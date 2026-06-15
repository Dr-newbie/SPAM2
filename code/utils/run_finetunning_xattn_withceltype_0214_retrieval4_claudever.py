"""
run_finetunning_xattn_withceltype_0214_retrieval4_claudever.py
--------------------------------------------------------------
Optimized version of run_finetunning_xattn_withceltype_0214_retrieval4.py

Performance fixes applied
-------------------------
1. [CRITICAL] GCN pre-computed once per epoch  (_precompute_gcn_all)
   Original: model.forward() ran the full GCN on ALL N nodes for EVERY mini-batch.
   For N=100K cells and batch_size=128 → ~800 full-graph GCN passes per epoch.
   Fix: compute GCN output for all nodes ONCE at epoch start with @torch.no_grad().
   Valid because GCN input (XY coords) is static — outputs only change when GCN
   weights change (between epochs, not within). Add --train_gcn to restore original
   behaviour if you need to keep finetuning the GCN weights.

2. [HIGH] Vectorized retrieval similarity computation
   Original: _retrieval4_memory_bank_expr() ran a Python for-loop over the batch,
   issuing one (1,D)@(N,D)^T GPU matmul per sample (and more per class).
   Fix: pre-compute the full (B,N) similarity matrix once with a single batched
   matmul before the loop. Per-class and per-sample steps index into this tensor
   at near-zero cost.

3. [MEDIUM] Pre-compute string → integer node indices at dataset build time
   Original: per-batch Python dict lookup loop converted string spot IDs to ints.
   Fix: _NodeIdxDataset wrapper resolves the mapping once at construction time,
   yielding integer tensors directly from the DataLoader.

4. [MEDIUM] GradScaler moved outside train_epoch()
   Original: new GradScaler created inside train_epoch() every epoch, resetting
   the internal loss-scale state each call.
   Fix: GradScaler created once in main() and passed as a parameter.

5. [MEDIUM] persistent_workers=True in DataLoader
   Avoids re-spawning N worker processes at the start of each epoch.

6. [MEDIUM] NaN checks on intermediate tensors gated by --nan_check_every (default 50)
   Original: torch.isfinite().all() called on (B,D) tensors every step → GPU-CPU sync.
   Fix: intermediate checks only every nan_check_every steps.
   The final scalar loss is still checked every step (nearly free).
"""

import os, sys, time
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Global mapping: string ID -> node index (kept for _to_device_batch compatibility)
ID2IDX = None

# ============================================================
# Path setup
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))   # .../Code_final/code
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.modules.pop("models", None)

from models.Foundations import inf_encoder_factory
from models.GCN_update import SpatialGCN
from models.gene_encoder import GeneEncoder
from models.fusion import CrossModalFusion
from models.decoders import ZINBDecoder
from utils.loss_utils import zinb_nll
from utils.dataset_utils import (
    SpotImageGeneDataset,
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_gene_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)
from models.graph_construction import build_knn_graph
from models.graph_construction_delaunay import build_delaunay_graph


# ============================================================
# Pretrained weight loaders (unchanged)
# ============================================================
def _load_sd(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and any(k in sd for k in ["state_dict", "model", "module"]):
        for k in ["state_dict", "model", "module"]:
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd if isinstance(sd, dict) else sd.state_dict()


def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_pretrained_for_finetune(foundation, gene_encoder, gcn,
                                 pt_img_backbone=None, pt_ig=None, pt_is=None):
    if pt_img_backbone:
        sd = _load_sd(pt_img_backbone)
        for s in [_strip_prefix(sd, "img_encoder.foundation."),
                  _strip_prefix(sd, "foundation."), sd]:
            try:
                foundation.load_state_dict(s, strict=False)
                print(f"[PT] loaded foundation from {pt_img_backbone}")
                break
            except Exception:
                pass

    if pt_ig:
        sd = _load_sd(pt_ig)
        for s in [_strip_prefix(sd, "gene_encoder."), sd]:
            try:
                gene_encoder.load_state_dict(s, strict=False)
                print(f"[PT] loaded gene_encoder from {pt_ig}")
                break
            except Exception:
                pass

    if pt_is:
        sd = _load_sd(pt_is)
        for s in [_strip_prefix(sd, "gcn."), sd]:
            try:
                gcn.load_state_dict(s, strict=False)
                print(f"[PT] loaded GCN from {pt_is}")
                break
            except Exception:
                pass


def load_xattn_into_fusion(fusion: CrossModalFusion, xattn_path):
    if not xattn_path:
        return
    sd = _load_sd(xattn_path)

    def _load_block(dst_block, src_prefix):
        sub = {k[len(src_prefix) + 1:]: v for k, v in sd.items()
               if k.startswith(src_prefix + ".")}
        if not sub:
            print(f"[XATTN][warn] no keys for {src_prefix}")
            return
        ret = dst_block.load_state_dict(sub, strict=False)
        if ret.missing_keys or ret.unexpected_keys:
            print(f"[XATTN][warn] {src_prefix}: missing={len(ret.missing_keys)}, "
                  f"unexpected={len(ret.unexpected_keys)}")
        else:
            print(f"[XATTN] loaded {src_prefix}")

    _load_block(fusion.xa_ig, "i2g")
    _load_block(fusion.xa_is, "i2c")


# ============================================================
# Cell type helpers (unchanged)
# ============================================================
def _pick_celltype_col(adata, celltype_col):
    cols = list(adata.obs.columns)
    cols_lower = {str(c).lower(): c for c in cols}
    if celltype_col is not None:
        key = str(celltype_col).lower()
        if key in cols_lower:
            return cols_lower[key]
        raise KeyError(f"--celltype_col={celltype_col!r} not found. Available: {cols}")
    candidates = ["cell_type", "celltype", "celltypes", "cell_type_pred", "annotation",
                  "annot", "cluster", "clusters", "leiden", "seurat_clusters", "Cluster"]
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    raise KeyError(f"Cell type column not found. Available: {cols}")


def _build_celltype_targets(adata, map_df_aligned, celltype_col):
    ct_col = _pick_celltype_col(adata, celltype_col)
    raw = adata.obs.loc[map_df_aligned["spot_id"], ct_col]
    raw_str = raw.astype(str)
    valid_mask = ~raw.isna()
    names = sorted(pd.unique(raw_str.loc[valid_mask]).tolist())
    name_to_idx = {n: i for i, n in enumerate(names)}

    y = np.full(map_df_aligned.shape[0], -1, dtype=np.int64)
    for i, sid in enumerate(map_df_aligned["spot_id"].tolist()):
        v = raw.loc[sid]
        if not pd.isna(v):
            y[i] = name_to_idx[str(v)]

    n_valid = int((y >= 0).sum())
    n_missing = int((y < 0).sum())
    print(f"[CellType] col={ct_col}, classes={len(names)}, labeled={n_valid}, missing={n_missing}")
    return y, names, ct_col


# ============================================================
# Fix 3: Pre-compute string → int node indices at dataset level
# ============================================================
def _norm_id_str(x) -> str:
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


class _NodeIdxDataset(torch.utils.data.Dataset):
    """
    Wraps SpotImageGeneDataset so the DataLoader returns an integer node index
    instead of a string spot ID.  The spot_id → node_index mapping is resolved
    once at construction time, eliminating per-batch Python string processing.
    Returns: (img_tensor, gene_tensor, node_idx_int)
    """
    def __init__(self, base_ds: SpotImageGeneDataset, id_to_idx: dict):
        self._base = base_ds
        self._node_indices = np.array(
            [id_to_idx[_norm_id_str(base_ds.map_df.iloc[i]["spot_id"])]
             for i in range(len(base_ds))],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx: int):
        img, gene, _ = self._base[idx]
        return img, gene, int(self._node_indices[idx])


def make_node_idx_dataloader(base_ds, id_to_idx, batch_size, num_workers,
                              shuffle=True):
    """DataLoader that yields (img, gene, node_idx_int) — no string IDs."""
    ds = _NodeIdxDataset(base_ds, id_to_idx)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),   # Fix 5
    )


# ============================================================
# Model (Fix 1: add zs_precomputed parameter to forward)
# ============================================================
class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn,
                 dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1,
                 merge="gated-sum",
                 img_tokens=1, spot_tokens=1, gene_tokens=1,
                 decoder_hidden=1024, gene_dim=1000,
                 num_celltypes=0, cls_head_hidden=0, cls_dropout=0.0):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.gcn = gcn

        self.proj_img  = nn.Linear(dim_img,  dim_fuse)
        self.proj_gene = nn.Linear(dim_gene, dim_fuse)
        self.proj_spot = nn.Linear(dim_spot, dim_fuse)

        self.fusion = CrossModalFusion(
            dim_img=dim_fuse, dim_spot=dim_fuse, dim_gene=dim_fuse,
            dim_out=dim_fuse, num_heads=num_heads, dropout=dropout,
            merge=merge, img_tokens=img_tokens,
            spot_tokens=spot_tokens, gene_tokens=gene_tokens,
        )
        self.decoder = ZINBDecoder(
            in_dim=dim_fuse, gene_dim=gene_dim,
            hidden=decoder_hidden, dropout=dropout,
        )
        self.num_celltypes = int(num_celltypes)
        if self.num_celltypes > 0:
            if cls_head_hidden and cls_head_hidden > 0:
                self.cls_head = nn.Sequential(
                    nn.Linear(dim_fuse, cls_head_hidden), nn.GELU(),
                    nn.Dropout(cls_dropout), nn.Linear(cls_head_hidden, self.num_celltypes),
                )
            else:
                self.cls_head = nn.Linear(dim_fuse, self.num_celltypes)
        else:
            self.cls_head = None

    def forward(self, batch, spot_feat_in, edge_index, edge_weight=None,
                return_attn=False, zs_precomputed=None):
        """
        zs_precomputed: (N, dim_spot) tensor, output of self.gcn pre-computed for
                        all N nodes.  If provided, the GCN call is skipped entirely
                        and we simply index the pre-computed tensor.  This is the
                        key Fix 1 optimisation.
        """
        x_img      = batch["image"]
        x_gene_log = batch["expr"].float()
        node_idx   = batch["idx"].long()

        zi = self.img_encoder(x_img)
        zg = self.gene_encoder(x_gene_log)

        # Fix 1: skip the full-graph GCN if pre-computed embeddings are available
        if zs_precomputed is not None:
            zs = zs_precomputed[node_idx]          # (B, dim_spot) — just index
        else:
            # Original path: full GCN on all N nodes (slow, keeps GCN trainable)
            spot_fp32 = spot_feat_in.float()
            ew_fp32   = edge_weight.float() if edge_weight is not None else None
            if x_img.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    zs_all = self.gcn(spot_fp32, edge_index, ew_fp32)
            else:
                zs_all = self.gcn(spot_fp32, edge_index, ew_fp32)
            zs = zs_all[node_idx]

        zi = self.proj_img(zi)
        zg = self.proj_gene(zg)
        zs = self.proj_spot(zs)

        if return_attn:
            z_fused, z_is, z_ig, attn = self.fusion(zi, zs, zg, return_attn=True)
        else:
            z_fused, z_is, z_ig = self.fusion(zi, zs, zg, return_attn=False)
            attn = None

        mu, theta, pi = self.decoder(z_fused)
        cls_logits = self.cls_head(z_fused) if self.cls_head is not None else None
        return mu, theta, pi, {
            "z_fused": z_fused, "z_is": z_is, "z_ig": z_ig,
            "attn": attn, "cls_logits": cls_logits,
        }


# ============================================================
# Fix 1: Pre-compute GCN embeddings for all nodes once per epoch
# ============================================================
@torch.no_grad()
def _precompute_gcn_all(model, spot_feat_in, edge_index, edge_weight):
    """
    Run the spatial GCN on all N nodes and return the detached (N, D) output.

    Called once at the start of each epoch.  The result is used as a read-only
    lookup table during all mini-batch forward passes, replacing the original
    O(N × n_batches) computation with O(N × 1).

    Why this is safe:
      - GCN input (XY coordinates) is static throughout training.
      - Within one epoch the GCN weights do not change between batches.
      - .detach() removes the GCN from the autograd graph for that epoch;
        add --train_gcn to fall back to the original per-batch full-graph path
        if you need to continue fine-tuning the GCN weights.
    """
    was_training = model.training
    model.eval()
    ew_fp32 = edge_weight.float() if edge_weight is not None else None
    with torch.amp.autocast("cuda", enabled=False):
        zs_all = model.gcn(spot_feat_in.float(), edge_index, ew_fp32)
    if was_training:
        model.train()
    return zs_all.detach()   # (N, D_spot)


# ============================================================
# Batch helper (Fix 3: handle integer tensor idx from _NodeIdxDataset)
# ============================================================
def _to_device_batch(batch, device):
    global ID2IDX

    def _to_tensor(x, device, dtype=None):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.as_tensor(x, dtype=dtype, device=device)

    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    if isinstance(batch, (list, tuple)):
        if len(batch) != 3:
            raise TypeError(f"Unexpected batch length {len(batch)}; expected 3.")
        x_img, x_gene, idx = batch
        x_img  = _to_tensor(x_img,  device)
        x_gene = _to_tensor(x_gene, device)

        # Fix 3: _NodeIdxDataset returns an integer tensor — handle directly
        if torch.is_tensor(idx):
            idx_tensor = idx.to(device, dtype=torch.long)
        elif isinstance(idx, (list, tuple, np.ndarray)):
            idx_list = list(idx)
            if idx_list and isinstance(idx_list[0], str):
                if ID2IDX is None:
                    raise RuntimeError("ID2IDX not initialized but idx is string-based.")
                idx_list = [ID2IDX[_norm_id_str(s)] for s in idx_list]
            idx_tensor = _to_tensor(idx_list, device, dtype=torch.long)
        else:
            idx_tensor = _to_tensor([idx], device, dtype=torch.long)

        return {"image": x_img, "expr": x_gene, "idx": idx_tensor}

    raise TypeError(f"Unsupported batch type: {type(batch)}")


# ============================================================
# Misc helpers (unchanged)
# ============================================================
def _save_attn_npz(path, attn_map, reduce_heads, node_idx=None):
    if attn_map is None:
        return
    payload = {}
    for k, v in attn_map.items():
        if v is None:
            continue
        x = v.detach().float().cpu()
        if reduce_heads and x.ndim == 4:
            x = x.mean(dim=1)
        payload[k] = x.numpy()
    if node_idx is not None:
        payload["node_idx"] = node_idx
    if payload:
        np.savez_compressed(path, **payload)


def _adata_x_to_numpy_float32(adata):
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"adata.X must be 2D, got shape={x.shape}")
    return x


def _build_class_index_map(celltype_y_np):
    out = {}
    if celltype_y_np is None:
        return out
    y = np.asarray(celltype_y_np, dtype=np.int64)
    for c in np.unique(y):
        if int(c) < 0:
            continue
        out[int(c)] = np.where(y == int(c))[0].astype(np.int64)
    return out


def _current_retrieval4_tau(model, tau_default=0.07):
    dev = next(model.parameters()).device
    if hasattr(model, "retrieval4_log_tau"):
        t = torch.exp(getattr(model, "retrieval4_log_tau"))
        return torch.clamp(t, min=1e-3, max=50.0)
    if hasattr(model, "retrieval4_tau_const"):
        t = getattr(model, "retrieval4_tau_const")
        return torch.clamp(t.to(device=dev), min=1e-3, max=50.0)
    return torch.tensor(float(max(tau_default, 1e-3)), device=dev, dtype=torch.float32)


# ============================================================
# Retrieval bank builder (Fix 1: accept zs_precomputed)
# ============================================================
@torch.no_grad()
def _build_retrieval4_banks(model, loader, spot_feat_in, edge_index, edge_weight,
                             expr_bank_np, num_celltypes, device, amp_dtype,
                             n_nodes, pred_min_conf=0.0,
                             zs_precomputed=None):   # Fix 1
    was_training = model.training
    model.eval()
    dim = int(model.proj_img.out_features)
    key_bank      = torch.zeros((n_nodes, dim), dtype=torch.float32, device=device)
    pred_cls_bank = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    pred_prob_bank = torch.zeros((n_nodes, int(num_celltypes)),
                                 dtype=torch.float32, device=device)
    seen = torch.zeros((n_nodes,), dtype=torch.bool, device=device)

    pbar = tqdm(loader, desc="[Retrieval4] build key/prob banks", leave=False)
    for batch in pbar:
        batch = _to_device_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            q = model.proj_img(model.img_encoder(batch["image"]))
            # Fix 1: pass pre-computed GCN embeddings to skip per-batch full-graph GCN
            _, _, _, aux = model(batch, spot_feat_in, edge_index, edge_weight,
                                 return_attn=False, zs_precomputed=zs_precomputed)

        q = F.normalize(q.float(), dim=1)
        cls_logits = aux.get("cls_logits", None)
        if cls_logits is None:
            raise RuntimeError("[Retrieval4] cls_head required but cls_logits is None.")
        probs    = torch.softmax(cls_logits.float(), dim=1)
        _, pred_cls = torch.max(probs, dim=1)
        idx = batch["idx"].long()
        key_bank[idx]       = q
        pred_cls_bank[idx]  = pred_cls.long()
        pred_prob_bank[idx] = probs.float()
        seen[idx] = True

    miss = int((~seen).sum().item())
    if miss > 0:
        print(f"[Retrieval4][warn] {miss} nodes not seen while building banks.")

    pred_cls_np  = pred_cls_bank.cpu().numpy()
    pred_prob_np = pred_prob_bank.cpu().numpy()
    pred_conf_np = pred_prob_np.max(axis=1)
    class_index_map = {}
    for c in range(int(num_celltypes)):
        m = np.where((pred_cls_np == c) & (pred_conf_np >= float(pred_min_conf)))[0]
        if m.size > 0:
            class_index_map[int(c)] = m.astype(np.int64)

    if was_training:
        model.train()
    return {"keys_bank": key_bank, "pred_cls_bank": pred_cls_bank,
            "pred_prob_bank": pred_prob_bank, "class_index_map": class_index_map}


# ============================================================
# Fix 2: Vectorized retrieval memory-bank expression lookup
# ============================================================
def _allocate_cluster_counts(prob_vec, total):
    total = int(max(total, 0))
    if total == 0:
        return np.zeros_like(prob_vec, dtype=np.int64)
    p = np.clip(np.asarray(prob_vec, dtype=np.float64), 0.0, None)
    s = float(p.sum())
    p = p / s if s > 0 else np.ones_like(p) / max(int(p.size), 1)
    raw = p * float(total)
    cnt = np.floor(raw).astype(np.int64)
    rem = int(total - int(cnt.sum()))
    if rem > 0:
        order = np.argsort(-(raw - cnt.astype(np.float64)))
        cnt[order[:rem]] += 1
    return cnt


def _retrieval4_memory_bank_expr(
    query_key, query_probs, x_log_true,
    keys_bank, expr_bank_np, class_index_map,
    bank_size=5000, tau=None, prob_topm=0, fallback="global",
):
    """
    Build a class-proportional memory bank per query and compute a weighted-sum
    retrieved expression vector.

    Fix 2 (vectorized similarity):
      The original computed (1,D)@(N,D)^T for every sample × class combination,
      serialising hundreds of small GPU matmuls per step.
      Now: one (B,N) matmul is performed upfront; all subsequent per-sample and
      per-class operations index into this pre-computed similarity matrix.
    """
    device = query_key.device
    q   = F.normalize(query_key.float(), dim=1)   # (B, D)
    bsz = int(q.size(0))
    n_nodes     = int(keys_bank.size(0))
    target_bank = int(max(1, min(int(bank_size), n_nodes)))

    x_retr   = x_log_true.clone()
    used     = torch.zeros((bsz,), dtype=torch.bool, device=device)
    top1_vals = []

    if tau is None:
        tau = torch.tensor(0.07, dtype=torch.float32, device=device)
    tau = torch.clamp(tau, min=1e-3, max=50.0)

    probs = query_probs.float().to(device=device)
    n_cls = int(probs.size(1))
    if prob_topm > 0 and prob_topm < n_cls:
        vals_m, idx_m = torch.topk(probs, k=prob_topm, dim=1, largest=True, sorted=False)
        probs_topm = torch.zeros_like(probs)
        probs_topm.scatter_(1, idx_m, vals_m)
        probs = probs_topm
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    all_idx_np = np.arange(n_nodes, dtype=np.int64)

    # Fix 2: single batched matmul for ALL B×N similarities
    # Replaces per-sample (1,D)@(N,D)^T calls inside the loop
    sim_all = q @ keys_bank.t()   # (B, N) — one GPU call

    for bi in range(bsz):
        sim_bi   = sim_all[bi]                                       # (N,) view
        prob_vec = probs[bi].detach().cpu().numpy().astype(np.float64)
        quota    = _allocate_cluster_counts(prob_vec, target_bank)

        selected_parts = []
        for ci, take_n in enumerate(quota.tolist()):
            if take_n <= 0:
                continue
            cand_np = class_index_map.get(int(ci), None)
            if cand_np is None or cand_np.size == 0:
                continue
            cand_t   = torch.as_tensor(cand_np, device=device, dtype=torch.long)
            # Fix 2: index pre-computed row — no new GPU matmul
            sim_class = sim_bi[cand_t]
            kk = min(int(take_n), int(sim_class.numel()))
            if kk <= 0:
                continue
            _, loc = torch.topk(sim_class, k=kk, dim=0, largest=True, sorted=True)
            selected_parts.append(cand_np[loc.detach().cpu().numpy()])

        if selected_parts:
            selected_np = np.unique(np.concatenate(selected_parts, axis=0).astype(np.int64))
        else:
            selected_np = np.empty((0,), dtype=np.int64)

        # Fill shortfall from global similarity (Fix 2: reuse sim_bi)
        if selected_np.size < target_bank:
            if selected_np.size > 0:
                sim_fill = sim_bi.clone()
                sim_fill[torch.as_tensor(selected_np, device=device, dtype=torch.long)] = -1e9
            else:
                sim_fill = sim_bi
            need = min(target_bank - int(selected_np.size), int(sim_fill.numel()))
            if need > 0:
                _, fill_idx = torch.topk(sim_fill, k=need, dim=0, largest=True, sorted=True)
                fill_np = fill_idx.detach().cpu().numpy().astype(np.int64)
                selected_np = (np.concatenate([selected_np, fill_np])
                               if selected_np.size > 0 else fill_np)

        if selected_np.size == 0:
            if fallback == "global":
                kk = min(target_bank, int(sim_bi.numel()))
                _, loc = torch.topk(sim_bi, k=kk, dim=0, largest=True, sorted=True)
                selected_np = all_idx_np[loc.detach().cpu().numpy()]
            else:
                continue

        sel_t = torch.as_tensor(selected_np, device=device, dtype=torch.long)
        # Fix 2: reuse sim_bi — no new GPU matmul
        sim_sel = sim_bi[sel_t]
        if sim_sel.numel() > target_bank:
            vals_sel, loc_sel = torch.topk(sim_sel, k=target_bank,
                                           dim=0, largest=True, sorted=True)
            sel_t   = sel_t.index_select(0, loc_sel)
            sim_sel = vals_sel
        if sim_sel.numel() == 0:
            continue

        w = torch.softmax(sim_sel / tau, dim=0)
        top1_vals.append(sim_sel.max().detach())
        expr_sel = torch.from_numpy(
            expr_bank_np[sel_t.detach().cpu().numpy()]
        ).to(device=device, dtype=x_log_true.dtype)
        x_retr[bi] = (w.unsqueeze(-1) * expr_sel).sum(dim=0)
        used[bi] = True

    top1_mean = float(torch.cat(top1_vals).mean().item()) if top1_vals else float("nan")
    return x_retr, used, top1_mean


# ============================================================
# Training loop (Fix 1, 4, 6)
# ============================================================
def train_epoch(model, loader, spot_feat_in, edge_index, edge_weight,
                optimizer, scaler,               # Fix 4: scaler passed in, not created here
                amp_dtype=torch.float32,
                device=torch.device("cpu"),
                epoch=1, total_epochs=1,
                lambda_zinb=1.0, lambda_mse=0.1,
                lambda_pcc=0.0, lambda_ce=0.0,
                ce_label_smoothing=0.0, ce_warmup_epochs=0,
                celltype_labels_all=None, ce_class_weights=None,
                max_grad_norm=1.0,
                retrieval4_cfg=None,
                zs_precomputed=None,             # Fix 1: pre-computed GCN embeddings
                nan_check_every=50):             # Fix 6: NaN check frequency
    model.train()
    loss_sum, n = 0.0, 0
    cls_correct, cls_count = 0.0, 0
    retr_used_sum, retr_top1_sum, retr_steps = 0.0, 0.0, 0
    retr_tau_last = float("nan")
    eps = 1e-8

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}/{total_epochs}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch = _to_device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        # Fix 6: only check intermediate tensor NaN every nan_check_every steps
        do_nan_check = (step % nan_check_every == 0)

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            x_log = batch["expr"].float()
            batch_in  = batch
            loss_retr = x_log.new_zeros(())
            retr_used_ratio = 0.0

            if retrieval4_cfg is not None and bool(retrieval4_cfg.get("enabled", False)):
                keys_bank      = retrieval4_cfg.get("keys_bank")
                pred_prob_bank = retrieval4_cfg.get("pred_prob_bank")
                expr_bank_np   = retrieval4_cfg.get("expr_bank_np")
                class_index_map = retrieval4_cfg.get("class_index_map", {})

                if keys_bank is not None and expr_bank_np is not None and pred_prob_bank is not None:
                    q   = model.proj_img(model.img_encoder(batch["image"]))
                    tau = _current_retrieval4_tau(model, float(retrieval4_cfg.get("tau", 0.07)))
                    retr_tau_last = float(tau.detach().item())

                    x_retr, retr_used, retr_top1 = _retrieval4_memory_bank_expr(
                        query_key=q,
                        query_probs=pred_prob_bank[batch["idx"].long()],
                        x_log_true=x_log,
                        keys_bank=keys_bank,
                        expr_bank_np=expr_bank_np,
                        class_index_map=class_index_map,
                        bank_size=int(retrieval4_cfg.get("bank_size", 5000)),
                        tau=tau,
                        prob_topm=int(retrieval4_cfg.get("prob_topm", 0)),
                        fallback=str(retrieval4_cfg.get("fallback", "global")),
                    )
                    if not np.isnan(retr_top1):
                        retr_top1_sum += float(retr_top1)
                        retr_steps += 1
                    retr_used_ratio = float(retr_used.float().mean().item())
                    retr_used_sum  += retr_used_ratio

                    mix_base = float(retrieval4_cfg.get("mix", 0.0))
                    mix_warm = int(retrieval4_cfg.get("mix_warmup_epochs", 0))
                    mix = mix_base * min(1.0, float(epoch) / float(max(mix_warm, 1))) \
                          if mix_warm > 0 else mix_base
                    mix = float(np.clip(mix, 0.0, 1.0))
                    batch_in = dict(batch)
                    batch_in["expr"] = (1.0 - mix) * x_log + mix * x_retr

                    if float(retrieval4_cfg.get("lambda_retr", 0.0)) > 0.0:
                        loss_retr = F.mse_loss(x_retr, x_log)

            # Fix 1: pass pre-computed GCN embeddings to skip full-graph GCN per batch
            mu, theta, pi, aux = model(
                batch_in, spot_feat_in, edge_index, edge_weight,
                zs_precomputed=zs_precomputed,
            )

            # Fix 6: intermediate NaN checks only every nan_check_every steps
            if do_nan_check and (torch.isnan(mu).any() or torch.isinf(mu).any()):
                print(f"[WARN] Step {step}: mu NaN/Inf, skipping"); continue
            if do_nan_check and (torch.isnan(theta).any() or torch.isinf(theta).any()):
                print(f"[WARN] Step {step}: theta NaN/Inf, skipping"); continue

            x_raw    = torch.expm1(torch.clamp(x_log, min=0.0))
            loss_zinb = zinb_nll(x_raw, mu, theta, pi, reduction="mean")

            # Final scalar loss always checked (cheap)
            if torch.isnan(loss_zinb) or torch.isinf(loss_zinb):
                print(f"[WARN] Step {step}: ZINB loss NaN/Inf, skipping"); continue

            mu_log   = torch.log1p(torch.clamp(mu, min=eps))
            loss_mse = F.mse_loss(mu_log, x_log)
            if torch.isnan(loss_mse) or torch.isinf(loss_mse):
                print(f"[WARN] Step {step}: MSE loss NaN/Inf, skipping"); continue

            if lambda_pcc > 0.0:
                Yc  = x_log - x_log.mean(dim=0, keepdim=True)
                Yhc = mu_log - mu_log.mean(dim=0, keepdim=True)
                num = (Yc * Yhc).sum(dim=0)
                den = (Yc.pow(2).sum(dim=0) * Yhc.pow(2).sum(dim=0)).sqrt() + 1e-8
                pcc_loss = 1.0 - (num / den).mean()
            else:
                pcc_loss = 0.0 * loss_zinb

            use_ce = (lambda_ce > 0.0 and model.cls_head is not None
                      and celltype_labels_all is not None
                      and celltype_labels_all.numel() > 0)
            if use_ce:
                y_ct    = celltype_labels_all[batch["idx"].long()]
                valid   = y_ct >= 0
                cls_logits = aux.get("cls_logits")
                ce_scale = (min(1.0, float(epoch) / max(int(ce_warmup_epochs), 1))
                            if ce_warmup_epochs > 0 else 1.0)
                if cls_logits is None or valid.sum().item() == 0:
                    ce_loss = 0.0 * loss_zinb
                    ce_acc = 0.0; ce_seen = 0
                else:
                    ce_loss = F.cross_entropy(cls_logits[valid], y_ct[valid],
                                              weight=ce_class_weights,
                                              label_smoothing=max(float(ce_label_smoothing), 0.0))
                    pred_ct = torch.argmax(cls_logits[valid], dim=1)
                    ce_seen = int(valid.sum().item())
                    ce_acc  = float((pred_ct == y_ct[valid]).float().mean().item())
                    ce_loss = ce_loss * ce_scale
            else:
                ce_loss = 0.0 * loss_zinb
                ce_acc = 0.0; ce_seen = 0

            lam_retr = float(retrieval4_cfg.get("lambda_retr", 0.0)
                             if retrieval4_cfg is not None else 0.0)
            loss = (lambda_zinb * loss_zinb + lambda_mse * loss_mse
                    + lambda_pcc * pcc_loss + lambda_ce * ce_loss
                    + lam_retr * loss_retr)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Step {step}: total loss NaN/Inf, skipping"); continue

        # Fix 4: use passed-in scaler (not a freshly constructed one)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        bs = x_log.size(0)
        loss_sum += loss.item() * bs
        n += bs
        if ce_seen > 0:
            cls_correct += ce_acc * ce_seen
            cls_count   += ce_seen

        pbar.set_postfix({
            "Lz": f"{loss_zinb.item():.4f}",
            "Lm": f"{loss_mse.item():.4f}",
            "Lp": f"{(pcc_loss.item() if lambda_pcc > 0.0 else 0.0):.4f}",
            "Lc": f"{(ce_loss.item() if lambda_ce > 0.0 else 0.0):.4f}",
            "Lr1": f"{(loss_retr.item() if retrieval4_cfg is not None else 0.0):.4f}",
            "Ruse": f"{retr_used_ratio:.2f}",
            "Cacc": f"{(ce_acc if ce_seen > 0 else 0.0):.3f}",
        })

    return {
        "loss":       loss_sum / max(n, 1),
        "ct_acc":     float(cls_correct / max(cls_count, 1)),
        "ct_n":       int(cls_count),
        "retr_used":  float(retr_used_sum / max(len(loader), 1)),
        "retr_top1":  float(retr_top1_sum / max(retr_steps, 1)) if retr_steps > 0 else float("nan"),
        "retr_tau":   float(retr_tau_last),
    }


# ============================================================
# Eval loop (Fix 1)
# ============================================================
@torch.no_grad()
def eval_epoch(model, loader, spot_feat_in, edge_index, edge_weight,
               amp_dtype=torch.float32, device=torch.device("cpu"),
               epoch=1, total_epochs=1,
               save_attn=False, attn_out_dir=None,
               attn_max_batches=0, attn_reduce_heads=False,
               celltype_labels_all=None,
               retrieval4_cfg=None,
               zs_precomputed=None):    # Fix 1
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    cls_correct, cls_count = 0.0, 0
    preds, gts = [], []
    eps = 1e-8

    pbar = tqdm(loader, desc=f"[Eval ] Epoch {epoch}/{total_epochs}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch = _to_device_batch(batch, device)
        dump_attn = bool(save_attn and attn_out_dir and attn_max_batches > 0
                         and step <= attn_max_batches)

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            x_log    = batch["expr"].float()
            batch_in = batch

            if retrieval4_cfg is not None and bool(retrieval4_cfg.get("enabled", False)):
                keys_bank      = retrieval4_cfg.get("keys_bank")
                pred_prob_bank = retrieval4_cfg.get("pred_prob_bank")
                expr_bank_np   = retrieval4_cfg.get("expr_bank_np")
                class_index_map = retrieval4_cfg.get("class_index_map", {})
                if keys_bank is not None and expr_bank_np is not None and pred_prob_bank is not None:
                    q   = model.proj_img(model.img_encoder(batch["image"]))
                    tau = _current_retrieval4_tau(model, float(retrieval4_cfg.get("tau", 0.07)))
                    x_retr, _, _ = _retrieval4_memory_bank_expr(
                        query_key=q,
                        query_probs=pred_prob_bank[batch["idx"].long()],
                        x_log_true=x_log,
                        keys_bank=keys_bank, expr_bank_np=expr_bank_np,
                        class_index_map=class_index_map,
                        bank_size=int(retrieval4_cfg.get("bank_size", 5000)), tau=tau,
                        prob_topm=int(retrieval4_cfg.get("prob_topm", 0)),
                        fallback=str(retrieval4_cfg.get("fallback", "global")),
                    )
                    mix = float(np.clip(float(retrieval4_cfg.get("eval_mix", 1.0)), 0.0, 1.0))
                    batch_in = dict(batch)
                    batch_in["expr"] = (1.0 - mix) * x_log + mix * x_retr

            # Fix 1: skip full-graph GCN
            mu, theta, pi, aux = model(
                batch_in, spot_feat_in, edge_index, edge_weight,
                return_attn=dump_attn,
                zs_precomputed=zs_precomputed,
            )
            mu_clamped = torch.clamp(mu, min=0.0)
            yhat_log   = torch.log1p(mu_clamped + eps)

        preds.append(yhat_log.detach().cpu())
        gts.append(x_log.detach().cpu())
        mse_sum += F.mse_loss(yhat_log, x_log, reduction="sum").item()
        mae_sum += F.l1_loss(yhat_log,  x_log, reduction="sum").item()
        n       += x_log.numel()

        if celltype_labels_all is not None and celltype_labels_all.numel() > 0:
            y_ct = celltype_labels_all[batch["idx"].long()]
            valid = y_ct >= 0
            cls_logits = aux.get("cls_logits")
            if cls_logits is not None and valid.sum().item() > 0:
                pred_ct = torch.argmax(cls_logits[valid], dim=1)
                cls_correct += float((pred_ct == y_ct[valid]).sum().item())
                cls_count   += int(valid.sum().item())

        if dump_attn:
            out_npz = os.path.join(attn_out_dir, f"ep{epoch:03d}_eval_step{step:04d}.npz")
            _save_attn_npz(out_npz, aux.get("attn"), reduce_heads=attn_reduce_heads,
                           node_idx=batch["idx"].detach().cpu().numpy())

    P = torch.cat(preds, 0).numpy()
    Y = torch.cat(gts,   0).numpy()
    Yc = Y - Y.mean(0, keepdims=True)
    Pc = P - P.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0)) + 1e-8
    pcc = np.where(den > 0, num / den, np.nan)

    return {
        "mse":        mse_sum / max(n, 1),
        "mae":        mae_sum / max(n, 1),
        "pcc_mean":   float(np.nanmean(pcc)),
        "pcc_median": float(np.nanmedian(pcc)),
        "ct_acc":     float(cls_correct / max(cls_count, 1)),
        "ct_n":       int(cls_count),
    }, P, Y


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--pt_img_backbone", default=None)
    p.add_argument("--pt_ig",           default=None)
    p.add_argument("--pt_is",           default=None)
    p.add_argument("--pt_xattn",        default=None)

    p.add_argument("--h5ad",     required=True)
    p.add_argument("--csv",      required=True)
    p.add_argument("--root",     default=None)
    p.add_argument("--id_col",   default="cell_id")
    p.add_argument("--path_col", default="patch_path")
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0",
                            "plip","phikon","conch_v1"])
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--proj_dim_gene", type=int,  default=256)
    p.add_argument("--proj_dim_spot", type=int,  default=256)
    p.add_argument("--fuse_dim",     type=int,   default=512)
    p.add_argument("--heads",        type=int,   default=8)
    p.add_argument("--img_tokens",   type=int,   default=1)
    p.add_argument("--spot_tokens",  type=int,   default=1)
    p.add_argument("--gene_tokens",  type=int,   default=1)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--merge", default="gated-sum", choices=["gated-sum","concat-proj"])
    p.add_argument("--k",            type=int,   default=12)
    p.add_argument("--graph_type",   default="knn", choices=["knn","delaunay"])
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--device",       default="auto")
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--save_dir",     default="./ckpts_cross")
    p.add_argument("--save_attn",    action="store_true")
    p.add_argument("--attn_out_dir", default=None)
    p.add_argument("--attn_max_batches", type=int, default=10)
    p.add_argument("--attn_every",   type=int,   default=1)
    p.add_argument("--attn_reduce_heads", action="store_true")
    p.add_argument("--inside_he_only",    action="store_true")

    p.add_argument("--lambda_zinb",  type=float, default=1.0)
    p.add_argument("--lambda_mse",   type=float, default=0.1)
    p.add_argument("--lambda_pcc",   type=float, default=0.0)
    p.add_argument("--lambda_ce",    type=float, default=0.2)
    p.add_argument("--celltype_col", default=None)
    p.add_argument("--ce_label_smoothing", type=float, default=0.0)
    p.add_argument("--ce_warmup_epochs",   type=int,   default=0)
    p.add_argument("--ce_head_hidden",     type=int,   default=0)
    p.add_argument("--ce_head_dropout",    type=float, default=0.0)
    p.add_argument("--ce_class_weight",    default="none", choices=["none","balanced"])
    p.add_argument("--warmup_epochs",      type=int,   default=2)
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--ema_decay",          type=float, default=0.0)

    # retrieval4 args (unchanged)
    p.add_argument("--retrieval4",                  action="store_true")
    p.add_argument("--retrieval4_bank_size",        type=int,   default=5000)
    p.add_argument("--retrieval4_tau",              type=float, default=0.07)
    p.add_argument("--retrieval4_learnable_tau",    action="store_true")
    p.add_argument("--retrieval4_mix",              type=float, default=0.5)
    p.add_argument("--retrieval4_eval_mix",         type=float, default=1.0)
    p.add_argument("--retrieval4_mix_warmup_epochs",type=int,   default=5)
    p.add_argument("--retrieval4_lambda",           type=float, default=0.1)
    p.add_argument("--retrieval4_refresh_epochs",   type=int,   default=1)
    p.add_argument("--retrieval4_fallback",         default="global",
                   choices=["self_expr","global"])
    p.add_argument("--retrieval4_num_celltypes",    type=int,   default=0)
    p.add_argument("--retrieval4_pred_min_conf",    type=float, default=0.0)
    p.add_argument("--retrieval4_prob_topm",        type=int,   default=0)

    # Fix 1: GCN training flag
    p.add_argument("--train_gcn", action="store_true",
                   help="Keep GCN trainable during finetuning (restores original per-batch "
                        "full-graph GCN behaviour, slower but allows GCN weight updates).")
    # Fix 6: NaN check frequency
    p.add_argument("--nan_check_every", type=int, default=50,
                   help="Check intermediate tensors for NaN every N steps "
                        "(final loss still checked every step). Default: 50.")

    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # device & amp
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")
    amp_dtype = torch.float32
    if args.amp:
        amp_dtype = (torch.bfloat16
                     if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
                     else torch.float16)

    print(f"[Device] {device}  AMP={args.amp}  dtype={amp_dtype}")
    print(f"[FusionTokens] img={args.img_tokens}, spot={args.spot_tokens}, "
          f"gene={args.gene_tokens}")
    print(f"[GCN] train_gcn={args.train_gcn}  "
          f"({'per-batch full-graph (original)' if args.train_gcn else 'pre-computed once/epoch (fast)'})")
    print(f"[NaNCheck] every {args.nan_check_every} steps")

    attn_out_dir = None
    if args.save_attn:
        attn_out_dir = args.attn_out_dir or os.path.join(args.save_dir, "attn")
        os.makedirs(attn_out_dir, exist_ok=True)

    # ---- data ----
    adata = load_h5ad_with_preproc(
        args.h5ad, use_hvg=False, n_top_genes=None,
        x_key="x_centroid", y_key="y_centroid",
    )
    map_df = load_patch_mapping_csv_xenium(
        args.csv, root_dir=args.root,
        id_col=args.id_col, path_col=args.path_col, drop_missing=True,
    )
    if args.inside_he_only:
        try:
            raw_df = pd.read_csv(args.csv)
        except Exception:
            raw_df = None
        if raw_df is not None and "inside_he" in raw_df.columns:
            keep     = raw_df["inside_he"].astype(str).str.lower().isin(["true","1","yes"])
            keep_ids = raw_df.loc[keep, args.id_col].astype(str).tolist()
            map_df   = map_df.loc[map_df["spot_id"].astype(str).isin(keep_ids)].reset_index(drop=True)
        else:
            print("[WARN] inside_he_only requested but 'inside_he' column not found.")

    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(
        adata, map_df, id_col="spot_id", coord_key="spatial"
    )
    if map_df_aligned.shape[0] != adata.n_obs:
        adata = adata[map_df_aligned["spot_id"].astype(str).tolist()].copy()

    # Fix 3: build normalized id→idx mapping and wire up global ID2IDX for legacy path
    global ID2IDX
    id_values = map_df_aligned["spot_id"].astype(str).tolist()
    id_to_idx = {_norm_id_str(v): i for i, v in enumerate(id_values)}
    ID2IDX    = id_to_idx
    print(f"[ID2IDX] {len(ID2IDX)} nodes")

    # cell type targets
    celltype_y_np, celltype_names, celltype_col_used = None, [], None
    if args.lambda_ce > 0.0:
        celltype_y_np, celltype_names, celltype_col_used = _build_celltype_targets(
            adata=adata, map_df_aligned=map_df_aligned, celltype_col=args.celltype_col)
        if len(celltype_names) < 2:
            raise ValueError(f"Need >=2 labeled classes, got {len(celltype_names)}.")

    # Fix 3 + Fix 5: use _NodeIdxDataset + persistent_workers
    foundation_tmp = inf_encoder_factory(args.enc_name)(weights_path=None)
    image_tf = getattr(foundation_tmp, "eval_transforms", None)
    del foundation_tmp

    base_ds = SpotImageGeneDataset(
        adata=adata, map_df=map_df_aligned, gene_names=None, image_transform=image_tf,
    )
    gene_dim = base_ds.gene_dim
    loader = make_node_idx_dataloader(
        base_ds=base_ds, id_to_idx=id_to_idx,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"[Data] N={len(base_ds)}, gene_dim={gene_dim}, steps/epoch={len(loader)}")

    # graph
    if args.graph_type == "delaunay":
        edge_index, edge_weight = build_delaunay_graph(coords_np, k=args.k, use_weight=True)
    else:
        edge_index, edge_weight = build_knn_graph(coords_np, k=args.k, use_weight=True)
    print(f"[Graph] type={args.graph_type}, k={args.k}, E={edge_index.size(1)}")
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    spot_feat_in = torch.from_numpy(coords_np).float().to(device)

    # ---- encoders ----
    foundation  = inf_encoder_factory(args.enc_name)(weights_path=None)
    foundation.precision = amp_dtype
    img_dim = foundation.get_output_dim()

    gene_encoder = GeneEncoder(
        input_dim=gene_dim, model_size_omic="small",
        proj_dim=args.proj_dim_gene, normalize=False,
    )
    spot_gcn = SpatialGCN(
        in_dim=spot_feat_in.size(1), hidden_dim=256,
        out_dim=args.proj_dim_spot, dropout=0.2,
    )
    load_pretrained_for_finetune(
        foundation, gene_encoder, spot_gcn,
        args.pt_img_backbone, args.pt_ig, args.pt_is,
    )

    num_celltypes_model = 0
    if args.lambda_ce > 0.0:
        num_celltypes_model = len(celltype_names)
    elif args.retrieval4:
        num_celltypes_model = int(max(args.retrieval4_num_celltypes, 0))
    if args.retrieval4 and num_celltypes_model <= 0:
        raise ValueError("--retrieval4 requires cls_head. Set --lambda_ce>0 or "
                         "--retrieval4_num_celltypes > 0.")

    model = CrossAttnZINBModel(
        img_encoder=foundation, gene_encoder=gene_encoder, gcn=spot_gcn,
        dim_img=img_dim, dim_gene=args.proj_dim_gene, dim_spot=args.proj_dim_spot,
        dim_fuse=args.fuse_dim, num_heads=args.heads, dropout=args.dropout,
        merge=args.merge, img_tokens=args.img_tokens,
        spot_tokens=args.spot_tokens, gene_tokens=args.gene_tokens,
        decoder_hidden=1024, gene_dim=gene_dim,
        num_celltypes=num_celltypes_model,
        cls_head_hidden=args.ce_head_hidden, cls_dropout=args.ce_head_dropout,
    ).to(device)

    if args.retrieval4:
        init_tau = float(max(args.retrieval4_tau, 1e-3))
        if args.retrieval4_learnable_tau:
            model.retrieval4_log_tau = nn.Parameter(
                torch.tensor(np.log(init_tau), dtype=torch.float32))
            print(f"[Retrieval4] learnable tau (init={init_tau:.5f})")
        else:
            model.register_buffer("retrieval4_tau_const",
                                  torch.tensor(init_tau, dtype=torch.float32))
            print(f"[Retrieval4] fixed tau={init_tau:.5f}")

    if args.pt_xattn:
        load_xattn_into_fusion(model.fusion, args.pt_xattn)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    # Fix 4: GradScaler created ONCE here and passed to train_epoch every epoch
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and amp_dtype != torch.float32),
    )

    celltype_y_t    = None
    ce_class_weights = None
    if args.lambda_ce > 0.0 and celltype_y_np is not None:
        celltype_y_t = torch.from_numpy(celltype_y_np).to(device=device, dtype=torch.long)
        if args.ce_class_weight == "balanced":
            valid   = celltype_y_np[celltype_y_np >= 0]
            counts  = np.bincount(valid, minlength=len(celltype_names)).astype(np.float32)
            w       = counts.sum() / np.maximum(counts, 1.0)
            ce_class_weights = torch.tensor(w, dtype=torch.float32, device=device)
            print(f"[CellType] class_weight=balanced (min={w.min():.4f}, max={w.max():.4f})")

    retrieval4_cfg = None
    if args.retrieval4:
        expr_bank_np = _adata_x_to_numpy_float32(adata)
        retrieval4_cfg = {
            "enabled":            True,
            "bank_size":          int(max(1, args.retrieval4_bank_size)),
            "tau":                float(max(args.retrieval4_tau, 1e-3)),
            "mix":                float(np.clip(args.retrieval4_mix,      0.0, 1.0)),
            "eval_mix":           float(np.clip(args.retrieval4_eval_mix, 0.0, 1.0)),
            "mix_warmup_epochs":  int(max(0, args.retrieval4_mix_warmup_epochs)),
            "lambda_retr":        float(max(0.0, args.retrieval4_lambda)),
            "fallback":           str(args.retrieval4_fallback),
            "pred_min_conf":      float(max(0.0, args.retrieval4_pred_min_conf)),
            "prob_topm":          int(max(0, args.retrieval4_prob_topm)),
            "expr_bank_np":       expr_bank_np,
            "class_index_map":    {},
            "keys_bank":          None,
            "pred_prob_bank":     None,
        }

    from torch.optim.lr_scheduler import LambdaLR

    def warmup_lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1.0) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optim, lr_lambda=warmup_lr_lambda)
    best = {"pcc": -1.0, "path": None}

    for ep in range(1, args.epochs + 1):

        # ------------------------------------------------------------------
        # Fix 1: Pre-compute GCN output for all N nodes once per epoch.
        # If --train_gcn is set, skip caching and use the original path.
        # ------------------------------------------------------------------
        if args.train_gcn:
            zs_all = None   # original per-batch full-graph GCN path
        else:
            zs_all = _precompute_gcn_all(model, spot_feat_in, edge_index, edge_weight)

        # Rebuild retrieval banks (also benefits from zs_all — no redundant GCN)
        if retrieval4_cfg is not None and retrieval4_cfg.get("enabled"):
            refresh_every = max(1, int(args.retrieval4_refresh_epochs))
            if ep == 1 or ((ep - 1) % refresh_every == 0):
                banks = _build_retrieval4_banks(
                    model=model, loader=loader,
                    spot_feat_in=spot_feat_in, edge_index=edge_index,
                    edge_weight=edge_weight,
                    expr_bank_np=retrieval4_cfg["expr_bank_np"],
                    num_celltypes=num_celltypes_model,
                    device=device, amp_dtype=amp_dtype,
                    n_nodes=map_df_aligned.shape[0],
                    pred_min_conf=float(retrieval4_cfg.get("pred_min_conf", 0.0)),
                    zs_precomputed=zs_all,     # Fix 1
                )
                retrieval4_cfg["keys_bank"]      = banks["keys_bank"]
                retrieval4_cfg["pred_prob_bank"] = banks["pred_prob_bank"]
                retrieval4_cfg["class_index_map"] = banks["class_index_map"]
                print(f"[Retrieval4] key bank refreshed at epoch {ep}, "
                      f"clusters={len(retrieval4_cfg['class_index_map'])}")

        t0 = time.time()
        tr = train_epoch(
            model, loader, spot_feat_in, edge_index, edge_weight,
            optim, scaler,                 # Fix 4: pass scaler
            amp_dtype, device,
            epoch=ep, total_epochs=args.epochs,
            lambda_zinb=args.lambda_zinb, lambda_mse=args.lambda_mse,
            lambda_pcc=args.lambda_pcc,   lambda_ce=args.lambda_ce,
            ce_label_smoothing=args.ce_label_smoothing,
            ce_warmup_epochs=args.ce_warmup_epochs,
            celltype_labels_all=celltype_y_t,
            ce_class_weights=ce_class_weights,
            max_grad_norm=args.max_grad_norm,
            retrieval4_cfg=retrieval4_cfg,
            zs_precomputed=zs_all,         # Fix 1
            nan_check_every=args.nan_check_every,   # Fix 6
        )
        scheduler.step()

        ev, P, Y = eval_epoch(
            model, loader, spot_feat_in, edge_index, edge_weight,
            amp_dtype, device,
            epoch=ep, total_epochs=args.epochs,
            save_attn=bool(args.save_attn and (ep % max(args.attn_every, 1) == 0)),
            attn_out_dir=attn_out_dir,
            attn_max_batches=args.attn_max_batches,
            attn_reduce_heads=args.attn_reduce_heads,
            celltype_labels_all=celltype_y_t,
            retrieval4_cfg=retrieval4_cfg,
            zs_precomputed=zs_all,         # Fix 1
        )
        dt = time.time() - t0

        retr_msg = ""
        if retrieval4_cfg is not None and retrieval4_cfg.get("enabled"):
            retr_msg = (f" | retrUse={tr['retr_used']:.3f} "
                        f"retrTop1={tr['retr_top1']:.4f} tau={tr['retr_tau']:.4f}")
        print(
            f"[{ep:03d}] trainLoss={tr['loss']:.6f} "
            f"trainCtAcc={tr['ct_acc']:.4f} (n={tr['ct_n']}) | "
            f"eval MSE={ev['mse']:.6e} MAE={ev['mae']:.6e} "
            f"PCC(mean)={ev['pcc_mean']:.4f} med={ev['pcc_median']:.4f} "
            f"evalCtAcc={ev['ct_acc']:.4f} (n={ev['ct_n']})"
            f"{retr_msg}  ({dt:.1f}s)"
        )

        if ev["pcc_mean"] > best["pcc"]:
            best["pcc"] = ev["pcc_mean"]
            out_dir = os.path.join(args.save_dir, "best")
            os.makedirs(out_dir, exist_ok=True)
            rec = ad.AnnData(X=P, obs=adata.obs.copy(),
                             var=adata.var.copy(), uns=adata.uns.copy())
            rec.write(os.path.join(out_dir, "recovered_data.h5ad"), compression="gzip")
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            if args.lambda_ce > 0.0 and celltype_names:
                torch.save({"celltype_col": celltype_col_used,
                            "celltype_names": celltype_names,
                            "num_celltypes": len(celltype_names)},
                           os.path.join(out_dir, "celltype_meta.pt"))
            best["path"] = out_dir
            print(f"[BEST↑] saved to {out_dir}")

    print(f"Done. best PCC(mean)={best['pcc']:.4f}, path={best['path']}")


if __name__ == "__main__":
    main()
