"""
run_finetunning_xattn_withceltype_0214_retrieval_final_version.py
-----------------------------------------------------------------
Final retrieval design — three core principles:

1. Per-query probability-proportional quota bank (dynamic, not fixed)
2. Pure cosine-similarity weighted sum (NO prototype mixing, NO averaging)
3. Adaptive per-query temperature τ_q = τ_base × (1 + α × H_norm(p_q))

Low entropy (confident) → τ ≈ τ_base → sharp NN-like retrieval
High entropy (uncertain) → τ ↑ → soft quasi-prototype smoothing, no explicit prototype term

Based on retrieval4_claudever.py; performance optimisations carried over:
  - GCN pre-computed once per epoch
  - Vectorized (B,N) similarity matrix
  - _NodeIdxDataset for pre-computed node indices
  - GradScaler created once in main()
  - persistent_workers=True
  - NaN checks gated by --nan_check_every
"""

import json
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

# Global mapping: string ID -> node index
ID2IDX = None

# Count-scale targets for the ZINB likelihood, indexed by node id and filled in
# main(). Kept module-level so _to_device_batch can attach them without changing
# the dataset's (img, gene, sid) contract.
COUNTS_ALL = None       # (N, G) float32, observed integer counts
SIZE_FACTOR_ALL = None  # (N,)   float32, library_size / target_sum


def pick_amp_dtype(amp_dtype: str, device: torch.device) -> torch.dtype:
    """Resolve AMP dtype from CLI string for the current device."""
    if device.type != "cuda":
        return torch.float32
    if amp_dtype == "auto":
        return (torch.bfloat16
                if torch.cuda.get_device_capability()[0] >= 8
                else torch.float16)
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported --amp_dtype: {amp_dtype}")


# ============================================================
# Path setup
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.modules.pop("models", None)

from models.Foundations import inf_encoder_factory
from models.GCN_update import SpatialGCN
from models.gene_encoder import GeneEncoder
from models.fusion import CrossModalFusion
from models.decoders import ZINBDecoder
from utils.loss_utils import zinb_nll
from utils.wandb_utils import wandb_init, wandb_log, wandb_summary, wandb_finish
from utils.lora_utils import attach_lora_to_foundation
from utils.val_split import checkerboard_split, summarize as val_split_summary
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
from models.spot_features import (build_histology_node_features,
                                  freeze_and_lock_eval, make_spot_in_proj)


# ============================================================
# Pretrained weight loaders
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


def _load_module_from_sd(module, sd, prefixes, name, path):
    """
    Load `module` from the first prefix in `prefixes` that yields overlapping
    keys, and fail loudly if nothing matches or a shape disagrees.

    The previous version wrapped load_state_dict(strict=False) in a bare
    `except Exception: pass`. strict=False only tolerates missing/unexpected
    keys — a shape mismatch still raises — so any architecture drift between
    pretraining and finetuning silently left the module randomly initialized.
    """
    tgt = module.state_dict()
    for pfx in prefixes:
        sub = _strip_prefix(sd, pfx) if pfx else dict(sd)
        common = [k for k in sub if k in tgt]
        if not common:
            continue
        bad = [(k, tuple(sub[k].shape), tuple(tgt[k].shape))
               for k in common
               if hasattr(sub[k], "shape") and sub[k].shape != tgt[k].shape]
        if bad:
            detail = "\n".join(f"    {k}: ckpt{a} vs model{b}" for k, a, b in bad[:8])
            raise RuntimeError(
                f"[PT] shape mismatch loading {name} from {path}:\n{detail}\n"
                "  The checkpoint was produced by a different architecture. "
                "Re-run pretraining with matching --proj_dim / --spot_hidden_dim, "
                "or point --pt_* at the matching checkpoint."
            )
        ret = module.load_state_dict(sub, strict=False)
        missing = [k for k in ret.missing_keys]
        print(f"[PT] loaded {name} from {path} "
              f"({len(common)}/{len(tgt)} tensors"
              + (f", {len(missing)} left at init" if missing else "") + ")")
        return True
    raise RuntimeError(
        f"[PT] no keys matching {name} found in {path}. "
        f"Tried prefixes {prefixes!r}; checkpoint has "
        f"{list(sd)[:5]}{' ...' if len(sd) > 5 else ''}"
    )


def load_pretrained_for_finetune(foundation, gene_encoder, gcn, spot_in_proj=None,
                                 pt_img_backbone=None, pt_ig=None, pt_is=None):
    if pt_img_backbone:
        sd = _load_sd(pt_img_backbone)
        _load_module_from_sd(
            foundation, sd,
            ["img_encoder.foundation.", "foundation.", ""],
            "foundation", pt_img_backbone,
        )
        # The pretrained image branch is foundation -> proj (ImageFoundationAdapter).
        # The GCN was trained on the post-projection embedding, so finetuning must
        # reuse the same proj to build H(0).
        if spot_in_proj is not None and not isinstance(spot_in_proj, nn.Identity):
            _load_module_from_sd(
                spot_in_proj, sd,
                ["img_encoder.proj.", "proj."],
                "spot_in_proj", pt_img_backbone,
            )

    if pt_ig:
        sd = _load_sd(pt_ig)
        _load_module_from_sd(
            gene_encoder, sd, ["gene_encoder.", ""], "gene_encoder", pt_ig,
        )

    if pt_is:
        sd = _load_sd(pt_is)
        _load_module_from_sd(gcn, sd, ["gcn.", ""], "spatial GCN", pt_is)


def load_xattn_into_fusion(fusion: CrossModalFusion, xattn_path):
    if not xattn_path:
        return
    sd = _load_sd(xattn_path)

    def _load_block(dst_block, *src_prefixes):
        for src_prefix in src_prefixes:
            sub = {k[len(src_prefix) + 1:]: v for k, v in sd.items()
                   if k.startswith(src_prefix + ".")}
            if not sub:
                continue
            ret = dst_block.load_state_dict(sub, strict=False)
            if ret.missing_keys or ret.unexpected_keys:
                print(f"[XATTN][warn] {src_prefix}: missing={len(ret.missing_keys)}, "
                      f"unexpected={len(ret.unexpected_keys)}")
            else:
                print(f"[XATTN] loaded {src_prefix}")
            return
        print(f"[XATTN][warn] no keys for {'/'.join(src_prefixes)}")

    _load_block(fusion.xa_ig, "i2g")
    # "i2s" is the current name; "i2c" is the pre-rename (coordinate-era) key.
    _load_block(fusion.xa_is, "i2s", "i2c")


# ============================================================
# Cell type helpers
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
# Pre-compute string → int node indices at dataset level
# ============================================================
def _norm_id_str(x) -> str:
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


class _NodeIdxDataset(torch.utils.data.Dataset):
    """
    Wraps SpotImageGeneDataset so the DataLoader returns an integer node index
    instead of a string spot ID.
    Returns: (img_tensor, gene_tensor, node_idx_int)
    """
    def __init__(self, base_ds: SpotImageGeneDataset, id_to_idx: dict,
                 skip_images: bool = False, subset: np.ndarray = None):
        self._base = base_ds
        self._skip_images = bool(skip_images)
        # `subset` holds node ids (not dataset positions); used to build the
        # disjoint train / validation loaders of the R1-4 spatial split.
        self._subset = None if subset is None else np.asarray(subset, dtype=np.int64)
        self._node_indices = np.array(
            [id_to_idx[_norm_id_str(base_ds.map_df.iloc[i]["spot_id"])]
             for i in range(len(base_ds))],
            dtype=np.int64,
        )
        if self._subset is None:
            self._positions = None
        else:
            want = set(self._subset.tolist())
            self._positions = np.array(
                [p for p, n in enumerate(self._node_indices) if n in want],
                dtype=np.int64,
            )

    def __len__(self):
        return len(self._base) if self._positions is None else len(self._positions)

    def __getitem__(self, i: int):
        idx = i if self._positions is None else int(self._positions[i])
        node = int(self._node_indices[idx])
        if self._skip_images:
            # Backbone frozen and cached -> never touch the PNG.
            row = self._base.map_df.iloc[idx]
            irow = self._base.adata.obs_names.get_loc(str(row["spot_id"]))
            x = self._base.adata.X[irow, self._base._gene_index]
            if hasattr(x, "toarray"):
                x = x.toarray()
            gene = torch.from_numpy(np.asarray(x).reshape(-1).astype(np.float32))
            return torch.empty(0), gene, node
        img, gene, _ = self._base[idx]
        return img, gene, node


def make_node_idx_dataloader(base_ds, id_to_idx, batch_size, num_workers,
                              shuffle=True, skip_images=False, subset=None):
    ds = _NodeIdxDataset(base_ds, id_to_idx, skip_images=skip_images, subset=subset)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


# ============================================================
# Model
# ============================================================
class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn,
                 dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1,
                 merge="gated-sum",
                 img_tokens=1, spot_tokens=1, gene_tokens=1,
                 decoder_hidden=1024, gene_dim=1000,
                 num_celltypes=0, cls_head_hidden=0, cls_dropout=0.0,
                 dim_spot_in=256, use_spot=True, use_gene=True):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.gcn = gcn
        # modality ablation (R2): drop the spot (coords/GCN) and/or gene branch in fusion
        self.use_spot = bool(use_spot)
        self.use_gene = bool(use_gene)

        # H(0) for the spatial GCN is the histology embedding of each spot, at
        # the width the GCN was pretrained on. Mirrors the pretraining
        # ImageFoundationAdapter's projection head so the weights transfer.
        self.dim_spot_in = int(dim_spot_in)
        self.spot_in_proj = make_spot_in_proj(dim_img, dim_spot_in)

        # Optional (N, dim_img) cache of backbone outputs, valid only while the
        # backbone is frozen. A plain attribute, not a buffer: it is derived
        # data and must stay out of the checkpoint.
        self.img_feat_cache = None

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

    def encode_image(self, batch):
        """
        Backbone embedding for the batch. Served from img_feat_cache when the
        backbone is frozen, which removes both the ViT forward and the patch
        decode from every step.
        """
        if self.img_feat_cache is not None:
            return self.img_feat_cache[batch["idx"].long()]
        return self.img_encoder(batch["image"])

    @torch.no_grad()
    def build_spot_features(self, img_loader, n_nodes, device, amp_dtype=None):
        """
        H(0) for every node: histology embedding -> spot_in_proj.
        `img_loader` must be an ordered loader yielding node indices.
        """
        if self.img_feat_cache is not None:
            return self.spot_in_proj(self.img_feat_cache.float()).float()
        return build_histology_node_features(
            img_encoder=self.img_encoder,
            loader=img_loader,
            n_nodes=n_nodes,
            out_dim=self.dim_spot_in,
            device=device,
            proj=self.spot_in_proj,
            amp_dtype=amp_dtype,
        )

    def forward(self, batch, spot_feat_in, edge_index, edge_weight=None,
                return_attn=False, zs_precomputed=None):
        """
        zs_precomputed: (N, dim_spot) tensor — pre-computed GCN output for all nodes.
        If provided, the GCN call is skipped (fix: avoids full-graph GCN per batch).
        """
        x_gene_log = batch["expr"].float()
        node_idx   = batch["idx"].long()

        zi = self.encode_image(batch)
        zg = self.gene_encoder(x_gene_log)

        if zs_precomputed is not None:
            zs = zs_precomputed[node_idx]
        else:
            spot_fp32 = spot_feat_in.float()
            ew_fp32   = edge_weight.float() if edge_weight is not None else None
            if node_idx.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    zs_all = self.gcn(spot_fp32, edge_index, ew_fp32)
            else:
                zs_all = self.gcn(spot_fp32, edge_index, ew_fp32)
            zs = zs_all[node_idx]

        zi = self.proj_img(zi)
        zg = self.proj_gene(zg)
        zs = self.proj_spot(zs)

        if return_attn:
            z_fused, z_is, z_ig, attn = self.fusion(zi, zs, zg, return_attn=True,
                                                    use_spot=self.use_spot, use_gene=self.use_gene)
        else:
            z_fused, z_is, z_ig = self.fusion(zi, zs, zg, return_attn=False,
                                              use_spot=self.use_spot, use_gene=self.use_gene)
            attn = None

        mu, theta, pi = self.decoder(z_fused)
        cls_logits = self.cls_head(z_fused) if self.cls_head is not None else None
        return mu, theta, pi, {
            "z_fused": z_fused, "z_is": z_is, "z_ig": z_ig,
            "attn": attn, "cls_logits": cls_logits,
        }


# ============================================================
# Pre-compute GCN embeddings once per epoch
# ============================================================
@torch.no_grad()
def _precompute_gcn_all(model, spot_feat_in, edge_index, edge_weight):
    was_training = model.training
    model.eval()
    ew_fp32 = edge_weight.float() if edge_weight is not None else None
    with torch.amp.autocast("cuda", enabled=False):
        zs_all = model.gcn(spot_feat_in.float(), edge_index, ew_fp32)
    if was_training:
        model.train()
    return zs_all.detach()


# ============================================================
# Batch helper
# ============================================================
def _to_device_batch(batch, device):
    global ID2IDX

    def _to_tensor(x, device, dtype=None):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.as_tensor(x, dtype=dtype, device=device)

    if isinstance(batch, dict):
        out = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in batch.items()}
        if COUNTS_ALL is not None and "counts" not in out and "idx" in out:
            cpu_idx = out["idx"].detach().cpu().long()
            out["counts"] = COUNTS_ALL[cpu_idx].to(device, non_blocking=True)
            out["size_factor"] = SIZE_FACTOR_ALL[cpu_idx].to(device, non_blocking=True)
        return out

    if isinstance(batch, (list, tuple)):
        if len(batch) != 3:
            raise TypeError(f"Unexpected batch length {len(batch)}; expected 3.")
        x_img, x_gene, idx = batch
        x_img  = _to_tensor(x_img,  device)
        x_gene = _to_tensor(x_gene, device)

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

        out = {"image": x_img, "expr": x_gene, "idx": idx_tensor}
        if COUNTS_ALL is not None:
            cpu_idx = idx_tensor.detach().cpu()
            out["counts"] = COUNTS_ALL[cpu_idx].to(device, non_blocking=True)
            out["size_factor"] = SIZE_FACTOR_ALL[cpu_idx].to(device, non_blocking=True)
        return out

    raise TypeError(f"Unsupported batch type: {type(batch)}")


# ============================================================
# Misc helpers
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


# ============================================================
# FINAL RETRIEVAL: entropy-adaptive per-query temperature
# ============================================================
def _compute_per_query_tau(query_probs, tau_base, tau_entropy_alpha):
    """
    Compute per-query temperature τ_q = τ_base × (1 + α × H_norm(p_q)).

    H(p) = -Σ p·log(p)
    H_norm = H / log(C)  ∈ [0, 1]  (0 = fully confident, 1 = uniform)

    Low entropy → τ_q ≈ τ_base  → sharp, NN-like retrieval
    High entropy → τ_q ↑        → soft quasi-prototype (no explicit prototype term)

    Returns: (B,) float32 tensor, clamped to [1e-3, 50].
    """
    p = query_probs.float().clamp(min=1e-9)  # (B, C)
    C = float(p.size(1))
    # entropy per query
    H = -(p * torch.log(p)).sum(dim=1)       # (B,)
    H_norm = H / float(np.log(C) if C > 1 else 1.0)  # (B,) in [0, 1]
    tau_q = tau_base * (1.0 + tau_entropy_alpha * H_norm)
    return tau_q.clamp(min=1e-3, max=50.0)   # (B,)


def _allocate_cluster_counts(prob_vec, total):
    """Integer quota allocation: floor + fractional remainder greedy."""
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


@torch.no_grad()
def _build_retrieval_final_banks(model, loader, spot_feat_in, edge_index, edge_weight,
                                  expr_bank_np, num_celltypes, device, amp_dtype,
                                  n_nodes, pred_min_conf=0.0,
                                  zs_precomputed=None, candidate_nodes=None):
    """
    Build the key bank (N, D) and probability bank (N, C), and the per-class
    candidate index map used for retrieval.

    `loader` must cover every node, because each node needs its own key and
    class posterior as a *query*. `candidate_nodes` restricts what may be
    *retrieved*: passing the training node ids keeps validation cells out of the
    bank, so a validation prediction can never retrieve its own expression
    (reviewer comment R1-4). Leaving it None reproduces the old behaviour where
    every node was retrievable.
    """
    was_training = model.training
    model.eval()
    dim = int(model.proj_img.out_features)
    key_bank       = torch.zeros((n_nodes, dim), dtype=torch.float32, device=device)
    pred_cls_bank  = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    pred_prob_bank = torch.zeros((n_nodes, int(num_celltypes)),
                                 dtype=torch.float32, device=device)
    seen = torch.zeros((n_nodes,), dtype=torch.bool, device=device)

    pbar = tqdm(loader, desc="[RetrFinal] build key/prob banks", leave=False)
    for batch in pbar:
        batch = _to_device_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            q = model.proj_img(model.encode_image(batch))
            _, _, _, aux = model(batch, spot_feat_in, edge_index, edge_weight,
                                 return_attn=False, zs_precomputed=zs_precomputed)

        q = F.normalize(q.float(), dim=1)
        cls_logits = aux.get("cls_logits", None)
        if cls_logits is None:
            raise RuntimeError("[RetrFinal] cls_head required but cls_logits is None.")
        probs   = torch.softmax(cls_logits.float(), dim=1)
        _, pred_cls = torch.max(probs, dim=1)
        idx = batch["idx"].long()
        key_bank[idx]       = q
        pred_cls_bank[idx]  = pred_cls.long()
        pred_prob_bank[idx] = probs.float()
        seen[idx] = True

    miss = int((~seen).sum().item())
    if miss > 0:
        print(f"[RetrFinal][warn] {miss} nodes not seen while building banks.")

    pred_cls_np  = pred_cls_bank.cpu().numpy()
    pred_prob_np = pred_prob_bank.cpu().numpy()
    pred_conf_np = pred_prob_np.max(axis=1)

    allowed = np.ones(n_nodes, dtype=bool)
    if candidate_nodes is not None:
        allowed[:] = False
        allowed[np.asarray(candidate_nodes, dtype=np.int64)] = True

    class_index_map = {}
    for c in range(int(num_celltypes)):
        m = np.where((pred_cls_np == c) & (pred_conf_np >= float(pred_min_conf))
                     & allowed)[0]
        if m.size > 0:
            class_index_map[int(c)] = m.astype(np.int64)

    if was_training:
        model.train()
    return {"keys_bank": key_bank, "pred_cls_bank": pred_cls_bank,
            "pred_prob_bank": pred_prob_bank, "class_index_map": class_index_map}


def _retrieval_final_expr(*args, **kwargs):
    """
    FP32 wrapper around the retrieval step.

    Callers run inside an fp16 autocast region, which would push the
    (B, N_cells) similarity matmul and the temperature-scaled softmax into half
    precision. Cosine similarities divided by tau~0.07 have almost no headroom
    in fp16, and on Blackwell the fp16 cublasGemmEx for this shape fails
    outright. The cost of forcing fp32 here is negligible next to the ZINB head.
    """
    with torch.amp.autocast("cuda", enabled=False):
        return _retrieval_final_expr_impl(*args, **kwargs)


def _retrieval_final_expr_impl(
    query_key, query_probs, x_log_true,
    keys_bank, expr_bank_np, class_index_map,
    bank_size=5000, tau_base=0.07, tau_entropy_alpha=1.0,
    prob_topm=0, fallback="global", allowed_mask=None,
):
    """
    Final retrieval: per-query entropy-adaptive temperature + pure cosine-weighted sum.

    Steps per query:
      1. Compute τ_q via _compute_per_query_tau
      2. Optionally mask to top-m classes, renormalize
      3. Quota allocation: quota[c] = _allocate_cluster_counts(prob_vec, target)
      4. Stage 1: for each class c, take topk cosine from class_index_map[c]
      5. Stage 2: fill remaining slots from global top-cosine (no duplicates)
      6. Fallback: global top-k if completely empty
      7. Pure weighted sum: w = softmax(cos(q, bank_sel) / τ_q), x_retr = w @ expr_bank[sel]
         — no prototype, no averaging, every row is a real training instance

    Returns: (x_retr, used_mask, metrics_dict)
    """
    device = query_key.device
    q   = F.normalize(query_key.float(), dim=1)  # (B, D)
    bsz = int(q.size(0))
    n_nodes     = int(keys_bank.size(0))
    target_bank = int(max(1, min(int(bank_size), n_nodes)))

    # Compute per-query tau (B,)
    tau_q = _compute_per_query_tau(query_probs, tau_base, tau_entropy_alpha)  # (B,)

    x_retr = x_log_true.clone()
    used   = torch.zeros((bsz,), dtype=torch.bool, device=device)
    top1_vals = []
    tau_vals  = []

    probs = query_probs.float().to(device=device)
    n_cls = int(probs.size(1))
    if prob_topm > 0 and prob_topm < n_cls:
        vals_m, idx_m = torch.topk(probs, k=prob_topm, dim=1, largest=True, sorted=False)
        probs_topm = torch.zeros_like(probs)
        probs_topm.scatter_(1, idx_m, vals_m)
        probs = probs_topm
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    # Single batched (B, N) similarity matrix — no per-sample GPU matmul in the loop
    sim_all = q @ keys_bank.float().t()  # (B, N), fp32
    # Stage 2 fills any shortfall from global similarity; without this mask it
    # would happily pull in validation cells that stage 1 excluded (R1-4).
    if allowed_mask is not None:
        sim_all = sim_all.masked_fill(~allowed_mask.to(sim_all.device).unsqueeze(0), -1e9)

    for bi in range(bsz):
        sim_bi   = sim_all[bi]                                   # (N,)
        prob_vec = probs[bi].detach().cpu().numpy().astype(np.float64)
        quota    = _allocate_cluster_counts(prob_vec, target_bank)
        tq_bi    = float(tau_q[bi].item())

        selected_parts = []
        for ci, take_n in enumerate(quota.tolist()):
            if take_n <= 0:
                continue
            cand_np = class_index_map.get(int(ci), None)
            if cand_np is None or cand_np.size == 0:
                continue
            cand_t   = torch.as_tensor(cand_np, device=device, dtype=torch.long)
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

        # Stage 2: fill shortfall from global similarity (reuse sim_bi)
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

        # Fallback
        if selected_np.size == 0:
            if fallback == "global":
                kk = min(target_bank, int(sim_bi.numel()))
                _, loc = torch.topk(sim_bi, k=kk, dim=0, largest=True, sorted=True)
                selected_np = loc.detach().cpu().numpy().astype(np.int64)
            else:
                continue

        sel_t   = torch.as_tensor(selected_np, device=device, dtype=torch.long)
        sim_sel = sim_bi[sel_t]
        if sim_sel.numel() > target_bank:
            vals_sel, loc_sel = torch.topk(sim_sel, k=target_bank,
                                           dim=0, largest=True, sorted=True)
            sel_t   = sel_t.index_select(0, loc_sel)
            sim_sel = vals_sel
        if sim_sel.numel() == 0:
            continue

        # Pure cosine-similarity weighted sum with per-query τ
        w = torch.softmax(sim_sel / tq_bi, dim=0)
        top1_vals.append(sim_sel.max().detach())
        tau_vals.append(tq_bi)
        expr_sel = torch.from_numpy(
            expr_bank_np[sel_t.detach().cpu().numpy()]
        ).to(device=device, dtype=x_log_true.dtype)
        x_retr[bi] = (w.unsqueeze(-1) * expr_sel).sum(dim=0)
        used[bi] = True

    top1_mean = float(torch.stack(top1_vals).mean().item()) if top1_vals else float("nan")
    tau_mean  = float(np.mean(tau_vals)) if tau_vals else float("nan")
    return x_retr, used, {"top1_mean": top1_mean, "tau_mean": tau_mean}


def _allocate_cluster_counts_batch(probs_np, total):
    """
    Vectorized `_allocate_cluster_counts` over a whole batch.

    Same rule: floor(p*total), then hand the remainder to the largest fractional
    parts. np.argsort is stable and ties in `raw - cnt` are broken by class index
    exactly as the per-sample loop did.
    """
    B, C = probs_np.shape
    total = int(max(total, 0))
    if total == 0:
        return np.zeros((B, C), dtype=np.int64)
    p = np.clip(probs_np.astype(np.float64), 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    p = np.where(s > 0, p / np.maximum(s, 1e-300), 1.0 / max(C, 1))
    raw = p * float(total)
    cnt = np.floor(raw).astype(np.int64)
    rem = total - cnt.sum(axis=1)                      # (B,)
    frac = raw - cnt.astype(np.float64)
    order = np.argsort(-frac, axis=1, kind="stable")   # (B, C)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(C)[None, :].repeat(B, 0), axis=1)
    cnt += (rank < rem[:, None]).astype(np.int64)
    return cnt


@torch.no_grad()
def _retrieval_final_expr_fast(
    query_key, query_probs, x_log_true,
    keys_bank, expr_bank_np, class_index_map,
    bank_size=5000, tau_base=0.07, tau_entropy_alpha=1.0,
    prob_topm=0, fallback="global", allowed_mask=None,
    cls_of=None, expr_bank_t=None,
):
    """
    Batched rewrite of _retrieval_final_expr_impl.

    The per-sample version issued roughly 2,700 GPU->CPU synchronisations per
    step (one per query for the probability vector, plus one per query per class
    for the top-k indices), which dominated the step time: 0.53 s/step measured
    with the GPU otherwise idle. This version does the same work with two syncs.

    It relies on one property of the original: class_index_map is built from
    argmax predictions, so the classes are disjoint. The np.unique() in the
    original therefore only sorted - it never actually removed duplicates - and
    the per-class selections can be concatenated directly.
    """
    device = query_key.device
    q = F.normalize(query_key.float(), dim=1)
    B = int(q.size(0))
    N = int(keys_bank.size(0))
    target = int(max(1, min(int(bank_size), N)))

    tau_q = _compute_per_query_tau(query_probs, tau_base, tau_entropy_alpha)   # (B,)

    probs = query_probs.float().to(device)
    C = int(probs.size(1))
    if prob_topm > 0 and prob_topm < C:
        vals_m, idx_m = torch.topk(probs, k=prob_topm, dim=1, largest=True, sorted=False)
        probs = torch.zeros_like(probs).scatter_(1, idx_m, vals_m)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    sim_all = q @ keys_bank.float().t()                                        # (B, N)
    if allowed_mask is not None:
        sim_all = sim_all.masked_fill(~allowed_mask.to(device).unsqueeze(0), -1e9)

    # sync #1: quotas are integer bookkeeping, cheapest on the host
    quota = _allocate_cluster_counts_batch(probs.detach().cpu().numpy(), target)  # (B, C)
    quota_t = torch.from_numpy(quota).to(device)

    NEG = torch.finfo(sim_all.dtype).min
    sel_idx_parts, sel_val_parts = [], []
    for c in range(C):
        k_c = int(quota[:, c].max())
        if k_c <= 0:
            continue
        members = cls_of.get(c) if isinstance(cls_of, dict) else None
        if members is None or members.numel() == 0:
            continue
        k_c = min(k_c, int(members.numel()))
        sub = sim_all.index_select(1, members)                                 # (B, |c|)
        vals, loc = torch.topk(sub, k=k_c, dim=1, largest=True, sorted=True)
        idx = members[loc]                                                     # (B, k_c)
        keep = (torch.arange(k_c, device=device).unsqueeze(0) < quota_t[:, c:c + 1])
        sel_idx_parts.append(torch.where(keep, idx, torch.full_like(idx, -1)))
        sel_val_parts.append(torch.where(keep, vals, torch.full_like(vals, NEG)))

    if sel_idx_parts:
        sel_idx = torch.cat(sel_idx_parts, dim=1)                              # (B, K)
        sel_val = torch.cat(sel_val_parts, dim=1)
    else:
        sel_idx = torch.full((B, 1), -1, dtype=torch.long, device=device)
        sel_val = torch.full((B, 1), NEG, dtype=sim_all.dtype, device=device)

    n_sel = (sel_idx >= 0).sum(dim=1)                                          # (B,)
    need = torch.clamp(target - n_sel, min=0)
    need_max = int(need.max().item())                                          # sync #2

    if need_max > 0:
        fill_src = sim_all.clone()
        valid = sel_idx >= 0
        fill_src.scatter_(1, torch.where(valid, sel_idx, torch.zeros_like(sel_idx)),
                          torch.where(valid, torch.full_like(sel_val, NEG),
                                      fill_src.gather(1, torch.where(valid, sel_idx,
                                                                     torch.zeros_like(sel_idx)))))
        fvals, fidx = torch.topk(fill_src, k=min(need_max, N), dim=1,
                                 largest=True, sorted=True)
        fkeep = (torch.arange(fvals.size(1), device=device).unsqueeze(0) < need.unsqueeze(1))
        sel_idx = torch.cat([sel_idx, torch.where(fkeep, fidx, torch.full_like(fidx, -1))], 1)
        sel_val = torch.cat([sel_val, torch.where(fkeep, fvals, torch.full_like(fvals, NEG))], 1)

    # keep the `target` best per query; invalid slots carry NEG and sort to the end
    k_final = min(target, sel_val.size(1))
    top_val, top_pos = torch.topk(sel_val, k=k_final, dim=1, largest=True, sorted=True)
    top_idx = sel_idx.gather(1, top_pos)
    valid = (top_idx >= 0) & (top_val > NEG / 2)

    w = torch.softmax(top_val.masked_fill(~valid, NEG) / tau_q.unsqueeze(1), dim=1)
    w = torch.where(valid, w, torch.zeros_like(w))
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

    gather_idx = torch.where(valid, top_idx, torch.zeros_like(top_idx))
    # Accumulate in chunks over the bank axis rather than as one batched matmul:
    # materialising (B, k, G) would be ~800 MB at k=5000, and cuBLAS's strided
    # batched GEMM fails outright on this machine (same defect as the fp16 path).
    G = int(expr_bank_t.size(1))
    x_retr = torch.zeros((B, G), dtype=torch.float32, device=device)
    step = max(1, min(512, k_final))
    for ks in range(0, k_final, step):
        ke = min(ks + step, k_final)
        sel = expr_bank_t[gather_idx[:, ks:ke].reshape(-1)].view(B, ke - ks, G).float()
        x_retr += (w[:, ks:ke].unsqueeze(-1).float() * sel).sum(dim=1)
    x_retr = x_retr.to(x_log_true.dtype)

    used = valid.any(dim=1)
    x_retr = torch.where(used.unsqueeze(1), x_retr, x_log_true)

    top1 = top_val.masked_fill(~valid, NEG).max(dim=1).values
    metrics = {"top1_mean": float(top1[used].mean().item()) if bool(used.any()) else float("nan"),
               "tau_mean": float(tau_q.mean().item())}
    return x_retr, used, metrics


def _current_retrieval_tau_base(model, tau_default=0.07):
    """Return the current tau_base scalar (learnable or fixed)."""
    dev = next(model.parameters()).device
    if hasattr(model, "retrieval_log_tau"):
        t = torch.exp(getattr(model, "retrieval_log_tau"))
        return float(torch.clamp(t, min=1e-3, max=50.0).item())
    if hasattr(model, "retrieval_tau_const"):
        t = getattr(model, "retrieval_tau_const")
        return float(torch.clamp(t.to(device=dev), min=1e-3, max=50.0).item())
    return float(max(tau_default, 1e-3))


# ============================================================
# Training loop
# ============================================================
def train_epoch(model, loader, spot_feat_in, edge_index, edge_weight,
                optimizer, scaler,
                amp_dtype=torch.float32,
                device=torch.device("cpu"),
                epoch=1, total_epochs=1,
                lambda_zinb=1.0, lambda_mse=0.1,
                lambda_pcc=0.0, lambda_ce=0.0,
                ce_label_smoothing=0.0, ce_warmup_epochs=0,
                pcc_warmup_epochs=0,
                celltype_labels_all=None, ce_class_weights=None,
                max_grad_norm=1.0,
                retrieval_cfg=None,
                zs_precomputed=None,
                nan_check_every=50):
    model.train()
    loss_sum, n = 0.0, 0
    cls_correct, cls_count = 0.0, 0
    retr_used_sum, retr_top1_sum, retr_tau_sum, retr_steps = 0.0, 0.0, 0.0, 0
    eps = 1e-8

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}/{total_epochs}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch = _to_device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        do_nan_check = (step % nan_check_every == 0)

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            x_log = batch["expr"].float()
            batch_in  = batch
            loss_retr = x_log.new_zeros(())
            retr_used_ratio = 0.0

            if retrieval_cfg is not None and bool(retrieval_cfg.get("enabled", False)):
                keys_bank      = retrieval_cfg.get("keys_bank")
                pred_prob_bank = retrieval_cfg.get("pred_prob_bank")
                expr_bank_np   = retrieval_cfg.get("expr_bank_np")
                class_index_map = retrieval_cfg.get("class_index_map", {})

                if keys_bank is not None and expr_bank_np is not None and pred_prob_bank is not None:
                    q        = model.proj_img(model.encode_image(batch))
                    tau_base = _current_retrieval_tau_base(model, float(retrieval_cfg.get("tau_base", 0.07)))
                    tau_alpha = float(retrieval_cfg.get("tau_entropy_alpha", 1.0))

                    x_retr, retr_used, retr_metrics = _retrieval_final_expr_fast(
                        query_key=q,
                        query_probs=pred_prob_bank[batch["idx"].long()],
                        x_log_true=x_log,
                        keys_bank=keys_bank,
                        expr_bank_np=expr_bank_np,
                        class_index_map=class_index_map,
                        bank_size=int(retrieval_cfg.get("bank_size", 5000)),
                        tau_base=tau_base,
                        tau_entropy_alpha=tau_alpha,
                        prob_topm=int(retrieval_cfg.get("prob_topm", 0)),
                        allowed_mask=retrieval_cfg.get("allowed_mask"),
                        cls_of=retrieval_cfg.get("cls_of"),
                        expr_bank_t=retrieval_cfg.get("expr_bank_t"),
                        fallback=str(retrieval_cfg.get("fallback", "global")),
                    )

                    t1 = retr_metrics.get("top1_mean", float("nan"))
                    tm = retr_metrics.get("tau_mean", float("nan"))
                    if not np.isnan(t1):
                        retr_top1_sum += float(t1)
                        retr_steps    += 1
                    if not np.isnan(tm):
                        retr_tau_sum  += float(tm)

                    retr_used_ratio = float(retr_used.float().mean().item())
                    retr_used_sum  += retr_used_ratio

                    mix_base = float(retrieval_cfg.get("mix", 0.0))
                    mix_warm = int(retrieval_cfg.get("mix_warmup_epochs", 0))
                    mix = mix_base * min(1.0, float(epoch) / float(max(mix_warm, 1))) \
                          if mix_warm > 0 else mix_base
                    mix = float(np.clip(mix, 0.0, 1.0))
                    batch_in = dict(batch)
                    batch_in["expr"] = (1.0 - mix) * x_log + mix * x_retr

                    if float(retrieval_cfg.get("lambda_retr", 0.0)) > 0.0:
                        loss_retr = F.mse_loss(x_retr, x_log)

            mu, theta, pi, aux = model(
                batch_in, spot_feat_in, edge_index, edge_weight,
                zs_precomputed=zs_precomputed,
            )

            if do_nan_check and (torch.isnan(mu).any() or torch.isinf(mu).any()):
                print(f"[WARN] Step {step}: mu NaN/Inf, skipping"); continue
            if do_nan_check and (torch.isnan(theta).any() or torch.isinf(theta).any()):
                print(f"[WARN] Step {step}: theta NaN/Inf, skipping"); continue

            # ---- ZINB on observed counts, depth carried by the mean ----------
            # Previously: x_raw = expm1(x_log). expm1 inverts log1p but NOT the
            # library-size normalization applied before it, so the target was a
            # continuous CP10K quantity, outside the support of a negative
            # binomial, and per-cell sequencing depth had already been divided
            # out before the likelihood saw it. Two cells with 1/500 and 10/5000
            # counts both became 20, despite differing threefold in measurement
            # noise, and a zero at depth 500 became indistinguishable from a zero
            # at depth 5000 — so pi learned depth rather than dropout.
            #
            # Standard count-model form instead (scVI, ZINB-WaVE):
            #     mu_ig = s_i * mu_hat_ig,   s_i = (sum_g x_ig) / 1e4
            # The decoder keeps predicting on the normalized scale it was fit to;
            # the size factor puts it on the count scale so Var = mu + mu^2/theta
            # adapts per cell and a single theta is valid across depths.
            x_count = batch["counts"].float()
            s_i = batch["size_factor"].float().unsqueeze(1)     # (B, 1)
            mu_count = torch.clamp(s_i * mu, min=1e-8)
            loss_zinb = zinb_nll(x_count, mu_count, theta, pi, reduction="mean")

            if torch.isnan(loss_zinb) or torch.isinf(loss_zinb):
                print(f"[WARN] Step {step}: ZINB loss NaN/Inf, skipping"); continue

            # The auxiliary reconstruction term stays in log1p(CP10K) — the same
            # space as x_log, as the encoder input and as every reported metric.
            # It is deliberately NOT moved to the count scale: mixing a
            # count-scale NLL with a count-scale MSE would let a handful of
            # high-depth cells dominate. The two terms therefore live on
            # different scales by design, which is why their weights are not
            # comparable numbers; both magnitudes are logged so lambda_zinb /
            # lambda_mse can be set from observed sizes rather than by analogy.
            mu_log   = torch.log1p(torch.clamp(mu, min=eps))
            loss_mse = F.mse_loss(mu_log, x_log)
            if torch.isnan(loss_mse) or torch.isinf(loss_mse):
                print(f"[WARN] Step {step}: MSE loss NaN/Inf, skipping"); continue

            if lambda_pcc > 0.0:
                # fp32, not the ambient autocast dtype. Under bf16 the centred
                # prediction Yhc rounds to exactly zero for ~80% of genes while
                # the decoder is still near-constant (measured: 435/541 genes at
                # batch 128), which makes num/den 0/eps for those genes and feeds
                # the backbone pure noise.
                x32  = x_log.float()
                mu32 = mu_log.float()
                Yc  = x32 - x32.mean(dim=0, keepdim=True)
                Yhc = mu32 - mu32.mean(dim=0, keepdim=True)
                num = (Yc * Yhc).sum(dim=0)
                # Clamp each variance *inside* the sqrt. The original
                # `sqrt(a*b) + 1e-8` lets the denominator approach zero whenever
                # the decoder output is near-constant across the batch — which is
                # exactly its state at initialisation — sending ‖∂L/∂mu‖ to ~4300x
                # the reconstruction gradient. clip_grad_norm_(max_norm=1.0) then
                # rescales the whole update by 1/‖g‖, so the ZINB/MSE signal is
                # annihilated and training sits at its initial loss indefinitely.
                den = (Yc.pow(2).sum(dim=0).clamp_min(1e-3)
                       * Yhc.pow(2).sum(dim=0).clamp_min(1e-3)).sqrt()
                pcc_loss = 1.0 - (num / den).clamp(-1.0, 1.0).mean()
                # Even clamped, the term still outweighs the reconstruction
                # gradient by ~150x while the decoder is untrained, so ramp it in
                # the way ce_loss is ramped rather than applying it from step 1.
                # Unlike ce_warmup_epochs this ramp starts at 0 for epoch 1, not
                # at 1/N. Epoch 1 is precisely when the decoder is degenerate: the
                # baseline needs ~60 steps to get ZINB from 10.5 down to 1.2, and
                # any PCC weight during that window suppresses the gradient that
                # does the work. From epoch 2 the output has real per-gene
                # variance and the term is ~70x better conditioned.
                if pcc_warmup_epochs > 0:
                    pcc_loss = pcc_loss * min(
                        1.0, max(0.0, float(epoch) - 1.0)
                        / float(max(int(pcc_warmup_epochs), 1)))
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

            lam_retr = float(retrieval_cfg.get("lambda_retr", 0.0)
                             if retrieval_cfg is not None else 0.0)
            loss = (lambda_zinb * loss_zinb + lambda_mse * loss_mse
                    + lambda_pcc * pcc_loss + lambda_ce * ce_loss
                    + lam_retr * loss_retr)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Step {step}: total loss NaN/Inf, skipping"); continue

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
            "Lz":   f"{loss_zinb.item():.4f}",
            "Lm":   f"{loss_mse.item():.4f}",
            "Lp":   f"{(pcc_loss.item() if lambda_pcc > 0.0 else 0.0):.4f}",
            "Lc":   f"{(ce_loss.item() if lambda_ce > 0.0 else 0.0):.4f}",
            "Lr":   f"{(loss_retr.item() if retrieval_cfg is not None else 0.0):.4f}",
            "Ruse": f"{retr_used_ratio:.2f}",
            "Cacc": f"{(ce_acc if ce_seen > 0 else 0.0):.3f}",
        })

    n_steps = max(len(loader), 1)
    return {
        "loss":       loss_sum / max(n, 1),
        "ct_acc":     float(cls_correct / max(cls_count, 1)),
        "ct_n":       int(cls_count),
        "retr_used":  float(retr_used_sum / n_steps),
        "retr_top1":  float(retr_top1_sum / max(retr_steps, 1)) if retr_steps > 0 else float("nan"),
        "retr_tau":   float(retr_tau_sum  / max(retr_steps, 1)) if retr_steps > 0 else float("nan"),
    }


# ============================================================
# Eval loop
# ============================================================
@torch.no_grad()
def eval_epoch(model, loader, spot_feat_in, edge_index, edge_weight,
               amp_dtype=torch.float32, device=torch.device("cpu"),
               epoch=1, total_epochs=1,
               save_attn=False, attn_out_dir=None,
               attn_max_batches=0, attn_reduce_heads=False,
               celltype_labels_all=None,
               retrieval_cfg=None,
               zs_precomputed=None):
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    cls_correct, cls_count = 0.0, 0
    preds, gts, node_idxs = [], [], []
    retr_top1_vals, retr_use_vals, x_retr_all = [], [], []
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

            if retrieval_cfg is not None and bool(retrieval_cfg.get("enabled", False)):
                keys_bank      = retrieval_cfg.get("keys_bank")
                pred_prob_bank = retrieval_cfg.get("pred_prob_bank")
                expr_bank_np   = retrieval_cfg.get("expr_bank_np")
                class_index_map = retrieval_cfg.get("class_index_map", {})
                if keys_bank is not None and expr_bank_np is not None and pred_prob_bank is not None:
                    q        = model.proj_img(model.encode_image(batch))
                    tau_base = _current_retrieval_tau_base(model, float(retrieval_cfg.get("tau_base", 0.07)))
                    tau_alpha = float(retrieval_cfg.get("tau_entropy_alpha", 1.0))
                    x_retr, retr_used_e, rm = _retrieval_final_expr_fast(
                        query_key=q,
                        query_probs=pred_prob_bank[batch["idx"].long()],
                        x_log_true=x_log,
                        keys_bank=keys_bank, expr_bank_np=expr_bank_np,
                        class_index_map=class_index_map,
                        bank_size=int(retrieval_cfg.get("bank_size", 5000)),
                        tau_base=tau_base,
                        tau_entropy_alpha=tau_alpha,
                        prob_topm=int(retrieval_cfg.get("prob_topm", 0)),
                        allowed_mask=retrieval_cfg.get("allowed_mask"),
                        cls_of=retrieval_cfg.get("cls_of"),
                        expr_bank_t=retrieval_cfg.get("expr_bank_t"),
                        fallback=str(retrieval_cfg.get("fallback", "global")),
                    )
                    mix = float(np.clip(float(retrieval_cfg.get("eval_mix", 1.0)), 0.0, 1.0))
                    batch_in = dict(batch)
                    batch_in["expr"] = (1.0 - mix) * x_log + mix * x_retr
                    # R1-3: the retrTop1 reported during training is measured on
                    # training cells, which ARE in the bank, so it is ~1.0 by
                    # construction and says nothing about generalization. These
                    # are the held-out equivalents: how similar the best bank
                    # entry is to a query the bank has never seen, and how much
                    # of the retrieved vector survives into the prediction.
                    retr_top1_vals.append(float(rm.get("top1_mean", float("nan"))))
                    retr_use_vals.append(float(retr_used_e.float().mean().item()))
                    x_retr_all.append(x_retr.detach().float().cpu())

            mu, theta, pi, aux = model(
                batch_in, spot_feat_in, edge_index, edge_weight,
                return_attn=dump_attn,
                zs_precomputed=zs_precomputed,
            )
            mu_clamped = torch.clamp(mu, min=0.0)
            yhat_log   = torch.log1p(mu_clamped + eps)

        preds.append(yhat_log.detach().cpu())
        gts.append(x_log.detach().cpu())
        node_idxs.append(batch["idx"].detach().cpu().long())
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

    # The eval loader inherits shuffle=True from make_node_idx_dataloader, so P/Y
    # come back in loader order, not node order. The metrics below are column-wise
    # and so order-invariant, but the caller pairs P with adata.obs when writing
    # recovered_data.h5ad — without this scatter those rows are misaligned (and
    # differently misaligned each epoch). Scatter by node index rather than
    # forcing shuffle=False so this stays correct however the loader is built.
    node_idx = torch.cat(node_idxs, 0).numpy()
    if node_idx.shape[0] == P.shape[0]:
        order = np.argsort(node_idx, kind="stable")
        P = P[order]
        Y = Y[order]

    Yc = Y - Y.mean(0, keepdims=True)
    Pc = P - P.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0)) + 1e-8
    pcc = np.where(den > 0, num / den, np.nan)

    # R1-3 (E3): correlation between the retrieved context vector and the final
    # prediction, on held-out cells. Near 1.0 would mean the network is passing
    # the retrieved profile through rather than conditioning on the image.
    retr_pred_corr = float("nan")
    if x_retr_all:
        XR = torch.cat(x_retr_all, 0).numpy()
        if node_idx.shape[0] == XR.shape[0]:
            XR = XR[order]
        if XR.shape == P.shape:
            a_ = XR - XR.mean(0, keepdims=True)
            b_ = P - P.mean(0, keepdims=True)
            dd = np.sqrt((a_ ** 2).sum(0) * (b_ ** 2).sum(0)) + 1e-8
            retr_pred_corr = float(np.nanmean(np.where(dd > 0, (a_ * b_).sum(0) / dd, np.nan)))

    return {
        "mse":        mse_sum / max(n, 1),
        "mae":        mae_sum / max(n, 1),
        "pcc_mean":   float(np.nanmean(pcc)),
        "retr_top1_heldout": (float(np.nanmean(retr_top1_vals)) if retr_top1_vals
                              else float("nan")),
        "retr_used_heldout": (float(np.nanmean(retr_use_vals)) if retr_use_vals
                              else float("nan")),
        "retr_pred_corr":    retr_pred_corr,
        "pcc_median": float(np.nanmedian(pcc)),
        "ct_acc":     float(cls_correct / max(cls_count, 1)),
        "ct_n":       int(cls_count),
    }, P, Y


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Final retrieval finetuning with entropy-adaptive per-query temperature."
    )

    p.add_argument("--pt_img_backbone", default=None,
                   help="Image_pretrained.pt — supplies both the backbone and "
                        "the projection used to build H(0).")
    p.add_argument("--pt_ig",           default=None, help="SNN_pretrained.pt")
    p.add_argument("--pt_is",           default=None,
                   help="Spatial_pretrained.pt (was Coord_pretrained.pt before "
                        "the GCN switched from coordinates to histology H(0); "
                        "old files are not loadable and will error out).")
    p.add_argument("--pt_xattn",        default=None,
                   help="LEGACY. Pretraining no longer trains cross-attention "
                        "(Eqs. 4-6 act on the encoder outputs), so no "
                        "XATTN_pretrained.pt is produced and the Eq. (8) fusion "
                        "is trained from scratch here. Leave unset.")

    p.add_argument("--h5ad",     required=True)
    p.add_argument("--csv",      required=True)
    p.add_argument("--root",     default=None)
    p.add_argument("--id_col",   default="cell_id")
    p.add_argument("--path_col", default="patch_path")
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0",
                            "plip","phikon","conch_v1"])
    p.add_argument("--epochs",       type=int,   default=250)  # Supp.: 250 for finetuning
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--proj_dim_gene", type=int,  default=256)
    p.add_argument("--proj_dim_spot", type=int,  default=256)
    p.add_argument("--fuse_dim",     type=int,   default=512)
    p.add_argument("--heads",        type=int,   default=8)
    p.add_argument("--img_tokens",   type=int,   default=10)   # Supp.: Image/Cell/Gene tokens = 10
    p.add_argument("--spot_tokens",  type=int,   default=10)
    p.add_argument("--gene_tokens",  type=int,   default=10)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--modalities", default="all",
                   choices=["all", "image", "image+coords", "image+gene"],
                   help="modality ablation (R2): all=image+coords+gene; image=image only; "
                        "image+coords=drop gene; image+gene=drop coords")
    p.add_argument("--merge", default="gated-sum", choices=["gated-sum","concat-proj"])
    p.add_argument("--k",            type=int,   default=12,
                   help="k for the KNN graph (also the Delaunay fallback)")
    p.add_argument("--graph_type",   default="delaunay", choices=["knn","delaunay"],
                   help="Spatial graph over the spot centroids. The manuscript "
                        "uses Delaunay triangulation; 'knn' is kept for ablation.")
    p.add_argument("--pt_proj_dim",  type=int,   default=256,
                   help="Pretraining --proj_dim. Sets the width of H(0) and the "
                        "spatial GCN input; must match Spatial_pretrained.pt.")
    p.add_argument("--spot_hidden_dim", type=int, default=256,
                   help="Hidden width of the spatial GCN; must match the "
                        "pretraining --spot_hidden_dim.")
    p.add_argument("--spot_feat_refresh_epochs", type=int, default=0,
                   help="Recompute the histology node features H(0) every N "
                        "epochs (0 = compute once). Use N>0 when the image "
                        "backbone is trainable (LoRA / no --freeze_backbone).")
    p.add_argument("--lr",           type=float, default=1e-5)  # Supp.: base lr 1e-5
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--device",       default="auto")
    p.add_argument("--amp",          action="store_true",
                   help="Enable AMP (legacy flag).")
    p.add_argument("--amp_dtype",    default=None,
                   choices=["auto", "bf16", "fp16", "fp32"],
                   help="AMP dtype. If set, overrides --amp behavior.")
    p.add_argument("--save_dir",     default="./ckpts_cross")
    p.add_argument("--save_every",   type=int, default=0,
                   help="Also save checkpoint every N epochs (0 = disabled).")
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
    p.add_argument("--pcc_warmup_epochs",  type=int,   default=0,
                   help="Ramp lambda_pcc from 0 over N epochs. The PCC term is "
                        "ill-conditioned while the decoder output is still "
                        "near-constant, so applying it from step 1 stalls training.")
    p.add_argument("--ce_head_hidden",     type=int,   default=0)
    p.add_argument("--ce_head_dropout",    type=float, default=0.0)
    p.add_argument("--ce_class_weight",    default="none", choices=["none","balanced"])
    p.add_argument("--warmup_epochs",      type=int,   default=2)
    p.add_argument("--val_frac",     type=float, default=0.2,
                   help="Fraction of cells held out as validation tiles (R1-4).")
    p.add_argument("--val_tile_um",  type=float, default=1000.0,
                   help="Checkerboard tile edge in microns.")
    p.add_argument("--val_buffer_um", type=float, default=75.0,
                   help="Cells this close to a tile edge join neither split.")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--ema_decay",          type=float, default=0.0)

    # ---- Final retrieval args (clean naming) ----
    p.add_argument("--retrieval",           action="store_true",
                   help="Enable retrieval-augmented training.")
    p.add_argument("--bank_size",           type=int,   default=5000,
                   help="Per-query quota bank size (max retrieved samples).")
    p.add_argument("--tau_base",            type=float, default=0.07,
                   help="Base retrieval temperature (τ_base).")
    p.add_argument("--tau_entropy_alpha",   type=float, default=0.3,  # Supp.: alpha = 0.3
                   help="α for τ_q = τ_base × (1 + α × H_norm). 0 = fixed τ.")
    p.add_argument("--tau_learnable",       action="store_true",
                   help="Make τ_base a learnable nn.Parameter.")
    p.add_argument("--mix",                 type=float, default=0.5,
                   help="Train blend ratio: (1-mix)×GT + mix×retrieved.")
    p.add_argument("--eval_mix",            type=float, default=1.0,
                   help="Eval blend ratio.")
    p.add_argument("--mix_warmup_epochs",   type=int,   default=5,
                   help="Ramp mix from 0 → mix over N epochs.")
    p.add_argument("--lambda_retr",         type=float, default=0.0,
                   help="Aux MSE loss weight: retrieved vs GT.")
    p.add_argument("--refresh_epochs",      type=int,   default=1,
                   help="Rebuild retrieval banks every N epochs.")
    p.add_argument("--pred_min_conf",       type=float, default=0.0,
                   help="Min softmax confidence for inclusion in class_index_map.")
    p.add_argument("--prob_topm",           type=int,   default=0,
                   help="Keep only top-m classes per query (0 = all).")
    p.add_argument("--fallback",            default="global",
                   choices=["global", "zero"],
                   help="Fallback when no class candidates found.")
    p.add_argument("--retrieval_num_celltypes", type=int, default=0,
                   help="cls_head size when --lambda_ce=0 (retrieval needs cell-type head).")

    # Performance / debug
    p.add_argument("--train_gcn", action="store_true",
                   help="Keep GCN trainable (per-batch full-graph GCN, slower).")
    p.add_argument("--nan_check_every", type=int, default=50,
                   help="Check intermediate tensors for NaN every N steps.")

    p.add_argument("--freeze_backbone", action="store_true",
                   help="freeze the pretrained image encoder; train only the "
                        "fusion/decoder/head modules")
    p.add_argument("--with_lora", action="store_true",
                   help="PEFT: freeze the backbone and train LoRA adapters on it")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--early_stopping", action="store_true",
                   help="stop when eval PCC(mean) has not improved for --patience epochs")
    p.add_argument("--patience", type=int, default=5,
                   help="epochs without improvement before stopping")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="minimum eval PCC(mean) gain that counts as an improvement")

    p.add_argument("--wandb", action="store_true", help="log this run to wandb")
    p.add_argument("--wandb_project", default="spam-xenium")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default="online", choices=["online","offline","disabled"])
    p.add_argument("--wandb_tags", default=None, help="comma-separated")
    p.add_argument("--wandb_notes", default=None)

    args = p.parse_args()

    wandb_init(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
        tags=[t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None,
        notes=args.wandb_notes,
        config=vars(args),
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # device & amp
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")
    amp_requested = bool(args.amp)
    if args.amp_dtype is not None:
        amp_requested = (args.amp_dtype != "fp32")
    amp_choice = args.amp_dtype if args.amp_dtype is not None else "auto"
    amp_dtype = pick_amp_dtype(amp_choice, device) if amp_requested else torch.float32
    use_amp = bool(device.type == "cuda" and amp_dtype != torch.float32)

    print(f"[Device] {device}  AMP={use_amp}  dtype={amp_dtype}")
    print(f"[FusionTokens] img={args.img_tokens}, spot={args.spot_tokens}, "
          f"gene={args.gene_tokens}")
    print(f"[GCN] train_gcn={args.train_gcn}  "
          f"({'per-batch full-graph' if args.train_gcn else 'pre-computed once/epoch (fast)'})")
    print(f"[NaNCheck] every {args.nan_check_every} steps")
    print(f"[Checkpoint] save_every={args.save_every}")
    if args.retrieval:
        print(f"[Retrieval] ENABLED  bank_size={args.bank_size}  "
              f"tau_base={args.tau_base}  tau_entropy_alpha={args.tau_entropy_alpha}  "
              f"tau_learnable={args.tau_learnable}  mix={args.mix}  "
              f"eval_mix={args.eval_mix}  refresh_epochs={args.refresh_epochs}")

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

    foundation_tmp = inf_encoder_factory(args.enc_name)(weights_path=None)
    image_tf = getattr(foundation_tmp, "eval_transforms", None)
    del foundation_tmp

    base_ds = SpotImageGeneDataset(
        adata=adata, map_df=map_df_aligned, gene_names=None, image_transform=image_tf,
    )
    gene_dim = base_ds.gene_dim

    # Count-scale targets for the ZINB likelihood (manuscript Eqs. 9-10 define
    # it on observed counts). Held on CPU and sliced per batch: for the largest
    # section this matrix is several GB, and only the batch rows are needed.
    global COUNTS_ALL, SIZE_FACTOR_ALL
    COUNTS_ALL = torch.from_numpy(base_ds.counts_matrix())
    SIZE_FACTOR_ALL = torch.from_numpy(base_ds.size_factors())
    _int_frac = float((COUNTS_ALL == COUNTS_ALL.round()).float().mean())
    print(f"[ZINB] counts {tuple(COUNTS_ALL.shape)} integer_frac={_int_frac:.4f} | "
          f"size_factor mean={SIZE_FACTOR_ALL.mean():.4f} "
          f"[{SIZE_FACTOR_ALL.min():.4f}, {SIZE_FACTOR_ALL.max():.4f}]")
    if _int_frac < 0.999:
        raise RuntimeError(
            "adata.layers['counts'] is not integer-valued — the h5ad was already "
            "normalized before load_h5ad_with_preproc() saw it. The ZINB "
            "likelihood needs observed counts.")
    # With the backbone frozen its outputs are cached once (below), so the
    # training loop never needs the patches again.
    _skip_img = bool(args.freeze_backbone)
    # ---- R1-4: spatially disjoint train / validation tiles ----------------
    # The submitted code trained and "evaluated" on one loader over the whole
    # section, so the best checkpoint was chosen by in-sample PCC. Now the
    # section is split into checkerboard tiles with a buffer; the graph is still
    # built over all N cells (transductive) and only the loss is masked.
    tr_nodes, va_nodes, buf_nodes = checkerboard_split(
        coords_np, val_frac=args.val_frac, tile_um=args.val_tile_um,
        buffer_um=args.val_buffer_um, seed=args.seed,
    )
    print(val_split_summary(coords_np, tr_nodes, va_nodes, buf_nodes,
                            args.val_tile_um, args.val_buffer_um))
    if len(va_nodes) == 0:
        raise RuntimeError("validation split is empty; lower --val_tile_um")

    loader = make_node_idx_dataloader(
        base_ds=base_ds, id_to_idx=id_to_idx,
        batch_size=args.batch_size, num_workers=args.num_workers,
        skip_images=_skip_img, subset=tr_nodes,
    )
    val_loader = make_node_idx_dataloader(
        base_ds=base_ds, id_to_idx=id_to_idx,
        batch_size=args.batch_size, num_workers=args.num_workers,
        skip_images=_skip_img, shuffle=False, subset=va_nodes,
    )
    # Every node needs a key and a class posterior as a *query*, including
    # validation cells; what they may *retrieve* is restricted separately.
    bank_loader = make_node_idx_dataloader(
        base_ds=base_ds, id_to_idx=id_to_idx,
        batch_size=args.batch_size, num_workers=args.num_workers,
        skip_images=_skip_img, shuffle=False,
    )
    # Ordered pass used to (re)build the histology node features H(0). Always
    # reads images: it is what fills the cache.
    spot_feat_loader = make_node_idx_dataloader(
        base_ds=base_ds, id_to_idx=id_to_idx,
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"[Data] N={len(base_ds)}, gene_dim={gene_dim}, "
          f"train steps/epoch={len(loader)}, val steps={len(val_loader)}")

    # graph
    if args.graph_type == "delaunay":
        edge_index, edge_weight = build_delaunay_graph(coords_np, k=args.k, use_weight=True)
    else:
        edge_index, edge_weight = build_knn_graph(coords_np, k=args.k, use_weight=True)
    print(f"[Graph] type={args.graph_type}, k={args.k}, "
          f"N={coords_np.shape[0]}, E={edge_index.size(1)}")
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None
    n_nodes = int(coords_np.shape[0])
    # Coordinates define the graph edges only. The GCN node features H(0) are
    # histology embeddings, built after the encoders are loaded (see below).

    # ---- encoders ----
    foundation  = inf_encoder_factory(args.enc_name)(weights_path=None)
    foundation.precision = amp_dtype

    # The optimizer below takes model.parameters() at args.lr. Left alone that
    # trains the whole ~310M-param pretrained backbone at the head's learning
    # rate (3e-4), which is ~30x the usual range for finetuning a pretrained ViT
    # and destroys the representation within one epoch. Either freeze it or
    # adapt it with LoRA.
    if args.freeze_backbone and args.with_lora:
        raise ValueError("--freeze_backbone and --with_lora are mutually exclusive")

    if args.with_lora:
        foundation = attach_lora_to_foundation(
            foundation, backbone_name=args.enc_name,
            r=args.lora_r, alpha=args.lora_alpha,
            dropout=args.lora_dropout, bias="none",
        )
        _tot = sum(p.numel() for p in foundation.encoder.parameters())
        _tr = sum(p.numel() for p in foundation.encoder.parameters() if p.requires_grad)
        print(f"[LoRA] r={args.lora_r} alpha={args.lora_alpha} on '{args.enc_name}': "
              f"{_tot/1e6:.1f}M total, {_tr/1e6:.2f}M trainable ({100*_tr/_tot:.2f}%)")
    elif args.freeze_backbone:
        # Pins eval mode too: train_epoch's model.train() would otherwise recurse
        # into the frozen backbone and re-enable its dropout each epoch.
        n_frozen = freeze_and_lock_eval(foundation.encoder)
        print(f"[Freeze] image backbone frozen: {n_frozen/1e6:.1f}M params require_grad=False")

    img_dim = foundation.get_output_dim()

    gene_encoder = GeneEncoder(
        input_dim=gene_dim, model_size_omic="small",
        proj_dim=args.proj_dim_gene, normalize=False,
    )
    # in_dim == the pretraining --proj_dim: the GCN consumes the projected
    # histology embedding, matching Spatial_pretrained.pt.
    spot_gcn = SpatialGCN(
        in_dim=args.pt_proj_dim, hidden_dim=args.spot_hidden_dim,
        out_dim=args.proj_dim_spot, dropout=0.2,
    )

    num_celltypes_model = 0
    if args.lambda_ce > 0.0:
        num_celltypes_model = len(celltype_names)
    elif args.retrieval:
        num_celltypes_model = int(max(args.retrieval_num_celltypes, 0))
    if args.retrieval and num_celltypes_model <= 0:
        raise ValueError(
            "--retrieval requires cls_head. Set --lambda_ce>0 or "
            "--retrieval_num_celltypes > 0."
        )

    model = CrossAttnZINBModel(
        img_encoder=foundation, gene_encoder=gene_encoder, gcn=spot_gcn,
        dim_img=img_dim, dim_gene=args.proj_dim_gene, dim_spot=args.proj_dim_spot,
        dim_fuse=args.fuse_dim, num_heads=args.heads, dropout=args.dropout,
        merge=args.merge, img_tokens=args.img_tokens,
        spot_tokens=args.spot_tokens, gene_tokens=args.gene_tokens,
        decoder_hidden=1024, gene_dim=gene_dim,
        num_celltypes=num_celltypes_model,
        cls_head_hidden=args.ce_head_hidden, cls_dropout=args.ce_head_dropout,
        dim_spot_in=args.pt_proj_dim,
        use_spot=args.modalities in ("all", "image+coords"),
        use_gene=args.modalities in ("all", "image+gene"),
    )
    if args.modalities != "all":
        print(f"[modality-ablation] modalities={args.modalities} "
              f"(use_spot={model.use_spot}, use_gene={model.use_gene})", flush=True)

    # Load pretrained weights before .to(device): spot_in_proj lives inside the
    # model, and it must come from the same Image_pretrained.pt as the backbone.
    load_pretrained_for_finetune(
        foundation, gene_encoder, spot_gcn, model.spot_in_proj,
        args.pt_img_backbone, args.pt_ig, args.pt_is,
    )
    model = model.to(device)

    if args.retrieval:
        init_tau = float(max(args.tau_base, 1e-3))
        if args.tau_learnable:
            model.retrieval_log_tau = nn.Parameter(
                torch.tensor(np.log(init_tau), dtype=torch.float32))
            print(f"[Retrieval] learnable tau_base (init={init_tau:.5f})")
        else:
            model.register_buffer("retrieval_tau_const",
                                  torch.tensor(init_tau, dtype=torch.float32))
            print(f"[Retrieval] fixed tau_base={init_tau:.5f}")

    if args.pt_xattn:
        load_xattn_into_fusion(model.fusion, args.pt_xattn)

    # ---- H(0): histology node features for the spatial GCN ----
    # spot_feat_in is consumed as a constant (the GCN is run over all nodes at
    # once, outside the per-batch autograd path), so spot_in_proj receives no
    # gradient. Freeze it explicitly rather than parking dead state in AdamW.
    for p in model.spot_in_proj.parameters():
        p.requires_grad = False
    if args.freeze_backbone:
        model.img_feat_cache = build_histology_node_features(
            img_encoder=model.img_encoder, loader=spot_feat_loader,
            n_nodes=n_nodes, out_dim=img_dim, device=device, amp_dtype=amp_dtype,
        )
        print(f"[Cache] frozen backbone -> cached {tuple(model.img_feat_cache.shape)} "
              f"backbone outputs ({model.img_feat_cache.numel()*4/1e9:.2f} GB); "
              "training loop no longer reads patches")
    spot_feat_in = model.build_spot_features(
        img_loader=spot_feat_loader, n_nodes=n_nodes,
        device=device, amp_dtype=amp_dtype,
    )
    if args.spot_feat_refresh_epochs > 0:
        print(f"[SpotFeat] H(0) refreshed every {args.spot_feat_refresh_epochs} epoch(s)")
    elif not args.freeze_backbone:
        print("[SpotFeat][warn] the image backbone is trainable but H(0) is "
              "computed once. Pass --spot_feat_refresh_epochs N to keep the "
              "node features in sync with the drifting encoder.")

    # Only hand the optimizer params that actually require grad, so a frozen
    # backbone is not carried in the optimizer state (AdamW would otherwise
    # allocate exp_avg/exp_avg_sq for all ~310M frozen params).
    _trainable = [p for p in model.parameters() if p.requires_grad]
    _n_all = sum(p.numel() for p in model.parameters())
    _n_tr = sum(p.numel() for p in _trainable)
    print(f"[Optim] trainable {_n_tr/1e6:.2f}M / {_n_all/1e6:.1f}M params "
          f"({100*_n_tr/_n_all:.2f}%) at lr={args.lr}")
    optim = torch.optim.AdamW(_trainable, lr=args.lr,
                               weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and amp_dtype != torch.float32),
    )

    celltype_y_t     = None
    celltype_y_eval_t = None
    ce_class_weights = None
    if args.lambda_ce > 0.0 and celltype_y_np is not None:
        # R1-4: the transcriptomic-state labels come from Leiden clustering of
        # the expression matrix, so keeping validation cells labelled would let
        # their expression shape the classifier the retrieval depends on. Mark
        # them -1 (ignore_index) instead of dropping the array, so node indices
        # stay aligned everywhere else.
        # Two label vectors, deliberately. The classifier must not see validation
        # cells (they define the retrieval bank's classes), but we still need
        # their true labels to *measure* held-out accuracy — masking both was
        # what produced "evalCtAcc=0.0000 (n=0)" and blinded the classifier
        # diagnostics that R2-3 depends on.
        celltype_y_eval_np = celltype_y_np.copy()          # measurement only
        celltype_y_np = celltype_y_np.copy()               # training targets
        n_lab_before = int((celltype_y_np >= 0).sum())
        celltype_y_np[va_nodes] = -1
        celltype_y_np[buf_nodes] = -1
        print(f"[CellType] train labels masked outside training tiles: "
              f"{n_lab_before} -> {int((celltype_y_np >= 0).sum())}; "
              f"eval labels kept: {int((celltype_y_eval_np >= 0).sum())}")
        celltype_y_t = torch.from_numpy(celltype_y_np).to(device=device, dtype=torch.long)
        celltype_y_eval_t = torch.from_numpy(celltype_y_eval_np).to(device=device, dtype=torch.long)
        if args.ce_class_weight == "balanced":
            valid   = celltype_y_np[celltype_y_np >= 0]
            counts  = np.bincount(valid, minlength=len(celltype_names)).astype(np.float32)
            w       = counts.sum() / np.maximum(counts, 1.0)
            ce_class_weights = torch.tensor(w, dtype=torch.float32, device=device)
            print(f"[CellType] class_weight=balanced (min={w.min():.4f}, max={w.max():.4f})")

    retrieval_cfg = None
    if args.retrieval:
        expr_bank_np = _adata_x_to_numpy_float32(adata)
        retrieval_cfg = {
            "enabled":            True,
            "bank_size":          int(max(1, args.bank_size)),
            "tau_base":           float(max(args.tau_base, 1e-3)),
            "tau_entropy_alpha":  float(args.tau_entropy_alpha),
            "mix":                float(np.clip(args.mix,      0.0, 1.0)),
            "eval_mix":           float(np.clip(args.eval_mix, 0.0, 1.0)),
            "mix_warmup_epochs":  int(max(0, args.mix_warmup_epochs)),
            "lambda_retr":        float(max(0.0, args.lambda_retr)),
            "fallback":           str(args.fallback),
            "pred_min_conf":      float(max(0.0, args.pred_min_conf)),
            "prob_topm":          int(max(0, args.prob_topm)),
            "expr_bank_np":       expr_bank_np,
            "class_index_map":    {},
            "keys_bank":          None,
            "pred_prob_bank":     None,
            # Retrievable cells = training tiles only (R1-4).
            "candidate_nodes":    tr_nodes,
            "cls_of":             None,
            "expr_bank_t":        None,
            "allowed_mask":       torch.zeros(map_df_aligned.shape[0],
                                              dtype=torch.bool).index_fill_(
                                                  0, torch.from_numpy(tr_nodes.astype(np.int64)), True),
        }

    from torch.optim.lr_scheduler import LambdaLR

    def warmup_lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1.0) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optim, lr_lambda=warmup_lr_lambda)
    best = {"pcc": -1.0, "path": None}
    # Early stopping tracks eval PCC(mean) — the same quantity that selects the
    # best checkpoint — so stopping never discards a better model than the one
    # already saved under save_dir/best.
    epochs_no_improve = 0
    best_epoch = 0

    for ep in range(1, args.epochs + 1):

        # Refresh H(0) when the image encoder is still moving
        if (args.spot_feat_refresh_epochs > 0 and ep > 1
                and (ep - 1) % args.spot_feat_refresh_epochs == 0):
            spot_feat_in = model.build_spot_features(
                img_loader=spot_feat_loader, n_nodes=n_nodes,
                device=device, amp_dtype=amp_dtype,
            )

        # Pre-compute GCN output for all nodes once per epoch
        if args.train_gcn:
            zs_all = None
        else:
            zs_all = _precompute_gcn_all(model, spot_feat_in, edge_index, edge_weight)

        # Rebuild retrieval banks
        if retrieval_cfg is not None and retrieval_cfg.get("enabled"):
            refresh_every = max(1, int(args.refresh_epochs))
            if ep == 1 or ((ep - 1) % refresh_every == 0):
                banks = _build_retrieval_final_banks(
                    model=model, loader=bank_loader,
                    spot_feat_in=spot_feat_in, edge_index=edge_index,
                    edge_weight=edge_weight,
                    expr_bank_np=retrieval_cfg["expr_bank_np"],
                    num_celltypes=num_celltypes_model,
                    device=device, amp_dtype=amp_dtype,
                    n_nodes=map_df_aligned.shape[0],
                    pred_min_conf=float(retrieval_cfg.get("pred_min_conf", 0.0)),
                    zs_precomputed=zs_all,
                    candidate_nodes=retrieval_cfg.get("candidate_nodes"),
                )
                retrieval_cfg["keys_bank"]       = banks["keys_bank"]
                retrieval_cfg["pred_prob_bank"]  = banks["pred_prob_bank"]
                retrieval_cfg["class_index_map"] = banks["class_index_map"]
                # Device-resident lookups for the vectorized retrieval: class
                # membership lists and the expression bank, moved once per
                # refresh instead of once per query.
                retrieval_cfg["cls_of"] = {
                    int(c): torch.from_numpy(np.asarray(v, dtype=np.int64)).to(device)
                    for c, v in banks["class_index_map"].items()
                }
                if retrieval_cfg.get("expr_bank_t") is None:
                    retrieval_cfg["expr_bank_t"] = torch.from_numpy(
                        np.ascontiguousarray(retrieval_cfg["expr_bank_np"])
                    ).to(device)
                print(f"[Retrieval] banks refreshed at epoch {ep}, "
                      f"clusters={len(retrieval_cfg['class_index_map'])}")

        t0 = time.time()
        tr = train_epoch(
            model, loader, spot_feat_in, edge_index, edge_weight,
            optim, scaler,
            amp_dtype, device,
            epoch=ep, total_epochs=args.epochs,
            lambda_zinb=args.lambda_zinb, lambda_mse=args.lambda_mse,
            lambda_pcc=args.lambda_pcc,   lambda_ce=args.lambda_ce,
            ce_label_smoothing=args.ce_label_smoothing,
            ce_warmup_epochs=args.ce_warmup_epochs,
            pcc_warmup_epochs=args.pcc_warmup_epochs,
            celltype_labels_all=celltype_y_t,
            ce_class_weights=ce_class_weights,
            max_grad_norm=args.max_grad_norm,
            retrieval_cfg=retrieval_cfg,
            zs_precomputed=zs_all,
            nan_check_every=args.nan_check_every,
        )
        scheduler.step()

        ev, P, Y = eval_epoch(
            model, val_loader, spot_feat_in, edge_index, edge_weight,
            amp_dtype, device,
            epoch=ep, total_epochs=args.epochs,
            save_attn=bool(args.save_attn and (ep % max(args.attn_every, 1) == 0)),
            attn_out_dir=attn_out_dir,
            attn_max_batches=args.attn_max_batches,
            attn_reduce_heads=args.attn_reduce_heads,
            celltype_labels_all=(celltype_y_eval_t if celltype_y_eval_t is not None
                                 else celltype_y_t),
            retrieval_cfg=retrieval_cfg,
            zs_precomputed=zs_all,
        )
        dt = time.time() - t0

        retr_msg = ""
        if retrieval_cfg is not None and retrieval_cfg.get("enabled"):
            # train-side numbers are in-bank by construction; the held-out
            # ones are what answer "retrieval or memorization" (R1-3).
            retr_msg = (f" | trainTop1={tr['retr_top1']:.4f} "
                        f"heldoutTop1={ev.get('retr_top1_heldout', float('nan')):.4f} "
                        f"corr(xref,yhat)={ev.get('retr_pred_corr', float('nan')):.4f} "
                        f"τ={tr['retr_tau']:.4f}")
        print(
            f"[{ep:03d}] trainLoss={tr['loss']:.6f} "
            f"trainCtAcc={tr['ct_acc']:.4f} (n={tr['ct_n']}) | "
            f"eval MSE={ev['mse']:.6e} MAE={ev['mae']:.6e} "
            f"PCC(mean)={ev['pcc_mean']:.4f} med={ev['pcc_median']:.4f} "
            f"evalCtAcc={ev['ct_acc']:.4f} (n={ev['ct_n']})"
            f"{retr_msg}  ({dt:.1f}s)"
        )

        _wb = {
            "epoch/index": ep,
            "train/loss": tr["loss"],
            "train/ct_acc": tr["ct_acc"],
            "eval/mse": ev["mse"],
            "eval/mae": ev["mae"],
            "eval/pcc_mean": ev["pcc_mean"],
            "eval/pcc_median": ev["pcc_median"],
            "eval/ct_acc": ev["ct_acc"],
            "epoch/time_sec": dt,
        }
        if retrieval_cfg is not None and retrieval_cfg.get("enabled"):
            _wb.update({
                "retr/used": tr["retr_used"],
                "retr/top1": tr["retr_top1"],
                "retr/tau_mean": tr["retr_tau"],
            })
        wandb_log(_wb, step=ep)

        improved = ev["pcc_mean"] > best["pcc"] + args.min_delta
        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        wandb_log({"epoch/epochs_no_improve": epochs_no_improve}, step=ep)

        if ev["pcc_mean"] > best["pcc"]:
            # R1-4: selection is now driven by the held-out validation tiles.
            # recovered_data.h5ad is NOT written here — P covers only the
            # validation subset, and the inference script consumes that file as
            # a whole-section bank. It is regenerated once after training from
            # the selected checkpoint (see below).
            best["pcc"] = ev["pcc_mean"]
            out_dir = os.path.join(args.save_dir, "best")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            if celltype_names:
                torch.save({"celltype_col":   celltype_col_used,
                            "celltype_names": celltype_names,
                            "num_celltypes":  len(celltype_names)},
                           os.path.join(out_dir, "celltype_meta.pt"))
            # R2-6: record exactly what produced this checkpoint.
            with open(os.path.join(out_dir, "config.json"), "w") as _f:
                json.dump({"args": vars(args), "epoch": ep,
                           "val_pcc_mean": float(ev["pcc_mean"]),
                           "val_pcc_median": float(ev["pcc_median"]),
                           "n_train_cells": int(len(tr_nodes)),
                           "n_val_cells": int(len(va_nodes)),
                           "n_buffer_cells": int(len(buf_nodes)),
                           "selection_metric": "val_pcc_mean"}, _f, indent=2)
            best["path"] = out_dir
            best_epoch = ep
            print(f"[BEST↑] val PCC(mean)={ev['pcc_mean']:.4f} @ epoch {ep} -> {out_dir}")

        if args.early_stopping and epochs_no_improve >= args.patience:
            print(f"[EarlyStop] eval PCC(mean) has not improved by >{args.min_delta} "
                  f"for {epochs_no_improve} epochs (patience={args.patience}). "
                  f"Stopping at epoch {ep}/{args.epochs}; "
                  f"best={best['pcc']:.4f} @ epoch {best_epoch}.")
            wandb_summary(early_stopped=True, stopped_at_epoch=ep,
                          best_epoch=best_epoch)
            break

        if args.save_every > 0 and ep % args.save_every == 0:
            out_dir = os.path.join(args.save_dir, f"ep{ep:04d}")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            if celltype_names:
                torch.save({"celltype_col":   celltype_col_used,
                            "celltype_names": celltype_names,
                            "num_celltypes":  len(celltype_names)},
                           os.path.join(out_dir, "celltype_meta.pt"))
            print(f"[CKPT] epoch {ep} saved to {out_dir}")

    print(f"Done. best val PCC(mean)={best['pcc']:.4f} @ epoch {best_epoch}, "
          f"path={best['path']}")

    # ---- whole-section recovery from the selected checkpoint ---------------
    # Written once, after selection, over every node — the inference script uses
    # this file as the training-section expression bank, so it must cover the
    # full section even though model selection only ever saw validation tiles.
    if best["path"] is not None:
        sd = torch.load(os.path.join(best["path"], "model.pt"), map_location="cpu",
                        weights_only=False)
        model.load_state_dict(sd)
        model.to(device)
        zs_full = (None if args.train_gcn
                   else _precompute_gcn_all(model, spot_feat_in, edge_index, edge_weight))
        ev_full, P_full, _ = eval_epoch(
            model, bank_loader, spot_feat_in, edge_index, edge_weight,
            amp_dtype, device, epoch=best_epoch, total_epochs=args.epochs,
            save_attn=False, attn_out_dir=None, attn_max_batches=0,
            attn_reduce_heads=False,
            celltype_labels_all=(celltype_y_eval_t if celltype_y_eval_t is not None
                                 else celltype_y_t),
            retrieval_cfg=retrieval_cfg, zs_precomputed=zs_full,
        )
        rec = ad.AnnData(X=P_full, obs=adata.obs.copy(),
                         var=adata.var.copy(), uns=adata.uns.copy())
        rec.obs["spam_split"] = "buffer"
        rec.obs.iloc[tr_nodes, rec.obs.columns.get_loc("spam_split")] = "train"
        rec.obs.iloc[va_nodes, rec.obs.columns.get_loc("spam_split")] = "val"
        rec.write(os.path.join(best["path"], "recovered_data.h5ad"), compression="gzip")
        print(f"[Recover] whole-section predictions written "
              f"({P_full.shape[0]} cells); in-sample PCC(mean)={ev_full['pcc_mean']:.4f} "
              f"(reported for reference only — selection used the validation tiles)")

    wandb_summary(best_pcc_mean=best["pcc"], best_path=best["path"],
                  best_epoch=best_epoch, epochs_run=args.epochs)
    wandb_finish()


if __name__ == "__main__":
    main()
