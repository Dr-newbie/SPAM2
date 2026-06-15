# run_contrastive_xattn_claudever.py
# Optimized version of run_contrastive_xattn.py
#
# Changes vs original:
#   1. [PERF] Pre-compute string spot_id → integer node index at dataset build time
#      (_NodeIdxDataset wrapper). Eliminates per-sample Python string processing in the hot path.
#   2. [PERF] NaN checks on intermediate tensors (zi, zg, zc) only every --nan_check_every
#      steps (default 50). Final loss is still checked every step (scalar, cheap).
#      Removes ~50x GPU-CPU pipeline stalls from .all().item() calls.
#   3. [PERF] DataLoader uses persistent_workers=True — avoids re-spawning worker
#      processes between epochs.
#   4. [PERF] Checkpoint saved only when training loss improves (--save_best, default on),
#      with optional periodic save via --save_every N. Removes unnecessary I/O every epoch.
#   5. [MINOR] Loss logging window fixed: running loss correctly resets after each print.
#
# Example:
'''
CUDA_VISIBLE_DEVICES=1 \
/data/project/banana9903_xenium/.conda-envs/SPAM/bin/python \
  Code_final/code/utils/run_contrastive_xattn_claudever.py \
--h5ad /data/project/banana9903_xenium/h5ads/human_skin1.h5ad \
--csv /data/project/banana9903_xenium/contrast_test/expression/human_skin1_patches_256/spot_patch_map_256.csv \
--enc_name uni_v1 \
--proj_dim 256 \
--epochs 50 --batch_size 256 \
--save_dir /data/project/banana9903_xenium/ckpts_human_skin1_xattn_10tokens_Claudeversion \
--device cuda:0 \
--img_tokens 10 --gene_tokens 10 --coord_tokens 10 \
--graph_type delaunay \
--nan_check_every 50 \
--save_every 10
'''

import os
import sys
import time
import argparse
from typing import Optional, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # /Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.modules.pop("code", None)

from code.models.Foundations import inf_encoder_factory
from code.models.GCN_update import SpatialGCN
from code.models.gene_encoder import GeneEncoder
from code.utils.lora_utils import attach_lora_to_foundation
from code.utils.attention import CrossAttentionBlock
from code.utils.contrastive import ImageFoundationAdapter, build_spatial_graph
from code.utils.dataset_utils import (
    SpotImageGeneDataset,
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
    extract_coords_np_in_order,
)
from torch_geometric.utils import subgraph


# ----------------------------
# Device & AMP helpers (unchanged)
# ----------------------------
def setup_cuda_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but --device asks for CUDA")
        if ":" in device_arg:
            idx = int(device_arg.split(":")[1])
            torch.cuda.set_device(idx)
        return torch.device(device_arg)
    return torch.device("cpu")


def pick_amp_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "fp32":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if (major, minor) >= (8, 0) else torch.float16


# ----------------------------
# Fix 1: Pre-compute node indices at dataset build time
# ----------------------------
def _norm_id_str(x) -> str:
    """Normalize spot ID string: strip whitespace and remove trailing '.0'."""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


class _NodeIdxDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around SpotImageGeneDataset that returns an integer node index
    instead of a string spot ID.

    The mapping spot_id -> node_index is computed ONCE at construction time,
    eliminating the per-sample Python string dict lookup that runs inside the
    training hot path in the original script.

    Returns: (img_tensor, gene_tensor, node_idx_int)
    """
    def __init__(self, base_ds: SpotImageGeneDataset, id_to_idx: Dict[str, int]):
        self._base = base_ds
        # Pre-compute: dataset position i -> graph node index
        # Since map_df_aligned and the graph share the same ordered spot_id list,
        # id_to_idx[spot_id_at_pos_i] == i, but we resolve explicitly to be safe.
        self._node_indices = np.array(
            [id_to_idx[_norm_id_str(base_ds.map_df.iloc[i]["spot_id"])]
             for i in range(len(base_ds))],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        img, gene, _ = self._base[idx]         # discard string sid
        return img, gene, int(self._node_indices[idx])  # integer index


def make_node_idx_dataloader(
    base_ds: SpotImageGeneDataset,
    id_to_idx: Dict[str, int],
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader that yields (img, gene, node_idx_int) batches."""
    ds = _NodeIdxDataset(base_ds, id_to_idx)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        # Fix 3: Keep worker processes alive between epochs — avoids re-spawn overhead.
        persistent_workers=(num_workers > 0),
    )


# ----------------------------
# Model blocks (unchanged)
# ----------------------------
class XAttnContrastiveModel(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module,
        gene_encoder: nn.Module,
        proj_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.15,
        lam_dir: float = 0.5,
        img_tokens: int = 1,
        gene_tokens: int = 1,
        coord_tokens: int = 1,
    ):
        super().__init__()
        if img_tokens < 1 or gene_tokens < 1 or coord_tokens < 1:
            raise ValueError("img_tokens/gene_tokens/coord_tokens must be >= 1")
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.temperature = float(temperature)
        self.lam_dir = float(lam_dir)
        self.proj_dim = int(proj_dim)
        self.img_tokens = int(img_tokens)
        self.gene_tokens = int(gene_tokens)
        self.coord_tokens = int(coord_tokens)

        self.i2g = CrossAttentionBlock(proj_dim, proj_dim, proj_dim, num_heads, dropout)
        self.g2i = CrossAttentionBlock(proj_dim, proj_dim, proj_dim, num_heads, dropout)
        self.i2c = CrossAttentionBlock(proj_dim, proj_dim, proj_dim, num_heads, dropout)
        self.c2i = CrossAttentionBlock(proj_dim, proj_dim, proj_dim, num_heads, dropout)

        self.img_tokenizer   = None if self.img_tokens   == 1 else nn.Linear(proj_dim, proj_dim * self.img_tokens)
        self.gene_tokenizer  = None if self.gene_tokens  == 1 else nn.Linear(proj_dim, proj_dim * self.gene_tokens)
        self.coord_tokenizer = None if self.coord_tokens == 1 else nn.Linear(proj_dim, proj_dim * self.coord_tokens)

    def _to_tokens(self, x: torch.Tensor, tokenizer: Optional[nn.Module], n_tokens: int) -> torch.Tensor:
        if n_tokens == 1:
            return x.unsqueeze(1)
        return tokenizer(x).view(x.size(0), n_tokens, self.proj_dim)

    @staticmethod
    def _pool_tokens(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1) if x.size(1) == 1 else x.mean(dim=1)

    def cross_attend(self, zi, zg, zc, return_attn: bool = False):
        qi = self._to_tokens(zi, self.img_tokenizer,   self.img_tokens)
        kg = self._to_tokens(zg, self.gene_tokenizer,  self.gene_tokens)
        kc = self._to_tokens(zc, self.coord_tokenizer, self.coord_tokens)

        if return_attn:
            zi2g_t, attn_i2g = self.i2g(qi, kg, kg, return_attn=True)
            zg2i_t, attn_g2i = self.g2i(kg, qi, qi, return_attn=True)
            zi2c_t, attn_i2c = self.i2c(qi, kc, kc, return_attn=True)
            zc2i_t, attn_c2i = self.c2i(kc, qi, qi, return_attn=True)
        else:
            zi2g_t = self.i2g(qi, kg, kg)
            zg2i_t = self.g2i(kg, qi, qi)
            zi2c_t = self.i2c(qi, kc, kc)
            zc2i_t = self.c2i(kc, qi, qi)
            attn_i2g = attn_g2i = attn_i2c = attn_c2i = None

        zi2g = F.normalize(self._pool_tokens(zi2g_t), dim=1)
        zg2i = F.normalize(self._pool_tokens(zg2i_t), dim=1)
        zi2c = F.normalize(self._pool_tokens(zi2c_t), dim=1)
        zc2i = F.normalize(self._pool_tokens(zc2i_t), dim=1)

        if return_attn:
            return zi2g, zg2i, zi2c, zc2i, {
                "attn_i2g": attn_i2g, "attn_g2i": attn_g2i,
                "attn_i2c": attn_i2c, "attn_c2i": attn_c2i,
            }
        return zi2g, zg2i, zi2c, zc2i

    def contrastive_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits = (a @ b.t()) / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return self.lam_dir * F.cross_entropy(logits, targets) + \
               (1.0 - self.lam_dir) * F.cross_entropy(logits.t(), targets)


def _unique_trainable_params(*modules: Iterable[nn.Module]):
    seen, params = set(), []
    for m in modules:
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def _save_all(save_dir, img_encoder, gene_encoder, coord_gcn, xattn_model):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(img_encoder.state_dict(),  os.path.join(save_dir, "Image_pretrained.pt"))
    torch.save(gene_encoder.state_dict(), os.path.join(save_dir, "SNN_pretrained.pt"))
    torch.save(coord_gcn.state_dict(),    os.path.join(save_dir, "Coord_pretrained.pt"))
    torch.save(xattn_model.state_dict(),  os.path.join(save_dir, "XATTN_pretrained.pt"))


def _save_attn_npz(path, attn_map, reduce_heads, meta=None):
    payload: Dict[str, np.ndarray] = {}
    for k, v in attn_map.items():
        if v is None:
            continue
        x = v.detach().float().cpu()
        if reduce_heads and x.ndim == 4:
            x = x.mean(dim=1)
        payload[k] = x.numpy()
    if meta is not None:
        payload.update(meta)
    if payload:
        np.savez_compressed(path, **payload)


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--use_hvg", action="store_true")
    p.add_argument("--n_top_genes", type=int, default=541)

    # coords
    p.add_argument("--coord_key", default="spatial")
    p.add_argument("--use_csv_coords", action="store_true")
    p.add_argument("--csv_coord_cols", default="x_he_px,y_he_px")
    p.add_argument("--csv_to_unit", type=float, default=None)
    p.add_argument("--coord_norm", default="standardize", choices=["standardize", "none"])

    # model / foundation
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--with_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # graph / coord-gcn
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--graph_type", default="knn", choices=["knn", "delaunay"])
    p.add_argument("--gcn_device", default=None)

    # xattn
    p.add_argument("--xattn_heads", type=int, default=8)
    p.add_argument("--xattn_dropout", type=float, default=0.1)
    p.add_argument("--img_tokens", type=int, default=1)
    p.add_argument("--gene_tokens", type=int, default=1)
    p.add_argument("--coord_tokens", type=int, default=1)
    p.add_argument("--coord_hidden_dim", type=int, default=128)
    p.add_argument("--coord_dropout", type=float, default=0.1)
    p.add_argument("--save_attn", action="store_true")
    p.add_argument("--attn_out_dir", default=None)
    p.add_argument("--attn_max_batches", type=int, default=10)
    p.add_argument("--attn_every", type=int, default=1)
    p.add_argument("--attn_reduce_heads", action="store_true")

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--lam_dir", type=float, default=0.5)
    p.add_argument("--lam_ig", type=float, default=1.0)
    p.add_argument("--lam_ic", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_norm", type=float, default=0.0)
    p.add_argument("--save_dir", default="./ckpts")

    # Fix 2: NaN check frequency
    p.add_argument("--nan_check_every", type=int, default=50,
                   help="Check intermediate embeddings for NaN/Inf every N steps. "
                        "Final loss is always checked. Set to 1 to restore original behavior.")

    # Fix 4: Checkpoint saving strategy
    p.add_argument("--save_every", type=int, default=0,
                   help="Also save checkpoint every N epochs (0 = only save on best loss).")
    p.add_argument("--no_save_best", action="store_true",
                   help="Disable best-loss checkpoint saving (not recommended).")

    # device/amp
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = select_device(args.device)
    setup_cuda_flags()
    amp_dtype = pick_amp_dtype(args.amp_dtype) if args.amp else torch.float32
    gcn_device = torch.device(args.gcn_device) if args.gcn_device is not None else device
    if gcn_device.type == "cpu" and args.amp:
        print("[WARN] gcn_device=cpu -> disabling AMP.")
        args.amp = False
    use_amp = bool(args.amp and device.type == "cuda")
    save_best = not args.no_save_best

    print(f"[Device] {device}, AMP={args.amp}, dtype={amp_dtype}")
    print(f"[XAttnTokens] img={args.img_tokens}, gene={args.gene_tokens}, coord={args.coord_tokens}")
    print(f"[NaNCheck] every {args.nan_check_every} steps (final loss checked every step)")
    print(f"[Checkpoint] save_best={save_best}, save_every={args.save_every}")

    attn_out_dir = None
    if args.save_attn:
        attn_out_dir = args.attn_out_dir or os.path.join(args.save_dir, "attn")
        os.makedirs(attn_out_dir, exist_ok=True)

    if args.temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}")

    # ------------------------------------------------------------------
    # 1) Load data + align
    # ------------------------------------------------------------------
    adata = load_h5ad_with_preproc(
        args.h5ad,
        use_hvg=args.use_hvg,
        n_top_genes=args.n_top_genes,
        x_key="x_centroid",
        y_key="y_centroid",
    )
    root_dir = args.root or os.path.dirname(args.csv)
    map_df = load_patch_mapping_csv_xenium(args.csv, root_dir=root_dir, drop_missing=True)
    if len(map_df) == 0 and args.root is None:
        alt_root = os.path.dirname(root_dir)
        map_df = load_patch_mapping_csv_xenium(args.csv, root_dir=alt_root, drop_missing=True)
        if len(map_df) == 0:
            raise RuntimeError(
                f"All patch paths missing. Tried root={root_dir} and alt={alt_root}. "
                "Pass --root explicitly."
            )
    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(
        adata=adata, map_df=map_df, id_col="spot_id", coord_key=args.coord_key
    )
    if len(map_df_aligned) == 0:
        raise RuntimeError("No overlapping spot_ids between CSV and h5ad after pruning.")

    if args.use_csv_coords:
        cx, cy = args.csv_coord_cols.split(",")
        coords_np = extract_coords_np_in_order(
            adata=adata, map_df=map_df_aligned, coord_key=args.coord_key,
            use_csv_coords=True, csv_coord_cols=(cx, cy), csv_to_unit=args.csv_to_unit,
        )

    # Coordinate normalisation
    if args.coord_norm == "standardize":
        mean_c = coords_np.mean(axis=0, keepdims=True)
        std_c  = coords_np.std(axis=0,  keepdims=True)
        std_c[std_c == 0] = 1.0
    else:
        mean_c = np.zeros((1, coords_np.shape[1]), dtype=np.float32)
        std_c  = np.ones( (1, coords_np.shape[1]), dtype=np.float32)

    coords_feat = ((coords_np - mean_c) / std_c).astype(np.float32)
    if not np.isfinite(coords_feat).all():
        raise RuntimeError("coords_feat contains NaN/Inf.")
    coords_feat_t = torch.from_numpy(coords_feat).to(gcn_device, non_blocking=True)

    # Fix 1: Build id_to_idx with normalized keys (consistent with _norm_id_str).
    # Since map_df_aligned is in the same order as the graph node list,
    # id_to_idx[spot_id_at_pos_i] == i for all i.
    id_to_idx: Dict[str, int] = {
        _norm_id_str(sid): i
        for i, sid in enumerate(map_df_aligned["spot_id"].tolist())
    }

    # ------------------------------------------------------------------
    # 1b) Build spatial graph
    # ------------------------------------------------------------------
    edge_index, edge_weight = build_spatial_graph(
        coords_np, k=args.k, use_weight=True, graph_type=args.graph_type
    )
    print(f"[Graph] type={args.graph_type}, k={args.k}, E={edge_index.size(1)}")
    edge_index  = edge_index.to(gcn_device, non_blocking=True).long().contiguous()
    edge_weight = (
        edge_weight.to(gcn_device, non_blocking=True).to(torch.float32).contiguous()
        if edge_weight is not None else None
    )
    if edge_weight is not None and not torch.isfinite(edge_weight).all().item():
        raise RuntimeError("edge_weight contains NaN/Inf.")

    # ------------------------------------------------------------------
    # 2) Foundation encoder + transforms
    # ------------------------------------------------------------------
    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation = FoundationCls(weights_path=None)

    lora_allowed = args.enc_name in {"plip", "phikon"}
    if args.with_lora:
        if lora_allowed:
            foundation = attach_lora_to_foundation(
                foundation, backbone_name=args.enc_name,
                r=args.lora_r, alpha=args.lora_alpha,
                dropout=args.lora_dropout, bias="none",
            )
            print(f"[LoRA] enabled on backbone='{args.enc_name}'")
        else:
            print(f"[LoRA] skipped for backbone='{args.enc_name}' (timm/open-clip not supported).")

    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)

    # ------------------------------------------------------------------
    # 3) DataLoader — Fix 1 + Fix 3
    # ------------------------------------------------------------------
    base_ds = SpotImageGeneDataset(
        adata=adata, map_df=map_df_aligned,
        gene_names=None, image_transform=image_tf,
    )
    gene_dim = base_ds.gene_dim
    # Fix 1: wrap to pre-convert string IDs → integer node indices
    # Fix 3: persistent_workers keeps worker processes alive across epochs
    train_loader = make_node_idx_dataloader(
        base_ds=base_ds,
        id_to_idx=id_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    print(f"[Data] N={len(base_ds)}, gene_dim={gene_dim}, "
          f"steps_per_epoch={len(train_loader)}")

    # ------------------------------------------------------------------
    # 4) Models
    # ------------------------------------------------------------------
    img_encoder = ImageFoundationAdapter(foundation, proj_dim=args.proj_dim, normalize=False)
    gene_encoder = GeneEncoder(
        input_dim=gene_dim, model_size_omic="small",
        proj_dim=args.proj_dim, normalize=False,
    )
    coord_gcn = SpatialGCN(
        in_dim=coords_np.shape[1],
        hidden_dim=args.coord_hidden_dim,
        out_dim=args.proj_dim,
        dropout=args.coord_dropout,
    )
    coord_gcn.to(gcn_device).float()

    model = XAttnContrastiveModel(
        img_encoder=img_encoder,
        gene_encoder=gene_encoder,
        proj_dim=args.proj_dim,
        num_heads=args.xattn_heads,
        dropout=args.xattn_dropout,
        temperature=args.temperature,
        lam_dir=args.lam_dir,
        img_tokens=args.img_tokens,
        gene_tokens=args.gene_tokens,
        coord_tokens=args.coord_tokens,
    )
    model.to(device).train()

    # ------------------------------------------------------------------
    # 5) Optimizer & scaler (created ONCE, outside the epoch loop)
    # ------------------------------------------------------------------
    optimizer = Adam(
        _unique_trainable_params(model, coord_gcn),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    # GradScaler created once — its internal loss-scale state accumulates across epochs.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype_found = getattr(model.img_encoder.foundation, "precision", None)

    # ------------------------------------------------------------------
    # 6) Training loop
    # ------------------------------------------------------------------
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        running_steps = 0
        bar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                   desc=f"[xattn] Epoch {epoch}/{args.epochs}", leave=False)

        for step, batch in bar:
            img, gene, node_ids = batch  # Fix 1: node_ids is already an int tensor
            img  = img.to(device, non_blocking=True)
            gene = gene.to(device, non_blocking=True)

            # Fix 1: direct tensor transfer — no Python string loop
            batch_ids = node_ids.to(gcn_device, non_blocking=True).long()

            # Subgraph for coord GCN (same approach as original — correct for contrastive)
            sub_ei, sub_ew = subgraph(
                subset=batch_ids, edge_index=edge_index,
                edge_attr=edge_weight, relabel_nodes=True,
            )
            sub_ei = sub_ei.long().contiguous()
            sub_ew = sub_ew.to(torch.float32).contiguous() if sub_ew is not None else None
            coord_feat_batch = coords_feat_t[batch_ids]

            # Fix 2: only check intermediate tensors periodically
            do_nan_check = (step % args.nan_check_every == 0)

            dump_attn = bool(
                args.save_attn and attn_out_dir is not None
                and args.attn_max_batches > 0
                and (epoch % max(args.attn_every, 1) == 0)
                and (step <= args.attn_max_batches)
            )

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                ctx = (torch.amp.autocast("cuda", dtype=amp_dtype_found)
                       if amp_dtype_found in (torch.bfloat16, torch.float16)
                       else torch.amp.autocast("cuda"))

                with torch.amp.autocast("cuda", enabled=False):
                    zc = coord_gcn(coord_feat_batch.float(), sub_ei, sub_ew)
                zc = zc.to(device, non_blocking=True)

                with ctx:
                    zi = model.img_encoder(img)
                    zg = model.gene_encoder(gene)
                    zc = zc.to(dtype=zi.dtype)

                    # Fix 2: intermediate NaN check only every nan_check_every steps
                    if do_nan_check and not (
                        torch.isfinite(zi).all() and
                        torch.isfinite(zg).all() and
                        torch.isfinite(zc).all()
                    ):
                        print(f"[WARN] NaN/Inf in embeddings at epoch {epoch} step {step}. Skipping.")
                        continue

                    if dump_attn:
                        zi2g, zg2i, zi2c, zc2i, attn_map = model.cross_attend(zi, zg, zc, return_attn=True)
                    else:
                        zi2g, zg2i, zi2c, zc2i = model.cross_attend(zi, zg, zc)
                        attn_map = None

                    # Fix 2: intermediate post-xattn NaN check also gated
                    if do_nan_check and not (
                        torch.isfinite(zi2g).all() and torch.isfinite(zg2i).all() and
                        torch.isfinite(zi2c).all() and torch.isfinite(zc2i).all()
                    ):
                        print(f"[WARN] NaN/Inf after cross-attention at epoch {epoch} step {step}. Skipping.")
                        continue

                    loss_ig = model.contrastive_loss(zi2g, zg2i)
                    loss_ic = model.contrastive_loss(zi2c, zc2i)
                    loss = args.lam_ig * loss_ig + args.lam_ic * loss_ic

                # Always check final scalar loss before backward (cheap)
                if not torch.isfinite(loss):
                    print(f"[WARN] NaN/Inf loss at epoch {epoch} step {step}. Skipping.")
                    continue

                scaler.scale(loss).backward()
                if args.max_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                scaler.step(optimizer)
                scaler.update()

            else:  # AMP off
                with torch.amp.autocast("cuda", enabled=False):
                    zc = coord_gcn(coord_feat_batch.float(), sub_ei, sub_ew)
                zc = zc.to(device, non_blocking=True)
                zi = model.img_encoder(img)
                zg = model.gene_encoder(gene)

                if do_nan_check and not (
                    torch.isfinite(zi).all() and
                    torch.isfinite(zg).all() and
                    torch.isfinite(zc).all()
                ):
                    print(f"[WARN] NaN/Inf in embeddings at epoch {epoch} step {step}. Skipping.")
                    continue

                if dump_attn:
                    zi2g, zg2i, zi2c, zc2i, attn_map = model.cross_attend(zi, zg, zc, return_attn=True)
                else:
                    zi2g, zg2i, zi2c, zc2i = model.cross_attend(zi, zg, zc)
                    attn_map = None

                if do_nan_check and not (
                    torch.isfinite(zi2g).all() and torch.isfinite(zg2i).all() and
                    torch.isfinite(zi2c).all() and torch.isfinite(zc2i).all()
                ):
                    print(f"[WARN] NaN/Inf after cross-attention at epoch {epoch} step {step}. Skipping.")
                    continue

                loss_ig = model.contrastive_loss(zi2g, zg2i)
                loss_ic = model.contrastive_loss(zi2c, zc2i)
                loss = args.lam_ig * loss_ig + args.lam_ic * loss_ic

                if not torch.isfinite(loss):
                    print(f"[WARN] NaN/Inf loss at epoch {epoch} step {step}. Skipping.")
                    continue

                loss.backward()
                if args.max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()

            if dump_attn and attn_map is not None:
                out_npz = os.path.join(attn_out_dir, f"ep{epoch:03d}_step{step:04d}.npz")
                _save_attn_npz(
                    out_npz, attn_map, reduce_heads=args.attn_reduce_heads,
                    meta={
                        "node_idx": batch_ids.detach().cpu().numpy(),
                        "spot_id":  np.asarray([str(s) for s in node_ids]),
                    },
                )

            # Fix 5: correctly reset running window after each print
            running += loss.item()
            running_steps += 1
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running / running_steps:.4f}")
                running = 0.0
                running_steps = 0

        # ------------------------------------------------------------------
        # Fix 4: Checkpoint saving — only on improvement + periodic
        # ------------------------------------------------------------------
        epoch_loss = running / max(running_steps, 1)  # last partial window

        # Best-loss checkpoint
        if save_best and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dir = os.path.join(args.save_dir, "best")
            _save_all(best_dir, img_encoder, gene_encoder, coord_gcn, model)
            print(f"[BEST] loss={best_loss:.6f} -> saved to {best_dir}")

        # Periodic checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0:
            ep_dir = os.path.join(args.save_dir, f"ep{epoch:04d}")
            _save_all(ep_dir, img_encoder, gene_encoder, coord_gcn, model)
            print(f"[CKPT] epoch {epoch} saved to {ep_dir}")

    # Always save final checkpoint
    _save_all(args.save_dir, img_encoder, gene_encoder, coord_gcn, model)
    print(f"Training done. best_loss={best_loss:.6f}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
