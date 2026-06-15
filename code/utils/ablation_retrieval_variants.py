#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_retrieval_variants.py
-------------------------------
Ablation study: compare 5 retrieval strategies for the CrossAttn-ZINB model.

Variants tested (input expression to the model inference):
  A) Mean       : global mean of training expression (no retrieval)
  B) Global     : global cosine top-1 retrieval, no cell-type constraint
  C) CT-Hard    : cell-type guided hard top-1 (argmax predicted class, then top-1 cosine)
  D) CT-WS-Fixed: cell-type quota + cosine-weighted sum, FIXED temperature  τ=τ_base
  E) CT-WS-Adapt: cell-type quota + cosine-weighted sum, ENTROPY-ADAPTIVE τ  (full method)

For each variant the model decoder is run with the constructed expression input
and μ (mean of ZINB) is compared against the ground-truth test expression.

Metrics (per variant):
  • Mean PCC across genes     (log1p space, as used during training)
  • Median PCC
  • MSE  (log1p space)
  • MAE  (log1p space)
  • Per-cell-type mean PCC    (if cell-type annotations available)

Saved outputs:
  <out_dir>/ablation_summary.json        — all scalar metrics for all variants
  <out_dir>/ablation_summary.csv         — same, one row per variant (convenient for tables)
  <out_dir>/per_celltype_pcc.csv         — per-class PCC for each variant
  <out_dir>/per_gene_pcc_<variant>.npy   — (n_genes,) PCC array per variant

Example:
  python ablation_retrieval_variants.py \\
    --train_h5ad /data/.../train.h5ad \\
    --train_csv  /data/.../train_patches.csv \\
    --test_h5ad  /data/.../test.h5ad \\
    --test_csv   /data/.../test_patches.csv \\
    --ft_model_ckpt /data/.../best/model.pt \\
    --celltype_meta_pt /data/.../best/celltype_meta.pt \\
    --celltype_col cell_type \\
    --enc_name uni_v1 \\
    --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \\
    --img_tokens 10 --spot_tokens 10 --gene_tokens 10 \\
    --graph_type delaunay --k 12 \\
    --bank_size 500 --tau_base 0.07 --tau_entropy_alpha 1.0 \\
    --variants A B C D E \\
    --batch_size 128 --num_workers 4 \\
    --out_dir ./ablation/retrieval_variants
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.modules.pop("models", None)

from models.Foundations import inf_encoder_factory
from models.GCN_update import SpatialGCN
from models.gene_encoder import GeneEncoder
from models.fusion import CrossModalFusion
from models.decoders import ZINBDecoder
from models.graph_construction import build_knn_graph
from models.graph_construction_delaunay import build_delaunay_graph
from utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
    make_img_only_dataloader,
)


# ============================================================
# Minimal image-only dataset for embedding
# ============================================================
class _ImgIdxDataset(Dataset):
    def __init__(self, map_df, image_transform=None):
        self.df = map_df.reset_index(drop=True)
        self.tf = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["patch_path"]).convert("RGB")
        img = self.tf(img) if self.tf is not None else (
            torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0)
        return img, idx


# ============================================================
# Model (identical to run_finetunning_xattn_withceltype_0214_retrieval_final_version.py)
# ============================================================
class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn,
                 dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1, merge="gated-sum",
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
            dim_out=dim_fuse, num_heads=num_heads, dropout=dropout, merge=merge,
            img_tokens=img_tokens, spot_tokens=spot_tokens, gene_tokens=gene_tokens,
        )
        self.decoder = ZINBDecoder(in_dim=dim_fuse, gene_dim=gene_dim,
                                    hidden=decoder_hidden, dropout=dropout)
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

    def forward(self, batch, spot_feat_in, edge_index, edge_weight=None):
        x_img    = batch["image"]
        x_gene   = batch["expr"].float()
        node_idx = batch["idx"].long()
        zi = self.img_encoder(x_img)
        zg = self.gene_encoder(x_gene)
        zs_all = self.gcn(spot_feat_in.float(), edge_index,
                          edge_weight.float() if edge_weight is not None else None)
        zs = zs_all[node_idx]
        zi = self.proj_img(zi);  zg = self.proj_gene(zg);  zs = self.proj_spot(zs)
        z_fused, _, _ = self.fusion(zi, zs, zg, return_attn=False)
        mu, theta, pi = self.decoder(z_fused)
        cls_logits = self.cls_head(z_fused) if self.cls_head is not None else None
        return mu, theta, pi, cls_logits


# ============================================================
# Utilities
# ============================================================
def _load_sd(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and any(k in sd for k in ["state_dict", "model", "module"]):
        for k in ["state_dict", "model", "module"]:
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd if isinstance(sd, dict) else sd.state_dict()


def _infer_cls_head_config(sd):
    if not isinstance(sd, dict):
        return 0, 0
    if "cls_head.weight" in sd:
        return int(sd["cls_head.weight"].shape[0]), 0
    if "cls_head.0.weight" in sd and "cls_head.3.weight" in sd:
        return int(sd["cls_head.3.weight"].shape[0]), int(sd["cls_head.0.weight"].shape[0])
    return 0, 0


def select_device(arg):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if ":" in arg and arg.startswith("cuda"):
        torch.cuda.set_device(int(arg.split(":")[1]))
    return torch.device(arg)


def build_graph(coords_np, k, graph_type):
    if graph_type.lower() == "delaunay":
        return build_delaunay_graph(coords_np, k=k, use_weight=True)
    return build_knn_graph(coords_np, k=k, use_weight=True)


def _adata_x(adata):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def pcc_per_gene(pred: np.ndarray, true: np.ndarray):
    Yc = true - true.mean(0, keepdims=True)
    Pc = pred - pred.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0)) + 1e-8
    return num / den


def eval_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pcc = pcc_per_gene(pred, true)
    mse = float(((pred - true) ** 2).mean())
    mae = float(np.abs(pred - true).mean())
    return {
        "pcc_mean":   round(float(np.nanmean(pcc)), 4),
        "pcc_median": round(float(np.nanmedian(pcc)), 4),
        "mse":        round(mse, 6),
        "mae":        round(mae, 6),
    }, pcc


# ============================================================
# Entropy-adaptive per-query temperature (identical to finetuning script)
# ============================================================
def _compute_per_query_tau(query_probs: np.ndarray, tau_base: float,
                            tau_entropy_alpha: float) -> np.ndarray:
    """τ_q = τ_base × (1 + α × H_norm(p_q));  (N,) float array."""
    p = np.clip(query_probs.astype(np.float64), 1e-9, None)
    C = float(p.shape[1])
    H = -(p * np.log(p)).sum(axis=1)
    H_norm = H / (np.log(C) if C > 1 else 1.0)
    tau_q = tau_base * (1.0 + tau_entropy_alpha * H_norm)
    return np.clip(tau_q, 1e-3, 50.0).astype(np.float32)


def _allocate_cluster_counts(prob_vec: np.ndarray, total: int) -> np.ndarray:
    """Integer quota: floor + fractional-remainder greedy."""
    total = max(int(total), 0)
    if total == 0:
        return np.zeros_like(prob_vec, dtype=np.int64)
    p = np.clip(prob_vec.astype(np.float64), 0.0, None)
    s = float(p.sum())
    p = p / s if s > 0 else np.ones_like(p) / max(int(p.size), 1)
    raw = p * float(total)
    cnt = np.floor(raw).astype(np.int64)
    rem = int(total - int(cnt.sum()))
    if rem > 0:
        order = np.argsort(-(raw - cnt.astype(np.float64)))
        cnt[order[:rem]] += 1
    return cnt


# ============================================================
# Build projected image key bank from training set
# ============================================================
@torch.no_grad()
def build_key_bank(model, train_map, image_tf, device, batch_size, num_workers, amp_dtype):
    """Returns (N_train, D) normalized projected image embeddings."""
    ds  = _ImgIdxDataset(train_map, image_transform=image_tf)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=True, drop_last=False)
    dim = model.proj_img.out_features
    bank = torch.zeros((len(ds), dim), dtype=torch.float32, device=device)
    use_amp = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
    model.eval()
    for imgs, idxs in tqdm(dl, desc="build key_bank", leave=False):
        imgs = imgs.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            q = model.proj_img(model.img_encoder(imgs))
        q = F.normalize(q.float(), dim=1)
        bank[idxs.to(device)] = q.detach()
    return bank  # (N, D)  on device


# ============================================================
# Build cls_head probability bank from training set
# (model already loaded, uses mean expression as dummy gene input)
# ============================================================
@torch.no_grad()
def build_prob_bank(model, train_map, train_coords_np, image_tf,
                    mean_expr_np, device, batch_size, num_workers, amp_dtype,
                    edge_index, edge_weight, spot_feat_in):
    """
    Returns (N_train, C) softmax probability matrix.
    mean_expr_np : (G,) mean training expression for dummy gene input.
    """
    if model.cls_head is None:
        raise RuntimeError("Model has no cls_head. Cannot build prob bank.")
    n_train = len(train_map)
    num_ct  = model.num_celltypes
    prob_bank = np.zeros((n_train, num_ct), dtype=np.float32)
    mean_expr_t = torch.from_numpy(
        np.repeat(mean_expr_np[None, :], n_train, axis=0).astype(np.float32)
    ).to(device)

    ds  = _ImgIdxDataset(train_map, image_transform=image_tf)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=True, drop_last=False)
    use_amp = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
    model.eval()
    for imgs, idxs in tqdm(dl, desc="build prob_bank", leave=False):
        imgs = imgs.to(device)
        idxs_d = idxs.to(device).long()
        expr = mean_expr_t[idxs_d]
        batch = {"image": imgs, "expr": expr, "idx": idxs_d}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, _, _, cls_logits = model(batch, spot_feat_in, edge_index, edge_weight)
        if cls_logits is None:
            raise RuntimeError("cls_logits is None from model.")
        probs = torch.softmax(cls_logits.float(), dim=1).detach().cpu().numpy()
        prob_bank[idxs.numpy()] = probs
    return prob_bank  # (N, C)


# ============================================================
# Build class_index_map {cls_idx: np.int64[]} from prob bank (argmax)
# ============================================================
def build_class_index_map(prob_bank: np.ndarray, min_conf: float = 0.0) -> dict:
    pred_cls  = prob_bank.argmax(axis=1).astype(np.int64)
    pred_conf = prob_bank.max(axis=1)
    n_cls = prob_bank.shape[1]
    cmap = {}
    for c in range(n_cls):
        inds = np.where((pred_cls == c) & (pred_conf >= min_conf))[0]
        if inds.size > 0:
            cmap[c] = inds.astype(np.int64)
    return cmap


# ============================================================
# Expression construction functions for each variant
# ============================================================

# ── A: Mean ──────────────────────────────────────────────────────────────────
def build_expr_mean(mean_expr: np.ndarray, n_test: int) -> np.ndarray:
    return np.repeat(mean_expr[None, :], n_test, axis=0).astype(np.float32)


# ── B: Global top-1 ──────────────────────────────────────────────────────────
def build_expr_global_top1(sim_mat: np.ndarray, train_expr: np.ndarray) -> np.ndarray:
    best_idx = sim_mat.argmax(axis=1)
    return train_expr[best_idx].astype(np.float32)


# ── C: CT-Hard (cell-type guided top-1, hard argmax assignment) ───────────────
def build_expr_ct_hard(sim_mat: np.ndarray, pred_probs: np.ndarray,
                        class_index_map: dict, train_expr: np.ndarray) -> np.ndarray:
    """
    For each query: argmax predicted class → top-1 cosine within that class.
    Falls back to global top-1 if the class has no members in the bank.
    """
    n_test = sim_mat.shape[0]
    out = np.zeros((n_test, train_expr.shape[1]), dtype=np.float32)
    pred_cls = pred_probs.argmax(axis=1).astype(np.int64)
    for qi in range(n_test):
        c    = int(pred_cls[qi])
        cand = class_index_map.get(c, None)
        if cand is not None and cand.size > 0:
            local_sims = sim_mat[qi, cand]
            best_local = int(cand[local_sims.argmax()])
        else:
            best_local = int(sim_mat[qi].argmax())
        out[qi] = train_expr[best_local]
    return out


# ── D: CT-WS-Fixed (quota + cosine-weighted sum, fixed temperature) ──────────
def build_expr_ct_ws_fixed(sim_mat: np.ndarray, pred_probs: np.ndarray,
                            class_index_map: dict, train_expr: np.ndarray,
                            bank_size: int, tau: float) -> np.ndarray:
    """
    Probability-proportional quota retrieval + cosine-weighted sum, fixed τ.
    """
    n_test = sim_mat.shape[0]
    out = np.zeros((n_test, train_expr.shape[1]), dtype=np.float32)
    for qi in range(n_test):
        prob_vec = pred_probs[qi].astype(np.float64)
        quota    = _allocate_cluster_counts(prob_vec, bank_size)
        selected = []
        for c, take_n in enumerate(quota.tolist()):
            if take_n <= 0:
                continue
            cand = class_index_map.get(c, None)
            if cand is None or cand.size == 0:
                continue
            sims = sim_mat[qi, cand]
            kk   = min(int(take_n), len(cand))
            top  = cand[np.argpartition(-sims, kth=kk-1)[:kk]]
            selected.append(top)
        if selected:
            sel = np.unique(np.concatenate(selected).astype(np.int64))
        else:
            sel = np.empty(0, dtype=np.int64)
        # fill shortfall from global similarity
        if sel.size < bank_size:
            sim_fill = sim_mat[qi].copy()
            if sel.size > 0:
                sim_fill[sel] = -1e9
            need   = min(bank_size - sel.size, sim_fill.shape[0])
            fill   = np.argpartition(-sim_fill, kth=need-1)[:need]
            sel    = np.concatenate([sel, fill]) if sel.size > 0 else fill
        if sel.size == 0:
            out[qi] = train_expr[sim_mat[qi].argmax()]
            continue
        sims_sel = sim_mat[qi, sel]
        if sims_sel.size > bank_size:
            top_loc  = np.argpartition(-sims_sel, kth=bank_size-1)[:bank_size]
            sel      = sel[top_loc]
            sims_sel = sims_sel[top_loc]
        w = np.exp(sims_sel / tau)
        w = w / (w.sum() + 1e-9)
        out[qi] = (w[:, None] * train_expr[sel]).sum(0)
    return out


# ── E: CT-WS-Adapt (quota + cosine-weighted sum, entropy-adaptive τ) ─────────
def build_expr_ct_ws_adapt(sim_mat: np.ndarray, pred_probs: np.ndarray,
                            class_index_map: dict, train_expr: np.ndarray,
                            bank_size: int, tau_base: float,
                            tau_entropy_alpha: float) -> np.ndarray:
    """Full method: per-query entropy-adaptive temperature."""
    n_test = sim_mat.shape[0]
    out = np.zeros((n_test, train_expr.shape[1]), dtype=np.float32)
    tau_q = _compute_per_query_tau(pred_probs, tau_base, tau_entropy_alpha)  # (N,)
    for qi in range(n_test):
        prob_vec = pred_probs[qi].astype(np.float64)
        quota    = _allocate_cluster_counts(prob_vec, bank_size)
        selected = []
        for c, take_n in enumerate(quota.tolist()):
            if take_n <= 0:
                continue
            cand = class_index_map.get(c, None)
            if cand is None or cand.size == 0:
                continue
            sims = sim_mat[qi, cand]
            kk   = min(int(take_n), len(cand))
            top  = cand[np.argpartition(-sims, kth=kk-1)[:kk]]
            selected.append(top)
        if selected:
            sel = np.unique(np.concatenate(selected).astype(np.int64))
        else:
            sel = np.empty(0, dtype=np.int64)
        if sel.size < bank_size:
            sim_fill = sim_mat[qi].copy()
            if sel.size > 0:
                sim_fill[sel] = -1e9
            need   = min(bank_size - sel.size, sim_fill.shape[0])
            fill   = np.argpartition(-sim_fill, kth=need-1)[:need]
            sel    = np.concatenate([sel, fill]) if sel.size > 0 else fill
        if sel.size == 0:
            out[qi] = train_expr[sim_mat[qi].argmax()]
            continue
        sims_sel = sim_mat[qi, sel]
        if sims_sel.size > bank_size:
            top_loc  = np.argpartition(-sims_sel, kth=bank_size-1)[:bank_size]
            sel      = sel[top_loc]
            sims_sel = sims_sel[top_loc]
        tq = float(tau_q[qi])
        w = np.exp(sims_sel / tq)
        w = w / (w.sum() + 1e-9)
        out[qi] = (w[:, None] * train_expr[sel]).sum(0)
    return out


# ============================================================
# Run model inference with a given expression matrix → μ predictions
# ============================================================
@torch.no_grad()
def run_inference(model, test_map, test_coords_np, expr_matrix_np,
                  image_tf, device, batch_size, num_workers, amp_dtype,
                  edge_index, edge_weight, spot_feat_in):
    """Returns (N_test, G) μ predictions in log1p space."""
    n_test  = len(test_map)
    gene_dim = expr_matrix_np.shape[1]
    preds = []
    expr_t = torch.from_numpy(expr_matrix_np).to(device)

    ds = _ImgIdxDataset(test_map, image_transform=image_tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)
    use_amp = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
    model.eval()
    for imgs, idxs in tqdm(dl, desc="inference", leave=False):
        imgs   = imgs.to(device)
        idxs_d = idxs.to(device).long()
        expr   = expr_t[idxs_d]
        batch  = {"image": imgs, "expr": expr, "idx": idxs_d}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            mu, _, _, _ = model(batch, spot_feat_in, edge_index, edge_weight)
        mu_log = torch.log1p(torch.clamp(mu.float(), min=0.0))
        preds.append(mu_log.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Ablation: compare retrieval variants")
    # data
    ap.add_argument("--train_h5ad",  required=True)
    ap.add_argument("--train_csv",   required=True)
    ap.add_argument("--train_root",  default=None)
    ap.add_argument("--test_h5ad",   required=True)
    ap.add_argument("--test_csv",    required=True)
    ap.add_argument("--test_root",   default=None)
    ap.add_argument("--id_col",      default="cell_id")
    ap.add_argument("--path_col",    default="patch_path")
    ap.add_argument("--celltype_col", default=None)
    ap.add_argument("--celltype_meta_pt", default=None)
    # model
    ap.add_argument("--enc_name",      default="uni_v1")
    ap.add_argument("--proj_dim_gene", type=int, default=256)
    ap.add_argument("--proj_dim_spot", type=int, default=256)
    ap.add_argument("--fuse_dim",      type=int, default=256)
    ap.add_argument("--heads",         type=int, default=8)
    ap.add_argument("--dropout",       type=float, default=0.1)
    ap.add_argument("--merge",         default="gated-sum")
    ap.add_argument("--img_tokens",  type=int, default=1)
    ap.add_argument("--spot_tokens", type=int, default=1)
    ap.add_argument("--gene_tokens", type=int, default=1)
    ap.add_argument("--graph_type",  default="delaunay", choices=["knn","delaunay"])
    ap.add_argument("--k",           type=int, default=12)
    ap.add_argument("--ft_model_ckpt", required=True)
    # retrieval params
    ap.add_argument("--bank_size",         type=int,   default=500,
                    help="Number of training samples retrieved per query")
    ap.add_argument("--tau_base",          type=float, default=0.07,
                    help="Base temperature for variants D and E")
    ap.add_argument("--tau_entropy_alpha", type=float, default=1.0,
                    help="Entropy scaling factor for variant E (adaptive τ)")
    ap.add_argument("--min_conf",          type=float, default=0.0,
                    help="Minimum predicted class confidence to include in class_index_map")
    # which variants to run
    ap.add_argument("--variants", nargs="+", default=["A","B","C","D","E"],
                    choices=["A","B","C","D","E"],
                    help="Retrieval variants to evaluate")
    # run
    ap.add_argument("--batch_size",  type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--amp",         action="store_true")
    ap.add_argument("--sim_chunk",   type=int, default=512,
                    help="Chunk size for computing (N_test, N_train) similarity matrix")
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = select_device(args.device)
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16) if (args.amp and device.type == "cuda") else torch.float32
    print(f"[Device] {device}  AMP={args.amp}  dtype={amp_dtype}")
    print(f"[Variants] {args.variants}")

    # ── load data ─────────────────────────────────────────────────────────────
    train_adata = load_h5ad_with_preproc(args.train_h5ad, use_hvg=False, n_top_genes=None,
                                         x_key="x_centroid", y_key="y_centroid")
    test_adata  = load_h5ad_with_preproc(args.test_h5ad,  use_hvg=False, n_top_genes=None,
                                         x_key="x_centroid", y_key="y_centroid")
    common    = train_adata.var_names.intersection(test_adata.var_names)
    gene_list = [g for g in train_adata.var_names if g in common]
    train_adata = train_adata[:, gene_list].copy()
    test_adata  = test_adata[:,  gene_list].copy()
    gene_dim = len(gene_list)
    print(f"[Genes] aligned gene_dim={gene_dim}")

    def _prep(adata, csv, root):
        map_df = load_patch_mapping_csv_xenium(csv, root_dir=root,
                                               id_col=args.id_col, path_col=args.path_col,
                                               drop_missing=True)
        map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
        map_df, coords = reorder_and_prune_by_obs(adata, map_df, id_col="spot_id",
                                                   coord_key="spatial")
        return map_df, coords

    train_map, train_coords = _prep(train_adata, args.train_csv, args.train_root)
    test_map,  test_coords  = _prep(test_adata,  args.test_csv,  args.test_root)
    if test_map.shape[0] != test_adata.n_obs:
        test_adata = test_adata[test_map["spot_id"].astype(str).tolist()].copy()

    train_expr = _adata_x(train_adata)
    test_expr  = _adata_x(test_adata)  # ground-truth test expression (log1p)
    n_train    = len(train_map)
    n_test     = len(test_map)

    # mean training expression (for variant A and prob bank input)
    mean_expr  = train_expr.mean(axis=0).astype(np.float32)

    # ── celltype labels (for per-class PCC reporting) ─────────────────────────
    ct_meta_path = args.celltype_meta_pt
    if ct_meta_path is None and args.ft_model_ckpt:
        cand = os.path.join(os.path.dirname(args.ft_model_ckpt), "celltype_meta.pt")
        if os.path.isfile(cand):
            ct_meta_path = cand
    ct_meta       = torch.load(ct_meta_path, map_location="cpu") if (ct_meta_path and os.path.isfile(ct_meta_path)) else {}
    celltype_names = [str(x) for x in ct_meta.get("celltype_names", [])]

    def _get_labels(adata, map_df):
        if not celltype_names:
            return None
        hint_col = args.celltype_col
        cols_lower = {str(c).lower(): c for c in adata.obs.columns}
        hints = [hint_col] if hint_col else []
        hints += ["cell_type","celltype","celltypes","annotation","cluster","leiden"]
        ct_col = None
        for h in hints:
            if h and h.lower() in cols_lower:
                ct_col = cols_lower[h.lower()]; break
        if ct_col is None:
            return None
        sids   = map_df["spot_id"].astype(str).tolist()
        gt_str = adata.obs.loc[sids, ct_col].astype(str).to_numpy()
        n2i    = {n: i for i, n in enumerate(celltype_names)}
        return np.array([n2i.get(n, -1) for n in gt_str], dtype=np.int64)

    test_labels = _get_labels(test_adata, test_map)

    # ── model ──────────────────────────────────────────────────────────────────
    ft_sd  = _load_sd(args.ft_model_ckpt)
    ckpt_num_ct, ckpt_ct_hidden = _infer_cls_head_config(ft_sd)
    meta_num_ct = int(ct_meta.get("num_celltypes", 0)) if ct_meta else 0
    num_celltypes = max(meta_num_ct, ckpt_num_ct)

    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation    = FoundationCls(weights_path=None)
    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)
    img_dim  = foundation.get_output_dim()

    # Test graph (for model inference on test set)
    test_edge_idx, test_edge_w = build_graph(test_coords, k=args.k, graph_type=args.graph_type)
    test_edge_idx = test_edge_idx.to(device)
    test_edge_w   = test_edge_w.to(device)  if test_edge_w is not None  else None
    test_spot_feat = torch.from_numpy(test_coords).float().to(device)

    gcn = SpatialGCN(in_dim=test_coords.shape[1], hidden_dim=256,
                     out_dim=args.proj_dim_spot, dropout=0.2)
    gene_encoder = GeneEncoder(input_dim=gene_dim, model_size_omic="small",
                               proj_dim=args.proj_dim_gene, normalize=False)
    model = CrossAttnZINBModel(
        img_encoder=foundation, gene_encoder=gene_encoder, gcn=gcn,
        dim_img=img_dim, dim_gene=args.proj_dim_gene, dim_spot=args.proj_dim_spot,
        dim_fuse=args.fuse_dim, num_heads=args.heads, dropout=args.dropout, merge=args.merge,
        img_tokens=args.img_tokens, spot_tokens=args.spot_tokens, gene_tokens=args.gene_tokens,
        decoder_hidden=1024, gene_dim=gene_dim,
        num_celltypes=num_celltypes, cls_head_hidden=ckpt_ct_hidden,
    ).to(device)
    model.load_state_dict(ft_sd, strict=False)
    model.eval()
    print(f"[Model] loaded  num_celltypes={num_celltypes}  gene_dim={gene_dim}")

    # ── train graph (for prob bank) ────────────────────────────────────────────
    train_edge_idx, train_edge_w = build_graph(train_coords, k=args.k, graph_type=args.graph_type)
    train_edge_idx = train_edge_idx.to(device)
    train_edge_w   = train_edge_w.to(device)  if train_edge_w is not None  else None
    train_spot_feat = torch.from_numpy(train_coords).float().to(device)

    # ── precompute: key bank, prob bank, class index map, sim matrix ───────────
    needs_retrieval = any(v in args.variants for v in ["B","C","D","E"])
    needs_ct_bank   = any(v in args.variants for v in ["C","D","E"])

    key_bank  = None
    prob_bank = None
    class_idx_map = None
    sim_mat   = None

    if needs_retrieval:
        print("[Build] projected image key bank (training) ...")
        key_bank_t = build_key_bank(model, train_map, image_tf, device,
                                     args.batch_size, args.num_workers, amp_dtype)
        print("[Build] projected image queries (test) ...")
        ds_test = _ImgIdxDataset(test_map, image_transform=image_tf)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
        use_amp_flag = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
        query_bank = torch.zeros((n_test, model.proj_img.out_features),
                                  dtype=torch.float32, device=device)
        with torch.no_grad():
            for imgs, idxs in tqdm(dl_test, desc="embed(test queries)", leave=False):
                imgs = imgs.to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp_flag):
                    q = model.proj_img(model.img_encoder(imgs))
                q = F.normalize(q.float(), dim=1)
                query_bank[idxs.to(device)] = q.detach()

        # similarity matrix
        print("[Build] similarity matrix (test × train) ...")
        sim_mat = np.zeros((n_test, n_train), dtype=np.float32)
        train_cpu = key_bank_t.cpu()
        for s in tqdm(range(0, n_test, args.sim_chunk), desc="sim", leave=False):
            e = min(n_test, s + args.sim_chunk)
            q_chunk = query_bank[s:e].cpu()
            sim_mat[s:e] = (q_chunk @ train_cpu.t()).numpy()
        del key_bank_t, query_bank

    if needs_ct_bank:
        if num_celltypes <= 0:
            print("[WARN] No cls_head in checkpoint; variants C/D/E require cell-type predictions. "
                  "Skipping these variants.")
            args.variants = [v for v in args.variants if v not in ("C","D","E")]
        else:
            print("[Build] predicted prob bank (training, cls_head) ...")
            prob_bank = build_prob_bank(
                model, train_map, train_coords, image_tf, mean_expr, device,
                args.batch_size, args.num_workers, amp_dtype,
                train_edge_idx, train_edge_w, train_spot_feat,
            )
            class_idx_map = build_class_index_map(prob_bank, min_conf=args.min_conf)
            print(f"[Build] class_index_map: {len(class_idx_map)} classes populated")

    # ── also build test-side prob bank for variants C/D/E ─────────────────────
    # We need the predicted probs for TEST patches to compute their entropy (variant E)
    # and class assignment (variants C/D/E).
    test_prob_bank = None
    if any(v in args.variants for v in ["C","D","E"]) and num_celltypes > 0:
        print("[Build] predicted prob bank (test, cls_head) ...")
        test_prob_bank = build_prob_bank(
            model, test_map, test_coords, image_tf, mean_expr, device,
            args.batch_size, args.num_workers, amp_dtype,
            test_edge_idx, test_edge_w, test_spot_feat,
        )

    # ── run variants ───────────────────────────────────────────────────────────
    VARIANT_NAMES = {
        "A": "Mean",
        "B": "Global-Top1",
        "C": "CT-Hard-Top1",
        "D": "CT-WS-Fixed-τ",
        "E": "CT-WS-Adaptive-τ (Full)",
    }

    all_results  = {}  # variant → metrics dict
    all_pcc_gene = {}  # variant → (n_genes,) array
    per_ct_data  = {}  # variant → {class_name: mean_pcc}

    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"[Variant {variant}] {VARIANT_NAMES[variant]}")
        t_start = time.time()

        # ── build input expression matrix ─────────────────────────────────────
        if variant == "A":
            expr_in = build_expr_mean(mean_expr, n_test)

        elif variant == "B":
            expr_in = build_expr_global_top1(sim_mat, train_expr)

        elif variant == "C":
            expr_in = build_expr_ct_hard(
                sim_mat, test_prob_bank, class_idx_map, train_expr)

        elif variant == "D":
            expr_in = build_expr_ct_ws_fixed(
                sim_mat, test_prob_bank, class_idx_map, train_expr,
                bank_size=args.bank_size, tau=args.tau_base)

        elif variant == "E":
            expr_in = build_expr_ct_ws_adapt(
                sim_mat, test_prob_bank, class_idx_map, train_expr,
                bank_size=args.bank_size, tau_base=args.tau_base,
                tau_entropy_alpha=args.tau_entropy_alpha)

        else:
            print(f"[SKIP] Unknown variant {variant}")
            continue

        print(f"  expr_in shape={expr_in.shape}  time={time.time()-t_start:.1f}s")

        # ── run model decoder with this expression ────────────────────────────
        t_inf = time.time()
        mu_pred = run_inference(
            model, test_map, test_coords, expr_in, image_tf, device,
            args.batch_size, args.num_workers, amp_dtype,
            test_edge_idx, test_edge_w, test_spot_feat,
        )
        print(f"  inference done  time={time.time()-t_inf:.1f}s")

        # ── compute metrics against ground-truth test expression ──────────────
        m, pcc_arr = eval_metrics(mu_pred, test_expr)
        all_results[variant]  = {"name": VARIANT_NAMES[variant], **m}
        all_pcc_gene[variant] = pcc_arr

        print(f"  PCC mean={m['pcc_mean']:.4f}  PCC median={m['pcc_median']:.4f}  "
              f"MSE={m['mse']:.6f}  MAE={m['mae']:.6f}")

        # ── per-cell-type PCC ─────────────────────────────────────────────────
        if test_labels is not None and celltype_names:
            per_ct_data[variant] = {}
            for ci, cname in enumerate(celltype_names):
                sel = np.where(test_labels == ci)[0]
                if sel.size == 0:
                    continue
                pcc_ci = pcc_per_gene(mu_pred[sel], test_expr[sel])
                per_ct_data[variant][cname] = round(float(np.nanmean(pcc_ci)), 4)

        # ── save per-gene PCC ─────────────────────────────────────────────────
        np.save(os.path.join(args.out_dir, f"per_gene_pcc_{variant}.npy"), pcc_arr)

    # ── save summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[Summary]")
    summary_rows = []
    for v in args.variants:
        if v not in all_results:
            continue
        r = all_results[v]
        print(f"  {v} ({r['name']:30s})  PCC={r['pcc_mean']:.4f}  "
              f"MSE={r['mse']:.6f}  MAE={r['mae']:.6f}")
        summary_rows.append({"variant": v, **r})

    with open(os.path.join(args.out_dir, "ablation_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, "ablation_summary.csv"), index=False)

    # ── per-class PCC table ───────────────────────────────────────────────────
    if per_ct_data:
        ct_rows = []
        for cname in celltype_names:
            row = {"class_name": cname}
            for v in args.variants:
                row[f"pcc_{v}_{VARIANT_NAMES.get(v,'?').replace(' ','_')}"] = \
                    per_ct_data.get(v, {}).get(cname, float("nan"))
            ct_rows.append(row)
        pd.DataFrame(ct_rows).to_csv(
            os.path.join(args.out_dir, "per_celltype_pcc.csv"), index=False)

    print(f"\n[DONE] all results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
