#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_retrieval_quality.py
--------------------------
Evaluate memory-bank retrieval quality.

For each test patch query, the model ranks ALL training patches by cosine
similarity (in the projected image embedding space).  Cell type is used as
a proxy for ground-truth relevance (same cell type = relevant).

Metrics reported
  • Recall@K (K = 1, 5, 10, 20, 50)   — fraction of queries with ≥1 same-celltype
                                          training sample in top-K
  • Mean Average Precision (mAP) @ 20  — area under precision-recall curve
  • Mean Rank of first relevant sample  — lower is better
  • Median Rank of first relevant sample
  • nDCG@20                             — normalized discounted cumulative gain
  • Mean cosine similarity of top-1 retrieved sample (regardless of cell type)
  • Expression PCC (per-gene Pearson) between
      - GT test expression and top-1 retrieved training expression
      - GT test expression and cosine-weighted sum of retrieved training expression
  • Per-cell-type Recall@1 and mean PCC

Saved outputs
  <out_dir>/retrieval_metrics.json     — all scalar metrics
  <out_dir>/per_celltype_retrieval.csv — per-class Recall@K + expression PCC

Example:
  python eval_retrieval_quality.py \\
    --train_h5ad /data/.../human_skin1.h5ad \\
    --train_csv  /data/.../train_patches.csv \\
    --test_h5ad  /data/.../human_skin1_test.h5ad \\
    --test_csv   /data/.../test_patches.csv \\
    --ft_model_ckpt /data/.../best/model.pt \\
    --celltype_meta_pt /data/.../best/celltype_meta.pt \\
    --celltype_col cell_type \\
    --enc_name uni_v1 \\
    --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \\
    --img_tokens 10 --spot_tokens 10 --gene_tokens 10 \\
    --graph_type delaunay --k 12 \\
    --top_k 1 5 10 20 50 \\
    --bank_size_for_weighted 500 \\
    --out_dir ./ablation/retrieval_quality
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
)


# ============================================================
# Minimal image-only dataset (no dependency on make_img_only_dataloader)
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
        if self.tf is not None:
            img = self.tf(img)
        else:
            img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        return img, idx


# ============================================================
# Model
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


# ============================================================
# Helpers
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


# ============================================================
# Embed all patches in a map_df using the model's image encoder
# ============================================================
@torch.no_grad()
def embed_patches(img_encoder, map_df, image_tf, device, batch_size, num_workers, amp_dtype):
    ds     = _ImgIdxDataset(map_df, image_transform=image_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    img_encoder.eval()
    all_emb = np.zeros((len(ds), img_encoder.get_output_dim()), dtype=np.float32)
    use_amp = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
    for imgs, idxs in tqdm(loader, desc="embed", leave=False):
        imgs = imgs.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            emb = img_encoder(imgs).float()
        emb = F.normalize(emb, dim=1)
        all_emb[idxs.numpy()] = emb.detach().cpu().numpy()
    return all_emb  # (N, D)


@torch.no_grad()
def embed_patches_proj(proj_img_layer, img_encoder, map_df, image_tf,
                        device, batch_size, num_workers, amp_dtype):
    """Embed patches in the PROJECTED space (after proj_img linear layer)."""
    ds     = _ImgIdxDataset(map_df, image_transform=image_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    img_encoder.eval(); proj_img_layer.eval()
    dim = proj_img_layer.out_features
    all_emb = np.zeros((len(ds), dim), dtype=np.float32)
    use_amp = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
    for imgs, idxs in tqdm(loader, desc="embed(proj)", leave=False):
        imgs = imgs.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            emb = proj_img_layer(img_encoder(imgs)).float()
        emb = F.normalize(emb, dim=1)
        all_emb[idxs.numpy()] = emb.detach().cpu().numpy()
    return all_emb  # (N, D)


# ============================================================
# Retrieval evaluation metrics
# ============================================================
def recall_at_k(sim_matrix: np.ndarray, train_labels: np.ndarray,
                test_labels: np.ndarray, k_list: list) -> dict:
    """
    sim_matrix : (N_test, N_train)  cosine similarity
    train_labels : (N_train,)  integer cell-type index
    test_labels  : (N_test,)   integer cell-type index

    Returns Recall@K for each K: fraction of test queries with ≥1 same-label
    training sample in their top-K.
    """
    n_test = sim_matrix.shape[0]
    results = {}
    for k in k_list:
        topk_idx = np.argpartition(-sim_matrix, kth=min(k, sim_matrix.shape[1]-1), axis=1)[:, :k]
        hits = 0
        for qi in range(n_test):
            if test_labels[qi] < 0:
                continue
            retrieved_labels = train_labels[topk_idx[qi]]
            if (retrieved_labels == test_labels[qi]).any():
                hits += 1
        n_valid = int((test_labels >= 0).sum())
        results[f"R@{k}"] = round(hits / max(n_valid, 1), 4)
    return results


def mean_average_precision(sim_matrix: np.ndarray, train_labels: np.ndarray,
                            test_labels: np.ndarray, max_k: int = 20) -> float:
    """mAP@max_k."""
    n_test = sim_matrix.shape[0]
    ap_list = []
    for qi in range(n_test):
        if test_labels[qi] < 0:
            continue
        order = np.argsort(-sim_matrix[qi])[:max_k]
        relevant = (train_labels[order] == test_labels[qi]).astype(float)
        hits = 0.0; prec_sum = 0.0
        for rank_k, rel in enumerate(relevant, start=1):
            if rel > 0:
                hits += 1
                prec_sum += hits / rank_k
        n_rel = int((train_labels == test_labels[qi]).sum())
        if n_rel > 0:
            ap_list.append(prec_sum / min(n_rel, max_k))
    return float(np.mean(ap_list)) if ap_list else 0.0


def mean_rank_first_relevant(sim_matrix: np.ndarray, train_labels: np.ndarray,
                              test_labels: np.ndarray):
    """Mean/Median rank of the first same-label training sample (1-based)."""
    ranks = []
    for qi in range(sim_matrix.shape[0]):
        if test_labels[qi] < 0:
            continue
        order = np.argsort(-sim_matrix[qi])
        for rank_0, ti in enumerate(order):
            if train_labels[ti] == test_labels[qi]:
                ranks.append(rank_0 + 1)
                break
    return (float(np.mean(ranks)), float(np.median(ranks))) if ranks else (float("nan"), float("nan"))


def ndcg_at_k(sim_matrix: np.ndarray, train_labels: np.ndarray,
              test_labels: np.ndarray, k: int = 20) -> float:
    """nDCG@k  (binary relevance: same cell type = 1)."""
    def _dcg(rel, k_):
        rel = rel[:k_]
        gains  = rel / np.log2(np.arange(2, len(rel) + 2))
        return float(gains.sum())

    ndcg_vals = []
    for qi in range(sim_matrix.shape[0]):
        if test_labels[qi] < 0:
            continue
        order   = np.argsort(-sim_matrix[qi])[:k]
        rel     = (train_labels[order] == test_labels[qi]).astype(float)
        n_rel   = int((train_labels == test_labels[qi]).sum())
        ideal_rel = np.ones(min(n_rel, k), dtype=float)
        idcg = _dcg(ideal_rel, k)
        if idcg == 0:
            continue
        ndcg_vals.append(_dcg(rel, k) / idcg)
    return float(np.mean(ndcg_vals)) if ndcg_vals else float("nan")


def expression_pcc(pred_expr: np.ndarray, true_expr: np.ndarray) -> tuple:
    """Mean/median gene-level PCC across all test spots."""
    Yc = true_expr - true_expr.mean(0, keepdims=True)
    Pc = pred_expr - pred_expr.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0)) + 1e-8
    pcc = num / den
    return float(np.nanmean(pcc)), float(np.nanmedian(pcc))


def top1_retrieved_expr(sim_matrix: np.ndarray, train_expr: np.ndarray) -> np.ndarray:
    """Top-1 cosine-retrieved expression for each test query."""
    best_idx = sim_matrix.argmax(axis=1)
    return train_expr[best_idx]


def weighted_sum_retrieved_expr(sim_row: np.ndarray, train_expr: np.ndarray,
                                 bank_size: int, tau: float = 0.07) -> np.ndarray:
    """Cosine-weighted sum of top-bank_size training expressions."""
    if bank_size < sim_row.shape[0]:
        topk_idx = np.argpartition(-sim_row, kth=bank_size)[:bank_size]
    else:
        topk_idx = np.arange(len(sim_row))
    sims = sim_row[topk_idx]
    w = np.exp(sims / tau)
    w = w / (w.sum() + 1e-9)
    return (w[:, None] * train_expr[topk_idx]).sum(0)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate memory-bank retrieval quality")
    ap.add_argument("--train_h5ad",  required=True)
    ap.add_argument("--train_csv",   required=True)
    ap.add_argument("--train_root",  default=None)
    ap.add_argument("--test_h5ad",   required=True)
    ap.add_argument("--test_csv",    required=True)
    ap.add_argument("--test_root",   default=None)
    ap.add_argument("--id_col",   default="cell_id")
    ap.add_argument("--path_col", default="patch_path")
    ap.add_argument("--celltype_col",   default=None)
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
    # eval params
    ap.add_argument("--top_k",    type=int, nargs="+", default=[1, 5, 10, 20, 50],
                    help="K values for Recall@K")
    ap.add_argument("--map_k",    type=int, default=20, help="K for mAP computation")
    ap.add_argument("--ndcg_k",   type=int, default=20, help="K for nDCG computation")
    ap.add_argument("--bank_size_for_weighted", type=int, default=500,
                    help="Bank size for cosine-weighted-sum expression")
    ap.add_argument("--tau",      type=float, default=0.07,
                    help="Temperature for weighted-sum softmax")
    ap.add_argument("--sim_chunk", type=int, default=512,
                    help="Chunk size for computing similarity matrix (CPU memory control)")
    ap.add_argument("--embedding_space", default="projected",
                    choices=["raw","projected"],
                    help="'raw'=foundation encoder output, 'projected'=after proj_img linear")
    # run
    ap.add_argument("--batch_size",  type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--amp",         action="store_true")
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = select_device(args.device)
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16) if (args.amp and device.type == "cuda") else torch.float32

    # ── load data ────────────────────────────────────────────────────────────
    train_adata = load_h5ad_with_preproc(args.train_h5ad, use_hvg=False, n_top_genes=None,
                                         x_key="x_centroid", y_key="y_centroid")
    test_adata  = load_h5ad_with_preproc(args.test_h5ad,  use_hvg=False, n_top_genes=None,
                                         x_key="x_centroid", y_key="y_centroid")
    # align genes
    common      = train_adata.var_names.intersection(test_adata.var_names)
    gene_list   = [g for g in train_adata.var_names if g in common]
    train_adata = train_adata[:, gene_list].copy()
    test_adata  = test_adata[:,  gene_list].copy()

    for split, h5ad, csv, root in [
        ("train", args.train_h5ad, args.train_csv, args.train_root),
        ("test",  args.test_h5ad,  args.test_csv,  args.test_root),
    ]:
        pass  # loaded above

    def _prep_map(adata, csv_path, root, id_col, path_col):
        map_df = load_patch_mapping_csv_xenium(csv_path, root_dir=root,
                                               id_col=id_col, path_col=path_col,
                                               drop_missing=True)
        map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
        map_df, coords = reorder_and_prune_by_obs(adata, map_df, id_col="spot_id",
                                                   coord_key="spatial")
        return map_df, coords

    train_map, _   = _prep_map(train_adata, args.train_csv, args.train_root, args.id_col, args.path_col)
    test_map,  test_coords = _prep_map(test_adata, args.test_csv, args.test_root, args.id_col, args.path_col)

    if test_map.shape[0] != test_adata.n_obs:
        test_ids = test_map["spot_id"].astype(str).tolist()
        test_adata = test_adata[test_ids].copy()

    train_expr = _adata_x(train_adata)
    test_expr  = _adata_x(test_adata)

    # ── celltype labels ───────────────────────────────────────────────────────
    ct_meta_path = args.celltype_meta_pt
    if ct_meta_path is None and args.ft_model_ckpt:
        cand = os.path.join(os.path.dirname(args.ft_model_ckpt), "celltype_meta.pt")
        if os.path.isfile(cand):
            ct_meta_path = cand
    ct_meta       = torch.load(ct_meta_path, map_location="cpu") if ct_meta_path and os.path.isfile(ct_meta_path) else {}
    celltype_names = [str(x) for x in ct_meta.get("celltype_names", [])]

    def _get_ct_labels(adata, map_df, ct_col_hint):
        if not ct_col_hint and not celltype_names:
            return None
        cols_lower = {str(c).lower(): c for c in adata.obs.columns}
        hints = [ct_col_hint] if ct_col_hint else []
        hints += ["cell_type","celltype","celltypes","annotation","cluster","leiden"]
        ct_col = None
        for h in hints:
            if h and h.lower() in cols_lower:
                ct_col = cols_lower[h.lower()]; break
        if ct_col is None:
            return None
        spot_ids = map_df["spot_id"].astype(str).tolist()
        gt_str   = adata.obs.loc[spot_ids, ct_col].astype(str).to_numpy()
        if not celltype_names:
            return None  # need celltype_names to map to indices
        name2idx = {n: i for i, n in enumerate(celltype_names)}
        return np.array([name2idx.get(n, -1) for n in gt_str], dtype=np.int64)

    train_labels = _get_ct_labels(train_adata, train_map, args.celltype_col)
    test_labels  = _get_ct_labels(test_adata,  test_map,  args.celltype_col)

    # ── model ─────────────────────────────────────────────────────────────────
    ft_sd  = _load_sd(args.ft_model_ckpt)
    ckpt_num_ct, ckpt_ct_hidden = _infer_cls_head_config(ft_sd)
    meta_num_ct = int(ct_meta.get("num_celltypes", 0)) if ct_meta else 0
    num_celltypes = max(meta_num_ct, ckpt_num_ct)

    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation    = FoundationCls(weights_path=None)
    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)
    img_dim  = foundation.get_output_dim()
    gene_dim = len(gene_list)

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

    # ── embed all patches ─────────────────────────────────────────────────────
    t0 = time.time()
    if args.embedding_space == "projected":
        train_emb = embed_patches_proj(model.proj_img, model.img_encoder, train_map, image_tf,
                                        device, args.batch_size, args.num_workers, amp_dtype)
        test_emb  = embed_patches_proj(model.proj_img, model.img_encoder, test_map,  image_tf,
                                        device, args.batch_size, args.num_workers, amp_dtype)
    else:
        train_emb = embed_patches(model.img_encoder, train_map, image_tf,
                                   device, args.batch_size, args.num_workers, amp_dtype)
        test_emb  = embed_patches(model.img_encoder, test_map,  image_tf,
                                   device, args.batch_size, args.num_workers, amp_dtype)
    print(f"[Embed] done in {time.time()-t0:.1f}s  "
          f"train={train_emb.shape}  test={test_emb.shape}")

    # ── chunked GPU evaluation — never materialise the full sim matrix ────────
    n_test  = test_emb.shape[0]
    n_train = train_emb.shape[0]
    max_k_needed = max(max(args.top_k), args.map_k, args.ndcg_k, args.bank_size_for_weighted)
    max_k_needed = min(max_k_needed, n_train)

    # Accumulators (keep only top-K per query)
    topK_idx  = np.zeros((n_test, max_k_needed), dtype=np.int32)
    topK_sim  = np.zeros((n_test, max_k_needed), dtype=np.float32)
    top1_sims = np.zeros(n_test, dtype=np.float32)

    train_t = torch.from_numpy(train_emb).to(device)
    test_t  = torch.from_numpy(test_emb)

    print(f"[Sim] computing top-{max_k_needed} similarities for {n_test} queries "
          f"(chunk={args.sim_chunk}) ...")
    for s in tqdm(range(0, n_test, args.sim_chunk), desc="similarity (GPU)"):
        e = min(n_test, s + args.sim_chunk)
        q_t = test_t[s:e].to(device)
        with torch.no_grad():
            sim_chunk = (q_t @ train_t.t())          # (chunk, N_train) on GPU
        top1_sims[s:e] = sim_chunk.max(dim=1).values.cpu().numpy()
        vals, idx = torch.topk(sim_chunk, k=max_k_needed, dim=1, largest=True, sorted=True)
        topK_sim[s:e] = vals.cpu().numpy()
        topK_idx[s:e] = idx.cpu().numpy()

    top1_sim = float(top1_sims.mean())

    # ── retrieval IR metrics (require cell-type labels) ───────────────────────
    ir_metrics = {}
    per_ct_rows = []
    if train_labels is not None and test_labels is not None:
        valid_test = test_labels >= 0
        print(f"[CT] valid test queries: {valid_test.sum()} / {n_test}")

        # Recall@K — vectorised over stored top-K
        recall = {}
        for k in args.top_k:
            k = min(k, max_k_needed)
            hits = 0
            n_valid = int((test_labels >= 0).sum())
            for qi in range(n_test):
                if test_labels[qi] < 0:
                    continue
                if (train_labels[topK_idx[qi, :k]] == test_labels[qi]).any():
                    hits += 1
            recall[f"R@{k}"] = round(hits / max(n_valid, 1), 4)

        # mAP@map_k
        k_map = min(args.map_k, max_k_needed)
        ap_list = []
        for qi in range(n_test):
            if test_labels[qi] < 0:
                continue
            rel = (train_labels[topK_idx[qi, :k_map]] == test_labels[qi]).astype(float)
            hits_ = 0.0; prec_sum = 0.0
            for rank_k, r in enumerate(rel, start=1):
                if r > 0:
                    hits_ += 1
                    prec_sum += hits_ / rank_k
            n_rel = int((train_labels == test_labels[qi]).sum())
            if n_rel > 0:
                ap_list.append(prec_sum / min(n_rel, k_map))
        mAP = float(np.mean(ap_list)) if ap_list else 0.0

        # nDCG@ndcg_k
        k_ndcg = min(args.ndcg_k, max_k_needed)
        def _dcg(rel_arr, k_):
            rel_arr = rel_arr[:k_]
            return float((rel_arr / np.log2(np.arange(2, len(rel_arr) + 2))).sum())
        ndcg_vals = []
        for qi in range(n_test):
            if test_labels[qi] < 0:
                continue
            rel = (train_labels[topK_idx[qi, :k_ndcg]] == test_labels[qi]).astype(float)
            n_rel = int((train_labels == test_labels[qi]).sum())
            ideal_rel = np.ones(min(n_rel, k_ndcg), dtype=float)
            idcg = _dcg(ideal_rel, k_ndcg)
            if idcg > 0:
                ndcg_vals.append(_dcg(rel, k_ndcg) / idcg)
        ndcg = float(np.mean(ndcg_vals)) if ndcg_vals else float("nan")

        # Mean rank — use top-K as upper bound; rank is position+1 within stored topK
        ranks = []
        for qi in range(n_test):
            if test_labels[qi] < 0:
                continue
            lab_row = train_labels[topK_idx[qi]]
            pos = np.where(lab_row == test_labels[qi])[0]
            if pos.size > 0:
                ranks.append(int(pos[0]) + 1)
            else:
                ranks.append(max_k_needed + 1)   # outside top-K
        mr  = float(np.mean(ranks))  if ranks else float("nan")
        mdr = float(np.median(ranks)) if ranks else float("nan")

        ir_metrics = {
            **recall,
            f"mAP@{args.map_k}": round(mAP, 4),
            f"nDCG@{args.ndcg_k}": round(ndcg, 4),
            "mean_rank":   round(mr,  2),
            "median_rank": round(mdr, 2),
        }
        print(f"[IR]  " + "  ".join(f"{k}={v:.4f}" for k, v in recall.items()))
        print(f"[IR]  mAP@{args.map_k}={mAP:.4f}  nDCG@{args.ndcg_k}={ndcg:.4f}  "
              f"MeanRank={mr:.1f}  MedianRank={mdr:.1f}")

        # per-cell-type Recall@1 + expression PCC
        for ci, cname in enumerate(celltype_names):
            test_sel = np.where(test_labels == ci)[0]
            if test_sel.size == 0:
                continue
            train_sel = np.where(train_labels == ci)[0]
            hits = int(sum(train_labels[topK_idx[qi, 0]] == ci for qi in test_sel))
            rec1 = hits / len(test_sel)
            top1_expr_sel = train_expr[topK_idx[test_sel, 0]]
            pcc_m, pcc_med = expression_pcc(top1_expr_sel, test_expr[test_sel])
            per_ct_rows.append({
                "class_idx": ci, "class_name": cname,
                "n_test": len(test_sel), "n_train": len(train_sel),
                "recall@1": round(rec1, 4),
                "pcc_top1_mean": round(pcc_m, 4),
                "pcc_top1_median": round(pcc_med, 4),
            })

    # ── expression quality metrics ────────────────────────────────────────────
    t1_expr = train_expr[topK_idx[:, 0]]
    pcc_top1_m, pcc_top1_med = expression_pcc(t1_expr, test_expr)

    # weighted-sum using stored top-K sims (vectorised, no Python loop)
    k_ws = min(args.bank_size_for_weighted, max_k_needed)
    ws_sims = topK_sim[:, :k_ws]                          # (N_test, k_ws)
    ws_w = np.exp(ws_sims / args.tau)
    ws_w = ws_w / (ws_w.sum(axis=1, keepdims=True) + 1e-9)  # (N_test, k_ws)
    ws_idx = topK_idx[:, :k_ws]                           # (N_test, k_ws)
    # batched gather to avoid building N_test × k_ws × G tensor all at once
    chunk_ws = 2048
    ws_pred = np.zeros_like(test_expr)
    for s in tqdm(range(0, n_test, chunk_ws), desc="weighted-sum expr", leave=False):
        e = min(n_test, s + chunk_ws)
        ws_pred[s:e] = (ws_w[s:e, :, None] * train_expr[ws_idx[s:e]]).sum(axis=1)
    pcc_ws_m, pcc_ws_med = expression_pcc(ws_pred, test_expr)

    expr_metrics = {
        "top1_pcc_mean":         round(pcc_top1_m,   4),
        "top1_pcc_median":       round(pcc_top1_med, 4),
        "weighted_sum_pcc_mean": round(pcc_ws_m,     4),
        "weighted_sum_pcc_median": round(pcc_ws_med, 4),
        "mean_top1_cosine_sim":  round(float(top1_sim), 4),
    }
    print(f"[Expr] Top-1 PCC mean={pcc_top1_m:.4f}  "
          f"Weighted-sum PCC mean={pcc_ws_m:.4f}")

    # ── save results ─────────────────────────────────────────────────────────
    summary = {
        "embedding_space": args.embedding_space,
        "n_train": n_train, "n_test": n_test,
        "top_k": args.top_k,
        "bank_size_for_weighted": args.bank_size_for_weighted,
        "tau": args.tau,
        "ir_metrics":   ir_metrics,
        "expr_metrics": expr_metrics,
    }
    with open(os.path.join(args.out_dir, "retrieval_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if per_ct_rows:
        pd.DataFrame(per_ct_rows).to_csv(
            os.path.join(args.out_dir, "per_celltype_retrieval.csv"), index=False)

    print(f"[DONE] results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
