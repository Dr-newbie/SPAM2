#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
celltype_inference.py

Purpose
- Evaluate celltype prediction quality from image-driven inference model.
- Save confusion matrices and classification reports.
- Build expression embeddings using the gene encoder weights used in inference.
- Run label transfer (train -> eval) on expression embeddings.
- Visualize UMAP and spatial label assignment quality.

Example
CUDA_VISIBLE_DEVICES=0 python Code_final/code/utils/celltype_inference.py \
  --h5ad /data/project/banana9903_xenium/xenium_rep1_io/data/merged_with_celltype_raw.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --id_col cell_id --path_col patch_path \
  --train_h5ad /data/project/banana9903_xenium/xenium_rep1_io/data/merged_with_celltype_raw.h5ad \
  --train_csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --enc_name uni_v1 \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \
  --graph_type delaunay --k 12 \
  --ft_model_ckpt /data/project/banana9903_xenium/ckpts_cross_human_breast_cancer_xattn_delaunay_10tokens_with_celltype/best/model.pt \
  --celltype_pred_expr_source eval \
  --batch_size 256 --num_workers 4 \
  --out_dir /data/project/banana9903_xenium/inference/celltype_eval_breast
"""

import os
import sys
import json
import argparse
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.modules.pop("code", None)

from code.models.Foundations import inf_encoder_factory
from code.models.GCN_update import SpatialGCN
from code.models.gene_encoder import GeneEncoder
from code.utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_only_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)
from code.utils.image_representation_gathering import (
    _expression_matrix_for_spots,
    ImageMapDataset,
    gather_image_representations,
)
from code.utils.run_inference_xattn_withcelltype import (
    CrossAttnZINBModel,
    _load_sd,
    load_pretrained_blocks,
    load_xattn_into_fusion,
    select_device,
    pick_amp_dtype,
    align_genes_for_inference,
    build_spatial_graph,
    _mean_gene_vector,
    _resolve_obs_col,
    _load_celltype_meta,
    _infer_cls_head_config,
    _predict_celltype_probs_from_base_expr,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_celltype_col(adata: ad.AnnData, preferred: str | None, fallback: str | None = None) -> str:
    for cand in [preferred, fallback]:
        col = _resolve_obs_col(adata, cand)
        if col is not None:
            return col
    candidates = [
        "cell_type", "celltype", "celltypes", "cell_type_pred", "annotation",
        "annot", "cluster", "clusters", "leiden", "seurat_clusters", "Cluster"
    ]
    cols = list(adata.obs.columns)
    low = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"Cell type column not found. Available columns: {cols}")


@torch.no_grad()
def _encode_gene_embeddings(
    gene_encoder: torch.nn.Module,
    expr_np: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    gene_encoder.eval()
    gene_encoder.to(device)
    n = int(expr_np.shape[0])
    out_dim = int(getattr(gene_encoder, "out_dim", expr_np.shape[1]))
    out = np.zeros((n, out_dim), dtype=np.float32)
    for s in tqdm(range(0, n, batch_size), desc="encode gene", leave=False):
        e = min(n, s + batch_size)
        x = torch.from_numpy(expr_np[s:e]).to(device=device, dtype=torch.float32)
        z = gene_encoder(x)
        if isinstance(z, tuple):
            z = z[-1]
        z = F.normalize(z.float(), dim=1)
        out[s:e] = z.detach().cpu().numpy()
    return out


def _knn_label_transfer_cosine(
    ref_emb: np.ndarray,
    ref_y_idx: np.ndarray,
    qry_emb: np.ndarray,
    k: int = 5,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ref_emb.shape[0] == 0:
        raise ValueError("Empty reference embeddings.")
    k = max(1, min(int(k), int(ref_emb.shape[0])))

    ref = ref_emb.astype(np.float32, copy=False)
    qry = qry_emb.astype(np.float32, copy=False)
    ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    qry = qry / (np.linalg.norm(qry, axis=1, keepdims=True) + 1e-8)

    n_q = qry.shape[0]
    n_cls = int(ref_y_idx.max()) + 1
    pred_idx = np.zeros(n_q, dtype=np.int64)
    pred_conf = np.zeros(n_q, dtype=np.float32)
    top1_sim = np.zeros(n_q, dtype=np.float32)

    ref_t = torch.from_numpy(ref)
    for s in tqdm(range(0, n_q, chunk_size), desc="label transfer", leave=False):
        e = min(n_q, s + chunk_size)
        qry_t = torch.from_numpy(qry[s:e])
        sims = qry_t @ ref_t.t()  # (B, Nref)

        topk_vals, topk_idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        topk_idx_np = topk_idx.numpy()
        topk_vals_np = topk_vals.numpy()
        topk_lab = ref_y_idx[topk_idx_np]

        bsz = e - s
        vote = np.zeros((bsz, n_cls), dtype=np.int32)
        rows = np.arange(bsz)
        for j in range(k):
            np.add.at(vote, (rows, topk_lab[:, j]), 1)

        cls = np.argmax(vote, axis=1).astype(np.int64)
        conf = vote[rows, cls].astype(np.float32) / float(k)

        pred_idx[s:e] = cls
        pred_conf[s:e] = conf
        top1_sim[s:e] = topk_vals_np[:, 0].astype(np.float32)

    return pred_idx, pred_conf, top1_sim


def _build_palette(labels: list[str]) -> dict[str, Any]:
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    palette: dict[str, Any] = {}
    for i, lab in enumerate(uniq):
        palette[lab] = cmap(i % 20)
    if "__NA__" in uniq:
        palette["__NA__"] = (0.6, 0.6, 0.6, 1.0)
    return palette


def _plot_categorical_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    palette: dict[str, Any],
    title: str,
    point_size: float = 2.0,
    alpha: float = 0.8,
    invert_y: bool = False,
):
    labels = labels.astype(str)
    uniq = sorted(set(labels.tolist()))
    for lab in uniq:
        m = labels == lab
        ax.scatter(
            x[m], y[m], s=point_size, alpha=alpha, linewidths=0,
            c=[palette.get(lab, (0.2, 0.2, 0.2, 1.0))], label=str(lab),
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()


def _save_confusion_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_dir: str,
    prefix: str,
):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(out_dir, f"{prefix}_confusion_counts.csv"))

    cm_norm = cm.astype(np.float32)
    row_sum = cm_norm.sum(axis=1, keepdims=True) + 1e-8
    cm_norm = cm_norm / row_sum
    cmn_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cmn_df.to_csv(os.path.join(out_dir, f"{prefix}_confusion_row_norm.csv"))

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(os.path.join(out_dir, f"{prefix}_classification_report.csv"))

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "balanced_accuracy": bacc}, f, indent=2)

    n_cls = max(2, len(labels))
    fig_w = max(8, min(30, n_cls * 0.42))
    fig_h = max(6, min(24, n_cls * 0.36))

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm_df, cmap="Blues", annot=(n_cls <= 25), fmt="d", cbar=True,
        linewidths=0.2, linecolor="white"
    )
    plt.title(f"{prefix} confusion (count)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_counts.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        cmn_df, cmap="magma", annot=(n_cls <= 20), fmt=".2f", cbar=True,
        linewidths=0.2, linecolor="white", vmin=0.0, vmax=1.0
    )
    plt.title(f"{prefix} confusion (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_row_norm.png"), dpi=300)
    plt.close()

    return {"accuracy": acc, "balanced_accuracy": bacc}


def _run_umap(
    X: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    metric: str = "cosine",
    pca_n_comps: int = 50,
    seed: int = 0,
) -> np.ndarray:
    adata = ad.AnnData(X=np.asarray(X, dtype=np.float32))
    if X.shape[1] > pca_n_comps:
        sc.pp.pca(adata, n_comps=pca_n_comps, svd_solver="arpack", random_state=seed)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric, use_rep="X_pca")
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric)
    sc.tl.umap(adata, min_dist=min_dist, random_state=seed)
    return np.asarray(adata.obsm["X_umap"], dtype=np.float32)


def _subsample_idx(n: int, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--id_col", default="cell_id")
    ap.add_argument("--path_col", default="patch_path")

    ap.add_argument("--train_h5ad", required=True)
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--train_root", default=None)
    ap.add_argument("--train_id_col", default=None)
    ap.add_argument("--train_path_col", default=None)

    ap.add_argument("--enc_name", default="uni_v1",
                    choices=["uni_v1", "virchow", "virchow2", "gigapath", "hoptimus0", "plip", "phikon", "conch_v1"])
    ap.add_argument("--enc_weights_path", default=None)

    ap.add_argument("--proj_dim_gene", type=int, default=256)
    ap.add_argument("--proj_dim_spot", type=int, default=256)
    ap.add_argument("--fuse_dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--img_tokens", type=int, default=1)
    ap.add_argument("--spot_tokens", type=int, default=1)
    ap.add_argument("--gene_tokens", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--merge", default="gated-sum", choices=["gated-sum", "concat-proj"])

    ap.add_argument("--graph_type", default="knn", choices=["knn", "delaunay"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--ft_model_ckpt", required=True)
    ap.add_argument("--pt_img_backbone", default=None)
    ap.add_argument("--pt_ig", default=None)
    ap.add_argument("--pt_is", default=None)
    ap.add_argument("--pt_xattn", default=None)

    ap.add_argument("--celltype_meta_pt", default=None)
    ap.add_argument("--celltype_col", default=None, help="train celltype column override")
    ap.add_argument("--eval_celltype_col", default=None, help="eval celltype column override")
    ap.add_argument("--celltype_pred_expr_source", default="eval", choices=["mean", "eval"])

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gene_batch_size", type=int, default=2048)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--sim_chunk", type=int, default=1024)

    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.3)
    ap.add_argument("--umap_metric", default="cosine")
    ap.add_argument("--umap_pca_comps", type=int, default=50)
    ap.add_argument("--umap_max_points", type=int, default=50000)
    ap.add_argument("--point_size", type=float, default=2.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="celltype_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = select_device(args.device)
    amp_dtype = pick_amp_dtype(args.amp)
    print(f"[Device] {device}, AMP={args.amp}, dtype={amp_dtype}")

    # 1) data
    eval_adata = load_h5ad_with_preproc(
        args.h5ad, use_hvg=False, n_top_genes=None, x_key="x_centroid", y_key="y_centroid"
    )
    train_adata = load_h5ad_with_preproc(
        args.train_h5ad, use_hvg=False, n_top_genes=None, x_key="x_centroid", y_key="y_centroid"
    )
    train_adata, eval_adata = align_genes_for_inference(train_adata, eval_adata)

    map_df = load_patch_mapping_csv_xenium(
        args.csv, root_dir=args.root, id_col=args.id_col, path_col=args.path_col, drop_missing=True
    )
    map_df = align_map_ids_to_obs(eval_adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(eval_adata, map_df, id_col="spot_id", coord_key="spatial")
    if map_df_aligned.shape[0] != eval_adata.n_obs:
        eval_keep = map_df_aligned["spot_id"].astype(str).tolist()
        eval_adata = eval_adata[eval_keep].copy()

    eval_spot_ids = map_df_aligned["spot_id"].astype(str).tolist()
    print(f"[Data] eval n={len(eval_spot_ids)}, genes={eval_adata.n_vars}")

    train_spot_ids: list[str]
    train_map_df_aligned = None
    if args.train_csv:
        train_id_col = args.train_id_col or args.id_col
        train_path_col = args.train_path_col or args.path_col
        train_map_df = load_patch_mapping_csv_xenium(
            args.train_csv,
            root_dir=args.train_root or args.root,
            id_col=train_id_col,
            path_col=train_path_col,
            drop_missing=True,
        )
        train_map_df = align_map_ids_to_obs(train_adata, train_map_df, id_col="spot_id")
        train_map_df, _ = reorder_and_prune_by_obs(train_adata, train_map_df, id_col="spot_id", coord_key="spatial")
        train_map_df_aligned = train_map_df
        train_spot_ids = train_map_df["spot_id"].astype(str).tolist()
    else:
        train_spot_ids = train_adata.obs_names.astype(str).tolist()
    print(f"[Data] train ref n={len(train_spot_ids)}")

    # 2) model
    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation = FoundationCls(weights_path=args.enc_weights_path)
    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)

    img_loader = make_img_only_dataloader(
        map_df=map_df_aligned,
        batch_size=max(1, args.batch_size),
        num_workers=max(1, args.num_workers),
        image_transform=image_tf,
        shuffle=False,
    )

    edge_index, edge_weight = build_spatial_graph(coords_np, k=args.k, graph_type=args.graph_type)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None
    spot_feat_in = torch.from_numpy(coords_np).float().to(device)

    img_dim = foundation.get_output_dim()
    gene_dim = eval_adata.n_vars
    gene_encoder = GeneEncoder(
        input_dim=gene_dim, model_size_omic="small", proj_dim=args.proj_dim_gene, normalize=False
    )
    gcn = SpatialGCN(in_dim=spot_feat_in.size(1), hidden_dim=256, out_dim=args.proj_dim_spot, dropout=0.2)

    load_pretrained_blocks(
        foundation, gene_encoder, gcn,
        pt_img_backbone=args.pt_img_backbone, pt_ig=args.pt_ig, pt_is=args.pt_is
    )

    ft_sd = _load_sd(args.ft_model_ckpt)
    ckpt_num_ct, ckpt_ct_hidden = _infer_cls_head_config(ft_sd)

    celltype_meta_path = args.celltype_meta_pt
    if celltype_meta_path is None and args.ft_model_ckpt:
        cand = os.path.join(os.path.dirname(args.ft_model_ckpt), "celltype_meta.pt")
        if os.path.isfile(cand):
            celltype_meta_path = cand
    ct_meta = _load_celltype_meta(celltype_meta_path)
    meta_num_ct = int(ct_meta.get("num_celltypes", 0)) if isinstance(ct_meta, dict) else 0
    num_celltypes = max(meta_num_ct, ckpt_num_ct)
    if num_celltypes <= 0:
        raise ValueError("No cls_head info found in checkpoint/meta. Need celltype-trained model.")

    model = CrossAttnZINBModel(
        img_encoder=foundation,
        gene_encoder=gene_encoder,
        gcn=gcn,
        dim_img=img_dim,
        dim_gene=args.proj_dim_gene,
        dim_spot=args.proj_dim_spot,
        dim_fuse=args.fuse_dim,
        num_heads=args.heads,
        dropout=args.dropout,
        merge=args.merge,
        img_tokens=args.img_tokens,
        spot_tokens=args.spot_tokens,
        gene_tokens=args.gene_tokens,
        decoder_hidden=1024,
        gene_dim=gene_dim,
        num_celltypes=num_celltypes,
        cls_head_hidden=ckpt_ct_hidden,
        cls_dropout=0.0,
    ).to(device)

    if args.pt_xattn:
        load_xattn_into_fusion(model.fusion, args.pt_xattn)
    model.load_state_dict(ft_sd, strict=False)
    model.eval()
    print(f"[Model] loaded, num_celltypes={num_celltypes}")

    celltype_names = []
    if isinstance(ct_meta, dict):
        celltype_names = [str(x) for x in ct_meta.get("celltype_names", [])]
    if len(celltype_names) != num_celltypes:
        if len(celltype_names) == 0:
            celltype_names = [f"class_{i}" for i in range(num_celltypes)]
        else:
            raise ValueError(
                f"celltype_names length {len(celltype_names)} != num_celltypes {num_celltypes}"
            )

    ct_meta_col = None
    if isinstance(ct_meta, dict):
        ct_meta_col = ct_meta.get("celltype_col", None)
    ct_col_train = _pick_celltype_col(train_adata, args.celltype_col, ct_meta_col)
    ct_col_eval = _pick_celltype_col(eval_adata, args.eval_celltype_col, args.celltype_col or ct_meta_col)
    print(f"[CellType] train_col={ct_col_train}, eval_col={ct_col_eval}")

    # 3) image-driven celltype prediction confusion
    if args.celltype_pred_expr_source == "eval":
        expr_base = _expression_matrix_for_spots(eval_adata, eval_spot_ids).astype(np.float32)
    else:
        gene_mean = _mean_gene_vector(train_adata).astype(np.float32)
        expr_base = np.repeat(gene_mean[None, :], len(eval_spot_ids), axis=0)

    pred_probs = _predict_celltype_probs_from_base_expr(
        model=model,
        img_loader=img_loader,
        spot_feat_in=spot_feat_in,
        edge_index=edge_index,
        edge_weight=edge_weight,
        expr_base_np=expr_base,
        device=device,
        amp=args.amp,
        amp_dtype=amp_dtype,
    )
    pred_idx = np.argmax(pred_probs, axis=1).astype(np.int64)
    pred_conf = np.max(pred_probs, axis=1).astype(np.float32)
    pred_name = np.asarray([celltype_names[i] for i in pred_idx], dtype=object)

    true_series = eval_adata.obs.loc[eval_spot_ids, ct_col_eval]
    valid_true = ~true_series.isna().to_numpy()
    true_name = true_series.astype(str).to_numpy(dtype=object)

    image_pred_df = pd.DataFrame({
        "spot_id": eval_spot_ids,
        "true_celltype": true_name,
        "pred_celltype_img": pred_name,
        "pred_celltype_img_conf": pred_conf,
        "x": coords_np[:, 0],
        "y": coords_np[:, 1],
    })
    image_pred_csv = os.path.join(args.out_dir, f"{args.prefix}_image_celltype_predictions.csv")
    image_pred_df.to_csv(image_pred_csv, index=False)
    print(f"[Save] {image_pred_csv}")

    valid_idx = np.where(valid_true)[0]
    if valid_idx.size == 0:
        raise ValueError(
            f"No valid labels in eval set for column '{ct_col_eval}'. "
            "Use --eval_celltype_col to set a labeled column."
        )
    y_true_img = true_name[valid_idx].astype(str)
    y_pred_img = pred_name[valid_idx].astype(str)
    labels_img = sorted(set(y_true_img.tolist()) | set(y_pred_img.tolist()))
    metrics_img = _save_confusion_artifacts(
        y_true=y_true_img,
        y_pred=y_pred_img,
        labels=labels_img,
        out_dir=args.out_dir,
        prefix=f"{args.prefix}_image_cls",
    )
    print(f"[ImageCls] acc={metrics_img['accuracy']:.4f}, bacc={metrics_img['balanced_accuracy']:.4f}")

    # 4) expression embedding label transfer
    train_expr_all = _expression_matrix_for_spots(train_adata, train_spot_ids).astype(np.float32)
    eval_expr_all = _expression_matrix_for_spots(eval_adata, eval_spot_ids).astype(np.float32)

    train_ct_series = train_adata.obs.loc[train_spot_ids, ct_col_train]
    train_valid = ~train_ct_series.isna().to_numpy()
    train_label_name = train_ct_series.astype(str).to_numpy(dtype=object)
    if train_valid.sum() == 0:
        raise ValueError("No valid labeled samples in train reference.")

    train_expr_ref = train_expr_all[train_valid]
    train_label_ref = train_label_name[train_valid].astype(str)

    label_names_ref = sorted(set(train_label_ref.tolist()))
    label_to_idx = {n: i for i, n in enumerate(label_names_ref)}
    train_y_idx = np.asarray([label_to_idx[x] for x in train_label_ref], dtype=np.int64)

    train_emb = _encode_gene_embeddings(
        model.gene_encoder, train_expr_ref, device=device, batch_size=args.gene_batch_size
    )
    eval_emb = _encode_gene_embeddings(
        model.gene_encoder, eval_expr_all, device=device, batch_size=args.gene_batch_size
    )

    pred_tr_idx, pred_tr_conf, pred_tr_sim = _knn_label_transfer_cosine(
        ref_emb=train_emb,
        ref_y_idx=train_y_idx,
        qry_emb=eval_emb,
        k=args.knn_k,
        chunk_size=args.sim_chunk,
    )
    pred_tr_name = np.asarray([label_names_ref[i] for i in pred_tr_idx], dtype=object)

    transfer_df = pd.DataFrame({
        "spot_id": eval_spot_ids,
        "true_celltype": true_name,
        "pred_celltype_expr_transfer": pred_tr_name,
        "pred_celltype_expr_conf": pred_tr_conf,
        "pred_celltype_expr_top1_sim": pred_tr_sim,
        "x": coords_np[:, 0],
        "y": coords_np[:, 1],
    })
    transfer_csv = os.path.join(args.out_dir, f"{args.prefix}_expr_transfer_predictions.csv")
    transfer_df.to_csv(transfer_csv, index=False)
    print(f"[Save] {transfer_csv}")

    y_true_tr = true_name[valid_idx].astype(str)
    y_pred_tr = pred_tr_name[valid_idx].astype(str)
    labels_tr = sorted(set(y_true_tr.tolist()) | set(y_pred_tr.tolist()))
    metrics_tr = _save_confusion_artifacts(
        y_true=y_true_tr,
        y_pred=y_pred_tr,
        labels=labels_tr,
        out_dir=args.out_dir,
        prefix=f"{args.prefix}_expr_transfer",
    )
    print(f"[ExprTransfer] acc={metrics_tr['accuracy']:.4f}, bacc={metrics_tr['balanced_accuracy']:.4f}")

    # 4-1) image representation -> expression celltype alignment (kNN transfer on image embeddings)
    metrics_img_repr = None
    pred_img_repr_name = None
    eval_img_rep = None
    if train_map_df_aligned is not None:
        train_img_loader = DataLoader(
            ImageMapDataset(train_map_df_aligned, image_transform=image_tf, id_col="spot_id"),
            batch_size=max(1, args.batch_size),
            shuffle=False,
            num_workers=max(1, args.num_workers),
            pin_memory=True,
            drop_last=False,
        )
        eval_img_loader = DataLoader(
            ImageMapDataset(map_df_aligned, image_transform=image_tf, id_col="spot_id"),
            batch_size=max(1, args.batch_size),
            shuffle=False,
            num_workers=max(1, args.num_workers),
            pin_memory=True,
            drop_last=False,
        )

        train_img_rep, train_img_ids = gather_image_representations(
            model.img_encoder,
            train_img_loader,
            device=device,
            amp_dtype=amp_dtype if args.amp else torch.float32,
            normalize=True,
            show_progress=True,
            desc="embed(train_img)",
        )
        eval_img_rep, eval_img_ids = gather_image_representations(
            model.img_encoder,
            eval_img_loader,
            device=device,
            amp_dtype=amp_dtype if args.amp else torch.float32,
            normalize=True,
            show_progress=True,
            desc="embed(eval_img)",
        )
        if eval_img_ids != eval_spot_ids:
            raise RuntimeError("eval image embedding ids are not aligned with eval_spot_ids.")

        train_img_ct_series = train_adata.obs.loc[train_img_ids, ct_col_train]
        train_img_valid = ~train_img_ct_series.isna().to_numpy()
        train_img_ct_name = train_img_ct_series.astype(str).to_numpy(dtype=object)
        if int(train_img_valid.sum()) <= 0:
            raise ValueError("No valid labeled samples in train image reference.")

        ref_img_emb = train_img_rep[train_img_valid]
        ref_img_label = train_img_ct_name[train_img_valid].astype(str)
        img_ref_names = sorted(set(ref_img_label.tolist()))
        img_ref_to_idx = {n: i for i, n in enumerate(img_ref_names)}
        ref_img_y_idx = np.asarray([img_ref_to_idx[x] for x in ref_img_label], dtype=np.int64)

        pred_img_repr_idx, pred_img_repr_conf, pred_img_repr_sim = _knn_label_transfer_cosine(
            ref_emb=ref_img_emb,
            ref_y_idx=ref_img_y_idx,
            qry_emb=eval_img_rep,
            k=args.knn_k,
            chunk_size=args.sim_chunk,
        )
        pred_img_repr_name = np.asarray([img_ref_names[i] for i in pred_img_repr_idx], dtype=object)

        image_repr_df = pd.DataFrame({
            "spot_id": eval_spot_ids,
            "true_celltype": true_name,
            "pred_celltype_img_repr_transfer": pred_img_repr_name,
            "pred_celltype_img_repr_conf": pred_img_repr_conf,
            "pred_celltype_img_repr_top1_sim": pred_img_repr_sim,
            "x": coords_np[:, 0],
            "y": coords_np[:, 1],
        })
        image_repr_csv = os.path.join(args.out_dir, f"{args.prefix}_image_repr_transfer_predictions.csv")
        image_repr_df.to_csv(image_repr_csv, index=False)
        print(f"[Save] {image_repr_csv}")

        y_true_ir = true_name[valid_idx].astype(str)
        y_pred_ir = pred_img_repr_name[valid_idx].astype(str)
        labels_ir = sorted(set(y_true_ir.tolist()) | set(y_pred_ir.tolist()))
        metrics_img_repr = _save_confusion_artifacts(
            y_true=y_true_ir,
            y_pred=y_pred_ir,
            labels=labels_ir,
            out_dir=args.out_dir,
            prefix=f"{args.prefix}_image_repr_transfer",
        )
        print(
            f"[ImageReprTransfer] acc={metrics_img_repr['accuracy']:.4f}, "
            f"bacc={metrics_img_repr['balanced_accuracy']:.4f}"
        )
    else:
        print("[ImageReprTransfer][skip] --train_csv not provided.")

    # 5) UMAP + spatial visualization
    true_for_plot = true_name.astype(str).copy()
    true_for_plot[~valid_true] = "__NA__"
    pred_img_for_plot = pred_name.astype(str).copy()
    pred_tr_for_plot = pred_tr_name.astype(str).copy()
    pred_img_repr_for_plot = (
        pred_img_repr_name.astype(str).copy() if pred_img_repr_name is not None else np.asarray(["__NA__"] * len(true_for_plot), dtype=object)
    )
    all_labs = (
        list(true_for_plot.tolist())
        + list(pred_img_for_plot.tolist())
        + list(pred_tr_for_plot.tolist())
        + list(pred_img_repr_for_plot.tolist())
    )
    palette = _build_palette(all_labs)

    idx_umap = _subsample_idx(len(eval_spot_ids), args.umap_max_points, args.seed)
    eval_expr_umap = _run_umap(
        eval_expr_all[idx_umap],
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        pca_n_comps=args.umap_pca_comps,
        seed=args.seed,
    )
    eval_emb_umap = _run_umap(
        eval_emb[idx_umap],
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        pca_n_comps=args.umap_pca_comps,
        seed=args.seed,
    )

    np.save(os.path.join(args.out_dir, f"{args.prefix}_umap_raw_expr.npy"), eval_expr_umap)
    np.save(os.path.join(args.out_dir, f"{args.prefix}_umap_gene_emb.npy"), eval_emb_umap)
    np.save(os.path.join(args.out_dir, f"{args.prefix}_umap_indices.npy"), idx_umap)
    if eval_img_rep is not None:
        eval_img_umap = _run_umap(
            eval_img_rep[idx_umap],
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            pca_n_comps=args.umap_pca_comps,
            seed=args.seed,
        )
        np.save(os.path.join(args.out_dir, f"{args.prefix}_umap_image_emb.npy"), eval_img_umap)
    else:
        eval_img_umap = None

    # raw expression UMAP: true vs transferred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_categorical_scatter(
        axes[0], eval_expr_umap[:, 0], eval_expr_umap[:, 1], true_for_plot[idx_umap],
        palette=palette, title="Raw Expression UMAP (True)", point_size=args.point_size
    )
    _plot_categorical_scatter(
        axes[1], eval_expr_umap[:, 0], eval_expr_umap[:, 1], pred_tr_for_plot[idx_umap],
        palette=palette, title="Raw Expression UMAP (Transfer Pred)", point_size=args.point_size
    )
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    raw_umap_png = os.path.join(args.out_dir, f"{args.prefix}_umap_raw_expr_true_vs_transfer.png")
    plt.savefig(raw_umap_png, dpi=300)
    plt.close()

    # gene embedding UMAP: true vs transferred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_categorical_scatter(
        axes[0], eval_emb_umap[:, 0], eval_emb_umap[:, 1], true_for_plot[idx_umap],
        palette=palette, title="Gene Embedding UMAP (True)", point_size=args.point_size
    )
    _plot_categorical_scatter(
        axes[1], eval_emb_umap[:, 0], eval_emb_umap[:, 1], pred_tr_for_plot[idx_umap],
        palette=palette, title="Gene Embedding UMAP (Transfer Pred)", point_size=args.point_size
    )
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    emb_umap_png = os.path.join(args.out_dir, f"{args.prefix}_umap_gene_emb_true_vs_transfer.png")
    plt.savefig(emb_umap_png, dpi=300)
    plt.close()

    # image embedding UMAP: true vs image-repr transfer
    if eval_img_umap is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _plot_categorical_scatter(
            axes[0], eval_img_umap[:, 0], eval_img_umap[:, 1], true_for_plot[idx_umap],
            palette=palette, title="Image Embedding UMAP (True)", point_size=args.point_size
        )
        _plot_categorical_scatter(
            axes[1], eval_img_umap[:, 0], eval_img_umap[:, 1], pred_img_repr_for_plot[idx_umap],
            palette=palette, title="Image Embedding UMAP (Image-Repr Pred)", point_size=args.point_size
        )
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
        plt.tight_layout()
        img_umap_png = os.path.join(args.out_dir, f"{args.prefix}_umap_image_emb_true_vs_imgreprpred.png")
        plt.savefig(img_umap_png, dpi=300)
        plt.close()

    # spatial: true vs image pred vs expr transfer pred vs image-repr pred
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    _plot_categorical_scatter(
        axes[0], coords_np[:, 0], coords_np[:, 1], true_for_plot, palette=palette,
        title="Spatial (True)", point_size=args.point_size, invert_y=True
    )
    _plot_categorical_scatter(
        axes[1], coords_np[:, 0], coords_np[:, 1], pred_img_for_plot, palette=palette,
        title="Spatial (Image Pred)", point_size=args.point_size, invert_y=True
    )
    _plot_categorical_scatter(
        axes[2], coords_np[:, 0], coords_np[:, 1], pred_tr_for_plot, palette=palette,
        title="Spatial (Expr Transfer Pred)", point_size=args.point_size, invert_y=True
    )
    _plot_categorical_scatter(
        axes[3], coords_np[:, 0], coords_np[:, 1], pred_img_repr_for_plot, palette=palette,
        title="Spatial (Image-Repr Pred)", point_size=args.point_size, invert_y=True
    )
    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    spatial_png = os.path.join(args.out_dir, f"{args.prefix}_spatial_true_vs_preds.png")
    plt.savefig(spatial_png, dpi=300)
    plt.close()

    summary = {
        "image_cls": metrics_img,
        "expr_transfer": metrics_tr,
        "image_repr_transfer": metrics_img_repr,
        "n_eval": int(len(eval_spot_ids)),
        "n_eval_labeled": int(valid_true.sum()),
        "n_train_ref": int(len(train_spot_ids)),
        "n_train_ref_labeled": int(train_valid.sum()),
        "celltype_train_col": ct_col_train,
        "celltype_eval_col": ct_col_eval,
    }
    with open(os.path.join(args.out_dir, f"{args.prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
