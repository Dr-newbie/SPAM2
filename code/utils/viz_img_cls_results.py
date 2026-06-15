#!/usr/bin/env python3
"""
Visualize img-cls predictions and compute accuracy/F1/confusion matrix.

Example:
python Code_final/code/utils/viz_img_cls_results.py \
  --pred_csv /data/project/banana9903_xenium/ckpts_img_cls2/infer_pred.csv \
  --h5ad /data/project/watercar99/Data/Spatial/xenium_rep1_io/data/merged_with_celltype_symbol_SPAM.h5ad \
  --cluster_col leiden \
  --out_dir /data/project/banana9903_xenium/ckpts_img_cls2/viz
"""

import os
import json
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc


def _norm_id_str(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _pick_cluster_col(adata, cluster_col: Optional[str]) -> str:
    cols = list(adata.obs.columns)
    cols_lower = {c.lower(): c for c in cols}
    if cluster_col:
        key = cluster_col.lower()
        if key in cols_lower:
            return cols_lower[key]
    cand = ["leiden", "cluster", "clusters", "seurat_clusters", "kmeans", "cluster_k8"]
    for c in cand:
        if c in cols_lower:
            return cols_lower[c]
    raise KeyError(f"cluster column not found in adata.obs. Available: {cols}")


def _coords_from_pred_or_adata(pred_df: pd.DataFrame, adata, id_col: str) -> Tuple[np.ndarray, np.ndarray]:
    # prefer CSV coords if available
    for cx, cy in [("x_he_px", "y_he_px"), ("x_morph_px", "y_morph_px"), ("x_um", "y_um")]:
        if cx in pred_df.columns and cy in pred_df.columns:
            return pred_df[cx].to_numpy(), pred_df[cy].to_numpy()
    # fallback to adata.obsm['spatial']
    if "spatial" not in adata.obsm:
        raise KeyError("No coords found in pred_csv and adata.obsm['spatial'] missing.")
    obs_ids = [_norm_id_str(x) for x in adata.obs_names.tolist()]
    obs_to_pos = {s: i for i, s in enumerate(obs_ids)}
    idxs = [obs_to_pos[_norm_id_str(s)] for s in pred_df[id_col].tolist()]
    coords = adata.obsm["spatial"][idxs, :]
    return coords[:, 0], coords[:, 1]


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp == 0 and (fp == 0 or fn == 0):
            f1s.append(0.0)
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"acc": acc, "macro_f1": macro_f1}


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1
    return cm


def _plot_confusion(cm: np.ndarray, labels: List[str], out_png: str, normalize: bool = False):
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_spatial(
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    out_png: str,
    title: str,
    max_points: int = 0,
    seed: int = 42,
    show_legend: bool = False,
):
    n = len(labels)
    if max_points and n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
        labels = [labels[i] for i in idx]

    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    label_to_color = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    colors = [label_to_color[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=2, c=colors, alpha=0.8, linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    if show_legend:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=lab,
                       markerfacecolor=label_to_color[lab], markersize=6)
            for lab in uniq
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_umap(
    umap_xy: np.ndarray,
    labels: List[str],
    out_png: str,
    title: str,
    max_points: int = 0,
    seed: int = 42,
    show_legend: bool = False,
):
    n = len(labels)
    if max_points and n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        umap_xy = umap_xy[idx, :]
        labels = [labels[i] for i in idx]

    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    label_to_color = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    colors = [label_to_color[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], s=2, c=colors, alpha=0.8, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if show_legend:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=lab,
                       markerfacecolor=label_to_color[lab], markersize=6)
            for lab in uniq
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _get_umap_coords(
    adata,
    pred_df: pd.DataFrame,
    id_col: str,
    umap_key: str,
    compute: bool,
    n_neighbors: int,
    min_dist: float,
) -> Optional[np.ndarray]:
    if umap_key not in adata.obsm:
        if not compute:
            print(f"[warn] UMAP key '{umap_key}' not found in adata.obsm. Skipping UMAP plots.")
            return None
        print(f"[info] computing UMAP (key={umap_key}) ...")
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=50)
        sc.tl.umap(adata, min_dist=min_dist)
        if "X_umap" in adata.obsm and umap_key != "X_umap":
            adata.obsm[umap_key] = adata.obsm["X_umap"]

    obs_ids = [_norm_id_str(x) for x in adata.obs_names.tolist()]
    obs_to_pos = {s: i for i, s in enumerate(obs_ids)}
    idxs = [obs_to_pos[_norm_id_str(s)] for s in pred_df[id_col].tolist()]
    return np.asarray(adata.obsm[umap_key][idxs, :], dtype=np.float64)


def _load_merge_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"merge map not found: {path}")
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("merge map json must be a dict: {orig_label: merged_label}")
        return {str(k): str(v) for k, v in obj.items()}
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("merge map csv must have at least 2 columns: orig, merged")
        return {str(k): str(v) for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}
    raise ValueError("merge map must be .json or .csv")


def _apply_merge_map(
    labels: np.ndarray,
    merge_map: Dict[str, str],
    default_label: Optional[str],
    drop_unmapped: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    mapped = []
    keep = []
    for i, l in enumerate(labels):
        if l in merge_map:
            mapped.append(merge_map[l])
            keep.append(True)
        else:
            if drop_unmapped:
                keep.append(False)
            else:
                mapped.append(default_label if default_label is not None else l)
                keep.append(True)
    return np.array(mapped, dtype=object), np.array(keep, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--cluster_col", default=None)
    ap.add_argument("--h5ad_id_col", default=None,
                    help="Use adata.obs[... ] as ID column. If not set, use obs_names.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--merge_map", default=None,
                    help="Path to label merge map (.json or .csv).")
    ap.add_argument("--merge_default", default="other",
                    help="Default merged label for unmapped classes (ignored if --merge_drop_unmapped).")
    ap.add_argument("--merge_drop_unmapped", action="store_true",
                    help="Drop samples with labels not in merge map.")
    ap.add_argument("--umap_key", default="X_umap", help="UMAP key in adata.obsm (default: X_umap)")
    ap.add_argument("--compute_umap", action="store_true", help="Compute UMAP if missing (can be slow)")
    ap.add_argument("--umap_n_neighbors", type=int, default=15)
    ap.add_argument("--umap_min_dist", type=float, default=0.5)
    ap.add_argument("--max_points", type=int, default=0, help="Optional downsample for plotting (0=all)")
    ap.add_argument("--legend", action="store_true", help="Show legend in plots (can be crowded)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_df = pd.read_csv(args.pred_csv)
    if "spot_id" not in pred_df.columns:
        raise KeyError(f"pred_csv missing 'spot_id': {pred_df.columns.tolist()}")

    adata = sc.read_h5ad(args.h5ad)
    adata.var_names_make_unique()
    clus_col = _pick_cluster_col(adata, args.cluster_col)

    if args.h5ad_id_col and args.h5ad_id_col in adata.obs.columns:
        adata_ids = adata.obs[args.h5ad_id_col].astype(str).map(_norm_id_str)
    else:
        adata_ids = pd.Index([_norm_id_str(x) for x in adata.obs_names.tolist()])
    id_to_label = {k: v for k, v in zip(adata_ids, adata.obs[clus_col].astype(str).tolist())}

    pred_df["spot_id"] = pred_df["spot_id"].astype(str).map(_norm_id_str)
    pred_df["true_label"] = pred_df["spot_id"].map(id_to_label)
    valid = pred_df["true_label"].notna().values
    if not valid.any():
        raise ValueError("No matching IDs between pred_csv and h5ad.")
    pred_df = pred_df.loc[valid].reset_index(drop=True)

    # labels ordering
    true_labels = pred_df["true_label"].astype(str).values
    pred_labels = pred_df["pred_label"].astype(str).values
    labels = sorted(pd.unique(true_labels).tolist())
    extra = [x for x in sorted(pd.unique(pred_labels).tolist()) if x not in labels]
    labels = labels + extra
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_true = np.array([label_to_idx[x] for x in true_labels], dtype=np.int64)
    y_pred = np.array([label_to_idx[x] for x in pred_labels], dtype=np.int64)

    metrics = _eval_metrics(y_true, y_pred, num_classes=len(labels))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    cm = _confusion_matrix(y_true, y_pred, n=len(labels))
    np.savetxt(os.path.join(args.out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    _plot_confusion(cm, labels, os.path.join(args.out_dir, "confusion_matrix.png"), normalize=False)
    _plot_confusion(cm, labels, os.path.join(args.out_dir, "confusion_matrix_norm.png"), normalize=True)

    # spatial plots
    try:
        x, y = _coords_from_pred_or_adata(pred_df, adata, id_col="spot_id")
        _plot_spatial(
            x, y, pred_labels.tolist(),
            os.path.join(args.out_dir, "spatial_pred.png"),
            "Predicted",
            max_points=args.max_points,
            seed=args.seed,
            show_legend=args.legend,
        )
        _plot_spatial(
            x, y, true_labels.tolist(),
            os.path.join(args.out_dir, "spatial_true.png"),
            "True",
            max_points=args.max_points,
            seed=args.seed,
            show_legend=args.legend,
        )
    except Exception as e:
        print(f"[warn] spatial plot skipped: {e}")

    # UMAP plots
    try:
        umap_xy = _get_umap_coords(
            adata,
            pred_df,
            id_col="spot_id",
            umap_key=args.umap_key,
            compute=args.compute_umap,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
        )
        if umap_xy is not None:
            _plot_umap(
                umap_xy, pred_labels.tolist(),
                os.path.join(args.out_dir, "umap_pred.png"),
                "UMAP Predicted",
                max_points=args.max_points,
                seed=args.seed,
                show_legend=args.legend,
            )
            _plot_umap(
                umap_xy, true_labels.tolist(),
                os.path.join(args.out_dir, "umap_true.png"),
                "UMAP True",
                max_points=args.max_points,
                seed=args.seed,
                show_legend=args.legend,
            )
    except Exception as e:
        print(f"[warn] umap plot skipped: {e}")

    # merged labels (optional)
    merge_map = _load_merge_map(args.merge_map)
    if merge_map:
        true_m, keep_t = _apply_merge_map(true_labels, merge_map, args.merge_default, args.merge_drop_unmapped)
        pred_m, keep_p = _apply_merge_map(pred_labels, merge_map, args.merge_default, args.merge_drop_unmapped)
        keep = keep_t & keep_p
        true_m = true_m[keep]
        pred_m = pred_m[keep]
        if not keep.all():
            print(f"[merge] dropped {np.sum(~keep)} samples due to unmapped labels")
        if true_m.size == 0:
            print("[warn] merged labels produced 0 samples; skipping merged metrics.")
            return
        # quick coverage summary
        uniq, cnt = np.unique(true_m, return_counts=True)
        top = sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda x: -x[1])[:10]
        print(f"[merge] true label top counts: {top}")

        merged_labels = sorted(pd.unique(true_m).tolist())
        extra = [x for x in sorted(pd.unique(pred_m).tolist()) if x not in merged_labels]
        merged_labels = merged_labels + extra
        m2i = {l: i for i, l in enumerate(merged_labels)}
        y_true_m = np.array([m2i[x] for x in true_m], dtype=np.int64)
        y_pred_m = np.array([m2i[x] for x in pred_m], dtype=np.int64)
        metrics_m = _eval_metrics(y_true_m, y_pred_m, num_classes=len(merged_labels))
        with open(os.path.join(args.out_dir, "metrics_merged.json"), "w") as f:
            json.dump(metrics_m, f, indent=2)
        print(f"[metrics_merged] acc={metrics_m['acc']:.4f} macro_f1={metrics_m['macro_f1']:.4f}")
        cm_m = _confusion_matrix(y_true_m, y_pred_m, n=len(merged_labels))
        np.savetxt(os.path.join(args.out_dir, "confusion_matrix_merged.csv"), cm_m, fmt="%d", delimiter=",")
        _plot_confusion(cm_m, merged_labels, os.path.join(args.out_dir, "confusion_matrix_merged.png"), normalize=False)
        _plot_confusion(cm_m, merged_labels, os.path.join(args.out_dir, "confusion_matrix_merged_norm.png"), normalize=True)
        if "umap_xy" in locals() and umap_xy is not None:
            _plot_umap(
                umap_xy, pred_m.tolist(),
                os.path.join(args.out_dir, "umap_pred_merged.png"),
                "UMAP Pred (Merged)",
                max_points=args.max_points,
                seed=args.seed,
                show_legend=args.legend,
            )
            _plot_umap(
                umap_xy, true_m.tolist(),
                os.path.join(args.out_dir, "umap_true_merged.png"),
                "UMAP True (Merged)",
                max_points=args.max_points,
                seed=args.seed,
                show_legend=args.legend,
            )
        # spatial merged plots
        try:
            x_m, y_m = _coords_from_pred_or_adata(pred_df.loc[keep].reset_index(drop=True), adata, id_col="spot_id")
            _plot_spatial(
                x_m, y_m, pred_m.tolist(),
                os.path.join(args.out_dir, "spatial_pred_merged.png"),
                "Predicted (Merged)",
                max_points=args.max_points,
                seed=args.seed,
                show_legend=args.legend,
            )
            _plot_spatial(
                x_m, y_m, true_m.tolist(),
                os.path.join(args.out_dir, "spatial_true_merged.png"),
                "True (Merged)",
                max_points=args.max_points,
                seed=args.seed,
                show_legend=args.legend,
            )
        except Exception as e:
            print(f"[warn] spatial merged plot skipped: {e}")


if __name__ == "__main__":
    main()
