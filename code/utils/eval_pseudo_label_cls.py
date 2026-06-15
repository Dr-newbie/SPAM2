#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pseudo_label_cls.py
------------------------
Ablation / evaluation script for pseudo-label (cell-type) classification quality.

Evaluates the cls_head predictions at TWO granularities:
  1. Fine-grained  — original checkpoint cell-type labels
  2. Coarse        — 10 integrated categories (configurable; default = breast cancer)

Metrics
  • Accuracy, Macro-F1, Weighted-F1, ECE  (both fine + coarse)
  • Per-class Precision / Recall / F1
  • Entropy distribution of predictions

Visualizations saved to <out_dir>/figures/
  confusion_matrix_fine.png    — normalized confusion matrix (fine labels)
  confusion_matrix_coarse.png  — normalized confusion matrix (coarse labels)
  per_class_f1_fine.png        — horizontal F1 bar chart (fine)
  per_class_f1_coarse.png      — horizontal F1 bar chart (coarse)
  celltype_distribution.png    — GT vs Predicted distribution (coarse)
  entropy_histogram.png        — H_norm distribution with confidence threshold lines
  calibration_curve.png        — reliability diagram (ECE)
  umap_gt_fine.png             — UMAP of fused embeddings, colored by GT fine label
  umap_gt_coarse.png           — UMAP colored by GT coarse label
  umap_pred_coarse.png         — UMAP colored by predicted coarse label
  umap_confidence.png          — UMAP colored by prediction confidence (H_norm)

Example:
  python eval_pseudo_label_cls.py \\
    --h5ad   /data/.../best/recovered_data.h5ad \\
    --csv    /data/.../spot_patch_map_256.csv \\
    --root   /data/.../expression \\
    --ft_model_ckpt    /data/.../best/model.pt \\
    --celltype_meta_pt /data/.../best/celltype_meta.pt \\
    --celltype_col     Cluster \\
    --enc_name uni_v1 \\
    --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \\
    --img_tokens 10 --spot_tokens 10 --gene_tokens 10 \\
    --graph_type delaunay --k 12 \\
    --expr_source eval \\
    --batch_size 128 --num_workers 4 \\
    --out_dir ./ablation/pseudo_label_cls_breast_cancer
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

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
# Default coarse cell-type mapping (breast cancer)
# Fine-grained (20 classes) → Coarse (10 categories)
# ============================================================
BREAST_CANCER_COARSE_MAP = {
    "B_Cells":                 "B Cells",
    "CD4+_T_Cells":            "T Cells",
    "CD8+_T_Cells":            "T Cells",
    "DCIS_1":                  "Malignant",
    "DCIS_2":                  "Malignant",
    "Endothelial":             "Endothelial",
    "IRF7+_DCs":               "Myeloid",
    "Invasive_Tumor":          "Malignant",
    "LAMP3+_DCs":              "Myeloid",
    "Macrophages_1":           "Macrophage",
    "Macrophages_2":           "Macrophage",
    "Mast_Cells":              "Myeloid",
    "Myoepi_ACTA2+":           "Epithelial",
    "Myoepi_KRT15+":           "Epithelial",
    "Perivascular-Like":       "Perivascular/Mural",
    "Prolif_Invasive_Tumor":   "Malignant",
    "Stromal":                 "Stromal",
    "Stromal_&_T_Cell_Hybrid": "Unassigned",
    "T_Cell_&_Tumor_Hybrid":   "Unassigned",
    "Unlabeled":               "Unassigned",
}

COARSE_CATEGORIES = [
    "B Cells", "T Cells", "Unassigned", "Endothelial",
    "Perivascular/Mural", "Epithelial", "Myeloid",
    "Stromal", "Macrophage", "Malignant",
]

# Consistent color palette for coarse categories
COARSE_PALETTE = {
    "B Cells":           "#4878CF",
    "T Cells":           "#6ACC65",
    "Unassigned":        "#B8B8B8",
    "Endothelial":       "#D65F5F",
    "Perivascular/Mural":"#EE854A",
    "Epithelial":        "#82C6E2",
    "Myeloid":           "#956CB4",
    "Stromal":           "#C4AD66",
    "Macrophage":        "#DC7EC0",
    "Malignant":         "#797979",
}


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
        self.decoder = ZINBDecoder(
            in_dim=dim_fuse, gene_dim=gene_dim, hidden=decoder_hidden, dropout=dropout
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
        return mu, theta, pi, z_fused, cls_logits


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


def _mean_gene_vector(adata):
    m = adata.X.mean(axis=0)
    m = np.asarray(m.A1 if hasattr(m, "A1") else m, dtype=np.float32).ravel()
    return m


# ============================================================
# Metrics (no sklearn dependency)
# ============================================================
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1);  pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def per_class_metrics(y_true, y_pred, n_classes):
    results = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp + 1e-9);  rec = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        results[c] = {"precision": prec, "recall": rec, "f1": f1,
                      "support": int((y_true == c).sum())}
    return results


def macro_f1(per_cls):
    vals = [v["f1"] for v in per_cls.values() if v["support"] > 0]
    return float(np.mean(vals)) if vals else 0.0


def weighted_f1(per_cls):
    total = sum(v["support"] for v in per_cls.values())
    if total == 0: return 0.0
    return float(sum(v["f1"] * v["support"] for v in per_cls.values()) / total)


def confusion_matrix_np(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


# ============================================================
# Coarse mapping helpers
# ============================================================
def build_coarse_map(fine_names: list, coarse_map_dict: dict):
    """
    Returns:
      fine_to_coarse_idx : (n_fine,) int array  — maps fine idx → coarse idx
      coarse_names       : list of coarse category names (in order)
    """
    # collect ordered coarse names (preserve COARSE_CATEGORIES order where possible)
    coarse_set = set(coarse_map_dict.values())
    ordered = [c for c in COARSE_CATEGORIES if c in coarse_set]
    remaining = sorted(c for c in coarse_set if c not in ordered)
    coarse_names = ordered + remaining
    coarse_idx_map = {c: i for i, c in enumerate(coarse_names)}

    fine_to_coarse = np.full(len(fine_names), -1, dtype=np.int64)
    for fi, fname in enumerate(fine_names):
        cname = coarse_map_dict.get(fname, None)
        if cname is not None and cname in coarse_idx_map:
            fine_to_coarse[fi] = coarse_idx_map[cname]
    return fine_to_coarse, coarse_names


# ============================================================
# Visualization helpers
# ============================================================
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update({
        "figure.dpi":       150,
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
    })
    return plt, sns


def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str,
                           out_path: str, figsize_per_class: float = 0.55):
    plt, sns = _setup_mpl()
    n = len(labels)
    fs = max(6, n * figsize_per_class)
    fig, ax = plt.subplots(figsize=(fs + 1.5, fs))

    # row-normalize
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.where(row_sum > 0, cm / np.maximum(row_sum, 1), 0.0)

    sns.heatmap(
        cm_norm, annot=(n <= 20), fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.3 if n <= 20 else 0.0,
        ax=ax, cbar_kws={"shrink": 0.7},
        annot_kws={"size": max(6, 9 - n // 5)},
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("Ground Truth", labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {os.path.basename(out_path)}")


def plot_per_class_f1(per_cls: dict, names: list, title: str, out_path: str,
                       palette: dict = None):
    plt, sns = _setup_mpl()
    labels  = [names[c] for c in sorted(per_cls.keys()) if per_cls[c]["support"] > 0]
    f1_vals = [per_cls[c]["f1"] for c in sorted(per_cls.keys()) if per_cls[c]["support"] > 0]
    supports= [per_cls[c]["support"] for c in sorted(per_cls.keys()) if per_cls[c]["support"] > 0]

    order = np.argsort(f1_vals)
    labels   = [labels[i]   for i in order]
    f1_vals  = [f1_vals[i]  for i in order]
    supports = [supports[i] for i in order]

    colors = [palette.get(l, "#4C72B0") if palette else "#4C72B0" for l in labels]

    fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.38)))
    bars = ax.barh(labels, f1_vals, color=colors, edgecolor="white", height=0.7)
    for bar, sup in zip(bars, supports):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={sup:,}", va="center", ha="left", fontsize=8, color="#555555")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 Score")
    ax.set_title(title, fontweight="bold")
    ax.axvline(np.mean(f1_vals), color="red", linestyle="--", linewidth=1.2,
               label=f"Macro-F1 = {np.mean(f1_vals):.3f}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {os.path.basename(out_path)}")


def plot_celltype_distribution(gt_labels, pred_labels, coarse_names, out_path):
    """Side-by-side GT vs Predicted coarse cell type distribution."""
    plt, _ = _setup_mpl()
    n_cls  = len(coarse_names)
    gt_counts   = np.array([(gt_labels   == c).sum() for c in range(n_cls)])
    pred_counts = np.array([(pred_labels == c).sum() for c in range(n_cls)])
    total = max(gt_counts.sum(), 1)

    x     = np.arange(n_cls)
    width = 0.38
    colors = [COARSE_PALETTE.get(c, "#888888") for c in coarse_names]

    fig, ax = plt.subplots(figsize=(max(8, n_cls * 0.9), 5))
    ax.bar(x - width/2, gt_counts   / total * 100, width, label="Ground Truth",
           color=colors, edgecolor="white", alpha=0.85)
    ax.bar(x + width/2, pred_counts / total * 100, width, label="Predicted",
           color=colors, edgecolor="white", alpha=0.5, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels(coarse_names, rotation=35, ha="right")
    ax.set_ylabel("% of cells")
    ax.set_title("Cell-type Distribution: Ground Truth vs Predicted (Coarse)",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {os.path.basename(out_path)}")


def plot_entropy_histogram(H_norm: np.ndarray, out_path: str):
    plt, _ = _setup_mpl()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(H_norm, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(0.3, color="orange", linestyle="--", linewidth=1.5,
               label="H_norm = 0.3  (high confidence)")
    ax.axvline(0.5, color="red",    linestyle="--", linewidth=1.5,
               label="H_norm = 0.5  (uncertain)")
    pct_conf = float((H_norm < 0.3).mean()) * 100
    ax.set_xlabel("Normalized Entropy  H_norm  (0 = certain, 1 = uniform)")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Prediction Entropy Distribution\n"
                 f"({pct_conf:.1f}% of cells with H_norm < 0.3)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {os.path.basename(out_path)}")


def plot_calibration_curve(probs: np.ndarray, gt_labels: np.ndarray,
                            out_path: str, n_bins: int = 15):
    plt, _ = _setup_mpl()
    conf    = probs.max(axis=1)
    pred    = probs.argmax(axis=1)
    correct = (pred == gt_labels).astype(float)
    bins    = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_acc, bin_conf, bin_cnt = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_acc.append(correct[mask].mean())
        bin_conf.append(conf[mask].mean())
        bin_cnt.append(int(mask.sum()))

    ece = compute_ece(probs, gt_labels, n_bins=n_bins)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax1.plot(bin_conf, bin_acc, "o-", color="#D65F5F", linewidth=2,
             markersize=6, label=f"Model (ECE={ece:.4f})")
    ax1.fill_between(bin_conf, bin_conf, bin_acc, alpha=0.15, color="#D65F5F")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction Correct")
    ax1.set_title("Calibration Curve (Reliability Diagram)", fontweight="bold")
    ax1.legend(fontsize=9);  ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1);  ax1.set_ylim(0, 1)

    # confidence histogram
    ax2.bar(bin_centers, bin_cnt, width=(bins[1] - bins[0]) * 0.85,
            color="#4C72B0", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Predicted Confidence");  ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {os.path.basename(out_path)}")


def plot_umap(embeddings: np.ndarray, color_labels: np.ndarray,
              label_names: list, title: str, out_path: str,
              palette: dict = None, point_size: float = 3.0,
              n_neighbors: int = 30, min_dist: float = 0.3):
    """Compute UMAP and save scatter colored by label_names."""
    plt, _ = _setup_mpl()
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                 n_components=2, random_state=42, verbose=False)
        emb2d = reducer.fit_transform(embeddings.astype(np.float32))
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50, random_state=42)
        emb50 = pca.fit_transform(embeddings.astype(np.float32))
        from sklearn.manifold import TSNE
        emb2d = TSNE(n_components=2, random_state=42, perplexity=40,
                     n_iter=1000).fit_transform(emb50)
        method = "t-SNE"

    unique_labels = sorted(set(color_labels.tolist()))
    if palette:
        cmap = [palette.get(label_names[l], "#888888") if l < len(label_names) else "#888888"
                for l in unique_labels]
    else:
        import matplotlib.cm as cm
        cmap_fn = cm.get_cmap("tab20", len(unique_labels))
        cmap = [cmap_fn(i) for i in range(len(unique_labels))]

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, ul in enumerate(unique_labels):
        mask = color_labels == ul
        lname = label_names[ul] if ul < len(label_names) else f"cls_{ul}"
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   s=point_size, alpha=0.55, color=cmap[i], label=lname, rasterized=True)

    ax.set_title(f"{method} — {title}", fontweight="bold")
    ax.set_xlabel(f"{method} 1");  ax.set_ylabel(f"{method} 2")
    ax.axis("off")

    ncol = max(1, len(unique_labels) // 25 + 1)
    legend = ax.legend(markerscale=4, fontsize=7, ncol=ncol,
                       loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       borderaxespad=0, framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [viz] {os.path.basename(out_path)}")


def plot_umap_continuous(embeddings: np.ndarray, values: np.ndarray,
                          title: str, out_path: str,
                          cmap_name: str = "RdYlGn_r",
                          n_neighbors: int = 30, min_dist: float = 0.3,
                          umap_coords: np.ndarray = None):
    """UMAP scatter with continuous color (e.g., confidence / entropy)."""
    plt, _ = _setup_mpl()
    if umap_coords is not None:
        emb2d = umap_coords
        method = "UMAP"
    else:
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                     n_components=2, random_state=42, verbose=False)
            emb2d = reducer.fit_transform(embeddings.astype(np.float32))
            method = "UMAP"
        except Exception:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            emb50 = PCA(n_components=50, random_state=42).fit_transform(embeddings.astype(np.float32))
            emb2d = TSNE(n_components=2, random_state=42, perplexity=40).fit_transform(emb50)
            method = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=values, cmap=cmap_name,
                    s=3.0, alpha=0.6, rasterized=True, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_title(f"{method} — {title}", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [viz] {os.path.basename(out_path)}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate pseudo-label cell-type classification "
                                             "with fine-grained + coarse metrics and visualizations")
    # data
    ap.add_argument("--h5ad",      required=True)
    ap.add_argument("--csv",       required=True)
    ap.add_argument("--root",      default=None)
    ap.add_argument("--id_col",    default="cell_id")
    ap.add_argument("--path_col",  default="patch_path")
    ap.add_argument("--celltype_col", default=None)
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
    # ckpt
    ap.add_argument("--ft_model_ckpt",    required=True)
    ap.add_argument("--celltype_meta_pt", default=None)
    # expr source
    ap.add_argument("--expr_source", default="mean", choices=["mean","eval"])
    ap.add_argument("--train_h5ad",  default=None)
    # coarse mapping
    ap.add_argument("--coarse_map_json", default=None,
                    help="JSON file mapping fine class name → coarse category name. "
                         "If omitted, the default breast-cancer mapping is used.")
    ap.add_argument("--no_coarse", action="store_true",
                    help="Skip coarse-level analysis entirely.")
    # viz
    ap.add_argument("--umap_subsample", type=int, default=5000,
                    help="Max cells to include in UMAP plots (0 = all).")
    ap.add_argument("--umap_n_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist",    type=float, default=0.3)
    ap.add_argument("--no_umap",          action="store_true")
    # run
    ap.add_argument("--batch_size",  type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--amp",         action="store_true")
    ap.add_argument("--ece_bins",    type=int, default=15)
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    device = select_device(args.device)
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16) if (args.amp and device.type == "cuda") else torch.float32
    print(f"[Device] {device}  AMP={args.amp}  dtype={amp_dtype}")

    # ── load data ─────────────────────────────────────────────────────────────
    adata = load_h5ad_with_preproc(args.h5ad, use_hvg=False, n_top_genes=None,
                                   x_key="x_centroid", y_key="y_centroid")
    if args.expr_source == "mean" and args.train_h5ad:
        t_adata = load_h5ad_with_preproc(args.train_h5ad, use_hvg=False, n_top_genes=None,
                                          x_key="x_centroid", y_key="y_centroid")
        common  = adata.var_names.intersection(t_adata.var_names)
        t_adata = t_adata[:, [g for g in t_adata.var_names if g in common]].copy()
        adata   = adata[:,  [g for g in adata.var_names   if g in common]].copy()
        expr_mean = _mean_gene_vector(t_adata)
    else:
        expr_mean = _mean_gene_vector(adata)

    map_df = load_patch_mapping_csv_xenium(args.csv, root_dir=args.root,
                                           id_col=args.id_col, path_col=args.path_col,
                                           drop_missing=True)
    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(adata, map_df, id_col="spot_id",
                                                          coord_key="spatial")
    if map_df_aligned.shape[0] != adata.n_obs:
        adata = adata[map_df_aligned["spot_id"].astype(str).tolist()].copy()
    n_spots  = map_df_aligned.shape[0]
    gene_dim = adata.n_vars

    # ── celltype meta ──────────────────────────────────────────────────────────
    meta_path = args.celltype_meta_pt
    if meta_path is None and args.ft_model_ckpt:
        cand = os.path.join(os.path.dirname(args.ft_model_ckpt), "celltype_meta.pt")
        if os.path.isfile(cand):
            meta_path = cand
    ct_meta = torch.load(meta_path, map_location="cpu") if meta_path and os.path.isfile(meta_path) else {}
    celltype_names = [str(x) for x in ct_meta.get("celltype_names", [])]

    # ── coarse mapping ─────────────────────────────────────────────────────────
    if not args.no_coarse and celltype_names:
        if args.coarse_map_json and os.path.isfile(args.coarse_map_json):
            with open(args.coarse_map_json) as f:
                coarse_map_dict = json.load(f)
            print(f"[Coarse] loaded mapping from {args.coarse_map_json}")
        else:
            coarse_map_dict = BREAST_CANCER_COARSE_MAP
            print("[Coarse] using default breast-cancer coarse mapping")
        fine_to_coarse_idx, coarse_names = build_coarse_map(celltype_names, coarse_map_dict)
        unmapped = [celltype_names[i] for i in range(len(celltype_names))
                    if fine_to_coarse_idx[i] < 0]
        if unmapped:
            print(f"[Coarse][warn] {len(unmapped)} fine classes not in coarse map: {unmapped}")
        coarse_palette = {c: COARSE_PALETTE.get(c, "#888888") for c in coarse_names}
    else:
        fine_to_coarse_idx = None
        coarse_names       = []
        coarse_palette     = {}

    # ── ground-truth labels ────────────────────────────────────────────────────
    gt_col = None
    if args.celltype_col:
        cols_lower = {str(c).lower(): c for c in adata.obs.columns}
        gt_col = cols_lower.get(args.celltype_col.lower(), None)
        if gt_col is None:
            print(f"[WARN] --celltype_col={args.celltype_col!r} not found. "
                  f"Available: {list(adata.obs.columns)}")

    # ── model ──────────────────────────────────────────────────────────────────
    ft_sd  = _load_sd(args.ft_model_ckpt)
    ckpt_num_ct, ckpt_ct_hidden = _infer_cls_head_config(ft_sd)
    num_celltypes = max(int(ct_meta.get("num_celltypes", 0)) if ct_meta else 0, ckpt_num_ct)
    if num_celltypes <= 0:
        raise RuntimeError("No cls_head in checkpoint. Train with --lambda_ce > 0.")

    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation    = FoundationCls(weights_path=None)
    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)
    img_dim  = foundation.get_output_dim()

    gcn = SpatialGCN(in_dim=coords_np.shape[1], hidden_dim=256,
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
    print(f"[Model] num_celltypes={num_celltypes}  gene_dim={gene_dim}")

    # ── graph ──────────────────────────────────────────────────────────────────
    edge_index, edge_weight = build_graph(coords_np, k=args.k, graph_type=args.graph_type)
    edge_index   = edge_index.to(device)
    edge_weight  = edge_weight.to(device) if edge_weight is not None else None
    spot_feat_in = torch.from_numpy(coords_np).float().to(device)

    # ── expression matrix ──────────────────────────────────────────────────────
    if args.expr_source == "eval":
        X = adata.X
        if hasattr(X, "toarray"): X = X.toarray()
        expr_matrix = np.asarray(X, dtype=np.float32)
    else:
        expr_matrix = np.repeat(expr_mean[None, :], n_spots, axis=0)

    # ── inference ──────────────────────────────────────────────────────────────
    img_loader = make_img_only_dataloader(
        map_df=map_df_aligned, batch_size=args.batch_size,
        num_workers=args.num_workers, image_transform=image_tf, shuffle=False,
    )
    expr_t = torch.from_numpy(expr_matrix).to(device)
    all_probs = np.zeros((n_spots, num_celltypes), dtype=np.float32)
    all_fused = []
    t0 = time.time()

    with torch.no_grad():
        for batch_imgs in tqdm(img_loader, desc="[CLS] inference"):
            if isinstance(batch_imgs, dict):
                images = batch_imgs["image"].to(device);  idxs = batch_imgs["idx"].to(device).long()
            else:
                images, idxs = batch_imgs;  images = images.to(device);  idxs = idxs.to(device).long()
            batch = {"image": images, "expr": expr_t[idxs], "idx": idxs}
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(device.type == "cuda" and args.amp)):
                _, _, _, z_fused, cls_logits = model(batch, spot_feat_in, edge_index, edge_weight)
            all_probs[idxs.detach().cpu().numpy()] = \
                torch.softmax(cls_logits.float(), dim=1).detach().cpu().numpy()
            all_fused.append(z_fused.detach().float().cpu())

    print(f"[CLS] inference done  {time.time()-t0:.1f}s")
    all_fused_np = torch.cat(all_fused, dim=0).numpy()  # (N, D)

    pred_idx  = all_probs.argmax(axis=1).astype(np.int64)
    pred_conf = all_probs.max(axis=1).astype(np.float32)
    log_C  = float(np.log(num_celltypes) if num_celltypes > 1 else 1.0)
    H      = -(all_probs * np.log(np.clip(all_probs, 1e-9, None))).sum(axis=1)
    H_norm = (H / log_C).astype(np.float32)

    # ── metrics ────────────────────────────────────────────────────────────────
    metrics       = {}
    gt_fine_idx   = None
    gt_coarse_idx = None
    pred_coarse_idx = None
    spot_ids = map_df_aligned["spot_id"].astype(str).tolist()

    if gt_col is not None:
        gt_str = adata.obs.loc[spot_ids, gt_col].astype(str).to_numpy()
        if not celltype_names:
            celltype_names = sorted(set(gt_str.tolist()))
        n2i = {n: i for i, n in enumerate(celltype_names)}
        gt_fine_idx = np.array([n2i.get(n, -1) for n in gt_str], dtype=np.int64)
        valid = gt_fine_idx >= 0
        print(f"[GT] {valid.sum()} / {len(valid)} spots have matching GT label")

        # ── fine-grained metrics ───────────────────────────────────────────────
        pc_fine = per_class_metrics(gt_fine_idx[valid], pred_idx[valid], num_celltypes)
        acc_fine = float((gt_fine_idx[valid] == pred_idx[valid]).mean())
        mf1_fine = macro_f1(pc_fine);  wf1_fine = weighted_f1(pc_fine)
        ece_fine = compute_ece(all_probs[valid], gt_fine_idx[valid], n_bins=args.ece_bins)
        metrics["fine"] = {
            "n_samples": int(valid.sum()), "n_classes": num_celltypes,
            "accuracy": round(acc_fine, 4), "macro_f1": round(mf1_fine, 4),
            "weighted_f1": round(wf1_fine, 4), "ece": round(ece_fine, 4),
        }
        print(f"[Fine]  Acc={acc_fine:.4f}  Macro-F1={mf1_fine:.4f}  "
              f"Weighted-F1={wf1_fine:.4f}  ECE={ece_fine:.4f}")

        # per-class CSV (fine)
        pc_rows = [{"class": celltype_names[c], **{k: round(v, 4) if isinstance(v, float) else v
                    for k, v in pc_fine[c].items()}}
                   for c in range(num_celltypes)]
        pd.DataFrame(pc_rows).to_csv(
            os.path.join(args.out_dir, "per_class_metrics_fine.csv"), index=False)

        # confusion matrix (raw CSV, fine)
        cm_fine = confusion_matrix_np(gt_fine_idx[valid], pred_idx[valid], num_celltypes)
        pd.DataFrame(cm_fine, index=celltype_names, columns=celltype_names).to_csv(
            os.path.join(args.out_dir, "confusion_matrix_fine.csv"))

        # ── coarse metrics ─────────────────────────────────────────────────────
        if fine_to_coarse_idx is not None and len(coarse_names) > 0:
            gt_coarse_idx   = fine_to_coarse_idx[gt_fine_idx]
            pred_coarse_idx = fine_to_coarse_idx[pred_idx]
            coarse_valid    = (gt_coarse_idx >= 0) & valid
            n_coarse = len(coarse_names)

            pc_coarse = per_class_metrics(gt_coarse_idx[coarse_valid],
                                           pred_coarse_idx[coarse_valid], n_coarse)
            acc_c  = float((gt_coarse_idx[coarse_valid] == pred_coarse_idx[coarse_valid]).mean())
            mf1_c  = macro_f1(pc_coarse);  wf1_c = weighted_f1(pc_coarse)
            metrics["coarse"] = {
                "n_samples": int(coarse_valid.sum()), "n_classes": n_coarse,
                "accuracy": round(acc_c, 4), "macro_f1": round(mf1_c, 4),
                "weighted_f1": round(wf1_c, 4),
            }
            print(f"[Coarse] Acc={acc_c:.4f}  Macro-F1={mf1_c:.4f}  Weighted-F1={wf1_c:.4f}")

            pc_rows_c = [{"class": coarse_names[c], **{k: round(v, 4) if isinstance(v, float) else v
                          for k, v in pc_coarse[c].items()}}
                         for c in range(n_coarse)]
            pd.DataFrame(pc_rows_c).to_csv(
                os.path.join(args.out_dir, "per_class_metrics_coarse.csv"), index=False)
            cm_coarse = confusion_matrix_np(gt_coarse_idx[coarse_valid],
                                             pred_coarse_idx[coarse_valid], n_coarse)
            pd.DataFrame(cm_coarse, index=coarse_names, columns=coarse_names).to_csv(
                os.path.join(args.out_dir, "confusion_matrix_coarse.csv"))

    # entropy stats
    metrics["entropy"] = {
        "mean_H_norm": round(float(H_norm.mean()), 4),
        "std_H_norm":  round(float(H_norm.std()),  4),
        "pct_conf_lt_0.3": round(float((H_norm < 0.3).mean()), 4),
        "pct_conf_lt_0.5": round(float((H_norm < 0.5).mean()), 4),
    }

    # per-spot predictions
    pred_names = np.array([celltype_names[i] if i < len(celltype_names) else f"cls_{i}"
                            for i in pred_idx], dtype=object)
    pred_df = pd.DataFrame({
        "spot_id":       spot_ids,
        "pred_celltype": pred_names,
        "confidence":    pred_conf,
        "H_norm":        H_norm,
    })
    if gt_col is not None and gt_fine_idx is not None:
        pred_df.insert(1, "gt_celltype",
                        adata.obs.loc[spot_ids, gt_col].astype(str).to_numpy())
        pred_df["correct"] = (gt_fine_idx == pred_idx).astype(int)
        if gt_coarse_idx is not None:
            pred_df["gt_coarse"]   = [coarse_names[i] if 0 <= i < len(coarse_names) else "?"
                                       for i in gt_coarse_idx]
            pred_df["pred_coarse"] = [coarse_names[i] if 0 <= i < len(coarse_names) else "?"
                                       for i in pred_coarse_idx]
    pred_df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    np.savez_compressed(os.path.join(args.out_dir, "prob_matrix.npz"),
                        probs=all_probs, pred_idx=pred_idx, H_norm=H_norm,
                        celltype_names=np.array(celltype_names, dtype=object))

    # save summary JSON
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Metrics Summary]")
    print(json.dumps(metrics, indent=2))

    # ── visualizations ─────────────────────────────────────────────────────────
    print("\n[Viz] generating figures ...")

    # 1. confusion matrices
    if gt_fine_idx is not None:
        plot_confusion_matrix(
            cm_fine, celltype_names,
            title="Confusion Matrix — Fine-grained Cell Types (row-normalized)",
            out_path=os.path.join(fig_dir, "confusion_matrix_fine.png"),
        )
        if gt_coarse_idx is not None:
            plot_confusion_matrix(
                cm_coarse, coarse_names,
                title="Confusion Matrix — Coarse Cell Types (row-normalized)",
                out_path=os.path.join(fig_dir, "confusion_matrix_coarse.png"),
                figsize_per_class=0.65,
            )

        # 2. per-class F1 bar charts
        plot_per_class_f1(
            pc_fine, celltype_names,
            title="Per-class F1 — Fine-grained",
            out_path=os.path.join(fig_dir, "per_class_f1_fine.png"),
        )
        if gt_coarse_idx is not None:
            plot_per_class_f1(
                pc_coarse, coarse_names,
                title="Per-class F1 — Coarse",
                out_path=os.path.join(fig_dir, "per_class_f1_coarse.png"),
                palette=coarse_palette,
            )

        # 3. cell-type distribution
        if gt_coarse_idx is not None and pred_coarse_idx is not None:
            coarse_valid_mask = gt_coarse_idx >= 0
            plot_celltype_distribution(
                gt_coarse_idx[coarse_valid_mask],
                pred_coarse_idx[coarse_valid_mask],
                coarse_names,
                out_path=os.path.join(fig_dir, "celltype_distribution.png"),
            )

        # 4. calibration curve
        plot_calibration_curve(
            all_probs[valid], gt_fine_idx[valid],
            out_path=os.path.join(fig_dir, "calibration_curve.png"),
            n_bins=args.ece_bins,
        )

    # 5. entropy histogram (always)
    plot_entropy_histogram(H_norm, out_path=os.path.join(fig_dir, "entropy_histogram.png"))

    # 6. UMAP
    if not args.no_umap and all_fused_np.shape[0] > 0:
        # subsample for speed
        n_total = all_fused_np.shape[0]
        if args.umap_subsample > 0 and n_total > args.umap_subsample:
            rng   = np.random.RandomState(42)
            sel   = rng.choice(n_total, args.umap_subsample, replace=False)
            emb_s = all_fused_np[sel]
        else:
            sel   = np.arange(n_total)
            emb_s = all_fused_np

        print(f"  [UMAP] fitting on {len(sel)} cells ...")
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                n_components=2, random_state=42, verbose=False,
            )
            umap_coords = reducer.fit_transform(emb_s.astype(np.float32))
            umap_method = "UMAP"
        except Exception as e:
            print(f"  [UMAP] UMAP failed ({e}), falling back to t-SNE")
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            emb50 = PCA(n_components=50, random_state=42).fit_transform(emb_s.astype(np.float32))
            umap_coords = TSNE(n_components=2, random_state=42, perplexity=40,
                               n_iter=1000).fit_transform(emb50)
            umap_method = "t-SNE"

        np.save(os.path.join(args.out_dir, "umap_coords.npy"), umap_coords)
        np.save(os.path.join(args.out_dir, "umap_sel_idx.npy"), sel)

        def _save_umap_scatter(color_idx, names, palette, suffix, title):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm_mod
            unique = sorted(set(color_idx.tolist()))
            if palette:
                col_map = [palette.get(names[u], "#888888") if u < len(names) else "#888888"
                           for u in unique]
            else:
                cfn = cm_mod.get_cmap("tab20", max(len(unique), 1))
                col_map = [cfn(i) for i in range(len(unique))]
            fig, ax = plt.subplots(figsize=(9, 7))
            for i, u in enumerate(unique):
                mask = color_idx == u
                lname = names[u] if u < len(names) else f"cls_{u}"
                ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                           s=3.0, alpha=0.55, color=col_map[i], label=lname, rasterized=True)
            ncol = max(1, len(unique) // 25 + 1)
            ax.legend(markerscale=4, fontsize=7, ncol=ncol,
                      loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      borderaxespad=0, framealpha=0.8)
            ax.set_title(f"{umap_method} — {title}", fontweight="bold")
            ax.axis("off")
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"umap_{suffix}.png"),
                        bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  [viz] umap_{suffix}.png")

        # GT fine
        if gt_fine_idx is not None:
            _save_umap_scatter(gt_fine_idx[sel], celltype_names, None,
                               "gt_fine", "Ground Truth — Fine")
        # GT coarse
        if gt_coarse_idx is not None:
            valid_sel = gt_coarse_idx[sel] >= 0
            if valid_sel.any():
                _save_umap_scatter(gt_coarse_idx[sel], coarse_names, coarse_palette,
                                   "gt_coarse", "Ground Truth — Coarse")
        # Predicted coarse
        if pred_coarse_idx is not None:
            _save_umap_scatter(pred_coarse_idx[sel], coarse_names, coarse_palette,
                               "pred_coarse", "Predicted — Coarse")
        # Confidence (H_norm) — continuous
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=H_norm[sel], cmap="RdYlGn_r",
                        s=3.0, alpha=0.6, rasterized=True, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="H_norm (0=confident, 1=uncertain)")
        ax.set_title(f"{umap_method} — Prediction Entropy", fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "umap_confidence.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("  [viz] umap_confidence.png")

    print(f"\n[DONE] All outputs saved to: {args.out_dir}")
    print(f"       Figures in:            {fig_dir}/")


if __name__ == "__main__":
    main()
