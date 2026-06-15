#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_coarse_confusion.py

Load an existing celltype_inference.py predictions CSV and re-plot
confusion matrices at coarse (merged) cell-type granularity.

Usage:
  python Code_final/code/utils/plot_coarse_confusion.py \
    --csv /data/project/banana9903_xenium/inference/celltype_eval_breast/celltype_eval_image_repr_transfer_predictions.csv \
    --true_col true_celltype \
    --pred_col pred_celltype_img_repr_transfer \
    --out_dir /data/project/banana9903_xenium/inference/celltype_eval_breast \
    --prefix celltype_eval_image_repr_transfer
"""

import os
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────────────────────────────
# Breast cancer coarse cell-type mapping  (fine → coarse)
# ──────────────────────────────────────────────────────────────────────────────
BREAST_CANCER_COARSE_MAP = {
    "B_Cells": "B Cells",
    "CD4+_T_Cells": "T Cells",
    "CD8+_T_Cells": "T Cells",
    "DCIS_1": "Malignant",
    "DCIS_2": "Malignant",
    "Endothelial": "Endothelial",
    "IRF7+_DCs": "Myeloid",
    "Invasive_Tumor": "Malignant",
    "LAMP3+_DCs": "Myeloid",
    "Macrophages_1": "Macrophage",
    "Macrophages_2": "Macrophage",
    "Mast_Cells": "Myeloid",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "Perivascular-Like": "Perivascular/Mural",
    "Prolif_Invasive_Tumor": "Malignant",
    "Stromal": "Stromal",
    "Stromal_&_T_Cell_Hybrid": "Unassigned",
    "T_Cell_&_Tumor_Hybrid": "Unassigned",
    "Unlabeled": "Unassigned",
}

COARSE_ORDER = [
    "B Cells",
    "T Cells",
    "Myeloid",
    "Macrophage",
    "Malignant",
    "Epithelial",
    "Endothelial",
    "Perivascular/Mural",
    "Stromal",
    "Unassigned",
]

# Colour palette for coarse categories
COARSE_PALETTE = {
    "B Cells": "#4e79a7",
    "T Cells": "#f28e2b",
    "Myeloid": "#e15759",
    "Macrophage": "#76b7b2",
    "Malignant": "#59a14f",
    "Epithelial": "#edc948",
    "Endothelial": "#b07aa1",
    "Perivascular/Mural": "#ff9da7",
    "Stromal": "#9c755f",
    "Unassigned": "#bab0ac",
}


def apply_coarse_map(series: pd.Series, coarse_map: dict) -> pd.Series:
    return series.map(lambda x: coarse_map.get(str(x), str(x)))


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    out_dir: str,
    prefix: str,
    tag: str,
    palette: dict | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # row-normalised
    cm_norm = cm.astype(np.float32)
    row_sum = cm_norm.sum(axis=1, keepdims=True) + 1e-8
    cm_norm /= row_sum
    cmn_df = pd.DataFrame(cm_norm, index=labels, columns=labels)

    cm_df.to_csv(os.path.join(out_dir, f"{prefix}_{tag}_confusion_counts.csv"))
    cmn_df.to_csv(os.path.join(out_dir, f"{prefix}_{tag}_confusion_row_norm.csv"))

    n_cls = len(labels)
    fig_w = max(8, min(30, n_cls * 0.75 + 2))
    fig_h = max(6, min(24, n_cls * 0.65 + 2))

    # row-normalised heatmap
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    annot = n_cls <= 20
    sns.heatmap(
        cmn_df,
        cmap="magma",
        annot=annot,
        fmt=".2f" if annot else "",
        cbar=True,
        linewidths=0.4,
        linecolor="white",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title(f"{prefix}_{tag} confusion (row-normalized)", fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{prefix}_{tag}_confusion_row_norm.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[Save] {out_png}")

    # count heatmap
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm_df,
        cmap="Blues",
        annot=annot,
        fmt="d" if annot else "",
        cbar=True,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(f"{prefix}_{tag} confusion (count)", fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    out_count_png = os.path.join(out_dir, f"{prefix}_{tag}_confusion_counts.png")
    plt.savefig(out_count_png, dpi=300)
    plt.close()
    print(f"[Save] {out_count_png}")

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        os.path.join(out_dir, f"{prefix}_{tag}_classification_report.csv")
    )

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    metrics = {"accuracy": acc, "balanced_accuracy": bacc}
    with open(os.path.join(out_dir, f"{prefix}_{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{tag}] acc={acc:.4f}  bacc={bacc:.4f}")
    return metrics


def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    out_dir: str,
    prefix: str,
    tag: str,
    palette: dict | None = None,
):
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    f1s = [report.get(lab, {}).get("f1-score", 0.0) for lab in labels]
    colors = [palette.get(lab, "#888888") if palette else "#4878cf" for lab in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    ax.bar(labels, f1s, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title(f"{prefix}_{tag} per-class F1", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{prefix}_{tag}_per_class_f1.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[Save] {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description="Re-plot coarse confusion matrix from existing predictions CSV"
    )
    ap.add_argument("--csv", required=True, help="Predictions CSV from celltype_inference.py")
    ap.add_argument("--true_col", default="true_celltype")
    ap.add_argument("--pred_col", required=True, help="Column name for predicted cell type")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="celltype_eval")
    ap.add_argument(
        "--coarse_map_json",
        default=None,
        help="Optional JSON file with custom fine->coarse mapping (overrides default)",
    )
    ap.add_argument(
        "--no_fine",
        action="store_true",
        help="Skip fine-grained confusion matrix (only plot coarse)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[Load] {len(df)} rows from {args.csv}")

    # drop rows where true or pred is NaN
    df = df.dropna(subset=[args.true_col, args.pred_col])
    y_true_fine = df[args.true_col].astype(str).to_numpy()
    y_pred_fine = df[args.pred_col].astype(str).to_numpy()

    # ── Fine-grained ──────────────────────────────────────────────────────────
    if not args.no_fine:
        labels_fine = sorted(set(y_true_fine.tolist()) | set(y_pred_fine.tolist()))
        save_confusion_matrix(
            y_true_fine, y_pred_fine, labels_fine,
            args.out_dir, args.prefix, "fine",
        )
        plot_per_class_f1(
            y_true_fine, y_pred_fine, labels_fine,
            args.out_dir, args.prefix, "fine",
        )

    # ── Coarse mapping ────────────────────────────────────────────────────────
    coarse_map = dict(BREAST_CANCER_COARSE_MAP)
    if args.coarse_map_json:
        with open(args.coarse_map_json) as f:
            coarse_map.update(json.load(f))

    y_true_coarse = np.asarray([coarse_map.get(x, x) for x in y_true_fine], dtype=object)
    y_pred_coarse = np.asarray([coarse_map.get(x, x) for x in y_pred_fine], dtype=object)

    # Use canonical order, then append any unmapped categories alphabetically
    present = set(y_true_coarse.tolist()) | set(y_pred_coarse.tolist())
    labels_coarse = [c for c in COARSE_ORDER if c in present]
    labels_coarse += sorted(present - set(labels_coarse))

    save_confusion_matrix(
        y_true_coarse, y_pred_coarse, labels_coarse,
        args.out_dir, args.prefix, "coarse",
        palette=COARSE_PALETTE,
    )
    plot_per_class_f1(
        y_true_coarse, y_pred_coarse, labels_coarse,
        args.out_dir, args.prefix, "coarse",
        palette=COARSE_PALETTE,
    )

    print("[Done]")


if __name__ == "__main__":
    main()
