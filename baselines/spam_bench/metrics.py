# -*- coding: utf-8 -*-
"""
Evaluation shared by every baseline and by SPAM itself.

Reviewer comment R2-2 asks how metrics are aggregated and how variable they
are, so this module fixes one definition and reports the spread rather than a
bare mean. PCC is computed per gene across cells (the axis SPAM reports);
genes with zero variance in either vector are dropped rather than counted as
zero, since a constant gene has undefined correlation.
"""
from typing import Dict, Optional

import numpy as np


def per_gene_pcc(pred: np.ndarray, true: np.ndarray,
                 min_std: float = 1e-8) -> np.ndarray:
    """(G,) per-gene Pearson r across cells. NaN where undefined."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    pc = p - p.mean(axis=0, keepdims=True)
    tc = t - t.mean(axis=0, keepdims=True)
    ps = np.sqrt((pc ** 2).sum(axis=0))
    ts = np.sqrt((tc ** 2).sum(axis=0))
    denom = ps * ts
    out = np.full(p.shape[1], np.nan)
    ok = (ps > min_std) & (ts > min_std) & (denom > 0)
    out[ok] = (pc[:, ok] * tc[:, ok]).sum(axis=0) / denom[ok]
    return out


def per_cell_pcc(pred: np.ndarray, true: np.ndarray, min_std: float = 1e-8) -> np.ndarray:
    """(N,) per-cell Pearson r across genes."""
    return per_gene_pcc(pred.T, true.T, min_std=min_std)


def evaluate(pred: np.ndarray, true: np.ndarray,
             gene_names: Optional[list] = None,
             top_k: int = 50) -> Dict[str, float]:
    """
    Returns the metric set reported for every method.

    `top_k` mirrors the manuscript's "top-N HEG" tables: genes ranked by mean
    observed expression in the evaluation set.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    g = per_gene_pcc(pred, true)
    c = per_cell_pcc(pred, true)
    valid = ~np.isnan(g)

    heg = np.argsort(-true.mean(axis=0))[:top_k]
    heg_valid = heg[~np.isnan(g[heg])]

    out = {
        "mse": float(np.mean((pred - true) ** 2)),
        "mae": float(np.mean(np.abs(pred - true))),
        "pcc_gene_mean": float(np.nanmean(g)),
        "pcc_gene_median": float(np.nanmedian(g)),
        "pcc_gene_std": float(np.nanstd(g)),
        "pcc_gene_q25": float(np.nanpercentile(g, 25)),
        "pcc_gene_q75": float(np.nanpercentile(g, 75)),
        "pcc_cell_mean": float(np.nanmean(c)),
        "pcc_cell_median": float(np.nanmedian(c)),
        f"pcc_top{top_k}heg_mean": float(np.nanmean(g[heg_valid])) if heg_valid.size else float("nan"),
        "n_genes_scored": int(valid.sum()),
        "n_genes_undefined": int((~valid).sum()),
        "n_cells": int(pred.shape[0]),
    }
    return out


def save_predictions(path: str, pred: np.ndarray, true: np.ndarray,
                     cell_ids: list, gene_names: list) -> None:
    """Persist raw predictions so metrics can be recomputed without retraining."""
    # savez_compressed on ~1M-cell matrices duplicates arrays (astype) and
    # buffers the zip stream — RAM spike killed ovarian bleep/stmcl at save.
    np.savez(path, pred=pred.astype(np.float32, copy=False),
                        true=true.astype(np.float32, copy=False),
                        cell_ids=np.asarray(cell_ids, dtype=object),
                        gene_names=np.asarray(gene_names, dtype=object))
