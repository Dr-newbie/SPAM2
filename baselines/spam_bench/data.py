# -*- coding: utf-8 -*-
"""
Common Xenium dataset for the baseline comparison (reviewer comment R2-1).

Every baseline in this comparison solves the same task — map a histology patch
to that spot's expression vector — but each upstream repo hard-codes a Visium
directory layout (tissue_positions_list.csv, scalefactors, a whole-slide .tif).
Our data is Xenium single-cell: one 256x256 patch per cell, expression in an
h5ad. Rather than fabricate fake Visium trees, we keep each baseline's *model
and training procedure* verbatim and swap only the dataset layer, so the
comparison differs in method and nothing else.

The split is the same spatial-block scheme SPAM uses, so no baseline sees a
different train/test partition.
"""
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Same filter the SPAM pipeline uses, imported rather than reimplemented so the
# two cannot drift apart — that is what makes "same genes, same metric" true.
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "code"))
from utils.gene_panel import drop_control_probes  # noqa: E402


def _norm_id(x) -> str:
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def load_xenium_table(h5ad_path: str, csv_path: str, root: Optional[str] = None,
                      id_col: str = "cell_id", path_col: str = "patch_path",
                      n_top_genes: Optional[int] = None,
                      inside_he_only: bool = False):
    """
    Returns (expr, coords, patch_paths, gene_names).

    expr is log1p(CP10K) — the same preprocessing SPAM's loader applies — plus
    the raw integer counts, since some baselines (and the corrected ZINB) need
    counts rather than normalized values.
    """
    import anndata as ad
    import scanpy as sc

    adata = ad.read_h5ad(h5ad_path)
    # Xenium control probes are dropped before normalization, exactly as in the
    # SPAM pipeline: they are near-zero by construction and would otherwise add
    # hundreds of unpredictable columns to every per-gene average.
    adata = drop_control_probes(adata, verbose=True)
    map_df = pd.read_csv(csv_path)
    if inside_he_only and "inside_he" in map_df.columns:
        keep = map_df["inside_he"].astype(str).str.lower().isin(["true", "1", "yes"])
        map_df = map_df.loc[keep].reset_index(drop=True)

    root = root or os.path.dirname(csv_path)
    paths = [p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))
             for p in map_df[path_col].astype(str)]
    map_df = map_df.assign(_abs_path=paths)
    exists = np.fromiter((os.path.isfile(p) for p in paths), dtype=bool, count=len(paths))
    if not exists.all():
        map_df = map_df.loc[exists].reset_index(drop=True)

    obs_pos = {_norm_id(v): i for i, v in enumerate(adata.obs_names.astype(str))}
    rows, keep_idx = [], []
    for i, sid in enumerate(map_df[id_col].astype(str)):
        j = obs_pos.get(_norm_id(sid))
        if j is not None:
            rows.append(j)
            keep_idx.append(i)
    map_df = map_df.iloc[keep_idx].reset_index(drop=True)
    adata = adata[rows].copy()

    counts = adata.X
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    counts = counts.astype(np.float32)

    adata.layers["counts"] = counts.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    expr = adata.X
    expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)
    expr = expr.astype(np.float32)

    gene_names = adata.var_names.astype(str).tolist()
    if n_top_genes is not None and n_top_genes < expr.shape[1]:
        # Rank by mean expression — deterministic and shared across baselines.
        order = np.argsort(-expr.mean(axis=0))[:n_top_genes]
        order = np.sort(order)
        expr, counts = expr[:, order], counts[:, order]
        gene_names = [gene_names[i] for i in order]

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    return dict(expr=expr, counts=counts, coords=coords,
                paths=map_df["_abs_path"].tolist(),
                ids=map_df[id_col].astype(str).tolist(),
                gene_names=gene_names)


def restrict_to_shared_genes(train: dict, test: dict):
    """
    Intersect two sections' panels and reorder both to the same gene list.

    Cross-section evaluation is only meaningful on genes both panels measure;
    the paired Xenium sections here range from an identical 377-gene panel
    (skin1/skin2, tonsil) to a 480-vs-5001 overlap (lung). Sorting the
    intersection makes the column order reproducible across methods.
    """
    shared = sorted(set(train["gene_names"]) & set(test["gene_names"]))
    if not shared:
        raise ValueError("train and test sections share no genes")
    for tab in (train, test):
        pos = {g: i for i, g in enumerate(tab["gene_names"])}
        cols = np.array([pos[g] for g in shared])
        tab["expr"] = tab["expr"][:, cols]
        tab["counts"] = tab["counts"][:, cols]
        tab["gene_names"] = list(shared)
    return train, test


def load_pair(args, id_col="cell_id", path_col="patch_path"):
    """
    Load the training section and, if given, the paired held-out section.

    Returns (train_tab, test_tab_or_None). When a test section is supplied the
    panels are intersected, so `n_genes` reflects the shared panel.
    """
    train = load_xenium_table(args.h5ad, args.csv, args.root, id_col, path_col,
                              getattr(args, "n_genes", None),
                              getattr(args, "inside_he_only", False))
    test = None
    if getattr(args, "test_h5ad", None):
        test = load_xenium_table(args.test_h5ad, args.test_csv,
                                 getattr(args, "test_root", None) or args.root,
                                 id_col, path_col, None,
                                 getattr(args, "inside_he_only", False))
        train, test = restrict_to_shared_genes(train, test)
    return train, test


def spatial_block_split(coords: np.ndarray, val_frac: float = 0.2,
                        tile_um: float = 500.0, buffer_um: float = 50.0,
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checkerboard spatial split with a boundary buffer.

    A random cell-level split would put Delaunay neighbours on both sides, and
    a single straight cut biases tissue composition (one half can be all
    stroma). Tiles of ~500um are far larger than a 2-hop neighbourhood (~50um),
    so blocks are genuinely separated while composition stays balanced.

    Returns (train_idx, val_idx); cells inside the buffer belong to neither.
    """
    xy = coords[:, :2]
    tx = np.floor((xy[:, 0] - xy[:, 0].min()) / tile_um).astype(np.int64)
    ty = np.floor((xy[:, 1] - xy[:, 1].min()) / tile_um).astype(np.int64)
    checker = (tx + ty) % 2 == 1

    rng = np.random.default_rng(seed)
    tiles = np.unique(np.stack([tx, ty], 1)[checker], axis=0)
    n_val_tiles = max(1, int(round(len(tiles) * val_frac * 2)))
    chosen = set(map(tuple, tiles[rng.permutation(len(tiles))[:n_val_tiles]]))
    is_val = np.fromiter(((tx[i], ty[i]) in chosen for i in range(len(tx))),
                         dtype=bool, count=len(tx))

    # buffer: distance to the nearest tile edge
    dx = np.minimum((xy[:, 0] - xy[:, 0].min()) % tile_um,
                    tile_um - (xy[:, 0] - xy[:, 0].min()) % tile_um)
    dy = np.minimum((xy[:, 1] - xy[:, 1].min()) % tile_um,
                    tile_um - (xy[:, 1] - xy[:, 1].min()) % tile_um)
    in_buffer = (dx < buffer_um) | (dy < buffer_um)

    train_idx = np.where(~is_val & ~in_buffer)[0]
    val_idx = np.where(is_val & ~in_buffer)[0]
    return train_idx, val_idx


class PatchExprDataset(Dataset):
    """(patch tensor, expression vector, index). Backbone-agnostic."""

    def __init__(self, paths: Sequence[str], expr: np.ndarray,
                 indices: Optional[Sequence[int]] = None,
                 transform=None, size: int = 224, counts: Optional[np.ndarray] = None):
        self.paths = list(paths)
        self.expr = expr
        self.counts = counts
        self.indices = np.arange(len(self.paths)) if indices is None else np.asarray(indices)
        self.size = int(size)
        if transform is None:
            from torchvision import transforms as T
            transform = T.Compose([
                T.Resize((self.size, self.size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        img = Image.open(self.paths[j]).convert("RGB")
        x = self.transform(img)
        y = torch.from_numpy(np.asarray(self.expr[j], dtype=np.float32))
        return x, y, j


class FeatureExprDataset(Dataset):
    """Precomputed-feature variant, for baselines that train on frozen CNN features."""

    def __init__(self, feats: np.ndarray, expr: np.ndarray,
                 indices: Optional[Sequence[int]] = None):
        self.feats, self.expr = feats, expr
        self.indices = np.arange(len(feats)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return (torch.from_numpy(np.asarray(self.feats[j], dtype=np.float32)),
                torch.from_numpy(np.asarray(self.expr[j], dtype=np.float32)), j)
