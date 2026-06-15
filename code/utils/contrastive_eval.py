#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate base vs contrastive embeddings with:
1) kNN overlap within modality (base vs contrastive)
2) Leiden stability (ARI vs reference labels across seeds, optional subsample)
3) Cross-modality kNN agreement (image vs gene)

Expected inputs:
- Embeddings saved by test_contrastive.py with --save_embeddings:
  emb_image_base.npy, emb_image_contrastive.npy, emb_gene_base.npy, emb_gene_contrastive.npy
- Leiden labels saved by test_contrastive.py with --leiden:
  leiden_image_base.csv, leiden_image_contrastive.csv, leiden_gene_base.csv, leiden_gene_contrastive.csv
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import scanpy as sc
import anndata as ad


EMB_NAMES = [
    "image_base",
    "image_contrastive",
    "gene_base",
    "gene_contrastive",
]


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x


def _load_embedding(emb_dir: Path, name: str) -> Optional[np.ndarray]:
    path = emb_dir / f"emb_{name}.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    return _ensure_float32(arr)


def _load_leiden_csv(label_dir: Path, name: str) -> Optional[Tuple[List[str], np.ndarray]]:
    path = label_dir / f"leiden_{name}.csv"
    if not path.exists():
        return None
    spot_ids: List[str] = []
    labels: List[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spot_ids.append(row["spot_id"])
            labels.append(row["leiden"])
    return spot_ids, np.asarray(labels, dtype=object)


def _align_labels_to_order(
    spot_ids_ref: List[str],
    labels_ref: np.ndarray,
    spot_ids_target: List[str],
) -> np.ndarray:
    if spot_ids_ref == spot_ids_target:
        return labels_ref
    index = {sid: i for i, sid in enumerate(spot_ids_ref)}
    out = np.empty(len(spot_ids_target), dtype=object)
    for i, sid in enumerate(spot_ids_target):
        out[i] = labels_ref[index[sid]]
    return out


def _build_neighbors(
    X: np.ndarray,
    n_neighbors: int,
    metric: str,
    seed: int,
    pca_n_comps: Optional[int] = 50,
) -> ad.AnnData:
    adata = ad.AnnData(X)
    if pca_n_comps is not None and X.shape[1] > pca_n_comps:
        sc.pp.pca(adata, n_comps=pca_n_comps, svd_solver="arpack", random_state=seed)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric, use_rep="X_pca")
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric)
    return adata


def _leiden_with_target_clusters_on_adata(
    adata: ad.AnnData,
    target_n: int,
    seed: int,
    res_min: float,
    res_max: float,
    max_iter: int,
    key_added: str = "leiden",
) -> np.ndarray:
    def _count(res: float) -> int:
        sc.tl.leiden(adata, resolution=float(res), key_added=key_added, random_state=seed)
        return int(adata.obs[key_added].nunique())

    best_res, best_k, best_labels = None, None, None
    lo, hi = float(res_min), float(res_max)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        k = _count(mid)
        labels = adata.obs[key_added].astype(str).to_numpy()
        if best_k is None or abs(k - target_n) < abs(best_k - target_n):
            best_k, best_res, best_labels = k, mid, labels
        if k == target_n:
            break
        if k < target_n:
            lo = mid
        else:
            hi = mid
    return best_labels


def _ari(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    nij = {}
    ai = {}
    bj = {}
    for x, y in zip(a, b):
        nij[(x, y)] = nij.get((x, y), 0) + 1
        ai[x] = ai.get(x, 0) + 1
        bj[y] = bj.get(y, 0) + 1

    def comb2(k: int) -> float:
        return k * (k - 1) / 2

    sum_nij = sum(comb2(v) for v in nij.values())
    sum_ai = sum(comb2(v) for v in ai.values())
    sum_bj = sum(comb2(v) for v in bj.values())
    comb_n = comb2(n)
    if comb_n == 0:
        return 0.0
    expected = (sum_ai * sum_bj) / comb_n
    max_index = 0.5 * (sum_ai + sum_bj)
    denom = max_index - expected
    return (sum_nij - expected) / denom if denom != 0 else 0.0


def _get_knn_indices(
    X: np.ndarray,
    k: int,
    metric: str,
    backend: str = "auto",
) -> np.ndarray:
    # Try sklearn first (most common in this env)
    if backend in ("auto", "sklearn"):
        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception:
            if backend == "sklearn":
                raise
        else:
            nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
            nn.fit(X)
            _, inds = nn.kneighbors(X, return_distance=True)
            # drop self if present
            if np.all(inds[:, 0] == np.arange(X.shape[0])):
                inds = inds[:, 1:k + 1]
            else:
                inds = inds[:, :k]
            return inds

    # Fallback: faiss
    if backend in ("auto", "faiss"):
        try:
            import faiss
        except Exception:
            if backend == "faiss":
                raise
        else:
            Xf = _ensure_float32(X)
            if metric == "cosine":
                norms = np.linalg.norm(Xf, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                Xf = Xf / norms
                index = faiss.IndexFlatIP(Xf.shape[1])
            else:
                index = faiss.IndexFlatL2(Xf.shape[1])
            index.add(Xf)
            _, inds = index.search(Xf, k + 1)
            if np.all(inds[:, 0] == np.arange(Xf.shape[0])):
                inds = inds[:, 1:k + 1]
            else:
                inds = inds[:, :k]
            return inds

    raise RuntimeError("No kNN backend available. Install scikit-learn or faiss.")


def _mean_overlap(idx_a: np.ndarray, idx_b: np.ndarray, k: int) -> float:
    n = idx_a.shape[0]
    total = 0
    for i in range(n):
        a = idx_a[i, :k]
        bset = set(idx_b[i, :k].tolist())
        total += sum(1 for x in a if x in bset)
    return total / (n * k)


def _save_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, help="Output directory for metrics/plots")
    p.add_argument("--emb_dir", default=None, help="Directory with emb_*.npy (default: out_dir)")
    p.add_argument("--leiden_dir", default=None, help="Directory with leiden_*.csv (default: out_dir)")

    p.add_argument("--k_list", default="15,30,50", help="Comma-separated k values")
    p.add_argument("--knn_metric", default="cosine")
    p.add_argument("--knn_backend", default="auto", choices=["auto", "sklearn", "faiss"])

    p.add_argument("--stability_seeds", default="0,1,2,3,4")
    p.add_argument("--stability_subsample", type=int, default=20000,
                   help="Subsample size for stability (0 or >=N uses full)")
    p.add_argument("--leiden_neighbors", type=int, default=30)
    p.add_argument("--leiden_metric", default="cosine")
    p.add_argument("--leiden_pca_comps", type=int, default=50)
    p.add_argument("--leiden_res_min", type=float, default=0.01)
    p.add_argument("--leiden_res_max", type=float, default=5.0)
    p.add_argument("--leiden_max_iter", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.emb_dir) if args.emb_dir else out_dir
    label_dir = Path(args.leiden_dir) if args.leiden_dir else out_dir

    k_list = _parse_int_list(args.k_list)
    if not k_list:
        raise ValueError("--k_list must contain at least one integer")
    k_max = max(k_list)
    stability_seeds = _parse_int_list(args.stability_seeds)
    if not stability_seeds:
        raise ValueError("--stability_seeds must contain at least one integer")

    # Load embeddings
    embeddings: Dict[str, np.ndarray] = {}
    for name in EMB_NAMES:
        arr = _load_embedding(emb_dir, name)
        if arr is not None:
            embeddings[name] = arr
    if not embeddings:
        raise FileNotFoundError(
            "No embeddings found. Re-run test_contrastive.py with --save_embeddings."
        )

    # Load Leiden labels (if available)
    labels: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for name in EMB_NAMES:
        item = _load_leiden_csv(label_dir, name)
        if item is not None:
            labels[name] = item

    # kNN overlap: base vs contrastive
    knn_rows: List[Dict[str, object]] = []
    knn_cache: Dict[str, np.ndarray] = {}
    for name, emb in embeddings.items():
        knn_cache[name] = _get_knn_indices(
            emb, k=k_max, metric=args.knn_metric, backend=args.knn_backend
        )

    def _knn_overlap_pair(name_a: str, name_b: str) -> Optional[Dict[int, float]]:
        if name_a not in knn_cache or name_b not in knn_cache:
            return None
        idx_a = knn_cache[name_a]
        idx_b = knn_cache[name_b]
        return {k: _mean_overlap(idx_a, idx_b, k) for k in k_list}

    # within-modality base vs contrastive
    within_pairs = [
        ("image_base", "image_contrastive"),
        ("gene_base", "gene_contrastive"),
    ]
    within_results = {}
    for a, b in within_pairs:
        res = _knn_overlap_pair(a, b)
        if res is None:
            continue
        within_results[(a, b)] = res
        for k, val in res.items():
            knn_rows.append(
                {"pair": f"{a} vs {b}", "k": k, "overlap": float(val)}
            )

    # cross-modality agreement
    cross_pairs = [
        ("image_base", "gene_base"),
        ("image_contrastive", "gene_contrastive"),
    ]
    cross_results = {}
    for a, b in cross_pairs:
        res = _knn_overlap_pair(a, b)
        if res is None:
            continue
        cross_results[(a, b)] = res
        for k, val in res.items():
            knn_rows.append(
                {"pair": f"{a} vs {b}", "k": k, "overlap": float(val)}
            )

    _save_csv(knn_rows, out_dir / "knn_overlap.csv")

    # Stability: ARI vs reference labels across seeds
    stability_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(args.seed)

    for name, emb in embeddings.items():
        if name not in labels:
            continue
        spot_ids, ref_labels = labels[name]
        n = emb.shape[0]
        if len(ref_labels) != n:
            # align by spot_id order if possible
            spot_ids_ref, ref_labels_ref = labels[name]
            ref_labels = _align_labels_to_order(spot_ids_ref, ref_labels_ref, spot_ids)

        if args.stability_subsample and args.stability_subsample < n:
            sub_idx = rng.choice(n, size=args.stability_subsample, replace=False)
            sub_idx.sort()
        else:
            sub_idx = np.arange(n)

        X_sub = emb[sub_idx]
        ref_sub = ref_labels[sub_idx].tolist()
        target_n = int(len(set(ref_sub)))

        adata = _build_neighbors(
            X_sub,
            n_neighbors=args.leiden_neighbors,
            metric=args.leiden_metric,
            seed=args.seed,
            pca_n_comps=args.leiden_pca_comps,
        )
        for seed in stability_seeds:
            pred = _leiden_with_target_clusters_on_adata(
                adata,
                target_n=target_n,
                seed=seed,
                res_min=args.leiden_res_min,
                res_max=args.leiden_res_max,
                max_iter=args.leiden_max_iter,
            )
            ari = _ari(ref_sub, pred.tolist())
            stability_rows.append(
                {"embedding": name, "seed": seed, "ari_vs_ref": float(ari)}
            )

    _save_csv(stability_rows, out_dir / "stability_ari.csv")

    # Plots
    import matplotlib.pyplot as plt

    if knn_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes = axes.ravel()

        # within-modality
        ax = axes[0]
        for (a, b), res in within_results.items():
            ks = sorted(res.keys())
            vals = [res[k] for k in ks]
            ax.plot(ks, vals, marker="o", label=f"{a} vs {b}")
        ax.set_title("Within-modality kNN overlap")
        ax.set_xlabel("k")
        ax.set_ylabel("Mean overlap")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

        # cross-modality
        ax = axes[1]
        for (a, b), res in cross_results.items():
            ks = sorted(res.keys())
            vals = [res[k] for k in ks]
            ax.plot(ks, vals, marker="o", label=f"{a} vs {b}")
        ax.set_title("Cross-modality kNN agreement")
        ax.set_xlabel("k")
        ax.set_ylabel("Mean overlap")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(out_dir / "knn_overlap.png", dpi=200)
        plt.close()

    if stability_rows:
        # simple box plot with scatter
        by_name: Dict[str, List[float]] = {}
        for row in stability_rows:
            by_name.setdefault(row["embedding"], []).append(float(row["ari_vs_ref"]))

        names = sorted(by_name.keys())
        data = [by_name[n] for n in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(data, labels=names, showfliers=False)
        for i, vals in enumerate(data, start=1):
            x = np.random.normal(i, 0.04, size=len(vals))
            ax.scatter(x, vals, s=10, alpha=0.7)
        ax.set_title("Leiden stability (ARI vs reference)")
        ax.set_ylabel("ARI")
        ax.set_ylim(-0.05, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / "stability_ari.png", dpi=200)
        plt.close()

    print(f"[Done] Saved metrics to: {out_dir}")


if __name__ == "__main__":
    main()
