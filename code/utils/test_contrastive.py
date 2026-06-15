#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare representations before/after contrastive training via UMAP.

Example:
CUDA_VISIBLE_DEVICES=1 \
/data/project/banana9903_xenium/.conda-envs/SPAM/bin/python /data/project/banana9903_xenium/Code_final/code/utils/test_contrastive.py \
--h5ad /data/project/banana9903_xenium/h5ads/skin_melanoma.h5ad \
--csv /data/project/banana9903_xenium/contrast_test/expression/human_skin_melanoma_patches_256/spot_patch_map_256.csv \
--enc_name uni_v1 \
--proj_dim 256 \
--ckpt_dir /data/project/banana9903_xenium/ckpts_human_skin_melanoma \
--out_dir /data/project/banana9903_xenium/contrast_test/umap_compare \
--device cuda:0 \
--batch_size 128 \
--num_workers 128 \
--n_top_genes 541 \
--leiden \
--leiden_ref gene_base \
--leiden_ref_resolution 1.0 \
--save_embeddings

"""

import os
import sys
import argparse
import random
from typing import Optional, Dict

import numpy as np
import torch
from tqdm import tqdm

import scanpy as sc
import anndata as ad


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # /Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 표준모듈 'code'와 충돌 방지: 이미 로드돼 있으면 제거
sys.modules.pop("code", None)

from code.models.Foundations import inf_encoder_factory
from code.models.gene_encoder import GeneEncoder
from code.utils.contrastive import ImageFoundationAdapter
from code.utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_only_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _ensure_numpy(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _get_param_dtypes(m: torch.nn.Module) -> set:
    return {p.dtype for p in m.parameters()}


def _normalize_encoder_precision(foundation):
    dtypes = _get_param_dtypes(foundation.encoder)
    if len(dtypes) > 1:
        print(f"[WARN] Mixed parameter dtypes in encoder {dtypes}. Casting to float32.")
        foundation.encoder = foundation.encoder.float()
        dtypes = {torch.float32}
    foundation.precision = next(iter(dtypes)) if dtypes else torch.float32


@torch.no_grad()
def encode_images_foundation(
    foundation,
    img_loader,
    device: torch.device,
) -> np.ndarray:
    foundation.eval()
    foundation.to(device)
    out_dim = foundation.get_output_dim()
    n = len(img_loader.dataset)
    feats = np.zeros((n, out_dim), dtype=np.float32)
    for imgs, idxs in tqdm(img_loader, desc="Encode images (base)", leave=False):
        dtype = getattr(foundation, "precision", torch.float32)
        imgs = imgs.to(device=device, dtype=dtype, non_blocking=True)
        z = foundation.forward_features(imgs)
        z = z.detach().float().cpu().numpy()
        feats[idxs.numpy()] = z
    return feats


@torch.no_grad()
def encode_images_contrastive(
    img_encoder: ImageFoundationAdapter,
    img_loader,
    device: torch.device,
) -> np.ndarray:
    img_encoder.eval()
    img_encoder.to(device)
    out_dim = int(getattr(img_encoder, "proj_dim", 0)) or None
    n = len(img_loader.dataset)
    if out_dim is None:
        raise ValueError("Cannot infer output dim from img_encoder.")
    feats = np.zeros((n, out_dim), dtype=np.float32)
    for imgs, idxs in tqdm(img_loader, desc="Encode images (contrastive)", leave=False):
        imgs = imgs.to(device=device, non_blocking=True)
        z = img_encoder(imgs)
        z = z.detach().float().cpu().numpy()
        feats[idxs.numpy()] = z
    return feats


@torch.no_grad()
def encode_genes_contrastive(
    gene_encoder: GeneEncoder,
    gene_mat: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    gene_encoder.eval()
    gene_encoder.to(device)
    n, d = gene_mat.shape
    feats = np.zeros((n, int(gene_encoder.out_dim)), dtype=np.float32)
    for i in tqdm(range(0, n, batch_size), desc="Encode genes (contrastive)", leave=False):
        x = torch.from_numpy(gene_mat[i:i + batch_size]).to(device=device, dtype=torch.float32)
        z = gene_encoder(x)
        z = z.detach().float().cpu().numpy()
        feats[i:i + batch_size] = z
    return feats


def run_umap(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
    pca_n_comps: Optional[int] = 50,
) -> np.ndarray:
    if X.shape[0] < 2:
        raise ValueError(f"UMAP needs at least 2 samples, got n={X.shape[0]}.")
    adata = ad.AnnData(X)
    if pca_n_comps is not None and X.shape[1] > pca_n_comps:
        sc.pp.pca(adata, n_comps=pca_n_comps, svd_solver="arpack", random_state=seed)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric, use_rep="X_pca")
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric)
    sc.tl.umap(adata, min_dist=min_dist, random_state=seed)
    return np.asarray(adata.obsm["X_umap"], dtype=np.float32)


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


def leiden_with_target_clusters(
    X: np.ndarray,
    target_n: int,
    n_neighbors: int,
    metric: str,
    seed: int,
    pca_n_comps: Optional[int] = 50,
    res_min: float = 0.01,
    res_max: float = 5.0,
    max_iter: int = 20,
    key_added: str = "leiden",
) -> np.ndarray:
    if target_n < 1:
        raise ValueError(f"target_n must be >=1, got {target_n}")
    adata = _build_neighbors(X, n_neighbors=n_neighbors, metric=metric, seed=seed, pca_n_comps=pca_n_comps)

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


def leiden_with_resolution(
    X: np.ndarray,
    resolution: float,
    n_neighbors: int,
    metric: str,
    seed: int,
    pca_n_comps: Optional[int] = 50,
    key_added: str = "leiden",
) -> np.ndarray:
    adata = _build_neighbors(X, n_neighbors=n_neighbors, metric=metric, seed=seed, pca_n_comps=pca_n_comps)
    sc.tl.leiden(adata, resolution=float(resolution), key_added=key_added, random_state=seed)
    return adata.obs[key_added].astype(str).to_numpy()


def save_umap_plot_with_labels(
    umap_xy: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    point_size: float = 2.0,
):
    import matplotlib.pyplot as plt

    labels = np.asarray(labels)
    uniq = np.unique(labels)
    lut = {k: i for i, k in enumerate(uniq)}
    colors = np.vectorize(lut.get)(labels)
    plt.figure(figsize=(6, 6))
    plt.scatter(umap_xy[:, 0], umap_xy[:, 1], c=colors, s=point_size, linewidths=0, alpha=0.9, cmap="tab20")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_umap_plot(
    umap_xy: np.ndarray,
    title: str,
    out_path: str,
    point_size: float = 2.0,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(umap_xy[:, 0], umap_xy[:, 1], s=point_size, linewidths=0, alpha=0.9)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None, help="CSV patch_path가 상대경로면 합칠 루트")
    p.add_argument("--csv_id_col", default="cell_id", help="CSV id column (default: cell_id)")
    p.add_argument("--csv_path_col", default="patch_path", help="CSV path column (default: patch_path)")
    p.add_argument("--use_hvg", action="store_true", help="HVG로 유전자 선택 (기본은 평균상위)")
    p.add_argument("--n_top_genes", type=int, default=541)

    # model / weights
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--ckpt_dir", default=None, help="contrastive ckpt dir (Image_pretrained.pt, SNN_pretrained.pt)")
    p.add_argument("--foundation_weights", default=None, help="optional foundation weights path")

    # compute
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gene_batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)

    # umap
    p.add_argument("--umap_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist", type=float, default=0.3)
    p.add_argument("--umap_metric", default="cosine")
    p.add_argument("--umap_pca_comps", type=int, default=50)
    p.add_argument("--point_size", type=float, default=2.0)

    # leiden
    p.add_argument("--leiden", action="store_true", help="run Leiden clustering and color UMAPs")
    p.add_argument("--leiden_ref", default="gene_base",
                   choices=["image_base", "gene_base", "image_contrastive", "gene_contrastive"])
    p.add_argument("--leiden_ref_resolution", type=float, default=1.0,
                   help="resolution to compute target cluster count from ref embedding")
    p.add_argument("--leiden_n_clusters", type=int, default=None,
                   help="if set, force target cluster count for all embeddings")
    p.add_argument("--leiden_res_min", type=float, default=0.01)
    p.add_argument("--leiden_res_max", type=float, default=5.0)
    p.add_argument("--leiden_max_iter", type=int, default=20)

    # output
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_embeddings", action="store_true", help="save embeddings/umap as .npy")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)

    # 1) Load data + align
    adata = load_h5ad_with_preproc(
        args.h5ad,
        use_hvg=args.use_hvg,
        n_top_genes=args.n_top_genes,
        x_key="x_centroid",
        y_key="y_centroid",
    )
    root_dir = args.root or os.path.dirname(args.csv)
    map_df = load_patch_mapping_csv_xenium(
        args.csv,
        root_dir=root_dir,
        id_col=args.csv_id_col,
        path_col=args.csv_path_col,
        drop_missing=True,
    )
    if len(map_df) == 0 and args.root is None:
        # Common case: patch_path already contains the folder name; default root joins it twice.
        alt_root = os.path.dirname(root_dir)
        map_df = load_patch_mapping_csv_xenium(
            args.csv,
            root_dir=alt_root,
            id_col=args.csv_id_col,
            path_col=args.csv_path_col,
            drop_missing=True,
        )
        if len(map_df) == 0:
            raise RuntimeError(
                "All patch paths are missing after auto root fallback. "
                f"Tried root_dir={root_dir} and alt_root={alt_root}. "
                "Pass the correct --root explicitly."
            )
    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, _ = reorder_and_prune_by_obs(
        adata=adata, map_df=map_df, id_col="spot_id", coord_key="spatial"
    )
    if len(map_df_aligned) == 0:
        obs_head = adata.obs_names.astype(str).tolist()[:5]
        map_head = map_df["spot_id"].astype(str).tolist()[:5]
        raise RuntimeError(
            "No overlapping spot_ids between CSV and h5ad. "
            f"Check id columns. obs_names head={obs_head}, csv spot_id head={map_head}. "
            "If CSV id column is not 'cell_id', pass --csv_id_col."
        )

    # ordered obs
    spot_ids = map_df_aligned["spot_id"].astype(str).tolist()
    adata_sub = adata[spot_ids].copy()

    # 2) Base gene matrix (from h5ad)
    if "selected" in adata.var.columns:
        sel = adata.var["selected"].values
        gene_base = _ensure_numpy(adata_sub[:, sel].X)
    else:
        gene_base = _ensure_numpy(adata_sub.X)

    # 3) Foundation + image transforms
    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation = FoundationCls(weights_path=args.foundation_weights)
    _normalize_encoder_precision(foundation)
    image_tf = getattr(foundation, "eval_transforms", None)

    img_loader = make_img_only_dataloader(
        map_df=map_df_aligned,
        batch_size=max(1, args.batch_size),
        num_workers=max(0, args.num_workers),
        image_transform=image_tf,
        shuffle=False,
    )

    # 4) Encode base image reps
    image_base = encode_images_foundation(foundation, img_loader, device)

    # 5) Contrastive encoders (optional)
    image_contrastive = None
    gene_contrastive = None
    if args.ckpt_dir:
        image_ckpt = os.path.join(args.ckpt_dir, "Image_pretrained.pt")
        if not os.path.isfile(image_ckpt):
            raise FileNotFoundError(f"Image checkpoint not found: {image_ckpt}")

        img_encoder = ImageFoundationAdapter(foundation, proj_dim=args.proj_dim, normalize=False)
        state = torch.load(image_ckpt, map_location="cpu")
        img_encoder.load_state_dict(state, strict=True)
        image_contrastive = encode_images_contrastive(img_encoder, img_loader, device)

        gene_ckpt = os.path.join(args.ckpt_dir, "SNN_pretrained.pt")
        if os.path.isfile(gene_ckpt):
            gene_encoder = GeneEncoder(
                input_dim=gene_base.shape[1],
                model_size_omic="small",
                proj_dim=args.proj_dim,
                normalize=False,
            )
            gstate = torch.load(gene_ckpt, map_location="cpu")
            gene_encoder.load_state_dict(gstate, strict=True)
            gene_contrastive = encode_genes_contrastive(
                gene_encoder, gene_base, device, batch_size=args.gene_batch_size
            )
        else:
            print(f"[WARN] Gene checkpoint not found, skip contrastive gene: {gene_ckpt}")

    # 6) UMAP + save
    outputs: Dict[str, np.ndarray] = {}
    outputs["umap_image_base"] = run_umap(
        image_base,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
        pca_n_comps=args.umap_pca_comps,
    )
    save_umap_plot(
        outputs["umap_image_base"],
        title="Image UMAP (base)",
        out_path=os.path.join(args.out_dir, "umap_image_base.png"),
        point_size=args.point_size,
    )

    outputs["umap_gene_base"] = run_umap(
        gene_base,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
        pca_n_comps=args.umap_pca_comps,
    )
    save_umap_plot(
        outputs["umap_gene_base"],
        title="H5AD UMAP (base)",
        out_path=os.path.join(args.out_dir, "umap_gene_base.png"),
        point_size=args.point_size,
    )

    if image_contrastive is not None:
        outputs["umap_image_contrastive"] = run_umap(
            image_contrastive,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            seed=args.seed,
            pca_n_comps=args.umap_pca_comps,
        )
        save_umap_plot(
            outputs["umap_image_contrastive"],
            title="Image UMAP (contrastive)",
            out_path=os.path.join(args.out_dir, "umap_image_contrastive.png"),
            point_size=args.point_size,
        )

    if gene_contrastive is not None:
        outputs["umap_gene_contrastive"] = run_umap(
            gene_contrastive,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            seed=args.seed,
            pca_n_comps=args.umap_pca_comps,
        )
        save_umap_plot(
            outputs["umap_gene_contrastive"],
            title="H5AD UMAP (contrastive)",
            out_path=os.path.join(args.out_dir, "umap_gene_contrastive.png"),
            point_size=args.point_size,
        )

    # Leiden clustering with matched cluster count
    if args.leiden:
        emb_map = {
            "image_base": image_base,
            "gene_base": gene_base,
        }
        if image_contrastive is not None:
            emb_map["image_contrastive"] = image_contrastive
        if gene_contrastive is not None:
            emb_map["gene_contrastive"] = gene_contrastive

        if args.leiden_ref not in emb_map:
            raise RuntimeError(f"leiden_ref '{args.leiden_ref}' not available (missing embedding).")

        if args.leiden_n_clusters is not None:
            target_n = int(args.leiden_n_clusters)
            ref_labels = None
        else:
            ref_labels = leiden_with_resolution(
                emb_map[args.leiden_ref],
                resolution=args.leiden_ref_resolution,
                n_neighbors=args.umap_neighbors,
                metric=args.umap_metric,
                seed=args.seed,
                pca_n_comps=args.umap_pca_comps,
            )
            target_n = int(np.unique(ref_labels).size)

        leiden_labels = {}
        for name, emb in emb_map.items():
            if ref_labels is not None and name == args.leiden_ref:
                labels = ref_labels
            else:
                labels = leiden_with_target_clusters(
                    emb,
                    target_n=target_n,
                    n_neighbors=args.umap_neighbors,
                    metric=args.umap_metric,
                    seed=args.seed,
                    pca_n_comps=args.umap_pca_comps,
                    res_min=args.leiden_res_min,
                    res_max=args.leiden_res_max,
                    max_iter=args.leiden_max_iter,
                )
            leiden_labels[name] = labels

        # save colored UMAPs + labels
        for name, labels in leiden_labels.items():
            umap_key = f"umap_{name}"
            if umap_key in outputs:
                save_umap_plot_with_labels(
                    outputs[umap_key],
                    labels,
                    title=f"{name.replace('_', ' ').title()} UMAP (Leiden, k={target_n})",
                    out_path=os.path.join(args.out_dir, f"umap_{name}_leiden.png"),
                    point_size=args.point_size,
                )
            # label table
            out_csv = os.path.join(args.out_dir, f"leiden_{name}.csv")
            with open(out_csv, "w", encoding="utf-8") as f:
                f.write("spot_id,leiden\n")
                for sid, lab in zip(spot_ids, labels):
                    f.write(f"{sid},{lab}\n")

    if args.save_embeddings:
        np.save(os.path.join(args.out_dir, "emb_image_base.npy"), image_base)
        np.save(os.path.join(args.out_dir, "emb_gene_base.npy"), gene_base)
        if image_contrastive is not None:
            np.save(os.path.join(args.out_dir, "emb_image_contrastive.npy"), image_contrastive)
        if gene_contrastive is not None:
            np.save(os.path.join(args.out_dir, "emb_gene_contrastive.npy"), gene_contrastive)
        for k, v in outputs.items():
            np.save(os.path.join(args.out_dir, f"{k}.npy"), v)

    # Grid plot (when all 4 exist)
    if image_contrastive is not None and gene_contrastive is not None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()
        titles = [
            ("Image UMAP (base)", outputs["umap_image_base"]),
            ("H5AD UMAP (base)", outputs["umap_gene_base"]),
            ("Image UMAP (contrastive)", outputs["umap_image_contrastive"]),
            ("H5AD UMAP (contrastive)", outputs["umap_gene_contrastive"]),
        ]
        for ax, (title, umap_xy) in zip(axes, titles):
            ax.scatter(umap_xy[:, 0], umap_xy[:, 1], s=args.point_size, linewidths=0, alpha=0.9)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "umap_grid.png"), dpi=300)
        plt.close()

    print(f"[Done] UMAP outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
