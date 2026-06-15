#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_inference_xattn_withtrainclustercentroid.py
- xattn 기반으로 파인튜닝된 Cross-Attn + ZINB 모델 추론 스크립트
- 블록별 ckpt(Image/SNN/Coord/XATTN) 또는 전체 ft model ckpt 로드 지원
- gene 입력 소스:
  - mean: train 데이터의 gene-wise mean
  - train_cluster_centroid: train cluster centroid 평균 기반 벡터
  - similarity: 이미지 임베딩 유사도 기반 매칭
  - celltype_similarity: celltype-aware similarity 매칭(명시적)

예시:

CUDA_VISIBLE_DEVICES=1 python /data/project/banana9903_xenium/Code_final/code/utils/run_inference_xattn_withtrainclustercentroid.py \
  --h5ad /data/project/banana9903_xenium/h5ads/ovarian_cancer_merged_with_celltype_symbol.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/ov_cancer_patches_256/spot_patch_map_256.csv \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --id_col cell_id --path_col patch_path \
  --train_h5ad /data/project/banana9903_xenium/h5ads/ovarian_cancer_merged_with_celltype_symbol.h5ad \
  --train_csv /data/project/banana9903_xenium/contrast_test/expression/ov_cancer_patches_256spot_patch_map_256.csv \
  --enc_name uni_v1 \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \
  --graph_type delaunay --k 12 \
  --img_tokens 10 --gene_tokens 10 --spot_tokens 10 \
  --ft_model_ckpt /data/project/banana9903_xenium/ckpts_cross_human_ovarian_cancer_xattn_delaunay_10tokens_with_celltype_softcluster_retrieval5_0224/best/model.pt \
  --expr_source train_cluster_centroid \
  --cluster_col leiden \
  --cluster_centroid_agg size_weighted \
  --batch_size 256 --num_workers 4 \
  --out_h5ad /data/project/banana9903_xenium/inference/0303_ovariancancer_recovered_xattn_retrieval_rawexpression.h5ad \
  --celltype_meta_pt /data/project/banana9903_xenium/ckpts_cross_human_ovarian_cancer_xattn_delaunay_10tokens_with_celltype_softcluster_retrieval5_0224/best/celltype_meta.pt



token 있는거 사용!!
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import anndata as ad
from tqdm import tqdm
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 표준 모듈 'code' 이름 충돌 방지
sys.modules.pop("code", None)

# ---- 프로젝트 유틸/모델 ----
from code.models.Foundations import inf_encoder_factory
from code.models.GCN_update import SpatialGCN
from code.models.gene_encoder import GeneEncoder
from code.models.fusion import CrossModalFusion
from code.models.decoders import ZINBDecoder
from code.models.graph_construction import build_knn_graph
from code.models.graph_construction_delaunay import build_delaunay_graph

from code.utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_only_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)
from code.utils.image_representation_gathering import (
    predict_expression_by_image_similarity,
    ImageMapDataset,
    gather_image_representations,
    top1_cosine_match,
    _expression_matrix_for_spots,
)


class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn, dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1, merge="gated-sum",
                 img_tokens=1, spot_tokens=1, gene_tokens=1,
                 decoder_hidden=1024, gene_dim=1000,
                 num_celltypes: int = 0, cls_head_hidden: int = 0, cls_dropout: float = 0.0):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.gcn = gcn

        self.proj_img = nn.Linear(dim_img, dim_fuse)
        self.proj_gene = nn.Linear(dim_gene, dim_fuse)
        self.proj_spot = nn.Linear(dim_spot, dim_fuse)

        self.fusion = CrossModalFusion(
            dim_img=dim_fuse,
            dim_spot=dim_fuse,
            dim_gene=dim_fuse,
            dim_out=dim_fuse,
            num_heads=num_heads,
            dropout=dropout,
            merge=merge,
            img_tokens=img_tokens,
            spot_tokens=spot_tokens,
            gene_tokens=gene_tokens,
        )
        self.decoder = ZINBDecoder(
            in_dim=dim_fuse,
            gene_dim=gene_dim,
            hidden=decoder_hidden,
            dropout=dropout,
        )
        self.num_celltypes = int(num_celltypes)
        if self.num_celltypes > 0:
            if cls_head_hidden and cls_head_hidden > 0:
                self.cls_head = nn.Sequential(
                    nn.Linear(dim_fuse, cls_head_hidden),
                    nn.GELU(),
                    nn.Dropout(cls_dropout),
                    nn.Linear(cls_head_hidden, self.num_celltypes),
                )
            else:
                self.cls_head = nn.Linear(dim_fuse, self.num_celltypes)
        else:
            self.cls_head = None

    def forward(
        self,
        batch,
        spot_feat_in,
        edge_index,
        edge_weight=None,
        return_attn: bool = False,
        return_aux: bool = False,
    ):
        x_img = batch["image"]            # (B, C, H, W)
        x_gene = batch["expr"].float()    # (B, G)
        node_idx = batch["idx"].long()    # (B,)

        zi = self.img_encoder(x_img)
        zg = self.gene_encoder(x_gene)
        zs_all_nodes = self.gcn(spot_feat_in, edge_index, edge_weight)  # (N, Ds)
        zs = zs_all_nodes[node_idx]                                      # (B, Ds)

        zi = self.proj_img(zi)
        zg = self.proj_gene(zg)
        zs = self.proj_spot(zs)

        if return_attn:
            z_fused, _, _, attn = self.fusion(zi, zs, zg, return_attn=True)
        else:
            z_fused, _, _ = self.fusion(zi, zs, zg, return_attn=False)
            attn = None
        mu, theta, pi = self.decoder(z_fused)
        cls_logits = self.cls_head(z_fused) if self.cls_head is not None else None
        if return_aux:
            return mu, theta, pi, {"attn": attn, "cls_logits": cls_logits}
        if return_attn:
            return mu, theta, pi, attn
        return mu, theta, pi


def _load_sd(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and any(k in sd for k in ["state_dict", "model", "module"]):
        for k in ["state_dict", "model", "module"]:
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd if isinstance(sd, dict) else sd.state_dict()


def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_pretrained_blocks(foundation, gene_encoder, gcn,
                           pt_img_backbone=None, pt_ig=None, pt_is=None):
    if pt_img_backbone:
        sd = _load_sd(pt_img_backbone)
        for cand in [_strip_prefix(sd, "img_encoder.foundation."),
                     _strip_prefix(sd, "foundation."),
                     sd]:
            try:
                foundation.load_state_dict(cand, strict=False)
                print(f"[PT] foundation loaded from {pt_img_backbone} (strict=False)")
                break
            except Exception:
                pass

    if pt_ig:
        sd = _load_sd(pt_ig)
        for cand in [_strip_prefix(sd, "gene_encoder."), sd]:
            try:
                gene_encoder.load_state_dict(cand, strict=False)
                print(f"[PT] gene_encoder loaded from {pt_ig} (strict=False)")
                break
            except Exception:
                pass

    if pt_is:
        sd = _load_sd(pt_is)
        for cand in [_strip_prefix(sd, "gcn."), sd]:
            try:
                gcn.load_state_dict(cand, strict=False)
                print(f"[PT] gcn loaded from {pt_is} (strict=False)")
                break
            except Exception:
                pass


def load_xattn_into_fusion(fusion: CrossModalFusion, xattn_path: str | None):
    if not xattn_path:
        return
    sd = _load_sd(xattn_path)

    def _load_block(dst_block: nn.Module, src_prefix: str):
        sub = {k[len(src_prefix) + 1:]: v for k, v in sd.items() if k.startswith(src_prefix + ".")}
        if not sub:
            print(f"[XATTN][warn] no keys found for {src_prefix} in {xattn_path}")
            return
        ret = dst_block.load_state_dict(sub, strict=False)
        missing = len(ret.missing_keys)
        unexpected = len(ret.unexpected_keys)
        if missing or unexpected:
            print(f"[XATTN][warn] {src_prefix} -> fusion (missing={missing}, unexpected={unexpected})")
        else:
            print(f"[XATTN] loaded {src_prefix} into fusion")

    _load_block(fusion.xa_ig, "i2g")
    _load_block(fusion.xa_is, "i2c")


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if ":" in device_arg:
            torch.cuda.set_device(int(device_arg.split(":")[1]))
        return torch.device(device_arg)
    return torch.device("cpu")


def pick_amp_dtype(amp: bool) -> torch.dtype:
    if not amp:
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if (major, minor) >= (8, 0) else torch.float16


def align_genes_for_inference(train_adata: ad.AnnData, eval_adata: ad.AnnData):
    common = eval_adata.var_names.intersection(train_adata.var_names)
    if len(common) == 0:
        raise ValueError("No overlapping genes between train and eval datasets.")
    eval_genes = [g for g in eval_adata.var_names if g in common]
    train_sub = train_adata[:, eval_genes].copy()
    eval_sub = eval_adata[:, eval_genes].copy()
    return train_sub, eval_sub


def build_spatial_graph(coords_np: np.ndarray, k: int, graph_type: str):
    graph_key = str(graph_type).strip().lower()
    if graph_key == "delaunay":
        return build_delaunay_graph(coords_np, k=k, use_weight=True)
    return build_knn_graph(coords_np, k=k, use_weight=True)


def _save_attn_npz(path: str, attn_map: dict | None, reduce_heads: bool, node_idx: np.ndarray | None = None):
    if attn_map is None:
        return
    payload = {}
    for k, v in attn_map.items():
        if v is None:
            continue
        x = v.detach().float().cpu()
        if reduce_heads and x.ndim == 4:
            x = x.mean(dim=1)
        payload[k] = x.numpy()
    if node_idx is not None:
        payload["node_idx"] = node_idx
    if payload:
        np.savez_compressed(path, **payload)


def _mean_gene_vector(adata: ad.AnnData) -> np.ndarray:
    m = adata.X.mean(axis=0)
    if hasattr(m, "A1"):
        return np.asarray(m.A1, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    if m.ndim > 1:
        m = m.reshape(-1)
    return m.astype(np.float32)


def _resolve_obs_col(adata: ad.AnnData, col: str | None) -> str | None:
    if col is None:
        return None
    cols = list(adata.obs.columns)
    if col in cols:
        return col
    low = {str(c).lower(): c for c in cols}
    return low.get(str(col).lower(), None)


def _resolve_cluster_col(adata: ad.AnnData, preferred: str | None) -> str:
    if preferred is not None:
        resolved = _resolve_obs_col(adata, preferred)
        if resolved is not None:
            return resolved

    candidates = [
        "leiden", "cluster", "clusters", "seurat_clusters", "Cluster",
        "cell_type", "celltype", "celltypes", "annotation", "annot",
    ]
    for cand in candidates:
        resolved = _resolve_obs_col(adata, cand)
        if resolved is not None:
            return resolved
    raise ValueError(
        f"cluster column not found in train_h5ad.obs. available={list(adata.obs.columns)}"
    )


def _build_cluster_centroid_reference_vector(
    adata: ad.AnnData,
    cluster_col: str,
    agg: str = "mean",
) -> tuple[np.ndarray, list[str], list[int], np.ndarray]:
    raw = adata.obs[cluster_col]
    valid_mask = ~raw.isna()
    if int(valid_mask.sum()) <= 0:
        raise ValueError(f"cluster column '{cluster_col}' has no valid labels.")

    labels = raw.loc[valid_mask].astype(str)
    cluster_names = sorted(pd.unique(labels).tolist())
    centroids = []
    cluster_sizes = []
    for cname in cluster_names:
        spot_ids = labels.index[labels == cname].tolist()
        if len(spot_ids) == 0:
            continue
        centroids.append(_mean_gene_vector(adata[spot_ids]))
        cluster_sizes.append(len(spot_ids))

    if len(centroids) == 0:
        raise ValueError(f"failed to build any cluster centroid from '{cluster_col}'.")

    centroid_mat = np.stack(centroids, axis=0).astype(np.float32)
    if agg == "size_weighted":
        weights = np.asarray(cluster_sizes, dtype=np.float32)
        weights = weights / max(weights.sum(), 1.0)
    else:
        weights = np.ones(len(centroids), dtype=np.float32) / float(len(centroids))

    ref_vec = (centroid_mat * weights[:, None]).sum(axis=0).astype(np.float32)
    return ref_vec, cluster_names, cluster_sizes, weights


def _load_celltype_meta(meta_path: str | None) -> dict | None:
    if meta_path is None or not os.path.isfile(meta_path):
        return None
    try:
        obj = torch.load(meta_path, map_location="cpu")
    except Exception as e:
        print(f"[CT][warn] failed to load celltype_meta: {meta_path} ({e})")
        return None
    if not isinstance(obj, dict):
        print(f"[CT][warn] unexpected celltype_meta format: {type(obj)}")
        return None
    return obj


def _infer_cls_head_config(sd: dict | None) -> tuple[int, int]:
    if not isinstance(sd, dict):
        return 0, 0
    if "cls_head.weight" in sd and getattr(sd["cls_head.weight"], "ndim", 0) == 2:
        num_ct = int(sd["cls_head.weight"].shape[0])
        return num_ct, 0
    if "cls_head.0.weight" in sd and "cls_head.3.weight" in sd:
        w0 = sd["cls_head.0.weight"]
        w3 = sd["cls_head.3.weight"]
        if getattr(w0, "ndim", 0) == 2 and getattr(w3, "ndim", 0) == 2:
            return int(w3.shape[0]), int(w0.shape[0])
    return 0, 0


@torch.no_grad()
def _predict_celltype_probs_from_base_expr(
    model: nn.Module,
    img_loader,
    spot_feat_in: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    expr_base_np: np.ndarray,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    if getattr(model, "cls_head", None) is None or getattr(model, "num_celltypes", 0) <= 0:
        raise RuntimeError("model has no cls_head but celltype prediction was requested.")
    model.eval()
    num_nodes = int(expr_base_np.shape[0])
    num_ct = int(model.num_celltypes)
    probs_all = np.zeros((num_nodes, num_ct), dtype=np.float32)
    expr_base_t = torch.from_numpy(expr_base_np).to(device)

    for batch_imgs in tqdm(img_loader, desc="predict celltype", leave=False):
        if isinstance(batch_imgs, dict):
            images = batch_imgs["image"].to(device)
            idxs = batch_imgs["idx"].to(device).long()
        else:
            images, idxs = batch_imgs
            images = images.to(device)
            idxs = idxs.to(device).long()
        expr = expr_base_t[idxs]
        batch = {"image": images, "expr": expr, "idx": idxs}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp)):
            _, _, _, aux = model(
                batch, spot_feat_in, edge_index, edge_weight, return_attn=False, return_aux=True
            )
        logits = aux.get("cls_logits", None)
        if logits is None:
            raise RuntimeError("cls_logits is None from model; check cls_head configuration.")
        probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
        probs_all[idxs.detach().cpu().numpy()] = probs
    return probs_all


def predict_expression_by_celltype_similarity(
    adata: ad.AnnData,
    train_map_df: pd.DataFrame,
    test_map_df: pd.DataFrame,
    train_celltypes: np.ndarray,
    pred_celltype_probs: np.ndarray,
    celltype_names: list[str],
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
    amp_dtype: torch.dtype | None = None,
    sim_chunk: int = 1024,
    sim_device: str = "cpu",
    meta_cols: list[str] | None = None,
    image_transform: nn.Module | None = None,
    img_encoder: nn.Module | None = None,
    min_conf: float = 0.0,
    fallback: str = "similarity",
) -> dict:
    if test_map_df is None:
        raise ValueError("test_map_df is required for similarity matching.")
    if img_encoder is None:
        raise ValueError("img_encoder is required for celltype-aware similarity.")

    device_t = torch.device(device)
    train_map_df = align_map_ids_to_obs(adata, train_map_df, id_col="spot_id")
    train_map_df, _ = reorder_and_prune_by_obs(adata, train_map_df, id_col="spot_id")
    train_spot_ids = train_map_df["spot_id"].astype(str).tolist()
    if len(train_spot_ids) != len(train_celltypes):
        raise ValueError(
            f"train_celltypes length mismatch: {len(train_celltypes)} vs train_map_df {len(train_spot_ids)}"
        )

    train_loader = DataLoader(
        ImageMapDataset(train_map_df, image_transform=image_transform, id_col="spot_id"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        ImageMapDataset(test_map_df, image_transform=image_transform, id_col="spot_id"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_rep, train_ids = gather_image_representations(
        img_encoder, train_loader, device=device_t, amp_dtype=amp_dtype, normalize=True,
        show_progress=True, desc="embed(train)"
    )
    test_rep, test_ids = gather_image_representations(
        img_encoder, test_loader, device=device_t, amp_dtype=amp_dtype, normalize=True,
        show_progress=True, desc="embed(test)"
    )
    if pred_celltype_probs.shape[0] != test_rep.shape[0]:
        raise ValueError(
            f"pred_celltype_probs rows ({pred_celltype_probs.shape[0]}) "
            f"!= number of test samples ({test_rep.shape[0]})."
        )

    global_idx, global_sim = top1_cosine_match(
        train_rep, test_rep, chunk_size=sim_chunk, device=sim_device,
        show_progress=True, desc="similarity(global)"
    )

    pred_ct_idx = np.argmax(pred_celltype_probs, axis=1).astype(np.int64)
    pred_ct_conf = np.max(pred_celltype_probs, axis=1).astype(np.float32)
    pred_ct_name = np.asarray(
        [celltype_names[i] if i < len(celltype_names) else f"class_{i}" for i in pred_ct_idx],
        dtype=object,
    )

    selected_idx = global_idx.copy()
    selected_sim = global_sim.copy()
    used_ct_constraint = np.zeros(test_rep.shape[0], dtype=bool)
    train_celltypes = np.asarray(train_celltypes, dtype=object)

    class_to_train = {}
    for ci, cname in enumerate(celltype_names):
        inds = np.where(train_celltypes == str(cname))[0]
        class_to_train[ci] = inds.astype(np.int64)

    for ci, cname in enumerate(celltype_names):
        candidate_train = class_to_train.get(ci, np.array([], dtype=np.int64))
        if candidate_train.size == 0:
            continue
        test_sel = np.where((pred_ct_idx == ci) & (pred_ct_conf >= float(min_conf)))[0]
        if test_sel.size == 0:
            continue
        local_idx, local_sim = top1_cosine_match(
            train_rep[candidate_train],
            test_rep[test_sel],
            chunk_size=sim_chunk,
            device=sim_device,
            show_progress=False,
            desc=f"ct:{cname}",
        )
        selected_idx[test_sel] = candidate_train[local_idx]
        selected_sim[test_sel] = local_sim
        used_ct_constraint[test_sel] = True

    expr_train = _expression_matrix_for_spots(adata, train_spot_ids)
    pred_expr = expr_train[selected_idx].astype(np.float32)
    if fallback == "mean":
        fallback_mask = ~used_ct_constraint
        if fallback_mask.any():
            pred_expr[fallback_mask] = _mean_gene_vector(adata)[None, :]

    matched_train_spot_ids = np.asarray(train_spot_ids, dtype=object)[selected_idx]
    matched_train_celltype = train_celltypes[selected_idx]

    matched_meta = None
    if meta_cols:
        cols = [c for c in meta_cols if c in train_map_df.columns]
        if cols:
            matched_meta = train_map_df.iloc[selected_idx][cols].reset_index(drop=True)
        else:
            print(f"[WARN] None of meta_cols found in train_map_df: {meta_cols}")

    return {
        "pred_expression": pred_expr,
        "best_train_index": selected_idx,
        "best_similarity": selected_sim,
        "matched_train_spot_id": matched_train_spot_ids,
        "matched_train_meta": matched_meta,
        "matched_train_celltype": matched_train_celltype,
        "pred_celltype_index": pred_ct_idx,
        "pred_celltype_name": pred_ct_name,
        "pred_celltype_conf": pred_ct_conf,
        "used_celltype_constraint": used_ct_constraint.astype(np.int64),
        "test_ids": test_ids,
        "train_ids": train_ids,
        "train_spot_ids": train_spot_ids,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="추론 대상 h5ad")
    ap.add_argument("--csv", required=True, help="추론 대상 이미지-스팟 매핑 CSV")
    ap.add_argument("--root", default=None, help="eval CSV patch_path 상대경로 루트")
    ap.add_argument("--id_col", default="cell_id", help="eval CSV ID 컬럼")
    ap.add_argument("--path_col", default="patch_path", help="eval CSV 이미지 경로 컬럼")

    ap.add_argument("--train_h5ad", required=True, help="train h5ad (mean/similarity 표현용)")
    ap.add_argument("--train_csv", default=None, help="similarity 모드용 train CSV")
    ap.add_argument("--train_root", default=None, help="train CSV patch_path 상대경로 루트")
    ap.add_argument("--train_id_col", default=None, help="train CSV ID 컬럼 (기본: --id_col)")
    ap.add_argument("--train_path_col", default=None, help="train CSV 경로 컬럼 (기본: --path_col)")

    ap.add_argument("--enc_name", default="uni_v1",
                    choices=["uni_v1", "virchow", "virchow2", "gigapath", "hoptimus0", "plip", "phikon", "conch_v1"])
    ap.add_argument("--enc_weights_path", default=None)

    ap.add_argument("--proj_dim_gene", type=int, default=256)
    ap.add_argument("--proj_dim_spot", type=int, default=256)
    ap.add_argument("--fuse_dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--img_tokens", type=int, default=1,
                    help="number of image query tokens in fusion cross-attention")
    ap.add_argument("--spot_tokens", type=int, default=1,
                    help="number of spatial tokens in fusion cross-attention")
    ap.add_argument("--gene_tokens", type=int, default=1,
                    help="number of gene tokens in fusion cross-attention")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--merge", default="gated-sum", choices=["gated-sum", "concat-proj"])

    ap.add_argument("--graph_type", default="knn", choices=["knn", "delaunay"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--ft_model_ckpt", default=None, help="finetune된 전체 모델 ckpt (model.pt)")
    ap.add_argument("--pt_img_backbone", default=None, help="pretrained image backbone ckpt")
    ap.add_argument("--pt_ig", default=None, help="pretrained img↔gene ckpt")
    ap.add_argument("--pt_is", default=None, help="pretrained img↔spatial ckpt")
    ap.add_argument("--pt_xattn", default=None, help="pretrained xattn ckpt (XATTN_pretrained.pt)")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument(
        "--expr_source",
        default="mean",
        choices=["mean", "train_cluster_centroid", "similarity", "celltype_similarity"],
    )
    ap.add_argument("--cluster_col", default="leiden",
                    help="train_h5ad cluster column for --expr_source train_cluster_centroid")
    ap.add_argument("--cluster_centroid_agg", default="mean", choices=["mean", "size_weighted"],
                    help="aggregation over train cluster centroids (mean=uniform over clusters)")
    ap.add_argument("--sim_chunk", type=int, default=1024)
    ap.add_argument("--sim_device", default="cpu")
    ap.add_argument("--sim_batch_size", type=int, default=None)
    ap.add_argument("--sim_meta_cols", default=None, help="comma-separated train CSV columns to attach")
    ap.add_argument("--sim_meta_prefix", default="nn_", help="prefix for attached similarity metadata")
    ap.add_argument("--celltype_meta_pt", default=None,
                    help="path to celltype_meta.pt (default: dirname(--ft_model_ckpt)/celltype_meta.pt)")
    ap.add_argument("--celltype_col", default=None,
                    help="reference(train_h5ad) celltype column override")
    ap.add_argument("--celltype_min_conf", type=float, default=0.0,
                    help="minimum predicted celltype confidence to apply ct-constrained retrieval")
    ap.add_argument("--celltype_fallback", default="similarity", choices=["similarity", "mean"],
                    help="fallback expr source when ct constraint is not applied")
    ap.add_argument(
        "--celltype_pred_expr_source",
        default="mean",
        choices=["mean", "eval"],
        help="expr source to predict celltype in celltype_similarity "
             "(mean=train mean, eval=eval h5ad expression)",
    )

    ap.add_argument("--out_h5ad", required=True)
    ap.add_argument("--save_attn", action="store_true",
                    help="save attention weights during inference")
    ap.add_argument("--attn_out_dir", default=None,
                    help="directory to save attention npz files (default: <out_h5ad_dir>/attn)")
    ap.add_argument("--attn_max_batches", type=int, default=20,
                    help="max inference batches to dump attention")
    ap.add_argument("--attn_reduce_heads", action="store_true",
                    help="store mean over heads instead of per-head attention")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)

    device = select_device(args.device)
    amp_dtype = pick_amp_dtype(args.amp)
    print(f"[Device] {device}, AMP={args.amp}, dtype={amp_dtype}")
    print(f"[FusionTokens] img={args.img_tokens}, spot={args.spot_tokens}, gene={args.gene_tokens}")
    attn_out_dir = None
    if args.save_attn:
        base_dir = os.path.dirname(args.out_h5ad) or "."
        attn_out_dir = args.attn_out_dir or os.path.join(base_dir, "attn")
        os.makedirs(attn_out_dir, exist_ok=True)
        print(f"[AttnSave] dir={attn_out_dir} max_batches={args.attn_max_batches}")

    eval_adata = load_h5ad_with_preproc(
        args.h5ad, use_hvg=False, n_top_genes=None, x_key="x_centroid", y_key="y_centroid"
    )
    train_adata = load_h5ad_with_preproc(
        args.train_h5ad, use_hvg=False, n_top_genes=None, x_key="x_centroid", y_key="y_centroid"
    )
    train_adata, eval_adata = align_genes_for_inference(train_adata, eval_adata)

    map_df = load_patch_mapping_csv_xenium(
        args.csv,
        root_dir=args.root,
        id_col=args.id_col,
        path_col=args.path_col,
        drop_missing=True,
    )
    map_df = align_map_ids_to_obs(eval_adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(eval_adata, map_df, id_col="spot_id", coord_key="spatial")
    if map_df_aligned.shape[0] != eval_adata.n_obs:
        eval_obs_ids = map_df_aligned["spot_id"].astype(str).tolist()
        eval_adata = eval_adata[eval_obs_ids].copy()

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
    print(f"[Graph] type={args.graph_type}, k={args.k}, E={edge_index.size(1)}")
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    img_dim = foundation.get_output_dim()
    gene_dim = eval_adata.n_vars
    spot_feat_in = torch.from_numpy(coords_np).float().to(device)
    gene_encoder = GeneEncoder(input_dim=gene_dim, model_size_omic="small",
                               proj_dim=args.proj_dim_gene, normalize=False)
    gcn = SpatialGCN(in_dim=spot_feat_in.size(1), hidden_dim=256,
                     out_dim=args.proj_dim_spot, dropout=0.2)

    load_pretrained_blocks(
        foundation, gene_encoder, gcn,
        pt_img_backbone=args.pt_img_backbone,
        pt_ig=args.pt_ig,
        pt_is=args.pt_is,
    )

    ft_sd = _load_sd(args.ft_model_ckpt) if args.ft_model_ckpt else None
    ckpt_num_ct, ckpt_ct_hidden = _infer_cls_head_config(ft_sd)

    celltype_meta_path = args.celltype_meta_pt
    if celltype_meta_path is None and args.ft_model_ckpt:
        cand = os.path.join(os.path.dirname(args.ft_model_ckpt), "celltype_meta.pt")
        if os.path.isfile(cand):
            celltype_meta_path = cand
    ct_meta = _load_celltype_meta(celltype_meta_path)
    meta_num_ct = int(ct_meta.get("num_celltypes", 0)) if isinstance(ct_meta, dict) else 0
    num_celltypes = max(meta_num_ct, ckpt_num_ct)
    cls_head_hidden = ckpt_ct_hidden
    if num_celltypes > 0:
        print(
            f"[CT] enable cls_head num_celltypes={num_celltypes}, "
            f"hidden={cls_head_hidden}, meta={celltype_meta_path}"
        )

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
        cls_head_hidden=cls_head_hidden,
        cls_dropout=0.0,
    ).to(device)

    if args.pt_xattn:
        load_xattn_into_fusion(model.fusion, args.pt_xattn)

    if args.ft_model_ckpt:
        sd = ft_sd if ft_sd is not None else _load_sd(args.ft_model_ckpt)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"[FT] loaded full model state from {args.ft_model_ckpt} (strict=False)")
        except Exception as e:
            print(f"[FT][warn] failed to load full model from {args.ft_model_ckpt}: {e}")

    model.eval()

    sim_out = None
    cluster_out = None
    gene_mean = _mean_gene_vector(train_adata)
    if args.expr_source == "mean":
        expr_matrix = np.repeat(gene_mean[None, :], map_df_aligned.shape[0], axis=0)
    elif args.expr_source == "train_cluster_centroid":
        cluster_col = _resolve_cluster_col(train_adata, args.cluster_col)
        centroid_vec, cluster_names, cluster_sizes, cluster_weights = _build_cluster_centroid_reference_vector(
            train_adata,
            cluster_col=cluster_col,
            agg=args.cluster_centroid_agg,
        )
        expr_matrix = np.repeat(centroid_vec[None, :], map_df_aligned.shape[0], axis=0).astype(np.float32)
        cluster_out = {
            "cluster_col": cluster_col,
            "agg": args.cluster_centroid_agg,
            "cluster_names": cluster_names,
            "cluster_sizes": [int(x) for x in cluster_sizes],
            "cluster_weights": cluster_weights.astype(np.float32).tolist(),
        }
        print(
            f"[ClusterCentroid] col={cluster_col}, n_clusters={len(cluster_names)}, "
            f"agg={args.cluster_centroid_agg}"
        )
    else:
        if not args.train_csv:
            raise ValueError("--train_csv is required when --expr_source uses similarity")

        train_id_col = args.train_id_col or args.id_col
        train_path_col = args.train_path_col or args.path_col
        train_map_df = load_patch_mapping_csv_xenium(
            args.train_csv,
            root_dir=args.train_root or args.root,
            id_col=train_id_col,
            path_col=train_path_col,
            drop_missing=True,
        )
        sim_bs = args.sim_batch_size or args.batch_size
        meta_cols = None
        if args.sim_meta_cols:
            meta_cols = [c.strip() for c in args.sim_meta_cols.split(",") if c.strip()]

        if args.expr_source == "similarity":
            sim_out = predict_expression_by_image_similarity(
                adata=train_adata,
                train_map_df=train_map_df,
                test_map_df=map_df_aligned,
                enc_name=args.enc_name,
                weights_path=args.enc_weights_path,
                batch_size=sim_bs,
                num_workers=args.num_workers,
                device=str(device),
                amp_dtype=amp_dtype if args.amp else torch.float32,
                sim_chunk=args.sim_chunk,
                sim_device=args.sim_device,
                meta_cols=meta_cols,
                img_encoder=model.img_encoder,
                image_transform=image_tf,
            )
            expr_matrix = sim_out["pred_expression"]
        else:
            if num_celltypes <= 0 or model.cls_head is None:
                raise ValueError(
                    "--expr_source=celltype_similarity requires cls_head in checkpoint "
                    "(train with run_finetuning_xattn_withcelltype.py)."
                )
            if ct_meta is None or "celltype_names" not in ct_meta:
                raise ValueError(
                    "celltype_meta.pt with 'celltype_names' is required for celltype_similarity."
                )
            celltype_names = [str(x) for x in ct_meta.get("celltype_names", [])]
            if len(celltype_names) != num_celltypes:
                raise ValueError(
                    f"celltype_names length ({len(celltype_names)}) != num_celltypes ({num_celltypes})."
                )
            ct_col = _resolve_obs_col(train_adata, args.celltype_col)
            if ct_col is None:
                ct_col = _resolve_obs_col(train_adata, ct_meta.get("celltype_col", None))
            if ct_col is None:
                raise ValueError(
                    "celltype column not found in train_h5ad. "
                    "Set --celltype_col or check celltype_meta.pt."
                )
            train_map_df_aligned = align_map_ids_to_obs(train_adata, train_map_df, id_col="spot_id")
            train_map_df_aligned, _ = reorder_and_prune_by_obs(
                train_adata, train_map_df_aligned, id_col="spot_id", coord_key="spatial"
            )
            train_spot_ids = train_map_df_aligned["spot_id"].astype(str).tolist()
            train_ct = (
                train_adata.obs.loc[train_spot_ids, ct_col]
                .astype(str)
                .to_numpy()
            )
            print(f"[CT] reference celltype_col={ct_col}, n_classes(meta)={len(celltype_names)}")

            if args.celltype_pred_expr_source == "eval":
                eval_spot_ids = map_df_aligned["spot_id"].astype(str).tolist()
                expr_base_for_ct = _expression_matrix_for_spots(eval_adata, eval_spot_ids).astype(np.float32)
                print("[CT] pred expr base=eval expression")
            else:
                expr_base_for_ct = np.repeat(gene_mean[None, :], map_df_aligned.shape[0], axis=0).astype(np.float32)
                print("[CT] pred expr base=train mean expression")
            pred_ct_probs = _predict_celltype_probs_from_base_expr(
                model=model,
                img_loader=img_loader,
                spot_feat_in=spot_feat_in,
                edge_index=edge_index,
                edge_weight=edge_weight,
                expr_base_np=expr_base_for_ct,
                device=device,
                amp=args.amp,
                amp_dtype=amp_dtype,
            )
            pred_idx_dbg = np.argmax(pred_ct_probs, axis=1).astype(np.int64)
            u_dbg, c_dbg = np.unique(pred_idx_dbg, return_counts=True)
            order_dbg = np.argsort(-c_dbg)
            topk = min(5, order_dbg.shape[0])
            parts = []
            for oi in order_dbg[:topk]:
                ci = int(u_dbg[oi])
                cname = celltype_names[ci] if ci < len(celltype_names) else f"class_{ci}"
                parts.append(f"{cname}:{int(c_dbg[oi])}")
            print(f"[CT] predicted class distribution(top{topk}): " + ", ".join(parts))

            sim_out = predict_expression_by_celltype_similarity(
                adata=train_adata,
                train_map_df=train_map_df,
                test_map_df=map_df_aligned,
                train_celltypes=train_ct,
                pred_celltype_probs=pred_ct_probs,
                celltype_names=celltype_names,
                batch_size=sim_bs,
                num_workers=args.num_workers,
                device=str(device),
                amp_dtype=amp_dtype if args.amp else torch.float32,
                sim_chunk=args.sim_chunk,
                sim_device=args.sim_device,
                meta_cols=meta_cols,
                image_transform=image_tf,
                img_encoder=model.img_encoder,
                min_conf=args.celltype_min_conf,
                fallback=args.celltype_fallback,
            )
            expr_matrix = sim_out["pred_expression"]

    preds = []
    expr_matrix_t = torch.from_numpy(expr_matrix).to(device)
    with torch.no_grad():
        pbar = tqdm(img_loader, desc="inference", leave=False)
        for step, batch_imgs in enumerate(pbar, start=1):
            if isinstance(batch_imgs, dict):
                images = batch_imgs["image"].to(device)
                idxs = batch_imgs["idx"].to(device).long()
            else:
                images, idxs = batch_imgs
                images = images.to(device)
                idxs = idxs.to(device).long()

            expr = expr_matrix_t[idxs]
            batch = {"image": images, "expr": expr, "idx": idxs}

            t0 = time.time()
            dump_attn_this_batch = bool(args.save_attn and attn_out_dir and args.attn_max_batches > 0 and step <= args.attn_max_batches)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and args.amp)
            ):
                if dump_attn_this_batch:
                    mu, theta, pi, attn_map = model(
                        batch, spot_feat_in, edge_index, edge_weight, return_attn=True
                    )
                else:
                    mu, theta, pi = model(batch, spot_feat_in, edge_index, edge_weight, return_attn=False)
                    attn_map = None
            preds.append(mu.detach().cpu())
            if dump_attn_this_batch and attn_map is not None:
                out_npz = os.path.join(attn_out_dir, f"infer_step{step:04d}.npz")
                _save_attn_npz(
                    out_npz,
                    attn_map,
                    reduce_heads=args.attn_reduce_heads,
                    node_idx=idxs.detach().cpu().numpy(),
                )
            pbar.set_postfix(batch_s=f"{time.time()-t0:.3f}")

    P = torch.cat(preds, dim=0).numpy()

    rec = ad.AnnData(
        X=P,
        obs=eval_adata.obs.copy(),
        var=eval_adata.var.copy(),
        uns=eval_adata.uns.copy(),
    )
    if "x_centroid" in rec.obs.columns and "y_centroid" in rec.obs.columns:
        rec.obsm["spatial"] = rec.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)

    if sim_out is not None:
        pref = args.sim_meta_prefix
        rec.obs[pref + "train_index"] = sim_out["best_train_index"]
        rec.obs[pref + "similarity"] = sim_out["best_similarity"]
        rec.obs[pref + "train_spot_id"] = sim_out["matched_train_spot_id"]
        if "matched_train_celltype" in sim_out:
            rec.obs[pref + "train_celltype"] = sim_out["matched_train_celltype"]
        if "pred_celltype_index" in sim_out:
            rec.obs[pref + "pred_celltype_index"] = sim_out["pred_celltype_index"]
        if "pred_celltype_name" in sim_out:
            rec.obs[pref + "pred_celltype_name"] = sim_out["pred_celltype_name"]
        if "pred_celltype_conf" in sim_out:
            rec.obs[pref + "pred_celltype_conf"] = sim_out["pred_celltype_conf"]
        if "used_celltype_constraint" in sim_out:
            rec.obs[pref + "used_celltype_constraint"] = sim_out["used_celltype_constraint"]
        matched_meta = sim_out.get("matched_train_meta", None)
        if matched_meta is not None:
            for col in matched_meta.columns:
                rec.obs[pref + col] = matched_meta[col].values
    if cluster_out is not None:
        rec.uns["train_cluster_centroid"] = {
            "cluster_col": cluster_out["cluster_col"],
            "agg": cluster_out["agg"],
            "n_clusters": int(len(cluster_out["cluster_names"])),
            "cluster_names": cluster_out["cluster_names"],
            "cluster_sizes": cluster_out["cluster_sizes"],
            "cluster_weights": cluster_out["cluster_weights"],
        }

    rec.write(args.out_h5ad, compression="gzip")
    print(f"[DONE] saved: {args.out_h5ad}  (shape={P.shape})")


if __name__ == "__main__":
    main()
