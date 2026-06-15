#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_inference_xattn.py
- xattn 기반으로 파인튜닝된 Cross-Attn + ZINB 모델 추론 스크립트
- 블록별 ckpt(Image/SNN/Coord/XATTN) 또는 전체 ft model ckpt 로드 지원
- gene 입력 소스:
  - mean: train 데이터의 gene-wise mean
  - similarity: 이미지 임베딩 유사도 기반 매칭

예시:
CUDA_VISIBLE_DEVICES=0 \
python Code_final/code/utils/run_inference_xattn.py \
  --h5ad /path/eval.h5ad \
  --csv /path/eval_map.csv \
  --root /path/root \
  --train_h5ad /path/train.h5ad \
  --train_csv /path/train_map.csv \
  --enc_name uni_v1 \
  --id_col cell_id --path_col patch_path \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \
  --graph_type knn --k 12 \
  --pt_img_backbone /path/Image_pretrained.pt \
  --pt_ig /path/SNN_pretrained.pt \
  --pt_is /path/Coord_pretrained.pt \
  --pt_xattn /path/XATTN_pretrained.pt \
  --ft_model_ckpt /path/best/model.pt \
  --expr_source similarity \
  --out_h5ad /path/recovered_eval.h5ad

CUDA_VISIBLE_DEVICES=1 \
python Code_final/code/utils/run_inference_xattn.py \
  --h5ad   /data/project/banana9903_xenium/h5ads/human_lung_addon.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/human_lung_addon_patches_256/spot_patch_map_256.csv \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --train_h5ad /data/project/banana9903_xenium/h5ads/human_lung_addon.h5ad\
  --train_csv /data/project/banana9903_xenium/contrast_test/expression/human_lung_addon_patches_256/spot_patch_map_256.csv \
  --enc_name uni_v1 \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \
  --graph_type delaunay \
  --ft_model_ckpt /data/project/banana9903_xenium/ckpts_cross_human_lung_addon_xattn_delaunay_10tokens/best/model.pt \
  --expr_source similarity \
  --sim_meta_cols cell_id,patch_path,x_he_px,y_he_px,inside_he \
  --batch_size 1024 --num_workers 120 \
  --out_h5ad /data/project/banana9903_xenium/inference/recovered_human_human_lung_addon_eval_0212.h5ad \
  --img_tokens 10 \
  --gene_tokens 10 \
  --spot_tokens 10


CUDA_VISIBLE_DEVICES=1 python Code_final/code/utils/run_inference_xattn.py \
  --h5ad /data/project/banana9903_xenium/h5ads/human_lung_addon.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/human_lung_addon_patches_256/spot_patch_map_256.csv  \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --id_col cell_id --path_col patch_path \
  --train_h5ad /data/project/banana9903_xenium/h5ads/human_lung_addon.h5ad \
  --train_csv /data/project/banana9903_xenium/contrast_test/expression/human_lung_addon_patches_256/spot_patch_map_256.csv \
  --enc_name uni_v1 \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \
  --graph_type knn --k 12 \
  --pt_img_backbone /data/project/banana9903_xenium/ckpts_human_breast_cancer_xattn/Image_pretrained.pt \
  --pt_ig /data/project/banana9903_xenium/ckpts_human_breast_cancer_xattn/SNN_pretrained.pt \
  --pt_is /data/project/banana9903_xenium/ckpts_human_breast_cancer_xattn/Coord_pretrained.pt \
  --pt_xattn /data/project/banana9903_xenium/ckpts_human_breast_cancer_xattn/XATTN_pretrained.pt \
  --ft_model_ckpt /data/project/banana9903_xenium/ckpts_cross_human_breast_cancer_xattn/best/model.pt \
  --expr_source similarity \
  --batch_size 256 --num_workers 4 \
  --out_h5ad /data/project/banana9903_xenium/inference/recovered_breast_xattn.h5ad \
  --img_tokens 10 \
  --gene_tokens 10 \
  --spot_tokens 10

token 있는거 사용!!
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import anndata as ad
from tqdm import tqdm

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
from code.utils.image_representation_gathering import predict_expression_by_image_similarity


class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn, dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1, merge="gated-sum",
                 img_tokens=1, spot_tokens=1, gene_tokens=1,
                 decoder_hidden=1024, gene_dim=1000):
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

    def forward(self, batch, spot_feat_in, edge_index, edge_weight=None, return_attn: bool = False):
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

    ap.add_argument("--expr_source", default="mean", choices=["mean", "similarity"])
    ap.add_argument("--sim_chunk", type=int, default=1024)
    ap.add_argument("--sim_device", default="cpu")
    ap.add_argument("--sim_batch_size", type=int, default=None)
    ap.add_argument("--sim_meta_cols", default=None, help="comma-separated train CSV columns to attach")
    ap.add_argument("--sim_meta_prefix", default="nn_", help="prefix for attached similarity metadata")

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
    ).to(device)

    if args.pt_xattn:
        load_xattn_into_fusion(model.fusion, args.pt_xattn)

    if args.ft_model_ckpt:
        sd = _load_sd(args.ft_model_ckpt)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"[FT] loaded full model state from {args.ft_model_ckpt} (strict=False)")
        except Exception as e:
            print(f"[FT][warn] failed to load full model from {args.ft_model_ckpt}: {e}")

    model.eval()

    sim_out = None
    if args.expr_source == "mean":
        gene_mean = np.asarray(train_adata.X).mean(axis=0).astype(np.float32)
        expr_matrix = np.repeat(gene_mean[None, :], map_df_aligned.shape[0], axis=0)
    else:
        if not args.train_csv:
            raise ValueError("--train_csv is required when --expr_source=similarity")

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
        matched_meta = sim_out.get("matched_train_meta", None)
        if matched_meta is not None:
            for col in matched_meta.columns:
                rec.obs[pref + col] = matched_meta[col].values

    rec.write(args.out_h5ad, compression="gzip")
    print(f"[DONE] saved: {args.out_h5ad}  (shape={P.shape})")


if __name__ == "__main__":
    main()
