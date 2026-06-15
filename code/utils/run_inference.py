#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_inference.py — 파인튜닝된 Cross-Attn + ZINB 모델로 추론 실행
- 3개 모달리티 인코더(이미지 백본, gene encoder, GCN) 개별 ckpt 로드 지원
- 추론 시 gene expression 입력은
  - mean: train 데이터의 gene-wise mean 벡터
  - similarity: 이미지 임베딩 유사도 기반으로 train cell expression을 매칭
- 최종 결과를 예측 expression 행렬(X=mu)로 갖는 h5ad로 저장

사용 예)
CUDA_VISIBLE_DEVICES=1 \
python Code_final/code/utils/run_inference.py \
  --h5ad  /data/project/banana9903_xenium/h5ads/human_reactive_follicular.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/human_reactive_follicular_patches_256/spot_patch_map_256.csv \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --train_h5ad /data/project/banana9903_xenium/h5ads/human_follicular_lymphoid.h5ad \
  --train_csv /data/project/banana9903_xenium/contrast_test/expression/human_follicular_lymphoid_patches_256/spot_patch_map_256.csv \
  --enc_name uni_v1 \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 512 --heads 8 \
  --ft_model_ckpt /data/project/banana9903_xenium/ckpts_cross_human_follycular_lymphoid/best/model.pt \
  --expr_source similarity \
  --sim_meta_cols cell_id,patch_path,x_he_px,y_he_px,inside_he \
  --batch_size 1024 --num_workers 120 \
  --out_h5ad /data/project/banana9903_xenium/inference/recovered_human_human_reactive_follicular_eval_0124.h5ad

"""

import os, sys, time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from code.utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_only_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)
from code.utils.image_representation_gathering import predict_expression_by_image_similarity

# ----------------------------
# 모듈 정의 (train과 동일 구조)
# ----------------------------
class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn, dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1, merge="gated-sum",
                 decoder_hidden=1024, gene_dim=1000):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.gcn = gcn

        self.proj_img  = nn.Linear(dim_img, dim_fuse)
        self.proj_gene = nn.Linear(dim_gene, dim_fuse)
        self.proj_spot = nn.Linear(dim_spot, dim_fuse)

        self.fusion = CrossModalFusion(dim_img=dim_fuse, dim_spot=dim_fuse, dim_gene=dim_fuse,
                                       dim_out=dim_fuse, num_heads=num_heads, dropout=dropout,
                                       merge=merge)
        self.decoder = ZINBDecoder(in_dim=dim_fuse, gene_dim=gene_dim, hidden=decoder_hidden, dropout=dropout)

    def forward(self, batch, spot_feat_in, edge_index, edge_weight=None):
        x_img = batch["image"]            # (B, C, H, W)
        x_gene = batch["expr"].float()    # (B, G) — 여기선 mean 벡터 반복
        node_idx = batch["idx"].long()    # (B,)

        zi = self.img_encoder(x_img)
        zg = self.gene_encoder(x_gene)
        zs_all = self.gcn(spot_feat_in, edge_index, edge_weight)  # (N, Ds)
        zs_all = zs_all[node_idx]  # (B, Ds)

        zi = self.proj_img(zi)
        zg = self.proj_gene(zg)
        zs = self.proj_spot(zs_all)

        z_fused, _, _ = self.fusion(zi, zs, zg)
        mu, theta, pi = self.decoder(z_fused)
        return mu, theta, pi


# ----------------------------
# 로딩 유틸
# ----------------------------
def _load_sd(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and any(k in sd for k in ["state_dict","model","module"]):
        for k in ["state_dict","model","module"]:
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd if isinstance(sd, dict) else sd.state_dict()


def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_pretrained_blocks(foundation, gene_encoder, gcn,
                           pt_img_backbone=None, pt_ig=None, pt_is=None):
    if pt_img_backbone:
        sd = _load_sd(pt_img_backbone)
        for cand in [
            _strip_prefix(sd, "img_encoder.foundation."),
            _strip_prefix(sd, "foundation."),
            sd,
        ]:
            try:
                foundation.load_state_dict(cand, strict=False)
                print(f"[PT] foundation loaded from {pt_img_backbone} (strict=False)")
                break
            except Exception:
                pass

    if pt_ig:
        sd = _load_sd(pt_ig)
        for cand in [
            _strip_prefix(sd, "gene_encoder."),
            sd,
        ]:
            try:
                gene_encoder.load_state_dict(cand, strict=False)
                print(f"[PT] gene_encoder loaded from {pt_ig} (strict=False)")
                break
            except Exception:
                pass

    if pt_is:
        sd = _load_sd(pt_is)
        for cand in [
            _strip_prefix(sd, "gcn."),
            sd,
        ]:
            try:
                gcn.load_state_dict(cand, strict=False)
                print(f"[PT] gcn loaded from {pt_is} (strict=False)")
                break
            except Exception:
                pass


# ----------------------------
# 기타 유틸
# ----------------------------
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
    """train과 eval의 교집합 유전자 순서를 eval 기준으로 맞춰 반환.
    모델은 eval_adata의 var_names 순서대로 예측을 출력하게 됨.
    """
    common = eval_adata.var_names.intersection(train_adata.var_names)
    if len(common) == 0:
        raise ValueError("No overlapping genes between train and eval datasets.")
    # eval 순서 유지
    eval_genes = [g for g in eval_adata.var_names if g in common]
    train_genes = eval_genes  # 같은 순서로
    train_sub = train_adata[:, train_genes].copy()
    eval_sub = eval_adata[:, eval_genes].copy()
    return train_sub, eval_sub


# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="추론 대상 h5ad")
    ap.add_argument("--csv", required=True, help="이미지-스팟 매핑 CSV")
    ap.add_argument("--root", default=None, help="CSV patch_path 상대경로 루트")
    ap.add_argument("--train_h5ad", required=True, help="train h5ad (mean/similarity 표현용)")
    ap.add_argument("--train_csv", default=None, help="similarity 모드에서 사용할 train patch map CSV")
    ap.add_argument("--train_root", default=None, help="train CSV patch_path 상대경로 루트")

    ap.add_argument("--enc_name", default="uni_v1",
                    choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    ap.add_argument("--enc_weights_path", default=None,
                    help="encoder weights dir (e.g., local UNI folder with pytorch_model.bin)")

    ap.add_argument("--proj_dim_gene", type=int, default=256)
    ap.add_argument("--proj_dim_spot", type=int, default=256)
    ap.add_argument("--fuse_dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--merge", default="gated-sum", choices=["gated-sum","concat-proj"])

    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--ft_model_ckpt", default=None, help="finetune된 전체 모델 state_dict (model.pt)")
    ap.add_argument("--pt_img_backbone", default=None, help="pretrained image backbone ckpt")
    ap.add_argument("--pt_ig", default=None, help="pretrained img↔gene ckpt")
    ap.add_argument("--pt_is", default=None, help="pretrained img↔spatial ckpt")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--expr_source", default="mean", choices=["mean", "similarity"],
                    help="gene 입력 소스: mean(train 평균) 또는 similarity(이미지 유사도 기반)")
    ap.add_argument("--sim_chunk", type=int, default=1024, help="similarity 매칭 chunk size")
    ap.add_argument("--sim_device", default="cpu", help="similarity 계산 디바이스 (cpu/cuda)")
    ap.add_argument("--sim_batch_size", type=int, default=None, help="similarity 임베딩 배치 (기본: --batch_size)")
    ap.add_argument("--sim_meta_cols", default=None,
                    help="comma-separated train CSV columns to attach to output obs (similarity mode)")
    ap.add_argument("--sim_meta_prefix", default="nn_", help="prefix for attached similarity metadata columns")

    ap.add_argument("--out_h5ad", required=True)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)

    device = select_device(args.device)
    amp_dtype = pick_amp_dtype(args.amp)
    print(f"[Device] {device}, AMP={args.amp}, dtype={amp_dtype}")

    # 1) 데이터 로딩: eval + train(평균 벡터용)
    eval_adata = load_h5ad_with_preproc(args.h5ad, use_hvg=False, n_top_genes=None,
                                        x_key="x_centroid", y_key="y_centroid")
    train_adata = load_h5ad_with_preproc(args.train_h5ad, use_hvg=False, n_top_genes=None,
                                         x_key="x_centroid", y_key="y_centroid")
    # 유전자 정렬(교집합, eval 순서)
    train_adata, eval_adata = align_genes_for_inference(train_adata, eval_adata)

    # 매핑 로딩 + 정렬
    map_df = load_patch_mapping_csv_xenium(args.csv, root_dir=args.root, drop_missing=True)
    map_df = align_map_ids_to_obs(eval_adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(eval_adata, map_df, id_col="spot_id", coord_key="spatial")
    if map_df_aligned.shape[0] != eval_adata.n_obs:
        eval_obs_ids = map_df_aligned["spot_id"].astype(str).tolist()
        eval_adata = eval_adata[eval_obs_ids].copy()

    # 2) 로더 (이미지 전용)
    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation = FoundationCls(weights_path=args.enc_weights_path)
    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)

    img_loader = make_img_only_dataloader(
        map_df=map_df_aligned,
        batch_size=max(1, args.batch_size),
        num_workers=max(1, args.num_workers),
        image_transform=image_tf,
        shuffle=False
    )

    # 그래프
    edge_index, edge_weight = build_knn_graph(coords_np, k=args.k, use_weight=True)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    # 3) 모듈 구성
    # img encoder output dim 질의
    img_dim = foundation.get_output_dim()
    gene_dim = eval_adata.n_vars

    gene_encoder = GeneEncoder(input_dim=gene_dim, model_size_omic="small",
                               proj_dim=args.proj_dim_gene, normalize=False)
    spot_feat_in = torch.from_numpy(coords_np).float().to(device)
    gcn = SpatialGCN(in_dim=spot_feat_in.size(1), hidden_dim=256,
                     out_dim=args.proj_dim_spot, dropout=0.2)

    # (선택) 프리트레인 블록 로드
    load_pretrained_blocks(foundation, gene_encoder, gcn,
                           pt_img_backbone=args.pt_img_backbone,
                           pt_ig=args.pt_ig,
                           pt_is=args.pt_is)

    # 파인튜닝 전체 모델 래퍼
    model = CrossAttnZINBModel(
        img_encoder=foundation,
        gene_encoder=gene_encoder,
        gcn=gcn,
        dim_img=img_dim, dim_gene=args.proj_dim_gene, dim_spot=args.proj_dim_spot,
        dim_fuse=args.fuse_dim, num_heads=args.heads, dropout=args.dropout,
        merge=args.merge, decoder_hidden=1024, gene_dim=gene_dim
    ).to(device)

    # (선택) 파인튜닝 전체 ckpt 로드
    if args.ft_model_ckpt:
        sd = _load_sd(args.ft_model_ckpt)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"[FT] loaded full model state from {args.ft_model_ckpt} (strict=False)")
        except Exception as e:
            print(f"[FT][warn] failed to load full model from {args.ft_model_ckpt}: {e}")

    model.eval()

    sim_out = None
    # gene 입력 준비 (mean or similarity)
    if args.expr_source == "mean":
        gene_mean = np.asarray(train_adata.X).mean(axis=0).astype(np.float32)
        expr_matrix = np.repeat(gene_mean[None, :], map_df_aligned.shape[0], axis=0)
    else:
        if not args.train_csv:
            raise ValueError("--train_csv is required when --expr_source=similarity")
        train_map_df = load_patch_mapping_csv_xenium(
            args.train_csv,
            root_dir=args.train_root or args.root,
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

    # 4) 추론 루프
    preds = []
    idx_cursor = 0
    expr_matrix_t = torch.from_numpy(expr_matrix).to(device)

    with torch.no_grad():
        pbar = tqdm(img_loader, desc="inference", leave=False)
        for batch_imgs in pbar:
            # img_loader가 (images, indices) 형태라고 가정 → 실제 구현에 맞춰 조정 필요
            # Code_final의 make_img_only_dataloader는 통상 {'image':..., 'idx':...} 반환하도록 설계됨.
            if isinstance(batch_imgs, dict):
                images = batch_imgs["image"].to(device)
                idxs   = batch_imgs["idx"].to(device).long()
            else:
                # (images, idx)
                images, idxs = batch_imgs
                images = images.to(device)
                idxs   = idxs.to(device).long()

            B = images.size(0)
            expr = expr_matrix_t[idxs]  # (B, G) mean or similarity 기반 입력

            batch = {"image": images, "expr": expr, "idx": idxs}
            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type=="cuda" and args.amp)):
                mu, theta, pi = model(batch, spot_feat_in, edge_index, edge_weight)
            preds.append(mu.detach().cpu())
            idx_cursor += B
            pbar.set_postfix(batch_s=f"{time.time()-t0:.3f}")

    P = torch.cat(preds, dim=0).numpy()

    # 5) 저장 (eval_adata 순서와 동일)
    rec = ad.AnnData(X=P, obs=eval_adata.obs.copy(), var=eval_adata.var.copy(), uns=eval_adata.uns.copy())
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
