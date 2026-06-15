# run_contrastive.py
# 실행 예:
# Output Files

'''
Code_final.code.utils.run_contrastive의 Docstring

CUDA_VISIBLE_DEVICES=1 \
/data/project/banana9903_xenium/.conda-envs/SPAM/bin/python Code_final/code/utils/run_contrastive.py \
  --h5ad /data/project/banana9903_xenium/xenium_rep1_io/data/merged_with_celltype_raw.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --enc_name uni_v1 \
  --mode joint \
  --epochs 20 --batch_size 16 --img_batch_size 256 \
  --save_dir /data/project/banana9903_xenium/ckpts_human_lung _cancer_delaunay2 \
  --device cuda:0 \
  --gcn_device cuda:0 \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --num_workers 4
  --graph_type delaunay 

'''


import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # /Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 표준모듈 'code'와 충돌 방지: 이미 로드돼 있으면 제거
sys.modules.pop("code", None)

import argparse
import torch

# --- contrastive 학습 루프 & 래퍼 ---
from code.utils.contrastive import (
    ImageFoundationAdapter,
    img2gene_ContrastiveModel,
    img2spot_ContrastiveModel,
    build_spatial_graph,
    train_contrastive,
    train_img_spatial_contrastive,
    train_joint_img_gene_spatial,
)

# --- 프로젝트 모듈 ---
from code.models.Foundations import inf_encoder_factory
from code.models.GCN_update import SpatialGCN
from code.models.gene_encoder import GeneEncoder
from code.utils.lora_utils import attach_lora_to_foundation

from code.utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_gene_dataloader,
    make_img_only_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)


# ----------------------------
# Device & AMP helpers
# ----------------------------
def setup_cuda_flags():
    # 최신 PyTorch 권장 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def select_device(device_arg: str) -> torch.device:
    """
    device_arg: "auto" | "cpu" | "cuda" | "cuda:0" | "cuda:1" ...
    """
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

def pick_amp_dtype(amp_dtype: str) -> torch.dtype:
    """
    amp_dtype: "auto"|"bf16"|"fp16"|"fp32"
    - auto: bf16 if compute capability >= 8.0 (Ampere+), else fp16
    - fp32: 사실상 AMP off
    """
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "fp32":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if (major, minor) >= (8, 0) else torch.float16


def main():
    p = argparse.ArgumentParser()
    # ----- data -----
    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None, help="CSV patch_path가 상대경로면 합칠 루트")
    p.add_argument("--use_hvg", action="store_true", help="HVG로 유전자 선택 (기본은 평균상위)")
    p.add_argument("--n_top_genes", type=int, default=541)

    # ----- model / foundation -----
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--with_lora", action="store_true",
                   help="HF 기반 백본(plip/phikon)에서만 활성화. timm 백본은 자동으로 스킵됨.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # ----- training -----
    p.add_argument("--mode", default="joint", choices=["gene","spatial","joint"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64, help="img↔gene 미니배치 크기")
    p.add_argument("--img_batch_size", type=int, default=256, help="img↔spatial(또는 joint에서 IS) 미니배치 크기")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--k", type=int, default=12, help="k for KNN (or Delaunay fallback KNN)")
    p.add_argument("--graph_type", default="knn", choices=["knn", "delaunay"],
                   help="Spatial graph type for img↔spatial contrastive.")
    p.add_argument("--save_dir", default="./ckpts")

    # ----- device/amp -----
    p.add_argument("--device", default="auto", help='"auto", "cpu", "cuda", "cuda:0" ...')
    p.add_argument("--gcn_device", default=None,
                   help='GCN device override: "cpu", "cuda", "cuda:0" ... (default: use --device)')
    p.add_argument("--amp", action="store_true", help="Enable AMP (bf16/fp16)")
    p.add_argument("--amp_dtype", default="auto", choices=["auto","bf16","fp16","fp32"],
                   help="AMP dtype; auto=bf16(Ampere+), else fp16. fp32 = no autocast.")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 0) Device & AMP
    device = select_device(args.device)
    setup_cuda_flags()
    amp_dtype = pick_amp_dtype(args.amp_dtype) if args.amp else torch.float32
    if args.gcn_device == "cpu":
        # GCN을 CPU로 돌리면 AMP를 끄므로, 이미지 인코더 입력도 FP32로 고정
        amp_dtype = torch.float32
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"[Device] {device}, AMP={args.amp}, dtype={amp_dtype}")

    # 1) Load data
    adata = load_h5ad_with_preproc(
        args.h5ad,
        use_hvg=args.use_hvg,
        n_top_genes=args.n_top_genes,
        x_key="x_centroid",
        y_key="y_centroid",
    )
    # root_dir 미지정 시 CSV가 있는 디렉터리를 기준으로 상대경로 해석
    map_df = load_patch_mapping_csv_xenium(
        args.csv,
        root_dir=args.root or os.path.dirname(args.csv),
        drop_missing=True,
    )

    # (1) id 정규화 + off-by-one 자동교정
    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    # (2) 교집합만 남기고 map_df 순서대로 좌표 추출
    map_df_aligned, coords_np = reorder_and_prune_by_obs(
        adata=adata, map_df=map_df, id_col="spot_id", coord_key="spatial"
    )

    # 2) Foundation & transforms
    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation = FoundationCls(weights_path=None)

    # --- LoRA: HF기반 백본만 허용 (timm 백본은 충돌 방지 차원에서 스킵) ---
    lora_allowed = args.enc_name in {"plip", "phikon"}  # conch_v1/open_clip도 일반적으로 peft 미호환
    if args.with_lora:
        if lora_allowed:
            foundation = attach_lora_to_foundation(
                foundation,
                backbone_name=args.enc_name,
                r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, bias="none"
            )
            print(f"[LoRA] enabled on backbone='{args.enc_name}'")
        else:
            print(f"[LoRA] skipped for backbone='{args.enc_name}' (timm/open-clip backbone not supported by peft).")

    # AMP dtype을 foundation.precision에 반영 (ImageFoundationAdapter에서 이 dtype으로 캐스팅)
    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)

    # 3) DataLoaders (※ img↔spatial은 순서 중요 → map_df_aligned 사용)
    train_loader, gene_dim = make_img_gene_dataloader(
        adata=adata, map_df=map_df_aligned,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_transform=image_tf
    )
    img_loader = make_img_only_dataloader(
        map_df=map_df_aligned,
        batch_size=max(1, args.img_batch_size),      # 풀배치 대신 미니배치(메모리 안정)
        num_workers=max(1, args.num_workers // 2),
        image_transform=image_tf,
        shuffle=False
    )

    # 4) Build spatial graph (coords_np는 map_df_aligned 순서와 일치)
    edge_index, edge_weight = build_spatial_graph(
        coords_np, k=args.k, use_weight=True, graph_type=args.graph_type
    )
    print(f"[Graph] type={args.graph_type}, k={args.k}, E={edge_index.size(1)}")

    # 5) Models (공유 img_encoder)
    img_encoder = ImageFoundationAdapter(foundation, proj_dim=args.proj_dim, normalize=False)
    gene_encoder = GeneEncoder(input_dim=gene_dim, model_size_omic='small',
                               proj_dim=args.proj_dim, normalize=False)
    gcn = SpatialGCN(in_dim=args.proj_dim, hidden_dim=256, out_dim=args.proj_dim, dropout=0.2)

    ig_model = img2gene_ContrastiveModel(img_encoder=img_encoder, gene_encoder=gene_encoder,
                                         proj_dim=args.proj_dim)
    is_model = img2spot_ContrastiveModel(img_encoder=img_encoder, gcn=gcn, temperature=0.15, lam=0.5)

    # 6) Train (각 train_* 내부에서 device/AMP 처리 및 ckpt 저장)
    common = dict(
        epochs=args.epochs,
        lr=1e-3,
        save_dir=args.save_dir,
        device=str(device),
        use_amp=use_amp,
    )
    if args.mode == "gene":
        train_contrastive(ig_model, train_loader, **common)
    elif args.mode == "spatial":
        train_img_spatial_contrastive(
            is_model, img_loader, edge_index=edge_index, edge_weight=edge_weight,
            gcn_device=args.gcn_device, **common
        )
    else:
        train_joint_img_gene_spatial(
            is_model=is_model, ig_model=ig_model,
            img_loader=img_loader, gene_loader=train_loader,
            edge_index=edge_index, edge_weight=edge_weight,
            lam_is=1.0, lam_ig=1.0, gcn_device=args.gcn_device, **common
        )

if __name__ == "__main__":
    main()
