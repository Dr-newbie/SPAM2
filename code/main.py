#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified main.py — 하나의 진입점으로 3 파이프라인 실행
  1) pretrain   : code/utils/run_contrastive.py
  2) finetune   : code/utils/run_finetuning.py
  3) inference  : code/utils/run_inference.py

예시:
# 1) Contrastive Pretraining (joint)
python main.py pretrain \
  --h5ad /data/project/banana9903_xenium/xenium_rep1_io/data/merged_with_celltype_raw.h5ad \
  --csv  /data/project/banana9903_xenium/xenium_rep1_io/patches_224/spot_patch_map_224.csv \
  --root  \
  --enc_name uni_v1 \
  --mode joint \
  --epochs 500 --batch_size 64 --img_batch_size 256 \
  --save_dir ./ckpts \
  --device auto --amp

# 2) Finetuning (ZINB) + PT 가중치 주입
python main.py finetune \
  --h5ad /path/section.h5ad \
  --csv  /path/patch_map.csv \
  --root /path \
  --enc_name uni_v1 \
  --epochs 20 --batch_size 128 \
  --k 12 --lr 3e-4 --weight_decay 0.05 \
  --device auto --amp --save_dir ./ckpts_cross \
  --pt_img_backbone ./ckpts/joint/best_img_encoder.pt \
  --pt_ig ./ckpts/joint/best_ig_model.pt \
  --pt_is ./ckpts/joint/best_is_model.pt

# 3) Inference (train gene-wise mean vector 사용)
python main.py inference \
  --h5ad /path/section_eval.h5ad \
  --csv  /path/patch_map.csv \
  --root /path \
  --train_h5ad /path/section_train.h5ad \
  --enc_name uni_v1 \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 512 --heads 8 \
  --k 12 --device auto --amp \
  --ft_model_ckpt ./ckpts_cross/best/model.pt \
  --pt_img_backbone ./ckpts/joint/best_img_encoder.pt \
  --pt_ig ./ckpts/joint/best_ig_model.pt \
  --pt_is ./ckpts/joint/best_is_model.pt \
  --out_h5ad ./inference/recovered_eval.h5ad
"""

import os
import sys
import argparse
import importlib.util
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR))  # e.g., /path/to/Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 표준 모듈명 'code' 충돌 방지
sys.modules.pop("code", None)

# ------------------------------------------------------------
# utils: import 외부 파일의 main()을 호출
# ------------------------------------------------------------
def _load_module_from_path(mod_name: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for {mod_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module

def _run_foreign_main(main_func, argv: List[str]):
    prev_argv = sys.argv[:]
    try:
        sys.argv = [main_func.__module__] + argv
        return main_func()
    finally:
        sys.argv = prev_argv

# ------------------------------------------------------------
# Subcommand: pretrain → code/utils/run_contrastive.py
# ------------------------------------------------------------
def add_pretrain_args(sp: argparse.ArgumentParser):
    sp.add_argument("--h5ad", required=True)
    sp.add_argument("--csv", required=True)
    sp.add_argument("--root", default=None)
    sp.add_argument("--use_hvg", action="store_true")
    sp.add_argument("--n_top_genes", type=int, default=541)

    sp.add_argument("--enc_name", default="uni_v1",
                    choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    sp.add_argument("--proj_dim", type=int, default=256)
    sp.add_argument("--with_lora", action="store_true")
    sp.add_argument("--lora_r", type=int, default=16)
    sp.add_argument("--lora_alpha", type=int, default=32)
    sp.add_argument("--lora_dropout", type=float, default=0.05)

    sp.add_argument("--mode", default="joint", choices=["gene","spatial","joint"])
    sp.add_argument("--epochs", type=int, default=50)
    sp.add_argument("--batch_size", type=int, default=64)
    sp.add_argument("--img_batch_size", type=int, default=256)
    sp.add_argument("--num_workers", type=int, default=8)
    sp.add_argument("--k", type=int, default=12)
    sp.add_argument("--save_dir", default="./ckpts")

    sp.add_argument("--device", default="auto")
    sp.add_argument("--amp", action="store_true")
    sp.add_argument("--amp_dtype", default="auto", choices=["auto","bf16","fp16","fp32"])

def run_pretrain(args: argparse.Namespace):
    target_path = os.path.join(PROJECT_ROOT, "utils", "run_contrastive.py")
    mod = _load_module_from_path("run_contrastive", target_path)
    argv = [
        "--h5ad", args.h5ad,
        "--csv", args.csv,
        "--root", args.root if args.root is not None else "",
        "--enc_name", args.enc_name,
        "--proj_dim", str(args.proj_dim),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--img_batch_size", str(args.img_batch_size),
        "--num_workers", str(args.num_workers),
        "--k", str(args.k),
        "--save_dir", args.save_dir,
        "--device", args.device,
        "--amp_dtype", args.amp_dtype,
        "--mode", args.mode,
    ]
    if args.use_hvg:
        argv.append("--use_hvg")
    if args.with_lora:
        argv += ["--with_lora", "--lora_r", str(args.lora_r),
                 "--lora_alpha", str(args.lora_alpha),
                 "--lora_dropout", str(args.lora_dropout)]
    if args.amp:
        argv.append("--amp")

    argv = [x for x in argv if x != ""]
    return _run_foreign_main(mod.main, argv)

# ------------------------------------------------------------
# Subcommand: finetune → code/utils/run_finetuning.py
# ------------------------------------------------------------
def add_finetune_args(sp: argparse.ArgumentParser):
    sp.add_argument("--h5ad", required=True)
    sp.add_argument("--csv", required=True)
    sp.add_argument("--root", default=None)
    sp.add_argument("--id_col", default="spot_id")

    sp.add_argument("--enc_name", default="uni_v1",
                    choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])

    sp.add_argument("--epochs", type=int, default=20)
    sp.add_argument("--batch_size", type=int, default=128)
    sp.add_argument("--num_workers", type=int, default=8)

    sp.add_argument("--proj_dim_gene", type=int, default=256)
    sp.add_argument("--proj_dim_spot", type=int, default=256)
    sp.add_argument("--fuse_dim", type=int, default=512)
    sp.add_argument("--heads", type=int, default=8)
    sp.add_argument("--dropout", type=float, default=0.1)
    sp.add_argument("--merge", default="gated-sum", choices=["gated-sum","concat-proj"])

    sp.add_argument("--k", type=int, default=12)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--weight_decay", type=float, default=0.05)

    sp.add_argument("--device", default="auto")
    sp.add_argument("--amp", action="store_true")
    sp.add_argument("--save_dir", default="./ckpts_cross")

    # ===== pretraining weights for finetune =====
    sp.add_argument("--pt_img_backbone", default=None, help="pretrained image backbone ckpt")
    sp.add_argument("--pt_ig", default=None, help="pretrained img↔gene ckpt")
    sp.add_argument("--pt_is", default=None, help="pretrained img↔spatial ckpt")

def run_finetune(args: argparse.Namespace):
    target_path = os.path.join(PROJECT_ROOT, "utils", "run_finetuning.py")
    mod = _load_module_from_path("run_finetuning", target_path)

    argv = [
        "--h5ad", args.h5ad,
        "--csv", args.csv,
        "--root", args.root if args.root is not None else "",
        "--id_col", args.id_col,
        "--enc_name", args.enc_name,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--proj_dim_gene", str(args.proj_dim_gene),
        "--proj_dim_spot", str(args.proj_dim_spot),
        "--fuse_dim", str(args.fuse_dim),
        "--heads", str(args.heads),
        "--dropout", str(args.dropout),
        "--merge", args.merge,
        "--k", str(args.k),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--device", args.device,
        "--save_dir", args.save_dir,
    ]
    if args.amp:
        argv.append("--amp")
    # PT weights (옵션)
    if args.pt_img_backbone:
        argv += ["--pt_img_backbone", args.pt_img_backbone]
    if args.pt_ig:
        argv += ["--pt_ig", args.pt_ig]
    if args.pt_is:
        argv += ["--pt_is", args.pt_is]

    argv = [x for x in argv if x != ""]
    return _run_foreign_main(mod.main, argv)

# ------------------------------------------------------------
# Subcommand: inference → code/utils/run_inference.py
# ------------------------------------------------------------
def add_inference_args(sp: argparse.ArgumentParser):
    sp.add_argument("--h5ad", required=True, help="추론 대상 h5ad")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--root", default=None)
    sp.add_argument("--train_h5ad", required=True, help="gene-wise mean 계산용 train h5ad")

    sp.add_argument("--enc_name", default="uni_v1",
                    choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])

    sp.add_argument("--proj_dim_gene", type=int, default=256)
    sp.add_argument("--proj_dim_spot", type=int, default=256)
    sp.add_argument("--fuse_dim", type=int, default=512)
    sp.add_argument("--heads", type=int, default=8)
    sp.add_argument("--dropout", type=float, default=0.1)
    sp.add_argument("--merge", default="gated-sum", choices=["gated-sum","concat-proj"])

    sp.add_argument("--k", type=int, default=12)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--amp", action="store_true")

    sp.add_argument("--ft_model_ckpt", default=None, help="finetune된 전체 모델 state_dict (model.pt)")
    sp.add_argument("--pt_img_backbone", default=None)
    sp.add_argument("--pt_ig", default=None)
    sp.add_argument("--pt_is", default=None)

    sp.add_argument("--batch_size", type=int, default=256)
    sp.add_argument("--num_workers", type=int, default=8)

    sp.add_argument("--out_h5ad", required=True)

def run_inference(args: argparse.Namespace):
    target_path = os.path.join(PROJECT_ROOT, "code", "utils", "run_inference.py")
    mod = _load_module_from_path("run_inference", target_path)

    argv = [
        "--h5ad", args.h5ad,
        "--csv", args.csv,
        "--root", args.root if args.root is not None else "",
        "--train_h5ad", args.train_h5ad,
        "--enc_name", args.enc_name,
        "--proj_dim_gene", str(args.proj_dim_gene),
        "--proj_dim_spot", str(args.proj_dim_spot),
        "--fuse_dim", str(args.fuse_dim),
        "--heads", str(args.heads),
        "--dropout", str(args.dropout),
        "--merge", args.merge,
        "--k", str(args.k),
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--out_h5ad", args.out_h5ad,
    ]
    if args.amp:
        argv.append("--amp")
    # ckpt 옵션
    if args.ft_model_ckpt:
        argv += ["--ft_model_ckpt", args.ft_model_ckpt]
    if args.pt_img_backbone:
        argv += ["--pt_img_backbone", args.pt_img_backbone]
    if args.pt_ig:
        argv += ["--pt_ig", args.pt_ig]
    if args.pt_is:
        argv += ["--pt_is", args.pt_is]

    argv = [x for x in argv if x != ""]
    return _run_foreign_main(mod.main, argv)

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified runner: pretrain / finetune / inference")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_pre = sub.add_parser("pretrain", help="Contrastive pretraining (code/utils/run_contrastive.py)")
    add_pretrain_args(sp_pre)

    sp_ft = sub.add_parser("finetune", help="Cross-Attn + ZINB finetuning (code/utils/run_finetuning.py)")
    add_finetune_args(sp_ft)

    sp_ft = sub.add_parser("finetune2", help="Cross-Attn + ZINB finetuning (code/utils/run_finetuning.py)")
    add_finetune_args(sp_ft)

    sp_inf = sub.add_parser("inference", help="Run inference to save predicted expression h5ad (code/utils/run_inference.py)")
    add_inference_args(sp_inf)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "pretrain":
        return run_pretrain(args)
    elif args.cmd == "finetune":
        return run_finetune(args)
    elif args.cmd == "finetune2":
        return run_finetune(args)
    elif args.cmd == "inference":
        return run_inference(args)
    else:
        raise ValueError(f"Unknown subcommand: {args.cmd}")

if __name__ == "__main__":
    main()
