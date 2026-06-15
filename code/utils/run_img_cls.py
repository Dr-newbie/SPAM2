#!/usr/bin/env python3
"""
Image-only supervised training using cluster labels from h5ad.

Example:
CUDA_VISIBLE_DEVICES=1 \
/data/project/banana9903_xenium/.conda-envs/SPAM/bin/python /data/project/banana9903_xenium/Code_final/code/utils/run_img_cls.py \
  --h5ad /data/project/watercar99/Data/Spatial/xenium_rep1_io/data/merged_with_celltype_symbol_SPAM.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --enc_name hoptimus0 \
  --cluster_col leiden \
  --epochs 50 --batch_size 32 --num_workers 128 \
  --save_dir /data/project/banana9903_xenium/ckpts_img_cls2 \
  --device cuda:0  --amp_dtype fp16 \
  --img_size 224 --cls_weight balanced
"""

import os
import sys
import argparse
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # /Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 표준모듈 'code'와 충돌 방지
sys.modules.pop("code", None)

from code.models.Foundations import inf_encoder_factory  # noqa: E402
from code.utils.contrastive import (  # noqa: E402
    ImageFoundationAdapter,
    img2cluster_Classifier,
    train_img_classification,
)
from code.utils.dataset_utils import (  # noqa: E402
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)


# ----------------------------
# Device & AMP helpers
# ----------------------------
def setup_cuda_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


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


def pick_amp_dtype(amp_dtype: str) -> torch.dtype:
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


# ----------------------------
# Dataset
# ----------------------------
_DEFAULT_IMG_TF = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class ImgClusterDataset(Dataset):
    def __init__(
        self,
        map_df: pd.DataFrame,
        labels: np.ndarray,
        image_transform=None,
        img_size: Optional[int] = None,
    ):
        self.map_df = map_df.reset_index(drop=True)
        self.labels = labels.astype(np.int64)
        self.image_transform = image_transform if image_transform is not None else _DEFAULT_IMG_TF
        self.img_size = img_size

    def __len__(self):
        return len(self.map_df)

    def __getitem__(self, idx: int):
        img_path = self.map_df.iloc[idx]["patch_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[warn] image not found: {img_path} — using blank placeholder")
            size = self.img_size if self.img_size is not None else 224
            img = Image.new("RGB", (size, size), (0, 0, 0))
        except Exception as e:
            print(f"[warn] failed to open image {img_path}: {e} — using blank placeholder")
            size = self.img_size if self.img_size is not None else 224
            img = Image.new("RGB", (size, size), (0, 0, 0))

        if self.img_size is not None:
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        img = self.image_transform(img)
        label = int(self.labels[idx])
        return img, torch.tensor(label, dtype=torch.long)


def _pick_cluster_col(adata, cluster_col: Optional[str]) -> str:
    cols = list(adata.obs.columns)
    cols_lower = {c.lower(): c for c in cols}
    if cluster_col:
        key = cluster_col.lower()
        if key in cols_lower:
            return cols_lower[key]
    cand = ["leiden", "cluster", "clusters", "seurat_clusters", "kmeans", "cluster_k8"]
    for c in cand:
        if c in cols_lower:
            return cols_lower[c]
    raise KeyError(f"cluster column not found in adata.obs. Available: {cols}")


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None, help="CSV patch_path가 상대경로면 합칠 루트")
    p.add_argument("--cluster_col", default=None, help="h5ad obs 클러스터 컬럼 (없으면 자동 탐색)")

    # foundation
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    p.add_argument("--weights_path", default=None, help="foundation weights path (optional)")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--img_size", type=int, default=None, help="이미지 패치 크기 강제 (예: 224)")

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir", default="./ckpts_img_cls")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--cls_head_hidden", type=int, default=0, help="classifier hidden dim (0이면 linear)")
    p.add_argument("--cls_dropout", type=float, default=0.0)
    p.add_argument("--cls_weight", default="none", choices=["none", "balanced"])

    # device/amp
    p.add_argument("--device", default="auto", help='"auto", "cpu", "cuda", "cuda:0" ...')
    p.add_argument("--amp", action="store_true", help="Enable AMP (bf16/fp16)")
    p.add_argument("--amp_dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = select_device(args.device)
    setup_cuda_flags()
    amp_dtype = pick_amp_dtype(args.amp_dtype) if args.amp else torch.float32
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"[Device] {device}, AMP={args.amp}, dtype={amp_dtype}")

    # 1) Load data + align IDs
    adata = load_h5ad_with_preproc(
        args.h5ad,
        use_hvg=True,
        n_top_genes=541,
        x_key="x_centroid",
        y_key="y_centroid",
    )
    map_df = load_patch_mapping_csv_xenium(
        args.csv,
        root_dir=args.root or os.path.dirname(args.csv),
        drop_missing=True,
    )
    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, _coords = reorder_and_prune_by_obs(
        adata=adata, map_df=map_df, id_col="spot_id", coord_key="spatial"
    )

    # 2) Labels
    clus_col = _pick_cluster_col(adata, args.cluster_col)
    labels_raw = adata.obs.loc[map_df_aligned["spot_id"], clus_col]
    valid_mask = ~labels_raw.isna()
    if not valid_mask.all():
        map_df_aligned = map_df_aligned.loc[valid_mask.values].reset_index(drop=True)
        labels_raw = labels_raw.loc[valid_mask]
        print(f"[warn] dropped {np.sum(~valid_mask.values)} samples with NaN labels")

    labels_str = labels_raw.astype(str).values
    label_names = sorted(pd.unique(labels_str).tolist())
    if len(label_names) < 2:
        raise ValueError(f"Need at least 2 classes, got {len(label_names)}: {label_names}")
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    labels_idx = np.array([label_to_idx[x] for x in labels_str], dtype=np.int64)

    # 3) Foundation & dataset
    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation = FoundationCls(weights_path=args.weights_path)
    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)

    ds = ImgClusterDataset(
        map_df_aligned,
        labels_idx,
        image_transform=image_tf,
        img_size=args.img_size,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    img_encoder = ImageFoundationAdapter(foundation, proj_dim=args.proj_dim, normalize=False)
    if args.freeze_backbone:
        for p in img_encoder.foundation.parameters():
            p.requires_grad = False

    class_weights = None
    if args.cls_weight == "balanced":
        counts = np.bincount(labels_idx, minlength=len(label_names)).astype(np.float32)
        class_weights = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32)

    model = img2cluster_Classifier(
        img_encoder=img_encoder,
        num_classes=len(label_names),
        hidden_dim=args.cls_head_hidden,
        dropout=args.cls_dropout,
    )

    # 4) Train
    train_img_classification(
        model,
        loader,
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=0.0,
        max_norm=0.0,
        use_amp=use_amp,
        device=str(device),
        log_every=args.log_every,
        save_dir=args.save_dir,
        class_weights=class_weights,
        label_names=label_names,
    )


if __name__ == "__main__":
    main()
