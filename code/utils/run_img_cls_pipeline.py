#!/usr/bin/env python3
"""
Image classification pipeline (train / eval / infer) using cluster labels from h5ad.

Examples:
1) Train only
CUDA_VISIBLE_DEVICES=1 \
python Code_final/code/utils/run_img_cls_pipeline.py \
  --mode train \
  --train_h5ad /path/to/train.h5ad \
  --train_csv /path/to/spot_patch_map_256.csv \
  --train_root /path/to/expression \
  --cluster_col leiden \
  --enc_name hoptimus0 \
  --epochs 50 --batch_size 32 --num_workers 128 \
  --save_dir /data/project/banana9903_xenium/ckpts_img_cls2 \
  --device cuda:0 --amp --amp_dtype fp16 \
  --img_size 224 --cls_weight balanced

2) Eval only (using existing ckpt)
CUDA_VISIBLE_DEVICES=1 /data/project/banana9903_xenium/.conda-envs/SPAM/bin/python \
  /data/project/banana9903_xenium/Code_final/code/utils/run_img_cls_pipeline.py \
  --mode infer \
  --ckpt_dir /data/project/banana9903_xenium/ckpts_img_cls2 \
  --infer_h5ad /data/project/watercar99/Data/Spatial/xenium_rep1_io/data/merged_with_celltype_symbol_SPAM.h5ad \
  --infer_csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --infer_root /data/project/banana9903_xenium/contrast_test/expression \
  --enc_name hoptimus0 --proj_dim 256 --img_size 224 \
  --device cuda:0 --amp --amp_dtype fp16 \
  --batch_size 64 --num_workers 16 \
  --fast_h5ad \
  --infer_out_csv /data/project/banana9903_xenium/ckpts_img_cls2/infer_pred.csv


3) All (train -> eval -> infer)
CUDA_VISIBLE_DEVICES=1 \
python Code_final/code/utils/run_img_cls_pipeline.py \
  --mode all \
  --train_h5ad ... --train_csv ... --train_root ... \
  --eval_h5ad ... --eval_csv ... --eval_root ... \
  --infer_h5ad ... --infer_csv ... --infer_root ... \
  --cluster_col leiden \
  --enc_name hoptimus0 --proj_dim 256 --img_size 224 \
  --epochs 50 --batch_size 32 --num_workers 128 \
  --save_dir /data/project/banana9903_xenium/ckpts_img_cls2 \
  --device cuda:0 --amp --amp_dtype fp16 \
  --eval_out_csv /path/to/eval_pred.csv \
  --infer_out_csv /path/to/infer_pred.csv
"""

import os
import sys
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc

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
        labels: Optional[np.ndarray] = None,
        image_transform=None,
        img_size: Optional[int] = None,
    ):
        self.map_df = map_df.reset_index(drop=True)
        self.labels = None if labels is None else labels.astype(np.int64)
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

        if self.img_size is not None and img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        img = self.image_transform(img)

        if self.labels is None:
            return img
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


def _load_h5ad_minimal(h5ad_path: str, x_key: str = "x_centroid", y_key: str = "y_centroid"):
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()
    if (x_key in adata.obs.columns) and (y_key in adata.obs.columns):
        xy = adata.obs[[x_key, y_key]].to_numpy().astype(float)
        adata.obsm["spatial"] = xy
        xy_min, xy_max = xy.min(0), xy.max(0)
        adata.obsm["coord"] = (xy - xy_min) / (xy_max - xy_min + 1e-9)
    else:
        raise KeyError(f"x_key={x_key!r}, y_key={y_key!r} not found in adata.obs")
    return adata


def _load_and_align(
    h5ad_path: str,
    csv_path: str,
    root_dir: Optional[str],
    id_col: str,
    path_col: str,
    fast_h5ad: bool = False,
    max_samples: int = 0,
) -> Tuple["ad.AnnData", pd.DataFrame]:
    if fast_h5ad:
        adata = _load_h5ad_minimal(h5ad_path, x_key="x_centroid", y_key="y_centroid")
    else:
        adata = load_h5ad_with_preproc(
            h5ad_path,
            use_hvg=True,
            n_top_genes=541,
            x_key="x_centroid",
            y_key="y_centroid",
        )
    map_df = load_patch_mapping_csv_xenium(
        csv_path,
        root_dir=root_dir or os.path.dirname(csv_path),
        id_col=id_col,
        path_col=path_col,
        drop_missing=True,
    )
    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, _coords = reorder_and_prune_by_obs(
        adata=adata, map_df=map_df, id_col="spot_id", coord_key="spatial"
    )
    if map_df_aligned.shape[0] != adata.n_obs:
        obs_ids = map_df_aligned["spot_id"].astype(str).tolist()
        adata = adata[obs_ids].copy()
    if max_samples and map_df_aligned.shape[0] > max_samples:
        map_df_aligned = map_df_aligned.iloc[:max_samples].reset_index(drop=True)
        obs_ids = map_df_aligned["spot_id"].astype(str).tolist()
        adata = adata[obs_ids].copy()
    print(f"[data] h5ad={h5ad_path} n_obs={adata.n_obs} n_vars={adata.n_vars}")
    print(f"[data] map_df_aligned={map_df_aligned.shape[0]} rows")
    return adata, map_df_aligned


def _build_labels(
    adata,
    map_df_aligned: pd.DataFrame,
    cluster_col: Optional[str],
    label_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    clus_col = _pick_cluster_col(adata, cluster_col)
    labels_raw = adata.obs.loc[map_df_aligned["spot_id"], clus_col]
    valid_mask = ~labels_raw.isna()
    if not valid_mask.all():
        map_df_aligned = map_df_aligned.loc[valid_mask.values].reset_index(drop=True)
        labels_raw = labels_raw.loc[valid_mask]
        print(f"[warn] dropped {np.sum(~valid_mask.values)} samples with NaN labels")

    labels_str = labels_raw.astype(str).values
    if label_names is None:
        label_names = sorted(pd.unique(labels_str).tolist())

    label_to_idx = {name: i for i, name in enumerate(label_names)}
    known_mask = np.array([x in label_to_idx for x in labels_str], dtype=bool)
    if not np.all(known_mask):
        unknown = sorted(pd.unique(labels_str[~known_mask]).tolist())
        print(f"[warn] dropped {np.sum(~known_mask)} samples with unknown labels: {unknown}")
        labels_str = labels_str[known_mask]
        map_df_aligned = map_df_aligned.loc[known_mask].reset_index(drop=True)

    labels_idx = np.array([label_to_idx[x] for x in labels_str], dtype=np.int64)
    return labels_idx, label_names, map_df_aligned


def _build_model(
    enc_name: str,
    weights_path: Optional[str],
    proj_dim: int,
    num_classes: int,
    cls_head_hidden: int,
    cls_dropout: float,
    amp_dtype: torch.dtype,
    freeze_backbone: bool,
) -> Tuple[img2cluster_Classifier, Optional[torch.nn.Module]]:
    FoundationCls = inf_encoder_factory(enc_name)
    foundation = FoundationCls(weights_path=weights_path)
    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)

    img_encoder = ImageFoundationAdapter(foundation, proj_dim=proj_dim, normalize=False)
    if freeze_backbone:
        for p in img_encoder.foundation.parameters():
            p.requires_grad = False

    model = img2cluster_Classifier(
        img_encoder=img_encoder,
        num_classes=num_classes,
        hidden_dim=cls_head_hidden,
        dropout=cls_dropout,
    )
    return model, image_tf


def _infer_num_classes_from_classifier_sd(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    if "weight" in sd and sd["weight"].ndim == 2:
        return int(sd["weight"].shape[0])
    weight_keys = [k for k in sd.keys() if k.endswith(".weight")]
    if not weight_keys:
        return None
    # pick the layer with the highest numeric prefix if possible
    def _key_rank(k: str) -> int:
        head = k.split(".")[0]
        return int(head) if head.isdigit() else -1
    weight_keys.sort(key=_key_rank)
    w = sd[weight_keys[-1]]
    if w.ndim == 2:
        return int(w.shape[0])
    return None


def _load_label_names_and_cls_sd(ckpt_dir: str) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    cls_ckpt = os.path.join(ckpt_dir, "ImgClassifier.pt")
    label_path = os.path.join(ckpt_dir, "label_names.pt")
    if not os.path.isfile(cls_ckpt):
        raise FileNotFoundError(f"ImgClassifier.pt not found in {ckpt_dir}")

    label_names = None
    if os.path.isfile(label_path):
        obj = torch.load(label_path, map_location="cpu")
        if isinstance(obj, dict) and "label_names" in obj:
            label_names = obj["label_names"]
        elif isinstance(obj, (list, tuple)):
            label_names = list(obj)

    cls_sd = torch.load(cls_ckpt, map_location="cpu")
    if label_names is None:
        n = _infer_num_classes_from_classifier_sd(cls_sd)
        n = n or 0
        label_names = [f"class_{i}" for i in range(n)]
        print(f"[warn] label_names.pt not found. Using placeholders: {label_names}")
    return label_names, cls_sd


def _load_img_cls_ckpt(model, ckpt_dir: str, cls_sd: Optional[Dict[str, torch.Tensor]] = None) -> None:
    img_ckpt = os.path.join(ckpt_dir, "Image_pretrained.pt")

    if not os.path.isfile(img_ckpt):
        raise FileNotFoundError(f"Image_pretrained.pt not found in {ckpt_dir}")

    model.img_encoder.load_state_dict(torch.load(img_ckpt, map_location="cpu"), strict=False)
    if cls_sd is None:
        cls_sd = torch.load(os.path.join(ckpt_dir, "ImgClassifier.pt"), map_location="cpu")
    model.classifier.load_state_dict(cls_sd, strict=False)


def _predict(
    model: img2cluster_Classifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    model.to(device).eval()
    preds = []
    probs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(imgs)
                if not torch.isfinite(logits).all():
                    if not hasattr(_predict, "_warned_nan"):
                        print("[warn] Non-finite logits in AMP. Recomputing this batch in FP32.")
                        _predict._warned_nan = True
                    with torch.amp.autocast("cuda", enabled=False):
                        logits = model(imgs.float())
            else:
                logits = model(imgs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits even in FP32. Check input images / model weights.")
            p = torch.softmax(logits.float(), dim=1)
            preds.append(torch.argmax(p, dim=1).cpu().numpy())
            probs.append(torch.max(p, dim=1).values.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(probs, axis=0)


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    acc = float((y_true == y_pred).mean())
    # macro F1
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp == 0 and (fp == 0 or fn == 0):
            f1s.append(0.0)
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"acc": acc, "macro_f1": macro_f1}


def _save_predictions(
    out_csv: Optional[str],
    out_h5ad: Optional[str],
    adata,
    map_df: pd.DataFrame,
    pred_idx: np.ndarray,
    pred_prob: np.ndarray,
    label_names: List[str],
    true_idx: Optional[np.ndarray] = None,
    true_label: Optional[np.ndarray] = None,
):
    pred_label = np.array([label_names[i] if i < len(label_names) else f"class_{i}" for i in pred_idx])

    df = map_df.copy()
    df["pred_idx"] = pred_idx
    df["pred_label"] = pred_label
    df["pred_prob"] = pred_prob
    if true_idx is not None:
        df["true_idx"] = true_idx
    if true_label is not None:
        df["true_label"] = true_label

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[save] predictions csv -> {out_csv}")

    if out_h5ad:
        os.makedirs(os.path.dirname(out_h5ad) or ".", exist_ok=True)
        adata = adata.copy()
        adata.obs["pred_idx"] = pred_idx
        adata.obs["pred_label"] = pred_label
        adata.obs["pred_prob"] = pred_prob
        if true_label is not None:
            adata.obs["true_label"] = true_label
        if true_idx is not None:
            adata.obs["true_idx"] = true_idx
        adata.write(out_h5ad, compression="gzip")
        print(f"[save] predictions h5ad -> {out_h5ad}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="train", choices=["train", "eval", "infer", "all"])

    # common
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    p.add_argument("--weights_path", default=None)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cluster_col", default=None)

    # csv id/path columns
    p.add_argument("--id_col", default="cell_id")
    p.add_argument("--path_col", default="patch_path")
    p.add_argument("--fast_h5ad", action="store_true",
                   help="Skip normalization/HVG; only load obs/coords (faster)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Optional cap for samples (0=all)")

    # train
    p.add_argument("--train_h5ad", default=None)
    p.add_argument("--train_csv", default=None)
    p.add_argument("--train_root", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--save_dir", default="./ckpts_img_cls")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--cls_head_hidden", type=int, default=0)
    p.add_argument("--cls_dropout", type=float, default=0.0)
    p.add_argument("--cls_weight", default="none", choices=["none", "balanced"])

    # eval
    p.add_argument("--eval_h5ad", default=None)
    p.add_argument("--eval_csv", default=None)
    p.add_argument("--eval_root", default=None)
    p.add_argument("--eval_out_csv", default=None)
    p.add_argument("--eval_out_h5ad", default=None)

    # infer
    p.add_argument("--infer_h5ad", default=None)
    p.add_argument("--infer_csv", default=None)
    p.add_argument("--infer_root", default=None)
    p.add_argument("--infer_out_csv", default=None)
    p.add_argument("--infer_out_h5ad", default=None)

    # ckpt for eval/infer
    p.add_argument("--ckpt_dir", default=None, help="Path with Image_pretrained.pt / ImgClassifier.pt")
    args = p.parse_args()

    device = select_device(args.device)
    setup_cuda_flags()
    amp_dtype = pick_amp_dtype(args.amp_dtype) if args.amp else torch.float32
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"[Device] {device}, AMP={args.amp}, dtype={amp_dtype}")

    # ----------------------------
    # Train
    # ----------------------------
    label_names = None
    if args.mode in ("train", "all"):
        if not args.train_h5ad or not args.train_csv:
            raise ValueError("--train_h5ad and --train_csv are required for train")

        adata_tr, map_df_tr = _load_and_align(
            args.train_h5ad, args.train_csv, args.train_root, args.id_col, args.path_col,
            fast_h5ad=args.fast_h5ad, max_samples=args.max_samples
        )
        labels_idx, label_names, map_df_tr = _build_labels(
            adata_tr, map_df_tr, args.cluster_col, label_names=None
        )
        if len(label_names) < 2:
            raise ValueError(f"Need at least 2 classes, got {len(label_names)}: {label_names}")

        model, image_tf = _build_model(
            args.enc_name,
            args.weights_path,
            args.proj_dim,
            num_classes=len(label_names),
            cls_head_hidden=args.cls_head_hidden,
            cls_dropout=args.cls_dropout,
            amp_dtype=amp_dtype,
            freeze_backbone=args.freeze_backbone,
        )

        ds = ImgClusterDataset(map_df_tr, labels_idx, image_transform=image_tf, img_size=args.img_size)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        class_weights = None
        if args.cls_weight == "balanced":
            counts = np.bincount(labels_idx, minlength=len(label_names)).astype(np.float32)
            class_weights = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32)

        os.makedirs(args.save_dir, exist_ok=True)
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

    # ----------------------------
    # Eval
    # ----------------------------
    if args.mode in ("eval", "all"):
        ckpt_dir = args.ckpt_dir or args.save_dir
        if not ckpt_dir:
            raise ValueError("--ckpt_dir is required for eval")
        if not args.eval_h5ad or not args.eval_csv:
            if args.mode == "all":
                print("[warn] eval data not provided. Skipping eval.")
            else:
                raise ValueError("--eval_h5ad and --eval_csv are required for eval")
        else:
            adata_ev, map_df_ev = _load_and_align(
                args.eval_h5ad, args.eval_csv, args.eval_root, args.id_col, args.path_col,
                fast_h5ad=args.fast_h5ad, max_samples=args.max_samples
            )

            label_names, cls_sd = _load_label_names_and_cls_sd(ckpt_dir)
            if len(label_names) == 0:
                raise ValueError("Failed to infer num_classes from checkpoint. Check label_names.pt or classifier ckpt.")
            model, image_tf = _build_model(
                args.enc_name,
                args.weights_path,
                args.proj_dim,
                num_classes=len(label_names),
                cls_head_hidden=args.cls_head_hidden,
                cls_dropout=args.cls_dropout,
                amp_dtype=amp_dtype,
                freeze_backbone=False,
            )
            _load_img_cls_ckpt(model, ckpt_dir, cls_sd=cls_sd)

            labels_idx, label_names, map_df_ev = _build_labels(
                adata_ev, map_df_ev, args.cluster_col, label_names=label_names
            )
            if len(labels_idx) == 0:
                raise ValueError("No valid eval samples after label filtering.")

            ds = ImgClusterDataset(map_df_ev, labels_idx, image_transform=image_tf, img_size=args.img_size)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )

            pred_idx, pred_prob = _predict(model, loader, device, use_amp, amp_dtype)
            metrics = _eval_metrics(labels_idx, pred_idx, num_classes=len(label_names))
            print(f"[eval] acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f}")

            true_label = np.array([label_names[i] for i in labels_idx])
            _save_predictions(
                args.eval_out_csv,
                args.eval_out_h5ad,
                adata_ev,
                map_df_ev,
                pred_idx,
                pred_prob,
                label_names,
                true_idx=labels_idx,
                true_label=true_label,
            )

    # ----------------------------
    # Infer
    # ----------------------------
    if args.mode in ("infer", "all"):
        ckpt_dir = args.ckpt_dir or args.save_dir
        if not ckpt_dir:
            raise ValueError("--ckpt_dir is required for infer")
        if not args.infer_h5ad or not args.infer_csv:
            if args.mode == "all":
                print("[warn] infer data not provided. Skipping infer.")
            else:
                raise ValueError("--infer_h5ad and --infer_csv are required for infer")
        else:
            adata_inf, map_df_inf = _load_and_align(
                args.infer_h5ad, args.infer_csv, args.infer_root, args.id_col, args.path_col,
                fast_h5ad=args.fast_h5ad, max_samples=args.max_samples
            )

            label_names, cls_sd = _load_label_names_and_cls_sd(ckpt_dir)
            if len(label_names) == 0:
                raise ValueError("Failed to infer num_classes from checkpoint. Check label_names.pt or classifier ckpt.")
            model, image_tf = _build_model(
                args.enc_name,
                args.weights_path,
                args.proj_dim,
                num_classes=len(label_names),
                cls_head_hidden=args.cls_head_hidden,
                cls_dropout=args.cls_dropout,
                amp_dtype=amp_dtype,
                freeze_backbone=False,
            )
            _load_img_cls_ckpt(model, ckpt_dir, cls_sd=cls_sd)

            ds = ImgClusterDataset(map_df_inf, labels=None, image_transform=image_tf, img_size=args.img_size)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )

            pred_idx, pred_prob = _predict(model, loader, device, use_amp, amp_dtype)
            _save_predictions(
                args.infer_out_csv,
                args.infer_out_h5ad,
                adata_inf,
                map_df_inf,
                pred_idx,
                pred_prob,
                label_names,
            )


if __name__ == "__main__":
    main()
