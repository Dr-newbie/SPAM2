import os, sys, time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 표준 모듈 'code' 이름 충돌 방지
sys.modules.pop("code", None)

from code.models.Foundations import inf_encoder_factory
from code.utils.dataset_utils import (
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)

class ImageMapDataset(Dataset):
    def __init__(
        self,
        map_df: pd.DataFrame,
        image_transform: nn.Module | None = None,
        id_col: str = "spot_id",
    ):
        super().__init__()
        self.map_df = map_df.reset_index(drop=True)
        self.image_transform = image_transform
        self.id_col = id_col

    def __len__(self):
        return len(self.map_df)

    def __getitem__(self, idx: int):
        row = self.map_df.iloc[idx]
        img_path = row["patch_path"]
        img = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)
        else:
            img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        sample_id = str(row[self.id_col]) if self.id_col in self.map_df.columns else str(idx)
        return img, sample_id


def gather_image_representations(
    model: nn.Module,
    img_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    normalize: bool = True,
    show_progress: bool = True,
    desc: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    이미지 패치들의 표현을 모아서 반환합니다.
    
    Args:
        model (nn.Module): 이미지 표현 추출 모델
        img_loader (DataLoader): 이미지 패치들을 로드하는 DataLoader
        device (torch.device): 연산에 사용할 디바이스
        amp_dtype (torch.dtype): 자동 혼합 정밀도에 사용할 데이터 타입 (기본값: torch.float16)
    
    Returns:
        (np.ndarray, list[str]): (이미지 표현 배열, 샘플 id 리스트)
    """
    model.eval()
    all_representations = []
    all_ids: list[str] = []
    target_dtype = amp_dtype if amp_dtype is not None else getattr(model, "precision", torch.float32)
    if device.type == "cpu":
        target_dtype = torch.float32
    use_amp = device.type == "cuda" and target_dtype in (torch.float16, torch.bfloat16)

    with torch.no_grad():
        it = img_loader
        if show_progress:
            it = tqdm(img_loader, desc=desc or "embed", leave=False)
        for batch in it:
            if isinstance(batch, (list, tuple)):
                images, ids = batch
            else:
                images = batch["image"]
                ids = batch.get("id", None)

            images = images.to(device, dtype=target_dtype, non_blocking=True)

            ctx = torch.autocast(device_type="cuda", dtype=target_dtype, enabled=use_amp)
            with ctx:
                img_reps = model(images)
            if normalize:
                img_reps = F.normalize(img_reps, dim=1)

            all_representations.append(img_reps.float().cpu().numpy())
            if ids is None:
                all_ids.extend([str(i) for i in range(len(images))])
            else:
                all_ids.extend([str(x) for x in ids])

    all_representations = np.concatenate(all_representations, axis=0)
    return all_representations, all_ids

def top1_cosine_match(
    train_rep: np.ndarray,
    test_rep: np.ndarray,
    chunk_size: int = 1024,
    device: str = "cpu",
    show_progress: bool = True,
    desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity 기반으로 test->train top1 매칭.
    반환: (best_indices, best_scores)
    """
    dev = torch.device(device)
    train_t = torch.from_numpy(train_rep).to(dev, dtype=torch.float32)
    test_t = torch.from_numpy(test_rep).to(dev, dtype=torch.float32)

    train_t = F.normalize(train_t, dim=1)
    test_t = F.normalize(test_t, dim=1)

    best_idx = torch.empty(test_t.size(0), dtype=torch.long, device=dev)
    best_val = torch.empty(test_t.size(0), dtype=torch.float32, device=dev)

    steps = range(0, test_t.size(0), chunk_size)
    if show_progress:
        steps = tqdm(steps, desc=desc or "similarity", leave=False)
    for s in steps:
        e = min(test_t.size(0), s + chunk_size)
        sims = test_t[s:e] @ train_t.t()
        vals, idxs = torch.max(sims, dim=1)
        best_idx[s:e] = idxs
        best_val[s:e] = vals

    return best_idx.cpu().numpy(), best_val.cpu().numpy()


def _expression_matrix_for_spots(adata: ad.AnnData, spot_ids: list[str]) -> np.ndarray:
    obs_to_pos = {str(s): i for i, s in enumerate(adata.obs_names.astype(str).tolist())}
    idxs = [obs_to_pos[str(s)] for s in spot_ids]
    X = adata.X[idxs, :]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def predict_expression_by_image_similarity(
    adata: ad.AnnData,
    train_map_df: pd.DataFrame,
    test_map_df: pd.DataFrame,
    enc_name: str = "uni_v1",
    weights_path: str | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
    amp_dtype: torch.dtype | None = None,
    sim_chunk: int = 1024,
    sim_device: str = "cpu",
    meta_cols: list[str] | None = None,
    img_encoder: nn.Module | None = None,
    image_transform: nn.Module | None = None,
) -> dict:
    """
    train/test 이미지 패치 임베딩을 뽑아 cosine similarity로 매칭 후,
    test 각 패치에 가장 유사한 train cell의 expression을 반환.
    """
    if test_map_df is None:
        raise ValueError("test_map_df is required for similarity matching.")
    device_t = torch.device(device)

    train_map_df = align_map_ids_to_obs(adata, train_map_df, id_col="spot_id")
    train_map_df, _ = reorder_and_prune_by_obs(adata, train_map_df, id_col="spot_id")

    if img_encoder is None:
        FoundationCls = inf_encoder_factory(enc_name)
        img_encoder = FoundationCls(weights_path=weights_path)
        if image_transform is None:
            image_transform = getattr(img_encoder, "eval_transforms", None)
        if amp_dtype is None:
            amp_dtype = getattr(img_encoder, "precision", torch.float32)
        img_encoder.precision = amp_dtype
    else:
        if image_transform is None:
            image_transform = getattr(img_encoder, "eval_transforms", None)
        if amp_dtype is None:
            amp_dtype = getattr(img_encoder, "precision", torch.float32)

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

    best_idx, best_sim = top1_cosine_match(
        train_rep, test_rep, chunk_size=sim_chunk, device=sim_device,
        show_progress=True, desc="similarity"
    )

    train_spot_ids = train_map_df["spot_id"].astype(str).tolist()
    expr_train = _expression_matrix_for_spots(adata, train_spot_ids)
    pred_expr = expr_train[best_idx]
    matched_train_spot_ids = np.asarray(train_spot_ids, dtype=object)[best_idx]

    matched_meta = None
    if meta_cols:
        cols = [c for c in meta_cols if c in train_map_df.columns]
        if cols:
            matched_meta = train_map_df.iloc[best_idx][cols].reset_index(drop=True)
        else:
            print(f"[WARN] None of meta_cols found in train_map_df: {meta_cols}")

    return {
        "pred_expression": pred_expr,
        "best_train_index": best_idx,
        "best_similarity": best_sim,
        "matched_train_spot_id": matched_train_spot_ids,
        "matched_train_meta": matched_meta,
        "test_ids": test_ids,
        "train_ids": train_ids,
        "train_spot_ids": train_spot_ids,
    }
