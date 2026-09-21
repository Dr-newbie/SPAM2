# dataset_utils.py
# -*- coding: utf-8 -*-

import os
import warnings
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import scanpy as sc

try:
    from utils.gene_panel import drop_control_probes
except ModuleNotFoundError:
    from code.utils.gene_panel import drop_control_probes
import anndata as ad


# ----------------------------
# 이미지 유틸
# ----------------------------
_DEFAULT_IMG_TF = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def _check_file(p: str):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"File not found: {p}")

def _infer_spot_col(df: pd.DataFrame) -> str:
    cand = ["spot_id", "barcode", "cell_id", "obs_name", "id"]
    for c in cand:
        if c in df.columns:
            return c
    return df.columns[0]

def _infer_path_col(df: pd.DataFrame) -> str:
    cand = ["patch_path", "image_path", "path", "filepath", "file"]
    for c in cand:
        if c in df.columns:
            return c
    if df.shape[1] > 1:
        return df.columns[1]
    raise ValueError("Cannot infer image path column from CSV.")


# ----------------------------
# h5ad 로드 + 전처리
# ----------------------------
def load_h5ad_with_preproc(
    h5ad_path: str,
    use_hvg: bool = True,
    n_top_genes: int = 1000,
    log1p: bool = True,
    target_sum: float = 1e4,
    x_key: str = "x_centroid",
    y_key: str = "y_centroid",
    drop_controls: bool = True,
) -> ad.AnnData:
    """
    - HVG 사용 시: seurat_v3로 상위 n_top_genes 선정
    - 미사용 시: 평균 발현 상위 n_top_genes 선택
    - obs의 x_key, y_key를 사용해 adata.obsm['spatial']와 ['coord'] 설정
    """
    _check_file(h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    # Drop Xenium control probes before anything else. They are near-zero by
    # design (0.07% of counts on the breast panel), are not valid ZINB targets,
    # and would otherwise contribute 228 unpredictable columns to every
    # per-gene metric average. Doing it here also means the size factor below
    # measures transcript depth, not depth plus control noise.
    if drop_controls:
        adata = drop_control_probes(adata, verbose=True)

    # Keep the observed integer counts and the per-cell library size BEFORE
    # normalization. The ZINB likelihood is defined on counts, and
    # normalize_total() is destructive: log1p can be inverted with expm1 but the
    # library-size division cannot, so a loss computed after this point can
    # never recover sequencing depth. adata.X stays log1p(CP10K) for the
    # encoders; the loss reads adata.layers["counts"] and obs["size_factor"].
    _counts = adata.X.copy()
    _lib = np.asarray(_counts.sum(axis=1)).ravel().astype(np.float64)
    adata.layers["counts"] = _counts
    # s_i = (sum_g x_ig) / target_sum, so that s_i * mu_hat lives on the count
    # scale while mu_hat stays on the normalized scale the decoder was fit to.
    adata.obs["size_factor"] = np.maximum(_lib, 1.0) / float(target_sum)
    adata.obs["library_size"] = _lib

    sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)

    if use_hvg:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        adata.var["selected"] = adata.var["highly_variable"].copy()
    else:
        gene_means = np.array(adata.X.mean(axis=0)).ravel()
        top_idx = np.argsort(gene_means)[::-1][:n_top_genes]
        selected = np.zeros(adata.n_vars, dtype=bool)
        selected[top_idx] = True
        adata.var["selected"] = selected

    if (x_key in adata.obs.columns) and (y_key in adata.obs.columns):
        xy = adata.obs[[x_key, y_key]].to_numpy().astype(float)
        adata.obsm["spatial"] = xy
        xy_min, xy_max = xy.min(0), xy.max(0)
        adata.obsm["coord"] = (xy - xy_min) / (xy_max - xy_min + 1e-9)
    else:
        raise KeyError(f"x_key={x_key!r}, y_key={y_key!r} not found in adata.obs")

    return adata


# ----------------------------
# 매핑 CSV 로드 (일반)
# ----------------------------
def load_patch_mapping_csv(csv_path: str) -> pd.DataFrame:
    """
    CSV는 최소한 (spot_id / patch_path) 두 컬럼을 가져야 함.
    컬럼명이 다르면 자동 추론.
    반환: ['spot_id', 'patch_path']로 정규화된 DataFrame
    """
    _check_file(csv_path)
    df = pd.read_csv(csv_path)
    spot_col = _infer_spot_col(df)
    path_col = _infer_path_col(df)

    out = df[[spot_col, path_col]].copy()
    out.columns = ["spot_id", "patch_path"]
    for p in out["patch_path"].tolist():
        _check_file(p)
    return out


# ----------------------------
# Xenium 스타일 CSV 로더 (좌표 포함 가능)
# ----------------------------
def load_patch_mapping_csv_xenium(
    csv_path: str,
    root_dir: str | None = None,
    id_col: str = "cell_id",
    path_col: str = "patch_path",
    make_absolute: bool = True,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """
    스크린샷과 같은 CSV 전용 로더.
    반환: DataFrame[spot_id, patch_path, x_he_px, y_he_px] (좌표 컬럼은 있으면 포함)
    - root_dir가 주어지면 patch_path가 상대경로일 때 root_dir와 join해서 절대경로로 만듦.
    """
    _check_file(csv_path)
    df = pd.read_csv(csv_path)

    if id_col not in df.columns or path_col not in df.columns:
        raise KeyError(f"{id_col!r} 또는 {path_col!r} 컬럼을 찾을 수 없음. 실제 컬럼: {df.columns.tolist()}")

    out = pd.DataFrame({
        "spot_id": df[id_col].astype(str).values,
        "patch_path": df[path_col].astype(str).values,
    })

    for cx, cy in [("x_he_px", "y_he_px"), ("x_morph_px", "y_morph_px"), ("x_um","y_um")]:
        if cx in df.columns and cy in df.columns:
            out[cx] = df[cx].astype(float).values
            out[cy] = df[cy].astype(float).values
            break

    if make_absolute and root_dir is not None:
        out["patch_path"] = [
            os.path.normpath(os.path.join(root_dir, p)) if not os.path.isabs(p) else p
            for p in out["patch_path"].tolist()
        ]

    exists = out["patch_path"].map(os.path.isfile)
    missing = out.loc[~exists, "patch_path"].tolist()
    if missing:
        warnings.warn(f"[load_patch_mapping_csv_xenium] {len(missing)}개 경로가 존재하지 않습니다. 첫 번째: {missing[0]}")
        if drop_missing:
            out = out.loc[exists].reset_index(drop=True)

    return out


# ----------------------------
# Dataset (img, gene) 페어
# ----------------------------
class SpotImageGeneDataset(Dataset):
    """
    - adata: AnnData (X: cells×genes, obs_names는 spot_id와 매칭)
    - map_df: ['spot_id','patch_path']
    - gene_names: None면 adata.var['selected']==True를 사용
    - image_transform: 파운데이션 eval_transform 넣어도 됨(입력은 PIL.Image)
    """
    def __init__(
        self,
        adata: ad.AnnData,
        map_df: pd.DataFrame,
        gene_names: Optional[List[str]] = None,
        image_transform: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.adata = adata
        self.map_df = map_df.reset_index(drop=True)
        self.image_transform = image_transform if image_transform is not None else _DEFAULT_IMG_TF

        obs_index = set(adata.obs_names.astype(str).tolist())
        # NOTE: pd.Series.isin segfaults on >1M-row object series in this env
        # (broken pandas C hashtable ABI); plain-Python membership is safe.
        ids = self.map_df["spot_id"].astype(str).to_numpy()
        keep = np.fromiter((i in obs_index for i in ids), dtype=bool, count=len(ids))
        self.map_df = self.map_df[keep].reset_index(drop=True)

        if gene_names is None:
            if "selected" in adata.var.columns:
                sel = adata.var["selected"].values
                self.gene_names = adata.var_names[sel].tolist()
            else:
                self.gene_names = adata.var_names.tolist()
        else:
            self.gene_names = list(gene_names)

        self._gene_index = adata.var_names.get_indexer(self.gene_names)
        if np.any(self._gene_index < 0):
            missing = [g for g, idx in zip(self.gene_names, self._gene_index) if idx < 0]
            raise KeyError(f"Some genes not found in adata.var_names: {missing[:10]} ...")

        self.gene_dim = len(self.gene_names)

        # Count-scale targets for the ZINB likelihood, kept alongside the
        # log1p(CP10K) encoder input. Absent only for AnnData written before
        # load_h5ad_with_preproc started preserving them.
        self.has_counts = "counts" in getattr(adata, "layers", {})
        self._size_factor = (
            adata.obs["size_factor"].to_numpy().astype(np.float32)
            if "size_factor" in adata.obs.columns else None
        )

    def __len__(self):
        return len(self.map_df)

    def __getitem__(self, idx: int):
        row = self.map_df.iloc[idx]
        sid = str(row["spot_id"])
        img_path = row["patch_path"]

        img = Image.open(img_path).convert("RGB")
        img_t = self.image_transform(img)  # (C,H,W)

        irow = self.adata.obs_names.get_loc(sid)
        x = self.adata.X[irow, self._gene_index]
        if hasattr(x, "toarray"):
            x = x.toarray()
        gene_vec = torch.from_numpy(np.asarray(x).reshape(-1).astype(np.float32))  # (G,)

        return img_t, gene_vec, sid  # (기존 형식 유지)

    # ------------------------------------------------------------------
    # Count-scale targets, aligned to map_df order.
    #
    # Returned as whole-section arrays rather than per-item tensors so the
    # __getitem__ contract (img, gene, sid) stays intact for every existing
    # caller; the training loop indexes these with the batch's node index.
    # ------------------------------------------------------------------
    def counts_matrix(self) -> np.ndarray:
        """(N, G) observed integer counts in map_df order."""
        if not self.has_counts:
            raise RuntimeError(
                "adata has no 'counts' layer. Reload through "
                "load_h5ad_with_preproc(), which preserves raw counts before "
                "normalize_total()."
            )
        rows = self.adata.obs_names.get_indexer(self.map_df["spot_id"].astype(str))
        if np.any(rows < 0):
            raise KeyError("some spot_ids are missing from adata.obs_names")
        x = self.adata.layers["counts"][rows][:, self._gene_index]
        if hasattr(x, "toarray"):
            x = x.toarray()
        return np.asarray(x, dtype=np.float32)

    def size_factors(self) -> np.ndarray:
        """
        (N,) per-cell size factor s_i = library_size / target_sum, in map_df
        order. Computed on the full panel before gene selection, so it measures
        sequencing depth rather than the depth of the retained genes.
        """
        rows = self.adata.obs_names.get_indexer(self.map_df["spot_id"].astype(str))
        if self._size_factor is None:
            raise RuntimeError(
                "adata.obs has no 'size_factor'. Reload through "
                "load_h5ad_with_preproc()."
            )
        return self._size_factor[rows].astype(np.float32)


# ----------------------------
# Dataset (img only, 순서 고정)  → 전역 인덱스 함께 반환
# ----------------------------
class ImgOnlyDataset(Dataset):
    """
    - img↔spatial contrastive에서 쓰기 위한 단일 슬라이드 이미지 시퀀스
    - 반환: (이미지 텐서, 전역 인덱스)
    - 순서는 map_df에 정의된 그 순서 그대로 유지(이 순서로 coords_np, KNN 그래프를 맞춰야 함)
    """
    def __init__(
        self,
        map_df: pd.DataFrame,
        image_transform: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.map_df = map_df.reset_index(drop=True)
        self.image_transform = image_transform if image_transform is not None else _DEFAULT_IMG_TF

    def __len__(self):
        return len(self.map_df)

    def __getitem__(self, idx: int):
        img_path = self.map_df.iloc[idx]["patch_path"]
        img = Image.open(img_path).convert("RGB")
        img_t = self.image_transform(img)
        # 전역 인덱스도 같이 반환 (서브그래프 미니배치에서 직접 사용 가능)
        return img_t, torch.tensor(idx, dtype=torch.long)


# ----------------------------
# 좌표 추출 (map_df 순서와 동일) - 두 모드 지원
# ----------------------------
def extract_coords_np_in_order(
    adata: ad.AnnData,
    map_df: pd.DataFrame,
    coord_key: str = "spatial",
    use_csv_coords: bool = False,
    csv_coord_cols: tuple[str, str] = ("x_he_px", "y_he_px"),
    csv_to_unit: float | None = None,
) -> np.ndarray:
    """
    기본은 adata.obsm[coord_key]에서 좌표를 가져오되,
    use_csv_coords=True이면 map_df의 좌표 컬럼(csv_coord_cols)을 사용.
    csv_to_unit: 픽셀→마이크론/정규화 변환이 필요하면 스칼라 곱 적용(예: 0.5µm/px이면 0.5).
    반환: coords_np (N,2) — map_df 순서
    """
    spot_ids = map_df["spot_id"].astype(str).tolist()

    if use_csv_coords:
        cx, cy = csv_coord_cols
        if cx not in map_df.columns or cy not in map_df.columns:
            raise KeyError(f"CSV 좌표 컬럼 {csv_coord_cols}을 map_df에서 찾을 수 없음.")
        coords = map_df[[cx, cy]].to_numpy(dtype=np.float64)
        if csv_to_unit is not None:
            coords = coords * float(csv_to_unit)
        return coords

    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key!r} not found in adata.obsm")

    obs_to_pos = {str(s): i for i, s in enumerate(adata.obs_names.astype(str).tolist())}
    idxs = [obs_to_pos[s] for s in spot_ids]
    coords = adata.obsm[coord_key][idxs, :]
    return np.asarray(coords, dtype=np.float64)


# ----------------------------
# Dataloader helpers
# ----------------------------
def make_img_gene_dataloader(
    adata: ad.AnnData,
    map_df: pd.DataFrame,
    batch_size: int = 128,
    num_workers: int = 8,
    shuffle: bool = True,
    image_transform: Optional[torch.nn.Module] = None,
) -> Tuple[DataLoader, int]:
    """
    img↔gene contrastive용 로더 생성
    반환: train_loader, gene_dim
    """
    ds = SpotImageGeneDataset(
        adata=adata,
        map_df=map_df,
        gene_names=None,
        image_transform=image_transform,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, ds.gene_dim


def make_img_only_dataloader(
    map_df: pd.DataFrame,
    batch_size: Optional[int] = None,  # None이면 풀배치 (하지만 OOM 위험)
    num_workers: int = 4,
    image_transform: Optional[torch.nn.Module] = None,
    shuffle: bool = False,   # 순서가 중요하므로 False 권장
) -> DataLoader:
    """
    img↔spatial(/joint)용 로더 생성(순서 고정).
    이제 배치는 (imgs, idxs) 형태로 반환됨.
    """
    ds = ImgOnlyDataset(map_df=map_df, image_transform=image_transform)
    if batch_size is None:
        batch_size = len(ds)  # 풀배치 (필요시 OOM 주의)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


# ----------------------------
# ID 정규화 & 정렬 유틸
# ----------------------------
def _norm_id_str(x) -> str:
    """공백 제거, float형 '123.0' → '123' 보정."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def align_map_ids_to_obs(
    adata: ad.AnnData,
    map_df: pd.DataFrame,
    id_col: str = "spot_id",
) -> pd.DataFrame:
    """
    CSV의 id를 adata.obs_names와 최대한 맞춰서 정규화한다.
    - 숫자형이면 int→str로 변환
    - 1-based 오프셋이 의심되면 (id-1)로도 시도해보고, 더 많이 매칭되는 쪽을 채택
    반환: 정렬/삭제 없이, id 문자열만 교정된 DataFrame
    """
    df = map_df.copy()
    ids_raw = df[id_col].values
    try:
        ids_int = pd.to_numeric(ids_raw, errors="raise").astype(np.int64)
        cand_a = [_norm_id_str(x) for x in ids_int]       # 0-based 가정
        cand_b = [_norm_id_str(x - 1) for x in ids_int]   # 1-based → 0-based 교정
        use_int = True
    except Exception:
        cand_a = [_norm_id_str(x) for x in ids_raw]
        cand_b = None
        use_int = False

    obs_set = set([_norm_id_str(x) for x in adata.obs_names.tolist()])
    hit_a = sum(1 for s in cand_a if s in obs_set)
    hit_b = sum(1 for s in (cand_b or []) if s in obs_set)

    fixed = cand_b if (use_int and hit_b > hit_a) else cand_a
    df[id_col] = fixed
    return df


def reorder_and_prune_by_obs(
    adata: ad.AnnData,
    map_df: pd.DataFrame,
    id_col: str = "spot_id",
    coord_key: str = "spatial",
):
    """
    adata와 교집합만 남기고, map_df 순서대로 좌표를 반환.
    반환: map_df_kept, coords_np
    """
    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key!r} not in adata.obsm")

    obs_names = [_norm_id_str(x) for x in adata.obs_names.tolist()]
    obs_to_pos = {s: i for i, s in enumerate(obs_names)}

    spot_ids = [_norm_id_str(x) for x in map_df[id_col].tolist()]

    kept_rows, coord_rows, miss = [], [], 0
    for i, sid in enumerate(spot_ids):
        j = obs_to_pos.get(sid, None)
        if j is None:
            miss += 1
            continue
        kept_rows.append(i)
        coord_rows.append(j)

    if miss:
        print(f"[dataset_utils] pruned {miss} rows not found in adata.obs_names "
              f"({len(spot_ids)-miss} kept)")

    map_df_kept = map_df.iloc[kept_rows].reset_index(drop=True)
    coords_np = np.asarray(adata.obsm[coord_key][coord_rows, :], dtype=np.float64)
    return map_df_kept, coords_np
