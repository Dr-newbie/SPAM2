
"""
run_finetunning_xattn_withceltype_0214_retrieval4.py
----------------------------------------------------
Cross-Attn + ZINB 파인튜닝 스크립트(+ cell type 보조 분류).
retrieval4(확률 비례 cluster memory-bank + weighted sum) 실험 버전.
선택적으로 contrastive 단계에서 학습된 XATTN_pretrained.pt를
fusion 모듈의 cross-attn 가중치로 로드할 수 있음.

예시:
CUDA_VISIBLE_DEVICES=0 python Code_final/code/utils/run_finetunning_xattn_withceltype_0214_retrieval4.py \
  --h5ad /data/project/watercar99/Data/Spatial/xenium_rep1_io/data/merged_with_celltype_symbol_SPAM.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/breast1_patches_256/spot_patch_map_256.csv \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --id_col cell_id --path_col patch_path \
  --enc_name uni_v1 \
  --pt_img_backbone /data/project/banana9903_xenium/ckpts_human_breast_cancer_xattn_10tokens/Image_pretrained.pt \
  --pt_ig /data/project/banana9903_xenium/ckpts_human_breast_cancer_xattn_10tokens/SNN_pretrained.pt \
  --pt_is /data/project/banana9903_xenium/ckpts_human_breast_cancer_xattn_10tokens/Coord_pretrained.pt \
  --pt_xattn /data/project/banana9903_xenium/ckpts_human_breast_cancer_xattn_10tokens/XATTN_pretrained.pt \
  --epochs 50 --batch_size 128 --num_workers 128 \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \
  --graph_type delaunay --k 12 --lr 3e-4 --weight_decay 0.05 \
  --max_grad_norm 1.0 \
  --inside_he_only \
  --save_dir /data/project/banana9903_xenium/ckpts_cross_human_breast_cancer_xattn_delaunay_10tokens_with_celltype_softcluster_retrieval4 \
  --img_tokens 10 \
  --gene_tokens 10 \
  --spot_tokens 10 \
  --celltype_col Cluster \
  --lambda_ce 0.2

"""


import os, sys, time
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # ✅ 학습 로그용 진행바

# 전역 매핑: 문자열 ID -> 노드 인덱스
ID2IDX = None

# ============================================
# 패키지 경로 설정: .../Code_final/code 를 sys.path에 추가
# ============================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../Code_final/code/utils
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))          # .../Code_final/code

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 혹시 다른 곳에서 불러온 'models' 패키지가 있으면 제거해서
# 우리 프로젝트의 models/ 를 새로 로드하게 만듦
sys.modules.pop("models", None)

# ============================================
# 프로젝트 모듈 import
# ============================================
from models.Foundations import inf_encoder_factory
from models.GCN_update import SpatialGCN
from models.gene_encoder import GeneEncoder
from models.fusion import CrossModalFusion
from models.decoders import ZINBDecoder
from utils.loss_utils import zinb_nll
from utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_gene_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)
from models.graph_construction import build_knn_graph
from models.graph_construction_delaunay import build_delaunay_graph


# ============================================
# 프리트레인 가중치 로더 유틸
# ============================================
def _load_sd(path):
    sd = torch.load(path, map_location="cpu")
    # sd가 {'state_dict': ...} 형태일 수도 있음
    if isinstance(sd, dict) and any(k in sd for k in ["state_dict", "model", "module"]):
        for k in ["state_dict", "model", "module"]:
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd if isinstance(sd, dict) else sd.state_dict()


def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_pretrained_for_finetune(foundation, gene_encoder, gcn,
                                 pt_img_backbone=None, pt_ig=None, pt_is=None):
    # 1) 이미지 백본/어댑터 → foundation에 주입
    if pt_img_backbone:
        sd = _load_sd(pt_img_backbone)

        # 케이스 A: ImageFoundationAdapter로 저장된 경우
        #   - 'img_encoder.foundation.' 또는 'foundation.' 프리픽스
        cand = [
            _strip_prefix(sd, "img_encoder.foundation."),
            _strip_prefix(sd, "foundation."),
            sd,  # 혹시 바로 backbone 키일 수도 있음
        ]
        loaded = False
        for s in cand:
            try:
                foundation.load_state_dict(s, strict=False)
                print(f"[PT] loaded into foundation (strict=False) from {pt_img_backbone}")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            print(f"[PT][warn] could not map backbone keys from {pt_img_backbone}")

    # 2) img↔gene ckpt → gene_encoder에 주입
    if pt_ig:
        sd = _load_sd(pt_ig)
        cand = [
            _strip_prefix(sd, "gene_encoder."),
            sd,
        ]
        loaded = False
        for s in cand:
            try:
                gene_encoder.load_state_dict(s, strict=False)
                print(f"[PT] loaded gene_encoder from {pt_ig} (strict=False)")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            print(f"[PT][warn] gene_encoder keys not found in {pt_ig}")

    # 3) img↔spatial ckpt → gcn에 주입
    if pt_is:
        sd = _load_sd(pt_is)
        cand = [
            _strip_prefix(sd, "gcn."),
            sd,
        ]
        loaded = False
        for s in cand:
            try:
                gcn.load_state_dict(s, strict=False)
                print(f"[PT] loaded GCN from {pt_is} (strict=False)")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            print(f"[PT][warn] gcn keys not found in {pt_is}")


def load_xattn_into_fusion(fusion: CrossModalFusion, xattn_path: str | None):
    if not xattn_path:
        return
    sd = _load_sd(xattn_path)

    def _load_block(dst_block: nn.Module, src_prefix: str):
        sub = {k[len(src_prefix) + 1:]: v for k, v in sd.items()
               if k.startswith(src_prefix + ".")}
        if not sub:
            print(f"[XATTN][warn] no keys found for {src_prefix} in {xattn_path}")
            return
        ret = dst_block.load_state_dict(sub, strict=False)
        missing = len(ret.missing_keys)
        unexpected = len(ret.unexpected_keys)
        if missing or unexpected:
            print(f"[XATTN][warn] {src_prefix} -> fusion "
                  f"(missing={missing}, unexpected={unexpected})")
        else:
            print(f"[XATTN] loaded {src_prefix} into fusion")

    # XATTN_pretrained: i2g, g2i, i2c, c2i
    # fusion: xa_ig (img<-gene), xa_is (img<-spot/coord)
    _load_block(fusion.xa_ig, "i2g")
    _load_block(fusion.xa_is, "i2c")


def _pick_celltype_col(adata, celltype_col: str | None):
    cols = list(adata.obs.columns)
    cols_lower = {str(c).lower(): c for c in cols}
    if celltype_col is not None:
        key = str(celltype_col).lower()
        if key in cols_lower:
            return cols_lower[key]
        raise KeyError(f"--celltype_col={celltype_col!r} not found in adata.obs. Available: {cols}")

    candidates = [
        "cell_type", "celltype", "celltypes", "cell_type_pred", "annotation",
        "annot", "cluster", "clusters", "leiden", "seurat_clusters", 'Cluster'
    ]
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    raise KeyError(f"Cell type column not found in adata.obs. Available: {cols}")


def _build_celltype_targets(
    adata,
    map_df_aligned: pd.DataFrame,
    celltype_col: str | None,
):
    ct_col = _pick_celltype_col(adata, celltype_col)
    raw = adata.obs.loc[map_df_aligned["spot_id"], ct_col]
    raw_str = raw.astype(str)
    valid_mask = ~raw.isna()
    valid_vals = raw_str.loc[valid_mask]
    names = sorted(pd.unique(valid_vals).tolist())
    name_to_idx = {n: i for i, n in enumerate(names)}

    y = np.full(map_df_aligned.shape[0], -1, dtype=np.int64)
    if len(names) > 0:
        for i, sid in enumerate(map_df_aligned["spot_id"].tolist()):
            v = raw.loc[sid]
            if pd.isna(v):
                continue
            y[i] = name_to_idx[str(v)]

    n_valid = int((y >= 0).sum())
    n_missing = int((y < 0).sum())
    print(f"[CellType] col={ct_col}, classes={len(names)}, labeled={n_valid}, missing={n_missing}")
    return y, names, ct_col


# ============================================
# 모델 래퍼
# ============================================
class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn,
                 dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1,
                 merge="gated-sum",
                 img_tokens=1, spot_tokens=1, gene_tokens=1,
                 decoder_hidden=1024, gene_dim=1000,
                 num_celltypes: int = 0, cls_head_hidden: int = 0, cls_dropout: float = 0.0):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.gcn = gcn   # SpatialGCN: (X_all, edge_index, edge_weight) -> (N, dim_spot)

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

    def forward(self, batch, spot_feat_in, edge_index, edge_weight=None, return_attn: bool = False):
        """
        batch: dict
          - image: (B, C, H, W)
          - expr : (B, G)  # ⬅ log1p(count) 라고 가정
          - idx  : (B,) long  — 그래프 노드 인덱스(=spot index in current order)
        spot_feat_in: (N, F_in)  — 모든 노드에 대한 입력 feature (ex. coords)
        """
        x_img = batch["image"]
        x_gene_log = batch["expr"].float()
        node_idx = batch["idx"].long()   # (B,)

        # 1) enc
        zi = self.img_encoder(x_img)            # (B, Di)
        zg = self.gene_encoder(x_gene_log)      # (B, Dg)  # log-space 표현

        # 1') 전체 노드 feature에 대해 GCN 수행 → (N, Ds)
        # Keep GCN in FP32 to avoid bf16 GEMM instability on some CUDA stacks.
        spot_feat_in = spot_feat_in.float()
        edge_weight_fp32 = edge_weight.float() if edge_weight is not None else None
        if x_img.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                zs_all_nodes = self.gcn(spot_feat_in, edge_index, edge_weight_fp32)  # (N, Ds)
        else:
            zs_all_nodes = self.gcn(spot_feat_in, edge_index, edge_weight_fp32)  # (N, Ds)
        zs = zs_all_nodes[node_idx]                                     # (B, Ds)

        # 2) same dim
        zi = self.proj_img(zi)
        zg = self.proj_gene(zg)
        zs = self.proj_spot(zs)

        # 3) cross-attn fusion
        if return_attn:
            z_fused, z_is, z_ig, attn = self.fusion(zi, zs, zg, return_attn=True)
        else:
            z_fused, z_is, z_ig = self.fusion(zi, zs, zg, return_attn=False)
            attn = None

        # 4) decode (ZINB params, count-space mean mu)
        mu, theta, pi = self.decoder(z_fused)
        cls_logits = self.cls_head(z_fused) if self.cls_head is not None else None
        return mu, theta, pi, {
            "z_fused": z_fused,
            "z_is": z_is,
            "z_ig": z_ig,
            "attn": attn,
            "cls_logits": cls_logits,
        }


# ============================================
# 배치 정규화 헬퍼: dict / tuple 둘 다 지원 + 문자열 ID 처리
# ============================================
def _to_device_batch(batch, device):
    """
    DataLoader batch를 dict 형태로 통일하고 device로 올리는 함수.
    - dict 인 경우: 그대로 key별로 .to(device)
    - (image, expr, idx) tuple/list 인 경우:
        * image, expr: tensor 또는 array -> tensor
        * idx: 정수 / 리스트 / 문자열 ID 리스트 모두 처리
    """
    global ID2IDX

    def _to_tensor(x, device, dtype=None):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.as_tensor(x, dtype=dtype, device=device)

    # 1) 이미 dict 인 경우
    if isinstance(batch, dict):
        return {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    # 2) (image, expr, idx) 형태의 tuple / list 인 경우
    if isinstance(batch, (list, tuple)):
        if len(batch) != 3:
            raise TypeError(f"Unexpected batch length {len(batch)}; expected 3.")

        x_img, x_gene, idx = batch

        # image / expr는 보통 이미 tensor일 것, 그래도 안전하게 처리
        x_img = _to_tensor(x_img, device)
        x_gene = _to_tensor(x_gene, device)

        # ---- idx 처리: 문자열인지, 숫자인지에 따라 분기 ----
        if not isinstance(idx, (list, tuple, np.ndarray)):
            idx_list = [idx]
        else:
            idx_list = list(idx)

        if len(idx_list) > 0 and isinstance(idx_list[0], str):
            if ID2IDX is None:
                raise RuntimeError("ID2IDX mapping is not initialized but idx is string-based.")
            try:
                idx_mapped = [ID2IDX[str(s)] for s in idx_list]
            except KeyError as e:
                raise KeyError(f"idx value {e} not found in ID2IDX mapping") from e
        else:
            idx_mapped = idx_list

        idx_tensor = _to_tensor(idx_mapped, device, dtype=torch.long)

        return {
            "image": x_img,
            "expr": x_gene,
            "idx": idx_tensor,
        }

    # 3) 그 외 타입은 에러
    raise TypeError(f"Unsupported batch type: {type(batch)}")


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


def _adata_x_to_numpy_float32(adata) -> np.ndarray:
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"adata.X must be 2D, got shape={x.shape}")
    return x


def _build_class_index_map(celltype_y_np: np.ndarray | None) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if celltype_y_np is None:
        return out
    y = np.asarray(celltype_y_np, dtype=np.int64)
    for c in np.unique(y):
        if int(c) < 0:
            continue
        out[int(c)] = np.where(y == int(c))[0].astype(np.int64)
    return out


def _current_retrieval4_tau(model: nn.Module, tau_default: float = 0.07) -> torch.Tensor:
    dev = next(model.parameters()).device
    if hasattr(model, "retrieval4_log_tau"):
        t = torch.exp(getattr(model, "retrieval4_log_tau"))
        return torch.clamp(t, min=1e-3, max=50.0)
    if hasattr(model, "retrieval4_tau_const"):
        t = getattr(model, "retrieval4_tau_const")
        return torch.clamp(t.to(device=dev), min=1e-3, max=50.0)
    return torch.tensor(float(max(tau_default, 1e-3)), device=dev, dtype=torch.float32)


@torch.no_grad()
def _build_retrieval4_banks(
    model: nn.Module,
    loader,
    spot_feat_in: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    expr_bank_np: np.ndarray,
    num_celltypes: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    n_nodes: int,
    pred_min_conf: float = 0.0,
) -> dict:
    was_training = model.training
    model.eval()
    dim = int(model.proj_img.out_features)
    key_bank = torch.zeros((n_nodes, dim), dtype=torch.float32, device=device)
    pred_cls_bank = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    pred_prob_bank = torch.zeros((n_nodes, int(num_celltypes)), dtype=torch.float32, device=device)
    seen = torch.zeros((n_nodes,), dtype=torch.bool, device=device)
    pbar = tqdm(loader, desc="[Retrieval4] build key/prob banks", leave=False)
    for batch in pbar:
        batch = _to_device_batch(batch, device)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=(device.type == "cuda" and amp_dtype != torch.float32),
        ):
            q = model.proj_img(model.img_encoder(batch["image"]))
            _, _, _, aux = model(batch, spot_feat_in, edge_index, edge_weight, return_attn=False)
        q = F.normalize(q.float(), dim=1)
        cls_logits = aux.get("cls_logits", None)
        if cls_logits is None:
            raise RuntimeError("[Retrieval4] cls_head is required but cls_logits is None.")
        probs = torch.softmax(cls_logits.float(), dim=1)
        conf, pred_cls = torch.max(probs, dim=1)
        idx = batch["idx"].long()
        key_bank[idx] = q
        pred_cls_bank[idx] = pred_cls.long()
        pred_prob_bank[idx] = probs.float()
        seen[idx] = True
    miss = int((~seen).sum().item())
    if miss > 0:
        print(f"[Retrieval4][warn] {miss} nodes were not seen while building key/prob banks.")

    pred_cls_np = pred_cls_bank.detach().cpu().numpy()
    pred_prob_np = pred_prob_bank.detach().cpu().numpy()
    pred_conf_np = pred_prob_np.max(axis=1) if pred_prob_np.size > 0 else np.zeros((n_nodes,), dtype=np.float32)
    class_index_map: dict[int, np.ndarray] = {}
    for c in range(int(num_celltypes)):
        m = np.where((pred_cls_np == c) & (pred_conf_np >= float(pred_min_conf)))[0]
        if m.size > 0:
            class_index_map[int(c)] = m.astype(np.int64)

    if was_training:
        model.train()
    return {
        "keys_bank": key_bank,
        "pred_cls_bank": pred_cls_bank,
        "pred_prob_bank": pred_prob_bank,
        "class_index_map": class_index_map,
    }


def _allocate_cluster_counts(
    prob_vec: np.ndarray,
    total: int,
) -> np.ndarray:
    total = int(max(total, 0))
    if total == 0:
        return np.zeros_like(prob_vec, dtype=np.int64)
    p = np.asarray(prob_vec, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0.0:
        p = np.ones_like(p, dtype=np.float64) / max(int(p.size), 1)
    else:
        p = p / s

    raw = p * float(total)
    cnt = np.floor(raw).astype(np.int64)
    rem = int(total - int(cnt.sum()))
    if rem > 0:
        frac = raw - cnt.astype(np.float64)
        order = np.argsort(-frac)
        cnt[order[:rem]] += 1
    return cnt


def _retrieval4_memory_bank_expr(
    query_key: torch.Tensor,
    query_probs: torch.Tensor,
    x_log_true: torch.Tensor,
    keys_bank: torch.Tensor,
    expr_bank_np: np.ndarray,
    class_index_map: dict[int, np.ndarray],
    bank_size: int = 5000,
    tau: torch.Tensor | None = None,
    prob_topm: int = 0,
    fallback: str = "global",
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Query별 memory bank(최대 bank_size)를 구성한 뒤 weighted sum으로 retrieval expr 생성.
    1) query의 cluster 확률로 클래스별 quota를 배정
    2) 각 클래스에서 cosine 상위 샘플을 quota만큼 선택
    3) 선택된 전체 bank와 query를 다시 cosine 비교 후 softmax weighted sum
    """
    device = query_key.device
    q = F.normalize(query_key.float(), dim=1)
    bsz = int(q.size(0))
    n_nodes = int(keys_bank.size(0))
    target_bank = int(max(1, min(int(bank_size), n_nodes)))
    x_retr = x_log_true.clone()
    used = torch.zeros((bsz,), dtype=torch.bool, device=device)
    top1_vals = []

    if tau is None:
        tau = torch.tensor(0.07, dtype=torch.float32, device=device)
    tau = torch.clamp(tau, min=1e-3, max=50.0)

    probs = query_probs.float().to(device=device)
    n_cls = int(probs.size(1))
    prob_topm = int(max(0, prob_topm))
    if prob_topm > 0 and prob_topm < n_cls:
        vals_m, idx_m = torch.topk(probs, k=prob_topm, dim=1, largest=True, sorted=False)
        probs_topm = torch.zeros_like(probs)
        probs_topm.scatter_(1, idx_m, vals_m)
        probs = probs_topm
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    all_idx_np = np.arange(n_nodes, dtype=np.int64)

    for bi in range(bsz):
        qv = q[bi:bi + 1]  # (1, D)
        prob_vec = probs[bi].detach().cpu().numpy().astype(np.float64)
        quota = _allocate_cluster_counts(prob_vec, target_bank)

        selected_parts: list[np.ndarray] = []
        for ci, take_n in enumerate(quota.tolist()):
            if take_n <= 0:
                continue
            cand_np = class_index_map.get(int(ci), None)
            if cand_np is None or cand_np.size == 0:
                continue
            cand_t = torch.as_tensor(cand_np, device=device, dtype=torch.long)
            cand_keys = keys_bank.index_select(0, cand_t)
            sim = (qv @ cand_keys.t()).squeeze(0)
            kk = min(int(take_n), int(sim.numel()))
            if kk <= 0:
                continue
            _, loc = torch.topk(sim, k=kk, dim=0, largest=True, sorted=True)
            selected_parts.append(cand_np[loc.detach().cpu().numpy()])

        if selected_parts:
            selected_np = np.concatenate(selected_parts, axis=0).astype(np.int64)
            # 중복 제거(클래스별 집합 merge)
            if selected_np.size > 1:
                selected_np = np.unique(selected_np)
        else:
            selected_np = np.empty((0,), dtype=np.int64)

        # 부족분은 global 유사도 상위로 채워서 bank size 맞춤
        if selected_np.size < target_bank:
            sim_all = (qv @ keys_bank.t()).squeeze(0)
            if selected_np.size > 0:
                mask = torch.ones((n_nodes,), dtype=torch.bool, device=device)
                mask[torch.as_tensor(selected_np, device=device, dtype=torch.long)] = False
                sim_fill = sim_all.clone()
                sim_fill[~mask] = -1e9
            else:
                sim_fill = sim_all
            need = min(target_bank - int(selected_np.size), int(sim_fill.numel()))
            if need > 0:
                _, fill_idx = torch.topk(sim_fill, k=need, dim=0, largest=True, sorted=True)
                fill_np = fill_idx.detach().cpu().numpy().astype(np.int64)
                if selected_np.size > 0:
                    selected_np = np.concatenate([selected_np, fill_np], axis=0)
                else:
                    selected_np = fill_np

        if selected_np.size == 0:
            if fallback == "global":
                # cluster 정보가 비어도 global top-bank는 사용
                sim_all = (qv @ keys_bank.t()).squeeze(0)
                kk = min(target_bank, int(sim_all.numel()))
                _, loc = torch.topk(sim_all, k=kk, dim=0, largest=True, sorted=True)
                selected_np = all_idx_np[loc.detach().cpu().numpy()]
            else:
                # self_expr: retrieval 미사용
                continue

        sel_t = torch.as_tensor(selected_np, device=device, dtype=torch.long)
        cand_keys = keys_bank.index_select(0, sel_t)
        sim_sel = (qv @ cand_keys.t()).squeeze(0)
        if sim_sel.numel() > target_bank:
            vals_sel, loc_sel = torch.topk(sim_sel, k=target_bank, dim=0, largest=True, sorted=True)
            sel_t = sel_t.index_select(0, loc_sel)
            sim_sel = vals_sel

        if sim_sel.numel() == 0:
            continue
        w = torch.softmax(sim_sel / tau, dim=0)
        top1_vals.append(sim_sel.max().detach())
        expr_sel = torch.from_numpy(expr_bank_np[sel_t.detach().cpu().numpy()]).to(
            device=device, dtype=x_log_true.dtype
        )
        x_retr[bi] = (w.unsqueeze(-1) * expr_sel).sum(dim=0)
        used[bi] = True

    if top1_vals:
        top1_mean = float(torch.cat(top1_vals, dim=0).mean().item())
    else:
        top1_mean = float("nan")
    return x_retr, used, top1_mean


# ============================================
# 학습 / 평가 루프
# ============================================
def train_epoch(model, loader, spot_feat_in, edge_index, edge_weight,
                optimizer, amp_dtype=torch.float32,
                device=torch.device("cpu"),
                epoch: int = 1, total_epochs: int = 1,
                lambda_zinb: float = 1.0,
                lambda_mse: float = 0.1,
                lambda_pcc: float = 0.0,
                lambda_ce: float = 0.0,
                ce_label_smoothing: float = 0.0,
                ce_warmup_epochs: int = 0,
                celltype_labels_all: torch.Tensor | None = None,
                ce_class_weights: torch.Tensor | None = None,
                max_grad_norm: float = 1.0,
                retrieval4_cfg: dict | None = None):
    """
    한 epoch 학습.
    tqdm으로 배치별 loss를 progress bar에 띄워서 모니터링.
    - batch['expr']는 log1p(count)라고 가정
    - ZINB는 expm1(logX) 기반 raw-like count로 계산
    - MSE/PCC는 log-space에서 계산
    """
    model.train()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and amp_dtype != torch.float32)
    )
    loss_sum, n = 0.0, 0
    cls_correct, cls_count = 0.0, 0
    retr_used_sum, retr_top1_sum, retr_steps = 0.0, 0.0, 0
    retr_tau_last = float("nan")

    eps = 1e-8

    # ✅ tqdm으로 DataLoader 감싸기
    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}/{total_epochs}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch = _to_device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=(device.type == "cuda" and amp_dtype != torch.float32),
        ):
            # x_log: log1p(count) 값 (load_h5ad_with_preproc에서 생성된 adata.X)
            x_log = batch["expr"].float()
            batch_in = batch
            loss_retr = x_log.new_zeros(())
            retr_used_ratio = 0.0

            # retrieval4: query 확률 비례로 클래스별 bank를 구성한 뒤 weighted sum retrieval
            if retrieval4_cfg is not None and bool(retrieval4_cfg.get("enabled", False)):
                keys_bank = retrieval4_cfg.get("keys_bank", None)
                pred_prob_bank = retrieval4_cfg.get("pred_prob_bank", None)
                expr_bank_np = retrieval4_cfg.get("expr_bank_np", None)
                class_index_map = retrieval4_cfg.get("class_index_map", {})
                if keys_bank is not None and expr_bank_np is not None and pred_prob_bank is not None:
                    q = model.proj_img(model.img_encoder(batch["image"]))
                    tau = _current_retrieval4_tau(model, float(retrieval4_cfg.get("tau", 0.07)))
                    retr_tau_last = float(tau.detach().item())
                    x_retr, retr_used, retr_top1 = _retrieval4_memory_bank_expr(
                        query_key=q,
                        query_probs=pred_prob_bank[batch["idx"].long()],
                        x_log_true=x_log,
                        keys_bank=keys_bank,
                        expr_bank_np=expr_bank_np,
                        class_index_map=class_index_map,
                        bank_size=int(retrieval4_cfg.get("bank_size", 5000)),
                        tau=tau,
                        prob_topm=int(retrieval4_cfg.get("prob_topm", 0)),
                        fallback=str(retrieval4_cfg.get("fallback", "global")),
                    )
                    retr_top1_sum += 0.0 if np.isnan(retr_top1) else float(retr_top1)
                    retr_steps += 0 if np.isnan(retr_top1) else 1
                    retr_used_ratio = float(retr_used.float().mean().item())
                    retr_used_sum += retr_used_ratio

                    mix_base = float(retrieval4_cfg.get("mix", 0.0))
                    mix_warm = int(retrieval4_cfg.get("mix_warmup_epochs", 0))
                    if mix_warm > 0:
                        mix = mix_base * min(1.0, float(epoch) / float(max(mix_warm, 1)))
                    else:
                        mix = mix_base
                    mix = float(np.clip(mix, 0.0, 1.0))
                    batch_in = dict(batch)
                    batch_in["expr"] = (1.0 - mix) * x_log + mix * x_retr

                    if float(retrieval4_cfg.get("lambda_retr", 0.0)) > 0.0:
                        loss_retr = F.mse_loss(x_retr, x_log)

            mu, theta, pi, aux = model(batch_in, spot_feat_in, edge_index, edge_weight)

            # ✅ 입력 값 검증
            if torch.isnan(mu).any() or torch.isinf(mu).any():
                print(f"[WARN] Step {step}: mu contains NaN/Inf, skipping this batch")
                continue
            if torch.isnan(theta).any() or torch.isinf(theta).any():
                print(f"[WARN] Step {step}: theta contains NaN/Inf, skipping this batch")
                continue

            # ZINB는 count-space 가정 → expm1으로 raw-like count 복원
            x_raw = torch.expm1(torch.clamp(x_log, min=0.0))

            # 1) ZINB loss (raw-like count 기준)
            loss_zinb = zinb_nll(x_raw, mu, theta, pi, reduction="mean")

            # ✅ Loss 값 검증: NaN/Inf 감지 시 skip
            if torch.isnan(loss_zinb) or torch.isinf(loss_zinb):
                print(f"[WARN] Step {step}: ZINB loss is NaN/Inf ({loss_zinb.item():.4e}), skipping")
                continue

            # 2) log1p 기반 MSE (direct reconstruction은 log-space)
            mu_clamped = torch.clamp(mu, min=eps)
            mu_log = torch.log1p(mu_clamped)
            loss_mse = F.mse_loss(mu_log, x_log)

            # ✅ MSE 검증
            if torch.isnan(loss_mse) or torch.isinf(loss_mse):
                print(f"[WARN] Step {step}: MSE loss is NaN/Inf ({loss_mse.item():.4e}), skipping")
                continue

            # 3) (옵션) PCC loss – log-space에서 per gene PCC의 평균을 최대화(= 1 - mean(PCC) 최소화)
            if lambda_pcc > 0.0:
                Y = x_log
                Yh = mu_log
                Yc = Y - Y.mean(dim=0, keepdim=True)
                Yhc = Yh - Yh.mean(dim=0, keepdim=True)
                num = (Yc * Yhc).sum(dim=0)
                den = (Yc.pow(2).sum(dim=0) * Yhc.pow(2).sum(dim=0)).sqrt() + 1e-8
                pcc = num / den
                pcc_loss = 1.0 - pcc.mean()  # PCC ↑ 를 원하므로 1 - mean(PCC) 최소화
            else:
                # 그래프 끊기지 않게 텐서 유지
                pcc_loss = 0.0 * loss_zinb

            use_ce = (
                lambda_ce > 0.0
                and model.cls_head is not None
                and celltype_labels_all is not None
                and celltype_labels_all.numel() > 0
            )
            if use_ce:
                y_ct = celltype_labels_all[batch["idx"].long()]
                valid = y_ct >= 0
                cls_logits = aux.get("cls_logits", None)
                ce_scale = min(1.0, float(epoch) / max(int(ce_warmup_epochs), 1)) if ce_warmup_epochs > 0 else 1.0
                if cls_logits is None or valid.sum().item() == 0:
                    ce_loss = 0.0 * loss_zinb
                    ce_acc = 0.0
                    ce_seen = 0
                else:
                    ce_loss = F.cross_entropy(
                        cls_logits[valid],
                        y_ct[valid],
                        weight=ce_class_weights,
                        label_smoothing=max(float(ce_label_smoothing), 0.0),
                    )
                    pred_ct = torch.argmax(cls_logits[valid], dim=1)
                    ce_seen = int(valid.sum().item())
                    ce_acc = float((pred_ct == y_ct[valid]).float().mean().item()) if ce_seen > 0 else 0.0
                    ce_loss = ce_loss * ce_scale
            else:
                ce_loss = 0.0 * loss_zinb
                ce_acc = 0.0
                ce_seen = 0

            # 최종 혼합 loss
            loss = (
                lambda_zinb * loss_zinb +
                lambda_mse * loss_mse +
                lambda_pcc * pcc_loss +
                lambda_ce * ce_loss +
                float(retrieval4_cfg.get("lambda_retr", 0.0) if retrieval4_cfg is not None else 0.0) * loss_retr
            )

            # ✅ 최종 loss 검증
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Step {step}: Total loss is NaN/Inf ({loss.item():.4e}), skipping")
                continue

        # ✅ Gradient clipping
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        bs = x_log.size(0)
        loss_sum += loss.item() * bs
        n += bs
        if ce_seen > 0:
            cls_correct += ce_acc * ce_seen
            cls_count += ce_seen

        # ✅ 진행바에 각 loss 컴포넌트 표시
        pbar.set_postfix({
            "Lz": f"{loss_zinb.item():.4f}",
            "Lm": f"{loss_mse.item():.4f}",
            "Lp": f"{(pcc_loss.item() if lambda_pcc > 0.0 else 0.0):.4f}",
            "Lc": f"{(ce_loss.item() if lambda_ce > 0.0 else 0.0):.4f}",
            "Lr1": f"{(loss_retr.item() if retrieval4_cfg is not None else 0.0):.4f}",
            "Ruse": f"{retr_used_ratio:.2f}",
            "Cacc": f"{(ce_acc if ce_seen > 0 else 0.0):.3f}",
        })

    return {
        "loss": loss_sum / max(n, 1),
        "ct_acc": float(cls_correct / max(cls_count, 1)),
        "ct_n": int(cls_count),
        "retr_used": float(retr_used_sum / max(len(loader), 1)),
        "retr_top1": float(retr_top1_sum / max(retr_steps, 1)) if retr_steps > 0 else float("nan"),
        "retr_tau": float(retr_tau_last),
    }


@torch.no_grad()
def eval_epoch(model, loader, spot_feat_in, edge_index, edge_weight,
               amp_dtype=torch.float32,
               device=torch.device("cpu"),
               epoch: int = 1, total_epochs: int = 1,
               save_attn: bool = False,
               attn_out_dir: str | None = None,
               attn_max_batches: int = 0,
               attn_reduce_heads: bool = False,
               celltype_labels_all: torch.Tensor | None = None,
               retrieval4_cfg: dict | None = None):
    """
    한 epoch 평가.
    - batch['expr']는 log1p(count)
    - mu는 count-space mean → log1p(mu) vs logX 로 MSE/MAE/PCC 계산
    """
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    cls_correct, cls_count = 0.0, 0
    preds, gts = [], []

    eps = 1e-8

    pbar = tqdm(loader, desc=f"[Eval ] Epoch {epoch}/{total_epochs}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch = _to_device_batch(batch, device)
        dump_attn_this_batch = bool(save_attn and attn_out_dir and attn_max_batches > 0 and step <= attn_max_batches)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=(device.type == "cuda" and amp_dtype != torch.float32),
        ):
            x_log = batch["expr"].float()
            batch_in = batch
            if retrieval4_cfg is not None and bool(retrieval4_cfg.get("enabled", False)):
                keys_bank = retrieval4_cfg.get("keys_bank", None)
                pred_prob_bank = retrieval4_cfg.get("pred_prob_bank", None)
                expr_bank_np = retrieval4_cfg.get("expr_bank_np", None)
                class_index_map = retrieval4_cfg.get("class_index_map", {})
                if keys_bank is not None and expr_bank_np is not None and pred_prob_bank is not None:
                    q = model.proj_img(model.img_encoder(batch["image"]))
                    tau = _current_retrieval4_tau(model, float(retrieval4_cfg.get("tau", 0.07)))
                    x_retr, _, _ = _retrieval4_memory_bank_expr(
                        query_key=q,
                        query_probs=pred_prob_bank[batch["idx"].long()],
                        x_log_true=x_log,
                        keys_bank=keys_bank,
                        expr_bank_np=expr_bank_np,
                        class_index_map=class_index_map,
                        bank_size=int(retrieval4_cfg.get("bank_size", 5000)),
                        tau=tau,
                        prob_topm=int(retrieval4_cfg.get("prob_topm", 0)),
                        fallback=str(retrieval4_cfg.get("fallback", "global")),
                    )
                    mix = float(np.clip(float(retrieval4_cfg.get("eval_mix", 1.0)), 0.0, 1.0))
                    batch_in = dict(batch)
                    batch_in["expr"] = (1.0 - mix) * x_log + mix * x_retr

            mu, theta, pi, aux = model(
                batch_in, spot_feat_in, edge_index, edge_weight,
                return_attn=dump_attn_this_batch,
            )

            # mu는 count-space mean → log1p로 변환
            mu_clamped = torch.clamp(mu, min=0.0)
            yhat_log = torch.log1p(mu_clamped + eps)

        preds.append(yhat_log.detach().cpu())
        gts.append(x_log.detach().cpu())
        mse_sum += F.mse_loss(yhat_log, x_log, reduction="sum").item()
        mae_sum += F.l1_loss(yhat_log, x_log, reduction="sum").item()
        n += x_log.numel()
        if celltype_labels_all is not None and celltype_labels_all.numel() > 0:
            y_ct = celltype_labels_all[batch["idx"].long()]
            valid = y_ct >= 0
            cls_logits = aux.get("cls_logits", None)
            if cls_logits is not None and valid.sum().item() > 0:
                pred_ct = torch.argmax(cls_logits[valid], dim=1)
                cls_correct += float((pred_ct == y_ct[valid]).sum().item())
                cls_count += int(valid.sum().item())
        if dump_attn_this_batch:
            out_npz = os.path.join(attn_out_dir, f"ep{epoch:03d}_eval_step{step:04d}.npz")
            _save_attn_npz(
                out_npz,
                aux.get("attn", None),
                reduce_heads=attn_reduce_heads,
                node_idx=batch["idx"].detach().cpu().numpy(),
            )

    P = torch.cat(preds, 0).numpy()  # log-pred
    Y = torch.cat(gts, 0).numpy()    # log-GT
    Yc, Pc = Y - Y.mean(0, keepdims=True), P - P.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0)) + 1e-8
    pcc = np.where(den > 0, num / den, np.nan)

    return {
        "mse": mse_sum / max(n, 1),
        "mae": mae_sum / max(n, 1),
        "pcc_mean": float(np.nanmean(pcc)),
        "pcc_median": float(np.nanmedian(pcc)),
        "ct_acc": float(cls_correct / max(cls_count, 1)),
        "ct_n": int(cls_count),
    }, P, Y


# ============================================
# main
# ============================================
def main():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument("--pt_img_backbone", default=None,
                   help="contrastive에서 학습된 이미지 백본/어댑터 ckpt")
    p.add_argument("--pt_ig", default=None,
                   help="img↔gene contrastive(ig_model) ckpt 경로")
    p.add_argument("--pt_is", default=None,
                   help="img↔spatial contrastive(is_model) ckpt 경로")
    p.add_argument("--pt_xattn", default=None,
                   help="contrastive xattn ckpt (XATTN_pretrained.pt)")

    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--id_col", default="cell_id")
    p.add_argument("--path_col", default="patch_path")
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1", "virchow", "virchow2",
                            "gigapath", "hoptimus0",
                            "plip", "phikon", "conch_v1"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--proj_dim_gene", type=int, default=256)
    p.add_argument("--proj_dim_spot", type=int, default=256)
    p.add_argument("--fuse_dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--img_tokens", type=int, default=1,
                   help="number of image query tokens in fusion cross-attention")
    p.add_argument("--spot_tokens", type=int, default=1,
                   help="number of spatial tokens in fusion cross-attention")
    p.add_argument("--gene_tokens", type=int, default=1,
                   help="number of gene tokens in fusion cross-attention")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--merge", default="gated-sum",
                   choices=["gated-sum", "concat-proj"])
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--graph_type", default="knn", choices=["knn", "delaunay"],
                   help="Spatial graph type for GCN input.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_dir", default="./ckpts_cross")
    p.add_argument("--save_attn", action="store_true",
                   help="save attention weights from eval loop")
    p.add_argument("--attn_out_dir", default=None,
                   help="directory to save attention npz files (default: <save_dir>/attn)")
    p.add_argument("--attn_max_batches", type=int, default=10,
                   help="max eval batches per epoch to dump attention")
    p.add_argument("--attn_every", type=int, default=1,
                   help="save attention every N epochs")
    p.add_argument("--attn_reduce_heads", action="store_true",
                   help="store mean over heads instead of per-head attention")
    p.add_argument("--inside_he_only", action="store_true",
                   help="CSV에서 inside_he=True인 패치만 사용")

    # 🔧 ZINB + MSE + PCC 가중치
    p.add_argument("--lambda_zinb", type=float, default=1.0,
                   help="ZINB NLL loss 가중치")
    p.add_argument("--lambda_mse", type=float, default=0.1,
                   help="log-space MSE 재구성 loss 가중치")
    p.add_argument("--lambda_pcc", type=float, default=0.0,
                   help="PCC 기반 loss 가중치 (1 - mean(PCC))")
    p.add_argument("--lambda_ce", type=float, default=0.2,
                   help="cell type 분류 CE loss 가중치")
    p.add_argument("--celltype_col", default=None,
                   help="adata.obs cell type 컬럼명 (기본: 자동 탐색)")
    p.add_argument("--ce_label_smoothing", type=float, default=0.0,
                   help="cell type CE label smoothing")
    p.add_argument("--ce_warmup_epochs", type=int, default=0,
                   help="cell type CE loss warmup epoch 수 (0=off)")
    p.add_argument("--ce_head_hidden", type=int, default=0,
                   help="cell type 분류 head hidden dim (0이면 linear)")
    p.add_argument("--ce_head_dropout", type=float, default=0.0,
                   help="cell type 분류 head dropout")
    p.add_argument("--ce_class_weight", default="none", choices=["none", "balanced"],
                   help="cell type CE class weight 사용 여부")
    
    # ✅ 안정화 파라미터
    p.add_argument("--warmup_epochs", type=int, default=2,
                   help="학습 초기 warmup epoch 수 (lr을 천천히 증가)")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Gradient clipping max norm (안정성)")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA (Exponential Moving Average) decay rate for model parameters. Set to 0 to disable.")
    p.add_argument("--retrieval4", action="store_true",
                   help="enable retrieval4: probability-guided memory-bank retrieval")
    p.add_argument("--retrieval4_bank_size", type=int, default=5000,
                   help="total memory bank size per query (allocated by query class probabilities)")
    p.add_argument("--retrieval4_tau", type=float, default=0.07,
                   help="softmax temperature for retrieval4 weighting")
    p.add_argument("--retrieval4_learnable_tau", action="store_true",
                   help="make retrieval4 temperature learnable")
    p.add_argument("--retrieval4_mix", type=float, default=0.5,
                   help="mix ratio for retrieved expr into model input expr during train")
    p.add_argument("--retrieval4_eval_mix", type=float, default=1.0,
                   help="mix ratio for retrieved expr during eval")
    p.add_argument("--retrieval4_mix_warmup_epochs", type=int, default=5,
                   help="warmup epochs to ramp retrieval4 mix ratio")
    p.add_argument("--retrieval4_lambda", type=float, default=0.1,
                   help="auxiliary MSE weight between retrieved expr and GT log expr")
    p.add_argument("--retrieval4_refresh_epochs", type=int, default=1,
                   help="refresh key bank every N epochs")
    p.add_argument("--retrieval4_fallback", default="global", choices=["self_expr", "global"],
                   help="fallback strategy when no valid celltype retrieval candidate")
    p.add_argument("--retrieval4_num_celltypes", type=int, default=0,
                   help="num classes for cls_head when --lambda_ce=0 and --retrieval4 is used")
    p.add_argument("--retrieval4_pred_min_conf", type=float, default=0.0,
                   help="minimum confidence to include node in predicted-label cluster")
    p.add_argument("--retrieval4_prob_topm", type=int, default=0,
                   help="use only top-m classes from query probabilities for bank allocation (0=use all)")

    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # device & amp
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda"
            and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )
    if not args.amp:
        amp_dtype = torch.float32

    print(f"[Device] {device} AMP={args.amp} dtype={amp_dtype}")
    print(f"[FusionTokens] img={args.img_tokens}, spot={args.spot_tokens}, gene={args.gene_tokens}")
    attn_out_dir = None
    if args.save_attn:
        attn_out_dir = args.attn_out_dir or os.path.join(args.save_dir, "attn")
        os.makedirs(attn_out_dir, exist_ok=True)
        print(f"[AttnSave] dir={attn_out_dir} max_batches={args.attn_max_batches} every={args.attn_every}")

    # ---------------- data ----------------
    adata = load_h5ad_with_preproc(
        args.h5ad,
        use_hvg=False,
        n_top_genes=None,
        x_key="x_centroid",
        y_key="y_centroid",
    )
    map_df = load_patch_mapping_csv_xenium(
        args.csv,
        root_dir=args.root,
        id_col=args.id_col,
        path_col=args.path_col,
        drop_missing=True,
    )
    if args.inside_he_only:
        if "inside_he" not in map_df.columns:
            try:
                raw_df = pd.read_csv(args.csv)
            except Exception:
                raw_df = None
            if raw_df is None or "inside_he" not in raw_df.columns:
                print("[WARN] inside_he_only requested but 'inside_he' column not found in CSV.")
            else:
                keep = raw_df["inside_he"].astype(str).str.lower().isin(["true", "1", "yes"])
                keep_ids = raw_df.loc[keep, args.id_col].astype(str).tolist()
                map_df = map_df.loc[map_df["spot_id"].astype(str).isin(keep_ids)].reset_index(drop=True)
        else:
            keep = map_df["inside_he"].astype(str).str.lower().isin(["true", "1", "yes"])
            map_df = map_df.loc[keep].reset_index(drop=True)
    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(
        adata, map_df, id_col="spot_id", coord_key="spatial"
    )
    if map_df_aligned.shape[0] != adata.n_obs:
        obs_ids = map_df_aligned["spot_id"].astype(str).tolist()
        adata = adata[obs_ids].copy()

    # --- 문자열 ID → 정수 인덱스 매핑 생성 ---
    global ID2IDX
    id_values = map_df_aligned["spot_id"].astype(str).tolist()
    ID2IDX = {v: i for i, v in enumerate(id_values)}
    print(f"[ID2IDX] built mapping for {len(ID2IDX)} nodes based on 'spot_id'")

    # cell type targets (map_df_aligned 순서와 동일한 node index 기준)
    celltype_y_np = None
    celltype_names = []
    celltype_col_used = None
    need_celltype_labels = bool(args.lambda_ce > 0.0)
    if need_celltype_labels:
        celltype_y_np, celltype_names, celltype_col_used = _build_celltype_targets(
            adata=adata,
            map_df_aligned=map_df_aligned,
            celltype_col=args.celltype_col,
        )
        if args.lambda_ce > 0.0 and len(celltype_names) < 2:
            raise ValueError(
                f"Need >=2 labeled celltype classes when --lambda_ce>0, got {len(celltype_names)}."
            )

    # dataloader
    loader, gene_dim = make_img_gene_dataloader(
        adata=adata,
        map_df=map_df_aligned,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_transform=None,  # foundation 내부 transform 사용 시 adapter에서 호출
    )

    # graph
    if args.graph_type == "delaunay":
        edge_index, edge_weight = build_delaunay_graph(coords_np, k=args.k, use_weight=True)
    else:
        edge_index, edge_weight = build_knn_graph(coords_np, k=args.k, use_weight=True)
    print(f"[Graph] type={args.graph_type}, k={args.k}, E={edge_index.size(1)}")
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    # spot input feature: 여기서는 좌표를 그대로 사용 (N, 2)
    spot_feat_in = torch.from_numpy(coords_np).float().to(device)

    # ---------------- encoders ----------------
    Foundation = inf_encoder_factory(args.enc_name)
    foundation = Foundation(weights_path=None)     # img_encoder
    img_dim = foundation.get_output_dim()

    gene_encoder = GeneEncoder(
        input_dim=gene_dim,
        model_size_omic="small",
        proj_dim=args.proj_dim_gene,
        normalize=False,
    )

    # GCN은 좌표(2차원)를 입력으로 받아 proj_dim_spot 차원 embedding 출력
    spot_gcn = SpatialGCN(
        in_dim=spot_feat_in.size(1),       # ex. 2
        hidden_dim=256,
        out_dim=args.proj_dim_spot,
        dropout=0.2,
    )

    load_pretrained_for_finetune(
        foundation=foundation,
        gene_encoder=gene_encoder,
        gcn=spot_gcn,
        pt_img_backbone=args.pt_img_backbone,
        pt_ig=args.pt_ig,
        pt_is=args.pt_is,
    )

    num_celltypes_model = 0
    if args.lambda_ce > 0.0:
        num_celltypes_model = len(celltype_names)
    elif args.retrieval4:
        num_celltypes_model = int(max(args.retrieval4_num_celltypes, 0))
    if args.retrieval4 and num_celltypes_model <= 0:
        raise ValueError(
            "--retrieval4 requires cls_head. Set --lambda_ce>0 (with labels) "
            "or provide --retrieval4_num_celltypes > 0."
        )

    # 래퍼 모델
    model = CrossAttnZINBModel(
        img_encoder=foundation,
        gene_encoder=gene_encoder,
        gcn=spot_gcn,
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
        num_celltypes=num_celltypes_model,
        cls_head_hidden=args.ce_head_hidden,
        cls_dropout=args.ce_head_dropout,
    )

    if args.retrieval4:
        init_tau = float(max(args.retrieval4_tau, 1e-3))
        if args.retrieval4_learnable_tau:
            model.retrieval4_log_tau = nn.Parameter(torch.tensor(np.log(init_tau), dtype=torch.float32))
            print(f"[Retrieval4] learnable tau enabled (init={init_tau:.5f})")
        else:
            model.register_buffer("retrieval4_tau_const", torch.tensor(init_tau, dtype=torch.float32))
            print(f"[Retrieval4] fixed tau={init_tau:.5f}")

    model = model.to(device)

    # optional: load xattn weights into fusion
    if args.pt_xattn:
        load_xattn_into_fusion(model.fusion, args.pt_xattn)

    # 🔧 필요하면 여기서 encoder / decoder LR 분리도 가능 (지금은 전체 동일)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    celltype_y_t = None
    ce_class_weights = None
    if args.lambda_ce > 0.0 and celltype_y_np is not None:
        celltype_y_t = torch.from_numpy(celltype_y_np).to(device=device, dtype=torch.long)
        if args.ce_class_weight == "balanced":
            valid = celltype_y_np[celltype_y_np >= 0]
            counts = np.bincount(valid, minlength=len(celltype_names)).astype(np.float32)
            w = counts.sum() / np.maximum(counts, 1.0)
            ce_class_weights = torch.tensor(w, dtype=torch.float32, device=device)
            print(f"[CellType] class_weight=balanced (min={w.min():.4f}, max={w.max():.4f})")

    retrieval4_cfg = None
    if args.retrieval4:
        expr_bank_np = _adata_x_to_numpy_float32(adata)
        retrieval4_cfg = {
            "enabled": True,
            "bank_size": int(max(1, args.retrieval4_bank_size)),
            "tau": float(max(args.retrieval4_tau, 1e-3)),
            "mix": float(np.clip(args.retrieval4_mix, 0.0, 1.0)),
            "eval_mix": float(np.clip(args.retrieval4_eval_mix, 0.0, 1.0)),
            "mix_warmup_epochs": int(max(0, args.retrieval4_mix_warmup_epochs)),
            "lambda_retr": float(max(0.0, args.retrieval4_lambda)),
            "fallback": str(args.retrieval4_fallback),
            "pred_min_conf": float(max(0.0, args.retrieval4_pred_min_conf)),
            "prob_topm": int(max(0, args.retrieval4_prob_topm)),
            "expr_bank_np": expr_bank_np,
            "class_index_map": {},
            "keys_bank": None,
            "pred_prob_bank": None,
        }
        print(
            f"[Retrieval4] enabled bank_size={retrieval4_cfg['bank_size']} "
            f"mix(train/eval)={retrieval4_cfg['mix']:.2f}/{retrieval4_cfg['eval_mix']:.2f} "
            f"lambda={retrieval4_cfg['lambda_retr']:.4f} fallback={retrieval4_cfg['fallback']} "
            f"prob_topm={retrieval4_cfg['prob_topm']}"
        )
    
    # ✅ Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    
    def warmup_lr_lambda(epoch):
        """선형 warmup 후 cosine annealing"""
        if epoch < args.warmup_epochs:
            # Linear warmup
            return (epoch + 1.0) / args.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optim, lr_lambda=warmup_lr_lambda)

    best = {"pcc": -1.0, "path": None}
    for ep in range(1, args.epochs + 1):
        if retrieval4_cfg is not None and bool(retrieval4_cfg.get("enabled", False)):
            refresh_every = max(1, int(args.retrieval4_refresh_epochs))
            if ep == 1 or ((ep - 1) % refresh_every == 0):
                banks = _build_retrieval4_banks(
                    model=model,
                    loader=loader,
                    spot_feat_in=spot_feat_in,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    expr_bank_np=retrieval4_cfg["expr_bank_np"],
                    num_celltypes=num_celltypes_model,
                    device=device,
                    amp_dtype=amp_dtype,
                    n_nodes=map_df_aligned.shape[0],
                    pred_min_conf=float(retrieval4_cfg.get("pred_min_conf", 0.0)),
                )
                retrieval4_cfg["keys_bank"] = banks["keys_bank"]
                retrieval4_cfg["pred_prob_bank"] = banks["pred_prob_bank"]
                retrieval4_cfg["class_index_map"] = banks["class_index_map"]
                print(f"[Retrieval4] predicted clusters={len(retrieval4_cfg['class_index_map'])}")
                print(f"[Retrieval4] key bank refreshed at epoch {ep}")

        t0 = time.time()
        tr = train_epoch(
            model, loader, spot_feat_in, edge_index, edge_weight,
            optim, amp_dtype, device,
            epoch=ep, total_epochs=args.epochs,
            lambda_zinb=args.lambda_zinb,
            lambda_mse=args.lambda_mse,
            lambda_pcc=args.lambda_pcc,
            lambda_ce=args.lambda_ce,
            ce_label_smoothing=args.ce_label_smoothing,
            ce_warmup_epochs=args.ce_warmup_epochs,
            celltype_labels_all=celltype_y_t,
            ce_class_weights=ce_class_weights,
            max_grad_norm=args.max_grad_norm,
            retrieval4_cfg=retrieval4_cfg,
        )
        
        # ✅ LR 스케줄 업데이트
        scheduler.step()
        
        ev, P, Y = eval_epoch(
            model, loader, spot_feat_in, edge_index, edge_weight,
            amp_dtype, device,
            epoch=ep, total_epochs=args.epochs,
            save_attn=bool(args.save_attn and (ep % max(args.attn_every, 1) == 0)),
            attn_out_dir=attn_out_dir,
            attn_max_batches=args.attn_max_batches,
            attn_reduce_heads=args.attn_reduce_heads,
            celltype_labels_all=celltype_y_t,
            retrieval4_cfg=retrieval4_cfg,
        )
        dt = time.time() - t0
        retr_msg = ""
        if retrieval4_cfg is not None and bool(retrieval4_cfg.get("enabled", False)):
            retr_msg = (
                f" | retrUse={tr['retr_used']:.3f} "
                f"retrTop1={tr['retr_top1']:.4f} tau={tr['retr_tau']:.4f}"
            )
        print(
            f"[{ep:03d}] trainLoss={tr['loss']:.6f} "
            f"trainCtAcc={tr['ct_acc']:.4f} (n={tr['ct_n']}) | "
            f"eval MSE={ev['mse']:.6e} MAE={ev['mae']:.6e} "
            f"PCC(mean)={ev['pcc_mean']:.4f} med={ev['pcc_median']:.4f} "
            f"evalCtAcc={ev['ct_acc']:.4f} (n={ev['ct_n']})"
            f"{retr_msg}  ({dt:.1f}s)"
        )

        if ev["pcc_mean"] > best["pcc"]:
            best["pcc"] = ev["pcc_mean"]
            out_dir = os.path.join(args.save_dir, "best")
            os.makedirs(out_dir, exist_ok=True)

            # P, Y 둘 다 log-space (log1p) 이므로, recovered_data.h5ad도 log-space 예측 저장
            rec = ad.AnnData(
                X=P,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
                uns=adata.uns.copy(),
            )
            rec.write(
                os.path.join(out_dir, "recovered_data.h5ad"),
                compression="gzip",
            )
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            if args.lambda_ce > 0.0 and len(celltype_names) > 0:
                torch.save(
                    {
                        "celltype_col": celltype_col_used,
                        "celltype_names": celltype_names,
                        "num_celltypes": len(celltype_names),
                    },
                    os.path.join(out_dir, "celltype_meta.pt"),
                )
            best["path"] = out_dir
            print(f"[BEST↑] saved to {out_dir}")

    print(f"Done. best PCC(mean)={best['pcc']:.4f}, path={best['path']}")


if __name__ == "__main__":
    main()
