#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_inference_xattn_withcelltype.py
- xattn 기반으로 파인튜닝된 Cross-Attn + ZINB 모델 추론 스크립트
- 블록별 ckpt(Image/SNN/Coord/XATTN) 또는 전체 ft model ckpt 로드 지원
- gene 입력 소스:
  - mean: train 데이터의 gene-wise mean
  - similarity: 이미지 임베딩 유사도 기반 매칭
  - celltype_similarity: celltype-aware similarity 매칭(명시적)

예시:

CUDA_VISIBLE_DEVICES=1 python /data/project/banana9903_xenium/Code_final/code/utils/run_inference_xattn_withcelltype.py \
  --h5ad /data/project/banana9903_xenium/h5ads/human_lung_addon.h5ad \
  --csv /data/project/banana9903_xenium/contrast_test/expression/human_lung_addon_patches_256/spot_patch_map_256.csv \
  --root /data/project/banana9903_xenium/contrast_test/expression \
  --id_col cell_id --path_col patch_path \
  --train_h5ad /data/project/banana9903_xenium/ckpts_cross_human_lung_addon_xattn_delaunay_10tokens_with_celltype_softcluster_retrieval5_0224/best/recovered_data.h5ad \
  --train_csv /data/project/banana9903_xenium/contrast_test/expression/human_lung_addon_patches_256/spot_patch_map_256.csv \
  --enc_name uni_v1 \
  --proj_dim_gene 256 --proj_dim_spot 256 --fuse_dim 256 --heads 8 \
  --graph_type delaunay --k 12 \
  --img_tokens 10 --gene_tokens 10 --spot_tokens 10 \
  --ft_model_ckpt /data/project/banana9903_xenium/ckpts_cross_human_lung_addon_xattn_delaunay_10tokens_with_celltype_softcluster_retrieval5_0224/best/model.pt \
  --expr_source celltype_similarity \
  --celltype_pred_expr_source eval \
  --batch_size 256 --num_workers 4 \
  --out_h5ad /data/project/banana9903_xenium/inference/recovered_xattn_retrieval_human_lung_addon_0227.h5ad \
  --celltype_meta_pt /data/project/banana9903_xenium/ckpts_cross_human_lung_addon_xattn_delaunay_10tokens_with_celltype_softcluster_retrieval5_0224/best/celltype_meta.pt



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
import torch.nn.functional as F
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
from code.models.spot_features import build_histology_node_features, make_spot_in_proj

from code.utils.lora_utils import attach_lora_to_foundation
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
    topk_cosine_match,
    _expression_matrix_for_spots,
)


class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn, dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1, merge="gated-sum",
                 img_tokens=1, spot_tokens=1, gene_tokens=1,
                 decoder_hidden=1024, gene_dim=1000,
                 num_celltypes: int = 0, cls_head_hidden: int = 0, cls_dropout: float = 0.0,
                 dim_spot_in: int = 256, use_spot: bool = True, use_gene: bool = True):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.gcn = gcn
        # modality ablation (R2): drop the spot (coords/GCN) and/or gene branch in fusion
        self.use_spot = bool(use_spot)
        self.use_gene = bool(use_gene)

        # H(0) for the spatial GCN is the histology embedding of each spot,
        # projected to the width the GCN was pretrained on.
        self.dim_spot_in = int(dim_spot_in)
        self.spot_in_proj = make_spot_in_proj(dim_img, dim_spot_in)

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

    @torch.no_grad()
    def build_spot_features(self, img_loader, n_nodes, device, amp_dtype=None):
        """H(0) for every node: histology embedding -> spot_in_proj."""
        return build_histology_node_features(
            img_encoder=self.img_encoder,
            loader=img_loader,
            n_nodes=n_nodes,
            out_dim=self.dim_spot_in,
            device=device,
            proj=self.spot_in_proj,
            amp_dtype=amp_dtype,
        )

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
            z_fused, _, _, attn = self.fusion(zi, zs, zg, return_attn=True,
                                              use_spot=self.use_spot, use_gene=self.use_gene)
        else:
            z_fused, _, _ = self.fusion(zi, zs, zg, return_attn=False,
                                        use_spot=self.use_spot, use_gene=self.use_gene)
            attn = None
        mu, theta, pi = self.decoder(z_fused)
        cls_logits = self.cls_head(z_fused) if self.cls_head is not None else None
        if return_aux:
            return mu, theta, pi, {"attn": attn, "cls_logits": cls_logits,
                                   "zi": zi, "zg": zg, "zs": zs}
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


def _load_module_from_sd(module, sd, prefixes, name, path):
    """
    Load `module` from the first prefix that yields overlapping keys, failing
    loudly on shape mismatch. load_state_dict(strict=False) still raises on a
    size mismatch, so the old `except Exception: pass` around it silently left
    modules randomly initialized whenever the architecture drifted.
    """
    tgt = module.state_dict()
    for pfx in prefixes:
        sub = _strip_prefix(sd, pfx) if pfx else dict(sd)
        common = [k for k in sub if k in tgt]
        if not common:
            continue
        bad = [(k, tuple(sub[k].shape), tuple(tgt[k].shape))
               for k in common
               if hasattr(sub[k], "shape") and sub[k].shape != tgt[k].shape]
        if bad:
            detail = "\n".join(f"    {k}: ckpt{a} vs model{b}" for k, a, b in bad[:8])
            raise RuntimeError(
                f"[PT] shape mismatch loading {name} from {path}:\n{detail}\n"
                "  Check --pt_proj_dim / --spot_hidden_dim against the checkpoint."
            )
        ret = module.load_state_dict(sub, strict=False)
        print(f"[PT] loaded {name} from {path} "
              f"({len(common)}/{len(tgt)} tensors"
              + (f", {len(ret.missing_keys)} left at init" if ret.missing_keys else "") + ")")
        return True
    raise RuntimeError(
        f"[PT] no keys matching {name} found in {path}. Tried prefixes {prefixes!r}."
    )


def load_pretrained_blocks(foundation, gene_encoder, gcn, spot_in_proj=None,
                           pt_img_backbone=None, pt_ig=None, pt_is=None):
    if pt_img_backbone:
        sd = _load_sd(pt_img_backbone)
        _load_module_from_sd(
            foundation, sd,
            ["img_encoder.foundation.", "foundation.", ""],
            "foundation", pt_img_backbone,
        )
        if spot_in_proj is not None and not isinstance(spot_in_proj, nn.Identity):
            _load_module_from_sd(
                spot_in_proj, sd, ["img_encoder.proj.", "proj."],
                "spot_in_proj", pt_img_backbone,
            )

    if pt_ig:
        sd = _load_sd(pt_ig)
        _load_module_from_sd(gene_encoder, sd, ["gene_encoder.", ""], "gene_encoder", pt_ig)

    if pt_is:
        sd = _load_sd(pt_is)
        _load_module_from_sd(gcn, sd, ["gcn.", ""], "spatial GCN", pt_is)


def load_xattn_into_fusion(fusion: CrossModalFusion, xattn_path: str | None):
    if not xattn_path:
        return
    sd = _load_sd(xattn_path)

    def _load_block(dst_block: nn.Module, *src_prefixes: str):
        for src_prefix in src_prefixes:
            sub = {k[len(src_prefix) + 1:]: v for k, v in sd.items()
                   if k.startswith(src_prefix + ".")}
            if not sub:
                continue
            ret = dst_block.load_state_dict(sub, strict=False)
            missing = len(ret.missing_keys)
            unexpected = len(ret.unexpected_keys)
            if missing or unexpected:
                print(f"[XATTN][warn] {src_prefix} -> fusion (missing={missing}, unexpected={unexpected})")
            else:
                print(f"[XATTN] loaded {src_prefix} into fusion")
            return
        print(f"[XATTN][warn] no keys found for {'/'.join(src_prefixes)} in {xattn_path}")

    _load_block(fusion.xa_ig, "i2g")
    # "i2s" is the current name; "i2c" is the pre-rename (coordinate-era) key.
    _load_block(fusion.xa_is, "i2s", "i2c")


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


def _infer_lora_config(sd: dict | None) -> tuple[bool, int]:
    """Detect a LoRA-adapted image backbone from finetune-checkpoint key names.

    A --with_lora stage-2 run saves the encoder as PEFT rewrote it: every adapted
    Linear becomes '<name>.base_layer.weight' plus '<name>.lora_A.default.weight'
    of shape (r, in_features) and '<name>.lora_B.default.weight' of shape
    (out_features, r). r is therefore recoverable from the checkpoint itself.

    alpha is NOT. It lives in LoraConfig and only reaches the weights through the
    alpha/r scaling applied in forward, so it has to come from the caller
    (--lora_alpha). A wrong value silently rescales every adapted layer instead
    of failing, which is why the default here matches the stage-2 trainer's.
    """
    if not isinstance(sd, dict):
        return False, 0
    ranks = {int(v.shape[0]) for k, v in sd.items()
             if ".lora_A." in k and getattr(v, "ndim", 0) == 2}
    if not ranks:
        return False, 0
    if len(ranks) > 1:
        print(f"[LoRA][warn] checkpoint mixes ranks {sorted(ranks)}; using max")
    return True, max(ranks)


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
    # Crash-resume cache (same knob as SpotFeat): this stage costs ~1h on 1M+
    # cells and the box loses power mid-run.
    _cdir = os.environ.get("SPAM_SPOTFEAT_CACHE")
    _cpath = os.path.join(_cdir, f"ctprobs_{num_nodes}x{num_ct}.npy") if _cdir else None
    if _cpath and os.path.isfile(_cpath):
        try:
            arr = np.load(_cpath)
            if arr.shape == (num_nodes, num_ct) and np.isfinite(arr).all():
                print(f"[cache] celltype probs hit: {_cpath}")
                return arr
        except Exception as e:
            print(f"[cache] celltype probs unreadable ({e}), rebuilding: {_cpath}")
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
    if _cpath:
        np.save(_cpath, probs_all)
        print(f"[cache] celltype probs saved: {_cpath}")
    return probs_all


# ---- SPAM's ACTUAL retrieval (ported from the finetune
# run_finetunning_xattn_withceltype_0214_retrieval_final_version.py):
# per-query entropy-adaptive tau, celltype-probability quota over clusters,
# then a pure cosine-softmax WEIGHTED SUM of the selected bank expression.
# This is the mechanism the manuscript reports; the celltype-top-1 copy is only
# a simplification that the obsbank inference happened to use. ----
def _per_query_tau_np(query_probs, tau_base, tau_alpha):
    p = np.clip(np.asarray(query_probs, dtype=np.float64), 1e-9, None)
    C = p.shape[1]
    H = -(p * np.log(p)).sum(1)
    Hn = H / (np.log(C) if C > 1 else 1.0)
    return np.clip(tau_base * (1.0 + tau_alpha * Hn), 1e-3, 50.0)


def _allocate_cluster_counts(prob_vec, total):
    total = int(max(total, 0))
    if total == 0:
        return np.zeros_like(prob_vec, dtype=np.int64)
    p = np.clip(np.asarray(prob_vec, dtype=np.float64), 0.0, None)
    s = float(p.sum())
    p = p / s if s > 0 else np.ones_like(p) / max(int(p.size), 1)
    raw = p * float(total)
    cnt = np.floor(raw).astype(np.int64)
    rem = int(total - int(cnt.sum()))
    if rem > 0:
        order = np.argsort(-(raw - cnt.astype(np.float64)))
        cnt[order[:rem]] += 1
    return cnt


@torch.no_grad()
def weighted_retrieval_expr(test_rep, pred_probs, train_rep, expr_train, class_to_train,
                            bank_size=5000, tau_base=0.07, tau_alpha=0.3,
                            device="cuda", sim_chunk=512):
    # GPU-resident to avoid per-query host<->device syncs. expr_bank and the
    # per-class candidate index tensors live on-device; only the finished chunk
    # of predictions is copied back to host.
    dev = torch.device(device)
    q = F.normalize(torch.from_numpy(np.ascontiguousarray(test_rep, np.float32)).to(dev), dim=1)
    bank = F.normalize(torch.from_numpy(np.ascontiguousarray(train_rep, np.float32)).to(dev), dim=1)
    B, N = q.size(0), bank.size(0)
    target = int(min(bank_size, N))
    probs = np.asarray(pred_probs, dtype=np.float64)
    tau_q = torch.from_numpy(_per_query_tau_np(probs, tau_base, tau_alpha).astype(np.float32)).to(dev)
    # expr_bank can be huge (e.g. ovarian 1.16M x 1833 = 8.5 GB); on a SHARED GPU
    # keep it on the device only if it comfortably fits, else gather on CPU.
    expr_np = np.ascontiguousarray(expr_train, np.float32)
    G = expr_np.shape[1]
    expr_bytes = expr_np.nbytes
    # Decide GPU vs CPU by ACTUAL free device memory, not a fixed cap: the CPU
    # gather path is ~50x slower (per-query host<->device sync), so use the GPU
    # whenever the bank fits with a safety margin.
    use_gpu_expr = False
    if dev.type == "cuda":
        try:
            free_b, _ = torch.cuda.mem_get_info(dev)
        except Exception:
            free_b = 0
        use_gpu_expr = expr_bytes < 0.7 * free_b
        print(f"[weighted] expr_bank {expr_bytes/1e9:.1f}GB, GPU free {free_b/1e9:.1f}GB "
              f"-> {'GPU' if use_gpu_expr else 'CPU (gather)'}")
    else:
        print(f"[weighted] expr_bank {expr_bytes/1e9:.1f}GB on CPU (gather)")
    expr_gpu = torch.from_numpy(expr_np).to(dev) if use_gpu_expr else None
    cmap = {int(c): torch.as_tensor(np.asarray(v, np.int64), device=dev, dtype=torch.long)
            for c, v in class_to_train.items() if len(v) > 0}
    out = np.zeros((B, G), dtype=np.float32)
    for s in tqdm(range(0, B, sim_chunk), desc=f"weighted-retr(bank={target})", leave=False):
        e = min(B, s + sim_chunk)
        sim = q[s:e] @ bank.t()                       # (chunk, N) fp32, on GPU
        buf = torch.empty((e - s, G), device=dev, dtype=torch.float32)
        for j in range(e - s):
            bi = s + j
            sim_bi = sim[j]                           # (N,) GPU
            quota = _allocate_cluster_counts(probs[bi], target)
            parts = []
            for ci in np.nonzero(quota)[0]:
                cand = cmap.get(int(ci))
                if cand is None:
                    continue
                kk = min(int(quota[ci]), int(cand.numel()))
                if kk <= 0:
                    continue
                _, loc = torch.topk(sim_bi[cand], kk)
                parts.append(cand.index_select(0, loc))
            sel = torch.unique(torch.cat(parts)) if parts else torch.empty(0, dtype=torch.long, device=dev)
            if int(sel.numel()) < target:             # stage-2 global fill
                sf = sim_bi if sel.numel() == 0 else sim_bi.clone().index_fill_(0, sel, -1e9)
                need = min(target - int(sel.numel()), int(sf.numel()))
                if need > 0:
                    _, fi = torch.topk(sf, need)
                    sel = fi if sel.numel() == 0 else torch.cat([sel, fi])
            if sel.numel() == 0:
                _, sel = torch.topk(sim_bi, min(target, N))
            ss = sim_bi.index_select(0, sel)
            if int(ss.numel()) > target:
                vs, ls = torch.topk(ss, target); sel = sel.index_select(0, ls); ss = vs
            w = torch.softmax(ss / tau_q[bi], dim=0)  # (k,)
            if use_gpu_expr:
                buf[j] = (w.unsqueeze(-1) * expr_gpu.index_select(0, sel)).sum(0)
            else:
                w_np = w.detach().cpu().numpy()
                sel_np = sel.detach().cpu().numpy()
                out[bi] = (w_np[:, None] * expr_np[sel_np]).sum(0)
        if use_gpu_expr:
            out[s:e] = buf.cpu().numpy()
    return out


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
    test_image_transform: nn.Module | None = None,
    bank_subsample: int | None = None,
    bank_seed: int = 0,
    ablation: bool = False,
    topk: int = 10,
    tau: float = 0.07,
    weighted: bool = False,
    w_bank_size: int = 5000,
    w_tau_base: float = 0.07,
    w_tau_alpha: float = 0.3,
) -> dict:
    if test_map_df is None:
        raise ValueError("test_map_df is required for similarity matching.")
    if img_encoder is None:
        raise ValueError("img_encoder is required for celltype-aware similarity.")
    # E7 rotation: test queries may use a rotated transform; train bank never rotates.
    if test_image_transform is None:
        test_image_transform = image_transform

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
        ImageMapDataset(test_map_df, image_transform=test_image_transform, id_col="spot_id"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Crash-resume cache for the embedding passes: the machine has died in the
    # similarity stage repeatedly, so persist the (deterministic, frozen-model)
    # embeddings and resume there instead of recomputing hours of forwards.
    _cdir = os.environ.get("SPAM_SPOTFEAT_CACHE")

    def _cached_embed(loader, n_rows, tag, desc):
        path = os.path.join(_cdir, f"emb_{tag}_{n_rows}.npz") if _cdir else None
        if path and os.path.isfile(path):
            try:
                z = np.load(path, allow_pickle=True)
                rep, ids = z["rep"], [str(s) for s in z["ids"]]
                if rep.shape[0] == n_rows and np.isfinite(rep).all():
                    print(f"[cache] {desc} hit: {path}")
                    return rep, ids
                print(f"[cache] {desc} invalid, rebuilding: {path}")
            except Exception as e:
                print(f"[cache] {desc} unreadable ({e}), rebuilding: {path}")
        rep, ids = gather_image_representations(
            img_encoder, loader, device=device_t, amp_dtype=amp_dtype, normalize=True,
            show_progress=True, desc=desc,
        )
        if path:
            np.savez(path, rep=rep, ids=np.asarray(ids, dtype=object))
            print(f"[cache] {desc} saved: {path}")
        return rep, ids

    train_rep, train_ids = _cached_embed(train_loader, len(train_map_df), "train", "embed(train)")
    test_rep, test_ids = _cached_embed(test_loader, len(test_map_df), "test", "embed(test)")

    # R2-4 bank ablation: randomly keep `bank_subsample` cells in the retrieval
    # bank (train_rep + celltypes + spot_ids + expr all restricted consistently).
    if bank_subsample is not None and 0 < bank_subsample < len(train_rep):
        rng = np.random.default_rng(bank_seed)
        keep = np.sort(rng.choice(len(train_rep), int(bank_subsample), replace=False))
        train_rep = train_rep[keep]
        train_celltypes = np.asarray(train_celltypes, dtype=object)[keep]
        train_spot_ids = [train_spot_ids[i] for i in keep]
        train_map_df = train_map_df.iloc[keep].reset_index(drop=True)
        print(f"[bank] subsampled retrieval bank to {len(train_rep)} cells (seed={bank_seed})")
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
    if weighted:
        # SPAM's actual retrieval: prob-quota + cosine-softmax weighted sum.
        print(f"[retrieval] WEIGHTED (bank={w_bank_size}, tau_base={w_tau_base}, "
              f"tau_alpha={w_tau_alpha}) — SPAM's trained mechanism")
        pred_expr = weighted_retrieval_expr(
            test_rep, pred_celltype_probs, train_rep, np.asarray(expr_train, np.float32),
            class_to_train, bank_size=w_bank_size, tau_base=w_tau_base, tau_alpha=w_tau_alpha,
            device=sim_device if sim_device != "cpu" else "cpu", sim_chunk=sim_chunk,
        )
    else:
        pred_expr = expr_train[selected_idx].astype(np.float32)
    if fallback == "mean":
        fallback_mask = ~used_ct_constraint
        if fallback_mask.any():
            pred_expr[fallback_mask] = _mean_gene_vector(adata)[None, :]

    matched_train_spot_ids = np.asarray(train_spot_ids, dtype=object)[selected_idx]
    matched_train_celltype = train_celltypes[selected_idx]

    # Retrieval-strategy ablation (R2): compute alternative aggregations in the
    # same pass, reusing train_rep/test_rep/expr_train. Strategies:
    #   global_top1      = expr of single nearest bank cell (no celltype)
    #   global_topk      = softmax(sim/tau)-weighted mean of top-K nearest (no celltype)
    #   global_mean      = mean over the whole bank (no retrieval)
    #   celltype_centroid= mean over the predicted celltype's bank cells
    #   celltype_top1    = current SPAM (pred_expr, celltype-constrained top-1)
    ablation_preds = None
    if ablation:
        gmean = _mean_gene_vector(adata).astype(np.float32)
        # global top-1
        p_g1 = expr_train[global_idx].astype(np.float32)
        # global top-K weighted
        tk_idx, tk_sim = topk_cosine_match(
            train_rep, test_rep, k=topk, chunk_size=sim_chunk,
            device=sim_device, show_progress=True, desc=f"topk={topk}(global)")
        w = np.exp((tk_sim - tk_sim.max(1, keepdims=True)) / float(tau))
        w = (w / w.sum(1, keepdims=True)).astype(np.float32)          # (N,K)
        p_gk = np.einsum("nk,nkg->ng", w, expr_train[tk_idx].astype(np.float32)).astype(np.float32)
        # celltype centroid (mean expr within predicted celltype; fallback=global mean)
        ct_cent = np.repeat(gmean[None, :], len(celltype_names), axis=0).astype(np.float32)
        for ci in range(len(celltype_names)):
            inds = class_to_train.get(ci, np.array([], dtype=np.int64))
            if inds.size: ct_cent[ci] = expr_train[inds].mean(0)
        p_cc = ct_cent[pred_ct_idx].astype(np.float32)
        # global mean
        p_gm = np.repeat(gmean[None, :], test_rep.shape[0], axis=0).astype(np.float32)
        ablation_preds = {
            "global_top1": p_g1, "global_topk": p_gk,
            "global_mean": p_gm, "celltype_centroid": p_cc,
            "celltype_top1": pred_expr,
        }

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
        "ablation_preds": ablation_preds,
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
    ap.add_argument("--img_tokens", type=int, default=10,   # Supp.: 10
                    help="number of image query tokens in fusion cross-attention")
    ap.add_argument("--spot_tokens", type=int, default=10,
                    help="number of spatial tokens in fusion cross-attention")
    ap.add_argument("--gene_tokens", type=int, default=10,
                    help="number of gene tokens in fusion cross-attention")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--merge", default="gated-sum", choices=["gated-sum", "concat-proj"])
    ap.add_argument("--modalities", default="all",
                    choices=["all", "image", "image+coords", "image+gene"],
                    help="modality ablation (R2): all=image+coords+gene; "
                         "image=image only; image+coords=drop gene; image+gene=drop coords")

    ap.add_argument("--graph_type", default="delaunay", choices=["knn", "delaunay"],
                    help="Must match the graph used at finetuning time.")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--pt_proj_dim", type=int, default=256,
                    help="Pretraining --proj_dim: width of H(0) and of the "
                         "spatial GCN input. Must match the finetuned model.")
    ap.add_argument("--spot_hidden_dim", type=int, default=256,
                    help="Hidden width of the spatial GCN; must match finetuning.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--lora_alpha", type=int, default=32,
                    help="LoRA alpha used at stage 2. Not stored in the checkpoint, "
                         "so it must match the training run (trainer default: 32).")
    ap.add_argument("--lora_r", type=int, default=0,
                    help="LoRA rank override; 0 = infer from the checkpoint.")
    ap.add_argument("--no_lora", action="store_true",
                    help="Skip LoRA attachment even if the checkpoint carries adapters "
                         "(evaluates the un-adapted backbone; for ablation only).")

    ap.add_argument("--ft_model_ckpt", default=None, help="finetune된 전체 모델 ckpt (model.pt)")
    ap.add_argument("--pt_img_backbone", default=None, help="pretrained image backbone ckpt")
    ap.add_argument("--pt_ig", default=None, help="pretrained img↔gene ckpt")
    ap.add_argument("--pt_is", default=None, help="pretrained img↔spatial ckpt")
    ap.add_argument("--pt_xattn", default=None, help="pretrained xattn ckpt (XATTN_pretrained.pt)")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--rotate_test", type=int, default=0, choices=[0, 90, 180, 270],
                    help="E7: rotate ONLY test query patches by this many degrees "
                         "(loss-less PIL transpose). Train bank never rotates.")
    ap.add_argument("--hflip_test", action="store_true",
                    help="E7: horizontal-flip test query patches.")
    ap.add_argument("--bank_subsample", type=int, default=None,
                    help="R2-4: randomly keep this many cells in the retrieval bank.")
    ap.add_argument("--bank_seed", type=int, default=0,
                    help="R2-4: RNG seed for bank subsampling.")

    ap.add_argument("--expr_source", default="mean", choices=["mean", "similarity", "celltype_similarity"])
    ap.add_argument("--weighted_retrieval", action="store_true",
                    help="Use SPAM's ACTUAL trained retrieval (celltype-prob quota + "
                         "cosine-softmax weighted sum) instead of the simplified top-1 copy. "
                         "Matches the manuscript-reported mechanism.")
    ap.add_argument("--w_bank_size", type=int, default=5000)
    ap.add_argument("--w_tau_base", type=float, default=0.07)
    ap.add_argument("--w_tau_alpha", type=float, default=0.3)
    ap.add_argument("--retrieval_ablation", action="store_true",
                    help="R2: with --expr_source celltype_similarity, also compute & save 5 "
                         "aggregation strategies (global top-1/top-K/mean, celltype centroid/top-1) "
                         "to <out>_retrieval_ablation.npz")
    ap.add_argument("--retr_topk", type=int, default=10, help="K for global top-K weighted strategy")
    ap.add_argument("--retr_tau", type=float, default=0.07, help="softmax temperature for top-K weights")
    ap.add_argument("--keep_control_probes", action="store_true",
                    help="Build the model on the FULL panel (do not drop control probes). "
                         "Required to match finetune checkpoints trained on the 541-gene panel; "
                         "otherwise gene_encoder/cls_head shape-mismatch and celltype prediction corrupts.")
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
    ap.add_argument("--dump_embeddings", type=str, default=None,
                    help="If set, dump per-cell projected z_img/z_gene/z_sp (from true expr) "
                         "to this .npz and exit before retrieval (for the alignment UMAP).")
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

    _drop_ctrl = not args.keep_control_probes
    eval_adata = load_h5ad_with_preproc(
        args.h5ad, use_hvg=False, n_top_genes=None, x_key="x_centroid", y_key="y_centroid",
        drop_controls=_drop_ctrl,
    )
    train_adata = load_h5ad_with_preproc(
        args.train_h5ad, use_hvg=False, n_top_genes=None, x_key="x_centroid", y_key="y_centroid",
        drop_controls=_drop_ctrl,
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

    # E7: build a test-only rotated transform (loss-less PIL transpose then the
    # foundation eval transform). Train bank keeps `image_tf` unrotated.
    test_image_tf = image_tf
    if args.rotate_test or args.hflip_test:
        from PIL import Image as _PILImage
        _ops = []
        if args.rotate_test:
            _ops.append({90: _PILImage.ROTATE_90, 180: _PILImage.ROTATE_180,
                         270: _PILImage.ROTATE_270}[args.rotate_test])
        if args.hflip_test:
            _ops.append(_PILImage.FLIP_LEFT_RIGHT)

        def _rot_tf(img, _base=image_tf, _ops=_ops):
            for op in _ops:
                img = img.transpose(op)
            return _base(img)
        test_image_tf = _rot_tf
        print(f"[E7] test queries rotated: rotate={args.rotate_test} hflip={args.hflip_test}")

    img_loader = make_img_only_dataloader(
        map_df=map_df_aligned,
        batch_size=max(1, args.batch_size),
        num_workers=max(1, args.num_workers),
        image_transform=test_image_tf,
        shuffle=False,
    )

    edge_index, edge_weight = build_spatial_graph(coords_np, k=args.k, graph_type=args.graph_type)
    print(f"[Graph] type={args.graph_type}, k={args.k}, E={edge_index.size(1)}")
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    img_dim = foundation.get_output_dim()
    gene_dim = eval_adata.n_vars
    n_nodes = int(coords_np.shape[0])
    # Coordinates define the graph edges only; H(0) is built from histology
    # once the encoder weights are in place (after the ft checkpoint is loaded).
    gene_encoder = GeneEncoder(input_dim=gene_dim, model_size_omic="small",
                               proj_dim=args.proj_dim_gene, normalize=False)
    gcn = SpatialGCN(in_dim=args.pt_proj_dim, hidden_dim=args.spot_hidden_dim,
                     out_dim=args.proj_dim_spot, dropout=0.2)

    load_pretrained_blocks(
        foundation, gene_encoder, gcn,
        pt_img_backbone=args.pt_img_backbone,
        pt_ig=args.pt_ig,
        pt_is=args.pt_is,
    )

    ft_sd = _load_sd(args.ft_model_ckpt) if args.ft_model_ckpt else None
    ckpt_num_ct, ckpt_ct_hidden = _infer_cls_head_config(ft_sd)

    # A --with_lora checkpoint only lines up with a backbone that has the same
    # adapters injected. Attach them here — after load_pretrained_blocks, whose
    # pt_* state dicts use the pre-PEFT names, and before the full checkpoint is
    # applied below.
    ckpt_has_lora, ckpt_lora_r = _infer_lora_config(ft_sd)
    if ckpt_has_lora and not args.no_lora:
        lora_r = args.lora_r if args.lora_r > 0 else ckpt_lora_r
        foundation = attach_lora_to_foundation(
            foundation, backbone_name=args.enc_name,
            r=lora_r, alpha=args.lora_alpha, dropout=0.0, bias="none",
        )
        r_src = "--lora_r" if args.lora_r > 0 else "checkpoint"
        print(f"[LoRA] adapters attached: r={lora_r} (from {r_src}), "
              f"alpha={args.lora_alpha}, scaling={args.lora_alpha / max(lora_r, 1):.3f}")
    elif ckpt_has_lora:
        print("[LoRA][warn] --no_lora set: checkpoint adapters (r="
              f"{ckpt_lora_r}) will be dropped")

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
        dim_spot_in=args.pt_proj_dim,
        use_spot=args.modalities in ("all", "image+coords"),
        use_gene=args.modalities in ("all", "image+gene"),
    )
    if args.modalities != "all":
        print(f"[modality-ablation] modalities={args.modalities} "
              f"(use_spot={model.use_spot}, use_gene={model.use_gene})")

    # spot_in_proj lives inside the model, so it is loaded here rather than in
    # load_pretrained_blocks above (which must run before the LoRA attach, since
    # the pt_* state dicts use pre-PEFT names). A full --ft_model_ckpt overrides
    # it below; this only matters when running from block checkpoints.
    if args.pt_img_backbone and not isinstance(model.spot_in_proj, nn.Identity):
        _load_module_from_sd(
            model.spot_in_proj, _load_sd(args.pt_img_backbone),
            ["img_encoder.proj.", "proj."], "spot_in_proj", args.pt_img_backbone,
        )
    model = model.to(device)

    if args.pt_xattn:
        load_xattn_into_fusion(model.fusion, args.pt_xattn)

    if args.ft_model_ckpt:
        sd = ft_sd if ft_sd is not None else _load_sd(args.ft_model_ckpt)
        try:
            incompatible = model.load_state_dict(sd, strict=False)
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            print(f"[FT] loaded full model state from {args.ft_model_ckpt} "
                  f"(strict=False): {len(sd)} ckpt tensors, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

            # strict=False is deliberate (cls_head/decoder shapes differ across
            # runs), but it also means an architecture mismatch produces no error
            # at all — a LoRA checkpoint loaded into an un-adapted backbone lands
            # here with all 384 img_encoder keys unexpected and silently scores
            # the raw pretrained encoder. Surface that instead of hiding it.
            for label, keys in (("missing", missing), ("unexpected", unexpected)):
                enc_keys = [k for k in keys if k.startswith("img_encoder.")]
                if enc_keys:
                    print(f"[FT][warn] {len(enc_keys)} {label} img_encoder keys, "
                          f"e.g. {enc_keys[:3]}")
            if any(".lora_A." in k for k in unexpected):
                raise SystemExit(
                    "ABORT: the checkpoint carries LoRA adapters but the model "
                    "has none, so the image encoder would silently fall back to "
                    "base weights. Re-run without --no_lora."
                )
        except Exception as e:
            print(f"[FT][warn] failed to load full model from {args.ft_model_ckpt}: {e}")

    model.eval()

    # H(0) for the spatial GCN — built with the final (post-checkpoint) encoder
    # so it reproduces exactly what finetuning fed the GCN.
    spot_feat_in = model.build_spot_features(
        img_loader=img_loader, n_nodes=n_nodes,
        device=device, amp_dtype=amp_dtype if args.amp else torch.float32,
    )

    sim_out = None
    gene_mean = _mean_gene_vector(train_adata)
    if args.expr_source == "mean":
        expr_matrix = np.repeat(gene_mean[None, :], map_df_aligned.shape[0], axis=0)
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
            if args.dump_embeddings:
                print(f"[dump] extracting z_img/z_gene/z_sp (true expr) -> {args.dump_embeddings}", flush=True)
                model.eval()
                N_d = int(expr_base_for_ct.shape[0])
                expr_t = torch.from_numpy(expr_base_for_ct).to(device)
                idx_l, zi_l, zg_l, zs_l = [], [], [], []
                with torch.no_grad():
                    for bimg in tqdm(img_loader, desc="dump emb", leave=False):
                        if isinstance(bimg, dict):
                            imgs = bimg["image"].to(device); idxs = bimg["idx"].to(device).long()
                        else:
                            imgs, idxs = bimg; imgs = imgs.to(device); idxs = idxs.to(device).long()
                        b = {"image": imgs, "expr": expr_t[idxs], "idx": idxs}
                        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                            enabled=(device.type == "cuda" and args.amp)):
                            _, _, _, aux_d = model(b, spot_feat_in, edge_index, edge_weight, return_aux=True)
                        idx_l.append(idxs.cpu().numpy())
                        zi_l.append(aux_d["zi"].float().cpu().numpy())
                        zg_l.append(aux_d["zg"].float().cpu().numpy())
                        zs_l.append(aux_d["zs"].float().cpu().numpy())
                allidx = np.concatenate(idx_l)
                ZI = np.zeros((N_d, zi_l[0].shape[1]), np.float32); ZI[allidx] = np.concatenate(zi_l)
                ZG = np.zeros((N_d, zg_l[0].shape[1]), np.float32); ZG[allidx] = np.concatenate(zg_l)
                ZS = np.zeros((N_d, zs_l[0].shape[1]), np.float32); ZS[allidx] = np.concatenate(zs_l)
                ids_d = map_df_aligned["spot_id"].astype(str).to_numpy()
                np.savez(args.dump_embeddings, zi=ZI, zg=ZG, zs=ZS, ids=np.asarray(ids_d, dtype=object))
                print(f"[dump] saved {args.dump_embeddings}  zi{ZI.shape} zg{ZG.shape} zs{ZS.shape}", flush=True)
                import sys as _sys; _sys.exit(0)
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
                test_image_transform=test_image_tf,
                img_encoder=model.img_encoder,
                min_conf=args.celltype_min_conf,
                fallback=args.celltype_fallback,
                bank_subsample=args.bank_subsample,
                bank_seed=args.bank_seed,
                weighted=args.weighted_retrieval,
                w_bank_size=args.w_bank_size,
                w_tau_base=args.w_tau_base,
                w_tau_alpha=args.w_tau_alpha,
                ablation=args.retrieval_ablation,
                topk=args.retr_topk,
                tau=args.retr_tau,
            )
            expr_matrix = sim_out["pred_expression"]
            if args.retrieval_ablation and sim_out.get("ablation_preds"):
                import os as _os
                ap = sim_out["ablation_preds"]
                stem = _os.path.splitext(args.out_h5ad)[0]
                outp = f"{stem}_retrieval_ablation.npz"
                np.savez(outp,
                         test_ids=np.asarray(sim_out["test_ids"], dtype=object),
                         gene_names=np.asarray(list(train_adata.var_names), dtype=object),
                         **{k: v.astype(np.float32) for k, v in ap.items()})
                print(f"[retrieval-ablation] saved 5 strategies -> {outp}")

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
            # Store log1p(mu): the finetuning evaluation scores in log1p(CP10K)
            # space (eval_epoch: yhat_log = log1p(clamp(mu))), and every metric
            # in the manuscript is computed there. Saving raw mu put predictions
            # on a different scale from every downstream consumer.
            preds.append(torch.log1p(torch.clamp(mu, min=0.0)).detach().cpu())
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

    rec.write(args.out_h5ad, compression="gzip")
    print(f"[DONE] saved: {args.out_h5ad}  (shape={P.shape})")

    # Retrieval-strategy ablation (R2, corrected): SPAM's prediction is the model
    # DECODER OUTPUT, not the retrieved copy. So for each alternative aggregation
    # strategy we feed its retrieved expression as the gene-token INPUT and run
    # the same decoder forward, then save the OUTPUT (comparable to the 0.46
    # headline). celltype_top1's output == args.out_h5ad above.
    if args.retrieval_ablation and sim_out is not None and sim_out.get("ablation_preds"):
        stem = os.path.splitext(args.out_h5ad)[0]
        for strat, strat_expr in sim_out["ablation_preds"].items():
            if strat == "celltype_top1":
                continue  # identical to the main output already saved
            em_t = torch.from_numpy(np.ascontiguousarray(strat_expr, dtype=np.float32)).to(device)
            preds2 = []
            with torch.no_grad():
                for batch_imgs in tqdm(img_loader, desc=f"decode:{strat}", leave=False):
                    if isinstance(batch_imgs, dict):
                        images = batch_imgs["image"].to(device); idxs = batch_imgs["idx"].to(device).long()
                    else:
                        images, idxs = batch_imgs; images = images.to(device); idxs = idxs.to(device).long()
                    batch = {"image": images, "expr": em_t[idxs], "idx": idxs}
                    with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                        enabled=(device.type == "cuda" and args.amp)):
                        mu, theta, pi = model(batch, spot_feat_in, edge_index, edge_weight, return_attn=False)
                    preds2.append(torch.log1p(torch.clamp(mu, min=0.0)).detach().cpu())
            P2 = torch.cat(preds2, dim=0).numpy()
            rec2 = ad.AnnData(X=P2, obs=eval_adata.obs.copy(), var=eval_adata.var.copy())
            outp = f"{stem}_out_{strat}.h5ad"
            rec2.write(outp, compression="gzip")
            print(f"[retrieval-ablation] decoder output {strat} -> {outp}  (shape={P2.shape})")
        print("===== retrieval-ablation decoder outputs DONE =====")


if __name__ == "__main__":
    main()
