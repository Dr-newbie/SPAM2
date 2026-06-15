# run_contrastive2.py
# Cross-attention fusion before contrastive (image↔gene, image↔coord)
#
# Example:
'''
CUDA_VISIBLE_DEVICES=0 \
/data/project/banana9903_xenium/.conda-envs/SPAM/bin/python Code_final/code/utils/run_contrastive_xattn.py \
--h5ad /data/project/banana9903_xenium/h5ads/human_reactive_follicular.h5ad \
--csv /data/project/banana9903_xenium/contrast_test/expression/human_reactive_follicular_patches_256/spot_patch_map_256.csv \
--enc_name uni_v1 \
--proj_dim 256 \
--epochs 50 --batch_size 256 \
--save_dir /data/project/banana9903_xenium/ckpts_human_reactive_follicular_xattn_10tokens \
--device cuda:0 \
--img_tokens 10 \
--gene_tokens 10 \
--coord_tokens 10 \
--graph_type delaunay

'''
import os
import sys
import time
import argparse
from typing import Optional, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # /Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 표준모듈 'code'와 충돌 방지: 이미 로드돼 있으면 제거
sys.modules.pop("code", None)

from code.models.Foundations import inf_encoder_factory
from code.models.GCN_update import SpatialGCN
from code.models.graph_construction import build_knn_graph
from code.models.gene_encoder import GeneEncoder
from code.utils.lora_utils import attach_lora_to_foundation
from code.utils.attention import CrossAttentionBlock
from code.utils.contrastive import ImageFoundationAdapter, build_spatial_graph
from code.utils.dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_gene_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
    extract_coords_np_in_order,
)
from torch_geometric.utils import subgraph


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
# Model blocks
# ----------------------------
class XAttnContrastiveModel(nn.Module):
    def __init__(
        self,
        img_encoder: nn.Module,
        gene_encoder: nn.Module,
        proj_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.15,
        lam_dir: float = 0.5,
        img_tokens: int = 1,
        gene_tokens: int = 1,
        coord_tokens: int = 1,
    ):
        super().__init__()
        if img_tokens < 1 or gene_tokens < 1 or coord_tokens < 1:
            raise ValueError("img_tokens/gene_tokens/coord_tokens must be >= 1")
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.temperature = float(temperature)
        self.lam_dir = float(lam_dir)
        self.proj_dim = int(proj_dim)
        self.img_tokens = int(img_tokens)
        self.gene_tokens = int(gene_tokens)
        self.coord_tokens = int(coord_tokens)

        self.i2g = CrossAttentionBlock(proj_dim, proj_dim, proj_dim, num_heads, dropout)
        self.g2i = CrossAttentionBlock(proj_dim, proj_dim, proj_dim, num_heads, dropout)
        self.i2c = CrossAttentionBlock(proj_dim, proj_dim, proj_dim, num_heads, dropout)
        self.c2i = CrossAttentionBlock(proj_dim, proj_dim, proj_dim, num_heads, dropout)

        self.img_tokenizer = None if self.img_tokens == 1 else nn.Linear(proj_dim, proj_dim * self.img_tokens)
        self.gene_tokenizer = None if self.gene_tokens == 1 else nn.Linear(proj_dim, proj_dim * self.gene_tokens)
        self.coord_tokenizer = None if self.coord_tokens == 1 else nn.Linear(proj_dim, proj_dim * self.coord_tokens)

    def _to_tokens(self, x: torch.Tensor, tokenizer: Optional[nn.Module], n_tokens: int) -> torch.Tensor:
        if n_tokens == 1:
            return x.unsqueeze(1)
        t = tokenizer(x)
        return t.view(x.size(0), n_tokens, self.proj_dim)

    @staticmethod
    def _pool_tokens(x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x.squeeze(1)
        return x.mean(dim=1)

    def cross_attend(self, zi: torch.Tensor, zg: torch.Tensor, zc: torch.Tensor, return_attn: bool = False):
        qi = self._to_tokens(zi, self.img_tokenizer, self.img_tokens)
        kg = self._to_tokens(zg, self.gene_tokenizer, self.gene_tokens)
        kc = self._to_tokens(zc, self.coord_tokenizer, self.coord_tokens)

        if return_attn:
            zi2g_t, attn_i2g = self.i2g(qi, kg, kg, return_attn=True)
            zg2i_t, attn_g2i = self.g2i(kg, qi, qi, return_attn=True)
            zi2c_t, attn_i2c = self.i2c(qi, kc, kc, return_attn=True)
            zc2i_t, attn_c2i = self.c2i(kc, qi, qi, return_attn=True)
        else:
            zi2g_t = self.i2g(qi, kg, kg)
            zg2i_t = self.g2i(kg, qi, qi)
            zi2c_t = self.i2c(qi, kc, kc)
            zc2i_t = self.c2i(kc, qi, qi)
            attn_i2g = attn_g2i = attn_i2c = attn_c2i = None

        zi2g = self._pool_tokens(zi2g_t)
        zg2i = self._pool_tokens(zg2i_t)
        zi2c = self._pool_tokens(zi2c_t)
        zc2i = self._pool_tokens(zc2i_t)

        zi2g = F.normalize(zi2g, dim=1)
        zg2i = F.normalize(zg2i, dim=1)
        zi2c = F.normalize(zi2c, dim=1)
        zc2i = F.normalize(zc2i, dim=1)
        if return_attn:
            return zi2g, zg2i, zi2c, zc2i, {
                "attn_i2g": attn_i2g,
                "attn_g2i": attn_g2i,
                "attn_i2c": attn_i2c,
                "attn_c2i": attn_c2i,
            }
        return zi2g, zg2i, zi2c, zc2i

    def contrastive_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits = (a @ b.t()) / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.t(), targets)
        return self.lam_dir * loss_ab + (1.0 - self.lam_dir) * loss_ba


def _unique_trainable_params(*modules: Iterable[nn.Module]):
    seen, params = set(), []
    for m in modules:
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def _save_all(
    save_dir: str,
    img_encoder: nn.Module,
    gene_encoder: nn.Module,
    coord_gcn: nn.Module,
    xattn_model: nn.Module,
):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(img_encoder.state_dict(), os.path.join(save_dir, "Image_pretrained.pt"))
    torch.save(gene_encoder.state_dict(), os.path.join(save_dir, "SNN_pretrained.pt"))
    torch.save(coord_gcn.state_dict(), os.path.join(save_dir, "Coord_pretrained.pt"))
    torch.save(xattn_model.state_dict(), os.path.join(save_dir, "XATTN_pretrained.pt"))

def _is_finite(*tensors: torch.Tensor) -> bool:
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all().item():
            return False
    return True


def _save_attn_npz(path: str, attn_map: Dict[str, torch.Tensor], reduce_heads: bool, meta: Dict[str, np.ndarray] | None = None):
    payload: Dict[str, np.ndarray] = {}
    for k, v in attn_map.items():
        if v is None:
            continue
        x = v.detach().float().cpu()
        if reduce_heads and x.ndim == 4:
            x = x.mean(dim=1)
        payload[k] = x.numpy()
    if meta is not None:
        payload.update(meta)
    if not payload:
        return
    np.savez_compressed(path, **payload)


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None, help="CSV patch_path가 상대경로면 합칠 루트")
    p.add_argument("--use_hvg", action="store_true", help="HVG로 유전자 선택 (기본은 평균상위)")
    p.add_argument("--n_top_genes", type=int, default=541)

    # coords
    p.add_argument("--coord_key", default="spatial")
    p.add_argument("--use_csv_coords", action="store_true",
                   help="CSV 좌표 컬럼을 사용 (x_he_px, y_he_px 등)")
    p.add_argument("--csv_coord_cols", default="x_he_px,y_he_px")
    p.add_argument("--csv_to_unit", type=float, default=None)
    p.add_argument("--coord_norm", default="standardize", choices=["standardize", "none"])

    # model / foundation
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--with_lora", action="store_true",
                   help="HF 기반 백본(plip/phikon)에서만 활성화. timm 백본은 자동으로 스킵됨.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # graph / coord-gcn
    p.add_argument("--k", type=int, default=12, help="KNN k for spatial graph")
    p.add_argument("--graph_type", default="knn", choices=["knn", "delaunay"],
                   help="Spatial graph type for coordinate-based GCN.")
    p.add_argument("--gcn_device", default=None,
                   help='GCN device override: "cpu", "cuda", "cuda:0" ... (default: use --device)')

    # xattn
    p.add_argument("--xattn_heads", type=int, default=8)
    p.add_argument("--xattn_dropout", type=float, default=0.1)
    p.add_argument("--img_tokens", type=int, default=1,
                   help="number of image query tokens for cross-attention")
    p.add_argument("--gene_tokens", type=int, default=1,
                   help="number of gene tokens for cross-attention")
    p.add_argument("--coord_tokens", type=int, default=1,
                   help="number of coordinate tokens for cross-attention")
    p.add_argument("--coord_hidden_dim", type=int, default=128)
    p.add_argument("--coord_dropout", type=float, default=0.1)
    p.add_argument("--save_attn", action="store_true",
                   help="save attention weights for sampled training batches")
    p.add_argument("--attn_out_dir", default=None,
                   help="directory to save attention npz files (default: <save_dir>/attn)")
    p.add_argument("--attn_max_batches", type=int, default=10,
                   help="max number of batches to dump per epoch when --save_attn")
    p.add_argument("--attn_every", type=int, default=1,
                   help="save attention every N epochs when --save_attn")
    p.add_argument("--attn_reduce_heads", action="store_true",
                   help="store mean over heads instead of per-head attention")

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--lam_dir", type=float, default=0.5,
                   help="방향성 가중치 (A→B vs B→A). 0.5=대칭")
    p.add_argument("--lam_ig", type=float, default=1.0, help="image↔gene loss weight")
    p.add_argument("--lam_ic", type=float, default=1.0, help="image↔coord loss weight")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_norm", type=float, default=0.0)
    p.add_argument("--save_dir", default="./ckpts")

    # device/amp
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = select_device(args.device)
    setup_cuda_flags()
    amp_dtype = pick_amp_dtype(args.amp_dtype) if args.amp else torch.float32
    gcn_device = torch.device(args.gcn_device) if args.gcn_device is not None else device
    if gcn_device.type == "cpu" and args.amp:
        print("[WARN] gcn_device=cpu -> disabling AMP to avoid GradScaler on CPU params.")
        args.amp = False
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"[Device] {device}, AMP={args.amp}, dtype={amp_dtype}")
    print(f"[XAttnTokens] img={args.img_tokens}, gene={args.gene_tokens}, coord={args.coord_tokens}")
    attn_out_dir = None
    if args.save_attn:
        attn_out_dir = args.attn_out_dir or os.path.join(args.save_dir, "attn")
        os.makedirs(attn_out_dir, exist_ok=True)
        print(f"[AttnSave] dir={attn_out_dir} max_batches={args.attn_max_batches} every={args.attn_every}")
    if args.temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}")

    # 1) Load data + align
    adata = load_h5ad_with_preproc(
        args.h5ad,
        use_hvg=args.use_hvg,
        n_top_genes=args.n_top_genes,
        x_key="x_centroid",
        y_key="y_centroid",
    )
    root_dir = args.root or os.path.dirname(args.csv)
    map_df = load_patch_mapping_csv_xenium(
        args.csv,
        root_dir=root_dir,
        drop_missing=True,
    )
    if len(map_df) == 0 and args.root is None:
        # Common case: patch_path already includes folder name; default root joins it twice.
        alt_root = os.path.dirname(root_dir)
        map_df = load_patch_mapping_csv_xenium(
            args.csv,
            root_dir=alt_root,
            drop_missing=True,
        )
        if len(map_df) == 0:
            raise RuntimeError(
                "All patch paths are missing after auto root fallback. "
                f"Tried root_dir={root_dir} and alt_root={alt_root}. "
                "Pass the correct --root explicitly."
            )
    map_df = align_map_ids_to_obs(adata, map_df, id_col="spot_id")
    map_df_aligned, coords_np = reorder_and_prune_by_obs(
        adata=adata, map_df=map_df, id_col="spot_id", coord_key=args.coord_key
    )
    if len(map_df_aligned) == 0:
        obs_head = adata.obs_names.astype(str).tolist()[:5]
        map_head = map_df["spot_id"].astype(str).tolist()[:5]
        raise RuntimeError(
            "No overlapping spot_ids between CSV and h5ad after pruning. "
            f"obs_names head={obs_head}, csv spot_id head={map_head}. "
            "If CSV id column is not 'cell_id', ensure your CSV uses the right id column."
        )

    if args.use_csv_coords:
        cx, cy = args.csv_coord_cols.split(",")
        coords_np = extract_coords_np_in_order(
            adata=adata,
            map_df=map_df_aligned,
            coord_key=args.coord_key,
            use_csv_coords=True,
            csv_coord_cols=(cx, cy),
            csv_to_unit=args.csv_to_unit,
        )

    if args.coord_norm == "standardize":
        mean = coords_np.mean(axis=0, keepdims=True)
        std = coords_np.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
    else:
        mean = np.zeros((1, coords_np.shape[1]), dtype=np.float32)
        std = np.ones((1, coords_np.shape[1]), dtype=np.float32)

    coords_feat = ((coords_np - mean) / std).astype(np.float32)
    if not np.isfinite(coords_feat).all():
        raise RuntimeError("coords_feat contains NaN/Inf. Check coord normalization or input coords.")
    coords_feat_t = torch.from_numpy(coords_feat)
    id_to_idx: Dict[str, int] = {
        str(sid): i for i, sid in enumerate(map_df_aligned["spot_id"].tolist())
    }

    # 1b) Build spatial graph for GCN (use raw coords for neighbor search)
    edge_index, edge_weight = build_spatial_graph(
        coords_np, k=args.k, use_weight=True, graph_type=args.graph_type
    )
    print(f"[Graph] type={args.graph_type}, k={args.k}, E={edge_index.size(1)}")
    edge_index = edge_index.to(gcn_device, non_blocking=True).long().contiguous()
    edge_weight = (
        edge_weight.to(gcn_device, non_blocking=True).to(torch.float32).contiguous()
        if edge_weight is not None
        else None
    )
    if edge_weight is not None and not torch.isfinite(edge_weight).all().item():
        raise RuntimeError("edge_weight contains NaN/Inf. Check coords and KNN graph construction.")
    coords_feat_t = coords_feat_t.to(gcn_device, non_blocking=True)

    # 2) Foundation + transforms
    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation = FoundationCls(weights_path=None)

    # --- LoRA: HF기반 백본만 허용 (timm 백본은 스킵) ---
    lora_allowed = args.enc_name in {"plip", "phikon"}
    if args.with_lora:
        if lora_allowed:
            foundation = attach_lora_to_foundation(
                foundation,
                backbone_name=args.enc_name,
                r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, bias="none"
            )
            print(f"[LoRA] enabled on backbone='{args.enc_name}'")
        else:
            print(f"[LoRA] skipped for backbone='{args.enc_name}' (timm/open-clip backbone not supported).")

    foundation.precision = amp_dtype
    image_tf = getattr(foundation, "eval_transforms", None)

    # 3) DataLoader (img, gene, sid)
    train_loader, gene_dim = make_img_gene_dataloader(
        adata=adata, map_df=map_df_aligned,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_transform=image_tf
    )

    # 4) Models
    img_encoder = ImageFoundationAdapter(foundation, proj_dim=args.proj_dim, normalize=False)
    gene_encoder = GeneEncoder(
        input_dim=gene_dim,
        model_size_omic="small",
        proj_dim=args.proj_dim,
        normalize=False,
    )
    coord_gcn = SpatialGCN(
        in_dim=coords_np.shape[1],
        hidden_dim=args.coord_hidden_dim,
        out_dim=args.proj_dim,
        dropout=args.coord_dropout,
    )
    coord_gcn.to(gcn_device).float()
    model = XAttnContrastiveModel(
        img_encoder=img_encoder,
        gene_encoder=gene_encoder,
        proj_dim=args.proj_dim,
        num_heads=args.xattn_heads,
        dropout=args.xattn_dropout,
        temperature=args.temperature,
        lam_dir=args.lam_dir,
        img_tokens=args.img_tokens,
        gene_tokens=args.gene_tokens,
        coord_tokens=args.coord_tokens,
    )

    # 5) Train
    model.to(device).train()
    optimizer = Adam(_unique_trainable_params(model, coord_gcn), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype_found = getattr(model.img_encoder.foundation, "precision", None)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        bar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                   desc=f"[xattn] Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in bar:
            if len(batch) == 3:
                img, gene, sid = batch
            else:
                img, gene = batch
                sid = None
            img = img.to(device, non_blocking=True)
            gene = gene.to(device, non_blocking=True)

            if sid is None:
                raise RuntimeError("sid not found in dataloader batch. Check dataset.")
            batch_ids = torch.as_tensor(
                [id_to_idx[str(s).strip().replace(".0", "")] for s in sid],
                dtype=torch.long,
                device=gcn_device,
            )
            sub_ei, sub_ew = subgraph(
                subset=batch_ids, edge_index=edge_index, edge_attr=edge_weight, relabel_nodes=True
            )
            sub_ei = sub_ei.long().contiguous()
            sub_ew = sub_ew.to(torch.float32).contiguous() if sub_ew is not None else None
            coord_feat_batch = coords_feat_t[batch_ids]
            dump_attn_this_batch = bool(
                args.save_attn
                and attn_out_dir is not None
                and args.attn_max_batches > 0
                and (epoch % max(args.attn_every, 1) == 0)
                and (step <= args.attn_max_batches)
            )

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                if amp_dtype_found in (torch.bfloat16, torch.float16):
                    ctx = torch.amp.autocast("cuda", dtype=amp_dtype_found)
                else:
                    ctx = torch.amp.autocast("cuda")
                # GCN은 FP32로 강제
                with torch.amp.autocast("cuda", enabled=False):
                    zc = coord_gcn(coord_feat_batch.float(), sub_ei, sub_ew)
                zc = zc.to(device, non_blocking=True)
                with ctx:
                    zi = model.img_encoder(img)
                    zg = model.gene_encoder(gene)
                    zc = zc.to(dtype=zi.dtype)
                    if not _is_finite(zi, zg, zc):
                        print("[WARN] NaN/Inf detected in embeddings (AMP). Skipping batch.")
                        continue
                    if dump_attn_this_batch:
                        zi2g, zg2i, zi2c, zc2i, attn_map = model.cross_attend(zi, zg, zc, return_attn=True)
                    else:
                        zi2g, zg2i, zi2c, zc2i = model.cross_attend(zi, zg, zc, return_attn=False)
                        attn_map = None
                    if not _is_finite(zi2g, zg2i, zi2c, zc2i):
                        print("[WARN] NaN/Inf detected after cross-attention (AMP). Skipping batch.")
                        continue
                    loss_ig = model.contrastive_loss(zi2g, zg2i)
                    loss_ic = model.contrastive_loss(zi2c, zc2i)
                    loss = args.lam_ig * loss_ig + args.lam_ic * loss_ic
                    if not _is_finite(loss):
                        print("[WARN] NaN/Inf loss detected (AMP). Skipping batch.")
                        continue
                scaler.scale(loss).backward()
                if args.max_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                zc = coord_gcn(coord_feat_batch.float(), sub_ei, sub_ew)
                zc = zc.to(device, non_blocking=True)
                zi = model.img_encoder(img)
                zg = model.gene_encoder(gene)
                if not _is_finite(zi, zg, zc):
                    print("[WARN] NaN/Inf detected in embeddings. Skipping batch.")
                    continue
                if dump_attn_this_batch:
                    zi2g, zg2i, zi2c, zc2i, attn_map = model.cross_attend(zi, zg, zc, return_attn=True)
                else:
                    zi2g, zg2i, zi2c, zc2i = model.cross_attend(zi, zg, zc, return_attn=False)
                    attn_map = None
                if not _is_finite(zi2g, zg2i, zi2c, zc2i):
                    print("[WARN] NaN/Inf detected after cross-attention. Skipping batch.")
                    continue
                loss_ig = model.contrastive_loss(zi2g, zg2i)
                loss_ic = model.contrastive_loss(zi2c, zc2i)
                loss = args.lam_ig * loss_ig + args.lam_ic * loss_ic
                if not _is_finite(loss):
                    print("[WARN] NaN/Inf loss detected. Skipping batch.")
                    continue
                loss.backward()
                if args.max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()

            if dump_attn_this_batch and attn_map is not None:
                out_npz = os.path.join(attn_out_dir, f"ep{epoch:03d}_step{step:04d}.npz")
                _save_attn_npz(
                    out_npz,
                    attn_map,
                    reduce_heads=args.attn_reduce_heads,
                    meta={
                        "node_idx": batch_ids.detach().cpu().numpy(),
                        "spot_id": np.asarray([str(s) for s in sid]),
                    },
                )

            running += loss.item()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running/50:.4f}")
                running = 0.0

        _save_all(args.save_dir, img_encoder, gene_encoder, coord_gcn, model)

    print(f"Training done. ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
