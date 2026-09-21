# run_contrastive_xattn.py
# Stage 1: multi-branch contrastive pretraining (manuscript Eqs. 1-6).
#
# NOTE ON THE FILENAME: this script no longer trains any cross-attention. Per
# Fig. 1 and Eqs. (4)-(6) the pretraining objective is plain InfoNCE between the
# ENCODER OUTPUTS; cross-attention is the finetuning fusion of Eq. (8). The
# "_xattn" suffix is kept only so existing commands keep resolving.
#
# Spatial branch:
#   - graph edges E     : Delaunay triangulation over the cell centroids  (Eq. 1)
#   - node features H(0): the histology embedding of each cell            (Eq. 2)
#   Coordinates build E only; they are never GCN node features.
#
# Checkpoints written to --save_dir:
#   Image_pretrained.pt    image encoder + its projection head
#   SNN_pretrained.pt      gene encoder
#   Spatial_pretrained.pt  spatial GCN (in_dim == --proj_dim)
#
# Example:
'''
CUDA_VISIBLE_DEVICES=0 \
/data/project/banana9903_xenium/.conda-envs/SPAM/bin/python Code_final/code/utils/run_contrastive_xattn.py \
--h5ad /data/project/banana9903_xenium/h5ads/human_reactive_follicular.h5ad \
--csv /data/project/banana9903_xenium/contrast_test/expression/human_reactive_follicular_patches_256/spot_patch_map_256.csv \
--enc_name uni_v1 \
--proj_dim 256 \
--epochs 100 --batch_size 256 \
--save_dir /data/project/banana9903_xenium/ckpts_human_reactive_follicular_pretrain \
--device cuda:0 \
--spot_hidden_dim 256 \
--graph_type delaunay --k 12

# Token counts (--img_tokens / --gene_tokens / --spot_tokens) belong to the
# finetuning fusion, not here; passing them prints a warning and does nothing.
'''
import contextlib
import os
import sys
import time
import argparse
from typing import Optional, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # /Code_final
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 표준모듈 'code'와 충돌 방지: 이미 로드돼 있으면 제거
sys.modules.pop("code", None)

from code.models.Foundations import inf_encoder_factory
from code.models.GCN_update import SpatialGCN, precompute_gcn_norm
from code.models.graph_construction import build_knn_graph
from code.models.gene_encoder import GeneEncoder
from code.utils.lora_utils import attach_lora_to_foundation
from code.models.spot_features import build_histology_node_features, freeze_and_lock_eval
from code.utils.contrastive import ImageFoundationAdapter, build_spatial_graph
from code.utils.dataset_utils import (
    SpotImageGeneDataset,
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_gene_dataloader,
    make_img_only_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
    extract_coords_np_in_order,
)
from torch_geometric.utils import k_hop_subgraph, subgraph


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
class MultiBranchContrastiveModel(nn.Module):
    """
    Multi-branch contrastive pretraining, manuscript Eqs. (4)-(6).

    The image embedding is the anchor and is aligned against the gene embedding
    and the spatial(GCN) embedding by two InfoNCE branches computed on the
    ENCODER OUTPUTS:

        L_img-gene    = InfoNCE(z_i, z_g)                              (4)
        L_img-spatial = InfoNCE(z_i, z_s)                              (5)
        L             = lam_ig * L_img-gene + lam_is * L_img-spatial   (6)

    Cross-attention belongs to the finetuning fusion (Eq. 8), not here. The
    earlier XAttnContrastiveModel put the cross-attention *before* the loss,
    which made the objective degenerate: CrossAttention adds a residual from its
    query (attention.py: `norm(x + q)`), so both sides of the InfoNCE carried the
    same sample's image AND gene. On synthetic data with zero mutual information
    between the two modalities and fresh batches every step — where nothing is
    learnable or memorizable — that loss still trained from log(64)=4.16 down to
    0.13, i.e. it was solvable by a shortcut and measured no alignment at all.
    """

    def __init__(
        self,
        img_encoder: nn.Module,
        gene_encoder: nn.Module,
        proj_dim: int,
        temperature: float = 0.15,
        lam_dir: float = 0.5,
    ):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.temperature = float(temperature)
        self.lam_dir = float(lam_dir)
        self.proj_dim = int(proj_dim)

    @staticmethod
    def normalize(*zs: torch.Tensor):
        """Cosine-similarity space: sim(.,.) in Eqs. (4)-(5)."""
        out = tuple(F.normalize(z, dim=1) for z in zs)
        return out[0] if len(out) == 1 else out

    def contrastive_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE over in-batch negatives. lam_dir=1.0 is the image-anchored form
        printed as Eqs. (4)-(5); 0.5 is the symmetric CLIP form.

        The similarity matrix and its softmax are computed in FP32. It is only
        (B, B), so the cost is nil, and dividing cosine similarities by
        tau=0.15 before a softmax is exactly where half precision loses
        resolution. It also sidesteps an fp16 cublasGemmEx failure seen on
        Blackwell for this matmul (attention.py carries the same workaround).
        """
        with torch.amp.autocast("cuda", enabled=False):
            a = a.float()
            b = b.float()
            logits = (a @ b.t()) / self.temperature
            targets = torch.arange(logits.size(0), device=logits.device)
            loss_ab = F.cross_entropy(logits, targets)
            if self.lam_dir >= 1.0:
                return loss_ab
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
    spot_gcn: nn.Module,
):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(img_encoder.state_dict(), os.path.join(save_dir, "Image_pretrained.pt"))
    torch.save(gene_encoder.state_dict(), os.path.join(save_dir, "SNN_pretrained.pt"))
    # Renamed from Coord_pretrained.pt: the GCN now consumes histology features,
    # not coordinates, so the old file is not loadable into the new architecture.
    torch.save(spot_gcn.state_dict(), os.path.join(save_dir, "Spatial_pretrained.pt"))
    # No XATTN_pretrained.pt: pretraining has no cross-attention (Eqs. 4-6).
    # The finetuning fusion (Eq. 8) is trained from scratch in stage 2.


def gcn_receptive_field(batch_ids: torch.Tensor,
                        edge_index: torch.Tensor,
                        edge_weight: Optional[torch.Tensor],
                        n_nodes: int,
                        num_hops: int):
    """
    Exact receptive field of `batch_ids` under a `num_hops`-layer GCN.

    A random mini-batch of a few hundred nodes drawn from a slide-scale graph has
    an almost empty induced subgraph (expected intra-batch edges ~ E*(B/N)^2), so
    running the GCN on `subgraph(batch_ids, ...)` degenerates to the self-loops
    and the spatial branch learns nothing spatial. Instead we take every node
    within `num_hops` of the batch and run the GCN on the induced subgraph of
    that set — the outputs at the batch rows are then bit-identical to a
    full-graph forward pass.

    Returns:
        subset:  (S,) node ids in the receptive field
        sub_ei:  (2, E') edge index relabelled to [0, S)
        sub_ew:  (E',) edge weights or None
        mapping: (B,) row of each batch node inside `subset`
    """
    subset, _, _, _ = k_hop_subgraph(
        batch_ids, num_hops, edge_index, relabel_nodes=False, num_nodes=n_nodes
    )
    sub_ei, sub_ew = subgraph(
        subset, edge_index, edge_attr=edge_weight,
        relabel_nodes=True, num_nodes=n_nodes,
    )
    pos = torch.full((n_nodes,), -1, dtype=torch.long, device=subset.device)
    pos[subset] = torch.arange(subset.numel(), device=subset.device)
    mapping = pos[batch_ids]
    if int((mapping < 0).sum()) > 0:
        raise RuntimeError("batch node missing from its own receptive field")
    sub_ei = sub_ei.long().contiguous()
    sub_ew = sub_ew.to(torch.float32).contiguous() if sub_ew is not None else None
    return subset, sub_ei, sub_ew, mapping

def _is_finite(*tensors: torch.Tensor) -> bool:
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all().item():
            return False
    return True


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
    p.add_argument("--coord_norm", default="standardize", choices=["standardize", "none"],
                   help="DEPRECATED / no effect. Coordinates are only used to build "
                        "the graph edges; GCN node features are histology embeddings.")

    # model / foundation
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    p.add_argument("--enc_weights_path", default=None,
                   help="Local foundation weights dir (containing pytorch_model.bin). "
                        "Unset falls back to $SPAM_UNI_WEIGHTS, then to a HF-hub "
                        "download. Nothing here silently random-inits: a failure "
                        "to obtain the pretrained weights raises.")
    p.add_argument("--proj_dim", type=int, default=256)
    _bb = p.add_mutually_exclusive_group()
    _bb.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true",
                     default=True,
                     help="Freeze the foundation backbone; train only the projection "
                          "head (default, matches Fig. 1).")
    _bb.add_argument("--train_backbone", dest="freeze_backbone", action="store_false",
                     help="Optimize the whole backbone as well (previous behaviour).")
    p.add_argument("--with_lora", action="store_true",
                   help="HF 기반 백본(plip/phikon)에서만 활성화. timm 백본은 자동으로 스킵됨.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # graph / spatial gcn
    p.add_argument("--k", type=int, default=12,
                   help="k for the KNN graph (also the Delaunay fallback)")
    p.add_argument("--graph_type", default="delaunay", choices=["knn", "delaunay"],
                   help="Spatial graph over the spot centroids. The manuscript uses "
                        "Delaunay triangulation; 'knn' is kept for ablation.")
    p.add_argument("--gcn_hops", type=int, default=2,
                   help="Receptive field per mini-batch. Must equal the GCN depth "
                        "(SpatialGCN has 2 layers) for exact full-graph outputs.")
    p.add_argument("--gcn_device", default=None,
                   help='GCN device override: "cpu", "cuda", "cuda:0" ... (default: use --device)')

    # spatial GCN
    p.add_argument("--spot_hidden_dim", "--coord_hidden_dim", dest="spot_hidden_dim",
                   type=int, default=256,
                   help="hidden width of the spatial GCN. Must match the finetuning "
                        "script's --spot_hidden_dim or the checkpoint will not load.")
    p.add_argument("--spot_dropout", "--coord_dropout", dest="spot_dropout",
                   type=float, default=0.1)

    # Deprecated: pretraining has no cross-attention (Eqs. 4-6). Accepted so
    # existing command lines keep running; ignored with a warning. Token counts
    # and attention dumps belong to the finetuning fusion (Eq. 8).
    for _dead in ["--xattn_heads", "--xattn_dropout", "--img_tokens",
                  "--gene_tokens", "--spot_tokens", "--coord_tokens",
                  "--attn_out_dir", "--attn_max_batches", "--attn_every"]:
        p.add_argument(_dead, dest=_dead.lstrip("-").replace("-", "_"),
                       default=None, help=argparse.SUPPRESS)
    for _dead in ["--save_attn", "--attn_reduce_heads"]:
        p.add_argument(_dead, dest=_dead.lstrip("-").replace("-", "_"),
                       action="store_true", help=argparse.SUPPRESS)

    # training
    p.add_argument("--epochs", type=int, default=100)      # Supp.: 100 for pretraining
    p.add_argument("--batch_size", type=int, default=128)  # Supp.: 128
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--lam_dir", type=float, default=0.5,
                   help="InfoNCE 방향성. 1.0 = 논문 Eq.(4)-(5) 그대로 image-anchored, "
                        "0.5 = 대칭 CLIP 형태(기본).")
    p.add_argument("--lam_ig", type=float, default=1.0,
                   help="lambda_g in Eq. (6): image<->gene branch weight")
    p.add_argument("--lam_is", "--lam_ic", dest="lam_is", type=float, default=1.0,
                   help="image↔spatial(GCN) loss weight")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)  # Supp.: 0.05
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

    _ignored = [f for f in ["xattn_heads", "xattn_dropout", "img_tokens", "gene_tokens",
                            "spot_tokens", "coord_tokens", "save_attn", "attn_out_dir",
                            "attn_max_batches", "attn_every", "attn_reduce_heads"]
                if getattr(args, f, None)]
    if _ignored:
        print(f"[WARN] ignoring cross-attention flags {_ignored}: pretraining is "
              "plain InfoNCE on the encoder outputs (Eqs. 4-6). Token counts and "
              "attention dumps only apply to the finetuning fusion (Eq. 8).")

    _dir = "image-anchored (Eq. 4-5)" if args.lam_dir >= 1.0 else "symmetric CLIP"
    print(f"[Loss] L = {args.lam_ig}*InfoNCE(z_img, z_gene) + "
          f"{args.lam_is}*InfoNCE(z_img, z_spatial), tau={args.temperature}, {_dir}")

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

    if not np.isfinite(coords_np).all():
        raise RuntimeError("coords contain NaN/Inf. Check adata.obsm[coord_key].")
    n_nodes = int(coords_np.shape[0])
    id_to_idx: Dict[str, int] = {
        str(sid): i for i, sid in enumerate(map_df_aligned["spot_id"].tolist())
    }

    # 1b) Spatial graph. Coordinates define the EDGES only — the GCN node
    #     features H(0) are the histology embeddings built further below.
    edge_index, edge_weight = build_spatial_graph(
        coords_np, k=args.k, use_weight=True, graph_type=args.graph_type
    )
    print(f"[Graph] type={args.graph_type}, k={args.k}, N={n_nodes}, E={edge_index.size(1)}")
    edge_index = edge_index.to(gcn_device, non_blocking=True).long().contiguous()
    edge_weight = (
        edge_weight.to(gcn_device, non_blocking=True).to(torch.float32).contiguous()
        if edge_weight is not None
        else None
    )
    if edge_weight is not None and not torch.isfinite(edge_weight).all().item():
        raise RuntimeError("edge_weight contains NaN/Inf. Check coords and graph construction.")

    # Normalize once on the full graph so that mini-batch subgraphs reproduce the
    # full-graph forward pass exactly (see precompute_gcn_norm).
    edge_index, edge_weight = precompute_gcn_norm(edge_index, edge_weight, n_nodes)
    edge_index = edge_index.long().contiguous()
    edge_weight = edge_weight.to(torch.float32).contiguous()

    # 2) Foundation + transforms
    FoundationCls = inf_encoder_factory(args.enc_name)
    foundation = FoundationCls(weights_path=args.enc_weights_path)
    print(f"[Backbone] {args.enc_name}, weights="
          f"{args.enc_weights_path or os.environ.get('SPAM_UNI_WEIGHTS') or 'hf-hub (download)'}")

    # --- LoRA: HF기반 백본만 허용 (timm 백본은 스킵) ---
    lora_allowed = args.enc_name in {"plip", "phikon"}
    if args.with_lora:
        if args.freeze_backbone:
            raise ValueError("--with_lora and --freeze_backbone are mutually exclusive")
        if lora_allowed:
            foundation = attach_lora_to_foundation(
                foundation,
                backbone_name=args.enc_name,
                r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, bias="none"
            )
            print(f"[LoRA] enabled on backbone='{args.enc_name}'")
        else:
            print(f"[LoRA] skipped for backbone='{args.enc_name}' (timm/open-clip backbone not supported).")

    # Fig. 1 marks the pretraining Image Encoder frozen (❄️) with only the
    # Projection head trainable (🔥). Left unfrozen, the full ~303M ViT-L is
    # optimized by the contrastive objective on a single slide's cells, which
    # both drifts the foundation representation off-distribution and is what the
    # figure does not depict.
    if args.freeze_backbone:
        n_frozen = freeze_and_lock_eval(foundation.encoder)
        print(f"[Freeze] image backbone frozen: {n_frozen/1e6:.1f}M params "
              "(only the projection head trains, per Fig. 1)")
    else:
        n_bb = sum(p_bb.numel() for p_bb in foundation.encoder.parameters())
        print(f"[Freeze][warn] --train_backbone: optimizing {n_bb/1e6:.1f}M backbone "
              f"params at lr={args.lr}. Fig. 1 shows this encoder frozen.")

    foundation.precision = amp_dtype
    img_dim = foundation.get_output_dim()
    image_tf = getattr(foundation, "eval_transforms", None)

    # 3) DataLoader (img, gene, sid)
    train_loader, gene_dim = make_img_gene_dataloader(
        adata=adata, map_df=map_df_aligned,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_transform=image_tf
    )

    # Dense gene matrix in node order, so the frozen-backbone path can drop the
    # image pipeline entirely and still serve the gene branch.
    gene_ds = SpotImageGeneDataset(
        adata=adata, map_df=map_df_aligned, gene_names=None, image_transform=image_tf,
    )
    _gx = adata[map_df_aligned["spot_id"].astype(str).tolist()].X[:, gene_ds._gene_index]
    if hasattr(_gx, "toarray"):
        _gx = _gx.toarray()
    gene_matrix = torch.from_numpy(np.asarray(_gx, dtype=np.float32))
    assert gene_matrix.shape == (n_nodes, gene_dim), (gene_matrix.shape, n_nodes, gene_dim)

    # 4) Models
    img_encoder = ImageFoundationAdapter(foundation, proj_dim=args.proj_dim, normalize=False)
    gene_encoder = GeneEncoder(
        input_dim=gene_dim,
        model_size_omic="small",
        proj_dim=args.proj_dim,
        normalize=False,
    )
    # H(0) = histology embedding (post-projection), so in_dim == proj_dim.
    spot_gcn = SpatialGCN(
        in_dim=args.proj_dim,
        hidden_dim=args.spot_hidden_dim,
        out_dim=args.proj_dim,
        dropout=args.spot_dropout,
        normalize=False,   # edge weights were pre-normalized on the full graph
    )
    spot_gcn.to(gcn_device).float()
    model = MultiBranchContrastiveModel(
        img_encoder=img_encoder,
        gene_encoder=gene_encoder,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        lam_dir=args.lam_dir,
    )

    # 5) Train
    model.to(device).train()
    # Manuscript (Supp.): AdamW, lr 1e-5, weight decay 0.05.
    optimizer = AdamW(_unique_trainable_params(model, spot_gcn),
                      lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype_found = getattr(model.img_encoder.foundation, "precision", None)

    # 5b) H(0) source for the spatial GCN.
    #
    # A mini-batch carries live embeddings only for its own nodes, but a 2-layer
    # GCN over a batch node also needs its 2-hop neighbours' features.
    ordered_img_loader = make_img_only_dataloader(
        map_df=map_df_aligned,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_transform=image_tf,
        shuffle=False,
    )

    if args.freeze_backbone:
        # The backbone is constant, so its output can be cached once and the
        # image pipeline dropped from the training loop entirely: H(0) for any
        # node set is just proj(raw_feat[nodes]), computed live from the current
        # (trainable) projection. Exact — no stale feature bank — and it removes
        # both the ViT forward and the PNG decode from every step.
        raw_feat = build_histology_node_features(
            img_encoder=foundation,
            loader=ordered_img_loader,
            n_nodes=n_nodes,
            out_dim=img_dim,
            device=device,
            amp_dtype=amp_dtype if use_amp else None,
        ).to(gcn_device)
        feat_bank = None
        gene_matrix_g = gene_matrix.to(device)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.arange(n_nodes, dtype=torch.long)),
            batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False,
        )
        print(f"[Cache] frozen backbone -> cached {tuple(raw_feat.shape)} raw features "
              f"({raw_feat.numel()*4/1e9:.2f} GB); training loop no longer reads images")
    else:
        # Trainable backbone: neighbours come from a write-through bank seeded by
        # one no-grad pass, with the batch rows substituted live so gradients
        # still reach the image encoder.
        raw_feat = None
        gene_matrix_g = None
        feat_bank = build_histology_node_features(
            img_encoder=model.img_encoder,
            loader=ordered_img_loader,
            n_nodes=n_nodes,
            out_dim=args.proj_dim,
            device=device,
            amp_dtype=amp_dtype if use_amp else None,
        ).to(gcn_device)
    model.train()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        bar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                   desc=f"[xattn] Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in bar:
            if raw_feat is not None:
                # Frozen-backbone path: batch is just node indices.
                batch_ids = batch[0].to(gcn_device, non_blocking=True)
                gene = gene_matrix_g[batch_ids.to(device)]
                img = None
            else:
                if len(batch) == 3:
                    img, gene, sid = batch
                else:
                    raise RuntimeError("sid not found in dataloader batch. Check dataset.")
                img = img.to(device, non_blocking=True)
                gene = gene.to(device, non_blocking=True)
                batch_ids = torch.as_tensor(
                    [id_to_idx[str(s).strip().replace(".0", "")] for s in sid],
                    dtype=torch.long,
                    device=gcn_device,
                )
            subset, sub_ei, sub_ew, mapping = gcn_receptive_field(
                batch_ids, edge_index, edge_weight, n_nodes, args.gcn_hops
            )
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                if amp_dtype_found in (torch.bfloat16, torch.float16):
                    ctx = torch.amp.autocast("cuda", dtype=amp_dtype_found)
                else:
                    ctx = torch.amp.autocast("cuda")
            else:
                ctx = contextlib.nullcontext()

            # GCN은 FP32로 강제.
            if raw_feat is not None:
                # H(0) for the whole receptive field straight from the cache,
                # through the live projection head. Exact, no staleness.
                with torch.amp.autocast("cuda", enabled=False):
                    h0 = model.img_encoder.proj(raw_feat[subset].float())
                    zs_sub = spot_gcn(h0, sub_ei, sub_ew)
                zi = h0[mapping].to(device, non_blocking=True)
                with ctx:
                    zg = model.gene_encoder(gene)
                zs = zs_sub[mapping].to(device, non_blocking=True).to(dtype=zi.dtype)
            else:
                with ctx:
                    zi = model.img_encoder(img)
                    zg = model.gene_encoder(gene)
                # H(0): 이웃은 bank, 배치 노드는 live zi.
                with torch.amp.autocast("cuda", enabled=False):
                    h0 = feat_bank[subset].index_copy(0, mapping, zi.float().to(gcn_device))
                    zs_sub = spot_gcn(h0, sub_ei, sub_ew)
                zs = zs_sub[mapping].to(device, non_blocking=True).to(dtype=zi.dtype)

            if not _is_finite(zi, zg, zs):
                print("[WARN] NaN/Inf detected in embeddings. Skipping batch.")
                continue

            with ctx:
                # Eqs. (4)-(6): InfoNCE directly on the encoder outputs.
                zi_n, zg_n, zs_n = model.normalize(zi, zg, zs)
                loss_ig = model.contrastive_loss(zi_n, zg_n)
                loss_is = model.contrastive_loss(zi_n, zs_n)
                loss = args.lam_ig * loss_ig + args.lam_is * loss_is
            if not _is_finite(loss):
                print("[WARN] NaN/Inf loss detected. Skipping batch.")
                continue

            if use_amp:
                scaler.scale(loss).backward()
                if args.max_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()

            if feat_bank is not None:
                # Write-through: the bank must track the encoder as it drifts.
                feat_bank[batch_ids] = zi.detach().float().to(gcn_device)

            running += loss.item()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running/50:.4f}")
                running = 0.0

        _save_all(args.save_dir, img_encoder, gene_encoder, spot_gcn)

    print(f"Training done. ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
