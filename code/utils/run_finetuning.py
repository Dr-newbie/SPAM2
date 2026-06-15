import os, sys, time
import numpy as np
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


# ============================================
# 모델 래퍼
# ============================================
class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn,
                 dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1,
                 merge="gated-sum",
                 decoder_hidden=1024, gene_dim=1000):
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
        )
        self.decoder = ZINBDecoder(
            in_dim=dim_fuse,
            gene_dim=gene_dim,
            hidden=decoder_hidden,
            dropout=dropout,
        )

    def forward(self, batch, spot_feat_in, edge_index, edge_weight=None):
        """
        batch: dict
          - image: (B, C, H, W)
          - expr : (B, G)
          - idx  : (B,) long  — 그래프 노드 인덱스(=spot index in current order)
        spot_feat_in: (N, F_in)  — 모든 노드에 대한 입력 feature (ex. coords)
        """
        x_img = batch["image"]
        x_gene = batch["expr"].float()
        node_idx = batch["idx"].long()   # (B,)

        # 1) enc
        zi = self.img_encoder(x_img)                    # (B, Di)
        zg = self.gene_encoder(x_gene)                  # (B, Dg)

        # 1') 전체 노드 feature에 대해 GCN 수행 → (N, Ds)
        zs_all_nodes = self.gcn(spot_feat_in, edge_index, edge_weight)  # (N, Ds)
        zs = zs_all_nodes[node_idx]                                       # (B, Ds)

        # 2) same dim
        zi = self.proj_img(zi)
        zg = self.proj_gene(zg)
        zs = self.proj_spot(zs)

        # 3) cross-attn fusion
        z_fused, z_is, z_ig = self.fusion(zi, zs, zg)

        # 4) decode (ZINB params)
        mu, theta, pi = self.decoder(z_fused)
        return mu, theta, pi, {"z_fused": z_fused, "z_is": z_is, "z_ig": z_ig}


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


# ============================================
# 학습 / 평가 루프
# ============================================
def train_epoch(model, loader, spot_feat_in, edge_index, edge_weight,
                optimizer, amp_dtype=torch.float32,
                device=torch.device("cpu"),
                epoch: int = 1, total_epochs: int = 1):
    """
    한 epoch 학습.
    tqdm으로 배치별 loss를 progress bar에 띄워서 모니터링.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and amp_dtype != torch.float32)
    )
    loss_sum, n = 0.0, 0

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
            mu, theta, pi, _ = model(batch, spot_feat_in, edge_index, edge_weight)
            x = batch["expr"].float()
            loss = zinb_nll(x, mu, theta, pi, reduction="mean")

        if scaler.is_enabled():
            scaler.scale(loss).step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        loss_sum += loss.item() * bs
        n += bs

        # ✅ 진행바에 현재 배치 loss 표시
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, spot_feat_in, edge_index, edge_weight,
               amp_dtype=torch.float32,
               device=torch.device("cpu"),
               epoch: int = 1, total_epochs: int = 1):
    """
    한 epoch 평가.
    tqdm으로 진행만 보여주고, 최종 metric은 epoch 끝날 때 반환.
    """
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    preds, gts = [], []

    pbar = tqdm(loader, desc=f"[Eval ] Epoch {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        batch = _to_device_batch(batch, device)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=(device.type == "cuda" and amp_dtype != torch.float32),
        ):
            mu, theta, pi, _ = model(batch, spot_feat_in, edge_index, edge_weight)
            x = batch["expr"].float()
            yhat = mu

        preds.append(yhat.detach().cpu())
        gts.append(x.detach().cpu())
        mse_sum += F.mse_loss(yhat, x, reduction="sum").item()
        mae_sum += F.l1_loss(yhat, x, reduction="sum").item()
        n += x.numel()

    P = torch.cat(preds, 0).numpy()
    Y = torch.cat(gts, 0).numpy()
    Yc, Pc = Y - Y.mean(0, keepdims=True), P - P.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0)) + 1e-8
    pcc = np.where(den > 0, num / den, np.nan)

    return {
        "mse": mse_sum / max(n, 1),
        "mae": mae_sum / max(n, 1),
        "pcc_mean": float(np.nanmean(pcc)),
        "pcc_median": float(np.nanmedian(pcc)),
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

    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--id_col", default="spot_id")
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
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--merge", default="gated-sum",
                   choices=["gated-sum", "concat-proj"])
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--graph_type", default="knn", choices=["knn", "delaunay"],
                   help="Spatial graph type for fine-tuning.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_dir", default="./ckpts_cross")
    p.add_argument("--inside_he_only", action="store_true",
                   help="CSV에서 inside_he=True인 패치만 사용")
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

    # ---------------- data ----------------
    adata = load_h5ad_with_preproc(
        args.h5ad,
        use_hvg=False,
        n_top_genes=None,
        x_key="x_centroid",
        y_key="y_centroid",
    )
    map_df = load_patch_mapping_csv_xenium(args.csv, root_dir=args.root, drop_missing=True)
    if args.inside_he_only:
        if "inside_he" not in map_df.columns:
            print("[WARN] inside_he_only requested but 'inside_he' column not found in CSV.")
        else:
            keep = map_df["inside_he"].astype(str).str.lower().isin(["true", "1", "yes"])
            map_df = map_df.loc[keep].reset_index(drop=True)
    map_df = align_map_ids_to_obs(adata, map_df, id_col=args.id_col)
    map_df_aligned, coords_np = reorder_and_prune_by_obs(
        adata, map_df, id_col=args.id_col, coord_key="spatial"
    )
    if map_df_aligned.shape[0] != adata.n_obs:
        obs_ids = map_df_aligned[args.id_col].astype(str).tolist()
        adata = adata[obs_ids].copy()

    # --- 문자열 ID → 정수 인덱스 매핑 생성 ---
    global ID2IDX
    id_values = map_df_aligned[args.id_col].astype(str).tolist()
    ID2IDX = {v: i for i, v in enumerate(id_values)}
    print(f"[ID2IDX] built mapping for {len(ID2IDX)} nodes based on '{args.id_col}'")

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
        decoder_hidden=1024,
        gene_dim=gene_dim,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best = {"pcc": -1.0, "path": None}
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(
            model, loader, spot_feat_in, edge_index, edge_weight,
            optim, amp_dtype, device,
            epoch=ep, total_epochs=args.epochs
        )
        ev, P, Y = eval_epoch(
            model, loader, spot_feat_in, edge_index, edge_weight,
            amp_dtype, device,
            epoch=ep, total_epochs=args.epochs
        )
        dt = time.time() - t0
        print(
            f"[{ep:03d}] trainZINB={tr:.6f} | "
            f"eval MSE={ev['mse']:.6e} MAE={ev['mae']:.6e} "
            f"PCC(mean)={ev['pcc_mean']:.4f} med={ev['pcc_median']:.4f}  ({dt:.1f}s)"
        )

        if ev["pcc_mean"] > best["pcc"]:
            best["pcc"] = ev["pcc_mean"]
            out_dir = os.path.join(args.save_dir, "best")
            os.makedirs(out_dir, exist_ok=True)

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
            best["path"] = out_dir
            print(f"[BEST↑] saved to {out_dir}")

    print(f"Done. best PCC(mean)={best['pcc']:.4f}, path={best['path']}")


if __name__ == "__main__":
    main()
