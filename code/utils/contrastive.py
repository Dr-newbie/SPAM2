# contrastive.py
# -*- coding: utf-8 -*-

import os
import sys
import time
from typing import Optional, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# --- project imports (run_contrastive에서 sys.path 처리 가정) ---
from code.models.gene_encoder import GeneEncoder
from code.models.Foundations import inf_encoder_factory
from code.models.graph_construction import build_knn_graph
from code.models.graph_construction_delaunay import build_delaunay_graph
from code.models.GCN_update import SpatialGCN
from code.utils.lora_utils import attach_lora_to_foundation
from code.utils.wandb_utils import wandb_log, wandb_summary

# PyG: 서브그래프 유틸 (미니배치용)
from torch_geometric.utils import subgraph


def build_spatial_graph(
    coords,
    k: int = 12,
    use_weight: bool = True,
    graph_type: str = "knn",
):
    """
    Spatial graph builder wrapper.
    graph_type:
      - "knn": 기존 KNN 그래프
      - "delaunay": Delaunay triangulation 기반 그래프 (실패 시 내부 KNN fallback)
    """
    graph_key = str(graph_type).strip().lower()
    if graph_key == "knn":
        return build_knn_graph(coords, k=k, use_weight=use_weight)
    if graph_key == "delaunay":
        return build_delaunay_graph(coords, k=k, use_weight=use_weight)
    raise ValueError(f"Unknown graph_type='{graph_type}'. Use one of ['knn', 'delaunay'].")


# ----------------------------
# Foundation adapter
# ----------------------------
class ImageFoundationAdapter(nn.Module):
    def __init__(self, foundation, proj_dim: int, normalize: bool = False):
        super().__init__()
        self.foundation = foundation
        self.normalize = normalize  # 기본 False: 모델 encode에서 한 번만 정규화
        self.proj_dim = int(proj_dim)
        out_dim = self.foundation.get_output_dim()
        self.proj = nn.Linear(out_dim, proj_dim) if out_dim != proj_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp16/bf16 등 precision 캐스팅 (GPU+AMP 권장)
        dtype = getattr(self.foundation, "precision", torch.float32)
        x = x.to(dtype=dtype)
        z = self.foundation.forward_features(x)  # (B, out_dim)
        z = self.proj(z)                         # (B, proj_dim)
        if self.normalize:
            z = F.normalize(z, dim=1)
        return z


# ----------------------------
# img ↔ gene contrastive
# ----------------------------
class img2gene_ContrastiveModel(nn.Module):
    def __init__(self, img_encoder: nn.Module, gene_encoder: nn.Module,
                 proj_dim: int, temperature: float = 0.15, lam: float = 0.5, gcn_force_fp32: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.lam = float(lam)
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.gcn_force_fp32 = gcn_force_fp32

    def encode(self, img, gene):
        img_z = self.img_encoder(img)     # (B, D)
        gene_z = self.gene_encoder(gene)  # (B, D)
        img_z = F.normalize(img_z, dim=1)
        gene_z = F.normalize(gene_z, dim=1)
        return img_z, gene_z

    def forward(self, img, gene):
        return self.encode(img, gene)

    def contrastive_loss(self, img_z, gene_z):
        logits = (img_z @ gene_z.t()) / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i2g = F.cross_entropy(logits, targets)
        loss_g2i = F.cross_entropy(logits.t(), targets)
        return self.lam * loss_i2g + (1.0 - self.lam) * loss_g2i


# ----------------------------
# img ↔ spatial(GCN) contrastive
# ----------------------------
class img2spot_ContrastiveModel(nn.Module):
    def __init__(self, img_encoder: nn.Module, gcn: nn.Module,
                 temperature: float = 0.15, lam: float = 0.5, gcn_force_fp32: bool = False,**kwargs,):
        super().__init__()
        self.img_encoder = img_encoder
        self.gcn = gcn
        self.temperature = float(temperature)
        self.lam = float(lam)
        self.gcn_force_fp32 = bool(gcn_force_fp32)

    def encode(self, img: torch.Tensor, edge_index: torch.Tensor,
               edge_weight: Optional[torch.Tensor] = None):
        z_img = self.img_encoder(img)                              # (N, D)
        # GCN은 FP32 강제(autocast 비활성화)
        with torch.amp.autocast('cuda', enabled=False):
            zi32 = z_img.float().contiguous()
            ew32 = edge_weight.float() if edge_weight is not None else None
            z_spa = self.gcn(zi32, edge_index, ew32)               # (N, D) FP32
        z_img = F.normalize(z_img, dim=1)
        z_spa = F.normalize(z_spa, dim=1).to(z_img.dtype)          # dtype 맞추기
        return z_img, z_spa

    def forward(self, img, edge_index, edge_weight=None):
        return self.encode(img, edge_index, edge_weight)

    def contrastive_loss(self, zi, zs):
        logits = (zi @ zs.t()) / self.temperature
        targets = torch.arange(zi.size(0), device=zi.device)
        loss_i2s = F.cross_entropy(logits, targets)
        loss_s2i = F.cross_entropy(logits.t(), targets)
        return self.lam * loss_i2s + (1.0 - self.lam) * loss_s2i


# ----------------------------
# img → cluster classification
# ----------------------------
class img2cluster_Classifier(nn.Module):
    def __init__(self, img_encoder: nn.Module, num_classes: int,
                 hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.img_encoder = img_encoder
        in_dim = getattr(img_encoder, "proj_dim", None)
        if in_dim is None:
            raise AttributeError("img_encoder must define proj_dim for classification head.")
        if hidden_dim and hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        z = self.img_encoder(img)
        return self.classifier(z)


# ----------------------------
# Utils
# ----------------------------
def _unique_trainable_params(*modules: Iterable[nn.Module]):
    """여러 모델의 파라미터를 합치되, trainable만, 중복(id) 제거."""
    seen, params = set(), []
    for m in modules:
        for p in m.parameters():
            if p.requires_grad and (id(p) not in seen):
                seen.add(id(p))
                params.append(p)
    return params


def _save_all(save_dir: str, img_encoder: ImageFoundationAdapter,
              gene_encoder: Optional[nn.Module] = None,
              gcn: Optional[nn.Module] = None):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(img_encoder.state_dict(), os.path.join(save_dir, "Image_pretrained.pt"))
    if gene_encoder is not None:
        torch.save(gene_encoder.state_dict(), os.path.join(save_dir, "SNN_pretrained.pt"))
    if gcn is not None:
        torch.save(gcn.state_dict(), os.path.join(save_dir, "KNN_pretrained.pt"))

def _debug_gcn_inputs(x, edge_index, edge_weight, B, gcn_module, tag=""):
    probs = []
    def info(t):
        return f"shape={tuple(t.shape)}, dtype={t.dtype}, dev={t.device}, contig={t.is_contiguous()}, stride={t.stride()}"

    # x 검사
    if not torch.is_tensor(x):
        probs.append("x not tensor")
    else:
        if x.dtype != torch.float32: probs.append(f"x.dtype={x.dtype}")
        if not x.is_contiguous(): probs.append("x not contiguous")
        if x.dim()!=2: probs.append(f"x.dim={x.dim()}")
        if x.size(0)!=B: probs.append(f"x.B={x.size(0)} != B={B}")
        if x.numel()==0: probs.append("x is empty (numel=0)")

    # edge_index 검사
    if not torch.is_tensor(edge_index):
        probs.append("edge_index not tensor")
    else:
        if edge_index.dtype!=torch.long: probs.append(f"edge_index.dtype={edge_index.dtype}")
        if not edge_index.is_contiguous(): probs.append("edge_index not contiguous")
        if edge_index.dim()!=2 or edge_index.size(0)!=2:
            probs.append(f"edge_index shape={tuple(edge_index.shape)} (expect [2,E])")

    # edge_weight 검사
    if edge_weight is not None:
        if not torch.is_tensor(edge_weight): probs.append("edge_weight not tensor")
        else:
            if edge_weight.dtype!=torch.float32: probs.append(f"edge_weight.dtype={edge_weight.dtype}")
            if not edge_weight.is_contiguous(): probs.append("edge_weight not contiguous")
            if edge_weight.dim()!=1: probs.append(f"edge_weight.dim={edge_weight.dim()}")
            if torch.is_tensor(edge_index) and edge_index.dim()==2 and edge_index.size(1)!=edge_weight.numel():
                probs.append(f"|E| mismatch: edge_index.E={edge_index.size(1)} vs edge_weight.numel()={edge_weight.numel()}")

    # 서브그래프 relabel 검증(로컬 노드 수 == B?)
    local_nodes = int(edge_index.max().item())+1 if torch.is_tensor(edge_index) and edge_index.numel()>0 else B
    if local_nodes!=B:
        probs.append(f"local_nodes={local_nodes} != B={B} (relabel/batch_ids 문제)")

    # GCN in_channels vs x D 검증
    D = int(x.size(1)) if torch.is_tensor(x) and x.dim()==2 else None
    in_ch = None
    try:
        if hasattr(gcn_module, "conv1"):
            if hasattr(gcn_module.conv1, "in_channels"):
                in_ch = gcn_module.conv1.in_channels
            elif hasattr(gcn_module.conv1, "lin") and hasattr(gcn_module.conv1.lin, "weight"):
                in_ch = gcn_module.conv1.lin.weight.shape[1]
    except Exception as e:
        probs.append(f"read gcn in_channels failed: {e!r}")

    if D is not None and in_ch is not None and D!=in_ch:
        probs.append(f"feature dim mismatch x.D={D} vs gcn.in_channels={in_ch}")

    if probs:
        print(f"\n[GCN DEBUG] {tag}")
        print("x      :", info(x) if torch.is_tensor(x) else "NA")
        print("ei     :", info(edge_index) if torch.is_tensor(edge_index) else "NA")
        if edge_weight is not None:
            print("ew     :", info(edge_weight))
        try:
            lin = gcn_module.conv1.lin
            print("W dtype:", lin.weight.dtype, "W shape:", tuple(lin.weight.shape), "dev:", lin.weight.device)
            if lin.bias is not None:
                print("b dtype:", lin.bias.dtype, "b shape:", tuple(lin.bias.shape), "dev:", lin.bias.device)
        except Exception as e:
            print("inspect weights failed:", repr(e))
        print("Problems:", probs)
        raise RuntimeError("GCN input validation failed.")

def _warmup_linear(x, gcn_module, tag=""):
    try:
        lin = gcn_module.conv1.lin
        W = lin.weight
        b = lin.bias
        # 모두 FP32 & 동일 디바이스로 캐스팅/복사
        x_ = x.to(torch.float32).contiguous()
        W_ = W.to(torch.float32).contiguous()
        b_ = b.to(torch.float32).contiguous() if b is not None else None

        # 차원 일치 확인
        assert x_.dim()==2 and W_.dim()==2, f"x.dim={x_.dim()}, W.dim={W_.dim()}"
        assert x_.size(1)==W_.size(1), f"x.D={x_.size(1)} vs W.in={W_.size(1)}"
        assert x_.device==W_.device, f"dev mismatch x:{x_.device} W:{W_.device}"

        # 실제로 한 번 곱해봄
        y = F.linear(x_, W_, b_)
        if y.numel()==0:
            raise RuntimeError("F.linear produced empty output")
    except Exception as e:
        print(f"\n[GCN WARMUP FAILED] {tag} :: {type(e).__name__}: {e}")
        print(f"  x  : shape={tuple(x.shape)}, dtype={x.dtype}, dev={x.device}, contig={x.is_contiguous()}, stride={x.stride()}")
        try:
            print(f"  W  : shape={tuple(gcn_module.conv1.lin.weight.shape)}, dtype={gcn_module.conv1.lin.weight.dtype}, dev={gcn_module.conv1.lin.weight.device}, contig={gcn_module.conv1.lin.weight.is_contiguous()}, stride={gcn_module.conv1.lin.weight.stride()}")
            if gcn_module.conv1.lin.bias is not None:
                print(f"  b  : shape={tuple(gcn_module.conv1.lin.bias.shape)}, dtype={gcn_module.conv1.lin.bias.dtype}, dev={gcn_module.conv1.lin.bias.device}, contig={gcn_module.conv1.lin.bias.is_contiguous()}, stride={gcn_module.conv1.lin.bias.stride()}")
        except Exception as ee:
            print("  (weight inspect failed)", repr(ee))
        raise  # 여기서 멈춰서 스택을 이 지점으로 고정


# ----------------------------
# Train loops (img↔gene)
# ----------------------------
def train_contrastive(
    model: img2gene_ContrastiveModel,
    train_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_norm: float = 0.0,
    use_amp: bool = True,
    device: Optional[str] = None,
    log_every: int = 50,
    save_dir: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    optimizer = Adam(_unique_trainable_params(model), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # autocast dtype: foundation.precision (bf16/fp16) 우선
    amp_dtype = getattr(model.img_encoder.foundation, "precision", None)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        running = 0.0
        bar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                   desc=f"[img↔gene] Epoch {epoch}/{epochs}", leave=False)
        for step, batch in bar:
            if len(batch) == 3:
                img, gene, _ = batch
            else:
                img, gene = batch
            img = img.to(device, non_blocking=True)
            gene = gene.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                # dtype 자동선택: 지정 가능하면 지정
                if amp_dtype in (torch.bfloat16, torch.float16):
                    ctx = torch.amp.autocast('cuda', dtype=amp_dtype)
                else:
                    ctx = torch.amp.autocast('cuda')
                with ctx:
                    img_z, gene_z = model.encode(img, gene)
                    loss = model.contrastive_loss(img_z, gene_z)
                scaler.scale(loss).backward()
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer); scaler.update()
            else:
                img_z, gene_z = model.encode(img, gene)
                loss = model.contrastive_loss(img_z, gene_z)
                loss.backward()
                if max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            running += loss.item()
            if step % log_every == 0:
                bar.set_postfix(loss=f"{running/log_every:.4f}")
                running = 0.0

        # 에폭마다 저장
        if save_dir:
            _save_all(save_dir, model.img_encoder, model.gene_encoder, None)

    print(f"Training img↔gene done. ({time.time()-t0:.1f}s)")
    return model


# ----------------------------
# 내부 헬퍼: 배치에서 (imgs, node_ids) 추출
# ----------------------------
def _split_batch_for_spatial(batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    DataLoader 배치가 (imgs, node_ids) 또는 imgs 형태로 들어올 수 있음.
    반환: imgs, node_ids(or None)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2 and torch.is_tensor(batch[1]):
            return batch[0], batch[1].long()
        return batch[0], None
    return batch, None


# ----------------------------
# Train loops (img classification)
# ----------------------------
def train_img_classification(
    model: img2cluster_Classifier,
    train_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_norm: float = 0.0,
    use_amp: bool = True,
    device: Optional[str] = None,
    log_every: int = 50,
    save_dir: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    label_names: Optional[list] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    optimizer = Adam(_unique_trainable_params(model), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # autocast dtype: foundation.precision (bf16/fp16) 우선
    amp_dtype = getattr(model.img_encoder.foundation, "precision", None)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_acc = 0.0
        seen = 0
        bar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                   desc=f"[img→cls] Epoch {epoch}/{epochs}", leave=False)
        for step, batch in bar:
            img, y = batch
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            try:
                if use_amp:
                    if amp_dtype in (torch.bfloat16, torch.float16):
                        ctx = torch.amp.autocast('cuda', dtype=amp_dtype)
                    else:
                        ctx = torch.amp.autocast('cuda')
                    with ctx:
                        logits = model(img)
                        loss = F.cross_entropy(logits, y, weight=class_weights)
                    scaler.scale(loss).backward()
                    if max_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    logits = model(img)
                    loss = F.cross_entropy(logits, y, weight=class_weights)
                    loss.backward()
                    if max_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
            except RuntimeError as e:
                if "CUBLAS_STATUS_INVALID_VALUE" not in str(e):
                    raise
                if not hasattr(train_img_classification, "_warned_bf16_fallback"):
                    print("[WARN] CUBLAS_STATUS_INVALID_VALUE in AMP; retrying this batch in FP32.")
                    train_img_classification._warned_bf16_fallback = True
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=False):
                    logits = model(img.float())
                    loss = F.cross_entropy(logits, y, weight=class_weights)
                loss.backward()
                if max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            running_acc += (preds == y).sum().item()
            seen += y.size(0)
            if step % log_every == 0:
                avg_loss = running_loss / max(1, seen)
                avg_acc = running_acc / max(1, seen)
                bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}")

        if save_dir:
            _save_all(save_dir, model.img_encoder, None, None)
            torch.save(model.classifier.state_dict(), os.path.join(save_dir, "ImgClassifier.pt"))
            if label_names is not None:
                torch.save({"label_names": label_names}, os.path.join(save_dir, "label_names.pt"))

    print(f"Training img→cls done. ({time.time()-t0:.1f}s)")
    return model


# ----------------------------
# Train loops (img↔spatial) — 미니배치 + 서브그래프
# ----------------------------
def train_img_spatial_contrastive(
    model: img2spot_ContrastiveModel,
    img_loader,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    gcn_device: Optional[str] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_norm: float = 0.0,
    use_amp: bool = True,
    device: Optional[str] = None,
    log_every: int = 50,
    save_dir: Optional[str] = None,
):
    """
    미니배치 방식의 img↔spatial 학습.
    - 이미지 인코더는 AMP( bf16/fp16 ) 사용 가능
    - GCN 경로는 항상 FP32 로 고정(Blackwell + bf16에서의 cublas 오류 회피)
    - 그래프 텐서는 dtype/contiguous 보장(long / float32)
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gcn_device = torch.device(gcn_device) if gcn_device is not None else device
    if gcn_device.type == "cpu" and use_amp:
        print("[WARN] gcn_device=cpu -> disabling AMP to avoid GradScaler on CPU params.")
        use_amp = False
    model.to(device).train()

    # --- 그래프 텐서 dtype 고정 및 contiguous 보장 ---
    edge_index = edge_index.to(gcn_device, non_blocking=True).long().contiguous()
    edge_weight = edge_weight.to(gcn_device, non_blocking=True).to(torch.float32).contiguous() if edge_weight is not None else None

    # --- GCN은 항상 FP32로 동작(가중치 포함) ---
    model.gcn.to(gcn_device).float()

    optimizer = Adam(_unique_trainable_params(model), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype = getattr(model.img_encoder.foundation, "precision", None)  # bf16/fp16/None

    t0 = time.time()
    base = 0
    N = int(edge_index.max().item())+1
    for epoch in range(1, epochs + 1):
        running = 0.0
        base = 0
        bar = tqdm(enumerate(img_loader, 0), total=len(img_loader),
                   desc=f"[img↔spatial] Epoch {epoch}/{epochs}", leave=False)

        for step, batch in bar:
            # 배치 파싱 (imgs, node_ids | imgs)
            imgs, node_ids = _split_batch_for_spatial(batch)
            B = imgs.size(0)

            # 배치의 전역 노드 id
            if node_ids is None:
                batch_ids = torch.arange(base, base + B, device=gcn_device, dtype=torch.long) % N
            else:
                batch_ids = node_ids.to(gcn_device, non_blocking=True).long()
            base += B

            # 배치 서브그래프 (로컬 relabel: 0..B-1)
            sub_ei, sub_ew = subgraph(
                subset=batch_ids, edge_index=edge_index, edge_attr=edge_weight, relabel_nodes=True
            )
            sub_ei = sub_ei.long().contiguous()
            sub_ew = sub_ew.to(torch.float32).contiguous() if sub_ew is not None else None

            imgs = imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                # 인코더는 AMP, GCN은 FP32로 강제
                ctx = (torch.amp.autocast('cuda', dtype=amp_dtype)
                       if amp_dtype in (torch.bfloat16, torch.float16) else
                       torch.amp.autocast('cuda'))

                with ctx:
                    zi = model.img_encoder(imgs)       # AMP(bf16/fp16)
                    zi = F.normalize(zi, dim=1)

                # --- GCN 경로는 autocast 끄고 FP32 명시 ---
                zi32 = zi.to(torch.float32).contiguous()
                zi32_g = zi32.to(gcn_device) if gcn_device != device else zi32

                _debug_gcn_inputs(zi32_g, sub_ei, sub_ew, B, model.gcn, tag=f"img_spatial step={step}")
                _warmup_linear(zi32_g, model.gcn, tag=f"img_spatial step={step}")

                with torch.amp.autocast('cuda', enabled=False):
                    zs = model.gcn(zi32_g, sub_ei, sub_ew)   # FP32 연산
                    zs = F.normalize(zs, dim=1)

                    # 로짓/로스도 FP32에서 계산(안정성↑)
                    logits = (zi32_g @ zs.t()) / float(model.temperature)
                    targets = torch.arange(B, device=gcn_device)
                    loss = model.lam * F.cross_entropy(logits, targets) + \
                           (1.0 - model.lam) * F.cross_entropy(logits.t(), targets)
                    loss = loss.to(device)

                scaler.scale(loss).backward()
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer); scaler.update()

            else:
                # AMP OFF: GCN 포함 전부 FP32로
                zi = model.img_encoder(imgs)
                zi = F.normalize(zi, dim=1)

                zi32 = zi.to(torch.float32).contiguous()
                zi32_g = zi32.to(gcn_device) if gcn_device != device else zi32
                zs = model.gcn(zi32_g, sub_ei, sub_ew)       # FP32
                zs = F.normalize(zs, dim=1)

                logits = (zi32_g @ zs.t()) / float(model.temperature)
                targets = torch.arange(B, device=gcn_device)
                loss = model.lam * F.cross_entropy(logits, targets) + \
                       (1.0 - model.lam) * F.cross_entropy(logits.t(), targets)
                loss = loss.to(device)

                loss.backward()
                if max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            running += float(loss.item())
            if step % log_every == 0:
                bar.set_postfix(loss=f"{running/max(1,log_every):.4f}")
                running = 0.0

        if save_dir:
            _save_all(save_dir, model.img_encoder, None, model.gcn)

    print(f"Training img↔spatial done. ({time.time()-t0:.1f}s)")
    return model


# ----------------------------
# Joint training (멀티태스크: IS + IG) — 미니배치 + 서브그래프
# ----------------------------
def train_joint_img_gene_spatial(
    is_model: img2spot_ContrastiveModel,
    ig_model: img2gene_ContrastiveModel,
    img_loader,                 # (imgs, node_ids) 또는 imgs, 순서 보존
    gene_loader,                # (img, gene, id) or (img, gene)
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    gcn_device: Optional[str] = None,
    lam_is: float = 1.0,
    lam_ig: float = 1.0,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_norm: float = 0.0,
    use_amp: bool = True,
    device: Optional[str] = None,
    log_every: int = 50,
    save_dir: Optional[str] = None,
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 0.0,
    save_best: bool = True,
):
    """
    early_stopping / save_best 는 **train loss** 기준으로 동작한다.
    이 파이프라인에는 held-out validation split이 없어서 감시할 val 지표가 없다.
    train loss 기준 early stopping은 발산/정체는 잡아주지만 과적합은 잡지 못한다.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gcn_device = torch.device(gcn_device) if gcn_device is not None else device
    if gcn_device.type == "cpu" and use_amp:
        print("[WARN] gcn_device=cpu -> disabling AMP to avoid GradScaler on CPU params.")
        use_amp = False
    is_model.to(device).train()
    ig_model.to(device).train()

    # --- 그래프 텐서 dtype 고정 ---
    edge_index = edge_index.to(gcn_device, non_blocking=True).long().contiguous()
    edge_weight = edge_weight.to(gcn_device, non_blocking=True).to(torch.float32).contiguous() if edge_weight is not None else None

    # --- GCN은 항상 FP32로 운영(가중치도 FP32). Blackwell + bf16에서 안전한 설정 ---
    is_model.gcn.to(gcn_device).float()

    params = _unique_trainable_params(is_model, ig_model)
    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    amp_dtype_is = getattr(is_model.img_encoder.foundation, "precision", None)
    amp_dtype_ig = getattr(ig_model.img_encoder.foundation, "precision", None)

    def _infinite(loader):
        while True:
            for b in loader:
                yield b
    gene_iter = _infinite(gene_loader)

    t0 = time.time()
    base = 0
    N = int(edge_index.max().item())+1
    global_step = 0
    best_epoch_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False

    # nohup/파일 리다이렉트에서는 tqdm이 \r 로 로그를 도배하므로 갱신 주기를 늘린다.
    _is_tty = sys.stderr.isatty()
    _tqdm_kw = dict(mininterval=0.1 if _is_tty else 30.0,
                    dynamic_ncols=True, file=sys.stderr)

    epoch_bar = tqdm(range(1, epochs + 1), desc="[JOINT] epochs",
                     unit="ep", leave=True, **_tqdm_kw)
    for epoch in epoch_bar:
        running = 0.0
        running_n = 0
        base = 0
        epoch_sum, epoch_n = 0.0, 0
        epoch_seen = 0
        epoch_t0 = time.time()
        bar = tqdm(enumerate(img_loader, 0), total=len(img_loader),
                   desc=f"[JOINT] Epoch {epoch}/{epochs}", leave=False, **_tqdm_kw)
        for step, img_batch in bar:
            imgs_b, node_ids = _split_batch_for_spatial(img_batch)
            B = imgs_b.size(0)

            if node_ids is None:
                batch_ids = torch.arange(base, base + B, device=gcn_device, dtype=torch.long) % N
            else:
                batch_ids = node_ids.to(gcn_device, non_blocking=True).long()
            base += B

            # 배치 서브그래프: 로컬 인덱스로 relabel
            sub_ei, sub_ew = subgraph(
                subset=batch_ids, edge_index=edge_index, edge_attr=edge_weight, relabel_nodes=True
            )
            sub_ei = sub_ei.long().contiguous()
            sub_ew = sub_ew.to(torch.float32).contiguous() if sub_ew is not None else None

            gene_batch = next(gene_iter)
            if len(gene_batch) == 3:
                imgs_g, gene_g, _ = gene_batch
            else:
                imgs_g, gene_g = gene_batch

            imgs_b = imgs_b.to(device, non_blocking=True)
            imgs_g = imgs_g.to(device, non_blocking=True)
            gene_g = gene_g.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                ctx_is = (torch.amp.autocast('cuda', dtype=amp_dtype_is)
                          if amp_dtype_is in (torch.bfloat16, torch.float16) else
                          torch.amp.autocast('cuda'))
                ctx_ig = (torch.amp.autocast('cuda', dtype=amp_dtype_ig)
                          if amp_dtype_ig in (torch.bfloat16, torch.float16) else
                          torch.amp.autocast('cuda'))

                # --- IS 경로: 이미지 인코더는 AMP로, GCN은 FP32로 ---
                with ctx_is:
                    zi = is_model.img_encoder(imgs_b)                  # bf16/fp16 (autocast)
                    zi = F.normalize(zi, dim=1)
                # GCN 입력/가중치/엣지 모두 FP32 + autocast OFF + contiguous
                zi32 = zi.to(torch.float32).contiguous()
                zi32_g = zi32.to(gcn_device) if gcn_device != device else zi32
                # === 여기 추가 ===
                _debug_gcn_inputs(zi32_g, sub_ei, sub_ew, B, is_model.gcn, tag=f"joint step={step}")
                _warmup_linear(zi32_g, is_model.gcn, tag=f"img_spatial step={step}")
                
                with torch.amp.autocast('cuda', enabled=False):
                    zs = is_model.gcn(zi32_g, sub_ei, sub_ew)           # FP32로 연산
                    zs = F.normalize(zs, dim=1)
                    # 로짓도 FP32에서 계산(안정성↑)
                    logits_is = (zi32_g @ zs.t()) / float(is_model.temperature)
                    targets_b = torch.arange(B, device=gcn_device)
                    loss_is = is_model.lam * F.cross_entropy(logits_is, targets_b) + \
                              (1.0 - is_model.lam) * F.cross_entropy(logits_is.t(), targets_b)
                    loss_is = loss_is.to(device)

                # --- IG 경로: 전부 AMP ---
                with ctx_ig:
                    zimg, zg = ig_model.encode(imgs_g, gene_g)
                    loss_ig = ig_model.contrastive_loss(zimg, zg)

                loss = lam_is * loss_is + lam_ig * loss_ig
                scaler.scale(loss).backward()
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params, max_norm)
                scaler.step(optimizer); scaler.update()
            else:
                # AMP off 경로: GCN은 FP32, 로짓도 FP32에서
                zi = is_model.img_encoder(imgs_b); zi = F.normalize(zi, dim=1)
                zi32 = zi.to(torch.float32).contiguous()
                zi32_g = zi32.to(gcn_device) if gcn_device != device else zi32
                zs = is_model.gcn(zi32_g, sub_ei, sub_ew)
                zs = F.normalize(zs, dim=1)
                logits_is = (zi32_g @ zs.t()) / float(is_model.temperature)
                targets_b = torch.arange(B, device=gcn_device)
                loss_is = is_model.lam * F.cross_entropy(logits_is, targets_b) + \
                          (1.0 - is_model.lam) * F.cross_entropy(logits_is.t(), targets_b)
                loss_is = loss_is.to(device)

                zimg, zg = ig_model.encode(imgs_g, gene_g)
                loss_ig = ig_model.contrastive_loss(zimg, zg)

                loss = lam_is * loss_is + lam_ig * loss_ig
                loss.backward()
                if max_norm > 0:
                    nn.utils.clip_grad_norm_(params, max_norm)
                optimizer.step()

            loss_f = float(loss.item())
            running += loss_f
            running_n += 1
            epoch_sum += loss_f
            epoch_n += 1
            epoch_seen += B
            global_step += 1

            # --- wandb: step 단위 (비활성 시 no-op) ---
            wandb_log({
                "train/loss": loss_f,
                "train/loss_is": float(loss_is.item()),
                "train/loss_ig": float(loss_ig.item()),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }, step=global_step)

            if step % log_every == 0:
                # 실제 누적 스텝 수로 나눈다 (log_every로 나누면 step 0에서 50배 작게 보임)
                bar.set_postfix(loss=f"{running/max(1,running_n):.4f}",
                                is_=f"{float(loss_is.item()):.4f}",
                                ig=f"{float(loss_ig.item()):.4f}")
                running = 0.0
                running_n = 0

        bar.close()

        # --- epoch 집계 ---
        epoch_loss = epoch_sum / max(1, epoch_n)
        epoch_time = time.time() - epoch_t0

        # --- early stopping / best 판정 (train loss 기준) ---
        improved = epoch_loss < (best_epoch_loss - min_delta)
        if improved:
            best_epoch_loss = epoch_loss
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_metrics = {
            "epoch/loss": epoch_loss,
            "epoch/time_sec": epoch_time,
            "epoch/samples_per_sec": epoch_seen / max(1e-9, epoch_time),
            "epoch/index": epoch,
            "epoch/best_loss": best_epoch_loss,
            "epoch/epochs_no_improve": epochs_no_improve,
        }
        if device.type == "cuda":
            epoch_metrics["epoch/peak_mem_alloc_GB"] = torch.cuda.max_memory_allocated(device) / 1024**3
            epoch_metrics["epoch/peak_mem_reserved_GB"] = torch.cuda.max_memory_reserved(device) / 1024**3
        wandb_log(epoch_metrics, step=global_step)

        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}",
                              best=f"{best_epoch_loss:.4f}@{best_epoch}",
                              bad=epochs_no_improve)
        print(f"[JOINT] Epoch {epoch}/{epochs} loss={epoch_loss:.4f} "
              f"best={best_epoch_loss:.4f}@ep{best_epoch} no_improve={epochs_no_improve} "
              f"({epoch_time:.1f}s, {epoch_seen/max(1e-9, epoch_time):.1f} samples/s)", flush=True)

        # --- 저장: last는 항상, best는 개선됐을 때만 별도 디렉터리에 ---
        if save_dir:
            _save_all(save_dir, is_model.img_encoder, ig_model.gene_encoder, is_model.gcn)
            if save_best and improved:
                best_dir = os.path.join(save_dir, "best")
                _save_all(best_dir, is_model.img_encoder, ig_model.gene_encoder, is_model.gcn)
                print(f"[JOINT] best 갱신 → {best_dir} (epoch {epoch}, loss {epoch_loss:.4f})", flush=True)

        # --- early stop ---
        if early_stopping and epochs_no_improve >= patience:
            stopped_early = True
            print(f"[JOINT] Early stopping: {patience} epoch 동안 개선 없음 "
                  f"(best={best_epoch_loss:.4f} @ epoch {best_epoch}).", flush=True)
            epoch_bar.close()
            break

    if not stopped_early:
        epoch_bar.close()

    total_sec = time.time() - t0
    wandb_summary(final_epoch_loss=epoch_loss,
                  best_epoch_loss=best_epoch_loss,
                  best_epoch=best_epoch,
                  epochs_completed=epoch,
                  stopped_early=stopped_early,
                  total_train_sec=total_sec)
    print(f"Joint training (img↔spatial + img↔gene) done. "
          f"({total_sec:.1f}s, {epoch}/{epochs} epochs, best={best_epoch_loss:.4f} @ ep{best_epoch})")
    return is_model, ig_model
