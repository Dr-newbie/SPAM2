# -*- coding: utf-8 -*-
"""
Histology-derived node features H(0) for the spatial GCN.

Manuscript definition of the spatial graph:
  - E    : Delaunay triangulation over the spot centroids
  - H(0) : the histology (image-foundation) embedding of each spot

Spot coordinates therefore determine E only; they are never fed to the GCN as
node features. This module builds the H(0) matrix for every node of a slide, in
node-index order, so pretraining, finetuning and inference all consume the same
representation.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


def unpack_img_and_index(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accept (img, node_idx) or (img, gene, node_idx) batches."""
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unsupported batch type {type(batch)!r}; expected tuple/list.")
    if len(batch) == 2:
        img, idx = batch
    elif len(batch) == 3:
        img, _, idx = batch
    else:
        raise ValueError(f"Unsupported batch arity {len(batch)}; expected 2 or 3.")
    if not torch.is_tensor(idx):
        raise TypeError(
            "The loader must yield integer node indices, not string spot IDs. "
            "Use a dataset whose __getitem__ returns the node index "
            "(ImgOnlyDataset or _NodeIdxDataset)."
        )
    return img, idx


@torch.no_grad()
def build_histology_node_features(
    img_encoder: nn.Module,
    loader,
    n_nodes: int,
    out_dim: int,
    device: torch.device,
    proj: Optional[nn.Module] = None,
    amp_dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Run `img_encoder` (optionally followed by `proj`) over every spot and scatter
    the embeddings into a dense (n_nodes, out_dim) matrix indexed by node id.

    Args:
        img_encoder: module mapping a batch of patches to (B, D). May be a bare
            foundation encoder or an ImageFoundationAdapter.
        loader: ordered DataLoader yielding (img, node_idx) or
            (img, gene, node_idx). Shuffling is harmless — rows are scattered by
            node index — but every node must be covered exactly once.
        n_nodes: number of graph nodes (== number of aligned spots).
        out_dim: feature width after `proj`.
        proj: optional projection applied to the encoder output, used to match
            the width the pretrained GCN was trained on.
        amp_dtype: autocast dtype for the encoder forward (CUDA only).

    Returns:
        (n_nodes, out_dim) float32 tensor on `device`.
    """
    device = torch.device(device)

    # Optional crash-resume cache (SPAM_SPOTFEAT_CACHE=<dir>): this box loses
    # power mid-run; H(0) is the longest inference stage, so persist it keyed
    # by (n_nodes, out_dim), which is unique per section within one run dir.
    import os as _os
    import numpy as _np
    _cache_dir = _os.environ.get("SPAM_SPOTFEAT_CACHE")
    _cache_path = None
    if _cache_dir:
        _os.makedirs(_cache_dir, exist_ok=True)
        _cache_path = _os.path.join(_cache_dir, f"spotfeat_{n_nodes}x{out_dim}.npy")
        if _os.path.isfile(_cache_path):
            arr = _np.load(_cache_path)
            if arr.shape == (n_nodes, out_dim) and _np.isfinite(arr).all():
                print(f"[SpotFeat] cache hit: {_cache_path}")
                return torch.from_numpy(arr).to(device)
            print(f"[SpotFeat] cache invalid, rebuilding: {_cache_path}")

    feats = torch.zeros(n_nodes, out_dim, dtype=torch.float32, device=device)
    filled = torch.zeros(n_nodes, dtype=torch.bool, device=device)

    modules = [img_encoder] + ([proj] if proj is not None else [])
    was_training = [m.training for m in modules]
    for m in modules:
        m.eval()

    # Mirror ImageFoundationAdapter: cast the input to the backbone precision.
    enc_precision = getattr(img_encoder, "precision", None)
    if enc_precision is None:
        enc_precision = getattr(getattr(img_encoder, "foundation", None), "precision", None)

    use_autocast = (device.type == "cuda"
                    and amp_dtype in (torch.float16, torch.bfloat16))

    it = loader
    if verbose:
        try:
            from tqdm import tqdm
            it = tqdm(loader, desc="[SpotFeat] histology H(0)", leave=False)
        except Exception:
            pass

    try:
        for batch in it:
            img, idx = unpack_img_and_index(batch)
            img = img.to(device, non_blocking=True)
            if enc_precision is not None and not use_autocast:
                img = img.to(dtype=enc_precision)
            idx = idx.to(device, non_blocking=True).long().view(-1)

            if use_autocast:
                with torch.autocast("cuda", dtype=amp_dtype):
                    z = img_encoder(img)
            else:
                z = img_encoder(img)

            z = z.float()
            if proj is not None:
                z = proj(z).float()
            if z.size(1) != out_dim:
                raise RuntimeError(
                    f"Histology feature width {z.size(1)} != expected {out_dim}. "
                    "Check --pt_proj_dim against the pretrained GCN input dim."
                )
            feats[idx] = z
            filled[idx] = True
    finally:
        for m, flag in zip(modules, was_training):
            m.train(flag)

    n_missing = int((~filled).sum().item())
    if n_missing:
        raise RuntimeError(
            f"[SpotFeat] {n_missing}/{n_nodes} nodes were never visited by the "
            "loader. The image loader and the graph must cover the same spots "
            "in the same node indexing."
        )
    if not torch.isfinite(feats).all():
        raise RuntimeError("[SpotFeat] histology features contain NaN/Inf.")

    if verbose:
        print(f"[SpotFeat] H(0) built: {tuple(feats.shape)} "
              f"(mean={feats.mean().item():.4f}, std={feats.std().item():.4f})")
    if _cache_path:
        _np.save(_cache_path, feats.cpu().numpy())
        print(f"[SpotFeat] cache saved: {_cache_path}")
    return feats


def freeze_and_lock_eval(module: nn.Module) -> int:
    """
    Freeze `module` and pin it to eval mode.

    Setting requires_grad=False is not enough on its own: a later
    `model.train()` recurses into every child and puts the frozen backbone back
    into train mode, re-enabling its dropout. UNI happens to be unaffected
    (drop_rate=0, drop_path=Identity — verified identical outputs in both
    modes), but Phikon/PLIP are HF encoders with dropout 0.1, so a frozen
    backbone would silently emit noisy features. Overriding .train() makes the
    mode stick.

    Returns the number of frozen parameters.
    """
    n = 0
    for p in module.parameters():
        p.requires_grad = False
        n += p.numel()
    module.eval()
    module.train = lambda mode=True: module   # nn.Module.train() recurses via children()
    return n


def make_spot_in_proj(dim_img: int, dim_spot_in: int) -> nn.Module:
    """
    Projection that maps the image-foundation output to the width the spatial
    GCN was pretrained on. Mirrors ImageFoundationAdapter.proj so the pretrained
    weights transfer 1:1.
    """
    if int(dim_img) == int(dim_spot_in):
        return nn.Identity()
    return nn.Linear(int(dim_img), int(dim_spot_in))
