"""Shared helper: extract frozen UNI (uni_v1) patch embeddings for the baselines,
so each baseline's native image encoder can be swapped for the same histology
foundation model SPAM uses (R2 fair-encoder ablation). Output is (N, 1024) fp32,
cached to disk. Runs fp32 (this box's fp16/bf16 GEMM is broken — see memory).
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# SPAM model code (Foundations.inf_encoder_factory) lives under Code_final/code.
_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code"))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)


def _uni_encoder(device):
    # The baseline harness imports its own `spam_bench/models.py` first, caching
    # it as sys.modules['models']; that shadows the code/models package, so a
    # plain `from models.Foundations import ...` resolves to the wrong module.
    # Foundations also uses a relative import (`.transforms_utils`), so it must
    # be imported *as a submodule of the real code/models package*, not stand-
    # alone. Temporarily evict the bench `models.*` entries, put code/ on the
    # path so `models` resolves to code/models, build the encoder, then restore
    # the bench modules so the surrounding baseline code keeps working.
    import importlib
    _saved = {k: sys.modules[k] for k in list(sys.modules)
              if k == "models" or k.startswith("models.")}
    for k in _saved:
        del sys.modules[k]
    _added = False
    if _CODE not in sys.path:
        sys.path.insert(0, _CODE); _added = True
    else:
        sys.path.remove(_CODE); sys.path.insert(0, _CODE)
    try:
        from models.Foundations import inf_encoder_factory
        weights = os.environ.get("SPAM_UNI_WEIGHTS")
        enc = inf_encoder_factory("uni_v1")(weights_path=weights)
        enc = enc.to(device).eval()
        tf = getattr(enc, "eval_transforms", None)
        if tf is None:
            raise RuntimeError("UNI encoder has no eval_transforms")
    finally:
        # drop the code/models.* we just imported and restore the bench ones
        for k in list(sys.modules):
            if k == "models" or k.startswith("models."):
                del sys.modules[k]
        sys.modules.update(_saved)
        if _added and _CODE in sys.path:
            sys.path.remove(_CODE)
    return enc, tf


class _PatchTfDataset(Dataset):
    def __init__(self, paths, tf):
        self.paths = list(paths)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), i


@torch.no_grad()
def uni_features(paths, cache_path, device="cuda", batch_size=128, num_workers=8):
    """Return (N, 1024) frozen UNI embeddings for `paths`, cached at `cache_path`.

    Order matches `paths` (row i = paths[i]). fp32 forward (no autocast) because
    fp16/bf16 GEMM is broken on this machine.
    """
    if os.path.isfile(cache_path):
        return np.load(cache_path)
    enc, tf = _uni_encoder(device)
    dim = int(enc.get_output_dim())
    ds = _PatchTfDataset(paths, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    out = np.zeros((len(ds), dim), dtype=np.float32)
    from tqdm import tqdm
    for x, idx in tqdm(dl, desc="[UNI] features", leave=False):
        f = enc.forward_features(x.to(device).float())
        out[idx.numpy()] = f.float().cpu().numpy()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, out)
    return out
