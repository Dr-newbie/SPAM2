# -*- coding: utf-8 -*-
"""
Baseline model construction.

Each entry imports the architecture from the upstream repository rather than
reimplementing it, so the comparison reflects the published methods. Only the
output width is rebound to our gene panel and, where a repo hard-codes a Visium
constant, that constant is passed in.
"""
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn

REPOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repos")


@contextmanager
def _on_path(*dirs):
    """Temporarily prepend repo dirs; several repos use bare `import config`."""
    added = [d for d in dirs if d not in sys.path]
    for d in added:
        sys.path.insert(0, d)
    saved = dict(sys.modules)
    try:
        yield
    finally:
        for d in added:
            try:
                sys.path.remove(d)
            except ValueError:
                pass
        # drop modules the repo injected under generic names (config, models, ...)
        for k in list(sys.modules):
            if k not in saved and k.split(".")[0] in {
                "config", "models", "modules", "model", "dataset", "utils",
                "DeepSpaceLib", "BasicLib", "model_AE", "model_MLP", "geb",
                "csra", "heteroconv", "train",
            }:
                del sys.modules[k]


# ----------------------------------------------------------------------------
# ST-Net (He et al., Nat Biomed Eng 2020) — DenseNet121 with a linear head.
# ----------------------------------------------------------------------------
def build_stnet(n_genes: int) -> nn.Module:
    import torchvision
    net = torchvision.models.densenet121(weights="DEFAULT")
    net.classifier = nn.Linear(net.classifier.in_features, n_genes)
    return net


# ----------------------------------------------------------------------------
# DeepSpaCE (Monjo et al., Sci Rep 2022) — make_model() from DeepSpaceLib.
# ----------------------------------------------------------------------------
def _albumentations_compat():
    """
    DeepSpaCE imports `albumentations.pytorch.ToTensor`, renamed to ToTensorV2 in
    albumentations >= 1.0. Alias it rather than pinning the old release, which
    would drag in an incompatible numpy. Only the import chain needs it —
    make_model() itself does no augmentation.
    """
    try:
        import albumentations.pytorch as ap
        if not hasattr(ap, "ToTensor"):
            ap.ToTensor = ap.ToTensorV2
    except Exception:
        pass


def _torchvision_pretrained_compat():
    """DeepSpaCE calls vgg16(pretrained=True); the kwarg was dropped in tv 0.15+."""
    import torchvision.models as tvm
    for name, wname in (("vgg16", "VGG16_Weights"), ("densenet121", "DenseNet121_Weights")):
        fn = getattr(tvm, name)
        if getattr(fn, "_spam_patched", False):
            continue
        weights_enum = getattr(tvm, wname)

        def wrapper(*a, _fn=fn, _w=weights_enum, pretrained=None, **kw):
            if pretrained is not None and "weights" not in kw:
                kw["weights"] = _w.DEFAULT if pretrained else None
            return _fn(*a, **kw)

        wrapper._spam_patched = True
        setattr(tvm, name, wrapper)


def build_deepspace(n_genes: int, arch: str = "DenseNet121") -> nn.Module:
    _albumentations_compat()
    _torchvision_pretrained_compat()
    with _on_path(os.path.join(REPOS, "DeepSpaCE")):
        import DeepSpaceLib
        import contextlib, io
        # make_model prints every parameter tensor; silence that, and take the
        # net from its (net, params_to_update) return.
        with contextlib.redirect_stdout(io.StringIO()):
            # use_pretrained=True, transfer=False -> full finetuning (paper setting)
            net, _ = DeepSpaceLib.make_model(True, n_genes, False, arch)
        return net


# ----------------------------------------------------------------------------
# BLEEP (Xie et al., NeurIPS 2023) — CLIP-style image/expression contrastive.
# ----------------------------------------------------------------------------
def build_bleep(n_genes: int):
    d = os.path.join(REPOS, "BLEEP")
    with _on_path(d):
        import config as CFG
        CFG.spot_embedding = int(n_genes)   # repo hard-codes 3467 shared HVGs
        import importlib
        import models as bleep_models
        importlib.reload(bleep_models)
        return bleep_models.CLIPModel(spot_embedding=int(n_genes))


# ----------------------------------------------------------------------------
# STMCL (Tan et al., Bioinformatics 2024) — multimodal contrastive.
# ----------------------------------------------------------------------------
def build_stmcl(n_genes: int, temperature: float = 1.0,
                image_embedding: int = 1024, projection_dim: int = 256,
                dropout: float = 0.1):
    """
    STMCL adds learned x/y positional embeddings to the expression vector, so
    the batch dict must carry "position" as integer bin indices in [0, 128).
    image_embedding=1024 is the DenseNet121 feature width used by its
    ImageEncoder.
    """
    d = os.path.join(REPOS, "STMCL")
    with _on_path(d):
        import model as stmcl_model
        return stmcl_model.STMCL(
            temperature=temperature, image_embedding=image_embedding,
            spot_embedding=int(n_genes), projection_dim=projection_dim,
            dropout=dropout,
        )


# ----------------------------------------------------------------------------
# DeepPT (Hoang et al., Nat Cancer 2024) — ResNet50 features -> AE(512) -> MLP.
# The upstream pipeline is three separate stages run over TCGA slides; the
# architecture is what we reuse.
# ----------------------------------------------------------------------------
class DeepPTModel(nn.Module):
    """
    AE bottleneck + MLP regressor over frozen ResNet50 features.

    Two faithful-adaptation notes:

    * Upstream trains the autoencoder as a separate stage and then feeds the
      512-d *encoder* output to the MLP. We keep the same modules and use
      `ae.encoder`, training both jointly, since our stage-1 features are fixed.
    * `MLP_regression.forward` ends in `torch.mean(x, dim=0)`, which collapses
      all tiles of a slide into one bulk profile — DeepPT predicts bulk mRNA
      from a whole slide. Spatial prediction needs one profile per tile, so the
      tile-averaging is dropped here. This is the same adaptation the published
      ST benchmarks make when they include DeepPT.
    """

    def __init__(self, n_inputs: int, n_genes: int, n_hiddens: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        with _on_path(os.path.join(REPOS, "DeepPT", "12AE"),
                      os.path.join(REPOS, "DeepPT", "13DeepPT_train")):
            from model_AE import AutoEncoder
            from model_MLP import MLP_regression
            self.ae = AutoEncoder(n_inputs, n_hiddens, n_hiddens)
            self.mlp = MLP_regression(n_hiddens, n_hiddens, n_genes,
                                      dropout, bias_init=None)

    def forward(self, x):
        z = self.ae.encoder(x)          # 2048 -> 512, upstream's compressed code
        h = self.mlp.layer0(z)
        return self.mlp.layer1(h)       # per-tile output; no mean over tiles


def build_deeppt(n_genes: int, n_inputs: int = 2048):
    return DeepPTModel(n_inputs=n_inputs, n_genes=n_genes)


BUILDERS = {
    "stnet": build_stnet,
    "deepspace": build_deepspace,
    "bleep": build_bleep,
    "stmcl": build_stmcl,
    "deeppt": build_deeppt,
}
