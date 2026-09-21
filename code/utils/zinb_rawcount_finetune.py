# -*- coding: utf-8 -*-
"""Raw-count ZINB objective for SPAM finetuning (submission reference).

Reviewer R1-5 correctly noted that a negative-binomial (NB) likelihood is a
*count* model and must be evaluated on RAW COUNTS, not on library-size-normalized
(CP10K) values. This module implements the corrected objective used in the paper:

  Notation (per cell i, gene g)
    x_ig            raw integer count
    l_i = sum_g x_ig            library size
    s_i = l_i / 1e4             per-cell size factor
    x_ig / s_i = CP10K          library-size-normalized expression
    encoder input   = log1p(CP10K) = log1p(x_ig / s_i)     (unchanged)
    decoder output  = (mu_hat_ig, theta_ig, pi_ig)         mu_hat on the CP10K scale
    COUNT-SCALE MEAN:  mu_ig = s_i * mu_hat_ig             <-- size-factor restores counts

  ZINB likelihood ON RAW COUNTS (paper Eqs. 9-10)
    log NB(x | mu, theta) = lgamma(x+theta) - lgamma(theta) - lgamma(x+1)
                            + theta*(log theta - log(theta+mu))
                            +     x*(log mu    - log(theta+mu))
    P(x | mu, theta, pi) = pi * 1[x=0] + (1-pi) * NB(x | mu, theta)
    L_ZINB = - (1/NG) sum_{i,g} log P(x_ig | mu_ig, theta_ig, pi_ig)

  Total finetuning loss
    L = lambda_zinb * L_ZINB
        + lambda_mse * || log1p(mu) - log1p(x) ||^2      (auxiliary, log-count space)
        + lambda_ce  * CE(celltype)                       (auxiliary head, optional)

Evaluation predicts the zero-inflated count mean xhat = (1-pi)*mu and compares to
the ground truth in log1p(CP10K) space.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def zinb_nll_rawcount(
    x: torch.Tensor,          # (B, G) RAW counts  (>= 0, integer-valued)
    mu: torch.Tensor,         # (B, G) COUNT-scale mean  (> 0)  == s_i * mu_hat
    theta: torch.Tensor,      # (B, G) dispersion (> 0)
    pi: torch.Tensor,         # (B, G) zero-inflation prob in [0,1]
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """ZINB negative log-likelihood on raw counts (log-space, numerically stable)."""
    mu = torch.clamp(mu, min=eps, max=1e8)
    theta = torch.clamp(theta, min=eps, max=1e8)
    pi = torch.clamp(pi, min=eps, max=1 - eps)
    x = torch.clamp(x, min=0.0)

    log_theta_mu = torch.log(theta + mu + eps)
    nb_log_prob = (
        torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )
    # ZI mixture: x==0 -> log(pi + (1-pi) NB(0)); x>0 -> log(1-pi) + log NB(x)
    log_prob_zero = torch.logsumexp(
        torch.stack([torch.log(pi + eps), torch.log(1.0 - pi + eps) + nb_log_prob], dim=0), dim=0
    )
    log_prob_nonzero = torch.log(1.0 - pi + eps) + nb_log_prob
    log_prob = torch.where(x <= eps, log_prob_zero, log_prob_nonzero)

    nll = -log_prob
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


def raw_counts_from_log1p(x_log: torch.Tensor, size_factor: torch.Tensor) -> torch.Tensor:
    """Recover raw counts from the log1p(CP10K) input.
    CP10K = expm1(x_log);  raw count x = CP10K * s_i  (s_i = library/1e4).
    size_factor: (B,) or (B,1). Use a stored integer counts layer instead when available.
    """
    if size_factor.dim() == 1:
        size_factor = size_factor.unsqueeze(1)
    cp10k = torch.expm1(torch.clamp(x_log, min=0.0))
    return cp10k * size_factor


def finetune_loss_rawcount(
    x_log: torch.Tensor,          # (B, G) encoder input = log1p(CP10K)
    size_factor: torch.Tensor,    # (B,)  s_i = library_size / 1e4
    mu_hat: torch.Tensor,         # (B, G) decoder mean on the CP10K scale (> 0)
    theta: torch.Tensor,          # (B, G) dispersion (> 0)
    pi: torch.Tensor,             # (B, G) zero-inflation prob
    x_raw: torch.Tensor | None = None,   # (B, G) raw counts if available (preferred)
    lambda_zinb: float = 1.0,
    lambda_mse: float = 0.1,
    eps: float = 1e-8,
):
    """Corrected raw-count ZINB finetuning loss (+ auxiliary log-space MSE).

    Returns (total, {"zinb":..., "mse":...}). The cell-type CE head, if used, is
    added by the caller exactly as before.
    """
    sf = size_factor.unsqueeze(1) if size_factor.dim() == 1 else size_factor
    if x_raw is None:
        x_raw = raw_counts_from_log1p(x_log, sf)          # CP10K * s_i
    mu = sf * mu_hat                                        # COUNT-scale mean  mu_ig = s_i * mu_hat_ig

    loss_zinb = zinb_nll_rawcount(x_raw, mu, theta, pi, reduction="mean", eps=eps)
    # auxiliary reconstruction in log-count space (both sides count scale)
    loss_mse = F.mse_loss(torch.log1p(torch.clamp(mu, min=eps)), torch.log1p(x_raw))
    total = lambda_zinb * loss_zinb + lambda_mse * loss_mse
    return total, {"zinb": loss_zinb.detach(), "mse": loss_mse.detach()}


# ---------------------------------------------------------------------------
# Drop-in replacement for the ZINB block in run_finetuning_xattn_withcelltype.py
# (previously the loss was evaluated on CP10K via x_raw = expm1(x_log) with an
#  un-scaled mu — i.e. an NB on normalized values, which R1-5 flagged). Replace
# that block with:
#
#     mu_hat, theta, pi, aux = model(batch, spot_feat_in, edge_index, edge_weight)
#     x_log = batch["expr"].float()                 # log1p(CP10K)
#     s_i   = batch["size_factor"].float()          # library/1e4  (add to the loader)
#     loss, parts = finetune_loss_rawcount(x_log, s_i, mu_hat, theta, pi,
#                                          x_raw=batch.get("counts"),  # optional raw-count layer
#                                          lambda_zinb=1.0, lambda_mse=0.1)
#     # + lambda_ce * CE(aux["cls_logits"], y_celltype)   (unchanged)
#
# `size_factor` is already stored in adata.obs["size_factor"] by
# dataset_utils.load_h5ad_with_preproc (s_i = library/1e4); expose it per sample
# in the DataLoader batch. A raw integer counts layer, if kept at preprocessing
# time, should be passed as `x_raw` and is preferred over the expm1 recovery.
# ---------------------------------------------------------------------------
