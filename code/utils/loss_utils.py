# -*- coding: utf-8 -*-
# code/utils/loss_util.py
import torch
import torch.nn.functional as F

def zinb_nll(
    x: torch.Tensor,          # (B, G) >= 0  (연속값 허용, count면 정규화/스케일은 호출부에서 결정)
    mu: torch.Tensor,         # (B, G) > 0
    theta: torch.Tensor,      # (B, G) > 0  (dispersion)
    pi: torch.Tensor,         # (B, G) in [0,1]
    mask: torch.Tensor | None = None,  # (B, G) bool
    gene_weight: torch.Tensor | None = None,  # (G,) or (B,G)
    reduction: str = "mean",
    eps: float = 1e-8,
):
    """
    Zero-Inflated Negative Binomial Negative Log Likelihood.
    - x가 연속(로그정규화 등)일 때도 근사 사용. count라면 그대로 사용 가능.
    - ✅ 강화된 수치 안정성 처리
    """
    # ✅ 더 강한 안전 클램프 (극단값 방지)
    mu = torch.clamp(mu, min=eps, max=1e8)
    theta = torch.clamp(theta, min=eps, max=1e8)
    pi = torch.clamp(pi, min=eps, max=1 - eps)
    x = torch.clamp(x, min=0.0)

    # ✅ NB log probability 계산을 더 안정적으로
    # NB(x) = C(x+θ-1, x) * (θ/(θ+μ))^θ * (μ/(θ+μ))^x
    
    # lgamma 연산에서 overflow 방지
    t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
    t1 = torch.where(torch.isnan(t1) | torch.isinf(t1), torch.zeros_like(t1), t1)
    
    # log-space 연산으로 수치 안정성 강화
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    
    nb_log_prob = t1 + t2 + t3              # log NB(x)
    
    # ✅ NB log prob에서 extreme 값 처리
    nb_log_prob = torch.where(torch.isnan(nb_log_prob) | torch.isinf(nb_log_prob), 
                               torch.full_like(nb_log_prob, -1e6), nb_log_prob)

    # ZI mixture
    is_zero = (x <= eps)
    
    # ✅ log-sum-exp trick으로 log(a + b) 안정적 계산
    log_prob_zero = torch.logsumexp(
        torch.stack([
            torch.log(pi + eps),
            torch.log(1.0 - pi + eps) + nb_log_prob
        ], dim=0),
        dim=0
    )
    log_prob_nonzero = torch.log(1.0 - pi + eps) + nb_log_prob
    log_prob = torch.where(is_zero, log_prob_zero, log_prob_nonzero)

    nll = -log_prob
    
    # ✅ NLL에서 extreme 값 제거
    nll = torch.where(torch.isnan(nll) | torch.isinf(nll), 
                      torch.full_like(nll, 1e6), nll)
    
    if mask is not None:
        nll = torch.where(mask, nll, torch.zeros_like(nll))

    if gene_weight is not None:
        if gene_weight.dim() == 1:
            nll = nll * gene_weight.view(1, -1)
        else:
            nll = nll * gene_weight

    if reduction == "mean":
        denom = (mask.float().sum() if mask is not None else nll.numel())
        denom = max(float(denom), 1.0)
        result = nll.sum() / denom
    elif reduction == "sum":
        result = nll.sum()
    elif reduction == "none":
        result = nll
    else:
        raise ValueError(f"unknown reduction: {reduction}")
    
    # ✅ 최종 결과 검증
    if torch.isnan(result) or torch.isinf(result):
        print(f"[WARN] zinb_nll returned NaN/Inf: {result.item()}")
        return torch.tensor(1e6, device=x.device, dtype=x.dtype)
    
    return result
