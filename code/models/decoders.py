# -*- coding: utf-8 -*-
# code/models/decoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZINBDecoder(nn.Module):
    """
    입력: (B, D) → (mu, theta, pi) 각 (B, G)
    ✅ 안정적인 초기화 및 출력 범위 제한
    """
    def __init__(self, in_dim: int, gene_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden // 2, gene_dim)
        self.theta = nn.Linear(hidden // 2, gene_dim)
        self.pi = nn.Linear(hidden // 2, gene_dim)
        
        # ✅ 초기값 안정화: 작은 가중치로 시작
        with torch.no_grad():
            nn.init.normal_(self.mu.weight, 0, 0.01)
            nn.init.normal_(self.theta.weight, 0, 0.01)
            nn.init.normal_(self.pi.weight, 0, 0.01)
            self.mu.bias.fill_(0.5)       # log(exp(0.5)) ≈ 1.6, softplus 후 약 1.6
            self.theta.bias.fill_(0.0)    # log(exp(0)) = 1, softplus 후 약 1.3
            self.pi.bias.fill_(0.0)       # sigmoid(0) = 0.5

    def forward(self, z):
        h = self.backbone(z)
        
        # ✅ 범위 제한된 출력
        # mu: softplus로 항상 양수, 너무 크지 않도록 제한
        mu = F.softplus(self.mu(h)) + 1e-4
        mu = torch.clamp(mu, min=1e-4, max=1e4)
        
        # theta: dispersion parameter, 너무 크면 gradient 불안정
        theta = F.softplus(self.theta(h)) + 1e-4
        theta = torch.clamp(theta, min=1e-4, max=1e4)
        
        # pi: zero-inflation rate, [0, 1]에 제한
        pi = torch.sigmoid(self.pi(h))
        
        return mu, theta, pi
