# gene_encoder.py
# -*- coding: utf-8 -*-
# The code based on mCat, mahmoodlab.

import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F


__all__ = [
    "SNNBackbone",
    "GeneEncoder",
    "initialize_weights",
    "init_max_weights",
    "Reg_Block",
]


# ----------------------------
# Blocks & Init
# ----------------------------
def Reg_Block(dim1: int, dim2: int, dropout: float = 0.25) -> nn.Sequential:
    """
    Multilayer reception block (Linear -> ReLU -> Dropout)
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout, inplace=False),
    )


def initialize_weights(module: nn.Module):
    """
    Xavier init for Linear, and identity-like init for LayerNorm
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def init_max_weights(module: nn.Module):
    """
    Alternative init using N(0, 1/sqrt(fan_in))
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            if m.bias is not None:
                m.bias.data.zero_()


# ----------------------------
# MCAT-style SNN backbone (encoder only)
# ----------------------------
class SNNBackbone(nn.Module):
    """
    MCAT/MAHMOODLAB 스타일의 Genomic SNN에서 classifier를 제거하고
    fc_omic(=인코더)만 남긴 백본.

    Args:
        input_dim: 유전자(또는 omic) 입력 차원
        model_size_omic: ['small', 'medium', 'big', 'huge', 'giant']
        dropout_hidden: 은닉층 dropout 비율 (첫 블록 제외)
    """
    SIZE_DICT = {
        "small":  [256, 256, 256],
        "medium": [512, 512, 512],
        "big":    [1024, 1024, 256],
        "huge":   [1024, 1024, 512],
        "giant":  [1024, 1024, 1024],
    }

    def __init__(
        self,
        input_dim: int,
        model_size_omic: str = "small",
        dropout_hidden: float = 0.25,
    ):
        super().__init__()
        if model_size_omic not in self.SIZE_DICT:
            raise ValueError(f"Unknown model_size_omic: {model_size_omic}")

        hidden: List[int] = self.SIZE_DICT[model_size_omic]

        # 첫 블록(입력 -> 첫 은닉)
        fc_omic = [Reg_Block(dim1=input_dim, dim2=hidden[0])]

        # 중간 블록들
        for i, _ in enumerate(hidden[1:-1]):
            fc_omic.append(Reg_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=dropout_hidden))

        # 마지막 선형 (hidden[-2] -> hidden[-1]) 은 활성 없이 마무리
        fc_omic.append(nn.Linear(hidden[-2], hidden[-1]))

        self.fc_omic = nn.Sequential(*fc_omic)
        initialize_weights(self)

        self.output_dim = hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            features: (B, output_dim)
        """
        features = self.fc_omic(x)
        return features


# ----------------------------
# Gene Encoder (with optional projection head)
# ----------------------------
class GeneEncoder(nn.Module):
    """
    Contrastive 학습에 바로 투입 가능한 유전자 인코더.

    - 백본: SNNBackbone (fc_omic)
    - 옵션: proj_dim을 주면 (output_dim -> proj_dim) projection head를 추가
    - 옵션: normalize=True면 L2 normalize 후 반환

    Args:
        input_dim: 유전자(omic) 입력 차원
        model_size_omic: 백본 크기 프리셋
        proj_dim: None이면 백본 출력 차원 그대로 사용, 정수면 추가 선형 투영
        normalize: True면 L2 정규화 반환
        dropout_hidden: SNNBackbone 내부 숨김층 dropout(첫 블록 제외)
    """

    def __init__(
        self,
        input_dim: int,
        model_size_omic: str = "small",
        proj_dim: Optional[int] = None,
        normalize: bool = True,
        dropout_hidden: float = 0.25,
    ):
        super().__init__()
        self.normalize = normalize

        # Backbone
        self.backbone = SNNBackbone(
            input_dim=input_dim,
            model_size_omic=model_size_omic,
            dropout_hidden=dropout_hidden,
        )

        # Projection head (optional)
        self.proj: Optional[nn.Module]
        if proj_dim is not None and proj_dim > 0 and proj_dim != self.backbone.output_dim:
            self.proj = nn.Linear(self.backbone.output_dim, proj_dim)
            initialize_weights(self.proj)
            self.out_dim = proj_dim
        else:
            self.proj = None
            self.out_dim = self.backbone.output_dim

    def forward(self, x: torch.Tensor, return_preproj: bool = False):
        """
        Args:
            x: (B, input_dim)
            return_preproj: True면 (pre_features, z) 튜플 반환

        Returns:
            z: (B, out_dim)  [normalize=True면 L2 정규화]
            또는 (pre_features, z)
        """
        feats = self.backbone(x)  # (B, backbone_dim)
        if self.proj is not None:
            z = self.proj(feats)
        else:
            z = feats

        if self.normalize:
            z = F.normalize(z, dim=1)

        if return_preproj:
            return feats, z
        return z
