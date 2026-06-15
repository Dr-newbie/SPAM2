import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel


def _collect_lora_targets_for_vit(enc, backbone_name: str):
    """
    timm ViT / OpenCLIP / DINO / Virchow / UNI / Conch 등:
    - qkv / proj / fc1 / fc2 등을 대상으로 LoRA
    HF Vision (CLIPVisionModel/ViTModel) 계열:
    - query/key/value/out_proj, attention.output.dense, intermediate.dense 등
    """
    targets = []
    for n, m in enc.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        # 공통 패턴
        if any(k in n for k in ["qkv", "proj", "fc1", "fc2"]):
            targets.append(n)
            continue
        # HF-CLIPVision/HF-ViT 패턴
        if any(k in n for k in ["query", "key", "value", "out_proj", "output.dense", "intermediate.dense", "dense"]):
            targets.append(n)
    # 중복 제거
    targets = sorted(set(targets))
    return targets

def attach_lora_to_foundation(
    foundation,
    backbone_name: str,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
):
    enc = foundation.encoder
    targets = _collect_lora_targets_for_vit(enc, backbone_name)

    if len(targets) == 0:
        raise ValueError(f"No target Linear layers found for LoRA in {backbone_name}. Check naming.")

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=targets,
        task_type="FEATURE_EXTRACTION",  # 분류/LM이 아니므로
    )
    foundation.encoder = get_peft_model(enc, cfg)  # 기본적으로 base weight는 freeze, LoRA만 학습
    return foundation