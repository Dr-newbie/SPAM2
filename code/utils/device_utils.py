# device_utils.py (새 파일로 빼도 OK)
import torch, os

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def setup_cuda_flags():
    # TF32: Ampere+에서 속도↑, 정확도 거의 동일. Blackwell에서도 권장
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # 고정 크기 배치에 유리
    # 선택: 더 공격적으로
    try:
        torch.set_float32_matmul_precision("high")  # "medium"/"high"
    except Exception:
        pass

def pick_amp_dtype() -> torch.dtype:
    # Blackwell(차세대) 포함 대부분에서 bf16 안정/빠름
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # bf16은 Ampere(8.0) 이상. 그 미만이면 fp16로
        if (major, minor) >= (8, 0):
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32

class AutocastAmp:
    """bf16이면 GradScaler 불필요, fp16이면 스케일러 사용"""
    def __init__(self, enabled=True):
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.dtype = pick_amp_dtype()
        # bf16은 스케일러 불필요
        self.use_scaler = (self.enabled and self.dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

    def autocast(self):
        return torch.cuda.amp.autocast(enabled=self.enabled, dtype=self.dtype)

    def scale(self, loss):
        return self.scaler.scale(loss) if self.use_scaler else loss

    def unscale_(self, optimizer):
        if self.use_scaler:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer):
        if self.use_scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
