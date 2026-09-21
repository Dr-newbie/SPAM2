# -*- coding: utf-8 -*-
"""
Weights & Biases 트래킹 헬퍼.

설계 원칙:
  1) wandb가 설치되어 있지 않거나 --wandb 를 주지 않으면 **전부 no-op**.
     → 기존 실행 경로/결과는 전혀 바뀌지 않는다.
  2) 로깅 실패(네트워크 끊김, 인증 실패 등)가 학습을 죽이지 않는다.
     실험 트래킹이 실험 자체를 중단시키면 안 되므로 모든 호출을 감싼다.
  3) 상태는 모듈 전역 하나(_RUN)로 두어, 학습 루프가 run 객체를
     인자로 들고 다니지 않아도 되게 한다.

사용:
    from code.utils.wandb_utils import wandb_init, wandb_log, wandb_finish
    wandb_init(enabled=args.wandb, project=..., config={...})
    wandb_log({"train/loss": 0.1}, step=global_step)
    wandb_finish()
"""

from typing import Optional, Dict, Any

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

_RUN = None
_WARNED = False


def wandb_available() -> bool:
    return _wandb is not None


def _warn_once(msg: str):
    global _WARNED
    if not _WARNED:
        print(f"[wandb] {msg}")
        _WARNED = True


def wandb_init(enabled: bool = False,
               project: str = "spam-xenium",
               entity: Optional[str] = None,
               run_name: Optional[str] = None,
               mode: str = "online",
               tags: Optional[list] = None,
               notes: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None):
    """wandb run 시작. enabled=False면 아무것도 하지 않고 None 반환."""
    global _RUN
    if not enabled:
        return None
    if _wandb is None:
        _warn_once("wandb가 설치되어 있지 않아 트래킹을 건너뜁니다. "
                   "설치: pip install wandb")
        return None
    try:
        _RUN = _wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,              # online / offline / disabled
            tags=tags or [],
            notes=notes,
            config=config or {},
        )
        print(f"[wandb] run 시작: {getattr(_RUN, 'name', '?')} "
              f"(project={project}, mode={mode})")
        print(f"[wandb] URL: {getattr(_RUN, 'url', 'n/a')}")
        return _RUN
    except Exception as e:
        _warn_once(f"init 실패 → 트래킹 없이 계속 진행합니다: {type(e).__name__}: {e}")
        _RUN = None
        return None


def wandb_log(metrics: Dict[str, Any], step: Optional[int] = None):
    """활성 run이 없으면 no-op. 실패해도 학습을 중단시키지 않는다."""
    if _RUN is None:
        return
    try:
        _RUN.log(metrics, step=step)
    except Exception as e:
        _warn_once(f"log 실패(이후 무시): {type(e).__name__}: {e}")


def wandb_summary(**kwargs):
    """run summary(최종 요약값) 갱신."""
    if _RUN is None:
        return
    try:
        for k, v in kwargs.items():
            _RUN.summary[k] = v
    except Exception as e:
        _warn_once(f"summary 실패: {type(e).__name__}: {e}")


def wandb_save_file(path: str):
    """체크포인트 등 산출물을 run에 첨부."""
    if _RUN is None:
        return
    try:
        _RUN.save(path, policy="now")
    except Exception as e:
        _warn_once(f"save 실패: {type(e).__name__}: {e}")


def wandb_finish():
    global _RUN
    if _RUN is None:
        return
    try:
        _RUN.finish()
    except Exception as e:
        _warn_once(f"finish 실패: {type(e).__name__}: {e}")
    finally:
        _RUN = None
