from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


try:
    import transformer_engine.pytorch as te  # type: ignore
except Exception:
    te = None


def fp8_available() -> bool:
    return te is not None and hasattr(te, "fp8_autocast")


def fp8_supported_on_current_device() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return int(major) >= 9


def fp8_autocast(enabled: bool = False, **kwargs: Any):
    if bool(enabled) and fp8_available():
        return te.fp8_autocast(enabled=True, **kwargs)
    return nullcontext()
