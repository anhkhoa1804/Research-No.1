"""Open-vocabulary relation grounding research code.

This package contains the training pipeline, datasets, models, and evaluation utilities.
"""

from .config import TrainConfig
from .phase_audit import pure_phase_audit, summarize_pure_phase_audit

__all__ = [
    "TrainConfig",
    "pure_phase_audit",
    "summarize_pure_phase_audit",
]
