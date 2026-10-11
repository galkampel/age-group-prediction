"""Analysis of a fitted model.

:mod:`~age_group_prediction.inspection.bootstrap`
    The bootstrap mean and percentile interval of a metric on an evaluation
    set, resampling rows or whole groups.

The design and its decisions are in ``docs/MODEL_INSPECTION_PLAN.md``.
"""

from __future__ import annotations

from .bootstrap import BootstrapEvaluator, BootstrapResult

__all__ = ["BootstrapEvaluator", "BootstrapResult"]
