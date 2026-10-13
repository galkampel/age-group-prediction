"""Analysis of a fitted model.

:mod:`~age_group_prediction.inspection.bootstrap`
    The bootstrap mean and percentile interval of a metric on an evaluation
    set, resampling rows or whole groups.
:mod:`~age_group_prediction.inspection.permutation`
    Permutation importance per raw input column, with a scorer built from a
    :class:`~age_group_prediction.scoring.Metric`.

The design and its decisions are in ``docs/MODEL_INSPECTION_PLAN.md`` and
``docs/FEATURE_IMPORTANCE_PLAN.md``.
"""

from __future__ import annotations

from .bootstrap import BootstrapEvaluator, BootstrapResult
from .permutation import PermutationImportance

__all__ = ["BootstrapEvaluator", "BootstrapResult", "PermutationImportance"]
