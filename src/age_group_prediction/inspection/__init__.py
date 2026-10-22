"""Analysis of a fitted model.

:mod:`~age_group_prediction.inspection.bootstrap`
    The bootstrap mean and percentile interval of every metric named per
    call on an evaluation set, resampling rows or whole groups.
:mod:`~age_group_prediction.inspection.permutation`
    Permutation importance per raw input column, with a scorer built from a
    :class:`~age_group_prediction.scoring.Metric`.
:mod:`~age_group_prediction.inspection.shap_values`
    SHAP values of the rate per raw input column, with the cohort output
    names set.

The design and its decisions are in ``docs/MODEL_INSPECTION_PLAN.md`` and
``docs/FEATURE_IMPORTANCE_PLAN.md``.
"""

from __future__ import annotations

from .bootstrap import BootstrapEvaluator
from .permutation import PermutationImportance
from .shap_values import Method, ShapExplainer

__all__ = [
    "BootstrapEvaluator",
    "Method",
    "PermutationImportance",
    "ShapExplainer",
]
