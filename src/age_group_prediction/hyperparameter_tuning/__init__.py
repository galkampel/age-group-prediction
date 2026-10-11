"""Optuna hyperparameter tuning in independent parts.

:mod:`~age_group_prediction.hyperparameter_tuning.parameters`
    Which values each hyperparameter may take.
:mod:`~age_group_prediction.hyperparameter_tuning.evaluator`
    How one trial is scored: its parameters are set once, then every
    cross-validation fold is fitted and scored.
:mod:`~age_group_prediction.hyperparameter_tuning.aggregation`
    How a trial's fold scores are combined into one value.
:mod:`~age_group_prediction.hyperparameter_tuning.study`
    How Optuna searches, and which trial won.

Each part can be replaced without touching the others. The design and its
decisions are in ``docs/HYPERPARAMETER_TUNING_PLAN.md``.
"""

from __future__ import annotations

from .aggregation import (
    Aggregator,
    Mean,
    UpperBound,
    WeightedMean,
    corrected_std_error,
)
from .evaluator import CVHyperparameterEvaluator
from .parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    Parameter,
)
from .study import HyperparameterStudy, TrialRecord, TuningResult

__all__ = [
    "Aggregator",
    "CVHyperparameterEvaluator",
    "CategoricalParameter",
    "FloatParameter",
    "HyperparameterStudy",
    "IntParameter",
    "Mean",
    "Parameter",
    "TrialRecord",
    "TuningResult",
    "UpperBound",
    "WeightedMean",
    "corrected_std_error",
]
