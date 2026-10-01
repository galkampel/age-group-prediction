"""Metrics: a named scoring function of ``(y_true, y_pred)`` and its direction.

Any callable works, from :mod:`sklearn.metrics` or custom, so the direction is
stored rather than assumed: a tuner reads ``greater_is_better`` to know whether
to maximize the value or its negation.

Shared by :mod:`~age_group_prediction.modeling` and
:mod:`~age_group_prediction.hyperparameter_tuning`. Not the ``Metric`` protocol
the package root exports from the old ``metrics.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlogy
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    root_mean_squared_error,
)

__all__ = [
    "COHORT_LOG_LOSS",
    "MAE",
    "POISSON_DEVIANCE",
    "RMSE",
    "Metric",
    "cohort_log_loss",
]


@dataclass(frozen=True)
class Metric:
    """A scoring function ``function(y_true, y_pred) -> float``, its name and direction."""

    name: str
    function: Callable[[ArrayLike, ArrayLike], float | np.floating]
    greater_is_better: bool = False

    def __post_init__(self) -> None:
        # Checked here, because a wrong value would otherwise surface only when
        # scoring, or never: an int direction would silently flip a tuner's sign.
        if not self.name:
            raise ValueError("name must not be empty")
        if not callable(self.function):
            raise TypeError(f"function must be callable, got {self.function!r}")
        if not isinstance(self.greater_is_better, bool):
            raise TypeError(
                f"greater_is_better must be a bool, got {self.greater_is_better!r}"
            )


def cohort_log_loss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """The mean negative log probability per child: ``−Σ n_bk log p_bk / Σ n_bk``.

    ``y_true`` holds the cohort counts, one row per building and one column per
    cohort; ``y_pred`` the predicted cohorts in the same layout, paired by
    position. Each row of ``y_pred`` is divided by its sum, so expected counts
    per cohort score as their proportions and probabilities are unchanged. Per
    child, not per building: a building with no children adds nothing, given
    a positive prediction row (hence ``xlogy``, where a plain ``0 · log 0`` is
    nan). Equals scikit-learn's ``log_loss`` on one row per child.
    """
    counts = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    # numpy would broadcast one column, or one row, over every cohort silently.
    if counts.shape != predictions.shape:
        raise ValueError(
            "y_true and y_pred must both be (buildings, cohorts) arrays of the "
            f"same shape, got {counts.shape} and {predictions.shape}"
        )
    probabilities = predictions / predictions.sum(axis=1, keepdims=True)
    return -float(xlogy(counts, probabilities).sum() / counts.sum())


POISSON_DEVIANCE = Metric("poisson_deviance", mean_poisson_deviance)
RMSE = Metric("rmse", root_mean_squared_error)
MAE = Metric("mae", mean_absolute_error)
COHORT_LOG_LOSS = Metric("cohort_log_loss", cohort_log_loss)
