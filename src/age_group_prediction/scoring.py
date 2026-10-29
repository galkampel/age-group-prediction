"""Metrics: a named scoring function of ``(y_true, y_pred)`` and its direction.

Any callable works, from :mod:`sklearn.metrics` or custom, so the direction is
stored rather than assumed: a tuner reads ``direction``, in Optuna's terms,
to know whether to minimize the value (a loss) or maximize it (a score).

Shared by :mod:`~age_group_prediction.modeling`,
:mod:`~age_group_prediction.inspection` and the experiment notebook, which
build their metrics from these. Not the ``Metric`` protocol the package root
exports from the old ``metrics.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlogy
from sklearn.metrics import mean_poisson_deviance

__all__ = [
    "Direction",
    "Metric",
    "cohort_log_loss",
    "cohort_poisson_deviance",
]


type Direction = Literal["minimize", "maximize"]  # Optuna's terms: a loss, a score


@dataclass(frozen=True)
class Metric:
    """A scoring function ``function(y_true, y_pred) -> float``, its name and direction.

    Calling the metric applies the function: ``metric(y_true, y_pred)``.
    """

    name: str
    function: Callable[[ArrayLike, ArrayLike], float | np.floating]
    direction: Direction = "minimize"

    def __post_init__(self) -> None:
        # Checked here, because a wrong value would otherwise surface only when
        # scoring, or never: a misspelled direction would silently flip a tuner's sign.
        if not self.name:
            raise ValueError("name must not be empty")
        if not callable(self.function):
            raise TypeError(f"function must be callable, got {self.function!r}")
        if self.direction not in get_args(Direction.__value__):
            raise ValueError(
                f"direction must be 'minimize' or 'maximize', got {self.direction!r}"
            )

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """``function(y_true, y_pred)`` as a Python float, not a NumPy scalar."""
        return float(self.function(y_true, y_pred))


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


def cohort_poisson_deviance(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """The mean Poisson deviance over every (building, cohort) cell.

    ``y_true`` holds the cohort counts, one row per building and one column per
    cohort; ``y_pred`` the predicted counts in the same layout, paired by
    position. Each cohort weighs as its own ``mean_poisson_deviance`` would: the
    value is the mean of the per-cohort deviances.
    """
    counts = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    # A transposed or flattened prediction has as many cells, so after ravel it
    # would be scored mispaired, silently.
    if counts.shape != predictions.shape:
        raise ValueError(
            "y_true and y_pred must both be (buildings, cohorts) arrays of the "
            f"same shape, got {counts.shape} and {predictions.shape}"
        )
    return float(mean_poisson_deviance(counts.ravel(), predictions.ravel()))
