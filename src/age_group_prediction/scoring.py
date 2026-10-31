"""Metrics: a named scoring function of ``(y_true, y_pred)`` and its direction.

Any callable works, from :mod:`sklearn.metrics` or custom, so the direction is
stored rather than assumed: a tuner reads ``direction``, in Optuna's terms,
to know whether to minimize the value (a loss) or maximize it (a score).

The functions below pair ``y_true`` and ``y_pred`` by position and reject a
NaN or an infinity. The count metrics score the cohort matrix,
``(n_samples, n_cohorts)``, or a single target, ``(n_samples,)``, and pool
every cell; the log loss scores the matrix of at least two cohorts. With
``cohort``, a column name of ``y_true``, they score that cohort alone: its
column, or for the log loss the cohort against the rest of its sample.
``evaluate`` scores a list of metrics for each cohort asked for, and the
composite. Shared by :mod:`~age_group_prediction.modeling`,
:mod:`~age_group_prediction.inspection` and the experiment notebook, which
build their metrics from these. Not the ``Metric`` protocol the package root
exports from the old ``metrics.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    root_mean_squared_error,
)

from .utils import check_paired_arrays, column_position

__all__ = [
    "Direction",
    "Metric",
    "evaluate",
    "mean_absolute_error_loss",
    "mean_bias_error",
    "mean_poisson_deviance_loss",
    "multinomial_log_loss",
    "root_mean_squared_error_loss",
    "wmape_loss",
]


type Direction = Literal["minimize", "maximize"]  # Optuna's terms: a loss, a score


@dataclass(frozen=True)
class Metric:
    """A scoring function ``function(y_true, y_pred) -> float``, its name and direction.

    Calling the metric applies the function: ``metric(y_true, y_pred)``.
    """

    name: str
    # (y_true, y_pred) for the whole of y; the functions below also take cohort=.
    function: Callable[..., float | np.floating]
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

    def __call__(
        self, y_true: ArrayLike, y_pred: ArrayLike, *, cohort: str | None = None
    ) -> float:
        """``function(y_true, y_pred)`` as a Python float, not a NumPy scalar.

        ``cohort`` is passed on only when given, so a two-argument function
        scores the whole of ``y`` and raises ``TypeError`` if asked for a cohort.
        """
        if cohort is None:
            return float(self.function(y_true, y_pred))
        return float(self.function(y_true, y_pred, cohort=cohort))


def multinomial_log_loss(
    y_true: ArrayLike, y_pred: ArrayLike, *, cohort: str | None = None
) -> float:
    """The mean negative log probability per child: ``−Σ n_sk log p_sk / Σ n_sk``.

    ``y_true`` holds the cohort counts, ``y_pred`` the predicted cohorts, both
    ``(n_samples, n_cohorts)`` with at least two cohorts. Each row of ``y_pred``
    is divided by its sum, so expected counts score as their proportions. Per
    child, not per sample: scikit-learn's ``log_loss`` on one row per (sample,
    cohort) with children, weighted by the count. With ``cohort``, the cohort
    against the sum of the others, ``[n_k, Σn − n_k]`` and ``[ŷ_k, Σŷ − ŷ_k]``:
    the binomial log loss of its share, not a part of the multinomial one.
    """
    counts, predictions = check_paired_arrays(y_true, y_pred)
    # One cohort normalizes to probability 1 and would score 0 whatever the
    # prediction; checked before cohort=, whose rest column would be all zeros.
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError(
            "y_true and y_pred must be (n_samples, n_cohorts) matrices with at least "
            f"two cohorts; score one cohort with cohort=, got shape {counts.shape}"
        )
    if cohort is not None:
        position = column_position(y_true, cohort)
        counts = np.column_stack(
            [counts[:, position], counts.sum(axis=1) - counts[:, position]]
        )
        predictions = np.column_stack(
            [
                predictions[:, position],
                predictions.sum(axis=1) - predictions[:, position],
            ]
        )
    probabilities = predictions / predictions.sum(axis=1, keepdims=True)
    sample_positions, cohort_positions = np.nonzero(counts)
    return float(
        log_loss(
            y_true=cohort_positions,
            y_proba=probabilities[sample_positions],
            sample_weight=counts[sample_positions, cohort_positions],
            # Every cohort, also one without a child in y_true.
            labels=range(counts.shape[1]),
        )
    )


def mean_poisson_deviance_loss(
    y_true: ArrayLike, y_pred: ArrayLike, *, cohort: str | None = None
) -> float:
    """The mean Poisson deviance over the cells.

    For the cohort matrix, each cohort weighs as its own ``mean_poisson_deviance``
    would: the value is the mean of the per-cohort deviances.
    """
    counts, predictions = check_paired_arrays(y_true, y_pred)
    if cohort is not None:
        position = column_position(y_true, cohort)
        counts, predictions = counts[:, position], predictions[:, position]
    return float(mean_poisson_deviance(counts.ravel(), predictions.ravel()))


def mean_absolute_error_loss(
    y_true: ArrayLike, y_pred: ArrayLike, *, cohort: str | None = None
) -> float:
    """The mean absolute error over the cells, ``mean|ŷ − y|``."""
    counts, predictions = check_paired_arrays(y_true, y_pred)
    if cohort is not None:
        position = column_position(y_true, cohort)
        counts, predictions = counts[:, position], predictions[:, position]
    return float(mean_absolute_error(counts.ravel(), predictions.ravel()))


def root_mean_squared_error_loss(
    y_true: ArrayLike, y_pred: ArrayLike, *, cohort: str | None = None
) -> float:
    """The root of the mean squared error over the cells, ``sqrt(mean (ŷ − y)²)``.

    Pooled: for the cohort matrix, not the mean of the per-cohort RMSEs.
    """
    counts, predictions = check_paired_arrays(y_true, y_pred)
    if cohort is not None:
        position = column_position(y_true, cohort)
        counts, predictions = counts[:, position], predictions[:, position]
    return float(root_mean_squared_error(counts.ravel(), predictions.ravel()))


def wmape_loss(
    y_true: ArrayLike, y_pred: ArrayLike, *, cohort: str | None = None
) -> float:
    """The weighted mean absolute percentage error over the cells, ``Σ|ŷ − y| / Σ|y|``.

    The absolute error as a share of the children; not finite when there are
    none (numpy warns).
    """
    counts, predictions = check_paired_arrays(y_true, y_pred)
    if cohort is not None:
        position = column_position(y_true, cohort)
        counts, predictions = counts[:, position], predictions[:, position]
    return float(np.abs(predictions - counts).sum() / np.abs(counts).sum())


def mean_bias_error(
    y_true: ArrayLike, y_pred: ArrayLike, *, cohort: str | None = None
) -> float:
    """The mean signed error over the cells, ``mean(ŷ − y)``: 0 is unbiased.

    Signed, so neither direction is better; a model is not tuned on it.
    """
    counts, predictions = check_paired_arrays(y_true, y_pred)
    if cohort is not None:
        position = column_position(y_true, cohort)
        counts, predictions = counts[:, position], predictions[:, position]
    return float(np.mean(predictions - counts))


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metrics: Sequence[Metric],
    cohorts: str | Sequence[str] = "composite",
) -> pd.DataFrame:
    """Each metric's score (a column, by the metric's name) for each cohort (a row).

    A cohort is a column name of ``y_true``, scored alone (``cohort=``), or
    ``"composite"``, the whole of ``y_true``; a single name gives one row.
    """
    if isinstance(cohorts, str):
        cohorts = [cohorts]
    return pd.DataFrame(
        [
            [
                metric(y_true, y_pred, cohort=None if cohort == "composite" else cohort)
                for metric in metrics
            ]
            for cohort in cohorts
        ],
        index=pd.Index(cohorts, name="cohort"),
        columns=pd.Index([metric.name for metric in metrics], name="metric"),
    )
