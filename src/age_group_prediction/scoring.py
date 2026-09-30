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
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    root_mean_squared_error,
)

__all__ = ["MAE", "POISSON_DEVIANCE", "RMSE", "Metric"]


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


POISSON_DEVIANCE = Metric("poisson_deviance", mean_poisson_deviance)
RMSE = Metric("rmse", root_mean_squared_error)
MAE = Metric("mae", mean_absolute_error)
