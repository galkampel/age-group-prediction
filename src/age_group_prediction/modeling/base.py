"""The contract every age-group model shares: fit, predict and evaluate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from ..scoring import Metric

__all__ = ["BaseAgeGroupModel"]


class BaseAgeGroupModel(BaseEstimator, ABC):
    """A scikit-learn-style model of one count target.

    Subclasses take their settings in ``__init__`` and store them verbatim, so
    ``get_params`` and ``set_params`` (inherited from ``BaseEstimator``) and
    ``sklearn.base.clone`` work, and a tuner can build a fresh model per trial.
    Data are only ever method arguments (``X``, ``y``, and ``exposure`` where a
    model uses one), and fitted state lives in trailing-underscore attributes.
    ``fit`` replaces all fitted state, so a refit equals a fresh fit: a tuner
    reuses one copy across the folds of a trial.
    """

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        """Learn from the rows of ``X`` and their targets ``y``.

        ``exposure`` is each row's raw exposure ``n``, for a model with an
        offset. A model raises if one is passed when it uses none, or missing
        when it needs one, rather than silently ignoring or dropping it.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> np.ndarray:
        """The predicted mean for each row of ``X``; ``exposure`` as in :meth:`fit`."""

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, metric: Metric) -> float:
        """Score ``y_pred`` against ``y_true`` with one metric.

        A method rather than a free function so a model can change how it is
        scored. The cast turns a custom metric's numpy scalar into a float.
        """
        return float(metric.function(y_true, y_pred))
