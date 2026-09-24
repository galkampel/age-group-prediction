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
    ``get_params``, ``set_params`` and ``clone`` work and a tuner can build a
    fresh model per trial. Data are only ever method arguments, and fitted
    state lives in trailing-underscore attributes.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Learn from the rows of ``X`` and their targets ``y``."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """The predicted mean for each row of ``X``."""

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, metric: Metric) -> float:
        """Score ``y_pred`` against ``y_true`` with one metric.

        A method rather than a free function so a model can change how it is
        scored. The cast turns a custom metric's numpy scalar into a float.
        """
        return float(metric.function(y_true, y_pred))
