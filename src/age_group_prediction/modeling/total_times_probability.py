"""Model 2: each cohort's mean as the predicted total times the cohort's predicted probability."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from .base import BaseAgeGroupModel
from .cohort_probability import CohortProbabilityModel

__all__ = ["TotalTimesProbabilityModel"]


class TotalTimesProbabilityModel(BaseAgeGroupModel):
    """Predict each cohort as the total model's prediction times the probability model's.

    ``total_model`` (e.g. a ``CountModel``) is fitted on the row sums of ``y``,
    ``probability_model`` on ``y``, the cohort counts; both take the raw table
    ``X`` and the same ``exposure``. Nested ``set_params`` names reach both
    (``total_model__estimator__alpha``, ``probability_model__calibration_method``).
    ``predict`` returns a DataFrame with ``y``'s columns at fit, indexed like
    ``X``, each row summing to the predicted total.
    """

    def __init__(
        self,
        *,
        total_model: BaseAgeGroupModel,
        probability_model: CohortProbabilityModel,
    ) -> None:
        # Stored verbatim: set_params assigns attributes without re-entering here.
        self.total_model = total_model
        self.probability_model = probability_model

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit a copy of each model: the total model on ``y``'s row sums, the probability model on ``y``."""
        total_model = clone(self.total_model).fit(X, y.sum(axis=1), exposure=exposure)
        probability_model = clone(self.probability_model).fit(X, y, exposure=exposure)
        # Set together, only once both succeeded.
        self.total_model_: BaseAgeGroupModel = total_model
        self.probability_model_: CohortProbabilityModel = probability_model
        return self

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> pd.DataFrame:
        """The predicted mean of every cohort for each row of ``X``."""
        check_is_fitted(self)
        # As an array: a total with its own index would be realigned to X's,
        # into NaN, silently.
        total = np.asarray(self.total_model_.predict(X, exposure=exposure), dtype=float)
        probabilities = self.probability_model_.predict(X, exposure=exposure)
        return probabilities * total[:, None]
