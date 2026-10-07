"""Model A, and Model 2's total: one regressor for one count column, the exposure as a weighted rate."""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ..feature_engineering import FeatureTransformer
from .base import BaseAgeGroupModel

__all__ = ["CountModel", "Regressor"]


class Regressor(Protocol):
    """What ``estimator`` must do: scikit-learn's regressor API with ``sample_weight``."""

    def fit(
        self, X: pd.DataFrame, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self: ...

    def predict(self, X: pd.DataFrame) -> ArrayLike: ...


class CountModel(BaseAgeGroupModel):
    """One regressor, with a Poisson or Gaussian loss, for one count column: a cohort's or the total.

    ``estimator`` carries its own loss and hyperparameters (``estimator__…``
    in ``set_params``). ``X`` is the raw table when ``feature_transformer`` is
    given, otherwise the finished design matrix.

    With ``use_exposure``, the raw exposure passed to ``fit`` and ``predict``
    scales the mean: ``fit`` regresses ``y / exposure`` with
    ``sample_weight=exposure``, and ``predict`` multiplies by the exposure. For
    a Poisson loss this is the offset model ``log(exposure)``
    (``docs/DIRECT_COHORT_MODEL.md`` §0.1). A penalized scikit-learn GLM
    normalizes ``sample_weight``, so its ``alpha`` acts as
    ``alpha * mean(exposure)``. Without ``use_exposure`` a passed exposure is
    ignored.
    """

    def __init__(
        self,
        *,
        estimator: Regressor,
        use_exposure: bool = False,
        feature_transformer: FeatureTransformer | None = None,
    ) -> None:
        # Stored verbatim: set_params assigns attributes without re-entering here.
        self.estimator = estimator
        self.use_exposure = use_exposure
        self.feature_transformer = feature_transformer

    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit a copy of ``estimator`` on ``X`` and ``y``; ``exposure`` is raw, not its log."""
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure)
        feature_transformer, X = self._fit_features(X, y)
        # A copy, so the caller's template stays unfitted across folds.
        estimator: Regressor = clone(self.estimator)
        if exposure_values is None:
            estimator.fit(X, y)
        else:
            # The rate per apartment, weighted by the apartments.
            estimator.fit(X, y / exposure_values, sample_weight=exposure_values)
        # Set together, only once fitting succeeded.
        self.estimator_ = estimator
        self.use_exposure_: bool = exposure_values is not None
        self.feature_transformer_ = feature_transformer
        return self

    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> np.ndarray:
        """The predicted mean count for each row of ``X``."""
        check_is_fitted(self)
        # Follows how the model was fitted, not the current use_exposure, which
        # set_params may have changed since.
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure_)
        prediction = np.asarray(
            self.estimator_.predict(self._transform_features(X)), dtype=float
        )
        if exposure_values is None:
            return prediction
        return prediction * exposure_values
