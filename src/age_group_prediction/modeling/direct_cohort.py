"""Model A: one regressor for one cohort's child count, the exposure as a weighted rate."""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ..feature_engineering import FeatureTransformer
from .base import BaseAgeGroupModel

__all__ = ["DirectCohortModel", "Regressor"]


class Regressor(Protocol):
    """What ``estimator`` must do: scikit-learn's regressor API with ``sample_weight``."""

    def fit(
        self, X: pd.DataFrame, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self: ...

    def predict(self, X: pd.DataFrame) -> ArrayLike: ...


class DirectCohortModel(BaseAgeGroupModel):
    """Any regressor with a Poisson or Gaussian loss, fitted on one cohort.

    ``X`` is the raw table when ``feature_transformer`` is given, otherwise the
    finished design matrix. Cohorts are independent, so each gets its own instance.
    ``estimator`` is always given, and carries its own loss and
    hyperparameters, e.g. ``LGBMRegressor(objective="poisson")``,
    ``HistGradientBoostingRegressor(loss="poisson")`` or
    ``PoissonRegressor()``. A tuner reaches its settings by nested names
    (``estimator__n_estimators``) through ``set_params``; nothing is searched
    in ``fit``. LightGBM ignores ``subsample`` unless ``subsample_freq >= 1``
    is set on the estimator too.

    ``use_exposure`` makes the raw exposure (apartments), passed to ``fit`` and
    ``predict``, scale the mean: the estimator learns the rate per apartment
    from ``y / exposure`` with ``sample_weight=exposure``, and ``predict``
    multiplies it back by the exposure to give a count. For a Poisson loss this
    is the offset model ``log(exposure)`` itself, since the two negative
    log-likelihoods differ by a constant, ``L_offset = L_rate - sum(y log
    exposure)``; for a Gaussian loss it is least squares of the count with a
    variance proportional to the exposure. Both derivations are in
    ``docs/DIRECT_COHORT_MODEL.md`` §0.1. The equivalence is of the
    likelihoods: a penalized scikit-learn GLM (``PoissonRegressor(alpha=…)``)
    normalizes ``sample_weight`` to sum to one, so its ``alpha`` acts as
    ``alpha * mean(exposure)`` in the offset model; LightGBM and
    ``HistGradientBoostingRegressor`` use the weights as given. Unlike
    LightGBM's ``init_score``, it needs only ``sample_weight``: a regressor
    without one fails at ``fit`` with its own error. With
    ``use_exposure=False`` a passed exposure is ignored, so a caller can pass
    one exposure to every model, and a tuner can compare with and without.
    """

    def __init__(
        self,
        *,
        estimator: Regressor,
        use_exposure: bool = False,
        feature_transformer: FeatureTransformer | None = None,
    ) -> None:
        # Stored verbatim, unvalidated: set_params assigns attributes without
        # re-entering __init__, so fit is where the configuration is checked.
        self.estimator = estimator
        self.use_exposure = use_exposure
        self.feature_transformer = feature_transformer

    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit a copy of ``estimator`` on ``X`` and ``y``; ``exposure`` is raw, not its log.

        What the loss cannot fit is left to the estimator: LightGBM, for one,
        rejects an all-zero ``y`` under a Poisson loss.
        """
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure)
        feature_transformer, X = self._fit_features(X, y)
        # A copy, so the caller's template stays unfitted across folds.
        estimator: Regressor = clone(self.estimator)
        if exposure_values is None:
            estimator.fit(X, y)
        else:
            # The rate per apartment, weighted by the apartments: the offset
            # model's likelihood for a Poisson loss, least squares of the count
            # for a Gaussian one; any regressor that takes sample_weight.
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
