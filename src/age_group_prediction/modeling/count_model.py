"""Model A, and Model 2's total: one regressor for one count column, the exposure as a weighted rate or an exposure argument."""

from __future__ import annotations

from typing import Protocol, Self, TypeGuard

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from ..feature_engineering import FeatureTransformer
from .base import BaseAgeGroupModel

__all__ = ["CountModel", "ExposureRegressor", "Regressor"]


class Regressor(Protocol):
    """What ``estimator`` must do: scikit-learn's regressor API with ``sample_weight``."""

    def fit(
        self, X: pd.DataFrame, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self: ...

    def predict(self, X: pd.DataFrame) -> ArrayLike: ...


class ExposureRegressor(Protocol):
    """A regressor that takes the raw exposure itself, e.g. ``NegativeBinomialRegressor``."""

    def fit(
        self, X: pd.DataFrame, y: ArrayLike, exposure: ArrayLike | None = None
    ) -> Self: ...

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> ArrayLike: ...


# scikit-learn's own test of an estimator's fit signature (as BaggingRegressor
# uses it for sample_weight). TypeGuard, not TypeIs: mypy treats the two
# protocols as overlapping, so TypeIs would not narrow a case to its type.
def _takes_exposure(
    estimator: Regressor | ExposureRegressor,
) -> TypeGuard[ExposureRegressor]:
    return has_fit_parameter(estimator, "exposure")


def _takes_sample_weight(
    estimator: Regressor | ExposureRegressor,
) -> TypeGuard[Regressor]:
    return has_fit_parameter(estimator, "sample_weight")


class CountModel(BaseAgeGroupModel):
    """One regressor, with a Poisson, NB2 or Gaussian loss, for one count column: a cohort's or the total.

    ``estimator`` carries its own loss and hyperparameters (``estimator__…``
    in ``set_params``). ``X`` is the raw table when ``feature_transformer`` is
    given, otherwise the finished design matrix.

    With ``use_exposure``, the raw exposure passed to ``fit`` and ``predict``
    scales the mean, by the first case the estimator's ``fit`` matches:

    - it takes ``exposure`` (``ExposureRegressor``): the exposure as is, at
      ``fit`` and ``predict``;
    - it takes ``sample_weight`` (``Regressor``): the weighted rate, ``y /
      exposure`` with ``sample_weight=exposure``, the prediction multiplied by
      the exposure. For a Poisson loss this is the offset model
      ``log(exposure)`` (``docs/DIRECT_COHORT_MODEL.md`` §0.1), for NB2 it is
      not, hence the first case. A penalized scikit-learn GLM normalizes
      ``sample_weight``, so its ``alpha`` acts as ``alpha * mean(exposure)``;
    - neither: ``fit`` raises ``TypeError``. The cases are read from the
      estimator's own ``fit`` signature (``has_fit_parameter``), so a wrapper
      whose ``fit`` takes ``**fit_params`` (``Pipeline``,
      ``TransformedTargetRegressor``, ``GridSearchCV``) is refused even if it
      would forward ``sample_weight``: pass the regressor itself.

    Without ``use_exposure`` a passed exposure is ignored.
    """

    def __init__(
        self,
        *,
        estimator: Regressor | ExposureRegressor,
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
        estimator: Regressor | ExposureRegressor = clone(self.estimator)
        # The cases of the class docstring, each named; any other raises.
        if exposure_values is None:
            estimator.fit(X, y)
        elif _takes_exposure(estimator):
            estimator.fit(X, y, exposure=exposure_values)
        elif _takes_sample_weight(estimator):
            # The rate per apartment, weighted by the apartments.
            estimator.fit(X, y / exposure_values, sample_weight=exposure_values)
        else:
            raise TypeError(
                f"{type(estimator).__name__}.fit takes neither `exposure` nor "
                "`sample_weight`, so use_exposure=True cannot be applied"
            )
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
        X = self._transform_features(X)
        estimator = self.estimator_
        if exposure_values is None:
            prediction = estimator.predict(X)
        elif _takes_exposure(estimator):
            prediction = estimator.predict(X, exposure=exposure_values)
        elif _takes_sample_weight(estimator):
            prediction = np.asarray(estimator.predict(X), dtype=float) * exposure_values
        else:
            # Unreachable after a fit that succeeded: fit raised for it.
            raise TypeError(
                f"{type(estimator).__name__}.fit takes neither `exposure` nor "
                "`sample_weight`"
            )
        return np.asarray(prediction, dtype=float)
