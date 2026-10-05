"""Model 2's first half: a penalized Poisson or NB2 GLM for a building's total children."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted, check_X_y, validate_data
from torch.distributions import NegativeBinomial, Poisson

from ..feature_engineering import FeatureTransformer
from .base import BaseAgeGroupModel
from .optimization import Minimizer, Solver, single_threaded_torch

__all__ = ["Family", "TotalChildrenModel"]

type Family = Literal["poisson", "nb2"]

# α → 0 is the Poisson limit, the best NB2 fit for data without overdispersion:
# unbounded, log α runs off (to −16…−23 on Poisson draws), where
# lgamma(1/α + y) − lgamma(1/α) loses its precision and BFGS fails.
_MIN_DISPERSION = 1e-6
# Where log α starts: the dispersion measured on the simulated totals.
_START_DISPERSION = 0.1


class TotalChildrenModel(BaseAgeGroupModel):
    """A count regression of the total, ``log μ = offset + b + Xβ``.

    ``X`` is the raw table when ``feature_transformer`` is given, otherwise
    the finished design matrix. With ``use_exposure`` (the default:
    Model 2's specification) the offset is ``log(exposure)``, the raw exposure
    passed to ``fit`` and ``predict``, so ``b + Xβ`` is the log rate per
    apartment; a forgotten exposure raises. With ``use_exposure=False`` there
    is no offset and a passed exposure is ignored.

    ``family`` is the count distribution: ``"poisson"``, or ``"nb2"``, the
    negative binomial with variance ``μ(1 + αμ)``, whose dispersion ``α`` is
    fitted with ``b`` and ``β`` and stored as ``dispersion_``. ``α`` is kept at
    or above 1e-6, the Poisson limit in practice, so NB2 takes only
    ``solver="lbfgs"``, the solver with bounds. ``predict`` is the mean for
    either family.

    ``l2_penalty`` multiplies ``½‖β‖²``, added to the mean negative
    log-likelihood per building, so its meaning does not change with the
    number of buildings; ``b`` and ``α`` are not penalized. ``solver`` is
    the scipy method, named as scikit-learn names it; ``tol`` bounds the
    gradient of the objective at the fit. A fit that does not converge raises
    rather than returning a wrong point.
    """

    def __init__(
        self,
        *,
        family: Family = "poisson",
        solver: Solver = "lbfgs",
        use_exposure: bool = True,
        l2_penalty: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        feature_transformer: FeatureTransformer | None = None,
    ) -> None:
        self.family = family
        self.solver = solver
        self.use_exposure = use_exposure
        self.l2_penalty = l2_penalty
        self.max_iter = max_iter
        self.tol = tol
        self.feature_transformer = feature_transformer

    @staticmethod
    def _objective(
        parameters: np.ndarray,
        X: torch.Tensor,
        y: torch.Tensor,
        offset: torch.Tensor,
        l2_penalty: float,
        family: Family,
    ) -> tuple[float, np.ndarray]:
        """The penalized mean negative log-likelihood and its gradient.

        scipy's optimizer calls it with a numpy point ``[b, β]``, and
        ``[b, β, log α]`` for NB2; torch gives the vectorized likelihood and
        its exact gradient.
        """
        params = torch.tensor(parameters, requires_grad=True)
        coefficients = params[1 : 1 + X.shape[1]]
        log_mean = offset + params[0] + X @ coefficients
        if family == "nb2":
            log_dispersion = params[-1]
            # torch's NB counts successes before 1/α failures; with these
            # log-odds its mean is μ and its variance μ(1 + αμ). Logits, not
            # probabilities, which round to 1 at a large αμ.
            distribution: Poisson | NegativeBinomial = NegativeBinomial(
                total_count=torch.exp(-log_dispersion),
                logits=log_dispersion + log_mean,
            )
        else:
            distribution = Poisson(torch.exp(log_mean))
        value = -distribution.log_prob(y).mean() + 0.5 * l2_penalty * (
            coefficients @ coefficients
        )
        value.backward()
        assert params.grad is not None
        return value.item(), params.grad.numpy()

    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit ``b`` and ``β`` (and ``α`` for NB2) on ``X`` and the totals ``y``."""
        # Any other name would fit the Poisson silently.
        if self.family not in ("poisson", "nb2"):
            raise ValueError(f"unknown family {self.family!r}; use 'poisson' or 'nb2'")
        if self.l2_penalty < 0:
            raise ValueError(f"l2_penalty must be at least 0, got {self.l2_penalty}")
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure)
        feature_transformer, X = self._fit_features(X, y)
        X_values, y_values = check_X_y(X, y, y_numeric=True)
        # The start is the fit without features.
        if exposure_values is None:
            offset = np.zeros(len(y_values))
            start_intercept = np.log(y_values.mean())
        else:
            offset = np.log(exposure_values)
            start_intercept = np.log(y_values.sum() / exposure_values.sum())
        start = np.concatenate([[start_intercept], np.zeros(X_values.shape[1])])
        bounds = None
        if self.family == "nb2":
            start = np.append(start, np.log(_START_DISPERSION))
            bounds = [(None, None)] * (len(start) - 1) + [
                (np.log(_MIN_DISPERSION), None)
            ]
        with single_threaded_torch():
            X_tensor, y_tensor, offset_tensor = (
                torch.tensor(values, dtype=torch.float64)
                for values in (X_values, y_values, offset)
            )
            parameters = Minimizer(self.solver, self.max_iter, self.tol).minimize(
                lambda candidate: self._objective(
                    candidate,
                    X_tensor,
                    y_tensor,
                    offset_tensor,
                    self.l2_penalty,
                    self.family,
                ),
                start,
                bounds,
            )
        self.intercept_ = float(parameters[0])
        self.coef_: np.ndarray = parameters[1 : 1 + X_values.shape[1]]
        self.use_exposure_: bool = self.use_exposure
        self.feature_transformer_ = feature_transformer
        # Only an NB2 fit has a dispersion; a refit as Poisson drops the old one.
        if self.family == "nb2":
            self.dispersion_ = float(np.exp(parameters[-1]))
        else:
            self.__dict__.pop("dispersion_", None)
        # Recorded after success, so a failed refit leaves the previous fit whole.
        validate_data(self, X, reset=True, skip_check_array=True)
        return self

    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> np.ndarray:
        """The predicted mean total for each row of ``X``."""
        check_is_fitted(self)
        # Rejects columns in another order, which would be silent.
        X_values = validate_data(self, self._transform_features(X), reset=False)
        # The fitted state, not use_exposure, which set_params may have changed.
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure_)
        log_mean = self.intercept_ + X_values @ self.coef_
        if exposure_values is not None:
            log_mean = log_mean + np.log(exposure_values)
        return np.asarray(np.exp(log_mean))
