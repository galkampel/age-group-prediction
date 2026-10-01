"""Model 2's first half: a penalized Poisson GLM for a building's total children."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import poisson
from sklearn.utils.validation import check_is_fitted, check_X_y, validate_data

from .base import BaseAgeGroupModel
from .optimization import Method, Minimizer

__all__ = ["Solver", "TotalChildrenModel"]

# scikit-learn's names for the scipy methods; see optimization.Method.
Solver = Literal["lbfgs", "bfgs"]
_METHODS: dict[Solver, Method] = {"lbfgs": "L-BFGS-B", "bfgs": "BFGS"}


class TotalChildrenModel(BaseAgeGroupModel):
    """A Poisson regression of the total, ``log μ = offset + b + Xβ``.

    ``X`` is the finished design matrix. With ``use_exposure`` (the default:
    Model 2's specification) the offset is ``log(exposure)``, the raw exposure
    passed to ``fit`` and ``predict``, so ``b + Xβ`` is the log rate per
    apartment; a forgotten exposure raises. With ``use_exposure=False`` there
    is no offset and a passed exposure is ignored.

    ``l2_penalty`` multiplies ``½‖β‖²``, added to the mean negative
    log-likelihood per building, so its meaning does not change with the
    number of buildings; the intercept ``b`` is not penalized. ``solver`` is
    the scipy method, named as scikit-learn names it; ``tol`` bounds the
    gradient of the objective at the fit. A fit that does not converge raises
    rather than returning a wrong point.
    """

    def __init__(
        self,
        *,
        solver: Solver = "lbfgs",
        use_exposure: bool = True,
        l2_penalty: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        # Stored verbatim: set_params assigns attributes without re-entering here.
        self.solver = solver
        self.use_exposure = use_exposure
        self.l2_penalty = l2_penalty
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _objective(
        parameters: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        offset: np.ndarray | float,
        l2_penalty: float,
    ) -> tuple[float, np.ndarray]:
        """The penalized mean negative log-likelihood and its gradient.

        ``parameters`` is ``[b, β]``.
        """
        intercept, coefficients = parameters[0], parameters[1:]
        mean = np.exp(offset + intercept + X @ coefficients)
        value = -np.mean(poisson.logpmf(y, mean)) + 0.5 * l2_penalty * float(
            coefficients @ coefficients
        )
        residual = mean - y
        gradient = np.concatenate(
            [[residual.mean()], X.T @ residual / len(y) + l2_penalty * coefficients]
        )
        return float(value), gradient

    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit ``b`` and ``β`` on ``X`` and the totals ``y``."""
        if self.solver not in _METHODS:
            raise ValueError(
                f"unknown solver {self.solver!r}; use one of {sorted(_METHODS)}"
            )
        # A negative penalty rewards large coefficients, and the fit still
        # converges, so nothing else would notice.
        if self.l2_penalty < 0:
            raise ValueError(f"l2_penalty must be at least 0, got {self.l2_penalty}")
        # As float arrays; the columns are recorded below, once the fit succeeded.
        X_values, y_values = check_X_y(X, y, y_numeric=True)
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure)
        # Without a child the best log rate is -inf, yet L-BFGS-B stops at a
        # finite intercept and reports success.
        if not np.any(y_values > 0):
            raise ValueError("y has no children, so no rate can be fitted")
        # The start is the fit without features: the log of the mean total,
        # or of the mean rate per apartment with the offset; every β at 0.
        if exposure_values is None:
            offset: np.ndarray | float = 0.0
            start_intercept = np.log(y_values.mean())
        else:
            offset = np.log(exposure_values)
            start_intercept = np.log(y_values.sum() / exposure_values.sum())
        start = np.concatenate([[start_intercept], np.zeros(X_values.shape[1])])
        parameters = Minimizer(_METHODS[self.solver], self.max_iter, self.tol).minimize(
            lambda candidate: self._objective(
                candidate, X_values, y_values, offset, self.l2_penalty
            ),
            start,
        )
        # Set together, only once the fit succeeded. The columns are fitted
        # state too (feature_names_in_, n_features_in_): predict checks X
        # against them.
        self.intercept_ = float(parameters[0])
        self.coef_: np.ndarray = parameters[1:]
        self.use_exposure_: bool = self.use_exposure
        validate_data(self, X, reset=True, skip_check_array=True)
        return self

    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> np.ndarray:
        """The predicted mean total for each row of ``X``."""
        check_is_fitted(self)
        # Columns in another order would meet the wrong coefficients silently.
        X_values = validate_data(self, X, reset=False)
        # Follows how the model was fitted, not the current use_exposure, which
        # set_params may have changed since.
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure_)
        log_mean = self.intercept_ + X_values @ self.coef_
        if exposure_values is not None:
            log_mean = log_mean + np.log(exposure_values)
        return np.asarray(np.exp(log_mean))
