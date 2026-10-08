"""NB2 regression as a scikit-learn regressor, with an optional exposure; statsmodels inside."""

from __future__ import annotations

import warnings
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import ConvergenceWarning

__all__ = ["NegativeBinomialRegressor"]


class NegativeBinomialRegressor(RegressorMixin, BaseEstimator):
    """Unpenalized NB2: ``log μ = log(exposure) + b + Xβ``, ``Var = μ(1 + αμ)``, α by maximum likelihood.

    ``y`` is a count. ``exposure`` is raw, as statsmodels' own (the log is
    taken inside, with coefficient 1), positive, one value per row, at ``fit``
    and ``predict`` alike; ``CountModel`` passes it when ``use_exposure`` is
    on. statsmodels fits it with BFGS, its preliminary Poisson fit included; a
    fit that does not converge in ``max_iter`` iterations raises ``RuntimeError``.
    """

    def __init__(self, *, max_iter: int = 500) -> None:
        self.max_iter = max_iter

    def fit(
        self, X: ArrayLike, y: ArrayLike, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit the coefficients and α on ``X`` and the counts ``y``."""
        X, y = validate_data(self, X, y, reset=True, y_numeric=True)
        # "add": the default "skip" adds no intercept beside a constant column,
        # which would shift the parameters by one.
        design = add_constant(X, has_constant="add")
        with warnings.catch_warnings():
            # Raised below instead, as an error.
            warnings.simplefilter("ignore", ConvergenceWarning)
            results = NegativeBinomial(
                y, design, loglike_method="nb2", exposure=exposure
            ).fit(
                method="bfgs",
                maxiter=self.max_iter,
                disp=0,
                # Newton's preliminary fit fails on an all-zero or constant column.
                optim_kwds_prelim={"method": "bfgs"},
                # The Hessian gives only standard errors, which are not used; it
                # cannot be inverted with such a column.
                skip_hessian=True,
            )
        if not results.mle_retvals["converged"]:
            raise RuntimeError(
                f"NB2 did not converge in max_iter={self.max_iter} iterations"
            )
        # statsmodels orders them: intercept, coefficients, α.
        params = np.asarray(results.params, dtype=float)
        self.intercept_ = float(params[0])
        self.coef_ = params[1:-1]
        self.dispersion_ = float(params[-1])
        return self

    def predict(self, X: ArrayLike, exposure: ArrayLike | None = None) -> np.ndarray:
        """The mean count ``exposure · exp(b + Xβ)`` for each row of ``X``."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        mean = np.exp(self.intercept_ + X @ self.coef_)
        if exposure is None:
            return mean
        exposure = np.asarray(exposure, dtype=float)
        # A column (n, 1) would broadcast against the (n,) rows to (n, n) silently.
        if exposure.shape != mean.shape:
            raise ValueError(
                f"exposure has shape {exposure.shape}; expected ({len(mean)},), "
                "one value per row of X"
            )
        return mean * exposure
