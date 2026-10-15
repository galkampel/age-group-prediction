"""NB2 regression as a scikit-learn regressor, with an optional exposure; statsmodels inside."""

from __future__ import annotations

import warnings
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.families import NegativeBinomial as NegativeBinomialFamily
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import ConvergenceWarning

__all__ = ["NegativeBinomialRegressor"]

# The coordinate descent of the penalized fit. statsmodels' defaults (50
# iterations, tolerance 1e-10) stop short on small penalties; these gave the
# same path without a warning (measured, docs/INDEPENDENT_COHORT_MODELS.md §2.3).
_PENALIZED_MAX_ITER = 200
_PENALIZED_TOL = 1e-6


class NegativeBinomialRegressor(RegressorMixin, BaseEstimator):
    """NB2: ``log μ = log(exposure) + b + Xβ``, ``Var = μ(1 + αμ)``; α free or given, an L1 penalty on β.

    ``y`` is a count. ``exposure`` is raw, as statsmodels' own (the log is
    taken inside, with coefficient 1), positive, one value per row, at ``fit``
    and ``predict`` alike; ``CountModel`` passes it when ``use_exposure`` is
    on.

    ``dispersion=None`` fits α by maximum likelihood with the rest (statsmodels'
    discrete model, BFGS with its preliminary Poisson fit). A given
    ``dispersion`` fixes α and fits ``b`` and ``β`` as a GLM at it: by IRLS,
    or with ``alpha > 0`` by coordinate descent under the L1 penalty
    ``alpha · Σ|β_j|`` per observation, the intercept free. The two are the
    hyperparameters a search tunes together; a penalty needs a given
    dispersion, because the free-α penalized solver does not reach its optimum
    (``docs/INDEPENDENT_COHORT_MODELS.md`` §2.3). A fit that does not converge
    raises ``RuntimeError``; ``max_iter`` bounds the likelihood and IRLS fits.
    """

    def __init__(
        self,
        *,
        dispersion: float | None = None,
        alpha: float = 0.0,
        max_iter: int = 500,
    ) -> None:
        self.dispersion = dispersion
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(
        self, X: ArrayLike, y: ArrayLike, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit the coefficients, and α unless given, on ``X`` and the counts ``y``."""
        self._check_settings()
        X, y = validate_data(self, X, y, reset=True, y_numeric=True)
        # "add": the default "skip" adds no intercept beside a constant column,
        # which would shift the parameters by one.
        design = add_constant(X, has_constant="add")
        if self.dispersion is None:
            # statsmodels orders them: intercept, coefficients, α.
            params = self._fit_free_dispersion(y, design, exposure)
            coefficients, dispersion = params[:-1], float(params[-1])
        else:
            coefficients = self._fit_fixed_dispersion(y, design, exposure)
            dispersion = float(self.dispersion)
        self.intercept_ = float(coefficients[0])
        self.coef_ = coefficients[1:]
        self.dispersion_ = dispersion
        return self

    def _check_settings(self) -> None:
        # statsmodels fails on these late and obscurely: a dispersion of 0 as a
        # ZeroDivisionError, a negative one as "the first guess on the deviance
        # function returned a nan"; a negative penalty would be a reward.
        if self.dispersion is not None and self.dispersion <= 0:
            raise ValueError(f"dispersion must be positive, got {self.dispersion}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.alpha > 0 and self.dispersion is None:
            raise ValueError(
                "give a dispersion to penalize: alpha > 0 needs a fixed dispersion"
            )

    def _fit_free_dispersion(
        self, y: ArrayLike, design: np.ndarray, exposure: ArrayLike | None
    ) -> np.ndarray:
        """Intercept, coefficients and α by maximum likelihood."""
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
        return np.asarray(results.params, dtype=float)

    def _fit_fixed_dispersion(
        self, y: ArrayLike, design: np.ndarray, exposure: ArrayLike | None
    ) -> np.ndarray:
        """Intercept and coefficients as a GLM at the given dispersion, penalized if ``alpha > 0``."""
        glm = GLM(
            y,
            design,
            family=NegativeBinomialFamily(alpha=self.dispersion),
            exposure=exposure,
        )
        if self.alpha == 0:
            results = glm.fit(maxiter=self.max_iter)
            limit = self.max_iter
        else:
            # A zero for the intercept: statsmodels penalizes every column it is
            # given, the constant included.
            penalties = np.r_[0.0, np.full(design.shape[1] - 1, self.alpha)]
            with warnings.catch_warnings():
                # Raised below instead, as an error.
                warnings.filterwarnings(
                    "ignore", message="Elastic net fitting did not converge"
                )
                results = glm.fit_regularized(
                    method="elastic_net",
                    alpha=penalties,
                    L1_wt=1.0,
                    maxiter=_PENALIZED_MAX_ITER,
                    cnvrg_tol=_PENALIZED_TOL,
                )
            limit = _PENALIZED_MAX_ITER
        # IRLS sets the flag without any warning.
        if not results.converged:
            raise RuntimeError(
                f"NB2 at dispersion {self.dispersion} did not converge in {limit} "
                f"iterations (alpha={self.alpha})"
            )
        return np.asarray(results.params, dtype=float)

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
