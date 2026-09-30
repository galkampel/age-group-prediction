"""L-BFGS-B minimization for the models fitted by maximum likelihood."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = ["Bounds", "LBFGSMinimizer", "ObjectiveWithGradient"]

# The value and its gradient at a point, as scipy's jac=True expects.
type ObjectiveWithGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]
# One (lower, upper) pair per parameter; None leaves that side open.
type Bounds = Sequence[tuple[float | None, float | None]]


@dataclass(frozen=True)
class LBFGSMinimizer:
    """Minimize a smooth objective with an analytic gradient, or raise.

    A component the models build in ``fit`` from their own ``max_iter`` and
    ``tol``, not an estimator. ``tol`` means what it means in scikit-learn's
    ``PoissonRegressor`` and ``LogisticRegression``, which use the same options.

    Nothing is clipped. A fit that overflows, stops early or ends at a
    non-finite objective raises ``RuntimeError`` instead of returning a wrong
    point: scipy reports an overflow as a failure at the start point, and a
    NaN objective with a zero gradient as converged.
    """

    max_iter: int
    tol: float

    def minimize(
        self,
        objective: ObjectiveWithGradient,
        start: np.ndarray,
        bounds: Bounds | None = None,
    ) -> np.ndarray:
        """The point where ``objective`` is smallest, searched from ``start``."""
        try:
            with np.errstate(over="raise", invalid="raise"):
                result = minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    # scikit-learn's mapping: tol bounds the projected gradient,
                    # and ftol stays just above float64 precision so that it
                    # never stops the search first.
                    options={
                        "maxiter": self.max_iter,
                        "gtol": self.tol,
                        "ftol": 64 * np.finfo(float).eps,
                        "maxls": 50,
                    },
                )
        except FloatingPointError as error:
            raise RuntimeError(
                f"the fit overflowed or took an invalid value ({error}); the "
                "features are likely on too large a scale: standardize them"
            ) from error
        if not result.success:
            # Status 1 is the iteration limit; any other failure (a line search
            # that found no descent) is not cured by more iterations.
            advice = (
                "raise max_iter"
                if result.status == 1
                else "check the data, the gradient and the features' scale"
            )
            raise RuntimeError(f"the fit did not converge: {result.message}; {advice}")
        if not np.isfinite(result.fun):
            raise RuntimeError(
                f"the fit ended at a non-finite objective ({result.fun}); "
                "check the data and the features' scale"
            )
        return np.asarray(result.x)
