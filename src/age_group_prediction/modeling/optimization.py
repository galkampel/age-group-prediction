"""Gradient-based minimization for the models fitted by maximum likelihood."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.optimize import minimize

__all__ = [
    "Bounds",
    "Minimizer",
    "ObjectiveWithGradient",
    "Solver",
    "single_threaded_torch",
]

# The value and its gradient at a point, as scipy's jac=True expects.
type ObjectiveWithGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]
# One (lower, upper) pair per parameter; None leaves that side open.
type Bounds = Sequence[tuple[float | None, float | None]]
# scikit-learn's names for the scipy methods that reached its oracles with an
# analytic gradient and no Hessian: L-BFGS-B and BFGS, on the Poisson (1e-7)
# and the multinomial (1e-6) objectives. Only lbfgs takes bounds.
type Solver = Literal["lbfgs", "bfgs"]

# Line-search steps per iteration, as scikit-learn uses (scipy's default is 20).
_MAX_LINE_SEARCH_STEPS = 50


@contextmanager
def single_threaded_torch() -> Iterator[None]:
    """Run torch on one thread, then restore the count.

    LightGBM, scikit-learn and torch each load their own OpenMP runtime, and
    torch's threaded kernels then crash. A model's ``fit`` wraps every torch
    call in this, the tensors' construction included; one thread is also the
    faster setting at the sizes fitted here.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def _lbfgsb_options(max_iter: int, tol: float) -> dict[str, float]:
    # scikit-learn's mapping: tol bounds the projected gradient, and ftol stays
    # just above float64 precision so that it never stops the search first.
    # scipy's cap on evaluations (15000) also ends with status 1, where "raise
    # max_iter" would not help: set it above what max_iter iterations can use,
    # each with up to two line searches (one retried after a reset).
    return {
        "maxiter": max_iter,
        "gtol": tol,
        "ftol": 64 * np.finfo(float).eps,
        "maxls": _MAX_LINE_SEARCH_STEPS,
        "maxfun": (max_iter + 1) * (2 * _MAX_LINE_SEARCH_STEPS + 1),
    }


def _bfgs_options(max_iter: int, tol: float) -> dict[str, float]:
    # gtol bounds the gradient's largest component, as L-BFGS-B's does.
    return {"maxiter": max_iter, "gtol": tol}


# Each solver's scipy method, and how a model's max_iter and tol become that
# method's own options: scipy's one-size tol= argument also sets L-BFGS-B's
# ftol, which stops the search early (2.7e-4 from the oracle instead of 6e-8).
_METHODS: dict[Solver, tuple[str, Callable[[int, float], dict[str, float]]]] = {
    "lbfgs": ("L-BFGS-B", _lbfgsb_options),
    "bfgs": ("BFGS", _bfgs_options),
}


@dataclass(frozen=True)
class Minimizer:
    """Minimize a smooth objective with an analytic gradient, or raise.

    A component the models build in ``fit`` from their own ``solver``,
    ``max_iter`` and ``tol``, not an estimator. ``solver`` is scikit-learn's
    name for the scipy method. ``tol`` bounds the gradient's largest component
    at the returned point, for every solver.

    Nothing is clipped. A fit that overflows, stops early or ends at a
    non-finite objective raises ``RuntimeError`` instead of returning a wrong
    point: scipy reports an overflow as a failure at the start point, and a
    NaN objective with a zero gradient as converged.
    """

    solver: Solver
    max_iter: int
    tol: float

    def minimize(
        self,
        objective: ObjectiveWithGradient,
        start: np.ndarray,
        bounds: Bounds | None = None,
    ) -> np.ndarray:
        """The point where ``objective`` is smallest, searched from ``start``."""
        if self.solver not in _METHODS:
            raise ValueError(
                f"unknown solver {self.solver!r}; use one of {sorted(_METHODS)}"
            )
        # scipy only warns and then ignores bounds a method cannot handle.
        if bounds is not None and self.solver != "lbfgs":
            raise ValueError(f"solver {self.solver!r} takes no bounds; use 'lbfgs'")
        method, options = _METHODS[self.solver]
        try:
            with np.errstate(over="raise", invalid="raise"):
                result = minimize(
                    objective,
                    start,
                    method=method,
                    jac=True,
                    bounds=bounds,
                    options=options(self.max_iter, self.tol),
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
