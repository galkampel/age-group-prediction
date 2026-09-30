"""LBFGSMinimizer: each test names the mistake in our code it would catch."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import rosen, rosen_der

from age_group_prediction.modeling.optimization import LBFGSMinimizer

MINIMIZER = LBFGSMinimizer(max_iter=500, tol=1e-6)
TARGET = np.array([1.5, -2.0])


def _quadratic(beta: np.ndarray) -> tuple[float, np.ndarray]:
    """``½‖beta − TARGET‖²``, smallest at ``TARGET``."""
    difference = beta - TARGET
    return 0.5 * float(difference @ difference), difference


def _rosenbrock(beta: np.ndarray) -> tuple[float, np.ndarray]:
    return float(rosen(beta)), rosen_der(beta)


def test_the_minimum_is_returned() -> None:
    # Returning the start point or scipy's objective value would pass a fit
    # that never moved.
    found = MINIMIZER.minimize(_quadratic, np.zeros(2))

    np.testing.assert_allclose(found, TARGET, atol=1e-6)


def test_bounds_are_honored() -> None:
    # A bounded parameter (B8's log dispersion) that left its range would reach
    # values the model cannot use.
    found = MINIMIZER.minimize(
        _quadratic, np.zeros(2), bounds=[(None, 1.0), (-1.0, None)]
    )

    np.testing.assert_allclose(found, [1.0, -1.0], atol=1e-8)


def test_the_gradient_at_the_minimum_is_within_tol() -> None:
    # tol is the models' accuracy setting, as in scikit-learn (gtol=tol). With
    # scipy's own gtol or ftol the search stops at a gradient of 6.4e-6 here.
    tol = 1e-6

    found = LBFGSMinimizer(max_iter=500, tol=tol).minimize(
        _rosenbrock, np.full(5, -1.2)
    )

    assert np.abs(rosen_der(found)).max() <= tol


def test_an_iteration_limit_raises() -> None:
    # scipy returns the last point with success=False; used silently, it is a
    # fit that stopped halfway.
    with pytest.raises(RuntimeError, match="did not converge.*max_iter"):
        LBFGSMinimizer(max_iter=2, tol=1e-6).minimize(_rosenbrock, np.zeros(2))


def test_a_failed_line_search_does_not_advise_more_iterations() -> None:
    # A NaN gradient stops scipy abnormally; "raise max_iter" would send the
    # user after the wrong cause.
    def no_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return 1.0, np.full_like(beta, np.nan)

    with pytest.raises(RuntimeError, match="did not converge") as raised:
        MINIMIZER.minimize(no_gradient, np.zeros(1))
    assert "max_iter" not in str(raised.value)


def test_an_overflow_raises_naming_the_feature_scale() -> None:
    # Without errstate scipy reports the overflow as a failure at the start
    # point, with a NaN objective and no hint at the cause (N9).
    features = 1e3 * np.arange(1.0, 6.0)
    counts = np.array([1.0, 0.0, 3.0, 2.0, 5.0])

    def poisson(beta: np.ndarray) -> tuple[float, np.ndarray]:
        mean = np.exp(beta[0] * features)
        return float(np.sum(mean - counts * beta[0] * features)), np.array(
            [np.sum((mean - counts) * features)]
        )

    with pytest.raises(RuntimeError, match="overflowed.*scale"):
        MINIMIZER.minimize(poisson, np.zeros(1))


def test_an_invalid_value_raises() -> None:
    # The log of a negative mean gives NaN; without errstate scipy may step
    # past it and report a fit.
    def negative_mean(beta: np.ndarray) -> tuple[float, np.ndarray]:
        mean = np.exp(beta[0]) * np.array([1.0, -2.0])
        return float(np.sum(mean - np.log(mean))), np.zeros(1)

    with pytest.raises(RuntimeError, match="invalid value"):
        MINIMIZER.minimize(negative_mean, np.zeros(1))


def test_a_non_finite_objective_raises() -> None:
    # A NaN objective with a zero gradient is "converged" to scipy
    # (success=True); returned, it would be a fit made of NaN.
    def not_a_number(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return float("nan"), np.zeros_like(beta)

    with pytest.raises(RuntimeError, match="non-finite"):
        MINIMIZER.minimize(not_a_number, np.zeros(2))
