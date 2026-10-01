"""Minimizer: each test names the mistake in our code it would catch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.optimize import minimize, rosen, rosen_der

from age_group_prediction.modeling import optimization
from age_group_prediction.modeling.optimization import Method, Minimizer

METHODS: list[Method] = ["L-BFGS-B", "BFGS"]
MINIMIZER = Minimizer("L-BFGS-B", max_iter=500, tol=1e-6)
TARGET = np.array([1.5, -2.0])


def _quadratic(beta: np.ndarray) -> tuple[float, np.ndarray]:
    """``½‖beta − TARGET‖²``, smallest at ``TARGET``."""
    difference = beta - TARGET
    return 0.5 * float(difference @ difference), difference


def _rosenbrock(beta: np.ndarray) -> tuple[float, np.ndarray]:
    return float(rosen(beta)), rosen_der(beta)


@pytest.mark.parametrize("method", METHODS)
def test_the_minimum_is_returned(method: Method) -> None:
    # Returning the start point or scipy's objective value would pass a fit
    # that never moved; a method name misspelt in the table never runs.
    found = Minimizer(method, max_iter=500, tol=1e-6).minimize(_quadratic, np.zeros(2))

    np.testing.assert_allclose(found, TARGET, atol=1e-6)


def test_an_unknown_method_raises() -> None:
    # scipy would raise its own error; ours lists the methods this table maps.
    with pytest.raises(ValueError, match="unknown method.*BFGS"):
        Minimizer("Nelder-Mead", max_iter=500, tol=1e-6).minimize(  # type: ignore[arg-type]
            _quadratic, np.zeros(2)
        )


def test_bounds_with_a_method_that_ignores_them_raise() -> None:
    # scipy only warns, then minimizes without them: a bounded parameter
    # (B8's log dispersion) would leave its range silently.
    with pytest.raises(ValueError, match="takes no bounds"):
        Minimizer("BFGS", max_iter=500, tol=1e-6).minimize(
            _quadratic, np.zeros(2), bounds=[(None, 1.0), (-1.0, None)]
        )


def test_bounds_are_honored() -> None:
    # A bounded parameter (B8's log dispersion) that left its range would reach
    # values the model cannot use.
    found = MINIMIZER.minimize(
        _quadratic, np.zeros(2), bounds=[(None, 1.0), (-1.0, None)]
    )

    np.testing.assert_allclose(found, [1.0, -1.0], atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_the_gradient_at_the_minimum_is_within_tol(method: Method) -> None:
    # tol is the models' accuracy setting; each method's table entry must map
    # it to that method's gradient tolerance. With scipy's own defaults the
    # search stops at a gradient of 6.4e-6 here (L-BFGS-B) or 1e-5 (BFGS).
    tol = 1e-6

    found = Minimizer(method, max_iter=500, tol=tol).minimize(
        _rosenbrock, np.full(5, -1.2)
    )

    assert np.abs(rosen_der(found)).max() <= tol


def test_an_iteration_limit_raises() -> None:
    # scipy returns the last point with success=False; used silently, it is a
    # fit that stopped halfway.
    with pytest.raises(RuntimeError, match="did not converge.*max_iter"):
        Minimizer("L-BFGS-B", max_iter=2, tol=1e-6).minimize(_rosenbrock, np.zeros(2))


def test_the_iteration_limit_binds_before_the_evaluation_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # scipy's cap on evaluations (maxfun, 15000 by default) also ends with
    # status 1, where the advice "raise max_iter" would not help. After a
    # failed line search L-BFGS-B resets and searches again once.
    options: dict[str, float] = {}

    def recording_minimize(*args: object, **kwargs: Any) -> object:
        options.update(kwargs["options"])
        return minimize(*args, **kwargs)

    monkeypatch.setattr(optimization, "minimize", recording_minimize)
    Minimizer("L-BFGS-B", max_iter=1000, tol=1e-6).minimize(_quadratic, np.zeros(2))

    assert options.get("maxfun", 15_000) >= (1000 + 1) * (2 * options["maxls"] + 1)


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
