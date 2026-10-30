"""NegativeBinomialRegressor: each test names the mistake in our code it would catch."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.utils.estimator_checks import (
    check_do_not_raise_errors_in_init_or_set_params,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

from age_group_prediction.modeling import NegativeBinomialRegressor, negative_binomial

INTERCEPT, COEF, ALPHA = -2.0, np.array([0.3, -0.2, 0.1]), 0.1


def _nb2_data(rows: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """NB2 counts, ``Var = μ(1 + αμ)``, with ``μ = exposure · exp(b + Xβ)``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(rows, 3)), columns=["a", "b", "c"])
    exposure = rng.uniform(5, 40, size=rows)
    mean = exposure * np.exp(INTERCEPT + X.to_numpy() @ COEF)
    # numpy's (n, p) parametrization: n = 1/α, p = n / (n + μ).
    y = rng.negative_binomial(1 / ALPHA, 1 / (1 + ALPHA * mean))
    return X, y, exposure


X, Y, EXPOSURE = _nb2_data(300)


def test_recovers_the_coefficients_and_alpha() -> None:
    # Catches a wrong reading of statsmodels' parameters: their order
    # (intercept, coefficients, α) or α reported as 1/α or log α.
    X, y, exposure = _nb2_data(5000)

    model = NegativeBinomialRegressor().fit(X, y, exposure=exposure)

    assert model.intercept_ == pytest.approx(INTERCEPT, abs=0.05)
    np.testing.assert_allclose(model.coef_, COEF, atol=0.03)
    assert model.dispersion_ == pytest.approx(ALPHA, abs=0.02)


def test_doubling_the_exposure_doubles_the_prediction() -> None:
    # Catches predict ignoring the exposure, or taking its log as if it were
    # an offset: the mean is proportional to the raw exposure.
    model = NegativeBinomialRegressor().fit(X, Y, exposure=EXPOSURE)

    np.testing.assert_allclose(
        model.predict(X, exposure=2 * EXPOSURE),
        2 * model.predict(X, exposure=EXPOSURE),
    )


def test_a_fixed_dispersion_at_the_estimate_reproduces_the_free_fit() -> None:
    # The GLM stage must be the same model at a fixed α: a wrong family, or
    # the exposure given as an offset instead of its log, would move the
    # coefficients by orders more than the two solvers' tolerances (2.7e-6
    # apart on this data).
    free = NegativeBinomialRegressor().fit(X, Y, exposure=EXPOSURE)

    fixed = NegativeBinomialRegressor(dispersion=free.dispersion_).fit(
        X, Y, exposure=EXPOSURE
    )

    assert fixed.dispersion_ == free.dispersion_
    np.testing.assert_allclose(
        np.r_[fixed.intercept_, fixed.coef_],
        np.r_[free.intercept_, free.coef_],
        atol=1e-4,
    )


@pytest.mark.filterwarnings("error")
def test_the_penalty_equals_statsmodels_with_a_free_intercept() -> None:
    # Catches a penalty on the intercept (statsmodels penalizes every column
    # unless told), or a penalty not applied at all: statsmodels called
    # directly is the oracle, and the penalized coefficients are smaller.
    model = NegativeBinomialRegressor(dispersion=0.1, alpha=0.05).fit(
        X, Y, exposure=EXPOSURE
    )
    unpenalized = NegativeBinomialRegressor(dispersion=0.1).fit(X, Y, exposure=EXPOSURE)

    oracle = sm.GLM(
        Y,
        sm.add_constant(X.to_numpy()),
        family=sm.families.NegativeBinomial(alpha=0.1),
        exposure=EXPOSURE,
    ).fit_regularized(
        method="elastic_net",
        alpha=np.r_[0.0, 0.05, 0.05, 0.05],
        L1_wt=1.0,
        maxiter=200,
        cnvrg_tol=1e-6,
    )

    assert model.dispersion_ == 0.1
    np.testing.assert_allclose(
        np.r_[model.intercept_, model.coef_], oracle.params, rtol=1e-10
    )
    assert np.abs(model.coef_).sum() < np.abs(unpenalized.coef_).sum()


def test_a_large_penalty_zeroes_the_coefficients_but_not_the_intercept() -> None:
    model = NegativeBinomialRegressor(dispersion=0.1, alpha=10.0).fit(
        X, Y, exposure=EXPOSURE
    )

    np.testing.assert_array_equal(model.coef_, np.zeros(3))
    assert model.intercept_ != 0.0


def test_a_negative_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        NegativeBinomialRegressor(dispersion=0.1, alpha=-0.1).fit(
            X, Y, exposure=EXPOSURE
        )


@pytest.mark.parametrize("dispersion", [0.0, -0.1], ids=["zero", "negative"])
def test_a_non_positive_dispersion_raises(dispersion: float) -> None:
    # statsmodels fails on these late and obscurely: a ZeroDivisionError, or
    # "the first guess on the deviance function returned a nan".
    with pytest.raises(ValueError, match="dispersion must be positive"):
        NegativeBinomialRegressor(dispersion=dispersion).fit(X, Y, exposure=EXPOSURE)


def test_a_penalty_without_a_dispersion_raises() -> None:
    # The free-α penalized solver is the one measured not to reach its optimum;
    # silently fitting unpenalized, or estimating α first, would hide that.
    with pytest.raises(ValueError, match="give a dispersion to penalize"):
        NegativeBinomialRegressor(alpha=0.05).fit(X, Y, exposure=EXPOSURE)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("alpha", [0.0, 0.05], ids=["irls", "penalized"])
def test_a_fixed_dispersion_fit_that_does_not_converge_raises(
    alpha: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # IRLS sets its flag with no warning and the elastic net only warns (a
    # UserWarning, which -W error would surface before our check), so a
    # stopped fit would otherwise pass as a fit.
    monkeypatch.setattr(negative_binomial, "_PENALIZED_MAX_ITER", 1)

    with pytest.raises(RuntimeError, match="did not converge in 1 iterations"):
        NegativeBinomialRegressor(dispersion=0.1, alpha=alpha, max_iter=1).fit(
            X, Y, exposure=EXPOSURE
        )


@pytest.mark.filterwarnings("error")
def test_a_fit_that_does_not_converge_raises() -> None:
    # Catches a missing convergence check (statsmodels only warns), and a
    # missing warning filter or Hessian skip, under which statsmodels'
    # ConvergenceWarning or HessianInversionWarning would surface first.
    with pytest.raises(RuntimeError, match="max_iter=1"):
        NegativeBinomialRegressor(max_iter=1).fit(X, Y, exposure=EXPOSURE)


def test_columns_in_another_order_raise_at_predict() -> None:
    # Catches predict skipping the feature-name check: the coefficients are
    # positional, so reordered columns would be predicted silently wrong.
    model = NegativeBinomialRegressor().fit(X, Y, exposure=EXPOSURE)

    with pytest.raises(ValueError, match="feature names"):
        model.predict(X[["b", "a", "c"]], exposure=EXPOSURE)


def test_an_exposure_of_another_shape_raises_at_predict() -> None:
    # Catches a column exposure (n, 1) broadcast against the rows to an (n, n)
    # prediction, which numpy does silently.
    model = NegativeBinomialRegressor().fit(X, Y, exposure=EXPOSURE)

    with pytest.raises(ValueError, match="exposure has shape"):
        model.predict(X, exposure=EXPOSURE[:, None])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("value", [0.0, 1.0], ids=["all-zero", "all-ones"])
def test_a_constant_column_does_not_change_the_fit(value: float) -> None:
    # A dummy absent from a training fold is all zeros. Catches Newton's
    # preliminary fit, which raises LinAlgError on it, and statsmodels'
    # has_constant="skip", which adds no intercept beside an all-ones column
    # and shifts the parameters by one.
    with_column = X.assign(constant=value)

    model = NegativeBinomialRegressor().fit(with_column, Y, exposure=EXPOSURE)
    reference = NegativeBinomialRegressor().fit(X, Y, exposure=EXPOSURE)

    np.testing.assert_allclose(
        model.predict(with_column, exposure=EXPOSURE),
        reference.predict(X, exposure=EXPOSURE),
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "check",
    [
        check_parameters_default_constructible,
        check_no_attributes_set_in_init,
        check_set_params,
        check_do_not_raise_errors_in_init_or_set_params,
    ],
    ids=lambda f: f.__name__,
)
def test_keeps_the_parameter_contract(
    check: Callable[[str, NegativeBinomialRegressor], None],
) -> None:
    # The checks the contract test runs on every model: clone and set_params
    # (the tuner's nested estimator__max_iter) rely on them.
    check("NegativeBinomialRegressor", NegativeBinomialRegressor())
