"""NegativeBinomialRegressor: each test names the mistake in our code it would catch."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import (
    check_do_not_raise_errors_in_init_or_set_params,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

from age_group_prediction.modeling import NegativeBinomialRegressor

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
