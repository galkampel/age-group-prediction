"""TotalChildrenModel: each test names the mistake in our code it would catch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.modeling import TotalChildrenModel
from age_group_prediction.modeling.total_children import Solver


def _data(rows: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Totals proportional to building size, at a rate set by two of three features."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "ses": rng.normal(size=rows),
            "size": rng.normal(size=rows),
            "noise": rng.normal(size=rows),
        }
    )
    exposure = rng.integers(12, 80, size=rows).astype(float)
    y = pd.Series(rng.poisson(exposure * np.exp(-2 + 0.4 * X["ses"] - 0.3 * X["size"])))
    return X, y, exposure


X, Y, EXPOSURE = _data()


@pytest.mark.parametrize("solver", ["lbfgs", "bfgs"])
@pytest.mark.parametrize("l2_penalty", [0.0, 0.01, 1.0])
def test_the_fit_matches_scikit_learn_with_the_exposure(
    l2_penalty: float, solver: Solver
) -> None:
    # A wrong gradient, or a penalty on another scale than N8's (mean per
    # building), moves the coefficients, whichever solver runs.
    # PoissonRegressor weighs each rate by its exposure, so its penalty is ours
    # over the mean exposure.
    model = TotalChildrenModel(solver=solver, l2_penalty=l2_penalty).fit(
        X, Y, exposure=EXPOSURE
    )
    oracle = PoissonRegressor(
        alpha=l2_penalty / EXPOSURE.mean(), tol=1e-12, max_iter=10_000
    ).fit(X, Y / EXPOSURE, sample_weight=EXPOSURE)

    assert model.intercept_ == pytest.approx(oracle.intercept_, abs=1e-6)
    np.testing.assert_allclose(model.coef_, oracle.coef_, atol=1e-6)


def test_the_fit_matches_scikit_learn_without_the_exposure() -> None:
    # Without the offset the penalty is PoissonRegressor's alpha itself.
    model = TotalChildrenModel(use_exposure=False, l2_penalty=0.01).fit(X, Y)
    oracle = PoissonRegressor(alpha=0.01, tol=1e-12, max_iter=10_000).fit(X, Y)

    assert model.intercept_ == pytest.approx(oracle.intercept_, abs=1e-6)
    np.testing.assert_allclose(model.coef_, oracle.coef_, atol=1e-6)


def test_the_solver_setting_picks_the_scipy_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both methods reach the same fit, so only the call shows which one ran: a
    # solver mapped to the wrong method would never be compared in a study.
    from age_group_prediction.modeling import optimization

    methods: list[str] = []
    scipy_minimize = optimization.minimize

    def recording_minimize(*args: object, **kwargs: Any) -> object:
        methods.append(kwargs["method"])
        return scipy_minimize(*args, **kwargs)

    monkeypatch.setattr(optimization, "minimize", recording_minimize)
    TotalChildrenModel(solver="bfgs").fit(X, Y, exposure=EXPOSURE)

    assert methods == ["BFGS"]


def test_the_gradient_matches_the_objective() -> None:
    # The fit trusts the analytic gradient; a wrong one can still "converge".
    rng = np.random.default_rng(1)
    X_values = X.to_numpy(dtype=float)
    y = Y.to_numpy(dtype=float)
    offset = np.log(EXPOSURE)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return TotalChildrenModel._objective(parameters, X_values, y, offset, 0.5)

    point = rng.normal(scale=0.3, size=X.shape[1] + 1) + np.array([-2, 0, 0, 0])

    assert (
        check_grad(lambda p: objective(p)[0], lambda p: objective(p)[1], point) < 1e-5
    )


def test_a_huge_penalty_leaves_the_constant_rate() -> None:
    # Every β goes to 0 but the intercept must not: it is unpenalized, so the
    # prediction is the training rate per apartment times the exposure.
    model = TotalChildrenModel(l2_penalty=1e8).fit(X, Y, exposure=EXPOSURE)

    np.testing.assert_allclose(
        model.predict(X, exposure=EXPOSURE),
        EXPOSURE * Y.sum() / EXPOSURE.sum(),
        rtol=1e-6,
    )


def test_doubling_the_exposure_doubles_the_mean() -> None:
    # Exact because the exposure is not a feature; fails if predict drops the
    # offset.
    model = TotalChildrenModel().fit(X, Y, exposure=EXPOSURE)

    np.testing.assert_allclose(
        model.predict(X, exposure=2 * EXPOSURE),
        2 * model.predict(X, exposure=EXPOSURE),
        rtol=1e-12,
    )


def test_predict_follows_how_the_model_was_fitted() -> None:
    # Turning use_exposure off after fitting must not silently return rates per
    # apartment instead of totals.
    model = TotalChildrenModel().fit(X, Y, exposure=EXPOSURE)
    before = model.predict(X, exposure=EXPOSURE)
    model.set_params(use_exposure=False)

    with pytest.raises(ValueError, match="pass `exposure`"):
        model.predict(X)
    np.testing.assert_array_equal(model.predict(X, exposure=EXPOSURE), before)


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (
            lambda: TotalChildrenModel().fit(X, Y * 0, exposure=EXPOSURE),
            "no children",
        ),
        (
            lambda: TotalChildrenModel(l2_penalty=-0.01).fit(X, Y, exposure=EXPOSURE),
            "l2_penalty",
        ),
        (
            lambda: TotalChildrenModel(solver="newton").fit(  # type: ignore[arg-type]
                X, Y, exposure=EXPOSURE
            ),
            "unknown solver",
        ),
        (
            lambda: (
                TotalChildrenModel()
                .fit(X, Y, exposure=EXPOSURE)
                .predict(X[["noise", "size", "ses"]], exposure=EXPOSURE)
            ),
            "feature names",
        ),
    ],
    ids=["all-zero-y", "negative-penalty", "unknown-solver", "reordered-columns"],
)
def test_input_that_would_fit_silently_raises(
    run: Callable[[], object], message: str
) -> None:
    # An all-zero y "converges" to a finite intercept; a negative penalty
    # rewards large coefficients and still converges; a solver name set by
    # set_params is only read at fit; reordered columns would meet the wrong
    # coefficients.
    with pytest.raises(ValueError, match=message):
        run()


def test_a_failed_refit_leaves_the_previous_fit_intact() -> None:
    # A refit on other columns that fails must not leave their names beside
    # the old coefficients: predict would then accept the new columns and
    # use the old coefficients on them, silently.
    model = TotalChildrenModel().fit(X, Y, exposure=EXPOSURE)
    before = model.predict(X, exposure=EXPOSURE)
    renamed = X.rename(columns={"ses": "a", "size": "b", "noise": "c"})

    with pytest.raises(ValueError, match="no children"):
        model.fit(renamed, Y * 0, exposure=EXPOSURE)

    np.testing.assert_array_equal(model.predict(X, exposure=EXPOSURE), before)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(renamed, exposure=EXPOSURE)


def test_the_model_beats_the_constant_rate_on_new_buildings() -> None:
    # A fit that never left its start point is the constant rate itself.
    X_new, y_new, exposure_new = _data(seed=1)
    model = TotalChildrenModel().fit(X, Y, exposure=EXPOSURE)
    constant_rate = exposure_new * Y.sum() / EXPOSURE.sum()

    model_deviance = mean_poisson_deviance(
        y_new, model.predict(X_new, exposure=exposure_new)
    )

    assert model_deviance < 0.8 * mean_poisson_deviance(y_new, constant_rate)


@pytest.mark.parametrize(
    ("exposure", "message"),
    [
        (None, "pass `exposure`"),
        (EXPOSURE[:, None], "one-dimensional"),
        (EXPOSURE[:1], "inconsistent numbers of samples"),
    ],
    ids=["missing", "two-dimensional", "length-one"],
)
def test_exposure_misuse_raises_at_fit_and_at_predict(
    exposure: np.ndarray | None, message: str
) -> None:
    # A dropped offset, an (n, n) broadcast, or one exposure for every row.
    with pytest.raises(ValueError, match=message):
        TotalChildrenModel().fit(X, Y, exposure=exposure)
    model = TotalChildrenModel().fit(X, Y, exposure=EXPOSURE)
    with pytest.raises(ValueError, match=message):
        model.predict(X, exposure=exposure)


def test_an_unused_exposure_is_ignored() -> None:
    # A caller passes one exposure to every model; with use_exposure=False it
    # must not enter the fit or the prediction.
    without = TotalChildrenModel(use_exposure=False).fit(X, Y).predict(X)

    model = TotalChildrenModel(use_exposure=False).fit(X, Y, exposure=EXPOSURE)

    np.testing.assert_array_equal(model.predict(X), without)
    np.testing.assert_array_equal(model.predict(X, exposure=EXPOSURE), without)
