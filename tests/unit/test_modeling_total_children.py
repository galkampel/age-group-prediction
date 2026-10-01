"""TotalChildrenModel: each test names the mistake in our code it would catch."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import check_grad
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.modeling import TotalChildrenModel
from age_group_prediction.modeling.optimization import Solver
from age_group_prediction.modeling.total_children import Family


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
FAMILIES: list[Family] = ["poisson", "nb2"]


@pytest.fixture(autouse=True, scope="module")
def _torch_single_threaded() -> Iterator[None]:
    # The suite loads LightGBM before torch, and torch's threaded kernels then
    # crash. fit sets one thread; the tests that call _objective directly
    # need it too.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _nb2_data(
    dispersion: float, rows: int = 2000, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Totals drawn from an NB2 with a known dispersion, on _data's mean."""
    X_nb2, _, exposure = _data(rows, seed)
    mean = exposure * np.exp(-2 + 0.4 * X_nb2["ses"] - 0.3 * X_nb2["size"])
    rng = np.random.default_rng(seed + 100)
    y = pd.Series(
        nbinom.rvs(1 / dispersion, 1 / (1 + dispersion * mean), random_state=rng)
    )
    return X_nb2, y, exposure


def _tensors(
    X_values: pd.DataFrame, y: pd.Series, exposure: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What fit hands _objective: X, y and the offset, as float64 tensors."""
    X_tensor, y_tensor, offset = (
        torch.tensor(values, dtype=torch.float64)
        for values in (X_values.to_numpy(), y.to_numpy(), np.log(exposure))
    )
    return X_tensor, y_tensor, offset


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


@pytest.mark.parametrize("family", FAMILIES)
def test_the_gradient_matches_the_objective(family: Family) -> None:
    # The fit trusts the autograd gradient; one of another value than the one
    # returned (e.g. a term added after backward) can still "converge".
    rng = np.random.default_rng(1)
    X_values, y, offset = _tensors(X, Y, EXPOSURE)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return TotalChildrenModel._objective(
            parameters, X_values, y, offset, 0.5, family
        )

    point = rng.normal(scale=0.3, size=X.shape[1] + 1) + np.array([-2, 0, 0, 0])
    if family == "nb2":
        point = np.append(point, np.log(0.3))

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


@pytest.mark.parametrize("family", FAMILIES)
def test_doubling_the_exposure_doubles_the_mean(family: Family) -> None:
    # Exact because the exposure is not a feature; fails if predict drops the
    # offset.
    model = TotalChildrenModel(family=family).fit(X, Y, exposure=EXPOSURE)

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
            lambda: TotalChildrenModel(family="gamma").fit(  # type: ignore[arg-type]
                X, Y, exposure=EXPOSURE
            ),
            "unknown family",
        ),
        (
            lambda: TotalChildrenModel(family="nb2", solver="bfgs").fit(
                X, Y, exposure=EXPOSURE
            ),
            "takes no bounds",
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
    ids=[
        "negative-penalty",
        "unknown-solver",
        "unknown-family",
        "nb2-with-bfgs",
        "reordered-columns",
    ],
)
def test_input_that_would_fit_silently_raises(
    run: Callable[[], object], message: str
) -> None:
    # A negative penalty rewards large coefficients and still converges; a
    # solver or family name set by set_params is only read at fit (an unknown
    # family would fit the Poisson); BFGS would drop NB2's bound on α (scipy
    # only warns); reordered columns would meet the wrong coefficients.
    with pytest.raises(ValueError, match=message):
        run()


def test_a_failed_refit_leaves_the_previous_fit_intact() -> None:
    # A refit on other columns that fails must not leave their names beside
    # the old coefficients: predict would then accept the new columns and
    # use the old coefficients on them, silently.
    model = TotalChildrenModel().fit(X, Y, exposure=EXPOSURE)
    before = model.predict(X, exposure=EXPOSURE)
    renamed = X.rename(columns={"ses": "a", "size": "b", "noise": "c"})

    with pytest.raises(RuntimeError, match="did not converge"):
        model.set_params(max_iter=1).fit(renamed, Y, exposure=EXPOSURE)

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


def test_the_nb2_objective_is_the_nb2_log_likelihood() -> None:
    # The NB2 of statsmodels and of §6, in scipy's form: a wrong torch
    # parameterization (1/α and α swapped, probabilities for odds) still fits.
    rng = np.random.default_rng(2)
    X_values, y, offset = _tensors(X, Y, EXPOSURE)
    point = np.r_[-2, rng.normal(scale=0.3, size=X.shape[1]), np.log(0.3)]
    intercept, coefficients, dispersion = point[0], point[1:-1], np.exp(point[-1])
    mean = EXPOSURE * np.exp(intercept + X.to_numpy() @ coefficients)
    expected = (
        -np.mean(nbinom.logpmf(Y, 1 / dispersion, 1 / (1 + dispersion * mean)))
        + 0.25 * coefficients @ coefficients
    )

    value, _ = TotalChildrenModel._objective(point, X_values, y, offset, 0.5, "nb2")

    assert value == pytest.approx(expected, rel=1e-10)


def test_nb2_recovers_a_known_dispersion() -> None:
    # A Poisson fitted under the name nb2, or α reported as 1/α, misses it.
    X_nb2, y_nb2, exposure_nb2 = _nb2_data(dispersion=0.5)

    model = TotalChildrenModel(family="nb2").fit(X_nb2, y_nb2, exposure=exposure_nb2)

    assert model.dispersion_ == pytest.approx(0.5, abs=0.1)
    np.testing.assert_allclose(model.coef_, [0.4, -0.3, 0.0], atol=0.1)


def test_nb2_on_poisson_data_stops_at_the_dispersion_floor() -> None:
    # Without overdispersion the likelihood keeps rising as α → 0; unbounded,
    # log α runs off to where lgamma loses its precision. The floor holds it
    # at the Poisson limit, with the Poisson's coefficients.
    nb2 = TotalChildrenModel(family="nb2").fit(X, Y, exposure=EXPOSURE)
    poisson = TotalChildrenModel().fit(X, Y, exposure=EXPOSURE)

    assert nb2.dispersion_ == pytest.approx(1e-6, rel=1e-6)
    np.testing.assert_allclose(nb2.coef_, poisson.coef_, atol=1e-4)


def test_only_an_nb2_fit_has_a_dispersion() -> None:
    # A Poisson fit with a dispersion, or one left over from an earlier NB2
    # fit, would describe a model that was not fitted.
    model = TotalChildrenModel(family="nb2").fit(X, Y, exposure=EXPOSURE)
    assert hasattr(model, "dispersion_")

    model.set_params(family="poisson").fit(X, Y, exposure=EXPOSURE)

    assert not hasattr(model, "dispersion_")


def test_an_nb2_fit_survives_lightgbm_loaded_first() -> None:
    # Importing the package loads LightGBM before torch, and their two OpenMP
    # runtimes make torch's threaded kernels segfault: NB2's logsigmoid even on
    # a few rows, and building a tensor above ~32k elements. The whole fit must
    # run torch single-threaded. A subprocess, because the import order is the
    # trigger and this module runs under one thread.
    script = """
import lightgbm, torch
import numpy as np, pandas as pd
from age_group_prediction.modeling import TotalChildrenModel
rng = np.random.default_rng(0)
X = pd.DataFrame({"x": rng.normal(size=40_000)})
exposure = rng.integers(12, 80, size=40_000).astype(float)
y = pd.Series(rng.poisson(exposure * np.exp(-2 + 0.4 * X["x"])))
TotalChildrenModel(family="nb2").fit(X, y, exposure=exposure)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        capture_output=True,
        text=True,
        check=False,  # the return code is the assertion, with the crash output
    )

    assert completed.returncode == 0, completed.stderr[-2000:]


def test_the_thread_count_is_restored_after_the_fit_even_a_failed_one() -> None:
    # The fit runs torch single-threaded; leaving it so would slow every
    # later torch user in the process, silently. A failed fit (features on a
    # huge scale) is the path that skips a restore not placed in finally.
    torch.set_num_threads(3)
    try:
        TotalChildrenModel().fit(X, Y, exposure=EXPOSURE)
        assert torch.get_num_threads() == 3
        with pytest.raises(RuntimeError):
            TotalChildrenModel().fit(X * 1e4, Y, exposure=EXPOSURE)
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(1)  # the module fixture's setting
