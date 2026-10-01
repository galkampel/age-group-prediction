"""CohortProbabilityModel: each test names the mistake in our code it would catch."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import approx_fprime
from scipy.special import softmax
from scipy.stats import dirichlet

from age_group_prediction.modeling import CohortProbabilityModel
from age_group_prediction.scoring import cohort_log_loss

COHORTS = ["n_kindergarten", "n_elementary", "n_highschool", "n_other"]
INTERCEPTS = np.array([1.0, 1.5, 0.5, 1.0])


def _data(
    rows: int, n_cohorts: int = 3, seed: int = 0, total: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Counts whose composition is Dirichlet, with a known intercept and W.

    Two of three features move the concentration; the totals are this data's
    (``2 + Poisson(17)``) unless ``total`` fixes them, large for recovery.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "ses": rng.normal(size=rows),
            "size": rng.normal(size=rows),
            "noise": rng.normal(size=rows),
        }
    )
    intercepts = INTERCEPTS[:n_cohorts]
    coefficients = rng.normal(scale=0.5, size=(3, n_cohorts))
    coefficients[2] = 0.0
    concentration = np.exp(intercepts + X.to_numpy() @ coefficients)
    shares = np.vstack([rng.dirichlet(row) for row in concentration])
    totals = np.full(rows, total) if total else 2 + rng.poisson(17, size=rows)
    counts = np.vstack([rng.multinomial(n, s) for n, s in zip(totals, shares)])
    return (
        X,
        pd.DataFrame(counts, columns=COHORTS[:n_cohorts]),
        intercepts,
        coefficients,
    )


X, Y, _, _ = _data(300)


@pytest.fixture(autouse=True, scope="module")
def _torch_single_threaded() -> Iterator[None]:
    # The suite loads LightGBM before torch, and torch's threaded kernels then
    # crash. fit sets one thread; the tests that call _objective directly
    # need it too.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _compressed_shares(counts: pd.DataFrame) -> np.ndarray:
    """The shares as fit sees them: Smithson & Verkuilen's compression."""
    values = counts.to_numpy()
    rows, n_cohorts = values.shape
    shares = values / values.sum(axis=1, keepdims=True)
    return (shares * (rows - 1) + 1 / n_cohorts) / rows


def _objective_at(point: np.ndarray, l2_penalty: float) -> tuple[float, np.ndarray]:
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float64)
    shares = torch.tensor(_compressed_shares(Y), dtype=torch.float64)
    return CohortProbabilityModel._objective(point, X_tensor, shares, l2_penalty)


def test_the_objective_is_the_mean_dirichlet_log_density() -> None:
    # Another density, a sum where the mean belongs, or a penalty on another
    # scale than N8's (per building) still fits something.
    n_cohorts = Y.shape[1]
    point = np.random.default_rng(1).normal(scale=0.3, size=n_cohorts * 4)
    coefficients = point[n_cohorts:].reshape(-1, n_cohorts)
    concentration = np.exp(point[:n_cohorts] + X.to_numpy() @ coefficients)
    expected = (
        -np.mean(
            [
                dirichlet.logpdf(share, row)
                for share, row in zip(_compressed_shares(Y), concentration)
            ]
        )
        + 0.25 * (coefficients**2).sum()
    )

    value, _ = _objective_at(point, 0.5)

    assert value == pytest.approx(expected, rel=1e-10)


def test_the_gradient_matches_the_objective() -> None:
    # The fit trusts the autograd gradient; one of another value than the one
    # returned (e.g. a term added after backward) can still "converge".
    point = np.random.default_rng(2).normal(scale=0.3, size=Y.shape[1] * 4)

    _, gradient = _objective_at(point, 0.5)

    numerical = approx_fprime(point, lambda p: _objective_at(p, 0.5)[0], 1e-7)
    np.testing.assert_allclose(gradient, numerical, atol=1e-5)


@pytest.mark.parametrize("n_cohorts", [2, 3, 4])
def test_known_parameters_are_recovered(n_cohorts: int) -> None:
    # Shares not normalized, exp missing, or a cohort count hard-coded to 3
    # all miss the parameters that generated the composition.
    X_big, Y_big, intercepts, coefficients = _data(2000, n_cohorts, seed=1, total=1000)

    model = CohortProbabilityModel().fit(X_big, Y_big)

    np.testing.assert_allclose(model.intercept_, intercepts, atol=0.15)
    np.testing.assert_allclose(model.coef_, coefficients, atol=0.15)


def test_shares_on_the_boundary_are_compressed() -> None:
    # This data has buildings with a zero cohort; a raw share of 0 has no
    # Dirichlet density, so the fit would end non-finite.
    assert (Y.to_numpy() == 0).any(axis=1).sum() > 0

    model = CohortProbabilityModel().fit(X, Y)

    assert np.isfinite(model.coef_).all()


def test_fit_compresses_the_shares_as_the_pinned_objective_does() -> None:
    # The objective test feeds its own compressed shares; fit's own
    # compression could differ (another floor or scale) and every other test
    # still pass. At fit's point the pinned objective must be stationary.
    model = CohortProbabilityModel(tol=1e-8).fit(X, Y)
    point = np.concatenate([model.intercept_, model.coef_.ravel()])

    _, gradient = _objective_at(point, 0.0)

    assert np.abs(gradient).max() < 1e-6


def test_scaling_every_count_leaves_the_fit_unchanged() -> None:
    # The observation is the composition: counts where shares should be would
    # move the fit.
    model = CohortProbabilityModel().fit(X, Y)
    scaled = CohortProbabilityModel().fit(X, 10 * Y)

    np.testing.assert_allclose(scaled.coef_, model.coef_, atol=1e-6)
    np.testing.assert_allclose(scaled.intercept_, model.intercept_, atol=1e-6)


def test_a_huge_penalty_leaves_the_intercept_only_fit() -> None:
    # Every coefficient goes to 0 but the intercepts must not: they are
    # unpenalized, so the fit is the one without features.
    model = CohortProbabilityModel(l2_penalty=1e8).fit(X, Y)
    intercept_only = CohortProbabilityModel().fit(X * 0, Y)

    np.testing.assert_allclose(model.intercept_, intercept_only.intercept_, atol=1e-5)
    assert np.abs(model.coef_).max() < 1e-6


def test_probabilities_are_named_like_y_and_indexed_like_x() -> None:
    # Misnamed columns or misaligned rows would be combined with the wrong
    # cohort or building by Model 2; logits out of step with the
    # probabilities would miscalibrate.
    X_new = X.iloc[:5].set_index(pd.Index([10, 20, 30, 40, 50]))
    model = CohortProbabilityModel().fit(X, Y)

    probabilities = model.predict(X_new)

    assert list(probabilities.columns) == list(Y.columns)
    assert probabilities.index.equals(X_new.index)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        probabilities.to_numpy(),
        softmax(model.predict_logits(X_new).to_numpy(), axis=1),
        rtol=1e-12,
    )


def test_cohorts_from_an_array_are_numbered() -> None:
    # Reading .columns from an array y would fail; the names must come from
    # what y has.
    model = CohortProbabilityModel().fit(X, Y.to_numpy())

    assert model.cohorts_ == [0, 1, 2]
    assert list(model.predict(X).columns) == [0, 1, 2]


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (
            lambda: CohortProbabilityModel().fit(X, Y.assign(n_highschool=0)),
            r"no child.*\['n_highschool'\]",
        ),
        (lambda: CohortProbabilityModel(l2_penalty=-0.01).fit(X, Y), "l2_penalty"),
        (
            lambda: (
                CohortProbabilityModel().fit(X, Y).predict(X[["noise", "size", "ses"]])
            ),
            "feature names",
        ),
    ],
    ids=["unobserved-cohort", "negative-penalty", "reordered-columns"],
)
def test_input_that_would_fit_silently_raises(
    run: Callable[[], object], message: str
) -> None:
    # An unobserved cohort would be fitted to the compressed floor; a negative
    # penalty rewards large coefficients and still converges; reordered
    # columns would meet the wrong coefficients.
    with pytest.raises(ValueError, match=message):
        run()


def test_non_convergence_raises() -> None:
    # A fit that stopped halfway, used silently, is a wrong composition.
    with pytest.raises(RuntimeError, match="did not converge"):
        CohortProbabilityModel(max_iter=1).fit(X, Y)


def test_a_failed_refit_leaves_the_previous_fit_intact() -> None:
    # A refit on other columns that fails must not leave their names beside
    # the old coefficients: predict would then accept the new columns.
    model = CohortProbabilityModel().fit(X, Y)
    before = model.predict(X)
    renamed = X.rename(columns={"ses": "a", "size": "b", "noise": "c"})

    with pytest.raises(RuntimeError, match="did not converge"):
        model.set_params(max_iter=1).fit(renamed, Y)

    pd.testing.assert_frame_equal(model.predict(X), before)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(renamed)


def test_the_solver_setting_picks_the_scipy_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both methods reach the same fit, so only the call shows which one ran.
    from age_group_prediction.modeling import optimization

    methods: list[str] = []
    scipy_minimize = optimization.minimize

    def recording_minimize(*args: object, **kwargs: Any) -> object:
        methods.append(kwargs["method"])
        return scipy_minimize(*args, **kwargs)

    monkeypatch.setattr(optimization, "minimize", recording_minimize)
    CohortProbabilityModel(solver="bfgs").fit(X, Y)

    assert methods == ["BFGS"]


def test_the_model_beats_the_marginal_shares_on_new_buildings() -> None:
    # A fit that never left its start point predicts equal shares; one that
    # ignores the features predicts the marginal shares.
    X_all, Y_all, _, _ = _data(2000, seed=3)
    X_fit, Y_fit, X_new, Y_new = X_all[:1000], Y_all[:1000], X_all[1000:], Y_all[1000:]
    model = CohortProbabilityModel().fit(X_fit, Y_fit)
    marginal = np.tile(Y_fit.sum() / Y_fit.to_numpy().sum(), (len(X_new), 1))

    model_loss = cohort_log_loss(Y_new.to_numpy(), model.predict(X_new).to_numpy())

    assert model_loss < 0.95 * cohort_log_loss(Y_new.to_numpy(), marginal)


def test_a_passed_exposure_is_ignored() -> None:
    # A caller passes one exposure to every model; this one has no offset.
    model = CohortProbabilityModel().fit(X, Y)
    with_exposure = CohortProbabilityModel().fit(X, Y, exposure=np.ones(len(X)))

    np.testing.assert_array_equal(with_exposure.coef_, model.coef_)
    pd.testing.assert_frame_equal(
        with_exposure.predict(X, exposure=np.ones(len(X))), model.predict(X)
    )


def test_the_thread_count_is_restored_after_the_fit_even_a_failed_one() -> None:
    # The fit runs torch single-threaded; leaving it so would slow every
    # later torch user in the process, silently.
    torch.set_num_threads(3)
    try:
        CohortProbabilityModel().fit(X, Y)
        assert torch.get_num_threads() == 3
        with pytest.raises(RuntimeError):
            CohortProbabilityModel(max_iter=1).fit(X, Y)
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(1)  # the module fixture's setting
