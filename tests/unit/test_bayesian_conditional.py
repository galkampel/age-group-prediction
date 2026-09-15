"""Focused tests for Bayesian conditional model contracts that do not run MCMC."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from age_group_prediction import (
    DEFAULT_TOTAL_FEATURE_SPEC,
    BayesianConditionalModel,
    PredictionConfig,
)
from age_group_prediction.modeling_config import (
    DEFAULT_BAYESIAN_CONDITIONAL_CONFIG,
    BayesianDiagnosticConfig,
    BayesianPriorPredictiveConfig,
)
from age_group_prediction.models import bayesian_conditional as bayesian_module
from age_group_prediction.models.bayesian_inference import (
    _summarize_diagnostics,
    evaluate_stage_diagnostics,
)


def test_stage_diagnostics_accept_valid_payload() -> None:
    diagnostics = {
        "worst_rhat": 1.01,
        "minimum_effective_sample_size": 120.0,
        "divergences": 0,
        "minimum_mean_accept_prob": 0.9,
        "maximum_mean_accept_prob": 0.92,
        "tree_depth_saturation": 0.0,
    }

    assert evaluate_stage_diagnostics(
        diagnostics, BayesianDiagnosticConfig(action="error")
    ) == ()


def test_stage_diagnostics_report_each_failed_threshold() -> None:
    diagnostics = {
        "worst_rhat": 1.2,
        "minimum_effective_sample_size": 20.0,
        "divergences": 2,
        "minimum_mean_accept_prob": 0.2,
        "maximum_mean_accept_prob": 0.999,
        "tree_depth_saturation": 0.5,
    }

    failures = evaluate_stage_diagnostics(
        diagnostics, BayesianDiagnosticConfig(action="warn")
    )

    assert len(failures) == 6
    assert all("threshold" in failure for failure in failures)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_stage_diagnostics_reject_unavailable_values(missing: float | None) -> None:
    diagnostics = {
        "worst_rhat": missing,
        "minimum_effective_sample_size": 120.0,
        "divergences": 0,
        "minimum_mean_accept_prob": 0.9,
        "maximum_mean_accept_prob": 0.92,
        "tree_depth_saturation": 0.0,
    }

    failures = evaluate_stage_diagnostics(
        diagnostics, BayesianDiagnosticConfig(action="warn")
    )

    assert failures == ("worst_rhat is unavailable",)


@pytest.fixture
def modeling_df() -> pd.DataFrame:
    rng = np.random.default_rng(18)
    row_count = 12
    counts = rng.integers(0, 4, size=(row_count, 3))
    frame = pd.DataFrame(
        {
            "building_id": np.arange(row_count),
            "neighborhood_id": np.repeat(["north", "south"], row_count // 2),
            "ses": rng.normal(size=row_count),
            "avg_household_size": rng.uniform(1.8, 3.8, size=row_count),
            "median_age": rng.uniform(24.0, 58.0, size=row_count),
            "n_daycares_500m": rng.integers(0, 6, size=row_count),
            "n_apartments": rng.integers(8, 60, size=row_count),
            "3_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "school_status": rng.choice(
                ["none", "existing", "planned"], size=row_count
            ),
            "n_kindergarten": counts[:, 0],
            "n_elementary": counts[:, 1],
            "n_highschool": counts[:, 2],
        }
    )
    frame["n_children_total"] = counts.sum(axis=1)
    return frame


def _valid_diagnostics(seed: int) -> dict[str, object]:
    return {
        "worst_rhat": 1.01,
        "minimum_effective_sample_size": 100.0,
        "divergences": 0,
        "minimum_mean_accept_prob": 0.9,
        "maximum_mean_accept_prob": 0.92,
        "tree_depth_saturation": 0.0,
        "seed": seed,
    }


def test_fit_predict_draws_reconcile_and_unseen_neighborhoods_fallback(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[str] = []

    def fake_run_nuts(model, model_args, *, config, seed):
        del config
        sample_count = 8
        calls.append(model.__name__)
        feature_count = model_args[0].shape[1]
        if model.__name__ == "_total_model":
            neighborhood_count = model_args[3]
            return {
                "total_intercept": torch.full((sample_count,), -2.0),
                "total_coefficients": torch.zeros((sample_count, feature_count)),
                "log_total_concentration": torch.zeros(sample_count),
                "neighborhood_scale": torch.full((sample_count,), 0.1),
                "neighborhood_raw": torch.zeros(
                    (sample_count, neighborhood_count)
                ),
            }, _valid_diagnostics(seed)
        return {
            "composition_intercepts": torch.zeros((sample_count, 2)),
            "composition_coefficients": torch.zeros(
                (sample_count, feature_count, 2)
            ),
            "log_composition_concentration": torch.full((sample_count,), 2.0),
        }, _valid_diagnostics(seed)

    monkeypatch.setattr(bayesian_module, "_run_nuts", fake_run_nuts)
    monkeypatch.setattr(
        bayesian_module,
        "_run_prior_predictive",
        lambda *args, **kwargs: {
            "seed": kwargs["seed"],
            "draws": 8,
            "children_per_apartment_quantiles": [0.0, 0.1, 0.5],
            "cohort_share_quantiles": [[0.1, 0.3, 0.8]] * 3,
            "violations": [],
        },
    )
    model = BayesianConditionalModel().fit(
        modeling_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(5),
    )
    evaluation = modeling_df.iloc[:3].copy()
    evaluation.loc[evaluation.index[0], "neighborhood_id"] = "unseen"
    prediction = model.predict(
        evaluation,
        prediction_config=PredictionConfig(
            n_predictive_draws=25,
            interval_levels=(0.8,),
            include_pointwise_log_probabilities=True,
        ),
        rng=np.random.default_rng(6),
    )

    assert calls == ["_total_model", "_composition_model"]
    assert prediction.predictive_draws is not None
    total_draws = prediction.predictive_draws["total"]
    cohort_draws = np.stack(
        [prediction.predictive_draws[name] for name in prediction.cohort_names],
        axis=-1,
    )
    assert total_draws.shape == (3, 25)
    assert np.array_equal(cohort_draws.sum(axis=2), total_draws)
    assert np.equal(total_draws, np.floor(total_draws)).all()
    assert prediction.pointwise_log_probabilities is not None
    assert set(prediction.pointwise_log_probabilities) == {
        "total",
        *prediction.cohort_names,
    }
    assert prediction.pointwise_log_probability_scope == "sequential_joint"
    metadata = model.get_metadata()
    assert metadata["model"]["last_prediction_unseen_neighborhood_count"] == 1
    assert metadata["model"]["probability_preprocessing"]["fit_row_count"] == 12
    assert metadata["model"]["prior_predictive"]["draws"] == 8
    json.dumps(metadata)


def test_fit_raises_when_observed_cohorts_do_not_sum_to_observed_total(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    pytest.importorskip("torch")
    broken_df = modeling_df.copy()
    broken_df.loc[broken_df.index[0], "n_kindergarten"] += 1

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("MCMC should not run when the accounting check fails")

    monkeypatch.setattr(bayesian_module, "_run_nuts", fail_if_called)
    monkeypatch.setattr(bayesian_module, "_run_prior_predictive", fail_if_called)

    with pytest.raises(ValueError, match="must sum to the observed total"):
        BayesianConditionalModel().fit(
            broken_df,
            feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
            rng=np.random.default_rng(5),
        )


def _fit_with_stubbed_backend(modeling_df: pd.DataFrame) -> BayesianConditionalModel:
    """Fit a model whose NUTS and prior-predictive stages are stubbed out."""
    return BayesianConditionalModel().fit(
        modeling_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(5),
    )


def _stub_bayesian_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch the NUTS and prior-predictive stages so fit avoids MCMC."""
    torch = pytest.importorskip("torch")

    def fake_run_nuts(model, model_args, *, config, seed):
        del config
        sample_count = 8
        feature_count = model_args[0].shape[1]
        if model.__name__ == "_total_model":
            neighborhood_count = model_args[3]
            return {
                "total_intercept": torch.full((sample_count,), -2.0),
                "total_coefficients": torch.zeros((sample_count, feature_count)),
                "log_total_concentration": torch.zeros(sample_count),
                "neighborhood_scale": torch.full((sample_count,), 0.1),
                "neighborhood_raw": torch.zeros(
                    (sample_count, neighborhood_count)
                ),
            }, _valid_diagnostics(seed)
        return {
            "composition_intercepts": torch.zeros((sample_count, 2)),
            "composition_coefficients": torch.zeros(
                (sample_count, feature_count, 2)
            ),
            "log_composition_concentration": torch.full((sample_count,), 2.0),
        }, _valid_diagnostics(seed)

    monkeypatch.setattr(bayesian_module, "_run_nuts", fake_run_nuts)
    monkeypatch.setattr(
        bayesian_module,
        "_run_prior_predictive",
        lambda *args, **kwargs: {
            "seed": kwargs["seed"],
            "draws": 8,
            "children_per_apartment_quantiles": [0.0, 0.1, 0.5],
            "cohort_share_quantiles": [[0.1, 0.3, 0.8]] * 3,
            "violations": [],
        },
    )


def test_fit_restores_torch_runtime_settings_after_a_successful_fit(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    torch = pytest.importorskip("torch")
    _stub_bayesian_backend(monkeypatch)
    previous_threads = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    torch.set_num_threads(3)
    torch.set_default_dtype(torch.float32)
    try:
        _fit_with_stubbed_backend(modeling_df)

        assert torch.get_num_threads() == 3
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_default_dtype(previous_dtype)


def test_fit_restores_torch_runtime_settings_when_fit_raises(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    torch = pytest.importorskip("torch")

    def raising_prior_predictive(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        bayesian_module, "_run_prior_predictive", raising_prior_predictive
    )
    previous_threads = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    torch.set_num_threads(3)
    torch.set_default_dtype(torch.float32)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            _fit_with_stubbed_backend(modeling_df)

        assert torch.get_num_threads() == 3
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_default_dtype(previous_dtype)


def test_summarize_diagnostics_reports_none_tree_depth_when_all_realized_depths_are_zero() -> (
    None
):
    """An all-zero, nonempty tree-depth list means ``_build_tree`` never ran.

    ``sample()`` appends a realized depth on every iteration regardless of
    whether the recursive ``_build_tree`` hook fired, so a nonempty list of
    zeros cannot be a healthy shallow chain — it can only mean a future Pyro
    NUTS replaced the recursive doubling this instrumentation depends on.
    Reporting a plain 0 would silently pass the tree-depth-saturation policy
    check, so the summary must report ``None`` (unavailable) instead.
    """
    diagnostics = _summarize_diagnostics({}, [0, 0, 0, 0], maximum_depth=10)

    assert diagnostics["maximum_observed_tree_depth"] is None
    assert diagnostics["tree_depth_saturation"] is None

    policy = BayesianDiagnosticConfig(action="warn")
    failures = evaluate_stage_diagnostics(
        {**_valid_diagnostics(seed=0), **diagnostics}, policy
    )
    assert "tree_depth_saturation is unavailable" in failures


def test_summarize_diagnostics_reports_a_real_tree_depth_when_build_tree_fired() -> None:
    diagnostics = _summarize_diagnostics({}, [1, 2, 3, 10, 10], maximum_depth=10)

    assert diagnostics["maximum_observed_tree_depth"] == 10
    assert diagnostics["tree_depth_saturation"] == pytest.approx(2 / 5)


def test_summarize_diagnostics_emits_maximum_mean_accept_prob_as_max_across_chains() -> (
    None
):
    raw = {
        "mean accept prob": {"chain 0": 0.907, "chain 1": 0.923},
    }

    diagnostics = _summarize_diagnostics(raw, [], maximum_depth=10)

    assert diagnostics["minimum_mean_accept_prob"] == pytest.approx(0.907)
    assert diagnostics["maximum_mean_accept_prob"] == pytest.approx(0.923)

def _prior_predictive_args(
    row_count: int = 24, feature_count: int = 9
) -> tuple[object, object]:
    """Build minimal total-model arguments for a prior-predictive run."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    features = torch.as_tensor(
        rng.normal(size=(row_count, feature_count)), dtype=torch.float64
    )
    log_exposure = torch.as_tensor(
        np.log(np.full(row_count, 40.0)), dtype=torch.float64
    )
    index = torch.as_tensor(np.repeat([0, 1], row_count // 2), dtype=torch.long)
    return features, (features, log_exposure, index, 2)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, None, id="configured-priors-pass"),
        pytest.param(
            {"kappa_log_loc": -20.0},
            "composition concentration is too small",
            id="degenerate-concentration",
        ),
        pytest.param(
            {"composition_coefficient_scale": 50.0},
            "drives buildings toward a single cohort",
            id="diffuse-composition-coefficients",
        ),
        pytest.param(
            {"total_intercept_loc": 8.0},
            "children per apartment exceeds",
            id="implausible-child-rate",
        ),
    ],
)
def test_prior_predictive_checks_detect_degenerate_priors(
    overrides: dict[str, float], expected: str | None
) -> None:
    """Configured priors pass, and each pathological prior raises a violation.

    Averaging composition probabilities over draws cannot detect a degenerate
    prior on its own, because the zero-mean logit priors keep that average near
    uniform however diffuse they are. These cases pin the statistics that do
    have power.
    """
    pytest.importorskip("pyro")
    import dataclasses

    from age_group_prediction.models.bayesian_components import _run_prior_predictive

    features, total_args = _prior_predictive_args()
    priors = dataclasses.replace(DEFAULT_BAYESIAN_CONDITIONAL_CONFIG.priors, **overrides)
    config = dataclasses.replace(
        DEFAULT_BAYESIAN_CONDITIONAL_CONFIG,
        priors=priors,
        prior_predictive=BayesianPriorPredictiveConfig(draws=60),
    )

    summary = _run_prior_predictive(
        (*total_args, priors, None), features, 3, config=config, seed=7
    )

    if expected is None:
        assert summary["violations"] == []
        assert (
            summary["expected_dominant_share"]
            < config.prior_predictive.maximum_expected_dominant_share
        )
        assert (
            summary["concentration_quantile"]
            > config.prior_predictive.minimum_concentration_quantile
        )
    else:
        assert any(expected in violation for violation in summary["violations"]), (
            f"expected a violation mentioning {expected!r}, got "
            f"{summary['violations']}"
        )


def _informative_posterior_stub(
    monkeypatch: pytest.MonkeyPatch, sample_count: int = 24, seed: int = 7
) -> None:
    """Stub NUTS with a posterior whose composition genuinely varies by row.

    The other stubs in this module return all-zero composition intercepts and
    coefficients, which makes every cohort probability exactly uniform. A test
    driven by that fixture cannot detect a composition stage that ignores its
    features, because uniform is what the sabotage would produce anyway.
    """
    torch = pytest.importorskip("torch")
    generator = np.random.default_rng(seed)

    def fake_run_nuts(model, model_args, *, config, seed):
        del config, seed
        feature_count = model_args[0].shape[1]
        if model.__name__ == "_total_model":
            neighborhood_count = model_args[3]
            return {
                "total_intercept": torch.as_tensor(
                    generator.normal(-2.0, 0.3, sample_count)
                ),
                "total_coefficients": torch.as_tensor(
                    generator.normal(0.0, 0.25, (sample_count, feature_count))
                ),
                "log_total_concentration": torch.as_tensor(
                    generator.normal(0.8, 0.2, sample_count)
                ),
                "neighborhood_scale": torch.as_tensor(
                    np.abs(generator.normal(0.4, 0.1, sample_count))
                ),
                "neighborhood_raw": torch.as_tensor(
                    generator.normal(0.0, 1.0, (sample_count, neighborhood_count))
                ),
            }, _valid_diagnostics(1)
        return {
            "composition_intercepts": torch.as_tensor(
                generator.normal(0.0, 0.8, (sample_count, 2))
            ),
            "composition_coefficients": torch.as_tensor(
                generator.normal(0.0, 0.6, (sample_count, feature_count, 2))
            ),
            "log_composition_concentration": torch.as_tensor(
                generator.normal(2.0, 0.4, sample_count)
            ),
        }, _valid_diagnostics(1)

    monkeypatch.setattr(bayesian_module, "_run_nuts", fake_run_nuts)
    monkeypatch.setattr(
        bayesian_module,
        "_run_prior_predictive",
        lambda *args, **kwargs: {
            "seed": kwargs["seed"],
            "draws": 8,
            "children_per_apartment_quantiles": [0.0, 0.1, 0.5],
            "cohort_share_quantiles": [[0.1, 0.3, 0.8]] * 3,
            "violations": [],
        },
    )


def test_pointwise_cohort_keys_sum_to_an_independent_joint_dirichlet_multinomial(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    """Cohort keys must decompose the posterior-integrated joint composition score.

    The oracle is independent: `scipy.stats.dirichlet_multinomial.logpmf`,
    integrated over posterior samples here rather than by the code under test.
    Gate 6 compares these keys against Model B's, so the decomposition, the
    zero final key, and the posterior integration all have to hold.
    """
    pytest.importorskip("torch")
    from scipy.special import logsumexp
    from scipy.stats import dirichlet_multinomial

    _informative_posterior_stub(monkeypatch)
    model = BayesianConditionalModel().fit(
        modeling_df, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(5)
    )
    prediction = model.predict(
        modeling_df,
        prediction_config=PredictionConfig(
            n_predictive_draws=0,
            interval_levels=(),
            include_pointwise_log_probabilities=True,
        ),
        rng=np.random.default_rng(6),
    )
    pointwise = prediction.pointwise_log_probabilities
    assert pointwise is not None
    cohorts = list(prediction.cohort_names)

    _, composition_posterior = model._posterior_arrays()
    probabilities = model._composition_probabilities(
        model._probability_transformer.transform(modeling_df).to_numpy(),
        composition_posterior,
    )
    # Guard the fixture itself: a uniform composition would make this test
    # pass against a model that ignored every feature.
    spread = probabilities.mean(axis=0).max(axis=1) - probabilities.mean(axis=0).min(axis=1)
    assert spread.mean() > 0.02, "fixture composition is too close to uniform to have power"

    kappa = np.exp(composition_posterior["log_composition_concentration"])
    observed = modeling_df.loc[:, cohorts].to_numpy(dtype=float)
    per_sample = np.array(
        [
            [
                dirichlet_multinomial.logpmf(
                    observed[row], kappa[sample] * probabilities[sample, row],
                    int(observed[row].sum()),
                )
                if observed[row].sum() > 0
                else 0.0
                for row in range(len(modeling_df))
            ]
            for sample in range(len(kappa))
        ]
    )
    oracle_joint = logsumexp(per_sample, axis=0) - np.log(len(kappa))

    model_joint = sum(pointwise[cohort] for cohort in cohorts)
    np.testing.assert_allclose(model_joint, oracle_joint, rtol=0.0, atol=1e-10)
    # The final cohort is determined once the total and preceding cohorts are
    # known, so its conditional score carries no information.
    np.testing.assert_allclose(
        pointwise[cohorts[-1]], 0.0, rtol=0.0, atol=1e-10
    )


def test_pointwise_total_key_matches_an_independent_nb2_reference(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    """The total key must be the posterior-integrated NB2 log mass.

    The reference is written from the stable ``log1p`` form here rather than
    reusing any project helper, so it fails if the model's parameterization
    drifts.
    """
    pytest.importorskip("torch")
    from scipy.special import gammaln, logsumexp

    _informative_posterior_stub(monkeypatch)
    model = BayesianConditionalModel().fit(
        modeling_df, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(5)
    )
    prediction = model.predict(
        modeling_df,
        prediction_config=PredictionConfig(
            n_predictive_draws=0,
            interval_levels=(),
            include_pointwise_log_probabilities=True,
        ),
        rng=np.random.default_rng(6),
    )
    pointwise = prediction.pointwise_log_probabilities
    assert pointwise is not None

    total_posterior, _ = model._posterior_arrays()
    transformer = model.feature_transformer
    total_means = model._total_posterior_means(
        modeling_df,
        transformer.transform(modeling_df).to_numpy(),
        transformer.get_log_exposure(modeling_df).to_numpy(),
        total_posterior,
        np.random.default_rng(6),
    )
    concentration = np.exp(total_posterior["log_total_concentration"])
    observed = modeling_df["n_children_total"].to_numpy(dtype=float)
    ratio = total_means / concentration[:, None]
    log1p_ratio = np.log1p(ratio)
    reference = (
        gammaln(observed + concentration[:, None])
        - gammaln(concentration[:, None])
        - gammaln(observed + 1.0)
        - concentration[:, None] * log1p_ratio
        + observed * (np.log(total_means) - np.log(concentration[:, None]) - log1p_ratio)
    )
    expected = logsumexp(reference, axis=0) - np.log(len(concentration))

    np.testing.assert_allclose(pointwise["total"], expected, rtol=0.0, atol=1e-8)


def test_total_model_factor_matches_an_independent_nb2_log_mass() -> None:
    """The Pyro total stage's closed-form NB2 factor must be the NB2 log mass.

    ``_total_model`` scores observed totals with a hand-written ``lgamma``
    expression through ``pyro.factor``. Nothing else in the suite executes that
    expression, so it is compared here against an independent ``log1p``
    reference across a wide parameter grid.
    """
    torch = pytest.importorskip("torch")
    pyro = pytest.importorskip("pyro")
    from scipy.special import gammaln

    from age_group_prediction.models.bayesian_components import _total_model

    observed = np.array([0.0, 1.0, 3.0, 12.0, 60.0])
    for mean, concentration in ((0.05, 0.5), (1.0, 2.7), (8.0, 1.0), (90.0, 30.0)):
        features = torch.zeros((len(observed), 1), dtype=torch.float64)
        log_exposure = torch.full(
            (len(observed),), float(np.log(mean)), dtype=torch.float64
        )
        index = torch.zeros(len(observed), dtype=torch.long)

        class _Priors:
            total_intercept_loc = 0.0
            total_intercept_scale = 1e-9
            total_coefficient_scale = 1e-9
            dispersion_log_loc = float(np.log(concentration))
            dispersion_log_scale = 1e-9
            neighborhood_scale = 1e-9

        trace = pyro.poutine.trace(
            pyro.poutine.condition(
                _total_model,
                data={
                    "total_intercept": torch.tensor(0.0, dtype=torch.float64),
                    "total_coefficients": torch.zeros(1, dtype=torch.float64),
                    "log_total_concentration": torch.tensor(
                        float(np.log(concentration)), dtype=torch.float64
                    ),
                    "neighborhood_scale": torch.tensor(0.0, dtype=torch.float64),
                    "neighborhood_raw": torch.zeros(1, dtype=torch.float64),
                },
            )
        ).get_trace(
            features,
            log_exposure,
            index,
            1,
            _Priors(),
            torch.as_tensor(observed, dtype=torch.float64),
        )
        scored = float(
            trace.nodes["total_count_log_likelihood"]["fn"].log_factor
            if "fn" in trace.nodes["total_count_log_likelihood"]
            else trace.nodes["total_count_log_likelihood"]["log_prob_sum"]
        )

        ratio = mean / concentration
        log1p_ratio = np.log1p(ratio)
        reference = (
            gammaln(observed + concentration)
            - gammaln(concentration)
            - gammaln(observed + 1.0)
            - concentration * log1p_ratio
            + observed * (np.log(mean) - np.log(concentration) - log1p_ratio)
        ).sum()
        assert abs(scored - reference) < 1e-8, (
            f"pyro.factor NB2 log mass disagrees with the log1p reference at "
            f"mean={mean}, concentration={concentration}: {scored} vs {reference}"
        )


def test_diagnostic_policy_outcome_is_recorded_even_when_it_only_warns(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    """A warned convergence failure must stay visible in recorded metadata.

    Plan section 9.3 makes convergence an explicit selection constraint. Under
    the reduced profile the action is ``warn``, so without a recorded outcome a
    non-converged fit is indistinguishable downstream from a clean one.
    """
    torch = pytest.importorskip("torch")
    failing = {
        "worst_rhat": 1.9,
        "minimum_effective_sample_size": 4.0,
        "divergences": 37,
        "minimum_mean_accept_prob": 0.9,
        "maximum_mean_accept_prob": 0.92,
        "tree_depth_saturation": 0.0,
        "seed": 1,
    }

    def fake_run_nuts(model, model_args, *, config, seed):
        del config, seed
        feature_count = model_args[0].shape[1]
        if model.__name__ == "_total_model":
            neighborhood_count = model_args[3]
            return {
                "total_intercept": torch.full((8,), -2.0),
                "total_coefficients": torch.zeros((8, feature_count)),
                "log_total_concentration": torch.zeros(8),
                "neighborhood_scale": torch.full((8,), 0.1),
                "neighborhood_raw": torch.zeros((8, neighborhood_count)),
            }, dict(failing)
        return {
            "composition_intercepts": torch.zeros((8, 2)),
            "composition_coefficients": torch.zeros((8, feature_count, 2)),
            "log_composition_concentration": torch.full((8,), 2.0),
        }, dict(failing)

    monkeypatch.setattr(bayesian_module, "_run_nuts", fake_run_nuts)
    monkeypatch.setattr(
        bayesian_module,
        "_run_prior_predictive",
        lambda *args, **kwargs: {
            "seed": kwargs["seed"],
            "draws": 8,
            "children_per_apartment_quantiles": [0.0, 0.1, 0.5],
            "cohort_share_quantiles": [[0.1, 0.3, 0.8]] * 3,
            "violations": [],
        },
    )

    model = BayesianConditionalModel()
    assert model.bayesian_config.diagnostic_policy.action == "warn"
    with pytest.warns(RuntimeWarning, match="diagnostics failed"):
        model.fit(
            modeling_df,
            feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
            rng=np.random.default_rng(5),
        )

    diagnostics = model.get_metadata()["model"]["diagnostics"]
    for stage in ("total", "composition"):
        payload = diagnostics[stage]
        assert payload["policy_passed"] is False
        assert payload["active_profile"] == "reduced"
        assert payload["policy_thresholds"]["action"] == "warn"
        failures = payload["policy_failures"]
        assert any("worst_rhat" in failure for failure in failures)
        assert any("divergences" in failure for failure in failures)
        assert any(
            "minimum_effective_sample_size" in failure for failure in failures
        )
    json.dumps(model.get_metadata())


def test_diagnostic_policy_outcome_records_a_clean_pass(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    """A converged fit records an explicit pass, not merely an absent failure."""
    pytest.importorskip("torch")
    _informative_posterior_stub(monkeypatch)

    model = BayesianConditionalModel().fit(
        modeling_df, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(5)
    )

    diagnostics = model.get_metadata()["model"]["diagnostics"]
    for stage in ("total", "composition"):
        assert diagnostics[stage]["policy_passed"] is True
        assert diagnostics[stage]["policy_failures"] == []


def test_bayesian_and_independent_keys_are_the_same_conditional_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model B and this model must emit key-by-key comparable pointwise scores.

    Gate 6 ranks these two families against each other key by key, which is only
    meaningful if both decompose composition the same way: conditional on the
    preceding cohorts in schema order, including the multinomial coefficient,
    and summing to a joint score. Model A's marginal keys are deliberately not
    comparable and are checked to declare a different scope.
    """
    pytest.importorskip("torch")
    from age_group_prediction import (
        DEFAULT_TREE_FEATURE_SPEC,
        DirectCohortModel,
        IndependentTotalProbabilityModel,
    )

    rng = np.random.default_rng(3)
    row_count = 80
    counts = rng.integers(0, 9, size=(row_count, 3))
    frame = pd.DataFrame(
        {
            "building_id": np.arange(row_count),
            "neighborhood_id": np.repeat([f"nb{index}" for index in range(8)], 10),
            "ses": rng.normal(size=row_count),
            "avg_household_size": rng.uniform(1.8, 3.8, size=row_count),
            "median_age": rng.uniform(24.0, 58.0, size=row_count),
            "n_daycares_500m": rng.integers(0, 6, size=row_count),
            "n_apartments": rng.integers(8, 60, size=row_count),
            "3_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "school_status": rng.choice(["none", "existing", "planned"], size=row_count),
            "n_kindergarten": counts[:, 0],
            "n_elementary": counts[:, 1],
            "n_highschool": counts[:, 2],
        }
    )
    frame["n_children_total"] = counts.sum(axis=1)
    config = PredictionConfig(
        n_predictive_draws=0,
        interval_levels=(),
        include_pointwise_log_probabilities=True,
    )

    _informative_posterior_stub(monkeypatch)
    bayesian = BayesianConditionalModel().fit(
        frame, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(5)
    )
    bayesian_prediction = bayesian.predict(
        frame, prediction_config=config, rng=np.random.default_rng(6)
    )
    independent = IndependentTotalProbabilityModel().fit(
        frame, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(5)
    )
    independent_prediction = independent.predict(
        frame, prediction_config=config, rng=np.random.default_rng(6)
    )

    cohorts = list(bayesian_prediction.cohort_names)
    for prediction in (bayesian_prediction, independent_prediction):
        pointwise = prediction.pointwise_log_probabilities
        assert pointwise is not None
        assert prediction.pointwise_log_probability_scope == "sequential_joint"
        assert set(pointwise) == {"total", *cohorts}
        # Conditional decomposition: once the total and the preceding cohorts
        # are known, the last cohort carries no information.
        np.testing.assert_allclose(pointwise[cohorts[-1]], 0.0, rtol=0.0, atol=1e-10)

    direct = DirectCohortModel().fit(
        frame, feature_spec=DEFAULT_TREE_FEATURE_SPEC, rng=np.random.default_rng(5)
    )
    direct_prediction = direct.predict(
        frame, prediction_config=config, rng=np.random.default_rng(6)
    )
    direct_pointwise = direct_prediction.pointwise_log_probabilities
    assert direct_pointwise is not None
    assert direct_prediction.pointwise_log_probability_scope == "marginal"
    # Model A's last cohort key is a marginal log mass and is emphatically not
    # near zero, which is exactly why a per-target table would flatter the other
    # two families by that amount for no modeling reason.
    assert np.abs(direct_pointwise[cohorts[-1]]).mean() > 0.5


def test_total_posterior_mean_responds_to_exposure_features_and_neighborhood(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    """Predicted totals must actually depend on their inputs.

    Deliberately built from the public ``predict`` payload rather than from
    ``_total_posterior_means``: a test that computes its own reference with the
    same helper it is checking would agree with a helper that ignored every
    input. This is what a constant-mean sabotage has to fail.
    """
    pytest.importorskip("torch")
    _informative_posterior_stub(monkeypatch)
    model = BayesianConditionalModel().fit(
        modeling_df, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(5)
    )
    config = PredictionConfig(
        n_predictive_draws=0, interval_levels=(), include_pointwise_log_probabilities=False
    )

    def total_mean_for(frame: pd.DataFrame) -> np.ndarray:
        return model.predict(
            frame, prediction_config=config, rng=np.random.default_rng(6)
        ).total_mean

    baseline_row = modeling_df.iloc[[0]].copy()
    baseline = total_mean_for(baseline_row)

    # The log-apartment exposure offset carries a fixed unit coefficient, so a
    # building with more apartments must be predicted to hold more children.
    larger = baseline_row.copy()
    larger.loc[larger.index[0], "n_apartments"] = (
        int(baseline_row["n_apartments"].iloc[0]) * 4
    )
    assert total_mean_for(larger)[0] > baseline[0] * 1.5

    # A different known neighborhood draws a different fitted random effect.
    other_neighborhood = baseline_row.copy()
    other_neighborhood.loc[other_neighborhood.index[0], "neighborhood_id"] = "south"
    assert not np.isclose(total_mean_for(other_neighborhood)[0], baseline[0])

    # Contextual features move the mean too.
    richer = baseline_row.copy()
    richer.loc[richer.index[0], "ses"] = float(baseline_row["ses"].iloc[0]) + 3.0
    assert not np.isclose(total_mean_for(richer)[0], baseline[0])

    # And distinct buildings must not all receive the same prediction.
    assert np.std(total_mean_for(modeling_df)) > 1e-6


# State-bundle round trips for this model are covered, together with the other
# model families, in `test_model_state_bundles.py`.
