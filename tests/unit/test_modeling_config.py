"""Tests for model-runtime configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from age_group_prediction.modeling_config import (
    BayesianConditionalConfig,
    BayesianDiagnosticConfig,
    BayesianPriorConfig,
    BayesianPriorPredictiveConfig,
    BayesianStabilizationConfig,
    NUTSProfileConfig,
)


def test_bayesian_conditional_defaults_pair_profiles_and_diagnostics() -> None:
    config = BayesianConditionalConfig()

    assert config.inference_profile is config.reduced_profile
    assert config.diagnostic_policy.action == "warn"
    assert config.stabilization.clip_total_log_mean is False
    assert config.stabilization.clip_composition_logits is False


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: BayesianPriorConfig(kappa_log_scale=0.0), "kappa_log_scale"),
        (lambda: NUTSProfileConfig(chains=1, warmup_steps=5, posterior_samples=5), "two chains"),
        (lambda: NUTSProfileConfig(chains=2, warmup_steps=5, posterior_samples=3), "four"),
        (lambda: BayesianDiagnosticConfig(action="ignore"), "action"),
        (lambda: BayesianDiagnosticConfig(action="warn", maximum_rhat=0.99), "maximum_rhat"),
        (
            lambda: BayesianDiagnosticConfig(action="warn", minimum_mean_accept_prob=-0.1),
            "minimum_mean_accept_prob",
        ),
        (
            lambda: BayesianDiagnosticConfig(action="warn", minimum_mean_accept_prob=1.1),
            "minimum_mean_accept_prob",
        ),
        (lambda: BayesianPriorPredictiveConfig(draws=0), "draws"),
        (lambda: BayesianPriorPredictiveConfig(action="ignore"), "action"),
        (
            lambda: BayesianPriorPredictiveConfig(minimum_expected_share_ratio=0.0),
            "minimum_expected_share_ratio",
        ),
        (
            lambda: BayesianPriorPredictiveConfig(minimum_expected_share_ratio=1.0),
            "minimum_expected_share_ratio",
        ),
        (
            lambda: BayesianPriorPredictiveConfig(maximum_expected_share_ratio=0.5),
            "maximum_expected_share_ratio",
        ),
        (
            lambda: BayesianStabilizationConfig(total_log_mean_bounds=(1.0, 1.0)),
            "total_log_mean_bounds",
        ),
        (lambda: BayesianConditionalConfig(active_profile="quick"), "active_profile"),
    ],
)
def test_bayesian_configuration_rejects_invalid_values(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize("name", ["total_feature_spec", "probability_feature_spec"])
def test_bayesian_configuration_does_not_accept_feature_spec_aliases(name: str) -> None:
    """Feature forms come from fit() and the constructor, never from config."""
    with pytest.raises(TypeError, match=name):
        BayesianConditionalConfig(**{name: "default_total"})


def test_bayesian_prior_predictive_and_diagnostic_defaults_are_valid() -> None:
    prior_predictive = BayesianPriorPredictiveConfig()
    diagnostics = BayesianDiagnosticConfig(action="warn")

    assert prior_predictive.action == "warn"
    assert prior_predictive.minimum_expected_share_ratio == 0.25
    assert prior_predictive.maximum_expected_share_ratio == 2.0
    assert diagnostics.minimum_mean_accept_prob == 0.6


def test_bayesian_prior_predictive_accepts_explicit_valid_ratios() -> None:
    config = BayesianPriorPredictiveConfig(
        action="error",
        minimum_expected_share_ratio=0.5,
        maximum_expected_share_ratio=1.5,
    )

    assert config.action == "error"
    assert config.minimum_expected_share_ratio == 0.5
    assert config.maximum_expected_share_ratio == 1.5

from age_group_prediction import load_experiment_config


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "modeling.toml"


def test_load_experiment_config_returns_typed_sections() -> None:
    config = load_experiment_config(_config_path())

    assert config.randomness.default_seed == 42
    assert config.outer_split.strategy_version == "known-neighborhood-v1"
    assert config.prediction.n_predictive_draws == 0
    assert config.prediction_validation.reconciliation_tolerance == 1e-9
    assert config.evaluation.bootstrap_replicates == 200
    assert config.evaluation.interval_method == "percentile"
    assert config.tuning.n_trials == 30
    assert config.direct_cohort.family == "poisson"
    assert config.direct_cohort.search_space.max_depth == (3, 8)
    assert config.independent_total_probability.total_family == "nb2"
    assert config.independent_total_probability.tuning is config.tuning
    assert config.independent_total_probability.search_space.probability_c == (
        0.01,
        100.0,
    )
    assert config.independent_total_probability.bootstrap_replicates == 200
    assert config.bayesian_conditional.active_profile == "reduced"
    assert config.bayesian_conditional.inference_profile.chains == 2
    assert config.bayesian_conditional.full_profile.chains == 4
    assert config.bayesian_conditional.diagnostic_policy.action == "warn"
    assert config.bayesian_conditional.stabilization.clip_total_log_mean is False


def test_loader_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    text = _config_path().read_text() + "\n[unknown]\nvalue = 1\n"
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text(text)

    with pytest.raises(ValueError, match="unknown key: config.unknown"):
        load_experiment_config(cfg_path)


def test_loader_rejects_missing_section(tmp_path: Path) -> None:
    text = _config_path().read_text().replace("[prediction]\n", "")
    cfg_path = tmp_path / "missing.toml"
    cfg_path.write_text(text)

    with pytest.raises(ValueError, match="missing key: config.prediction"):
        load_experiment_config(cfg_path)


def test_loader_surfaces_invalid_nested_values(tmp_path: Path) -> None:
    text = (
        _config_path()
        .read_text()
        .replace(
            "reconciliation_tolerance = 1e-9",
            "reconciliation_tolerance = 0.0",
        )
    )
    cfg_path = tmp_path / "invalid.toml"
    cfg_path.write_text(text)

    with pytest.raises(ValueError, match="reconciliation_tolerance"):
        load_experiment_config(cfg_path)


def test_loader_surfaces_invalid_bayesian_nested_values(tmp_path: Path) -> None:
    text = _config_path().read_text().replace(
        "maximum_rhat = 1.05",
        "maximum_rhat = 0.99",
        1,
    )
    cfg_path = tmp_path / "invalid-bayesian.toml"
    cfg_path.write_text(text)

    with pytest.raises(ValueError, match="maximum_rhat"):
        load_experiment_config(cfg_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "kappa_log_scale = 1.0",
            "",
            "missing key: config.bayesian_priors.kappa_log_scale",
        ),
        (
            "draws = 200",
            "draws = 200\nunknown_option = true",
            "unknown key: config.bayesian_prior_predictive.unknown_option",
        ),
        (
            "minimum_expected_share_ratio = 0.25",
            "maximum_boundary_cohort_share = 1.5",
            "unknown key: config.bayesian_prior_predictive.maximum_boundary_cohort_share",
        ),
    ],
)
def test_loader_requires_exact_bayesian_section_keys(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    cfg_path = tmp_path / "invalid-bayesian-keys.toml"
    cfg_path.write_text(_config_path().read_text().replace(old, new, 1))

    with pytest.raises(ValueError, match=message):
        load_experiment_config(cfg_path)
