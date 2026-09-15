"""Strict loader for experiment runtime defaults used by model contracts."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .modeling_config import (
    BayesianConditionalConfig,
    BayesianDiagnosticConfig,
    BayesianPriorConfig,
    BayesianPriorPredictiveConfig,
    BayesianStabilizationConfig,
    DirectCohortConfig,
    EvaluationConfig,
    FoldConfig,
    IndependentSearchSpaceConfig,
    IndependentTotalProbabilityConfig,
    LightGBMSearchSpaceConfig,
    NUTSProfileConfig,
    OptunaTuningConfig,
    OuterSplitConfig,
    PredictionConfig,
    PredictionValidationConfig,
    RandomnessConfig,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable composition of split, randomness, and prediction settings."""

    outer_split: OuterSplitConfig
    folds: FoldConfig
    randomness: RandomnessConfig
    prediction: PredictionConfig
    prediction_validation: PredictionValidationConfig
    evaluation: EvaluationConfig
    tuning: OptunaTuningConfig
    direct_cohort: DirectCohortConfig
    independent_total_probability: IndependentTotalProbabilityConfig
    bayesian_conditional: BayesianConditionalConfig


def _load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file_obj:
        return tomllib.load(file_obj)


def _assert_exact_keys(values: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(values)
    missing = expected.difference(actual)
    unknown = actual.difference(expected)
    messages = [
        *(f"missing key: {path}.{name}" for name in sorted(missing)),
        *(f"unknown key: {path}.{name}" for name in sorted(unknown)),
    ]
    if messages:
        raise ValueError("; ".join(messages))


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise TypeError(f"missing key: config.{name}")
    return section


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load strict model-runtime defaults from a TOML file."""
    raw = _load_toml(path)
    if not isinstance(raw, dict):
        raise TypeError("Experiment config must be a TOML table")
    _assert_exact_keys(
        raw,
        {
            "outer_split",
            "folds",
            "randomness",
            "prediction",
            "prediction_validation",
            "evaluation",
            "tuning",
            "direct_cohort",
            "direct_cohort_search_space",
            "independent_total_probability",
            "independent_total_probability_search_space",
            "bayesian_conditional",
            "bayesian_priors",
            "bayesian_reduced_profile",
            "bayesian_full_profile",
            "bayesian_reduced_diagnostics",
            "bayesian_full_diagnostics",
            "bayesian_prior_predictive",
            "bayesian_stabilization",
        },
        "config",
    )

    outer_split_raw = _require_section(raw, "outer_split")
    folds_raw = _require_section(raw, "folds")
    randomness_raw = _require_section(raw, "randomness")
    prediction_raw = _require_section(raw, "prediction")
    prediction_validation_raw = _require_section(raw, "prediction_validation")
    evaluation_raw = _require_section(raw, "evaluation")
    tuning_raw = _require_section(raw, "tuning")
    direct_cohort_raw = _require_section(raw, "direct_cohort")
    search_space_raw = _require_section(raw, "direct_cohort_search_space")
    independent_raw = _require_section(raw, "independent_total_probability")
    independent_search_raw = _require_section(
        raw, "independent_total_probability_search_space"
    )
    bayesian_raw = _require_section(raw, "bayesian_conditional")
    bayesian_priors_raw = _require_section(raw, "bayesian_priors")
    bayesian_reduced_profile_raw = _require_section(
        raw, "bayesian_reduced_profile"
    )
    bayesian_full_profile_raw = _require_section(raw, "bayesian_full_profile")
    bayesian_reduced_diagnostics_raw = _require_section(
        raw, "bayesian_reduced_diagnostics"
    )
    bayesian_full_diagnostics_raw = _require_section(
        raw, "bayesian_full_diagnostics"
    )
    bayesian_prior_predictive_raw = _require_section(
        raw, "bayesian_prior_predictive"
    )
    bayesian_stabilization_raw = _require_section(raw, "bayesian_stabilization")
    _assert_exact_keys(
        bayesian_raw,
        {"active_profile"},
        "config.bayesian_conditional",
    )
    for section_name, section, config_type in (
        ("bayesian_priors", bayesian_priors_raw, BayesianPriorConfig),
        ("bayesian_reduced_profile", bayesian_reduced_profile_raw, NUTSProfileConfig),
        ("bayesian_full_profile", bayesian_full_profile_raw, NUTSProfileConfig),
        (
            "bayesian_reduced_diagnostics",
            bayesian_reduced_diagnostics_raw,
            BayesianDiagnosticConfig,
        ),
        (
            "bayesian_full_diagnostics",
            bayesian_full_diagnostics_raw,
            BayesianDiagnosticConfig,
        ),
        (
            "bayesian_prior_predictive",
            bayesian_prior_predictive_raw,
            BayesianPriorPredictiveConfig,
        ),
        (
            "bayesian_stabilization",
            bayesian_stabilization_raw,
            BayesianStabilizationConfig,
        ),
    ):
        _assert_exact_keys(
            section,
            {field.name for field in fields(config_type)},
            f"config.{section_name}",
        )

    try:
        tuning = OptunaTuningConfig(**tuning_raw)
        search_space = LightGBMSearchSpaceConfig(
            **{name: tuple(value) for name, value in search_space_raw.items()}
        )
        return ExperimentConfig(
            outer_split=OuterSplitConfig(**outer_split_raw),
            folds=FoldConfig(**folds_raw),
            randomness=RandomnessConfig(**randomness_raw),
            prediction=PredictionConfig(**prediction_raw),
            prediction_validation=PredictionValidationConfig(
                **prediction_validation_raw
            ),
            evaluation=EvaluationConfig(**evaluation_raw),
            tuning=tuning,
            direct_cohort=DirectCohortConfig(
                **direct_cohort_raw,
                tuning=tuning,
                search_space=search_space,
            ),
            independent_total_probability=IndependentTotalProbabilityConfig(
                **{
                    name: tuple(value) if isinstance(value, list) else value
                    for name, value in independent_raw.items()
                },
                tuning=tuning,
                search_space=IndependentSearchSpaceConfig(
                    **{
                        name: tuple(value)
                        for name, value in independent_search_raw.items()
                    }
                ),
                tuning_folds=FoldConfig(**folds_raw),
            ),
            bayesian_conditional=BayesianConditionalConfig(
                **bayesian_raw,
                priors=BayesianPriorConfig(**bayesian_priors_raw),
                reduced_profile=NUTSProfileConfig(**bayesian_reduced_profile_raw),
                full_profile=NUTSProfileConfig(**bayesian_full_profile_raw),
                reduced_diagnostics=BayesianDiagnosticConfig(
                    **bayesian_reduced_diagnostics_raw
                ),
                full_diagnostics=BayesianDiagnosticConfig(
                    **bayesian_full_diagnostics_raw
                ),
                prior_predictive=BayesianPriorPredictiveConfig(
                    **bayesian_prior_predictive_raw
                ),
                stabilization=BayesianStabilizationConfig(
                    **{
                        name: tuple(value) if isinstance(value, list) else value
                        for name, value in bayesian_stabilization_raw.items()
                    }
                ),
            ),
        )
    except TypeError as error:
        # TypeError here means unexpected keys for dataclass constructors.
        message = str(error)
        raise ValueError(message) from error
