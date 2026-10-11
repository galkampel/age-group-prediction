"""Tests for the study's settings and defaults."""

from __future__ import annotations

from typing import Any

import optuna
import pytest
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler

from age_group_prediction.hyperparameter_tuning import HyperparameterStudy

# Past TPE's 10 random start-up trials, so the model-based draws are compared.
N_TRIALS = 12


def _objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0.0, 1.0)
    k = trial.suggest_int("k", 0, 9)
    return -((x - 0.3) ** 2) - (k - 4) ** 2


def _params(study: optuna.Study) -> list[dict[str, Any]]:
    study.optimize(_objective, n_trials=N_TRIALS)
    return [trial.params for trial in study.trials]


def test_the_default_sampler_is_seeded_by_the_study() -> None:
    study = HyperparameterStudy(seed=7)
    first, second = study._create_optuna_study(), study._create_optuna_study()
    assert isinstance(first.sampler, TPESampler)
    assert _params(first) == _params(second)


def test_the_default_sampler_is_multivariate() -> None:
    sampler = HyperparameterStudy(seed=0)._create_optuna_study().sampler
    assert isinstance(sampler, TPESampler)
    # Optuna keeps the flag private; there is no public getter.
    assert sampler._multivariate is True


def test_the_default_pruner_never_prunes() -> None:
    assert isinstance(
        HyperparameterStudy(seed=0)._create_optuna_study().pruner, NopPruner
    )


def test_a_given_sampler_and_pruner_are_used_as_given() -> None:
    sampler = optuna.samplers.RandomSampler(seed=0)
    pruner = optuna.pruners.MedianPruner()
    study = HyperparameterStudy(
        seed=0, sampler=sampler, pruner=pruner
    )._create_optuna_study()
    assert study.sampler is sampler
    assert study.pruner is pruner


@pytest.mark.filterwarnings("error")
def test_the_default_sampler_does_not_warn() -> None:
    HyperparameterStudy(seed=0)._create_optuna_study()


def test_a_study_with_no_stopping_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="never stop"):
        HyperparameterStudy(seed=0, n_trials=None)


@pytest.mark.parametrize("n_trials", [0, -1])
def test_fewer_than_one_trial_is_rejected(n_trials: int) -> None:
    with pytest.raises(ValueError, match="n_trials must be at least 1"):
        HyperparameterStudy(seed=0, n_trials=n_trials)


@pytest.mark.parametrize("timeout_seconds", [0, -1.0])
def test_a_non_positive_timeout_is_rejected(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        HyperparameterStudy(seed=0, n_trials=None, timeout_seconds=timeout_seconds)


def test_a_misspelt_setting_is_rejected() -> None:
    # Pydantic's default ignores extras: n_trials would silently stay 50.
    with pytest.raises(ValueError, match="Unexpected keyword argument"):
        HyperparameterStudy(seed=0, n_trial=10)  # type: ignore[call-arg]


def test_a_bool_seed_is_rejected() -> None:
    # Lax pydantic would read True as 1.
    with pytest.raises(ValueError, match="valid integer"):
        HyperparameterStudy(seed=True)
