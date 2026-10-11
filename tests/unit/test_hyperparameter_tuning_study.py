"""Tests for the study's settings, defaults and search."""

from __future__ import annotations

import time
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


def test_more_initial_params_than_trials_are_rejected() -> None:
    with pytest.raises(ValueError, match="the rest would never run"):
        HyperparameterStudy(seed=0, n_trials=1, initial_params=[{"x": 0.1}, {"x": 0.2}])


def test_the_same_seed_gives_the_same_result() -> None:
    study = HyperparameterStudy(seed=7, n_trials=N_TRIALS)
    first = study.optimize(_objective)
    assert study.optimize(_objective) == first
    assert HyperparameterStudy(seed=7, n_trials=N_TRIALS).optimize(_objective) == first


def test_a_pruned_trial_never_wins() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 1:
            # Optuna keeps this as the pruned trial's value.
            trial.report(100.0, step=0)
            raise optuna.TrialPruned()
        return float(trial.number)

    result = HyperparameterStudy(seed=0, n_trials=3).optimize(objective)
    assert result.best_trial_number == 2


def test_a_tie_goes_to_the_lowest_trial_number() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 0:
            # Trial 0 finishes after trial 1: Optuna's best_trial would be 1.
            time.sleep(0.1)
        return 1.0

    study = HyperparameterStudy(seed=0, n_trials=2, n_jobs=2)
    assert study.optimize(objective).best_trial_number == 0


def test_no_completed_trial_raises() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise optuna.TrialPruned()

    with pytest.raises(RuntimeError, match="no trial completed"):
        HyperparameterStudy(seed=0, n_trials=2).optimize(objective)


def test_initial_params_run_first_in_order() -> None:
    initial = [{"x": 0.9, "k": 1}, {"x": 0.1, "k": 8}]
    study = HyperparameterStudy(seed=0, n_trials=4, initial_params=initial)
    study.optimize(_objective)
    assert [trial.params for trial in study.optuna_study_.trials[:2]] == initial


def test_a_misspelt_initial_param_stops_the_study() -> None:
    study = HyperparameterStudy(
        seed=0, n_trials=5, initial_params=[{"x": 0.5, "k": 2, "kk": 3}]
    )
    with pytest.raises(ValueError, match=r"initial_params\[0\]: .*\['kk'\]"):
        study.optimize(_objective)
    assert len(study.optuna_study_.trials) == 1


def test_the_optuna_study_is_kept() -> None:
    study = HyperparameterStudy(seed=0, n_trials=3)
    study.optimize(_objective)
    assert len(study.optuna_study_.trials) == 3


def test_editing_the_result_leaves_the_study_intact() -> None:
    study = HyperparameterStudy(seed=0, n_trials=3)
    result = study.optimize(_objective)
    result.best_params["x"] = -1.0
    assert study.optuna_study_.trials[result.best_trial_number].params["x"] != -1.0
