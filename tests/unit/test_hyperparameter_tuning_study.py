"""Tests for the study's settings, defaults and search."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import optuna
import pytest
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from optuna.trial import TrialState

from age_group_prediction.hyperparameter_tuning import (
    HyperparameterStudy,
    corrected_std_error,
)

# Past TPE's 10 random start-up trials, so the model-based draws are compared.
N_TRIALS = 12


def _objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0.0, 1.0)
    k = trial.suggest_int("k", 0, 9)
    return (x - 0.3) ** 2 + (k - 4) ** 2


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


def test_the_sampler_searches_toward_lower_scores() -> None:
    # The best trial is chosen by our own min, so only the sampler and the
    # pruner see the direction; a "maximize" study would search the wrong way.
    study = HyperparameterStudy(seed=0)._create_optuna_study()
    assert study.direction == StudyDirection.MINIMIZE


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
            # Optuna keeps this as the pruned trial's value, below every other.
            trial.report(-100.0, step=0)
            raise optuna.TrialPruned()
        return float(trial.number)

    result = HyperparameterStudy(seed=0, n_trials=3).optimize(objective)
    assert result.best_trial_number == 0


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


FOLD_SCORES = [1.0, 1.5, 0.8]
FOLD_SIZES = [10, 12, 9]


def _evaluator_like(trial: optuna.Trial) -> float:
    """Records what CVHyperparameterEvaluator records, for a loss."""
    trial.suggest_float("x", 0.0, 1.0)
    trial.set_user_attr("fold_scores", FOLD_SCORES)
    trial.set_user_attr("fold_sizes", FOLD_SIZES)
    trial.set_user_attr("metric_name", "poisson_deviance")
    trial.set_user_attr("greater_is_better", False)
    return 1.1


def _negated_r2_like(trial: optuna.Trial) -> float:
    """Records what the evaluator records for a greater-is-better metric."""
    trial.suggest_float("x", 0.0, 1.0)
    trial.set_user_attr("fold_scores", [-0.6, -0.7, -0.8])
    trial.set_user_attr("fold_sizes", FOLD_SIZES)
    trial.set_user_attr("metric_name", "r2")
    trial.set_user_attr("greater_is_better", True)
    return -0.7


def test_a_complete_record_carries_its_folds_and_their_std_error() -> None:
    record = HyperparameterStudy(seed=0, n_trials=1).optimize(_evaluator_like).trials[0]
    assert record.fold_scores == tuple(FOLD_SCORES)
    assert record.fold_sizes == tuple(FOLD_SIZES)
    assert record.std_error == corrected_std_error(FOLD_SCORES)


def test_a_pruned_record_reports_no_score() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 1:
            # Optuna keeps this as the pruned trial's value.
            trial.report(-100.0, step=0)
            raise optuna.TrialPruned()
        return 1.0

    record = HyperparameterStudy(seed=0, n_trials=2).optimize(objective).trials[1]
    assert record.state == TrialState.PRUNED
    assert record.score is None


def test_a_failed_trial_is_recorded() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        # Optuna marks a NaN trial FAIL and goes on, with only a log line.
        return math.nan if trial.number == 0 else 1.0

    result = HyperparameterStudy(seed=0, n_trials=2).optimize(objective)
    assert [record.state for record in result.trials] == [
        TrialState.FAIL,
        TrialState.COMPLETE,
    ]


def test_trials_that_never_ran_are_not_recorded() -> None:
    def objective(trial: optuna.Trial) -> float:
        time.sleep(0.05)
        return trial.suggest_float("x", 0.0, 1.0)

    # The timeout comes during the first trial; the other two stay WAITING.
    study = HyperparameterStudy(
        seed=0,
        n_trials=None,
        timeout_seconds=0.01,
        initial_params=[{"x": 0.1}, {"x": 0.2}, {"x": 0.3}],
    )
    result = study.optimize(objective)
    assert len(study.optuna_study_.trials) == 3
    assert [record.number for record in result.trials] == [0]


def test_editing_a_record_leaves_the_study_intact() -> None:
    study = HyperparameterStudy(seed=0, n_trials=1)
    record = study.optimize(_objective).trials[0]
    record.params["x"] = -1.0
    assert study.optuna_study_.trials[0].params["x"] != -1.0


def test_to_dict_round_trips_through_json() -> None:
    result = HyperparameterStudy(seed=0, n_trials=3).optimize(_evaluator_like)
    as_dict = result.to_dict()
    assert json.loads(json.dumps(as_dict, allow_nan=False)) == as_dict


def test_to_dict_names_the_state() -> None:
    # TrialState is an IntEnum: json would write 1, and 1 == TrialState.COMPLETE,
    # so the round trip alone cannot catch it.
    result = HyperparameterStudy(seed=0, n_trials=1).optimize(_objective)
    assert result.to_dict()["trials"][0]["state"] == "COMPLETE"


def test_a_loss_is_reported_as_it_is() -> None:
    result = HyperparameterStudy(seed=0, n_trials=1).optimize(_evaluator_like)
    record = result.trials[0]
    assert (result.best_score, result.best_value) == (1.1, 1.1)
    assert (record.score, record.value) == (1.1, 1.1)
    assert record.fold_values == record.fold_scores == (1.0, 1.5, 0.8)
    assert result.metric_name == record.metric_name == "poisson_deviance"


def test_a_greater_is_better_metric_is_reported_in_its_own_sign() -> None:
    result = HyperparameterStudy(seed=0, n_trials=1).optimize(_negated_r2_like)
    record = result.trials[0]
    assert (result.best_score, result.best_value) == (-0.7, 0.7)
    assert (record.score, record.value) == (-0.7, 0.7)
    assert record.fold_values == (0.6, 0.7, 0.8)


def test_each_trial_is_reported_in_its_own_metric_sign() -> None:
    # An objective that switches metric between trials: each record must use
    # its own trial's sign, not the best trial's.
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        trial.set_user_attr("greater_is_better", trial.number == 1)
        return 2.0 if trial.number == 0 else -3.0

    result = HyperparameterStudy(seed=0, n_trials=2).optimize(objective)
    assert [record.value for record in result.trials] == [2.0, 3.0]


def test_a_plain_objective_is_taken_as_a_score() -> None:
    result = HyperparameterStudy(seed=0, n_trials=1).optimize(_objective)
    assert result.metric_name is None
    assert result.best_value == result.best_score
    assert result.trials[0].value == result.trials[0].score


@pytest.mark.parametrize(
    ("settings", "reproducible"),
    [
        ({"n_trials": 2}, True),
        ({"n_trials": 2, "timeout_seconds": 60.0}, False),
        ({"n_trials": 2, "n_jobs": 2}, False),
    ],
)
def test_is_reproducible_only_without_a_timeout_or_parallel_jobs(
    settings: dict[str, Any], reproducible: bool
) -> None:
    result = HyperparameterStudy(seed=0, **settings).optimize(_objective)
    assert result.is_reproducible is reproducible
