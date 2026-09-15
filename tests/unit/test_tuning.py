"""Tests for reusable deterministic Optuna tuning."""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from age_group_prediction.modeling_config import OptunaTuningConfig
from age_group_prediction.tuning import run_optuna_study


def test_seeded_study_replays_trials_and_winner() -> None:
    config = OptunaTuningConfig(n_trials=8)

    def objective(trial: object) -> float:
        value = trial.suggest_float("value", -2.0, 2.0)  # type: ignore[attr-defined]
        return float((value - 0.25) ** 2)

    first = run_optuna_study(objective, config=config, seed=17)
    second = run_optuna_study(objective, config=config, seed=17)

    assert first == second
    assert len(first.trials) == config.n_trials


def test_partially_evaluated_pruned_trial_never_wins() -> None:
    """A pruned trial is scored on fewer folds, so it must not be selectable.

    Folds here get progressively harder, which is what makes the defect bite:
    the objective reports a running mean, so a trial pruned early is scored on
    its easy folds only and records a lower value than any completed trial.
    """
    difficulty = (0.2, 0.6, 1.0, 1.4, 1.8)

    def objective(trial: optuna.Trial) -> float:
        candidate = trial.suggest_float("value", 0.0, 1.0)
        quality = 1.0 + 2.0 * candidate  # lower `value` is genuinely better
        scores: list[float] = []
        for fold, fold_difficulty in enumerate(difficulty):
            scores.append(quality * fold_difficulty)
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    result = run_optuna_study(
        objective,
        config=OptunaTuningConfig(n_trials=40, enable_pruning=True),
        seed=5,
    )

    states = {record.number: record.state for record in result.trials}
    assert "PRUNED" in states.values(), "the pruning path must actually be exercised"
    assert states[result.best_trial_number] == "COMPLETE"

    # The winner must be the best trial that was scored on every fold.
    completed = [record for record in result.trials if record.state == "COMPLETE"]
    best_completed = min(completed, key=lambda record: (record.value, record.number))
    assert result.best_trial_number == best_completed.number
    assert result.best_value == best_completed.value

    # Pruned trials are still retained as evidence, with their partial scores.
    pruned = [record for record in result.trials if record.state == "PRUNED"]
    assert pruned and all(record.value is not None for record in pruned)
    assert all(len(record.intermediate_values) < len(difficulty) for record in pruned)


def test_study_without_a_completed_trial_is_rejected() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("value", 0.0, 1.0)
        trial.report(0.0, step=0)
        raise optuna.TrialPruned()

    with pytest.raises(RuntimeError, match="no completed trial"):
        run_optuna_study(
            objective,
            config=OptunaTuningConfig(n_trials=4, enable_pruning=True),
            seed=1,
        )


def test_reproducible_config_rejects_parallelism_and_timeout() -> None:
    with pytest.raises(ValueError, match="n_jobs=1"):
        OptunaTuningConfig(n_jobs=2)
    with pytest.raises(ValueError, match="timeout"):
        OptunaTuningConfig(timeout_seconds=10.0)