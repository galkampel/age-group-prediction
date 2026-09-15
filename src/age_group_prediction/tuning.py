"""Deterministic Optuna study execution shared by model implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import optuna

from .modeling_config import OptunaTuningConfig


@dataclass(frozen=True)
class TrialRecord:
    """JSON-safe summary of one Optuna trial."""

    number: int
    state: str
    value: float | None
    params: dict[str, int | float | str]
    intermediate_values: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class TuningResult:
    """Best parameters and ordered evidence from a completed study."""

    best_params: dict[str, int | float | str]
    best_value: float
    best_trial_number: int
    trials: tuple[TrialRecord, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TuningResult:
        """Rebuild a result from its ``dataclasses.asdict`` JSON round trip."""
        return cls(
            best_params=dict(payload["best_params"]),
            best_value=float(payload["best_value"]),
            best_trial_number=int(payload["best_trial_number"]),
            trials=tuple(
                TrialRecord(
                    number=int(trial["number"]),
                    state=str(trial["state"]),
                    value=None if trial["value"] is None else float(trial["value"]),
                    params=dict(trial["params"]),
                    intermediate_values=tuple(
                        (int(step), float(value))
                        for step, value in trial["intermediate_values"]
                    ),
                )
                for trial in payload["trials"]
            ),
        )


def run_optuna_study(
    objective: Callable[[optuna.Trial], float],
    *,
    config: OptunaTuningConfig,
    seed: int,
) -> TuningResult:
    """Run a seeded single-job minimization study with stable trial summaries."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner: optuna.pruners.BasePruner
    if config.enable_pruning:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    else:
        pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=config.n_trials,
        n_jobs=config.n_jobs,
        timeout=config.timeout_seconds,
        show_progress_bar=False,
    )
    records = tuple(
        TrialRecord(
            number=trial.number,
            state=trial.state.name,
            value=None if trial.value is None else float(trial.value),
            params={key: _json_scalar(value) for key, value in trial.params.items()},
            intermediate_values=tuple(
                (int(step), float(value))
                for step, value in sorted(trial.intermediate_values.items())
            ),
        )
        for trial in study.trials
    )
    # Only completed trials are comparable. Optuna gives a PRUNED trial the
    # value of its last intermediate report, so `value is not None` does not
    # exclude one: callers that report a running mean over folds would let a
    # trial scored on two folds win against trials scored on five. Ties break
    # on trial number so the winner does not depend on iteration order.
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        raise RuntimeError(
            "Optuna study produced no completed trial; every trial was pruned "
            "or failed, so no hyperparameters can be selected"
        )
    best = min(completed, key=lambda trial: (float(trial.value), trial.number))
    return TuningResult(
        best_params={key: _json_scalar(value) for key, value in best.params.items()},
        best_value=float(best.value),
        best_trial_number=best.number,
        trials=records,
    )


def _json_scalar(value: Any) -> int | float | str:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    raise TypeError(f"Optuna parameter is not JSON-safe: {type(value).__name__}")


__all__ = ["TrialRecord", "TuningResult", "run_optuna_study"]