"""How Optuna searches, and which trial won.

The study accepts any ``trial -> float`` objective, lower is better, such as
a lambda around :meth:`CVHyperparameterEvaluator.evaluate`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import optuna
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import FrozenTrial, TrialState
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .._config import STRICT
from .aggregation import corrected_std_error

__all__ = ["HyperparameterStudy", "TrialRecord", "TuningResult"]


@dataclass(frozen=True)
class TrialRecord:
    """One trial that ran, with the metric in its own sign.

    ``value`` and ``fold_values`` undo the negation the evaluator applies to
    a greater-is-better metric (D12), from the sign it recorded on this trial;
    the search compared ``-value`` then, kept in ``optuna_study_``. All but
    ``number``, ``state`` and ``params`` are ``None`` unless the trial is
    COMPLETE; the folds are ``None`` too for an objective that does not
    record them.
    """

    number: int
    state: TrialState
    params: dict[str, Any]
    value: float | None
    fold_values: tuple[float, ...] | None
    fold_sizes: tuple[int, ...] | None
    std_error: float | None


@dataclass(frozen=True)
class TuningResult:
    """The outcome of one ``optimize``: the best COMPLETE trial and every trial that ran.

    ``best_value`` is the best trial's aggregated metric in its own sign;
    ``metric_name`` is the evaluator's, ``None`` for a plain objective.
    """

    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int
    metric_name: str | None
    trials: tuple[TrialRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dict, e.g. for MLflow: the state by name, tuples as lists."""
        return {
            "best_params": dict(self.best_params),
            "best_value": self.best_value,
            "best_trial_number": self.best_trial_number,
            "metric_name": self.metric_name,
            "trials": [_record_dict(record) for record in self.trials],
        }


def _record_dict(record: TrialRecord) -> dict[str, Any]:
    as_dict = asdict(record)
    # TrialState is an IntEnum: json would write it as a bare number.
    as_dict["state"] = record.state.name
    for key in ("fold_values", "fold_sizes"):
        if as_dict[key] is not None:
            as_dict[key] = list(as_dict[key])
    return as_dict


def _sign(trial: FrozenTrial) -> float:
    # The evaluator negates a greater-is-better metric (D12); this undoes it.
    # Absent (a plain objective): the value is the score.
    return -1.0 if trial.user_attrs.get("greater_is_better", False) else 1.0


def _trial_record(trial: FrozenTrial) -> TrialRecord:
    # Copies: get_trials(deepcopy=False) returns Optuna's own objects.
    if trial.state != TrialState.COMPLETE:
        # A pruned trial's Optuna value is its last intermediate, not a CV score.
        return TrialRecord(trial.number, trial.state, dict(trial.params), *(None,) * 4)
    sign = _sign(trial)
    fold_scores = trial.user_attrs.get("fold_scores")
    fold_sizes = trial.user_attrs.get("fold_sizes")
    return TrialRecord(
        number=trial.number,
        state=trial.state,
        params=dict(trial.params),
        value=sign * trial.values[0],
        fold_values=None
        if fold_scores is None
        else tuple(sign * score for score in fold_scores),
        fold_sizes=None if fold_sizes is None else tuple(fold_sizes),
        std_error=None if fold_scores is None else corrected_std_error(fold_scores),
    )


# Not frozen: optimize keeps the Optuna study as ``optuna_study_``.
@pydantic_dataclass(
    kw_only=True, config=ConfigDict(**STRICT, arbitrary_types_allowed=True)
)
class HyperparameterStudy:
    """Settings of one Optuna search; the study always minimizes.

    ``seed`` seeds the default sampler, ``TPESampler(seed=seed,
    multivariate=True)``, which is built afresh for each ``optimize``, so two
    calls on one study draw the same trials. With a given ``sampler`` the seed
    is unused, but still recorded. The default pruner, ``NopPruner``, prunes
    nothing. The trials run one after the other, so a seeded sampler
    reproduces the search.

    After ``optimize``, the Optuna study is ``optuna_study_`` (e.g. for
    ``optuna.visualization``), kept even when the search raised.
    """

    seed: int
    n_trials: int = 50
    sampler: BaseSampler | None = None
    pruner: BasePruner | None = None

    def __post_init__(self) -> None:
        # Optuna accepts this and runs no trial at all.
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {self.n_trials}")

    def optimize(self, objective: Callable[[optuna.Trial], float]) -> TuningResult:
        """Run the search and return its best COMPLETE trial.

        A tie goes to the lowest trial number. Raises ``RuntimeError`` when no
        trial completed.
        """
        study = self._create_optuna_study()
        self.optuna_study_ = study
        study.optimize(objective, n_trials=self.n_trials)
        complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if len(complete) == 0:
            raise RuntimeError("no trial completed: every trial was pruned")
        # COMPLETE only (D8): a pruned trial's value is its last intermediate.
        # The trials are in number order, and min keeps the first minimum.
        best = min(complete, key=lambda trial: trial.values[0])
        # FAIL too: a plain objective's NaN fails its trial and the search goes on.
        ran = study.get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
        )
        return TuningResult(
            # A copy: Optuna's own dict, so editing the result would edit the study.
            best_params=dict(best.params),
            best_value=_sign(best) * best.values[0],
            best_trial_number=best.number,
            metric_name=best.user_attrs.get("metric_name"),
            trials=tuple(_trial_record(trial) for trial in ran),
        )

    def _create_optuna_study(self) -> optuna.Study:
        """A fresh Optuna study with the defaults resolved."""
        if self.sampler is None:
            with warnings.catch_warnings():
                # Optuna marks multivariate TPE experimental. The warning changes
                # no draw, so it is silenced here only.
                warnings.filterwarnings("ignore", message="Argument ``multivariate``")
                sampler: BaseSampler = TPESampler(seed=self.seed, multivariate=True)
        else:
            sampler = self.sampler
        pruner = NopPruner() if self.pruner is None else self.pruner
        return optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
