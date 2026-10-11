"""How Optuna searches, and which trial won.

The study accepts any ``trial -> float`` objective, greater is better, such as
a lambda around :meth:`CVHyperparameterEvaluator.evaluate`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import optuna
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import FrozenTrial, TrialState
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ._config import _STRICT

__all__ = ["HyperparameterStudy", "TuningResult"]


@dataclass(frozen=True)
class TuningResult:
    """The best COMPLETE trial of one ``optimize``."""

    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int


# Not frozen: optimize keeps the Optuna study as ``optuna_study_``.
@pydantic_dataclass(
    kw_only=True, config=ConfigDict(**_STRICT, arbitrary_types_allowed=True)
)
class HyperparameterStudy:
    """Settings of one Optuna search; the study always maximizes.

    ``seed`` seeds the default sampler, ``TPESampler(seed=seed,
    multivariate=True)``, which is built afresh for each ``optimize``, so two
    calls on one study draw the same trials. With a given ``sampler`` the seed
    is unused, but still recorded. The default pruner, ``NopPruner``, prunes
    nothing. The search stops after ``n_trials`` trials or ``timeout_seconds``,
    whichever comes first. ``initial_params`` are tried first, in order.

    After ``optimize``, the Optuna study is ``optuna_study_`` (e.g. for
    ``optuna.visualization``), kept even when the search raised.
    """

    seed: int
    n_trials: int | None = 50
    timeout_seconds: float | None = None
    n_jobs: int = 1
    sampler: BaseSampler | None = None
    pruner: BasePruner | None = None
    # A list or tuple; a set's order varies between runs.
    initial_params: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        # Optuna accepts these and runs forever, or runs no trial at all.
        if self.n_trials is None and self.timeout_seconds is None:
            raise ValueError(
                "n_trials and timeout_seconds are both None: the study would never stop"
            )
        if self.n_trials is not None and self.n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {self.n_trials}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        # Optuna leaves the extra enqueued trials waiting, never run.
        if self.n_trials is not None and len(self.initial_params) > self.n_trials:
            raise ValueError(
                f"initial_params holds {len(self.initial_params)} trials but "
                f"n_trials is {self.n_trials}: the rest would never run"
            )

    def optimize(self, objective: Callable[[optuna.Trial], float]) -> TuningResult:
        """Run the search and return its best COMPLETE trial.

        A tie goes to the lowest trial number. Raises ``RuntimeError`` when no
        trial completed, and ``ValueError`` when an ``initial_params`` name was
        not suggested by its trial (no conditional parameters, D9).
        """
        study = self._create_optuna_study()
        for params in self.initial_params:
            study.enqueue_trial(dict(params))
        self.optuna_study_ = study
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            n_jobs=self.n_jobs,
            callbacks=[self._check_initial_params],
        )
        complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if len(complete) == 0:
            raise RuntimeError(
                "no trial completed: every trial was pruned or the timeout came first"
            )
        # Not study.best_trial: Optuna gives a tie to the first trial to finish,
        # which with n_jobs > 1 need not be the lowest number. The trials are in
        # number order, and max keeps the first maximum.
        best = max(complete, key=lambda trial: trial.values[0])
        return TuningResult(
            # A copy: Optuna's own dict, so editing the result would edit the study.
            best_params=dict(best.params),
            best_value=best.values[0],
            best_trial_number=best.number,
        )

    def _check_initial_params(self, study: optuna.Study, trial: FrozenTrial) -> None:
        # Optuna drops an enqueued name that no suggest asks for, silently.
        # Enqueued trials are numbered 0..k-1 when enqueued. COMPLETE only: a
        # pruned objective may stop before suggesting every name.
        if trial.state != TrialState.COMPLETE or trial.number >= len(
            self.initial_params
        ):
            return
        unused = sorted(set(self.initial_params[trial.number]) - set(trial.params))
        if unused:
            raise ValueError(
                f"initial_params[{trial.number}]: no trial suggested {unused}; "
                "check the names"
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
        return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
