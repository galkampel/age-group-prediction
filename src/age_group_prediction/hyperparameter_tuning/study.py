"""How Optuna searches: the study's settings and its seeded defaults.

The study accepts any ``trial -> float`` objective, greater is better, such as
a lambda around :meth:`CVHyperparameterEvaluator.evaluate`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import optuna
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler, TPESampler
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ._config import _STRICT

__all__ = ["HyperparameterStudy"]


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
