"""Every Optuna call: the search of one component's choices.

Given a component's name and family, a search space samples its choices
from a trial under ``<name>__<parameter>`` and returns them by value or name
(D10, D11). The trial's params, under those names, are the record the
components read, so a study's ``best_params`` is read the same way (D25);
the search space reads nothing back and builds nothing.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar, cast

from optuna.trial import BaseTrial

from ..modeling import CalibrationMethod
from .components import CountFamily, ProbabilityFamily
from .params import (
    EstimatorParams,
    LGBMParams,
    LogisticParams,
    NB2Params,
    PoissonGLMParams,
)

__all__ = ["ComponentSearchSpace", "CountSearchSpace", "ProbabilitySearchSpace"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ComponentSearchSpace(abc.ABC):
    """The search of one component's choices, given its name, family and transformer candidates."""

    name: str
    family: str
    # The family's candidates by name, in the order a trial draws them; the
    # caller resolves the drawn name through its dict.
    feature_transformer_names: tuple[str, ...]

    @property
    def prefix(self) -> str:
        """The trial's parameter names are ``<name>__<parameter>``."""
        return f"{self.name}__"

    def suggest_feature_transformer_name(self, trial: BaseTrial) -> str:
        """The chosen transformer's name, drawn from the candidates (Optuna's choices must be primitives)."""
        return trial.suggest_categorical(
            f"{self.prefix}feature_transformer",
            list(self.feature_transformer_names),
        )

    @abc.abstractmethod
    def suggest_hyperparameters(self, trial: BaseTrial) -> EstimatorParams:
        """The family's params record: its ``suggest_<family>_hyperparams``."""

    def suggest_lgbm_hyperparams(self, trial: BaseTrial) -> LGBMParams:
        prefix = self.prefix
        return LGBMParams(
            max_depth=trial.suggest_int(f"{prefix}max_depth", 3, 8),
            num_leaves=trial.suggest_int(f"{prefix}num_leaves", 7, 63),
            min_child_samples=trial.suggest_int(f"{prefix}min_child_samples", 5, 40),
            learning_rate=trial.suggest_float(
                f"{prefix}learning_rate", 0.01, 0.2, log=True
            ),
            n_estimators=trial.suggest_int(f"{prefix}n_estimators", 50, 400),
            reg_alpha=trial.suggest_float(f"{prefix}reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float(f"{prefix}reg_lambda", 1e-8, 10.0, log=True),
            min_split_gain=trial.suggest_float(f"{prefix}min_split_gain", 0.0, 1.0),
            subsample=trial.suggest_float(f"{prefix}subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(f"{prefix}colsample_bytree", 0.7, 1.0),
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CountSearchSpace(ComponentSearchSpace):
    """A count component's search: its hyperparameters and feature transformer."""

    family: CountFamily

    def suggest_hyperparameters(
        self, trial: BaseTrial
    ) -> LGBMParams | PoissonGLMParams | NB2Params:
        if self.family in ("lgbm_poisson", "lgbm_gaussian"):
            return self.suggest_lgbm_hyperparams(trial)
        elif self.family == "poisson_glm":
            return self.suggest_poisson_glm_hyperparams(trial)
        elif self.family == "nb2":
            return self.suggest_nb2_hyperparams(trial)
        else:
            raise ValueError(f"unknown count family {self.family!r}")

    def suggest_poisson_glm_hyperparams(self, trial: BaseTrial) -> PoissonGLMParams:
        return PoissonGLMParams(
            alpha=trial.suggest_float(f"{self.prefix}alpha", 1e-4, 10.0, log=True)
        )

    def suggest_nb2_hyperparams(self, trial: BaseTrial) -> NB2Params:
        prefix = self.prefix
        # The estimated dispersions are 0.12 to 0.18; the penalty's path runs
        # from the unpenalized fit at 1e-3 to all-zero coefficients at 1.
        return NB2Params(
            dispersion=trial.suggest_float(f"{prefix}dispersion", 0.01, 1.0, log=True),
            alpha=trial.suggest_float(f"{prefix}alpha", 1e-3, 1.0, log=True),
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ProbabilitySearchSpace(ComponentSearchSpace):
    """A probability component's search: its hyperparameters, calibration and feature transformer."""

    family: ProbabilityFamily
    # No isotonic: its lowest score block can hold no child of a cohort and
    # then maps a building there to probability 0, an infinite log loss
    # (measured 2026-10-15 with the LightGBM classifier).
    CALIBRATION_METHODS: ClassVar[tuple[CalibrationMethod | None, ...]] = (
        None,
        "temperature",
        "sigmoid",
    )

    def suggest_hyperparameters(self, trial: BaseTrial) -> LGBMParams | LogisticParams:
        if self.family == "lgbm":
            return self.suggest_lgbm_hyperparams(trial)
        elif self.family == "logistic":
            return self.suggest_logistic_hyperparams(trial)
        else:
            raise ValueError(f"unknown probability family {self.family!r}")

    def suggest_logistic_hyperparams(self, trial: BaseTrial) -> LogisticParams:
        return LogisticParams(
            C=trial.suggest_float(f"{self.prefix}C", 1e-3, 100.0, log=True)
        )

    def suggest_calibration_method(self, trial: BaseTrial) -> CalibrationMethod | None:
        # Optuna types a mixed list's draw as any choice type; the draw is one
        # of CALIBRATION_METHODS.
        return cast(
            CalibrationMethod | None,
            trial.suggest_categorical(
                f"{self.prefix}calibration_method", list(self.CALIBRATION_METHODS)
            ),
        )
