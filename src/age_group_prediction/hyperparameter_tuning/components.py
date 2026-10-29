"""The components of a composite model: complete records of one part of a run.

A component is set by its constructor from a trial's draws (the objective) or
by ``from_params`` from a study's ``best_params`` (the refit); no Optuna here.
It makes its estimator; the models are built from components elsewhere.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping
from typing import Any, Literal, Self

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from ..feature_engineering import FeatureTransformer
from ..modeling import (
    CalibrationMethod,
    Classifier,
    ExposureRegressor,
    NegativeBinomialRegressor,
    Regressor,
)
from .params import (
    EstimatorParams,
    LGBMParams,
    LogisticParams,
    NB2Params,
    PoissonGLMParams,
)

__all__ = [
    "Component",
    "CountComponent",
    "CountFamily",
    "ProbabilityComponent",
    "ProbabilityFamily",
]

# LightGBM's two losses are separate families (D6).
type CountFamily = Literal["lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2"]
type ProbabilityFamily = Literal["logistic", "lgbm"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Component(abc.ABC):
    """A named part of a composite: its family, feature transformer and estimator params."""

    name: str
    family: str
    # CountModel and CohortProbabilityModel clone it at fit, so the shared
    # declaration is safe.
    feature_transformer: FeatureTransformer

    @staticmethod
    def set_estimator_params[P: EstimatorParams](
        params: Mapping[str, Any], params_class: type[P]
    ) -> P:
        """The record ``params_class`` filled from ``params`` by its fields; a missing one raises KeyError."""
        return params_class(
            **{
                field.name: params[field.name]
                for field in dataclasses.fields(params_class)
            }
        )

    @abc.abstractmethod
    def make_estimator(
        self, *, random_state: int | None = None
    ) -> Regressor | ExposureRegressor | Classifier:
        """The unfitted estimator: the family's fixed settings and ``estimator_params``."""

    def to_dict(self) -> dict[str, Any]:
        """The record as plain data: the transformer by its declaration (plans, interactions)."""
        record = dataclasses.asdict(self)
        record["feature_transformer"] = self.feature_transformer.to_dict()
        return record


@dataclasses.dataclass(frozen=True, kw_only=True)
class CountComponent(Component):
    """A CountModel's part of a run; ``use_exposure`` makes the target a weighted rate."""

    family: CountFamily
    use_exposure: bool
    estimator_params: LGBMParams | PoissonGLMParams | NB2Params

    @classmethod
    def from_params(
        cls,
        best_params: Mapping[str, Any],
        *,
        name: str,
        family: CountFamily,
        use_exposure: bool,
        feature_transformer_dict: Mapping[str, FeatureTransformer],
    ) -> Self:
        """The component ``best_params`` names under ``name``: its family's record and its transformer, through ``feature_transformer_dict``.

        A choice the independent cohorts model shares sits under
        ``shared__`` and counts as the component's own.
        """
        params = {
            parameter_name.split("__", 1)[1]: value
            for parameter_name, value in best_params.items()
            if parameter_name.split("__", 1)[0] in (name, "shared")
        }
        estimator_params: LGBMParams | PoissonGLMParams | NB2Params
        if family in ("lgbm_poisson", "lgbm_gaussian"):
            estimator_params = cls.set_estimator_params(params, LGBMParams)
        elif family == "poisson_glm":
            estimator_params = cls.set_estimator_params(params, PoissonGLMParams)
        elif family == "nb2":
            estimator_params = cls.set_estimator_params(params, NB2Params)
        else:
            raise ValueError(f"unknown count family {family!r}")
        return cls(
            name=name,
            family=family,
            use_exposure=use_exposure,
            estimator_params=estimator_params,
            feature_transformer=feature_transformer_dict[params["feature_transformer"]],
        )

    def make_estimator(
        self, *, random_state: int | None = None
    ) -> Regressor | ExposureRegressor:
        estimator_params = dataclasses.asdict(self.estimator_params)
        if self.family in ("lgbm_poisson", "lgbm_gaussian"):
            # LightGBM's name for the loss, which the family fixes (D6).
            objective = "poisson" if self.family == "lgbm_poisson" else "regression"
            # subsample_freq=1: LightGBM ignores subsample without it; n_jobs=1: one
            # thread per fit, so the parallel folds do not oversubscribe the cores.
            return LGBMRegressor(
                objective=objective,
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=random_state,
                **estimator_params,
            )
        elif self.family == "poisson_glm":
            return PoissonRegressor(max_iter=1000, **estimator_params)
        elif self.family == "nb2":
            return NegativeBinomialRegressor(**estimator_params)
        else:
            raise ValueError(f"unknown count family {self.family!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ProbabilityComponent(Component):
    """The total times probability model's CohortProbabilityModel part of a run; the exposure does not apply."""

    family: ProbabilityFamily
    calibration_method: CalibrationMethod | None
    estimator_params: LGBMParams | LogisticParams

    @classmethod
    def from_params(
        cls,
        best_params: Mapping[str, Any],
        *,
        name: str,
        family: ProbabilityFamily,
        feature_transformer_dict: Mapping[str, FeatureTransformer],
    ) -> Self:
        """The component ``best_params`` names under ``name``: its family's record, calibration and transformer, through ``feature_transformer_dict``."""
        params = {
            parameter_name.split("__", 1)[1]: value
            for parameter_name, value in best_params.items()
            if parameter_name.split("__", 1)[0] in (name, "shared")
        }
        estimator_params: LGBMParams | LogisticParams
        if family == "lgbm":
            estimator_params = cls.set_estimator_params(params, LGBMParams)
        elif family == "logistic":
            estimator_params = cls.set_estimator_params(params, LogisticParams)
        else:
            raise ValueError(f"unknown probability family {family!r}")
        return cls(
            name=name,
            family=family,
            calibration_method=params["calibration_method"],
            estimator_params=estimator_params,
            feature_transformer=feature_transformer_dict[params["feature_transformer"]],
        )

    def make_estimator(self, *, random_state: int | None = None) -> Classifier:
        estimator_params = dataclasses.asdict(self.estimator_params)
        if self.family == "lgbm":
            return LGBMClassifier(
                objective="multiclass",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=random_state,
                **estimator_params,
            )
        elif self.family == "logistic":
            return LogisticRegression(max_iter=1000, **estimator_params)
        else:
            raise ValueError(f"unknown probability family {self.family!r}")
