"""The searched hyperparameters of each estimator family, as plain records.

Hyperparameters only, no Optuna: the field names are the estimator's
keywords, so a component makes its estimator as
``Estimator(<fixed settings>, **asdict(params))``. A typo raises ``TypeError``
when the record is built, where LightGBM would accept it silently.
"""

from __future__ import annotations

import dataclasses

__all__ = [
    "EstimatorParams",
    "LGBMParams",
    "LogisticParams",
    "NB2Params",
    "PoissonGLMParams",
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LGBMParams:
    """LightGBM's searched hyperparameters, for the regressor and the classifier alike."""

    max_depth: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    n_estimators: int
    reg_alpha: float
    reg_lambda: float
    min_split_gain: float
    subsample: float
    colsample_bytree: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class PoissonGLMParams:
    alpha: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class NB2Params:
    dispersion: float
    alpha: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class LogisticParams:
    C: float


# The searched hyperparameters of any family: one of the records above.
type EstimatorParams = LGBMParams | PoissonGLMParams | NB2Params | LogisticParams
