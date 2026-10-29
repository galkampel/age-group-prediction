"""Hyperparameter tuning of the rebuilt models with plain Optuna.

:mod:`~age_group_prediction.hyperparameter_tuning.params`
    The searched hyperparameters of each estimator family, as records whose
    fields are the estimator's keywords.
:mod:`~age_group_prediction.hyperparameter_tuning.components`
    A composite's parts as complete records, set from a trial's draws or
    read back from a study's ``best_params``; each makes its estimator.
:mod:`~age_group_prediction.hyperparameter_tuning.search_spaces`
    Every Optuna call: one component's choices, sampled under
    ``<name>__<parameter>``.

The code comes from the experiment notebook's section 5
(``notebooks/02_experiments.py``); the design and its decisions are in
``docs/NOTEBOOK_TUNING_PLAN.md`` (D21, D25).
"""

from __future__ import annotations

from .components import (
    Component,
    CountComponent,
    CountFamily,
    ProbabilityComponent,
    ProbabilityFamily,
)
from .params import (
    EstimatorParams,
    LGBMParams,
    LogisticParams,
    NB2Params,
    PoissonGLMParams,
)
from .search_spaces import (
    ComponentSearchSpace,
    CountSearchSpace,
    ProbabilitySearchSpace,
)

__all__ = [
    "Component",
    "ComponentSearchSpace",
    "CountComponent",
    "CountFamily",
    "CountSearchSpace",
    "EstimatorParams",
    "LGBMParams",
    "LogisticParams",
    "NB2Params",
    "PoissonGLMParams",
    "ProbabilityComponent",
    "ProbabilityFamily",
    "ProbabilitySearchSpace",
]
