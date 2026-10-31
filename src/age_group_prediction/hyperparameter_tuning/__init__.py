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
:mod:`~age_group_prediction.hyperparameter_tuning.builders`
    The two composite models' components, drawn from a trial, and the
    models built from components.
:mod:`~age_group_prediction.hyperparameter_tuning.cross_validation`
    The tuner: a model's cross-validated score and the study of each model.
:mod:`~age_group_prediction.hyperparameter_tuning.evaluation`
    The refit of a run's best model on the training part, scored on
    training and validation.
:mod:`~age_group_prediction.hyperparameter_tuning.aggregation`
    How a trial's fold scores are combined into its value.

The code comes from the experiment notebook's section 5
(``notebooks/02_experiments.py``); the design and its decisions are in
``docs/NOTEBOOK_TUNING_PLAN.md`` (D21, D25, D26).
"""

from __future__ import annotations

from .aggregation import (
    Aggregator,
    Mean,
    UpperBound,
    WeightedMean,
    corrected_std_error,
)
from .builders import (
    build_independent_cohorts_model,
    build_total_times_probability_model,
    suggest_count_component,
    suggest_independent_cohorts_components,
    suggest_probability_component,
    suggest_total_times_probability_components,
)
from .components import (
    Component,
    CountComponent,
    CountFamily,
    ProbabilityComponent,
    ProbabilityFamily,
)
from .cross_validation import CrossValidationTuner
from .evaluation import fit_and_score
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
    "Aggregator",
    "Component",
    "ComponentSearchSpace",
    "CountComponent",
    "CountFamily",
    "CountSearchSpace",
    "CrossValidationTuner",
    "EstimatorParams",
    "LGBMParams",
    "LogisticParams",
    "Mean",
    "NB2Params",
    "PoissonGLMParams",
    "ProbabilityComponent",
    "ProbabilityFamily",
    "ProbabilitySearchSpace",
    "UpperBound",
    "WeightedMean",
    "build_independent_cohorts_model",
    "build_total_times_probability_model",
    "corrected_std_error",
    "fit_and_score",
    "suggest_count_component",
    "suggest_independent_cohorts_components",
    "suggest_probability_component",
    "suggest_total_times_probability_components",
]
