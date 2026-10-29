"""The two composite models, built from components or from a trial's draws.

The suggest functions call a search space's methods explicitly and set the
components with constructor calls, the transformer resolved through the
candidate dict; the build functions turn components into models, with no
Optuna. A study is one family combination and exposure (D12): the objective
is ``suggest_independent_cohorts_model`` /
``suggest_total_times_probability_model``, and the best run's refit sets its
components from the study's ``best_params`` with the component types'
``from_params`` (D25) and builds the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from optuna.trial import BaseTrial

from ..feature_engineering import FeatureTransformer
from ..modeling import (
    CohortProbabilityModel,
    CountModel,
    IndependentCohortModels,
    TotalTimesProbabilityModel,
)
from .components import (
    CountComponent,
    CountFamily,
    ProbabilityComponent,
    ProbabilityFamily,
)
from .search_spaces import CountSearchSpace, ProbabilitySearchSpace

__all__ = [
    "build_independent_cohorts_model",
    "build_total_times_probability_model",
    "suggest_independent_cohorts_model",
    "suggest_total_times_probability_model",
]


def build_independent_cohorts_model(
    components: Iterable[CountComponent], *, y_min: float, random_state: int | None
) -> IndependentCohortModels:
    """The independent cohorts model: one CountModel per component, keyed by the component's name, a cohort."""
    return IndependentCohortModels(
        {
            component.name: CountModel(
                estimator=component.make_estimator(random_state=random_state),
                use_exposure=component.use_exposure,
                feature_transformer=component.feature_transformer,
                y_min=y_min,
            )
            for component in components
        }
    )


def suggest_independent_cohorts_model(
    trial: BaseTrial,
    *,
    family: CountFamily,
    use_exposure: bool,
    cohorts: Sequence[str],
    feature_transformer_dict: Mapping[str, FeatureTransformer],
    share_hyperparams: bool,
    share_feature_transformer: bool,
    y_min: float,
    random_state: int | None,
) -> IndependentCohortModels:
    """The independent cohorts model of the components ``trial`` draws for one family and exposure.

    ``share_hyperparams``: one draw of the hyperparameters in every component;
    ``share_feature_transformer``: one draw of the transformer in every
    component; each drawn once, before the loop, under ``shared__``. A
    cohort's own choices are drawn under ``<cohort>__``, its hyperparameters
    then its transformer.
    """
    # A seeded study's result depends on the draw order; with both flags
    # False it is the order the 24 Check C baselines were recorded with (D24.2).
    feature_transformer_names = tuple(feature_transformer_dict)
    shared_search_space = CountSearchSpace(
        name="shared",
        family=family,
        feature_transformer_names=feature_transformer_names,
    )
    if share_hyperparams:
        shared_estimator_params = shared_search_space.suggest_hyperparameters(trial)
    if share_feature_transformer:
        shared_feature_transformer_name = (
            shared_search_space.suggest_feature_transformer_name(trial)
        )
    cohort_components = []
    for cohort in cohorts:
        cohort_search_space = CountSearchSpace(
            name=cohort,
            family=family,
            feature_transformer_names=feature_transformer_names,
        )
        estimator_params = (
            shared_estimator_params
            if share_hyperparams
            else cohort_search_space.suggest_hyperparameters(trial)
        )
        feature_transformer_name = (
            shared_feature_transformer_name
            if share_feature_transformer
            else cohort_search_space.suggest_feature_transformer_name(trial)
        )
        cohort_components.append(
            CountComponent(
                name=cohort,
                family=family,
                use_exposure=use_exposure,
                estimator_params=estimator_params,
                feature_transformer=feature_transformer_dict[feature_transformer_name],
            )
        )
    independent_cohorts_model = build_independent_cohorts_model(
        cohort_components, y_min=y_min, random_state=random_state
    )
    return independent_cohorts_model


def build_total_times_probability_model(
    total_component: CountComponent,
    probability_component: ProbabilityComponent,
    *,
    y_min: float,
    random_state: int | None,
) -> TotalTimesProbabilityModel:
    """The total times probability model: the total's CountModel and the cohort probabilities' classifier."""
    return TotalTimesProbabilityModel(
        total_model=CountModel(
            estimator=total_component.make_estimator(random_state=random_state),
            use_exposure=total_component.use_exposure,
            feature_transformer=total_component.feature_transformer,
            y_min=y_min,
        ),
        # The replication is fixed, not searched: for the logistic
        # classifier the two replications are the same fit, for LightGBM
        # they differed within noise and "per_child" is 2-3x slower
        # (docs/TOTAL_TIMES_PROBABILITY_MODEL.md §3.1).
        probability_model=CohortProbabilityModel(
            estimator=probability_component.make_estimator(random_state=random_state),
            replication_method="weighted",
            calibration_method=probability_component.calibration_method,
            feature_transformer=probability_component.feature_transformer,
        ),
    )


def suggest_total_times_probability_model(
    trial: BaseTrial,
    *,
    total_family: CountFamily,
    probability_family: ProbabilityFamily,
    use_exposure: bool,
    total_feature_transformer_dict: Mapping[str, FeatureTransformer],
    probability_feature_transformer_dict: Mapping[str, FeatureTransformer],
    y_min: float,
    random_state: int | None,
) -> TotalTimesProbabilityModel:
    """The total times probability model of the components ``trial`` draws for one family pair and exposure."""
    # The draws, in order: the total's hyperparameters and transformer under
    # `total__`, then the probabilities' hyperparameters, calibration and
    # transformer under `probability__`.
    total_search_space = CountSearchSpace(
        name="total",
        family=total_family,
        feature_transformer_names=tuple(total_feature_transformer_dict),
    )
    total_component = CountComponent(
        name=total_search_space.name,
        family=total_family,
        use_exposure=use_exposure,
        estimator_params=total_search_space.suggest_hyperparameters(trial),
        feature_transformer=total_feature_transformer_dict[
            total_search_space.suggest_feature_transformer_name(trial)
        ],
    )
    probability_search_space = ProbabilitySearchSpace(
        name="probability",
        family=probability_family,
        feature_transformer_names=tuple(probability_feature_transformer_dict),
    )
    probability_component = ProbabilityComponent(
        name=probability_search_space.name,
        family=probability_family,
        estimator_params=probability_search_space.suggest_hyperparameters(trial),
        calibration_method=probability_search_space.suggest_calibration_method(trial),
        feature_transformer=probability_feature_transformer_dict[
            probability_search_space.suggest_feature_transformer_name(trial)
        ],
    )
    total_times_probability_model = build_total_times_probability_model(
        total_component,
        probability_component,
        y_min=y_min,
        random_state=random_state,
    )
    return total_times_probability_model
