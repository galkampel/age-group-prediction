"""Tests for the builders: the sharing cases, the read-back of a trial's draws, what the build passes on."""

from __future__ import annotations

import dataclasses
from typing import Any

import optuna
import pytest
from optuna.samplers import RandomSampler
from optuna.trial import Trial

from age_group_prediction.feature_engineering import FeatureTransformer, Plan
from age_group_prediction.hyperparameter_tuning import (
    CountComponent,
    LGBMParams,
    LogisticParams,
    NB2Params,
    ProbabilityComponent,
    build_independent_cohorts_model,
    build_total_times_probability_model,
    suggest_independent_cohorts_model,
    suggest_total_times_probability_model,
)

PASSTHROUGH = Plan(name="passthrough")
TRANSFORMERS = {
    "ses_only": FeatureTransformer(column_plans=((PASSTHROUGH, ["ses"]),)),
    "median_age_only": FeatureTransformer(
        column_plans=((PASSTHROUGH, ["median_age"]),)
    ),
}
# The probabilities' own candidates, under other names, as in the notebook: a
# part resolving its draw through the other part's dict fails.
PROBABILITY_TRANSFORMERS = {
    "probability__ses": FeatureTransformer(column_plans=((PASSTHROUGH, ["ses"]),)),
    "probability__median_age": FeatureTransformer(
        column_plans=((PASSTHROUGH, ["median_age"]),)
    ),
}
COHORTS = ("n_kindergarten", "n_elementary", "n_highschool")
Y_MIN = 1e-9
SHARING = [(False, False), (True, False), (True, True)]
LGBM = LGBMParams(
    max_depth=3,
    num_leaves=7,
    min_child_samples=5,
    learning_rate=0.1,
    n_estimators=50,
    reg_alpha=1e-8,
    reg_lambda=1e-8,
    min_split_gain=0.0,
    subsample=0.8,
    colsample_bytree=0.9,
)


def _trial(seed: int) -> Trial:
    return optuna.create_study(sampler=RandomSampler(seed=seed)).ask()


def _count_components(params: dict[str, Any], *, family: Any) -> list[CountComponent]:
    """The components ``CountComponent.from_params`` reads back from ``params``, one per cohort."""
    return [
        CountComponent.from_params(
            params,
            name=cohort,
            family=family,
            use_exposure=True,
            feature_transformer_dict=TRANSFORMERS,
        )
        for cohort in COHORTS
    ]


@pytest.mark.parametrize(("share_hyperparams", "share_feature_transformer"), SHARING)
def test_a_shared_choice_is_one_draw_in_every_cohorts_component(
    share_hyperparams: bool, share_feature_transformer: bool
) -> None:
    # Independent: each of the three cohorts draws LightGBM's ten
    # hyperparameters and its transformer, 33 parameters. Shared
    # hyperparameters: one draw under `shared__`, each cohort still drawing its
    # transformer, 13. Both shared: 11, one transformer in every component.
    trial = _trial(0)
    suggest_independent_cohorts_model(
        trial,
        family="lgbm_poisson",
        use_exposure=True,
        cohorts=COHORTS,
        feature_transformer_dict=TRANSFORMERS,
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=Y_MIN,
        random_state=0,
    )
    names = sorted(trial.params)
    shared_names = [name for name in names if name.startswith("shared__")]
    transformer_names = [
        name for name in names if name.endswith("__feature_transformer")
    ]
    components = _count_components(trial.params, family="lgbm_poisson")
    if share_feature_transformer:
        assert transformer_names == ["shared__feature_transformer"]
        assert len({id(c.feature_transformer) for c in components}) == 1
    else:
        assert transformer_names == sorted(
            f"{cohort}__feature_transformer" for cohort in COHORTS
        )
    if share_hyperparams:
        assert len(names) == (11 if share_feature_transformer else 13)
        assert len(shared_names) == (11 if share_feature_transformer else 10)
        assert len({c.estimator_params for c in components}) == 1
    else:
        assert len(names) == 33
        assert shared_names == []
        assert all(name.split("__")[0] in COHORTS for name in names)


@pytest.mark.parametrize("family", ["lgbm_poisson", "nb2"])
@pytest.mark.parametrize(("share_hyperparams", "share_feature_transformer"), SHARING)
def test_the_independent_cohorts_model_read_back_from_a_trials_params_is_the_one_drawn(
    family: str, share_hyperparams: bool, share_feature_transformer: bool
) -> None:
    # The refit reads the best components from best_params (D25): the model
    # built from them must be the model the objective built from the draws.
    trial = _trial(1)
    drawn = suggest_independent_cohorts_model(
        trial,
        family=family,  # type: ignore[arg-type]
        use_exposure=True,
        cohorts=COHORTS,
        feature_transformer_dict=TRANSFORMERS,
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=Y_MIN,
        random_state=0,
    )
    read = build_independent_cohorts_model(
        _count_components(trial.params, family=family), y_min=Y_MIN, random_state=0
    )
    assert list(drawn.cohort_models) == list(read.cohort_models) == list(COHORTS)
    for cohort in COHORTS:
        drawn_model, read_model = (
            drawn.cohort_models[cohort],
            read.cohort_models[cohort],
        )
        assert read_model.estimator.get_params() == drawn_model.estimator.get_params()
        assert read_model.feature_transformer is drawn_model.feature_transformer


@pytest.mark.parametrize("probability_family", ["lgbm", "logistic"])
def test_the_total_times_probability_model_read_back_from_a_trials_params_is_the_one_drawn(
    probability_family: str,
) -> None:
    trial = _trial(2)
    drawn = suggest_total_times_probability_model(
        trial,
        total_family="lgbm_gaussian",
        probability_family=probability_family,  # type: ignore[arg-type]
        use_exposure=True,
        total_feature_transformer_dict=TRANSFORMERS,
        probability_feature_transformer_dict=PROBABILITY_TRANSFORMERS,
        y_min=Y_MIN,
        random_state=0,
    )
    # The two parts share LightGBM's parameter names; each reads its own.
    total = CountComponent.from_params(
        trial.params,
        name="total",
        family="lgbm_gaussian",
        use_exposure=True,
        feature_transformer_dict=TRANSFORMERS,
    )
    probability = ProbabilityComponent.from_params(
        trial.params,
        name="probability",
        family=probability_family,  # type: ignore[arg-type]
        feature_transformer_dict=PROBABILITY_TRANSFORMERS,
    )
    read = build_total_times_probability_model(
        total, probability, y_min=Y_MIN, random_state=0
    )
    for part in ("total_model", "probability_model"):
        drawn_part, read_part = getattr(drawn, part), getattr(read, part)
        assert read_part.estimator.get_params() == drawn_part.estimator.get_params()
        assert read_part.feature_transformer is drawn_part.feature_transformer
    assert (
        read.probability_model.calibration_method
        == drawn.probability_model.calibration_method
    )


def test_the_independent_cohorts_build_passes_each_components_settings_on() -> None:
    lgbm_component = CountComponent(
        name="n_highschool",
        family="lgbm_poisson",
        use_exposure=True,
        estimator_params=LGBM,
        feature_transformer=TRANSFORMERS["ses_only"],
    )
    components = [
        CountComponent(
            name=cohort,
            family="nb2",
            use_exposure=use_exposure,
            estimator_params=NB2Params(dispersion=0.15, alpha=alpha),
            feature_transformer=TRANSFORMERS[name],
        )
        for cohort, use_exposure, alpha, name in [
            ("n_kindergarten", True, 0.01, "ses_only"),
            ("n_elementary", False, 0.1, "median_age_only"),
        ]
    ]
    model = build_independent_cohorts_model(
        [*components, lgbm_component], y_min=Y_MIN, random_state=3
    )
    assert list(model.cohort_models) == [
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
    ]
    assert (
        model.cohort_models["n_highschool"].estimator.get_params()["random_state"] == 3
    )
    for component, alpha in zip(components, (0.01, 0.1), strict=True):
        count_model = model.cohort_models[component.name]
        assert count_model.use_exposure is component.use_exposure
        assert count_model.feature_transformer is component.feature_transformer
        assert count_model.y_min == Y_MIN
        assert count_model.estimator.get_params()["alpha"] == alpha


def test_the_total_times_probability_build_fixes_the_replication_and_passes_the_rest_on() -> (
    None
):
    total = CountComponent(
        name="total",
        family="lgbm_poisson",
        use_exposure=False,
        estimator_params=LGBM,
        feature_transformer=TRANSFORMERS["ses_only"],
    )
    probability = ProbabilityComponent(
        name="probability",
        family="logistic",
        calibration_method="temperature",
        estimator_params=LogisticParams(C=2.0),
        feature_transformer=TRANSFORMERS["median_age_only"],
    )
    model = build_total_times_probability_model(
        total, probability, y_min=Y_MIN, random_state=5
    )
    assert model.total_model.use_exposure is False
    assert model.total_model.y_min == Y_MIN
    assert model.total_model.estimator.get_params()["random_state"] == 5
    assert model.total_model.feature_transformer is TRANSFORMERS["ses_only"]
    # Fixed, not searched (D24.1).
    assert model.probability_model.replication_method == "weighted"
    assert model.probability_model.calibration_method == "temperature"
    assert model.probability_model.estimator.get_params()["C"] == 2.0
    assert (
        model.probability_model.feature_transformer is TRANSFORMERS["median_age_only"]
    )
    lgbm_probability = dataclasses.replace(
        probability, family="lgbm", estimator_params=LGBM
    )
    model = build_total_times_probability_model(
        total, lgbm_probability, y_min=Y_MIN, random_state=5
    )
    assert model.probability_model.estimator.get_params()["random_state"] == 5
