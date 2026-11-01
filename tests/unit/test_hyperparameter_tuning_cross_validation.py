"""Tests for the tuner: parallel folds, the aggregator, the study's settings, the model each tune method builds."""

from __future__ import annotations

import dataclasses
import itertools
import os

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from optuna.trial import TrialState

from age_group_prediction.feature_engineering import FeatureTransformer, Plan
from age_group_prediction.hyperparameter_tuning import (
    CountComponent,
    CrossValidationTuner,
    LogisticParams,
    Mean,
    PoissonGLMParams,
    ProbabilityComponent,
    WeightedMean,
    build_independent_cohorts_model,
    build_total_times_probability_model,
)
from age_group_prediction.modeling import IndependentCohortModels
from age_group_prediction.scoring import (
    Metric,
    mean_absolute_error_loss,
    mean_poisson_deviance_loss,
    root_mean_squared_error_loss,
)
from age_group_prediction.splitting import Splitter

PASSTHROUGH = Plan(name="passthrough")
TRANSFORMERS = {
    "ses_only": FeatureTransformer(column_plans=((PASSTHROUGH, ["ses"]),)),
    "median_age_only": FeatureTransformer(
        column_plans=((PASSTHROUGH, ["median_age"]),)
    ),
}
COHORTS = ("n_kindergarten", "n_elementary", "n_highschool")
METRIC = Metric(
    "cohort_poisson_deviance", mean_poisson_deviance_loss, direction="minimize"
)
CV = Splitter("random").cv(n_splits=3, random_state=0)
TUNER = CrossValidationTuner(
    cv=CV, objective_metric=METRIC, n_trials=3, n_startup_trials=2
)
# The independent cohorts model's settings, GLMs for speed.
INDEPENDENT_COHORTS = {
    "family": "poisson_glm",
    "use_exposure": True,
    "cohorts": COHORTS,
    "feature_transformer_dict": TRANSFORMERS,
    "share_hyperparams": False,
    "share_feature_transformer": False,
}


def _table(n_rows: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Features, cohort counts that depend on them, and the exposure."""
    rng = np.random.default_rng(0)
    exposure = pd.Series(rng.integers(5, 40, n_rows).astype(float))
    X = pd.DataFrame(
        {"ses": rng.normal(size=n_rows), "median_age": rng.normal(size=n_rows)}
    )
    Y = pd.DataFrame(
        {
            "n_kindergarten": rng.poisson(0.1 * exposure * np.exp(0.5 * X["ses"])),
            "n_elementary": rng.poisson(
                0.2 * exposure * np.exp(-0.3 * X["median_age"])
            ),
            "n_highschool": rng.poisson(0.1 * exposure),
        }
    )
    return X, Y, exposure


def _model() -> IndependentCohortModels:
    return build_independent_cohorts_model(
        [
            CountComponent(
                name=cohort,
                family="poisson_glm",
                use_exposure=True,
                estimator_params=PoissonGLMParams(alpha=0.1),
                feature_transformer=TRANSFORMERS["ses_only"],
            )
            for cohort in COHORTS
        ],
        y_min=1e-9,
        random_state=0,
    )


def test_the_folds_fitted_in_parallel_score_as_one_after_another() -> None:
    # D13: the folds of a trial in processes give the same value, so a seeded
    # study is unchanged by n_jobs.
    X, Y, exposure = _table()
    one_after_another = TUNER.cross_validate(_model(), X, Y, exposure)[METRIC.name]
    in_parallel = dataclasses.replace(TUNER, n_jobs=2).cross_validate(
        _model(), X, Y, exposure
    )[METRIC.name]
    assert in_parallel == one_after_another


def _process_id(y_true: object, y_pred: object) -> float:
    """A metric whose value is the process that scored the fold."""
    return float(os.getpid())


def _not_converged(y_true: object, y_pred: object) -> float:
    raise RuntimeError("the fit did not converge")


def test_the_folds_are_fitted_in_n_jobs_processes() -> None:
    # Equal scores cannot show n_jobs ignored; the scoring process can.
    X, Y, exposure = _table()
    tuner = dataclasses.replace(
        TUNER, objective_metric=Metric("process_id", _process_id)
    )
    assert tuner.cross_validate(_model(), X, Y, exposure)["process_id"] == os.getpid()
    in_parallel = dataclasses.replace(tuner, n_jobs=2).cross_validate(
        _model(), X, Y, exposure
    )["process_id"]
    assert in_parallel != os.getpid()


def test_the_groups_reach_the_split() -> None:
    # A grouped split without the groups raises; with them, every group lies in
    # one fold's validation rows.
    X, Y, exposure = _table()
    groups = pd.Series(np.arange(len(X)) // 10)
    tuner = dataclasses.replace(
        TUNER, cv=Splitter("grouped").cv(n_splits=3, random_state=0)
    )
    with pytest.raises(ValueError, match="groups"):
        tuner.cross_validate(_model(), X, Y, exposure)[METRIC.name]
    assert np.isfinite(
        tuner.cross_validate(_model(), X, Y, exposure, groups)[METRIC.name]
    )


def test_a_runtime_error_in_a_folds_process_fails_its_trial() -> None:
    # NB2's non-convergence is raised in a fold's process and comes back as a
    # RuntimeError, so `catch` marks the trial FAIL there too (D13).
    X, Y, exposure = _table()
    tuner = dataclasses.replace(
        TUNER,
        objective_metric=Metric("not_converged", _not_converged),
        n_trials=2,
        n_jobs=2,
    )
    study, _ = tuner.tune_independent_cohorts_model(
        X,
        Y,
        exposure,
        **INDEPENDENT_COHORTS,
        y_min=1e-9,
        random_state=0,
        study_seed=0,
    )
    assert [trial.state for trial in study.trials] == [TrialState.FAIL] * 2


def test_the_fold_scores_are_combined_by_the_aggregator_the_plain_mean_by_default() -> (
    None
):
    X, Y, exposure = _table()
    folds = list(CV.split(X, Y))
    scores = [
        TUNER.score_fold(_model(), X, Y, exposure, train_index, val_index)[METRIC.name]
        for train_index, val_index in folds
    ]
    sizes = [len(val_index) for _, val_index in folds]
    assert len(set(sizes)) > 1  # unequal folds, so the two means differ
    assert TUNER.aggregator == Mean()
    cv_scores = TUNER.cross_validate(_model(), X, Y, exposure)
    assert cv_scores[METRIC.name] == pytest.approx(np.mean(scores))
    weighted = dataclasses.replace(TUNER, aggregator=WeightedMean())
    weighted_cv_scores = weighted.cross_validate(_model(), X, Y, exposure)
    assert weighted_cv_scores[METRIC.name] == pytest.approx(
        np.average(scores, weights=sizes)
    )


def test_a_trial_records_every_metric_and_its_value_is_the_objectives() -> None:
    # Each recorded metric's user attribute is the cv a study on that metric
    # would score, from the same folds; the value is the objective's.
    X, Y, exposure = _table()
    recorded = (
        Metric("mae", mean_absolute_error_loss),
        Metric("rmse", root_mean_squared_error_loss),
    )
    tuner = dataclasses.replace(TUNER, n_trials=2, recorded_metrics=recorded)
    study, _ = tuner.tune_independent_cohorts_model(
        X, Y, exposure, **INDEPENDENT_COHORTS, y_min=1e-9, random_state=0, study_seed=0
    )
    for trial in _completed(study):
        components = [
            CountComponent.from_params(
                trial.params,
                name=cohort,
                family="poisson_glm",
                use_exposure=True,
                feature_transformer_dict=TRANSFORMERS,
            )
            for cohort in COHORTS
        ]
        model = build_independent_cohorts_model(components, y_min=1e-9, random_state=0)
        assert set(trial.user_attrs) == {"mae", "rmse", METRIC.name}
        for metric in (*recorded, METRIC):
            assert (
                trial.user_attrs[metric.name]
                == dataclasses.replace(TUNER, objective_metric=metric).cross_validate(
                    model, X, Y, exposure
                )[metric.name]
            )
        assert trial.value == trial.user_attrs[METRIC.name]


def test_a_recorded_metric_named_as_the_objective_does_not_replace_its_score() -> None:
    X, Y, exposure = _table()
    namesake = Metric(METRIC.name, mean_absolute_error_loss)
    tuner = dataclasses.replace(TUNER, recorded_metrics=(namesake,))
    assert (
        tuner.cross_validate(_model(), X, Y, exposure)[METRIC.name]
        == TUNER.cross_validate(_model(), X, Y, exposure)[METRIC.name]
    )


def test_a_trial_whose_fit_raises_runtime_error_fails_and_the_study_goes_on() -> None:
    # An NB2 fit that does not converge raises RuntimeError: that trial is
    # marked FAIL and the study runs its other trials (D7, D19). In this
    # process, the second trial's first fold raises.
    X, Y, exposure = _table()
    calls = itertools.count()

    def fails_once(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        if next(calls) == CV.get_n_splits():
            raise RuntimeError("the fit did not converge")
        return mean_poisson_deviance_loss(y_true, y_pred)

    tuner = dataclasses.replace(
        TUNER, objective_metric=Metric("fails_once", fails_once)
    )
    study, seconds = tuner.tune_independent_cohorts_model(
        X,
        Y,
        exposure,
        **INDEPENDENT_COHORTS,
        y_min=1e-9,
        random_state=0,
        study_seed=0,
    )
    assert [trial.state for trial in study.trials] == [
        TrialState.COMPLETE,
        TrialState.FAIL,
        TrialState.COMPLETE,
    ]
    assert seconds > 0


def test_any_other_error_in_a_trial_ends_the_study() -> None:
    # Only RuntimeError is caught: a bug should stop the run, not hide in it.
    X, Y, exposure = _table()

    def a_bug(y_true: object, y_pred: object) -> float:
        raise ValueError("a bug")

    tuner = dataclasses.replace(TUNER, objective_metric=Metric("a_bug", a_bug))
    with pytest.raises(ValueError, match="a bug"):
        tuner.tune_independent_cohorts_model(
            X,
            Y,
            exposure,
            **INDEPENDENT_COHORTS,
            y_min=1e-9,
            random_state=0,
            study_seed=0,
        )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_the_study_takes_the_metrics_direction_and_the_sampler_its_settings(
    direction: str,
) -> None:
    X, Y, exposure = _table()
    tuner = dataclasses.replace(
        TUNER,
        objective_metric=Metric(
            "cohort_poisson_deviance", mean_poisson_deviance_loss, direction
        ),  # type: ignore[arg-type]
        n_trials=2,
        n_startup_trials=5,
    )
    studies = [
        tuner.tune_independent_cohorts_model(
            X,
            Y,
            exposure,
            **INDEPENDENT_COHORTS,
            y_min=1e-9,
            random_state=0,
            study_seed=7,
        )[0]
        for _ in range(2)
    ]
    study = studies[0]
    assert study.direction == StudyDirection[direction.upper()]
    assert len(study.trials) == 2
    assert isinstance(study.sampler, TPESampler)
    # optuna keeps the setting in a private attribute; no public reader exists.
    assert study.sampler._n_startup_trials == 5
    # The seed reaches the sampler: a second study draws the same trials.
    assert [trial.params for trial in studies[1].trials] == [
        trial.params for trial in study.trials
    ]


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    assert len(trials) == len(study.trials)
    return trials


def test_the_independent_cohorts_study_scores_the_model_its_components_build() -> None:
    # Each trial's value is the score of the model built from the trial's
    # components with the given settings, y_min and random_state: rebuilt here
    # from the trial's params, as the refit does (D25). The hyperparameters are
    # shared and the transformers the cohorts' own, so the names show both
    # flags reached the draw. A floor of 2.0 binds on these counts, so it shows too.
    X, Y, exposure = _table()
    settings = INDEPENDENT_COHORTS | {
        "family": "lgbm_poisson",
        "use_exposure": False,
        "share_hyperparams": True,
    }
    study, _ = TUNER.tune_independent_cohorts_model(
        X, Y, exposure, **settings, y_min=2.0, random_state=3, study_seed=0
    )
    for trial in _completed(study):
        assert {name.split("__")[0] for name in trial.params} == {"shared", *COHORTS}
        components = [
            CountComponent.from_params(
                trial.params,
                name=cohort,
                family="lgbm_poisson",
                use_exposure=False,
                feature_transformer_dict=TRANSFORMERS,
            )
            for cohort in COHORTS
        ]
        model = build_independent_cohorts_model(components, y_min=2.0, random_state=3)
        assert trial.value == TUNER.cross_validate(model, X, Y, exposure)[METRIC.name]


def test_the_total_times_probability_study_scores_the_model_its_components_build() -> (
    None
):
    X, Y, exposure = _table()
    study, _ = TUNER.tune_total_times_probability_model(
        X,
        Y,
        exposure,
        total_family="lgbm_gaussian",
        probability_family="lgbm",
        use_exposure=True,
        total_feature_transformer_dict=TRANSFORMERS,
        probability_feature_transformer_dict=TRANSFORMERS,
        y_min=2.0,
        random_state=3,
        study_seed=0,
    )
    for trial in _completed(study):
        model = build_total_times_probability_model(
            CountComponent.from_params(
                trial.params,
                name="total",
                family="lgbm_gaussian",
                use_exposure=True,
                feature_transformer_dict=TRANSFORMERS,
            ),
            ProbabilityComponent.from_params(
                trial.params,
                name="probability",
                family="lgbm",
                feature_transformer_dict=TRANSFORMERS,
            ),
            y_min=2.0,
            random_state=3,
        )
        assert trial.value == TUNER.cross_validate(model, X, Y, exposure)[METRIC.name]


@pytest.mark.parametrize("use_exposure", [True, False])
def test_the_total_study_scores_its_total_with_the_fixed_probability_component(
    use_exposure: bool,
) -> None:
    # Coordinate descent's step for the total: the trials draw the total only,
    # and each value is the score of the drawn total with the given
    # probabilities, rebuilt as the refit does; with the exposure on and off,
    # so a dropped setting fails.
    X, Y, exposure = _table()
    probability = ProbabilityComponent(
        name="probability",
        family="logistic",
        calibration_method="temperature",
        estimator_params=LogisticParams(C=2.0),
        feature_transformer=TRANSFORMERS["median_age_only"],
    )
    study, _ = TUNER.tune_total_component(
        X,
        Y,
        exposure,
        total_family="lgbm_gaussian",
        use_exposure=use_exposure,
        total_feature_transformer_dict=TRANSFORMERS,
        probability_component=probability,
        y_min=2.0,
        random_state=3,
        study_seed=0,
    )
    for trial in _completed(study):
        assert {name.split("__")[0] for name in trial.params} == {"total"}
        total = CountComponent.from_params(
            trial.params,
            name="total",
            family="lgbm_gaussian",
            use_exposure=use_exposure,
            feature_transformer_dict=TRANSFORMERS,
        )
        model = build_total_times_probability_model(
            total, probability, y_min=2.0, random_state=3
        )
        assert trial.value == TUNER.cross_validate(model, X, Y, exposure)[METRIC.name]


@pytest.mark.parametrize("probability_family", ["lgbm", "logistic"])
def test_the_probability_study_scores_its_probabilities_with_the_fixed_total_component(
    probability_family: str,
) -> None:
    # Coordinate descent's step for the probabilities, the mirror of the total's;
    # with both classifier families, so a dropped family fails.
    X, Y, exposure = _table()
    total = CountComponent(
        name="total",
        family="poisson_glm",
        use_exposure=True,
        estimator_params=PoissonGLMParams(alpha=0.1),
        feature_transformer=TRANSFORMERS["ses_only"],
    )
    study, _ = TUNER.tune_probability_component(
        X,
        Y,
        exposure,
        total_component=total,
        probability_family=probability_family,  # type: ignore[arg-type]
        probability_feature_transformer_dict=TRANSFORMERS,
        y_min=2.0,
        random_state=3,
        study_seed=0,
    )
    for trial in _completed(study):
        assert {name.split("__")[0] for name in trial.params} == {"probability"}
        probability = ProbabilityComponent.from_params(
            trial.params,
            name="probability",
            family=probability_family,  # type: ignore[arg-type]
            feature_transformer_dict=TRANSFORMERS,
        )
        model = build_total_times_probability_model(
            total, probability, y_min=2.0, random_state=3
        )
        assert trial.value == TUNER.cross_validate(model, X, Y, exposure)[METRIC.name]
