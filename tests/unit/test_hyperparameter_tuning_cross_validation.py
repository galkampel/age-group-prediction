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
    Mean,
    PoissonGLMParams,
    ProbabilityComponent,
    WeightedMean,
    build_independent_cohorts_model,
    build_total_times_probability_model,
)
from age_group_prediction.modeling import IndependentCohortModels
from age_group_prediction.scoring import Metric, mean_poisson_deviance_loss
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
TUNER = CrossValidationTuner(cv=CV, metric=METRIC, n_trials=3, n_startup_trials=2)
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
    one_after_another = TUNER.cv_score(_model(), X, Y, exposure)
    in_parallel = dataclasses.replace(TUNER, n_jobs=2).cv_score(
        _model(), X, Y, exposure
    )
    assert in_parallel == one_after_another


def _process_id(y_true: object, y_pred: object) -> float:
    """A metric whose value is the process that scored the fold."""
    return float(os.getpid())


def _not_converged(y_true: object, y_pred: object) -> float:
    raise RuntimeError("the fit did not converge")


def test_the_folds_are_fitted_in_n_jobs_processes() -> None:
    # Equal scores cannot show n_jobs ignored; the scoring process can.
    X, Y, exposure = _table()
    tuner = dataclasses.replace(TUNER, metric=Metric("process_id", _process_id))
    assert tuner.cv_score(_model(), X, Y, exposure) == os.getpid()
    in_parallel = dataclasses.replace(tuner, n_jobs=2).cv_score(
        _model(), X, Y, exposure
    )
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
        tuner.cv_score(_model(), X, Y, exposure)
    assert np.isfinite(tuner.cv_score(_model(), X, Y, exposure, groups))


def test_a_runtime_error_in_a_folds_process_fails_its_trial() -> None:
    # NB2's non-convergence is raised in a fold's process and comes back as a
    # RuntimeError, so `catch` marks the trial FAIL there too (D13).
    X, Y, exposure = _table()
    tuner = dataclasses.replace(
        TUNER, metric=Metric("not_converged", _not_converged), n_trials=2, n_jobs=2
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
        TUNER.fold_score(_model(), X, Y, exposure, train_index, val_index)
        for train_index, val_index in folds
    ]
    sizes = [len(val_index) for _, val_index in folds]
    assert len(set(sizes)) > 1  # unequal folds, so the two means differ
    assert TUNER.aggregator == Mean()
    assert TUNER.cv_score(_model(), X, Y, exposure) == pytest.approx(np.mean(scores))
    weighted = dataclasses.replace(TUNER, aggregator=WeightedMean())
    assert weighted.cv_score(_model(), X, Y, exposure) == pytest.approx(
        np.average(scores, weights=sizes)
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

    tuner = dataclasses.replace(TUNER, metric=Metric("fails_once", fails_once))
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

    tuner = dataclasses.replace(TUNER, metric=Metric("a_bug", a_bug))
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
        metric=Metric("cohort_poisson_deviance", mean_poisson_deviance_loss, direction),  # type: ignore[arg-type]
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
        assert trial.value == TUNER.cv_score(model, X, Y, exposure)


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
        assert trial.value == TUNER.cv_score(model, X, Y, exposure)
