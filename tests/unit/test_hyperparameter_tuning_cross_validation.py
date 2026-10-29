"""Tests for the cross-validation loop: parallel folds, the aggregator, the study's settings, the refit's table."""

from __future__ import annotations

import os
from functools import partial

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.feature_engineering import FeatureTransformer, Plan
from age_group_prediction.hyperparameter_tuning import (
    CountComponent,
    PoissonGLMParams,
    WeightedMean,
    build_independent_cohorts_model,
    cv_score,
    fit_and_score,
    fold_score,
    suggest_independent_cohorts_model,
    tune_model,
)
from age_group_prediction.modeling import IndependentCohortModels
from age_group_prediction.scoring import Metric, cohort_poisson_deviance
from age_group_prediction.splitting import Splitter
from age_group_prediction.utils import AgeGroupDataset

PASSTHROUGH = Plan(name="passthrough")
TRANSFORMERS = {
    "ses_only": FeatureTransformer(column_plans=((PASSTHROUGH, ["ses"]),)),
    "median_age_only": FeatureTransformer(
        column_plans=((PASSTHROUGH, ["median_age"]),)
    ),
}
COHORTS = ("n_kindergarten", "n_elementary", "n_highschool")
METRIC = Metric(
    "cohort_poisson_deviance", cohort_poisson_deviance, direction="minimize"
)
CV = Splitter("random").cv(n_splits=3, random_state=0)


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


SUGGEST_MODEL = partial(
    suggest_independent_cohorts_model,
    family="poisson_glm",
    use_exposure=True,
    cohorts=COHORTS,
    feature_transformer_dict=TRANSFORMERS,
    share_hyperparams=False,
    share_feature_transformer=False,
    y_min=1e-9,
    random_state=0,
)


def test_the_folds_fitted_in_parallel_score_as_one_after_another() -> None:
    # D13: the folds of a trial in processes give the same value, so a seeded
    # study is unchanged by n_jobs.
    X, Y, exposure = _table()
    one_after_another = cv_score(_model(), X, Y, exposure, CV, None, METRIC)
    in_parallel = cv_score(_model(), X, Y, exposure, CV, None, METRIC, n_jobs=2)
    assert in_parallel == one_after_another


def _process_id(y_true: object, y_pred: object) -> float:
    """A metric whose value is the process that scored the fold."""
    return float(os.getpid())


def _not_converged(y_true: object, y_pred: object) -> float:
    raise RuntimeError("the fit did not converge")


def test_the_folds_are_fitted_in_n_jobs_processes() -> None:
    # Equal scores cannot show n_jobs ignored; the scoring process can.
    X, Y, exposure = _table()
    metric = Metric("process_id", _process_id)
    in_process = cv_score(_model(), X, Y, exposure, CV, None, metric)
    in_parallel = cv_score(_model(), X, Y, exposure, CV, None, metric, n_jobs=2)
    assert in_process == os.getpid()
    assert in_parallel != os.getpid()


def test_a_runtime_error_in_a_folds_process_fails_its_trial() -> None:
    # NB2's non-convergence is raised in a fold's process and comes back as a
    # RuntimeError, so `catch` marks the trial FAIL there too (D13).
    X, Y, exposure = _table()
    study, _ = tune_model(
        SUGGEST_MODEL,
        X,
        Y,
        exposure,
        CV,
        None,
        Metric("not_converged", _not_converged),
        study_seed=0,
        n_trials=2,
        n_startup_trials=2,
        n_jobs=2,
    )
    assert [trial.state for trial in study.trials] == [TrialState.FAIL] * 2


def test_the_fold_scores_are_combined_by_the_aggregator_the_plain_mean_by_default() -> (
    None
):
    X, Y, exposure = _table()
    folds = list(CV.split(X, Y))
    scores = [
        fold_score(_model(), X, Y, exposure, train_index, val_index, METRIC)
        for train_index, val_index in folds
    ]
    sizes = [len(val_index) for _, val_index in folds]
    assert len(set(sizes)) > 1  # unequal folds, so the two means differ
    assert cv_score(_model(), X, Y, exposure, CV, None, METRIC) == pytest.approx(
        np.mean(scores)
    )
    assert cv_score(
        _model(), X, Y, exposure, CV, None, METRIC, aggregator=WeightedMean()
    ) == pytest.approx(np.average(scores, weights=sizes))


def test_a_trial_whose_fit_raises_runtime_error_fails_and_the_study_goes_on() -> None:
    # An NB2 fit that does not converge raises RuntimeError: that trial is
    # marked FAIL and the study runs its other trials (D7, D19).
    X, Y, exposure = _table()

    def suggest_model(trial: optuna.Trial) -> IndependentCohortModels:
        if trial.number == 1:
            raise RuntimeError("the fit did not converge")
        return SUGGEST_MODEL(trial)

    study, seconds = tune_model(
        suggest_model,
        X,
        Y,
        exposure,
        CV,
        None,
        METRIC,
        study_seed=0,
        n_trials=3,
        n_startup_trials=2,
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

    def suggest_model(trial: optuna.Trial) -> IndependentCohortModels:
        raise ValueError("a bug")

    with pytest.raises(ValueError, match="a bug"):
        tune_model(
            suggest_model,
            X,
            Y,
            exposure,
            CV,
            None,
            METRIC,
            study_seed=0,
            n_trials=2,
            n_startup_trials=2,
        )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_the_study_takes_the_metrics_direction_and_the_sampler_its_settings(
    direction: str,
) -> None:
    X, Y, exposure = _table()
    metric = Metric("cohort_poisson_deviance", cohort_poisson_deviance, direction)  # type: ignore[arg-type]
    studies = [
        tune_model(
            SUGGEST_MODEL,
            X,
            Y,
            exposure,
            CV,
            None,
            metric,
            study_seed=7,
            n_trials=2,
            n_startup_trials=5,
        )[0]
        for _ in range(2)
    ]
    study = studies[0]
    assert study.direction == StudyDirection[direction.upper()]
    assert isinstance(study.sampler, TPESampler)
    # optuna keeps the setting in a private attribute; no public reader exists.
    assert study.sampler._n_startup_trials == 5
    # The seed reaches the sampler: a second study draws the same trials.
    assert [trial.params for trial in studies[1].trials] == [
        trial.params for trial in study.trials
    ]


def test_the_refit_scores_each_cohort_then_the_composite_on_both_parts() -> None:
    X, Y, exposure = _table()
    groups = pd.Series(np.arange(len(X)))
    train = AgeGroupDataset(
        X[:150], Y[:150], exposure[:150], groups[:150], groups[:150]
    )
    val = AgeGroupDataset(X[150:], Y[150:], exposure[150:], groups[150:], groups[150:])
    cohort_metric = Metric(
        "poisson_deviance", mean_poisson_deviance, direction="minimize"
    )
    model = _model()
    fitted, score_table = fit_and_score(
        model, train, val, cohort_metric=cohort_metric, metric=METRIC
    )
    assert fitted is model
    assert [row["cohort"] for row in score_table] == [*COHORTS, "composite"]
    y_pred_val = model.predict(val.X, exposure=val.exposure)
    assert score_table[0]["val"] == mean_poisson_deviance(
        val.Y["n_kindergarten"], y_pred_val["n_kindergarten"]
    )
    assert score_table[-1]["val"] == cohort_poisson_deviance(val.Y, y_pred_val)
    assert score_table[-1]["train"] == cohort_poisson_deviance(
        train.Y, model.predict(train.X, exposure=train.exposure)
    )
