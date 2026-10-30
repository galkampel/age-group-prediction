"""Tests for the refit of a run's best model: the score table on training and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.feature_engineering import FeatureTransformer, Plan
from age_group_prediction.hyperparameter_tuning import (
    CountComponent,
    PoissonGLMParams,
    build_independent_cohorts_model,
    fit_and_score,
)
from age_group_prediction.scoring import Metric, mean_poisson_deviance_loss
from age_group_prediction.utils import AgeGroupDataset

SES_ONLY = FeatureTransformer(column_plans=((Plan(name="passthrough"), ["ses"]),))
COHORTS = ("n_kindergarten", "n_elementary", "n_highschool")
METRIC = Metric(
    "cohort_poisson_deviance", mean_poisson_deviance_loss, direction="minimize"
)


def test_the_refit_scores_each_cohort_then_the_composite_on_both_parts() -> None:
    rng = np.random.default_rng(0)
    exposure = pd.Series(rng.integers(5, 40, 200).astype(float))
    X = pd.DataFrame({"ses": rng.normal(size=200)})
    Y = pd.DataFrame(
        {
            cohort: rng.poisson(0.1 * exposure * np.exp(0.5 * X["ses"]))
            for cohort in COHORTS
        }
    )
    groups = pd.Series(np.arange(len(X)))
    train = AgeGroupDataset(
        X[:150], Y[:150], exposure[:150], groups[:150], groups[:150]
    )
    val = AgeGroupDataset(X[150:], Y[150:], exposure[150:], groups[150:], groups[150:])
    cohort_metric = Metric(
        "poisson_deviance", mean_poisson_deviance, direction="minimize"
    )
    model = build_independent_cohorts_model(
        [
            CountComponent(
                name=cohort,
                family="poisson_glm",
                use_exposure=True,
                estimator_params=PoissonGLMParams(alpha=0.1),
                feature_transformer=SES_ONLY,
            )
            for cohort in COHORTS
        ],
        y_min=1e-9,
        random_state=0,
    )
    fitted, score_table = fit_and_score(
        model, train, val, cohort_metric=cohort_metric, metric=METRIC
    )
    assert fitted is model
    assert [row["cohort"] for row in score_table] == [*COHORTS, "composite"]
    y_pred_val = model.predict(val.X, exposure=val.exposure)
    assert score_table[0]["val"] == mean_poisson_deviance(
        val.Y["n_kindergarten"], y_pred_val["n_kindergarten"]
    )
    assert score_table[-1]["val"] == mean_poisson_deviance_loss(val.Y, y_pred_val)
    assert score_table[-1]["train"] == mean_poisson_deviance_loss(
        train.Y, model.predict(train.X, exposure=train.exposure)
    )
