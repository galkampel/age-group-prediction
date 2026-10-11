"""The worked example of ``docs/HYPERPARAMETER_TUNING_PLAN.md`` §5, end to end.

One study per cohort on a small simulated table, then each best parameter set
rebuilt, refitted on the training rows and used to predict the held-out rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from optuna.trial import TrialState

from age_group_prediction.feature_engineering import (
    ColumnPlan,
    FeatureTransformer,
    OneHot,
)
from age_group_prediction.hyperparameter_tuning import (
    CVHyperparameterEvaluator,
    FloatParameter,
    HyperparameterStudy,
    IntParameter,
)
from age_group_prediction.modeling import CountModel
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from age_group_prediction.scoring import POISSON_DEVIANCE
from age_group_prediction.splitting import Splitter
from age_group_prediction.utils import take_rows
from student_simulator import StudentPopulationSimulator, load_simulation_config

COHORTS = ("n_kindergarten", "n_elementary", "n_highschool")

# FEATURE_TRANSFORMATIONS.md §8.0–8.1, copied: the declarations are not in
# src yet (EXPERIMENT_NOTEBOOK_PLAN.md D1 moves them there).
ROOM_SHARES = ("4_rooms_share", "5_rooms_share", "6_rooms_share")
SCHOOL = OneHot(categories=("none", "existing", "planned"), reference_category="none")
tree = FeatureTransformer(
    plans=(
        ColumnPlan(
            name="numeric",
            columns=(
                *ROOM_SHARES,
                "ses",
                "avg_household_size",
                "median_age",
                "n_daycares_500m",
                "n_apartments",
            ),
        ),
        # A tuple: the doc's columns="school_status" runs (a validator wraps
        # it) but ColumnPlan.columns is typed tuple[str, ...], so mypy rejects it.
        ColumnPlan(name="school", columns=("school_status",), transforms=(SCHOOL,)),
    ),
)

# The §5 search space.
PARAMETERS = [
    IntParameter("estimator__max_depth", 3, 8),
    IntParameter("estimator__num_leaves", 7, 63),
    IntParameter("estimator__min_child_samples", 5, 40),
    FloatParameter("estimator__learning_rate", 0.01, 0.2, log=True),
    IntParameter("estimator__n_estimators", 50, 400),
    FloatParameter("estimator__reg_alpha", 1e-8, 10.0, log=True),
    FloatParameter("estimator__reg_lambda", 1e-8, 10.0, log=True),
    FloatParameter("estimator__min_split_gain", 0.0, 1.0),
    FloatParameter("estimator__subsample", 0.7, 1.0),
    FloatParameter("estimator__colsample_bytree", 0.7, 1.0),
]


def _table() -> pd.DataFrame:
    """The simulator at 20 neighborhoods (79 buildings), room shares added."""
    path = Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"
    config = load_simulation_config(path)
    config = config.model_copy(
        update={
            "simulation": config.simulation.model_copy(update={"n_neighborhoods": 20})
        }
    )
    raw = StudentPopulationSimulator(config).run()
    shares = ShareTransformer(
        ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
    )
    return shares.fit_transform(raw)


@dataclass(frozen=True)
class _Split:
    train: pd.DataFrame
    test: pd.DataFrame
    groups_train: pd.Series
    exposure_train: pd.Series
    exposure_test: pd.Series
    evaluator: CVHyperparameterEvaluator


@pytest.fixture(scope="module")
def split() -> _Split:
    df = _table()
    groups = df["neighborhood_id"]
    # On the full table, before the split; then taken by the split's positions.
    exposure = ExposureTransformer("n_apartments").fit_transform(df)
    splitter = Splitter("stratified_by_group")
    train_index, test_index = splitter.train_test_indices(
        df, groups, test_size=0.2, random_state=42
    )
    evaluator = CVHyperparameterEvaluator(
        CountModel(
            # subsample_freq=1: LightGBM ignores subsample without it.
            estimator=LGBMRegressor(
                objective="poisson",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=0,
            ),
            use_exposure=True,
            feature_transformer=tree,
        ),
        PARAMETERS,
        cv=splitter.cv(n_splits=5, random_state=42),
        metric=POISSON_DEVIANCE,
    )
    return _Split(
        train=take_rows(df, train_index),
        test=take_rows(df, test_index),
        groups_train=take_rows(groups, train_index),
        exposure_train=take_rows(exposure, train_index),
        exposure_test=take_rows(exposure, test_index),
        evaluator=evaluator,
    )


@pytest.mark.slow
@pytest.mark.parametrize(("seed", "cohort"), list(enumerate(COHORTS)))
def test_the_refit_model_carries_each_cohorts_best_parameters(
    split: _Split, seed: int, cohort: str
) -> None:
    y_train = split.train[cohort]
    result = HyperparameterStudy(seed=seed, n_trials=3).optimize(
        lambda trial: split.evaluator.evaluate(
            trial,
            split.train,
            y_train,
            split.groups_train,
            exposure=split.exposure_train,
        )
    )
    assert result.metric_name == "poisson_deviance"
    assert len(result.trials) == 3
    values = [r.value for r in result.trials if r.state == TrialState.COMPLETE]
    assert result.best_value == min(v for v in values if v is not None)
    assert result.best_params == result.trials[result.best_trial_number].params

    model = split.evaluator.build_model(result.best_params)
    # The refit model is the best trial's, not the template's defaults.
    params = model.get_params()
    assert {name: params[name] for name in result.best_params} == result.best_params
    prediction = np.asarray(
        model.fit(split.train, y_train, exposure=split.exposure_train).predict(
            split.test, exposure=split.exposure_test
        )
    )
    assert prediction.shape == (len(split.test),)
    assert all(math.isfinite(p) and p >= 0 for p in prediction)
