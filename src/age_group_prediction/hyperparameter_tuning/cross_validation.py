"""The cross-validated score of a model, the study that tunes it, and the refit of the best.

Each trial fits its folds in ``n_jobs`` parallel processes, and the trials
run one after another (D13, D19). ``tune_model`` takes the objective's model as a
callable, so it holds no model or search space of its own.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from sklearn.model_selection import BaseCrossValidator

from ..modeling import BaseAgeGroupModel
from ..scoring import Metric
from ..utils import AgeGroupDataset, Groups
from .aggregation import Aggregator, Mean

__all__ = ["cv_score", "fit_and_score", "fold_score", "tune_model"]


def fold_score(
    model: BaseAgeGroupModel,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    exposure: pd.Series,
    train_index: np.ndarray,
    val_index: np.ndarray,
    metric: Metric,
) -> float:
    """The score on one fold's validation rows of ``model`` fitted on its training rows."""
    X_train, Y_train, exposure_train = (
        X.iloc[train_index],
        Y.iloc[train_index],
        exposure.iloc[train_index],
    )
    X_val, Y_val, exposure_val = (
        X.iloc[val_index],
        Y.iloc[val_index],
        exposure.iloc[val_index],
    )
    model.fit(X_train, Y_train, exposure=exposure_train)
    Y_val_pred = model.predict(X_val, exposure=exposure_val)
    return model.evaluate(Y_val, Y_val_pred, metric)


def cv_score(
    model: BaseAgeGroupModel,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    exposure: pd.Series,
    cv: BaseCrossValidator,
    cv_groups: Groups | None,
    metric: Metric,
    aggregator: Aggregator | None = None,
    n_jobs: int = 1,
) -> float:
    """The aggregated fold score of ``model`` refit on each fold's training rows (``fit`` replaces all state).

    ``aggregator`` combines the fold scores; None means the plain mean.
    ``n_jobs`` processes fit the folds in parallel; the scores are the
    same as one after another, so a seeded study is unchanged. The
    processes do not inherit a script's ``-W`` filters: a warning in a fold
    raises there only under the environment's ``PYTHONWARNINGS`` (D15).
    """
    # The aggregator decides the selection (D9); the plain mean by default,
    # since the folds differ by one row at most.
    aggregator = Mean() if aggregator is None else aggregator
    folds = list(cv.split(X, Y, cv_groups))
    scores = Parallel(n_jobs=n_jobs)(
        delayed(fold_score)(model, X, Y, exposure, train_index, val_index, metric)
        for train_index, val_index in folds
    )
    fold_sizes = [len(val_index) for _, val_index in folds]
    return aggregator.aggregate_scores(scores, fold_sizes)


def tune_model(
    suggest_model: Callable[[optuna.Trial], BaseAgeGroupModel],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    exposure: pd.Series,
    cv: BaseCrossValidator,
    cv_groups: Groups | None,
    metric: Metric,
    *,
    study_seed: int,
    n_trials: int,
    n_startup_trials: int,
    aggregator: Aggregator | None = None,
    n_jobs: int = 1,
) -> tuple[optuna.Study, float]:
    """One study of the model ``suggest_model(trial)`` builds, on the rows of ``X``; the study and its seconds.

    A trial whose fit raises RuntimeError (an NB2 fit that does not
    converge, re-raised as such from a fold's process) is marked FAIL and
    the study goes on; without ``catch`` one such trial would end the
    study. The first ``n_startup_trials`` are drawn at random, before TPE's
    model guides the draws. ``aggregator`` and ``n_jobs`` go to ``cv_score``.
    """

    def objective(trial: optuna.Trial) -> float:
        # Optuna passes the trial alone; the data is bound here.
        return cv_score(
            suggest_model(trial),
            X,
            Y,
            exposure,
            cv,
            cv_groups,
            metric,
            aggregator,
            n_jobs,
        )

    study = optuna.create_study(
        direction=metric.direction,
        sampler=TPESampler(seed=study_seed, n_startup_trials=n_startup_trials),
    )
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
    return study, time.perf_counter() - start


def fit_and_score[M: BaseAgeGroupModel](
    model: M,
    train_age_group: AgeGroupDataset,
    val_age_group: AgeGroupDataset,
    *,
    cohort_metric: Metric,
    metric: Metric,
) -> tuple[M, list[dict[str, Any]]]:
    """Fit ``model`` on the training part; it and its score table on training and validation.

    A row per cohort column, scored by ``cohort_metric``, then the
    composite, scored by ``metric``.
    """
    model.fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
    y_pred_train = model.predict(train_age_group.X, exposure=train_age_group.exposure)
    y_pred_val = model.predict(val_age_group.X, exposure=val_age_group.exposure)
    score_table = [
        {
            "cohort": cohort,
            "train": model.evaluate(
                train_age_group.Y[cohort], y_pred_train[cohort], cohort_metric
            ),
            "val": model.evaluate(
                val_age_group.Y[cohort], y_pred_val[cohort], cohort_metric
            ),
        }
        for cohort in train_age_group.Y.columns
    ]
    score_table.append(
        {
            "cohort": "composite",
            "train": model.evaluate(train_age_group.Y, y_pred_train, metric),
            "val": model.evaluate(val_age_group.Y, y_pred_val, metric),
        }
    )
    return model, score_table
