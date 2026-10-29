"""The refit of a run's best model on the training part, scored on training and validation."""

from __future__ import annotations

from typing import Any

from ..modeling import BaseAgeGroupModel
from ..scoring import Metric
from ..utils import AgeGroupDataset

__all__ = ["fit_and_score"]


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
