"""The bootstrap intervals of a fitted model's metrics on an evaluation set.

The model predicts once; only the evaluation rows are resampled, so the
intervals are conditional on the fitted model (no refit). Every metric named
in a call scores the same replicates, so their intervals are paired.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from sklearn.utils.validation import check_consistent_length

from .._config import STRICT
from ..modeling import BaseAgeGroupModel
from ..scoring import Metric
from ..utils import DesignMatrix, Exposure, Groups, Target, take_rows

__all__ = ["BootstrapEvaluator"]


# Settings are checked when the evaluator is built, not after the replicates:
# strictly (a bool is not an int), and a misspelt keyword is rejected.
@pydantic_dataclass(frozen=True, config=STRICT)
class BootstrapEvaluator:
    """The resampling design: ``n_resamples`` replicates and a ``1 - alpha`` percentile interval.

    The metrics are named per call, as ``cross_validate``'s ``scoring``.
    Each replicate draws the evaluation rows with replacement, or whole
    groups when ``groups`` is given, and scores the fixed predictions on
    them with every metric. Scores are reported raw, in each metric's own
    direction.
    """

    n_resamples: Annotated[int, Field(ge=1)]
    alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    random_state: int | None = None

    def evaluate(
        self,
        model: BaseAgeGroupModel,
        X: DesignMatrix,
        y: Target,
        groups: Groups | None = None,
        *,
        exposure: Exposure | None = None,
        metrics: Sequence[Metric],
    ) -> pd.DataFrame:
        """Predict ``X`` once with the fitted ``model``, then bootstrap every metric.

        ``X``, ``y``, ``groups`` and ``exposure`` are paired by position.
        ``groups`` holds one label per row (e.g. the neighborhood), and each
        replicate draws as many groups as there are, keeping all their rows.
        Returns one row per metric, indexed by its name: ``score`` on the
        unresampled rows, and the replicates' ``mean``, ``lower`` and
        ``upper``.
        """
        names = [metric.name for metric in metrics]
        # An empty frame, or two rows under one label, would pass silently.
        if not names:
            raise ValueError("metrics must hold at least one Metric")
        if len(set(names)) < len(names):
            raise ValueError(f"metric names must be distinct, got {names}")
        # A longer y or groups would be silently mispaired with the predictions,
        # or leave rows out of every replicate.
        check_consistent_length(X, y, groups)
        y_pred = model.predict(X, exposure=exposure)
        rng = np.random.default_rng(self.random_state)
        if groups is None:
            group_rows = None
        else:
            # The row positions of each group, one array per group.
            _, group_codes, group_sizes = np.unique(
                np.asarray(groups), return_inverse=True, return_counts=True
            )
            # With one group every replicate is the whole set: a zero-width
            # interval that looks certain.
            if len(group_sizes) < 2:
                raise ValueError(
                    f"groups has {len(group_sizes)} distinct value(s); the "
                    "group bootstrap needs at least 2"
                )
            group_rows = np.split(
                np.argsort(group_codes, kind="stable"), np.cumsum(group_sizes)[:-1]
            )
        scores = np.empty((self.n_resamples, len(metrics)))
        for replicate in range(self.n_resamples):
            if group_rows is None:
                index = rng.integers(0, len(X), size=len(X))
            else:
                drawn = rng.integers(0, len(group_rows), size=len(group_rows))
                index = np.concatenate([group_rows[group] for group in drawn])
            y_drawn, y_pred_drawn = take_rows(y, index), take_rows(y_pred, index)
            for column, metric in enumerate(metrics):
                scores[replicate, column] = model.evaluate(
                    y_drawn, y_pred_drawn, metric
                )
        # np.quantile would return nan bounds silently.
        non_finite = np.sum(~np.isfinite(scores), axis=0)
        if non_finite.any():
            failed = ", ".join(
                f"{name}: {count} of {self.n_resamples}"
                for name, count in zip(names, non_finite)
                if count
            )
            raise ValueError(
                f"replicate scores are not finite ({failed}); check the data and "
                "that the metric suits the model"
            )
        lower, upper = np.quantile(scores, [self.alpha / 2, 1 - self.alpha / 2], axis=0)
        return pd.DataFrame(
            {
                "score": [model.evaluate(y, y_pred, metric) for metric in metrics],
                "mean": scores.mean(axis=0),
                "lower": lower,
                "upper": upper,
            },
            index=pd.Index(names, name="metric"),
        )
