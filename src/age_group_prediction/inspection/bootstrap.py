"""The bootstrap interval of a fitted model's metric on an evaluation set.

The model predicts once; only the evaluation rows are resampled, so the
interval is conditional on the fitted model (no refit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from sklearn.utils.validation import check_consistent_length

from .._config import STRICT
from ..modeling import BaseAgeGroupModel
from ..scoring import Metric
from ..utils import DesignMatrix, Exposure, Groups, Target, take_rows

__all__ = ["BootstrapEvaluator", "BootstrapResult"]


@dataclass(frozen=True)
class BootstrapResult:
    """The mean of the replicate scores and their percentile interval."""

    mean: float
    lower: float
    upper: float


# Settings are checked when the evaluator is built, not after the replicates:
# strictly (a bool is not an int), and a misspelt keyword is rejected.
@pydantic_dataclass(frozen=True, config=STRICT)
class BootstrapEvaluator:
    """Bootstrap a metric of a fitted model: its mean and ``1 - alpha`` percentile interval.

    Each of ``n_resamples`` replicates draws the evaluation rows with
    replacement, or whole groups when ``groups`` is given, and scores the
    fixed predictions on them. Scores are reported raw, in the metric's own
    direction.
    """

    metric: Metric
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
    ) -> BootstrapResult:
        """Predict ``X`` once with the fitted ``model``, then bootstrap the metric.

        ``X``, ``y``, ``groups`` and ``exposure`` are paired by position.
        ``groups`` holds one label per row (e.g. the neighborhood), and each
        replicate draws as many groups as there are, keeping all their rows.
        """
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
        scores = np.empty(self.n_resamples)
        for replicate in range(self.n_resamples):
            if group_rows is None:
                index = rng.integers(0, len(X), size=len(X))
            else:
                drawn = rng.integers(0, len(group_rows), size=len(group_rows))
                index = np.concatenate([group_rows[group] for group in drawn])
            scores[replicate] = model.evaluate(
                take_rows(y, index), take_rows(y_pred, index), self.metric
            )
        # np.quantile would return nan bounds silently.
        non_finite = int(np.sum(~np.isfinite(scores)))
        if non_finite:
            raise ValueError(
                f"{non_finite} of {self.n_resamples} replicate scores are not "
                "finite; check the data and that the metric suits the model"
            )
        lower, upper = np.quantile(scores, [self.alpha / 2, 1 - self.alpha / 2])
        return BootstrapResult(float(scores.mean()), float(lower), float(upper))
