"""The bootstrap intervals of a fitted model's metrics on an evaluation set.

The caller predicts once and passes the predictions; only the evaluation rows
are resampled, so the intervals are conditional on the fitted model (no
refit). Every metric named in a call scores the same replicates, so their
intervals are paired.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from sklearn.utils.validation import check_consistent_length

from .. import scoring
from .._config import STRICT
from ..scoring import Metric
from ..utils import Groups, Target, take_rows

__all__ = ["BootstrapEvaluator"]


# Settings are checked when the evaluator is built, not after the replicates:
# strictly (a bool is not an int), and a misspelt keyword is rejected.
@pydantic_dataclass(frozen=True, config=STRICT)
class BootstrapEvaluator:
    """The resampling design: ``n_resamples`` replicates and a ``1 - alpha`` percentile interval.

    The metrics and cohorts are named per call, as ``cross_validate``'s
    ``scoring``. Each replicate draws the evaluation rows with replacement, or
    whole groups when ``groups`` is given, and scores the fixed predictions on
    them with every metric for every cohort (``scoring.evaluate``). Scores are
    reported raw, in each metric's own direction.
    """

    n_resamples: Annotated[int, Field(ge=1)]
    alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    random_state: int | None = None

    def evaluate(
        self,
        y_true: Target,
        y_pred: Target,
        groups: Groups | None = None,
        *,
        metrics: Sequence[Metric],
        cohorts: str | Sequence[str] = "composite",
    ) -> pd.DataFrame:
        """Bootstrap every metric of the fixed predictions ``y_pred`` against ``y_true``, per cohort.

        ``y_true``, ``y_pred`` and ``groups`` are paired by position.
        ``groups`` holds one label per row (e.g. the neighborhood), and each
        replicate draws as many groups as there are, keeping all their rows.
        ``cohorts`` as in ``scoring.evaluate``: column names of ``y_true``,
        each scored alone, or ``"composite"``. Returns one row per (cohort,
        metric): the replicates' ``mean``, ``lower`` and ``upper``; the score
        on the rows themselves is ``scoring.evaluate``'s.
        """
        names = [metric.name for metric in metrics]
        # An empty frame, or two rows under one label, would pass silently.
        if not names:
            raise ValueError("metrics must hold at least one Metric")
        if len(set(names)) < len(names):
            raise ValueError(f"metric names must be distinct, got {names}")
        # A longer y_true, y_pred or groups would be silently mispaired, or
        # leave rows out of every replicate.
        check_consistent_length(y_true, y_pred, groups)
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
        replicate_scores = []
        for replicate in range(self.n_resamples):
            if group_rows is None:
                index = rng.integers(0, len(y_true), size=len(y_true))
            else:
                drawn = rng.integers(0, len(group_rows), size=len(group_rows))
                index = np.concatenate([group_rows[group] for group in drawn])
            y_drawn, y_pred_drawn = take_rows(y_true, index), take_rows(y_pred, index)
            # Every cohort and metric on the same rows, so their intervals are
            # paired.
            replicate_scores.append(
                scoring.evaluate(y_drawn, y_pred_drawn, metrics, cohorts).stack()
            )
        # One column per (cohort, metric), in the stacked table's order.
        labels = replicate_scores[0].index
        scores = np.array([replicate.to_numpy() for replicate in replicate_scores])
        # np.quantile would return nan bounds silently.
        non_finite = np.sum(~np.isfinite(scores), axis=0)
        if non_finite.any():
            failed = ", ".join(
                f"{cohort} {metric}: {count} of {self.n_resamples}"
                for (cohort, metric), count in zip(labels, non_finite)
                if count
            )
            raise ValueError(
                f"replicate scores are not finite ({failed}); check the data and "
                "that the metric suits the model"
            )
        lower, upper = np.quantile(scores, [self.alpha / 2, 1 - self.alpha / 2], axis=0)
        return pd.DataFrame(
            {
                "mean": scores.mean(axis=0),
                "lower": lower,
                "upper": upper,
            },
            index=labels,
        )
