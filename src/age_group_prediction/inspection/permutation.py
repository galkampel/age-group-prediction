"""Permutation importance of a fitted model, per raw column of ``X``.

A column's importance is the drop of the score when its values are shuffled,
averaged over repeats. The model predicts through its own feature transformer,
so one-hot or interaction columns are credited to their source column.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass
from sklearn.inspection import permutation_importance

from .._config import STRICT
from ..modeling import BaseAgeGroupModel
from ..scoring import Metric
from ..utils import Exposure, Target

__all__ = ["PermutationImportance"]


# Settings are checked when the object is built, not after the repeats.
@pydantic_dataclass(frozen=True, config=STRICT)
class PermutationImportance:
    """Permutation importance of a fitted model, one row per column of ``X``.

    Each column is shuffled ``n_repeats`` times and the model scored on
    ``metric`` each time. With a score where greater is better,

        importance = score(original X) - score(X with the column shuffled)

    averaged over the repeats; a loss (``direction="minimize"``) enters
    negated, so the importance is in the metric's units and positive means
    the column helps, whichever way the metric points. Held-out rows answer
    "what matters for the prediction"; the training rows show what the model
    fitted on.
    """

    metric: Metric
    # No lower bound here: scikit-learn raises on n_repeats < 1 before any
    # work. Strict pydantic rejects the bool scikit-learn would read as 1.
    n_repeats: int
    random_state: int | None = None

    def compute_importances(
        self,
        model: BaseAgeGroupModel,
        X: pd.DataFrame,
        y: Target,
        *,
        exposure: Exposure | None = None,
    ) -> pd.DataFrame:
        """Shuffle each column of ``X`` in turn and score the fitted ``model`` on ``y``.

        ``X`` is the raw table (the model transforms it inside ``predict``);
        ``X``, ``y`` and ``exposure`` are paired by position. Returns
        ``importance_mean`` and ``importance_std`` over the repeats, indexed
        by ``X``'s columns.
        """
        # scikit-learn scores "greater is better", so a loss is negated: the
        # drop of the score is then positive for a column the model uses.
        sign = 1.0 if self.metric.direction == "maximize" else -1.0

        def score(
            estimator: BaseAgeGroupModel, X_permuted: pd.DataFrame, y_true: Any
        ) -> float:
            # X_permuted is X for the baseline, then a copy with one column
            # shuffled; the rows stay in place, so the exposure taken by
            # position still matches.
            y_pred = estimator.predict(X_permuted, exposure=exposure)
            return sign * self.metric(y_true, y_pred)

        result = permutation_importance(
            model,
            X,
            y,
            scoring=score,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        # scikit-learn averages a NaN score into NaN importances silently.
        if not np.isfinite(result.importances).all():
            raise ValueError(
                "some permutation scores are not finite; check the data and "
                "that the metric suits the model"
            )
        return pd.DataFrame(
            {
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            },
            index=X.columns,
        )
