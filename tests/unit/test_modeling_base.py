"""The shared ``evaluate`` of every age-group model."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from age_group_prediction.modeling import BaseAgeGroupModel
from age_group_prediction.scoring import Metric


class _Model(BaseAgeGroupModel):
    """A concrete stand-in, since the base class is abstract; only evaluate runs."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        return self

    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> np.ndarray:
        raise NotImplementedError


def _max_error(y_true: ArrayLike, y_pred: ArrayLike) -> np.floating:
    return np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def test_evaluate_applies_a_custom_metric_to_y_true_and_y_pred() -> None:
    metric = Metric("max_error", _max_error)

    score = _Model().evaluate(
        np.array([1.0, 4.0, 2.0]), np.array([1.5, 2.0, 2.0]), metric
    )

    assert score == 2.0
    # The numpy scalar the metric returns is cast to a plain float.
    assert type(score) is float
