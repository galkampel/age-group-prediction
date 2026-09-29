"""Model 1: one independent model per cohort, fitted and used on the raw table."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from .base import BaseAgeGroupModel

__all__ = ["CohortModels", "IndependentCohortModels"]

# Each model takes the raw table, so each cohort keeps its own features: a
# ModelPipeline.
type CohortModels = Mapping[str, BaseAgeGroupModel]


class IndependentCohortModels(BaseAgeGroupModel):
    """Fit one model per cohort on its own column of ``y``, and predict them all.

    The cohorts are independent: each has its own feature transformer, model
    and hyperparameters, so each is tuned on its own and the tuned models are
    assembled here. Nested ``set_params`` names do not reach into the mapping;
    replace it with ``set_params(cohort_models=...)``, before ``fit``, which
    fits copies.

    The same ``exposure`` goes to every cohort; a model without an exposure
    offset ignores it. Rows are paired by position, as in
    :class:`~age_group_prediction.modeling.ModelPipeline`. ``predict`` returns a DataFrame with one column per
    cohort, in ``y``'s column order at fit, indexed like ``X``.
    """

    def __init__(self, cohort_models: CohortModels) -> None:
        # Stored verbatim: set_params assigns attributes without re-entering here.
        self.cohort_models = cohort_models

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        exposure: ArrayLike | None = None,
    ) -> Self:
        """Fit a copy of each cohort's model on the raw table ``X`` and ``y[cohort]``."""
        # Counted, not as sets: a duplicated column of y would pass a set check
        # and reach its model as a DataFrame.
        if Counter(y.columns) != Counter(list(self.cohort_models)):
            raise ValueError(
                f"cohort_models has {list(self.cohort_models)}, but y's columns "
                f"are {list(y.columns)}; give one model per column of y"
            )
        cohort_models = {
            cohort: clone(self.cohort_models[cohort]).fit(
                X, y[cohort], exposure=exposure
            )
            for cohort in y.columns
        }
        # Set only once every cohort succeeded.
        self.cohort_models_: dict[str, BaseAgeGroupModel] = cohort_models
        return self

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> pd.DataFrame:
        """The predicted mean of every cohort for each row of ``X``."""
        check_is_fitted(self)
        # As arrays: a prediction with its own index would be realigned to X's,
        # into NaN, silently.
        return pd.DataFrame(
            {
                cohort: np.asarray(model.predict(X, exposure=exposure))
                for cohort, model in self.cohort_models_.items()
            },
            index=X.index,
        )
