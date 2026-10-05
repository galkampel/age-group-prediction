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

# Each model takes the raw table, usually through its own feature_transformer.
type CohortModels = Mapping[str, BaseAgeGroupModel]


class IndependentCohortModels(BaseAgeGroupModel):
    """Fit one model per cohort on its own column of ``y``, and predict them all.

    Each cohort's model has its own features and hyperparameters. Nested
    ``set_params`` names do not reach into the mapping: tune each model on its
    own and replace the mapping with ``set_params(cohort_models=...)``, before
    ``fit``, which fits copies.

    The same ``exposure`` goes to every cohort; a model without an exposure
    offset ignores it. Rows are paired by position, as in
    :meth:`~age_group_prediction.modeling.BaseAgeGroupModel.fit`. ``predict``
    returns a DataFrame with one column per cohort, in ``y``'s column order at
    fit whatever the mapping's order (so a plain ``dict`` suffices), indexed
    like ``X``.
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
