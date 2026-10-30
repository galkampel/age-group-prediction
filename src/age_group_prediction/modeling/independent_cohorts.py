"""Model 1: one independent model per cohort, fitted and used on the raw table."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any, Self

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
    ``set_params`` names reach them: ``cohort_models__<cohort>__<param>`` sets
    one model's parameter (``cohort_models__n_kindergarten__estimator__alpha``),
    ``cohort_models__<cohort>`` replaces one model and ``cohort_models`` the
    mapping, so one study tunes every cohort on the composite's score. A cohort
    name must not contain ``__``. Set before ``fit``, which fits copies.

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

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """The parameters; with ``deep``, each cohort's model and its own parameters too."""
        params: dict[str, Any] = super().get_params(deep=deep)
        # sklearn descends only into a value that has get_params; a dict has none.
        if deep:
            for cohort, model in self.cohort_models.items():
                params[f"cohort_models__{cohort}"] = model
                params.update(
                    (f"cohort_models__{cohort}__{name}", value)
                    for name, value in model.get_params(deep=True).items()
                )
        return params

    def set_params(self, **params: Any) -> Self:
        """Set parameters, nested ``cohort_models__…`` names included; see the class docstring."""
        # The mapping first, then a cohort's model, then its nested names
        # (sklearn's order for its own nested names): a name set on an old
        # model would otherwise be overwritten by the new mapping silently.
        if "cohort_models" in params:
            self.cohort_models = params.pop("cohort_models")
        replaced: dict[str, BaseAgeGroupModel] = {}
        nested: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        for key in [key for key in params if key.startswith("cohort_models__")]:
            cohort, delim, name = key.removeprefix("cohort_models__").partition("__")
            if delim:
                nested[cohort][name] = params.pop(key)
            else:
                replaced[cohort] = params.pop(key)
        if replaced:
            # A new dict: the caller's mapping is left as given.
            self.cohort_models = {**self.cohort_models, **replaced}
        for cohort, sub_params in nested.items():
            # A misspelt cohort raises KeyError; a misspelt name, sklearn's
            # "Invalid parameter" naming that cohort's model.
            self.cohort_models[cohort].set_params(**sub_params)
        super().set_params(**params)
        return self

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
