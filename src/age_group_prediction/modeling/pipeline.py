"""A feature transformer, then a model, fitted and used on the raw table."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ..feature_engineering import FeatureTransformer
from .base import BaseAgeGroupModel

__all__ = ["ModelPipeline"]


class ModelPipeline(BaseAgeGroupModel):
    """Transform the raw table with ``feature_transformer``, then apply ``model``.

    ``exposure`` is passed through to the model. Build it on the full table,
    before splitting, with
    :class:`~age_group_prediction.preprocessing.ExposureTransformer`, which
    rejects the values LightGBM would accept silently. ``fit`` fits copies, so
    the templates stay unfitted; ``predict`` uses the fitted copies
    ``feature_transformer_`` and ``model_``. A setting changed by
    ``set_params`` after ``fit`` therefore reaches only the next ``fit``.

    Not scikit-learn's ``Pipeline``: that one names the exposure differently in
    ``fit`` and ``predict``, and has no ``evaluate``.
    """

    def __init__(
        self, feature_transformer: FeatureTransformer, model: BaseAgeGroupModel
    ) -> None:
        # Stored verbatim: set_params assigns attributes without re-entering here.
        self.feature_transformer = feature_transformer
        self.model = model

    @property
    def uses_exposure(self) -> bool:
        """The model's setting, so a caller knows whether to pass an exposure.

        It is the template's, so it describes the next ``fit``: after
        ``set_params`` a fitted pipeline may differ, and its ``predict`` then
        raises on a missing or unexpected exposure rather than misusing one.
        """
        return self.model.uses_exposure

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        exposure: ArrayLike | None = None,
    ) -> Self:
        """Fit copies of the transformer and the model on the raw table ``X``."""
        self._check_aligned(X, y, "y")
        self._check_aligned(X, exposure, "exposure")
        # y as sklearn's Pipeline passes it, so a supervised transformer works too.
        feature_transformer = clone(self.feature_transformer).fit(X, y)
        model = clone(self.model).fit(
            feature_transformer.transform(X), y, exposure=exposure
        )
        # Set together, only once both succeeded.
        self.feature_transformer_: FeatureTransformer = feature_transformer
        self.model_: BaseAgeGroupModel = model
        return self

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> np.ndarray | pd.DataFrame:
        """Predict from the raw table ``X`` with the fitted copies.

        ``X`` is transformed with the statistics learned at fit, never
        refitted, so a row's prediction does not depend on the rows beside it.
        """
        check_is_fitted(self)
        self._check_aligned(X, exposure, "exposure")
        return self.model_.predict(
            self.feature_transformer_.transform(X), exposure=exposure
        )

    @staticmethod
    def _check_aligned(X: pd.DataFrame, values: object, name: str) -> None:
        # The model reads y and the exposure by position, so values labelled for
        # other rows would land on the wrong buildings silently. An array has
        # no labels to check; whether an exposure is needed is the model's check.
        if isinstance(values, pd.Series | pd.DataFrame) and not values.index.equals(
            X.index
        ):
            raise ValueError(
                f"{name}'s index differs from X's; take both from the same rows "
                "of the same table"
            )
