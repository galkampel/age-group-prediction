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
    rejects the values LightGBM would accept silently. As in scikit-learn,
    the rows of ``X``, ``y`` and the exposure are paired by position, not by
    index: take them all by the same positions, e.g. those of
    :meth:`~age_group_prediction.splitting.Splitter.train_test_indices`.
    ``fit`` fits copies, so the templates stay unfitted; ``predict`` uses the
    fitted copies ``feature_transformer_`` and ``model_``. A setting changed by
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

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        exposure: ArrayLike | None = None,
    ) -> Self:
        """Fit copies of the transformer and the model on the raw table ``X``."""
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
        return self.model_.predict(
            self.feature_transformer_.transform(X), exposure=exposure
        )

    def predict_logits(self, X: pd.DataFrame) -> pd.DataFrame:
        """The model's logits for the raw table ``X``, what a calibrator takes.

        Only for a model that has them, such as
        :class:`~age_group_prediction.modeling.CohortProbabilityModel`.
        """
        check_is_fitted(self)
        return self.model_.predict_logits(  # type: ignore[attr-defined]
            self.feature_transformer_.transform(X)
        )
