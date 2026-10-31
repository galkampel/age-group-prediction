"""The shared feature helpers of every age-group model."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.feature_engineering import (
    Center,
    FeatureTransformer,
    Plan,
)
from age_group_prediction.modeling import BaseAgeGroupModel


class _FeatureModel(BaseAgeGroupModel):
    """A stand-in that only transforms: ``predict`` returns the design matrix."""

    def __init__(
        self, *, feature_transformer: FeatureTransformer | None = None
    ) -> None:
        self.feature_transformer = feature_transformer

    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        feature_transformer, design_matrix = self._fit_features(X, y)
        self.feature_transformer_ = feature_transformer
        self.design_matrix_ = design_matrix
        return self

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> pd.DataFrame:
        return self._transform_features(X)


def _centering() -> FeatureTransformer:
    return FeatureTransformer(((Plan(name="x", transforms=(Center(),)), ["x"]),))


def _table() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"x": [1.0, 2.0, 6.0]}, index=[10, 11, 12])
    return X, pd.Series([0, 1, 2], index=X.index)


def test_without_a_transformer_the_helpers_return_x_itself() -> None:
    # A copy or conversion of X when there is nothing to transform would cost
    # memory on every fit and predict, and could change dtypes or the index.
    X, y = _table()
    model = _FeatureModel().fit(X, y)

    assert model.feature_transformer_ is None
    assert model.design_matrix_ is X
    assert model.predict(X) is X


def test_fit_features_fits_a_copy_and_leaves_the_template_unfitted() -> None:
    # Fitting the caller's transformer in place would leak one fold's
    # statistics into the next fold and into the caller's later use.
    X, y = _table()
    template = _centering()

    model = _FeatureModel(feature_transformer=template).fit(X, y)

    assert model.feature_transformer_ is not template
    check_is_fitted(model.feature_transformer_)
    with pytest.raises(NotFittedError):
        check_is_fitted(template)


def test_transform_features_uses_the_statistics_learned_at_fit() -> None:
    # A transformer refitted at predict would centre the new rows on their own
    # mean, so a row's prediction would depend on the rows predicted with it.
    # The last check catches fit handing the model the raw rows untransformed.
    X, y = _table()
    model = _FeatureModel(feature_transformer=_centering()).fit(X, y)
    new_rows = pd.DataFrame({"x": [0.0, 10.0]}, index=[20, 21])

    design = model.predict(new_rows)

    training_mean = X["x"].mean()
    np.testing.assert_allclose(design.iloc[:, 0], new_rows["x"] - training_mean)
    np.testing.assert_allclose(model.design_matrix_.iloc[:, 0], X["x"] - training_mean)
