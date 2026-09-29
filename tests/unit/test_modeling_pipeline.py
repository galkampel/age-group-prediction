"""ModelPipeline: each test names the mistake in our code it would catch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.modeling import DirectCohortModel, ModelPipeline

# Centered, so the transformer learns a statistic that a refit would change.
# The exposure is not a feature: it is then the only way size enters.
FEATURES = FeatureTransformer(
    (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
)


def _table(n_rows: int = 200) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """A raw table on a non-default index, its counts and its exposure."""
    rng = np.random.default_rng(0)
    index = pd.RangeIndex(100, 100 + n_rows)
    X = pd.DataFrame({"x": rng.normal(size=n_rows)}, index=index)
    exposure = pd.Series(rng.integers(5, 40, n_rows).astype(float), index=index)
    y = pd.Series(rng.poisson(0.2 * exposure * np.exp(0.5 * X["x"])), index=index)
    return X, y, exposure


def _pipeline() -> ModelPipeline:
    return ModelPipeline(FEATURES, DirectCohortModel(use_exposure=True))


def test_the_templates_stay_unfitted() -> None:
    # Fitting the caller's objects in place would leak one fold's fit into the
    # next and into the caller's later use.
    X, y, exposure = _table()
    pipeline = _pipeline()

    pipeline.fit(X, y, exposure=exposure)

    for template in (pipeline.feature_transformer, pipeline.model):
        with pytest.raises(NotFittedError):
            check_is_fitted(template)


def test_doubling_the_exposure_doubles_the_prediction() -> None:
    # Catches the exposure being lost or replaced at predict: with the offset,
    # and size in no feature, the mean is exactly proportional to it.
    X, y, exposure = _table()
    pipeline = _pipeline().fit(X, y, exposure=exposure)

    np.testing.assert_allclose(
        pipeline.predict(X, exposure=2 * exposure),
        2 * pipeline.predict(X, exposure=exposure),
        rtol=1e-12,
    )


def test_rows_are_transformed_with_the_training_statistics() -> None:
    # A transformer refitted at predict would center each batch on itself, so a
    # row's prediction would depend on the rows predicted with it.
    X, y, exposure = _table()
    pipeline = _pipeline().fit(X.iloc[:150], y.iloc[:150], exposure=exposure.iloc[:150])
    X_val, exposure_val = X.iloc[150:], exposure.iloc[150:]

    batch = pipeline.predict(X_val, exposure=exposure_val)
    row_by_row = [
        pipeline.predict(X_val.iloc[[i]], exposure=exposure_val.iloc[[i]])[0]
        for i in range(len(X_val))
    ]

    np.testing.assert_array_equal(batch, row_by_row)


def test_predict_follows_the_fitted_copies_after_set_params() -> None:
    # Using the template at predict would ignore the fitted model, once
    # set_params has changed the template's setting.
    X, y, exposure = _table()
    pipeline = _pipeline().fit(X, y, exposure=exposure)
    expected = pipeline.predict(X, exposure=exposure)

    pipeline.set_params(model__use_exposure=False)

    np.testing.assert_array_equal(pipeline.predict(X, exposure=exposure), expected)
