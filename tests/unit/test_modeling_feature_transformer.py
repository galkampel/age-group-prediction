"""The feature transformer inside each leaf model: each test names the mistake it catches."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.modeling import (
    BaseAgeGroupModel,
    CohortProbabilityModel,
    CountModel,
    TotalChildrenModel,
)

# Centered x only. The table also holds columns the transformer drops, and x
# has a mean far from 0: on x alone a skipped transformer would go unseen,
# since trees ignore a shift and an intercept absorbs it.
FEATURES = FeatureTransformer(
    (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
)


def _table(n_rows: int = 200) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """A raw table on a non-default index, its exposure and its cohort counts."""
    rng = np.random.default_rng(0)
    index = pd.RangeIndex(100, 100 + n_rows)
    table = pd.DataFrame(
        {
            "x": rng.normal(loc=2.0, size=n_rows),
            "z": rng.normal(size=n_rows),
            "n_apartments": rng.integers(5, 40, n_rows).astype(float),
        },
        index=index,
    )
    rate = 0.1 * table["n_apartments"]
    counts = pd.DataFrame(
        {
            "a": rng.poisson(rate * np.exp(0.5 * (table["x"] - 2))),
            "b": rng.poisson(rate * np.exp(-0.3 * (table["x"] - 2))),
        },
        index=index,
    )
    return table, table["n_apartments"], counts


type Build = Callable[[FeatureTransformer | None], BaseAgeGroupModel]
type Target = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]

# Each leaf model with the target it takes, from the cohort counts.
MODELS: dict[str, tuple[Build, Target]] = {
    "CountModel": (
        lambda features: CountModel(
            # One thread: more OpenMP threads crash alongside torch on macOS.
            estimator=LGBMRegressor(
                objective="poisson", n_estimators=20, n_jobs=1, verbosity=-1
            ),
            use_exposure=True,
            feature_transformer=features,
        ),
        lambda counts: counts["a"],
    ),
    "TotalChildrenModel": (
        lambda features: TotalChildrenModel(feature_transformer=features),
        lambda counts: counts.sum(axis=1),
    ),
    "CohortProbabilityModel": (
        lambda features: CohortProbabilityModel(
            estimator=LogisticRegression(), feature_transformer=features
        ),
        lambda counts: counts,
    ),
}


@pytest.mark.parametrize(("build", "target"), MODELS.values(), ids=list(MODELS))
def test_rows_are_transformed_with_the_training_statistics(
    build: Build, target: Target
) -> None:
    # The model on the raw table must equal the model on a design matrix built
    # by hand. Catches the transformer skipped at fit, at predict or both (the
    # dropped columns would then reach the model), refitted at predict (new
    # rows centred on their own mean), or applied twice.
    table, exposure, counts = _table()
    y = target(counts)
    train, test = slice(0, 100), slice(100, None)
    model = build(FEATURES).fit(
        table.iloc[train], y.iloc[train], exposure=exposure.iloc[train]
    )
    features = clone(FEATURES).fit(table.iloc[train])
    by_hand = build(None).fit(
        features.transform(table.iloc[train]),
        y.iloc[train],
        exposure=exposure.iloc[train],
    )

    np.testing.assert_allclose(
        np.asarray(model.predict(table.iloc[test], exposure=exposure.iloc[test])),
        np.asarray(
            by_hand.predict(
                features.transform(table.iloc[test]), exposure=exposure.iloc[test]
            )
        ),
        rtol=1e-12,
    )


@pytest.mark.parametrize(("build", "target"), MODELS.values(), ids=list(MODELS))
def test_the_template_transformer_stays_unfitted(build: Build, target: Target) -> None:
    # Fitting the caller's transformer in place would leak one fold's
    # statistics into the next fold and into the caller's later use.
    table, exposure, counts = _table()
    template = clone(FEATURES)

    model = build(template).fit(table, target(counts), exposure=exposure)

    assert model.feature_transformer_ is not template
    with pytest.raises(NotFittedError):
        check_is_fitted(template)


@pytest.mark.parametrize(("build", "target"), MODELS.values(), ids=list(MODELS))
def test_predict_follows_the_transformer_fitted_at_fit(
    build: Build, target: Target
) -> None:
    # Predict must follow the fitted copy, not the current setting: checking
    # the template for None would, after set_params(feature_transformer=None),
    # predict on the raw table, silently when the transformer keeps the columns.
    table, exposure, counts = _table()
    model = build(FEATURES).fit(table[["x"]], target(counts), exposure=exposure)
    expected = np.asarray(model.predict(table[["x"]], exposure=exposure))

    model.set_params(feature_transformer=None)

    np.testing.assert_array_equal(
        np.asarray(model.predict(table[["x"]], exposure=exposure)), expected
    )
