"""PR #12 validation V4: claims in DIRECT_COHORT_MODEL.md §0.2/§0.3/§0.6, INDEPENDENT_TOTAL_PROBABILITY_MODEL.md
§0.2/§0.4 and MODULE_REFERENCE.md checked against the code."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from lightgbm.basic import LightGBMError
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from age_group_prediction import modeling
from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.modeling import (
    CohortProbabilityModel,
    DirectCohortModel,
    IndependentCohortModels,
    IndependentTotalProbabilityModel,
    TotalChildrenModel,
)

rng = np.random.default_rng(0)
N = 300
X = pd.DataFrame({"x": rng.normal(size=N)})
E = rng.integers(5, 40, N).astype(float)
Y = pd.Series(rng.poisson(0.2 * E * np.exp(0.3 * X["x"])))


def _lgbm() -> LGBMRegressor:
    return LGBMRegressor(objective="poisson", n_estimators=5, n_jobs=1, verbosity=-1)


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf], ids=["zero", "nan", "inf"])
def test_s03_lightgbm_fits_a_bad_exposure_silently_and_hgb_raises(bad: float) -> None:
    e = E.copy()
    e[0] = bad
    with np.errstate(all="ignore"):
        DirectCohortModel(estimator=_lgbm(), use_exposure=True).fit(
            X, Y, exposure=e
        )  # no error
        with pytest.raises(ValueError):
            DirectCohortModel(
                estimator=HistGradientBoostingRegressor(loss="poisson", max_iter=5),
                use_exposure=True,
            ).fit(X, Y, exposure=e)


def test_s03_sklearn_pipeline_raises_value_error_for_sample_weight() -> None:
    pipe = Pipeline([("s", StandardScaler()), ("m", _lgbm())])
    with pytest.raises(ValueError):
        DirectCohortModel(estimator=pipe, use_exposure=True).fit(X, Y, exposure=E)


def test_s03_unknown_objective_and_all_zero_y_raise_in_lightgbm() -> None:
    with pytest.raises(LightGBMError):
        DirectCohortModel(
            estimator=LGBMRegressor(objective="nope", n_jobs=1, verbosity=-1)
        ).fit(X, Y)
    with pytest.raises(LightGBMError, match="sum of labels is zero"):
        DirectCohortModel(estimator=_lgbm()).fit(X, Y * 0)


def test_s06_nested_name_into_the_mapping_raises_attribute_error() -> None:
    model = IndependentCohortModels({"a": DirectCohortModel(estimator=_lgbm())})
    with pytest.raises(AttributeError):
        model.set_params(cohort_models__a__estimator__learning_rate=0.05)
    DirectCohortModel(estimator=_lgbm()).set_params(estimator__learning_rate=0.05)


def test_s06_targets_may_stay_in_x_with_remainder_drop() -> None:
    tree = FeatureTransformer(
        (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
    )
    table = X.assign(a=Y.to_numpy())
    model = IndependentCohortModels(
        {"a": DirectCohortModel(estimator=_lgbm(), feature_transformer=tree)}
    )
    model.fit(table, table[["a"]])
    pd.testing.assert_frame_equal(model.predict(table), model.predict(table[["x"]]))


def test_itp_s02_fitted_state_and_nested_names() -> None:
    tree = FeatureTransformer(
        (ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),)
    )
    y = pd.DataFrame({"a": 1 + Y.to_numpy(), "b": rng.poisson(2, N)})
    model = IndependentTotalProbabilityModel(
        total_children_model=TotalChildrenModel(feature_transformer=tree),
        cohort_probability_model=CohortProbabilityModel(feature_transformer=tree),
    )
    tuned = clone(model).set_params(cohort_probability_model__l2_penalty=1e-3)
    assert tuned.cohort_probability_model.l2_penalty == 1e-3
    fitted = model.fit(X, y, exposure=E)
    total = fitted.total_children_model_
    for name in (
        "intercept_",
        "coef_",
        "use_exposure_",
        "feature_transformer_",
        "feature_names_in_",
        "n_features_in_",
    ):
        assert hasattr(total, name), name
    assert not hasattr(total, "dispersion_")  # Poisson
    prob = fitted.cohort_probability_model_
    for name in (
        "intercept_",
        "coef_",
        "cohorts_",
        "feature_transformer_",
        "feature_names_in_",
    ):
        assert hasattr(prob, name), name
    assert prob.coef_.shape == (1, 2) and prob.intercept_.shape == (2,)
    assert fitted.cohorts_ == ["a", "b"]


def test_module_reference_init_row_matches_all() -> None:
    row = next(
        l
        for l in Path("docs/MODULE_REFERENCE.md").read_text().splitlines()
        if l.startswith("| `__init__.py` | Public surface of the rebuilt models")
    )
    listed = re.findall(r"`([A-Za-z_]+)`", row.split("|")[3])
    assert listed == sorted(modeling.__all__) == list(modeling.__all__)


def test_module_reference_has_no_pipeline_row_and_direct_cohort_row_is_current() -> (
    None
):
    text = Path("docs/MODULE_REFERENCE.md").read_text()
    assert "modeling/pipeline.py" not in text and "ModelPipeline" not in text
    row = next(l for l in text.splitlines() if l.startswith("| `direct_cohort.py`"))
    for word in (
        "estimator",
        "per-apartment rate",
        "sample_weight=exposure",
        "feature_transformer",
        "Regressor",
    ):
        assert word in row, word
