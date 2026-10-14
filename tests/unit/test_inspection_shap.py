"""Tests for ``ShapExplainer``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from age_group_prediction.feature_engineering import (
    ColumnPlan,
    FeatureTransformer,
    OneHot,
    Standardize,
)
from age_group_prediction.inspection import ShapExplainer
from age_group_prediction.modeling import (
    BaseAgeGroupModel,
    CohortProbabilityModel,
    CountModel,
    TotalTimesProbabilityModel,
)
from age_group_prediction.preprocessing import ExposureTransformer
from student_simulator import StudentPopulationSimulator, load_simulation_config

# A shuffled index that is not 0..n-1, so code pairing rows by label fails.
# "c" appears only in X: the codes must come from X and the background.
_RNG = np.random.default_rng(0)
INDEX = _RNG.permutation(12) + 1000
X = pd.DataFrame(
    {
        "x": _RNG.normal(size=12),
        "count": _RNG.integers(0, 5, size=12),
        "flag": _RNG.random(12) > 0.5,
        "kind": np.repeat(["a", "b", "c"], 4),
    },
    index=INDEX,
)
BACKGROUND = pd.DataFrame(
    {
        "x": _RNG.normal(size=20),
        "count": _RNG.integers(0, 5, size=20),
        "flag": _RNG.random(20) > 0.5,
        "kind": np.tile(["a", "b"], 10),
    }
)
# Both methods at their cheapest: permutation at its minimum 2 * p + 1.
METHODS = [
    pytest.param({"method": "exact"}, id="exact"),
    pytest.param(
        {
            "method": "permutation",
            "max_evals": 2 * len(X.columns) + 1,
            "random_state": 0,
        },
        id="permutation",
    ),
]


class _SpyModel(BaseAgeGroupModel):
    """Predicts ``x * (1 + count)``, doubled for kind "b", times the exposure.

    Three columns interact, so one permutation's forward and backward sweep
    depends on its order (with two, any sweep is already exact) and a seed
    shows. Records what each predict call received.
    """

    def __init__(self) -> None:
        self.dtypes_seen: list[dict[str, str]] = []
        self.kinds_seen: set[str] = set()
        self.x_seen: set[float] = set()
        self.exposure_all_ones: list[bool] = []

    def fit(self, X: Any, y: Any, exposure: ArrayLike | None = None) -> Self:
        return self

    def predict(self, X: Any, exposure: ArrayLike | None = None) -> np.ndarray:
        self.dtypes_seen.append(dict(X.dtypes.astype(str)))
        self.kinds_seen.update(X["kind"])
        self.x_seen.update(X["x"].tolist())
        self.exposure_all_ones.append(bool(np.all(np.asarray(exposure) == 1.0)))
        rate = (
            np.asarray(X["x"], dtype=float)
            * (1.0 + np.asarray(X["count"], dtype=float))
            * np.where(X["kind"] == "b", 2.0, 1.0)
        )
        return rate * np.asarray(exposure, dtype=float)


class _CohortSpy(_SpyModel):
    """The spy's rate times 1, 2 and 3 as the cohort columns ``k``, ``e``, ``h``."""

    def predict(self, X: Any, exposure: ArrayLike | None = None) -> pd.DataFrame:
        rate = super().predict(X, exposure)
        return pd.DataFrame(
            {"k": rate, "e": 2.0 * rate, "h": 3.0 * rate}, index=X.index
        )


def _fitted_count_model() -> tuple[CountModel, pd.DataFrame, np.ndarray]:
    # A Poisson GLM with exposure on two float columns (no non-numeric
    # column, so the decode step has nothing to do).
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame({"x": rng.normal(size=60), "z": rng.normal(size=60)})
    exposure = rng.uniform(5.0, 50.0, size=60)
    y = rng.poisson(np.exp(0.5 * X_train["x"] - 0.3 * X_train["z"]) * exposure)
    features = FeatureTransformer(
        plans=(ColumnPlan(name="x", columns=("x",), transforms=(Standardize(),)),)
    )
    model = CountModel(
        estimator=PoissonRegressor(), use_exposure=True, feature_transformer=features
    ).fit(X_train, y, exposure=exposure)
    return model, X_train, exposure


@pytest.mark.parametrize("settings", METHODS)
def test_values_add_up_to_the_rate_without_exposure(settings: dict[str, Any]) -> None:
    # Catches explaining the count instead of the rate (the exposure not set
    # to one inside the prediction function) or a wrong base value: the
    # values must sum to predict(X, exposure=1), not to the count.
    model, X_train, exposure = _fitted_count_model()
    rows, rows_exposure = X_train.iloc[:8], exposure[:8]
    explanation = ShapExplainer(**settings).explain(model, rows, X_train.iloc[20:40])
    total = explanation.base_values + explanation.values.sum(axis=1)
    rate = model.predict(rows, exposure=np.ones(8))
    np.testing.assert_allclose(total, rate, atol=1e-10)
    assert not np.allclose(total, model.predict(rows, exposure=rows_exposure))


@pytest.mark.parametrize("settings", METHODS)
def test_values_add_up_to_the_count_with_exposure(settings: dict[str, Any]) -> None:
    # Catches the exposure not applied, applied to the values but not the
    # base values, or paired with the wrong rows: the values must sum to
    # predict(X, exposure) and differ from the rate.
    model, X_train, exposure = _fitted_count_model()
    rows, rows_exposure = X_train.iloc[:8], exposure[:8]
    explanation = ShapExplainer(**settings).explain(
        model, rows, X_train.iloc[20:40], exposure=rows_exposure
    )
    total = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(
        total, model.predict(rows, exposure=rows_exposure), atol=1e-10
    )
    assert not np.allclose(total, model.predict(rows, exposure=np.ones(8)))


@pytest.mark.parametrize("settings", METHODS)
def test_cohort_outputs_have_the_cohort_names_and_shape(
    settings: dict[str, Any],
) -> None:
    # Catches the output names left as shap's "Output 0..2" or set in the
    # wrong order: the cohort "e" must be the doubled rate.
    explanation = ShapExplainer(**settings).explain(_CohortSpy(), X, BACKGROUND)
    assert explanation.values.shape == (12, 4, 3)
    assert explanation.base_values.shape == (12, 3)
    assert explanation.output_names == ["k", "e", "h"]
    np.testing.assert_allclose(
        explanation[:, :, "e"].values, 2.0 * explanation[:, :, "k"].values
    )


@pytest.mark.parametrize("settings", METHODS)
def test_non_numeric_and_bool_columns_round_trip_through_the_codes(
    settings: dict[str, Any],
) -> None:
    # Catches codes leaking into the model, dtypes lost on the way back, or
    # the codes taken from the background alone ("c" is only in X).
    model = _SpyModel()
    explanation = ShapExplainer(**settings).explain(model, X, BACKGROUND)
    assert explanation.values.shape == (12, 4)
    assert all(model.exposure_all_ones)
    expected = {"x": "float64", "count": "int64", "flag": "bool", "kind": "str"}
    assert all(dtypes == expected for dtypes in model.dtypes_seen)
    assert "c" in model.kinds_seen
    assert model.kinds_seen <= {"a", "b", "c"}


@pytest.mark.parametrize("settings", METHODS)
def test_the_whole_background_is_used(settings: dict[str, Any]) -> None:
    # Catches shap's default cap of 100 background rows, which drops the
    # rest silently: every one of 120 distinct background values must reach
    # the model.
    background = pd.DataFrame(
        {
            "x": np.arange(120, dtype=float),
            "count": np.zeros(120, dtype=int),
            "flag": np.zeros(120, dtype=bool),
            "kind": np.tile(["a", "b"], 60),
        }
    )
    model = _SpyModel()
    ShapExplainer(**settings).explain(model, X.iloc[:2], background)
    assert set(range(120)) <= model.x_seen


def test_same_seed_gives_identical_values() -> None:
    # Catches a seed that is not passed to the permutation explainer.
    def values(random_state: int) -> np.ndarray:
        explainer = ShapExplainer("permutation", max_evals=9, random_state=random_state)
        return explainer.explain(_SpyModel(), X, BACKGROUND).values

    np.testing.assert_array_equal(values(0), values(0))
    assert not np.array_equal(values(0), values(1))


def test_parallel_rows_give_the_same_values() -> None:
    # Catches rows assembled out of order or an exposure paired with the
    # wrong row when the rows are explained in worker processes.
    model, X_train, exposure = _fitted_count_model()
    rows, rows_exposure, background = (
        X_train.iloc[:8],
        exposure[:8],
        X_train.iloc[20:40],
    )
    sequential = ShapExplainer("exact", n_jobs=1).explain(
        model, rows, background, exposure=rows_exposure
    )
    parallel = ShapExplainer("exact", n_jobs=2).explain(
        model, rows, background, exposure=rows_exposure
    )
    np.testing.assert_array_equal(parallel.values, sequential.values)
    np.testing.assert_array_equal(parallel.base_values, sequential.base_values)


def test_parallel_rows_run_in_other_processes() -> None:
    # Catches n_jobs not reaching joblib: with workers, the spy's records
    # stay in the workers and the parent's copy sees only the one predict
    # call that reads the output names.
    sequential, parallel = _SpyModel(), _SpyModel()
    ShapExplainer("exact", n_jobs=1).explain(sequential, X.iloc[:2], BACKGROUND)
    ShapExplainer("exact", n_jobs=2).explain(parallel, X.iloc[:2], BACKGROUND)
    assert len(sequential.dtypes_seen) > 1
    assert len(parallel.dtypes_seen) == 1


class _AdditiveModel(BaseAgeGroupModel):
    """Predicts ``2 x + 3 count``; the exact SHAP value of a column is then known."""

    def fit(self, X: Any, y: Any, exposure: ArrayLike | None = None) -> Self:
        return self

    def predict(self, X: Any, exposure: ArrayLike | None = None) -> np.ndarray:
        return 2.0 * np.asarray(X["x"], dtype=float) + 3.0 * np.asarray(
            X["count"], dtype=float
        )


def test_exact_values_of_an_additive_model_and_a_reordered_background() -> None:
    # Catches a column paired with another column's values (the masker's
    # columns are taken by position): for an additive model the exact SHAP
    # value is coef * (x - mean of the background column), and 0 for an
    # unused column, whatever the background's column order.
    expected = np.column_stack(
        [
            2.0 * (X["x"] - BACKGROUND["x"].mean()),
            3.0 * (X["count"] - BACKGROUND["count"].mean()),
            np.zeros(len(X)),
            np.zeros(len(X)),
        ]
    )
    for background in (BACKGROUND, BACKGROUND[["kind", "count", "flag", "x"]]):
        explanation = ShapExplainer("exact").explain(_AdditiveModel(), X, background)
        np.testing.assert_allclose(explanation.values, expected, atol=1e-12)
    assert explanation.display_data[0].tolist() == X.iloc[0].tolist()


def test_a_missing_value_in_a_string_column_reaches_the_model() -> None:
    # Catches the decode step choking on NaN (scikit-learn writes into the
    # read-only array pandas hands out) or turning it into a level.
    X_missing = X.copy()
    X_missing.loc[X_missing.index[0], "kind"] = None
    model = _SpyModel()
    ShapExplainer("exact").explain(model, X_missing, BACKGROUND)
    assert any(isinstance(kind, float) and np.isnan(kind) for kind in model.kinds_seen)
    assert {"a", "b", "c"} <= model.kinds_seen


def test_a_categorical_column_keeps_a_background_only_level() -> None:
    # Catches the cast back to X's categorical dtype, whose categories lack
    # the background's level "d": it would become NaN without an error.
    X_categorical = X.copy()
    X_categorical["kind"] = pd.Categorical(X_categorical["kind"])
    background = BACKGROUND.copy()
    background["kind"] = np.tile(["a", "d"], 10)
    model = _SpyModel()
    ShapExplainer("exact").explain(model, X_categorical, background)
    assert "d" in model.kinds_seen
    assert all(dtypes["kind"] == "category" for dtypes in model.dtypes_seen)


def test_exposure_of_the_wrong_length_raises() -> None:
    # Catches a shorter exposure explaining fewer rows silently: the rows
    # are explained one exposure value at a time.
    with pytest.raises(ValueError, match="exposure has 2 values for 12 rows"):
        ShapExplainer("exact").explain(_SpyModel(), X, BACKGROUND, exposure=np.ones(2))


@pytest.mark.parametrize(
    "settings",
    [
        {"method": "exact", "max_eval": 10},
        {"method": "permutation", "max_evals": True},
        {"method": "permutation", "max_evals": "50"},
        {"method": "permutation", "random_state": 0.5},
        {"method": "kernel"},
        {"method": "exact", "random_state": 0},
        {},
        {"method": "exact", "n_jobs": True},
        {"method": "exact", "n_jobs": "2"},
    ],
    ids=[
        "misspelt-keyword",
        "bool-max-evals",
        "string-max-evals",
        "float-seed",
        "unknown-method",
        "seed-with-exact",
        "missing-method",
        "bool-n-jobs",
        "string-n-jobs",
    ],
)
def test_invalid_settings_raise(settings: dict[str, Any]) -> None:
    # Catches lax settings: pydantic's default ignores an extra keyword,
    # reads True as 1 and "50" as 50; the exact method has no seed to take.
    with pytest.raises(ValidationError):
        ShapExplainer(**settings)


@pytest.mark.parametrize("settings", METHODS)
def test_total_times_probability_model_with_a_string_column(
    settings: dict[str, Any],
) -> None:
    # Catches the real-model path: the transformer inside predict must see
    # the string column school_status as fitted, and the explanation must
    # carry the cohort names.
    path = Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"
    config = load_simulation_config(path)
    config = config.model_copy(
        update={
            "simulation": config.simulation.model_copy(update={"n_neighborhoods": 20})
        }
    )
    table = StudentPopulationSimulator.from_config(config).simulate().dataset
    numeric = ("ses", "avg_household_size", "median_age", "n_daycares_500m")
    features = FeatureTransformer(
        plans=(
            ColumnPlan(name="numeric", columns=numeric, transforms=(Standardize(),)),
            ColumnPlan(
                name="school",
                columns=("school_status",),
                transforms=(
                    OneHot(
                        categories=("none", "existing", "planned"),
                        reference_category="none",
                    ),
                ),
            ),
        )
    )
    X_table = table[[*numeric, "school_status"]]
    Y_table = table[["n_kindergarten", "n_elementary", "n_highschool"]]
    exposure = ExposureTransformer("n_apartments").fit_transform(table)
    model = TotalTimesProbabilityModel(
        total_model=CountModel(
            estimator=PoissonRegressor(),
            use_exposure=True,
            feature_transformer=features,
        ),
        probability_model=CohortProbabilityModel(
            estimator=LogisticRegression(max_iter=1000),
            feature_transformer=features,
        ),
    ).fit(X_table, Y_table, exposure=exposure)
    settings = settings | (
        {"max_evals": 11} if settings["method"] == "permutation" else {}
    )
    explanation = ShapExplainer(**settings).explain(
        model, X_table.iloc[:8], X_table.iloc[20:40]
    )
    assert explanation.values.shape == (8, 5, 3)
    assert explanation.output_names == list(Y_table.columns)
    assert list(explanation.feature_names) == list(X_table.columns)
    assert np.isfinite(explanation.values).all()
