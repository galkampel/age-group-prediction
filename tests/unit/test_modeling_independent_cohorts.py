"""IndependentCohortModels: each test names the mistake in our code it would catch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike
from sklearn.base import clone

from age_group_prediction.feature_engineering import (
    Center,
    FeatureTransformer,
    Plan,
)
from age_group_prediction.modeling import (
    CountModel,
    IndependentCohortModels,
)

COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]


def _lightgbm() -> LGBMRegressor:
    # One thread: more OpenMP threads crash alongside torch on macOS.
    return LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1)


def _features(column: str) -> FeatureTransformer:
    return FeatureTransformer(((Plan(name=column, transforms=(Center(),)), [column]),))


def _table(n_rows: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """A raw table with its targets on a non-default index, the targets, the exposure.

    Each cohort depends on its own column, and the exposure is in no feature,
    so it is the only way building size enters.
    """
    rng = np.random.default_rng(0)
    index = pd.RangeIndex(100, 100 + n_rows)
    exposure = pd.Series(rng.integers(5, 40, n_rows).astype(float), index=index)
    table = pd.DataFrame(
        {"x": rng.normal(size=n_rows), "z": rng.normal(size=n_rows)}, index=index
    )
    table["n_apartments"] = exposure
    table["n_kindergarten"] = rng.poisson(0.1 * exposure * np.exp(0.5 * table["x"]))
    table["n_elementary"] = rng.poisson(np.exp(1 - 0.5 * table["z"]))
    table["n_highschool"] = rng.poisson(0.2 * exposure * np.exp(-0.3 * table["x"]))
    return table, table[COHORTS], exposure


def _cohort_models() -> dict[str, CountModel]:
    """Different features and offsets per cohort, in another order than y's columns."""
    return {
        "n_highschool": CountModel(
            estimator=_lightgbm(), use_exposure=True, feature_transformer=_features("x")
        ),
        "n_elementary": CountModel(
            estimator=_lightgbm(), feature_transformer=_features("z")
        ),
        "n_kindergarten": CountModel(
            estimator=_lightgbm(), use_exposure=True, feature_transformer=_features("x")
        ),
    }


# Also the check that cohorts with different features and offsets raise no warning.
@pytest.mark.filterwarnings("error")
def test_each_column_equals_that_cohorts_model_fitted_alone() -> None:
    # Catches a cohort fitted on another cohort's target, or given another
    # cohort's model.
    table, y, exposure = _table()
    cohort_models = _cohort_models()

    predictions = (
        IndependentCohortModels(cohort_models)
        .fit(table, y, exposure=exposure)
        .predict(table, exposure=exposure)
    )

    for cohort, model in cohort_models.items():
        alone = clone(model).fit(table, y[cohort], exposure=exposure)
        np.testing.assert_array_equal(
            predictions[cohort], alone.predict(table, exposure=exposure)
        )


@pytest.mark.parametrize(
    "columns",
    [COHORTS[:2], [*COHORTS, "n_total"], [*COHORTS, "n_highschool"]],
    ids=["model-without-column", "column-without-model", "duplicated-column"],
)
def test_models_that_do_not_match_ys_columns_raise(columns: list[str]) -> None:
    # A cohort without a model would be dropped from the output silently, and a
    # duplicated column would reach its model as a DataFrame.
    table, _, exposure = _table()
    table["n_total"] = table[COHORTS].sum(axis=1)

    with pytest.raises(ValueError, match="one model per column of y"):
        IndependentCohortModels(_cohort_models()).fit(
            table, table[columns], exposure=exposure
        )


def test_a_failing_cohort_leaves_the_previous_fit_intact() -> None:
    # Fitted state assigned cohort by cohort would pair a new fit of the first
    # cohorts with the old fit of the rest.
    table, y, exposure = _table()
    model = IndependentCohortModels(_cohort_models()).fit(table, y, exposure=exposure)
    before = model.predict(table, exposure=exposure)

    # n_elementary fits on the new rows; then n_kindergarten needs the exposure.
    with pytest.raises(ValueError, match="pass `exposure`"):
        model.fit(table.iloc[:100], y[["n_elementary", *COHORTS[::2]]].iloc[:100])

    pd.testing.assert_frame_equal(model.predict(table, exposure=exposure), before)


def test_columns_follow_y_and_rows_follow_X() -> None:
    # Columns in the mapping's order, or a default index, would misname cohorts
    # or misalign rows with the table.
    table, y, exposure = _table()

    predictions = (
        IndependentCohortModels(_cohort_models())
        .fit(table, y, exposure=exposure)
        .predict(table.iloc[::2], exposure=exposure.iloc[::2])
    )

    assert list(predictions.columns) == COHORTS
    pd.testing.assert_index_equal(predictions.index, table.index[::2])


def test_predict_needs_no_target_columns() -> None:
    # The table at fit holds the targets; one at predict does not. Reading a
    # target from X would fail on new buildings or leak the answer.
    table, y, exposure = _table()
    model = IndependentCohortModels(_cohort_models()).fit(table, y, exposure=exposure)

    pd.testing.assert_frame_equal(
        model.predict(table.drop(columns=COHORTS), exposure=exposure),
        model.predict(table, exposure=exposure),
    )


def test_doubling_the_exposure_doubles_only_the_cohorts_with_an_offset() -> None:
    # Catches the exposure lost at predict, or applied to a cohort without an
    # offset.
    table, y, exposure = _table()
    model = IndependentCohortModels(_cohort_models()).fit(table, y, exposure=exposure)

    single = model.predict(table, exposure=exposure)
    double = model.predict(table, exposure=2 * exposure)

    for cohort in ("n_kindergarten", "n_highschool"):
        np.testing.assert_allclose(double[cohort], 2 * single[cohort], rtol=1e-12)
    np.testing.assert_array_equal(double["n_elementary"], single["n_elementary"])


class _SeriesPredictions(CountModel):
    """A cohort model whose predictions carry their own default index."""

    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> pd.Series:
        return pd.Series(super().predict(X, exposure=exposure))


def test_predictions_are_placed_by_position_not_by_their_own_index() -> None:
    # A prediction with its own index would be realigned to X's, into NaN,
    # silently.
    table, y, _ = _table()
    X = table[["x", "z"]]
    model = IndependentCohortModels(
        {"n_elementary": _SeriesPredictions(estimator=_lightgbm())}
    )

    predictions = model.fit(X, y[["n_elementary"]]).predict(X)

    assert not predictions["n_elementary"].isna().any()


def test_nested_names_reach_each_cohorts_model() -> None:
    # A name not forwarded into the mapping, or forwarded to another cohort's
    # model, would leave a study tuning a model it never changes.
    model = IndependentCohortModels(_cohort_models())

    model.set_params(
        cohort_models__n_kindergarten__estimator__learning_rate=0.01,
        cohort_models__n_highschool__estimator__learning_rate=0.3,
        cohort_models__n_elementary__use_exposure=True,
    )

    params = model.get_params(deep=True)
    assert params["cohort_models__n_kindergarten__estimator__learning_rate"] == 0.01
    assert params["cohort_models__n_highschool__estimator__learning_rate"] == 0.3
    assert model.cohort_models["n_elementary"].use_exposure is True
    assert params["cohort_models__n_elementary__estimator__learning_rate"] == 0.1


def test_set_params_on_a_clone_leaves_the_template_untouched() -> None:
    # The tuner sets each trial's parameters on a clone; a clone that shared
    # the template's models would carry one trial's values into the next.
    template = IndependentCohortModels(_cohort_models())

    clone(template).set_params(
        cohort_models__n_kindergarten__estimator__learning_rate=0.01
    )

    assert (
        template.get_params()["cohort_models__n_kindergarten__estimator__learning_rate"]
        == 0.1
    )


def test_the_whole_mapping_is_replaced_before_nested_names() -> None:
    # In key order, the nested name would land on the old model and the new
    # mapping would then drop it silently.
    model = IndependentCohortModels(_cohort_models())
    new_models = _cohort_models()

    model.set_params(
        cohort_models__n_kindergarten__estimator__learning_rate=0.01,
        cohort_models=new_models,
    )

    assert model.cohort_models is new_models
    assert new_models["n_kindergarten"].get_params()["estimator__learning_rate"] == 0.01


def test_a_whole_cohort_model_can_be_replaced() -> None:
    # A cohort key without a sub-name is one model, not a parameter of the
    # mapping; the caller's mapping is left as given.
    given = _cohort_models()
    model = IndependentCohortModels(given)
    replacement = CountModel(estimator=_lightgbm())

    model.set_params(cohort_models__n_elementary=replacement)

    assert model.cohort_models["n_elementary"] is replacement
    assert given["n_elementary"] is not replacement
    assert model.cohort_models.keys() == given.keys()


def test_a_misspelt_cohort_raises() -> None:
    # Skipped silently, a misspelt cohort would leave its model untuned for a
    # whole study.
    model = IndependentCohortModels(_cohort_models())

    with pytest.raises(KeyError, match="n_kindergarden"):
        model.set_params(cohort_models__n_kindergarden__estimator__learning_rate=0.01)
