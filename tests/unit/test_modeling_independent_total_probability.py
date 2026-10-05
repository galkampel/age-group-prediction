"""IndependentTotalProbabilityModel: each test names the mistake in our code it would catch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.frozen import FrozenEstimator
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.modeling import (
    CohortProbabilityModel,
    IndependentTotalProbabilityModel,
    TemperatureCalibrator,
    TotalChildrenModel,
)

COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]


def _features(column: str) -> FeatureTransformer:
    return FeatureTransformer(
        (ColumnPlan(name=column, columns=(column,), transforms=(Center(),)),)
    )


def _table(n_rows: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """A raw table on a non-default index, its cohort counts, its exposure.

    The total grows with the exposure and ``x``; the composition moves with
    ``z``; the exposure is in no feature, so it enters through the offset only.
    Every building has a child.
    """
    rng = np.random.default_rng(0)
    index = pd.RangeIndex(100, 100 + n_rows)
    exposure = pd.Series(rng.integers(5, 40, n_rows).astype(float), index=index)
    table = pd.DataFrame(
        {"x": rng.normal(size=n_rows), "z": rng.normal(size=n_rows)}, index=index
    )
    table["n_apartments"] = exposure
    totals = 1 + rng.poisson(0.4 * exposure * np.exp(0.3 * table["x"]))
    concentration = np.exp(
        np.array([1.0, 1.5, 0.5]) + np.outer(table["z"], [0.5, 0, -0.5])
    )
    shares = np.vstack([rng.dirichlet(row) for row in concentration])
    counts = np.vstack([rng.multinomial(n, s) for n, s in zip(totals, shares)])
    table[COHORTS] = counts
    return table, table[COHORTS], exposure


def _model(
    calibrator: TemperatureCalibrator | FrozenEstimator | None = None,
) -> IndependentTotalProbabilityModel:
    return IndependentTotalProbabilityModel(
        total_children_model=TotalChildrenModel(feature_transformer=_features("x")),
        cohort_probability_model=CohortProbabilityModel(
            feature_transformer=_features("z")
        ),
        temperature_calibrator=calibrator,
    )


@pytest.mark.filterwarnings("error")
def test_the_output_is_the_total_times_the_shares_of_the_models_fitted_alone() -> None:
    # Catches a model fitted on the wrong target or with the other model's
    # features, or predictions combined other than by multiplication: every
    # cohort must sum to the total.
    table, y, exposure = _table()
    model = _model().fit(table, y, exposure=exposure)
    total = clone(model.total_children_model).fit(
        table, y.sum(axis=1), exposure=exposure
    )
    shares = clone(model.cohort_probability_model).fit(table, y)

    predictions = model.predict(table, exposure=exposure)

    np.testing.assert_allclose(
        predictions.to_numpy(),
        total.predict(table, exposure=exposure)[:, None]
        * shares.predict(table).to_numpy(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        predictions.sum(axis=1), total.predict(table, exposure=exposure), rtol=1e-12
    )


def test_the_total_is_the_row_sum_of_y_and_predict_needs_no_target_columns() -> None:
    # One cohort taken as the total, or the total read from X, would fail on
    # new buildings or predict a fraction of the children.
    table, y, exposure = _table()
    model = _model().fit(table, y, exposure=exposure)
    total_alone = TotalChildrenModel(feature_transformer=_features("x")).fit(
        table, y.sum(axis=1), exposure=exposure
    )

    predictions = model.predict(table.drop(columns=COHORTS), exposure=exposure)

    np.testing.assert_allclose(
        predictions.sum(axis=1),
        total_alone.predict(table, exposure=exposure),
        rtol=1e-12,
    )


def test_doubling_the_exposure_doubles_every_cohort() -> None:
    # Exact because the exposure is in no feature and reaches the total model
    # only: it would be lost at predict, or change the shares if it reached
    # the probability model.
    table, y, exposure = _table()
    model = _model().fit(table, y, exposure=exposure)

    single = model.predict(table, exposure=exposure)
    double = model.predict(table, exposure=2 * exposure)

    np.testing.assert_allclose(double.to_numpy(), 2 * single.to_numpy(), rtol=1e-12)


def test_nested_settings_reach_the_next_fit() -> None:
    # The sub-models are named parameters, so a tuner sets their
    # settings through nested names; fit must fit the templates as set.
    table, y, exposure = _table()
    model = _model()
    tuned = clone(model).set_params(cohort_probability_model__l2_penalty=10.0)

    fitted = model.fit(table, y, exposure=exposure)
    fitted_tuned = tuned.fit(table, y, exposure=exposure)

    assert fitted_tuned.cohort_probability_model_.l2_penalty == 10.0
    assert not np.allclose(
        fitted.predict(table, exposure=exposure),
        fitted_tuned.predict(table, exposure=exposure),
    )


def test_a_given_calibrator_is_applied_to_the_shares() -> None:
    # The calibrator ignored, or applied to something other than the
    # probability model's logits, would leave the shares uncalibrated.
    table, y, exposure = _table()
    calibrator = TemperatureCalibrator()
    calibrator.temperature_ = 3.0
    plain = _model().fit(table, y, exposure=exposure)
    calibrated = _model(calibrator).fit(table, y, exposure=exposure)

    predictions = calibrated.predict(table, exposure=exposure)

    expected = (
        plain.predict(table, exposure=exposure).sum(axis=1).to_numpy()[:, None]
        * calibrator.predict(
            plain.cohort_probability_model_.predict_logits(table)  # type: ignore[attr-defined]
        ).to_numpy()
    )
    np.testing.assert_allclose(predictions.to_numpy(), expected, rtol=1e-12)
    assert not np.allclose(predictions, plain.predict(table, exposure=exposure))


def test_an_unfitted_calibrator_raises_and_a_frozen_one_survives_clone() -> None:
    # A clone (a tuner's) drops the calibrator's fit, so a cloned model must
    # raise (the calibrator's own NotFittedError) rather than predict with
    # some temperature; FrozenEstimator is the documented way to keep the fit.
    table, y, exposure = _table()
    rng = np.random.default_rng(1)
    fitted = TemperatureCalibrator().fit(
        rng.normal(size=(50, 3)), rng.integers(1, 9, size=(50, 3))
    )

    with pytest.raises(NotFittedError):
        clone(_model(fitted)).fit(table, y, exposure=exposure).predict(
            table, exposure=exposure
        )
    frozen = clone(_model(FrozenEstimator(fitted))).fit(table, y, exposure=exposure)
    pd.testing.assert_frame_equal(
        frozen.predict(table, exposure=exposure),
        _model(fitted)
        .fit(table, y, exposure=exposure)
        .predict(table, exposure=exposure),
    )


def test_predict_follows_the_calibrator_given_at_fit() -> None:
    # Every other setting reaches only the next fit (the models are cloned
    # there); a calibrator read at predict would change a fitted model's
    # output without a refit.
    table, y, exposure = _table()
    calibrator = TemperatureCalibrator()
    calibrator.temperature_ = 3.0
    model = _model().fit(table, y, exposure=exposure)
    before = model.predict(table, exposure=exposure)

    model.set_params(temperature_calibrator=calibrator)

    pd.testing.assert_frame_equal(model.predict(table, exposure=exposure), before)
    model.fit(table, y, exposure=exposure)
    assert not np.allclose(model.predict(table, exposure=exposure), before)


def test_the_templates_stay_unfitted() -> None:
    # Fitting the caller's sub-models in place would leak one fold's fit into
    # the next.
    table, y, exposure = _table()
    model = _model().fit(table, y, exposure=exposure)

    for template in (model.total_children_model, model.cohort_probability_model):
        with pytest.raises(NotFittedError):
            check_is_fitted(template)


def test_a_failing_fit_leaves_the_previous_fit_intact() -> None:
    # Fitted state assigned model by model would pair a new total with the old
    # shares.
    table, y, exposure = _table()
    model = _model().fit(table, y, exposure=exposure)
    before = model.predict(table, exposure=exposure)

    # The total fits; then the probability model needs every cohort observed.
    with pytest.raises(ValueError, match="no child"):
        model.fit(table, y.assign(n_highschool=0), exposure=exposure)

    pd.testing.assert_frame_equal(model.predict(table, exposure=exposure), before)


def test_columns_follow_y_and_rows_follow_x() -> None:
    # A default index or the model's own column order would misalign rows or
    # misname cohorts.
    table, y, exposure = _table()
    model = _model().fit(table, y[COHORTS[::-1]], exposure=exposure)

    predictions = model.predict(table.iloc[::2], exposure=exposure.iloc[::2])

    assert list(predictions.columns) == COHORTS[::-1]
    pd.testing.assert_index_equal(predictions.index, table.index[::2])


class _RenamedShares(CohortProbabilityModel):
    """A probability model whose output names its cohorts differently."""

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> pd.DataFrame:
        return super().predict(X).rename(columns=str.upper)


def test_a_probability_model_with_other_cohorts_raises() -> None:
    # Shares under other names, or in another order, would be multiplied into
    # y's cohorts by position, silently.
    table, y, exposure = _table()
    model = IndependentTotalProbabilityModel(
        total_children_model=TotalChildrenModel(feature_transformer=_features("x")),
        cohort_probability_model=_RenamedShares(feature_transformer=_features("z")),
    ).fit(table, y, exposure=exposure)

    with pytest.raises(ValueError, match="predicts cohorts"):
        model.predict(table, exposure=exposure)


class _SeriesTotal(TotalChildrenModel):
    """A total model whose predictions carry their own default index."""

    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> pd.Series:
        return pd.Series(super().predict(X, exposure=exposure))


def test_predictions_are_placed_by_position_not_by_their_own_index() -> None:
    # A total with its own index would be realigned to X's, into NaN, silently.
    table, y, exposure = _table()
    model = IndependentTotalProbabilityModel(
        total_children_model=_SeriesTotal(feature_transformer=_features("x")),
        cohort_probability_model=CohortProbabilityModel(
            feature_transformer=_features("z")
        ),
    ).fit(table, y, exposure=exposure)

    predictions = model.predict(table, exposure=exposure)

    assert not predictions.isna().any().any()
