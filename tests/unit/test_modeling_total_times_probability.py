"""TotalTimesProbabilityModel: each test names the mistake in our code it would catch."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.modeling import (
    BaseAgeGroupModel,
    CohortProbabilityModel,
    CountModel,
    NegativeBinomialRegressor,
    TotalTimesProbabilityModel,
)

# Not in sorted order, so columns sorted anywhere would show.
COHORTS = ["kg", "el", "hs"]


def _table(n_rows: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Features on a non-default index, the cohort counts and the exposure."""
    rng = np.random.default_rng(0)
    index = pd.RangeIndex(100, 100 + n_rows)
    X = pd.DataFrame({"x": rng.normal(size=n_rows)}, index=index)
    exposure = pd.Series(rng.integers(5, 40, n_rows).astype(float), index=index)
    total = rng.poisson(0.2 * exposure * np.exp(0.3 * X["x"]))
    logits = np.column_stack([0.8 * X["x"], np.zeros(n_rows), -0.5 * X["x"]])
    shares = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    counts = np.array([rng.multinomial(n, p) for n, p in zip(total, shares)])
    return X, pd.DataFrame(counts, columns=COHORTS, index=index), exposure


def _model(total_model: BaseAgeGroupModel | None = None) -> TotalTimesProbabilityModel:
    return TotalTimesProbabilityModel(
        total_model=total_model
        or CountModel(estimator=PoissonRegressor(), use_exposure=True),
        probability_model=CohortProbabilityModel(estimator=LogisticRegression()),
    )


class _TotalSpy(BaseAgeGroupModel):
    """Records its arguments; predicts 2 per row."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        self.y_ = y
        self.exposure_ = exposure
        return self

    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> np.ndarray:
        self.predict_exposure_ = exposure
        return np.full(len(X), 2.0)


class _ProbabilitySpy(CohortProbabilityModel):
    """A CohortProbabilityModel that records its arguments."""

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> Self:
        self.y_ = y
        self.exposure_ = exposure
        return super().fit(X, y, exposure=exposure)

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> pd.DataFrame:
        self.predict_exposure_ = exposure
        return super().predict(X, exposure=exposure)


def _spied_model() -> TotalTimesProbabilityModel:
    return TotalTimesProbabilityModel(
        total_model=_TotalSpy(),
        probability_model=_ProbabilitySpy(estimator=LogisticRegression()),
    )


def test_fit_gives_the_row_sums_to_the_total_and_y_to_the_probabilities() -> None:
    # Catches a total from one column, the two targets swapped, or the
    # exposure withheld from either half.
    X, y, exposure = _table()

    model = _spied_model().fit(X, y, exposure=exposure)

    total_model = model.total_model_
    probability_model = model.probability_model_
    assert isinstance(total_model, _TotalSpy)
    assert isinstance(probability_model, _ProbabilitySpy)
    pd.testing.assert_series_equal(total_model.y_, y.sum(axis=1))
    pd.testing.assert_frame_equal(probability_model.y_, y)
    assert total_model.exposure_ is exposure
    assert probability_model.exposure_ is exposure


def test_predict_passes_the_exposure_to_both_halves() -> None:
    # Catches the exposure dropped at predict, which would drop the total's offset.
    X, y, exposure = _table()
    model = _spied_model().fit(X, y, exposure=exposure)

    model.predict(X, exposure=exposure)

    assert isinstance(model.total_model_, _TotalSpy)
    assert isinstance(model.probability_model_, _ProbabilitySpy)
    assert model.total_model_.predict_exposure_ is exposure
    assert model.probability_model_.predict_exposure_ is exposure


def test_predict_is_the_total_times_each_probability() -> None:
    # Against the two halves fitted alone. Catches the total broadcast along
    # the cohorts instead of the rows, and columns or index lost or reordered.
    X, y, exposure = _table()
    total_model = CountModel(estimator=PoissonRegressor(), use_exposure=True)
    probability_model = CohortProbabilityModel(estimator=LogisticRegression())

    predictions = (
        TotalTimesProbabilityModel(
            total_model=total_model, probability_model=probability_model
        )
        .fit(X, y, exposure=exposure)
        .predict(X, exposure=exposure)
    )

    total = clone(total_model).fit(X, y.sum(axis=1), exposure=exposure)
    probabilities = clone(probability_model).fit(X, y).predict(X)
    expected = pd.DataFrame(
        probabilities.to_numpy() * total.predict(X, exposure=exposure)[:, None],
        columns=COHORTS,
        index=X.index,
    )
    pd.testing.assert_frame_equal(predictions, expected)


@pytest.mark.parametrize(
    ("total_model", "use_exposure"),
    [
        (CountModel(estimator=PoissonRegressor(), use_exposure=True), True),
        (CountModel(estimator=PoissonRegressor()), False),
        (
            CountModel(estimator=NegativeBinomialRegressor(), use_exposure=True),
            True,
        ),
    ],
    ids=["poisson-exposure", "poisson", "nb2-exposure"],
)
def test_each_row_sums_to_the_predicted_total(
    total_model: CountModel, use_exposure: bool
) -> None:
    # Each total variant: the cohorts must split the total, not rescale it.
    X, y, exposure = _table()
    passed = exposure if use_exposure else None

    model = _model(total_model).fit(X, y, exposure=passed)

    np.testing.assert_allclose(
        model.predict(X, exposure=passed).sum(axis=1).to_numpy(),
        model.total_model_.predict(X, exposure=passed),
        rtol=1e-12,
    )


def test_the_templates_stay_unfitted() -> None:
    # Fitting the caller's models in place would carry one fold's fit into the next.
    X, y, exposure = _table()
    model = _model()

    model.fit(X, y, exposure=exposure)

    assert model.total_model_ is not model.total_model
    assert model.probability_model_ is not model.probability_model
    for template in (model.total_model, model.probability_model):
        with pytest.raises(NotFittedError):
            check_is_fitted(template)


def test_a_failing_half_leaves_the_previous_fit_intact() -> None:
    # The total fits before the probabilities fail (a cohort without a child),
    # so a total set at once would pair the new total with the old probabilities.
    X, y, exposure = _table()
    model = _model().fit(X, y, exposure=exposure)
    before = model.predict(X, exposure=exposure)

    with pytest.raises(ValueError, match="no child is observed"):
        model.fit(X, y.assign(hs=0), exposure=exposure)

    pd.testing.assert_frame_equal(model.predict(X, exposure=exposure), before)


def test_nested_settings_reach_the_fitted_halves() -> None:
    # The tuner sets both halves through nested names; fit must use them.
    X, y, exposure = _table()
    model = _model().set_params(
        total_model__estimator__alpha=10.0,
        probability_model__calibration_method="temperature",
    )

    model.fit(X, y, exposure=exposure)

    total_model = model.total_model_
    assert isinstance(total_model, CountModel)
    assert isinstance(total_model.estimator_, PoissonRegressor)
    assert total_model.estimator_.alpha == 10.0
    assert isinstance(model.probability_model_.estimator_, CalibratedClassifierCV)
