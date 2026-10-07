"""CountModel: each test names the mistake in our code it would catch."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from lightgbm.basic import LightGBMError
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.modeling import CountModel, Regressor


def _data(rows: int = 300) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Counts proportional to building size, with a rate that depends on ``ses``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"ses": rng.normal(size=rows), "noise": rng.normal(size=rows)})
    exposure = rng.integers(12, 80, size=rows).astype(float)
    y = pd.Series(rng.poisson(exposure * np.exp(-2 + 0.4 * X["ses"])))
    return X, y, exposure


X, Y, EXPOSURE = _data()


def _lightgbm(n_estimators: int = 20) -> LGBMRegressor:
    return LGBMRegressor(
        objective="poisson",
        n_estimators=n_estimators,
        # 1: more OpenMP threads crash alongside torch on macOS.
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        random_state=42,
    )


# One per kind of estimator the model must serve; the model clones each, so
# sharing the instances across tests is safe.
POISSON_ESTIMATORS = {
    "lightgbm": _lightgbm(),
    "hist-gradient-boosting": HistGradientBoostingRegressor(
        loss="poisson", max_iter=20, random_state=0
    ),
    # Unpenalized: its alpha would act as alpha * mean(exposure) in the offset
    # model (it normalizes sample_weight), shrinking ses almost to zero here.
    "poisson-glm": PoissonRegressor(alpha=0.0),
}
ESTIMATORS = {
    **POISSON_ESTIMATORS,
    "lightgbm-gaussian": LGBMRegressor(
        objective="regression", n_estimators=20, n_jobs=1, verbosity=-1
    ),
}


class _Recorder(BaseEstimator):
    """A regressor that keeps what ``fit`` received and predicts 1 per row."""

    def fit(
        self, X: pd.DataFrame, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self:
        self.y_ = y
        self.sample_weight_ = sample_weight
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X))


class _WithoutSampleWeight(BaseEstimator):
    """A regressor whose ``fit`` takes no ``sample_weight``."""

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> Self:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X))


def test_clone_and_set_params_change_only_the_copy() -> None:
    # A tuner builds each trial's model this way; it breaks if __init__ alters
    # or drops an argument, or if the copy shares the template's estimator.
    model = CountModel(estimator=_lightgbm(50), use_exposure=True)

    trial = clone(model).set_params(estimator__n_estimators=5)

    assert trial.get_params()["estimator__n_estimators"] == 5
    assert trial.get_params()["use_exposure"] is True
    assert model.get_params()["estimator__n_estimators"] == 50


def test_the_template_estimator_stays_unfitted() -> None:
    # Fitting the caller's estimator in place would leak one fold's fit into
    # the next and into the caller's later use.
    template = _lightgbm()

    model = CountModel(estimator=template).fit(X, Y)

    assert model.estimator_ is not template
    with pytest.raises(NotFittedError):
        check_is_fitted(template)


@pytest.mark.parametrize(
    ("exposure", "message"),
    [
        (None, "pass `exposure`"),
        (EXPOSURE[:, None], "one-dimensional"),
        (EXPOSURE[:1], "inconsistent numbers of samples"),
        (EXPOSURE[:-1], "inconsistent numbers of samples"),
    ],
    ids=["missing-while-on", "two-dimensional", "length-one", "one-row-short"],
)
def test_exposure_misuse_raises(exposure: np.ndarray | None, message: str) -> None:
    # The first two would pass silently: a dropped exposure, or a column that
    # broadcasts into an (n, n) target. A wrong length is rejected at fit by
    # our check, as scikit-learn checks every per-row array, not only by the
    # estimator. The values are checked by preprocessing.ExposureTransformer
    # and its tests.
    with pytest.raises(ValueError, match=message):
        CountModel(estimator=_lightgbm(), use_exposure=True).fit(
            X, Y, exposure=exposure
        )


@pytest.mark.parametrize(
    "exposure", [EXPOSURE[:1], EXPOSURE[:-1]], ids=["length-one", "one-row-short"]
)
def test_a_wrong_length_exposure_raises_at_predict(exposure: np.ndarray) -> None:
    # Without the check numpy would broadcast a length-1 exposure to every
    # building silently, and fail on another length with a broadcast error.
    model = CountModel(estimator=_lightgbm(5), use_exposure=True).fit(
        X, Y, exposure=EXPOSURE
    )

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        model.predict(X, exposure=exposure)


def test_an_unused_exposure_is_ignored() -> None:
    # A caller passes one exposure to every model; with use_exposure=False it
    # must not enter the fit or the prediction.
    model = CountModel(estimator=_lightgbm())
    without = clone(model).fit(X, Y).predict(X)

    with_exposure = clone(model).fit(X, Y, exposure=EXPOSURE)

    np.testing.assert_array_equal(with_exposure.predict(X), without)
    np.testing.assert_array_equal(with_exposure.predict(X, exposure=EXPOSURE), without)


def test_fit_receives_the_rate_with_the_exposure_as_weight() -> None:
    # The whole exposure model is this one call: a missing weight, the count
    # in place of the rate, or y * exposure would each fit another model, by a
    # margin a fitted estimator's mean can hide (0.5% without the weight).
    model = CountModel(estimator=_Recorder(), use_exposure=True).fit(
        X, Y, exposure=EXPOSURE
    )

    recorder = model.estimator_
    assert isinstance(recorder, _Recorder)
    np.testing.assert_array_equal(recorder.y_, Y.to_numpy() / EXPOSURE)
    np.testing.assert_array_equal(recorder.sample_weight_, EXPOSURE)


@pytest.mark.parametrize("estimator", ESTIMATORS.values(), ids=list(ESTIMATORS))
def test_doubling_the_exposure_doubles_the_prediction(estimator: Regressor) -> None:
    # Exact because the exposure is not a feature; fails if predict drops the
    # exposure, for every loss, the Gaussian one included.
    model = CountModel(estimator=estimator, use_exposure=True).fit(
        X, Y, exposure=EXPOSURE
    )

    np.testing.assert_allclose(
        model.predict(X, exposure=2 * EXPOSURE),
        2 * model.predict(X, exposure=EXPOSURE),
        rtol=1e-12,
    )


@pytest.mark.parametrize("use_exposure", [True, False], ids=["exposure", "plain"])
@pytest.mark.parametrize(
    "estimator", POISSON_ESTIMATORS.values(), ids=list(POISSON_ESTIMATORS)
)
def test_training_mean_prediction_matches_the_target_mean(
    estimator: Regressor, use_exposure: bool
) -> None:
    # A Poisson fit reproduces the training mean (within 0.2% here). Catches a
    # prediction on the wrong scale, e.g. rates returned as counts or the
    # exposure applied twice. The plain cases are the only cover of the path
    # without exposure.
    exposure = EXPOSURE if use_exposure else None
    model = CountModel(estimator=estimator, use_exposure=use_exposure).fit(
        X, Y, exposure=exposure
    )

    predictions = model.predict(X, exposure=exposure)

    assert predictions.mean() == pytest.approx(Y.mean(), rel=0.01)


def test_the_weighted_rate_equals_lightgbms_offset() -> None:
    # Catches a reformulation that is not the offset model log(exposure): the
    # two are the same likelihood, so LightGBM grows the same trees (measured
    # agreement 1.7e-8, floating point only).
    model = CountModel(estimator=_lightgbm(), use_exposure=True).fit(
        X, Y, exposure=EXPOSURE
    )
    # The offset model by hand: LightGBM skips boost_from_average once given an
    # init_score, so the intercept log(sum y / sum exposure) goes in with it.
    offset = np.log(EXPOSURE) + np.log(Y.sum() / EXPOSURE.sum())
    by_hand = _lightgbm().fit(X, Y, init_score=offset)

    np.testing.assert_allclose(
        model.predict(X, exposure=EXPOSURE),
        np.exp(by_hand.predict(X, raw_score=True) + offset),
        rtol=1e-6,
    )


def test_the_gaussian_weighted_rate_is_least_squares_of_the_count() -> None:
    # The Gaussian model with an exposure: mean exposure * f(x), variance
    # proportional to the exposure, i.e. least squares of the count on
    # [exposure, exposure * x] with weights 1 / exposure. Catches a Gaussian
    # path that fits another model, e.g. without the weight.
    model = CountModel(estimator=LinearRegression(), use_exposure=True).fit(
        X, Y, exposure=EXPOSURE
    )
    design = np.column_stack([EXPOSURE, EXPOSURE[:, None] * X.to_numpy()])
    count_model = LinearRegression(fit_intercept=False).fit(
        design, Y, sample_weight=1 / EXPOSURE
    )

    np.testing.assert_allclose(
        model.predict(X, exposure=EXPOSURE), count_model.predict(design), rtol=1e-8
    )


def test_predict_follows_how_the_model_was_fitted() -> None:
    # Turning use_exposure off after fitting must not silently return rates per
    # apartment instead of counts.
    model = CountModel(estimator=_lightgbm(5), use_exposure=True).fit(
        X, Y, exposure=EXPOSURE
    )
    model.set_params(use_exposure=False)

    with pytest.raises(ValueError, match="pass `exposure`"):
        model.predict(X)
    assert model.predict(X, exposure=EXPOSURE).mean() == pytest.approx(
        Y.mean(), rel=0.05
    )


def test_a_regressor_without_sample_weight_surfaces_the_library_error() -> None:
    # The model does not pre-check the estimator: Python's own error names the
    # missing argument, so a check of ours would only repeat it.
    model = CountModel(
        estimator=_WithoutSampleWeight(),
        use_exposure=True,
    )

    with pytest.raises(TypeError, match="sample_weight"):
        model.fit(X, Y, exposure=EXPOSURE)


def test_an_all_zero_target_surfaces_lightgbms_error() -> None:
    # Pinned because the model deliberately leaves this check to the estimator.
    with pytest.raises(LightGBMError, match="sum of labels is zero"):
        CountModel(estimator=_lightgbm(), use_exposure=True).fit(
            X, Y * 0, exposure=EXPOSURE
        )
