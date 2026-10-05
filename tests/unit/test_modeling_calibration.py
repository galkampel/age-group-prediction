"""TemperatureCalibrator: each test names the mistake in our code it would catch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.base import clone
from sklearn.calibration import _TemperatureScaling
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

from age_group_prediction.feature_engineering import (
    Center,
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.modeling import (
    CohortProbabilityModel,
    TemperatureCalibrator,
)

COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]


def _data(
    rows: int = 400, factor: float = 1.0, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Logits ``factor · log p`` of Dirichlet shares, and multinomial counts drawn from ``p``.

    A factor of 1 is calibrated; another factor is the temperature to recover.
    The totals vary (``2 + Poisson(17)``, this data's), so that a weight per
    child differs from one per building.
    """
    rng = np.random.default_rng(seed)
    shares = rng.dirichlet(np.full(len(COHORTS), 2.0), size=rows)
    totals = 2 + rng.poisson(17, size=rows)
    counts = np.vstack([rng.multinomial(n, row) for n, row in zip(totals, shares)])
    index = pd.RangeIndex(100, 100 + rows)
    return (
        pd.DataFrame(factor * np.log(shares), columns=COHORTS, index=index),
        pd.DataFrame(counts, columns=COHORTS, index=index),
    )


LOGITS, COUNTS = _data(factor=2.5)


def test_the_fit_equals_scikit_learns_temperature_scaling_per_building() -> None:
    # Another objective, the inverse temperature stored as T, or a weight per
    # child instead of per building still gives a plausible T. scikit-learn's
    # class fits one row per (building, cohort); weighted by the building's
    # observed share, every building counts once.
    n_cohorts = len(COHORTS)
    shares = COUNTS.to_numpy() / COUNTS.to_numpy().sum(axis=1, keepdims=True)
    oracle = _TemperatureScaling().fit(
        np.repeat(LOGITS.to_numpy(), n_cohorts, axis=0),
        np.tile(np.arange(n_cohorts), len(LOGITS)),
        sample_weight=shares.ravel(),
    )

    calibrator = TemperatureCalibrator().fit(LOGITS, COUNTS)

    assert calibrator.temperature_ == pytest.approx(1 / oracle.beta_, abs=1e-7)
    np.testing.assert_allclose(
        calibrator.predict(LOGITS).to_numpy(),
        oracle.predict(LOGITS.to_numpy()),
        atol=1e-6,
    )


def test_logits_scaled_by_a_known_factor_recover_it() -> None:
    # The scaling applied the wrong way round, or a search that never leaves
    # T = 1, misses the factor the logits were sharpened by.
    calibrator = TemperatureCalibrator().fit(LOGITS, COUNTS)

    assert calibrator.temperature_ == pytest.approx(2.5, abs=0.1)


def test_calibrated_logits_give_a_temperature_near_one() -> None:
    # An objective off by a normalization would move T away from 1 on logits
    # that are the log of the probabilities the counts were drawn from.
    logits, counts = _data(factor=1.0)

    calibrator = TemperatureCalibrator().fit(logits, counts)

    assert calibrator.temperature_ == pytest.approx(1.0, abs=0.05)


@pytest.mark.filterwarnings("error")
def test_extreme_logits_are_fitted_in_log_space() -> None:
    # A probability taken before the log underflows to 0 where the count is
    # positive: the loss is inf, and scipy's bounded search then multiplies
    # inf − inf (a warning, here an error) instead of fitting.
    logits = np.array([[800.0, 0.0, 0.0], [0.0, 800.0, 0.0]])
    counts = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])

    calibrator = TemperatureCalibrator().fit(logits, counts)

    assert np.isfinite(calibrator.temperature_)


def test_uninformative_logits_flatten_instead_of_failing() -> None:
    # Logits unrelated to the counts have no finite best T: the search must
    # end at the bound of the decided range, log(1/T) in (−10, 10), where the
    # probabilities are near uniform, not raise or stop at a sharp fit.
    rng = np.random.default_rng(1)
    noise = 3 * rng.normal(size=COUNTS.shape)

    calibrator = TemperatureCalibrator().fit(noise, COUNTS)

    assert calibrator.temperature_ == pytest.approx(np.exp(10), rel=1e-3)
    assert np.abs(calibrator.predict(noise).to_numpy() - 1 / len(COHORTS)).max() < 1e-3


def test_a_temperature_of_one_is_the_plain_softmax() -> None:
    # The temperature multiplied, or applied twice, would change the model's
    # own probabilities even when the fit found nothing to correct.
    calibrator = TemperatureCalibrator()
    calibrator.temperature_ = 1.0

    np.testing.assert_allclose(
        calibrator.predict(LOGITS).to_numpy(),
        softmax(LOGITS.to_numpy(), axis=1),
        rtol=1e-12,
    )


def test_probabilities_keep_the_logits_columns_and_index_and_sum_to_one() -> None:
    # Misnamed or misaligned probabilities would be multiplied with the wrong
    # cohort or building by Model 2.
    new_logits = LOGITS.iloc[:5].set_index(pd.Index([10, 20, 30, 40, 50]))

    probabilities = TemperatureCalibrator().fit(LOGITS, COUNTS).predict(new_logits)

    assert list(probabilities.columns) == COHORTS
    assert probabilities.index.equals(new_logits.index)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-12)


def test_rows_are_paired_by_position_not_by_label() -> None:
    # Two DataFrames with different indexes, multiplied as frames, would be
    # aligned by label into NaN or zeros; arrays and frames must agree.
    counts_relabelled = COUNTS.set_index(pd.RangeIndex(len(COUNTS)))

    from_frames = TemperatureCalibrator().fit(LOGITS, counts_relabelled)
    from_arrays = TemperatureCalibrator().fit(LOGITS.to_numpy(), COUNTS.to_numpy())

    assert from_frames.temperature_ == from_arrays.temperature_


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (COUNTS.sum(axis=1).to_frame(), "same shape"),
        (COUNTS[COHORTS[::-1]], "same order"),
    ],
    ids=["one-column", "reordered-cohorts"],
)
def test_input_that_would_fit_silently_raises(
    counts: pd.DataFrame, message: str
) -> None:
    # numpy would broadcast one column over every cohort; cohorts in another
    # order would be scored against the wrong logits.
    with pytest.raises(ValueError, match=message):
        TemperatureCalibrator().fit(LOGITS, counts)


def test_predict_before_fit_raises() -> None:
    # Without the check, predict would raise an AttributeError from inside
    # numpy, not the scikit-learn error callers test for.
    with pytest.raises(NotFittedError):
        TemperatureCalibrator().predict(LOGITS)


def test_a_failed_search_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # A search that stopped early would hand over an arbitrary temperature
    # silently.
    from age_group_prediction.modeling import calibration

    scipy_minimize_scalar = calibration.minimize_scalar

    def failing_minimize_scalar(*args: object, **kwargs: Any) -> object:
        result = scipy_minimize_scalar(*args, **kwargs)
        result.success = False
        return result

    monkeypatch.setattr(calibration, "minimize_scalar", failing_minimize_scalar)
    with pytest.raises(RuntimeError, match="temperature search failed"):
        TemperatureCalibrator().fit(LOGITS, COUNTS)


def test_nan_logits_raise() -> None:
    # scipy reports a nan objective as a failed search; used silently it is
    # an arbitrary temperature.
    logits = LOGITS.to_numpy().copy()
    logits[0, 0] = np.nan

    with pytest.raises(RuntimeError, match="temperature search failed"):
        TemperatureCalibrator().fit(logits, COUNTS)


@pytest.mark.filterwarnings("error")
def test_the_out_of_fold_loop_runs_as_documented() -> None:
    # The documented flow (MULTI_COHORT_MODELS_PLAN.md §7): a model's fold
    # copies give out-of-fold logits, the calibrator is fitted on them, and
    # then applied to the full fit's logits.
    rng = np.random.default_rng(3)
    rows = 300
    table = pd.DataFrame({"ses": rng.normal(size=rows), "size": rng.normal(size=rows)})
    concentration = np.exp(
        np.array([1.0, 1.5, 0.5])
        + table.to_numpy() @ rng.normal(scale=0.5, size=(2, len(COHORTS)))
    )
    shares = np.vstack([rng.dirichlet(row) for row in concentration])
    totals = 2 + rng.poisson(17, size=rows)
    counts = pd.DataFrame(
        np.vstack([rng.multinomial(n, s) for n, s in zip(totals, shares)]),
        columns=COHORTS,
    )
    model = CohortProbabilityModel(
        feature_transformer=FeatureTransformer(
            tuple(
                ColumnPlan(name=c, columns=(c,), transforms=(Center(),))
                for c in ("ses", "size")
            )
        )
    )

    logits_val, counts_val = [], []
    for fit_index, val_index in KFold(5, shuffle=True, random_state=0).split(table):
        fold = clone(model).fit(table.iloc[fit_index], counts.iloc[fit_index])
        logits_val.append(fold.predict_logits(table.iloc[val_index]))
        counts_val.append(counts.iloc[val_index])
    calibrator = TemperatureCalibrator().fit(
        pd.concat(logits_val), pd.concat(counts_val)
    )
    fitted = clone(model).fit(table, counts)
    probabilities = calibrator.predict(fitted.predict_logits(table))

    assert np.isfinite(calibrator.temperature_)
    assert list(probabilities.columns) == COHORTS
    assert probabilities.index.equals(table.index)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-12)
