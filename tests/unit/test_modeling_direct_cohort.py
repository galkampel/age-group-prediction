"""DirectCohortModel: each test names the mistake in our code it would catch."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from lightgbm.basic import LightGBMError
from sklearn.base import clone

from age_group_prediction.modeling import DirectCohortModel, Objective


def _data(rows: int = 300) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Counts proportional to building size, with a rate that depends on ``ses``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"ses": rng.normal(size=rows), "noise": rng.normal(size=rows)})
    n = rng.integers(12, 80, size=rows).astype(float)
    y = pd.Series(rng.poisson(n * np.exp(-2 + 0.4 * X["ses"])))
    return X, y, n


X, Y, N = _data()


def test_clone_and_set_params_change_only_the_copy() -> None:
    # A tuner builds each trial's model this way; it breaks if __init__ alters
    # or drops an argument.
    model = DirectCohortModel(use_exposure=True, n_estimators=50)

    trial = clone(model).set_params(n_estimators=5)

    assert trial.get_params()["n_estimators"] == 5
    assert trial.get_params()["use_exposure"] is True
    assert model.get_params()["n_estimators"] == 50


@pytest.mark.parametrize(
    ("use_exposure", "exposure"),
    [
        (True, None),
        (False, N),
        (True, np.r_[0.0, N[1:]]),
        (True, np.r_[-1.0, N[1:]]),
        (True, np.r_[np.inf, N[1:]]),
    ],
    ids=["missing-while-on", "given-while-off", "zero", "negative", "infinite"],
)
def test_exposure_misuse_raises(
    use_exposure: bool, exposure: np.ndarray | None
) -> None:
    # Each would otherwise pass silently: a dropped or ignored offset, or a
    # -inf/nan init_score that LightGBM accepts.
    with pytest.raises(ValueError):
        DirectCohortModel(use_exposure=use_exposure).fit(X, Y, exposure=exposure)


@pytest.mark.parametrize(
    ("objective", "use_exposure"),
    [("poisson", True), ("poisson", False), ("regression", False)],
)
def test_training_mean_prediction_matches_the_target_mean(
    objective: Objective, use_exposure: bool
) -> None:
    # Few trees, so a missing starting rate (base_log_rate_) or a broken offset
    # leaves the mean far off: 21.75 against 6.68 with 20 trees. The two cases
    # without exposure are the only cover of the plain predict path.
    exposure = N if use_exposure else None
    model = DirectCohortModel(
        objective=objective, use_exposure=use_exposure, n_estimators=20
    ).fit(X, Y, exposure=exposure)

    predictions = model.predict(X, exposure=exposure)

    assert predictions.mean() == pytest.approx(Y.mean(), rel=0.01)


def test_predict_follows_how_the_model_was_fitted() -> None:
    # Turning use_exposure off after fitting must not silently return rates per
    # apartment instead of counts.
    model = DirectCohortModel(use_exposure=True, n_estimators=5).fit(X, Y, exposure=N)
    model.set_params(use_exposure=False)

    with pytest.raises(ValueError):
        model.predict(X)
    assert model.predict(X, exposure=N).mean() == pytest.approx(Y.mean(), rel=0.05)


def test_doubling_the_exposure_doubles_the_prediction() -> None:
    # Exact because the exposure is not a feature; fails if predict drops the
    # offset.
    model = DirectCohortModel(use_exposure=True, n_estimators=20).fit(X, Y, exposure=N)

    np.testing.assert_allclose(
        model.predict(X, exposure=2 * N), 2 * model.predict(X, exposure=N), rtol=1e-12
    )


def test_subsample_below_one_changes_the_model() -> None:
    # LightGBM ignores subsample unless subsample_freq is set.
    full = DirectCohortModel(n_estimators=20).fit(X, Y).predict(X)
    bagged = DirectCohortModel(n_estimators=20, subsample=0.5).fit(X, Y).predict(X)

    assert not np.allclose(full, bagged)


@pytest.mark.parametrize(
    "fit",
    [
        lambda: DirectCohortModel(objective="regression", use_exposure=True).fit(
            X, Y, exposure=N
        ),
        lambda: DirectCohortModel(objective="not_an_objective").fit(X, Y),  # type: ignore[arg-type]
        lambda: DirectCohortModel(use_exposure=True).fit(X, Y, exposure=N[:10]),
        lambda: DirectCohortModel(use_exposure=True).fit(X, Y * 0, exposure=N),
    ],
    ids=["regression-with-exposure", "unknown-objective", "wrong-length", "all-zero-y"],
)
# log(sum y / sum n) = log 0 warns before LightGBM rejects the all-zero y.
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_invalid_input_surfaces_an_error(fit: Callable[[], DirectCohortModel]) -> None:
    # Only the first is our own check; the rest are LightGBM's, pinned here
    # because the model deliberately relies on them.
    with pytest.raises((ValueError, LightGBMError)):
        fit()
