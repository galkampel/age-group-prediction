"""Tests for ``CVHyperparameterEvaluator``."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.pruners import BasePruner
from optuna.trial import FixedTrial
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.hyperparameter_tuning import (
    CVHyperparameterEvaluator,
    FloatParameter,
    IntParameter,
)
from age_group_prediction.splitting import Splitter

PARAMETERS = [FloatParameter("alpha", 0.1, 10.0, log=True)]

# Column 0 is the row id, so a recording step can tell which rows it saw.
# Strata of 3, 5, 8 and 12 rows give folds of unequal size.
GROUPS = np.repeat(np.arange(4), [3, 5, 8, 12])
X = np.column_stack([np.arange(28.0), np.linspace(-1.0, 1.0, 28)])
Y = np.arange(28.0) ** 1.5


class _RecordingRegressor(RegressorMixin, BaseEstimator):
    """Records ``alpha`` and the row ids at each fit; predicts zero."""

    fits: ClassVar[list[tuple[float, list[int]]]] = []

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> _RecordingRegressor:
        _RecordingRegressor.fits.append((self.alpha, _ids(X)))
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))


class _RecordingTransformer(TransformerMixin, BaseEstimator):
    """Records the row ids at each fit; passes rows through."""

    fits: ClassVar[list[list[int]]] = []

    def fit(self, X: Any, y: Any = None) -> _RecordingTransformer:
        _RecordingTransformer.fits.append(_ids(X))
        return self

    def transform(self, X: Any) -> Any:
        return X


class _AlwaysPrune(BasePruner):
    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return True


def _ids(X: Any) -> list[int]:
    return sorted(np.asarray(X)[:, 0].astype(int).tolist())


@pytest.fixture(autouse=True)
def _reset_recordings() -> None:
    _RecordingRegressor.fits.clear()
    _RecordingTransformer.fits.clear()


def _cv() -> Any:
    return Splitter("stratified_by_group").cv(n_splits=3, random_state=0)


def _folds() -> list[tuple[np.ndarray, np.ndarray]]:
    # Identical to the evaluator's folds: Splitter.cv is seeded (D3).
    return list(_cv().split(X, Y, GROUPS))


def _evaluator(**overrides: Any) -> CVHyperparameterEvaluator:
    kwargs: dict[str, Any] = {
        "cv": _cv(),
        "scoring": "neg_mean_squared_error",
    } | overrides
    estimator = kwargs.pop("estimator", _RecordingRegressor())
    parameters = kwargs.pop("parameters", PARAMETERS)
    return CVHyperparameterEvaluator(estimator, parameters, **kwargs)


def _optimize(
    evaluator: CVHyperparameterEvaluator,
    n_trials: int,
    pruner: BasePruner | None = None,
) -> optuna.Study:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner,
    )
    study.optimize(lambda trial: evaluator.evaluate(trial, X, Y, GROUPS), n_trials)
    return study


def _mean_of_y(estimator: Any, X: Any, y: Any) -> float:
    return float(np.mean(y))


# --- The constructor --------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parameters": []}, "parameters must not be empty"),
        (
            {
                "parameters": [
                    FloatParameter("alpha", 0.1, 1.0),
                    IntParameter("alpha", 1, 9),
                    FloatParameter("tol", 1e-4, 1e-2),
                ]
            },
            r"unique, repeated: \['alpha'\]",
        ),
        ({"aggregation": "median"}, "unknown aggregation 'median'"),
        ({"z": 2.0}, "z applies only to 'lower_bound'"),
        ({"aggregation": "mean", "z": 2.0}, "z applies only to 'lower_bound'"),
        ({"aggregation": "lower_bound", "z": 0.0}, "positive finite"),
        ({"aggregation": "lower_bound", "z": -1.0}, "positive finite"),
        ({"aggregation": "lower_bound", "z": float("nan")}, "positive finite"),
    ],
    ids=[
        "no-parameters",
        "duplicate-names",
        "unknown-aggregation",
        "z-with-weighted-mean",
        "z-with-mean",
        "z-zero",
        "z-negative",
        "z-nan",
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _evaluator(**overrides)


@pytest.mark.parametrize(("z", "expected"), [(None, 1.0), (2.0, 2.0)])
def test_lower_bound_z_defaults_to_one(z: float | None, expected: float) -> None:
    assert _evaluator(aggregation="lower_bound", z=z).z == expected


def test_scoring_accepts_a_callable() -> None:
    def scorer(estimator: Any, X: Any, y: Any) -> float:
        return 0.0

    assert _evaluator(scoring=scorer)._scorer is scorer


def test_build_estimator_returns_an_unfitted_clone() -> None:
    template = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    before = template.get_params(deep=False)["steps"]
    evaluator = _evaluator(
        estimator=template, parameters=[FloatParameter("model__alpha", 0.1, 10.0)]
    )

    built = evaluator.build_estimator({"model__alpha": 3.0})

    assert built is not template
    assert built.get_params()["model__alpha"] == 3.0
    assert template.get_params()["model__alpha"] == 1.0
    assert template.get_params(deep=False)["steps"] == before
    with pytest.raises(NotFittedError):
        check_is_fitted(built)


# --- evaluate ---------------------------------------------------------


def test_parameters_are_suggested_once_per_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Optuna caches repeated suggests, so equal values alone would not show it.
    calls: list[str] = []
    original = FloatParameter.suggest

    def counting(self: FloatParameter, trial: Any) -> float:
        calls.append(self.name)
        return original(self, trial)

    monkeypatch.setattr(FloatParameter, "suggest", counting)

    study = _optimize(_evaluator(), n_trials=2)

    assert calls == ["alpha", "alpha"]
    fitted = [alpha for alpha, _ in _RecordingRegressor.fits]
    expected = [t.params["alpha"] for t in study.trials for _ in range(3)]
    assert fitted == expected


def test_every_trial_sees_identical_folds() -> None:
    # D3 end to end: each trial re-splits, and the seeded validator repeats.
    _optimize(_evaluator(), n_trials=3)

    rows = [ids for _, ids in _RecordingRegressor.fits]
    assert len(rows) == 9
    assert rows[0:3] == rows[3:6] == rows[6:9]
    assert rows[0:3] == [sorted(train.tolist()) for train, _ in _folds()]


def test_pipeline_preprocessing_is_fitted_on_training_rows_only() -> None:
    pipeline = Pipeline(
        [("record", _RecordingTransformer()), ("model", _RecordingRegressor())]
    )
    evaluator = _evaluator(
        estimator=pipeline, parameters=[FloatParameter("model__alpha", 0.1, 10.0)]
    )

    evaluator.evaluate(FixedTrial({"model__alpha": 1.0}), X, Y, GROUPS)

    folds = _folds()
    assert _RecordingTransformer.fits == [sorted(t.tolist()) for t, _ in folds]
    for seen, (_, validation_index) in zip(
        _RecordingTransformer.fits, folds, strict=True
    ):
        assert not set(seen) & set(validation_index.tolist())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_score_raises_and_skips_the_remaining_folds(bad: float) -> None:
    evaluator = _evaluator(scoring=lambda estimator, X, y: bad)

    with pytest.raises(ValueError, match=r"trial 0, fold 0: score is"):
        evaluator.evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS)
    assert len(_RecordingRegressor.fits) == 1


@pytest.mark.parametrize("aggregation", ["weighted_mean", "mean"])
def test_each_fold_reports_the_running_mean(aggregation: str) -> None:
    study = _optimize(
        _evaluator(scoring=_mean_of_y, aggregation=aggregation), n_trials=1
    )

    folds = _folds()
    scores = [float(np.mean(Y[validation_index])) for _, validation_index in folds]
    sizes = [len(validation_index) for _, validation_index in folds]
    assert len(set(sizes)) > 1  # the folds really differ in size
    weights = sizes if aggregation == "weighted_mean" else [1] * len(sizes)
    expected = [
        float(np.average(scores[: k + 1], weights=weights[: k + 1]))
        for k in range(len(scores))
    ]
    reported = study.trials[0].intermediate_values
    assert list(reported) == [0, 1, 2]
    assert list(reported.values()) == pytest.approx(expected)
    assert study.trials[0].value == pytest.approx(expected[-1])


def test_a_pruned_trial_stops_after_the_first_fold() -> None:
    study = _optimize(_evaluator(), n_trials=1, pruner=_AlwaysPrune())

    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert len(_RecordingRegressor.fits) == 1


def test_pandas_and_numpy_inputs_score_the_same() -> None:
    evaluator = _evaluator(estimator=Ridge())
    trial = FixedTrial({"alpha": 1.0})

    from_numpy = evaluator.evaluate(trial, X, Y, GROUPS)
    from_pandas = evaluator.evaluate(
        trial,
        pd.DataFrame(X, columns=["id", "x"]),
        pd.Series(Y),
        pd.Series(GROUPS),
    )

    assert from_pandas == pytest.approx(from_numpy)


def test_a_fixed_trial_scores_one_parameter_set() -> None:
    value = _evaluator(estimator=Ridge()).evaluate(
        FixedTrial({"alpha": 2.0}), X, Y, GROUPS
    )

    scores, sizes = [], []
    for train_index, validation_index in _folds():
        model = Ridge(alpha=2.0).fit(X[train_index], Y[train_index])
        predicted = model.predict(X[validation_index])
        scores.append(-mean_squared_error(Y[validation_index], predicted))
        sizes.append(len(validation_index))
    assert value == pytest.approx(np.average(scores, weights=sizes))


def test_one_evaluator_scores_several_datasets() -> None:
    # Settings in the constructor, data per call: e.g. one evaluator for every
    # cohort's target, each result equal to a freshly built evaluator's.
    shared = _evaluator(estimator=Ridge())
    trial = FixedTrial({"alpha": 1.0})

    for y in (Y, np.sqrt(Y)):
        fresh = _evaluator(estimator=Ridge()).evaluate(trial, X, y, GROUPS)
        assert shared.evaluate(trial, X, y, GROUPS) == fresh
