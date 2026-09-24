"""Tests for ``CVHyperparameterEvaluator``."""

from __future__ import annotations

import math
from typing import Any, ClassVar, Self

import numpy as np
import optuna
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from optuna.pruners import BasePruner
from optuna.trial import FixedTrial
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from age_group_prediction.feature_engineering.transformer import (
    ColumnPlan,
    FeatureTransformer,
)
from age_group_prediction.feature_engineering.transforms import Standardize
from age_group_prediction.hyperparameter_tuning import (
    CVHyperparameterEvaluator,
    FloatParameter,
    IntParameter,
)
from age_group_prediction.modeling import BaseAgeGroupModel, DirectCohortModel
from age_group_prediction.scoring import POISSON_DEVIANCE, Metric
from age_group_prediction.splitting import Splitter

PARAMETERS = [FloatParameter("alpha", 0.1, 10.0, log=True)]

# Column "id" is the row id, so a recording step can tell which rows it saw.
# Strata of 3, 5, 8 and 12 rows give folds of unequal size.
GROUPS = np.repeat(np.arange(4), [3, 5, 8, 12])
X = pd.DataFrame({"id": np.arange(28.0), "x": np.linspace(-1.0, 1.0, 28)})
Y = np.arange(28.0) ** 1.5
N = np.arange(1.0, 29.0)  # an exposure, distinct per row

MSE = Metric("mse", mean_squared_error)


class _RecordingModel(BaseAgeGroupModel):
    """Records what fit and predict receive; predicts the training mean of y."""

    fits: ClassVar[list[tuple[float, list[int], list[float] | None]]] = []
    predicts: ClassVar[list[tuple[list[float] | None, list[int]]]] = []

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any, exposure: ArrayLike | None = None) -> Self:
        _RecordingModel.fits.append((self.alpha, _ids(X), _listed(exposure)))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any, exposure: ArrayLike | None = None) -> np.ndarray:
        # The fitted_by column, if a _RecordingTransformer added it, tells
        # which fit transformed these rows.
        seen = (
            sorted(set(X["fitted_by"]))
            if "fitted_by" in getattr(X, "columns", ())
            else []
        )
        _RecordingModel.predicts.append((_listed(exposure), seen))
        return np.full(len(X), self.mean_)


class _RecordingTransformer(FeatureTransformer):
    # Replaces fit and transform wholesale; only these two are called.
    """Records the row ids and targets at each fit; stamps rows with that fit's number."""

    fits: ClassVar[list[tuple[list[int], list[float]]]] = []

    def fit(self, X: Any, y: Any = None) -> _RecordingTransformer:
        self.fit_number_ = len(_RecordingTransformer.fits)
        _RecordingTransformer.fits.append((_ids(X), np.asarray(y).tolist()))
        return self

    def transform(self, X: Any) -> Any:
        return X.assign(fitted_by=self.fit_number_)


class _AlwaysPrune(BasePruner):
    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return True


def _ids(X: Any) -> list[int]:
    return sorted(np.asarray(X)[:, 0].astype(int).tolist())


def _listed(exposure: ArrayLike | None) -> list[float] | None:
    return None if exposure is None else np.asarray(exposure).tolist()


def _mean_of_y(*, greater_is_better: bool = True) -> Metric:
    # The fold score is the mean target of the validation rows, which a test
    # can compute from the folds alone.
    return Metric(
        "mean_of_y",
        lambda y_true, y_pred: float(np.mean(np.asarray(y_true, dtype=float))),
        greater_is_better=greater_is_better,
    )


@pytest.fixture(autouse=True)
def _reset_recordings() -> None:
    _RecordingModel.fits.clear()
    _RecordingModel.predicts.clear()
    _RecordingTransformer.fits.clear()


def _cv() -> Any:
    return Splitter("stratified_by_group").cv(n_splits=3, random_state=0)


def _folds() -> list[tuple[np.ndarray, np.ndarray]]:
    # Identical to the evaluator's folds: Splitter.cv is seeded (D3).
    return list(_cv().split(X, Y, GROUPS))


def _evaluator(**overrides: Any) -> CVHyperparameterEvaluator:
    kwargs: dict[str, Any] = {
        "cv": _cv(),
        "metric": MSE,
        # Passes the columns through: most tests need no transformation.
        "features": FeatureTransformer(remainder="passthrough"),
    } | overrides
    model = kwargs.pop("model", _RecordingModel())
    parameters = kwargs.pop("parameters", PARAMETERS)
    return CVHyperparameterEvaluator(model, parameters, **kwargs)


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


def test_metric_must_be_a_metric() -> None:
    # The old sklearn API took scoring strings.
    with pytest.raises(TypeError, match="metric must be a Metric"):
        _evaluator(metric="neg_mean_squared_error")


def test_features_must_be_a_feature_transformer() -> None:
    with pytest.raises(TypeError, match="features must be a FeatureTransformer"):
        _evaluator(features=StandardScaler())


def test_build_features_and_model_returns_unfitted_copies() -> None:
    features_template = FeatureTransformer(remainder="passthrough")
    model_template = _RecordingModel(alpha=1.0)
    evaluator = _evaluator(features=features_template, model=model_template)

    features, model = evaluator.build_features_and_model({"alpha": 3.0})

    assert features is not features_template
    assert model is not model_template
    assert model.get_params()["alpha"] == 3.0
    assert model_template.get_params()["alpha"] == 1.0
    for built in (features, model):
        with pytest.raises(NotFittedError):
            check_is_fitted(built)


def test_build_features_and_model_rejects_an_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Invalid parameter 'beta'"):
        _evaluator().build_features_and_model({"beta": 1.0})


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
    fitted = [alpha for alpha, _, _ in _RecordingModel.fits]
    expected = [t.params["alpha"] for t in study.trials for _ in range(3)]
    assert fitted == expected


def test_every_trial_sees_identical_folds() -> None:
    # D3 end to end: each trial re-splits, and the seeded validator repeats.
    _optimize(_evaluator(), n_trials=3)

    rows = [ids for _, ids, _ in _RecordingModel.fits]
    assert len(rows) == 9
    assert rows[0:3] == rows[3:6] == rows[6:9]
    assert rows[0:3] == [sorted(train.tolist()) for train, _ in _folds()]


def test_features_are_fitted_per_fold_on_training_rows_only() -> None:
    _evaluator(features=_RecordingTransformer()).evaluate(
        FixedTrial({"alpha": 1.0}), X, Y, GROUPS
    )

    folds = _folds()
    # One fit per fold, on its training rows and their targets only.
    assert _RecordingTransformer.fits == [
        (sorted(t.tolist()), Y[t].tolist()) for t, _ in folds
    ]
    # Each fold's validation rows were transformed by that fold's own fit.
    assert [seen for _, seen in _RecordingModel.predicts] == [[0], [1], [2]]


def test_the_features_template_is_left_unfitted() -> None:
    # Each fold fits a clone: a shared, refitted template would carry one
    # trial's statistics into another's when trials run in parallel.
    evaluator = _evaluator(features=FeatureTransformer(remainder="passthrough"))

    evaluator.evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS)

    with pytest.raises(NotFittedError):
        check_is_fitted(evaluator.features)


def test_exposure_is_sliced_to_each_fold() -> None:
    # A shuffled index, so slicing by label would pick the wrong rows.
    exposure = pd.Series(N, index=np.random.default_rng(0).permutation(len(N)))

    _evaluator().evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS, exposure=exposure)

    folds = _folds()
    assert [e for _, _, e in _RecordingModel.fits] == [N[t].tolist() for t, _ in folds]
    assert [e for e, _ in _RecordingModel.predicts] == [N[v].tolist() for _, v in folds]


def test_no_exposure_arrives_as_none() -> None:
    _evaluator().evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS)

    assert [e for _, _, e in _RecordingModel.fits] == [None] * 3
    assert [e for e, _ in _RecordingModel.predicts] == [None] * 3


@pytest.mark.parametrize(
    "exposure",
    [np.concatenate([N, N]), N[:-1], N[:, None]],
    ids=["long", "short", "2-d"],
)
def test_an_exposure_not_one_per_row_is_rejected(exposure: np.ndarray) -> None:
    # Each fold would slice a longer exposure to the right size, silently.
    with pytest.raises(ValueError, match="one value per row of X"):
        _evaluator().evaluate(
            FixedTrial({"alpha": 1.0}), X, Y, GROUPS, exposure=exposure
        )


def test_a_missing_exposure_raises_the_models_error() -> None:
    evaluator = _evaluator(
        model=DirectCohortModel(use_exposure=True),
        parameters=[IntParameter("n_estimators", 1, 5)],
    )

    with pytest.raises(ValueError, match="pass `exposure` exactly when"):
        evaluator.evaluate(FixedTrial({"n_estimators": 2}), X, Y, GROUPS)


@pytest.mark.parametrize("greater_is_better", [True, False])
def test_the_score_is_signed_by_the_metrics_direction(greater_is_better: bool) -> None:
    metric = _mean_of_y(greater_is_better=greater_is_better)

    value = _evaluator(metric=metric).evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS)

    folds = _folds()
    pooled = float(
        np.average(
            [np.mean(Y[v]) for _, v in folds], weights=[len(v) for _, v in folds]
        )
    )
    assert pooled > 0
    assert value == pytest.approx(pooled if greater_is_better else -pooled)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_score_raises_and_skips_the_remaining_folds(bad: float) -> None:
    evaluator = _evaluator(metric=Metric("bad", lambda y_true, y_pred: bad))

    with pytest.raises(ValueError, match=r"trial 0, fold 0: score is"):
        evaluator.evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS)
    assert len(_RecordingModel.fits) == 1


@pytest.mark.parametrize("aggregation", ["weighted_mean", "mean"])
def test_each_fold_reports_the_running_mean(aggregation: str) -> None:
    study = _optimize(
        _evaluator(metric=_mean_of_y(), aggregation=aggregation), n_trials=1
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
    assert len(_RecordingModel.fits) == 1


def test_pandas_and_numpy_inputs_score_the_same() -> None:
    evaluator = _evaluator()
    trial = FixedTrial({"alpha": 1.0})

    # A reversed index, so label-based selection would pick the wrong rows.
    index = np.arange(len(Y))[::-1]
    from_pandas = evaluator.evaluate(
        trial,
        X.set_axis(index),
        pd.Series(Y, index=index),
        pd.Series(GROUPS, index=index),
    )
    rows_from_pandas = [ids for _, ids, _ in _RecordingModel.fits]
    _RecordingModel.fits.clear()
    from_numpy = evaluator.evaluate(trial, X.to_numpy(), Y, GROUPS)

    assert from_pandas == pytest.approx(from_numpy)
    assert rows_from_pandas == [ids for _, ids, _ in _RecordingModel.fits]


def test_matches_a_hand_written_fold_loop() -> None:
    # The real pieces, end to end: fold-local features, the exposure offset,
    # and a lower-is-better metric that is negated. The loop below builds
    # fresh objects per fold, so this also catches any state the evaluator's
    # one-copy-per-trial refits might carry from one fold to the next.
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), [5, 10, 20, 30, 45, 10])  # folds of 30, 40, 50
    rows = len(groups)
    df = pd.DataFrame(
        {"ses": rng.normal(size=rows), "n": rng.integers(12, 80, rows).astype(float)},
        index=np.arange(rows)[::-1],  # labels unlike positions
    )
    y = pd.Series(rng.poisson(df["n"] * np.exp(-2 + 0.4 * df["ses"])), index=df.index)
    tree = FeatureTransformer(
        plans=(
            ColumnPlan(name="num", columns=("ses", "n"), transforms=(Standardize(),)),
        )
    )
    cv = Splitter("grouped").cv(n_splits=3, random_state=0)
    model = DirectCohortModel(use_exposure=True)
    evaluator = CVHyperparameterEvaluator(
        model,
        [IntParameter("n_estimators", 5, 20)],
        cv=cv,
        metric=POISSON_DEVIANCE,
        features=tree,
    )

    value = evaluator.evaluate(
        FixedTrial({"n_estimators": 10}), df, y, groups, exposure=df["n"]
    )

    n = df["n"].to_numpy()
    scores: list[float] = []
    sizes: list[int] = []
    for train, validation in cv.split(df, y, groups):
        features = clone(tree).fit(df.iloc[train], y.iloc[train])
        fitted = clone(model).set_params(n_estimators=10)
        fitted.fit(features.transform(df.iloc[train]), y.iloc[train], exposure=n[train])
        predicted = fitted.predict(
            features.transform(df.iloc[validation]), exposure=n[validation]
        )
        scores.append(-float(POISSON_DEVIANCE.function(y.iloc[validation], predicted)))
        sizes.append(len(validation))
    assert len(set(sizes)) > 1
    assert value == pytest.approx(np.average(scores, weights=sizes))


def test_one_evaluator_scores_several_datasets() -> None:
    # Settings in the constructor, data per call: e.g. one evaluator for every
    # cohort's target, each result equal to a freshly built evaluator's.
    shared = _evaluator()
    trial = FixedTrial({"alpha": 1.0})

    for y in (Y, np.sqrt(Y)):
        fresh = _evaluator().evaluate(trial, X, y, GROUPS)
        assert shared.evaluate(trial, X, y, GROUPS) == fresh
