"""Tests for ``CVHyperparameterEvaluator``."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from typing import Any, ClassVar, Self

import numpy as np
import optuna
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from optuna.pruners import BasePruner
from optuna.trial import FixedTrial
from pydantic import ValidationError
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
    Aggregation,
    CVHyperparameterEvaluator,
    FloatParameter,
    IntParameter,
    LowerBound,
    Mean,
    WeightedMean,
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
EXPOSURE = np.arange(1.0, 29.0)  # distinct per row

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


class _Median(Aggregation):
    def aggregate(self, scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
        return float(np.median(scores))


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
        "feature_transformer": FeatureTransformer(remainder="passthrough"),
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
    ("overrides", "field"),
    [
        ({"parameters": []}, "parameters"),
        ({"parameters": set(PARAMETERS)}, "parameters"),
        ({"model": StandardScaler()}, "model"),
        ({"metric": "neg_mean_squared_error"}, "metric"),  # the old sklearn API
        ({"metric": {"name": "mse", "function": mean_squared_error}}, "metric"),
        ({"cv": 5}, "cv"),
        ({"feature_transformer": StandardScaler()}, "feature_transformer"),
        ({"aggregation": "mean"}, "aggregation"),  # the old string API
    ],
    ids=[
        "no-parameters",
        "parameter-set",
        "model",
        "metric-string",
        "metric-dict",
        "cv",
        "feature-transformer",
        "aggregation-string",
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError) as error:
        _evaluator(**overrides)
    # Positional arguments are named by position: model 0, parameters 1.
    location = {"model": 0, "parameters": 1}.get(field, field)
    assert error.value.errors()[0]["loc"][0] == location


def test_duplicate_parameter_names_are_named() -> None:
    parameters = [FloatParameter("alpha", 0.1, 1.0), IntParameter("alpha", 1, 9)]
    with pytest.raises(ValidationError, match=r"unique, repeated: \['alpha'\]"):
        _evaluator(parameters=parameters)


def test_settings_are_stored_as_given_and_frozen() -> None:
    model = _RecordingModel()
    transformer = FeatureTransformer(remainder="passthrough")
    evaluator = _evaluator(
        parameters=list(PARAMETERS), model=model, feature_transformer=transformer
    )

    assert evaluator.parameters == tuple(PARAMETERS)
    # Not copied here: the templates are cloned per trial.
    assert evaluator.model is model
    assert evaluator.feature_transformer is transformer
    assert evaluator.metric is MSE
    assert evaluator.aggregation == WeightedMean()  # the default
    with pytest.raises(dataclasses.FrozenInstanceError):
        evaluator.metric = MSE  # type: ignore[misc]


def test_build_feature_transformer_and_model_returns_unfitted_copies() -> None:
    transformer_template = FeatureTransformer(remainder="passthrough")
    model_template = _RecordingModel(alpha=1.0)
    evaluator = _evaluator(
        feature_transformer=transformer_template, model=model_template
    )

    transformer, model = evaluator.build_feature_transformer_and_model({"alpha": 3.0})

    assert transformer is not transformer_template
    assert model is not model_template
    assert model.get_params()["alpha"] == 3.0
    assert model_template.get_params()["alpha"] == 1.0
    for built in (transformer, model):
        with pytest.raises(NotFittedError):
            check_is_fitted(built)


def test_build_feature_transformer_and_model_rejects_an_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Invalid parameter 'beta'"):
        _evaluator().build_feature_transformer_and_model({"beta": 1.0})


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


def test_the_transformer_is_fitted_per_fold_on_training_rows_only() -> None:
    _evaluator(feature_transformer=_RecordingTransformer()).evaluate(
        FixedTrial({"alpha": 1.0}), X, Y, GROUPS
    )

    folds = _folds()
    # One fit per fold, on its training rows and their targets only.
    assert _RecordingTransformer.fits == [
        (sorted(t.tolist()), Y[t].tolist()) for t, _ in folds
    ]
    # Each fold's validation rows were transformed by that fold's own fit.
    assert [seen for _, seen in _RecordingModel.predicts] == [[0], [1], [2]]


def test_the_transformer_template_is_left_unfitted() -> None:
    # Each fold fits a clone: a shared, refitted template would carry one
    # trial's statistics into another's when trials run in parallel.
    evaluator = _evaluator(
        feature_transformer=FeatureTransformer(remainder="passthrough")
    )

    evaluator.evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS)

    with pytest.raises(NotFittedError):
        check_is_fitted(evaluator.feature_transformer)


def test_exposure_is_sliced_to_each_fold() -> None:
    # A shuffled index, so slicing by label would pick the wrong rows.
    exposure = pd.Series(
        EXPOSURE, index=np.random.default_rng(0).permutation(len(EXPOSURE))
    )

    _evaluator().evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS, exposure=exposure)

    folds = _folds()
    assert [e for _, _, e in _RecordingModel.fits] == [
        EXPOSURE[t].tolist() for t, _ in folds
    ]
    assert [e for e, _ in _RecordingModel.predicts] == [
        EXPOSURE[v].tolist() for _, v in folds
    ]


def test_no_exposure_arrives_as_none() -> None:
    _evaluator().evaluate(FixedTrial({"alpha": 1.0}), X, Y, GROUPS)

    assert [e for _, _, e in _RecordingModel.fits] == [None] * 3
    assert [e for e, _ in _RecordingModel.predicts] == [None] * 3


@pytest.mark.parametrize(
    "exposure",
    [np.concatenate([EXPOSURE, EXPOSURE]), EXPOSURE[:-1], EXPOSURE[:, None]],
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


def _fold_scores() -> tuple[list[float], list[int]]:
    """Each fold's score under _mean_of_y (the validation rows' mean y), and size."""
    folds = _folds()
    return [float(np.mean(Y[v])) for _, v in folds], [len(v) for _, v in folds]


@pytest.mark.parametrize(
    ("aggregation", "rule"),
    [
        (WeightedMean(), lambda scores, sizes: np.average(scores, weights=sizes)),
        (Mean(), lambda scores, sizes: np.mean(scores)),
        (
            LowerBound(z=2),
            # 3 folds partition the rows: the K-fold corrected SE.
            lambda scores, sizes: (
                np.average(scores, weights=sizes)
                - 2 * np.std(scores, ddof=1) * math.sqrt(1 / 3 + 1 / 2)
            ),
        ),
        (_Median(), lambda scores, sizes: np.median(scores)),
    ],
    ids=["weighted-mean", "mean", "lower-bound", "median"],
)
def test_each_fold_reports_its_score_and_the_value_is_the_aggregation(
    aggregation: Aggregation, rule: Any
) -> None:
    study = _optimize(
        _evaluator(metric=_mean_of_y(), aggregation=aggregation), n_trials=1
    )

    scores, sizes = _fold_scores()
    assert len(set(sizes)) > 1  # the folds really differ in size
    reported = study.trials[0].intermediate_values
    assert list(reported) == [0, 1, 2]
    assert list(reported.values()) == pytest.approx(scores)
    assert study.trials[0].value == pytest.approx(rule(scores, sizes))


def test_a_completed_trial_records_its_folds() -> None:
    evaluator = _evaluator(metric=_mean_of_y())
    study = _optimize(evaluator, n_trials=1)
    fixed = FixedTrial({"alpha": 1.0})
    evaluator.evaluate(fixed, X, Y, GROUPS)

    scores, sizes = _fold_scores()
    for attrs in (study.trials[0].user_attrs, fixed.user_attrs):
        assert set(attrs) == {"fold_scores", "fold_sizes"}
        assert attrs["fold_scores"] == pytest.approx(scores)
        assert attrs["fold_sizes"] == sizes
        # JSON-native, as sqlite storage requires.
        assert type(attrs["fold_scores"]) is list
        assert {type(v) for v in attrs["fold_scores"]} == {float}
        assert {type(v) for v in attrs["fold_sizes"]} == {int}
        json.dumps(attrs)


def test_a_pruned_trial_stops_after_the_first_fold() -> None:
    study = _optimize(_evaluator(), n_trials=1, pruner=_AlwaysPrune())

    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert len(_RecordingModel.fits) == 1
    assert study.trials[0].user_attrs == {}  # only a completed trial records folds


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
        feature_transformer=tree,
    )

    value = evaluator.evaluate(
        FixedTrial({"n_estimators": 10}), df, y, groups, exposure=df["n"]
    )

    exposure = df["n"].to_numpy()
    scores: list[float] = []
    sizes: list[int] = []
    for train, val in cv.split(df, y, groups):
        transformer = clone(tree).fit(df.iloc[train], y.iloc[train])
        fitted = clone(model).set_params(n_estimators=10)
        fitted.fit(
            transformer.transform(df.iloc[train]),
            y.iloc[train],
            exposure=exposure[train],
        )
        y_val_pred = fitted.predict(
            transformer.transform(df.iloc[val]), exposure=exposure[val]
        )
        scores.append(-float(POISSON_DEVIANCE.function(y.iloc[val], y_val_pred)))
        sizes.append(len(val))
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
