"""Tests for ``BootstrapEvaluator``."""

from __future__ import annotations

from collections import Counter
from typing import Any, Self

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

from age_group_prediction.inspection import BootstrapEvaluator, BootstrapResult
from age_group_prediction.modeling import BaseAgeGroupModel, CohortProbabilityModel
from age_group_prediction.scoring import COHORT_LOG_LOSS, Metric

# y is the row id, and the spy model predicts it too, so a metric can tell
# which rows a replicate drew and whether y and the predictions stay paired.
# Groups of 3, 5, 8 and 12 rows with unsorted labels, interleaved as real
# neighborhoods are, so code that assumes contiguous groups fails.
GROUPS = np.random.default_rng(0).permutation(np.repeat([7, 2, 9, 4], [3, 5, 8, 12]))
X = pd.DataFrame({"id": np.arange(28.0)})
Y = np.arange(28.0)
EXPOSURE = np.arange(1.0, 29.0)


class _SpyModel(BaseAgeGroupModel):
    """Predicts each row's id; records every predict call's exposure."""

    def __init__(self) -> None:
        self.predict_exposures: list[ArrayLike | None] = []

    def fit(self, X: Any, y: Any, exposure: ArrayLike | None = None) -> Self:
        return self

    def predict(self, X: Any, exposure: ArrayLike | None = None) -> np.ndarray:
        self.predict_exposures.append(exposure)
        return np.asarray(X["id"], dtype=float)


class _RecordingMetric:
    """Scores the mean of ``y_true``; records each replicate's row ids and score."""

    def __init__(self) -> None:
        self.replicates: list[np.ndarray] = []
        self.scores: list[float] = []
        self.paired = True

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        ids = np.asarray(y_true)
        self.replicates.append(ids)
        self.paired &= bool(np.array_equal(ids, np.asarray(y_pred)))
        self.scores.append(float(ids.mean()))
        return self.scores[-1]


def _evaluate(
    recorder: _RecordingMetric,
    groups: np.ndarray | None = None,
    *,
    greater_is_better: bool = False,
    **settings: Any,
) -> BootstrapResult:
    metric = Metric("mean_id", recorder, greater_is_better=greater_is_better)
    kwargs: dict[str, Any] = {"metric": metric, "n_resamples": 50, "random_state": 0}
    evaluator = BootstrapEvaluator(**(kwargs | settings))
    return evaluator.evaluate(_SpyModel(), X, Y, groups)


def test_groups_are_drawn_whole() -> None:
    # Catches row resampling when groups are given: each replicate must hold
    # every row of a drawn group, as often as the group was drawn, and draw as
    # many groups as there are.
    recorder = _RecordingMetric()
    _evaluate(recorder, GROUPS)
    for ids in recorder.replicates:
        counts = Counter(ids.astype(int).tolist())
        draws = []
        for group in np.unique(GROUPS):
            rows = np.flatnonzero(GROUPS == group)
            row_draws = {counts[row] for row in rows}
            assert len(row_draws) == 1, f"group {group} split: {row_draws}"
            draws.append(row_draws.pop())
        assert sum(draws) == 4


def test_rows_are_drawn_with_replacement() -> None:
    # Catches a permutation (no replacement), which gives every replicate the
    # same score and a zero-width interval.
    recorder = _RecordingMetric()
    result = _evaluate(recorder)
    assert all(len(ids) == len(X) for ids in recorder.replicates)
    assert any(len(set(ids.tolist())) < len(X) for ids in recorder.replicates)
    assert result.lower < result.upper


def test_predictions_stay_paired_with_targets() -> None:
    # Catches y and the predictions being taken at different rows.
    recorder = _RecordingMetric()
    _evaluate(recorder, GROUPS)
    _evaluate(recorder)
    assert recorder.paired


def test_same_seed_gives_identical_results() -> None:
    # Catches an unseeded generator, or a seed that is ignored.
    first, second, other = _RecordingMetric(), _RecordingMetric(), _RecordingMetric()
    assert _evaluate(first, GROUPS) == _evaluate(second, GROUPS)
    assert _evaluate(other, GROUPS, random_state=1) != _evaluate(first, GROUPS)


def test_bounds_are_the_percentiles_of_the_replicates() -> None:
    # Catches a wrong quantile (e.g. alpha instead of alpha / 2) or a mean of
    # something other than the replicate scores.
    recorder = _RecordingMetric()
    result = _evaluate(recorder, alpha=0.2, n_resamples=200)
    scores = np.array(recorder.scores)
    assert len(scores) == 200
    assert result.mean == pytest.approx(scores.mean())
    assert result.lower == pytest.approx(np.quantile(scores, 0.1))
    assert result.upper == pytest.approx(np.quantile(scores, 0.9))
    assert result.lower <= result.mean <= result.upper


def test_predict_is_called_once_with_the_exposure() -> None:
    # Catches predicting per replicate, which is costly and, for a model with
    # an exposure, needs the exposure resampled with the rows.
    model = _SpyModel()
    evaluator = BootstrapEvaluator(Metric("mean_id", _RecordingMetric()), 20)
    evaluator.evaluate(model, X, Y, GROUPS, exposure=EXPOSURE)
    assert len(model.predict_exposures) == 1
    assert model.predict_exposures[0] is EXPOSURE


def test_greater_is_better_metric_is_reported_raw() -> None:
    # Catches the tuner's sign flip being applied here: the interval is in the
    # metric's own units.
    recorder = _RecordingMetric()
    result = _evaluate(recorder, greater_is_better=True)
    assert result.mean == pytest.approx(np.mean(recorder.scores))
    assert result.mean > 0


def test_non_finite_replicate_raises() -> None:
    # Catches nan bounds returned silently, e.g. a cohort log loss of one
    # replicate with no children (0 / 0), among finite ones.
    calls = iter(range(10))
    metric = Metric(
        "one_nan", lambda y_true, y_pred: float("nan") if next(calls) == 3 else 1.0
    )
    evaluator = BootstrapEvaluator(metric, 10)
    with pytest.raises(ValueError, match="1 of 10 replicate scores are not finite"):
        evaluator.evaluate(_SpyModel(), X, Y)


@pytest.mark.parametrize(
    ("y", "groups"),
    [(Y[:-1], None), (np.append(Y, 28.0), None), (Y, GROUPS[:-1])],
    ids=["short-y", "long-y", "short-groups"],
)
def test_length_mismatch_raises(y: np.ndarray, groups: np.ndarray | None) -> None:
    # Catches a y or groups of another length: a longer y would be silently
    # mispaired with the predictions, shorter groups would drop rows.
    evaluator = BootstrapEvaluator(Metric("mean_id", _RecordingMetric()), 10)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        evaluator.evaluate(_SpyModel(), X, y, groups)


def test_fewer_than_two_groups_raises() -> None:
    # Catches a single group passing silently: every replicate is then the
    # whole set and the interval has zero width.
    evaluator = BootstrapEvaluator(Metric("mean_id", _RecordingMetric()), 10)
    with pytest.raises(ValueError, match="1 distinct value"):
        evaluator.evaluate(_SpyModel(), X, Y, np.zeros(len(X), dtype=int))


@pytest.mark.parametrize(
    "settings",
    [
        {"n_resamples": 0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"metric": {"name": "m", "function": len}},
        {"random_stat": 0},
        {"n_resamples": True},
    ],
    ids=[
        "no-resamples",
        "alpha-0",
        "alpha-1",
        "metric-dict",
        "misspelt-keyword",
        "bool-resamples",
    ],
)
def test_invalid_settings_raise(settings: dict[str, Any]) -> None:
    # Catches settings that would fail only after the predictions, or give a
    # degenerate interval. Pydantic's default ignores extras and reads True as 1.
    kwargs: dict[str, Any] = {"metric": Metric("mean_id", _RecordingMetric())}
    with pytest.raises(ValidationError):
        BootstrapEvaluator(**(kwargs | {"n_resamples": 10} | settings))


def test_cohort_model_with_cohort_log_loss() -> None:
    # Catches DataFrame targets and predictions being taken by label rather
    # than by position: the frames' index is shuffled and not 0..n-1.
    rng = np.random.default_rng(0)
    index = rng.permutation(200) + 1000
    X_cohort = pd.DataFrame({"x": rng.normal(size=200)}, index=index)
    y_cohort = pd.DataFrame(
        {
            "a": rng.poisson(np.exp(1 + 0.5 * X_cohort["x"])),
            "b": rng.poisson(np.exp(0.5 - 0.3 * X_cohort["x"])),
        },
        index=index,
    )
    groups = np.repeat(np.arange(20), 10)
    model = CohortProbabilityModel(estimator=LogisticRegression()).fit(
        X_cohort, y_cohort
    )
    evaluator = BootstrapEvaluator(COHORT_LOG_LOSS, 100, random_state=0)
    result = evaluator.evaluate(model, X_cohort, y_cohort, groups)
    full = model.evaluate(y_cohort, model.predict(X_cohort), COHORT_LOG_LOSS)
    assert result.lower < full < result.upper
