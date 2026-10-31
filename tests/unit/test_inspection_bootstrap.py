"""Tests for ``BootstrapEvaluator``."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

from age_group_prediction.inspection import BootstrapEvaluator
from age_group_prediction.modeling import CohortProbabilityModel
from age_group_prediction.scoring import (
    Direction,
    Metric,
    mean_absolute_error_loss,
    multinomial_log_loss,
)

COHORT_LOG_LOSS = Metric("cohort_log_loss", multinomial_log_loss)

# y_true is the row id, and so is the prediction, so a metric can tell which
# rows a replicate drew and whether y_true and y_pred stay paired. Groups of 3,
# 5, 8 and 12 rows with unsorted labels, interleaved as real neighborhoods are,
# so code that assumes contiguous groups fails.
GROUPS = np.random.default_rng(0).permutation(np.repeat([7, 2, 9, 4], [3, 5, 8, 12]))
Y = np.arange(28.0)
Y_PRED = np.arange(28.0)


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
    direction: Direction = "minimize",
    **settings: Any,
) -> pd.Series:
    """The one-metric row of the recorder's bootstrap: ``mean``, ``lower``, ``upper``."""
    metric = Metric("mean_id", recorder, direction=direction)
    kwargs: dict[str, Any] = {"n_resamples": 50, "random_state": 0}
    evaluator = BootstrapEvaluator(**(kwargs | settings))
    return evaluator.evaluate(Y, Y_PRED, groups, metrics=[metric]).loc[
        ("composite", "mean_id")
    ]


def _two_metrics() -> list[Metric]:
    return [Metric("mean_id", _RecordingMetric()), Metric("twice", _RecordingMetric())]


def test_every_metric_scores_the_same_replicates() -> None:
    # Catches the replicates being drawn per metric: the intervals of one call
    # must be paired, so both recorders see the same rows in the same order.
    first, second = _RecordingMetric(), _RecordingMetric()
    evaluator = BootstrapEvaluator(20, random_state=0)
    result = evaluator.evaluate(
        Y, Y_PRED, GROUPS, metrics=[Metric("a", first), Metric("b", second)]
    )
    # Every call of each recorder scores a replicate.
    assert len(first.replicates) == len(second.replicates) == 20
    assert all(
        np.array_equal(ids_a, ids_b)
        for ids_a, ids_b in zip(first.replicates, second.replicates)
    )
    assert list(result.index) == [("composite", "a"), ("composite", "b")]
    assert result.index.names == ["cohort", "metric"]


def test_the_rows_follow_the_metrics_and_hold_the_interval() -> None:
    # Catches the metrics' rows in another order than given, or a column
    # other than the interval's.
    result = BootstrapEvaluator(10).evaluate(
        Y,
        Y_PRED,
        metrics=[
            Metric("mean_id", lambda y_true, y_pred: float(np.mean(y_true))),
            Metric("max_id", lambda y_true, y_pred: float(np.max(y_true))),
        ],
    )
    assert list(result.columns) == ["mean", "lower", "upper"]
    assert list(result.index) == [("composite", "mean_id"), ("composite", "max_id")]


@pytest.mark.parametrize(
    ("metrics", "match"),
    [
        ([], "at least one Metric"),
        ([Metric("m", len), Metric("m", len)], "must be distinct"),
    ],
    ids=["empty", "duplicate-names"],
)
def test_empty_or_duplicate_metrics_raise(metrics: list[Metric], match: str) -> None:
    # Catches an empty frame, or two rows under one label, returned silently.
    with pytest.raises(ValueError, match=match):
        BootstrapEvaluator(10).evaluate(Y, Y_PRED, metrics=metrics)


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
    assert all(len(ids) == len(Y) for ids in recorder.replicates)
    assert any(len(set(ids.tolist())) < len(Y) for ids in recorder.replicates)
    assert result["lower"] < result["upper"]


def test_predictions_stay_paired_with_targets() -> None:
    # Catches y_true and y_pred being taken at different rows.
    recorder = _RecordingMetric()
    _evaluate(recorder, GROUPS)
    _evaluate(recorder)
    assert recorder.paired


def test_same_seed_gives_identical_results() -> None:
    # Catches an unseeded generator, or a seed that is ignored.
    first, second, other = _RecordingMetric(), _RecordingMetric(), _RecordingMetric()
    assert _evaluate(first, GROUPS).equals(_evaluate(second, GROUPS))
    assert not _evaluate(other, GROUPS, random_state=1).equals(_evaluate(first, GROUPS))


def test_bounds_are_the_percentiles_of_the_replicates() -> None:
    # Catches a wrong quantile (e.g. alpha instead of alpha / 2) or a mean of
    # something other than the replicate scores.
    recorder = _RecordingMetric()
    result = _evaluate(recorder, alpha=0.2, n_resamples=200)
    scores = np.array(recorder.scores)
    assert len(scores) == 200
    assert result["mean"] == pytest.approx(scores.mean())
    assert result["lower"] == pytest.approx(np.quantile(scores, 0.1))
    assert result["upper"] == pytest.approx(np.quantile(scores, 0.9))
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_a_metric_to_maximize_is_reported_raw() -> None:
    # Catches the tuner's sign flip being applied here: the interval is in the
    # metric's own units.
    recorder = _RecordingMetric()
    result = _evaluate(recorder, direction="maximize")
    assert result["mean"] == pytest.approx(np.mean(recorder.scores))
    assert result["mean"] > 0


def test_non_finite_replicate_raises() -> None:
    # Catches nan bounds returned silently, e.g. a cohort log loss of one
    # replicate with no children (0 / 0), among finite ones.
    calls = iter(range(10))
    metric = Metric(
        "one_nan", lambda y_true, y_pred: float("nan") if next(calls) == 3 else 1.0
    )
    evaluator = BootstrapEvaluator(10)
    with pytest.raises(ValueError, match=r"not finite \(composite one_nan: 1 of 10\)"):
        evaluator.evaluate(
            Y, Y_PRED, metrics=[Metric("mean_id", _RecordingMetric()), metric]
        )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "groups"),
    [
        (Y[:-1], Y_PRED, None),
        (np.append(Y, 28.0), Y_PRED, None),
        (Y, Y_PRED[:-1], None),
        (Y, Y_PRED, GROUPS[:-1]),
    ],
    ids=["short-y_true", "long-y_true", "short-y_pred", "short-groups"],
)
def test_length_mismatch_raises(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray | None
) -> None:
    # Catches a y_true, y_pred or groups of another length: a longer one would
    # be silently mispaired, shorter groups would drop rows.
    evaluator = BootstrapEvaluator(10)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        evaluator.evaluate(y_true, y_pred, groups, metrics=_two_metrics()[:1])


def test_fewer_than_two_groups_raises() -> None:
    # Catches a single group passing silently: every replicate is then the
    # whole set and the interval has zero width.
    evaluator = BootstrapEvaluator(10)
    with pytest.raises(ValueError, match="1 distinct value"):
        evaluator.evaluate(
            Y, Y_PRED, np.zeros(len(Y), dtype=int), metrics=_two_metrics()[:1]
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"n_resamples": 0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"random_stat": 0},
        {"n_resamples": True},
    ],
    ids=[
        "no-resamples",
        "alpha-0",
        "alpha-1",
        "misspelt-keyword",
        "bool-resamples",
    ],
)
def test_invalid_settings_raise(settings: dict[str, Any]) -> None:
    # Catches settings that would fail only after the replicates, or give a
    # degenerate interval. Pydantic's default ignores extras and reads True as 1.
    with pytest.raises(ValidationError):
        BootstrapEvaluator(**({"n_resamples": 10} | settings))


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
    y_pred = model.predict(X_cohort)
    evaluator = BootstrapEvaluator(100, random_state=0)
    result = evaluator.evaluate(
        y_cohort, y_pred, groups, metrics=[COHORT_LOG_LOSS]
    ).loc[("composite", "cohort_log_loss")]
    full = COHORT_LOG_LOSS(y_cohort, y_pred)
    assert result["lower"] < full < result["upper"]


def test_each_cohort_is_scored_on_the_composites_replicates() -> None:
    # Catches a cohort scored as the whole of y, or replicates drawn per
    # cohort: with one seed, the composite rows equal a composite-only call,
    # and a cohort's row equals the bootstrap of its column alone.
    rng = np.random.default_rng(1)
    y_true = pd.DataFrame(
        {"a": rng.poisson(3.0, 28), "b": rng.poisson(6.0, 28)}, dtype=float
    )
    y_pred = pd.DataFrame(
        {"a": rng.uniform(1, 5, 28), "b": rng.uniform(4, 8, 28)}, dtype=float
    )
    metrics = [Metric("mae", mean_absolute_error_loss)]
    evaluator = BootstrapEvaluator(50, random_state=0)

    both = evaluator.evaluate(
        y_true, y_pred, GROUPS, metrics=metrics, cohorts=["a", "composite"]
    )
    composite = evaluator.evaluate(y_true, y_pred, GROUPS, metrics=metrics)

    assert list(both.index) == [("a", "mae"), ("composite", "mae")]
    # The same replicates; the mean's last bit depends on the array's width
    # (NumPy sums one column pairwise, several row by row).
    composite_rows = both.loc[[("composite", "mae")]]
    assert composite_rows.index.equals(composite.index)
    np.testing.assert_allclose(composite_rows, composite, rtol=1e-12, atol=0)
    column_alone = evaluator.evaluate(y_true["a"], y_pred["a"], GROUPS, metrics=metrics)
    np.testing.assert_allclose(
        both.loc[[("a", "mae")]], column_alone, rtol=1e-12, atol=0
    )
    assert both.loc[("a", "mae"), "mean"] != both.loc[("composite", "mae"), "mean"]
