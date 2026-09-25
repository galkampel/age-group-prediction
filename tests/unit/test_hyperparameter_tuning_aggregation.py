"""Tests for the fold aggregations and the corrected standard error."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from age_group_prediction.hyperparameter_tuning import (
    Aggregation,
    LowerBound,
    Mean,
    WeightedMean,
    corrected_std_error,
)
from age_group_prediction.splitting import Splitter

# Unequal folds: the `grouped` sizes on the evaluator tests' 28 rows.
SCORES = [-3.0, -5.0, -4.5]
FOLD_SIZES = [20, 5, 3]
# Worked by hand:
WEIGHTED_MEAN = (20 * -3.0 + 5 * -5.0 + 3 * -4.5) / 28  # -98.5 / 28
MEAN = (-3.0 - 5.0 - 4.5) / 3
# s^2 = 13/12 (ddof=1); the folds partition 28 rows, so
# n_val/n_train = (28/3) / (56/3) = 1/2 = 1/(K-1).
STD_ERROR = math.sqrt(13 / 12 * (1 / 3 + 1 / 2))


class _Median(Aggregation):
    """A custom rule: only aggregate is defined."""

    def aggregate(self, scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
        return float(np.median(scores))


def test_corrected_std_error_matches_the_hand_value() -> None:
    assert corrected_std_error(SCORES) == pytest.approx(STD_ERROR)


def _size_ratio(method: str, groups: np.ndarray) -> float:
    """``mean(n_val) / mean(n_train)`` over a 5-fold ``Splitter.cv`` split."""
    X = pd.DataFrame({"x": np.arange(len(groups), dtype=float)})
    folds = (
        Splitter(method)
        .cv(n_splits=5, random_state=0)
        .split(X, np.zeros(len(groups)), None if method == "random" else groups)
    )
    sizes = np.array([(len(v), len(t)) for t, v in folds])
    return float(sizes[:, 0].mean() / sizes[:, 1].mean())


@pytest.mark.parametrize("method", ["random", "stratified_by_group", "grouped"])
def test_the_splitter_gives_the_ratio_the_std_error_assumes(method: str) -> None:
    groups = np.repeat(np.arange(10), 4)
    assert _size_ratio(method, groups) == pytest.approx(1 / (5 - 1))


def test_singleton_strata_make_the_std_error_conservative() -> None:
    # Never validated, so the true ratio is below 1/(K-1): the SE is too large.
    groups = np.concatenate([np.repeat(np.arange(10), 4), np.arange(10, 20)])
    assert _size_ratio("stratified_by_group", groups) < 1 / (5 - 1)


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        (WeightedMean(), WEIGHTED_MEAN),
        (Mean(), MEAN),
        (LowerBound(), WEIGHTED_MEAN - STD_ERROR),
        (LowerBound(z=2), WEIGHTED_MEAN - 2 * STD_ERROR),
        (_Median(), -4.5),
    ],
    ids=["weighted-mean", "mean", "lower-bound", "lower-bound-z2", "median"],
)
def test_each_aggregation_matches_the_hand_value(
    aggregation: Aggregation, expected: float
) -> None:
    value = aggregation.aggregate(SCORES, FOLD_SIZES)
    assert value == pytest.approx(expected)
    # Plain Python numbers: sqlite storage rejects numpy scalars as user attrs.
    assert type(value) is float


@pytest.mark.parametrize("aggregation", [WeightedMean(), Mean(), LowerBound()])
def test_numpy_arrays_score_like_lists(aggregation: Aggregation) -> None:
    arrays = aggregation.aggregate(np.array(SCORES), np.array(FOLD_SIZES))
    assert arrays == aggregation.aggregate(SCORES, FOLD_SIZES)


@pytest.mark.parametrize("aggregation", [WeightedMean(), Mean(), LowerBound()])
def test_no_folds_are_rejected(aggregation: Aggregation) -> None:
    with pytest.raises(ValueError, match="at least one fold"):
        aggregation.aggregate([], [])


@pytest.mark.parametrize("aggregation", [WeightedMean(), Mean(), LowerBound()])
def test_unequal_lengths_are_rejected(aggregation: Aggregation) -> None:
    with pytest.raises(ValueError, match="equal lengths, got 3 and 2"):
        aggregation.aggregate(SCORES, [20, 5])


@pytest.mark.filterwarnings("error")
def test_corrected_std_error_needs_two_folds() -> None:
    with pytest.raises(ValueError, match="at least 2 folds, got 1"):
        corrected_std_error([-3.0])


@pytest.mark.parametrize(
    "z",
    [0, -1.0, float("nan"), float("inf"), "1", True],
    ids=["zero", "negative", "nan", "inf", "string", "bool"],
)
def test_lower_bound_rejects_an_invalid_z(z: Any) -> None:
    with pytest.raises(ValueError, match="LowerBound\nz\n"):
        LowerBound(z=z)


def test_aggregations_are_frozen_and_compare_by_value() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        LowerBound().z = 2.0  # type: ignore[misc]
    assert LowerBound(2) == LowerBound(2.0)
    assert WeightedMean() == WeightedMean()
    assert repr(WeightedMean()) == "WeightedMean()"


def test_aggregation_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Aggregation()  # type: ignore[abstract]
