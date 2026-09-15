"""Unit tests for shared predictive payload helpers."""

from __future__ import annotations

import numpy as np

from age_group_prediction.predictive import central_prediction_intervals


def test_central_prediction_intervals_values() -> None:
    draws = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    levels = (0.5, 0.8)

    intervals = central_prediction_intervals(draws, levels)

    expected = np.array(
        [
            [[1.75, 3.25], [1.3, 3.7]],
            [[17.5, 32.5], [13.0, 37.0]],
        ]
    )
    np.testing.assert_allclose(intervals, expected)


def test_central_prediction_intervals_empty_levels() -> None:
    draws = np.array([[0.0, 1.0], [2.0, 3.0]])
    intervals = central_prediction_intervals(draws, ())

    assert intervals.shape == (2, 0, 2)
