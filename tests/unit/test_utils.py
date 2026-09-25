"""Tests for the shared helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from age_group_prediction.utils import take_rows

INDEX = np.array([2, 0])


def test_take_rows_indexes_numpy_by_position() -> None:
    array = np.array([[10, 11], [20, 21], [30, 31]])

    assert take_rows(array, INDEX).tolist() == [[30, 31], [10, 11]]


def test_take_rows_ignores_the_pandas_labels() -> None:
    # Fold indices are positions; label-based selection would pick rows 2 and 0
    # of the index labels, i.e. the wrong rows here.
    frame = pd.DataFrame({"x": [10, 20, 30]}, index=[2, 1, 0])
    series = frame["x"]

    assert take_rows(frame, INDEX)["x"].tolist() == [30, 10]
    assert take_rows(series, INDEX).tolist() == [30, 10]
    assert take_rows(series, INDEX).index.tolist() == [0, 2]
