"""Tests for the shared helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from age_group_prediction.utils import check_paired_arrays, column_position, take_rows

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


def test_check_paired_arrays_returns_float_arrays_of_one_shape() -> None:
    # Lists convert; a transposed prediction has as many cells as y_true, so
    # once raveled it would be scored mispaired, silently.
    y_true, y_pred = check_paired_arrays([[1, 2], [3, 4]], [[1.5, 2.5], [3.5, 4.5]])

    assert y_true.dtype == y_pred.dtype == np.float64
    assert y_true.shape == y_pred.shape == (2, 2)
    with pytest.raises(ValueError, match="same shape"):
        check_paired_arrays(np.zeros((3, 2)), np.zeros((2, 3)))


def test_column_position_finds_a_column_name_of_a_dataframe() -> None:
    # A misspelt name, and an array without names to look the column up in.
    table = pd.DataFrame(np.zeros((2, 3)), columns=["k", "e", "h"])

    assert column_position(table, "h") == 2
    with pytest.raises(KeyError, match="n_elementary"):
        column_position(table, "n_elementary")
    with pytest.raises(TypeError, match="DataFrame"):
        column_position(table.to_numpy(), "h")
