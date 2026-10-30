"""Helpers shared across packages: table types, the modeling dataset, positional row selection, and the checks that pair ``y_true`` and ``y_pred``."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "AgeGroupDataset",
    "DesignMatrix",
    "Exposure",
    "Groups",
    "Target",
    "check_paired_arrays",
    "column_position",
    "take_rows",
]

type DesignMatrix = pd.DataFrame | np.ndarray  # the features, one row per unit
type Target = pd.Series | pd.DataFrame | np.ndarray  # one column, or several
type Groups = pd.Series | np.ndarray  # the split key, one label per row
type Exposure = pd.Series | np.ndarray  # a model's raw exposure, one per row


def check_paired_arrays(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """``y_true`` and ``y_pred`` as finite float arrays of one shape, paired by position."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    # numpy would broadcast one column over every other, and a transposed or
    # flattened prediction has as many cells: either is scored mispaired, silently.
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, (n_samples,) or "
            f"(n_samples, n_outputs), got {y_true_array.shape} and {y_pred_array.shape}"
        )
    # A NaN or an infinity would pass through a sum as a NaN or inf score, as
    # scikit-learn's check_array rejects them.
    if not (np.isfinite(y_true_array).all() and np.isfinite(y_pred_array).all()):
        raise ValueError("y_true and y_pred must be finite, got a NaN or an infinity")
    return y_true_array, y_pred_array


def column_position(table: ArrayLike, column: str) -> int:
    """The position of the column named ``column`` in ``table``, a DataFrame."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"{column!r} is a column name, so table must be a DataFrame, "
            f"got {type(table).__name__}"
        )
    # A missing name raises pandas' KeyError naming it.
    return table.columns.get_loc(column)


def take_rows[T](array: T, index: np.ndarray) -> T:
    """The rows at positions ``index``: ``.iloc`` for pandas, plain indexing otherwise."""
    rows: Any = array
    return rows.iloc[index] if hasattr(array, "iloc") else rows[index]


class AgeGroupDataset(NamedTuple):
    """One table's modeling inputs, row-aligned: features, cohort counts, exposure, split groups, building ids."""

    X: pd.DataFrame
    Y: pd.DataFrame
    exposure: pd.Series
    groups: pd.Series
    ids: pd.Series
