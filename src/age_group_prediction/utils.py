"""Helpers shared across packages: table types, the modeling dataset, and positional row selection."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "AgeGroupDataset",
    "DesignMatrix",
    "Exposure",
    "Groups",
    "Target",
    "take_rows",
]

type DesignMatrix = pd.DataFrame | np.ndarray  # the features, one row per unit
type Target = pd.Series | pd.DataFrame | np.ndarray  # one column, or several
type Groups = pd.Series | np.ndarray  # the split key, one label per row
type Exposure = pd.Series | np.ndarray  # a model's raw exposure, one per row


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
