"""Stable content hashes used to tie artifacts back to the data behind them."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

__all__ = [
    "column_schema_hash",
    "table_hash",
]


def table_hash(df: pd.DataFrame, *, id_column: str) -> str:
    """Hash a table's contents independently of its row order.

    Rows are canonicalized by ``id_column`` before hashing, so the same rows in
    a different order hash identically. Column order and dtypes do affect the
    result, which is intended: a table whose columns were reordered or retyped
    is not the table the hash was taken from.
    """
    canonical_df = df.sort_values(id_column).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(canonical_df, index=False)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


def column_schema_hash(df: pd.DataFrame) -> str:
    """Hash a table's column names and dtypes in order."""
    schema = [(column, str(dtype)) for column, dtype in df.dtypes.items()]
    return hashlib.sha256(json.dumps(schema).encode()).hexdigest()
