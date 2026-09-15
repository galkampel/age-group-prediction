"""Explicit construction of the building-level modeling table."""

from __future__ import annotations

import pandas as pd

from .modeling_config import DEFAULT_MODELING_SCHEMA, ModelingSchema


def build_modeling_table(
    source_df: pd.DataFrame,
    *,
    schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
) -> pd.DataFrame:
    """Select raw columns, derive room shares, order, and validate the table."""
    schema.validate_definition()

    required_source_columns = {
        *schema.identifier_columns,
        *schema.raw_numeric_features,
        *schema.categorical_features,
        *schema.room_count_columns,
        *schema.target_columns,
    }
    missing_columns = required_source_columns.difference(source_df.columns)
    if missing_columns:
        raise ValueError(f"Missing source columns: {sorted(missing_columns)}")

    base_columns = (
        *schema.identifier_columns,
        *schema.raw_numeric_features,
        *schema.categorical_features,
        *schema.target_columns,
    )
    modeling_df = source_df.loc[:, list(base_columns)].copy()
    # Compute room share columns based on the exposure column.
    for room_column, share_column in schema.room_share_columns:
        modeling_df[share_column] = (
            source_df[room_column] / source_df[schema.exposure_column]
        )

    modeling_df = modeling_df.loc[:, list(schema.table_columns)].copy()
    schema.validate_table(modeling_df)
    return modeling_df
