"""Capture observable contracts for the Jupyter EDA reference population."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from student_simulator import (
    StudentPopulationSimulator,
    load_simulation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "simulation.toml"
BASELINE_PATH = (
    Path(__file__).with_name("fixtures") / "eda_jupyter_baseline.json"
)
EDA_NEIGHBORHOOD_COUNT = 150
EDA_BUILDING_RATE = 9.0

ROOM_COLUMNS = ("3_rooms", "4_rooms", "5_rooms", "6_rooms")
COHORT_COLUMNS = ("n_kindergarten", "n_elementary", "n_highschool")
TARGET_COLUMNS = (*COHORT_COLUMNS, "n_children_total")
NUMERIC_FEATURE_COLUMNS = (
    "ses",
    "avg_household_size",
    "median_age",
    "n_daycares_500m",
    *ROOM_COLUMNS,
    "n_apartments",
)
PLOT_INVENTORY = (
    "target_probability_histograms",
    "target_empirical_cdfs",
    "numeric_feature_histograms",
    "school_status_frequency",
    "pearson_correlation_heatmap",
    "spearman_correlation_heatmap",
)
STRING_COLUMNS = ("building_id", "neighborhood_id", "school_status")


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _dataframe_hash(frame: pd.DataFrame) -> str:
    """Hash values, index, column order, and dtypes without exporting data."""
    frame = _canonicalize_string_columns(frame)
    digest = sha256()
    digest.update("|".join(map(repr, frame.columns)).encode())
    digest.update("|".join(map(str, frame.dtypes)).encode())
    digest.update(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    )
    return digest.hexdigest()


def _canonicalize_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize logical strings across pandas storage settings."""
    return frame.astype(
        {column: "string" for column in STRING_COLUMNS if column in frame},
    )


def _summary_records(
    frame: pd.DataFrame,
) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for column in TARGET_COLUMNS:
        series = frame[column].astype(float)
        mean = float(series.mean())
        variance = float(series.var(ddof=1))
        records.append(
            {
                "target": column,
                "count": int(series.size),
                "mean": mean,
                "std": float(series.std(ddof=1)),
                "q25": float(series.quantile(0.25)),
                "q50": float(series.quantile(0.5)),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "zero_share": float((series == 0).mean()),
                "variance_to_mean": variance / mean if mean > 0 else np.nan,
            }
        )
    return records


def _describe_records(series: pd.Series) -> dict[str, float]:
    description = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return {name: float(value) for name, value in description.items()}


def _quality_checks(frame: pd.DataFrame) -> dict[str, bool]:
    room_sum = frame[list(ROOM_COLUMNS)].sum(axis=1)
    cohort_sum = frame[list(COHORT_COLUMNS)].sum(axis=1)
    context_unique = frame.groupby("neighborhood_id")[
        [
            "ses",
            "avg_household_size",
            "median_age",
            "n_daycares_500m",
            "school_status",
        ]
    ].nunique(dropna=False)
    return {
        "building_id_unique": bool(frame["building_id"].is_unique),
        "neighborhood_id_not_null": bool(
            frame["neighborhood_id"].notna().all()
        ),
        "targets_integer_dtype": all(
            is_integer_dtype(frame[column]) for column in TARGET_COLUMNS
        ),
        "targets_non_negative": bool(
            (frame[list(TARGET_COLUMNS)] >= 0).all().all()
        ),
        "n_apartments_positive": bool((frame["n_apartments"] > 0).all()),
        "rooms_sum_equals_n_apartments": bool(
            (room_sum == frame["n_apartments"]).all()
        ),
        "cohorts_sum_equals_total": bool(
            (cohort_sum == frame["n_children_total"]).all()
        ),
        "no_missing_values": bool(not frame.isna().any().any()),
        "context_repeats_within_neighborhood": bool(
            (context_unique <= 1).all().all()
        ),
    }


def capture_eda_baseline() -> dict[str, Any]:
    """Run the reference EDA configuration and return observable artifacts."""
    base_config = load_simulation_config(CONFIG_PATH)
    eda_config = base_config.model_copy(
        update={
            "simulation": base_config.simulation.model_copy(
                update={"n_neighborhoods": EDA_NEIGHBORHOOD_COUNT}
            ),
            "building": base_config.building.model_copy(
                update={"buildings_per_neighborhood_rate": EDA_BUILDING_RATE}
            ),
        }
    )
    simulator = StudentPopulationSimulator(eda_config)
    final_df = simulator.run()
    canonical_final_df = _canonicalize_string_columns(final_df)

    buildings_per_neighborhood = (
        final_df["neighborhood_id"].value_counts().sort_index()
    )
    positive_total = final_df["n_children_total"] > 0
    cohort_shares = final_df.loc[positive_total, list(COHORT_COLUMNS)].div(
        final_df.loc[positive_total, "n_children_total"],
        axis=0,
    )
    correlation_columns = [*NUMERIC_FEATURE_COLUMNS, *TARGET_COLUMNS]
    correlations = pd.concat(
        {
            "pearson": final_df[correlation_columns].corr(method="pearson"),
            "spearman": final_df[correlation_columns].corr(method="spearman"),
        },
        axis=1,
    )

    return {
        "config": {
            "path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "content_sha256": _file_hash(CONFIG_PATH),
            "seed": eda_config.simulation.seed,
            "overrides": {
                "n_neighborhoods": EDA_NEIGHBORHOOD_COUNT,
                "buildings_per_neighborhood_rate": EDA_BUILDING_RATE,
            },
        },
        "final_dataframe": {
            "shape": list(canonical_final_df.shape),
            "columns": list(canonical_final_df.columns),
            "dtypes": {
                column: str(dtype)
                for column, dtype in canonical_final_df.dtypes.items()
            },
            "sha256": _dataframe_hash(canonical_final_df),
            "unique_neighborhoods": int(final_df["neighborhood_id"].nunique()),
        },
        "quality_checks": _quality_checks(final_df),
        "summaries": {
            "buildings_per_neighborhood": _describe_records(
                buildings_per_neighborhood
            ),
            "targets": _summary_records(final_df),
            "cohort_share_describe_sha256": _dataframe_hash(
                cohort_shares.describe(percentiles=[0.1, 0.5, 0.9]).T
            ),
            "feature_target_correlations_sha256": _dataframe_hash(
                correlations
            ),
        },
        "plot_inventory": list(PLOT_INVENTORY),
        "comparison_policy": {
            "exact": [
                "config",
                "final_dataframe",
                "quality_checks",
                "summaries",
            ],
            "semantic_only": ["plot_inventory"],
            "excluded": ["private_latent_effects", "rendered_plot_pixels"],
        },
    }
