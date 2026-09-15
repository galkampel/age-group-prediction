"""Fold-metric aggregation, bootstrap evidence, and importance summaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..modeling_config import EvaluationConfig


def _aggregate_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_id", "approach", "metric_name", "target", "aggregation_level"]
    return (
        fold_metrics.groupby(keys, sort=True, as_index=False)["value"]
        .agg(mean="mean", standard_deviation="std", fold_count="count")
        .fillna({"standard_deviation": 0.0})
    )


def _concat_or_empty(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bootstrap_interval_evidence(
    *,
    fold_intervals: Sequence[pd.DataFrame],
    replicate_frames: Sequence[pd.DataFrame],
    aggregate_metrics: pd.DataFrame,
    evaluation_config: EvaluationConfig | None,
) -> pd.DataFrame:
    if evaluation_config is None:
        return pd.DataFrame()
    fold_evidence = _concat_or_empty(fold_intervals).assign(
        scope="fold",
        interval_basis="within_fold_neighborhood_cluster_bootstrap",
        assumes_independent_folds=False,
    )
    replicates = _concat_or_empty(replicate_frames)
    if replicates.empty:
        return fold_evidence
    group_keys = ["candidate_id", "approach", "metric_name", "target"]
    if "aggregation_level" in replicates.columns:
        group_keys.append("aggregation_level")
    cross_validation_replicates = (
        replicates.groupby([*group_keys, "replicate"], sort=True, as_index=False)[
            "value"
        ].mean()
    )
    alpha = 1.0 - evaluation_config.confidence_level
    records: list[dict[str, Any]] = []
    for key, group in cross_validation_replicates.groupby(group_keys, sort=True):
        key_values = dict(zip(group_keys, key, strict=True))
        values = group["value"].to_numpy(dtype=float)
        mask = aggregate_metrics["candidate_id"] == key_values["candidate_id"]
        for column in ("metric_name", "target", "aggregation_level"):
            if column in key_values and column in aggregate_metrics.columns:
                mask &= aggregate_metrics[column] == key_values[column]
        point = aggregate_metrics.loc[mask, "mean"]
        # An ambiguous mask means the key columns do not identify one metric,
        # so taking the first row would silently attach an interval to the
        # wrong point estimate.
        if len(point) != 1:
            raise ValueError(
                f"Expected exactly one aggregate for {key_values}; found {len(point)}"
            )
        lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        records.append(
            {
                **key_values,
                "value": float(point.iloc[0]),
                "confidence_lower": float(lower),
                "confidence_upper": float(upper),
                "successful_replicates": len(values),
                "fold_index": None,
                "scope": "cross_validation",
                # Replicate r is an independent resample in each fold, so pairing
                # replicates by index and averaging treats the folds as
                # independent. Validation folds are repeated overlapping splits,
                # so this band is narrower than the true sampling variability of
                # the fold mean. Fold-scope intervals remain the primary evidence.
                "interval_basis": "index_paired_replicate_mean_across_folds",
                "assumes_independent_folds": True,
            }
        )
    return pd.concat([fold_evidence, pd.DataFrame(records)], ignore_index=True)


def _summarize_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    keys = [
        "candidate_id",
        "approach",
        "component",
        "metric_name",
        "target",
        "feature_block",
        "columns",
        "optimization_direction",
    ]
    grouped = importance.groupby(keys, sort=True)["degradation"]
    summary = grouped.agg(
        mean_degradation="mean",
        standard_deviation="std",
        observation_count="count",
    ).reset_index()
    quantiles = grouped.quantile([0.025, 0.975]).unstack().reset_index()
    quantiles = quantiles.rename(columns={0.025: "quantile_025", 0.975: "quantile_975"})
    return summary.merge(quantiles, on=keys, validate="one_to_one").fillna(
        {"standard_deviation": 0.0}
    )
