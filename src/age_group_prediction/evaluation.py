"""Metric orchestration and neighborhood-cluster bootstrap evaluation."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from .metrics import (
    Metric,
    MetricResult,
    PredictionCapability,
    available_prediction_capabilities,
)
from .modeling_config import EvaluationConfig
from .resampling import NeighborhoodClusterResampler
from .results import (
    EvaluationResult,
    ParametricDistributionSpec,
    PredictionResult,
)

MissingCapabilityPolicy = Literal["error", "skip"]


@dataclass(frozen=True)
class BootstrapEvaluationResult:
    """Point metrics and percentile intervals from cluster resampling."""

    point_evaluation: EvaluationResult
    intervals_df: pd.DataFrame
    replicate_metrics_df: pd.DataFrame
    failed_replicates: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Copy frames and freeze serializable bootstrap metadata."""
        intervals = self.intervals_df.copy(deep=True)
        replicates = self.replicate_metrics_df.copy(deep=True)
        metadata = copy.deepcopy(dict(self.metadata))
        json.dumps(metadata)
        object.__setattr__(self, "intervals_df", intervals)
        object.__setattr__(self, "replicate_metrics_df", replicates)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


def evaluate_predictions(
    observed: pd.DataFrame,
    prediction: PredictionResult,
    metrics: Sequence[Metric],
    *,
    rng: np.random.Generator | None = None,
    on_missing_capability: MissingCapabilityPolicy = "error",
) -> EvaluationResult:
    """Evaluate fixed predictions without fitting or transforming features."""
    selected_metrics, skipped_metrics = _resolve_metrics(
        prediction,
        metrics,
        on_missing_capability=on_missing_capability,
    )
    metric_results = [
        metric.compute(observed, prediction, rng=rng) for metric in selected_metrics
    ]
    metrics_df = pd.DataFrame(
        [result.to_record() for result in metric_results],
        columns=[
            "metric_name",
            "value",
            "aggregation_level",
            "sample_count",
            "target",
        ],
    )
    diagnostics = {
        _metric_key(result): dict(result.metadata)
        for result in metric_results
        if result.metadata
    }
    return EvaluationResult(
        metrics_df=metrics_df,
        diagnostics=diagnostics,
        metadata={
            "metric_count": len(metric_results),
            "metric_definitions": [
                _metric_definition(metric) for metric in selected_metrics
            ],
            "missing_capability_policy": on_missing_capability,
            "skipped_metric_definitions": [
                _metric_definition(metric) for metric in skipped_metrics
            ],
        },
    )


def neighborhood_cluster_bootstrap(
    observed: pd.DataFrame,
    prediction: PredictionResult,
    metrics: Sequence[Metric],
    *,
    config: EvaluationConfig,
    default_seed: int,
    rng: np.random.Generator | None = None,
    on_missing_capability: MissingCapabilityPolicy = "error",
) -> BootstrapEvaluationResult:
    """Bootstrap fixed predictions by resampling whole neighborhoods."""
    if default_seed < 0:
        raise ValueError("default_seed must be nonnegative")
    if len(observed) != len(prediction.building_ids):
        raise ValueError("Observed rows must match prediction rows")
    resampler = NeighborhoodClusterResampler.from_frame(
        observed, config.neighborhood_id_column
    )

    resolved_rng = np.random.default_rng(default_seed) if rng is None else rng
    point_evaluation = evaluate_predictions(
        observed,
        prediction,
        metrics,
        rng=resolved_rng,
        on_missing_capability=on_missing_capability,
    )
    selected_metrics, _ = _resolve_metrics(
        prediction,
        metrics,
        on_missing_capability=on_missing_capability,
    )
    replicate_records: list[dict[str, object]] = []
    failed_replicates = 0
    for replicate_index in range(config.bootstrap_replicates):
        row_indices = resampler.sample_row_indices(resolved_rng)
        try:
            sampled_observed = observed.iloc[row_indices].reset_index(drop=True)
            sampled_prediction = _subset_prediction(prediction, row_indices)
            results = [
                metric.compute(sampled_observed, sampled_prediction, rng=resolved_rng)
                for metric in selected_metrics
            ]
        except (ValueError, FloatingPointError):
            failed_replicates += 1
            continue
        for result in results:
            replicate_records.append(
                {
                    "replicate": replicate_index,
                    **result.to_record(),
                }
            )

    if failed_replicates / config.bootstrap_replicates > config.max_failed_fraction:
        raise RuntimeError("Too many neighborhood-bootstrap replicates failed")
    replicate_df = pd.DataFrame(replicate_records)
    intervals_df = _percentile_intervals(
        point_evaluation.metrics_df,
        replicate_df,
        confidence_level=config.confidence_level,
    )
    return BootstrapEvaluationResult(
        point_evaluation=point_evaluation,
        intervals_df=intervals_df,
        replicate_metrics_df=replicate_df,
        failed_replicates=failed_replicates,
        metadata={
            "bootstrap_unit": "neighborhood",
            "replicate_count": config.bootstrap_replicates,
            "confidence_level": config.confidence_level,
            "interval_method": config.interval_method,
            "failed_replicates": failed_replicates,
            "default_seed": default_seed if rng is None else None,
            "neighborhood_id_column": config.neighborhood_id_column,
        },
    )


def _subset_prediction(
    prediction: PredictionResult, row_indices: np.ndarray
) -> PredictionResult:
    """Restrict a prediction to resampled rows, preserving payload contracts."""

    def subset_mapping(
        values: Mapping[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray] | None:
        if values is None:
            return None
        return {key: value[row_indices] for key, value in values.items()}

    def subset_parameter(value: float | np.ndarray | None) -> float | np.ndarray | None:
        # Models fit one dispersion or scale per target and broadcast it across
        # buildings, so these arrive as per-building arrays of the original
        # length. Leaving them unsubset makes every differently sized resample
        # fail the per-building length check, and silently misaligns the
        # parameters with the resampled rows whenever the lengths coincide.
        if value is None or np.ndim(value) == 0:
            return value
        return np.asarray(value)[row_indices]

    def subset_distributions(
        values: Mapping[str, ParametricDistributionSpec] | None,
    ) -> dict[str, ParametricDistributionSpec] | None:
        if values is None:
            return None
        return {
            target: ParametricDistributionSpec(
                family=spec.family,
                dispersion=subset_parameter(spec.dispersion),
                scale=subset_parameter(spec.scale),
            )
            for target, spec in values.items()
        }

    return PredictionResult(
        building_ids=np.arange(len(row_indices)),
        cohort_names=prediction.cohort_names,
        total_mean=prediction.total_mean[row_indices],
        cohort_means=prediction.cohort_means[row_indices],
        age_group_probabilities=prediction.age_group_probabilities[row_indices],
        reconciliation_error=prediction.reconciliation_error[row_indices],
        predictive_draws=subset_mapping(prediction.predictive_draws),
        prediction_intervals=subset_mapping(prediction.prediction_intervals),
        interval_levels=prediction.interval_levels,
        pointwise_log_probabilities=subset_mapping(
            prediction.pointwise_log_probabilities
        ),
        parametric_distributions=subset_distributions(
            prediction.parametric_distributions
        ),
        validation_config=prediction.validation_config,
        pointwise_log_probability_scope=prediction.pointwise_log_probability_scope,
    )


def _percentile_intervals(
    point_metrics: pd.DataFrame,
    replicate_metrics: pd.DataFrame,
    *,
    confidence_level: float,
) -> pd.DataFrame:
    """Summarize replicate metric values as percentile confidence intervals."""
    alpha = 1.0 - confidence_level
    records: list[dict[str, object]] = []
    for row in point_metrics.itertuples(index=False):
        # The aggregation level is part of a metric's identity: two metrics
        # sharing a name and target at different levels are different
        # quantities, and pooling their replicates would mix them. It is also
        # emitted, so fold-scope rows carry the same key columns as the
        # cross-validation scope they are stacked with.
        mask = (replicate_metrics["metric_name"] == row.metric_name) & (
            replicate_metrics["target"] == row.target
        )
        if "aggregation_level" in replicate_metrics.columns:
            mask &= replicate_metrics["aggregation_level"] == row.aggregation_level
        values = replicate_metrics.loc[mask, "value"].to_numpy(dtype=float)
        if not len(values):
            continue
        lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        records.append(
            {
                "metric_name": row.metric_name,
                "target": row.target,
                "aggregation_level": row.aggregation_level,
                "value": row.value,
                "confidence_lower": float(lower),
                "confidence_upper": float(upper),
                "successful_replicates": len(values),
            }
        )
    return pd.DataFrame(records)


def _metric_key(result: MetricResult) -> str:
    """Build a stable diagnostics key from metric name and target."""
    return f"{result.metric_name}:{result.target}"


def _resolve_metrics(
    prediction: PredictionResult,
    metrics: Sequence[Metric],
    *,
    on_missing_capability: MissingCapabilityPolicy,
) -> tuple[tuple[Metric, ...], tuple[Metric, ...]]:
    """Validate metric capabilities and return selected and skipped metrics."""
    if on_missing_capability not in {"error", "skip"}:
        raise ValueError("on_missing_capability must be 'error' or 'skip'")
    available = available_prediction_capabilities(prediction)
    selected: list[Metric] = []
    skipped: list[Metric] = []
    for metric in metrics:
        if _metric_capability_available(prediction, metric, available):
            selected.append(metric)
            continue
        if on_missing_capability == "error":
            raise ValueError(
                f"Metric '{metric.name}' for target '{metric.target}' requires "
                f"missing prediction capability '{metric.required_capability}'"
            )
        skipped.append(metric)
    return tuple(selected), tuple(skipped)


def _metric_capability_available(
    prediction: PredictionResult,
    metric: Metric,
    available: frozenset[PredictionCapability],
) -> bool:
    """Return whether a capability is available for the metric's target."""
    capability = metric.required_capability
    if capability not in available:
        return False
    payload_key = "total" if metric.target in {"total", "n_children_total"} else metric.target
    payloads: dict[PredictionCapability, Mapping[str, object] | None] = {
        "pointwise_log_probabilities": prediction.pointwise_log_probabilities,
        "predictive_draws": prediction.predictive_draws,
        "prediction_intervals": prediction.prediction_intervals,
        "parametric_distributions": prediction.parametric_distributions,
    }
    payload = payloads.get(capability)
    return payload is None or payload_key in payload


def _metric_definition(metric: Metric) -> dict[str, Any]:
    """Describe one metric so evaluation metadata is reproducible."""
    definition: dict[str, Any] = {}
    if is_dataclass(metric) and not isinstance(metric, type):
        definition.update(asdict(metric))
    definition.update(
        {
            "name": metric.name,
            "target": metric.target,
            "required_capability": metric.required_capability,
            "optimization_direction": metric.optimization_direction,
            "aggregation_level": metric.aggregation_level,
        }
    )
    return definition
