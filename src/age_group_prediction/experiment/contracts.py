"""Typed, immutable declarations for candidates and selection policies."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from ..metrics import Metric
from ..modeling_config import FeatureSpec, PredictionConfig
from ..models.base import BaseAgeGroupModel

ApproachName = Literal[
    "DirectCohortModel",
    "IndependentTotalProbabilityModel",
    "BayesianConditionalModel",
]
SelectionRole = Literal["eligible", "diagnostic_comparator"]
OptimizationDirection = Literal["minimize", "maximize"]

_APPROACH_NAMES = {
    "DirectCohortModel",
    "IndependentTotalProbabilityModel",
    "BayesianConditionalModel",
}


@dataclass(frozen=True)
class FoldIdentity:
    """Serializable identity of one precomputed validation fold."""

    fold_index: int
    fit_building_ids: tuple[object, ...]
    validation_building_ids: tuple[object, ...]
    fingerprint: str


@dataclass(frozen=True)
class FeatureBlock:
    """Named raw columns permuted together for validation-only importance."""

    name: str
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature block name must not be empty")
        if not self.columns or any(not column for column in self.columns):
            raise ValueError("Feature blocks must contain named columns")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Feature block columns must be unique")


@dataclass(frozen=True)
class PermutationImportanceSpec:
    """One component, metric, and block set for repeated importance."""

    component: str
    metric: Metric
    feature_blocks: tuple[FeatureBlock, ...]
    repeats: int = 5

    def __post_init__(self) -> None:
        if not self.component:
            raise ValueError("Importance component must not be empty")
        if self.repeats < 1:
            raise ValueError("Importance repeats must be at least one")
        if not self.feature_blocks:
            raise ValueError("Importance requires at least one feature block")
        if self.metric.optimization_direction not in {"minimize", "maximize"}:
            raise ValueError(
                "Importance metrics must have minimize or maximize direction"
            )


@dataclass(frozen=True)
class CandidateDefinition:
    """Explicit configuration and factory for one auditable model candidate."""

    candidate_id: str
    approach: ApproachName
    model_factory: Callable[[], BaseAgeGroupModel] = field(repr=False, compare=False)
    fit_feature_spec: FeatureSpec
    component_feature_specs: tuple[FeatureSpec, ...]
    metrics: tuple[Metric, ...]
    configuration: Mapping[str, Any]
    prediction_config: PredictionConfig = field(default_factory=PredictionConfig)
    selection_role: SelectionRole = "eligible"
    importance_specs: tuple[PermutationImportanceSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("Candidate ID must not be empty")
        if self.approach not in _APPROACH_NAMES:
            raise ValueError(f"Unknown model approach: {self.approach}")
        if self.selection_role not in {"eligible", "diagnostic_comparator"}:
            raise ValueError(f"Unknown selection role: {self.selection_role}")
        if not self.component_feature_specs:
            raise ValueError("Candidates must declare their component feature specs")
        if self.fit_feature_spec not in self.component_feature_specs:
            raise ValueError("Fit feature spec must be among component feature specs")
        components = [spec.component for spec in self.component_feature_specs]
        if len(components) != len(set(components)):
            raise ValueError("Candidate feature-spec components must be unique")
        if not self.metrics:
            raise ValueError("Candidates must declare at least one metric")
        metric_keys = [
            (metric.name, metric.target, metric.aggregation_level)
            for metric in self.metrics
        ]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError(
                "Candidate metrics must have unique name, target, and aggregation keys"
            )
        configuration = dict(self.configuration)
        json.dumps(configuration)
        object.__setattr__(self, "configuration", MappingProxyType(configuration))

    def to_descriptor(self) -> dict[str, Any]:
        """Return the serializable candidate identity without its runtime factory."""
        return {
            "candidate_id": self.candidate_id,
            "approach": self.approach,
            "selection_role": self.selection_role,
            "fit_feature_spec": asdict(self.fit_feature_spec),
            "component_feature_specs": [
                asdict(spec) for spec in self.component_feature_specs
            ],
            "configuration": dict(self.configuration),
            "prediction_config": asdict(self.prediction_config),
            "metrics": [_metric_descriptor(metric) for metric in self.metrics],
            "importance": [
                {
                    "component": importance.component,
                    "metric": _metric_descriptor(importance.metric),
                    "feature_blocks": [asdict(block) for block in importance.feature_blocks],
                    "repeats": importance.repeats,
                }
                for importance in self.importance_specs
            ],
        }


@dataclass(frozen=True)
class MetricReference:
    """Stable metric-table key used by one selection criterion."""

    metric_name: str
    target: str
    aggregation_level: str = "building"

    def __post_init__(self) -> None:
        if not self.metric_name or not self.target or not self.aggregation_level:
            raise ValueError(
                "Metric references require a name, target, and aggregation level"
            )

    @classmethod
    def from_metric(cls, metric: Metric) -> MetricReference:
        """Derive a reference from a metric so the three keys cannot drift apart.

        Aggregation levels are declared by the metric class (for example
        ``composition_log_loss`` scores per child, not per building), so building
        references by hand silently produces keys that match no metric row.
        """
        return cls(metric.name, metric.target, metric.aggregation_level)


@dataclass(frozen=True)
class SelectionCriterion:
    """One ordered selection criterion composed from visible fold metrics."""

    name: str
    metric_references: tuple[MetricReference, ...]
    optimization_direction: OptimizationDirection
    reduction: Literal["mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        if not self.name or not self.metric_references:
            raise ValueError("Selection criteria require a name and metric references")
        if self.optimization_direction not in {"minimize", "maximize"}:
            raise ValueError("Selection direction must be minimize or maximize")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("Selection reduction must be mean or sum")


@dataclass(frozen=True)
class SelectionPolicy:
    """Ordered within-approach selection criteria and likelihood semantics."""

    approach: ApproachName
    criteria: tuple[SelectionCriterion, ...]
    likelihood_comparability: str

    def __post_init__(self) -> None:
        if self.approach not in _APPROACH_NAMES:
            raise ValueError(f"Unknown model approach: {self.approach}")
        if not self.criteria or not self.likelihood_comparability:
            raise ValueError("Selection policy requires criteria and comparability text")


def _metric_descriptor(metric: Metric) -> dict[str, Any]:
    descriptor = asdict(metric) if hasattr(metric, "__dataclass_fields__") else {}
    descriptor.update(
        {
            "name": metric.name,
            "target": metric.target,
            "required_capability": metric.required_capability,
            "optimization_direction": metric.optimization_direction,
            "aggregation_level": metric.aggregation_level,
        }
    )
    json.dumps(descriptor)
    return descriptor
