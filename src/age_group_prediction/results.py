"""Model-independent prediction and evaluation result contracts."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .modeling_config import PredictionValidationConfig

_REQUIRED_METRIC_COLUMNS = (
    "metric_name",
    "value",
    "aggregation_level",
    "sample_count",
    "target",
)

PredictiveFamily = Literal["poisson", "nb2", "normal"]
_VALID_PREDICTIVE_FAMILIES = {"poisson", "nb2", "normal"}
# What the pointwise log-probability keys of one prediction mean.
# ``marginal``: each key scores its own target alone, so keys cannot be summed
# into a joint score. ``sequential_joint``: ``total`` is the marginal total score
# and each cohort key is conditional on the total and the preceding cohorts, so
# the keys sum to the joint score of the total and the full cohort vector.
PointwiseLogProbabilityScope = Literal["marginal", "sequential_joint"]
_VALID_POINTWISE_LOG_PROBABILITY_SCOPES = {"marginal", "sequential_joint"}


def _readonly_array(
    values: object, *, dtype: npt.DTypeLike | None = None
) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _copy_array_mapping(
    values: Mapping[str, np.ndarray] | None,
    *,
    row_count: int,
    allowed_targets: set[str],
    value_name: str,
) -> Mapping[str, np.ndarray] | None:
    """Create a read-only copy of a mapping of arrays, validating row count and allowed targets."""
    if values is None:
        return None
    unknown_targets = set(values).difference(allowed_targets)
    if unknown_targets:
        raise ValueError(f"Unknown {value_name} targets: {sorted(unknown_targets)}")
    copied: dict[str, np.ndarray] = {}
    for target, target_values in values.items():
        array = _readonly_array(target_values, dtype=float)
        if array.ndim == 0 or array.shape[0] != row_count:
            raise ValueError(f"{value_name} for {target} must have one row per building")
        if not np.isfinite(array).all():
            raise ValueError(f"{value_name} for {target} must be finite")
        copied[target] = array
    return MappingProxyType(copied)


def _resolve_positive_parameter(
    values: object,
    *,
    row_count: int,
    parameter_name: str,
) -> np.ndarray:
    """Return a read-only positive per-row parameter vector."""
    if np.isscalar(values):
        try:
            scalar = float(np.asarray(values, dtype=float).item())
        except (TypeError, ValueError) as error:
            raise ValueError(f"{parameter_name} must be finite and positive") from error
        if not np.isfinite(scalar) or scalar <= 0:
            raise ValueError(f"{parameter_name} must be finite and positive")
        return _readonly_array(np.full(row_count, scalar, dtype=float), dtype=float)
    array = _readonly_array(values, dtype=float)
    if array.shape != (row_count,):
        raise ValueError(f"{parameter_name} must be scalar or one value per building")
    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError(f"{parameter_name} must be finite and positive")
    return array


@dataclass(frozen=True)
class ParametricDistributionSpec:
    """Per-target predictive distribution declaration and parameters."""

    family: PredictiveFamily
    dispersion: float | np.ndarray | None = None
    scale: float | np.ndarray | None = None

    def __post_init__(self) -> None:
        # Without this check an unrecognized family reaches a dispatch chain
        # and is scored as Normal, substituting a log density for a log mass.
        if self.family not in _VALID_PREDICTIVE_FAMILIES:
            raise ValueError(
                f"family must be one of {sorted(_VALID_PREDICTIVE_FAMILIES)}; "
                f"got {self.family!r}"
            )


def _copy_parametric_distribution_mapping(
    values: Mapping[str, ParametricDistributionSpec] | None,
    *,
    row_count: int,
    allowed_targets: set[str],
) -> Mapping[str, ParametricDistributionSpec] | None:
    """Create a validated read-only mapping of parametric distribution specs."""
    if values is None:
        return None
    unknown_targets = set(values).difference(allowed_targets)
    if unknown_targets:
        raise ValueError(
            f"Unknown parametric distribution targets: {sorted(unknown_targets)}"
        )
    copied: dict[str, ParametricDistributionSpec] = {}
    for target, spec in values.items():
        if not isinstance(spec, ParametricDistributionSpec):
            raise TypeError(
                "Parametric distribution values must be ParametricDistributionSpec"
            )
        if spec.family == "poisson":
            if spec.dispersion is not None or spec.scale is not None:
                raise ValueError("Poisson distributions must not define dispersion/scale")
            copied[target] = ParametricDistributionSpec(family="poisson")
            continue
        if spec.family == "nb2":
            if spec.scale is not None:
                raise ValueError("NB2 distributions must not define a normal scale")
            if spec.dispersion is None:
                raise ValueError("NB2 distributions require a positive dispersion")
            copied[target] = ParametricDistributionSpec(
                family="nb2",
                dispersion=_resolve_positive_parameter(
                    spec.dispersion,
                    row_count=row_count,
                    parameter_name="NB2 dispersion",
                ),
            )
            continue
        if spec.family == "normal":
            if spec.dispersion is not None:
                raise ValueError("Normal distributions must not define NB2 dispersion")
            if spec.scale is None:
                raise ValueError("Normal distributions require a positive scale")
            copied[target] = ParametricDistributionSpec(
                family="normal",
                scale=_resolve_positive_parameter(
                    spec.scale,
                    row_count=row_count,
                    parameter_name="Normal scale",
                ),
            )
            continue
        # Unreachable while the dataclass validates, and deliberately so: this
        # chain must never absorb an unknown family into the Normal branch.
        raise ValueError(f"Unsupported predictive family: {spec.family!r}")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class PredictionResult:
    """Validated predictions shared by every age-group model family."""

    building_ids: np.ndarray
    cohort_names: tuple[str, ...]
    total_mean: np.ndarray
    cohort_means: np.ndarray
    age_group_probabilities: np.ndarray
    reconciliation_error: np.ndarray
    predictive_draws: Mapping[str, np.ndarray] | None = None
    prediction_intervals: Mapping[str, np.ndarray] | None = None
    interval_levels: tuple[float, ...] = ()
    pointwise_log_probabilities: Mapping[str, np.ndarray] | None = None
    parametric_distributions: Mapping[str, ParametricDistributionSpec] | None = None
    validation_config: PredictionValidationConfig = field(
        default_factory=PredictionValidationConfig
    )
    pointwise_log_probability_scope: PointwiseLogProbabilityScope | None = None

    def __post_init__(self) -> None:
        building_ids = _readonly_array(self.building_ids)
        total_mean = _readonly_array(self.total_mean, dtype=float)
        cohort_means = _readonly_array(self.cohort_means, dtype=float)
        probabilities = _readonly_array(self.age_group_probabilities, dtype=float)
        reconciliation_error = _readonly_array(
            self.reconciliation_error, dtype=float
        )
        row_count = len(building_ids)
        cohort_count = len(self.cohort_names)

        if building_ids.ndim != 1:
            raise ValueError("building_ids must be one-dimensional")
        if not pd.Index(building_ids).is_unique:
            raise ValueError("building_ids must be unique")
        if not self.cohort_names or cohort_count != len(set(self.cohort_names)):
            raise ValueError("cohort_names must be nonempty and unique")
        if total_mean.shape != (row_count,):
            raise ValueError("total_mean must have one value per building")
        expected_matrix_shape = (row_count, cohort_count)
        if cohort_means.shape != expected_matrix_shape:
            raise ValueError("cohort_means shape must match buildings and cohorts")
        if probabilities.shape != expected_matrix_shape:
            raise ValueError(
                "age_group_probabilities shape must match buildings and cohorts"
            )
        if reconciliation_error.shape != (row_count,):
            raise ValueError(
                "reconciliation_error must have one value per building"
            )
        for name, array in (
            ("total_mean", total_mean),
            ("cohort_means", cohort_means),
            ("age_group_probabilities", probabilities),
            ("reconciliation_error", reconciliation_error),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            if (array < 0).any():
                raise ValueError(f"{name} must be nonnegative")
        if (probabilities > 1).any() or not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=self.validation_config.probability_sum_tolerance,
        ):
            raise ValueError("Age-group probabilities must sum to one per building")

        actual_error = np.abs(cohort_means.sum(axis=1) - total_mean)
        if not np.allclose(
            reconciliation_error,
            actual_error,
            rtol=0.0,
            atol=self.validation_config.reported_error_tolerance,
        ):
            raise ValueError("reconciliation_error does not match prediction means")
        if (actual_error > self.validation_config.reconciliation_tolerance).any():
            raise ValueError("Cohort and total means do not reconcile")

        allowed_targets = {"total", *self.cohort_names}
        predictive_draws = _copy_array_mapping(
            self.predictive_draws,
            row_count=row_count,
            allowed_targets=allowed_targets,
            value_name="predictive draws",
        )
        log_probabilities = _copy_array_mapping(
            self.pointwise_log_probabilities,
            row_count=row_count,
            allowed_targets=allowed_targets,
            value_name="pointwise log probabilities",
        )
        scope = self.pointwise_log_probability_scope
        if scope is not None:
            if scope not in _VALID_POINTWISE_LOG_PROBABILITY_SCOPES:
                raise ValueError(
                    "pointwise_log_probability_scope must be one of "
                    f"{sorted(_VALID_POINTWISE_LOG_PROBABILITY_SCOPES)}"
                )
            if log_probabilities is None:
                raise ValueError(
                    "pointwise_log_probability_scope requires pointwise log probabilities"
                )
            if scope == "sequential_joint":
                if set(log_probabilities) != allowed_targets:
                    raise ValueError(
                        "A sequential_joint scope requires pointwise log "
                        "probabilities for the total and every cohort"
                    )
                if len({values.shape for values in log_probabilities.values()}) != 1:
                    raise ValueError(
                        "A sequential_joint scope requires every pointwise log "
                        "probability entry to share one shape"
                    )
        intervals = _copy_array_mapping(
            self.prediction_intervals,
            row_count=row_count,
            allowed_targets=allowed_targets,
            value_name="prediction intervals",
        )
        parametric_distributions = _copy_parametric_distribution_mapping(
            self.parametric_distributions,
            row_count=row_count,
            allowed_targets=allowed_targets,
        )
        if len(self.interval_levels) != len(set(self.interval_levels)) or any(
            not 0 < level < 1 for level in self.interval_levels
        ):
            raise ValueError("Interval levels must be unique and between zero and one")
        if intervals is None and self.interval_levels:
            raise ValueError("Interval levels require prediction intervals")
        if intervals is not None:
            expected_interval_shape = (row_count, len(self.interval_levels), 2)
            for target, target_intervals in intervals.items():
                if target_intervals.shape != expected_interval_shape:
                    raise ValueError(
                        f"Prediction intervals for {target} have an invalid shape"
                    )
                if (target_intervals[..., 0] > target_intervals[..., 1]).any():
                    raise ValueError(
                        f"Prediction interval lower bounds exceed upper bounds for {target}"
                    )

        object.__setattr__(self, "building_ids", building_ids)
        object.__setattr__(self, "total_mean", total_mean)
        object.__setattr__(self, "cohort_means", cohort_means)
        object.__setattr__(self, "age_group_probabilities", probabilities)
        object.__setattr__(self, "reconciliation_error", reconciliation_error)
        object.__setattr__(self, "predictive_draws", predictive_draws)
        object.__setattr__(self, "prediction_intervals", intervals)
        object.__setattr__(self, "pointwise_log_probabilities", log_probabilities)
        object.__setattr__(
            self,
            "parametric_distributions",
            parametric_distributions,
        )

    @classmethod
    def from_means(
        cls,
        *,
        building_ids: object,
        cohort_names: tuple[str, ...],
        total_mean: object,
        cohort_means: object,
        predictive_draws: Mapping[str, np.ndarray] | None = None,
        prediction_intervals: Mapping[str, np.ndarray] | None = None,
        interval_levels: tuple[float, ...] = (),
        pointwise_log_probabilities: Mapping[str, np.ndarray] | None = None,
        parametric_distributions: Mapping[str, ParametricDistributionSpec] | None = None,
        validation_config: PredictionValidationConfig | None = None,
        pointwise_log_probability_scope: PointwiseLogProbabilityScope | None = None,
    ) -> PredictionResult:
        """Construct a result and derive probabilities and reconciliation."""
        total_array = np.asarray(total_mean, dtype=float)
        cohort_array = np.asarray(cohort_means, dtype=float)
        if cohort_array.ndim != 2:
            raise ValueError("cohort_means must be two-dimensional")
        cohort_sums = cohort_array.sum(axis=1)
        resolved_validation = validation_config or PredictionValidationConfig()
        probabilities = np.full(
            cohort_array.shape,
            1.0 / cohort_array.shape[1],
            dtype=float,
        )
        positive = cohort_sums > 0
        probabilities[positive] = cohort_array[positive] / cohort_sums[positive, None]
        if (
            resolved_validation.zero_total_probability_policy != "uniform"
            and (~positive).any()
        ):
            raise ValueError("Unsupported zero-total probability policy")
        return cls(
            building_ids=np.asarray(building_ids),
            cohort_names=cohort_names,
            total_mean=total_array,
            cohort_means=cohort_array,
            age_group_probabilities=probabilities,
            reconciliation_error=np.abs(cohort_sums - total_array),
            predictive_draws=predictive_draws,
            prediction_intervals=prediction_intervals,
            interval_levels=interval_levels,
            pointwise_log_probabilities=pointwise_log_probabilities,
            parametric_distributions=parametric_distributions,
            validation_config=resolved_validation,
            pointwise_log_probability_scope=pointwise_log_probability_scope,
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Metric table and serializable context produced by model evaluation."""

    metrics_df: pd.DataFrame
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        missing = set(_REQUIRED_METRIC_COLUMNS).difference(self.metrics_df.columns)
        if missing:
            raise ValueError(f"Missing evaluation metric columns: {sorted(missing)}")
        metrics_df = self.metrics_df.loc[:, list(_REQUIRED_METRIC_COLUMNS)].copy(
            deep=True
        )
        if (metrics_df["sample_count"] < 0).any():
            raise ValueError("Metric sample counts must be nonnegative")
        diagnostics = copy.deepcopy(dict(self.diagnostics))
        metadata = copy.deepcopy(dict(self.metadata))
        json.dumps({"diagnostics": diagnostics, "metadata": metadata})
        object.__setattr__(self, "metrics_df", metrics_df)
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))