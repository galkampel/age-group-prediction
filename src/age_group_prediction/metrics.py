"""Typed metrics over observed outcomes and shared prediction results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import kstest, nbinom, norm, poisson

# Closed-form metrics below are computed directly: inputs are already validated
# here, so sklearn's generic input checking would dominate bootstrap runtime.
from sklearn.metrics import mean_poisson_deviance, r2_score

from .distributions import nb2_scipy_parameters, pointwise_log_probability
from .modeling_config import EvaluationConfig, PredictionValidationConfig
from .results import ParametricDistributionSpec, PredictionResult

OptimizationDirection = Literal["minimize", "maximize", "target_zero"]
PredictionCapability = Literal[
    "means",
    "probabilities",
    "pointwise_log_probabilities",
    "joint_pointwise_log_probabilities",
    "predictive_draws",
    "prediction_intervals",
    "parametric_distributions",
    "reconciliation",
]


_JSON_SCALARS = (str, int, float, bool)


def available_prediction_capabilities(
    prediction: PredictionResult,
) -> frozenset[PredictionCapability]:
    """Return the payload capabilities exposed by one prediction result."""
    capabilities: set[PredictionCapability] = {
        "means",
        "probabilities",
        "reconciliation",
    }
    if prediction.pointwise_log_probabilities is not None:
        capabilities.add("pointwise_log_probabilities")
    # Only keys that decompose one joint score can be summed. Marginal or
    # undeclared per-target scores must not be offered to a joint metric.
    if prediction.pointwise_log_probability_scope == "sequential_joint":
        capabilities.add("joint_pointwise_log_probabilities")
    if prediction.predictive_draws is not None:
        capabilities.add("predictive_draws")
    if prediction.prediction_intervals is not None:
        capabilities.add("prediction_intervals")
    if prediction.parametric_distributions is not None:
        capabilities.add("parametric_distributions")
    return frozenset(capabilities)


def _json_safe_copy(value: Any) -> Any:
    """Copy metadata and prove JSON-serializability in a single traversal."""
    if value is None or isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, Mapping):
        copied: dict[Any, Any] = {}
        for key, item in value.items():
            if key is not None and not isinstance(key, _JSON_SCALARS):
                raise TypeError(
                    f"Metric metadata keys must be JSON-safe: {type(key).__name__}"
                )
            copied[key] = _json_safe_copy(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(item) for item in value]
    raise TypeError(
        f"Metric metadata is not JSON-serializable: {type(value).__name__}"
    )


@dataclass(frozen=True)
class MetricResult:
    """One scalar metric value with JSON-serializable diagnostic context."""

    metric_name: str
    value: float
    target: str
    aggregation_level: str
    sample_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze JSON-serializable metric metadata."""
        if not self.metric_name:
            raise ValueError("metric_name must not be empty")
        if not np.isfinite(self.value):
            raise ValueError("Metric value must be finite")
        if self.sample_count < 0:
            raise ValueError("Metric sample_count must be nonnegative")
        metadata = _json_safe_copy(dict(self.metadata))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_record(self) -> dict[str, object]:
        """Return the columns required by the shared evaluation result."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "aggregation_level": self.aggregation_level,
            "sample_count": self.sample_count,
            "target": self.target,
        }


@runtime_checkable
class Metric(Protocol):
    """Immutable scorer that consumes outcomes and a prediction result."""

    @property
    def name(self) -> str:
        """Metric name used in evaluation outputs."""
        ...

    @property
    def target(self) -> str:
        """Observed/predicted target that this metric scores."""
        ...

    @property
    def required_capability(self) -> PredictionCapability:
        """Prediction payload requirement for this metric."""
        ...

    @property
    def optimization_direction(self) -> OptimizationDirection:
        """Whether lower, higher, or zero-centered values are preferred."""
        ...

    @property
    def aggregation_level(self) -> str:
        """Unit at which metric contributions are aggregated."""
        ...

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute one aggregate score without mutating predictions."""
        ...


def _validate_observed(
    observed: pd.DataFrame,
    prediction: PredictionResult,
    target: str,
) -> np.ndarray:
    """Return one nonnegative finite observed target vector aligned to predictions."""
    if len(observed) != len(prediction.building_ids):
        raise ValueError("Observed rows must match prediction rows")
    if target not in observed:
        raise ValueError(f"Missing observed target column: {target}")
    values = observed[target].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Observed target {target} must be finite and nonnegative")
    return values


def _predicted_mean(prediction: PredictionResult, target: str) -> np.ndarray:
    """Return a predicted mean vector for total or cohort targets."""
    if target == "total" or target == "n_children_total":
        return prediction.total_mean
    try:
        cohort_index = prediction.cohort_names.index(target)
    except ValueError as error:
        raise ValueError(f"Unknown prediction target: {target}") from error
    return prediction.cohort_means[:, cohort_index]


def _prediction_payload_key(target: str) -> str:
    """Map observed total-column names to prediction payload keys."""
    return "total" if target == "n_children_total" else target


def _require_concrete_variant(metric: object) -> None:
    """Reject instantiating a source-dispatching base instead of a variant.

    The base classes carry the shared identity and reduction; only the concrete
    variants declare a ``required_capability`` and a scoring path, which is what
    keeps the two from disagreeing.
    """
    if not hasattr(metric, "required_capability"):
        raise TypeError(
            f"{type(metric).__name__} is abstract; construct a concrete variant"
        )


def _metric_result(
    metric: Metric,
    value: float | np.floating[Any],
    sample_count: int,
    *,
    metadata: dict[str, Any] | None = None,
) -> MetricResult:
    """Construct one immutable metric result with optional metadata."""
    return MetricResult(
        metric_name=metric.name,
        value=float(value),
        target=metric.target,
        aggregation_level=metric.aggregation_level,
        sample_count=sample_count,
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class MeanAbsoluteError:
    """Building-level mean absolute error over predicted means."""

    target: str
    name: str = field(default="mae", init=False)
    required_capability: PredictionCapability = field(default="means", init=False)
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute the mean absolute error of predicted means."""
        del rng
        actual = _validate_observed(observed, prediction, self.target)
        value = np.mean(np.abs(_predicted_mean(prediction, self.target) - actual))
        return _metric_result(self, value, len(actual))


@dataclass(frozen=True)
class RootMeanSquaredError:
    """Building-level root mean squared error over predicted means."""

    target: str
    name: str = field(default="rmse", init=False)
    required_capability: PredictionCapability = field(default="means", init=False)
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute the root mean squared error of predicted means."""
        del rng
        actual = _validate_observed(observed, prediction, self.target)
        residual = _predicted_mean(prediction, self.target) - actual
        return _metric_result(self, np.sqrt(np.mean(residual**2)), len(actual))


@dataclass(frozen=True)
class MeanBias:
    """Building-level signed mean error over predicted means."""

    target: str
    name: str = field(default="mean_bias", init=False)
    required_capability: PredictionCapability = field(default="means", init=False)
    optimization_direction: OptimizationDirection = field(
        default="target_zero", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute the average signed residual (predicted minus observed)."""
        del rng
        actual = _validate_observed(observed, prediction, self.target)
        value = np.mean(_predicted_mean(prediction, self.target) - actual)
        return _metric_result(self, value, len(actual))


@dataclass(frozen=True)
class R2:
    """Building-level coefficient of determination for context reporting."""

    target: str
    name: str = field(default="r2", init=False)
    required_capability: PredictionCapability = field(default="means", init=False)
    optimization_direction: OptimizationDirection = field(
        default="maximize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute $R^2$ using the scikit-learn implementation."""
        del rng
        actual = _validate_observed(observed, prediction, self.target)
        value = r2_score(actual, _predicted_mean(prediction, self.target))
        return _metric_result(self, value, len(actual))


@dataclass(frozen=True)
class MeanPoissonDeviance:
    """Building-level mean Poisson deviance with positive-mean clipping."""

    target: str
    minimum_mean: float
    name: str = field(default="mean_poisson_deviance", init=False)
    required_capability: PredictionCapability = field(default="means", init=False)
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def __post_init__(self) -> None:
        """Require a finite positive clipping floor for predicted means."""
        if not np.isfinite(self.minimum_mean) or self.minimum_mean <= 0:
            raise ValueError("minimum_mean must be finite and positive")

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute mean Poisson deviance through scikit-learn."""
        del rng
        actual = _validate_observed(observed, prediction, self.target)
        predicted = np.clip(
            _predicted_mean(prediction, self.target), self.minimum_mean, None
        )
        return _metric_result(
            self, mean_poisson_deviance(actual, predicted), len(actual)
        )


def _require_parametric_distribution(
    prediction: PredictionResult,
    target: str,
) -> ParametricDistributionSpec:
    """Return the declared parametric predictive family for one target."""
    distributions = prediction.parametric_distributions
    payload_key = _prediction_payload_key(target)
    if distributions is None or payload_key not in distributions:
        raise ValueError(f"Missing parametric distribution for {target}")
    return distributions[payload_key]


def _nb2_r_and_p(
    mean: np.ndarray,
    dispersion: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert NB2 mean/dispersion parameters to SciPy ``(r, p)`` form."""
    return nb2_scipy_parameters(mean, dispersion)


@dataclass(frozen=True)
class PredictiveNegativeLogLikelihood:
    """Shared identity and reduction for the predictive NLL variants.

    The scoring source is a class-level constant per variant rather than a field
    branched on at runtime, so the declared capability and the scoring path
    cannot disagree. The metric-table key (name, target, aggregation level) is
    deliberately identical across variants: it identifies the score, not how the
    model supplied it.
    """

    target: str
    interpretation: str
    name: str = field(default="predictive_nll", init=False)
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def __post_init__(self) -> None:
        """Require a concrete variant and a non-empty interpretation."""
        _require_concrete_variant(self)
        if not self.interpretation:
            raise ValueError("NLL interpretation must not be empty")

    def _log_probabilities(
        self, actual: np.ndarray, prediction: PredictionResult
    ) -> tuple[np.ndarray, str, str]:
        """Return pointwise log probabilities, integration label, and family."""
        raise NotImplementedError

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute mean NLL with variant-specific integration metadata."""
        del rng
        actual = _validate_observed(observed, prediction, self.target)
        log_probabilities, integration, family = self._log_probabilities(
            actual, prediction
        )
        return _metric_result(
            self,
            -np.mean(log_probabilities),
            len(actual),
            metadata={
                "interpretation": self.interpretation,
                "integration": integration,
                "family": family,
            },
        )


@dataclass(frozen=True)
class ParametricPredictiveNegativeLogLikelihood(PredictiveNegativeLogLikelihood):
    """Mean predictive NLL evaluated from declared parametric families."""

    source: str = field(default="parametric", init=False)
    required_capability: PredictionCapability = field(
        default="parametric_distributions", init=False
    )

    def _log_probabilities(
        self, actual: np.ndarray, prediction: PredictionResult
    ) -> tuple[np.ndarray, str, str]:
        """Score the observation under the declared predictive family."""
        spec = _require_parametric_distribution(prediction, self.target)
        mean = _predicted_mean(prediction, self.target)
        if spec.family == "poisson":
            if not np.equal(actual, np.floor(actual)).all():
                raise ValueError("Poisson NLL requires integer observed counts")
            log_probabilities = pointwise_log_probability("poisson", actual, mean)
        elif spec.family == "nb2":
            if not np.equal(actual, np.floor(actual)).all():
                raise ValueError("NB2 NLL requires integer observed counts")
            assert spec.dispersion is not None
            log_probabilities = pointwise_log_probability(
                "nb2", actual, mean, dispersion=spec.dispersion
            )
        elif spec.family == "normal":
            assert spec.scale is not None
            log_probabilities = pointwise_log_probability(
                "normal", actual, mean, scale=spec.scale
            )
        else:
            # Never silently score an unknown family as Normal: that swaps a
            # log density for a log mass and the two cannot be ranked together.
            raise ValueError(f"Unsupported predictive family: {spec.family!r}")
        return log_probabilities, "parametric", spec.family


@dataclass(frozen=True)
class PointwisePredictiveNegativeLogLikelihood(PredictiveNegativeLogLikelihood):
    """Mean predictive NLL from log probabilities the model supplied directly."""

    source: str = field(default="pointwise", init=False)
    required_capability: PredictionCapability = field(
        default="pointwise_log_probabilities", init=False
    )

    def _log_probabilities(
        self, actual: np.ndarray, prediction: PredictionResult
    ) -> tuple[np.ndarray, str, str]:
        """Read supplied values, integrating over any draw axis."""
        del actual
        values = prediction.pointwise_log_probabilities
        payload_key = _prediction_payload_key(self.target)
        if values is None or payload_key not in values:
            raise ValueError(f"Missing pointwise log probabilities for {self.target}")
        supplied = np.asarray(values[payload_key], dtype=float)
        if supplied.ndim == 1:
            return supplied, "pointwise", "generic"
        if supplied.ndim == 2:
            if supplied.shape[1] < 1:
                raise ValueError("Pointwise log probabilities must contain draws")
            log_probabilities = logsumexp(supplied, axis=1) - np.log(supplied.shape[1])
            return log_probabilities, "posterior-log-mean-exp", "generic"
        raise ValueError("Pointwise log probabilities must be 1D or 2D")


@dataclass(frozen=True)
class JointPredictiveNegativeLogLikelihood:
    r"""Mean negative joint log score of each building's total and cohort vector.

    The joint score $\log p(Y_b, \mathbf C_b)$ is the sum of a prediction's
    pointwise entries when its scope is ``"sequential_joint"``: a marginal total
    score plus cohort scores conditional on the total and the preceding
    cohorts. The sum does not depend on cohort order, although the individual
    cohort entries do. Marginal per-target scores, such as independent cohort
    likelihoods, cannot be summed into a joint score, so predictions that do
    not declare a sequential-joint scope lack the required capability.

    Entries with a draw axis are summed within each draw before the stable
    log-mean-exp, because the per-draw joint score is the sum of that draw's
    entries while the log-mean-exp of a sum is not the sum of log-mean-exps.
    """

    interpretation: str
    target: str = field(default="joint", init=False)
    name: str = field(default="joint_predictive_nll", init=False)
    required_capability: PredictionCapability = field(
        default="joint_pointwise_log_probabilities", init=False
    )
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def __post_init__(self) -> None:
        """Require a non-empty description of the scored joint distribution."""
        if not self.interpretation:
            raise ValueError("NLL interpretation must not be empty")

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Sum the sequential-joint entries per building and average the NLL."""
        del rng
        if len(observed) != len(prediction.building_ids):
            raise ValueError("Observed rows must match prediction rows")
        values = prediction.pointwise_log_probabilities
        if (
            prediction.pointwise_log_probability_scope != "sequential_joint"
            or values is None
        ):
            raise ValueError(
                "Joint predictive NLL requires pointwise log probabilities with a "
                "'sequential_joint' scope"
            )
        keys = ("total", *prediction.cohort_names)
        joint = np.sum([np.asarray(values[key], dtype=float) for key in keys], axis=0)
        if joint.ndim == 1:
            log_scores = joint
            integration = "pointwise"
        elif joint.ndim == 2:
            if joint.shape[1] < 1:
                raise ValueError("Pointwise log probabilities must contain draws")
            log_scores = logsumexp(joint, axis=1) - np.log(joint.shape[1])
            integration = "posterior-log-mean-exp"
        else:
            raise ValueError("Pointwise log probabilities must be 1D or 2D")
        return _metric_result(
            self,
            -np.mean(log_scores),
            len(log_scores),
            metadata={
                "interpretation": self.interpretation,
                "integration": integration,
                "scope": "sequential_joint",
                "summed_keys": list(keys),
            },
        )


def _observed_cohort_counts(
    observed: pd.DataFrame, prediction: PredictionResult
) -> np.ndarray:
    """Return finite nonnegative observed cohort counts aligned to predictions."""
    missing = set(prediction.cohort_names).difference(observed.columns)
    if missing:
        raise ValueError(f"Missing observed cohort columns: {sorted(missing)}")
    counts = observed.loc[:, list(prediction.cohort_names)].to_numpy(dtype=float)
    if len(counts) != len(prediction.building_ids):
        raise ValueError("Observed rows must match prediction rows")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Observed cohort counts must be finite and nonnegative")
    return counts


def _composition_arrays(
    counts: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned float cohort counts and matching per-cohort values."""
    counts = np.asarray(counts, dtype=float)
    values = np.asarray(values, dtype=float)
    if counts.shape != values.shape:
        raise ValueError("Composition counts and per-cohort values must share a shape")
    return counts, values


def _positive_child_total(counts: np.ndarray) -> float:
    """Return the total observed child weight, requiring it to be positive."""
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("Composition metrics require positive observed totals")
    return total


def composition_log_loss(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute child-weighted multiclass cross-entropy from cohort probabilities."""
    counts, probabilities = _composition_arrays(counts, probabilities)
    total = _positive_child_total(counts)
    positive = counts > 0
    if (probabilities[positive] <= 0).any():
        raise ValueError("Positive observed cohorts require positive probabilities")
    return -float(np.sum(counts[positive] * np.log(probabilities[positive])) / total)


def composition_log_loss_from_log_probabilities(
    counts: np.ndarray, log_probabilities: np.ndarray
) -> float:
    """Compute child-weighted cross-entropy from normalized log probabilities."""
    counts, log_probabilities = _composition_arrays(counts, log_probabilities)
    total = _positive_child_total(counts)
    positive = counts > 0
    return -float(np.sum(counts[positive] * log_probabilities[positive]) / total)


def composition_brier_score(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute the per-child multiclass Brier score on the ``[0, 2]`` scale."""
    counts, probabilities = _composition_arrays(counts, probabilities)
    row_totals = counts.sum(axis=1)
    total = _positive_child_total(row_totals)
    score = np.sum(
        counts * (1.0 - probabilities) ** 2
        + (row_totals[:, None] - counts) * probabilities**2
    )
    return float(score / total)


@dataclass(frozen=True)
class CompositionLogLoss:
    """Child-weighted multiclass cross-entropy over cohort probabilities."""

    target: str = field(default="composition", init=False)
    name: str = field(default="composition_log_loss", init=False)
    required_capability: PredictionCapability = field(
        default="probabilities", init=False
    )
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="child", init=False)

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute count-weighted multiclass log loss over cohort probabilities."""
        del rng
        counts = _observed_cohort_counts(observed, prediction)
        value = composition_log_loss(counts, prediction.age_group_probabilities)
        return _metric_result(self, value, len(counts))


@dataclass(frozen=True)
class CompositionBrierScore:
    """Child-weighted multiclass Brier score over cohort probabilities."""

    target: str = field(default="composition", init=False)
    name: str = field(default="composition_brier", init=False)
    required_capability: PredictionCapability = field(
        default="probabilities", init=False
    )
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="child", init=False)

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute count-weighted multiclass Brier score on the [0, 2] scale."""
        del rng
        counts = _observed_cohort_counts(observed, prediction)
        value = composition_brier_score(counts, prediction.age_group_probabilities)
        return _metric_result(self, value, len(counts))


@dataclass(frozen=True)
class ReconciliationError:
    """Accounting diagnostic on cohort-to-total mean consistency per building."""

    statistic: Literal["mean", "max", "count_above_tolerance"]
    tolerance: float = 0.0
    target: str = field(default="all", init=False)
    required_capability: PredictionCapability = field(
        default="reconciliation", init=False
    )
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def __post_init__(self) -> None:
        """Validate supported statistic names and finite tolerance."""
        if self.statistic not in {"mean", "max", "count_above_tolerance"}:
            raise ValueError("Unknown reconciliation statistic")
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")

    @property
    def name(self) -> str:
        """Return a stable metric identifier for the selected statistic."""
        if self.statistic == "count_above_tolerance":
            return "reconciliation_count_above_tolerance"
        return f"{self.statistic}_reconciliation_error"

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Aggregate per-building reconciliation errors for diagnostics."""
        del observed, rng
        errors = prediction.reconciliation_error
        count_above = int(np.sum(errors > self.tolerance))
        if self.statistic == "mean":
            value = np.mean(errors)
        elif self.statistic == "max":
            value = np.max(errors)
        else:
            value = float(count_above)
        return _metric_result(
            self,
            value,
            len(prediction.building_ids),
            metadata={
                "tolerance": self.tolerance,
                "count_above_tolerance": count_above,
            },
        )


def _prediction_interval(
    prediction: PredictionResult, target: str, level: float
) -> np.ndarray:
    """Return one target's prediction interval matrix for a requested level."""
    intervals = prediction.prediction_intervals
    payload_key = _prediction_payload_key(target)
    if intervals is None or payload_key not in intervals:
        raise ValueError(f"Missing prediction intervals for {target}")
    matching = [
        index
        for index, candidate in enumerate(prediction.interval_levels)
        if np.isclose(candidate, level, rtol=0.0, atol=1e-12)
    ]
    if len(matching) != 1:
        raise ValueError(f"Prediction interval level {level} is unavailable")
    return intervals[payload_key][:, matching[0], :]


@dataclass(frozen=True)
class IntervalCoverage:
    """Empirical interval coverage at one nominal probability level."""

    target: str
    level: float
    name: str = field(default="interval_coverage", init=False)
    required_capability: PredictionCapability = field(
        default="prediction_intervals", init=False
    )
    optimization_direction: OptimizationDirection = field(
        default="maximize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def __post_init__(self) -> None:
        """Require a nominal level strictly inside $(0,1)$."""
        if not 0 < self.level < 1:
            raise ValueError("Interval coverage level must be between zero and one")

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute the fraction of observations inside predicted intervals."""
        del rng
        actual = _validate_observed(observed, prediction, self.target)
        interval = _prediction_interval(prediction, self.target, self.level)
        covered = (actual >= interval[:, 0]) & (actual <= interval[:, 1])
        return _metric_result(
            self,
            np.mean(covered),
            len(actual),
            metadata={"interval_level": self.level},
        )


@dataclass(frozen=True)
class MeanIntervalWidth:
    """Average width of predictive intervals at one nominal level."""

    target: str
    level: float
    name: str = field(default="mean_interval_width", init=False)
    required_capability: PredictionCapability = field(
        default="prediction_intervals", init=False
    )
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute the mean upper-minus-lower predictive interval width."""
        del observed, rng
        interval = _prediction_interval(prediction, self.target, self.level)
        return _metric_result(
            self,
            np.mean(interval[:, 1] - interval[:, 0]),
            len(interval),
            metadata={"interval_level": self.level},
        )


@dataclass(frozen=True)
class WeightedIntervalScore:
    """Weighted interval score (WIS) for one predictive interval level."""

    target: str
    level: float
    name: str = field(default="weighted_interval_score", init=False)
    required_capability: PredictionCapability = field(
        default="prediction_intervals", init=False
    )
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def __post_init__(self) -> None:
        """Require a nominal level strictly inside $(0,1)$."""
        if not 0 < self.level < 1:
            raise ValueError(
                "Weighted interval score level must be between zero and one"
            )

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute one-level WIS, equal to pinball losses at alpha/2 and 1-alpha/2."""
        del rng
        actual = _validate_observed(observed, prediction, self.target)
        interval = _prediction_interval(prediction, self.target, self.level)
        lower = interval[:, 0]
        upper = interval[:, 1]
        alpha = 1.0 - self.level
        interval_score = (
            upper
            - lower
            + (2.0 / alpha) * (lower - actual) * (actual < lower)
            + (2.0 / alpha) * (actual - upper) * (actual > upper)
        )
        return _metric_result(
            self,
            np.mean((alpha / 2.0) * interval_score),
            len(actual),
            metadata={"interval_level": self.level},
        )


@dataclass(frozen=True)
class RandomizedPIT:
    """Shared randomization and KS reduction for the randomized PIT variants.

    As with the NLL variants, the source is a per-class constant so the declared
    capability and the PIT construction cannot drift apart.
    """

    target: str
    default_seed: int
    histogram_bins: int
    name: str = field(default="randomized_pit_ks", init=False)
    optimization_direction: OptimizationDirection = field(
        default="minimize", init=False
    )
    aggregation_level: str = field(default="building", init=False)

    def __post_init__(self) -> None:
        """Require a concrete variant, RNG defaults, and histogram resolution."""
        _require_concrete_variant(self)
        if self.default_seed < 0:
            raise ValueError("default_seed must be nonnegative")
        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least two")

    def _pit_values(
        self,
        actual: np.ndarray,
        prediction: PredictionResult,
        uniforms: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Return randomized PIT values and the family label."""
        raise NotImplementedError

    def compute(
        self,
        observed: pd.DataFrame,
        prediction: PredictionResult,
        *,
        rng: np.random.Generator | None = None,
    ) -> MetricResult:
        """Compute randomized PIT values and a uniformity KS diagnostic."""
        actual = _validate_observed(observed, prediction, self.target)
        resolved_rng = np.random.default_rng(self.default_seed) if rng is None else rng
        uniforms = resolved_rng.uniform(size=len(actual))
        pit, family = self._pit_values(actual, prediction, uniforms)
        pit = np.clip(pit, 0.0, 1.0)
        statistic, p_value = kstest(pit, "uniform")
        counts, edges = np.histogram(pit, bins=self.histogram_bins, range=(0.0, 1.0))
        return _metric_result(
            self,
            statistic,
            len(actual),
            metadata={
                "p_value": float(p_value),
                "pit_values": pit.tolist(),
                "histogram_counts": counts.tolist(),
                "histogram_edges": edges.tolist(),
                "source": self.source,
                "family": family,
            },
        )


@dataclass(frozen=True)
class ParametricRandomizedPIT(RandomizedPIT):
    """Randomized PIT built from declared parametric predictive CDFs."""

    source: str = field(default="parametric", init=False)
    required_capability: PredictionCapability = field(
        default="parametric_distributions", init=False
    )

    def _pit_values(
        self,
        actual: np.ndarray,
        prediction: PredictionResult,
        uniforms: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Randomize within the CDF jump for discrete families."""
        spec = _require_parametric_distribution(prediction, self.target)
        mean = _predicted_mean(prediction, self.target)
        if spec.family == "poisson":
            if not np.equal(actual, np.floor(actual)).all():
                raise ValueError("Poisson PIT requires integer observed counts")
            lower = poisson.cdf(actual - 1, mean)
            upper = poisson.cdf(actual, mean)
            pit = lower + uniforms * (upper - lower)
        elif spec.family == "nb2":
            if not np.equal(actual, np.floor(actual)).all():
                raise ValueError("NB2 PIT requires integer observed counts")
            assert spec.dispersion is not None
            r, p = _nb2_r_and_p(mean, spec.dispersion)
            lower = nbinom.cdf(actual - 1, r, p)
            upper = nbinom.cdf(actual, r, p)
            pit = lower + uniforms * (upper - lower)
        elif spec.family == "normal":
            assert spec.scale is not None
            pit = norm.cdf(actual, loc=mean, scale=spec.scale)
        else:
            raise ValueError(f"Unsupported predictive family: {spec.family!r}")
        return pit, spec.family


@dataclass(frozen=True)
class DrawsRandomizedPIT(RandomizedPIT):
    """Randomized PIT estimated from posterior predictive draws."""

    source: str = field(default="draws", init=False)
    required_capability: PredictionCapability = field(
        default="predictive_draws", init=False
    )

    def _pit_values(
        self,
        actual: np.ndarray,
        prediction: PredictionResult,
        uniforms: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Estimate the randomized rank of the observation among draws."""
        draws = prediction.predictive_draws
        payload_key = _prediction_payload_key(self.target)
        if draws is None or payload_key not in draws:
            raise ValueError(f"Missing predictive draws for {self.target}")
        target_draws = np.asarray(draws[payload_key], dtype=float)
        if target_draws.ndim != 2 or target_draws.shape[1] < 1:
            raise ValueError("Predictive draws for PIT must be two-dimensional")
        less = (target_draws < actual[:, None]).sum(axis=1)
        equal = (target_draws == actual[:, None]).sum(axis=1)
        pit = (less + uniforms * equal) / target_draws.shape[1]
        return pit, "draws"


_NLL_VARIANTS: dict[str, type[PredictiveNegativeLogLikelihood]] = {
    "parametric": ParametricPredictiveNegativeLogLikelihood,
    "pointwise": PointwisePredictiveNegativeLogLikelihood,
}
_PIT_VARIANTS: dict[str, type[RandomizedPIT]] = {
    "parametric": ParametricRandomizedPIT,
    "draws": DrawsRandomizedPIT,
}


def default_metric_set(
    targets: Sequence[str],
    *,
    nll_source: Literal["parametric", "pointwise"],
    nll_interpretation: str,
    evaluation_config: EvaluationConfig,
    prediction_validation_config: PredictionValidationConfig,
    interval_levels: Sequence[float] = (),
    pit_source: Literal["parametric", "draws"] | None = None,
    pit_default_seed: int | None = None,
    joint_nll_interpretation: str | None = None,
) -> tuple[Metric, ...]:
    """Build the shared model-comparison metrics in deterministic order.

    Passing ``joint_nll_interpretation`` adds the joint predictive NLL, which
    only predictions with a ``"sequential_joint"`` pointwise scope can supply.
    """
    resolved_targets = tuple(targets)
    resolved_levels = tuple(interval_levels)
    if not resolved_targets or any(not target for target in resolved_targets):
        raise ValueError("Metric targets must be nonempty")
    if len(resolved_targets) != len(set(resolved_targets)):
        raise ValueError("Metric targets must be unique")
    if len(resolved_levels) != len(set(resolved_levels)):
        raise ValueError("Metric interval levels must be unique")
    if pit_source is not None and pit_default_seed is None:
        raise ValueError("PIT metrics require pit_default_seed")
    # This factory is the one place a source string is parsed into a metric
    # class, so an unknown value is rejected here rather than silently
    # selecting a scoring path downstream.
    if nll_source not in _NLL_VARIANTS:
        raise ValueError(
            f"nll_source must be one of {sorted(_NLL_VARIANTS)}; got {nll_source!r}"
        )
    if pit_source is not None and pit_source not in _PIT_VARIANTS:
        raise ValueError(
            f"pit_source must be one of {sorted(_PIT_VARIANTS)}; got {pit_source!r}"
        )

    metrics: list[Metric] = []
    for target in resolved_targets:
        metrics.extend(
            (
                MeanAbsoluteError(target),
                RootMeanSquaredError(target),
                MeanBias(target),
                R2(target),
                MeanPoissonDeviance(
                    target,
                    minimum_mean=evaluation_config.poisson_minimum_mean,
                ),
                _NLL_VARIANTS[nll_source](
                    target=target,
                    interpretation=nll_interpretation,
                ),
            )
        )
    if joint_nll_interpretation is not None:
        metrics.append(
            JointPredictiveNegativeLogLikelihood(interpretation=joint_nll_interpretation)
        )

    metrics.extend((CompositionLogLoss(), CompositionBrierScore()))
    reconciliation_tolerance = (
        prediction_validation_config.reconciliation_tolerance
    )
    metrics.extend(
        (
            ReconciliationError("mean", tolerance=reconciliation_tolerance),
            ReconciliationError("max", tolerance=reconciliation_tolerance),
            ReconciliationError(
                "count_above_tolerance",
                tolerance=reconciliation_tolerance,
            ),
        )
    )

    for target in resolved_targets:
        for level in resolved_levels:
            metrics.extend(
                (
                    IntervalCoverage(target, level),
                    MeanIntervalWidth(target, level),
                    WeightedIntervalScore(target, level),
                )
            )
        if pit_source is not None:
            assert pit_default_seed is not None
            metrics.append(
                _PIT_VARIANTS[pit_source](
                    target=target,
                    default_seed=pit_default_seed,
                    histogram_bins=evaluation_config.pit_histogram_bins,
                )
            )
    return tuple(metrics)
