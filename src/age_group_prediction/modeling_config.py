"""Typed configuration and invariant validation for building-level modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_SEED = 42

FeatureComponent = Literal["tree", "total_count", "age_probability"]
SesForm = Literal["linear", "quadratic", "spline"]
DaycareForm = Literal["linear", "log1p"]
FeatureInteraction = Literal[
    "ses_x_household_size",
    "daycare_x_median_age",
    "room_share_x_household_size",
    "room_share_x_median_age",
]
IntervalMethod = Literal["percentile"]
UnknownCategoryPolicy = Literal["error", "treat_as_reference"]
ZeroTotalProbabilityPolicy = Literal["uniform"]
DirectCohortFamily = Literal["normal", "poisson", "nb2"]
IndependentTotalFamily = Literal["poisson", "nb2"]
BayesianInferenceProfile = Literal["reduced", "full"]
BayesianDiagnosticAction = Literal["warn", "error"]

_VALID_COMPONENTS = {"tree", "total_count", "age_probability"}
_VALID_SES_FORMS = {"linear", "quadratic", "spline"}
_VALID_DAYCARE_FORMS = {"linear", "log1p"}
_VALID_INTERACTIONS = {
    "ses_x_household_size",
    "daycare_x_median_age",
    "room_share_x_household_size",
    "room_share_x_median_age",
}
_VALID_INTERVAL_METHODS = {"percentile"}
_VALID_UNKNOWN_CATEGORY_POLICIES = {"error", "treat_as_reference"}
_VALID_ZERO_TOTAL_POLICIES = {"uniform"}
_VALID_DIRECT_COHORT_FAMILIES = {"normal", "poisson", "nb2"}
_VALID_INDEPENDENT_TOTAL_FAMILIES = {"poisson", "nb2"}
_VALID_BAYESIAN_INFERENCE_PROFILES = {"reduced", "full"}
_VALID_BAYESIAN_DIAGNOSTIC_ACTIONS = {"warn", "error"}


@dataclass(frozen=True)
class OuterSplitConfig:
    """Immutable settings for the one-time outer test-lockbox split."""

    test_fraction: float = 0.2
    building_id_column: str = "building_id"
    neighborhood_id_column: str = "neighborhood_id"
    strategy_version: str = "known-neighborhood-v1"

    def __post_init__(self) -> None:
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be strictly between zero and one")
        if self.building_id_column == self.neighborhood_id_column:
            raise ValueError("Building and neighborhood ID columns must differ")


@dataclass(frozen=True)
class FoldConfig:
    """Immutable settings for train-only validation folds.

    There is deliberately no validation fraction. Folds rotate each
    neighborhood's buildings across ``n_folds``, so the validation share is
    ``1 / n_folds`` by construction and cannot disagree with what was asked
    for.
    """

    n_folds: int = 5
    building_id_column: str = "building_id"
    neighborhood_id_column: str = "neighborhood_id"

    def __post_init__(self) -> None:
        # One fold would make every building its own validation set and leave
        # its neighborhood unrepresented in the fit partition.
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")
        if self.building_id_column == self.neighborhood_id_column:
            raise ValueError("Building and neighborhood ID columns must differ")


DEFAULT_OUTER_SPLIT_CONFIG = OuterSplitConfig()
DEFAULT_FOLD_CONFIG = FoldConfig()


@dataclass(frozen=True)
class RandomnessConfig:
    """Immutable defaults for model APIs that accept optional RNGs."""

    default_seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.default_seed < 0:
            raise ValueError("default_seed must be nonnegative")


@dataclass(frozen=True)
class PredictionConfig:
    """Immutable, model-agnostic prediction output requests."""

    n_predictive_draws: int = 0
    interval_levels: tuple[float, ...] = ()
    include_pointwise_log_probabilities: bool = False

    def __post_init__(self) -> None:
        if self.n_predictive_draws < 0:
            raise ValueError("n_predictive_draws must be nonnegative")
        if len(self.interval_levels) != len(set(self.interval_levels)):
            raise ValueError("Prediction interval levels must be unique")
        if any(not 0 < level < 1 for level in self.interval_levels):
            raise ValueError("Prediction interval levels must be between zero and one")


@dataclass(frozen=True)
class PredictionValidationConfig:
    """Immutable validation policy for shared prediction-result invariants."""

    probability_sum_tolerance: float = 1e-12
    reported_error_tolerance: float = 1e-12
    reconciliation_tolerance: float = 1e-9
    zero_total_probability_policy: ZeroTotalProbabilityPolicy = "uniform"

    def __post_init__(self) -> None:
        for name, value in (
            ("probability_sum_tolerance", self.probability_sum_tolerance),
            ("reported_error_tolerance", self.reported_error_tolerance),
            ("reconciliation_tolerance", self.reconciliation_tolerance),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if self.zero_total_probability_policy not in _VALID_ZERO_TOTAL_POLICIES:
            raise ValueError(
                "zero_total_probability_policy must be one of "
                f"{sorted(_VALID_ZERO_TOTAL_POLICIES)}"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable neighborhood-bootstrap and interval-scoring settings."""

    bootstrap_replicates: int
    confidence_level: float
    neighborhood_id_column: str
    interval_method: IntervalMethod
    max_failed_fraction: float
    poisson_minimum_mean: float
    pit_histogram_bins: int

    def __post_init__(self) -> None:
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be at least one")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if not self.neighborhood_id_column:
            raise ValueError("neighborhood_id_column must not be empty")
        if self.interval_method not in _VALID_INTERVAL_METHODS:
            raise ValueError(
                f"interval_method must be one of {sorted(_VALID_INTERVAL_METHODS)}"
            )
        if not 0 <= self.max_failed_fraction < 1:
            raise ValueError("max_failed_fraction must be in [0, 1)")
        if not np.isfinite(self.poisson_minimum_mean) or self.poisson_minimum_mean <= 0:
            raise ValueError("poisson_minimum_mean must be finite and positive")
        if self.pit_histogram_bins < 2:
            raise ValueError("pit_histogram_bins must be at least two")


@dataclass(frozen=True)
class OptunaTuningConfig:
    """Reproducible Optuna study settings shared by model families."""

    n_trials: int = 30
    n_jobs: int = 1
    timeout_seconds: float | None = None
    enable_pruning: bool = False

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least one")
        if self.n_jobs != 1:
            raise ValueError("Reproducible tuning requires n_jobs=1")
        if self.timeout_seconds is not None:
            raise ValueError("Reproducible tuning does not support timeout stopping")


@dataclass(frozen=True)
class LightGBMSearchSpaceConfig:
    """Bounded search space for direct-cohort LightGBM regressors."""

    num_leaves: tuple[int, int] = (7, 63)
    max_depth: tuple[int, int] = (3, 8)
    min_child_samples: tuple[int, int] = (5, 40)
    learning_rate: tuple[float, float] = (0.01, 0.2)
    n_estimators: tuple[int, int] = (50, 400)
    reg_alpha: tuple[float, float] = (1e-8, 10.0)
    reg_lambda: tuple[float, float] = (1e-8, 10.0)
    min_split_gain: tuple[float, float] = (0.0, 1.0)
    subsample: tuple[float, float] = (0.7, 1.0)
    colsample_bytree: tuple[float, float] = (0.7, 1.0)
    nb2_dispersion: tuple[float, float] = (1e-4, 5.0)

    def __post_init__(self) -> None:
        for name in ("num_leaves", "max_depth", "min_child_samples", "n_estimators"):
            lower, upper = getattr(self, name)
            if lower < 1 or upper < lower:
                raise ValueError(f"{name} must be an ordered positive integer range")
        for name in (
            "learning_rate",
            "reg_alpha",
            "reg_lambda",
            "min_split_gain",
            "subsample",
            "colsample_bytree",
            "nb2_dispersion",
        ):
            lower, upper = getattr(self, name)
            if not np.isfinite((lower, upper)).all() or lower < 0 or upper < lower:
                raise ValueError(f"{name} must be an ordered finite nonnegative range")
        if self.learning_rate[0] <= 0 or self.nb2_dispersion[0] <= 0:
            raise ValueError("learning_rate and nb2_dispersion must be strictly positive")
        if self.num_leaves[0] > 2 ** self.max_depth[0]:
            raise ValueError("num_leaves lower bound must fit the minimum max_depth")
        if self.subsample[0] <= 0 or self.subsample[1] > 1:
            raise ValueError("subsample must lie in (0, 1]")
        if self.colsample_bytree[0] <= 0 or self.colsample_bytree[1] > 1:
            raise ValueError("colsample_bytree must lie in (0, 1]")


@dataclass(frozen=True)
class DirectCohortConfig:
    """Immutable distributional-tree tuning and bootstrap settings."""

    family: DirectCohortFamily = "poisson"
    tuning: OptunaTuningConfig = OptunaTuningConfig()
    search_space: LightGBMSearchSpaceConfig = LightGBMSearchSpaceConfig()
    tuning_folds: FoldConfig = DEFAULT_FOLD_CONFIG
    bootstrap_replicates: int = 200
    lightgbm_n_jobs: int = 1
    minimum_mean: float = 1e-8
    minimum_scale: float = 1e-8

    def __post_init__(self) -> None:
        if self.family not in _VALID_DIRECT_COHORT_FAMILIES:
            raise ValueError(
                f"family must be one of {sorted(_VALID_DIRECT_COHORT_FAMILIES)}"
            )
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be at least one")
        if self.lightgbm_n_jobs < 1:
            raise ValueError("lightgbm_n_jobs must be at least one")
        for name, value in (
            ("minimum_mean", self.minimum_mean),
            ("minimum_scale", self.minimum_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")


@dataclass(frozen=True)
class IndependentSearchSpaceConfig:
    """Bounded Optuna search space for independent total/probability regularization."""

    total_l2_penalty: tuple[float, float] = (0.0, 1.0)
    probability_c: tuple[float, float] = (0.01, 100.0)

    def __post_init__(self) -> None:
        for name in ("total_l2_penalty", "probability_c"):
            lower, upper = getattr(self, name)
            if not np.isfinite((lower, upper)).all() or lower < 0 or upper < lower:
                raise ValueError(f"{name} must be an ordered finite nonnegative range")
        if self.probability_c[0] <= 0:
            raise ValueError("probability_c must be strictly positive")


@dataclass(frozen=True)
class IndependentTotalProbabilityConfig:
    """Immutable independent total/probability family, tuning, calibration, and bootstrap settings."""

    total_family: IndependentTotalFamily = "nb2"
    tuning: OptunaTuningConfig = OptunaTuningConfig()
    search_space: IndependentSearchSpaceConfig = IndependentSearchSpaceConfig()
    tuning_folds: FoldConfig = DEFAULT_FOLD_CONFIG
    optimizer_max_iterations: int = 500
    optimizer_tolerance: float = 1e-8
    dispersion_bounds: tuple[float, float] = (1e-4, 5.0)
    minimum_mean: float = 1e-8
    calibration_temperature_bounds: tuple[float, float] = (0.25, 4.0)
    # Temperature is retained on a likelihood-ratio test at one degree of
    # freedom, not on a raw improvement threshold. T = 1 lies strictly inside
    # `calibration_temperature_bounds`, so the fitted NLL is <= the raw NLL by
    # construction and any positive tolerance retains noise as evidence.
    calibration_significance_level: float = 0.05
    bootstrap_replicates: int = 200
    # A neighborhood-cluster resample can omit every child of one cohort, which
    # leaves the composition model unfittable for that replicate. Such
    # replicates are skipped and counted rather than failing the prediction.
    bootstrap_max_failed_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.total_family not in _VALID_INDEPENDENT_TOTAL_FAMILIES:
            raise ValueError(
                "total_family must be one of "
                f"{sorted(_VALID_INDEPENDENT_TOTAL_FAMILIES)}"
            )
        if self.optimizer_max_iterations < 1:
            raise ValueError("optimizer_max_iterations must be at least one")
        for name, value in (
            ("optimizer_tolerance", self.optimizer_tolerance),
            ("minimum_mean", self.minimum_mean),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if (
            not np.isfinite(self.calibration_significance_level)
            or not 0 < self.calibration_significance_level < 1
        ):
            raise ValueError(
                "calibration_significance_level must be finite and in (0, 1)"
            )
        for name, bounds in (
            ("dispersion_bounds", self.dispersion_bounds),
            ("calibration_temperature_bounds", self.calibration_temperature_bounds),
        ):
            lower, upper = bounds
            if not np.isfinite(bounds).all() or lower <= 0 or upper <= lower:
                raise ValueError(f"{name} must be ordered, finite, and positive")
        if not 0 <= self.bootstrap_max_failed_fraction < 1:
            raise ValueError("bootstrap_max_failed_fraction must be in [0, 1)")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be at least one")


@dataclass(frozen=True)
class BayesianPriorConfig:
    """Regularizing priors for both stages of the Bayesian conditional model."""

    total_intercept_loc: float = -2.0
    total_intercept_scale: float = 1.0
    total_coefficient_scale: float = 0.5
    dispersion_log_loc: float = 0.0
    dispersion_log_scale: float = 1.0
    neighborhood_scale: float = 0.5
    composition_intercept_scale: float = 1.0
    composition_coefficient_scale: float = 0.5
    kappa_log_loc: float = 2.0
    kappa_log_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("total_intercept_loc", self.total_intercept_loc),
            ("dispersion_log_loc", self.dispersion_log_loc),
            ("kappa_log_loc", self.kappa_log_loc),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in (
            ("total_intercept_scale", self.total_intercept_scale),
            ("total_coefficient_scale", self.total_coefficient_scale),
            ("dispersion_log_scale", self.dispersion_log_scale),
            ("neighborhood_scale", self.neighborhood_scale),
            ("composition_intercept_scale", self.composition_intercept_scale),
            ("composition_coefficient_scale", self.composition_coefficient_scale),
            ("kappa_log_scale", self.kappa_log_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")


@dataclass(frozen=True)
class NUTSProfileConfig:
    """One reproducible, diagnostic-valid Pyro NUTS execution profile."""

    chains: int
    warmup_steps: int
    posterior_samples: int
    target_acceptance: float = 0.9
    max_tree_depth: int = 10
    full_mass: bool = False
    jit_compile: bool = False
    disable_progress_bar: bool = True

    def __post_init__(self) -> None:
        if self.chains < 2:
            raise ValueError("Diagnostic-valid NUTS profiles require at least two chains")
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be at least one")
        if self.posterior_samples < 4:
            raise ValueError(
                "posterior_samples must be at least four for split R-hat diagnostics"
            )
        if not np.isfinite(self.target_acceptance) or not 0 < self.target_acceptance < 1:
            raise ValueError("target_acceptance must be finite and between zero and one")
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be at least one")


@dataclass(frozen=True)
class BayesianDiagnosticConfig:
    """Convergence thresholds and failure action for one inference profile."""

    action: BayesianDiagnosticAction
    maximum_rhat: float = 1.05
    minimum_effective_sample_size: float = 100.0
    maximum_divergences: int = 0
    minimum_mean_accept_prob: float = 0.6
    maximum_mean_accept_prob: float = 0.98
    maximum_tree_depth_saturation: float = 0.05

    def __post_init__(self) -> None:
        if self.action not in _VALID_BAYESIAN_DIAGNOSTIC_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_VALID_BAYESIAN_DIAGNOSTIC_ACTIONS)}"
            )
        if not np.isfinite(self.maximum_rhat) or self.maximum_rhat < 1:
            raise ValueError("maximum_rhat must be finite and at least one")
        if (
            not np.isfinite(self.minimum_effective_sample_size)
            or self.minimum_effective_sample_size <= 0
        ):
            raise ValueError(
                "minimum_effective_sample_size must be finite and strictly positive"
            )
        if self.maximum_divergences < 0:
            raise ValueError("maximum_divergences must be nonnegative")
        for name, value in (
            ("minimum_mean_accept_prob", self.minimum_mean_accept_prob),
            ("maximum_mean_accept_prob", self.maximum_mean_accept_prob),
            ("maximum_tree_depth_saturation", self.maximum_tree_depth_saturation),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.maximum_mean_accept_prob <= self.minimum_mean_accept_prob:
            raise ValueError(
                "maximum_mean_accept_prob must be strictly greater than "
                "minimum_mean_accept_prob"
            )


@dataclass(frozen=True)
class BayesianPriorPredictiveConfig:
    """Prior-predictive sample size, plausibility bounds, and failure action.

    The expected-share ratios bound the prior expected cohort composition as a
    multiple of the uniform share ``1 / K``, so they stay meaningful for any
    cohort count. They deliberately do not constrain realized cohort shares,
    whose dispersion is a modeled consequence of the Dirichlet concentration.
    """

    draws: int = 200
    action: BayesianDiagnosticAction = "warn"
    maximum_children_per_apartment: float = 10.0
    minimum_expected_share_ratio: float = 0.25
    maximum_expected_share_ratio: float = 2.0
    maximum_expected_dominant_share: float = 0.85
    minimum_concentration_quantile: float = 1.0

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise ValueError("Prior-predictive draws must be at least one")
        if self.action not in _VALID_BAYESIAN_DIAGNOSTIC_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_VALID_BAYESIAN_DIAGNOSTIC_ACTIONS)}"
            )
        if (
            not np.isfinite(self.maximum_children_per_apartment)
            or self.maximum_children_per_apartment <= 0
        ):
            raise ValueError(
                "maximum_children_per_apartment must be finite and strictly positive"
            )
        if (
            not np.isfinite(self.minimum_expected_share_ratio)
            or not 0 < self.minimum_expected_share_ratio < 1
        ):
            raise ValueError(
                "minimum_expected_share_ratio must be finite and in (0, 1)"
            )
        if (
            not np.isfinite(self.maximum_expected_share_ratio)
            or self.maximum_expected_share_ratio < 1
        ):
            raise ValueError(
                "maximum_expected_share_ratio must be finite and at least one"
            )
        if (
            not np.isfinite(self.maximum_expected_dominant_share)
            or not 0 < self.maximum_expected_dominant_share < 1
        ):
            raise ValueError(
                "maximum_expected_dominant_share must be finite and in (0, 1)"
            )
        if (
            not np.isfinite(self.minimum_concentration_quantile)
            or self.minimum_concentration_quantile <= 0
        ):
            raise ValueError(
                "minimum_concentration_quantile must be finite and strictly positive"
            )


@dataclass(frozen=True)
class BayesianStabilizationConfig:
    """Explicit, observable numerical clipping policy for Bayesian predictors."""

    clip_total_log_mean: bool = False
    total_log_mean_bounds: tuple[float, float] = (-20.0, 20.0)
    clip_composition_logits: bool = False
    composition_logit_bounds: tuple[float, float] = (-20.0, 20.0)

    def __post_init__(self) -> None:
        for name, bounds in (
            ("total_log_mean_bounds", self.total_log_mean_bounds),
            ("composition_logit_bounds", self.composition_logit_bounds),
        ):
            lower, upper = bounds
            if not np.isfinite(bounds).all() or lower >= upper:
                raise ValueError(f"{name} must be ordered and finite")


@dataclass(frozen=True)
class BayesianConditionalConfig:
    """Runtime policy for the Bayesian NB2 + Dirichlet-multinomial model.

    Feature forms are deliberately not configured here. The total spec is the
    ``feature_spec`` passed to ``fit`` and the probability spec is a model
    constructor argument, so the caller supplies the frozen
    ``IndependentTotalProbabilityModel`` forms.
    """

    active_profile: BayesianInferenceProfile = "reduced"
    priors: BayesianPriorConfig = BayesianPriorConfig()
    reduced_profile: NUTSProfileConfig = NUTSProfileConfig(
        chains=2,
        warmup_steps=150,
        posterior_samples=150,
    )
    full_profile: NUTSProfileConfig = NUTSProfileConfig(
        chains=4,
        warmup_steps=1_000,
        posterior_samples=1_000,
    )
    reduced_diagnostics: BayesianDiagnosticConfig = BayesianDiagnosticConfig(
        action="warn",
        minimum_effective_sample_size=50.0,
    )
    full_diagnostics: BayesianDiagnosticConfig = BayesianDiagnosticConfig(
        action="error"
    )
    prior_predictive: BayesianPriorPredictiveConfig = BayesianPriorPredictiveConfig()
    stabilization: BayesianStabilizationConfig = BayesianStabilizationConfig()

    def __post_init__(self) -> None:
        if self.active_profile not in _VALID_BAYESIAN_INFERENCE_PROFILES:
            raise ValueError(
                "active_profile must be one of "
                f"{sorted(_VALID_BAYESIAN_INFERENCE_PROFILES)}"
            )

    @property
    def inference_profile(self) -> NUTSProfileConfig:
        """Return the selected NUTS profile."""
        return (
            self.reduced_profile
            if self.active_profile == "reduced"
            else self.full_profile
        )

    @property
    def diagnostic_policy(self) -> BayesianDiagnosticConfig:
        """Return diagnostics paired with the selected NUTS profile."""
        return (
            self.reduced_diagnostics
            if self.active_profile == "reduced"
            else self.full_diagnostics
        )


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable declaration of one model component's feature representation.

    ``unknown_category_policy`` is named for what it does. Because categorical
    encoding drops the reference level, a category not seen during fit is
    encoded exactly like the reference level and cannot be distinguished from
    it. ``"treat_as_reference"`` says so; ``"error"`` (the default) refuses
    instead.
    """

    component: FeatureComponent
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    ses_form: SesForm = "linear"
    daycare_form: DaycareForm = "linear"
    interactions: tuple[FeatureInteraction, ...] = ()
    scale_numeric: bool = False
    exposure_column: str | None = None
    unknown_category_policy: UnknownCategoryPolicy = "error"
    spline_n_knots: int = 4
    ses_column: str = "ses"
    household_size_column: str = "avg_household_size"
    median_age_column: str = "median_age"
    daycare_column: str = "n_daycares_500m"
    specification_version: str = "1"

    def __post_init__(self) -> None:
        if self.component not in _VALID_COMPONENTS:
            raise ValueError(f"Unknown feature component: {self.component}")
        if self.ses_form not in _VALID_SES_FORMS:
            raise ValueError(f"Unknown SES form: {self.ses_form}")
        if self.daycare_form not in _VALID_DAYCARE_FORMS:
            raise ValueError(f"Unknown daycare form: {self.daycare_form}")
        if self.unknown_category_policy not in _VALID_UNKNOWN_CATEGORY_POLICIES:
            raise ValueError(f"Unknown category policy: {self.unknown_category_policy}")
        if len(self.numeric_features) != len(set(self.numeric_features)):
            raise ValueError("Numeric feature columns must be unique")
        if len(self.categorical_features) != len(set(self.categorical_features)):
            raise ValueError("Categorical feature columns must be unique")
        if set(self.numeric_features) & set(self.categorical_features):
            raise ValueError("Numeric and categorical feature columns must be disjoint")
        if len(self.interactions) != len(set(self.interactions)):
            raise ValueError("Feature interactions must be unique")
        unknown_interactions = set(self.interactions).difference(_VALID_INTERACTIONS)
        if unknown_interactions:
            raise ValueError(f"Unknown feature interactions: {unknown_interactions}")
        if self.ses_form == "spline" and self.spline_n_knots < 3:
            raise ValueError("SES splines require at least three knots")
        if self.ses_form == "spline" and "ses_x_household_size" in self.interactions:
            # A spline replaces the linear SES column, so a linear SES
            # interaction would appear without its own main effect. Expressing
            # it properly needs a tensor-product basis, which is a modeling
            # extension rather than a default; forbid the pair instead of
            # silently emitting an unsupported term.
            raise ValueError(
                "Spline SES cannot be combined with the linear "
                "ses_x_household_size interaction"
            )
        if self.component == "age_probability" and (
            "room_share_x_household_size" in self.interactions
        ):
            raise ValueError(
                "Room-share by household-size interactions are total-count only"
            )
        if self.component == "total_count" and (
            "room_share_x_median_age" in self.interactions
        ):
            raise ValueError(
                "Room-share by median-age interactions are age-probability only"
            )
        if self.exposure_column is not None:
            if self.component != "total_count":
                raise ValueError("Only total-count features may define an exposure")
            if self.exposure_column in self.numeric_features:
                raise ValueError(
                    "Exposure must be an offset, not an ordinary numeric predictor"
                )

    def validate_for_schema(self, schema: ModelingSchema) -> None:
        """Validate selected columns and interaction hierarchy against a schema."""
        schema.validate_definition()
        selected = {*self.numeric_features, *self.categorical_features}
        selected_room_counts = selected.intersection(schema.room_count_columns)
        selected_room_shares = selected.intersection(schema.room_share_features)
        if selected_room_counts and selected_room_shares:
            raise ValueError("Raw room counts and room shares are redundant features")
        allowed = set(schema.feature_columns)
        unknown = selected.difference(allowed)
        if unknown:
            raise ValueError(
                f"Feature specification columns are not modeled: {unknown}"
            )
        forbidden = selected.intersection(
            {
                *schema.identifier_columns,
                *schema.target_columns,
                *schema.forbidden_feature_columns,
            }
        )
        if forbidden:
            raise ValueError(f"Forbidden feature specification columns: {forbidden}")
        if (
            self.exposure_column is not None
            and self.exposure_column != schema.exposure_column
        ):
            raise ValueError("Feature exposure must match the modeling schema")
        required_numeric = {self.ses_column}
        if "ses_x_household_size" in self.interactions:
            required_numeric.add(self.household_size_column)
        if "daycare_x_median_age" in self.interactions:
            required_numeric.update((self.daycare_column, self.median_age_column))
        if "room_share_x_household_size" in self.interactions:
            required_numeric.update(
                (*schema.room_share_features, self.household_size_column)
            )
        if "room_share_x_median_age" in self.interactions:
            required_numeric.update(
                (*schema.room_share_features, self.median_age_column)
            )
        if self.daycare_form != "linear":
            required_numeric.add(self.daycare_column)
        missing_main_effects = required_numeric.difference(self.numeric_features)
        if missing_main_effects:
            raise ValueError(
                f"Feature interactions require main effects: {missing_main_effects}"
            )


@dataclass(frozen=True)
class CategoricalFeatureSpec:
    """Allowed levels and reference category for one categorical feature."""

    column: str
    categories: tuple[str, ...]
    reference_category: str

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("Categorical feature column must not be empty")
        if not self.categories:
            raise ValueError(f"{self.column} must define at least one category")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError(f"{self.column} categories must be unique")
        if self.reference_category not in self.categories:
            raise ValueError(
                f"{self.column} reference category must be an allowed category"
            )


@dataclass(frozen=True)
class ModelingSchema:
    """Immutable column schema for one modeling experiment."""

    building_id_column: str = "building_id"
    neighborhood_id_column: str = "neighborhood_id"
    raw_numeric_features: tuple[str, ...] = (
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "n_apartments",
    )
    derived_numeric_features: tuple[str, ...] = (
        "3_rooms_share",
        "4_rooms_share",
        "5_rooms_share",
    )
    categorical_feature_specs: tuple[CategoricalFeatureSpec, ...] = (
        CategoricalFeatureSpec(
            column="school_status",
            categories=("none", "existing", "planned"),
            reference_category="none",
        ),
    )
    cohort_target_columns: tuple[str, ...] = (
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
    )
    total_target_column: str = "n_children_total"
    exposure_column: str = "n_apartments"
    room_count_columns: tuple[str, ...] = (
        "3_rooms",
        "4_rooms",
        "5_rooms",
        "6_rooms",
    )
    room_reference_column: str = "6_rooms"
    forbidden_feature_columns: tuple[str, ...] = ("u_b", "w_j", "v_b")
    schema_version: str = "1"

    @property
    def identifier_columns(self) -> tuple[str, ...]:
        return self.building_id_column, self.neighborhood_id_column

    @property
    def room_share_features(self) -> tuple[str, ...]:
        return self.derived_numeric_features

    @property
    def room_share_columns(self) -> tuple[tuple[str, str], ...]:
        non_reference_columns = tuple(
            column
            for column in self.room_count_columns
            if column != self.room_reference_column
        )
        return tuple(zip(non_reference_columns, self.derived_numeric_features))

    @property
    def numeric_features(self) -> tuple[str, ...]:
        return self.raw_numeric_features + self.derived_numeric_features

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return tuple(spec.column for spec in self.categorical_feature_specs)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    @property
    def target_columns(self) -> tuple[str, ...]:
        return self.cohort_target_columns + (self.total_target_column,)

    @property
    def table_columns(self) -> tuple[str, ...]:
        return self.identifier_columns + self.feature_columns + self.target_columns

    def validate_definition(self) -> None:
        """Reject schemas that leak identifiers, targets, or latent effects."""
        if self.building_id_column == self.neighborhood_id_column:
            raise ValueError("Building and neighborhood ID columns must differ")
        if self.exposure_column not in self.raw_numeric_features:
            raise ValueError("Exposure column must be a raw numeric feature")
        if len(self.room_count_columns) != len(set(self.room_count_columns)):
            raise ValueError("Room count columns must be unique")
        if self.room_reference_column not in self.room_count_columns:
            raise ValueError("Room reference column must be a room count column")
        if len(self.derived_numeric_features) != len(self.room_count_columns) - 1:
            raise ValueError(
                "Derived numeric features must name each non-reference room share"
            )
        if len(self.categorical_features) != len(set(self.categorical_features)):
            raise ValueError("Categorical feature columns must be unique")

        features = set(self.feature_columns)
        forbidden = {
            *self.identifier_columns,
            *self.target_columns,
            *self.forbidden_feature_columns,
        }
        overlap = features.intersection(forbidden)
        if overlap:
            raise ValueError(f"Forbidden modeling features: {sorted(overlap)}")
        if len(self.table_columns) != len(set(self.table_columns)):
            raise ValueError("Modeling schema columns must be unique")

    def validate_table(self, df: pd.DataFrame) -> None:
        """Validate the canonical modeling-table invariants."""
        self.validate_definition()
        missing_columns = set(self.table_columns).difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing modeling columns: {sorted(missing_columns)}")

        if not df[self.building_id_column].is_unique:
            raise ValueError(f"{self.building_id_column} must be unique")
        if (df[self.exposure_column] <= 0).any():
            raise ValueError(f"{self.exposure_column} must be positive")

        numeric_columns = [*self.numeric_features, *self.target_columns]
        numeric_values = df.loc[:, numeric_columns].to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise ValueError("Modeling numeric columns must be finite")

        target_values = df.loc[:, self.target_columns].to_numpy(dtype=float)
        if (target_values < 0).any() or not np.equal(
            target_values, np.floor(target_values)
        ).all():
            raise ValueError("Modeling targets must be nonnegative integers")

        for spec in self.categorical_feature_specs:
            if df[spec.column].isna().any():
                raise ValueError(f"{spec.column} must not be missing")
            observed_categories = set(df[spec.column].unique())
            unknown_categories = observed_categories.difference(spec.categories)
            if unknown_categories:
                raise ValueError(
                    f"Unknown {spec.column} values: {sorted(unknown_categories)}"
                )

        cohort_count = len(self.cohort_target_columns)
        cohort_sum = target_values[:, :cohort_count].sum(axis=1)
        total = df[self.total_target_column].to_numpy(dtype=float)
        if not np.array_equal(cohort_sum, total):
            raise ValueError(
                f"{self.total_target_column} must equal the sum of cohort targets"
            )


DEFAULT_MODELING_SCHEMA = ModelingSchema()
DEFAULT_MODELING_SCHEMA.validate_definition()

_BASE_NUMERIC_FEATURES = DEFAULT_MODELING_SCHEMA.numeric_features
_NON_EXPOSURE_NUMERIC_FEATURES = tuple(
    column
    for column in _BASE_NUMERIC_FEATURES
    if column != DEFAULT_MODELING_SCHEMA.exposure_column
)
DEFAULT_TREE_FEATURE_SPEC = FeatureSpec(
    component="tree",
    numeric_features=_BASE_NUMERIC_FEATURES,
    categorical_features=DEFAULT_MODELING_SCHEMA.categorical_features,
)
DEFAULT_TOTAL_FEATURE_SPEC = FeatureSpec(
    component="total_count",
    numeric_features=_NON_EXPOSURE_NUMERIC_FEATURES,
    categorical_features=DEFAULT_MODELING_SCHEMA.categorical_features,
    scale_numeric=True,
    exposure_column=DEFAULT_MODELING_SCHEMA.exposure_column,
)
DEFAULT_PROBABILITY_FEATURE_SPEC = FeatureSpec(
    component="age_probability",
    numeric_features=_NON_EXPOSURE_NUMERIC_FEATURES,
    categorical_features=DEFAULT_MODELING_SCHEMA.categorical_features,
    scale_numeric=True,
)
DEFAULT_BAYESIAN_CONDITIONAL_CONFIG = BayesianConditionalConfig()

IDENTIFIER_COLUMNS = DEFAULT_MODELING_SCHEMA.identifier_columns
NUMERIC_FEATURES = DEFAULT_MODELING_SCHEMA.numeric_features
CATEGORICAL_FEATURES = DEFAULT_MODELING_SCHEMA.categorical_features
MODELING_FEATURE_COLUMNS = DEFAULT_MODELING_SCHEMA.feature_columns
COHORT_TARGET_COLUMNS = DEFAULT_MODELING_SCHEMA.cohort_target_columns
TOTAL_TARGET_COLUMN = DEFAULT_MODELING_SCHEMA.total_target_column
MODELING_TARGET_COLUMNS = DEFAULT_MODELING_SCHEMA.target_columns

DEFAULT_DIRECT_COHORT_CONFIG = DirectCohortConfig()
DEFAULT_INDEPENDENT_TOTAL_PROBABILITY_CONFIG = IndependentTotalProbabilityConfig()
