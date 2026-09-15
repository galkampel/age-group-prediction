"""Independent NB2 totals and grouped age probabilities."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
import scipy
from scipy.special import softmax
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression

from ..data_splitting import ValidationFold, make_validation_folds
from ..distributions import (
    draw_outcomes,
    multinomial_prefix_log_masses,
    pointwise_log_probability,
)
from ..feature_engineering import FittedFeatureTransformer
from ..modeling_config import (
    DEFAULT_INDEPENDENT_TOTAL_PROBABILITY_CONFIG,
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_SEED,
    FeatureSpec,
    IndependentTotalProbabilityConfig,
    ModelingSchema,
    PredictionConfig,
    PredictionValidationConfig,
)
from ..predictive import central_prediction_intervals
from ..resampling import NeighborhoodClusterResampler
from ..results import ParametricDistributionSpec, PredictionResult
from ..state_bundle import restore_config, restore_feature_spec
from ..tuning import TuningResult, run_optuna_study
from .base import BaseAgeGroupModel
from .count_regression import (
    _fit_total_regression,
    _NB2Fit,
    _predict_total_mean,
    _total_fit_from_state,
    _total_fit_to_state,
    _TotalFit,
)
from .fold_scoring import _probability_fold_loss, _total_fold_loss
from .grouped_multinomial import (
    _decision_logits,
    _fit_grouped_multinomial,
    _multinomial_from_state,
    _multinomial_to_state,
)
from .probability_calibration import _fit_calibration_temperature

_IMPLEMENTATION_VERSION = "2"


class IndependentTotalProbabilityModel(BaseAgeGroupModel):
    """Explicit count-family total and grouped multinomial probability components."""

    implementation_version = _IMPLEMENTATION_VERSION
    _bundle_dependencies = ("scipy",)

    def __init__(
        self,
        *,
        schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
        default_rng_seed: int = DEFAULT_SEED,
        default_prediction_config: PredictionConfig | None = None,
        prediction_validation_config: PredictionValidationConfig | None = None,
        probability_feature_spec: FeatureSpec = DEFAULT_PROBABILITY_FEATURE_SPEC,
        independent_config: IndependentTotalProbabilityConfig = (
            DEFAULT_INDEPENDENT_TOTAL_PROBABILITY_CONFIG
        ),
    ) -> None:
        """Initialize the independent total/probability model's feature and estimation configuration."""
        super().__init__(
            schema=schema,
            default_rng_seed=default_rng_seed,
            default_prediction_config=default_prediction_config,
            prediction_validation_config=prediction_validation_config,
        )
        if probability_feature_spec.component != "age_probability":
            raise ValueError("probability_feature_spec must target age_probability")
        probability_feature_spec.validate_for_schema(schema)
        self.probability_feature_spec = probability_feature_spec
        self.independent_config = independent_config
        self._reset_model_state()

    def _reset_model_state(self) -> None:
        """Clear learned independent-model state before fitting or after a failed fit."""
        self._total_fit: _TotalFit | None = None
        self._probability_estimator: LogisticRegression | None = None
        self._probability_transformer: FittedFeatureTransformer | None = None
        self._selected_probability_spec: FeatureSpec | None = None
        self._selected_total_l2: float | None = None
        self._selected_probability_c: float | None = None
        self._total_tuning_result: TuningResult | None = None
        self._probability_tuning_result: TuningResult | None = None
        self._temperature = 1.0
        self._selection_folds: list[ValidationFold] = []
        self._unvalidated_building_ids: tuple[object, ...] = ()
        self._selection_diagnostics: dict[str, object] = {}
        self._calibration_diagnostics: dict[str, object] = {}
        self._bootstrap_failed_replicates = 0
        self._bootstrap_successful_replicates = 0
        self._train_df: pd.DataFrame | None = None

    def _select_feature_spec(
        self,
        *,
        train_df: pd.DataFrame,
        feature_spec: FeatureSpec,
        rng: np.random.Generator,
    ) -> FeatureSpec:
        """Tune fixed total and composition specifications on identical training folds."""
        if feature_spec.component != "total_count" or feature_spec.exposure_column is None:
            raise ValueError(
                "IndependentTotalProbabilityModel requires total-count features with exposure"
            )
        fold_seed = self._derive_backend_seed(rng, purpose="model-b/validation-folds")
        fold_plan = make_validation_folds(
            train_df,
            config=self.independent_config.tuning_folds,
            rng=np.random.default_rng(fold_seed),
        )
        self._selection_folds = list(fold_plan.folds)
        self._unvalidated_building_ids = fold_plan.unvalidated_building_ids
        self._selected_probability_spec = self.probability_feature_spec
        self._total_tuning_result = self._tune_component(
            spec=feature_spec, component="total_count", rng=rng
        )
        self._probability_tuning_result = self._tune_component(
            spec=self.probability_feature_spec, component="age_probability", rng=rng
        )
        self._selected_total_l2 = float(
            self._total_tuning_result.best_params["total_l2_penalty"]
        )
        self._selected_probability_c = float(
            self._probability_tuning_result.best_params["probability_c"]
        )
        self._selection_diagnostics = {
            "fold_count": len(self._selection_folds),
            "unvalidated_building_count": len(self._unvalidated_building_ids),
            "total_feature_spec": asdict(feature_spec),
            "probability_feature_spec": asdict(self.probability_feature_spec),
            "total_tuning": asdict(self._total_tuning_result),
            "probability_tuning": asdict(self._probability_tuning_result),
            # Without its bounds a chosen penalty is not reproducible from the
            # run's own record. Same key as DirectCohortModel so a tracked
            # comparison reads one column across model families.
            "search_space": asdict(self.independent_config.search_space),
        }
        return feature_spec

    def _tune_component(
        self,
        *,
        spec: FeatureSpec,
        component: str,
        rng: np.random.Generator,
    ) -> TuningResult:
        """Tune one component's regularization with seeded Optuna CV."""
        spec.validate_for_schema(self.schema)
        search = self.independent_config.search_space
        parameter_name = (
            "total_l2_penalty" if component == "total_count" else "probability_c"
        )
        bounds = getattr(search, parameter_name)
        sampler_seed = self._derive_backend_seed(
            rng, purpose=f"model-b/optuna-sampler/{component}"
        )

        def objective(trial: optuna.Trial) -> float:
            """Return one trial's mean held-out loss across the selection folds."""
            regularization = trial.suggest_float(
                parameter_name,
                *bounds,
                log=component == "age_probability",
            )
            scores: list[float] = []
            for fold in self._selection_folds:
                scores.append(
                    self._score_fold(
                        spec,
                        component,
                        regularization,
                        fold,
                        rng,
                        f"trial-{trial.number}",
                    )
                )
                trial.report(float(np.mean(scores)), step=fold.fold_index)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        return run_optuna_study(
            objective,
            config=self.independent_config.tuning,
            seed=sampler_seed,
        )

    def _score_fold(
        self,
        spec: FeatureSpec,
        component: str,
        regularization: float,
        fold: ValidationFold,
        rng: np.random.Generator,
        purpose: str,
    ) -> float:
        """Fit one component on a fold and return its held-out loss.

        Only the probability component draws a seed; the fitting and scoring
        live in `fold_scoring`.
        """
        if component == "total_count":
            return _total_fold_loss(
                spec,
                regularization,
                fold,
                schema=self.schema,
                config=self.independent_config,
            )
        # Feature preprocessing consumes no randomness, so drawing the seed
        # before the fold's features are built leaves every seed of a
        # successful fit unchanged.
        seed = self._derive_backend_seed(
            rng, purpose=f"model-b/selection/{purpose}/fold-{fold.fold_index}"
        )
        return _probability_fold_loss(
            spec,
            regularization,
            fold,
            schema=self.schema,
            config=self.independent_config,
            seed=seed,
        )

    def _fit_temperature(self, rng: np.random.Generator) -> None:
        """Calibrate composition confidence with one scalar temperature, $T$.

        Cross-fitted logits $z$ are converted to probabilities as $softmax(z / T)$.
        Values above one flatten overconfident probabilities, while values below one
        sharpen underconfident probabilities. The fitted $T$ minimizes child-count-
        weighted out-of-fold multinomial NLL and is retained only when a
        likelihood-ratio test at one degree of freedom rejects $T = 1$ at
        `calibration_significance_level`.
        """
        spec = self._selected_probability_spec
        c_value = self._selected_probability_c
        assert spec is not None and c_value is not None
        logits_parts: list[np.ndarray] = []
        count_parts: list[np.ndarray] = []
        for fold in self._selection_folds:
            transformer = FittedFeatureTransformer(spec, schema=self.schema)
            fit_features = transformer.fit_transform(fold.fit_df)
            validation_features = transformer.transform(fold.validation_df)
            seed = self._derive_backend_seed(
                rng, purpose=f"model-b/calibration/fold-{fold.fold_index}"
            )
            estimator = _fit_grouped_multinomial(
                fit_features,
                fold.fit_df.loc[:, self.schema.cohort_target_columns].to_numpy(
                    dtype=float
                ),
                c_value=c_value,
                config=self.independent_config,
                seed=seed,
            )
            logits_parts.append(_decision_logits(estimator, validation_features))
            count_parts.append(
                fold.validation_df.loc[:, self.schema.cohort_target_columns].to_numpy(
                    dtype=float
                )
            )
        # The fit, the likelihood-ratio retention test and the diagnostics are
        # pure functions of these rows, in `probability_calibration`.
        self._temperature, self._calibration_diagnostics = _fit_calibration_temperature(
            np.vstack(logits_parts),
            np.vstack(count_parts),
            config=self.independent_config,
        )

    def _fit_model(
        self,
        *,
        train_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        rng: np.random.Generator,
    ) -> None:
        """Refit selected total and composition components on all training data."""
        probability_spec = self._selected_probability_spec
        total_l2 = self._selected_total_l2
        probability_c = self._selected_probability_c
        if log_exposure is None or probability_spec is None:
            raise RuntimeError(
                "Independent total/probability selection did not produce both feature specs"
            )
        assert total_l2 is not None and probability_c is not None
        # Calibration needs the selected regularization and the selection folds,
        # and nothing downstream of it, so it runs here rather than inside
        # `_select_feature_spec`, which selects a specification.
        self._fit_temperature(rng)
        self._total_fit = _fit_total_regression(
            self.independent_config.total_family,
            features,
            train_df[self.schema.total_target_column].to_numpy(dtype=float),
            log_exposure,
            l2_penalty=total_l2,
            config=self.independent_config,
        )
        probability_transformer = FittedFeatureTransformer(
            probability_spec, schema=self.schema
        )
        probability_features = probability_transformer.fit_transform(train_df)
        final_seed = self._derive_backend_seed(
            rng, purpose="model-b/final-probability-refit"
        )
        self._probability_estimator = _fit_grouped_multinomial(
            probability_features,
            train_df.loc[:, self.schema.cohort_target_columns].to_numpy(dtype=float),
            c_value=probability_c,
            config=self.independent_config,
            seed=final_seed,
        )
        self._probability_transformer = probability_transformer
        self._train_df = train_df.copy(deep=True)

    def _predict_model(
        self,
        *,
        eval_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        prediction_config: PredictionConfig,
        rng: np.random.Generator,
    ) -> PredictionResult:
        """Produce reconciled mean predictions and optional predictive uncertainty."""
        total_fit = self._total_fit
        probability_estimator = self._probability_estimator
        probability_transformer = self._probability_transformer
        if (
            total_fit is None
            or probability_estimator is None
            or probability_transformer is None
            or log_exposure is None
        ):
            raise RuntimeError("Independent total/probability model must be fitted before prediction")
        total_mean = _predict_total_mean(
            total_fit,
            features,
            log_exposure,
            minimum_mean=self.independent_config.minimum_mean,
        )
        probability_features = probability_transformer.transform(eval_df)
        logits = _decision_logits(probability_estimator, probability_features)
        probabilities = softmax(logits / self._temperature, axis=1)
        cohort_means = total_mean[:, None] * probabilities

        pointwise = None
        if prediction_config.include_pointwise_log_probabilities:
            pointwise = self._pointwise_log_probabilities(
                eval_df, total_mean=total_mean, probabilities=probabilities
            )

        predictive_draws = None
        prediction_intervals = None
        if prediction_config.n_predictive_draws or prediction_config.interval_levels:
            n_refits = max(
                self.independent_config.bootstrap_replicates,
                prediction_config.n_predictive_draws,
            )
            all_draws = self._bootstrap_predictive_draws(
                eval_df=eval_df, n_refits=n_refits, rng=rng
            )
            n_reported = prediction_config.n_predictive_draws or n_refits
            predictive_draws = {
                target: values[:, :n_reported] for target, values in all_draws.items()
            }
            if prediction_config.interval_levels:
                prediction_intervals = {
                    target: central_prediction_intervals(
                        values, prediction_config.interval_levels
                    )
                    for target, values in all_draws.items()
                }

        return PredictionResult.from_means(
            building_ids=eval_df[self.schema.building_id_column].to_numpy(),
            cohort_names=self.schema.cohort_target_columns,
            total_mean=total_mean,
            cohort_means=cohort_means,
            predictive_draws=predictive_draws,
            prediction_intervals=prediction_intervals,
            interval_levels=prediction_config.interval_levels,
            pointwise_log_probabilities=pointwise,
            parametric_distributions={
                "total": ParametricDistributionSpec(
                    self.independent_config.total_family,
                    dispersion=(
                        total_fit.dispersion
                        if isinstance(total_fit, _NB2Fit)
                        else None
                    ),
                )
            },
            validation_config=self.prediction_validation_config,
            # The NB2/Poisson total plus sequential conditional multinomial
            # cohort scores sum to the joint log mass of total and cohorts.
            pointwise_log_probability_scope=(
                "sequential_joint" if pointwise is not None else None
            ),
        )

    def _pointwise_log_probabilities(
        self,
        eval_df: pd.DataFrame,
        *,
        total_mean: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute per-target log contributions for observed evaluation rows."""
        total_fit = self._total_fit
        assert total_fit is not None
        required = (self.schema.total_target_column, *self.schema.cohort_target_columns)
        missing = set(required).difference(eval_df.columns)
        if missing:
            raise ValueError(
                f"Pointwise log probabilities require observed targets: {sorted(missing)}"
            )
        result = {
            "total": pointwise_log_probability(
                self.independent_config.total_family,
                eval_df[self.schema.total_target_column].to_numpy(dtype=float),
                total_mean,
                dispersion=(
                    total_fit.dispersion if isinstance(total_fit, _NB2Fit) else None
                ),
            )
        }
        # Each cohort entry is a conditional multinomial log mass given the
        # preceding cohorts, so the cohort entries sum to the building's
        # multinomial composition log mass, including its coefficient. Omitting
        # that coefficient would leave a log kernel rather than a log mass, and
        # would offset every comparison against a model that reports one.
        observed_cohorts = eval_df.loc[
            :, list(self.schema.cohort_target_columns)
        ].to_numpy(dtype=float)
        prefixes = multinomial_prefix_log_masses(observed_cohorts, probabilities)
        increments = np.diff(prefixes, axis=1, prepend=0.0)
        for index, cohort in enumerate(self.schema.cohort_target_columns):
            result[cohort] = increments[:, index]
        return result

    def _bootstrap_predictive_draws(
        self,
        *,
        eval_df: pd.DataFrame,
        n_refits: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Generate cluster-refit NB2-total and conditional-multinomial cohort draws."""
        train_df = self._require_training_frame(self._train_df)
        total_spec = self._feature_spec
        probability_spec = self._selected_probability_spec
        total_l2 = self._selected_total_l2
        probability_c = self._selected_probability_c
        if total_spec is None or probability_spec is None:
            raise RuntimeError(
                "Independent total/probability model must be fitted before bootstrap prediction"
            )
        assert total_l2 is not None and probability_c is not None
        resampler = NeighborhoodClusterResampler.from_frame(
            train_df, self.schema.neighborhood_id_column
        )
        cohorts = self.schema.cohort_target_columns
        total_columns: list[np.ndarray] = []
        cohort_columns: dict[str, list[np.ndarray]] = {cohort: [] for cohort in cohorts}
        failed_replicates = 0
        for refit_index in range(n_refits):
            row_indices = resampler.sample_row_indices(rng)
            resampled = train_df.iloc[row_indices].reset_index(drop=True)
            total_transformer = FittedFeatureTransformer(total_spec, schema=self.schema)
            total_features = total_transformer.fit_transform(resampled)
            eval_total_features = total_transformer.transform(eval_df)
            fit_offset = total_transformer.get_log_exposure(resampled)
            eval_offset = total_transformer.get_log_exposure(eval_df)
            assert fit_offset is not None and eval_offset is not None
            total_fit = _fit_total_regression(
                self.independent_config.total_family,
                total_features,
                resampled[self.schema.total_target_column].to_numpy(dtype=float),
                fit_offset,
                l2_penalty=total_l2,
                config=self.independent_config,
            )
            total_means = _predict_total_mean(
                total_fit,
                eval_total_features,
                eval_offset,
                minimum_mean=self.independent_config.minimum_mean,
            )
            probability_transformer = FittedFeatureTransformer(
                probability_spec, schema=self.schema
            )
            probability_features = probability_transformer.fit_transform(resampled)
            eval_probability_features = probability_transformer.transform(eval_df)
            fit_seed = self._derive_backend_seed(
                rng, purpose=f"model-b/bootstrap-probability/{refit_index}"
            )
            try:
                probability_estimator = _fit_grouped_multinomial(
                    probability_features,
                    resampled.loc[:, cohorts].to_numpy(dtype=float),
                    c_value=probability_c,
                    config=self.independent_config,
                    seed=fit_seed,
                )
            except ValueError:
                # This resample contains no child of some cohort, so the
                # composition model is unidentified for it. Skipping conditions
                # the interval on resamples that span every cohort; the
                # alternative is failing a prediction the caller cannot fix.
                failed_replicates += 1
                continue
            probabilities = softmax(
                _decision_logits(probability_estimator, eval_probability_features)
                / self._temperature,
                axis=1,
            )
            outcome_seed = self._derive_backend_seed(
                rng, purpose=f"model-b/bootstrap-outcome/{refit_index}"
            )
            outcome_rng = np.random.default_rng(outcome_seed)
            sampled_totals = draw_outcomes(
                self.independent_config.total_family,
                total_means,
                rng=outcome_rng,
                dispersion=(
                    total_fit.dispersion if isinstance(total_fit, _NB2Fit) else None
                ),
            ).astype(int)
            total_columns.append(sampled_totals.astype(float))
            sampled_cohorts = np.vstack(
                [
                    outcome_rng.multinomial(int(total), probability)
                    for total, probability in zip(sampled_totals, probabilities)
                ]
            )
            for cohort_index, cohort in enumerate(cohorts):
                cohort_columns[cohort].append(
                    sampled_cohorts[:, cohort_index].astype(float)
                )
        if failed_replicates / n_refits > self.independent_config.bootstrap_max_failed_fraction:
            raise RuntimeError(
                f"{failed_replicates} of {n_refits} bootstrap replicates could not "
                "fit the composition model because a resample omitted an age "
                "cohort entirely; the cohort is too concentrated in a few "
                "neighborhoods for cluster resampling"
            )
        self._bootstrap_failed_replicates = failed_replicates
        self._bootstrap_successful_replicates = n_refits - failed_replicates
        return {
            "total": np.column_stack(total_columns),
            **{
                cohort: np.column_stack(columns)
                for cohort, columns in cohort_columns.items()
            },
        }

    def _model_configuration(self) -> Mapping[str, object]:
        """Return the probability feature spec and the independent-model configuration."""
        return {
            "probability_feature_spec": asdict(self.probability_feature_spec),
            "independent_config": asdict(self.independent_config),
        }

    def _export_model_state(self) -> dict[str, object]:
        """Return both fitted components, calibration, and selection evidence.

        The count regression is stored as coefficients (plus NB2 dispersion) and
        the multinomial as its coefficients, intercepts, and classes, so neither
        optimizer re-runs on load. The internal selection folds and the retained
        training frame are deliberately omitted: both hold training rows.
        """
        assert self._total_fit is not None
        assert self._probability_estimator is not None
        assert self._probability_transformer is not None
        assert self._selected_probability_spec is not None
        assert self._total_tuning_result is not None
        assert self._probability_tuning_result is not None
        return {
            "probability_feature_spec": asdict(self.probability_feature_spec),
            "independent_config": asdict(self.independent_config),
            "total_fit": _total_fit_to_state(self._total_fit),
            "probability_estimator": _multinomial_to_state(self._probability_estimator),
            "probability_transformer": self._probability_transformer.to_state(),
            "selected_probability_spec": asdict(self._selected_probability_spec),
            "selected_total_l2": self._selected_total_l2,
            "selected_probability_c": self._selected_probability_c,
            "temperature": self._temperature,
            "total_tuning_result": asdict(self._total_tuning_result),
            "probability_tuning_result": asdict(self._probability_tuning_result),
            "selection_diagnostics": self._selection_diagnostics,
            "calibration_diagnostics": self._calibration_diagnostics,
        }

    @classmethod
    def _from_model_state(cls, state: Mapping[str, Any], **base_arguments: Any) -> Self:
        """Rebuild both components from exported state; refuse a mismatched family."""
        model = cls(
            probability_feature_spec=restore_feature_spec(
                state["probability_feature_spec"]
            ),
            independent_config=restore_config(
                DEFAULT_INDEPENDENT_TOTAL_PROBABILITY_CONFIG, state["independent_config"]
            ),
            **base_arguments,
        )
        total_fit = _total_fit_from_state(state["total_fit"])
        if state["total_fit"]["family"] != model.independent_config.total_family:
            raise ValueError(
                "State bundle total regression family does not match its configuration"
            )
        model._total_fit = total_fit
        model._probability_estimator = _multinomial_from_state(
            state["probability_estimator"]
        )
        model._probability_transformer = FittedFeatureTransformer.from_state(
            state["probability_transformer"], schema=model.schema
        )
        model._selected_probability_spec = restore_feature_spec(
            state["selected_probability_spec"]
        )
        model._selected_total_l2 = float(state["selected_total_l2"])
        model._selected_probability_c = float(state["selected_probability_c"])
        model._temperature = float(state["temperature"])
        model._total_tuning_result = TuningResult.from_dict(state["total_tuning_result"])
        model._probability_tuning_result = TuningResult.from_dict(
            state["probability_tuning_result"]
        )
        model._selection_diagnostics = copy.deepcopy(dict(state["selection_diagnostics"]))
        model._calibration_diagnostics = copy.deepcopy(
            dict(state["calibration_diagnostics"])
        )
        return model

    def _attach_training_frame(self, train_df: pd.DataFrame) -> None:
        """Retain a deep copy of the training frame for bootstrap refits."""
        self._train_df = train_df.copy(deep=True)

    def _get_model_metadata(self) -> Mapping[str, object]:
        """Return fitted independent-model settings, calibration, and diagnostic metadata."""
        total_fit = self._total_fit
        family = self.independent_config.total_family
        return {
            "implementation_version": _IMPLEMENTATION_VERSION,
            "likelihood": (
                f"{family} total with conditional grouped multinomial composition"
            ),
            "parameterization": (
                "Var(Y)=mu+alpha*mu^2; log apartment exposure offset"
                if family == "nb2"
                else "Var(Y)=mu; log apartment exposure offset"
            ),
            "hyperparameters": {
                "total_family": family,
                "total_l2_penalty": self._selected_total_l2,
                "probability_c": self._selected_probability_c,
                "dispersion_alpha": (
                    total_fit.dispersion if isinstance(total_fit, _NB2Fit) else None
                ),
            },
            "priors": None,
            "calibration": self._calibration_diagnostics,
            "dependency_versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn_version,
            },
            "uncertainty_method": (
                f"neighborhood-cluster bootstrap refits with {family} totals and "
                "conditional multinomial cohorts"
            ),
            "diagnostics": {
                "selection": self._selection_diagnostics,
                "probability_feature_spec": (
                    asdict(self._selected_probability_spec)
                    if self._selected_probability_spec is not None
                    else None
                ),
                "probability_preprocessing": (
                    self._probability_transformer.get_metadata()
                    if self._probability_transformer is not None
                    else None
                ),
                "zero_total_policy": (
                    "included in total and preprocessing; zero composition weight"
                ),
                "bootstrap_successful_replicates": (
                    self._bootstrap_successful_replicates
                ),
                "bootstrap_failed_replicates": self._bootstrap_failed_replicates,
                "cohort_distribution_declaration": (
                    "joint predictive draws only; no marginal NB2 declaration"
                ),
            },
        }


__all__ = ["IndependentTotalProbabilityModel"]