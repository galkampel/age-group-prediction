"""Bayesian NB2 totals with conditional Dirichlet-multinomial composition."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from dataclasses import asdict
from importlib.metadata import version
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, softmax

from ..distributions import (
    dirichlet_multinomial_prefix_log_masses,
    pointwise_log_probability,
)
from ..fitted_features import FittedFeatureTransformer
from ..modeling_config import (
    DEFAULT_BAYESIAN_CONDITIONAL_CONFIG,
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_SEED,
    BayesianConditionalConfig,
    FeatureSpec,
    ModelingSchema,
    PredictionConfig,
    PredictionValidationConfig,
)
from ..predictive import central_prediction_intervals
from ..results import PredictionResult
from ..state_bundle import restore_config, restore_feature_spec
from .base import BaseAgeGroupModel
from .bayesian_components import (
    PriorPredictiveSummary,
    _composition_model,
    _run_prior_predictive,
    _total_model,
)
from .bayesian_inference import (
    _run_nuts,
    _scoped_torch_runtime_settings,
    evaluate_stage_diagnostics,
)

_IMPLEMENTATION_VERSION = "1"

# The posterior sites prediction reads, and therefore the ones a state bundle
# stores. Sites NUTS may add for its own bookkeeping are not needed to predict.
_POSTERIOR_SITES = {
    "total": (
        "total_intercept",
        "total_coefficients",
        "log_total_concentration",
        "neighborhood_scale",
        "neighborhood_raw",
    ),
    "composition": (
        "composition_intercepts",
        "composition_coefficients",
        "log_composition_concentration",
    ),
}

# Composition scores evaluate a (samples, rows, cohorts) tensor. Chunking rows
# keeps the temporaries small at the full inference profile, where the posterior
# holds several thousand samples.
_COMPOSITION_ROW_CHUNK = 512


class BayesianConditionalModel(BaseAgeGroupModel):
    """Hierarchical NB2 total model plus conditional Dirichlet-multinomial."""

    implementation_version = _IMPLEMENTATION_VERSION
    _bundle_dependencies = ("scipy", "torch", "pyro-ppl")

    def __init__(
        self,
        *,
        schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
        default_rng_seed: int = DEFAULT_SEED,
        default_prediction_config: PredictionConfig | None = None,
        prediction_validation_config: PredictionValidationConfig | None = None,
        probability_feature_spec: FeatureSpec = DEFAULT_PROBABILITY_FEATURE_SPEC,
        bayesian_config: BayesianConditionalConfig = DEFAULT_BAYESIAN_CONDITIONAL_CONFIG,
    ) -> None:
        """Initialize the model and validate its composition feature contract."""
        super().__init__(
            schema=schema,
            default_rng_seed=default_rng_seed,
            default_prediction_config=default_prediction_config,
            prediction_validation_config=prediction_validation_config,
        )
        if probability_feature_spec.component != "composition":
            raise ValueError("probability_feature_spec must target composition")
        probability_feature_spec.validate_for_schema(schema)
        self.probability_feature_spec = probability_feature_spec
        self.bayesian_config = bayesian_config
        self._reset_model_state()

    def _reset_model_state(self) -> None:
        """Clear posterior, preprocessing, diagnostics, and fallback state."""
        self._total_posterior: dict[str, Any] | None = None
        self._composition_posterior: dict[str, Any] | None = None
        self._probability_transformer: FittedFeatureTransformer | None = None
        self._neighborhood_lookup: dict[object, int] = {}
        self._stage_diagnostics: dict[str, object] = {}
        self._prior_predictive_summary: PriorPredictiveSummary | dict[str, object] = {}
        self._prediction_fallback_count = 0

    def _fit_model(
        self,
        *,
        train_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        rng: np.random.Generator,
    ) -> None:
        """Run prior checks and fit the total and composition NUTS stages.

        The total design comes from the base transformer; composition owns a
        separate transformer fitted on the same training partition.
        """
        if log_exposure is None:
            raise ValueError("Bayesian total features must define an exposure offset")
        probability_transformer = FittedFeatureTransformer(
            self.probability_feature_spec, schema=self.schema
        )
        probability_features = probability_transformer.fit_transform(train_df)
        neighborhood_values = train_df[self.schema.neighborhood_id_column].to_numpy()
        neighborhoods = pd.unique(neighborhood_values)
        # Native Python keys, so a state bundle can serialize numeric IDs:
        # `json` rejects NumPy scalars. Lookups by NumPy scalars still match,
        # because equal numbers hash equally.
        self._neighborhood_lookup = {
            (value.item() if isinstance(value, np.generic) else value): index
            for index, value in enumerate(neighborhoods)
        }
        neighborhood_index = np.asarray(
            [self._neighborhood_lookup[value] for value in neighborhood_values]
        )
        total_args = (
            torch.as_tensor(features.to_numpy(copy=True), dtype=torch.float64),
            torch.as_tensor(log_exposure.to_numpy(copy=True), dtype=torch.float64),
            torch.as_tensor(neighborhood_index, dtype=torch.long),
            len(neighborhoods),
            self.bayesian_config.priors,
            torch.as_tensor(
                train_df[self.schema.total_target_column].to_numpy(copy=True),
                dtype=torch.float64,
            ),
        )
        cohort_sums = train_df.loc[:, self.schema.cohort_target_columns].sum(axis=1)
        mismatched = train_df.loc[
            ~np.isclose(cohort_sums, train_df[self.schema.total_target_column]),
            self.schema.building_id_column,
        ]
        if not mismatched.empty:
            raise ValueError(
                "Observed cohort columns must sum to the observed total for every "
                f"building; violated by building(s): {mismatched.tolist()}"
            )
        composition_args = (
            torch.as_tensor(
                probability_features.to_numpy(copy=True), dtype=torch.float64
            ),
            torch.as_tensor(
                train_df[self.schema.total_target_column].to_numpy(copy=True),
                dtype=torch.float64,
            ),
            len(self.schema.cohort_target_columns),
            self.bayesian_config.priors,
            torch.as_tensor(
                train_df.loc[:, self.schema.cohort_target_columns].to_numpy(copy=True),
                dtype=torch.float64,
            ),
        )
        total_seed = self._derive_backend_seed(
            rng, purpose="bayesian-conditional/nuts/total"
        )
        composition_seed = self._derive_backend_seed(
            rng, purpose="bayesian-conditional/nuts/composition"
        )
        prior_seed = self._derive_backend_seed(
            rng, purpose="bayesian-conditional/prior-predictive"
        )
        with _scoped_torch_runtime_settings():
            prior_summary = _run_prior_predictive(
                total_args,
                composition_args[0],
                len(self.schema.cohort_target_columns),
                config=self.bayesian_config,
                seed=prior_seed,
            )
            prior_violations = prior_summary["violations"]
            if prior_violations:
                message = "Bayesian prior-predictive checks failed: " + "; ".join(
                    str(value) for value in prior_violations
                )
                if self.bayesian_config.prior_predictive.action == "error":
                    raise RuntimeError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            total_posterior, total_diagnostics = _run_nuts(
                _total_model, total_args, config=self.bayesian_config, seed=total_seed
            )
            composition_posterior, composition_diagnostics = _run_nuts(
                _composition_model,
                composition_args,
                config=self.bayesian_config,
                seed=composition_seed,
            )
        # Record the evidence before enforcing policy. A failing stage clears
        # fitted state, so applying the policy first would discard the very
        # diagnostics needed to understand the failure.
        total_recorded, total_failures = self._evaluate_stage_policy(total_diagnostics)
        composition_recorded, composition_failures = self._evaluate_stage_policy(
            composition_diagnostics
        )
        self._stage_diagnostics = {
            "total": total_recorded,
            "composition": composition_recorded,
        }
        self._prior_predictive_summary = prior_summary
        self._apply_diagnostic_policy("total", total_recorded, total_failures)
        self._apply_diagnostic_policy("composition", composition_recorded, composition_failures)
        self._total_posterior = total_posterior
        self._composition_posterior = composition_posterior
        self._probability_transformer = probability_transformer

    def _evaluate_stage_policy(
        self, diagnostics: Mapping[str, object]
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        """Attach the convergence-policy outcome to one stage's diagnostics.

        The outcome is recorded on the passing *and* failing paths, and before
        the warn-or-raise branch, so a run that only warned still carries
        machine-readable evidence of which thresholds it missed. Plan section
        9.3 makes convergence an explicit selection constraint, and a consumer
        must not have to re-derive the policy from raw numbers to apply it.
        """
        policy = self.bayesian_config.diagnostic_policy
        failures = evaluate_stage_diagnostics(diagnostics, policy)
        recorded = dict(diagnostics)
        recorded.update(
            {
                "active_profile": self.bayesian_config.active_profile,
                "policy_passed": not failures,
                "policy_failures": list(failures),
                "policy_thresholds": asdict(policy),
            }
        )
        return recorded, failures

    def _apply_diagnostic_policy(
        self,
        stage: str,
        diagnostics: Mapping[str, object],
        failures: tuple[str, ...],
    ) -> None:
        """Warn or raise when one stage violates configured diagnostics."""
        if not failures:
            return
        message = f"Bayesian {stage} diagnostics failed: {'; '.join(failures)}"
        if self.bayesian_config.diagnostic_policy.action == "error":
            # A failed fit clears fitted state, so carry the diagnostics on the
            # exception itself. Otherwise an expensive run that misses one
            # threshold leaves no evidence of why it missed it.
            error = RuntimeError(message)
            error.stage = stage
            error.diagnostics = dict(diagnostics)
            error.failures = failures
            raise error
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _posterior_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return fitted total and composition posterior tensors as NumPy arrays."""
        if self._total_posterior is None or self._composition_posterior is None:
            raise RuntimeError("Bayesian model must be fitted before prediction")
        total = {
            name: value.detach().cpu().numpy()
            for name, value in self._total_posterior.items()
        }
        composition = {
            name: value.detach().cpu().numpy()
            for name, value in self._composition_posterior.items()
        }
        return total, composition

    def _predict_model(
        self,
        *,
        eval_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        prediction_config: PredictionConfig,
        rng: np.random.Generator,
    ) -> PredictionResult:
        """Build posterior means, optional draws, intervals, and total log scores."""
        if log_exposure is None or self._probability_transformer is None:
            raise RuntimeError("Bayesian model must be fitted before prediction")
        total, composition = self._posterior_arrays()
        total_means = self._total_posterior_means(
            eval_df, features.to_numpy(), log_exposure.to_numpy(), total, rng
        )
        probability_features = self._probability_transformer.transform(eval_df)
        probabilities = self._composition_probabilities(
            probability_features.to_numpy(), composition
        )
        total_mean = total_means.mean(axis=0)
        mean_probabilities = probabilities.mean(axis=0)
        cohort_means = total_mean[:, None] * mean_probabilities

        all_draws: dict[str, np.ndarray] | None = None
        reported_draws = None
        intervals = None
        internal_draw_count = max(
            prediction_config.n_predictive_draws,
            500 if prediction_config.interval_levels else 0,
        )
        if internal_draw_count:
            all_draws = self._predictive_draws(
                total_means,
                probabilities,
                internal_draw_count,
                rng,
                total_posterior=total,
                composition_posterior=composition,
            )
            if prediction_config.n_predictive_draws:
                reported_draws = {
                    name: values[:, : prediction_config.n_predictive_draws]
                    for name, values in all_draws.items()
                }
            if prediction_config.interval_levels:
                intervals = {
                    name: central_prediction_intervals(
                        values, prediction_config.interval_levels
                    )
                    for name, values in all_draws.items()
                }

        pointwise = None
        if prediction_config.include_pointwise_log_probabilities:
            pointwise = self._pointwise_log_probabilities(
                eval_df,
                total_means=total_means,
                probabilities=probabilities,
                total_posterior=total,
                composition_posterior=composition,
            )

        return PredictionResult.from_means(
            building_ids=eval_df[self.schema.building_id_column].to_numpy(),
            cohort_names=self.schema.cohort_target_columns,
            total_mean=total_mean,
            cohort_means=cohort_means,
            predictive_draws=reported_draws,
            prediction_intervals=intervals,
            interval_levels=prediction_config.interval_levels,
            pointwise_log_probabilities=pointwise,
            validation_config=self.prediction_validation_config,
            pointwise_log_probability_scope=(
                "sequential_joint" if pointwise is not None else None
            ),
        )

    def _total_posterior_means(
        self,
        eval_df: pd.DataFrame,
        features: np.ndarray,
        log_exposure: np.ndarray,
        posterior: Mapping[str, np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute total means for every posterior sample and evaluation row.

        Known neighborhoods reuse fitted effects; unseen rows draw from the
        fitted population distribution before optional log-mean clipping.
        """
        intercept = posterior["total_intercept"][:, None]
        coefficients = posterior["total_coefficients"]
        raw_effect = posterior["neighborhood_raw"]
        scale = posterior["neighborhood_scale"]
        effects = np.empty((len(intercept), len(eval_df)))
        unseen_count = 0
        for row, value in enumerate(eval_df[self.schema.neighborhood_id_column]):
            index = self._neighborhood_lookup.get(value)
            if index is None:
                effects[:, row] = rng.normal(0.0, scale)
                unseen_count += 1
            else:
                effects[:, row] = scale * raw_effect[:, index]
        self._prediction_fallback_count = unseen_count
        log_mean = log_exposure[None, :] + intercept + coefficients @ features.T + effects
        stabilization = self.bayesian_config.stabilization
        if stabilization.clip_total_log_mean:
            log_mean = np.clip(log_mean, *stabilization.total_log_mean_bounds)
        return np.exp(log_mean)

    def _composition_probabilities(
        self, features: np.ndarray, posterior: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Compute reference-softmax probabilities for each sample and row."""
        logits = (
            posterior["composition_intercepts"][:, None, :]
            + np.einsum(
                "np,spk->snk",
                features,
                posterior["composition_coefficients"],
            )
        )
        logits = np.concatenate((logits, np.zeros((*logits.shape[:-1], 1))), axis=-1)
        stabilization = self.bayesian_config.stabilization
        if stabilization.clip_composition_logits:
            logits = np.clip(logits, *stabilization.composition_logit_bounds)
        return softmax(logits, axis=-1)

    def _pointwise_log_probabilities(
        self,
        eval_df: pd.DataFrame,
        *,
        total_means: np.ndarray,
        probabilities: np.ndarray,
        total_posterior: Mapping[str, np.ndarray],
        composition_posterior: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Return posterior-integrated NB2 total and composition log scores.

        Each cohort entry is a conditional Dirichlet-multinomial log mass given
        the preceding cohorts, so the cohort entries sum to the building's
        posterior-integrated joint composition score. The final cohort is
        determined by the total and therefore scores zero up to rounding.
        """
        cohorts = self.schema.cohort_target_columns
        required = (self.schema.total_target_column, *cohorts)
        missing = [column for column in required if column not in eval_df.columns]
        if missing:
            raise ValueError(
                f"Pointwise log probabilities require observed targets: {sorted(missing)}"
            )
        observed_total = eval_df[self.schema.total_target_column].to_numpy(dtype=float)
        observed_cohorts = eval_df.loc[:, list(cohorts)].to_numpy(dtype=float)
        # Matches the tolerance `_fit_model` applies to the same invariant, so
        # a frame accepted at fit cannot be rejected here.
        if not np.isclose(observed_cohorts.sum(axis=1), observed_total).all():
            raise ValueError(
                "Pointwise log probabilities require observed cohorts to sum to "
                "the observed total"
            )

        # Route through the shared NB2 entry point that Models A and B also use,
        # rather than re-deriving SciPy's size/probability parameters here, so
        # all three families score the total under one implementation. Evaluated
        # for every posterior sample at once: the former per-sample Python loop
        # ran once per draw, i.e. 4,000 times at the full inference profile.
        concentration = np.exp(total_posterior["log_total_concentration"])
        log_masses = pointwise_log_probability(
            "nb2",
            observed_total[None, :],
            total_means,
            dispersion=1.0 / concentration[:, None],
        )
        result = {"total": logsumexp(log_masses, axis=0) - np.log(len(concentration))}

        kappa = np.exp(composition_posterior["log_composition_concentration"])
        sample_count = len(kappa)
        prefix_scores = np.empty((len(eval_df), len(cohorts)), dtype=float)
        # Integrate the prefix masses before differencing them. Integrating each
        # cohort increment separately would not recover the joint score.
        for start in range(0, len(eval_df), _COMPOSITION_ROW_CHUNK):
            rows = slice(start, start + _COMPOSITION_ROW_CHUNK)
            alpha = kappa[:, None, None] * probabilities[:, rows, :]
            prefixes = dirichlet_multinomial_prefix_log_masses(
                observed_cohorts[None, rows, :], alpha
            )
            prefix_scores[rows] = logsumexp(prefixes, axis=0) - np.log(sample_count)
        increments = np.diff(prefix_scores, axis=1, prepend=0.0)
        result.update(
            {cohort: increments[:, index] for index, cohort in enumerate(cohorts)}
        )
        return result

    def _predictive_draws(
        self,
        total_means: np.ndarray,
        probabilities: np.ndarray,
        draw_count: int,
        rng: np.random.Generator,
        *,
        total_posterior: Mapping[str, np.ndarray],
        composition_posterior: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Draw NB2 totals followed by conditional cohort compositions.

        The returned arrays have shape ``(rows, draws)`` and cohort draws must
        reconcile exactly to their corresponding total draws. The posteriors are
        passed in rather than re-read, because ``_posterior_arrays`` copies every
        tensor from Torch to NumPy and the caller has already done that.
        """
        row_count = total_means.shape[1]
        total_draws = np.empty((row_count, draw_count))
        cohort_draws = np.empty((row_count, draw_count, probabilities.shape[-1]))
        total, composition = total_posterior, composition_posterior
        total_indices = rng.integers(0, len(total_means), size=draw_count)
        composition_indices = rng.integers(0, len(probabilities), size=draw_count)
        concentrations = np.exp(composition["log_composition_concentration"])
        total_concentrations = np.exp(total["log_total_concentration"])
        for draw, (total_index, composition_index) in enumerate(
            zip(total_indices, composition_indices, strict=True)
        ):
            means = total_means[total_index]
            phi = total_concentrations[total_index]
            totals = rng.negative_binomial(phi, phi / (phi + means))
            total_draws[:, draw] = totals
            for row in range(row_count):
                shares = rng.dirichlet(
                    concentrations[composition_index]
                    * probabilities[composition_index, row]
                )
                cohort_draws[row, draw] = rng.multinomial(totals[row], shares)
        if not np.array_equal(cohort_draws.sum(axis=2), total_draws):
            raise RuntimeError("Conditional posterior draws failed exact reconciliation")
        result = {"total": total_draws}
        result.update(
            {
                cohort: cohort_draws[:, :, index]
                for index, cohort in enumerate(self.schema.cohort_target_columns)
            }
        )
        return result

    def _model_configuration(self) -> Mapping[str, object]:
        """Return the probability feature spec and the Bayesian configuration.

        The configuration carries the priors, both NUTS profiles, both
        diagnostic policies, and the active profile.
        """
        return {
            "probability_feature_spec": asdict(self.probability_feature_spec),
            "bayesian_config": asdict(self.bayesian_config),
        }

    def _export_model_state(self) -> dict[str, object]:
        """Return posterior samples, composition preprocessing, and diagnostics.

        Posteriors are plain nested lists of the sites prediction reads, so a
        reloaded model predicts without re-running NUTS. Posterior predictive
        draws come from these samples, so the training frame is never needed.
        """
        assert self._probability_transformer is not None
        total, composition = self._posterior_arrays()
        return {
            "probability_feature_spec": asdict(self.probability_feature_spec),
            "bayesian_config": asdict(self.bayesian_config),
            "probability_transformer": self._probability_transformer.to_state(),
            "posteriors": {
                stage: {name: arrays[name].tolist() for name in _POSTERIOR_SITES[stage]}
                for stage, arrays in (("total", total), ("composition", composition))
            },
            # A list of pairs rather than a dict: JSON coerces every object key
            # to a string, which would break lookup for numeric neighborhood IDs.
            "neighborhood_lookup": [
                [value, index] for value, index in self._neighborhood_lookup.items()
            ],
            "diagnostics": self._stage_diagnostics,
            "prior_predictive": self._prior_predictive_summary,
        }

    @classmethod
    def _from_model_state(cls, state: Mapping[str, Any], **base_arguments: Any) -> Self:
        """Restore posterior tensors, neighborhood lookup, and preprocessing."""
        model = cls(
            probability_feature_spec=restore_feature_spec(
                state["probability_feature_spec"]
            ),
            bayesian_config=restore_config(
                DEFAULT_BAYESIAN_CONDITIONAL_CONFIG, state["bayesian_config"]
            ),
            **base_arguments,
        )
        model._total_posterior, model._composition_posterior = (
            {
                name: torch.as_tensor(np.asarray(values, dtype=float), dtype=torch.float64)
                for name, values in state["posteriors"][stage].items()
            }
            for stage in ("total", "composition")
        )
        model._neighborhood_lookup = {
            value: int(index) for value, index in state["neighborhood_lookup"]
        }
        model._stage_diagnostics = copy.deepcopy(dict(state["diagnostics"]))
        model._prior_predictive_summary = copy.deepcopy(dict(state["prior_predictive"]))
        model._probability_transformer = FittedFeatureTransformer.from_state(
            state["probability_transformer"], schema=model.schema
        )
        return model

    def _get_model_metadata(self) -> Mapping[str, object]:
        """Return JSON-safe model configuration, diagnostics, and fit metadata."""
        return {
            "implementation_version": _IMPLEMENTATION_VERSION,
            "likelihood": {
                "total": "hierarchical_nb2",
                "composition": "dirichlet_multinomial_conditioned_on_total",
            },
            "parameterization": "noncentered_neighborhood_and_reference_softmax",
            "pointwise_log_probability_scope": {
                "keys": ["total", *self.schema.cohort_target_columns],
                "prediction_scope": "sequential_joint",
                "total": "posterior_integrated_nb2_log_mass",
                "cohort": "conditional_dirichlet_multinomial_log_mass_given_"
                "preceding_cohorts",
                "joint": "sum_of_all_keys_equals_posterior_integrated_nb2_total_"
                "plus_dirichlet_multinomial_composition",
                "posterior_integration": "log_mean_exp_over_posterior_samples",
                "cohort_order": list(self.schema.cohort_target_columns),
                "includes_multinomial_coefficient": True,
            },
            "hyperparameters": asdict(self.bayesian_config),
            "priors": asdict(self.bayesian_config.priors),
            "dependency_versions": {
                "torch": version("torch"),
                "pyro": version("pyro-ppl"),
            },
            "uncertainty_method": "separate_nuts_posterior_predictive",
            "torch_num_threads": 1,
            "diagnostics": self._stage_diagnostics,
            "prior_predictive": self._prior_predictive_summary,
            "known_neighborhood_count": len(self._neighborhood_lookup),
            "last_prediction_unseen_neighborhood_count": self._prediction_fallback_count,
            "probability_preprocessing": (
                self._probability_transformer.get_metadata()
                if self._probability_transformer is not None
                else None
            ),
        }


__all__ = ["BayesianConditionalModel"]