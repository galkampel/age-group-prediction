"""Independently tuned distributional trees for three age cohorts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Self

import lightgbm
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor

from ..data_splitting import ValidationFold, make_validation_folds
from ..distributions import (
    draw_outcomes,
    nb2_gradient_hessian,
    nb2_total_log_probability,
    pointwise_log_probability,
)
from ..feature_engineering import FittedFeatureTransformer
from ..modeling_config import (
    DEFAULT_DIRECT_COHORT_CONFIG,
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_SEED,
    DirectCohortConfig,
    ModelingSchema,
    PredictionConfig,
    PredictionValidationConfig,
)
from ..predictive import central_prediction_intervals
from ..resampling import NeighborhoodClusterResampler
from ..results import ParametricDistributionSpec, PredictionResult
from ..state_bundle import restore_config
from ..tuning import TuningResult, run_optuna_study
from .base import BaseAgeGroupModel

_IMPLEMENTATION_VERSION = "2"
_RAW_SCORE_BOUNDS = (-30.0, 30.0)
# The LightGBM objective each family trains with. NB2 has no built-in
# objective, so it optimizes a custom gradient and Hessian.
_LIGHTGBM_OBJECTIVES = {
    "poisson": "poisson",
    "normal": "regression",
    "nb2": "custom_nb2_gradient",
}
_FoldTransform = tuple[ValidationFold, pd.DataFrame, pd.DataFrame]


class DirectCohortModel(BaseAgeGroupModel):
    """One independently tuned LightGBM regressor per cohort target."""

    implementation_version = _IMPLEMENTATION_VERSION
    _bundle_dependencies = ("lightgbm",)

    def __init__(
        self,
        *,
        schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
        default_rng_seed: int = DEFAULT_SEED,
        default_prediction_config: PredictionConfig | None = None,
        prediction_validation_config: PredictionValidationConfig | None = None,
        direct_cohort_config: DirectCohortConfig = DEFAULT_DIRECT_COHORT_CONFIG,
    ) -> None:
        """Store the direct-cohort configuration and start unfitted."""
        super().__init__(
            schema=schema,
            default_rng_seed=default_rng_seed,
            default_prediction_config=default_prediction_config,
            prediction_validation_config=prediction_validation_config,
        )
        self.direct_cohort_config = direct_cohort_config
        self._reset_model_state()

    def _reset_model_state(self) -> None:
        """Clear per-cohort boosters, parameters, tuning results, and training frame."""
        # The fitted Booster rather than its scikit-learn wrapper: prediction
        # needs only the trees, and a Booster is exactly what LightGBM rebuilds
        # from a state bundle, so fitted and reloaded models share one path.
        self._fitted_boosters: dict[str, lightgbm.Booster] = {}
        self._selected_parameters: dict[str, dict[str, int | float | str]] = {}
        self._ancillary_parameters: dict[str, float] = {}
        self._tuning_results: dict[str, TuningResult] = {}
        self._unvalidated_building_ids: tuple[object, ...] = ()
        self._train_df: pd.DataFrame | None = None

    def _fit_model(
        self,
        *,
        train_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        rng: np.random.Generator,
    ) -> None:
        """Tune and refit one LightGBM regressor per cohort on shared folds."""
        feature_spec = self._feature_spec
        assert feature_spec is not None
        if feature_spec.component != "tree" or log_exposure is not None:
            raise ValueError(
                "DirectCohortModel requires tree features without an exposure offset"
            )
        cohort_targets = self.schema.cohort_target_columns
        missing_targets = set(cohort_targets).difference(train_df.columns)
        if missing_targets:
            raise ValueError(f"Missing cohort target columns: {sorted(missing_targets)}")

        fold_plan = make_validation_folds(
            train_df, config=self.direct_cohort_config.tuning_folds, rng=rng
        )
        self._unvalidated_building_ids = fold_plan.unvalidated_building_ids
        fold_transforms: list[_FoldTransform] = []
        for fold in fold_plan.folds:
            transformer = FittedFeatureTransformer(feature_spec, schema=self.schema)
            fold_transforms.append(
                (
                    fold,
                    transformer.fit_transform(fold.fit_df),
                    transformer.transform(fold.validation_df),
                )
            )

        for cohort in cohort_targets:
            sampler_seed = self._derive_backend_seed(
                rng, purpose=f"optuna-sampler/{cohort}"
            )

            def objective(trial: optuna.Trial, *, target: str = cohort) -> float:
                """Return the mean fold validation loss of one trial for ``target``."""
                parameters = self._suggest_parameters(trial)
                scores: list[float] = []
                for fold, fit_features, validation_features in fold_transforms:
                    fit_seed = self._derive_backend_seed(
                        rng,
                        purpose=f"tuning/{target}/trial-{trial.number}/fold-{fold.fold_index}",
                    )
                    estimator = self._make_estimator(parameters, seed=fit_seed)
                    estimator.fit(fit_features, fold.fit_df[target].to_numpy())
                    predictions = self._predict_mean(
                        estimator.booster_, validation_features
                    )
                    observed = fold.validation_df[target].to_numpy(dtype=float)
                    scores.append(self._validation_score(observed, predictions, parameters))
                    trial.report(float(np.mean(scores)), step=fold.fold_index)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return float(np.mean(scores))

            tuning_result = run_optuna_study(
                objective,
                config=self.direct_cohort_config.tuning,
                seed=sampler_seed,
            )
            parameters = dict(tuning_result.best_params)
            final_seed = self._derive_backend_seed(rng, purpose=f"final-refit/{cohort}")
            estimator = self._make_estimator(parameters, seed=final_seed)
            estimator.fit(features, train_df[cohort].to_numpy())
            self._fitted_boosters[cohort] = estimator.booster_
            self._selected_parameters[cohort] = parameters
            self._tuning_results[cohort] = tuning_result
            if self.direct_cohort_config.family == "normal":
                self._ancillary_parameters[cohort] = self._cross_fitted_normal_scale(
                    cohort, parameters, fold_transforms, rng
                )
            elif self.direct_cohort_config.family == "nb2":
                self._ancillary_parameters[cohort] = float(parameters["dispersion"])

        self._train_df = train_df.copy(deep=True)

    def _suggest_parameters(
        self, trial: optuna.Trial
    ) -> dict[str, int | float | str]:
        """Sample LightGBM hyperparameters (and NB2 dispersion) for one trial."""
        search = self.direct_cohort_config.search_space
        max_depth = trial.suggest_int("max_depth", *search.max_depth)
        parameters: dict[str, int | float | str] = {
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int(
                "num_leaves",
                search.num_leaves[0],
                min(search.num_leaves[1], 2**max_depth),
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *search.min_child_samples
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *search.learning_rate, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", *search.n_estimators),
            "reg_alpha": trial.suggest_float("reg_alpha", *search.reg_alpha, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *search.reg_lambda, log=True
            ),
            "min_split_gain": trial.suggest_float(
                "min_split_gain", *search.min_split_gain
            ),
            "subsample": trial.suggest_float("subsample", *search.subsample),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search.colsample_bytree
            ),
        }
        if self.direct_cohort_config.family == "nb2":
            parameters["dispersion"] = trial.suggest_float(
                "dispersion", *search.nb2_dispersion, log=True
            )
        return parameters

    def _make_estimator(
        self, parameters: Mapping[str, int | float | str], *, seed: int
    ) -> LGBMRegressor:
        """Build a seeded LightGBM regressor with this family's objective."""
        family = self.direct_cohort_config.family
        estimator_parameters = dict(parameters)
        dispersion = estimator_parameters.pop("dispersion", None)
        if family in ("poisson", "normal"):
            objective: str | object = _LIGHTGBM_OBJECTIVES[family]
        elif family == "nb2":
            if dispersion is None:
                raise ValueError(
                    "NB2 estimators require a tuned `dispersion` hyperparameter"
                )

            def objective(
                y_true: np.ndarray, y_pred: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
                """Return NB2 gradients and Hessians at the tuned dispersion."""
                return nb2_gradient_hessian(
                    y_true,
                    y_pred,
                    dispersion=float(dispersion),
                    raw_score_bounds=_RAW_SCORE_BOUNDS,
                )
        else:
            raise ValueError(f"Unsupported direct-cohort family: {family}")

        return LGBMRegressor(
            objective=objective,
            random_state=seed,
            n_jobs=self.direct_cohort_config.lightgbm_n_jobs,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            subsample_freq=1 if float(estimator_parameters["subsample"]) < 1 else 0,
            **estimator_parameters,
        )

    def _ancillary_kwargs(self, cohort: str) -> dict[str, float | None]:
        """Return the ancillary arguments this family's kernels take.

        `pointwise_log_probability` and `draw_outcomes` share one signature
        across families, but each family fills a different slot: NB2 a
        dispersion, Normal a scale, Poisson neither. Building the pair here
        keeps the three call sites identical and leaves one place to change.
        """
        family = self.direct_cohort_config.family
        if family == "nb2":
            return {"dispersion": self._ancillary_parameters[cohort], "scale": None}
        if family == "normal":
            return {"dispersion": None, "scale": self._ancillary_parameters[cohort]}
        return {"dispersion": None, "scale": None}

    def _predict_mean(
        self, booster: lightgbm.Booster, features: pd.DataFrame
    ) -> np.ndarray:
        """Predict cohort means from fitted trees; the one prediction path.

        Every caller passes a Booster (``estimator.booster_`` for a freshly
        fitted wrapper) so tuning, bootstrap refits, fitted, and reloaded models
        predict identically. Built-in objectives apply their link inside
        ``predict``; the custom NB2 objective returns raw scores, exponentiated
        here. The thread count is explicit because a bare Booster otherwise
        uses every OpenMP thread, unlike the scikit-learn wrapper, and that
        crashes alongside torch's OpenMP runtime on macOS.
        """
        predictions = np.asarray(
            booster.predict(
                features, num_threads=self.direct_cohort_config.lightgbm_n_jobs
            ),
            dtype=float,
        )
        if self.direct_cohort_config.family == "nb2":
            predictions = np.exp(np.clip(predictions, *_RAW_SCORE_BOUNDS))
        return np.clip(predictions, self.direct_cohort_config.minimum_mean, None)

    def _validation_score(
        self,
        observed: np.ndarray,
        predictions: np.ndarray,
        parameters: Mapping[str, int | float | str],
    ) -> float:
        """Return RMSE for Normal, else mean negative log probability."""
        family = self.direct_cohort_config.family
        if family == "normal":
            return float(np.sqrt(np.mean((observed - predictions) ** 2)))
        dispersion = float(parameters["dispersion"]) if family == "nb2" else None
        return float(
            -np.mean(
                pointwise_log_probability(
                    family, observed, predictions, dispersion=dispersion
                )
            )
        )

    def _cross_fitted_normal_scale(
        self,
        cohort: str,
        parameters: Mapping[str, int | float | str],
        fold_transforms: list[_FoldTransform],
        rng: np.random.Generator,
    ) -> float:
        """Estimate a Normal scale as the RMSE of out-of-fold residuals, floored."""
        residuals: list[np.ndarray] = []
        for fold, fit_features, validation_features in fold_transforms:
            fit_seed = self._derive_backend_seed(
                rng, purpose=f"normal-scale/{cohort}/fold-{fold.fold_index}"
            )
            estimator = self._make_estimator(parameters, seed=fit_seed)
            estimator.fit(fit_features, fold.fit_df[cohort].to_numpy())
            residuals.append(
                fold.validation_df[cohort].to_numpy(dtype=float)
                - self._predict_mean(estimator.booster_, validation_features)
            )
        scale = float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))
        return max(scale, self.direct_cohort_config.minimum_scale)

    def _predict_model(
        self,
        *,
        eval_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        prediction_config: PredictionConfig,
        rng: np.random.Generator,
    ) -> PredictionResult:
        """Predict cohort means and their total, plus requested scores and draws."""
        if log_exposure is not None:
            raise ValueError("DirectCohortModel does not accept an exposure offset")
        cohorts = self.schema.cohort_target_columns
        cohort_means = np.column_stack(
            [
                self._predict_mean(self._fitted_boosters[cohort], features)
                for cohort in cohorts
            ]
        )
        total_mean = cohort_means.sum(axis=1)

        pointwise = None
        if prediction_config.include_pointwise_log_probabilities:
            pointwise = self._pointwise_log_probabilities(
                eval_df, cohort_means=cohort_means, total_mean=total_mean
            )

        predictive_draws = None
        prediction_intervals = None
        if prediction_config.n_predictive_draws or prediction_config.interval_levels:
            n_refits = max(
                self.direct_cohort_config.bootstrap_replicates,
                prediction_config.n_predictive_draws,
            )
            draws = self._bootstrap_predictive_draws(
                eval_df=eval_df, n_refits=n_refits, rng=rng
            )
            n_reported = prediction_config.n_predictive_draws or n_refits
            predictive_draws = {
                name: values[:, :n_reported] for name, values in draws.items()
            }
            if prediction_config.interval_levels:
                prediction_intervals = {
                    name: central_prediction_intervals(
                        values, prediction_config.interval_levels
                    )
                    for name, values in draws.items()
                }

        return PredictionResult.from_means(
            building_ids=eval_df[self.schema.building_id_column].to_numpy(),
            cohort_names=cohorts,
            total_mean=total_mean,
            cohort_means=cohort_means,
            predictive_draws=predictive_draws,
            prediction_intervals=prediction_intervals,
            interval_levels=prediction_config.interval_levels,
            pointwise_log_probabilities=pointwise,
            parametric_distributions=self._parametric_distributions(),
            validation_config=self.prediction_validation_config,
            # Independent marginal cohort scores plus a convolved total: the
            # keys double-count the total and do not sum to a joint score.
            pointwise_log_probability_scope=(
                "marginal" if pointwise is not None else None
            ),
        )

    def _parametric_distributions(self) -> dict[str, ParametricDistributionSpec]:
        """Declare each target's distribution; NB2 declares no ``total`` entry."""
        family = self.direct_cohort_config.family
        cohorts = self.schema.cohort_target_columns
        if family == "poisson":
            return {
                "total": ParametricDistributionSpec("poisson"),
                **{cohort: ParametricDistributionSpec("poisson") for cohort in cohorts},
            }
        if family == "normal":
            scales = np.array([self._ancillary_parameters[cohort] for cohort in cohorts])
            return {
                "total": ParametricDistributionSpec(
                    "normal", scale=float(np.sqrt(np.sum(scales**2)))
                ),
                **{
                    cohort: ParametricDistributionSpec(
                        "normal", scale=self._ancillary_parameters[cohort]
                    )
                    for cohort in cohorts
                },
            }
        if family == "nb2":
            # No `total` entry: a sum of independent NB2 variables is not NB2,
            # so the observed total is scored by convolution instead of by a
            # declared family (plan section 6, item 6).
            return {
                cohort: ParametricDistributionSpec(
                    "nb2", dispersion=self._ancillary_parameters[cohort]
                )
                for cohort in cohorts
            }
        raise ValueError(f"Unsupported direct-cohort family: {family}")

    def _pointwise_log_probabilities(
        self,
        eval_df: pd.DataFrame,
        *,
        cohort_means: np.ndarray,
        total_mean: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Score each observed cohort and the total; NB2 totals use convolution."""
        cohorts = self.schema.cohort_target_columns
        required = (self.schema.total_target_column, *cohorts)
        missing = [column for column in required if column not in eval_df.columns]
        if missing:
            raise ValueError(
                f"Pointwise log probabilities require observed targets: {sorted(missing)}"
            )
        family = self.direct_cohort_config.family
        result: dict[str, np.ndarray] = {}
        for index, cohort in enumerate(cohorts):
            result[cohort] = pointwise_log_probability(
                family,
                eval_df[cohort].to_numpy(dtype=float),
                cohort_means[:, index],
                **self._ancillary_kwargs(cohort),
            )
        observed_total = eval_df[self.schema.total_target_column].to_numpy(dtype=float)
        if family == "nb2":
            result["total"] = nb2_total_log_probability(
                observed_total,
                cohort_means,
                np.array([self._ancillary_parameters[cohort] for cohort in cohorts]),
            )
        else:
            total_scale = None
            if family == "normal":
                scales = np.array([self._ancillary_parameters[cohort] for cohort in cohorts])
                total_scale = float(np.sqrt(np.sum(scales**2)))
            result["total"] = pointwise_log_probability(
                family, observed_total, total_mean, scale=total_scale
            )
        return result

    def _bootstrap_predictive_draws(
        self,
        *,
        eval_df: pd.DataFrame,
        n_refits: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Refit selected trees on resampled neighborhoods and draw outcomes."""
        train_df = self._require_training_frame(self._train_df)
        feature_spec = self._feature_spec
        if feature_spec is None:
            raise RuntimeError("Model must be fitted before generating predictive draws")
        resampler = NeighborhoodClusterResampler.from_frame(
            train_df, self.schema.neighborhood_id_column
        )
        cohorts = self.schema.cohort_target_columns
        cohort_draws = {
            cohort: np.empty((len(eval_df), n_refits), dtype=float) for cohort in cohorts
        }
        family = self.direct_cohort_config.family
        for refit_index in range(n_refits):
            row_indices = resampler.sample_row_indices(rng)
            resampled = train_df.iloc[row_indices].reset_index(drop=True)
            transformer = FittedFeatureTransformer(feature_spec, schema=self.schema)
            refit_features = transformer.fit_transform(resampled)
            eval_features = transformer.transform(eval_df)
            for cohort in cohorts:
                fit_seed = self._derive_backend_seed(
                    rng, purpose=f"bootstrap-refit/{cohort}/{refit_index}"
                )
                estimator = self._make_estimator(
                    self._selected_parameters[cohort], seed=fit_seed
                )
                estimator.fit(refit_features, resampled[cohort].to_numpy())
                means = self._predict_mean(estimator.booster_, eval_features)
                outcome_seed = self._derive_backend_seed(
                    rng, purpose=f"bootstrap-outcome/{cohort}/{refit_index}"
                )
                cohort_draws[cohort][:, refit_index] = draw_outcomes(
                    family,
                    means,
                    rng=np.random.default_rng(outcome_seed),
                    **self._ancillary_kwargs(cohort),
                )
        return {"total": sum(cohort_draws.values()), **cohort_draws}

    def _model_configuration(self) -> Mapping[str, object]:
        """Return the direct-cohort configuration: family, tuning, and search space."""
        return {"direct_cohort_config": asdict(self.direct_cohort_config)}

    def _export_model_state(self) -> dict[str, object]:
        """Return the fitted trees, family parameters, and tuning evidence.

        Each cohort's trees are stored in LightGBM's own text model format,
        which its loader reads back directly; nothing is pickled. Tuning results
        are included so a reloaded model reports the same metadata. Fold
        building IDs and the retained training frame are deliberately omitted.
        """
        return {
            "direct_cohort_config": asdict(self.direct_cohort_config),
            "boosters": {
                cohort: booster.model_to_string()
                for cohort, booster in self._fitted_boosters.items()
            },
            "selected_parameters": self._selected_parameters,
            "ancillary_parameters": self._ancillary_parameters,
            "tuning_results": {
                cohort: asdict(result) for cohort, result in self._tuning_results.items()
            },
        }

    @classmethod
    def _from_model_state(cls, state: Mapping[str, Any], **base_arguments: Any) -> Self:
        """Rebuild LightGBM boosters and fitted parameters from exported state."""
        model = cls(
            direct_cohort_config=restore_config(
                DEFAULT_DIRECT_COHORT_CONFIG, state["direct_cohort_config"]
            ),
            **base_arguments,
        )
        model._fitted_boosters = {
            cohort: lightgbm.Booster(model_str=text)
            for cohort, text in state["boosters"].items()
        }
        model._selected_parameters = copy.deepcopy(
            {cohort: dict(values) for cohort, values in state["selected_parameters"].items()}
        )
        model._ancillary_parameters = {
            cohort: float(value) for cohort, value in state["ancillary_parameters"].items()
        }
        model._tuning_results = {
            cohort: TuningResult.from_dict(payload)
            for cohort, payload in state["tuning_results"].items()
        }
        return model

    def _attach_training_frame(self, train_df: pd.DataFrame) -> None:
        """Retain a deep copy of the training frame for bootstrap refits."""
        self._train_df = train_df.copy(deep=True)

    def _get_model_metadata(self) -> Mapping[str, object]:
        """Return the family, selected hyperparameters, and tuning diagnostics."""
        family = self.direct_cohort_config.family
        family_name = {"poisson": "Poisson", "normal": "Normal", "nb2": "NB2"}[
            family
        ]
        return {
            "implementation_version": _IMPLEMENTATION_VERSION,
            "likelihood": f"three independent {family_name} cohort likelihoods",
            "parameterization": f"cohort-specific LightGBM {family} objective",
            "hyperparameters": self._selected_parameters,
            "priors": None,
            "calibration": None,
            "dependency_versions": {
                "lightgbm": lightgbm.__version__,
                "optuna": optuna.__version__,
                "numpy": np.__version__,
            },
            "uncertainty_method": (
                f"neighborhood-cluster bootstrap refits with {family} outcome draws"
            ),
            "diagnostics": {
                "family": family,
                # Recorded because it differs from the family's name for
                # Normal, and tracking tags each family run with it.
                "lightgbm_objective": _LIGHTGBM_OBJECTIVES[family],
                "ancillary_parameters": self._ancillary_parameters,
                "tuning": {
                    cohort: asdict(result)
                    for cohort, result in self._tuning_results.items()
                },
                "search_space": asdict(self.direct_cohort_config.search_space),
                "normal_mean_policy": "clip to positive minimum",
            },
        }


__all__ = ["DirectCohortModel"]