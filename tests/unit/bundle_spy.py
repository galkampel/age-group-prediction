"""A state-bundle-capable spy model and a small five-candidate spy experiment.

The spy models in ``test_experiment.py`` implement no state-bundle hooks, so a
run with ``capture_artifacts=True`` cannot use them. This spy does, and its
scripted metadata imitates what each real model family records (tuning
studies, search spaces, Bayesian convergence diagnostics), so artifact and
tracking tests run in milliseconds. Real models are covered by the slow
validation tests.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from age_group_prediction.data_splitting import (
    FoldConfig,
    OuterSplitConfig,
    make_validation_folds,
    split_known_neighborhood_buildings,
)
from age_group_prediction.experiment import (
    CandidateDefinition,
    CrossValidationExperimentResult,
    FeatureBlock,
    MetricReference,
    PermutationImportanceSpec,
    SelectionCriterion,
    SelectionPolicy,
    run_cross_model_validation,
)
from age_group_prediction.experiment_config import load_experiment_config
from age_group_prediction.metrics import (
    MeanAbsoluteError,
    ParametricPredictiveNegativeLogLikelihood,
)
from age_group_prediction.modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
)
from age_group_prediction.models.base import BaseAgeGroupModel
from age_group_prediction.results import ParametricDistributionSpec, PredictionResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COHORTS = tuple(DEFAULT_MODELING_SCHEMA.cohort_target_columns)
TOTAL = DEFAULT_MODELING_SCHEMA.total_target_column
# A negative ESS, as Pyro's classic estimator produced on a short profile.
NEGATIVE_ESS = -3554.19


class BundleSpyModel(BaseAgeGroupModel):
    """Deterministic means from ``ses``, with scripted diagnostics and a bundle."""

    implementation_version = "bundle-spy-1"

    def __init__(
        self,
        *,
        offset: float = 0.0,
        family: str | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        fit_error: BaseException | None = None,
        **base_arguments: Any,
    ) -> None:
        """Store the offset, predictive family, scripted diagnostics, and fit error.

        ``fit_error``, when given, is raised from ``_fit_model`` instead of
        fitting, so a Gate 8 final-refit failure path (a Bayesian diagnostic
        error carrying ``stage``/``diagnostics``/``failures``, as
        ``models.bayesian_conditional`` raises) can be exercised without a
        real NUTS run.
        """
        super().__init__(**base_arguments)
        self.offset = offset
        self.family = family
        self.diagnostics = copy.deepcopy(dict(diagnostics or {}))
        self.fit_error = fit_error

    def _reset_model_state(self) -> None:
        """Nothing to reset: the spy's state is its constructor arguments."""

    def _fit_model(self, *, train_df, features, log_exposure, rng) -> None:
        """Fit nothing, or raise the scripted ``fit_error`` if one was given."""
        del train_df, features, log_exposure, rng
        if self.fit_error is not None:
            raise self.fit_error

    def _predict_model(
        self, *, eval_df, features, log_exposure, prediction_config, rng
    ) -> PredictionResult:
        """Predict ``ses + offset`` totals, split 20/50/30 across cohorts."""
        del features, log_exposure, prediction_config, rng
        total = np.clip(eval_df["ses"].to_numpy(dtype=float) + self.offset, 0.01, None)
        distributions = None
        if self.family is not None:
            spec = ParametricDistributionSpec(
                family=self.family,
                dispersion=0.5 if self.family == "nb2" else None,
                scale=1.0 if self.family == "normal" else None,
            )
            distributions = {target: spec for target in ("total", *COHORTS)}
        return PredictionResult.from_means(
            building_ids=eval_df["building_id"].to_numpy(),
            cohort_names=COHORTS,
            total_mean=total,
            cohort_means=total[:, None] * np.array([0.2, 0.5, 0.3]),
            parametric_distributions=distributions,
        )

    def _get_model_metadata(self) -> Mapping[str, object]:
        """Return metadata shaped like a real model's, with scripted diagnostics."""
        return {
            "implementation_version": self.implementation_version,
            "likelihood": f"spy {self.family}",
            "parameterization": "offset",
            "hyperparameters": {"offset": self.offset},
            "priors": None,
            "calibration": None,
            "dependency_versions": {},
            "uncertainty_method": "none",
            "diagnostics": copy.deepcopy(self.diagnostics),
        }

    def _model_configuration(self) -> Mapping[str, object]:
        """Return the constructor arguments that decide what the spy predicts."""
        return {
            "offset": self.offset,
            "family": self.family,
            "diagnostics": copy.deepcopy(self.diagnostics),
            "fit_error": None if self.fit_error is None else repr(self.fit_error),
        }

    def _export_model_state(self) -> dict[str, object]:
        """Bundle the constructor arguments, which are the whole fitted state."""
        return {
            "offset": self.offset,
            "family": self.family,
            "diagnostics": copy.deepcopy(self.diagnostics),
        }

    @classmethod
    def _from_model_state(cls, state: Mapping[str, Any], **base_arguments: Any):
        """Rebuild the spy from ``_export_model_state`` output."""
        return cls(
            offset=float(state["offset"]),
            family=state["family"],
            diagnostics=state["diagnostics"],
            **base_arguments,
        )


def tuning_record(best_value: float) -> dict[str, Any]:
    """``asdict(TuningResult)``-shaped evidence with one pruned trial."""
    return {
        "best_params": {"learning_rate": 0.1},
        "best_value": best_value,
        "best_trial_number": 1,
        "trials": [
            {
                "number": 0,
                "state": "COMPLETE",
                "value": best_value + 0.2,
                "params": {"learning_rate": 0.05},
                "intermediate_values": [[0, best_value + 0.3], [1, best_value + 0.2]],
            },
            {
                "number": 1,
                "state": "COMPLETE",
                "value": best_value,
                "params": {"learning_rate": 0.1},
                "intermediate_values": [[0, best_value + 0.1], [1, best_value]],
            },
            {
                "number": 2,
                "state": "PRUNED",
                "value": best_value + 0.5,
                "params": {"learning_rate": 0.3},
                "intermediate_values": [[0, best_value + 0.5]],
            },
        ],
    }


SEARCH_SPACE = {"learning_rate": [0.01, 0.3]}
# Distinct per family, so a test can tell one family's tuning evidence from
# another's; identical studies would hide evidence logged into the wrong run.
DIRECT_TUNING_BEST_VALUES = {"poisson": 1.0, "nb2": 1.4, "normal": 1.8}


def direct_diagnostics(family: str) -> dict[str, Any]:
    """Diagnostics shaped like ``DirectCohortModel``'s: per-cohort tuning."""
    return {
        "family": family,
        "lightgbm_objective": {"normal": "regression"}.get(family, family),
        "tuning": {
            cohort: tuning_record(DIRECT_TUNING_BEST_VALUES[family])
            for cohort in COHORTS
        },
        "search_space": SEARCH_SPACE,
    }


def independent_diagnostics() -> dict[str, Any]:
    """Diagnostics shaped like Model B's: tuning under ``selection``."""
    return {
        "selection": {
            "total_tuning": tuning_record(2.0),
            "probability_tuning": tuning_record(0.8),
            "search_space": SEARCH_SPACE,
        }
    }


def bayesian_diagnostics() -> dict[str, Any]:
    """Per-stage convergence diagnostics, with a negative ESS on the total stage."""
    return {
        "total": {
            "worst_rhat": 1.2,
            "minimum_effective_sample_size": NEGATIVE_ESS,
            "policy_passed": False,
            "policy_failures": ["worst_rhat=1.2 violates threshold 1.05"],
        },
        "composition": {
            "worst_rhat": 1.01,
            "minimum_effective_sample_size": 400.0,
            "policy_passed": True,
            "policy_failures": [],
        },
    }


def modeling_table() -> pd.DataFrame:
    """Twelve buildings in three neighborhoods, valid under the default schema."""
    rows: list[dict[str, object]] = []
    for neighborhood_id in range(3):
        for local_id in range(4):
            kindergarten = local_id % 2
            elementary = 1 + (local_id % 3)
            highschool = (local_id + 1) % 2
            rows.append(
                {
                    "building_id": neighborhood_id * 10 + local_id,
                    "neighborhood_id": neighborhood_id,
                    "ses": 0.2 + 0.1 * local_id,
                    "avg_household_size": 2.0 + 0.1 * neighborhood_id,
                    "median_age": 30.0 + local_id,
                    "n_daycares_500m": local_id,
                    "n_apartments": 20 + local_id,
                    "3_rooms_share": 0.2,
                    "4_rooms_share": 0.3,
                    "5_rooms_share": 0.25,
                    "school_status": ("none", "existing", "planned")[neighborhood_id],
                    "n_kindergarten": kindergarten,
                    "n_elementary": elementary,
                    "n_highschool": highschool,
                    "n_children_total": kindergarten + elementary + highschool,
                }
            )
    table = pd.DataFrame(rows, columns=DEFAULT_MODELING_SCHEMA.table_columns)
    DEFAULT_MODELING_SCHEMA.validate_table(table)
    return table


def split_and_folds():
    """Split the table into outer training and holdout, then two training folds."""
    split = split_known_neighborhood_buildings(
        modeling_table(),
        config=OuterSplitConfig(test_fraction=0.25),
        rng=np.random.default_rng(4),
    )
    folds = make_validation_folds(
        split.train_df, config=FoldConfig(n_folds=2), rng=np.random.default_rng(9)
    ).folds
    return split, folds


def _importance(component: str) -> tuple[PermutationImportanceSpec, ...]:
    """One single-repeat SES permutation-importance spec for ``component``."""
    return (
        PermutationImportanceSpec(
            component=component,
            metric=MeanAbsoluteError(TOTAL),
            feature_blocks=(FeatureBlock("ses", ("ses",)),),
            repeats=1,
        ),
    )


def spy_candidates(
    model_class: type[BundleSpyModel] = BundleSpyModel,
) -> tuple[CandidateDefinition, ...]:
    """Three direct families (Normal as a comparator) plus B and Bayesian stand-ins."""

    def direct(family: str, offset: float, role: str) -> CandidateDefinition:
        """A direct-cohort candidate for one predictive family."""
        return CandidateDefinition(
            candidate_id=f"direct-{family}",
            approach="DirectCohortModel",
            model_factory=lambda: model_class(
                offset=offset, family=family, diagnostics=direct_diagnostics(family)
            ),
            fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
            component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
            metrics=(
                MeanAbsoluteError(TOTAL),
                ParametricPredictiveNegativeLogLikelihood(
                    target=TOTAL, interpretation=f"{family} total"
                ),
            ),
            configuration={"family": family},
            selection_role=role,
            importance_specs=_importance("tree"),
        )

    conditional_specs = (DEFAULT_TOTAL_FEATURE_SPEC, DEFAULT_PROBABILITY_FEATURE_SPEC)
    return (
        direct("poisson", 0.0, "eligible"),
        direct("nb2", 0.3, "eligible"),
        direct("normal", 0.1, "diagnostic_comparator"),
        CandidateDefinition(
            candidate_id="independent-nb2",
            approach="IndependentTotalProbabilityModel",
            model_factory=lambda: model_class(
                offset=0.2, family="nb2", diagnostics=independent_diagnostics()
            ),
            fit_feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
            component_feature_specs=conditional_specs,
            metrics=(MeanAbsoluteError(TOTAL),),
            configuration={"total_family": "nb2"},
            importance_specs=_importance("total_count"),
        ),
        CandidateDefinition(
            candidate_id="bayesian-reduced",
            approach="BayesianConditionalModel",
            model_factory=lambda: model_class(
                offset=0.1, diagnostics=bayesian_diagnostics()
            ),
            fit_feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
            component_feature_specs=conditional_specs,
            metrics=(MeanAbsoluteError(TOTAL),),
            configuration={"profile": "reduced"},
            importance_specs=_importance("total_count"),
        ),
    )


def spy_policies() -> tuple[SelectionPolicy, ...]:
    """One total-MAE selection policy per public approach."""

    def policy(approach: str) -> SelectionPolicy:
        """Rank one approach's eligible candidates by total MAE."""
        return SelectionPolicy(
            approach=approach,
            criteria=(
                SelectionCriterion(
                    name="total_mae",
                    metric_references=(MetricReference("mae", TOTAL),),
                    optimization_direction="minimize",
                ),
            ),
            likelihood_comparability="Point accuracy within the approach.",
        )

    return tuple(
        policy(approach)
        for approach in (
            "DirectCohortModel",
            "IndependentTotalProbabilityModel",
            "BayesianConditionalModel",
        )
    )


def run_spy_experiment(
    *,
    capture_artifacts: bool = True,
    model_class: type[BundleSpyModel] = BundleSpyModel,
) -> tuple[Any, CrossValidationExperimentResult]:
    """Run the five spy candidates; return the outer split and the result."""
    split, folds = split_and_folds()
    config = load_experiment_config(REPOSITORY_ROOT / "configs" / "modeling.toml")
    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=spy_candidates(model_class),
        selection_policies=spy_policies(),
        master_seed=31,
        evaluation_config=replace(config.evaluation, bootstrap_replicates=3),
        # The Bayesian stand-in scripts a failed convergence policy.
        require_convergence=False,
        capture_artifacts=capture_artifacts,
    )
    return split, result
