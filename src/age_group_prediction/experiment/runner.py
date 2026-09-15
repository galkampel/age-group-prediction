"""Model-agnostic cross-validation orchestration over fixed training folds."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ..data_splitting import SplitManifest, ValidationFold
from ..evaluation import evaluate_predictions, neighborhood_cluster_bootstrap
from ..experiment_config import ExperimentConfig
from ..modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    EvaluationConfig,
    ModelingSchema,
)
from ..models.base import BaseAgeGroupModel
from .aggregation import (
    _aggregate_fold_metrics,
    _bootstrap_interval_evidence,
    _summarize_importance,
)
from .artifacts import FoldArtifactEvidence, capture_fold_artifact
from .contracts import (
    _APPROACH_NAMES,
    ApproachName,
    CandidateDefinition,
    SelectionPolicy,
)
from .evidence import (
    CrossValidationExperimentResult,
    FoldRunEvidence,
    SelectionFreeze,
    _build_provenance,
    _calibration_record,
    _manifest_fingerprint,
    _prediction_records,
)
from .importance import _compute_selected_importance, _validate_importance_specs
from .partitions import (
    _assert_prediction_alignment,
    _fold_coverage,
    _sorted_ids,
    validate_experiment_partitions,
)
from .seeds import _stable_seed
from .selection import (
    _convergence_failures,
    _select_candidates,
    _validate_bayesian_feature_freeze,
    _validate_likelihood_comparability,
)


def _resolve_master_seed(
    master_seed: int | None,
    experiment_config: ExperimentConfig | None,
) -> tuple[int, str]:
    """Resolve the experiment seed and report where it came from."""
    if master_seed is not None:
        resolved, source = master_seed, "explicit_argument"
    elif experiment_config is not None:
        resolved = experiment_config.randomness.default_seed
        source = "experiment_config.randomness.default_seed"
    else:
        raise ValueError(
            "Provide master_seed or experiment_config so the run is never "
            "seeded by an implicit default"
        )
    if resolved < 0:
        raise ValueError("Master seed must be nonnegative")
    return resolved, source


def run_cross_model_validation(
    outer_train_df: pd.DataFrame,
    *,
    split_manifest: SplitManifest,
    validation_folds: Sequence[ValidationFold],
    candidates: Sequence[CandidateDefinition],
    selection_policies: Sequence[SelectionPolicy],
    master_seed: int | None = None,
    experiment_config: ExperimentConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
    schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
    required_approaches: Sequence[ApproachName] = (
        "DirectCohortModel",
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    ),
    require_convergence: bool = True,
    capture_artifacts: bool = False,
) -> CrossValidationExperimentResult:
    """Compare candidates on fixed training-only folds and freeze selections.

    The master seed and evaluation config resolve in the precedence declared in
    section 4: explicit argument, then loaded ``ExperimentConfig``, and never a
    silent default.

    Section 9.3 makes convergence an explicit selection constraint, so a
    candidate whose fitted model recorded ``policy_passed=False`` for any stage
    on any fold is excluded from ranking and recorded as rejected with its
    failures. Models that record no such diagnostics are unconstrained. Pass
    ``require_convergence=False`` to rank unconverged candidates anyway, which
    an exploratory run may legitimately want.

    Pass ``capture_artifacts=True`` to export every candidate's state bundle on
    every fold and prove, before the model is released, that the bundle
    reloads to identical point predictions (``experiment.artifacts``). The
    check uses its own generators, so it changes no result. A bundle that does
    not reload aborts the run. Tracking requires this evidence; the default
    skips it because bundles for every candidate and fold are held in memory.
    """
    master_seed, seed_source = _resolve_master_seed(master_seed, experiment_config)
    if evaluation_config is None and experiment_config is not None:
        evaluation_config = experiment_config.evaluation
    fold_identities = validate_experiment_partitions(
        outer_train_df,
        split_manifest=split_manifest,
        validation_folds=validation_folds,
        schema=schema,
    )
    candidate_by_id, policy_by_approach = _validate_candidate_registry(
        candidates,
        selection_policies,
        schema=schema,
        required_approaches=required_approaches,
    )
    folds_by_index = {fold.fold_index: fold for fold in validation_folds}
    fitted_models: dict[tuple[str, int], BaseAgeGroupModel] = {}
    fold_runs: list[FoldRunEvidence] = []
    metric_frames: list[pd.DataFrame] = []
    prediction_records: list[dict[str, Any]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    bootstrap_replicate_frames: list[pd.DataFrame] = []
    calibration_records: list[dict[str, Any]] = []
    artifacts: list[FoldArtifactEvidence] = []

    for candidate_id in sorted(candidate_by_id):
        candidate = candidate_by_id[candidate_id]
        for identity in fold_identities:
            fold = folds_by_index[identity.fold_index]
            seeds = {
                operation: _stable_seed(
                    master_seed,
                    f"candidate/{candidate_id}/fold/{identity.fold_index}/{operation}",
                )
                for operation in ("fit", "predict", "evaluate", "bootstrap")
            }
            model = candidate.model_factory()
            if not isinstance(model, BaseAgeGroupModel):
                raise TypeError("Candidate factories must return BaseAgeGroupModel")
            # A model failing mid-run raises from deep inside an estimator or a
            # third-party library, where nothing names the candidate or the
            # fold. Without this context the caller cannot tell which of a
            # dozen candidate-fold combinations died.
            with _fold_failure_context(candidate_id, identity.fold_index, "fit"):
                model.fit(
                    fold.fit_df,
                    feature_spec=candidate.fit_feature_spec,
                    rng=np.random.default_rng(seeds["fit"]),
                )
            with _fold_failure_context(candidate_id, identity.fold_index, "predict"):
                prediction = model.predict(
                    fold.validation_df,
                    prediction_config=candidate.prediction_config,
                    rng=np.random.default_rng(seeds["predict"]),
                )
                _assert_prediction_alignment(
                    prediction,
                    fold.validation_df,
                    id_column=schema.building_id_column,
                )
            with _fold_failure_context(candidate_id, identity.fold_index, "evaluate"):
                evaluation = evaluate_predictions(
                    fold.validation_df,
                    prediction,
                    candidate.metrics,
                    rng=np.random.default_rng(seeds["evaluate"]),
                )
            bootstrap = None
            if evaluation_config is not None:
                with _fold_failure_context(
                    candidate_id, identity.fold_index, "bootstrap"
                ):
                    bootstrap = neighborhood_cluster_bootstrap(
                        fold.validation_df,
                        prediction,
                        candidate.metrics,
                        config=evaluation_config,
                        # Pass the seed only: supplying an rng makes the
                        # bootstrap record default_seed=None and lose its seed
                        # provenance.
                        default_seed=seeds["bootstrap"],
                    )
                intervals = bootstrap.intervals_df.assign(
                    candidate_id=candidate_id,
                    approach=candidate.approach,
                    fold_index=identity.fold_index,
                )
                bootstrap_frames.append(intervals)
                bootstrap_replicate_frames.append(
                    bootstrap.replicate_metrics_df.assign(
                        candidate_id=candidate_id,
                        approach=candidate.approach,
                        fold_index=identity.fold_index,
                    )
                )

            metadata = model.get_metadata()
            # After the metadata is taken, because the reload check predicts
            # again and a prediction updates the model's duration records.
            if capture_artifacts:
                with _fold_failure_context(
                    candidate_id, identity.fold_index, "capture_artifacts"
                ):
                    artifacts.append(
                        capture_fold_artifact(
                            model,
                            fold.validation_df,
                            candidate_id=candidate_id,
                            fold_index=identity.fold_index,
                        )
                    )
            fold_runs.append(
                FoldRunEvidence(
                    candidate_id=candidate_id,
                    approach=candidate.approach,
                    fold_identity=identity,
                    prediction=prediction,
                    evaluation=evaluation,
                    bootstrap=bootstrap,
                    model_metadata=metadata,
                    seeds=seeds,
                )
            )
            fitted_models[(candidate_id, identity.fold_index)] = model
            metric_frames.append(
                evaluation.metrics_df.assign(
                    candidate_id=candidate_id,
                    approach=candidate.approach,
                    fold_index=identity.fold_index,
                )
            )
            prediction_records.extend(
                _prediction_records(candidate, identity, prediction)
            )
            calibration_records.append(
                _calibration_record(candidate, identity, metadata)
            )

    fold_metrics = pd.concat(metric_frames, ignore_index=True)
    aggregate_metrics = _aggregate_fold_metrics(fold_metrics)
    _validate_likelihood_comparability(
        candidate_by_id, policy_by_approach, fold_runs
    )
    selections = _select_candidates(
        candidate_by_id,
        policy_by_approach,
        aggregate_metrics,
        _convergence_failures(fold_runs),
        require_convergence=require_convergence,
    )
    _validate_bayesian_feature_freeze(candidate_by_id, selections)
    # Only selected candidates are interpreted, so the rest are released before
    # importance starts. A Bayesian posterior runs to tens of megabytes per
    # fold, and the losers are never consulted again.
    selected_ids = {selection.selected_candidate_id for selection in selections}
    fitted_models = {
        key: model for key, model in fitted_models.items() if key[0] in selected_ids
    }
    importance = _compute_selected_importance(
        candidates=candidate_by_id,
        selections=selections,
        fold_identities=fold_identities,
        folds_by_index=folds_by_index,
        fitted_models=fitted_models,
        master_seed=master_seed,
        schema=schema,
    )
    bootstrap_intervals = _bootstrap_interval_evidence(
        fold_intervals=bootstrap_frames,
        replicate_frames=bootstrap_replicate_frames,
        aggregate_metrics=aggregate_metrics,
        evaluation_config=evaluation_config,
    )
    freeze = SelectionFreeze(
        manifest_fingerprint=_manifest_fingerprint(split_manifest),
        outer_training_building_ids=_sorted_ids(set(outer_train_df[schema.building_id_column])),
        fold_identities=fold_identities,
        candidate_descriptors=tuple(
            candidate_by_id[candidate_id].to_descriptor()
            for candidate_id in sorted(candidate_by_id)
        ),
        selections=selections,
        master_seed=master_seed,
        master_seed_source=seed_source,
    )
    return CrossValidationExperimentResult(
        fold_runs=tuple(fold_runs),
        fold_metrics_df=fold_metrics,
        aggregate_metrics_df=aggregate_metrics,
        bootstrap_intervals_df=bootstrap_intervals,
        predictions_df=pd.DataFrame(prediction_records),
        calibration_df=pd.DataFrame(calibration_records),
        importance_df=importance,
        importance_summary_df=_summarize_importance(importance),
        fold_coverage_df=_fold_coverage(
            freeze.outer_training_building_ids, fold_identities
        ),
        selections=selections,
        freeze=freeze,
        provenance=_build_provenance(
            outer_train_df,
            schema=schema,
            split_manifest=split_manifest,
            candidates=candidate_by_id,
            master_seed=master_seed,
            master_seed_source=seed_source,
            run_settings={
                # Recorded so a consumer can re-hash the training frame the same
                # way without being handed the schema separately.
                "building_id_column": schema.building_id_column,
                "evaluation_config": (
                    asdict(evaluation_config) if evaluation_config is not None else None
                ),
                "require_convergence": require_convergence,
                "required_approaches": list(required_approaches),
                "capture_artifacts": capture_artifacts,
                "fold_count": len(fold_identities),
            },
        ),
        artifacts=tuple(artifacts),
    )


@contextmanager
def _fold_failure_context(
    candidate_id: str, fold_index: int, operation: str
) -> Iterator[None]:
    """Name the candidate, fold and operation on any failure inside the block.

    The run still aborts: with one candidate's evidence missing there is no
    defined way to rank the rest. Containing the failure and continuing would
    change what selection compares, so the failure is surfaced rather than
    swallowed.
    """
    try:
        yield
    except Exception as error:
        raise RuntimeError(
            f"Candidate '{candidate_id}' failed during {operation} on fold "
            f"{fold_index}: {type(error).__name__}: {error}"
        ) from error


def _validate_candidate_registry(
    candidates: Sequence[CandidateDefinition],
    selection_policies: Sequence[SelectionPolicy],
    *,
    schema: ModelingSchema,
    required_approaches: Sequence[ApproachName],
) -> tuple[dict[str, CandidateDefinition], dict[str, SelectionPolicy]]:
    """Check candidates and policies cover the required approaches consistently."""
    if not candidates:
        raise ValueError("At least one model candidate is required")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    policy_by_approach = {policy.approach: policy for policy in selection_policies}
    if len(policy_by_approach) != len(selection_policies):
        raise ValueError("Selection policies must have unique approaches")
    eligible_approaches = {
        candidate.approach
        for candidate in candidates
        if candidate.selection_role == "eligible"
    }
    required = set(required_approaches)
    if len(required) != len(required_approaches) or not required <= _APPROACH_NAMES:
        raise ValueError("Required approaches must be unique public model names")
    if eligible_approaches != required:
        raise ValueError(
            "Eligible candidates must exactly cover the required model approaches"
        )
    if eligible_approaches != set(policy_by_approach):
        raise ValueError(
            "Selection policies must exactly cover approaches with eligible candidates"
        )
    for candidate in candidates:
        components = {spec.component for spec in candidate.component_feature_specs}
        expected_components = (
            {"tree"}
            if candidate.approach == "DirectCohortModel"
            else {"total_count", "age_probability"}
        )
        if components != expected_components:
            raise ValueError(
                f"{candidate.approach} must declare feature components "
                f"{sorted(expected_components)}"
            )
        for spec in candidate.component_feature_specs:
            spec.validate_for_schema(schema)
        _validate_importance_specs(candidate, schema=schema)
        if candidate.selection_role == "eligible":
            _validate_policy_metrics(candidate, policy_by_approach[candidate.approach])
    return candidate_by_id, policy_by_approach


def _validate_policy_metrics(
    candidate: CandidateDefinition,
    policy: SelectionPolicy,
) -> None:
    """Check a candidate declares every policy metric with a matching direction."""
    available = {
        (metric.name, metric.target, metric.aggregation_level): metric
        for metric in candidate.metrics
    }
    for criterion in policy.criteria:
        for reference in criterion.metric_references:
            key = (
                reference.metric_name,
                reference.target,
                reference.aggregation_level,
            )
            metric = available.get(key)
            if metric is None:
                raise ValueError(
                    f"Candidate '{candidate.candidate_id}' does not declare selection "
                    f"metric {reference.metric_name}:{reference.target}:"
                    f"{reference.aggregation_level}"
                )
            # A criterion that disagrees with the metric's own declared direction
            # silently inverts the ranking and selects the worst candidate.
            if metric.optimization_direction != criterion.optimization_direction:
                raise ValueError(
                    f"Selection criterion '{criterion.name}' declares "
                    f"{criterion.optimization_direction} for metric "
                    f"{reference.metric_name}:{reference.target}:"
                    f"{reference.aggregation_level}, which declares "
                    f"{metric.optimization_direction}"
                )
