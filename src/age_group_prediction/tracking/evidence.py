"""Log a finished comparison: preflight checks, the parent run, then one child per candidate.

The child runs themselves are written by ``candidates``.
"""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import mlflow.data
import pandas as pd

from ..experiment import CrossValidationExperimentResult, FoldRunEvidence
from ..experiment.selection import (
    _CONTINUOUS_FAMILIES,
    _DISCRETE_FAMILIES,
    _DISTRIBUTIONAL_CAPABILITIES,
    _candidate_measure_kinds,
)
from ..hashing import column_schema_hash, table_hash
from ._files import _flatten, _log_params, _write_csv, _write_json
from ._metadata import _data_tags, _declared_family, _diagnostics
from .candidates import _log_candidate_run
from .runs import (
    _PARENT_TAG,
    EVIDENCE_COMPLETE_TAG,
    ComparisonRunHandle,
    TrackingContext,
    tracked_comparison,
)
from .settings import TrackingSettings


def log_experiment_result(
    result: CrossValidationExperimentResult,
    *,
    handle: ComparisonRunHandle,
    train_df: pd.DataFrame | None = None,
) -> None:
    """Log a finished comparison into the parent run ``handle`` names.

    The result must come from ``run_cross_model_validation(...,
    capture_artifacts=True)``: without a reload-checked bundle for every
    candidate and fold the evidence would be incomplete, so it is refused
    before anything is written. ``train_df``, when given, must hash to the
    recorded outer training frame and is logged as the run's training input.
    """
    _require_loggable(result, train_df)
    active = mlflow.active_run()
    if active is None or active.info.run_id != handle.run_id:
        raise RuntimeError(
            "log_experiment_result must be called inside the tracked_comparison "
            "block that opened this handle"
        )
    # One comparison per parent: a second call would add a second set of
    # children and blur which evidence the parent's tags describe.
    existing_children = mlflow.search_runs(
        experiment_ids=[handle.experiment_id],
        filter_string=f"tags.`{_PARENT_TAG}` = '{handle.run_id}'",
        output_format="list",
    )
    if existing_children or EVIDENCE_COMPLETE_TAG in active.data.tags:
        raise RuntimeError(
            "This comparison run already holds logged evidence; open a new "
            "tracked_comparison for another result"
        )
    freeze = result.freeze.to_dict()
    provenance = result.provenance.to_dict()
    _log_parent_evidence(result, freeze=freeze, provenance=provenance)
    if train_df is not None:
        _log_training_input(train_df)
    for descriptor in freeze["candidate_descriptors"]:
        _log_candidate_run(
            result,
            descriptor=descriptor,
            freeze=freeze,
            provenance=provenance,
            handle=handle,
        )
    # Last write: the parent is complete only once every child is.
    mlflow.set_tag(EVIDENCE_COMPLETE_TAG, "true")


def log_cross_validation_experiment(
    result: CrossValidationExperimentResult,
    context: TrackingContext,
    *,
    train_df: pd.DataFrame | None = None,
    settings: TrackingSettings | None = None,
) -> str:
    """Log a result already in hand as a new comparison; return the parent run ID.

    Prefer ``tracked_comparison`` around the run itself, which also records a
    comparison that fails. This convenience is for a result computed earlier.
    """
    # Checked before a run is opened, so a refused result leaves no failed run.
    _require_loggable(result, train_df)
    with tracked_comparison(context, settings=settings) as handle:
        log_experiment_result(result, handle=handle, train_df=train_df)
    return handle.run_id


# --- Preflight checks ---------------------------------------------------------


def _require_loggable(
    result: CrossValidationExperimentResult, train_df: pd.DataFrame | None
) -> None:
    """Refuse incomplete evidence or a mismatched frame before writing anything."""
    expected = {
        (run.candidate_id, run.fold_identity.fold_index) for run in result.fold_runs
    }
    captured = [(item.candidate_id, item.fold_index) for item in result.artifacts]
    if not expected or len(captured) != len(set(captured)) or set(captured) != expected:
        raise ValueError(
            "The result does not hold one state bundle per candidate and fold. "
            "Run run_cross_model_validation(..., capture_artifacts=True) before "
            "tracking it."
        )
    if not all(item.reload_check.passed for item in result.artifacts):
        raise ValueError("The result holds a state bundle that failed its reload check")
    _require_consistent_families(result)
    if train_df is None:
        return
    provenance = result.provenance
    id_column = provenance.run_settings["building_id_column"]
    observed = {
        "training_data_hash": table_hash(train_df, id_column=id_column),
        "training_schema_hash": column_schema_hash(train_df),
    }
    for label, value in observed.items():
        if getattr(provenance, label) != value:
            raise ValueError(
                f"train_df does not match the comparison: {label} is {value} but "
                f"the result records {getattr(provenance, label)}"
            )


def _require_consistent_families(result: CrossValidationExperimentResult) -> None:
    """Refuse a candidate whose declared family differs from its fitted model's.

    The ``family`` tag comes from the candidate's configuration, a label the
    caller writes. A model that records its own family (``DirectCohortModel``
    does) must agree with it on every fold, or the run would be tagged with a
    family it did not fit.
    """
    for descriptor in result.freeze.candidate_descriptors:
        declared = _declared_family(descriptor["configuration"])
        if declared is None:
            continue
        for run in result.fold_runs:
            if run.candidate_id != descriptor["candidate_id"]:
                continue
            fitted = _diagnostics(run.model_metadata).get("family")
            if fitted is not None and str(fitted) != declared:
                raise ValueError(
                    f"Candidate '{run.candidate_id}' declares family '{declared}' "
                    f"but its fitted model reports '{fitted}' on fold "
                    f"{run.fold_identity.fold_index}; correct the candidate "
                    "configuration before tracking it"
                )


# --- Parent run ---------------------------------------------------------------


def _log_training_input(train_df: pd.DataFrame) -> None:
    """Record the outer training frame as the active run's training dataset."""
    # MLflow's dataset profiling emits a pandas deprecation warning and an
    # integer-column schema hint from inside its own modules, about its own
    # internals. Only warnings raised by MLflow code are silenced, for this call.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"mlflow\.")
        dataset = mlflow.data.from_pandas(train_df, name="outer_training")
        mlflow.log_input(dataset, context="training")


def _log_parent_evidence(
    result: CrossValidationExperimentResult,
    *,
    freeze: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Write the parent run's identity tags, params, and comparison artifacts."""
    mlflow.set_tags(_data_tags(provenance))
    descriptors = freeze["candidate_descriptors"]
    params = {
        "candidate_count": str(len(descriptors)),
        "eligible_candidate_count": str(
            sum(item["selection_role"] == "eligible" for item in descriptors)
        ),
        "fold_count": str(len(freeze["fold_identities"])),
        "outer_training_building_count": str(
            len(freeze["outer_training_building_ids"])
        ),
        "selection_count": str(len(freeze["selections"])),
        "mlflow_version": mlflow.__version__,
        **_flatten("split", provenance["split_summary"]),
        **_flatten("run_settings", provenance["run_settings"]),
        **_flatten("package", provenance["package_versions"]),
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _log_params(params, root)
        _write_json(root / "freeze.json", freeze)
        _write_json(root / "provenance.json", provenance)
        _write_json(root / "selections.json", freeze["selections"])
        _write_json(
            root / "fold_identities.json",
            {
                f"fold_{identity['fold_index']}": identity
                for identity in freeze["fold_identities"]
            },
        )
        _write_json(
            root / "metric_comparability.json",
            _metric_comparability(result, freeze),
        )
        for name, frame in (
            ("aggregate_metrics", result.aggregate_metrics_df),
            ("bootstrap_intervals", result.bootstrap_intervals_df),
            ("fold_coverage", result.fold_coverage_df),
            ("importance_summary", result.importance_summary_df),
        ):
            _write_csv(root / f"{name}.csv", frame)
        mlflow.log_artifacts(directory)


def _metric_comparability(
    result: CrossValidationExperimentResult, freeze: Mapping[str, Any]
) -> dict[str, Any]:
    """State which scores may be compared, from the rules selection enforces.

    The family sets and capability names are the ones
    ``experiment.selection`` uses to refuse ranking log densities against log
    masses, so this record cannot drift from the behavior it describes.
    """
    return {
        "rules": {
            "selection_scope": "one selection per approach; approaches are never "
            "ranked against each other",
            "discrete_families": sorted(_DISCRETE_FAMILIES),
            "continuous_families": sorted(_CONTINUOUS_FAMILIES),
            "distributional_capabilities": sorted(_DISTRIBUTIONAL_CAPABILITIES),
            "distributional_criteria_may_mix_discrete_and_continuous": False,
            "diagnostic_comparators_are_ranked": False,
            # Model A's pointwise keys are marginal log masses; Model B's and
            # the Bayesian model's are conditional on the preceding cohorts, so
            # a per-target predictive_nll is not one quantity across approaches.
            "predictive_nll_is_ranked_across_approaches": False,
            "predictive_measure_kinds_basis": (
                "families of the parametric distributions a candidate declares; "
                "empty when it declares none and is scored through pointwise log "
                "probabilities instead"
            ),
        },
        "policies": {
            selection["approach"]: {
                "likelihood_comparability": selection["policy"][
                    "likelihood_comparability"
                ],
                "criteria": selection["policy"]["criteria"],
            }
            for selection in freeze["selections"]
        },
        "candidates": [
            {
                "candidate_id": descriptor["candidate_id"],
                "approach": descriptor["approach"],
                "selection_role": descriptor["selection_role"],
                "predictive_measure_kinds": sorted(
                    _candidate_measure_kinds(
                        descriptor["candidate_id"], result.fold_runs
                    )
                ),
                "pointwise_log_probability_scopes": _pointwise_scopes(
                    descriptor["candidate_id"], result.fold_runs
                ),
                "metrics": [
                    {
                        key: metric.get(key)
                        for key in (
                            "name",
                            "target",
                            "aggregation_level",
                            "required_capability",
                            "optimization_direction",
                            "interpretation",
                        )
                    }
                    for metric in descriptor["metrics"]
                ],
            }
            for descriptor in freeze["candidate_descriptors"]
        ],
    }


def _pointwise_scopes(
    candidate_id: str, fold_runs: Sequence[FoldRunEvidence]
) -> list[str]:
    """The pointwise log-probability scopes a candidate's predictions declared."""
    return sorted(
        {
            run.prediction.pointwise_log_probability_scope
            for run in fold_runs
            if run.candidate_id == candidate_id
            and run.prediction.pointwise_log_probability_scope is not None
        }
    )
