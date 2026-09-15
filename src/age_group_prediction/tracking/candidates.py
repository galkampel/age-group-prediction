"""Child runs: one nested run per candidate with its tags, metrics and artifacts."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from ..experiment import CrossValidationExperimentResult, FoldRunEvidence
from ._files import (
    _flatten,
    _log_params,
    _tag_bool,
    _trials_table,
    _write_csv,
    _write_gzipped_json,
    _write_json,
)
from ._metadata import (
    _data_tags,
    _declared_family,
    _diagnostics,
    _search_space,
    _tuning_results,
)
from .runs import (
    _MAX_FAILURE_MESSAGE_LENGTH,
    _PARENT_TAG,
    EVIDENCE_COMPLETE_TAG,
    ComparisonRunHandle,
    _terminating_run,
)


def _log_candidate_run(
    result: CrossValidationExperimentResult,
    *,
    descriptor: Mapping[str, Any],
    freeze: Mapping[str, Any],
    provenance: Mapping[str, Any],
    handle: ComparisonRunHandle,
) -> None:
    """Log one candidate as a nested child run, marking it complete last."""
    candidate_id = descriptor["candidate_id"]
    runs = sorted(
        (run for run in result.fold_runs if run.candidate_id == candidate_id),
        key=lambda run: run.fold_identity.fold_index,
    )
    with _terminating_run(
        handle.context,
        tags={"run_role": "candidate"},
        # A child does not inherit its parent's experiment; without this it
        # lands in MLflow's default experiment and artifact location.
        experiment_id=handle.experiment_id,
        run_name=candidate_id,
        nested=True,
        parent_run_id=handle.run_id,
    ) as run_id:
        # MLflow silently creates a top-level run if nesting fails, so the
        # hierarchy is verified rather than assumed.
        child = mlflow.get_run(run_id)
        if (
            child.data.tags.get(_PARENT_TAG) != handle.run_id
            or child.info.experiment_id != handle.experiment_id
        ):
            raise RuntimeError(
                f"Child run for '{candidate_id}' is not nested under the comparison"
            )
        mlflow.set_tags(
            {
                **_data_tags(provenance),
                **_candidate_tags(descriptor, freeze, runs),
            }
        )
        _log_candidate_metrics(result, candidate_id, runs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _log_params(_flatten("", descriptor), root)
            _write_candidate_artifacts(result, candidate_id, runs, root)
            mlflow.log_artifacts(directory)
        mlflow.set_tag(EVIDENCE_COMPLETE_TAG, "true")


def _candidate_tags(
    descriptor: Mapping[str, Any],
    freeze: Mapping[str, Any],
    runs: Sequence[FoldRunEvidence],
) -> dict[str, str]:
    """Identity, family, and selection-outcome tags for one candidate run."""
    candidate_id = descriptor["candidate_id"]
    metadata = runs[0].model_metadata
    diagnostics = _diagnostics(metadata)
    configuration = descriptor["configuration"]
    tags = {
        "candidate_id": candidate_id,
        "approach": descriptor["approach"],
        "model_class": str(metadata["model_class"]),
        "selection_role": descriptor["selection_role"],
    }
    # Family is a candidate configuration choice, not a candidate field; the
    # fitted model's own diagnostics are the fallback.
    family = _declared_family(configuration) or diagnostics.get("family")
    if family is not None:
        tags["family"] = str(family)
    if "lightgbm_objective" in diagnostics:
        # The objective LightGBM actually optimized, which is not always the
        # family's name (Normal trains with `regression`).
        tags["objective_family"] = str(diagnostics["lightgbm_objective"])

    selection = next(
        (
            item
            for item in freeze["selections"]
            if item["approach"] == descriptor["approach"]
        ),
        None,
    )
    tags["selected"] = _tag_bool(
        selection is not None and selection["selected_candidate_id"] == candidate_id
    )
    if descriptor["selection_role"] != "eligible":
        tags["rejection_reason"] = "diagnostic comparator; reported, never ranked"
    elif selection is not None and candidate_id in selection["rejection_reasons"]:
        tags["rejection_reason"] = str(selection["rejection_reasons"][candidate_id])[
            :_MAX_FAILURE_MESSAGE_LENGTH
        ]
    return tags


def _log_candidate_metrics(
    result: CrossValidationExperimentResult,
    candidate_id: str,
    runs: Sequence[FoldRunEvidence],
) -> None:
    """Log per-fold metrics (stepped by fold index) and cross-validation summaries."""
    fold_metrics = result.fold_metrics_df[
        result.fold_metrics_df["candidate_id"] == candidate_id
    ]
    for run in runs:
        fold_index = run.fold_identity.fold_index
        metadata = run.model_metadata
        metrics = {
            _metric_key("fold", row): float(row["value"])
            for _, row in fold_metrics[
                fold_metrics["fold_index"] == fold_index
            ].iterrows()
        }
        # Logged separately and never summed: evaluate time contains predict
        # time when a model evaluates itself.
        for operation in ("fit", "predict", "evaluate"):
            duration = metadata.get(f"{operation}_duration_seconds")
            if duration is not None:
                metrics[f"duration/{operation}_seconds"] = float(duration)
        for component, tuning in _tuning_results(metadata).items():
            metrics[f"tuning/{component}/best_value"] = float(tuning["best_value"])
        metrics.update(_convergence_metrics(_diagnostics(metadata)))
        mlflow.log_metrics(metrics, step=fold_index)

    aggregate = result.aggregate_metrics_df[
        result.aggregate_metrics_df["candidate_id"] == candidate_id
    ]
    summary: dict[str, float] = {}
    for _, row in aggregate.iterrows():
        summary[_metric_key("cv_mean", row)] = float(row["mean"])
        summary[_metric_key("cv_std", row)] = float(row["standard_deviation"])
    mlflow.log_metrics(summary)


def _metric_key(prefix: str, row: pd.Series) -> str:
    """Build ``{prefix}/{metric}/{target}/{level}`` from a metric-table row."""
    return f"{prefix}/{row['metric_name']}/{row['target']}/{row['aggregation_level']}"


def _convergence_metrics(diagnostics: Mapping[str, Any]) -> dict[str, float]:
    """Chartable convergence summaries for stages that record a policy outcome.

    Pyro's classic ESS is not constrained to be positive; a short profile has
    produced -3554. A negative value is not a sample size, so the chart shows
    it clamped at zero with ``ess_valid=0``, and the raw value stays in the
    fold's ``metadata.json``.
    """
    metrics: dict[str, float] = {}
    for stage, payload in sorted(diagnostics.items()):
        if not isinstance(payload, Mapping) or "policy_passed" not in payload:
            continue
        rhat = payload.get("worst_rhat")
        if rhat is not None and math.isfinite(rhat):
            metrics[f"diag/{stage}/max_rhat"] = float(rhat)
        ess = payload.get("minimum_effective_sample_size")
        ess_valid = ess is not None and math.isfinite(ess) and ess > 0
        metrics[f"diag/{stage}/min_ess_clamped"] = float(ess) if ess_valid else 0.0
        metrics[f"diag/{stage}/ess_valid"] = float(ess_valid)
        metrics[f"diag/{stage}/policy_passed"] = float(bool(payload["policy_passed"]))
    return metrics


def _write_candidate_artifacts(
    result: CrossValidationExperimentResult,
    candidate_id: str,
    runs: Sequence[FoldRunEvidence],
    root: Path,
) -> None:
    """Write one candidate's fold evidence, tuning tables, and table slices."""
    bundles = {
        item.fold_index: item
        for item in result.artifacts
        if item.candidate_id == candidate_id
    }
    for run in runs:
        fold_index = run.fold_identity.fold_index
        fold_dir = root / "folds" / f"fold_{fold_index}"
        _write_json(fold_dir / "metadata.json", dict(run.model_metadata))
        # The runner's per-purpose seeds; the model's own derived seeds are in
        # the metadata.
        _write_json(fold_dir / "seeds.json", dict(run.seeds))
        _write_gzipped_json(
            fold_dir / "state_bundle.json.gz", dict(bundles[fold_index].state_bundle)
        )
        _write_json(
            fold_dir / "reload_check.json", bundles[fold_index].reload_check.to_dict()
        )
        for component, tuning in _tuning_results(run.model_metadata).items():
            _write_csv(
                root / "tuning" / f"fold_{fold_index}" / f"{component}_trials.csv",
                _trials_table(tuning),
            )

    search_space = _search_space(runs[0].model_metadata)
    if search_space is not None:
        _write_json(root / "search_space.json", search_space)

    importance = result.importance_df
    if "validation_building_ids" in importance.columns:
        # A complete ID list repeated on every row; the parent's
        # fold_identities.json holds each fold's validation IDs once.
        importance = importance.drop(columns="validation_building_ids")
    for name, frame in (
        ("predictions", result.predictions_df),
        ("fold_metrics", result.fold_metrics_df),
        ("calibration", result.calibration_df),
        ("importance", importance),
    ):
        if "candidate_id" in frame.columns:
            _write_csv(
                root / f"{name}.csv", frame[frame["candidate_id"] == candidate_id]
            )
