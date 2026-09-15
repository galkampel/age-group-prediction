"""Run lifecycle: the caller's context, the parent run, and how every run ends."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mlflow
from mlflow.entities import ViewType

from ._files import _tag_bool, _write_json
from .settings import (
    ARTIFACT_LOCATION_ENV,
    TrackingSettings,
    _normalize_location,
    resolve_tracking_settings,
)

EVIDENCE_COMPLETE_TAG = "evidence_complete"
_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"

# MLflow's tag value limit is 8000 characters; failure messages stay well below.
_MAX_FAILURE_MESSAGE_LENGTH = 5000
_PARENT_TAG = "mlflow.parentRunId"
FINAL_RUN_ROLE = "final_evaluation"
# Tags this package owns. Caller-supplied extra tags may not overwrite them.
_RESERVED_TAGS = frozenset(
    {
        EVIDENCE_COMPLETE_TAG,
        "failure_type",
        "failure_message",
        "run_role",
        "test_lock_status",
        "source_dirty",
        "manifest_fingerprint",
        "training_data_hash",
        "training_schema_hash",
        "master_seed",
        "master_seed_source",
        "source_cv_run_id",
        "cross_family_rule_version",
        "selected_candidate_id",
        "selected_approach",
        "cross_family_decision_hash",
        "final_attempt_fingerprint",
        "retry_of_run_ids",
    }
)


@dataclass(frozen=True)
class TrackingContext:
    """Run identity the package cannot know and the caller must supply.

    The package never runs Git: ``source_revision`` and ``source_dirty`` come
    from the caller, and are left unset when the code is not committed.
    """

    source_revision: str | None = None
    source_dirty: bool | None = None
    test_lock_status: str = "locked"
    run_name: str | None = None
    extra_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse extra tags that would overwrite tags the adapter owns."""
        tags = {str(key): str(value) for key, value in self.extra_tags.items()}
        clashes = sorted(
            key for key in tags if key in _RESERVED_TAGS or key.startswith("mlflow.")
        )
        if clashes:
            raise ValueError(f"extra_tags may not set reserved tags: {clashes}")
        object.__setattr__(self, "extra_tags", MappingProxyType(tags))

    def with_test_lock_status(self, status: str) -> TrackingContext:
        """Return this context with a different ``test_lock_status`` tag value."""
        return replace(self, test_lock_status=status)


@dataclass(frozen=True)
class ComparisonRunHandle:
    """The open parent run that a comparison's evidence is logged into."""

    run_id: str
    experiment_id: str
    settings: TrackingSettings
    context: TrackingContext


@contextmanager
def tracked_comparison(
    context: TrackingContext,
    *,
    settings: TrackingSettings | None = None,
) -> Iterator[ComparisonRunHandle]:
    """Open the parent run for one comparison and record how it ends.

    Run the comparison and ``log_experiment_result`` inside the block. An
    exception ends the run ``FAILED`` and a ``KeyboardInterrupt`` ends it
    ``KILLED``, both with failure tags, and either is re-raised. MLflow's own
    context manager would mark an interrupt ``FAILED``, which is why the run
    is ended here explicitly.
    """
    settings = settings or resolve_tracking_settings()
    if mlflow.active_run() is not None:
        raise RuntimeError(
            "An MLflow run is already active; a comparison must be its own "
            "top-level run. End the active run first."
        )
    previous_environment_uri = os.environ.get(_TRACKING_URI_ENV)
    mlflow.set_tracking_uri(settings.tracking_uri)
    try:
        experiment_id = _ensure_experiment(settings)
        with _terminating_run(
            context,
            experiment_id=experiment_id,
            run_name=context.run_name,
            tags={"run_role": "comparison"},
        ) as run_id:
            yield ComparisonRunHandle(
                run_id=run_id,
                experiment_id=experiment_id,
                settings=settings,
                context=context,
            )
    finally:
        _restore_tracking_uri(previous_environment_uri)


@dataclass(frozen=True)
class FinalRunHandle:
    """The open top-level final-evaluation parent that Gate 8 evidence logs into."""

    run_id: str
    experiment_id: str
    settings: TrackingSettings
    context: TrackingContext
    source_cv_run_id: str
    manifest_fingerprint: str


@contextmanager
def tracked_final_evaluation(
    context: TrackingContext,
    *,
    source_cv_run_id: str,
    manifest_fingerprint: str,
    cross_family_rule_version: str,
    selected_candidate_id: str,
    selected_approach: str,
    pretest_freeze: Mapping[str, Any],
    final_attempt_fingerprint: str,
    settings: TrackingSettings | None = None,
) -> Iterator[FinalRunHandle]:
    """Open a new top-level final-evaluation parent, linked to its source CV run.

    Follows the order the brief requires (session handoff section 9.1):
    the source CV parent must already be a complete ``run_role=comparison``
    run; no complete final run may already exist for the same
    ``source_cv_run_id`` and ``manifest_fingerprint``; the parent then opens
    ``test_lock_status=locked`` and the pretest freeze (which must already
    carry the cross-family decision) is logged before the tag is flipped to
    ``opened`` and control is handed to the caller. This never reopens or
    appends to the completed CV parent.

    The pretest freeze must carry a cross-family decision matching the
    selected-candidate tags; its hash is tagged ``cross_family_decision_hash``.
    ``final_attempt_fingerprint`` identifies everything the attempt refits and
    scores (``experiment.final_evaluation.final_attempt_fingerprint``) and is
    tagged under that name. Once any final run on this manifest has reached
    ``opened``, a new run is refused unless it has the identical source CV run,
    decision, and attempt fingerprint. An allowed retry is tagged
    ``retry_of_run_ids``.
    """
    if context.test_lock_status != "locked":
        raise ValueError(
            "A final-evaluation context must start 'locked'; the lockbox is "
            "opened by this function only after the pretest freeze is logged"
        )
    decision = pretest_freeze.get("cross_family_selection")
    if decision is None:
        raise ValueError(
            "The pretest freeze must carry the cross-family decision before a "
            "final-evaluation run may open"
        )
    if (
        decision.get("selected_candidate_id") != selected_candidate_id
        or decision.get("selected_approach") != selected_approach
    ):
        raise ValueError(
            "The selected candidate tags do not match the pretest freeze's "
            "cross-family decision"
        )
    if not final_attempt_fingerprint:
        raise ValueError("A final-evaluation run requires a final attempt fingerprint")
    decision_hash = _decision_hash(decision)
    settings = settings or resolve_tracking_settings()
    if mlflow.active_run() is not None:
        raise RuntimeError(
            "An MLflow run is already active; a final evaluation must be its "
            "own top-level run. End the active run first."
        )
    previous_environment_uri = os.environ.get(_TRACKING_URI_ENV)
    mlflow.set_tracking_uri(settings.tracking_uri)
    try:
        experiment_id = _ensure_experiment(settings)
        _require_complete_source_run(experiment_id, source_cv_run_id)
        _refuse_duplicate_final_run(experiment_id, source_cv_run_id, manifest_fingerprint)
        retry_of_run_ids = _retryable_opened_attempts(
            experiment_id,
            source_cv_run_id=source_cv_run_id,
            manifest_fingerprint=manifest_fingerprint,
            decision_hash=decision_hash,
            final_attempt_fingerprint=final_attempt_fingerprint,
        )
        tags = {
            "run_role": FINAL_RUN_ROLE,
            "source_cv_run_id": source_cv_run_id,
            "manifest_fingerprint": manifest_fingerprint,
            "cross_family_rule_version": cross_family_rule_version,
            "selected_candidate_id": selected_candidate_id,
            "selected_approach": selected_approach,
            "cross_family_decision_hash": decision_hash,
            "final_attempt_fingerprint": final_attempt_fingerprint,
        }
        if retry_of_run_ids:
            tags["retry_of_run_ids"] = ",".join(retry_of_run_ids)
        with _terminating_run(
            context,
            experiment_id=experiment_id,
            run_name=context.run_name,
            tags=tags,
        ) as run_id:
            with tempfile.TemporaryDirectory() as directory:
                _write_json(Path(directory) / "pretest_freeze.json", dict(pretest_freeze))
                mlflow.log_artifacts(directory)
            # Only after the pretest freeze is durably logged does the
            # lockbox open; a crash before this line leaves the run
            # incomplete rather than opened with nothing recorded.
            mlflow.set_tag("test_lock_status", "opened")
            yield FinalRunHandle(
                run_id=run_id,
                experiment_id=experiment_id,
                settings=settings,
                context=context,
                source_cv_run_id=source_cv_run_id,
                manifest_fingerprint=manifest_fingerprint,
            )
    finally:
        _restore_tracking_uri(previous_environment_uri)


def _require_complete_source_run(experiment_id: str, source_cv_run_id: str) -> None:
    """Refuse a source CV run that is missing, foreign, or not a finished comparison."""
    run = mlflow.get_run(source_cv_run_id)
    if run.info.experiment_id != experiment_id:
        raise ValueError(
            f"Source CV run {source_cv_run_id!r} belongs to a different experiment"
        )
    if (
        run.info.status != "FINISHED"
        or run.data.tags.get("run_role") != "comparison"
        or run.data.tags.get(EVIDENCE_COMPLETE_TAG) != "true"
    ):
        raise ValueError(
            f"Source CV run {source_cv_run_id!r} is not a complete comparison run"
        )


def _refuse_duplicate_final_run(
    experiment_id: str, source_cv_run_id: str, manifest_fingerprint: str
) -> None:
    """Refuse a second complete final run for the same source run and manifest."""
    existing = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.run_role = '{FINAL_RUN_ROLE}' and "
            f"tags.source_cv_run_id = '{source_cv_run_id}' and "
            f"tags.manifest_fingerprint = '{manifest_fingerprint}' and "
            f"tags.{EVIDENCE_COMPLETE_TAG} = 'true'"
        ),
        # Deleting a completed run does not make its evaluation unseen.
        run_view_type=ViewType.ALL,
        output_format="list",
    )
    if existing:
        raise RuntimeError(
            "A completed final evaluation run already exists for this source "
            "CV run and manifest fingerprint"
        )


def _decision_hash(decision: Mapping[str, Any]) -> str:
    """SHA-256 of a cross-family decision's canonical JSON, comparable across runs."""
    payload = json.dumps(decision, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _retryable_opened_attempts(
    experiment_id: str,
    *,
    source_cv_run_id: str,
    manifest_fingerprint: str,
    decision_hash: str,
    final_attempt_fingerprint: str,
) -> list[str]:
    """Return earlier opened final runs this run may retry; refuse any other.

    Once a final run has flipped ``test_lock_status`` to ``opened`` for a
    manifest, test evidence exists even if that run later failed: a failed run
    keeps whatever children, predictions and metrics it logged. A later run on
    the same manifest is therefore only a retry of the identical attempt: the
    same source CV run, the same decision, and the same final attempt
    fingerprint (the refit models' configuration, seeds, prediction and
    evaluation settings). Otherwise a different decision, a new CV run, or a
    re-tuned or re-seeded refit could be chosen after seeing that evidence.
    Deleted runs are searched too, because deleting a run does not unsee its
    results.
    """
    attempts = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.run_role = '{FINAL_RUN_ROLE}' and "
            f"tags.manifest_fingerprint = '{manifest_fingerprint}' and "
            "tags.test_lock_status = 'opened'"
        ),
        run_view_type=ViewType.ALL,
        output_format="list",
    )
    for attempt in attempts:
        tags = attempt.data.tags
        if (
            tags.get("source_cv_run_id") != source_cv_run_id
            or tags.get("cross_family_decision_hash") != decision_hash
            or tags.get("final_attempt_fingerprint") != final_attempt_fingerprint
        ):
            raise RuntimeError(
                "The lockbox was already opened for this manifest by final run "
                f"{attempt.info.run_id!r} with a different source CV run, "
                "cross-family decision, or refit and scoring configuration; "
                "test evidence from that attempt must not feed a new choice"
            )
    return sorted(attempt.info.run_id for attempt in attempts)


def _restore_tracking_uri(environment_uri: str | None) -> None:
    """Undo ``mlflow.set_tracking_uri``, which is process-global.

    MLflow also writes the URI into ``MLFLOW_TRACKING_URI``. Restoring only
    MLflow's own setting would pin it, and the environment, to this
    comparison's store, so a later change to the variable would be silently
    ignored. The pinned setting is cleared instead and the variable is put back
    exactly as found (absent included), so MLflow reads the environment again.
    """
    # Annotated `str | Path`, but MLflow's implementation treats None as "unset".
    mlflow.set_tracking_uri(None)  # type: ignore[arg-type]
    os.environ.pop(_TRACKING_URI_ENV, None)
    if environment_uri is not None:
        os.environ[_TRACKING_URI_ENV] = environment_uri


def _ensure_experiment(settings: TrackingSettings) -> str:
    """Return the configured experiment's ID, creating it if it does not exist."""
    client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        return client.create_experiment(
            settings.experiment_name, artifact_location=settings.artifact_location
        )
    if experiment.lifecycle_stage != "active":
        raise ValueError(
            f"MLflow experiment '{settings.experiment_name}' is deleted; restore it "
            "or set MLFLOW_EXPERIMENT_NAME to a different name"
        )
    if settings.artifact_location is not None and (
        _normalize_location(experiment.artifact_location) != settings.artifact_location
    ):
        raise ValueError(
            f"MLflow experiment '{settings.experiment_name}' stores artifacts in "
            f"{experiment.artifact_location}, but the configured location is "
            f"{settings.artifact_location}. MLflow cannot move an experiment's "
            "artifacts; set MLFLOW_EXPERIMENT_NAME to a new name, or set "
            f"{ARTIFACT_LOCATION_ENV} to the existing location."
        )
    return experiment.experiment_id


def _context_tags(context: TrackingContext) -> dict[str, str]:
    """Tags from the caller's context, written on every run."""
    tags = {"test_lock_status": context.test_lock_status}
    if context.source_revision is not None:
        tags["mlflow.source.git.commit"] = context.source_revision
    if context.source_dirty is not None:
        tags["source_dirty"] = _tag_bool(context.source_dirty)
    tags.update(context.extra_tags)
    return tags


@contextmanager
def _terminating_run(
    context: TrackingContext, *, tags: Mapping[str, str], **start_arguments: Any
) -> Iterator[str]:
    """Start a run and end it with a status that tells failures apart.

    Both parent and child runs use this, so every run carries the caller's
    context tags, and an interrupt during a child's logging marks the child
    ``KILLED`` as well as the parent.
    """
    supplied_tags = {**tags, **_context_tags(context)}
    run = mlflow.start_run(tags=supplied_tags, **start_arguments)
    try:
        _drop_inferred_git_tags(run.info.run_id, supplied_tags)
        yield run.info.run_id
    except KeyboardInterrupt as error:
        _end_unsuccessful_run(error, status="KILLED")
        raise
    except BaseException as error:
        _end_unsuccessful_run(error, status="FAILED")
        raise
    mlflow.end_run("FINISHED")


def _drop_inferred_git_tags(run_id: str, supplied_tags: Mapping[str, str]) -> None:
    """Remove the Git tags MLflow infers from the working directory's HEAD.

    Source identity comes only from the caller. MLflow tags the commit, branch
    and remote of whatever repository the process started in, which describes
    uncommitted code as the last commit, so any such tag the caller did not
    supply is deleted.
    """
    client = mlflow.MlflowClient()
    for key in mlflow.get_run(run_id).data.tags:
        if key.startswith("mlflow.source.git.") and key not in supplied_tags:
            client.delete_tag(run_id, key)


def _end_unsuccessful_run(error: BaseException, *, status: str) -> None:
    """Tag the active run with the failure, mark it incomplete, and end it."""
    try:
        mlflow.set_tags(
            {
                # Overwritten even if logging had finished: a failed or
                # interrupted run must never read as complete.
                EVIDENCE_COMPLETE_TAG: "false",
                "failure_type": type(error).__name__,
                "failure_message": str(error)[:_MAX_FAILURE_MESSAGE_LENGTH],
            }
        )
    finally:
        # The status matters more than the tags, so it is set even if tagging
        # itself fails (for example because the store is unreachable).
        mlflow.end_run(status)
