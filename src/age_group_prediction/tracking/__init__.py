"""Log a finished cross-validation comparison to MLflow.

Tracking is an after-the-fact adapter (plan section 11): the experiment runner
never imports MLflow and gains no tracking hook, so a comparison produces the
same ``CrossValidationExperimentResult`` whether or not it is tracked. This
package only reads that result. Install it with ``uv sync --group tracking``.

Typical use::

    with tracked_comparison(TrackingContext(source_revision=sha)) as handle:
        result = run_cross_model_validation(..., capture_artifacts=True)
        log_experiment_result(result, handle=handle, train_df=train_df)

Running the comparison inside the block means a failure or an interrupt is
recorded on the parent run (``FAILED`` or ``KILLED``, with ``failure_type`` and
``failure_message`` tags) instead of leaving no trace.

**Configuration** comes only from the environment:

=====================================  =======================================
``MLFLOW_TRACKING_URI``                default ``sqlite:///mlflow.db``
``MLFLOW_EXPERIMENT_NAME``             default ``age-group-prediction``
``AGE_GROUP_MLFLOW_ARTIFACT_LOCATION`` default ``mlartifacts/`` beside a SQLite
                                       database file; for any other URI the
                                       tracking server decides
=====================================  =======================================

The artifact variable is project-specific because MLflow defines none: without
it, client-side artifacts land in ``./mlruns`` relative to whatever directory
the process started in. An existing experiment whose artifact location differs
is refused rather than silently used.

**Run layout.** One parent run per comparison and one nested child run per
candidate, so each ``DirectCohortModel`` family is its own sibling run.

* Parent: data, split and seed tags; run-setting, split-summary and
  package-version params; ``freeze.json``, ``provenance.json``,
  ``selections.json``,
  ``fold_identities.json`` (the only copy of each fold's building IDs),
  ``metric_comparability.json``, and comparison-wide tables.
* Child: candidate identity and selection outcome tags; the candidate
  descriptor as params; per-fold, cross-validation, duration, tuning and
  convergence metrics; each fold's metadata, runner seeds, gzipped state
  bundle and reload check; tuning trial tables; this candidate's slices of the
  result tables.

Tables are written only when they have rows: importance, for example, exists
only for selected candidates.

**Completeness.** Every run's final write is the tag ``evidence_complete=true``,
children before the parent. A run that fails or is interrupted, even after
logging finished, is overwritten to ``evidence_complete=false``. A process
killed outright leaves a run ``RUNNING`` without the tag. Treat a comparison as
complete only when its status is ``FINISHED`` *and* the tag is ``true``.

Every run also carries the caller's ``TrackingContext`` tags. MLflow's own
inferred ``mlflow.source.git.*`` tags are removed, because they describe the
working directory's last commit rather than the code that ran.

Fold models get no MLflow LoggedModel or model flavor. They are evaluation
evidence, stored as plain JSON state bundles; a loadable model wrapper belongs
to the final full-training refit (Gate 8).

**Package layout.** Import only from this package; the submodules are
internal. Dependencies run one way, from top to bottom:

* ``_files``: param flattening and JSON, gzipped JSON and CSV writers.
* ``settings``: ``TrackingSettings`` and environment resolution.
* ``_metadata``: readers for a result's provenance and model metadata.
* ``runs``: ``TrackingContext``, ``tracked_comparison`` and how runs end.
* ``candidates``: one nested child run per candidate.
* ``evidence``: the logging entry points, preflight checks and parent run.
"""

from __future__ import annotations

from .evidence import log_cross_validation_experiment, log_experiment_result
from .final import log_final_evaluation_result, run_final_evaluation
from .runs import (
    EVIDENCE_COMPLETE_TAG,
    FINAL_RUN_ROLE,
    ComparisonRunHandle,
    FinalRunHandle,
    TrackingContext,
    tracked_comparison,
    tracked_final_evaluation,
)
from .settings import (
    ARTIFACT_LOCATION_ENV,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    TrackingSettings,
    resolve_tracking_settings,
)

__all__ = [
    "ARTIFACT_LOCATION_ENV",
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_TRACKING_URI",
    "EVIDENCE_COMPLETE_TAG",
    "FINAL_RUN_ROLE",
    "ComparisonRunHandle",
    "FinalRunHandle",
    "TrackingContext",
    "TrackingSettings",
    "log_cross_validation_experiment",
    "log_experiment_result",
    "log_final_evaluation_result",
    "resolve_tracking_settings",
    "run_final_evaluation",
    "tracked_comparison",
    "tracked_final_evaluation",
]
