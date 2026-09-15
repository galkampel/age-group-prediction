"""Log a finished lockbox evaluation: parent evidence, then one child per winner.

This is the Gate 8 counterpart to ``evidence.py``/``candidates.py``, but for
the one-time final evaluation rather than a cross-validation comparison: one
call logs ``FinalEvaluationResult`` (paired with the live refit models in
``refit_result``) into the parent run ``tracked_final_evaluation`` opened,
then logs each approach winner's full-training refit as a loadable
models-from-code pyfunc and proves the reload reproduces its point
predictions exactly.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from ..data_splitting import SplitManifest
from ..experiment.contracts import CandidateDefinition
from ..experiment.evidence import CrossValidationExperimentResult, SelectionFreeze
from ..experiment.final_evaluation import (
    FinalCandidateEvaluation,
    FinalEvaluationResult,
    FinalModelArtifact,
    evaluate_frozen_models_on_lockbox,
    final_attempt_fingerprint,
    refit_frozen_approach_winners,
    verify_pretest_freeze,
)
from ..modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    EvaluationConfig,
    ModelingSchema,
    PredictionConfig,
)
from ..models.base import BaseAgeGroupModel
from ..state_bundle import dependency_versions
from . import pyfunc_model
from ._files import _flatten, _log_params, _write_csv, _write_gzipped_json, _write_json
from .candidates import _metric_key
from .runs import (
    _PARENT_TAG,
    EVIDENCE_COMPLETE_TAG,
    FinalRunHandle,
    TrackingContext,
    _terminating_run,
    tracked_final_evaluation,
)
from .settings import TrackingSettings

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def log_final_evaluation_result(
    refit_result: Sequence[FinalModelArtifact],
    evaluation_result: FinalEvaluationResult,
    *,
    handle: FinalRunHandle,
    split_manifest: SplitManifest,
) -> None:
    """Log one finished lockbox evaluation into the final parent ``handle`` names.

    Refuses a finalized freeze whose manifest fingerprint does not match the
    run ``handle`` was opened with, and refuses to log into a final run that
    already holds evidence -- mirroring ``log_experiment_result``'s
    one-comparison-per-parent guard, but for the one-time final evaluation.
    """
    freeze = evaluation_result.finalized_freeze
    if freeze.manifest_fingerprint != handle.manifest_fingerprint:
        raise ValueError("Finalized freeze does not match this final run's manifest")
    if freeze.cross_family_selection is None:
        raise ValueError("Finalized freeze has no recorded cross-family selection")
    active = mlflow.active_run()
    if active is None or active.info.run_id != handle.run_id:
        raise RuntimeError(
            "log_final_evaluation_result must be called inside the "
            "tracked_final_evaluation block that opened this handle"
        )
    existing_children = mlflow.search_runs(
        experiment_ids=[handle.experiment_id],
        filter_string=f"tags.`{_PARENT_TAG}` = '{handle.run_id}'",
        output_format="list",
    )
    if existing_children or EVIDENCE_COMPLETE_TAG in active.data.tags:
        raise RuntimeError(
            "This final run already holds logged evidence; open a new "
            "tracked_final_evaluation for another result"
        )
    _log_final_parent_evidence(evaluation_result, split_manifest=split_manifest)
    artifact_by_id = {artifact.candidate_id: artifact for artifact in refit_result}
    for evaluation in evaluation_result.evaluations:
        _log_final_candidate_run(
            evaluation,
            artifact=artifact_by_id[evaluation.candidate_id],
            evaluation_result=evaluation_result,
            handle=handle,
        )
    # Last write: the parent is complete only once every child is.
    mlflow.set_tag(EVIDENCE_COMPLETE_TAG, "true")


# --- Parent run -----------------------------------------------------------


def _log_final_parent_evidence(
    evaluation_result: FinalEvaluationResult, *, split_manifest: SplitManifest
) -> None:
    """Write the final parent's params and comparison-wide artifacts."""
    freeze = evaluation_result.finalized_freeze
    cross_family = freeze.cross_family_selection
    assert cross_family is not None  # checked by the caller before this runs
    params = {
        "candidate_count": str(len(evaluation_result.evaluations)),
        "holdout_building_count": str(len(evaluation_result.holdout_building_ids)),
        "mlflow_version": mlflow.__version__,
        "selected_candidate_id": cross_family.selected_candidate_id,
        "selected_approach": cross_family.selected_approach,
        "cross_family_rule_name": cross_family.rule.name,
        "cross_family_rule_version": cross_family.rule.version,
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _log_params(params, root)
        _write_json(root / "split_manifest.json", split_manifest.to_dict())
        _write_json(root / "finalized_freeze.json", freeze.to_dict())
        _write_json(root / "cross_family_rule.json", cross_family.rule.to_dict())
        _write_json(
            root / "comparability.json",
            {
                "likelihood_comparability": cross_family.rule.likelihood_comparability,
                "candidates": {
                    evaluation.candidate_id: evaluation.metric_comparability
                    for evaluation in evaluation_result.evaluations
                },
            },
        )
        _write_json(
            root / "provenance.json",
            {
                "mlflow_version": mlflow.__version__,
                "dependency_versions": dependency_versions(
                    "numpy", "pandas", "scipy", "scikit-learn"
                ),
            },
        )
        test_metrics = pd.concat(
            [evaluation.metrics_df for evaluation in evaluation_result.evaluations],
            ignore_index=True,
        )
        _write_csv(root / "test_metrics.csv", test_metrics)
        mlflow.log_artifacts(directory)


# --- Children ---------------------------------------------------------------


def _log_final_candidate_run(
    evaluation: FinalCandidateEvaluation,
    *,
    artifact: FinalModelArtifact,
    evaluation_result: FinalEvaluationResult,
    handle: FinalRunHandle,
) -> None:
    """Log one approach winner as a nested child, marking it complete last."""
    # A child holds test predictions and metrics, so it is tagged as opened
    # evidence; the context's "locked" describes only how the parent started.
    with _terminating_run(
        handle.context.with_test_lock_status("opened"),
        tags={"run_role": "final_candidate"},
        experiment_id=handle.experiment_id,
        run_name=evaluation.candidate_id,
        nested=True,
        parent_run_id=handle.run_id,
    ) as run_id:
        child = mlflow.get_run(run_id)
        if (
            child.data.tags.get(_PARENT_TAG) != handle.run_id
            or child.info.experiment_id != handle.experiment_id
        ):
            raise RuntimeError(
                f"Final child run for '{evaluation.candidate_id}' is not "
                "nested under the final evaluation run"
            )
        mlflow.set_tags(
            {
                "candidate_id": evaluation.candidate_id,
                "approach": evaluation.approach,
                "role": evaluation.role,
            }
        )
        mlflow.log_metrics(
            {
                _metric_key("test", row): float(row["value"])
                for _, row in evaluation.metrics_df.iterrows()
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _log_params(_flatten("seeds", dict(evaluation.seeds)), root)
            _write_json(
                root / "refit_metadata.json", dict(artifact.evidence.model_metadata)
            )
            _write_json(root / "seeds.json", dict(artifact.evidence.seeds))
            _write_gzipped_json(
                root / "state_bundle.json.gz", dict(artifact.evidence.state_bundle)
            )
            _write_json(
                root / "reload_check.json", artifact.evidence.reload_check.to_dict()
            )
            _write_json(
                root / "evaluation_metadata.json", dict(evaluation.evaluation_metadata)
            )
            _write_csv(root / "predictions.csv", evaluation.predictions_df)
            _write_csv(root / "metrics.csv", evaluation.metrics_df)
            _write_csv(root / "intervals.csv", evaluation.intervals_df)
            mlflow.log_artifacts(directory)
        _log_final_model(evaluation, artifact=artifact, evaluation_result=evaluation_result)
        mlflow.set_tag(EVIDENCE_COMPLETE_TAG, "true")


def _log_final_model(
    evaluation: FinalCandidateEvaluation,
    *,
    artifact: FinalModelArtifact,
    evaluation_result: FinalEvaluationResult,
) -> None:
    """Log one refit as a loadable models-from-code pyfunc, then prove it reloads exactly.

    Ships the project's own source as ``code_paths`` and the state bundle as
    a plain JSON artifact; no cloudpickle, no native model flavor.

    The equality target is a *fresh* prediction from the live ``artifact.model``
    on ``evaluation_result.holdout_input_df``, called with no explicit RNG --
    not ``evaluation.predictions_df``, which was computed with a distinct,
    purpose-scoped seed (``final/{candidate_id}/predict``). ``PythonModel.predict``
    takes no seed argument, so the loaded pyfunc always predicts with
    ``np.random.default_rng(model.default_rng_seed)``; comparing against a
    differently-seeded prediction would only coincidentally match today (the
    project's split strategy never hands a Bayesian refit a holdout building
    in an unseen neighborhood, the one case where a point mean itself draws a
    random effect -- see ``models.bayesian_conditional``) and could
    legitimately diverge otherwise. This mirrors the identically-seeded
    comparison ``experiment.artifacts.capture_fold_artifact`` and
    ``final_evaluation._final_reload_check`` already use for the same reason.

    Every logged model is loaded back immediately, **in-process**: this
    proves the state bundle and the wrapper's predict contract round-trip
    exactly, but it does not exercise ``code_paths`` as a fresh interpreter
    would -- ``age_group_prediction`` is already imported in this process, so
    the model-from-code script's imports resolve from ``sys.modules`` rather
    than from the code copied into the artifact, and native libraries
    (LightGBM's OpenMP runtime) are already initialized. Neither a missing
    ``code_paths`` entry nor a fresh-process load failure is caught here;
    ``tests/unit/test_gate8_tracking.py::test_logged_pyfunc_loads_and_predicts_in_a_fresh_process``
    covers both.
    """
    reference = artifact.model.predict(
        evaluation_result.holdout_input_df, prediction_config=PredictionConfig()
    )
    expected = pyfunc_model.prediction_to_frame(reference)
    input_example = evaluation_result.holdout_input_df.head(5)
    signature = infer_signature(input_example, expected.head(5))
    with tempfile.TemporaryDirectory() as directory:
        bundle_path = Path(directory) / "state_bundle.json"
        bundle_path.write_text(
            json.dumps(dict(artifact.evidence.state_bundle)), encoding="utf-8"
        )
        model_info = mlflow.pyfunc.log_model(
            name=f"{evaluation.candidate_id}-final-model",
            python_model=str(Path(pyfunc_model.__file__).resolve()),
            artifacts={"state_bundle": str(bundle_path)},
            code_paths=[str(_PACKAGE_ROOT)],
            signature=signature,
            input_example=input_example,
        )
    loaded = mlflow.pyfunc.load_model(model_info.model_uri)
    reloaded = loaded.predict(evaluation_result.holdout_input_df).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        reloaded[expected.columns], expected, check_exact=True, check_dtype=False
    )
    mlflow.set_tag("final_model_uri", model_info.model_uri)


# --- Orchestrator -------------------------------------------------------------


def run_final_evaluation(
    outer_train_df: pd.DataFrame,
    modeling_table: pd.DataFrame,
    *,
    split_manifest: SplitManifest,
    cv_result: CrossValidationExperimentResult,
    pretest_freeze: SelectionFreeze,
    candidates: Sequence[CandidateDefinition],
    final_refit_factories: Mapping[str, Callable[[], BaseAgeGroupModel]],
    context: TrackingContext,
    source_cv_run_id: str,
    evaluation_config: EvaluationConfig | None,
    schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
    settings: TrackingSettings | None = None,
) -> FinalEvaluationResult:
    """Refit, replay the lockbox, evaluate, and log -- the one guarded backend call.

    This is the single function a caller (the marimo notebook) needs for
    Gate 8's expensive final action: ``experiment/`` stays MLflow-free and
    ``tracking/`` stays statistics-free, and no other function in this
    package may materialize the holdout partition.

    The full-training refit runs *before* any MLflow run opens, so a refit
    failure (including a Bayesian full-profile diagnostic error) never
    leaves a half-open final run behind. Everything from the parent run's
    opening onward -- logging the pretest freeze, flipping
    ``test_lock_status`` to ``opened``, replaying the manifest, evaluating,
    and logging -- happens inside one ``tracked_final_evaluation`` block, so
    a reactive rerun of the caller cannot route test evidence anywhere but
    through this guarded path.
    """
    # Before anything is refit or any run opens, the decision must be exactly
    # what the rule produces from this CV result, recorded on exactly this CV
    # result's freeze. A hand-built or edited decision never reaches the lockbox.
    verify_pretest_freeze(pretest_freeze, cv_result)
    refit_result = refit_frozen_approach_winners(
        outer_train_df,
        split_manifest=split_manifest,
        cv_result=cv_result,
        candidates=candidates,
        final_refit_factories=final_refit_factories,
        schema=schema,
    )
    # Computed before any run opens, from what was actually refit: a retry after
    # the lockbox opened must refit and score exactly the same attempt.
    attempt_fingerprint = final_attempt_fingerprint(
        pretest_freeze,
        refit_result,
        candidates=candidates,
        evaluation_config=evaluation_config,
        schema=schema,
    )
    cross_family = pretest_freeze.cross_family_selection
    with tracked_final_evaluation(
        context,
        source_cv_run_id=source_cv_run_id,
        manifest_fingerprint=pretest_freeze.manifest_fingerprint,
        cross_family_rule_version=cross_family.rule.version,
        selected_candidate_id=cross_family.selected_candidate_id,
        selected_approach=cross_family.selected_approach,
        pretest_freeze=pretest_freeze.to_dict(),
        final_attempt_fingerprint=attempt_fingerprint,
        settings=settings,
    ) as handle:
        evaluation_result = evaluate_frozen_models_on_lockbox(
            modeling_table,
            split_manifest=split_manifest,
            pretest_freeze=pretest_freeze,
            refit_result=refit_result,
            candidates=candidates,
            evaluation_config=evaluation_config,
            schema=schema,
        )
        log_final_evaluation_result(
            refit_result, evaluation_result, handle=handle, split_manifest=split_manifest
        )
    return evaluation_result
