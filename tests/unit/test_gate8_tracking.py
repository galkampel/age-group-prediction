"""MLflow tracking for the Gate 8 final evaluation: final run lifecycle, evidence
logging, and the models-from-code pyfunc wrapper.

Run mechanics (lifecycle, duplicate refusal, evidence shape, one-time
semantics) reuse the same fast ``bundle_spy`` fixtures Phase 3's own tests
use, plus ``test_final_evaluation``'s spy refit/pretest-freeze helpers. Those
mechanics tests monkeypatch ``tracking.final._log_final_model`` to a cheap
stub, because ``BundleSpyModel`` bundles cannot dispatch through
``pyfunc_model``'s real-class registry (the model-from-code loader always
executes ``pyfunc_model.py`` fresh against the *real* three classes, never
against a monkeypatched copy of an already-imported module -- see the
module's own docstring). The pyfunc wrapper's dispatch/predict contract and
the actual ``mlflow.pyfunc.log_model``/``load_model`` round trip are instead
proven with one small, fast, real ``DirectCohortModel`` fit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
mlflow = pytest.importorskip("mlflow")

from age_group_prediction.experiment import (
    evaluate_frozen_models_on_lockbox,
    refit_frozen_approach_winners,
)
from age_group_prediction.experiment import final_evaluation as final_evaluation_module
from age_group_prediction.experiment.contracts import CandidateDefinition
from age_group_prediction.experiment.final_evaluation import (
    FinalCandidateEvaluation,
    FinalModelArtifact,
    FinalRefitEvidence,
    _final_prediction_records,
    _final_reload_check,
    _smoke_frame,
)
from age_group_prediction.hashing import column_schema_hash, table_hash
from age_group_prediction.metrics import MeanAbsoluteError
from age_group_prediction.modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_TREE_FEATURE_SPEC,
    DirectCohortConfig,
    OptunaTuningConfig,
)
from age_group_prediction.models.direct_cohort import DirectCohortModel
from age_group_prediction.tracking import (
    ARTIFACT_LOCATION_ENV,
    EVIDENCE_COMPLETE_TAG,
    TrackingContext,
    log_experiment_result,
    pyfunc_model,
    run_final_evaluation,
    tracked_comparison,
    tracked_final_evaluation,
)
from age_group_prediction.tracking import final as final_tracking
from tests.unit.bundle_spy import (
    BundleSpyModel,
    direct_diagnostics,
    modeling_table,
    run_spy_experiment,
    spy_candidates,
)
from tests.unit.test_final_evaluation import (
    _evaluation_config,
    _final_refit_factories,
    _stub_cross_family_selection,
)

TOTAL = DEFAULT_MODELING_SCHEMA.total_target_column
_CONTEXT = TrackingContext(
    source_revision="abc1234", source_dirty=False, run_name="gate8-final"
)


def _point_store(monkeypatch: pytest.MonkeyPatch, root: Path) -> str:
    """Point MLflow at a fresh SQLite store and artifact directory under ``root``."""
    uri = f"sqlite:///{root / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv(ARTIFACT_LOCATION_ENV, str(root / "artifacts"))
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    return uri


def _all_runs(client) -> list:
    experiments = [item.experiment_id for item in client.search_experiments()]
    return client.search_runs(experiments) if experiments else []


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh store per test."""
    uri = _point_store(monkeypatch, tmp_path)
    yield mlflow.MlflowClient(uri)
    while mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture(scope="module")
def spy_cv_result():
    """A fast, deterministic CV comparison result with captured fold artifacts.

    Captured artifacts are required both by ``refit_frozen_approach_winners``
    (Phase 3) and by ``log_experiment_result`` (Gate 7), which this module
    uses to build a real, loggable source CV parent run.
    """
    return run_spy_experiment(capture_artifacts=True)


@pytest.fixture
def refit_and_evaluation(spy_cv_result):
    """A full spy refit, pretest freeze, and lockbox evaluation, matched to one source."""
    split, result = spy_cv_result
    artifacts = refit_frozen_approach_winners(
        split.train_df,
        split_manifest=split.manifest,
        cv_result=result,
        candidates=spy_candidates(),
        final_refit_factories=_final_refit_factories(),
    )
    selection = _stub_cross_family_selection(
        manifest_fingerprint=result.freeze.manifest_fingerprint,
        selections=result.selections,
    )
    pretest_freeze = result.freeze.with_cross_family_selection(selection)
    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )
    return split, result, artifacts, pretest_freeze, evaluation_result


@pytest.fixture
def stub_log_final_model(monkeypatch):
    """Replace the real pyfunc logging with a cheap tag, for run-mechanics tests.

    ``BundleSpyModel`` bundles cannot dispatch through ``pyfunc_model``'s
    real-class registry (module docstring), so mechanics tests that are not
    themselves about the pyfunc contract stub this step out.
    """
    calls: list[str] = []

    def fake(evaluation, *, artifact, evaluation_result) -> None:
        del artifact, evaluation_result
        calls.append(evaluation.candidate_id)
        mlflow.set_tag("final_model_uri", f"stub://{evaluation.candidate_id}")

    monkeypatch.setattr(final_tracking, "_log_final_model", fake)
    return calls


def _source_cv_run(result, *, train_df=None) -> str:
    """Log a completed source CV comparison; return its parent run ID."""
    with tracked_comparison(_CONTEXT) as handle:
        log_experiment_result(result, handle=handle, train_df=train_df)
    return handle.run_id


def _open_final_run(
    pretest_freeze, *, source_cv_run_id: str, final_attempt_fingerprint: str = "attempt-a"
):
    """Open a final-evaluation run for ``pretest_freeze``, matching its manifest."""
    selection = pretest_freeze.cross_family_selection
    return tracked_final_evaluation(
        _CONTEXT,
        source_cv_run_id=source_cv_run_id,
        manifest_fingerprint=pretest_freeze.manifest_fingerprint,
        cross_family_rule_version=selection.rule.version,
        selected_candidate_id=selection.selected_candidate_id,
        selected_approach=selection.selected_approach,
        pretest_freeze=pretest_freeze.to_dict(),
        final_attempt_fingerprint=final_attempt_fingerprint,
    )


# --- Final run lifecycle ----------------------------------------------------


def test_final_run_is_separate_from_and_linked_to_its_source_cv_run(
    refit_and_evaluation, store
) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)

    with _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle:
        assert handle.run_id != source_run_id

    final_run = store.get_run(handle.run_id)
    assert final_run.data.tags["run_role"] == "final_evaluation"
    assert final_run.data.tags["source_cv_run_id"] == source_run_id
    assert final_run.data.tags["manifest_fingerprint"] == pretest_freeze.manifest_fingerprint
    # The source CV run is untouched: still the same tags it finished with.
    source_run = store.get_run(source_run_id)
    assert source_run.data.tags["run_role"] == "comparison"
    assert source_run.data.tags[EVIDENCE_COMPLETE_TAG] == "true"


def test_incomplete_source_cv_run_is_refused(refit_and_evaluation, store) -> None:
    _split, _result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    with tracked_comparison(_CONTEXT) as handle:
        # Never logged: the comparison run ends without evidence_complete=true.
        pass

    with (
        pytest.raises(ValueError, match="not a complete comparison"),
        _open_final_run(pretest_freeze, source_cv_run_id=handle.run_id),
    ):
        pass
    assert not any(
        run.data.tags.get("run_role") == "final_evaluation" for run in _all_runs(store)
    )


def test_unknown_source_cv_run_is_refused(refit_and_evaluation, store) -> None:
    _split, _result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    with (
        pytest.raises(mlflow.MlflowException),
        _open_final_run(pretest_freeze, source_cv_run_id="does-not-exist"),
    ):
        pass


def test_pretest_freeze_is_logged_before_the_lockbox_opens(
    refit_and_evaluation, store
) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)

    with _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle:
        opened = mlflow.get_run(handle.run_id)
        assert opened.data.tags["test_lock_status"] == "opened"
        artifact_paths = {
            item.path for item in store.list_artifacts(handle.run_id)
        }
        assert "pretest_freeze.json" in artifact_paths


def test_a_final_context_must_start_locked(refit_and_evaluation, store) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    unlocked = TrackingContext(test_lock_status="opened")

    with pytest.raises(ValueError, match="must start 'locked'"):
        selection = pretest_freeze.cross_family_selection
        with tracked_final_evaluation(
            unlocked,
            source_cv_run_id=source_run_id,
            manifest_fingerprint=pretest_freeze.manifest_fingerprint,
            cross_family_rule_version=selection.rule.version,
            selected_candidate_id=selection.selected_candidate_id,
            selected_approach=selection.selected_approach,
            pretest_freeze=pretest_freeze.to_dict(),
            final_attempt_fingerprint="attempt-a",
        ):
            pass
    assert not any(
        run.data.tags.get("run_role") == "final_evaluation" for run in _all_runs(store)
    )


def test_reserved_final_tags_cannot_be_forged(refit_and_evaluation) -> None:
    _split, _result, _artifacts, _pretest_freeze, _evaluation = refit_and_evaluation
    with pytest.raises(ValueError, match="reserved tags"):
        TrackingContext(extra_tags={"source_cv_run_id": "forged"})


def test_a_second_complete_final_run_is_refused(
    refit_and_evaluation, store, stub_log_final_model
) -> None:
    _split, result, artifacts, pretest_freeze, evaluation_result = refit_and_evaluation
    source_run_id = _source_cv_run(result)

    with _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle:
        final_tracking.log_final_evaluation_result(
            artifacts, evaluation_result, handle=handle, split_manifest=_split.manifest
        )

    with (
        pytest.raises(RuntimeError, match="already exists"),
        _open_final_run(pretest_freeze, source_cv_run_id=source_run_id),
    ):
        pass


# --- Evidence logging ---------------------------------------------------------


@pytest.fixture
def logged_final(refit_and_evaluation, store, stub_log_final_model):
    """One logged final run, for the read-only evidence-shape tests."""
    split, result, artifacts, pretest_freeze, evaluation_result = refit_and_evaluation
    source_run_id = _source_cv_run(result, train_df=split.train_df)
    with _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle:
        final_tracking.log_final_evaluation_result(
            artifacts, evaluation_result, handle=handle, split_manifest=split.manifest
        )
    runs = _all_runs(store)
    return {
        "client": store,
        "handle": handle,
        "parent": store.get_run(handle.run_id),
        "children": {
            run.data.tags["candidate_id"]: run
            for run in runs
            if run.data.tags.get("mlflow.parentRunId") == handle.run_id
        },
        "evaluation_result": evaluation_result,
        "pretest_freeze": pretest_freeze,
    }


def test_one_child_per_approach_winner(logged_final) -> None:
    children = logged_final["children"]
    assert len(children) == 3
    assert {run.data.tags["approach"] for run in children.values()} == {
        "DirectCohortModel",
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    }
    for run in children.values():
        assert run.info.status == "FINISHED"
        assert run.data.tags[EVIDENCE_COMPLETE_TAG] == "true"


def test_selected_versus_comparator_roles_are_tagged(logged_final) -> None:
    children = logged_final["children"]
    pretest_freeze = logged_final["pretest_freeze"]
    selected_id = pretest_freeze.cross_family_selection.selected_candidate_id
    roles = {candidate_id: run.data.tags["role"] for candidate_id, run in children.items()}
    assert roles[selected_id] == "selected"
    assert sum(role == "comparator" for role in roles.values()) == 2


def test_parent_holds_split_manifest_and_finalized_freeze_artifacts(logged_final) -> None:
    client, handle = logged_final["client"], logged_final["handle"]
    paths = {item.path for item in client.list_artifacts(handle.run_id)}
    assert {
        "split_manifest.json",
        "finalized_freeze.json",
        "cross_family_rule.json",
        "comparability.json",
        "provenance.json",
        "test_metrics.csv",
    } <= paths
    assert client.get_run(handle.run_id).data.tags[EVIDENCE_COMPLETE_TAG] == "true"


def test_child_artifacts_hold_bundle_metadata_and_result_slices(logged_final) -> None:
    client = logged_final["client"]
    evaluation_result = logged_final["evaluation_result"]
    candidate_id = evaluation_result.evaluations[0].candidate_id
    run_id = logged_final["children"][candidate_id].info.run_id
    paths = {item.path for item in client.list_artifacts(run_id)}
    assert {
        "refit_metadata.json",
        "seeds.json",
        "state_bundle.json.gz",
        "reload_check.json",
        "evaluation_metadata.json",
        "predictions.csv",
        "metrics.csv",
    } <= paths


def test_metrics_are_keyed_by_metric_target_and_level(logged_final) -> None:
    evaluation_result = logged_final["evaluation_result"]
    evaluation = evaluation_result.evaluations[0]
    run_id = logged_final["children"][evaluation.candidate_id].info.run_id
    metrics = logged_final["client"].get_run(run_id).data.metrics
    for _, row in evaluation.metrics_df.iterrows():
        key = f"test/{row['metric_name']}/{row['target']}/{row['aggregation_level']}"
        assert metrics[key] == pytest.approx(float(row["value"]))


def test_logging_twice_into_one_final_run_is_refused(
    refit_and_evaluation, store, stub_log_final_model
) -> None:
    split, result, artifacts, pretest_freeze, evaluation_result = refit_and_evaluation
    source_run_id = _source_cv_run(result)

    with (
        pytest.raises(RuntimeError, match="already holds logged evidence"),
        _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle,
    ):
        final_tracking.log_final_evaluation_result(
            artifacts, evaluation_result, handle=handle, split_manifest=split.manifest
        )
        final_tracking.log_final_evaluation_result(
            artifacts, evaluation_result, handle=handle, split_manifest=split.manifest
        )
    parent = store.get_run(handle.run_id)
    assert parent.info.status == "FAILED"
    assert parent.data.tags[EVIDENCE_COMPLETE_TAG] == "false"


def test_a_manifest_mismatch_is_refused_before_any_child_logs(
    refit_and_evaluation, store, stub_log_final_model
) -> None:
    split, result, artifacts, pretest_freeze, evaluation_result = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    other_selection = pretest_freeze.cross_family_selection
    with (
        pytest.raises(ValueError, match="does not match this final run's manifest"),
        tracked_final_evaluation(
            _CONTEXT,
            source_cv_run_id=source_run_id,
            manifest_fingerprint="not-the-real-fingerprint",
            cross_family_rule_version=other_selection.rule.version,
            selected_candidate_id=other_selection.selected_candidate_id,
            selected_approach=other_selection.selected_approach,
            pretest_freeze=pretest_freeze.to_dict(),
            final_attempt_fingerprint="attempt-a",
        ) as handle,
    ):
        final_tracking.log_final_evaluation_result(
            artifacts, evaluation_result, handle=handle, split_manifest=split.manifest
        )


def test_a_failure_midway_through_children_never_looks_complete(
    refit_and_evaluation, store, monkeypatch
) -> None:
    split, result, artifacts, pretest_freeze, evaluation_result = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    logged: list[str] = []

    def fail_on_second(evaluation, *, artifact, evaluation_result) -> None:
        del artifact, evaluation_result
        if len(logged) == 1:
            raise RuntimeError("simulated model-logging failure")
        logged.append(evaluation.candidate_id)
        mlflow.set_tag("final_model_uri", f"stub://{evaluation.candidate_id}")

    monkeypatch.setattr(final_tracking, "_log_final_model", fail_on_second)

    with (
        pytest.raises(RuntimeError, match="simulated model-logging failure"),
        _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle,
    ):
        final_tracking.log_final_evaluation_result(
            artifacts, evaluation_result, handle=handle, split_manifest=split.manifest
        )

    runs = _all_runs(store)
    parent = next(run for run in runs if run.info.run_id == handle.run_id)
    children = {
        run.data.tags.get("candidate_id"): run
        for run in runs
        if run.data.tags.get("mlflow.parentRunId") == handle.run_id
    }
    assert parent.info.status == "FAILED"
    assert parent.data.tags[EVIDENCE_COMPLETE_TAG] == "false"
    assert len(children) == 2
    finished = [run for run in children.values() if run.info.status == "FINISHED"]
    failed = [run for run in children.values() if run.info.status == "FAILED"]
    assert len(finished) == 1
    assert len(failed) == 1
    assert finished[0].data.tags[EVIDENCE_COMPLETE_TAG] == "true"
    assert failed[0].data.tags[EVIDENCE_COMPLETE_TAG] == "false"


def test_source_cv_result_is_unchanged_by_final_tracking(
    refit_and_evaluation, store, stub_log_final_model
) -> None:
    split, result, artifacts, pretest_freeze, evaluation_result = refit_and_evaluation
    before = result.freeze.to_dict()
    source_run_id = _source_cv_run(result)
    with _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle:
        final_tracking.log_final_evaluation_result(
            artifacts, evaluation_result, handle=handle, split_manifest=split.manifest
        )
    assert result.freeze.to_dict() == before


# --- Target-free holdout input ------------------------------------------------


def test_holdout_input_frame_carries_no_target_column(refit_and_evaluation) -> None:
    _split, _result, _artifacts, _pretest_freeze, evaluation_result = refit_and_evaluation
    leaked = set(evaluation_result.holdout_input_df.columns) & set(
        DEFAULT_MODELING_SCHEMA.target_columns
    )
    assert leaked == set()
    assert "building_id" in evaluation_result.holdout_input_df.columns


# --- Models-from-code pyfunc wrapper (real model classes) --------------------


def _fast_direct_cohort_fixture():
    """A quickly fitted real ``DirectCohortModel`` and its full-training refit evidence."""
    train_df = modeling_table()
    config = DirectCohortConfig(tuning=OptunaTuningConfig(n_trials=2))
    model = DirectCohortModel(direct_cohort_config=config)
    model.fit(train_df, feature_spec=DEFAULT_TREE_FEATURE_SPEC, rng=np.random.default_rng(0))
    bundle = model.to_state_bundle()
    smoke_df = _smoke_frame(train_df, schema=DEFAULT_MODELING_SCHEMA)
    reload_check = _final_reload_check(model, bundle, smoke_df, seed=0)
    evidence = FinalRefitEvidence(
        candidate_id="direct-poisson-real",
        approach="DirectCohortModel",
        manifest_fingerprint="test-fingerprint",
        training_data_hash=table_hash(train_df, id_column="building_id"),
        training_schema_hash=column_schema_hash(train_df),
        seeds={"fit": 0, "reload_check": 0},
        model_metadata=model.get_metadata(),
        state_bundle=bundle,
        reload_check=reload_check,
        smoke_frame_building_ids=tuple(smoke_df["building_id"]),
    )
    artifact = FinalModelArtifact(
        candidate_id="direct-poisson-real",
        approach="DirectCohortModel",
        model=model,
        evidence=evidence,
    )
    candidate = CandidateDefinition(
        candidate_id="direct-poisson-real",
        approach="DirectCohortModel",
        model_factory=lambda: DirectCohortModel(direct_cohort_config=config),
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(MeanAbsoluteError(TOTAL),),
        configuration={"family": "poisson"},
    )
    holdout_df = train_df.drop(columns=list(DEFAULT_MODELING_SCHEMA.target_columns))
    prediction = model.predict(train_df)
    predictions_df = pd.DataFrame(_final_prediction_records(candidate, prediction))
    evaluation = FinalCandidateEvaluation(
        candidate_id="direct-poisson-real",
        approach="DirectCohortModel",
        role="selected",
        seeds={"predict": 0, "evaluate": 0},
        metrics_df=pd.DataFrame(
            [{"metric_name": "mae", "target": TOTAL, "aggregation_level": "building", "value": 0.0}]
        ),
        intervals_df=pd.DataFrame(),
        predictions_df=predictions_df,
        metric_comparability="point accuracy",
        model_metadata=model.get_metadata(),
        evaluation_metadata={},
    )
    return artifact, evaluation, holdout_df


def test_pyfunc_model_class_dispatches_and_predicts_from_a_bundle() -> None:
    """The wrapper's own ``load_context``/``predict`` reproduce the fitted model exactly."""
    artifact, _evaluation, holdout_df = _fast_direct_cohort_fixture()
    wrapper = pyfunc_model.AgeGroupPyfuncModel()

    import json
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as directory:
        bundle_path = Path(directory) / "state_bundle.json"
        bundle_path.write_text(
            json.dumps(dict(artifact.evidence.state_bundle)), encoding="utf-8"
        )
        context = SimpleNamespace(artifacts={"state_bundle": str(bundle_path)})
        wrapper.load_context(context)

    predicted = wrapper.predict(context, holdout_df)
    expected = artifact.model.predict(holdout_df)
    assert list(predicted["building_id"]) == list(expected.building_ids)
    np.testing.assert_array_equal(
        predicted["total_mean"].to_numpy(), expected.total_mean
    )


def test_unsupported_bundle_model_class_is_refused() -> None:
    artifact, _evaluation, _holdout_df = _fast_direct_cohort_fixture()
    bundle = dict(artifact.evidence.state_bundle)
    bundle["model_class"] = "NotARealModel"
    wrapper = pyfunc_model.AgeGroupPyfuncModel()

    import json
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as directory:
        bundle_path = Path(directory) / "state_bundle.json"
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        context = SimpleNamespace(artifacts={"state_bundle": str(bundle_path)})
        with pytest.raises(ValueError, match="Unsupported state bundle model_class"):
            wrapper.load_context(context)


def test_logged_pyfunc_reloads_and_reproduces_predictions_exactly(store) -> None:
    """The real ``mlflow.pyfunc.log_model``/``load_model`` round trip via ``_log_final_model``."""
    artifact, evaluation, holdout_df = _fast_direct_cohort_fixture()

    class _StubResult:
        holdout_input_df = holdout_df.reset_index(drop=True)

    with tracked_comparison(_CONTEXT) as handle:
        final_tracking._log_final_model(
            evaluation, artifact=artifact, evaluation_result=_StubResult()
        )
        run = mlflow.get_run(handle.run_id)
        assert run.data.tags["final_model_uri"].startswith("runs:/") or run.data.tags[
            "final_model_uri"
        ].startswith("models:/")


# --- Independent-validation remediation (F1, F2, F3, F4, M5, M14, M20) --------


@pytest.fixture
def rule_returns_stub(monkeypatch, spy_cv_result):
    """Make the cross-family rule return the stub decision.

    The spy CV result declares none of the rule's metrics, so without this the
    real rule would refuse it; the rule binding itself is proven separately in
    ``test_final_evaluation``.
    """
    _split, result = spy_cv_result
    selection = _stub_cross_family_selection(
        manifest_fingerprint=result.freeze.manifest_fingerprint,
        selections=result.selections,
    )
    monkeypatch.setattr(
        final_evaluation_module, "select_cross_family_winner", lambda cv_result: selection
    )
    return selection


def _run_final(split, result, pretest_freeze, *, source_cv_run_id: str, run_name: str):
    return run_final_evaluation(
        split.train_df,
        modeling_table(),
        split_manifest=split.manifest,
        cv_result=result,
        pretest_freeze=pretest_freeze,
        candidates=spy_candidates(),
        final_refit_factories=_final_refit_factories(),
        context=TrackingContext(run_name=run_name),
        source_cv_run_id=source_cv_run_id,
        evaluation_config=_evaluation_config(),
    )


def _final_parents(client) -> list:
    return [
        run for run in _all_runs(client) if run.data.tags.get("run_role") == "final_evaluation"
    ]


def test_run_final_evaluation_logs_the_decision_before_the_lockbox_opens(
    spy_cv_result, store, stub_log_final_model, rule_returns_stub, monkeypatch
) -> None:
    """Observe the order of events, not only the state they leave behind.

    The pretest freeze artifact, carrying the decision, must be logged before
    ``test_lock_status`` flips to ``opened``, which must happen before the
    manifest is replayed.
    """
    split, result = spy_cv_result
    pretest_freeze = result.freeze.with_cross_family_selection(rule_returns_stub)
    source_run_id = _source_cv_run(result)
    events: list[tuple[str, object]] = []
    real_log_artifacts, real_set_tag = mlflow.log_artifacts, mlflow.set_tag
    real_replay = final_evaluation_module.replay_split_manifest

    def log_artifacts(local_dir, *args, **kwargs):
        pretest_path = Path(local_dir) / "pretest_freeze.json"
        if pretest_path.exists():
            events.append(("pretest_freeze", json.loads(pretest_path.read_text())))
        return real_log_artifacts(local_dir, *args, **kwargs)

    def set_tag(key, value, *args, **kwargs):
        if key == "test_lock_status":
            events.append(("test_lock_status", value))
        return real_set_tag(key, value, *args, **kwargs)

    def replay(*args, **kwargs):
        events.append(("replay", None))
        return real_replay(*args, **kwargs)

    monkeypatch.setattr(mlflow, "log_artifacts", log_artifacts)
    monkeypatch.setattr(mlflow, "set_tag", set_tag)
    monkeypatch.setattr(final_evaluation_module, "replay_split_manifest", replay)

    _run_final(split, result, pretest_freeze, source_cv_run_id=source_run_id, run_name="order")

    assert [kind for kind, _ in events[:3]] == ["pretest_freeze", "test_lock_status", "replay"]
    assert events[1][1] == "opened"
    logged_pretest = events[0][1]
    assert logged_pretest["cross_family_selection"] is not None
    assert logged_pretest == json.loads(json.dumps(pretest_freeze.to_dict()))


def test_run_final_evaluation_refuses_a_decision_the_rule_did_not_produce(
    spy_cv_result, store, stub_log_final_model, rule_returns_stub
) -> None:
    split, result = spy_cv_result
    source_run_id = _source_cv_run(result)
    alternative_id = next(
        s.selected_candidate_id
        for s in result.selections
        if s.approach == "IndependentTotalProbabilityModel"
    )
    edited = result.freeze.with_cross_family_selection(
        replace(
            rule_returns_stub,
            selected_candidate_id=alternative_id,
            selected_approach="IndependentTotalProbabilityModel",
        )
    )

    with pytest.raises(ValueError, match="not the one the rule produces"):
        _run_final(split, result, edited, source_cv_run_id=source_run_id, run_name="edited")

    assert _final_parents(store) == []


def test_final_children_are_tagged_as_opened_test_evidence(logged_final) -> None:
    for run in logged_final["children"].values():
        assert run.data.tags["test_lock_status"] == "opened"


def _fail_after_opening(pretest_freeze, *, source_cv_run_id: str) -> None:
    with (
        pytest.raises(RuntimeError, match="simulated failure after opening"),
        _open_final_run(pretest_freeze, source_cv_run_id=source_cv_run_id),
    ):
        raise RuntimeError("simulated failure after opening")


def _alternative_pretest_freeze(result, pretest_freeze):
    alternative_id = next(
        s.selected_candidate_id
        for s in result.selections
        if s.approach == "IndependentTotalProbabilityModel"
    )
    return result.freeze.with_cross_family_selection(
        replace(
            pretest_freeze.cross_family_selection,
            selected_candidate_id=alternative_id,
            selected_approach="IndependentTotalProbabilityModel",
        )
    )


def test_a_different_decision_after_the_lockbox_opened_is_refused(
    refit_and_evaluation, store
) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    _fail_after_opening(pretest_freeze, source_cv_run_id=source_run_id)

    with (
        pytest.raises(RuntimeError, match="lockbox was already opened"),
        _open_final_run(
            _alternative_pretest_freeze(result, pretest_freeze), source_cv_run_id=source_run_id
        ),
    ):
        pass

    assert len(_final_parents(store)) == 1


def test_a_different_source_cv_run_after_the_lockbox_opened_is_refused(
    refit_and_evaluation, store
) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    first_source = _source_cv_run(result)
    _fail_after_opening(pretest_freeze, source_cv_run_id=first_source)
    second_source = _source_cv_run(result)

    with (
        pytest.raises(RuntimeError, match="lockbox was already opened"),
        _open_final_run(pretest_freeze, source_cv_run_id=second_source),
    ):
        pass


def test_an_identical_retry_after_the_lockbox_opened_is_allowed_and_linked(
    refit_and_evaluation, store
) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    _fail_after_opening(pretest_freeze, source_cv_run_id=source_run_id)
    (failed,) = _final_parents(store)
    assert failed.info.status == "FAILED"

    with _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle:
        retry = mlflow.get_run(handle.run_id)

    assert retry.data.tags["retry_of_run_ids"] == failed.info.run_id
    assert (
        retry.data.tags["cross_family_decision_hash"]
        == failed.data.tags["cross_family_decision_hash"]
    )


def test_a_pretest_freeze_without_a_decision_never_opens_a_final_run(
    refit_and_evaluation, store
) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    selection = pretest_freeze.cross_family_selection

    with (
        pytest.raises(ValueError, match="cross-family decision"),
        tracked_final_evaluation(
            _CONTEXT,
            source_cv_run_id=source_run_id,
            manifest_fingerprint=pretest_freeze.manifest_fingerprint,
            cross_family_rule_version=selection.rule.version,
            selected_candidate_id=selection.selected_candidate_id,
            selected_approach=selection.selected_approach,
            pretest_freeze=result.freeze.to_dict(),
            final_attempt_fingerprint="attempt-a",
        ),
    ):
        pass

    assert _final_parents(store) == []


def test_log_final_model_refuses_a_pyfunc_that_does_not_reproduce(store, monkeypatch) -> None:
    """The in-run reload check must fail when the loaded model's output drifts."""
    artifact, evaluation, holdout_df = _fast_direct_cohort_fixture()

    class _StubResult:
        holdout_input_df = holdout_df.reset_index(drop=True)

    real_load_model = mlflow.pyfunc.load_model

    class _Drifting:
        def __init__(self, loaded) -> None:
            self._loaded = loaded

        def predict(self, frame):
            output = self._loaded.predict(frame)
            output["total_mean"] = output["total_mean"] + 1e-9
            return output

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: _Drifting(real_load_model(uri)))

    with pytest.raises(AssertionError), tracked_comparison(_CONTEXT):
        final_tracking._log_final_model(
            evaluation, artifact=artifact, evaluation_result=_StubResult()
        )


_FRESH_PROCESS_LOADER = """
import json
import sys

import mlflow
from mlflow.models import Model

path = sys.argv[1]
model = mlflow.pyfunc.load_model(path)
example = Model.load(path).load_input_example(path)
prediction = model.predict(example)
print(json.dumps({
    "module_file": sys.modules["age_group_prediction"].__file__,
    "prediction": prediction.to_dict(orient="list"),
}))
"""


def test_logged_pyfunc_loads_and_predicts_in_a_fresh_process(store, tmp_path) -> None:
    """Load a logged model where nothing is pre-imported, as a consumer would.

    The in-process reload reuses the already-imported package and an
    already-initialized LightGBM, so it can prove neither that ``code_paths``
    is sufficient nor that the model loads at all in a new interpreter.
    """
    artifact, evaluation, holdout_df = _fast_direct_cohort_fixture()

    class _StubResult:
        holdout_input_df = holdout_df.reset_index(drop=True)

    with tracked_comparison(_CONTEXT) as handle:
        final_tracking._log_final_model(
            evaluation, artifact=artifact, evaluation_result=_StubResult()
        )
    model_uri = store.get_run(handle.run_id).data.tags["final_model_uri"]
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri, dst_path=str(tmp_path / "model")
    )
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_LOADER, local_path],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-3000:]
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(payload["module_file"]).resolve().is_relative_to(Path(local_path).resolve())
    expected = pyfunc_model.prediction_to_frame(
        artifact.model.predict(holdout_df.head(5).reset_index(drop=True))
    )
    assert payload["prediction"] == expected.to_dict(orient="list")


def test_a_deleted_complete_final_run_still_refuses_a_second_evaluation(
    refit_and_evaluation, store, stub_log_final_model
) -> None:
    split, result, artifacts, pretest_freeze, evaluation_result = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    with _open_final_run(pretest_freeze, source_cv_run_id=source_run_id) as handle:
        final_tracking.log_final_evaluation_result(
            artifacts, evaluation_result, handle=handle, split_manifest=split.manifest
        )
    store.delete_run(handle.run_id)

    with (
        pytest.raises(RuntimeError, match="already exists"),
        _open_final_run(pretest_freeze, source_cv_run_id=source_run_id),
    ):
        pass


# --- Independent remediation review (R1-R4) -----------------------------------


def _failed_opened_final_attempt(split, result, pretest_freeze, monkeypatch) -> str:
    """Run one guarded final attempt that fails after the lockbox opened.

    Returns the source CV run ID. The failure is injected into the logging
    step, which runs only after ``test_lock_status`` has flipped to ``opened``,
    and fires once, so a later attempt in the same test runs normally.
    """
    source_run_id = _source_cv_run(result)
    real_log = final_tracking.log_final_evaluation_result
    failures: list[str] = []

    def fail_once(*args, **kwargs):
        if not failures:
            failures.append("failed")
            raise RuntimeError("simulated failure after the lockbox opened")
        return real_log(*args, **kwargs)

    monkeypatch.setattr(final_tracking, "log_final_evaluation_result", fail_once)
    with pytest.raises(RuntimeError, match="simulated failure after the lockbox opened"):
        _run_final(split, result, pretest_freeze, source_cv_run_id=source_run_id, run_name="first")
    return source_run_id


def test_a_retry_that_refits_different_models_after_the_lockbox_opened_is_refused(
    spy_cv_result, store, stub_log_final_model, rule_returns_stub, monkeypatch
) -> None:
    """The same decision refit with other model settings is a new choice, not a retry."""
    split, result = spy_cv_result
    pretest_freeze = result.freeze.with_cross_family_selection(rule_returns_stub)
    source_run_id = _failed_opened_final_attempt(split, result, pretest_freeze, monkeypatch)
    retuned = dict(_final_refit_factories())
    retuned[rule_returns_stub.selected_candidate_id] = lambda: BundleSpyModel(
        offset=0.9, family="poisson", diagnostics=direct_diagnostics("poisson")
    )

    with pytest.raises(RuntimeError, match="lockbox was already opened"):
        run_final_evaluation(
            split.train_df,
            modeling_table(),
            split_manifest=split.manifest,
            cv_result=result,
            pretest_freeze=pretest_freeze,
            candidates=spy_candidates(),
            final_refit_factories=retuned,
            context=TrackingContext(run_name="retuned"),
            source_cv_run_id=source_run_id,
            evaluation_config=_evaluation_config(),
        )

    assert len(_final_parents(store)) == 1


def test_a_retry_with_a_different_master_seed_after_the_lockbox_opened_is_refused(
    spy_cv_result, store, stub_log_final_model, rule_returns_stub, monkeypatch
) -> None:
    """A consistently re-seeded CV result would re-roll every refit and test seed."""
    split, result = spy_cv_result
    pretest_freeze = result.freeze.with_cross_family_selection(rule_returns_stub)
    source_run_id = _failed_opened_final_attempt(split, result, pretest_freeze, monkeypatch)
    reseeded = replace(
        result,
        freeze=replace(result.freeze, master_seed=result.freeze.master_seed + 1),
        provenance=replace(result.provenance, master_seed=result.provenance.master_seed + 1),
    )

    with pytest.raises(RuntimeError, match="lockbox was already opened"):
        _run_final(
            split,
            reseeded,
            reseeded.freeze.with_cross_family_selection(rule_returns_stub),
            source_cv_run_id=source_run_id,
            run_name="reseeded",
        )

    assert len(_final_parents(store)) == 1


def test_an_identical_guarded_retry_after_the_lockbox_opened_is_allowed_and_linked(
    spy_cv_result, store, stub_log_final_model, rule_returns_stub, monkeypatch
) -> None:
    """The attempt fingerprint is stable, so a true retry is never refused."""
    split, result = spy_cv_result
    pretest_freeze = result.freeze.with_cross_family_selection(rule_returns_stub)
    source_run_id = _failed_opened_final_attempt(split, result, pretest_freeze, monkeypatch)
    (failed,) = _final_parents(store)

    _run_final(split, result, pretest_freeze, source_cv_run_id=source_run_id, run_name="retry")

    (retry,) = [run for run in _final_parents(store) if run.info.run_id != failed.info.run_id]
    assert failed.info.status == "FAILED"
    assert retry.data.tags[EVIDENCE_COMPLETE_TAG] == "true"
    assert retry.data.tags["retry_of_run_ids"] == failed.info.run_id
    assert (
        retry.data.tags["final_attempt_fingerprint"]
        == failed.data.tags["final_attempt_fingerprint"]
    )


def test_a_different_attempt_fingerprint_after_the_lockbox_opened_is_refused(
    refit_and_evaluation, store
) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    _fail_after_opening(pretest_freeze, source_cv_run_id=source_run_id)

    with (
        pytest.raises(RuntimeError, match="lockbox was already opened"),
        _open_final_run(
            pretest_freeze, source_cv_run_id=source_run_id, final_attempt_fingerprint="attempt-b"
        ),
    ):
        pass

    assert len(_final_parents(store)) == 1


def test_a_deleted_opened_attempt_still_blocks_a_different_decision(
    refit_and_evaluation, store
) -> None:
    """Deleting an attempt that opened the lockbox does not unsee its evidence."""
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    _fail_after_opening(pretest_freeze, source_cv_run_id=source_run_id)
    (failed,) = _final_parents(store)
    store.delete_run(failed.info.run_id)

    with (
        pytest.raises(RuntimeError, match="lockbox was already opened"),
        _open_final_run(
            _alternative_pretest_freeze(result, pretest_freeze), source_cv_run_id=source_run_id
        ),
    ):
        pass


def test_selected_tags_that_do_not_match_the_decision_never_open_a_final_run(
    refit_and_evaluation, store
) -> None:
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    selection = pretest_freeze.cross_family_selection
    assert selection.selected_approach != "IndependentTotalProbabilityModel"
    other_id = next(
        s.selected_candidate_id
        for s in result.selections
        if s.approach == "IndependentTotalProbabilityModel"
    )

    with (
        pytest.raises(ValueError, match="do not match the pretest freeze"),
        tracked_final_evaluation(
            _CONTEXT,
            source_cv_run_id=source_run_id,
            manifest_fingerprint=pretest_freeze.manifest_fingerprint,
            cross_family_rule_version=selection.rule.version,
            selected_candidate_id=other_id,
            selected_approach="IndependentTotalProbabilityModel",
            pretest_freeze=pretest_freeze.to_dict(),
            final_attempt_fingerprint="attempt-a",
        ),
    ):
        pass

    assert _final_parents(store) == []


def test_an_attempt_that_failed_before_opening_does_not_block_a_new_decision(
    refit_and_evaluation, store, monkeypatch
) -> None:
    """No test evidence exists until the lockbox opens, so nothing was seen to re-choose on."""
    _split, result, _artifacts, pretest_freeze, _evaluation = refit_and_evaluation
    source_run_id = _source_cv_run(result)
    real_log_artifacts = mlflow.log_artifacts

    def unavailable(*args, **kwargs):
        raise OSError("artifact store unavailable")

    monkeypatch.setattr(mlflow, "log_artifacts", unavailable)
    with pytest.raises(OSError), _open_final_run(pretest_freeze, source_cv_run_id=source_run_id):
        pass
    monkeypatch.setattr(mlflow, "log_artifacts", real_log_artifacts)
    (failed,) = _final_parents(store)
    assert failed.data.tags["test_lock_status"] == "locked"

    alternative = _alternative_pretest_freeze(result, pretest_freeze)
    with _open_final_run(alternative, source_cv_run_id=source_run_id) as handle:
        opened = mlflow.get_run(handle.run_id)

    selected = alternative.cross_family_selection.selected_candidate_id
    assert opened.data.tags["selected_candidate_id"] == selected
    assert "retry_of_run_ids" not in opened.data.tags
