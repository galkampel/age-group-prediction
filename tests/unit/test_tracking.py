"""MLflow tracking adapter against an isolated SQLite store (Gate 7).

The spy experiment in ``bundle_spy`` declares Poisson, NB2 and Normal
direct-cohort candidates (Normal as a diagnostic comparator) plus Model B and
Bayesian stand-ins, so every run shape the adapter writes is exercised in
seconds. Real models are covered by
``tests/validation/test_tracking_real_models.py``.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Set before MLflow is imported: the hint prints at import time.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
mlflow = pytest.importorskip("mlflow")

from age_group_prediction import tracking
from age_group_prediction.tracking import (
    ARTIFACT_LOCATION_ENV,
    DEFAULT_EXPERIMENT_NAME,
    TrackingContext,
    _files,
    candidates,
    log_cross_validation_experiment,
    log_experiment_result,
    resolve_tracking_settings,
    tracked_comparison,
)
from tests.unit import bundle_spy
from tests.unit.bundle_spy import (
    COHORTS,
    NEGATIVE_ESS,
    BundleSpyModel,
    run_spy_experiment,
)

# `tracking` is a package, so its `__init__.py` sits one level below the root.
PACKAGE_ROOT = Path(tracking.__file__).resolve().parent.parent
_RESULT_FRAMES = (
    "fold_metrics_df",
    "aggregate_metrics_df",
    "bootstrap_intervals_df",
    "predictions_df",
    "calibration_df",
    "importance_df",
    "importance_summary_df",
    "fold_coverage_df",
)
_CONTEXT = TrackingContext(
    source_revision="abc1234",
    source_dirty=True,
    run_name="spy-comparison",
    extra_tags={"purpose": "unit-test"},
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
    """Every run in every experiment of the store."""
    experiments = [item.experiment_id for item in client.search_experiments()]
    return client.search_runs(experiments) if experiments else []


def _download_text(client, run_id: str, path: str, destination: Path) -> str:
    """Download one text artifact and return its contents."""
    local = client.download_artifacts(run_id, path, str(destination))
    return Path(local).read_text(encoding="utf-8")


def _artifact_paths(client, run_id: str, path: str | None = None) -> set[str]:
    """Every artifact file path under ``path``, recursively."""
    paths: set[str] = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            paths |= _artifact_paths(client, run_id, item.path)
        else:
            paths.add(item.path)
    return paths


@pytest.fixture(scope="module")
def spy_run():
    """The five-candidate spy comparison, with artifacts captured."""
    return run_spy_experiment(capture_artifacts=True)


@pytest.fixture(scope="module")
def logged(spy_run, tmp_path_factory):
    """The spy comparison logged once, for the read-only structure tests."""
    split, result = spy_run
    root = tmp_path_factory.mktemp("tracking_store")
    with pytest.MonkeyPatch.context() as patch:
        uri = _point_store(patch, root)
        with tracked_comparison(_CONTEXT) as handle:
            log_experiment_result(result, handle=handle, train_df=split.train_df)
        client = mlflow.MlflowClient(uri)
        runs = _all_runs(client)
        yield {
            "client": client,
            "handle": handle,
            "parent": client.get_run(handle.run_id),
            "children": {
                run.data.tags["candidate_id"]: run
                for run in runs
                if run.data.tags.get("mlflow.parentRunId") == handle.run_id
            },
            "runs": runs,
            "result": result,
            "split": split,
            "downloads": tmp_path_factory.mktemp("downloads"),
        }


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh store per test, for tests that write or fail."""
    uri = _point_store(monkeypatch, tmp_path)
    yield mlflow.MlflowClient(uri)
    while mlflow.active_run() is not None:
        mlflow.end_run()


# --- Run structure and tags ---------------------------------------------------


def test_one_parent_and_one_nested_child_per_candidate(logged) -> None:
    """One complete parent and one complete sibling run per candidate and family."""
    parent, children, runs = logged["parent"], logged["children"], logged["runs"]

    assert len(runs) == 6
    assert {run.info.experiment_id for run in runs} == {logged["handle"].experiment_id}
    assert logged["client"].get_experiment(parent.info.experiment_id).name == (
        DEFAULT_EXPERIMENT_NAME
    )
    assert "mlflow.parentRunId" not in parent.data.tags
    assert set(children) == {
        "direct-poisson",
        "direct-nb2",
        "direct-normal",
        "independent-nb2",
        "bayesian-reduced",
    }
    for run in runs:
        assert run.info.status == "FINISHED"
        assert run.data.tags["evidence_complete"] == "true"

    # Each loss family is its own sibling run, never one run choosing a family.
    direct = {
        run.data.tags["family"]: run.data.tags["objective_family"]
        for run in children.values()
        if run.data.tags["approach"] == "DirectCohortModel"
    }
    assert direct == {"poisson": "poisson", "nb2": "nb2", "normal": "regression"}


def test_every_run_carries_identity_and_context_tags(logged) -> None:
    """Data, seed, lock and source tags on every run; selection outcome on children."""
    provenance = logged["result"].provenance
    identity = {
        "manifest_fingerprint": provenance.manifest_fingerprint,
        "training_data_hash": provenance.training_data_hash,
        "training_schema_hash": provenance.training_schema_hash,
        "master_seed": str(provenance.master_seed),
        "master_seed_source": provenance.master_seed_source,
        "test_lock_status": "locked",
        "source_dirty": "true",
        "mlflow.source.git.commit": "abc1234",
        "purpose": "unit-test",
    }
    for run in logged["runs"]:
        tags = run.data.tags
        assert {key: tags.get(key) for key in identity} == identity
        # Only the caller-supplied revision; nothing inferred from HEAD.
        assert {key for key in tags if key.startswith("mlflow.source.git.")} == {
            "mlflow.source.git.commit"
        }

    assert logged["parent"].data.tags["run_role"] == "comparison"
    children = logged["children"]
    for candidate_id, run in children.items():
        tags = run.data.tags
        assert tags["run_role"] == "candidate"
        assert tags["candidate_id"] == candidate_id
        assert tags["model_class"] == "BundleSpyModel"
    assert children["independent-nb2"].data.tags["family"] == "nb2"
    assert children["direct-normal"].data.tags["selection_role"] == (
        "diagnostic_comparator"
    )
    assert "never ranked" in children["direct-normal"].data.tags["rejection_reason"]

    for selection in logged["result"].selections:
        for candidate_id, run in children.items():
            if run.data.tags["approach"] != selection.approach:
                continue
            is_selected = candidate_id == selection.selected_candidate_id
            assert run.data.tags["selected"] == ("true" if is_selected else "false")
            if candidate_id in selection.rejection_reasons:
                assert (
                    run.data.tags["rejection_reason"]
                    == (selection.rejection_reasons[candidate_id])
                )


def test_extra_tags_cannot_overwrite_tags_the_adapter_owns() -> None:
    """A caller cannot forge completeness, lock status, or source revision."""
    for key in ("evidence_complete", "test_lock_status", "mlflow.source.git.commit"):
        with pytest.raises(ValueError, match="reserved tags"):
            TrackingContext(extra_tags={key: "x"})


# --- Parent evidence ----------------------------------------------------------


def test_parent_logs_params_training_input_and_comparison_artifacts(logged) -> None:
    """The parent's params, dataset input, and artifacts match the result."""
    client, parent, result = logged["client"], logged["parent"], logged["result"]
    run_id = parent.info.run_id
    downloads = logged["downloads"] / "parent"

    assert _artifact_paths(client, run_id) == {
        "freeze.json",
        "provenance.json",
        "selections.json",
        "fold_identities.json",
        "metric_comparability.json",
        "aggregate_metrics.csv",
        "bootstrap_intervals.csv",
        "fold_coverage.csv",
        "importance_summary.csv",
    }
    params = parent.data.params
    assert params["candidate_count"] == "5"
    assert params["eligible_candidate_count"] == "4"
    assert params["fold_count"] == "2"
    assert params["run_settings.capture_artifacts"] == "true"
    assert params["run_settings.evaluation_config.bootstrap_replicates"] == "3"
    assert params["split.holdout_building_count"] == str(
        len(logged["split"].manifest.holdout_building_ids)
    )
    assert params["split.source_table_hash"] == (
        logged["split"].manifest.source_table_hash
    )
    assert params["package.numpy"] == np.__version__
    assert [item.dataset.name for item in parent.inputs.dataset_inputs] == [
        "outer_training"
    ]

    def load(path: str):
        """Download and parse one of the parent's JSON artifacts."""
        return json.loads(_download_text(client, run_id, path, downloads))

    assert load("freeze.json") == json.loads(json.dumps(result.freeze.to_dict()))
    assert load("freeze.json")["test_metrics"] is None
    assert load("provenance.json") == json.loads(
        json.dumps(result.provenance.to_dict())
    )
    folds = load("fold_identities.json")
    assert set(folds) == {"fold_0", "fold_1"}
    assert set(folds["fold_0"]) == {
        "fold_index",
        "fit_building_ids",
        "validation_building_ids",
        "fingerprint",
    }
    aggregate = pd.read_csv(
        client.download_artifacts(run_id, "aggregate_metrics.csv", str(downloads))
    )
    assert list(aggregate.columns) == list(result.aggregate_metrics_df.columns)
    assert len(aggregate) == len(result.aggregate_metrics_df)


def test_metric_comparability_is_derived_from_the_selection_rules(logged) -> None:
    """Densities and masses are classified per candidate, with the policies' rules."""
    client, run_id = logged["client"], logged["parent"].info.run_id
    record = json.loads(
        _download_text(
            client, run_id, "metric_comparability.json", logged["downloads"] / "mc"
        )
    )

    assert record["rules"]["discrete_families"] == ["nb2", "poisson"]
    assert record["rules"]["continuous_families"] == ["normal"]
    assert record["rules"]["diagnostic_comparators_are_ranked"] is False
    assert record["rules"]["predictive_nll_is_ranked_across_approaches"] is False
    assert "predictive_measure_kinds_basis" in record["rules"]
    # The spy declares no pointwise log probabilities; the real-model test
    # checks the conditional models' sequential-joint scope.
    assert all(
        item["pointwise_log_probability_scopes"] == [] for item in record["candidates"]
    )
    kinds = {
        item["candidate_id"]: item["predictive_measure_kinds"]
        for item in record["candidates"]
    }
    assert kinds["direct-poisson"] == ["discrete"]
    assert kinds["direct-nb2"] == ["discrete"]
    assert kinds["direct-normal"] == ["continuous"]
    assert kinds["bayesian-reduced"] == []
    nll = next(
        metric
        for item in record["candidates"]
        if item["candidate_id"] == "direct-normal"
        for metric in item["metrics"]
        if metric["name"] == "predictive_nll"
    )
    assert nll["interpretation"] == "normal total"
    assert set(record["policies"]) == {
        "DirectCohortModel",
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    }


# --- Child evidence -----------------------------------------------------------


def test_candidate_artifacts_hold_bundles_checks_tuning_and_slices(logged) -> None:
    """A child's artifact tree, file schemas, and a reloadable stored bundle."""
    client, result = logged["client"], logged["result"]
    run_id = logged["children"]["direct-nb2"].info.run_id
    downloads = logged["downloads"] / "direct-nb2"

    per_fold = {
        f"folds/fold_{fold}/{name}"
        for fold in (0, 1)
        for name in (
            "metadata.json",
            "seeds.json",
            "state_bundle.json.gz",
            "reload_check.json",
        )
    }
    trials = {
        f"tuning/fold_{fold}/{cohort}_trials.csv"
        for fold in (0, 1)
        for cohort in COHORTS
    }
    assert _artifact_paths(client, run_id) == per_fold | trials | {
        "search_space.json",
        "predictions.csv",
        "fold_metrics.csv",
        "calibration.csv",
        "importance.csv",
    }

    evidence = next(
        run
        for run in result.fold_runs
        if run.candidate_id == "direct-nb2" and run.fold_identity.fold_index == 1
    )
    metadata = json.loads(
        _download_text(client, run_id, "folds/fold_1/metadata.json", downloads)
    )
    assert metadata == json.loads(json.dumps(dict(evidence.model_metadata)))
    seeds = json.loads(
        _download_text(client, run_id, "folds/fold_1/seeds.json", downloads)
    )
    assert seeds == dict(evidence.seeds)
    reload_check = json.loads(
        _download_text(client, run_id, "folds/fold_1/reload_check.json", downloads)
    )
    assert reload_check["passed"] is True
    assert set(reload_check["max_abs_differences"].values()) == {0.0}

    bundle_path = client.download_artifacts(
        run_id, "folds/fold_1/state_bundle.json.gz", str(downloads)
    )
    with gzip.open(bundle_path, "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)
    validation = (
        logged["split"]
        .train_df.set_index("building_id")
        .loc[list(evidence.fold_identity.validation_building_ids)]
    )
    reloaded = BundleSpyModel.from_state_bundle(bundle).predict(
        validation.reset_index()
    )
    np.testing.assert_array_equal(reloaded.total_mean, evidence.prediction.total_mean)

    table = pd.read_csv(
        client.download_artifacts(
            run_id, "tuning/fold_0/n_kindergarten_trials.csv", str(downloads)
        )
    )
    assert list(table.columns) == [
        "number",
        "state",
        "value",
        "param_learning_rate",
        "intermediate_values",
    ]
    assert list(table["state"]) == ["COMPLETE", "COMPLETE", "PRUNED"]
    pruned = next(
        trial
        for trial in evidence.model_metadata["model"]["diagnostics"]["tuning"][
            "n_kindergarten"
        ]["trials"]
        if trial["state"] == "PRUNED"
    )
    assert json.loads(table.loc[2, "intermediate_values"]) == [
        [step, pytest.approx(value)] for step, value in pruned["intermediate_values"]
    ]

    predictions = pd.read_csv(
        client.download_artifacts(run_id, "predictions.csv", str(downloads))
    )
    assert set(predictions["candidate_id"]) == {"direct-nb2"}
    importance = pd.read_csv(
        client.download_artifacts(run_id, "importance.csv", str(downloads))
    )
    assert "validation_building_ids" not in importance.columns
    assert not importance.empty


def test_each_family_run_logs_its_own_tuning_evidence(logged) -> None:
    """Tuning metrics and trial tables come from that candidate's own studies.

    Each spy family has a distinct best value, so evidence logged into the
    wrong family's run would not match that run's own metadata.
    """
    client, result = logged["client"], logged["result"]
    best_values: dict[str, set[float]] = {}
    for candidate_id in ("direct-poisson", "direct-nb2", "direct-normal"):
        run_id = logged["children"][candidate_id].info.run_id
        for evidence in result.fold_runs:
            if evidence.candidate_id != candidate_id:
                continue
            fold_index = evidence.fold_identity.fold_index
            studies = evidence.model_metadata["model"]["diagnostics"]["tuning"]
            for cohort, study in studies.items():
                history = {
                    metric.step: metric.value
                    for metric in client.get_metric_history(
                        run_id, f"tuning/{cohort}/best_value"
                    )
                }
                assert history[fold_index] == pytest.approx(study["best_value"])
                table = pd.read_csv(
                    client.download_artifacts(
                        run_id,
                        f"tuning/fold_{fold_index}/{cohort}_trials.csv",
                        str(logged["downloads"] / "tuning" / candidate_id),
                    )
                )
                assert table["value"].tolist() == pytest.approx(
                    [trial["value"] for trial in study["trials"]]
                )
                best_values.setdefault(candidate_id, set()).add(study["best_value"])
    assert len({frozenset(values) for values in best_values.values()}) == 3


def test_importance_building_ids_are_logged_once_per_fold(logged) -> None:
    """The dropped importance column must be recoverable from fold identities."""
    client, result = logged["client"], logged["result"]
    folds = json.loads(
        _download_text(
            client,
            logged["parent"].info.run_id,
            "fold_identities.json",
            logged["downloads"] / "ids",
        )
    )
    # Same IDs; the freeze orders them canonically and importance keeps the
    # validation frame's order, so they are compared as sets.
    for _, row in result.importance_df.iterrows():
        logged_ids = folds[f"fold_{row['fold_index']}"]["validation_building_ids"]
        importance_ids = json.loads(row["validation_building_ids"])
        assert len(importance_ids) == len(logged_ids)
        assert set(importance_ids) == set(logged_ids)


def test_candidates_without_tuning_or_importance_omit_those_artifacts(logged) -> None:
    """Artifacts exist only where the evidence does."""
    client = logged["client"]
    bayesian = _artifact_paths(
        client, logged["children"]["bayesian-reduced"].info.run_id
    )
    assert not any(path.startswith("tuning/") for path in bayesian)
    assert "search_space.json" not in bayesian

    independent = _artifact_paths(
        client, logged["children"]["independent-nb2"].info.run_id
    )
    assert {
        "tuning/fold_0/total_trials.csv",
        "tuning/fold_0/probability_trials.csv",
    } <= independent
    # Importance runs for selected candidates only.
    comparator = _artifact_paths(
        client, logged["children"]["direct-normal"].info.run_id
    )
    assert "importance.csv" not in comparator


def test_metrics_are_keyed_by_metric_target_level_and_stepped_by_fold(logged) -> None:
    """Fold metrics use the fold index as step and match the result's tables."""
    client, result = logged["client"], logged["result"]
    run = logged["children"]["direct-nb2"]
    run_id = run.info.run_id

    history = client.get_metric_history(run_id, "fold/mae/n_children_total/building")
    expected = result.fold_metrics_df[
        (result.fold_metrics_df["candidate_id"] == "direct-nb2")
        & (result.fold_metrics_df["metric_name"] == "mae")
    ].set_index("fold_index")["value"]
    assert {metric.step: metric.value for metric in history} == pytest.approx(
        expected.to_dict()
    )
    aggregate = result.aggregate_metrics_df.set_index(["candidate_id", "metric_name"])
    assert run.data.metrics["cv_mean/mae/n_children_total/building"] == pytest.approx(
        aggregate.loc[("direct-nb2", "mae"), "mean"]
    )
    assert "cv_std/predictive_nll/n_children_total/building" in run.data.metrics
    for key in ("duration/fit_seconds", "tuning/n_kindergarten/best_value"):
        assert sorted(m.step for m in client.get_metric_history(run_id, key)) == [0, 1]
    assert not any(key.startswith("diag/") for key in run.data.metrics)


def test_negative_ess_is_charted_as_zero_and_flagged_invalid(logged) -> None:
    """A negative ESS charts as 0 with ess_valid=0; the raw value stays in metadata."""
    run = logged["children"]["bayesian-reduced"]
    metrics = run.data.metrics

    assert metrics["diag/total/min_ess_clamped"] == 0.0
    assert metrics["diag/total/ess_valid"] == 0.0
    assert metrics["diag/total/policy_passed"] == 0.0
    assert metrics["diag/total/max_rhat"] == pytest.approx(1.2)
    assert metrics["diag/composition/min_ess_clamped"] == 400.0
    assert metrics["diag/composition/ess_valid"] == 1.0
    assert metrics["diag/composition/policy_passed"] == 1.0

    # The raw value is kept, but only in the fold's metadata artifact.
    metadata = json.loads(
        _download_text(
            logged["client"],
            run.info.run_id,
            "folds/fold_0/metadata.json",
            logged["downloads"] / "bayesian",
        )
    )
    assert metadata["model"]["diagnostics"]["total"][
        "minimum_effective_sample_size"
    ] == pytest.approx(NEGATIVE_ESS)


def test_logging_leaves_every_result_field_unchanged(spy_run, store) -> None:
    """Tracking only reads the result: a deep snapshot is equal after logging."""
    split, result = spy_run
    before = _snapshot(result)

    log_cross_validation_experiment(result, _CONTEXT, train_df=split.train_df)

    after = _snapshot(result)
    for name in _RESULT_FRAMES:
        pd.testing.assert_frame_equal(before["frames"][name], after["frames"][name])
    assert before["records"] == after["records"]
    for left, right in zip(before["arrays"], after["arrays"], strict=True):
        np.testing.assert_array_equal(left, right)


def _snapshot(result) -> dict[str, object]:
    """A deep, independent copy of everything a logger could reach."""
    return {
        "frames": {
            name: getattr(result, name).copy(deep=True) for name in _RESULT_FRAMES
        },
        "records": json.dumps(
            {
                "freeze": result.freeze.to_dict(),
                "provenance": result.provenance.to_dict(),
                "fold_runs": [
                    {
                        "candidate_id": run.candidate_id,
                        "fold_index": run.fold_identity.fold_index,
                        "metadata": dict(run.model_metadata),
                        "seeds": dict(run.seeds),
                        "metrics": run.evaluation.metrics_df.to_dict("records"),
                    }
                    for run in result.fold_runs
                ],
                "artifacts": [
                    {
                        "candidate_id": item.candidate_id,
                        "fold_index": item.fold_index,
                        "bundle": dict(item.state_bundle),
                        "check": item.reload_check.to_dict(),
                    }
                    for item in result.artifacts
                ],
            },
            sort_keys=True,
            default=str,
        ),
        "arrays": [
            np.array(array, copy=True)
            for run in result.fold_runs
            for array in (
                run.prediction.total_mean,
                run.prediction.cohort_means,
                run.prediction.age_group_probabilities,
            )
        ],
    }


# --- Settings -----------------------------------------------------------------


def test_settings_default_to_a_local_sqlite_store_with_artifacts_beside_it() -> None:
    """Documented defaults, and artifact placement for other tracking URIs."""
    defaults = resolve_tracking_settings({})
    assert defaults.tracking_uri == "sqlite:///mlflow.db"
    assert defaults.experiment_name == "age-group-prediction"
    assert defaults.artifact_location == (Path.cwd() / "mlartifacts").resolve().as_uri()

    absolute = resolve_tracking_settings(
        {"MLFLOW_TRACKING_URI": "sqlite:////data/runs/store.db?timeout=5"}
    )
    assert (
        absolute.artifact_location == Path("/data/runs/mlartifacts").resolve().as_uri()
    )
    for uri in ("http://localhost:5000", "sqlite:///:memory:"):
        assert (
            resolve_tracking_settings({"MLFLOW_TRACKING_URI": uri}).artifact_location
            is None
        )


def test_explicit_artifact_location_is_normalized(tmp_path) -> None:
    """A path and its file URI, spaces included, resolve to one location."""
    spaced = tmp_path / "with space"
    settings = resolve_tracking_settings(
        {
            "MLFLOW_EXPERIMENT_NAME": "named",
            ARTIFACT_LOCATION_ENV: spaced.as_uri(),
        }
    )
    assert settings.experiment_name == "named"
    assert settings.artifact_location == spaced.resolve().as_uri()
    # A plain path and its file URI name the same location.
    assert resolve_tracking_settings(
        {ARTIFACT_LOCATION_ENV: str(spaced)}
    ) == settings.__class__(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="age-group-prediction",
        artifact_location=settings.artifact_location,
    )


def test_existing_experiment_with_another_artifact_location_is_refused(
    spy_run, store, tmp_path
) -> None:
    """Artifacts never silently land somewhere other than configured."""
    store.create_experiment(
        DEFAULT_EXPERIMENT_NAME, artifact_location=str(tmp_path / "elsewhere")
    )

    with pytest.raises(ValueError, match="cannot move an experiment's artifacts"):
        log_cross_validation_experiment(spy_run[1], _CONTEXT)
    assert _all_runs(store) == []


def test_reusing_an_experiment_in_a_path_with_a_space_is_accepted(
    spy_run, monkeypatch, tmp_path
) -> None:
    """A second comparison into a store whose path has a space is accepted."""
    uri = _point_store(monkeypatch, tmp_path / "with space")
    first = log_cross_validation_experiment(spy_run[1], _CONTEXT)
    second = log_cross_validation_experiment(spy_run[1], _CONTEXT)

    client = mlflow.MlflowClient(uri)
    assert first != second
    assert {client.get_run(run_id).info.status for run_id in (first, second)} == {
        "FINISHED"
    }


def test_a_comparison_leaves_the_tracking_uri_as_the_environment_says(
    spy_run, monkeypatch, tmp_path
) -> None:
    """A later change to MLFLOW_TRACKING_URI must reach the next comparison.

    ``mlflow.set_tracking_uri`` also rewrites the environment variable, so a
    naive restore once pinned every later comparison to the first store.
    """
    first_uri = _point_store(monkeypatch, tmp_path / "first")
    log_cross_validation_experiment(spy_run[1], _CONTEXT)
    assert os.environ["MLFLOW_TRACKING_URI"] == first_uri
    assert mlflow.get_tracking_uri() == first_uri

    second_uri = _point_store(monkeypatch, tmp_path / "second")
    run_id = log_cross_validation_experiment(spy_run[1], _CONTEXT)
    assert mlflow.MlflowClient(second_uri).get_run(run_id).info.status == "FINISHED"
    assert _all_runs(mlflow.MlflowClient(first_uri)) != []
    assert len(_all_runs(mlflow.MlflowClient(second_uri))) == 6

    # With the variable unset, explicit settings must not leave it set behind.
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    explicit = resolve_tracking_settings(
        {
            "MLFLOW_TRACKING_URI": second_uri,
            ARTIFACT_LOCATION_ENV: str(tmp_path / "second" / "artifacts"),
        }
    )
    with tracked_comparison(_CONTEXT, settings=explicit):
        pass
    assert "MLFLOW_TRACKING_URI" not in os.environ


# --- Failures and refusals ------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status"), [(RuntimeError, "FAILED"), (KeyboardInterrupt, "KILLED")]
)
def test_a_comparison_that_fails_inside_the_block_is_recorded(
    store, monkeypatch, error, status
) -> None:
    """A runner failure or interrupt ends the parent FAILED or KILLED, incomplete."""

    def failing_fit(self, *, train_df, features, log_exposure, rng) -> None:
        """Fail every fit, as a broken model would."""
        raise error("simulated fit failure")

    monkeypatch.setattr(BundleSpyModel, "_fit_model", failing_fit)

    with (
        pytest.raises((RuntimeError, KeyboardInterrupt)),
        tracked_comparison(_CONTEXT) as handle,
    ):
        run_spy_experiment(capture_artifacts=True)

    parent = store.get_run(handle.run_id)
    assert parent.info.status == status
    assert parent.data.tags["evidence_complete"] == "false"
    assert parent.data.tags["failure_type"] in {"RuntimeError", "KeyboardInterrupt"}
    assert "simulated fit failure" in parent.data.tags["failure_message"]
    assert mlflow.active_run() is None


@pytest.mark.parametrize(
    ("error", "status"), [(RuntimeError, "FAILED"), (KeyboardInterrupt, "KILLED")]
)
def test_a_failure_midway_through_children_never_looks_complete(
    spy_run, store, monkeypatch, error, status
) -> None:
    """Children logged before a failure stay complete; nothing after reads complete."""
    original = candidates._write_candidate_artifacts
    written: list[str] = []

    def fail_on_third(result, candidate_id, runs, root):
        """Write the first two candidates' artifacts, then fail."""
        if len(written) == 2:
            raise error("simulated upload failure")
        written.append(candidate_id)
        original(result, candidate_id, runs, root)

    # Patched where `_log_candidate_run` looks the name up, or it never fires.
    monkeypatch.setattr(candidates, "_write_candidate_artifacts", fail_on_third)

    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        log_cross_validation_experiment(spy_run[1], _CONTEXT)

    runs = _all_runs(store)
    parent = next(run for run in runs if "mlflow.parentRunId" not in run.data.tags)
    children = {
        run.data.tags["candidate_id"]: run
        for run in runs
        if "mlflow.parentRunId" in run.data.tags
    }
    assert parent.info.status == status
    assert parent.data.tags["evidence_complete"] == "false"
    assert len(children) == 3
    for candidate_id, run in children.items():
        if candidate_id in written:
            assert run.info.status == "FINISHED"
            assert run.data.tags["evidence_complete"] == "true"
        else:
            assert run.info.status == status
            assert run.data.tags["evidence_complete"] == "false"
    assert mlflow.active_run() is None


def test_a_result_without_captured_artifacts_is_refused_before_any_run(store) -> None:
    """Incomplete evidence is refused before MLflow is touched."""
    _, result = run_spy_experiment(capture_artifacts=False)

    with pytest.raises(ValueError, match="capture_artifacts=True"):
        log_cross_validation_experiment(result, _CONTEXT)
    assert _all_runs(store) == []


def test_a_different_training_frame_is_refused_before_any_run(spy_run, store) -> None:
    """A training frame that does not hash to the result's is refused by name."""
    split, result = spy_run

    with pytest.raises(ValueError, match="training_data_hash"):
        log_cross_validation_experiment(
            result, _CONTEXT, train_df=split.train_df.iloc[1:]
        )
    assert _all_runs(store) == []


def test_an_already_active_run_is_refused(store) -> None:
    """A comparison must be a top-level run, never nested in someone else's."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.start_run()
    try:
        with (
            pytest.raises(RuntimeError, match="already active"),
            tracked_comparison(_CONTEXT),
        ):
            pass
    finally:
        mlflow.end_run()


def test_one_comparison_run_accepts_one_result(spy_run, store) -> None:
    """Logging twice into one parent is refused."""
    result = spy_run[1]

    with (
        pytest.raises(RuntimeError, match="already holds logged evidence"),
        tracked_comparison(_CONTEXT) as handle,
    ):
        log_experiment_result(result, handle=handle)
        log_experiment_result(result, handle=handle)
    parent = store.get_run(handle.run_id)
    assert parent.info.status == "FAILED"
    # The first call had marked it complete; the failure must overwrite that.
    assert parent.data.tags["evidence_complete"] == "false"


def test_an_interrupt_after_logging_finished_still_reads_incomplete(
    spy_run, store
) -> None:
    """Completeness is overwritten, not merely withheld, when the block fails late."""
    with (
        pytest.raises(KeyboardInterrupt),
        tracked_comparison(_CONTEXT) as handle,
    ):
        log_experiment_result(spy_run[1], handle=handle)
        assert mlflow.get_run(handle.run_id).data.tags["evidence_complete"] == "true"
        raise KeyboardInterrupt

    parent = store.get_run(handle.run_id)
    assert parent.info.status == "KILLED"
    assert parent.data.tags["evidence_complete"] == "false"


def test_a_candidate_whose_fitted_family_disagrees_is_refused(
    store, monkeypatch
) -> None:
    """A mislabeled candidate is refused rather than tagged with the wrong family."""
    original = bundle_spy.direct_diagnostics

    def mislabeled(family: str) -> dict:
        """Make the candidate declared as Poisson report NB2 as its fitted family."""
        return original("nb2" if family == "poisson" else family)

    monkeypatch.setattr(bundle_spy, "direct_diagnostics", mislabeled)
    _, result = run_spy_experiment(capture_artifacts=True)

    with pytest.raises(ValueError, match="declares family 'poisson'"):
        log_cross_validation_experiment(result, _CONTEXT)
    assert _all_runs(store) == []


def test_logging_outside_the_opening_block_is_refused(spy_run, store) -> None:
    """A handle is usable only while its parent run is active."""
    with tracked_comparison(_CONTEXT) as handle:
        pass

    with pytest.raises(RuntimeError, match="inside the tracked_comparison block"):
        log_experiment_result(spy_run[1], handle=handle)


# --- Serialization helpers ------------------------------------------------------


def test_params_are_flattened_without_collisions_or_lost_empties() -> None:
    """Nested params flatten to dotted names; empties survive; collisions fail."""
    assert _files._flatten("", {"a": {"b": 1, "c": [1, 2]}, "d": {}, "e": None}) == {
        "a.b": "1",
        "a.c": "[1, 2]",
        "d": "{}",
        "e": "null",
    }
    with pytest.raises(ValueError, match="flatten to the name 'a.b'"):
        _files._flatten("", {"a.b": 1, "a": {"b": 2}})


def test_overlong_params_are_truncated_deliberately_with_a_full_copy(
    store, tmp_path
) -> None:
    """Params over 6000 characters are cut, with the full values kept as JSON."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    long_value = "x" * 7000
    with mlflow.start_run() as run:
        _files._log_params({"short": "1", "long": long_value}, tmp_path)

    params = store.get_run(run.info.run_id).data.params
    assert params["short"] == "1"
    assert params["long"] == long_value[:6000]
    full = json.loads((tmp_path / "params_full.json").read_text(encoding="utf-8"))
    assert full == {
        "truncated_keys": ["long"],
        "params": {"short": "1", "long": long_value},
    }


# --- Boundary -------------------------------------------------------------------


def test_experiment_and_model_code_never_import_mlflow() -> None:
    """Tracking must stay an after-the-fact adapter (handoff decision 1)."""
    offenders: list[str] = []
    for directory in ("experiment", "models"):
        for path in sorted((PACKAGE_ROOT / directory).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                if any(name.split(".")[0] == "mlflow" for name in names):
                    offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert offenders == []

    package_init = ast.parse((PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(package_init)
        if isinstance(node, ast.ImportFrom)
    }
    assert "tracking" not in imported
