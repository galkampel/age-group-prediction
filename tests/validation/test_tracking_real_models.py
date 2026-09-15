"""Gate 7 tracking against all three real model families.

The unit tests drive a spy model. This test logs a real comparison (direct
Poisson, NB2 and Normal as a diagnostic comparator, independent NB2, and the
reduced Bayesian profile) to a temporary MLflow store, then proves from what
MLflow stored, not from memory, that every fold's state bundle reloads without
the training frame and reproduces the runner's predictions exactly.
"""

from __future__ import annotations

import gzip
import json
import os
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
mlflow = pytest.importorskip("mlflow")

from age_group_prediction.data_splitting import (
    make_validation_folds,
    split_known_neighborhood_buildings,
)
from age_group_prediction.experiment import (
    CandidateDefinition,
    direct_cohort_selection_policy,
    run_cross_model_validation,
    select_cross_family_winner,
    sequential_joint_selection_policy,
)
from age_group_prediction.metrics import default_metric_set
from age_group_prediction.modeling_config import (
    DEFAULT_TREE_FEATURE_SPEC,
    BayesianConditionalConfig,
    DirectCohortConfig,
    FoldConfig,
    IndependentTotalProbabilityConfig,
    OuterSplitConfig,
    PredictionConfig,
)
from age_group_prediction.models import (
    BayesianConditionalModel,
    DirectCohortModel,
    IndependentTotalProbabilityModel,
)
from age_group_prediction.tracking import (
    ARTIFACT_LOCATION_ENV,
    TrackingContext,
    log_experiment_result,
    run_final_evaluation,
    tracked_comparison,
)
from tests.validation.test_experiment_real_models import (
    COHORTS,
    FAST_TUNING,
    _experiment_config,
    _importance_spec,
    _learnable_frame,
    _real_candidates,
)

_MODEL_CLASSES = {
    model.__name__: model
    for model in (
        DirectCohortModel,
        IndependentTotalProbabilityModel,
        BayesianConditionalModel,
    )
}


def _direct_candidate(config, family: str, role: str) -> CandidateDefinition:
    """A real direct-cohort candidate for one family, with fast tuning."""
    return CandidateDefinition(
        candidate_id=f"direct-{family}",
        approach="DirectCohortModel",
        model_factory=lambda: DirectCohortModel(
            direct_cohort_config=DirectCohortConfig(
                family=family, tuning=FAST_TUNING, bootstrap_replicates=4
            )
        ),
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=default_metric_set(
            COHORTS,
            nll_source="parametric",
            nll_interpretation=f"independent direct-cohort {family} likelihood",
            evaluation_config=config.evaluation,
            prediction_validation_config=config.prediction_validation,
        ),
        configuration={"family": family},
        prediction_config=PredictionConfig(),
        selection_role=role,
        importance_specs=(_importance_spec("tree"),),
    )


def _run_real_comparison(*, capture_artifacts: bool):
    """Run the five real candidates once; return the split, folds and result."""
    config = _experiment_config()
    split = split_known_neighborhood_buildings(
        _learnable_frame(),
        config=OuterSplitConfig(test_fraction=0.25),
        rng=np.random.default_rng(4),
    )
    folds = make_validation_folds(
        split.train_df, config=FoldConfig(n_folds=2), rng=np.random.default_rng(9)
    ).folds
    # The thirty-sample Bayesian profile is expected to miss its convergence
    # thresholds; see test_experiment_real_models.py.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Bayesian (total|composition) diagnostics failed",
            category=RuntimeWarning,
        )
        result = run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=(
                *_real_candidates(config),
                _direct_candidate(config, "nb2", "eligible"),
                _direct_candidate(config, "normal", "diagnostic_comparator"),
            ),
            selection_policies=(
                direct_cohort_selection_policy(COHORTS),
                sequential_joint_selection_policy("IndependentTotalProbabilityModel"),
                sequential_joint_selection_policy("BayesianConditionalModel"),
            ),
            master_seed=23,
            evaluation_config=replace(config.evaluation, bootstrap_replicates=6),
            require_convergence=False,
            capture_artifacts=capture_artifacts,
        )
    return split, folds, result


@pytest.fixture(scope="module")
def tracked_real_run(tmp_path_factory):
    """Run the five real candidates inside a tracked comparison and log them."""
    root = tmp_path_factory.mktemp("real_tracking_store")
    with pytest.MonkeyPatch.context() as patch:
        uri = f"sqlite:///{root / 'mlflow.db'}"
        patch.setenv("MLFLOW_TRACKING_URI", uri)
        patch.setenv(ARTIFACT_LOCATION_ENV, str(root / "artifacts"))
        patch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        with tracked_comparison(TrackingContext(run_name="real")) as handle:
            split, folds, result = _run_real_comparison(capture_artifacts=True)
            log_experiment_result(result, handle=handle, train_df=split.train_df)
        client = mlflow.MlflowClient(uri)
        children = {
            run.data.tags["candidate_id"]: run
            for run in client.search_runs([handle.experiment_id])
            if run.data.tags.get("mlflow.parentRunId") == handle.run_id
        }
        yield {
            "client": client,
            "parent": client.get_run(handle.run_id),
            "children": children,
            "result": result,
            "split": split,
            "folds": {fold.fold_index: fold for fold in folds},
            "downloads": tmp_path_factory.mktemp("real_downloads"),
        }


@pytest.mark.slow
def test_real_comparison_logs_one_complete_run_per_family(tracked_real_run) -> None:
    """Every real candidate is a complete sibling run with its real objective."""
    parent, children = tracked_real_run["parent"], tracked_real_run["children"]

    assert parent.info.status == "FINISHED"
    assert parent.data.tags["evidence_complete"] == "true"
    assert set(children) == {
        "direct-poisson",
        "direct-nb2",
        "direct-normal",
        "independent-nb2",
        "bayesian-reduced",
    }
    for run in children.values():
        assert run.info.status == "FINISHED"
        assert run.data.tags["evidence_complete"] == "true"
    objectives = {
        run.data.tags["family"]: run.data.tags["objective_family"]
        for run in children.values()
        if run.data.tags["approach"] == "DirectCohortModel"
    }
    assert objectives == {
        "poisson": "poisson",
        "nb2": "custom_nb2_gradient",
        "normal": "regression",
    }
    assert children["bayesian-reduced"].data.tags["model_class"] == (
        "BayesianConditionalModel"
    )


@pytest.mark.slow
def test_every_stored_bundle_reloads_to_the_runner_predictions(
    tracked_real_run,
) -> None:
    """Frequentist and Pyro artifacts pass reload/predict smoke tests from MLflow."""
    client, result = tracked_real_run["client"], tracked_real_run["result"]
    predictions = result.predictions_df.set_index(
        ["candidate_id", "fold_index", "building_id"]
    )
    cohort_columns = [f"{cohort}_mean" for cohort in COHORTS]
    probability_columns = [f"{cohort}_probability" for cohort in COHORTS]
    checked = 0
    for candidate_id, run in tracked_real_run["children"].items():
        for fold_index, fold in tracked_real_run["folds"].items():
            path = client.download_artifacts(
                run.info.run_id,
                f"folds/fold_{fold_index}/state_bundle.json.gz",
                str(tracked_real_run["downloads"] / candidate_id),
            )
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                bundle = json.load(stream)
            model = _MODEL_CLASSES[bundle["model_class"]].from_state_bundle(bundle)
            reloaded = model.predict(
                fold.validation_df,
                prediction_config=PredictionConfig(),
                rng=np.random.default_rng(0),
            )
            expected = predictions.loc[
                [
                    (candidate_id, fold_index, building)
                    for building in reloaded.building_ids
                ]
            ]
            np.testing.assert_array_equal(
                reloaded.total_mean, expected["total_mean"].to_numpy()
            )
            np.testing.assert_array_equal(
                reloaded.cohort_means, expected[cohort_columns].to_numpy()
            )
            np.testing.assert_array_equal(
                reloaded.age_group_probabilities,
                expected[probability_columns].to_numpy(),
            )
            checked += 1
    assert checked == 10


@pytest.mark.slow
def test_each_loss_family_records_its_own_train_only_tuning(tracked_real_run) -> None:
    """Each tuned run stores its own per-fold trial tables and best values."""
    client, result = tracked_real_run["client"], tracked_real_run["result"]
    direct_best_values: dict[str, list[float]] = {}
    components = {
        "direct-poisson": COHORTS,
        "direct-nb2": COHORTS,
        "direct-normal": COHORTS,
        "independent-nb2": ("total", "probability"),
    }
    for candidate_id, names in components.items():
        run_id = tracked_real_run["children"][candidate_id].info.run_id
        for fold_index in tracked_real_run["folds"]:
            evidence = next(
                item
                for item in result.fold_runs
                if item.candidate_id == candidate_id
                and item.fold_identity.fold_index == fold_index
            )
            diagnostics = evidence.model_metadata["model"]["diagnostics"]
            studies = diagnostics.get("tuning") or {
                "total": diagnostics["selection"]["total_tuning"],
                "probability": diagnostics["selection"]["probability_tuning"],
            }
            for name in names:
                table = pd.read_csv(
                    client.download_artifacts(
                        run_id,
                        f"tuning/fold_{fold_index}/{name}_trials.csv",
                        str(tracked_real_run["downloads"] / "tuning" / candidate_id),
                    )
                )
                trials = studies[name]["trials"]
                assert list(table["number"]) == [trial["number"] for trial in trials]
                # Scores, not only trial numbers: every study numbers its trials
                # 0 and 1, so only the values tell one family's studies apart.
                assert table["value"].tolist() == pytest.approx(
                    [
                        np.nan if trial["value"] is None else trial["value"]
                        for trial in trials
                    ],
                    nan_ok=True,
                )
                history = {
                    metric.step: metric.value
                    for metric in client.get_metric_history(
                        run_id, f"tuning/{name}/best_value"
                    )
                }
                assert history[fold_index] == pytest.approx(studies[name]["best_value"])
                if candidate_id.startswith("direct-"):
                    direct_best_values.setdefault(candidate_id, []).append(
                        studies[name]["best_value"]
                    )
    # Each loss family tuned its own studies, so their scores differ.
    assert len({tuple(values) for values in direct_best_values.values()}) == 3


@pytest.mark.slow
def test_real_bayesian_run_charts_convergence_without_negative_ess(
    tracked_real_run,
) -> None:
    """Real convergence metrics are charted with ESS clamped at zero."""
    metrics = tracked_real_run["children"]["bayesian-reduced"].data.metrics
    for stage in ("total", "composition"):
        assert metrics[f"diag/{stage}/min_ess_clamped"] >= 0.0
        assert metrics[f"diag/{stage}/ess_valid"] in {0.0, 1.0}
        assert f"diag/{stage}/policy_passed" in metrics


@pytest.mark.slow
def test_real_metric_comparability_separates_densities_from_masses(
    tracked_real_run,
) -> None:
    """The stored comparability record classifies real families correctly."""
    path = tracked_real_run["client"].download_artifacts(
        tracked_real_run["parent"].info.run_id,
        "metric_comparability.json",
        str(tracked_real_run["downloads"] / "parent"),
    )
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    kinds = {
        item["candidate_id"]: item["predictive_measure_kinds"]
        for item in record["candidates"]
    }
    assert kinds["direct-poisson"] == ["discrete"]
    assert kinds["direct-nb2"] == ["discrete"]
    assert kinds["direct-normal"] == ["continuous"]
    scopes = {
        item["candidate_id"]: item["pointwise_log_probability_scopes"]
        for item in record["candidates"]
    }
    assert scopes["independent-nb2"] == ["sequential_joint"]
    assert scopes["bayesian-reduced"] == ["sequential_joint"]
    assert record["rules"]["predictive_nll_is_ranked_across_approaches"] is False


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


@pytest.mark.slow
def test_real_capture_changes_no_comparison_result(tracked_real_run) -> None:
    """Tracked and untracked real comparisons are the same comparison.

    Real models hold state a spy cannot imitate (LightGBM boosters, torch and
    Pyro generators), so capture's extra export, reload and predict is checked
    against a second, untracked run of the same real candidates.
    """
    _, _, plain = _run_real_comparison(capture_artifacts=False)
    tracked = tracked_real_run["result"]

    assert plain.artifacts == ()
    for name in _RESULT_FRAMES:
        pd.testing.assert_frame_equal(getattr(tracked, name), getattr(plain, name))
    assert tracked.freeze.to_dict() == plain.freeze.to_dict()
    for with_capture, without_capture in zip(
        tracked.fold_runs, plain.fold_runs, strict=True
    ):
        assert with_capture.seeds == without_capture.seeds
        np.testing.assert_array_equal(
            with_capture.prediction.cohort_means,
            without_capture.prediction.cohort_means,
        )


# --- Gate 8 final evaluation, disposable real-model run ----------------------
#
# This never touches `artifacts/lockbox/`: the split manifest, table and
# candidates are the same disposable synthetic fixtures `tracked_real_run`
# already uses. The Bayesian final refit is still forced to the full profile
# (Gate 8's own rule), but with a fast full-profile config so this stays a
# disposable validation run rather than the canonical one-time lockbox.


def _gate8_final_refit_factories():
    """Fresh, unfitted real factories for every winner the real comparison may pick.

    Direct and independent factories reuse the CV candidates' fast tuning
    (full-training refit tunes exactly like CV). The Bayesian factory is
    forced to the full profile with its full default draw count: Gate 8's
    own strict policy (`action == "error"`) means an under-sampled profile
    fails its convergence thresholds rather than passing, as the module
    docstring's first attempt (a shortened full profile) found. This is
    still a disposable run against a tiny synthetic table, never the
    canonical lockbox.
    """
    return {
        "direct-poisson": lambda: DirectCohortModel(
            direct_cohort_config=DirectCohortConfig(
                family="poisson", tuning=FAST_TUNING, bootstrap_replicates=4
            )
        ),
        "direct-nb2": lambda: DirectCohortModel(
            direct_cohort_config=DirectCohortConfig(
                family="nb2", tuning=FAST_TUNING, bootstrap_replicates=4
            )
        ),
        "independent-nb2": lambda: IndependentTotalProbabilityModel(
            independent_config=IndependentTotalProbabilityConfig(
                total_family="nb2", tuning=FAST_TUNING, bootstrap_replicates=4
            )
        ),
        "bayesian-reduced": lambda: BayesianConditionalModel(
            bayesian_config=BayesianConditionalConfig(active_profile="full")
        ),
    }


def _gate8_candidates(config):
    """The same five real candidates `_run_real_comparison` declared."""
    return (
        *_real_candidates(config),
        _direct_candidate(config, "nb2", "eligible"),
        _direct_candidate(config, "normal", "diagnostic_comparator"),
    )


@pytest.fixture(scope="module")
def gate8_final_run(tracked_real_run):
    """A disposable Gate 8 final evaluation, linked to the real tracked CV run."""
    result = tracked_real_run["result"]
    split = tracked_real_run["split"]
    config = _experiment_config()
    selection = select_cross_family_winner(result)
    pretest_freeze = result.freeze.with_cross_family_selection(selection)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Bayesian (total|composition) diagnostics failed",
            category=RuntimeWarning,
        )
        evaluation_result = run_final_evaluation(
            split.train_df,
            _learnable_frame(),
            split_manifest=split.manifest,
            cv_result=result,
            pretest_freeze=pretest_freeze,
            candidates=_gate8_candidates(config),
            final_refit_factories=_gate8_final_refit_factories(),
            context=TrackingContext(run_name="gate8-final-disposable"),
            source_cv_run_id=tracked_real_run["parent"].info.run_id,
            evaluation_config=replace(config.evaluation, bootstrap_replicates=6),
        )
    client = tracked_real_run["client"]
    final_runs = [
        run
        for run in client.search_runs([tracked_real_run["parent"].info.experiment_id])
        if run.data.tags.get("run_role") == "final_evaluation"
    ]
    assert len(final_runs) == 1
    parent = final_runs[0]
    children = {
        run.data.tags["candidate_id"]: run
        for run in client.search_runs([parent.info.experiment_id])
        if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    }
    return {
        "client": client,
        "parent": parent,
        "children": children,
        "evaluation_result": evaluation_result,
        "pretest_freeze": pretest_freeze,
        "split": split,
    }


@pytest.mark.slow
def test_gate8_final_run_is_linked_and_complete(gate8_final_run) -> None:
    """The final parent links to the source CV run and finishes with three children."""
    parent, children = gate8_final_run["parent"], gate8_final_run["children"]
    assert parent.info.status == "FINISHED"
    assert parent.data.tags["evidence_complete"] == "true"
    assert parent.data.tags["test_lock_status"] == "opened"
    assert len(children) == 3
    assert set(children) <= {"direct-poisson", "direct-nb2", "independent-nb2", "bayesian-reduced"}
    assert {"independent-nb2", "bayesian-reduced"} <= set(children)
    assert len({"direct-poisson", "direct-nb2"} & set(children)) == 1
    for run in children.values():
        assert run.info.status == "FINISHED"
        assert run.data.tags["evidence_complete"] == "true"
        assert run.data.tags["role"] in {"selected", "comparator"}


@pytest.mark.slow
def test_gate8_final_predictions_share_identical_holdout_ids(gate8_final_run) -> None:
    """Every approach winner predicted the same ordered IDs: the manifest's holdout.

    The manifest is the oracle. ``holdout_building_ids`` is derived from the
    frame the evaluator scored, so comparing against it alone is circular.
    """
    evaluation_result = gate8_final_run["evaluation_result"]
    manifest_holdout = sorted(gate8_final_run["split"].manifest.holdout_building_ids)
    id_sets = [
        tuple(evaluation.predictions_df["building_id"])
        for evaluation in evaluation_result.evaluations
    ]
    assert len(set(id_sets)) == 1
    assert sorted(id_sets[0]) == manifest_holdout
    assert sorted(evaluation_result.holdout_building_ids) == manifest_holdout


@pytest.mark.slow
def test_gate8_final_models_are_loadable_pyfuncs_with_matching_predictions(
    gate8_final_run,
) -> None:
    """Every logged pyfunc reloads and reproduces its evaluation's point predictions."""
    import numpy as np

    evaluation_result = gate8_final_run["evaluation_result"]
    evaluations = {
        evaluation.candidate_id: evaluation for evaluation in evaluation_result.evaluations
    }
    for candidate_id, run in gate8_final_run["children"].items():
        loaded = mlflow.pyfunc.load_model(run.data.tags["final_model_uri"])
        reloaded = loaded.predict(evaluation_result.holdout_input_df).set_index("building_id")
        logged = evaluations[candidate_id].predictions_df.set_index("building_id")
        assert list(reloaded.index) == list(logged.index)
        columns = list(reloaded.columns)
        np.testing.assert_allclose(
            reloaded[columns].to_numpy(dtype=float),
            logged[columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        )
