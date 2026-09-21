"""Fitted-fold bundle capture and run provenance (Gate 7, MLflow-free)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from age_group_prediction.experiment import FoldArtifactEvidence, ReloadCheck
from age_group_prediction.experiment.evidence import _feature_spec_fingerprint
from age_group_prediction.hashing import column_schema_hash, table_hash
from age_group_prediction.modeling_config import (
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
)
from tests.unit.bundle_spy import BundleSpyModel, run_spy_experiment

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


class _RandomPointSpyModel(BundleSpyModel):
    """Consumes the prediction generator for point means, like the Bayesian
    model does for rows in neighborhoods it never saw."""

    implementation_version = "random-point-spy-1"

    def _predict_model(self, **kwargs):
        """Add generator-drawn noise to the spy's deterministic totals."""
        result = super()._predict_model(**kwargs)
        noise = kwargs["rng"].uniform(0.0, 0.1, size=len(result.total_mean))
        total = result.total_mean + noise
        return type(result).from_means(
            building_ids=result.building_ids,
            cohort_names=result.cohort_names,
            total_mean=total,
            cohort_means=total[:, None] * np.array([0.2, 0.5, 0.3]),
            parametric_distributions=result.parametric_distributions,
        )


def test_capture_records_a_reload_checked_bundle_for_every_candidate_and_fold() -> None:
    """Capture yields one exactly matching, JSON-safe bundle per candidate and fold."""
    _, result = run_spy_experiment(capture_artifacts=True)

    expected = {
        (run.candidate_id, run.fold_identity.fold_index) for run in result.fold_runs
    }
    captured = [(item.candidate_id, item.fold_index) for item in result.artifacts]
    assert len(captured) == len(expected) == 10
    assert set(captured) == expected
    for item in result.artifacts:
        check = item.reload_check
        assert check.passed and check.tolerance == 0.0
        assert dict(check.max_abs_differences) == {
            "total_mean": 0.0,
            "cohort_means": 0.0,
            "age_group_probabilities": 0.0,
            "parametric_distributions": 0.0,
        }
        assert item.state_bundle["model_class"] == "BundleSpyModel"
        # Written as JSON, so it must survive a strict round trip.
        json.dumps(dict(item.state_bundle), allow_nan=False)


def test_default_run_captures_nothing_and_capture_changes_no_result() -> None:
    """Tracking-on and tracking-off comparisons must be the same comparison.

    The model consumes its generator for point predictions, so the check fails
    if the reload check ever shared a stream with the runner.
    """
    _, captured = run_spy_experiment(
        capture_artifacts=True, model_class=_RandomPointSpyModel
    )
    _, plain = run_spy_experiment(
        capture_artifacts=False, model_class=_RandomPointSpyModel
    )

    assert plain.artifacts == ()
    assert len(captured.artifacts) == 10
    assert all(item.reload_check.passed for item in captured.artifacts)
    for name in _RESULT_FRAMES:
        pd.testing.assert_frame_equal(getattr(captured, name), getattr(plain, name))
    assert captured.freeze.to_dict() == plain.freeze.to_dict()
    assert captured.selections == plain.selections
    for with_capture, without_capture in zip(
        captured.fold_runs, plain.fold_runs, strict=True
    ):
        assert with_capture.seeds == without_capture.seeds
        np.testing.assert_array_equal(
            with_capture.prediction.total_mean, without_capture.prediction.total_mean
        )

    captured_provenance = captured.provenance.to_dict()
    plain_provenance = plain.provenance.to_dict()
    assert captured_provenance["run_settings"].pop("capture_artifacts") is True
    assert plain_provenance["run_settings"].pop("capture_artifacts") is False
    assert captured_provenance == plain_provenance


def test_a_bundle_that_does_not_reload_aborts_the_run_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload whose predictions drift aborts the run, naming the operation."""
    original = BundleSpyModel._from_model_state.__func__

    def drifted(cls, state, **base_arguments):
        """Reload faithfully, then shift the offset so predictions differ."""
        model = original(cls, state, **base_arguments)
        model.offset += 1.0
        return model

    monkeypatch.setattr(BundleSpyModel, "_from_model_state", classmethod(drifted))

    with pytest.raises(
        RuntimeError, match="failed during capture_artifacts on fold"
    ) as caught:
        run_spy_experiment(capture_artifacts=True)
    assert "does not reproduce" in str(caught.value)


def test_reload_check_sees_distribution_parameters_not_only_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bundle that loses NB2 dispersion keeps every mean yet scores differently."""
    original = BundleSpyModel._from_model_state.__func__

    def dropped_dispersion(cls, state, **base_arguments):
        """Reload faithfully, then turn an NB2 spy into a Poisson one."""
        model = original(cls, state, **base_arguments)
        if model.family == "nb2":
            model.family = "poisson"
        return model

    monkeypatch.setattr(
        BundleSpyModel, "_from_model_state", classmethod(dropped_dispersion)
    )

    with pytest.raises(RuntimeError, match="during capture_artifacts") as caught:
        run_spy_experiment(capture_artifacts=True)
    assert "'parametric_distributions': None" in str(caught.value)


def test_artifact_evidence_does_not_share_state_with_its_input() -> None:
    """Recorded bundles are private, read-only, and strictly JSON-safe."""
    bundle = {"model_class": "X", "model_state": {"offset": 1.0}}
    evidence = FoldArtifactEvidence(
        candidate_id="c",
        fold_index=0,
        state_bundle=bundle,
        reload_check=ReloadCheck({"total_mean": 0.0}, 0.0, True),
    )

    bundle["model_state"]["offset"] = 2.0
    assert evidence.state_bundle["model_state"]["offset"] == 1.0
    with pytest.raises(TypeError):
        evidence.state_bundle["model_class"] = "Y"  # type: ignore[index]
    with pytest.raises(ValueError):
        FoldArtifactEvidence(
            candidate_id="c",
            fold_index=0,
            state_bundle={"value": float("nan")},
            reload_check=ReloadCheck({"total_mean": 0.0}, 0.0, True),
        )


def test_provenance_ties_the_result_to_its_data_split_specs_and_settings() -> None:
    """Provenance hashes, fingerprints and settings are correct and reproducible."""
    split, result = run_spy_experiment(capture_artifacts=True)
    provenance = result.provenance

    assert provenance.training_data_hash == table_hash(
        split.train_df, id_column="building_id"
    )
    assert provenance.training_schema_hash == column_schema_hash(split.train_df)
    assert provenance.manifest_fingerprint == result.freeze.manifest_fingerprint
    summary = provenance.split_summary
    assert summary["source_table_hash"] == split.manifest.source_table_hash
    assert summary["holdout_building_count"] == len(split.manifest.holdout_building_ids)
    assert summary["training_building_count"] == len(split.train_df)
    assert not any("building_ids" in key for key in summary)
    assert (provenance.master_seed, provenance.master_seed_source) == (
        31,
        "explicit_argument",
    )
    assert provenance.feature_spec_fingerprints["direct-poisson"] == {
        "tree": _feature_spec_fingerprint(DEFAULT_TREE_FEATURE_SPEC)
    }
    assert set(provenance.feature_spec_fingerprints["bayesian-reduced"]) == {
        "total_count",
        "composition",
    }
    assert _feature_spec_fingerprint(DEFAULT_TREE_FEATURE_SPEC) != (
        _feature_spec_fingerprint(DEFAULT_TOTAL_FEATURE_SPEC)
    )
    assert set(provenance.package_versions) == {
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "optuna",
        "torch",
        "pyro-ppl",
    }
    settings = provenance.to_dict()["run_settings"]
    assert settings["building_id_column"] == "building_id"
    assert settings["evaluation_config"]["bootstrap_replicates"] == 3
    assert settings["require_convergence"] is False
    assert settings["capture_artifacts"] is True
    assert settings["fold_count"] == 2
    json.dumps(provenance.to_dict())

    _, repeated = run_spy_experiment(capture_artifacts=True)
    assert repeated.provenance == provenance
