"""Fold, selection, and freeze evidence records."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..data_splitting import SplitManifest
from ..evaluation import BootstrapEvaluationResult
from ..hashing import column_schema_hash, table_hash
from ..modeling_config import FeatureSpec, ModelingSchema
from ..results import EvaluationResult, PredictionResult
from ..state_bundle import dependency_versions
from .artifacts import FoldArtifactEvidence
from .contracts import ApproachName, CandidateDefinition, FoldIdentity, SelectionPolicy

if TYPE_CHECKING:
    # Deferred to break the import cycle: `final_selection` reads
    # `CrossValidationExperimentResult` from this module, so this module must
    # not import it back at runtime. `from __future__ import annotations`
    # keeps every annotation below a string, so this import is type-checking
    # only.
    from .final_selection import CrossFamilySelection

# Distributions whose versions identify the numerical stack behind a comparison.
# MLflow is deliberately absent: tracking records its own version, and the
# experiment layer does not depend on it.
_PROVENANCE_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "optuna",
    "torch",
    "pyro-ppl",
)


@dataclass(frozen=True)
class FoldRunEvidence:
    """One candidate's immutable evidence from one validation fold."""

    candidate_id: str
    approach: ApproachName
    fold_identity: FoldIdentity
    prediction: PredictionResult
    evaluation: EvaluationResult
    bootstrap: BootstrapEvaluationResult | None
    model_metadata: Mapping[str, Any]
    seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        """Check metadata and seeds are JSON-safe and freeze read-only copies."""
        metadata = dict(self.model_metadata)
        seeds = dict(self.seeds)
        json.dumps({"model_metadata": metadata, "seeds": seeds})
        object.__setattr__(self, "model_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "seeds", MappingProxyType(seeds))


@dataclass(frozen=True)
class FrozenApproachSelection:
    """Selected candidate and complete within-approach decision evidence."""

    approach: ApproachName
    selected_candidate_id: str
    rejected_candidate_ids: tuple[str, ...]
    rejection_reasons: Mapping[str, str]
    criterion_values: Mapping[str, Mapping[str, float]]
    policy: SelectionPolicy

    def __post_init__(self) -> None:
        """Require one reason per rejection and freeze read-only copies."""
        copied = {
            candidate_id: dict(values)
            for candidate_id, values in self.criterion_values.items()
        }
        reasons = dict(self.rejection_reasons)
        if set(reasons) != set(self.rejected_candidate_ids):
            raise ValueError("Every rejected candidate requires one rejection reason")
        json.dumps({"criterion_values": copied, "rejection_reasons": reasons})
        object.__setattr__(self, "criterion_values", MappingProxyType(copied))
        object.__setattr__(self, "rejection_reasons", MappingProxyType(reasons))


@dataclass(frozen=True)
class SelectionFreeze:
    """Serializable proof that training-only choices are frozen.

    Gate 6 always constructs ``cross_family_selection`` and ``test_metrics``
    as ``None``. Gate 8 advances a freeze through exactly two transitions,
    each producing a new immutable value rather than mutating this one:
    Gate 6 freeze -> pretest freeze (``with_cross_family_selection``) ->
    finalized freeze (``with_test_metrics``). Neither transition can be
    replayed: a freeze that already carries a decision refuses a second one,
    and a freeze that already carries test metrics refuses to be finalized
    again (plan section 11, "one-time semantics").
    """

    manifest_fingerprint: str
    outer_training_building_ids: tuple[object, ...]
    fold_identities: tuple[FoldIdentity, ...]
    candidate_descriptors: tuple[Mapping[str, Any], ...]
    selections: tuple[FrozenApproachSelection, ...]
    master_seed: int
    master_seed_source: str = "explicit_argument"
    cross_family_selection: CrossFamilySelection | None = None
    test_metrics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Freeze ``test_metrics`` into a read-only, JSON-safe mapping if set.

        Also enforces the ordering invariant at construction time, not only
        through ``with_test_metrics``: no path, including a future
        deserialization constructor, may produce a freeze with test metrics
        but no recorded decision.
        """
        if self.test_metrics is not None:
            if self.cross_family_selection is None:
                raise ValueError(
                    "A freeze cannot carry test metrics without a prior "
                    "cross-family selection"
                )
            payload = dict(self.test_metrics)
            json.dumps(payload)
            object.__setattr__(self, "test_metrics", MappingProxyType(payload))

    def with_cross_family_selection(
        self, selection: CrossFamilySelection
    ) -> SelectionFreeze:
        """Return a new freeze recording the cross-family decision.

        The decision must be made before any test metric exists and must
        reference this freeze's own manifest and candidates; a second call
        on an already-decided freeze is refused rather than silently
        replacing the recorded decision.
        """
        if self.cross_family_selection is not None:
            raise ValueError(
                "This freeze already records a cross-family selection; a "
                "decision cannot be replaced"
            )
        if selection.manifest_fingerprint != self.manifest_fingerprint:
            raise ValueError(
                "Cross-family selection manifest fingerprint does not match "
                "this freeze"
            )
        known_ids = {descriptor["candidate_id"] for descriptor in self.candidate_descriptors}
        unknown_ids = set(selection.compared_candidate_ids).difference(known_ids)
        if unknown_ids:
            raise ValueError(
                f"Cross-family selection references unknown candidate IDs: "
                f"{sorted(unknown_ids)}"
            )
        return replace(self, cross_family_selection=selection)

    def with_test_metrics(self, test_metrics: Mapping[str, Any]) -> SelectionFreeze:
        """Return a new, finalized freeze recording the one-time test summary.

        Requires a prior cross-family decision and refuses a second test
        result on an already-finalized freeze.
        """
        if self.cross_family_selection is None:
            raise ValueError(
                "Test metrics require a prior cross-family selection; the "
                "decision must predate any test evidence"
            )
        if self.test_metrics is not None:
            raise ValueError(
                "This freeze is already finalized with test metrics; a "
                "second test result cannot be recorded"
            )
        return replace(self, test_metrics=dict(test_metrics))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe freeze evidence for later tracking and final refit."""
        result = {
            "manifest_fingerprint": self.manifest_fingerprint,
            "outer_training_building_ids": list(self.outer_training_building_ids),
            "fold_identities": [asdict(identity) for identity in self.fold_identities],
            "candidate_descriptors": [dict(value) for value in self.candidate_descriptors],
            "selections": [
                {
                    "approach": selection.approach,
                    "selected_candidate_id": selection.selected_candidate_id,
                    "rejected_candidate_ids": list(selection.rejected_candidate_ids),
                    "rejection_reasons": dict(selection.rejection_reasons),
                    "criterion_values": {
                        key: dict(value)
                        for key, value in selection.criterion_values.items()
                    },
                    "policy": asdict(selection.policy),
                }
                for selection in self.selections
            ],
            "master_seed": self.master_seed,
            "master_seed_source": self.master_seed_source,
            "cross_family_selection": (
                self.cross_family_selection.to_dict()
                if self.cross_family_selection is not None
                else None
            ),
            "test_metrics": (
                dict(self.test_metrics) if self.test_metrics is not None else None
            ),
        }
        json.dumps(result)
        return result


@dataclass(frozen=True)
class ExperimentProvenance:
    """What a comparison ran on and with, as content hashes and settings.

    Run identity in this project is content-based: no timestamp, run ID or git
    commit exists inside the package, and a tracking layer supplies those. This
    record holds everything the runner itself can vouch for, so a logged
    comparison can be tied back to its exact data, split, candidate feature
    specifications, seed, settings, and numerical stack. ``split_summary``
    describes the outer split (source table, sizes, holdout fractions) without
    any building ID: the lockbox IDs are pinned by ``manifest_fingerprint``.
    """

    training_data_hash: str
    training_schema_hash: str
    manifest_fingerprint: str
    split_summary: Mapping[str, Any]
    master_seed: int
    master_seed_source: str
    feature_spec_fingerprints: Mapping[str, Mapping[str, str]]
    package_versions: Mapping[str, str]
    run_settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Check the mappings are JSON-safe and freeze read-only copies."""
        summary = copy.deepcopy(dict(self.split_summary))
        fingerprints = {
            candidate_id: MappingProxyType(dict(components))
            for candidate_id, components in self.feature_spec_fingerprints.items()
        }
        versions = dict(self.package_versions)
        settings = copy.deepcopy(dict(self.run_settings))
        json.dumps(
            {
                "summary": summary,
                "fingerprints": {key: dict(value) for key, value in fingerprints.items()},
                "versions": versions,
                "settings": settings,
            }
        )
        object.__setattr__(self, "split_summary", MappingProxyType(summary))
        object.__setattr__(
            self, "feature_spec_fingerprints", MappingProxyType(fingerprints)
        )
        object.__setattr__(self, "package_versions", MappingProxyType(versions))
        object.__setattr__(self, "run_settings", MappingProxyType(settings))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe provenance for tracking."""
        result = {
            "training_data_hash": self.training_data_hash,
            "training_schema_hash": self.training_schema_hash,
            "manifest_fingerprint": self.manifest_fingerprint,
            "split_summary": copy.deepcopy(dict(self.split_summary)),
            "master_seed": self.master_seed,
            "master_seed_source": self.master_seed_source,
            "feature_spec_fingerprints": {
                candidate_id: dict(components)
                for candidate_id, components in self.feature_spec_fingerprints.items()
            },
            "package_versions": dict(self.package_versions),
            "run_settings": copy.deepcopy(dict(self.run_settings)),
        }
        json.dumps(result)
        return result


@dataclass(frozen=True)
class CrossValidationExperimentResult:
    """Training-only fold evidence, selections, and validation importance.

    ``artifacts`` is empty unless the run was started with
    ``capture_artifacts=True``; it then holds one reload-checked state bundle
    per candidate and fold.
    """

    fold_runs: tuple[FoldRunEvidence, ...]
    fold_metrics_df: pd.DataFrame
    aggregate_metrics_df: pd.DataFrame
    bootstrap_intervals_df: pd.DataFrame
    predictions_df: pd.DataFrame
    calibration_df: pd.DataFrame
    importance_df: pd.DataFrame
    importance_summary_df: pd.DataFrame
    fold_coverage_df: pd.DataFrame
    selections: tuple[FrozenApproachSelection, ...]
    freeze: SelectionFreeze
    provenance: ExperimentProvenance
    artifacts: tuple[FoldArtifactEvidence, ...] = ()

    def __post_init__(self) -> None:
        """Deep-copy every dataframe so callers cannot edit recorded evidence."""
        for name in (
            "fold_metrics_df",
            "aggregate_metrics_df",
            "bootstrap_intervals_df",
            "predictions_df",
            "calibration_df",
            "importance_df",
            "importance_summary_df",
            "fold_coverage_df",
        ):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))


def _prediction_records(
    candidate: CandidateDefinition,
    identity: FoldIdentity,
    prediction: PredictionResult,
) -> list[dict[str, Any]]:
    """One tidy prediction row per validation building for one candidate and fold."""
    records: list[dict[str, Any]] = []
    for row_index, building_id in enumerate(prediction.building_ids):
        record: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "approach": candidate.approach,
            "fold_index": identity.fold_index,
            "building_id": building_id,
            "total_mean": float(prediction.total_mean[row_index]),
        }
        for cohort_index, cohort in enumerate(prediction.cohort_names):
            record[f"{cohort}_mean"] = float(
                prediction.cohort_means[row_index, cohort_index]
            )
            record[f"{cohort}_probability"] = float(
                prediction.age_group_probabilities[row_index, cohort_index]
            )
        records.append(record)
    return records


def _calibration_record(
    candidate: CandidateDefinition,
    identity: FoldIdentity,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize one fold's calibration evidence, promoting the decision fields."""
    model_metadata = metadata.get("model")
    calibration = (
        model_metadata.get("calibration")
        if isinstance(model_metadata, Mapping)
        else None
    )
    calibration_supported = candidate.approach == "IndependentTotalProbabilityModel"
    status = (
        "recorded"
        if calibration is not None
        else "missing" if calibration_supported else "not_applicable"
    )
    # Promote the decision fields downstream tracking needs so they are not
    # buried inside the serialized payload.
    details = calibration if isinstance(calibration, Mapping) else {}
    return {
        "candidate_id": candidate.candidate_id,
        "approach": candidate.approach,
        "fold_index": identity.fold_index,
        "status": status,
        "calibration_supported": calibration_supported,
        "calibration_method": details.get("method"),
        "temperature": details.get("temperature"),
        "fitted_temperature": details.get("fitted_temperature"),
        "retained": details.get("retained"),
        "raw_weighted_nll": details.get("raw_weighted_nll"),
        "selected_weighted_nll": details.get("selected_weighted_nll"),
        "calibration": json.dumps(calibration, sort_keys=True),
    }


def _manifest_fingerprint(manifest: SplitManifest) -> str:
    """SHA-256 of the split manifest's canonical JSON."""
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_summary(manifest: SplitManifest) -> dict[str, Any]:
    """The outer split's source, settings and sizes, without any building ID."""
    return {
        "strategy": manifest.strategy,
        "source_table_hash": manifest.source_table_hash,
        "column_schema_hash": manifest.column_schema_hash,
        "n_rows": manifest.n_rows,
        "n_neighborhoods": manifest.n_neighborhoods,
        "requested_holdout_fraction": manifest.requested_holdout_fraction,
        "realized_holdout_fraction": manifest.realized_holdout_fraction,
        "training_building_count": len(manifest.training_building_ids),
        "holdout_building_count": len(manifest.holdout_building_ids),
        "seed_source": manifest.seed_source,
        "deployment_claim": manifest.deployment_claim,
    }


def _feature_spec_fingerprint(spec: FeatureSpec) -> str:
    """SHA-256 of a feature specification's canonical JSON."""
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_provenance(
    outer_train_df: pd.DataFrame,
    *,
    schema: ModelingSchema,
    split_manifest: SplitManifest,
    candidates: Mapping[str, CandidateDefinition],
    master_seed: int,
    master_seed_source: str,
    run_settings: Mapping[str, Any],
) -> ExperimentProvenance:
    """Hash the outer training frame and record the comparison's settings."""
    return ExperimentProvenance(
        training_data_hash=table_hash(
            outer_train_df, id_column=schema.building_id_column
        ),
        training_schema_hash=column_schema_hash(outer_train_df),
        manifest_fingerprint=_manifest_fingerprint(split_manifest),
        split_summary=_split_summary(split_manifest),
        master_seed=master_seed,
        master_seed_source=master_seed_source,
        feature_spec_fingerprints={
            candidate_id: {
                spec.component: _feature_spec_fingerprint(spec)
                for spec in candidates[candidate_id].component_feature_specs
            }
            for candidate_id in sorted(candidates)
        },
        package_versions=dependency_versions(*_PROVENANCE_PACKAGES),
        run_settings=run_settings,
    )
