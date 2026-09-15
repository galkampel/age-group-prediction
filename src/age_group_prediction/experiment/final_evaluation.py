"""Full-training refit and one-time lockbox evaluation (Gate 8, phase 3).

Gate 6/7 only ever fit a candidate on one fold's fit partition; those fold
models are training-only evidence, never a deployment model (session handoff
section 1). This module does the two things that must happen after a
cross-family decision is frozen and before Gate 8 is complete:

1. :func:`refit_frozen_approach_winners` refits exactly one winner per
   approach on the *complete* outer-training partition, proves each refit's
   state bundle reproduces the fitted model's point predictions exactly, and
   enforces that a Bayesian refit actually ran under the full NUTS profile and
   strict diagnostic policy (decisions 3.1/4.4 of the handoff).
2. :func:`evaluate_frozen_models_on_lockbox` is the *only* Gate 8 API that
   receives the full modeling table and may materialize the holdout
   partition. It replays the persisted split manifest, predicts and scores
   all three refit winners once on the identical ordered holdout rows, and
   returns a finalized freeze plus immutable test evidence.

Both functions are pure: no MLflow import, no test-driven ranking (the
cross-family decision was already frozen by
:mod:`age_group_prediction.experiment.final_selection` before either function
runs), and no mutation of the ``CrossValidationExperimentResult`` or
``SelectionFreeze`` they are given — every transition returns a new value.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..data_splitting import SplitManifest, replay_split_manifest
from ..evaluation import evaluate_predictions, neighborhood_cluster_bootstrap
from ..hashing import column_schema_hash, table_hash
from ..modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    BayesianConditionalConfig,
    EvaluationConfig,
    ModelingSchema,
    OuterSplitConfig,
    PredictionConfig,
)
from ..models.base import BaseAgeGroupModel
from ..results import PredictionResult
from .artifacts import _RELOAD_TOLERANCE, ReloadCheck, _prediction_differences
from .contracts import ApproachName, CandidateDefinition
from .evidence import (
    CrossValidationExperimentResult,
    SelectionFreeze,
    _manifest_fingerprint,
)
from .final_selection import select_cross_family_winner
from .partitions import _assert_prediction_alignment, _sorted_ids
from .seeds import _stable_seed

__all__ = [
    "FinalCandidateEvaluation",
    "FinalEvaluationResult",
    "FinalModelArtifact",
    "FinalRefitEvidence",
    "evaluate_frozen_models_on_lockbox",
    "final_attempt_fingerprint",
    "refit_frozen_approach_winners",
    "verify_pretest_freeze",
]

_REQUIRED_APPROACHES: frozenset[ApproachName] = frozenset(
    {"DirectCohortModel", "IndependentTotalProbabilityModel", "BayesianConditionalModel"}
)
_BAYESIAN_DIAGNOSTIC_STAGES = ("total", "composition")
# Small enough to predict in milliseconds, large enough that a broken feature
# transformer or a shape bug would surface rather than hide in one row.
_SMOKE_FRAME_SIZE = 20


@dataclass(frozen=True)
class FinalRefitEvidence:
    """Immutable, JSON-safe evidence for one approach winner's full refit."""

    candidate_id: str
    approach: ApproachName
    manifest_fingerprint: str
    training_data_hash: str
    training_schema_hash: str
    seeds: Mapping[str, int]
    model_metadata: Mapping[str, Any]
    state_bundle: Mapping[str, Any]
    reload_check: ReloadCheck
    smoke_frame_building_ids: tuple[object, ...]

    def __post_init__(self) -> None:
        """Freeze mappings and require the bundle/metadata to be strictly JSON-safe."""
        seeds = dict(self.seeds)
        metadata = copy.deepcopy(dict(self.model_metadata))
        bundle = copy.deepcopy(dict(self.state_bundle))
        json.dumps({"seeds": seeds, "model_metadata": metadata})
        json.dumps(bundle, allow_nan=False)
        object.__setattr__(self, "seeds", MappingProxyType(seeds))
        object.__setattr__(self, "model_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "state_bundle", MappingProxyType(bundle))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe refit evidence."""
        return {
            "candidate_id": self.candidate_id,
            "approach": self.approach,
            "manifest_fingerprint": self.manifest_fingerprint,
            "training_data_hash": self.training_data_hash,
            "training_schema_hash": self.training_schema_hash,
            "seeds": dict(self.seeds),
            "model_metadata": dict(self.model_metadata),
            "state_bundle": dict(self.state_bundle),
            "reload_check": self.reload_check.to_dict(),
            "smoke_frame_building_ids": list(self.smoke_frame_building_ids),
        }


@dataclass(frozen=True)
class FinalModelArtifact:
    """One approach winner's live refit model, paired with its immutable evidence.

    ``model`` is the only place in Gate 8 evidence that holds a live, mutable
    fitted object; it is excluded from equality/repr so nothing accidentally
    tries to compare or print it, and every other field here (and everything
    inside ``evidence``) is JSON-safe.
    """

    candidate_id: str
    approach: ApproachName
    model: BaseAgeGroupModel = field(repr=False, compare=False)
    evidence: FinalRefitEvidence


@dataclass(frozen=True)
class FinalCandidateEvaluation:
    """One approach winner's one-time holdout predictions, metrics, and intervals."""

    candidate_id: str
    approach: ApproachName
    role: Literal["selected", "comparator"]
    seeds: Mapping[str, int]
    # `compare=False`: a DataFrame's `==` returns a DataFrame, not a bool, so
    # the dataclass-generated `__eq__` would raise on any of these fields.
    metrics_df: pd.DataFrame = field(compare=False)
    intervals_df: pd.DataFrame = field(compare=False)
    predictions_df: pd.DataFrame = field(compare=False)
    metric_comparability: str
    model_metadata: Mapping[str, Any]
    evaluation_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Deep-copy frames and freeze JSON-safe mappings."""
        if self.role not in {"selected", "comparator"}:
            raise ValueError("Final candidate evaluation role must be 'selected' or 'comparator'")
        object.__setattr__(self, "metrics_df", self.metrics_df.copy(deep=True))
        object.__setattr__(self, "intervals_df", self.intervals_df.copy(deep=True))
        object.__setattr__(self, "predictions_df", self.predictions_df.copy(deep=True))
        seeds = dict(self.seeds)
        metadata = copy.deepcopy(dict(self.model_metadata))
        evaluation_metadata = copy.deepcopy(dict(self.evaluation_metadata))
        json.dumps(
            {
                "seeds": seeds,
                "model_metadata": metadata,
                "evaluation_metadata": evaluation_metadata,
            }
        )
        object.__setattr__(self, "seeds", MappingProxyType(seeds))
        object.__setattr__(self, "model_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "evaluation_metadata", MappingProxyType(evaluation_metadata))


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Immutable evidence from one lockbox evaluation: all three approach winners.

    ``holdout_input_df`` is the target-free frame every refit model actually
    predicted from: the holdout building, feature and exposure columns, in
    holdout row order, with every target column and every other modeling
    column excluded. It exists so a caller (tracking's models-from-code
    logger) can prove a loaded pyfunc reproduces these predictions without
    replaying the split manifest a second time (Boundary: only
    ``evaluate_frozen_models_on_lockbox`` may materialize the holdout).
    """

    manifest_fingerprint: str
    holdout_building_ids: tuple[object, ...]
    evaluations: tuple[FinalCandidateEvaluation, ...]
    finalized_freeze: SelectionFreeze
    holdout_input_df: pd.DataFrame = field(compare=False)
    # The schema the evaluation actually used: leakage is checked against its
    # target columns, never the default schema's.
    schema: ModelingSchema = field(default=DEFAULT_MODELING_SCHEMA, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Deep-copy the holdout input frame and refuse a leaked target column."""
        leaked = set(self.holdout_input_df.columns) & set(self.schema.target_columns)
        if leaked:
            raise ValueError(
                f"holdout_input_df must not carry target columns: {sorted(leaked)}"
            )
        object.__setattr__(
            self, "holdout_input_df", self.holdout_input_df.copy(deep=True)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe summary evidence (the fitted models are not included)."""
        return {
            "manifest_fingerprint": self.manifest_fingerprint,
            "holdout_building_ids": list(self.holdout_building_ids),
            "evaluations": [
                {
                    "candidate_id": evaluation.candidate_id,
                    "approach": evaluation.approach,
                    "role": evaluation.role,
                    "seeds": dict(evaluation.seeds),
                    "metric_comparability": evaluation.metric_comparability,
                }
                for evaluation in self.evaluations
            ],
            "finalized_freeze": self.finalized_freeze.to_dict(),
        }


@contextmanager
def _final_failure_context(candidate_id: str, operation: str) -> Iterator[None]:
    """Name the candidate and stage on failure, preserving Bayesian diagnostic context.

    A Bayesian diagnostic failure attaches ``stage``, ``diagnostics``, and
    ``failures`` to the raised exception (``models.bayesian_conditional``); an
    expensive full-profile run that fails deserves to keep that evidence
    rather than have it discarded by a generic wrapper.
    """
    try:
        yield
    except Exception as error:
        wrapped = RuntimeError(
            f"Candidate '{candidate_id}' failed during final {operation}: "
            f"{type(error).__name__}: {error}"
        )
        for attribute in ("stage", "diagnostics", "failures"):
            if hasattr(error, attribute):
                setattr(wrapped, attribute, getattr(error, attribute))
        raise wrapped from error


def _smoke_frame(outer_train_df: pd.DataFrame, *, schema: ModelingSchema) -> pd.DataFrame:
    """A small, deterministic, training-only frame for the final reload check."""
    ordered = outer_train_df.sort_values(schema.building_id_column, ignore_index=True)
    return ordered.head(min(_SMOKE_FRAME_SIZE, len(ordered))).reset_index(drop=True)


def _replay_config(*, schema: ModelingSchema) -> OuterSplitConfig:
    """The canonical outer-split config used only to replay a persisted manifest.

    Takes no manifest argument on purpose. ``replay_split_manifest`` checks a
    manifest's recorded strategy against this config's ``strategy_version``,
    and there is no manifest field this function could read that would not
    make that check compare a value to itself; only the column names, which
    name this modeling table's schema rather than describe the splitting
    algorithm, are ever overridden from their canonical defaults.
    """
    return OuterSplitConfig(
        building_id_column=schema.building_id_column,
        neighborhood_id_column=schema.neighborhood_id_column,
    )


def _final_reload_check(
    model: BaseAgeGroupModel,
    bundle: Mapping[str, Any],
    smoke_df: pd.DataFrame,
    *,
    seed: int,
) -> ReloadCheck:
    """Reload a bundle from its JSON text and require exact point-prediction equality.

    Mirrors ``experiment.artifacts.capture_fold_artifact``: the reloaded model
    is rebuilt **without** ``train_df`` (the bundle must be self-contained),
    and both models predict the same smoke frame with identically seeded
    fresh generators so any randomness a model consumes runs the same stream
    on both sides.
    """
    reloaded = type(model).from_state_bundle(json.loads(json.dumps(bundle)))
    original, restored = (
        candidate.predict(
            smoke_df,
            prediction_config=PredictionConfig(),
            rng=np.random.default_rng(seed),
        )
        for candidate in (model, reloaded)
    )
    differences = _prediction_differences(original, restored)
    check = ReloadCheck(
        max_abs_differences=differences,
        tolerance=_RELOAD_TOLERANCE,
        passed=max(differences.values()) <= _RELOAD_TOLERANCE,
    )
    if not check.passed:
        raise ValueError(
            "Final refit state bundle does not reproduce the fitted model's "
            f"predictions: {check.to_dict()}"
        )
    return check


# Which way is stricter for each diagnostic threshold: a recorded value may
# never be looser than the package's default full-profile policy.
_THRESHOLD_STRICTER_DIRECTION: dict[str, Literal["lower", "higher"]] = {
    "maximum_rhat": "lower",
    "minimum_effective_sample_size": "higher",
    "maximum_divergences": "lower",
    "minimum_mean_accept_prob": "higher",
    "maximum_mean_accept_prob": "lower",
    "maximum_tree_depth_saturation": "lower",
}
_DRAW_COUNT_KEYS = ("chains", "warmup_steps", "posterior_samples")


def _require_full_bayesian_policy(model: BaseAgeGroupModel, *, candidate_id: str) -> None:
    """Confirm a fitted Bayesian model actually ran the full profile and strict policy.

    The ``active_profile`` label and the policy ``action`` alone are not proof:
    a configuration can label a profile ``"full"`` while giving it reduced
    draws or loose thresholds. Each stage's *recorded* chains, warmup and
    posterior samples must therefore be at least the package's default full
    profile, and each recorded threshold at least as strict as its default
    full diagnostics. A stricter configuration passes; a weaker one does not.
    """
    defaults = BayesianConditionalConfig()
    diagnostics = model.get_metadata()["model"].get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise TypeError(f"Candidate {candidate_id!r} reports no Bayesian diagnostics")
    for stage in _BAYESIAN_DIAGNOSTIC_STAGES:
        stage_diagnostics = diagnostics.get(stage)
        if not isinstance(stage_diagnostics, Mapping):
            raise TypeError(f"Candidate {candidate_id!r} is missing {stage!r} diagnostics")
        if stage_diagnostics.get("active_profile") != "full":
            raise ValueError(
                f"Candidate {candidate_id!r} final refit did not run the full "
                f"Bayesian profile for stage {stage!r}"
            )
        thresholds = stage_diagnostics.get("policy_thresholds")
        if not isinstance(thresholds, Mapping) or thresholds.get("action") != "error":
            raise ValueError(
                f"Candidate {candidate_id!r} final refit did not use the strict "
                f"diagnostic policy (action='error') for stage {stage!r}"
            )
        for key in _DRAW_COUNT_KEYS:
            recorded = stage_diagnostics.get(key)
            minimum = getattr(defaults.full_profile, key)
            if not isinstance(recorded, int) or recorded < minimum:
                raise ValueError(
                    f"Candidate {candidate_id!r} final refit ran {key}={recorded!r} "
                    f"for stage {stage!r}, below the full-profile minimum {minimum}"
                )
        for key, stricter in _THRESHOLD_STRICTER_DIRECTION.items():
            recorded = thresholds.get(key)
            default = getattr(defaults.full_diagnostics, key)
            looser = (
                not isinstance(recorded, int | float)
                or (stricter == "lower" and recorded > default)
                or (stricter == "higher" and recorded < default)
            )
            if looser:
                raise ValueError(
                    f"Candidate {candidate_id!r} final refit used {key}={recorded!r} "
                    f"for stage {stage!r}, looser than the full-profile policy's {default}"
                )


def _verify_refit_preconditions(
    outer_train_df: pd.DataFrame,
    *,
    split_manifest: SplitManifest,
    cv_result: CrossValidationExperimentResult,
    candidates: Sequence[CandidateDefinition],
    final_refit_factories: Mapping[str, Callable[[], BaseAgeGroupModel]],
    schema: ModelingSchema,
) -> None:
    """Verify runtime refit inputs reproduce exactly what Gate 6/7 froze.

    Boundary 1 requires the cross-family decision to predate any holdout
    access; this check instead protects the earlier boundary that a full
    refit must train on the *same* rows, schema, candidates, and seed the
    frozen CV comparison used, or the refit is not evidence about the
    candidate that was actually selected.
    """
    schema.validate_table(outer_train_df)
    freeze = cv_result.freeze
    manifest_fingerprint = _manifest_fingerprint(split_manifest)
    if manifest_fingerprint != freeze.manifest_fingerprint:
        raise ValueError("Split manifest does not match the frozen CV evidence")
    if manifest_fingerprint != cv_result.provenance.manifest_fingerprint:
        raise ValueError("Split manifest does not match the CV provenance record")

    outer_ids = _sorted_ids(set(outer_train_df[schema.building_id_column]))
    if outer_ids != freeze.outer_training_building_ids:
        raise ValueError("Outer-training building IDs do not match the frozen CV evidence")

    if table_hash(outer_train_df, id_column=schema.building_id_column) != (
        cv_result.provenance.training_data_hash
    ):
        raise ValueError("Outer-training data does not match the CV provenance hash")
    if column_schema_hash(outer_train_df) != cv_result.provenance.training_schema_hash:
        raise ValueError("Outer-training schema does not match the CV provenance hash")

    approaches = {selection.approach for selection in cv_result.selections}
    if approaches != _REQUIRED_APPROACHES:
        raise ValueError("Cross-validation result must freeze exactly one winner per approach")

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    descriptor_by_id = {
        descriptor["candidate_id"]: descriptor for descriptor in freeze.candidate_descriptors
    }
    selected_ids = {selection.selected_candidate_id for selection in cv_result.selections}
    missing_candidates = selected_ids.difference(candidate_by_id)
    if missing_candidates:
        raise ValueError(
            f"Supplied candidates are missing the frozen winners: {sorted(missing_candidates)}"
        )
    missing_factories = selected_ids.difference(final_refit_factories)
    if missing_factories:
        raise ValueError(
            "final_refit_factories is missing the frozen winners: "
            f"{sorted(missing_factories)}"
        )
    for candidate_id in selected_ids:
        descriptor = descriptor_by_id.get(candidate_id)
        if descriptor is None:
            raise ValueError(f"No frozen descriptor recorded for candidate {candidate_id!r}")
        if candidate_by_id[candidate_id].to_descriptor() != dict(descriptor):
            raise ValueError(
                f"Candidate {candidate_id!r}'s runtime definition (feature specs, "
                "metrics, or configuration) does not match the descriptor Gate 6 froze"
            )

    if freeze.master_seed < 0:
        raise ValueError("Frozen master seed must be nonnegative")
    if not freeze.master_seed_source:
        raise ValueError("Frozen master seed source must be recorded")
    # Every refit and evaluation seed derives from the freeze's master seed, so
    # it must be the seed the CV comparison actually ran with.
    if (freeze.master_seed, freeze.master_seed_source) != (
        cv_result.provenance.master_seed,
        cv_result.provenance.master_seed_source,
    ):
        raise ValueError("Frozen master seed does not match the CV provenance record")


def verify_pretest_freeze(
    pretest_freeze: SelectionFreeze, cv_result: CrossValidationExperimentResult
) -> None:
    """Refuse a pretest freeze that is not the rule's decision on this CV result.

    ``SelectionFreeze.with_cross_family_selection`` accepts any decision whose
    manifest fingerprint and candidate IDs fit the freeze, including a
    hand-built one. Before the lockbox opens, the decision must instead be
    exactly what :func:`select_cross_family_winner` produces from
    ``cv_result``, recorded on exactly ``cv_result``'s freeze: the same
    selections, descriptors, folds and master seed. Without this, someone who
    had already seen test evidence could hand the guarded path a different
    choice, or a seed that no longer matches the refit's.
    """
    decision = pretest_freeze.cross_family_selection
    if decision is None:
        raise ValueError(
            "The final evaluation requires a pretest freeze with a recorded "
            "cross-family selection"
        )
    if pretest_freeze.test_metrics is not None:
        raise ValueError("This pretest freeze already carries test metrics")
    undecided = {**pretest_freeze.to_dict(), "cross_family_selection": None}
    if undecided != cv_result.freeze.to_dict():
        raise ValueError("Pretest freeze does not extend this CV result's freeze")
    if decision.to_dict() != select_cross_family_winner(cv_result).to_dict():
        raise ValueError(
            "Pretest cross-family decision is not the one the rule produces "
            "from this CV result"
        )


def final_attempt_fingerprint(
    pretest_freeze: SelectionFreeze,
    refit_result: Sequence[FinalModelArtifact],
    *,
    candidates: Sequence[CandidateDefinition],
    evaluation_config: EvaluationConfig | None,
    schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
) -> str:
    """Return the SHA-256 of everything one final attempt refits and scores.

    The cross-family decision alone does not pin a final attempt. The same
    decision can be refit with other hyperparameters, priors or seeds, or
    scored with other prediction or bootstrap settings, and each changes the
    test numbers. This hashes:

    - the pretest freeze (decision, selections, descriptors, folds, master seed);
    - each refit's seeds, training hashes, constructor configuration
      (``BaseAgeGroupModel.configuration_record``, never fitted values), and
      the dependency versions its state bundle records;
    - the selected candidates' descriptors and prediction settings;
    - the evaluation settings and the schema.

    Tracking refuses a retry after the lockbox opened unless this matches.
    Model source code is not hashed: a retry assumes the same package code,
    identified only by each model's ``implementation_version``.
    """
    selected_ids = {selection.selected_candidate_id for selection in pretest_freeze.selections}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    missing_candidates = selected_ids.difference(candidate_by_id)
    if missing_candidates:
        raise ValueError(
            f"Supplied candidates are missing the frozen winners: {sorted(missing_candidates)}"
        )
    payload = {
        "pretest_freeze": pretest_freeze.to_dict(),
        "refits": [
            {
                "candidate_id": artifact.candidate_id,
                "approach": artifact.approach,
                "manifest_fingerprint": artifact.evidence.manifest_fingerprint,
                "training_data_hash": artifact.evidence.training_data_hash,
                "training_schema_hash": artifact.evidence.training_schema_hash,
                "seeds": dict(artifact.evidence.seeds),
                "dependency_versions": artifact.evidence.state_bundle.get(
                    "dependency_versions"
                ),
                "model_configuration": artifact.model.configuration_record(),
            }
            for artifact in sorted(refit_result, key=lambda artifact: artifact.candidate_id)
        ],
        "candidates": [
            {
                "descriptor": candidate_by_id[candidate_id].to_descriptor(),
                "prediction_config": asdict(candidate_by_id[candidate_id].prediction_config),
            }
            for candidate_id in sorted(selected_ids)
        ],
        "evaluation_config": (
            asdict(evaluation_config) if evaluation_config is not None else None
        ),
        "schema": asdict(schema),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def refit_frozen_approach_winners(
    outer_train_df: pd.DataFrame,
    *,
    split_manifest: SplitManifest,
    cv_result: CrossValidationExperimentResult,
    candidates: Sequence[CandidateDefinition],
    final_refit_factories: Mapping[str, Callable[[], BaseAgeGroupModel]],
    schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
) -> tuple[FinalModelArtifact, ...]:
    """Refit exactly one winner per approach on the complete outer-training data.

    Reuses ``cv_result.freeze.master_seed`` rather than taking a new seed
    argument, scoped under a ``final/{candidate_id}/{operation}`` purpose
    string distinct from every CV-fold purpose that seed already produced
    (section 4, "purpose-scoped randomness"). Never replays the split or
    reads a test target: this function only ever sees ``outer_train_df``.
    """
    _verify_refit_preconditions(
        outer_train_df,
        split_manifest=split_manifest,
        cv_result=cv_result,
        candidates=candidates,
        final_refit_factories=final_refit_factories,
        schema=schema,
    )
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    master_seed = cv_result.freeze.master_seed
    manifest_fingerprint = _manifest_fingerprint(split_manifest)
    smoke_df = _smoke_frame(outer_train_df, schema=schema)
    training_data_hash = table_hash(outer_train_df, id_column=schema.building_id_column)
    training_schema_hash = column_schema_hash(outer_train_df)

    artifacts: list[FinalModelArtifact] = []
    for selection in sorted(cv_result.selections, key=lambda selection: selection.approach):
        candidate_id = selection.selected_candidate_id
        candidate = candidate_by_id[candidate_id]
        seeds = {
            "fit": _stable_seed(master_seed, f"final/{candidate_id}/fit"),
            "reload_check": _stable_seed(master_seed, f"final/{candidate_id}/reload_check"),
        }
        model = final_refit_factories[candidate_id]()
        if not isinstance(model, BaseAgeGroupModel):
            raise TypeError("Final refit factories must return BaseAgeGroupModel")
        with _final_failure_context(candidate_id, "fit"):
            model.fit(
                outer_train_df,
                feature_spec=candidate.fit_feature_spec,
                rng=np.random.default_rng(seeds["fit"]),
            )
        if selection.approach == "BayesianConditionalModel":
            _require_full_bayesian_policy(model, candidate_id=candidate_id)
        metadata = model.get_metadata()
        bundle = model.to_state_bundle()
        with _final_failure_context(candidate_id, "reload_check"):
            reload_check = _final_reload_check(
                model, bundle, smoke_df, seed=seeds["reload_check"]
            )
        evidence = FinalRefitEvidence(
            candidate_id=candidate_id,
            approach=selection.approach,
            manifest_fingerprint=manifest_fingerprint,
            training_data_hash=training_data_hash,
            training_schema_hash=training_schema_hash,
            seeds=seeds,
            model_metadata=metadata,
            state_bundle=bundle,
            reload_check=reload_check,
            smoke_frame_building_ids=tuple(smoke_df[schema.building_id_column]),
        )
        artifacts.append(
            FinalModelArtifact(
                candidate_id=candidate_id,
                approach=selection.approach,
                model=model,
                evidence=evidence,
            )
        )
    return tuple(artifacts)


def _final_prediction_records(
    candidate: CandidateDefinition, prediction: PredictionResult
) -> list[dict[str, Any]]:
    """One tidy prediction row per holdout building for one final candidate."""
    records: list[dict[str, Any]] = []
    for row_index, building_id in enumerate(prediction.building_ids):
        record: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "approach": candidate.approach,
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


def _test_metrics_summary(evaluations: Sequence[FinalCandidateEvaluation]) -> dict[str, Any]:
    """Build a compact, JSON-safe test-metric summary keyed by candidate ID."""
    summary: dict[str, Any] = {}
    for evaluation in evaluations:
        records = evaluation.metrics_df.to_dict(orient="records")
        summary[evaluation.candidate_id] = {
            f"{record['metric_name']}:{record['target']}:{record['aggregation_level']}": (
                float(record["value"])
            )
            for record in records
        }
    return summary


def evaluate_frozen_models_on_lockbox(
    modeling_table: pd.DataFrame,
    *,
    split_manifest: SplitManifest,
    pretest_freeze: SelectionFreeze,
    refit_result: Sequence[FinalModelArtifact],
    candidates: Sequence[CandidateDefinition],
    evaluation_config: EvaluationConfig | None,
    schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
) -> FinalEvaluationResult:
    """Replay the manifest, score all three refit winners once, and finalize the freeze.

    This is the only Gate 8 API that receives the full modeling table and may
    materialize the holdout partition (decision 3.4). It refuses a freeze
    without a recorded cross-family decision and refuses one whose test
    evidence is already populated (section 8.3, "one-time behavior"): a
    failed call can always be retried from the same ``pretest_freeze``,
    because nothing here mutates it before every candidate has finished.
    """
    if pretest_freeze.cross_family_selection is None:
        raise ValueError(
            "Lockbox evaluation requires a pretest freeze with a recorded "
            "cross-family selection"
        )
    if pretest_freeze.test_metrics is not None:
        raise ValueError(
            "This freeze already carries test metrics; a completed evaluation "
            "cannot be repeated"
        )
    schema.validate_table(modeling_table)
    manifest_fingerprint = _manifest_fingerprint(split_manifest)
    if manifest_fingerprint != pretest_freeze.manifest_fingerprint:
        raise ValueError("Split manifest does not match the pretest freeze")
    if manifest_fingerprint != pretest_freeze.cross_family_selection.manifest_fingerprint:
        raise ValueError("Cross-family selection does not match this split manifest")

    approaches = {selection.approach for selection in pretest_freeze.selections}
    if approaches != _REQUIRED_APPROACHES:
        raise ValueError("Pretest freeze must record exactly one winner per approach")
    selected_ids = {selection.selected_candidate_id for selection in pretest_freeze.selections}
    artifact_by_id = {artifact.candidate_id: artifact for artifact in refit_result}
    if set(artifact_by_id) != selected_ids:
        raise ValueError("refit_result must contain exactly the frozen approach winners")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    missing_candidates = selected_ids.difference(candidate_by_id)
    if missing_candidates:
        raise ValueError(
            f"Supplied candidates are missing the frozen winners: {sorted(missing_candidates)}"
        )

    split = replay_split_manifest(
        modeling_table, split_manifest, config=_replay_config(schema=schema)
    )

    training_ids = _sorted_ids(set(split.train_df[schema.building_id_column]))
    if training_ids != pretest_freeze.outer_training_building_ids:
        raise ValueError("Replayed training partition does not match the frozen CV evidence")
    training_data_hash = table_hash(split.train_df, id_column=schema.building_id_column)
    training_schema_hash = column_schema_hash(split.train_df)
    for artifact in refit_result:
        if artifact.evidence.manifest_fingerprint != manifest_fingerprint:
            raise ValueError(
                f"Candidate {artifact.candidate_id!r} refit evidence manifest "
                "fingerprint does not match this replay"
            )
        if artifact.evidence.training_data_hash != training_data_hash:
            raise ValueError(
                f"Candidate {artifact.candidate_id!r} refit evidence does not match "
                "the replayed training data"
            )
        if artifact.evidence.training_schema_hash != training_schema_hash:
            raise ValueError(
                f"Candidate {artifact.candidate_id!r} refit evidence does not match "
                "the replayed training schema"
            )

    holdout_ids = _sorted_ids(set(split.test_df[schema.building_id_column]))
    # Which rows get scored is checked against the manifest itself, never
    # against a value derived from the frame about to be predicted: a check
    # built from that frame would agree with whatever rows it happened to hold.
    if holdout_ids != _sorted_ids(set(split_manifest.holdout_building_ids)):
        raise ValueError("Replayed holdout partition does not match the split manifest")
    master_seed = pretest_freeze.master_seed
    selected_candidate_id = pretest_freeze.cross_family_selection.selected_candidate_id

    evaluations: list[FinalCandidateEvaluation] = []
    reference_ids: np.ndarray | None = None
    for selection in sorted(pretest_freeze.selections, key=lambda selection: selection.approach):
        candidate_id = selection.selected_candidate_id
        candidate = candidate_by_id[candidate_id]
        artifact = artifact_by_id[candidate_id]
        seeds = {
            "predict": _stable_seed(master_seed, f"final/{candidate_id}/predict"),
            "evaluate": _stable_seed(master_seed, f"final/{candidate_id}/evaluate"),
            "bootstrap": _stable_seed(master_seed, f"final/{candidate_id}/bootstrap"),
        }
        with _final_failure_context(candidate_id, "predict"):
            prediction = artifact.model.predict(
                split.test_df,
                prediction_config=candidate.prediction_config,
                rng=np.random.default_rng(seeds["predict"]),
            )
            _assert_prediction_alignment(
                prediction, split.test_df, id_column=schema.building_id_column
            )
        if reference_ids is None:
            reference_ids = prediction.building_ids
        elif not np.array_equal(prediction.building_ids, reference_ids):
            raise ValueError(
                "All three final models must receive identical ordered holdout IDs"
            )
        with _final_failure_context(candidate_id, "evaluate"):
            evaluation = evaluate_predictions(
                split.test_df,
                prediction,
                candidate.metrics,
                rng=np.random.default_rng(seeds["evaluate"]),
            )
        intervals_df = pd.DataFrame()
        if evaluation_config is not None:
            with _final_failure_context(candidate_id, "bootstrap"):
                bootstrap = neighborhood_cluster_bootstrap(
                    split.test_df,
                    prediction,
                    candidate.metrics,
                    config=evaluation_config,
                    default_seed=seeds["bootstrap"],
                )
            intervals_df = bootstrap.intervals_df.assign(
                candidate_id=candidate_id, approach=candidate.approach
            )
        role: Literal["selected", "comparator"] = (
            "selected" if candidate_id == selected_candidate_id else "comparator"
        )
        evaluations.append(
            FinalCandidateEvaluation(
                candidate_id=candidate_id,
                approach=candidate.approach,
                role=role,
                seeds=seeds,
                metrics_df=evaluation.metrics_df.assign(
                    candidate_id=candidate_id, approach=candidate.approach
                ),
                intervals_df=intervals_df,
                predictions_df=pd.DataFrame(
                    _final_prediction_records(candidate, prediction)
                ),
                metric_comparability=selection.policy.likelihood_comparability,
                model_metadata=artifact.evidence.model_metadata,
                evaluation_metadata=dict(evaluation.metadata),
            )
        )

    finalized_freeze = pretest_freeze.with_test_metrics(_test_metrics_summary(evaluations))
    target_free_columns = [*schema.identifier_columns, *schema.feature_columns]
    holdout_input_df = split.test_df.loc[:, target_free_columns].reset_index(drop=True)
    return FinalEvaluationResult(
        manifest_fingerprint=manifest_fingerprint,
        holdout_building_ids=holdout_ids,
        evaluations=tuple(evaluations),
        finalized_freeze=finalized_freeze,
        holdout_input_df=holdout_input_df,
        schema=schema,
    )
