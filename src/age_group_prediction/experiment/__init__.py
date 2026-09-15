"""Model-agnostic cross-validation, selection, and interpretation workflows."""

from __future__ import annotations

from .artifacts import FoldArtifactEvidence, ReloadCheck
from .candidate_registry import CandidateRegistry, build_canonical_candidate_registry
from .candidates import FeatureSpecCandidate, enumerate_feature_specs
from .contracts import (
    ApproachName as ApproachName,
)
from .contracts import (
    CandidateDefinition,
    FeatureBlock,
    FoldIdentity,
    MetricReference,
    PermutationImportanceSpec,
    SelectionCriterion,
    SelectionPolicy,
)
from .contracts import (
    OptimizationDirection as OptimizationDirection,
)
from .contracts import (
    SelectionRole as SelectionRole,
)
from .evidence import (
    CrossValidationExperimentResult,
    ExperimentProvenance,
    FoldRunEvidence,
    FrozenApproachSelection,
    SelectionFreeze,
)
from .final_evaluation import (
    FinalCandidateEvaluation,
    FinalEvaluationResult,
    FinalModelArtifact,
    FinalRefitEvidence,
    evaluate_frozen_models_on_lockbox,
    refit_frozen_approach_winners,
)
from .final_selection import (
    CrossFamilySelection,
    CrossFamilySelectionRule,
    select_cross_family_winner,
)
from .partitions import validate_experiment_partitions
from .policies import (
    direct_cohort_selection_policy,
    sequential_joint_selection_policy,
)
from .runner import run_cross_model_validation

__all__ = [
    "CandidateDefinition",
    "CandidateRegistry",
    "CrossFamilySelection",
    "CrossFamilySelectionRule",
    "CrossValidationExperimentResult",
    "ExperimentProvenance",
    "FeatureBlock",
    "FeatureSpecCandidate",
    "FinalCandidateEvaluation",
    "FinalEvaluationResult",
    "FinalModelArtifact",
    "FinalRefitEvidence",
    "FoldArtifactEvidence",
    "FoldIdentity",
    "FoldRunEvidence",
    "FrozenApproachSelection",
    "MetricReference",
    "PermutationImportanceSpec",
    "ReloadCheck",
    "SelectionCriterion",
    "SelectionFreeze",
    "SelectionPolicy",
    "build_canonical_candidate_registry",
    "direct_cohort_selection_policy",
    "enumerate_feature_specs",
    "evaluate_frozen_models_on_lockbox",
    "refit_frozen_approach_winners",
    "run_cross_model_validation",
    "select_cross_family_winner",
    "sequential_joint_selection_policy",
    "validate_experiment_partitions",
]