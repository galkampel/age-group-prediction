"""Canonical, production candidate registry for cross-model validation.

Gate 6/7 tests each wired their own one-candidate-per-approach comparison by
hand (see ``tests/validation/test_experiment_real_models.py::_real_candidates``
for the pattern this module promotes). That duplication meant every caller,
including a future notebook, could silently drift from the wiring that was
actually validated. This module is the single place real candidates,
policies, and full-training refit factories are declared, so a Gate 8 caller
consumes the same objects Gate 6/7 already exercised.

Nothing here fits a model, runs a fold, or picks a winner: it only declares
the ordered candidate set that ``run_cross_model_validation`` and the Gate 8
final refit consume.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from ..experiment_config import ExperimentConfig
from ..metrics import Metric, default_metric_set
from ..modeling_config import DEFAULT_MODELING_SCHEMA, FeatureSpec
from ..models.base import BaseAgeGroupModel
from ..models.bayesian_conditional import BayesianConditionalModel
from ..models.direct_cohort import DirectCohortModel
from ..models.independent_total_probability import IndependentTotalProbabilityModel
from .candidates import enumerate_feature_specs
from .contracts import CandidateDefinition, SelectionPolicy
from .policies import direct_cohort_selection_policy, sequential_joint_selection_policy

__all__ = [
    "CandidateRegistry",
    "build_canonical_candidate_registry",
]

# The predeclared, unmodified SES form is always enumerated first (section
# 5.3), so indexing it by name rather than position keeps the registry
# readable and immune to a reordering of `enumerate_feature_specs`.
_TREE_SPEC_NAME = "tree__ses_linear"
_TOTAL_SPEC_NAME = "total_count__ses_linear"
_PROBABILITY_SPEC_NAME = "composition__ses_linear"

CANDIDATE_SET_NAME = "gate8-canonical-v1"


def _named_spec(component: str, name: str) -> FeatureSpec:
    """Return one enumerated feature spec by its stable predeclared name."""
    candidates = {
        candidate.name: candidate.spec
        for candidate in enumerate_feature_specs(component)
    }
    if name not in candidates:
        raise ValueError(
            f"Unknown predeclared spec {name!r} for component {component!r}; "
            f"available: {sorted(candidates)}"
        )
    return candidates[name]


@dataclass(frozen=True)
class CandidateRegistry:
    """The ordered canonical candidates, their policies, and refit factories.

    ``final_refit_factories`` builds one fresh, unfitted model per candidate
    for the Gate 8 full-training refit. It is keyed by ``candidate_id`` and
    differs from each candidate's own ``model_factory`` only for the Bayesian
    candidate, whose CV factory uses the configured reduced NUTS profile while
    its final factory is forced to the full profile and strict diagnostics
    (plan section 3, decision 3.1; handoff section 6.1).
    """

    candidates: tuple[CandidateDefinition, ...]
    selection_policies: tuple[SelectionPolicy, ...]
    final_refit_factories: Mapping[str, Callable[[], BaseAgeGroupModel]]
    candidate_set_name: str
    candidate_set_fingerprint: str

    def __post_init__(self) -> None:
        """Validate candidate/factory alignment and freeze the factory mapping."""
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate registry IDs must be unique")
        factories = dict(self.final_refit_factories)
        if set(factories) != set(candidate_ids):
            raise ValueError(
                "Final refit factories must cover exactly the registered candidates"
            )
        object.__setattr__(self, "final_refit_factories", MappingProxyType(factories))

    def candidate_by_id(self, candidate_id: str) -> CandidateDefinition:
        """Look up one registered candidate by its stable ID."""
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(f"Unknown candidate ID: {candidate_id!r}")


def _fingerprint(candidates: tuple[CandidateDefinition, ...]) -> str:
    """SHA-256 of the ordered candidate descriptors' canonical JSON.

    Provenance for the candidate *set*: a change to any candidate's approach,
    feature specs, metrics, or configuration changes this fingerprint, so a
    persisted decision or logged run can be checked against the registry that
    produced it.
    """
    payload = json.dumps(
        [candidate.to_descriptor() for candidate in candidates],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_canonical_candidate_registry(config: ExperimentConfig) -> CandidateRegistry:
    """Build the one-candidate-per-approach production registry.

    Reuses ``enumerate_feature_specs`` for the predeclared feature forms,
    ``default_metric_set`` for the metric declarations, and the shipped
    ``direct_cohort_selection_policy``/``sequential_joint_selection_policy``
    helpers, so this registry cannot silently diverge from the metrics and
    policies Gate 6 selection already validates against.
    """
    evaluation = config.evaluation
    validation = config.prediction_validation
    cohort_targets = tuple(DEFAULT_MODELING_SCHEMA.cohort_target_columns)
    total_target = DEFAULT_MODELING_SCHEMA.total_target_column

    tree_spec = _named_spec("tree", _TREE_SPEC_NAME)
    total_spec = _named_spec("total_count", _TOTAL_SPEC_NAME)
    probability_spec = _named_spec("composition", _PROBABILITY_SPEC_NAME)

    marginal_metrics = default_metric_set(
        cohort_targets,
        nll_source="parametric",
        nll_interpretation="independent direct-cohort likelihood",
        evaluation_config=evaluation,
        prediction_validation_config=validation,
    )

    def joint_metrics(interpretation: str) -> tuple[Metric, ...]:
        """Build the total-plus-composition joint metric set for one candidate."""
        return default_metric_set(
            (total_target, *cohort_targets),
            nll_source="pointwise",
            nll_interpretation=interpretation,
            evaluation_config=evaluation,
            prediction_validation_config=validation,
            joint_nll_interpretation=(
                "sequential joint total-plus-composition log mass"
            ),
        )

    conditional_prediction = replace(
        config.prediction, include_pointwise_log_probabilities=True
    )

    direct_cohort_config = config.direct_cohort
    independent_config = config.independent_total_probability
    reduced_bayesian_config = config.bayesian_conditional
    full_bayesian_config = replace(reduced_bayesian_config, active_profile="full")

    def direct_cohort_factory() -> DirectCohortModel:
        """Build a fresh, unfitted direct-cohort candidate model."""
        return DirectCohortModel(direct_cohort_config=direct_cohort_config)

    def independent_factory() -> IndependentTotalProbabilityModel:
        """Build a fresh, unfitted independent total/probability candidate model."""
        return IndependentTotalProbabilityModel(independent_config=independent_config)

    def bayesian_cv_factory() -> BayesianConditionalModel:
        """Build a fresh Bayesian candidate model under the configured CV profile."""
        return BayesianConditionalModel(bayesian_config=reduced_bayesian_config)

    def bayesian_final_factory() -> BayesianConditionalModel:
        """Build a fresh Bayesian candidate model forced to the full NUTS profile."""
        return BayesianConditionalModel(bayesian_config=full_bayesian_config)

    candidates = (
        CandidateDefinition(
            candidate_id="direct-poisson",
            approach="DirectCohortModel",
            model_factory=direct_cohort_factory,
            fit_feature_spec=tree_spec,
            component_feature_specs=(tree_spec,),
            metrics=marginal_metrics,
            configuration={"family": direct_cohort_config.family},
            prediction_config=config.prediction,
        ),
        CandidateDefinition(
            candidate_id="independent-nb2",
            approach="IndependentTotalProbabilityModel",
            model_factory=independent_factory,
            fit_feature_spec=total_spec,
            component_feature_specs=(total_spec, probability_spec),
            metrics=joint_metrics("NB2 total plus multinomial conditional composition"),
            configuration={"total_family": independent_config.total_family},
            prediction_config=conditional_prediction,
        ),
        CandidateDefinition(
            candidate_id="bayesian-reduced",
            approach="BayesianConditionalModel",
            model_factory=bayesian_cv_factory,
            fit_feature_spec=total_spec,
            component_feature_specs=(total_spec, probability_spec),
            metrics=joint_metrics(
                "posterior-integrated NB2 total plus Dirichlet-multinomial composition"
            ),
            configuration={"profile": reduced_bayesian_config.active_profile},
            prediction_config=conditional_prediction,
        ),
    )

    selection_policies = (
        direct_cohort_selection_policy(cohort_targets),
        sequential_joint_selection_policy("IndependentTotalProbabilityModel"),
        sequential_joint_selection_policy("BayesianConditionalModel"),
    )

    final_refit_factories: dict[str, Callable[[], BaseAgeGroupModel]] = {
        "direct-poisson": direct_cohort_factory,
        "independent-nb2": independent_factory,
        "bayesian-reduced": bayesian_final_factory,
    }

    return CandidateRegistry(
        candidates=candidates,
        selection_policies=selection_policies,
        final_refit_factories=final_refit_factories,
        candidate_set_name=CANDIDATE_SET_NAME,
        candidate_set_fingerprint=_fingerprint(candidates),
    )
