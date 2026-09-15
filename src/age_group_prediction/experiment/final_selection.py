"""Deterministic cross-family selection, frozen before the lockbox opens.

Gate 6 freezes one winner *per approach*: `FrozenApproachSelection` is
within-approach by construction, so a validated comparison ends with three
winners, not one. This module makes the remaining choice across approaches,
reading only training/CV evidence that Gate 6/7 already produced, and reads no
test frame at all. Plan section 11 requires that choice to be recorded before
any holdout row is materialized, so nothing here is optional plumbing: the
:class:`CrossFamilySelection` a caller obtains here is what
`experiment.evidence.SelectionFreeze.with_cross_family_selection` accepts, and
that freeze is what the Gate 8 final refit/evaluation must be given before it
may see a holdout building.

The rule (plan section "Gate 8", decision 3.1 of the session handoff):

1. Rank the selected `IndependentTotalProbabilityModel` against the selected
   `BayesianConditionalModel` by minimum ``joint_predictive_nll``. These two
   score the same sequential-joint object; `DirectCohortModel` does not
   produce it and is excluded from this stage by construction.
2. Compare that conditional winner with the selected `DirectCohortModel`
   lexicographically by minimum composition log loss, then minimum mean
   cohort RMSE, then minimum mean cohort MAE, then candidate ID.
3. Cohort RMSE and MAE are reduced over the cohort targets the selected
   `DirectCohortModel` candidate itself declares, never the total: Model A's
   own declared metric set has no total-scoped entry, so the cohort list is
   read from its descriptor rather than assumed.
4. `predictive_nll` is never read: Model A's per-cohort keys are marginal log
   masses while Model B's and the Bayesian model's are a schema-ordered
   conditional decomposition, so a per-target table would hand the
   conditional models their near-zero last key as free winnings.
5. No test evidence, bootstrap interval width, or runtime ever enters this
   module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..modeling_config import DEFAULT_MODELING_SCHEMA
from .contracts import ApproachName, MetricReference
from .evidence import CrossValidationExperimentResult

__all__ = [
    "CROSS_FAMILY_RULE_NAME",
    "CROSS_FAMILY_RULE_VERSION",
    "CrossFamilySelection",
    "CrossFamilySelectionRule",
    "select_cross_family_winner",
]

CROSS_FAMILY_RULE_NAME = "composition-first-lexicographic"
CROSS_FAMILY_RULE_VERSION = "1.0"

_CONDITIONAL_APPROACHES: tuple[ApproachName, ApproachName] = (
    "IndependentTotalProbabilityModel",
    "BayesianConditionalModel",
)
_JOINT_NLL_REFERENCE = MetricReference("joint_predictive_nll", "joint", "building")
_COMPOSITION_REFERENCE = MetricReference("composition_log_loss", "composition", "child")

_LIKELIHOOD_COMPARABILITY = (
    "joint_predictive_nll scores the same sequential-joint total-plus-"
    "composition object for the independent and Bayesian conditional "
    "candidates only; DirectCohortModel declares no such object and is "
    "excluded from that comparison by capability, not by a special case. "
    "composition_log_loss and cohort RMSE/MAE are the same quantity for all "
    "three approaches and carry the DirectCohortModel comparison. Per-target "
    "predictive_nll is never read: DirectCohortModel's cohort keys are "
    "marginal log masses while the conditional approaches' are a "
    "schema-ordered conditional decomposition whose last key is ~0 by "
    "construction."
)


@dataclass(frozen=True)
class CrossFamilySelectionRule:
    """Versioned, ordered description of the cross-family selection rule."""

    name: str
    version: str
    conditional_criterion: str
    final_criteria: tuple[str, ...]
    reductions: Mapping[str, str]
    likelihood_comparability: str

    def __post_init__(self) -> None:
        """Validate the declared criteria are non-empty and freeze reductions."""
        if not self.name or not self.version:
            raise ValueError("Cross-family rule requires a name and version")
        if not self.conditional_criterion or not self.final_criteria:
            raise ValueError("Cross-family rule requires its ordered criteria")
        if not self.likelihood_comparability:
            raise ValueError("Cross-family rule requires a comparability explanation")
        reductions = dict(self.reductions)
        json.dumps(reductions)
        object.__setattr__(self, "reductions", MappingProxyType(reductions))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe rule evidence."""
        return {
            "name": self.name,
            "version": self.version,
            "conditional_criterion": self.conditional_criterion,
            "final_criteria": list(self.final_criteria),
            "reductions": dict(self.reductions),
            "likelihood_comparability": self.likelihood_comparability,
        }


@dataclass(frozen=True)
class CrossFamilySelection:
    """Immutable, JSON-safe record of one cross-family decision.

    Records every value the rule compared, not only the winner: an auditor
    must be able to recompute the decision from this object and the rule
    alone, without rerunning the comparison.
    """

    rule: CrossFamilySelectionRule
    manifest_fingerprint: str
    compared_candidate_ids: tuple[str, ...]
    candidate_approaches: Mapping[str, ApproachName]
    criterion_metric_references: Mapping[str, tuple[Mapping[str, str], ...]]
    criterion_values: Mapping[str, Mapping[str, float]]
    conditional_winner_candidate_id: str
    conditional_tie_break_used: bool
    selected_candidate_id: str
    selected_approach: ApproachName
    decisive_final_criterion: str | None
    final_tie_break_used: bool

    def __post_init__(self) -> None:
        """Check every mapping is JSON-safe and freeze read-only copies."""
        approaches = dict(self.candidate_approaches)
        references = {
            criterion: tuple(dict(reference) for reference in refs)
            for criterion, refs in self.criterion_metric_references.items()
        }
        values = {
            candidate_id: dict(criteria)
            for candidate_id, criteria in self.criterion_values.items()
        }
        json.dumps(
            {
                "candidate_approaches": approaches,
                "criterion_metric_references": {
                    key: [dict(entry) for entry in refs]
                    for key, refs in references.items()
                },
                "criterion_values": values,
            }
        )
        object.__setattr__(self, "candidate_approaches", MappingProxyType(approaches))
        object.__setattr__(
            self, "criterion_metric_references", MappingProxyType(references)
        )
        object.__setattr__(self, "criterion_values", MappingProxyType(values))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe cross-family selection evidence."""
        result = {
            "rule": self.rule.to_dict(),
            "manifest_fingerprint": self.manifest_fingerprint,
            "compared_candidate_ids": list(self.compared_candidate_ids),
            "candidate_approaches": dict(self.candidate_approaches),
            "criterion_metric_references": {
                criterion: [dict(reference) for reference in refs]
                for criterion, refs in self.criterion_metric_references.items()
            },
            "criterion_values": {
                candidate_id: dict(criteria)
                for candidate_id, criteria in self.criterion_values.items()
            },
            "conditional_winner_candidate_id": self.conditional_winner_candidate_id,
            "conditional_tie_break_used": self.conditional_tie_break_used,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_approach": self.selected_approach,
            "decisive_final_criterion": self.decisive_final_criterion,
            "final_tie_break_used": self.final_tie_break_used,
        }
        json.dumps(result)
        return result


def _descriptor_metric_keys(descriptor: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    """Return the (name, target, aggregation_level) keys one candidate declares."""
    return {
        (metric["name"], metric["target"], metric["aggregation_level"])
        for metric in descriptor["metrics"]
    }


def _require_declared(
    descriptor: Mapping[str, Any], reference: MetricReference, candidate_id: str
) -> None:
    """Refuse to read a metric a candidate never declared it would produce."""
    key = (reference.metric_name, reference.target, reference.aggregation_level)
    if key not in _descriptor_metric_keys(descriptor):
        raise ValueError(
            f"Candidate {candidate_id!r} does not declare metric {key}; the "
            "cross-family rule cannot read an undeclared criterion"
        )


def _metric_mean(
    aggregate_metrics: pd.DataFrame, candidate_id: str, reference: MetricReference
) -> float:
    """Read exactly one finite aggregate metric mean for one candidate."""
    rows = aggregate_metrics.loc[
        (aggregate_metrics["candidate_id"] == candidate_id)
        & (aggregate_metrics["metric_name"] == reference.metric_name)
        & (aggregate_metrics["target"] == reference.target)
        & (aggregate_metrics["aggregation_level"] == reference.aggregation_level),
        "mean",
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one {reference.metric_name}:{reference.target}:"
            f"{reference.aggregation_level} aggregate for candidate "
            f"{candidate_id!r}; found {len(rows)}"
        )
    value = float(rows.iloc[0])
    if not np.isfinite(value):
        raise ValueError(
            f"Candidate {candidate_id!r} metric {reference.metric_name}:"
            f"{reference.target} is not finite"
        )
    return value


def _cohort_references(
    direct_descriptor: Mapping[str, Any], metric_name: str
) -> tuple[MetricReference, ...]:
    """Read the building-level cohort references DirectCohortModel declares.

    DirectCohortModel's canonical metric set never includes a total-scoped
    entry, so reading the cohort list from its descriptor rather than
    hardcoding one keeps this rule tied to what was actually validated.
    ``total_target_column`` is still excluded explicitly: a future
    DirectCohortModel candidate that also declares a diagnostic total-scoped
    RMSE/MAE (as ``sequential_joint_selection_policy`` does for the
    conditional approaches) must not silently fold the total into a "cohort"
    reduction.
    """
    references = sorted(
        (
            MetricReference(metric["name"], metric["target"], metric["aggregation_level"])
            for metric in direct_descriptor["metrics"]
            if metric["name"] == metric_name
            and metric["aggregation_level"] == "building"
            and metric["target"] != DEFAULT_MODELING_SCHEMA.total_target_column
        ),
        key=lambda reference: reference.target,
    )
    if not references:
        raise ValueError(
            f"DirectCohortModel candidate declares no building-level {metric_name!r} "
            "cohort metrics; the cross-family rule has no cohort target to reduce over"
        )
    return tuple(references)


def _mean_over_references(
    aggregate_metrics: pd.DataFrame,
    candidate_id: str,
    descriptor: Mapping[str, Any],
    references: Sequence[MetricReference],
) -> float:
    """Average one candidate's per-cohort metric means, after checking declaration."""
    values = []
    for reference in references:
        _require_declared(descriptor, reference, candidate_id)
        values.append(_metric_mean(aggregate_metrics, candidate_id, reference))
    return float(np.mean(values))


def select_cross_family_winner(
    cv_result: CrossValidationExperimentResult,
) -> CrossFamilySelection:
    """Choose one cross-family winner from Gate 6/7 training-and-CV evidence.

    Reads only ``cv_result.aggregate_metrics_df``, the three
    ``FrozenApproachSelection`` winners, their candidate descriptors, and the
    manifest fingerprint; it accepts no test frame and computes nothing a test
    prediction could influence.
    """
    selections_by_approach = {
        selection.approach: selection for selection in cv_result.selections
    }
    required_approaches = {"DirectCohortModel", *_CONDITIONAL_APPROACHES}
    missing_approaches = required_approaches.difference(selections_by_approach)
    if missing_approaches:
        raise ValueError(
            f"Cross-family selection requires one winner per approach; missing "
            f"{sorted(missing_approaches)}"
        )

    descriptors_by_id = {
        descriptor["candidate_id"]: descriptor
        for descriptor in cv_result.freeze.candidate_descriptors
    }
    candidate_ids = {
        approach: selections_by_approach[approach].selected_candidate_id
        for approach in required_approaches
    }
    unknown_ids = set(candidate_ids.values()).difference(descriptors_by_id)
    if unknown_ids:
        raise ValueError(
            f"Cross-family selection cannot resolve candidate descriptors for "
            f"{sorted(unknown_ids)}"
        )

    direct_id = candidate_ids["DirectCohortModel"]
    independent_id = candidate_ids["IndependentTotalProbabilityModel"]
    bayesian_id = candidate_ids["BayesianConditionalModel"]
    direct_descriptor = descriptors_by_id[direct_id]
    aggregate_metrics = cv_result.aggregate_metrics_df

    # Stage 1: rank the two conditional candidates on the joint score only.
    for candidate_id in (independent_id, bayesian_id):
        _require_declared(
            descriptors_by_id[candidate_id], _JOINT_NLL_REFERENCE, candidate_id
        )
    joint_values = {
        candidate_id: _metric_mean(aggregate_metrics, candidate_id, _JOINT_NLL_REFERENCE)
        for candidate_id in (independent_id, bayesian_id)
    }
    conditional_ranked = sorted(
        (independent_id, bayesian_id),
        key=lambda candidate_id: (joint_values[candidate_id], candidate_id),
    )
    conditional_winner_id = conditional_ranked[0]
    conditional_tie_break_used = (
        joint_values[independent_id] == joint_values[bayesian_id]
    )

    # Stage 2: compare the conditional winner against DirectCohortModel by
    # composition log loss, then mean cohort RMSE, then mean cohort MAE, then
    # candidate ID, in that order.
    rmse_references = _cohort_references(direct_descriptor, "rmse")
    mae_references = _cohort_references(direct_descriptor, "mae")
    final_candidates = (conditional_winner_id, direct_id)
    criterion_values: dict[str, dict[str, float]] = {
        candidate_id: {} for candidate_id in (independent_id, bayesian_id, direct_id)
    }
    criterion_values[independent_id]["joint_predictive_nll"] = joint_values[
        independent_id
    ]
    criterion_values[bayesian_id]["joint_predictive_nll"] = joint_values[bayesian_id]
    for candidate_id in final_candidates:
        descriptor = descriptors_by_id[candidate_id]
        _require_declared(descriptor, _COMPOSITION_REFERENCE, candidate_id)
        criterion_values[candidate_id]["composition_log_loss"] = _metric_mean(
            aggregate_metrics, candidate_id, _COMPOSITION_REFERENCE
        )
        criterion_values[candidate_id]["mean_cohort_rmse"] = _mean_over_references(
            aggregate_metrics, candidate_id, descriptor, rmse_references
        )
        criterion_values[candidate_id]["mean_cohort_mae"] = _mean_over_references(
            aggregate_metrics, candidate_id, descriptor, mae_references
        )

    ordered_final_criteria = ("composition_log_loss", "mean_cohort_rmse", "mean_cohort_mae")
    ranked_final = sorted(
        final_candidates,
        key=lambda candidate_id: (
            *(criterion_values[candidate_id][name] for name in ordered_final_criteria),
            candidate_id,
        ),
    )
    selected_id = ranked_final[0]
    decisive_final_criterion: str | None = None
    for name in ordered_final_criteria:
        if (
            criterion_values[ranked_final[0]][name]
            != criterion_values[ranked_final[1]][name]
        ):
            decisive_final_criterion = name
            break
    final_tie_break_used = decisive_final_criterion is None

    rule = CrossFamilySelectionRule(
        name=CROSS_FAMILY_RULE_NAME,
        version=CROSS_FAMILY_RULE_VERSION,
        conditional_criterion="joint_predictive_nll",
        final_criteria=(*ordered_final_criteria, "candidate_id"),
        reductions={
            "joint_predictive_nll": "single_building_level_value",
            "composition_log_loss": "single_child_level_value",
            "mean_cohort_rmse": "mean_over_declared_cohort_targets",
            "mean_cohort_mae": "mean_over_declared_cohort_targets",
            "candidate_id": "lexicographic_tie_break",
        },
        likelihood_comparability=_LIKELIHOOD_COMPARABILITY,
    )
    candidate_approaches = {
        direct_id: "DirectCohortModel",
        independent_id: "IndependentTotalProbabilityModel",
        bayesian_id: "BayesianConditionalModel",
    }
    criterion_metric_references = {
        "joint_predictive_nll": (asdict(_JOINT_NLL_REFERENCE),),
        "composition_log_loss": (asdict(_COMPOSITION_REFERENCE),),
        "mean_cohort_rmse": tuple(asdict(reference) for reference in rmse_references),
        "mean_cohort_mae": tuple(asdict(reference) for reference in mae_references),
    }

    return CrossFamilySelection(
        rule=rule,
        manifest_fingerprint=cv_result.freeze.manifest_fingerprint,
        compared_candidate_ids=(direct_id, independent_id, bayesian_id),
        candidate_approaches=candidate_approaches,
        criterion_metric_references=criterion_metric_references,
        criterion_values=criterion_values,
        conditional_winner_candidate_id=conditional_winner_id,
        conditional_tie_break_used=conditional_tie_break_used,
        selected_candidate_id=selected_id,
        selected_approach=candidate_approaches[selected_id],
        decisive_final_criterion=decisive_final_criterion,
        final_tie_break_used=final_tie_break_used,
    )
