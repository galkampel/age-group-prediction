"""Deterministic within-approach selection and likelihood comparability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd

from .contracts import CandidateDefinition, SelectionCriterion, SelectionPolicy
from .evidence import FoldRunEvidence, FrozenApproachSelection


def _convergence_failures(
    fold_runs: Sequence[FoldRunEvidence],
) -> dict[str, tuple[str, ...]]:
    """Collect per-candidate convergence failures recorded by fitted models.

    A model that records no ``policy_passed`` for a stage declares no
    convergence evidence and is therefore unconstrained; only an explicit
    ``False`` counts as a failure.
    """
    failures: dict[str, list[str]] = {}
    for run in fold_runs:
        model_metadata = run.model_metadata.get("model")
        if not isinstance(model_metadata, Mapping):
            continue
        diagnostics = model_metadata.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        for stage, payload in sorted(diagnostics.items()):
            if not isinstance(payload, Mapping) or "policy_passed" not in payload:
                continue
            if payload["policy_passed"]:
                continue
            reasons = tuple(payload.get("policy_failures") or ("unspecified",))
            failures.setdefault(run.candidate_id, []).extend(
                f"fold {run.fold_identity.fold_index} {stage}: {reason}"
                for reason in reasons
            )
    return {
        candidate_id: tuple(reasons) for candidate_id, reasons in failures.items()
    }


def _select_candidates(
    candidates: Mapping[str, CandidateDefinition],
    policies: Mapping[str, SelectionPolicy],
    aggregate_metrics: pd.DataFrame,
    convergence_failures: Mapping[str, tuple[str, ...]] | None = None,
    *,
    require_convergence: bool = True,
) -> tuple[FrozenApproachSelection, ...]:
    resolved_failures = dict(convergence_failures or {})
    selections: list[FrozenApproachSelection] = []
    for approach in sorted(policies):
        policy = policies[approach]
        declared = sorted(
            candidate.candidate_id
            for candidate in candidates.values()
            if candidate.approach == approach and candidate.selection_role == "eligible"
        )
        # Plan section 9.3 makes convergence an explicit selection constraint.
        # A candidate whose own diagnostics failed is excluded before ranking
        # rather than allowed to win on a score computed from an unconverged
        # posterior.
        unconverged = (
            [
                candidate_id
                for candidate_id in declared
                if candidate_id in resolved_failures
            ]
            if require_convergence
            else []
        )
        eligible = [
            candidate_id for candidate_id in declared if candidate_id not in unconverged
        ]
        if not eligible:
            detail = "; ".join(
                f"{candidate_id}: {', '.join(resolved_failures[candidate_id])}"
                for candidate_id in unconverged
            )
            raise ValueError(
                f"Every eligible {approach} candidate failed its convergence "
                f"policy, so none can be selected ({detail}). Fix convergence or "
                "pass require_convergence=False to rank them anyway."
            )
        criterion_values = {
            candidate_id: {
                criterion.name: _criterion_value(
                    aggregate_metrics, candidate_id, criterion
                )
                for criterion in policy.criteria
            }
            for candidate_id in eligible
        }
        ranked = sorted(
            eligible,
            key=lambda candidate_id: (
                *(
                    criterion_values[candidate_id][criterion.name]
                    if criterion.optimization_direction == "minimize"
                    else -criterion_values[candidate_id][criterion.name]
                    for criterion in policy.criteria
                ),
                candidate_id,
            ),
        )
        selected = ranked[0]
        outranked = [
            candidate_id for candidate_id in ranked if candidate_id != selected
        ]
        reasons = {
            candidate_id: (
                f"Ranked below '{selected}' under the declared ordered "
                "training-only selection criteria: "
                + _margin_summary(
                    criterion_values[candidate_id],
                    criterion_values[selected],
                    policy,
                )
            )
            for candidate_id in outranked
        }
        reasons.update(
            {
                candidate_id: (
                    "Excluded by the convergence selection constraint: "
                    + "; ".join(resolved_failures[candidate_id])
                )
                for candidate_id in unconverged
            }
        )
        selections.append(
            FrozenApproachSelection(
                approach=approach,
                selected_candidate_id=selected,
                rejected_candidate_ids=(*outranked, *unconverged),
                rejection_reasons=reasons,
                criterion_values=criterion_values,
                policy=policy,
            )
        )
    return tuple(selections)


def _margin_summary(
    candidate_values: Mapping[str, float],
    selected_values: Mapping[str, float],
    policy: SelectionPolicy,
) -> str:
    """Describe the first criterion that separated a candidate from the winner.

    A bare "ranked below" reason reads identically whether a candidate lost by
    three nats or by 1e-12, so the deciding criterion and its margin are named.
    """
    for criterion in policy.criteria:
        candidate_value = candidate_values[criterion.name]
        selected_value = selected_values[criterion.name]
        if candidate_value == selected_value:
            continue
        margin = abs(candidate_value - selected_value)
        return (
            f"{criterion.name} {candidate_value:.12g} versus "
            f"{selected_value:.12g} (margin {margin:.3g}, "
            f"{criterion.optimization_direction})"
        )
    return "tied on every declared criterion; ordered by candidate id"


_DISCRETE_FAMILIES = {"poisson", "nb2"}
_CONTINUOUS_FAMILIES = {"normal"}
_DISTRIBUTIONAL_CAPABILITIES = {
    "parametric_distributions",
    "pointwise_log_probabilities",
    "joint_pointwise_log_probabilities",
}


def _candidate_measure_kinds(
    candidate_id: str, fold_runs: Sequence[FoldRunEvidence]
) -> set[str]:
    """Classify a candidate's declared predictive families as mass or density."""
    kinds: set[str] = set()
    for run in fold_runs:
        if run.candidate_id != candidate_id:
            continue
        for spec in (run.prediction.parametric_distributions or {}).values():
            if spec.family in _DISCRETE_FAMILIES:
                kinds.add("discrete")
            elif spec.family in _CONTINUOUS_FAMILIES:
                kinds.add("continuous")
    return kinds


def _validate_likelihood_comparability(
    candidates: Mapping[str, CandidateDefinition],
    policies: Mapping[str, SelectionPolicy],
    fold_runs: Sequence[FoldRunEvidence],
) -> None:
    """Reject ranking continuous log densities against discrete log masses.

    Section 9.1 forbids ranking unlike likelihood definitions. A Normal
    direct-cohort candidate scores log densities while Poisson and NB2 candidates
    score log masses, so a distributional criterion cannot order them. Such a
    candidate belongs in the run as a ``diagnostic_comparator``.
    """
    for approach, policy in policies.items():
        eligible = sorted(
            candidate_id
            for candidate_id, candidate in candidates.items()
            if candidate.approach == approach
            and candidate.selection_role == "eligible"
        )
        uses_distributional_criterion = any(
            metric.required_capability in _DISTRIBUTIONAL_CAPABILITIES
            for candidate_id in eligible
            for metric in candidates[candidate_id].metrics
            for criterion in policy.criteria
            for reference in criterion.metric_references
            if (metric.name, metric.target, metric.aggregation_level)
            == (reference.metric_name, reference.target, reference.aggregation_level)
        )
        if not uses_distributional_criterion:
            continue
        kinds_by_candidate = {
            candidate_id: _candidate_measure_kinds(candidate_id, fold_runs)
            for candidate_id in eligible
        }
        observed = set().union(*kinds_by_candidate.values()) if eligible else set()
        if len(observed) > 1:
            detail = ", ".join(
                f"{candidate_id}={sorted(kinds) or ['unspecified']}"
                for candidate_id, kinds in sorted(kinds_by_candidate.items())
            )
            raise ValueError(
                f"Approach '{approach}' ranks continuous and discrete candidates "
                f"under a distributional criterion ({detail}); log densities are "
                "not comparable to log masses. Declare the continuous candidate "
                "with selection_role='diagnostic_comparator'."
            )


def _criterion_value(
    aggregate_metrics: pd.DataFrame,
    candidate_id: str,
    criterion: SelectionCriterion,
) -> float:
    values: list[float] = []
    for reference in criterion.metric_references:
        rows = aggregate_metrics.loc[
            (aggregate_metrics["candidate_id"] == candidate_id)
            & (aggregate_metrics["metric_name"] == reference.metric_name)
            & (aggregate_metrics["target"] == reference.target)
            & (aggregate_metrics["aggregation_level"] == reference.aggregation_level),
            "mean",
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Candidate '{candidate_id}' does not provide exactly one "
                f"{reference.metric_name}:{reference.target}:"
                f"{reference.aggregation_level} aggregate"
            )
        values.append(float(rows.iloc[0]))
    reducer = np.mean if criterion.reduction == "mean" else np.sum
    return float(reducer(values))


def _validate_bayesian_feature_freeze(
    candidates: Mapping[str, CandidateDefinition],
    selections: Sequence[FrozenApproachSelection],
) -> None:
    selected = {
        selection.approach: candidates[selection.selected_candidate_id]
        for selection in selections
    }
    independent = selected.get("IndependentTotalProbabilityModel")
    bayesian = selected.get("BayesianConditionalModel")
    if independent is None or bayesian is None:
        return
    independent_specs = {
        spec.component: asdict(spec) for spec in independent.component_feature_specs
    }
    bayesian_specs = {
        spec.component: asdict(spec) for spec in bayesian.component_feature_specs
    }
    for component in ("total_count", "composition"):
        if independent_specs.get(component) != bayesian_specs.get(component):
            raise ValueError(
                "BayesianConditionalModel must consume the selected independent "
                f"model's frozen {component} feature spec"
            )
