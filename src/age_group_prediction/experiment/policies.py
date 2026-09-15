"""Predeclared, approach-specific selection policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ..metrics import (
    CompositionLogLoss,
    JointPredictiveNegativeLogLikelihood,
    ParametricPredictiveNegativeLogLikelihood,
    RootMeanSquaredError,
)
from ..modeling_config import DEFAULT_MODELING_SCHEMA
from .contracts import MetricReference, SelectionCriterion, SelectionPolicy


def direct_cohort_selection_policy(
    cohort_names: Sequence[str],
) -> SelectionPolicy:
    """Select discrete direct-cohort candidates without double-counting totals."""
    resolved_cohorts = tuple(cohort_names)
    if not resolved_cohorts or len(resolved_cohorts) != len(set(resolved_cohorts)):
        raise ValueError("Direct-cohort selection requires unique cohort names")
    return SelectionPolicy(
        approach="DirectCohortModel",
        criteria=(
            SelectionCriterion(
                name="independent_cohort_joint_nll",
                metric_references=tuple(
                    MetricReference.from_metric(
                        ParametricPredictiveNegativeLogLikelihood(
                            target=cohort,
                            interpretation="independent direct-cohort likelihood",
                        )
                    )
                    for cohort in resolved_cohorts
                ),
                optimization_direction="minimize",
                reduction="sum",
            ),
            SelectionCriterion(
                name="mean_cohort_rmse",
                metric_references=tuple(
                    MetricReference.from_metric(RootMeanSquaredError(target=cohort))
                    for cohort in resolved_cohorts
                ),
                optimization_direction="minimize",
            ),
        ),
        likelihood_comparability=(
            "Sum of independent cohort log masses within discrete direct-cohort "
            "candidates; the separately scored total is excluded to avoid double "
            "counting and is not comparable to sequential conditional keys."
        ),
    )


def sequential_joint_selection_policy(
    approach: Literal[
        "IndependentTotalProbabilityModel", "BayesianConditionalModel"
    ],
) -> SelectionPolicy:
    """Select conditional candidates by their posterior/predictive joint score."""
    if approach not in {
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    }:
        raise ValueError("Sequential-joint selection requires a conditional approach")
    return SelectionPolicy(
        approach=approach,
        criteria=(
            SelectionCriterion(
                name="joint_predictive_nll",
                metric_references=(
                    MetricReference.from_metric(
                        JointPredictiveNegativeLogLikelihood(
                            interpretation=(
                                "sequential joint total-plus-composition log mass"
                            )
                        )
                    ),
                ),
                optimization_direction="minimize",
            ),
            SelectionCriterion(
                name="composition_log_loss",
                metric_references=(
                    MetricReference.from_metric(CompositionLogLoss()),
                ),
                optimization_direction="minimize",
            ),
            SelectionCriterion(
                name="total_rmse",
                metric_references=(
                    MetricReference.from_metric(
                        RootMeanSquaredError(
                            target=DEFAULT_MODELING_SCHEMA.total_target_column
                        )
                    ),
                ),
                optimization_direction="minimize",
            ),
        ),
        likelihood_comparability=(
            "Joint total-plus-composition log mass with sequential_joint pointwise "
            "scope; comparable between independent total/probability and Bayesian "
            "conditional candidates, with interpretation retained in metric metadata."
        ),
    )
