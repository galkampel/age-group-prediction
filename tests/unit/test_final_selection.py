"""Tests for the Gate 8 cross-family selection rule and freeze transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from age_group_prediction.experiment import (
    CandidateDefinition,
    CrossValidationExperimentResult,
    ExperimentProvenance,
    FoldIdentity,
    FrozenApproachSelection,
    MetricReference,
    SelectionCriterion,
    SelectionFreeze,
    SelectionPolicy,
    select_cross_family_winner,
)
from age_group_prediction.metrics import (
    CompositionLogLoss,
    JointPredictiveNegativeLogLikelihood,
    MeanAbsoluteError,
    ParametricPredictiveNegativeLogLikelihood,
    RootMeanSquaredError,
)
from age_group_prediction.modeling_config import (
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
)

COHORTS = ("n_kindergarten", "n_elementary", "n_highschool")
TOTAL = "n_children_total"
MANIFEST_FINGERPRINT = "manifest-fp-1"


def _policy(approach: str) -> SelectionPolicy:
    """A minimal single-criterion policy, sufficient for a frozen selection."""
    return SelectionPolicy(
        approach=approach,
        criteria=(
            SelectionCriterion(
                name="placeholder",
                metric_references=(MetricReference("mae", TOTAL),),
                optimization_direction="minimize",
            ),
        ),
        likelihood_comparability="test fixture",
    )


def _direct_candidate(candidate_id: str = "direct-poisson") -> CandidateDefinition:
    """A DirectCohortModel candidate declaring only cohort-scoped metrics."""
    return CandidateDefinition(
        candidate_id=candidate_id,
        approach="DirectCohortModel",
        model_factory=lambda: None,
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(
            *(MeanAbsoluteError(cohort) for cohort in COHORTS),
            *(RootMeanSquaredError(cohort) for cohort in COHORTS),
            *(
                ParametricPredictiveNegativeLogLikelihood(
                    target=cohort, interpretation="independent direct-cohort likelihood"
                )
                for cohort in COHORTS
            ),
            CompositionLogLoss(),
        ),
        configuration={"family": "poisson"},
    )


def _conditional_candidate(
    candidate_id: str, approach: str, interpretation: str = "conditional"
) -> CandidateDefinition:
    """An IndependentTotalProbabilityModel/BayesianConditionalModel candidate."""
    return CandidateDefinition(
        candidate_id=candidate_id,
        approach=approach,
        model_factory=lambda: None,
        fit_feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TOTAL_FEATURE_SPEC, DEFAULT_PROBABILITY_FEATURE_SPEC),
        metrics=(
            *(MeanAbsoluteError(target) for target in (TOTAL, *COHORTS)),
            *(RootMeanSquaredError(target) for target in (TOTAL, *COHORTS)),
            JointPredictiveNegativeLogLikelihood(interpretation=interpretation),
            CompositionLogLoss(),
        ),
        configuration={},
    )


def _metric_row(candidate_id: str, approach: str, metric, mean: float) -> dict:
    """One aggregate-metrics row shaped like ``aggregation.py`` produces."""
    return {
        "candidate_id": candidate_id,
        "approach": approach,
        "metric_name": metric.name,
        "target": metric.target,
        "aggregation_level": metric.aggregation_level,
        "mean": mean,
        "standard_deviation": 0.0,
        "fold_count": 2,
    }


def _selection(
    approach: str, selected_candidate_id: str, policy: SelectionPolicy | None = None
) -> FrozenApproachSelection:
    """A frozen within-approach selection with no rejected candidates."""
    return FrozenApproachSelection(
        approach=approach,
        selected_candidate_id=selected_candidate_id,
        rejected_candidate_ids=(),
        rejection_reasons={},
        criterion_values={selected_candidate_id: {"placeholder": 0.0}},
        policy=policy or _policy(approach),
    )


def _provenance() -> ExperimentProvenance:
    """Minimal, JSON-safe provenance sufficient to build a comparison result."""
    return ExperimentProvenance(
        training_data_hash="data-hash",
        training_schema_hash="schema-hash",
        manifest_fingerprint=MANIFEST_FINGERPRINT,
        split_summary={"strategy": "test"},
        master_seed=1,
        master_seed_source="explicit_argument",
        feature_spec_fingerprints={},
        package_versions={},
        run_settings={},
    )


def _cv_result(
    *,
    candidates: Sequence[CandidateDefinition],
    selections: Sequence[FrozenApproachSelection],
    metric_rows: Sequence[dict],
    manifest_fingerprint: str = MANIFEST_FINGERPRINT,
) -> CrossValidationExperimentResult:
    """Build a minimal comparison result carrying only what selection reads."""
    empty = pd.DataFrame()
    freeze = SelectionFreeze(
        manifest_fingerprint=manifest_fingerprint,
        outer_training_building_ids=(1, 2, 3),
        fold_identities=(
            FoldIdentity(
                fold_index=0,
                fit_building_ids=(1, 2),
                validation_building_ids=(3,),
                fingerprint="fold-0",
            ),
        ),
        candidate_descriptors=tuple(
            candidate.to_descriptor() for candidate in candidates
        ),
        selections=tuple(selections),
        master_seed=1,
    )
    return CrossValidationExperimentResult(
        fold_runs=(),
        fold_metrics_df=empty,
        aggregate_metrics_df=pd.DataFrame(metric_rows),
        bootstrap_intervals_df=empty,
        predictions_df=empty,
        calibration_df=empty,
        importance_df=empty,
        importance_summary_df=empty,
        fold_coverage_df=empty,
        selections=tuple(selections),
        freeze=freeze,
        provenance=_provenance(),
    )


def _default_scenario(
    *,
    joint_independent: float = 2.0,
    joint_bayesian: float = 1.5,
    composition: dict[str, float] | None = None,
    cohort_rmse: dict[str, float] | None = None,
    cohort_mae: dict[str, float] | None = None,
) -> CrossValidationExperimentResult:
    """A three-candidate scenario: Bayesian wins the conditional stage by default."""
    direct = _direct_candidate()
    independent = _conditional_candidate("independent-nb2", "IndependentTotalProbabilityModel")
    bayesian = _conditional_candidate("bayesian-reduced", "BayesianConditionalModel")
    candidates = (direct, independent, bayesian)
    selections = (
        _selection("DirectCohortModel", "direct-poisson"),
        _selection("IndependentTotalProbabilityModel", "independent-nb2"),
        _selection("BayesianConditionalModel", "bayesian-reduced"),
    )
    composition = composition or {
        "direct-poisson": 1.0,
        "independent-nb2": 0.9,
        "bayesian-reduced": 0.8,
    }
    cohort_rmse = cohort_rmse or dict.fromkeys(
        ("direct-poisson", "independent-nb2", "bayesian-reduced"), 1.0
    )
    cohort_mae = cohort_mae or dict.fromkeys(
        ("direct-poisson", "independent-nb2", "bayesian-reduced"), 0.5
    )
    rows: list[dict] = []
    joint_nll = {"independent-nb2": joint_independent, "bayesian-reduced": joint_bayesian}
    for candidate in (independent, bayesian):
        rows.append(
            _metric_row(
                candidate.candidate_id,
                candidate.approach,
                JointPredictiveNegativeLogLikelihood(interpretation="x"),
                joint_nll[candidate.candidate_id],
            )
        )
    for candidate in candidates:
        rows.append(
            _metric_row(
                candidate.candidate_id,
                candidate.approach,
                CompositionLogLoss(),
                composition[candidate.candidate_id],
            )
        )
        for cohort in COHORTS:
            rows.append(
                _metric_row(
                    candidate.candidate_id,
                    candidate.approach,
                    RootMeanSquaredError(cohort),
                    cohort_rmse[candidate.candidate_id],
                )
            )
            rows.append(
                _metric_row(
                    candidate.candidate_id,
                    candidate.approach,
                    MeanAbsoluteError(cohort),
                    cohort_mae[candidate.candidate_id],
                )
            )
        if candidate.approach != "DirectCohortModel":
            # Total-scoped RMSE/MAE exist in production descriptors but must
            # never enter a "cohort" reduction.
            rows.append(
                _metric_row(candidate.candidate_id, candidate.approach, RootMeanSquaredError(TOTAL), -999.0)
            )
            rows.append(
                _metric_row(candidate.candidate_id, candidate.approach, MeanAbsoluteError(TOTAL), -999.0)
            )
            rows.append(
                _metric_row(
                    candidate.candidate_id,
                    candidate.approach,
                    ParametricPredictiveNegativeLogLikelihood(
                        target=COHORTS[-1], interpretation="decoy"
                    ),
                    999.0,
                )
            )
    return _cv_result(candidates=candidates, selections=selections, metric_rows=rows)


# --- Conditional (Model B vs Bayesian) stage ---------------------------


def test_bayesian_wins_the_conditional_stage_on_lower_joint_nll() -> None:
    result = _default_scenario(joint_independent=2.0, joint_bayesian=1.5)
    selection = select_cross_family_winner(result)

    assert selection.conditional_winner_candidate_id == "bayesian-reduced"
    assert not selection.conditional_tie_break_used


def test_independent_wins_the_conditional_stage_on_lower_joint_nll() -> None:
    result = _default_scenario(joint_independent=1.1, joint_bayesian=1.5)
    selection = select_cross_family_winner(result)

    assert selection.conditional_winner_candidate_id == "independent-nb2"


def test_conditional_stage_tie_breaks_by_candidate_id() -> None:
    result = _default_scenario(joint_independent=1.2, joint_bayesian=1.2)
    selection = select_cross_family_winner(result)

    assert selection.conditional_tie_break_used
    # "bayesian-reduced" < "independent-nb2" lexicographically.
    assert selection.conditional_winner_candidate_id == "bayesian-reduced"


# --- Final (conditional winner vs DirectCohortModel) stage --------------


def test_composition_log_loss_decides_the_final_stage_first() -> None:
    """Direct loses on RMSE/MAE but wins on composition, and composition wins."""
    result = _default_scenario(
        composition={"direct-poisson": 0.3, "independent-nb2": 0.9, "bayesian-reduced": 0.9},
        cohort_rmse={"direct-poisson": 5.0, "independent-nb2": 5.0, "bayesian-reduced": 0.1},
        cohort_mae={"direct-poisson": 5.0, "independent-nb2": 5.0, "bayesian-reduced": 0.1},
    )
    selection = select_cross_family_winner(result)

    assert selection.conditional_winner_candidate_id == "bayesian-reduced"
    assert selection.selected_candidate_id == "direct-poisson"
    assert selection.decisive_final_criterion == "composition_log_loss"
    assert not selection.final_tie_break_used


def test_mean_cohort_rmse_decides_when_composition_ties() -> None:
    result = _default_scenario(
        composition={"direct-poisson": 0.5, "independent-nb2": 0.9, "bayesian-reduced": 0.5},
        cohort_rmse={"direct-poisson": 2.0, "independent-nb2": 9.0, "bayesian-reduced": 1.0},
        cohort_mae={"direct-poisson": 9.0, "independent-nb2": 9.0, "bayesian-reduced": 0.1},
    )
    selection = select_cross_family_winner(result)

    assert selection.decisive_final_criterion == "mean_cohort_rmse"
    assert selection.selected_candidate_id == "bayesian-reduced"


def test_mean_cohort_mae_decides_when_composition_and_rmse_tie() -> None:
    result = _default_scenario(
        composition={"direct-poisson": 0.5, "independent-nb2": 0.9, "bayesian-reduced": 0.5},
        cohort_rmse={"direct-poisson": 2.0, "independent-nb2": 9.0, "bayesian-reduced": 2.0},
        cohort_mae={"direct-poisson": 3.0, "independent-nb2": 9.0, "bayesian-reduced": 1.0},
    )
    selection = select_cross_family_winner(result)

    assert selection.decisive_final_criterion == "mean_cohort_mae"
    assert selection.selected_candidate_id == "bayesian-reduced"


def test_final_stage_tie_breaks_by_candidate_id() -> None:
    result = _default_scenario(
        composition={"direct-poisson": 0.5, "independent-nb2": 0.9, "bayesian-reduced": 0.5},
        cohort_rmse={"direct-poisson": 2.0, "independent-nb2": 9.0, "bayesian-reduced": 2.0},
        cohort_mae={"direct-poisson": 3.0, "independent-nb2": 9.0, "bayesian-reduced": 3.0},
    )
    selection = select_cross_family_winner(result)

    assert selection.decisive_final_criterion is None
    assert selection.final_tie_break_used
    # "bayesian-reduced" < "direct-poisson" lexicographically.
    assert selection.selected_candidate_id == "bayesian-reduced"


def test_candidate_order_invariance() -> None:
    """Shuffling ``selections`` must not change the decision."""
    result = _default_scenario()
    shuffled = replace(
        result,
        selections=tuple(reversed(result.selections)),
        freeze=replace(result.freeze, selections=tuple(reversed(result.selections))),
    )

    first = select_cross_family_winner(result)
    second = select_cross_family_winner(shuffled)

    assert first.selected_candidate_id == second.selected_candidate_id
    assert first.criterion_values == second.criterion_values


# --- Structural safety: never total, never per-target NLL, never comparators


def test_total_scoped_rmse_and_mae_never_enter_the_cohort_reduction() -> None:
    """The default scenario plants a total RMSE/MAE decoy of -999; it must be ignored."""
    result = _default_scenario()
    selection = select_cross_family_winner(result)

    # Only the conditional winner and DirectCohortModel reach the final stage.
    for candidate_id in ("bayesian-reduced", "direct-poisson"):
        assert selection.criterion_values[candidate_id]["mean_cohort_rmse"] >= 0.0
        assert selection.criterion_values[candidate_id]["mean_cohort_mae"] >= 0.0
    references = selection.criterion_metric_references["mean_cohort_rmse"]
    assert {reference["target"] for reference in references} == set(COHORTS)


def test_total_metrics_declared_by_direct_cohort_never_enter_the_cohort_reduction() -> None:
    """A total RMSE/MAE decoy on the DirectCohortModel candidate itself must be ignored.

    The cohort target list is read from the DirectCohortModel descriptor, so
    decoys planted only on the conditional candidates can never reach the
    reduction. This decoy is declared by, and logged for, the candidate the
    list is actually read from.
    """
    base = _default_scenario()
    direct = _direct_candidate()
    direct = replace(
        direct,
        metrics=(*direct.metrics, RootMeanSquaredError(TOTAL), MeanAbsoluteError(TOTAL)),
    )
    descriptors = tuple(
        direct.to_descriptor() if descriptor["candidate_id"] == direct.candidate_id else descriptor
        for descriptor in base.freeze.candidate_descriptors
    )
    decoys = pd.DataFrame(
        [
            _metric_row(direct.candidate_id, direct.approach, RootMeanSquaredError(TOTAL), -999.0),
            _metric_row(direct.candidate_id, direct.approach, MeanAbsoluteError(TOTAL), -999.0),
        ]
    )
    result = replace(
        base,
        freeze=replace(base.freeze, candidate_descriptors=descriptors),
        aggregate_metrics_df=pd.concat([base.aggregate_metrics_df, decoys], ignore_index=True),
    )

    selection = select_cross_family_winner(result)

    assert selection.criterion_values["direct-poisson"]["mean_cohort_rmse"] == 1.0
    assert selection.criterion_values["direct-poisson"]["mean_cohort_mae"] == 0.5
    for criterion in ("mean_cohort_rmse", "mean_cohort_mae"):
        targets = {ref["target"] for ref in selection.criterion_metric_references[criterion]}
        assert targets == set(COHORTS)


def test_wrong_aggregation_level_row_is_never_read_as_the_real_metric() -> None:
    """A same name/target row at the wrong aggregation level must be ignored.

    ``composition_log_loss`` is declared at the ``child`` level; a decoy row
    with the same name and target but logged at ``building`` level must never
    be picked up in its place.
    """
    result = _default_scenario()
    decoy = result.aggregate_metrics_df.iloc[0:1].copy()
    decoy["candidate_id"] = "bayesian-reduced"
    decoy["metric_name"] = "composition_log_loss"
    decoy["target"] = "composition"
    decoy["aggregation_level"] = "building"
    decoy["mean"] = -999.0
    with_decoy = replace(
        result,
        aggregate_metrics_df=pd.concat(
            [result.aggregate_metrics_df, decoy], ignore_index=True
        ),
    )

    selection = select_cross_family_winner(with_decoy)

    assert selection.criterion_values["bayesian-reduced"]["composition_log_loss"] != -999.0


def test_per_target_predictive_nll_is_never_read() -> None:
    """The default scenario plants a decoy predictive_nll of 999; it must not appear."""
    result = _default_scenario()
    selection = select_cross_family_winner(result)

    assert "predictive_nll" not in selection.rule.final_criteria
    assert "predictive_nll" != selection.rule.conditional_criterion
    for criteria in selection.criterion_values.values():
        assert "predictive_nll" not in criteria


def test_diagnostic_comparator_candidates_are_ignored() -> None:
    """An unselected comparator descriptor must not influence the decision."""
    result = _default_scenario()
    comparator = CandidateDefinition(
        candidate_id="direct-normal",
        approach="DirectCohortModel",
        model_factory=lambda: None,
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(MeanAbsoluteError(TOTAL),),
        configuration={"family": "normal"},
        selection_role="diagnostic_comparator",
    )
    with_comparator = replace(
        result,
        freeze=replace(
            result.freeze,
            candidate_descriptors=(
                *result.freeze.candidate_descriptors,
                comparator.to_descriptor(),
            ),
        ),
    )

    baseline = select_cross_family_winner(result)
    augmented = select_cross_family_winner(with_comparator)

    assert baseline.selected_candidate_id == augmented.selected_candidate_id
    assert baseline.criterion_values == augmented.criterion_values


# --- Validation ----------------------------------------------------------


def test_rejects_missing_composition_declaration() -> None:
    """A candidate that declares cohort RMSE/MAE but not composition_log_loss."""
    direct = CandidateDefinition(
        candidate_id="direct-poisson",
        approach="DirectCohortModel",
        model_factory=lambda: None,
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(
            *(RootMeanSquaredError(cohort) for cohort in COHORTS),
            *(MeanAbsoluteError(cohort) for cohort in COHORTS),
        ),
        configuration={},
    )
    independent = _conditional_candidate("independent-nb2", "IndependentTotalProbabilityModel")
    bayesian = _conditional_candidate("bayesian-reduced", "BayesianConditionalModel")
    rows = [
        _metric_row(
            "independent-nb2",
            "IndependentTotalProbabilityModel",
            JointPredictiveNegativeLogLikelihood(interpretation="x"),
            2.0,
        ),
        # "bayesian-reduced" < "independent-nb2" lexicographically, so an exact
        # tie here makes it the conditional winner and the first candidate the
        # final-stage loop reaches.
        _metric_row(
            "bayesian-reduced",
            "BayesianConditionalModel",
            JointPredictiveNegativeLogLikelihood(interpretation="x"),
            2.0,
        ),
        _metric_row("bayesian-reduced", "BayesianConditionalModel", CompositionLogLoss(), 0.5),
    ]
    for cohort in COHORTS:
        rows.append(
            _metric_row("bayesian-reduced", "BayesianConditionalModel", RootMeanSquaredError(cohort), 1.0)
        )
        rows.append(
            _metric_row("bayesian-reduced", "BayesianConditionalModel", MeanAbsoluteError(cohort), 0.5)
        )
    result = _cv_result(
        candidates=(direct, independent, bayesian),
        selections=(
            _selection("DirectCohortModel", "direct-poisson"),
            _selection("IndependentTotalProbabilityModel", "independent-nb2"),
            _selection("BayesianConditionalModel", "bayesian-reduced"),
        ),
        metric_rows=rows,
    )

    with pytest.raises(ValueError, match="does not declare metric"):
        select_cross_family_winner(result)


def test_rejects_a_missing_aggregate_metric() -> None:
    result = _default_scenario()
    dropped = replace(
        result,
        aggregate_metrics_df=result.aggregate_metrics_df.loc[
            ~(
                (result.aggregate_metrics_df["candidate_id"] == "bayesian-reduced")
                & (result.aggregate_metrics_df["metric_name"] == "joint_predictive_nll")
            )
        ],
    )

    with pytest.raises(ValueError, match="Expected exactly one"):
        select_cross_family_winner(dropped)


def test_rejects_a_duplicate_aggregate_metric() -> None:
    result = _default_scenario()
    duplicate_row = result.aggregate_metrics_df[
        (result.aggregate_metrics_df["candidate_id"] == "bayesian-reduced")
        & (result.aggregate_metrics_df["metric_name"] == "joint_predictive_nll")
    ]
    duplicated = replace(
        result,
        aggregate_metrics_df=pd.concat(
            [result.aggregate_metrics_df, duplicate_row], ignore_index=True
        ),
    )

    with pytest.raises(ValueError, match="Expected exactly one"):
        select_cross_family_winner(duplicated)


def test_rejects_a_non_finite_metric_value() -> None:
    result = _default_scenario()
    mutated = result.aggregate_metrics_df.copy()
    mask = (mutated["candidate_id"] == "bayesian-reduced") & (
        mutated["metric_name"] == "joint_predictive_nll"
    )
    mutated.loc[mask, "mean"] = np.nan
    non_finite = replace(result, aggregate_metrics_df=mutated)

    with pytest.raises(ValueError, match="not finite"):
        select_cross_family_winner(non_finite)


def test_rejects_when_an_approach_has_no_winner() -> None:
    result = _default_scenario()
    missing_approach = replace(
        result,
        selections=tuple(
            selection
            for selection in result.selections
            if selection.approach != "BayesianConditionalModel"
        ),
    )

    with pytest.raises(ValueError, match="missing"):
        select_cross_family_winner(missing_approach)


# --- SelectionFreeze transitions -----------------------------------------


def test_freeze_records_a_cross_family_decision_once() -> None:
    result = _default_scenario()
    selection = select_cross_family_winner(result)

    pretest = result.freeze.with_cross_family_selection(selection)

    assert pretest.cross_family_selection == selection
    assert pretest.test_metrics is None
    with pytest.raises(ValueError, match="already records"):
        pretest.with_cross_family_selection(selection)


def test_freeze_rejects_a_decision_with_the_wrong_manifest_fingerprint() -> None:
    result = _default_scenario()
    selection = select_cross_family_winner(result)
    mismatched = replace(selection, manifest_fingerprint="other-manifest")

    with pytest.raises(ValueError, match="manifest fingerprint"):
        result.freeze.with_cross_family_selection(mismatched)


def test_freeze_rejects_a_decision_with_unknown_candidate_ids() -> None:
    result = _default_scenario()
    selection = select_cross_family_winner(result)
    unknown = replace(selection, compared_candidate_ids=(*selection.compared_candidate_ids, "ghost"))

    with pytest.raises(ValueError, match="unknown candidate IDs"):
        result.freeze.with_cross_family_selection(unknown)


def test_freeze_rejects_test_metrics_before_a_decision() -> None:
    result = _default_scenario()

    with pytest.raises(ValueError, match="prior cross-family selection"):
        result.freeze.with_test_metrics({"rmse": 1.0})


def test_freeze_rejects_a_second_test_result() -> None:
    result = _default_scenario()
    selection = select_cross_family_winner(result)
    pretest = result.freeze.with_cross_family_selection(selection)
    finalized = pretest.with_test_metrics({"rmse": 1.0})

    assert finalized.test_metrics == {"rmse": 1.0}
    with pytest.raises(ValueError, match="already finalized"):
        finalized.with_test_metrics({"rmse": 2.0})


def test_freeze_to_dict_reports_cross_family_selection_and_test_metrics() -> None:
    result = _default_scenario()
    selection = select_cross_family_winner(result)
    finalized = result.freeze.with_cross_family_selection(selection).with_test_metrics(
        {"rmse": 1.0}
    )

    payload = finalized.to_dict()
    assert payload["cross_family_selection"]["selected_candidate_id"] == (
        selection.selected_candidate_id
    )
    assert payload["test_metrics"] == {"rmse": 1.0}


def test_gate6_freeze_still_carries_no_decision_or_test_metrics() -> None:
    """`SelectionFreeze` built the Gate 6 way must be unaffected by the new fields."""
    result = _default_scenario()

    assert result.freeze.cross_family_selection is None
    assert result.freeze.test_metrics is None
    assert result.freeze.to_dict()["cross_family_selection"] is None
    assert result.freeze.to_dict()["test_metrics"] is None
