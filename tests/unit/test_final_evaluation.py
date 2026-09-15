"""Tests for Gate 8 full-training refit and one-time lockbox evaluation.

Uses ``tests/unit/bundle_spy.py``'s spy experiment for the same reason the
Gate 7 tracking tests do: it runs in milliseconds and exercises every run
shape (a diagnostic comparator, Model B/Bayesian stand-ins) without a real
likelihood. Its ``spy_candidates()`` metrics (total MAE, a per-total NLL) are
sufficient here because neither ``refit_frozen_approach_winners`` nor
``evaluate_frozen_models_on_lockbox`` reads ``aggregate_metrics_df`` or scores
a cross-family ranking; only ``final_selection.select_cross_family_winner``
needs the richer composition/joint metric set, and this module never calls
it. The cross-family decision itself is stubbed with a minimal
``CrossFamilySelection`` that names one real winner, exactly as a caller who
already ran Phase 2 would hand this module a real one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace

import pytest

from age_group_prediction.experiment import (
    evaluate_frozen_models_on_lockbox,
    final_evaluation,
    refit_frozen_approach_winners,
)
from age_group_prediction.experiment.evidence import FrozenApproachSelection
from age_group_prediction.experiment.final_selection import (
    CrossFamilySelection,
    CrossFamilySelectionRule,
)
from age_group_prediction.experiment_config import load_experiment_config
from age_group_prediction.modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    BayesianConditionalConfig,
    OuterSplitConfig,
)
from tests.unit.bundle_spy import (
    REPOSITORY_ROOT,
    BundleSpyModel,
    direct_diagnostics,
    independent_diagnostics,
    modeling_table,
    run_spy_experiment,
    spy_candidates,
)

TOTAL = DEFAULT_MODELING_SCHEMA.total_target_column


def _full_bayesian_diagnostics(
    *,
    active_profile: str = "full",
    action: str = "error",
    stage_overrides: dict | None = None,
    threshold_overrides: dict | None = None,
) -> dict:
    """A Bayesian diagnostics payload shaped like a passing full-profile refit.

    Draw counts and thresholds default to the package's full profile, the floor
    ``_require_full_bayesian_policy`` enforces. The overrides replace
    individual stage keys (such as ``chains``) or policy thresholds (such as
    ``maximum_rhat``).
    """
    defaults = BayesianConditionalConfig()
    thresholds = {**asdict(defaults.full_diagnostics), "action": action}
    thresholds.update(threshold_overrides or {})
    stage = {
        "worst_rhat": 1.01,
        "minimum_effective_sample_size": 500.0,
        "policy_passed": True,
        "policy_failures": [],
        "active_profile": active_profile,
        "chains": defaults.full_profile.chains,
        "warmup_steps": defaults.full_profile.warmup_steps,
        "posterior_samples": defaults.full_profile.posterior_samples,
    }
    stage.update(stage_overrides or {})
    return {
        name: {**stage, "policy_thresholds": dict(thresholds)}
        for name in ("total", "composition")
    }


def _final_refit_factories(
    *, bayesian_diagnostics: dict | None = None, bayesian_fit_error: BaseException | None = None
) -> dict[str, Callable[[], BundleSpyModel]]:
    """Fresh, unfitted spy factories for every candidate ``spy_candidates`` declares."""
    return {
        "direct-poisson": lambda: BundleSpyModel(
            offset=0.0, family="poisson", diagnostics=direct_diagnostics("poisson")
        ),
        "direct-nb2": lambda: BundleSpyModel(
            offset=0.3, family="nb2", diagnostics=direct_diagnostics("nb2")
        ),
        "direct-normal": lambda: BundleSpyModel(
            offset=0.1, family="normal", diagnostics=direct_diagnostics("normal")
        ),
        "independent-nb2": lambda: BundleSpyModel(
            offset=0.2, family="nb2", diagnostics=independent_diagnostics()
        ),
        "bayesian-reduced": lambda: BundleSpyModel(
            offset=0.1,
            diagnostics=bayesian_diagnostics or _full_bayesian_diagnostics(),
            fit_error=bayesian_fit_error,
        ),
    }


def _stub_cross_family_selection(
    *, manifest_fingerprint: str, selections: tuple[FrozenApproachSelection, ...]
) -> CrossFamilySelection:
    """A minimal, JSON-safe cross-family decision naming DirectCohortModel's winner.

    Phase 3 never inspects the decision's own criteria; it only needs
    ``manifest_fingerprint`` and ``selected_candidate_id`` to match a real
    ``SelectionFreeze``, exactly as a caller who already ran
    ``final_selection.select_cross_family_winner`` would supply.
    """
    approaches = {selection.selected_candidate_id: selection.approach for selection in selections}
    direct = next(s for s in selections if s.approach == "DirectCohortModel")
    independent = next(
        s for s in selections if s.approach == "IndependentTotalProbabilityModel"
    )
    rule = CrossFamilySelectionRule(
        name="test-stub",
        version="1.0",
        conditional_criterion="joint_predictive_nll",
        final_criteria=("composition_log_loss", "mean_cohort_rmse", "mean_cohort_mae", "candidate_id"),
        reductions={"composition_log_loss": "single_child_level_value"},
        likelihood_comparability="test fixture",
    )
    return CrossFamilySelection(
        rule=rule,
        manifest_fingerprint=manifest_fingerprint,
        compared_candidate_ids=tuple(approaches),
        candidate_approaches=approaches,
        criterion_metric_references={},
        criterion_values={candidate_id: {} for candidate_id in approaches},
        conditional_winner_candidate_id=independent.selected_candidate_id,
        conditional_tie_break_used=False,
        selected_candidate_id=direct.selected_candidate_id,
        selected_approach="DirectCohortModel",
        decisive_final_criterion="composition_log_loss",
        final_tie_break_used=False,
    )


@pytest.fixture(scope="module")
def spy_cv_result():
    """A fast, deterministic CV comparison result, reused read-only by every test."""
    return run_spy_experiment(capture_artifacts=False)


# --- Full-training refit --------------------------------------------------


def test_refit_produces_exactly_three_full_refits(spy_cv_result) -> None:
    split, result = spy_cv_result
    artifacts = refit_frozen_approach_winners(
        split.train_df,
        split_manifest=split.manifest,
        cv_result=result,
        candidates=spy_candidates(),
        final_refit_factories=_final_refit_factories(),
    )

    assert len(artifacts) == 3
    assert {artifact.approach for artifact in artifacts} == {
        "DirectCohortModel",
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    }
    assert {artifact.candidate_id for artifact in artifacts} == {
        selection.selected_candidate_id for selection in result.selections
    }
    for artifact in artifacts:
        assert artifact.model.is_fitted
        assert artifact.evidence.reload_check.passed
        assert artifact.evidence.manifest_fingerprint == result.freeze.manifest_fingerprint


def test_refit_state_bundle_reload_reproduces_smoke_predictions(spy_cv_result) -> None:
    split, result = spy_cv_result
    artifacts = refit_frozen_approach_winners(
        split.train_df,
        split_manifest=split.manifest,
        cv_result=result,
        candidates=spy_candidates(),
        final_refit_factories=_final_refit_factories(),
    )

    for artifact in artifacts:
        differences = artifact.evidence.reload_check.max_abs_differences
        assert all(value == 0.0 for value in differences.values())
        assert artifact.evidence.smoke_frame_building_ids


def test_refit_seeds_are_reproducible_and_candidate_order_invariant(spy_cv_result) -> None:
    split, result = spy_cv_result
    factories = _final_refit_factories()

    first = refit_frozen_approach_winners(
        split.train_df,
        split_manifest=split.manifest,
        cv_result=result,
        candidates=spy_candidates(),
        final_refit_factories=factories,
    )
    reordered_result = replace(result, selections=tuple(reversed(result.selections)))
    second = refit_frozen_approach_winners(
        split.train_df,
        split_manifest=split.manifest,
        cv_result=reordered_result,
        candidates=spy_candidates(),
        final_refit_factories=factories,
    )

    first_by_id = {artifact.candidate_id: artifact for artifact in first}
    second_by_id = {artifact.candidate_id: artifact for artifact in second}
    assert set(first_by_id) == set(second_by_id)
    for candidate_id, artifact in first_by_id.items():
        assert artifact.evidence.seeds == second_by_id[candidate_id].evidence.seeds


def test_refit_enforces_the_full_bayesian_profile(spy_cv_result) -> None:
    split, result = spy_cv_result
    factories = _final_refit_factories(
        bayesian_diagnostics=_full_bayesian_diagnostics(active_profile="reduced")
    )

    with pytest.raises(ValueError, match="full Bayesian profile"):
        refit_frozen_approach_winners(
            split.train_df,
            split_manifest=split.manifest,
            cv_result=result,
            candidates=spy_candidates(),
            final_refit_factories=factories,
        )


def test_refit_enforces_the_strict_bayesian_diagnostic_policy(spy_cv_result) -> None:
    split, result = spy_cv_result
    factories = _final_refit_factories(
        bayesian_diagnostics=_full_bayesian_diagnostics(action="warn")
    )

    with pytest.raises(ValueError, match="strict diagnostic policy|action='error'"):
        refit_frozen_approach_winners(
            split.train_df,
            split_manifest=split.manifest,
            cv_result=result,
            candidates=spy_candidates(),
            final_refit_factories=factories,
        )


def test_bayesian_policy_failure_preserves_stage_diagnostics_and_failures(
    spy_cv_result,
) -> None:
    """A failed fit must not replay the manifest; the run stops before it can."""
    split, result = spy_cv_result
    scripted = RuntimeError("Bayesian total diagnostics failed: worst_rhat=1.2")
    scripted.stage = "total"
    scripted.diagnostics = {"worst_rhat": 1.2}
    scripted.failures = ("worst_rhat=1.2 violates threshold 1.05",)
    factories = _final_refit_factories(bayesian_fit_error=scripted)

    with pytest.raises(RuntimeError) as excinfo:
        refit_frozen_approach_winners(
            split.train_df,
            split_manifest=split.manifest,
            cv_result=result,
            candidates=spy_candidates(),
            final_refit_factories=factories,
        )

    assert excinfo.value.stage == "total"
    assert excinfo.value.diagnostics == {"worst_rhat": 1.2}
    assert excinfo.value.failures == scripted.failures
    assert excinfo.value.__cause__ is scripted


def test_refit_rejects_a_manifest_fingerprint_mismatch(spy_cv_result) -> None:
    split, result = spy_cv_result
    tampered_manifest = replace(split.manifest, requested_holdout_fraction=0.99)

    with pytest.raises(ValueError, match="manifest"):
        refit_frozen_approach_winners(
            split.train_df,
            split_manifest=tampered_manifest,
            cv_result=result,
            candidates=spy_candidates(),
            final_refit_factories=_final_refit_factories(),
        )


def test_refit_rejects_outer_training_ids_that_do_not_match_the_freeze(spy_cv_result) -> None:
    split, result = spy_cv_result
    truncated = split.train_df.iloc[:-1].copy()

    with pytest.raises(ValueError, match="building IDs"):
        refit_frozen_approach_winners(
            truncated,
            split_manifest=split.manifest,
            cv_result=result,
            candidates=spy_candidates(),
            final_refit_factories=_final_refit_factories(),
        )


def test_refit_rejects_a_candidate_descriptor_mismatch(spy_cv_result) -> None:
    split, result = spy_cv_result
    winner_id = next(
        s.selected_candidate_id for s in result.selections if s.approach == "DirectCohortModel"
    )
    mutated_candidates = [
        replace(candidate, configuration={"family": "mutated"})
        if candidate.candidate_id == winner_id
        else candidate
        for candidate in spy_candidates()
    ]

    with pytest.raises(ValueError, match="does not match the descriptor"):
        refit_frozen_approach_winners(
            split.train_df,
            split_manifest=split.manifest,
            cv_result=result,
            candidates=mutated_candidates,
            final_refit_factories=_final_refit_factories(),
        )


def test_refit_rejects_missing_factories_for_a_frozen_winner(spy_cv_result) -> None:
    split, result = spy_cv_result
    factories = _final_refit_factories()
    winner_id = next(
        s.selected_candidate_id for s in result.selections if s.approach == "DirectCohortModel"
    )
    del factories[winner_id]

    with pytest.raises(ValueError, match="final_refit_factories"):
        refit_frozen_approach_winners(
            split.train_df,
            split_manifest=split.manifest,
            cv_result=result,
            candidates=spy_candidates(),
            final_refit_factories=factories,
        )


def test_refit_factory_must_return_a_base_age_group_model(spy_cv_result) -> None:
    split, result = spy_cv_result
    factories = _final_refit_factories()
    winner_id = next(
        s.selected_candidate_id for s in result.selections if s.approach == "DirectCohortModel"
    )
    factories[winner_id] = lambda: object()

    with pytest.raises(TypeError, match="BaseAgeGroupModel"):
        refit_frozen_approach_winners(
            split.train_df,
            split_manifest=split.manifest,
            cv_result=result,
            candidates=spy_candidates(),
            final_refit_factories=factories,
        )


def test_refit_does_not_mutate_the_source_cv_result(spy_cv_result) -> None:
    split, result = spy_cv_result
    before = result.freeze.to_dict()

    refit_frozen_approach_winners(
        split.train_df,
        split_manifest=split.manifest,
        cv_result=result,
        candidates=spy_candidates(),
        final_refit_factories=_final_refit_factories(),
    )

    assert result.freeze.to_dict() == before


# --- Lockbox evaluation ----------------------------------------------------


@pytest.fixture()
def refit_and_pretest_freeze(spy_cv_result):
    """A full refit plus a pretest freeze naming DirectCohortModel as selected."""
    split, result = spy_cv_result
    artifacts = refit_frozen_approach_winners(
        split.train_df,
        split_manifest=split.manifest,
        cv_result=result,
        candidates=spy_candidates(),
        final_refit_factories=_final_refit_factories(),
    )
    selection = _stub_cross_family_selection(
        manifest_fingerprint=result.freeze.manifest_fingerprint,
        selections=result.selections,
    )
    pretest_freeze = result.freeze.with_cross_family_selection(selection)
    return split, result, artifacts, pretest_freeze


def _evaluation_config():
    config = load_experiment_config(REPOSITORY_ROOT / "configs" / "modeling.toml")
    return replace(config.evaluation, bootstrap_replicates=3)


def test_lockbox_evaluation_scores_all_three_winners_once(refit_and_pretest_freeze) -> None:
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze

    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )

    assert len(evaluation_result.evaluations) == 3
    assert {e.approach for e in evaluation_result.evaluations} == {
        "DirectCohortModel",
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    }
    roles = {e.candidate_id: e.role for e in evaluation_result.evaluations}
    assert roles[pretest_freeze.cross_family_selection.selected_candidate_id] == "selected"
    assert sum(role == "comparator" for role in roles.values()) == 2


def test_lockbox_evaluation_gives_every_model_identical_ordered_holdout_ids(
    refit_and_pretest_freeze,
) -> None:
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze

    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )

    id_sets = [
        tuple(evaluation.predictions_df["building_id"]) for evaluation in evaluation_result.evaluations
    ]
    assert len(set(id_sets)) == 1
    assert set(id_sets[0]) == set(evaluation_result.holdout_building_ids)


def test_lockbox_evaluation_labels_metric_scope_per_approach(refit_and_pretest_freeze) -> None:
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze

    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )

    selection_by_approach = {selection.approach: selection for selection in pretest_freeze.selections}
    for evaluation in evaluation_result.evaluations:
        assert evaluation.metric_comparability
        assert (
            evaluation.metric_comparability
            == selection_by_approach[evaluation.approach].policy.likelihood_comparability
        )


def test_lockbox_evaluation_does_not_rank_across_families(refit_and_pretest_freeze) -> None:
    """The selected role must come only from the pretest decision, never test metrics."""
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze

    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )

    selected = next(e for e in evaluation_result.evaluations if e.role == "selected")
    assert selected.candidate_id == pretest_freeze.cross_family_selection.selected_candidate_id
    # The pretest decision is unchanged: this is the value select_cross_family_winner
    # would have produced, and nothing here recomputes or overwrites it.
    assert (
        evaluation_result.finalized_freeze.cross_family_selection.selected_candidate_id
        == pretest_freeze.cross_family_selection.selected_candidate_id
    )


def test_lockbox_evaluation_finalizes_the_freeze_with_test_metrics(
    refit_and_pretest_freeze,
) -> None:
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze

    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )

    finalized = evaluation_result.finalized_freeze
    assert finalized.test_metrics is not None
    assert set(finalized.test_metrics) == {a.candidate_id for a in artifacts}
    # The statistical decision is byte-for-byte unchanged after finalization.
    assert finalized.cross_family_selection.to_dict() == pretest_freeze.cross_family_selection.to_dict()


def test_lockbox_evaluation_state_bundles_reload_checked_before_evaluation(
    refit_and_pretest_freeze,
) -> None:
    """Every evaluation carries the refit's own reload-checked model metadata."""
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze
    artifact_by_id = {a.candidate_id: a for a in artifacts}

    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )

    for evaluation in evaluation_result.evaluations:
        artifact = artifact_by_id[evaluation.candidate_id]
        assert artifact.evidence.reload_check.passed
        assert dict(evaluation.model_metadata) == dict(artifact.evidence.model_metadata)


def test_lockbox_evaluation_refuses_a_freeze_without_a_decision(
    refit_and_pretest_freeze,
) -> None:
    split, result, artifacts, _pretest_freeze = refit_and_pretest_freeze

    with pytest.raises(ValueError, match="cross-family selection"):
        evaluate_frozen_models_on_lockbox(
            modeling_table(),
            split_manifest=split.manifest,
            pretest_freeze=result.freeze,
            refit_result=artifacts,
            candidates=spy_candidates(),
            evaluation_config=_evaluation_config(),
        )


def test_lockbox_evaluation_refuses_a_second_evaluation(refit_and_pretest_freeze) -> None:
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze
    finalized = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    ).finalized_freeze

    with pytest.raises(ValueError, match="already carries test metrics"):
        evaluate_frozen_models_on_lockbox(
            modeling_table(),
            split_manifest=split.manifest,
            pretest_freeze=finalized,
            refit_result=artifacts,
            candidates=spy_candidates(),
            evaluation_config=_evaluation_config(),
        )


def test_lockbox_evaluation_rejects_a_manifest_mismatch(refit_and_pretest_freeze) -> None:
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze
    tampered_manifest = replace(split.manifest, requested_holdout_fraction=0.99)

    with pytest.raises(ValueError, match="manifest"):
        evaluate_frozen_models_on_lockbox(
            modeling_table(),
            split_manifest=tampered_manifest,
            pretest_freeze=pretest_freeze,
            refit_result=artifacts,
            candidates=spy_candidates(),
            evaluation_config=_evaluation_config(),
        )


def test_lockbox_evaluation_rejects_a_replayed_training_partition_mismatch(
    refit_and_pretest_freeze,
) -> None:
    """A pretest freeze frozen against a different table's training IDs is refused."""
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze
    tampered_freeze = replace(
        pretest_freeze, outer_training_building_ids=("nonexistent-id",)
    )

    with pytest.raises(ValueError, match="training partition"):
        evaluate_frozen_models_on_lockbox(
            modeling_table(),
            split_manifest=split.manifest,
            pretest_freeze=tampered_freeze,
            refit_result=artifacts,
            candidates=spy_candidates(),
            evaluation_config=_evaluation_config(),
        )


def test_lockbox_evaluation_does_not_mutate_the_pretest_freeze(
    refit_and_pretest_freeze,
) -> None:
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze
    before = pretest_freeze.to_dict()

    evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )

    assert pretest_freeze.to_dict() == before


def test_replay_config_cannot_depend_on_the_manifest_it_replays() -> None:
    """``_replay_config`` must not source ``strategy_version`` from a manifest.

    ``replay_split_manifest`` rejects a manifest whose recorded strategy no
    longer matches the config's strategy. If that config's own strategy were
    ever built from the manifest being checked, the comparison would compare
    a value to itself and a manifest stamped with a strategy the codebase no
    longer implements would replay silently instead of being refused. Taking
    no manifest parameter at all makes that mistake structurally impossible
    rather than merely absent today.
    """
    import inspect

    signature = inspect.signature(final_evaluation._replay_config)
    assert "manifest" not in signature.parameters
    assert "split_manifest" not in signature.parameters

    config = final_evaluation._replay_config(schema=DEFAULT_MODELING_SCHEMA)
    assert config.strategy_version == OuterSplitConfig().strategy_version
    assert config.building_id_column == DEFAULT_MODELING_SCHEMA.building_id_column
    assert config.neighborhood_id_column == DEFAULT_MODELING_SCHEMA.neighborhood_id_column


# --- Independent-validation remediation (F2, F12, M13, M25) -------------------


def test_lockbox_evaluation_scores_exactly_the_manifest_holdout(refit_and_pretest_freeze) -> None:
    """Scored IDs must equal the manifest's holdout IDs.

    The manifest is an oracle independent of the frame being scored; comparing
    against ``evaluation_result.holdout_building_ids`` alone would agree with
    whatever rows the evaluator happened to predict.
    """
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze

    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
    )

    manifest_holdout = sorted(split.manifest.holdout_building_ids)
    assert sorted(evaluation_result.holdout_building_ids) == manifest_holdout
    for evaluation in evaluation_result.evaluations:
        assert sorted(evaluation.predictions_df["building_id"]) == manifest_holdout


def test_lockbox_evaluation_refuses_a_replay_whose_holdout_is_not_the_manifest(
    refit_and_pretest_freeze, monkeypatch
) -> None:
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze
    real_replay = final_evaluation.replay_split_manifest

    def replay_with_wrong_holdout(*args, **kwargs):
        replayed = real_replay(*args, **kwargs)
        return replace(replayed, test_df=replayed.train_df.head(2))

    monkeypatch.setattr(final_evaluation, "replay_split_manifest", replay_with_wrong_holdout)

    with pytest.raises(ValueError, match="holdout partition does not match the split manifest"):
        evaluate_frozen_models_on_lockbox(
            modeling_table(),
            split_manifest=split.manifest,
            pretest_freeze=pretest_freeze,
            refit_result=artifacts,
            candidates=spy_candidates(),
            evaluation_config=_evaluation_config(),
        )


def test_refit_rejects_altered_training_values_with_the_same_ids(spy_cv_result) -> None:
    split, result = spy_cv_result
    altered = split.train_df.copy()
    altered["ses"] = altered["ses"] + 0.5

    with pytest.raises(ValueError, match="CV provenance hash"):
        refit_frozen_approach_winners(
            altered,
            split_manifest=split.manifest,
            cv_result=result,
            candidates=spy_candidates(),
            final_refit_factories=_final_refit_factories(),
        )


def test_final_evaluation_result_refuses_a_target_column(refit_and_pretest_freeze) -> None:
    import pandas as pd

    _split, _result, _artifacts, pretest_freeze = refit_and_pretest_freeze

    with pytest.raises(ValueError, match="must not carry target columns"):
        final_evaluation.FinalEvaluationResult(
            manifest_fingerprint=pretest_freeze.manifest_fingerprint,
            holdout_building_ids=(),
            evaluations=(),
            finalized_freeze=pretest_freeze,
            holdout_input_df=pd.DataFrame({"building_id": [1], TOTAL: [3]}),
        )


def _stub_decision(result) -> CrossFamilySelection:
    return _stub_cross_family_selection(
        manifest_fingerprint=result.freeze.manifest_fingerprint,
        selections=result.selections,
    )


def _rule_returns(monkeypatch, selection: CrossFamilySelection) -> None:
    """Make the rule return ``selection``: the spy CV result lacks the rule's metrics."""
    monkeypatch.setattr(
        final_evaluation, "select_cross_family_winner", lambda cv_result: selection
    )


def test_verify_pretest_freeze_uses_the_real_cross_family_rule() -> None:
    from age_group_prediction.experiment import final_selection

    assert final_evaluation.select_cross_family_winner is final_selection.select_cross_family_winner


def test_verify_pretest_freeze_accepts_the_rule_decision_on_the_same_freeze(
    spy_cv_result, monkeypatch
) -> None:
    _split, result = spy_cv_result
    selection = _stub_decision(result)
    _rule_returns(monkeypatch, selection)

    final_evaluation.verify_pretest_freeze(
        result.freeze.with_cross_family_selection(selection), result
    )


def test_verify_pretest_freeze_refuses_a_decision_the_rule_did_not_produce(
    spy_cv_result, monkeypatch
) -> None:
    _split, result = spy_cv_result
    selection = _stub_decision(result)
    _rule_returns(monkeypatch, selection)
    alternative_id = next(
        s.selected_candidate_id
        for s in result.selections
        if s.approach == "IndependentTotalProbabilityModel"
    )
    edited = replace(
        selection,
        selected_candidate_id=alternative_id,
        selected_approach="IndependentTotalProbabilityModel",
    )

    with pytest.raises(ValueError, match="not the one the rule produces"):
        final_evaluation.verify_pretest_freeze(
            result.freeze.with_cross_family_selection(edited), result
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "master_seed",
        "master_seed_source",
        "selections",
        "candidate_descriptors",
        "fold_identities",
        "outer_training_building_ids",
    ],
)
def test_verify_pretest_freeze_refuses_a_freeze_that_is_not_this_cv_results(
    spy_cv_result, monkeypatch, tamper
) -> None:
    """Every field of the Gate 6 freeze must be carried over unchanged."""
    _split, result = spy_cv_result
    selection = _stub_decision(result)
    _rule_returns(monkeypatch, selection)
    pretest_freeze = result.freeze.with_cross_family_selection(selection)
    tampered_values = {
        "master_seed": lambda: pretest_freeze.master_seed + 1,
        "master_seed_source": lambda: "forged_source",
        "selections": lambda: tuple(
            replace(s, criterion_values={s.selected_candidate_id: {"forged": 0.0}})
            for s in pretest_freeze.selections
        ),
        "candidate_descriptors": lambda: tuple(
            {**descriptor, "configuration": {"forged": True}}
            for descriptor in pretest_freeze.candidate_descriptors
        ),
        "fold_identities": lambda: pretest_freeze.fold_identities[:-1],
        "outer_training_building_ids": lambda: pretest_freeze.outer_training_building_ids[:-1],
    }
    tampered = replace(pretest_freeze, **{tamper: tampered_values[tamper]()})

    with pytest.raises(ValueError, match="does not extend this CV result's freeze"):
        final_evaluation.verify_pretest_freeze(tampered, result)


def test_verify_pretest_freeze_refuses_a_freeze_without_a_decision(spy_cv_result) -> None:
    _split, result = spy_cv_result

    with pytest.raises(ValueError, match="recorded cross-family selection"):
        final_evaluation.verify_pretest_freeze(result.freeze, result)


def test_verify_pretest_freeze_refuses_a_finalized_freeze(spy_cv_result, monkeypatch) -> None:
    _split, result = spy_cv_result
    selection = _stub_decision(result)
    _rule_returns(monkeypatch, selection)
    finalized = result.freeze.with_cross_family_selection(selection).with_test_metrics({"x": 1.0})

    with pytest.raises(ValueError, match="already carries test metrics"):
        final_evaluation.verify_pretest_freeze(finalized, result)


# --- Independent-validation remediation, second pass (F7, F9) ----------------


def _refit_with_bayesian_diagnostics(split, result, diagnostics: dict):
    return refit_frozen_approach_winners(
        split.train_df,
        split_manifest=split.manifest,
        cv_result=result,
        candidates=spy_candidates(),
        final_refit_factories=_final_refit_factories(bayesian_diagnostics=diagnostics),
    )


@pytest.mark.parametrize(
    "stage_overrides",
    [
        {"chains": 2},
        {"chains": 3},  # one below the floor: the comparison is strict
        {"warmup_steps": 150},
        {"posterior_samples": 150},
        {"chains": None},
    ],
)
def test_refit_refuses_a_full_label_with_fewer_draws_than_the_full_profile(
    spy_cv_result, stage_overrides
) -> None:
    """A profile labelled ``full`` that ran reduced draws is not the full profile."""
    split, result = spy_cv_result

    with pytest.raises(ValueError, match="below the full-profile minimum"):
        _refit_with_bayesian_diagnostics(
            split, result, _full_bayesian_diagnostics(stage_overrides=stage_overrides)
        )


@pytest.mark.parametrize(
    "threshold_overrides",
    [
        {"maximum_rhat": 10.0},
        {"minimum_effective_sample_size": 1.0},
        {"maximum_divergences": 5},
        {"minimum_mean_accept_prob": 0.1},
        {"maximum_mean_accept_prob": 0.999},
        {"maximum_tree_depth_saturation": 0.5},
    ],
)
def test_refit_refuses_a_full_label_with_looser_thresholds(
    spy_cv_result, threshold_overrides
) -> None:
    """``action='error'`` over thresholds loose enough to pass anything is not strict."""
    split, result = spy_cv_result

    with pytest.raises(ValueError, match="looser than the full-profile policy"):
        _refit_with_bayesian_diagnostics(
            split, result, _full_bayesian_diagnostics(threshold_overrides=threshold_overrides)
        )


def test_refit_accepts_a_stricter_than_default_full_profile(spy_cv_result) -> None:
    split, result = spy_cv_result
    diagnostics = _full_bayesian_diagnostics(
        stage_overrides={"chains": 8, "posterior_samples": 2_000},
        threshold_overrides={"maximum_rhat": 1.01, "minimum_effective_sample_size": 400.0},
    )

    artifacts = _refit_with_bayesian_diagnostics(split, result, diagnostics)

    assert len(artifacts) == 3


def test_final_evaluation_result_checks_leakage_against_the_schema_it_is_given(
    refit_and_pretest_freeze,
) -> None:
    import pandas as pd

    _split, _result, _artifacts, pretest_freeze = refit_and_pretest_freeze
    custom_schema = replace(DEFAULT_MODELING_SCHEMA, total_target_column="y_custom_total")

    with pytest.raises(ValueError, match="must not carry target columns"):
        final_evaluation.FinalEvaluationResult(
            manifest_fingerprint=pretest_freeze.manifest_fingerprint,
            holdout_building_ids=(),
            evaluations=(),
            finalized_freeze=pretest_freeze,
            holdout_input_df=pd.DataFrame({"building_id": [1], "y_custom_total": [5]}),
            schema=custom_schema,
        )


def test_lockbox_evaluation_result_carries_the_schema_it_evaluated_with(
    refit_and_pretest_freeze,
) -> None:
    """The evaluator must hand its own schema to the result's leakage check.

    The schema here has the default columns but a distinct version, so a result
    that silently fell back to the default schema compares unequal.
    """
    split, _result, artifacts, pretest_freeze = refit_and_pretest_freeze
    custom_schema = replace(DEFAULT_MODELING_SCHEMA, schema_version="validation-custom")

    evaluation_result = evaluate_frozen_models_on_lockbox(
        modeling_table(),
        split_manifest=split.manifest,
        pretest_freeze=pretest_freeze,
        refit_result=artifacts,
        candidates=spy_candidates(),
        evaluation_config=_evaluation_config(),
        schema=custom_schema,
    )

    assert evaluation_result.schema == custom_schema
    assert evaluation_result.schema != DEFAULT_MODELING_SCHEMA


# --- Independent remediation review (R1) --------------------------------------


@pytest.mark.parametrize("field_name", ["master_seed", "master_seed_source"])
def test_refit_rejects_a_frozen_seed_that_is_not_the_cv_provenance_seed(
    spy_cv_result, field_name
) -> None:
    """Every final seed derives from the freeze's master seed, so it must be the CV's."""
    split, result = spy_cv_result
    forged = {
        "master_seed": result.freeze.master_seed + 1,
        "master_seed_source": "forged_source",
    }[field_name]
    reseeded = replace(result, freeze=replace(result.freeze, **{field_name: forged}))

    with pytest.raises(ValueError, match="master seed does not match the CV provenance"):
        refit_frozen_approach_winners(
            split.train_df,
            split_manifest=split.manifest,
            cv_result=reseeded,
            candidates=spy_candidates(),
            final_refit_factories=_final_refit_factories(),
        )


def test_final_attempt_fingerprint_is_stable_and_covers_what_is_refit_and_scored(
    spy_cv_result,
) -> None:
    split, result = spy_cv_result
    pretest_freeze = result.freeze.with_cross_family_selection(_stub_decision(result))
    selected = pretest_freeze.cross_family_selection.selected_candidate_id

    def refit(factories=None, cv_result=result):
        return refit_frozen_approach_winners(
            split.train_df,
            split_manifest=split.manifest,
            cv_result=cv_result,
            candidates=spy_candidates(),
            final_refit_factories=factories or _final_refit_factories(),
        )

    def fingerprint(artifacts, *, freeze=pretest_freeze, candidates=None, **overrides):
        return final_evaluation.final_attempt_fingerprint(
            freeze,
            artifacts,
            candidates=candidates or spy_candidates(),
            evaluation_config=overrides.get("evaluation_config", _evaluation_config()),
            schema=overrides.get("schema", DEFAULT_MODELING_SCHEMA),
        )

    baseline = fingerprint(refit())
    # Two independent refits of the same attempt, as a genuine retry would run.
    assert fingerprint(refit()) == baseline

    retuned = dict(_final_refit_factories())
    retuned[selected] = lambda: BundleSpyModel(
        offset=0.9, family="poisson", diagnostics=direct_diagnostics("poisson")
    )
    reseeded = replace(
        result,
        freeze=replace(result.freeze, master_seed=result.freeze.master_seed + 1),
        provenance=replace(result.provenance, master_seed=result.provenance.master_seed + 1),
    )
    other_prediction = tuple(
        replace(
            candidate,
            prediction_config=replace(
                candidate.prediction_config,
                n_predictive_draws=candidate.prediction_config.n_predictive_draws + 1,
            ),
        )
        if candidate.candidate_id == selected
        else candidate
        for candidate in spy_candidates()
    )
    evaluation_config = _evaluation_config()

    def with_other_dependency_versions(artifacts):
        return tuple(
            replace(
                artifact,
                evidence=replace(
                    artifact.evidence,
                    state_bundle={
                        **artifact.evidence.state_bundle,
                        "dependency_versions": {"numpy": "0.0.0"},
                    },
                ),
            )
            for artifact in artifacts
        )

    variants = {
        "refit model configuration": fingerprint(refit(retuned)),
        "dependency versions": fingerprint(with_other_dependency_versions(refit())),
        "master seed": fingerprint(
            refit(cv_result=reseeded),
            freeze=reseeded.freeze.with_cross_family_selection(_stub_decision(reseeded)),
        ),
        "prediction settings": fingerprint(refit(), candidates=other_prediction),
        "evaluation settings": fingerprint(
            refit(),
            evaluation_config=replace(
                evaluation_config,
                bootstrap_replicates=evaluation_config.bootstrap_replicates + 1,
            ),
        ),
        "schema": fingerprint(
            refit(), schema=replace(DEFAULT_MODELING_SCHEMA, schema_version="other")
        ),
    }
    for label, value in variants.items():
        assert value != baseline, label


def test_real_models_record_their_constructor_configuration() -> None:
    """The fingerprint relies on each real model's record following its configuration."""
    from age_group_prediction.models import (
        BayesianConditionalModel,
        DirectCohortModel,
        IndependentTotalProbabilityModel,
    )

    direct = DirectCohortModel()
    independent = IndependentTotalProbabilityModel()
    bayesian = BayesianConditionalModel()
    changed_models = {
        direct: DirectCohortModel(
            direct_cohort_config=replace(
                direct.direct_cohort_config,
                bootstrap_replicates=direct.direct_cohort_config.bootstrap_replicates + 1,
            )
        ),
        independent: IndependentTotalProbabilityModel(
            independent_config=replace(
                independent.independent_config,
                bootstrap_replicates=independent.independent_config.bootstrap_replicates + 1,
            )
        ),
        bayesian: BayesianConditionalModel(
            bayesian_config=replace(bayesian.bayesian_config, active_profile="full")
        ),
    }
    for original, changed in changed_models.items():
        record = original.configuration_record()
        json.dumps(record, allow_nan=False)
        assert record == type(original)().configuration_record()
        assert changed.configuration_record() != record
    assert DirectCohortModel(default_rng_seed=7).configuration_record() != (
        direct.configuration_record()
    )
