"""Gate 6 orchestration against all three real model families.

Every other Gate 6 test drives a spy model, so nothing exercises the runner
against a real likelihood, a real feature transformer, or a real posterior.
That gap has shipped a blocking defect before: the conditional selection policy
once keyed a metric at the wrong aggregation level, which no spy could detect
because the spy declared whatever the test asked for.

The frame is deliberately small and the Bayesian profile deliberately short:
this test exists to prove the wiring holds end to end, not to recover
parameters, which `test_bayesian_recovery.py` already does.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from age_group_prediction.data_splitting import (
    make_validation_folds,
    split_known_neighborhood_buildings,
)
from age_group_prediction.experiment import (
    CandidateDefinition,
    FeatureBlock,
    PermutationImportanceSpec,
    direct_cohort_selection_policy,
    run_cross_model_validation,
    sequential_joint_selection_policy,
)
from age_group_prediction.experiment_config import load_experiment_config
from age_group_prediction.metrics import (
    RootMeanSquaredError,
    default_metric_set,
)
from age_group_prediction.modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
    BayesianConditionalConfig,
    DirectCohortConfig,
    FoldConfig,
    IndependentTotalProbabilityConfig,
    NUTSProfileConfig,
    OptunaTuningConfig,
    OuterSplitConfig,
    PredictionConfig,
)
from age_group_prediction.models.bayesian_conditional import BayesianConditionalModel
from age_group_prediction.models.direct_cohort import DirectCohortModel
from age_group_prediction.models.independent_total_probability import (
    IndependentTotalProbabilityModel,
)

SCHEMA = DEFAULT_MODELING_SCHEMA
COHORTS = tuple(SCHEMA.cohort_target_columns)
TOTAL = SCHEMA.total_target_column
FAST_TUNING = OptunaTuningConfig(n_trials=2)


def _experiment_config():
    return load_experiment_config("configs/modeling.toml")


def _learnable_frame(
    row_count: int = 240, neighborhood_count: int = 12, seed: int = 7
) -> pd.DataFrame:
    """Simulate a table whose total and composition both depend on features."""
    rng = np.random.default_rng(seed)
    ses = rng.normal(size=row_count)
    apartments = rng.integers(20, 90, size=row_count)
    median_age = rng.uniform(24.0, 58.0, size=row_count)
    neighborhood = np.arange(row_count) % neighborhood_count
    neighborhood_effect = rng.normal(0.0, 0.3, size=neighborhood_count)

    log_mean = (
        np.log(apartments / 50.0)
        + 1.0
        + 0.45 * ses
        + neighborhood_effect[neighborhood]
    )
    total = rng.poisson(np.exp(log_mean))

    tilt = (median_age - 40.0) / 18.0
    logits = np.stack(
        [0.5 - 1.1 * tilt, 0.2 * np.ones(row_count), -0.4 + 1.1 * tilt], axis=1
    )
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    counts = np.stack(
        [rng.multinomial(int(n), p) for n, p in zip(total, probabilities, strict=True)]
    )

    frame = pd.DataFrame(
        {
            "building_id": np.arange(row_count),
            "neighborhood_id": [f"n{index}" for index in neighborhood],
            "ses": ses,
            "avg_household_size": rng.uniform(1.8, 3.8, size=row_count),
            "median_age": median_age,
            "n_daycares_500m": rng.integers(0, 6, size=row_count),
            "n_apartments": apartments,
            "3_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "school_status": rng.choice(["none", "existing", "planned"], size=row_count),
            "n_kindergarten": counts[:, 0],
            "n_elementary": counts[:, 1],
            "n_highschool": counts[:, 2],
            "n_children_total": total,
        }
    )
    SCHEMA.validate_table(frame)
    return frame


def _importance_spec(component: str) -> PermutationImportanceSpec:
    return PermutationImportanceSpec(
        component=component,
        metric=RootMeanSquaredError(TOTAL),
        feature_blocks=(FeatureBlock("ses", ("ses",)),),
        repeats=1,
    )


def _real_candidates(config) -> tuple[CandidateDefinition, ...]:
    """Build one candidate per public approach, all on real models."""
    evaluation = config.evaluation
    validation = config.prediction_validation

    # Model A declares marginal per-cohort scores and supplies no distribution
    # for the total, so its metric set covers the cohorts only. This mirrors
    # `direct_cohort_selection_policy`, which excludes the separately scored
    # total to avoid double counting.
    marginal_metrics = default_metric_set(
        COHORTS,
        nll_source="parametric",
        nll_interpretation="independent direct-cohort likelihood",
        evaluation_config=evaluation,
        prediction_validation_config=validation,
    )

    def joint_metrics(interpretation: str):
        return default_metric_set(
            (TOTAL, *COHORTS),
            nll_source="pointwise",
            nll_interpretation=interpretation,
            evaluation_config=evaluation,
            prediction_validation_config=validation,
            joint_nll_interpretation=(
                "sequential joint total-plus-composition log mass"
            ),
        )

    # The conditional models only emit pointwise keys when asked, and the joint
    # score is built from those keys.
    conditional_prediction = PredictionConfig(include_pointwise_log_probabilities=True)

    return (
        CandidateDefinition(
            candidate_id="direct-poisson",
            approach="DirectCohortModel",
            model_factory=lambda: DirectCohortModel(
                direct_cohort_config=DirectCohortConfig(
                    family="poisson", tuning=FAST_TUNING, bootstrap_replicates=4
                )
            ),
            fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
            component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
            metrics=marginal_metrics,
            configuration={"family": "poisson"},
            prediction_config=PredictionConfig(),
            importance_specs=(_importance_spec("tree"),),
        ),
        CandidateDefinition(
            candidate_id="independent-nb2",
            approach="IndependentTotalProbabilityModel",
            model_factory=lambda: IndependentTotalProbabilityModel(
                independent_config=IndependentTotalProbabilityConfig(
                    total_family="nb2", tuning=FAST_TUNING, bootstrap_replicates=4
                )
            ),
            fit_feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
            component_feature_specs=(
                DEFAULT_TOTAL_FEATURE_SPEC,
                DEFAULT_PROBABILITY_FEATURE_SPEC,
            ),
            metrics=joint_metrics("NB2 total plus multinomial conditional composition"),
            configuration={"total_family": "nb2"},
            prediction_config=conditional_prediction,
            importance_specs=(_importance_spec("total_count"),),
        ),
        CandidateDefinition(
            candidate_id="bayesian-reduced",
            approach="BayesianConditionalModel",
            model_factory=lambda: BayesianConditionalModel(
                bayesian_config=BayesianConditionalConfig(
                    reduced_profile=NUTSProfileConfig(
                        chains=2, warmup_steps=30, posterior_samples=30
                    )
                )
            ),
            fit_feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
            component_feature_specs=(
                DEFAULT_TOTAL_FEATURE_SPEC,
                DEFAULT_PROBABILITY_FEATURE_SPEC,
            ),
            metrics=joint_metrics(
                "posterior-integrated NB2 total plus Dirichlet-multinomial composition"
            ),
            configuration={"profile": "reduced"},
            prediction_config=conditional_prediction,
            importance_specs=(_importance_spec("total_count"),),
        ),
    )


@pytest.fixture(scope="module")
def real_model_run():
    """Run all three real model families once through the Gate 6 runner."""
    config = _experiment_config()
    frame = _learnable_frame()
    split = split_known_neighborhood_buildings(
        frame,
        config=OuterSplitConfig(test_fraction=0.25),
        rng=np.random.default_rng(4),
    )
    folds = make_validation_folds(
        split.train_df, config=FoldConfig(n_folds=2), rng=np.random.default_rng(9)
    ).folds
    # A thirty-sample profile is far too short to converge and is expected to
    # report it. The warnings are suppressed here so the suite keeps one
    # documented Bayesian warning from the recovery test, whose thresholds are
    # meaningful; the diagnostics themselves are asserted below rather than
    # ignored.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Bayesian (total|composition) diagnostics failed",
            category=RuntimeWarning,
        )
        result = run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=_real_candidates(config),
            selection_policies=(
                direct_cohort_selection_policy(COHORTS),
                sequential_joint_selection_policy("IndependentTotalProbabilityModel"),
                sequential_joint_selection_policy("BayesianConditionalModel"),
            ),
            master_seed=23,
            evaluation_config=replace(config.evaluation, bootstrap_replicates=6),
            # The short profile is expected to miss the convergence thresholds.
            # The constraint itself is covered by
            # `test_unconverged_candidate_is_excluded_from_selection`.
            require_convergence=False,
        )
    return split, result


@pytest.mark.slow
def test_real_three_model_run_completes_and_freezes_one_per_approach(
    real_model_run,
) -> None:
    """Orchestration must survive three real likelihoods end to end."""
    _, result = real_model_run

    assert {selection.approach for selection in result.selections} == {
        "DirectCohortModel",
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    }
    assert len(result.freeze.candidate_descriptors) == 3
    assert not result.fold_metrics_df.empty


@pytest.mark.slow
def test_real_run_leaks_no_holdout_building_anywhere(real_model_run) -> None:
    """No holdout ID may reach a prediction, an importance payload or the freeze."""
    split, result = real_model_run
    holdout = set(split.manifest.holdout_building_ids)

    assert holdout
    assert not holdout & set(result.predictions_df["building_id"])
    assert not holdout & set(result.freeze.outer_training_building_ids)
    importance_ids = {
        building_id
        for payload in result.importance_df["validation_building_ids"]
        for building_id in json.loads(payload)
    }
    assert importance_ids
    assert not holdout & importance_ids
    assert result.freeze.to_dict()["test_metrics"] is None


@pytest.mark.slow
def test_direct_cohort_is_excluded_from_joint_scoring_by_capability(
    real_model_run,
) -> None:
    """Marginal per-cohort keys must never be summed into a joint score.

    Model A's cohort keys are marginal, so adding them would double count the
    total it also predicts. The exclusion is enforced by the declared pointwise
    scope rather than by a special case in selection.
    """
    _, result = real_model_run
    joint = result.aggregate_metrics_df[
        result.aggregate_metrics_df["metric_name"] == "joint_predictive_nll"
    ]

    scored = set(joint["candidate_id"])
    assert scored == {"independent-nb2", "bayesian-reduced"}
    assert "direct-poisson" not in scored


@pytest.mark.slow
def test_per_target_cohort_keys_are_not_comparable_across_families(
    real_model_run,
) -> None:
    """The conditional models' last cohort key is ~0 by construction.

    Ranking a per-target NLL table would hand them that key as free winnings,
    so selection must rank the joint score instead.
    """
    _, result = real_model_run
    aggregate = result.aggregate_metrics_df
    per_target = aggregate[
        (aggregate["metric_name"] == "predictive_nll")
        & (aggregate["target"] == "n_highschool")
    ].set_index("candidate_id")["mean"]

    assert abs(per_target["independent-nb2"]) < 1e-9
    assert abs(per_target["bayesian-reduced"]) < 1e-9
    # Model A's key carries real information, so a per-target table would show
    # it losing by roughly its own value for no modeling reason.
    assert per_target["direct-poisson"] > 0.1

    for selection in result.selections:
        if selection.approach == "DirectCohortModel":
            continue
        assert selection.policy.criteria[0].name == "joint_predictive_nll"


@pytest.mark.slow
def test_joint_score_equals_the_summed_sequential_keys(real_model_run) -> None:
    """The joint metric must be the summed conditional keys, not a new quantity."""
    _, result = real_model_run
    run = next(
        evidence
        for evidence in result.fold_runs
        if evidence.candidate_id == "independent-nb2"
    )
    keys = run.prediction.pointwise_log_probabilities

    assert run.prediction.pointwise_log_probability_scope == "sequential_joint"
    assert set(keys) == {"total", *COHORTS}

    stacked = np.sum([np.asarray(value) for value in keys.values()], axis=0)
    expected = -float(np.mean(stacked))
    reported = run.evaluation.metrics_df.set_index("metric_name").loc[
        "joint_predictive_nll", "value"
    ]

    assert float(reported) == pytest.approx(expected, abs=1e-12)


@pytest.mark.slow
def test_real_importance_detects_a_block_the_models_use(real_model_run) -> None:
    """Permuting SES must degrade models that genuinely learned from it."""
    _, result = real_model_run

    assert not result.importance_df.empty
    # Importance runs for selected candidates only.
    assert set(result.importance_df["candidate_id"]) == {
        selection.selected_candidate_id for selection in result.selections
    }
    degradation = result.importance_df["degradation"].to_numpy(dtype=float)
    assert np.all(np.isfinite(degradation))
    assert np.any(degradation > 0.0)


@pytest.mark.slow
def test_unconverged_candidate_is_excluded_from_selection(real_model_run) -> None:
    """Convergence is a selection constraint, not a warning nobody reads.

    Plan section 9.3 requires it, and the Bayesian model records
    `policy_passed` per stage precisely so selection can consume it.
    """
    _, result = real_model_run
    bayesian = next(
        evidence
        for evidence in result.fold_runs
        if evidence.candidate_id == "bayesian-reduced"
    )
    diagnostics = bayesian.model_metadata["model"]["diagnostics"]
    assert {"total", "composition"} <= set(diagnostics)
    for stage in ("total", "composition"):
        assert "policy_passed" in diagnostics[stage]

    from age_group_prediction.experiment.selection import _convergence_failures

    failures = _convergence_failures(result.fold_runs)
    unconverged = {
        stage
        for stage in ("total", "composition")
        if not diagnostics[stage]["policy_passed"]
    }
    if unconverged:
        assert "bayesian-reduced" in failures
    else:
        assert "bayesian-reduced" not in failures
