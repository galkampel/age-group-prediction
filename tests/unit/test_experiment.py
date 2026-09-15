"""Tests for cross-model validation, selection, and interpretation."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from age_group_prediction.data_splitting import (
    FoldConfig,
    OuterSplitConfig,
    ValidationFold,
    make_validation_folds,
    split_known_neighborhood_buildings,
)
from age_group_prediction.experiment import (
    CandidateDefinition,
    FeatureBlock,
    MetricReference,
    PermutationImportanceSpec,
    SelectionCriterion,
    SelectionPolicy,
    build_canonical_candidate_registry,
    direct_cohort_selection_policy,
    enumerate_feature_specs,
    run_cross_model_validation,
    sequential_joint_selection_policy,
    validate_experiment_partitions,
)
from age_group_prediction.experiment_config import load_experiment_config
from age_group_prediction.metrics import (
    CompositionLogLoss,
    JointPredictiveNegativeLogLikelihood,
    MeanAbsoluteError,
    ParametricPredictiveNegativeLogLikelihood,
    RootMeanSquaredError,
)
from age_group_prediction.modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
    EvaluationConfig,
    PredictionConfig,
)
from age_group_prediction.models.base import BaseAgeGroupModel
from age_group_prediction.results import ParametricDistributionSpec, PredictionResult


class _SpyModel(BaseAgeGroupModel):
    """Small contract model that records fold-local fit and prediction rows."""

    def __init__(
        self,
        *,
        offset: float,
        fit_log: list[tuple[int, ...]],
        predict_log: list[tuple[int, ...]],
        calibration: dict[str, object] | None = None,
        prediction_frame_log: list[pd.DataFrame] | None = None,
    ) -> None:
        super().__init__()
        self.offset = offset
        self.fit_log = fit_log
        self.predict_log = predict_log
        self.calibration = calibration
        self.prediction_frame_log = prediction_frame_log

    def _reset_model_state(self) -> None:
        pass

    def _fit_model(self, *, train_df, features, log_exposure, rng) -> None:
        del features, log_exposure, rng
        self.fit_log.append(tuple(train_df["building_id"]))

    def _predict_model(
        self,
        *,
        eval_df,
        features,
        log_exposure,
        prediction_config,
        rng,
    ) -> PredictionResult:
        del features, log_exposure, prediction_config, rng
        self.predict_log.append(tuple(eval_df["building_id"]))
        if self.prediction_frame_log is not None:
            self.prediction_frame_log.append(eval_df.copy(deep=True))
        total = np.clip(eval_df["ses"].to_numpy(dtype=float) + self.offset, 0.01, None)
        probabilities = np.array([0.2, 0.5, 0.3])
        return PredictionResult.from_means(
            building_ids=eval_df["building_id"].to_numpy(),
            cohort_names=DEFAULT_MODELING_SCHEMA.cohort_target_columns,
            total_mean=total,
            cohort_means=total[:, None] * probabilities,
        )

    def _get_model_metadata(self):
        return {
            "implementation_version": "test",
            "likelihood": "deterministic test means",
            "parameterization": "offset",
            "hyperparameters": {"offset": self.offset},
            "priors": None,
            "calibration": self.calibration,
            "dependency_versions": {},
            "uncertainty_method": "none",
            "diagnostics": {},
        }


class _LateReorderingSpyModel(_SpyModel):
    """Preserve initial fold order, then violate it during importance."""

    def _predict_model(self, **kwargs) -> PredictionResult:
        result = super()._predict_model(**kwargs)
        if len(self.predict_log) < 2:
            return result
        return PredictionResult.from_means(
            building_ids=result.building_ids[::-1],
            cohort_names=result.cohort_names,
            total_mean=result.total_mean,
            cohort_means=result.cohort_means,
        )


def _modeling_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for neighborhood_id in range(3):
        for local_id in range(4):
            kindergarten = local_id % 2
            elementary = 1 + (local_id % 3)
            highschool = (local_id + 1) % 2
            rows.append(
                {
                    "building_id": neighborhood_id * 10 + local_id,
                    "neighborhood_id": neighborhood_id,
                    "ses": 0.2 + 0.1 * local_id,
                    "avg_household_size": 2.0 + 0.1 * neighborhood_id,
                    "median_age": 30.0 + local_id,
                    "n_daycares_500m": local_id,
                    "n_apartments": 20 + local_id,
                    "3_rooms_share": 0.2,
                    "4_rooms_share": 0.3,
                    "5_rooms_share": 0.25,
                    "school_status": ("none", "existing", "planned")[
                        neighborhood_id
                    ],
                    "n_kindergarten": kindergarten,
                    "n_elementary": elementary,
                    "n_highschool": highschool,
                    "n_children_total": kindergarten + elementary + highschool,
                }
            )
    table = pd.DataFrame(rows, columns=DEFAULT_MODELING_SCHEMA.table_columns)
    DEFAULT_MODELING_SCHEMA.validate_table(table)
    return table


def test_partition_validation_rejects_holdout_rows_before_modeling() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(
        table,
        config=OuterSplitConfig(test_fraction=0.25),
        rng=np.random.default_rng(4),
    )
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    contaminated_validation = pd.concat(
        [folds[0].validation_df.iloc[:-1], split.test_df.iloc[[0]]],
        ignore_index=True,
    )
    contaminated_folds = (
        ValidationFold(
            fold_index=folds[0].fold_index,
            fit_df=folds[0].fit_df,
            validation_df=contaminated_validation,
        ),
        folds[1],
    )

    with pytest.raises(ValueError, match="partition the outer training IDs"):
        validate_experiment_partitions(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=contaminated_folds,
        )


def test_run_rejects_noncanonical_outer_training_before_factory_call() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(split.train_df, rng=np.random.default_rng(9)).folds
    factory_calls = 0

    def factory() -> BaseAgeGroupModel:
        nonlocal factory_calls
        factory_calls += 1
        return _SpyModel(offset=0.0, fit_log=[], predict_log=[])

    candidate = CandidateDefinition(
        candidate_id="never-built",
        approach="DirectCohortModel",
        model_factory=factory,
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(MeanAbsoluteError("n_children_total"),),
        configuration={},
    )
    contaminated = pd.concat(
        [split.train_df.iloc[:-1], split.test_df.iloc[[0]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="match the split manifest"):
        run_cross_model_validation(
            contaminated,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=(candidate,),
            selection_policies=(_policy(),),
            master_seed=2,
            required_approaches=("DirectCohortModel",),
        )

    assert factory_calls == 0


def _candidate(
    candidate_id: str,
    *,
    offset: float,
    fit_log: list[tuple[int, ...]],
    predict_log: list[tuple[int, ...]],
) -> CandidateDefinition:
    metric = MeanAbsoluteError("n_children_total")
    return CandidateDefinition(
        candidate_id=candidate_id,
        approach="DirectCohortModel",
        model_factory=lambda: _SpyModel(
            offset=offset,
            fit_log=fit_log,
            predict_log=predict_log,
        ),
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(metric,),
        configuration={"offset": offset},
        prediction_config=PredictionConfig(),
        importance_specs=(
            PermutationImportanceSpec(
                component="tree",
                metric=metric,
                feature_blocks=(FeatureBlock("ses", ("ses",)),),
                repeats=2,
            ),
        ),
    )


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        approach="DirectCohortModel",
        criteria=(
            SelectionCriterion(
                name="total_mae",
                metric_references=(MetricReference("mae", "n_children_total"),),
                optimization_direction="minimize",
            ),
        ),
        likelihood_comparability="Point accuracy within direct-cohort candidates.",
    )


def test_candidates_share_folds_and_freeze_training_only_selection() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(
        table,
        config=OuterSplitConfig(test_fraction=0.25),
        rng=np.random.default_rng(4),
    )
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    fit_log_a: list[tuple[int, ...]] = []
    fit_log_b: list[tuple[int, ...]] = []
    predict_log_a: list[tuple[int, ...]] = []
    predict_log_b: list[tuple[int, ...]] = []
    candidates = (
        _candidate(
            "direct-low-offset",
            offset=0.0,
            fit_log=fit_log_a,
            predict_log=predict_log_a,
        ),
        _candidate(
            "direct-high-offset",
            offset=10.0,
            fit_log=fit_log_b,
            predict_log=predict_log_b,
        ),
    )

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=candidates,
        selection_policies=(_policy(),),
        master_seed=17,
        required_approaches=("DirectCohortModel",),
    )

    assert fit_log_a == fit_log_b == [tuple(fold.fit_df["building_id"]) for fold in folds]
    expected_validation = [tuple(fold.validation_df["building_id"]) for fold in folds]
    assert predict_log_a[:2] == expected_validation
    assert predict_log_b[:2] == expected_validation
    assert result.selections[0].selected_candidate_id == "direct-low-offset"
    reason = result.selections[0].rejection_reasons["direct-high-offset"]
    assert set(result.selections[0].rejection_reasons) == {"direct-high-offset"}
    assert reason.startswith(
        "Ranked below 'direct-low-offset' under the declared ordered "
        "training-only selection criteria: "
    )
    # The deciding criterion and its margin must be readable from the reason
    # alone, so a candidate that lost by a hair is distinguishable from one
    # that lost outright.
    assert "total_mae" in reason
    assert "margin 4.9" in reason
    assert result.freeze.to_dict()["test_metrics"] is None
    assert result.freeze.to_dict()["candidate_descriptors"][0]["metrics"][0] == {
        "target": "n_children_total",
        "name": "mae",
        "required_capability": "means",
        "optimization_direction": "minimize",
        "aggregation_level": "building",
    }
    assert set(result.predictions_df["building_id"]).isdisjoint(
        split.manifest.holdout_building_ids
    )
    assert set(result.importance_df["candidate_id"]) == {"direct-low-offset"}
    importance_ids = {
        building_id
        for encoded in result.importance_df["validation_building_ids"]
        for building_id in json.loads(encoded)
    }
    assert importance_ids.isdisjoint(split.manifest.holdout_building_ids)


def test_cross_validation_is_reproducible_and_candidate_order_invariant() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds

    def run(order: tuple[float, float]):
        candidates = tuple(
            _candidate(
                f"direct-{offset}",
                offset=offset,
                fit_log=[],
                predict_log=[],
            )
            for offset in order
        )
        return run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=candidates,
            selection_policies=(_policy(),),
            master_seed=23,
            required_approaches=("DirectCohortModel",),
        )

    first = run((0.0, 3.0))
    second = run((3.0, 0.0))

    assert first.freeze.to_dict() == second.freeze.to_dict()
    pd.testing.assert_frame_equal(first.fold_metrics_df, second.fold_metrics_df)
    pd.testing.assert_frame_equal(first.importance_df, second.importance_df)


def test_policy_helpers_make_likelihood_comparability_explicit() -> None:
    direct = direct_cohort_selection_policy(
        DEFAULT_MODELING_SCHEMA.cohort_target_columns
    )
    direct_targets = {
        reference.target
        for reference in direct.criteria[0].metric_references
    }
    assert direct_targets == set(DEFAULT_MODELING_SCHEMA.cohort_target_columns)
    assert "n_children_total" not in direct_targets
    assert "double counting" in direct.likelihood_comparability

    conditional = sequential_joint_selection_policy(
        "IndependentTotalProbabilityModel"
    )
    assert conditional.criteria[0].metric_references == (
        MetricReference("joint_predictive_nll", "joint"),
    )
    assert "sequential_joint" in conditional.likelihood_comparability


def test_candidate_records_calibration_and_declared_feature_blocks() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    calibration = {
        "temperature": 1.2,
        "retained": True,
        "raw_weighted_nll": 1.1,
        "calibrated_weighted_nll": 1.0,
    }
    metric = MeanAbsoluteError("n_children_total")
    candidate = CandidateDefinition(
        candidate_id="independent-calibrated",
        approach="IndependentTotalProbabilityModel",
        model_factory=lambda: _SpyModel(
            offset=0.0,
            fit_log=[],
            predict_log=[],
            calibration=calibration,
        ),
        fit_feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        component_feature_specs=(
            DEFAULT_TOTAL_FEATURE_SPEC,
            DEFAULT_PROBABILITY_FEATURE_SPEC,
        ),
        metrics=(metric,),
        configuration={"calibration_policy": "evidence_gated_temperature"},
        importance_specs=(
            PermutationImportanceSpec(
                component="total_count",
                metric=metric,
                feature_blocks=(FeatureBlock("context", ("ses", "median_age")),),
                repeats=1,
            ),
        ),
    )
    policy = SelectionPolicy(
        approach="IndependentTotalProbabilityModel",
        criteria=(
            SelectionCriterion(
                "total_mae",
                (MetricReference("mae", "n_children_total"),),
                "minimize",
            ),
        ),
        likelihood_comparability="Test-only point comparison.",
    )

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(policy,),
        master_seed=31,
        required_approaches=("IndependentTotalProbabilityModel",),
    )

    assert result.calibration_df.loc[0, "status"] == "recorded"
    assert json.loads(result.calibration_df.loc[0, "calibration"]) == calibration
    descriptor = result.freeze.to_dict()["candidate_descriptors"][0]
    assert descriptor["importance"][0]["feature_blocks"] == [
        {"name": "context", "columns": ("ses", "median_age")}
    ]
    assert result.importance_summary_df.loc[0, "observation_count"] == 2


def _run_with_importance_block(block: FeatureBlock) -> None:
    """Run one candidate whose only importance block is the one supplied.

    Importance specs are validated by the runner's candidate registry rather
    than by ``CandidateDefinition`` construction, so the block must be carried
    into an actual run for the guard to fire.
    """
    metric = MeanAbsoluteError("n_children_total")
    candidate = CandidateDefinition(
        candidate_id="invalid-importance",
        approach="DirectCohortModel",
        model_factory=lambda: _SpyModel(offset=0.0, fit_log=[], predict_log=[]),
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(metric,),
        configuration={},
        importance_specs=(
            PermutationImportanceSpec(
                component="tree",
                metric=metric,
                feature_blocks=(block,),
            ),
        ),
    )
    split, folds = _split_and_folds()
    run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(_policy(),),
        master_seed=1,
        required_approaches=("DirectCohortModel",),
    )


def test_importance_rejects_forbidden_columns() -> None:
    """A target column must never be permutable as a feature."""
    with pytest.raises(ValueError, match="forbidden columns"):
        _run_with_importance_block(FeatureBlock("target", ("n_children_total",)))


def test_importance_rejects_columns_outside_their_component_spec() -> None:
    """A block naming a column its own component never fits is rejected."""
    with pytest.raises(ValueError, match="outside its component spec"):
        _run_with_importance_block(FeatureBlock("stray", ("not_a_feature",)))


def test_bayesian_selection_must_use_frozen_independent_feature_specs() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    metric = MeanAbsoluteError("n_children_total")

    def conditional_candidate(candidate_id, approach, probability_spec):
        return CandidateDefinition(
            candidate_id=candidate_id,
            approach=approach,
            model_factory=lambda: _SpyModel(offset=0.0, fit_log=[], predict_log=[]),
            fit_feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
            component_feature_specs=(DEFAULT_TOTAL_FEATURE_SPEC, probability_spec),
            metrics=(metric,),
            configuration={},
        )

    policies = tuple(
        SelectionPolicy(
            approach=approach,
            criteria=(
                SelectionCriterion(
                    "total_mae",
                    (MetricReference("mae", "n_children_total"),),
                    "minimize",
                ),
            ),
            likelihood_comparability="Test-only point comparison.",
        )
        for approach in (
            "IndependentTotalProbabilityModel",
            "BayesianConditionalModel",
        )
    )

    with pytest.raises(ValueError, match="frozen age_probability feature spec"):
        run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=(
                conditional_candidate(
                    "independent",
                    "IndependentTotalProbabilityModel",
                    DEFAULT_PROBABILITY_FEATURE_SPEC,
                ),
                conditional_candidate(
                    "bayesian",
                    "BayesianConditionalModel",
                    replace(DEFAULT_PROBABILITY_FEATURE_SPEC, ses_form="quadratic"),
                ),
            ),
            selection_policies=policies,
            master_seed=1,
            required_approaches=(
                "IndependentTotalProbabilityModel",
                "BayesianConditionalModel",
            ),
        )


def test_bootstrap_evidence_includes_cross_validation_intervals() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    candidate = _candidate(
        "direct",
        offset=0.0,
        fit_log=[],
        predict_log=[],
    )
    evaluation_config = EvaluationConfig(
        bootstrap_replicates=4,
        confidence_level=0.8,
        neighborhood_id_column="neighborhood_id",
        interval_method="percentile",
        max_failed_fraction=0.5,
        poisson_minimum_mean=1e-9,
        pit_histogram_bins=10,
    )

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(_policy(),),
        master_seed=8,
        evaluation_config=evaluation_config,
        required_approaches=("DirectCohortModel",),
    )

    assert set(result.bootstrap_intervals_df["scope"]) == {
        "fold",
        "cross_validation",
    }


def test_all_public_approaches_receive_identical_fixed_folds() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    metric = MeanAbsoluteError("n_children_total")
    fit_logs: dict[str, list[tuple[int, ...]]] = {
        approach: []
        for approach in (
            "DirectCohortModel",
            "IndependentTotalProbabilityModel",
            "BayesianConditionalModel",
        )
    }

    def candidate(approach):
        specs = (
            (DEFAULT_TREE_FEATURE_SPEC,)
            if approach == "DirectCohortModel"
            else (DEFAULT_TOTAL_FEATURE_SPEC, DEFAULT_PROBABILITY_FEATURE_SPEC)
        )
        return CandidateDefinition(
            candidate_id=approach.lower(),
            approach=approach,
            model_factory=lambda: _SpyModel(
                offset=0.0,
                fit_log=fit_logs[approach],
                predict_log=[],
            ),
            fit_feature_spec=specs[0],
            component_feature_specs=specs,
            metrics=(metric,),
            configuration={},
        )

    candidates = tuple(candidate(approach) for approach in fit_logs)
    policies = tuple(
        SelectionPolicy(
            approach=approach,
            criteria=(
                SelectionCriterion(
                    "total_mae",
                    (MetricReference("mae", "n_children_total"),),
                    "minimize",
                ),
            ),
            likelihood_comparability="Common point metric for fold-identity test.",
        )
        for approach in fit_logs
    )

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=candidates,
        selection_policies=policies,
        master_seed=7,
    )

    expected = [tuple(fold.fit_df["building_id"]) for fold in folds]
    assert all(fit_log == expected for fit_log in fit_logs.values())
    assert {selection.approach for selection in result.selections} == set(fit_logs)


def test_grouped_importance_permutation_preserves_nonfeature_rows() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    prediction_frames: list[pd.DataFrame] = []
    metric = MeanAbsoluteError("n_children_total")
    candidate = CandidateDefinition(
        candidate_id="direct-grouped-importance",
        approach="DirectCohortModel",
        model_factory=lambda: _SpyModel(
            offset=0.0,
            fit_log=[],
            predict_log=[],
            prediction_frame_log=prediction_frames,
        ),
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(metric,),
        configuration={},
        importance_specs=(
            PermutationImportanceSpec(
                component="tree",
                metric=metric,
                feature_blocks=(FeatureBlock("context", ("ses", "median_age")),),
                repeats=1,
            ),
        ),
    )

    run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(_policy(),),
        master_seed=41,
        required_approaches=("DirectCohortModel",),
    )

    baseline = prediction_frames[-2]
    permuted = prediction_frames[-1]
    unchanged = [
        "building_id",
        "neighborhood_id",
        *DEFAULT_MODELING_SCHEMA.target_columns,
    ]
    pd.testing.assert_frame_equal(baseline.loc[:, unchanged], permuted.loc[:, unchanged])
    assert sorted(zip(permuted["ses"], permuted["median_age"], strict=True)) == sorted(
        zip(baseline["ses"], baseline["median_age"], strict=True)
    )
    assert set(permuted["building_id"]).isdisjoint(
        split.manifest.holdout_building_ids
    )


def test_importance_rejects_prediction_row_reordering() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    metric = MeanAbsoluteError("n_children_total")
    candidate = CandidateDefinition(
        candidate_id="late-reordering",
        approach="DirectCohortModel",
        model_factory=lambda: _LateReorderingSpyModel(
            offset=0.0,
            fit_log=[],
            predict_log=[],
        ),
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(metric,),
        configuration={},
        importance_specs=(
            PermutationImportanceSpec(
                component="tree",
                metric=metric,
                feature_blocks=(FeatureBlock("ses", ("ses",)),),
                repeats=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="preserve validation-row order"):
        run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=(candidate,),
            selection_policies=(_policy(),),
            master_seed=5,
            required_approaches=("DirectCohortModel",),
        )


def test_selection_policy_metrics_are_validated_before_factory_call() -> None:
    table = _modeling_table()
    split = split_known_neighborhood_buildings(table, rng=np.random.default_rng(4))
    folds = make_validation_folds(split.train_df, rng=np.random.default_rng(9)).folds
    factory_calls = 0

    def factory() -> BaseAgeGroupModel:
        nonlocal factory_calls
        factory_calls += 1
        return _SpyModel(offset=0.0, fit_log=[], predict_log=[])

    candidate = CandidateDefinition(
        candidate_id="missing-policy-metrics",
        approach="DirectCohortModel",
        model_factory=factory,
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(MeanAbsoluteError("n_children_total"),),
        configuration={},
    )

    with pytest.raises(ValueError, match="does not declare selection metric"):
        run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=(candidate,),
            selection_policies=(
                direct_cohort_selection_policy(
                    DEFAULT_MODELING_SCHEMA.cohort_target_columns
                ),
            ),
            master_seed=6,
            required_approaches=("DirectCohortModel",),
        )

    assert factory_calls == 0

def _split_and_folds():
    table = _modeling_table()
    split = split_known_neighborhood_buildings(
        table,
        config=OuterSplitConfig(test_fraction=0.25),
        rng=np.random.default_rng(4),
    )
    folds = make_validation_folds(
        split.train_df,
        config=FoldConfig(n_folds=2),
        rng=np.random.default_rng(9),
    ).folds
    return split, folds


def test_shipped_policies_reference_declared_metric_aggregation_levels() -> None:
    """Policy helpers must key metrics the way the metric classes declare them."""
    declared = {
        (metric.name, metric.target, metric.aggregation_level)
        for metric in (
            JointPredictiveNegativeLogLikelihood(interpretation="joint"),
            CompositionLogLoss(),
            RootMeanSquaredError(target="n_children_total"),
            *(
                ParametricPredictiveNegativeLogLikelihood(
                    target=cohort,
                    interpretation="cohort",
                )
                for cohort in DEFAULT_MODELING_SCHEMA.cohort_target_columns
            ),
            *(
                RootMeanSquaredError(target=cohort)
                for cohort in DEFAULT_MODELING_SCHEMA.cohort_target_columns
            ),
        )
    }
    policies = (
        sequential_joint_selection_policy("IndependentTotalProbabilityModel"),
        sequential_joint_selection_policy("BayesianConditionalModel"),
        direct_cohort_selection_policy(DEFAULT_MODELING_SCHEMA.cohort_target_columns),
    )
    for policy in policies:
        for criterion in policy.criteria:
            for reference in criterion.metric_references:
                key = (
                    reference.metric_name,
                    reference.target,
                    reference.aggregation_level,
                )
                assert key in declared, f"{policy.approach} references unknown {key}"

    composition = next(
        reference
        for criterion in policies[0].criteria
        for reference in criterion.metric_references
        if reference.metric_name == "composition_log_loss"
    )
    assert composition.aggregation_level == "child"


def test_selection_criterion_direction_must_match_metric_direction() -> None:
    split, folds = _split_and_folds()
    factory_calls = 0

    def factory() -> BaseAgeGroupModel:
        nonlocal factory_calls
        factory_calls += 1
        return _SpyModel(offset=0.0, fit_log=[], predict_log=[])

    candidate = CandidateDefinition(
        candidate_id="inverted",
        approach="DirectCohortModel",
        model_factory=factory,
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(MeanAbsoluteError("n_children_total"),),
        configuration={},
    )
    inverted = SelectionPolicy(
        approach="DirectCohortModel",
        criteria=(
            SelectionCriterion(
                name="total_mae",
                metric_references=(MetricReference("mae", "n_children_total"),),
                optimization_direction="maximize",
            ),
        ),
        likelihood_comparability="Point accuracy.",
    )

    with pytest.raises(ValueError, match="which declares minimize"):
        run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=(candidate,),
            selection_policies=(inverted,),
            master_seed=3,
            required_approaches=("DirectCohortModel",),
        )

    assert factory_calls == 0


def test_master_seed_resolves_from_experiment_config_and_is_recorded() -> None:
    split, folds = _split_and_folds()
    fit_log: list[tuple[int, ...]] = []
    predict_log: list[tuple[int, ...]] = []
    candidate = _candidate(
        "seeded", offset=0.0, fit_log=fit_log, predict_log=predict_log
    )
    config = load_experiment_config("configs/modeling.toml")

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(_policy(),),
        experiment_config=config,
        required_approaches=("DirectCohortModel",),
    )

    freeze = result.freeze.to_dict()
    assert freeze["master_seed"] == config.randomness.default_seed
    assert freeze["master_seed_source"] == "experiment_config.randomness.default_seed"

    with pytest.raises(ValueError, match="never seeded by an implicit default"):
        run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=(candidate,),
            selection_policies=(_policy(),),
            required_approaches=("DirectCohortModel",),
        )


def test_bootstrap_evidence_records_its_seed_and_interval_basis() -> None:
    split, folds = _split_and_folds()
    candidate = _candidate("boot", offset=0.0, fit_log=[], predict_log=[])

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(_policy(),),
        master_seed=11,
        evaluation_config=EvaluationConfig(
            bootstrap_replicates=4,
            confidence_level=0.8,
            neighborhood_id_column="neighborhood_id",
            interval_method="percentile",
            max_failed_fraction=0.5,
            poisson_minimum_mean=1e-8,
            pit_histogram_bins=10,
        ),
        required_approaches=("DirectCohortModel",),
    )

    seeds = {
        run.bootstrap.metadata["default_seed"]
        for run in result.fold_runs
        if run.bootstrap is not None
    }
    assert seeds and None not in seeds

    intervals = result.bootstrap_intervals_df
    assert "aggregation_level" in intervals.columns
    basis = dict(
        zip(intervals["scope"], intervals["interval_basis"], strict=True)
    )
    assert basis["fold"] == "within_fold_neighborhood_cluster_bootstrap"
    assert (
        basis["cross_validation"] == "index_paired_replicate_mean_across_folds"
    )
    cross = intervals.loc[intervals["scope"] == "cross_validation"]
    assert bool(cross["assumes_independent_folds"].all())


def _table_with_singleton_neighborhood() -> pd.DataFrame:
    """Return the fixture plus a neighborhood holding exactly one building.

    Folds rotate, so every non-singleton training building is validated exactly
    once and a coverage table over the plain fixture is the constant 1. A
    building alone in its neighborhood is never validated, which is the only
    way this fixture can distinguish a real coverage count from a constant.
    """
    table = _modeling_table()
    singleton = table.iloc[[0]].copy()
    singleton["building_id"] = 99
    singleton["neighborhood_id"] = 3
    extended = pd.concat([table, singleton], ignore_index=True)
    DEFAULT_MODELING_SCHEMA.validate_table(extended)
    return extended


def test_fold_coverage_reports_unvalidated_and_repeated_buildings() -> None:
    table = _table_with_singleton_neighborhood()
    split = split_known_neighborhood_buildings(
        table, config=OuterSplitConfig(test_fraction=0.25), rng=np.random.default_rng(4)
    )
    folds = make_validation_folds(
        split.train_df, config=FoldConfig(n_folds=2), rng=np.random.default_rng(9)
    ).folds
    candidate = _candidate("coverage", offset=0.0, fit_log=[], predict_log=[])

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(_policy(),),
        master_seed=5,
        required_approaches=("DirectCohortModel",),
    )

    coverage = result.fold_coverage_df
    assert set(coverage["building_id"]) == set(split.manifest.training_building_ids)
    assert coverage["validation_fold_count"].max() <= len(folds)
    # A constant coverage column must not satisfy this test: the singleton
    # neighborhood contributes no out-of-fold evidence and has to show as zero.
    counts = dict(
        zip(coverage["building_id"], coverage["validation_fold_count"], strict=True)
    )
    assert counts[99] == 0
    assert set(counts.values()) == {0, 1}
    expected = {
        building_id: sum(
            building_id in set(fold.validation_df["building_id"]) for fold in folds
        )
        for building_id in split.manifest.training_building_ids
    }
    actual = dict(
        zip(coverage["building_id"], coverage["validation_fold_count"], strict=True)
    )
    assert actual == expected


def test_calibration_evidence_exposes_decision_fields() -> None:
    split, folds = _split_and_folds()
    calibration = {
        "method": "evidence-gated multiclass temperature scaling",
        "temperature": 1.0,
        "fitted_temperature": 1.4,
        "retained": False,
        "raw_weighted_nll": 1.2,
        "selected_weighted_nll": 1.2,
    }
    metric = MeanAbsoluteError("n_children_total")
    candidate = CandidateDefinition(
        candidate_id="calibrated",
        approach="IndependentTotalProbabilityModel",
        model_factory=lambda: _SpyModel(
            offset=0.0, fit_log=[], predict_log=[], calibration=calibration
        ),
        fit_feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        component_feature_specs=(
            DEFAULT_TOTAL_FEATURE_SPEC,
            DEFAULT_PROBABILITY_FEATURE_SPEC,
        ),
        metrics=(metric,),
        configuration={},
    )
    policy = SelectionPolicy(
        approach="IndependentTotalProbabilityModel",
        criteria=(
            SelectionCriterion(
                name="total_mae",
                metric_references=(MetricReference("mae", "n_children_total"),),
                optimization_direction="minimize",
            ),
        ),
        likelihood_comparability="Point accuracy.",
    )

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(policy,),
        master_seed=7,
        required_approaches=("IndependentTotalProbabilityModel",),
    )

    row = result.calibration_df.iloc[0]
    assert row["status"] == "recorded"
    # The retention decision must be readable without parsing the JSON payload.
    assert bool(row["retained"]) is False
    assert row["temperature"] == 1.0
    assert row["fitted_temperature"] == 1.4
    assert json.loads(row["calibration"]) == calibration


class _ParametricSpyModel(_SpyModel):
    """Spy model that declares an explicit predictive family."""

    def __init__(self, *, family: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.family = family

    def _predict_model(self, **kwargs) -> PredictionResult:
        result = super()._predict_model(**kwargs)
        spec = (
            ParametricDistributionSpec(family="normal", scale=1.0)
            if self.family == "normal"
            else ParametricDistributionSpec(family="poisson")
        )
        return PredictionResult.from_means(
            building_ids=result.building_ids,
            cohort_names=result.cohort_names,
            total_mean=result.total_mean,
            cohort_means=result.cohort_means,
            parametric_distributions={
                target: spec
                for target in ("total", *DEFAULT_MODELING_SCHEMA.cohort_target_columns)
            },
        )


def _parametric_candidate(candidate_id: str, family: str, role: str):
    metric = ParametricPredictiveNegativeLogLikelihood(
        target="n_children_total",
        interpretation=family,
    )
    return CandidateDefinition(
        candidate_id=candidate_id,
        approach="DirectCohortModel",
        model_factory=lambda: _ParametricSpyModel(
            family=family, offset=0.0, fit_log=[], predict_log=[]
        ),
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(metric,),
        configuration={"family": family},
        prediction_config=PredictionConfig(),
        selection_role=role,
    )


def _nll_policy() -> SelectionPolicy:
    return SelectionPolicy(
        approach="DirectCohortModel",
        criteria=(
            SelectionCriterion(
                name="total_nll",
                metric_references=(
                    MetricReference("predictive_nll", "n_children_total"),
                ),
                optimization_direction="minimize",
            ),
        ),
        likelihood_comparability="Discrete direct-cohort log masses.",
    )


def test_continuous_candidate_cannot_be_ranked_against_discrete_candidate() -> None:
    """Log densities and log masses are not on a common scale (section 9.1)."""
    split, folds = _split_and_folds()

    with pytest.raises(ValueError, match="not comparable to log masses"):
        run_cross_model_validation(
            split.train_df,
            split_manifest=split.manifest,
            validation_folds=folds,
            candidates=(
                _parametric_candidate("discrete", "poisson", "eligible"),
                _parametric_candidate("continuous", "normal", "eligible"),
            ),
            selection_policies=(_nll_policy(),),
            master_seed=8,
            required_approaches=("DirectCohortModel",),
        )


def test_continuous_candidate_is_allowed_as_diagnostic_comparator() -> None:
    split, folds = _split_and_folds()

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(
            _parametric_candidate("discrete", "poisson", "eligible"),
            _parametric_candidate("continuous", "normal", "diagnostic_comparator"),
        ),
        selection_policies=(_nll_policy(),),
        master_seed=8,
        required_approaches=("DirectCohortModel",),
    )

    selection = result.selections[0]
    assert selection.selected_candidate_id == "discrete"
    # The comparator is never ranked, so it is neither selected nor rejected.
    assert selection.rejected_candidate_ids == ()
    assert "continuous" in {run.candidate_id for run in result.fold_runs}


def test_same_measure_candidates_are_still_comparable() -> None:
    split, folds = _split_and_folds()

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(
            _parametric_candidate("poisson_a", "poisson", "eligible"),
            _parametric_candidate("poisson_b", "poisson", "eligible"),
        ),
        selection_policies=(_nll_policy(),),
        master_seed=8,
        required_approaches=("DirectCohortModel",),
    )

    assert len(result.selections[0].rejected_candidate_ids) == 1


def test_purpose_scoped_seeds_are_distinct_per_candidate_fold_and_operation() -> None:
    """Child seeds must separate every purpose, not merely be recorded.

    A single constant substituted for the whole derivation would still be
    reproducible and still be recorded, so only distinctness detects it.
    """
    split, folds = _split_and_folds()
    candidates = (
        _candidate("seeds-a", offset=0.0, fit_log=[], predict_log=[]),
        _candidate("seeds-b", offset=3.0, fit_log=[], predict_log=[]),
    )

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=candidates,
        selection_policies=(_policy(),),
        master_seed=23,
        required_approaches=("DirectCohortModel",),
    )

    observed = [
        (run.candidate_id, run.fold_identity.fold_index, operation, seed)
        for run in result.fold_runs
        for operation, seed in run.seeds.items()
    ]
    assert observed
    assert len({entry[3] for entry in observed}) == len(observed)

    # The same purpose under the same master seed must still reproduce.
    repeated = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=candidates,
        selection_policies=(_policy(),),
        master_seed=23,
        required_approaches=("DirectCohortModel",),
    )
    assert [run.seeds for run in repeated.fold_runs] == [
        run.seeds for run in result.fold_runs
    ]


def test_importance_degradation_is_nonzero_for_a_block_that_matters() -> None:
    """Permuting a block the model actually uses must change the score.

    Two degeneracies have to be avoided for this test to have any power. The
    offset puts predictions among the observations, because when every
    prediction sits below every observation the absolute error collapses to
    ``mean(observed) - mean(predicted)``, which no permutation can change. And
    the metric is squared error, which is sensitive to the pairing that a
    permutation destroys.
    """
    split, folds = _split_and_folds()
    metric = RootMeanSquaredError("n_children_total")
    candidate = CandidateDefinition(
        candidate_id="importance-power",
        approach="DirectCohortModel",
        model_factory=lambda: _SpyModel(offset=2.5, fit_log=[], predict_log=[]),
        fit_feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        component_feature_specs=(DEFAULT_TREE_FEATURE_SPEC,),
        metrics=(MeanAbsoluteError("n_children_total"), metric),
        configuration={},
        prediction_config=PredictionConfig(),
        importance_specs=(
            PermutationImportanceSpec(
                component="tree",
                metric=metric,
                feature_blocks=(FeatureBlock("ses", ("ses",)),),
                repeats=2,
            ),
        ),
    )

    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=folds,
        candidates=(candidate,),
        selection_policies=(_policy(),),
        master_seed=23,
        required_approaches=("DirectCohortModel",),
    )

    degradation = result.importance_df["degradation"].to_numpy(dtype=float)
    assert len(degradation)
    assert np.any(degradation != 0.0)
    assert result.importance_df["baseline_value"].notna().all()
    assert (
        result.importance_df["permuted_value"].to_numpy(dtype=float)
        != result.importance_df["baseline_value"].to_numpy(dtype=float)
    ).any()


def test_candidate_factory_enumerates_declared_specs_in_a_stable_order() -> None:
    """Section 5.3's sequence must be enumerated, not rebuilt per call site."""
    first = enumerate_feature_specs("total_count")
    second = enumerate_feature_specs("total_count")

    assert [candidate.name for candidate in first] == [
        candidate.name for candidate in second
    ]
    assert [candidate.name for candidate in first] == [
        "total_count__ses_linear",
        "total_count__ses_quadratic",
        "total_count__ses_spline",
        "total_count__ses_x_household_size",
        "total_count__daycare_x_median_age",
        "total_count__room_share_x_household_size",
    ]
    assert [candidate.spec.ses_form for candidate in first[:3]] == [
        "linear",
        "quadratic",
        "spline",
    ]


def test_candidate_factory_respects_component_specific_interactions() -> None:
    """Room-composition terms are declared per component and must not cross."""
    total = {candidate.name for candidate in enumerate_feature_specs("total_count")}
    probability = {
        candidate.name for candidate in enumerate_feature_specs("age_probability")
    }

    assert "total_count__room_share_x_household_size" in total
    assert "total_count__room_share_x_median_age" not in total
    assert "age_probability__room_share_x_median_age" in probability
    assert "age_probability__room_share_x_household_size" not in probability
    # Trees represent interactions and transforms themselves.
    assert len(enumerate_feature_specs("tree")) == 3


def test_candidate_factory_emits_only_valid_specifications() -> None:
    """Every enumerated spec must satisfy the schema it will be fitted under.

    The spline SES form cannot carry the linear SES interaction, so an
    enumeration that paired them would raise here rather than at fit time.
    """
    for component in ("tree", "total_count", "age_probability"):
        for candidate in enumerate_feature_specs(
            component, include_daycare_saturation=True
        ):
            assert candidate.spec.component == component
            candidate.spec.validate_for_schema(DEFAULT_MODELING_SCHEMA)


def test_candidate_factory_rejects_unknown_components_and_interactions() -> None:
    """An undeclared component or interaction is a caller error, not a default."""
    with pytest.raises(ValueError, match="Unknown feature component"):
        enumerate_feature_specs("not_a_component")
    with pytest.raises(ValueError, match="not declared for component"):
        enumerate_feature_specs(
            "age_probability", interactions=("room_share_x_household_size",)
        )


def test_candidate_factory_saturation_curve_is_opt_in() -> None:
    """Section 5.3 admits the daycare curve only when residuals justify it."""
    default = {candidate.name for candidate in enumerate_feature_specs("total_count")}
    opted_in = {
        candidate.name
        for candidate in enumerate_feature_specs(
            "total_count", include_daycare_saturation=True
        )
    }

    assert "total_count__daycare_log1p" not in default
    assert opted_in - default == {"total_count__daycare_log1p"}


# --- Canonical candidate registry (Gate 8) ------------------------------


def _canonical_config():
    return load_experiment_config("configs/modeling.toml")


def test_canonical_registry_has_stable_candidate_ids_and_order() -> None:
    """The registry's order must not depend on dict iteration or set order."""
    registry = build_canonical_candidate_registry(_canonical_config())

    assert tuple(candidate.candidate_id for candidate in registry.candidates) == (
        "direct-poisson",
        "independent-nb2",
        "bayesian-reduced",
    )
    assert {candidate.approach for candidate in registry.candidates} == {
        "DirectCohortModel",
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    }
    second = build_canonical_candidate_registry(_canonical_config())
    assert registry.candidate_set_fingerprint == second.candidate_set_fingerprint


def test_canonical_registry_declares_required_metrics_and_likelihood_scopes() -> None:
    """Model A stays marginal-only; the conditional candidates stay joint-scored."""
    registry = build_canonical_candidate_registry(_canonical_config())
    by_id = {candidate.candidate_id: candidate for candidate in registry.candidates}

    direct_metric_names = {metric.name for metric in by_id["direct-poisson"].metrics}
    assert "joint_predictive_nll" not in direct_metric_names
    assert "predictive_nll" in direct_metric_names

    for candidate_id in ("independent-nb2", "bayesian-reduced"):
        joint_names = {metric.name for metric in by_id[candidate_id].metrics}
        assert "joint_predictive_nll" in joint_names
        assert by_id[candidate_id].prediction_config.include_pointwise_log_probabilities

    approaches = {policy.approach for policy in registry.selection_policies}
    assert approaches == {
        "DirectCohortModel",
        "IndependentTotalProbabilityModel",
        "BayesianConditionalModel",
    }


def test_canonical_registry_shares_model_b_and_bayesian_feature_specs() -> None:
    """Section 3's frozen feature-spec equality must hold by construction."""
    registry = build_canonical_candidate_registry(_canonical_config())
    by_id = {candidate.candidate_id: candidate for candidate in registry.candidates}

    independent_specs = {
        spec.component: spec
        for spec in by_id["independent-nb2"].component_feature_specs
    }
    bayesian_specs = {
        spec.component: spec
        for spec in by_id["bayesian-reduced"].component_feature_specs
    }
    assert independent_specs == bayesian_specs


def test_canonical_registry_final_bayesian_factory_forces_full_profile() -> None:
    """CV uses the configured reduced profile; the final refit must not."""
    registry = build_canonical_candidate_registry(_canonical_config())

    cv_model = registry.candidate_by_id("bayesian-reduced").model_factory()
    final_model = registry.final_refit_factories["bayesian-reduced"]()

    assert cv_model.bayesian_config.active_profile == "reduced"
    assert final_model.bayesian_config.active_profile == "full"
    assert final_model.bayesian_config.diagnostic_policy.action == "error"


def test_canonical_registry_final_refit_factories_cover_every_candidate() -> None:
    registry = build_canonical_candidate_registry(_canonical_config())

    assert set(registry.final_refit_factories) == {
        candidate.candidate_id for candidate in registry.candidates
    }
    for candidate_id, factory in registry.final_refit_factories.items():
        model = factory()
        assert type(model).__name__ == registry.candidate_by_id(candidate_id).approach


def test_canonical_registry_rejects_a_factory_mapping_that_misses_a_candidate() -> None:
    from age_group_prediction.experiment.candidate_registry import CandidateRegistry

    registry = build_canonical_candidate_registry(_canonical_config())
    with pytest.raises(ValueError, match="Final refit factories"):
        CandidateRegistry(
            candidates=registry.candidates,
            selection_policies=registry.selection_policies,
            final_refit_factories={
                "direct-poisson": registry.final_refit_factories["direct-poisson"]
            },
            candidate_set_name=registry.candidate_set_name,
            candidate_set_fingerprint=registry.candidate_set_fingerprint,
        )
