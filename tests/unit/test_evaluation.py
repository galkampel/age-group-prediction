"""Tests for metric orchestration and neighborhood-cluster bootstrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from age_group_prediction import load_experiment_config
from age_group_prediction.evaluation import (
    _subset_prediction,
    evaluate_predictions,
    neighborhood_cluster_bootstrap,
)
from age_group_prediction.metrics import (
    IntervalCoverage,
    JointPredictiveNegativeLogLikelihood,
    MeanAbsoluteError,
    MeanBias,
    ParametricPredictiveNegativeLogLikelihood,
)
from age_group_prediction.models import ParametricDistributionSpec, PredictionResult


def _inputs() -> tuple[pd.DataFrame, PredictionResult]:
    """Return a small deterministic observed/prediction pair for bootstrap tests."""
    observed = pd.DataFrame(
        {
            "neighborhood_id": [1, 1, 2, 2, 3, 3],
            "n_children_total": [1, 2, 4, 5, 7, 8],
        }
    )
    total_mean = np.array([1.0, 3.0, 3.0, 6.0, 6.0, 9.0])
    probabilities = np.array([0.2, 0.3, 0.5])
    prediction = PredictionResult.from_means(
        building_ids=np.arange(len(observed)),
        cohort_names=("n_kindergarten", "n_elementary", "n_highschool"),
        total_mean=total_mean,
        cohort_means=total_mean[:, None] * probabilities,
        parametric_distributions={"total": ParametricDistributionSpec("poisson")},
    )
    return observed, prediction


def _config():
    """Load shared experiment configuration from the project TOML file."""
    path = Path(__file__).resolve().parents[2] / "configs" / "modeling.toml"
    return load_experiment_config(path)


def test_evaluate_predictions_returns_shared_result_without_features() -> None:
    """Evaluation should consume observed outcomes plus PredictionResult only."""
    observed, prediction = _inputs()
    result = evaluate_predictions(
        observed,
        prediction,
        [MeanAbsoluteError("n_children_total"), MeanBias("n_children_total")],
    )

    assert list(result.metrics_df["metric_name"]) == ["mae", "mean_bias"]
    assert result.metadata["metric_count"] == 2
    assert result.diagnostics == {}


def test_evaluate_predictions_rejects_missing_capability_clearly() -> None:
    """Unsupported metrics should fail before any partial evaluation is returned."""
    observed, prediction = _inputs()

    with pytest.raises(
        ValueError,
        match=(
            "Metric 'interval_coverage' for target 'n_children_total' requires "
            "missing prediction capability 'prediction_intervals'"
        ),
    ):
        evaluate_predictions(
            observed,
            prediction,
            [IntervalCoverage("n_children_total", 0.8)],
        )


def test_evaluate_predictions_can_explicitly_skip_missing_capabilities() -> None:
    """Mixed-capability comparisons may opt in to an auditable skip policy."""
    observed, prediction = _inputs()
    result = evaluate_predictions(
        observed,
        prediction,
        [
            MeanAbsoluteError("n_children_total"),
            IntervalCoverage("n_children_total", 0.8),
        ],
        on_missing_capability="skip",
    )

    assert list(result.metrics_df["metric_name"]) == ["mae"]
    assert result.metadata["missing_capability_policy"] == "skip"
    assert result.metadata["skipped_metric_definitions"] == [
        {
            "target": "n_children_total",
            "level": 0.8,
            "name": "interval_coverage",
            "required_capability": "prediction_intervals",
            "optimization_direction": "maximize",
            "aggregation_level": "building",
        }
    ]


def test_evaluate_predictions_checks_capability_for_each_target() -> None:
    """A payload for one target must not satisfy a different target's metric."""
    observed, prediction = _inputs()

    with pytest.raises(ValueError, match="target 'n_kindergarten'.*parametric"):
        evaluate_predictions(
            observed,
            prediction,
            [
                ParametricPredictiveNegativeLogLikelihood(
                    target="n_kindergarten",
                    interpretation="cohort likelihood",
                )
            ],
        )


def test_cluster_bootstrap_is_reproducible_with_default_rng() -> None:
    """Default-seeded bootstrap runs should produce identical outputs."""
    observed, prediction = _inputs()
    config = _config()
    metrics = [MeanAbsoluteError("n_children_total")]

    first = neighborhood_cluster_bootstrap(
        observed,
        prediction,
        metrics,
        config=config.evaluation,
        default_seed=config.randomness.default_seed,
    )
    second = neighborhood_cluster_bootstrap(
        observed,
        prediction,
        metrics,
        config=config.evaluation,
        default_seed=config.randomness.default_seed,
    )

    pd.testing.assert_frame_equal(first.intervals_df, second.intervals_df)
    pd.testing.assert_frame_equal(
        first.replicate_metrics_df, second.replicate_metrics_df
    )
    assert first.metadata["bootstrap_unit"] == "neighborhood"
    assert first.metadata["default_seed"] == config.randomness.default_seed


def test_cluster_bootstrap_advances_equal_caller_rngs_deterministically() -> None:
    """Equivalent caller RNGs should yield the same bootstrap stream and state."""
    observed, prediction = _inputs()
    config = _config()
    metrics = [MeanAbsoluteError("n_children_total")]
    rng_a = np.random.default_rng(81)
    rng_b = np.random.default_rng(81)

    first = neighborhood_cluster_bootstrap(
        observed,
        prediction,
        metrics,
        config=config.evaluation,
        default_seed=config.randomness.default_seed,
        rng=rng_a,
    )
    second = neighborhood_cluster_bootstrap(
        observed,
        prediction,
        metrics,
        config=config.evaluation,
        default_seed=config.randomness.default_seed,
        rng=rng_b,
    )

    pd.testing.assert_frame_equal(first.intervals_df, second.intervals_df)
    assert rng_a.random() == rng_b.random()
    assert first.metadata["default_seed"] is None


def test_bootstrap_replicates_resample_complete_neighborhood_clusters() -> None:
    """Each replicate should preserve row count by resampling whole neighborhoods."""
    observed, prediction = _inputs()
    config = _config()
    result = neighborhood_cluster_bootstrap(
        observed,
        prediction,
        [MeanAbsoluteError("n_children_total")],
        config=config.evaluation,
        default_seed=config.randomness.default_seed,
    )

    assert set(result.replicate_metrics_df["sample_count"]) == {len(observed)}
    assert result.failed_replicates == 0
    assert len(result.replicate_metrics_df) == config.evaluation.bootstrap_replicates
    assert (
        result.intervals_df["confidence_lower"]
        <= result.intervals_df["confidence_upper"]
    ).all()


def test_cluster_bootstrap_supports_parametric_nll_metric() -> None:
    """Neighborhood bootstrap should support distribution-aware NLL metrics."""
    observed, prediction = _inputs()
    config = _config()
    result = neighborhood_cluster_bootstrap(
        observed,
        prediction,
        [
            ParametricPredictiveNegativeLogLikelihood(
                target="n_children_total",
                interpretation="Poisson plug-in",
            )
        ],
        config=config.evaluation,
        default_seed=config.randomness.default_seed,
    )

    assert result.point_evaluation.metrics_df.loc[0, "metric_name"] == "predictive_nll"
    assert result.failed_replicates == 0


def _joint_inputs(scope: str) -> tuple[pd.DataFrame, PredictionResult]:
    """Return the bootstrap fixture with full pointwise scores under one scope."""
    observed, base = _inputs()
    rng = np.random.default_rng(12)
    pointwise = {
        key: -rng.uniform(0.1, 2.0, size=len(observed))
        for key in ("total", *base.cohort_names)
    }
    prediction = PredictionResult.from_means(
        building_ids=base.building_ids,
        cohort_names=base.cohort_names,
        total_mean=base.total_mean,
        cohort_means=base.cohort_means,
        pointwise_log_probabilities=pointwise,
        pointwise_log_probability_scope=scope,
    )
    return observed, prediction


def test_evaluate_predictions_gates_joint_nll_on_declared_scope() -> None:
    """Marginal per-target scores must not be summed into a joint NLL."""
    observed, prediction = _joint_inputs("marginal")
    metric = JointPredictiveNegativeLogLikelihood(interpretation="joint score")

    with pytest.raises(
        ValueError,
        match=(
            "Metric 'joint_predictive_nll' for target 'joint' requires missing "
            "prediction capability 'joint_pointwise_log_probabilities'"
        ),
    ):
        evaluate_predictions(observed, prediction, [metric])

    skipped = evaluate_predictions(
        observed, prediction, [metric], on_missing_capability="skip"
    )
    assert skipped.metrics_df.empty
    assert skipped.metadata["skipped_metric_definitions"][0]["name"] == (
        "joint_predictive_nll"
    )


def test_cluster_bootstrap_preserves_joint_scope_in_replicates() -> None:
    """Resampled predictions must keep the scope the joint metric depends on."""
    observed, prediction = _joint_inputs("sequential_joint")
    config = _config()
    result = neighborhood_cluster_bootstrap(
        observed,
        prediction,
        [JointPredictiveNegativeLogLikelihood(interpretation="joint score")],
        config=config.evaluation,
        default_seed=config.randomness.default_seed,
    )

    assert result.point_evaluation.metrics_df.loc[0, "target"] == "joint"
    assert result.failed_replicates == 0
    assert len(result.replicate_metrics_df) == config.evaluation.bootstrap_replicates


def _array_dispersion_inputs() -> tuple[pd.DataFrame, PredictionResult]:
    """Return a fixture whose NB2 dispersion is a per-building array.

    The neighborhood cluster sizes are deliberately unequal so resampled
    replicates differ in length from the original frame. With equal clusters
    every replicate has the original row count, and an unsubset per-building
    parameter is never caught.
    """
    observed = pd.DataFrame(
        {
            "neighborhood_id": [1, 2, 2, 3, 3, 3],
            "n_children_total": [1, 2, 4, 5, 7, 8],
        }
    )
    total_mean = np.array([1.0, 3.0, 3.0, 6.0, 6.0, 9.0])
    probabilities = np.array([0.2, 0.3, 0.5])
    row_count = len(observed)
    return observed, PredictionResult.from_means(
        building_ids=np.arange(row_count),
        cohort_names=("n_kindergarten", "n_elementary", "n_highschool"),
        total_mean=total_mean,
        cohort_means=total_mean[:, None] * probabilities,
        parametric_distributions={
            "total": ParametricDistributionSpec(
                family="nb2",
                dispersion=np.full(row_count, 0.25),
            )
        },
    )


def test_cluster_bootstrap_subsets_per_building_distribution_parameters() -> None:
    """A per-building dispersion must follow its rows through every replicate.

    Left unsubset, each resample of a different size fails the per-building
    length check, so the interval is built from whichever replicates happened
    to match the original row count.
    """
    observed, prediction = _array_dispersion_inputs()
    config = _config()
    result = neighborhood_cluster_bootstrap(
        observed,
        prediction,
        [
            ParametricPredictiveNegativeLogLikelihood(
                target="n_children_total",
                interpretation="NB2 plug-in",
            )
        ],
        config=config.evaluation,
        default_seed=config.randomness.default_seed,
    )

    assert result.failed_replicates == 0
    assert len(result.replicate_metrics_df) == config.evaluation.bootstrap_replicates


def test_subset_prediction_aligns_distribution_parameters_with_rows() -> None:
    """Subset parameters must belong to the rows they are scored against."""
    _, base = _inputs()
    row_count = len(base.total_mean)
    dispersion = np.arange(1.0, row_count + 1.0)
    prediction = PredictionResult.from_means(
        building_ids=base.building_ids,
        cohort_names=base.cohort_names,
        total_mean=base.total_mean,
        cohort_means=base.cohort_means,
        parametric_distributions={
            "total": ParametricDistributionSpec(
                family="nb2", dispersion=dispersion
            )
        },
    )

    rows = np.array([4, 2, 0])
    subset = _subset_prediction(prediction, rows)
    spec = subset.parametric_distributions["total"]

    assert np.array_equal(np.asarray(spec.dispersion), dispersion[rows])


def test_subset_prediction_rebroadcasts_scalar_distribution_parameters() -> None:
    """A scalar parameter applies to every row, so it survives any resample.

    `PredictionResult` broadcasts scalars to per-building arrays on
    construction, which is why an unsubset parameter fails the length check for
    every model rather than only those declaring per-building values.
    """
    _, base = _inputs()
    prediction = PredictionResult.from_means(
        building_ids=base.building_ids,
        cohort_names=base.cohort_names,
        total_mean=base.total_mean,
        cohort_means=base.cohort_means,
        parametric_distributions={
            "total": ParametricDistributionSpec(family="normal", scale=1.5)
        },
    )

    rows = np.array([3, 3, 1])
    subset = _subset_prediction(prediction, rows)
    spec = subset.parametric_distributions["total"]

    assert np.array_equal(np.asarray(spec.scale), np.full(len(rows), 1.5))
