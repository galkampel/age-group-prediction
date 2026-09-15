"""Golden tests for typed metrics and stochastic prediction transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import nbinom, norm
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_pinball_loss,
    root_mean_squared_error,
)

from age_group_prediction.distributions import dirichlet_multinomial_prefix_log_masses
from age_group_prediction.metrics import (
    CompositionBrierScore,
    CompositionLogLoss,
    DrawsRandomizedPIT,
    IntervalCoverage,
    JointPredictiveNegativeLogLikelihood,
    MeanAbsoluteError,
    MeanBias,
    MeanIntervalWidth,
    MeanPoissonDeviance,
    MetricResult,
    ParametricPredictiveNegativeLogLikelihood,
    ParametricRandomizedPIT,
    PointwisePredictiveNegativeLogLikelihood,
    PredictiveNegativeLogLikelihood,
    RandomizedPIT,
    ReconciliationError,
    RootMeanSquaredError,
    WeightedIntervalScore,
    available_prediction_capabilities,
    default_metric_set,
)
from age_group_prediction.modeling_config import (
    EvaluationConfig,
    PredictionValidationConfig,
)
from age_group_prediction.models import ParametricDistributionSpec, PredictionResult

COHORTS = ("n_kindergarten", "n_elementary", "n_highschool")


def _evaluation_config() -> EvaluationConfig:
    return EvaluationConfig(
        bootstrap_replicates=10,
        confidence_level=0.9,
        neighborhood_id_column="neighborhood_id",
        interval_method="percentile",
        max_failed_fraction=0.2,
        poisson_minimum_mean=0.125,
        pit_histogram_bins=7,
    )


def _observed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_children_total": [0, 3, 4],
            "n_kindergarten": [0, 1, 4],
            "n_elementary": [0, 1, 0],
            "n_highschool": [0, 1, 0],
        }
    )


def _prediction(
    *,
    pointwise: dict[str, np.ndarray] | None = None,
    draws: dict[str, np.ndarray] | None = None,
    intervals: dict[str, np.ndarray] | None = None,
    interval_levels: tuple[float, ...] = (),
    parametric: dict[str, ParametricDistributionSpec] | None = None,
    scope: str | None = None,
) -> PredictionResult:
    cohort_means = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 0.0, 0.0]])
    return PredictionResult.from_means(
        building_ids=np.array([1, 2, 3]),
        cohort_names=COHORTS,
        total_mean=np.array([0.0, 3.0, 4.0]),
        cohort_means=cohort_means,
        pointwise_log_probabilities=pointwise,
        predictive_draws=draws,
        prediction_intervals=intervals,
        interval_levels=interval_levels,
        parametric_distributions=parametric,
        pointwise_log_probability_scope=scope,
    )


def test_point_metrics_have_exact_values_and_handle_zeros() -> None:
    observed = _observed()
    prediction = _prediction()

    assert MeanAbsoluteError("n_children_total").compute(observed, prediction).value == 0
    assert RootMeanSquaredError("n_children_total").compute(observed, prediction).value == 0
    assert MeanBias("n_children_total").compute(observed, prediction).value == 0
    assert MeanPoissonDeviance("n_children_total", minimum_mean=1e-8).compute(
        observed, prediction
    ).value == pytest.approx(2e-8 / 3)


def test_metrics_reject_invalid_observed_values() -> None:
    observed = _observed()
    observed.loc[0, "n_children_total"] = -1

    with pytest.raises(ValueError, match="finite and nonnegative"):
        MeanAbsoluteError("n_children_total").compute(observed, _prediction())


def test_poisson_nll_is_finite_for_zero_mean_and_zero_observation() -> None:
    result = ParametricPredictiveNegativeLogLikelihood(
        target="n_children_total",
        interpretation="plug-in Poisson total",
    ).compute(
        _observed(),
        _prediction(parametric={"total": ParametricDistributionSpec("poisson")}),
    )

    assert np.isfinite(result.value)
    assert result.metadata["interpretation"] == "plug-in Poisson total"


def test_posterior_nll_uses_stable_log_mean_exp() -> None:
    log_probabilities = np.array([[-1000.0, -1001.0], [-2.0, -4.0], [-3.0, -3.0]])
    prediction = _prediction(pointwise={"total": log_probabilities})
    result = PointwisePredictiveNegativeLogLikelihood(
        target="n_children_total",
        interpretation="posterior-integrated count likelihood",
    ).compute(_observed(), prediction)

    expected = -np.mean(logsumexp(log_probabilities, axis=1) - np.log(2))
    assert result.value == pytest.approx(expected)
    assert result.metadata["integration"] == "posterior-log-mean-exp"


def test_parametric_nb2_and_normal_nll_match_scipy() -> None:
    observed = _observed()
    base = _prediction()

    nb2_prediction = _prediction(
        parametric={"total": ParametricDistributionSpec("nb2", dispersion=0.25)}
    )
    nb2_result = ParametricPredictiveNegativeLogLikelihood(
        target="n_children_total",
        interpretation="NB2 plug-in",
    ).compute(observed, nb2_prediction)
    r = np.full(len(observed), 1.0 / 0.25)
    p = r / (r + base.total_mean)
    expected_nb2 = -np.mean(nbinom.logpmf(observed["n_children_total"], r, p))
    assert nb2_result.value == pytest.approx(expected_nb2)
    assert nb2_result.metadata["family"] == "nb2"

    normal_prediction = _prediction(
        parametric={"total": ParametricDistributionSpec("normal", scale=1.5)}
    )
    normal_result = ParametricPredictiveNegativeLogLikelihood(
        target="n_children_total",
        interpretation="Normal approximation",
    ).compute(observed, normal_prediction)
    expected_normal = -np.mean(
        norm.logpdf(observed["n_children_total"], loc=base.total_mean, scale=1.5)
    )
    assert normal_result.value == pytest.approx(expected_normal)
    assert normal_result.metadata["family"] == "normal"


def test_composition_scores_match_golden_values() -> None:
    observed = _observed()
    prediction = _prediction()

    assert CompositionLogLoss().compute(observed, prediction).value == pytest.approx(
        3 * np.log(3) / 7
    )
    assert CompositionBrierScore().compute(observed, prediction).value == pytest.approx(
        2 / 7
    )


def test_reconciliation_metrics_are_zero() -> None:
    prediction = _prediction()

    assert ReconciliationError("mean").compute(_observed(), prediction).value == 0
    assert ReconciliationError("max").compute(_observed(), prediction).value == 0
    assert (
        ReconciliationError("count_above_tolerance", tolerance=1e-12)
        .compute(_observed(), prediction)
        .value
        == 0
    )


def test_randomized_poisson_pit_default_and_caller_rng_are_reproducible() -> None:
    metric = ParametricRandomizedPIT(
        target="n_children_total",
        default_seed=17,
        histogram_bins=5,
    )

    prediction = _prediction(parametric={"total": ParametricDistributionSpec("poisson")})
    default_a = metric.compute(_observed(), prediction)
    default_b = metric.compute(_observed(), prediction)
    assert default_a.metadata["pit_values"] == default_b.metadata["pit_values"]

    rng_a = np.random.default_rng(31)
    rng_b = np.random.default_rng(31)
    caller_a = metric.compute(_observed(), prediction, rng=rng_a)
    caller_b = metric.compute(_observed(), prediction, rng=rng_b)
    assert caller_a.metadata["pit_values"] == caller_b.metadata["pit_values"]
    assert rng_a.random() == rng_b.random()


def test_randomized_draw_pit_uses_predictive_draws() -> None:
    draws = np.array([[0, 0, 1], [2, 3, 4], [3, 4, 5]], dtype=float)
    metric = DrawsRandomizedPIT(
        target="n_children_total",
        default_seed=9,
        histogram_bins=4,
    )
    result = metric.compute(_observed(), _prediction(draws={"total": draws}))

    pit = np.asarray(result.metadata["pit_values"], dtype=float)
    assert np.logical_and(pit >= 0, pit <= 1).all()
    assert len(result.metadata["histogram_counts"]) == 4


def test_draw_pit_matches_rank_formula_without_pseudo_counts() -> None:
    observed = _observed()
    draws = np.array([[0, 1], [3, 4], [2, 4]], dtype=float)
    metric = DrawsRandomizedPIT(
        target="n_children_total",
        default_seed=123,
        histogram_bins=4,
    )
    prediction = _prediction(draws={"total": draws})
    result = metric.compute(observed, prediction)

    uniform = np.random.default_rng(123).uniform(size=len(observed))
    actual = observed["n_children_total"].to_numpy(dtype=float)
    less = (draws < actual[:, None]).sum(axis=1)
    equal = (draws == actual[:, None]).sum(axis=1)
    expected = (less + uniform * equal) / draws.shape[1]
    np.testing.assert_allclose(np.asarray(result.metadata["pit_values"]), expected)


def test_interval_metrics_match_golden_values() -> None:
    intervals = np.array(
        [
            [[0.0, 0.0]],
            [[2.0, 4.0]],
            [[2.0, 3.0]],
        ]
    )
    prediction = _prediction(
        intervals={"total": intervals},
        interval_levels=(0.8,),
    )

    assert IntervalCoverage("n_children_total", 0.8).compute(
        _observed(), prediction
    ).value == pytest.approx(2 / 3)
    assert MeanIntervalWidth("n_children_total", 0.8).compute(
        _observed(), prediction
    ).value == pytest.approx(1.0)
    assert WeightedIntervalScore("n_children_total", 0.8).compute(
        _observed(), prediction
    ).value == pytest.approx(13 / 30)


def test_metric_result_metadata_is_validated_and_isolated() -> None:
    """Metadata must be rejected unless JSON-safe, and copied defensively."""
    nested = {"pit_values": [0.1, 0.2], "inner": {"bins": (1, 2)}}
    result = MetricResult("m", 1.0, "total", "building", 2, nested)

    nested["pit_values"].append(0.9)
    assert result.metadata["pit_values"] == [0.1, 0.2]
    assert result.metadata["inner"]["bins"] == [1, 2]

    with pytest.raises(TypeError, match="not JSON-serializable"):
        MetricResult("m", 1.0, "total", "building", 2, {"array": np.arange(3)})


def test_point_metrics_match_sklearn() -> None:
    observed = _observed()
    prediction = _prediction()
    actual = observed["n_children_total"].to_numpy(dtype=float)
    pred = prediction.total_mean

    mae = MeanAbsoluteError("n_children_total").compute(observed, prediction).value
    rmse = RootMeanSquaredError("n_children_total").compute(observed, prediction).value
    assert mae == pytest.approx(mean_absolute_error(actual, pred))
    assert rmse == pytest.approx(root_mean_squared_error(actual, pred))


def test_composition_metrics_match_sklearn_oracle() -> None:
    """Vectorized composition scores must equal sklearn's weighted multiclass forms."""
    observed = _observed()
    prediction = _prediction()
    counts = observed.loc[:, list(COHORTS)].to_numpy(dtype=float)
    n_cohorts = counts.shape[1]
    y_true = np.tile(np.arange(n_cohorts), counts.shape[0])
    y_proba = np.repeat(prediction.age_group_probabilities, n_cohorts, axis=0)
    weights = counts.reshape(-1)
    labels = list(range(n_cohorts))

    assert CompositionLogLoss().compute(observed, prediction).value == pytest.approx(
        log_loss(y_true, y_proba, sample_weight=weights, labels=labels)
    )
    assert CompositionBrierScore().compute(
        observed, prediction
    ).value == pytest.approx(
        brier_score_loss(
            y_true,
            y_proba,
            sample_weight=weights,
            labels=labels,
            scale_by_half=False,
        )
    )


def test_weighted_interval_score_matches_pinball_loss_oracle() -> None:
    """WIS at level alpha equals pinball losses at alpha/2 and 1-alpha/2."""
    intervals = np.array([[[0.0, 0.0]], [[2.0, 4.0]], [[2.0, 3.0]]])
    level = 0.8
    prediction = _prediction(
        intervals={"total": intervals}, interval_levels=(level,)
    )
    observed = _observed()
    actual = observed["n_children_total"].to_numpy(dtype=float)
    alpha = 1.0 - level

    expected = mean_pinball_loss(
        actual, intervals[:, 0, 0], alpha=alpha / 2.0
    ) + mean_pinball_loss(actual, intervals[:, 0, 1], alpha=1.0 - alpha / 2.0)
    assert WeightedIntervalScore("n_children_total", level).compute(
        observed, prediction
    ).value == pytest.approx(expected)


def test_default_metric_set_contains_shared_metrics_and_configured_tolerances() -> None:
    """The canonical set should cover every shared point and accounting metric."""
    validation_config = PredictionValidationConfig(reconciliation_tolerance=0.25)
    metrics = default_metric_set(
        ("n_children_total", "n_kindergarten"),
        nll_source="parametric",
        nll_interpretation="plug-in count likelihood",
        evaluation_config=_evaluation_config(),
        prediction_validation_config=validation_config,
    )

    expected_per_target = {
        "mae",
        "rmse",
        "mean_bias",
        "r2",
        "mean_poisson_deviance",
        "predictive_nll",
    }
    for target in ("n_children_total", "n_kindergarten"):
        assert {metric.name for metric in metrics if metric.target == target} == (
            expected_per_target
        )
    assert {metric.name for metric in metrics if metric.target == "composition"} == {
        "composition_log_loss",
        "composition_brier",
    }
    reconciliation = [
        metric for metric in metrics if isinstance(metric, ReconciliationError)
    ]
    assert [metric.name for metric in reconciliation] == [
        "mean_reconciliation_error",
        "max_reconciliation_error",
        "reconciliation_count_above_tolerance",
    ]
    assert {metric.tolerance for metric in reconciliation} == {0.25}
    poisson_metrics = [
        metric for metric in metrics if isinstance(metric, MeanPoissonDeviance)
    ]
    assert {metric.minimum_mean for metric in poisson_metrics} == {0.125}
    nll_metrics = [
        metric
        for metric in metrics
        if isinstance(metric, PredictiveNegativeLogLikelihood)
    ]
    assert {metric.interpretation for metric in nll_metrics} == {
        "plug-in count likelihood"
    }
    assert not any(isinstance(metric, IntervalCoverage) for metric in metrics)
    assert not any(isinstance(metric, RandomizedPIT) for metric in metrics)


def test_default_metric_set_adds_requested_distributional_metrics() -> None:
    """Interval and PIT metrics should be present only when explicitly requested."""
    metrics = default_metric_set(
        ("n_children_total",),
        nll_source="pointwise",
        nll_interpretation="posterior predictive log score",
        evaluation_config=_evaluation_config(),
        prediction_validation_config=PredictionValidationConfig(),
        interval_levels=(0.8, 0.95),
        pit_source="draws",
        pit_default_seed=73,
    )

    assert sum(isinstance(metric, IntervalCoverage) for metric in metrics) == 2
    assert sum(isinstance(metric, MeanIntervalWidth) for metric in metrics) == 2
    assert sum(isinstance(metric, WeightedIntervalScore) for metric in metrics) == 2
    pit = next(metric for metric in metrics if isinstance(metric, RandomizedPIT))
    assert pit.source == "draws"
    assert pit.default_seed == 73
    assert pit.histogram_bins == 7


def test_default_metric_set_keeps_shared_names_across_model_nll_definitions() -> None:
    """Model-specific NLL semantics must not change the comparison table shape."""
    kwargs = {
        "targets": ("n_children_total", *COHORTS),
        "evaluation_config": _evaluation_config(),
        "prediction_validation_config": PredictionValidationConfig(),
    }
    model_a = default_metric_set(
        nll_source="parametric",
        nll_interpretation="independent plug-in Poisson",
        **kwargs,
    )
    model_b = default_metric_set(
        nll_source="parametric",
        nll_interpretation="NB2 plus multinomial conditional composition",
        **kwargs,
    )
    model_c = default_metric_set(
        nll_source="pointwise",
        nll_interpretation="posterior-integrated joint likelihood",
        **kwargs,
    )

    identity = lambda metric: (metric.name, metric.target)
    assert list(map(identity, model_a)) == list(map(identity, model_b))
    assert list(map(identity, model_b)) == list(map(identity, model_c))
    assert {
        metric.required_capability
        for metric in model_c
        if isinstance(metric, PredictiveNegativeLogLikelihood)
    } == {"pointwise_log_probabilities"}


def test_available_prediction_capabilities_reflect_optional_payloads() -> None:
    """Capability discovery should report only payloads actually supplied."""
    base = _prediction()
    assert available_prediction_capabilities(base) == {
        "means",
        "probabilities",
        "reconciliation",
    }

    enriched = _prediction(
        draws={"total": np.ones((3, 2))},
        parametric={"total": ParametricDistributionSpec("poisson")},
    )
    assert available_prediction_capabilities(enriched) == {
        "means",
        "probabilities",
        "predictive_draws",
        "parametric_distributions",
        "reconciliation",
    }


def _sequential_pointwise() -> dict[str, np.ndarray]:
    return {
        "total": np.array([-0.5, -1.0, -2.0]),
        "n_kindergarten": np.array([0.0, -0.7, -0.2]),
        "n_elementary": np.array([0.0, -0.4, -0.1]),
        "n_highschool": np.zeros(3),
    }


def test_joint_nll_sums_sequential_entries_per_building() -> None:
    pointwise = _sequential_pointwise()

    result = JointPredictiveNegativeLogLikelihood(
        interpretation="NB2 total plus conditional composition"
    ).compute(_observed(), _prediction(pointwise=pointwise, scope="sequential_joint"))

    assert result.value == pytest.approx(-np.mean(sum(pointwise.values())))
    assert result.target == "joint"
    assert result.sample_count == 3
    assert result.metadata["integration"] == "pointwise"
    assert result.metadata["summed_keys"] == ["total", *COHORTS]


def test_joint_nll_sums_within_each_draw_before_log_mean_exp() -> None:
    rng = np.random.default_rng(4)
    pointwise = {
        key: rng.normal(-1.0, 0.8, size=(3, 5)) for key in ("total", *COHORTS)
    }

    result = JointPredictiveNegativeLogLikelihood(
        interpretation="posterior joint"
    ).compute(_observed(), _prediction(pointwise=pointwise, scope="sequential_joint"))

    joint_draws = sum(pointwise.values())
    expected = -np.mean(logsumexp(joint_draws, axis=1) - np.log(5))
    summed_per_key = -np.mean(
        sum(logsumexp(values, axis=1) - np.log(5) for values in pointwise.values())
    )
    assert result.value == pytest.approx(expected)
    assert result.value != pytest.approx(summed_per_key)
    assert result.metadata["integration"] == "posterior-log-mean-exp"


def test_joint_nll_is_invariant_to_cohort_order_while_entries_are_not() -> None:
    """Sequential entries move between keys under reordering; their sum does not."""
    counts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 0.0, 0.0]])
    alpha = np.array([[1.0, 2.0, 3.0], [0.5, 4.0, 1.5], [2.0, 2.0, 2.0]])
    cohort_means = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 0.0, 0.0]])
    metric = JointPredictiveNegativeLogLikelihood(interpretation="Dirichlet-multinomial")
    values = []
    highschool_entries = []
    for order in ((0, 1, 2), (2, 0, 1)):
        names = tuple(COHORTS[index] for index in order)
        prefixes = dirichlet_multinomial_prefix_log_masses(
            counts[:, order], alpha[:, order]
        )
        increments = np.diff(prefixes, axis=1, prepend=0.0)
        pointwise = {
            "total": np.array([-0.3, -1.2, -1.9]),
            **{name: increments[:, index] for index, name in enumerate(names)},
        }
        prediction = PredictionResult.from_means(
            building_ids=np.array([1, 2, 3]),
            cohort_names=names,
            total_mean=np.array([0.0, 3.0, 4.0]),
            cohort_means=cohort_means[:, order],
            pointwise_log_probabilities=pointwise,
            pointwise_log_probability_scope="sequential_joint",
        )
        values.append(metric.compute(_observed(), prediction).value)
        highschool_entries.append(pointwise["n_highschool"])

    assert values[0] == pytest.approx(values[1], abs=1e-12)
    # Last in schema order, the entry is determined by the total and is zero.
    np.testing.assert_allclose(highschool_entries[0], 0.0, rtol=0, atol=1e-12)
    assert not np.allclose(highschool_entries[0], highschool_entries[1])


@pytest.mark.parametrize("scope", ["marginal", None])
def test_joint_nll_is_unavailable_for_marginal_or_undeclared_scores(
    scope: str | None,
) -> None:
    prediction = _prediction(pointwise=_sequential_pointwise(), scope=scope)

    assert "joint_pointwise_log_probabilities" not in (
        available_prediction_capabilities(prediction)
    )
    with pytest.raises(ValueError, match="sequential_joint"):
        JointPredictiveNegativeLogLikelihood(interpretation="joint").compute(
            _observed(), prediction
        )


def test_joint_capability_is_reported_for_sequential_joint_scores() -> None:
    prediction = _prediction(pointwise=_sequential_pointwise(), scope="sequential_joint")

    assert "joint_pointwise_log_probabilities" in (
        available_prediction_capabilities(prediction)
    )


def test_joint_nll_requires_an_interpretation() -> None:
    with pytest.raises(ValueError, match="interpretation"):
        JointPredictiveNegativeLogLikelihood(interpretation="")


@pytest.mark.parametrize(
    ("pointwise", "scope", "message"),
    [
        (None, "sequential_joint", "requires pointwise"),
        (None, "marginal", "requires pointwise"),
        ({"total": np.zeros(3)}, "sequential_joint", "total and every cohort"),
        (
            {"total": np.zeros(3), **{name: np.zeros((3, 2)) for name in COHORTS}},
            "sequential_joint",
            "share one shape",
        ),
        ({"total": np.zeros(3)}, "joint", "must be one of"),
    ],
)
def test_prediction_result_validates_pointwise_scope(
    pointwise: dict[str, np.ndarray] | None, scope: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _prediction(pointwise=pointwise, scope=scope)


def test_default_metric_set_adds_joint_nll_only_when_requested() -> None:
    kwargs = {
        "targets": ("n_children_total", *COHORTS),
        "nll_source": "pointwise",
        "nll_interpretation": "posterior-integrated likelihood",
        "evaluation_config": _evaluation_config(),
        "prediction_validation_config": PredictionValidationConfig(),
    }
    without_joint = default_metric_set(**kwargs)
    with_joint = default_metric_set(
        **kwargs, joint_nll_interpretation="NB2 total plus conditional composition"
    )

    assert not any(
        isinstance(metric, JointPredictiveNegativeLogLikelihood)
        for metric in without_joint
    )
    joint = [
        metric
        for metric in with_joint
        if isinstance(metric, JointPredictiveNegativeLogLikelihood)
    ]
    assert len(joint) == 1
    assert joint[0].required_capability == "joint_pointwise_log_probabilities"
    assert len(with_joint) == len(without_joint) + 1


def test_predictive_nll_variants_cannot_disagree_about_their_source() -> None:
    """The scoring path and the declared capability come from the same class."""
    parametric = ParametricPredictiveNegativeLogLikelihood(
        target="n_children_total", interpretation="poisson"
    )
    pointwise = PointwisePredictiveNegativeLogLikelihood(
        target="n_children_total", interpretation="supplied"
    )
    assert parametric.source == "parametric"
    assert parametric.required_capability == "parametric_distributions"
    assert pointwise.source == "pointwise"
    assert pointwise.required_capability == "pointwise_log_probabilities"

    # The metric-table key identifies the score, not how it was supplied.
    for metric in (parametric, pointwise):
        assert metric.name == "predictive_nll"
        assert metric.aggregation_level == "building"

    # A mistyped source can no longer select a scoring path.
    with pytest.raises(TypeError):
        ParametricPredictiveNegativeLogLikelihood(
            target="n_children_total", interpretation="x", source="typo"
        )


def test_randomized_pit_variants_declare_their_own_capability() -> None:
    parametric = ParametricRandomizedPIT(
        target="n_children_total", default_seed=1, histogram_bins=5
    )
    draws = DrawsRandomizedPIT(
        target="n_children_total", default_seed=1, histogram_bins=5
    )
    assert parametric.required_capability == "parametric_distributions"
    assert draws.required_capability == "predictive_draws"
    assert parametric.name == draws.name == "randomized_pit_ks"


def test_source_dispatching_bases_are_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        PredictiveNegativeLogLikelihood(target="n_children_total", interpretation="x")
    with pytest.raises(TypeError, match="abstract"):
        RandomizedPIT(target="n_children_total", default_seed=1, histogram_bins=5)


def test_default_metric_set_rejects_unknown_source_strings() -> None:
    """The factory is the single place a source string is parsed."""
    config = EvaluationConfig(
        bootstrap_replicates=2,
        confidence_level=0.9,
        neighborhood_id_column="neighborhood_id",
        interval_method="percentile",
        max_failed_fraction=0.5,
        poisson_minimum_mean=1e-8,
        pit_histogram_bins=10,
    )
    with pytest.raises(ValueError, match="nll_source must be one of"):
        default_metric_set(
            ("n_children_total",),
            nll_source="parametric_distributions",
            nll_interpretation="x",
            evaluation_config=config,
            prediction_validation_config=PredictionValidationConfig(),
        )
    with pytest.raises(ValueError, match="pit_source must be one of"):
        default_metric_set(
            ("n_children_total",),
            nll_source="parametric",
            nll_interpretation="x",
            evaluation_config=config,
            prediction_validation_config=PredictionValidationConfig(),
            pit_source="predictive_draws",
            pit_default_seed=1,
        )


@pytest.mark.parametrize("family", ["gamma", "NORMAL", "student_t", ""])
def test_parametric_metrics_reject_unknown_families(family: str) -> None:
    """An unknown family must not reach a metric disguised as Normal.

    The dispatch chains here end in an explicit branch per family, so even if a
    spec were built without validation the metric refuses to score it rather
    than substituting a log density for a log mass.
    """
    with pytest.raises(ValueError, match="family must be one of"):
        ParametricDistributionSpec(family, scale=1.0)


def test_parametric_metric_dispatch_has_no_catch_all_normal_branch() -> None:
    """Guard the fix: bypassing construction still must not score as Normal."""
    prediction = _prediction(
        parametric={"total": ParametricDistributionSpec("normal", scale=1.5)}
    )
    # Corrupt the stored spec after construction, so this exercises the metric
    # dispatch rather than the PredictionResult guard that normally stops it.
    object.__setattr__(prediction.parametric_distributions["total"], "family", "gamma")
    observed = _observed()

    nll = ParametricPredictiveNegativeLogLikelihood(
        target="n_children_total", interpretation="total marginal"
    )
    with pytest.raises(ValueError, match="Unsupported predictive family"):
        nll.compute(observed, prediction)

    pit = ParametricRandomizedPIT(
        target="n_children_total", default_seed=0, histogram_bins=7
    )
    with pytest.raises(ValueError, match="Unsupported predictive family"):
        pit.compute(observed, prediction, rng=np.random.default_rng(0))
