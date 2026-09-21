"""Focused tests for independent totals, grouped probabilities, and uncertainty."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from age_group_prediction import (
    COHORT_TARGET_COLUMNS,
    DEFAULT_TOTAL_FEATURE_SPEC,
    TOTAL_TARGET_COLUMN,
    FoldConfig,
    IndependentSearchSpaceConfig,
    IndependentTotalFamily,
    IndependentTotalProbabilityConfig,
    IndependentTotalProbabilityModel,
    OptunaTuningConfig,
    PredictionConfig,
)
from age_group_prediction.models.count_regression import (
    _fit_nb2_regression,
    _nb2_log_probability,
    _predict_total_mean,
)
from age_group_prediction.models.grouped_multinomial import (
    _decision_logits,
    _fit_grouped_multinomial,
    _grouped_multinomial_rows,
)


def _config(
    *, total_family: IndependentTotalFamily = "nb2"
) -> IndependentTotalProbabilityConfig:
    """A fast config: fixed regularization, two trials, two folds, three refits."""
    return IndependentTotalProbabilityConfig(
        total_family=total_family,
        tuning=OptunaTuningConfig(n_trials=2),
        search_space=IndependentSearchSpaceConfig(
            total_l2_penalty=(0.01, 0.01),
            probability_c=(1.0, 1.0),
        ),
        tuning_folds=FoldConfig(n_folds=2),
        optimizer_max_iterations=300,
        bootstrap_replicates=3,
    )


@pytest.fixture(scope="module")
def modeling_df() -> pd.DataFrame:
    """A small random table with three zero-total buildings."""
    rng = np.random.default_rng(18)
    row_count = 36
    counts = rng.integers(1, 5, size=(row_count, 3))
    counts[:3] = 0
    frame = pd.DataFrame(
        {
            "building_id": np.arange(row_count),
            "neighborhood_id": np.repeat(np.arange(3), row_count // 3),
            "ses": rng.normal(size=row_count),
            "avg_household_size": rng.uniform(1.8, 3.8, size=row_count),
            "median_age": rng.uniform(24.0, 58.0, size=row_count),
            "n_daycares_500m": rng.integers(0, 6, size=row_count),
            "n_apartments": rng.integers(8, 60, size=row_count),
            "3_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "school_status": rng.choice(
                ["none", "existing", "planned"], size=row_count
            ),
            "n_kindergarten": counts[:, 0],
            "n_elementary": counts[:, 1],
            "n_highschool": counts[:, 2],
        }
    )
    frame[TOTAL_TARGET_COLUMN] = counts.sum(axis=1)
    return frame


@pytest.fixture(scope="module")
def fitted_model(modeling_df: pd.DataFrame) -> IndependentTotalProbabilityModel:
    """The model fitted once on ``modeling_df`` with seed 44."""
    return IndependentTotalProbabilityModel(independent_config=_config()).fit(
        modeling_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(44),
    )


def test_nb2_log_probability_matches_shared_parameterization() -> None:
    """The regression's NB2 log mass matches the shared distribution module."""
    observed = np.array([0.0, 1.0, 4.0])
    means = np.array([0.5, 2.0, 5.0])
    dispersion = 0.4

    from age_group_prediction.distributions import pointwise_log_probability

    np.testing.assert_allclose(
        _nb2_log_probability(observed, means, dispersion),
        pointwise_log_probability(
            "nb2", observed, means, dispersion=dispersion
        ),
    )


def test_log_exposure_has_fixed_unit_coefficient() -> None:
    """Doubling exposure doubles the predicted total mean."""
    features = pd.DataFrame({"x": np.zeros(12)})
    observed = np.arange(1, 13, dtype=float)
    exposure = pd.Series(np.full(12, 10.0))
    fit = _fit_nb2_regression(
        features,
        observed,
        np.log(exposure),
        l2_penalty=0.0,
        config=_config(),
    )
    baseline = _predict_total_mean(
        fit, features.iloc[:2], pd.Series(np.log([10.0, 10.0])), minimum_mean=1e-8
    )
    doubled = _predict_total_mean(
        fit, features.iloc[:2], pd.Series(np.log([20.0, 20.0])), minimum_mean=1e-8
    )

    np.testing.assert_allclose(doubled, 2.0 * baseline)


@pytest.mark.parametrize("c_value", [0.01, 1.5, 100.0])
def test_grouped_and_literal_expansion_are_equivalent(c_value: float) -> None:
    """Equivalence must hold across the whole default C range.

    Low C is the stress case: regularization is applied to the summed loss, so
    a mistake in how sample weights scale against the penalty shows up there
    first. Zero-total buildings are included to confirm they drop out cleanly.
    """
    features = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0, 0.5]})
    counts = np.array([[4, 2, 1], [1, 3, 2], [2, 1, 5], [3, 4, 2], [0, 0, 0]])
    grouped = _fit_grouped_multinomial(
        features,
        counts,
        c_value=c_value,
        config=_config(),
        seed=3,
    )
    grouped_features, labels, weights = _grouped_multinomial_rows(features, counts)
    expanded_features = np.repeat(grouped_features, weights.astype(int), axis=0)
    expanded_labels = np.repeat(labels, weights.astype(int))
    expanded = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        tol=_config().optimizer_tolerance,
        max_iter=_config().optimizer_max_iterations,
        random_state=3,
    ).fit(expanded_features, expanded_labels)

    np.testing.assert_allclose(grouped.coef_, expanded.coef_, atol=1e-8)
    np.testing.assert_allclose(grouped.intercept_, expanded.intercept_, atol=1e-8)
    np.testing.assert_allclose(
        _decision_logits(grouped, features),
        _decision_logits(expanded, features),
        atol=1e-8,
    )


def test_zero_totals_do_not_create_grouped_rows() -> None:
    """A building with no children contributes no grouped likelihood rows."""
    features = pd.DataFrame({"x": [0.0, 1.0]})
    grouped_features, labels, weights = _grouped_multinomial_rows(
        features, np.array([[0, 0, 0], [2, 1, 3]])
    )

    assert grouped_features.shape == (3, 1)
    np.testing.assert_array_equal(labels, [0, 1, 2])
    np.testing.assert_array_equal(weights, [2, 1, 3])


def test_probability_fit_rejects_an_unobserved_age_class() -> None:
    """The multinomial fit refuses data in which an age class never occurs."""
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    counts = np.array([[2, 1, 0], [1, 3, 0], [4, 2, 0]])

    with pytest.raises(ValueError, match="every age class"):
        _fit_grouped_multinomial(
            features,
            counts,
            c_value=1.0,
            config=_config(),
            seed=4,
        )


def test_predictions_normalize_and_reconcile_exactly(
    fitted_model: IndependentTotalProbabilityModel, modeling_df: pd.DataFrame
) -> None:
    """Probabilities sum to one and cohort means sum to the total mean."""
    result = fitted_model.predict(modeling_df)

    np.testing.assert_allclose(result.age_group_probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.cohort_means.sum(axis=1), result.total_mean)
    assert np.max(result.reconciliation_error) <= 1e-9
    assert result.parametric_distributions is not None
    assert set(result.parametric_distributions) == {"total"}
    assert result.parametric_distributions["total"].family == "nb2"


def test_targets_never_enter_either_predictor_matrix(
    fitted_model: IndependentTotalProbabilityModel,
) -> None:
    """Neither component's features include a target or the building ID."""
    total_names = set(fitted_model.feature_transformer.get_feature_names_out())
    metadata = fitted_model.get_metadata()
    probability_names = set(
        metadata["model"]["diagnostics"]["probability_preprocessing"][  # type: ignore[index]
            "feature_names"
        ]
    )
    forbidden = {TOTAL_TARGET_COLUMN, *COHORT_TARGET_COLUMNS, "building_id"}

    assert total_names.isdisjoint(forbidden)
    assert probability_names.isdisjoint(forbidden)


def test_calibration_and_metadata_are_json_serializable(
    fitted_model: IndependentTotalProbabilityModel,
) -> None:
    """Metadata serializes to JSON and records calibration and tuning evidence."""
    metadata = fitted_model.get_metadata()
    json.dumps(metadata)
    calibration = metadata["model"]["calibration"]  # type: ignore[index]

    assert calibration["temperature"] > 0
    assert calibration["selected_weighted_nll"] <= calibration["raw_weighted_nll"]
    assert calibration["zero_total_validation_rows"] >= 0
    selection = metadata["model"]["diagnostics"]["selection"]  # type: ignore[index]
    assert len(selection["total_tuning"]["trials"]) == 2
    assert len(selection["probability_tuning"]["trials"]) == 2
    assert all(
        len(trial["intermediate_values"]) == 2
        for trial in selection["total_tuning"]["trials"]
    )


def test_fixed_specs_are_tuned_independently_with_optuna(
    modeling_df: pd.DataFrame,
) -> None:
    """Each component gets its own Optuna study over its own parameter."""
    config = IndependentTotalProbabilityConfig(
        tuning=OptunaTuningConfig(n_trials=2),
        search_space=IndependentSearchSpaceConfig(
            total_l2_penalty=(0.0, 0.1),
            probability_c=(0.1, 10.0),
        ),
        tuning_folds=FoldConfig(n_folds=2),
        optimizer_max_iterations=300,
        bootstrap_replicates=2,
    )
    model = IndependentTotalProbabilityModel(independent_config=config).fit(
        modeling_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(52),
    )
    selection = model.get_metadata()["model"]["diagnostics"]["selection"]  # type: ignore[index]

    assert selection["fold_count"] == 2
    assert selection["total_feature_spec"]["component"] == "total_count"
    assert selection["probability_feature_spec"]["component"] == "composition"
    assert len(selection["total_tuning"]["trials"]) == 2
    assert len(selection["probability_tuning"]["trials"]) == 2
    assert set(selection["total_tuning"]["best_params"]) == {"total_l2_penalty"}
    assert set(selection["probability_tuning"]["best_params"]) == {"probability_c"}


@pytest.mark.parametrize("family", ["poisson", "nb2"])
def test_total_family_is_an_explicit_run_configuration(
    modeling_df: pd.DataFrame, family: IndependentTotalFamily
) -> None:
    """The configured total family reaches predictions and metadata."""
    model = IndependentTotalProbabilityModel(
        independent_config=_config(total_family=family)
    ).fit(
        modeling_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(53),
    )
    result = model.predict(modeling_df)
    metadata = model.get_metadata()["model"]

    assert result.parametric_distributions is not None
    assert result.parametric_distributions["total"].family == family
    assert metadata["hyperparameters"]["total_family"] == family  # type: ignore[index]
    assert (metadata["hyperparameters"]["dispersion_alpha"] is not None) == (  # type: ignore[index]
        family == "nb2"
    )


def test_pointwise_scores_have_correct_distributional_scope(
    fitted_model: IndependentTotalProbabilityModel, modeling_df: pd.DataFrame
) -> None:
    """Pointwise scores are finite, sequential-joint, and match the joint NLL."""
    result = fitted_model.predict(
        modeling_df,
        prediction_config=PredictionConfig(include_pointwise_log_probabilities=True),
    )

    assert result.pointwise_log_probabilities is not None
    assert set(result.pointwise_log_probabilities) == {
        "total",
        *COHORT_TARGET_COLUMNS,
    }
    assert np.isfinite(
        np.column_stack(list(result.pointwise_log_probabilities.values()))
    ).all()

    from age_group_prediction.metrics import JointPredictiveNegativeLogLikelihood

    assert result.pointwise_log_probability_scope == "sequential_joint"
    joint = JointPredictiveNegativeLogLikelihood(
        interpretation="NB2 total plus conditional multinomial composition"
    ).compute(modeling_df, result)
    assert joint.value == pytest.approx(
        -np.mean(
            np.column_stack(list(result.pointwise_log_probabilities.values())).sum(
                axis=1
            )
        )
    )


def test_same_seed_reproduces_fit_predictions_and_seed_provenance(
    modeling_df: pd.DataFrame,
) -> None:
    """Two fits with the same seed give the same predictions and derived seeds."""
    models = [
        IndependentTotalProbabilityModel(independent_config=_config()).fit(
            modeling_df,
            feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
            rng=np.random.default_rng(71),
        )
        for _ in range(2)
    ]

    predictions = [model.predict(modeling_df) for model in models]
    np.testing.assert_allclose(predictions[0].total_mean, predictions[1].total_mean)
    np.testing.assert_allclose(
        predictions[0].age_group_probabilities,
        predictions[1].age_group_probabilities,
    )
    assert models[0].get_metadata()["derived_seeds"] == models[1].get_metadata()[
        "derived_seeds"
    ]


def test_bootstrap_draws_are_integer_and_reconcile_draw_by_draw(
    fitted_model: IndependentTotalProbabilityModel, modeling_df: pd.DataFrame
) -> None:
    """Predictive draws are integers whose cohorts sum to the total in each draw."""
    result = fitted_model.predict(
        modeling_df.iloc[:5],
        prediction_config=PredictionConfig(
            n_predictive_draws=3, interval_levels=(0.8,)
        ),
        rng=np.random.default_rng(91),
    )

    assert result.predictive_draws is not None
    assert result.prediction_intervals is not None
    total_draws = result.predictive_draws["total"]
    cohort_sum = sum(
        result.predictive_draws[cohort] for cohort in COHORT_TARGET_COLUMNS
    )
    np.testing.assert_array_equal(total_draws, cohort_sum)
    assert total_draws.shape == (5, 3)
    assert np.equal(total_draws, np.floor(total_draws)).all()
    for cohort in COHORT_TARGET_COLUMNS:
        assert result.prediction_intervals[cohort].shape == (5, 1, 2)


def test_public_import_paths_are_stable() -> None:
    """The model is importable from both the package and ``models``."""
    from age_group_prediction import models

    assert IndependentTotalProbabilityModel is models.IndependentTotalProbabilityModel


# --- Statistical contracts ---------------------------------------------------


@pytest.fixture(scope="module")
def signal_df() -> pd.DataFrame:
    """Return a table generated through the model's own exposure-offset form."""
    rng = np.random.default_rng(20260912)
    n, n_hoods = 240, 12
    frame = pd.DataFrame(
        {
            "building_id": np.arange(n),
            "neighborhood_id": np.repeat(np.arange(n_hoods), n // n_hoods),
            "ses": rng.normal(size=n),
            "avg_household_size": rng.uniform(1.8, 3.8, size=n),
            "median_age": rng.uniform(24.0, 58.0, size=n),
            "n_daycares_500m": rng.integers(0, 6, size=n),
            "n_apartments": rng.integers(20, 90, size=n),
            "3_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "school_status": rng.choice(["none", "existing", "planned"], size=n),
        }
    )
    apartments = frame["n_apartments"].to_numpy(dtype=float)
    ses = frame["ses"].to_numpy()
    age = (frame["median_age"].to_numpy() - 40.0) / 20.0
    totals = rng.poisson(apartments * np.exp(-1.6 + 0.30 * ses - 0.25 * age))
    logits = np.column_stack(
        [0.4 * ses - 0.6 * age, np.zeros(n), -0.3 * ses + 0.5 * age]
    )
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    counts = np.vstack(
        [rng.multinomial(int(t), p) for t, p in zip(totals, probabilities)]
    )
    for index, cohort in enumerate(COHORT_TARGET_COLUMNS):
        frame[cohort] = counts[:, index]
    frame[TOTAL_TARGET_COLUMN] = counts.sum(axis=1)
    frame.attrs["true_probabilities"] = probabilities
    return frame


def test_pointwise_keys_sum_to_the_true_joint_log_mass(
    signal_df: pd.DataFrame,
) -> None:
    """The four keys must equal NB2(total) x multinomial(cohorts | total).

    Asserting that the joint metric equals the sum of the same arrays it was
    built from is circular; scipy is the external oracle. A per-cohort score
    that dropped its multinomial coefficient would still pass the circular
    check and fail this one.
    """
    from scipy.stats import multinomial, nbinom

    from age_group_prediction.distributions import nb2_scipy_parameters

    model = IndependentTotalProbabilityModel(independent_config=_config()).fit(
        signal_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(44),
    )
    result = model.predict(
        signal_df,
        prediction_config=PredictionConfig(include_pointwise_log_probabilities=True),
    )
    assert result.pointwise_log_probabilities is not None
    assert result.pointwise_log_probability_scope == "sequential_joint"
    keys = result.pointwise_log_probabilities
    actual = np.column_stack(
        [keys["total"], *[keys[cohort] for cohort in COHORT_TARGET_COLUMNS]]
    ).sum(axis=1)

    observed_total = signal_df[TOTAL_TARGET_COLUMN].to_numpy(dtype=float)
    observed_cohorts = signal_df[list(COHORT_TARGET_COLUMNS)].to_numpy(dtype=int)
    size, probability = nb2_scipy_parameters(
        result.total_mean, model._total_fit.dispersion
    )
    expected = nbinom.logpmf(observed_total, size, probability) + np.array(
        [
            multinomial.logpmf(
                observed_cohorts[row],
                int(observed_total[row]),
                result.age_group_probabilities[row],
            )
            for row in range(len(signal_df))
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


def test_both_components_beat_their_constant_baselines(
    signal_df: pd.DataFrame,
) -> None:
    """Guard against a component that fits without using its features.

    Neither a constant total mean nor uniform probabilities is detectable from
    the shape, reconciliation and serialization assertions elsewhere in this
    file.
    """
    model = IndependentTotalProbabilityModel(independent_config=_config()).fit(
        signal_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(44),
    )
    result = model.predict(signal_df)

    observed_total = signal_df[TOTAL_TARGET_COLUMN].to_numpy(dtype=float)
    total_rmse = float(np.sqrt(np.mean((result.total_mean - observed_total) ** 2)))
    constant_rmse = float(
        np.sqrt(np.mean((observed_total.mean() - observed_total) ** 2))
    )
    assert total_rmse < constant_rmse
    assert float(np.corrcoef(result.total_mean, observed_total)[0, 1]) > 0.5

    # A constant-share baseline is NOT a sufficient comparison here: the true
    # composition averages to roughly uniform, so a predictor returning 1/3 for
    # every building scores about as well as the observed overall share. The
    # discriminating property is that predictions must VARY with the features
    # and track the truth building by building.
    true_probabilities = signal_df.attrs["true_probabilities"]
    predicted = result.age_group_probabilities
    for index, cohort in enumerate(COHORT_TARGET_COLUMNS):
        assert predicted[:, index].std() > 0.01, (
            f"{cohort} probabilities are effectively constant across buildings"
        )
        correlation = float(np.corrcoef(predicted[:, index], true_probabilities[:, index])[0, 1])
        assert correlation > 0.5, f"{cohort} probability correlation {correlation:.3f}"
    model_error = float(np.mean(np.abs(predicted - true_probabilities)))
    uniform_error = float(np.mean(np.abs(1.0 / 3.0 - true_probabilities)))
    assert model_error < uniform_error


def test_tuning_selects_from_a_recorded_search_space(
    signal_df: pd.DataFrame,
) -> None:
    """Non-degenerate bounds, so selection is real and reproducible from the run.

    Every other test in this file pins both bounds to a single point, which
    makes tuning a no-op and leaves the recorded search space untested.
    """
    config = IndependentTotalProbabilityConfig(
        total_family="nb2",
        tuning=OptunaTuningConfig(n_trials=6),
        search_space=IndependentSearchSpaceConfig(
            total_l2_penalty=(1e-6, 2.0), probability_c=(0.01, 100.0)
        ),
        tuning_folds=FoldConfig(n_folds=3),
        optimizer_max_iterations=300,
        bootstrap_replicates=2,
    )
    model = IndependentTotalProbabilityModel(independent_config=config).fit(
        signal_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(44),
    )
    selection = model.get_metadata()["model"]["diagnostics"]["selection"]

    recorded = selection["search_space"]
    assert tuple(recorded["total_l2_penalty"]) == (1e-6, 2.0)
    assert tuple(recorded["probability_c"]) == (0.01, 100.0)
    # A chosen value without its bounds is not reproducible from the record.
    for name, bounds in (
        ("total_l2_penalty", (1e-6, 2.0)),
        ("probability_c", (0.01, 100.0)),
    ):
        chosen = model.get_metadata()["model"]["hyperparameters"][name]
        assert bounds[0] <= chosen <= bounds[1]
    # Both components must be scored on the same folds.
    assert selection["fold_count"] == 3
    for component in ("total_tuning", "probability_tuning"):
        winner = next(
            trial
            for trial in selection[component]["trials"]
            if trial["number"] == selection[component]["best_trial_number"]
        )
        assert winner["state"] == "COMPLETE"


def test_calibration_applies_exactly_one_when_it_is_not_retained(
    modeling_df: pd.DataFrame,
) -> None:
    """Pin the calibration mechanism, independent of the retention threshold.

    This does not assert that the retain/reject rule is well chosen -- it is
    seed-sensitive, which is recorded as a finding -- only that a rejected
    temperature leaves probabilities untouched and a retained one never
    worsens the score it was selected on.
    """
    model = IndependentTotalProbabilityModel(independent_config=_config()).fit(
        modeling_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(44),
    )
    calibration = model.get_metadata()["model"]["calibration"]

    if calibration["retained"]:
        assert calibration["temperature"] == calibration["fitted_temperature"]
        assert calibration["selected_weighted_nll"] <= calibration["raw_weighted_nll"]
    else:
        assert calibration["temperature"] == 1.0
        assert calibration["selected_weighted_nll"] == calibration["raw_weighted_nll"]
    assert calibration["temperature"] > 0.0
    # Raw and calibrated must both be recorded for either score, or "did
    # calibration help" cannot be answered from the run.
    for key in (
        "raw_weighted_nll",
        "selected_weighted_nll",
        "raw_weighted_share_error",
        "selected_weighted_share_error",
    ):
        assert key in calibration


def test_bootstrap_skips_replicates_that_omit_a_cohort(
    concentrated_cohort_df: pd.DataFrame,
) -> None:
    """A cluster resample can omit every child of one cohort.

    The composition model is unidentified for such a replicate. It must be
    skipped and counted rather than failing a prediction the caller cannot fix.
    """
    config = IndependentTotalProbabilityConfig(
        total_family="nb2",
        tuning=OptunaTuningConfig(n_trials=2),
        search_space=IndependentSearchSpaceConfig(
            total_l2_penalty=(0.01, 0.01), probability_c=(1.0, 1.0)
        ),
        tuning_folds=FoldConfig(n_folds=2),
        optimizer_max_iterations=300,
        bootstrap_replicates=40,
    )
    model = IndependentTotalProbabilityModel(independent_config=config).fit(
        concentrated_cohort_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(1),
    )
    result = model.predict(
        concentrated_cohort_df,
        prediction_config=PredictionConfig(interval_levels=(0.8,)),
        rng=np.random.default_rng(2),
    )
    diagnostics = model.get_metadata()["model"]["diagnostics"]

    assert diagnostics["bootstrap_failed_replicates"] > 0, (
        "fixture no longer provokes a failed replicate; the skip path is untested"
    )
    assert diagnostics["bootstrap_successful_replicates"] == (
        40 - diagnostics["bootstrap_failed_replicates"]
    )
    assert result.predictive_draws is not None
    width = result.predictive_draws["total"].shape[1]
    assert width == diagnostics["bootstrap_successful_replicates"]
    cohort_sum = sum(
        result.predictive_draws[cohort] for cohort in COHORT_TARGET_COLUMNS
    )
    np.testing.assert_array_equal(result.predictive_draws["total"], cohort_sum)


def test_bootstrap_raises_when_too_many_replicates_fail(
    concentrated_cohort_df: pd.DataFrame,
) -> None:
    """Too many resamples missing a cohort make bootstrap prediction fail loudly."""
    frame = concentrated_cohort_df.copy()
    highschool = np.zeros(len(frame), dtype=int)
    highschool[frame["neighborhood_id"] == 0] = 2  # a single neighborhood
    frame["n_highschool"] = highschool
    frame[TOTAL_TARGET_COLUMN] = frame[list(COHORT_TARGET_COLUMNS)].sum(axis=1)
    config = IndependentTotalProbabilityConfig(
        total_family="nb2",
        tuning=OptunaTuningConfig(n_trials=2),
        search_space=IndependentSearchSpaceConfig(
            total_l2_penalty=(0.01, 0.01), probability_c=(1.0, 1.0)
        ),
        tuning_folds=FoldConfig(n_folds=2),
        optimizer_max_iterations=300,
        bootstrap_replicates=40,
    )
    model = IndependentTotalProbabilityModel(independent_config=config).fit(
        frame, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(1)
    )
    with pytest.raises(RuntimeError, match="omitted an age"):
        model.predict(
            frame,
            prediction_config=PredictionConfig(interval_levels=(0.8,)),
            rng=np.random.default_rng(2),
        )


@pytest.fixture(scope="module")
def concentrated_cohort_df() -> pd.DataFrame:
    """Return a table where highschool children live in only some neighborhoods."""
    rng = np.random.default_rng(9)
    n, n_hoods = 60, 6
    frame = pd.DataFrame(
        {
            "building_id": np.arange(n),
            "neighborhood_id": np.repeat(np.arange(n_hoods), n // n_hoods),
            "ses": rng.normal(size=n),
            "avg_household_size": rng.uniform(1.8, 3.8, size=n),
            "median_age": rng.uniform(24.0, 58.0, size=n),
            "n_daycares_500m": rng.integers(0, 6, size=n),
            "n_apartments": rng.integers(20, 90, size=n),
            "3_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "school_status": rng.choice(["none", "existing", "planned"], size=n),
        }
    )
    frame["n_kindergarten"] = rng.integers(1, 6, size=n)
    frame["n_elementary"] = rng.integers(1, 6, size=n)
    highschool = np.zeros(n, dtype=int)
    highschool[np.isin(frame["neighborhood_id"], [0, 1])] = 2
    frame["n_highschool"] = highschool
    frame[TOTAL_TARGET_COLUMN] = frame[list(COHORT_TARGET_COLUMNS)].sum(axis=1)
    return frame


@pytest.mark.parametrize("max_iterations", [1, 2])
def test_total_optimizers_report_non_convergence(max_iterations: int) -> None:
    """Both total-count optimizers raise when they run out of iterations."""
    from age_group_prediction.models.count_regression import (
        _fit_poisson_regression,
        _fit_total_regression,
    )

    features = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    observed = np.arange(1, 21, dtype=float)
    log_exposure = pd.Series(np.log(np.full(20, 25.0)))
    config = IndependentTotalProbabilityConfig(
        optimizer_max_iterations=max_iterations
    )
    for family, message in (("nb2", "NB2"), ("poisson", "Poisson")):
        with pytest.raises(RuntimeError, match=f"{message} optimization failed"):
            _fit_total_regression(
                family,
                features,
                observed,
                log_exposure,
                l2_penalty=0.01,
                config=config,
            )
    assert _fit_poisson_regression is not None

def test_calibration_retention_uses_a_likelihood_ratio_test(
    fitted_model: IndependentTotalProbabilityModel,
) -> None:
    """The retention decision must be the recorded 1-df likelihood-ratio test."""
    diagnostics = fitted_model.get_metadata()["model"]["calibration"]

    assert diagnostics["retention_rule"] == "likelihood_ratio_test_1df"
    statistic = diagnostics["likelihood_ratio_statistic"]
    critical = diagnostics["likelihood_ratio_critical_value"]
    # chi2(1) at the default 5% level.
    assert critical == pytest.approx(3.841458820694124)
    assert diagnostics["retained"] is bool(statistic > critical)
    # The statistic is 2 * W * (raw - fitted) on a per-child mean NLL.
    expected = (
        2.0
        * diagnostics["weighted_child_count"]
        * (diagnostics["raw_weighted_nll"] - diagnostics["selected_weighted_nll"])
    )
    if diagnostics["retained"]:
        assert statistic == pytest.approx(expected, rel=1e-9)


def test_noise_scale_calibration_gain_is_rejected(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    """A gain far below the chi-square threshold must not retain a temperature.

    This is the defect the likelihood-ratio test replaced: ``T = 1`` is interior
    to the search bounds, so the fitted NLL is never worse than the raw NLL, and
    the former ``1e-6`` improvement threshold accepted differences at that noise
    scale as evidence. On perfectly calibrated logits the old rule retained 143
    of 150 replicates; this one retained 8 of 150.
    """
    from scipy.optimize import OptimizeResult

    # Patched in the module whose `_fit_calibration_temperature` calls it.
    from age_group_prediction.models import probability_calibration as module

    real_minimize = module.minimize_scalar
    captured: dict[str, float] = {}

    def tiny_improvement(function, **kwargs):
        """Report T = 1.35 with a per-child NLL gain of only 1e-5."""
        result = real_minimize(function, **kwargs)
        # Report a temperature away from one that buys a negligible gain: the
        # per-child NLL improves by 1e-5, well above the retired 1e-6 tolerance
        # and far below the chi-square threshold at this sample size.
        raw = float(function(0.0))
        captured["raw"] = raw
        return OptimizeResult(
            x=float(np.log(1.35)), fun=raw - 1e-5, success=True, nit=result.nit
        )

    monkeypatch.setattr(module, "minimize_scalar", tiny_improvement)

    model = IndependentTotalProbabilityModel(independent_config=_config()).fit(
        modeling_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(44),
    )
    diagnostics = model.get_metadata()["model"]["calibration"]

    assert diagnostics["fitted_temperature"] == pytest.approx(1.35)
    assert diagnostics["likelihood_ratio_statistic"] < 3.841458820694124
    assert diagnostics["retained"] is False
    assert diagnostics["temperature"] == 1.0


def test_large_calibration_gain_is_retained(
    monkeypatch: pytest.MonkeyPatch, modeling_df: pd.DataFrame
) -> None:
    """A genuine improvement must still be retained, so the test is not vacuous."""
    from scipy.optimize import OptimizeResult

    # Patched in the module whose `_fit_calibration_temperature` calls it.
    from age_group_prediction.models import probability_calibration as module

    real_minimize = module.minimize_scalar

    def large_improvement(function, **kwargs):
        """Report T = 1.8 with a per-child NLL gain of 0.5."""
        result = real_minimize(function, **kwargs)
        raw = float(function(0.0))
        return OptimizeResult(
            x=float(np.log(1.8)), fun=raw - 0.5, success=True, nit=result.nit
        )

    monkeypatch.setattr(module, "minimize_scalar", large_improvement)

    model = IndependentTotalProbabilityModel(independent_config=_config()).fit(
        modeling_df,
        feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
        rng=np.random.default_rng(44),
    )
    diagnostics = model.get_metadata()["model"]["calibration"]

    assert diagnostics["likelihood_ratio_statistic"] > 3.841458820694124
    assert diagnostics["retained"] is True
    assert diagnostics["temperature"] == pytest.approx(1.8)
