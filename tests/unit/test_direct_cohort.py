"""Focused tests for the direct cohort model's distributional trees."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from age_group_prediction import (
    COHORT_TARGET_COLUMNS,
    DEFAULT_TREE_FEATURE_SPEC,
    TOTAL_TARGET_COLUMN,
    DirectCohortConfig,
    DirectCohortModel,
    FeatureSpec,
    FoldConfig,
    LightGBMSearchSpaceConfig,
    OptunaTuningConfig,
    PredictionConfig,
)
from age_group_prediction.feature_engineering import FittedFeatureTransformer
from age_group_prediction.metrics import (
    CompositionLogLoss,
    MeanAbsoluteError,
    ParametricPredictiveNegativeLogLikelihood,
    ReconciliationError,
)


def _make_config(*, family: str = "poisson") -> DirectCohortConfig:
    """Return a small, fast-fitting direct cohort model configuration for tests."""
    return DirectCohortConfig(
        family=family,  # type: ignore[arg-type]
        tuning=OptunaTuningConfig(n_trials=2),
        search_space=LightGBMSearchSpaceConfig(
            num_leaves=(4, 6),
            max_depth=(2, 3),
            min_child_samples=(2, 4),
            learning_rate=(0.05, 0.15),
            n_estimators=(5, 12),
            reg_alpha=(1e-8, 0.1),
            reg_lambda=(1e-8, 0.1),
            min_split_gain=(0.0, 0.1),
            subsample=(0.8, 1.0),
            colsample_bytree=(0.8, 1.0),
            nb2_dispersion=(0.05, 1.0),
        ),
        tuning_folds=FoldConfig(n_folds=2),
        bootstrap_replicates=3,
    )


@pytest.fixture(scope="module")
def modeling_df() -> pd.DataFrame:
    """Return a small deterministic modeling table spanning two neighborhoods."""
    rng = np.random.default_rng(0)
    n = 24
    df = pd.DataFrame(
        {
            "building_id": np.arange(n),
            "neighborhood_id": np.repeat([0, 1], n // 2),
            "ses": rng.normal(size=n),
            "avg_household_size": rng.uniform(1.5, 4.0, size=n),
            "median_age": rng.uniform(20.0, 60.0, size=n),
            "n_daycares_500m": rng.integers(0, 5, size=n),
            "n_apartments": rng.integers(5, 40, size=n),
            "3_rooms_share": rng.uniform(0.0, 0.3, size=n),
            "4_rooms_share": rng.uniform(0.0, 0.3, size=n),
            "5_rooms_share": rng.uniform(0.0, 0.3, size=n),
            "school_status": rng.choice(["none", "existing", "planned"], size=n),
            "n_kindergarten": rng.integers(0, 5, size=n),
            "n_elementary": rng.integers(0, 5, size=n),
            "n_highschool": rng.integers(0, 5, size=n),
        }
    )
    df[TOTAL_TARGET_COLUMN] = df[list(COHORT_TARGET_COLUMNS)].sum(axis=1)
    return df


def _learning_config(*, family: str = "poisson") -> DirectCohortConfig:
    """Return a configuration with enough capacity to fit a real signal.

    `_make_config` is deliberately tiny so the contract tests stay fast; it is
    too weak to learn anything, which is exactly why the recovery test needs
    its own.
    """
    return DirectCohortConfig(
        family=family,  # type: ignore[arg-type]
        tuning=OptunaTuningConfig(n_trials=2),
        search_space=LightGBMSearchSpaceConfig(
            num_leaves=(7, 15),
            max_depth=(3, 4),
            min_child_samples=(5, 10),
            learning_rate=(0.1, 0.2),
            n_estimators=(80, 150),
            reg_alpha=(1e-8, 0.1),
            reg_lambda=(1e-8, 0.1),
            min_split_gain=(0.0, 0.05),
            subsample=(0.9, 1.0),
            colsample_bytree=(0.9, 1.0),
            nb2_dispersion=(1e-3, 1.0),
        ),
        tuning_folds=FoldConfig(n_folds=2),
        bootstrap_replicates=2,
    )


@pytest.fixture(scope="module")
def signal_df() -> pd.DataFrame:
    """Return a table whose cohort counts genuinely depend on the features."""
    rng = np.random.default_rng(20260912)
    n = 160
    df = pd.DataFrame(
        {
            "building_id": np.arange(n),
            "neighborhood_id": np.repeat(np.arange(8), n // 8),
            "ses": rng.normal(size=n),
            "avg_household_size": rng.uniform(1.8, 3.8, size=n),
            "median_age": rng.uniform(24.0, 58.0, size=n),
            "n_daycares_500m": rng.integers(0, 5, size=n),
            "n_apartments": rng.integers(20, 90, size=n),
            "3_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=n),
            "school_status": rng.choice(["none", "existing", "planned"], size=n),
        }
    )
    apartments = df["n_apartments"].to_numpy() / 50.0
    ses = df["ses"].to_numpy()
    age = (df["median_age"].to_numpy() - 40.0) / 20.0
    means = {
        "n_kindergarten": np.exp(0.7 + 0.9 * np.log(apartments) + 0.35 * ses - 0.5 * age),
        "n_elementary": np.exp(1.1 + 0.8 * np.log(apartments) + 0.20 * ses - 0.2 * age),
        "n_highschool": np.exp(0.5 + 0.7 * np.log(apartments) - 0.15 * ses + 0.4 * age),
    }
    for cohort, mean in means.items():
        df[cohort] = rng.poisson(mean)
    df[TOTAL_TARGET_COLUMN] = df[list(COHORT_TARGET_COLUMNS)].sum(axis=1)
    return df


@pytest.mark.parametrize("family", ["poisson", "nb2", "normal"])
def test_fitted_model_beats_an_intercept_only_baseline(
    signal_df: pd.DataFrame, family: str
) -> None:
    """Guard against a model that fits without learning.

    Every other test here uses targets drawn independently of the features, so
    a predictor returning a constant passes all of them. This one fails unless
    the fitted trees actually use the features.
    """
    model = DirectCohortModel(direct_cohort_config=_learning_config(family=family)).fit(
        signal_df,
        feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        rng=np.random.default_rng(2024),
    )
    result = model.predict(signal_df)

    for index, cohort in enumerate(COHORT_TARGET_COLUMNS):
        observed = signal_df[cohort].to_numpy(dtype=float)
        predicted = result.cohort_means[:, index]
        model_rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
        intercept_rmse = float(np.sqrt(np.mean((observed.mean() - observed) ** 2)))
        assert model_rmse < intercept_rmse, (
            f"{family}/{cohort}: RMSE {model_rmse:.3f} did not beat the "
            f"intercept-only baseline {intercept_rmse:.3f}"
        )
        # A constant predictor has no covariance with the outcome at all.
        assert float(np.corrcoef(predicted, observed)[0, 1]) > 0.5


@pytest.fixture(scope="module")
def fitted_model(modeling_df: pd.DataFrame) -> DirectCohortModel:
    """Return a direct cohort model instance fitted once and reused by read-only tests."""
    model = DirectCohortModel(direct_cohort_config=_make_config())
    return model.fit(
        modeling_df, feature_spec=DEFAULT_TREE_FEATURE_SPEC, rng=np.random.default_rng(123)
    )


# --- Base lifecycle integration ---------------------------------------------


def test_public_import_paths_are_stable() -> None:
    from age_group_prediction import models

    assert DirectCohortModel is models.DirectCohortModel


def test_predict_and_evaluate_before_fit_raise_clearly(modeling_df: pd.DataFrame) -> None:
    model = DirectCohortModel(direct_cohort_config=_make_config())

    with pytest.raises(RuntimeError, match="fitted before predict"):
        model.predict(modeling_df)
    with pytest.raises(RuntimeError, match="fitted before evaluate"):
        model.evaluate(modeling_df, metrics=[])


def test_fit_rejects_invalid_feature_spec(modeling_df: pd.DataFrame) -> None:
    model = DirectCohortModel(direct_cohort_config=_make_config())
    leaking_spec = FeatureSpec(
        component="tree",
        numeric_features=("ses", "n_kindergarten"),
        categorical_features=(),
    )

    with pytest.raises(ValueError, match="are not modeled"):
        model.fit(modeling_df, feature_spec=leaking_spec, rng=np.random.default_rng(0))


def test_failed_fit_clears_all_state(modeling_df: pd.DataFrame) -> None:
    model = DirectCohortModel(direct_cohort_config=_make_config())
    broken_df = modeling_df.drop(columns=["n_kindergarten"])

    with pytest.raises(ValueError, match="Missing cohort target columns"):
        model.fit(broken_df, feature_spec=DEFAULT_TREE_FEATURE_SPEC, rng=np.random.default_rng(0))

    assert not model.is_fitted
    assert model.get_metadata()["fit_duration_seconds"] is None


def test_fit_owns_final_preprocessing_on_the_full_training_partition(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame
) -> None:
    metadata = fitted_model.feature_transformer.get_metadata()
    assert metadata["fit_row_count"] == len(modeling_df)
    feature_names = fitted_model.feature_transformer.get_feature_names_out()
    assert not set(COHORT_TARGET_COLUMNS).intersection(feature_names)
    assert TOTAL_TARGET_COLUMN not in feature_names
    assert "building_id" not in feature_names


# --- Output contract ---------------------------------------------------------


def test_predictions_are_finite_nonnegative_and_reconciled(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame
) -> None:
    result = fitted_model.predict(modeling_df)

    assert result.cohort_means.shape == (len(modeling_df), 3)
    assert np.isfinite(result.cohort_means).all()
    assert (result.cohort_means >= 0).all()
    np.testing.assert_allclose(result.total_mean, result.cohort_means.sum(axis=1))
    assert np.max(result.reconciliation_error) <= 1e-9
    np.testing.assert_allclose(result.age_group_probabilities.sum(axis=1), 1.0)


def test_zero_total_probabilities_fall_back_to_uniform(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force every cohort estimator to predict zero, then verify the uniform fallback."""
    eval_df = modeling_df.iloc[:2]
    zero_predictions = np.zeros(len(eval_df))
    for cohort in COHORT_TARGET_COLUMNS:
        monkeypatch.setattr(
            fitted_model._fitted_boosters[cohort],
            "predict",
            lambda _features, **_kwargs: zero_predictions,
        )

    result = fitted_model.predict(eval_df)

    np.testing.assert_allclose(result.age_group_probabilities, np.full((2, 3), 1 / 3))


def test_parametric_distributions_declare_independent_poisson_targets(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame
) -> None:
    result = fitted_model.predict(modeling_df)

    assert result.parametric_distributions is not None
    expected_targets = {"total", *COHORT_TARGET_COLUMNS}
    assert set(result.parametric_distributions) == expected_targets
    assert {spec.family for spec in result.parametric_distributions.values()} == {"poisson"}


def test_metadata_is_json_serializable_and_declares_likelihood(
    fitted_model: DirectCohortModel,
) -> None:
    metadata = fitted_model.get_metadata()
    json.dumps(metadata)

    assert metadata["model_class"] == "DirectCohortModel"
    assert metadata["model"]["likelihood"] == (
        "three independent Poisson cohort likelihoods"
    )
    assert set(metadata["model"]["hyperparameters"]) == set(COHORT_TARGET_COLUMNS)
    assert metadata["model"]["dependency_versions"]["lightgbm"]


# --- Leakage and selection ---------------------------------------------------


def test_every_cohort_records_a_completed_tuning_study(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame
) -> None:
    diagnostics = fitted_model.get_metadata()["model"]["diagnostics"]["tuning"]
    for cohort in COHORT_TARGET_COLUMNS:
        assert len(diagnostics[cohort]["trials"]) == 2
        assert np.isfinite(diagnostics[cohort]["best_value"])


def test_tuning_folds_come_only_from_supplied_training_data(
    modeling_df: pd.DataFrame,
) -> None:
    """Every fold transformer must learn from training rows only.

    Recording which rows each fold's transformer is fitted on is the only way
    to see this: the fold split happens inside `fit`, so a leak would be
    invisible from the outside.
    """
    fitted_on: list[set[int]] = []
    original_fit_transform = FittedFeatureTransformer.fit_transform

    def recording_fit_transform(
        self: FittedFeatureTransformer, frame: pd.DataFrame, *args: object, **kwargs: object
    ) -> pd.DataFrame:
        fitted_on.append(set(frame["building_id"]))
        return original_fit_transform(self, frame, *args, **kwargs)

    training_ids = set(modeling_df["building_id"])
    model = DirectCohortModel(direct_cohort_config=_make_config())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            FittedFeatureTransformer, "fit_transform", recording_fit_transform
        )
        model.fit(
            modeling_df,
            feature_spec=DEFAULT_TREE_FEATURE_SPEC,
            rng=np.random.default_rng(123),
        )

    assert fitted_on, "no transformer was fitted"
    assert all(seen <= training_ids for seen in fitted_on)
    # The final transformer sees the whole partition; the fold ones see less.
    fold_fits = [seen for seen in fitted_on if seen != training_ids]
    assert len(fold_fits) == _make_config().tuning_folds.n_folds
    assert all(seen < training_ids for seen in fold_fits)


@pytest.mark.parametrize("family", ["normal", "nb2"])
def test_alternative_families_fit_with_matching_predictive_contracts(
    modeling_df: pd.DataFrame, family: str
) -> None:
    model = DirectCohortModel(direct_cohort_config=_make_config(family=family)).fit(
        modeling_df,
        feature_spec=DEFAULT_TREE_FEATURE_SPEC,
        rng=np.random.default_rng(5),
    )
    result = model.predict(
        modeling_df,
        prediction_config=PredictionConfig(
            n_predictive_draws=2, include_pointwise_log_probabilities=True
        ),
    )

    assert result.parametric_distributions is not None
    assert result.pointwise_log_probabilities is not None
    assert result.predictive_draws is not None
    assert {spec.family for spec in result.parametric_distributions.values()} == {family}
    if family == "normal":
        assert "total" in result.parametric_distributions
    else:
        assert "total" not in result.parametric_distributions
        for cohort in COHORT_TARGET_COLUMNS:
            assert np.equal(
                result.predictive_draws[cohort],
                np.floor(result.predictive_draws[cohort]),
            ).all()
    np.testing.assert_allclose(
        result.predictive_draws["total"],
        sum(result.predictive_draws[cohort] for cohort in COHORT_TARGET_COLUMNS),
    )
    assert model.get_metadata()["model"]["diagnostics"]["family"] == family


# --- RNG and uncertainty -----------------------------------------------------


@pytest.mark.parametrize("family", ["poisson", "normal", "nb2"])
def test_same_seed_reproduces_fit_and_predictions(
    modeling_df: pd.DataFrame, family: str
) -> None:
    model_a = DirectCohortModel(direct_cohort_config=_make_config(family=family)).fit(
        modeling_df, feature_spec=DEFAULT_TREE_FEATURE_SPEC, rng=np.random.default_rng(7)
    )
    model_b = DirectCohortModel(direct_cohort_config=_make_config(family=family)).fit(
        modeling_df, feature_spec=DEFAULT_TREE_FEATURE_SPEC, rng=np.random.default_rng(7)
    )

    result_a = model_a.predict(modeling_df, rng=np.random.default_rng(9))
    result_b = model_b.predict(modeling_df, rng=np.random.default_rng(9))

    np.testing.assert_allclose(result_a.cohort_means, result_b.cohort_means)
    assert model_a.get_metadata()["derived_seeds"] == model_b.get_metadata()["derived_seeds"]
    assert (
        model_a.get_metadata()["model"]["diagnostics"]["tuning"]
        == model_b.get_metadata()["model"]["diagnostics"]["tuning"]
    )


def test_different_seed_changes_bootstrap_draws(modeling_df: pd.DataFrame) -> None:
    model = DirectCohortModel(direct_cohort_config=_make_config()).fit(
        modeling_df, feature_spec=DEFAULT_TREE_FEATURE_SPEC, rng=np.random.default_rng(7)
    )
    prediction_config = PredictionConfig(n_predictive_draws=3)

    result_a = model.predict(modeling_df, prediction_config=prediction_config, rng=np.random.default_rng(1))
    result_b = model.predict(modeling_df, prediction_config=prediction_config, rng=np.random.default_rng(2))

    assert result_a.predictive_draws is not None
    assert result_b.predictive_draws is not None
    assert not np.allclose(
        result_a.predictive_draws["total"], result_b.predictive_draws["total"]
    )


def test_predictive_draws_and_intervals_are_well_formed(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame
) -> None:
    prediction_config = PredictionConfig(n_predictive_draws=4, interval_levels=(0.8,))
    result = fitted_model.predict(
        modeling_df, prediction_config=prediction_config, rng=np.random.default_rng(11)
    )

    assert result.predictive_draws is not None
    assert result.prediction_intervals is not None
    expected_targets = {"total", *COHORT_TARGET_COLUMNS}
    assert set(result.predictive_draws) == expected_targets
    assert set(result.prediction_intervals) == expected_targets
    for target in expected_targets:
        assert result.predictive_draws[target].shape == (len(modeling_df), 4)
        assert result.prediction_intervals[target].shape == (len(modeling_df), 1, 2)
        assert (
            result.prediction_intervals[target][..., 0]
            <= result.prediction_intervals[target][..., 1]
        ).all()
    total_draws = result.predictive_draws["total"]
    cohort_draw_sum = sum(result.predictive_draws[c] for c in COHORT_TARGET_COLUMNS)
    np.testing.assert_allclose(total_draws, cohort_draw_sum)


def test_interval_request_without_explicit_draws_returns_full_bootstrap_sample(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame
) -> None:
    prediction_config = PredictionConfig(interval_levels=(0.8,))
    result = fitted_model.predict(
        modeling_df, prediction_config=prediction_config, rng=np.random.default_rng(3)
    )

    assert result.predictive_draws is not None
    replicates = fitted_model.direct_cohort_config.bootstrap_replicates
    assert result.predictive_draws["total"].shape == (len(modeling_df), replicates)


def test_pointwise_log_probabilities_require_observed_targets(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame
) -> None:
    prediction_config = PredictionConfig(include_pointwise_log_probabilities=True)
    features_only_df = modeling_df.drop(columns=[TOTAL_TARGET_COLUMN])

    with pytest.raises(ValueError, match="require observed targets"):
        fitted_model.predict(features_only_df, prediction_config=prediction_config)

    result = fitted_model.predict(modeling_df, prediction_config=prediction_config)
    assert result.pointwise_log_probabilities is not None
    assert set(result.pointwise_log_probabilities) == {"total", *COHORT_TARGET_COLUMNS}
    # Marginal cohort scores plus a convolved total cannot be summed jointly.
    assert result.pointwise_log_probability_scope == "marginal"


# --- Shared evaluation integration -------------------------------------------


def test_evaluate_delegates_to_shared_metrics(
    fitted_model: DirectCohortModel, modeling_df: pd.DataFrame
) -> None:
    metrics = [
        MeanAbsoluteError(TOTAL_TARGET_COLUMN),
        CompositionLogLoss(),
        ReconciliationError("max"),
        ParametricPredictiveNegativeLogLikelihood(
            target=TOTAL_TARGET_COLUMN,
            interpretation="independent Poisson total likelihood",
        ),
    ]

    evaluation = fitted_model.evaluate(modeling_df, metrics=metrics)

    assert set(evaluation.metrics_df["metric_name"]) == {
        "mae",
        "composition_log_loss",
        "max_reconciliation_error",
        "predictive_nll",
    }
