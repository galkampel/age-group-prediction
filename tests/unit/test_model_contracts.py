"""Tests for shared model lifecycle and result contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from age_group_prediction import (
    BaseAgeGroupModel,
    EvaluationResult,
    ExperimentConfig,
    FeatureSpec,
    ParametricDistributionSpec,
    PredictionConfig,
    PredictionResult,
    load_experiment_config,
)
from age_group_prediction.metrics import MeanAbsoluteError

COHORT_NAMES = ("n_kindergarten", "n_elementary", "n_highschool")
FEATURE_SPEC = FeatureSpec(
    component="total_count",
    numeric_features=("ses",),
    categorical_features=(),
    scale_numeric=True,
    exposure_column="n_apartments",
)


@pytest.fixture
def modeling_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": [11, 12, 13, 14],
            "ses": [1.0, 2.0, 3.0, 4.0],
            "n_apartments": [10, 20, 30, 40],
            "n_kindergarten": [1, 2, 3, 4],
            "n_elementary": [2, 3, 4, 5],
            "n_highschool": [1, 1, 2, 2],
            "n_children_total": [4, 6, 9, 11],
        }
    )


@pytest.fixture(scope="module")
def runtime_config() -> ExperimentConfig:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "modeling.toml"
    return load_experiment_config(config_path)


def test_result_contract_import_paths_are_compatible() -> None:
    """Public and legacy imports resolve to the canonical result classes."""
    from age_group_prediction import models, results
    from age_group_prediction.models import base

    assert PredictionResult is models.PredictionResult is base.PredictionResult
    assert PredictionResult is results.PredictionResult
    assert EvaluationResult is models.EvaluationResult is base.EvaluationResult
    assert EvaluationResult is results.EvaluationResult
    assert ParametricDistributionSpec is models.ParametricDistributionSpec
    assert ParametricDistributionSpec is base.ParametricDistributionSpec
    assert ParametricDistributionSpec is results.ParametricDistributionSpec


class SyntheticModel(BaseAgeGroupModel):
    def __init__(self, *, runtime_config: ExperimentConfig) -> None:
        super().__init__(
            default_rng_seed=runtime_config.randomness.default_seed,
            default_prediction_config=runtime_config.prediction,
            prediction_validation_config=runtime_config.prediction_validation,
        )
        self.fit_feature_mean: float | None = None
        self.fit_seed: int | None = None

    def _reset_model_state(self) -> None:
        self.fit_feature_mean = None
        self.fit_seed = None

    def _fit_model(
        self,
        *,
        train_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        rng: np.random.Generator,
    ) -> None:
        if "fail_fit" in train_df:
            raise ValueError("synthetic fit failure")
        assert log_exposure is not None
        self.fit_feature_mean = float(features["ses"].mean())
        self.fit_seed = self._derive_backend_seed(rng, purpose="synthetic-fit")

    def _predict_model(
        self,
        *,
        eval_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        prediction_config: PredictionConfig,
        rng: np.random.Generator,
    ) -> PredictionResult:
        del features, log_exposure
        child_seed = self._derive_backend_seed(rng, purpose="synthetic-predict")
        jitter = np.random.default_rng(child_seed).uniform(0.0, 0.01, len(eval_df))
        total_mean = eval_df["n_children_total"].to_numpy(dtype=float) + jitter
        probabilities = np.array([0.2, 0.3, 0.5])
        cohort_means = total_mean[:, None] * probabilities
        draws = None
        if prediction_config.n_predictive_draws:
            draws = {
                "total": np.repeat(
                    total_mean[:, None],
                    prediction_config.n_predictive_draws,
                    axis=1,
                )
            }
        return PredictionResult.from_means(
            building_ids=eval_df["building_id"].to_numpy(),
            cohort_names=COHORT_NAMES,
            total_mean=total_mean,
            cohort_means=cohort_means,
            predictive_draws=draws,
            parametric_distributions={"total": ParametricDistributionSpec("poisson")},
            validation_config=self.prediction_validation_config,
        )

    def _get_model_metadata(self) -> Mapping[str, object]:
        return {
            "implementation_version": "synthetic-v1",
            "likelihood": "deterministic",
            "parameterization": "cohort proportions",
            "hyperparameters": {},
            "priors": None,
            "calibration": None,
            "dependency_versions": {"numpy": np.__version__},
            "uncertainty_method": "none",
            "diagnostics": {"fit_feature_mean": self.fit_feature_mean},
        }


class SelectingSyntheticModel(SyntheticModel):
    def _select_feature_spec(
        self,
        *,
        train_df: pd.DataFrame,
        feature_spec: FeatureSpec,
        rng: np.random.Generator,
    ) -> FeatureSpec:
        del train_df, rng
        return replace(feature_spec, ses_form="quadratic")


def test_abstract_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseAgeGroupModel()  # type: ignore[abstract]


def test_prediction_config_is_immutable_and_validated() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        PredictionConfig(n_predictive_draws=-1)
    with pytest.raises(ValueError, match="unique"):
        PredictionConfig(interval_levels=(0.8, 0.8))
    with pytest.raises(ValueError, match="between"):
        PredictionConfig(interval_levels=(1.0,))
    with pytest.raises(AttributeError):
        PredictionConfig().n_predictive_draws = 2  # type: ignore[misc]


def test_unfitted_operations_fail_clearly(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    model = SyntheticModel(runtime_config=runtime_config)

    with pytest.raises(RuntimeError, match="fitted before predict"):
        model.predict(modeling_df)
    with pytest.raises(RuntimeError, match="fitted before evaluate"):
        model.evaluate(modeling_df, metrics=[])


def test_fit_owns_preprocessing_and_returns_self(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    fit_df = modeling_df.iloc[:3]
    model = SyntheticModel(runtime_config=runtime_config)

    assert model.fit(fit_df, feature_spec=FEATURE_SPEC) is model
    assert model.fit_feature_mean == pytest.approx(0.0, abs=1e-12)
    assert model.feature_transformer.get_metadata()["fit_row_count"] == len(fit_df)

    model.predict(modeling_df.iloc[3:])
    assert model.feature_transformer.get_metadata()["fit_row_count"] == len(fit_df)


def test_pre_fit_selection_controls_final_preprocessing_and_metadata(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    model = SelectingSyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )

    assert "ses_squared" in model.feature_transformer.get_feature_names_out()
    assert model.get_metadata()["feature_spec"]["ses_form"] == "quadratic"  # type: ignore[index]


def test_failed_refit_clears_all_fitted_state(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    model = SyntheticModel(runtime_config=runtime_config)
    model.fit(modeling_df, feature_spec=FEATURE_SPEC)
    failing_df = modeling_df.assign(fail_fit=True)

    with pytest.raises(ValueError, match="synthetic fit failure"):
        model.fit(failing_df, feature_spec=FEATURE_SPEC)

    assert not model.is_fitted
    assert model.fit_seed is None
    assert model.get_metadata()["fit_duration_seconds"] is None
    with pytest.raises(RuntimeError, match="fitted before predict"):
        model.predict(modeling_df)


def test_operation_durations_are_recorded_only_on_success(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    """Fit/predict/evaluate durations should commit only when the operation succeeds."""
    model = SyntheticModel(runtime_config=runtime_config)
    model.fit(modeling_df, feature_spec=FEATURE_SPEC)
    model.predict(modeling_df)
    model.evaluate(modeling_df, metrics=[MeanAbsoluteError("n_children_total")])

    metadata = model.get_metadata()
    assert metadata["fit_duration_seconds"] >= 0  # type: ignore[operator]
    assert metadata["predict_duration_seconds"] >= 0  # type: ignore[operator]
    assert metadata["evaluate_duration_seconds"] >= 0  # type: ignore[operator]

    failing_df = modeling_df.assign(fail_fit=True)
    with pytest.raises(ValueError, match="synthetic fit failure"):
        model.fit(failing_df, feature_spec=FEATURE_SPEC)

    cleared_metadata = model.get_metadata()
    assert cleared_metadata["fit_duration_seconds"] is None
    assert cleared_metadata["predict_duration_seconds"] is None
    assert cleared_metadata["evaluate_duration_seconds"] is None


def test_prediction_shapes_order_and_optional_draws(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    model = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )
    result = model.predict(
        modeling_df,
        prediction_config=PredictionConfig(n_predictive_draws=5),
    )

    np.testing.assert_array_equal(result.building_ids, modeling_df["building_id"])
    assert result.total_mean.shape == (len(modeling_df),)
    assert result.cohort_means.shape == (len(modeling_df), len(COHORT_NAMES))
    np.testing.assert_allclose(result.age_group_probabilities.sum(axis=1), 1.0)
    assert np.max(result.reconciliation_error) <= 1e-9
    assert result.predictive_draws is not None
    assert result.predictive_draws["total"].shape == (len(modeling_df), 5)


def test_zero_total_probabilities_use_uniform_fallback() -> None:
    result = PredictionResult.from_means(
        building_ids=np.array([1]),
        cohort_names=COHORT_NAMES,
        total_mean=np.array([0.0]),
        cohort_means=np.zeros((1, 3)),
    )

    np.testing.assert_allclose(result.age_group_probabilities, [[1 / 3] * 3])


def test_prediction_result_defensively_copies_arrays() -> None:
    totals = np.array([3.0])
    cohorts = np.array([[1.0, 1.0, 1.0]])
    result = PredictionResult.from_means(
        building_ids=np.array([1]),
        cohort_names=COHORT_NAMES,
        total_mean=totals,
        cohort_means=cohorts,
    )
    totals[0] = 9.0
    cohorts[0, 0] = 9.0

    assert result.total_mean[0] == 3.0
    assert result.cohort_means[0, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        result.total_mean[0] = 2.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"total_mean": np.array([-1.0])}, "nonnegative"),
        ({"cohort_means": np.ones((1, 2))}, "shape"),
        ({"age_group_probabilities": np.array([[0.2, 0.2, 0.2]])}, "sum to one"),
        ({"reconciliation_error": np.array([0.5])}, "does not match"),
        # Truthfully reported but unreconciled means: only the tolerance check
        # can refuse this, so it guards cohort/total accounting on its own.
        (
            {"total_mean": np.array([3.5]), "reconciliation_error": np.array([0.5])},
            "do not reconcile",
        ),
    ],
)
def test_prediction_result_rejects_invalid_values(
    changes: dict[str, np.ndarray], message: str
) -> None:
    values = {
        "building_ids": np.array([1]),
        "cohort_names": COHORT_NAMES,
        "total_mean": np.array([3.0]),
        "cohort_means": np.ones((1, 3)),
        "age_group_probabilities": np.full((1, 3), 1 / 3),
        "reconciliation_error": np.array([0.0]),
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        PredictionResult(**values)  # type: ignore[arg-type]


def test_optional_prediction_payloads_validate_targets_and_shapes() -> None:
    with pytest.raises(ValueError, match="Unknown predictive draws targets"):
        PredictionResult.from_means(
            building_ids=np.array([1]),
            cohort_names=COHORT_NAMES,
            total_mean=np.array([3.0]),
            cohort_means=np.ones((1, 3)),
            predictive_draws={"unknown": np.ones((1, 2))},
        )
    with pytest.raises(ValueError, match="invalid shape"):
        PredictionResult.from_means(
            building_ids=np.array([1]),
            cohort_names=COHORT_NAMES,
            total_mean=np.array([3.0]),
            cohort_means=np.ones((1, 3)),
            prediction_intervals={"total": np.ones((1, 2))},
            interval_levels=(0.8,),
        )


def test_parametric_distribution_payload_validation_and_copying() -> None:
    result = PredictionResult.from_means(
        building_ids=np.array([1, 2]),
        cohort_names=COHORT_NAMES,
        total_mean=np.array([3.0, 4.0]),
        cohort_means=np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
        parametric_distributions={
            "total": ParametricDistributionSpec("nb2", dispersion=0.25),
            "n_kindergarten": ParametricDistributionSpec(
                "normal", scale=np.array([1.0, 2.0])
            ),
        },
    )

    assert result.parametric_distributions is not None
    nb2_dispersion = result.parametric_distributions["total"].dispersion
    assert isinstance(nb2_dispersion, np.ndarray)
    np.testing.assert_allclose(nb2_dispersion, [0.25, 0.25])
    with pytest.raises(ValueError, match="read-only"):
        nb2_dispersion[0] = 0.1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"total": ParametricDistributionSpec("nb2")}, "require a positive dispersion"),
        (
            {"total": ParametricDistributionSpec("normal")},
            "require a positive scale",
        ),
        (
            {"total": ParametricDistributionSpec("poisson", scale=np.array([1.0]))},
            "must not define dispersion/scale",
        ),
        (
            {"unknown": ParametricDistributionSpec("poisson")},
            "Unknown parametric distribution targets",
        ),
    ],
)
def test_parametric_distribution_payload_rejects_invalid_inputs(
    payload: dict[str, ParametricDistributionSpec],
    message: str,
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        PredictionResult.from_means(
            building_ids=np.array([1]),
            cohort_names=COHORT_NAMES,
            total_mean=np.array([3.0]),
            cohort_means=np.ones((1, 3)),
            parametric_distributions=payload,
        )


def test_evaluation_result_and_metadata_are_serializable(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    model = SyntheticModel(runtime_config=runtime_config)
    json.dumps(model.get_metadata())
    model.fit(modeling_df, feature_spec=FEATURE_SPEC)

    evaluation = model.evaluate(
        modeling_df, metrics=[MeanAbsoluteError("n_children_total")]
    )
    assert list(evaluation.metrics_df.columns) == [
        "metric_name",
        "value",
        "aggregation_level",
        "sample_count",
        "target",
    ]
    assert evaluation.metrics_df.loc[0, "sample_count"] == len(modeling_df)
    assert evaluation.metrics_df.loc[0, "metric_name"] == "mae"
    assert evaluation.metrics_df["value"].to_numpy(dtype=float)[0] < 0.01
    metadata = model.get_metadata()
    json.dumps(metadata)
    assert metadata["feature_spec"]["numeric_features"] == ("ses",)  # type: ignore[index]
    assert metadata["preprocessing"]["fit_row_count"] == len(modeling_df)  # type: ignore[index]


def test_model_evaluation_delegates_without_feature_matrices(
    modeling_df: pd.DataFrame,
    runtime_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model boundary should pass outcomes and predictions to evaluation."""
    from age_group_prediction.models import base

    delegated: dict[str, object] = {}
    shared_evaluator = base.evaluate_predictions

    def recording_evaluator(
        observed: pd.DataFrame,
        prediction: PredictionResult,
        metrics: list[MeanAbsoluteError],
        *,
        rng: np.random.Generator,
    ) -> EvaluationResult:
        delegated.update(observed=observed, prediction=prediction, metrics=metrics)
        return shared_evaluator(observed, prediction, metrics, rng=rng)

    monkeypatch.setattr(base, "evaluate_predictions", recording_evaluator)
    model = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )
    metrics = [MeanAbsoluteError("n_children_total")]

    model.evaluate(modeling_df, metrics=metrics)

    assert delegated["observed"] is modeling_df
    assert isinstance(delegated["prediction"], PredictionResult)
    assert delegated["metrics"] is metrics


def test_default_and_caller_rng_behavior_is_reproducible(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    default_a = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )
    default_b = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )
    result_a = default_a.predict(modeling_df)
    result_b = default_b.predict(modeling_df)
    np.testing.assert_array_equal(result_a.total_mean, result_b.total_mean)

    rng_a = np.random.default_rng(73)
    rng_b = np.random.default_rng(73)
    caller_a = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC, rng=rng_a
    )
    caller_b = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC, rng=rng_b
    )
    np.testing.assert_array_equal(
        caller_a.predict(modeling_df, rng=rng_a).total_mean,
        caller_b.predict(modeling_df, rng=rng_b).total_mean,
    )
    assert rng_a.random() == rng_b.random()


def test_recorded_child_seed_replays_model_rng(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    model = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df,
        feature_spec=FEATURE_SPEC,
        rng=np.random.default_rng(19),
    )
    result = model.predict(modeling_df, rng=np.random.default_rng(23))
    predict_seed = model.get_metadata()["derived_seeds"][-1]["seed"]  # type: ignore[index]
    replayed_jitter = np.random.default_rng(predict_seed).uniform(
        0.0, 0.01, len(modeling_df)
    )

    np.testing.assert_allclose(
        result.total_mean,
        modeling_df["n_children_total"].to_numpy() + replayed_jitter,
    )


class JointScopeSyntheticModel(SyntheticModel):
    """Return sequential-joint scores built under the default validation config."""

    def _predict_model(
        self,
        *,
        eval_df: pd.DataFrame,
        features: pd.DataFrame,
        log_exposure: pd.Series | None,
        prediction_config: PredictionConfig,
        rng: np.random.Generator,
    ) -> PredictionResult:
        del features, log_exposure, prediction_config, rng
        total_mean = eval_df["n_children_total"].to_numpy(dtype=float)
        return PredictionResult.from_means(
            building_ids=eval_df["building_id"].to_numpy(),
            cohort_names=COHORT_NAMES,
            total_mean=total_mean,
            cohort_means=total_mean[:, None] * np.array([0.2, 0.3, 0.5]),
            pointwise_log_probabilities={
                key: np.full(len(eval_df), -1.0) for key in ("total", *COHORT_NAMES)
            },
            pointwise_log_probability_scope="sequential_joint",
        )


def test_validation_config_rebuild_preserves_pointwise_scope(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    """Re-validating under the model's config must not drop the joint scope."""
    from age_group_prediction import PredictionValidationConfig

    config = replace(
        runtime_config,
        prediction_validation=replace(
            runtime_config.prediction_validation, reconciliation_tolerance=1e-8
        ),
    )
    assert config.prediction_validation != PredictionValidationConfig()
    model = JointScopeSyntheticModel(runtime_config=config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )

    result = model.predict(modeling_df)

    # The hook returned the default config, so predict() must have rebuilt it.
    assert result.validation_config == config.prediction_validation
    assert result.pointwise_log_probability_scope == "sequential_joint"

@pytest.mark.parametrize("family", ["gamma", "NORMAL", "student_t", ""])
def test_unknown_predictive_family_is_rejected_not_reclassified(family: str) -> None:
    """An unrecognized family must raise, never silently become Normal.

    Left unchecked, these strings fell through the dispatch chain and were
    scored with a log density instead of a log mass.
    """
    with pytest.raises(ValueError, match="family must be one of"):
        ParametricDistributionSpec(family=family, scale=1.0)


def test_training_hashes_identify_the_rows_a_model_was_fitted_on(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    model = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )
    metadata = model.get_metadata()

    assert metadata["training_data_hash"]
    assert metadata["training_schema_hash"]

    # Same rows in a different order are the same training data.
    shuffled = modeling_df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    reordered = SyntheticModel(runtime_config=runtime_config).fit(
        shuffled, feature_spec=FEATURE_SPEC
    )
    assert (
        reordered.get_metadata()["training_data_hash"]
        == metadata["training_data_hash"]
    )

    # Different rows must not. Without this the test would pass on a constant.
    changed = modeling_df.copy()
    changed.loc[changed.index[0], "ses"] += 0.5
    other = SyntheticModel(runtime_config=runtime_config).fit(
        changed, feature_spec=FEATURE_SPEC
    )
    assert other.get_metadata()["training_data_hash"] != metadata["training_data_hash"]

    retyped = modeling_df.astype({"n_apartments": float})
    retyped_model = SyntheticModel(runtime_config=runtime_config).fit(
        retyped, feature_spec=FEATURE_SPEC
    )
    assert (
        retyped_model.get_metadata()["training_schema_hash"]
        != metadata["training_schema_hash"]
    )


def test_training_hashes_are_cleared_when_a_fit_fails(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    model = SyntheticModel(runtime_config=runtime_config)
    with pytest.raises(ValueError, match="synthetic fit failure"):
        model.fit(modeling_df.assign(fail_fit=True), feature_spec=FEATURE_SPEC)

    metadata = model.get_metadata()
    assert metadata["training_data_hash"] is None
    assert metadata["training_schema_hash"] is None


def test_repeated_predict_does_not_accumulate_duplicate_seed_records(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    """Re-running an operation replaces its seed records rather than appending."""
    model = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )
    fit_seeds = [
        record
        for record in model.get_metadata()["derived_seeds"]
        if record["operation"] == "fit"
    ]

    counts = []
    for _ in range(4):
        model.predict(modeling_df)
        counts.append(len(model.get_metadata()["derived_seeds"]))

    assert len(set(counts)) == 1, f"seed records grew across predict calls: {counts}"
    # A later predict must not erase what fit recorded.
    assert [
        record
        for record in model.get_metadata()["derived_seeds"]
        if record["operation"] == "fit"
    ] == fit_seeds


def test_metadata_covers_every_required_contract_field(
    modeling_df: pd.DataFrame, runtime_config: ExperimentConfig
) -> None:
    """Plan section 4.3 enumerates what get_metadata must carry."""
    model = SyntheticModel(runtime_config=runtime_config).fit(
        modeling_df, feature_spec=FEATURE_SPEC
    )
    metadata = model.get_metadata()
    flattened = {**metadata, **metadata["model"]}

    required = (
        "implementation_version",
        "likelihood",
        "parameterization",
        "feature_spec",
        "preprocessing",
        "hyperparameters",
        "priors",
        "calibration",
        "derived_seeds",
        "dependency_versions",
        "training_data_hash",
        "training_schema_hash",
        "fit_duration_seconds",
        "uncertainty_method",
        "diagnostics",
    )
    missing = [name for name in required if name not in flattened]
    assert not missing, f"missing required metadata fields: {missing}"
    json.dumps(metadata)
