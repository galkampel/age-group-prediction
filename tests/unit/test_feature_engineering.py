"""Tests for immutable feature specs and fold-fitted transforms."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from age_group_prediction import (
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_TOTAL_FEATURE_SPEC,
    DEFAULT_TREE_FEATURE_SPEC,
    CategoricalFeatureSpec,
    FeatureSpec,
    FittedFeatureTransformer,
    ModelingSchema,
    build_modeling_table,
)
from student_simulator import StudentPopulationSimulator, load_simulation_config


@pytest.fixture(scope="module")
def modeling_df() -> pd.DataFrame:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"
    config = load_simulation_config(config_path)
    return build_modeling_table(StudentPopulationSimulator(config).run())


def test_named_defaults_define_distinct_component_policies() -> None:
    assert DEFAULT_TREE_FEATURE_SPEC.component == "tree"
    assert not DEFAULT_TREE_FEATURE_SPEC.scale_numeric
    assert DEFAULT_MODELING_SCHEMA.exposure_column in (
        DEFAULT_TREE_FEATURE_SPEC.numeric_features
    )

    assert DEFAULT_TOTAL_FEATURE_SPEC.component == "total_count"
    assert DEFAULT_TOTAL_FEATURE_SPEC.scale_numeric
    assert (
        DEFAULT_TOTAL_FEATURE_SPEC.exposure_column
        == DEFAULT_MODELING_SCHEMA.exposure_column
    )
    assert (
        DEFAULT_MODELING_SCHEMA.exposure_column
        not in DEFAULT_TOTAL_FEATURE_SPEC.numeric_features
    )

    assert DEFAULT_PROBABILITY_FEATURE_SPEC.component == "age_probability"
    assert DEFAULT_PROBABILITY_FEATURE_SPEC.exposure_column is None
    assert (
        DEFAULT_MODELING_SCHEMA.exposure_column
        not in DEFAULT_PROBABILITY_FEATURE_SPEC.numeric_features
    )


def test_transform_requires_fit(modeling_df: pd.DataFrame) -> None:
    transformer = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC)

    with pytest.raises(RuntimeError, match="fitted"):
        transformer.transform(modeling_df)
    with pytest.raises(RuntimeError, match="fitted"):
        transformer.get_feature_names_out()
    with pytest.raises(RuntimeError, match="fitted"):
        transformer.get_metadata()


def test_default_total_transform_is_numeric_stable_and_target_free(
    modeling_df: pd.DataFrame,
) -> None:
    fit_df = modeling_df.iloc[:150]
    validation_df = modeling_df.iloc[150:]
    transformer = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC)

    fit_features = transformer.fit_transform(fit_df)
    validation_features = transformer.transform(validation_df)

    assert tuple(fit_features.columns) == transformer.get_feature_names_out()
    assert tuple(validation_features.columns) == transformer.get_feature_names_out()
    assert np.isfinite(fit_features.to_numpy()).all()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in fit_features.dtypes)
    assert set(DEFAULT_MODELING_SCHEMA.identifier_columns).isdisjoint(fit_features)
    assert set(DEFAULT_MODELING_SCHEMA.target_columns).isdisjoint(fit_features)
    assert DEFAULT_MODELING_SCHEMA.exposure_column not in fit_features
    assert np.allclose(
        fit_features.loc[:, DEFAULT_TOTAL_FEATURE_SPEC.numeric_features].mean(),
        0.0,
        atol=1e-12,
    )


def test_transforming_validation_data_does_not_change_fitted_state(
    modeling_df: pd.DataFrame,
) -> None:
    fit_df = modeling_df.iloc[:150]
    validation_df = modeling_df.iloc[150:].copy()
    transformer = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC).fit(fit_df)
    metadata_before = transformer.get_metadata()
    fit_features_before = transformer.transform(fit_df)

    validation_df.loc[:, "ses"] = validation_df["ses"] + 1000
    transformer.transform(validation_df)

    assert transformer.get_metadata() == metadata_before
    pd.testing.assert_frame_equal(transformer.transform(fit_df), fit_features_before)


def test_fitted_preprocessing_is_row_order_invariant_and_excludes_targets(
    modeling_df: pd.DataFrame,
) -> None:
    changed_targets = modeling_df.copy()
    changed_targets.loc[:, "n_kindergarten"] += 1
    # The total spec scales, so this exercises learned means and scales too.
    transformer_a = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC).fit(modeling_df)
    transformer_b = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC).fit(
        changed_targets.sample(frac=1.0, random_state=4)
    )

    metadata_a = transformer_a.get_metadata()
    metadata_b = transformer_b.get_metadata()
    assert metadata_a["feature_spec"] == metadata_b["feature_spec"]
    assert metadata_a["feature_names"] == metadata_b["feature_names"]
    assert metadata_a["fit_row_count"] == metadata_b["fit_row_count"]
    assert metadata_a["category_references"] == metadata_b["category_references"]
    np.testing.assert_allclose(
        list(metadata_a["numeric_means"].values()),
        list(metadata_b["numeric_means"].values()),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        list(metadata_a["numeric_scales"].values()),
        list(metadata_b["numeric_scales"].values()),
        rtol=1e-14,
    )
    assert (
        transformer_a.get_feature_names_out()
        == transformer_b.get_feature_names_out()
    )


def test_categorical_reference_and_unknown_policy_are_schema_driven(
    modeling_df: pd.DataFrame,
) -> None:
    transformer = FittedFeatureTransformer(DEFAULT_TREE_FEATURE_SPEC).fit(modeling_df)
    features = transformer.transform(modeling_df)

    assert "school_status_none" not in features
    assert "school_status_existing" in features
    assert "school_status_planned" in features

    unknown_df = modeling_df.copy()
    unknown_df.loc[unknown_df.index[0], "school_status"] = "unknown"
    with pytest.raises(ValueError, match="Unknown school_status"):
        transformer.transform(unknown_df)

    # "treat_as_reference" is named for what the encoding actually does: with
    # the reference level dropped, an unseen level is indistinguishable from it.
    reference_spec = replace(
        DEFAULT_TREE_FEATURE_SPEC, unknown_category_policy="treat_as_reference"
    )
    reference_transformer = FittedFeatureTransformer(reference_spec).fit(modeling_df)
    with pytest.warns(UserWarning, match="unknown categories"):
        encoded = reference_transformer.transform(unknown_df)
    encoded_columns = ["school_status_existing", "school_status_planned"]
    assert encoded.loc[unknown_df.index[0], encoded_columns].sum() == 0

    reference_rows = modeling_df[modeling_df["school_status"] == "none"]
    reference_encoded = reference_transformer.transform(reference_rows)
    np.testing.assert_allclose(
        encoded.loc[unknown_df.index[0], encoded_columns].to_numpy(dtype=float),
        reference_encoded.iloc[0][encoded_columns].to_numpy(dtype=float),
    )


def test_custom_schema_controls_semantic_columns_and_category_reference() -> None:
    schema = ModelingSchema(
        raw_numeric_features=("income", "household", "age", "services", "units"),
        derived_numeric_features=("small_mix",),
        categorical_feature_specs=(
            CategoricalFeatureSpec(
                column="zone",
                categories=("rural", "urban"),
                reference_category="rural",
            ),
        ),
        exposure_column="units",
        room_count_columns=("large", "small"),
        room_reference_column="large",
    )
    spec = FeatureSpec(
        component="total_count",
        numeric_features=("income", "household", "age", "services", "small_mix"),
        categorical_features=("zone",),
        interactions=("ses_x_household_size",),
        scale_numeric=True,
        exposure_column="units",
        ses_column="income",
        household_size_column="household",
        median_age_column="age",
        daycare_column="services",
    )
    feature_df = pd.DataFrame(
        {
            "income": [1.0, 2.0, 3.0],
            "household": [2.0, 2.5, 3.0],
            "age": [30.0, 40.0, 50.0],
            "services": [0.0, 1.0, 2.0],
            "units": [10, 20, 30],
            "small_mix": [0.2, 0.4, 0.6],
            "zone": ["rural", "urban", "rural"],
        }
    )

    transformed = FittedFeatureTransformer(spec, schema=schema).fit_transform(
        feature_df
    )

    assert "zone_rural" not in transformed
    assert "zone_urban" in transformed
    assert "income_x_household" in transformed
    assert "log_units" not in transformed


def test_absent_fit_category_still_has_stable_output_column(
    modeling_df: pd.DataFrame,
) -> None:
    fit_df = modeling_df.loc[modeling_df["school_status"] != "planned"]
    transformer = FittedFeatureTransformer(DEFAULT_TREE_FEATURE_SPEC).fit(fit_df)

    assert "school_status_planned" in transformer.get_feature_names_out()
    assert tuple(transformer.transform(modeling_df).columns) == (
        transformer.get_feature_names_out()
    )


def test_log_exposure_is_separate_and_positive(modeling_df: pd.DataFrame) -> None:
    transformer = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC).fit(modeling_df)
    raw_exposure = modeling_df[DEFAULT_MODELING_SCHEMA.exposure_column].copy()
    log_exposure = transformer.get_log_exposure(modeling_df)

    assert log_exposure is not None
    np.testing.assert_allclose(
        log_exposure,
        np.log(modeling_df[DEFAULT_MODELING_SCHEMA.exposure_column]),
    )
    pd.testing.assert_series_equal(
        modeling_df[DEFAULT_MODELING_SCHEMA.exposure_column],
        raw_exposure,
        check_names=True,
    )
    invalid_df = modeling_df.copy()
    invalid_df.loc[invalid_df.index[0], "n_apartments"] = 0
    with pytest.raises(ValueError, match="finite and positive"):
        transformer.transform(invalid_df)


def test_component_specific_interactions_fail_during_spec_construction() -> None:
    with pytest.raises(ValueError, match="total-count only"):
        replace(
            DEFAULT_PROBABILITY_FEATURE_SPEC,
            interactions=("room_share_x_household_size",),
        )
    with pytest.raises(ValueError, match="age-probability only"):
        replace(
            DEFAULT_TOTAL_FEATURE_SPEC,
            interactions=("room_share_x_median_age",),
        )


def test_interactions_require_selected_main_effects() -> None:
    spec = FeatureSpec(
        component="tree",
        numeric_features=("ses",),
        categorical_features=(),
        interactions=("ses_x_household_size",),
    )

    with pytest.raises(ValueError, match="main effects"):
        spec.validate_for_schema(DEFAULT_MODELING_SCHEMA)


def test_raw_room_counts_cannot_be_mixed_with_room_shares() -> None:
    spec = replace(
        DEFAULT_TREE_FEATURE_SPEC,
        numeric_features=(
            *DEFAULT_TREE_FEATURE_SPEC.numeric_features,
            "3_rooms",
        ),
    )

    with pytest.raises(ValueError, match="redundant"):
        spec.validate_for_schema(DEFAULT_MODELING_SCHEMA)


@pytest.mark.parametrize(
    "spec",
    [
        replace(DEFAULT_TOTAL_FEATURE_SPEC, ses_form="quadratic"),
        replace(DEFAULT_TOTAL_FEATURE_SPEC, ses_form="spline"),
        replace(
            DEFAULT_TOTAL_FEATURE_SPEC,
            interactions=(
                "ses_x_household_size",
                "daycare_x_median_age",
                "room_share_x_household_size",
            ),
        ),
        replace(DEFAULT_TOTAL_FEATURE_SPEC, daycare_form="log1p"),
        replace(
            DEFAULT_PROBABILITY_FEATURE_SPEC,
            interactions=("room_share_x_median_age",),
        ),
    ],
)
def test_candidate_transforms_are_finite_and_have_stable_names(
    modeling_df: pd.DataFrame, spec: FeatureSpec
) -> None:
    transformer = FittedFeatureTransformer(spec)
    transformed = transformer.fit_transform(modeling_df.iloc[:150])
    validation = transformer.transform(modeling_df.iloc[150:])

    assert tuple(transformed.columns) == tuple(validation.columns)
    assert np.isfinite(transformed.to_numpy()).all()


def test_probability_preprocessing_can_fit_before_zero_total_filtering(
    modeling_df: pd.DataFrame,
) -> None:
    fit_df = modeling_df.copy()
    fit_df.loc[fit_df.index[:5], list(DEFAULT_MODELING_SCHEMA.target_columns)] = 0
    transformer = FittedFeatureTransformer(DEFAULT_PROBABILITY_FEATURE_SPEC).fit(fit_df)

    assert transformer.get_metadata()["fit_row_count"] == len(fit_df)
    assert len(transformer.transform(fit_df)) == len(fit_df)


def test_spec_and_fitted_metadata_are_json_serializable(
    modeling_df: pd.DataFrame,
) -> None:
    spec = replace(DEFAULT_TOTAL_FEATURE_SPEC, ses_form="spline")
    transformer = FittedFeatureTransformer(spec).fit(modeling_df)

    json.dumps(transformer.get_metadata())
    assert transformer.get_metadata()["spline_knots"]

def test_unscaled_specs_report_no_scaling_state(modeling_df: pd.DataFrame) -> None:
    """Metadata must not advertise means and scales it never applied."""
    metadata = FittedFeatureTransformer(DEFAULT_TREE_FEATURE_SPEC).fit(
        modeling_df
    ).get_metadata()

    assert metadata["scale_numeric"] is False
    assert metadata["numeric_means"] is None
    assert metadata["numeric_scales"] is None
    np.testing.assert_allclose(
        FittedFeatureTransformer(DEFAULT_TREE_FEATURE_SPEC)
        .fit_transform(modeling_df)["ses"]
        .to_numpy(),
        modeling_df["ses"].astype(float).to_numpy(),
    )


def test_interactions_use_the_transformed_main_effect(
    modeling_df: pd.DataFrame,
) -> None:
    """An interaction must multiply the representation the matrix carries."""
    spec = FeatureSpec(
        component="total_count",
        numeric_features=("ses", "avg_household_size", "median_age", "n_daycares_500m"),
        categorical_features=(),
        daycare_form="log1p",
        interactions=("daycare_x_median_age",),
        exposure_column="n_apartments",
    )
    transformed = FittedFeatureTransformer(spec).fit_transform(modeling_df)

    raw = modeling_df["n_daycares_500m"].astype(float).to_numpy()
    age = modeling_df["median_age"].astype(float).to_numpy()
    interaction = transformed["n_daycares_500m_x_median_age"].to_numpy()

    np.testing.assert_allclose(interaction, np.log1p(raw) * age)
    assert not np.allclose(interaction, raw * age)
    # The linear main effect it would otherwise imply is absent by design.
    assert "n_daycares_500m" not in transformed.columns
    assert "n_daycares_500m_log1p" in transformed.columns


def test_spline_ses_cannot_carry_a_linear_ses_interaction() -> None:
    """A spline replaces the linear column, so the linear interaction is illegal."""
    with pytest.raises(ValueError, match="Spline SES cannot be combined"):
        FeatureSpec(
            component="total_count",
            numeric_features=("ses", "avg_household_size"),
            categorical_features=(),
            ses_form="spline",
            interactions=("ses_x_household_size",),
            exposure_column="n_apartments",
        )


def test_fit_transform_matches_transform_without_redoing_the_work(
    modeling_df: pd.DataFrame,
) -> None:
    transformer = FittedFeatureTransformer(DEFAULT_TOTAL_FEATURE_SPEC)
    fit_transformed = transformer.fit_transform(modeling_df)

    pd.testing.assert_frame_equal(fit_transformed, transformer.transform(modeling_df))
    # The returned frame is a copy, so mutating it cannot corrupt fitted state.
    fit_transformed.iloc[0, 0] = 12345.0
    assert transformer.transform(modeling_df).iloc[0, 0] != 12345.0


def test_missing_category_value_is_reported_as_missing(
    modeling_df: pd.DataFrame,
) -> None:
    spec = FeatureSpec(
        component="age_probability",
        numeric_features=("ses",),
        categorical_features=("school_status",),
    )
    missing_df = modeling_df.copy()
    missing_df["school_status"] = missing_df["school_status"].astype(object)
    missing_df.loc[missing_df.index[0], "school_status"] = None

    with pytest.raises(ValueError, match="school_status must not be missing"):
        FittedFeatureTransformer(spec).fit(missing_df)
