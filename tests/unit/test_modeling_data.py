"""Tests for modeling configuration and known-neighborhood splitting.

These tests prove no target, target-derived column, ID, or latent effect
enters the modeling features; that room shares are non-collinear; and that
outer and validation splits are deterministic, replayable, and isolated.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from age_group_prediction import (
    CATEGORICAL_FEATURES,
    COHORT_TARGET_COLUMNS,
    DEFAULT_MODELING_SCHEMA,
    DEFAULT_SEED,
    IDENTIFIER_COLUMNS,
    MODELING_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TOTAL_TARGET_COLUMN,
    CategoricalFeatureSpec,
    FoldConfig,
    ModelingSchema,
    OuterSplitConfig,
    SplitManifest,
    build_modeling_table,
    load_split_manifest,
    make_validation_folds,
    persist_split_manifest,
    replay_split_manifest,
    split_known_neighborhood_buildings,
)
from student_simulator import StudentPopulationSimulator, load_simulation_config


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"


@pytest.fixture(scope="module")
def source_df() -> pd.DataFrame:
    config = load_simulation_config(_config_path())
    return StudentPopulationSimulator(config).run()


@pytest.fixture(scope="module")
def modeling_df(source_df: pd.DataFrame) -> pd.DataFrame:
    return build_modeling_table(source_df)


# --- Feature contract -------------------------------------------------


def test_modeling_feature_columns_are_the_nine_agreed_features() -> None:
    assert NUMERIC_FEATURES == (
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "n_apartments",
        "3_rooms_share",
        "4_rooms_share",
        "5_rooms_share",
    )
    assert CATEGORICAL_FEATURES == ("school_status",)
    assert len(MODELING_FEATURE_COLUMNS) == 9


def test_modeling_schema_rejects_target_as_feature() -> None:
    schema = ModelingSchema(
        raw_numeric_features=(
            *DEFAULT_MODELING_SCHEMA.raw_numeric_features,
            TOTAL_TARGET_COLUMN,
        )
    )
    with pytest.raises(ValueError, match="Forbidden modeling features"):
        schema.validate_definition()


def test_custom_modeling_schema_controls_all_column_semantics() -> None:
    schema = ModelingSchema(
        building_id_column="building_key",
        neighborhood_id_column="area_key",
        raw_numeric_features=("signal", "units"),
        derived_numeric_features=("small_mix",),
        categorical_feature_specs=(
            CategoricalFeatureSpec(
                column="status",
                categories=("baseline", "active"),
                reference_category="baseline",
            ),
        ),
        cohort_target_columns=("younger", "older"),
        total_target_column="children",
        exposure_column="units",
        room_count_columns=("large_units", "small_units"),
        room_reference_column="large_units",
        forbidden_feature_columns=("private_effect",),
        schema_version="custom-1",
    )
    source_df = pd.DataFrame(
        {
            "building_key": ["b1", "b2"],
            "area_key": ["a1", "a1"],
            "signal": [0.2, 0.7],
            "units": [10, 20],
            "status": ["baseline", "active"],
            "large_units": [4, 5],
            "small_units": [6, 15],
            "younger": [2, 3],
            "older": [1, 4],
            "children": [3, 7],
        }
    )

    modeling_df = build_modeling_table(source_df, schema=schema)

    assert schema.identifier_columns == ("building_key", "area_key")
    assert schema.categorical_features == ("status",)
    assert schema.room_share_features == ("small_mix",)
    assert schema.room_share_columns == (("small_units", "small_mix"),)
    assert tuple(modeling_df.columns) == schema.table_columns
    assert "large_units_share" not in modeling_df
    np.testing.assert_allclose(modeling_df["small_mix"], [0.6, 0.75])


def test_modeling_table_excludes_ids_targets_and_latent_effects(
    modeling_df: pd.DataFrame,
) -> None:
    forbidden_columns = {
        *IDENTIFIER_COLUMNS,
        *COHORT_TARGET_COLUMNS,
        TOTAL_TARGET_COLUMN,
        "u_b",
        "w_j",
        "v_b",
    }
    feature_columns = set(MODELING_FEATURE_COLUMNS)
    assert feature_columns.isdisjoint(forbidden_columns)
    # The modeling table still carries IDs and targets for splitting/evaluation,
    # but only as their own dedicated columns, never inside the feature list.
    assert set(modeling_df.columns) == {
        *IDENTIFIER_COLUMNS,
        *MODELING_FEATURE_COLUMNS,
        *COHORT_TARGET_COLUMNS,
        TOTAL_TARGET_COLUMN,
    }


def test_room_shares_omit_the_six_room_reference_category(
    source_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
) -> None:
    assert "6_rooms_share" not in modeling_df.columns
    for room_column in ["3_rooms", "4_rooms", "5_rooms"]:
        assert f"{room_column}_share" in modeling_df.columns

    expected_share = source_df["3_rooms"] / source_df["n_apartments"]
    pd.testing.assert_series_equal(
        modeling_df["3_rooms_share"], expected_share.rename("3_rooms_share")
    )


def test_build_modeling_table_rejects_missing_source_columns(
    source_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Missing source columns"):
        build_modeling_table(source_df.drop(columns=["ses"]))

    with pytest.raises(ValueError, match="Missing source columns"):
        build_modeling_table(source_df.drop(columns=["4_rooms"]))


def test_modeling_table_rejects_invalid_accounting_identity(
    modeling_df: pd.DataFrame,
) -> None:
    invalid_df = modeling_df.copy()
    invalid_df.loc[invalid_df.index[0], TOTAL_TARGET_COLUMN] += 1
    with pytest.raises(ValueError, match="must equal"):
        DEFAULT_MODELING_SCHEMA.validate_table(invalid_df)


def test_modeling_table_rejects_noninteger_targets(
    modeling_df: pd.DataFrame,
) -> None:
    invalid_df = modeling_df.copy()
    invalid_df[[COHORT_TARGET_COLUMNS[0], TOTAL_TARGET_COLUMN]] = invalid_df[
        [COHORT_TARGET_COLUMNS[0], TOTAL_TARGET_COLUMN]
    ].astype(float)
    invalid_df.loc[invalid_df.index[0], COHORT_TARGET_COLUMNS[0]] += 0.5
    invalid_df.loc[invalid_df.index[0], TOTAL_TARGET_COLUMN] += 0.5
    with pytest.raises(ValueError, match="nonnegative integers"):
        DEFAULT_MODELING_SCHEMA.validate_table(invalid_df)


def test_modeling_table_accepts_integer_valued_float_targets(
    modeling_df: pd.DataFrame,
) -> None:
    float_target_df = modeling_df.copy()
    float_target_df[list(DEFAULT_MODELING_SCHEMA.target_columns)] = float_target_df[
        list(DEFAULT_MODELING_SCHEMA.target_columns)
    ].astype(float)

    DEFAULT_MODELING_SCHEMA.validate_table(float_target_df)


def test_modeling_table_rejects_unknown_school_status(
    modeling_df: pd.DataFrame,
) -> None:
    invalid_df = modeling_df.copy()
    invalid_df.loc[invalid_df.index[0], "school_status"] = "unknown"
    with pytest.raises(ValueError, match="Unknown school_status"):
        DEFAULT_MODELING_SCHEMA.validate_table(invalid_df)


def test_modeling_table_rejects_nonpositive_exposure(
    modeling_df: pd.DataFrame,
) -> None:
    invalid_df = modeling_df.copy()
    invalid_df.loc[invalid_df.index[0], "n_apartments"] = 0
    with pytest.raises(ValueError, match="n_apartments must be positive"):
        DEFAULT_MODELING_SCHEMA.validate_table(invalid_df)


# --- Known-neighborhood split -------------------------------------------------


def test_split_is_deterministic_and_invariant_to_row_order(
    modeling_df: pd.DataFrame,
) -> None:
    shuffled_df = modeling_df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    split_a = split_known_neighborhood_buildings(
        modeling_df, rng=np.random.default_rng(DEFAULT_SEED)
    )
    split_b = split_known_neighborhood_buildings(
        shuffled_df, rng=np.random.default_rng(DEFAULT_SEED)
    )

    assert set(split_a.train_df["building_id"]) == set(
        split_b.train_df["building_id"]
    )
    assert set(split_a.test_df["building_id"]) == set(split_b.test_df["building_id"])
    assert split_a.manifest == split_b.manifest


def test_split_default_rng_is_reproducible(modeling_df: pd.DataFrame) -> None:
    split_a = split_known_neighborhood_buildings(modeling_df)
    split_b = split_known_neighborhood_buildings(modeling_df)

    assert split_a.manifest == split_b.manifest


def test_split_partitions_are_disjoint_and_cover_every_building(
    modeling_df: pd.DataFrame,
) -> None:
    split = split_known_neighborhood_buildings(modeling_df)

    train_ids = set(split.train_df["building_id"])
    test_ids = set(split.test_df["building_id"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(modeling_df["building_id"])


def test_non_singleton_neighborhoods_appear_in_both_partitions(
    modeling_df: pd.DataFrame,
) -> None:
    split = split_known_neighborhood_buildings(modeling_df)

    neighborhood_sizes = modeling_df.groupby("neighborhood_id").size()
    non_singleton_neighborhoods = neighborhood_sizes[neighborhood_sizes >= 2].index

    train_neighborhoods = set(split.train_df["neighborhood_id"])
    test_neighborhoods = set(split.test_df["neighborhood_id"])
    for neighborhood_id in non_singleton_neighborhoods:
        assert neighborhood_id in train_neighborhoods
        assert neighborhood_id in test_neighborhoods


def test_singleton_neighborhoods_remain_entirely_in_training(
    modeling_df: pd.DataFrame,
) -> None:
    split = split_known_neighborhood_buildings(modeling_df)

    neighborhood_sizes = modeling_df.groupby("neighborhood_id").size()
    singleton_neighborhoods = set(neighborhood_sizes[neighborhood_sizes == 1].index)

    assert singleton_neighborhoods.isdisjoint(
        set(split.test_df["neighborhood_id"])
    )
    assert singleton_neighborhoods <= set(split.train_df["neighborhood_id"])


@pytest.mark.parametrize("invalid_fraction", [0.0, 1.0, -0.1, 1.1])
def test_split_rejects_invalid_test_fraction(
    modeling_df: pd.DataFrame, invalid_fraction: float
) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        OuterSplitConfig(test_fraction=invalid_fraction)


def test_split_rejects_duplicate_building_ids(modeling_df: pd.DataFrame) -> None:
    duplicated_df = pd.concat([modeling_df, modeling_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        split_known_neighborhood_buildings(duplicated_df)


def test_split_rejects_missing_required_columns(modeling_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing split columns"):
        split_known_neighborhood_buildings(
            modeling_df.drop(columns=["neighborhood_id"]),
        )


def test_split_manifest_is_json_serializable_and_replayable(
    modeling_df: pd.DataFrame,
) -> None:
    split = split_known_neighborhood_buildings(
        modeling_df, rng=np.random.default_rng(123)
    )
    json.dumps(split.manifest.to_dict())

    shuffled_df = modeling_df.sample(frac=1.0, random_state=91).reset_index(drop=True)
    replayed = replay_split_manifest(shuffled_df, split.manifest)

    assert set(replayed.train_df["building_id"]) == set(split.train_df["building_id"])
    assert set(replayed.test_df["building_id"]) == set(split.test_df["building_id"])


def test_manifest_records_requested_and_realized_holdout_fractions(
    modeling_df: pd.DataFrame,
) -> None:
    """Small neighborhoods hold out more than asked for; both numbers are kept."""
    config = OuterSplitConfig(test_fraction=0.1)
    split = split_known_neighborhood_buildings(modeling_df, config=config)
    manifest = split.manifest

    assert manifest.requested_holdout_fraction == 0.1
    assert manifest.realized_holdout_fraction == pytest.approx(
        len(split.test_df) / len(modeling_df)
    )
    # The floor of one holdout per non-singleton neighborhood inflates the share.
    assert manifest.realized_holdout_fraction > manifest.requested_holdout_fraction


def test_manifest_records_how_randomness_was_obtained(
    modeling_df: pd.DataFrame,
) -> None:
    """A lockbox should say whether its seed was chosen or fell back."""
    defaulted = split_known_neighborhood_buildings(modeling_df)
    supplied = split_known_neighborhood_buildings(
        modeling_df, rng=np.random.default_rng(DEFAULT_SEED)
    )

    assert defaulted.manifest.seed_source == "project_default"
    assert supplied.manifest.seed_source == "caller_generator"
    # Same randomness, different provenance: the assignments still agree.
    assert (
        defaulted.manifest.holdout_building_ids
        == supplied.manifest.holdout_building_ids
    )


def test_manifest_rejects_an_unknown_seed_source(
    modeling_df: pd.DataFrame,
) -> None:
    manifest = split_known_neighborhood_buildings(modeling_df).manifest
    with pytest.raises(ValueError, match="seed_source must be one of"):
        replace(manifest, seed_source="guessed")


def test_split_manifest_round_trips_through_json(
    modeling_df: pd.DataFrame,
) -> None:
    split = split_known_neighborhood_buildings(modeling_df)
    restored = SplitManifest.from_dict(json.loads(json.dumps(split.manifest.to_dict())))

    assert restored == split.manifest
    replayed = replay_split_manifest(modeling_df, restored)
    assert set(replayed.test_df["building_id"]) == set(split.test_df["building_id"])


def test_split_manifest_rejects_mismatched_provenance(
    modeling_df: pd.DataFrame,
) -> None:
    """Hashes prove the rows; these fields must prove the manifest describes them."""
    manifest = split_known_neighborhood_buildings(modeling_df).manifest

    with pytest.raises(ValueError, match="rows"):
        replay_split_manifest(modeling_df, replace(manifest, n_rows=999_999))
    with pytest.raises(ValueError, match="neighborhoods"):
        replay_split_manifest(modeling_df, replace(manifest, n_neighborhoods=1))
    with pytest.raises(ValueError, match="strategy"):
        replay_split_manifest(modeling_df, replace(manifest, strategy="other-v9"))
    with pytest.raises(ValueError, match="realized holdout fraction"):
        replay_split_manifest(
            modeling_df, replace(manifest, realized_holdout_fraction=0.99)
        )


def test_split_manifest_rejects_swapped_partitions(
    modeling_df: pd.DataFrame,
) -> None:
    """A swapped manifest is still a partition, so only the fraction catches it."""
    manifest = split_known_neighborhood_buildings(modeling_df).manifest
    swapped = replace(
        manifest,
        training_building_ids=manifest.holdout_building_ids,
        holdout_building_ids=manifest.training_building_ids,
    )

    with pytest.raises(ValueError, match="realized holdout fraction"):
        replay_split_manifest(modeling_df, swapped)


def test_split_manifest_rejects_changed_source_data(
    modeling_df: pd.DataFrame,
) -> None:
    split = split_known_neighborhood_buildings(modeling_df)
    changed_df = modeling_df.copy()
    changed_df.loc[changed_df.index[0], "ses"] += 0.01

    with pytest.raises(ValueError, match="does not match"):
        replay_split_manifest(changed_df, split.manifest)


# --- Lockbox manifest persistence (Gate 8) -----------------------------


def test_persist_split_manifest_writes_a_reloadable_json_file(
    modeling_df: pd.DataFrame, tmp_path: Path
) -> None:
    manifest = split_known_neighborhood_buildings(modeling_df).manifest
    target = tmp_path / "lockbox" / "split_manifest.json"

    written = persist_split_manifest(target, manifest)

    assert written == manifest
    assert target.exists()
    assert load_split_manifest(target) == manifest


def test_persist_split_manifest_is_idempotent_for_an_exact_match(
    modeling_df: pd.DataFrame, tmp_path: Path
) -> None:
    manifest = split_known_neighborhood_buildings(modeling_df).manifest
    target = tmp_path / "split_manifest.json"

    first = persist_split_manifest(target, manifest)
    second = persist_split_manifest(target, manifest)

    assert first == second == manifest


def test_persist_split_manifest_refuses_a_conflicting_manifest(
    modeling_df: pd.DataFrame, tmp_path: Path
) -> None:
    manifest = split_known_neighborhood_buildings(modeling_df).manifest
    conflicting = replace(manifest, requested_holdout_fraction=0.99)
    target = tmp_path / "split_manifest.json"
    persist_split_manifest(target, manifest)

    with pytest.raises(ValueError, match="already persisted"):
        persist_split_manifest(target, conflicting)

    assert load_split_manifest(target) == manifest


def test_validation_folds_only_use_outer_training_buildings(
    modeling_df: pd.DataFrame,
) -> None:
    outer_split = split_known_neighborhood_buildings(modeling_df)
    config = FoldConfig(n_folds=3)
    plan = make_validation_folds(
        outer_split.train_df,
        config=config,
        rng=np.random.default_rng(77),
    )
    outer_train_ids = set(outer_split.train_df["building_id"])
    outer_test_ids = set(outer_split.test_df["building_id"])

    assert len(plan.folds) == config.n_folds
    for fold in plan.folds:
        fit_ids = set(fold.fit_df["building_id"])
        validation_ids = set(fold.validation_df["building_id"])
        assert fit_ids.isdisjoint(validation_ids)
        assert fit_ids | validation_ids == outer_train_ids
        assert validation_ids.isdisjoint(outer_test_ids)


def test_every_non_singleton_building_is_validated_exactly_once(
    modeling_df: pd.DataFrame,
) -> None:
    """Folds rotate rather than resample, so validation exposure is uniform."""
    outer_split = split_known_neighborhood_buildings(modeling_df)
    train_df = outer_split.train_df
    plan = make_validation_folds(train_df, rng=np.random.default_rng(77))

    exposure = Counter(
        building_id
        for fold in plan.folds
        for building_id in fold.validation_df["building_id"]
    )
    unvalidated = set(plan.unvalidated_building_ids)
    for building_id in train_df["building_id"]:
        expected = 0 if building_id in unvalidated else 1
        assert exposure.get(building_id, 0) == expected


def test_unvalidated_buildings_are_exactly_the_training_singletons(
    modeling_df: pd.DataFrame,
) -> None:
    """A lone building cannot validate without emptying its neighborhood."""
    train_df = split_known_neighborhood_buildings(modeling_df).train_df
    plan = make_validation_folds(train_df, rng=np.random.default_rng(77))

    neighborhood_sizes = train_df.groupby("neighborhood_id").size()
    singleton_neighborhoods = set(neighborhood_sizes[neighborhood_sizes == 1].index)
    expected = set(
        train_df.loc[
            train_df["neighborhood_id"].isin(singleton_neighborhoods), "building_id"
        ]
    )
    assert set(plan.unvalidated_building_ids) == expected


def test_every_neighborhood_is_represented_in_every_fit_fold(
    modeling_df: pd.DataFrame,
) -> None:
    """The known-neighborhood claim requires each fold to have seen every group."""
    train_df = split_known_neighborhood_buildings(modeling_df).train_df
    plan = make_validation_folds(train_df, rng=np.random.default_rng(77))

    expected = set(train_df["neighborhood_id"])
    for fold in plan.folds:
        assert set(fold.fit_df["neighborhood_id"]) == expected


def test_validation_folds_are_invariant_to_input_row_order(
    modeling_df: pd.DataFrame,
) -> None:
    train_df = split_known_neighborhood_buildings(modeling_df).train_df
    shuffled_df = train_df.sample(frac=1.0, random_state=11).reset_index(drop=True)

    plan_a = make_validation_folds(train_df, rng=np.random.default_rng(5))
    plan_b = make_validation_folds(shuffled_df, rng=np.random.default_rng(5))

    assert [
        set(fold.validation_df["building_id"]) for fold in plan_a.folds
    ] == [set(fold.validation_df["building_id"]) for fold in plan_b.folds]


def test_caller_generator_advances_between_fold_requests(
    modeling_df: pd.DataFrame,
) -> None:
    """Plan section 4.2: a supplied generator advances as folds are created."""
    train_df = split_known_neighborhood_buildings(modeling_df).train_df
    rng = np.random.default_rng(2024)

    first = make_validation_folds(train_df, rng=rng)
    second = make_validation_folds(train_df, rng=rng)

    assert [set(f.validation_df["building_id"]) for f in first.folds] != [
        set(f.validation_df["building_id"]) for f in second.folds
    ]


def test_single_fold_is_rejected(modeling_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_folds must be at least two"):
        FoldConfig(n_folds=1)


def test_validation_folds_are_reproducible_with_fresh_equal_rngs(
    modeling_df: pd.DataFrame,
) -> None:
    config = FoldConfig(n_folds=3)
    plan_a = make_validation_folds(
        modeling_df, config=config, rng=np.random.default_rng(9)
    )
    plan_b = make_validation_folds(
        modeling_df, config=config, rng=np.random.default_rng(9)
    )

    assert [
        set(fold.validation_df["building_id"]) for fold in plan_a.folds
    ] == [set(fold.validation_df["building_id"]) for fold in plan_b.folds]
    assert len(
        {frozenset(fold.validation_df["building_id"]) for fold in plan_a.folds}
    ) == config.n_folds
