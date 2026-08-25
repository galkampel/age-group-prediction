"""Behavioral baseline for the compact simulator migration.

These tests describe generated data contracts rather than current stage/state
implementation details. They may be updated only when the intended statistical
model or final DataFrame contract changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from student_simulator import (
    StudentPopulationSimulator,
    load_simulation_config,
)


def _config_path() -> Path:
    """Return the canonical simulation configuration path."""
    return Path(__file__).resolve().parents[2] / "configs" / "stage1.toml"


def _run(seed: int = 42, n_neighborhoods: int = 8):
    """Run a small deterministic simulation for contract checks."""
    config = load_simulation_config(_config_path())
    config = config.model_copy(
        update={
            "simulation": config.simulation.model_copy(
                update={"seed": seed, "n_neighborhoods": n_neighborhoods}
            )
        }
    )
    simulator = StudentPopulationSimulator(config)
    simulator.run()
    result = simulator.last_result
    assert result is not None
    return result


def test_canonical_tables_have_expected_columns() -> None:
    """Describe observable tables and fields to preserve during migration."""
    result = _run()

    expected_columns = {
        "neighborhoods": {
            "neighborhood_id",
            "ses",
            "avg_household_size",
            "median_age",
            "n_daycares_500m",
            "school_status",
        },
        "buildings": {
            "building_id",
            "neighborhood_id",
            "3_rooms",
            "4_rooms",
            "5_rooms",
            "6_rooms",
            "n_apartments",
        },
        "apartments": {
            "apartment_id",
            "building_id",
            "rooms",
            "n_children_total",
            "n_kindergarten",
            "n_elementary",
            "n_highschool",
        },
    }

    tables = {
        "neighborhoods": result.neighborhoods,
        "buildings": result.buildings,
        "apartments": result.apartments,
    }
    for table_name, columns in expected_columns.items():
        assert columns <= set(tables[table_name].columns)
    assert {
        "building_id",
        "n_apartments",
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
        "n_children_total",
    } <= set(result.final_df.columns)


def test_canonical_tables_preserve_accounting_identities() -> None:
    """Lock the data relationships needed by the compact final DataFrame."""
    result = _run()
    neighborhoods = result.neighborhoods
    buildings = result.buildings
    apartments = result.apartments
    building_targets = result.final_df

    config = load_simulation_config(_config_path())
    supported_rooms = set(
        config.building.room_mix.supported_rooms
    )

    # Key uniqueness and key completeness.
    assert neighborhoods["neighborhood_id"].is_unique
    assert buildings["building_id"].is_unique
    assert apartments["apartment_id"].is_unique
    assert building_targets["building_id"].is_unique
    assert neighborhoods["neighborhood_id"].notna().all()
    assert buildings["building_id"].notna().all()
    assert buildings["neighborhood_id"].notna().all()
    assert apartments["apartment_id"].notna().all()
    assert apartments["building_id"].notna().all()
    assert building_targets["building_id"].notna().all()

    # Foreign-key identities.
    assert buildings["neighborhood_id"].isin(
        neighborhoods["neighborhood_id"]
    ).all()
    assert apartments["building_id"].isin(buildings["building_id"]).all()

    # Supported room categories are explicit model contracts.
    wide_rooms = {
        int(column.removesuffix("_rooms"))
        for column in buildings.columns
        if column.endswith("_rooms")
    }
    assert wide_rooms == supported_rooms
    assert set(apartments["rooms"].unique()).issubset(supported_rooms)

    # Dtype and non-negativity contracts for integer outcomes.
    building_count_columns = [
        "3_rooms",
        "4_rooms",
        "5_rooms",
        "6_rooms",
        "n_apartments",
    ]
    for column in building_count_columns:
        assert pd.api.types.is_integer_dtype(buildings[column])
        assert (buildings[column] >= 0).all()
    integer_columns = [
        "rooms",
        "n_children_total",
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
    ]
    for column in integer_columns:
        assert pd.api.types.is_integer_dtype(apartments[column])
        assert (apartments[column] >= 0).all()

    target_columns = [
        "n_apartments",
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
        "n_children_total",
    ]
    for column in target_columns:
        assert pd.api.types.is_integer_dtype(building_targets[column])
        assert (building_targets[column] >= 0).all()

    room_totals = buildings.set_index("building_id")[
        ["3_rooms", "4_rooms", "5_rooms", "6_rooms"]
    ].sum(axis=1)
    apartment_totals = apartments.groupby("building_id")["apartment_id"].size()
    pd.testing.assert_series_equal(
        room_totals,
        apartment_totals,
        check_names=False,
    )

    assert (
        apartments["n_kindergarten"]
        + apartments["n_elementary"]
        + apartments["n_highschool"]
    ).equals(apartments["n_children_total"])

    expected_targets = (
        apartments.groupby("building_id", as_index=False)
        .agg(
            n_apartments=("apartment_id", "size"),
            n_kindergarten=("n_kindergarten", "sum"),
            n_elementary=("n_elementary", "sum"),
            n_highschool=("n_highschool", "sum"),
            n_children_total=("n_children_total", "sum"),
        )
        .sort_values("building_id")
        .reset_index(drop=True)
    )
    observed_targets = building_targets[
        [
            "building_id",
            "n_apartments",
            "n_kindergarten",
            "n_elementary",
            "n_highschool",
            "n_children_total",
        ]
    ]
    pd.testing.assert_frame_equal(observed_targets, expected_targets)


def test_pipeline_reproducibility_and_seed_variation() -> None:
    """Preserve reproducibility without permanent historical hashes."""
    first = _run(seed=42)
    second = _run(seed=42)
    different_seed = _run(seed=43)

    for table_name in ("neighborhoods", "buildings", "apartments", "final_df"):
        pd.testing.assert_frame_equal(
            getattr(first, table_name),
            getattr(second, table_name),
        )

    assert not first.neighborhoods.equals(different_seed.neighborhoods)
