"""Behavioral baseline for the compact simulator migration.

These tests describe generated data contracts rather than current stage/state
implementation details. They may be updated only when the intended statistical
model or final DataFrame contract changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from student_simulator import (
    SimulatedPopulation,
    StudentPopulationSimulator,
    load_simulation_config,
)


def _config_path() -> Path:
    """Return the canonical simulation configuration path."""
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"


def _run(seed: int = 42, n_neighborhoods: int = 8) -> SimulatedPopulation:
    """Run a small deterministic simulation for contract checks."""
    config = load_simulation_config(_config_path())
    config = config.model_copy(
        update={
            "simulation": config.simulation.model_copy(
                update={"seed": seed, "n_neighborhoods": n_neighborhoods}
            )
        }
    )
    return StudentPopulationSimulator.from_config(config).create_dataset()


def test_dataset_has_expected_columns() -> None:
    """Describe the observable table and fields to preserve during migration."""
    dataset = _run().dataset

    assert {
        "building_id",
        "neighborhood_id",
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "school_status",
        "3_rooms",
        "4_rooms",
        "5_rooms",
        "6_rooms",
        "n_apartments",
        "n_children_total",
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
    } <= set(dataset.columns)


def test_dataset_preserves_accounting_identities() -> None:
    """Lock the data relationships needed by the compact final DataFrame."""
    result = _run()
    dataset = result.dataset

    config = load_simulation_config(_config_path())
    supported_rooms = set(config.building.room_mix.supported_rooms)

    # Key uniqueness and key completeness.
    assert dataset["building_id"].is_unique
    assert dataset["building_id"].notna().all()
    assert dataset["neighborhood_id"].notna().all()

    # The effects belong to the table's buildings and neighborhoods.
    assert result.building_effects.index.isin(dataset["building_id"]).all()
    assert result.neighborhood_effects.index.isin(dataset["neighborhood_id"]).all()

    # Supported room categories are explicit model contracts.
    wide_rooms = {
        int(column.removesuffix("_rooms"))
        for column in dataset.columns
        if column.endswith("_rooms")
    }
    assert wide_rooms == supported_rooms

    # Dtype and non-negativity contracts for integer counts.
    count_columns = [
        "3_rooms",
        "4_rooms",
        "5_rooms",
        "6_rooms",
        "n_apartments",
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
        "n_children_total",
    ]
    for column in count_columns:
        assert pd.api.types.is_integer_dtype(dataset[column])
        assert (dataset[column] >= 0).all()

    room_totals = dataset[["3_rooms", "4_rooms", "5_rooms", "6_rooms"]].sum(axis=1)
    assert room_totals.equals(dataset["n_apartments"])
    cohort_totals = dataset[["n_kindergarten", "n_elementary", "n_highschool"]].sum(
        axis=1
    )
    assert cohort_totals.equals(dataset["n_children_total"])


def test_pipeline_reproducibility_and_seed_variation() -> None:
    """Preserve reproducibility without permanent historical hashes."""
    first = _run(seed=42)
    second = _run(seed=42)
    different_seed = _run(seed=43)

    pd.testing.assert_frame_equal(first.dataset, second.dataset)
    pd.testing.assert_frame_equal(first.building_effects, second.building_effects)
    pd.testing.assert_frame_equal(
        first.neighborhood_effects, second.neighborhood_effects
    )

    assert not first.dataset.equals(different_seed.dataset)
