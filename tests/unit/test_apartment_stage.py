from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from student_simulator import ApartmentSimulator, StudentPopulationSimulator, load_simulation_config

from tests.unit.helpers import make_apartment_simulator, with_overrides


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "stage1.toml"


def _small_simulator(seed: int = 42) -> StudentPopulationSimulator:
    config = load_simulation_config(_config_path())
    config = with_overrides(
        config,
        simulation={"seed": seed, "n_neighborhoods": 5},
        building={"buildings_per_neighborhood_rate": 1.0},
    )
    return StudentPopulationSimulator(config)


def _small_component(seed: int = 42) -> ApartmentSimulator:
    config = load_simulation_config(_config_path())
    config = with_overrides(config, simulation={"seed": seed})
    return make_apartment_simulator(config)


def _small_result(seed: int = 42):
    simulator = _small_simulator(seed)
    simulator.run()
    result = simulator.last_result
    assert result is not None
    return result


def test_run_keeps_apartments_and_latent_effects_separate() -> None:
    result = _small_result()

    assert set(result.latent_effects) == {
        "building_total_count_effect",
        "neighborhood_total_count_effect",
    }
    assert not any("effect" in column for column in result.apartments)


def test_apartments_exactly_expand_wide_building_room_counts() -> None:
    result = _small_result()
    neighborhoods = result.neighborhoods
    buildings = result.buildings
    apartments = result.apartments

    assert buildings["building_id"].is_unique
    assert apartments["apartment_id"].is_unique
    assert buildings["neighborhood_id"].isin(
        neighborhoods["neighborhood_id"]
    ).all()
    assert apartments["building_id"].isin(buildings["building_id"]).all()

    actual_counts = apartments.groupby(["building_id", "rooms"]).size()
    for _, building in buildings.iterrows():
        for room in (3, 4, 5, 6):
            assert actual_counts.get(
                (building["building_id"], room), 0
            ) == building[f"{room}_rooms"]

    expected_total = int(buildings["n_apartments"].sum())
    assert len(apartments) == expected_total
    assert pd.api.types.is_integer_dtype(apartments["rooms"])


def test_apartment_component_generate_matches_building_room_counts() -> None:
    buildings = pd.DataFrame(
        {
            "building_id": ["B_000001", "B_000002"],
            "neighborhood_id": ["N_001", "N_002"],
            "3_rooms": [1, 2],
            "4_rooms": [1, 0],
            "5_rooms": [0, 1],
            "6_rooms": [2, 0],
            "n_apartments": [4, 3],
        }
    )
    simulator = _small_component(seed=101)

    apartments = simulator.generate(buildings)

    assert apartments["apartment_id"].is_unique
    assert apartments["building_id"].isin(buildings["building_id"]).all()
    assert pd.api.types.is_integer_dtype(apartments["rooms"])

    actual_counts = apartments.groupby(["building_id", "rooms"]).size()
    for _, building in buildings.iterrows():
        for room in (3, 4, 5, 6):
            assert actual_counts.get((building["building_id"], room), 0) == building[f"{room}_rooms"]


def test_apartment_component_with_context_adds_expected_columns() -> None:
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_00000001", "A_00000002"],
            "building_id": ["B_000001", "B_000002"],
            "rooms": [3, 6],
        }
    )
    buildings = pd.DataFrame(
        {
            "building_id": ["B_000001", "B_000002"],
            "neighborhood_id": ["N_001", "N_002"],
        }
    )
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002"],
            "ses": [0.2, -0.4],
            "avg_household_size": [2.6, 3.1],
            "median_age": [35.0, 44.0],
            "n_daycares_500m": [1, 4],
            "school_status": ["none", "existing"],
        }
    )

    out = ApartmentSimulator.with_context(apartments, buildings, neighborhoods)

    assert list(out.columns) == [
        "apartment_id",
        "building_id",
        "rooms",
        "neighborhood_id",
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "school_status",
    ]
    assert len(out) == 2
    assert out["neighborhood_id"].tolist() == ["N_001", "N_002"]


def test_apartment_component_with_context_enforces_many_to_one_cardinality() -> None:
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_00000001"],
            "building_id": ["B_000001"],
            "rooms": [4],
        }
    )
    duplicate_buildings = pd.DataFrame(
        {
            "building_id": ["B_000001", "B_000001"],
            "neighborhood_id": ["N_001", "N_001"],
        }
    )
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001"],
            "ses": [0.0],
            "avg_household_size": [2.7],
            "median_age": [37.0],
            "n_daycares_500m": [2],
            "school_status": ["planned"],
        }
    )

    with pytest.raises(pd.errors.MergeError):
        ApartmentSimulator.with_context(apartments, duplicate_buildings, neighborhoods)


def test_apartment_component_to_cohort_long_preserves_rowwise_totals() -> None:
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_00000001", "A_00000002"],
            "building_id": ["B_000001", "B_000002"],
            "rooms": [3, 5],
            "n_children_total": [3, 2],
            "n_kindergarten": [1, 0],
            "n_elementary": [1, 1],
            "n_highschool": [1, 1],
        }
    )

    long_view = ApartmentSimulator.to_cohort_long(apartments)

    assert set(long_view["cohort"].unique()) == {"kindergarten", "elementary", "highschool"}
    assert len(long_view) == len(apartments) * 3

    grouped = long_view.groupby("apartment_id")["n_children"].sum().sort_index()
    expected = apartments.set_index("apartment_id")["n_children_total"].sort_index()
    pd.testing.assert_series_equal(grouped, expected, check_names=False)


def test_apartment_component_to_cohort_long_requires_columns() -> None:
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_00000001"],
            "building_id": ["B_000001"],
            "rooms": [3],
            "n_children_total": [1],
            "n_kindergarten": [1],
            "n_elementary": [0],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        ApartmentSimulator.to_cohort_long(apartments)
