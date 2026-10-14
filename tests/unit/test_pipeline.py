from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import (
    ApartmentSimulator,
    BuildingSimulator,
    CohortCompositionSimulator,
    NeighborhoodSimulator,
    StudentPopulationSimulator,
    TotalChildrenSimulator,
    aggregate_to_buildings,
    load_simulation_config,
)
from student_simulator.config import Config
from tests.unit.helpers import with_overrides


def _config(seed: int = 42) -> Config:
    config = load_simulation_config(
        Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"
    )
    return with_overrides(config, simulation={"seed": seed, "n_neighborhoods": 5})


def _assert_same_population(first, second) -> None:
    pd.testing.assert_frame_equal(first.dataset, second.dataset)
    pd.testing.assert_frame_equal(first.building_effects, second.building_effects)
    pd.testing.assert_frame_equal(
        first.neighborhood_effects, second.neighborhood_effects
    )


def test_simulate_returns_the_dataset_with_the_canonical_columns() -> None:
    dataset = StudentPopulationSimulator.from_config(_config()).simulate().dataset

    assert list(dataset.columns) == [
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
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
    ]
    assert dataset["building_id"].is_unique


def test_effect_frames_have_one_row_per_owner() -> None:
    population = StudentPopulationSimulator.from_config(_config()).simulate()
    dataset = population.dataset

    assert list(population.building_effects.columns) == ["total_count", "cohort"]
    assert list(population.neighborhood_effects.columns) == ["total_count"]
    assert population.building_effects.index.name == "building_id"
    assert population.neighborhood_effects.index.name == "neighborhood_id"
    assert population.building_effects.index.tolist() == (
        dataset["building_id"].tolist()
    )
    assert population.neighborhood_effects.index.tolist() == (
        dataset["neighborhood_id"].unique().tolist()
    )
    assert population.building_effects.notna().all(axis=None)


def test_the_same_seed_reproduces_the_dataset_and_another_seed_changes_it() -> None:
    simulator = StudentPopulationSimulator.from_config(_config(seed=42))
    other_seed = StudentPopulationSimulator.from_config(_config(seed=43))

    _assert_same_population(simulator.simulate(), simulator.simulate())
    assert not simulator.simulate().dataset.equals(other_seed.simulate().dataset)


def test_a_passed_generator_replaces_the_seed_and_is_advanced() -> None:
    simulator = StudentPopulationSimulator.from_config(_config(seed=42))

    _assert_same_population(
        simulator.simulate(np.random.default_rng(42)),
        simulator.simulate(),
    )
    assert not simulator.simulate(np.random.default_rng(7)).dataset.equals(
        simulator.simulate().dataset
    )
    shared = np.random.default_rng(7)
    assert not simulator.simulate(shared).dataset.equals(
        simulator.simulate(shared).dataset
    )


def test_from_config_builds_the_same_dataset_as_explicit_construction() -> None:
    config = _config()
    simulation = config.simulation
    explicit = StudentPopulationSimulator(
        NeighborhoodSimulator(simulation, config.neighborhood),
        BuildingSimulator(simulation, config.building, config.transforms),
        ApartmentSimulator(simulation, config.building.room_mix),
        TotalChildrenSimulator(simulation, config.total_children, config.transforms),
        CohortCompositionSimulator(
            simulation, config.cohort, config.transforms, config.total_children
        ),
        n_neighborhoods=simulation.n_neighborhoods,
        seed=simulation.seed,
    )

    _assert_same_population(
        explicit.simulate(),
        StudentPopulationSimulator.from_config(config).simulate(),
    )


def test_components_see_one_shared_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = StudentPopulationSimulator.from_config(_config())
    seen: list[np.random.Generator] = []
    for component, method in (
        (simulator.neighborhood_simulator, "sample_neighborhoods"),
        (simulator.building_simulator, "sample_buildings"),
        (simulator.apartment_simulator, "expand_apartments"),
        (simulator.total_children_simulator, "sample_total_children"),
        (simulator.cohort_composition_simulator, "sample_cohorts"),
    ):
        original = getattr(component, method)

        def spy(data, rng, _original=original):
            seen.append(rng)
            return _original(data, rng)

        monkeypatch.setattr(component, method, spy)

    simulator.simulate()

    assert len(seen) == 5
    assert all(rng is seen[0] for rng in seen)


def test_a_building_without_apartments_gets_zero_counts() -> None:
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001"],
            "ses": [0.0],
            "avg_household_size": [2.6],
            "median_age": [37.0],
            "n_daycares_500m": [0],
            "school_status": ["none"],
        }
    )
    buildings = pd.DataFrame(
        {
            "building_id": ["B_000001", "B_000002"],
            "neighborhood_id": ["N_001", "N_001"],
            "3_rooms": [1, 0],
            "4_rooms": [0, 0],
            "5_rooms": [0, 0],
            "6_rooms": [0, 0],
            "n_apartments": [1, 0],
        }
    )
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_00000001"],
            "building_id": ["B_000001"],
            "n_children_total": [2],
            "n_kindergarten": [1],
            "n_elementary": [1],
            "n_highschool": [0],
        }
    )

    dataset = aggregate_to_buildings(apartments, buildings, neighborhoods)

    counts = dataset.set_index("building_id").loc[
        "B_000002",
        ["n_kindergarten", "n_elementary", "n_highschool"],
    ]
    assert counts.tolist() == [0, 0, 0]
    assert pd.api.types.is_integer_dtype(dataset["n_highschool"])
