from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from student_simulator import (
    ApartmentSimulator,
    BuildingSimulator,
    CohortCompositionSimulator,
    NeighborhoodSimulator,
    StudentPopulationSimulator,
    TotalChildrenSimulator,
    load_simulation_config,
)

from tests.unit.helpers import (
    make_apartment_simulator,
    make_building_simulator,
    make_cohort_simulator,
    make_neighborhood_simulator,
    make_total_children_simulator,
    with_overrides,
)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"


def _config(seed: int = 123, n_neighborhoods: int = 6):
    cfg = load_simulation_config(_config_path())
    return with_overrides(
        cfg, simulation={"seed": seed, "n_neighborhoods": n_neighborhoods}
    )


def _base_neighborhoods() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002", "N_003", "N_004"],
            "ses": [-0.8, -0.1, 0.6, 1.0],
            "avg_household_size": [2.3, 2.6, 2.9, 3.1],
            "median_age": [34.0, 37.0, 43.0, 46.0],
            "n_daycares_500m": [1, 2, 3, 4],
            "school_status": ["none", "planned", "existing", "existing"],
        }
    )


def _base_buildings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": ["B_000001", "B_000002"],
            "neighborhood_id": ["N_001", "N_002"],
            "3_rooms": [2, 1],
            "4_rooms": [2, 2],
            "5_rooms": [2, 1],
            "6_rooms": [1, 2],
            "n_apartments": [7, 6],
        }
    )


def _result_before_total_children() -> tuple[object, object]:
    cfg = _config(seed=777, n_neighborhoods=5)
    simulator = StudentPopulationSimulator(cfg)
    simulator.run()
    result = simulator.last_result
    assert result is not None
    return result, cfg


def _result_before_cohort() -> tuple[object, object]:
    cfg = _config(seed=888, n_neighborhoods=5)
    simulator = StudentPopulationSimulator(cfg)
    simulator.run()
    result = simulator.last_result
    assert result is not None
    return result, cfg


def test_neighborhood_component_rng_convention() -> None:
    cfg = _config()
    simulator = make_neighborhood_simulator(cfg)
    none_a = simulator.generate(
        cfg.simulation.n_neighborhoods
    )
    none_b = simulator.generate(
        cfg.simulation.n_neighborhoods
    )
    pd.testing.assert_frame_equal(none_a, none_b)

    seed_a = simulator.generate(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(123),
    )
    seed_b = simulator.generate(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(123),
    )
    seed_c = simulator.generate(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(124),
    )

    pd.testing.assert_frame_equal(seed_a, seed_b)
    assert not seed_a.equals(seed_c)


def test_building_stage_rng_convention() -> None:
    cfg = _config()
    simulator = make_building_simulator(cfg)
    neighborhoods = _base_neighborhoods()

    none_a = simulator.generate(neighborhoods)
    none_b = simulator.generate(neighborhoods)
    pd.testing.assert_frame_equal(none_a, none_b)

    seed_a = simulator.generate(
        neighborhoods, rng=np.random.default_rng(222)
    )
    seed_b = simulator.generate(
        neighborhoods, rng=np.random.default_rng(222)
    )
    seed_c = simulator.generate(
        neighborhoods, rng=np.random.default_rng(223)
    )

    pd.testing.assert_frame_equal(seed_a, seed_b)
    assert not seed_a.equals(seed_c)


def test_apartment_component_rng_convention() -> None:
    cfg = _config()
    simulator = make_apartment_simulator(cfg)
    buildings = _base_buildings()

    pd.testing.assert_frame_equal(
        simulator.generate(buildings),
        simulator.generate(buildings),
    )

    seed_a = simulator.generate(buildings, rng=np.random.default_rng(333))
    seed_b = simulator.generate(buildings, rng=np.random.default_rng(333))
    seed_c = simulator.generate(buildings, rng=np.random.default_rng(334))

    pd.testing.assert_frame_equal(seed_a, seed_b)
    assert not seed_a.equals(seed_c)


def test_total_children_component_rng_convention() -> None:
    result, cfg = _result_before_total_children()
    simulator = make_total_children_simulator(cfg)

    context = ApartmentSimulator.with_context(
        result.apartments,
        result.buildings,
        result.neighborhoods,
    )

    pd.testing.assert_frame_equal(
        simulator.generate(context),
        simulator.generate(context),
    )

    seed_a = simulator.generate(context, rng=np.random.default_rng(444))
    seed_b = simulator.generate(context, rng=np.random.default_rng(444))
    seed_c = simulator.generate(context, rng=np.random.default_rng(445))

    pd.testing.assert_frame_equal(seed_a, seed_b)
    assert not seed_a.equals(seed_c)


def test_cohort_component_rng_convention() -> None:
    result, cfg = _result_before_cohort()
    simulator = make_cohort_simulator(cfg)
    context = ApartmentSimulator.with_context(
        result.apartments,
        result.buildings,
        result.neighborhoods,
    )

    pd.testing.assert_frame_equal(
        simulator.generate(context),
        simulator.generate(context),
    )

    seed_a = simulator.generate(context, rng=np.random.default_rng(555))
    seed_b = simulator.generate(context, rng=np.random.default_rng(555))
    seed_c = simulator.generate(context, rng=np.random.default_rng(556))

    pd.testing.assert_frame_equal(seed_a, seed_b)
    assert not seed_a.equals(seed_c)
