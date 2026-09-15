from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from student_simulator import (
    ApartmentSimulator,
    StudentPopulationSimulator,
    TotalChildrenSimulator,
    load_simulation_config,
)

from tests.unit.helpers import make_total_children_simulator


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"


def _state_for_log_mu_checks(
) -> tuple[StudentPopulationSimulator, dict[str, pd.DataFrame]]:
    cfg = load_simulation_config(_config_path())
    sim = StudentPopulationSimulator(cfg)

    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002"],
            "ses": [0.0, 1.0],
            "avg_household_size": [2.6, 3.1],
            "median_age": [37.0, 47.0],
            "n_daycares_500m": [0, 3],
            "school_status": ["none", "existing"],
        }
    )
    buildings = pd.DataFrame(
        {
            "building_id": ["B_001", "B_002"],
            "neighborhood_id": ["N_001", "N_002"],
        }
    )
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_001", "A_002", "A_003", "A_004"],
            "building_id": ["B_001", "B_001", "B_002", "B_002"],
            "rooms": [3, 4, 5, 6],
        }
    )
    building_effect = pd.Series([0.0, 0.0], index=["B_001", "B_002"])
    neighborhood_effect = pd.Series([0.0, 0.0], index=["N_001", "N_002"])

    context = ApartmentSimulator.with_context(
        apartments, buildings, neighborhoods
    )
    return sim, {
        "context": context,
        "building_effect": building_effect,
        "neighborhood_effect": neighborhood_effect,
    }


def test_total_children_component_adds_nonnegative_integer_counts() -> None:
    cfg = load_simulation_config(_config_path())
    sim = StudentPopulationSimulator(cfg)
    sim.run()
    result = sim.last_result
    assert result is not None
    apartments = result.apartments

    assert "n_children_total" in apartments.columns
    assert pd.api.types.is_integer_dtype(apartments["n_children_total"])
    assert (apartments["n_children_total"] >= 0).all()


def test_log_mu_finite_and_positive_mu() -> None:
    sim, parts = _state_for_log_mu_checks()
    simulator = make_total_children_simulator(sim.config)

    log_mu = simulator.compute_log_mu(
        parts["context"],
        parts["building_effect"],
        parts["neighborhood_effect"],
    )
    mu = np.exp(log_mu)

    assert np.isfinite(log_mu).all()
    assert np.isfinite(mu).all()
    assert (mu > 0).all()


def test_total_children_component_keeps_latent_effects_out_of_columns(
) -> None:
    sim, parts = _state_for_log_mu_checks()
    out = make_total_children_simulator(sim.config).generate(
        parts["context"],
        rng=np.random.default_rng(123),
    )

    assert "n_children_total" in out.columns
    assert not any("effect" in column for column in out.columns)


def test_generated_random_effects_use_named_owner_sets() -> None:
    sim, parts = _state_for_log_mu_checks()
    simulator = make_total_children_simulator(sim.config)
    context = parts["context"]

    simulator.generate(
        context,
        rng=np.random.default_rng(10),
    )
    effects = simulator.latent_effects()
    building_effect = effects["building_total_count_effect"]
    neighborhood_effect = effects["neighborhood_total_count_effect"]

    assert set(effects) == {
        "building_total_count_effect",
        "neighborhood_total_count_effect",
    }
    assert building_effect.index.tolist() == ["B_001", "B_002"]
    assert neighborhood_effect.index.tolist() == ["N_001", "N_002"]
    assert building_effect.index.name == "building_id"
    assert neighborhood_effect.index.name == "neighborhood_id"


def test_room_effect_log_mean_differences_match_coefficients() -> None:
    sim, _ = _state_for_log_mu_checks()
    simulator = make_total_children_simulator(sim.config)

    # Keep contextual terms fixed so room effects can be read directly.
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
        {"building_id": ["B_001"], "neighborhood_id": ["N_001"]}
    )
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_3", "A_4", "A_5", "A_6"],
            "building_id": ["B_001", "B_001", "B_001", "B_001"],
            "rooms": [3, 4, 5, 6],
        }
    )
    building_effect = pd.Series([0.0], index=["B_001"])
    neighborhood_effect = pd.Series([0.0], index=["N_001"])

    context = ApartmentSimulator.with_context(
        apartments, buildings, neighborhoods
    )
    log_mu = simulator.compute_log_mu(
        context,
        building_effect,
        neighborhood_effect,
    )

    tcfg = sim.config.total_children
    assert np.isclose(log_mu[1] - log_mu[0], tcfg.room_log_mean_effects[4])
    assert np.isclose(log_mu[2] - log_mu[0], tcfg.room_log_mean_effects[5])
    assert np.isclose(log_mu[3] - log_mu[0], tcfg.room_log_mean_effects[6])


def test_household_size_main_effect_positive_for_supported_rooms() -> None:
    cfg = load_simulation_config(_config_path())
    for room in cfg.building.room_mix.supported_rooms:
        slope = (
            cfg.total_children.household_size_coef
            + cfg.total_children.rooms_household_interaction_coef * (room - 4)
        )
        assert slope > 0


def test_shared_effects_constant_for_same_owner_descendants() -> None:
    sim, _ = _state_for_log_mu_checks()
    simulator = make_total_children_simulator(sim.config)

    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_001", "A_002", "A_003"],
            "building_id": ["B_001", "B_001", "B_002"],
            "rooms": [4, 4, 4],
        }
    )
    buildings = pd.DataFrame(
        {
            "building_id": ["B_001", "B_002"],
            "neighborhood_id": ["N_001", "N_002"],
        }
    )
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002"],
            "ses": [0.0, 0.0],
            "avg_household_size": [2.6, 2.6],
            "median_age": [37.0, 37.0],
            "n_daycares_500m": [0, 0],
            "school_status": ["none", "none"],
        }
    )
    building_effect = pd.Series([0.7, -0.2], index=["B_001", "B_002"])
    neighborhood_effect = pd.Series([0.5, -0.1], index=["N_001", "N_002"])

    context = ApartmentSimulator.with_context(
        apartments, buildings, neighborhoods
    )
    log_mu = simulator.compute_log_mu(
        context,
        building_effect,
        neighborhood_effect,
    )

    assert np.isclose(log_mu[0], log_mu[1])
    assert not np.isclose(log_mu[0], log_mu[2])
