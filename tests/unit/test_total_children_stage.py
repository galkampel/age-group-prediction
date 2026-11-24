from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import (
    ApartmentSimulator,
    Config,
    load_simulation_config,
)
from student_simulator.features import log_household_size_excess
from tests.unit.helpers import (
    make_total_children_simulator,
    simulate_stages,
    with_overrides,
)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"


def _state_for_log_mu_checks() -> tuple[Config, dict[str, pd.DataFrame]]:
    cfg = load_simulation_config(_config_path())

    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002"],
            "ses": [0.0, 1.0],
            "avg_household_size": [3.0, 3.1],
            "median_age": [31.0, 47.0],
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

    context = ApartmentSimulator.join_context(
        apartments, buildings, neighborhoods
    )
    return cfg, {
        "context": context,
        "building_effect": building_effect,
        "neighborhood_effect": neighborhood_effect,
    }


def test_total_children_component_adds_nonnegative_integer_counts() -> None:
    cfg = load_simulation_config(_config_path())
    apartments = simulate_stages(cfg).apartments_with_totals

    assert "n_children_total" in apartments.columns
    assert pd.api.types.is_integer_dtype(apartments["n_children_total"])
    assert (apartments["n_children_total"] >= 0).all()


def test_log_mu_finite_and_positive_mu() -> None:
    cfg, parts = _state_for_log_mu_checks()
    simulator = make_total_children_simulator(cfg)

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
    cfg, parts = _state_for_log_mu_checks()
    out, _ = make_total_children_simulator(cfg).sample_total_children(
        parts["context"],
        rng=np.random.default_rng(123),
    )

    assert "n_children_total" in out.columns
    assert not any("effect" in column for column in out.columns)


def test_generated_random_effects_use_named_owner_sets() -> None:
    cfg, parts = _state_for_log_mu_checks()
    simulator = make_total_children_simulator(cfg)
    context = parts["context"]

    _, effects = simulator.sample_total_children(
        context,
        rng=np.random.default_rng(10),
    )
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
    cfg, _ = _state_for_log_mu_checks()
    simulator = make_total_children_simulator(cfg)

    # Keep contextual terms fixed so room effects can be read directly; at the
    # reference household size the rooms x household interaction is zero.
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001"],
            "ses": [0.0],
            "avg_household_size": [3.0],
            "median_age": [31.0],
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

    context = ApartmentSimulator.join_context(
        apartments, buildings, neighborhoods
    )
    log_mu = simulator.compute_log_mu(
        context,
        building_effect,
        neighborhood_effect,
    )

    tcfg = cfg.total_children
    assert np.isclose(log_mu[1] - log_mu[0], tcfg.room_log_mean_effects[4])
    assert np.isclose(log_mu[2] - log_mu[0], tcfg.room_log_mean_effects[5])
    assert np.isclose(log_mu[3] - log_mu[0], tcfg.room_log_mean_effects[6])


def test_household_size_main_effect_positive_for_supported_rooms() -> None:
    # The slope of log mu in x = log((H - 1.7) / 1.3), room by room: from
    # H = 3.0 (x = 0) to H = 4.3 (x = log 2). 12.1: 1.0 - 0.15 * (rooms - 4).
    cfg = load_simulation_config(_config_path())
    simulator = make_total_children_simulator(cfg)
    rooms = [3, 4, 5, 6]
    effects = (pd.Series([0.0], index=["B_001"]), pd.Series([0.0], index=["N_001"]))
    log_mu = {
        household_size: simulator.compute_log_mu(
            _one_neighborhood_context(household_size, rooms), *effects
        )
        for household_size in (3.0, 4.3)
    }

    slope = (log_mu[4.3] - log_mu[3.0]) / np.log(2.0)

    assert (slope > 0).all()
    assert np.allclose(slope, [1.15, 1.0, 0.85, 0.70])


def _one_neighborhood_context(
    household_size: float, rooms: list[int]
) -> pd.DataFrame:
    """One reference neighborhood (SES 0, age 31, no daycare or school)."""
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001"],
            "ses": [0.0],
            "avg_household_size": [household_size],
            "median_age": [31.0],
            "n_daycares_500m": [0],
            "school_status": ["none"],
        }
    )
    buildings = pd.DataFrame(
        {"building_id": ["B_001"], "neighborhood_id": ["N_001"]}
    )
    apartments = pd.DataFrame(
        {
            "apartment_id": [f"A_{index}" for index in range(len(rooms))],
            "building_id": ["B_001"] * len(rooms),
            "rooms": rooms,
        }
    )
    return ApartmentSimulator.join_context(apartments, buildings, neighborhoods)


def test_log_household_size_term_is_zero_at_the_reference_and_adds_log_two() -> None:
    cfg = load_simulation_config(_config_path())
    reference = cfg.transforms.household_size_reference
    childless = cfg.total_children.childless_household_size
    doubled = childless + 2 * (reference - childless)

    values = log_household_size_excess(
        [reference, doubled], reference=reference, childless=childless
    )

    assert np.isclose(values[0], 0.0)
    assert np.isclose(values[1] - values[0], np.log(2.0))


def test_room_differences_include_the_crowding_interaction() -> None:
    cfg = load_simulation_config(_config_path())
    tcfg = cfg.total_children
    household_size = 4.0
    context = _one_neighborhood_context(household_size, [3, 4, 5, 6])
    log_mu = make_total_children_simulator(cfg).compute_log_mu(
        context,
        pd.Series([0.0], index=["B_001"]),
        pd.Series([0.0], index=["N_001"]),
    )

    x = np.log(
        (household_size - tcfg.childless_household_size)
        / (cfg.transforms.household_size_reference - tcfg.childless_household_size)
    )
    effects = {3: 0.0, **tcfg.room_log_mean_effects}
    expected = np.array(
        [
            effects[room]
            + tcfg.rooms_household_interaction_coef
            * (room - cfg.transforms.room_reference)
            * x
            for room in (3, 4, 5, 6)
        ]
    )
    assert np.allclose(log_mu - log_mu[0], expected - expected[0])


def test_counts_are_the_replayed_draw_clipped_at_the_cap() -> None:
    # An intercept of 2.0 puts mu near the cap, so some draws are clipped and
    # some are not.
    cfg = with_overrides(
        load_simulation_config(_config_path()), total_children={"intercept": 2.0}
    )
    tcfg = cfg.total_children
    simulator = make_total_children_simulator(cfg)
    context = _one_neighborhood_context(3.0, [3, 4, 5, 6] * 100)

    out, effects = simulator.sample_total_children(
        context, rng=np.random.default_rng(31)
    )

    # Replay: the building effect, the neighborhood effect, then one NB2 draw
    # per apartment.
    replay = np.random.default_rng(31)
    replay.normal(0.0, tcfg.building_effect_sd, 1)
    replay.normal(0.0, tcfg.neighborhood_effect_sd, 1)
    mu = np.exp(
        simulator.compute_log_mu(
            context,
            effects["building_total_count_effect"],
            effects["neighborhood_total_count_effect"],
        )
    )
    phi = tcfg.nb_dispersion_phi
    draw = replay.negative_binomial(n=phi, p=phi / (phi + mu))

    counts = out["n_children_total"].to_numpy()
    assert np.array_equal(counts, np.minimum(draw, tcfg.max_children))
    assert (draw > tcfg.max_children).any()
    assert (draw < tcfg.max_children).any()


def test_a_very_large_intercept_puts_every_count_at_the_cap() -> None:
    cfg = with_overrides(
        load_simulation_config(_config_path()), total_children={"intercept": 8.0}
    )
    context = _one_neighborhood_context(3.0, [3, 4, 5, 6] * 25)

    out, _ = make_total_children_simulator(cfg).sample_total_children(
        context, rng=np.random.default_rng(8)
    )

    assert (out["n_children_total"] == cfg.total_children.max_children).all()


def test_a_reference_at_the_childless_size_raises() -> None:
    with pytest.raises(ValueError, match="reference must exceed"):
        log_household_size_excess([3.0], reference=1.7, childless=1.7)


@pytest.mark.parametrize("household_size", [1.7, 1.5])
def test_a_household_size_at_or_below_the_childless_size_raises(
    household_size: float,
) -> None:
    cfg = load_simulation_config(_config_path())
    context = _one_neighborhood_context(household_size, [3, 4])

    with pytest.raises(ValueError, match="childless household size"):
        make_total_children_simulator(cfg).sample_total_children(
            context, rng=np.random.default_rng(0)
        )


def test_shared_effects_constant_for_same_owner_descendants() -> None:
    cfg, _ = _state_for_log_mu_checks()
    simulator = make_total_children_simulator(cfg)

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
            "avg_household_size": [3.0, 3.0],
            "median_age": [31.0, 31.0],
            "n_daycares_500m": [0, 0],
            "school_status": ["none", "none"],
        }
    )
    building_effect = pd.Series([0.7, -0.2], index=["B_001", "B_002"])
    neighborhood_effect = pd.Series([0.5, -0.1], index=["N_001", "N_002"])

    context = ApartmentSimulator.join_context(
        apartments, buildings, neighborhoods
    )
    log_mu = simulator.compute_log_mu(
        context,
        building_effect,
        neighborhood_effect,
    )

    assert np.isclose(log_mu[0], log_mu[1])
    assert not np.isclose(log_mu[0], log_mu[2])
