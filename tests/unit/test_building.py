from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import (
    BuildingSimulator,
    ConfigError,
    load_simulation_config,
)
from student_simulator.features import expected_room_probabilities

from tests.unit.helpers import make_building_simulator, with_overrides


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"


def _config(seed: int = 42):
    config = load_simulation_config(_config_path())
    return with_overrides(config, simulation={"seed": seed})


def _neighborhoods() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002", "N_003"],
            "ses": [-0.8, 0.0, 0.9],
            "avg_household_size": [2.2, 2.6, 3.1],
            "median_age": [27.0, 31.0, 36.0],
            "school_status": ["none", "planned", "existing"],
        }
    )


def test_sample_buildings_contract_and_accounting() -> None:
    config = _config()
    buildings = make_building_simulator(config).sample_buildings(
        _neighborhoods(),
        rng=np.random.default_rng(42),
    )

    assert list(buildings.columns) == [
        "building_id",
        "neighborhood_id",
        "3_rooms",
        "4_rooms",
        "5_rooms",
        "6_rooms",
        "n_apartments",
    ]
    assert buildings["building_id"].is_unique
    assert buildings["building_id"].iloc[0] == "B_000001"
    assert buildings["neighborhood_id"].isin(
        _neighborhoods()["neighborhood_id"]
    ).all()

    room_columns = ["3_rooms", "4_rooms", "5_rooms", "6_rooms"]
    for column in [*room_columns, "n_apartments"]:
        assert pd.api.types.is_integer_dtype(buildings[column])
        assert (buildings[column] >= 0).all()

    building_config = config.building
    assert buildings["n_apartments"].between(
        building_config.apartments_per_building_min,
        building_config.apartments_per_building_max,
    ).all()
    assert buildings[room_columns].sum(axis=1).equals(
        buildings["n_apartments"]
    )


def test_building_rng_convention_for_component_calls() -> None:
    config = _config(seed=999)
    simulator = make_building_simulator(config)

    none_a = simulator.sample_buildings(_neighborhoods())
    none_b = simulator.sample_buildings(_neighborhoods())
    pd.testing.assert_frame_equal(none_a, none_b)

    seed_a = simulator.sample_buildings(
        _neighborhoods(), rng=np.random.default_rng(123)
    )
    seed_b = simulator.sample_buildings(
        _neighborhoods(), rng=np.random.default_rng(123)
    )
    seed_c = simulator.sample_buildings(
        _neighborhoods(), rng=np.random.default_rng(124)
    )
    pd.testing.assert_frame_equal(seed_a, seed_b)
    assert not seed_a.equals(seed_c)


def test_building_exact_overrides_are_respected() -> None:
    config = _config()
    config = with_overrides(
        config,
        building={
            "buildings_per_neighborhood": (1, 2, 1),
            "apartments_per_building": (12, 13, 14, 15),
        },
    )

    buildings = make_building_simulator(config).sample_buildings(_neighborhoods())

    assert len(buildings) == 4
    assert buildings["n_apartments"].tolist() == [12, 13, 14, 15]


def test_building_apartment_override_length_mismatch_raises() -> None:
    config = _config()
    config = with_overrides(
        config,
        building={
            "buildings_per_neighborhood": (1, 2, 1),
            "apartments_per_building": (12, 13, 14),
        },
    )

    with pytest.raises(ConfigError, match="apartments_per_building"):
        make_building_simulator(config).sample_buildings(_neighborhoods())



def _replay_buildings(config, neighborhoods: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Redraw the building stage with numpy alone, in the documented order."""
    building_config = config.building
    room_config = building_config.room_mix
    transforms = config.transforms
    rng = np.random.default_rng(seed)
    n_neighborhoods = len(neighborhoods)
    counts = building_config.buildings_per_neighborhood_min + rng.poisson(
        building_config.buildings_per_neighborhood_rate, n_neighborhoods
    )
    rows = neighborhoods.loc[neighborhoods.index.repeat(counts)].reset_index(drop=True)
    tower_propensity = np.repeat(
        rng.normal(0.0, building_config.tower_propensity_sd, n_neighborhoods), counts
    )
    texture = np.repeat(
        rng.normal(0.0, building_config.apartments_neighborhood_log_sd, n_neighborhoods),
        counts,
    )
    tower_log_odds = (
        building_config.tower_log_odds_intercept
        + building_config.tower_ses_coef * rows["ses"]
        + building_config.tower_median_age_coef
        * (rows["median_age"] - transforms.median_age_reference)
        / transforms.median_age_scale
        + building_config.tower_planned_school_coef * (rows["school_status"] == "planned")
        + tower_propensity
    ).to_numpy()
    is_tower = rng.random(len(rows)) < 1.0 / (1.0 + np.exp(-tower_log_odds))
    median = np.where(
        is_tower,
        building_config.tower_apartments_median,
        building_config.texture_apartments_median,
    )
    building_noise = rng.normal(0.0, building_config.apartments_building_log_sd, len(rows))
    totals = np.clip(
        np.rint(median * np.exp(texture + building_noise)).astype(int),
        building_config.apartments_per_building_min,
        building_config.apartments_per_building_max,
    )
    expected_mix = expected_room_probabilities(
        supported_rooms=room_config.supported_rooms,
        base_shares=room_config.base_shares,
        household_size_tilt=room_config.household_size_tilt,
        ses_tilt=room_config.ses_tilt,
        building_size_tilt=room_config.building_size_tilt,
        room_distance_divisor=room_config.room_distance_divisor,
        household_size_scaled=(
            rows["avg_household_size"] - transforms.household_size_reference
        ).to_numpy()
        / transforms.household_size_scale,
        ses=rows["ses"].to_numpy(),
        log_relative_building_size=np.log(
            totals / building_config.texture_apartments_median
        ),
    )
    room_counts = np.array(
        [
            rng.multinomial(
                int(total),
                rng.dirichlet(room_config.dirichlet_concentration * building_mix),
            )
            for total, building_mix in zip(totals, expected_mix, strict=True)
        ]
    )
    replayed = pd.DataFrame({"neighborhood_id": rows["neighborhood_id"]})
    for index, room in enumerate(room_config.supported_rooms):
        replayed[f"{room}_rooms"] = room_counts[:, index]
    replayed["n_apartments"] = totals
    return replayed


def test_buildings_equal_a_replay_of_the_documented_draws() -> None:
    config = _config()
    neighborhoods = _neighborhoods()

    buildings = make_building_simulator(config).sample_buildings(
        neighborhoods, rng=np.random.default_rng(7)
    )
    expected = _replay_buildings(config, neighborhoods, seed=7)

    pd.testing.assert_frame_equal(
        buildings.drop(columns="building_id"), expected, check_dtype=False
    )


def test_the_tower_probability_follows_its_logistic_formula() -> None:
    # With the hidden effects and the size noise near zero (the config requires
    # SDs above 0), a building's size is exactly its type's median, so the type
    # can be read from the size. Four kinds of neighborhood: the reference, with
    # a planned school, one SES unit higher, ten years older.
    kinds = [
        (0.0, 31.0, "none"),
        (0.0, 31.0, "planned"),
        (1.0, 31.0, "existing"),
        (0.0, 41.0, "none"),
    ]
    per_kind, per_neighborhood = 2_000, 5
    neighborhoods = pd.DataFrame(
        [
            {
                "neighborhood_id": f"N_{kind_index}_{index:05d}",
                "ses": ses,
                "avg_household_size": 3.0,
                "median_age": median_age,
                "school_status": school_status,
            }
            for kind_index, (ses, median_age, school_status) in enumerate(kinds)
            for index in range(per_kind)
        ]
    )
    config = with_overrides(
        _config(),
        building={
            "tower_propensity_sd": 1e-9,
            "apartments_neighborhood_log_sd": 1e-9,
            "apartments_building_log_sd": 1e-9,
            "buildings_per_neighborhood": (per_neighborhood,) * len(neighborhoods),
        },
    )
    buildings = make_building_simulator(config).sample_buildings(
        neighborhoods, rng=np.random.default_rng(11)
    )
    building_config = config.building
    kind_of_building = np.repeat(np.arange(len(kinds)), per_kind * per_neighborhood)
    tower_share = (
        (buildings["n_apartments"] == building_config.tower_apartments_median)
        .groupby(kind_of_building)
        .mean()
    )
    log_odds = building_config.tower_log_odds_intercept + np.array(
        [
            0.0,
            building_config.tower_planned_school_coef,
            building_config.tower_ses_coef,
            building_config.tower_median_age_coef,
        ]
    )

    assert set(buildings["n_apartments"]) == {
        building_config.texture_apartments_median,
        building_config.tower_apartments_median,
    }
    # 10,000 buildings per kind: a standard error of at most 0.004.
    assert np.allclose(tower_share.to_numpy(), 1.0 / (1.0 + np.exp(-log_odds)), atol=0.015)
