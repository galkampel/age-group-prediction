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

from tests.unit.helpers import make_building_simulator, with_overrides


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "stage1.toml"


def _config(seed: int = 42):
    config = load_simulation_config(_config_path())
    return with_overrides(config, simulation={"seed": seed})


def _neighborhoods() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002", "N_003"],
            "ses": [-0.8, 0.0, 0.9],
            "avg_household_size": [2.2, 2.6, 3.1],
        }
    )


def test_building_generate_contract_and_accounting() -> None:
    config = _config()
    buildings = make_building_simulator(config).generate(
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

    none_a = simulator.generate(_neighborhoods())
    none_b = simulator.generate(_neighborhoods())
    pd.testing.assert_frame_equal(none_a, none_b)

    seed_a = simulator.generate(
        _neighborhoods(), rng=np.random.default_rng(123)
    )
    seed_b = simulator.generate(
        _neighborhoods(), rng=np.random.default_rng(123)
    )
    seed_c = simulator.generate(
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

    buildings = make_building_simulator(config).generate(_neighborhoods())

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
        make_building_simulator(config).generate(_neighborhoods())
