"""Shapes and signs of the building draws on one large, fixed-seed sample.

The plan's real-world targets are approximate (docs/SIMULATOR_REALISM_PLAN.md,
R10 and 11.2), so these tests assert directions and coarse ranges, not exact
numbers. The building type is hidden; a building counts as tower-sized at 52
apartments or more, between the texture buildings' 95th percentile and the
towers' 5th.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import load_simulation_config
from student_simulator.config import Config
from tests.unit.helpers import (
    make_building_simulator,
    make_neighborhood_simulator,
    with_overrides,
)

N_NEIGHBORHOODS = 20_000
BUILDINGS_RATE = 9.0
SEED = 20_261_125
TOWER_SIZED = 52
ROOM_COLUMNS = ["3_rooms", "4_rooms", "5_rooms", "6_rooms"]
ROOMS = np.array([3, 4, 5, 6])


def _config() -> Config:
    path = Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    return with_overrides(
        load_simulation_config(path),
        building={"buildings_per_neighborhood_rate": BUILDINGS_RATE},
    )


@pytest.fixture(scope="module")
def buildings() -> pd.DataFrame:
    config = _config()
    rng = np.random.default_rng(SEED)
    neighborhoods = make_neighborhood_simulator(config).sample_neighborhoods(
        N_NEIGHBORHOODS, rng=rng
    )
    buildings = make_building_simulator(config).sample_buildings(
        neighborhoods, rng=rng
    )
    return buildings.merge(neighborhoods, on="neighborhood_id", validate="many_to_one")


def _mean_rooms(frame: pd.DataFrame) -> float:
    """Mean rooms per apartment over the frame's apartments."""
    return float(frame[ROOM_COLUMNS].sum().to_numpy() @ ROOMS / frame["n_apartments"].sum())


def test_few_sizes_are_clipped(buildings: pd.DataFrame) -> None:
    building_config = _config().building
    bounds = [
        building_config.apartments_per_building_min,
        building_config.apartments_per_building_max,
    ]

    assert buildings["n_apartments"].isin(bounds).mean() < 0.02


def test_sizes_are_right_skewed_around_forty(buildings: pd.DataFrame) -> None:
    size = buildings["n_apartments"]

    assert size.skew() > 1.0
    assert 36 <= size.mean() <= 45
    assert 28 <= size.median() <= 36


def test_buildings_of_one_neighborhood_are_alike(buildings: pd.DataFrame) -> None:
    # One-way ANOVA estimate of the intraclass correlation of log size.
    log_size = np.log(buildings["n_apartments"])
    groups = log_size.groupby(buildings["neighborhood_id"])
    within = groups.var(ddof=1).mean()
    n_per_group = groups.size().mean()
    between = groups.mean().var(ddof=1) - within / n_per_group

    assert between / (between + within) > 0.15


def test_towers_are_a_minority(buildings: pd.DataFrame) -> None:
    assert 0.08 <= (buildings["n_apartments"] >= TOWER_SIZED).mean() <= 0.22


def test_towers_rise_with_ses(buildings: pd.DataFrame) -> None:
    tower = buildings["n_apartments"] >= TOWER_SIZED
    share = tower.groupby(pd.qcut(buildings["ses"], 3, labels=False)).mean()

    assert share.is_monotonic_increasing
    assert share.iloc[0] < share.iloc[-1]


def test_towers_rise_with_age_at_fixed_ses(buildings: pd.DataFrame) -> None:
    # Median age rises with SES, so age is compared inside the middle SES tercile.
    frame = buildings[pd.qcut(buildings["ses"], 3, labels=False) == 1]
    tower = frame["n_apartments"] >= TOWER_SIZED
    share = tower.groupby(pd.qcut(frame["median_age"], 3, labels=False)).mean()

    assert share.is_monotonic_increasing
    assert share.iloc[0] < share.iloc[-1]


def test_a_planned_school_raises_the_tower_share(buildings: pd.DataFrame) -> None:
    middle = pd.qcut(buildings["ses"], 3, labels=False) == 1
    frame = buildings[middle]
    tower = frame["n_apartments"] >= TOWER_SIZED
    planned = frame["school_status"] == "planned"

    assert tower[planned].mean() > tower[~planned].mean()


def test_apartments_grow_with_ses(buildings: pd.DataFrame) -> None:
    quintiles = pd.qcut(buildings["ses"], 5, labels=False)
    by_quintile = pd.Series(
        [_mean_rooms(buildings[quintiles == q]) for q in range(5)]
    )

    assert by_quintile.is_monotonic_increasing


def test_large_households_are_crowded(buildings: pd.DataFrame) -> None:
    large = buildings[buildings["avg_household_size"] > 4.5]
    people_per_room = large["avg_household_size"].mean() / _mean_rooms(large)

    assert len(large) > 1_000
    assert people_per_room > 1.15


def test_six_room_apartments_are_rare(buildings: pd.DataFrame) -> None:
    assert buildings["6_rooms"].sum() / buildings["n_apartments"].sum() < 0.08


def test_larger_buildings_hold_smaller_apartments(buildings: pd.DataFrame) -> None:
    terciles = pd.qcut(buildings["ses"], 3, labels=False)
    tower = buildings["n_apartments"] >= TOWER_SIZED
    for tercile in range(3):
        in_tercile = terciles == tercile
        assert _mean_rooms(buildings[in_tercile & tower]) < _mean_rooms(
            buildings[in_tercile & ~tower]
        )

