"""Shapes and signs of the children counts on one large, fixed-seed sample.

The plan's real-world targets are approximate (docs/SIMULATOR_REALISM_PLAN.md,
R10 and 12.2), so these tests assert directions and coarse ranges, not exact
numbers. Each but the level test fails on the model before step 3 (12.5); the
level guards the intercept. The household-size bands have fixed edges whose
band means are about 12.2's (2.05 ... 4.98).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import ApartmentSimulator, load_simulation_config
from student_simulator.config import Config
from tests.unit.helpers import (
    make_apartment_simulator,
    make_building_simulator,
    make_neighborhood_simulator,
    make_total_children_simulator,
    with_overrides,
)

N_NEIGHBORHOODS = 1_000
BUILDINGS_RATE = 9.0
SEED = 42
HOUSEHOLD_SIZE_BAND_EDGES = [-np.inf, 2.25, 2.625, 3.0, 3.45, 3.95, 4.5, np.inf]
MIDDLE_BAND = 3
# 12.5's numbers, not the config's, so a changed config value fails the tests.
CAP = 10
CHILDLESS_HOUSEHOLD_SIZE = 1.7


def _config() -> Config:
    path = Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    return with_overrides(
        load_simulation_config(path),
        building={"buildings_per_neighborhood_rate": BUILDINGS_RATE},
    )


@pytest.fixture(scope="module")
def config() -> Config:
    return _config()


@pytest.fixture(scope="module")
def apartments(config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    neighborhoods = make_neighborhood_simulator(config).sample_neighborhoods(
        N_NEIGHBORHOODS, rng=rng
    )
    buildings = make_building_simulator(config).sample_buildings(neighborhoods, rng=rng)
    apartments = make_apartment_simulator(config).expand_apartments(buildings, rng)
    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)
    totals, _ = make_total_children_simulator(config).sample_total_children(
        context, rng
    )
    return totals


def _room_ratio(frame: pd.DataFrame) -> float:
    """Mean children in a 6-room apartment over a 3-room one."""
    by_rooms = frame.groupby("rooms")["n_children_total"].mean()
    return float(by_rooms[6] / by_rooms[3])


def test_small_households_hold_few_children(apartments: pd.DataFrame) -> None:
    small = apartments["avg_household_size"] < 2.3

    assert apartments.loc[small, "n_children_total"].mean() < 0.35


def test_children_are_proportional_to_household_size_beyond_the_childless(
    apartments: pd.DataFrame,
) -> None:
    band = pd.cut(
        apartments["avg_household_size"], HOUSEHOLD_SIZE_BAND_EDGES, labels=False
    )
    means = apartments.groupby(band)[["n_children_total", "avg_household_size"]].mean()
    per_person = means["n_children_total"] / (
        means["avg_household_size"] - CHILDLESS_HOUSEHOLD_SIZE
    )
    relative = per_person / per_person[MIDDLE_BAND]

    assert len(relative) == len(HOUSEHOLD_SIZE_BAND_EDGES) - 1
    assert relative.between(0.75, 1.25).all(), relative


def test_crowding_shrinks_the_room_gap_where_households_are_large(
    apartments: pd.DataFrame,
) -> None:
    household_size = apartments["avg_household_size"]
    large = _room_ratio(apartments[household_size > 4.5])
    small = _room_ratio(apartments[household_size < 2.6])

    assert large < small
    assert large < 2.6


def test_counts_are_capped(apartments: pd.DataFrame) -> None:
    counts = apartments["n_children_total"]

    assert counts.max() <= CAP
    assert (counts == CAP).mean() < 0.003


def test_the_level_is_about_nine_tenths(apartments: pd.DataFrame) -> None:
    assert 0.80 <= apartments["n_children_total"].mean() <= 0.95
