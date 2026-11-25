"""Shapes and signs of the cohort split on one large, fixed-seed sample.

The plan's real-world targets are approximate (docs/SIMULATOR_REALISM_PLAN.md,
R10 and 13.2), so these tests assert directions and coarse ranges on realized
shares, not exact numbers. The first four fail on the model before step 4
(13.5); the last three are guards that pass on both. The band edges are fixed:
the household-size bands are 13.2's, and the median-age bands reproduce 13.5's
numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import load_simulation_config
from tests.unit.helpers import simulate_stages, with_overrides

N_NEIGHBORHOODS = 1_000
BUILDINGS_RATE = 9.0
SEED = 42
HOUSEHOLD_SIZE_BAND_EDGES = [-np.inf, 2.25, 2.625, 3.0, 3.45, 3.95, 4.5, np.inf]
REFERENCE_BAND = 3
TOP_BAND = 6
MEDIAN_AGE_BAND_EDGES = [-np.inf, 24, 28, 31, 34, 38, np.inf]
LARGE_BUILDING = 60
MIN_BUILDING_CHILDREN = 20


@pytest.fixture(scope="module")
def apartments() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    config = with_overrides(
        load_simulation_config(path),
        simulation={"seed": SEED, "n_neighborhoods": N_NEIGHBORHOODS},
        building={"buildings_per_neighborhood_rate": BUILDINGS_RATE},
    )
    stages = simulate_stages(config)
    frame = stages.apartments_with_cohorts.copy()
    n_apartments = stages.buildings_df.set_index("building_id")["n_apartments"]
    frame["n_apartments"] = frame["building_id"].map(n_apartments)
    return frame


def realized_shares(frame: pd.DataFrame) -> pd.Series:
    """Kindergarten, elementary and high-school shares of all children in `frame`."""
    counts = frame[["n_kindergarten", "n_elementary", "n_highschool"]].sum()
    return counts / frame["n_children_total"].sum()


def kindergarten_share_by(frame: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """The realized kindergarten share of each group."""
    sums = frame.groupby(groups)[["n_kindergarten", "n_children_total"]].sum()
    return sums["n_kindergarten"] / sums["n_children_total"]


def test_the_level(apartments: pd.DataFrame) -> None:
    shares = realized_shares(apartments)

    assert 0.24 <= shares["n_kindergarten"] <= 0.30, shares
    assert 0.35 <= shares["n_elementary"] <= 0.41, shares
    assert 0.32 <= shares["n_highschool"] <= 0.38, shares


@pytest.mark.parametrize("median_age_band", [None, (28.0, 34.0)])
def test_a_planned_school_holds_younger_children(
    apartments: pd.DataFrame, median_age_band: tuple[float, float] | None
) -> None:
    frame = apartments
    if median_age_band is not None:
        frame = frame[frame["median_age"].between(*median_age_band)]
    planned = frame["school_status"] == "planned"
    difference = realized_shares(frame[planned]) - realized_shares(frame[~planned])

    assert difference["n_kindergarten"] > 0.04, difference
    assert difference["n_highschool"] < -0.04, difference


def test_large_households_tilt_to_the_young(apartments: pd.DataFrame) -> None:
    band = pd.cut(
        apartments["avg_household_size"], HOUSEHOLD_SIZE_BAND_EDGES, labels=False
    )
    kindergarten = kindergarten_share_by(apartments, band)

    assert kindergarten[TOP_BAND] - kindergarten[REFERENCE_BAND] > 0.02, kindergarten


def test_the_room_gap_is_moderate(apartments: pd.DataFrame) -> None:
    kindergarten = kindergarten_share_by(apartments, apartments["rooms"])

    assert list(kindergarten.index) == [3, 4, 5, 6]
    assert (kindergarten.diff().dropna() < 0).all(), kindergarten
    assert 0.08 <= kindergarten[3] - kindergarten[6] <= 0.22, kindergarten


def test_older_areas_hold_older_children(apartments: pd.DataFrame) -> None:
    band = pd.cut(apartments["median_age"], MEDIAN_AGE_BAND_EDGES, labels=False)
    kindergarten = kindergarten_share_by(apartments, band)

    assert len(kindergarten) == len(MEDIAN_AGE_BAND_EDGES) - 1
    assert (kindergarten.diff().dropna() < 0).all(), kindergarten
    assert kindergarten.iloc[0] - kindergarten.iloc[-1] > 0.06, kindergarten


def test_building_size_has_no_effect(apartments: pd.DataFrame) -> None:
    large = apartments["n_apartments"] >= LARGE_BUILDING
    kindergarten = kindergarten_share_by(apartments, large)

    assert abs(kindergarten[True] - kindergarten[False]) < 0.03, kindergarten


def test_buildings_differ(apartments: pd.DataFrame) -> None:
    sums = apartments.groupby("building_id")[
        ["n_kindergarten", "n_children_total"]
    ].sum()
    sums = sums[sums["n_children_total"] >= MIN_BUILDING_CHILDREN]
    kindergarten = sums["n_kindergarten"] / sums["n_children_total"]

    assert 0.07 <= kindergarten.std() <= 0.14, kindergarten.describe()
