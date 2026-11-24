from pathlib import Path

import pytest

from student_simulator import load_simulation_config
from tests.unit.helpers import simulate_stages
from tests.validation.helpers import VALIDATION, run_calibration_checks


@pytest.mark.calibration
def test_canonical_calibration_ranges_pass() -> None:
    """Verify configured statistical ranges on the canonical full-size seed."""
    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    )
    config = load_simulation_config(config_path)
    calibration = run_calibration_checks(
        simulate_stages(config).apartments_with_totals, VALIDATION
    )

    assert calibration
    assert all(check.passed for check in calibration), calibration


@pytest.mark.calibration
def test_children_per_apartment_leave_room_for_the_adults() -> None:
    """In 95% of neighborhoods, mean children per apartment is at most H - 1.5.

    Run on the benchmark's setup (150 neighborhoods, building rate 9, seed 42).
    It catches gross miscalibration; the earlier household-size scale (0.5)
    fails it by one neighborhood on this seed.
    """
    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    )
    config = load_simulation_config(config_path).with_overrides(
        {
            "simulation": {"n_neighborhoods": 150},
            "building": {"buildings_per_neighborhood_rate": 9.0},
        }
    )
    apartments = simulate_stages(config).apartments_with_totals
    by_neighborhood = apartments.groupby("neighborhood_id").agg(
        mean_children=("n_children_total", "mean"),
        household_size=("avg_household_size", "first"),
    )
    fits = (
        by_neighborhood["mean_children"]
        <= by_neighborhood["household_size"] - 1.5
    )

    assert fits.mean() >= 0.95, by_neighborhood[~fits]


@pytest.mark.calibration
def test_large_household_neighborhoods_leave_room_for_the_adults() -> None:
    """Of the neighborhoods with H > 4.5, at least 80% average at most H - 1.5 children.

    The rule over all neighborhoods barely moves when the household-size scale
    is too small; the large-household tail does (measured: 0.95-0.97 at scale
    0.75, 0.24-0.49 at 0.5). 1000 neighborhoods give about 39 such areas.
    """
    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    )
    config = load_simulation_config(config_path).with_overrides(
        {"simulation": {"n_neighborhoods": 1000}}
    )
    apartments = simulate_stages(config).apartments_with_totals
    by_neighborhood = apartments.groupby("neighborhood_id").agg(
        mean_children=("n_children_total", "mean"),
        household_size=("avg_household_size", "first"),
    )
    tail = by_neighborhood[by_neighborhood["household_size"] > 4.5]
    fits = tail["mean_children"] <= tail["household_size"] - 1.5

    assert len(tail) >= 20
    assert fits.mean() >= 0.80, tail[~fits]
