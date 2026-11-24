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
