from pathlib import Path

import pytest

from student_simulator import load_simulation_config
from student_simulator.config import load_validation_config
from tests.unit.helpers import simulate_stages
from tests.validation.helpers import run_calibration_checks


@pytest.mark.calibration
def test_canonical_calibration_ranges_pass() -> None:
    """Verify configured statistical ranges on the canonical full-size seed."""
    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"
    )
    config = load_simulation_config(config_path)
    validation = load_validation_config(config_path.with_name("validation.toml"))
    calibration = run_calibration_checks(
        simulate_stages(config).apartments_with_totals, validation
    )

    assert calibration
    assert all(check.passed for check in calibration), calibration
