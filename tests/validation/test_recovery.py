from pathlib import Path

import pytest

from student_simulator import (
    StudentPopulationSimulator,
    load_simulation_config,
)
from student_simulator.config import load_validation_config
from tests.validation.helpers import run_recovery_checks

pytest.importorskip("statsmodels")


@pytest.mark.slow
def test_oracle_nb2_recovers_fixed_effects() -> None:
    """Verify oracle recovery with generated offsets from the generator."""
    config = load_simulation_config(
        Path(__file__).resolve().parents[2] / "configs" / "stage1.toml"
    )
    validation = load_validation_config(
        Path(__file__).resolve().parents[2] / "configs" / "validation.toml"
    )
    simulator = StudentPopulationSimulator(config)
    simulator.run()
    result = simulator.last_result
    assert result is not None

    results = run_recovery_checks(result, config, validation)
    oracle = next(
        check
        for check in results
        if check.name == "oracle_nb2_fixed_effect_recovery"
    )
    assert oracle.passed, oracle.observed
