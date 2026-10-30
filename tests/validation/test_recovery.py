from pathlib import Path

import pytest

from student_simulator import load_simulation_config
from tests.unit.helpers import simulate_stages
from tests.validation.helpers import VALIDATION, run_recovery_checks


@pytest.mark.slow
def test_oracle_nb2_recovers_fixed_effects() -> None:
    """Verify oracle recovery with generated offsets from the generator."""
    config = load_simulation_config(
        Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    )
    stages = simulate_stages(config)

    results = run_recovery_checks(
        stages.apartments_with_totals, stages.total_effects, config, VALIDATION
    )
    oracle = next(
        check
        for check in results
        if check.name == "oracle_nb2_fixed_effect_recovery"
    )
    assert oracle.passed, oracle.observed
