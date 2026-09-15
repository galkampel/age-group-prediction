from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from student_simulator import (
    StudentPopulationSimulator,
    load_simulation_config,
)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"


def _simulator(seed: int = 42) -> StudentPopulationSimulator:
    config = load_simulation_config(_config_path())
    config = config.model_copy(
        update={
            "simulation": config.simulation.model_copy(
                update={"seed": seed, "n_neighborhoods": 5}
            )
        }
    )
    return StudentPopulationSimulator(config)


def test_run_returns_final_building_dataframe() -> None:
    final_df = _simulator().run()

    assert list(final_df.columns) == [
        "building_id",
        "neighborhood_id",
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "school_status",
        "3_rooms",
        "4_rooms",
        "5_rooms",
        "6_rooms",
        "n_apartments",
        "n_kindergarten",
        "n_elementary",
        "n_highschool",
        "n_children_total",
    ]
    assert final_df["building_id"].is_unique


def test_run_retains_named_intermediate_tables() -> None:
    simulator = _simulator()
    simulator.run()
    result = simulator.last_result

    assert result is not None
    assert result.neighborhoods["neighborhood_id"].is_unique
    assert result.buildings["building_id"].is_unique
    assert result.apartments["apartment_id"].is_unique
    assert result.final_df["building_id"].is_unique
    assert set(result.latent_effects) == {
        "building_total_count_effect",
        "neighborhood_total_count_effect",
    }


def test_run_and_retained_final_dataframe_match() -> None:
    simulator = _simulator()
    final_df = simulator.run()
    result = simulator.last_result

    assert result is not None
    pd.testing.assert_frame_equal(
        final_df,
        result.final_df,
    )


def test_last_result_is_none_before_the_first_run() -> None:
    assert _simulator().last_result is None


def test_last_result_returns_defensive_copies() -> None:
    simulator = _simulator()
    simulator.run()
    first_result = simulator.last_result
    assert first_result is not None
    first_result.final_df.loc[:, "n_children_total"] = -1
    first_result.latent_effects["building_total_count_effect"].iloc[0] = -1

    second_result = simulator.last_result
    assert second_result is not None
    assert (second_result.final_df["n_children_total"] >= 0).all()
    assert (
        second_result.latent_effects["building_total_count_effect"].iloc[0]
        != -1
    )


def test_repeated_runs_are_reproducible() -> None:
    simulator = _simulator()
    pd.testing.assert_frame_equal(simulator.run(), simulator.run())
    assert not simulator.run().equals(_simulator(seed=43).run())


def test_run_honors_the_supplied_rng() -> None:
    first = _simulator().run(rng=np.random.default_rng(101))
    second = _simulator().run(rng=np.random.default_rng(101))

    pd.testing.assert_frame_equal(first, second)


def test_run_advances_a_shared_rng() -> None:
    simulator = _simulator()
    rng = np.random.default_rng(101)

    assert not simulator.run(rng=rng).equals(simulator.run(rng=rng))
