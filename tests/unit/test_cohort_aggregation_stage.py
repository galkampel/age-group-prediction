from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import (
    ApartmentSimulator,
    CohortCompositionSimulator,
    SimulationResult,
    StudentPopulationSimulator,
    aggregate_to_buildings,
    load_simulation_config,
)
from student_simulator.config import Config

from tests.unit.helpers import make_cohort_simulator, simulate_stages, with_overrides


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"


def _config(seed: int = 42) -> Config:
    cfg = load_simulation_config(_config_path())
    return with_overrides(cfg, simulation={"seed": seed, "n_neighborhoods": 6})


def _population(seed: int = 42) -> SimulationResult:
    return StudentPopulationSimulator.from_config(_config(seed)).simulate()


def test_simulate_returns_a_nonempty_dataset() -> None:
    assert not _population().dataset.empty


def test_cohort_counts_nonnegative_and_sum_to_total_per_apartment() -> None:
    apartments = simulate_stages(_config()).apartments_with_cohorts

    for col in ("n_kindergarten", "n_elementary", "n_highschool", "n_children_total"):
        assert pd.api.types.is_integer_dtype(apartments[col])
        assert (apartments[col] >= 0).all()

    cohort_sum = apartments["n_kindergarten"] + apartments["n_elementary"] + apartments["n_highschool"]
    assert np.array_equal(cohort_sum.to_numpy(), apartments["n_children_total"].to_numpy())


def test_cohort_component_returns_counts_without_latent_columns() -> None:
    cfg = load_simulation_config(_config_path())
    context = simulate_stages(cfg).apartments_with_totals

    out, _ = make_cohort_simulator(cfg).sample_cohorts(
        context,
        rng=np.random.default_rng(71),
    )

    cohort_sum = out[["n_kindergarten", "n_elementary", "n_highschool"]].sum(axis=1)
    assert np.array_equal(cohort_sum.to_numpy(), out["n_children_total"].to_numpy())
    assert not any("effect" in column for column in out.columns)


def test_cohort_effect_is_returned_per_building_and_not_as_a_column() -> None:
    cfg = load_simulation_config(_config_path())
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_00000001", "A_00000002", "A_00000003"],
            "building_id": ["B_000001", "B_000001", "B_000002"],
            "rooms": [3, 5, 4],
            "n_children_total": [2, 0, 3],
        }
    )
    buildings = pd.DataFrame(
        {
            "building_id": ["B_000001", "B_000002"],
            "neighborhood_id": ["N_001", "N_002"],
        }
    )
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002"],
            "ses": [0.0, 0.5],
            "avg_household_size": [2.6, 3.0],
            "median_age": [37.0, 42.0],
            "n_daycares_500m": [0, 2],
            "school_status": ["none", "existing"],
        }
    )
    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)

    out, effects = make_cohort_simulator(cfg).sample_cohorts(
        context, rng=np.random.default_rng(5)
    )

    assert set(effects) == {"building_cohort_effect"}
    building_effect = effects["building_cohort_effect"]
    assert building_effect.index.tolist() == ["B_000001", "B_000002"]
    assert building_effect.index.name == "building_id"
    assert not any("effect" in column for column in out.columns)


def test_zero_child_apartment_receives_all_zero_cohorts() -> None:
    cfg = load_simulation_config(_config_path())
    simulator = make_cohort_simulator(cfg)
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_00000001"],
            "building_id": ["B_000001"],
            "rooms": [4],
            "n_children_total": [0],
        }
    )
    buildings = pd.DataFrame(
        {"building_id": ["B_000001"], "neighborhood_id": ["N_001"]}
    )
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001"],
            "ses": [0.0],
            "avg_household_size": [2.6],
            "median_age": [37.0],
            "n_daycares_500m": [0],
            "school_status": ["none"],
        }
    )
    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)
    out, _ = simulator.sample_cohorts(context, rng=np.random.default_rng(123))
    assert int(out.loc[0, "n_kindergarten"]) == 0
    assert int(out.loc[0, "n_elementary"]) == 0
    assert int(out.loc[0, "n_highschool"]) == 0


def test_cohort_probabilities_finite_sum_to_one_and_shift_with_theta() -> None:
    cfg = load_simulation_config(_config_path())
    simulator = make_cohort_simulator(cfg)

    theta = np.array([-1.0, 0.0, 1.0], dtype=float)
    probs = simulator.cohort_probabilities(theta)
    assert np.isfinite(probs).all()
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Positive lifecycle shift should reduce kindergarten mass and increase high-school mass.
    assert probs[2, 0] < probs[1, 0] < probs[0, 0]
    assert probs[2, 2] > probs[1, 2] > probs[0, 2]

    ref = np.array(
        [
            cfg.cohort.reference_shares["kindergarten"],
            cfg.cohort.reference_shares["elementary"],
            cfg.cohort.reference_shares["highschool"],
        ]
    )
    assert np.allclose(probs[1], ref)


def test_cohort_probabilities_remain_finite_for_extreme_lifecycle_scores() -> None:
    cfg = load_simulation_config(_config_path())
    simulator = make_cohort_simulator(cfg)

    probabilities = simulator.cohort_probabilities(
        np.array([-1_000.0, 1_000.0])
    )

    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def one_apartment_per_neighborhood(
    rooms: list[int],
    avg_household_size: list[float],
    median_age: list[float],
    n_daycares_500m: list[int],
    school_status: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Each apartment in its own building and neighborhood, with zero building effects."""
    building_ids = [f"B_{index:06d}" for index in range(1, len(rooms) + 1)]
    neighborhood_ids = [f"N_{index:03d}" for index in range(1, len(rooms) + 1)]
    apartments = pd.DataFrame(
        {
            "apartment_id": [f"A_{index:08d}" for index in range(1, len(rooms) + 1)],
            "building_id": building_ids,
            "rooms": rooms,
            "n_children_total": [2] * len(rooms),
        }
    )
    buildings = pd.DataFrame(
        {"building_id": building_ids, "neighborhood_id": neighborhood_ids}
    )
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": neighborhood_ids,
            "ses": [0.0] * len(rooms),
            "avg_household_size": avg_household_size,
            "median_age": median_age,
            "n_daycares_500m": n_daycares_500m,
            "school_status": school_status,
        }
    )
    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)
    zero_effect = pd.Series(0.0, index=pd.Index(building_ids, name="building_id"))
    return context, zero_effect


@pytest.mark.parametrize("school_status", ["none", "existing"])
def test_the_reference_apartment_gets_the_reference_shares(school_status: str) -> None:
    simulator = make_cohort_simulator(load_simulation_config(_config_path()))
    context, zero_effect = one_apartment_per_neighborhood(
        rooms=[4],
        avg_household_size=[3.0],
        median_age=[31.0],
        n_daycares_500m=[0],
        school_status=[school_status],
    )

    theta = simulator.compute_theta(context, zero_effect)

    assert np.allclose(theta, [0.0])
    assert np.allclose(simulator.cohort_probabilities(theta), [[0.225, 0.385, 0.390]])


def test_each_score_term_follows_the_plan() -> None:
    # The reference apartment, then one apartment per term moved off its reference.
    simulator = make_cohort_simulator(load_simulation_config(_config_path()))
    context, zero_effect = one_apartment_per_neighborhood(
        rooms=[4, 6, 4, 4, 4, 4],
        avg_household_size=[3.0, 3.0, 3.0, 3.0, 3.75, 3.0],
        median_age=[31.0, 31.0, 31.0, 41.0, 31.0, 31.0],
        n_daycares_500m=[0, 0, 3, 0, 0, 0],
        school_status=["none", "none", "none", "none", "none", "planned"],
    )

    theta = simulator.compute_theta(context, zero_effect)

    # 13.1: 0.20 (r - 4) - 0.20 (1 - exp(-D / 3)) + 0.25 (M - 31) / 10
    #       + 0.05 (H - 3.0) / 0.75 - 0.25 P
    expected = [0.0, 0.40, -0.20 * (1 - np.exp(-1.0)), 0.25, 0.05, -0.25]
    assert np.allclose(theta, expected)


def test_a_planned_school_lowers_the_score_and_shifts_the_split_younger() -> None:
    simulator = make_cohort_simulator(load_simulation_config(_config_path()))
    context, zero_effect = one_apartment_per_neighborhood(
        rooms=[4, 4],
        avg_household_size=[3.4, 3.4],
        median_age=[35.0, 35.0],
        n_daycares_500m=[2, 2],
        school_status=["existing", "planned"],
    )

    theta = simulator.compute_theta(context, zero_effect)
    probabilities = simulator.cohort_probabilities(theta)

    assert np.isclose(theta[1] - theta[0], -0.25)
    assert probabilities[1, 0] > probabilities[0, 0]
    assert probabilities[1, 2] < probabilities[0, 2]


def test_aggregate_to_buildings_sums_the_apartment_counts() -> None:
    stages = simulate_stages(_config())
    apartments = stages.apartments_with_cohorts
    targets = aggregate_to_buildings(
        apartments, stages.buildings_df, stages.neighborhoods_df
    )

    grouped = (
        apartments.groupby("building_id", as_index=False)
        .agg(
            n_apartments=("apartment_id", "size"),
            n_kindergarten=("n_kindergarten", "sum"),
            n_elementary=("n_elementary", "sum"),
            n_highschool=("n_highschool", "sum"),
        )
        .sort_values("building_id")
        .reset_index(drop=True)
    )

    expected = targets[
        [
            "building_id",
            "n_apartments",
            "n_kindergarten",
            "n_elementary",
            "n_highschool",
        ]
    ].sort_values("building_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, grouped)


def test_long_cohort_view_has_one_row_per_apartment_and_cohort() -> None:
    apartments = simulate_stages(_config()).apartments_with_cohorts

    long_view = ApartmentSimulator.to_cohort_long(apartments)
    assert len(long_view) == len(apartments) * 3
    assert set(long_view["cohort"].unique()) == {"kindergarten", "elementary", "highschool"}
