from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from student_simulator import (
    ApartmentSimulator,
    CohortCompositionSimulator,
    StudentPopulationSimulator,
    load_simulation_config,
)

from tests.unit.helpers import make_cohort_simulator, with_overrides


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "stage1.toml"


def _sim(seed: int = 42) -> StudentPopulationSimulator:
    cfg = load_simulation_config(_config_path())
    cfg = with_overrides(cfg, simulation={"seed": seed, "n_neighborhoods": 6})
    return StudentPopulationSimulator(cfg)


def _result(simulator: StudentPopulationSimulator):
    simulator.run()
    result = simulator.last_result
    assert result is not None
    return result


def test_run_retains_apartments_and_final_buildings() -> None:
    result = _result(_sim())

    assert not result.apartments.empty
    assert not result.final_df.empty


def test_cohort_counts_nonnegative_and_sum_to_total_per_apartment() -> None:
    sim = _sim()
    apartments = _result(sim).apartments

    for col in ("n_kindergarten", "n_elementary", "n_highschool", "n_children_total"):
        assert pd.api.types.is_integer_dtype(apartments[col])
        assert (apartments[col] >= 0).all()

    cohort_sum = apartments["n_kindergarten"] + apartments["n_elementary"] + apartments["n_highschool"]
    assert np.array_equal(cohort_sum.to_numpy(), apartments["n_children_total"].to_numpy())


def test_cohort_component_returns_counts_without_latent_columns() -> None:
    cfg = load_simulation_config(_config_path())
    result = _result(StudentPopulationSimulator(cfg))
    context = ApartmentSimulator.with_context(
        result.apartments,
        result.buildings,
        result.neighborhoods,
    )

    out = make_cohort_simulator(cfg).generate(
        context,
        rng=np.random.default_rng(71),
    )

    cohort_sum = out[["n_kindergarten", "n_elementary", "n_highschool"]].sum(axis=1)
    assert np.array_equal(cohort_sum.to_numpy(), out["n_children_total"].to_numpy())
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
    context = ApartmentSimulator.with_context(apartments, buildings, neighborhoods)
    out = simulator.generate(context, rng=np.random.default_rng(123))
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


def test_building_targets_equal_grouped_apartment_targets() -> None:
    sim = _sim()
    result = _result(sim)
    apartments = result.apartments
    targets = result.final_df

    grouped = (
        apartments.groupby("building_id", as_index=False)
        .agg(
            n_apartments=("apartment_id", "size"),
            n_kindergarten=("n_kindergarten", "sum"),
            n_elementary=("n_elementary", "sum"),
            n_highschool=("n_highschool", "sum"),
            n_children_total=("n_children_total", "sum"),
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
            "n_children_total",
        ]
    ].sort_values("building_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, grouped)


def test_building_targets_include_context_in_stable_order() -> None:
    sim = _sim()
    targets = _result(sim).final_df

    expected_columns = [
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
    assert list(targets.columns) == expected_columns
    assert targets["building_id"].is_unique


def test_derived_apartment_and_long_cohort_views() -> None:
    sim = _sim()
    result = _result(sim)
    apartments = result.apartments
    buildings = result.buildings
    neighborhoods = result.neighborhoods

    model_view = ApartmentSimulator.with_context(
        apartments,
        buildings,
        neighborhoods,
    )
    assert "neighborhood_id" in model_view.columns
    assert len(model_view) == len(apartments)

    long_view = ApartmentSimulator.to_cohort_long(apartments)
    assert len(long_view) == len(apartments) * 3
    assert set(long_view["cohort"].unique()) == {"kindergarten", "elementary", "highschool"}
