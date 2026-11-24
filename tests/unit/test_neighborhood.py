from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import load_simulation_config
from tests.unit.helpers import make_neighborhood_simulator, with_overrides


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"


def _config(seed: int = 42, n_neighborhoods: int = 7):
    cfg = load_simulation_config(_config_path())
    return with_overrides(
        cfg, simulation={"seed": seed, "n_neighborhoods": n_neighborhoods}
    )


def test_sample_neighborhoods_contract_and_bounds() -> None:
    cfg = _config(seed=42, n_neighborhoods=9)
    simulator = make_neighborhood_simulator(cfg)
    neighborhoods = simulator.sample_neighborhoods(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(42),
    )

    assert len(neighborhoods) == cfg.simulation.n_neighborhoods
    assert list(neighborhoods.columns) == [
        "neighborhood_id",
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "school_status",
    ]

    assert neighborhoods["neighborhood_id"].is_unique
    assert neighborhoods["neighborhood_id"].iloc[0] == "N_001"

    ncfg = cfg.neighborhood
    assert neighborhoods["ses"].between(
        ncfg.ses_min, ncfg.ses_max
    ).all()
    assert neighborhoods["median_age"].between(
        ncfg.median_age_min,
        ncfg.median_age_max,
    ).all()
    assert neighborhoods["avg_household_size"].between(
        ncfg.household_size_min, ncfg.household_size_max
    ).all()

    assert pd.api.types.is_integer_dtype(neighborhoods["n_daycares_500m"])
    assert (neighborhoods["n_daycares_500m"] >= 0).all()
    statuses = set(neighborhoods["school_status"].unique())
    assert statuses.issubset({"existing", "planned", "none"})


def test_neighborhood_rng_convention_for_component_calls() -> None:
    cfg = _config(seed=999, n_neighborhoods=6)

    simulator = make_neighborhood_simulator(cfg)
    none_a = simulator.sample_neighborhoods(
        cfg.simulation.n_neighborhoods
    )
    none_b = simulator.sample_neighborhoods(
        cfg.simulation.n_neighborhoods
    )
    pd.testing.assert_frame_equal(none_a, none_b)

    seed_a = simulator.sample_neighborhoods(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(123),
    )
    seed_b = simulator.sample_neighborhoods(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(123),
    )
    seed_c = simulator.sample_neighborhoods(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(124),
    )

    pd.testing.assert_frame_equal(seed_a, seed_b)
    assert not seed_a.equals(seed_c)


def test_sample_neighborhoods_has_no_latent_effect_columns() -> None:
    cfg = _config(seed=42, n_neighborhoods=4)
    simulator = make_neighborhood_simulator(cfg)
    neighborhoods = simulator.sample_neighborhoods(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(42),
    )

    forbidden = {
        "neighborhood_total_count_effect",
        "total_children_effect",
        "cohort_effect",
    }
    assert forbidden.isdisjoint(set(neighborhoods.columns))


def test_sample_neighborhoods_rejects_non_positive_count() -> None:
    cfg = _config()

    with pytest.raises(ValueError, match="greater than zero"):
        make_neighborhood_simulator(cfg).sample_neighborhoods(0)


def test_sample_neighborhoods_follows_the_documented_draws_and_rng_order() -> None:
    # The oracle rebuilds every column from one generator in the documented
    # order: SES, age noise, household noise, the Poisson, one uniform per row.
    cfg = _config(seed=42, n_neighborhoods=50)
    ncfg = cfg.neighborhood
    neighborhoods = make_neighborhood_simulator(cfg).sample_neighborhoods(
        50, rng=np.random.default_rng(42)
    )

    rng = np.random.default_rng(42)
    ses = np.clip(
        rng.normal(ncfg.ses_mean, ncfg.ses_sd, 50), ncfg.ses_min, ncfg.ses_max
    )
    age = np.clip(
        ncfg.median_age_baseline
        + ncfg.median_age_ses_slope * ses
        + rng.normal(0.0, ncfg.median_age_noise_sd, 50),
        ncfg.median_age_min,
        ncfg.median_age_max,
    )
    age_offset = age - ncfg.median_age_baseline
    household_size = np.clip(
        ncfg.household_size_baseline
        + ncfg.household_size_age_slope * age_offset
        + ncfg.household_size_ses_slope * ses
        + ncfg.household_size_ses_quadratic_coef * ses**2
        + rng.normal(0.0, ncfg.household_size_noise_sd, 50),
        ncfg.household_size_min,
        ncfg.household_size_max,
    )
    household_offset = household_size - ncfg.household_size_baseline
    daycares = rng.poisson(
        np.exp(
            ncfg.daycare_log_intercept
            + ncfg.daycare_household_size_coef * household_offset
            + ncfg.daycare_ses_coef * ses
            + ncfg.daycare_age_coef * age_offset
        )
    )
    shares = ncfg.school_status_reference_shares
    weights = np.column_stack(
        [
            shares["existing"]
            * np.exp(ncfg.school_existing_household_size_coef * household_offset),
            shares["planned"] * np.exp(ncfg.school_planned_age_coef * age_offset),
            np.full(50, shares["none"]),
        ]
    )
    cumulative = np.cumsum(weights / weights.sum(axis=1, keepdims=True), axis=1)
    uniforms = rng.random(50)
    labels = np.array(["existing", "planned", "none"], dtype=object)
    school_status = labels[(uniforms[:, None] >= cumulative[:, :2]).sum(axis=1)]

    np.testing.assert_array_equal(neighborhoods["ses"], ses)
    np.testing.assert_allclose(neighborhoods["median_age"], age, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        neighborhoods["avg_household_size"], household_size, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(neighborhoods["n_daycares_500m"], daycares)
    np.testing.assert_array_equal(neighborhoods["school_status"], school_status)


def test_ses_draws_are_unchanged_by_the_new_model() -> None:
    # SES is the first draw and keeps its formula. The literals are the
    # first five SES values the previous model produced for seed 42.
    cfg = _config(seed=42, n_neighborhoods=150)
    neighborhoods = make_neighborhood_simulator(cfg).sample_neighborhoods(
        150, rng=np.random.default_rng(42)
    )

    np.testing.assert_array_equal(
        neighborhoods["ses"].to_numpy()[:5],
        [
            0.30471707975443135,
            -1.0399841062404955,
            0.7504511958064572,
            0.9405647163912139,
            -1.9510351886538364,
        ],
    )


def test_reference_neighborhood_gets_the_reference_shares_and_rate() -> None:
    # With near-zero spread every row is the reference neighborhood
    # (household size and age at their baselines), where the school shares
    # are the reference shares and the daycare rate is exp(log intercept).
    cfg = _config()
    cfg = with_overrides(
        cfg,
        neighborhood={
            "ses_sd": 1e-9,
            "median_age_noise_sd": 1e-9,
            "household_size_noise_sd": 1e-9,
        },
    )
    ncfg = cfg.neighborhood
    neighborhoods = make_neighborhood_simulator(cfg).sample_neighborhoods(
        40_000, rng=np.random.default_rng(5)
    )

    np.testing.assert_allclose(
        neighborhoods["avg_household_size"], ncfg.household_size_baseline, atol=1e-6
    )
    np.testing.assert_allclose(
        neighborhoods["median_age"], ncfg.median_age_baseline, atol=1e-6
    )
    shares = neighborhoods["school_status"].value_counts(normalize=True)
    for label, reference_share in ncfg.school_status_reference_shares.items():
        assert shares[label] == pytest.approx(reference_share, abs=0.01)
    assert neighborhoods["n_daycares_500m"].mean() == pytest.approx(
        np.exp(ncfg.daycare_log_intercept), abs=0.03
    )
