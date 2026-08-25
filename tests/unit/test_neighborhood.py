from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_simulator import NeighborhoodSimulator, load_simulation_config

from tests.unit.helpers import make_neighborhood_simulator, with_overrides


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "stage1.toml"


def _config(seed: int = 42, n_neighborhoods: int = 7):
    cfg = load_simulation_config(_config_path())
    return with_overrides(
        cfg, simulation={"seed": seed, "n_neighborhoods": n_neighborhoods}
    )


def test_neighborhood_generate_contract_and_bounds() -> None:
    cfg = _config(seed=42, n_neighborhoods=9)
    simulator = make_neighborhood_simulator(cfg)
    neighborhoods = simulator.generate(
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
    none_a = simulator.generate(
        cfg.simulation.n_neighborhoods
    )
    none_b = simulator.generate(
        cfg.simulation.n_neighborhoods
    )
    pd.testing.assert_frame_equal(none_a, none_b)

    seed_a = simulator.generate(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(123),
    )
    seed_b = simulator.generate(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(123),
    )
    seed_c = simulator.generate(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(124),
    )

    pd.testing.assert_frame_equal(seed_a, seed_b)
    assert not seed_a.equals(seed_c)


def test_neighborhood_generation_has_no_latent_effect_columns() -> None:
    cfg = _config(seed=42, n_neighborhoods=4)
    simulator = make_neighborhood_simulator(cfg)
    neighborhoods = simulator.generate(
        cfg.simulation.n_neighborhoods,
        rng=np.random.default_rng(42),
    )

    forbidden = {
        "neighborhood_total_count_effect",
        "total_children_effect",
        "cohort_effect",
    }
    assert forbidden.isdisjoint(set(neighborhoods.columns))


def test_neighborhood_generate_rejects_non_positive_count() -> None:
    cfg = _config()

    with pytest.raises(ValueError, match="greater than zero"):
        make_neighborhood_simulator(cfg).generate(0)
