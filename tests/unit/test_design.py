from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from student_simulator import ApartmentSimulator
from student_simulator.config import load_simulation_config
from tests.unit.helpers import make_cohort_simulator, make_total_children_simulator


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"


def _sample_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002"],
            "ses": [0.3, -0.8],
            # Row 0 sits at the reference neighborhood (H 3.0, age 31).
            "avg_household_size": [3.0, 3.1],
            "median_age": [31.0, 42.0],
            "n_daycares_500m": [0, 5],
            "school_status": ["existing", "planned"],
            # Proxy truth must never leak into observed design matrices.
            "research_truth_noise": [0.1, -0.2],
        }
    )
    buildings = pd.DataFrame(
        {
            "building_id": ["B_001", "B_002"],
            "neighborhood_id": ["N_001", "N_002"],
            "research_truth_building": [0.5, -0.3],
        }
    )
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_001", "A_002", "A_003"],
            "building_id": ["B_001", "B_001", "B_002"],
            "rooms": [3, 5, 6],
            "research_truth_apartment": [1.0, 2.0, 3.0],
        }
    )
    return apartments, buildings, neighborhoods


def test_total_children_design_columns_match_config_keys_order() -> None:
    config = load_simulation_config(_config_path())
    simulator = make_total_children_simulator(config)
    apartments, buildings, neighborhoods = _sample_tables()
    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)

    design = simulator.build_design_matrix(context)

    assert list(design.columns) == [
        "intercept",
        "rooms_4",
        "rooms_5",
        "rooms_6",
        "ses_squared",
        "log_household_size_excess",
        "daycare_saturation",
        "school_existing",
        "school_planned",
        "median_age_scaled",
        "rooms_household_interaction",
    ]


def test_total_children_school_dummies_mutually_exclusive() -> None:
    config = load_simulation_config(_config_path())
    simulator = make_total_children_simulator(config)
    apartments, buildings, neighborhoods = _sample_tables()
    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)

    design = simulator.build_design_matrix(context)
    both = design["school_existing"] + design["school_planned"]
    assert np.all((both == 0) | (both == 1))
    assert (both > 1).sum() == 0


def test_cohort_lifecycle_features_have_semantic_columns() -> None:
    config = load_simulation_config(_config_path())
    simulator = make_cohort_simulator(config)
    apartments, buildings, neighborhoods = _sample_tables()
    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)

    features = simulator._build_lifecycle_features(context)
    assert list(features.columns) == [
        "rooms_centered",
        "daycare_saturation",
        "median_age_scaled",
        "household_size_scaled",
        "school_planned",
    ]


def test_cohort_school_planned_is_one_only_for_a_planned_school() -> None:
    config = load_simulation_config(_config_path())
    simulator = make_cohort_simulator(config)
    apartments = pd.DataFrame(
        {
            "apartment_id": ["A_001", "A_002", "A_003"],
            "building_id": ["B_001", "B_002", "B_003"],
            "rooms": [4, 4, 4],
        }
    )
    buildings = pd.DataFrame(
        {
            "building_id": ["B_001", "B_002", "B_003"],
            "neighborhood_id": ["N_001", "N_002", "N_003"],
        }
    )
    neighborhoods = pd.DataFrame(
        {
            "neighborhood_id": ["N_001", "N_002", "N_003"],
            "ses": [0.0, 0.0, 0.0],
            "avg_household_size": [3.0, 3.0, 3.0],
            "median_age": [31.0, 31.0, 31.0],
            "n_daycares_500m": [1, 1, 1],
            "school_status": ["planned", "existing", "none"],
        }
    )

    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)
    features = simulator._build_lifecycle_features(context)
    assert features["school_planned"].tolist() == [1.0, 0.0, 0.0]


def test_design_matrices_exclude_research_truth_columns() -> None:
    config = load_simulation_config(_config_path())
    total_simulator = make_total_children_simulator(config)
    cohort_simulator = make_cohort_simulator(config)
    apartments, buildings, neighborhoods = _sample_tables()

    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)
    total_design = total_simulator.build_design_matrix(context)
    cohort_features = cohort_simulator._build_lifecycle_features(context)

    for col in total_design.columns:
        assert "research_truth" not in col
    for col in cohort_features.columns:
        assert "research_truth" not in col


def test_cohort_lifecycle_features_follow_configured_transform_flow() -> None:
    config = load_simulation_config(_config_path())
    apartments, buildings, neighborhoods = _sample_tables()
    context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)

    features = make_cohort_simulator(config)._build_lifecycle_features(context)

    assert list(features.columns) == [
        "rooms_centered",
        "daycare_saturation",
        "median_age_scaled",
        "household_size_scaled",
        "school_planned",
    ]
    assert features.loc[0, "rooms_centered"] == -1.0
    assert features.loc[0, "household_size_scaled"] == 0.0
    assert features.loc[0, "median_age_scaled"] == 0.0
    assert features.loc[0, "daycare_saturation"] == 0.0
    # Row 0 is in the neighborhood with an existing school, row 2 in the planned one.
    assert features.loc[0, "school_planned"] == 0.0
    assert features.loc[2, "school_planned"] == 1.0
