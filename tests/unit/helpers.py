"""Shared test-only helpers for building simulator components from Config."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from student_simulator import (
    ApartmentSimulator,
    BuildingSimulator,
    CohortCompositionSimulator,
    LatentEffects,
    NeighborhoodSimulator,
    TotalChildrenSimulator,
)
from student_simulator.config import Config


def make_neighborhood_simulator(config: Config) -> NeighborhoodSimulator:
    """Build a NeighborhoodSimulator from the settings it owns."""
    return NeighborhoodSimulator(config.simulation, config.neighborhood)


def make_building_simulator(config: Config) -> BuildingSimulator:
    """Build a BuildingSimulator from the settings it owns."""
    return BuildingSimulator(config.simulation, config.building, config.transforms)


def make_apartment_simulator(config: Config) -> ApartmentSimulator:
    """Build an ApartmentSimulator from the settings it owns."""
    return ApartmentSimulator(config.simulation, config.building.room_mix)


def make_total_children_simulator(config: Config) -> TotalChildrenSimulator:
    """Build a TotalChildrenSimulator from the settings it owns."""
    return TotalChildrenSimulator(
        config.simulation, config.total_children, config.transforms
    )


def make_cohort_simulator(config: Config) -> CohortCompositionSimulator:
    """Build a CohortCompositionSimulator from the settings it owns."""
    return CohortCompositionSimulator(
        config.simulation,
        config.cohort,
        config.transforms,
        config.total_children,
    )


def with_overrides(config: Config, **section_overrides: dict[str, Any]) -> Config:
    """Return a copy of config with each named section's fields overridden."""
    update = {
        section: getattr(config, section).model_copy(update=fields)
        for section, fields in section_overrides.items()
    }
    return config.model_copy(update=update)


class Stages(NamedTuple):
    """Every intermediate frame of one simulated population."""

    neighborhoods_df: pd.DataFrame
    buildings_df: pd.DataFrame
    apartment_context: pd.DataFrame
    apartments_with_totals: pd.DataFrame
    apartments_with_cohorts: pd.DataFrame
    total_effects: LatentEffects
    cohort_effects: LatentEffects


def simulate_stages(config: Config) -> Stages:
    """Run the five stages in `create_dataset`'s order on one seeded generator.

    The population's output is building-level; tests of the apartment-level
    stages take their rows from here.
    """
    rng = np.random.default_rng(config.simulation.seed)
    neighborhoods_df = make_neighborhood_simulator(config).sample_neighborhoods(
        config.simulation.n_neighborhoods, rng
    )
    buildings_df = make_building_simulator(config).sample_buildings(
        neighborhoods_df, rng
    )
    apartments_df = make_apartment_simulator(config).expand_apartments(
        buildings_df, rng
    )
    apartment_context = ApartmentSimulator.join_context(
        apartments_df, buildings_df, neighborhoods_df
    )
    apartments_with_totals, total_effects = make_total_children_simulator(
        config
    ).sample_total_children(apartment_context, rng)
    apartments_with_cohorts, cohort_effects = make_cohort_simulator(
        config
    ).sample_cohorts(apartments_with_totals, rng)
    return Stages(
        neighborhoods_df,
        buildings_df,
        apartment_context,
        apartments_with_totals,
        apartments_with_cohorts,
        total_effects,
        cohort_effects,
    )
