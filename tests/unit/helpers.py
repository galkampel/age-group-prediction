"""Shared test-only helpers for building simulator components from Config."""

from __future__ import annotations

from typing import Any

from student_simulator import (
    ApartmentSimulator,
    BuildingSimulator,
    CohortCompositionSimulator,
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
