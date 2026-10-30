from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .apartment import ApartmentSimulator
from .building import BuildingSimulator
from .config import Config
from .neighborhood import NeighborhoodSimulator
from .outcomes import (
    COHORT_COLUMNS,
    CohortCompositionSimulator,
    TotalChildrenSimulator,
)


@dataclass(frozen=True)
class SimulationResult:
    """The dataset of one simulated population and the latent effects behind it.

    ``dataset`` has one row per building: its neighborhood context, room
    counts and the three cohort counts (the building's total is their sum,
    so it is not a column). The effects are one frame per owner level
    (lme4's ``ranef`` layout), indexed by the owner id so they join to
    ``dataset``: ``building_effects`` (``building_id``; columns
    ``total_count``, ``cohort``) and ``neighborhood_effects``
    (``neighborhood_id``; column ``total_count``). They exist for
    coefficient-recovery checks and are never columns of ``dataset``. A
    building without apartments has no effect row.
    """

    dataset: pd.DataFrame
    building_effects: pd.DataFrame = field(repr=False)
    neighborhood_effects: pd.DataFrame = field(repr=False)


def aggregate_to_buildings(
    apartments_df: pd.DataFrame,
    buildings_df: pd.DataFrame,
    neighborhoods_df: pd.DataFrame,
) -> pd.DataFrame:
    """Sum the apartments' child counts into one row per building.

    Columns: `building_id`, the neighborhood columns, the building's room
    counts and `n_apartments`, then the cohort counts. The total is their sum
    and is left out. The accounting identities (room counts and apartment rows
    against `n_apartments`, cohorts against each apartment's total) are
    properties of the components and are tested there.
    """
    apartment_sums = (
        apartments_df.groupby("building_id")[list(COHORT_COLUMNS)]
        .sum()
        # A building without apartments has no group; its counts are zero.
        .reindex(buildings_df["building_id"], fill_value=0)
    )
    df = buildings_df.merge(
        neighborhoods_df, on="neighborhood_id", how="left", validate="many_to_one"
    ).merge(apartment_sums, on="building_id", validate="one_to_one")
    rooms_and_n_apartments = buildings_df.columns.drop(
        ["building_id", "neighborhood_id"]
    )
    return df[
        [
            "building_id",
            *neighborhoods_df.columns,
            *rooms_and_n_apartments,
            *apartment_sums.columns,
        ]
    ]


class StudentPopulationSimulator:
    """The flow from neighborhoods to the dataset, one generator for every draw."""

    def __init__(
        self,
        neighborhood_simulator: NeighborhoodSimulator,
        building_simulator: BuildingSimulator,
        apartment_simulator: ApartmentSimulator,
        total_children_simulator: TotalChildrenSimulator,
        cohort_composition_simulator: CohortCompositionSimulator,
        *,
        n_neighborhoods: int,
        seed: int,
    ) -> None:
        self.neighborhood_simulator = neighborhood_simulator
        self.building_simulator = building_simulator
        self.apartment_simulator = apartment_simulator
        self.total_children_simulator = total_children_simulator
        self.cohort_composition_simulator = cohort_composition_simulator
        self.n_neighborhoods = n_neighborhoods
        self.seed = seed

    @classmethod
    def from_config(cls, config: Config) -> StudentPopulationSimulator:
        """Build the five components, the size and the seed from one config."""
        simulation = config.simulation
        return cls(
            NeighborhoodSimulator(simulation, config.neighborhood),
            BuildingSimulator(simulation, config.building, config.transforms),
            ApartmentSimulator(simulation, config.building.room_mix),
            TotalChildrenSimulator(
                simulation, config.total_children, config.transforms
            ),
            CohortCompositionSimulator(
                simulation, config.cohort, config.transforms, config.total_children
            ),
            n_neighborhoods=simulation.n_neighborhoods,
            seed=simulation.seed,
        )

    def simulate(self, rng: np.random.Generator | None = None) -> SimulationResult:
        """Sample one population, every stage drawing from one generator.

        Args:
            rng: Optional generator. When omitted, a fresh generator is created
                from `seed`, so repeated calls return the same population.
        """
        generator = np.random.default_rng(self.seed) if rng is None else rng
        # 1. Context: one row per neighborhood, then the buildings inside them,
        #    then one row per apartment with its room count.
        neighborhoods_df = self.neighborhood_simulator.sample_neighborhoods(
            self.n_neighborhoods, generator
        )
        buildings_df = self.building_simulator.sample_buildings(
            neighborhoods_df, generator
        )
        apartments_df = self.apartment_simulator.expand_apartments(
            buildings_df, generator
        )
        # 2. Outcomes per apartment: the total first, then its split into
        #    cohorts. Both stages read the apartment joined to its building and
        #    neighborhood, and each returns that frame with its counts added.
        apartment_context = ApartmentSimulator.join_context(
            apartments_df, buildings_df, neighborhoods_df
        )
        apartments_with_totals, total_effects = (
            self.total_children_simulator.sample_total_children(
                apartment_context, generator
            )
        )
        apartments_with_cohorts, cohort_effects = (
            self.cohort_composition_simulator.sample_cohorts(
                apartments_with_totals, generator
            )
        )
        # 3. The dataset: apartments summed into their buildings.
        dataset = aggregate_to_buildings(
            apartments_with_cohorts, buildings_df, neighborhoods_df
        )
        building_effects = pd.DataFrame(
            {
                "total_count": total_effects["building_total_count_effect"],
                "cohort": cohort_effects["building_cohort_effect"],
            }
        )
        neighborhood_effects = total_effects[
            "neighborhood_total_count_effect"
        ].to_frame("total_count")
        return SimulationResult(
            dataset=dataset,
            building_effects=building_effects,
            neighborhood_effects=neighborhood_effects,
        )
