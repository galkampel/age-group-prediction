from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .apartment import ApartmentSimulator
from .building import BuildingSimulator
from .config import Config
from .neighborhood import NeighborhoodSimulator
from .outcomes import CohortCompositionSimulator, TotalChildrenSimulator


_FINAL_BUILDING_COLUMNS: tuple[str, ...] = (
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
)
_ROOM_COUNT_COLUMNS: tuple[str, ...] = (
    "3_rooms",
    "4_rooms",
    "5_rooms",
    "6_rooms",
)
_COHORT_COUNT_COLUMNS: tuple[str, ...] = (
    "n_kindergarten",
    "n_elementary",
    "n_highschool",
)
_OUTCOME_COUNT_COLUMNS: tuple[str, ...] = (
    "n_children_total",
    *_COHORT_COUNT_COLUMNS,
)
_NEIGHBORHOOD_CONTEXT_COLUMNS: tuple[str, ...] = (
    "ses",
    "avg_household_size",
    "median_age",
    "n_daycares_500m",
    "school_status",
)


@dataclass(frozen=True)
class SimulationResult:
    """Named tables from one full simulation run for research workflows.

    Random effects support test-only coefficient recovery and never become
    columns in generated tables or the final building-level output.
    """

    neighborhoods: pd.DataFrame
    buildings: pd.DataFrame
    apartments: pd.DataFrame
    final_df: pd.DataFrame
    latent_effects: dict[str, pd.Series] = field(repr=False)


def _copy_result(result: SimulationResult) -> SimulationResult:
    """Return a deep copy that callers may inspect and mutate safely."""
    return SimulationResult(
        neighborhoods=result.neighborhoods.copy(deep=True),
        buildings=result.buildings.copy(deep=True),
        apartments=result.apartments.copy(deep=True),
        final_df=result.final_df.copy(deep=True),
        latent_effects={
            name: effect.copy(deep=True)
            for name, effect in result.latent_effects.items()
        },
    )


def _to_building_level(
    apartments: pd.DataFrame,
    buildings: pd.DataFrame,
    neighborhoods: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per building with context and aggregated outcomes."""
    grouped = (
        apartments.groupby("building_id", as_index=False)
        .agg(
            apartment_rows=("apartment_id", "size"),
            n_kindergarten=("n_kindergarten", "sum"),
            n_elementary=("n_elementary", "sum"),
            n_highschool=("n_highschool", "sum"),
            n_children_total=("n_children_total", "sum"),
        )
        .sort_values("building_id")
        .reset_index(drop=True)
    )
    merged = buildings.merge(
        grouped,
        on="building_id",
        how="left",
        validate="one_to_one",
    )
    count_columns = ("apartment_rows", *_OUTCOME_COUNT_COLUMNS)
    for column in count_columns:
        merged[column] = merged[column].fillna(0).astype(int)

    if (merged["apartment_rows"] != merged["n_apartments"]).any():
        raise ValueError("apartment row counts do not match n_apartments")
    room_totals = merged[list(_ROOM_COUNT_COLUMNS)].sum(axis=1)
    if (room_totals != merged["n_apartments"]).any():
        raise ValueError("room totals do not equal n_apartments")
    cohort_sum = merged[list(_COHORT_COUNT_COLUMNS)].sum(axis=1)
    if not np.array_equal(
        cohort_sum.to_numpy(),
        merged["n_children_total"].to_numpy(),
    ):
        raise ValueError("cohort totals do not equal n_children_total")

    with_neighborhood = merged.merge(
        neighborhoods[
            [
                "neighborhood_id",
                *_NEIGHBORHOOD_CONTEXT_COLUMNS,
            ]
        ],
        on="neighborhood_id",
        how="left",
        validate="many_to_one",
    )
    if (
        with_neighborhood[list(_NEIGHBORHOOD_CONTEXT_COLUMNS)]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("building rows are missing neighborhood context")
    if not with_neighborhood["building_id"].is_unique:
        raise ValueError(
            "building aggregation must produce one row per building"
        )

    with_neighborhood["n_apartments"] = with_neighborhood["apartment_rows"]
    out = with_neighborhood.drop(columns=["apartment_rows"])
    out = out.loc[:, _FINAL_BUILDING_COLUMNS]
    return out.set_index("building_id", drop=False).sort_index().reset_index(
        drop=True
    )


class StudentPopulationSimulator:
    """Generate a complete student population with one shared run-level RNG."""

    def __init__(self, config: Config) -> None:
        """Store the complete typed generation configuration."""
        self.config = config
        self._last_result: SimulationResult | None = None

    def run(
        self,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate one population run and return its building-level table.

        1. Generate neighborhoods, buildings, and apartments with one RNG.
        2. Add total child counts, then rebuild apartment context for cohorts.
        3. Allocate cohorts, aggregate apartments to buildings, and retain
           details.

        Args:
            rng: Optional shared generator. When omitted, a fresh generator is
                created from the configured simulation seed.
        """
        generator = (
            np.random.default_rng(self.config.simulation.seed)
            if rng is None
            else rng
        )
        neighborhoods = NeighborhoodSimulator(
            self.config.simulation,
            self.config.neighborhood,
        ).generate(
            self.config.simulation.n_neighborhoods,
            rng=generator,
        )
        buildings = BuildingSimulator(
            self.config.simulation,
            self.config.building,
            self.config.transforms,
        ).generate(
            neighborhoods,
            rng=generator,
        )
        apartments = ApartmentSimulator(
            self.config.simulation,
            self.config.building.room_mix,
        ).generate(
            buildings,
            rng=generator,
        )
        context = ApartmentSimulator.with_context(
            apartments,
            buildings,
            neighborhoods,
        )
        total_children = TotalChildrenSimulator(
            self.config.simulation,
            self.config.total_children,
            self.config.transforms,
        )
        generated_totals = total_children.generate(context, rng=generator)
        apartments_with_totals = apartments.copy(deep=True)
        apartments_with_totals["n_children_total"] = generated_totals[
            "n_children_total"
        ].to_numpy()
        cohort_context = ApartmentSimulator.with_context(
            apartments_with_totals,
            buildings,
            neighborhoods,
        )
        generated_cohorts = CohortCompositionSimulator(
            self.config.simulation,
            self.config.cohort,
            self.config.transforms,
            self.config.total_children,
        ).generate(
            cohort_context,
            rng=generator,
        )
        apartments_with_cohorts = apartments_with_totals.copy(deep=True)
        for column in _COHORT_COUNT_COLUMNS:
            apartments_with_cohorts[column] = generated_cohorts[
                column
            ].to_numpy()
        final_df = _to_building_level(
            apartments_with_cohorts,
            buildings,
            neighborhoods,
        )
        result = SimulationResult(
            neighborhoods=neighborhoods.copy(deep=True),
            buildings=buildings.copy(deep=True),
            apartments=apartments_with_cohorts.copy(deep=True),
            final_df=final_df.copy(deep=True),
            latent_effects=total_children.latent_effects(),
        )
        self._last_result = result
        return result.final_df.copy(deep=True)

    @property
    def last_result(self) -> SimulationResult | None:
        """Return safely copied artifacts from the latest successful run."""
        if self._last_result is None:
            return None
        return _copy_result(self._last_result)
