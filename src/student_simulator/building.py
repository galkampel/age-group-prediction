from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BuildingSettings,
    ConfigError,
    SimulationSettings,
    TransformSettings,
)
from .features import expected_room_probabilities, scale_household_size


class BuildingSimulator:
    """Generate wide building rows with apartment room composition."""

    def __init__(
        self,
        simulation: SimulationSettings,
        settings: BuildingSettings,
        transforms: TransformSettings,
    ) -> None:
        """Extract seed, building, room-mix, and shared feature settings."""
        self._default_seed = simulation.seed
        self._settings = settings
        self._room_mix = settings.room_mix
        self._transforms = transforms

    def generate(
        self,
        neighborhoods: pd.DataFrame,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate buildings for the supplied neighborhoods.

        Args:
            neighborhoods: Public neighborhood rows containing IDs, SES, and
                average household size.
            rng: Optional shared generator. When omitted, a fresh generator is
                created from the configured simulation seed.

        Returns:
            One row per building with neighborhood foreign key, wide room-count
            columns, and total apartment count.

        Raises:
            ConfigError: If configured overrides do not match generated owner
                counts or generation produces no buildings.
        """
        generator = (
            np.random.default_rng(self._default_seed) if rng is None else rng
        )
        # Sample one building count per neighborhood, honoring any configured
        buildings_per_neighborhood = self._sample_building_counts(
            len(neighborhoods), generator
        )
        # Expand building counts into stable IDs and neighborhood foreign keys.
        buildings = self._build_rows(
            neighborhoods,
            buildings_per_neighborhood,
        )
        # Sample apartment totals and room counts for each building, honoring any
        # configured overrides.
        apartment_totals = self._sample_apartment_totals(
            len(buildings), generator
        )
        room_counts = self._sample_room_counts(
            buildings,
            neighborhoods,
            apartment_totals,
            generator,
        )

        for index, room in enumerate(self._room_mix.supported_rooms):
            buildings[f"{room}_rooms"] = room_counts[:, index].astype(int)
        buildings["n_apartments"] = apartment_totals.astype(int)
        return buildings

    def _sample_building_counts(
        self,
        n_neighborhoods: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return one building count per neighborhood, honoring overrides."""
        override = self._settings.buildings_per_neighborhood
        if override:
            if len(override) != n_neighborhoods:
                raise ConfigError(
                    "override length mismatch: "
                    "building.buildings_per_neighborhood must match "
                    "number of neighborhoods"
                )
            return np.asarray(override, dtype=int)

        return self._settings.buildings_per_neighborhood_min + rng.poisson(
            lam=self._settings.buildings_per_neighborhood_rate,
            size=n_neighborhoods,
        )

    def _build_rows(
        self,
        neighborhoods: pd.DataFrame,
        buildings_per_neighborhood: np.ndarray,
    ) -> pd.DataFrame:
        """Expand building counts into stable IDs and neighborhood keys."""
        rows: list[dict[str, str]] = []
        building_id = 1
        neighborhood_rows = neighborhoods.reset_index(drop=True)
        for position, (_, neighborhood) in enumerate(
            neighborhood_rows.iterrows()
        ):
            for _ in range(int(buildings_per_neighborhood[position])):
                rows.append(
                    {
                        "building_id": f"B_{building_id:06d}",
                        "neighborhood_id": str(
                            neighborhood["neighborhood_id"]
                        ),
                    }
                )
                building_id += 1

        if not rows:
            raise ConfigError(
                "generated zero buildings; check building configuration"
            )
        return pd.DataFrame(rows)

    def _sample_apartment_totals(
        self,
        n_buildings: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return apartment totals per building, honoring exact overrides."""
        override = self._settings.apartments_per_building
        if override:
            if len(override) != n_buildings:
                raise ConfigError(
                    "override length mismatch: "
                    "building.apartments_per_building must match number of "
                    "generated buildings"
                )
            return np.asarray(override, dtype=int)

        # Round normal draws before clipping so every building total is an
        # integer within the configured physical bounds.
        totals = np.rint(
            rng.normal(
                loc=self._settings.apartments_per_building_mean,
                scale=self._settings.apartments_per_building_sd,
                size=n_buildings,
            )
        ).astype(int)
        return np.clip(
            totals,
            self._settings.apartments_per_building_min,
            self._settings.apartments_per_building_max,
        )

    def _sample_room_counts(
        self,
        buildings: pd.DataFrame,
        neighborhoods: pd.DataFrame,
        apartment_totals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample integer room counts whose rows equal apartment totals.

        The configured neighborhood tilt defines expected room probabilities.
        A building-specific Dirichlet draw adds composition heterogeneity, and
        a multinomial draw converts that mix into exact integer counts.
        """
        context = buildings.merge(
            neighborhoods[
                ["neighborhood_id", "ses", "avg_household_size"]
            ],
            on="neighborhood_id",
            how="left",
            validate="many_to_one",
        )
        # Scale household size to the model reference before combining it with
        # SES in the room-mix tilt.
        household_size_scaled = scale_household_size(
            context["avg_household_size"].to_numpy(),
            reference=self._transforms.household_size_reference,
            scale=self._transforms.household_size_scale,
        )
        room_config = self._room_mix
        expected_probs = expected_room_probabilities(
            supported_rooms=room_config.supported_rooms,
            base_shares=room_config.base_shares,
            household_size_tilt=room_config.household_size_tilt,
            ses_tilt=room_config.ses_tilt,
            room_distance_divisor=room_config.room_distance_divisor,
            household_size_scaled=household_size_scaled,
            ses=context["ses"].to_numpy(dtype=float),
        )

        room_counts = np.zeros(
            (len(buildings), len(room_config.supported_rooms)),
            dtype=int,
        )
        for index, n_apartments in enumerate(apartment_totals):
            # Concentration controls variation around the neighborhood-level
            # expected mix; multinomial sampling preserves the apartment total.
            alpha = (
                room_config.dirichlet_concentration
                * expected_probs[index]
            )
            building_mix = rng.dirichlet(alpha)
            room_counts[index] = rng.multinomial(
                int(n_apartments), building_mix
            )
        return room_counts
