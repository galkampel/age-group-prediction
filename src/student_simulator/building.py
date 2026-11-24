from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import (
    BuildingSettings,
    ConfigError,
    SimulationSettings,
    TransformSettings,
)
from .features import (
    expected_room_probabilities,
    scale_household_size,
    scale_median_age,
)


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

    def sample_buildings(
        self,
        neighborhoods_df: pd.DataFrame,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Sample the buildings of each neighborhood and their room counts.

        Draws, in order: a building count per neighborhood; all the
        neighborhoods' hidden tower propensities, then all their hidden
        log-size textures; one uniform per building for its hidden type
        (texture or tower), then one normal per building for its log size;
        then per building a Dirichlet room mix (tilted by SES, household size
        and the building's size) and a multinomial split of the total into
        room counts. Configured overrides replace the counts
        and, for the apartment totals, all of the type and size draws.

        Args:
            neighborhoods_df: Public neighborhood rows containing IDs, SES,
                average household size, median age and school status.
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
        # override.
        buildings_per_neighborhood = self._sample_building_counts(
            len(neighborhoods_df), generator
        )
        # Expand building counts into stable IDs and neighborhood foreign keys.
        buildings_df = self._build_rows(
            neighborhoods_df,
            buildings_per_neighborhood,
        )
        # One row of neighborhood context per building, shared by both stages.
        context = buildings_df.merge(
            neighborhoods_df[
                [
                    "neighborhood_id",
                    "ses",
                    "avg_household_size",
                    "median_age",
                    "school_status",
                ]
            ],
            on="neighborhood_id",
            how="left",
            validate="many_to_one",
        )
        # Sample apartment totals and room counts for each building, honoring any
        # configured overrides.
        apartment_totals = self._sample_apartment_totals(
            context, buildings_per_neighborhood, generator
        )
        room_counts = self._sample_room_counts(
            context,
            apartment_totals,
            generator,
        )

        for index, room in enumerate(self._room_mix.supported_rooms):
            buildings_df[f"{room}_rooms"] = room_counts[:, index].astype(int)
        buildings_df["n_apartments"] = apartment_totals.astype(int)
        return buildings_df

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
        neighborhoods_df: pd.DataFrame,
        buildings_per_neighborhood: np.ndarray,
    ) -> pd.DataFrame:
        """Expand building counts into stable IDs and neighborhood keys."""
        rows: list[dict[str, str]] = []
        building_id = 1
        neighborhood_rows = neighborhoods_df.reset_index(drop=True)
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
        context: pd.DataFrame,
        buildings_per_neighborhood: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return apartment totals per building.

        An override replaces every type and size draw. The hidden type is
        not returned.
        """
        n_buildings = len(context)
        override = self._settings.apartments_per_building
        if override:
            if len(override) != n_buildings:
                raise ConfigError(
                    "override length mismatch: "
                    "building.apartments_per_building must match number of "
                    "generated buildings"
                )
            return np.asarray(override, dtype=int)

        settings = self._settings
        # Context rows run neighborhood by neighborhood (see _build_rows), so
        # repeating each neighborhood's draw by its count aligns it.
        n_neighborhoods = len(buildings_per_neighborhood)
        tower_propensity = np.repeat(
            rng.normal(0.0, settings.tower_propensity_sd, n_neighborhoods),
            buildings_per_neighborhood,
        )
        texture = np.repeat(
            rng.normal(
                0.0, settings.apartments_neighborhood_log_sd, n_neighborhoods
            ),
            buildings_per_neighborhood,
        )
        median_age_scaled = scale_median_age(
            context["median_age"].to_numpy(),
            reference=self._transforms.median_age_reference,
            scale=self._transforms.median_age_scale,
        )
        tower_log_odds = (
            settings.tower_log_odds_intercept
            + settings.tower_ses_coef * context["ses"].to_numpy()
            + settings.tower_median_age_coef * median_age_scaled
            + settings.tower_planned_school_coef
            * (context["school_status"] == "planned").to_numpy()
            + tower_propensity
        )
        is_tower = rng.random(n_buildings) < expit(tower_log_odds)
        median = np.where(
            is_tower,
            settings.tower_apartments_median,
            settings.texture_apartments_median,
        )
        building_noise = rng.normal(
            0.0, settings.apartments_building_log_sd, n_buildings
        )
        # Round before clipping so every building total is an integer within
        # the configured physical bounds.
        totals = np.rint(median * np.exp(texture + building_noise)).astype(int)
        return np.clip(
            totals,
            settings.apartments_per_building_min,
            settings.apartments_per_building_max,
        )

    def _sample_room_counts(
        self,
        context: pd.DataFrame,
        apartment_totals: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample integer room counts whose rows equal apartment totals.

        The configured tilt (SES, household size and the building's size
        relative to the texture median) defines expected room probabilities.
        A building-specific Dirichlet draw adds composition heterogeneity, and
        a multinomial draw converts that mix into exact integer counts.
        """
        # Scale household size to the model reference before combining it with
        # SES and the building's size in the room-mix tilt.
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
            building_size_tilt=room_config.building_size_tilt,
            room_distance_divisor=room_config.room_distance_divisor,
            household_size_scaled=household_size_scaled,
            ses=context["ses"].to_numpy(dtype=float),
            log_relative_building_size=np.log(
                apartment_totals / self._settings.texture_apartments_median
            ),
        )

        room_counts = np.zeros(
            (len(context), len(room_config.supported_rooms)),
            dtype=int,
        )
        for index, n_apartments in enumerate(apartment_totals):
            # Concentration controls variation around the building's expected
            # mix; multinomial sampling preserves the apartment total.
            alpha = (
                room_config.dirichlet_concentration
                * expected_probs[index]
            )
            building_mix = rng.dirichlet(alpha)
            room_counts[index] = rng.multinomial(
                int(n_apartments), building_mix
            )
        return room_counts
