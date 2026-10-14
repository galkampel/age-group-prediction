from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RoomMixSettings, SimulationSettings


class ApartmentSimulator:
    """Generate apartment rows and apartment-level derived views."""

    def __init__(
        self,
        simulation: SimulationSettings,
        room_mix: RoomMixSettings,
    ) -> None:
        """Extract seed and room settings owned by this component."""
        self._default_seed = simulation.seed
        self._supported_rooms = room_mix.supported_rooms

    def expand_apartments(
        self,
        buildings_df: pd.DataFrame,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Expand wide building room counts into canonical apartment rows.

        Draws one permutation per building, which orders its apartments.

        Args:
            buildings_df: One row per building with `{room}_rooms` columns and
                `building_id`.
            rng: Optional shared generator. When omitted, a fresh generator is
                created from the configured simulation seed.

        Returns:
            Apartment rows with `apartment_id`, `building_id`, and `rooms`.
        """
        generator = (
            np.random.default_rng(self._default_seed) if rng is None else rng
        )

        apartment_rows: list[dict[str, int | str]] = []
        apartment_id_counter = 1
        for _, building in buildings_df.iterrows():
            rooms = self._expand_building_rooms(building)
            if rooms:
                rooms_array = np.asarray(rooms, dtype=int)
                # Tie apartment room ordering to the shared run RNG.
                rooms_array = rooms_array[generator.permutation(len(rooms_array))]
            else:
                rooms_array = np.asarray([], dtype=int)

            for room in rooms_array:
                apartment_rows.append(
                    {
                        "apartment_id": f"A_{apartment_id_counter:08d}",
                        "building_id": str(building["building_id"]),
                        "rooms": int(room),
                    }
                )
                apartment_id_counter += 1

        return pd.DataFrame(apartment_rows)

    @staticmethod
    def join_context(
        apartments_df: pd.DataFrame,
        buildings_df: pd.DataFrame,
        neighborhoods_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join apartments with building and neighborhood context columns.

        This method uses explicit merge keys and cardinality validation to keep
        relationship assumptions observable.
        """
        merged = apartments_df.merge(
            buildings_df[["building_id", "neighborhood_id"]],
            on="building_id",
            how="left",
            validate="many_to_one",
        )
        return merged.merge(
            neighborhoods_df,
            on="neighborhood_id",
            how="left",
            validate="many_to_one",
        )

    @staticmethod
    def to_cohort_long(apartments_df: pd.DataFrame) -> pd.DataFrame:
        """Return long-format apartment cohort counts for analysis workflows.

        Raises:
            ValueError: If one or more required cohort columns are missing.
        """
        required = {
            "apartment_id",
            "building_id",
            "rooms",
            "n_children_total",
            "n_kindergarten",
            "n_elementary",
            "n_highschool",
        }
        missing = sorted(required - set(apartments_df.columns))
        if missing:
            missing_columns = ", ".join(missing)
            raise ValueError(
                "apartments missing required columns for cohort long view: "
                f"{missing_columns}"
            )

        cohort_map = {
            "n_kindergarten": "kindergarten",
            "n_elementary": "elementary",
            "n_highschool": "highschool",
        }
        long_df = apartments_df.melt(
            id_vars=["apartment_id", "building_id", "rooms", "n_children_total"],
            value_vars=list(cohort_map.keys()),
            var_name="cohort_target",
            value_name="n_children",
        )
        long_df["cohort"] = long_df["cohort_target"].map(cohort_map)
        return long_df.drop(columns=["cohort_target"])

    def _expand_building_rooms(self, building: pd.Series) -> list[int]:
        """Expand `{room}_rooms` counts for one building into a room list."""
        room_labels: list[int] = []
        for room in self._supported_rooms:
            count = int(building[f"{room}_rooms"])
            if count > 0:
                room_labels.extend([int(room)] * count)
        return room_labels