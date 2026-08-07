import numpy as np
import pandas as pd


class SyntheticDataSimulator:
    """Class-based synthetic data simulator for neighborhood and apartment data."""

    def __init__(
        self,
        seed: int = 42,
        avg_family_size_min: float = 1.5,
        avg_family_size_max: float = 4.5,
        schools_min: int = 0,
        schools_max: int = 5,
        parks_min: int = 0,
        parks_max: int = 10,
        room_min: int = 1,
        room_max: int = 10,
        surface_noise_loc: float = 10.0,
        surface_noise_scale: float = 15.0,
        surface_min: float = 15.0,
        surface_max: float = 300.0,
        kindergarten_lambda: float = 1.2,
        elementary_school_factor: float = 0.5,
        elementary_min_lambda: float = 0.5,
        highschool_school_factor: float = 0.3,
        highschool_min_lambda: float = 0.3,
        apartment_dist_alpha_min: float = 2.0,
        apartment_dist_alpha_max: float = 10.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)

        # Statistical parameters.
        self.avg_family_size_min = avg_family_size_min
        self.avg_family_size_max = avg_family_size_max
        self.schools_min = schools_min
        self.schools_max = schools_max
        self.parks_min = parks_min
        self.parks_max = parks_max
        self.room_min = room_min
        self.room_max = room_max
        self.surface_noise_loc = surface_noise_loc
        self.surface_noise_scale = surface_noise_scale
        self.surface_min = surface_min
        self.surface_max = surface_max
        self.kindergarten_lambda = kindergarten_lambda
        self.elementary_school_factor = elementary_school_factor
        self.elementary_min_lambda = elementary_min_lambda
        self.highschool_school_factor = highschool_school_factor
        self.highschool_min_lambda = highschool_min_lambda
        self.apartment_dist_alpha_min = apartment_dist_alpha_min
        self.apartment_dist_alpha_max = apartment_dist_alpha_max

    def create_neighborhood_data(self, num_neighborhoods: int) -> pd.DataFrame:
        """Create neighborhood-level data for the requested number of neighborhoods."""
        neighborhood_ids = [f"N_{i+1:02d}" for i in range(num_neighborhoods)]

        avg_family_sizes = self.rng.uniform(
            self.avg_family_size_min,
            self.avg_family_size_max,
            size=num_neighborhoods,
        )
        cnt_schools = self.rng.integers(
            self.schools_min,
            self.schools_max + 1,
            size=num_neighborhoods,
        )
        cnt_parks = self.rng.integers(
            self.parks_min,
            self.parks_max + 1,
            size=num_neighborhoods,
        )

        return pd.DataFrame(
            {
                "neighborhood": neighborhood_ids,
                "avg_family_size": avg_family_sizes,
                "cnt_schools": cnt_schools,
                "cnt_parks": cnt_parks,
            }
        )

    def create_apartment_data(
        self, df_neighborhoods: pd.DataFrame, num_apartments: int
    ) -> pd.DataFrame:
        """Create apartment-level data based on neighborhood-level features."""
        neighborhood_ids = df_neighborhoods["neighborhood"].tolist()
        num_neighborhoods = len(neighborhood_ids)

        alpha = self.rng.uniform(
            self.apartment_dist_alpha_min,
            self.apartment_dist_alpha_max,
            size=num_neighborhoods,
        )
        weights = self.rng.dirichlet(alpha)
        apartments_per_neighborhood = self.rng.multinomial(num_apartments, weights)

        apartment_records: list[dict[str, int | float | str]] = []

        for i, neighborhood_id in enumerate(neighborhood_ids):
            count = apartments_per_neighborhood[i]
            if count == 0:
                continue

            fam_size = float(df_neighborhoods.iloc[i]["avg_family_size"])
            n_schools = int(df_neighborhoods.iloc[i]["cnt_schools"])

            cnt_rooms = self.rng.poisson(lam=fam_size, size=count)
            cnt_rooms = np.clip(cnt_rooms, self.room_min, self.room_max)

            base_surface = cnt_rooms * 20.0
            surface = base_surface + self.rng.normal(
                loc=self.surface_noise_loc,
                scale=self.surface_noise_scale,
                size=count,
            )
            surface = np.clip(surface, self.surface_min, self.surface_max)

            cnt_kindergarten = self.rng.poisson(
                lam=self.kindergarten_lambda,
                size=count,
            )
            ele_lam = max(self.elementary_min_lambda, n_schools * self.elementary_school_factor)
            high_lam = max(self.highschool_min_lambda, n_schools * self.highschool_school_factor)

            cnt_elementary = self.rng.poisson(lam=ele_lam, size=count)
            cnt_highschool = self.rng.poisson(lam=high_lam, size=count)

            for idx in range(count):
                apartment_records.append(
                    {
                        "cnt_rooms": int(cnt_rooms[idx]),
                        "surface": float(surface[idx]),
                        "neighborhood": neighborhood_id,
                        "cnt_kindergarten": int(cnt_kindergarten[idx]),
                        "cnt_elementary": int(cnt_elementary[idx]),
                        "cnt_highschool": int(cnt_highschool[idx]),
                    }
                )

        return pd.DataFrame(apartment_records)

    def create_data(
        self, num_neighborhoods: int = 20, num_apartments: int = 10000
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create neighborhood and apartment data sequentially."""
        df_neighborhoods = self.create_neighborhood_data(num_neighborhoods)
        df_apartments = self.create_apartment_data(df_neighborhoods, num_apartments)
        return df_neighborhoods, df_apartments
