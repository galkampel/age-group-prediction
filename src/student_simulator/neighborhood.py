from __future__ import annotations

import numpy as np
import pandas as pd

from .config import NeighborhoodSettings, SimulationSettings


class NeighborhoodSimulator:
    """Generate configured neighborhood context as a public DataFrame."""

    def __init__(
        self,
        simulation: SimulationSettings,
        settings: NeighborhoodSettings,
    ) -> None:
        """Extract the seed, row-count default, and neighborhood settings."""
        self._default_seed = simulation.seed
        self._default_n_neighborhoods = simulation.n_neighborhoods
        self._settings = settings

    def generate(
        self,
        n_neighborhoods: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate one public row per neighborhood.

        Args:
            n_neighborhoods: Number of rows to generate. Uses the configured
                simulation default when omitted.
            rng: Optional shared generator. When omitted, a fresh generator is
                created from the configured simulation seed.

        Returns:
            Neighborhood features with stable IDs and no latent model effects.

        Raises:
            ValueError: If `n_neighborhoods` is not positive.
        """
        n = (
            self._default_n_neighborhoods
            if n_neighborhoods is None
            else n_neighborhoods
        )
        if n <= 0:
            raise ValueError("n_neighborhoods must be greater than zero")
        generator = (
            np.random.default_rng(self._default_seed) if rng is None else rng
        )

        neighborhood_ids = self._generate_ids(n)
        ses = self._sample_ses(n, generator)
        median_age = self._sample_median_age(n, generator)
        household_size = self._sample_household_size(median_age, generator)
        n_daycares = self._sample_daycares(
            ses=ses,
            median_age=median_age,
            rng=generator,
        )
        school_status = self._sample_school_status(n, generator)

        return pd.DataFrame(
            {
                "neighborhood_id": neighborhood_ids,
                "ses": ses,
                "avg_household_size": household_size,
                "median_age": median_age,
                "n_daycares_500m": n_daycares,
                "school_status": school_status,
            }
        )

    def _generate_ids(self, n_neighborhoods: int) -> list[str]:
        """Return sequential, zero-padded IDs for generated neighborhoods."""
        return [f"N_{idx:03d}" for idx in range(1, n_neighborhoods + 1)]

    def _sample_ses(
        self,
        n_neighborhoods: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw SES values from the configured clipped normal distribution."""
        ncfg = self._settings
        return np.clip(
            rng.normal(
                loc=ncfg.ses_mean,
                scale=ncfg.ses_sd,
                size=n_neighborhoods,
            ),
            ncfg.ses_min,
            ncfg.ses_max,
        )

    def _sample_median_age(
        self,
        n_neighborhoods: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw median ages from the configured clipped normal distribution."""
        ncfg = self._settings
        return np.clip(
            rng.normal(
                loc=ncfg.median_age_mean,
                scale=ncfg.median_age_sd,
                size=n_neighborhoods,
            ),
            ncfg.median_age_min,
            ncfg.median_age_max,
        )

    def _sample_household_size(
        self,
        median_age: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw age-dependent household sizes within configured bounds."""
        ncfg = self._settings
        hh_noise = rng.normal(
            loc=0.0,
            scale=ncfg.household_size_noise_sd,
            size=len(median_age),
        )
        # Center age at its configured mean so the baseline remains the
        # expected household size for a reference-age neighborhood.
        return np.clip(
            ncfg.household_size_baseline
            + ncfg.household_size_age_slope
            * (median_age - ncfg.median_age_mean)
            + hh_noise,
            ncfg.household_size_min,
            ncfg.household_size_max,
        )

    def _sample_daycares(
        self,
        ses: np.ndarray,
        median_age: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw daycare counts from the configured log-linear Poisson model."""
        ncfg = self._settings
        # Age is centered so the intercept is the reference-age log rate;
        # exponentiation converts that predictor to a positive Poisson rate.
        daycare_log_mu = (
            ncfg.daycare_log_intercept
            + ncfg.daycare_ses_coef * ses
            + ncfg.daycare_age_coef
            * (median_age - ncfg.median_age_mean)
        )
        daycare_lambda = np.exp(daycare_log_mu)
        return rng.poisson(lam=daycare_lambda, size=len(ses)).astype(int)

    def _sample_school_status(
        self,
        n_neighborhoods: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw school status from configured categorical probabilities."""
        ncfg = self._settings
        status_probs = ncfg.school_status_probs
        school_labels = np.array(["existing", "planned", "none"], dtype=object)
        school_probs = np.array(
            [
                status_probs["existing"],
                status_probs["planned"],
                status_probs["none"],
            ],
            dtype=float,
        )
        return rng.choice(
            school_labels,
            size=n_neighborhoods,
            p=school_probs,
        )
