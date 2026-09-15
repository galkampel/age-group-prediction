from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CohortSettings,
    SimulationSettings,
    TotalChildrenSettings,
    TransformSettings,
)
from .features import (
    center_rooms,
    daycare_saturation,
    scale_household_size,
    scale_median_age,
    stable_row_normalize,
)


def _sample_random_effect(
    entity_ids: pd.Series,
    standard_deviation: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Sample one zero-centered Gaussian effect per unique entity ID."""
    unique_entity_ids = pd.Index(
        pd.unique(entity_ids),
        dtype="object",
        name=entity_ids.name,
    )
    return pd.Series(
        rng.normal(
            loc=0.0,
            scale=standard_deviation,
            size=len(unique_entity_ids),
        ),
        index=unique_entity_ids,
        dtype=float,
    )


class TotalChildrenSimulator:
    """Generate apartment-level total child counts from the NB2 model."""

    def __init__(
        self,
        simulation: SimulationSettings,
        settings: TotalChildrenSettings,
        transforms: TransformSettings,
    ) -> None:
        """Extract total-children settings and shared feature scales."""
        self._default_seed = simulation.seed
        self._settings = settings
        self._transforms = transforms
        self._latent_effects: dict[str, pd.Series] = {}

    def generate(
        self,
        apartment_context: pd.DataFrame,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Sample total child counts for a joined apartment context.

        Args:
            apartment_context: Apartment rows joined to building and
                neighborhood features. It must contain `building_id` and
                `neighborhood_id` for latent-effect ownership.
            rng: Optional shared generator. When omitted, a fresh generator is
                created from the configured simulation seed.

        Returns:
            A copy of `apartment_context` with `n_children_total` added. Latent
            effects remain private and are never returned as columns.
        """
        generator = (
            np.random.default_rng(self._default_seed) if rng is None else rng
        )
        building_effect = _sample_random_effect(
            apartment_context["building_id"],
            self._settings.building_effect_sd,
            rng=generator,
        )
        neighborhood_effect = _sample_random_effect(
            apartment_context["neighborhood_id"],
            self._settings.neighborhood_effect_sd,
            rng=generator,
        )
        log_mu = self.compute_log_mu(
            apartment_context,
            building_effect,
            neighborhood_effect,
        )
        if not np.isfinite(log_mu).all():
            raise ValueError(
                "total_children linear predictor contains non-finite values"
            )

        mu = np.exp(log_mu)
        if not np.isfinite(mu).all() or np.any(mu <= 0):
            raise ValueError(
                "total_children expected counts must be finite and > 0"
            )

        out = apartment_context.copy(deep=True)
        out["n_children_total"] = self._sample_nb2(generator, mu).astype(int)
        self._latent_effects = {
            "building_total_count_effect": building_effect,
            "neighborhood_total_count_effect": neighborhood_effect,
        }
        return out

    def compute_log_mu(
        self,
        apartment_context: pd.DataFrame,
        building_effect: pd.Series,
        neighborhood_effect: pd.Series,
    ) -> np.ndarray:
        """Compute deterministic log expected counts before NB2 sampling."""
        design = self.build_design_matrix(apartment_context)
        log_mu = np.zeros(len(design), dtype=float)
        for feature, coefficient in self._feature_coefficients().items():
            log_mu += coefficient * design[feature].to_numpy(dtype=float)

        building_values = apartment_context["building_id"].map(
            building_effect
        ).to_numpy(dtype=float)
        neighborhood_values = apartment_context["neighborhood_id"].map(
            neighborhood_effect
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(building_values).all()
            or not np.isfinite(neighborhood_values).all()
        ):
            raise ValueError(
                "missing internal random effect for apartment owner"
            )
        return log_mu + building_values + neighborhood_values

    def build_design_matrix(
        self, apartment_context: pd.DataFrame
    ) -> pd.DataFrame:
        """Build observed total-children terms in coefficient-stable order."""
        features = self._build_features(apartment_context)
        design = pd.DataFrame(index=apartment_context.index)
        design["intercept"] = 1.0
        for room in sorted(self._settings.room_log_mean_effects):
            design[f"rooms_{room}"] = (
                apartment_context["rooms"] == room
            ).astype(float)

        ses = apartment_context["ses"].to_numpy(dtype=float)
        design["ses_squared"] = ses**2
        design["household_size_scaled"] = features[
            "household_size_scaled"
        ]
        design["daycare_saturation"] = features["daycare_saturation"]
        design["school_existing"] = (
            apartment_context["school_status"] == "existing"
        ).astype(float)
        design["school_planned"] = (
            apartment_context["school_status"] == "planned"
        ).astype(float)
        design["median_age_scaled"] = features["median_age_scaled"]
        design["rooms_household_interaction"] = (
            features["rooms_centered"] * features["household_size_scaled"]
        )
        return design.loc[:, list(self._feature_coefficients())]

    def latent_effects(self) -> dict[str, pd.Series]:
        """Return named copies of effects from the latest generation call.

        This supports the temporary pipeline recovery adapter while keeping
        latent values out of generated DataFrame columns. Named effects avoid
        positional coupling when an effect is added or removed.

        Raises:
            RuntimeError: If called before `generate` creates effects.
        """
        if not self._latent_effects:
            raise RuntimeError(
                "latent effects are unavailable before generate()"
            )
        return {
            name: effect.copy()
            for name, effect in self._latent_effects.items()
        }

    def _build_features(self, apartment_context: pd.DataFrame) -> pd.DataFrame:
        """Apply configured transforms used by the expected-count formula."""
        features = pd.DataFrame(index=apartment_context.index)
        features["rooms_centered"] = center_rooms(
            apartment_context["rooms"].to_numpy(),
            reference=self._transforms.room_reference,
        )
        features["household_size_scaled"] = scale_household_size(
            apartment_context["avg_household_size"].to_numpy(),
            reference=self._transforms.household_size_reference,
            scale=self._transforms.household_size_scale,
        )
        features["median_age_scaled"] = scale_median_age(
            apartment_context["median_age"].to_numpy(),
            reference=self._transforms.median_age_reference,
            scale=self._transforms.median_age_scale,
        )
        features["daycare_saturation"] = daycare_saturation(
            apartment_context["n_daycares_500m"].to_numpy(),
            saturation_scale=self._settings.daycare_saturation_scale,
        )
        return features

    def _feature_coefficients(self) -> dict[str, float]:
        """Map semantic design features to their configured coefficients."""
        return {
            "intercept": self._settings.intercept,
            **{
                f"rooms_{room}": coefficient
                for room, coefficient in sorted(
                    self._settings.room_log_mean_effects.items()
                )
            },
            "ses_squared": self._settings.ses_quadratic_coef,
            "household_size_scaled": self._settings.household_size_coef,
            "daycare_saturation": self._settings.daycare_saturation_coef,
            "school_existing": self._settings.existing_school_coef,
            "school_planned": self._settings.planned_school_coef,
            "median_age_scaled": self._settings.median_age_coef,
            "rooms_household_interaction": (
                self._settings.rooms_household_interaction_coef
            ),
        }

    def _sample_nb2(
        self, rng: np.random.Generator, mu: np.ndarray
    ) -> np.ndarray:
        """Sample NB2 counts with variance `mu + mu**2 / phi`."""
        phi = self._settings.nb_dispersion_phi
        if phi <= 0:
            raise ValueError("phi must be > 0")
        if np.any(mu <= 0):
            raise ValueError("mu values must be > 0")
        probability = phi / (phi + mu)
        return rng.negative_binomial(n=phi, p=probability)


class CohortCompositionSimulator:
    """Generate apartment-level child counts for three age cohorts."""

    def __init__(
        self,
        simulation: SimulationSettings,
        settings: CohortSettings,
        transforms: TransformSettings,
        total_children: TotalChildrenSettings,
    ) -> None:
        """Extract cohort settings and shared feature scales."""
        self._default_seed = simulation.seed
        self._settings = settings
        self._transforms = transforms
        self._total_children = total_children

    def generate(
        self,
        apartment_context: pd.DataFrame,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Sample cohort counts that sum to each apartment's child total.

        Args:
            apartment_context: Apartment rows joined to building and
                neighborhood features, including `n_children_total`.
            rng: Optional shared generator. When omitted, a fresh generator is
                created from the configured simulation seed.

        Returns:
            A copy of `apartment_context` with kindergarten, elementary, and
            high-school counts. The latent building effect is not returned.
        """
        generator = (
            np.random.default_rng(self._default_seed) if rng is None else rng
        )
        building_effect = _sample_random_effect(
            apartment_context["building_id"],
            self._settings.building_effect_sd,
            rng=generator,
        )
        # Compute lifecycle scores and cohort probabilities for each apartment.
        theta = self.compute_theta(apartment_context, building_effect)
        probabilities = self.cohort_probabilities(theta)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError(
                "cohort probabilities must be finite and non-negative"
            )
        if not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError(
                "cohort probabilities must sum to one per apartment"
            )

        totals = apartment_context["n_children_total"].to_numpy(dtype=int)
        draws = np.zeros((len(apartment_context), 3), dtype=int)
        for index, total in enumerate(totals):
            draws[index, :] = generator.multinomial(
                int(total), probabilities[index]
            )

        out = apartment_context.copy(deep=True)
        out["n_kindergarten"] = draws[:, 0]
        out["n_elementary"] = draws[:, 1]
        out["n_highschool"] = draws[:, 2]
        return out

    def compute_theta(
        self,
        apartment_context: pd.DataFrame,
        building_effect: pd.Series,
    ) -> np.ndarray:
        """Compute deterministic lifecycle scores before cohort sampling."""
        features = self._build_lifecycle_features(apartment_context)
        theta = np.zeros(len(features), dtype=float)
        for feature, coefficient in self._feature_coefficients().items():
            theta += coefficient * features[feature].to_numpy(dtype=float)

        effect_values = apartment_context["building_id"].map(
            building_effect
        ).to_numpy(dtype=float)
        if not np.isfinite(effect_values).all():
            raise ValueError(
                "missing internal cohort effect for apartment building"
            )
        return theta + effect_values

    def cohort_probabilities(self, theta: np.ndarray) -> np.ndarray:
        """Map lifecycle scores to probabilities for the three age cohorts."""
        reference_shares = self._settings.reference_shares
        logits = np.column_stack(
            (
                np.log(reference_shares["kindergarten"])
                + self._settings.kindergarten_lifecycle_slope * theta,
                np.full_like(
                    theta,
                    np.log(reference_shares["elementary"]),
                    dtype=float,
                ),
                np.log(reference_shares["highschool"])
                + self._settings.highschool_lifecycle_slope * theta,
            )
        )
        return stable_row_normalize(logits)

    def _build_lifecycle_features(
        self, apartment_context: pd.DataFrame
    ) -> pd.DataFrame:
        """Build lifecycle features from observed apartment context."""
        rooms_centered = center_rooms(
            apartment_context["rooms"].to_numpy(),
            reference=self._transforms.room_reference,
        )
        household_size_scaled = scale_household_size(
            apartment_context["avg_household_size"].to_numpy(),
            reference=self._transforms.household_size_reference,
            scale=self._transforms.household_size_scale,
        )
        median_age_scaled = scale_median_age(
            apartment_context["median_age"].to_numpy(),
            reference=self._transforms.median_age_reference,
            scale=self._transforms.median_age_scale,
        )
        daycare = daycare_saturation(
            apartment_context["n_daycares_500m"].to_numpy(),
            saturation_scale=self._total_children.daycare_saturation_scale,
        )
        features = pd.DataFrame(index=apartment_context.index)
        features["rooms_centered"] = rooms_centered
        features["daycare_saturation"] = daycare
        features["median_age_scaled"] = median_age_scaled
        features["household_size_scaled"] = household_size_scaled
        features["rooms_median_age_positive_interaction"] = (
            np.maximum(rooms_centered, 0.0)
            * np.maximum(median_age_scaled, 0.0)
        )
        return features

    def _feature_coefficients(self) -> dict[str, float]:
        """Map lifecycle features to their configured coefficients."""
        return {
            "rooms_centered": self._settings.room_coef,
            "daycare_saturation": self._settings.daycare_coef,
            "median_age_scaled": self._settings.median_age_coef,
            "household_size_scaled": self._settings.household_size_coef,
            "rooms_median_age_positive_interaction": (
                self._settings.room_age_positive_interaction_coef
            ),
        }

