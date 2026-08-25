from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping
import tomllib

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


SCHEMA_VERSION = "stage1-v1"


class ConfigError(ValueError):
    """Raised when configuration data fails validation."""


class _Settings(BaseModel):
    """Strict, immutable base for all validated settings sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SimulationSettings(_Settings):
    """Global controls for one population-generation run."""

    schema_version: Literal["stage1-v1"] = SCHEMA_VERSION
    seed: int = Field(ge=0)
    n_neighborhoods: int = Field(gt=0)


class TransformSettings(_Settings):
    """Shared feature centering and scaling parameters."""

    household_size_reference: float
    household_size_scale: float = Field(gt=0)
    median_age_reference: float
    median_age_scale: float = Field(gt=0)
    room_reference: float


class NeighborhoodSettings(_Settings):
    """Neighborhood context distribution parameters."""

    ses_mean: float
    ses_sd: float = Field(gt=0)
    ses_min: float
    ses_max: float
    median_age_mean: float
    median_age_sd: float = Field(gt=0)
    median_age_min: float
    median_age_max: float
    household_size_baseline: float
    household_size_age_slope: float
    household_size_noise_sd: float = Field(gt=0)
    household_size_min: float
    household_size_max: float
    daycare_log_intercept: float
    daycare_ses_coef: float
    daycare_age_coef: float
    school_status_probs: dict[str, float]

    @model_validator(mode="after")
    def validate_ranges_and_probabilities(self) -> NeighborhoodSettings:
        _validate_min_max(self.ses_min, self.ses_max, "neighborhood.ses")
        _validate_min_max(
            self.median_age_min,
            self.median_age_max,
            "neighborhood.median_age",
        )
        _validate_min_max(
            self.household_size_min,
            self.household_size_max,
            "neighborhood.household_size",
        )
        _validate_probability_map(
            self.school_status_probs,
            {"existing", "planned", "none"},
            "neighborhood.school_status_probs",
        )
        return self


class RoomMixSettings(_Settings):
    """Apartment room-composition distribution within a building."""

    supported_rooms: tuple[int, ...]
    base_shares: dict[int, float]
    household_size_tilt: float
    ses_tilt: float
    room_distance_divisor: float = Field(gt=0)
    dirichlet_concentration: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_rooms_and_shares(self) -> RoomMixSettings:
        if not self.supported_rooms:
            raise ValueError("supported rooms must not be empty: room_mix")
        _validate_probability_map(
            self.base_shares,
            set(self.supported_rooms),
            "room_mix.base_shares",
        )
        return self


class BuildingSettings(_Settings):
    """Building-count, apartment-count, and room-mix settings."""

    buildings_per_neighborhood_min: int = Field(gt=0)
    buildings_per_neighborhood_rate: float = Field(gt=0)
    apartments_per_building_mean: int = Field(gt=0)
    apartments_per_building_sd: float = Field(gt=0)
    apartments_per_building_min: int
    apartments_per_building_max: int
    buildings_per_neighborhood: tuple[int, ...]
    apartments_per_building: tuple[int, ...]
    room_mix: RoomMixSettings

    @model_validator(mode="after")
    def validate_ranges_and_overrides(self) -> BuildingSettings:
        _validate_min_max(
            self.apartments_per_building_min,
            self.apartments_per_building_max,
            "building.apartments_per_building",
        )
        if any(value <= 0 for value in self.buildings_per_neighborhood):
            raise ValueError("building counts must be positive")
        for value in self.apartments_per_building:
            if not (
                self.apartments_per_building_min
                <= value
                <= self.apartments_per_building_max
            ):
                raise ValueError(
                    "building apartment overrides must be within configured "
                    "bounds"
                )
        return self


class ApartmentSettings(_Settings):
    """Reserved settings boundary for future apartment generation controls."""


class TotalChildrenSettings(_Settings):
    """NB2 total-children model coefficients and dispersion settings."""

    intercept: float
    room_log_mean_effects: dict[int, float]
    ses_quadratic_coef: float
    household_size_coef: float
    daycare_saturation_coef: float
    existing_school_coef: float
    planned_school_coef: float
    median_age_coef: float
    rooms_household_interaction_coef: float
    daycare_saturation_scale: float = Field(gt=0)
    building_effect_sd: float = Field(gt=0)
    neighborhood_effect_sd: float = Field(gt=0)
    nb_dispersion_phi: float = Field(gt=0)


class CohortSettings(_Settings):
    """Age-cohort composition model coefficients and reference shares."""

    reference_shares: dict[str, float]
    room_coef: float
    daycare_coef: float
    median_age_coef: float
    household_size_coef: float
    room_age_positive_interaction_coef: float
    building_effect_sd: float = Field(gt=0)
    kindergarten_lifecycle_slope: float
    highschool_lifecycle_slope: float

    @model_validator(mode="after")
    def validate_reference_shares(self) -> CohortSettings:
        _validate_probability_map(
            self.reference_shares,
            {"kindergarten", "elementary", "highschool"},
            "cohort.reference_shares",
        )
        return self


class ValidationSettings(_Settings):
    """Test-only calibration and coefficient-recovery acceptance thresholds."""

    mean_children_min: float
    mean_children_max: float
    zero_share_min: float
    zero_share_max: float
    recovery_abs_tolerance: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> ValidationSettings:
        _validate_min_max(
            self.mean_children_min,
            self.mean_children_max,
            "validation.mean_children",
        )
        _validate_min_max(
            self.zero_share_min,
            self.zero_share_max,
            "validation.zero_share",
        )
        return self


class Config(_Settings):
    """Complete generation configuration loaded from the stage TOML file."""

    simulation: SimulationSettings
    transforms: TransformSettings = Field(validation_alias="features")
    neighborhood: NeighborhoodSettings
    building: BuildingSettings
    apartment: ApartmentSettings
    total_children: TotalChildrenSettings
    cohort: CohortSettings

    @model_validator(mode="after")
    def validate_cross_section_contracts(self) -> Config:
        building_override = self.building.buildings_per_neighborhood
        if (
            building_override
            and len(building_override) != self.simulation.n_neighborhoods
        ):
            raise ValueError(
                "building.buildings_per_neighborhood length must match "
                "simulation.n_neighborhoods"
            )

        supported_rooms = set(self.building.room_mix.supported_rooms)
        reference_room = min(supported_rooms)
        expected_effect_rooms = supported_rooms - {reference_room}
        if (
            set(self.total_children.room_log_mean_effects)
            != expected_effect_rooms
        ):
            raise ValueError(
                "total_children.room_log_mean_effects must include every "
                "supported room except the smallest reference room"
            )
        return self


def _validate_min_max(minimum: float, maximum: float, path: str) -> None:
    """Require a coherent inclusive numeric range."""
    if minimum > maximum:
        raise ValueError(f"minimum must not exceed maximum: {path}")


def _validate_probability_map(
    values: Mapping[str | int, float],
    expected_keys: set[str | int],
    path: str,
) -> None:
    """Require exact labels, non-negative values, and a unit total."""
    actual_keys = set(values)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=str)
        unknown = sorted(actual_keys - expected_keys, key=str)
        messages = [
            *(f"missing key: {path}.{key}" for key in missing),
            *(f"unknown key: {path}.{key}" for key in unknown),
        ]
        raise ValueError("; ".join(messages))
    if any(value < 0 for value in values.values()):
        raise ValueError(f"probabilities must be non-negative: {path}")
    total = sum(values.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"probabilities must sum to 1.0: {path} (got {total})"
        )


def _load_toml(path: str | Path) -> dict[str, object]:
    """Load TOML into the raw mapping accepted by the typed models."""
    with Path(path).open("rb") as file_obj:
        return tomllib.load(file_obj)


def _raise_config_error(error: ValidationError) -> None:
    """Present Pydantic validation failures through the public error type."""
    messages: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        message = item["msg"]
        messages.append(f"{location}: {message}" if location else message)
    raise ConfigError("; ".join(messages)) from error


def load_simulation_config(path: str | Path) -> Config:
    """Load strict immutable generation settings from a stage TOML file."""
    raw = _load_toml(path)
    try:
        return Config.model_validate(raw)
    except ValidationError as error:
        _raise_config_error(error)
        raise AssertionError("unreachable")


def load_validation_config(path: str | Path) -> ValidationSettings:
    """Load strict test-only calibration and recovery thresholds."""
    raw = _load_toml(path)
    try:
        return ValidationSettings.model_validate(raw["validation"])
    except (KeyError, ValidationError) as error:
        if isinstance(error, ValidationError):
            _raise_config_error(error)
        raise ConfigError("missing key: config.validation") from error