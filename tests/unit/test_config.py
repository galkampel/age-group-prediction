from __future__ import annotations

from pathlib import Path

import pytest

from student_simulator.config import (
    Config,
    ConfigError,
    load_simulation_config,
)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"


def test_load_simulation_config_returns_typed_settings() -> None:
    config = load_simulation_config(_config_path())

    assert isinstance(config, Config)
    assert config.neighborhood.school_status_reference_shares == {
        "existing": 0.55,
        "planned": 0.20,
        "none": 0.25,
    }
    room_mix = config.building.room_mix
    assert room_mix.supported_rooms == (3, 4, 5, 6)
    assert set(room_mix.base_shares) == {3, 4, 5, 6}
    assert config.transforms.household_size_reference == 3.0
    assert config.transforms.room_reference == 4.0
    assert config.total_children.nb_dispersion_phi > 0


def test_simulation_loader_returns_generation_settings_only() -> None:
    config = load_simulation_config(_config_path())

    assert config.simulation.seed == 42
    assert config.transforms.room_reference == 4.0
    assert config.building.room_mix.supported_rooms == (3, 4, 5, 6)


def test_runtime_config_can_be_copied_with_overrides() -> None:
    config = load_simulation_config(_config_path())
    overridden = config.model_copy(
        update={"simulation": config.simulation.model_copy(update={"seed": 7})}
    )

    assert overridden.simulation.seed == 7
    assert config.simulation.seed == 42


def test_unknown_key_reports_full_path(tmp_path: Path) -> None:
    raw_text = _config_path().read_text()
    text = raw_text.replace(
        "  n_neighborhoods: 60",
        "  n_neighborhoods: 60\n  unknown_setting: 123",
    )
    cfg_path = tmp_path / "bad.yml"
    cfg_path.write_text(text)

    with pytest.raises(ConfigError, match=r"simulation.unknown_setting"):
        load_simulation_config(cfg_path)


def test_missing_key_reports_full_path(tmp_path: Path) -> None:
    raw_text = _config_path().read_text()
    text = "\n".join(
        line
        for line in raw_text.splitlines()
        if not line.strip().startswith("nb_dispersion_phi:")
    )
    cfg_path = tmp_path / "bad_missing.yml"
    cfg_path.write_text(text)

    with pytest.raises(ConfigError, match=r"total_children.nb_dispersion_phi"):
        load_simulation_config(cfg_path)


def test_invalid_probability_sum_fails(tmp_path: Path) -> None:
    raw_text = _config_path().read_text()
    text = raw_text.replace("none: 0.25", "none: 0.20")
    cfg_path = tmp_path / "bad_probs.yml"
    cfg_path.write_text(text)

    with pytest.raises(
        ConfigError,
        match=r"probabilities must sum to 1.0: "
        r"neighborhood.school_status_reference_shares",
    ):
        load_simulation_config(cfg_path)


def test_override_lengths_and_values_are_validated(tmp_path: Path) -> None:
    raw_text = _config_path().read_text()
    text = raw_text.replace(
        "buildings_per_neighborhood: []",
        "buildings_per_neighborhood: [1, 2]",
    )
    cfg_path = tmp_path / "bad_override.yml"
    cfg_path.write_text(text)

    with pytest.raises(ConfigError, match=r"building.buildings_per_neighborhood"):
        load_simulation_config(cfg_path)


def test_invalid_feature_scale_fails(tmp_path: Path) -> None:
    text = _config_path().read_text().replace(
        "median_age_scale: 10.0",
        "median_age_scale: 0.0",
    )
    cfg_path = tmp_path / "bad_scale.yml"
    cfg_path.write_text(text)

    with pytest.raises(ConfigError, match=r"features.median_age_scale"):
        load_simulation_config(cfg_path)


def test_missing_nested_room_mix_reports_full_path(tmp_path: Path) -> None:
    text = _config_path().read_text().replace(
        "  room_mix:",
        "  removed_room_mix:",
    )
    cfg_path = tmp_path / "missing_room_mix.yml"
    cfg_path.write_text(text)

    with pytest.raises(ConfigError, match=r"building.room_mix"):
        load_simulation_config(cfg_path)


def test_negative_seed_fails_before_simulation(tmp_path: Path) -> None:
    text = _config_path().read_text().replace("seed: 42", "seed: -1")
    cfg_path = tmp_path / "negative_seed.yml"
    cfg_path.write_text(text)

    with pytest.raises(ConfigError, match=r"simulation.seed"):
        load_simulation_config(cfg_path)


def test_yaml_comments_do_not_change_values(tmp_path: Path) -> None:
    text = _config_path().read_text().replace(
        "  seed: 42",
        "  # An additional experiment comment.\n  seed: 42  # And an inline one.",
    )
    cfg_path = tmp_path / "commented.yml"
    cfg_path.write_text(text)

    config = load_simulation_config(cfg_path)
    assert config.simulation.seed == 42


def test_an_empty_file_fails_as_a_config_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "empty.yml"
    cfg_path.write_text("")

    with pytest.raises(ConfigError, match=r"valid dictionary"):
        load_simulation_config(cfg_path)


def test_an_override_is_coerced_to_its_field_type() -> None:
    # model_copy(update=...) would keep the int, and the notebook's config
    # hash would then differ for the same data.
    config = load_simulation_config(_config_path())
    as_int = config.with_overrides({"building": {"buildings_per_neighborhood_rate": 9}})

    assert isinstance(as_int.building.buildings_per_neighborhood_rate, float)


@pytest.mark.parametrize("section", ["transforms", "features"])
def test_the_transform_settings_take_either_name(section: str) -> None:
    # The attribute and the file's key name the same section; an override
    # under the file's key was once overwritten by the dump's, silently.
    config = load_simulation_config(_config_path())
    overridden = config.with_overrides({section: {"room_reference": 5.0}})

    assert overridden.transforms.room_reference == 5.0


def test_an_override_runs_the_cross_section_checks() -> None:
    # model_copy(update=...) accepts two values for 60 neighborhoods silently.
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=r"building.buildings_per_neighborhood"):
        config.with_overrides({"building": {"buildings_per_neighborhood": (1, 2)}})


def test_an_override_of_an_unknown_field_is_rejected() -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=r"simulation.n_neighbourhoods"):
        config.with_overrides({"simulation": {"n_neighbourhoods": 150}})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("median_age_mean", 37.0),
        ("median_age_sd", 5.0),
        ("school_status_probs", {"existing": 0.55, "planned": 0.20, "none": 0.25}),
    ],
)
def test_a_removed_neighborhood_field_is_rejected(field: str, value: object) -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=rf"neighborhood.{field}"):
        config.with_overrides({"neighborhood": {field: value}})


def test_a_zero_school_reference_share_is_rejected() -> None:
    # The draw takes the log of each share; the map still sums to 1.
    config = load_simulation_config(_config_path())
    shares = {"existing": 0.55, "planned": 0.0, "none": 0.45}

    with pytest.raises(
        ConfigError,
        match=r"reference shares must be positive: "
        r"neighborhood.school_status_reference_shares",
    ):
        config.with_overrides(
            {"neighborhood": {"school_status_reference_shares": shares}}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("median_age_baseline", 15.0),
        ("median_age_baseline", 60.0),
        ("household_size_baseline", 1.5),
        ("household_size_baseline", 7.0),
    ],
)
def test_a_baseline_outside_its_bounds_is_rejected(field: str, value: float) -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(
        ConfigError, match=rf"baseline must be within its bounds: neighborhood.{field}"
    ):
        config.with_overrides({"neighborhood": {field: value}})


def test_a_household_size_minimum_of_one_is_rejected() -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(
        ConfigError,
        match=r"household size minimum must exceed 1: neighborhood.household_size_min",
    ):
        config.with_overrides({"neighborhood": {"household_size_min": 1.0}})


@pytest.mark.parametrize(
    ("section", "field", "value", "path"),
    [
        ("features", "household_size_reference", 2.6, "features.household_size_reference"),
        ("features", "median_age_reference", 37.0, "features.median_age_reference"),
        ("neighborhood", "household_size_baseline", 2.6, "features.household_size_reference"),
        ("neighborhood", "median_age_baseline", 37.0, "features.median_age_reference"),
    ],
)
def test_the_reference_neighborhood_must_equal_the_baselines(
    section: str, field: str, value: float, path: str
) -> None:
    # Either side changed alone leaves the outcome models centered on a
    # neighborhood the draws do not center on.
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=rf"{path} must equal neighborhood\."):
        config.with_overrides({section: {field: value}})


def test_the_reference_neighborhood_may_move_with_the_baselines() -> None:
    config = load_simulation_config(_config_path())

    moved = config.with_overrides(
        {
            "features": {"household_size_reference": 2.6, "median_age_reference": 37.0},
            "neighborhood": {"household_size_baseline": 2.6, "median_age_baseline": 37.0},
        }
    )

    assert moved.transforms.household_size_reference == 2.6
    assert moved.neighborhood.median_age_baseline == 37.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("apartments_per_building_mean", 40), ("apartments_per_building_sd", 12.0)],
)
def test_a_removed_building_field_is_rejected(field: str, value: object) -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=rf"building.{field}"):
        config.with_overrides({"building": {field: value}})


def test_a_zero_apartment_minimum_is_rejected() -> None:
    # The room tilt takes the log of each building's size.
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=r"building.apartments_per_building_min"):
        config.with_overrides({"building": {"apartments_per_building_min": 0}})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("texture_apartments_median", 7.0),
        ("tower_apartments_median", 161.0),
    ],
)
def test_a_building_median_outside_the_bounds_is_rejected(
    field: str, value: float
) -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(
        ConfigError,
        match=rf"median must be within the apartment bounds: building.{field}",
    ):
        config.with_overrides({"building": {field: value}})


@pytest.mark.parametrize(
    "field",
    [
        "tower_propensity_sd",
        "apartments_neighborhood_log_sd",
        "apartments_building_log_sd",
    ],
)
def test_a_zero_building_sd_is_rejected(field: str) -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=rf"building.{field}"):
        config.with_overrides({"building": {field: 0.0}})


@pytest.mark.parametrize("value", [1.8, 2.0])
def test_a_childless_household_size_at_or_above_the_minimum_is_rejected(
    value: float,
) -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(
        ConfigError,
        match=r"total_children.childless_household_size must be below "
        r"neighborhood.household_size_min",
    ):
        config.with_overrides({"total_children": {"childless_household_size": value}})


@pytest.mark.parametrize("value", [0, 2.5])
def test_a_max_children_below_one_or_fractional_is_rejected(value: float) -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=r"total_children.max_children"):
        config.with_overrides({"total_children": {"max_children": value}})


def test_the_removed_cohort_bonus_is_rejected() -> None:
    config = load_simulation_config(_config_path())

    with pytest.raises(ConfigError, match=r"cohort.room_age_positive_interaction_coef"):
        config.with_overrides({"cohort": {"room_age_positive_interaction_coef": 0.12}})


def test_a_config_without_the_cohort_planned_school_coef_is_rejected(
    tmp_path: Path,
) -> None:
    raw_text = _config_path().read_text()
    text = raw_text.replace("  planned_school_coef: -0.25\n", "", 1)
    # The total-children section keeps its own field of the same name.
    assert text.count("\n  planned_school_coef:") == 1
    cfg_path = tmp_path / "bad_missing_cohort.yml"
    cfg_path.write_text(text)

    with pytest.raises(ConfigError, match=r"cohort.planned_school_coef"):
        load_simulation_config(cfg_path)
