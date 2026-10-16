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
    assert config.neighborhood.school_status_probs == {
        "existing": 0.55,
        "planned": 0.20,
        "none": 0.25,
    }
    room_mix = config.building.room_mix
    assert room_mix.supported_rooms == (3, 4, 5, 6)
    assert set(room_mix.base_shares) == {3, 4, 5, 6}
    assert config.transforms.household_size_reference == 2.6
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

    with pytest.raises(ConfigError, match=r"probabilities must sum to 1.0: neighborhood.school_status_probs"):
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
