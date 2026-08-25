from __future__ import annotations

import numpy as np

from student_simulator.features import (
    center_rooms,
    daycare_saturation,
    expected_room_probabilities,
    positive_part_interaction,
    scale_household_size,
    scale_median_age,
)


def test_named_scales_zero_at_references() -> None:
    hh = np.array([2.6])
    age = np.array([37.0])
    rooms = np.array([4])

    assert np.allclose(scale_household_size(hh, reference=2.6, scale=0.5), 0.0)
    assert np.allclose(scale_median_age(age, reference=37.0, scale=10.0), 0.0)
    assert np.allclose(center_rooms(rooms, reference=4.0), 0.0)


def test_daycare_saturation_properties() -> None:
    values = np.array([0, 1, 3, 10], dtype=float)
    sat = daycare_saturation(values, saturation_scale=3.0)

    assert np.isclose(sat[0], 0.0)
    assert np.all(np.diff(sat) > 0)
    assert sat[-1] < 1.0
    assert sat[-1] > 0.95


def test_positive_part_interaction_activation_quadrants() -> None:
    lhs = np.array([-1.0, -1.0, 2.0, 2.0])
    rhs = np.array([-1.0, 3.0, -1.0, 3.0])
    interaction = positive_part_interaction(lhs, rhs)
    assert np.array_equal(interaction, np.array([0.0, 0.0, 0.0, 6.0]))


def test_expected_room_probabilities_are_valid_and_tilt() -> None:
    probs = expected_room_probabilities(
        supported_rooms=(3, 4, 5, 6),
        base_shares={3: 0.30, 4: 0.38, 5: 0.22, 6: 0.10},
        household_size_tilt=0.35,
        ses_tilt=0.20,
        room_distance_divisor=2.0,
        household_size_scaled=np.array([-1.0, 1.5]),
        ses=np.array([-1.0, 1.0]),
    )

    assert probs.shape == (2, 4)
    assert np.all(np.isfinite(probs))
    assert np.all(probs > 0)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Higher household-size/SES row should have more mass on larger rooms.
    assert probs[1, 3] > probs[0, 3]
    assert probs[1, 0] < probs[0, 0]


def test_nb2_sampling_matches_mean_and_variance() -> None:
    from pathlib import Path

    from student_simulator import load_simulation_config

    from tests.unit.helpers import make_total_children_simulator, with_overrides

    rng = np.random.default_rng(123)
    mu_values = np.array([0.5, 1.2, 2.5, 5.0])
    phi_values = np.array([2.0, 3.0, 5.0])
    config_path = Path(__file__).resolve().parents[2] / "configs" / "stage1.toml"

    n = 200_000
    for mu in mu_values:
        for phi in phi_values:
            config = load_simulation_config(config_path)
            config = with_overrides(
                config, total_children={"nb_dispersion_phi": phi}
            )
            simulator = make_total_children_simulator(config)
            draws = simulator._sample_nb2(rng, np.full(n, mu))
            sample_mean = draws.mean()
            sample_var = draws.var()
            expected_mean = mu
            expected_var = mu + (mu**2) / phi

            # Tolerances are deliberately generous but deterministic.
            assert abs(sample_mean - expected_mean) < 0.06 + 0.03 * expected_mean
            assert abs(sample_var - expected_var) < 0.25 + 0.10 * expected_var
