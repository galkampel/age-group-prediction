from __future__ import annotations

from typing import Iterable

import numpy as np


def scale_household_size(
    values: np.ndarray | Iterable[float],
    reference: float,
    scale: float,
) -> np.ndarray:
    """Scale household size into the named model scale."""
    arr = np.asarray(values, dtype=float)
    return (arr - reference) / scale


def scale_median_age(
    values: np.ndarray | Iterable[float],
    reference: float,
    scale: float,
) -> np.ndarray:
    """Scale neighborhood median age into the named model scale."""
    arr = np.asarray(values, dtype=float)
    return (arr - reference) / scale


def center_rooms(
    values: np.ndarray | Iterable[float],
    reference: float,
) -> np.ndarray:
    """Center room counts around the configured reference apartment size."""
    arr = np.asarray(values, dtype=float)
    return arr - reference


def daycare_saturation(
    values: np.ndarray | Iterable[float],
    saturation_scale: float,
) -> np.ndarray:
    """Compute daycare saturation as $1 - exp(-D / scale)$."""
    arr = np.asarray(values, dtype=float)
    return 1.0 - np.exp(-arr / saturation_scale)


def positive_part(values: np.ndarray | Iterable[float]) -> np.ndarray:
    """Apply the element-wise positive-part transform max(x, 0)."""
    arr = np.asarray(values, dtype=float)
    return np.maximum(arr, 0.0)


def positive_part_interaction(
    lhs: np.ndarray | Iterable[float],
    rhs: np.ndarray | Iterable[float],
) -> np.ndarray:
    """Compute the one-sided interaction max(lhs, 0) * max(rhs, 0)."""
    return positive_part(lhs) * positive_part(rhs)


def normalize_positive_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize non-negative weights to probabilities along the last axis."""
    arr = np.asarray(weights, dtype=float)
    if np.any(arr < 0):
        raise ValueError("weights must be non-negative")
    denominator = arr.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError("weights must have a strictly positive sum")
    return arr / denominator


def stable_row_normalize(values: np.ndarray) -> np.ndarray:
    """Convert rows of logits to stable probability distributions."""
    arr = np.asarray(values, dtype=float)
    row_max = np.max(arr, axis=1, keepdims=True)
    exp_values = np.exp(arr - row_max)
    denominator = exp_values.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0) or np.any(~np.isfinite(denominator)):
        raise ValueError(
            "row normalization failed due to non-finite denominator"
        )
    return exp_values / denominator


def expected_room_probabilities(
    supported_rooms: tuple[int, ...],
    base_shares: dict[int, float],
    household_size_tilt: float,
    ses_tilt: float,
    room_distance_divisor: float,
    household_size_scaled: np.ndarray | Iterable[float],
    ses: np.ndarray | Iterable[float],
) -> np.ndarray:
    """Compute room probabilities before Dirichlet/multinomial draws."""
    household_size = np.asarray(household_size_scaled, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    if household_size.shape != ses_arr.shape:
        raise ValueError(
            "household_size_scaled and ses must have the same shape"
        )

    rooms = np.asarray(supported_rooms, dtype=float)
    base = np.asarray(
        [base_shares[int(room)] for room in supported_rooms],
        dtype=float,
    )
    tilt = household_size_tilt * household_size + ses_tilt * ses_arr
    distance = (rooms - 4.0) / room_distance_divisor
    weights = base[None, :] * np.exp(tilt[:, None] * distance[None, :])
    return normalize_positive_weights(weights)
