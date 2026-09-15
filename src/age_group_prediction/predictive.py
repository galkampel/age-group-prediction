"""Shared helpers for predictive payload post-processing."""

from __future__ import annotations

import numpy as np


def central_prediction_intervals(
    draws: np.ndarray, levels: tuple[float, ...]
) -> np.ndarray:
    """Compute central intervals with shape ``(rows, levels, lower_or_upper)``."""
    intervals = np.empty((draws.shape[0], len(levels), 2), dtype=float)
    for index, level in enumerate(levels):
        alpha = 1.0 - level
        lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0], axis=1)
        intervals[:, index, 0] = lower
        intervals[:, index, 1] = upper
    return intervals


__all__ = ["central_prediction_intervals"]