"""Unit tests for neighborhood-cluster bootstrap resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from age_group_prediction.resampling import NeighborhoodClusterResampler


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neighborhood_id": [0, 0, 1, 1, 1, 2],
            "value": [10, 11, 20, 21, 22, 30],
        }
    )


def test_from_frame_requires_neighborhood_column() -> None:
    frame = pd.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing neighborhood column"):
        NeighborhoodClusterResampler.from_frame(frame, "neighborhood_id")


def test_from_frame_requires_multiple_neighborhoods() -> None:
    frame = pd.DataFrame({"neighborhood_id": [7, 7, 7]})
    with pytest.raises(
        ValueError,
        match="Neighborhood-cluster bootstrap requires at least two neighborhoods",
    ):
        NeighborhoodClusterResampler.from_frame(frame, "neighborhood_id")


def test_sample_row_indices_is_seed_reproducible() -> None:
    resampler = NeighborhoodClusterResampler.from_frame(_frame(), "neighborhood_id")

    first = resampler.sample_row_indices(np.random.default_rng(1234))
    second = resampler.sample_row_indices(np.random.default_rng(1234))

    np.testing.assert_array_equal(first, second)


def test_sample_row_indices_preserves_whole_clusters() -> None:
    frame = _frame()
    resampler = NeighborhoodClusterResampler.from_frame(frame, "neighborhood_id")
    sampled = resampler.sample_row_indices(np.random.default_rng(5))

    sampled_neighborhoods = frame.iloc[sampled]["neighborhood_id"].to_numpy()
    sampled_counts = {
        int(neighborhood): int((sampled_neighborhoods == neighborhood).sum())
        for neighborhood in np.unique(sampled_neighborhoods)
    }
    original = {
        int(neighborhood): int((frame["neighborhood_id"] == neighborhood).sum())
        for neighborhood in np.unique(frame["neighborhood_id"])
    }
    assert set(sampled_counts).issubset(set(original))
    for neighborhood, count in sampled_counts.items():
        assert count % original[neighborhood] == 0
