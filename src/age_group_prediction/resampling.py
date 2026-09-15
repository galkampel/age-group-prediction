"""Shared neighborhood-cluster resampling for bootstrap workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeighborhoodClusterResampler:
    """Cache neighborhood row clusters and sample bootstrap row indices."""

    neighborhoods: np.ndarray
    cluster_rows: Mapping[object, np.ndarray]

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, neighborhood_column: str
    ) -> NeighborhoodClusterResampler:
        """Build a cluster index from one frame and neighborhood column."""
        if neighborhood_column not in frame:
            raise ValueError(f"Missing neighborhood column: {neighborhood_column}")
        neighborhoods = frame[neighborhood_column].drop_duplicates().to_numpy()
        if len(neighborhoods) < 2:
            raise ValueError(
                "Neighborhood-cluster bootstrap requires at least two neighborhoods"
            )
        neighborhood_values = frame[neighborhood_column].to_numpy()
        cluster_rows = {
            neighborhood: np.flatnonzero(neighborhood_values == neighborhood)
            for neighborhood in neighborhoods
        }
        return cls(neighborhoods=neighborhoods, cluster_rows=cluster_rows)

    def sample_row_indices(self, rng: np.random.Generator) -> np.ndarray:
        """Sample full neighborhoods with replacement and return row indices."""
        sampled = rng.choice(
            self.neighborhoods, size=len(self.neighborhoods), replace=True
        )
        return np.concatenate([self.cluster_rows[neighborhood] for neighborhood in sampled])


__all__ = ["NeighborhoodClusterResampler"]