"""Fit-free, row-wise preprocessing, safe to run on the full table before splitting.

Nothing here learns from the data. Anything that learns a statistic must be
fitted per fold instead, in :mod:`age_group_prediction.feature_engineering`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["ShareTransformer"]


class ShareTransformer(TransformerMixin, BaseEstimator):
    """Replace a group of count columns with their row-wise shares.

    The denominator is the row sum of ``share_columns``, not some outside
    total. ``reference_column``, when given, is the share left out, so the rest
    read as contrasts against it.

    No ``inverse_transform``: the row sum is discarded.
    """

    def __init__(
        self,
        share_columns: Sequence[str],
        *,
        reference_column: str | None = None,
        suffix: str = "_share",
        drop_inputs: bool = True,
    ) -> None:
        # Stored verbatim: a clone assigns attributes without re-entering here.
        self.share_columns = share_columns
        self.reference_column = reference_column
        self.suffix = suffix
        self.drop_inputs = drop_inputs

    def fit(self, X: pd.DataFrame, y: object = None) -> ShareTransformer:
        """Record the output columns; nothing is learned from ``X``."""
        self.feature_names_out_ = tuple(self._kept(X)) + self._share_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` with the counts replaced by their shares."""
        check_is_fitted(self)
        totals = X.loc[:, list(self.share_columns)].to_numpy(dtype=float).sum(axis=1)
        if (totals <= 0).any():
            # numpy would return nan here and only warn, so the failure would
            # surface far from its cause.
            raise ValueError(
                f"share_columns sum to zero for rows "
                f"{X.index[totals <= 0].tolist()[:5]}; their shares are undefined"
            )
        emitted = [name for name in self.share_columns if name != self.reference_column]
        shares = pd.DataFrame(
            X.loc[:, emitted].to_numpy(dtype=float) / totals[:, None],
            index=X.index,
            columns=list(self._share_names()),
        )
        # .loc, not a numpy round trip, so passthrough dtypes survive.
        return pd.concat([X.loc[:, self._kept(X)], shares], axis=1)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """The output columns, in order."""
        check_is_fitted(self)
        return np.asarray(self.feature_names_out_, dtype=object)

    def _kept(self, X: pd.DataFrame) -> list[str]:
        """The columns that survive beside the shares."""
        consumed = set(self.share_columns) if self.drop_inputs else set()
        return [name for name in X.columns if name not in consumed]

    def _share_names(self) -> tuple[str, ...]:
        """Name each share for its count. Used by both transform and fit."""
        return tuple(
            f"{name}{self.suffix}"
            for name in self.share_columns
            if name != self.reference_column
        )
