"""Model 2's second half: each cohort's probability for a building, from a classifier."""

from __future__ import annotations

from typing import Literal, Protocol, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from ..feature_engineering import FeatureTransformer
from ..utils import take_rows
from .base import BaseAgeGroupModel

__all__ = ["Classifier", "CohortProbabilityModel", "ReplicationType"]

# How the cohort counts become categorical rows (see CohortProbabilityModel).
type ReplicationType = Literal["per_child", "weighted"]


class Classifier(Protocol):
    """What ``estimator`` must do: scikit-learn's classifier API; ``sample_weight`` for the weighted rows."""

    classes_: np.ndarray

    def fit(
        self, X: pd.DataFrame, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self: ...

    def predict_proba(self, X: pd.DataFrame) -> ArrayLike: ...


class CohortProbabilityModel(BaseAgeGroupModel):
    """Each cohort's probability for a building: a classifier on the children, labelled by cohort position.

    ``estimator`` is any multi-class classifier, with its own hyperparameters
    (``estimator__…`` in ``set_params``). ``X`` is the raw table when
    ``feature_transformer`` is given, otherwise the finished design matrix;
    ``y`` is a DataFrame of cohort counts, one column per cohort (at least
    two), non-negative and finite: not checked here, since a data value
    belongs where the data is prepared, yet a negative count would be a
    negative weight, which most classifiers fit silently. Rows are paired by
    position.

    ``fit`` turns the counts into categorical rows
    (:meth:`multinomial_to_categorical`), by ``replication``:

    - ``"weighted"``: one row per sample and cohort with a child, weighted by
      its count (``sample_weight``); a classifier without ``sample_weight``
      raises here;
    - ``"per_child"``: one row per child, unweighted, so a classifier without
      ``sample_weight`` fits too; the counts must be of an integer dtype that
      numpy casts safely to ``int64`` (``np.repeat`` rejects floats,
      ``uint64`` and pandas' ``Int64``).

    Either way each child counts once. A row's label is its cohort's column
    position in ``y``, so the classifier's ``classes_`` are ``0 … K−1`` in
    ``y``'s order (cohort names of any type, never sorted). The feature
    transformer is fitted on the samples first; a sample with no children
    adds no row. ``predict`` returns the classifier's probabilities with
    ``y``'s columns at fit, and raises if a row does not sum to 1 (a
    one-vs-all objective such as LightGBM's ``"multiclassova"``).
    A ``OneVsRestClassifier`` receives the weights only with scikit-learn's
    metadata routing enabled. The model has no exposure, so a passed one is
    ignored.
    """

    def __init__(
        self,
        *,
        estimator: Classifier,
        replication: ReplicationType = "weighted",
        feature_transformer: FeatureTransformer | None = None,
    ) -> None:
        # Stored verbatim: set_params assigns attributes without re-entering here.
        self.estimator = estimator
        self.replication = replication
        self.feature_transformer = feature_transformer

    @staticmethod
    def multinomial_to_categorical(
        y: pd.DataFrame, replication: ReplicationType
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The sample positions in ``y``, cohort positions (the labels) and weights of the categorical rows.

        ``"weighted"``: every non-zero cell, its count the weight.
        ``"per_child"``: each cell repeated by its count, no weights.
        """
        counts = y.to_numpy()
        sample_positions, cohort_positions = np.nonzero(counts)
        cell_counts = counts[sample_positions, cohort_positions]
        if replication == "weighted":
            return sample_positions, cohort_positions, cell_counts
        elif replication == "per_child":
            return (
                np.repeat(sample_positions, cell_counts),
                np.repeat(cohort_positions, cell_counts),
                None,
            )
        else:
            raise ValueError(
                f"replication must be 'weighted' or 'per_child', got {replication!r}"
            )

    @staticmethod
    def _check_every_cohort_observed(y: pd.DataFrame) -> None:
        """Raise naming the cohorts without a child: the classifier would never predict them."""
        observed = (y.to_numpy() > 0).any(axis=0)
        unobserved = list(y.columns[~observed])
        # predict would lack their columns, and fail only there.
        if unobserved:
            raise ValueError(
                f"no child is observed in cohorts {unobserved}; drop them from y"
            )

    @staticmethod
    def _check_rows_sum_to_one(
        probabilities: np.ndarray, estimator: Classifier
    ) -> None:
        """Raise if a row does not sum to 1, which Model 2 would multiply by the total silently."""
        row_sums = probabilities.sum(axis=1)
        # A one-vs-all objective (LightGBM's "multiclassova") scores each cohort alone.
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{type(estimator).__name__}'s probabilities do not sum to 1 "
                f"in every row (row sums {np.min(row_sums)} to {np.max(row_sums)}; "
                "nan if any is nan); use a softmax objective (e.g. LightGBM's "
                "'multiclass') or wrap the classifier in OneVsRestClassifier"
            )

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit a copy of ``estimator`` on ``X`` and the cohort counts ``y``; ``exposure`` is ignored."""
        # The rows are taken from y's cells, so a shorter y would drop X's last rows silently.
        check_consistent_length(X, y)
        self._check_every_cohort_observed(y)
        # On the samples: the categorical rows would weight its statistics by children.
        feature_transformer, X = self._fit_features(X, y)
        sample_positions, cohort_positions, weights = self.multinomial_to_categorical(
            y, self.replication
        )
        X_categorical = take_rows(X, sample_positions)
        # A copy, so the caller's template stays unfitted across folds.
        estimator: Classifier = clone(self.estimator)
        if self.replication == "weighted":
            estimator.fit(X_categorical, cohort_positions, sample_weight=weights)
        elif self.replication == "per_child":
            # No keyword: a classifier without sample_weight fits too.
            estimator.fit(X_categorical, cohort_positions)
        else:
            # Unreachable: multinomial_to_categorical raised for it.
            raise ValueError(f"unknown replication {self.replication!r}")
        # Set together, only once fitting succeeded.
        self.estimator_ = estimator
        self.cohorts_: list[object] = list(y.columns)
        self.feature_transformer_ = feature_transformer
        return self

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> pd.DataFrame:
        """Each cohort's probability for each row of ``X``; ``exposure`` is ignored."""
        check_is_fitted(self)
        X = self._transform_features(X)
        probabilities = np.asarray(self.estimator_.predict_proba(X), dtype=float)
        self._check_rows_sum_to_one(probabilities, self.estimator_)
        # classes_ are the cohort positions 0 … K−1, y's order: the classifier
        # was fitted on them, and every cohort was observed.
        return pd.DataFrame(probabilities, columns=self.cohorts_, index=X.index)
