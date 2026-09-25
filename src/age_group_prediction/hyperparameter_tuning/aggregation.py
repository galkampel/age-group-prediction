"""How a trial's fold scores are combined into the one value the study maximizes.

An aggregation knows nothing about the data: it receives each fold's score
(greater is better) and size (its validation rows), computed by the
evaluator, and returns a number.

- :class:`WeightedMean` (the default): the size-weighted mean. For a metric
  that is a mean over rows (MAE, Poisson deviance) it equals the pooled
  per-row score, the measure the test set reports.
- :class:`Mean`: the plain mean, as ``cross_val_score(...).mean()``.
- :class:`LowerBound`: the weighted mean minus ``z`` standard errors, to prefer
  stable settings over the best average. Its ranking is noisier: the SE is
  estimated from only K scores.

Subclass :class:`Aggregation` for another rule, e.g. a median.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Annotated

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ._config import _STRICT

__all__ = [
    "Aggregation",
    "LowerBound",
    "Mean",
    "WeightedMean",
    "corrected_std_error",
]


def corrected_std_error(scores: Sequence[float]) -> float:
    """The corrected standard error of the mean fold score, for K-fold splits.

    ``s * sqrt(1/K + 1/(K-1))``, with ``s`` the scores' std (ddof=1): the
    Nadeau–Bengio correction in its K-fold form (Bouckaert & Frank, 2004),
    where ``n_val/n_train = 1/(K-1)``. The naive ``s/sqrt(K)`` is too small:
    the folds share most of their training rows. The ratio is exact when
    every row is validated once (``random``, ``grouped``). Under
    ``stratified_by_group``, rows alone in their stratum are never validated,
    so the SE comes out slightly large, i.e. conservative. Needs 2 folds.
    """
    k = len(scores)
    if k < 2:
        # numpy would warn and return NaN.
        raise ValueError(f"the standard error needs at least 2 folds, got {k}")
    return float(np.std(scores, ddof=1)) * math.sqrt(1 / k + 1 / (k - 1))


def _weighted_mean(scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
    _check_folds(scores, fold_sizes)
    return float(np.average(scores, weights=fold_sizes))


def _check_folds(scores: Sequence[float], fold_sizes: Sequence[int]) -> None:
    # numpy's own errors here are unclear ("Weights sum to zero",
    # "Axis must be specified when shapes of a and weights differ").
    if len(scores) != len(fold_sizes):
        raise ValueError(
            "scores and fold_sizes must have equal lengths, "
            f"got {len(scores)} and {len(fold_sizes)}"
        )
    if len(scores) == 0:  # not `not scores`: ambiguous for a numpy array
        raise ValueError("at least one fold is required")


class Aggregation(ABC):
    """A rule that combines a trial's fold scores into its value.

    A plain ABC: it holds no data. Subclass it; for a validated setting,
    decorate the subclass as :class:`LowerBound` is.
    """

    @abstractmethod
    def aggregate(self, scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
        """The trial's value, from every fold's score and size."""


# Field-less classes are pydantic dataclasses too, so they print as
# "WeightedMean()" and compare by value, like LowerBound.
@pydantic_dataclass(frozen=True, config=_STRICT)
class WeightedMean(Aggregation):
    """The mean score weighted by fold size."""

    def aggregate(self, scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
        """The size-weighted mean."""
        return _weighted_mean(scores, fold_sizes)


@pydantic_dataclass(frozen=True, config=_STRICT)
class Mean(Aggregation):
    """The unweighted mean of the fold scores."""

    def aggregate(self, scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
        """The unweighted mean."""
        _check_folds(scores, fold_sizes)
        return float(np.mean(scores))


@pydantic_dataclass(frozen=True, config=_STRICT)
class LowerBound(Aggregation):
    """The weighted mean minus ``z`` corrected standard errors.

    ``z=1`` is the one-SE rule; ``z=1.645`` roughly a one-sided 95% bound. The
    SE is the unweighted mean's (:func:`corrected_std_error`), an
    approximation when the fold sizes differ.
    """

    z: Annotated[float, Field(gt=0)] = 1.0

    def aggregate(self, scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
        """The weighted mean minus ``z`` standard errors."""
        return _weighted_mean(scores, fold_sizes) - self.z * corrected_std_error(scores)
