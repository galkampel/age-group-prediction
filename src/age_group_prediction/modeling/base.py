"""The contract every age-group model shares: fit, predict and evaluate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_consistent_length

from ..scoring import Metric

__all__ = ["BaseAgeGroupModel"]


class BaseAgeGroupModel(BaseEstimator, ABC):
    """A scikit-learn-style model of count targets.

    ``y`` is one count target (a Series) or one per cohort (a DataFrame, one
    column each).

    Subclasses take their settings in ``__init__`` and store them verbatim, so
    ``get_params`` and ``set_params`` (inherited from ``BaseEstimator``) and
    ``sklearn.base.clone`` work, and a tuner can build a fresh model per trial.
    Data are only ever method arguments (``X``, ``y``, and ``exposure`` where a
    model uses one), and fitted state lives in trailing-underscore attributes.
    ``fit`` replaces all fitted state, so a refit equals a fresh fit: a tuner
    reuses one copy across the folds of a trial.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        exposure: ArrayLike | None = None,
    ) -> Self:
        """Learn from the rows of ``X`` and their targets ``y``.

        ``exposure`` is each row's raw exposure, not its log. A model whose
        setting gives it an offset raises if it is missing, rather than
        silently dropping the offset; any other model ignores it. So a caller
        holding several models passes the same exposure to each.
        """

    @abstractmethod
    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> np.ndarray | pd.DataFrame:
        """The predicted mean for each row of ``X``; ``exposure`` as in :meth:`fit`.

        An array for one target; a DataFrame with ``y``'s columns for several.
        """

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, metric: Metric) -> float:
        """Score ``y_pred`` against ``y_true`` with one metric.

        A method rather than a free function so a model can change how it is
        scored. The cast turns a custom metric's numpy scalar into a float.
        """
        return float(metric.function(y_true, y_pred))

    @staticmethod
    def _check_exposure(
        X: pd.DataFrame, exposure: ArrayLike | None, *, expected: bool
    ) -> np.ndarray | None:
        """Return the exposure as floats, or ``None`` when none is ``expected``.

        The rule of :meth:`fit`, for a model with an offset: ``expected`` is
        its setting at fit and its fitted state at predict. Only what would
        otherwise pass silently is checked; the values are validated where the
        data is prepared, by
        :class:`~age_group_prediction.preprocessing.ExposureTransformer`.
        """
        if not expected:
            return None
        # A forgotten exposure would silently drop the offset.
        if exposure is None:
            raise ValueError(
                "the model uses an exposure offset (use_exposure=True at fit); "
                "pass `exposure`"
            )
        exposure_values = np.asarray(exposure, dtype=float)
        # A column (n, 1), e.g. a one-column DataFrame, would broadcast against
        # the (n,) linear predictor into an (n, n) result.
        if exposure_values.ndim != 1:
            raise ValueError(
                f"exposure must be one-dimensional, got shape {exposure_values.shape}"
            )
        # Rows are paired by position, and numpy broadcasts a length-1 exposure
        # to every row silently.
        check_consistent_length(X, exposure_values)
        return exposure_values
