"""Model 2: a building's predicted total times its predicted cohort shares."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from sklearn.utils.validation import check_is_fitted

from .base import BaseAgeGroupModel
from .calibration import TemperatureCalibrator

__all__ = ["IndependentTotalProbabilityModel"]


class IndependentTotalProbabilityModel(BaseAgeGroupModel):
    """Predict each cohort as ``total × share``: two independent models, multiplied.

    ``total_children_model`` predicts a building's total children from the
    raw table (the row sum of ``y`` at fit); ``cohort_probability_model``
    predicts its cohort shares (``y``'s columns). Both are usually a
    :class:`~age_group_prediction.modeling.ModelPipeline`, each with its own
    features; their settings are reached by nested ``set_params`` names
    (``cohort_probability_model__model__l2_penalty``), before ``fit``, which
    fits copies. The same ``exposure`` goes to both; the probability model
    has no offset and ignores it.

    ``temperature_calibrator`` is an already fitted
    :class:`~age_group_prediction.modeling.TemperatureCalibrator` (on
    out-of-fold logits, by the caller), applied to the probability model's
    logits at ``predict``; without one the shares are the model's own. It is
    used as given, and ``sklearn.base.clone`` drops a fit: wrap it in
    ``sklearn.frozen.FrozenEstimator`` when this model is cloned, e.g. by a
    tuner. ``predict`` uses the calibrator ``fit`` was given; one set
    afterwards reaches the next ``fit``.

    Rows are paired by position. ``predict`` returns a DataFrame with ``y``'s
    columns at fit, indexed like ``X``.
    """

    def __init__(
        self,
        *,
        total_children_model: BaseAgeGroupModel,
        cohort_probability_model: BaseAgeGroupModel,
        temperature_calibrator: TemperatureCalibrator | FrozenEstimator | None = None,
    ) -> None:
        # Stored verbatim: set_params assigns attributes without re-entering here.
        self.total_children_model = total_children_model
        self.cohort_probability_model = cohort_probability_model
        self.temperature_calibrator = temperature_calibrator

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        exposure: ArrayLike | None = None,
    ) -> Self:
        """Fit copies of both models on the raw table ``X`` and the cohort counts ``y``."""
        cohorts = list(y.columns)
        total_children_model = clone(self.total_children_model).fit(
            X, y.sum(axis=1), exposure=exposure
        )
        cohort_probability_model = clone(self.cohort_probability_model).fit(
            X, y, exposure=exposure
        )
        # Set together, only once both succeeded.
        self.total_children_model_: BaseAgeGroupModel = total_children_model
        self.cohort_probability_model_: BaseAgeGroupModel = cohort_probability_model
        self.cohorts_: list[object] = cohorts
        # The fitted state is complete: predict follows it, not a later set_params.
        self.temperature_calibrator_ = self.temperature_calibrator
        return self

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> pd.DataFrame:
        """The predicted mean of every cohort for each row of ``X``."""
        check_is_fitted(self)
        if self.temperature_calibrator_ is None:
            shares = pd.DataFrame(
                self.cohort_probability_model_.predict(X, exposure=exposure)
            )
        else:
            shares = self.temperature_calibrator_.predict(
                self.cohort_probability_model_.predict_logits(X)  # type: ignore[attr-defined]
            )
        # A probability model with other cohorts would be combined silently.
        if list(shares.columns) != self.cohorts_:
            raise ValueError(
                f"the probability model predicts cohorts {list(shares.columns)}, "
                f"but y's columns at fit were {self.cohorts_}"
            )
        # As arrays: a prediction with its own index would be realigned to X's,
        # into NaN, silently.
        total = np.asarray(self.total_children_model_.predict(X, exposure=exposure))
        return pd.DataFrame(
            total[:, None] * np.asarray(shares), columns=self.cohorts_, index=X.index
        )
