"""Post-hoc temperature scaling of cohort logits, fitted on out-of-fold logits."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

__all__ = ["TemperatureCalibrator"]


class TemperatureCalibrator(BaseEstimator):
    """Map cohort logits to probabilities ``softmax(logits / T)``, with ``T`` fitted.

    The one parameter ``T`` is fitted on **out-of-fold** logits of an already
    fitted :class:`~age_group_prediction.modeling.CohortProbabilityModel`
    (``predict_logits``) and the cohort counts of the same buildings; the
    model itself is left as is, as scikit-learn's
    ``CalibratedClassifierCV(method="temperature")`` does. ``fit`` minimizes
    the negative log probability of each building's observed composition
    ``s_b = n_b / Σ_k n_bk``, averaged over buildings (as the Dirichlet fit
    weighs them), ``−mean_b Σ_k s_bk log softmax(logits_b / T)_k``, in log
    space so that a sharp logit never underflows to a zero probability, over
    ``log(1/T)`` in (−10, 10). The fitted value is always used (no gate); on logits that
    carry nothing it grows large and the probabilities flatten.

    No settings: ``sklearn.base.clone`` returns an unfitted copy, so a model
    holding a fitted calibrator uses it as given. Rows of ``logits`` and
    ``counts`` are paired by position; the counts are nonnegative, with at
    least one child per building, as validated where the data is prepared.
    """

    def fit(
        self, logits: pd.DataFrame | np.ndarray, counts: pd.DataFrame | np.ndarray
    ) -> Self:
        """Fit ``T`` on out-of-fold ``logits`` and the cohort ``counts`` of the same rows."""
        logit_values = np.asarray(logits, dtype=float)
        count_values = np.asarray(counts, dtype=float)
        # numpy would broadcast one column over every cohort; two DataFrames
        # with their cohorts in different orders would pair them wrongly.
        if logit_values.shape != count_values.shape:
            raise ValueError(
                "logits and counts must both be (buildings, cohorts) arrays of the "
                f"same shape, got {logit_values.shape} and {count_values.shape}"
            )
        if (
            isinstance(logits, pd.DataFrame)
            and isinstance(counts, pd.DataFrame)
            and list(logits.columns) != list(counts.columns)
        ):
            raise ValueError(
                f"logits' cohorts {list(logits.columns)} differ from counts' "
                f"{list(counts.columns)}; give both in the same order"
            )

        shares = count_values / count_values.sum(axis=1, keepdims=True)

        def objective(log_inverse_temperature: float) -> float:
            scaled = logit_values * np.exp(log_inverse_temperature)
            return float(-(shares * log_softmax(scaled, axis=1)).sum(axis=1).mean())

        # scikit-learn's search for its own temperature scaling.
        result = minimize_scalar(
            objective,
            bounds=(-10.0, 10.0),
            method="bounded",
            options={"xatol": 64 * np.finfo(float).eps},
        )
        # A search that stopped early (scipy also reports a nan objective this
        # way) would otherwise hand over an arbitrary temperature.
        if not result.success:
            raise RuntimeError(
                f"the temperature search failed ({result.message}); check the "
                "counts and the logits"
            )
        self.temperature_ = float(np.exp(-result.x))
        return self

    def predict(self, logits: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """``softmax(logits / T)``, one column per cohort, with the logits' columns and index."""
        check_is_fitted(self)
        return pd.DataFrame(
            softmax(np.asarray(logits, dtype=float) / self.temperature_, axis=1),
            columns=getattr(logits, "columns", None),
            index=getattr(logits, "index", None),
        )
