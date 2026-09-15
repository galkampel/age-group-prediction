"""Grouped-multinomial helpers for the independent total/probability model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..modeling_config import IndependentTotalProbabilityConfig


def _grouped_multinomial_rows(
    features: pd.DataFrame, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert building cohorts to at most three positive-weight likelihood rows."""
    values = features.to_numpy(dtype=float)
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (len(features), 3):
        raise ValueError("Grouped cohort counts must have three columns")
    expanded_features = np.repeat(values, 3, axis=0)
    labels = np.tile(np.arange(3), len(features))
    weights = counts.reshape(-1)
    positive = weights > 0
    return expanded_features[positive], labels[positive], weights[positive]


def _fit_grouped_multinomial(
    features: pd.DataFrame,
    counts: np.ndarray,
    *,
    c_value: float,
    config: IndependentTotalProbabilityConfig,
    seed: int,
) -> LogisticRegression:
    """Fit a weighted multinomial age-composition model."""
    grouped_features, labels, weights = _grouped_multinomial_rows(features, counts)
    if weights.size == 0:
        raise ValueError("Probability fitting requires at least one observed child")
    if set(labels) != {0, 1, 2}:
        raise ValueError("Probability fitting requires every age class to be observed")
    estimator = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        tol=config.optimizer_tolerance,
        max_iter=config.optimizer_max_iterations,
        random_state=seed,
    )
    estimator.fit(grouped_features, labels, sample_weight=weights)
    return estimator


def _decision_logits(
    estimator: LogisticRegression, features: pd.DataFrame
) -> np.ndarray:
    """Return validated three-class logits from a fitted multinomial estimator."""
    logits = np.asarray(estimator.decision_function(features.to_numpy(dtype=float)))
    if logits.shape != (len(features), 3):
        raise RuntimeError("Grouped multinomial model did not produce three-class logits")
    return logits


def _multinomial_to_state(estimator: LogisticRegression) -> dict[str, object]:
    """Return what ``decision_function`` reads, as JSON-safe primitives.

    A fitted ``LogisticRegression`` predicts from ``coef_``, ``intercept_`` and
    ``classes_`` alone; the solver settings only mattered while fitting.
    """
    return {
        "coefficients": estimator.coef_.tolist(),
        "intercepts": estimator.intercept_.tolist(),
        "classes": estimator.classes_.tolist(),
    }


def _multinomial_from_state(state: Mapping[str, object]) -> LogisticRegression:
    """Rebuild a fitted estimator by assigning its learned attributes."""
    estimator = LogisticRegression()
    estimator.coef_ = np.asarray(state["coefficients"], dtype=float)
    estimator.intercept_ = np.asarray(state["intercepts"], dtype=float)
    estimator.classes_ = np.asarray(state["classes"])
    estimator.n_features_in_ = estimator.coef_.shape[1]
    return estimator


__all__ = [
    "_decision_logits",
    "_fit_grouped_multinomial",
    "_grouped_multinomial_rows",
    "_multinomial_from_state",
    "_multinomial_to_state",
]