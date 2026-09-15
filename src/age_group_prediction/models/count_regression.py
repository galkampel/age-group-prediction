"""Count-regression helpers for the independent total/probability model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from ..modeling_config import (
    IndependentTotalFamily,
    IndependentTotalProbabilityConfig,
)

_RAW_SCORE_BOUNDS = (-30.0, 30.0)


@dataclass(frozen=True)
class _NB2Fit:
    coefficients: np.ndarray
    dispersion: float
    objective_value: float


@dataclass(frozen=True)
class _PoissonFit:
    coefficients: np.ndarray
    objective_value: float


_TotalFit = _NB2Fit | _PoissonFit


def _design_matrix(features: pd.DataFrame) -> np.ndarray:
    """Add an intercept column to numeric feature values."""
    values = features.to_numpy(dtype=float)
    return np.column_stack((np.ones(len(values)), values))


def _nb2_log_probability(
    observed: np.ndarray, mean: np.ndarray, dispersion: float
) -> np.ndarray:
    """Compute NB2 log probabilities under the project's alpha parameterization."""
    observed = np.asarray(observed, dtype=float)
    mean = np.asarray(mean, dtype=float)
    size = 1.0 / dispersion
    return (
        gammaln(observed + size)
        - gammaln(size)
        - gammaln(observed + 1.0)
        + size * (np.log(size) - np.log(size + mean))
        + observed * (np.log(mean) - np.log(size + mean))
    )


def _fit_nb2_regression(
    features: pd.DataFrame,
    observed: np.ndarray,
    log_exposure: pd.Series,
    *,
    l2_penalty: float,
    config: IndependentTotalProbabilityConfig,
) -> _NB2Fit:
    """Fit penalized NB2 total-count regression with a fixed exposure offset."""
    design = _design_matrix(features)
    observed = np.asarray(observed, dtype=float)
    offset = log_exposure.to_numpy(dtype=float)
    exposure = np.exp(offset)
    initial_rate = max(float(observed.sum() / exposure.sum()), config.minimum_mean)
    initial = np.zeros(design.shape[1] + 1, dtype=float)
    initial[0] = np.log(initial_rate)
    initial[-1] = np.log(np.sqrt(np.prod(config.dispersion_bounds)))

    def objective(parameters: np.ndarray) -> float:
        coefficients = parameters[:-1]
        dispersion = float(np.exp(parameters[-1]))
        raw_mean = offset + design @ coefficients
        mean = np.exp(np.clip(raw_mean, *_RAW_SCORE_BOUNDS))
        nll = -float(np.mean(_nb2_log_probability(observed, mean, dispersion)))
        return nll + 0.5 * l2_penalty * float(coefficients[1:] @ coefficients[1:])

    lower, upper = np.log(config.dispersion_bounds)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(None, None)] * design.shape[1] + [(lower, upper)],
        options={
            "maxiter": config.optimizer_max_iterations,
            "ftol": config.optimizer_tolerance,
        },
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"NB2 optimization failed: {result.message}")
    return _NB2Fit(
        coefficients=np.asarray(result.x[:-1], dtype=float),
        dispersion=float(np.exp(result.x[-1])),
        objective_value=float(result.fun),
    )


def _fit_poisson_regression(
    features: pd.DataFrame,
    observed: np.ndarray,
    log_exposure: pd.Series,
    *,
    l2_penalty: float,
    config: IndependentTotalProbabilityConfig,
) -> _PoissonFit:
    """Fit penalized Poisson total-count regression with a fixed exposure offset."""
    design = _design_matrix(features)
    observed = np.asarray(observed, dtype=float)
    offset = log_exposure.to_numpy(dtype=float)
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = np.log(
        max(float(observed.sum() / np.exp(offset).sum()), config.minimum_mean)
    )

    def objective(coefficients: np.ndarray) -> float:
        mean = np.exp(np.clip(offset + design @ coefficients, *_RAW_SCORE_BOUNDS))
        log_probability = observed * np.log(mean) - mean - gammaln(observed + 1.0)
        return -float(np.mean(log_probability)) + 0.5 * l2_penalty * float(
            coefficients[1:] @ coefficients[1:]
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        options={
            "maxiter": config.optimizer_max_iterations,
            "ftol": config.optimizer_tolerance,
        },
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Poisson optimization failed: {result.message}")
    return _PoissonFit(
        coefficients=np.asarray(result.x, dtype=float),
        objective_value=float(result.fun),
    )


def _fit_total_regression(
    family: IndependentTotalFamily,
    features: pd.DataFrame,
    observed: np.ndarray,
    log_exposure: pd.Series,
    *,
    l2_penalty: float,
    config: IndependentTotalProbabilityConfig,
) -> _TotalFit:
    """Fit the explicitly configured total-count likelihood."""
    if family == "poisson":
        return _fit_poisson_regression(
            features,
            observed,
            log_exposure,
            l2_penalty=l2_penalty,
            config=config,
        )
    return _fit_nb2_regression(
        features,
        observed,
        log_exposure,
        l2_penalty=l2_penalty,
        config=config,
    )


def _predict_total_mean(
    fit: _TotalFit,
    features: pd.DataFrame,
    log_exposure: pd.Series,
    *,
    minimum_mean: float,
) -> np.ndarray:
    """Predict total means for either supported count family."""
    raw_mean = (
        log_exposure.to_numpy(dtype=float) + _design_matrix(features) @ fit.coefficients
    )
    return np.clip(np.exp(np.clip(raw_mean, *_RAW_SCORE_BOUNDS)), minimum_mean, None)


def _total_fit_to_state(fit: _TotalFit) -> dict[str, object]:
    """Return a fitted total regression as JSON-safe primitives."""
    state: dict[str, object] = {
        "family": "nb2" if isinstance(fit, _NB2Fit) else "poisson",
        "coefficients": fit.coefficients.tolist(),
        "objective_value": fit.objective_value,
    }
    if isinstance(fit, _NB2Fit):
        state["dispersion"] = fit.dispersion
    return state


def _total_fit_from_state(state: Mapping[str, object]) -> _TotalFit:
    """Rebuild a fitted total regression without re-running the optimizer."""
    coefficients = np.asarray(state["coefficients"], dtype=float)
    objective_value = float(state["objective_value"])
    if state["family"] == "nb2":
        return _NB2Fit(
            coefficients=coefficients,
            dispersion=float(state["dispersion"]),
            objective_value=objective_value,
        )
    if state["family"] == "poisson":
        return _PoissonFit(coefficients=coefficients, objective_value=objective_value)
    raise ValueError(f"Unknown total regression family: {state['family']!r}")


__all__ = [
    "_NB2Fit",
    "_PoissonFit",
    "_TotalFit",
    "_design_matrix",
    "_fit_nb2_regression",
    "_fit_poisson_regression",
    "_fit_total_regression",
    "_nb2_log_probability",
    "_predict_total_mean",
    "_total_fit_from_state",
    "_total_fit_to_state",
]