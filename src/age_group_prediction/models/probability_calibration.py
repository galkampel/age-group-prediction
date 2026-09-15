"""Composition loss kernels and temperature calibration for the independent model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from scipy.stats import chi2

from ..metrics import composition_log_loss_from_log_probabilities
from ..modeling_config import IndependentTotalProbabilityConfig


def _weighted_composition_nll(logits: np.ndarray, counts: np.ndarray) -> float:
    """Calculate child-count-weighted multinomial negative log likelihood."""
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    return composition_log_loss_from_log_probabilities(counts, log_probabilities)


def _weighted_proportion_squared_error(
    probabilities: np.ndarray, counts: np.ndarray
) -> float:
    """Calculate child-weighted squared error between predicted and observed shares.

    This building-level share error differs from the per-child Brier score in
    `metrics.composition_brier_score`.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1)
    positive = totals > 0
    observed_probabilities = counts[positive] / totals[positive, None]
    squared_error = np.sum(
        (probabilities[positive] - observed_probabilities) ** 2, axis=1
    )
    return float(np.average(squared_error, weights=totals[positive]))


def _fit_calibration_temperature(
    logits: np.ndarray,
    counts: np.ndarray,
    *,
    config: IndependentTotalProbabilityConfig,
) -> tuple[float, dict[str, object]]:
    """Fit one softmax temperature to cross-fitted logits and record the evidence.

    Returns the temperature to apply, which is the fitted one only when the
    optimizer succeeded and a likelihood-ratio test rejects ``T = 1``, and
    ``1.0`` otherwise, together with the calibration diagnostics.
    """
    raw_nll = _weighted_composition_nll(logits, counts)
    lower, upper = np.log(config.calibration_temperature_bounds)
    result = minimize_scalar(
        lambda log_temperature: _weighted_composition_nll(
            logits / np.exp(log_temperature), counts
        ),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": config.optimizer_tolerance},
    )
    fitted_temperature = float(np.exp(result.x))
    fitted_nll = float(result.fun)
    # Retention is decided by a likelihood-ratio test at one degree of
    # freedom (the temperature), not by a raw improvement threshold.
    # T = 1 lies strictly inside `calibration_temperature_bounds`, so
    # `fitted_nll <= raw_nll` holds by construction and the comparison is
    # made on the same cross-fitted rows T was fitted on. A plain
    # improvement threshold therefore accepts noise: on perfectly
    # calibrated logits, where T = 1 is correct, the former 1e-6 rule
    # retained 143 of 150 replicates while this test retained 8 of 150,
    # matching its nominal 5% size.
    #
    # `_weighted_composition_nll` returns a per-child mean, so the
    # deviance statistic is 2 * (number of children) * (raw - fitted).
    child_count = float(counts.sum())
    likelihood_ratio_statistic = 2.0 * child_count * (raw_nll - fitted_nll)
    critical_value = float(chi2.ppf(1.0 - config.calibration_significance_level, 1))
    retained = bool(result.success and likelihood_ratio_statistic > critical_value)
    temperature = fitted_temperature if retained else 1.0
    calibrated_probabilities = softmax(logits / temperature, axis=1)
    diagnostics: dict[str, object] = {
        "method": "evidence-gated multiclass temperature scaling",
        "temperature": temperature,
        "fitted_temperature": fitted_temperature,
        "retained": bool(retained),
        "retention_rule": "likelihood_ratio_test_1df",
        "likelihood_ratio_statistic": likelihood_ratio_statistic,
        "likelihood_ratio_critical_value": critical_value,
        "likelihood_ratio_p_value": float(chi2.sf(likelihood_ratio_statistic, 1)),
        "calibration_significance_level": config.calibration_significance_level,
        "weighted_child_count": child_count,
        "raw_weighted_nll": raw_nll,
        "selected_weighted_nll": _weighted_composition_nll(
            logits / temperature, counts
        ),
        # Named for what it measures: a building-level weighted squared
        # error between predicted and observed shares. It is NOT the
        # per-child Brier score in `metrics.composition_brier_score`.
        "raw_weighted_share_error": _weighted_proportion_squared_error(
            softmax(logits, axis=1), counts
        ),
        "selected_weighted_share_error": _weighted_proportion_squared_error(
            calibrated_probabilities, counts
        ),
        "validation_rows": len(logits),
        "zero_total_validation_rows": int((counts.sum(axis=1) == 0).sum()),
    }
    return temperature, diagnostics


__all__ = [
    "_fit_calibration_temperature",
    "_weighted_composition_nll",
    "_weighted_proportion_squared_error",
]
