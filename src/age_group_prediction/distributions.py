"""Shared predictive-distribution operations for count modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import nbinom, norm, poisson

# Multinomial probabilities come from a softmax, which sums to one within a few
# ulps. This tolerance only needs to reject a vector that is not a simplex.
_SIMPLEX_SUM_TOLERANCE = 1e-9


def nb2_scipy_parameters(
    mean: np.ndarray, dispersion: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return SciPy's size/probability parameters for an NB2 distribution."""
    mean = np.asarray(mean, dtype=float)
    if not np.isfinite(mean).all() or (mean < 0).any():
        raise ValueError("NB2 means must be finite and nonnegative")
    dispersion = np.asarray(dispersion, dtype=float)
    if not np.isfinite(dispersion).all() or (dispersion <= 0).any():
        raise ValueError("NB2 dispersion must be finite and positive")
    size = 1.0 / dispersion
    probability = size / (size + mean)
    return size, probability


def nb2_torch_parameters(
    mean: Any, concentration: Any
) -> tuple[Any, Any]:
    r"""Return Torch NB2 concentration/logits without importing Torch eagerly.

    Here ``concentration`` is $\phi$ in
    $Var(Y)=\mu+\mu^2/\phi$, the reciprocal of the project's SciPy
    ``dispersion`` parameter.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Torch NB2 parameters require torch, which is a core project "
            "dependency; reinstall the project with `uv sync`."
        ) from error

    mean_tensor = torch.as_tensor(mean)
    concentration_tensor = torch.as_tensor(
        concentration,
        dtype=mean_tensor.dtype,
        device=mean_tensor.device,
    )
    if not mean_tensor.is_floating_point():
        mean_tensor = mean_tensor.to(dtype=torch.float64)
        concentration_tensor = concentration_tensor.to(dtype=torch.float64)
    if not bool(torch.isfinite(mean_tensor).all()) or bool((mean_tensor <= 0).any()):
        raise ValueError("Torch NB2 means must be finite and strictly positive")
    if not bool(torch.isfinite(concentration_tensor).all()) or bool(
        (concentration_tensor <= 0).any()
    ):
        raise ValueError("Torch NB2 concentration must be finite and positive")
    try:
        torch.broadcast_shapes(mean_tensor.shape, concentration_tensor.shape)
    except RuntimeError as error:
        raise ValueError(
            "Torch NB2 mean and concentration must be broadcast-compatible"
        ) from error
    logits = torch.log(mean_tensor) - torch.log(concentration_tensor)
    return concentration_tensor, logits


def pointwise_log_probability(
    family: str,
    observed: np.ndarray,
    mean: np.ndarray,
    *,
    dispersion: float | np.ndarray | None = None,
    scale: float | np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate one supported predictive log density or log mass."""
    observed = np.asarray(observed, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if family == "poisson":
        return np.asarray(poisson.logpmf(observed, mean), dtype=float)
    if family == "normal":
        if scale is None or not np.isfinite(scale).all() or (np.asarray(scale) <= 0).any():
            raise ValueError("Normal scale must be finite and positive")
        return np.asarray(norm.logpdf(observed, loc=mean, scale=scale), dtype=float)
    if family == "nb2":
        if dispersion is None:
            raise ValueError("NB2 log probability requires dispersion")
        size, probability = nb2_scipy_parameters(mean, dispersion)
        return np.asarray(nbinom.logpmf(observed, size, probability), dtype=float)
    raise ValueError(f"Unsupported predictive family: {family}")


def draw_outcomes(
    family: str,
    mean: np.ndarray,
    *,
    rng: np.random.Generator,
    dispersion: float | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Draw independent outcomes from one supported predictive family."""
    mean = np.asarray(mean, dtype=float)
    if family == "poisson":
        return np.asarray(rng.poisson(mean), dtype=float)
    if family == "normal":
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Normal scale must be finite and positive")
        return np.asarray(rng.normal(mean, scale), dtype=float)
    if family == "nb2":
        if dispersion is None:
            raise ValueError("NB2 draws require dispersion")
        size, probability = nb2_scipy_parameters(mean, dispersion)
        return np.asarray(rng.negative_binomial(size, probability), dtype=float)
    raise ValueError(f"Unsupported predictive family: {family}")


def nb2_total_log_probability(
    observed_total: np.ndarray,
    cohort_means: np.ndarray,
    dispersions: np.ndarray,
) -> np.ndarray:
    """Evaluate total-count masses by convolving independent NB2 cohorts."""
    observed_total = np.asarray(observed_total, dtype=float)
    cohort_means = np.asarray(cohort_means, dtype=float)
    dispersions = np.asarray(dispersions, dtype=float)
    if cohort_means.ndim != 2 or cohort_means.shape[0] != len(observed_total):
        raise ValueError("cohort_means must have one row per observed total")
    if dispersions.shape != (cohort_means.shape[1],):
        raise ValueError("dispersions must have one value per cohort")
    if (observed_total < 0).any() or not np.equal(
        observed_total, np.floor(observed_total)
    ).all():
        raise ValueError("Observed totals must be nonnegative integers")

    result = np.empty(len(observed_total), dtype=float)
    for row_index, total_value in enumerate(observed_total.astype(int)):
        support = np.arange(total_value + 1)
        convolved_log_mass = np.array([0.0])
        for cohort_index in range(cohort_means.shape[1]):
            size, probability = nb2_scipy_parameters(
                np.array([cohort_means[row_index, cohort_index]]),
                float(dispersions[cohort_index]),
            )
            log_mass = nbinom.logpmf(support, size, probability.item())
            next_log_mass = np.empty(total_value + 1, dtype=float)
            for subtotal in support:
                previous_support = min(subtotal, len(convolved_log_mass) - 1)
                previous_indices = np.arange(previous_support + 1)
                next_log_mass[subtotal] = logsumexp(
                    convolved_log_mass[previous_indices]
                    + log_mass[subtotal - previous_indices]
                )
            convolved_log_mass = next_log_mass
        result[row_index] = convolved_log_mass[total_value]
    return result


def dirichlet_multinomial_prefix_log_masses(
    counts: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    r"""Return Dirichlet-multinomial log masses for each cumulative cohort prefix.

    Entry ``[..., k]`` is the log mass of observing ``counts[..., :k + 1]`` with
    every later cohort lumped into a single remaining category. Lumping
    categories of a Dirichlet-multinomial yields another Dirichlet-multinomial,
    so each entry is a genuine log mass of the same distribution, and the final
    entry is the full joint log mass.

    Successive differences of these values are conditional log masses, so they
    can be reported per cohort while still summing to the joint score.
    """
    counts = np.asarray(counts, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if counts.shape[-1] != alpha.shape[-1]:
        raise ValueError("counts and alpha must share a cohort axis")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Dirichlet-multinomial counts must be finite and nonnegative")
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Dirichlet-multinomial counts must be integers")
    if not np.isfinite(alpha).all() or (alpha <= 0).any():
        raise ValueError(
            "Dirichlet-multinomial concentration must be finite and positive"
        )

    total = counts.sum(axis=-1, keepdims=True)
    concentration = alpha.sum(axis=-1, keepdims=True)
    # Accumulate the tail concentration by reversed cumulative sum rather than
    # subtracting from the total, which would cancel catastrophically once the
    # leading concentrations dominate.
    reversed_tail = np.cumsum(alpha[..., ::-1], axis=-1)[..., ::-1]
    tail_alpha = np.concatenate(
        (reversed_tail[..., 1:], np.zeros_like(reversed_tail[..., :1])), axis=-1
    )
    tail_counts = total - np.cumsum(counts, axis=-1)

    head = gammaln(total + 1.0) + gammaln(concentration) - gammaln(total + concentration)
    observed_terms = np.cumsum(
        gammaln(counts + alpha) - gammaln(alpha) - gammaln(counts + 1.0), axis=-1
    )
    # The final prefix has no remaining category, so it contributes nothing.
    remainder = np.zeros_like(observed_terms)
    remainder[..., :-1] = (
        gammaln(tail_counts[..., :-1] + tail_alpha[..., :-1])
        - gammaln(tail_alpha[..., :-1])
        - gammaln(tail_counts[..., :-1] + 1.0)
    )
    return head + observed_terms + remainder


def dirichlet_multinomial_log_probability(
    counts: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Return the joint Dirichlet-multinomial log mass of observed cohort counts."""
    return dirichlet_multinomial_prefix_log_masses(counts, alpha)[..., -1]


def multinomial_prefix_log_masses(
    counts: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Return multinomial log masses for each cumulative cohort prefix.

    This is the multinomial counterpart of
    :func:`dirichlet_multinomial_prefix_log_masses`, so the two composition
    likelihoods decompose per cohort in the same way and their per-cohort
    scores are directly comparable.
    """
    counts = np.asarray(counts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if counts.shape[-1] != probabilities.shape[-1]:
        raise ValueError("counts and probabilities must share a cohort axis")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Multinomial counts must be finite and nonnegative")
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Multinomial counts must be integers")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Multinomial probabilities must be finite and nonnegative")
    if not np.allclose(
        probabilities.sum(axis=-1), 1.0, rtol=0.0, atol=_SIMPLEX_SUM_TOLERANCE
    ):
        raise ValueError("Multinomial probabilities must sum to one over cohorts")

    total = counts.sum(axis=-1, keepdims=True)
    reversed_tail = np.cumsum(probabilities[..., ::-1], axis=-1)[..., ::-1]
    tail_probabilities = np.concatenate(
        (reversed_tail[..., 1:], np.zeros_like(reversed_tail[..., :1])), axis=-1
    )
    tail_counts = total - np.cumsum(counts, axis=-1)

    # ``xlogy`` scores a zero count as 0 even against a zero probability, and a
    # positive count against a zero probability as -inf. Clipping the
    # probability away from zero would instead cap that penalty at a finite
    # value and hide an impossible outcome from any NLL comparison.
    head = gammaln(total + 1.0)
    observed_terms = np.cumsum(
        xlogy(counts, probabilities) - gammaln(counts + 1.0), axis=-1
    )
    # The final prefix has no remaining category, so it contributes nothing.
    remainder = np.zeros_like(observed_terms)
    remainder[..., :-1] = xlogy(
        tail_counts[..., :-1], tail_probabilities[..., :-1]
    ) - gammaln(tail_counts[..., :-1] + 1.0)
    return head + observed_terms + remainder


def nb2_gradient_hessian(
    observed: np.ndarray,
    raw_prediction: np.ndarray,
    *,
    dispersion: float,
    raw_score_bounds: tuple[float, float] = (-30.0, 30.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return NB2 NLL derivatives for the log-mean raw score."""
    observed = np.asarray(observed, dtype=float)
    raw_prediction = np.asarray(raw_prediction, dtype=float)
    if (observed < 0).any() or not np.equal(observed, np.floor(observed)).all():
        raise ValueError("NB2 targets must be nonnegative integers")
    if not np.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("NB2 dispersion must be finite and positive")
    mean = np.exp(np.clip(raw_prediction, *raw_score_bounds))
    denominator = 1.0 + dispersion * mean
    gradient = (mean - observed) / denominator
    hessian = mean * (1.0 + dispersion * observed) / denominator**2
    return gradient, hessian


__all__ = [
    "dirichlet_multinomial_log_probability",
    "dirichlet_multinomial_prefix_log_masses",
    "draw_outcomes",
    "multinomial_prefix_log_masses",
    "nb2_gradient_hessian",
    "nb2_scipy_parameters",
    "nb2_torch_parameters",
    "nb2_total_log_probability",
    "pointwise_log_probability",
]