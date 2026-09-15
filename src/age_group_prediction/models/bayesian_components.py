"""Pyro model definitions and prior simulation for the Bayesian model."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive

from ..modeling_config import BayesianConditionalConfig


class PriorPredictiveSummary(TypedDict):
    """Plausibility summary returned by ``_run_prior_predictive``."""

    seed: int
    draws: int
    children_per_apartment_quantiles: list[float]
    expected_cohort_shares: list[float]
    expected_dominant_share: float
    concentration_quantile: float
    cohort_share_quantiles: list[list[float]]
    violations: list[str]


def _total_model(
    features: Any,
    log_exposure: Any,
    neighborhood_index: Any,
    neighborhood_count: int,
    priors: Any,
    observed: Any | None = None,
    sample_outcome: bool = True,
) -> None:
    """Define the hierarchical NB2 model for building-level child totals.

    The model samples fixed and non-centered neighborhood effects, records the
    mean and concentration, and either conditions on totals or generates them.
    """
    coefficient_count = features.shape[1]
    intercept = pyro.sample(
        "total_intercept",
        dist.Normal(priors.total_intercept_loc, priors.total_intercept_scale),
    )
    coefficients = pyro.sample(
        "total_coefficients",
        dist.Normal(
            torch.zeros(coefficient_count, dtype=torch.float64),
            priors.total_coefficient_scale,
        ).to_event(1),
    )
    # log(\phi)- dispersion parameter for the Negative Binomial distribution
    log_concentration = pyro.sample(
        "log_total_concentration",
        dist.Normal(priors.dispersion_log_loc, priors.dispersion_log_scale),
    )
    # non-centered parameterization for neighborhood effects
    neighborhood_scale = pyro.sample(
        "neighborhood_scale", dist.HalfNormal(priors.neighborhood_scale)
    )
    neighborhood_raw = pyro.sample(
        "neighborhood_raw",
        dist.Normal(
            torch.zeros(neighborhood_count, dtype=torch.float64), 1.0
        ).to_event(1),
    )
    log_mean = (
        log_exposure
        + intercept
        + features @ coefficients
        + neighborhood_scale * neighborhood_raw[neighborhood_index]
    )
    concentration = torch.exp(log_concentration)
    pyro.deterministic("total_log_mean", log_mean)
    pyro.deterministic("total_concentration", concentration)
    if not sample_outcome:
        return
    if observed is not None:
        log_denominator = torch.logaddexp(log_concentration, log_mean)
        log_probability = (
            torch.lgamma(observed + concentration)
            - torch.lgamma(concentration)
            - torch.lgamma(observed + 1.0)
            + concentration * (log_concentration - log_denominator)
            + observed * (log_mean - log_denominator)
        )
        pyro.factor("total_count_log_likelihood", log_probability.sum())
        return
    # total_counts- total number of failures until the experiment is stopped
    # logits- log(p/ (1-p)) for the Negative Binomial distribution
    pyro.sample(
        "total_count",
        dist.NegativeBinomial(
            total_count=concentration,
            logits=log_mean - log_concentration,
        ).to_event(1),
        obs=observed,
    )


def _composition_model(
    features: Any,
    totals: Any,
    cohort_count: int,
    priors: Any,
    observed: Any | None = None,
    sample_outcome: bool = True,
) -> None:
    """Define cohort composition conditional on each building's total count.

    A reference-category softmax supplies probabilities to a
    Dirichlet-multinomial likelihood or generative outcome site.
    """
    coefficient_count = features.shape[1]
    modeled_cohorts = cohort_count - 1
    intercepts = pyro.sample(
        "composition_intercepts",
        dist.Normal(
            torch.zeros(modeled_cohorts, dtype=torch.float64),
            priors.composition_intercept_scale,
        ).to_event(1),
    )
    coefficients = pyro.sample(
        "composition_coefficients",
        dist.Normal(
            torch.zeros(
                (coefficient_count, modeled_cohorts), dtype=torch.float64
            ),
            priors.composition_coefficient_scale,
        ).to_event(2),
    )
    log_kappa = pyro.sample(
        "log_composition_concentration",
        dist.Normal(priors.kappa_log_loc, priors.kappa_log_scale),
    )
    reference = torch.zeros((*features.shape[:-1], 1), dtype=torch.float64)
    # reference category for the softmax
    logits = torch.cat((intercepts + features @ coefficients, reference), dim=-1)
    probabilities = torch.exp(
        logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    )
    pyro.deterministic("composition_probabilities", probabilities)
    if not sample_outcome:
        return
    pyro.sample(
        "cohort_count",
        dist.DirichletMultinomial(
            concentration=torch.exp(log_kappa) * probabilities,
            total_count=totals,
        ).to_event(1),
        obs=observed,
    )


def _run_prior_predictive(
    total_args: tuple[object, ...],
    composition_features: Any,
    cohort_count: int,
    *,
    config: BayesianConditionalConfig,
    seed: int,
) -> PriorPredictiveSummary:
    """Simulate from both prior stages and summarize plausibility checks.

    Totals use the exact Gamma-Poisson NB2 representation, and cohort counts
    use the equivalent Dirichlet-then-multinomial construction.
    """
    pyro.set_rng_seed(seed)
    draws = config.prior_predictive.draws
    total_samples = Predictive(
        _total_model,
        num_samples=draws,
        return_sites=(
            "total_log_mean",
            "total_concentration",
            "neighborhood_scale",
        ),
    )(*total_args[:-1], None, False)
    log_means = (
        total_samples["total_log_mean"].reshape(draws, -1).detach().cpu().numpy()
    )
    total_concentration = (
        total_samples["total_concentration"].reshape(draws).detach().cpu().numpy()
    )
    if not np.isfinite(log_means).all():
        raise RuntimeError("Prior-predictive total log means must be finite")
    outcome_rng = np.random.default_rng(seed)
    means = np.exp(log_means)
    latent_rates = outcome_rng.gamma(
        shape=total_concentration[:, None],
        scale=means / total_concentration[:, None],
    )
    sampled_totals = outcome_rng.poisson(latent_rates)
    composition_samples = Predictive(
        _composition_model,
        num_samples=draws,
        return_sites=(
            "composition_probabilities",
            "log_composition_concentration",
        ),
    )(
        composition_features,
        torch.as_tensor(sampled_totals, dtype=torch.float64),
        cohort_count,
        config.priors,
        None,
        False,
    )
    probabilities = (
        composition_samples["composition_probabilities"]
        .reshape(draws, *sampled_totals.shape[1:], cohort_count)
        .detach()
        .cpu()
        .numpy()
    )
    composition_concentration = np.exp(
        composition_samples["log_composition_concentration"]
        .reshape(draws)
        .detach()
        .cpu()
        .numpy()
    )
    sampled_cohorts = np.empty((*sampled_totals.shape, cohort_count), dtype=float)
    for draw_index in range(draws):
        for row_index in range(sampled_totals.shape[1]):
            total_count = int(sampled_totals[draw_index, row_index])
            if total_count == 0:
                sampled_cohorts[draw_index, row_index] = 0
                continue
            concentration = (
                composition_concentration[draw_index]
                * probabilities[draw_index, row_index]
            )
            shares = outcome_rng.dirichlet(concentration)
            sampled_cohorts[draw_index, row_index] = outcome_rng.multinomial(
                total_count, shares
            )
    if not np.isfinite(sampled_totals).all() or (sampled_totals < 0).any():
        raise RuntimeError("Prior-predictive totals must be finite and nonnegative")
    if not np.isfinite(sampled_cohorts).all() or (sampled_cohorts < 0).any():
        raise RuntimeError("Prior-predictive cohort counts must be finite and nonnegative")
    if not np.array_equal(sampled_cohorts.sum(axis=-1), sampled_totals):
        raise RuntimeError("Prior-predictive cohort counts must reconcile exactly")
    if not np.isfinite(total_concentration).all() or (total_concentration <= 0).any():
        raise RuntimeError("Prior-predictive total concentration must be positive")
    if not np.isfinite(composition_concentration).all() or (
        composition_concentration <= 0
    ).any():
        raise RuntimeError("Prior-predictive composition concentration must be positive")
    exposure = torch.exp(total_args[1]).detach().cpu().numpy()
    rates = sampled_totals / exposure
    positive = sampled_totals > 0
    if not positive.any():
        raise RuntimeError("Prior prediction must produce at least one positive total")
    share_values = sampled_cohorts[positive] / sampled_totals[positive, None]
    rate_quantiles = np.quantile(rates, (0.05, 0.5, 0.95)).tolist()
    share_quantiles = np.quantile(
        share_values, (0.05, 0.5, 0.95), axis=0
    ).T.tolist()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Prior-predictive composition probabilities must be finite")
    # Prior expected composition, averaged over draws and rows. Unlike quantiles
    # of realized shares, this is insensitive to the legitimate Dirichlet
    # dispersion implied by the concentration prior, so it measures only whether
    # the priors systematically favour or exclude a cohort.
    expected_shares = probabilities.reshape(-1, cohort_count).mean(axis=0)
    uniform_share = 1.0 / cohort_count
    minimum_expected = (
        config.prior_predictive.minimum_expected_share_ratio * uniform_share
    )
    maximum_expected = (
        config.prior_predictive.maximum_expected_share_ratio * uniform_share
    )
    # Averaging probabilities over draws hides dispersion: zero-mean logit
    # priors keep the average near uniform however diffuse they are. These two
    # statistics are the ones with power over a degenerate composition prior.
    # The dominant share responds to diffuse logits driving single buildings to
    # a simplex vertex, and the concentration quantile responds to a kappa
    # prior that makes individual buildings effectively single-cohort.
    expected_dominant_share = float(probabilities.max(axis=-1).mean())
    concentration_quantile = float(np.quantile(composition_concentration, 0.05))
    violations: list[str] = []
    if rate_quantiles[-1] > config.prior_predictive.maximum_children_per_apartment:
        violations.append(
            "95th percentile children per apartment exceeds configured maximum"
        )
    if (expected_shares < minimum_expected).any() or (
        expected_shares > maximum_expected
    ).any():
        violations.append(
            "prior systematically favors or excludes a cohort: expected shares "
            f"{np.round(expected_shares, 4).tolist()} fall outside "
            f"[{minimum_expected:.4f}, {maximum_expected:.4f}]"
        )
    if expected_dominant_share > config.prior_predictive.maximum_expected_dominant_share:
        violations.append(
            "prior drives buildings toward a single cohort: expected dominant "
            f"share {expected_dominant_share:.4f} exceeds "
            f"{config.prior_predictive.maximum_expected_dominant_share}"
        )
    if concentration_quantile < config.prior_predictive.minimum_concentration_quantile:
        violations.append(
            "prior composition concentration is too small: 5th percentile "
            f"{concentration_quantile:.4g} is below "
            f"{config.prior_predictive.minimum_concentration_quantile}"
        )
    return {
        "seed": seed,
        "draws": draws,
        "children_per_apartment_quantiles": rate_quantiles,
        "expected_cohort_shares": expected_shares.tolist(),
        "expected_dominant_share": expected_dominant_share,
        "concentration_quantile": concentration_quantile,
        "cohort_share_quantiles": share_quantiles,
        "violations": violations,
    }


__all__ = [
    "PriorPredictiveSummary",
    "_composition_model",
    "_run_prior_predictive",
    "_total_model",
]
