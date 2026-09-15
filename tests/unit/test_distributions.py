"""Tests for shared predictive-distribution operations."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import binom, dirichlet_multinomial, multinomial, nbinom

from age_group_prediction.distributions import (
    dirichlet_multinomial_log_probability,
    dirichlet_multinomial_prefix_log_masses,
    multinomial_prefix_log_masses,
    nb2_gradient_hessian,
    nb2_scipy_parameters,
    nb2_torch_parameters,
    nb2_total_log_probability,
)


def test_nb2_parameterization_has_declared_moments() -> None:
    mean = np.array([2.0, 5.0])
    dispersion = 0.4
    size, probability = nb2_scipy_parameters(mean, dispersion)

    np.testing.assert_allclose(nbinom.mean(size, probability), mean)
    np.testing.assert_allclose(
        nbinom.var(size, probability), mean + dispersion * mean**2
    )


def test_torch_nb2_parameterization_matches_scipy() -> None:
    pytest.importorskip("torch")
    script = """
import numpy as np
import torch
from scipy.stats import nbinom
from age_group_prediction.distributions import nb2_scipy_parameters, nb2_torch_parameters
mean = torch.tensor([0.5, 2.0, 7.0], dtype=torch.float64)
concentration = torch.tensor(2.5, dtype=torch.float64)
total_count, logits = nb2_torch_parameters(mean, concentration)
distribution = torch.distributions.NegativeBinomial(total_count=total_count, logits=logits)
np.testing.assert_allclose(distribution.mean.numpy(), mean.numpy())
np.testing.assert_allclose(distribution.variance.numpy(), (mean + mean.square() / concentration).numpy())
observed = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
size, probability = nb2_scipy_parameters(mean.numpy(), 1.0 / concentration.item())
np.testing.assert_allclose(distribution.log_prob(observed).numpy(), nbinom.logpmf(observed.numpy(), size, probability), rtol=1e-12, atol=1e-12)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_torch_nb2_parameters_preserve_broadcasting_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    mean = torch.tensor([[1.0], [3.0]], dtype=torch.float64, requires_grad=True)
    concentration = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    total_count, logits = nb2_torch_parameters(mean, concentration)

    assert total_count.shape == (3,)
    assert logits.shape == (2, 3)
    logits.sum().backward()
    assert mean.grad is not None


@pytest.mark.parametrize(
    ("mean", "concentration", "message"),
    [
        ([0.0], 1.0, "means"),
        ([1.0], 0.0, "concentration"),
        ([[1.0, 2.0]], [1.0, 2.0, 3.0], "broadcast-compatible"),
    ],
)
def test_torch_nb2_parameters_reject_invalid_inputs(
    mean: object, concentration: object, message: str
) -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match=message):
        nb2_torch_parameters(mean, concentration)


def test_nb2_total_probability_matches_manual_convolution() -> None:
    means = np.array([[1.0, 2.0, 3.0]])
    dispersions = np.array([0.2, 0.4, 0.8])
    observed = np.array([4.0])
    actual = np.exp(nb2_total_log_probability(observed, means, dispersions))[0]

    support = np.arange(5)
    masses = []
    for mean, dispersion in zip(means[0], dispersions, strict=True):
        size, probability = nb2_scipy_parameters(np.array([mean]), dispersion)
        masses.append(nbinom.pmf(support, size, probability.item()))
    expected = np.convolve(np.convolve(masses[0], masses[1]), masses[2])[4]
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)


def test_nb2_derivatives_match_finite_differences() -> None:
    observed = np.array([0.0, 2.0, 7.0])
    raw = np.array([-0.4, 0.3, 1.1])
    dispersion = 0.35
    gradient, hessian = nb2_gradient_hessian(
        observed, raw, dispersion=dispersion
    )

    def loss(values: np.ndarray) -> float:
        mean = np.exp(values)
        size, probability = nb2_scipy_parameters(mean, dispersion)
        return float(-nbinom.logpmf(observed, size, probability).sum())

    epsilon = 1e-5
    numeric_gradient = approx_fprime(raw, loss, epsilon)
    numeric_hessian = approx_fprime(
        raw,
        lambda values: approx_fprime(values, loss, epsilon),
        epsilon,
    ).diagonal()
    np.testing.assert_allclose(gradient, numeric_gradient, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(hessian, numeric_hessian, rtol=2e-3, atol=2e-3)


def test_dirichlet_multinomial_joint_matches_scipy_oracle() -> None:
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0.5, 3.0, size=(4, 3, 5))
    counts = rng.integers(0, 8, size=(4, 3, 5)).astype(float)
    n = counts.sum(axis=-1)

    actual = dirichlet_multinomial_log_probability(counts, alpha)
    expected = dirichlet_multinomial.logpmf(counts, alpha, n)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_dirichlet_multinomial_prefixes_match_explicitly_lumped_scipy_calls() -> None:
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0.5, 3.0, size=(4, 3, 5))
    counts = rng.integers(0, 8, size=(4, 3, 5)).astype(float)
    n_cohorts = counts.shape[-1]

    prefix = dirichlet_multinomial_prefix_log_masses(counts, alpha)
    joint = dirichlet_multinomial_log_probability(counts, alpha)

    for k in range(n_cohorts - 1):
        lumped_alpha = np.concatenate(
            (alpha[..., : k + 1], alpha[..., k + 1 :].sum(axis=-1, keepdims=True)),
            axis=-1,
        )
        lumped_counts = np.concatenate(
            (counts[..., : k + 1], counts[..., k + 1 :].sum(axis=-1, keepdims=True)),
            axis=-1,
        )
        expected = dirichlet_multinomial.logpmf(
            lumped_counts, lumped_alpha, lumped_counts.sum(axis=-1)
        )
        np.testing.assert_allclose(prefix[..., k], expected, rtol=0, atol=1e-12)

    # The final prefix has no remaining tail category to lump, so SciPy would
    # reject a zero concentration entry; compare against the unlumped joint.
    np.testing.assert_allclose(prefix[..., -1], joint, rtol=0, atol=1e-12)


def test_dirichlet_multinomial_prefix_last_entry_is_the_joint() -> None:
    rng = np.random.default_rng(1)
    alpha = rng.uniform(0.5, 3.0, size=(2, 6))
    counts = rng.integers(0, 10, size=(2, 6)).astype(float)

    prefix = dirichlet_multinomial_prefix_log_masses(counts, alpha)
    joint = dirichlet_multinomial_log_probability(counts, alpha)

    np.testing.assert_allclose(prefix[..., -1], joint, rtol=0, atol=0.0)


@pytest.mark.parametrize(
    ("kappa", "atol"),
    [
        (1e3, 2e-2),
        (1e6, 2e-5),
    ],
)
def test_dirichlet_multinomial_approaches_multinomial_at_large_concentration(
    kappa: float, atol: float
) -> None:
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(5))
    counts = rng.multinomial(20, p)
    n = counts.sum()
    alpha = kappa * p

    actual = dirichlet_multinomial_log_probability(counts.astype(float), alpha)
    expected = multinomial.logpmf(counts, n, p)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def test_dirichlet_multinomial_zero_total_gives_exact_zero_log_mass() -> None:
    counts = np.zeros(5)
    alpha = np.array([1.0, 2.0, 0.5, 3.0, 1.5])

    prefix = dirichlet_multinomial_prefix_log_masses(counts, alpha)

    assert (prefix == 0.0).all()


@pytest.mark.parametrize(
    ("counts", "alpha", "message"),
    [
        ([-1.0, 2.0], [1.0, 1.0], "nonnegative"),
        ([1.5, 2.0], [1.0, 1.0], "integers"),
        ([1.0, 2.0], [1.0, 0.0], "positive"),
        ([1.0, 2.0], [1.0, 1.0, 1.0], "cohort axis"),
    ],
)
def test_dirichlet_multinomial_rejects_invalid_inputs(
    counts: list[float], alpha: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dirichlet_multinomial_prefix_log_masses(np.array(counts), np.array(alpha))


def test_multinomial_prefix_last_entry_matches_scipy_oracle() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        cohort_count = int(rng.integers(2, 6))
        n = int(rng.integers(0, 60))
        p = rng.dirichlet(np.ones(cohort_count))
        counts = rng.multinomial(n, p)

        prefix = multinomial_prefix_log_masses(counts.astype(float), p)

        np.testing.assert_allclose(
            prefix[-1], multinomial.logpmf(counts, n, p), rtol=0, atol=1e-10
        )


def test_multinomial_positive_count_against_zero_probability_is_impossible() -> None:
    counts = np.array([1.0, 1.0, 1.0])
    p = np.array([0.5, 0.5, 0.0])

    prefix = multinomial_prefix_log_masses(counts, p)

    # The first prefix lumps the last two cohorts, whose combined probability
    # is positive, so it stays an ordinary binomial mass.
    np.testing.assert_allclose(prefix[0], binom.logpmf(1, 3, 0.5), rtol=0, atol=1e-12)
    assert prefix[1] == -np.inf
    assert prefix[-1] == -np.inf
    assert multinomial.logpmf(counts, 3, p) == -np.inf


@pytest.mark.parametrize(
    ("counts", "p"),
    [
        ([1.0, 2.0, 0.0], [0.5, 0.5, 0.0]),
        ([1.0, 0.0, 2.0], [0.5, 0.0, 0.5]),
        ([0.0, 0.0, 3.0], [0.0, 0.0, 1.0]),
    ],
)
def test_multinomial_zero_count_against_zero_probability_matches_scipy(
    counts: list[float], p: list[float]
) -> None:
    prefix = multinomial_prefix_log_masses(np.array(counts), np.array(p))

    assert np.isfinite(prefix).all()
    np.testing.assert_allclose(
        prefix[-1],
        multinomial.logpmf(np.array(counts, dtype=int), int(sum(counts)), p),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("counts", "p", "message"),
    [
        ([1.0, 1.0, 1.0], [0.2, 0.2, 0.2], "sum to one"),
        ([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], "sum to one"),
        ([1.0, 1.0, 1.0], [0.6, 0.6, -0.2], "nonnegative"),
        ([-1.0, 2.0], [0.5, 0.5], "nonnegative"),
        ([1.5, 2.0], [0.5, 0.5], "integers"),
        ([1.0, 2.0], [0.2, 0.3, 0.5], "cohort axis"),
    ],
)
def test_multinomial_prefix_rejects_invalid_inputs(
    counts: list[float], p: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        multinomial_prefix_log_masses(np.array(counts), np.array(p))