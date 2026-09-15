"""NUTS execution and convergence diagnostics for the Bayesian model."""

from __future__ import annotations

import inspect
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pyro
import torch
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.util import diagnostics as compute_diagnostics

from ..modeling_config import BayesianConditionalConfig, BayesianDiagnosticConfig

# Tree-depth instrumentation reads Pyro's protected ``NUTS._build_tree`` by
# position. Pin the signature it was written against so a Pyro upgrade fails
# loudly instead of silently recording the wrong argument as the tree depth.
_NUTS_BUILD_TREE_PARAMETERS = (
    "self",
    "z",
    "r",
    "z_grads",
    "log_slice",
    "direction",
    "tree_depth",
    "energy_current",
)
_TREE_DEPTH_ARGUMENT_POSITION = _NUTS_BUILD_TREE_PARAMETERS.index("tree_depth") - 1


@contextmanager
def _scoped_torch_runtime_settings() -> Iterator[None]:
    """Scope the macOS ARM / Python 3.13 Torch workarounds to one fit.

    ``torch.set_num_threads`` and ``torch.set_default_dtype`` mutate
    process-global state with no built-in undo, so a fit in a long-lived
    process would otherwise permanently change the dtype and thread count
    seen by unrelated code that runs afterward. This restores both, even
    when the wrapped block raises.
    """
    previous_threads = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_default_dtype(previous_dtype)


def _assert_supported_nuts_internals() -> None:
    """Fail loudly when Pyro's private NUTS signature no longer matches the hook."""
    observed = tuple(inspect.signature(NUTS._build_tree).parameters)
    if observed != _NUTS_BUILD_TREE_PARAMETERS:
        raise RuntimeError(
            "Unsupported Pyro version: NUTS._build_tree parameters changed from "
            f"{_NUTS_BUILD_TREE_PARAMETERS} to {observed}; the tree-depth "
            "instrumentation in InstrumentedNUTS must be updated"
        )


def evaluate_stage_diagnostics(
    diagnostics: Mapping[str, object], policy: BayesianDiagnosticConfig
) -> tuple[str, ...]:
    """Return deterministic convergence failures for one MCMC stage."""
    failures: list[str] = []
    checks = (
        ("worst_rhat", policy.maximum_rhat, lambda value, limit: value > limit),
        (
            "minimum_effective_sample_size",
            policy.minimum_effective_sample_size,
            lambda value, limit: value < limit,
        ),
        (
            "divergences",
            policy.maximum_divergences,
            lambda value, limit: value > limit,
        ),
        (
            "minimum_mean_accept_prob",
            policy.minimum_mean_accept_prob,
            lambda value, limit: value < limit,
        ),
        (
            "maximum_mean_accept_prob",
            policy.maximum_mean_accept_prob,
            lambda value, limit: value > limit,
        ),
        (
            "tree_depth_saturation",
            policy.maximum_tree_depth_saturation,
            lambda value, limit: value > limit,
        ),
    )
    for name, limit, failed in checks:
        value = diagnostics.get(name)
        if value is None or not np.isfinite(float(value)):
            failures.append(f"{name} is unavailable")
        elif failed(float(value), float(limit)):
            failures.append(f"{name}={value} violates threshold {limit}")
    return tuple(failures)


def _summarize_diagnostics(
    raw: Mapping[str, object], tree_depths: list[int], maximum_depth: int
) -> dict[str, object]:
    """Reduce raw Pyro diagnostics to the convergence fields used by policy."""

    def finite_or_none(value: float) -> float | None:
        """Convert non-finite diagnostic values to a JSON-safe null."""
        return value if np.isfinite(value) else None

    rhats: list[float] = []
    effective_sizes: list[float] = []
    for site_diagnostics in raw.values():
        if not isinstance(site_diagnostics, Mapping):
            continue
        if "r_hat" in site_diagnostics:
            rhats.extend(np.asarray(site_diagnostics["r_hat"]).reshape(-1).tolist())
        if "n_eff" in site_diagnostics:
            effective_sizes.extend(
                np.asarray(site_diagnostics["n_eff"]).reshape(-1).tolist()
            )
    divergences = raw.get("divergences", {})
    mean_accept_probs = raw.get("mean accept prob", {})
    divergence_count = (
        sum(len(values) for values in divergences.values())
        if isinstance(divergences, Mapping)
        else 0
    )
    mean_accept_prob_values = (
        [float(value) for value in mean_accept_probs.values()]
        if isinstance(mean_accept_probs, Mapping)
        else []
    )
    observed_maximum_tree_depth = max(tree_depths, default=0)
    # ``sample()`` appends a realized depth on every iteration regardless of
    # whether ``_build_tree`` ever ran, so a nonempty all-zero list means the
    # recursive hook this instrumentation depends on was never invoked (e.g.
    # a future Pyro NUTS using an iterative doubling loop instead) rather
    # than a real chain that never expanded past the root. Report that as
    # unavailable instrumentation, not as a healthy zero, so the policy check
    # fails loudly instead of silently passing.
    tree_depth_instrumentation_fired = (
        bool(tree_depths) and observed_maximum_tree_depth >= 1
    )
    return {
        "worst_rhat": finite_or_none(max(rhats, default=float("nan"))),
        "minimum_effective_sample_size": finite_or_none(
            min(effective_sizes, default=float("nan"))
        ),
        "divergences": divergence_count,
        "minimum_mean_accept_prob": finite_or_none(
            min(mean_accept_prob_values, default=float("nan"))
        ),
        "maximum_mean_accept_prob": finite_or_none(
            max(mean_accept_prob_values, default=float("nan"))
        ),
        "maximum_observed_tree_depth": (
            observed_maximum_tree_depth if tree_depth_instrumentation_fired else None
        ),
        "tree_depth_saturation": (
            float(np.mean(np.asarray(tree_depths) >= maximum_depth))
            if tree_depth_instrumentation_fired
            else None
        ),
    }


def _run_nuts(
    model: Any,
    model_args: tuple[object, ...],
    *,
    config: BayesianConditionalConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Run configured NUTS chains sequentially and combine their diagnostics.

    Returned posterior tensors flatten the chain and sample dimensions, while
    R-hat and ESS are computed from the unflattened chain-aware tensors.
    """
    _assert_supported_nuts_internals()
    profile = config.inference_profile

    class InstrumentedNUTS(NUTS):
        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize NUTS with storage for realized per-iteration depths."""
            super().__init__(*args, **kwargs)
            self.realized_tree_depths: list[int] = []
            self.final_mean_accept_prob = float("nan")
            self._current_tree_depth = 0

        def _build_tree(self, *args: object, **kwargs: object) -> Any:
            """Record the deepest recursive tree level reached in one sample."""
            tree_depth = int(
                kwargs.get("tree_depth", args[_TREE_DEPTH_ARGUMENT_POSITION])
            )
            self._current_tree_depth = max(self._current_tree_depth, tree_depth + 1)
            return super()._build_tree(*args, **kwargs)

        def sample(self, params: object) -> Any:
            """Generate one NUTS state, retaining tree depth and accept probability.

            The kernel's running mean accept probability must be read here
            rather than after the run, because ``MCMC`` calls ``cleanup`` and
            resets it. Its denominator restarts once warmup ends, so the value
            left by the final iteration covers retained iterations only.
            """
            self._current_tree_depth = 0
            result = super().sample(params)
            self.realized_tree_depths.append(self._current_tree_depth)
            self.final_mean_accept_prob = float(
                getattr(self, "_mean_accept_prob", float("nan"))
            )
            return result

    started = time.perf_counter()
    grouped_samples: dict[str, list[Any]] = {}
    tree_depths: list[int] = []
    divergences: dict[str, object] = {}
    mean_accept_probs: dict[str, object] = {}
    for chain_index in range(profile.chains):
        pyro.set_rng_seed((seed + chain_index) % (int(np.iinfo(np.uint32).max) + 1))
        kernel = InstrumentedNUTS(
            model,
            target_accept_prob=profile.target_acceptance,
            max_tree_depth=profile.max_tree_depth,
            full_mass=profile.full_mass,
            jit_compile=profile.jit_compile,
        )
        mcmc = MCMC(
            kernel,
            num_samples=profile.posterior_samples,
            warmup_steps=profile.warmup_steps,
            num_chains=1,
            disable_progbar=profile.disable_progress_bar,
        )
        mcmc.run(*model_args)
        for name, values in mcmc.get_samples().items():
            grouped_samples.setdefault(name, []).append(values)
        raw_chain_diagnostics = mcmc.diagnostics()
        divergences[f"chain {chain_index}"] = raw_chain_diagnostics.get(
            "divergences", {}
        ).get("chain 0", [])
        # Pyro's "acceptance rate" diagnostic counts multinomial-sampler moves
        # and is close to 1.0 for every NUTS run, so it cannot detect a badly
        # behaved sampler. Report instead the mean accept probability that
        # target_accept_prob adapts to, captured per iteration by the kernel.
        mean_accept_probs[f"chain {chain_index}"] = kernel.final_mean_accept_prob
        tree_depths.extend(kernel.realized_tree_depths[profile.warmup_steps :])
    runtime = time.perf_counter() - started
    stacked_samples = {
        name: torch.stack(chain_values) for name, chain_values in grouped_samples.items()
    }
    raw_diagnostics = dict(compute_diagnostics(stacked_samples))
    raw_diagnostics["divergences"] = divergences
    raw_diagnostics["mean accept prob"] = mean_accept_probs
    diagnostics = _summarize_diagnostics(
        raw_diagnostics,
        tree_depths,
        profile.max_tree_depth,
    )
    diagnostics.update(
        {
            "seed": seed,
            "runtime_seconds": runtime,
            "chains": profile.chains,
            "warmup_steps": profile.warmup_steps,
            "posterior_samples": profile.posterior_samples,
            "target_acceptance": profile.target_acceptance,
            "maximum_tree_depth": profile.max_tree_depth,
        }
    )
    samples = {
        name: values.flatten(0, 1) for name, values in stacked_samples.items()
    }
    return samples, diagnostics


__all__ = [
    "_run_nuts",
    "_scoped_torch_runtime_settings",
    "_summarize_diagnostics",
    "evaluate_stage_diagnostics",
]
