"""Slow end-to-end NUTS checks for the Bayesian conditional model.

Every other Bayesian test monkeypatches inference, so these are the only checks
that run real NUTS. They are marked ``slow`` and are excluded from the fast
suite by ``pytest -m "not calibration and not slow"``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("pyro")

from age_group_prediction import (
    DEFAULT_PROBABILITY_FEATURE_SPEC,
    DEFAULT_TOTAL_FEATURE_SPEC,
    BayesianConditionalModel,
    PredictionConfig,
)
from age_group_prediction.fitted_features import (
    FittedFeatureTransformer,
)
from age_group_prediction.modeling_config import (
    DEFAULT_MODELING_SCHEMA,
)

# Generating values are deliberately placed well away from their prior means,
# so that a posterior which simply reproduced the prior would NOT cover them.
# An earlier version of this test drew every value from the prior's
# high-density region, which made it pass even with the total likelihood
# removed entirely.
#
# The offsets are moderate on purpose. Pushing them far into the prior tails
# makes the simulated building totals span orders of magnitude, which wrecks
# the posterior geometry and fails on correct code. The power of this test
# comes from the contraction assertions below, not from extreme truth values,
# so these only need to be off-centre enough that coverage is not trivial
# while the simulated data stay realistic.
#
# value                     prior            offset
# intercept        -1.5     Normal(-2, 1)    +0.5 sd
# log concentration 1.2     Normal(0, 1)     +1.2 sd
# log kappa         3.2     Normal(2, 1)     +1.2 sd
# neighborhood sd   0.5     HalfNormal(0.5)  upper ~32% tail
_TRUE_INTERCEPT = -1.5
_TRUE_LOG_CONCENTRATION = 1.2
_TRUE_NEIGHBORHOOD_SCALE = 0.5
_TRUE_LOG_KAPPA = 3.2

# Prior standard deviations, used for the posterior-contraction assertions.
# Coverage alone cannot distinguish a posterior that learned from one that
# stayed at the prior, so each scalar must also be substantially narrower
# than its prior.
_PRIOR_SCALAR_SDS = {
    "total_intercept": 1.0,
    "log_total_concentration": 1.0,
    "log_composition_concentration": 1.0,
    # HalfNormal(s) has sd = s * sqrt(1 - 2/pi).
    "neighborhood_scale": 0.5 * float(np.sqrt(1.0 - 2.0 / np.pi)),
}
_MAXIMUM_POSTERIOR_SD_RATIO = 0.4


def _raw_frame(row_count: int, neighborhood_count: int, seed: int) -> pd.DataFrame:
    """Build a raw modeling table with no outcome columns yet."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "building_id": np.arange(row_count),
            "neighborhood_id": [
                f"n{index % neighborhood_count}" for index in range(row_count)
            ],
            "ses": rng.normal(size=row_count),
            "avg_household_size": rng.uniform(1.8, 3.8, size=row_count),
            "median_age": rng.uniform(24.0, 58.0, size=row_count),
            "n_daycares_500m": rng.integers(0, 6, size=row_count),
            "n_apartments": rng.integers(20, 90, size=row_count),
            "3_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "4_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "5_rooms_share": rng.uniform(0.05, 0.3, size=row_count),
            "school_status": rng.choice(
                ["none", "existing", "planned"], size=row_count
            ),
            "n_kindergarten": 0,
            "n_elementary": 0,
            "n_highschool": 0,
            "n_children_total": 0,
        }
    )


def _simulate(
    row_count: int = 600,
    # The neighborhood scale is a group-level parameter, so its posterior is
    # informed by the number of neighborhoods, not the number of buildings.
    # Twelve groups leave it too weakly identified for the contraction
    # assertion to hold reliably across seeds. Thirty is both better posed and
    # closer to the real modeling table, which has sixty.
    neighborhood_count: int = 30,
    seed: int = 20250911,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | float]]:
    """Simulate outcomes from the model's own design matrices.

    Generating from the transformed design guarantees the fitted model is
    correctly specified, so a recovery failure indicates an inference defect
    rather than a mismatch between the simulation and the likelihood.
    """
    frame = _raw_frame(row_count, neighborhood_count, seed)
    rng = np.random.default_rng(seed + 1)
    schema = DEFAULT_MODELING_SCHEMA

    total_transformer = FittedFeatureTransformer(
        DEFAULT_TOTAL_FEATURE_SPEC, schema=schema
    )
    total_features = total_transformer.fit_transform(frame).to_numpy()
    log_exposure = total_transformer.get_log_exposure(frame).to_numpy()

    probability_transformer = FittedFeatureTransformer(
        DEFAULT_PROBABILITY_FEATURE_SPEC, schema=schema
    )
    composition_features = probability_transformer.fit_transform(frame).to_numpy()

    cohort_count = len(schema.cohort_target_columns)
    total_coefficients = rng.normal(0.0, 0.25, size=total_features.shape[1])
    neighborhood_raw = rng.normal(0.0, 1.0, size=neighborhood_count)
    neighborhood_index = np.array(
        [int(value[1:]) for value in frame["neighborhood_id"]]
    )

    log_mean = (
        log_exposure
        + _TRUE_INTERCEPT
        + total_features @ total_coefficients
        + _TRUE_NEIGHBORHOOD_SCALE * neighborhood_raw[neighborhood_index]
    )
    concentration = np.exp(_TRUE_LOG_CONCENTRATION)
    mean = np.exp(log_mean)
    totals = rng.negative_binomial(concentration, concentration / (concentration + mean))

    composition_intercepts = rng.normal(0.0, 0.4, size=cohort_count - 1)
    composition_coefficients = rng.normal(
        0.0, 0.3, size=(composition_features.shape[1], cohort_count - 1)
    )
    logits = np.concatenate(
        (
            composition_intercepts + composition_features @ composition_coefficients,
            np.zeros((len(frame), 1)),
        ),
        axis=1,
    )
    probabilities = np.exp(
        logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    )
    kappa = np.exp(_TRUE_LOG_KAPPA)
    cohorts = np.zeros((len(frame), cohort_count), dtype=int)
    for row, total in enumerate(totals):
        if total == 0:
            continue
        shares = rng.dirichlet(kappa * probabilities[row])
        cohorts[row] = rng.multinomial(int(total), shares)

    for index, column in enumerate(schema.cohort_target_columns):
        frame[column] = cohorts[:, index]
    frame[schema.total_target_column] = cohorts.sum(axis=1)

    truth: dict[str, np.ndarray | float] = {
        "total_intercept": _TRUE_INTERCEPT,
        "total_coefficients": total_coefficients,
        "log_total_concentration": _TRUE_LOG_CONCENTRATION,
        "neighborhood_scale": _TRUE_NEIGHBORHOOD_SCALE,
        "composition_intercepts": composition_intercepts,
        "composition_coefficients": composition_coefficients,
        "log_composition_concentration": _TRUE_LOG_KAPPA,
    }
    return frame, truth


def _covers(samples: np.ndarray, value: float, level: float = 0.95) -> bool:
    """Report whether a central posterior interval contains a true value."""
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    return bool(lower <= value <= upper)


@pytest.mark.slow
def test_real_nuts_fit_completes_both_stages_and_preserves_invariants() -> None:
    """Run real NUTS end to end and assert the documented payload invariants."""
    frame, _ = _simulate(row_count=150, neighborhood_count=6)
    model = BayesianConditionalModel()
    model.fit(frame, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(5))

    diagnostics = model.get_metadata()["model"]["diagnostics"]
    assert set(diagnostics) == {"total", "composition"}
    profile = model.bayesian_config.inference_profile
    expected_samples = profile.chains * profile.posterior_samples
    for stage in ("total", "composition"):
        payload = diagnostics[stage]
        assert payload["chains"] == profile.chains
        assert payload["posterior_samples"] == profile.posterior_samples
        # Convergence quality is deliberately not asserted here. The reduced
        # profile is documented as smoke-scale, so a stray divergence or a low
        # effective sample size is an expected outcome that the diagnostic
        # policy reports. This test owns the structural contract; recovery and
        # the full profile own convergence. Only check the plumbing produces a
        # usable value.
        assert isinstance(payload["divergences"], int)
        assert payload["divergences"] >= 0
        # The renamed diagnostic must report the adapted accept probability
        # rather than Pyro's move counter, which is always close to one.
        assert 0.5 < payload["minimum_mean_accept_prob"] <= 1.0
    total_posterior, composition_posterior = model._posterior_arrays()
    assert total_posterior["total_intercept"].shape == (expected_samples,)
    assert composition_posterior["log_composition_concentration"].shape == (
        expected_samples,
    )

    evaluation = frame.iloc[:40].copy()
    evaluation.loc[evaluation.index[0], "neighborhood_id"] = "unseen-neighborhood"
    prediction = model.predict(
        evaluation,
        prediction_config=PredictionConfig(
            n_predictive_draws=48,
            interval_levels=(0.8, 0.95),
            include_pointwise_log_probabilities=True,
        ),
        rng=np.random.default_rng(6),
    )

    draws = prediction.predictive_draws
    assert draws is not None
    total_draws = draws["total"]
    cohort_draws = np.stack(
        [draws[name] for name in prediction.cohort_names], axis=-1
    )
    assert np.array_equal(cohort_draws.sum(axis=2), total_draws)
    assert (total_draws >= 0).all()
    assert np.array_equal(total_draws, np.floor(total_draws))
    assert np.array_equal(cohort_draws, np.floor(cohort_draws))
    assert np.isfinite(prediction.total_mean).all()

    pointwise = prediction.pointwise_log_probabilities
    assert pointwise is not None
    assert set(pointwise) == {"total", *prediction.cohort_names}
    assert all(np.isfinite(values).all() for values in pointwise.values())

    metadata = model.get_metadata()
    assert metadata["model"]["last_prediction_unseen_neighborhood_count"] == 1
    json.dumps(metadata)


@pytest.mark.slow
def test_real_nuts_posterior_recovers_known_generating_parameters() -> None:
    """Check posterior credible intervals cover the parameters that generated data."""
    frame, truth = _simulate()
    model = BayesianConditionalModel()
    model.fit(
        frame, feature_spec=DEFAULT_TOTAL_FEATURE_SPEC, rng=np.random.default_rng(11)
    )
    total_posterior, composition_posterior = model._posterior_arrays()

    scalars = {
        "total_intercept": total_posterior["total_intercept"],
        "log_total_concentration": total_posterior["log_total_concentration"],
        "neighborhood_scale": total_posterior["neighborhood_scale"],
        "log_composition_concentration": composition_posterior[
            "log_composition_concentration"
        ],
    }

    # Contraction is the assertion with real power. Coverage alone passes when
    # the posterior simply reproduces the prior, which is what happens if a
    # likelihood term is silently disconnected. Every scalar must be
    # substantially narrower than its own prior.
    for name, samples in scalars.items():
        ratio = float(samples.std()) / _PRIOR_SCALAR_SDS[name]
        assert ratio < _MAXIMUM_POSTERIOR_SD_RATIO, (
            f"{name} posterior did not contract: posterior sd is {ratio:.2f} of "
            f"the prior sd, so the data barely informed it"
        )

    # Four independent 95% intervals miss at least one about 19% of the time,
    # so require a rate rather than every one individually. Combined with the
    # contraction assertions above, this is a check on centring, not on width.
    scalar_covered = sum(
        _covers(samples, float(truth[name])) for name, samples in scalars.items()
    )
    assert scalar_covered >= 3, (
        f"only {scalar_covered}/4 scalar credible intervals covered their "
        f"generating value: "
        + ", ".join(
            f"{name}={'hit' if _covers(samples, float(truth[name])) else 'MISS'}"
            for name, samples in scalars.items()
        )
    )

    # Vector parameters are checked by coverage rate rather than individually,
    # because several independent 95% intervals will occasionally miss.
    total_coefficients = np.asarray(truth["total_coefficients"])
    total_covered = np.mean(
        [
            _covers(total_posterior["total_coefficients"][:, index], value)
            for index, value in enumerate(total_coefficients)
        ]
    )
    composition_coefficients = np.asarray(truth["composition_coefficients"])
    composition_covered = np.mean(
        [
            _covers(
                composition_posterior["composition_coefficients"][:, row, column],
                composition_coefficients[row, column],
            )
            for row in range(composition_coefficients.shape[0])
            for column in range(composition_coefficients.shape[1])
        ]
    )
    # 0.70 sits below the binomial noise floor for 18 composition coefficients
    # (the 1% lower tail at p=0.95 is about 0.78), so it would not distinguish a
    # genuine mis-parameterization from sampling noise. Use a tighter bar there.
    assert total_covered >= 0.7, f"total coefficient coverage {total_covered:.2f}"
    assert composition_covered >= 0.8, (
        f"composition coefficient coverage {composition_covered:.2f}"
    )
