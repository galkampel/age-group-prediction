from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from student_simulator.config import Config, ValidationSettings
from student_simulator.outcomes import LatentEffects, TotalChildrenSimulator

# Test-only quality thresholds for the canonical configs/simulation.yml run;
# they do not control data generation.
VALIDATION = ValidationSettings(
    mean_children_min=0.40,
    mean_children_max=0.61,
    zero_share_min=0.60,
    zero_share_max=0.75,
    recovery_abs_tolerance=0.20,
)


@dataclass(frozen=True)
class CalibrationCheck:
    name: str
    passed: bool
    observed: float
    expected: tuple[float, float]


def run_calibration_checks(
    apartments_df: pd.DataFrame,
    validation: ValidationSettings,
) -> list[CalibrationCheck]:
    """Run warning-level calibration checks on apartment child totals."""
    counts = apartments_df["n_children_total"].to_numpy(dtype=float)
    mean_children = float(counts.mean())
    zero_share = float((counts == 0).mean())

    mean_bounds = (
        validation.mean_children_min,
        validation.mean_children_max,
    )
    zero_bounds = (
        validation.zero_share_min,
        validation.zero_share_max,
    )

    return [
        CalibrationCheck(
            name="mean_children_range",
            passed=mean_bounds[0] <= mean_children <= mean_bounds[1],
            observed=mean_children,
            expected=mean_bounds,
        ),
        CalibrationCheck(
            name="zero_share_range",
            passed=zero_bounds[0] <= zero_share <= zero_bounds[1],
            observed=zero_share,
            expected=zero_bounds,
        ),
    ]


@dataclass(frozen=True)
class RecoveryResult:
    name: str
    passed: bool
    observed: dict[str, object]


def run_recovery_checks(
    apartments_df: pd.DataFrame,
    total_effects: LatentEffects,
    config: Config,
    validation: ValidationSettings,
) -> list[RecoveryResult]:
    """Fit oracle and naive NB2 models for test-only coefficient recovery.

    `apartments_df` is the total-children stage's output (context, owner ids and
    `n_children_total`); `total_effects` are the effects that stage returned.
    """
    design = TotalChildrenSimulator(
        config.simulation,
        config.total_children,
        config.transforms,
    ).build_design_matrix(apartments_df)
    response = apartments_df["n_children_total"].to_numpy(dtype=float)

    offset = (
        apartments_df["building_id"].map(total_effects["building_total_count_effect"])
        + apartments_df["neighborhood_id"].map(
            total_effects["neighborhood_total_count_effect"]
        )
    ).to_numpy(dtype=float)

    alpha = 1.0 / config.total_children.nb_dispersion_phi
    family = sm.families.NegativeBinomial(alpha=alpha)
    oracle_fit = sm.GLM(response, design, family=family, offset=offset).fit()
    naive_fit = sm.GLM(response, design, family=family).fit()

    total = config.total_children
    expected = {
        "intercept": total.intercept,
        **{
            f"rooms_{room}": coefficient
            for room, coefficient in sorted(
                total.room_log_mean_effects.items()
            )
        },
        "ses_squared": total.ses_quadratic_coef,
        "household_size_scaled": total.household_size_coef,
        "daycare_saturation": total.daycare_saturation_coef,
        "school_existing": total.existing_school_coef,
        "school_planned": total.planned_school_coef,
        "median_age_scaled": total.median_age_coef,
        "rooms_household_interaction": total.rooms_household_interaction_coef,
    }
    oracle_estimates = dict(
        zip(design.columns, oracle_fit.params, strict=True)
    )
    naive_estimates = dict(zip(design.columns, naive_fit.params, strict=True))
    absolute_errors = {
        name: abs(float(oracle_estimates[name]) - truth)
        for name, truth in expected.items()
    }
    max_error = max(absolute_errors.values())
    tolerance = validation.recovery_abs_tolerance

    return [
        RecoveryResult(
            name="oracle_nb2_fixed_effect_recovery",
            passed=max_error <= tolerance,
            observed={
                "estimates": oracle_estimates,
                "absolute_errors": absolute_errors,
                "max_absolute_error": max_error,
            },
        ),
        RecoveryResult(
            name="naive_nb2_benchmark",
            passed=True,
            observed={"estimates": naive_estimates},
        ),
    ]
