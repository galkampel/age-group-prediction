from __future__ import annotations

from dataclasses import dataclass

from student_simulator.apartment import ApartmentSimulator
from student_simulator.config import Config, ValidationSettings
from student_simulator.outcomes import TotalChildrenSimulator
from student_simulator.pipeline import SimulationResult


@dataclass(frozen=True)
class CalibrationCheck:
    name: str
    passed: bool
    observed: float
    expected: tuple[float, float]


def run_calibration_checks(
    result: SimulationResult,
    validation: ValidationSettings,
) -> list[CalibrationCheck]:
    """Run warning-level calibration checks for canonical test gates."""
    counts = result.apartments["n_children_total"].to_numpy(dtype=float)
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
    result: SimulationResult,
    config: Config,
    validation: ValidationSettings,
) -> list[RecoveryResult]:
    """Fit oracle and naive NB2 models for test-only coefficient recovery."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError(
            "Coefficient recovery requires the validation dependency group: "
            "install with `uv sync --group validation`."
        ) from exc

    apartments = result.apartments
    buildings = result.buildings
    neighborhoods = result.neighborhoods
    context = ApartmentSimulator.with_context(
        apartments,
        buildings,
        neighborhoods,
    )
    design = TotalChildrenSimulator(
        config.simulation,
        config.total_children,
        config.transforms,
    ).build_design_matrix(context)
    response = apartments["n_children_total"].to_numpy(dtype=float)

    apartment_owners = apartments.merge(
        buildings[["building_id", "neighborhood_id"]],
        on="building_id",
        how="left",
        validate="many_to_one",
    )
    offset = (
        apartment_owners["building_id"].map(
            result.latent_effects["building_total_count_effect"]
        )
        + apartment_owners["neighborhood_id"].map(
            result.latent_effects["neighborhood_total_count_effect"]
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
