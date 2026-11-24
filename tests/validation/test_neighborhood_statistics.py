"""Shapes and signs of the neighborhood draws on one large, fixed-seed sample.

The plan's real-world targets are approximate (docs/SIMULATOR_REALISM_PLAN.md,
R10), so these tests assert directions and coarse ranges, not exact numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from student_simulator import load_simulation_config
from student_simulator.config import Config
from tests.unit.helpers import make_neighborhood_simulator

N_NEIGHBORHOODS = 20_000
SEED = 20_261_124


def _config() -> Config:
    path = Path(__file__).resolve().parents[2] / "configs" / "simulation.yml"
    return load_simulation_config(path)


def _sample() -> pd.DataFrame:
    return make_neighborhood_simulator(_config()).sample_neighborhoods(
        N_NEIGHBORHOODS, rng=np.random.default_rng(SEED)
    )


@pytest.fixture(scope="module")
def neighborhoods() -> pd.DataFrame:
    return _sample()


def test_ses_is_a_standardized_index(neighborhoods: pd.DataFrame) -> None:
    ses = neighborhoods["ses"]

    assert abs(ses.mean()) < 0.05
    assert 0.95 < ses.std() < 1.03


def test_median_age_rises_with_ses(neighborhoods: pd.DataFrame) -> None:
    ses = neighborhoods["ses"]
    age = neighborhoods["median_age"]
    deciles = pd.qcut(ses, 10, labels=False)
    age_by_decile = age.groupby(deciles).mean()

    assert np.corrcoef(ses, age)[0, 1] > 0.4
    assert age_by_decile.iloc[0] <= age_by_decile.iloc[-1] - 8


def test_household_size_falls_with_age(neighborhoods: pd.DataFrame) -> None:
    ses = neighborhoods["ses"]
    design = sm.add_constant(
        pd.DataFrame(
            {"median_age": neighborhoods["median_age"], "ses": ses, "ses_sq": ses**2}
        )
    )
    fit = sm.OLS(neighborhoods["avg_household_size"], design).fit()

    assert fit.params["median_age"] < 0


def test_household_size_is_a_reverse_j_in_ses(neighborhoods: pd.DataFrame) -> None:
    ses = neighborhoods["ses"]
    household_size = neighborhoods["avg_household_size"]
    # The plan's ten SES bins: <= -2, -2..-1.5, ..., 1.5..2, > 2.
    edges = [-np.inf, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf]
    bin_means = household_size.groupby(pd.cut(ses, edges), observed=False).mean()
    middle = household_size[ses.between(-0.5, 0.5)].mean()
    lowest, minimum, top = bin_means.iloc[0], bin_means.min(), bin_means.iloc[-1]

    assert lowest > middle > minimum
    assert top > minimum
    assert 3.0 <= household_size.mean() <= 3.4


def test_daycares_follow_demand_purchasing_power_and_young_areas(
    neighborhoods: pd.DataFrame,
) -> None:
    daycares = neighborhoods["n_daycares_500m"]
    design = sm.add_constant(neighborhoods[["avg_household_size", "ses", "median_age"]])
    fit = sm.GLM(daycares, design, family=sm.families.Poisson()).fit()
    quintiles = pd.qcut(neighborhoods["ses"], 5, labels=False)
    daycares_by_quintile = daycares.groupby(quintiles).mean()

    assert pd.api.types.is_integer_dtype(daycares)
    assert (daycares >= 0).all()
    assert fit.params["avg_household_size"] > 0
    assert fit.params["ses"] > 0
    assert fit.params["median_age"] < 0
    assert daycares_by_quintile.iloc[-1] > daycares_by_quintile.iloc[0]
    assert 0.02 <= (daycares == 0).mean() <= 0.15


def test_school_status_follows_household_size_and_age(
    neighborhoods: pd.DataFrame,
) -> None:
    status = neighborhoods["school_status"]
    household_size = neighborhoods["avg_household_size"]
    age = neighborhoods["median_age"]
    existing_by_tercile = (
        (status == "existing").groupby(pd.qcut(household_size, 3, labels=False)).mean()
    )
    planned_by_tercile = (
        (status == "planned").groupby(pd.qcut(age, 3, labels=False)).mean()
    )

    assert set(status.unique()) == {"existing", "planned", "none"}
    assert existing_by_tercile.is_monotonic_increasing
    assert existing_by_tercile.is_unique
    assert planned_by_tercile.is_monotonic_decreasing
    assert planned_by_tercile.is_unique


def test_school_shares_near_the_reference_neighborhood(
    neighborhoods: pd.DataFrame,
) -> None:
    ncfg = _config().neighborhood
    window = (
        neighborhoods["avg_household_size"].sub(ncfg.household_size_baseline).abs()
        < 0.15
    ) & (neighborhoods["median_age"].sub(ncfg.median_age_baseline).abs() < 2.0)
    shares = neighborhoods.loc[window, "school_status"].value_counts(normalize=True)

    assert window.sum() > 500
    for label, reference_share in ncfg.school_status_reference_shares.items():
        assert shares[label] == pytest.approx(reference_share, abs=0.05)


@pytest.mark.parametrize(
    ("column", "bounds"),
    [
        ("ses", ("ses_min", "ses_max")),
        ("median_age", ("median_age_min", "median_age_max")),
        ("avg_household_size", ("household_size_min", "household_size_max")),
    ],
)
def test_values_within_bounds_and_rarely_clipped(
    neighborhoods: pd.DataFrame, column: str, bounds: tuple[str, str]
) -> None:
    ncfg = _config().neighborhood
    minimum, maximum = (getattr(ncfg, name) for name in bounds)
    values = neighborhoods[column]

    assert values.between(minimum, maximum).all()
    assert ((values == minimum) | (values == maximum)).mean() < 0.02


def test_the_same_seed_gives_the_same_frame(neighborhoods: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(_sample(), neighborhoods)
