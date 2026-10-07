"""TotalChildrenModel against statsmodels, on a simulated population."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from age_group_prediction.modeling import TotalChildrenModel
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from student_simulator import StudentPopulationSimulator, load_simulation_config

FEATURES = [
    "ses",
    "avg_household_size",
    "median_age",
    "n_daycares_500m",
    "4_rooms_share",
    "5_rooms_share",
    "6_rooms_share",
]


def _simulated_table() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardized features, the totals and the exposure of seed 0's table."""
    config = load_simulation_config(
        Path(__file__).resolve().parents[2] / "configs" / "simulation.toml"
    )
    table = StudentPopulationSimulator(config).run(rng=np.random.default_rng(0))
    table = ShareTransformer(
        ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
    ).fit_transform(table)
    X = table[FEATURES]
    X = (X - X.mean()) / X.std()
    exposure = ExposureTransformer("n_apartments").fit_transform(table)
    return X, table["n_children_total"], exposure


@pytest.mark.slow
def test_the_unpenalized_fit_equals_statsmodels_poisson_glm() -> None:
    # A second, independent implementation of the Poisson likelihood with an
    # exposure offset, on the table the model is built for.
    X, y, exposure = _simulated_table()

    model = TotalChildrenModel().fit(X, y, exposure=exposure)
    oracle = sm.GLM(
        y.to_numpy(),
        sm.add_constant(X.to_numpy()),
        family=sm.families.Poisson(),
        exposure=exposure.to_numpy(),
    ).fit(tol=1e-12)

    np.testing.assert_allclose(
        np.r_[model.intercept_, model.coef_], oracle.params, atol=1e-6
    )


@pytest.mark.slow
def test_the_unpenalized_nb2_fit_equals_statsmodels() -> None:
    # statsmodels' NB2 (variance μ(1 + αμ)) with an exposure offset: the
    # coefficients and α itself, not 1/α. The simulated totals are
    # overdispersed (α ≈ 0.1), so the floor on α is not reached.
    X, y, exposure = _simulated_table()

    model = TotalChildrenModel(family="nb2").fit(X, y, exposure=exposure)
    oracle = sm.NegativeBinomial(
        y.to_numpy(),
        sm.add_constant(X.to_numpy()),
        exposure=exposure.to_numpy(),
        loglike_method="nb2",
    ).fit(method="newton", tol=1e-12, disp=0)

    np.testing.assert_allclose(
        np.r_[model.intercept_, model.coef_, model.dispersion_],
        oracle.params,
        atol=1e-5,
    )
