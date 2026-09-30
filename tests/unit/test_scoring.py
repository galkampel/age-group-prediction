"""The ready-made metrics and the validation of a custom one."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    root_mean_squared_error,
)

from age_group_prediction.scoring import MAE, POISSON_DEVIANCE, RMSE, Metric


@pytest.mark.parametrize(
    ("metric", "name", "function"),
    [
        (POISSON_DEVIANCE, "poisson_deviance", mean_poisson_deviance),
        (RMSE, "rmse", root_mean_squared_error),
        (MAE, "mae", mean_absolute_error),
    ],
)
def test_ready_made_metrics_wrap_sklearn(
    metric: Metric, name: str, function: Callable[..., float]
) -> None:
    assert metric.name == name
    assert metric.function is function
    # All three are losses; a tuner relies on this to negate them.
    assert metric.greater_is_better is False


@pytest.mark.parametrize(
    "build",
    [
        lambda: Metric("", mean_absolute_error),
        lambda: Metric("custom", 3),  # type: ignore[arg-type]
        lambda: Metric("custom", mean_absolute_error, 1),  # type: ignore[arg-type]
        lambda: Metric("custom", mean_absolute_error, greater_is_beter=True),  # type: ignore[call-arg]
    ],
    ids=["empty-name", "not-callable", "int-direction", "misspelled-field"],
)
def test_invalid_metric_is_rejected(build: Callable[[], Metric]) -> None:
    with pytest.raises((ValueError, TypeError)):
        build()
