"""The ready-made metrics and the validation of a custom one."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    root_mean_squared_error,
)

from age_group_prediction.scoring import (
    COHORT_LOG_LOSS,
    COHORT_POISSON_DEVIANCE,
    MAE,
    POISSON_DEVIANCE,
    RMSE,
    Metric,
    cohort_log_loss,
    cohort_poisson_deviance,
)

# Three buildings, the second without children; probabilities that differ per row.
COUNTS = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 3.0, 2.0]])
PROBABILITIES = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])


@pytest.mark.parametrize(
    ("metric", "name", "function"),
    [
        (POISSON_DEVIANCE, "poisson_deviance", mean_poisson_deviance),
        (RMSE, "rmse", root_mean_squared_error),
        (MAE, "mae", mean_absolute_error),
        (COHORT_LOG_LOSS, "cohort_log_loss", cohort_log_loss),
        (COHORT_POISSON_DEVIANCE, "cohort_poisson_deviance", cohort_poisson_deviance),
    ],
)
def test_ready_made_metrics_wrap_their_function(
    metric: Metric, name: str, function: Callable[..., float]
) -> None:
    assert metric.name == name
    assert metric.function is function
    # All are losses; a tuner relies on this to negate them.
    assert metric.greater_is_better is False


# --- cohort_log_loss --------------------------------------------------


def test_cohort_log_loss_is_the_mean_per_child() -> None:
    # Per child, not per building: 9 children, so dividing by the 3 buildings
    # (or by the 2 with children) would be wrong. Pinned to sklearn's log_loss
    # on one row per (building, cohort), weighted by its count.
    by_hand = (
        -(
            2 * np.log(0.5)
            + np.log(0.3)
            + np.log(0.1)
            + 3 * np.log(0.6)
            + 2 * np.log(0.3)
        )
        / 9
    )
    rows = [(b, k) for b in range(3) for k in range(3) if COUNTS[b, k] > 0]
    by_sklearn = log_loss(
        [k for _, k in rows],
        [PROBABILITIES[b] for b, _ in rows],
        sample_weight=[COUNTS[b, k] for b, k in rows],
        labels=[0, 1, 2],
    )

    value = cohort_log_loss(COUNTS, PROBABILITIES)

    assert value == pytest.approx(by_hand)
    assert value == pytest.approx(by_sklearn)


def test_a_building_without_children_adds_nothing() -> None:
    # A plain 0 * log(0) is nan, which would poison the whole loss (and a
    # probability of exactly 0 is what a zero-count cohort may well get).
    with_zero_row = cohort_log_loss(COUNTS, PROBABILITIES)
    without = cohort_log_loss(COUNTS[[0, 2]], PROBABILITIES[[0, 2]])
    zero_probability = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.1, 0.6, 0.3]])

    assert with_zero_row == pytest.approx(without)
    assert np.isfinite(cohort_log_loss(COUNTS, zero_probability))


def test_predicted_counts_score_as_their_proportions() -> None:
    # Model 1 and Model 2 predict expected children per cohort; each row is
    # read as its proportions, so a count matrix is not scored as if it were
    # probabilities (which would give a wrong number, silently).
    totals = np.array([[7.0], [3.0], [20.0]])

    assert cohort_log_loss(COUNTS, PROBABILITIES * totals) == pytest.approx(
        cohort_log_loss(COUNTS, PROBABILITIES)
    )


def test_dataframes_are_paired_by_position() -> None:
    # Rows and columns pair by position, as everywhere (N19): a DataFrame
    # with other labels is not realigned (pandas would, into NaN).
    counts = pd.DataFrame(COUNTS, index=[10, 11, 12], columns=["a", "b", "c"])
    probabilities = pd.DataFrame(PROBABILITIES, columns=["x", "y", "z"])

    assert cohort_log_loss(counts, probabilities) == pytest.approx(
        cohort_log_loss(COUNTS, PROBABILITIES)
    )


@pytest.mark.parametrize(
    "y_pred",
    [PROBABILITIES[:, :1], PROBABILITIES[:, 0]],
    ids=["one-column", "one-dimensional"],
)
def test_a_prediction_of_another_shape_is_rejected(y_pred: np.ndarray) -> None:
    # numpy broadcasts one column over every cohort (after the row
    # normalization it is all ones, so the loss is 0, silently).
    with pytest.raises(ValueError, match="same shape"):
        cohort_log_loss(COUNTS, y_pred)


# --- cohort_poisson_deviance ------------------------------------------

# Four buildings by three cohorts (not square, so a transposed prediction has
# another shape); expected counts that differ per cell, so a mispairing changes
# the value.
COHORT_COUNTS = np.array(
    [[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 3.0, 2.0], [4.0, 0.0, 1.0]]
)
EXPECTED_COUNTS = np.array(
    [[1.5, 0.8, 0.4], [0.3, 0.2, 0.6], [1.2, 2.5, 1.7], [3.1, 0.5, 0.9]]
)


def test_cohort_poisson_deviance_is_the_mean_of_the_cohort_deviances() -> None:
    # Each cohort weighs as its own CountModel's POISSON_DEVIANCE would.
    per_cohort = [
        mean_poisson_deviance(COHORT_COUNTS[:, k], EXPECTED_COUNTS[:, k])
        for k in range(3)
    ]

    assert cohort_poisson_deviance(COHORT_COUNTS, EXPECTED_COUNTS) == pytest.approx(
        np.mean(per_cohort)
    )


@pytest.mark.parametrize(
    "y_pred",
    [EXPECTED_COUNTS.T.copy(), EXPECTED_COUNTS.ravel(order="F")],
    ids=["transposed", "flattened"],
)
def test_a_transposed_or_flattened_prediction_is_rejected(y_pred: np.ndarray) -> None:
    # As many cells as y_true: once raveled, they would be scored mispaired,
    # without an error from scikit-learn.
    with pytest.raises(ValueError, match="same shape"):
        cohort_poisson_deviance(COHORT_COUNTS, y_pred)


def test_cohort_poisson_deviance_pairs_dataframes_by_position() -> None:
    # A DataFrame with other labels is not realigned (pandas would, into NaN).
    counts = pd.DataFrame(
        COHORT_COUNTS, index=[10, 11, 12, 13], columns=["a", "b", "c"]
    )
    expected = pd.DataFrame(EXPECTED_COUNTS, columns=["x", "y", "z"])

    assert cohort_poisson_deviance(counts, expected) == pytest.approx(
        cohort_poisson_deviance(COHORT_COUNTS, EXPECTED_COUNTS)
    )


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
