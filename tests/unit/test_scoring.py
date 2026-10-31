"""The scoring functions, the validation of a custom metric, and the score table of several metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, mean_absolute_error, mean_poisson_deviance

from age_group_prediction.scoring import (
    Metric,
    evaluate,
    mean_absolute_error_loss,
    mean_bias_error,
    mean_poisson_deviance_loss,
    multinomial_log_loss,
    root_mean_squared_error_loss,
    wmape_loss,
)

# Three samples, the second without children; probabilities that differ per row.
COUNTS = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 3.0, 2.0]])
PROBABILITIES = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])


def test_calling_a_metric_applies_its_function_as_a_float() -> None:
    # Catches a call that returns the NumPy scalar, which MLflow and json reject.
    metric = Metric("mae", lambda y_true, y_pred: np.float64(0.5))

    value = metric([1.0], [1.5])

    assert value == 0.5
    assert type(value) is float


def test_a_metric_passes_the_cohort_on_only_when_given() -> None:
    # A two-argument function (scikit-learn's, a lambda) keeps scoring the
    # whole of y; a cohort must reach the function, never be dropped silently.
    calls = []

    def score(y_true, y_pred, **kwargs) -> float:
        calls.append(kwargs)
        return 1.0

    Metric("score", score)([1.0], [1.0])
    Metric("score", score)([1.0], [1.0], cohort="n_elementary")

    assert calls == [{}, {"cohort": "n_elementary"}]
    with pytest.raises(TypeError, match="cohort"):
        Metric("mae", mean_absolute_error)([1.0], [1.0], cohort="n_elementary")


# --- multinomial_log_loss ---------------------------------------------


def test_multinomial_log_loss_is_the_mean_per_child() -> None:
    # Per child, not per sample: 9 children, so dividing by the 3 samples
    # (or by the 2 with children) would be wrong. Pinned to sklearn's log_loss
    # on one row per (sample, cohort), weighted by its count.
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
    rows = [(i, k) for i in range(3) for k in range(3) if COUNTS[i, k] > 0]
    by_sklearn = log_loss(
        y_true=[k for _, k in rows],
        y_proba=[PROBABILITIES[i] for i, _ in rows],
        sample_weight=[COUNTS[i, k] for i, k in rows],
        labels=[0, 1, 2],
    )

    value = multinomial_log_loss(COUNTS, PROBABILITIES)

    assert value == pytest.approx(by_hand)
    assert value == pytest.approx(by_sklearn)


def test_a_sample_without_children_adds_nothing() -> None:
    # A plain 0 * log(0) is nan, which would poison the whole loss (and a
    # probability of exactly 0 is what a zero-count cohort may well get).
    with_zero_row = multinomial_log_loss(COUNTS, PROBABILITIES)
    without = multinomial_log_loss(COUNTS[[0, 2]], PROBABILITIES[[0, 2]])
    zero_probability = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.1, 0.6, 0.3]])

    assert with_zero_row == pytest.approx(without)
    assert np.isfinite(multinomial_log_loss(COUNTS, zero_probability))


def test_a_cohort_without_a_child_is_scored_not_rejected() -> None:
    # A bootstrap replicate or a small part can hold no child of one cohort;
    # log_loss would infer two classes from y_true against three columns.
    counts = np.array([[2.0, 0.0, 1.0], [1.0, 0.0, 3.0]])
    by_hand = -(2 * np.log(0.5) + np.log(0.2) + np.log(0.1) + 3 * np.log(0.3)) / 7

    assert multinomial_log_loss(counts, PROBABILITIES[[0, 2]]) == pytest.approx(by_hand)


def test_predicted_counts_score_as_their_proportions() -> None:
    # Model 1 and Model 2 predict expected children per cohort; each row is
    # read as its proportions, so a count matrix is not scored as if it were
    # probabilities (which would give a wrong number, silently).
    totals = np.array([[7.0], [3.0], [20.0]])

    assert multinomial_log_loss(COUNTS, PROBABILITIES * totals) == pytest.approx(
        multinomial_log_loss(COUNTS, PROBABILITIES)
    )


def test_one_cohorts_log_loss_is_the_binomial_log_loss_of_its_share() -> None:
    # The cohort against the sum of the others, per child:
    # -[n_k log p_k + (N - n_k) log(1 - p_k)] / sum N. The predicted counts'
    # rows sum to 7, 3 and 20, so the rest's sum is scored as a proportion too.
    k, totals = 1, COUNTS.sum(axis=1)
    p = PROBABILITIES[:, k]
    by_hand = (
        -(COUNTS[:, k] * np.log(p) + (totals - COUNTS[:, k]) * np.log(1 - p)).sum()
        / totals.sum()
    )
    counts = pd.DataFrame(COUNTS, columns=["a", "b", "c"])
    predicted = PROBABILITIES * np.array([[7.0], [3.0], [20.0]])

    assert multinomial_log_loss(counts, predicted, cohort="b") == pytest.approx(by_hand)


def test_dataframes_and_lists_are_paired_by_position() -> None:
    # Rows and columns pair by position, as everywhere (N19): a DataFrame
    # with other labels is not realigned (pandas would, into NaN).
    counts = pd.DataFrame(COUNTS, index=[10, 11, 12], columns=["a", "b", "c"])
    probabilities = pd.DataFrame(PROBABILITIES, columns=["x", "y", "z"])
    expected = multinomial_log_loss(COUNTS, PROBABILITIES)

    assert multinomial_log_loss(counts, probabilities) == pytest.approx(expected)
    assert multinomial_log_loss(
        COUNTS.tolist(), PROBABILITIES.tolist()
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "y_pred",
    [PROBABILITIES[:, :1], PROBABILITIES[:, 0]],
    ids=["one-column", "one-dimensional"],
)
def test_a_prediction_of_another_shape_is_rejected(y_pred: np.ndarray) -> None:
    # numpy broadcasts one column over every cohort (after the row
    # normalization it is all ones, so the loss is 0, silently).
    with pytest.raises(ValueError, match="same shape"):
        multinomial_log_loss(COUNTS, y_pred)


def test_one_cohort_alone_is_rejected() -> None:
    # One cohort normalizes to probability 1 and would score 0 whatever the
    # prediction: a single column, a 1-D cohort, or a single column scored with
    # cohort= (its rest column is all zeros, so the cohort's share is 1).
    with pytest.raises(ValueError, match="at least two cohorts"):
        multinomial_log_loss(COUNTS[:, :1], PROBABILITIES[:, :1])
    with pytest.raises(ValueError, match="with cohort="):
        multinomial_log_loss(COUNTS[:, 0], PROBABILITIES[:, 0])
    with pytest.raises(ValueError, match="at least two cohorts"):
        multinomial_log_loss(
            pd.DataFrame(COUNTS[:, :1], columns=["a"]), PROBABILITIES[:, :1], cohort="a"
        )


# --- the count metrics ------------------------------------------------

# Four samples by three cohorts (not square, so a transposed prediction has
# another shape); expected counts that differ per cell, so a mispairing changes
# the value.
COHORT_COUNTS = np.array(
    [[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 3.0, 2.0], [4.0, 0.0, 1.0]]
)
EXPECTED_COUNTS = np.array(
    [[1.5, 0.8, 0.4], [0.3, 0.2, 0.6], [1.2, 2.5, 1.7], [3.1, 0.5, 0.9]]
)


def _pooled(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Each count metric by hand, over every cell."""
    error = (y_pred - y_true).ravel()
    return {
        "mean_poisson_deviance_loss": mean_poisson_deviance(
            y_true.ravel(), y_pred.ravel()
        ),
        "mean_absolute_error_loss": np.abs(error).mean(),
        "root_mean_squared_error_loss": np.sqrt((error**2).mean()),
        "wmape_loss": np.abs(error).sum() / y_true.sum(),
        "mean_bias_error": error.mean(),
    }


COUNT_METRICS = [
    mean_poisson_deviance_loss,
    mean_absolute_error_loss,
    root_mean_squared_error_loss,
    wmape_loss,
    mean_bias_error,
]
COUNT_METRIC_IDS = [function.__name__ for function in COUNT_METRICS]


@pytest.mark.parametrize("function", COUNT_METRICS, ids=COUNT_METRIC_IDS)
def test_a_count_metric_on_the_cohort_matrix_pools_every_cell(
    function: Callable[..., float],
) -> None:
    # Catches a metric scored per cohort and averaged, or on one column only.
    expected = _pooled(COHORT_COUNTS, EXPECTED_COUNTS)[function.__name__]

    assert function(COHORT_COUNTS, EXPECTED_COUNTS) == pytest.approx(expected)


@pytest.mark.parametrize("function", COUNT_METRICS, ids=COUNT_METRIC_IDS)
def test_a_count_metric_scores_one_cohort(
    function: Callable[..., float],
) -> None:
    # A cohort's column (or the total) is one-dimensional, the per-cohort rows
    # of a score table.
    k = 2
    expected = _pooled(COHORT_COUNTS[:, k], EXPECTED_COUNTS[:, k])[function.__name__]

    assert function(COHORT_COUNTS[:, k], EXPECTED_COUNTS[:, k]) == pytest.approx(
        expected
    )


COHORT_FRAME = pd.DataFrame(COHORT_COUNTS, columns=["k", "e", "h"])
# Other labels: the cohort is found among y_true's columns and y_pred is read
# at the same position, never realigned by label.
EXPECTED_FRAME = pd.DataFrame(EXPECTED_COUNTS, columns=["x", "y", "z"])
ALL_METRICS = [multinomial_log_loss, *COUNT_METRICS]
ALL_METRIC_IDS = [function.__name__ for function in ALL_METRICS]


@pytest.mark.parametrize("function", COUNT_METRICS, ids=COUNT_METRIC_IDS)
def test_a_count_metric_with_a_cohort_scores_that_cohorts_column(
    function: Callable[..., float],
) -> None:
    # Catches the cohort ignored (the whole matrix scored) or read at another
    # column: each column gives another value.
    expected = function(COHORT_COUNTS[:, 2], EXPECTED_COUNTS[:, 2])

    assert function(COHORT_FRAME, EXPECTED_FRAME, cohort="h") == pytest.approx(expected)
    assert function(COHORT_FRAME, EXPECTED_FRAME) != pytest.approx(expected)


@pytest.mark.parametrize("function", ALL_METRICS, ids=ALL_METRIC_IDS)
def test_a_cohort_y_true_does_not_name_is_rejected(
    function: Callable[..., float],
) -> None:
    # A misspelt name, and an array without names to look the cohort up in.
    with pytest.raises(KeyError, match="n_elementary"):
        function(COHORT_FRAME, EXPECTED_FRAME, cohort="n_elementary")
    with pytest.raises(TypeError, match="DataFrame"):
        function(COHORT_COUNTS, EXPECTED_COUNTS, cohort="h")


@pytest.mark.parametrize("function", ALL_METRICS, ids=ALL_METRIC_IDS)
@pytest.mark.parametrize("value", [np.nan, np.inf], ids=["nan", "inf"])
@pytest.mark.parametrize("side", ["y_true", "y_pred"])
def test_a_non_finite_value_is_rejected(
    function: Callable[..., float], value: float, side: str
) -> None:
    # Catches a NaN or an infinity passed through a sum as a NaN or inf score
    # (WMAPE and bias would), in either input.
    y_true, y_pred = COHORT_COUNTS.copy(), EXPECTED_COUNTS.copy()
    (y_true if side == "y_true" else y_pred)[0, 0] = value

    with pytest.raises(ValueError, match="must be finite"):
        function(y_true, y_pred)


def test_the_cohort_deviance_is_the_mean_of_the_cohort_deviances() -> None:
    # Each cohort weighs as its own CountModel's mean_poisson_deviance would.
    per_cohort = [
        mean_poisson_deviance(COHORT_COUNTS[:, k], EXPECTED_COUNTS[:, k])
        for k in range(3)
    ]

    assert mean_poisson_deviance_loss(COHORT_COUNTS, EXPECTED_COUNTS) == pytest.approx(
        np.mean(per_cohort)
    )


def test_the_cohort_rmse_pools_the_cells_not_the_cohort_rmses() -> None:
    # Unlike the MAE and the deviance, the mean of the per-cohort RMSEs differs
    # from the pooled one; scikit-learn's multi-output RMSE gives the former.
    per_cohort = [
        root_mean_squared_error_loss(COHORT_COUNTS[:, k], EXPECTED_COUNTS[:, k])
        for k in range(3)
    ]
    pooled = root_mean_squared_error_loss(COHORT_COUNTS, EXPECTED_COUNTS)

    assert pooled != pytest.approx(np.mean(per_cohort))
    assert pooled == pytest.approx(np.sqrt(np.mean([rmse**2 for rmse in per_cohort])))


def test_an_overprediction_has_a_positive_bias() -> None:
    # The sign is the metric: a flipped difference reads as under-prediction.
    assert mean_bias_error([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


@pytest.mark.parametrize("function", COUNT_METRICS, ids=COUNT_METRIC_IDS)
@pytest.mark.parametrize(
    "y_pred",
    [
        EXPECTED_COUNTS.T.copy(),
        EXPECTED_COUNTS.ravel(order="F"),
        EXPECTED_COUNTS[:, :1],
    ],
    ids=["transposed", "flattened", "one-column"],
)
def test_a_prediction_of_another_shape_is_rejected_by_every_count_metric(
    function: Callable[..., float], y_pred: np.ndarray
) -> None:
    # As many cells as y_true, or one column numpy would broadcast: once
    # raveled, scored mispaired, without an error from scikit-learn.
    with pytest.raises(ValueError, match="same shape"):
        function(COHORT_COUNTS, y_pred)


@pytest.mark.parametrize("function", COUNT_METRICS, ids=COUNT_METRIC_IDS)
def test_a_count_metric_pairs_dataframes_and_lists_by_position(
    function: Callable[..., float],
) -> None:
    # A DataFrame with other labels is not realigned (pandas would, into NaN).
    counts = pd.DataFrame(
        COHORT_COUNTS, index=[10, 11, 12, 13], columns=["a", "b", "c"]
    )
    expected = pd.DataFrame(EXPECTED_COUNTS, columns=["x", "y", "z"])
    reference = function(COHORT_COUNTS, EXPECTED_COUNTS)

    assert function(counts, expected) == pytest.approx(reference)
    assert function(COHORT_COUNTS.tolist(), EXPECTED_COUNTS.tolist()) == pytest.approx(
        reference
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: Metric("", mean_absolute_error),
        lambda: Metric("custom", 3),  # type: ignore[arg-type]
        lambda: Metric("custom", mean_absolute_error, "min"),  # type: ignore[arg-type]
        lambda: Metric("custom", mean_absolute_error, direction=True),  # type: ignore[arg-type]
        lambda: Metric("custom", mean_absolute_error, greater_is_better=True),  # type: ignore[call-arg]
    ],
    ids=[
        "empty-name",
        "not-callable",
        "unknown-direction",
        "bool-direction",
        "old-field",
    ],
)
def test_invalid_metric_is_rejected(build: Callable[[], Metric]) -> None:
    with pytest.raises((ValueError, TypeError)):
        build()


# --- evaluate --------------------------------------------------------------

EVALUATED = (
    Metric("poisson_deviance", mean_poisson_deviance_loss),
    Metric("mae", mean_absolute_error_loss),
)
Y_TRUE = pd.DataFrame(
    {"n_kindergarten": [1.0, 4.0, 2.0], "n_elementary": [2.0, 2.0, 5.0]}
)
Y_PRED = pd.DataFrame(
    {"n_kindergarten": [1.5, 2.0, 2.5], "n_elementary": [2.0, 3.0, 4.0]}
)


def test_evaluate_scores_every_metric_for_each_cohort_asked_for() -> None:
    cohorts = ["n_elementary", "composite", "n_kindergarten"]

    scores = evaluate(Y_TRUE, Y_PRED, EVALUATED, cohorts=cohorts)

    assert list(scores.index) == cohorts
    assert list(scores.columns) == ["poisson_deviance", "mae"]
    for cohort in cohorts:
        for metric in EVALUATED:
            expected = (
                metric(Y_TRUE, Y_PRED)
                if cohort == "composite"
                else metric(Y_TRUE, Y_PRED, cohort=cohort)
            )
            assert scores.loc[cohort, metric.name] == expected
    # A cohort's deviance is its column's alone: scikit-learn's on that column.
    assert scores.loc["n_kindergarten", "poisson_deviance"] == mean_poisson_deviance(
        Y_TRUE["n_kindergarten"], Y_PRED["n_kindergarten"]
    )


@pytest.mark.parametrize("cohorts", ["n_elementary", None])
def test_evaluate_of_one_name_or_the_default_is_one_row(cohorts: str | None) -> None:
    # The default is the composite; a name is one cohort, not its characters.
    scores = (
        evaluate(Y_TRUE, Y_PRED, EVALUATED)
        if cohorts is None
        else evaluate(Y_TRUE, Y_PRED, EVALUATED, cohorts=cohorts)
    )

    assert list(scores.index) == [cohorts or "composite"]
    assert scores.loc[cohorts or "composite", "mae"] == mean_absolute_error_loss(
        Y_TRUE, Y_PRED, cohort=cohorts
    )
