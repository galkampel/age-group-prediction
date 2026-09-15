"""Unit tests for shared composition scoring kernels."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.metrics import brier_score_loss, log_loss

from age_group_prediction.metrics import (
    composition_brier_score,
    composition_log_loss,
    composition_log_loss_from_log_probabilities,
)
from age_group_prediction.models.probability_calibration import (
    _weighted_composition_nll,
    _weighted_proportion_squared_error,
)

COUNTS = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
PROBABILITIES = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.25, 0.25, 0.5]])


def _sklearn_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the counts into one weighted sklearn row per building and cohort."""
    cohort_count = COUNTS.shape[1]
    labels = np.tile(np.arange(cohort_count), COUNTS.shape[0])
    probabilities = np.repeat(PROBABILITIES, cohort_count, axis=0)
    return labels, probabilities, COUNTS.reshape(-1)


def test_composition_log_loss_matches_sklearn_oracle() -> None:
    """Composition log loss equals sklearn's count-weighted log loss."""
    labels, probabilities, weights = _sklearn_inputs()

    assert composition_log_loss(COUNTS, PROBABILITIES) == pytest.approx(
        log_loss(labels, probabilities, sample_weight=weights, labels=[0, 1, 2])
    )


def test_composition_brier_score_matches_sklearn_oracle() -> None:
    """Composition Brier score equals sklearn's count-weighted multiclass Brier score."""
    labels, probabilities, weights = _sklearn_inputs()

    assert composition_brier_score(COUNTS, PROBABILITIES) == pytest.approx(
        brier_score_loss(
            labels,
            probabilities,
            sample_weight=weights,
            labels=[0, 1, 2],
            scale_by_half=False,
        )
    )


def test_log_probability_kernel_matches_probability_kernel() -> None:
    """The log-probability kernel agrees with the probability kernel."""
    log_probabilities = np.log(PROBABILITIES)

    assert composition_log_loss_from_log_probabilities(
        COUNTS, log_probabilities
    ) == pytest.approx(composition_log_loss(COUNTS, PROBABILITIES))


def test_kernels_reject_zero_observed_children() -> None:
    """Both kernels refuse a table with no observed children."""
    zeros = np.zeros_like(COUNTS)
    message = "Composition metrics require positive observed totals"

    with pytest.raises(ValueError, match=message):
        composition_log_loss(zeros, PROBABILITIES)
    with pytest.raises(ValueError, match=message):
        composition_brier_score(zeros, PROBABILITIES)


def test_kernels_reject_misaligned_shapes() -> None:
    """Counts and probabilities must have the same shape."""
    with pytest.raises(ValueError, match="must share a shape"):
        composition_log_loss(COUNTS, PROBABILITIES[:, :2])


def test_log_loss_rejects_zero_probability_for_observed_cohort() -> None:
    """A zero probability for a cohort with observed children is refused."""
    probabilities = PROBABILITIES.copy()
    probabilities[0, 0] = 0.0

    with pytest.raises(ValueError, match="require positive probabilities"):
        composition_log_loss(COUNTS, probabilities)


def test_model_composition_nll_matches_pre_refactor_formula() -> None:
    """Model B's weighted composition NLL equals the explicit softmax formula."""
    logits = np.array([[1.5, -0.5, 0.25], [0.0, 2.0, -1.0], [-0.75, 0.5, 1.25]])
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    expected = -float(np.sum(COUNTS * log_probabilities) / COUNTS.sum())

    assert _weighted_composition_nll(logits, COUNTS) == pytest.approx(expected)


def test_share_error_is_not_the_per_child_brier_score() -> None:
    """The calibration diagnostic scores building shares, not individual children."""
    share_error = _weighted_proportion_squared_error(PROBABILITIES, COUNTS)

    assert share_error != pytest.approx(composition_brier_score(COUNTS, PROBABILITIES))
