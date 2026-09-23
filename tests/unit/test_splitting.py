"""Tests for the stratified holdout and the stratified cross-validation folds.

Expected values are written out by hand, so a test fails when the arithmetic
changes rather than following it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate

from age_group_prediction.splitting import StratifiedFolds, StratifiedHoldout


def _groups() -> np.ndarray:
    """Ten rows: N1 has five, N2 three, N3 and N4 one each."""
    return np.array(["N1"] * 5 + ["N2"] * 3 + ["N3", "N4"])


def _X(n_rows: int = 10) -> np.ndarray:
    """A frame-shaped stand-in; the splitter reads only its length."""
    return np.arange(n_rows, dtype=float).reshape(-1, 1)


def _holdout(
    groups: np.ndarray, *, test_size: float = 0.2, random_state: int | None = 0
) -> tuple[np.ndarray, np.ndarray]:
    """The one split of StratifiedHoldout, as ``(train, test)``."""
    splitter = StratifiedHoldout(test_size=test_size, random_state=random_state)
    return next(splitter.split(_X(groups.size), groups=groups))


# --- Holdout values ---------------------------------------------------


def test_each_stratum_gives_up_the_rounded_fraction_of_its_rows() -> None:
    # N1: round(5 * 0.2) = 1. N2: round(3 * 0.2) = 1. The singletons give none.
    _, test_index = _holdout(_groups(), test_size=0.2)

    assert test_index.size == 2
    assert sorted(_groups()[test_index]) == ["N1", "N2"]


def test_a_stratum_of_two_gives_up_exactly_one_row_at_any_test_size() -> None:
    # The floor of one and the ceiling of n - 1 meet when n is two.
    groups = np.array(["A", "A", "B", "B"])

    for test_size in (0.05, 0.95):
        _, test_index = _holdout(groups, test_size=test_size)
        assert sorted(groups[test_index]) == ["A", "B"]


def test_a_singleton_stratum_is_never_held_out() -> None:
    # The one row of N3 or N4 would leave its stratum unrepresented in training.
    train_index, test_index = _holdout(_groups(), test_size=0.9)

    assert set(_groups()[test_index]) == {"N1", "N2"}
    assert {8, 9} <= set(train_index.tolist())


# --- Holdout structure ------------------------------------------------


def test_every_non_singleton_stratum_appears_on_both_sides() -> None:
    train_index, test_index = _holdout(_groups(), test_size=0.2)

    assert {"N1", "N2"} <= set(_groups()[train_index])
    assert set(_groups()[test_index]) == {"N1", "N2"}


def test_the_two_indices_partition_the_rows_exactly_once() -> None:
    train_index, test_index = _holdout(_groups(), test_size=0.4)

    assert sorted(np.concatenate([train_index, test_index]).tolist()) == list(range(10))


def test_the_same_random_state_reproduces_the_split_and_a_different_one_moves_it() -> (
    None
):
    first = _holdout(_groups(), test_size=0.4, random_state=7)
    again = _holdout(_groups(), test_size=0.4, random_state=7)
    other = _holdout(_groups(), test_size=0.4, random_state=8)

    np.testing.assert_array_equal(first[1], again[1])
    assert not np.array_equal(first[1], other[1])


# --- Fold values ------------------------------------------------------


def test_every_row_with_a_stratum_mate_is_validated_exactly_once() -> None:
    folds = list(
        StratifiedFolds(n_splits=3, random_state=0).split(_X(), groups=_groups())
    )

    validated = np.concatenate([test_index for _, test_index in folds])
    assert sorted(validated.tolist()) == list(range(8))


def test_a_singleton_is_in_every_fit_set_and_no_validation_set() -> None:
    folds = list(
        StratifiedFolds(n_splits=3, random_state=0).split(_X(), groups=_groups())
    )

    for train_index, test_index in folds:
        assert {8, 9} <= set(train_index.tolist())
        assert not {8, 9} & set(test_index.tolist())


def test_fold_sizes_differ_by_at_most_one() -> None:
    # Eight rows have a stratum-mate, so three folds hold three, three and two.
    folds = list(
        StratifiedFolds(n_splits=3, random_state=0).split(_X(), groups=_groups())
    )

    sizes = sorted(test_index.size for _, test_index in folds)
    assert sizes == [2, 3, 3]


def test_the_deal_carries_between_strata_instead_of_restarting() -> None:
    # Three strata of two, three folds. Dealing each stratum from its own start
    # would put every stratum's first row into fold 0 and its second into fold
    # 1, giving 3 / 3 / 0 and an empty fold.
    groups = np.array(["A", "A", "B", "B", "C", "C"])

    folds = list(
        StratifiedFolds(n_splits=3, random_state=0).split(_X(6), groups=groups)
    )

    assert [test_index.size for _, test_index in folds] == [2, 2, 2]


def test_no_stratum_has_all_its_rows_in_one_fold() -> None:
    # Five strata of two into five folds is the case that separates a
    # round-robin deal from np.array_split: contiguous blocks of two would put
    # each stratum wholly in one fold, absent from that fold's fit set. Every
    # other test in this file passes under either.
    groups = np.array([f"S{index}" for index in range(5) for _ in range(2)])

    for fit_index, validation_index in StratifiedFolds(5, random_state=0).split(
        _X(10), groups=groups
    ):
        assert set(groups[validation_index]) <= set(groups[fit_index])


# --- The deployment claim ---------------------------------------------


def test_every_validated_row_has_a_stratum_mate_among_the_fit_rows() -> None:
    # A new building in a known neighborhood: the claim the module exists for.
    groups = _groups()

    for train_index, test_index in StratifiedFolds(n_splits=3, random_state=0).split(
        _X(), groups=groups
    ):
        assert set(groups[test_index]) <= set(groups[train_index])


# --- scikit-learn contract --------------------------------------------


def test_get_n_splits_matches_the_number_of_folds_yielded() -> None:
    splitter = StratifiedFolds(n_splits=4, random_state=0)

    folds = list(splitter.split(_X(), groups=_groups()))
    assert len(folds) == splitter.get_n_splits()


def test_the_fit_and_validation_rows_of_a_fold_never_overlap() -> None:
    for train_index, test_index in StratifiedFolds(n_splits=3, random_state=0).split(
        _X(), groups=_groups()
    ):
        assert not set(train_index.tolist()) & set(test_index.tolist())
        assert sorted(np.concatenate([train_index, test_index]).tolist()) == list(
            range(10)
        )


def test_two_split_calls_on_one_instance_yield_identical_folds() -> None:
    # An Optuna study reuses one splitter across every trial; a stored Generator
    # would advance and make the trials incomparable.
    splitter = StratifiedFolds(n_splits=3, random_state=0)

    first = [test_index for _, test_index in splitter.split(_X(), groups=_groups())]
    again = [test_index for _, test_index in splitter.split(_X(), groups=_groups())]

    for one, other in zip(first, again, strict=True):
        np.testing.assert_array_equal(one, other)


def test_cross_validate_scores_every_fold() -> None:
    # Proves groups reaches split and that the complement is sklearn's own.
    scores = cross_validate(
        DummyRegressor(),
        _X(),
        np.arange(10, dtype=float),
        cv=StratifiedFolds(n_splits=3, random_state=0),
        groups=_groups(),
    )

    assert np.isfinite(scores["test_score"]).all()


# --- Failures ---------------------------------------------------------


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_a_test_size_outside_the_unit_interval_is_rejected(test_size: float) -> None:
    # Both ends return a plausible split of the wrong size if left unchecked.
    with pytest.raises(ValueError, match="test_size"):
        _holdout(_groups(), test_size=test_size)


def test_a_single_fold_is_rejected_at_construction() -> None:
    # It would validate everything and fit on the singletons alone. Rejected in
    # __init__, where sklearn's own _BaseKFold rejects it.
    with pytest.raises(ValueError, match="n_splits"):
        StratifiedFolds(n_splits=1)


def test_fewer_rows_with_a_stratum_mate_than_folds_is_rejected() -> None:
    # Two rows have a mate, so a third fold would be empty and score nan.
    groups = np.array(["A", "A", "B"])

    with pytest.raises(ValueError, match="fewer than"):
        list(StratifiedFolds(n_splits=3).split(_X(3), groups=groups))
