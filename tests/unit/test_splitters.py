"""Tests for the split methods.

Each method is defined by what it guarantees about a test row's group, so the
tests assert those guarantees directly rather than fixed index values.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    cross_validate,
)

from age_group_prediction.splitting import Splitter, StratifiedFolds, StratifiedHoldout
from age_group_prediction.utils import take_rows

METHODS = ("random", "stratified_by_group", "grouped")


def _frame(
    rows_per_group: int = 4, n_groups: int = 6
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """``(X, y, groups)`` on a non-default index, so row identity is visible."""
    size = rows_per_group * n_groups
    index = pd.RangeIndex(100, 100 + size)
    groups = pd.Series(
        [f"N{g}" for g in range(1, n_groups + 1) for _ in range(rows_per_group)],
        index=index,
    )
    X = pd.DataFrame({"a": np.arange(size, dtype=float)}, index=index)
    return X, pd.Series(np.arange(size, dtype=float), index=index), groups


def _pairs() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Five groups of two -- small enough for a random split to swallow one."""
    return _frame(rows_per_group=2, n_groups=5)


def _split(
    method: str, seed: int, test_size: float = 0.2, pairs: bool = False
) -> tuple[pd.Series, pd.Series]:
    """``(groups_train, groups_test)``, for whichever fixture the test wants."""
    X, _, groups = _pairs() if pairs else _frame()
    train_index, test_index = Splitter(method).train_test_indices(
        X, groups, test_size=test_size, random_state=seed
    )
    return groups.iloc[train_index], groups.iloc[test_index]


# --- What each method guarantees --------------------------------------


def test_stratified_by_group_always_leaves_every_test_group_represented() -> None:
    for seed in range(50):
        groups_train, groups_test = _split("stratified_by_group", seed, pairs=True)
        assert set(groups_test) <= set(groups_train)


def test_random_sometimes_leaves_a_group_unrepresented() -> None:
    # The difference from stratified_by_group, and the reason the Bayesian
    # model's fallback path exists. Scanned, not pinned to one lucky seed.
    swallowed = [
        seed
        for seed in range(50)
        if not set(_split("random", seed, pairs=True)[1])
        <= set(_split("random", seed, pairs=True)[0])
    ]
    assert swallowed, "expected at least one seed to hold out a whole group"


def test_grouped_never_leaves_a_test_group_represented() -> None:
    # The opposite guarantee: a test group is never among the fit rows.
    for seed in range(20):
        groups_train, groups_test = _split("grouped", seed)
        assert not set(groups_test) & set(groups_train)


# --- What train_test_indices returns ----------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_every_row_lands_on_exactly_one_side(method: str) -> None:
    # A row lost, or in both halves, would leak or waste data silently.
    X, _, groups = _frame()
    train_index, test_index = Splitter(method).train_test_indices(
        X, groups, test_size=0.25, random_state=0
    )

    np.testing.assert_array_equal(
        np.sort(np.concatenate([train_index, test_index])), np.arange(len(X))
    )


@pytest.mark.parametrize("method", METHODS)
def test_the_indices_are_positions_whatever_the_index(method: str) -> None:
    # Every array (the exposure too) is taken by these positions. Labels
    # returned as positions would take other rows, silently once the labels
    # fall in range; _frame's labels start at 100.
    X, _, groups = _frame()
    labelled = Splitter(method).train_test_indices(
        X, groups, test_size=0.25, random_state=0
    )
    plain = Splitter(method).train_test_indices(
        X.reset_index(drop=True),
        groups.reset_index(drop=True),
        test_size=0.25,
        random_state=0,
    )

    for index, expected in zip(labelled, plain, strict=True):
        np.testing.assert_array_equal(index, expected)


@pytest.mark.parametrize("method", METHODS)
def test_the_same_random_state_reproduces_the_split(method: str) -> None:
    X, _, groups = _frame()
    first, again = (
        Splitter(method).train_test_indices(X, groups, test_size=0.2, random_state=7)
        for _ in range(2)
    )

    for index, repeated in zip(first, again, strict=True):
        np.testing.assert_array_equal(index, repeated)


@pytest.mark.parametrize("method", METHODS)
def test_a_different_random_state_moves_the_split(method: str) -> None:
    # The reason the docstring says to draw it once.
    assert list(_split(method, 1)[1].index) != list(_split(method, 2)[1].index)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_holds_out_some_rows_and_keeps_most(method: str) -> None:
    groups_train, groups_test = _split(method, 0)

    assert 0 < len(groups_test) < len(groups_train)


def test_it_works_on_numpy_arrays_too() -> None:
    # No pandas anywhere: the indices take rows of plain arrays as well.
    groups = np.array([f"N{g}" for g in range(1, 7) for _ in range(4)])
    X = np.arange(24.0).reshape(-1, 1)
    train_index, test_index = Splitter("stratified_by_group").train_test_indices(
        X, groups, test_size=0.25, random_state=0
    )

    assert len(train_index) + len(test_index) == 24
    assert set(take_rows(groups, test_index)) <= set(take_rows(groups, train_index))


def test_random_takes_groups_none() -> None:
    # random never reads groups, and its validator warns if given them, so a
    # caller without groups must not have to invent some.
    X, _, _ = _frame()
    train_index, test_index = Splitter("random").train_test_indices(
        X, None, test_size=0.25, random_state=0
    )

    assert len(train_index) + len(test_index) == len(X)


# --- The cross-validator ----------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("random", KFold),
        ("stratified_by_group", StratifiedFolds),
        ("grouped", GroupKFold),
    ],
)
def test_cv_returns_the_sklearn_validator_the_method_names(
    method: str, expected: type
) -> None:
    validator = Splitter(method).cv(n_splits=3, random_state=0)

    assert isinstance(validator, expected)
    assert validator.get_n_splits() == 3


@pytest.mark.parametrize(
    ("method", "holdout", "validator"),
    [
        ("random", ShuffleSplit, KFold),
        ("stratified_by_group", StratifiedHoldout, StratifiedFolds),
        ("grouped", GroupShuffleSplit, GroupKFold),
    ],
)
def test_each_method_pairs_one_split_with_one_validator(
    method: str, holdout: type, validator: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dispatch lives in two methods, so nothing structural stops a future
    # edit pairing one method's split with another's folds. This is that guard.
    seen: list[type] = []
    for cls in (ShuffleSplit, StratifiedHoldout, GroupShuffleSplit):
        original = cls.__init__

        def record(self, *args, _cls=cls, _original=original, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(_cls)
            _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", record)

    X, _, groups = _frame()
    Splitter(method).train_test_indices(X, groups, test_size=0.2, random_state=0)

    assert seen == [holdout]
    assert isinstance(Splitter(method).cv(n_splits=3, random_state=0), validator)


@pytest.mark.parametrize("method", METHODS)
def test_the_validator_yields_the_folds_it_promises(method: str) -> None:
    X, _, groups = _frame()
    validator = Splitter(method).cv(n_splits=3, random_state=0)

    # groups only where the validator reads them; see the test below.
    keywords = {} if method == "random" else {"groups": groups}
    folds = list(validator.split(X, **keywords))

    assert len(folds) == 3
    for fit_index, validation_index in folds:
        assert not set(fit_index.tolist()) & set(validation_index.tolist())


def test_random_ignores_groups_and_says_so() -> None:
    # The wart of the uniform call site: passing groups to every method makes
    # KFold warn. Documented rather than papered over with a wrapper class.
    X, _, groups = _frame()

    with pytest.warns(UserWarning, match="groups parameter is ignored"):
        list(Splitter("random").cv(n_splits=3, random_state=0).split(X, groups=groups))


def test_the_grouped_validator_keeps_whole_groups_out_of_each_fit_set() -> None:
    X, _, groups = _frame()
    validator = Splitter("grouped").cv(n_splits=3, random_state=0)

    for fit_index, validation_index in validator.split(X, groups=groups):
        assert not set(groups.iloc[validation_index]) & set(groups.iloc[fit_index])


# --- The two halves together ------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_the_documented_two_step_flow_scores_every_fold(method: str) -> None:
    X, y, groups = _frame(rows_per_group=6, n_groups=12)
    # random reads no groups, and KFold warns if given them.
    groups = None if method == "random" else groups
    splitter = Splitter(method)

    train_index, _ = splitter.train_test_indices(
        X, groups, test_size=0.2, random_state=42
    )
    X_train, y_train = take_rows(X, train_index), take_rows(y, train_index)
    groups_train = None if groups is None else take_rows(groups, train_index)
    scores = cross_validate(
        DummyRegressor(),
        X_train,
        y_train,
        cv=splitter.cv(n_splits=5, random_state=42),
        groups=groups_train,
    )

    assert np.isfinite(scores["test_score"]).all()


# --- The configuration ------------------------------------------------


def test_the_repr_names_the_method() -> None:
    assert repr(Splitter("grouped")) == "Splitter(method='grouped')"


def test_the_method_is_frozen() -> None:
    # Both halves dispatch on it, so a mutable method could split them.
    splitter = Splitter("random")

    with pytest.raises(FrozenInstanceError):
        splitter.method = "grouped"  # type: ignore[misc]


# --- Failures ---------------------------------------------------------


def test_an_unknown_method_is_rejected_at_construction() -> None:
    # A config string is never type-checked, so the check is at runtime.
    with pytest.raises(ValueError, match="unknown method"):
        Splitter("temporal")  # type: ignore[arg-type]


def test_a_single_fold_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        Splitter("stratified_by_group").cv(n_splits=1, random_state=0)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_a_test_size_outside_the_unit_interval_is_rejected(test_size: float) -> None:
    # Left to each holdout class, which already names the parameter.
    with pytest.raises(ValueError, match="test_size|should be"):
        _split("stratified_by_group", 0, test_size=test_size)


@pytest.mark.parametrize("method", ["stratified_by_group", "grouped"])
def test_the_group_aware_methods_reject_groups_none(method: str) -> None:
    # One message for both: StratifiedHoldout raises sklearn's own wording.
    X, _, _ = _frame()

    with pytest.raises(ValueError, match="should not be None"):
        Splitter(method).train_test_indices(X, None, test_size=0.25, random_state=0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "random_state", [None, np.random.RandomState(0)], ids=["none", "random-state"]
)
def test_cv_requires_an_int_seed(method: str, random_state: object) -> None:
    # Either would reshuffle on every split() call, and tuning re-splits per trial.
    with pytest.raises(TypeError, match="random_state must be an int"):
        Splitter(method).cv(n_splits=3, random_state=random_state)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", METHODS)
def test_repeated_splits_give_identical_folds(method: str) -> None:
    X, _, groups = _frame()
    validator = Splitter(method).cv(n_splits=3, random_state=np.int64(0))
    keywords = {} if method == "random" else {"groups": groups}

    first, second = (
        [val.tolist() for _, val in validator.split(X, **keywords)] for _ in range(2)
    )

    assert first == second


def test_there_are_no_default_parameters() -> None:
    # Sizing and seeding are decisions, so the call site has to state them.
    X, _, groups = _frame()

    with pytest.raises(TypeError):
        Splitter("random").train_test_indices(X, groups)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Splitter("random").cv()  # type: ignore[call-arg]
