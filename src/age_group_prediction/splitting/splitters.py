"""Split methods: a train/test split paired with the validator that matches it.

``random``
    Groups are ignored, so a group can land wholly in the test set.
``stratified_by_group``
    Split within each group, so every group is on both sides.
``grouped``
    Split between groups, so a test group is never among the fit rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never, get_args

import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
)

from ..utils import DesignMatrix, Groups
from .stratified import StratifiedFolds, StratifiedHoldout

__all__ = ["Method", "Splitter"]

Method = Literal["random", "stratified_by_group", "grouped"]


@dataclass(frozen=True)
class Splitter:
    """A train/test split and the cross-validator that matches it.

    One ``Splitter`` drives both halves, so a run cannot pair one method's
    split with another's folds::

        splitter = Splitter("stratified_by_group")
        train_index, test_index = splitter.train_test_indices(
            X, groups, test_size=0.2, random_state=42
        )
        X_train, y_train = take_rows(X, train_index), take_rows(y, train_index)
        groups_train = take_rows(groups, train_index)
        cross_validate(
            estimator, X_train, y_train,
            cv=splitter.cv(n_splits=5, random_state=42), groups=groups_train,
        )

    Give the validator ``groups_train``, never ``groups``: it runs on the
    training rows only, and the whole table leaks the test set into tuning.
    """

    method: Method

    def __post_init__(self) -> None:
        # A method read from a config is never type-checked: fail here, not at
        # the first split.
        if self.method not in get_args(Method):
            raise ValueError(
                f"unknown method {self.method!r}; expected {list(get_args(Method))}"
            )

    def train_test_indices(
        self,
        X: DesignMatrix,
        groups: Groups | None,
        *,
        test_size: float,
        random_state: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The positions of the training rows and of the test rows of ``X``.

        Positions, as scikit-learn's splitters and :meth:`cv` return, so every
        array of the table's rows (``X``, ``y``, ``groups``, the exposure) is
        taken by the same positions, whatever its index, with
        :func:`~age_group_prediction.utils.take_rows`. ``groups`` may be
        ``None`` only for ``random``; the other methods split by groups and
        raise.
        """
        # ShuffleSplit ignores groups and warns if given them.
        keys = None if self.method == "random" else groups
        # n_splits=1 overrides sklearn defaults of 10 and 5: one draw, not many.
        if self.method == "random":
            holdout: BaseCrossValidator = ShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
        elif self.method == "stratified_by_group":
            holdout = StratifiedHoldout(test_size=test_size, random_state=random_state)
        elif self.method == "grouped":
            holdout = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
        else:
            # mypy flags this call if a new Method has no branch above.
            assert_never(self.method)
        train_index, test_index = next(holdout.split(X, groups=keys))
        return train_index, test_index

    def cv(self, *, n_splits: int, random_state: int) -> BaseCrossValidator:
        """The validator for the training rows. Give it ``groups_train``.

        ``random_state`` must be an int: every ``split()`` call then returns the
        same folds, which tuning relies on when it re-splits in every trial.
        ``None`` or a ``RandomState`` would reshuffle on each call.
        """
        # Not numbers.Integral: typeshed's int isn't one, so mypy skips the rest.
        if not isinstance(random_state, (int, np.integer)):
            raise TypeError(
                f"random_state must be an int, got {type(random_state).__name__}"
            )
        # Rows arrive sorted by group, so unshuffled folds would be grouped ones.
        if self.method == "random":
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if self.method == "stratified_by_group":
            return StratifiedFolds(n_splits=n_splits, random_state=random_state)
        if self.method == "grouped":
            return GroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        assert_never(self.method)
