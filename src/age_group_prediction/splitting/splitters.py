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
from numbers import Integral
from typing import Literal, get_args

from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
)

from ..utils import DesignMatrix, Groups, Target, take_rows
from .stratified import StratifiedFolds, StratifiedHoldout

__all__ = ["DesignMatrix", "Groups", "Method", "Splitter", "Target"]

Method = Literal["random", "stratified_by_group", "grouped"]


@dataclass(frozen=True)
class Splitter:
    """A train/test split and the cross-validator that matches it.

    One ``Splitter`` drives both halves, so a run cannot pair one method's
    split with another's folds::

        splitter = Splitter("stratified_by_group")
        X_train, X_test, y_train, y_test, groups_train, groups_test = (
            splitter.train_test_split(X, y, groups, test_size=0.2, random_state=42)
        )
        cross_validate(
            estimator, X_train, y_train,
            cv=splitter.cv(n_splits=5, random_state=42), groups=groups_train,
        )

    Give the validator ``groups_train``, never ``groups``: it runs on the
    training rows only, and the whole table leaks the test set into tuning.
    """

    method: Method

    def __post_init__(self) -> None:
        # An unknown method would otherwise fall through to the grouped branch.
        if self.method not in get_args(Method):
            raise ValueError(
                f"unknown method {self.method!r}; expected {list(get_args(Method))}"
            )

    def train_test_split(
        self,
        X: DesignMatrix,
        y: Target,
        groups: Groups | None,
        *,
        test_size: float,
        random_state: int | None,
    ) -> tuple[
        DesignMatrix, DesignMatrix, Target, Target, Groups | None, Groups | None
    ]:
        """Split ``X``, ``y`` and ``groups``, as scikit-learn's function does.

        Returns two per array in scikit-learn's order; the groups come back
        because :meth:`cv` needs ``groups_train``. ``groups`` may be ``None``
        only for ``random``, which then returns ``None`` for both group pieces;
        the other methods split by groups and raise.
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
        else:
            holdout = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
        train_index, test_index = next(holdout.split(X, groups=keys))
        X_train, X_test = take_rows(X, train_index), take_rows(X, test_index)
        y_train, y_test = take_rows(y, train_index), take_rows(y, test_index)
        groups_train, groups_test = (
            (None, None)
            if groups is None
            else (take_rows(groups, train_index), take_rows(groups, test_index))
        )
        return X_train, X_test, y_train, y_test, groups_train, groups_test

    def cv(self, *, n_splits: int, random_state: int) -> BaseCrossValidator:
        """The validator for the training rows. Give it ``groups_train``.

        ``random_state`` must be an int: every ``split()`` call then returns the
        same folds, which tuning relies on when it re-splits in every trial.
        ``None`` or a ``RandomState`` would reshuffle on each call.
        """
        if not isinstance(random_state, Integral):
            raise TypeError(
                f"random_state must be an int, got {type(random_state).__name__}"
            )
        # Rows arrive sorted by group, so unshuffled folds would be grouped ones.
        if self.method == "random":
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if self.method == "stratified_by_group":
            return StratifiedFolds(n_splits=n_splits, random_state=random_state)
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
