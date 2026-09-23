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
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
)

from .stratified import StratifiedFolds, StratifiedHoldout

__all__ = ["DesignMatrix", "Groups", "Method", "Splitter", "Target"]

Method = Literal["random", "stratified_by_group", "grouped"]

type DesignMatrix = pd.DataFrame | np.ndarray  # the features, one row per unit
type Target = pd.Series | pd.DataFrame | np.ndarray  # one column, or several
type Groups = pd.Series | np.ndarray  # the split key, one label per row


def _take[T](array: T, index: np.ndarray) -> T:
    """The rows at ``index``: ``.iloc`` for pandas, plain indexing otherwise."""
    rows: Any = array
    return rows.iloc[index] if hasattr(array, "iloc") else rows[index]


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
        groups: Groups,
        *,
        test_size: float,
        random_state: int | None,
    ) -> tuple[DesignMatrix, DesignMatrix, Target, Target, Groups, Groups]:
        """Split ``X``, ``y`` and ``groups``, as scikit-learn's function does.

        Returns two per array in scikit-learn's order; the groups come back
        because :meth:`cv` needs ``groups_train``.
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
        # Arrays outermost gives scikit-learn's order. Unpacked into names
        # because a comprehension is variadic and would not typecheck.
        X_train, X_test, y_train, y_test, groups_train, groups_test = (
            _take(array, index)
            for array in (X, y, groups)
            for index in (train_index, test_index)
        )
        return X_train, X_test, y_train, y_test, groups_train, groups_test

    def cv(self, *, n_splits: int, random_state: int | None) -> BaseCrossValidator:
        """The validator for the training rows. Give it ``groups_train``."""
        # Rows arrive sorted by group, so unshuffled folds would be grouped ones.
        if self.method == "random":
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if self.method == "stratified_by_group":
            return StratifiedFolds(n_splits=n_splits, random_state=random_state)
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
