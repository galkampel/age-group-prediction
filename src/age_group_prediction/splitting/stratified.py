"""Splits that keep every stratum represented on both sides.

A row being predicted must have a stratum-mate among the rows used to fit, so a
row alone in its stratum stays in the fitting partition throughout. Nothing in
scikit-learn does this.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import BaseCrossValidator

__all__ = ["StratifiedFolds", "StratifiedHoldout"]


def _strata(groups: np.ndarray, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Yield each stratum's row positions, shuffled, one block per stratum.

    Skipping singletons is how they reach every fit set: a position never
    yielded is validated nowhere, and ``split`` fits on the complement.
    """
    for label in np.unique(groups):
        positions = np.flatnonzero(groups == label)
        if positions.size > 1:
            yield rng.permutation(positions)


class StratifiedHoldout(BaseCrossValidator):
    """One split holding out part of every stratum.

    A stratum of ``n`` rows gives up ``max(1, min(round(n * test_size), n - 1))``,
    so the realized share runs above ``test_size`` for small strata and below it
    for every singleton.
    """

    def __init__(
        self, test_size: float = 0.2, *, random_state: int | None = None
    ) -> None:
        if not 0 < test_size < 1:
            # Outside the interval the clamp below still returns a plausible
            # split, of the wrong size.
            raise ValueError(f"test_size must lie in (0, 1), got {test_size}")
        self.test_size = test_size
        self.random_state = random_state

    def get_n_splits(
        self, X: object = None, y: object = None, groups: object = None
    ) -> int:
        """One, whatever the data: a holdout is drawn once."""
        return 1

    def _iter_test_indices(
        self, X: object = None, y: object = None, groups: ArrayLike | None = None
    ) -> list[np.ndarray]:
        """The held-out rows; ``split`` keeps the complement to train on."""
        held = [
            positions[
                : max(
                    1, min(round(positions.size * self.test_size), positions.size - 1)
                )
            ]
            for positions in _strata(
                np.asarray(groups), np.random.default_rng(self.random_state)
            )
        ]
        return [np.concatenate(held) if held else np.empty(0, dtype=int)]


class StratifiedFolds(BaseCrossValidator):
    """Folds that keep every stratum in every fit set.

    The strata are laid end to end and dealt round-robin, so every row with a
    stratum-mate is validated exactly once. Dealing each stratum from its own
    start would pile every small stratum into fold 0.

    A row alone in its stratum is in every fit set and no validation set.
    """

    def __init__(self, n_splits: int = 5, *, random_state: int | None = None) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        # random_state stays an int: a held Generator advances, so two split()
        # calls would deal different folds.
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(
        self, X: object = None, y: object = None, groups: object = None
    ) -> int:
        """The number of folds, whatever the data."""
        return self.n_splits

    def _iter_test_indices(
        self, X: object = None, y: object = None, groups: ArrayLike | None = None
    ) -> list[np.ndarray]:
        """The rows each fold holds out -- one array of positions per fold.

        Only this half is returned; ``split`` fits on the complement, so a
        position returned by no fold is fitted in every one, which is how
        singletons stay in training.
        """
        rng = np.random.default_rng(self.random_state)
        shuffled = list(_strata(np.asarray(groups), rng))
        ordered = np.concatenate(shuffled) if shuffled else np.empty(0, dtype=int)
        if ordered.size < self.n_splits:
            # Fold f takes a row only when f < size, so a shortfall is an empty
            # fold -- which scores nan, and argmax over a nan returns its index.
            raise ValueError(
                f"{ordered.size} rows have a stratum-mate, fewer than "
                f"{self.n_splits} folds"
            )
        # Round-robin over the strata end to end, never contiguous blocks:
        # array_split would put a whole small stratum in one fold.
        slot = np.arange(ordered.size) % self.n_splits
        # A list, not a generator, so the check above raises at split().
        return [ordered[slot == fold] for fold in range(self.n_splits)]
