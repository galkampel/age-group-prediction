"""Given a method, a train/test split and a K-fold cross-validator.

:class:`Splitter` pairs the two; :mod:`~age_group_prediction.splitting.splitters`
says what each method guarantees.

A split is a pure function of ``(row order, random_state)`` -- sort by
``building_id`` at the call site if it must survive a reordering of the table.
"""

from __future__ import annotations

from .splitters import Method, Splitter
from .stratified import StratifiedFolds, StratifiedHoldout

__all__ = ["Method", "Splitter", "StratifiedFolds", "StratifiedHoldout"]
