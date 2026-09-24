"""Model A: one LightGBM regressor for one cohort's child count."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from .base import BaseAgeGroupModel

__all__ = ["DirectCohortModel", "Objective"]

Objective = Literal["poisson", "regression"]


class DirectCohortModel(BaseAgeGroupModel):
    """A LightGBM regressor with fixed hyperparameters, fitted on one cohort.

    ``X`` is the finished design matrix: features are transformed before they
    reach the model. Cohorts are independent, so each gets its own instance and
    hyperparameters. These default to LightGBM's own and are tuned from outside
    through ``set_params``; nothing is searched in ``fit``.

    ``use_exposure`` (Poisson only) makes the raw exposure ``n`` (apartments),
    passed to ``fit`` and ``predict``, enter as the offset ``log n``. The trees
    then learn the cohort's rate per apartment rather than per building, and
    ``predict`` multiplies it back by ``n`` to give a count.
    """

    def __init__(
        self,
        *,
        objective: Objective = "poisson",
        use_exposure: bool = False,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_split_gain: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        # Stored verbatim, unvalidated: set_params assigns attributes without
        # re-entering __init__, so fit is where the configuration is checked.
        self.objective = objective
        self.use_exposure = use_exposure
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        # 1 by default: more OpenMP threads crash alongside torch on macOS.
        self.n_jobs = n_jobs

    def _check_exposure(self, exposure: ArrayLike | None) -> np.ndarray | None:
        """Return the exposure as floats, or ``None`` when ``use_exposure`` is off.

        Only what would otherwise pass silently is checked here; LightGBM itself
        rejects an unknown objective, a wrong-length exposure and an all-zero y.
        """
        if self.use_exposure and self.objective == "regression":
            raise ValueError(
                "an exposure offset needs a log link, which 'regression' lacks; "
                "set use_exposure=False or use objective='poisson'"
            )
        # A forgotten exposure would silently drop the offset, and an unexpected
        # one would be silently ignored.
        if (exposure is None) == self.use_exposure:
            raise ValueError("pass `exposure` exactly when use_exposure=True")
        if exposure is None:
            return None
        n = np.asarray(exposure, dtype=float)
        # log n of a non-positive value is -inf or nan, which LightGBM accepts.
        if not (np.isfinite(n) & (n > 0)).all():
            raise ValueError("exposure must be strictly positive and finite")
        return n

    def fit(
        self, X: pd.DataFrame, y: pd.Series, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit the trees on ``X`` and ``y``; ``exposure`` is the raw count ``n``."""
        n = self._check_exposure(exposure)
        init_score = None
        self.base_log_rate_: float | None = None
        if n is not None:
            # The intercept in log space: the average log rate per apartment,
            # log(sum y / sum n). LightGBM skips boost_from_average once given an
            # init_score, so without it the trees would start at 1 per apartment.
            self.base_log_rate_ = float(np.log(np.sum(y) / n.sum()))
            init_score = np.log(n) + self.base_log_rate_
        self.regressor_ = LGBMRegressor(
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_split_gain=self.min_split_gain,
            subsample=self.subsample,
            # LightGBM ignores subsample unless bagging runs at some frequency.
            subsample_freq=1 if self.subsample < 1 else 0,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            # Identical trees on every refit; LightGBM recommends force_col_wise with it.
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ).fit(X, y, init_score=init_score)
        return self

    def predict(self, X: pd.DataFrame, exposure: ArrayLike | None = None) -> np.ndarray:
        """The predicted mean count for each row of ``X``."""
        check_is_fitted(self)
        n = self._check_exposure(exposure)
        if n is None:
            return np.asarray(self.regressor_.predict(X))
        assert self.base_log_rate_ is not None
        # LightGBM's predict never adds the init_score back; the offset is ours.
        raw = self.regressor_.predict(X, raw_score=True)
        return np.asarray(np.exp(raw + np.log(n) + self.base_log_rate_))
