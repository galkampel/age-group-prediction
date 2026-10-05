"""Model 2's second half: a Dirichlet regression of a building's cohort shares."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from scipy.special import softmax
from sklearn.utils.validation import check_is_fitted, check_X_y, validate_data
from torch.distributions import Dirichlet

from ..feature_engineering import FeatureTransformer
from .base import BaseAgeGroupModel
from .optimization import Minimizer, Solver, single_threaded_torch

__all__ = ["CohortProbabilityModel"]


class CohortProbabilityModel(BaseAgeGroupModel):
    """A Dirichlet regression of the cohort shares, ``s_b ~ Dirichlet(α_b)``.

    ``X`` is the raw table when ``feature_transformer`` is given, otherwise
    the finished design matrix, and ``y`` the cohort counts, one
    column per cohort (at least two), at least one child per building and
    no negative count. The building's composition ``s_b = y_b / Σ_k y_bk`` is modelled
    as a Dirichlet with ``α_bk = exp(a_k + x_b β_k)``; the predicted share is
    the Dirichlet mean ``α_bk / Σ_j α_bj``, which Model 2 multiplies by the
    predicted total. A share of 0 has no density, so the shares are compressed
    toward the centre, ``(s (N − 1) + 1/K) / N`` (Smithson & Verkuilen 2006,
    the standard in Dirichlet regression).

    ``l2_penalty`` multiplies ``½‖W‖²``, added to the mean negative
    log-density per building; the intercepts are not penalized. ``solver``,
    ``max_iter`` and ``tol`` are :class:`~age_group_prediction.modeling.optimization.Minimizer`'s.
    Calibration is fitted afterwards on :meth:`predict_logits`. The model has
    no offset, so a passed exposure is ignored.
    """

    def __init__(
        self,
        *,
        solver: Solver = "lbfgs",
        l2_penalty: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        feature_transformer: FeatureTransformer | None = None,
    ) -> None:
        self.solver = solver
        self.l2_penalty = l2_penalty
        self.max_iter = max_iter
        self.tol = tol
        self.feature_transformer = feature_transformer

    @staticmethod
    def _objective(
        parameters: np.ndarray, X: torch.Tensor, shares: torch.Tensor, l2_penalty: float
    ) -> tuple[float, np.ndarray]:
        """The penalized mean negative Dirichlet log-density and its gradient.

        scipy's optimizer calls it with a numpy point ``[a, W]``; torch gives
        the vectorized density and its exact gradient.
        """
        n_cohorts = shares.shape[1]
        params = torch.tensor(parameters, requires_grad=True)
        coefficients = params[n_cohorts:].reshape(-1, n_cohorts)
        alpha = torch.exp(params[:n_cohorts] + X @ coefficients)
        value = (
            -Dirichlet(alpha).log_prob(shares).mean()
            + 0.5 * l2_penalty * (coefficients * coefficients).sum()
        )
        value.backward()
        assert params.grad is not None
        return value.item(), params.grad.numpy()

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> Self:
        """Fit ``a`` and ``W`` on ``X`` and the cohort counts ``y``; ``exposure`` is ignored."""
        if self.l2_penalty < 0:
            raise ValueError(f"l2_penalty must be at least 0, got {self.l2_penalty}")
        feature_transformer, X = self._fit_features(X, y)
        X_values, counts = check_X_y(X, y, multi_output=True, y_numeric=True)
        cohorts = list(getattr(y, "columns", range(counts.shape[1])))
        # A cohort never observed would be fitted to the compressed floor silently.
        unobserved = [c for c, total in zip(cohorts, counts.sum(axis=0)) if total <= 0]
        if unobserved:
            raise ValueError(
                f"no child is observed in cohorts {unobserved}; drop them from y"
            )
        n_rows, n_cohorts = counts.shape
        shares = counts / counts.sum(axis=1, keepdims=True)
        shares = (shares * (n_rows - 1) + 1 / n_cohorts) / n_rows
        with single_threaded_torch():
            X_tensor = torch.tensor(X_values, dtype=torch.float64)
            shares_tensor = torch.tensor(shares, dtype=torch.float64)
            parameters = Minimizer(self.solver, self.max_iter, self.tol).minimize(
                lambda candidate: self._objective(
                    candidate, X_tensor, shares_tensor, self.l2_penalty
                ),
                np.zeros(n_cohorts * (1 + X_values.shape[1])),
            )
        self.intercept_: np.ndarray = parameters[:n_cohorts]
        self.coef_: np.ndarray = parameters[n_cohorts:].reshape(-1, n_cohorts)
        self.cohorts_: list[object] = cohorts
        self.feature_transformer_ = feature_transformer
        # Recorded after success, so a failed refit leaves the previous fit whole.
        validate_data(self, X, reset=True, skip_check_array=True)
        return self

    def predict_logits(self, X: pd.DataFrame) -> pd.DataFrame:
        """``log α = a + XW``, one column per cohort; what a calibrator is fitted on."""
        check_is_fitted(self)
        # Rejects columns in another order, which would be silent.
        X_values = validate_data(self, self._transform_features(X), reset=False)
        return pd.DataFrame(
            self.intercept_ + X_values @ self.coef_,
            columns=self.cohorts_,
            index=getattr(X, "index", None),
        )

    def predict(
        self, X: pd.DataFrame, exposure: ArrayLike | None = None
    ) -> pd.DataFrame:
        """Each cohort's predicted share, the Dirichlet mean; ``exposure`` is ignored."""
        # The raw X: predict_logits transforms it. A second transform here would
        # pass silently, since the design keeps the raw column names.
        logits = self.predict_logits(X)
        return pd.DataFrame(
            softmax(logits.to_numpy(), axis=1),
            columns=logits.columns,
            index=logits.index,
        )
