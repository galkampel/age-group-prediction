"""SHAP values of a fitted model, per raw column of ``X``.

Shapley values are exact for the prediction function they explain. shap
predicts on synthetic rows that mix one row of ``X`` with ``background``
rows, and it hands those rows over without saying which row of ``X`` they
belong to; so the rows are explained one at a time, each with its own
exposure. Non-numeric columns reach shap as integer codes and are decoded
before each ``predict``; ``Explanation.data`` therefore holds codes and
``display_data`` the original values. Every value travels as float64, so an
integer column above 2**53 loses precision on the way.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from pydantic.dataclasses import dataclass as pydantic_dataclass
from sklearn.preprocessing import OrdinalEncoder

from .._config import STRICT
from ..modeling import BaseAgeGroupModel
from ..utils import Exposure

__all__ = ["Method", "ShapExplainer"]

Method = Literal["exact", "permutation"]


# Settings are checked when the object is built, not after the evaluations.
@pydantic_dataclass(frozen=True, config=STRICT)
class ShapExplainer:
    """Shapley values of the prediction for each row of ``X``, one per raw column.

    Both methods are model-agnostic and additive: ``base_values +
    values.sum(axis=1)`` equals the prediction. They score a row against
    ``background``, the rows whose values stand in for an absent column.
    ``"exact"`` evaluates every coalition of the columns, ``2 ** n_columns``
    model evaluations per row: exact and cheap up to about ten columns.
    ``"permutation"`` samples permutations within ``max_evals`` evaluations
    per row (shap's default 500, at least ``2 * n_columns + 1``):
    approximate, seeded by ``random_state``, for more columns. ``max_evals``
    left as ``None`` is shap's default for the method. Cost grows with
    ``len(background)`` per evaluation, so pass a few dozen background rows.
    Rows are explained one at a time (no progress bar), ``n_jobs`` of them
    in parallel processes as in scikit-learn (``-1`` for all cores): worth
    it for a run of minutes, slower than ``1`` for a run of seconds.
    """

    method: Method
    max_evals: int | None = None
    random_state: int | None = None
    n_jobs: int = 1

    def __post_init__(self) -> None:
        # The exact method is deterministic: a seed would be ignored silently.
        if self.method == "exact" and self.random_state is not None:
            raise ValueError("random_state has no effect with method='exact'")

    def explain(
        self,
        model: BaseAgeGroupModel,
        X: pd.DataFrame,
        background: pd.DataFrame,
        *,
        exposure: Exposure | None = None,
    ) -> shap.Explanation:
        """SHAP values of ``model.predict(X, exposure)`` against ``background``.

        Both frames hold the raw columns the model was fitted on; ``exposure``
        is paired with ``X`` by position. Without it the values explain the
        rate, ``predict(X, exposure=1)``; with it, the count. The result's
        ``feature_names`` are ``X``'s columns and, for a cohort model, its
        ``output_names`` are the cohort columns ``predict`` returns.
        """
        exposure_values = (
            np.ones(len(X)) if exposure is None else np.asarray(exposure, dtype=float)
        )
        # The loop below runs over the exposure, so a longer one would pass.
        if len(exposure_values) != len(X):
            raise ValueError(
                f"exposure has {len(exposure_values)} values for {len(X)} rows"
            )
        non_numeric_cols = [
            column
            for column in X.columns
            if not pd.api.types.is_numeric_dtype(X[column])
        ]
        # The levels of both frames, so a level seen only in X still decodes.
        encoder = OrdinalEncoder().fit(pd.concat([X, background])[non_numeric_cols])
        dtypes = X.dtypes.to_dict()
        for column, categories in zip(non_numeric_cols, encoder.categories_):
            # A categorical dtype of X alone would turn a background-only
            # level into NaN on the way back.
            if isinstance(dtypes[column], pd.CategoricalDtype):
                dtypes[column] = pd.CategoricalDtype(
                    categories, ordered=dtypes[column].ordered
                )

        def encode(X: pd.DataFrame) -> pd.DataFrame:
            # shap's masker needs one float matrix: a str or bool column breaks
            # it. predict restores the dtypes.
            X_encoded = X.copy()
            X_encoded[non_numeric_cols] = encoder.transform(X[non_numeric_cols])
            return X_encoded.astype(float)

        def predict(X_encoded: pd.DataFrame, exposure_value: float) -> np.ndarray:
            # shap builds a fresh frame with X's columns for every call; its
            # rows are variants of one row of X, hence one exposure value.
            X_rows = X_encoded.copy()
            # scikit-learn refuses inverse_transform on zero columns.
            if non_numeric_cols:
                # A copy: scikit-learn writes NaN codes into the array it is
                # given, and pandas hands out a read-only one.
                X_rows[non_numeric_cols] = encoder.inverse_transform(
                    X_rows[non_numeric_cols].to_numpy(copy=True)
                )
            y_pred = model.predict(
                X_rows.astype(dtypes), exposure=np.full(len(X_rows), exposure_value)
            )
            return np.asarray(y_pred, dtype=float)

        # The masker defines "column absent": Independent replaces the absent
        # columns of a row by each background row's values in turn and shap
        # averages the predictions, so base_values is the mean prediction
        # over the background. shap keeps 100 background rows by default,
        # silently.
        # shap pairs the masker's columns with X's by position.
        masker = shap.maskers.Independent(
            encode(background[X.columns]), max_samples=len(background)
        )
        # Both explainers validate max_evals against their own minimum.
        kwargs = {} if self.max_evals is None else {"max_evals": self.max_evals}
        X_encoded = encode(X)

        def explain_row(pos: int, exposure_value: float) -> shap.Explanation:
            predict_row = partial(predict, exposure_value=exposure_value)
            if self.method == "exact":
                explainer = shap.ExactExplainer(predict_row, masker)
            elif self.method == "permutation":
                explainer = shap.PermutationExplainer(
                    predict_row, masker, seed=self.random_state
                )
            else:
                raise ValueError(f"unknown method {self.method!r}")
            return explainer(X_encoded.iloc[[pos]], **kwargs)

        row_explanations = Parallel(n_jobs=self.n_jobs)(
            delayed(explain_row)(pos, exposure_value)
            for pos, exposure_value in enumerate(exposure_values)
        )
        y_pred = model.predict(X.iloc[:1], exposure=exposure_values[:1])
        return shap.Explanation(
            values=np.concatenate([row.values for row in row_explanations]),
            base_values=np.concatenate([row.base_values for row in row_explanations]),
            data=np.concatenate([row.data for row in row_explanations]),
            # The plots label with the original values, not the codes.
            display_data=X.to_numpy(dtype=object),
            feature_names=list(X.columns),
            # shap labels several outputs "Output 0..K-1".
            output_names=(
                list(y_pred.columns) if isinstance(y_pred, pd.DataFrame) else None
            ),
        )
