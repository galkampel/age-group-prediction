"""Declaring a design matrix and fitting it to data."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from .transforms import Transform

# One ColumnTransformer entry, in sklearn's (name, transformer, columns) form.
# Columns are names, except in the interaction step's base entry, which
# selects every column at fit time: the first step's output names aren't
# known until it has run.
_TransformerTuple = tuple[
    str,
    TransformerMixin | Pipeline | str,
    list[str] | Callable[[pd.DataFrame], list[str]],
]


class ColumnPlan(BaseModel):
    """One named group of columns and the transforms applied to them.

    Columns are grouped by shared treatment. The same column may appear in
    several plans, which is how a derived term joins the one it derives from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    columns: tuple[str, ...] = Field(min_length=1)
    transforms: tuple[Transform, ...] = ()

    @field_validator("columns", mode="before")
    @classmethod
    def _accept_a_bare_column(cls, value: str | Sequence[str]) -> Sequence[str]:
        """Let a one-column plan pass the name itself rather than a list."""
        return (value,) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_input_signs(self) -> ColumnPlan:
        """Reject a step needing non-negative input once negatives are certain."""
        # The step after which negatives are certain. Only then is the plan
        # refused; otherwise a bad value is the data's, caught at fit or
        # transform by the step's own input check or the finite-output check.
        negative_since: str | None = None
        for transform in self.transforms:
            if transform.needs_nonnegative_input and negative_since is not None:
                raise ValueError(
                    f"Plan {self.name!r}: {transform.kind} needs non-negative "
                    f"input, but {negative_since} before it always emits negative "
                    f"values; apply {transform.kind} before {negative_since}"
                )
            if transform.negative_output == "always":
                negative_since = transform.kind
            elif transform.negative_output in ("depends_on_data", "never"):
                negative_since = None
        return self

    def build(self) -> TransformerMixin | Pipeline | str:
        """Return the transformer for this plan's ``ColumnTransformer`` entry."""
        if not self.transforms:
            return "passthrough"
        if len(self.transforms) == 1:
            return self.transforms[0].build()
        # Indexed step names: a chain may repeat a kind, and sklearn rejects a
        # Pipeline whose steps share a name.
        return Pipeline(
            [
                (f"{index}_{transform.kind}", transform.build())
                for index, transform in enumerate(self.transforms)
            ]
        )


class Interaction(BaseModel):
    """One product column, multiplied out of two columns of the design matrix.

    Interactions run after the plans, on the matrix they produce, so ``left``
    and ``right`` name what a plan emits -- ``ses_squared``, not ``ses``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    left: str
    right: str

    @property
    def columns(self) -> tuple[str, str]:
        """The two columns it multiplies."""
        return (self.left, self.right)

    @property
    def name(self) -> str:
        """The design-matrix column it adds."""
        return _product_name(self.columns)

    def build(self) -> TransformerMixin:
        """Return the transformer that multiplies those two columns into one."""
        return FunctionTransformer(
            lambda df: df.prod(axis=1).to_frame(),
            validate=False,
            # The same helper name() uses, so the declared name and the fitted
            # one cannot disagree.
            feature_names_out=lambda _, columns: [_product_name(columns)],
        )


class FeatureTransformer(TransformerMixin, BaseEstimator):
    """Turns a set of column plans into a fitted design matrix.

    Statistics are learned in :meth:`fit` and reused by every
    :meth:`transform`, so a validation fold is centered by the training fold's
    means. ``sklearn.base.clone`` gives a fresh unfitted copy per fold.
    """

    def __init__(
        self,
        plans: tuple[ColumnPlan, ...] = (),
        *,
        interactions: Sequence[Interaction] = (),
        exposure_column: str | None = None,
        remainder: str = "drop",
    ) -> None:
        # Stored verbatim: set_params assigns attributes without re-entering
        # __init__, so validating here would be skipped by GridSearchCV.
        self.plans = plans
        self.interactions = interactions
        self.exposure_column = exposure_column
        self.remainder = remainder

    def validate(self) -> None:
        """Reject a declaration that would quietly build the wrong matrix.

        Only mistakes nothing else catches: everything scikit-learn rejects at
        fit is left to it, and whether the exposure is also a predictor is the
        caller's modeling choice, not this class's.
        """
        if not self.plans and self.remainder == "drop":
            # A legal but empty matrix, shape (n, 0), which nobody complains about.
            raise ValueError(
                "A transformer that drops unplanned columns needs at least one "
                "plan; otherwise the design matrix has no columns"
            )

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureTransformer:
        """Learn the transforms from ``X`` and record the resulting columns."""
        self.validate()
        self.pipeline_ = self._build().set_output(transform="pandas")
        self.pipeline_.fit(X)
        # Taken from the pipeline rather than assembled here: one derivation of
        # the column names, sklearn's, instead of two that could drift.
        self.feature_names_ = tuple(self.pipeline_.get_feature_names_out())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transforms, returning a named, index-aligned frame."""
        check_is_fitted(self)
        transformed = self.pipeline_.transform(X)
        _check_finite(transformed)
        return transformed

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """The design matrix's columns, in order."""
        check_is_fitted(self)
        return np.asarray(self.feature_names_, dtype=object)

    def log_exposure(self, X: pd.DataFrame) -> pd.Series | None:
        """The offset column, or ``None`` when no exposure is declared.

        Returned separately from :meth:`transform` so it cannot be mistaken for
        a predictor: callers hand it to the model's ``offset``.
        """
        if self.exposure_column is None:
            return None
        values = X[self.exposure_column].astype(float)
        if (values <= 0).any():
            # The offset is a log, and log of a non-positive number is -inf.
            raise ValueError(
                f"Exposure column {self.exposure_column!r} must be strictly "
                f"positive; it enters the model as its log"
            )
        return pd.Series(
            np.log(values.to_numpy()),
            index=X.index,
            name=f"log_{self.exposure_column}",
        )

    def _build(self) -> Pipeline:
        """Return the unfitted pipeline the plans and interactions describe."""
        transformers: list[_TransformerTuple] = []
        for plan in self.plans:
            transformers.append((plan.name, plan.build(), list(plan.columns)))
        column_transformer = ColumnTransformer(
            transformers,
            remainder=self.remainder,
            # Every transform that changes a column's meaning renames it, so a
            # prefix would only add noise. A real collision raises.
            verbose_feature_names_out=False,
        )
        steps = [("column_transformer", column_transformer)]
        if self.interactions:
            interaction_transformer = self._build_interactions()
            steps.append(("interaction_transformer", interaction_transformer))
        return Pipeline(steps)

    def _build_interactions(self) -> ColumnTransformer:
        """Return a ``ColumnTransformer`` with one entry per product, plus the base."""
        # The base entry is what keeps the main effects: a ColumnTransformer
        # drops whatever no entry claimed, and an operand is claimed by its
        # product. A column may appear in several entries.
        transformers: list[_TransformerTuple] = [("base", "passthrough", _all_columns)]
        for interaction in self.interactions:
            transformers.append(
                (interaction.name, interaction.build(), list(interaction.columns))
            )
        return ColumnTransformer(
            transformers,
            remainder="drop",
            # Entries emit in order, so the products follow the base columns. A
            # product shadowing a column, or declared twice, raises here.
            verbose_feature_names_out=False,
        )


def _product_name(columns: Sequence[str]) -> str:
    """Name a product for the columns it multiplies."""
    return "_x_".join(columns)


def _all_columns(df: pd.DataFrame) -> list[str]:
    """Select every column, so the base matrix survives the interaction step."""
    return list(df.columns)


def _check_finite(df: pd.DataFrame) -> None:
    """Reject a design matrix holding a non-finite value.

    numpy raises nothing for ``log(0)``; it returns ``-inf`` and poisons the
    matrix silently. Non-numeric columns are passed-through categoricals.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)
    offending = [name for name, ok in zip(numeric.columns, finite) if not ok]
    if offending:
        raise ValueError(
            f"Transformed columns hold non-finite values: {offending}. A log of "
            f"a non-positive value is the usual cause"
        )
