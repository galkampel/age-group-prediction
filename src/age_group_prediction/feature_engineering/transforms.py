"""The vocabulary of feature transformations.

Each is its own model carrying exactly the parameters it needs, so a missing or
nonsensical parameter is a construction-time error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils import Tags
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_is_fitted,
    validate_data,
)

# Declared here rather than imported, so the package depends on nothing in the
# project and can be used against any frame.
UnknownCategoryPolicy = Literal["error", "treat_as_reference"]

# Our policy names are ours; sklearn's are sklearn's. Map explicitly so a
# rename on either side fails loudly instead of silently changing behavior.
_SKLEARN_HANDLE_UNKNOWN: dict[UnknownCategoryPolicy, str] = {
    "error": "error",
    "treat_as_reference": "ignore",
}


class _TransformBase(BaseModel):
    """One transformation.

    Frozen so a spec can be shared across folds without aliasing;
    ``extra="forbid"`` turns a typo such as ``scalar=2`` into an error.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The sign of the output. "always": negative somewhere (or all zero) whatever
    # the data, as for a mean-zero column. "as_input": negative only where the
    # input is. "depends_on_data": either, so only the data can tell. "never".
    # ColumnPlan reads it to reject a log that cannot succeed.
    negative_output: ClassVar[
        Literal["always", "as_input", "depends_on_data", "never"]
    ] = "as_input"
    # Undefined on (some) negative input. Conservative: a declaration knows only
    # the sign, not how far below zero a value goes (log1p is fine down to -1).
    needs_nonnegative_input: ClassVar[bool] = False

    def build(self) -> TransformerMixin:
        """Return an unfitted scikit-learn transformer for this entry."""
        raise NotImplementedError


class Standardize(_TransformBase):
    """Subtract the fold mean and divide by the fold standard deviation."""

    kind: Literal["standardize"] = "standardize"
    negative_output = "always"

    def build(self) -> TransformerMixin:
        return StandardScaler(with_mean=True, with_std=True)


class Center(_TransformBase):
    """Subtract the fold mean, leaving the original units intact."""

    kind: Literal["center"] = "center"
    negative_output = "always"

    def build(self) -> TransformerMixin:
        return StandardScaler(with_mean=True, with_std=False)


class Quadratic(_TransformBase):
    """Square the column, renaming it so it cannot collide with its own input.

    Meant to follow ``Center``, ``Standardize`` or ``CenterByReferencePoint``:
    squaring a column whose centre is far from zero, relative to its spread,
    makes the quadratic term nearly collinear with the linear one. A reference
    point avoids that only when it lies near the centre of the data.
    """

    kind: Literal["quadratic"] = "quadratic"
    negative_output = "never"

    def build(self) -> TransformerMixin:
        # sqrt inverts squaring only on non-negative input; the sign is lost.
        return FunctionTransformer(
            lambda x: x**2,
            inverse_func=np.sqrt,
            check_inverse=False,
            validate=False,
            feature_names_out=_squared_names,
        )


class Log(_TransformBase):
    """Natural log. Needs strictly positive input.

    A log after centering, standardizing or relative saturation is rejected
    when the plan is declared: a column measured from its mean is negative
    somewhere. Any other non-positive value surfaces as a non-finite
    design-matrix column.
    """

    kind: Literal["log"] = "log"
    negative_output = "depends_on_data"
    needs_nonnegative_input = True

    def build(self) -> TransformerMixin:
        return FunctionTransformer(
            np.log,
            inverse_func=np.exp,
            validate=False,
            feature_names_out="one-to-one",
        )


class Log1p(_TransformBase):
    """``log(1 + x)``, defined at zero and so usable on counts.

    Needs input greater than -1. It is still rejected after centering,
    standardizing or relative saturation: the rule is on sign, and a log of a
    column measured from its mean is a mistake even where it stays above -1. Any other value at or below -1
    surfaces as a non-finite column.
    """

    kind: Literal["log1p"] = "log1p"
    needs_nonnegative_input = True

    def build(self) -> TransformerMixin:
        return FunctionTransformer(
            np.log1p,
            inverse_func=np.expm1,
            validate=False,
            feature_names_out="one-to-one",
        )


class DomainScale(_TransformBase):
    """Divide by a fixed constant chosen from domain knowledge.

    Learns nothing from the data, so the coefficient reads per declared unit
    and means the same thing in every fold — unlike a standardized one.
    """

    kind: Literal["domain_scale"] = "domain_scale"
    scale: float = Field(gt=0.0)

    def build(self) -> TransformerMixin:
        scale = self.scale
        return FunctionTransformer(
            lambda x: x / scale,
            inverse_func=lambda x: x * scale,
            validate=False,
            feature_names_out="one-to-one",
        )


class DomainMinMax(_TransformBase):
    """Rescale by a declared range, mapping ``minimum`` to 0 and ``maximum`` to 1.

    Nothing is clipped: an out-of-range value lands outside [0, 1] rather than
    saturating. Declared bounds, not observed ones, keep the scale fixed across
    folds.
    """

    kind: Literal["domain_min_max"] = "domain_min_max"
    negative_output = "depends_on_data"
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def _check_bounds(self) -> DomainMinMax:
        if self.maximum <= self.minimum:
            raise ValueError(
                f"domain_min_max needs maximum > minimum; got "
                f"minimum={self.minimum}, maximum={self.maximum}"
            )
        return self

    def build(self) -> TransformerMixin:
        minimum = self.minimum
        span = self.maximum - self.minimum
        return FunctionTransformer(
            lambda x: (x - minimum) / span,
            inverse_func=lambda x: x * span + minimum,
            validate=False,
            feature_names_out="one-to-one",
        )


class CenterByReferencePoint(_TransformBase):
    """Subtract a fixed reference point chosen from domain knowledge.

    Unlike ``Center``, it learns nothing from the data, so the output is the
    deviation from the declared reference and means the same in every fold.
    Its sign depends on where the data lie relative to the reference.
    """

    kind: Literal["center_by_reference_point"] = "center_by_reference_point"
    negative_output = "depends_on_data"
    # A non-finite reference makes every output non-finite, whatever the data.
    reference_point: float = Field(allow_inf_nan=False)

    def build(self) -> TransformerMixin:
        reference_point = self.reference_point
        return FunctionTransformer(
            lambda x: x - reference_point,
            inverse_func=lambda x: x + reference_point,
            validate=False,
            feature_names_out="one-to-one",
        )


class RelativeSaturation(_TransformBase):
    """Diminishing returns on a count, measured against the fold's typical value.

    Emits ``log1p(x) - log1p(mean(x))``. The reference is ``log1p`` of the
    mean, not the mean of ``log1p``: it keeps the reference expressed in the
    original counts, so the output is not mean-zero, but it is negative wherever
    ``x`` lies below the mean, which some value always does. Needs input greater
    than -1, checked at fit.
    """

    kind: Literal["relative_saturation"] = "relative_saturation"
    negative_output = "always"
    needs_nonnegative_input = True

    def build(self) -> TransformerMixin:
        return Log1pRatioScaler()


class OneHot(_TransformBase):
    """One-hot encode one categorical column against a declared reference level.

    The reference is dropped, so the remaining coefficients read as contrasts
    against it. Declaring ``categories`` rather than learning them keeps the
    columns identical across folds when a rare level is missing from one.
    """

    kind: Literal["ohe"] = "ohe"
    negative_output = "never"
    categories: tuple[str, ...]
    reference_category: str
    unknown_policy: UnknownCategoryPolicy = "error"

    @model_validator(mode="after")
    def _check_reference(self) -> OneHot:
        if not self.categories:
            raise ValueError("ohe needs at least one category")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError(f"ohe categories must be unique; got {self.categories}")
        if self.reference_category not in self.categories:
            raise ValueError(
                f"ohe reference_category {self.reference_category!r} must be one "
                f"of the declared categories {self.categories}"
            )
        return self

    @property
    def ordered_categories(self) -> tuple[str, ...]:
        """The declared levels with the reference first, for ``drop='first'``."""
        return (
            self.reference_category,
            *(
                category
                for category in self.categories
                if category != self.reference_category
            ),
        )

    def build(self) -> TransformerMixin:
        return OneHotEncoder(
            categories=[list(self.ordered_categories)],
            drop="first",
            sparse_output=False,
            handle_unknown=_SKLEARN_HANDLE_UNKNOWN[self.unknown_policy],
        )


# ``kind`` is the discriminator: it makes a bad payload report the one member it
# was meant to be, and lets a dumped spec reload as the same subclass.
Transform = Annotated[
    Standardize
    | Center
    | Quadratic
    | Log
    | Log1p
    | DomainScale
    | DomainMinMax
    | CenterByReferencePoint
    | RelativeSaturation
    | OneHot,
    Field(discriminator="kind"),
]


class Log1pRatioScaler(TransformerMixin, BaseEstimator):
    """Diminishing returns measured against the fit data's mean, per column.

    Emits ``log1p(x) - log1p(mean(x))``, where the mean is learned at fit. The
    fitted estimator behind :class:`RelativeSaturation`, and usable on its own.

    Input must be greater than -1, where ``log1p`` is defined; checked at fit.
    ``transform`` does not re-check: a value at or below -1 there gives ``nan``.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The per-column mean of the fit data.
    n_features_in_ : int
        The number of columns seen at fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        The column names seen at fit, when the input had string column names.
    """

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        # sklearn has no tag for "greater than -1"; positive_only is the nearest,
        # and data respecting it is always valid here. Only sklearn's estimator
        # checks read it (no meta-estimator does): they then feed non-negative
        # data. The one expecting sklearn's own "Negative values" error is an
        # expected failure in the tests.
        tags.input_tags.positive_only = True
        return tags

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> Log1pRatioScaler:
        # Records feature_names_in_, which lets transform reject a later fold
        # whose columns moved.
        values = validate_data(self, X, dtype=np.float64, ensure_2d=True)
        if np.any(values <= -1.0):
            raise ValueError(
                "Log1pRatioScaler needs values greater than -1; log1p is "
                "undefined at or below it"
            )
        self.mean_ = values.mean(axis=0)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        values = validate_data(self, X, dtype=np.float64, reset=False)
        return np.log1p(values) - np.log1p(self.mean_)

    def inverse_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        values = np.asarray(X, dtype=np.float64)
        return np.expm1(values + np.log1p(self.mean_))

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self)
        names = _check_feature_names_in(self, input_features)
        return np.asarray(_saturation_names(names), dtype=object)


def _squared_names(_transformer: object, input_features: Sequence[str]) -> list[str]:
    """Name a squared column for its input, so the pair stays distinguishable."""
    return [f"{name}_squared" for name in input_features]


def _saturation_names(input_features: Sequence[str]) -> list[str]:
    """Name a saturated column for its input; shared by the spec and the scaler."""
    return [f"{name}_sat" for name in input_features]
