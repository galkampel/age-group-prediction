"""Fold-fitted feature transformations for age-group models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from .modeling_config import (
    DEFAULT_MODELING_SCHEMA,
    CategoricalFeatureSpec,
    FeatureSpec,
    ModelingSchema,
    UnknownCategoryPolicy,
)
from .state_bundle import restore_feature_spec

# Our policy names are ours; sklearn's are sklearn's. Map explicitly so a
# rename on either side fails loudly instead of silently changing behavior.
_SKLEARN_HANDLE_UNKNOWN: dict[UnknownCategoryPolicy, str] = {
    "error": "error",
    "treat_as_reference": "ignore",
}

# Cubic SES splines. `to_state` strips this many padding knots from each end of
# the fitted knot vector to recover the base knots `from_state` rebuilds from.
_SPLINE_DEGREE = 3


class FittedFeatureTransformer:
    """Fit preprocessing state on one fold and transform later partitions."""

    def __init__(
        self,
        feature_spec: FeatureSpec,
        *,
        schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
    ) -> None:
        feature_spec.validate_for_schema(schema)
        self.feature_spec = feature_spec
        self.schema = schema
        self._numeric_means: pd.Series | None = None
        self._numeric_scales: pd.Series | None = None
        self._spline: SplineTransformer | None = None
        self._categorical_encoder: OneHotEncoder | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._fit_row_count: int | None = None
        self._fit_transformed: pd.DataFrame | None = None

    @property
    def is_fitted(self) -> bool:
        return self._feature_names is not None

    def fit(self, fit_df: pd.DataFrame) -> FittedFeatureTransformer:
        """Learn scaling, spline knots, and output columns from one fit fold."""
        self._validate_input(fit_df)
        numeric_df = fit_df.loc[:, list(self.feature_spec.numeric_features)].astype(
            float
        )
        if self.feature_spec.scale_numeric:
            self._numeric_means = numeric_df.mean()
            scales = numeric_df.std(ddof=0)
            # A constant column would divide by zero; leave it unscaled.
            self._numeric_scales = scales.mask(scales == 0, 1.0)

        if self.feature_spec.ses_form == "spline":
            self._spline = SplineTransformer(
                n_knots=self.feature_spec.spline_n_knots,
                degree=_SPLINE_DEGREE,
                include_bias=False,
            ).fit(numeric_df[[self.feature_spec.ses_column]])

        category_columns = list(self.feature_spec.categorical_features)
        if category_columns:
            self._categorical_encoder = self._make_categorical_encoder().fit(
                fit_df.loc[:, category_columns]
            )

        transformed = self._transform_fitted(fit_df)
        self._feature_names = tuple(transformed.columns)
        self._fit_row_count = len(fit_df)
        self._fit_transformed = transformed
        return self

    def transform(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Transform a partition using state learned by ``fit``."""
        if not self.is_fitted:
            raise RuntimeError("Feature transformer must be fitted before transform")
        self._validate_input(eval_df)
        transformed = self._transform_fitted(eval_df)
        if tuple(transformed.columns) != self._feature_names:
            raise RuntimeError("Transformed feature columns changed after fitting")
        return transformed

    def fit_transform(self, fit_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on and transform the same partition.

        Returns the frame ``fit`` already built rather than transforming the
        same rows a second time; this runs once per fold per tuning trial.
        """
        self.fit(fit_df)
        assert self._fit_transformed is not None
        return self._fit_transformed.copy()

    def get_feature_names_out(self) -> tuple[str, ...]:
        """Return deterministic transformed feature names."""
        if self._feature_names is None:
            raise RuntimeError("Feature transformer must be fitted first")
        return self._feature_names

    def get_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable feature and fitted-state metadata."""
        if not self.is_fitted:
            raise RuntimeError("Feature transformer must be fitted first")
        return {
            "feature_spec": asdict(self.feature_spec),
            "feature_names": self._feature_names,
            "fit_row_count": self._fit_row_count,
            # None rather than unused numbers when scaling is not applied.
            "scale_numeric": self.feature_spec.scale_numeric,
            "numeric_means": (
                self._numeric_means.to_dict()
                if self._numeric_means is not None
                else None
            ),
            "numeric_scales": (
                self._numeric_scales.to_dict()
                if self._numeric_scales is not None
                else None
            ),
            "spline_knots": (
                self._spline.bsplines_[0].t.tolist()
                if self._spline is not None
                else None
            ),
            "category_references": {
                spec.column: spec.reference_category
                for spec in self.schema.categorical_feature_specs
                if spec.column in self.feature_spec.categorical_features
            },
        }

    def to_state(self) -> dict[str, object]:
        """Return everything ``transform`` reads, as JSON-safe primitives.

        ``from_state`` rebuilds an equivalent transformer from this without the
        fit rows, which is what lets a model state bundle omit training data.
        Scaling is stored as per-column means and scales and the SES spline as
        its base knots. One-hot categories are not stored: they come from the
        schema, not from the data.
        """
        if not self.is_fitted:
            raise RuntimeError("Feature transformer must be fitted first")
        assert self._feature_names is not None
        return {
            "feature_spec": asdict(self.feature_spec),
            "feature_names": list(self._feature_names),
            "fit_row_count": self._fit_row_count,
            "numeric_means": _series_to_dict(self._numeric_means),
            "numeric_scales": _series_to_dict(self._numeric_scales),
            "spline_base_knots": (
                self._spline.bsplines_[0].t[_SPLINE_DEGREE:-_SPLINE_DEGREE].tolist()
                if self._spline is not None
                else None
            ),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
    ) -> FittedFeatureTransformer:
        """Rebuild a fitted transformer from ``to_state`` output without refitting."""
        transformer = cls(restore_feature_spec(state["feature_spec"]), schema=schema)
        spec = transformer.feature_spec
        if spec.scale_numeric != (state["numeric_means"] is not None):
            raise ValueError("Transformer state disagrees with its spec about scaling")
        if (spec.ses_form == "spline") != (state["spline_base_knots"] is not None):
            raise ValueError("Transformer state disagrees with its spec about the spline")

        if spec.scale_numeric:
            transformer._numeric_means = _series_from_state(
                state["numeric_means"], spec.numeric_features, label="numeric_means"
            )
            transformer._numeric_scales = _series_from_state(
                state["numeric_scales"], spec.numeric_features, label="numeric_scales"
            )
        if spec.ses_form == "spline":
            knots = np.asarray(state["spline_base_knots"], dtype=float)
            # With explicit knots, `fit` learns nothing from its rows beyond the
            # input column name, so fitting on the knots themselves is exact.
            transformer._spline = SplineTransformer(
                knots=knots.reshape(-1, 1),
                degree=_SPLINE_DEGREE,
                include_bias=False,
            ).fit(pd.DataFrame({spec.ses_column: knots}))
        if spec.categorical_features:
            transformer._categorical_encoder = (
                transformer._make_categorical_encoder().fit(
                    transformer._schema_category_frame()
                )
            )
        transformer._fit_row_count = int(state["fit_row_count"])
        # Assigned last: `is_fitted` keys off this attribute.
        transformer._feature_names = tuple(state["feature_names"])
        return transformer

    def get_log_exposure(self, df: pd.DataFrame) -> pd.Series | None:
        """Return a Poisson/NB2 offset from strictly positive raw exposure values.

        This uses ``log(exposure)`` (not ``log1p``) because the exposure is
        required to be positive and models consume it as a multiplicative
        offset. The input frame is read but never mutated.
        """
        exposure = self.feature_spec.exposure_column
        if exposure is None:
            return None
        if exposure not in df:
            raise ValueError(f"Missing exposure column: {exposure}")
        values = cast(pd.Series, df[exposure]).astype(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{exposure} must be finite and positive")
        return pd.Series(
            np.log(values.to_numpy()),
            index=values.index,
            name=f"log_{exposure}",
        )

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = {
            *self.feature_spec.numeric_features,
            *self.feature_spec.categorical_features,
        }
        if self.feature_spec.exposure_column is not None:
            required.add(self.feature_spec.exposure_column)
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        numeric = df.loc[:, list(self.feature_spec.numeric_features)].to_numpy(
            dtype=float
        )
        if not np.isfinite(numeric).all():
            raise ValueError("Numeric feature columns must be finite")
        for category_spec in self.schema.categorical_feature_specs:
            if category_spec.column not in self.feature_spec.categorical_features:
                continue
            values = df[category_spec.column]
            if values.isna().any():
                raise ValueError(f"{category_spec.column} must not be missing")
            unknown = set(values).difference(category_spec.categories)
            if unknown and self.feature_spec.unknown_category_policy == "error":
                raise ValueError(
                    f"Unknown {category_spec.column} values: {sorted(unknown)}"
                )
        self._validate_exposure(df)

    def _validate_exposure(self, df: pd.DataFrame) -> None:
        """Reject an unusable exposure column before any transformation runs."""
        self.get_log_exposure(df)

    def _transform_fitted(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.loc[:, list(self.feature_spec.numeric_features)].astype(
            float
        )
        if self.feature_spec.scale_numeric:
            assert self._numeric_means is not None
            assert self._numeric_scales is not None
            numeric_df = (numeric_df - self._numeric_means) / self._numeric_scales

        output = numeric_df.copy()
        # Interaction operands track each variable's selected main-effect form,
        # so an interaction never multiplies a representation the matrix does
        # not also carry as a main effect.
        operands = {column: numeric_df[column] for column in numeric_df.columns}

        ses_column = self.feature_spec.ses_column
        if self.feature_spec.ses_form == "quadratic":
            output[f"{ses_column}_squared"] = numeric_df[ses_column] ** 2
        elif self.feature_spec.ses_form == "spline":
            assert self._spline is not None
            spline_values = self._spline.transform(df[[ses_column]])
            output = output.drop(columns=[ses_column])
            for index in range(spline_values.shape[1]):
                output[f"{ses_column}_spline_{index}"] = spline_values[:, index]
            # FeatureSpec forbids pairing spline SES with its linear interaction.
            operands.pop(ses_column)

        daycare_column = self.feature_spec.daycare_column
        if self.feature_spec.daycare_form == "log1p":
            raw_daycare = df[daycare_column].astype(float)
            if (raw_daycare < 0).any():
                raise ValueError("Daycare log1p transformation requires nonnegative values")
            output = output.drop(columns=[daycare_column])
            log_daycare = pd.Series(
                np.log1p(raw_daycare.to_numpy()),
                index=raw_daycare.index,
            )
            output[f"{daycare_column}_log1p"] = log_daycare
            operands[daycare_column] = log_daycare

        self._add_interactions(output, operands)
        self._add_categorical_features(output, df)
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("Transformed features must be finite")
        return output

    def _add_interactions(
        self, output: pd.DataFrame, operands: dict[str, pd.Series]
    ) -> None:
        """Multiply main-effect representations, never raw stand-ins.

        ``operands`` holds each variable exactly as the matrix carries it as a
        main effect, so hierarchy holds in the output and not only in the spec.
        """
        spec = self.feature_spec
        if "ses_x_household_size" in spec.interactions:
            output[f"{spec.ses_column}_x_{spec.household_size_column}"] = (
                operands[spec.ses_column] * operands[spec.household_size_column]
            )
        if "daycare_x_median_age" in spec.interactions:
            output[f"{spec.daycare_column}_x_{spec.median_age_column}"] = (
                operands[spec.daycare_column] * operands[spec.median_age_column]
            )
        if "room_share_x_household_size" in spec.interactions:
            for room_share in self.schema.room_share_features:
                output[f"{room_share}_x_{spec.household_size_column}"] = (
                    operands[room_share] * operands[spec.household_size_column]
                )
        if "room_share_x_median_age" in spec.interactions:
            for room_share in self.schema.room_share_features:
                output[f"{room_share}_x_{spec.median_age_column}"] = (
                    operands[room_share] * operands[spec.median_age_column]
                )

    def _make_categorical_encoder(self) -> OneHotEncoder:
        """Return an unfitted encoder whose categories come from the schema.

        The declared reference level is placed first so ``drop="first"``
        removes it. Categories are listed in the spec's column order, which is
        the order the encoder's input columns arrive in; listing them in schema
        order would pair each column with another column's levels whenever the
        two orders differ. Shared by ``fit`` and ``from_state`` so both paths
        build the identical encoder.
        """
        specs = {spec.column: spec for spec in self.schema.categorical_feature_specs}
        return OneHotEncoder(
            categories=[
                _ordered_categories(specs[column])
                for column in self.feature_spec.categorical_features
            ],
            drop="first",
            handle_unknown=_SKLEARN_HANDLE_UNKNOWN[
                self.feature_spec.unknown_category_policy
            ],
            sparse_output=False,
        )

    def _schema_category_frame(self) -> pd.DataFrame:
        """Return a frame containing every declared level of each categorical column.

        An encoder with explicit categories learns only the column names from
        the rows it is fitted on, so this stands in for the fit rows.
        """
        specs = {spec.column: spec for spec in self.schema.categorical_feature_specs}
        columns = self.feature_spec.categorical_features
        length = max(len(specs[column].categories) for column in columns)
        return pd.DataFrame(
            {
                column: np.resize(np.asarray(specs[column].categories, dtype=object), length)
                for column in columns
            }
        )

    def _add_categorical_features(
        self, output: pd.DataFrame, df: pd.DataFrame
    ) -> None:
        if self._categorical_encoder is None:
            return
        category_columns = list(self.feature_spec.categorical_features)
        encoded = cast(
            np.ndarray,
            self._categorical_encoder.transform(df.loc[:, category_columns]),
        )
        encoded_df = pd.DataFrame(
            encoded,
            index=df.index,
            columns=self._categorical_encoder.get_feature_names_out(category_columns),
        )
        for column in encoded_df:
            output[column] = encoded_df[column]


def _series_to_dict(series: pd.Series | None) -> dict[str, float] | None:
    """Return a fitted per-column statistic as a JSON-safe mapping."""
    if series is None:
        return None
    return {str(column): float(value) for column, value in series.items()}


def _series_from_state(
    values: Mapping[str, float], columns: tuple[str, ...], *, label: str
) -> pd.Series:
    """Rebuild a per-column statistic in the spec's column order.

    Order comes from the spec, never from the mapping: a JSON writer may sort
    or reorder keys, and `_transform_fitted` would otherwise reorder its output
    columns to match.
    """
    if set(values) != set(columns):
        raise ValueError(
            f"Transformer state {label} must cover exactly the spec's numeric "
            f"features; missing {sorted(set(columns) - set(values))}, unknown "
            f"{sorted(set(values) - set(columns))}"
        )
    return pd.Series([float(values[column]) for column in columns], index=list(columns))


def _ordered_categories(category_spec: CategoricalFeatureSpec) -> list[str]:
    """Place the declared reference first for OneHotEncoder(drop='first')."""
    return [
        category_spec.reference_category,
        *(category for category in category_spec.categories if category != category_spec.reference_category),
    ]