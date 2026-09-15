"""Held-out fold losses for the independent model's total and probability components.

Each function fits one component on a selection fold's fit rows and scores the
fold's validation rows. The probability loss takes its seed from the model,
which derives and records it, so these functions stay free of model state.
"""

from __future__ import annotations

import numpy as np

from ..data_splitting import ValidationFold
from ..distributions import pointwise_log_probability
from ..feature_engineering import FittedFeatureTransformer
from ..modeling_config import (
    FeatureSpec,
    IndependentTotalProbabilityConfig,
    ModelingSchema,
)
from .count_regression import _fit_total_regression, _NB2Fit, _predict_total_mean
from .grouped_multinomial import _decision_logits, _fit_grouped_multinomial
from .probability_calibration import _weighted_composition_nll


def _total_fold_loss(
    spec: FeatureSpec,
    regularization: float,
    fold: ValidationFold,
    *,
    schema: ModelingSchema,
    config: IndependentTotalProbabilityConfig,
) -> float:
    """Fit the total-count regression on a fold and return its held-out mean NLL."""
    transformer = FittedFeatureTransformer(spec, schema=schema)
    fit_features = transformer.fit_transform(fold.fit_df)
    validation_features = transformer.transform(fold.validation_df)
    fit_offset = transformer.get_log_exposure(fold.fit_df)
    validation_offset = transformer.get_log_exposure(fold.validation_df)
    assert fit_offset is not None and validation_offset is not None
    fitted = _fit_total_regression(
        config.total_family,
        fit_features,
        fold.fit_df[schema.total_target_column].to_numpy(dtype=float),
        fit_offset,
        l2_penalty=regularization,
        config=config,
    )
    means = _predict_total_mean(
        fitted,
        validation_features,
        validation_offset,
        minimum_mean=config.minimum_mean,
    )
    observed = fold.validation_df[schema.total_target_column].to_numpy(dtype=float)
    return -float(
        np.mean(
            pointwise_log_probability(
                config.total_family,
                observed,
                means,
                dispersion=(fitted.dispersion if isinstance(fitted, _NB2Fit) else None),
            )
        )
    )


def _probability_fold_loss(
    spec: FeatureSpec,
    regularization: float,
    fold: ValidationFold,
    *,
    schema: ModelingSchema,
    config: IndependentTotalProbabilityConfig,
    seed: int,
) -> float:
    """Fit the grouped multinomial on a fold and return its held-out weighted NLL."""
    transformer = FittedFeatureTransformer(spec, schema=schema)
    fit_features = transformer.fit_transform(fold.fit_df)
    validation_features = transformer.transform(fold.validation_df)
    counts = fold.fit_df.loc[:, schema.cohort_target_columns].to_numpy(dtype=float)
    estimator = _fit_grouped_multinomial(
        fit_features,
        counts,
        c_value=regularization,
        config=config,
        seed=seed,
    )
    validation_counts = fold.validation_df.loc[
        :, schema.cohort_target_columns
    ].to_numpy(dtype=float)
    return _weighted_composition_nll(
        _decision_logits(estimator, validation_features), validation_counts
    )


__all__ = ["_probability_fold_loss", "_total_fold_loss"]
