"""Fold partition validation and row-alignment guards."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..data_splitting import SplitManifest, ValidationFold
from ..modeling_config import DEFAULT_MODELING_SCHEMA, ModelingSchema
from ..results import PredictionResult
from .contracts import FoldIdentity


def validate_experiment_partitions(
    outer_train_df: pd.DataFrame,
    *,
    split_manifest: SplitManifest,
    validation_folds: Sequence[ValidationFold],
    schema: ModelingSchema = DEFAULT_MODELING_SCHEMA,
) -> tuple[FoldIdentity, ...]:
    """Validate that fixed folds contain only canonical outer-training rows."""
    schema.validate_table(outer_train_df)
    if not validation_folds:
        raise ValueError("At least one validation fold is required")

    id_column = schema.building_id_column
    outer_ids = set(outer_train_df[id_column])
    manifest_train_ids = set(split_manifest.training_building_ids)
    holdout_ids = set(split_manifest.holdout_building_ids)
    if outer_ids != manifest_train_ids:
        raise ValueError("Outer training IDs must match the split manifest exactly")
    if outer_ids & holdout_ids:
        raise ValueError("Outer training data contains split-manifest holdout IDs")

    canonical = outer_train_df.set_index(id_column, drop=False)
    identities: list[FoldIdentity] = []
    seen_fold_indices: set[int] = set()
    for fold in validation_folds:
        if fold.fold_index in seen_fold_indices:
            raise ValueError("Validation fold indices must be unique")
        seen_fold_indices.add(fold.fold_index)

        fit_ids = set(fold.fit_df[id_column])
        validation_ids = set(fold.validation_df[id_column])
        if fit_ids & validation_ids:
            raise ValueError("Validation fold fit and validation IDs must be disjoint")
        if fit_ids | validation_ids != outer_ids:
            raise ValueError(
                "Every validation fold must partition the outer training IDs exactly"
            )
        if (fit_ids | validation_ids) & holdout_ids:
            raise ValueError("Validation folds contain split-manifest holdout IDs")

        _assert_canonical_rows(fold.fit_df, canonical, id_column=id_column)
        _assert_canonical_rows(fold.validation_df, canonical, id_column=id_column)
        ordered_fit_ids = _sorted_ids(fit_ids)
        ordered_validation_ids = _sorted_ids(validation_ids)
        fingerprint = hashlib.sha256(
            repr(
                (fold.fold_index, ordered_fit_ids, ordered_validation_ids)
            ).encode("utf-8")
        ).hexdigest()
        identities.append(
            FoldIdentity(
                fold_index=fold.fold_index,
                fit_building_ids=ordered_fit_ids,
                validation_building_ids=ordered_validation_ids,
                fingerprint=fingerprint,
            )
        )

    return tuple(sorted(identities, key=lambda identity: identity.fold_index))


def _assert_prediction_alignment(
    prediction: PredictionResult,
    validation_df: pd.DataFrame,
    *,
    id_column: str,
) -> None:
    expected = validation_df[id_column].to_numpy()
    if not np.array_equal(prediction.building_ids, expected):
        raise ValueError("Prediction building IDs must preserve validation-row order")


def _fold_coverage(
    outer_training_ids: Sequence[object],
    fold_identities: Sequence[FoldIdentity],
) -> pd.DataFrame:
    """Report how often each outer-training building is validated.

    ``make_validation_folds`` draws repeated overlapping splits rather than a
    disjoint K-fold partition, and singleton neighborhoods never enter a
    validation set. Fold means are therefore over correlated, unbalanced
    samples, and some buildings contribute no out-of-fold evidence at all.
    """
    counts: dict[object, int] = {building_id: 0 for building_id in outer_training_ids}
    for identity in fold_identities:
        for building_id in identity.validation_building_ids:
            counts[building_id] = counts.get(building_id, 0) + 1
    return pd.DataFrame(
        [
            {"building_id": building_id, "validation_fold_count": count}
            for building_id, count in counts.items()
        ]
    ).sort_values("building_id", key=lambda s: s.map(repr), ignore_index=True)


def _assert_canonical_rows(
    fold_df: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    id_column: str,
) -> None:
    if not fold_df[id_column].is_unique:
        raise ValueError(f"{id_column} must be unique within every validation fold")
    if tuple(fold_df.columns) != tuple(canonical.columns):
        raise ValueError("Validation fold columns must match outer training columns")
    fold_ids = fold_df[id_column].to_numpy()
    expected = canonical.loc[fold_ids].reset_index(drop=True)
    actual = fold_df.reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_like=False)
    except AssertionError as error:
        raise ValueError(
            "Validation fold rows must match canonical outer training rows"
        ) from error


def _sorted_ids(values: set[object]) -> tuple[object, ...]:
    return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))
