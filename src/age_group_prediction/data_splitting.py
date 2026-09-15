"""Deterministic outer splits and train-only validation folds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .hashing import column_schema_hash, table_hash
from .modeling_config import (
    DEFAULT_FOLD_CONFIG,
    DEFAULT_OUTER_SPLIT_CONFIG,
    DEFAULT_SEED,
    FoldConfig,
    OuterSplitConfig,
)

# How a split's randomness was obtained. This is evidence about a completed
# split rather than a configuration choice, so it lives here beside the
# manifest that records it, not in `modeling_config`.
SeedSource = Literal["caller_generator", "project_default"]
_VALID_SEED_SOURCES = {"caller_generator", "project_default"}

__all__ = [
    "DEFAULT_FOLD_CONFIG",
    "DEFAULT_OUTER_SPLIT_CONFIG",
    "DEFAULT_SEED",
    "DataSplit",
    "FoldConfig",
    "FoldPlan",
    "OuterSplitConfig",
    "SeedSource",
    "SplitManifest",
    "ValidationFold",
    "load_split_manifest",
    "make_validation_folds",
    "persist_split_manifest",
    "replay_split_manifest",
    "split_known_neighborhood_buildings",
]


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng(DEFAULT_SEED)
    return rng


def _seed_source(rng: np.random.Generator | None) -> SeedSource:
    """Report whether randomness was chosen deliberately or fell back.

    Only the source is recorded, never a seed value: a caller-supplied
    generator may already have been advanced, so no integer would faithfully
    describe it.
    """
    return "project_default" if rng is None else "caller_generator"


@dataclass(frozen=True)
class SplitManifest:
    """Serializable evidence for the immutable outer test partition.

    Both holdout fractions are recorded because they differ: every
    non-singleton neighborhood must contribute at least one test building, so
    a population of small neighborhoods holds out more than was requested.
    ``requested_holdout_fraction`` is the setting; ``realized_holdout_fraction``
    is what the split actually did.
    """

    strategy: str
    requested_holdout_fraction: float
    realized_holdout_fraction: float
    training_building_ids: tuple[object, ...]
    holdout_building_ids: tuple[object, ...]
    source_table_hash: str
    column_schema_hash: str
    n_rows: int
    n_neighborhoods: int
    seed_source: SeedSource
    deployment_claim: str = "new building in a known neighborhood"

    def __post_init__(self) -> None:
        if self.seed_source not in _VALID_SEED_SOURCES:
            raise ValueError(
                f"seed_source must be one of {sorted(_VALID_SEED_SOURCES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitManifest:
        """Rebuild a manifest from ``to_dict`` output or its JSON round trip."""
        expected = set(cls.__dataclass_fields__)
        unknown = set(payload).difference(expected)
        missing = expected.difference(payload).difference(
            {"deployment_claim"}
        )
        if unknown or missing:
            raise ValueError(
                "Split manifest payload keys do not match the manifest schema: "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        # JSON has no tuple type; restore it so round-tripped manifests
        # compare equal to the manifest they were serialized from.
        return cls(
            **{
                **payload,
                "training_building_ids": tuple(payload["training_building_ids"]),
                "holdout_building_ids": tuple(payload["holdout_building_ids"]),
            }
        )


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load a persisted manifest written by ``persist_split_manifest``."""
    payload = json.loads(Path(path).read_text())
    return SplitManifest.from_dict(payload)


def persist_split_manifest(path: str | Path, manifest: SplitManifest) -> SplitManifest:
    """Write a manifest to ``path``, or verify an exact match if one exists.

    The lockbox manifest is written once and read many times afterward: a
    second write from a fresh split must never silently replace it, since a
    replaced manifest would relabel which buildings are held out. Writing
    only when the file is absent, and otherwise requiring the incoming
    manifest to equal what is already there, gives that guarantee without a
    separate "did I already run this" flag to keep in sync.
    """
    resolved = Path(path)
    if resolved.exists():
        existing = load_split_manifest(resolved)
        if existing != manifest:
            raise ValueError(
                f"A different split manifest is already persisted at {resolved}; "
                "refusing to overwrite the lockbox partition"
            )
        return existing
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return manifest


@dataclass(frozen=True)
class DataSplit:
    """Outer split data and its replayable manifest."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    manifest: SplitManifest


@dataclass(frozen=True)
class ValidationFold:
    """One deterministic train-only fit/validation fold."""

    fold_index: int
    fit_df: pd.DataFrame
    validation_df: pd.DataFrame


@dataclass(frozen=True)
class FoldPlan:
    """Validation folds plus the training buildings no fold can validate.

    A building alone in its neighborhood cannot enter a validation set without
    leaving that neighborhood unrepresented in the fit partition, which is
    outside the supported "new building in a known neighborhood" claim. Those
    buildings still train; they are named here so their exclusion is recorded
    evidence rather than an emergent property of the fold draw.
    """

    folds: tuple[ValidationFold, ...]
    unvalidated_building_ids: tuple[object, ...]


def _validate_split_input(
    df: pd.DataFrame,
    *,
    building_id_column: str,
    neighborhood_id_column: str,
) -> None:
    missing_columns = {building_id_column, neighborhood_id_column}.difference(
        df.columns
    )
    if missing_columns:
        raise ValueError(f"Missing split columns: {sorted(missing_columns)}")
    if not df[building_id_column].is_unique:
        raise ValueError(f"{building_id_column} must be unique before splitting")


def _partition_outer_holdout(
    df: pd.DataFrame,
    *,
    config: OuterSplitConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out buildings within each neighborhood, keeping singletons in train."""
    holdout_building_ids: list[object] = []
    for _, neighborhood_df in df.groupby(config.neighborhood_id_column, sort=True):
        building_ids = np.sort(neighborhood_df[config.building_id_column].to_numpy())
        building_count = len(building_ids)
        if building_count == 1:
            continue
        # At least one building held out and at least one retained, so every
        # non-singleton neighborhood is represented on both sides.
        holdout_count = max(
            1,
            min(round(building_count * config.test_fraction), building_count - 1),
        )
        holdout_building_ids.extend(
            rng.permutation(building_ids)[:holdout_count].tolist()
        )

    holdout_mask = df[config.building_id_column].isin(holdout_building_ids)
    return df.loc[~holdout_mask].copy(), df.loc[holdout_mask].copy()


def _make_outer_manifest(
    modeling_table: pd.DataFrame,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: OuterSplitConfig,
    seed_source: SeedSource,
) -> SplitManifest:
    return SplitManifest(
        strategy=config.strategy_version,
        requested_holdout_fraction=config.test_fraction,
        realized_holdout_fraction=len(test_df) / len(modeling_table),
        training_building_ids=tuple(
            np.sort(train_df[config.building_id_column].to_numpy()).tolist()
        ),
        holdout_building_ids=tuple(
            np.sort(test_df[config.building_id_column].to_numpy()).tolist()
        ),
        source_table_hash=table_hash(
            modeling_table, id_column=config.building_id_column
        ),
        column_schema_hash=column_schema_hash(modeling_table),
        n_rows=len(modeling_table),
        n_neighborhoods=modeling_table[config.neighborhood_id_column].nunique(),
        seed_source=seed_source,
    )


def split_known_neighborhood_buildings(
    modeling_table: pd.DataFrame,
    *,
    config: OuterSplitConfig = DEFAULT_OUTER_SPLIT_CONFIG,
    rng: np.random.Generator | None = None,
) -> DataSplit:
    """Create the outer known-neighborhood split and its manifest."""
    _validate_split_input(
        modeling_table,
        building_id_column=config.building_id_column,
        neighborhood_id_column=config.neighborhood_id_column,
    )
    train_df, test_df = _partition_outer_holdout(
        modeling_table, config=config, rng=_resolve_rng(rng)
    )
    return DataSplit(
        train_df=train_df,
        test_df=test_df,
        manifest=_make_outer_manifest(
            modeling_table,
            train_df=train_df,
            test_df=test_df,
            config=config,
            seed_source=_seed_source(rng),
        ),
    )


def replay_split_manifest(
    modeling_table: pd.DataFrame,
    manifest: SplitManifest,
    *,
    config: OuterSplitConfig = DEFAULT_OUTER_SPLIT_CONFIG,
) -> DataSplit:
    """Reapply a manifest after verifying the table, schema, and provenance.

    ``seed_source`` is deliberately not checked. Every other recorded field
    describes the table and can be recomputed from it; how the original split
    was seeded is history, and replay reproduces assignments from the persisted
    IDs rather than by reseeding.
    """
    _validate_split_input(
        modeling_table,
        building_id_column=config.building_id_column,
        neighborhood_id_column=config.neighborhood_id_column,
    )
    if (
        table_hash(modeling_table, id_column=config.building_id_column)
        != manifest.source_table_hash
    ):
        raise ValueError("Modeling table does not match the split manifest")
    if column_schema_hash(modeling_table) != manifest.column_schema_hash:
        raise ValueError("Modeling schema does not match the split manifest")
    if manifest.strategy != config.strategy_version:
        raise ValueError(
            f"Split manifest strategy {manifest.strategy!r} does not match the "
            f"configured strategy {config.strategy_version!r}"
        )

    # The hashes prove the rows; these prove the manifest describes them.
    n_neighborhoods = modeling_table[config.neighborhood_id_column].nunique()
    if manifest.n_rows != len(modeling_table):
        raise ValueError(
            f"Split manifest claims {manifest.n_rows} rows but the table has "
            f"{len(modeling_table)}"
        )
    if manifest.n_neighborhoods != n_neighborhoods:
        raise ValueError(
            f"Split manifest claims {manifest.n_neighborhoods} neighborhoods but "
            f"the table has {n_neighborhoods}"
        )

    all_ids = set(modeling_table[config.building_id_column])
    training_ids = set(manifest.training_building_ids)
    holdout_ids = set(manifest.holdout_building_ids)
    if training_ids & holdout_ids or training_ids | holdout_ids != all_ids:
        raise ValueError("Split manifest IDs do not form an exact partition")

    realized = len(holdout_ids) / len(modeling_table)
    if not np.isclose(realized, manifest.realized_holdout_fraction):
        raise ValueError(
            f"Split manifest records a realized holdout fraction of "
            f"{manifest.realized_holdout_fraction} but its IDs give {realized}"
        )

    holdout_mask = modeling_table[config.building_id_column].isin(holdout_ids)
    return DataSplit(
        train_df=modeling_table.loc[~holdout_mask].copy(),
        test_df=modeling_table.loc[holdout_mask].copy(),
        manifest=manifest,
    )


def _assign_fold_indices(
    train_df: pd.DataFrame,
    *,
    config: FoldConfig,
    rng: np.random.Generator,
) -> dict[object, int]:
    """Deal each neighborhood's buildings round-robin across the folds.

    Rotation rather than repeated independent draws, so every non-singleton
    building is validated exactly once and each neighborhood keeps at least
    one building in every fit partition.

    The starting fold carries over between neighborhoods instead of resetting
    to zero. Most neighborhoods here are smaller than ``n_folds``, so resetting
    would give every one of them a building in fold 0 and few a building in the
    last fold, leaving fold sizes badly unbalanced.
    """
    assignments: dict[object, int] = {}
    cursor = 0
    for _, neighborhood_df in train_df.groupby(
        config.neighborhood_id_column, sort=True
    ):
        building_ids = np.sort(neighborhood_df[config.building_id_column].to_numpy())
        if len(building_ids) == 1:
            continue
        for position, building_id in enumerate(rng.permutation(building_ids)):
            assignments[building_id] = (cursor + position) % config.n_folds
        cursor = (cursor + len(building_ids)) % config.n_folds
    return assignments


def make_validation_folds(
    train_df: pd.DataFrame,
    *,
    config: FoldConfig = DEFAULT_FOLD_CONFIG,
    rng: np.random.Generator | None = None,
) -> FoldPlan:
    """Create reproducible known-neighborhood folds from outer training data.

    Each non-singleton training building appears in exactly one validation
    set. Buildings alone in their neighborhood are reported in
    ``FoldPlan.unvalidated_building_ids`` rather than silently dropped.
    """
    _validate_split_input(
        train_df,
        building_id_column=config.building_id_column,
        neighborhood_id_column=config.neighborhood_id_column,
    )
    assignments = _assign_fold_indices(
        train_df, config=config, rng=_resolve_rng(rng)
    )
    fold_index_by_row = train_df[config.building_id_column].map(assignments)

    folds: list[ValidationFold] = []
    for fold_index in range(config.n_folds):
        validation_mask = fold_index_by_row == fold_index
        if not validation_mask.any():
            raise ValueError(
                f"Fold {fold_index} has no validation rows; reduce n_folds "
                f"({config.n_folds}) or supply more buildings per neighborhood"
            )
        folds.append(
            ValidationFold(
                fold_index=fold_index,
                fit_df=train_df.loc[~validation_mask].copy(),
                validation_df=train_df.loc[validation_mask].copy(),
            )
        )

    unvalidated = train_df.loc[
        ~train_df[config.building_id_column].isin(assignments),
        config.building_id_column,
    ]
    return FoldPlan(
        folds=tuple(folds),
        unvalidated_building_ids=tuple(np.sort(unvalidated.to_numpy()).tolist()),
    )
