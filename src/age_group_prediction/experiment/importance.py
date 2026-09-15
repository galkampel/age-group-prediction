"""Validation-only repeated permutation importance over raw feature blocks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..data_splitting import ValidationFold
from ..modeling_config import ModelingSchema
from ..models.base import BaseAgeGroupModel
from .contracts import CandidateDefinition, FoldIdentity
from .evidence import FrozenApproachSelection
from .partitions import _assert_prediction_alignment
from .seeds import _stable_seed


def _validate_importance_specs(
    candidate: CandidateDefinition,
    *,
    schema: ModelingSchema,
) -> None:
    specs = {spec.component: spec for spec in candidate.component_feature_specs}
    forbidden = {
        *schema.identifier_columns,
        *schema.target_columns,
        *schema.forbidden_feature_columns,
    }
    for importance in candidate.importance_specs:
        if importance.component not in specs:
            raise ValueError(
                f"Importance component '{importance.component}' has no feature spec"
            )
        spec = specs[importance.component]
        allowed = {
            *spec.numeric_features,
            *spec.categorical_features,
            *(() if spec.exposure_column is None else (spec.exposure_column,)),
        }
        for block in importance.feature_blocks:
            columns = set(block.columns)
            if columns & forbidden:
                raise ValueError(f"Feature block '{block.name}' contains forbidden columns")
            unknown = columns - allowed
            if unknown:
                raise ValueError(
                    f"Feature block '{block.name}' is outside its component spec: "
                    f"{sorted(unknown)}"
                )


def _compute_selected_importance(
    *,
    candidates: Mapping[str, CandidateDefinition],
    selections: Sequence[FrozenApproachSelection],
    fold_identities: Sequence[FoldIdentity],
    folds_by_index: Mapping[int, ValidationFold],
    fitted_models: Mapping[tuple[str, int], BaseAgeGroupModel],
    master_seed: int,
    schema: ModelingSchema,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    selected_ids = {selection.selected_candidate_id for selection in selections}
    for candidate_id in sorted(selected_ids):
        candidate = candidates[candidate_id]
        for identity in fold_identities:
            model = fitted_models[(candidate_id, identity.fold_index)]
            validation_df = folds_by_index[identity.fold_index].validation_df
            validation_ids = tuple(validation_df[schema.building_id_column])
            for importance in candidate.importance_specs:
                for block in importance.feature_blocks:
                    for repeat in range(importance.repeats):
                        purpose = (
                            f"importance/{candidate_id}/fold/{identity.fold_index}/"
                            f"{importance.component}/{block.name}/repeat/{repeat}"
                        )
                        seed = _stable_seed(master_seed, purpose)
                        baseline_prediction = model.predict(
                            validation_df,
                            prediction_config=candidate.prediction_config,
                            rng=np.random.default_rng(seed),
                        )
                        _assert_prediction_alignment(
                            baseline_prediction,
                            validation_df,
                            id_column=schema.building_id_column,
                        )
                        baseline = importance.metric.compute(
                            validation_df,
                            baseline_prediction,
                            rng=np.random.default_rng(seed),
                        )
                        permuted_df = validation_df.copy(deep=True)
                        row_order = np.random.default_rng(seed).permutation(
                            len(permuted_df)
                        )
                        permuted_df.loc[:, list(block.columns)] = (
                            validation_df.loc[:, list(block.columns)]
                            .iloc[row_order]
                            .to_numpy()
                        )
                        permuted_prediction = model.predict(
                            permuted_df,
                            prediction_config=candidate.prediction_config,
                            rng=np.random.default_rng(seed),
                        )
                        _assert_prediction_alignment(
                            permuted_prediction,
                            permuted_df,
                            id_column=schema.building_id_column,
                        )
                        permuted = importance.metric.compute(
                            validation_df,
                            permuted_prediction,
                            rng=np.random.default_rng(seed),
                        )
                        degradation = (
                            permuted.value - baseline.value
                            if importance.metric.optimization_direction == "minimize"
                            else baseline.value - permuted.value
                        )
                        records.append(
                            {
                                "candidate_id": candidate_id,
                                "approach": candidate.approach,
                                "fold_index": identity.fold_index,
                                "component": importance.component,
                                "metric_name": importance.metric.name,
                                "target": importance.metric.target,
                                "feature_block": block.name,
                                "columns": json.dumps(list(block.columns)),
                                "repeat": repeat,
                                "seed": seed,
                                "purpose": purpose,
                                "validation_building_ids": json.dumps(
                                    list(validation_ids)
                                ),
                                "baseline_value": baseline.value,
                                "permuted_value": permuted.value,
                                "degradation": float(degradation),
                                "optimization_direction": (
                                    importance.metric.optimization_direction
                                ),
                            }
                        )
    return pd.DataFrame(records)
