"""Fitted-fold state bundles and their reload checks, captured during a run.

Plan section 11 asks for fitted artifacts together with load/predict smoke-test
results. A fold's model exists only inside the runner's fold loop (models that
are not selected are released before importance), so the bundle is exported
and proven reloadable there, while the fitted model is still in hand. The
tracking adapter later writes this evidence; nothing here knows about MLflow.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..modeling_config import PredictionConfig
from ..models.base import BaseAgeGroupModel
from ..results import ParametricDistributionSpec, PredictionResult

# Seed for the two generators the reload check hands to the evaluated and the
# reloaded model. Each model gets its own generator seeded identically, so a
# model that consumes randomness for point predictions (the Bayesian model
# draws random effects for rows in neighborhoods it never saw) consumes the
# same stream in both. These generators are separate from the runner's, whose
# seeds are hash-derived per purpose, so the check cannot change any result.
_RELOAD_CHECK_SEED = 0

# Phase 1 proved reloads reproduce point predictions exactly for every model
# family, so any difference at all is a defect rather than numerical noise.
_RELOAD_TOLERANCE = 0.0


@dataclass(frozen=True)
class ReloadCheck:
    """Outcome of reloading one fold's state bundle and predicting with it.

    ``max_abs_differences`` holds, per compared prediction field, the largest
    absolute difference between the fitted and the reloaded model: the means,
    the age-group probabilities, and the parameters of every parametric
    distribution (NB2 dispersion, Normal scale), which point means alone would
    not reveal. A mismatched shape, family or target counts as infinite.
    """

    max_abs_differences: Mapping[str, float]
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        """Copy the differences into a read-only mapping."""
        object.__setattr__(
            self,
            "max_abs_differences",
            MappingProxyType(dict(self.max_abs_differences)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the check as JSON-safe values (an infinity is written as null)."""
        return {
            "max_abs_differences": {
                field: value if np.isfinite(value) else None
                for field, value in self.max_abs_differences.items()
            },
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class FoldArtifactEvidence:
    """One candidate's state bundle for one fold, with its reload check."""

    candidate_id: str
    fold_index: int
    state_bundle: Mapping[str, Any]
    reload_check: ReloadCheck

    def __post_init__(self) -> None:
        """Freeze a private, strictly JSON-safe copy of the bundle."""
        # Deep-copied so later edits to the exporter's dictionaries cannot
        # change recorded evidence, and checked strictly because the bundle is
        # written as JSON.
        bundle = copy.deepcopy(dict(self.state_bundle))
        json.dumps(bundle, allow_nan=False)
        object.__setattr__(self, "state_bundle", MappingProxyType(bundle))


def capture_fold_artifact(
    model: BaseAgeGroupModel,
    validation_df: pd.DataFrame,
    *,
    candidate_id: str,
    fold_index: int,
) -> FoldArtifactEvidence:
    """Export a fitted model's bundle and prove it reloads to the same predictions.

    The bundle is reloaded from its JSON text, which is the form that gets
    stored, and **without the training frame**, which proves the artifact is
    self-contained. Both models point-predict the fold's validation rows with
    ``PredictionConfig()``: a candidate's own config may request predictive
    draws, which the frequentist models can only produce with the training
    frame. Raises ``ValueError`` when the predictions differ, so the run aborts
    rather than store an artifact that does not reproduce its model.
    """
    bundle = model.to_state_bundle()
    reloaded = type(model).from_state_bundle(json.loads(json.dumps(bundle)))
    original, restored = (
        candidate.predict(
            validation_df,
            prediction_config=PredictionConfig(),
            rng=np.random.default_rng(_RELOAD_CHECK_SEED),
        )
        for candidate in (model, reloaded)
    )
    differences = _prediction_differences(original, restored)
    check = ReloadCheck(
        max_abs_differences=differences,
        tolerance=_RELOAD_TOLERANCE,
        passed=max(differences.values()) <= _RELOAD_TOLERANCE,
    )
    if not check.passed:
        raise ValueError(
            "Reloaded state bundle does not reproduce the fitted model's "
            f"predictions: {check.to_dict()}"
        )
    return FoldArtifactEvidence(
        candidate_id=candidate_id,
        fold_index=fold_index,
        state_bundle=bundle,
        reload_check=check,
    )


def _prediction_differences(
    original: PredictionResult, restored: PredictionResult
) -> dict[str, float]:
    """Largest absolute difference per compared prediction field."""
    return {
        field: _max_abs_difference(getattr(original, field), getattr(restored, field))
        for field in ("total_mean", "cohort_means", "age_group_probabilities")
    } | {
        "parametric_distributions": _parametric_difference(
            original.parametric_distributions, restored.parametric_distributions
        )
    }


def _parametric_difference(
    original: Mapping[str, ParametricDistributionSpec] | None,
    restored: Mapping[str, ParametricDistributionSpec] | None,
) -> float:
    """Largest parameter difference across targets; infinite on any mismatch."""
    original, restored = original or {}, restored or {}
    if set(original) != set(restored):
        return float("inf")
    worst = 0.0
    for target, spec in original.items():
        other = restored[target]
        if spec.family != other.family:
            return float("inf")
        for parameter in ("dispersion", "scale"):
            left, right = getattr(spec, parameter), getattr(other, parameter)
            if (left is None) != (right is None):
                return float("inf")
            if left is not None:
                worst = max(worst, _max_abs_difference(left, right))
    return worst


def _max_abs_difference(left: Any, right: Any) -> float:
    """Largest absolute difference; infinite when the shapes disagree."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right), initial=0.0))


__all__ = ["FoldArtifactEvidence", "ReloadCheck", "capture_fold_artifact"]
