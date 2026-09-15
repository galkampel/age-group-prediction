"""Read tags and fields out of a result's provenance and model metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _data_tags(provenance: Mapping[str, Any]) -> dict[str, str]:
    """Tags that tie a run to its data, split and seed; shared by every run."""
    return {
        "manifest_fingerprint": provenance["manifest_fingerprint"],
        "training_data_hash": provenance["training_data_hash"],
        "training_schema_hash": provenance["training_schema_hash"],
        "master_seed": str(provenance["master_seed"]),
        "master_seed_source": provenance["master_seed_source"],
    }


def _declared_family(configuration: Mapping[str, Any]) -> str | None:
    """The family a candidate's configuration declares, if any."""
    family = configuration.get("family", configuration.get("total_family"))
    return None if family is None else str(family)


def _diagnostics(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``metadata["model"]["diagnostics"]``, or an empty mapping."""
    model = metadata.get("model")
    diagnostics = model.get("diagnostics") if isinstance(model, Mapping) else None
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _tuning_results(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Each tuned component's Optuna evidence, keyed by component name.

    Model A tunes one study per cohort under ``diagnostics.tuning``. Model B
    tunes its total and probability components under
    ``diagnostics.selection``. The Bayesian model tunes nothing.
    """
    diagnostics = _diagnostics(metadata)
    tuning = diagnostics.get("tuning")
    if isinstance(tuning, Mapping):
        return dict(tuning)
    selection = diagnostics.get("selection")
    if not isinstance(selection, Mapping):
        return {}
    return {
        component: selection[key]
        for key, component in (
            ("total_tuning", "total"),
            ("probability_tuning", "probability"),
        )
        if isinstance(selection.get(key), Mapping)
    }


def _search_space(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The tuning search space, which Model A and Model B record in different places."""
    diagnostics = _diagnostics(metadata)
    if isinstance(diagnostics.get("search_space"), Mapping):
        return diagnostics["search_space"]
    selection = diagnostics.get("selection")
    if isinstance(selection, Mapping) and isinstance(
        selection.get("search_space"), Mapping
    ):
        return selection["search_space"]
    return None
