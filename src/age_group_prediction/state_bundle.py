"""Shared helpers for JSON state bundles of fitted models.

A state bundle is the project's fitted-model artifact (plan section 11): one
JSON document that rebuilds a fitted model **without pickling and without the
training rows**. ``BaseAgeGroupModel.to_state_bundle`` and
``BaseAgeGroupModel.from_state_bundle`` own the lifecycle; this module holds
the pieces that belong to no single model, so that each model only describes
its own fitted state.

"No training rows" is precise: no per-building record and no building ID is
stored, and the training frame is identified only by hash. Fitted *summary*
statistics are model state and do appear:

* preprocessing: scaling means and scales, spline knots spanning the data
  range, and the fit row count;
* LightGBM trees: split thresholds, per-feature ``[min:max]`` bounds, and
  per-node sample counts (a small leaf can describe only a few buildings);
* the Bayesian model: the fitted neighborhood IDs, which index its random
  effects;
* fit evidence: per-fold tuning scores and calibration row and child counts.

A bundle is therefore a model artifact, not an anonymized one; store it with
the same care as the training data summaries it was fitted from.

Every bundle has the same top-level layout (format ``"2"``):

=========================  ====================================================
``bundle_format``          ``STATE_BUNDLE_FORMAT``; a loader refuses others
``model_class``            class that wrote the bundle; a loader refuses others
``implementation_version`` the class's ``implementation_version``; refused if
                           it differs, because fitted state from another
                           implementation may not mean the same thing
``schema``                 ``asdict(ModelingSchema)``
``default_rng_seed``       constructor seed
``feature_transformer``    ``FittedFeatureTransformer.to_state()``
``training_data_hash``     hashes of the frame the model was fitted on
``training_schema_hash``
``dependency_versions``    informational only; never enforced on load
``model_state``            the model-specific payload
=========================  ====================================================

Format history: ``"1"`` (Gate 5) stored no preprocessing state and required the
training frame to refit it. ``"2"`` (Gate 7) stores the fitted preprocessing, so
the training frame is optional. No format-1 bundle was ever persisted, so the
format-1 reader was removed rather than kept.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import version
from typing import Any

import pandas as pd

from .hashing import column_schema_hash, table_hash
from .modeling_config import DEFAULT_TOTAL_FEATURE_SPEC, FeatureSpec, ModelingSchema

STATE_BUNDLE_FORMAT = "2"

__all__ = [
    "STATE_BUNDLE_FORMAT",
    "check_bundle_header",
    "dependency_versions",
    "restore_config",
    "restore_feature_spec",
    "verify_training_frame",
]


def restore_config(template: Any, payload: Mapping[str, object]) -> Any:
    """Rebuild a frozen config dataclass from its JSON round trip.

    Accepts both ``dataclasses.asdict`` output (nested dicts, tuples) and its
    JSON round trip (nested dicts, lists): JSON has no tuple or dataclass type.
    ``template`` is an already-valid instance of the target class, used only to
    discover which fields are nested configs and which are tuples.

    The payload must name exactly the class's fields. A missing field would
    silently take its default and an unknown one would be silently dropped, and
    either would rebuild a different configuration than the one that was fitted.
    Every rebuilt class also re-runs its own ``__post_init__`` validation.
    """
    expected = {field.name for field in fields(template)}
    missing, unknown = expected - set(payload), set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"Cannot rebuild {type(template).__name__}: missing fields "
            f"{sorted(missing)}, unknown fields {sorted(unknown)}"
        )
    values: dict[str, object] = {}
    for field in fields(template):
        current = getattr(template, field.name)
        value = payload[field.name]
        if is_dataclass(current) and isinstance(value, Mapping):
            values[field.name] = restore_config(current, value)
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            values[field.name] = _restore_tuple(type(template), field.name, current, value)
        else:
            values[field.name] = value
    return type(template)(**values)


def _restore_tuple(
    owner: type,
    field_name: str,
    template: tuple[Any, ...],
    value: list[Any] | tuple[Any, ...],
) -> tuple[Any, ...]:
    """Restore a tuple field, rebuilding nested dataclass elements element-wise.

    A field such as ``ModelingSchema.categorical_feature_specs`` is a tuple of
    dataclasses, which JSON flattens to a list of dicts. The template's own
    first element supplies the class to rebuild them with.
    """
    if not any(isinstance(item, Mapping) for item in value):
        return tuple(value)
    if not template:
        raise ValueError(
            f"Cannot rebuild {owner.__name__}.{field_name} from the bundle: the "
            "template has no element to infer the nested type from"
        )
    return tuple(restore_config(template[0], item) for item in value)


def restore_feature_spec(payload: Mapping[str, object]) -> FeatureSpec:
    """Rebuild any ``FeatureSpec``; every field is a primitive or a tuple of them."""
    return restore_config(DEFAULT_TOTAL_FEATURE_SPEC, payload)


def check_bundle_header(
    bundle: Mapping[str, object], *, model_class: type, implementation_version: str
) -> None:
    """Refuse a bundle this loader was not written against, rather than guess."""
    if bundle.get("bundle_format") != STATE_BUNDLE_FORMAT:
        raise ValueError(
            f"Unsupported state bundle format {bundle.get('bundle_format')!r}; "
            f"this loader reads {STATE_BUNDLE_FORMAT!r}"
        )
    if bundle.get("model_class") != model_class.__name__:
        raise ValueError(
            f"State bundle was written by {bundle.get('model_class')!r}, "
            f"not {model_class.__name__!r}"
        )
    if bundle.get("implementation_version") != implementation_version:
        raise ValueError(
            f"State bundle was written by {model_class.__name__} implementation "
            f"version {bundle.get('implementation_version')!r}, but this code is "
            f"version {implementation_version!r}; refit instead of reusing "
            "fitted state from a different implementation"
        )


def verify_training_frame(
    train_df: pd.DataFrame, *, bundle: Mapping[str, object], schema: ModelingSchema
) -> None:
    """Check that ``train_df`` is exactly the frame the bundled model was fitted on.

    Only needed when the caller supplies the frame, which bootstrap-based
    predictive draws require. A different frame would silently produce
    different uncertainty, so it is rejected by name.
    """
    observed = {
        "training_data_hash": table_hash(train_df, id_column=schema.building_id_column),
        "training_schema_hash": column_schema_hash(train_df),
    }
    for label, value in observed.items():
        if bundle[label] != value:
            raise ValueError(
                f"train_df does not match the bundle: {label} is {value} but the "
                f"bundle records {bundle[label]}. The model was fitted on a "
                "different training partition."
            )


def dependency_versions(*distributions: str) -> dict[str, str]:
    """Return installed versions of the named distributions, for provenance."""
    return {name: version(name) for name in distributions}
