"""Serialization helpers: params, JSON, gzipped JSON, CSV and tag values."""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

# MLflow truncates longer param values with only a warning. Truncation here is
# deliberate, and the untruncated values are stored in `params_full.json`.
_MAX_PARAM_VALUE_LENGTH = 6000


def _flatten(prefix: str, value: Any) -> dict[str, str]:
    """Flatten nested mappings to dotted param names; lists become JSON text.

    An empty mapping is kept as ``{}`` rather than vanishing, and two paths
    that flatten to one name (a key ``"a.b"`` beside ``a -> b``) are refused.
    """
    if isinstance(value, Mapping) and value:
        flattened: dict[str, str] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            for flat_name, text in _flatten(name, item).items():
                if flat_name in flattened:
                    raise ValueError(f"Two params flatten to the name '{flat_name}'")
                flattened[flat_name] = text
        return flattened
    text = value if isinstance(value, str) else json.dumps(value)
    return {prefix: text}


def _log_params(params: Mapping[str, str], root: Path) -> None:
    """Log params, truncating overlong values and keeping the full set as JSON."""
    truncated = {
        key: value
        for key, value in params.items()
        if len(value) > _MAX_PARAM_VALUE_LENGTH
    }
    mlflow.log_params(
        {key: value[:_MAX_PARAM_VALUE_LENGTH] for key, value in params.items()}
    )
    if truncated:
        _write_json(
            root / "params_full.json",
            {"truncated_keys": sorted(truncated), "params": dict(params)},
        )


def _write_json(path: Path, payload: Any) -> None:
    """Write indented, key-sorted JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_gzipped_json(path: Path, payload: Any) -> None:
    """Write strict JSON through gzip, reproducibly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 so identical bundles produce identical bytes.
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(json.dumps(payload, allow_nan=False).encode("utf-8"))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a frame as CSV, or nothing when it has no rows."""
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _trials_table(tuning: Mapping[str, Any]) -> pd.DataFrame:
    """One row per Optuna trial, pruned trials included."""
    return pd.DataFrame(
        [
            {
                "number": trial["number"],
                "state": trial["state"],
                "value": trial["value"],
                **{f"param_{key}": value for key, value in trial["params"].items()},
                # (step, value) pairs, as JSON so the column stays one cell.
                "intermediate_values": json.dumps(
                    [list(pair) for pair in trial["intermediate_values"]]
                ),
            }
            for trial in tuning["trials"]
        ]
    )


def _tag_bool(value: bool) -> str:
    """Render a boolean as the lowercase tag value ``true`` or ``false``."""
    return "true" if value else "false"
