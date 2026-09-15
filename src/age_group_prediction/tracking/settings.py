"""Tracking settings read from the environment, and artifact-location handling."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "age-group-prediction"
ARTIFACT_LOCATION_ENV = "AGE_GROUP_MLFLOW_ARTIFACT_LOCATION"


@dataclass(frozen=True)
class TrackingSettings:
    """Where runs and artifacts are stored."""

    tracking_uri: str
    experiment_name: str
    artifact_location: str | None


def resolve_tracking_settings(
    environ: Mapping[str, str] | None = None,
) -> TrackingSettings:
    """Read tracking settings from the environment, with documented defaults."""
    environ = os.environ if environ is None else environ
    tracking_uri = environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    configured_location = environ.get(ARTIFACT_LOCATION_ENV)
    artifact_location = (
        _normalize_location(configured_location)
        if configured_location
        else _default_artifact_location(tracking_uri)
    )
    return TrackingSettings(
        tracking_uri=tracking_uri,
        experiment_name=environ.get("MLFLOW_EXPERIMENT_NAME")
        or DEFAULT_EXPERIMENT_NAME,
        artifact_location=artifact_location,
    )


def _default_artifact_location(tracking_uri: str) -> str | None:
    """Place artifacts beside a SQLite database file; defer to a server otherwise."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    database = tracking_uri[len(prefix) :].split("?", 1)[0]
    if not database or database == ":memory:":
        return None
    return _normalize_location(str(Path(database).parent / "mlartifacts"))


def _normalize_location(location: str) -> str:
    """Turn a local path or ``file:`` URI into one absolute ``file:`` URI.

    MLflow may store a local location as a plain path or a ``file:`` URI, so
    both sides are normalized before an existing experiment is compared.
    Remote URIs (``s3://``, ``mlflow-artifacts:`` and so on) are left as given.
    """
    parsed = urlparse(location)
    if parsed.scheme == "file":
        # Decoded first: `as_uri` percent-encodes (a space becomes %20), so an
        # undecoded path would be encoded twice and never match again.
        return Path(url2pathname(parsed.path)).resolve().as_uri()
    if parsed.scheme and len(parsed.scheme) > 1:
        return location
    return Path(location).expanduser().resolve().as_uri()
