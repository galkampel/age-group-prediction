"""The pydantic config shared by the package's user-built value objects."""

from __future__ import annotations

from pydantic import ConfigDict

# Strict: no silent conversion, so "0.1" is not read as 0.1, nor 1 as True.
# NaN and inf would otherwise pass here and surface only when a trial runs.
_STRICT = ConfigDict(strict=True, allow_inf_nan=False)
