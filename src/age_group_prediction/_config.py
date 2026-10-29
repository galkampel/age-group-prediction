"""The pydantic config shared by the user-built value objects of ``hyperparameter_tuning.aggregation`` and ``inspection``."""

from __future__ import annotations

from pydantic import ConfigDict

# Strict: no silent conversion, so "0.1" is not read as 0.1, nor 1 as True.
# NaN and inf would otherwise pass here and surface only when a trial runs.
# A misspelt keyword (pydantic's default is to ignore extras) would leave the
# intended setting at its default.
STRICT = ConfigDict(strict=True, allow_inf_nan=False, extra="forbid")
