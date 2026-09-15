"""Parity contract for the Jupyter EDA reference run."""

from __future__ import annotations

import json

import pytest
from eda_baseline import (
    BASELINE_PATH,
    capture_eda_baseline,
)


@pytest.fixture(name="captured_baseline", scope="module")
def _captured_baseline() -> dict[str, object]:
    """Capture the EDA population once for the focused parity checks."""
    return capture_eda_baseline()


def test_eda_reference_matches_committed_jupyter_baseline(
    captured_baseline: dict[str, object],
) -> None:
    """Prevent unreviewed drift from the frozen Jupyter EDA reference."""
    expected = json.loads(BASELINE_PATH.read_text())
    assert captured_baseline == expected


def test_eda_reference_is_reproducible(
    captured_baseline: dict[str, object],
) -> None:
    """Verify a clean second run preserves the exact data parity contract."""
    repeated_baseline = capture_eda_baseline()
    assert (
        repeated_baseline["final_dataframe"]
        == captured_baseline["final_dataframe"]
    )
    assert (
        repeated_baseline["quality_checks"]
        == captured_baseline["quality_checks"]
    )
    assert repeated_baseline["summaries"] == captured_baseline["summaries"]
