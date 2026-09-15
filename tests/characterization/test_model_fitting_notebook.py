"""Characterize the Gate 8 marimo client's default (non-interactive) behavior.

The notebook itself is a thin client (Gate 8 session handoff, section 10): a
plain script run must perform setup and validation only, and the notebook must
route both expensive actions through package APIs rather than calling a
model's own ``fit``/``predict``/``evaluate``. These tests describe that
contract from the outside -- by actually running the notebook as a script and
by reading its source -- rather than by importing marimo internals, so they
stay valid across marimo versions.

Running the notebook this way is slow (it builds the canonical 150-
neighborhood population and modeling table), so these tests are marked
``slow`` like the project's other real-model checks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
mlflow = pytest.importorskip("mlflow")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_model_fitting.py"
NOTEBOOK_SOURCE = NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_notebook_is_nonempty_and_declares_marimo_app() -> None:
    """The notebook must actually exist as a marimo app (not the old placeholder)."""
    assert NOTEBOOK_SOURCE.strip(), "notebooks/02_model_fitting.py must not be empty"
    assert "app = marimo.App()" in NOTEBOOK_SOURCE
    assert 'if __name__ == "__main__":' in NOTEBOOK_SOURCE
    assert "app.run()" in NOTEBOOK_SOURCE


def test_notebook_never_calls_model_fit_predict_or_evaluate_directly() -> None:
    """The notebook must route all model work through package APIs, never a model's own methods.

    A model instance is never in scope in this notebook (the registry exposes
    only factories and frozen candidate definitions), so a direct
    ``.fit(``/``.predict(``/``.evaluate(`` call would be a real boundary
    violation rather than a false positive on an unrelated method name.
    """
    for forbidden in (".fit(", ".predict(", ".evaluate("):
        assert forbidden not in NOTEBOOK_SOURCE, (
            f"notebook must not call a model method directly: found {forbidden!r}"
        )


def test_notebook_uses_the_lockbox_aware_orchestrator_for_final_work() -> None:
    """The one-time final action must go through ``run_final_evaluation``, not ad hoc calls."""
    assert "run_final_evaluation(" in NOTEBOOK_SOURCE
    assert "evaluate_frozen_models_on_lockbox(" not in NOTEBOOK_SOURCE
    assert "refit_frozen_approach_winners(" not in NOTEBOOK_SOURCE


def test_notebook_gates_both_expensive_actions_behind_run_buttons() -> None:
    """CV and the final action each need their own explicit button; final also needs a checkbox."""
    assert NOTEBOOK_SOURCE.count("mo.ui.run_button(") == 2
    assert NOTEBOOK_SOURCE.count("mo.ui.checkbox(") == 1
    assert "run_cv_button.value" in NOTEBOOK_SOURCE
    assert "run_final_button.value and confirm_final_checkbox.value" in NOTEBOOK_SOURCE


def test_notebook_never_displays_the_holdout_frame() -> None:
    """Setup must show only ``split.train_df``, never ``split.test_df``."""
    assert "split.test_df" not in NOTEBOOK_SOURCE


def test_notebook_never_previews_the_full_modeling_table_with_targets() -> None:
    """The pre-split modeling-table preview must withhold target columns.

    Every row of ``modeling_df`` may still become a holdout building at the
    point it is first built (the outer split has not run yet), so a bare
    ``modeling_df.head(...)`` would risk displaying a real target value for a
    building the manifest later assigns to the one-time lockbox holdout --
    caught independently of ``test_notebook_never_displays_the_holdout_frame``,
    which only looks for ``split.test_df`` and would miss this class of leak.
    """
    assert "modeling_df.head(" not in NOTEBOOK_SOURCE
    assert "DEFAULT_MODELING_SCHEMA.target_columns" in NOTEBOOK_SOURCE


@pytest.mark.slow
def test_default_script_execution_performs_setup_only(tmp_path, monkeypatch) -> None:
    """A plain script run must create no MLflow experiment, run, or artifact.

    Both `mo.ui.run_button` controls default to unclicked when a notebook
    runs as a script, so every cell gated behind one is skipped by marimo's
    own reactive `mo.stop` semantics -- this proves it end to end rather than
    trusting that the gating pattern is wired correctly.
    """
    store_root = tmp_path / "store"
    store_root.mkdir()
    tracking_uri = f"sqlite:///{store_root / 'mlflow.db'}"
    artifact_location = str(store_root / "artifacts")

    env = dict(os.environ)
    # `age_group_prediction`/`student_simulator`'s editable install does not
    # resolve for a plain `python <script>.py` invocation in this
    # environment (reproduced independently of this notebook: the untouched
    # `01_eda.py` fails identically with `ModuleNotFoundError`); `pytest`
    # itself only works because of `pyproject.toml`'s `pythonpath = ["src"]`.
    # This is a pre-existing environment characteristic, not something Gate 8
    # introduced, so the subprocess is given the same `src` path explicitly.
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["AGE_GROUP_MLFLOW_ARTIFACT_LOCATION"] = artifact_location
    env.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    env.pop("MLFLOW_EXPERIMENT_NAME", None)
    env.pop("MLFLOW_EXPERIMENT_ID", None)

    # The notebook derives its lockbox manifest path from its own directory, so
    # it runs from a throwaway project copy. Run from the repository, a setup
    # run on a checkout without a persisted manifest would create the canonical
    # lockbox as a side effect of the test suite.
    project_copy = tmp_path / "project"
    (project_copy / "notebooks").mkdir(parents=True)
    (project_copy / "configs").symlink_to(PROJECT_ROOT / "configs", target_is_directory=True)
    notebook_copy = project_copy / "notebooks" / NOTEBOOK_PATH.name
    notebook_copy.write_text(NOTEBOOK_SOURCE, encoding="utf-8")
    canonical_manifest = PROJECT_ROOT / "artifacts" / "lockbox" / "split_manifest.json"
    canonical_manifest_before = (
        canonical_manifest.stat().st_mtime_ns if canonical_manifest.exists() else None
    )

    completed = subprocess.run(
        [sys.executable, str(notebook_copy)],
        cwd=project_copy,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    # No MLflow database file means no run was ever opened: `tracked_comparison`
    # and `tracked_final_evaluation` both call `mlflow.start_run` only after
    # `_ensure_experiment`, which itself only runs once a run is about to open.
    assert not (store_root / "mlflow.db").exists()

    # Setup created a manifest inside the copy and left the repository's alone.
    assert (project_copy / "artifacts" / "lockbox" / "split_manifest.json").exists()
    canonical_manifest_after = (
        canonical_manifest.stat().st_mtime_ns if canonical_manifest.exists() else None
    )
    assert canonical_manifest_after == canonical_manifest_before
