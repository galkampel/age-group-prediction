"""Characterize the experiment notebook's script run: setup only, nothing fitted.

The notebook (``notebooks/02_experiments.py``) builds its data in setup cells
and will fit models only behind run buttons. A plain script run leaves every
button unclicked, so it must print the facts of the sections built so far
and fit nothing. These tests run the notebook as a subprocess and read its
stdout, rather than importing marimo internals, so they stay valid across
marimo versions.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from age_group_prediction.preprocessing import ShareTransformer
from student_simulator import StudentPopulationSimulator, load_simulation_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_experiments.py"


def _run(notebook: Path, scratch: Path) -> subprocess.CompletedProcess[str]:
    """Run ``notebook`` as a script against a scratch MLflow store under ``scratch``.

    The editable install does not resolve for a plain ``python <script>``
    invocation here (pytest works through ``pyproject.toml``'s ``pythonpath``),
    so ``src`` is passed explicitly. The MLflow variables point away from the
    repository's canonical store for the cells that will use them.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    env["MLFLOW_TRACKING_URI"] = f"sqlite:///{scratch / 'mlflow.db'}"
    env["MLFLOW_EXPERIMENT_NAME"] = "test-experiments-notebook"
    env["AGE_GROUP_MLFLOW_ARTIFACT_LOCATION"] = str(scratch / "artifacts")
    return subprocess.run(
        [sys.executable, str(notebook)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _printed(stdout: str, prefix: str) -> dict[str, str]:
    """The ``name=value`` pairs of the one stdout line starting with ``prefix``."""
    lines = [line for line in stdout.splitlines() if line.startswith(prefix + " ")]
    assert len(lines) == 1, f"expected one {prefix!r} line, got {lines!r}"
    pairs = (item.split("=", 1) for item in lines[0].split(" ")[1:])
    return {name: value for name, value in pairs}


def _canonical_hash() -> str:
    """The hash of the table the notebook's defaults build, computed here."""
    base = load_simulation_config(PROJECT_ROOT / "configs" / "simulation.toml")
    config = base.model_copy(
        update={
            "simulation": base.simulation.model_copy(update={"n_neighborhoods": 150}),
            "building": base.building.model_copy(
                update={"buildings_per_neighborhood_rate": 9.0}
            ),
        }
    )
    table = ShareTransformer(
        ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
    ).fit_transform(StudentPopulationSimulator.from_config(config).simulate().dataset)
    return hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()


@pytest.fixture(scope="module")
def default_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> subprocess.CompletedProcess[str]:
    """One script run of the notebook as committed, shared by the tests."""
    result = _run(NOTEBOOK_PATH, tmp_path_factory.mktemp("store"))
    assert result.returncode == 0, result.stderr
    return result


def test_a_setting_missing_or_changed_in_a_settings_mapping_is_caught(
    default_run: subprocess.CompletedProcess[str],
) -> None:
    # Every run will log DATA_SETTINGS and EXPERIMENT_SETTINGS, so a value
    # left out of them is a value left out of MLflow, and a default that
    # drifts from the notebook plan's §9 D1a goes unnoticed.
    assert _printed(default_run.stdout, "data_settings") == {
        "config_path": "configs/simulation.toml",
        "n_neighborhoods": "150",
        "buildings_per_neighborhood_rate": "9.0",
        "simulation_seed": "42",
    }
    assert _printed(default_run.stdout, "experiment_settings") == {
        "seed": "42",
        "split_method": "random",
        "test_size": "0.2",
        "val_size": "0.2",
        "n_splits": "5",
        "n_trials": "200",
        "n_resamples": "1000",
    }


def test_an_override_not_reaching_the_simulator_is_caught(
    default_run: subprocess.CompletedProcess[str],
) -> None:
    # The config alone gives 60 neighborhoods and far fewer rows; a hash over
    # less than the full table would also differ from the one computed here.
    assert _printed(default_run.stdout, "data") == {
        "rows": "1529",
        "neighborhoods": "150",
        "data_hash": _canonical_hash(),
    }


def test_a_dropped_position_mapping_in_the_split_is_caught(
    default_run: subprocess.CompletedProcess[str],
) -> None:
    # Measured on this table (random, seed 42): a second share of 0.2 instead
    # of val_size / (1 - test_size) = 0.25 gives 978/245/306, and the whole
    # table given to the second call gives 1146/383/306. The sizes do not
    # catch the second call's positions used as table positions (917/306/306
    # with 250 rows in two parts); the notebook maps them with
    # train_val_index[...] instead.
    assert _printed(default_run.stdout, "split") == {
        "n_train": "917",
        "n_val": "306",
        "n_test": "306",
    }


def test_a_seed_override_not_reaching_the_simulator_is_caught(tmp_path: Path) -> None:
    # The default keeps the config's seed, so only a run with the override set
    # shows whether it is wired: the population and the printed seed must change.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert source.count("simulation_seed = None") == 1
    # A copy beside a configs/ link, because the notebook finds the config
    # relative to its own directory.
    project_copy = tmp_path / "project"
    (project_copy / "notebooks").mkdir(parents=True)
    (project_copy / "configs").symlink_to(
        PROJECT_ROOT / "configs", target_is_directory=True
    )
    notebook_copy = project_copy / "notebooks" / NOTEBOOK_PATH.name
    notebook_copy.write_text(
        source.replace("simulation_seed = None", "simulation_seed = 7"),
        encoding="utf-8",
    )
    result = _run(notebook_copy, tmp_path / "store")
    assert result.returncode == 0, result.stderr
    assert _printed(result.stdout, "data_settings")["simulation_seed"] == "7"
    assert _printed(result.stdout, "data")["data_hash"] != _canonical_hash()


def test_a_tuning_cell_outside_a_run_button_is_caught() -> None:
    # A cell that tunes without `mo.stop(not <button>.value)` would fit three
    # studies on every script run, the notebook test included. A string check
    # on the cells that name `tune_model` (the run cell calls it through
    # joblib's `delayed`): the stop's presence, not its position; the script
    # run's wall time above is the behavioural guard.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = source.split("@app.cell")
    tuning_cells = [
        cell for cell in cells if "tune_model" in cell and "def tune_model(" not in cell
    ]
    assert len(tuning_cells) >= 1
    assert all("mo.stop(not " in cell for cell in tuning_cells)
