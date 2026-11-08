"""Characterize the objective EDA notebook's script run: setup only, nothing fitted.

The notebook (``notebooks/02_objective_eda.py``) builds its data in setup
cells and fits models only behind its run button. A plain script run leaves
the button unclicked, so it must print the facts of the setup and fit
nothing. These tests run the notebook as a subprocess and read its stdout,
rather than importing marimo internals, so they stay valid across marimo
versions; the last ones read the names a script run defines.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import optuna
import pytest
from optuna.samplers import RandomSampler
from optuna.trial import TrialState

from age_group_prediction.hyperparameter_tuning import (
    suggest_total_times_probability_components,
)
from student_simulator import load_simulation_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_objective_eda.py"


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
    env["MLFLOW_EXPERIMENT_NAME"] = "test-objective-eda-notebook"
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


def _flatten(mapping: dict, prefix: str = "") -> dict:
    """The notebook's ``flatten``, copied: the notebook cannot be imported here."""
    flat = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _canonical_hash() -> str:
    """The config hash of the notebook's defaults, computed here."""
    base = load_simulation_config(PROJECT_ROOT / "configs" / "simulation.yml")
    config = base.with_overrides(
        {
            "simulation": {"n_neighborhoods": 150},
            "building": {"buildings_per_neighborhood_rate": 9.0},
        }
    )
    config_params = _flatten(config.model_dump())
    return hashlib.sha256(
        json.dumps(config_params, sort_keys=True).encode()
    ).hexdigest()


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
    # left out of them is a value left out of MLflow. The EDA's settings are
    # the ones it reads; the study directory's name is built from them.
    assert _printed(default_run.stdout, "data_settings") == {
        "config_path": "configs/simulation.yml",
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
        "n_trials": "100",
        "n_startup_trials": "10",
    }


def test_an_override_not_reaching_the_simulator_is_caught(
    default_run: subprocess.CompletedProcess[str],
) -> None:
    # The config alone gives 60 neighborhoods and far fewer rows; a config
    # hash without the overrides would also differ from the one computed here.
    assert _printed(default_run.stdout, "data") == {
        "rows": "1529",
        "neighborhoods": "150",
        "config_hash": _canonical_hash(),
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
    # shows whether it is wired: the printed seed and the config hash must
    # change, and so must the population (1561 rows at seed 7, measured), which
    # the hash alone cannot show.
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
    printed = _printed(result.stdout, "data")
    assert printed["config_hash"] != _canonical_hash()
    assert printed["rows"] == "1561"


def test_a_tuning_cell_outside_a_run_button_is_caught() -> None:
    # A cell that tunes without `mo.stop(not <button>.value)` would fit its
    # run's studies on every script run, the notebook test included. A string
    # check on the cells that call the tuner's tune methods (each EDA run cell
    # calls one; the cell building the tuner calls none): the stop's presence,
    # not its position; the script run's wall time above is the behavioural
    # guard.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = source.split("@app.cell")
    tuning_cells = [cell for cell in cells if "tuner.tune_" in cell]
    assert len(tuning_cells) >= 1
    assert all(re.search(r"mo\.stop\(\s*not ", cell) for cell in tuning_cells)


@pytest.fixture(scope="module")
def notebook_defs() -> dict:
    """The names a script run defines, in this process; the button stays unclicked, so nothing is fitted."""
    spec = importlib.util.spec_from_file_location(
        "objective_eda_notebook", NOTEBOOK_PATH
    )
    assert spec is not None and spec.loader is not None
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    _, defs = notebook.app.run()
    return dict(defs)


def test_a_study_is_named_by_the_settings_that_shape_it(notebook_defs: dict) -> None:
    # A stored study is loaded by its name alone, so a setting left out of the
    # directory's or the file's name would load a stale study after a change.
    # The EDA's Model 1 run settings are the tuning notebook's first run's, so
    # its deviance study is that run's.
    assert notebook_defs["tuner"].storage_dir == (
        PROJECT_ROOT
        / "studies"
        / (
            f"data={_canonical_hash()[:8]}__split_method=random__test_size=0.2__"
            "val_size=0.2__n_splits=5__seed=42__n_trials=100__n_startup_trials=10"
        )
    )
    run_settings = {
        "family": "lgbm_poisson",
        "use_exposure": True,
        "share_hyperparams": True,
        "share_feature_transformer": False,
    }
    assert notebook_defs["build_study_name"](
        "independent_cohorts", run_settings, "poisson_deviance", 42
    ) == (
        "independent_cohorts__family=lgbm_poisson__use_exposure=True__"
        "share_hyperparams=True__share_feature_transformer=False__"
        "objective=poisson_deviance__seed=42"
    )


def test_every_trial_is_a_row_of_the_whole_model_it_scored(notebook_defs: dict) -> None:
    # PR 3's review (F3): build_trials_table names each component's
    # transformer by identity in its family's candidates and reads its
    # hyperparameters, the classifier's with its calibration. Three trials
    # drawn as the joint method draws them, the third failed: a failed trial
    # records no score.
    count_transformers = notebook_defs["COUNT_FEATURE_TRANSFORMERS"]["lgbm_poisson"]
    probability_transformers = notebook_defs["PROBABILITY_FEATURE_TRANSFORMERS"][
        "logistic"
    ]
    metric_names = ["poisson_deviance", "rmse"]
    study = optuna.create_study(sampler=RandomSampler(seed=0))
    components_by_trial = []
    for number in range(3):
        trial = study.ask()
        components_by_trial.append(
            suggest_total_times_probability_components(
                trial,
                total_family="lgbm_poisson",
                probability_family="logistic",
                use_exposure=True,
                total_feature_transformer_dict=count_transformers,
                probability_feature_transformer_dict=probability_transformers,
            )
        )
        if number < 2:
            trial.set_user_attr("poisson_deviance", 2.0 + number)
            trial.set_user_attr("rmse", 4.0 + number)
            study.tell(trial, 2.0 + number)
        else:
            study.tell(trial, state=TrialState.FAIL)

    table = notebook_defs["build_trials_table"](
        study,
        "poisson_deviance",
        metric_names,
        components_by_trial,
        {"total": count_transformers, "probability": probability_transformers},
    )

    assert list(table.columns) == [
        "id",
        "objective",
        "seconds",
        "state",
        *metric_names,
        "feature_transformers",
        "hyperparameters",
    ]
    assert table["id"].tolist() == [0, 1, 2]
    assert table["state"].tolist() == ["COMPLETE", "COMPLETE", "FAIL"]
    assert set(table["objective"]) == {"poisson_deviance"}
    assert table["poisson_deviance"].tolist()[:2] == [2.0, 3.0]
    assert table["rmse"].tolist()[:2] == [4.0, 5.0]
    assert all(math.isnan(table[name].iloc[2]) for name in metric_names)
    for row, trial, components in zip(
        table.itertuples(), study.trials, components_by_trial, strict=True
    ):
        total, probability = components
        assert row.feature_transformers == {
            "total": trial.params["total__feature_transformer"],
            "probability": trial.params["probability__feature_transformer"],
        }
        assert row.hyperparameters == {
            "total": dataclasses.asdict(total.estimator_params),
            "probability": {
                **dataclasses.asdict(probability.estimator_params),
                "calibration_method": probability.calibration_method,
            },
        }
