"""Characterize the hyperparameter tuning notebook's script run: setup only, nothing fitted.

The notebook (``notebooks/03_hyperparameter_tuning.py``) builds its data in
setup cells and fits models only behind run buttons. A plain script run leaves every
button unclicked, so it must print the facts of the sections built so far
and fit nothing. These tests run the notebook as a subprocess and read its
stdout, rather than importing marimo internals, so they stay valid across
marimo versions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import pytest
import shap
from optuna.samplers import RandomSampler
from scipy.stats import gaussian_kde
from sklearn.utils import Bunch

from age_group_prediction.hyperparameter_tuning import (
    suggest_independent_cohorts_components,
    suggest_total_times_probability_components,
)
from student_simulator import load_simulation_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_hyperparameter_tuning.py"
OBJECTIVE_EDA_PATH = PROJECT_ROOT / "notebooks" / "02_objective_eda.py"


def _mlflow_environment(scratch: Path) -> dict[str, str]:
    """The MLflow variables of a scratch store under ``scratch``, away from the canonical one."""
    return {
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "MLFLOW_TRACKING_URI": f"sqlite:///{scratch / 'mlflow.db'}",
        "MLFLOW_EXPERIMENT_NAME": "test-hyperparameter-tuning-notebook",
        "AGE_GROUP_MLFLOW_ARTIFACT_LOCATION": str(scratch / "artifacts"),
    }


def _run(
    notebook: Path,
    scratch: Path,
    cwd: Path = PROJECT_ROOT,
    environment: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``notebook`` as a script from ``cwd`` against a scratch MLflow store under ``scratch``.

    The editable install does not resolve for a plain ``python <script>``
    invocation here (pytest works through ``pyproject.toml``'s ``pythonpath``),
    so ``src`` is passed explicitly. The MLflow variables point away from the
    repository's canonical store; ``environment`` overrides them (a ``None``
    value removes one).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env.update(_mlflow_environment(scratch))
    for name, value in (environment or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return subprocess.run(
        [sys.executable, str(notebook)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _project_copy(tmp_path: Path, source: str) -> Path:
    """``source`` as the notebook of a project copy under ``tmp_path``; its path.

    The notebook finds the config and the code relative to its own directory,
    so the copy's ``configs`` and ``src`` link to the repository's.
    """
    project_copy = tmp_path / "project"
    (project_copy / "notebooks").mkdir(parents=True)
    for name in ("configs", "src"):
        (project_copy / name).symlink_to(PROJECT_ROOT / name, target_is_directory=True)
    notebook_copy = project_copy / "notebooks" / NOTEBOOK_PATH.name
    notebook_copy.write_text(source, encoding="utf-8")
    return notebook_copy


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
    # left out of them is a value left out of MLflow, and a default that
    # drifts from the notebook plan's §9 D1a goes unnoticed.
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
        "objective_metric": "rmse",
        "selection_metric": "rmse",
        "n_resamples": "1000",
        "alpha": "0.05",
        "n_repeats": "10",
        "n_shap_rows": "100",
        "n_shap_background": "50",
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
    notebook_copy = _project_copy(
        tmp_path, source.replace("simulation_seed = None", "simulation_seed = 7")
    )
    result = _run(notebook_copy, tmp_path / "store")
    assert result.returncode == 0, result.stderr
    assert _printed(result.stdout, "data_settings")["simulation_seed"] == "7"
    printed = _printed(result.stdout, "data")
    assert printed["config_hash"] != _canonical_hash()
    assert printed["rows"] == "1561"


def test_a_tuning_cell_outside_a_run_button_is_caught() -> None:
    # A cell that tunes without `mo.stop(not <button>.value)` would fit its
    # run's study on every script run, the notebook test included. A string
    # check on the cells that call the tuner's tune methods (each run cell
    # calls one; the cell building the tuner calls none): the stop's
    # presence, not its position; the script run's wall time above is the
    # behavioural guard.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = source.split("@app.cell")
    tuning_cells = [cell for cell in cells if "tuner.tune_" in cell]
    assert len(tuning_cells) >= 1
    assert all(re.search(r"mo\.stop\(\s*not ", cell) for cell in tuning_cells)


@pytest.fixture(scope="module")
def notebook_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The scratch MLflow store of the in-process run below."""
    return tmp_path_factory.mktemp("notebook_store")


@pytest.fixture(scope="module")
def notebook_defs(notebook_store: Path) -> dict:
    """The names a script run defines, in this process; the buttons stay unclicked, so nothing is fitted.

    The MLflow variables point at a scratch store for the run only; the MLflow
    cell sets no global, so nothing reaches the tests that run after.
    """
    spec = importlib.util.spec_from_file_location(
        "hyperparameter_tuning_notebook", NOTEBOOK_PATH
    )
    assert spec is not None and spec.loader is not None
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    with pytest.MonkeyPatch.context() as patch:
        for name, value in _mlflow_environment(notebook_store).items():
            patch.setenv(name, value)
        _, defs = notebook.app.run()
    return dict(defs)


def test_a_script_run_opens_no_mlflow_run(
    notebook_defs: dict, notebook_store: Path
) -> None:
    # M1: the MLflow cell creates the experiment with its artifact location and
    # reads the code's tags (the notebook and git); it opens no run, and sets no
    # MLflow global (the tracking URI and the active experiment are read from the
    # environment at every call), so the old tracking tests that run after see
    # none.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(notebook_store)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    experiment = client.get_experiment_by_name(environment["MLFLOW_EXPERIMENT_NAME"])
    assert experiment is not None
    assert experiment.artifact_location == str((notebook_store / "artifacts").resolve())
    experiment_ids = [item.experiment_id for item in client.search_experiments()]
    assert client.search_runs(experiment_ids) == []
    code_tags = notebook_defs["code_tags"]
    assert set(code_tags) == {"notebook", "git_commit", "git_dirty"}
    assert code_tags["notebook"] == "notebooks/03_hyperparameter_tuning.py"
    assert re.fullmatch(r"[0-9a-f]{40}", code_tags["git_commit"])
    assert code_tags["git_dirty"] in {"true", "false"}
    assert os.environ.get("MLFLOW_TRACKING_URI") != environment["MLFLOW_TRACKING_URI"]
    assert mlflow.get_tracking_uri() != environment["MLFLOW_TRACKING_URI"]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MLFLOW_TRACKING_URI", None),
        ("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        ("AGE_GROUP_MLFLOW_ARTIFACT_LOCATION", "mlartifacts"),
    ],
)
def test_the_mlflow_cell_refuses_the_canonical_store(
    tmp_path: Path, name: str, value: str | None
) -> None:
    # D1e: the repo root's mlflow.db and mlartifacts/ are the old stack's final
    # evidence, and MLflow's default URI (unset) is sqlite:///mlflow.db in the
    # working directory. Each run is a copy run from the copy's root, so its
    # "canonical" store is the copy's: a missing refusal opens that one, never
    # the repository's.
    notebook_copy = _project_copy(tmp_path, NOTEBOOK_PATH.read_text(encoding="utf-8"))
    project_copy = notebook_copy.parents[1]
    result = _run(
        notebook_copy, tmp_path / "store", cwd=project_copy, environment={name: value}
    )
    error = result.stderr[-2000:]
    assert result.returncode != 0, error
    assert "ValueError" in error and name in error, error
    assert not (project_copy / "mlflow.db").exists()
    assert not (project_copy / "mlartifacts").exists()


def test_the_mlflow_cell_refuses_an_experiment_with_another_artifact_location(
    tmp_path: Path,
) -> None:
    # The review's F1: an experiment that exists keeps the artifact location it
    # was created with (MLflow creates a missing one with its default, from the
    # repository's root the canonical mlartifacts/), and MLflow uses it over the
    # variable: the cell must refuse it, so no run's artifacts land elsewhere.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(tmp_path / "store")
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    client.create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((tmp_path / "elsewhere").resolve()),
    )
    notebook_copy = _project_copy(tmp_path, NOTEBOOK_PATH.read_text(encoding="utf-8"))
    result = _run(notebook_copy, tmp_path / "store", cwd=notebook_copy.parents[1])
    error = result.stderr[-2000:]
    assert result.returncode != 0, error
    assert "ValueError" in error, error
    assert "AGE_GROUP_MLFLOW_ARTIFACT_LOCATION" in error, error
    experiment_ids = [item.experiment_id for item in client.search_experiments()]
    assert client.search_runs(experiment_ids) == []


def test_the_tuning_runs_are_logged_inside_the_two_tune_cells(
    notebook_defs: dict,
) -> None:
    # §8's L2 (MLFLOW_RUNS_PLAN.md): each Tune cell logs its runs as their
    # studies finish, behind the Tune button's mo.stop only; no logging button.
    # A script run builds what each run logs (the shared settings, the two
    # datasets) and opens no run (test_a_script_run_opens_no_mlflow_run).
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = [cell for cell in source.split("@app.cell") if "tuner.tune_" in cell]
    assert len(cells) == 2
    for cell, button in zip(cells, ("run_model1", "run_model2"), strict=True):
        assert re.search(rf"mo\.stop\(\s*not {button}\.value", cell), button
        assert "_run_id = log_tuning_run(" in cell, button
    # The file reads in the order it runs: the function before the Tune cells.
    assert source.index("def log_tuning_run(") < source.index("tuner.tune_")
    assert "log_tuning_runs" not in source
    assert "log_tuning_runs" not in notebook_defs
    assert notebook_defs["shared_params"] == {
        "config_hash": notebook_defs["config_hash"],
        **notebook_defs["DATA_SETTINGS"],
        **notebook_defs["EXPERIMENT_SETTINGS"],
        "n_train": 917,
        "n_val": 306,
        "n_test": 306,
        "prediction_floor": notebook_defs["PREDICTION_FLOOR"],
    }
    assert len(notebook_defs["shared_params"]) == 23
    dataset_inputs = notebook_defs["tuning_dataset_inputs"]
    assert {context: dataset.name for context, dataset in dataset_inputs.items()} == {
        "training": "train",
        "validation": "val",
    }
    assert all(len(dataset.digest) == 36 for dataset in dataset_inputs.values())


def test_the_best_runs_are_logged_by_the_cell_that_follows_the_inspection(
    notebook_defs: dict,
) -> None:
    # §8's L3: the best runs' run is logged by the cell that reads the
    # inspection's results, with no button: it runs once "Inspect the best
    # runs" has produced them. A script run defines the functions and opens no
    # run (test_a_script_run_opens_no_mlflow_run).
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = [
        cell for cell in source.split("@app.cell") if '"stage": "best_runs"' in cell
    ]
    assert len(cells) == 1
    assert ".value" not in cells[0]
    assert "\n    importance_df,\n" in cells[0]
    assert "_figures = build_dataset_figures(" in cells[0]
    assert "_n_metrics = log_evaluation_run(" in cells[0]
    # Its settings.json is the complete record: every shared setting.
    assert '"stage": "best_runs", **code_tags},\n        shared_params,\n' in cells[0]
    assert "log_best_runs" not in source
    assert callable(notebook_defs["build_dataset_figures"])
    assert callable(notebook_defs["log_evaluation_run"])
    assert "best_runs_permutation" not in notebook_defs
    assert "importance_df" not in notebook_defs


def test_the_final_run_is_logged_by_the_cell_that_follows_the_test_evaluation(
    notebook_defs: dict,
) -> None:
    # §8's L3: the final run is logged by the cell that reads the test
    # evaluation's results, with no button: it runs once "Evaluate on the test
    # part" has produced them, and its guard refuses a second final run. A
    # script run evaluates nothing on test.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = [cell for cell in source.split("@app.cell") if '"stage": "final"' in cell]
    assert len(cells) == 1
    # No menu's or button's `.value` (§9: the SHAP arrays' `.values` are read).
    assert re.search(r"\.value\b", cells[0]) is None
    assert "\n    test_scores_df,\n" in cells[0]
    assert "tags.final = 'true'" in cells[0]
    assert "_n_metrics = log_evaluation_run(" in cells[0]
    # Its settings.json: every shared setting and the refit's size.
    assert '{**shared_params, "n_train_val": len(train_val_age_group.X)},' in cells[0]
    assert "log_final_run" not in source
    for name in ("test_scores_df", "test_predictions", "val_against_test_df"):
        assert name not in notebook_defs, name
    # §9's W1: the final run also logs the test SHAP values and, at a fixed
    # rule, the waterfalls the display draws with the same function.
    assert '"test_shap_values": pd.concat(' in cells[0]
    assert "_n_worst_buildings = 5" in cells[0]
    assert "] = waterfall_figure(" in cells[0]
    displays = [
        cell
        for cell in source.split("@app.cell")
        if "    test_waterfall_figures = {" in cell
    ]
    assert len(displays) == 1
    assert "waterfall_figure(" in displays[0]
    assert "go.Waterfall(" not in displays[0]
    assert callable(notebook_defs["waterfall_figure"])


def test_a_tuning_run_is_logged_once_then_skipped(
    notebook_defs: dict, tmp_path: Path
) -> None:
    # The second review's G3: log_tuning_run called on a scratch store with a
    # 2-trial study and a small tuning run. The first call logs the run; the
    # second finds a FINISHED run with its key and logs nothing (M3's skip).
    # The cold read's T1 and T2: the metrics' values and a dataset input.
    # C1 (TUNING_NOTEBOOK_REVISION_PLAN.md, Task C): the kept parameters and
    # metrics, the rest in settings.json and scores.csv; a Model 2 run's names.
    # C3b: the study's seed a parameter, a cohort's validation score under the
    # objective a metric (its other scores not), Optuna's two figures and the
    # study_folder tag. The experiment is created first: MLflow would create it
    # at its default artifact location, from the repository's root the canonical
    # one.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(tmp_path)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    experiment_id = client.create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((tmp_path / "artifacts").resolve()),
    )
    study = optuna.create_study(study_name="a_study", sampler=RandomSampler(seed=0))
    study.optimize(
        lambda trial: trial.suggest_float("total__alpha", 1e-4, 1.0, log=True),
        n_trials=2,
    )
    studies_dir = tmp_path / "studies"
    studies_dir.mkdir()
    (studies_dir / "a_study.log").write_text("the study's journal\n", encoding="utf-8")
    # Values with no short decimal form: a rounded scores.csv fails the read-back.
    scores = pd.DataFrame(
        {"score": [1 / 3, 1 / 13, 2 / 7, 2 / 17, 3 / 11, 4 / 13, 5 / 19]},
        index=pd.MultiIndex.from_tuples(
            [
                ("train", "composite", "rmse"),
                ("train", "composite", "mae"),
                ("val", "composite", "rmse"),
                ("val", "composite", "mae"),
                ("val", "n_kindergarten", "rmse"),
                ("val", "n_kindergarten", "mae"),
                ("train", "n_kindergarten", "rmse"),
            ],
            names=["dataset", "cohort", "metric"],
        ),
    )
    # The notebook's own settings: five of the 23 are parameters.
    shared_params = notebook_defs["shared_params"]
    tuning_run = {
        "study": study,
        "study_seed": 7,
        "cv_best": study.best_value,
        "scores": scores,
        "seconds": 1.0,
        "n_complete": 2,
        "n_failed": 0,
        "components": [Bunch(name="total", to_dict=lambda: {"family": "poisson_glm"})],
    }
    code_tags = {
        "notebook": "notebooks/03_hyperparameter_tuning.py",
        "git_commit": "0" * 40,
        "git_dirty": "false",
    }
    # Floats: an integer column makes MLflow's schema inference warn.
    dataset = mlflow.data.from_pandas(
        pd.DataFrame({"x": [0.5, 1.5], "y": [2.0, 3.0]}), name="train", digest="d" * 36
    )
    arguments = (
        tuning_run,
        "independent_cohorts",
        {
            "family": "poisson_glm",
            "use_exposure": True,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        studies_dir,
        "rmse",
        shared_params,
        {"simulation.n_neighborhoods": 150},
        {"training": dataset},
        code_tags,
    )
    # A Model 2 run: its run settings name the two families.
    model2_study = optuna.create_study(
        study_name="b_study", sampler=RandomSampler(seed=1)
    )
    model2_study.optimize(
        lambda trial: (
            trial.suggest_float("total__alpha", 1e-4, 1.0, log=True)
            + trial.suggest_float("probability__C", 1e-2, 10.0, log=True)
        ),
        n_trials=2,
    )
    (studies_dir / "b_study.log").write_text("the study's journal\n", encoding="utf-8")
    model2_arguments = (
        {**tuning_run, "study": model2_study, "cv_best": model2_study.best_value},
        "total_times_probability",
        {
            "total_family": "poisson_glm",
            "probability_family": "logistic",
            "use_exposure": False,
        },
        *arguments[3:],
    )
    with pytest.MonkeyPatch.context() as patch:
        for name, value in environment.items():
            patch.setenv(name, value)
        run_id = notebook_defs["log_tuning_run"](*arguments)
        assert run_id is not None
        assert notebook_defs["log_tuning_run"](*arguments) is None
        model2_run_id = notebook_defs["log_tuning_run"](*model2_arguments)
    kept_settings = {
        name: str(shared_params[name])
        for name in (
            "config_hash",
            "seed",
            "split_method",
            "n_trials",
            "objective_metric",
        )
    }
    kept_settings["study_seed"] = "7"
    model2_run = client.get_run(model2_run_id)
    assert model2_run.info.run_name == (
        "total_times_probability poisson_glm+logistic exposure=False"
    )
    assert model2_run.data.params == {
        "total_family": "poisson_glm",
        "probability_family": "logistic",
        "use_exposure": "False",
        **kept_settings,
        "best.total__alpha": str(model2_study.best_params["total__alpha"]),
        "best.probability__C": str(model2_study.best_params["probability__C"]),
    }
    assert len(model2_run.data.params) == 9 + 2
    runs = client.search_runs([experiment_id], "tags.run_key = 'studies/a_study'")
    assert [run.info.run_id for run in runs] == [run_id]
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "independent_cohorts poisson_glm exposure=True"
    assert {
        name: value
        for name, value in run.data.tags.items()
        if not name.startswith("mlflow.")
    } == {
        "run_key": "studies/a_study",
        "study_folder": "studies",
        "stage": "tuning",
        "model": "independent_cohorts",
        "family": "poisson_glm",
        **code_tags,
    }
    assert run.data.params == {
        "family": "poisson_glm",
        "use_exposure": "True",
        "share_hyperparams": "True",
        "share_feature_transformer": "False",
        **kept_settings,
        "best.total__alpha": str(study.best_params["total__alpha"]),
    }
    assert len(run.data.params) == 10 + 1
    # The composite's scores and a cohort's validation score under the objective:
    # the cohort's other scores are in scores.csv.
    assert run.data.metrics == {
        "cv/rmse": study.best_value,
        "train/composite/rmse": 1 / 3,
        "train/composite/mae": 1 / 13,
        "val/composite/rmse": 2 / 7,
        "val/composite/mae": 2 / 17,
        "val/n_kindergarten/rmse": 3 / 11,
        "seconds": 1.0,
        "n_complete": 2.0,
        "n_failed": 0.0,
    }
    assert [
        (
            item.dataset.name,
            item.dataset.digest,
            {tag.key: tag.value for tag in item.tags},
        )
        for item in run.inputs.dataset_inputs
    ] == [("train", "d" * 36, {"mlflow.data.context": "training"})]
    artifacts = Path(run.info.artifact_uri)
    assert sorted(
        str(path.relative_to(artifacts))
        for path in artifacts.rglob("*")
        if path.is_file()
    ) == [
        "best_params.json",
        "components.json",
        "config.json",
        "figures/optimization_history.html",
        "figures/param_importances.html",
        "figures/slice.html",
        "scores.csv",
        "search_space.json",
        "settings.json",
        "study/a_study.log",
        "trials.csv",
    ]
    assert json.loads((artifacts / "best_params.json").read_text()) == study.best_params
    assert json.loads((artifacts / "settings.json").read_text()) == {
        **shared_params,
        "study_seed": 7,
    }
    pd.testing.assert_frame_equal(
        pd.read_csv(artifacts / "scores.csv", float_precision="round_trip"),
        scores.reset_index(),
        check_exact=True,
    )
    assert len(pd.read_csv(artifacts / "trials.csv")) == 2
    # M1's concern: no run is left open and the store is not left as MLflow's
    # tracking URI for the tests that run after.
    assert mlflow.active_run() is None
    assert mlflow.get_tracking_uri() != environment["MLFLOW_TRACKING_URI"]


def test_a_tuning_run_that_fails_while_logging_holds_no_metrics(
    notebook_defs: dict, tmp_path: Path
) -> None:
    # C3c (TUNING_NOTEBOOK_REVISION_PLAN.md, Task C, "Second review"): the
    # metrics are logged last, so a run that fails while logging (here its
    # journal missing) is FAILED with no metrics and never shows in a chart; it
    # is logged again once the journal is there. The parameter importances are
    # seeded with the study's seed, so the two runs' figures equal the one drawn
    # here with seed 7, but for the div id plotly draws at random. Three
    # parameters: fANOVA gives a single one an importance of 1 whatever the seed.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(tmp_path)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    experiment_id = client.create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((tmp_path / "artifacts").resolve()),
    )
    study = optuna.create_study(study_name="c_study", sampler=RandomSampler(seed=0))
    study.optimize(
        lambda trial: (
            trial.suggest_float("total__alpha", 1e-4, 1.0, log=True)
            + trial.suggest_float("total__l1_ratio", 0.0, 1.0)
            + (trial.suggest_categorical("total__transformer", ["a", "b"]) == "a")
        ),
        n_trials=5,
    )
    studies_dir = tmp_path / "studies"
    studies_dir.mkdir()
    tuning_run = {
        "study": study,
        "study_seed": 7,
        "cv_best": study.best_value,
        "scores": pd.DataFrame(
            {"score": [0.5]},
            index=pd.MultiIndex.from_tuples(
                [("val", "composite", "rmse")], names=["dataset", "cohort", "metric"]
            ),
        ),
        "seconds": 1.0,
        "n_complete": 5,
        "n_failed": 0,
        "components": [],
    }
    arguments = (
        tuning_run,
        "total_times_probability",
        {
            "total_family": "poisson_glm",
            "probability_family": "logistic",
            "use_exposure": False,
        },
        studies_dir,
        "rmse",
        notebook_defs["shared_params"],
        {},
        {},
        {},
    )
    with pytest.MonkeyPatch.context() as patch:
        for name, value in environment.items():
            patch.setenv(name, value)
        with pytest.raises(FileNotFoundError):
            notebook_defs["log_tuning_run"](*arguments)
        (studies_dir / "c_study.log").write_text(
            "the study's journal\n", encoding="utf-8"
        )
        run_id = notebook_defs["log_tuning_run"](*arguments)
    (failed_run,) = client.search_runs([experiment_id], "attributes.status = 'FAILED'")
    finished_run = client.get_run(run_id)
    assert failed_run.info.run_id != run_id
    assert failed_run.data.metrics == {}
    assert finished_run.info.status == "FINISHED"
    assert finished_run.data.metrics["val/composite/rmse"] == 0.5
    # As log_figure writes a plotly figure (include_plotlyjs="cdn").
    expected = optuna.visualization.plot_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=7)
    ).to_html(include_plotlyjs="cdn")
    figures = [
        re.sub(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", "<id>", html)
        for html in (
            expected,
            *(
                (
                    Path(run.info.artifact_uri) / "figures" / "param_importances.html"
                ).read_text(encoding="utf-8")
                for run in (failed_run, finished_run)
            ),
        )
    ]
    assert "total__l1_ratio" in figures[0]
    assert figures[1] == figures[0]
    assert figures[2] == figures[0]
    assert mlflow.active_run() is None


@pytest.mark.parametrize(
    ("n_trials", "objective"),
    [
        (1, lambda trial: trial.suggest_float("total__alpha", 1e-4, 1.0, log=True)),
        (2, lambda trial: trial.suggest_float("total__alpha", 1e-4, 1.0) * 0 + 1.0),
    ],
    ids=["one_complete_trial", "equal_values"],
)
def test_a_study_fanova_cannot_rank_is_logged_without_the_importances_figure(
    notebook_defs: dict, tmp_path: Path, n_trials: int, objective
) -> None:
    # C3c, the review's finding: fANOVA raises on a single complete trial or
    # on equal values, which would fail the run on every rerun (and refuse the
    # evaluation run). Such a study is logged without the importances figure.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(tmp_path)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    client.create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((tmp_path / "artifacts").resolve()),
    )
    study = optuna.create_study(study_name="d_study", sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    studies_dir = tmp_path / "studies"
    studies_dir.mkdir()
    (studies_dir / "d_study.log").write_text("the study's journal\n", encoding="utf-8")
    tuning_run = {
        "study": study,
        "study_seed": 7,
        "cv_best": study.best_value,
        "scores": pd.DataFrame(
            {"score": [0.5]},
            index=pd.MultiIndex.from_tuples(
                [("val", "composite", "rmse")], names=["dataset", "cohort", "metric"]
            ),
        ),
        "seconds": 1.0,
        "n_complete": n_trials,
        "n_failed": 0,
        "components": [],
    }
    with pytest.MonkeyPatch.context() as patch:
        for name, value in environment.items():
            patch.setenv(name, value)
        run_id = notebook_defs["log_tuning_run"](
            tuning_run,
            "independent_cohorts",
            {"family": "poisson_glm", "use_exposure": True},
            studies_dir,
            "rmse",
            notebook_defs["shared_params"],
            {},
            {},
            {},
        )
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert sorted(
        path.name for path in (Path(run.info.artifact_uri) / "figures").iterdir()
    ) == ["optimization_history.html", "slice.html"]
    assert mlflow.active_run() is None


def test_an_evaluation_run_logs_the_composite_and_each_best_runs_files(
    notebook_defs: dict, tmp_path: Path
) -> None:
    # The second review's G3: log_evaluation_run called on a scratch store
    # with two best runs and small frames. The composite rows (each model's
    # and the paired difference's) become metrics, a cohort's row does not;
    # each best run's study names a tag and gives its best_params file. The
    # cold read's T2: a dataset input and a figure. C1
    # (TUNING_NOTEBOOK_REVISION_PLAN.md, Task C): six settings as parameters,
    # all of them in settings.json; the score and its interval as metrics.
    # C3b: the study_folder tag, and each best run's FINISHED tuning run (found
    # by its run key, a FAILED one with the same key passed over) as a tag.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(tmp_path)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    experiment_id = client.create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((tmp_path / "artifacts").resolve()),
    )
    tuning_run_ids = {}
    for model in ("independent_cohorts", "total_times_probability"):
        tuning_run = client.create_run(
            experiment_id, tags={"run_key": f"studies/{model}_study"}
        )
        client.set_terminated(tuning_run.info.run_id)
        tuning_run_ids[model] = tuning_run.info.run_id
    # Started later, so a search without the status filter finds it first.
    failed_run = client.create_run(
        experiment_id,
        start_time=tuning_run.info.start_time + 1000,
        tags={"run_key": "studies/independent_cohorts_study"},
    )
    client.set_terminated(failed_run.info.run_id, status="FAILED")
    best_runs = {}
    for model in ("independent_cohorts", "total_times_probability"):
        study = optuna.create_study(
            study_name=f"{model}_study", sampler=RandomSampler(seed=0)
        )
        study.optimize(
            lambda trial: trial.suggest_float("total__alpha", 1e-4, 1.0, log=True),
            n_trials=1,
        )
        best_runs[model] = {
            "study": study,
            "components": [
                Bunch(name="total", to_dict=lambda: {"family": "poisson_glm"})
            ],
        }
    bound_names = notebook_defs["CI_BOUND_NAMES"]
    # The tables' statistics; the metrics keep the score and the two bounds.
    statistics = [
        "score",
        "mean",
        "median",
        "sd",
        bound_names["lower"],
        bound_names["upper"],
    ]
    kept_statistics = {"score", bound_names["lower"], bound_names["upper"]}
    # The final run's settings: the notebook's and its own n_train_val.
    settings = {**notebook_defs["shared_params"], "n_train_val": 1223}
    scores_df = pd.DataFrame(
        [
            {
                "model": model,
                "cohort": cohort,
                "metric": "rmse",
                **{statistic: value + i for i, statistic in enumerate(statistics)},
            }
            for model, cohort, value in (
                ("independent_cohorts", "composite", 4.0),
                ("total_times_probability", "composite", 4.5),
                ("independent_cohorts", "n_elementary", 3.0),
            )
        ]
    )
    difference_df = pd.DataFrame(
        [
            {
                "model": "independent_cohorts - total_times_probability",
                "cohort": "composite",
                "metric": "rmse",
                **{statistic: -0.5 + i for i, statistic in enumerate(statistics)},
            }
        ]
    )
    # Floats: an integer column makes MLflow's schema inference warn.
    dataset = mlflow.data.from_pandas(
        pd.DataFrame({"x": [0.5, 1.5], "y": [2.0, 3.0]}), name="val", digest="d" * 36
    )
    with pytest.MonkeyPatch.context() as patch:
        for name, value in environment.items():
            patch.setenv(name, value)
        n_metrics = notebook_defs["log_evaluation_run"](
            "best runs",
            {"run_key": "studies/best_runs", "stage": "best_runs"},
            settings,
            "val",
            scores_df,
            difference_df,
            bound_names,
            {"scores": scores_df},
            {"figures/a_figure.html": go.Figure()},
            best_runs,
            "studies",
            {"validation": dataset},
        )
    assert n_metrics == 9
    (run,) = client.search_runs([experiment_id], "tags.stage = 'best_runs'")
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "best runs"
    assert run.data.metrics == {
        f"val/{name}/rmse/{statistic}": value + i
        for name, value in (
            ("independent_cohorts", 4.0),
            ("total_times_probability", 4.5),
            ("difference", -0.5),
        )
        for i, statistic in enumerate(statistics)
        if statistic in kept_statistics
    }
    assert {
        name: value
        for name, value in run.data.tags.items()
        if not name.startswith("mlflow.")
    } == {
        "run_key": "studies/best_runs",
        "stage": "best_runs",
        "study_folder": "studies",
        "best.independent_cohorts": "independent_cohorts_study",
        "best.total_times_probability": "total_times_probability_study",
        "best.independent_cohorts.run_id": tuning_run_ids["independent_cohorts"],
        "best.total_times_probability.run_id": tuning_run_ids[
            "total_times_probability"
        ],
    }
    assert run.data.params == {
        name: str(settings[name])
        for name in (
            "config_hash",
            "seed",
            "split_method",
            "n_trials",
            "objective_metric",
            "selection_metric",
        )
    }
    assert [
        (
            item.dataset.name,
            item.dataset.digest,
            {tag.key: tag.value for tag in item.tags},
        )
        for item in run.inputs.dataset_inputs
    ] == [("val", "d" * 36, {"mlflow.data.context": "validation"})]
    artifacts = Path(run.info.artifact_uri)
    assert json.loads((artifacts / "settings.json").read_text()) == settings
    assert (artifacts / "tables" / "scores.csv").read_text() == scores_df.to_csv(
        index=False, float_format="%.6g"
    )
    assert (artifacts / "figures" / "a_figure.html").is_file()
    for model_name, best_run in best_runs.items():
        assert (
            json.loads((artifacts / "best_params" / f"{model_name}.json").read_text())
            == best_run["study"].best_params
        )
        assert json.loads(
            (artifacts / "components" / f"{model_name}.json").read_text()
        ) == {"total": {"family": "poisson_glm"}}
    assert mlflow.active_run() is None
    assert mlflow.get_tracking_uri() != environment["MLFLOW_TRACKING_URI"]


def test_an_evaluation_run_without_its_tuning_run_is_refused(
    notebook_defs: dict, tmp_path: Path
) -> None:
    # C3b: a best run whose FINISHED tuning run is not in the store (an
    # inconsistent store) raises a ValueError naming its run key, before any
    # evaluation run is opened.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(tmp_path)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    experiment_id = client.create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((tmp_path / "artifacts").resolve()),
    )
    tuning_run = client.create_run(
        experiment_id, tags={"run_key": "studies/independent_cohorts_study"}
    )
    client.set_terminated(tuning_run.info.run_id)
    best_runs = {
        model: {"study": Bunch(study_name=f"{model}_study"), "components": []}
        for model in ("independent_cohorts", "total_times_probability")
    }
    no_rows = pd.DataFrame(columns=["model", "cohort", "metric", "score"])
    with pytest.MonkeyPatch.context() as patch:
        for name, value in environment.items():
            patch.setenv(name, value)
        with pytest.raises(ValueError, match="'studies/total_times_probability_study'"):
            notebook_defs["log_evaluation_run"](
                "best runs",
                {"run_key": "studies/best_runs", "stage": "best_runs"},
                dict(notebook_defs["shared_params"]),
                "val",
                no_rows,
                no_rows,
                notebook_defs["CI_BOUND_NAMES"],
                {},
                {},
                best_runs,
                "studies",
                {},
            )
    assert [run.info.run_id for run in client.search_runs([experiment_id])] == [
        tuning_run.info.run_id
    ]
    assert mlflow.active_run() is None


def test_an_evaluation_run_that_fails_while_logging_holds_no_metrics(
    notebook_defs: dict, tmp_path: Path
) -> None:
    # C3c: the metrics are logged last, so an evaluation run that fails while
    # logging (here a figure log_figure cannot write) is FAILED with no metrics
    # and never shows in a chart.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(tmp_path)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    experiment_id = client.create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((tmp_path / "artifacts").resolve()),
    )
    best_runs = {}
    for model in ("independent_cohorts", "total_times_probability"):
        tuning_run = client.create_run(
            experiment_id, tags={"run_key": f"studies/{model}_study"}
        )
        client.set_terminated(tuning_run.info.run_id)
        best_runs[model] = {
            "study": Bunch(study_name=f"{model}_study", best_params={}),
            "components": [],
        }
    bound_names = notebook_defs["CI_BOUND_NAMES"]
    # A composite row per model: six metrics the run would log.
    scores_df = pd.DataFrame(
        [
            {
                "model": model,
                "cohort": "composite",
                "metric": "rmse",
                "score": 4.0,
                bound_names["lower"]: 3.5,
                bound_names["upper"]: 4.5,
            }
            for model in best_runs
        ]
    )
    with pytest.MonkeyPatch.context() as patch:
        for name, value in environment.items():
            patch.setenv(name, value)
        with pytest.raises(TypeError, match="Unsupported figure object type"):
            notebook_defs["log_evaluation_run"](
                "best runs",
                {"run_key": "studies/best_runs", "stage": "best_runs"},
                dict(notebook_defs["shared_params"]),
                "val",
                scores_df,
                scores_df.iloc[:0],
                bound_names,
                {},
                {"figures/a_figure.html": object()},
                best_runs,
                "studies",
                {},
            )
    (run,) = client.search_runs([experiment_id], "tags.stage = 'best_runs'")
    assert run.info.status == "FAILED"
    assert run.data.metrics == {}
    assert mlflow.active_run() is None


SHARING = [(False, False), (True, False), (True, True)]


def _count_components(notebook_defs: dict, params: dict, **settings) -> tuple:
    """The components ``CountComponent.from_params`` reads back from ``params``, one per cohort."""
    return tuple(
        notebook_defs["CountComponent"].from_params(params, name=cohort, **settings)
        for cohort in notebook_defs["COHORTS"]
    )


@pytest.mark.parametrize(("share_hyperparams", "share_feature_transformer"), SHARING)
def test_a_shared_choice_is_one_draw_in_every_cohorts_component(
    notebook_defs: dict, share_hyperparams: bool, share_feature_transformer: bool
) -> None:
    # Independent: each of the three cohorts draws LightGBM's nine hyperparameters
    # and its transformer, 30 parameters. Shared hyperparameters: one draw under
    # `shared__` in every component, each cohort still drawing its transformer,
    # 12. Both shared: 10, one transformer in every component. The package's
    # objective draws the components on the notebook's cohorts and candidates;
    # they are read back from the trial's params by the component type, as the
    # best run's are (D25, D26).
    trial = optuna.create_study(sampler=RandomSampler(seed=0)).ask()
    cohorts = notebook_defs["COHORTS"]
    settings = {
        "family": "lgbm_poisson",
        "use_exposure": True,
        "feature_transformer_dict": notebook_defs["COUNT_FEATURE_TRANSFORMERS"][
            "lgbm_poisson"
        ],
    }
    drawn = suggest_independent_cohorts_components(
        trial,
        cohorts=cohorts,
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        **settings,
    )
    names = sorted(trial.params)
    shared_names = [name for name in names if name.startswith("shared__")]
    transformer_names = [
        name for name in names if name.endswith("__feature_transformer")
    ]
    components = _count_components(notebook_defs, trial.params, **settings)
    assert components == drawn
    estimator_params = {component.estimator_params for component in components}
    transformers = {id(component.feature_transformer) for component in components}
    if share_feature_transformer:
        assert transformer_names == ["shared__feature_transformer"]
        assert len(transformers) == 1
    else:
        assert transformer_names == sorted(
            f"{cohort}__feature_transformer" for cohort in cohorts
        )
    if share_hyperparams:
        assert len(names) == (10 if share_feature_transformer else 12)
        assert len(shared_names) == (10 if share_feature_transformer else 9)
        assert len(estimator_params) == 1
    else:
        assert len(names) == 30
        assert shared_names == []
        assert all(name.split("__")[0] in cohorts for name in names)
    # The record of a run: plain data, the transformer by its declaration.
    for component in components:
        record = json.loads(json.dumps(component.to_dict()))
        assert record["feature_transformer"]["column_plans"]


@pytest.mark.parametrize(("share_hyperparams", "share_feature_transformer"), SHARING)
def test_the_best_components_are_read_back_from_the_trials_params(
    notebook_defs: dict, share_hyperparams: bool, share_feature_transformer: bool
) -> None:
    # The refit after a study reads the best components from best_params by
    # the component type (D25): they must be the components the objective drew
    # from the same trial, for either model; the build is the same for both.
    trial = optuna.create_study(sampler=RandomSampler(seed=1)).ask()
    settings = {
        "family": "nb2",
        "use_exposure": False,
        "feature_transformer_dict": notebook_defs["COUNT_FEATURE_TRANSFORMERS"]["nb2"],
    }
    drawn = suggest_independent_cohorts_components(
        trial,
        cohorts=notebook_defs["COHORTS"],
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        **settings,
    )
    assert _count_components(notebook_defs, trial.params, **settings) == drawn

    trial = optuna.create_study(sampler=RandomSampler(seed=2)).ask()
    count_transformers = notebook_defs["COUNT_FEATURE_TRANSFORMERS"]["lgbm_gaussian"]
    probability_transformers = notebook_defs["PROBABILITY_FEATURE_TRANSFORMERS"]["lgbm"]
    drawn = suggest_total_times_probability_components(
        trial,
        total_family="lgbm_gaussian",
        probability_family="lgbm",
        use_exposure=True,
        total_feature_transformer_dict=count_transformers,
        probability_feature_transformer_dict=probability_transformers,
    )
    # The two parts share LightGBM's parameter names; each reads its own.
    total = notebook_defs["CountComponent"].from_params(
        trial.params,
        name="total",
        family="lgbm_gaussian",
        use_exposure=True,
        feature_transformer_dict=count_transformers,
    )
    probability = notebook_defs["ProbabilityComponent"].from_params(
        trial.params,
        name="probability",
        family="lgbm",
        feature_transformer_dict=probability_transformers,
    )
    assert (total, probability) == drawn
    assert total.estimator_params.num_leaves == trial.params["total__num_leaves"]
    assert (
        probability.estimator_params.num_leaves
        == trial.params["probability__num_leaves"]
    )


def test_a_missing_entry_in_the_params_is_named(notebook_defs: dict) -> None:
    # The record is filled by its fields, so a params mapping without one of
    # them fails at the refit's read-back, naming the field.
    with pytest.raises(KeyError, match="dispersion"):
        notebook_defs["CountComponent"].from_params(
            {},
            name="total",
            family="nb2",
            use_exposure=True,
            feature_transformer_dict=notebook_defs["COUNT_FEATURE_TRANSFORMERS"]["nb2"],
        )


def test_a_study_is_named_by_the_settings_that_shape_it(notebook_defs: dict) -> None:
    # A stored study is loaded by its name alone, so a setting left out of the
    # directory's or the file's name would load a stale study after a change.
    # The file's name holds the objective, the tuner's (D42).
    assert notebook_defs["tuner"].objective_metric is notebook_defs["OBJECTIVE_METRIC"]
    assert notebook_defs["OBJECTIVE_METRIC"].name == "rmse"
    assert notebook_defs["tuner"].storage_dir == (
        PROJECT_ROOT
        / "studies"
        / (
            f"data={_canonical_hash()[:8]}__split_method=random__test_size=0.2__"
            "val_size=0.2__n_splits=5__seed=42__n_trials=100__n_startup_trials=10"
        )
    )
    assert notebook_defs["build_study_name"](
        "independent_cohorts",
        notebook_defs["MODEL1_RUN_SETTINGS"][0],
        "poisson_deviance",
        42,
    ) == (
        "independent_cohorts__family=lgbm_poisson__use_exposure=True__"
        "share_hyperparams=True__share_feature_transformer=False__"
        "objective=poisson_deviance__seed=42"
    )


def test_the_run_settings_are_the_ones_the_baselines_were_recorded_with(
    notebook_defs: dict,
) -> None:
    # Each run is seeded seed + i in its list's order, and Check C's baselines
    # were recorded with these entries: a reordered, dropped or flipped entry
    # changes the seeded results, and only a clicked run would show it.
    arms = [
        (family, use_exposure)
        for family in ("lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2")
        for use_exposure in (True, False)
    ]
    assert notebook_defs["MODEL1_RUN_SETTINGS"] == [
        {
            "family": family,
            "use_exposure": use_exposure,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        }
        for family, use_exposure in arms
    ]
    # The total times probability model's runs: a family pair × exposure, the
    # first the objective EDA's joint run (seed + 0), so its study is shared.
    assert notebook_defs["MODEL2_RUN_SETTINGS"] == [
        {
            "total_family": total_family,
            "probability_family": probability_family,
            "use_exposure": use_exposure,
        }
        for total_family in ("lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2")
        for probability_family in ("logistic", "lgbm")
        for use_exposure in (True, False)
    ]


@pytest.fixture(scope="module")
def objective_eda_defs() -> dict:
    """The names a script run of the objective EDA notebook defines; nothing is fitted."""
    spec = importlib.util.spec_from_file_location(
        "objective_eda_notebook", OBJECTIVE_EDA_PATH
    )
    assert spec is not None and spec.loader is not None
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    _, defs = notebook.app.run()
    return dict(defs)


def test_the_two_notebooks_share_one_setup(
    notebook_defs: dict, objective_eda_defs: dict
) -> None:
    # D41: both notebooks copy the setup, and a study either one stores is
    # loaded by its name alone, which holds the settings but not the data,
    # the split, the candidates or the tuner's design. A copy that drifts
    # would load a study tuned on the other copy's setup, silently: the two
    # must build the same objects.
    tuning, eda = notebook_defs, objective_eda_defs
    for name in (
        "DATA_SETTINGS",
        "config_hash",
        "seed",
        "split_method",
        "test_size",
        "val_size",
        "n_splits",
        "n_trials",
        "n_startup_trials",
        "objective_metric",
        "COHORTS",
        "PREDICTION_FLOOR",
        "cv_groups",
    ):
        assert tuning[name] == eda[name], name
    for dataset_name in ("train_age_group", "val_age_group", "test_age_group"):
        for field, tuning_values, eda_values in zip(
            tuning[dataset_name]._fields,
            tuning[dataset_name],
            eda[dataset_name],
            strict=True,
        ):
            assert tuning_values.equals(eda_values), (dataset_name, field)
    assert repr(tuning["cv"]) == repr(eda["cv"])
    for name in ("COUNT_FEATURE_TRANSFORMERS", "PROBABILITY_FEATURE_TRANSFORMERS"):
        assert {
            family: {key: t.to_dict() for key, t in candidates.items()}
            for family, candidates in tuning[name].items()
        } == {
            family: {key: t.to_dict() for key, t in candidates.items()}
            for family, candidates in eda[name].items()
        }, name
    assert [m.name for m in tuning["METRICS"]] == [m.name for m in eda["METRICS"]]
    assert tuning["OBJECTIVE_METRIC"].name == eda["OBJECTIVE_METRIC"].name
    tuning_tuner, eda_tuner = tuning["tuner"], eda["tuner"]
    assert tuning_tuner.storage_dir == eda_tuner.storage_dir
    assert tuning_tuner.objective_metric.name == eda_tuner.objective_metric.name
    assert [m.name for m in tuning_tuner.recorded_metrics] == [
        m.name for m in eda_tuner.recorded_metrics
    ]
    for field in ("n_trials", "n_startup_trials", "n_jobs"):
        assert getattr(tuning_tuner, field) == getattr(eda_tuner, field), field
    study_names = [
        (
            "independent_cohorts",
            {
                "family": "lgbm_poisson",
                "use_exposure": True,
                "share_hyperparams": True,
                "share_feature_transformer": False,
            },
            "rmse",
            42,
        ),
        (
            "total_times_probability_joint",
            {
                "total_family": "lgbm_poisson",
                "probability_family": "logistic",
                "use_exposure": True,
            },
            "rmse",
            42,
        ),
    ]
    for arguments in study_names:
        assert tuning["build_study_name"](*arguments) == eda["build_study_name"](
            *arguments
        )
    # F4: the transformer's name and the hyperparameters are read by one pair
    # of functions, copied in both notebooks: the same results on the same
    # components (a cohort's count component, and a joint draw's total and
    # probabilities, the classifier with its calibration).
    trial = optuna.create_study(sampler=RandomSampler(seed=3)).ask()
    count_transformers = tuning["COUNT_FEATURE_TRANSFORMERS"]["lgbm_gaussian"]
    probability_transformers = tuning["PROBABILITY_FEATURE_TRANSFORMERS"]["lgbm"]
    components = suggest_total_times_probability_components(
        trial,
        total_family="lgbm_gaussian",
        probability_family="lgbm",
        use_exposure=True,
        total_feature_transformer_dict=count_transformers,
        probability_feature_transformer_dict=probability_transformers,
    )
    trial = optuna.create_study(sampler=RandomSampler(seed=4)).ask()
    nb2_transformers = tuning["COUNT_FEATURE_TRANSFORMERS"]["nb2"]
    components += suggest_independent_cohorts_components(
        trial,
        family="nb2",
        use_exposure=False,
        cohorts=tuning["COHORTS"],
        feature_transformer_dict=nb2_transformers,
        share_hyperparams=False,
        share_feature_transformer=False,
    )
    candidates = (count_transformers, probability_transformers) + (
        nb2_transformers,
    ) * len(tuning["COHORTS"])
    for component, component_candidates in zip(components, candidates, strict=True):
        assert tuning["find_feature_transformer_name"](
            component, component_candidates
        ) == eda["find_feature_transformer_name"](component, component_candidates)
        assert tuning["read_hyperparameters"](component) == eda["read_hyperparameters"](
            component
        )
    assert "calibration_method" in tuning["read_hyperparameters"](components[1])


def test_the_best_runs_tables_show_the_composite_and_every_metric_by_default(
    notebook_defs: dict,
) -> None:
    # 7.0b: one cohort and one metric choice filter both models' best-run
    # tables; a click must open on the composite with every metric shown.
    cohort = notebook_defs["best_runs_cohort"]
    metrics = notebook_defs["best_runs_metrics"]
    metric_names = [metric.name for metric in notebook_defs["METRICS"]]
    assert list(cohort.options) == [*notebook_defs["COHORTS"], "composite"]
    assert cohort.value == "composite"
    assert list(metrics.options) == metric_names
    assert list(metrics.value) == metric_names


def test_the_forest_figure_draws_a_row_per_cohort_and_a_trace_per_model(
    notebook_defs: dict,
) -> None:
    # 5.4a: a row per cohort, the composite first; in each row a dot per model
    # at its score, the bar its interval; one legend entry per model; the
    # rule in every row. The rows of another metric are not drawn.
    forest_figure = notebook_defs["forest_figure"]
    palette = notebook_defs["PALETTE"]
    scores = pd.DataFrame(
        [
            {"model": model, "cohort": cohort, "metric": metric}
            for metric in ("rmse", "mae")
            for model in ("model_a", "model_b")
            for cohort in ("cohort_x", "cohort_y", "composite")
        ]
    )
    scores["score"] = range(len(scores))
    scores["lower"] = scores["score"] - 0.5
    scores["upper"] = scores["score"] + 1.0
    figure = forest_figure(scores, "rmse", "lower", "upper", palette)
    cohorts = ["composite", "cohort_x", "cohort_y"]
    assert [annotation.text for annotation in figure.layout.annotations] == cohorts
    assert [trace.name for trace in figure.data if trace.showlegend] == [
        "model_a",
        "model_b",
    ]
    assert figure.layout.showlegend
    rmse = scores[scores["metric"] == "rmse"].set_index(["cohort", "model"])
    traces = iter(figure.data)
    for row, cohort in enumerate(cohorts, start=1):
        for position, model in enumerate(["model_a", "model_b"]):
            trace = next(traces)
            axis = "" if row == 1 else str(row)
            assert (trace.name, trace.xaxis, trace.yaxis) == (
                model,
                f"x{axis}",
                f"y{axis}",
            )
            assert list(trace.y) == [position]
            assert list(trace.x) == [rmse.loc[(cohort, model), "score"]]
            assert list(trace.error_x.array) == [1.0]
            assert list(trace.error_x.arrayminus) == [0.5]
    assert next(traces, None) is None
    assert not figure.layout.shapes
    ruled = forest_figure(
        scores, "rmse", "lower", "upper", palette, share_x_axis=True, rule_at=0
    )
    assert len(ruled.layout.shapes) == 3
    assert all(shape.x0 == shape.x1 == 0 for shape in ruled.layout.shapes)
    one_model = forest_figure(
        scores[scores["model"] == "model_a"], "rmse", "lower", "upper", palette
    )
    assert not one_model.layout.showlegend


def test_the_ridge_draws_a_density_per_model_in_a_row_per_cohort(
    notebook_defs: dict,
) -> None:
    # 5.4a: a row per cohort, the composite first; in each row each model's
    # replicates as a density (a Gaussian KDE between their extremes), the
    # row's highest density 1; no dots or bars; one legend entry per model;
    # the rule in every row. The columns of another metric are not drawn.
    ridge_figure = notebook_defs["ridge_figure"]
    palette = notebook_defs["PALETTE"]
    models = ["model_a", "model_b"]
    columns = pd.MultiIndex.from_product(
        [["cohort_x", "cohort_y", "composite"], ["rmse", "mae"]],
        names=["cohort", "metric"],
    )
    rng = np.random.default_rng(0)
    replicate_scores = {
        model: pd.DataFrame(rng.normal(4.5, 0.3, (200, len(columns))), columns=columns)
        for model in models
    }
    figure = ridge_figure(replicate_scores, "rmse", palette)
    cohorts = ["composite", "cohort_x", "cohort_y"]
    assert [annotation.text for annotation in figure.layout.annotations] == cohorts
    assert len(figure.data) == 6 and all(trace.mode == "lines" for trace in figure.data)
    assert [trace.name for trace in figure.data if trace.showlegend] == models
    for row, cohort in enumerate(cohorts):
        curves = figure.data[2 * row : 2 * row + 2]
        densities = {}
        for model in models:
            replicates = replicate_scores[model][(cohort, "rmse")]
            x_values = np.linspace(replicates.min(), replicates.max(), 200)
            densities[model] = (x_values, gaussian_kde(replicates)(x_values))
        highest = max(density.max() for _, density in densities.values())
        for model, curve in zip(models, curves, strict=True):
            x_values, density = densities[model]
            axis = "" if row == 0 else str(row + 1)
            assert (curve.name, curve.xaxis) == (model, f"x{axis}")
            np.testing.assert_array_equal(curve.x, x_values)
            np.testing.assert_allclose(curve.y, density / highest)
        assert max(max(curve.y) for curve in curves) == pytest.approx(1.0)
    assert not figure.layout.shapes
    ruled = ridge_figure(
        replicate_scores, "rmse", palette, share_x_axis=True, rule_at=0
    )
    assert len(ruled.layout.shapes) == 3
    assert all(shape.x0 == shape.x1 == 0 for shape in ruled.layout.shapes)


def test_the_plots_open_on_their_forms_and_the_selection_metric(
    notebook_defs: dict,
) -> None:
    # 5.4b: the best runs' plot opens on the forest, the paired difference's
    # on the ridge; both metric menus on the selection metric. S4: both offer
    # all six metrics; the runs' menu only draws (the permutation importance
    # scores the selection metric).
    assert notebook_defs["best_runs_plot"].value == "forest"
    assert notebook_defs["best_difference_plot"].value == "ridge"
    selection = notebook_defs["SELECTION_METRIC"].name
    metric_names = [metric.name for metric in notebook_defs["METRICS"]]
    runs_metric = notebook_defs["best_runs_metric"]
    assert list(runs_metric.options) == metric_names
    difference_metric = notebook_defs["best_difference_metric"]
    assert list(difference_metric.options) == metric_names
    assert runs_metric.value == difference_metric.value == selection == "rmse"


def test_the_shap_beeswarm_opens_on_the_first_cohort(
    notebook_defs: dict,
) -> None:
    # 7.1b: the beeswarm's menu needs no click and offers the cohorts (SHAP
    # explains each cohort's count; no composite); its figures exist only
    # after the inspection's click. 7.1e: the permutation figure's menu, the
    # composite first, then each cohort.
    cohort = notebook_defs["shap_cohort"]
    assert list(cohort.options) == notebook_defs["COHORTS"]
    assert cohort.value == notebook_defs["COHORTS"][0]
    permutation_cohort = notebook_defs["permutation_cohort"]
    assert list(permutation_cohort.options) == ["composite", *notebook_defs["COHORTS"]]
    assert permutation_cohort.value == "composite"
    for name in (
        "shap_beeswarm_figures",
        "importance_df",
        "best_runs_permutation",
        "best_runs_shap",
        "permutation_figures",
    ):
        assert name not in notebook_defs, name


def test_the_test_evaluation_is_behind_its_button_and_opens_on_the_forest(
    notebook_defs: dict,
) -> None:
    # 7.2 (E11): the test part is scored once, on a click; a script run
    # evaluates nothing. The test section's plot opens on the forest and the
    # selection metric, any of the six offered.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = [
        cell for cell in source.split("@app.cell") if "train_val_age_group = " in cell
    ]
    assert len(cells) == 1
    assert "mo.stop(\n        not run_test_evaluation.value" in cells[0]
    assert "test_scores_df" not in notebook_defs
    assert notebook_defs["test_plot"].value == "forest"
    metric_names = [metric.name for metric in notebook_defs["METRICS"]]
    assert list(notebook_defs["test_metric"].options) == metric_names
    assert notebook_defs["test_metric"].value == notebook_defs["SELECTION_METRIC"].name


def test_the_test_reading_opens_on_its_defaults(
    notebook_defs: dict,
) -> None:
    # 7.3: the reading of the test predictions needs no click for its menus;
    # its frames and figures exist only after the test evaluation's click. The
    # tables show each model's worst 5 by a count loss finite on every
    # building; the figures have their own model and cohort; the statistics
    # cover all the test buildings or one neighborhood.
    assert notebook_defs["RANKING_METRIC_NAMES"] == ["rmse", "mae", "poisson_deviance"]
    end = notebook_defs["test_reading_end"]
    assert list(end.options) == ["best", "worst"]
    assert end.value == "worst"
    cohort = notebook_defs["test_reading_cohort"]
    assert list(cohort.options) == [*notebook_defs["COHORTS"], "composite"]
    assert cohort.value == "composite"
    metric = notebook_defs["test_reading_metric"]
    assert list(metric.options) == notebook_defs["RANKING_METRIC_NAMES"]
    assert metric.value == notebook_defs["SELECTION_METRIC"].name == "rmse"
    k = notebook_defs["test_reading_k"]
    assert (k.start, k.stop, k.value) == (1, 50, 5)
    model = notebook_defs["test_reading_model"]
    assert list(model.options) == ["independent_cohorts", "total_times_probability"]
    assert model.value == "independent_cohorts"
    figures_cohort = notebook_defs["test_figures_cohort"]
    assert list(figures_cohort.options) == list(cohort.options)
    assert figures_cohort.value == "composite"
    neighborhood = notebook_defs["test_neighborhood"]
    test_neighborhoods = sorted(notebook_defs["test_age_group"].groups.unique())
    assert list(neighborhood.options) == ["all", *test_neighborhoods]
    assert neighborhood.value == "all"
    assert "test_figures_level" not in notebook_defs
    for name in (
        "test_buildings_df",
        "test_building_scores_df",
        "test_ranked_buildings",
        "test_counts_histogram",
        "test_selection_counts_df",
    ):
        assert name not in notebook_defs, name


def test_the_test_explanation_is_behind_its_button(
    notebook_defs: dict,
) -> None:
    # 7.1c (E11): the test evaluation's click also explains the refit models;
    # a script run explains nothing. The test beeswarm's cohort and the
    # waterfalls' end and cohort need no click: the waterfalls open on each
    # model's worst buildings for the first cohort (the composite optional);
    # their building menus exist only after the click.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = [
        cell
        for cell in source.split("@app.cell")
        if "test_shap, test_permutation = " in cell
    ]
    assert len(cells) == 1
    assert "mo.stop(not run_test_evaluation.value)" in cells[0]
    cohorts = notebook_defs["COHORTS"]
    beeswarm_cohort = notebook_defs["test_beeswarm_cohort"]
    assert list(beeswarm_cohort.options) == cohorts
    assert beeswarm_cohort.value == cohorts[0]
    permutation_cohort = notebook_defs["test_permutation_cohort"]
    assert list(permutation_cohort.options) == ["composite", *cohorts]
    assert permutation_cohort.value == "composite"
    end = notebook_defs["test_waterfall_end"]
    assert list(end.options) == ["best", "worst"]
    assert end.value == "worst"
    waterfall_cohort = notebook_defs["test_waterfall_cohort"]
    assert list(waterfall_cohort.options) == [*cohorts, "composite"]
    assert waterfall_cohort.value == cohorts[0]
    for name in (
        "test_models",
        "test_shap",
        "test_permutation",
        "test_importance_df",
        "test_permutation_figures",
        "test_beeswarm_figures",
        "test_waterfall_buildings",
        "test_waterfall_figures",
    ):
        assert name not in notebook_defs, name


def test_the_beeswarm_figure_draws_a_row_per_column_and_per_level(
    notebook_defs: dict,
) -> None:
    # 7.1b, a function since 7.1c (section 5's and the test part's): the
    # columns by the cohort's mean |SHAP|, the largest on top; a category a
    # row per sorted level, kept at its place, in gray; one colorbar for the
    # numeric rows; the dots' jitter seeded.
    beeswarm_figure = notebook_defs["beeswarm_figure"]
    palette = notebook_defs["PALETTE"]
    X = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "s": ["b", "a", "b", "a"], "z": [5.0] * 4}
    )
    values = np.empty((4, 3, 2))
    values[:, :, 0] = [0.1, -0.5, 1.0]
    values[:, :, 1] = [-1.0, 0.5, 0.1]
    explanation = shap.Explanation(
        values=values,
        base_values=np.zeros((4, 2)),
        feature_names=list(X.columns),
        output_names=["cohort_a", "cohort_b"],
    )
    figure = beeswarm_figure(explanation, X, "cohort_a", 0, palette)
    labels = ["z", "s = a", "s = b", "x"]
    assert [trace.name for trace in figure.data] == labels
    assert list(figure.layout.yaxis.ticktext) == labels
    level_rows = [trace for trace in figure.data if " = " in trace.name]
    assert [trace.marker.color for trace in level_rows] == ["#898781"] * 2
    assert [list(trace.customdata) for trace in level_rows] == [["a", "a"], ["b", "b"]]
    assert sum(bool(trace.marker.showscale) for trace in figure.data) == 1
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(X))
    np.testing.assert_array_equal(figure.data[0].y, 3 + jitter)
    np.testing.assert_array_equal(figure.data[3].x, [0.1] * 4)
    other_cohort = beeswarm_figure(explanation, X, "cohort_b", 0, palette)
    assert [trace.name for trace in other_cohort.data] == [
        "x",
        "s = a",
        "s = b",
        "z",
    ]


def test_the_beeswarm_colors_follow_shaps_scale_clipped_at_the_percentiles(
    notebook_defs: dict,
) -> None:
    # A1: shap's blue-to-red (8 stops sampled from shap.plots.colors.red_blue),
    # the value clipped to its 5th-95th percentile, then min-max.
    palette = notebook_defs["PALETTE"]
    positions = [stop[0] for stop in palette["shap"]]
    assert len(positions) == 8
    assert positions[0] == 0 and positions[-1] == 1 and positions == sorted(positions)
    assert palette["shap"][0][1] == "#008bfb" == palette["lowers"]
    assert palette["shap"][-1][1] == "#ff0051" == palette["raises"]
    x = np.append(np.arange(20.0), 1000.0)
    X = pd.DataFrame({"x": x})
    explanation = shap.Explanation(
        values=np.zeros((len(x), 1, 1)),
        base_values=np.zeros((len(x), 1)),
        feature_names=["x"],
        output_names=["cohort_a"],
    )
    figure = notebook_defs["beeswarm_figure"](explanation, X, "cohort_a", 0, palette)
    (trace,) = figure.data
    assert [list(stop) for stop in trace.marker.colorscale] == palette["shap"]
    low, high = np.percentile(x, [5, 95])
    color = np.asarray(trace.marker.color)
    np.testing.assert_allclose(color, (np.clip(x, low, high) - low) / (high - low))
    assert color.min() == 0 and color.max() == 1 and color[-1] == 1
    assert 0 < color[10] < 1


def test_the_waterfall_figure_steps_from_the_base_to_the_prediction(
    notebook_defs: dict,
) -> None:
    # §9's W1 (the display's waterfall as a function, also logged by the final
    # run): the building found by its id; from its base a step per column, the
    # largest |SHAP| on top, to its prediction; the composite the cohorts'
    # values and bases summed (the total).
    waterfall_figure = notebook_defs["waterfall_figure"]
    palette = notebook_defs["PALETTE"]
    dataset = notebook_defs["AgeGroupDataset"](
        X=pd.DataFrame({"x": [1.0, 2.5], "s": ["a", "b"]}),
        Y=pd.DataFrame({"cohort_a": [2, 4], "cohort_b": [1, 3]}),
        exposure=pd.Series([10.0, 20.0]),
        groups=pd.Series(["n0", "n1"]),
        ids=pd.Series(["b0", "b1"]),
    )
    values = np.zeros((2, 2, 2))
    values[1, :, 0] = [0.5, -1.5]
    values[1, :, 1] = [1.0, 0.25]
    explanation = shap.Explanation(
        values=values,
        base_values=np.array([[1.0, 2.0], [3.0, 1.5]]),
        feature_names=["x", "s"],
        output_names=["cohort_a", "cohort_b"],
    )
    predictions = pd.DataFrame({"cohort_a": [1.0, 2.0], "cohort_b": [2.0, 2.75]})
    for cohort, labels, steps, base, prediction, title in (
        ("cohort_a", ["s = b", "x = 2.5"], [-1.5, 0.5], 3.0, 2.0, "cohort_a: true 4"),
        (
            "composite",
            ["x = 2.5", "s = b"],
            [1.5, -1.25],
            4.5,
            4.75,
            "the total: true 7",
        ),
    ):
        figure = waterfall_figure(
            explanation, dataset, predictions, "b1", cohort, palette
        )
        (trace,) = figure.data
        assert trace.type == "waterfall" and trace.base == base
        assert list(trace.y) == labels
        np.testing.assert_array_equal(trace.x, steps)
        assert [shape.x0 for shape in figure.layout.shapes] == [base, prediction]
        assert (
            figure.layout.title.text == f"b1 (n1), {title}, predicted {prediction:.2f}"
        )


def test_the_permutation_figure_draws_a_box_per_split_ordered_by_the_last(
    notebook_defs: dict,
) -> None:
    # 7.1d: per split a box of each column's drops over the repeats, the
    # columns by the last split's mean (the largest on top: a category axis
    # draws its first category at the bottom), the last split in blue, an
    # earlier one in gray, a dotted rule at 0.
    permutation_figure = notebook_defs["permutation_figure"]
    palette = notebook_defs["PALETTE"]
    train = Bunch(
        importances=np.array([[0.9, 1.1], [0.0, 0.2], [0.4, 0.6]]),
        importances_mean=np.array([1.0, 0.1, 0.5]),
    )
    val = Bunch(
        importances=np.array([[0.1, 0.3], [-0.1, 0.1], [0.5, 0.7]]),
        importances_mean=np.array([0.2, 0.0, 0.6]),
    )
    figure = permutation_figure(
        {"train": train, "val": val}, ["a", "b", "c"], "composite rmse", palette
    )
    assert [trace.name for trace in figure.data] == ["train", "val"]
    assert [trace.marker.color for trace in figure.data] == ["#898781", "#2a78d6"]
    assert list(figure.layout.yaxis.categoryarray) == ["b", "a", "c"]
    assert list(figure.data[1].y) == ["b", "b", "a", "a", "c", "c"]
    np.testing.assert_array_equal(figure.data[1].x, [-0.1, 0.1, 0.1, 0.3, 0.5, 0.7])
    np.testing.assert_array_equal(figure.data[0].x, [0.0, 0.2, 0.9, 1.1, 0.4, 0.6])
    assert [(shape.x0, shape.line.dash) for shape in figure.layout.shapes] == [
        (0, "dot")
    ]
    assert figure.layout.xaxis.title.text == "drop in the composite rmse when shuffled"
    test_only = permutation_figure(
        {"test": train}, ["a", "b", "c"], "n_x rmse", palette
    )
    assert list(test_only.layout.yaxis.categoryarray) == ["b", "c", "a"]
    assert test_only.layout.xaxis.title.text == "drop in the n_x rmse when shuffled"
    assert [trace.marker.color for trace in test_only.data] == ["#2a78d6"]


def test_the_count_figures_and_the_shares_bars_are_built_by_functions(
    notebook_defs: dict,
) -> None:
    # M5: the test reading's figures are functions, so the final run logs them
    # at a fixed state; a script run defines them without the test click.
    palette, model_colors = notebook_defs["PALETTE"], notebook_defs["MODEL_COLORS"]
    buildings = pd.DataFrame(
        {
            "building_id": ["b0", "b1", "b2"],
            "neighborhood_id": ["n0", "n0", "n1"],
            "total_true": [3, 5, 8],
            "total_predicted": [2.5, 5.5, 7.0],
        }
    )
    histogram = notebook_defs["counts_histogram_figure"](
        buildings, "total", "#123456", palette
    )
    assert [trace.name for trace in histogram.data] == ["true", "predicted"]
    assert list(histogram.data[0].x) == [3, 5, 8]
    assert list(histogram.data[1].x) == [2.5, 5.5, 7.0]
    assert histogram.data[1].marker.color == "#123456"
    scatter = notebook_defs["counts_scatter_figure"](buildings, "total", palette)
    # The markers: x the true counts, y the predictions (the first trace is the
    # dotted line of equality).
    assert list(scatter.data[1].x) == [3, 5, 8]
    assert list(scatter.data[1].y) == [2.5, 5.5, 7.0]
    assert scatter.layout.xaxis.title.text == "true total"
    true_counts = pd.DataFrame({"a": [1, 3], "b": [2, 2]})
    predicted_counts = {model: true_counts * 1.0 for model in model_colors}
    bars = notebook_defs["selection_shares_bars_figure"](
        true_counts, predicted_counts, model_colors, palette
    )
    assert [trace.name for trace in bars.data] == ["true", *model_colors]
    assert list(bars.data[0].x) == ["a", "b"] and list(bars.data[0].y) == [0.5, 0.5]
    assert bars.data[0].error_y.array is not None
    assert all(trace.error_y.array is None for trace in bars.data[1:])
