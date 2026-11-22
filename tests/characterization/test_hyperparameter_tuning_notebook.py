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

import optuna
import pandas as pd
import pytest
from optuna.samplers import RandomSampler
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
    assert len(notebook_defs["shared_params"]) == 17
    dataset_inputs = notebook_defs["tuning_dataset_inputs"]
    assert {context: dataset.name for context, dataset in dataset_inputs.items()} == {
        "training": "train",
        "validation": "val",
    }
    assert all(len(dataset.digest) == 36 for dataset in dataset_inputs.values())


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
    # The notebook's own settings: five of the 17 are parameters.
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
