"""Characterize the model inspection notebook: 03's setup, and its guarded read of the tuning runs.

The notebook (``notebooks/04_model_inspection.py``) copies the tuning notebook's
setup, reads the tuning runs that notebook logs to MLflow, and rebuilds each
model's best run from what the run logged. A script run on an empty store stops
at the read; the read's functions are tested on a scratch store holding a few
real tuning runs, logged by the tuning notebook's own ``log_tuning_run``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.samplers import RandomSampler

from age_group_prediction.hyperparameter_tuning import (
    suggest_independent_cohorts_components,
    suggest_total_times_probability_components,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_model_inspection.py"
TUNING_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_hyperparameter_tuning.py"


def _mlflow_environment(scratch: Path) -> dict[str, str]:
    """The MLflow variables of a scratch store under ``scratch``, away from the canonical one."""
    return {
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "MLFLOW_TRACKING_URI": f"sqlite:///{scratch / 'mlflow.db'}",
        "MLFLOW_EXPERIMENT_NAME": "test-model-inspection-notebook",
        "AGE_GROUP_MLFLOW_ARTIFACT_LOCATION": str(scratch / "artifacts"),
    }


@contextlib.contextmanager
def _environment(environment: dict[str, str]) -> Iterator[None]:
    """``environment`` set for the block only: MLflow reads it at every call, and no global is left."""
    with pytest.MonkeyPatch.context() as patch:
        for name, value in environment.items():
            patch.setenv(name, value)
        yield


def _run_notebook(notebook_path: Path, environment: dict[str, str]) -> tuple[dict, str]:
    """The names a script run of ``notebook_path`` defines, and what it prints, against a scratch store.

    The run is in this process (the notebook's buttons stay unclicked); the
    MLflow variables (``environment``) point at the scratch store for the run
    only.
    """
    spec = importlib.util.spec_from_file_location(notebook_path.stem, notebook_path)
    assert spec is not None and spec.loader is not None
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    stdout = io.StringIO()
    with _environment(environment), contextlib.redirect_stdout(stdout):
        _, defs = notebook.app.run()
    return dict(defs), stdout.getvalue()


def _project_copy(tmp_path: Path, source: str) -> Path:
    """``source`` as the notebook of a project copy under ``tmp_path``; its path.

    The notebook finds the config and the code relative to its own directory,
    so the copy's ``configs`` and ``src`` link to the repository's.
    """
    project_copy = tmp_path / "project"
    (project_copy / "notebooks").mkdir(parents=True)
    for name in ("configs", "src"):
        project_copy.joinpath(name).symlink_to(
            PROJECT_ROOT / name, target_is_directory=True
        )
    notebook_copy = project_copy / "notebooks" / NOTEBOOK_PATH.name
    notebook_copy.write_text(source, encoding="utf-8")
    return notebook_copy


def _printed(stdout: str, prefix: str) -> dict[str, str]:
    """The ``name=value`` pairs of the one stdout line starting with ``prefix``."""
    lines = [line for line in stdout.splitlines() if line.startswith(prefix + " ")]
    assert len(lines) == 1, f"expected one {prefix!r} line, got {lines!r}"
    pairs = (item.split("=", 1) for item in lines[0].split(" ")[1:])
    return {name: value for name, value in pairs}


@pytest.fixture(scope="module")
def notebook_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The empty scratch MLflow store of the script run below."""
    return tmp_path_factory.mktemp("notebook_store")


@pytest.fixture(scope="module")
def notebook_run(notebook_store: Path) -> tuple[dict, str]:
    """One script run of the notebook as committed, on an empty store: its names and its output."""
    return _run_notebook(NOTEBOOK_PATH, _mlflow_environment(notebook_store))


@pytest.fixture(scope="module")
def notebook_defs(notebook_run: tuple[dict, str]) -> dict:
    """The names the script run defines: the setup, the read's functions, not the read's results."""
    return notebook_run[0]


@pytest.fixture(scope="module")
def tuning_notebook_defs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The names a script run of the tuning notebook defines; nothing is tuned."""
    defs, _ = _run_notebook(
        TUNING_NOTEBOOK_PATH,
        _mlflow_environment(tmp_path_factory.mktemp("tuning_notebook_store")),
    )
    return defs


def test_a_script_run_on_an_empty_store_stops_at_the_read(
    notebook_run: tuple[dict, str], notebook_store: Path
) -> None:
    # D1's decision 1 (TUNING_NOTEBOOK_REVISION_PLAN.md, Task D): the read
    # counts the store's tuning runs and stops when any is missing, so nothing
    # is selected; the cells before it run. The MLflow cell creates the
    # experiment with its artifact location and opens no run; the code's tags
    # name this notebook (decision 8).
    mlflow = pytest.importorskip("mlflow")
    defs, stdout = notebook_run
    assert _printed(stdout, "tuning_runs") == {
        "independent_cohorts": "0",
        "total_times_probability": "0",
    }
    # marimo keeps what the read cell assigned before its stop (the empty
    # search), not what follows it.
    assert defs["tuning_runs_df"].empty
    assert "best_runs" not in defs
    for name in (
        "shared_params",
        "tuning_dataset_inputs",
        "study_folder",
        "N_TUNING_RUNS",
        "select_best_runs",
        "read_tuning_run",
        "rebuild_best_run",
        "check_best_run",
    ):
        assert name in defs, name
    environment = _mlflow_environment(notebook_store)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    experiment = client.get_experiment_by_name(environment["MLFLOW_EXPERIMENT_NAME"])
    assert experiment is not None
    assert experiment.artifact_location == str((notebook_store / "artifacts").resolve())
    experiment_ids = [item.experiment_id for item in client.search_experiments()]
    assert client.search_runs(experiment_ids) == []
    code_tags = defs["code_tags"]
    assert set(code_tags) == {"notebook", "git_commit", "git_dirty"}
    assert code_tags["notebook"] == "notebooks/04_model_inspection.py"
    assert re.fullmatch(r"[0-9a-f]{40}", code_tags["git_commit"])
    assert os.environ.get("MLFLOW_TRACKING_URI") != environment["MLFLOW_TRACKING_URI"]
    assert mlflow.get_tracking_uri() != environment["MLFLOW_TRACKING_URI"]


def test_the_read_stops_unless_every_tuning_run_is_in_the_store() -> None:
    # The stop is the read cell's (the empty-store run above shows it stop): one
    # search of the setup's FINISHED tuning runs, the count per model and the
    # distinct run keys per model both against N_TUNING_RUNS, before anything
    # is selected; no button, no run opened.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = [
        cell for cell in source.split("@app.cell") if "mlflow.search_runs(" in cell
    ]
    assert len(cells) == 1
    cell = cells[0]
    assert cell.count("mlflow.search_runs(") == 1
    for clause in (
        "\"tags.stage = 'tuning' \"",
        "f\"and tags.study_folder = '{study_folder}' \"",
        "f\"and params.objective_metric = '{objective_metric}' \"",
        "\"and attributes.status = 'FINISHED'\"",
    ):
        assert clause in cell, clause
    assert ".nunique()" in cell
    assert re.search(
        r"mo\.stop\(\s*_n_runs != N_TUNING_RUNS or _n_run_keys != N_TUNING_RUNS,", cell
    )
    for call in (
        "select_best_runs(",
        "read_tuning_run(",
        "rebuild_best_run(",
        "check_best_run(",
    ):
        assert cell.index("mo.stop(") < cell.index(call), call
    assert re.search(r"\.value\b", cell) is None
    assert "start_run" not in cell


# Each cell 04 copies from 03, by a line found in that cell only (decision 5).
COPIED_CELLS = {
    "sys.path": '_src = str(mo.notebook_dir().parent / "src")',
    "config": "base_config = load_simulation_config(config_path)",
    "data": "simulator = StudentPopulationSimulator(",
    "exposure": 'exposure_col = "n_apartments"',
    "config hash": "config_hash = hashlib.sha256(",
    "MLflow": "import mlflow\n",
    "split": "splitter = Splitter(split_method)",
    "plans": "passthrough_plan = Plan(",
    "feature variants": "COUNT_FEATURE_TRANSFORMERS = {",
    "cohorts": "PREDICTION_FLOOR = float(",
    "shared settings": "shared_params = {",
}


def _cell(source: str, marker: str) -> str:
    """The one cell of ``source`` holding ``marker``."""
    cells = [cell for cell in source.split("@app.cell") if marker in cell]
    assert len(cells) == 1, (marker, len(cells))
    return cells[0]


def test_the_inspection_notebook_copies_the_tuning_notebooks_setup(
    notebook_defs: dict, tuning_notebook_defs: dict
) -> None:
    # Decision 4: 04 reads the runs 03 tuned, so its data, split, feature
    # transformers and settings must be 03's. The copied cells are equal as
    # text (the MLflow cell's notebook tag aside); the cells 04 writes itself
    # are compared by their values. The guard (below) compares 04 with the
    # setup that tuned the store, which this test cannot.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    tuning_source = TUNING_NOTEBOOK_PATH.read_text(encoding="utf-8")
    for name, marker in COPIED_CELLS.items():
        tuning_cell = _cell(tuning_source, marker)
        if name == "MLflow":
            assert tuning_cell.count('"notebooks/03_hyperparameter_tuning.py"') == 1
            tuning_cell = tuning_cell.replace(
                '"notebooks/03_hyperparameter_tuning.py"',
                '"notebooks/04_model_inspection.py"',
            )
        assert _cell(source, marker) == tuning_cell, name
    inspection, tuning = notebook_defs, tuning_notebook_defs
    for name in (
        "seed",
        "split_method",
        "test_size",
        "val_size",
        "n_splits",
        "n_trials",
        "n_startup_trials",
        "objective_metric",
    ):
        assert inspection[name] == tuning[name], name
    # As objects: each metric's name, function and direction (both notebooks
    # import the same functions in this process).
    assert inspection["METRICS"] == tuning["METRICS"]
    assert inspection["OBJECTIVE_METRIC"] == tuning["OBJECTIVE_METRIC"]
    # The folder 03 stores its studies in and tags its runs with.
    assert inspection["study_folder"] == tuning["tuner"].storage_dir.name
    assert inspection["N_TUNING_RUNS"] == {
        "independent_cohorts": len(tuning["MODEL1_RUN_SETTINGS"]),
        "total_times_probability": len(tuning["MODEL2_RUN_SETTINGS"]),
    }


def _tuning_run(
    defs: dict,
    model_name: str,
    run_settings: dict,
    sampler_seed: int,
    study_seed: int,
    study_name: str,
) -> dict:
    """A real tuning run of ``model_name``, as the tuning notebook's Tune cells build it.

    A 2-trial study whose trials draw the run's components as the package's
    objective does (its value is the trial's number, so nothing is fitted in
    the study); the best components read back by ``from_params``; the model
    built, refit on the training part and scored on training and validation,
    by the tuning notebook's names (``defs``).
    """
    cohorts = defs["COHORTS"]
    count_transformers = defs["COUNT_FEATURE_TRANSFORMERS"]
    probability_transformers = defs["PROBABILITY_FEATURE_TRANSFORMERS"]
    use_exposure = run_settings["use_exposure"]

    def objective(trial: optuna.Trial) -> float:
        if model_name == "independent_cohorts":
            suggest_independent_cohorts_components(
                trial,
                family=run_settings["family"],
                use_exposure=use_exposure,
                cohorts=cohorts,
                feature_transformer_dict=count_transformers[run_settings["family"]],
                share_hyperparams=run_settings["share_hyperparams"],
                share_feature_transformer=run_settings["share_feature_transformer"],
            )
        else:
            suggest_total_times_probability_components(
                trial,
                total_family=run_settings["total_family"],
                probability_family=run_settings["probability_family"],
                use_exposure=use_exposure,
                total_feature_transformer_dict=count_transformers[
                    run_settings["total_family"]
                ],
                probability_feature_transformer_dict=probability_transformers[
                    run_settings["probability_family"]
                ],
            )
        return float(trial.number)

    study = optuna.create_study(
        study_name=study_name, sampler=RandomSampler(seed=sampler_seed)
    )
    study.optimize(objective, n_trials=2)
    if model_name == "independent_cohorts":
        components = tuple(
            defs["CountComponent"].from_params(
                study.best_params,
                name=cohort,
                family=run_settings["family"],
                use_exposure=use_exposure,
                feature_transformer_dict=count_transformers[run_settings["family"]],
            )
            for cohort in cohorts
        )
        model = defs["build_independent_cohorts_model"](
            components, y_min=defs["PREDICTION_FLOOR"], random_state=defs["seed"]
        )
    else:
        components = (
            defs["CountComponent"].from_params(
                study.best_params,
                name="total",
                family=run_settings["total_family"],
                use_exposure=use_exposure,
                feature_transformer_dict=count_transformers[
                    run_settings["total_family"]
                ],
            ),
            defs["ProbabilityComponent"].from_params(
                study.best_params,
                name="probability",
                family=run_settings["probability_family"],
                feature_transformer_dict=probability_transformers[
                    run_settings["probability_family"]
                ],
            ),
        )
        model = defs["build_total_times_probability_model"](
            *components, y_min=defs["PREDICTION_FLOOR"], random_state=defs["seed"]
        )
    train, val = defs["train_age_group"], defs["val_age_group"]
    model.fit(train.X, train.Y, exposure=train.exposure)
    scores = pd.concat(
        {
            dataset_name: defs["compute_metrics"](
                dataset.Y,
                model.predict(dataset.X, exposure=dataset.exposure),
                defs["METRICS"],
                cohorts=[*cohorts, "composite"],
            )
            .stack()
            .to_frame("score")
            for dataset_name, dataset in (("train", train), ("val", val))
        },
        names=["dataset"],
    )
    return {
        "study": study,
        "study_seed": study_seed,
        "cv_best": study.best_value,
        "scores": scores,
        "seconds": 1.0,
        "n_complete": 2,
        "n_failed": 0,
        "components": components,
    }


# The store's runs by label: the model, its run settings, the sampler's seed and
# the study's seed. A and B are one draw (one sampler seed) under two study
# names, so one score: their study seeds 10 and 9 order the other way as text.
TUNING_RUNS = {
    "A": (
        "independent_cohorts",
        {
            "family": "lgbm_poisson",
            "use_exposure": True,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        0,
        10,
    ),
    "B": (
        "independent_cohorts",
        {
            "family": "lgbm_poisson",
            "use_exposure": True,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        0,
        9,
    ),
    "C": (
        "independent_cohorts",
        {
            "family": "poisson_glm",
            "use_exposure": False,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        1,
        11,
    ),
    # The Gaussian loss can predict a negative count: the family the prediction
    # floor (y_min) exists for.
    "E": (
        "independent_cohorts",
        {
            "family": "lgbm_gaussian",
            "use_exposure": False,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        3,
        12,
    ),
    "D": (
        "total_times_probability",
        {
            "total_family": "lgbm_poisson",
            "probability_family": "logistic",
            "use_exposure": True,
        },
        2,
        42,
    ),
}


@pytest.fixture(scope="module")
def tuning_store(
    notebook_defs: dict,
    tuning_notebook_defs: dict,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """A scratch store holding the five real tuning runs of TUNING_RUNS, each logged by the
    tuning notebook's ``log_tuning_run`` with its datasets; the store's environment and the
    runs' ids by label."""
    mlflow = pytest.importorskip("mlflow")
    scratch = tmp_path_factory.mktemp("tuning_store")
    environment = _mlflow_environment(scratch)
    # Created first: MLflow would create it at its default artifact location.
    mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"]).create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((scratch / "artifacts").resolve()),
    )
    tuning = tuning_notebook_defs
    studies_dir = scratch / notebook_defs["study_folder"]
    studies_dir.mkdir()
    code_tags = {
        "notebook": "notebooks/03_hyperparameter_tuning.py",
        "git_commit": "0" * 40,
        "git_dirty": "false",
    }
    run_ids = {}
    # The tuning notebook's MLflow cell filters this warning of log_input (an
    # integer column in the data); pytest resets the filters at every test.
    with _environment(environment), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Hint: Inferred schema contains integer",
            module="mlflow.types.utils",
        )
        for label, (
            model_name,
            run_settings,
            sampler_seed,
            study_seed,
        ) in TUNING_RUNS.items():
            step = (
                "independent_cohorts"
                if model_name == "independent_cohorts"
                else "total_times_probability_joint"
            )
            study_name = tuning["build_study_name"](
                step, run_settings, "rmse", study_seed
            )
            (studies_dir / f"{study_name}.log").write_text(
                "the study's journal\n", encoding="utf-8"
            )
            run_ids[label] = tuning["log_tuning_run"](
                _tuning_run(
                    tuning,
                    model_name,
                    run_settings,
                    sampler_seed,
                    study_seed,
                    study_name,
                ),
                model_name,
                run_settings,
                studies_dir,
                "rmse",
                tuning["shared_params"],
                tuning["config_params"],
                tuning["tuning_dataset_inputs"],
                code_tags,
            )
    return {"environment": environment, "run_ids": run_ids}


@pytest.fixture(scope="module")
def rebuilt_runs(notebook_defs: dict, tuning_store: dict) -> dict:
    """Runs A, C, E and D (both models, four families) as the notebook reads and rebuilds
    them: by label, the run's id, its logged record and its rebuild."""
    rebuilt = {}
    with _environment(tuning_store["environment"]):
        for label in ("A", "C", "E", "D"):
            run_id = tuning_store["run_ids"][label]
            logged = notebook_defs["read_tuning_run"](run_id)
            rebuilt[label] = (
                run_id,
                logged,
                notebook_defs["rebuild_best_run"](
                    run_id,
                    TUNING_RUNS[label][0],
                    logged["components"],
                    logged["best_params"],
                ),
            )
    return rebuilt


@pytest.mark.parametrize("selection_metric_name", ["rmse", "poisson_deviance"])
def test_each_models_best_run_is_its_lowest_validation_score_a_tie_to_the_lower_seed(
    notebook_defs: dict, tuning_store: dict, selection_metric_name: str
) -> None:
    # Decision 7: per model, the lowest validation score of the composite on
    # the selection metric; a tie goes to the lower study seed, an integer (03
    # took the first in its list's order, the lowest seed). The study seed is a
    # parameter, so the frame holds it as text, where "10" sorts before "9".
    mlflow = pytest.importorskip("mlflow")
    with _environment(tuning_store["environment"]):
        runs_df = mlflow.search_runs(filter_string="tags.stage = 'tuning'")
    run_ids = tuning_store["run_ids"]
    select_best_runs = notebook_defs["select_best_runs"]
    metric = f"metrics.val/composite/{selection_metric_name}"
    by_id = runs_df.set_index("run_id")
    assert by_id.loc[run_ids["A"], metric] == by_id.loc[run_ids["B"], metric]
    tied = runs_df[runs_df["run_id"].isin([run_ids["A"], run_ids["B"]])]
    # In either order of the frame.
    for frame in (tied, tied.iloc[::-1]):
        (best_run,) = select_best_runs(frame, selection_metric_name).values()
        assert best_run["run_id"] == run_ids["B"]
    best_runs = select_best_runs(runs_df, selection_metric_name)
    assert list(best_runs) == ["independent_cohorts", "total_times_probability"]
    model1_ids = [run_ids[label] for label in ("A", "B", "C", "E")]
    expected = min(
        model1_ids,
        key=lambda run_id: (
            by_id.loc[run_id, metric],
            int(by_id.loc[run_id, "params.study_seed"]),
        ),
    )
    assert best_runs["independent_cohorts"]["run_id"] == expected
    assert best_runs["total_times_probability"]["run_id"] == run_ids["D"]


@pytest.mark.parametrize("label", ["A", "C", "E", "D"])
def test_a_rebuilt_best_run_reproduces_its_logged_scores(
    notebook_defs: dict, rebuilt_runs: dict, label: str
) -> None:
    # The probe (Task D): a run rebuilt from components.json + best_params.json
    # with from_params, refit on the training part, gives its logged scores
    # exactly; its exposure is a bool (the MLflow parameter is text). The
    # guard passes it.
    run_id, logged, rebuilt = rebuilt_runs[label]
    _, run_settings, _, _ = TUNING_RUNS[label]
    model_name = TUNING_RUNS[label][0]
    assert logged["scores"]["score"].dtype == np.float64
    assert len(logged["scores"]) == 48
    pd.testing.assert_frame_equal(
        rebuilt["scores"].reset_index(),
        logged["scores"],
        check_exact=True,
        check_dtype=False,
    )
    assert [component.name for component in rebuilt["components"]] == list(
        logged["components"]
    )
    assert rebuilt["use_exposure"] is run_settings["use_exposure"]
    notebook_defs["check_best_run"](
        run_id, logged, rebuilt, notebook_defs["tuning_dataset_inputs"]
    )
    # The floor changes a score only where a prediction falls below it, which
    # none of these runs' predictions do (a floor of 1e-3 gives the same 48
    # scores), so the rebuilt count models are read: built with 03's floor, and
    # a LightGBM estimator with the notebook's seed.
    model = rebuilt["model"]
    if model_name == "independent_cohorts":
        count_models = list(model.cohort_models.values())
    else:
        count_models = [model.total_model]
    for count_model in count_models:
        assert count_model.y_min == notebook_defs["PREDICTION_FLOOR"]
        if "random_state" in count_model.estimator.get_params():
            assert (
                count_model.estimator.get_params()["random_state"]
                == (notebook_defs["seed"])
            )


def test_the_guard_refuses_a_run_logged_on_other_data(
    notebook_defs: dict, rebuilt_runs: dict
) -> None:
    # Decision 4's guard: a run tuned on other data or another split (here
    # one of this notebook's parts replaced by the test part, either one) is
    # refused, naming the run.
    run_id, logged, rebuilt = rebuilt_runs["A"]
    for context, name in (("training", "train"), ("validation", "val")):
        dataset_inputs = {
            **notebook_defs["tuning_dataset_inputs"],
            context: notebook_defs["build_mlflow_dataset"](
                notebook_defs["test_age_group"], name
            ),
        }
        with pytest.raises(ValueError, match=rf"tuning run {run_id}'s dataset inputs"):
            notebook_defs["check_best_run"](run_id, logged, rebuilt, dataset_inputs)


def test_a_run_naming_what_this_notebook_lacks_is_refused_naming_the_run(
    notebook_defs: dict, rebuilt_runs: dict
) -> None:
    # A feature transformer renamed or removed since the run was tuned: the
    # run's best parameters name it, from_params cannot find it, and the
    # rebuild refuses the run as the guard does, naming it and the name.
    run_id, logged, _ = rebuilt_runs["D"]
    best_params = {
        **logged["best_params"],
        "total__feature_transformer": "a_removed_transformer",
    }
    with pytest.raises(
        ValueError,
        match=rf"tuning run {run_id}'s components cannot be rebuilt: "
        r"this notebook has no 'a_removed_transformer'",
    ):
        notebook_defs["rebuild_best_run"](
            run_id, "total_times_probability", logged["components"], best_params
        )


def test_the_guard_refuses_a_run_whose_components_differ_from_their_rebuild(
    notebook_defs: dict, rebuilt_runs: dict
) -> None:
    # A feature transformer whose declaration changed since the run was tuned
    # (here a cohort's logged transformer is another candidate's): the rebuild
    # reads the name and gets this notebook's declaration, which differs.
    run_id, logged, rebuilt = rebuilt_runs["A"]
    cohort = notebook_defs["COHORTS"][0]
    record = logged["components"][cohort]
    other = next(
        json.loads(json.dumps(transformer.to_dict()))
        for transformer in notebook_defs["COUNT_FEATURE_TRANSFORMERS"][
            "lgbm_poisson"
        ].values()
        if json.loads(json.dumps(transformer.to_dict()))
        != record["feature_transformer"]
    )
    tampered = {
        **logged,
        "components": {
            **logged["components"],
            cohort: {**record, "feature_transformer": other},
        },
    }
    with pytest.raises(
        ValueError, match=rf"tuning run {run_id}'s components \['{cohort}'\] differ"
    ):
        notebook_defs["check_best_run"](
            run_id, tampered, rebuilt, notebook_defs["tuning_dataset_inputs"]
        )


def test_the_guard_refuses_a_run_whose_scores_differ_from_their_rebuild(
    notebook_defs: dict, rebuilt_runs: dict
) -> None:
    # The scores are compared exactly: one logged score a float's step away
    # (a changed floor, seed or library would move far more) is refused, and
    # so is a scores.csv without one of its rows.
    run_id, logged, rebuilt = rebuilt_runs["D"]
    dataset_inputs = notebook_defs["tuning_dataset_inputs"]
    scores = logged["scores"].copy()
    scores.loc[0, "score"] = np.nextafter(scores.loc[0, "score"], np.inf)
    with pytest.raises(
        ValueError,
        match=rf"tuning run {run_id}'s scores differ from its rebuild's: 1 of 48",
    ):
        notebook_defs["check_best_run"](
            run_id, {**logged, "scores": scores}, rebuilt, dataset_inputs
        )
    with pytest.raises(
        ValueError, match=rf"tuning run {run_id}'s scores.csv rows differ"
    ):
        notebook_defs["check_best_run"](
            run_id,
            {**logged, "scores": logged["scores"].iloc[1:]},
            rebuilt,
            dataset_inputs,
        )


N_TUNING_RUNS_LINE = (
    '    N_TUNING_RUNS = {"independent_cohorts": 8, "total_times_probability": 16}\n'
)


def _run_on_the_tuning_store(
    tmp_path: Path, tuning_store: dict, n_tuning_runs: dict[str, int]
) -> tuple[dict, str]:
    """A script run, on the tuning store, of a copy of the notebook that expects ``n_tuning_runs``."""
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert source.count(N_TUNING_RUNS_LINE) == 1
    notebook_copy = _project_copy(
        tmp_path,
        source.replace(N_TUNING_RUNS_LINE, f"    N_TUNING_RUNS = {n_tuning_runs!r}\n"),
    )
    return _run_notebook(notebook_copy, tuning_store["environment"])


def test_the_read_cell_records_each_models_best_run(
    notebook_defs: dict,
    tuning_notebook_defs: dict,
    tuning_store: dict,
    tmp_path: Path,
) -> None:
    # The read cell past its stop (decision 6's record, which the inspection
    # reads): a copy expecting the store's 4 + 1 runs selects, rebuilds and
    # guards each model's best run; its record holds the run's id and study
    # name, the family tag, the exposure as a bool, the logged best
    # parameters, the rebuilt components and model, and the rebuilt scores.
    mlflow = pytest.importorskip("mlflow")
    defs, stdout = _run_on_the_tuning_store(
        tmp_path,
        tuning_store,
        {"independent_cohorts": 4, "total_times_probability": 1},
    )
    assert _printed(stdout, "tuning_runs") == {
        "independent_cohorts": "4",
        "total_times_probability": "1",
    }
    best_runs = defs["best_runs"]
    assert list(best_runs) == ["independent_cohorts", "total_times_probability"]
    with _environment(tuning_store["environment"]):
        runs_df = mlflow.search_runs(filter_string="tags.stage = 'tuning'")
        expected = notebook_defs["select_best_runs"](runs_df, "rmse")
        logged_by_model = {
            model_name: notebook_defs["read_tuning_run"](record["run_id"])
            for model_name, record in best_runs.items()
        }
    labels = {run_id: label for label, run_id in tuning_store["run_ids"].items()}
    val = defs["val_age_group"]
    for model_name, record in best_runs.items():
        assert set(record) == {
            "run_id",
            "study_name",
            "family",
            "use_exposure",
            "best_params",
            "components",
            "model",
            "scores",
        }
        assert record["run_id"] == expected[model_name]["run_id"]
        _, run_settings, _, study_seed = TUNING_RUNS[labels[record["run_id"]]]
        if model_name == "independent_cohorts":
            step, family = "independent_cohorts", run_settings["family"]
        else:
            step = "total_times_probability_joint"
            family = (
                f"{run_settings['total_family']}+{run_settings['probability_family']}"
            )
        assert record["study_name"] == tuning_notebook_defs["build_study_name"](
            step, run_settings, "rmse", study_seed
        )
        assert record["family"] == family
        assert record["use_exposure"] is run_settings["use_exposure"]
        logged = logged_by_model[model_name]
        assert record["best_params"] == logged["best_params"]
        assert {
            component.name: json.loads(json.dumps(component.to_dict()))
            for component in record["components"]
        } == logged["components"]
        pd.testing.assert_frame_equal(
            record["scores"].reset_index(),
            logged["scores"],
            check_exact=True,
            check_dtype=False,
        )
        # The record's model is the fitted one its scores were computed from.
        val_scores = (
            defs["compute_metrics"](
                val.Y,
                record["model"].predict(val.X, exposure=val.exposure),
                defs["METRICS"],
                cohorts=[*defs["COHORTS"], "composite"],
            )
            .stack()
            .to_numpy()
        )
        assert np.array_equal(
            val_scores, record["scores"].loc["val", "score"].to_numpy()
        )


def test_the_read_cell_stops_when_a_model_has_other_than_its_runs(
    tuning_store: dict, tmp_path: Path
) -> None:
    # Decision 1 on a store that is not empty: a copy expecting 3 runs of the
    # independent cohorts model finds 4 and stops, selecting nothing.
    defs, stdout = _run_on_the_tuning_store(
        tmp_path,
        tuning_store,
        {"independent_cohorts": 3, "total_times_probability": 1},
    )
    assert _printed(stdout, "tuning_runs") == {
        "independent_cohorts": "4",
        "total_times_probability": "1",
    }
    assert len(defs["tuning_runs_df"]) == 5
    assert "best_runs" not in defs
