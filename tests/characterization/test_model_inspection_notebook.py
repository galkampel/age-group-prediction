"""Characterize the model inspection notebook: 03's setup, its guarded read of the tuning runs, the inspection.

The notebook (``notebooks/04_model_inspection.py``) copies the tuning notebook's
setup, reads the tuning runs that notebook logs to MLflow, and rebuilds each
model's best run from what the run logged; then it bootstraps and inspects the
two best runs, evaluates them on the test part, and logs the best runs' run and
the final run. A script run on an empty store stops at the read; the read's
functions are tested on a scratch store holding a few real tuning runs, logged
by the tuning notebook's own ``log_tuning_run``; the figures, the menus and the
evaluation runs' logger on small inputs.
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
    # is selected; no button, no run opened. The cell found by the tuning
    # runs' stage: the two evaluation runs' cells search the store too (the
    # best runs' skip, the final run's guard).
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = [
        cell
        for cell in source.split("@app.cell")
        if "\"tags.stage = 'tuning' \"" in cell
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
    # D3: the validation bootstrap, which needs no click, reads the record: a
    # row per best run with its model, family and exposure. The inspection and
    # the two evaluation runs wait for their buttons, so only the tuning runs
    # are in the store.
    assert sorted(
        defs["best_runs_val_scores_df"][["model", "family", "use_exposure"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == [
        (model_name, record["family"], record["use_exposure"])
        for model_name, record in best_runs.items()
    ]
    assert "importance_df" not in defs
    assert "test_scores_df" not in defs
    with _environment(tuning_store["environment"]):
        stages = mlflow.search_runs(search_all_experiments=True)["tags.stage"]
    assert stages.tolist() == ["tuning"] * len(tuning_store["run_ids"])


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


def test_the_best_runs_are_logged_by_the_cell_that_follows_the_inspection(
    notebook_defs: dict,
) -> None:
    # §8's L3: the best runs' run is logged by the cell that reads the
    # inspection's results, with no button: it runs once "Inspect the best
    # runs" has produced them. A script run defines the functions and opens no
    # run (test_a_script_run_on_an_empty_store_stops_at_the_read).
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
    # C3b: the study_folder tag, and each best run's tuning run id as a tag.
    # D3: the best runs are the read's records (Task D, decision 6), so their
    # study name, tuning run id and best parameters are read from the record.
    mlflow = pytest.importorskip("mlflow")
    environment = _mlflow_environment(tmp_path)
    client = mlflow.MlflowClient(environment["MLFLOW_TRACKING_URI"])
    experiment_id = client.create_experiment(
        environment["MLFLOW_EXPERIMENT_NAME"],
        artifact_location=str((tmp_path / "artifacts").resolve()),
    )
    best_runs = {
        model: {
            "study_name": f"{model}_study",
            "run_id": f"{model}_run_id",
            # A value per model, so a swap between the two is caught.
            "best_params": {
                "total__alpha": alpha,
                "total__feature_transformer": "glm_base",
            },
            "components": [
                Bunch(name="total", to_dict=lambda: {"family": "poisson_glm"})
            ],
        }
        for model, alpha in (
            ("independent_cohorts", 0.01),
            ("total_times_probability", 0.25),
        )
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
    settings = {**notebook_defs["shared_params"], "n_train_val": 1218}
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
        "best.independent_cohorts.run_id": "independent_cohorts_run_id",
        "best.total_times_probability.run_id": "total_times_probability_run_id",
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
            == best_run["best_params"]
        )
        assert json.loads(
            (artifacts / "components" / f"{model_name}.json").read_text()
        ) == {"total": {"family": "poisson_glm"}}
    assert mlflow.active_run() is None
    assert mlflow.get_tracking_uri() != environment["MLFLOW_TRACKING_URI"]


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
    best_runs = {
        model: {
            "study_name": f"{model}_study",
            "run_id": f"{model}_run_id",
            "best_params": {},
            "components": [],
        }
        for model in ("independent_cohorts", "total_times_probability")
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
