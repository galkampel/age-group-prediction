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
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import optuna
import pytest
from optuna.samplers import RandomSampler

from student_simulator import load_simulation_config

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
        "n_trials": "400",
        "n_startup_trials": "20",
        "share_hyperparams": "True",
        "share_feature_transformer": "False",
        "n_resamples": "1000",
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
    # run's study on every script run, the notebook test included. A string
    # check on the cells that name `tune_model` (each run cell calls it): the
    # stop's presence, not its position; the script run's wall time above is
    # the behavioural guard.
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    cells = source.split("@app.cell")
    tuning_cells = [
        cell for cell in cells if "tune_model" in cell and "def tune_model(" not in cell
    ]
    assert len(tuning_cells) >= 1
    assert all(re.search(r"mo\.stop\(\s*not ", cell) for cell in tuning_cells)


@pytest.fixture(scope="module")
def notebook_defs() -> dict:
    """The names a script run defines, in this process; the buttons stay unclicked, so nothing is fitted."""
    spec = importlib.util.spec_from_file_location("experiments_notebook", NOTEBOOK_PATH)
    assert spec is not None and spec.loader is not None
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    _, defs = notebook.app.run()
    return dict(defs)


SHARING = [(False, False), (True, False), (True, True)]


def _count_components(notebook_defs: dict, params: dict, **settings) -> list:
    """The components ``CountComponent.from_params`` reads back from ``params``, one per cohort."""
    return [
        notebook_defs["CountComponent"].from_params(params, name=cohort, **settings)
        for cohort in notebook_defs["COHORTS"]
    ]


@pytest.mark.parametrize(("share_hyperparams", "share_feature_transformer"), SHARING)
def test_a_shared_choice_is_one_draw_in_every_cohorts_component(
    notebook_defs: dict, share_hyperparams: bool, share_feature_transformer: bool
) -> None:
    # Independent: each of the three cohorts draws LightGBM's ten hyperparameters
    # and its transformer, 33 parameters. Shared hyperparameters: one draw under
    # `shared__` in every component, each cohort still drawing its transformer,
    # 13. Both shared: 11, one transformer in every component. The objective
    # builds the model from the draws; the components are read back from the
    # trial's params by the component type, as the best run's are (D25).
    trial = optuna.create_study(sampler=RandomSampler(seed=0)).ask()
    cohorts = notebook_defs["COHORTS"]
    settings = {
        "family": "lgbm_poisson",
        "use_exposure": True,
        "feature_transformer_dict": notebook_defs["COUNT_FEATURE_TRANSFORMERS"][
            "lgbm_poisson"
        ],
    }
    model = notebook_defs["suggest_independent_cohorts_model"](
        trial,
        cohorts=cohorts,
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=notebook_defs["PREDICTION_FLOOR"],
        random_state=0,
        **settings,
    )
    names = sorted(trial.params)
    shared_names = [name for name in names if name.startswith("shared__")]
    transformer_names = [
        name for name in names if name.endswith("__feature_transformer")
    ]
    components = _count_components(notebook_defs, trial.params, **settings)
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
        assert len(names) == (11 if share_feature_transformer else 13)
        assert len(shared_names) == (11 if share_feature_transformer else 10)
        assert len(estimator_params) == 1
    else:
        assert len(names) == 33
        assert shared_names == []
        assert all(name.split("__")[0] in cohorts for name in names)
    # The read-back components make the estimators the objective made.
    for cohort, component in zip(cohorts, components, strict=True):
        assert (
            component.make_estimator(random_state=0).get_params()
            == model.cohort_models[cohort].estimator.get_params()
        )
    # The record of a run: plain data, the transformer by its declaration.
    for component in components:
        record = json.loads(json.dumps(component.to_dict()))
        assert record["feature_transformer"]["column_plans"]


@pytest.mark.parametrize(("share_hyperparams", "share_feature_transformer"), SHARING)
def test_the_best_components_are_read_back_from_the_trials_params(
    notebook_defs: dict, share_hyperparams: bool, share_feature_transformer: bool
) -> None:
    # The refit after a study reads the best components from best_params by
    # the component type (D25): the model built from them must be the model
    # the objective built from the same draws, for either model.
    trial = optuna.create_study(sampler=RandomSampler(seed=1)).ask()
    cohorts = notebook_defs["COHORTS"]
    settings = {
        "family": "nb2",
        "use_exposure": False,
        "feature_transformer_dict": notebook_defs["COUNT_FEATURE_TRANSFORMERS"]["nb2"],
    }
    drawn = notebook_defs["suggest_independent_cohorts_model"](
        trial,
        cohorts=cohorts,
        share_hyperparams=share_hyperparams,
        share_feature_transformer=share_feature_transformer,
        y_min=notebook_defs["PREDICTION_FLOOR"],
        random_state=0,
        **settings,
    )
    read = notebook_defs["build_independent_cohorts_model"](
        _count_components(notebook_defs, trial.params, **settings),
        y_min=notebook_defs["PREDICTION_FLOOR"],
        random_state=0,
    )
    for cohort in cohorts:
        drawn_model, read_model = (
            drawn.cohort_models[cohort],
            read.cohort_models[cohort],
        )
        assert read_model.estimator.get_params() == drawn_model.estimator.get_params()
        assert read_model.feature_transformer is drawn_model.feature_transformer

    trial = optuna.create_study(sampler=RandomSampler(seed=2)).ask()
    count_transformers = notebook_defs["COUNT_FEATURE_TRANSFORMERS"]["lgbm_gaussian"]
    probability_transformers = notebook_defs["PROBABILITY_FEATURE_TRANSFORMERS"]["lgbm"]
    drawn = notebook_defs["suggest_total_times_probability_model"](
        trial,
        total_family="lgbm_gaussian",
        probability_family="lgbm",
        use_exposure=True,
        total_feature_transformer_dict=count_transformers,
        probability_feature_transformer_dict=probability_transformers,
        y_min=notebook_defs["PREDICTION_FLOOR"],
        random_state=0,
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
    read = notebook_defs["build_total_times_probability_model"](
        total, probability, y_min=notebook_defs["PREDICTION_FLOOR"], random_state=0
    )
    assert (
        read.total_model.estimator.get_params()
        == drawn.total_model.estimator.get_params()
    )
    assert (
        read.probability_model.estimator.get_params()
        == drawn.probability_model.estimator.get_params()
    )
    assert read.total_model.feature_transformer is drawn.total_model.feature_transformer
    assert (
        read.probability_model.calibration_method
        == drawn.probability_model.calibration_method
    )
    assert total.estimator_params.max_depth == trial.params["total__max_depth"]
    assert (
        probability.estimator_params.max_depth == trial.params["probability__max_depth"]
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
