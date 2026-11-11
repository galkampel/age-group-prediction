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
import pytest
from optuna.samplers import RandomSampler
from scipy.stats import gaussian_kde

from age_group_prediction.hyperparameter_tuning import (
    suggest_independent_cohorts_components,
    suggest_total_times_probability_components,
)
from student_simulator import load_simulation_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_hyperparameter_tuning.py"
OBJECTIVE_EDA_PATH = PROJECT_ROOT / "notebooks" / "02_objective_eda.py"


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
    env["MLFLOW_EXPERIMENT_NAME"] = "test-hyperparameter-tuning-notebook"
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
def notebook_defs() -> dict:
    """The names a script run defines, in this process; the buttons stay unclicked, so nothing is fitted."""
    spec = importlib.util.spec_from_file_location(
        "hyperparameter_tuning_notebook", NOTEBOOK_PATH
    )
    assert spec is not None and spec.loader is not None
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    _, defs = notebook.app.run()
    return dict(defs)


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
    # Independent: each of the three cohorts draws LightGBM's ten hyperparameters
    # and its transformer, 33 parameters. Shared hyperparameters: one draw under
    # `shared__` in every component, each cohort still drawing its transformer,
    # 13. Both shared: 11, one transformer in every component. The package's
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
        assert len(names) == (11 if share_feature_transformer else 13)
        assert len(shared_names) == (11 if share_feature_transformer else 10)
        assert len(estimator_params) == 1
    else:
        assert len(names) == 33
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
    for part in ("train_age_group", "val_age_group", "test_age_group"):
        for field, tuning_values, eda_values in zip(
            tuning[part]._fields, tuning[part], eda[part], strict=True
        ):
            assert tuning_values.equals(eda_values), (part, field)
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
    figure = forest_figure(scores, "rmse", "lower", "upper")
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
        scores, "rmse", "lower", "upper", share_x_axis=True, rule_at=0
    )
    assert len(ruled.layout.shapes) == 3
    assert all(shape.x0 == shape.x1 == 0 for shape in ruled.layout.shapes)
    one_model = forest_figure(
        scores[scores["model"] == "model_a"], "rmse", "lower", "upper"
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
    figure = ridge_figure(replicate_scores, "rmse")
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
    ruled = ridge_figure(replicate_scores, "rmse", share_x_axis=True, rule_at=0)
    assert len(ruled.layout.shapes) == 3
    assert all(shape.x0 == shape.x1 == 0 for shape in ruled.layout.shapes)


def test_the_plots_open_on_their_forms_and_the_selection_metric(
    notebook_defs: dict,
) -> None:
    # 5.4b: the best runs' plot opens on the forest, the paired difference's
    # on the ridge; both metric menus on the selection metric. The runs' menu
    # (which also sets the permutation importance's metric, bias excepted)
    # offers the two selection metrics and bias; the difference's, all six.
    assert notebook_defs["best_runs_plot"].value == "forest"
    assert notebook_defs["best_difference_plot"].value == "ridge"
    selection = notebook_defs["SELECTION_METRIC"].name
    runs_metric = notebook_defs["best_runs_metric"]
    assert list(runs_metric.options) == ["rmse", "poisson_deviance", "bias"]
    difference_metric = notebook_defs["best_difference_metric"]
    assert list(difference_metric.options) == [
        metric.name for metric in notebook_defs["METRICS"]
    ]
    assert runs_metric.value == difference_metric.value == selection == "rmse"


def test_the_shap_beeswarm_opens_on_the_first_cohort(
    notebook_defs: dict,
) -> None:
    # 7.1b: the beeswarm's menu needs no click and offers the cohorts (SHAP
    # explains each cohort's count; no composite); its figures exist only
    # after the inspection's click.
    cohort = notebook_defs["shap_cohort"]
    assert list(cohort.options) == notebook_defs["COHORTS"]
    assert cohort.value == notebook_defs["COHORTS"][0]
    assert "shap_beeswarm_figures" not in notebook_defs
    assert "importance_df" not in notebook_defs


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
