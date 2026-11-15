# MLflow runs of the tuning notebook

How `notebooks/03_hyperparameter_tuning.py` logs its results to MLflow, how to open them, and
how to rebuild a model from a run. The decisions behind it are in
[MLFLOW_RUNS_PLAN.md](MLFLOW_RUNS_PLAN.md) (§1) and its two reviews,
[MLFLOW_RUNS_REVIEW.md](MLFLOW_RUNS_REVIEW.md).

The old stack's guide, [MLFLOW_EXPERIMENTS_GUIDE.md](MLFLOW_EXPERIMENTS_GUIDE.md), and its
`age_group_prediction.tracking` package are not used here: the notebook logs from its own
cells (decision 5).

## 1. What is logged, and when

Three buttons in the notebook each log one kind of run, all in the one experiment that
`MLFLOW_EXPERIMENT_NAME` names. Nothing is logged without a click, and a script run of the
notebook logs nothing.

| Run kind | Runs | Button (click after) | `stage` tag | Run name |
|---|---|---|---|---|
| a tuning run | 24, one per study | "Log the tuning runs" (both "Tune …" buttons) | `tuning` | `<model> <family> exposure=<True or False>` |
| the best runs' run | 1 | "Log the best runs" ("Inspect the best runs") | `best_runs` | `best runs` |
| the final run | 1 | "Log the final run" ("Evaluate on the test part") | `final` | `final` |

- **A tuning run** holds one study: its settings, its best parameters, its cross-validation
  and point scores, its trials table and journal, and the components' declaration.
- **The best runs' run** holds the comparison of each model's best run on the validation
  part: the bootstrap statistics, the paired difference, the permutation importance and
  SHAP figures, and the 24 runs' tables.
- **The final run** holds the same evaluation on the test part (E11). It is the only run with
  test metrics.

No fitted model is logged (decision 4): a run holds what rebuilds its models (§9).

## 2. The store

The notebook reads the store from four variables, set in `.env` (from `.env.example`):

| Variable | Meaning |
|---|---|
| `MLFLOW_TRACKING_URI` | a `sqlite:///` file, e.g. `sqlite:///.mlflow-local/mlflow.db` |
| `MLFLOW_EXPERIMENT_NAME` | the experiment, e.g. `age-group-local` |
| `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` | the experiment's artifact folder, e.g. `.mlflow-local/mlartifacts` |
| `MLFLOW_DISABLE_AGENT_HINT` | `1`: silences MLflow's agent hint |

The repository root's `mlflow.db` and `mlartifacts/` are the old stack's canonical store,
which must not change. The notebook's MLflow cell (after the data cell) refuses to run, with
a `ValueError` naming the variable, when:
- `MLFLOW_TRACKING_URI` is unset (MLflow's default is `sqlite:///mlflow.db` in the working
  directory, the canonical store), is not a `sqlite:///` URI, or names the root's `mlflow.db`;
- `MLFLOW_EXPERIMENT_NAME` is unset;
- `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` is unset or names the root's `mlartifacts/`;
- the experiment exists with another artifact location. An experiment keeps the location it
  was created with, and MLflow uses it over the variable; one created by a bare MLflow call
  from the repository root would hold the canonical `./mlartifacts`. Use that location, or
  another experiment name.

A missing experiment is created there, with the variable's location. The cell sets no MLflow
global (no `set_tracking_uri`, no `set_experiment`): MLflow reads the two `MLFLOW_` variables
at every call.

It also sets two warning filters, each by its category, message and module, so that any
other warning still shows. Under pandas 3, two MLflow calls warn on every use:
- `log_table`'s `select_dtypes` on a string column (`Pandas4Warning`);
- `log_input`'s schema inference on an integer column (`UserWarning`, "Hint: Inferred schema
  contains integer…"; under `-W error` MLflow drops the schema silently).

A script run checks the store without logging. It prints the line
`experiment=<name> tracking_uri=<uri>`, and the notebook's facts:

```bash
uv run --env-file .env --group notebook --group tracking \
  python notebooks/03_hyperparameter_tuning.py
```

To open the notebook and click the buttons, follow the root
[README](../README.md#run-the-modeling-notebook), step 3.

## 3. The MLflow UI

From the repository root, with `.env`'s variables exported to the shell:

```bash
set -a; source .env; set +a
uv run --group tracking mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --artifacts-destination "$AGE_GROUP_MLFLOW_ARTIFACT_LOCATION" --port 5001
```

Then open `http://127.0.0.1:5001` and select the experiment.
- **`--artifacts-destination` is given on purpose:** run from the repository root, it
  defaults to `./mlartifacts`, the canonical folder. The runs' own artifacts are read from
  the experiment's location either way.
- **The port:** 5000 is often taken on macOS (the AirPlay receiver).
- **The UI's mode:** MLflow 3.16 opens in its "GenAI" mode with an "MLflow Assistant"
  panel; the runs are under the experiment's runs list.
- **Figures** are plotly HTML that load plotly.js from `cdn.plot.ly`: they need the network.
  The UI shows each in its artifact viewer, and a `.csv` file as a table.
- **The `Default` experiment** is MLflow's own, created with the store. Its artifact
  location is `<working directory>/mlruns/0`; nothing logs to it and no folder is created.

## 4. What each run kind holds

**Every run:**
- **tags:** `run_key` (§6), `stage`, and the code's version: `notebook`
  (`notebooks/03_hyperparameter_tuning.py`), `git_commit` (HEAD) and `git_dirty` (`true` when
  a tracked file differs from HEAD);
- **23 shared parameters:** `config_hash` and the data settings (`config_path`,
  `n_neighborhoods`, `buildings_per_neighborhood_rate`, `simulation_seed`); the experiment
  settings (`seed`, `split_method`, `test_size`, `val_size`, `n_splits`, `n_trials`,
  `n_startup_trials`, `objective_metric`, `selection_metric`, `n_resamples`, `alpha`,
  `n_repeats`, `n_shap_rows`, `n_shap_background`); `n_train`, `n_val`, `n_test`; and
  `prediction_floor` (every model's `y_min`);
- **two dataset inputs,** each the part's X and Y side by side. Its digest is the first 36
  characters of the SHA-256 of its rows (MLflow's limit), so a changed split is a new
  dataset.

| | A tuning run | The best runs' run | The final run |
|---|---|---|---|
| more tags | `model`, `family` (`<total>+<probability>` for Model 2) | `best.<model>`: each best run's study name | `best.<model>`; `final=true` |
| more parameters | the run's settings (`family` or `total_family` and `probability_family`, `use_exposure`, Model 1's sharing flags), `study_seed`, `best.<name>` per best parameter | none | `n_train_val` |
| metrics | 52 (§5) | 108 (§5) | 108 (§5) |
| artifacts | `trials.csv`, `best_params.json`, `components.json`, `search_space.json`, `config.json`, `study/<study name>.log` | 7 tables in `tables/`, 18 figures in `figures/`, `best_params/<model>.json`, `components/<model>.json` | 6 tables, 23 figures, `best_params/<model>.json`, `components/<model>.json` |
| inputs (context: name) | `training`: `train`, `validation`: `val` | the same | `training`: `train_val`, `testing`: `test` |

- **The artifacts:**
  - `trials.csv` has a row per trial: `number`, `value`, `seconds`, `params_*`,
    `user_attrs_*`, `state`;
  - `search_space.json` holds each drawn parameter's distribution, as Optuna's JSON
    (`optuna.distributions.json_to_distribution(json.dumps(value))` reads one back);
  - `config.json` holds the simulator config's dotted keys, the same in every run
    (`config_hash` stands for it as a parameter);
  - `study/<study name>.log` is the study's Optuna journal;
  - `components.json` and `components/<model>.json` hold the components' `to_dict()`.
- **The best runs' tables:** `best_runs_val_scores`, `best_difference`, `importance`,
  `model1_runs`, `model1_scores`, `model2_runs`, `model2_scores`. A runs table's
  `feature_transformers` and `hyperparameters` cells are JSON text.
- **The final run's tables:** `test_scores`, `test_difference`, `val_against_test`,
  `test_importance`, `test_buildings`, `test_building_scores`.
- **The figures:**
  - the forest and the ridge of the selection metric, for the two runs and for their
    difference;
  - per model, the permutation figure of the composite and of each cohort, and the beeswarm
    of each cohort;
  - the final run adds the two count figures of the total per model and the shares bars of
    every test building.
- **The tables are CSV at 6 significant digits,** for reading in the UI. The exact values are
  in the journal, `best_params` and the `best.*` parameters.

## 5. The metric names

- **A tuning run (52):** `cv/<objective>` (the best trial's cross-validation score); `<train
  or val>/<cohort or composite>/<metric>` (the best model refit on the training part, 2 × 4 ×
  6); `seconds`, `n_complete`, `n_failed`.
- **An evaluation run (108):** `<val or test>/<model or difference>/<metric>/<statistic>`, for
  the composite only, with the statistics `score`, `mean`, `median`, `sd`, `ci95_lower`,
  `ci95_upper` (2 models and the difference × 6 metrics × 6 statistics). Every cohort's rows
  are in the tables.
- **`test/` metrics are in the final run only.**

The run's page groups the charts by the name's prefix (`val/difference/rmse`, …).

## 6. Logged once: the skip and the guard

Each run has a `run_key` tag. Its first part is the study folder's name (the data's hash, the
split, the seed and the budget: `data=<hash>__split_method=…__n_startup_trials=…`):

| Run kind | `run_key` |
|---|---|
| a tuning run | `<folder>/<study name>` |
| the best runs' run | `<folder>/best_runs__objective=<objective>__selection=<selection metric>` |
| the final run | `<folder>/final` |

- **The two skips:** a click logs a run only when no FINISHED run in the experiment has its
  key. The tuning button reports "Logged n tuning runs, skipped m (already logged)."; the best
  runs' button "The best runs' run is logged: skipped.". A run that failed mid-logging is
  marked FAILED in the UI and logged again on the next click.
- **The final run's guard:** "Log the final run" is refused when any FINISHED run tagged
  `final=true` exists in the experiment, whatever its key ("A final run exists in the
  experiment: nothing logged."). The test part stands for production and is evaluated once
  (E11).
- **A deleted run is not seen** by the skips or the guard (`search_runs` reads active runs
  only). Deleting a run in the UI is how it is logged again; deleting the final run is that
  decision.
- **Commit before logging:** a run's `git_commit` is the code that ran only when `git_dirty`
  is `false`.
- **`mlflow.source.name`** is the running script (the marimo kernel or a check script), not
  the notebook: read the `notebook` tag.

## 7. Comparing the 24 tuning runs in the UI

The run names are long and look alike in the runs list's default column width. To compare:
1. Search for `tags.stage = 'tuning'`.
2. In the columns menu, add the tags `model` and `family`, the parameter `use_exposure`, and
   the metric `val/composite/<the selection metric>` (`val/composite/rmse` by default).
3. Sort by that metric, ascending.

`family` is a tag in all 24 runs: the column that spans both models. Each model's best run by
that metric is the run its `best.<model>` tag names in the evaluation runs.

## 8. From an evaluation run to its tuning runs

An evaluation run's `best.<model>` tag is a study name. Behind its own key's folder, it is a
tuning run's `run_key`:

```python
import os

import mlflow

client = mlflow.MlflowClient()  # MLFLOW_TRACKING_URI
experiment = client.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])
(best_runs_run,) = client.search_runs(
    [experiment.experiment_id],
    "tags.stage = 'best_runs' and attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
folder = best_runs_run.data.tags["run_key"].split("/")[0]
for model in ("independent_cohorts", "total_times_probability"):
    run_key = f"{folder}/{best_runs_run.data.tags[f'best.{model}']}"
    tuning_runs = client.search_runs(
        [experiment.experiment_id], f"tags.run_key = '{run_key}'"
    )
    print(model, [run.info.run_name for run in tuning_runs])
```

The tuning run exists only if "Log the tuning runs" was clicked. The evaluation run does not
need it: it holds its own `best_params` and components.

## 9. Rebuilding a model from a run

Each run kind holds what rebuilds its models with the components' `from_params` and the
notebook's builders, as the notebook builds them:

| Argument | A tuning run | The best runs' run, the final run |
|---|---|---|
| `best_params` | `best_params.json` | `best_params/<model>.json` |
| `family` | the parameter `family` (Model 2: `total_family`, `probability_family`) | `components/<model>.json`: each component's `family` |
| `use_exposure` | the parameter `use_exposure` | `components/<model>.json`: each count component's `use_exposure` |
| `feature_transformer_dict` | the notebook's `COUNT_FEATURE_TRANSFORMERS[family]` or `PROBABILITY_FEATURE_TRANSFORMERS[family]` at the run's `git_commit` | the same |
| the model | `build_*_model(components, y_min=prediction_floor, random_state=seed)`, the two parameters | the same |
| fit on | the training part | the training part (best runs); training and validation (final) |

`components.json` is the readable declaration, also a check of the rebuild: the rebuilt
components' `to_dict()` equals it. The candidates and the data are the notebook's, so check
out the run's `git_commit` and keep its data settings (`config_hash` must match).

The recipe below loads the notebook's names from a run with every button unclicked (nothing is
tuned or logged), rebuilds both models of the latest best runs' run and of the final run and
the best tuning run by the selection metric, refits each, and asserts its composite scores
equal the run's metrics. Run on a 3-trial store, it was exact for all three run kinds. Save it
as a script and run it from the repository root like the script run in §2, with
`MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false` (else each artifact read prints a progress bar):

```python
import importlib.util
import os

import mlflow
import pandas as pd

# The notebook's names (the candidates, the builders, the data): every button unclicked.
spec = importlib.util.spec_from_file_location(
    "tuning_notebook", "notebooks/03_hyperparameter_tuning.py"
)
notebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notebook)
_, names = notebook.app.run()

client = mlflow.MlflowClient()  # MLFLOW_TRACKING_URI
experiment = client.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])


def first_run(filter_string, order_by="attributes.start_time DESC"):
    """The first FINISHED run of the experiment matching `filter_string`, by `order_by`
    (the latest by default)."""
    (run,) = client.search_runs(
        [experiment.experiment_id],
        f"{filter_string} and attributes.status = 'FINISHED'",
        order_by=[order_by],
        max_results=1,
    )
    return run


def read_json(run, path):
    return mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/{path}")


def rebuild_model(run, model_name):
    """The model `model_name` of `run`, unfitted: its components from `best_params`, each with its family
    and exposure (a tuning run's parameters; an evaluation run's components file)."""
    params = run.data.params
    if run.data.tags["stage"] == "tuning":
        best_params = read_json(run, "best_params.json")
        use_exposure = params["use_exposure"] == "True"
        if model_name == "independent_cohorts":
            declared = {
                cohort: {"family": params["family"], "use_exposure": use_exposure}
                for cohort in names["COHORTS"]
            }
        else:
            declared = {
                "total": {"family": params["total_family"], "use_exposure": use_exposure},
                "probability": {"family": params["probability_family"]},
            }
    else:
        best_params = read_json(run, f"best_params/{model_name}.json")
        declared = read_json(run, f"components/{model_name}.json")
    model_settings = {
        "y_min": float(params["prediction_floor"]),
        "random_state": int(params["seed"]),
    }
    if model_name == "independent_cohorts":
        components = [
            names["CountComponent"].from_params(
                best_params,
                name=cohort,
                family=declared[cohort]["family"],
                use_exposure=declared[cohort]["use_exposure"],
                feature_transformer_dict=names["COUNT_FEATURE_TRANSFORMERS"][
                    declared[cohort]["family"]
                ],
            )
            for cohort in names["COHORTS"]
        ]
        return names["build_independent_cohorts_model"](components, **model_settings)
    elif model_name == "total_times_probability":
        total = names["CountComponent"].from_params(
            best_params,
            name="total",
            family=declared["total"]["family"],
            use_exposure=declared["total"]["use_exposure"],
            feature_transformer_dict=names["COUNT_FEATURE_TRANSFORMERS"][
                declared["total"]["family"]
            ],
        )
        probability = names["ProbabilityComponent"].from_params(
            best_params,
            name="probability",
            family=declared["probability"]["family"],
            feature_transformer_dict=names["PROBABILITY_FEATURE_TRANSFORMERS"][
                declared["probability"]["family"]
            ],
        )
        return names["build_total_times_probability_model"](
            total, probability, **model_settings
        )
    else:
        raise ValueError(f"unknown model {model_name!r}")


def composite_scores(model, fit_on, score_on):
    """`model` fitted on `fit_on` and scored on `score_on`: {metric: composite score}."""
    model.fit(fit_on.X, fit_on.Y, exposure=fit_on.exposure)
    predictions = model.predict(score_on.X, exposure=score_on.exposure)
    scores = names["compute_metrics"](
        score_on.Y, predictions, names["METRICS"], cohorts=["composite"]
    )
    return scores.loc["composite"].to_dict()


train, val, test = names["train_age_group"], names["val_age_group"], names["test_age_group"]
train_val = names["AgeGroupDataset"](
    *(
        pd.concat([train_part, val_part])
        for train_part, val_part in zip(train, val, strict=True)
    )
)
selection = names["SELECTION_METRIC"].name

tuning_run = first_run("tags.stage = 'tuning'", f"metrics.`val/composite/{selection}` ASC")
best_runs_run = first_run("tags.stage = 'best_runs'")
final_run = first_run("tags.final = 'true'")
for run in (tuning_run, best_runs_run, final_run):
    assert run.data.params["config_hash"] == names["config_hash"], "another data version"

model_name = tuning_run.data.tags["model"]
scores = composite_scores(rebuild_model(tuning_run, model_name), train, val)
for metric, score in scores.items():
    assert score == tuning_run.data.metrics[f"val/composite/{metric}"]
print("tuning run", tuning_run.info.run_name, scores)

for run, fit_on, score_on, dataset in (
    (best_runs_run, train, val, "val"),
    (final_run, train_val, test, "test"),
):
    for model_name in ("independent_cohorts", "total_times_probability"):
        scores = composite_scores(rebuild_model(run, model_name), fit_on, score_on)
        for metric, score in scores.items():
            assert score == run.data.metrics[f"{dataset}/{model_name}/{metric}/score"]
        print(run.info.run_name, model_name, scores)
```
