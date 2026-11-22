# MLflow runs of the tuning and inspection notebooks

How `notebooks/03_hyperparameter_tuning.py` (the 24 tuning runs) and
`notebooks/04_model_inspection.py` (the best runs' run and the final run) log their results to
MLflow, how to open them, and how to rebuild a model from a run. To check the logging yourself, follow §10's steps. The
decisions behind it are in [MLFLOW_RUNS_PLAN.md](MLFLOW_RUNS_PLAN.md) (§1; §8 for the logging
without buttons) and its three reviews,
[MLFLOW_RUNS_REVIEW.md](MLFLOW_RUNS_REVIEW.md).

The old stack's guide, [MLFLOW_EXPERIMENTS_GUIDE.md](MLFLOW_EXPERIMENTS_GUIDE.md), and its
`age_group_prediction.tracking` package are not used here: the notebooks log from their own
cells (decision 5). `03` and `04` keep the same code for their setup, so both see the same data,
split and feature variants.

## 1. What is logged, and when

MLflow's terms, as this guide uses them:
- **the store:** a sqlite file for the runs' values (its address is the *tracking URI*) and a
  folder for their files (the *artifact location*);
- **an experiment:** a named group of runs; the notebooks log to one;
- **a run:** one logged result, here a study or an evaluation;
- **a parameter:** a setting, kept as text;
- **a metric:** a number, which the UI sorts by and charts;
- **a tag:** a label to filter runs by;
- **an artifact:** a file kept with the run (a table, a figure, JSON);
- **a dataset input:** the data a run used, by its name and hash (not the data itself).

Each run is logged by the click that produces its result, all in the one experiment that
`MLFLOW_EXPERIMENT_NAME` names: four clicks in two notebooks (`03` two, then `04` two), no
logging button. Nothing is logged without a click, and a script run of either notebook logs
nothing.

| Run kind | Runs | Notebook and click | `stage` tag | Run name |
|---|---|---|---|---|
| a tuning run | 24, one per study | `03`: "Tune the independent cohorts model" (8) and "Tune the total times probability model" (16), each run as its study finishes | `tuning` | `<model> <family> exposure=<True or False>` |
| the best runs' run | 1 | `04`: "Inspect the best runs", once the inspection is done | `best_runs` | `best runs` |
| the final run | 1 | `04`: "Evaluate on the test part", once the test evaluation is done | `final` | `final` |

`04` reads the 24 tuning runs back from the store (no button, no logging) and selects each
model's best run from them (§6, §8), so `03`'s two clicks come first.

A study loaded from the cache (a Tune click that finds its study stored) is logged like a
study just tuned.

- **A tuning run** holds one study: its settings, its best parameters, its cross-validation
  and point scores, its trials table and journal, the components' declaration, and Optuna's
  figures (three; `figures/param_importances.html` only when the complete trials hold two or more
  distinct values, §4).
- **The best runs' run** holds the comparison of each model's best run on the validation
  part: the bootstrap statistics, the paired difference, the permutation importance and
  SHAP figures, and 3 tables.
- **The final run** holds the same evaluation on the test part (E11), with each test
  building's SHAP values and the waterfalls of each model's 5 worst test buildings. It is the
  only run with test metrics.

No fitted model is logged (decision 4): a run holds what rebuilds its models (§9).

## 2. The store

The notebooks read the store from three variables, set in `.env` (from `.env.example`) beside
a fourth that MLflow reads itself:

| Variable | Meaning |
|---|---|
| `MLFLOW_TRACKING_URI` | a `sqlite:///` file, e.g. `sqlite:///.mlflow-local/mlflow.db` |
| `MLFLOW_EXPERIMENT_NAME` | the experiment, e.g. `age-group-local` |
| `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` | the experiment's artifact folder, e.g. `.mlflow-local/mlartifacts` |
| `MLFLOW_DISABLE_AGENT_HINT` | `1`: silences MLflow's agent hint |

The repository root's `mlflow.db` and `mlartifacts/` are the old stack's canonical store,
which must not change. Each notebook's MLflow cell (after the data cell) refuses to run, with
a `ValueError` naming the variable, when:
- `MLFLOW_TRACKING_URI` is unset (MLflow's default is `sqlite:///mlflow.db` in the working
  directory, the canonical store), is not a `sqlite:///` URI, or names the root's `mlflow.db`;
- `MLFLOW_EXPERIMENT_NAME` is unset;
- `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` is unset or names the root's `mlartifacts/`;
- the experiment exists with another artifact location. An experiment keeps the location it
  was created with, and MLflow uses it over the variable; one created by a bare MLflow call
  from the repository root would hold the canonical `./mlartifacts`. Use that location, or
  another experiment name.

A missing experiment is created there, with the variable's location. The cell also silences
two pandas 3 warnings inside MLflow; any other warning still shows.

A script run checks the store without logging. It prints the line
`experiment=<name> tracking_uri=<uri>`, and the notebook's facts:

```bash
uv run --env-file .env --group notebook --group tracking \
  python notebooks/03_hyperparameter_tuning.py
```

`notebooks/04_model_inspection.py` runs the same way. It also reads the store's tuning runs:
on a store that does not hold exactly the 24 (§10, step 4) the script stops at that read and prints
the counts.

To open a notebook and click the buttons, follow the root
[README](../README.md#run-the-modeling-notebook), step 3.

## 3. The MLflow UI

From the repository root, with `.env`'s variables exported to the shell:

```bash
set -a; source .env; set +a
uv run --group tracking mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --artifacts-destination "$AGE_GROUP_MLFLOW_ARTIFACT_LOCATION" --port 5001
```

Then open `http://127.0.0.1:5001`:
1. Click **Model training**, the toggle at the top left. MLflow 3.16 opens in its "GenAI"
   mode, where an experiment shows a traces page and no runs; the choice is kept.
2. Click the experiment: its runs list.
3. Click a run. Its page's tabs: **Overview** (the tags, the parameters, the metrics as a
   table, the datasets), **Model metrics** (the metrics as charts), **Artifacts** (the files:
   a `.csv` shown as a table, a figure drawn).
4. The file to open first: a tuning run's `trials.csv`; the best runs' run's
   `tables/best_runs_val_scores.csv` and `figures/val_forest_runs_<metric>.html`; the final
   run's `tables/test_scores.csv`.

- **`--artifacts-destination` is given on purpose:** run from the repository root, it
  defaults to `./mlartifacts`, the canonical folder. The runs' own artifacts are read from
  the experiment's location either way.
- **The port:** 5000 is often taken on macOS (the AirPlay receiver).
- **Figures** are plotly HTML that load plotly.js from `cdn.plot.ly`: they need the network.
- **The `Default` experiment** is MLflow's own, created with the store. Its artifact
  location is `<working directory>/mlruns/0`; nothing logs to it and no folder is created.

## 4. What each run kind holds

**The rule:** a parameter or a metric is something you would put on a chart's axis or filter
by; everything else is an artifact. A value identical in every run cannot separate runs in a
chart, so it stays in `settings.json`; the five parameters of "Every run" below are the
exception (identical across one store, logged as a filter or a grouping across stores).

**Every run:**
- **tags:** `run_key` and `study_folder` (§6), `stage`, and the code's version: `notebook`
  (the notebook that logged the run: `notebooks/03_hyperparameter_tuning.py` on a tuning run,
  `notebooks/04_model_inspection.py` on the best runs' run and the final run), `git_commit`
  (HEAD) and `git_dirty` (`true` when a tracked file differs from HEAD);
- **`settings.json`, the complete record of the run's settings:** `config_hash` and the
  data settings (`config_path`, `n_neighborhoods`, `buildings_per_neighborhood_rate`,
  `simulation_seed`); `n_train`, `n_val`, `n_test`; `prediction_floor` (every model's `y_min`);
  and the experiment settings. `03` has 8 of them (`seed`, `split_method`, `test_size`,
  `val_size`, `n_splits`, `n_trials`, `n_startup_trials`, `objective_metric`), so **a tuning
  run's holds 17 settings, plus `study_seed`**. `04` has all 14, adding `selection_metric`,
  `n_resamples`, `alpha`, `n_repeats`, `n_shap_rows` and `n_shap_background`, so **an
  evaluation run's holds 23 settings** (the final run's also `n_train_val`). The parameters are
  a subset of it;
- **at least these five as parameters in every run (the evaluation runs add `selection_metric`):** `config_hash`, `seed`, `split_method`,
  `n_trials`, `objective_metric`;
- **two dataset inputs,** each the part's X and Y side by side. Its digest is the first 36
  characters of the SHA-256 of its rows (MLflow's limit), so a changed split is a new
  dataset.

| | A tuning run | The best runs' run | The final run |
|---|---|---|---|
| more tags | `model`, `family` (`<total>+<probability>` for Model 2) | `best.<model>`: each best run's study name; `best.<model>.run_id`: its tuning run's id | the same; `final=true` |
| more parameters | the run's settings (`family` or `total_family` and `probability_family`, `use_exposure`, Model 1's sharing flags), `study_seed`, `best.<name>` per best parameter: 10 + `best.*` (Model 1), 9 + `best.*` (Model 2) | `selection_metric`: 6 in all | the same 6 |
| metrics | 19 (§5) | 54 (§5) | 54 (§5) |
| artifacts | `trials.csv`, `best_params.json`, `components.json`, `search_space.json`, `config.json`, `settings.json`, `scores.csv`, `study/<study name>.log`, `figures/optimization_history.html`, `figures/slice.html`, `figures/param_importances.html` | `settings.json`, 3 tables in `tables/`, 18 figures in `figures/`, `best_params/<model>.json`, `components/<model>.json` | the same, with 7 tables and 33 figures |
| inputs (context: name) | `training`: `train`, `validation`: `val` | the same | `training`: `train_val`, `testing`: `test` |

- **The metrics are logged last,** after every artifact and input: a run that fails while
  logging is FAILED with no metrics, so it never shows in a chart.
- **The artifacts:**
  - `trials.csv` has a row per trial: `number`, `value`, `seconds`, `params_*`,
    `user_attrs_*`, `state`;
  - `scores.csv` has the best model's 48 scores, written in full: `dataset`, `cohort`,
    `metric`, `score` (the metrics keep the composite's 12 and the validation score of each
    cohort under the objective);
  - `settings.json` is read back as a dictionary (§9 reads `prediction_floor` from it);
  - `figures/optimization_history.html`, `figures/slice.html` and
    `figures/param_importances.html` are Optuna's charts of the study (§7). The importances
    are fANOVA seeded with `study_seed`; a study whose complete trials hold fewer than two
    distinct values (fANOVA cannot rank them) has no such figure;
  - `search_space.json` holds each drawn parameter's distribution, as Optuna's JSON
    (`optuna.distributions.json_to_distribution(json.dumps(value))` reads one back);
  - `config.json` holds the simulator config's dotted keys, the same in every run
    (`config_hash` stands for it as a parameter);
  - `study/<study name>.log` is the study's Optuna journal (§7 reads it);
  - `components.json` and `components/<model>.json` hold the components' `to_dict()`.
- **The best runs' tables:** `best_runs_val_scores`, `best_difference`, `importance`. The
  24 tuning runs' own tables are not repeated: each tuning run holds its `trials.csv`,
  `scores.csv` and `components.json`.
- **The final run's tables:** `test_scores`, `test_difference`, `val_against_test`,
  `test_importance`, `test_buildings`, `test_building_scores`, `test_shap_values`.
  `test_shap_values` has a row per model, test building and cohort: `model`, `building_id`,
  `cohort`, `base_value`, then a SHAP value per feature (the base plus the values is the
  cohort's prediction). The features' own values and the counts are in `test_buildings`,
  joined on `model` and `building_id`.
- **The figures:**
  - the forest and the ridge of the selection metric, for the two runs and for their
    difference;
  - per model, the permutation figure of the composite and of each cohort, and the beeswarm
    of each cohort;
  - the final run adds the two count figures of the total per model, the shares bars of
    every test building, and per model the waterfalls of the total for its 5 worst test
    buildings by the selection metric on the composite
    (`figures/test_waterfall_<model>_total_<rank>_<building_id>.html`, rank 1 the worst). Any
    other building's or cohort's waterfall is rebuilt from `test_shap_values`: the cohort's
    row, or the three cohorts' rows summed for the total.
- **The tables are CSV at 6 significant digits,** for reading in the UI. The exact values are
  in the journal, `best_params`, the `best.*` parameters and a tuning run's `scores.csv`.

## 5. The metric names

- **A tuning run (19):** `cv/<objective>` (the best trial's cross-validation score);
  `<train or val>/composite/<metric>` (the best model refit on the training part, 2 × 6);
  `val/<cohort>/<objective>` (the 3 cohorts' validation score under the objective); `seconds`,
  `n_complete`, `n_failed`. The other cohort scores are in `scores.csv`.
- **An evaluation run (54):** `<val or test>/<model or difference>/<metric>/<statistic>`, for
  the composite only, with the statistics `score`, `ci95_lower`, `ci95_upper` (2 models and
  the difference × 6 metrics × 3 statistics). The mean, the median and the SD, and every
  cohort's rows, are in the tables.
- **`test/` metrics are in the final run only.**

The run's page groups the charts by the name's prefix (`val/difference/rmse`, …).

## 6. Logged once: the skip and the guard

Each run has a `run_key` tag. Its first part is the study folder's name (the data's hash, the
split, the seed and the budget: `data=<hash>__split_method=…__n_startup_trials=…`), which every
run also holds as the tag `study_folder`: `tags.study_folder = '<folder>'` is the exact filter
for the runs of one setup.

| Run kind | `run_key` |
|---|---|
| a tuning run | `<folder>/<study name>` |
| the best runs' run | `<folder>/best_runs__objective=<objective>__selection=<selection metric>` |
| the final run | `<folder>/final` |

- **The two skips:** a click logs a run only when no FINISHED run in the experiment has its
  key. Each Tune cell reports its own runs, "Logged n tuning runs, skipped m (already
  logged)." (8 for the independent cohorts model, 16 for the total times probability model);
  after "Inspect the best runs", "The best runs' run is logged: skipped.". A run that failed
  mid-logging is marked FAILED in the UI, holds no metrics (they are logged last, so it is
  not a point in a chart), and is logged again on the next click.
- **The final run's guard:** after "Evaluate on the test part", the final run is refused when
  any FINISHED run tagged `final=true` exists in the experiment, whatever its key ("A final
  run exists in the experiment: nothing logged."). The guard sits in the logging cell, not in
  the evaluation: the evaluation is seeded, and a reopened notebook can show the test results
  again without a second final run. The test part stands for production and is evaluated once
  (E11).
- **A deleted run is not seen** by the skips or the guard (`search_runs` reads active runs
  only). Deleting a run in the UI is how it is logged again; deleting the final run is that
  decision.
- **Commit before logging:** a run's `git_commit` is the code that ran only when `git_dirty`
  is `false`.
- **`04`'s selection:** each model's best run is its tuning run with the lowest
  `val/composite/<selection metric>`; a tie goes to the lower `study_seed`, compared as an
  integer (a parameter is text, below).
- **`mlflow.source.name`** is the notebook's absolute path under `marimo edit`, and the
  script's path when a script runs the notebook (a check script): read the `notebook` tag,
  the repository-relative path in both.

**Querying the runs** (`mlflow.search_runs`, `MlflowClient.search_runs`, the UI's search box):
- **A name with a slash needs backticks,** in `filter_string` and in `order_by`:
  ``metrics.`val/composite/rmse` < 4.5``; without them the query raises.
- **A parameter is text.** Only `=`, `!=`, `LIKE` and `ILIKE` apply to it
  (`params.n_trials > 3` raises); a `search_runs` DataFrame holds its column as strings, so
  `pd.to_numeric` before sorting or plotting. A metric is a number: filter and sort it.
- **A boolean parameter is `'True'` or `'False'`** (`params.use_exposure = 'True'`; `'true'`
  matches nothing). The tags the notebooks set as flags are lower case (`tags.final = 'true'`).
- **Add `attributes.status = 'FINISHED'`** to a query, as the notebooks' own skips do: a FAILED
  run stays in the store with the same `run_key` until it is deleted.
- **One quantity, two names.** A model's validation score is `val/composite/<metric>` in its
  tuning run and `val/<model>/<metric>/score` in the best runs' run; validation and test scores
  are `val/…` in one run and `test/…` in the other, so the UI cannot overlay them. The final
  run's `tables/val_against_test.csv` holds them side by side; or read both runs and join on
  the name with the `val/` or `test/` prefix stripped.

## 7. Comparing the 24 tuning runs in the UI

The run names are long and look alike in the runs list's default column width. To compare:
1. Search for `tags.stage = 'tuning' and attributes.status = 'FINISHED'`.
2. In the columns menu, add the tags `model` and `family`, the parameter `use_exposure`, and
   the metric `val/composite/<the selection metric>` (`val/composite/rmse` by default).
3. Sort by that metric, ascending.

`family` is a tag in all 24 runs: the column that spans both models. Each model's best run by
the selection metric is the run its `best.<model>` tag names in the evaluation runs (its id is the tag
`best.<model>.run_id`, §8). The code block at the end of this section opens the single best
tuning run across both models; §8 is the per-model route.

**Models and families are compared by score,** across runs:
- the runs table of the steps above (sorted, with the `family` and `use_exposure` columns);
- a bar chart of `val/composite/<the selection metric>`, one bar per run (24 bars);
- the runs grouped by the `family` or the `model` tag: grouped bars are one per tag value, each
  the mean of its runs (12 bars for `family`, 2 runs each).

**A hyperparameter across runs: a `best.*` scatter** (a `best.<name>` parameter on one axis, a
validation metric on the other). Each point is one study's best trial, with the family, the
exposure and the model differing between points, so it shows where the best settings landed,
not an effect. A scatter draws a run only if it holds the parameter, so it holds the runs of
one component role: the prefixed names (`total__`, `probability__`, `shared__` or a cohort's)
compare across the families of that role, never across the two models. For example
`best.total__num_leaves` against `val/composite/rmse` has the 8 runs whose total component is
a LightGBM family. Runs started with Model 1's sharing flags off (the check store) have no
`best.shared__…`; the numeric `best.*` values are read as numbers, the transformer names are
categories (a filter or a group, not an axis).

**A hyperparameter's effect is read within one study,** first in the tuning run's own figures:
`figures/slice.html` (each hyperparameter against the objective), `figures/param_importances.html`
(fANOVA's ranking) and `figures/optimization_history.html` (the best value over the trials), then
`trials.csv`.

**Any other Optuna chart of a study's trials.** Loaded with Optuna, a tuning run's journal
gives the study back: its trials and Optuna's charts (`plot_optimization_history`,
`plot_parallel_coordinate`, `plot_param_importances`, `plot_slice` in
`optuna.visualization`). The block reads the best tuning run's; it copies the file first,
since Optuna writes a lock file beside the file it opens:

```python
import tempfile

import mlflow
import optuna

# MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME name the store and the experiment.
(run,) = mlflow.search_runs(
    filter_string="tags.stage = 'tuning' and attributes.status = 'FINISHED'",
    order_by=["metrics.`val/composite/rmse` ASC"],
    max_results=1,
    output_format="list",
)
study_name = run.data.tags["run_key"].split("/")[1]
path = mlflow.artifacts.download_artifacts(
    run_id=run.info.run_id,
    artifact_path=f"study/{study_name}.log",
    dst_path=tempfile.mkdtemp(),
)
storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path))
study = optuna.load_study(study_name=study_name, storage=storage)
print(study_name, len(study.trials), "trials, best value", study.best_value)
optuna.visualization.plot_optimization_history(study)  # a notebook cell shows it
```

**What the UI check showed** (MLflow 3.16, chart view, the 3-trial check store). The eye icon in
the runs list's header sets how many runs the charts draw (first 10, first 20 or all); choose
all runs to see the 26. A scatter takes a `best.*` parameter on an axis and reads it as a
number; `best.total__num_leaves` against `val/composite/rmse` draws 8 points. A bar chart of
`val/composite/rmse` draws 24 bars; grouped by `family` it draws 12, the mean of each family's
runs. The grouped bars have no axis labels: the legend listed 5 of the 12 names and
colours repeat (hovering was not tried).

## 8. From an evaluation run to its tuning runs

An evaluation run holds, per model, two tags: `best.<model>.run_id`, the id of its best
tuning run (the direct link: `client.get_run(id)`; in the UI it is text, not a link), and
`best.<model>`, that run's study name. Behind its own key's folder, the study name is the
tuning run's `run_key`, the fallback when only the name is at hand. `04` takes both from the
FINISHED tuning run its read selected, so the run exists whenever an evaluation run does. The
block starts from the latest best runs' run, so it needs one logged:

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
    tuning_run = client.get_run(best_runs_run.data.tags[f"best.{model}.run_id"])
    print(model, tuning_run.info.run_name, tuning_run.info.status)
    # The fallback: the run_key search names the same run.
    run_key = f"{folder}/{best_runs_run.data.tags[f'best.{model}']}"
    (found,) = client.search_runs(
        [experiment.experiment_id],
        f"tags.run_key = '{run_key}' and attributes.status = 'FINISHED'",
    )
    assert found.info.run_id == tuning_run.info.run_id
```

The evaluation run does not need the tuning run to rebuild its models: it holds its own
`best_params` and components.

## 9. Rebuilding a model from a run

Each run kind holds what rebuilds its models with the components' `from_params` and the
notebook's builders, as the notebook builds them:

| Argument | A tuning run | The best runs' run, the final run |
|---|---|---|
| `best_params` | `best_params.json` | `best_params/<model>.json` |
| `family` | the parameter `family` (Model 2: `total_family`, `probability_family`) | `components/<model>.json`: each component's `family` |
| `use_exposure` | the parameter `use_exposure` | `components/<model>.json`: each count component's `use_exposure` |
| `feature_transformer_dict` | the notebook's `COUNT_FEATURE_TRANSFORMERS[family]` or `PROBABILITY_FEATURE_TRANSFORMERS[family]` at the run's `git_commit` | the same |
| the model | `build_*_model(components, y_min=prediction_floor, random_state=seed)`: `prediction_floor` from `settings.json` (every run kind), `seed` the parameter | the same |
| fit on | the training part | the training part (best runs); training and validation (final) |

`components.json` is the readable declaration, also a check of the rebuild: the rebuilt
components' `to_dict()` equals it. The candidates and the data are the notebook's, so check
out the run's `git_commit` and keep its data settings (`config_hash` must match).

The recipe below loads `04`'s names (the setup it shares with `03`, and `SELECTION_METRIC`,
which `03` no longer has) from a run with every button unclicked (nothing is tuned or logged).
`04` also reads the store while it loads, and rebuilds and checks its two best runs (§10, step
4) when the store holds the 24 tuning runs of its setup; on another store it stops at that read.
Every name the recipe uses is defined before the read, so the recipe runs either way. It
rebuilds both models of the latest best runs' run and of the final run and
the best tuning run by the selection metric, refits each, and asserts its composite scores
equal the run's metrics. It rebuilds the run kinds it finds and prints the ones it skips (the
final run before E11). Run on a 3-trial store, it was exact for all three run kinds. Save it
as a script and run it from the repository root like the script run in §2, with
`MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false` (else each artifact read prints a progress bar):

```python
import importlib.util
import os

import mlflow
import pandas as pd

# The notebook's names (the candidates, the builders, the data): every button unclicked.
spec = importlib.util.spec_from_file_location(
    "inspection_notebook", "notebooks/04_model_inspection.py"
)
notebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notebook)
_, names = notebook.app.run()

client = mlflow.MlflowClient()  # MLFLOW_TRACKING_URI
experiment = client.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])


def first_run(filter_string, order_by="attributes.start_time DESC"):
    """The first FINISHED run of the experiment matching `filter_string`, by `order_by`
    (the latest by default); None when there is none."""
    runs = client.search_runs(
        [experiment.experiment_id],
        f"{filter_string} and attributes.status = 'FINISHED'",
        order_by=[order_by],
        max_results=1,
    )
    return runs[0] if runs else None


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
        "y_min": float(read_json(run, "settings.json")["prediction_floor"]),
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
    if run is not None:
        assert run.data.params["config_hash"] == names["config_hash"], "another data version"

if tuning_run is None:
    print("skipped: no tuning run")
else:
    model_name = tuning_run.data.tags["model"]
    scores = composite_scores(rebuild_model(tuning_run, model_name), train, val)
    for metric, score in scores.items():
        assert score == tuning_run.data.metrics[f"val/composite/{metric}"]
    print("tuning run", tuning_run.info.run_name, scores)

for kind, run, fit_on, score_on, dataset in (
    ("best runs' run", best_runs_run, train, val, "val"),
    ("final run", final_run, train_val, test, "test"),
):
    if run is None:
        print(f"skipped: no {kind}")
        continue
    for model_name in ("independent_cohorts", "total_times_probability"):
        scores = composite_scores(rebuild_model(run, model_name), fit_on, score_on)
        for metric, score in scores.items():
            assert score == run.data.metrics[f"{dataset}/{model_name}/{metric}/score"]
        print(run.info.run_name, model_name, scores)
```

## 10. Checking the logging, step by step

A complete run on a store of its own, to see that every run kind is logged as this guide
says. Run every command from the repository root.

**Before you start:**
- **The test part.** Step 6 evaluates the test part, which is meant to be looked at once
  (E11). Stop after step 5 if this is not that look.
- **The studies.** `03`'s Tune click loads a study stored in `studies/` and tunes one that is not
  stored. Loading takes minutes; tuning takes far longer.
- **Commit first** (§6): a run then names the code that produced it (`git_dirty` `false`).

1. **Point `.env` at a check store.** Replace its three MLflow lines (keep the others):

   ```
   MLFLOW_TRACKING_URI=sqlite:///.mlflow-check/mlflow.db
   MLFLOW_EXPERIMENT_NAME=age-group-check
   AGE_GROUP_MLFLOW_ARTIFACT_LOCATION=.mlflow-check/mlartifacts
   ```

   `.mlflow-check/` is git-ignored (the patterns `mlflow.db` and `mlartifacts/`). Any folder
   works, except the root's `mlflow.db` and `mlartifacts/` (§2).

2. **Open the tuning notebook:**

   ```bash
   uv run --env-file .env --group notebook --group tracking \
     marimo edit notebooks/03_hyperparameter_tuning.py
   ```

   The setup cells run by themselves; nothing is tuned or logged. In the MLflow cell's
   output, check the line `experiment=age-group-check tracking_uri=sqlite:///.mlflow-check/mlflow.db`.
   A `ValueError` there names the variable to fix.

3. **Click the two Tune buttons,** one after the other. Under each Tune cell:

   | Click | You should see |
   |---|---|
   | "Tune the independent cohorts model" | "Logged 8 tuning runs, skipped 0 (already logged)." |
   | "Tune the total times probability model" | "Logged 16 tuning runs, skipped 0 (already logged)." |

   Close `03` (note the messages first); step 7 reopens it.

4. **Open the inspection notebook** on the same store:

   ```bash
   uv run --env-file .env --group notebook --group tracking \
     marimo edit notebooks/04_model_inspection.py
   ```

   Its setup cells run by themselves, and its read (section 5) needs no click: it finds the
   24 FINISHED tuning runs of its setup, shows them in a table and selects, rebuilds and checks
   each model's best run. Nothing is logged yet. Two outcomes other than success:
   - **A stop, no error:** the store holds other than 8 + 16 tuning runs of this setup (FINISHED,
     the same study folder and objective, each `run_key` once). The cell shows the counts per
     model (`expected`, `finished_runs`, `distinct_run_keys`) and nothing after it runs, so a
     partial store is never selected from and no run is logged. Tune the missing runs in `03`
     (a changed `n_trials`, split or seed changes the study folder, so the runs of `03`'s other
     setting do not count).
   - **A `ValueError` naming a tuning run,** from the guard that checks the selected run's
     rebuild against what the run logged:
     - *its dataset inputs differ from this notebook's:* the data or the split changed since the
       run (the run's training or validation digest is not `04`'s);
     - *its components differ from their rebuild:* the feature transformers of `04` are not the
       ones the run declared in `components.json`;
     - *its scores differ from its rebuild's:* the refit does not reproduce `scores.csv`
       exactly: the prediction floor, the seed or a library changed (or the `scores.csv` rows
       differ).

     A fourth `ValueError` names a run whose `components.json` or `best_params.json` holds a
     name `04` lacks, or lacks a hyperparameter `04` has (a transformer renamed or removed, a
     cohort or hyperparameter added). Each means the run belongs to other
     code or data than `04`'s: check out the run's `git_commit`, or tune again.

5. **Click "Inspect the best runs"** (the bootstrap, the permutation importance and SHAP:
   a few minutes). Under the section's last cell: "Logged the best runs' run: 54 metrics,
   3 tables, 18 figures."

6. **Click "Evaluate on the test part"** (the one look; a few minutes). Under the
   notebook's last cell: "Logged the final run: 54 metrics, 7 tables, 33 figures."

7. **Check that nothing is logged twice:**
   - reopen `03` (step 2's command; its setup cells run by themselves) and click a Tune button
     again (it loads the stored studies, which takes minutes): "Logged 0 tuning runs, skipped 8
     (already logged)." (or 16);
   - in `04`, re-run the cell under step 5's message (its ▶ button): "The best runs' run is
     logged: skipped.";
   - re-run `04`'s last cell: "A final run exists in the experiment: nothing logged."

8. **Open the MLflow UI** (§3: the command, then **Model training**). In the experiment
   `age-group-check` you should find:

   | Search | Runs | Open first |
   |---|---|---|
   | `tags.stage = 'tuning'` | 24 | `trials.csv` (a row per trial) |
   | `tags.stage = 'best_runs'` | 1 | `tables/best_runs_val_scores.csv`, then a figure |
   | `tags.stage = 'final'` | 1 | `tables/test_scores.csv` |

   Every run's status is FINISHED (a run that failed while logging is FAILED and holds no
   metrics). §4 and §5 list what each run holds; §7 compares the 24 tuning runs.

9. **Rebuild the models from the runs** (§9's recipe, saved as a script). It must print the
   tuning run, the best runs' two models and the final run's two models, with no "skipped"
   line and no `AssertionError`: each rebuilt model's scores equal its run's metrics.

10. **Clean up.** Close the notebooks and the UI, delete the store (`rm -rf .mlflow-check`),
   and put `.env`'s three lines back (`.env.example` has them).
