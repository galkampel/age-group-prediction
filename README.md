# Age Group Prediction

A configurable synthetic-data project for estimating resident children in new
residential buildings by age cohort.

## Current Simulator

The simulator generates static apartment-level outcomes and returns one final
building-level DataFrame:

- [Simplified model specification](docs/SIMPLIFIED_MODEL_PLAN.md) defines the
	current behavior and formulas.
- [Compact simulator migration plan](docs/COMPACT_SIMULATOR_MIGRATION_PLAN.md)
  records the compact component migration and its validation gates.
- [Marimo research workflow migration plan](docs/MARIMO_MIGRATION_PLAN.md)
	records the completed Jupyter cutover, package setup, notebook structure,
	validation gates, and model assignments.
- [EDA and predictive modeling plan](docs/EDA_AND_PREDICTIVE_MODELING_PLAN.md)
	defines the immediate EDA deliverable and modeling handoff.
- [Modeling guide](docs/MODELING_GUIDE.md) explains the supported pipeline,
  public APIs, configuration, evidence, and invariants.
- [MLflow experiments guide](docs/MLFLOW_EXPERIMENTS_GUIDE.md) covers scratch
  setup, run layout, searching, final-run semantics, and model loading.
- [Module reference](docs/MODULE_REFERENCE.md) lists every source module with
  its responsibility, public API, and dependencies.
- [Documentation index](docs/README.md) distinguishes current documents from
	advanced references and legacy source material.

The compact implementation has five generation components:
`NeighborhoodSimulator`, `BuildingSimulator`, `ApartmentSimulator`,
`TotalChildrenSimulator`, and `CohortCompositionSimulator`.
`StudentPopulationSimulator(config).run()` returns the final building-level
table and retains named intermediate tables for research inspection through
`simulator.last_result`. Random effects are not exported as DataFrame columns.

```python
from student_simulator import StudentPopulationSimulator, load_simulation_config

config = load_simulation_config("configs/simulation.toml")
simulator = StudentPopulationSimulator(config)
final_df = simulator.run()
details = simulator.last_result
assert details is not None
```

## Research Workflow

The research source is a pair of version-controlled marimo notebooks:

- `notebooks/01_eda.py` implements the complete EDA checklist and findings
	handoff;
- `notebooks/02_model_fitting.py` runs the accepted cross-validation,
  selection, and guarded final-evaluation workflow.

The Jupyter-to-marimo cutover is complete. `research.ipynb` has been removed;
marimo `.py` notebooks are the sole research source.

### Run The Modeling Notebook

1. Install the notebook and tracking dependencies once:

   ```bash
   uv sync --group notebook --group tracking
   ```

2. Create your local environment file once:

   ```bash
   cp .env.example .env
   ```

   `.env` is git-ignored. It sets `PYTHONPATH=src` (macOS can hide this
   checkout's editable-install `.pth` files) and points MLflow at a local
   scratch store in `.mlflow-local/`, also git-ignored. Without a scratch
   store, the notebook logs to the canonical `mlflow.db` and `mlartifacts/`,
   which must not change (see
   [the MLflow guide, Section 2](docs/MLFLOW_EXPERIMENTS_GUIDE.md#2-keep-scratch-and-canonical-stores-separate)).

3. Open the notebook in the browser editor from the repository root:

   ```bash
   uv run --env-file .env --group notebook --group tracking \
     marimo edit notebooks/02_model_fitting.py
   ```

   marimo prints a local URL and opens it. Add `--headless` to skip opening a
   browser or `--port <n>` to choose the port. The EDA notebook opens the same
   way with `marimo edit notebooks/01_eda.py`. `uv run` reads `.env` only when
   given `--env-file`. For a one-off throwaway store instead, export the
   Section 2 variables from the MLflow guide and prefix the command with
   `PYTHONPATH=src` in place of `--env-file .env`. To use the VS Code marimo
   extension, select `.venv/bin/python` and launch VS Code from a shell where
   the same variables are set.

4. Read the setup cells. They build the canonical population (150
   neighborhoods), the modeling table, the candidate registry, and training
   folds. They also load the persisted `artifacts/lockbox/split_manifest.json`
   (creating it if absent) and replay it in memory without displaying or
   evaluating any holdout row. Nothing is fitted yet.

5. Click **Run cross-validation** to fit and log the three-approach comparison
   to the scratch store. In the canonical run, fold fitting alone took about 12
   minutes (about 10 for the Bayesian model); evaluation, bootstrap intervals,
   importance, and logging add more. Inspect the result from the repository
   root with
   `uv run --env-file .env --group tracking mlflow ui --backend-store-uri sqlite:///.mlflow-local/mlflow.db`.

6. **Do not use the final-evaluation checkbox or button.** The canonical
   one-time lockbox evaluation has already run. The duplicate guard only sees
   runs in the current MLflow store, so in a scratch store it cannot detect the
   canonical run and would evaluate the holdout again.

`uv run --env-file .env notebooks/02_model_fitting.py` runs the notebook as a
script: setup only, with both buttons unclicked. marimo's runtime
configuration also loads `.env` by default when it finds `pyproject.toml`, so
never put canonical MLflow settings in `.env`.

### Run The Tests

Run the fast suite with:

```bash
uv run pytest -m "not calibration and not slow"
```

Run coefficient recovery after installing the validation dependency group:

```bash
uv sync --group validation
uv run pytest -m slow tests/validation/test_recovery.py -v
```

## Bayesian Conditional Model

`BayesianConditionalModel` fits two separate Pyro NUTS posteriors: a
hierarchical NB2 model for total children with a fixed log-apartment exposure
offset and neighborhood effects, followed by a Dirichlet-multinomial
age-composition model conditioned on observed totals. `torch` and `pyro-ppl`
are core project dependencies (not an optional group), so no extra `uv sync`
step is needed to use it:

```python
import numpy as np

from age_group_prediction import (
	DEFAULT_TOTAL_FEATURE_SPEC,
	BayesianConditionalModel,
	PredictionConfig,
)

model = BayesianConditionalModel().fit(
	train_df,
	feature_spec=DEFAULT_TOTAL_FEATURE_SPEC,
	rng=np.random.default_rng(42),
)
prediction = model.predict(
	evaluation_df,
	prediction_config=PredictionConfig(
		n_predictive_draws=500,
		interval_levels=(0.8, 0.95),
	),
	rng=np.random.default_rng(43),
)
```

The default reduced profile warns when convergence thresholds fail; the full
profile rejects the fit and attaches the failing diagnostics to the raised
error. The full profile has passed its strict policy on the 251-building
simulator dataset; see [the guide](docs/BAYESIAN_CONDITIONAL_MODEL.md#full-profile-acceptance-run).
Prior-predictive checks run before both likelihood fits, using their own
configured action independent of the convergence policy. Posterior prediction samples totals first and age composition second,
so integer cohort draws reconcile exactly without repair. Known neighborhoods
use their fitted effects; unseen neighborhoods receive fresh effects from the
learned population distribution, one independent draw per row rather than per
unique unseen neighborhood, and the count of affected rows is recorded in
model metadata. This is a documented, metadata-visible fallback path whose
claims are already limited to known neighborhoods; see [the Bayesian
conditional model guide](docs/BAYESIAN_CONDITIONAL_MODEL.md#implementation-cautions)
for details.

## Saving And Reloading Fitted Models

Every model (`DirectCohortModel`, `IndependentTotalProbabilityModel`,
`BayesianConditionalModel`) exports a JSON state bundle: configuration, fitted
preprocessing, and fitted parameters as plain JSON values. It contains no
pickles and no training rows. The training frame is identified only by its
hashes. The layout is documented in `src/age_group_prediction/state_bundle.py`.

```python
import json

from age_group_prediction import DirectCohortModel

bundle = model.to_state_bundle()
text = json.dumps(bundle)

reloaded = DirectCohortModel.from_state_bundle(json.loads(text))
point_prediction = reloaded.predict(evaluation_df)
```

The bundle alone reproduces point predictions, pointwise log probabilities, and
parametric distributions exactly. `DirectCohortModel` and
`IndependentTotalProbabilityModel` build predictive draws and intervals by
refitting on neighborhood-cluster bootstrap resamples, so requesting them needs
the original training frame. Pass it as
`from_state_bundle(bundle, train_df=train_df)`; it is checked against the
recorded hashes and a different frame is refused. `BayesianConditionalModel`
draws from its stored posterior and never needs the frame. A bundle written by
another bundle format, model class, or model `implementation_version` is
refused.

A bundle contains no rows, but it does contain fitted summary statistics.
These include feature ranges, tree split thresholds and node counts, and the
Bayesian model's neighborhood IDs. Store it with the same care as the data
it was fitted on.

## Tracking Experiments With MLflow

Tracking is optional: install it with `uv sync --group tracking`. It logs a
*finished* cross-validation comparison. The experiment and model code never
import MLflow, so a tracked and an untracked comparison produce the same
result. Import everything from `age_group_prediction.tracking`; its submodules
are internal. See the
[MLflow experiments guide](docs/MLFLOW_EXPERIMENTS_GUIDE.md) for a validated
scratch example, exact tags and artifacts, search filters, failure semantics,
and pyfunc loading.

Configuration comes only from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Where runs are stored |
| `MLFLOW_EXPERIMENT_NAME` | `age-group-prediction` | The experiment that holds the runs |
| `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` | `mlartifacts/` beside a SQLite database file | Where artifacts are stored; with a tracking server, the server decides |

MLflow has no variable for a client-side artifact location, so the third one
belongs to this project. If an experiment already exists with a different
artifact location, tracking refuses to use it rather than write elsewhere;
pick a new experiment name instead.

```python
from age_group_prediction.experiment import run_cross_model_validation
from age_group_prediction.tracking import (
    TrackingContext,
    log_experiment_result,
    tracked_comparison,
)

context = TrackingContext(source_revision="<git sha>", source_dirty=False)
with tracked_comparison(context) as handle:
    result = run_cross_model_validation(
        outer_train_df,
        # ...split manifest, folds, candidates, policies, seed...
        capture_artifacts=True,
    )
    log_experiment_result(result, handle=handle, train_df=outer_train_df)
```

Run the comparison inside the block, so a failure is recorded too: the run
ends `FAILED` (or `KILLED` on Ctrl-C) with `failure_type` and
`failure_message` tags. To log a result computed earlier, call
`log_cross_validation_experiment(result, context)`. The package never runs
Git, so pass the source revision yourself; MLflow's automatically inferred Git
tags are removed because they describe the last commit, not uncommitted code.

`capture_artifacts=True` saves every candidate's state bundle on every fold.
Before the fitted model is released, it checks that the bundle reloads
without the training frame and reproduces the model's predictions exactly.
Tracking refuses a result without this evidence. It is off by default because
every bundle is held in memory. Tracking also refuses a candidate whose
configured `family` disagrees with the family its fitted model reports, so a
run is never tagged with a family it did not fit.

Browse runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Each comparison is one parent run with one nested child run per candidate, so
Poisson, NB2 and Normal `DirectCohortModel` candidates are separate sibling
runs:

- **Parent:** data, split and seed tags; run settings, a split summary (source
  table hash, sizes, holdout fractions; no building IDs) and package versions
  as params; the selection freeze, provenance, fold identities (the only copy
  of each fold's building IDs), and the comparison-wide metric, bootstrap,
  coverage and importance tables. `metric_comparability.json` states which
  candidates score log masses or log densities, each candidate's pointwise
  log-probability scope, and that `predictive_nll` is never ranked across
  approaches, so do not compare those metrics across runs of different
  approaches in the MLflow UI.
- **Child:** candidate, family, LightGBM objective and selection-outcome tags;
  `fold/{metric}/{target}/{level}` metrics stepped by fold index, plus
  `cv_mean/`, `cv_std/`, `duration/`, `tuning/` and Bayesian `diag/` metrics;
  each fold's metadata, runner seeds, gzipped state bundle and reload check;
  Optuna trial tables; and this candidate's predictions, fold metrics,
  calibration and importance.

A comparison is complete only when its **parent** run is `FINISHED` **and**
tagged `evidence_complete=true`. That tag is each run's last write, children
before the parent. A failed or interrupted run is tagged `false`. A process
killed outright leaves its runs `RUNNING`, possibly with some children already
complete, which is why a complete child never implies a complete comparison. Bayesian effective sample sizes
are charted clamped at zero with an `ess_valid` flag, because Pyro's estimator
can go negative; the raw value stays in the fold's metadata. Fold models are
evaluation evidence stored as JSON bundles: no MLflow model flavor, registered
model, or pickle is written.

## Cross-Family Selection And Final Evaluation

The Gate 6/7 comparison above freezes one winner *per approach*
(`DirectCohortModel`, `IndependentTotalProbabilityModel`,
`BayesianConditionalModel`), not one winner overall. Gate 8 adds the
remaining pieces: choosing across approaches, refitting the winners on all
of the training data, evaluating them once on a held-out test partition, and
logging that as a second, linked MLflow run. The ordered workflow and its
public contracts are documented in the
[modeling guide](docs/MODELING_GUIDE.md); operational tracking details are in
the [MLflow experiments guide](docs/MLFLOW_EXPERIMENTS_GUIDE.md).

**Split manifest persistence.** The one-time outer test/train split is
computed once and persisted as JSON at `artifacts/lockbox/split_manifest.json`
(git-ignored). `persist_split_manifest(path, manifest)` writes the file only
if it is absent; if a manifest is already there, its parsed contents (every
recorded field, including both building-ID lists) must equal the one given, or
the write is refused, so a later run can never silently relabel which
buildings are held out. Formatting differences in the file itself are not
compared. `load_split_manifest(path)` reads it
back, and every later access replays it through `replay_split_manifest(...)`,
never by re-splitting.

**Cross-family selection.** `experiment.select_cross_family_winner(cv_result)`
reads only the three frozen winners and their aggregate cross-validation
metrics — no test data. It ranks the two conditional approaches
(`IndependentTotalProbabilityModel`, `BayesianConditionalModel`) by
`joint_predictive_nll`, then compares that winner against `DirectCohortModel`
lexicographically by composition log loss, then mean cohort RMSE, then mean
cohort MAE, then candidate ID. The resulting `CrossFamilySelection` is
attached to the Gate 6 freeze with
`freeze.with_cross_family_selection(selection)` *before* any holdout row is
read, producing a "pretest freeze" that a later test result can only extend,
never revise.

**CV-to-final run linkage.** The final action never reopens the completed
Gate 6/7 comparison run. `tracking.run_final_evaluation(...)` is the single
guarded entry point: it refits the three frozen winners on the complete
training partition (the Bayesian winner forced to the full NUTS profile and
its strict, `action="error"` diagnostic policy), then opens a *new*,
top-level MLflow parent tagged `run_role=final_evaluation`,
`source_cv_run_id=<the Gate 6/7 parent's run ID>`, and
`manifest_fingerprint=<the same split's fingerprint>`. Before refitting, it
requires the pretest freeze's decision to be exactly what
`select_cross_family_winner` produces from the supplied CV result, recorded
on that CV result's own freeze, so a hand-built or edited decision is refused.
A second complete final run for the same source run and manifest is refused,
including when the earlier run was deleted.

Within one MLflow experiment, once any final run has opened the lockbox for a
manifest, a later run on that manifest is accepted only as a retry of the
identical attempt. A failed attempt keeps whatever test evidence it logged, and
a deleted attempt still counts. An identical attempt has the same source CV run,
the same cross-family decision, and the same `final_attempt_fingerprint`. That
fingerprint is a hash, computed after the refit and before the run opens, of:

- the pretest freeze;
- each refit's seeds, training hashes, constructor configuration
  (`BaseAgeGroupModel.configuration_record`) and recorded dependency versions;
- the selected candidates' descriptors and prediction settings;
- the evaluation settings and the schema.

A retry that re-tunes, re-priors or re-seeds the refit, or runs under other
library versions, is therefore refused. Model source code is not hashed, so a
retry assumes the same package code. A final run that opened the lockbox
before the fingerprint existed carries no fingerprint tag, so every later
attempt on its manifest is refused. The refit also requires the freeze's master
seed to equal the CV provenance seed.
The check cannot see other experiments or tracking stores, so pointing
`MLFLOW_EXPERIMENT_NAME` elsewhere is outside this guarantee. Each final parent
is tagged `cross_family_decision_hash` and `final_attempt_fingerprint`, and a
retry is also tagged `retry_of_run_ids`.

The parent opens `test_lock_status=locked`,
logs the pretest freeze, and only then flips to `test_lock_status=opened`, so
the lockbox opens only once that evidence is durably recorded. Manifest
replay, one-time holdout evaluation, and logging all happen inside that one
guarded call, and the evaluator refuses a replay whose holdout IDs differ from
the manifest's. One nested child run per approach winner is tagged
`role=selected` or `role=comparator` (a comparator is a predeclared
alternative, not a rejected candidate) and `test_lock_status=opened`, because
it holds test evidence. Each child, then the parent, is marked
`evidence_complete=true` last.

**Pyfunc output contract.** Each full-training refit is logged as a
[models-from-code](https://mlflow.org/docs/latest/ml/model/models-from-code/)
`mlflow.pyfunc` model (`tracking/pyfunc_model.py`), never a pickle: its
`load_context` reads the refit's JSON state bundle artifact, validates the
bundle header, and dispatches to the matching model class's
`from_state_bundle(...)`, without the training frame. `predict` accepts a
target-free DataFrame (building ID, features, exposure, neighborhood) and
returns one row per building with:

- `building_id`
- `total_mean`
- one `{cohort}_mean` column per cohort
- one `{cohort}_probability` column per cohort

Immediately after logging, every pyfunc is reloaded in-process with
`mlflow.pyfunc.load_model(...)` and required to reproduce exactly a fresh
point prediction from the live refit on the same input. The pyfunc `predict`
API takes no seed argument, so neither side passes one, mirroring how
`experiment.artifacts.capture_fold_artifact` proves a fold bundle reloads.
That fresh prediction equals the evaluator's logged predictions whenever no
holdout building sits in an unseen neighborhood, which the known-neighborhood
split guarantees. The in-process reload cannot prove the model loads
elsewhere, so a separate test loads a logged model in a fresh interpreter,
where only the artifact's `code_paths` can supply the package.

`age_group_prediction.models` imports the LightGBM model before the torch
model. With torch's OpenMP runtime loaded first, LightGBM segfaults restoring
a booster in a new process. The canonical `direct-poisson` model was logged
before that fix, so its bundled code needs `import lightgbm` before
`mlflow.pyfunc.load_model(...)` in a fresh process. Each model's tagged
`final_model_uri` is on its child run.

**Notebook controls.** `notebooks/02_model_fitting.py` is a thin marimo
client: its modeling steps call only `age_group_prediction` package APIs,
never a model's own `fit`/`predict`/`evaluate` (its last cell uses the MLflow
client only to display run links). A plain script run
(`uv run notebooks/02_model_fitting.py`) performs setup only — building the
canonical population and modeling table, creating or loading the persisted
manifest and replaying it in memory to obtain the training partition,
building the candidate registry and validation folds — and never displays or
evaluates a holdout row, starts cross-validation, fits the full Bayesian
model, or opens an MLflow run. Cross-validation and the final action each need an
explicit `mo.ui.run_button` click; the final action additionally requires a
confirmation checkbox, since it is one-time and cannot be undone for a given
manifest.