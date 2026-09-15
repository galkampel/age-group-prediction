# MLflow Experiments Guide

This guide covers the optional MLflow adapter for cross-validation evidence and
final models. For the modeling pipeline itself, start with
[MODELING_GUIDE.md](MODELING_GUIDE.md).

## 1. Install And Configure Tracking

Install the optional dependency group:

```bash
uv sync --group tracking
```

Import only from `age_group_prediction.tracking`; its submodules are internal.
`resolve_tracking_settings()` reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend store |
| `MLFLOW_EXPERIMENT_NAME` | `age-group-prediction` | Experiment name |
| `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` | `mlartifacts/` beside a local SQLite database | Artifact root; a nonlocal tracking server decides when unset |

Local paths and `file:` URIs are normalized to absolute file URIs. If an
experiment already exists with a different artifact location, the adapter
refuses it because MLflow cannot move an experiment's existing artifacts. Use
the existing location or a new experiment name.

## 2. Keep Scratch And Canonical Stores Separate

Tutorials, tests, and exploratory runs must use a scratch store. The following
shell setup creates an isolated location and leaves the canonical `mlflow.db`,
`mlartifacts/`, and lockbox untouched:

```bash
export SCRATCH_DIR="$(mktemp -d /tmp/age-group-mlflow-guide.XXXXXX)"
export MLFLOW_TRACKING_URI="sqlite:///$SCRATCH_DIR/mlflow.db"
export MLFLOW_EXPERIMENT_NAME="age-group-guide-$(date +%Y%m%d-%H%M%S)"
export AGE_GROUP_MLFLOW_ARTIFACT_LOCATION="$SCRATCH_DIR/mlartifacts"
export MLFLOW_DISABLE_AGENT_HINT=1
```

For a persistent local store, copy `.env.example` to the git-ignored `.env`
instead and pass `uv run --env-file .env ...` from the repository root; it
uses `.mlflow-local/` (see
[Run The Modeling Notebook](../README.md#run-the-modeling-notebook)).

Print `$SCRATCH_DIR` and keep it for the UI command below. Never point a guide,
test, or exploratory process at the repository defaults. Changing the
experiment name or tracking URI also changes the scope in which duplicate
final attempts can be detected; that is isolation for scratch work, not a way
to repeat canonical evaluation.

## 3. Launch The UI

With the scratch variables still set:

```bash
uv run --group tracking mlflow ui \
  --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --port 5000
```

Open `http://127.0.0.1:5000`, select the value of
`MLFLOW_EXPERIMENT_NAME`, and group nested runs by parent. Choose another port
if 5000 is occupied.

## 4. Run Cross-Validation

### From The Notebook

Open [notebooks/02_model_fitting.py](../notebooks/02_model_fitting.py) with the
scratch environment configured before starting the kernel:

```bash
PYTHONPATH=src uv run --group notebook --group tracking \
  marimo edit notebooks/02_model_fitting.py
```

Step-by-step instructions, including expected run time, are in the
[README](../README.md#run-the-modeling-notebook). The notebook does
setup without fitting. The scratch variables redirect only MLflow: setup always
loads the repository's persisted `artifacts/lockbox/split_manifest.json`
(creating it if absent) and replays it in memory to obtain the training
partition, without displaying or evaluating any holdout row. Click **Run cross-validation** to execute the canonical
three-approach comparison and log it. The notebook passes
`capture_artifacts=True`, which tracking requires.

Do not click the final-evaluation controls during a tutorial. They replay the
canonical persisted manifest and open the one-time lockbox.

### From Python: Bounded Scratch Example

The following example uses a small synthetic subset, two outer validation
folds, one tuning trial, and only the direct-cohort candidate. The
direct-cohort model's inner tuning folds stay at the package default of five
(`DEFAULT_FOLD_CONFIG`); `configs/modeling.toml` does not control them. It creates no persisted split
manifest and never calls a final-evaluation API. Run it only after exporting
the scratch variables from Section 2.

```bash
PYTHONPATH=src uv run --frozen --group tracking python - <<'PY'
from dataclasses import replace

import numpy as np

from age_group_prediction import (
    build_modeling_table,
    load_experiment_config,
    make_validation_folds,
    split_known_neighborhood_buildings,
)
from age_group_prediction.experiment import (
    build_canonical_candidate_registry,
    run_cross_model_validation,
)
from age_group_prediction.tracking import (
    TrackingContext,
    log_experiment_result,
    tracked_comparison,
)
from student_simulator import StudentPopulationSimulator, load_simulation_config

config = load_experiment_config("configs/modeling.toml")
quick_tuning = replace(config.tuning, n_trials=1)
config = replace(
    config,
    folds=replace(config.folds, n_folds=2),
    evaluation=replace(config.evaluation, bootstrap_replicates=2),
    tuning=quick_tuning,
    direct_cohort=replace(
        config.direct_cohort,
        tuning=quick_tuning,
        bootstrap_replicates=2,
    ),
)

source_df = StudentPopulationSimulator(
    load_simulation_config("configs/simulation.toml")
).run()
neighborhoods = sorted(source_df["neighborhood_id"].unique())[:8]
scratch_source = source_df[source_df["neighborhood_id"].isin(neighborhoods)]
modeling_table = build_modeling_table(scratch_source)

split = split_known_neighborhood_buildings(
    modeling_table,
    config=config.outer_split,
    rng=np.random.default_rng(42),
)
fold_plan = make_validation_folds(
    split.train_df,
    config=config.folds,
    rng=np.random.default_rng(43),
)
registry = build_canonical_candidate_registry(config)
candidate = registry.candidate_by_id("direct-poisson")
policy = next(
    item
    for item in registry.selection_policies
    if item.approach == "DirectCohortModel"
)

context = TrackingContext(
    run_name="guide-direct-only",
    extra_tags={"purpose": "documentation-smoke"},
)
with tracked_comparison(context) as handle:
    result = run_cross_model_validation(
        split.train_df,
        split_manifest=split.manifest,
        validation_folds=fold_plan.folds,
        candidates=(candidate,),
        selection_policies=(policy,),
        experiment_config=config,
        required_approaches=("DirectCohortModel",),
        capture_artifacts=True,
    )
    log_experiment_result(result, handle=handle, train_df=split.train_df)

print(handle.run_id)
PY
```

Keep the comparison inside `tracked_comparison`. Then failures and interrupts
have a parent run instead of disappearing. If a complete result already exists
in memory, `log_cross_validation_experiment(result, context, train_df=...)` is
the convenience entry point, but it cannot record how computation failed.

## 5. Understand The CV Run Layout

One comparison is a top-level parent with `run_role=comparison`; each candidate
is a nested child with `run_role=candidate`.

The parent carries the common tags `manifest_fingerprint`,
`training_data_hash`, `training_schema_hash`, `master_seed`,
`master_seed_source`, `test_lock_status`, and optional caller source tags. Its
artifacts are:

- `freeze.json`
- `provenance.json`
- `selections.json`
- `fold_identities.json`
- `metric_comparability.json`
- `aggregate_metrics.csv`, `bootstrap_intervals.csv`, `fold_coverage.csv`, and
  `importance_summary.csv` when their tables have rows

Each child adds `candidate_id`, `approach`, `model_class`, `selection_role`,
`selected`, optional `family` and `objective_family`, and an optional
`rejection_reason`. Child artifacts include:

- `folds/fold_{index}/metadata.json`
- `folds/fold_{index}/seeds.json`
- `folds/fold_{index}/state_bundle.json.gz`
- `folds/fold_{index}/reload_check.json`
- `tuning/fold_{index}/{component}_trials.csv` when tuning evidence exists
- optional `search_space.json`
- candidate slices in `predictions.csv`, `fold_metrics.csv`, `calibration.csv`,
  and `importance.csv` when rows exist

Fold bundles are evaluation evidence, not MLflow LoggedModels. Loadable pyfunc
models are created only from final full-training refits.

## 6. Read Metrics And Completeness

Candidate metric keys are stable:

| Pattern | Meaning |
|---|---|
| `fold/{metric}/{target}/{aggregation_level}` | Fold value, with MLflow step equal to fold index |
| `cv_mean/{metric}/{target}/{aggregation_level}` | Mean across folds |
| `cv_std/{metric}/{target}/{aggregation_level}` | Standard deviation across folds |
| `duration/{fit|predict|evaluate}_seconds` | Operation duration; evaluation includes prediction time |
| `tuning/{component}/best_value` | Best tuning objective |
| `diag/{stage}/max_rhat` | Worst finite R-hat |
| `diag/{stage}/min_ess_clamped` | ESS clamped to zero for charting |
| `diag/{stage}/ess_valid` | `1.0` only for a finite positive ESS |
| `diag/{stage}/policy_passed` | Diagnostic-policy outcome as `0.0` or `1.0` |

Treat a run as complete only when its MLflow status is `FINISHED` and its
`evidence_complete` tag is `true`. That tag is the last write: children finish
before their parent. An ordinary exception produces `FAILED`; Ctrl-C produces
`KILLED`; both record `failure_type`, `failure_message`, and
`evidence_complete=false`. A hard process termination can leave `RUNNING`
without a completeness tag.

Never infer parent completeness from a finished child. Also read
`metric_comparability.json` before comparing likelihood metrics: per-target
`predictive_nll` is not comparable across all three approaches.

## 7. Search And Compare Runs

Use the public MLflow client after resolving the same settings:

```python
import mlflow

from age_group_prediction.tracking import resolve_tracking_settings

settings = resolve_tracking_settings()
client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
experiment = client.get_experiment_by_name(settings.experiment_name)
assert experiment is not None

complete_comparisons = client.search_runs(
    [experiment.experiment_id],
    filter_string=(
        "tags.run_role = 'comparison' and "
        "tags.evidence_complete = 'true'"
    ),
)
```

Useful exact filters include `tags.run_role = 'candidate'`,
`tags.candidate_id = 'direct-poisson'`,
`tags.test_lock_status = 'opened'`, and
`tags.source_cv_run_id = '<comparison-run-id>'`. Candidate children carry
MLflow's `mlflow.parentRunId` tag. The final duplicate/retry guards search all
lifecycle stages, so deleting a run does not erase the fact that its evidence
was seen.

## 8. Understand Final-Run Semantics

The notebook first computes `select_cross_family_winner(cv_result)` from CV
evidence and creates a pretest freeze. The guarded
`age_group_prediction.tracking.run_final_evaluation` function then:

1. verifies the decision against the source CV result;
2. refits all three within-approach winners on outer training data;
3. refuses a Bayesian refit that did not run the full profile under its strict
   diagnostic policy (the registry's final-refit factory is what forces that
   profile);
4. fingerprints the exact refits and scoring settings;
5. opens a new top-level run linked by `source_cv_run_id`;
6. logs `pretest_freeze.json` while `test_lock_status=locked`;
7. flips the parent to `test_lock_status=opened` before replaying the holdout;
8. logs final evidence and marks children, then parent, complete.

Final parents use `run_role=final_evaluation` and tags including
`manifest_fingerprint`, `cross_family_rule_version`, `selected_candidate_id`,
`selected_approach`, `cross_family_decision_hash`, and
`final_attempt_fingerprint`. An allowed identical retry also has
`retry_of_run_ids`.

Final parent artifacts are `pretest_freeze.json`, `split_manifest.json`,
`finalized_freeze.json`, `cross_family_rule.json`, `comparability.json`,
`provenance.json`, and `test_metrics.csv`. Nested children use
`run_role=final_candidate`, `role=selected` or `role=comparator`, and
`test_lock_status=opened`. Their metrics use
`test/{metric}/{target}/{aggregation_level}`. Each child stores refit metadata,
seeds, state bundle, reload check, evaluation metadata, predictions, metrics,
and intervals, plus its logged pyfunc model.

This guide intentionally provides no runnable final-evaluation snippet. Use
the notebook's confirmation-gated action only for the authorized canonical
run.

## 9. Load A Final Model

The `final_model_uri` tag is stored on each `final_candidate` child after its
pyfunc is logged and reloaded successfully.

```python
import mlflow

child = mlflow.get_run("<final-candidate-run-id>")
model_uri = child.data.tags["final_model_uri"]
loaded_model = mlflow.pyfunc.load_model(model_uri)
prediction_df = loaded_model.predict(target_free_modeling_df)
```

The input is a raw, target-free modeling frame containing building and
neighborhood IDs plus the features required by the fitted transformer. The
output column order is `building_id`, `total_mean`, then each cohort's
`{cohort}_mean` and `{cohort}_probability` in schema order. Serving is
deterministic point prediction; predictive draws and intervals remain offline
evaluation evidence.

For the canonical `direct-poisson` artifact, import LightGBM before loading in
a fresh process:

```python
import lightgbm  # Load its OpenMP runtime before the bundled torch imports.
import mlflow

loaded_model = mlflow.pyfunc.load_model(model_uri)
```

The package itself preserves this import order, but the already-logged
canonical artifact predates that correction.

## 10. Troubleshoot Safely

**Experiment artifact-location mismatch.** Set
`AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` to the experiment's existing location or
choose a new `MLFLOW_EXPERIMENT_NAME`. Do not move an experiment by editing the
database.

**Experiment is deleted.** Restore it through MLflow or choose a new scratch
name. The adapter refuses deleted experiments.

**An MLflow run is already active.** End it before calling
`tracked_comparison` or the final guard. Both require their own top-level run.

**Tracking refuses missing bundles.** Rerun CV with
`capture_artifacts=True`. There must be exactly one passing reload check per
candidate and fold.

**A run is `RUNNING` after a crash.** Treat it as incomplete. Do not add
evidence to it. Start a new scratch comparison; canonical final retries remain
subject to the opened-attempt fingerprint rule.

**A model URI does not load in a fresh process.** Confirm the tracking and
artifact stores are reachable, install the `tracking` group, and apply the
LightGBM-first import rule above for `direct-poisson`. A successful in-process
reload does not by itself prove fresh-process code-path or native-runtime
loading.

**Imports fail in this checkout.** Use `PYTHONPATH=src` with `uv run` if the
editable installation is not visible to the interpreter. This changes import
resolution only; it must not be used to redirect tracking or lockbox paths.