# Final Evaluation

This document describes the one-time step after the pretest freeze: refitting
each approach's frozen winner on the full training partition, fingerprinting
that exact attempt, opening the lockbox once, scoring every refit on the same
holdout, recording the result under guarded MLflow runs, and serving each refit
as a pickle-free pyfunc model. Everything here is derived from
`src/age_group_prediction/experiment/final_evaluation.py`,
`experiment/evidence.py`, `tracking/final.py`, `tracking/runs.py`,
`tracking/pyfunc_model.py`, and the final controls in
`notebooks/02_model_fitting.py`.

**There is intentionally no runnable example.** Opening the canonical lockbox
is an irreversible project action.

## 1. Purpose And Project Status

The holdout exists to give one unbiased estimate after every choice has been
frozen on training data. Evaluating it more than once, or letting it influence
any choice, would turn it into a selection set. The code therefore makes
evaluation one-way: the freeze can accept test metrics once, and the tracking
guards refuse duplicate or divergent attempts.

**Status.** Project records ([README](../README.md#run-the-modeling-notebook),
[MLflow guide](MLFLOW_EXPERIMENTS_GUIDE.md#8-understand-final-run-semantics),
and [docs/README.md](README.md#current-status)) state that the canonical final
evaluation has already run and selected the Bayesian conditional model, with
the other two approaches as comparators. The code does not record this, and
it was not re-verified for this document because the canonical store is not
opened. **Never rerun it.**

## 2. Step Order

Operational callers use one entry point:

```text
tracking.run_final_evaluation(
    outer_train_df, modeling_table, *, split_manifest, cv_result,
    pretest_freeze, candidates, final_refit_factories, context,
    source_cv_run_id, evaluation_config, schema=DEFAULT_MODELING_SCHEMA,
    settings=None,
) -> FinalEvaluationResult
```

1. `verify_pretest_freeze(pretest_freeze, cv_result)`.
2. `refit_frozen_approach_winners(...)` — before any MLflow run exists.
3. `final_attempt_fingerprint(...)`.
4. Inside `tracked_final_evaluation(...)`:
   `evaluate_frozen_models_on_lockbox(...)`, then
   `log_final_evaluation_result(...)`.

`verify_pretest_freeze` and `final_attempt_fingerprint` are imported from
`age_group_prediction.experiment.final_evaluation`; they are not re-exported
from the package root. The lower-level functions exist for tested composition;
calling them directly bypasses the tracking guards.

## 3. Verifying The Pretest Freeze

`verify_pretest_freeze` raises `ValueError` when:

- the freeze has no cross-family decision;
- the freeze already has test metrics;
- the freeze, with its decision removed, differs from `cv_result.freeze`
  (selections, descriptors, fold identities, seed, manifest);
- the decision differs from a fresh `select_cross_family_winner(cv_result)`.

Only `run_final_evaluation` calls it. `evaluate_frozen_models_on_lockbox`
has its own refusals (Section 6) but never compares the decision with a
recomputed one.

## 4. Refitting The Frozen Winners

```text
refit_frozen_approach_winners(
    outer_train_df, *, split_manifest, cv_result, candidates,
    final_refit_factories, schema=DEFAULT_MODELING_SCHEMA,
) -> tuple[FinalModelArtifact, ...]
```

**Preconditions** (each raises `ValueError`): the table passes schema
validation; the manifest fingerprint equals the freeze's and the provenance
record's; training IDs equal the freeze's; `table_hash` and
`column_schema_hash` equal the recorded training hashes; exactly the three
approaches are frozen; each winner exists in `candidates` and
`final_refit_factories` and its runtime descriptor equals the frozen
descriptor; and the master seed is nonnegative with a source equal to the
provenance record's `(master_seed, master_seed_source)`.

**Refit.** Each winner's final factory builds a fresh model, which is fitted on
the **entire** outer training partition with
`rng = default_rng(stable_seed(master, "final/{id}/fit"))`. It never sees a
holdout row. Model A and Model B re-run their nested tuning there with seeds
derived from the frozen master seed; the Bayesian model has none.

**Bayesian guard.** For the `total` and `composition` stages, the refit's
recorded diagnostics must show `active_profile == "full"` and
`action == "error"`; `chains`, `warmup_steps`, and `posterior_samples` must be
at least the **package default** full profile
(`BayesianConditionalConfig().full_profile`), and every diagnostic threshold at
least as strict as the package default full diagnostics. Stricter settings
pass. The TOML full profile currently equals the package defaults.

**Reload check.** As for fold artifacts
([details](CROSS_VALIDATION_AND_SELECTION.md#4-fold-artifacts-and-reload-checks)),
on a smoke frame of the first 20 training rows sorted by building ID, with
seed purpose `final/{id}/reload_check` and tolerance 0.

**Failures** in `fit` and the reload check are re-raised as
`RuntimeError("Candidate '{id}' failed during final {operation}: ...")`,
copying the original's `stage`, `diagnostics`, and `failures` attributes when
present. The Bayesian guard runs outside that wrapper, so its errors surface
unwrapped.

Each `FinalModelArtifact(candidate_id, approach, model, evidence)` carries
`FinalRefitEvidence`: `candidate_id`, `approach`, `manifest_fingerprint`,
`training_data_hash`, `training_schema_hash`, `seeds`, `model_metadata`,
`state_bundle`, `reload_check`, and `smoke_frame_building_ids`.

## 5. The Final Attempt Fingerprint

```text
final_attempt_fingerprint(pretest_freeze, refit_result, *, candidates,
                          evaluation_config, schema=DEFAULT_MODELING_SCHEMA) -> str
```

SHA-256 of canonical JSON containing:

- `pretest_freeze.to_dict()`;
- for each refit, sorted by ID: `candidate_id`, `approach`,
  `manifest_fingerprint`, `training_data_hash`, `training_schema_hash`,
  `seeds`, the bundle's `dependency_versions`, and
  `model.configuration_record()`;
- each selected candidate's descriptor and prediction config;
- the evaluation config (or `null`) and the schema.

It identifies *what would be evaluated*, so a retry that would tune, seed,
configure, or score differently gets a different fingerprint and is refused
(Section 7). Model source code is not hashed.

## 6. Evaluating Once On The Lockbox

```text
evaluate_frozen_models_on_lockbox(
    modeling_table, *, split_manifest, pretest_freeze, refit_result,
    candidates, evaluation_config, schema=DEFAULT_MODELING_SCHEMA,
) -> FinalEvaluationResult
```

**Refusals before replay:** no decision or existing test metrics; schema
validation failure; manifest fingerprint differing from the freeze's or the
decision's; not exactly three frozen approaches; refits that are not exactly
the frozen winners; missing candidates.

**Replay checks.** `replay_split_manifest` rebuilds the split; replayed
training IDs must equal the freeze's; every refit's manifest fingerprint and
training data and schema hashes must match the replayed training partition;
replayed holdout IDs must equal the manifest's.

**Scoring.** Every refit predicts the **same** replayed holdout frame
(`split.test_df`, targets included so metrics can be computed), and its
prediction IDs must equal the first model's. Seeds use
purposes `final/{id}/predict`, `evaluate`, and `bootstrap`; the bootstrap runs
only when `evaluation_config` is given.

**No cross-family ranking on test data.** Each evaluation's `role` is
`selected` when it is the frozen `selected_candidate_id` and `comparator`
otherwise, and its `metric_comparability` is that approach's policy text.
Test metrics are reported, never used to choose.

**Finalization.** The freeze becomes `pretest_freeze.with_test_metrics(summary)`,
with values keyed `metric:target:aggregation_level` per candidate.

`FinalEvaluationResult` fields: `manifest_fingerprint`,
`holdout_building_ids` (sorted), `evaluations` (`FinalCandidateEvaluation`:
`candidate_id`, `approach`, `role`, `seeds`, `metrics_df`, `intervals_df`,
`predictions_df`, `metric_comparability`, `model_metadata`,
`evaluation_metadata`), `finalized_freeze`, `holdout_input_df` (IDs and
features only, built after scoring for pyfunc logging; target columns
refused), and `schema`. `to_dict()` omits the data frames.

## 7. Tracked Lifecycle And Guards

`tracked_final_evaluation(context, *, source_cv_run_id, manifest_fingerprint,
cross_family_rule_version, selected_candidate_id, selected_approach,
pretest_freeze, final_attempt_fingerprint, settings=None)` is a context
manager yielding `FinalRunHandle`.

**Before opening** it refuses: a context whose `test_lock_status` is not
`locked` (the default); a freeze without a decision; selected ID or approach
not matching the decision; an empty fingerprint; an already active run.

**Source run.** The CV run must be in the same experiment, `FINISHED`, with
`run_role=comparison` and `evidence_complete=true`.

**Duplicate refusal.** A final run with the same `source_cv_run_id`,
`manifest_fingerprint`, and `evidence_complete=true` is refused, searched with
deleted runs included.

**Identical-retry rule.** Every final run on the manifest whose
`test_lock_status` is `opened` (deleted runs included) must match the new
attempt's `source_cv_run_id`, `cross_family_decision_hash`, and
`final_attempt_fingerprint`; otherwise `RuntimeError`. A permitted retry
lists them in `retry_of_run_ids`. A run that failed before the lock flipped
stays `locked` and does not constrain later attempts.

**Opening.** The run starts with tags `run_role=final_evaluation`,
`source_cv_run_id`, `manifest_fingerprint`, `cross_family_rule_version`,
`selected_candidate_id`, `selected_approach`, `cross_family_decision_hash`,
`final_attempt_fingerprint`, and `test_lock_status=locked`; logs
`pretest_freeze.json` **while still locked**; then sets
`test_lock_status=opened` and yields. Failures end the run `FAILED` (or
`KILLED` on interrupt) with `evidence_complete=false`.

**Scope limitation.** All searches are limited to the current tracking URI
and experiment. A different store or experiment name cannot see earlier final
runs, so none of these guards protect the canonical lockbox from a scratch
store.

## 8. Final Evidence

`log_final_evaluation_result(refit_result, evaluation_result, *, handle,
split_manifest)` requires the active run to be the handle's run with no
children and no completeness tag, and a freeze with a cross-family decision
whose manifest fingerprint matches the handle's.

- **Parent:** parameters (`candidate_count`, `holdout_building_count`,
  `selected_candidate_id`, `selected_approach`, rule name and version,
  `mlflow_version`); artifacts `split_manifest.json`, `finalized_freeze.json`,
  `cross_family_rule.json`, `comparability.json`, `provenance.json`, and
  `test_metrics.csv`; `evidence_complete=true` last.
- **One child per candidate** (`run_role=final_candidate`, `candidate_id`,
  `approach`, `role`): metrics `test/{metric_name}/{target}/{aggregation_level}`,
  seed parameters, refit metadata, seeds, compressed state bundle, reload
  check, evaluation metadata, predictions, metrics, intervals, the pyfunc
  model, tag `final_model_uri`, and `evidence_complete=true`.

Search and loading examples are in
[MLFLOW_EXPERIMENTS_GUIDE.md](MLFLOW_EXPERIMENTS_GUIDE.md#8-understand-final-run-semantics).

## 9. Pyfunc Serving Contract

Each refit is logged with `mlflow.pyfunc.log_model` as models-from-code:
`python_model` is the path of `tracking/pyfunc_model.py`, the state bundle is
the only artifact, the package source is included via `code_paths`, and the
signature and input example come from the first five holdout input rows.
Nothing is pickled.

- **Loading.** `AgeGroupPyfuncModel.load_context` reads the bundle JSON,
  dispatches on `model_class` to one of the three models (others refused),
  checks the bundle header, and calls `from_state_bundle` without `train_df`.
- **Input.** A raw modeling frame with IDs and features. Targets are expected
  to be absent, but the pyfunc does not itself refuse them.
- **Output.** `predict` uses `PredictionConfig()` without a generator and
  returns `prediction_to_frame(prediction)`: `building_id`, `total_mean`, then
  for each cohort in order `{cohort}_mean` and `{cohort}_probability`.
  Draws and intervals are not served.
- **In-process check.** Before logging completes, the loaded pyfunc's output
  must equal a fresh `model.predict(holdout_input_df, PredictionConfig())`
  frame exactly (`check_exact=True`, dtype-insensitive).
- **Fresh-process check.** A test loads the model in a subprocess without
  `PYTHONPATH` and verifies the package came from the artifact's code path.
- **Import order.** `pyfunc_model.py` and `models/__init__.py` import the
  LightGBM model before the torch model; reloading a booster after torch can
  crash the process. See the MLflow guide's
  [loading section](MLFLOW_EXPERIMENTS_GUIDE.md#9-load-a-final-model) for the
  canonical artifact, which predates this ordering.

## 10. Configuration

| TOML section / field | Effect here |
|---|---|
| `[evaluation]` | Holdout bootstrap, passed as `evaluation_config` (included in the fingerprint) |
| `[bayesian_full_profile]`, `[bayesian_full_diagnostics]` | Used by the Bayesian final factory; must be at least the package defaults |
| `[randomness] default_seed` | Reaches the refit only through the frozen master seed |
| `[outer_split]` | Not read here: replay uses the package-default `strategy_version` with the schema's ID columns; the TOML value is used by the notebook's own replay |

Tracking location settings are described in
[MLFLOW_EXPERIMENTS_GUIDE.md](MLFLOW_EXPERIMENTS_GUIDE.md#1-install-and-configure-tracking).

## 11. Pitfalls

- **Never use the notebook's final controls against a scratch store.** The
  notebook always uses the canonical
  `artifacts/lockbox/split_manifest.json` (loading it, or creating it if
  missing) and replays it, whatever the MLflow settings, and
  the guards cannot see runs in other stores or experiments.
- **Do not rerun cross-validation to "retry".** After an opened attempt, a new
  `source_cv_run_id` in the same experiment is refused.
- **Deleting a run does not reset the lock.** Deleted runs are searched.
- **Test metrics are not a selection criterion.** Comparators are reported,
  not ranked.
- **Do not call the lower-level functions operationally.** They skip the
  pretest-freeze verification and every tracking guard.

## 12. Tests And Related Documents

- Tests: `tests/unit/test_final_evaluation.py` (preconditions, Bayesian guard,
  fingerprint, replay, roles), `tests/unit/test_gate8_tracking.py` (lifecycle,
  duplicate and retry guards, evidence, pyfunc in-process and fresh-process
  reload), `tests/unit/test_final_selection.py` (freeze transitions), and
  `tests/validation/test_tracking_real_models.py`. All use temporary stores.
- [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md),
  [DATA_AND_SPLITTING.md](DATA_AND_SPLITTING.md),
  [EVALUATION_AND_METRICS.md](EVALUATION_AND_METRICS.md),
  [MODELING_GUIDE.md](MODELING_GUIDE.md) Sections 12–13,
  [MLFLOW_EXPERIMENTS_GUIDE.md](MLFLOW_EXPERIMENTS_GUIDE.md).
