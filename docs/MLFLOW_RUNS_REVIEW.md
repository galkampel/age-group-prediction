# Review: the MLflow logging, before M6 (overcomplication and completeness)

**For:** the user, who accepts or rejects each finding; then M6's planning session, which
builds the accepted ones in. The review was done by Fable 5.2, read-only, on 2026-11-15. It ran
on HEAD `94a47c3` (M5) and on M5's Check C scratch store (24 tuning runs, 1 best runs' run).
Claude Opus 5.5 checked each finding against the code before writing it here. Line numbers are
in `notebooks/03_hyperparameter_tuning.py` unless another file is named.

**Status (2026-11-15): decided.** The user accepted every recommendation (do not re-ask):
- F1, F2 and F3;
- F4 with F6: `log_evaluation_run` and `code_tags`;
- F5 (a), (b) and (c);
- F7 (b), F8, F9, F10 (a) and (c);
- F10 (b) left.

All go into M6's change, built and checked once (§6). M7 takes F5 (c).

**Implemented in M6 (2026-11-15), all checks passed** (the plan's M6). The one correction:
M4's cell keeps `json`, which its runs tables' dict columns need. F5 (c) is in M7's steps.

## 1. The verdict

M1–M5 log every item that [MLFLOW_RUNS_PLAN.md](MLFLOW_RUNS_PLAN.md) §5 places in the tuning
runs and the best runs' run. Each was seen on the store; nothing planned is missing. The code is
mostly simple enough, but four things should be settled before M6:
- a hole in the store's guard (F1);
- a stale skip of the best runs' run (F2);
- a missing `return` in M6's plan (F3);
- M6 repeating about 70 lines of M4's logging cell, where decision 5 asks for a function (F4).

One gap bears on decision 4: the best runs' run and the final run cannot be rebuilt with
`from_params` from what they log (F5). The rest is minor.

## 2. The findings recommended to accept

### F1. The guard checks the artifact variable, but an existing experiment keeps its own location (bug)
- **The code:** lines 280–283 create the experiment only when it is missing. When it exists,
  `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` passes its two refusals and is then read by nothing; the
  artifacts go to the experiment's stored location.
- **How it goes wrong:**
  - MLflow creates a missing `MLFLOW_EXPERIMENT_NAME` with no location
    (`fluent._get_experiment_id_from_env`), and from the repo root that defaults to the
    canonical `./mlartifacts`.
  - An experiment created that way by any earlier MLflow call would take all three buttons'
    artifacts into the canonical folder, with every refusal passing.
- **Precedent:** the old stack refuses this case (`src/age_group_prediction/tracking/runs.py`,
  `_ensure_experiment`).
- **The change:** keep the experiment, and raise when its `artifact_location` differs from the
  resolved variable. One refusal test with a pre-created experiment.
- **Effort and scope:** small; touches no decision (it completes D1e).

### F2. The best runs' skip key ignores the objective and the selection metric (bug)
- **The code:** line 2769, `_run_key = f"{tuner.storage_dir.name}/best_runs"`. The folder name
  (lines 884–900) holds the data, the split, the seed and the budget only. `objective_metric`
  and `selection_metric` are settings (lines 357–358); the objective is in the study names, the
  selection metric nowhere.
- **The effect:**
  - Change either setting, tune again, click "Log the best runs", and the cell says "skipped",
    although the best runs, their metrics and their figures differ.
  - The tuning runs are safe: their key ends in the study name, which holds `objective=`.
- **The change:** the key gains `__objective=<name>__selection=<name>`. Check D's expected key
  follows.
- **Effort and scope:** small; touches decision 3 (what "a run that exists" means).

### F3. M6's plan reads `val_against_test_df`, which its cell does not return (a bug in the plan)
- **The code:** the cell at 3107–3149 ends in a bare `return`. M4 met the same thing and
  recorded it; M6's steps do not.
- **The change:**
  - M6 restores `return (val_against_test_df,)`. ruff on `03` then goes 15 + 1 (the logging
    cell's bare `return`) − 1 = 15.
  - Drop M6's hedge "a dict column, if any": none of its six tables has one.
- **Effort and scope:** small; touches only choice 16's count.

### F4. M6 as designed repeats M4's logging; one function for both (overcomplication)
- **The code:** M4's lines 2781–2805 (the metrics) and 2847–2875 (the `with` block). M6 would
  repeat them for the test part:
  - the same: the statistics, the composite rows, the CSV tables, the figure, components and
    input loops;
  - different: the `val/` or `test/` prefix, the run name, the tags, the parameters, which
    tables, figures and datasets.
- **The change:**
  - one function in its own cell, `log_evaluation_run(run_name, tags, params, dataset,
    scores_df, difference_df, bound_names, tables, figures, best_runs, datasets)`: it builds
    the composite metrics, opens the run and logs everything;
  - M4's cell keeps its button, its skip, its tables and figures, and calls it;
  - M6's cell is its guard, its tables and figures, and one call.
- **The proof:** Check D already reads the best runs' run back item by item.
- **Effort and scope:** medium; applies decision 5.

### F5. The best runs' run and the final run cannot be rebuilt with `from_params` (gap)
- **Why not:**
  - `to_dict()` has no inverse (no `from_dict`).
  - `CountComponent.from_params` (`components.py:102`) needs the flat `best_params`,
    including each component's `feature_transformer` name, looked up among the code's
    candidates.
  - `components/<model>.json` holds the transformer's declaration, not that name.
  - The two runs hold no `best_params`: the name is found only in `tables/model1_runs.csv`, or,
    for the final run, by following the `best.<model>` tag to a tuning run, which exists only if
    that button was clicked.
  - `y_min=PREDICTION_FLOOR` (line 857) is logged nowhere.
- **The change:**
  - (a) the shared function also logs `best_params/<model>.json` per best run;
  - (b) `"prediction_floor"` joins `shared_params` (22 → 23 parameters; the M3 test and Check D
    follow);
  - (c) M7's guide states the rebuild recipe: `from_params(best_params, …)` then the model
    with `y_min` and `random_state=seed`, with `components.json` the readable declaration.
    Correct §5's wording to match.
- **Effort and scope:** small; makes decision 4 true for all three run kinds.

### F6. The notebook's path is a literal repeated in each run's tags (minor)
- **The code:** lines 1621 and 2856 (M6 would add a third) each repeat
  `"notebook": "notebooks/03_hyperparameter_tuning.py", **git_tags`.
- **The change:** the MLflow cell's one dict holds it (`code_tags`: the notebook, the commit,
  the dirty flag), and each run spreads it. Its test and Check B/D follow.
- **Effort and scope:** small; touches choice 14's name only.

### F8. `search_space.json` holds JSON text inside JSON (minor)
- **The evidence:** on the store, `"total__max_depth": "{\"name\": \"IntDistribution\", …}"`.
  `distribution_to_json` returns a string, so the UI shows backslashes.
- **The change:** `json.loads(optuna.distributions.distribution_to_json(distribution))`.
  `json_to_distribution(json.dumps(value))` reads it back.
- **Effort and scope:** small.

### F10 (a) and (c). M6's guard and a file name (minor)
- **(a)** The guard is an `if` only because its message indexes `_finals[0]`. The run's name is
  always "final", so the message can say "A final run exists in the experiment: nothing logged"
  with M4's `mo.stop(bool(…), …)` form.
- **(c)** The count figures' files are named `…_composite.html`, but the figures show the total
  (their axes say "true total"): name them `…_total.html`.

## 3. The findings put to the user, without a firm recommendation

### F7. `study_seed` is recomputed in the logging cell (overcomplication, modest)
- **The code:** line 1722 rebuilds `seed + _i`, a third copy of the rule the tune cells apply
  (lines 1112, 1381). A change in a tune cell would silently log a wrong `study_seed`.
- **Options:**
  - (b) the record keeps `"study_seed"` and `log_tuning_run` reads it, one parameter fewer.
    Check C's record keys follow.
  - (a) inline `log_tuning_run` into its one caller. Fable leaves (a) to taste: the function
    documents one run's content in one place.
- **Recommended:** (b).

### F9. `feature_transformer.<component>` repeats `best.<component>__feature_transformer` (minor)
- **The evidence:** on the store, `best.total__feature_transformer = count__room_share_x_ses`
  beside `feature_transformer.total = count__room_share_x_ses`. Every run setting has
  `share_feature_transformer: False`, so the two are equal in every run: 2–3 duplicate columns
  in the UI.
- **The change:** drop `feature_transformer.*`. §5's "their names as parameters" is met by
  `best.*`, and `components.json` keeps the declaration.
- **Recommended:** drop.

## 4. The finding recommended to leave

### F10 (b). One tag for the final run
`final=true` and `stage=final` say the same thing. Keep both: `final=true` is decision 8's and
E11's, and the M6 guard is written for it.

## 5. Considered, no finding

- **The tuning runs on the store:** every §5 parameter, the 52 metrics, the six artifacts, the
  seven tags, the two inputs (36-character digests, a schema with `long`).
- **The best runs' run on the store:** 108 metrics, 7 CSV tables (the dict columns as JSON
  text), 18 figures, 2 components, 22 parameters, the `best.<model>` tags, 2 inputs.
- **The 24 run names** are unique.
- **The MLflow cell:** its six refusals, two filters and the git tags are each needed; no global
  is set.
- **The skip queries:** 24 of them, 0.1 s; FINISHED-only, as decided.
- **Reuse:** `build_dataset` is reused (M6 adds 2 calls); `build_dataset_figures`' 16
  parameters follow the functions-as-arguments pattern, and M6's call matches it.
- **M5's three functions** match M6's calls.
- **`shared_params`** carries six settings a tuning run does not use: constant columns,
  harmless; "every run" is the plan's choice.
- **`config.json`,** repeated in the 24 runs, and the scores tables repeating the tuning runs'
  metrics: both intended in §5.
- **The MLflow tests:** each asserts something the code can break.
- **M7:** nothing wrong; F1's refusal and F5's recipe go into its guide.

## 6. Where the accepted findings go

All into M6's change, so the checks run once (the user's rule):
- first the fixes and the shared function (F1, F2, F4, F5, F6, F7, F8, F9, as accepted);
- then the final run on top of it (F3, F10).

M7 takes F5 (c).
