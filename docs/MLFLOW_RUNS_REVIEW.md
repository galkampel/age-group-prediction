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

**A second review, of the whole flow after M6 (2026-11-15): §7.** Three findings, decided:
G1 and G2 go into M7's guide, G3 is two tests; no notebook change.

**Built in M7 (2026-11-15):** G3's two tests (M3's mutation now caught by the tuning test),
G1 and G2 and §7.5's list in [MLFLOW_RUNS.md](MLFLOW_RUNS.md); the plan's §5 rows corrected.

**A third review, a cold read of the guide and the two tests (2026-11-15): §8.** Five findings
on the guide (H1–H5), two on the tests (T1, T2), all accepted (§8.6); decisions 4 and 1
explained and kept. Not built yet.

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

## 7. The second review: the whole flow, before M7 (2026-11-15)

**For:** the user, who decides each finding; then the Opus 5.5 session that builds the accepted
ones and M7. Done by Fable 5.2, read-only, on HEAD `42a1d1b` (the handoff after M6, `486891a`)
and on a copy of M6's scratch store (26 runs, read with `MlflowClient`; the UI on the copy).
Line numbers are in `notebooks/03_hyperparameter_tuning.py`.

**Status (2026-11-15): decided** (§7.6; do not re-ask). No notebook change.

### 7.1 The verdict

**The flow is correct, complete and simple enough: no bug, and nothing to remove.** On the
store, each run kind holds what §5 places in it, and a model of every run kind was rebuilt from
the run alone and reproduced the run's logged scores exactly (§7.2). Three findings remain, none
in the logging's logic:
- G1: the rebuild recipe M7 plans is one step short for the two evaluation runs (the guide);
- G2: the 24 run names look alike in the UI's default runs list (the guide, or a rename);
- G3: no test in the repository calls the two logging functions (one or two tests).

### 7.2 What was checked

- **The store** (`review2/inspect_store.py` in the scratchpad): 24 + 1 + 1 FINISHED runs.
  - A tuning run: 11 tags, 32–61 parameters (23 shared, the run's settings, `study_seed`,
    `best.*`), 52 metrics, 6 artifacts, 2 inputs. The 24 names are unique.
  - The best runs' run: 23 parameters, 108 `val/` metrics, 29 artifacts.
  - The final run: 24 parameters, 108 `test/` metrics, 33 artifacts, the only `final=true`.
  - Each `best.<model>` tag, behind the run key's folder, equals exactly one tuning run's
    `run_key`.
- **The rebuild, run for real** (`review2/rebuild_from_store.py`): a tuning run of each model,
  and both models of the best runs' run and of the final run, built with `from_params` from the
  run's `best_params` file, `prediction_floor` and `seed`, and refit. For each of the six
  models: `to_dict()` equals the run's `components` file, and the six composite scores equal
  the run's metrics exactly (`val/composite/<metric>`; `<val or test>/<model>/<metric>/score`).
  The notebook copy was run unclicked, for the code's candidates and the data only.
- **The UI** (MLflow 3.16 on the copy; screenshots in the scratchpad's `review2/shots/`): the
  runs list, a run of each kind, a CSV table, a figure.

### 7.3 The findings

#### G1. An evaluation run names its models' family and exposure only inside a file (the guide's recipe)
- **Location:** `log_evaluation_run` (lines 2773–2804); the plan's M7 steps ("`components.json`
  is the readable declaration, not an input").
- **Evidence:** `from_params` needs `family` and `use_exposure`. A tuning run holds them as
  parameters. The best runs' run and the final run hold no such parameter (the store: none
  named `family` or `exposure`): they are in `components/<model>.json` (each component's
  `family` and `use_exposure`) and inside the study name of the `best.<model>` tag. The rebuild
  above read them from `components/<model>.json` and was exact.
- **The change, two options:**
  - (a) the guide only: for an evaluation run, the recipe reads `family` and `use_exposure`
    from `components/<model>.json`; M7's sentence and §5's row of the fitted model are corrected. No
    code.
  - (b) code: `log_evaluation_run` also logs them as parameters per model (about 8 lines,
    Check D, the full runner).
- **Effort:** (a) none beyond M7; (b) small, but a notebook change.
- **Decision touched:** 4 (the declaration only), and the first review's F5 (c): completed,
  not reopened.
- **Recommendation:** (a). The file is already there and already exact.

#### G2. The 24 tuning runs' names look alike in the runs list
- **Location:** `log_tuning_run`, line 1628: `f"{model} {family} exposure=…"`.
- **Evidence:** the UI's Run Name column shows about 24 characters at its default width
  (`review2/shots/1_runs.png`): the 16 runs of Model 2 all read `total_times_probability …`,
  the 8 of Model 1 `independent_cohorts n…`. What tells them apart is cut.
- **The change, two options:**
  - (a) the guide only: the columns to add in the runs list (the tags `model`, `family`; the
    parameter `use_exposure`; the metric `val/composite/<selection metric>`, sorted), which is
    how MLflow runs are compared anyway. No code.
  - (b) a rename, the family first: `f"{family} exposure=… ({model})"`. One line, Check D's
    expected name, the full runner. It helps half: the longest family
    (`lgbm_gaussian+logistic`, 22 characters) fills the column, so `exposure=` is still cut.
- **Effort:** (a) none beyond M7; (b) small, but a notebook change.
- **Decision touched:** none numbered (the name is M3's as built).
- **Recommendation:** (a).

#### G3. No test in the repository calls `log_tuning_run` or `log_evaluation_run`
- **Location:** `tests/characterization/test_hyperparameter_tuning_notebook.py`, lines 338–409.
- **Evidence:** the three "logged only behind their button" tests read the cells' source text
  and the unclicked defs. What a run holds, and the skip, are asserted only by Check D, a
  script in `/private/tmp` (the handoff: "the scratchpad can vanish"). The plan's own
  mutations show it: M3's (the skip's `return None` made `pass`) and M6's (the guard disabled)
  were caught by Check D only; neither changes a line the pytest tests read.
- **The change:** both functions read no global and are in `notebook_defs`, so a test can call
  them on a `tmp_path` store (the variables set with `monkeypatch`, as `notebook_defs` does):
  - `log_tuning_run` twice with a 2-trial in-memory study and a small record: the second
    call returns `None` and the store holds one run; its key, name, tags, parameter and
    metric names, its six artifacts;
  - `log_evaluation_run` once with two small frames: the metric names
    (`<dataset>/<model or difference>/<metric>/<statistic>`), the `best.<model>` tags, the
    table, `best_params` and `components` files.
  The best runs' skip and the final run's guard stay Check D's (they are in the cells, behind
  the clicks).
- **Effort:** small to medium; tests only, no notebook edit, so no runner: the touched test
  file and the non-slow suite once.
- **Decision touched:** the first review's §5 ("the MLflow tests: each asserts something the
  code can break": still true; this is what they do not reach). Marked as such, with the
  evidence above.
- **Recommendation:** accept. If the in-process call leaves any MLflow global set (M1's
  concern), the session stops and asks.

### 7.4 Considered, no finding

- **Simplicity.** Nothing to remove.
  - The filter string appears three times (two skips, one guard), each with its own key: a
    function for it would be longer than the three.
  - The components' comprehension appears twice, four lines each.
  - `log_evaluation_run`'s returned count feeds its cell's message only: harmless.
  - The final run's `run_key` is read by no guard, but it names the tuning setup, and a reader
    needs it to turn a `best.<model>` tag into a tuning run's exact key.
  - `family` is a parameter in Model 1's 8 runs and a tag in all 24: the tag is the column
    that spans both models.
- **MLflow's practices.** Settings are parameters, outcomes metrics, identity and code version
  tags, tables and figures artifacts, the parts dataset inputs with a context: as the docs
  advise. `seconds`, `n_complete`, `n_failed` as metrics are numbers a reader sorts by. A
  tuning run's `val/<cohort>/<metric>` and an evaluation run's `val/<model>/<metric>/<statistic>`
  differ in their second part, but the two kinds are never charted together.
- **No run description** ("No description" in the UI): the `stage` tag and the run's name say
  what a run is; a description per kind would repeat the guide.
- **Completeness against §5:** every row found on the store. One wording to correct in M7:
  "the model's label, the family, the stage … every run, tags": `model` and `family` are the
  tuning runs' tags; an evaluation run has `best.<model>` instead.
- **Closed decisions** (1–17, F1–F10): nothing on the store argues against any of them.

### 7.5 What M7's guide must say (beyond its planned steps)

- **The rebuild recipe per run kind** (G1), with where each argument comes from; that it was
  run and was exact; that the candidates (`COUNT_FEATURE_TRANSFORMERS`,
  `PROBABILITY_FEATURE_TRANSFORMERS`) are the notebook's at the run's `git_commit`.
- **Comparing the 24 runs in the UI** (G2): the columns to add and the sort.
- **From an evaluation run to its tuning runs:** the filter `tags.run_key = '<the run key's
  folder>/<the best.<model> tag>'`. The tuning run exists only if its button was clicked; the
  evaluation run does not need it (it holds its own `best_params`).
- **The metric names of the two kinds,** and that `test/` metrics are in the final run only.
- **`git_dirty`:** commit before logging, so the run's `git_commit` is the code that ran (the
  scratch store's runs say `true`).
- **A deleted run is not seen** by the skips or the guard (`search_runs` reads active runs
  only): deleting a run in the UI is how one is logged again, the final run included, so
  deleting the final run is a decision (E11).
- **The store's `Default` experiment** is MLflow's own, created with the store; its artifact
  location is `<working directory>/mlruns/0`. Nothing logs to it and no folder is created.
- **`mlflow.source.name`** is the running script, not the notebook's path: the `notebook` tag
  is the one to read.

### 7.6 The user's decisions (2026-11-15; do not re-ask)

- **G1: the guide only.** For an evaluation run the recipe reads `family` and `use_exposure`
  from `components/<model>.json`. No parameter is added.
- **G2: the guide only.** The guide names the columns to add in the runs list. The run names
  stay.
- **G3: both tests.** `log_tuning_run` called twice and `log_evaluation_run` once, on a
  `tmp_path` store.

**Where they go:** all into M7's session, before the guide. The notebook is not edited:
- first G3's two tests, with a mutation (M3's: the skip's `return None` made `pass`, on a
  copy) that the new tuning test must now catch;
- then the guide with G1, G2 and §7.5's list, and §5's two corrected rows.

## 8. The third review: a cold read of the guide and the two tests, before the PR (2026-11-15)

**For:** the user, who decides each finding; then the session that builds the accepted ones (the
guide and the test file only). Done by Fable 5.2, read-only, on HEAD `3ac23b1` (on `d59fa47`,
M7) and on a copy of M6's store (`mlflow_M7_guide.db`, 26 runs; the UI on the copy). Line
numbers are in [MLFLOW_RUNS.md](MLFLOW_RUNS.md) unless a file is named. The checks ran once
(`review3/run_review3.sh` and its log in the scratchpad; the screenshots in `review3/shots/`).

**Status (2026-11-15): decided** (§8.6; do not re-ask). Not built yet: the guide and the test
file change in the next session.

### 8.1 The verdict

**The guide is correct against the code and the store; a reader new to MLflow gets lost in two
places.** What each button logs (§1, §4), the skip and the guard (§6), the comparison (§7) and
the rebuild (§9) are clear enough and true: every count and name was read back from the store,
and the guide's commands ran as written (§8.2). The two places:
- §3 does not get the reader from the UI's first page to a run (H1);
- §8's and §9's code stops with an unreadable error on a store without a best runs' run or a
  final run, which is the user's own store until E11 (H2).

The two tests are readable and assert what matters, with two holes shown by mutations (T1,
T2). Decisions 4 and 1 are both right for this project now; nothing found argues against
either (§8.5).

### 8.2 What was checked

- **The guide's commands as written,** with `guide.env`: the script run (exit 0, the
  `experiment=` line, 26 runs before and after); `mlflow ui` (§3's command, from the
  scratchpad); §8's block (one tuning run per `best.<model>`); §9's recipe (the best tuning
  run, both models of the best runs' run and of the final run, each exact).
- **Every count and name of §1 and §4–§7 on the store** (`review3/check_guide.py`): 24 + 1 + 1
  runs; the 23 shared parameters and the 5 tags in all 26; a tuning run's 7 tags, 52 metrics,
  6 artifacts, `trials.csv`'s columns, 2 inputs; the best runs' run's 108 `val/` metrics,
  7 tables, 18 figures; the final run's 108 `test/` metrics, 6 tables, 23 figures,
  `n_train_val`, its inputs `train_val` and `test`; the three run keys' forms; §7's claim (each
  model's best tuning run by `val/composite/rmse` is the run its `best.<model>` tag names). All
  as the guide says.
- **The UI, followed as a new reader** (MLflow 3.16, headless Chrome): the first page, the
  experiment, the runs list, §7's filter, sort and columns menu, a run of each kind, a table,
  a figure. §7's three steps work as written.
- **The repository unchanged** by the commands: its files, its `mlflow.db` and `mlartifacts/`
  compared before and after; no `.mlflow-local/`; the notebooks' md5 as before.

### 8.3 The guide's findings

#### H1. §3 stops one click before the runs
- **Location:** lines 88–94 ("open … and select the experiment"; "the runs are under the
  experiment's runs list").
- **Evidence:** in the UI's default "GenAI" mode, selecting the experiment opens an "Overview"
  page of traces ("Traces 0", "No data available"), and the sidebar has no "Runs" entry
  (`shots/2_experiment.png`). A run's page there is filed under "Evaluation runs"
  (`shots/4_tuning_run.png`), the guide's own name for two of its run kinds. One click on
  **Model training** (the toggle at the top left) gives the sidebar "Runs, Models, Traces" and
  opens the runs list (`shots/9_model_training_mode.png`); Chrome kept the choice on the next
  page load. The guide also never says what a run's page holds: its tabs are Overview (the
  tags, the parameters, the metrics as a table), Model metrics (the charts) and Artifacts (the
  files).
- **The change:** §3's list becomes the path: click "Model training", open the experiment
  (its runs list), click a run; one line per tab; per run kind, the file to open first
  (`trials.csv`; `tables/best_runs_val_scores.csv` and `figures/val_forest_runs_<metric>.html`;
  `tables/test_scores.csv`). About 8 lines replace 2.
- **Effort:** small; the guide only.
- **Recommendation:** accept. It is the one place the cold read could not go on without
  guessing.

#### H2. §8's and §9's code needs all three run kinds, and fails without saying so
- **Location:** lines 211–216 (§8) and 276–281, 371–373 (§9): `(run,) = client.search_runs(…)`.
- **Evidence:** on a copy of the store with the final run deleted, §9's recipe stops at
  `first_run` with `ValueError: not enough values to unpack (expected 1, got 0)`; with the
  best runs' run deleted too, §8's block stops the same way (`review3/run_review3.log`, step
  3). After the PR the user's store holds tuning runs and a best runs' run and no final run
  until E11: the recipe as written cannot run on it.
- **The change, two options:**
  - (a) `first_run` returns `None` when no run matches, and §9's script rebuilds the run
    kinds it finds and prints which it skipped (about 8 lines changed); §8's block gets one
    sentence ("needs a logged best runs' run"). The recipe is run once more on both stores.
  - (b) a sentence before each block naming the runs it needs. No code.
- **Effort:** (a) small, with one run of the two blocks; (b) none.
- **Recommendation:** (a). The recipe is the proof of decision 4; it should run on the store
  the user will have.

#### H3. MLflow's words are used before they are explained
- **Location:** §1 and §2 (lines 14–59): experiment, run, tag, parameter, metric, artifact,
  dataset input, tracking URI, "the store".
- **Evidence:** the cold read. §1's table has a "`stage` tag" column, §2 a "tracking URI" and
  an "artifact location"; none is defined, and §4 depends on telling a parameter from a tag
  from a metric.
- **The change:** a short list at the top of §1, one line each: the store (a sqlite file for
  the runs' values, a folder for their files), an experiment (a named group of runs), a run
  (one logged result), a parameter (a setting, text), a metric (a number, sortable and
  charted), a tag (a label to filter by), an artifact (a file), a dataset input (the data's
  name and hash). About 9 lines.
- **Effort:** small; the guide only.
- **Recommendation:** accept.

#### H4. Three small corrections
- **Line 36:** "the notebook reads the store from four variables": its cell reads three
  (lines 237–239 of the notebook); `MLFLOW_DISABLE_AGENT_HINT` is read by MLflow itself.
- **Line 18:** the column heading "Button (click after)" is hard to parse: two columns, "Button"
  and "Needs first".
- **Lines 57–65:** "the cell sets no MLflow global" and the two warning filters are notes for
  whoever edits the cell, not for the notebook's reader; the plan's §1 holds both. One sentence
  ("the cell also silences two pandas 3 warnings inside MLflow; any other warning still
  shows") replaces 9 lines.
- **Effort:** small.
- **Recommendation:** accept all three. Nothing else is longer than it needs to be: §9's
  script is long, but it is the recipe, and it runs.

#### H5. The journal is logged, and the guide does not say how to read it
- **Location:** line 131 ("`study/<study name>.log` is the study's Optuna journal").
- **Evidence:** decision 1 rests on it (§8.5): the journal is what gives a study's charts. Run
  for real (`review3/journal.py`): the file, loaded with `optuna.load_study`, gave the 3
  trials, `best_value` equal to the run's `cv/rmse`, `best_params` equal to
  `best_params.json`, each trial's start time, and four of Optuna's own figures (the
  optimization history, the parallel coordinates, the parameter importances, the slices). A
  new reader would not know the four lines.
- **The change:** a short block under §4 or §7: download the file (or copy it from the
  artifact folder: Optuna's file backend takes a lock beside the file it opens), then
  `JournalStorage(JournalFileBackend(path))`, `load_study(study_name=<the file's stem>)`,
  `optuna.visualization.plot_optimization_history(study)`. About 10 lines; run once.
- **Effort:** small.
- **Recommendation:** accept. It is the answer to "where are the trial charts".

### 8.4 The two tests

**Readable, and each assertion is one the code can break.** The setup is written twice (the
experiment, the study, the stand-in component), about 15 lines each: left as is, each test
reads top to bottom without a helper. Two holes:

#### T1. The tuning test asserts the metrics' names, not their values
- **Location:** `tests/characterization/test_hyperparameter_tuning_notebook.py`, lines 497–504
  (`assert set(run.data.metrics) == {…}`); the evaluation test asserts the values (609–617).
- **Evidence:** a mutation on a copy (`review3/mut_values`): `cv/<objective>` logged from
  `record["seconds"]`, and the train and validation scores swapped. Both new tests pass.
- **The change:** the same assertion with the values: `run.data.metrics == {"cv/rmse":
  study.best_value, "train/composite/rmse": 1.5, "val/composite/rmse": 2.5, "seconds": 1.0,
  "n_complete": 2.0, "n_failed": 0.0}`. No line added.
- **Effort:** small; the test file and the mutation once.
- **Recommendation:** accept.

#### T2. No test logs a dataset input or a figure
- **Location:** the same file, lines 465, 601 and 603: both tests pass `{}` for the dataset
  inputs, and the evaluation test `{}` for the figures.
- **Evidence:** a mutation (`review3/mut_inputs`): both `mlflow.log_input` calls and
  `mlflow.log_figure` made `pass`. Both new tests pass. Only Check D, a script in
  `/private/tmp`, reads them back.
- **The change:** one small dataset (`mlflow.data.from_pandas` of a two-row float frame, a
  given 36-character digest) passed to both calls, and one empty plotly figure to the
  evaluation call; assert the run's inputs (name and context) and the figure's file. About 12
  lines. A float frame avoids the integer-schema warning the MLflow cell filters (pytest
  resets the filters per test).
- **Effort:** small.
- **Recommendation:** accept; smaller than T1 (a deleted line, not a wrong value).

### 8.5 The two decisions, against common practice

Neither is reopened: no check gave evidence against either.

#### Decision 4: no logged model. Right for now.
- **What logging a model is.** `log_model` writes a folder into the run: the fitted model's
  file, an `MLmodel` file naming how to load it, the input and output columns (the
  signature), and the Python packages it needs. With it:
  - any code loads the model by an address (`mlflow.pyfunc.load_model("models:/…")`) and
    calls `predict`, without the notebook;
  - the **Model Registry** gives the model a name, numbered versions and aliases
    ("champion"), so a consumer asks for "the current one";
  - `mlflow models serve` puts it behind an HTTP endpoint.
- **What the rebuild gives instead.** The run holds `best_params`, the components'
  declaration, `prediction_floor`, `seed` and `git_commit`; `from_params` and a refit give the
  same model. §9's recipe reproduced the logged scores exactly, for all three run kinds, in
  three sessions (6 s for five models here).
- **For the 24 tuning runs, this is the common practice too.** MLflow's own Optuna tutorial
  logs no model per trial or per candidate; it refits the best parameters once and logs that
  one model. Twenty-three of the 24 are never used again.
- **For a final model, common practice would log it,** and this project does not. What is
  missing: an address to load it from, a registry version, serving. Who needs them: code
  outside the notebook that predicts. There is none today (the data is simulated; the final
  run's test evaluation is the end of the flow).
- **Why not log it anyway:**
  - the fitted models do not pickle (a feature transformer holds a lambda; only
    `cloudpickle` works, the handoff's trap), and a cloudpickle file loads only with the same
    code and library versions, so it would not remove the need for the code at `git_commit`.
    MLflow 3.16 itself no longer defaults to pickle for scikit-learn models
    (`mlflow/sklearn/__init__.py`: `serialization_format=SERIALIZATION_FORMAT_SKOPS`);
  - the old stack did it without pickle
    (`src/age_group_prediction/tracking/pyfunc_model.py`): a "model from code" whose
    `load_context` reads a JSON state bundle and calls `from_state_bundle`. That needed a
    state bundle on every model class; the new `modeling/` package has none (no `state_bundle`,
    `pickle` or `save` in it). A logged final model here is that feature, not a logging line.
- **When to revisit:** when something outside the notebook must predict (a service, a batch
  job, another repository). Then: a pyfunc model from code for the final model only, plus the
  registry. The UI shows the empty places it would fill ("Logged models —", "Registered models
  None" on a run's page).

#### Decision 1: no nested runs. Right for now.
- **What nested runs are.** MLflow's Optuna tutorial opens one parent run per study and, inside
  the objective, one child run per trial (`mlflow.start_run(nested=True)`) holding the trial's
  parameters and its score. The UI folds the children under the parent, and its chart view
  compares them without code (a parallel-coordinates chart of parameters against the score, a
  scatter, sorting by any parameter).
- **What this project has instead.** One run per study, with:
  - `trials.csv`, which the UI shows as a sortable table (`shots/5_tuning_artifacts.png`);
  - the journal, which gives the same charts from Optuna itself (H5: run for real).
- **The cost of child runs here:**
  - **the count:** 24 studies × 100 trials (`n_trials = 100`, the notebook's line 339) = 2,400
    more runs beside 26. The tutorial's case is one study;
  - **the skip rule:** "a FINISHED run with this key exists" becomes "the parent and all its
    children exist and are complete"; a failure halfway leaves a parent with some children;
  - **§7's comparison** (filter `stage = 'tuning'`, sort by the validation metric) would need
    the children filtered out or given no such metric;
  - **no live view to gain:** the trials run inside `src`'s tuner and are logged afterwards,
    behind a button, so child runs would be written from the finished study, like the table.
  - the time to write 2,400 runs was not measured (one run with its files took 0.1–0.15 s at
    M0).
- **What is lost:** the trial charts inside the MLflow UI. They are one `load_study` away.
- **When to revisit:** if the trials should be watched in the MLflow UI while a study runs
  (logging moves into the objective), or if the experiment shrinks to a few studies.

### 8.6 The user's decisions (2026-11-15; do not re-ask)

Every recommendation accepted:
- **H1:** §3 gets the path (the "Model training" click, the runs list, a run's tabs, the file
  to open first per run kind).
- **H2 (a):** `first_run` returns `None` when no run matches; §9's script rebuilds the run
  kinds it finds and prints which it skipped; §8's block gets its sentence.
- **H3:** the list of MLflow's terms at the top of §1.
- **H4:** all three (the three variables, the two column headings, §2's notes cut to one
  sentence).
- **H5:** the block that reads a study's journal with Optuna.
- **T1:** the tuning test asserts the metrics' values.
- **T2:** one dataset input in each test and one figure in the evaluation test, asserted.
- **Decisions 4 and 1:** kept, as explained in §8.5.

**Where they go:** one session, one change, checked once. No notebook, no `src` file:
- first T1 and T2 in the test file, with the two mutations of §8.4 (`review3/run_review3.sh`,
  step 6, is the model): each must now fail a new test;
- then the guide (H1–H5);
- **the checks, once:** ruff and the format on the test file, the test file, the non-slow
  suite, mypy; the guide's commands and python blocks as written on the scratch store (§9's
  recipe also on a copy without its final run: it must finish and say what it skipped; H5's
  block); the two mutations. No Check B, C or D (no notebook changed).
