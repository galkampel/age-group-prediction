# Plan: the MLflow runs of the tuning notebook

**For:** the user, and the session that implements a sub-task (written so that Claude Opus 5.5
can implement one from this file and the handoff's §4). One sub-task per approval. Tick a
sub-task when it is done and update the status line.

**Status (2026-11-14):** written and decided; nothing implemented. Next: M0.

## 1. The question and the verdict

The question (the handoff's §7): how do the tuning notebook's results become MLflow runs,
with every item of §7.4's inventory placed?

**The verdict: 26 runs in one experiment, written by three buttons in `03`, from notebook
cells only.**

| Run | How many | Button | Holds |
|---|---|---|---|
| a tuning run (`stage=tuning`) | 24, one per study | "Log the tuning runs" | the settings, the best parameters, the CV and point scores, the trials table, the declaration, the journal file |
| the best runs' run (`stage=best_runs`) | 1 | "Log the best runs" | the two best runs' bootstrap, the paired difference, the inspection, their tables and figures, the 24 runs' tables |
| the final run (`stage=final`, `final=true`) | 1 | "Log the final run" | the test scores, tables and figures; the only run with test metrics |

### The user's decisions (2026-11-14; do not re-ask)

1. **Run structure:** 24 + 1 + 1 runs; a study's trials are a logged table, not child runs.
2. **Trigger:** one button per run kind, all in the one experiment.
3. **A run that exists is skipped;** the cell reports logged and skipped counts.
4. **The model:** the declaration only (the components' `to_dict()` and the best parameters as
   JSON). No pickle, no logged model. A model is rebuilt with `from_params` and refit.
5. **The code lives in notebook cells.** What two cells repeat is a function in a cell that
   reads no global. No `src/mlflow_runs.py`.
6. **No read-back cell** (D4's table): the comparison is the MLflow UI. The user was unsure;
   this is the common practice and a one-cell addition later if wanted.
7. **The EDA notebook logs nothing.**
8. **The final run:** its own button, refused when a `final=true` run exists in the experiment.
9. **The old `tracking/` package and its guide are left untouched.**
10. **Figures:** a fixed set built by the logging cell, independent of the menus. No waterfalls.

### What was checked, and what was not

Read from the MLflow 3.16 docs and the installed 3.16.0 source:

- `log_table(data, artifact_file, run_id=None)` writes JSON and **appends when the file
  exists** in the run; `load_table(artifact_file, run_ids=…)` reads it back.
- `log_figure(figure, artifact_file)` takes a plotly figure and writes `.html` with plotly.js
  from the CDN. It has **no `run_id` argument**: it needs the active run.
- `log_dict(dictionary, artifact_file, run_id=None)`.
- `mlflow.data.from_pandas(df, source, targets, name, digest)`; `log_input(dataset, context)`.
  **A digest is at most 36 characters;** `config_hash` is 64.
- `search_runs(experiment_names=…, filter_string="tags.x = '…'")` returns a frame.
- Names may hold slashes; a parameter value at most 6,000 characters, a tag 8,000; 100
  parameters per batch.
- **MLflow's default tracking URI is `sqlite:///mlflow.db`,** which from the repo root is the
  canonical store. An unset `MLFLOW_TRACKING_URI` must be refused like the canonical one.
- `kaleido` is not installed: no PNG.

**Not confirmed without a probe (M0 settles each):**
- that no logging call warns under `PYTHONWARNINGS=error`;
- how `study.trials_dataframe()` (timedelta and datetime columns) passes through `log_table`
  and how the UI shows it;
- what `from_pandas` records as the source when none is given;
- the wall time of logging 24 runs;
- whether a run record can keep the `optuna.Study` without breaking Check C's pickles.

### Facts of the notebook the steps rest on (HEAD `10570a8`)

- A run record's keys: the run settings, `feature_transformers`, `model`, `components`,
  `cv_best`, `n_complete`, `n_failed`, `seconds`, `scores`. **It keeps no study, study name
  or study seed.**
- `build_study_name(step, run_settings, objective_name, study_seed)` is a notebook function;
  the steps are `"independent_cohorts"` and `"total_times_probability_joint"`; run *i* has
  `study_seed = seed + i`.
- `tuner.tune_*` return `(optuna.Study, seconds)`. The journal file is
  `studies_dir / f"{study_name}.log"`. A study name is unique only inside its `studies_dir`.
- `flatten`, `config_params`, `EXPERIMENT_SETTINGS` exist but their cells do not return them.
- The figure functions `forest_figure`, `ridge_figure`, `permutation_figure`,
  `beeswarm_figure` take their data and the palette. The two count figures and the shares
  bars are built inline in cells that read menus.
- Neither notebook imports `mlflow`. The notebook tests already pass scratch `MLFLOW_*`
  variables to the script.

## 2. The sub-tasks at a glance

| # | Sub-task | What you see change | Size | Implement with |
|---|---|---|---|---|
| M0 | The probe: one run of each kind logged by hand on the 3-trial copy, read back | screenshots of the MLflow UI on a scratch store; nothing in the repo | medium | **Fable** |
| M1 | The MLflow cell: the store's guard and the experiment | a line naming the experiment and the store; a refusal on the canonical store | small | Opus 5.5 |
| M2 | A run record keeps its study | nothing | small | Opus 5.5 |
| M3 | "Log the tuning runs": the 24 runs | a button; 24 runs in the UI | medium | Opus 5.5 |
| M4 | "Log the best runs": the comparison's run | a button; 1 run with 18 figures | medium | Opus 5.5 |
| M5 | The count figures and the shares bars as functions | nothing | small | Opus 5.5 |
| M6 | "Log the final run", guarded | a button; 1 run tagged `final=true`; a refusal on a second | medium | Opus 5.5 |
| M7 | Docs and close | a guide, the links, the old plan's notes | small | Opus 5.5 |

**Why Fable for M0:** it makes the design choices the docs leave open (the five unconfirmed
points above, the trials table's form, the metric names as the UI shows them) and rewrites
M2–M6's exact calls from what it saw. The others are steps whose check says plainly whether
they are right.

**The order:** M0, M1, M2, M3, M4, M5, M6, M7. M3 needs M1 and M2; M6 needs M5.

**Which model for which session** (one session per sub-task; switch the model before starting):

| Session | Model | Prompt |
|---|---|---|
| M0 | **Fable** | §7, the first prompt |
| M1, M2, M3, M4, M5, M6, M7 | Opus 5.5 | §7, the second prompt, with the sub-task's number |

**If an Opus session meets a choice this file does not make** (a call that warns, a step that
does not fit the code as it is, a check that cannot say whether the result is right), it stops
and tells the user the sub-task needs a Fable session; it does not choose by itself. M0 exists
so that this does not happen: after it, M1–M6 hold the exact calls as they were seen to work.

## 3. How a sub-task is done

The standing rules are the handoff's §4 and the review plan's §3 (the state check first; tell
the user before editing a notebook; never commit, `checkout`, `stash`, `reset` or `mv`; new
cells formatted alone and spliced in; no new ruff finding; a notebook function reads no global;
no private helpers; the stop's summary, md5 and commit message). Added for this plan:

1. **The store.** Never write the repo root's `mlflow.db` or `mlartifacts/`, nor
   `.mlflow-local/` (the user's scratch store). Every check uses
   `MLFLOW_TRACKING_URI=sqlite:///$S/mlflow_<sub-task>.db` and an artifact location in `$S`,
   both removed with `rm` before a run.
2. **The runner:** copy `checks/run_S5_3.sh` (its EDA figure step set back to the exact
   `compare_figures.py`), rename the outputs, keep the references `check_c_s5_3_*` (or the
   previous sub-task's).
3. **Check C stays equal** to its reference except what the sub-task's "Validate" names. The
   patched `run_button` clicks the logging buttons too, so from M3 on Check C logs to the
   scratch store.
4. **Check D** (new in M3, `checks/check_d_mlflow.py`): reads the scratch store after Check
   C's run and asserts the runs against the notebook's own values (the copy's `defs`).
5. **The notebook tests** (`tests/characterization/test_hyperparameter_tuning_notebook.py`)
   gain a test per sub-task as named below. A script run must leave the store with no run.
6. **Never click "Evaluate on the test part" or "Log the final run" on the user's cache.**

## 4. The sub-tasks

### [ ] M0. The probe (Fable; read-only for the repo)

- **Model:** **Fable** (switch to Fable for this session). It makes design choices from a probe; see §2.
- **Evidence:** the five unconfirmed points of §1; the handoff's §7.8 asks for a first
  sub-task that logs by hand and reads back.
- **Steps:** on Check C's 3-trial copy of `03` (every button clicked but none of the new
  ones, which do not exist yet), in a script in the scratchpad with a store in the scratchpad
  and `PYTHONWARNINGS=error`:
  1. Set the experiment as M1 will; assert the unset URI resolves to `sqlite:///mlflow.db`.
  2. Log one tuning run by hand with M3's calls; one best runs' run with M4's; one final run
     with M6's.
  3. Read each back: `search_runs` by the `run_key` tag; `load_table` equal to the frame
     logged; the artifacts listed; a figure's HTML opened and screenshotted.
  4. Log the same tuning run a second time into the same run and show the table doubled (the
     reason for the skip rule).
  5. Time 24 tuning runs.
  6. Try a record holding its `Study` through Check C's pickling.
  7. Start `mlflow ui` on the scratch store, screenshot the run list, one run's page, a
     table and a figure for the user.
- **Decided here, with the user where it is a choice:** the trials table's form
  (`study.trials_dataframe()` as it is, or with columns converted); M2's form (the record
  keeps `study`, or the logging cell loads it); the dataset's source; any warning filter.
- **Validate:** every read-back equal; the repo unchanged (`git status` clean, md5 equal);
  the repo root's `mlflow.db` and `mlartifacts/` unchanged (their listing before and after).
- **The stop:** the findings written into this file's M1–M6; the screenshots.
- **Commit:**
  ```text
  docs(notebook): the MLflow probe's findings; the runs plan's calls confirmed (M0)
  ```

### [ ] M1. The MLflow cell

- **Model:** Opus 5.5.
- **Evidence:** D1e; MLflow's default URI is the canonical store.
- **Steps:**
  1. `import mlflow` in the imports cell.
  2. One new cell after the data cell: read `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`,
     `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION` from the environment. Raise a `ValueError` naming
     the variable when one is unset, when the URI's file resolves to `project_root /
     "mlflow.db"`, or when the artifact location resolves to `project_root / "mlartifacts"`.
     Then `mlflow.set_tracking_uri`, create the experiment with its artifact location if it
     does not exist, `mlflow.set_experiment`. Print `experiment=… tracking_uri=…`. No run is
     opened.
  3. Return `config_params` from the data cell and `EXPERIMENT_SETTINGS` from the settings
     cell (later cells log them).
  4. `.env.example`: the run command names `notebooks/03_hyperparameter_tuning.py`.
- **Tests:** the script opens no run (the scratch store has none); the canonical URI, the
  unset URI and the canonical artifact location are each refused; `pytest.importorskip("mlflow")`.
- **Validate:** the runner; Check C equal with nothing excepted; Check B's names gain `mlflow`.
- **Mutation:** delete the `raise` for the canonical URI; the refusal test fails.
- **Commit:**
  ```text
  feat(notebook): the MLflow cell sets the experiment and refuses the canonical store (M1)
  ```

### [ ] M2. A run record keeps its study

- **Model:** Opus 5.5 (its form confirmed by M0).
- **Evidence:** the logging needs the study's name (the run's key), its trials and its best
  parameters; the record drops the study the tuner returned. Rebuilding the name and the
  journal path in the logging cell would repeat the tune cell and `src`.
- **Steps** (as recommended; M0 confirms or replaces): in both tune cells add
  `"study": _study` to the record. Nothing else reads it.
- **Validate:** Check C equal except each record's new key `study` (the check scripts drop it
  before pickling and assert its `study_name` equals `build_study_name(...)` for run *i*).
- **Mutation:** store Model 2's study in every Model 1 record; the name assertion fails.
- **Commit:**
  ```text
  feat(notebook): a run record keeps its study (M2)
  ```

### [ ] M3. "Log the tuning runs"

- **Model:** Opus 5.5 (its calls confirmed by M0).
- **Steps:**
  1. A cell with the function `log_tuning_run(record, model, run_settings, study_cache,
     shared_params, datasets, commit)`; it returns the run's id, or `None` when skipped.
     - **The key:** `run_key = f"{study_cache}/{record['study'].study_name}"`
       (`study_cache` is `studies_dir.name`). If `mlflow.search_runs(filter_string=
       f"tags.run_key = '{run_key}'")` has a row, return `None`.
     - **The run:** `mlflow.start_run(run_name=<model, families, exposure>)`.
     - **Tags:** `run_key`, `stage="tuning"`, `model`, the families, `notebook`,
       `git_commit`.
     - **Parameters:** the run settings; `shared_params` (`config_hash`, `DATA_SETTINGS`,
       `EXPERIMENT_SETTINGS`, `n_train`, `n_val`, `n_test`); `study_seed`;
       `feature_transformer.<component>`; `best.<name>` for each of `study.best_params`.
     - **Metrics:** `cv/<objective>`; `<dataset>/<cohort>/<metric>` (48); `seconds`,
       `n_complete`, `n_failed`.
     - **Table:** `trials.json`, the study's `trials_dataframe()` (M0's form).
     - **JSON:** `components.json` (each component's `to_dict()`), `best_params.json`,
       `search_space.json` (the union of the trials' distributions,
       `optuna.distributions.distribution_to_json`), `config.json` (`config_params`).
     - **Artifact:** the journal file, under `study/`.
     - **Datasets:** `log_input` of the training and the validation part
       (`mlflow.data.from_pandas`, `digest=config_hash[:36]`, contexts `training`,
       `validation`).
  2. The button `log_tuning_runs` and its cell: `mo.stop(not log_tuning_runs.value)`; one
     loop over `MODEL1_RUNS` and `MODEL2_RUNS`; a line "logged n, skipped m".
- **Tests:** the logging cell holds `mo.stop(not `; a script run leaves no run.
- **Validate:** Check C equal with nothing excepted. Check D: 24 runs; each run's tags,
  parameters and metrics equal the record's values; `load_table` equal to the study's frame;
  the five artifacts present; a second run of the copy on the same store leaves 24 runs and
  every table's row count unchanged.
- **Mutation:** drop the skip; the second run gives 48 runs and Check D fails.
- **Commit:**
  ```text
  feat(notebook): the 24 tuning runs logged to MLflow behind a button (M3)
  ```

### [ ] M4. "Log the best runs"

- **Model:** Opus 5.5 (its calls confirmed by M0).
- **Steps:** the button `log_best_runs` and one cell (inline; nothing repeats it yet).
  `mo.stop` unless clicked; it needs the inspection's results. Skipped when
  `run_key = f"{study_cache}/best_runs"` exists.
  - **Tags:** `run_key`, `stage="best_runs"`, `best.<model>` the best run's study name,
    `notebook`, `git_commit`.
  - **Parameters:** `shared_params`; the selection metric.
  - **Metrics, the composite only:** `val/<model>/<metric>/<statistic>` for `score`, `mean`,
    `median`, `sd` and the two bounds (72), and `val/difference/<metric>/<statistic>` (36).
    The cohorts' rows are in the tables: 400 metrics would bury the run's page.
  - **Tables:** `best_runs_val_scores_df`, `best_difference_df`, `importance_df`,
    `model1_runs_df`, `model1_scores_df`, `model2_runs_df`, `model2_scores_df`.
  - **Figures (18, HTML):** the forest and the ridge on the selection metric for the runs
    and for the difference (4); per model the permutation figure for the composite and each
    cohort (8) and the beeswarm for each cohort (6). Built by calling the four figure
    functions; the menus are not read.
  - **JSON:** each best run's `components.json`.
  - **Datasets:** as M3.
- **Tests:** the cell holds `mo.stop(not `; a script run leaves no run.
- **Validate:** Check C equal; Check D: 25 runs; the tables equal the notebook's frames; the
  18 figure files present; the composite metrics equal the frames' rows; the second run skips.
- **Mutation:** log `best_difference_df` under the scores table's file name; Check D fails.
- **Commit:**
  ```text
  feat(notebook): the best runs' comparison logged to MLflow as one run (M4)
  ```

### [ ] M5. The count figures and the shares bars as functions

- **Model:** Opus 5.5.
- **Evidence:** the two count figures and the shares bars are built inline from menus; M6
  logs them at a fixed state, so each is now built twice: a function is justified.
- **Steps:** move each figure's body into a function in its cell (`counts_histogram_figure`,
  `counts_scatter_figure`, `selection_shares_bars_figure`), taking its data, the labels and
  the palette; the cell calls it with the menus' values. The exported names stay.
- **Validate:** Check C equal with nothing excepted; `03`'s 21 figures' JSON identical to
  `figures_s5_3_tuning.json`.
- **Mutation:** swap the scatter's axes in the function; the figure comparison fails.
- **Commit:**
  ```text
  refactor(notebook): the count figures and the shares bars each built by a function (M5)
  ```

### [ ] M6. "Log the final run"

- **Model:** Opus 5.5 (its calls confirmed by M0).
- **Steps:** the button `log_final_run` and one cell. `mo.stop` unless clicked; it needs the
  test evaluation's results. **The guard:** if `search_runs(filter_string="tags.final =
  'true'")` has a row, `mo.stop` with a message naming that run; nothing is logged.
  - **Tags:** `run_key`, `stage="final"`, `final="true"`, `best.<model>`, `notebook`,
    `git_commit`.
  - **Parameters:** `shared_params`; the selection metric; `n_train_val`.
  - **Metrics, the composite only:** `test/<model>/<metric>/<statistic>` and
    `test/difference/<metric>/<statistic>`. No other run holds a `test/` metric.
  - **Tables:** `test_scores_df`, `test_difference_df`, `val_against_test_df`,
    `test_importance_df`, `test_buildings_df`, `test_building_scores_df`.
  - **Figures:** as M4 on the test part (18); per model the two count figures for the
    composite (4); the shares bars for all neighborhoods.
  - **JSON:** each test model's `components.json`.
  - **Datasets:** train + validation as `training`, the test part as `testing`.
- **Tests:** the cell holds `mo.stop(not `; a script run leaves no run.
- **Validate:** Check C equal; Check D: 26 runs, one `final=true`; no `test/` metric in the
  other 25; the tables equal; a second run of the copy leaves 26 and shows the refusal.
- **Mutation:** drop the guard; the second run gives two final runs and Check D fails.
- **Commit:**
  ```text
  feat(notebook): the final run logged to MLflow once, refused when one exists (M6)
  ```

### [ ] M7. Docs and close

- **Model:** Opus 5.5.
- **Steps:** `docs/MLFLOW_RUNS.md` (how to run the notebook and the UI on the scratch store;
  what each of the three runs holds; the skip rule and the final run's guard; that
  `MLFLOW_EXPERIMENTS_GUIDE.md` and `tracking/` are the old stack's); `docs/README.md`,
  `MODULE_REFERENCE.md` and the root `README.md` link it; a dated note on E9, E10, D1e–D5 in
  `EXPERIMENT_NOTEBOOK_PLAN.md`; this plan's status "complete"; the handoff.
- **Validate:** the guide's commands run as written on a scratch store; the links resolve;
  the full runner once more.
- **Commit:**
  ```text
  docs(notebook): the MLflow runs guide; the experiment plan's MLflow decisions closed (M7)
  ```

## 5. Where each item of the inventory goes

| Item | Where | Note |
|---|---|---|
| the run's settings, `use_exposure`, the sharing flags | tuning run, parameters | |
| `config_hash`, the data settings | every run, parameters | |
| the config's dotted keys | tuning run, `config.json` | the same in every run; the hash stands for it as a parameter |
| the split's method and sizes, `n_train` / `n_val` / `n_test` | every run, parameters | |
| `seed`, the study's seed | parameters | |
| `n_trials`, `n_startup_trials`, `n_splits`, the objective | every run, parameters | |
| the search space | tuning run, `search_space.json` | too long for a parameter |
| the best parameters; the components' `to_dict()` | tuning run, parameters `best.*`, `best_params.json`, `components.json` | |
| `cv_best`, the point scores, `seconds`, `n_complete`, `n_failed` | tuning run, metrics | |
| the bootstrap statistics, the paired difference | best runs' run: the composite as metrics, all rows as tables | |
| the test scores | the final run only | |
| the trials dataframe of each study | tuning run, `trials.json` | |
| the runs and scores tables | best runs' run, tables | a row of them is a tuning run |
| `importance_df`; `test_importance_df` | best runs' run; final run | |
| `val_against_test_df`, `test_buildings_df`, `test_building_scores_df` | final run, tables | |
| the EDA's trials tables, `eda_best_df`, its 27 figures | **left out** | decision 7 |
| the forest and the ridge, the permutation figures, the beeswarms | best runs' run (validation), final run (test) | a fixed set |
| the two count figures, the shares bars | final run | the composite; all neighborhoods |
| the waterfalls | **left out** | each is one building (decision 10) |
| the fitted model | **left out** | decision 4; rebuilt from `components.json` |
| the study's journal file | tuning run, artifact | |
| the feature transformers' description | tuning run: their names as parameters, their declaration in `components.json` | |
| the bootstrap replicates | **left out** | no reader: the ridge figure shows them, and the seed recomputes them |
| the model's label, the family, the stage, the git commit, the notebook | every run, tags | |
| the dataset (its hash, its rows) | every run, `log_input` | digest the hash's first 36 characters |

## 6. Questions still open

- M0's five points (§1), each settled by the probe.
- A read-back table (decision 6) can be added after M6 if the UI alone proves not enough.
- The old `tracking/` package's deletion stays the roadmap's step.

## 7. The prompt for the first implementing session (M0, a Fable session)

```text
Read docs/MLFLOW_RUNS_PLAN.md in full, then docs/EXPERIMENT_NOTEBOOK_HANDOFF.md: the status at
the top, §4 and §7. Then the cells M3, M4 and M6 name in notebooks/03_hyperparameter_tuning.py.
Verify the state first: the branch feat/mlflow-runs, HEAD my commit of the MLflow plan,
`git status` clean; the notebooks' md5 as the handoff's top status; no iCloud-offloaded files.
If anything differs, stop and show me the diff. Copy the scratchpad the top status names into
yours, set each script's S= to it, compile the pycache.
Then do M0 only (the probe), planned first in plan mode: on Check C's 3-trial copy, with a
store in your scratchpad and PYTHONWARNINGS=error, log one run of each kind by hand with the
plan's calls and read each back. Settle the plan's five unconfirmed points; ask me, with the
interactive question tool and screenshots of the MLflow UI, where a point is a choice. Then
write the findings into the plan's M1-M6 so that Opus can implement them exactly.
Edit no notebook, no src file and no test. Never write the repo root's mlflow.db or
mlartifacts/, nor .mlflow-local/. Don't run the test evaluation on my cache (E11). Stop with a
summary, the notebooks' md5 (unchanged) and the commit message in a copyable code block; tick
M0 and update the status lines. I commit myself: never commit, no Co-Authored-By line, no
"Generated with Claude Code" in PR text, never git checkout/stash/reset/mv.
```

**The prompt for each later session** (an Opus 5.5 session; replace `<M>` with the sub-task,
M1 to M7, in order):

```text
Read docs/MLFLOW_RUNS_PLAN.md in full (the status, §1's decisions and facts, §2, §3 how a
sub-task is done, §4's <M>), then docs/EXPERIMENT_NOTEBOOK_HANDOFF.md: the status at the top
and §4 (how to work, git and the editor, the checks, the traps, the iCloud notes, the user's
style). Then the cells <M> names in notebooks/03_hyperparameter_tuning.py.
Verify the state first: the branch feat/mlflow-runs, HEAD my commit of the previous sub-task,
`git status` clean; the notebooks' md5 as the handoff's top status; no iCloud-offloaded files.
If anything differs, stop and show me the diff. Copy the scratchpad the top status names into
yours, set each script's S= to it, compile the pycache. If that scratchpad is gone, tell me
before rewriting anything.
Then do sub-task <M> only, exactly as the plan describes: plan it first in plan mode (re-verify
its facts and line numbers read-only, list the exact edits), wait for my go, implement it, run
the checks of the plan's §3 and <M>'s validation and mutation. Do not start the next sub-task.
If a step needs a choice the plan does not make, stop and tell me it needs a Fable session.
Never write the repo root's mlflow.db or mlartifacts/, nor .mlflow-local/: every check uses a
store in your scratchpad. Don't run the test evaluation or "Log the final run" on my cache
(E11). My marimo session may be open: tell me before editing the notebook and remind me to
reload it after. Keep it simple; no private helpers; a notebook function reads no notebook
global. Stop with a file-by-file summary, the checks, the notebooks' md5 and the commit message
in a copyable code block; tick <M> and update the status lines in the plan and at the top of
the handoff. I commit myself: never commit, no Co-Authored-By line, no "Generated with Claude
Code" in PR text, never git checkout/stash/reset/mv.
```
