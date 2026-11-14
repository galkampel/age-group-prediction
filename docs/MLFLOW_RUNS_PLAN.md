# Plan: the MLflow runs of the tuning notebook

**For:** the user, and the session that implements a sub-task (written so that Claude Opus 5.5
can implement one from this file and the handoff's §4). One sub-task per approval. Tick a
sub-task when it is done and update the status line.

**Status (2026-11-14):** M0 done (the probe; its findings in §1 and in M1–M6's exact calls; the
user's four choices 11–14), committed `8e1ea98`. M1 done (the MLflow cell; the user's choice
there: no MLflow global), committed `2451c8a`. M2 done (a run record keeps its study), checked,
awaiting the user's commit. Next: M3, an Opus 5.5 session (§7, the second prompt).

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

**Confirmed by M0 (2026-11-14; the probe on Check C's 3-trial copy, `PYTHONWARNINGS=error`,
a store in the scratchpad; every read-back asserted):**
- **Two calls warn under pandas 3.0.5, and both are filtered** (the user's choice, below):
  - `log_table` on any frame with a `str` column: MLflow's own
    `select_dtypes(include=["object"])` raises `pandas.errors.Pandas4Warning` ("For backward
    compatibility, 'str' dtypes are included by select_dtypes…", attributed to
    `mlflow.tracking.client`). Under `-W error` every table fails.
  - `log_input` of a frame with int columns (X's `n_daycares_500m`, `n_apartments`; Y):
    `UserWarning` "Hint: Inferred schema contains integer column(s)…" (attributed to
    `mlflow.types.utils`), raised inside a `try/except Exception` in MLflow, so under `-W error`
    the schema is silently dropped (`None`), and in the notebook the warning prints.
  - With both filtered, nothing else warns: `import mlflow`, the store's creation, every call of
    M3, M4 and M6, `search_runs`, `load_table`, `load_dict`.
  - **MLflow's computed digest also warns** (`digest_utils`: "all arguments of all will be
    keyword-only"): a dataset must be given its digest.
- **`trials_dataframe()` through `log_table`:** its datetimes come back as ISO strings and
  `duration` as `"P0DT0H0M2.636103S"` (not equal). The converted form (§4 M3) reads back
  exactly equal. The UI's Artifacts tab shows a `log_table` file as a table ("View as table").
- **The source with none given:** `CodeDatasetSource` holding the run's `mlflow.user`,
  `mlflow.source.name` (the running script's path) and `mlflow.source.type` (`LOCAL`); kept.
- **Wall time:** one tuning run 0.1–0.15 s; the 24 in 1.0 s; the second pass (24 skips)
  0.1 s; the best runs' run 0.3–0.4 s; the final run 0.3–0.4 s (3-trial journals).
- **A record keeping its `Study`:** `pickle` and cloudpickle both work (3,008 bytes for a
  3-trial study, reloaded with its trials). Check C's records still drop it (a `Study` has no
  `==` for the reference comparison).
- **Read back, each equal:** params as `str(value)` (63 in the copy's Model 1 run 0); metrics as
  `float`; tags; the four JSONs after a JSON round trip; the journal's bytes; 12 of the 13
  tables exactly (the dict columns of `model1_runs` too). `test_building_scores` differs only by
  its columns' index name (`metric`, from `compute_metrics`): JSON drops it.
- **Also seen:** a missing folder of the sqlite file is created by MLflow; the artifact folder
  is created at the first artifact; a new store prints two alembic `INFO` lines to stderr
  ("Creating initial MLflow database tables…"); `set_experiment` of a missing experiment prints
  an `INFO` line and creates it without our artifact location (so M1 creates it first); an unset
  URI resolves to `sqlite:///<cwd>/mlflow.db` and creates nothing until a store call; a run
  that raises inside `with mlflow.start_run()` ends `FAILED` with its tags; `log_table` into an
  existing file doubles the table (3 rows → 6); the run's page groups metric charts by the
  name's prefix (`val/difference/bias (6)`); a figure's HTML loads plotly.js from
  `cdn.plot.ly` (needs the network; 13–56 KB per file) and renders in the UI's viewer;
  `load_table` and `load_dict` print a tqdm "Downloading artifacts" bar on stderr unless
  `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false`.
- **The UI** (`mlflow ui`, 3.16): opens in its "GenAI" mode with an "MLflow Assistant" panel;
  the runs list under the experiment; the screenshots are named in the handoff's top status.

**The user's choices at M0 (2026-11-14; do not re-ask):**

11. **The trials table:** the converted form: `number, value, seconds, params_*,
    user_attrs_*, state` (the start and end times dropped; the journal keeps them).
12. **The two warnings:** two narrow filters (category, message start, module) set once in the
    MLflow cell; not a cast at every call.
13. **The datasets' digest:** the SHA-256 of the part's rows (pandas' row hashes), its first
    36 characters; `config_hash` stays a parameter. A changed split gets its own dataset record.
14. **The git tags:** `git_commit` (HEAD) and `git_dirty` (`"true"` when a tracked file is
    modified), read once in the MLflow cell with `git` run in `project_root / "src"` (in Check
    C's copy a symlink to the repo, where `project_root` itself is not a repository).

**Settled by the probe without a question:** M2's form, as the plan recommended: the record
keeps its study (it pickles; the logging needs its name, trials and best parameters, and loading
it again would repeat the tune cell's name). The skip rule and the final run's guard match a
`FINISHED` run only, so a run that failed mid-logging is logged again (decision 3's "a run that
exists" is a complete one); the failed run stays in the UI marked failed.

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
   C's run and asserts the runs against the notebook's own values (the copy's `defs`). The
   model is M0's probe (`probes/probe_m0.py` in the scratchpad the handoff names): its
   read-back block and `compare_table`. Run it with `MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false`
   (else a tqdm bar per download). Params compare as `str(value)`, metrics as `float(value)`,
   the JSONs after `json.loads(json.dumps(...))`, a table with `pd.testing.assert_frame_equal`
   (exact), `test_building_scores_df` after `.rename_axis(columns=None)`.
5. **The notebook tests** (`tests/characterization/test_hyperparameter_tuning_notebook.py`)
   gain a test per sub-task as named below. A script run must leave the store with no run.
6. **Never click "Evaluate on the test part" or "Log the final run" on the user's cache.**

## 4. The sub-tasks

### [x] M0. The probe (Fable; read-only for the repo)

**Done 2026-11-14.** The probe (`probes/probe_m0.py`, run by `probes/run_probe_m0.sh`; log
`run_probe_m0.log`, the first run's `m0_shots/run_probe_m0_first.log`) ended `PROBE M0 DONE`
under `-W error` with every read-back asserted; its first run stopped on `log_table`'s
`Pandas4Warning`, the second ran with the filter, the third with the user's four choices. The
findings are §1's "Confirmed by M0"; the choices §1's 11–14; M1–M6 below hold the calls as
run. The repo unchanged: `git status` clean, the notebooks' md5 equal, the root's `mlflow.db`
(md5, size, mtime) and `mlartifacts/` (411 entries, a stat listing) equal before and after, no
`.mlflow-local/`. The screenshots: `m0_shots/` (the run list, a tuning run's page, both trials
forms, a figure in the UI's viewer, the metric charts, the final run's table and figure).

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

### [x] M1. The MLflow cell

**Done 2026-11-14** (`03`'s md5 `659e410dc1b64a68501f4f3e331875a3`). `checks/run_M1.sh` (log
`run_M1.log`): the non-slow suite 1416 passed (S5.3's 1412 + 4), mypy clean, the notebook tests
36 (32 + 4), ruff `03` 15 / `02` 11 as at HEAD (none new), the format hunks the user's two, `marimo
check` clean, Check B on both (M1's assertions), Check C on both equal to S5.3 with nothing
excepted, both notebooks' 21 + 27 figures identical, the scratch store 0 runs, the repo's
`mlflow.db` and `mlartifacts/` unchanged. The mutation (`run_mutant_M1.sh`): the copy as is 3
passed; without the canonical URI's `raise`, the `sqlite:///mlflow.db` case fails.

- **Model:** Opus 5.5.
- **Evidence:** D1e; MLflow's default URI is the canonical store.
- **The user's choice at M1 (2026-11-14; do not re-ask): no MLflow global.** The notebook test
  fixture `notebook_defs` runs the notebook inside the pytest process, and the old `tracking/`
  tests (not marked slow either) run later in the same session; `mlflow.set_tracking_uri`
  writes a module global and `os.environ["MLFLOW_TRACKING_URI"]`, `set_experiment` a global
  active experiment. So the cell calls neither: MLflow 3.16 reads `MLFLOW_TRACKING_URI` and,
  with no active experiment, `MLFLOW_EXPERIMENT_NAME` at every call
  (`fluent._get_experiment_id` → `_get_experiment_id_from_env`). Confirmed before the edit by
  `probes/probe_m1_env.py` under `-W error`: the run in the named experiment, its artifacts
  under the artifact location, the table equal, the schema logged, the URI and the
  environment unchanged. **M3, M4 and M6 call no setter either** (their skip and guard
  queries, `start_run`, `load_table` use the active experiment the variable names).
- **Steps (done):**
  1. **The MLflow cell**, after the data cell (`def _(pd, project_root)`): its own imports
     (`os`, `subprocess`, `warnings`, `Path`, `mlflow`; not the setup's imports cell, `02`'s
     twin); the six refusals (each a `ValueError` naming the variable: the URI unset, not
     `sqlite:///`, or its file `project_root / "mlflow.db"`; the experiment unset; the
     artifact location unset or `project_root / "mlartifacts"`); the two warning filters (§1,
     12); `create_experiment(name, artifact_location=str(Path(location).resolve()))` when
     missing; `git_tags` (`git_commit`, `git_dirty`, `git` in `project_root / "src"`); the line
     `experiment=… tracking_uri=…`. It returns `git_tags, mlflow`: the logging cells read
     both, so marimo runs them after it (the filters and the experiment set first).
  2. The data cell returns `config_params`; the settings cell `EXPERIMENT_SETTINGS`.
  3. `.env.example`: the run command names `notebooks/03_hyperparameter_tuning.py`.
- **Tests (done):** `_mlflow_environment(scratch)` (the four variables of a scratch store);
  `_run(notebook, scratch, cwd, environment)` (a `None` removes a variable);
  `_project_copy(tmp_path, source)` (a copy with `configs` **and `src`** linked; the
  seed-override test's copy had no `src`, where the cell's `git` would raise);
  `notebook_store` and `notebook_defs` (the variables set for the in-process run only, in
  `pytest.MonkeyPatch.context()`); `test_a_script_run_opens_no_mlflow_run` (the experiment with
  its artifact location, no run, the git tags, no URI leaked into the environment or into
  `mlflow.get_tracking_uri()`); `test_the_mlflow_cell_refuses_the_canonical_store` (×3:
  `MLFLOW_TRACKING_URI` unset, `sqlite:///mlflow.db`, `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION=
  mlartifacts`; each on `_project_copy` run from the copy's root, so a missing refusal opens
  the copy's store, never the repo's; exit ≠ 0, `ValueError` and the variable in stderr, no
  `mlflow.db` or `mlartifacts` in the copy).
- **Validate:** `checks/run_M1.sh` (the four variables exported to `$S/mlflow_M1.db` and
  `$S/mlartifacts_M1`; Check B `check_b_m1_tuning.py`: `mlflow`, `git_tags`, `config_params`,
  `EXPERIMENT_SETTINGS`, the experiment with its artifact location, no run, no active
  experiment); Check C on both equal to S5.3 with nothing excepted; both notebooks' figures
  identical; the repo's store listed before and after.
- **Mutation:** `checks/run_mutant_M1.sh`: on a full copy, the canonical URI's `raise`
  replaced by `pass`; the `sqlite:///mlflow.db` refusal case fails.
- **The scripts every later check needs:** the four MLflow variables exported (the notebook
  refuses to run without them), as `run_M1.sh` does; `run_dump_m1.sh` is the figure dump
  without its inline URI.
- **Commit:**
  ```text
  feat(notebook): the MLflow cell sets the experiment and refuses the canonical store (M1)
  ```

### [x] M2. A run record keeps its study

**Done 2026-11-14** (`03`'s md5 `74b39c08ded5cafbbbb68016d7f87649`). Both tune cells' records
gain `"study": _study` after `"components"`, and their comment says the logging reads the
study's name, trials and best parameters. `checks/run_M2.sh` (log `run_M2.log`): the non-slow
suite 1416, mypy clean, the notebook tests 36, ruff `03` 15 / `02` 11 (none new), the two format
hunks, Check B on both, Check C on both equal to M1's run with nothing excepted (the records
compared without `study`), the 21 + 27 figures identical, the scratch store 0 runs, the repo's
store unchanged. Check C (`checks/check_c_m2_tuning.py`) asserts each record's study against the
study it recomputes in memory: its name (the check's own `study_file`, not the notebook's
`build_study_name`), its best value equal to `cv_best`, 3 trials, each trial's params; its
`record()` drops `study` before pickling. **The mutation as run** (the line below could not be
written: Model 1's cell has no Model 2 study): Model 1's records each keep the first run's study
(`run_mutant_M2.sh`); Check C fails at run 1 (seed 43, the study named `…seed=42`).

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
- **Steps** (the calls as M0 ran them, `probes/probe_m0.py`'s `log_tuning_run`; four new cells
  after Model 2's tables cell, in this order):
  1. **The runs' shared inputs** (no button; cheap): reads `DATA_SETTINGS`,
     `EXPERIMENT_SETTINGS`, `config_hash`, `hashlib`, `mlflow`, `pd`, the three parts.
     ```python
     # What every run logs as parameters: the data's version and settings, the experiment's
     # settings (the selection metric among them) and the parts' sizes.
     shared_params = {
         "config_hash": config_hash, **DATA_SETTINGS, **EXPERIMENT_SETTINGS,
         "n_train": len(train_age_group.X), "n_val": len(val_age_group.X), "n_test": len(test_age_group.X),
     }

     def build_dataset(part, name):
         """`part` as an MLflow dataset named `name`: its X and Y side by side, the digest the
         SHA-256 of its rows (pandas' row hashes), its first 36 characters (MLflow's limit)."""
         frame = part.X.join(part.Y)
         digest = hashlib.sha256(pd.util.hash_pandas_object(frame).to_numpy()).hexdigest()[:36]
         return mlflow.data.from_pandas(frame, name=name, digest=digest)

     tuning_datasets = {
         "training": build_dataset(train_age_group, "train"),
         "validation": build_dataset(val_age_group, "val"),
     }
     ```
     It returns `build_dataset`, `shared_params`, `tuning_datasets` (22 parameters; M6 builds
     its two datasets with `build_dataset`). `from_pandas` with a digest warns nothing; without
     one MLflow's digest raises under `-W error` (M0). No `source`: MLflow records the running
     script (`CodeDatasetSource`). `targets` stays unset (it takes one column; Y has three).
  2. **The function** (reads `mlflow`, `optuna`):
     ```python
     def log_tuning_run(record, model, run_settings, study_seed, studies_dir, objective_name,
                        shared_params, config_params, datasets, git_tags):
         """One study's run in the active experiment; its id, or None when a FINISHED run
         has its key (a FAILED one is logged again)."""
         study = record["study"]
         run_key = f"{studies_dir.name}/{study.study_name}"
         if mlflow.search_runs(
             filter_string=f"tags.run_key = '{run_key}' and attributes.status = 'FINISHED'",
             output_format="list",
         ):
             return None
         if model == "independent_cohorts":
             family = run_settings["family"]
         elif model == "total_times_probability":
             family = f"{run_settings['total_family']}+{run_settings['probability_family']}"
         else:
             raise ValueError(f"unknown model {model!r}")
         # The trials: a row per trial, its seconds a number (a timedelta would come back as
         # text); the start and end times are in the journal.
         trials = study.trials_dataframe(
             attrs=("number", "value", "duration", "params", "user_attrs", "state")
         )
         trials["duration"] = trials["duration"].dt.total_seconds()
         trials = trials.rename(columns={"duration": "seconds"})
         # The search space the trials drew: each parameter's distribution, as Optuna's JSON.
         search_space = {}
         for trial in study.trials:
             for name, distribution in trial.distributions.items():
                 search_space[name] = optuna.distributions.distribution_to_json(distribution)
         with mlflow.start_run(run_name=f"{model} {family} exposure={run_settings['use_exposure']}") as run:
             mlflow.set_tags({"run_key": run_key, "stage": "tuning", "model": model, "family": family,
                              "notebook": "notebooks/03_hyperparameter_tuning.py", **git_tags})
             mlflow.log_params({
                 **run_settings, **shared_params, "study_seed": study_seed,
                 **{f"feature_transformer.{name}": value for name, value in record["feature_transformers"].items()},
                 **{f"best.{name}": value for name, value in study.best_params.items()},
             })
             mlflow.log_metrics({
                 f"cv/{objective_name}": record["cv_best"],
                 **{f"{dataset}/{cohort}/{metric}": score
                    for (dataset, cohort, metric), score in record["scores"]["score"].items()},
                 "seconds": record["seconds"], "n_complete": record["n_complete"], "n_failed": record["n_failed"],
             })
             mlflow.log_table(trials, "trials.json")
             mlflow.log_dict({c.name: c.to_dict() for c in record["components"]}, "components.json")
             mlflow.log_dict(study.best_params, "best_params.json")
             mlflow.log_dict(search_space, "search_space.json")
             mlflow.log_dict(config_params, "config.json")
             mlflow.log_artifact(str(studies_dir / f"{study.study_name}.log"), artifact_path="study")
             for context, dataset in datasets.items():
                 mlflow.log_input(dataset, context=context)
         return run.info.run_id
     ```
     (M0 asserted the union has one distribution per name; a parameter no trial drew is
     absent.) The run key is 260 characters in the copy (a tag holds 8,000); the run has 52
     metrics and, in the copy's Model 1 run 0, 63 parameters (fewer with the shared
     hyperparameters).
  3. **The button** `log_tuning_runs = mo.ui.run_button(label="Log the tuning runs")`, with a
     line saying it logs the 24 runs to the experiment the MLflow cell names and skips a
     logged one.
  4. **The logging cell:** `mo.stop(not log_tuning_runs.value, mo.md("Click the button to
     log the tuning runs to MLflow."))`; then
     ```python
     _logged = _skipped = 0
     for _model, _settings_list, _runs in (
         ("independent_cohorts", MODEL1_RUN_SETTINGS, MODEL1_RUNS),
         ("total_times_probability", MODEL2_RUN_SETTINGS, MODEL2_RUNS),
     ):
         for _i, (_run_settings, _record) in enumerate(zip(_settings_list, _runs, strict=True)):
             _run_id = log_tuning_run(_record, _model, _run_settings, seed + _i, tuner.storage_dir,
                                      tuner.objective_metric.name, shared_params, config_params,
                                      tuning_datasets, git_tags)
             _logged += _run_id is not None
             _skipped += _run_id is None
     mo.md(f"Logged {_logged} tuning runs, skipped {_skipped} (already logged).")
     ```
     M0's times: 24 runs in 1.0 s, 24 skips in 0.1 s (3-trial journals).
- **Tests:** the logging cell holds `mo.stop(not `; a script run leaves no run.
- **Validate:** Check C equal with nothing excepted. Check D: 24 runs; each run's tags (with
  `git_commit`, `git_dirty`), parameters (`str`) and metrics (`float`) equal the record's
  values; `load_table("trials.json", run_ids=[id])` exactly equal to the converted trials of
  `record["study"]`; the four JSONs equal; the journal's bytes equal the copy's file; two
  dataset inputs (contexts `training`, `validation`), their digests `build_dataset`'s, their
  schema not empty (the filter works: under `-W error` without it the schema is silently
  `None`); a second run of the copy on the same store leaves 24 runs and every trials table 3
  rows.
- **Mutation:** drop the skip; the second run gives 48 runs and Check D fails.
- **Commit:**
  ```text
  feat(notebook): the 24 tuning runs logged to MLflow behind a button (M3)
  ```

### [ ] M4. "Log the best runs"

- **Model:** Opus 5.5 (its calls confirmed by M0).
- **Steps** (the calls as M0 ran them; three new cells after the beeswarm cell of section 5):
  1. **The figures' function** (M6 calls it again: decision 5). It reads no global; the four
     figure functions are its arguments:
     ```python
     def build_part_figures(part, scores_df, replicate_scores, difference_df, difference_replicates,
                            permutation, explanations, X, metric_name, bound_names, seed, palette,
                            forest_figure, ridge_figure, permutation_figure, beeswarm_figure):
         """The 18 figures of one part (`part`: "val" or "test") by artifact path: the forest
         and the ridge of `metric_name` for the two runs and for their difference; per model
         the permutation figure of the composite and each cohort and the beeswarm of each
         cohort. `permutation` is {model: {cohort (the composite first): {split: result}}}."""
         difference_name = difference_df["model"].iloc[0]
         figures = {
             f"figures/{part}_forest_runs_{metric_name}.html": forest_figure(
                 scores_df, metric_name, bound_names["lower"], bound_names["upper"], palette),
             f"figures/{part}_ridge_runs_{metric_name}.html": ridge_figure(
                 replicate_scores, metric_name, palette),
             f"figures/{part}_forest_difference_{metric_name}.html": forest_figure(
                 difference_df, metric_name, bound_names["lower"], bound_names["upper"], palette,
                 share_x_axis=True, rule_at=0),
             f"figures/{part}_ridge_difference_{metric_name}.html": ridge_figure(
                 {difference_name: difference_replicates}, metric_name, palette,
                 share_x_axis=True, rule_at=0),
         }
         for model, by_cohort in permutation.items():
             for cohort, by_split in by_cohort.items():
                 figures[f"figures/{part}_permutation_{model}_{cohort}.html"] = permutation_figure(
                     by_split, list(X.columns), f"{cohort} {metric_name}", palette)
             for cohort in explanations[model].output_names:
                 figures[f"figures/{part}_beeswarm_{model}_{cohort}.html"] = beeswarm_figure(
                     explanations[model], X, cohort, seed, palette)
         return figures
     ```
  2. **The button** `log_best_runs = mo.ui.run_button(label="Log the best runs")`.
  3. **The logging cell.** `mo.stop(not log_best_runs.value, …)`; it reads the inspection's
     `best_runs_permutation`, `best_runs_shap` (so it runs only after that click). Then:
     - **Skip:** `_key = f"{tuner.storage_dir.name}/best_runs"`; when
       `mlflow.search_runs(filter_string=f"tags.run_key = '{_key}' and attributes.status =
       'FINISHED'", output_format="list")` is not empty, `mo.stop(True, mo.md("The best runs'
       run is logged: skipped."))`.
     - `with mlflow.start_run(run_name="best runs"):`
     - **Tags:** `{"run_key": _key, "stage": "best_runs", **{f"best.{model}":
       run["study"].study_name for model, run in best_runs.items()}, "notebook":
       "notebooks/03_hyperparameter_tuning.py", **git_tags}`.
     - **Parameters:** `shared_params` (the selection metric is in it, from
       `EXPERIMENT_SETTINGS`).
     - **Metrics, the composite only (108):** with `_statistics = ["score", "mean", "median",
       "sd", CI_BOUND_NAMES["lower"], CI_BOUND_NAMES["upper"]]`, for each composite row of
       `best_runs_val_scores_df` `f"val/{row['model']}/{row['metric']}/{statistic}"` (72), and
       of `best_difference_df` `f"val/difference/{row['metric']}/{statistic}"` (36). The UI
       groups each model and metric's six statistics in one chart section. The cohorts' rows
       are in the tables.
     - **Tables:** `mlflow.log_table(frame, f"tables/{name}.json")` for `best_runs_val_scores`,
       `best_difference`, `importance`, `model1_runs`, `model1_scores`, `model2_runs`,
       `model2_scores` (each the `<name>_df` frame; all read back exactly equal in M0, the dict
       columns too).
     - **Figures (18):** `for path, figure in build_part_figures("val",
       best_runs_val_scores_df, best_replicate_scores, best_difference_df,
       best_difference_replicates, best_runs_permutation, best_runs_shap, shap_rows.X,
       SELECTION_METRIC.name, CI_BOUND_NAMES, seed, PALETTE, forest_figure, ridge_figure,
       permutation_figure, beeswarm_figure).items(): mlflow.log_figure(figure, path)`
       (`log_figure` needs the active run: inside the `with`). Each file 13–56 KB, plotly.js
       from the CDN.
     - **JSON:** `mlflow.log_dict({c.name: c.to_dict() for c in run["components"]},
       f"components/{model}.json")` for each best run.
     - **Datasets:** `tuning_datasets`, as M3.
  M0 logged it in 0.3–0.4 s.
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
- **Steps** (the calls as M0 ran them; two new cells after the shares bars' cell):
  1. **The button** `log_final_run = mo.ui.run_button(label="Log the final run")`, with a
     line saying it logs the test evaluation once per experiment.
  2. **The logging cell.** `mo.stop(not log_final_run.value, …)`; it reads the test
     evaluation's and the test explanation's names (so it runs only after that click).
     - **The guard:**
       ```python
       _finals = mlflow.search_runs(
           filter_string="tags.final = 'true' and attributes.status = 'FINISHED'",
           output_format="list",
       )
       if _finals:
           mo.stop(True, mo.md(f"A final run exists: {_finals[0].info.run_name} "
                               f"({_finals[0].info.run_id}); nothing logged."))
       ```
       (an `if`, not `mo.stop(bool(_finals), …)`: the message would index an empty list).
     - `with mlflow.start_run(run_name="final"):`
     - **Tags:** `{"run_key": f"{tuner.storage_dir.name}/final", "stage": "final", "final":
       "true", **{f"best.{model}": run["study"].study_name for model, run in
       best_runs.items()}, "notebook": "notebooks/03_hyperparameter_tuning.py", **git_tags}`.
     - **Parameters:** `{**shared_params, "n_train_val": len(train_val_age_group.X)}`.
     - **Metrics, the composite only (108):** as M4 with the prefix `test/`, from
       `test_scores_df` and `test_difference_df`. No other run holds a `test/` metric.
     - **Tables:** `tables/<name>.json` for `test_scores`, `test_difference`,
       `val_against_test`, `test_importance`, `test_buildings`, `test_building_scores`.
     - **Figures (23):** `build_part_figures("test", test_scores_df, test_replicate_scores,
       test_difference_df, test_difference_replicates, {model: {cohort: {"test": result} for
       cohort, result in by_cohort.items()} for model, by_cohort in test_permutation.items()},
       test_shap, test_age_group.X, SELECTION_METRIC.name, CI_BOUND_NAMES, seed, PALETTE,
       forest_figure, ridge_figure, permutation_figure, beeswarm_figure)` (18); per model M5's
       `counts_histogram_figure` and `counts_scatter_figure` for the composite, as
       `figures/test_counts_histogram_<model>_composite.html` and
       `figures/test_counts_scatter_<model>_composite.html` (4); M5's
       `selection_shares_bars_figure` for all the test buildings, as
       `figures/test_selection_shares_bars_all.html` (1).
     - **JSON:** `components/<model>.json` for each best run's components (the test models
       are refit from them).
     - **Datasets:** `{"training": build_dataset(train_val_age_group, "train_val"),
       "testing": build_dataset(test_age_group, "test")}`, each `log_input` with its context.
  M0 logged it in 0.3–0.4 s; a second click showed the refusal and left one final run.
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
  `MLFLOW_EXPERIMENTS_GUIDE.md` and `tracking/` are the old stack's; M0's notes on the UI:
  `mlflow ui --backend-store-uri <the .env URI> --artifacts-destination <a scratch folder>` run
  from the repo root would default its artifact destination to the canonical `./mlartifacts`,
  so the guide names one; the UI opens in its "GenAI" mode with an assistant panel; a
  figure loads plotly.js from the CDN, so it needs the network; the two warning filters and
  why); `docs/README.md`,
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
| the dataset (its hash, its rows) | every run, `log_input` | the digest the rows' SHA-256, its first 36 characters (choice 13); `config_hash` a parameter |

## 6. Questions still open

- None from M0: its five points are settled (§1, "Confirmed by M0"), its choices made (§1,
  11–14).
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
