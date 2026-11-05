# Plan: the study cache, the bootstrap replicates, the notebook split, SHAP per cohort and the test evaluation

**For:** the sessions that close PR 3 (3.3) and then build PRs 4–7, one sub-task per approval.
Kept updated with each sub-task's Done note; tick the sub-task and update the status line.

**Status (2026-11-04):** written after 3.4 (`afcd6f9`) and the handoff (`b7da88d`), with the user's
list of changes and their answers (§0). 3.3's checks already run: the non-slow suite 1387 passed,
1 skipped, 1 xfailed, 19 deselected (7 warnings, the old stack's); mypy 28 files, no issue. The
notebook's md5 is `676212a808f6c1f87f7926a58b3bffe6`. **Next: 3.3 (§2).**

The companion docs: [EXPERIMENT_NOTEBOOK_HANDOFF.md](EXPERIMENT_NOTEBOOK_HANDOFF.md) (the state,
how to work, the checks, the traps, iCloud), [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)
(PRs 1–3, M1–M14), [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §0 (D1–D30),
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) (E11, the final evaluation).

## Contents

- §0 Context and decisions (D31–D39)
- §1 How to work: the rules, the checks, the reference run
- §2 PR 3's close: 3.3
- §3 PR 4: the tuning budget and the study cache (4.1–4.3)
- §4 PR 5: the bootstrap replicates and the comparison metric (5.1–5.4)
- §5 PR 6: two notebooks (6.1–6.3)
- §6 PR 7: SHAP per cohort and the test evaluation (7.1–7.3)
- §7 Facts verified on 2026-11-04

## 0. Context and decisions

The experiment notebook (`notebooks/02_experiments.py`, marimo) has six sections: the data, the
settings, the split, the feature variants, the objective EDA (section 5, PR 3) and the tuning with
the inspection (section 6). Each tuning run keeps its best model with its bootstrap intervals (3.4,
M14). The user listed seven changes on 2026-11-04. They go in four PRs from `main` after PR 3's
merge, in this order (the user's choice): the `src` changes first (PRs 4, 5), the restructure
(PR 6) once the content is final, the new content (PR 7) in the split notebook.

**Decisions (2026-11-04, the user's; do not re-ask):**

- **D31. The budget: `n_trials = 100`, `n_startup_trials = 10`.** Reverses D20's "never
  decrease `n_trials`" (the user). The random start stays an explicit, logged setting (the test
  pins it; the tuner's field has no default) at Optuna's default of 10, 10% of the budget: D20
  measured the random start's effect within the search noise and its cost not (Model 2's run
  1.5–2× longer at 20, the random trials drawing large models). The question "can I remove it"
  is answered: only by giving `CrossValidationTuner.n_startup_trials` a default; one explicit line
  is simpler and logged.
- **D32. Model 2 stays by coordinate descent** (D29). At 100 trials a joint study of 23–33
  dimensions gets 3–4 trials per dimension; the two steps get 100 each, and the method is exact
  under the deviance. The joint tune method stays in the package (a valid path).
- **D33. The study cache: Optuna's journal storage, one file per study, loaded when complete.**
  `JournalStorage(JournalFileBackend(path))` writes a JSON-lines file and loads a real `Study`
  (trials, params, user attributes, durations, states), so the EDA's `trials_table(study, …)`
  and the run loops keep reading `study.trials` / `best_params` unchanged; no `ExperimentalWarning`
  under `-W error` (probed, Optuna 4.9.0). A parquet of the trials would lose the study object
  and force those readers to be rewritten. The file is named from the run's settings so `ls`
  answers whether a study exists; the directory is keyed by the data and split, so a changed
  population never serves a stale study. The cache lives under the project, git-ignored.
- **D34. The Poisson deviance stays the tuner's objective; RMSE is the comparison metric.**
  The user leaned to RMSE for both and asked for the decision on the EDA's evidence (3.2, 3.2b):
  on validation the RMSE-tuned models were last or near last on every metric including RMSE
  (Model 1 4.6070 against 4.5393 deviance-tuned; Model 2 4.5613 against 4.5450; one holdout,
  noise about 0.04); on cross-validation a tie within the 0.02 noise (cv RMSE 4.3663 against
  4.3722; 4.2800 against 4.3019); within its study the RMSE best is also the deviance's best
  (percentile 0.0); coordinate descent is exact under the deviance only (M10). Chosen: the
  deviance tunes, RMSE compares (the best-run rule, the forest plots, the paired difference, the
  test scores); all six metrics stay recorded.
- **D35. `scoring.evaluate` becomes `compute_metrics`.** The package's precedent is
  `PermutationImportance.compute_importances`; "calc" is an abbreviation. A rename only.
- **D36. The bootstrap evaluator returns its replicates; the statistics are computed outside.**
  `BootstrapEvaluator(n_resamples, random_state)`; `evaluate(y_true, y_pred, groups, *,
  metrics, cohorts)` returns the `n_resamples × (cohort, metric)` frame; `alpha` leaves the
  class. The notebook computes the mean, the SD and the percentile interval, draws forest plots,
  and the paired difference of two models per replicate (valid because one evaluator with one
  seed draws the same rows for every model: M14). `n_resamples` (1000 or 10,000) is decided by a
  measurement in 5.3, not by rule. The resampling unit (whole neighborhoods today; rows within
  neighborhoods, the user's pandas sketch `y_true.groupby(groups).sample(frac=1.0, replace=True)`)
  is a design question in 5.2, answered with measured interval widths; the NumPy loop stays
  (pandas' `groupby.sample` is no simpler and loses `np.ndarray` targets).
- **D37. Two notebooks:** `02_objective_eda.py` (sections 1–5) and `03_hyperparameter_tuning.py`
  (sections 1–4, 6, then PR 7's sections). How they share sections 1–4 is 6.1's design question.
- **D38. SHAP per cohort, values and plots** ("show values and plots for each cohort
  separately", to see whether the features' order differs per cohort): the explainer's values
  are already per row, column and cohort; the table and the figures stop averaging over cohorts.
- **D39. The test evaluation follows E11:** both models' best runs (by the comparison metric on
  validation) refit on train + validation, scored on the test part once, behind a button, in the
  tuning notebook; the test scores are reported, never used to choose.

## 1. How to work

The standing rules are the handoff's §4 (plan mode per sub-task; the facts re-verified with
read-only probes; a design question gets evidence and a recommendation; the steps and the commit
message in the chat; a file-by-file summary, the checks, the notebook's md5 and the commit message
at each stop; the user commits; no `Co-Authored-By`, no "Generated with Claude Code"; never
`git checkout`/`stash`/`reset`/`mv`; format new cells with `ruff format` one by one, never the
whole file; the iCloud checks before every long run). The user's style: the smallest change that
works; loops over settings; no function unless reused; names that say what a thing holds; no
private helpers; settings in the constructor, data in the call.

**The checks of every sub-task** (the handoff's §4 has the details and the script locations):
- **Check A:** the notebook test (`tests/characterization/test_experiments_notebook.py`), ruff
  (14 findings today; none new), `ruff format --diff` (the user's two hunks only), `marimo check`.
- **Check B:** `app.run()`'s defs of a script run; nothing fitted.
- **Check C:** the clicked run on a 3-trial copy (`make_copy.py`: `n_trials = 3`, Model 1's
  `share_hyperparams` False; configs and src symlinked; `PYTHONWARNINGS=error`), the 24 `cv_best`
  baselines (the handoff's §2) and every assertion since 3.1c, compared frame by frame with the
  reference pickle of the previous commit's run (`compare.py`, exact, `seconds` dropped). Before
  the first edit of a sub-task that changes `src` or a run cell, run Check C on HEAD and keep the
  pickle as the next reference. The reference for the first change after 3.4 is `check_c_3_4b.pkl`.
- **`src` changes:** the unit tests of the module, the non-slow suite (`pytest -m "not slow"`,
  about 2 min) and mypy (`--cache-dir <scratch>/mypy_cache`); two mutations caught per sub-task.
- **The scripts:** copied into the session's scratchpad from
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/65d10f12-872b-4ac3-a313-b2749802a1a3/scratchpad/checks`
  (the 2026-11-04 session's copy; the earlier copies are named in the handoff's §4); if gone,
  rewritten from the handoff's §4.
- **Settings pinned by the notebook test:** `n_trials`, `n_startup_trials`, `n_resamples`,
  `alpha` and the run-settings lists; a sub-task that changes one updates the test in the same
  commit.
- **A marimo session the user has open** holds the code it loaded: tell the user before editing
  the notebook; they rerun the changed cells or restart the kernel; they must not save over an
  edit from an open editor.

## 2. PR 3's close: 3.3 (on `feat/objective-eda`)

No code change (D34). One commit for the docs, one more if the review's findings are fixed.

### [ ] 3.3a The checks and the review
- **Done already (2026-11-04):** the non-slow suite 1387 passed, 1 skipped, 1 xfailed, 19
  deselected; mypy 28 files, no issue; notebook md5 `676212a808f6c1f87f7926a58b3bffe6`.
- **The review:** an independent subagent over `git diff main..HEAD` (18 commits, 11 files,
  +2196/−794: the notebook's sections 5–6, `cross_validation.py`'s recorded metrics, the docs),
  given the diff, the metrics plan's §2 M10–M14 and §5, and the handoff's §3, not the session's
  conclusions; correctness first, then reuse and simplification; every finding reproduced with a
  read-only probe before it is reported; no fix without the user's approval. Fixes, if any, in
  their own commit (`fix(notebook): the PR review's findings`), each with Check A–C.
- **Done when:** the findings are listed in the chat with their reproductions; the user has
  chosen what to fix.

### [ ] 3.3b The docs and the PR
- **Change:**
  - this plan copied to `docs/NOTEBOOK_CACHE_EVALUATION_PLAN.md`, listed in `docs/README.md`'s
    plans;
  - `docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md`: 3.3 ticked with its Done note (the deviance kept,
    D34's evidence, the checks' numbers, the review's findings); the status line;
  - `docs/EXPERIMENT_NOTEBOOK_HANDOFF.md`: rewritten for the session that starts PR 4 (§2 the
    state after the merge, §2b PR 3's commits and review, §4 the scripts' location above, §6 PR
    4's 4.1 and the prompt; MLflow stays deferred, E11 is PR 7);
  - `README.md:39`, `docs/README.md:24`, `docs/SPLITTING.md:267`: "section 5" → "section 6" for
    the tuning; `docs/HYPERPARAMETER_TUNING.md:8`: "rebuilt from section 5" → "from the tuning
    section (then 5, now 6)".
- **In the chat:** the PR's title and description, the squash-merge title and description, and
  after the merge `git push -d origin feat/objective-eda`, `git branch -d feat/objective-eda`
  (the user runs them).
- **Done when:** the docs name no stale section; the handoff's prompt starts 4.1.
- **Commit:** `docs(notebook): PR 3 closed: the deviance kept as the objective; the plan for the
  study cache, the bootstrap replicates, the split and the test evaluation`
- **PR title:** `feat(notebook): the objective EDA; each tuned run keeps its best model and its
  bootstrap intervals`

## 3. PR 4: the tuning budget and the study cache (`feat/study-cache`, from `main`)

### [ ] 4.1 The budget (notebook, test)
- **Change:** `notebooks/02_experiments.py` section 2: `n_trials = 100`, `n_startup_trials = 10`
  (the comment: 10% of the budget, Optuna's default); section 6's text ("`n_trials = 400`, the
  first `n_startup_trials = 20`" → the new values); `tests/characterization/test_experiments_notebook.py:124-125`
  pinned to `"100"`, `"10"`; the scratchpad's `make_copy.py` replaces `    n_trials = 100\n`
  (assert the count). D31 and D32 in NOTEBOOK_TUNING_PLAN.md §0 (one line each, pointing here).
- **Checks:** Check A (the test with the new pins); Check B; Check C against `check_c_3_4b.pkl`:
  every frame equal (a 3-trial study draws the same points at a random start of 10 or 20: D20).
- **Done when:** the pinned settings pass and Check C's frames equal 3.4's.
- **Commit:** `feat(notebook): 100 trials, 10 at random`

### [ ] 4.2 The tuner takes a study storage (`src`, tests, HYPERPARAMETER_TUNING.md)
- **Change:** `src/age_group_prediction/hyperparameter_tuning/cross_validation.py`:
  `CrossValidationTuner` gains `storage_dir: Path | None = None`; each tune method gains a
  keyword `study_name: str | None = None` passed to `_optimize`. In `_optimize`: without
  `storage_dir`, today's in-memory study; with it, `JournalStorage(JournalFileBackend(storage_dir /
  f"{study_name}.log"))`, `optuna.create_study(..., storage=, study_name=, load_if_exists=True)`;
  if the loaded study already holds `n_trials` trials, return it with 0.0 seconds (the cached
  run); else `optimize` the remaining trials (`n_trials - len(study.trials)`; a resumed study's
  later draws differ from a fresh run's, the sampler's state not being stored: say so in the
  docstring). `study_name` required when `storage_dir` is given (raise `ValueError` otherwise).
- **Design question (recommendation first):** resume a partial file (above) or ignore it and
  start fresh (delete the file)? Recommended: resume, Optuna's own semantics, one fewer branch;
  the notebook's runs never stop midway except on an error, and then a resume is what the user
  wants.
- **Tests** (`tests/unit/test_hyperparameter_tuning_cross_validation.py`): a second call with the
  same dir and name returns the stored trials and 0.0 seconds without calling the objective (a
  counting objective); a file with fewer trials resumes to `n_trials`; `storage_dir` without a
  name raises; no dir behaves as before (the existing tests).
- **Checks:** the module's tests, the suite, mypy; two mutations (the trial count compared with
  `>=` 1 off; the cached branch returning the fresh seconds).
- **Done when:** the tests pass; a probe shows two notebook-sized studies loading in under a
  second.
- **Commit:** `feat(tuning): a study is stored in a journal file and loaded when complete`

### [ ] 4.3 The notebook's runs through the cache (notebook, test, docs)
- **Change:** section 2 (or 5, where the tuner is built): `studies_dir = project_root /
  "studies" / f"{config_hash[:8]}_{split_method}_seed{seed}"` (a setting printed with the
  others; `.gitignore` gains `studies/`); the tuner cell passes `storage_dir=studies_dir`. Every
  tune call passes `study_name`: Model 1
  `f"model1_{family}_exposure_{'on' if use_exposure else 'off'}_{'shared' if share_hyperparams else 'own'}_hyperparams_{'shared' if share_feature_transformer else 'own'}_transformer_{tuner.objective_metric.name}_{n_trials}"`,
  Model 2's totals `model2_total_{family}_exposure_{on|off}_...`, its probabilities
  `model2_probability_{family}_...`, the EDA's `eda_model1_{objective}_{n_trials}` and
  `eda_model2_total_{objective}_...` / `eda_model2_probability_{objective}_...` (one f-string per
  loop, the names visible in the runs tables as a `study_name` column). Section 6's and 5's text:
  a run loads its study when its file exists; delete the file to rerun. Check C's copy uses the
  copy's own `studies` dir (the symlinked `configs` only), so the baselines are fresh runs.
- **Design question:** the name's parts above, or a shorter hash of the settings dict? Recommended:
  the readable parts (the user: "easy to check if the study exists").
- **Checks:** Check A, B; Check C twice on the same copy: the first run writes 22 files and equals
  3.4's frames; the second loads them (every `seconds` 0.0, every frame equal, under 2 min);
  the 100-trial click on the real notebook (both buttons, `PYTHONWARNINGS=error`): the seconds,
  the best runs, recorded in this plan's Done note and the handoff's §2.
- **Done when:** a second click of the notebook is seconds, not minutes.
- **Commit:** `feat(notebook): every study cached in a journal file named by its run settings`

## 4. PR 5: the bootstrap replicates and the comparison metric (`feat/bootstrap-replicates`)

### [ ] 5.1 `scoring.evaluate` → `compute_metrics` (`src`, tests, docs, notebook)
- **Change:** the rename in `scoring.py`, `inspection/bootstrap.py`, the notebook's four calls,
  `tests/unit/test_scoring.py`, `test_inspection_bootstrap.py`, `test_hyperparameter_tuning_cross_validation.py`
  (if it names it), `docs/HYPERPARAMETER_TUNING.md`, `MODEL_INSPECTION.md`, the handoff, D30's
  line (a note, not a rewrite). `grep -rn "evaluate(" src tests docs notebooks` (excluding
  `__pycache__`) finds no `scoring.evaluate` after.
- **Checks:** the suite, mypy, Check A–C (the frames equal the reference exactly).
- **Commit:** `refactor(scoring): evaluate renamed compute_metrics`

### [ ] 5.2 The evaluator returns the replicates (`src`, tests, MODEL_INSPECTION.md)
- **Change:** `BootstrapEvaluator(n_resamples, random_state=None)`; `evaluate` returns
  `pd.DataFrame(scores, columns=labels)` (one row per replicate, a `(cohort, metric)` column
  MultiIndex) after the non-finite check; the `mean/lower/upper` block and `alpha` go. The
  notebook's two loops keep the replicates in each record's `scores` (5.3 reshapes the tables).
  The docstrings and the guide's block rerun.
- **Design question, with evidence from a probe** (one fitted model, validation rows): the
  resampling unit as a constructor setting, `unit: Literal["rows", "groups"]`? Today `groups=None`
  gives rows and `groups` given gives whole groups; the user's sketch resamples rows within each
  group (stratified). Measure the three intervals' widths on the composite RMSE and deviance.
  Recommended: keep the two existing designs (the API already chooses by `groups`); add the
  stratified unit only if its width differs materially and the user wants it, as a third value
  of a `unit` setting.
- **Tests:** the frame's shape `(n_resamples, n_cohorts × n_metrics)`, its quantiles equal the
  old bounds (a fixture from the old implementation's numbers), the same seed the same rows, the
  non-finite error unchanged.
- **Checks:** the suite, mypy, Check C: each record's `scores` quantiles at 0.025/0.975 equal
  the reference's `ci95_lower/ci95_upper` exactly, the means equal.
- **Commit:** `refactor(inspection): the bootstrap evaluator returns its replicates`

### [ ] 5.3 The statistics in the notebook, and `n_resamples`
- **Change:** the two loops keep `score` (the point estimate) and the replicates per dataset;
  the tables cells compute `mean`, `sd`, `ci95_lower`, `ci95_upper` from the replicates
  (`alpha` stays a setting; `CI_BOUND_NAMES` from it) into `model1_scores_df` / `model2_scores_df`
  with the same rows as today plus `sd`; a forest plot per model (plotly): each run's comparison
  metric with its interval, per cohort and the composite, validation (3.1h's palette).
- **The `n_resamples` measurement (a read-only probe before the edit):** one model (the best
  `lgbm_poisson` of a 3-trial run), the validation rows, five seeds at 1000 and at 10,000: the
  spread (SD over seeds) of each bound against the interval's width, for RMSE and the deviance;
  the seconds per call. Recommended rule: keep 1000 if the bounds' spread is under 5% of the
  width; else 10,000 for the comparison section only (the best models) and 1000 for the 24 runs.
  The user decides on the numbers; the test's pin follows.
- **Checks:** Check A (the pin), B, C (the tables' `mean`, `ci95_*` equal the reference exactly;
  the forest figures' traces one per run, the x the metric's values).
- **Commit:** `feat(notebook): each run's scores with their bootstrap SD and interval; a forest plot per model`

### [ ] 5.4 RMSE compares: the best-run rule and the paired difference
- **Change:** the metrics cell gains `COMPARISON_METRIC = RMSE` (the comment: compares models,
  in children per building; the deviance tunes); the inspection's best-run rule and the
  permutation importance's score read it (today `POISSON_DEVIANCE` at the notebook's lines
  2234, 2240, 2277, 2320); the forest plots of 5.3 read it; a new cell after the inspection's
  choice: the two best models' paired difference per replicate on validation (`model2 − model1`
  per cohort and the composite, the same replicate rows), its mean and interval, a forest plot of
  the difference with a rule at 0 ("0 inside the interval: not significant"); section 6's text.
- **Checks:** Check A–C (the best runs by RMSE on validation named; the difference's rows equal
  `scores2 − scores1` per replicate exactly; the inspection's `importance_df` recomputed with
  RMSE, a new reference).
- **Commit:** `feat(notebook): RMSE compares the models; the best models' paired bootstrap difference`

## 5. PR 6: two notebooks (`refactor/two-notebooks`)

### [ ] 6.1 The shared setup: the design
- **Question, probed with evidence before the edit:** sections 1–4 (lines 1–626: the data,
  the settings, the split, the 260-line feature declarations) in both notebooks as (a) a
  duplicate, (b) a plain module beside the notebooks (`notebooks/experiment_setup.py`:
  `simulate_population(config)`, `split_parts(df, …)`, `feature_transformer_candidates()`,
  taking the settings as arguments, the settings cells staying in each notebook), or (c)
  marimo's reusable cells (`@app.function` / `@app.class_definition` imported from one notebook;
  probe that marimo 0.24 imports them from a file named with a leading digit via
  `importlib`, and what `marimo check` says). Measured: the lines duplicated, the names each
  notebook reads, whether the notebook test's script run still prints the settings lines.
  Recommended before the probe: (b), the package rule's spirit (mechanism out of the notebook,
  the settings in it), one source of truth for the declarations, no marimo-version coupling.
- **Done when:** the user has chosen; the chosen module or cells exist with the same outputs
  (the printed `data`, `experiment_settings`, `split` lines equal).
- **Commit:** `refactor(notebook): the experiment's setup shared by the notebooks` (if b or c)

### [ ] 6.2 `02_objective_eda.py`
- **Change:** sections 1–5 (and the shared setup); the EDA's button, runs, tables and figures;
  the notebook test split: `test_objective_eda_notebook.py` (the printed settings, the tuning
  cells behind `mo.stop`). `README.md`, `docs/README.md`, the handoff name the two notebooks.
- **Checks:** Check A per notebook; Check B; Check C's EDA assertions on the new file equal the
  reference's EDA frames exactly.
- **Commit:** `refactor(notebook): the objective EDA in its own notebook`

### [ ] 6.3 `03_hyperparameter_tuning.py`
- **Change:** sections 1–4 (shared), 6 (tuning, inspection, the comparison of 5.4); `02_experiments.py`
  removed; the characterization test renamed/split; `SPLITTING.md`, `HYPERPARAMETER_TUNING.md`,
  `MODEL_INSPECTION.md` references.
- **Checks:** Check A per notebook; Check C's section-6 assertions on the new file equal the
  reference's frames exactly; the 100-trial click loads the cached studies (the cache's
  directory is keyed by the data, not the notebook's name).
- **Commit:** `refactor(notebook): the tuning in its own notebook; 02_experiments.py removed`

## 6. PR 7: SHAP per cohort and the test evaluation (`feat/test-evaluation`)

### [ ] 7.1 SHAP per cohort (notebook)
- **Change:** the importance cell: `mean_abs_shap` per column × cohort (`np.abs(values).mean(axis=0)`,
  a frame columns × cohorts, the composite beside as today) and the rank per cohort; a
  beeswarm-style plotly figure per model and cohort (the SHAP value per explained row and
  column, the color the column's value, the columns ordered by the cohort's mean |SHAP|), shown
  in a dropdown of the cohorts or stacked; section 6's text.
- **Checks:** Check A–C (the per-cohort means average to the reference's `mean_abs_shap`
  exactly; the figures' traces one per column).
- **Commit:** `feat(notebook): SHAP per cohort, the values and the feature order`

### [ ] 7.2 The final refit and the test scores (E11)
- **Change:** a new section 7 "The test evaluation": a button "Evaluate on the test part"; a cell
  that refits each model's best run (by the comparison metric on validation; the stored
  components, `build_*_model`, `random_state=seed`) on `train + val` (`pd.concat` of the two
  parts' `AgeGroupDataset`s; `test_age_group` returned by the split cell), predicts the test
  part once, and keeps `test_predictions` and `test_scores` (every metric per cohort and the
  composite, with the bootstrap's replicates over the test rows' neighborhoods) per model; the
  two models' paired difference on test; the tables and forest plots as in section 6. The
  section's text: the test part stands for production, scored once, never used to choose
  (E11); the notebook test checks the cell is behind `mo.stop(not `.
- **Checks:** Check A (the new guard), B, C (on the copy: the test scores' rows 2 models × 4
  cohorts × 6 metrics; the refit's training rows 1223; the test predictions' length 306).
- **Commit:** `feat(notebook): the best models refit on train and validation, scored once on the test part`

### [ ] 7.3 The reading of the test predictions
- **Change:** cells after 7.2, per model: (a) the residual table, a row per test building
  (`building_id`, `neighborhood_id`, the features, the true and predicted counts per cohort,
  the composite's absolute error), and the best K and worst K (`K = 10`, a setting) with what
  they share: their neighborhoods' counts among the K, their exposure and share quantiles
  (an anomaly: a value outside the training rows' 1st–99th percentile), one table each; (b)
  the count "confusion matrix": per cohort, the true count × the rounded predicted count as a
  heatmap (plotly), the diagonal the exact hits; (c) the total true against predicted (a scatter
  with the identity line, colored by exposure); (d) the cohort shares true against predicted
  (per cohort, the true share `y / total` against the predicted share: Model 2's classifier's
  probabilities, Model 1's `pred / pred.sum`), a scatter per cohort with the identity line and
  a calibration curve by deciles.
- **Design questions, in the sub-task's plan:** the error that ranks the buildings (the
  composite's absolute error, recommended: readable; or the Poisson deviance per row); which
  hover text and which figure per cell.
- **Checks:** Check A–C (the residual table's rows 306, its errors equal `|y − ŷ|.sum(axis=1)`;
  each heatmap's cells sum to 306; the share scatter's x and y from the predictions).
- **Commit:** `feat(notebook): the test predictions read: best and worst buildings, the count heatmaps, the totals and the shares`

## 7. Facts verified on 2026-11-04 (read-only probes)

- Optuna 4.9.0 `JournalStorage(JournalFileBackend(path))`: one JSON-lines file per study;
  `create_study(..., storage, study_name, load_if_exists=True)` and `load_study` return a real
  `Study`; `get_all_study_names`; no warning under `-W error`. sqlalchemy 2.0.52 and pyarrow
  25.0.1 are installed (not needed).
- `CrossValidationTuner._optimize` (cross_validation.py:315–331) is the only `create_study`.
- D20: a 3-trial study draws the same points at a random start of 10 or 20, so 4.1 keeps the 24
  baselines.
- `tests/characterization/test_experiments_notebook.py:118-131` pins the experiment settings;
  `:189-200` requires `mo.stop(not ` in every cell holding `tuner.tune_`.
- `BootstrapEvaluator.evaluate` (bootstrap.py:45–127): cluster bootstrap over `groups` or rows;
  `scores` is `n_resamples × (cohort, metric)` before the quantiles.
- `ShapExplainer.explain` returns values per row, column and cohort (`output_names`); the
  notebook's `importance_df` averages over rows and cohorts (line 2368).
- `PermutationImportance.compute_importances`: the naming precedent for `compute_metrics`.
- `scoring.evaluate`'s callers: scoring.py, inspection/bootstrap.py, the notebook (4 calls),
  tests (test_scoring, test_inspection_bootstrap, test_hyperparameter_tuning_cross_validation),
  the guides (HYPERPARAMETER_TUNING.md, MODEL_INSPECTION.md), the handoff, the metrics plan.
- `POISSON_DEVIANCE`'s readers in the notebook: the tuner cell (794), the best-run rule (2234,
  2240), the permutation importance (2277, 2320).
- E11 (EXPERIMENT_NOTEBOOK_PLAN.md:423–433): the final refit on train + validation, scored once
  on test; the winner rebuilt from its components.
- `notebooks/` holds `01_eda.py` and `02_experiments.py`; sections 1–4 are lines 1–626; the
  split cell returns `train_age_group`, `val_age_group` (not `test_age_group`); `project_root`
  and `config_hash` exist in section 1.
- The 400-trial click: Model 1 821 s, Model 2 752 s (joint 2871 s), the bootstrap about 7 s per
  call (24 models × 2 datasets); a click about 31 min.
- `.gitignore` has no `studies/` entry.
- The non-slow suite at `b7da88d`: 1387 passed, 1 skipped, 1 xfailed, 19 deselected; mypy 28
  files, no issue.
