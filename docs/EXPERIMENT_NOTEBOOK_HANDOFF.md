# Handoff: the experiment notebook, PR 5b (Model 2's two tuning methods in the EDA): 5b.2 done, 5b.3 next

**Status (2026-11-07, 5b.2 done, before the user's commit):**
- **5b.1 is committed** (`82ccac2`). **5b.2 is implemented as planned** (the cache plan's 5b.2
  Done note): the joint Model 2 method in the objective EDA (`eda_model2_joint_df`,
  `eda_model2_joint_trials_df`, `model2_joint` in the summary, the figures and the tables);
  section 5's text. The notebook's md5 is `6731d509d2b5298e4d38a99480f700dc`. Next: 5b.3, the
  reading at 100 trials (the EDA click on the real cache: 3 joint studies tuned, the rest
  loaded).
- **Check C's reference** is `check_c_5b_2.pkl` with its two figures' JSON (`run_5b_2.sh`), in
  the scratchpad named below.

**Status (2026-11-07, 5b.1 done, before the user's commit):**
- **5b.1 is implemented as planned** (the cache plan's 5b.1 Done note): `trials_table` renamed
  `build_trials_table` (7 notebook lines; §1 and §5 here). The notebook's md5 is
  `e0f2ba6c32ead22ebee081a4f0a27e3f`. Next: 5b.2 (the cache plan's §4b).
- **This session's scratchpad** (copy from it first): `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/f38b920e-c1b1-4cff-a21c-fa1038947061/scratchpad`:
  `checks/` (`run_5b_1.sh`: Check A, B, C and the comparison; `check_b.py` with 5b.1's names),
  `check_c_5b_1.pkl` (5b.2's reference) with its two figures' JSON, `check_c_5_3.pkl`,
  `check_c_5_2b.pkl`, `check_c_3_1l.pkl`, `check_c_3_4_first.pkl`. A new session's
  `PYTHONPYCACHEPREFIX` is compiled once without `-W error` first (§4's iCloud notes).

**Status (2026-11-07, PR 5 merged; PR 5b planned; the next session starts at §6):**
- **PR 5 is merged** as #23, squashed into `main` as `c3bdee8` (identical to `8a90bf9`, 5.3);
  `feat/bootstrap-replicates` deleted locally and on GitHub. At the close: the non-slow suite
  1393 passed, 1 skipped, 1 xfailed, 19 deselected; mypy 28 files, no issue.
- **PR 5b is on `feat/model2-tuning-methods`** (from `main` at `c3bdee8`): its plan is the cache
  plan's §4b (5b.1 the rename, 5b.2 the joint method in the EDA, 5b.3 the reading); the order
  after it is §6's.
- **The scripts and pickles to copy first** (it may vanish on a restart; then rewrite them from
  §4): `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/095bc943-e3e7-4192-9f55-ee2e42ef19ba/scratchpad`:
  `checks/` (`run_5_3.sh`: Check A, B, C and the comparison; `check_c.py` takes `<copy>
  <out.pkl> check_c_3_1l.pkl check_c_3_4_first.pkl check_c_5_2b.pkl`), `check_c_5_3.pkl` (the
  reference) with its two figures' JSON, `check_c_5_2b.pkl`, `check_c_3_1l.pkl`,
  `check_c_3_4_first.pkl`; `probes/view_5_3.py` and `viewer/forest_5_3.py` (5.3's forest plot on
  the real cache).

**Status (2026-11-07, 5.3 done, before the user's commit):**
- **5.3 is implemented as planned** (the cache plan's 5.3 Done note): `selection_metric = "rmse"`
  (section 2, pinned) → `SELECTION_METRIC`; the runs keep point scores; a cell picks each
  model's best run by the validation composite and bootstraps the two (`best_runs_val_scores_df`,
  `best_replicate_scores`, `best_model1`, `best_model2`); `forest_figure` and the forest plot.
  The notebook's md5 is `9a66a347ed3fc928f14a43bde04da1b9`.
- **This session's scratchpad** (copy from it first):
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/095bc943-e3e7-4192-9f55-ee2e42ef19ba/scratchpad`:
  `checks/` (5.3's: `check_b.py`, `check_c.py` with argv[5] the 5.2b pickle, `compare.py` on the
  shared columns, `run_5_3.sh`, `mutate_5_3.py`, `run_mutants_5_3.sh`), `new_cells/`, the
  pickles `check_c_5_3.pkl` (5.4's reference) with its figures' JSON, `check_c_5_2b.pkl`,
  `check_c_3_1l.pkl`, `check_c_3_4_first.pkl`, `probe_5_2.pkl`.

**Status (2026-11-07, 5.3 planned, not implemented):**
- **5.3 is planned** in the cache plan's §4 (its 5.3 block, "Planned 2026-11-07"):
  - the `n_resamples` measurement is done, and 1000 is kept;
  - the user changed the plan: each model's best run is chosen by one configurable selection
    metric (RMSE by default) on its validation point score; only the two best are bootstrapped;
    one `forest_figure` function draws them;
  - the plan awaits approval in the next session; nothing in the repo changed but the two
    docs; 5.4 is re-scoped.
- **The 2026-11-07 session's scratchpad** (copy from it first):
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/6c4b7827-eaf3-4b57-8720-9133a2e32b80/scratchpad`:
  `checks/`, `probes/` (with `probe_5_3.py`), the pickles `check_c_5_2b.pkl` (5.3's Check C
  reference), `check_c_3_1l.pkl`, `check_c_3_4_first.pkl`, `probe_5_2.pkl`, and
  `probe_5_3.log`, `probe_5_3_bounds.csv`.

**Status (2026-11-06, 5.1 and 5.2 committed):**
- **PR 5 is on `feat/bootstrap-replicates`** (from `main` at `30bd4f7`, PR 4's merge #22):
  `466534e` this handoff after the merge, `571f721` 5.1 (`scoring.evaluate` →
  `compute_metrics`), `4cfe502` 5.2 (the bootstrap evaluator returns its replicates; each
  score's `mean`, `median`, `sd` and interval computed in the notebook). §2's "PR 5" block has
  the details; the cache plan's 5.1 and 5.2 Done notes the full record.
- **Next: 5.3** (the forest plots and the `n_resamples` measurement), then 5.4 (RMSE compares;
  the paired difference), then closing PR 5.
- The notebook's md5 is `d4a361a5535aaefd89b9dccd4f99e282`. The user's real study cache is
  `studies/` in the repo (git-ignored; 25 studies at 100 trials): a click loads it in about 7 min.
- **This session's scratchpad** (copy from it first; it may vanish on a restart):
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/6399f8d3-96a2-49b3-a707-91be30c7b8b4/scratchpad`: `checks/` (the scripts, §4), `check_c_5_2b.pkl` (5.3's Check C reference) and
  `probe_5_2.pkl` (the 100-trial runs' fitted models, for 5.3's `n_resamples` probe).


**For:** the session that continues PR 5 of
[NOTEBOOK_CACHE_EVALUATION_PLAN.md](NOTEBOOK_CACHE_EVALUATION_PLAN.md), the bootstrap
replicates and the comparison metric (5.1–5.4), at 5.3. That plan holds the user's additional
changes, planned 2026-11-04 as PRs 4–7: the tuning budget and the study cache, the bootstrap
replicates and RMSE as the comparison metric, the notebook split, SHAP per cohort and the test
evaluation; decisions D31–D39.

PR 3 of [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md) is closed (3.3,
2026-11-05):
- its EDA is built and read (3.1a–3.1l, 3.2, 3.2b);
- section 6 keeps each run's best model and its bootstrap intervals (3.4);
- the review's F2 is fixed; F1 and F5 are fixed in the docs.

The metrics plan's three PRs:
- PR 1, the metrics in `scoring`: merged as #18 into `feat/experiment-notebook`, which #19
  merged into `main`;
- PR 2, the runs in loops scored on every metric: merged as #20, squashed into `main` as
  `16773a5`;
- PR 3, the objective EDA: merged as #21, squashed into `main` as `0c94f16`.

The cache plan's PR 4, the study cache: merged as #22, squashed into `main` as `30bd4f7`.
PR 5 is on **`feat/bootstrap-replicates`**, created from `main` at `30bd4f7`.

MLflow stays deferred; E11's final evaluation is the cache plan's PR 7.

The next session starts at §6.

**Written:** 2026-10-29 after the tuning plan's T5; **rewritten** 2026-10-31 at the end of PR 2
(2.1–2.5); **updated** 2026-10-31 after PR 2's merge (#20), for a new session; **updated**
2026-11-01 in PR 3 (§2b, §6), for a new session; **updated** 2026-11-01 after 3.1h–3.1j (§1, §2b,
§3, §4, §6), for a new session; **updated** 2026-11-02 after 3.1k, 3.1l and 3.2b (§1, §2b, §4,
§5, §6), for a new session on section 6; **updated** 2026-11-04 after 3.4 (§2b, §4, §5, §6), for
a new session on the user's additional changes; **updated** 2026-11-04 again after those changes
were planned (§1's "For", §2b, §4, §6; the plan is NOTEBOOK_CACHE_EVALUATION_PLAN.md), for the
session that does 3.3 and starts PR 4; **updated** 2026-11-05 at PR 3's close (3.3: the header,
§2, §2b, §4, §5, §6), for the session that starts PR 4; **updated** 2026-11-05 after PR 3's
merge (#21, `0c94f16`: the header, §2, §2b, §6), as `feat/study-cache`'s first commit; **updated** 2026-11-06 after 5.2 (`4cfe502`: the header, §2's
"PR 5", §3, §4, §6), for the session that continues PR 5 at 5.3; **updated** 2026-11-07 after
5.3's planning (the header, §2's "PR 5", §3, §4, §6), for the session that implements it;
**updated** 2026-11-07 after 5.3 (the header, §1, §2's "PR 5", §4, §5, §6), for PR 5's close and PR 5b; **updated**
2026-11-07 after PR 5's merge (#23), as `feat/model2-tuning-methods`' first commit. The detailed history is in the metrics
plan (§2 M1–M14, §4 2.1–2.5, §5 3.1–3.4 and 3.3 with each
sub-task's "Done" note), in [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) (§0 D1–D30) and in
git. Re-verify every fact here against the code of the day.

## 1. What exists

**The notebook** (`notebooks/02_experiments.py`, marimo; a script run fits nothing):
1. **Data:** the simulator at 150 neighborhoods × 9 buildings (`configs/simulation.yml`, its
   seed 42 unless `simulation_seed` overrides it), the room shares, `X`, `Y` (three cohorts),
   the exposure (`n_apartments`), the groups (neighborhoods); the settings printed and hashed.
2. **Settings:** `seed = 42`, `split_method = "random"`, `test_size = val_size = 0.2`,
   `n_splits = 5`, `n_trials = 100`, `n_startup_trials = 10` (D31, PR 4),
   `selection_metric = "rmse"` (5.3; or `"poisson_deviance"`), `n_resamples = 1000`,
   `n_repeats = 10`, `n_shap_rows = 100`, `n_shap_background = 50`. The sharing flags are run
   settings now (below), not experiment settings.
3. **Split:** train 917 / validation 306 / test 306 rows (`AgeGroupDataset` per part), and
   `cv` / `cv_groups` from the same `Splitter`.
4. **Feature variants:** the declarations of FEATURE_TRANSFORMATIONS.md §8, and each family's
   candidates as data: `COUNT_FEATURE_TRANSFORMERS` and `PROBABILITY_FEATURE_TRANSFORMERS`
   (family → `{name: FeatureTransformer}`).
5. **Section 5, the objective EDA** (PR 3; the metrics plan's 3.1, M10–M13): its text; the
   tuning imports, the metrics cell (also `RMSE` and `EDA_OBJECTIVES = (LOG_LOSS,
   POISSON_DEVIANCE, RMSE)`), `COHORTS` / `PREDICTION_FLOOR`, the `tuner` and
   `MODEL2_FIXED_PROBABILITY_COMPONENT` (moved up from section 6, so every name is defined above
   where it is read); `build_trials_table(study, objective, metric_names, components_by_trial,
   feature_transformer_dict)` (`trials_table` until 5b.1); the button "Run the objective EDA"; Model 1's EDA run
   (`lgbm_poisson`, exposure on, the notebook's flags) and Model 2's (`lgbm_poisson+logistic`,
   exposure on, by coordinate descent), each one study (Model 2: two) per objective through
   `dataclasses.replace(tuner, objective_metric=objective)`, every study seeded `seed`; each best
   refit and scored on validation (`eda_model1_df`, `eda_model2_df`, 12 rows each; 5b.2: Model 2's joint method too,
   `eda_model2_joint_df`, one `tune_total_times_probability_model` study per objective stored as
   `total_times_probability_joint__…`); every trial
   kept as a row of the whole model it scored (`eda_model1_trials_df` 3 × `n_trials` rows,
   `eda_model2_trials_df` 6 × `n_trials`: `id, objective, seconds, state`, the six metrics,
   `feature_transformers` and `hyperparameters` per component; Model 2's with `step`, `total` rows
   holding the fixed classifier, `probability` rows step 1's best total, `id` each study's own
   number; `eda_model2_joint_trials_df` 3 × `n_trials`, no `step`); `eda_best_df` (9 rows: model
   (`model1`, `model2`, `model2_joint`) × objective, `best_id`, `cv_<metric>`, `val_<metric>`);
   `eda_figures` (a plotly figure per model and objective, six stacked panels each; the joint's
   like Model 1's, no step rule); `eda_objective`, an "Objective" dropdown (the tuner's objective
   by default) choosing which objective's three figures are shown, one per model, full width, stacked (3.1k);
   `eda_recorded_metric`, a "Recorded metric" dropdown, and `eda_metric_figures[model, metric]`
   (a box per objective's study with every trial and its best as a star, Model 2's steps grouped,
   the y-axis over the bulk; 3.1l; three per metric since 5b.2); a table cell with the three
   summaries and the three trials tables.
6. **Section 6, tuning and inspection:**
   - **the metrics cell:** `POISSON_DEVIANCE = Metric("poisson_deviance",
     mean_poisson_deviance_loss)` (the tuner's objective, the best-run rule, the permutation
     importance's score), `LOG_LOSS`, and `METRICS`, the six (`poisson_deviance`, `log_loss`,
     `mae`, `rmse`, `wmape`, `bias`; the primary first) for the score tables and the bootstrap;
     `COHORTS`, `PREDICTION_FLOOR`;
   - **the `tuner` cell** (in section 5): `CrossValidationTuner(cv=cv,
     objective_metric=POISSON_DEVIANCE, n_trials=n_trials, n_startup_trials=n_startup_trials,
     n_jobs=n_splits, recorded_metrics=METRICS, storage_dir=studies_dir)`: every trial also
     records its cross-validated score on the six metrics (M12); every study is stored in
     `studies/<data, split, folds, seed, budget>/<step>__<run settings>__objective=…__seed=….log`
     and loaded by a later click (D33, PR 4: `build_study_name` in the same cell; a study run in
     sections 5 and 6 is stored once; delete a file to tune its study again);
   - **Model 1:** `MODEL1_RUN_SETTINGS` (8 entries: 4 count families × exposure on / off, each
     with `share_hyperparams=True`, `share_feature_transformer=False`), a button (marimo's
     clickable control; the cells behind it run only when it is clicked), one loop cell
     (`tuner.tune_independent_cohorts_model`, seeded `seed + i` in the list's order) building
     `MODEL1_RUNS`;
   - **Model 2, by coordinate descent** (D29): a settings cell (`MODEL2_TOTAL_RUN_SETTINGS`, 8
     entries in Model 1's order; `MODEL2_PROBABILITY_RUN_SETTINGS`, logistic then lgbm;
     `MODEL2_FIXED_PROBABILITY_COMPONENT`, logistic `C = 1.0`, no calibration, `glm_base`), a
     button, and three cells:
     - step 1 (`tuner.tune_total_component`, the fixed classifier, `seed + 0 … 7`) →
       `MODEL2_TOTAL_RUNS`;
     - step 2 (step 1's best total by cv read back and fixed; `tuner.tune_probability_component`,
       `seed + 8`, `seed + 9`) → `MODEL2_PROBABILITY_RUNS`;
     - the 16 pairs (each total run × each probability run, read back from their
       `best_params`; `tuner.cross_validate(...)[tuner.objective_metric.name]` as `cv_best`;
       refit and scored; no search) → `MODEL2_RUNS`;
   - **the bootstrap evaluator cell** (3.4, M14): `bootstrap_evaluator =
     BootstrapEvaluator(n_resamples, random_state=seed)` (5.2: no `alpha`; `alpha = 0.05` in
     section 2) and `CI_BOUND_NAMES` (`lower` → `ci95_lower`, `upper` → `ci95_upper`); since
     5.3 only the best-runs cell reads them;
   - **the refit and the scores** (Model 1's loop and Model 2's pairs): the best model fitted
     on the training part; per dataset (train, val) predicted once, `_dataset_scores =
     compute_metrics(...).stack()`, `scores` per dataset, cohort and metric with the one column
     `score` (5.3: no bootstrap in the loops);
   - **the tables**, two per model, shown with `mo.ui.table` (marimo's interactive table: sort,
     filter, pages): the runs (`model1_runs_df` 8 rows, `model2_runs_df` 16: the settings,
     `feature_transformers`, `hyperparameters` per component, `cv_poisson_deviance`,
     `n_complete`, `n_failed`, `seconds`; a pair's counts and seconds its two studies' summed)
     and their scores (`model1_scores_df` 384 × 8, `model2_scores_df` 768 × 6: a row per run ×
     dataset × cohort × metric, `score`);
   - **the best runs** (5.3; no button: it runs once both models' runs exist): each model's
     best run by the validation composite on `SELECTION_METRIC` (`selection_metric`, RMSE by
     default) → `best_model1`, `best_model2`; the two bootstrapped on validation (2 calls, one
     evaluator: the same replicate rows) → `best_replicate_scores` (`{"independent_cohorts":
     …, "total_times_probability": …}`, 1000 rows × (cohort, metric)) and
     `best_runs_val_scores_df` (48 rows: `model` (`independent_cohorts` /
     `total_times_probability`), `family`, `use_exposure` (bool), `cohort, metric, score, mean,
     median, sd, ci95_lower, ci95_upper`; the user's names at the 5.3 review); a sentence naming
     the two and the table;
   - **`forest_figure(scores, metric_name, lower_column, upper_column, rule_at=None)`** (5.3):
     a panel per cohort, a row per model, the dot `score` and the bar its interval
     (`error_x` `array` / `arrayminus`), `rule_at` a vertical line per panel; the forest plot of
     the two best on `SELECTION_METRIC` (`best_forest_figure`);
   - **behind "Inspect the best runs":** the permutation importance (still the deviance until
     5.4) and exact SHAP of `best_model1`, `best_model2` (`importance_df`).

**The run records** (the input of the MLflow runs; the best model only since 3.4, M14):
- **Model 1** (`MODEL1_RUNS`): the run settings (`family`, `use_exposure`, the two sharing
  flags), `feature_transformers` ({cohort: candidate name}), `model` (fitted on training),
  `components` (the objects; `to_dict()` writes each transformer's declaration), `cv_best`,
  `n_complete`, `n_failed`, `seconds` (the study's), `scores` (dataset × cohort × metric:
  `score` only since 5.3; the two best runs' statistics are `best_runs_val_scores_df`'s).
- **Model 2's step runs** (`MODEL2_TOTAL_RUNS`, `MODEL2_PROBABILITY_RUNS`): their settings,
  `feature_transformers`, `component` (the best, the object step 2 and the pairs take),
  `cv_best`, `n_complete`, `n_failed`, `seconds`.
- **Model 2's pairs** (`MODEL2_RUNS`): `total_family`, `probability_family`, `use_exposure`,
  `feature_transformers` (the two steps'), `model`, `components`, `cv_best` (the pair's
  cross-validated objective), `n_complete`, `n_failed`, `seconds` (its two studies' summed),
  `scores`.
- No `trials`, `best_params`, `seed` or `y_min`: section 5 studies the trials; the model holds
  the rest.

**The packages it uses**, each with its guide:
- [`hyperparameter_tuning`](HYPERPARAMETER_TUNING.md): records, search spaces, builders (the
  joint suggest functions and `suggest_count_component` / `suggest_probability_component`), the
  tuner (a tune method per model, and `tune_total_component` / `tune_probability_component`;
  `storage_dir` and each call's `study_name` store a study in Optuna's journal file and load it
  when complete, PR 4), aggregation. No `fit_and_score` (D30).
- `scoring`: the six metric functions, `Metric`, and `compute_metrics(y_true, y_pred,
  metrics, cohorts="composite")` (D35, 5.1; `evaluate` until then).
- [`inspection`](MODEL_INSPECTION.md): `BootstrapEvaluator.evaluate(y_true, y_pred, groups=None,
  *, metrics, cohorts="composite")` (the caller predicts; since 5.2 the replicates, one row
  each, the caller computing the statistics),
  `PermutationImportance`, `ShapExplainer`.
- `modeling` ([Model 1](INDEPENDENT_COHORT_MODELS.md), [Model 2](TOTAL_TIMES_PROBABILITY_MODEL.md)):
  fit and predict; no `evaluate` (D30).
- [`splitting`](SPLITTING.md), `feature_engineering` ([FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md)).
  MLflow is not used yet; the old `tracking/` package belongs to the old stack.

## 2. State

### PR 5 (in progress, `feat/bootstrap-replicates`)

- **Commits** (from `main` at `30bd4f7`):

  | commit | sub-task |
  |---|---|
  | `466534e` | the handoff after PR #22's merge (the branch's first commit) |
  | `571f721` | 5.1 `scoring.evaluate` renamed `compute_metrics` (D35) |
  | `4cfe502` | 5.2 the bootstrap evaluator returns its replicates (D36); `mean`, `median`, `sd`, interval in the notebook |

- **5.1 (a rename only):** `scoring.py`, `bootstrap.py`'s call and docstrings, `test_scoring.py`
  (its import, calls and two test names), the notebook's import cell, four cells' inputs and
  four calls, the current guides; D30 got a note. Checks: the suite 1392 passed; Check C equal
  to 4.3's run exactly.
- **5.2:** `BootstrapEvaluator(n_resamples, random_state=None)` without `alpha`; `evaluate`
  returns one row per replicate, a `(cohort, metric)` column MultiIndex; the draws unchanged
  (a test pins the old implementation's mean and bounds). The notebook's two loops (Model 1's
  runs, Model 2's pairs): `_replicate_scores = bootstrap_evaluator.evaluate(...)`, then one
  frame per dataset, `score, mean, median, sd, ci95_lower, ci95_upper` (`sd` pandas' `std()`,
  `ddof=1`; the bounds `quantile(alpha / 2)`, `quantile(1 - alpha / 2)`); the records' `scores`
  keep those statistics, not the replicates (the tables and the best-run rule read them).
  The scores tables: 384 × 13, 768 × 11.
- **5.2's resampling-unit probe** (the 100-trial best models, validation, 1000 replicates,
  seed 42; 306 rows in 129 neighborhoods, median 2 rows, 37 with one row): the composite's 95%
  width, rows / whole neighborhoods / rows within neighborhoods: Model 1 deviance 0.507 / 0.542
  / 0.380, RMSE 1.006 / 1.095 / 0.582; Model 2 deviance 0.488 / 0.493 / 0.360, RMSE 0.885 /
  1.024 / 0.582. The user kept the two designs (rows, whole groups). One bootstrap call (6
  metrics × 4 cohorts, 1000 replicates) takes 6.2 s.
- **5.2's checks:** `test_inspection_bootstrap.py` 24 under `-W error`; the non-slow suite 1393
  passed, 1 skipped, 1 xfailed, 19 deselected; mypy 28 files; two mutations caught;
  MODEL_INSPECTION.md's blocks rerun, output unchanged; Check A (the notebook test 14, ruff 15,
  the two hunks, `marimo check`), B; Check C (414 s) equal to 5.1's run, `mean` at rtol 1e-12
  (pandas sums in another order: at most 1.4e-14 relative), `median` and `sd` checked against
  the replicates.
- **5.3's planning (2026-11-07; HEAD `6013339`, the notebook's md5 unchanged):**
  - **The `n_resamples` probe** (`probes/probe_5_3.py` on `probe_5_2.pkl`, read-only):
    validation, whole neighborhoods, seeds 42–46. The bounds' SD over seeds as % of the
    interval's width: at most 4.6% at 1000, 1.5% at 10,000 (the best models and their paired
    difference, the deviance and RMSE, per cohort and the composite). One call takes 6.9 s at
    1000 and 69.1 s at 10,000. **The user kept 1000.**
  - **The best runs:** Model 1's is the same by either metric (`lgbm_poisson`, exposure on).
    Model 2's is `lgbm_gaussian+logistic`, exposure on, by deviance and `lgbm_poisson+logistic`,
    exposure off, by RMSE (RMSE 4.5393 against 4.5388, within noise). `score` lies inside its
    interval in all 1152 rows. The composite paired difference contains 0 under both rules.
  - **The user's answers:**
    - one selection metric, a notebook setting (RMSE by default, the deviance the
      alternative), on the validation point score;
    - only the two best models bootstrapped, on every metric; the other runs keep point scores;
    - one `forest_figure` function, its metric a parameter (Model 1's and Model 2's tables
      share a shape: a second function only if they differed);
    - n_resamples 1000.
  - **The user's question, answered in the chat and recorded in the cache plan's 5.3:**
    bootstrap or point estimate to choose? The point estimate: the bootstrap's mean ranks the
    same; the bootstrap measures uncertainty; cv is no fairer, since each run was tuned on it;
    the chosen score is optimistic, and the test part gives the fair one.
  - **The plan** is the cache plan's 5.3 block; the user stopped before approving it, to
    implement in a new session.
- **5.3, done (2026-11-07; uncommitted at the stop, HEAD `1af184a`):** as planned (the cache
  plan's 5.3 Done note). Files: `notebooks/02_experiments.py` (the settings, the metrics cell,
  the evaluator's comment, the two loops, the two captions, three new cells, the inspection's
  selection cell removed, section 6's text); `tests/characterization/test_experiments_notebook.py`
  (the pin); the cache plan; this handoff. The checks are in the Done note.

### PR 4 (2026-11-06, closed: merged as #22, `30bd4f7`)

- **Commits on `feat/study-cache`** (from `main` at `0c94f16`; after the squash-merge GitHub
  keeps them under #22, and the docs cite them by hash):

  | commit | sub-task |
  |---|---|
  | `9662010` | the handoff after PR #21's merge (the branch's first commit) |
  | `a16c218` | 4.1 the budget: `n_trials = 100`, `n_startup_trials = 10` (D31, D32) |
  | `3806c70` | 4.2 `CrossValidationTuner.storage_dir` and each tune method's `study_name` |
  | `6e5aa3a` | the fix: a stored study is loaded without writing to its file (`load_study`) |
  | `383ee53` | 4.3 every study cached in a journal file named by its run settings |

- **The merge:** #22, squashed into `main` as `30bd4f7`, identical to `383ee53`;
  `feat/study-cache` deleted locally and on GitHub. A squash-merged branch is deleted with
  `git branch -D` (`-d` refuses).

- **4.1's checks:** Check A, B; Check C (410 s) equal to `check_c_3_3_fix.pkl` (`check_c_4_1.pkl`).
- **4.2** (the user's answers, the cache plan's 4.2):
  - a partial study file is deleted and rerun from its seed (a resumed study is not the seeded
    run);
  - `seconds` is the study's run, stored in the study and returned on load;
  - a loaded study skips `optimize` by an explicit branch;
  - no save switch: the cache is always used.

  Its checks: the module 21 passed under `-W error`; the non-slow suite 1391 passed, 1 skipped,
  1 xfailed, 19 deselected; mypy 28 files; three mutations caught; two 100-trial studies load
  in 0.02–0.03 s; Check C (460 s) equal to 4.1's (`check_c_4_2b.pkl`).
- **4.3, done** (the cache plan's 4.3 has the design, the user's answers and the Done note),
  committed as `6e5aa3a` and `383ee53`. The changes:
  - `notebooks/02_experiments.py`:
    - section 1's two cells export `project_root` and `config_hash`;
    - the tuner cell builds `studies_dir` (`studies/data=<hash8>__split_method=…__test_size=…__val_size=…__n_splits=…__seed=…__n_trials=…__n_startup_trials=…`),
      passes `storage_dir=studies_dir`, defines `build_study_name(step, run_settings,
      objective_name, study_seed)` and shows the directory;
    - every tune call passes `study_name`: steps `independent_cohorts`,
      `total_times_probability_coordinate_descent_total`,
      `total_times_probability_coordinate_descent_probability` (the user's renames of
      `name_study`, `model1`, `model2_coordinate_descent_*`). The EDA's Model 1 run settings are a
      dict in section 6's key order, so the two studies both sections run are stored once;
    - section 5's text (a storage paragraph) and section 6's (one sentence);
    - the EDA's two figure cells end in a bare `return` (marimo's trim, kept by the user);
  - `.gitignore`: `studies/`;
  - `src/.../hyperparameter_tuning/cross_validation.py`: a complete stored study is opened with
    `load_study` (read only) and returned; `create_study` only creates (the first Check C found
    `create_study(load_if_exists=True)` appending a line to every loaded file);
  - `tests/unit/test_hyperparameter_tuning_cross_validation.py`: the load test checks the file's
    bytes;
  - `tests/characterization/test_experiments_notebook.py`:
    `test_a_study_is_named_by_the_settings_that_shape_it`;
  - `docs/HYPERPARAMETER_TUNING.md` §6: the notebook's names, the read-only load;
  - `docs/NOTEBOOK_CACHE_EVALUATION_PLAN.md`: 4.3 ticked with its Done note;
  - this handoff.
- **4.3's checks:** the tuning module 21 passed under `-W error`, two mutations caught; the
  non-slow suite 1392 passed, 1 skipped, 1 xfailed, 19 deselected; mypy 28 files; Check A (the
  notebook test 14, ruff 15, the two hunks, `marimo check`); Check B; Check C twice on one copy
  (435 s, then 398 s loading: every frame equal to 4.2's run, then to the first with `seconds`;
  26 files, unchanged by the load).
- **The 100-trial clicks** (the real notebook, `PYTHONWARNINGS=error`): the first tuned and
  stored 25 studies (8.5 MiB; 27 tune calls, two shared by the EDA and section 6) in 1237 s, no
  failed trial; the second loaded them in 409 s (refits, bootstraps, inspection), every frame
  equal with `seconds`, the files unchanged. Best on validation: Model 1 `lgbm_poisson`, exposure
  on, val 2.1296 (cv 2.1402); Model 2 `lgbm_gaussian+logistic`, exposure on, val 2.1031 (cv
  2.0776). The user's real cache is `studies/` in the repo (git-ignored).
- **The scripts of this session** (if the scratchpad survived, copy from here first):
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/6f1d463f-a666-4a6a-be29-7ef29b80079c/scratchpad`:
  `checks/` (`make_copy.py` restores the two figure returns in the copy; `check_cache.py` with
  the new names; `run_all_4_3.sh`, `run_4_3.sh`, `run_clicks.sh`, `run_click.py`,
  `list_studies.py`, `compare_clicks.py`), the pickles `check_c_4_3.pkl` (4.3's run: the
  reference for 5.1), `check_c_4_3_loaded.pkl`, `click_first.pkl`, `click_second.pkl`.

### PR 3 (2026-11-05, closed)


- **Branches (2026-11-05):**
  - PR 3 is merged as **#21**, squashed into `main` as **`0c94f16`** (`feat(notebook): the
    objective EDA; each tuned run keeps its best model and its bootstrap intervals (#21)`). Its
    content is identical to the branch's last commit `92dd12c`, 3.3b's docs commit (`git diff`
    empty).
  - `feat/objective-eda` is deleted, locally and on GitHub.
  - **`feat/study-cache`** (PR 4) was created from `main` at `0c94f16`; its first commit is this
    handoff's update.
  - The user commits, pushes and merges. A squash-merged branch is deleted with
    `git branch -D` (`-d` refuses: git does not see it as merged).
- **The notebook's md5 at PR 3's close:** `95261326e5368bd451607d083a5ab366`.
- **Checks at the end of PR 3** (2.x's numbers below are PR 2's):
  - the non-slow suite 1387 passed, 1 skipped, 1 xfailed, 19 deselected (7 warnings, the old
    stack's);
  - mypy 28 files, no issue;
  - Check A: the notebook test 13 passed, ruff 14 (12 PLR1711, a B018, a RUF059), the user's two
    format hunks (`@@ -158`, `@@ -208`), `marimo check` no issue;
  - Check B;
  - Check C: the 24 baselines below and every assertion of §4, equal to 3.4's run.
- **PR 2 (history):** `main` was at `16773a5` (`refactor(notebook): the runs in loops scored on
  every metric; Model 2 by coordinate descent (#20)`), PR 2 squashed; its content was identical
  to the branch's last commit `eb0da75`. `feat/objective-eda` (PR 3) was created from it.
- **PR 2's commits** (each sub-task's "Done" note is in the metrics plan §4). They live inside
  the squash commit `16773a5`; GitHub keeps them under #20, and the docs cite them by hash:

  | commit | sub-task |
  |---|---|
  | `356246b` | the handoff after PR #18 and #19 (the branch's first commit) |
  | `9463651` | 2.1 a test: a shared choice reaches every fitted cohort model |
  | `b87e68b` | 2.2 Model 1's runs in one loop; the sharing flags as run settings (D27, D28) |
  | `5569d4d` | the handoff after 2.2 and the coordinate-descent measurement |
  | `4c75172` | 2.3a.1 the measurement recorded |
  | `a5a4234` | 2.3a.2 the component suggest functions and tune methods (D29) |
  | `9aed504` | 2.3 Model 2 by coordinate descent in three cells (D29) |
  | `04caa68` | 2.3's record and Model 2's new Check C baselines |
  | `c2843e3` | 2.4 `scoring.evaluate`; the notebook fits, predicts and evaluates; `fit_and_score` gone (D30) |
  | `8195afc` | 2.4b `model.evaluate` gone; the bootstrap takes the predictions and the cohorts (D30) |
  | `0a983cc` | 2.5a the guides, every changed block rerun |
  | `9862669` | the PR review's fixes |
  | `eb0da75` | 2.5b D27–D30, the metrics plan's status, this handoff |

- **The PR's review** (2.5, an independent subagent): no correctness bug; six findings, each
  reproduced; #1, #4–#6 fixed in `9862669`, #2 (the bootstrap does not check `cohorts` for
  duplicates or emptiness) left as is by the user, #3 (this handoff) rewritten here.
- **Checks at the end of PR 2:** the non-slow suite 1385 passed, 1 skipped, 1 xfailed, 19
  deselected (the 6 MLflow warnings of the old stack); mypy (28 files) no issues; Check A: the
  notebook test 13 passed, ruff the same 14 findings, `ruff format --diff` only the user's two
  hunks (`@@ -158`, `@@ -208`), `marimo check` the one empty cell. **The notebook's md5:**
  `b93448f43085733915262656f6ad1387` (read it via `cat notebooks/02_experiments.py | md5`, §4).
- **Check C baselines** (3 trials, both sharing flags off; `cv_best`). Model 1's are those
  recorded since T1.6; Model 2's are its pairs since 2.3 (coordinate descent).

  | Model 1 | exposure on | exposure off |
  |---|---|---|
  | `lgbm_poisson` | 2.1902 | 2.2814 |
  | `lgbm_gaussian` | 2.1906 | 2.3151 |
  | `poisson_glm` | 2.5703 | 2.8776 |
  | `nb2` | 2.2826 | 2.8904 |

  | Model 2 (pairs) | exposure on | exposure off |
  |---|---|---|
  | `lgbm_poisson+logistic` | 2.1306 | 2.1797 |
  | `lgbm_poisson+lgbm` | 2.1502 | 2.1993 |
  | `lgbm_gaussian+logistic` | 2.1777 | 2.2232 |
  | `lgbm_gaussian+lgbm` | 2.1973 | 2.2429 |
  | `poisson_glm+logistic` | 2.2540 | 2.8648 |
  | `poisson_glm+lgbm` | 2.2737 | 2.8845 |
  | `nb2+logistic` | 2.2381 | 2.8615 |
  | `nb2+lgbm` | 2.2577 | 2.8811 |

  Model 2's step runs at 3 trials: totals (on / off) `lgbm_poisson` 2.1361 / 2.1852,
  `lgbm_gaussian` 2.1832 / 2.2287, `poisson_glm` 2.2595 / 2.8703, `nb2` 2.2436 / 2.8670; the
  classifiers, with the best total (`lgbm_poisson`, on), logistic 2.1306, lgbm 2.1502 (each
  equal to its pair with that total). Model 1 with the notebook's flags (`share_hyperparams =
  True`): `lgbm_poisson` 2.1812 / 2.2193, `lgbm_gaussian` 2.2177 / 2.2805, `poisson_glm`
  2.4158 / 2.8685, `nb2` 2.2762 / 2.8707.
- **The 400-trial click:** Model 1 821 s (D20, 2026-10-22: best on validation `lgbm_poisson`,
  exposure on, cv 2.1280, val 2.1254); Model 2 by coordinate descent 752 s (2.3, 2026-10-31: the
  totals 26–70 s each, logistic 61 s, lgbm 309 s; no failed trial; the 16 pairs equal 2.3a.1's
  measurement; best on validation `lgbm_gaussian+logistic`, exposure on, val 2.0759). A full
  click is about 26 min, from 62.

## 2a. Model 2 by coordinate descent (decided and measured 2026-10-31)

The decision is D29; the measurement is the metrics plan's 2.3a.1, kept here as the record.

**Why one round is exact** (probed on the notebook's 5 folds, ≤ 3e-15 on each):
- Model 2's composite Poisson deviance is `D_total / K + 2 (C + N · log_loss) / (n K)`, where
  `D_total` is the total's deviance, `log_loss` the probabilities' per-child multinomial log
  loss, and `C` depends on `Y` alone;
- the prediction is `probabilities × total`, and `y_min` floors the total only; no
  hyperparameter reaches both parts;
- so the joint optimum is the pair of the two separate optima. Within a step the fixed part adds
  the same constant to every trial's value **under a mean aggregator** (`Mean`, the notebook's,
  or `WeightedMean`; not `UpperBound`, whose standard error the fixed part changes: the PR's
  review, reproduced). Step 1 may therefore fix the cheapest classifier;
- the argument is the Poisson deviance's: under another objective (RMSE, the log loss) the
  composite does not split this way, and a step is the best part given the other (§6, PR 3).

**Measured (400 trials, `PYTHONWARNINGS=error`; a scratchpad probe built from the package's
public pieces, then reproduced by the notebook's cells in 2.3).** The comparison with D20's joint
results:

| Pair | Exposure | cv joint | cv coordinate | val joint | val coordinate |
|---|---|---|---|---|---|
| lgbm_poisson+logistic | on / off | 2.0489 / 2.0878 | 2.0622 / 2.0813 | 2.1285 / 2.1777 | 2.0965 / 2.1963 |
| lgbm_poisson+lgbm | on / off | 2.0809 / 2.0976 | 2.0769 / 2.0960 | 2.1775 / 2.1679 | 2.1256 / 2.2254 |
| lgbm_gaussian+logistic | on / off | 2.0715 / 2.1336 | 2.0706 / 2.1094 | 2.1256 / 2.2040 | 2.0759 / 2.1721 |
| lgbm_gaussian+lgbm | on / off | 2.0962 / 2.1483 | 2.0853 / 2.1241 | 2.1202 / 2.2304 | 2.1050 / 2.2012 |
| poisson_glm+logistic | on / off | 2.2418 / 2.8511 | 2.2418 / 2.8511 | 2.1749 / 3.0380 | 2.1749 / 3.0378 |
| poisson_glm+lgbm | on / off | 2.2562 / 2.8656 | 2.2565 / 2.8658 | 2.2019 / 3.0672 | 2.2039 / 3.0669 |
| nb2+logistic | on / off | 2.2289 / 2.8457 | 2.2290 / 2.8457 | 2.1848 / 3.0489 | 2.1838 / 3.0489 |
| nb2+lgbm | on / off | 2.2479 / 2.8633 | 2.2437 / 2.8604 | 2.2132 / 3.0780 | 2.2128 / 3.0779 |

- **cv** (the objective): better in 9, the same in 6 (≤ 0.0005), worse in 1 (within the 0.02
  noise); the mean −0.004. **val:** better in 7, the same in 6, worse in 3 (one beyond the noise,
  holdout noise by its cv).
- **Why it is faster:** the classifier does not depend on the total, yet the joint design
  searched it in each of its 8 studies per classifier family: 3200 LightGBM-classifier trials,
  against 400 now; step 1's classifier is the cheapest fit. 6400 trials became 4000.
- **Step 2 does not depend on which total is fixed** (probed): with a cheap `poisson_glm` total
  fixed, step 2's logistic study picked the same `best_params` to the last digit, its values
  shifted by a constant. The user kept step 2 on the best total (its values are real pairs' cv).
- Optuna's `PartialFixedSampler` raises `ExperimentalWarning` under `-W error`, so the fixing is
  done in the package's code.

## 2b. PR 3 (closed 2026-11-05)

- **Commits on `feat/objective-eda`** (each sub-task's Done note is in the metrics plan §5; after
  the squash-merge GitHub keeps them under the PR, and the docs cite them by hash):

  | commit | sub-task |
  |---|---|
  | `55a9196` | the handoff after PR #20's merge |
  | `55dbb88` | 3.1a `RMSE` named; `EDA_OBJECTIVES` |
  | `17680ac` | 3.1b the EDA as section 5; Model 1's EDA run |
  | `1eab956` | 3.1c Model 2's EDA run, by coordinate descent |
  | `fecac25` | 3.1d the EDA's design and evidence (M10, M11) |
  | `c1cad4a` | 3.2 the 400-trial click and its reading; the deviance stays the objective |
  | `40ea139` | the log loss checked; the objectives cross-validated on every metric |
  | `ed9b5d5` | 3.1e every trial records every metric (`src`; M12) |
  | `4afb834` | 3.1f the trials tables |
  | `5c52a8a` | the handoff for 3.1h–3.1j |
  | `b3b2b4b` | 3.1h plotly 7.1.0 in the `notebook` group |
  | `b3b1c49` | 3.1i every trial as a row of the whole model; 3.1g removed |
  | `b4a0b77` | 3.1j the summary of the best models and the plotly figures; the handoff |
  | `52fa06d` | 3.1k the figures one column per model, the objective in a dropdown; 3.1l the objectives compared on a recorded metric |
  | `2d159c4` | 3.2b the trials read (docs) |
  | `36c6bbc` | the handoff for the section-6 session |
  | `afcd6f9` | 3.4 each run's best model, its transformers' names and its bootstrap intervals; a runs table and a scores table per model (M14) |
  | `b7da88d` | the handoff for the session on the user's additional changes |
  | `a006995` | the plan for PRs 4–7, the docs index, this handoff: for the session that does 3.3 |
  | `aa1e267` | 3.3a the review's F2: the "probability step" label once (subject typed `ix(notebook): …`) |
  | `92dd12c` | 3.3b PR 3 closed: the deviance kept; F1, F5; the stale sections; this handoff for PR 4 |

- **3.3 (2026-11-05):** no code change (D34). The review (3.3a), an independent subagent over
  `main..HEAD` at `a006995`, found no correctness bug and five findings, each reproduced twice:
  - F1: HYPERPARAMETER_TUNING.md named the removed `best_df` and said a run keeps every trial;
    fixed in 3.3b;
  - F2: Model 2's EDA figures printed the "probability step" label in all six panels; fixed in
    `aa1e267`, one `add_annotation` in the top panel;
  - F3, F4: left for PR 6 (§5);
  - F5: 3.1k's and 3.1l's commit lines named commits that do not exist; fixed in 3.3b.

  3.3b also replaced the stale "section 5" for the tuning: sections 5 and 6 run the tuner; the
  package was rebuilt from "the tuning section (then 5, now 6)", in README.md, docs/README.md,
  SPLITTING.md, HYPERPARAMETER_TUNING.md, MODULE_REFERENCE.md and the package's docstring.
  Checks on `a006995`: Check A, B, and Check C (396 s) equal to `check_c_3_4b.pkl` exactly. On
  `aa1e267`:
  - Check A, B;
  - Check C (400 s) every frame and record equal, Model 2's figures' labels 6 → 1;
  - both mutations caught by the new assertion.

  The notebook's md5 is `95261326e5368bd451607d083a5ab366`.

- **The planning session (2026-11-04, after `b7da88d`):** the user's seven changes were planned
  as PRs 4–7 in NOTEBOOK_CACHE_EVALUATION_PLAN.md with the decisions D31–D39 (its §0: the budget
  100 / 10, Model 2 by coordinate descent, the study cache by Optuna's journal files, the
  deviance kept as the objective with RMSE the comparison metric, `compute_metrics`, the bootstrap
  returning its replicates, two notebooks, SHAP per cohort, the test evaluation per E11). 3.3's
  checks ran on `b7da88d`: the non-slow suite 1387 passed, 1 skipped, 1 xfailed, 19 deselected
  (7 warnings, the old stack's); mypy 28 files, no issue; notebook md5
  `676212a808f6c1f87f7926a58b3bffe6`. Nothing was implemented then; 3.3 followed on 2026-11-05
  (above). The objective's evidence is D34; the user leaned to RMSE for both and chose the deviance
  on the EDA's numbers.

- **3.4 (2026-11-02–04, `afcd6f9`):** notebook md5 `676212a808f6c1f87f7926a58b3bffe6`. Its
  checks: the notebook test 13 passed; ruff 14 (none new); the two format hunks; `marimo check`
  no issue; Check B; Check C (3 trials, 405–500 s) with 3.4's assertions (§4); four mutations
  caught (the pairs' seed + 1, the transformer-name lookup inverted, the bounds swapped, a
  pair's seconds from one study). The user ran section 6 in marimo and approved it ("looks
  good"); a `KeyError: 'n_complete'` they saw first was stale kernel state (a `MODEL2_RUNS` built
  by the cell's older version), gone after a rerun. A 400-trial click is about 5.5 min longer
  than before 3.4 (the bootstrap about 7 s per call: train and val for each of the 24 kept models).

- **3.1i took four versions at the user's notes** (the metrics plan's 3.1i Done note): a
  `components_of` callable with `first_id`; the components set explicitly with one id sequence; a
  merged Optuna study of Model 2's two steps; finally each original trial kept, in the table rows,
  with the component its step held fixed, identified by `step` and its own study's number (§3).
  The user removed the two cells that displayed the trials tables before committing 3.1i.
- **The notebook at 3.1l:** md5 `affd69836a6d72cd453d862c2a2bb527` (3.1k's `8f14ab06395954d3fa6c7772e9de1f18`, 3.1j's `e14384d3ae57cb8b9d44eeffad1e811a`; `cat
  notebooks/02_experiments.py | md5`); `src` unchanged since 3.1e (`ed9b5d5`).
- **Checks at 3.1j:** the non-slow suite 1387 passed, 1 skipped, 1 xfailed, 19 deselected (at
  3.1h; `src` unchanged since); Check A: the notebook test 13 passed, **ruff 16** (14 PLR1711, a
  B018, a RUF059), the two format hunks, `marimo check` no issue; Check B; Check C (78 s) with
  3.1i's and 3.1j's assertions (§4); every frame and record equal to a reference run made before
  3.1i; the six figures render in a marimo kernel. The user looked at them (2026-11-02) and asked
  for 3.1k: its checks (the metrics plan's 3.1k) rerun everything above against 3.1j's run.
- **Checks at 3.1l (2026-11-02, the last notebook change):** the notebook test 13 passed; ruff 16
  (none new); the two format hunks; `marimo check` no issue; Check B; Check C (99 s) with every
  assertion of 3.1i–3.1l (§4); every frame and record equal to 3.1j's run exactly; the two
  dropdowns and four figures render; two mutations per sub-task caught. The non-slow suite was
  last run at 3.1h (`src` unchanged since 3.1e): 3.3 reruns it.
- **The 400-trial EDA of 3.2b:** rerun on `52fa06d` (709 s, no warning; 1200 + 2400 trials);
  its reading is the metrics plan's 3.2b. The user saw its figures through a small viewer app
  that loads the figures' JSON (`marimo edit <app>`, no tuning; §4).
- **The EDA's Check C values** (3 trials, `cv_best` by objective log loss / deviance / RMSE):
  Model 1 1.0855 / 2.1812 / 4.4406; Model 2 1.0833 / 2.1340 / 4.3941 (its deviance step 1
  2.1361; under the log loss its total step's trials all tie). Model 2's EDA equals an
  independent recomputation from the two tune methods exactly.
- **The 400-trial EDA (3.2)** and the cross-validated comparison are in the metrics plan's 3.2:
  the deviance-tuned models win every count metric on cv; the Poisson deviance stays the
  objective.

## 3. Decisions (do not re-ask)

The decisions and their evidence are in [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §0 (D1–D30)
and the metrics plan §2 (M1–M9); a later one supersedes an earlier one where it says so.

- D1: plain Optuna and a hand-written CV loop. D2–D3, D8, D10–D11, D23, D25: frozen,
  keyword-only records for the params and the components; search spaces hold every Optuna call;
  components hold the transformer object and are read back with `from_params`.
- D6, D12: one study per family combination and exposure, on the composite; LightGBM's two losses
  are families; the exposure a run setting, both arms. D7, D19: the plain mean of the folds, a
  univariate `TPESampler(seed)`, `catch=(RuntimeError,)`; no pruning, no other parallel place.
- D9: an aggregator combines the fold scores, `Mean` by default. D20: `n_trials = 400`,
  `n_startup_trials = 20`; all runs kept (the budget superseded by D31: 100 / 10).
- D21–D22: the package holds the mechanism, the notebook the experiment; `BootstrapEvaluator`
  holds the resampling design and takes the metrics per call. D24: `replication_method` fixed at
  `"weighted"`. D26: `CrossValidationTuner` with a named tune method per model.
- **PR 2 (2026-10-31): D27** the sharing flags are run settings; **D28** the runs in loops over
  run settings (supersedes one cell per run); **D29** Model 2 by coordinate descent, the joint
  method kept; **D30** fitting and evaluating apart (`scoring.evaluate`, no `fit_and_score`, no
  `model.evaluate`; the bootstrap on the predictions, per cohort, the interval alone).
- **PR 3 (2026-11-01), the metrics plan's M10–M13:** M10 the EDA tunes one setting per model
  (`lgbm_poisson`, exposure on), Model 2 by coordinate descent as the full tuning, logistic in its
  step 2, every study seeded `seed`; M11 the EDA is section 5, before the tuning; M12 every trial
  records every metric (`objective_metric`, `recorded_metrics`, `score_fold`, `cross_validate`,
  `score_trial`; `cv_score` removed); M13 the EDA's outputs (a row per trial of the whole model,
  a summary, plotly figures with stacked panels; no slices, no rank table). The Poisson
  deviance stays the tuner's objective (3.2). The button stays.
- **PR 3's 3.1h–3.1j (2026-11-01), the user's choices:** plotly 7.1.0 in the `notebook` group
  (marimo 0.24's frontend bundles plotly.js 3.8.2, plotly 7 targets 4.1.1; the same figure's JSON
  differs only in the default template's mapbox entries); `trials_table` takes the study and one
  tuple of components per trial, set explicitly in the objective loop with `from_params` (no
  callable, no helpers, no lambda); Model 2's rows keep each original trial with the fixed
  component in the row (no `src` change, no merged study), a `step` column and each study's own
  number; the summary's models `model1` / `model2` as section 6's `best_df`, Model 2's best in its
  `probability` step (the pair the refit holds); the figures' colors from the dataviz skill's
  reference palette (validated), dark mode not selected.
- **The user's choices in PR 2:** step 2 fixes step 1's best total (two loops in sequence), in
  three cells (step 1, step 2, the pairs); the pairs are not timed (`seconds` is a study's
  time); the score tables long, in `mo.ui.table`; the list named `METRICS`; `scoring.evaluate`
  returns a DataFrame and lives in `scoring`; the bootstrap's result always indexed (cohort,
  metric), no `score` column, every cohort bootstrapped in the notebook; the bootstrap's
  `cohorts` not checked (the review's #2); the run settings pinned by a notebook test.
- **Earlier choices** (2026-10-21 onward): `y_min` stays a build argument, recorded once per run
  with the seed; `Aggregator.aggregate_scores` is a named method; `num_leaves` above
  `2**max_depth` stays a known inefficiency; `scoring.Metric` keeps `direction` and `__call__`;
  `Metric.__call__` keeps its `cohort is None` branch; no private helpers, shared ones public in
  `utils.py`; `cohort` is a `str`, not a `Literal`.
- **PR 5 (2026-11-06):** D35 `compute_metrics` (done in 5.1); D36 the evaluator returns its
  replicates, `alpha` the notebook's (done in 5.2); **the two resampling designs stay** (rows,
  whole groups; no `unit` setting: rows within neighborhoods measured 0.36–0.82× as wide, an
  artefact of 1–2-row neighborhoods); the loops' result named `_replicate_scores`; the scores'
  columns `score, mean, median, sd, ci95_lower, ci95_upper` (the user's order); Check C compares
  `mean` at rtol 1e-12, the rest exactly. Commit messages are given in a copyable code block.
- **5.3's planning (2026-11-07):** `n_resamples = 1000` kept (measured); each model's best run
  by one selection metric (a setting, RMSE by default) on the validation point score; only the
  two best bootstrapped; one `forest_figure` function, its metric a parameter; 5.4 re-scoped to
  the paired difference and the importance on the selection metric.
- **The user's wish for this work:** do not overcomplicate; the smallest change that works.
  Keep a valid path working and available; reuse inside it is welcome (D29: the joint draw calls
  the component suggest functions).

## 4. How to work

The standing rules: [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §1 and
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3.

**Each sub-task:** plan mode first; re-verify the facts with read-only probes; a design question
gets a recommendation, the measured evidence and the trade-off (AskUserQuestion with code
previews works well; the user wants the question and the recommendation written so they can
select), and waits for the answer; write the steps and the commit message in the chat; one
sub-task per approval. At each stop: a file-by-file summary, the checks, the notebook's new
md5, the commit message. Tick the sub-task and update the status lines.

**Git and the editor:**
- The user commits, pushes and merges. No `Co-Authored-By` line, no "Generated with Claude
  Code" in PR text. Never `git checkout`, `stash`, `reset` or `mv`; delete with `rm`; restore a
  file with `git show HEAD:<path> > <path>` and `cmp`.
- **The user edits between turns** and sometimes commits before the stop: check `git log`,
  `git status` and the notebook's md5 before each sub-task and each edit; show any diff the plan
  does not name, and ask.
- **marimo's editor trims each cell's `return`** to the names other cells read when the user
  saves: ruff jumps to about 31 findings. The user has chosen to have HEAD's returns restored,
  after seeing the diff; in 4.3 the user kept two such bare `return`s (the EDA's two figure
  cells: `eda_figures`, `eda_metric_figures` are no longer exported), and Check C's copy
  restores them (`make_copy.py`) so it still reads the figures.
- **Format new cells and lines to `ruff format`'s output; never format the whole file** (the
  user's two hunks). Write new cells in the scratchpad, format each on its own, splice them in;
  for an edited line, apply exactly the hunk `ruff format --diff` prints.
- No new ruff finding. A cell lists every notebook name it uses as a parameter, sorted.

**The checks:**
- **Check A:** the notebook test, ruff (15 since 4.3: 13 PLR1711, a B018, a RUF059; none new), the
  format diff (the two hunks), `marimo check` (no issue since the user removed the empty cell).
- **Check B:** the names a script run defines (`app.run()`'s defs); nothing fitted.
- **Check C:** the clicked run on a copy. Build `<scratch>/project/notebooks/02_experiments.py`
  from the notebook with the line `    n_trials = 100` replaced by `    n_trials = 3` (by line),
  and each of the 8 lines `            "share_hyperparams": True,` (in `MODEL1_RUN_SETTINGS`)
  replaced by `False` (assert both counts); make `<scratch>/project/configs` a symlink to the
  repo's `configs`; run it with `marimo.ui.run_button` patched to a clicked value,
  `PYTHONWARNINGS=error`, `PYTHONPATH=<repo>/src`, in the background. About 75–180 s with the
  inspection and the EDA; the 24 `cv_best` must equal §2's. Since 3.1c it also asserts the EDA:
  the summaries' 12 rows and §2b's `cv_best`, and Model 2's EDA recomputed from the two tune
  methods (`dataclasses.replace(tuner, objective_metric=...)`) must equal its table exactly;
  since 3.1f the trials tables (rows, columns, each study's objective column's minimum its
  `cv_best`). Since 3.1i: 9 and 18 rows with the columns `id, objective, seconds, state`, the
  six metrics, `feature_transformers`, `hyperparameters` (Model 2's then `step`); Model 2's `step`
  `total` × 3 then `probability` × 3 per objective, ids 0–2 in each; `total` rows the fixed
  classifier (`glm_base`, `{"C": 1.0, "calibration_method": None}`), `probability` rows step 1's
  best total; the drawn hyperparameters equal the recomputed studies' params; each step's
  minimum its study's `cv_best`. Since 3.1j: `eda_best_df` 6 rows (`model1` × 3, `model2` × 3, the
  objectives in order), each `best_id` its study's `idxmin` (Model 2's in `probability`, equal to
  the recomputed study's `best_trial.number`), `cv_<objective>` its `cv_best`, `val_*` the
  composite rows; `eda_figures` 6 figures, each 6 y-axes titled in `METRICS`' order, each panel's
  `trials` trace its column, the `best: …` star at the best's position, Model 2's rule at 2.5 in
  6 shapes, none in Model 1's. Since 3.1k: no `best so far` trace; the trials' markers 5 px at
  opacity 0.6 without a ring, the stars 12 px; `eda_objective` a dropdown of the three objectives,
  `poisson_deviance` by default; the figures cell's output (from `app.run()`'s outputs) holds the
  two `tuned on poisson_deviance` figures and no other objective's. Since 3.1l:
  `eda_metric_figures` 12 keys (model × `METRICS`), each 3 boxes named the objectives in order with
  the palette's colors and symbols, each box's y its study's column (Model 2's x the `step`), 3
  stars at each best trial (Model 2's in `probability`), the y-range holding the 2nd–98th
  percentile and every star; `eda_recorded_metric` `poisson_deviance` by default; the display the
  two `poisson_deviance by objective` figures only. Since 3.4 (M14), against 3.1l's reference
  pickle (`check_c_3_1l.pkl`, the run before 3.4): the EDA's frames exactly; each record's keys;
  the components equal those `from_params` reads from 3.1l's `best_params`, the names its
  transformer entries; each `scores` 48 rows (2 datasets × 4 cohorts × 6 metrics), `score`
  equal 3.1l's `train_scores` / `val_scores` and the stored model's predictions, `ci95_lower ≤
  mean ≤ ci95_upper`; a pair's counts and seconds its two studies' sums; the runs tables 8 and
  16 rows (the hyperparameters the components' `estimator_params`), the scores tables 384 and
  768 rows equal to the records' `scores`; the two best runs' intervals and `importance_df`
  equal 3.1l's `intervals_df` and `importance_df` exactly; each model's last run's intervals
  recomputed with `random_state=seed`; every record's `scores` equal 3.4's first version's
  run (`check_c_3_4_first.pkl`) with the renames. The script lives in the scratchpad and must be rewritten in a new session
  (below). Since 3.3 (`aa1e267`) it also asserts the F2 fix: Model 2's EDA figures hold one
  "probability step" annotation (`x` the rule's, `yref` `"y domain"`, the top panel), Model 1's
  none. **For PR 4's first change,** the reference is 3.3's run of `aa1e267`,
  `check_c_3_3_fix.pkl` (every frame and record equal to 3.4's `check_c_3_4b.pkl`); if it is
  gone, run Check C on HEAD before the first edit and keep that pickle.
- **Since 5.2:** a scores frame has `score, mean, median, sd, ci95_lower, ci95_upper`;
  `compare.py` compares `mean` at rtol 1e-12 (pandas sums the replicates in another order than
  the old `np.mean`), the rest exactly, and drops the columns its reference lacks; `check_c.py`
  checks `ci95_lower ≤ median ≤ ci95_upper`, `sd > 0` and recomputes the last runs'
  statistics from `evaluate`'s replicates.
- **Since 5.3:** the records' `scores` and the scores tables hold `score` only
  (`SCORE_COLUMNS = ["score"]`); `compare.py` compares the shared columns; `check_c.py` takes a
  fifth argument, the 5.2b pickle, and checks the best runs: each the `min` of the validation
  composite `rmse` and `best_model*`; `best_runs_val_scores_df` (48 × 11) `score` the records', its
  statistics recomputed exactly from a fresh evaluator, equal to 5.2b's record of the run (`mean`
  at 1e-12) and to 3.1l's bounds; `best_replicate_scores` equal to the fresh replicates; the
  sentence once; `importance_df` equal to 3.1l's; the forest figure (4 traces on `x`…`x4`, the
  titles the cohorts, y the two labels, x the scores, `error_x` exactly, `customdata` the
  bounds, the y axis reversed; `rule_at=0` adds 4 lines at 0). `run_5_3.sh`:
  `check_c.py <copy> check_c_5_3.pkl check_c_3_1l.pkl check_c_3_4_first.pkl check_c_5_2b.pkl`.
- **Since 5b.2:** the 3.1j–3.1l assertions cover `model2_joint` like Model 1 (one study, no
  step; the displays three figures each); a 5b.2 block recomputes the three joint studies in
  memory (`storage_dir=None`): `eda_model2_joint_df` equal to their refit's scores, `cv_best`
  their `best_value` (the baselines below), the trials table's transformer names,
  hyperparameters and metrics equal to each trial's params and user attributes, `eda_best_df`'s
  joint rows (`best_id` the `best_trial.number`), and the copy's `studies/` holding the three
  `total_times_probability_joint__…` files. `compare.py` on the reference's frames;
  `compare_figures.py` the reference's figures' JSON key by key (the joint's are new).
  `run_5b_2.sh`: `check_c.py <copy> check_c_5b_2.pkl check_c_3_1l.pkl check_c_3_4_first.pkl
  check_c_5_2b.pkl`. **The joint EDA's Check C values** (3 trials, log loss / deviance / RMSE):
  1.0833 / 2.1543 / 4.4258 (the log loss equals coordinate descent's: only the probabilities
  count).
- **The scripts of 5.3's session (2026-11-07, the latest copy;** copy from here first): the
  scratchpad named in the status at the top (`checks/` adapted to 5.3; `check_c_5_3.pkl` is
  5.4's reference).
- **The scripts of 5.3's planning session (2026-11-07;** if it survived), in
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/6c4b7827-eaf3-4b57-8720-9133a2e32b80/scratchpad`:
  - `checks/`: the 2026-11-06 session's scripts, copied unchanged (`run_5_2b.sh` still names
    that session's scratchpad in `S=`; 5.3 writes `run_5_3.sh`), not yet adapted to 5.3;
  - `probes/probe_5_3.py`: the `n_resamples` measurement, the best runs by either metric, and
    the replicates' memory, on `probe_5_2.pkl` (about 10 min); its output `probe_5_3.log`, the
    per-seed bounds `probe_5_3_bounds.csv`;
  - the pickles `check_c_5_2b.pkl` (5.3's reference) with its figures' JSON,
    `check_c_3_1l.pkl`, `check_c_3_4_first.pkl` and `probe_5_2.pkl`.
- **The scripts of PR 5's session (2026-11-06;** if it survived, copy from here), in
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/6399f8d3-96a2-49b3-a707-91be30c7b8b4/scratchpad`:
  - `checks/`: `make_copy.py`, `check_b.py` (5.1's and 5.2's assertions), `check_c.py`,
    `compare.py`, `run_5_1.sh` / `run_5_2.sh` (the suite, mypy, a guide's blocks, Check A, B, C
    and the comparison), `run_5_2b.sh` (Check A, B, C only), `run_guide_blocks.py` (a guide's
    ```python blocks against its ```text blocks), the older `run_4_*.sh`; each `run_*.sh` names
    this scratchpad in `S=`: change it to the new session's;
  - the pickles `check_c_3_1l.pkl`, `check_c_3_4_first.pkl` (`check_c.py`'s references),
    `check_c_4_3.pkl`, `check_c_5_1.pkl`, `check_c_5_2.pkl`, `check_c_5_2b.pkl` (5.2's final
    run: 5.3's reference), with their figures' JSON;
  - `probes/probe_5_2.py` and `run_probe_5_2.sh` (the real notebook on a copy with a copy of
    `studies/`, every button clicked: about 7 min, nothing tuned), `probe_5_2.pkl`
    (`MODEL1_RUNS`, `MODEL2_RUNS` with their fitted models, `best_model1`, `best_model2`,
    `train_age_group`, `val_age_group`, `seed`, `METRICS`, `COHORTS`; load with
    `joblib.externals.cloudpickle`), `probe_5_2_widths.csv`.
- **The scripts of PR 4's session (2026-11-06;** if it survived, copy from
  here first), in `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/26bc3522-aa12-45fe-b71e-3e2c785cc628/scratchpad`:
  - `checks/`: 3.3's scripts with `make_copy.py` matching `    n_trials = 100`; `run_check_c.sh`
    with `S=` this scratchpad; `run_4_1.sh` / `run_4_2.sh` (Check A, B, C and the comparison);
    `run_4_3.sh` (Check A, B, then Check C twice on one copy); `check_cache.py` (the stored
    studies' names, computed independently); `compare_with_seconds.py`; `run_click.py` (the
    real notebook's click); `mutate.py` (4.2's mutations on `mut/`);
  - the pickles `check_c_3_1l.pkl`, `check_c_3_4_first.pkl` (`check_c.py`'s references),
    `check_c_3_3_fix.pkl`, `check_c_4_1.pkl`, `check_c_4_2.pkl`, `check_c_4_2b.pkl` (4.2's
    final run: the reference for 4.3), each with its figures' JSON;
  - `probe_load_4_2.py` (the load-time probe).
- **The scripts of the 3.3 session (2026-11-05, the latest copy;** if it survived, copy from
  here first), in
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/99f2beed-010f-4e31-a210-84ca1344c999/scratchpad`:
  - `checks/`: the planning session's scripts, with F2's assertion in `check_c.py`;
    `run_check_c.sh` with `S=` this scratchpad; `run_3_3a.sh` (Check A, B, C and the
    comparison on HEAD) and `run_fix.sh` (the same, then the mutants); its figures-JSON step
    has a quoting bug, so the comparison was run on its own;
  - the pickles `check_c_3_1l.pkl`, `check_c_3_4_first.pkl` (`check_c.py`'s two references),
    `check_c_3_4b.pkl`, `check_c_3_3a.pkl` (`a006995`'s run) and `check_c_3_3_fix.pkl`
    (`aa1e267`'s run: the reference for 4.1), each with its figures' JSON;
  - the review's probes in `review/`.

  `run_check_c.sh`'s `S=` must name the new session's scratchpad.
- **The scripts of the planning session (2026-11-04;** if it survived, copy from here first),
  in
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/65d10f12-872b-4ac3-a313-b2749802a1a3/scratchpad`:
  `checks/` (3.4's scripts, unchanged: `make_copy.py`, `check_b.py`, `check_c.py`, `compare.py`,
  `run_check_c.sh`, `run_all.sh`), the pickles `check_c_3_1l.pkl`, `check_c_3_4_first.pkl`,
  `check_c_3_4b.pkl` (3.4's final run: the reference for the first change after 3.4), and
  `suite_3_3.log` / `mypy_3_3.log` (3.3's checks). `run_check_c.sh` still names the 3.4 session's
  scratchpad in `S=`: change it to the new session's.
- **The scripts of 2026-11-04** (3.4's; if the scratchpad survived, copy them, check before
  rewriting), in
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/d5d309c8-a5ca-4f61-9fb4-e32eeb7a25f3/scratchpad`:
  `checks/` as below, with 3.4's assertions in `check_c.py`; `run_check_c.sh` passes the two
  references; the pickles `check_c_3_1l.pkl`, `check_c_3_4_first.pkl` and `check_c_3_4b.pkl`
  (3.4's final run); the mutants `mutant_*.py`.
- **The scripts of 2026-11-02** (if the scratchpad survived; check before rewriting), in
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/b6fef19f-9c0f-4e5f-b31f-4d53156a429c/scratchpad`:
  `checks/make_copy.py` (the 3-trial copy, both counts asserted), `checks/check_c.py` (every
  assertion above; `outputs, defs = app.run()`; pickles the frames and writes the figures' JSON),
  `checks/check_b.py`, `checks/compare.py` (two pickles, exact, `seconds` dropped),
  `checks/run_check_c.sh` and `checks/run_all.sh`; `run_eda400.py` (the 400-trial EDA on a copy
  with the EDA's stops opened); `read_trials_3_2b.py` and its log; `viewer/eda_figures_400.py`
  (the viewer app). The reference pickle of 3.1l's Check C is `check_c_3_1l.pkl` there.
- **A display's content:** `app.run()` returns `(outputs, defs)`; a cell's output is in
  `outputs` (its `.text`), so Check C can assert what a cell shows (3.1k, 3.1l). Match whole
  titles: "tuned **on**ce on" in the EDA button's label contains "tuned on".
- **A figure's rendering:** marimo registers its plotly formatter only inside a kernel
  (`running_in_notebook()`), so a script shows plotly's own iframe; run a small app through
  `marimo export html` and count `marimo-plotly data-figure=` (3.1j: Check C wrote the figures
  with `to_json()`, the app read them back with `plotly.io.from_json`). Count it without the
  `<`: inside an `mo.vstack` the export escapes it as `<`.
- **For the user to look at figures without a click:** run the cells on a copy (3 trials for the
  layout, or the 400-trial EDA), save the figures' JSON, and give the user a small marimo app that
  loads them (`.venv/bin/marimo edit <app>`): it opens in seconds and tunes nothing. marimo does
  not store outputs in the `.py`.
- **A reference-run comparison** (used since 2.4): before a change, run Check C on HEAD and
  pickle `eda_model1_df`, `eda_model2_df`, `model1_df`, `model2_df`, `model2_steps_df`, the two
  `*_arms_df`, `best_df`, `intervals_df`, `importance_df` and the records without `trials`,
  components or `seconds`; after it, compare frame by frame (`assert_frame_equal`, exact). Only Model 1's `seconds` (a study's wall
  time) may differ.
- **A guide's code blocks:** run each ```python block in order in one namespace, compare its
  output with the ```text block after it (the model guides need a prelude: the seed-0
  population and FEATURE_TRANSFORMATIONS §8's declarations).
- **Mutations** on a full copy (`src`, `tests`, `pyproject.toml` in the scratchpad; pytest with
  `--rootdir <copy> -c <copy>/pyproject.toml`), a sentinel mutation first to show the copy is
  the one imported.
- The machine can be heavily loaded: run the suite and notebook checks in the background with
  stdin closed, unbuffered, logged to the scratchpad; `timeout` does not exist on macOS.

**iCloud (found 2026-10-31):** the repo is in `~/Desktop`, which iCloud Drive syncs with storage
optimization; macOS offloads files ("dataless": the name stays, the content is downloaded on
read, seconds to 30+ s each).
- **Before a long run or a git command,** check `find .git src notebooks tests docs configs
  -flags +dataless -type f | grep -vc __pycache__` (the offloaded files in `mlartifacts`,
  `artifacts`, `dist` and the caches do not matter). A stalled process shows 0% CPU and its
  main thread in `read()` (`sample <pid>`; `lsof -p <pid>` names the file).
- **To bring a file back,** read it (`cat file > /dev/null`).
- **Long runs use `~/.venvs/age-group-prediction/bin/python`,** an identical environment outside
  iCloud, with `PYTHONPYCACHEPREFIX=<scratch>/pycache`. Compile that cache once without strict
  warnings (`python -m compileall -q <site-packages> src`): under `-W error` a fresh compile of
  `pyro` raises a `SyntaxError`.
- **mypy:** `~/.venvs/age-group-prediction/bin/mypy --cache-dir <scratch>/mypy_cache` (the 28
  configured files): the repo's `.mypy_cache` holds offloaded files and mypy crashes on them
  with an INTERNAL ERROR.
- **`grep -r`** must exclude `__pycache__` (and `.mypy_cache`, `.venv`): their offloaded files
  block it.
- **`uv sync` needs `--all-groups`.** The lasting fix is the user's: Finder's "Keep Downloaded"
  on the folder, turning off Optimize Mac Storage, or moving the repo out of `~/Desktop`.

**Traps:**
- **The fitted models need cloudpickle:** `pickle` fails on a feature transformer's lambda
  (`CenterByReferencePoint.build.<locals>.<lambda>`); `joblib.externals.cloudpickle.dump`
  works (5.2's probe).
- **A shell heredoc with an unquoted delimiter runs the backticks inside it:** write
  scripts with `<<'EOF'`, or from Python.
- **An indentation regex also matches a deeper indent** (12 spaces inside 16): anchor it to
  the line start, or replace the deeper block first.
- **The user's VS Code marimo kernel** (the marimo extension) may hold the notebook: tell the
  user before editing; the user edits between turns (5.2: a rename), so re-read the md5 first.
- **The scratchpad can vanish:** `/private/tmp` was emptied when the machine restarted
  (2026-11-01), taking the check scripts. Keep every number in the docs; rewrite the scripts.
  Another session's scratchpad may survive (2026-11-02: the previous one did); read its scripts,
  and copy them into the new session's scratchpad rather than writing there.
- **The user's open marimo session** holds the code it loaded: if it was not started with
  `--watch`, saving from it overwrites the edits on disk. Tell the user before editing the
  notebook while they have it open; a cell rerun in it keeps the EDA's results, a reopened
  notebook needs the click again.
- **A copy of the notebook run in the project's `.venv`** (the marimo kernel's environment)
  needs `src` beside it: its first cell adds `<notebook dir>/../src` to `sys.path`, and that
  environment does not import the package otherwise. Symlink `<scratch>/project/src` too.
- **A run button is "on" only for the run its click triggers:** a click runs every cell that
  reads it, one after another (the EDA's Model 2 cell starts after Model 1's), then resets; a
  cell run by hand sees it unclicked and stops at `mo.stop`. Cells added in between change
  nothing unless they read the button, define a name already defined, or close a cycle.
- In zsh, `echo ====` fails (`=` expansion): use `echo ----` in check scripts.
- Never run two tuner calls at once from threads in one process: joblib mixes their inputs (D19).
  Never edit `src` while a run that imports it is going: its fold processes re-import modules.
- Fold processes do not inherit `-W`; a warning there raises only under `PYTHONWARNINGS`.
- The characterization test requires every cell holding `tuner.tune_` to hold `mo.stop(not `;
  that string must not appear in a markdown cell or a comment outside the run cells.
- A function sent to joblib must not capture the notebook's defs (`_thread.lock`).
- pandas 3.0.5 turns a `None` calibration into NaN and `value_counts()` drops it: count by label
  after `fillna("None")` (D16). `json.dumps(record, default=str)` writes a bare `NaN`; use
  `trials.to_json(orient="records")`.
- NumPy sums one column pairwise and several row by row: a bootstrap mean over a different
  number of (cohort, metric) columns can differ in its last bit (the bootstrap test compares at
  1e-12).
- The user is not familiar with marimo's UI terms ("button"): explain them.
- **ruff's B023** (function-uses-loop-variable) is active: a lambda in a loop that reads a loop
  variable is flagged; a comprehension is not.
- `study.trials` returns deep copies (setting a copy's `params` leaves the study unchanged).

**The user's style:** loops over run settings, not one cell per run (D28); no function unless
it is reused; names that say what a thing holds (the composite is "the composite", a cohort a
cohort); a record holds the object and writes its declaration; one source of truth for a list of
candidates; no field or check that nothing reads; scikit-learn's terms; `if`/`elif` with a
raising `else`; code that may move to a package reads no notebook global; settings in the
constructor, data in the call; fitting and evaluating as separate steps; no private helpers.

## 5. Open items

- **The review's #2:** `BootstrapEvaluator.evaluate` does not check `cohorts` (duplicates give
  identical rows, `[]` a confusing `TypeError`); left as is by the user.
- **PR 3's review, F3 (for PR 6):** no repo test covers `build_trials_table`, `eda_best_df`, the EDA
  figures or section 6's records and tables; only Check C (a scratchpad script) does. A
  characterization test of `notebook_defs["build_trials_table"]` on a 3-trial study fits PR 6's test
  split (6.2: `test_objective_eda_notebook.py`).
- **PR 3's review, F4 (for PR 6):**
  - the lookup of a transformer's name by identity is written four times (`build_trials_table`,
    Model 1's run cell, Model 2's step 1 and step 2);
  - the hyperparameters are read two ways: `to_dict()["estimator_params"]` in `build_trials_table`,
    `dataclasses.asdict(component.estimator_params)` in section 6; the output is the same;
  - one shared function belongs with 6.1's shared setup, once the two notebooks exist.
- **T2.3's F5:** gone with 3.4 (the two arms tables were removed).
- **T1's F7:** `num_leaves` above `2**max_depth` changes nothing in LightGBM; revisit in a
  search-space pass.
- **The user's:** two empty cells and the bare `return`s marimo writes; ask before changing.
- **Stale text in section 4:** its header says "the check below prints it" and "fitted once …
  as a check", but the check cell is gone. The user's text; ask.
- **SPLITTING.md §3** and the simulator plan's Stop 4 doc examples (`.run()` → `simulate()`), and
  the simulator plan's cleanup stop (dead code) and format-only commit.
- **Univariate TPE can lock onto one region** (3.2b): under the deviance Model 2's step-1 total
  study spent 280 of 400 trials at `max_depth` 4 and 8 at depth 8, where the RMSE study found 15
  totals with a lower deviance (best 2.0601 against 2.0718). Candidates for a search-space pass,
  with T1's F7 below: Optuna's `multivariate=True`, more startup trials, a narrower space; not
  asked for.
- **The forest plots (5.3) look bad (the user, 2026-11-07, on the 100-trial runs through the
  viewer app):** to be fixed later (with 5.4's difference plot in PR 7); `forest_figure`'s
  layout is unchanged since 5.3.
- **The EDA figures:** dark mode not selected (the dataviz skill asks for it; marimo's dark theme
  shows them on the light surface); not looked at by the assistant (no browser). The user looked
  at 3.1j's, asked for 3.1k and 3.1l, and approved 3.1k's at 400 trials ("Looks good!").
- **A `log(n_apartments)` plan for the GLMs** (FEATURE_TRANSFORMATIONS.md §4.4); not asked for.
- **The package imports the old stack eagerly:** `age_group_prediction/__init__.py` imports the
  old modeling stack, which imports pyro; pyro's `SyntaxWarning` (an invalid escape sequence in
  its own source) shows at the notebook's first import, and each import costs about 3 s. The
  fix belongs to the roadmap's "delete the old stack" (a later PR); the user was told and left it.
- **What would shorten a click further** (about 31 min since 3.4), not asked for: dropping the GLM
  exposure-off arms (structurally worse by about 0.6); narrowing LightGBM's search after its
  fANOVA importances; fewer trials once the space is narrower.

## 6. Next, and the prompt

**Status (2026-11-07, PR 5 merged; PR 5b planned, not started):**
- PR 5 is merged as #23 (`c3bdee8` on `main`, identical to 5.3's `8a90bf9`); its branch is
  deleted. PR 5b is on `feat/model2-tuning-methods` from `main` at `c3bdee8`; its first commit is
  the docs (the cache plan's §4b, this handoff). The notebook's md5 is
  `9a66a347ed3fc928f14a43bde04da1b9` (unchanged since 5.3).
- Check C's reference for the next notebook change is `check_c_5b_2.pkl` since 5b.2 (with its
  figures' JSON; every shared frame and figure equal to `check_c_5b_1.pkl`'s), in the scratchpad named in the status at the top; `check_c.py`'s
  fifth argument stays `check_c_5_2b.pkl`. If they are gone, rewrite the scripts from §4 and run
  Check C on HEAD before the first edit, keeping that pickle.

Next, one step per approval, each stopped with a file-by-file summary, the checks, the
notebook's md5 and the commit message in a copyable code block:
1. **Close PR 5 at 5.3:** done (#23).
2. **PR 5b, the EDA** (`feat/model2-tuning-methods` from `main`; its first commit records this
   order and its plan in the cache plan as §4b):
   - 5b.1 `trials_table` renamed `build_trials_table` (done; the user: keep the function, a more
     informative name; the trials' scores are already saved during the runs as user
     attributes, the function rebuilds each trial's whole model from the flat params);
   - 5b.2 (done) the joint Model 2 method in the EDA (one `tune_total_times_probability_model` study
     per objective, `lgbm_poisson+logistic`, exposure on), `model2_joint` in the summary,
     figures and tables;
   - 5b.3 the reading at 100 trials; if the joint method wins beyond the noise, the
     `model2_tuning_method` setting and the joint loop go into the tuning notebook (PR 7).
3. **PR 6, the split** (the cache plan's §5; F3 and F4 there).
4. **PR 7, the tuning notebook's evaluation:** 5.4 first (the paired difference from
   `best_replicate_scores`, `total_times_probability − independent_cohorts` per replicate; its
   forest plot with `rule_at=0`; the importance on `SELECTION_METRIC`), the forest plots'
   layout fixed (§5), then SHAP per cohort and the test evaluation. MLflow and D4 stay
   deferred.

The user's preferences:
- plan docs in `docs/`, split into sub-tasks validated one at a time;
- each sub-task planned first (the facts re-verified with read-only probes; design questions
  with evidence and a recommendation, their context explained in the chat before asking), then
  implemented after approval;
- each sub-task stopped with a file-by-file summary, the checks, the notebook's md5 and the
  commit message in a copyable code block;
- marimo's UI terms explained;
- the implementation kept simple, with meaningful names;
- the user commits.

**The prompt for the next session:**

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full (the status at the top, §2's "PR 5" block,
§3, §4's checks, traps, iCloud notes and the latest scripts, §5, §6 first), then
docs/NOTEBOOK_CACHE_EVALUATION_PLAN.md (§0 D34–D36, §4b PR 5b in full, §4's 5.3 Done note and
5.4 moved to PR 7), then the notebook's section 5: the EDA's settings and tuner cells,
trials_table, Model 1's and Model 2's EDA run cells, the summary cell (eda_best_df), the two
figure cells and the tables cell; and CrossValidationTuner.tune_total_times_probability_model
in src/age_group_prediction/hyperparameter_tuning/cross_validation.py.
Verify the state first:
- the branch feat/model2-tuning-methods; HEAD my docs commit of PR 5b's plan on c3bdee8 (#23);
- `git status` clean; the notebook's md5 9a66a347ed3fc928f14a43bde04da1b9 via
  `cat notebooks/02_experiments.py | md5`;
- no iCloud-offloaded files: `find .git src notebooks tests docs configs -flags +dataless -type f
  | grep -vc __pycache__` is 0.
If anything differs, stop and show me the diff. Then copy the check scripts (checks/),
check_c_5_3.pkl with its figures' JSON, check_c_5_2b.pkl, check_c_3_1l.pkl and
check_c_3_4_first.pkl from the scratchpad the handoff names (or rewrite them from §4). In plan
mode, re-verify 5b.1 (trials_table renamed build_trials_table) against the code (line numbers
by grep), show it with the steps and the commit message in the chat, and implement it after my
approval. Then 5b.2 and 5b.3, one sub-task per approval. Keep the implementation simple. Stop
each sub-task with a file-by-file summary, the checks, the notebook's md5 and the commit message
in a copyable code block. I commit myself: never commit, no Co-Authored-By line, no "Generated
with Claude Code" in PR text, never git checkout/stash/reset/mv.
```
