# Tuning notebook revision: SHAP colours, LightGBM search space, MLflow content, notebook split

## Context

Four notes on `notebooks/03_hyperparameter_tuning.py` (4,293 lines, branch `feat/mlflow-runs`):
the beeswarm is not in shap's blue-to-red; the LightGBM search space does not follow the
library's guidance; the notebook mixes tuning with model inspection; MLflow runs carry too many
params and metrics to compare runs in the chart view.

This is the general plan. Each task opens with a short design stop (a table to approve) that
tightens it; then small build sub-tasks, one approval and one commit each.

**Decided in this session (do not re-ask):**
- The inspection notebook reads the tuning runs from MLflow, rebuilds each model's best run
  from `components.json` + `best_params.json`, and refits only those 2 models. This replaces
  the earlier "no read-back cell" decision.
- SHAP: beeswarm in shap's blue-to-red, colours clipped at each feature's 5th–95th percentile,
  waterfall in shap's red/blue. Categories stay gray (as shap does). Plotly stays.

**Assumed (say so if wrong):** the test evaluation (section 6) moves to the inspection notebook
with the rest; the legacy search space (`modeling_config.py`, `configs/modeling.toml`,
`models/direct_cohort.py`) is not touched (old stack, due for deletion in roadmap §5).

## Order and why

`0 → A (SHAP) → B (LightGBM) → C (MLflow content) → D (split) → E (one full run)`

- A is small and independent; it goes first.
- B changes `src` only, so it does not care about the notebook's shape.
- C goes before D: the inspection notebook will read the MLflow runs, so it should be built
  against the final logged content, not the content about to be slimmed.
- B, C and D each invalidate the stored studies or the check store, so the long run (24 studies
  × 100 trials, four clicks) happens **once**, in E. Sub-tasks are checked with the existing
  3-trial runs and the unit/characterization tests.

## Status

2026-11-20: PR #27 merged as e9ebd69; A1 committed (7524113); B1 committed (9df22d9); B2
committed (5b3f8d7); B3 done (the docs, the old studies moved to `studies/retired/`, Check C
re-recorded: the handoff's status of 2026-11-20; committed 9a8910b). Task B complete. C1
committed (bfa070e; the kept lists per run kind in Task C, the chart view measured on B3's
store, C2–C4's prompts refined). C2 committed (e299d66; the tuning runs log the kept params
and 16 metrics, `settings.json` and `scores.csv`). C3 committed (bb94acc; the evaluation runs
log 6 params, 54 metrics and `settings.json`). A Fable best-practice review of C2–C3 led to C3b
(four additions, Task C). C3b done (the tuning runs log `study_seed` as a param, 19 metrics and
Optuna's two figures; every run the `study_folder` tag; the evaluation runs
`best.<model>.run_id`, found before the run opens, a missing one a ValueError; committed
c09a18f). A second Fable review, of the whole MLflow content (the user's request), found the
design sound and led to C3c (the metrics logged last, a param-importances figure) and a query
cheat-sheet in C4 (Task C, "Second review"). C3c done (both loggers log the metrics last, so a
run that fails while logging holds none; a tuning run logs `figures/param_importances.html`,
fANOVA seeded with the study's seed, and none when the complete trials hold fewer than two
distinct values, where fANOVA raises: the review's finding, the user's choice; committed
7d586cd). C4 done 2026-11-21, awaiting the user's commit: `docs/MLFLOW_RUNS.md` §1, §4–§10 rewritten for
C2–C3c (the query cheat-sheet in §6; §8's recipe by `best.<model>.run_id`; §9's
`prediction_floor` from `settings.json`); `checks/check_store.py` adapted (the scratch store in
`~/age-group-prediction-scratch/c4/`), 0 failures on the 3-trial store (26 runs), and §8's and
§9's recipes ran from the doc on it. The user's look in the MLflow UI is recorded in Task C and
§7 (eye menu 10/20/all, the scatter reads a param as a number and draws 8 points, 24 bars, 12
mean bars grouped by `family`). An independent review of the doc against the notebook found no
factual discrepancy and six wording points, all applied. Next: D1 (Fable).

## Step 0 — clean start (done 2026-11-19)

- `03`'s uncommitted change (marimo's own re-save: trimmed returns, `@app.function`, an empty
  cell; no logic) restored from HEAD `932d43f` (md5 `5a3877b7…`), the user's choice.
- This plan copied to `docs/TUNING_NOTEBOOK_REVISION_PLAN.md`.
- Branch (the user's choice): `feat/mlflow-runs` is merged into main first, as it stands (the
  re-check of the final run on the old search space is skipped; Task E re-runs everything);
  tasks A–E go on a new branch from main, `feat/tuning-notebook-revision`. This file is
  committed on `feat/mlflow-runs` before the PR, so it reaches main with the merge.

## Task A — SHAP colours (Sonnet; one sub-task)

Files: `notebooks/03_hyperparameter_tuning.py` — `PALETTE` (l.1881–1891), `beeswarm_figure`
(l.2532–2624), the two captions (l.2647, l.3374); the waterfall reads `raises`/`lowers` only.

- **A1.** Add a `PALETTE["shap"]` colour scale sampled from shap's own colormap
  (`shap.plots.colors.red_blue`, ~8 stops written as hex literals, so no guess at the
  mid-tones); `beeswarm_figure` uses it instead of `palette["blues"]` (the counts scatter keeps
  the blues). Colour value = the feature clipped to its 5th–95th percentile, then min–max.
  `raises`/`lowers` → `#ff0051` / `#008bfb`. Captions and the two docs lines that say "blue
  ramp" updated.
  - Verify: the beeswarm test (`tests/characterization/test_hyperparameter_tuning_notebook.py`
    l.1276) still passes, plus one new assert for the colour scale and the clipping; a figure
    dump viewed by you.

## Task B — LightGBM search space

Files: `src/age_group_prediction/hyperparameter_tuning/search_spaces.py` (l.57–72, one method
shared by the regressor and the classifier), `params.py` (`LGBMParams`), `components.py`
(`make_estimator`, l.134–203), their unit tests, `docs/HYPERPARAMETER_TUNING.md` §4/§10.

Facts that drive the ranges: ~917 training rows, ~733 per CV fold, 10–11 features; no early
stopping (ruled out: the composite's `fit` has no `eval_set`); TPE, 100 trials, 10 startup.

**Final space (B1, 2026-11-20; measured on the current data and on copies of the stored studies):**

| Hyperparameter | Was | Final | Evidence |
|---|---|---|---|
| `max_depth` | 3–8 | **not tuned; fixed `-1`** | In the 30 stored LightGBM studies (34 LightGBM components, 3,400 component-trials) 35.6% of the component-trials had `num_leaves > 2**max_depth`, so `num_leaves` did nothing (95.6% at depth 3, 71.9% at 4, 47.0% at 5); 13 of the 34 best settings were among them. Its fANOVA share was 1.5% (mean over 29 studies; one had no variance). |
| `num_leaves` | 7–63 | **4–32, log** | CV RMSE with `max_depth=-1`, the rest fixed: the Poisson regressors are best at 12–16 (Model 1: 4.414 at 12; the total: 4.388 at 16), the Gaussian at 8 (4.442), the classifier at 4 (4.410). At 63 they are 0.9–2.7% worse than their best, at 2 leaves 0.6–4.3% worse; the training RMSE falls from 4.45 to 1.46 over the range (Poisson Model 1) while the CV score turns at 12. |
| `min_child_samples` | 5–40 | **5–40, log** (the user's choice) | With 733 rows a regressor's tree reaches on average about 570 / value leaves: 118 at 5, 57 at 10, 28 at 20, 14 at 40, 11 at 50 (the largest tree 5–10% more). The regressors are best at 1–5 and 2.1–2.2% worse than their best at 40 (8 leaves; 0.9–2.1% at 32 leaves); the classifier is flat over 5–50 (CV RMSE within 0.12% at 8 leaves, 0.35% at 32). For the regressors, uniform 5–40 puts 56% of the draws above 20 and 14% where `num_leaves` cannot be reached (20% with 5–50); the log scale 31% and 7%. Values of 1–3 scored 0.05–0.5% better than 5, the same sign in all six rows but inside the seed-to-seed range (0.4–0.8%), so the lower bound stays 5. |
| `n_estimators` | 50–400 | **50–200** | Your note; no problem found. 23 of the 34 old best settings used more than 200 trees, mostly at low rates (median 0.029; 5 of the 23 at 0.09–0.2). The grid has no cell above 200 trees, so that the cap costs nothing rests on the whole studies only (next row): Model 1 4.3842 / 4.3833 against the old space's 4.3828, the joint model 4.3521 against 4.3488 (+0.08%, one seed). |
| `learning_rate` | 0.01–0.2 log | **0.005–0.2 log** (the user's choice; the note said 0.001) | With at most 200 trees a Poisson regressor needs `learning_rate × n_estimators` of about 4: 0.95 of the gap between the constant model (CV RMSE 5.339) and the best cell is recovered at 4, 0.79 at 2 (+4.4%), 0.55 at 1 (+9.4%), 0.15 at 0.2 (+17.8%). The classifier is best at 0.01–0.02 with 100–200 trees and 5–13% worse at 0.2. No cell below 0.01 was the best for any model. Draws with a product of 1 or less: 40.5% from 0.001, 14.6% from 0.005, 3.5% from 0.01. Whole 100-trial studies on the new space (with `min_child_samples` still uniform 5–40; the final log scale was not run), best CV RMSE: Model 1 (Poisson, exposure; seeds 42, 43) 4.4008 / 4.4009 from 0.001, 4.3842 / 4.3833 from 0.005, 4.3841 / 4.3859 from 0.01 (old space 4.3828); the joint model (one seed) 4.3662 / 4.3521 / 4.3366 (old space 4.3488). |
| `reg_alpha`, `reg_lambda` | 1e-8–10 log | **1e-6–10 log** | Your note; confirmed. The CV RMSE is the same to four decimals at 1e-8 and 1e-6 and within 0.0002 at 1e-3; it first moves at 0.1. |
| `min_split_gain`, `subsample`, `colsample_bytree` | 0–1, 0.7–1, 0.7–1 | **kept, unchanged** (the user's choice) | At the 8-leaf base each moves the CV RMSE by at most 0.03 (about 0.7%), the size of the seed-to-seed range (0.4–0.8%; seed sd 0.007–0.013); fANOVA share 2–8% each against 21–45% for `learning_rate`. `min_split_gain` 0–1 is inert at 8 leaves except for the Gaussian with exposure (real split gains: median 6–13 for the Poisson fits and the classifier, 100 for the Gaussian, 2 for the Gaussian with exposure, where 12% of the splits are under 1) but prunes 32-leaf trees (Gaussian with exposure: 4.543 at 0 to 4.406 at 1). Not measured: whether a study without them does better. |
| `boosting_type` | default | **default (`gbdt`) only** | Not measured; the reason stands: `dart` needs more trees and has no stable stopping point, `goss` is a speed device for large data, `rf` is a different model. |

What `min_child_samples` counts (B1): LightGBM estimates a leaf's row count from its hessian, so
the bound is on the leaf's share of the hessian (the weights and, for Poisson, the predicted
mean), not on its rows; only the unweighted Gaussian is bounded in rows (smallest leaf at a bound
of 5: 2 rows for the Poisson fit with or without exposure, 1 for the Gaussian total with
exposure, exactly 5 for the unweighted Gaussian).
The classifier is fitted on one row per (building, cohort with a child), 2,161 rows a fold
weighted by 17,366 children; the bound counts those rows, not children, so it is about three
times looser than for the regressor (42 leaves reachable at 40 against 14). One shared range
serves both, since the classifier does not react to it.

The old journals (the user's choice): the directory
`studies/data=dd15b240__…__n_startup_trials=10/` (50 journals, 20 MB, git-ignored) is moved to
`studies/retired/2026-11-20_lgbm-space-v1/` in B3, after asking. No code change; `02` and `03`
both tune afresh on the next click. A later change of the space needs the same manual step
(`docs/HYPERPARAMETER_TUNING.md` §10 says so).

The measurement scripts and their outputs are in the B1 session's scratchpad
(`…/25384ce9-321d-4e2d-bfd8-d174714c7fbc/scratchpad/b1/`, temporary); the numbers above are the record.

- **B1 (Fable) — design stop. Done 2026-11-20.** The table above.
- **B2 (Opus).** `suggest_lgbm_hyperparams` with the final ranges (`num_leaves` and
  `min_child_samples` with `log=True`), `LGBMParams` without `max_depth`, `make_estimator` with
  `max_depth=-1` fixed beside `subsample_freq` in both the regressor and the classifier, the
  unit tests' names and fixtures (`test_hyperparameter_tuning_{search_spaces,params,components,builders}.py`;
  `components` l.212 and `builders` l.150–208 use `max_depth` as the changed hyperparameter and
  need another one; the notebook test's l.807–809 `max_depth` asserts).
  - Verify: the touched unit tests; a mutation per range and per scale.
- **B3 (Opus).** `docs/HYPERPARAMETER_TUNING.md` §4 table (the final ranges and scales, with the
  one-line reason per row), the parameter counts in §2 (l.~198) and §5 (l.~461) (LightGBM draws
  nine hyperparameters now: 30 / 12 / 10 parameters, not 33 / 13 / 11), and §10 hazard (the `2**max_depth` item goes; the "found by name
  only" item gains the manual retirement step); the handoff's two `2**max_depth` notes marked
  resolved; the old studies moved aside as decided, after asking; the 3-trial baselines
  re-recorded (Check C: 8 + 16 runs).
  - Verify: Check C against the new baselines, run once.
  - **Done 2026-11-20.** Also the handoff's third `max_depth` note (TPE locking onto depth 4)
    and a new status block at its top (the user's choice). Check C was re-recorded in full,
    not only the 24 numbers (the user's choice): W1's script adapted (A1's colours too), the 8
    baselines without LightGBM unchanged, the 16 others and the references for C2–D4 in the
    handoff's status.

**Note for you:** the check store's test scores (E11's one look) were produced with the old
space. A new space gives new best models, so the test set gets a second look. The change is
motivated by the library's docs, not by the test scores, but the handoff should record it.

## Task C — MLflow: fewer params and metrics

Files: `notebooks/03_hyperparameter_tuning.py` — `shared_params` (l.1021–1060),
`log_tuning_run` (l.1063–1168), `log_evaluation_run` (l.2729–2813), the two logging cells;
`docs/MLFLOW_RUNS.md`; the notebook tests at l.339, l.431, l.558; `check_store.py`.

Rule: **a param or metric is something you would put on a chart axis or filter by; everything
else is an artifact.** A value identical in every run cannot separate runs in a chart.

**The kept lists (C1, 2026-11-20):**

| Run kind | Params | Metrics | New artifacts |
|---|---|---|---|
| Tuning (24) | the run's settings (`family`, or `total_family` and `probability_family`; `use_exposure`; Model 1's `share_hyperparams` and `share_feature_transformer`); `config_hash`, `seed`, `split_method`, `n_trials`, `objective_metric`; `best.<name>` per best parameter, under the study's own names. 9 + `best.*` for Model 1, 8 + `best.*` for Model 2 (was 28 / 27 + `best.*`). | 16 (was 52): `cv/<objective>`, `{train,val}/composite/<6 metrics>`, `seconds`, `n_complete`, `n_failed` | `settings.json`, `scores.csv`; the six others unchanged |
| Best runs | 6 (was 23): `config_hash`, `seed`, `split_method`, `n_trials`, `objective_metric`, `selection_metric` | 54 (was 108): `val/<model or difference>/<metric>/{score, ci95_lower, ci95_upper}` | `settings.json` |
| Final | the same 6 (was 24) | 54 (was 108): `test/…`, the same names | `settings.json` |

- **`settings.json` is the complete record:** the 23 shared settings as they are now
  (`shared_params`, unchanged), plus `study_seed` in a tuning run and `n_train_val` in the
  final run. The params are a subset of it.
- **Only in `settings.json` from now on** (18, each one value in all 24 runs of a store):
  `config_path`, `n_neighborhoods`, `buildings_per_neighborhood_rate`, `simulation_seed`,
  `test_size`, `val_size`, `n_splits`, `n_startup_trials`, `n_resamples`, `alpha`, `n_repeats`,
  `n_shap_rows`, `n_shap_background`, `n_train`, `n_val`, `n_test`, `prediction_floor`, and in
  a tuning run `selection_metric`; also `study_seed` (it differs per run, and is in the
  `run_key`) and the final run's `n_train_val`.
- **`selection_metric` is a param of the evaluation runs only** (the user's choice): a tuning
  run does not depend on it and its `run_key` does not hold it, so a skipped run would keep
  the value of its first logging.
- **`scores.csv`** (the user's choice): all 48 rows of the run's scores, columns `dataset`,
  `cohort`, `metric`, `score`, the composite's 12 rows included, floats written in full (no
  `float_format`): the exact record of the 36 per-cohort scores that leave the metrics.
- **Unchanged:** the tags, the dataset inputs, the tuning run's six artifacts, the evaluation
  runs' tables (they keep `mean`, `median`, `sd`), figures and per-model JSON.

**Additions after a best-practice review (C3b, the user's choice, 2026-11-20).** A Fable review
of C2–C3 against MLflow's documentation (read-only) found the lists in line with it: the run
settings and `best.*` as params, as in MLflow's Optuna tutorial; the bootstrap mean, median and
SD rightly not metrics; the formats the documented ones. It found two deliberate deviations,
kept: no logged model (the "no pickle" decision) and the trials not as child runs (the tuning
run is the tutorial's parent run, the trials in `trials.csv` and the journal). Of its ranked
changes the user chose four:
1. **Optuna's figures per tuning run:** `figures/optimization_history.html`
   (`optuna.visualization.plot_optimization_history`) and `figures/slice.html` (`plot_slice`),
   by `log_figure`; 48 small files over the 24 runs.
2. **The cohorts' validation scores under the objective:** `val/<cohort>/<objective>` for the
   3 cohorts: 19 metrics (was 16). The objective, not the selection metric (the user's choice):
   the objective is in the `run_key`, so a skipped run never holds a stale name; with the
   defaults (both `rmse`) it is the same metric. The other per-cohort scores stay in
   `scores.csv`.
3. **Two tags:** `study_folder` (the study folder's name, the `run_key`'s first part) on every
   run, an exact filter for one setup; and on each evaluation run `best.<model>.run_id`, the
   best tuning run's id, found by its `run_key` (a FINISHED run; none found raises, the store
   then being inconsistent).
4. **`study_seed` a param again** (it differs per run): 10 + `best.*` for Model 1, 9 + `best.*`
   for Model 2; it stays in `settings.json` too.

Not taken: `mlflow.source.git.commit` set by the notebook. MLflow sets it itself (with the
branch and the repository's URL) when the notebook runs from the repository: all 26 runs of the
manual run on `.mlflow-check` carry it; only Check C's copies in a scratchpad lack it (the
user's choice). Also not taken: a model logged with a signature, trials as child runs, a
stepped convergence metric (the history figure covers it).

**Second review (C3c, the user's request and choices, 2026-11-20).** A Fable review of the whole
MLflow content after C3b (read-only; its queries on a scratch MLflow 3.16 store) found every
param, metric, tag and artifact relevant and both comparisons working: within a study
(`trials.csv`, the two Optuna figures, the journal) and between runs (the runs table and a bar
chart grouped by the `model` or `family` tag, `params.use_exposure`, a `best.*` scatter within
one component role, `best.<model>.run_id` from an evaluation run to its tuning runs). Its
findings, none blocking:
- **A FAILED run is a duplicate chart point:** both loggers log the metrics before the
  artifacts, so a failure in an artifact step (an offloaded journal, say) leaves a FAILED run
  with every metric, and the retry adds a FINISHED twin with the same `run_key`; the
  notebook's own reads filter on FINISHED, the UI does not. **Taken (C3c):** `log_metrics`
  last in both loggers.
- **The query pitfalls are not written down as a set:** a slash name needs backticks
  (``metrics.`val/composite/rmse` ``) in `filter_string` and `order_by`; params are strings
  (only `=`, `!=`, `LIKE`, `ILIKE`; `pd.to_numeric` on `search_runs`' columns); a boolean is
  `'True'` (`'true'` matches nothing); filter on `attributes.status = 'FINISHED'`; a model's
  validation score is `val/composite/<metric>` in its tuning run and
  `val/<model>/<metric>/score` in the best runs' run, and validation against test is the final
  run's `val_against_test` table or a pandas join with the prefix stripped (the UI cannot
  overlay disjoint names). **Taken (C4):** a cheat-sheet in `MLFLOW_RUNS.md`.
- **A hyperparameter ranking per study:** the slice plot is one-dimensional. **Taken (C3c):**
  `figures/param_importances.html` by `plot_param_importances` with
  `FanovaImportanceEvaluator(seed=study_seed)` (unseeded, a rerun would give other numbers;
  checked: it takes a seed and runs on a 2-trial study under `-W error`). 72 small files over
  the 24 runs. C3c's review found that fANOVA raises on a single complete trial
  (`ValueError`) and on equal values (`RuntimeError`), which would fail the run on every rerun
  and refuse the evaluation run; such a study logs no importances figure (the user's choice: an
  explicit check, no exception caught). On a 100-trial, 30-parameter study with conditional
  parameters and failed trials it runs in ~3 s with no warning.
- **Kept as they are:** the figures load plotly from a CDN (`log_figure`'s
  `include_plotlyjs="cdn"`), so they are blank offline (embedding costs ~3.5 MB a figure); the
  lookup takes the newest of two FINISHED runs with one key (the skip prevents a second);
  the harmless duplicates (`family` tag and param for Model 1, `best_params.json` and the
  `best.*` params, `final` and `stage=final`); no metric renamed.

**The chart view, measured (C1; B3's 3-trial store, 26 runs, read with sqlite, read-only).**
A scatter draws a run only if it holds both the param and the metric. Runs holding the param
(all hold `val/composite/rmse`):

| x axis | Check store (sharing off) | Real setup (sharing on) |
|---|---|---|
| `best.shared__<LightGBM name>` | none | 4 (Model 1: the 2 LightGBM families × exposure) |
| `best.<cohort>__<LightGBM name>` | 4 per cohort | none |
| `best.shared__alpha` (check: `best.<cohort>__alpha`) | 4 | 4 (2 GLM + 2 NB2; their ranges differ) |
| `best.total__<LightGBM name>` | 8 | 8 |
| `best.total__alpha` | 8 | 8 (4 GLM + 4 NB2; their ranges differ) |
| `best.total__dispersion` | 4 | 4 |
| `best.probability__<LightGBM name>` | 8 | 8 |
| `best.probability__C` | 8 | 8 |

- **The prefixed names compare across the families of one component role** (the Poisson and
  the Gaussian LightGBM share `shared__…`, and `total__…`), **never across the two models.**
  No alias is added (the user's choice, asked twice; a common `best.count__<name>` for a
  12-run scatter was offered and declined).
- **The check store has no `best.shared__…`** (Check C runs with both sharing flags off), so
  the scatter to open in C4 is `best.total__num_leaves` against `val/composite/rmse`, 8 points.
- **Every numeric `best.*` value is a number in every run that has it;** the transformer and
  calibration names are categories (a filter or a group, not a scatter axis).
- **Each point is one study's best trial,** so a scatter across runs shows where the best
  settings landed, with family, exposure and model differing between the points. A
  hyperparameter's effect is read within one study: `trials.csv`, or Optuna's `plot_slice` on
  the journal (`MLFLOW_RUNS.md` §7).
- **Models and families are compared by score:** the runs table sorted by
  `val/composite/<selection>` with the `family` tag (in all 24 runs) and `use_exposure` as
  columns; a bar chart of that metric, one bar per run; the runs grouped by the `family` or
  `model` tag.
- **Verified in C4's guided look** (MLflow 3.16 UI, chart view, on a copy of the 3-trial
  store, 26 runs): the eye icon in the runs list's header has a menu "first 10 / first 20 / all
  runs", so the limit exists; the default was not observed (the user thinks all runs), and
  "all runs" shows 26. The scatter of `best.total__num_leaves` against `val/composite/rmse`
  takes the param on an axis, reads it as a number (numeric ticks 5, 10, 15, 20) and draws 8
  points (two nearly coincide at 5). The bar chart of `val/composite/rmse` draws 24 bars, one per
  tuning run; grouped by the `family` tag it draws 12 bars, one per family value, each the
  mean of its 2 runs (checked against the store: 4.66 for lgbm_poisson, 5.16 for poisson_glm,
  4.56 for lgbm_gaussian+logistic). The bars carry no axis labels, only a colour and a legend that
  lists 5 of the 12 names, and colours repeat; hovering was not tried.

**Checked against Task D's read:**

| `04` needs | From | Kept |
|---|---|---|
| each model's tuning runs | the tags `stage`, `model` | yes |
| the runs of the current setup | `tags.run_key LIKE '<folder>/%'` and `params.objective_metric` | yes |
| each model's best | `metrics.val/composite/<selection>` | yes |
| the rebuild | `components.json` (each component's family, each count component's `use_exposure`), `best_params.json` | yes |
| the study name | the `run_key`'s second part | yes |
| `y_min`, `random_state` | `04`'s own constants, asserted against the run's `prediction_floor` (`settings.json`) and `seed` (a param) | `prediction_floor` moved, readable; `seed` kept |

`MLFLOW_RUNS.md` §9's recipe reads `prediction_floor` as a param of every run kind: after
C2–C3 it is in `settings.json` only (C4 changes that line and the table's row; the rest of
the recipe reads kept params and is unchanged). For D1: the best runs' and the final run's
keys share the folder prefix, so the `stage` filter is required.

**Run names.** A name is optional: without one MLflow generates a random one (like
`bold-cat-123`). The practice is to set a short readable name saying what distinguishes the
run, and to put everything you filter or group by in tags and params, never to parse the name.
Names need not be unique. The current names (`<model> <family> exposure=<bool>`, `best runs`,
`final`) already do this; the plan keeps them.

- **C1 (Fable) — design stop. Done 2026-11-20.** The lists, the measurements and the check
  above. No code.
- **C2 (Opus).** The tuning runs: `shared_params` stays the complete dict and is logged as
  `settings.json` (with `study_seed`); the params are the run's settings, the five named keys
  and `best.*`; 16 metrics; `scores.csv`; the tests at l.339 and l.431.
- **C3 (Opus).** The evaluation runs: `log_evaluation_run` logs the 6 params, `settings.json`
  and the statistics `score`, lower, upper (54 metrics); the test at l.558.
- **C3b (Opus).** The four additions above: the two Optuna figures and the three cohort
  metrics in `log_tuning_run`, `study_seed` a param again, the `study_folder` tag on every run,
  the `best.<model>.run_id` tags on the evaluation runs; the tests that pin them. **Done,
  committed c09a18f.**
- **C3c (Opus).** The second review's two code changes: `log_metrics` last in both loggers,
  the param-importances figure in `log_tuning_run`; the tests that pin them. **Done.**
- **C4 (Sonnet).** `docs/MLFLOW_RUNS.md` §4–§5, §7, §9's `prediction_floor` and §10's two
  messages; `check_store.py`'s counts
  and lists (copied into the session's scratchpad; it lives in an old one).
  - Verify (C2–C4): the touched tests; a 3-trial run into a scratch store, then the UI's chart
    view opened by you with the scatter and the bar chart.

## Task D — split into tuning and inspection notebooks

Cut: after the Model 2 tables (l.1736). `03_hyperparameter_tuning.py` keeps sections 1–5
(setup, studies, refits, point scores, the 24 tuning runs). New `04_model_inspection.py` gets
best-run selection, the validation bootstrap, permutation importance, SHAP, all figures, the
test evaluation, and the two evaluation runs' logging.

- **D1 (Fable) — design stop + probe.** Read-only probe on a scratch store: select each model's
  best run with `mlflow.search_runs` (`stage=tuning`, order by `val/composite/<selection>`),
  rebuild with `from_params`, refit, and show the scores equal the tuning run's logged ones.
  Decide: what `04` repeats from `03` (data, split, feature variants, metrics — duplicated and
  pinned equal by a test, the D41 precedent from the `02`/`03` split); what `04` does when the
  store has fewer than 24 finished runs; whether the `model1_runs`/`model2_runs` tables are
  still logged by the best runs' run or dropped (the tuning runs now hold them); the `notebook`
  tag of the evaluation runs.
- **D2 (Opus).** `04`: setup + the MLflow read + selection + refit, up to `best_runs`. `03`
  unchanged yet. Verify: `best_runs` in `04` equals `03`'s on the same store.
- **D3 (Opus).** Move the validation inspection cells and the best runs' logging to `04`;
  delete them from `03`. Tests re-pointed (they find cells by file).
- **D4 (Opus).** Move the test evaluation and the final run's logging. `03` ends at l.~1736.
- **D5 (Sonnet).** Docs: `MLFLOW_RUNS.md` (which notebook logs what, §10's steps),
  `HYPERPARAMETER_TUNING.md`, the handoff; the three-notebook setup pin.
  - Verify (D3–D4): logged content of the two evaluation runs equal to C's by `run_key`
    (`checks/compare_stores.py`), wall times masked.

## Task E — the one full run (you click; Opus guides)

Fresh check store, new studies: `03` two clicks (24 runs), `04` two clicks (best runs, final).
`check_store.py` 0 failures; the new test scores recorded in the handoff. Then the PR.

## Standing rules for every sub-task

Plan mode per sub-task with a chat summary before approval; static checks (ruff,
`marimo check`, the touched tests) before any long run; each check run once; an independent
review before the stop; commit message in a copyable block, committed by you; no writes to the
repo's `mlflow.db`, `mlartifacts/` or `.mlflow-local/`; `find . -flags +dataless` before a run.

## Session prompts

One session per sub-task, in the plan's order: A1 → B1 → B2 → B3 → C1 → C2 → C3 → C3b → C3c → C4 → D1 → D2
→ D3 → D4 → D5 → E. At its stop a session prints the **next** sub-task's prompt from this
section, verbatim, in a copyable block, with the model's name above it. It does not write a new
prompt. A design stop (B1, C1, D1) may refine the prompts of its own task's build sub-tasks
here, as part of its output.

**Every prompt starts with this opening** (replace `<ID>`):

```text
Read docs/TUNING_NOTEBOOK_REVISION_PLAN.md: the Status line, the decisions at the top, sub-task <ID>'s task section, the standing rules and the session prompts. Plan sub-task <ID> in plan mode and do only <ID>. Before the first edit: confirm the branch is feat/tuning-notebook-revision, the working tree is clean, and no source file is offloaded (find . -flags +dataless, caches aside). At the stop: a file-by-file summary, the check results, the commit message in a copyable block, the plan's Status line updated, and the next sub-task's prompt printed verbatim from the plan with its model named.
```

**A1 (Sonnet)** — the opening, then:

```text
A1: the SHAP colours in notebooks/03_hyperparameter_tuning.py. PALETTE["shap"] is a plotly colour scale of about 8 stops sampled from shap.plots.colors.red_blue, written as hex literals; show me the sampled values and how you got them. beeswarm_figure uses it instead of palette["blues"] (the counts scatter keeps the blues); the colour value is the feature clipped to its 5th–95th percentile, then min–max; category levels stay gray, one colour bar. PALETTE["raises"] / ["lowers"] become #ff0051 / #008bfb. Update the two beeswarm captions and the docs lines that say "blue ramp". The beeswarm test still passes; add asserts for the colour scale and the clipping. Verify: ruff, marimo check, the touched tests (run once), and a figure dump of one validation beeswarm and one waterfall for me to look at.
```

**B1 (Fable; design stop, no code change)** — the opening, then:

```text
B1: decide the LightGBM search space with measurements. Start from Task B's table; my notes are fixed (max_depth not tuned and set to -1, n_estimators 50–200, learning_rate lower bound 0.001, reg_alpha / reg_lambda lower bound 1e-6, boosting_type the default) unless a measurement shows a problem, which you put to me as a question. Measure on the current data and the stored studies (read-only, scripts in the scratchpad): the leaves a tree actually reaches against num_leaves and min_child_samples at ~733 rows a fold; the score against num_leaves; the score against learning_rate × n_estimators, and how many trials would land where the model cannot fit with at most 200 trees; what min_child_samples means for the weighted classifier; whether min_split_gain, subsample and colsample_bytree earn their place in 100 trials. Decide how the old journals in studies/ are retired (matched by name only; 02_objective_eda.py shares them). Output: the final table with the evidence per row written into Task B, the open choices asked with the question tool, and B2's and B3's prompts refined below.
```

**B2 (Opus)** — the opening, then:

```text
B2: build the search space B1 decided (Task B's final table): suggest_lgbm_hyperparams with num_leaves 4–32 log, min_child_samples 5–40 log, learning_rate 0.005–0.2 log, n_estimators 50–200, reg_alpha / reg_lambda 1e-6–10 log, min_split_gain 0–1, subsample 0.7–1, colsample_bytree 0.7–1, and no max_depth; LGBMParams without max_depth; make_estimator with max_depth=-1 fixed beside subsample_freq, for the regressor and the classifier. The unit tests' names and fixtures (test_hyperparameter_tuning_search_spaces, _params, _components, _builders): where a test uses max_depth as the hyperparameter it changes (components l.212, builders l.150–208, params l.34), use num_leaves; the notebook test's max_depth asserts (l.807–809). The legacy space (modeling_config.py, configs/modeling.toml, models/direct_cohort.py) is not touched, and nothing in studies/ is touched (B3 retires it). Verify: the touched unit tests, a mutation per range and per log scale; ruff and mypy on the touched files.
```

**B3 (Opus)** — the opening, then:

```text
B3: docs/HYPERPARAMETER_TUNING.md §4's table (Task B's final ranges and scales, one line of reason per row), the parameter counts in §2 (l.~198) and §5 (l.~461) (nine LightGBM hyperparameters: 30 / 12 / 10 parameters, not 33 / 13 / 11), and §10's hazards (the 2**max_depth item removed; the "found by name only" item says a changed space needs the studies directory moved aside); the two 2**max_depth notes in docs/EXPERIMENT_NOTEBOOK_HANDOFF.md marked resolved, and the note that the test set gets a second look. Retire the old studies as B1 decided: move studies/data=dd15b240__… to studies/retired/2026-11-20_lgbm-space-v1/ (ask me before moving; nothing is deleted), and check 02_objective_eda.py and 03 then find no stored study. Re-record the 3-trial baselines (Check C, 8 + 16 runs), run once.
```

**C1 (Fable; design stop, no code change)** — the opening, then:

```text
C1: decide exactly which params and metrics each run kind keeps, by the rule in Task C (a chart axis or a filter; the rest an artifact). Check the lists against what the inspection notebook's read will need (val/composite/<selection>, components.json, best_params.json, the study name) and against the chart view: on a scratch store with 3-trial runs, show which scatter plots work across runs (a best.* hyperparameter against a validation metric) and whether the prefixed names (shared__, total__, probability__) compare across families. Output: the kept lists per run kind written into Task C, what moves to settings.json and scores.csv, open choices asked with the question tool, and C2–C4's prompts refined below.
```

**C2 (Opus)** — the opening, then:

```text
C2: the tuning runs as C1 decided (Task C's kept lists), in notebooks/03_hyperparameter_tuning.py. shared_params stays the complete dict of 23 settings; log_tuning_run logs it with study_seed as settings.json, and as params only the run's settings, config_hash, seed, split_method, n_trials, objective_metric and best.<name> (no selection_metric, no study_seed, none of the other settings). Metrics: cv/<objective>, the composite's 12 train/val scores, seconds, n_complete, n_failed (16). scores.csv: all 48 rows of tuning_run["scores"] as dataset, cohort, metric, score, floats written in full (no float_format). The other six artifacts, the tags and the inputs are unchanged; log_evaluation_run and its two callers are not touched (C3). Tests: the shared_params pin (l.356–365) stays, and the tuning-run test (l.431) pins the exact param names of a Model 1 and a Model 2 run, the 16 metric names, settings.json equal to shared_params plus study_seed, and scores.csv read back (float_precision="round_trip") equal to the scores exactly. Verify: ruff, marimo check, the touched tests, run once; a mutation per kept list (a param added, a metric added, scores.csv rounded).
```

**C3 (Opus)** — the opening, then:

```text
C3: the evaluation runs as C1 decided (Task C's kept lists), in notebooks/03_hyperparameter_tuning.py. log_evaluation_run is given the complete settings (shared_params; the final run's with n_train_val) and logs them as settings.json, and as params only config_hash, seed, split_method, n_trials, objective_metric, selection_metric. Metrics: per model and for the difference, per metric, the statistics score and the two bounds (bound_names) only: 54 in a run; mean, median and sd stay in the tables, which are unchanged, as are the figures, the per-model JSON, the tags and the inputs. The two callers' messages need no change (they print the returned count). Tests: the evaluation-run test (l.558; its n_metrics 18 becomes 9) pins the 6 param names, the metric names and settings.json. Verify: ruff, marimo check, the touched tests, run once; a mutation per kept list.
```

**C3b (Opus)** — the opening, then:

```text
C3b: the four additions in Task C ("Additions after a best-practice review"), in notebooks/03_hyperparameter_tuning.py. log_tuning_run: study_seed back among the params (it stays in settings.json); the metrics gain val/<cohort>/<objective> for the 3 cohorts (19 in a real run; the objective is objective_metric_name, not the selection metric); two figures by mlflow.log_figure, figures/optimization_history.html (optuna.visualization.plot_optimization_history(study)) and figures/slice.html (plot_slice(study)). Every run gets the tag study_folder, the study folder's name (the run_key's first part): the tuning runs from studies_dir, the two evaluation runs from what their callers pass. log_evaluation_run tags each best run's tuning run id as best.<model>.run_id, found by search_runs on tags.run_key = '<folder>/<study name>' and status FINISHED; none found raises a ValueError naming the run_key. Nothing else changes. Tests: the tuning-run test (l.431) pins the 10 / 9 params, the metrics (a cohort's validation score under the objective kept, its other per-cohort scores and every train score of a cohort not), the two figures among the artifacts and the study_folder tag; the evaluation-run test (l.634) logs two tuning runs first (or FINISHED runs with the right run_key) and pins the two run_id tags and study_folder, plus a case where a tuning run is missing and the ValueError is raised. Verify: ruff, marimo check, the touched tests, run once; a mutation per addition.
```

**C3c (Opus)** — the opening, then:

```text
C3c: the second review's two code changes (Task C, "Second review"), in notebooks/03_hyperparameter_tuning.py. In log_tuning_run and log_evaluation_run, mlflow.log_metrics moves to the end of the run's block (after every artifact and input), so a run that fails while logging holds no metrics and never shows in a chart; the metrics dict is still built before the run opens where it is now. log_tuning_run logs a third Optuna figure, figures/param_importances.html: optuna.visualization.plot_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=tuning_run["study_seed"])). Nothing else changes (names, counts, return values, the skip and the lookup). Tests: the tuning-run test's artifact list gains the figure; a new test per logger in which an artifact step fails (the tuning run: the study's journal file missing; the evaluation run: a figure that log_figure cannot write) and the run is FAILED with no metrics; the importances figure equal across two logs of the same study (seeded). Verify: ruff, marimo check, the touched tests, run once; a mutation per change (log_metrics back before the artifacts in each logger; the evaluator unseeded or the figure dropped).
```

**C4 (Sonnet)** — the opening, then:

```text
C4: the docs and the checker for C2–C3c. docs/MLFLOW_RUNS.md: §4 (the params per run kind with study_seed, the tags study_folder and best.<model>.run_id, settings.json, scores.csv and the three Optuna figures among a tuning run's artifacts, what settings.json holds, the metrics logged last so a FAILED run holds none), §5 (19 and 54 metrics), §6 (study_folder as the exact filter for one setup, and the query cheat-sheet of Task C's "Second review": backticks for a slash name in filter_string and order_by, params as strings with pd.to_numeric, booleans as 'True', the FINISHED filter, validation against test from the final run's val_against_test table or a pandas join with the prefix stripped), §8 (the evaluation run's best.<model>.run_id as the direct link; the run_key search stays as the fallback), §7 (models and families are compared by score: the runs table, a bar chart of val/composite/<selection>, the runs grouped by the family or model tag; a best.* scatter holds only the runs of one component role, Task C's coverage table; a hyperparameter's effect is read within a study, now first in the run's figures/slice.html, figures/param_importances.html and figures/optimization_history.html, the journal recipe kept for the other Optuna charts), §9's table and recipe (prediction_floor from settings.json for every run kind; the rest unchanged, it reads kept params), §10's two messages (108 metrics becomes 54). check_store.py copied from /private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/1e116e6f-752d-40d8-bedc-ad034d51e94c/scratchpad/manual/ into this session's scratchpad: the param lists per run kind, 19 and 54 metrics, the new artifacts and figures, the two tags (each best.<model>.run_id naming a FINISHED tuning run whose study name is the best.<model> tag). Then a 3-trial run into a scratch store (never the repo's stores), check_store.py with 0 failures, and guide me in the UI's chart view, one step at a time: show all runs (the view may draw only the first 10), the scatter of best.total__num_leaves against val/composite/rmse (8 points expected), and a bar chart of val/composite/rmse over the 24 tuning runs grouped by the family tag. Record in Task C what the look showed for the points C1 could not verify (a param on a scatter axis read as a number, the first-10 default, the bar chart and the grouping).
```

**C4, resume (Sonnet)** — the validation of the UI look; the build is done and uncommitted:

```text
Read docs/TUNING_NOTEBOOK_REVISION_PLAN.md: the Status line, Task C (the chart-view measurements and "Not verified in C1"), sub-task C4, the standing rules. Continue C4 from its validation: its build is done and uncommitted (git status: docs/MLFLOW_RUNS.md modified; HEAD 7d586cd). Confirm the branch is feat/tuning-notebook-revision and that only docs/MLFLOW_RUNS.md differs from HEAD; no source file offloaded (find . -flags +dataless, caches aside). The checker is checks/check_store.py (in the repo, uncommitted; C2–C3c's counts and lists; usage in its docstring). The 3-trial scratch store is in ~/age-group-prediction-scratch/c4/ (outside /tmp and iCloud): mlflow_C4.db and mlartifacts_C4 (26 FINISHED runs, experiment check-c4, artifact paths already pointing there; check_store.py reports 0 failures on a copy of it), recipe_s8.py and recipe_s9.py (§8's and §9's code blocks run from the doc), check_c4.log, and run_C4.sh with make_copy.py and run_c4.py, which rebuild the store in about 2 minutes if it is ever lost (never into the repo's stores). Work on a COPY of mlflow_C4.db for the UI and the checks. Then: (1) start the MLflow UI on a copy of the store (mlflow ui --backend-store-uri sqlite:///<copy> --artifacts-destination <mlartifacts_C4> --port 5001) and guide me in the chart view, one step at a time, waiting for my answer: show all runs (the view may draw only the first 10), the scatter of best.total__num_leaves against val/composite/rmse (8 points expected; is a param read as a number on an axis?), a bar chart of val/composite/rmse over the 24 tuning runs grouped by the family tag. (2) Record what I see in Task C (the points C1 could not verify) and in the closing paragraph of MLFLOW_RUNS.md §7 (§7 already refers to "the UI check"; fix any sentence the look shows wrong, e.g. the scatter's 8 points or the first-10 default). (3) An independent review of MLFLOW_RUNS.md against notebooks/03_hyperparameter_tuning.py (an agent given the diff, not my conclusion). (4) Rerun check_store.py once only if the docs' numbers changed. At the stop: a file-by-file summary, the check results, the commit message in a copyable block (no Co-Authored-By, no "Generated with" line), the Status line updated (C4 done, awaiting commit), and D1's prompt printed verbatim from the plan with its model named.
```

**D1 (Fable; design stop + read-only probe)** — the opening, then:

```text
D1: design the split into 03 (tuning) and a new 04_model_inspection.py. Probe on a scratch store: select each model's best run with mlflow.search_runs, rebuild it from components.json + best_params.json with from_params, refit, and show the scores equal the tuning run's logged ones. Decide: what 04 repeats from 03 and how the copies are pinned equal (the D41 precedent); what 04 does when the store has fewer than 24 finished runs; whether the model1_runs / model2_runs tables stay in the best runs' run; the evaluation runs' notebook tag; which tests move. Output: the decisions written into Task D, open choices asked with the question tool, and D2–D5's prompts refined below.
```

**D2, D3, D4 (Opus), D5 (Sonnet)** — the opening, then the sub-task's lines from Task D as D1
left them.

**E (Opus guides, the user clicks)** — the opening, then: `E: guide me through the full run on a fresh check store (03 two clicks, 04 two clicks), one step at a time, and remind me to note each message before closing marimo; then check_store.py, the test scores into the handoff, and the PR's title and description.`

## Verification (end to end)

1. `pytest tests/unit -k hyperparameter_tuning` and
   `tests/characterization/test_hyperparameter_tuning_notebook.py` green after each sub-task.
2. Check C (3-trial runs) against the baselines re-recorded in B3.
3. Task E's full run: 26 FINISHED runs, `check_store.py` 0 failures, the chart view showing a
   hyperparameter against a validation metric across runs, the beeswarm blue-to-red.
