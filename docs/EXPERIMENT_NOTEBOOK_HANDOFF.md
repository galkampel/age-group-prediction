# Handoff: the experiment notebook, PR 3 in progress (the objective EDA)

**For:** the session that continues PR 3 of
[NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md), the objective EDA, at its
sub-task 3.1h (§2b, §6). That
plan has three PRs: the metrics in `scoring` (PR 1, merged as #18 into
`feat/experiment-notebook`, which #19 merged into `main`), the runs in loops scored on every
metric (PR 2, **merged as #20**, squashed into `main` as `16773a5`), and the objective EDA
(PR 3, on **`feat/objective-eda`**, created from `main` at `16773a5`). MLflow and E11's final
evaluation are deferred.

The next session starts at §6.

**Written:** 2026-10-29 after the tuning plan's T5; **rewritten** 2026-10-31 at the end of PR 2
(2.1–2.5); **updated** 2026-10-31 after PR 2's merge (#20), for a new session; **updated**
2026-11-01 in PR 3 (§2b, §6), for a new session. The detailed history is in the metrics plan (§2 M1–M9, §4 2.1–2.5 with each
sub-task's "Done" note), in [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) (§0 D1–D30) and in
git. Re-verify every fact here against the code of the day.

## 1. What exists

**The notebook** (`notebooks/02_experiments.py`, marimo; a script run fits nothing):
1. **Data:** the simulator at 150 neighborhoods × 9 buildings (`configs/simulation.yml`, its
   seed 42 unless `simulation_seed` overrides it), the room shares, `X`, `Y` (three cohorts),
   the exposure (`n_apartments`), the groups (neighborhoods); the settings printed and hashed.
2. **Settings:** `seed = 42`, `split_method = "random"`, `test_size = val_size = 0.2`,
   `n_splits = 5`, `n_trials = 400`, `n_startup_trials = 20`, `n_resamples = 1000`,
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
   where it is read); `trials_table`; the button "Run the objective EDA"; Model 1's EDA run
   (`lgbm_poisson`, exposure on, the notebook's flags) and Model 2's (`lgbm_poisson+logistic`,
   exposure on, by coordinate descent), each one study (Model 2: two) per objective through
   `dataclasses.replace(tuner, objective_metric=objective)`, every study seeded `seed`; each best
   refit and scored on validation (`eda_model1_df`, `eda_model2_df`, 12 rows each), every trial
   kept (`eda_model1_trials_df`, `eda_model2_trials_df`); a table cell. Its rework is next (§6).
6. **Section 6, tuning and inspection:**
   - **the metrics cell:** `POISSON_DEVIANCE = Metric("poisson_deviance",
     mean_poisson_deviance_loss)` (the tuner's objective, the best-run rule, the permutation
     importance's score), `LOG_LOSS`, and `METRICS`, the six (`poisson_deviance`, `log_loss`,
     `mae`, `rmse`, `wmape`, `bias`; the primary first) for the score tables and the bootstrap;
     `COHORTS`, `PREDICTION_FLOOR`;
   - **the `tuner` cell** (in section 5): `CrossValidationTuner(cv=cv,
     objective_metric=POISSON_DEVIANCE, n_trials=n_trials, n_startup_trials=n_startup_trials,
     n_jobs=n_splits, recorded_metrics=METRICS)`: every trial also records its cross-validated
     score on the six metrics (M12);
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
   - **the refit and the scores** (Model 1's loop and Model 2's pairs): the best model fitted on
     the training part, each part predicted, and `scoring.evaluate(<part>.Y, y_pred, METRICS,
     cohorts=[*COHORTS, "composite"])`: a cohort × metric DataFrame per part (D30);
   - **the tables:** `model1_df` and `model2_df`, long (one row per run × cohort × metric, with
     `train` and `val`; `cv_best` and Model 1's trial counts and seconds on the primary metric's
     composite row), shown with `mo.ui.table` (marimo's interactive table: sort, filter, pages);
     `model2_steps_df` (one row per step study: `step`, `family`, `use_exposure`, `cv_best`,
     `n_complete`, `n_failed`, `seconds`); `model1_arms_df` and `model2_arms_df` (trials per
     categorical choice; Model 2's counted on the 10 step studies); `best_df` (each model's best
     run on validation, by the primary metric's composite row);
   - **behind "Inspect the best runs":** each model's best run rebuilt from its `best_params` and
     refit on training; the bootstrap of the six metrics per cohort and for the composite, on
     training and validation (`intervals_df`, 96 rows: model, split, cohort, metric, `mean`,
     `lower`, `upper`, shown with `mo.ui.table`); the permutation importance and exact SHAP
     (`importance_df`).

**The run records** (the input of the MLflow runs):
- **Model 1** (`MODEL1_RUNS`): the run settings (`family`, `use_exposure`, the two sharing
  flags), `trials` (`study.trials_dataframe()`), `components` (each best component's
  `to_dict()`), `cv_best`, `best_params`, `train_scores`, `val_scores`, `seconds` (the study's),
  `seed`, `y_min`.
- **Model 2's step runs** (`MODEL2_TOTAL_RUNS`, `MODEL2_PROBABILITY_RUNS`): their settings,
  `trials`, `component` (the best), `fixed_component` (the other part, as the study held it),
  `cv_best`, `best_params`, `seconds`, `seed`, `y_min`.
- **Model 2's pairs** (`MODEL2_RUNS`): `total_family`, `probability_family`, `use_exposure`,
  `components`, `cv_best` (the pair's `cv_score`), `best_params` (the two runs' merged),
  `train_scores`, `val_scores`, `seed`, `y_min`. No `trials` or `seconds`: the step runs hold
  them, and `seconds` is a study's time in both models.

**The packages it uses**, each with its guide:
- [`hyperparameter_tuning`](HYPERPARAMETER_TUNING.md): records, search spaces, builders (the
  joint suggest functions and `suggest_count_component` / `suggest_probability_component`), the
  tuner (a tune method per model, and `tune_total_component` / `tune_probability_component`),
  aggregation. No `fit_and_score` (D30).
- `scoring`: the six metric functions, `Metric`, and `evaluate(y_true, y_pred, metrics,
  cohorts="composite")`.
- [`inspection`](MODEL_INSPECTION.md): `BootstrapEvaluator.evaluate(y_true, y_pred, groups=None,
  *, metrics, cohorts="composite")` (the caller predicts; the interval alone),
  `PermutationImportance`, `ShapExplainer`.
- `modeling` ([Model 1](INDEPENDENT_COHORT_MODELS.md), [Model 2](TOTAL_TIMES_PROBABILITY_MODEL.md)):
  fit and predict; no `evaluate` (D30).
- [`splitting`](SPLITTING.md), `feature_engineering` ([FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md)).
  MLflow is not used yet; the old `tracking/` package belongs to the old stack.

## 2. State (2026-10-31, end of PR 2)

- **Branches:** `main` at **`16773a5`** (`refactor(notebook): the runs in loops scored on every
  metric; Model 2 by coordinate descent (#20)`), PR 2 squashed; its content is identical to the
  branch's last commit `eb0da75`. **`feat/objective-eda`** (PR 3) was created from it; its first
  commit is this handoff's update. `refactor/notebook-run-loop` is deleted, locally and on
  GitHub. The user commits, pushes and merges.
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

## 2b. PR 3 so far (2026-11-01)

- **Commits on `feat/objective-eda`** (each sub-task's Done note is in the metrics plan §5):

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
  | (this handoff) | the handoff for 3.1h–3.1j |

- **The working tree is not clean, on purpose:** `notebooks/02_experiments.py` holds 3.1g
  (matplotlib plots, slices and a rank table; superseded by M13, never to be committed: its plot
  cell, the `Figure` import, a sentence in the EDA text) and the user's editor saves (marimo's
  stamp 0.25.2, one empty cell removed, two cells displaying `eda_model1_trials_df` and
  `eda_model2_trials_df`, the table cell's parameters reflowed). Its md5 at this handoff:
  `812fdb31b46b8ebb7ddf9d279688f981`. 3.1i removes 3.1g and keeps the user's edits.
- **Checks at 3.1g** (3.1e's suite and mypy still hold: `src` unchanged since): the non-slow
  suite 1387 passed, 1 skipped, 1 xfailed, 19 deselected; mypy no issues (28 files); Check A: the
  notebook test 13 passed, **ruff 16** (14 + 2 × PLR1711 on the EDA's bare-`return` text and
  table cells, which marimo writes), the two format hunks, `marimo check` 1.
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
  `n_startup_trials = 20`; never decrease `n_trials`; all runs kept.
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
  after seeing the diff.
- **Format new cells and lines to `ruff format`'s output; never format the whole file** (the
  user's two hunks). Write new cells in the scratchpad, format each on its own, splice them in;
  for an edited line, apply exactly the hunk `ruff format --diff` prints.
- No new ruff finding. A cell lists every notebook name it uses as a parameter, sorted.

**The checks:**
- **Check A:** the notebook test, ruff (16 since 3.1b: 14 + the EDA's two bare `return`s; none
  new), the format diff (the two hunks), `marimo check` (one empty cell).
- **Check B:** the names a script run defines (`app.run()`'s defs); nothing fitted.
- **Check C:** the clicked run on a copy. Build `<scratch>/project/notebooks/02_experiments.py`
  from the notebook with the line `    n_trials = 400` replaced by `    n_trials = 3` (by line),
  and each of the 8 lines `            "share_hyperparams": True,` (in `MODEL1_RUN_SETTINGS`)
  replaced by `False` (assert both counts); make `<scratch>/project/configs` a symlink to the
  repo's `configs`; run it with `marimo.ui.run_button` patched to a clicked value,
  `PYTHONWARNINGS=error`, `PYTHONPATH=<repo>/src`, in the background. About 75–180 s with the
  inspection and the EDA; the 24 `cv_best` must equal §2's. Since 3.1c it also asserts the EDA:
  the summaries' 12 rows and §2b's `cv_best`, and Model 2's EDA recomputed from the two tune
  methods (`dataclasses.replace(tuner, objective_metric=...)`) must equal its table exactly;
  since 3.1f the trials tables (rows, columns, each study's objective column's minimum its
  `cv_best`). The script lives in the scratchpad and must be rewritten in a new session (below).
- **A reference-run comparison** (used since 2.4): before a change, run Check C on HEAD and
  pickle `model1_df`, `model2_df`, `best_df`, `intervals_df`, `importance_df` and the records
  without `trials`; after it, compare frame by frame. Only Model 1's `seconds` (a study's wall
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
- **The scratchpad can vanish:** `/private/tmp` was emptied when the machine restarted
  (2026-11-01), taking the check scripts. Keep every number in the docs; rewrite the scripts.
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

**The user's style:** loops over run settings, not one cell per run (D28); no function unless
it is reused; names that say what a thing holds (the composite is "the composite", a cohort a
cohort); a record holds the object and writes its declaration; one source of truth for a list of
candidates; no field or check that nothing reads; scikit-learn's terms; `if`/`elif` with a
raising `else`; code that may move to a package reads no notebook global; settings in the
constructor, data in the call; fitting and evaluating as separate steps; no private helpers.

## 5. Open items

- **The review's #2:** `BootstrapEvaluator.evaluate` does not check `cohorts` (duplicates give
  identical rows, `[]` a confusing `TypeError`); left as is by the user.
- **T2.3's F5:** the two tables cells count arms with different idioms; left as is.
- **T1's F7:** `num_leaves` above `2**max_depth` changes nothing in LightGBM; revisit in a
  search-space pass.
- **The user's:** two empty cells and the bare `return`s marimo writes; ask before changing.
- **Stale text in section 4:** its header says "the check below prints it" and "fitted once …
  as a check", but the check cell is gone. The user's text; ask.
- **SPLITTING.md §3** and the simulator plan's Stop 4 doc examples (`.run()` → `simulate()`), and
  the simulator plan's cleanup stop (dead code) and format-only commit.
- **A `log(n_apartments)` plan for the GLMs** (FEATURE_TRANSFORMATIONS.md §4.4); not asked for.
- **What would shorten a click further** (about 26 min now), not asked for: dropping the GLM
  exposure-off arms (structurally worse by about 0.6); narrowing LightGBM's search after its
  fANOVA importances; fewer trials once the space is narrower.

## 6. Next, and the prompt

**Status (2026-11-01):** PR 3's EDA is built (3.1a–3.1f) and read once (3.2: the Poisson deviance
stays the objective). The user's notes asked for every trial, then for the trials as rows of the
whole model, a summary and plotly figures (M12, M13). Next, one sub-task per approval, as the
metrics plan §5 details them:
1. **3.1h plotly** in the `notebook` dependency group (pyproject.toml, uv.lock; the same version
   into `~/.venvs/age-group-prediction`; the user runs `uv sync --all-groups`). The user commits
   only those two files.
2. **3.1i every trial as a row of the whole model:** 3.1g's leftovers removed (the user's edits
   kept, §2b); `trials_table(study, objective, metric_names, components_of,
   feature_transformer_dict, first_id=0)`; Model 2's ids running across its two steps.
3. **3.1j the summary and the plotly figures:** `eda_best_df` (6 rows); per model three figures
   side by side, six stacked panels each, hover, the best trial a red star with a legend entry,
   Model 2's step line.
4. **3.2b the 400-trial click again**, and the trials read into the plan's 3.2.
5. **3.3:** the suite and the PR's independent review subagent (findings reproduced before any
   fix); the handoff rewritten for the deferred work (also README.md's and SPLITTING.md's "section
   5", now section 6 for the tuning); in the chat, the PR's title and description, the
   squash-merge title and description, and the commands that delete the remote and local
   branch.
6. **Deferred:** the MLflow runs (the D1e guard first: with `MLFLOW_TRACKING_URI` unset, MLflow
   3.16 writes to the repo's canonical `mlflow.db` and its artifacts to `./mlruns`), D4's
   comparison table, and E11's final evaluation.

The user's preferences: plan docs in `docs/`, split into sub-tasks validated one at a time;
each sub-task planned first (the facts re-verified with read-only probes, design questions with
evidence and a recommendation, the steps and the commit message in the chat), then implemented
after approval and stopped with a file-by-file summary, the checks, the notebook's md5 and the
commit message; explain marimo's UI terms; keep the implementation simple; meaningful names
(no `name` / `scores` placeholders); the user commits.

**The prompt for the next session:**

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full (§2b, §4's checks, traps and iCloud notes, and
§6 first), then docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md (§2 M10–M13, §5 3.1e–3.3), then section 5
of notebooks/02_experiments.py and
src/age_group_prediction/hyperparameter_tuning/cross_validation.py. Verify the state first:
- the branch feat/objective-eda, HEAD my commit of this handoff on top of 4afb834; if the
  handoff is uncommitted, stop and tell me;
- the only uncommitted file is notebooks/02_experiments.py (3.1g's leftovers and my editor
  saves, §2b), md5 812fdb31b46b8ebb7ddf9d279688f981 via `cat notebooks/02_experiments.py | md5`;
- no iCloud-offloaded files: `find .git src notebooks tests docs configs -flags +dataless -type f
  | grep -vc __pycache__` is 0.
If anything differs, stop and show me the diff. The scratchpad from the last session is gone:
rewrite the check scripts (§4) as needed. Then plan 3.1h in plan mode (re-verify the facts with
read-only probes, write the steps and the commit message in the chat) and wait for my approval
before editing. Then 3.1i, 3.1j, 3.2b and 3.3 in order, one per approval, each planned before
it is implemented. Keep the implementation simple. I commit myself: never commit, no
Co-Authored-By line, no "Generated with Claude Code" in PR text, never git checkout/stash/reset/mv.
```
