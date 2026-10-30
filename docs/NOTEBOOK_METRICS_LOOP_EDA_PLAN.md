# Plan: the metrics in `scoring`, the runs in a loop, the objective EDA

**Status (2026-10-30):** written; awaiting the user's review. No sub-task started.

The user's notes of 2026-10-30 on the experiment notebook. They replace the order of
[EXPERIMENT_NOTEBOOK_HANDOFF.md](EXPERIMENT_NOTEBOOK_HANDOFF.md) §6: MLflow and E11's final
evaluation wait for a later stage (§6 below).

1. MAE, RMSE, WMAPE and bias move into `scoring`, beside the log loss and the Poisson deviance.
   D² Poisson is dropped. Every metric works on one cohort and on all cohorts.
2. `share_hyperparams` and `share_feature_transformer` are run settings, not experiment settings.
3. Check that the two flags work.
4. An EDA on one run per model: the same study minimizing three objectives (cross-entropy, mean
   Poisson deviance, RMSE), compared on every metric.
5. All the runs in a for loop, each evaluated on every metric.

## Contents

1. How to work
2. Decisions
3. PR 1: the metrics in `scoring`
4. PR 2: the runs in a loop, scored on every metric
5. PR 3: the objective EDA
6. Deferred
7. Facts verified on 2026-10-30

## 1. How to work

The standing rules are those of [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3 and
[NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §1, with Checks A–C as the handoff §4 states them.

- **PRs:** three, in order. Each branches from `feat/experiment-notebook` after the previous one
  is merged into it, as PR #17 did. The user creates the branches, commits, pushes, opens the draft
  PRs and merges.
- **Sub-tasks:** one per approval. Each starts in plan mode (re-verify the facts, a short summary
  in the chat, AskUserQuestion for a design question) and ends with:
  - its "Verify" checks;
  - one mutation check per claimed behaviour;
  - a file-by-file summary, the notebook's new md5, and the commit message (no `Co-Authored-By`).
- **Each sub-task leaves the branch green:** the non-slow suite, mypy and Check A pass after every
  sub-task, so each commit can be validated on its own.
- **Before the last sub-task of each PR:** an independent review subagent (6–8 minutes); its
  findings are reproduced before they are fixed.
- **Before each edit:** `git status` and the notebook's md5; show any diff this plan does not name,
  and ask.

## 2. Decisions

- **M1. One rule for every metric.** `y_true` and `y_pred` have the same shape: `(n,)` (one cohort,
  or the total) or `(n, K)` (the cohort matrix). A shape mismatch raises. The metrics are computed
  over all cells pooled:
  - MAE: `mean|e|`;
  - RMSE: `sqrt(mean e²)`;
  - WMAPE: `Σ|e| / Σy`;
  - bias: `mean(ŷ − y)`, signed;
  - Poisson deviance: the mean over the cells.

  Pooled is what the notebook's `on_cells` computes today, so the numbers do not change. Pooled
  RMSE differs from the mean of the per-cohort RMSEs; MAE and the deviance are the same either way.
- **M2. D² Poisson is dropped.** On one evaluation set it is `1 − D / D_null`, where `D_null`
  depends on `y` alone: a monotone transform of the deviance, so it ranks models the same way.
- **M3. The log loss of one cohort is one-vs-rest** (the user's proposal). Cohort k is scored
  against the rest of its row: counts `[n_k, N − n_k]`, predictions `[ŷ_k, Σŷ − ŷ_k]`.
  - The multinomial log loss on that two-column pair equals the binary cross-entropy
    `−Σ[n_k log p_k + (N − n_k) log(1 − p_k)] / ΣN`. It also equals the binomial negative
    log-likelihood of the cohort's marginal, minus its constant (probed to 6 decimals, §7). So it
    is a proper score of cohort k's share, and needs no new formula.
  - It is not a part of the multinomial loss: the three cohorts' losses summed to 1.696 against
    0.956. Cohort rows compare with each other, not with the composite row.
  - Like the composite loss, it ignores the total.
  - A single column always scores 0, so `multinomial_log_loss` raises on 1-D or one-column input.
- **M4. Names** follow the user's `_loss` pattern: `multinomial_log_loss` and
  `mean_poisson_deviance_loss` (the user's), plus `mean_absolute_error_loss`,
  `root_mean_squared_error_loss`, `wmape_loss` and `mean_bias_error`. The user may rename them at
  1.1. `Metric` is not changed.
- **M5. The loop is built before the EDA (recommended).** The user's order (EDA first, then all
  runs in a loop) is the order of the clicks; it does not have to be the order of implementation.
  - Built after the EDA, the 24 run cells would be edited three times before being replaced: the
    sharing flags, the metric lists, then the loop.
  - Built first, those changes touch two loop cells. The EDA then reuses the same calls, and the
    full click still comes after the EDA.
  - If the user wants the EDA first, PR 3 moves before PR 2, and PR 3 edits the 24 cells for the
    sharing flags.
- **M6. The sharing flags live in each run's settings** (the user's note). They leave
  `EXPERIMENT_SETTINGS`; each entry of `MODEL1_RUN_SETTINGS` holds them, and the record keeps them,
  as it does today.
- **M7. The loop supersedes D13/D14's one cell per run** (the user's note). The seeds stay
  `seed + i` in the list's order, which is today's order of cells, so Check C's baselines still
  hold. Trade-off: an error other than `RuntimeError` (a bug, D7) ends the loop, and that model's
  records already made in the cell are lost.
- **M8. The EDA compares objectives on paired searches.** The three studies of a model share their
  `study_seed`, so the 20 random startup trials are the same draws.

## 3. PR 1: the metrics in `scoring`

Branch `refactor/scoring-metrics`, created from `feat/experiment-notebook` with the user's
uncommitted `scoring.py` and notebook edits, which carry over to the new branch.

### [ ] 1.1 The metric functions

- **Change:** `src/age_group_prediction/scoring.py`, starting from the user's edit:
  - `__all__` and the imports (ruff's F822 ×2, F401 `xlogy`, I001);
  - `np.asarray` before the shape check, so list inputs work;
  - `labels=range(K)` in `log_loss`, so a cohort with no child in `y_true` scores instead of
    raising;
  - the four new functions (M1, M4); the log loss raises on one column (M3).
- **Tests** (`tests/unit/test_scoring.py`):
  - each metric on 1-D and on 2-D equals a hand-computed value;
  - 2-D equals the same metric on the raveled cells;
  - a mismatched shape raises (non-square data);
  - lists work;
  - an absent cohort scores a finite value;
  - the log loss rejects one column;
  - the two-column pair equals the binary formula (M3).
- **Done when:** the tests pass under `-W error`; mypy and ruff are clean on `scoring.py`.
- **Verify, mutations:** drop `labels=`, and the absent-cohort test fails; drop the `ravel` in RMSE,
  and the 2-D test fails; drop the one-column check, and its test fails.
- **Commit:** `refactor(scoring): every metric on one cohort or the cohort matrix; MAE, RMSE, WMAPE and bias added`

### [ ] 1.2 The callers of the renamed functions

- **Change:** the old names give way to the new ones in:
  - `tests/unit/test_inspection_bootstrap.py` and `test_modeling_cohort_probability.py`;
  - the doc blocks of MODEL_INSPECTION.md and HYPERPARAMETER_TUNING.md (the HYPERPARAMETER_TUNING
    blocks are rerun as written, and their printed numbers must not change);
  - the notebook's import and metrics cells: the two names only.
- **Done when:**
  - the non-slow suite is green;
  - `grep` finds the old names only in finished plans' history;
  - Check A and Check B pass.
- **Verify:** Check C at 3 trials with the inspection gives the 24 baselines. Its `intervals_df` is
  saved to the scratchpad as the reference for 1.3.
- **Commit:** `refactor(scoring): the callers take multinomial_log_loss and mean_poisson_deviance_loss`

### [ ] 1.3 The notebook's metrics from `scoring`

- **Change:**
  - the import cell takes the new functions and drops `d2_tweedie_score`, `partial` and sklearn's
    MAE and RMSE;
  - the metrics cell drops `on_cells`, `wmape`, `bias` and D², so `INSPECTION_METRICS` has six;
  - section 5's markdown no longer names D². That is the user's text: show the diff and ask.
- **Done when:** Check A passes, against the 14 ruff findings and the two format hunks; Check B
  passes.
- **Verify:** Check C. The six metrics of `intervals_df` equal 1.2's reference exactly (M1:
  the same computation).
- **Before the stop:** the PR's review subagent.
- **Commit:** `refactor(notebook): the inspection metrics from scoring; D² Poisson dropped`

## 4. PR 2: the runs in a loop, scored on every metric

Branch `refactor/notebook-run-loop`, from `feat/experiment-notebook` after PR 1 is merged.

### [ ] 2.1 A test that a shared choice reaches every fitted cohort model

- **Change:** `tests/unit/test_hyperparameter_tuning_builders.py` gets one test, parametrized over
  the four flag pairs. It tests the wiring, not chance differences: independent draws can coincide.
  - Take a trial's params, change one entry, read the components back with `from_params`, build
    the model and fit it.
  - With a flag off, changing `n_kindergarten__max_depth` (or `__feature_transformer`) changes only
    kindergarten's fitted estimator (or its columns).
  - With a flag on, changing `shared__…` changes all three cohorts, and no `<cohort>__` key exists
    for that choice.
- **Done when:** the test passes under `-W error`.
- **Verify, mutation:** in `from_params`, read the cohort's key before `shared__`, and the test fails.
- **Commit:** `test(tuning): a shared choice reaches every fitted cohort model, an unshared one only its own`

### [ ] 2.2 Model 1's runs in a loop; the sharing flags as run settings

- **Change:**
  - `MODEL1_RUN_SETTINGS`: 8 dicts, each with `family`, `use_exposure`, `share_hyperparams` and
    `share_feature_transformer` (M6), in today's order of cells;
  - Model 1's 8 run cells become one cell behind the existing button, looping with `study_seed=seed + i`
    and building `MODEL1_RUNS`;
  - the tables cell takes the list instead of 8 names;
  - the two flags leave `EXPERIMENT_SETTINGS`, the settings cell and the settings test.
- **Done when:**
  - Check A passes; the run-cell rule holds (the loop cell holds `mo.stop(not `);
  - Check B passes, for `MODEL1_RUNS`, `model1_df` and `model1_arms_df`.
- **Verify:** Check C. Model 1's 8 values equal the handoff §2 baselines for `share_hyperparams=True`
  (`lgbm_poisson` 2.1812 / 2.2193, …), and a copy with both flags `False` gives the first table.
- **Commit:** `refactor(notebook): the independent cohorts model's runs in one loop; the sharing flags are run settings`

### [ ] 2.3 Model 2's runs in a loop

- **Change:** `MODEL2_RUN_SETTINGS` (16 dicts); Model 2's 16 run cells become one loop cell; the
  tables cell takes the list.
- **Done when:** Check A and Check B pass.
- **Verify:** Check C. Model 2's 16 values equal the baselines.
- **Commit:** `refactor(notebook): the total times probability model's runs in one loop`

### [ ] 2.4 `fit_and_score` with several metrics

- **Change:**
  - `fit_and_score(model, train, val, *, cohort_metrics, metrics)`: one fit, then one row per
    (cohort, metric) and per (composite, metric), each `{cohort, metric, train, val}`; with tests;
  - the two loop cells pass every metric;
  - the tables cells, `best_df` and the inspection's best-run rule read the primary metric's
    composite row, not `score_table[-1]`.
- **Design question, asked at planning:** how a cohort row gets the log loss's one-vs-rest pair
  (M3) while the other metrics get the cohort's column:
  - (a) a third list, `cohort_pair_metrics`, scored on each cohort's `[cohort, rest]` pair;
  - (b) the per-cohort log loss left out of `fit_and_score` and computed in the notebook's table;
  - (c) a field on `Metric` (this changes `Metric`, which the user kept as is).
- **Done when:** the suite passes; mypy passes, including the package path and the changed tests;
  Check A and Check B pass.
- **Verify:** Check C. The primary metric's 24 values are unchanged, and every other metric
  appears in the tables.
- **Commit:** `feat(tuning): fit_and_score scores every metric from one fit`

### [ ] 2.5 Docs

- **Change:**
  - NOTEBOOK_TUNING_PLAN.md §0: the new decisions (M6, M7) supersede D13/D14;
  - HYPERPARAMETER_TUNING.md §8 (`fit_and_score`'s new signature; the block rerun);
  - the handoff's §1–§2 and its checks;
  - this plan's status.
- **Done when:** the doc blocks run as written and the links resolve.
- **Before the stop:** the PR's review subagent.
- **Commit:** `docs(notebook): the run loop and fit_and_score's metrics`

## 5. PR 3: the objective EDA

Branch `feat/objective-eda`, from `feat/experiment-notebook` after PR 2 is merged.

### [ ] 3.1 The EDA cells

- **Change:**
  - a button, "Run the objective EDA" (a button is the clickable control that runs the cells
    behind it);
  - `EDA_OBJECTIVES = (log loss, Poisson deviance, RMSE)`, each a `Metric` on the cohort matrix;
  - one cell per model, each setting its run in the cell (Model 1 `lgbm_poisson`, exposure on,
    the sharing flags; Model 2 `lgbm_gaussian+lgbm`, exposure on: D20's best runs);
  - each cell loops over the objectives with `dataclasses.replace(tuner, metric=objective)` and the
    same `study_seed` (M8), refits the best and scores it on every metric;
  - a table per model: objective × metric on validation (the composite and each cohort), plus
    `cv_best` in its own metric and the seconds;
  - markdown with the expected caveat: the cross-entropy normalizes the rows, so Model 2's total
    gets no signal (its hyperparameters are drawn at random) and Model 1's scale is free.
- **Done when:**
  - Check A passes; the run-cell rule covers the new cells (`_tuner.tune_` contains `tuner.tune_`);
  - Check B passes.
- **Verify:** a 3-trial click of the EDA on a copy: 6 studies, and each table has 3 rows per model.
- **Commit:** `feat(notebook): the objective EDA: one run per model tuned on cross-entropy, Poisson deviance and RMSE`

### [ ] 3.2 The EDA click and its reading

- **Run:** the full click at 400 trials, about 3 × (98 + 247) s ≈ 17 min, in the background with
  `PYTHONWARNINGS=error`, on a copy (or the user clicks it in marimo).
- **Done when:** the two tables are in this plan, with what they show. The user then chooses the
  tuner's objective for the full click.
- **Commit:** `docs(notebook): the objective EDA's results`

### [ ] 3.3 The chosen objective, and the docs

- **Change:** the tuner's metric is set to the user's choice (if it is not the Poisson deviance);
  the handoff's §6 is rewritten for MLflow and E11.
- **Before the stop:** the PR's review subagent.
- **Commit:** `feat(notebook): the tuner's objective from the EDA`

## 6. Deferred

- **MLflow:** the D1e guard, the runs from the records, and D4's table.
- **E11:** the final evaluation.

Facts kept for them:
- with `MLFLOW_TRACKING_URI` unset, MLflow 3.16 writes to the repo's canonical `mlflow.db`, and its
  artifacts go to `./mlruns`;
- `notebook_defs` runs `app.run()` in the pytest process;
- `app.run()` re-raises a cell's error.

## 7. Facts verified on 2026-10-30

- **Git:** HEAD `8cb13b0`; the notebook's md5 is `f4569dcfe7c2d7815bcf3ba2d00fb8b6`.
- **The user's edits:** they removed an empty cell and dropped `optuna` from a return. They also
  renamed `cohort_log_loss` to `multinomial_log_loss` (via sklearn's `log_loss`) and
  `cohort_poisson_deviance` to `mean_poisson_deviance_loss`.
- **Callers of the old names:** the notebook, three test files and two guides.
- **Notebook baselines:** ruff finds 14 (the same findings, on shifted lines); `ruff format --diff`
  shows the user's two hunks; `marimo check` finds 1 empty cell.
- **The sharing flags, end to end** (draw, build, fit; 3 cohorts, 2 candidate transformers):

  | share_hyperparams | share_feature_transformer | params drawn | fitted estimator param sets | fitted column sets |
  |---|---|---|---|---|
  | False | False | 33 | 3 | 2 |
  | True | False | 13 | 1 | 2 |
  | True | True | 11 | 1 | 1 |
  | False | True | 31 | 3 | 1 |

  With both flags on, a 3-trial study's `best_params` holds only `shared__` keys.
- **M3, on 200 simulated rows:** cohort 0's two-column multinomial log loss, its binary formula and
  its binomial NLL minus the constant are each 0.451099 (cohort 1: 0.662504; cohort 2: 0.581915).
  The three sum to 1.695517; the multinomial loss is 0.956140.
- **Costs at 400 trials** (D20): Model 1 `lgbm_poisson`, exposure on: 98 s; Model 2
  `lgbm_gaussian+lgbm`, exposure on: 247 s.
- **Any metric can be the objective:** the tuner scores each fold with `self.metric` on the 2-D
  `(Y_val, Y_pred)` and takes the study's direction from it.
