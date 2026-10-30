# Plan: the metrics in `scoring`, the runs in a loop, the objective EDA

**Status (2026-10-30):** PR 1 on `refactor/scoring-metrics`: 1.1 done, revised at the user's notes (`cohort=`, `y_proba=`, samples; then the notes on the helpers: `utils.check_paired_arrays` checks, `utils.column_position` finds a cohort's column, each metric selects it, no repeated-name guard; suite 1346 passed; Check C the 24 baselines), committed as `185285e`; next 1.2.

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

- **M1. One rule for every metric.** `y_true` and `y_pred` have the same shape: `(n_samples,)` (a
  single target, such as the total) or `(n_samples, n_cohorts)` (the cohort matrix). A shape
  mismatch raises. "Samples", scikit-learn's term, not "buildings": the granularity of a row may
  change (the user's note). The metrics are computed over all cells pooled:
  - MAE: `mean|e|`;
  - RMSE: `sqrt(mean e²)`;
  - WMAPE: `Σ|e| / Σy`;
  - bias: `mean(ŷ − y)`, signed;
  - Poisson deviance: the mean over the cells.

  Pooled is what the notebook's `on_cells` computes today, so the numbers do not change. Pooled
  RMSE differs from the mean of the per-cohort RMSEs; MAE and the deviance are the same either way.
- **M2. D² Poisson is dropped.** On one evaluation set it is `1 − D / D_null`, where `D_null`
  depends on `y` alone: a monotone transform of the deviance, so it ranks models the same way.
- **M3. The log loss of one cohort is one-vs-rest** (the user's proposal). With `cohort=`, cohort k
  is scored against the sum of the others in its sample: counts `[n_k, N − n_k]`, predictions
  `[ŷ_k, Σŷ − ŷ_k]`.
  - The multinomial log loss on that two-column pair equals the binary cross-entropy
    `−Σ[n_k log p_k + (N − n_k) log(1 − p_k)] / ΣN`. It also equals the binomial negative
    log-likelihood of the cohort's marginal, minus its constant (probed to 6 decimals, §7). So it
    is a proper score of cohort k's share, and needs no new formula.
  - It is not a part of the multinomial loss: the three cohorts' losses summed to 1.696 against
    0.956. Cohort rows compare with each other, not with the composite row.
  - Like the composite loss, it ignores the total.
  - A single column always scores 0, so `multinomial_log_loss` raises on 1-D or one-column input.
  - scikit-learn's `log_loss` is called with keywords, `y_true=` and `y_proba=` (the user's note:
    1.9 deprecated `y_pred` for `y_proba`, removed in 1.11; 1.9.0 is pinned, no upgrade needed).
- **M4. Names** follow the user's `_loss` pattern: `multinomial_log_loss` and
  `mean_poisson_deviance_loss` (the user's), plus `mean_absolute_error_loss`,
  `root_mean_squared_error_loss`, `wmape_loss` and `mean_bias_error`. Each is
  `(y_true, y_pred, *, cohort: str | None = None)` (M9).
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
- **M9. A metric scores one cohort through `cohort=`** (the user's design, 2026-10-30). Every metric
  takes the whole `y_true` / `y_pred` and an optional `cohort`, and restricts itself: a count
  metric pools that cohort's column; the log loss scores the `[cohort, rest]` pair (M3). One metric
  list then serves a score table's cohort rows and its composite row (2.4).
  - **By name, not by position** (the user's experiment names its cohorts). scikit-learn's column
    selectors do the same (`ColumnTransformer`: "strings can reference DataFrame columns by
    name"). The name is looked up in `y_true`'s columns and `y_pred` is read at that position, so
    pairing by position stands. An array `y_true` raises `TypeError`, a missing name pandas'
    `KeyError`. A repeated name is not checked: the cohort names are unique by construction
    (`Y = df[targets]`, the simulator's distinct names), and data is validated where it is made.
  - **`str`, not a `Literal`:** the names are data (the notebook's `COHORTS = list(Y.columns)`), and
    `scoring` serves any target matrix.
  - **`Metric.__call__` and `BaseAgeGroupModel.evaluate` take `cohort` too.** `Metric` passes it on
    only when given, so a two-argument function (scikit-learn's, a lambda) still scores the whole
    of `y` and raises `TypeError` if asked for a cohort.
  - The cohort rows do not add up to the composite: RMSE and WMAPE pool, and the log loss's rows
    are binomial.

## 3. PR 1: the metrics in `scoring`

Branch `refactor/scoring-metrics`, created from `feat/experiment-notebook` with the user's
uncommitted `scoring.py` edit, which carried over. The plan (`ff6087c`) and the user's notebook
cleanup (`492f171`: an empty cell and an unused `optuna` return) are its first commits.

### [x] 1.1 The metric functions, and the code that calls them

The user's choice at 1.1's stop (2026-10-30): the code callers are renamed in 1.1, so its commit is
green; 1.2 is the docs. The rename breaks every importer of the old names: two test modules fail
to collect, and the notebook fails to import.

- **Change:** `src/age_group_prediction/scoring.py`, starting from the user's edit:
  - `__all__` and the imports (ruff's F822 ×2, F401 `xlogy`, I001);
  - one shape check for every function (`check_paired_arrays`), with `np.asarray` first, so
    list inputs work. With `cohort=`, `column_position` finds the column and each metric selects
    it: a count metric takes the column, the log loss stacks `[cohort, rest]` (the user's notes:
    no `_cohort_cells`). Both helpers are public in `utils.py`, beside `take_rows` (the user's
    note: not private), as scikit-learn's public validators live in `sklearn.utils`. The check
    runs inside each metric function, as scikit-learn's metrics validate their own inputs: a
    check in `Metric.__call__` would miss the direct calls;
  - `log_loss(y_true=..., y_proba=..., labels=range(n_cohorts))`, so a cohort with no child in
    `y_true` scores instead of raising (M3);
  - the four new functions (M1, M4). The log loss raises on 1-D input itself; on one column,
    scikit-learn's `log_loss` raises ("at least two labels");
  - `cohort=` on every function, `Metric.__call__` and `BaseAgeGroupModel.evaluate`
    (`src/age_group_prediction/modeling/base.py`) (M9);
  - "samples" in the docstrings, messages and names (M1).
- **Behaviour changes, from the user's switch to scikit-learn's `log_loss`:**
  - a zero probability for a cohort that has children is clipped: a large finite loss (36.04 in
    the probe), where the old `xlogy` version gave `inf`;
  - no child at all raises scikit-learn's `ValueError`, where the old version gave `nan`.
  The bootstrap still fails loudly either way.
- **The code callers,** function references only. The `Metric` labels (`"cohort_log_loss"`,
  `"cohort_poisson_deviance"`) are unchanged, so every table reads the same:
  - `tests/unit/test_hyperparameter_tuning_cross_validation.py`,
    `test_hyperparameter_tuning_evaluation.py`, `test_inspection_bootstrap.py`,
    `test_inspection_permutation.py`, `test_modeling_cohort_probability.py`;
  - the notebook's import and metrics cells (the names sorted, as ruff's I001 and marimo expect).
- **Tests** (`tests/unit/test_scoring.py`):
  - each count metric on the cohort matrix pools every cell, and scores one cohort (1-D);
  - the cohort deviance is the mean of the cohort deviances; the cohort RMSE is not the mean of the
    cohort RMSEs;
  - a transposed, flattened or one-column prediction is rejected by every count metric;
  - DataFrames and lists pair by position;
  - an over-prediction has a positive bias;
  - the log loss: the mean per child; a sample without children adds nothing; a cohort without
    a child scores; counts score as proportions; `cohort=` gives the binomial log loss of the
    cohort's share (M3); one cohort alone is rejected;
  - `cohort=` (M9): a count metric scores that cohort's column, read at its position in `y_pred`;
    a name `y_true` does not hold, or an array `y_true`, is rejected by all six;
    `Metric` passes the cohort on only when given; `evaluate` passes it to the metric
    (`tests/unit/test_modeling_base.py`).
- **Done when:**
  - the tests pass under `-W error`; mypy and ruff are clean on the changed files;
  - the non-slow suite is green;
  - Check A passes (the notebook's 14 ruff findings, the two format hunks, `marimo check`'s one
    empty cell);
  - Check C gives the 24 baselines.
- **Verify, mutations:** drop `labels=`, and the absent-cohort test fails; drop the `ravel` in RMSE,
  and the pooling tests fail; drop the 2-D check, and the one-cohort test fails; the count
  metrics ignore `cohort`, the pair is built from another column, or `Metric` drops `cohort`, and
  their tests fail.
- **Commit:** `refactor(scoring): every metric on the sample matrix or one cohort by name; MAE, RMSE, WMAPE and bias added`

### [ ] 1.2 The docs of the renamed functions

- **Change:** the new names in MODEL_INSPECTION.md, HYPERPARAMETER_TUNING.md,
  TOTAL_TIMES_PROBABILITY_MODEL.md (code blocks and prose) and MODULE_REFERENCE.md's `scoring.py`
  row (its functions, the one-cohort-or-matrix rule, and its dependency on `utils`), and its
  `utils.py` row (`check_paired_arrays`, `column_position`). The blocks that import the renamed
  functions are rerun as written, and their printed numbers must not change.
- **Done when:** `grep` finds the old function names only in finished plans' history and as
  `Metric` labels; the rerun blocks print the numbers the docs show.
- **Verify:** 1.1's Check C `intervals_df` is the reference for 1.3 (saved in the scratchpad).
- **Commit:** `docs(scoring): the guides take multinomial_log_loss and mean_poisson_deviance_loss`

### [ ] 1.3 The notebook's metrics from `scoring`

- **Change:**
  - the import cell takes the new functions and drops `d2_tweedie_score`, `partial` and sklearn's
    MAE and RMSE;
  - the metrics cell drops `on_cells`, `wmape`, `bias` and D², so `INSPECTION_METRICS` has six;
  - section 5's markdown no longer names D². That is the user's text: show the diff and ask.
- **Done when:** Check A passes, against the 14 ruff findings and the two format hunks; Check B
  passes.
- **Verify:** Check C. The six metrics of `intervals_df` equal 1.1's reference exactly (M1:
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
  - `fit_and_score(model, train, val, *, metrics)`: one fit, then one row per (cohort, metric),
    scored with `cohort=<name>` (M9), and per (composite, metric), each `{cohort, metric, train,
    val}`; with tests. `POISSON_DEVIANCE` goes: `COHORT_POISSON_DEVIANCE` with a cohort is that
    cohort's deviance;
  - the two loop cells pass every metric;
  - the tables cells, `best_df` and the inspection's best-run rule read the primary metric's
    composite row, not `score_table[-1]`.
- **Design question, settled 2026-10-30 by M9:** every metric restricts itself through `cohort=`,
  so one list serves both kinds of row.
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
