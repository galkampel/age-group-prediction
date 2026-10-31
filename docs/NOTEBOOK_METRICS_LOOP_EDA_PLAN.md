# Plan: the metrics in `scoring`, the runs in a loop, the objective EDA

**Status (2026-10-30):** PR 1 on `refactor/scoring-metrics`: 1.1 done, revised at the user's notes (`cohort=`, `y_proba=`, samples; then the notes on the helpers: `utils.check_paired_arrays` checks, `utils.column_position` finds a cohort's column, each metric selects it, no repeated-name guard; suite 1346 passed; Check C the 24 baselines), committed as `185285e`. 1.2 done (docs only; the three guides' blocks rerun as written with the same numbers; MODEL_INSPECTION.md's stale simulator line fixed with `from_config`, the user's choice), committed as `96c950b`. 1.3 done (the notebook's six inspection metrics from `scoring`, D² dropped; Check C the 24 baselines and the intervals equal, WMAPE and bias but for their last bit); committed as `0e9d0a1`. 1.4 done (the review's fixes in `scoring`: non-finite values and a lone cohort rejected, two docstrings; suite 1373 passed); PR 1 merged as #18 (`b1d8dbd`) into `feat/experiment-notebook`, and that branch into `main` as #19 (`3db1aa8`). PR 2 on `refactor/notebook-run-loop` from `main`: 2.1 done (2026-10-31; the sharing flags' test, a fitted model per flag pair; the Verify mutation moved to the build, at the user's choice), committed as `9463651`. 2.2 done (2026-10-31; Model 1's runs in one loop over `MODEL1_RUN_SETTINGS`, the flags per run; Check C both arms' baselines), committed as `b87e68b`. 2.3a.1 done (2026-10-31; Model 2 by coordinate descent measured against D20's joint studies at 400 trials: cv better or the same in 15 of 16 pairs, the one worse within noise; about 890 s against 2871 s; docs only), committed as `4c75172`. 2.3a.2 done (2026-10-31; `tune_total_component` and `tune_probability_component`, the other component fixed; the joint tuning stays, its draw now through `suggest_count_component` and `suggest_probability_component`, at the user's notes; they reproduce 2.3a.1's step values at 400 trials; suite 1381 passed; Check C the 24 baselines), committed as `a5a4234`. 2.3 done (2026-10-31; Model 2 by coordinate descent in three cells: step 1's 8 runs, step 2's 2 runs with the best total fixed, the 16 pairs scored; the 400-trial click reproduces 2.3a.1's 16 pairs in 752 s; Check C Model 1's 8 unchanged and Model 2's 16 re-recorded), committed as `9aed504`, its record as `04caa68`. 2.4 done (2026-10-31, revised at the user's notes: fit, predict and evaluate apart; `scoring.evaluate(y_true, y_pred, metrics, cohorts)` a cohort × metric table, `fit_and_score` deleted; `POISSON_DEVIANCE`, `LOG_LOSS`, `METRICS`, no `COHORT_` prefixes; the long score tables in `mo.ui.table`; Check C the primary metric's rows equal `04caa68`'s exactly). committed as `c2843e3`. 2.4b done (2026-10-31; `BaseAgeGroupModel.evaluate` removed, the callers call the metric; `BootstrapEvaluator.evaluate(y_true, y_pred, groups, *, metrics, cohorts)` takes the predictions and the cohorts and returns the interval alone; the notebook bootstraps every cohort; Check C equal to `c2843e3`'s run, the composite intervals exactly). Next: 2.5.

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

- **PRs:** three, in order. PR 1 branched from `feat/experiment-notebook` and was merged into it
  (#18), and that branch into `main` (#19). PRs 2 and 3 each branch from `main` after the previous
  one is merged into it. The user creates the branches, commits, pushes, opens the draft
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

  Pooled is what the notebook's `on_cells` computed, so the numbers do not change (WMAPE and
  bias only in their last bit: they sum in memory order, `on_cells` in row order; 1.3). Pooled
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

### [x] 1.2 The docs of the renamed functions

- **Change:** the new names in MODEL_INSPECTION.md, HYPERPARAMETER_TUNING.md,
  TOTAL_TIMES_PROBABILITY_MODEL.md (code blocks and prose) and MODULE_REFERENCE.md's `scoring.py`
  row (its functions, the one-cohort-or-matrix rule, and its dependency on `utils`), and its
  `utils.py` row (`check_paired_arrays`, `column_position`). The blocks that import the renamed
  functions are rerun as written, and their printed numbers must not change.
- **Done when:** `grep` finds the old function names only in finished plans' history and as
  `Metric` labels; the rerun blocks print the numbers the docs show.
- **Verify:** 1.1's Check C `intervals_df` is the reference for 1.3 (saved in the scratchpad).
- **Commit:** `docs(scoring): the guides take multinomial_log_loss and mean_poisson_deviance_loss`
- **Done 2026-10-30.** The user's choices at the plan: MODEL_INSPECTION.md's stale
  `StudentPopulationSimulator(config).run()` (a `TypeError` today) becomes the simulator plan's
  Stop 4 form, `StudentPopulationSimulator.from_config(config).simulate().dataset`, and Stop 4
  notes it; `cohort=` is documented in prose and signatures, and the blocks keep scikit-learn's
  per-column scoring (a `cohort=` example arrives with 2.4–2.5's `fit_and_score`).
  - Changed: MODEL_INSPECTION.md (import, the two metrics, the simulator line, the run date,
    §6's hazard on which cells a metric scores), HYPERPARAMETER_TUNING.md (import, the metric,
    the run date), TOTAL_TIMES_PROBABILITY_MODEL.md (prose ×4, the block, `evaluate`'s
    signature, the rerun note), MODULE_REFERENCE.md (the `scoring.py`, `utils.py` and
    `modeling/base.py` rows), INDEPENDENT_COHORT_MODELS.md §4 (`evaluate`'s signature and
    `scoring`'s functions), SIMULATOR_WHITE_BOX_PLAN.md (Stop 4's note).
  - Rerun as written under `-W error` and `PYTHONWARNINGS=error`: MODEL_INSPECTION.md 5/5
    blocks and HYPERPARAMETER_TUNING.md 8/8 print the docs' output; TOTAL_TIMES's §6 block
    (with a prelude: the seed-0 population by `from_config(...).simulate(rng=default_rng(0))`,
    FEATURE_TRANSFORMATIONS §8.0–8.3's transformers) gives 196 / 49 buildings, 1.855 / 1.958 /
    2.250, total 3.587, composition 1.084, α 0.0876. Mutation: a copy with one old name
    restored fails with `ImportError`.
  - The old names remain only in finished plans, the handoffs and as `Metric` labels
    (`"cohort_log_loss"`, `"cohort_poisson_deviance"`, MODEL_INSPECTION's column label).

### [x] 1.3 The notebook's metrics from `scoring`

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
- **Done 2026-10-30.**
  - **A fact found at the plan:** WMAPE and bias cannot equal the reference exactly. On 3000
    bootstrap resamples of DataFrames, MAE and RMSE were equal bit for bit, WMAPE differed in
    1043 and bias in 2107, by at most 3.5e-16 relative / 1.1e-16 absolute. `np.asarray` of a
    DataFrame is column-major; the old `on_cells` ravelled to row order before summing, while
    `wmape_loss` and `mean_bias_error` sum the 2-D array in memory order. The user's choice:
    compare at a tolerance (rtol = atol = 1e-12) and leave `scoring` as it is.
  - **The user's choice on the texts:** section 5's markdown drops the D² clause and says six
    metrics; the inspection button's text and the bootstrap cell's comment say six.
  - **Changed:** the import cell (`partial`, `d2_tweedie_score`, `mean_absolute_error`,
    `root_mean_squared_error` out; the four `scoring` functions in), the metrics cell
    (`on_cells`, `wmape`, `bias` and D² out; `INSPECTION_METRICS` the six `scoring` metrics,
    labels unchanged), the three texts.
  - **Checks:** Check A: the notebook test 12 passed; ruff the 14; the format diff the two
    hunks (`@@ -158`, `@@ -208`); `marimo check` the one empty cell. Check B: the six metrics,
    each its `scoring` function; the dropped names gone. Check C (75 s): the 24 baselines;
    `intervals_df` 24 rows; deviance, log loss, MAE and RMSE equal the reference exactly, WMAPE
    and bias within 1e-12 (largest gap 1.1e-16). Mutation: `"wmape"` mapped to
    `mean_absolute_error_loss` fails the comparison (gap 3.1).

### [x] 1.4 The review's fixes in `scoring`

Added on 2026-10-30 at the user's choice, after PR 1's review subagent (run before 1.3's stop).
Its findings, each reproduced:
- `wmape_loss` and `mean_bias_error` passed a NaN or an infinity through (nan, ±inf), while the
  other four raised scikit-learn's `ValueError`;
- `multinomial_log_loss` on a single-column frame with `cohort=` scored ≈ 0 (2.2e-16) whatever
  the prediction (the rest column is all zeros), where the same call without `cohort=` raised;
- the module docstring said every function takes a single target (the log loss rejects one);
  `wmape_loss`'s docstring said `Σ y` where the code divides by `Σ|y|`;
- not changed: a `y_pred` frame with reordered columns is mispaired, which is the documented
  pairing by position (N19), pinned by the tests.

- **Change:**
  - `utils.check_paired_arrays` rejects a non-finite value in either array (`ValueError`), as
    scikit-learn's `check_array` does by default, so all six metrics behave alike;
  - `multinomial_log_loss` checks for a matrix of at least two cohorts before `cohort=`;
  - the module docstring and `wmape_loss`'s formula (`Σ|y|`); MODULE_REFERENCE.md's
    `scoring.py` and `utils.py` rows;
  - tests: all six metrics reject NaN and inf in either input (24 cases); the one-cohort test
    adds the `cohort=` case and matches the new message; `check_paired_arrays` rejects NaN,
    inf and −inf in either array.
- **Behaviour:** the tuner, the bootstrap and the permutation importance already got a
  `ValueError` from the primary metric on such input; only WMAPE and bias (nan / inf before)
  and the log loss (a `RuntimeWarning` first for an inf prediction) change.
- **Checks (2026-10-30):** the two test files and `test_modeling_base.py` 96 passed under
  `-W error`; the non-slow suite 1373 passed, 1 skipped, 1 xfailed, 19 deselected (1346 + 27);
  mypy no issues (29 files); ruff and the format check clean on the four files; Check A the
  notebook unchanged (ruff 14, the two hunks, `marimo check` 1). Mutations on a full copy, each
  caught: the finite check dropped (27 failed), `y_pred` only checked (15 failed), `< 2`
  dropped (1 failed).
- **Commit:** `fix(scoring): every metric rejects non-finite values, the log loss a lone cohort also with cohort=; two docstrings corrected`

## 4. PR 2: the runs in a loop, scored on every metric

Branch `refactor/notebook-run-loop`, from `main` after PR 1 (#18) and #19 are merged.

### [x] 2.1 A test that a shared choice reaches every fitted cohort model

- **Change:** `tests/unit/test_hyperparameter_tuning_builders.py` gets one test, parametrized over
  the four flag pairs. It tests the wiring, not chance differences: independent draws can coincide.
  - Take a trial's params, change one entry, read the components back with `from_params`, build
    the model and fit it.
  - With a flag off, changing `n_kindergarten__max_depth` (or `__feature_transformer`) changes only
    kindergarten's fitted estimator (or its columns).
  - With a flag on, changing `shared__…` changes all three cohorts, and no `<cohort>__` key exists
    for that choice.
- **Done when:** the test passes under `-W error`.
- **Verify, mutations:** in `build_independent_cohorts_model`, every cohort takes the first
  component's estimator, and the cases with `share_hyperparams` off fail; every cohort takes the
  first component's transformer, and the cases with `share_feature_transformer` off fail.
- **Commit:** `test(tuning): a shared choice reaches every fitted cohort model, an unshared one only its own`
- **Done 2026-10-31.**
  - **A fact found at the plan:** the first Verify mutation (in `from_params`, read the cohort's
    key before `shared__`) cannot fail. A trial never holds both keys for one choice: a shared
    choice is drawn only under `shared__`, an unshared one only under `<cohort>__` (probed: once
    the prefixes are stripped, the two sets of keys never overlap, for all four pairs). So
    `from_params` finds one value whichever it reads first. The user chose best practice: break
    the code the test is about, the build (above).
  - **Changed:** `tests/unit/test_hyperparameter_tuning_builders.py`: one test, two stacked
    `parametrize`s (the four pairs; `SHARING` is left alone, since its first test expects 33
    params whenever `share_hyperparams` is off). It changes `<prefix>__max_depth` and
    `<prefix>__feature_transformer` in a trial's params, with prefix `shared` when the flag is on
    and `n_kindergarten` when it is off. It reads both param sets back, builds and fits them,
    and compares each cohort's `estimator_.get_params()` and `estimator_.feature_names_in_`.
  - **Checks:** the file 17 passed under `-W error` (13 + 4); ruff, the format check and mypy
    clean on it; the non-slow suite 1377 passed, 1 skipped, 1 xfailed, 19 deselected (1373 + 4; the 6 MLflow
    warnings); mypy (29 files) no issues; the notebook
    unchanged. Mutations on a full copy, each caught: first estimator, 2 of the 4 cases
    (`share_hyperparams` off) and the existing build test; first transformer, 2 of the 4
    (`share_feature_transformer` off) and the existing build test.

### [x] 2.2 Model 1's runs in a loop; the sharing flags as run settings

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
- **Done 2026-10-31.**
  - **The user's choices at the plan:** `MODEL1_RUN_SETTINGS` in its own cell, before the
    button; section 5's text changed in the same commit. Three sentences change:
    - "each in its own cell" goes;
    - "Two run settings decide …";
    - "Each entry of `MODEL1_RUN_SETTINGS` sets both: its eight runs take `True` and `False`, and
      a run with other values is one more entry in that list" replaces "the other arm … is a
      click with the setting changed";
    - the seeds' sentence names the loop.
  - **Changed:**
    - the notebook:
      - the settings cell (the flags, their comment, their two `EXPERIMENT_SETTINGS` entries and
        return names out);
      - the new `MODEL1_RUN_SETTINGS` cell (8 entries in the old order of cells, each `True` /
        `False`; the D24.4 comment moved there);
      - the button's caption;
      - the 8 run cells replaced by one loop cell (`study_seed=seed + i`, the flags from each
        entry, records `{**run_settings, …}` with the old keys in the old order);
      - the tables cell (takes `MODEL1_RUNS`; the 8-name list out);
      - section 5's text.
    - The notebook test: the two flags out of `experiment_settings`.
  - **Check C's copy recipe from now on:** the line `    n_trials = 400` → `3` (one line), and
    each `            "share_hyperparams": True,` → `False` (8 lines, in `MODEL1_RUN_SETTINGS`).
    The handoff's recipe changes in 2.5.
  - **Checks:**
    - Check A: the notebook test 12 passed; ruff the same 14 findings, on shifted lines; the
      format diff the two hunks (`@@ -158`, `@@ -208`); `marimo check` the one empty cell.
    - Check B: `MODEL1_RUN_SETTINGS` defined (8 entries, the order above), the two flags not
      defined, nothing fitted.
    - Check C, two copies:
      - (a) the entries as committed (83 s): Model 1's 8 equal §2's `share_hyperparams=True`
        line, Model 2's 16 their table, the records' keys the old ones, the inspection ran;
      - (b) the 8 flags `False` (70 s): Model 1's 8 equal §2's first table, Model 2 unchanged.
        That (b) moves to the other baselines is the check that each entry's flags reach its
        study.
    - The non-slow suite: 1377 passed, 1 skipped, 1 xfailed, 19 deselected.

### [x] 2.3a Model 2 by coordinate descent

Decided by the user on 2026-10-31
([EXPERIMENT_NOTEBOOK_HANDOFF.md](EXPERIMENT_NOTEBOOK_HANDOFF.md) §2a). It was asked because
Model 2's joint search takes 48 of the click's 62 min at 400 trials. T2.4 / D19 had already
measured and rejected pruning and more parallelism (slower or unsafe), and the user declined
splitting into two objectives (a total deviance and a log loss).

- **The decision:** the notebook tunes Model 2 in two steps.
  1. Tune the total, with the probability model fixed.
  2. Tune the probability model, with step 1's best total (by cv) fixed.

  Then each of the 16 pairs is built from its two winners and scored, with no search. **The
  joint tuning stays** in `src`: `CrossValidationTuner.tune_total_times_probability_model`,
  `suggest_total_times_probability_components` and their tests. Nothing is removed.
- **Why one round is exact** (probed on the notebook's 5 folds, ≤ 3e-15 on each):
  - Model 2's composite Poisson deviance is `D_total / K + 2 (C + N · log_loss) / (n K)`, where
    `D_total` is the total's deviance, `log_loss` the probabilities' per-child multinomial log
    loss, and `C` depends on `Y` alone;
  - the prediction is `probabilities × total`, and `y_min` floors the total only; no
    hyperparameter reaches both parts;
  - so the joint optimum is the pair of the two separate optima, and the second step cannot
    change the first's result;
  - within a step, the fixed part adds the same constant to every trial's value. Step 1 may
    therefore fix the cheapest classifier: logistic, `C = 1.0`, no calibration, transformer
    `glm_base`;
  - each step scores the full composite, as today. Only what is searched changes.
- **The studies:**
  - step 1: 8 studies (4 total families × exposure on / off), seeded `seed + 0 … 7` in Model 1's
    order (`lgbm_poisson`, `lgbm_gaussian`, `poisson_glm`, `nb2`; exposure on, then off);
  - step 2: 2 studies (logistic, lgbm), seeded `seed + 8`, `seed + 9`. The exposure does not
    reach the classifier;
  - the 16 pairs: each one's `cv_best` is `tuner.cv_score` of the pair, comparable with a joint
    `best_value`, plus `fit_and_score`.

#### [x] 2.3a.1 The measurement

- **Done 2026-10-31,** docs only: a scratchpad probe built from the package's public pieces, 400
  trials, `PYTHONWARNINGS=error`. Today's joint `lgbm_gaussian+lgbm` (exposure on, `seed + 5`)
  reproduced D20's 2.0962 first. The comparison with D20's joint results:

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

  - **cv** (the objective): better in 9, the same in 6 (≤ 0.0005), worse in 1
    (`lgbm_poisson+logistic` on, +0.013, within the 0.02 noise); the mean −0.004. The two
    differences beyond the noise are improvements (both `lgbm_gaussian` exposure-off pairs,
    −0.024). The GLM pairs equal the joint results to 4 decimals, which shows the same objective.
  - **val** (one holdout): better in 7, the same in 6, worse in 3. One worse pair is beyond the
    noise: `lgbm_poisson+lgbm` off, +0.058, while its cv is −0.002, so this is holdout noise.
    The best pair on validation is `lgbm_gaussian+logistic`, exposure on, 2.0759 (D20's best
    was `lgbm_gaussian+lgbm`, 2.1202).
  - **Time:** 2871 s joint against about 890 s coordinate: the totals 422 s (36–93 s each), the
    classifiers 454 s (logistic 64, lgbm 390), the pairs 14 s. That is 3.2× faster. A rerun
    under a machine load of about 18 took 1137 s, with identical values.
  - **The winners:**
    - step 1's best total by cv: `lgbm_poisson`, exposure on, 2.0718;
    - step 2, logistic: 2.0622, with `C` 0.1012, `sigmoid`,
      `probability__room_share_x_median_age`;
    - step 2, lgbm: 2.0769, with `max_depth` 3, `n_estimators` 215, `learning_rate` 0.0109,
      `sigmoid`, `tree`.
  - Optuna's `PartialFixedSampler` raises `ExperimentalWarning` under `-W error`, so the fixing
    is done in the package's code (2.3a.2).
- **Commit:** `docs(notebook): Model 2 by coordinate descent measured against the joint studies`

#### [x] 2.3a.2 The tuner tunes one part of Model 2 with the other fixed

- **Change** (`src`, the smallest that works):
  - `suggest_count_component` and `suggest_probability_component` (`builders.py`) each draw one
    component under a given name; the joint draw calls both, the total first;
  - the tuner can tune one part with the other part's component given and fixed. **The joint
    method stays.**
- **Design question for the user, with code previews:** two named tune methods (one per step,
  D26's pattern; recommended) or one method with an optional fixed component.
- **Tests:** a step's trials hold only that part's keys; the fixed component is the one in the
  model that is scored; the joint method's tests still pass.
- **Done when:** the non-slow suite and mypy pass; Check A passes (the notebook unchanged).
- **Verify:** Check C gives the 24 baselines unchanged (the notebook still runs the joint
  studies); one mutation per claimed behaviour.
- **Commit:** `feat(tuning): tune the total times probability model's total or probabilities with the other component fixed`
- **Done 2026-10-31.**
  - **The user's choices at the plan:**
    - two named tune methods;
    - **the joint tuning is valid (correct, only slow) and stays:** the new code is added, not
      put in its place.
  - **The user's notes after the first implementation** (which had copied the joint function's
    two halves into new functions beside it):
    - the joint `suggest_total_times_probability_components` calls the two new functions, so
      each draw exists once. Its signature and its draws are unchanged (the builders test,
      Check C);
    - `suggest_total_component` becomes `suggest_count_component`, the name an argument. At the
      session's recommendation `suggest_probability_component` takes a name too: each mirrors
      its component's `from_params` (`name`, `family`, `use_exposure`,
      `feature_transformer_dict`). The names `"total"` and `"probability"` are given where the
      model is assembled: the joint function and the two tune methods;
    - not used in `suggest_independent_cohorts_components`: its loop interleaves the shared
      draws with each cohort's own, and `suggest_count_component` fits only with both flags
      off.
  - **Facts found at the plan** (read-only probes):
    - the two parts, drawn in order on a same-seeded trial, give the joint's params and
      components exactly (lgbm and logistic); the reverse order draws other values;
    - **step 2 does not depend on which total is fixed.** Step 2's logistic study at 400 trials
      (`seed + 8`), with a cheap `poisson_glm` total (`alpha = 1`) fixed instead of step 1's
      best, gave the same `best_params` to the last digit (`C` 0.1012250205366361). The values
      shift by a constant (2.5191 against 2.0622); 46 s against 65 s. The user asked whether
      both steps can be one loop; that is 2.3's choice (below).
  - **Changed:**
    - `builders.py`: `suggest_count_component(trial, *, name, family, use_exposure,
      feature_transformer_dict)` and `suggest_probability_component(trial, *, name, family,
      feature_transformer_dict)`, before the joint function, whose body is now the two calls
      (`name="total"`, then `name="probability"`; the order comment kept); `__all__` and the
      module docstring;
    - `cross_validation.py`: `CrossValidationTuner.tune_total_component(..., total_family,
      use_exposure, total_feature_transformer_dict, probability_component, y_min,
      random_state, study_seed)` and `tune_probability_component(..., total_component,
      probability_family, probability_feature_transformer_dict, ...)`, each shaped like the
      joint method (the joint's keyword names, so a run's settings splat in; each passes its
      part's name); the docstrings say that under the Poisson deviance the fixed part adds a
      constant to every trial, so the best is the joint study's; the imports and the module
      docstring;
    - `__init__.py`: the two suggest functions exported;
    - tests: `test_the_two_parts_drawn_in_order_are_the_joint_draws` (builders, lgbm and
      logistic; it pins the joint's order of calls, and a count component drawn under another
      name holds only that name's keys); `test_the_total_study_scores_its_total_with_the_fixed_probability_component`
      and `test_the_probability_study_scores_its_probabilities_with_the_fixed_total_component`
      (the tuner: each trial's keys are its part's only, and its value is the `cv_score` of the
      read-back part with the fixed one).
  - **Checks:**
    - the two test files 32 passed under `-W error` (17 + 2, 11 + 2); the non-slow suite 1381
      passed, 1 skipped, 1 xfailed, 19 deselected (1377 + 4; the 6 MLflow warnings); mypy no
      issues (29 files; with a fresh `--cache-dir`: the repo's `.mypy_cache` had 9 offloaded
      files and mypy crashed on them); ruff and the format check clean on the five files;
    - Check A: the notebook unchanged (md5 `e5fea5dcd691aa1e69a94769f4cff3d5`), its test 12
      passed, ruff the 14, the two format hunks, `marimo check` the one empty cell;
    - Check C (87 s): the 24 baselines (the 8 flags `False`), `intervals_df` 24 rows; with the
      joint draw through the shared functions, Model 2's 16 are unchanged;
    - **the new methods reproduce 2.3a.1** (400 trials, `PYTHONWARNINGS=error`):
      `tune_total_component` for `lgbm_poisson`, exposure on, `seed + 0`, step 1's fixed
      logistic: 2.071780, the measured `best_params` (50 s); `tune_probability_component` for
      logistic, `seed + 8`, that total fixed: 2.062233, the measured `best_params` (63 s).
    - All of the above were run on the first implementation and again on the revised one, with
      the same results.
  - **Mutations** on a full copy (a sentinel first showed the copy is imported), each caught:
    - the joint draws the probabilities first: the builders test (2);
    - `suggest_count_component` ignores `name` (draws under `total__`): the builders test (2);
    - `tune_total_component` builds with another probability component (`C = 1.0`): the
      total study's test;
    - `tune_probability_component` draws the total too (the joint suggest): the probability
      study's test;
    - `tune_probability_component` builds with another total (the exposure flipped): the
      probability study's test.

### [x] 2.3 Model 2's runs in loops, by coordinate descent

Reshaped on 2026-10-31 for 2.3a.

- **Change:**
  - **settings:** `MODEL2_TOTAL_RUN_SETTINGS` (8 entries, `seed + 0 … 7`),
    `MODEL2_PROBABILITY_RUN_SETTINGS` (2 entries, `seed + 8`, `seed + 9`), and the fixed
    classifier of step 1 (2.3a);
  - **the cells:** a loop for step 1, a loop for step 2 (the best step-1 total by cv fixed), then
    the 16 pairs. `model2_df`, `best_df` and the inspection read the pairs' records as before.
    A pair's `best_params` merge the two steps' params, and its `cv_best` is `tuner.cv_score`
    of the pair;
  - **the arms table** counts the 10 step studies' trials;
  - section 5's text, with the user's approval.
- **Done when:** Check A and Check B pass.
- **Verify:** Check C. Model 1's 8 values are unchanged; Model 2's 16 are re-recorded (3 trials).
- **Commit:** `refactor(notebook): the total times probability model by coordinate descent, its runs in loops` (`9aed504`)
- **Done 2026-10-31.**
  - **The user's choices at the plan:**
    - two loops in sequence: step 2 fixes step 1's best total by cv. Rejected: a cheap fixed
      total in step 2 too (2.3a.2's probe: the same best params, the values shifted by a
      constant), which makes the 10 studies independent but step 2's values no pair's cv;
    - three cells: step 1, step 2, the pairs (each defines one list; the pairs can be re-scored
      without re-running the studies; the two tuning cells keep their `mo.stop(not `);
    - the pairs are not timed: `seconds` is a study's time in both models, as Model 1's (the
      tuner's clock around `study.optimize`); a pair's tuning cost is its two runs' seconds,
      in `model2_steps_df`. The plan had a pair's scoring seconds; dropped, since the notebook
      does not import `time` and the value would mean something else.
  - **Changed** (`notebooks/02_experiments.py`):
    - the import cell: `LogisticParams`;
    - a settings cell before the button: `MODEL2_TOTAL_RUN_SETTINGS` (Model 1's order of
      families, exposure on then off), `MODEL2_PROBABILITY_RUN_SETTINGS` (logistic, lgbm),
      `MODEL2_FIXED_PROBABILITY_COMPONENT` (logistic, `C = 1.0`, no calibration, `glm_base`);
    - the button's caption;
    - the 16 run cells replaced by three: step 1 (`tune_total_component`, records
      `MODEL2_TOTAL_RUNS`), step 2 (the best total read back with `from_params`;
      `tune_probability_component`, `MODEL2_PROBABILITY_RUNS`), the pairs (`cv_score` and
      `fit_and_score`, `MODEL2_RUNS`). A step record: its settings, `trials`, `component`,
      `fixed_component` (the declarations), `cv_best`, `best_params`, `seconds`, `seed`,
      `y_min`. A pair record: `total_family`, `probability_family`, `use_exposure`,
      `components`, `cv_best`, `best_params`, `score_table`, `seed`, `y_min` (no `trials`:
      the steps hold them);
    - the tables cell: `model2_df` from the pairs (without `n_complete` / `n_failed`, which
      belong to studies), the new `model2_steps_df` (one row per study: `step`, `family`,
      `use_exposure`, `cv_best`, `n_complete`, `n_failed`, `seconds`), `model2_arms_df` from
      the 10 studies; the display cell shows the three;
    - section 5's text in the approved wording: "(one part per run for the total times
      probability model)"; "(each model's in loops over its run settings)"; the Model 2
      paragraph on coordinate descent.
  - **Checks:**
    - Check A: the notebook test 12 passed; ruff the same 14 findings; the format diff the two
      hunks (`@@ -158`, `@@ -208`); `marimo check` the one empty cell. The new cells were
      written in `ruff format`'s form (each checked on its own) and spliced in.
    - Check B: the two settings lists (8, 2) in order, the fixed classifier
      (`LogisticParams(C=1.0)`, no calibration, `glm_base`); no `model2_<pair>` name left;
      nothing fitted.
    - Check C (57 s): Model 1's 8 equal the baselines (the 8 flags `False`); 10 step records,
      `model2_steps_df` 10 rows, `model2_arms_df` 45 rows (step 1: 2 × 4 + 6 × 4; step 2:
      5 + 3, 2 + 3); `intervals_df` 24 rows. **Consistency:** each step-2 run's `cv_best`
      equals its pair with the best total exactly (difference 0). The 10 step values equal
      2.3a.1's 3-trial smoke run of its probe.
    - **The 400-trial click of Model 2** (a copy with Model 1's and the inspection's buttons
      stopped, `PYTHONWARNINGS=error`): the 16 pairs' cv and val equal 2.3a.1's "coordinate"
      columns to 4 decimals; no failed trial; 752 s (the totals 26–70 s each, logistic 61 s,
      lgbm 309 s), against 2871 s for the joint studies.
    - The non-slow suite: 1381 passed, 1 skipped, 1 xfailed, 19 deselected.
  - **Model 2's new Check C baselines** (3 trials, `cv_best` of the pairs; Model 1's are
    unchanged). The handoff's table and recipe change in 2.5.

    | Model 2 | exposure on | exposure off |
    |---|---|---|
    | `lgbm_poisson+logistic` | 2.1306 | 2.1797 |
    | `lgbm_poisson+lgbm` | 2.1502 | 2.1993 |
    | `lgbm_gaussian+logistic` | 2.1777 | 2.2232 |
    | `lgbm_gaussian+lgbm` | 2.1973 | 2.2429 |
    | `poisson_glm+logistic` | 2.2540 | 2.8648 |
    | `poisson_glm+lgbm` | 2.2737 | 2.8845 |
    | `nb2+logistic` | 2.2381 | 2.8615 |
    | `nb2+lgbm` | 2.2577 | 2.8811 |

    The step runs at 3 trials: totals (on / off) `lgbm_poisson` 2.1361 / 2.1852,
    `lgbm_gaussian` 2.1832 / 2.2287, `poisson_glm` 2.2595 / 2.8703, `nb2` 2.2436 / 2.8670;
    the classifiers, with the best total (`lgbm_poisson`, exposure on), logistic 2.1306,
    lgbm 2.1502.

### [x] 2.4 Every metric from one fit: fit, predict and evaluate apart

- **Change (as planned on 2026-10-30):** every run scored on every metric, per cohort and for
  the composite, from one fit; the metric constants without the `COHORT_` prefix (the user's
  note): one `POISSON_DEVIANCE = Metric("poisson_deviance", mean_poisson_deviance_loss)` for the
  tuner, the cohort rows and the composite, and `LOG_LOSS`; the tables, `best_df` and the
  best-run rule read the primary metric's composite row.
- **Revised at the user's notes (2026-10-31),** after a first implementation with
  `fit_and_score(model, train, val, *, metrics)`: fitting and evaluating are separate steps.
  1. `BaseAgeGroupModel.evaluate` goes; a caller calls the metric itself (2.4b);
  2. the notebook fits the best model, then predicts on training and validation;
  3. `evaluate(y_true, y_pred, metrics, cohorts="composite")`: a cohort name or a list of them,
     each a column of `y_true` or `"composite"`; every metric's score.
- **The user's choices:** `evaluate` returns a DataFrame (rows the cohorts asked for, columns
  the metric names) and lives in `scoring` (it knows no model, so the tuner, the inspection and
  the notebook can use it); two sub-tasks (2.4, then 2.4b); the score tables long, shown with
  `mo.ui.table`, marimo's interactive table (sort, filter, pages); the six-metric list named
  `METRICS` (it serves the score tables and the bootstrap).
- **Commit:** `refactor(scoring): evaluate scores every metric per cohort; the notebook fits, predicts and evaluates apart`
- **Done 2026-10-31.**
  - **Facts probed at the plan:** on both models' validation predictions, one cohort's
    `mean_poisson_deviance_loss(..., cohort=c)` equals scikit-learn's `mean_poisson_deviance` on
    the column bit for bit; all six metrics score each cohort and the composite, finite; the
    24 scores take about 7 ms per fit.
  - **Changed:**
    - `scoring.py`: `evaluate(y_true, y_pred, metrics, cohorts="composite") -> DataFrame`
      (index `cohort`, columns `metric`; `"composite"` scores without `cohort=`); `__all__`,
      the module docstring;
    - deleted: `hyperparameter_tuning/evaluation.py` (`fit_and_score`), its test file, its
      export and line in the package docstring;
    - `tests/unit/test_scoring.py`: the table (the rows asked for in order, the metric names,
      each value the metric with `cohort=` or without for `"composite"`; a cohort's deviance
      equals scikit-learn's on its column), and one name or the default giving one row;
    - the notebook:
      - the import cell: `evaluate` from `scoring`; `fit_and_score` and scikit-learn's
        `mean_poisson_deviance` out;
      - the metrics cell: `POISSON_DEVIANCE`, `LOG_LOSS`, `METRICS` (the six, the primary
        first); `COHORT_POISSON_DEVIANCE`, the old `POISSON_DEVIANCE`, `COHORT_LOG_LOSS` and
        `INSPECTION_METRICS` gone; the tuner, bootstrap and importance cells take the new names;
      - Model 1's loop and Model 2's pairs: the model fitted on the training part, then
        `evaluate(<part>.Y, model.predict(<part>.X, exposure=…), METRICS,
        cohorts=[*COHORTS, "composite"])` for training and validation; the records keep
        `train_scores` and `val_scores` (in place of `score_table`);
      - the two tables cells: a run's long rows (`cohort`, `metric`, `train`, `val`) from its
        two frames; `cv_best` (Model 1: also `n_complete`, `n_failed`, `seconds`) on the
        primary metric's composite row; the two score tables shown with `mo.ui.table`;
      - `best_df`: the primary metric's composite rows; the best-run cells:
        `run["val_scores"].loc["composite", POISSON_DEVIANCE.name]`;
      - section 5's text: "…scored on training and validation on every metric, per cohort and
        for the composite".
    - Not changed here: HYPERPARAMETER_TUNING.md and MODULE_REFERENCE.md still name
      `fit_and_score` (2.5).
  - **Checks:**
    - `test_scoring.py` 87 passed under `-W error`; ruff and the format check clean on the
      changed files; mypy no issues (28 files, a fresh `--cache-dir`); the non-slow suite
      1383 passed, 1 skipped, 1 xfailed, 19 deselected (1381 − 1 for
      `fit_and_score`'s test + 3 for `evaluate`'s);
    - Check A: the notebook test 12 passed; ruff the 14; the format diff the two hunks;
      `marimo check` the one empty cell;
    - Check B: `evaluate` is `scoring.evaluate`; `METRICS` the six labels (`poisson_deviance`,
      `log_loss`, `mae`, `rmse`, `wmape`, `bias`), the primary first and the tuner's metric;
      the old names gone; nothing fitted;
    - **Check C against a reference run of `04caa68`** (both 3 trials, the 8 flags `False`):
      the 24 `cv_best` equal; the `poisson_deviance` rows of `model1_df` (32) and `model2_df`
      (64) equal the reference's exactly, but for Model 1's `seconds` (the studies' wall
      time); 6 metrics per run and cohort (192 and 384 rows), `cv_best` on 8 and 16 rows;
      `best_df` equal; `intervals_df` and `importance_df` equal but for the two relabelled
      metrics.
  - **Mutations** of `evaluate` on a full copy (a sentinel first showed the copy is imported),
    each caught: the cohort ignored (2 tests); the first metric only (3); a single name not
    wrapped in a list (2).

### [x] 2.4b `model.evaluate` removed; the bootstrap takes the predictions and the cohorts

The user's note 1 of 2.4 (2026-10-31). It reverses "`evaluate` is not changed" (the handoff
§3, decided 2026-10-21). Added by the user at its plan: `BootstrapEvaluator.evaluate` needs no
model and no `X`, it takes the predictions; and during the work: a `cohorts` argument, as
`scoring.evaluate`'s, and its `score` column dropped (below).
- **Change:** `modeling/base.py` loses `evaluate`; the tuner's `fold_score`, the bootstrap and
  the permutation importance call the metric; `BootstrapEvaluator.evaluate(y_true, y_pred,
  groups=None, *, metrics, cohorts="composite")`.
- **Done when:** the suite and mypy pass; Check A.
- **Verify:** Check C: the 24 `cv_best`, `intervals_df`'s composite rows and `importance_df`
  unchanged.
- **Commit:** `refactor(modeling): the model has no evaluate; the bootstrap takes the predictions and the cohorts, the callers call the metric`
- **Done 2026-10-31.**
  - **The user's choices:** the bootstrap's result always indexed (cohort, metric) (the default
    gives `("composite", <metric>)` rows), each replicate scored by `scoring.evaluate` (one
    implementation of the cohort rule; every cohort and metric on the same replicates); the
    notebook bootstraps every cohort and the composite.
  - **Changed:**
    - `modeling/base.py`: `evaluate` and the `scoring` import out; the module docstrings
      (`base.py`, `modeling/__init__.py`) say fit and predict, and that a caller scores the
      predictions with `scoring`;
    - `hyperparameter_tuning/cross_validation.py`: `fold_score` returns
      `self.metric(Y_val, Y_val_pred)`;
    - `inspection/permutation.py`: the scorer returns `sign * self.metric(y_true, y_pred)`;
    - `inspection/bootstrap.py`: `evaluate(y_true, y_pred, groups=None, *, metrics,
      cohorts="composite")`; no model, `X` or exposure (the caller predicts once); `y` renamed
      `y_true`; `check_consistent_length(y_true, y_pred, groups)` (the predictions now
      checked); each replicate `scoring.evaluate(...).stack()`; the non-finite message names
      the cohort and the metric; the unresampled score computed after the check, as before;
    - **`score` dropped, at the user's question** ("why return score; is it necessary?"): it
      was the metric on the unresampled rows, the interval's point estimate. Since the caller
      predicts, `scoring.evaluate` of the same predictions gives it: on Check C the bootstrap's
      `score` equalled the best runs' `train_scores` / `val_scores` in all 4 × 24 rows. The
      bootstrap now returns the interval alone (`mean`, `lower`, `upper`); the notebook's
      caption points to the score tables for the scores themselves;
    - tests: `test_inspection_bootstrap.py` (the ids passed as `y_true` and `y_pred`, which
      keeps the pairing check; `_SpyModel`, `X`, `EXPOSURE` and the predict-once test out; the
      (cohort, metric) index; a short `y_pred` case; the score test replaced by one on the
      rows' order and the interval's columns; every recorder call now a replicate; a new test:
      a cohort's row equals the bootstrap of its column alone, the composite rows a
      composite-only call of the same seed, to 1e-12: NumPy sums one column pairwise and
      several row by row, so the mean's last bit depends on the array's width); `test_modeling_base.py` (the stand-in model and its two `evaluate` tests
      out; the cohort's passing is tested on `Metric`); `test_modeling_count_model.py` (the
      metric called);
    - the notebook: the two bootstrap cells pass `<part>.Y`, `best_model.predict(<part>.X,
      exposure=…)`, `<part>.groups` and `cohorts=[*COHORTS, "composite"]` (the concat's names
      `split`, `cohort`, `metric`); `intervals_df` (96 rows) shown with `mo.ui.table`;
      section 5's text: "the interval of six metrics, per cohort and for the composite".
  - **Checks:**
    - the bootstrap tests 23 passed under `-W error` (with the modeling, permutation and tuner
      tests: 87 before the cohorts); ruff and the format check clean; mypy no issues (28
      files); the non-slow suite 1382 passed, 1 skipped, 1 xfailed, 19 deselected
      (1383 − 3 removed + the short `y_pred` case + the cohorts test);
    - Check A: the notebook test 12 passed; ruff the 14; the format diff the two hunks (two
      hunks of the new lines applied as `ruff format` printed them); `marimo check` the one
      empty cell;
    - **Check C against `c2843e3`'s run** (71 s; 77 s after `score` was dropped, the same
      results, `intervals_df` with `mean`, `lower`, `upper`): the 24 `cv_best`, every run's
      `train_scores` / `val_scores`, `model1_df` (but for `seconds`), `model2_df`, `best_df`
      and `importance_df` equal; `intervals_df` 96 rows (the three cohorts and the
      composite), its composite rows equal the reference's 24 exactly. Before the cohorts were
      added, the same comparison held with `intervals_df` equal as a whole.
  - **Mutations** of the bootstrap on a full copy (a sentinel first), each caught: `y_pred`
    not resampled (3 tests); `y_pred`'s length not checked (3); the score on the last
    replicate's rows (3, before `score` was dropped); the replicates scored without
    `cohorts` (1); rerun after the drop, the three that apply still caught.

### [ ] 2.5 Docs

- **Change:**
  - NOTEBOOK_TUNING_PLAN.md §0: the new decisions (M6, M7) supersede D13/D14; Model 2's
    coordinate descent (2.3a);
  - HYPERPARAMETER_TUNING.md: the new tune methods and suggest functions (2.3a.2); §8 without
    `fit_and_score`: the fit, the predictions and `scoring.evaluate` (2.4), the block rerun;
    `evaluate` out of the model (2.4b);
  - MODULE_REFERENCE.md (the `scoring.py`, `hyperparameter_tuning`, `modeling/base.py` rows) and
    the guides that call `model.evaluate` (INDEPENDENT_COHORT_MODELS.md,
    TOTAL_TIMES_PROBABILITY_MODEL.md, MODEL_INSPECTION.md), their blocks rerun;
  - the handoff's §1–§2 and its checks;
  - this plan's status.
- **Done when:** the doc blocks run as written and the links resolve.
- **Before the stop:** the PR's review subagent.
- **Commit:** `docs(notebook): the run loop and fit_and_score's metrics`

## 5. PR 3: the objective EDA

Branch `feat/objective-eda`, from `main` after PR 2 is merged.

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
