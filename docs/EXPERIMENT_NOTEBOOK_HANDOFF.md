# Handoff: the experiment notebook, after the tuning rebuild

**For:** the session that continues [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)
(three PRs: the metrics in `scoring`, the runs in a loop, the objective EDA). PR 1 is merged
(#18), and the experiment-notebook line is merged into `main` (#19, §2). The next session starts
PR 2 at sub-task 2.1, on `refactor/notebook-run-loop` from `main`. MLflow and E11's final
evaluation are deferred (§6).

**Written:** 2026-10-29, after T5 of [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md);
**updated** 2026-10-30 after sub-task 1.1 of the metrics plan, after the user's notes on its
helpers, after sub-tasks 1.2, 1.3 and 1.4, and after the merges of #18 and #19 (§2, §6). The detailed history of
the tuning work is in that plan (§0 D1–D26, §2–§6 the sub-tasks), in
[NOTEBOOK_T4_HANDOFF.md](NOTEBOOK_T4_HANDOFF.md) (T4–T5) and in git; the decisions of the current
work are M1–M9 of the metrics plan. Re-verify every fact here against the code of the day.

## 1. What exists

**History in one paragraph.** On 2026-10-18 the user rejected the first `hyperparameter_tuning`
package (`set_params` paths, too much machinery). Section 5 of the notebook was rewritten with
plain Optuna and a hand-written cross-validation loop for both models (T1–T2), the best runs got
their bootstrap intervals and importances (T2.5), the metric constants left `scoring.py` (T3),
and the design was reshaped at the user's reviews (D23–D25). The old package was then deleted
(T4a) and a new one built from the notebook's code (T4b.1–T4b.2), revised at the user's notes
(D26: components-only suggest functions, `CrossValidationTuner`), and imported by the notebook
(T4b.3). T5 wrote the guide.

**The notebook** (`notebooks/02_experiments.py`, marimo; a script run fits nothing):
1. **Data:** the simulator at 150 neighborhoods × 9 buildings (`configs/simulation.yml`, its
   seed 42 unless `simulation_seed` overrides it), the room shares, `X`, `Y` (three cohorts),
   the exposure (`n_apartments`), the groups (neighborhoods); the settings printed and hashed.
2. **Settings:** `seed = 42`, `split_method = "random"`, `test_size = val_size = 0.2`,
   `n_splits = 5`, `n_trials = 400`, `n_startup_trials = 20`, `share_hyperparams = True`,
   `share_feature_transformer = False`, `n_resamples = 1000`, `n_repeats = 10`,
   `n_shap_rows = 100`, `n_shap_background = 50`.
3. **Split:** train 917 / validation 306 / test 306 rows (`AgeGroupDataset` per part), and
   `cv` / `cv_groups` from the same `Splitter`.
4. **Feature variants:** the declarations of FEATURE_TRANSFORMATIONS.md §8, and each family's
   candidates as data: `COUNT_FEATURE_TRANSFORMERS` and `PROBABILITY_FEATURE_TRANSFORMERS`
   (family → `{name: FeatureTransformer}`).
5. **Tuning and inspection:**
   - the metrics cell (`COHORT_POISSON_DEVIANCE`, `POISSON_DEVIANCE`, `COHORT_LOG_LOSS`, the
     six `INSPECTION_METRICS`, all `scoring` functions), `COHORTS`, `PREDICTION_FLOOR`;
   - the `tuner` cell: `CrossValidationTuner(cv=cv, metric=COHORT_POISSON_DEVIANCE,
     n_trials=n_trials, n_startup_trials=n_startup_trials, n_jobs=n_splits)`;
   - behind the "Tune …" buttons, 8 Model 1 run cells (`tuner.tune_independent_cohorts_model`)
     and 16 Model 2 run cells (`tuner.tune_total_times_probability_model`), each keeping a
     record (below), seeded `seed + i` in the cells' order;
   - the tables: `MODEL1_RUNS` / `model1_df` / `model1_arms_df`, `MODEL2_RUNS` / `model2_df` /
     `model2_arms_df`, and `best_df` (each model's best run on validation);
   - behind "Inspect the best runs": each model's best run rebuilt from its `best_params`,
     refit on training, the bootstrap of the six metrics on training and validation
     (`intervals_df`), the permutation importance and exact SHAP (`importance_df`).

**A run record** (one dict per run cell; the input of the MLflow runs): the settings
(`family`, `use_exposure`, the two sharing flags for Model 1; `total_family`,
`probability_family`, `use_exposure` for Model 2), `trials` (`study.trials_dataframe()`, every
trial), `components` (each best component's `to_dict()`: the estimator params and the
transformer's declaration), `cv_best`, `best_params`, `score_table` (per cohort and the
composite, train and validation), `seconds`, `seed`, `y_min`.

**The packages it uses**, each with its guide:
[`hyperparameter_tuning`](HYPERPARAMETER_TUNING.md) (records, search spaces, builders, the
tuner, `fit_and_score`, aggregation), [`inspection`](MODEL_INSPECTION.md) (`BootstrapEvaluator`,
`PermutationImportance`, `ShapExplainer`), `modeling` ([Model 1](INDEPENDENT_COHORT_MODELS.md),
[Model 2](TOTAL_TIMES_PROBABILITY_MODEL.md)), [`splitting`](SPLITTING.md) and
`feature_engineering` ([FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md)). MLflow is not
used yet; the old `tracking/` package belongs to the old stack.

## 2. State (2026-10-30, end of day)

- **Branches:** `main` at `3db1aa8` (`Merge pull request #19 from galkampel/feat/experiment-notebook`,
  a merge commit, so every commit the docs cite by hash is on `main`) and, from it,
  **`refactor/notebook-run-loop`** (PR 2 of the metrics plan; the user opens it as a draft into
  `main`). `feat/experiment-notebook` and `refactor/scoring-metrics` are deleted, locally and on
  GitHub. The user commits, pushes and merges.
- **The merges:** #17 (the tuning package rebuilt) squashed as `a748cd4` and #18 (PR 1 of the
  metrics plan) squashed as `b1d8dbd`, both into `feat/experiment-notebook`; #19 merged it into
  `main` as `3db1aa8`. PR 1's sub-task commits (`185285e` 1.1, `96c950b` 1.2, `0e9d0a1` 1.3,
  `13a9681` 1.4) live inside `b1d8dbd`; GitHub keeps them under #18.
- **What PR 1 left in place** (the metrics plan's 1.1–1.4 hold the details and the checks):
  - `scoring`: `multinomial_log_loss`, `mean_poisson_deviance_loss`, `mean_absolute_error_loss`,
    `root_mean_squared_error_loss`, `wmape_loss`, `mean_bias_error`, each `(y_true, y_pred, *,
    cohort: str | None = None)`; finite inputs of one shape, paired by position; the count
    metrics pool every cell; `cohort=`, a column name of `y_true`, scores one cohort (the log
    loss one-vs-rest, M3; it needs at least two cohorts). `Metric.__call__` and
    `BaseAgeGroupModel.evaluate` pass `cohort` on only when given.
  - `utils`: public `check_paired_arrays` (finite float arrays of one shape) and
    `column_position` (a DataFrame column's position by name).
  - The notebook: `INSPECTION_METRICS` is the six `scoring` metrics, labels unchanged
    (`cohort_poisson_deviance`, `cohort_log_loss`, `mae`, `rmse`, `wmape`, `bias`); D² is gone.
    The metrics cell still defines `COHORT_POISSON_DEVIANCE`, `POISSON_DEVIANCE` (scikit-learn's
    `mean_poisson_deviance`, `fit_and_score`'s `cohort_metric=`) and `COHORT_LOG_LOSS`.
  - WMAPE and bias differ from the old notebook code in their last bit (≤ 1.1e-16): they sum a
    DataFrame's column-major array in memory order, the old `on_cells` in row order. The user
    chose a tolerance over changing `scoring`.
  - The guides take the new names and run as written; MODEL_INSPECTION.md's simulator line uses
    `from_config(...).simulate().dataset` (the simulator plan's Stop 4 notes it).
  - **2.4 amended by the user:** one `POISSON_DEVIANCE = Metric("poisson_deviance",
    mean_poisson_deviance_loss)` serves the tuner, the cohort rows and the composite;
    `COHORT_LOG_LOSS` becomes `LOG_LOSS`; no `COHORT_` prefixes; the labels change, the numbers
    do not.
- **Settled at PR 1 (do not re-ask):** `Metric.__call__` keeps its `cohort is None` branch (a
  two-argument function would raise on an always-passed `cohort=None`); the shape check runs
  inside each metric; no repeated-name guard (the cohort names are unique by construction);
  `cohort` is a `str`, not a `Literal`; no private helpers, shared ones public in `utils.py`.
- **Behaviour changes of PR 1** (scikit-learn's `log_loss`): a cohort with no child in `y_true`
  scores; a zero probability for a cohort that has children gives a large finite loss (36.04 in
  a probe), not `inf`; no child at all raises; a NaN or an infinity raises in every metric.
- **Notebook md5:** `b3287cd0e24b9b489b1b9f63fc7cbac0` (`main`).
- **Checks on `main`'s tree** (identical to PR 1's last tested tree):
  - the non-slow suite: 1373 passed, 1 skipped, 1 xfailed, 19 deselected (6 warnings: MLflow's
    integer-schema hint in the old stack's `test_gate8_tracking.py`);
  - mypy (`.venv/bin/mypy` with no arguments: 29 files): no issues;
  - Check A: the notebook test 12 passed; `ruff check notebooks/02_experiments.py` 14 findings,
    all pre-existing (12 marimo `return`s PLR1711, B018 at 129, RUF059 `test_age_group` at 347);
    `ruff format --diff` only the user's two hunks, `@@ -158` and `@@ -208`; `marimo check` 1
    issue (the remaining empty cell, line 637);
  - Check C: the 24 baselines below (75–80 s).
- **Check C baselines** (the regression check: the clicked run on a copy with `n_trials = 3` and
  `share_hyperparams = False`, both flags off, under `PYTHONWARNINGS=error`; `cv_best`). Every
  change since T1.6 has reproduced them to the fourth decimal, PR 1 included.

  | Model 1 | exposure on | exposure off |
  |---|---|---|
  | `lgbm_poisson` | 2.1902 | 2.2814 |
  | `lgbm_gaussian` | 2.1906 | 2.3151 |
  | `poisson_glm` | 2.5703 | 2.8776 |
  | `nb2` | 2.2826 | 2.8904 |

  | Model 2 | exposure on | exposure off |
  |---|---|---|
  | `lgbm_poisson+logistic` | 2.1543 | 2.1929 |
  | `lgbm_poisson+lgbm` | 2.2212 | 2.2505 |
  | `lgbm_gaussian+logistic` | 2.1318 | 2.1685 |
  | `lgbm_gaussian+lgbm` | 2.1706 | 2.2239 |
  | `poisson_glm+logistic` | 2.2646 | 2.8660 |
  | `poisson_glm+lgbm` | 2.2794 | 2.8828 |
  | `nb2+logistic` | 2.2487 | 2.8540 |
  | `nb2+lgbm` | 2.3039 | 2.9222 |

  Model 1's other arms at 3 trials (on / off): with `share_hyperparams = True` (the notebook's
  default) `lgbm_poisson` 2.1812 / 2.2193, `lgbm_gaussian` 2.2177 / 2.2805, `poisson_glm`
  2.4158 / 2.8685, `nb2` 2.2762 / 2.8707; with both flags `True` 2.1957 / 2.2710, 2.2279 /
  2.2805, 2.2655 / 2.8759, 2.2532 / 2.8740. Model 2 does not depend on the flags.
- **The 400-trial click** (2026-10-22, D20; both buttons, `PYTHONWARNINGS=error`, both sharing
  flags off): 3708 s (Model 1 821 s, Model 2 2871 s), no failed trial. The best run per model on
  validation: Model 1 `lgbm_poisson`, exposure on, cv 2.1280, val 2.1254 (98 s); Model 2
  `lgbm_gaussian+lgbm`, exposure on, cv 2.0962, val 2.1202 (247 s). The full tables are in the
  tuning plan's D20.
- **Earlier commits** (T1–T5, PR #17) are listed in the tuning plan's status line and in git.

## 3. Decisions (do not re-ask)

The decisions and their evidence are in [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §0; a
later one supersedes an earlier one where it says so.

- D1: plain Optuna and a hand-written CV loop. D2–D3: frozen, keyword-only dataclasses for the
  params and the components (revised by D8, D10, D11, D23, D25). D4: the metrics are built in
  the notebook; `scoring` keeps `Metric` and the two functions.
- D6: one study per family combination and exposure, on the composite; LightGBM's two losses
  are families; the exposure a run setting, both arms. D7: the facts settled by probe (the
  plain mean of the folds, a univariate `TPESampler(seed)`, `catch=(RuntimeError,)`).
- D8: a component is a named part of the composite; trial names `<name>__<parameter>`. D9: an
  aggregator combines the fold scores, `Mean` by default.
- D10–D11: complete components; search spaces hold every Optuna call and build nothing. D12:
  a study is one family combination and exposure.
- D13–D14: runs one after another with the folds in parallel; one cell per run; every trial
  kept, only the best refit. D15: strict checks with `PYTHONWARNINGS=error`; untried choices
  show 0. D16: `CALIBRATION_METHODS` a ClassVar; counting by label; a family pair as
  `"<total>+<probability>"`; `use_exposure` its own column.
- D17: the old package replaced (T4a, T4b). D18: the bootstrap and importance of the best runs;
  no wrapper for the two importance methods. D19: no pruning, no other parallel place
  (measured). D20: `n_trials = 400`, `n_startup_trials = 20`; never decrease `n_trials`; all 24
  runs kept.
- D21: what lives in the package (the mechanism) and what stays in the notebook (the
  experiment); one-way modules; evaluation stays outside the model. D22: `BootstrapEvaluator`
  holds the resampling design and takes the metrics per call; exact SHAP of 100 validation rows
  against 50 training rows.
- D23–D24: components hold the transformer object (`to_dict()` writes its declaration); the
  candidates injected from section 4; `replication_method` fixed at `"weighted"`; the two
  sharing flags as experiment settings; the builders named after the models.
- D25: the search spaces read nothing back; the field `estimator_params`; `from_params` per
  component type; the objectives one loop each.
- D26: `suggest_*_components` return the components; `CrossValidationTuner` holds the study's
  design, with one named tune method per model (the user chose them over one `tune` with a
  `Literal`); `fit_and_score` in `evaluation.py`.

**Decided by the user** (2026-10-21 onward): `y_min` stays a build argument, recorded once per
run with the seed; `evaluate` is not changed; `Aggregator.aggregate_scores` is a named method;
`num_leaves` above `2**max_depth` stays a known inefficiency; Model 2 has all 16 run cells and
no list of totals; all 24 runs are kept; `scoring.Metric` keeps `direction` and `__call__`.

**Dataclass vs TypedDict, for the record** (the records are frozen dataclasses):

| | frozen dataclass (chosen) | TypedDict |
|---|---|---|
| JSON out | `component.to_dict()`: the params nested, the transformer by its declaration | the dict itself |
| JSON in | `from_params(best_params, ...)` per component type; each field read by name, a missing one a `KeyError` | `json.loads`; nothing checks the keys |
| Typos | a misspelt field raises `TypeError` when built (LightGBM would accept it silently) | a typo is another key |
| Immutability | frozen and hashable | a plain dict, not hashable |
| Equality | by value: a component read back from a trial's params equals the one drawn (tested) | by value |
| Dispatch | on the `family` field, an `if`/`elif` with a raising `else` | the same, on a family key |
| Behaviour | `make_estimator`, `from_params`, `to_dict`; no Optuna (the search spaces hold it) | functions only |

## 4. How to work

The standing rules: [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §1 and
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3.

**Each sub-task:** plan mode first; re-verify the facts with read-only probes
(`PYTHONPATH=src .venv/bin/python -W error -c "..."`); a design question gets a recommendation,
the measured evidence and the trade-off (AskUserQuestion with code previews works well), and
waits for the answer; write the steps and the commit message in the chat; one sub-task per
approval. At each stop: a file-by-file summary, the checks (A, B, C; the tests and mypy where
`src` changes; ruff on the notebook against the 14; the format diff's two hunks), the notebook's
new md5, the commit message. Tick the sub-task and update the status lines.

**Git and the editor:**
- The user commits, pushes and merges. No `Co-Authored-By` line. Never `git checkout`,
  `stash`, `reset` or `mv`; delete with `rm`; restore a file with `git show HEAD:<path> >
  <path>` and `cmp`.
- **The user edits between turns.** Check `git status` and the notebook's md5 before each
  sub-task and each edit; show any diff the handoff does not name, and ask.
- **marimo's editor trims each cell's `return`** to the names other cells read when the user
  saves: ruff jumps to about 31 findings and Check B loses names. The user has chosen to have
  HEAD's returns restored, after seeing the diff.
- **Format new cells to `ruff format`'s output; never format the whole file** (the user's two
  hunks). For many similar cells, generate them with a script in the scratchpad, format each
  on its own, insert them, and check that no blank line is added.
- No new ruff finding. A cell that uses a name imported in another cell lists it as a
  parameter (marimo's runtime finds it anyway; ruff reports F821 otherwise).

**The checks:**
- **Check A:** the notebook test, ruff, the format diff, `marimo check`.
- **Check B:** the names a script run defines (`app.run()`'s defs).
- **Check C:** the clicked run on a copy. Build `<scratch>/project/notebooks/02_experiments.py`
  from the notebook with the line `    n_trials = 400` replaced by `    n_trials = 3` (by line: the
  header's markdown holds the string too) and `    share_hyperparams = True` by `False`, and
  `<scratch>/project/configs` a symlink to the repo's `configs`; run it with `marimo.ui.run_button`
  patched to a clicked value, `PYTHONWARNINGS=error`, `PYTHONPATH=src`, in the background. It
  takes 80–170 s with the inspection; the 24 values must equal §2's.
- The machine can be heavily loaded: run notebook checks and the suite in the background with
  stdin closed, unbuffered, logged to the scratchpad; `timeout` does not exist on macOS.

**Traps:**
- Never run two tuner calls (two `Parallel` calls) at once from threads in one process: joblib
  mixes their inputs and the scores are wrong (D19).
- Fold processes do not inherit `-W`; a warning there raises only under `PYTHONWARNINGS`.
- **Mutation checks need a full copy:** `pyproject.toml`'s pytest `pythonpath = ["src"]` puts the
  repo's `src` first, so a mutated copy on `PYTHONPATH` is ignored (every mutation "passes").
  Copy `src`, `tests` and `pyproject.toml` to the scratchpad, mutate there, run pytest with
  `--rootdir <copy> -c <copy>/pyproject.toml`, and first confirm the copy is imported. Never
  mutate the repo while Check C runs: its fold processes re-import the modules.
- **mypy:** run `.venv/bin/mypy` with no arguments (the 29 configured files); `mypy src` reports
  the old stack's 44 known errors.
- The characterization test requires every cell holding `tuner.tune_` to hold `mo.stop(not `;
  that string must not appear in a markdown cell or a comment outside the run cells.
- A function sent to joblib must not capture the notebook's defs (`_thread.lock`).
- pandas 3.0.5 turns a `None` calibration into NaN and `value_counts()` drops it: count by label
  after `fillna("None")` (D16).
- `json.dumps(record, default=str)` writes a bare `NaN`; use `trials.to_json(orient="records")`
  when a record is saved.
- The user is not familiar with marimo's UI terms ("button"): explain them.

**The user's style:** explicit cells, one per run, not loops over runs; no function unless it is
reused; names that say what a thing holds; a record holds the object and writes its declaration;
one source of truth for a list of candidates (the section-4 dicts; `CALIBRATION_METHODS`); no
field or check that nothing reads; scikit-learn's terms; `if`/`elif` with a raising `else`;
code that may move to a package reads no notebook global; settings in the constructor, data in
the call; no private helpers: a shared helper is public in `utils.py`, named in scikit-learn's
terms (`check_paired_arrays`, `column_position`).

## 5. Open items

- **T2.3's F5:** the two tables cells count with different idioms; left as is.
- **T1's F7:** `num_leaves` above `2**max_depth` changes nothing in LightGBM; revisit in a
  search-space pass.
- **The user's:** two empty cells and the bare `return`s marimo writes; ask before changing.
- **Stale text in section 4:** its header says "the check below prints it" and "fitted once …
  as a check", but the check cell is gone. The user's text; ask.
- **SPLITTING.md §3** and the simulator plan's Stop 4 doc examples (`.run()` → `simulate()`;
  MODEL_INSPECTION.md §2 was done in the metrics plan's 1.2),
  and the simulator plan's cleanup stop (dead code) and format-only commit.
- **A `log(n_apartments)` plan for the GLMs** (FEATURE_TRANSFORMATIONS.md §4.4); not asked for.
- **What would shorten a click** (62 min at 400 trials) is a modeling choice the user has not
  asked for: fewer calibration inner folds or a lower `n_estimators` ceiling (D19).

## 6. Next, and the prompt

The order is [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)'s, one
sub-task per approval:
1. **PR 1** (`refactor/scoring-metrics`): merged as #18 (`b1d8dbd`), and with the
   experiment-notebook line into `main` as #19 (`3db1aa8`).
2. **PR 2** (`refactor/notebook-run-loop`, from `main`, a draft into `main`): the sharing-flags
   test (2.1), the runs in a loop (2.2–2.3), and `fit_and_score(model, train, val, *, metrics)`
   scoring every metric per cohort (`cohort=`) and for the composite, with the user's 2.4
   amendment (one `POISSON_DEVIANCE`, `LOG_LOSS`, no `COHORT_` prefixes; §2).
3. **PR 3** (`feat/objective-eda`, from `main` after PR 2 is merged): the objective EDA
   (cross-entropy, Poisson deviance, RMSE on one run per model).
4. **Deferred:** the MLflow runs (the D1e guard first: with `MLFLOW_TRACKING_URI` unset, MLflow 3.16
   writes to the repo's canonical `mlflow.db` and its artifacts to `./mlruns`), D4's comparison
   table, and E11's final evaluation.

The user's preferences for this work: PR descriptions carry no "Generated with Claude Code" line
and commit messages no `Co-Authored-By`; plan docs live in `docs/`, split into sub-tasks the user
validates one at a time.

**The prompt for the next session:**

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full, then docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md in
full (§4, PR 2), src/age_group_prediction/hyperparameter_tuning/components.py and builders.py,
and tests/unit/test_hyperparameter_tuning_builders.py. Verify the state of the handoff's §2
first: branch refactor/notebook-run-loop from main (3db1aa8, the merge of #19), HEAD my commit
of this handoff on top of it, a clean tree, the notebook md5 b3287cd0e24b9b489b1b9f63fc7cbac0.
If anything differs, stop and show me the diff. Then plan sub-task 2.1 in plan mode: re-verify
the facts with read-only probes, give a recommendation with evidence and the trade-off for any
design question, write the steps and the commit message in the chat, and wait for my approval
before editing. Then each later sub-task of the plan in order, one per approval; I commit
myself (never commit, no Co-Authored-By line, no "Generated with Claude Code" in PR text,
never git checkout/stash/reset/mv).
```
