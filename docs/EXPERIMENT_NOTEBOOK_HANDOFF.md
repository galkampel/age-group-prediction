# Handoff: the experiment notebook, after the tuning rebuild

**For:** the session that continues [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)
(three PRs: the metrics in `scoring`, the runs in a loop, the objective EDA). It starts with the
user's validation of sub-task 1.1, which is implemented and **uncommitted** (§2). MLflow and E11's
final evaluation are deferred (§6).

**Written:** 2026-10-29, after T5 of [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md);
**updated** 2026-10-30 after sub-task 1.1 of the metrics plan (§2, §6). The detailed history of
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
     seven `INSPECTION_METRICS`), `COHORTS`, `PREDICTION_FLOOR`;
   - the `tuner` cell: `CrossValidationTuner(cv=cv, metric=COHORT_POISSON_DEVIANCE,
     n_trials=n_trials, n_startup_trials=n_startup_trials, n_jobs=n_splits)`;
   - behind the "Tune …" buttons, 8 Model 1 run cells (`tuner.tune_independent_cohorts_model`)
     and 16 Model 2 run cells (`tuner.tune_total_times_probability_model`), each keeping a
     record (below), seeded `seed + i` in the cells' order;
   - the tables: `MODEL1_RUNS` / `model1_df` / `model1_arms_df`, `MODEL2_RUNS` / `model2_df` /
     `model2_arms_df`, and `best_df` (each model's best run on validation);
   - behind "Inspect the best runs": each model's best run rebuilt from its `best_params`,
     refit on training, the bootstrap of the seven metrics on training and validation
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

## 2. State (2026-10-30)

- **Branches:** `feat/experiment-notebook` (from `main`, no PR into `main` yet; it holds T1–T5 and
  PR #17's squash `a748cd4`) and, from it, **`refactor/scoring-metrics`** (PR 1 of the metrics
  plan; the user opens it as a draft into `feat/experiment-notebook`). The user merges.
- **HEAD of `refactor/scoring-metrics`:** `492f171` (`chore(notebook): an empty cell and an unused
  optuna return removed`), after `ff6087c` (the metrics plan) and `8cb13b0` (this handoff's
  previous update).
- **Uncommitted: sub-task 1.1 of the metrics plan, awaiting the user's validation and commit.**
  `git status` shows 1.1's 11 modified files, plus this handoff unless the user committed it
  first (a docs commit of its own):
  - `src/age_group_prediction/scoring.py`:
    - `multinomial_log_loss` (was `cohort_log_loss`; scikit-learn's `log_loss` called with
      `y_true=`, `y_proba=`, `labels=range(n_cohorts)`) and `mean_poisson_deviance_loss` (was
      `cohort_poisson_deviance`);
    - new: `mean_absolute_error_loss`, `root_mean_squared_error_loss`, `wmape_loss`,
      `mean_bias_error`;
    - every function takes `(y_true, y_pred, *, cohort: str | None = None)`, with `cohort` a
      column name of `y_true` (M9), and pools every cell;
    - the helpers, public in `utils.py`: `check_paired_arrays` (the shape check) and
      `column_position` (the column of a name); each metric selects its cohort;
    - `Metric.__call__(..., *, cohort=None)` passes the cohort on only when given; "samples", not
      "buildings" (M1).
  - `src/age_group_prediction/modeling/base.py`: `evaluate(..., *, cohort=None)`.
  - `tests/unit/test_scoring.py` (rewritten), `tests/unit/test_modeling_base.py` (one test) and
    `tests/unit/test_utils.py` (the two helpers).
  - The code callers, function references only (the `Metric` labels `"cohort_log_loss"` and
    `"cohort_poisson_deviance"` are unchanged): `test_hyperparameter_tuning_cross_validation.py`,
    `test_hyperparameter_tuning_evaluation.py`, `test_inspection_bootstrap.py`,
    `test_inspection_permutation.py`, `test_modeling_cohort_probability.py`, and the notebook's
    import and metrics cells.
  - `docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md` (M1, M3, M4, M9, 1.1 ticked, 2.4 settled).
  - The suggested commit message is the plan's 1.1 "Commit" line, with the body given in the
    chat on 2026-10-30. The user commits only after validating.
- **Behaviour changes in 1.1** (from the user's switch to scikit-learn's `log_loss`): a cohort with
  no child in `y_true` now scores; a zero probability for a cohort that has children is clipped to
  a large finite loss (36.04 in a probe) instead of `inf`; no child at all raises scikit-learn's
  `ValueError` instead of returning `nan`.
- **Notebook md5:** `d3fe1197a66230e5aed51efd7caaf276` (the uncommitted renames).
- **Checks on the uncommitted tree** (2026-10-30):
  - the non-slow suite: 1346 passed, 1 skipped, 1 xfailed, 19 deselected (HEAD's baseline on a
    `git archive` copy: 1299 passed);
  - mypy: no issues in 29 source files;
  - `ruff check notebooks/02_experiments.py`: 14 findings, all pre-existing (12 marimo `return`s
    PLR1711, B018 at 129, RUF059 `test_age_group` at 347; the lines shifted when the user removed
    an empty cell);
  - `ruff format --diff`: only the user's two hunks, now `@@ -158` and `@@ -208`;
  - `marimo check`: 1 issue (the remaining empty cell, line 635);
  - mutations, each caught and restored by copy and `cmp`: dropping `labels=`; RMSE without
    `ravel`; dropping the 2-D check; the count metrics ignoring `cohort`; the pair built from
    another column; `Metric` dropping `cohort`;
  - Check C (below): the 24 baselines, 97 s, under `PYTHONWARNINGS=error`. The six inspection
    metrics' intervals are saved as the reference for sub-task 1.3, in the old session's
    scratchpad (`checks_1_1b.log`): regenerate them by Check C on the 1.1 tree before 1.3.
- **Check C baselines** (the regression check: the clicked run on a copy with `n_trials = 3` and
  `share_hyperparams = False`, both flags off, under `PYTHONWARNINGS=error`; `cv_best`). Every
  change since T1.6 has reproduced them to the fourth decimal, 1.1 included.

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
the call.

## 5. Open items

- **T2.3's F5:** the two tables cells count with different idioms; left as is.
- **T1's F7:** `num_leaves` above `2**max_depth` changes nothing in LightGBM; revisit in a
  search-space pass.
- **The user's:** two empty cells and the bare `return`s marimo writes; ask before changing.
- **Stale text in section 4:** its header says "the check below prints it" and "fitted once …
  as a check", but the check cell is gone. The user's text; ask.
- **SPLITTING.md §3** and the simulator plan's Stop 4 doc examples (`.run()` → `simulate()`),
  and the simulator plan's cleanup stop (dead code) and format-only commit.
- **A `log(n_apartments)` plan for the GLMs** (FEATURE_TRANSFORMATIONS.md §4.4); not asked for.
- **What would shorten a click** (62 min at 400 trials) is a modeling choice the user has not
  asked for: fewer calibration inner folds or a lower `n_estimators` ceiling (D19).

## 6. Next, and the prompt

The order is [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)'s, one
sub-task per approval:
1. **The user validates 1.1** (the uncommitted diff of §2) and commits it, or gives notes; a note
   is planned and fixed in 1.1 before anything else.
2. **PR 1:** 1.2 (the four guides take the new names and `cohort=`, their blocks rerun; first
   show the user how to handle MODEL_INSPECTION.md's stale `StudentPopulationSimulator(config).run()`),
   then 1.3 (the notebook's metrics from `scoring`, D² out; the six intervals equal 1.1's).
3. **PR 2** (`refactor/notebook-run-loop`): the sharing-flags test, the runs in a loop, and
   `fit_and_score(model, train, val, *, metrics)` scoring every metric per cohort (`cohort=`) and
   for the composite.
4. **PR 3** (`feat/objective-eda`): the objective EDA (cross-entropy, Poisson deviance, RMSE on
   one run per model).
5. **Deferred:** the MLflow runs (the D1e guard first: with `MLFLOW_TRACKING_URI` unset, MLflow 3.16
   writes to the repo's canonical `mlflow.db` and its artifacts to `./mlruns`), D4's comparison
   table, and E11's final evaluation.

The user's preferences for this work: PR descriptions carry no "Generated with Claude Code" line
and commit messages no `Co-Authored-By`; plan docs live in `docs/`, split into sub-tasks the user
validates one at a time.

**The prompt for the next session:**

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full, then docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md in
full, src/age_group_prediction/scoring.py, src/age_group_prediction/modeling/base.py (evaluate)
and tests/unit/test_scoring.py. Verify the state of the handoff's §2 first: branch
refactor/scoring-metrics, HEAD 492f171 (or my commit of this handoff right after it), sub-task
1.1 uncommitted (the 11 modified files listed there), the notebook md5
d3fe1197a66230e5aed51efd7caaf276. If anything differs, stop and show me
the diff. Then wait: I validate 1.1 first, and I will give notes or commit it. Plan each note,
and afterwards each sub-task of the plan in order (1.2 next), in plan mode: re-verify the facts
with read-only probes, give a recommendation with evidence and the trade-off for any design
question, write the steps and the commit message in the chat, and wait for my approval before
editing. One sub-task per approval; I commit myself (never commit, no Co-Authored-By line, no
"Generated with Claude Code" in PR text, never git checkout/stash/reset/mv).
```
