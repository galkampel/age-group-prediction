# Handoff: the experiment notebook, after the tuning rebuild

**For:** the session that adds the MLflow runs and the final evaluation to
`notebooks/02_experiments.py`: the runs logged from the notebook's run records, the comparison
table, and E11's refit on training + validation scored once on test.

**Written:** 2026-10-29, after T5 of [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md);
**updated** the same day after PR #17 was squash-merged (§2). The
detailed history of the tuning work is in that plan (§0 D1–D26, §2–§6 the sub-tasks), in
[NOTEBOOK_T4_HANDOFF.md](NOTEBOOK_T4_HANDOFF.md) (T4–T5) and in git. Re-verify every fact here
against the code of the day.

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

## 2. State (2026-10-29)

- **Branch:** `feat/experiment-notebook` (from `main`). PR #17 (T4a–T5) was squash-merged
  into it as `a748cd4` on 2026-10-29, and its branch `refactor/hyperparameter-tuning-package`
  was deleted locally and on GitHub. `feat/experiment-notebook` holds T1–T5 and has no PR into
  `main` yet. The user merges.
- **HEAD:** the user's commit of this handoff's update, right after `a748cd4`. **Notebook md5:**
  `771464b5583902cf0cf8279359ae0f23` (unchanged by the merge: the merged tree equals the PR
  branch's last commit, `b66a8ed`).
- **Checks at HEAD:**
  - the notebook test (`tests/characterization/test_experiments_notebook.py`) 12 passed;
  - `ruff check notebooks/02_experiments.py`: 14 findings, all pre-existing: 12 marimo
    `return`s (PLR1711, lines 27, 39, 225, 238, 315, 393, 712, 1595, 3093, 3117, 3431, 3487),
    B018 at 129, RUF059 `test_age_group` at 352;
  - `ruff format --diff`: only the user's two hunks, `@@ -158` and `@@ -213`;
  - `marimo check`: the two known empty-cell issues;
  - `uv run --frozen mypy`: no issues in 29 source files;
  - `pytest -m "not slow"`: 1299 passed, 1 skipped, 1 xfailed, 19 deselected (six MLflow
    `UserWarning` hints from `test_gate8_tracking.py` pre-exist).
- **Check C baselines** (the regression check: the clicked run on a copy with `n_trials = 3` and
  `share_hyperparams = False`, both flags off, under `PYTHONWARNINGS=error`; `cv_best`). Every
  change since T1.6 has reproduced them to the fourth decimal.

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
- **The 400-trial click** (2026-10-22, D20; both buttons, `PYTHONWARNINGS=error`, **both
  sharing flags off**: the flags did not exist yet, so a click with today's default
  `share_hyperparams = True` gives other Model 1 values): 3708 s (Model 1 821 s, Model 2
  2871 s), no failed trial, no warning, every choice tried at least 11 times. The best run per
  model on validation:

  | Model | run | cv_best | validation |
  |---|---|---|---|
  | Model 1 | `lgbm_poisson`, exposure on | 2.1280 | 2.1254 |
  | Model 2 | `lgbm_gaussian+lgbm`, exposure on | 2.0962 | 2.1202 |

  The full tables are in the tuning plan's D20. Model 2's four LightGBM pairs with the exposure
  on lie within 0.06 on validation, inside the search noise (about 0.02 between seeds); every
  family is worse with the exposure off, the GLMs by about 0.6.
- **The commits:**

  | Phase | Commits |
  |---|---|
  | T1 (Model 1 in the notebook, D9–D15) | `0416267`, `bee781a`, `7ae6314`, `8cc231d`, `a5ae06b`, `6316d91`, `59acae6`, `9eca96e`; handoff `58c3bcb` |
  | T2 (Model 2, D16–D20, T2.5 and D22) | `a2b8693`, `7580304`, `1811551`, `9075dd7`, `10d6e83`, `de94ac8` |
  | T3, D23–D25 and their docs | `15932d7` |
  | The T4 handoff | `e2a4f78`, `6f4b479` |
  | T4a.1 / T4a.2 / T4a.3 (the old package, its docs, the notes) | `5c0908a` / `ed1a770` / `6935416` |
  | T4b.1 / T4b.2 / T4b.2b / T4b.3 (the new package, D26, the notebook's import) | `2765196` / `2bdc808` / `362cab6` / `1370236` |
  | T5.1 / T5.2 (the guide; the plans and this handoff) | `6c89a61` / `b66a8ed` |
  | PR #17 squash-merged (T4a.1–T5.2 in one commit) | `a748cd4` |
  | This handoff's update after the merge | the user's next commit |

  The T4a–T5 hashes above are PR #17's commits: after the squash they are not in
  `feat/experiment-notebook`'s history, and GitHub keeps them under the PR.

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

The items left of [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md)'s PR D, each to be
planned with the user:
1. **The MLflow runs** from the 24 run records and the inspection: the plan's E9 (what a run
   logs) and E10 (inline calls first, a helper module only for what two cells repeat) revisited
   for one study per run cell, as the 2026-10-29 notes there say. The record already holds the
   settings, the trials, the components' declarations, `best_params`, `cv_best` and the score
   table; the data settings and the config hash are printed in section 1.
2. **D4's comparison table** of the runs, from MLflow or from the records.
3. **E11's final evaluation:** the winner rebuilt from its `best_params`
   (`from_params`, `build_*_model`), refit on training + validation, scored once on test with
   the bootstrap of the seven metrics, behind its own button; the validation scores stay the
   selection evidence.

**The prompt for the next session:**

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full, then docs/HYPERPARAMETER_TUNING.md,
docs/MODEL_INSPECTION.md, docs/EXPERIMENT_NOTEBOOK_PLAN.md §3, §4 (E3, E7–E12 and their
2026-10-29 notes) and §9 (D2–D5), and notebooks/02_experiments.py. Verify the state of §2 first
(branch feat/experiment-notebook, HEAD my commit of the handoff update right after a748cd4,
git status clean, the notebook md5 771464b5583902cf0cf8279359ae0f23);
if anything differs, stop and show me the diff. Then plan the items of §6 in order, starting with
the MLflow runs from the run records: plan mode first, re-verify the facts with read-only probes,
give a recommendation with evidence and the trade-off for any design question, write the steps
and the commit message in the chat, and wait for my approval before editing. One sub-task per
approval; I commit myself (never commit, no Co-Authored-By line, never git checkout/stash/
reset/mv).
```
