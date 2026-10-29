# Handoff: the experiment notebook, tuning re-implemented in the notebook

**For:** the session that continues `notebooks/02_experiments.py` with T2.5 (the best runs'
bootstrap intervals and feature importances) and the sub-tasks after it.

**Written:** 2026-10-21, after T2 (Model 2) was completed, reviewed and committed (T2.1–T2.3).
**Updated:** 2026-10-22, after T2.4 (D19: today's design kept), the budget sub-task (D20:
400 trials, a random start of 20) and the T4 design review (D21), all in one session.

**The plan:** [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) holds the sub-tasks, the code
and the checks. Read it in full before the first edit:
- §0 holds the decisions D1–D18. A later decision supersedes an earlier one where it says so.
- §1 holds the rules and the checks A, B and C.
- §2 holds T1 and §3 holds T2.1–T2.3, all done; their code blocks and results are the
  notebook's current state.
- §3 also holds T2.4 and T2.5, which are next.
- §4–§6 hold T3, T4a, T4b and T5; §7 holds the performance facts.

The older plan [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) still holds the
general working rules in its §3, and E3, E7, E8 and E11, which T2.5 brings into the notebook.
Its E4, E5, E11 and D2/D3 are rewritten in T5. Re-verify every fact here against the code of
the day.

## 1. What the task is

On 2026-10-18 the user rejected the `hyperparameter_tuning` package and the notebook section
built on it. Section 5 of the notebook now tunes with plain Optuna and a hand-written
cross-validation loop. Once that code is final, T4 replaces the package with it (D17).

As built for both models, after the user's reviews (D1–D18):

- **Params records** (`LGBMParams`, `PoissonGLMParams`, `NB2Params`, `LogisticParams`):
  frozen, `kw_only`, fields only. The field names are the estimator's keywords.
- **Components:**
  - `CountComponent(name, family, use_exposure, feature_transformer_name,
    hyperparameters)`;
  - `ProbabilityComponent(name, family, calibration_method, feature_transformer_name,
    hyperparameters)`;
  - an abstract `Component` base. `make_estimator(*, seed)` is an if/elif on the family with
    a raising `else`.
  - The family is a `Literal`, checked through `get_type_hints`.
- **Search spaces** (`CountSearchSpace`, `ProbabilitySearchSpace`) hold every Optuna call and
  build nothing.
  - Their methods: `suggest_hyperparameters`, `suggest_feature_transformer_name`, and
    `suggest_calibration_method` (probability only).
  - Their candidate lists are read-only ClassVars: `FEATURE_TRANSFORMER_NAMES` per family,
    and `CALIBRATION_METHODS = (None, "temperature", "sigmoid")` (D16).
- **Builders** assemble components from a trial's draws and turn them into models:
  - `suggest_count_component` and `suggest_probability_component`;
  - `suggest_model1_components` and `suggest_model2_components`;
  - `make_count_model`, `build_model1` and `build_model2`;
  - `suggest_model1` and `suggest_model2`.
- **Runs:** one cell per run, each behind its model's button. The runs go one after another,
  seeded `seed + 0`, `seed + 1`, … in each model's order, and each trial fits its 5 folds in
  parallel. There is no pruning (D19). `n_trials = 400` and `n_startup_trials = 20`, a
  setting logged with the others and a required keyword of `tune_model` (D20).
  - **Model 1:** 8 runs, `(family, use_exposure)`.
  - **Model 2:** 16 runs, `(total_family, probability_family, use_exposure)`. The order is
    total (`lgbm_poisson`, `lgbm_gaussian`, `poisson_glm`, `nb2`) → exposure (on, off) →
    probability (`logistic`, `lgbm`). There is no `MODEL2_TOTALS` list: the run cells
    present are the runs tuned.
  - **A run cell keeps** a record with the settings, `trials`
    (`study.trials_dataframe()`), `components`, `cv_best`, `score_table`, `seconds`, `seed`
    and `y_min`. Only the best components are refit (`fit_and_score`) and scored on training
    and validation.
- **Tables:**
  - `model1_df` / `model1_arms_df`, and `model2_df` / `model2_arms_df`. Model 2's `family`
    is `"<total>+<probability>"`, and `use_exposure` is its own column (D16).
  - The trials-per-choice tables list every candidate, so an untried one shows 0. Model 2
    counts by label, so a `None` calibration is counted.
  - `best_df` gives each model's best run on validation, with `use_exposure`.
- **Metrics** are built in the notebook (D4). The fold scores are combined by an aggregator
  (`src/age_group_prediction/aggregation.py`, `Mean` by default, D9).

**The order of work** (one sub-task per approval, plan mode before each; the user reviews the
design and commits):
1. T2.4 (done 2026-10-22, D19) and the budget decision (done, D20; awaiting commit).
2. **T2.5, done 2026-10-22 (awaiting the user's commit; D22 in the plan):** the best runs'
   bootstrap intervals and feature importances (D18). As built: `BootstrapEvaluator` holds
   the resampling design and `evaluate(..., metrics=[...])` returns a frame (one row per
   metric: `score`, `mean`, `lower`, `upper`); the evaluation and the importances are two
   separate components in the notebook, no `inspect_model`; exact SHAP of 100 validation
   rows against 50 training rows on all cores; three settings `n_repeats`, `n_shap_rows`,
   `n_shap_background`. The plan's T2.5 lists the cells. As planned before:
   - the recipe `inspect_model(model, splits, *, metrics, n_resamples, alpha, n_repeats,
     shap_explainer, random_state)` goes into `src/age_group_prediction/inspection/`, typed
     and tested; it builds one `BootstrapEvaluator` per metric, runs `PermutationImportance`
     per split and `ShapExplainer` once, and returns the intervals frame, the importances
     frame and the `Explanation`. Nothing is added to `BaseAgeGroupModel`;
   - the best model of each run is refit on the whole training part and evaluated on the
     training rows and the validation rows; the test part waits for the refit on training +
     validation (E11);
   - seven metrics, each explained in the header: Poisson deviance, cohort log loss, D²
     Poisson, MAE, RMSE, WMAPE, bias (scikit-learn's functions on the flattened cells
     through a one-line `on_cells`; `wmape` and `bias` one-line notebook functions, having
     no library form);
   - the 24 run records gain `"best_params": _study.best_params`, so the best model is
     rebuilt from them (since D24.3 read back through the search spaces, no `FixedTrial`); new settings `n_repeats`, `n_shap_rows`,
     `n_shap_background` (sizes confirmed by a timing probe: exact SHAP costs 512
     evaluations per row at 9 columns); a button; one explicit cell per model; two tables.
3. T3: the metric constants leave `scoring.py` (done 2026-10-22, awaiting the user's commit).
4. T4a deletes the old package; T4b.1–3 build the new one from the notebook's code (D17,
   D21: six one-way modules; the runs, metrics, settings and tables stay in the notebook).
5. T5: the docs.
6. Then the older items: E11's final evaluation on test; MLflow runs from the run records.

## 2. State of the branch (2026-10-22)

- **Branch:** `feat/experiment-notebook`, HEAD `de94ac8` (T2.5). T3 and the component redesign
  (D23, S1–S6) await the user's commit.
- **Notebook md5:** `d919a7838019a1494d92768f0bafd483` (after the budget sub-task);
  `1e3a9eeb6a3d9f807bdf3a8287212158` after T2.5 (`de94ac8`); `7477a8c9819c29d54f1747b0ef4c6b60` after the
  component redesign of 2026-10-25 (D23 S1–S5, [NOTEBOOK_COMPONENT_REDESIGN_PLAN.md](NOTEBOOK_COMPONENT_REDESIGN_PLAN.md));
  `1ce2db86efa3ffe83f0eb85f6859c772` after the user's edits that followed (renames, `fit_and_score`
  as `@app.function`, marimo-trimmed returns); `df5d373e868e79a9c7323dfe61ffb829` after S6
  (2026-10-26: D23.9 the candidates injected as `feature_transformers` from the section-4 dicts
  `COUNT_FEATURE_TRANSFORMERS` / `PROBABILITY_FEATURE_TRANSFORMERS`, D23.10 the shared draw inside
  the loop, `share_hyperparams`, HEAD's returns restored; Check C the 24 baselines);
  `c42c578ec48f5da3d6e1ae9ff1ba5e1f` after D24 (2026-10-27, the user's notes: `replication_method` fixed at
  `"weighted"`, `share_feature_transformer` (a shared choice drawn once before the loop under `shared__`),
  the best components read from `best_params` with
  `*_components_from_params` and no `FixedTrial`, both sharing flags experiment settings
  (`True` / `False`; the other arm a click), `feature_transformer_dict`, the builders
  `build_independent_cohorts_model` / `build_total_times_probability_model`; NOTEBOOK_TUNING_PLAN.md
  §0 D24.1–D24.5).
  `425b6cf3ff396a478ad08756c2211338` after D25 (2026-10-29, the user's review of D24: the search spaces read nothing
  back and hold `feature_transformer_names`, `suggest_feature_transformer_name`; the component's record is
  the field `estimator_params`, set by the constructor from a trial's draws or by
  `CountComponent.from_params` / `ProbabilityComponent.from_params` from `best_params` (the entries
  under `<name>__` and `shared__`, `Component.set_estimator_params`); the objectives are one loop
  each; the run cells set `_best_components` explicitly; the readers and
  `total_times_probability_search_spaces` removed; NOTEBOOK_TUNING_PLAN.md §0 D25.1–D25.4; the
  notebook test 12/12, Check C the 24 baselines, the shared probe 2.1812).
  T3, the redesign, D24, D25 and the docs committed together as `15932d7` (2026-10-29). T4a done
  2026-10-29 on `refactor/hyperparameter-tuning-package` (draft PR #17 into
  `feat/experiment-notebook`): T4a.1 `5c0908a`, T4a.2 `ed1a770`, T4a.3 `6935416`;
  the notebook unchanged. T4b.1 `2765196`, T4b.2 `2bdc808` (the notebook imports `Mean` from the
  package, md5 `e8a0942fff75f66e6173d6b47ba162a4`), T4b.2b (D26, the user's notes on T4b.2; the
  package only) awaiting the user's commit. **Next: T4b.3**, planned with T5 in
  [NOTEBOOK_T4_HANDOFF.md](NOTEBOOK_T4_HANDOFF.md) (2026-10-29), the handoff the implementing
  session reads first and updates.
- **The commits of the task:**

  | Sub-task | Commit |
  |---|---|
  | T1.1 | `0416267` |
  | T1.2 | `bee781a` |
  | T1.3 | `7ae6314` |
  | T1.4 | `8cc231d` |
  | T1.5 and D9 | `a5ae06b` |
  | T1.6 and D10–D12 | `6316d91` |
  | T1.7 and D13–D14 | `59acae6` |
  | T1.8 and D15 | `9eca96e` |
  | Handoff after T1 | `58c3bcb` |
  | T2.1–T2.2, D16–D18 | `a2b8693` |
  | T2.3 (review, F1–F8) | `7580304` |
  | Handoff after T2 | `1811551` |
  | T2.4 and D19 (docs only) | `9075dd7` |
  | The budget sub-task, D20, D21 (notebook, test, docs) | `10d6e83` |
  | T2.5 and D22 | `de94ac8` |
  | T3, D23–D25 and the docs | `15932d7` |
  | The T4 handoff | `e2a4f78`, `6f4b479` |
  | T4a.1 (the old package's code) | `5c0908a` |
  | T4a.2 (the current guides) | `ed1a770` |
  | T4a.3 (the finished plans, the status lines) | `6935416` |
  | T4b.1 (params, components, search spaces) | `2765196` |
  | T4b.2 (builders, the loop, the aggregators) | `2bdc808` |
  | T4b.2b (D26: components-only suggest functions, `CrossValidationTuner`, `evaluation.py`) | awaiting commit |

  Check `git status` and `git log -3` first; the user may have edited since.
- **Checks at HEAD:**
  - the notebook test passes 5/5;
  - `ruff check notebooks/02_experiments.py` gives 8 findings, all pre-existing: 7 marimo
    `return`s (PLR1711), B018 at line 129, RUF059 `test_age_group` at line 320;
  - `ruff format --diff` shows only the two known hunks, at lines 158 and 213;
  - `marimo check` shows the two known empty-cell issues;
  - mypy is clean;
  - Check C at 3 trials under `PYTHONWARNINGS=error`: 48 s, all 24 baselines below (a
    random start of 20 draws the same first trials as 10, measured).
- **The 400-trial run** (both buttons patched, `PYTHONWARNINGS=error`, 2026-10-22, D20):
  - 3708 s in all: Model 1 821 s, Model 2 2871 s.
  - No failed trial and no warning; every choice tried at least 11 times.
  - `lgbm_gaussian+lgbm` (exposure on) reproduced the planning probe's 2.0962.
  - Model 2's LightGBM-classifier runs took 224–442 s each and its logistic runs 48–89 s.

  | Best run on validation | `family`, exposure | cv_best | val |
  |---|---|---|---|
  | Model 1 | `lgbm_poisson`, on | 2.1280 | 2.1254 |
  | Model 2 | `lgbm_gaussian+lgbm`, on | 2.0962 | 2.1202 |

  The full tables are in the plan's D20. The four LightGBM pairs of Model 2 with the
  exposure on lie within 0.06 on validation, inside the search noise (about 0.02 between
  seeds), so the best pair is not settled by the table. Every family is worse with the
  exposure off: the GLMs by about 0.6 on CV in both models.
- **Check C baselines at `n_trials = 3`** (the regression check). Every change since T1.6 has
  reproduced them exactly.

  | Model 1 `cv_best` | Exposure on | Exposure off |
  |---|---|---|
  | `lgbm_poisson` | 2.1902 | 2.2814 |
  | `lgbm_gaussian` | 2.1906 | 2.3151 |
  | `poisson_glm` | 2.5703 | 2.8776 |
  | `nb2` | 2.2826 | 2.8904 |

  | Model 2 `cv_best` | Exposure on | Exposure off |
  |---|---|---|
  | `lgbm_poisson+logistic` | 2.1543 | 2.1929 |
  | `lgbm_poisson+lgbm` | 2.2212 | 2.2505 |
  | `lgbm_gaussian+logistic` | 2.1318 | 2.1685 |
  | `lgbm_gaussian+lgbm` | 2.1706 | 2.2239 |
  | `poisson_glm+logistic` | 2.2646 | 2.8660 |
  | `poisson_glm+lgbm` | 2.2794 | 2.8828 |
  | `nb2+logistic` | 2.2487 | 2.8540 |
  | `nb2+lgbm` | 2.3039 | 2.9222 |

  Check C of the whole notebook at 3 trials takes about 47–53 s.

  The table above is the arm with both sharing flags `False` (D24.4; the Model 2 values do not
  depend on the flags). The two arms the settings offer (2026-10-27, D24; `n_trials = 3`):

  | Model 1 `cv_best` | `share_hyperparams=True`, transformer per cohort (the default): on / off | both shared: on / off |
  |---|---|---|
  | `lgbm_poisson` | 2.1812 / 2.2193 | 2.1957 / 2.2710 |
  | `lgbm_gaussian` | 2.2177 / 2.2805 | 2.2279 / 2.2805 |
  | `poisson_glm` | 2.4158 / 2.8685 | 2.2655 / 2.8759 |
  | `nb2` | 2.2762 / 2.8707 | 2.2532 / 2.8740 |

## 3. Decisions (do not re-ask)

All are in the plan's §0 with their evidence:
- D1: plain Optuna. D2–D3: the records, as revised by D8, D10 and D11. D4: the metric is
  built in the notebook.
- D6: one study per family and exposure, both arms. LightGBM's loss is a family.
- D7: the facts settled by probe. D9: the aggregator.
- D10–D12: complete components, explicit search spaces, a study per family combination.
- D13–D14: runs one after another with parallel folds, no pruning; one cell per run, every
  trial and the best kept.
- D15: strict checks; untried choices shown as 0.
- D16: the calibration candidates as a ClassVar; counting by label; a family pair as one
  family; `use_exposure` its own column.
- D17: T4 replaces the package, as T4a and T4b.
- D18: the bootstrap and importance in T2.5; no wrapper for the two importance methods.
- D19 (T2.4): today's design stays (folds in parallel, no pruning, `catch=(RuntimeError,)`);
  the measurements in the plan's T2.4.
- D20: `n_trials = 400`, `n_startup_trials = 20`; all 24 runs kept (an elimination rule was
  considered and declined). Never decrease `n_trials`.
- D21: the T4 design (what moves, six one-way modules, T4b in three stops) and the
  evaluation recipe's home: `inspect_model` in the `inspection` package, nothing on the base
  model.
- D25: the search spaces read nothing back; `estimator_params` the component's record;
  `from_params` per component type with the split by `__` (`shared__` folded in) and
  `set_estimator_params` as the check; the objectives one loop each; the run cells set the
  best components explicitly. The T4b module table reflects it.

**Decided by the user, 2026-10-21:**
- `y_min` stays a builder argument, recorded once per run with the seed. The seed is an
  argument of `make_estimator`.
- `evaluate` is not changed. `Aggregator.aggregate` stays a named method (`aggregate_scores` since 2026-10-29, the user's note on T4b.2).
- `num_leaves` above `2**max_depth` (T1's F7) is left as a known inefficiency.
- Model 2 has all 16 run cells and no `MODEL2_TOTALS` list.
- **All 24 runs are kept,** so D6 stands. The GLM totals with the exposure off are about 0.6
  worse on CV in both models, and no run is dropped.
- `scoring.Metric` keeps `direction` and `__call__` (HEAD's form, from `8ee091a`).
- **TPE's random start and budget, decided 2026-10-22 (D20): `n_trials = 400`,
  `n_startup_trials = 20`,** a new experiment setting; all 24 runs kept (an elimination rule
  was considered and declined). The evidence at 400 trials is in the plan's D20; the
  evidence at 200 trials with the exposure on:

  | Run | Random start | Best | Found at trial | Least-tried choice | Time |
  |---|---|---|---|---|---|
  | Poisson GLM | 10 | 2.2490 | 199 | 6, 9, 9 | 25–30 s |
  | Poisson GLM | 40 | 2.2490 | 184 | 9, 10, 11 | 29 s |
  | NB2 | 10 | 2.2428 | 23 | 9, 9, 9 | 112 s |
  | NB2 | 40 | 2.2426 | 123 | 7, 9, 11 | 100 s |
  | Model 2 `lgbm_gaussian+lgbm` (T2.4) | 10 | 2.1066 | 148 | 16 | 97 s |
  | Model 2 `lgbm_gaussian+lgbm` (T2.4) | 40 | 2.1127 | 192 | 18 | 310 s |
  | Model 2 `lgbm_poisson+lgbm` (T2.4) | 10 | 2.0820 | 130 | 16 | 145 s |

  - 400 trials left the Poisson GLM's best at 2.2490. The best so far at trial 100 / 150 /
    200: `lgbm_gaussian+lgbm` 2.1068 / 2.1066 / 2.1066, `lgbm_poisson+lgbm` 2.1102 /
    2.0820 / 2.0820, the Poisson GLM 2.2519 / 2.2496 / 2.2490.
  - A random start of 40 draws the large slow LightGBM models at random (3× the time) and
    ended worse. T2.4's recommendation: keep 200 and 10.
  - Model 2's coverage at 10: the GLM totals' transformer 7–9, logistic's transformer 8–12,
    every LightGBM choice 16 or more, the calibration 12–34.
  - A change re-runs both models at 200 trials (about 30 min) and resets the Check C
    baselines.
  - It is set in `TPESampler(seed=study_seed, n_startup_trials=…)` in `tune_model`.

## 4. How to work

The standing rules are in [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §1 and
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3.

**Each sub-task**
- Plan mode first. Re-verify the plan's facts with read-only probes, summarize the sub-task in
  the chat, and ask for approval.
- Then implement, run the checks, and report file by file. Give the check results and the
  suggested commit message in the chat. Tick the sub-task and update the plan's status line.
  Stop.

**Git and the editor**
- The user commits and pushes. No `Co-Authored-By` line. Never `git checkout`, `stash`,
  `reset` or `mv`. Restore a file from HEAD with `git show HEAD:<path> > <path>` and `cmp`.
- **The user edits between turns, the notebook and `src` alike.** This session found
  `scoring.py` reverted to an older `Metric` (with `greater_is_better`) in the working tree;
  the user wanted HEAD's form restored. Check `git status` and the notebook's md5 before
  each sub-task and before each edit. Show any diff the handoff does not name, and ask.
- **marimo's editor trims each cell's `return`** to the names other cells read whenever the
  user saves. Ruff then jumps to about 31 findings and Check B loses names. The user chose
  to have the returns as at HEAD restored each time, after the diff is shown.
- **Format new cells to `ruff format`'s output; never format the whole file.** For many
  similar cells (T2.1's 16):
  1. generate them with a script into the scratchpad;
  2. run `ruff format` there;
  3. insert them before `if __name__`;
  4. check that the insertion leaves no extra blank line, which would add a third hunk to
     the format diff.
- No new ruff finding (baseline 14 since the user's D24 edits; 8 before). A display cell without a value needs no bare `return`,
  which PLR1711 would flag.

**Probes and checks**
- Probes are read-only: `PYTHONPATH=src .venv/bin/python -W error -c "..."`.
- The T2.5 cells drafted on 2026-10-22 lived in that session's scratchpad only; write them
  again from the plan's T2.5.
- Concurrent `Parallel` calls from threads in one process return wrong scores (T2.4): never
  run two `cv_score` calls at once.
- Scratch files go to the session scratchpad, **never into `notebooks/`**. Review subagents
  write nothing into the repository.
- `timeout` does not exist on macOS: run long commands in the background. The 200-trial run
  of both models takes about 30 min.
- **Check C runs with `PYTHONWARNINGS=error`** (D15). The fold processes do not inherit
  `-W`; the environment variable reaches them.
- **In a probe, a function sent to joblib must not capture the notebook's `defs`**: it fails
  to pickle (`_thread.lock`). Use the notebook's own functions (`partial(suggest_model1,
  …)` pickles).
- **pandas 3.0.5 and `None`:**
  - `None` in a str column becomes NaN, and `value_counts()` drops it.
  - `reindex([None, …])` misses a column where every trial drew `None` (object dtype).
  - Count by label: `fillna("None")`, then reindex on `str(candidate)` (D16).
- `json.dumps(record, default=str)` writes a `NaN` token, which strict JSON readers reject.
  Use `trials.to_json(orient="records")` when records are first saved (T2.3's F3).

**Design reviews.** Expect one at each sub-task; most of T1 and T2 was reshaped by the user.
- Answer a design question with a recommendation, the measured evidence and the trade-off
  (AskUserQuestion with code previews works well). Do not implement before the answer.
- When the user asks "what is the best practice", give the practice, why, and the
  measurement here.
- When the user suggests a change, check it against the measurements first. Example: the
  user expected LightGBM's Gaussian loss to be inferior, but it ties the Poisson loss.
- The user is not familiar with marimo's UI terms (e.g. "button"): explain them.

**The user's style, as asked during T1 and T2:**
- explicit cells, one per run, not loops over runs or lists of settings;
- no function unless it is reused;
- names that say what a thing holds (`score_table`, `y_pred_*`, `feature_transformer_dict`);
- the trial records a registered object's name; the component holds the object and its
  record writes the object's declaration (D23, 2026-10-25; before it, records referred to
  registered objects by name);
- one source of truth for a list of candidates: the feature transformers in the notebook's
  section-4 dicts, injected into the search space (D23.9); the calibration methods a ClassVar of
  the search space;
- no field or check that nothing reads;
- scikit-learn's terms (`make_estimator`);
- if/elif with a raising `else`;
- code that may move to a package reads no notebook global.

## 5. Open items

- **T2.5 done (2026-10-22):** see the plan's D22 and T2.5. Plots of the results are a later
  item; the inspection click costs about one to two minutes (SHAP decides it).
- **What would shorten a click** (62 min at 400 trials) is a modeling choice the user has
  not asked for: fewer calibration inner folds (`calibration_cv`, 5 today) or a lower
  `n_estimators` ceiling (400). Pruning and other parallel places were measured and
  rejected (D19).
- **T2.3's F5:** the two tables cells count with different idioms; left as is.
- **T1's F7:** `num_leaves` above `2**max_depth` changes nothing in LightGBM. Revisit in a
  search-space pass.
- **The user's:** two empty cells and the bare `return`s marimo writes; ask before changing.
- **Stale text in section 4:** its header still says "the check below prints it" and "fitted
  once … as a check", but the check cell is gone. This is the user's text; ask.
- **SPLITTING.md §3** and the simulator plan's Stop 4 doc examples (`.run()` → `simulate()`).
- **The simulator plan's cleanup stop** (dead code) and the format-only commit.
- **A `log(n_apartments)` plan for the GLMs** (FEATURE_TRANSFORMATIONS.md §4.4). Not asked
  for.

## 6. The prompt for the next session

**Since 2026-10-29 the prompt is the one in [NOTEBOOK_T4_HANDOFF.md](NOTEBOOK_T4_HANDOFF.md)**
(T4a.1 next); the prompt below was answered by the planning session that wrote that file and is
kept for the record.

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full, then docs/NOTEBOOK_TUNING_PLAN.md §0 (D1–D25,
D21 and D25 above all), §1 (the rules and the checks A/B/C) and §5–§6 (T4a, T4b.1–3, T5), then
docs/NOTEBOOK_COMPONENT_REDESIGN_PLAN.md §0 and §6 (the history of D23–D24; its code blocks are
superseded by the notebook). Then read notebooks/02_experiments.py from the "## 4) Feature
variants" cell to the end of section 5 (the class and function cells, one Model 1 run cell, one
Model 2 run cell, the two tables cells, the two inspection rebuild cells),
src/age_group_prediction/aggregation.py, src/age_group_prediction/hyperparameter_tuning/ (the old
package T4a deletes) and tests/characterization/test_experiments_notebook.py.

Expected state: branch feat/experiment-notebook, HEAD at my commits of T3, D23, D24, D25 and the
docs (read git log -3; de94ac8 is the last commit before them), a clean tree, notebook md5
`425b6cf3ff396a478ad08756c2211338`. Verify HEAD, git status and the md5 first; if anything differs from
what the handoff §2 names, stop and show me the diff. I commit, push and merge myself: never
commit, never add a Co-Authored-By line, never git checkout/stash/reset/mv.

Task: T4a, T4b.1, T4b.2, T4b.3, T5 of docs/NOTEBOOK_TUNING_PLAN.md §5–§6: delete the old
hyperparameter_tuning package, then rebuild it from the notebook's section 5 as it is now (D21's
six modules; the T4b table lists what each holds after D25), then make the notebook import it,
then the docs. One sub-task per approval. Before each sub-task: plan mode, re-verify the facts
with read-only probes (PYTHONPATH=src .venv/bin/python -W error -c "..."), write the sub-tasks
and the commit message of the sub-task in the chat, and ask for approval before editing. Do not
re-ask the decisions of the plans' §0. Move the notebook's code as it is; if you must change
it, say why with evidence before doing so. When I ask a design question, answer with a
recommendation, the evidence and the trade-off, and wait for my answer before editing. Use best
practices and do not over-complicate: add no mechanism the measurements do not pay for.

At each stop give me: a file-by-file summary; the check results (Check A, B, C with the 24
baselines in the handoff §2's table and the tuning plan's T2.1; the package tests and mypy
where src changes); ruff on the notebook (14 pre-existing findings, re-counted when cells are
removed, no new one); ruff format --diff (only the two known hunks at lines 158 and 213); the
notebook's new md5; and the commit message in the chat. Tick the sub-task and update the
status lines in the plan documents. The machine can be heavily loaded: run every notebook
check in the background with stdin closed, unbuffered and logged to the scratchpad; a Check C
copy outside the repo needs PYTHONPATH=src and configs/ linked beside notebooks/.
```
