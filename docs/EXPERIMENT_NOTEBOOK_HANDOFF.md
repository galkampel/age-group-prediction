# Handoff: the experiment notebook, after the tuning rebuild

**For:** the session that continues [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)
(three PRs: the metrics in `scoring`, the runs in a loop, the objective EDA). PR 1 is merged
(#18), and the experiment-notebook line is merged into `main` (#19, §2). PR 2 is in progress on
`refactor/notebook-run-loop`:
- 2.1 is committed (`9463651`);
- 2.2 is done and awaits the user's commit;
- Model 2 moves to coordinate descent (2.3a, §2a), measured on 2026-10-31.

The next session starts at §6. MLflow and E11's final evaluation are deferred.

**Written:** 2026-10-29, after T5 of [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md);
**updated** 2026-10-30 after sub-task 1.1 of the metrics plan, after the user's notes on its
helpers, after sub-tasks 1.2, 1.3 and 1.4, and after the merges of #18 and #19 (§2, §6);
**updated** 2026-10-31 after 2.1, 2.2 and the coordinate-descent measurement (§2, §2a, §4, §6). The detailed history of
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

## 2. State (2026-10-31, end of day; the 2026-10-30 entries follow)

- **PR 2 so far** (the metrics plan §4 holds each sub-task's "Done" note):
  - **2.1 is committed as `9463651`:** one test in `test_hyperparameter_tuning_builders.py`.
    For each of the four flag pairs, a changed `max_depth` and transformer reach every fitted
    cohort model when shared, and only kindergarten's when not. The plan's first mutation
    could not fail (a shared and a cohort key never coexist), so the mutation moved to the
    build, at the user's choice.
  - **2.2 is done and NOT committed.** The user commits it first. The tree holds its three
    files:
    - `notebooks/02_experiments.py`: `MODEL1_RUN_SETTINGS` (8 entries, the flags per run), one
      loop cell for Model 1, the tables cell reading `MODEL1_RUNS`, the flags out of
      `EXPERIMENT_SETTINGS`, three sentences of section 5 in the user's approved wording;
    - `tests/characterization/test_experiments_notebook.py`;
    - `docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md`.

    Its commit message:
    `refactor(notebook): the independent cohorts model's runs in one loop; the sharing flags are run settings`.
  - **The notebook's md5 with 2.2:** `e5fea5dcd691aa1e69a94769f4cff3d5`. Read it through `cat`
    (`cat notebooks/02_experiments.py | md5`): on an iCloud-offloaded file, `md5 -q` printed
    the empty file's hash (§4).
  - **Checks at 2.2:**
    - the non-slow suite 1377 passed, 1 skipped, 1 xfailed, 19 deselected; mypy clean (29 files);
    - ruff on the notebook the same 14 findings; the format diff the user's two hunks;
      `marimo check` the one empty cell;
    - Check C (new recipe, §4): Model 1 gives the `share_hyperparams=True` baselines below,
      with the 8 flags set to `False` the first table, and Model 2's 16 are unchanged.
- **The environment was rebuilt on 2026-10-31 (§4, iCloud):**
  - `.venv` was deleted and rebuilt with `uv sync --frozen --all-groups`: the same lock and
    versions, every group (marimo, ruff, pytest, mypy, mlflow);
  - a second, identical environment outside iCloud is at `~/.venvs/age-group-prediction`.
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
- **Check C baselines** (the regression check: the clicked run on a copy with `n_trials = 3`
  and both sharing flags off, under `PYTHONWARNINGS=error`; `cv_best`). Since 2.2 the flags are
  turned off in the 8 entries of `MODEL1_RUN_SETTINGS` (§4). Every change since T1.6 has
  reproduced the baselines to the fourth decimal, PR 1 and 2.2 included. Model 2's 16 will be
  re-recorded when its cells move to coordinate descent (§2a, §6).

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

## 2a. Model 2 by coordinate descent (decided 2026-10-31)

**The user's decision:** the notebook tunes Model 2 in two steps.
1. Tune the total, with the probability model fixed.
2. Tune the probability model, with the best total fixed.

Then each of the 16 pairs is built from its two winners and scored, with no search. **The
implementation that tunes both components at once stays** in `src`:
`CrossValidationTuner.tune_total_times_probability_model`, `suggest_total_times_probability_components`
and their tests. Nothing is removed. It was asked because Model 2's joint search takes 48 min
at 400 trials. T2.4 / D19 had already measured and rejected pruning and more parallelism (slower
or unsafe), and the user declined splitting into two objectives (a total deviance and a log
loss).

**Why one round is exact** (probed on the notebook's 5 folds, ≤ 3e-15 on each):
- Model 2's composite Poisson deviance is `D_total / K + 2 (C + N · log_loss) / (n K)`, where:
  - `D_total` is the total's deviance;
  - `log_loss` is the probabilities' per-child multinomial log loss;
  - `C` depends on `Y` alone.
- The prediction is `probabilities × total`, and `y_min` floors the total only. No
  hyperparameter reaches both parts.
- So the joint optimum is the pair of the two separate optima, and the second step cannot
  change the first's result.
- Within a step, the fixed part adds the same constant to every trial's value. Step 1 may
  therefore fix the cheapest classifier: logistic, `C = 1.0`, no calibration, transformer
  `glm_base`.
- Each step scores the full composite, as today. Only what is searched changes.

**Reuse:**
- step 1: 8 studies (4 total families × exposure on / off), seeded `seed + 0 … 7` in Model 1's
  order of families;
- step 2: 2 studies (logistic, lgbm), seeded `seed + 8`, `seed + 9`. The exposure does not reach
  the classifier;
- the 16 pairs: each one's `cv_best` is `tuner.cv_score` of the pair, comparable with a joint
  `best_value`, plus `fit_and_score`.

**Measured (2.3a.1, 400 trials, `PYTHONWARNINGS=error`; a scratchpad probe built from the
package's public pieces).** Today's joint `lgbm_gaussian+lgbm` (exposure on, `seed + 5`)
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
- Optuna's `PartialFixedSampler` raises `ExperimentalWarning` under `-W error`, so the fixing is
  done in the package's code.

**Not yet recorded in the metrics plan:** 2.2's uncommitted changes are in the same file, so the
entry waits for the user's 2.2 commit (§6, step 1).

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

**Decided by the user on 2026-10-31** (PR 2):
- the sharing flags are run settings, in `MODEL1_RUN_SETTINGS`, its own cell before the button
  (M6);
- Model 1's runs are one loop (M7);
- Model 2 is tuned by coordinate descent in the notebook, and the joint implementation stays in
  `src` (§2a);
- the user's wish for this work: do not overcomplicate the implementation; the smallest change
  that works.

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
  header's markdown holds the string too), and each of the 8 lines
  `            "share_hyperparams": True,` in `MODEL1_RUN_SETTINGS` replaced by `False` (assert
  the count, 8; before 2.2 it was the one line `    share_hyperparams = True`). Also make
  `<scratch>/project/configs` a symlink to the repo's `configs`; run it with `marimo.ui.run_button`
  patched to a clicked value, `PYTHONWARNINGS=error`, `PYTHONPATH=src`, in the background. It
  takes 80–170 s with the inspection; the 24 values must equal §2's.
- The machine can be heavily loaded: run notebook checks and the suite in the background with
  stdin closed, unbuffered, logged to the scratchpad; `timeout` does not exist on macOS.

**iCloud (found 2026-10-31):** the repo is in `~/Desktop`, which iCloud Drive syncs with storage
optimization. macOS offloads files ("dataless": the name stays, the content is downloaded on
read). On 2026-10-31 it had offloaded:
- about 19,600 `.venv` files;
- 43 source files, even ones edited that hour, the notebook among them;
- 1,152 of 2,562 `.git` files.

A read of an offloaded file blocks until iCloud delivers it (seconds to 30+ s each). That hung
a 400-trial probe on an `import` and hung `git diff`. Precautions:
- **Before a long run or a git command,** check `find . -flags +dataless -type f | wc -l`. A
  stalled process shows 0% CPU, its main thread is in `read()` (`sample <pid>`), and
  `lsof -p <pid>` names the file.
- **To bring a file back,** read it (`cat file > /dev/null`). `brctl download` did nothing.
- **Long runs use `~/.venvs/age-group-prediction/bin/python`,** an identical environment outside
  iCloud (`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/age-group-prediction uv sync --frozen --all-groups`),
  with `PYTHONPYCACHEPREFIX=<scratch>/pycache`, so offloaded `__pycache__` files are never read.
  Compile that cache once without strict warnings, `python -m compileall -q
  <site-packages> src`. Under `PYTHONWARNINGS=error`, a fresh compile of `pyro` raises a
  `SyntaxError` (an invalid escape in a docstring).
- **`uv sync` needs `--all-groups`:** marimo, ruff, pytest and mlflow are dependency groups.
- **The lasting fix is the user's:** Finder's "Keep Downloaded" on the folder, turning off
  Optimize Mac Storage, or moving the repo out of `~/Desktop`.

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
- **What would shorten a click** (62 min at 400 trials): Model 2's coordinate descent (§2a)
  takes Model 2 from 48 to about 15 min. Still open, and not asked for:
  - dropping the GLM exposure-off arms (structurally worse by about 0.6: the GLMs see
    `n_apartments` only as the exposure);
  - narrowing LightGBM's search after measuring its fANOVA importances (`num_leaves` above
    `2**max_depth`; `learning_rate` × `n_estimators`; the calibration's cost of ×5);
  - fewer trials once the space is narrower.

  The user asked about all of these on 2026-10-31 and chose coordinate descent first.

## 6. Next, and the prompt

The order is [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)'s, reshaped
for Model 2 on 2026-10-31. One sub-task per approval:
1. **The user commits 2.2.** First check that nothing is offloaded
   (`find . -flags +dataless -type f | wc -l`, §4), or the commit may stall on `.git`.
2. **2.3a.1, recorded in the plan (docs only).** A new 2.3a entry in the metrics plan's §4:
   - the decision, the measurement and the table of §2a;
   - the reshaped 2.3a.2 and 2.3 below;
   - the status line.

   Commit: `docs(notebook): Model 2 by coordinate descent measured against the joint studies`.
3. **2.3a.2, the `src` change, the smallest that works.**
   - **The change:** the tuner can tune one part of Model 2 with the other part's component
     given and fixed. The draws move into `suggest_total_component` and
     `suggest_probability_component`, and `suggest_total_times_probability_components` calls
     both, so the joint draws and every baseline stay. **The joint method stays.**
   - **The design question for the user, with code previews:** two named tune methods (one per
     step, D26's pattern; the session's recommendation) or one method with an optional fixed
     component.
   - **Tests:** a step's trials hold only that part's keys; the fixed component is the one in
     the model that is scored; the joint method's tests still pass.
4. **2.3 (reshaped), Model 2's runs in loops, by coordinate descent.**
   - **Settings:** `MODEL2_TOTAL_RUN_SETTINGS` (8 entries, `seed + 0 … 7`),
     `MODEL2_PROBABILITY_RUN_SETTINGS` (2 entries, `seed + 8`, `seed + 9`), and the fixed
     classifier of step 1 (§2a).
   - **The cells:** a loop for step 1, a loop for step 2 (the best step-1 total by cv fixed),
     then the 16 pairs. Each pair's record keeps today's keys, so `model2_df`, `best_df` and the
     inspection are unchanged. Its `best_params` merge the two steps' params, and its `cv_best`
     is `tuner.cv_score` of the pair.
   - **The arms table** counts the 10 step studies' trials.
   - **Text and checks:** section 5's text is updated with the user's approval. Model 2's 16
     Check C baselines are re-recorded (3 trials), and Model 1's 8 must be unchanged.
5. **2.4:** `fit_and_score(model, train, val, *, metrics)`, every metric per cohort and for the
   composite (the user's amendment: one `POISSON_DEVIANCE`, `LOG_LOSS`, no `COHORT_` prefixes).
6. **2.5:** the docs (the decisions M6, M7 and Model 2's coordinate descent into the tuning
   plan's §0; HYPERPARAMETER_TUNING.md's new methods and `fit_and_score`; this handoff's checks),
   then the PR's review subagent. Then the user opens PR 2 as a draft into `main`.
7. **PR 3** (`feat/objective-eda`, from `main` after PR 2 is merged): the objective EDA.
8. **Deferred:** the MLflow runs (the D1e guard first: with `MLFLOW_TRACKING_URI` unset, MLflow
   3.16 writes to the repo's canonical `mlflow.db` and its artifacts to `./mlruns`), D4's
   comparison table, and E11's final evaluation.

The user's preferences for this work: PR descriptions carry no "Generated with Claude Code" line
and commit messages no `Co-Authored-By`; plan docs live in `docs/`, split into sub-tasks the user
validates one at a time; the user is new to marimo's UI terms (explain "button", "click"); keep
the implementation simple.

**The prompt for the next session:**

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full (§2a and §4's iCloud notes first), then
docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md (§4, PR 2), and
src/age_group_prediction/hyperparameter_tuning/cross_validation.py and builders.py. Verify the
state first:
- branch refactor/notebook-run-loop;
- HEAD my commits of 2.2 and of this handoff on top of 9463651 (if either is uncommitted,
  stop and tell me);
- a clean tree;
- the notebook md5 e5fea5dcd691aa1e69a94769f4cff3d5, read via `cat notebooks/02_experiments.py | md5`;
- no iCloud-offloaded files (`find . -flags +dataless -type f | wc -l` is 0; otherwise stop and
  tell me).
If anything differs, stop and show me the diff. Then do step 2 of the handoff's §6 (record 2.3a.1
in the plan, docs only) and wait for my commit. Then plan 2.3a.2 in plan mode: re-verify the
facts with read-only probes, give a recommendation with evidence and the trade-off for any
design question (code previews), write the steps and the commit message in the chat, and wait
for my approval before editing. Keep the implementation simple and do not remove the joint
tuning of both components. Then each later sub-task in order, one per approval. I commit myself:
never commit, no Co-Authored-By line, no "Generated with Claude Code" in PR text, never git
checkout/stash/reset/mv.
```
