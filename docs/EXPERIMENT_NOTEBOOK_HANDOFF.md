# Handoff: the experiment notebook, tuning re-implemented in the notebook

**For:** the session that continues `notebooks/02_experiments.py` with Model 2 (T2.1 onward).

**Written:** 2026-10-21, after T1 (Model 1) was completed and committed (T1.1–T1.8).

**The plan:** [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md), with the sub-tasks, the code
and the checks. Read it in full before the first edit:
- §0 holds the decisions D1–D15; a later decision supersedes an earlier one where it says
  so.
- §1 holds the rules and the checks A, B and C.
- §2 holds T1, done; its code blocks are the notebook's current code.
- §3–§6 hold T2–T5.

The older plan [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3 still holds the
general working rules; its §4 decisions E4, E5, E11 and §9 D2/D3 are superseded and are
rewritten in T5. Re-verify every fact here against the code of the day.

## 1. What the task is

On 2026-10-18 the user rejected the `hyperparameter_tuning` package and the notebook section
built on it. In its place, section 5 of the notebook tunes with plain Optuna and a
hand-written cross-validation loop; the package is deleted at T4. As built for Model 1, after
the user's reviews (D1–D15):

- **Params records** (`LGBMParams`, `PoissonGLMParams`, `NB2Params`, `LogisticParams`):
  frozen, `kw_only`, fields only. The field names are the estimator's keywords. They catch a
  LightGBM typo, which LightGBM itself accepts silently.
- **Components** are complete, saveable records of one part of a run, built by hand or from a
  trial:
  - `CountComponent(name, family, use_exposure, feature_transformer_name, hyperparameters)`;
  - `ProbabilityComponent(name, family, calibration_method, feature_transformer_name,
    hyperparameters)`;
  - an abstract `Component` base. `make_estimator(*, seed)` is an if/elif on the family with
    a raising `else`.
  - The family is a `Literal`, checked through `get_type_hints`. The transformer is a name in
    `FEATURE_TRANSFORMERS`, resolved by the builder.
- **Search spaces** (`CountSearchSpace(name, family)`, `ProbabilitySearchSpace(name,
  family)`) hold every Optuna call and build nothing:
  - `suggest_hyperparameters`, `suggest_feature_transformer_name`, and
    `suggest_calibration_method` (probability only).
  - The candidate transformer names per family are a read-only ClassVar,
    `FEATURE_TRANSFORMER_NAMES`.
  - Trial names are `<name>__<parameter>`.
- **Builders** assemble components from a trial's draws and turn them into models; they are
  the objective and the refit alike:
  - `suggest_count_component`, `suggest_probability_component`;
  - `suggest_model1_components`, `suggest_model2_components`;
  - `make_count_model`, `build_model1`, `build_model2`;
  - `suggest_model1`, `suggest_model2`.
- **A run** is one family combination and exposure:
  - Model 1: `(family, use_exposure)`, 8 runs.
  - Model 2: `(total_family, probability_family, use_exposure)`.
  - **Each run is its own cell,** behind the shared run button. The runs go one after
    another, run i seeded `seed + i`.
  - **Each trial fits its 5 folds in parallel:** `fold_score`, then `cv_score(…, n_jobs)`.
    No pruning.
  - **The fold scores are combined by an aggregator:** `src/age_group_prediction/aggregation.py`,
    `Mean` by default (D9).
  - **A run cell keeps** `{family, use_exposure, trials (study.trials_dataframe()),
    components, cv_best, score_table, seconds, seed, y_min}`. Only the best components are
    refit (`fit_and_score`) and scored on training and validation.
  - **A tables cell** collects the runs into `model1_df`, and `model1_arms_df`: the completed
    trials per feature transformer, every candidate listed, so 0 shows.
- **Metrics** are built in the notebook (D4): `COHORT_POISSON_DEVIANCE` for the objective and
  the composite, `POISSON_DEVIANCE` per cohort. `evaluate` stays `metric(y_true, y_pred)`.

**The order of work** (one sub-task per approval, plan mode before each, the user reviews the
design and commits):
1. T2.1 → T2.3 (Model 2);
2. then the `n_startup_trials` decision (§3);
3. T3, T4, T5;
4. then the older items: the bootstrap and importance check, then MLflow runs from the run
   records.

## 2. State of the branch (2026-10-21)

- **Branch:** `feat/experiment-notebook`, HEAD `9eca96e` (T1.8), tree clean.
- **Notebook md5:** `9d49fe78f99d7ff4d96e17742ea7d277`.
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

  Check `git status` and `git log -3` first; the user may have edited since.
- **Checks at HEAD:**
  - notebook test 5/5;
  - `ruff check notebooks/02_experiments.py`: 8 findings, all pre-existing (7 marimo
    `return`s PLR1711, B018 at line 129, RUF059 `test_age_group` at line 320);
  - `ruff format --diff`: only the two known hunks at lines 158 and 213;
  - `marimo check`: the two known empty-cell issues;
  - mypy clean;
  - non-slow suite at T1.5: 1357 passed, 1 skipped, 1 xfailed.
- **New in `src`:**
  - `src/age_group_prediction/aggregation.py` (D9: the user's `Aggregator`, `Mean`,
    `WeightedMean`, `UpperBound`, moved out of the package) and
    `tests/unit/test_aggregation.py`;
  - mypy's `files` and strict overrides list the module.
  - T4 keeps these and deletes the package's copies.
- **Model 1 at 200 trials** (the clicked path, measured 2026-10-21): 380 s, no failed trial.
  Every transformer was tried in every run and cohort: the least-tried got 6–9 trials in the
  GLM families and 17 or more in LightGBM.

  | Run, exposure on | cv_best | val (composite) |
  |---|---|---|
  | `lgbm_poisson` | 2.1459 | 2.1604 |
  | `lgbm_gaussian` | 2.1323 | 2.1646 |
  | `poisson_glm` | 2.2490 | 2.1882 |
  | `nb2` | 2.2428 | 2.1962 |

  Every family is worse with the exposure off, the GLMs by about 0.85 on validation.
- **Check C baseline at `n_trials = 3`** (the regression check): every redesign since T1.6
  reproduced these 8 best values exactly.

  | Family | Exposure on | Exposure off |
  |---|---|---|
  | `lgbm_poisson` | 2.1902 | 2.2814 |
  | `lgbm_gaussian` | 2.1906 | 2.3151 |
  | `poisson_glm` | 2.5703 | 2.8776 |
  | `nb2` | 2.2826 | 2.8904 |

- **Model 2's builders exist and were probed** (T1.8's reviewer):
  - (`poisson_glm` total, `logistic`) and (`poisson_glm`, `lgbm`) build, rebuild from
    `FixedTrial` and fit, with composite validation 2.195 and 2.358 at one trial.
  - There are no Model 2 run cells yet.

## 3. Decisions (do not re-ask)

All are in the plan's §0 with their evidence:
- D1: plain Optuna.
- D2–D3: the records, as revised by D8, D10 and D11.
- D4: the metric is built in the notebook.
- D6: one study per family and exposure, both arms; LightGBM's loss as a family.
- D7: the facts settled by probe.
- D9: the aggregator.
- D10–D12: complete components, explicit search spaces, a study per family combination,
  `feature_transformer_dict`.
- D13: runs one after another with parallel folds, no pruning.
- D14: one cell per run; every trial and the best kept.
- D15: strict checks; untried transformers shown as 0.

**Decided by the user, 2026-10-21:**
- `y_min` stays a builder argument, recorded once per run with the seed.
- The seed is an argument of `make_estimator`, not a field.
- `evaluate` is not changed. `cohort_log_loss` needs the matrix, and the models predict the
  target's column order.
- `Aggregator.aggregate` stays a named method, not `__call__`.
- `num_leaves` above `2**max_depth` (finding F7) is left as a known inefficiency.

**Pending, to decide after Model 2 (after T2.2): TPE's random start, `n_startup_trials`, 40
or 10 (today the default, 10).**

Put the choice to the user with this evidence, measured 2026-10-21 at 200 trials, exposure
on:

| Run | Random start | Best | Found at trial | Least-tried transformer per cohort | Time |
|---|---|---|---|---|---|
| Poisson GLM | 10 | 2.2490 | 199 | 6, 9, 9 | 25 s |
| Poisson GLM | 40 | 2.2490 | 184 | 9, 10, 11 | 29 s |
| NB2 | 10 | 2.2428 | 23 | 9, 9, 9 | 112 s |
| NB2 | 40 | 2.2426 | 123 | 7, 9, 11 | 100 s |

- **More trials are not needed:** 400 trials left the Poisson GLM's best at 2.2490 (found at
  trial 296).
- **40 makes almost no difference to the best** (at most 0.0002). It evens the Poisson GLM's
  coverage but not NB2's, and costs no time.
- **The cost of changing it:** every seeded result changes, so Model 1 and Model 2 must be
  re-run at 200 and a new Check C baseline recorded.
- **Where:** `TPESampler(seed=study_seed, n_startup_trials=…)` in `tune_model` (T1.5's cell).

Add the Model 2 runs' coverage to the evidence before asking.

## 4. How to work

The standing rules are in [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §1 and
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3. In short:

**Each sub-task**
- Plan mode first. Re-verify the plan's facts with read-only probes, summarize the sub-task in
  the chat, and ask for approval.
- Then implement, run the checks, and report file by file with the check results and the
  suggested commit command. Tick the sub-task and update the plan's status line. Stop.

**Git and the editor**
- The user commits and pushes. No `Co-Authored-By` line. Never `git checkout`, `stash`,
  `reset` or `mv`. Restore a file from HEAD with `git show HEAD:<path> > <path>` and `cmp`.
- The user edits the notebook between turns. Record its md5 after every edit of yours and
  check it before the next; if it changed, show the diff and ask.
- **marimo's editor trims each cell's `return`** to the names other cells read whenever the
  user saves. Ruff then jumps to about 31 findings and Check B loses names. The user chose
  (2026-10-21) to have the returns as at HEAD restored each time, after the diff is shown.
- Format new cells to `ruff format`'s output by formatting a scratch copy; never format the
  whole file. No new ruff finding (baseline 8). A display cell without a value needs no
  bare `return`, which PLR1711 would flag.

**Probes and checks**
- Probes are read-only: `PYTHONPATH=src .venv/bin/python -W error -c "..."`.
- Scratch files go to the session scratchpad, **never into `notebooks/`**. Review subagents
  write nothing into the repository.
- `timeout` does not exist on macOS: run long commands in the background.
- **Check C runs with `PYTHONWARNINGS=error`** (D15). The fold processes do not inherit
  `-W`; the environment variable reaches them.
- **In a probe, a function sent to joblib must not capture the notebook's `defs`**: it fails
  to pickle (`_thread.lock`). Pass the data in as arguments, or use the notebook's own
  functions (`partial(suggest_model1, …)` pickles).

**Design reviews.** Expect one at each sub-task; most of T1 was reworked after the user read
it.
- Answer a design question with a recommendation, the measured evidence and the trade-off
  (AskUserQuestion with code previews works well), and do not implement before the answer.
- When the user asks "what is the best practice", give the practice, why, and the
  measurement here.
- The user is not familiar with marimo's UI terms (e.g. "button"): explain them.

**The user's style, as asked during T1:**
- explicit cells, one per run, not loops over runs;
- no function unless it is reused;
- names that say what a thing holds (`score_table`, `y_pred_*`, `feature_transformer_dict`);
- records that refer to registered objects by name;
- no field or check that nothing reads;
- scikit-learn's terms (`make_estimator`);
- if/elif with a raising `else`;
- code that may move to a package reads no notebook global.

## 5. Open items outside this flow

- **T2.1's first question:** with one cell per run, how the Model 2 totals are chosen after
  Model 1's table: by which run cells exist, or by an edited list. Settle it in T2.1's plan.
- **F7:** `num_leaves` above `2**max_depth` changes nothing in LightGBM (D15). Revisit in a
  search-space pass.
- **The user's:** two empty cells and the bare `return`s marimo writes; ask before changing.
- **Stale text in section 4:** its header still says "the check below prints it" and "fitted
  once … as a check", but the check cell is gone (the user's text; ask).
- **SPLITTING.md §3** and the simulator plan's Stop 4 doc examples (`.run()` → `simulate()`).
- **The simulator plan's cleanup stop** (dead code) and the format-only commit.
- **A `log(n_apartments)` plan for the GLMs** (FEATURE_TRANSFORMATIONS.md §4.4). Not asked
  for.

## 6. The prompt for the next session

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full, then docs/NOTEBOOK_TUNING_PLAN.md: §0 (the
decisions D1–D15), §1 (the rules and the checks A/B/C), §3 (T2), §4 (T3), §5 (T4) and §6 (T5);
§2 holds T1 and the notebook's current code. Then read notebooks/02_experiments.py section 5
(from the "## 5) Hyperparameter tuning" cell to the end), src/age_group_prediction/scoring.py,
src/age_group_prediction/aggregation.py, and the Model 2 classes in
src/age_group_prediction/modeling/ (CohortProbabilityModel, TotalTimesProbabilityModel).

Expected state: branch feat/experiment-notebook, HEAD 9eca96e (T1.8) or a later commit of
mine, a clean tree, notebook md5 9d49fe78f99d7ff4d96e17742ea7d277. Verify HEAD, git status and
the md5 first; if anything has changes the handoff does not name, stop and show me the diff.
I commit, push and merge myself: never commit, never add a Co-Authored-By line, never git
checkout/stash/reset/mv.

Task: sub-task T2.1 of docs/NOTEBOOK_TUNING_PLAN.md (Model 2's run cells); its first
question is how the total families are chosen now that each run has its own cell (handoff
§5). Then one sub-task per approval in the plan's order (T2.2, T2.3, then the n_startup_trials
decision of handoff §3, then T3, T4, T5). Work in plan mode for each: re-verify the plan's
facts with read-only probes (PYTHONPATH=src .venv/bin/python -W error -c "..."), summarize the
sub-task in the chat, and ask for approval before editing. Do not re-ask the decisions of the
plan's §0 or the handoff's §3. Write the code as the plan gives it; if you must deviate, say
why with measured evidence before doing so. I review the design of each sub-task: when I ask
a design question, answer with a recommendation, the evidence and the trade-off, and wait for
my answer before editing.

At each sub-task's stop give me: a file-by-file summary; the results of the plan's checks for
that sub-task (Check A, Check B, and Check C run under PYTHONWARNINGS=error); ruff on the
notebook (8 pre-existing findings, no new one); ruff format --diff (only the two known hunks
at lines 158 and 213); at T2.3 an independent review subagent's findings reproduced (tell it
to write nothing into the repository); the notebook's new md5; and the suggested commit
command. Tick the sub-task and update the status line in docs/NOTEBOOK_TUNING_PLAN.md. After
T2.2, put the n_startup_trials decision (40 or 10) to me with the evidence of handoff §3 and
Model 2's coverage.
```
