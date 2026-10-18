# Handoff: the experiment notebook, tuning re-implemented in the notebook

**For:** the session that continues `notebooks/02_experiments.py`. **Written:** 2026-10-18,
after sub-task T1.1 of the re-implementation. **The plan with the sub-tasks, the code to write
and the checks:** [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) (read it in full before
the first edit; its §0 holds the decisions, §1 the rules and the three checks, §2–§6 the
sub-tasks T1.1–T5, §7 the facts for a later performance pass). The older plan
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3 still holds the general working
rules; its §4 decisions E4, E5, E11 and §9 D2/D3 are superseded and are rewritten in T5.
Re-verify every fact here against the code of the day.

## 1. What the task is

On 2026-10-18 the user rejected the `hyperparameter_tuning` package and the notebook section
built on it (S1–S4 of the previous handoff, commits `a7b8dc5`…`9491a31`): too much machinery
between the notebook and Optuna, and the composites assembled through `set_params` strings
(`cohort_models__<cohort>__…`) instead of being built explicitly. The replacement, approved as a
plan the same day:

- **Plain Optuna and a hand-written cross-validation loop in the notebook.** Nothing of the
  package is imported; the package, its five test files and its four docs are deleted (T4).
- **A trial's choices are frozen dataclasses** (one per estimator holding only its searched
  hyperparameters, no `build`; a component dataclass with the feature-variant key, the
  exposure flag or the calibration; and a composite `Model1Run` / `Model2Run`), each with
  `suggest(trial, prefix, …)` calling `trial.suggest_*` inline; the component and run
  dataclasses have `build(…)` calling the constructors. LightGBM's loss is a family
  (`lgbm_poisson`, `lgbm_gaussian`), not a sampled choice (2026-10-18; plan D6).
  `dataclasses.asdict(run)` is the JSON record.
- **The metric is built in the notebook:** `Metric("cohort_poisson_deviance",
  cohort_poisson_deviance, direction="minimize")`, and `optuna.create_study(direction=metric.direction)`;
  the objective returns the metric's value unchanged. The five module-level constants leave
  `scoring.py` (T3); the `Metric` class and the two functions stay.
- **Pruning and parallelization beyond today's parallel studies are not decided.** Once the
  implementation satisfies the user, a performance pass considers them (plan §7).
- Still in force from 2026-10-15: one joint study per family on the composite; a trial chooses
  per component its exposure and variant (LightGBM's loss as a family, plan D6); `n_trials = 200`; the studies of one
  model in parallel with joblib, study i seeded `seed + i`; only the best run is refit and
  scored on training and validation; Model 2's totals come from the edited list
  `MODEL2_TOTALS`; `split_method = "random"`.

**The order of work** (one sub-task per approval, plan mode before each, the user verifies
each in the IDE and commits): T1.2 → T1.8 (Model 1), T2.1 → T2.3 (Model 2), T3, T4, T5. Then
the performance pass; then the handoff's older items 4–5 (bootstrap and importance check,
MLflow runs).

## 2. State of the branch (2026-10-18)

- **Branch:** `feat/experiment-notebook`, HEAD `8ee091a`. **Uncommitted:** T1.1 of the plan
  (`notebooks/02_experiments.py`, md5 `a2fd2ac858c935ce5a9c7511af996703`), this file and
  `docs/NOTEBOOK_TUNING_PLAN.md`. The user may commit them before the next session starts; check
  `git status` and `git log -3` first.
- **Baseline (before T1.1):** `uv run pytest -m "not slow"`: 1333 passed, 1 skipped, 1 xfailed
  (≈ 105 s). After T1.1 one notebook test fails on purpose until T1.6 (below).
- **The notebook after T1.1:** sections 1–4 unchanged except the dropped `clone` import in
  the section-1 imports cell; section 5 holds four cells: the header (line 598), the imports
  (plain Optuna: `TPESampler`, `FixedTrial`, `TrialState`, `CategoricalDistribution`; joblib;
  the estimators; the five models; `Metric`, `cohort_poisson_deviance`,
  `mean_poisson_deviance`), the two `Metric` objects `COHORT_POISSON_DEVIANCE` and
  `POISSON_DEVIANCE` (line 684), and the constants (`COHORTS`, `PREDICTION_FLOOR`,
  `TREE_VARIANTS`, `COUNT_GLM_VARIANTS`, `PROBABILITY_GLM_VARIANTS`, `COUNT_FAMILIES`,
  `PROBABILITY_FAMILIES`). The user's empty cell before section 5 (lines 593–595) stays.
- **Checks after T1.1:** ruff 8 findings, all pre-existing (7 marimo `return`s PLR1711, one
  B018 at line 129; RUF059 `test_age_group` at line 320); `ruff format --diff` shows only the
  two known hunks in the user's cells (lines 158, 213); `marimo check` the two known
  empty-cell warnings; the `app.run()` defs hold the nine new names.
- **The notebook test** (`tests/characterization/test_experiments_notebook.py`): four of five
  pass. `test_a_tuning_cell_outside_a_run_button_is_caught` requires at least one cell naming
  `tune_model` behind `mo.stop(not …)`; no such cell exists until T1.6, so it fails from T1.1
  to T1.5 by design. Keep the function name `tune_model`, and do not write that string in the
  markdown cell or in comments outside the run cells.
- **`src` is untouched** by the task so far. The S1 `get_params`/`set_params` forwarding in
  `IndependentCohortModels` stays (tested; `clone`-safe); the notebook no longer uses it.

## 3. Decisions settled by probe (2026-10-18; Optuna 4.9.0, joblib 1.5.3; do not re-ask)

- The trial value is the plain mean of the fold scores (KFold on 917 rows: 184/183-row folds).
- The sampler is `TPESampler(seed=study_seed)`; `multivariate=True` raises
  `ExperimentalWarning` under `-W error`.
- The best run is rebuilt with `suggest(optuna.trial.FixedTrial(study.best_trial.params))`:
  exact for `None`/`bool`/`str` choices, and a missing name or a foreign value raises
  `ValueError`. Not a user attribute: notebook classes are pickled by value into the loky
  worker, so an instance coming back is of another class object.
- `study.optimize(..., catch=(RuntimeError,))`: an NB2 fit that does not converge marks its
  trial FAIL and the study goes on. Without `catch` the study stops and `Parallel` discards the
  sibling studies; returning `inf` would hide the traceback. The tables show `n_failed`.
- An in-memory `optuna.Study` pickles back whole from a loky worker (`trials`,
  `best_trial.distributions`, `best_params`), so the worker returns `(study, seconds)`.
- `CountModel` clones its `feature_transformer` at `fit`, so `build` passes the shared
  `FEATURE_VARIANTS[key]` as is.
- "Remove the metrics from scoring" is read as the five constants only. Confirm with the user
  at T3's plan summary.

## 4. How to work

The standing rules are in [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §1 and
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3. In short:
- Plan mode first for each sub-task: re-verify the plan's facts, summarize the sub-task in the
  chat as steps the user can verify, ask for approval, implement, run the sub-task's checks,
  report file by file with the check results and the suggested commit command, stop.
- **Git:** the user commits and pushes. No `Co-Authored-By` line. Never `git checkout`,
  `stash`, `reset` or `mv`; restore a file from HEAD with `git show HEAD:<path> > <path>` and
  `cmp`.
- **The user edits the notebook between turns** in the marimo editor. Record its md5 after
  every edit of yours and `cmp` before the next; if it changed, show the diff and ask.
- marimo's editor trims each cell's `return` to the names other cells read; a new cell that
  reads a name re-adds it to the producing cell's return. An undefined name in a cell fails
  silently in a script run: run the plan's Check B after every notebook edit.
- Format new cells by hand to `ruff format`'s output; never whole-file `ruff format`. No new
  ruff finding (baseline 8).
- Probes are read-only: `PYTHONPATH=src .venv/bin/python -W error -c "..."`. Scratch files
  go to the session scratchpad, never into `notebooks/`. Review subagents write nothing into
  the repository. `timeout` does not exist on macOS: run long commands in the background.
- A plain `.venv/bin/python` does not see `src` (Python 3.13 skips the hidden `.pth`); the
  notebook adds `src` itself, `.env` sets `PYTHONPATH=src`, the notebook test passes it.

## 5. Open items outside this flow

- Two empty cells and the bare `return`s marimo writes are the user's; ask before changing.
- The section-4 header still says "the check below prints it" and "fitted once … as a check";
  the check cell is gone (user's text; ask).
- SPLITTING.md §3 and the simulator plan's Stop 4 doc examples (`.run()` → `simulate()`).
- The simulator plan's cleanup stop (dead code) and the format-only commit.
- A `log(n_apartments)` plan for the GLMs (FEATURE_TRANSFORMATIONS.md §4.4). Not asked for.

## 6. The prompt for the next session

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full, then docs/NOTEBOOK_TUNING_PLAN.md in full
(§0 decisions, §1 rules and the checks A/B/C, §2 the sub-tasks of T1). Then read
notebooks/02_experiments.py from line 590 to the end, src/age_group_prediction/scoring.py,
and the constructors of CountModel, CohortProbabilityModel, IndependentCohortModels and
TotalTimesProbabilityModel in src/age_group_prediction/modeling/.

Expected state: branch feat/experiment-notebook, HEAD 8ee091a or one commit later; T1.1 of
the plan done in the notebook (md5 a2fd2ac858c935ce5a9c7511af996703 if uncommitted). Verify
HEAD, git status and the md5 first; if anything the handoff does not name has changes, stop
and show me the diff. I commit, push and merge myself: never commit, never add a
Co-Authored-By line, never git checkout/stash/reset/mv.

Task: sub-task T1.2 of docs/NOTEBOOK_TUNING_PLAN.md (the estimator dataclasses), then one
sub-task per approval in the plan's order (T1.3 … T1.8, T2, T3, T4, T5). Work in plan mode
for each: re-verify the plan's facts for it with read-only probes
(PYTHONPATH=src .venv/bin/python -W error -c "..."), summarize the sub-task in the chat,
and ask for approval before editing. Do not re-ask the decisions of the plan's §0 or the
handoff's §3. Write the code as the plan gives it; if you must deviate, say why with
measured evidence before doing so.

At each sub-task's stop give me: a file-by-file summary; the results of the plan's checks
for that sub-task (Check A, Check B, and Check C from T1.6 on), ruff on the notebook (8
pre-existing findings, no new one), ruff format --diff (only the two known hunks at lines
158 and 213); at T1.8 and T2.3 an independent review subagent's findings reproduced (tell
it to write nothing into the repository); the notebook's new md5; and the suggested commit
command. Tick the sub-task and update the status line in docs/NOTEBOOK_TUNING_PLAN.md.
```
