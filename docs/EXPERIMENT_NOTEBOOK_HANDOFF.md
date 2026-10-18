# Handoff: the experiment notebook, tuning re-implemented in the notebook

**For:** the session that continues `notebooks/02_experiments.py`. **Written:** 2026-10-18,
after sub-task T1.3 of the re-implementation (T1.1–T1.3 committed). **The plan with the sub-tasks, the code to write
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
- **Frozen dataclasses, no `build` on any** (plan D2, D3): one per estimator holding only its
  searched hyperparameters, with `suggest(trial, prefix, …)`; and a component holding a
  study's fixed settings, `CountComponent(family, use_exposure)` /
  `ProbabilityComponent(family)` (a `Literal` family), whose `suggest_*` methods sample the
  family's params and the feature transformer. The family → transformer-names mapping
  (`COUNT_FAMILY_TRANSFORMERS`, the one place a family is paired with its transformers) is an
  argument of `suggest_feature_transformer`, so the classes read no notebook global. The
  objective builds the model explicitly, in `suggest_model1` / `suggest_model2` (estimator,
  then `CountModel` / `CohortProbabilityModel`, then the composite). The best run's refit calls the same function on `FixedTrial(best_params)`.
  LightGBM's loss is a family (`lgbm_poisson`, `lgbm_gaussian`), and the exposure is a study
  setting with both arms (2026-10-18; plan D6). The record is
  `{"settings": asdict(component), "params": study.best_params}`.
- **The metric is built in the notebook:** `Metric("cohort_poisson_deviance",
  cohort_poisson_deviance, direction="minimize")`, and `optuna.create_study(direction=metric.direction)`;
  the objective returns the metric's value unchanged. The five module-level constants leave
  `scoring.py` (T3); the `Metric` class and the two functions stay.
- **Pruning and parallelization beyond today's parallel studies are not decided.** Once the
  implementation satisfies the user, a performance pass considers them (plan §7).
- Still in force from 2026-10-15: one joint study per family on the composite; a trial chooses
  per component its feature transformer and hyperparameters (LightGBM's loss a family, the
  exposure a study setting; plan D6); `n_trials = 200`; the studies of one model in parallel
  with joblib, study i seeded `seed + i`; only the best run is refit and
  scored on training and validation; Model 2's totals come from the edited list
  `MODEL2_TOTALS`; `split_method = "random"`.

**The order of work** (one sub-task per approval, plan mode before each, the user verifies
each in the IDE and commits): T1.4 → T1.8 (Model 1), T2.1 → T2.3 (Model 2), T3, T4, T5. Then
the performance pass; then the handoff's older items 4–5 (bootstrap and importance check,
MLflow runs).

## 2. State of the branch (2026-10-18, end of day)

- **Branch:** `feat/experiment-notebook`, HEAD `7ae6314` (T1.3), tree clean, notebook md5
  `389df50917ba2b32c4f949f252888ae2`. Commits of the task: `0416267` T1.1, `bee781a` T1.2,
  `7ae6314` T1.3. Check `git status` and `git log -3` first; the user may have edited the
  notebook since.
- **Baseline (before T1.1):** `uv run pytest -m "not slow"`: 1333 passed, 1 skipped, 1 xfailed
  (≈ 105 s). One notebook test fails on purpose until T1.6 (below).
- **Section 4:** its dict of feature transformers is `FEATURE_TRANSFORMERS` (renamed from
  `FEATURE_VARIANTS` on 2026-10-18; its text is the user's and unchanged).
- **Section 5 today** (from line 591; T1.4 appends after the last cell):
  1. the user's empty cell (591), kept;
  2. the header (596, `hide_code=True`);
  3. the imports (630): plain Optuna (`TPESampler`, `FixedTrial`, `TrialState`,
     `CategoricalDistribution`), joblib, the estimators, the five models, `Metric`,
     `cohort_poisson_deviance`, `mean_poisson_deviance`, `Literal`, `get_args`;
  4. the metrics `COHORT_POISSON_DEVIANCE`, `POISSON_DEVIANCE` (690);
  5. the constants (703): `COHORTS`, `PREDICTION_FLOOR`, the transformer-name tuples
     `TREE_TRANSFORMERS`, `COUNT_GLM_TRANSFORMERS`, `PROBABILITY_GLM_TRANSFORMERS`; the
     families `CountFamily` / `ProbabilityFamily` (plain `Literal`s) and
     `COUNT_FAMILIES = get_args(CountFamily)` / `PROBABILITY_FAMILIES`; the family → names
     mappings `COUNT_FAMILY_TRANSFORMERS`, `PROBABILITY_FAMILY_TRANSFORMERS`;
  6. the estimator params (762): `LGBMParams` (10 boosting fields, `suggest(trial, prefix,
     **fixed)`), `LGBMRegressorParams` (+ `objective`, fixed by the family),
     `LGBMClassifierParams`, `PoissonGLMParams`, `NB2Params`, `LogisticParams`;
  7. the components (860): `CountComponent(family, use_exposure)`,
     `ProbabilityComponent(family)`, with `suggest_params`, `suggest_feature_transformer(trial,
     prefix, family_transformers)` and (probability) `suggest_calibration_method`.
- **Checks after T1.3:** ruff 8 findings, all pre-existing (7 marimo `return`s PLR1711, one
  B018 at line 129; RUF059 `test_age_group` at line 320); `ruff format --diff` shows only the
  two known hunks in the user's cells (lines 158, 213); `marimo check` the two known
  empty-cell issues; Check B finds every name above.
- **The notebook test** (`tests/characterization/test_experiments_notebook.py`): four of five
  pass. `test_a_tuning_cell_outside_a_run_button_is_caught` requires at least one cell naming
  `tune_model` behind `mo.stop(not …)`; no such cell exists until T1.6, so it fails until then
  by design. Keep the function name `tune_model`, and do not write that string in the
  markdown cell or in comments outside the run cells.
- **The IDE's type checker** reports `object.__init__` errors on the notebook's dataclasses:
  an artifact (in a marimo cell `dataclasses` is a parameter, so the checker cannot see the
  decorator). The probes show they work.
- **`src` is untouched** by the task so far. The S1 `get_params`/`set_params` forwarding in
  `IndependentCohortModels` stays (tested; `clone`-safe); the notebook no longer uses it.
- **Next: T1.4** (plan §2): `count_estimator`, `probability_estimator`, `suggest_count_model`,
  `suggest_model1`, `suggest_model2`. The plan's sketch was executed against the notebook's
  `defs` on 2026-10-18: both models build and fit.

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
- `CountModel` clones its `feature_transformer` at `fit`, so the builder passes the shared
  `FEATURE_TRANSFORMERS[name]` as is (`FEATURE_VARIANTS` before 2026-10-18).
- "Remove the metrics from scoring" is read as the five constants only. Confirm with the user
  at T3's plan summary.

**Settled at the user's reviews of T1.2 and T1.3 (2026-10-18; plan D2, D3, D6; do not re-ask):**
- An estimator dataclass holds only its searched hyperparameters; field names are the
  estimator's keywords, so a builder calls `Estimator(<fixed settings>, **asdict(params))`.
  The LightGBM regressor and classifier have separate classes over a shared `LGBMParams`.
- **No dataclass builds a model.** The objective builds it through `suggest_model1` /
  `suggest_model2`, which sample from the trial and call the constructors explicitly; the
  best run's refit calls the same function on `FixedTrial(study.best_params)`.
- LightGBM's loss is a family: `lgbm_poisson`, `lgbm_gaussian` (TPE here is univariate, so it
  cannot learn hyperparameters per loss; a mixed 30-trial study did worse than either loss).
- The exposure is a study setting, and every count family runs with it on and off (the
  user's choice). Measured on validation, one fit per cohort: the GLMs lose 30–50% with it
  off (their transformers hold no `n_apartments`), LightGBM about 3%. Model 1: 8 studies;
  Model 2: `MODEL2_TOTALS` × exposure × `PROBABILITY_FAMILIES` (16 at four totals).
- A component holds a study's fixed settings only: `CountComponent(family, use_exposure)`,
  `ProbabilityComponent(family)`. The family → transformer-names mapping is an argument of
  `suggest_feature_transformer`, the one place a family is paired with its transformers, so
  a component cannot be given another family's (a field of names could). The classes read no
  notebook global, so they can move to a package.
- The family is a `Literal` (`CountFamily`, `ProbabilityFamily`), a **plain assignment**:
  cloudpickle cannot pickle a class annotated with a `type` alias defined in a cell, and the
  joblib workers receive the components by cloudpickle (measured). An unknown family raises
  at construction (`__post_init__`); `suggest_params` is `if`/`elif` with a raising `else`.
- Names: "feature transformer", not "feature variant" (`FEATURE_TRANSFORMERS`,
  `*_TRANSFORMERS`, the trial parameter `<prefix>feature_transformer`).
- The record of a study's best run: `{"settings": asdict(component), "params":
  study.best_params}` (all JSON).

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
- **Expect a design review of each sub-task.** On 2026-10-18 each of T1.2 and T1.3 was
  reworked after the user read it. Answer a design question with a recommendation, the
  measured evidence and the trade-off (AskUserQuestion with code previews worked well); do
  not implement before the answer.
- Code that may move to a package reads no notebook global: settings fixed for an object go
  in its constructor, a mapping it consults comes in as an argument.
- After a redesign, update the plan doc's later sketches in the same step, and run the next
  sub-task's sketch against the notebook's `defs` as a check (exec the plan's code block in a
  namespace built from `defs`).

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
(§0 decisions, §1 rules and the checks A/B/C, §2 the sub-tasks of T1, §3 T2). Then read
notebooks/02_experiments.py from line 590 to the end, src/age_group_prediction/scoring.py,
and the constructors of CountModel, CohortProbabilityModel, IndependentCohortModels and
TotalTimesProbabilityModel in src/age_group_prediction/modeling/.

Expected state: branch feat/experiment-notebook, HEAD 7ae6314 (T1.3), clean tree, notebook
md5 389df50917ba2b32c4f949f252888ae2. Verify HEAD, git status and the md5 first; if
anything has changes the handoff does not name, stop and show me the diff. I commit, push
and merge myself: never commit, never add a Co-Authored-By line, never git
checkout/stash/reset/mv.

Task: sub-task T1.4 of docs/NOTEBOOK_TUNING_PLAN.md (the model builders: count_estimator,
probability_estimator, suggest_count_model, suggest_model1, suggest_model2), then one
sub-task per approval in the plan's order (T1.5 … T1.8, T2, T3, T4, T5). Work in plan mode
for each: re-verify the plan's facts for it with read-only probes
(PYTHONPATH=src .venv/bin/python -W error -c "..."), summarize the sub-task in the chat,
and ask for approval before editing. Do not re-ask the decisions of the plan's §0 or the
handoff's §3. Write the code as the plan gives it; if you must deviate, say why with
measured evidence before doing so. I review the design of each sub-task: when I ask a
design question, answer with a recommendation, the evidence and the trade-off, and wait
for my answer before editing.

At each sub-task's stop give me: a file-by-file summary; the results of the plan's checks
for that sub-task (Check A, Check B, and Check C from T1.6 on), ruff on the notebook (8
pre-existing findings, no new one), ruff format --diff (only the two known hunks at lines
158 and 213); at T1.8 and T2.3 an independent review subagent's findings reproduced (tell
it to write nothing into the repository); the notebook's new md5; and the suggested commit
command. Tick the sub-task and update the status line in docs/NOTEBOOK_TUNING_PLAN.md.
```
