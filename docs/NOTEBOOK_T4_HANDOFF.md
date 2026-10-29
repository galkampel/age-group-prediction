# Handoff: T4a (three sub-tasks), then T4b.1–T4b.3 and T5 of NOTEBOOK_TUNING_PLAN.md

**Written for:** the model that implements it (Claude Opus 5.5) and the user who validates each
sub-task. Self-contained: it assumes no memory of the planning conversation. Every fact below was
verified on 2026-10-29 with read-only probes; re-verify the ones marked *(re-verify)* before
relying on them, since the user edits between turns.

**Status (2026-10-29):** written and approved by the user after the reading of 2026-10-29 (the
planning session verified the state and measured the baselines of §1; it edited nothing else).
**T4a.1 committed `5c0908a` (2026-10-29)** on the branch `refactor/hyperparameter-tuning-package`
(from `e2a4f78`, pushed), whose draft PR #17 "refactor(tuning): rebuild the hyperparameter_tuning
package from the experiment notebook" goes into `feat/experiment-notebook` and covers T4a.1–T5, one
commit per sub-task; the notebook md5 unchanged `425b6cf3ff396a478ad08756c2211338`. The user
paused there. **T4a.2 committed `ed1a770`** (the four docs deleted, the five guides point at the
notebook). **T4a.3 committed `6935416`** (the 11 notes, the status lines, Check A and Check C);
T4a is complete. **T4b.1 done 2026-10-29** (`params.py`, `components.py`, `search_spaces.py`
and their tests; §3's T4b.1 result), awaiting the user's commit; the notebook md5 unchanged.
**Next: T4b.2.** The implementing session updates this line, ticks the sub-tasks of
§2, and records each sub-task's commit and the notebook's md5 in §1 as it goes; this file is the
handoff for any session that continues the work.

**The prompt for the implementing session:**

```text
Read docs/NOTEBOOK_T4_HANDOFF.md in full, then the documents its reading order lists. Verify
its §1 state first (branch refactor/hyperparameter-tuning-package, HEAD my commit of the
handoff update right after 5c0908a, git status clean, the notebook md5); if anything differs,
stop and show me the diff. Then T4a.2 in plan
mode: re-verify its facts with read-only probes, write the steps and the commit message in the
chat, and wait for my approval before editing. One sub-task per approval; I commit myself. At
each stop also give me the updated lines of draft PR #17's body.
```

**Repository:** `/Users/galkampel/Desktop/Projects/age-group-prediction`, branch
`refactor/hyperparameter-tuning-package` (stacked on `feat/experiment-notebook`, draft PR #17).
The source-of-truth documents, in reading order:
1. `docs/EXPERIMENT_NOTEBOOK_HANDOFF.md` (all of it: §2 the state and the 24 Check C baselines,
   §3 the decisions, §4 how to work, §6 the prompt this plan answers);
2. `docs/NOTEBOOK_TUNING_PLAN.md` §0 (D1–D25; D21 and D25 above all), §1 (rules, Checks A/B/C),
   §5–§6 (T4a, T4b, T5);
3. `docs/NOTEBOOK_COMPONENT_REDESIGN_PLAN.md` §0 and §6 (the history of D23–D24; its code blocks
   are superseded by the notebook);
4. `notebooks/02_experiments.py` from the `## 4) Feature variants` cell (line 378) to the end
   (line 4446); `src/age_group_prediction/aggregation.py`;
   `tests/characterization/test_experiments_notebook.py`.

## 0. The rules (the user's, standing)

- **Plan mode before each sub-task.** Re-verify the facts with read-only probes
  (`PYTHONPATH=src .venv/bin/python -W error -c "..."`), write the sub-task's steps and its commit
  message in the chat, ask for approval, then edit. One sub-task per approval. Stop after each.
- **The user commits, pushes and merges.** Never commit, never add a `Co-Authored-By` line, never
  `git checkout` / `stash` / `reset` / `mv`. Delete with plain `rm`. Restore a file from HEAD with
  `git show HEAD:<path> > <path>` and `cmp`.
- **Do not re-ask the decisions of the plans' §0.** Move the notebook's code as it is; a change
  needs a stated reason with evidence first. A design question gets a recommendation, the
  evidence and the trade-off, and waits for the user's answer before any edit. No mechanism the
  measurements do not pay for.
- **Before each sub-task and each edit:** `git status` (clean except what the previous sub-task
  left), `md5 -q notebooks/02_experiments.py` against the last recorded value. A difference the
  handoff does not name: show the diff and ask.
- **At each stop give:** a file-by-file summary; the check results (Check A, B, C with the 24
  baselines; the package tests and mypy where src changes); ruff on the notebook (14 pre-existing
  findings, re-counted when cells are removed, no new one); `ruff format --diff` (only the two
  known hunks at lines 158 and 213); the notebook's new md5; the commit message. Tick the
  sub-task and update the status lines in NOTEBOOK_TUNING_PLAN.md (its first line and the §5/§6
  checkboxes), EXPERIMENT_NOTEBOOK_HANDOFF.md §2 and NOTEBOOK_COMPONENT_REDESIGN_PLAN.md's status.
- **The machine can be heavily loaded** (load average 31–62 on 8 cores when this was written).
  Run every notebook check and the test suite in the background with stdin closed, unbuffered,
  logged to the session scratchpad (`... > <scratch>/<name>.log 2>&1 < /dev/null`), and poll
  the log. `timeout` does not exist on macOS.
- **Scratch files** go to the session scratchpad, never into `notebooks/` or the repository.
  Review subagents write nothing into the repository.
- **Format new code to `ruff format`'s output** (new cells by hand or via a scratch file); never
  `ruff format` the whole notebook (the user's two hunks). The notebook is outside mypy's `files`.
- Tools: `.venv/bin/python` 3.13.9, `.venv/bin/ruff` 0.16.4, `.venv/bin/pytest`, `.venv/bin/mypy`,
  `.venv/bin/marimo`; `uv` at `~/.local/bin/uv` (`uv run --frozen mypy`). optuna 4.9.0, joblib
  1.5.3, lightgbm 4.7.0, scikit-learn 1.9.0, pandas 3.0.5.
- **The pull request** is draft #17 from `refactor/hyperparameter-tuning-package` into
  `feat/experiment-notebook` (the user opened it after T4a.1). At each stop also give the
  updated lines of its body (the sub-task's checkbox ticked with its commit, the verification
  results) for the user to paste. Never push or edit the PR yourself.
- `uv run --frozen mypy` re-syncs the project's editable install in `.venv` (no dependency
  changes; harmless). Use mypy's default cache: a cold cache in the scratchpad took over 6 min
  under load.

## 1. State verified 2026-10-29, after T4a.1 *(re-verify first)*

```sh
cd /Users/galkampel/Desktop/Projects/age-group-prediction
git branch --show-current          # refactor/hyperparameter-tuning-package
git log -3 --oneline               # <the user's commit of this handoff update>; 5c0908a refactor(tuning): T4a.1 ...; e2a4f78 docs(notebook): the T4 handoff ...
git status --short                 # clean
md5 -q notebooks/02_experiments.py # 425b6cf3ff396a478ad08756c2211338
```

- `e2a4f78` (this handoff, docs only) follows `15932d7`, which holds T3, D23, D24, D25 in one
  commit; `5c0908a` is T4a.1. The branch is pushed; draft PR #17 into `feat/experiment-notebook`.
- Baselines after T4a.1 (measured): `uv run --frozen mypy` clean, **22** source files;
  `pytest -m "not slow"` **1226 passed, 1 skipped, 1 xfailed, 19 deselected** (1228 collected;
  the 141 removed tests were 138 non-slow and 3 slow; the six MLflow `UserWarning` hints from
  `test_gate8_tracking.py` pre-exist); the notebook test **12 passed** (the file holds 12 tests;
  the "17" written earlier here and in the plans was a miscount, T4a.3 corrects the plans).
- Baselines at `15932d7`, the notebook unchanged since: `ruff check notebooks/02_experiments.py`
  14 findings; `ruff format --diff` the hunks `@@ -158,9 +158,7 @@` and `@@ -213,8 +211,10 @@`
  only.
- *(Before T4a.1, kept for the record:)* mypy 27 files; the five old test files 141 tests;
  `pytest -m "not slow" --collect-only` 1366 collected (22 deselected).
- *(Before T4a.1)* The old package `src/age_group_prediction/hyperparameter_tuning/` holds `__init__.py`,
  `aggregation.py`, `evaluator.py`, `parameters.py`, `study.py` (+ `__pycache__`). Its
  `aggregation.py` differs from `src/age_group_prediction/aggregation.py` only by one docstring
  word ("(the default)") and the `_config` import; `tests/unit/test_hyperparameter_tuning_aggregation.py`
  differs from `tests/unit/test_aggregation.py` only by its import line.
- Nothing in `src` imports the old package except its own modules; `src/age_group_prediction/__init__.py`
  does not import it. The old stack's `src/age_group_prediction/tuning.py` (`TuningResult`,
  `TrialRecord`, `run_optuna_study`) is unrelated and stays.
- The notebook imports `from age_group_prediction.aggregation import Mean` (line 736) and nothing
  from the old package.
- Every file naming the old package (`grep -rln "hyperparameter_tuning\|HYPERPARAMETER_TUNING"`,
  excluding `.venv` and the package itself):
  - **code and config:** `pyproject.toml` (lines 56, 76), `src/age_group_prediction/_config.py`
    (line 1 docstring), `tests/unit/test_modeling_count_model.py` (line 331, a comment), the five
    `tests/unit/test_hyperparameter_tuning_*.py`;
  - **the docs to delete:** `docs/HYPERPARAMETER_TUNING.md`, `_PLAN.md`, `_REVIEW.md`, `_VALIDATION.md`;
  - **current guides:** `README.md` (36–39), `docs/README.md` (23–25, 44, 74),
    `docs/MODULE_REFERENCE.md` (5, 11, 74–77), `docs/INDEPENDENT_COHORT_MODELS.md` (299–300),
    `docs/SPLITTING.md` (266–268);
  - **finished plans (one-line note only):** `docs/EXPERIMENT_NOTEBOOK_PLAN.md`,
    `docs/MODEL_INSPECTION_PLAN.md`, `docs/TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md`,
    `docs/DIRECT_COHORT_GENERALIZATION_PLAN.md`, `docs/DIRECT_COHORT_GENERALIZATION_HANDOFF.md`,
    `docs/DIRECT_COHORT_GENERALIZATION_VALIDATION.md`,
    `docs/DIRECT_COHORT_GENERALIZATION_VALIDATION_REPORT.md`, `docs/SPLITTING_FIX_PLAN.md`,
    `docs/SIMULATOR_WHITE_BOX_PLAN.md`, `docs/MODEL_REIMPLEMENTATION_PLAN.md`,
    `docs/MULTI_COHORT_MODELS_PLAN.md` (11 files);
  - **the living plans** (status lines only): `docs/NOTEBOOK_TUNING_PLAN.md`,
    `docs/EXPERIMENT_NOTEBOOK_HANDOFF.md`, `docs/NOTEBOOK_COMPONENT_REDESIGN_PLAN.md`.
- `README.md:32` links `docs/MODELING_GUIDE.md#hyperparameter-tuning`, the old stack's Models A/B:
  untouched.

## 2. T4a: delete the old package (D17), as three sub-tasks

The notebook is untouched by all three: its md5 stays `425b6cf3ff396a478ad08756c2211338`, Check B
is not needed (no cell changes). `src/age_group_prediction/aggregation.py` and
`tests/unit/test_aggregation.py` stay (D9's interim home; T4b.2 moves them into the new
package). `optuna` stays in `pyproject.toml`.

### [x] T4a.1 The code: the package, its tests, the config

1. `rm -r src/age_group_prediction/hyperparameter_tuning`
2. `rm tests/unit/test_hyperparameter_tuning_aggregation.py tests/unit/test_hyperparameter_tuning_evaluator.py tests/unit/test_hyperparameter_tuning_integration.py tests/unit/test_hyperparameter_tuning_parameters.py tests/unit/test_hyperparameter_tuning_study.py`
3. `pyproject.toml`: remove `"src/age_group_prediction/hyperparameter_tuning",` from
   `[tool.mypy] files` and `"age_group_prediction.hyperparameter_tuning.*",` from the
   `[[tool.mypy.overrides]] module` list. Nothing else there changes.
4. `src/age_group_prediction/_config.py` line 1: the docstring names the users that remain:
   `"""The pydantic config shared by the user-built value objects of ``aggregation`` and ``inspection``."""`
5. `tests/unit/test_modeling_count_model.py:329–331`, the comment of
   `test_a_negative_gaussian_prediction_breaking_the_deviance_is_caught`: replace
   `docs/HYPERPARAMETER_TUNING.md §10` with `docs/INDEPENDENT_COHORT_MODELS.md`, whose `y_min`
   paragraph (around line 107–110: "an exposure predicted a negative count in the fourth trial
   of a study and stopped it") records the same hazard. Comment text only.

**Done when:**
- `grep -rn "hyperparameter_tuning\|HYPERPARAMETER_TUNING" src tests` prints nothing;
- `uv run --frozen mypy` → `Success: no issues found in 22 source files`;
- `.venv/bin/pytest -m "not slow" -q -p no:cacheprovider` (background, logged) → all passed,
  no warning turned error;
- `.venv/bin/pytest tests/characterization/test_experiments_notebook.py -q` → 12 passed;
- `git status` shows only the deletions and the three edits; the notebook md5 unchanged.

**Result (2026-10-29), committed `5c0908a`:** the grep empty; mypy 22 files; the non-slow suite
1226 passed, 1 skipped, 1 xfailed, 19 deselected in 121 s; the notebook test 12 passed; md5
unchanged. The handoff's expected 1225 and 17 were miscounts (see §1).

**Commit message (the user commits):**
```
refactor(tuning): T4a.1, delete the old hyperparameter_tuning package and its tests

The package (parameters, evaluator, study and its copy of aggregation) and
its five test files (141 tests) go: the notebook's section 5 is the tuning
now (D1, D17), and T4b rebuilds the package from it. The aggregators stay in
age_group_prediction/aggregation.py (D9's interim home) until T4b.2. mypy's
files and overrides drop the package; one test comment points at
INDEPENDENT_COHORT_MODELS.md instead of the guide T4a.2 deletes.
```

### [x] T4a.2 The docs that state current API

1. `rm docs/HYPERPARAMETER_TUNING.md docs/HYPERPARAMETER_TUNING_PLAN.md docs/HYPERPARAMETER_TUNING_REVIEW.md docs/HYPERPARAMETER_TUNING_VALIDATION.md`
2. `README.md:36–39`: the bullet "[Hyperparameter tuning package](docs/HYPERPARAMETER_TUNING.md):
   the `hyperparameter_tuning` package that tunes the rebuilt `modeling` models …" becomes one
   bullet on the notebook's tuning: section 5 of `notebooks/02_experiments.py` tunes the rebuilt
   models with plain Optuna (plan: `docs/NOTEBOOK_TUNING_PLAN.md`); the `hyperparameter_tuning`
   package was removed on 2026-10-29 and is rebuilt from that code (T4b); its guide returns with
   T5. Keep the bullet's style (two to four lines, a markdown link).
3. `docs/README.md`:
   - lines 23–25 (the in-progress bullet "the `hyperparameter_tuning` package is complete (PR #5
     …)") become one line: the package was removed on 2026-10-29 and is rebuilt from the
     notebook's section 5 (`NOTEBOOK_TUNING_PLAN.md` D17, T4b);
   - line 44 (the table row linking `HYPERPARAMETER_TUNING.md`) and line 74 (the row linking
     `HYPERPARAMETER_TUNING_PLAN.md`) are removed. Do not add rows for other documents here;
     T5 owns the docs index.
4. `docs/MODULE_REFERENCE.md`:
   - line 5: drop `2026-10-11 (`hyperparameter_tuning`), ` from the date list;
   - line 11: drop `` `age_group_prediction.hyperparameter_tuning`, `` from the import list;
   - rows 74–77 (`hyperparameter_tuning/parameters.py`, `evaluator.py`, `aggregation.py`,
     `study.py`) are removed. No row is added for the root `aggregation.py` (T4b.2 moves it into
     the new package and documents it there). Check the mermaid block (lines ~17–35) and
     "Other Code Locations" for the package name; the grep found none, but re-check.
5. `docs/INDEPENDENT_COHORT_MODELS.md:299–300`: "One study then tunes every cohort on the
   composite's score ([Hyperparameter tuning §9](…))." → "One study can then tune every cohort
   on the composite's score through these names; the experiment notebook builds its components
   explicitly instead (`NOTEBOOK_TUNING_PLAN.md` D1)." The `set_params` forwarding stays documented.
6. `docs/SPLITTING.md:266–268`: "The new tuning package already consumes `cv`: every trial
   re-splits with the same seeded validator ([Hyperparameter tuning §4](…))." → "The experiment
   notebook's tuning loop (`cv_score`, section 5 of `notebooks/02_experiments.py`) already
   consumes `cv`: every trial splits with the same seeded validator."

**Done when:**
- `grep -rn "hyperparameter_tuning\|HYPERPARAMETER_TUNING" README.md docs/README.md docs/MODULE_REFERENCE.md docs/INDEPENDENT_COHORT_MODELS.md docs/SPLITTING.md` prints nothing;
- `ls docs/HYPERPARAMETER_TUNING*` finds nothing;
- every link the edited lines keep resolves (`NOTEBOOK_TUNING_PLAN.md` exists);
- `git status` shows the four deletions and the five edits; the notebook md5 unchanged.

**Result (2026-10-29), awaiting the user's commit:** the grep empty; the four docs gone; every
relative link of the five guides resolves; `git status` the four deletions, the five edits and
this file; md5 unchanged. Two departures from the wording above, approved with the plan: the new
README and docs/README text says "the old tuning package" (steps 2–3 named the package, which
the done-when grep forbids), and `INDEPENDENT_COHORT_MODELS.md` says "a tuner's
`clone(model).set_params(...)`" for "the tuner's" (the tuner was the deleted evaluator; the file
says "a tuner" elsewhere). Left for T5: `MODULE_REFERENCE.md`'s `scoring.py` row still lists the
five constants T3 removed.

**Commit message:**
```
docs(tuning): T4a.2, the old package's four docs deleted, the guides point at the notebook

HYPERPARAMETER_TUNING.md and its plan, review and validation go with the
package (T4a.1). README, docs/README, MODULE_REFERENCE,
INDEPENDENT_COHORT_MODELS and SPLITTING name the notebook's section 5 as
the tuning; the new guide comes with T5.
```

### [x] T4a.3 The finished plans, the status lines, the notebook checks

1. In each of the 11 finished plans of §1 add one line, directly under the title (or under the
   existing dated notes some of them already carry), nothing else:
   `> **2026-10-29:** the `hyperparameter_tuning` package this document names was removed (NOTEBOOK_TUNING_PLAN.md D17, T4a); T4b rebuilds it from the notebook's section 5.`
2. `docs/NOTEBOOK_TUNING_PLAN.md`: tick `### [x] T4a Delete the old package`; append to the
   status line (line 3): "T4a done 2026-10-29 (T4a.1 the code, T4a.2 the current guides, T4a.3
   the finished plans; the non-slow suite 1226 passed, mypy 22 files, the notebook unchanged, Check C
   the 24 baselines). Next: T4b.1."
3. `docs/EXPERIMENT_NOTEBOOK_HANDOFF.md` §2: the "… committed together as `15932d7` … **Next:
   T4a.1**, planned with …" sentence becomes the commits of T4a.1–T4a.3 (T4a.1 `5c0908a`, on
   `refactor/hyperparameter-tuning-package`, draft PR #17) (hashes if the user has committed; otherwise "awaiting the
   user's commit") and "**Next: T4b.1.**"; add the three rows to the commits table.
4. `docs/NOTEBOOK_COMPONENT_REDESIGN_PLAN.md` status: "**Next: T4a**" → "**Next: T4b.1**".
4a. The notebook test's count is 12, not 17 (miscounted at D25): correct "(17 tests)" in
   `NOTEBOOK_TUNING_PLAN.md` D25.4 (line ~645) and its status line ("the notebook test 17/17"),
   and "notebook test 17/17" in `EXPERIMENT_NOTEBOOK_HANDOFF.md` §2 (line ~133), each to 12.
5. Run the notebook checks on the unchanged notebook (they prove nothing it imports left with
   the package):
   - **Check A:** `.venv/bin/pytest tests/characterization/test_experiments_notebook.py -q` →
     12 passed; `.venv/bin/ruff check notebooks/02_experiments.py | tail -2` → 14 findings;
     `.venv/bin/ruff format --diff notebooks/02_experiments.py | grep '^@@'` → the two hunks;
     `.venv/bin/marimo check notebooks/02_experiments.py` → the two known empty-cell issues.
   - **Check C** (background, logged): the recipe of §4 below, the 24 baselines of §5.

**Done when:** the 11 notes present and nothing else changed in those files (`git diff --stat`
shows +1 per file); the status lines updated; Check A and Check C as above; md5 unchanged.

**Result (2026-10-29), awaiting the user's commit:** the 11 notes, each `2 0` in
`git diff --numstat` (the note and a blank line: a `>` line directly after a paragraph or
another quote merges into it), the note linking the plan and saying the docs went too; the
status lines; 17 → 12 in the three places. Also, approved with the plan: the commits table of
EXPERIMENT_NOTEBOOK_HANDOFF.md §2 lost its stale row (the budget sub-task is `10d6e83`) and its
gap (T2.5, `15932d7`, the T4 handoff). Check A: the notebook test 12 passed, ruff 14, the format
hunks -158 and -213 only, `marimo check` the two known empty cells. Check C (both sharing flags
`False` in the copy, so the arm of the 24 baselines; the copy's `n_trials = 400` replaced by
line, since the markdown at line 672 holds the same string): all 24 baselines to the fourth
decimal, no warning under `PYTHONWARNINGS=error`, 117.7 s with the inspection. md5 unchanged.

**Commit message:**
```
docs(tuning): T4a.3, the finished plans note the package's removal; T4a ticked

A one-line dated note in the eleven plans that name the old package;
NOTEBOOK_TUNING_PLAN.md ticks T4a, the handoff and the redesign plan point
at T4b.1. Check A and Check C (the 24 baselines) pass on the unchanged
notebook.
```

## 3. What follows T4a (plan each in plan mode, from NOTEBOOK_TUNING_PLAN.md §5–§6)

The orientation facts, verified 2026-10-29 *(re-verify line numbers; they move)*. The design is
D21 (six one-way modules, what moves and what stays), D25 (the module table in §5 T4b), and the
user's style in the handoff §4.

**The notebook's section 5 cells that move** (line ranges at md5 `425b6cf3…`):

| Cell | Lines | Holds | Target module (T4b) |
|---|---|---|---|
| imports | 715–789 | `abc`, `dataclasses`, `time`, `partial`, `ClassVar`, `Literal`, numpy, optuna, joblib, lightgbm, `TPESampler`, sklearn estimators and metrics, the project imports; `optuna.logging.set_verbosity(WARNING)` | stays, trimmed to what the remaining cells use |
| metrics | 792–842 | `COHORT_POISSON_DEVIANCE`, `POISSON_DEVIANCE`, `COHORT_LOG_LOSS`, `on_cells`, `wmape`, `bias`, `INSPECTION_METRICS` | stays (D4) |
| constants and families | 845–856 | `COHORTS`, `PREDICTION_FLOOR`, `CountFamily`, `ProbabilityFamily` (`Literal` assignments) | `COHORTS`, `PREDICTION_FLOOR` stay; the families → `components.py` as `type` statements (D3: they pickle from a module) |
| params | 859–892 | `LGBMParams`, `PoissonGLMParams`, `NB2Params`, `LogisticParams` (frozen, `kw_only`) | `params.py` |
| components | 895–1056 | `Component` (`set_estimator_params`, `make_estimator` abstract, `to_dict`), `CountComponent`, `ProbabilityComponent` (each `from_params`, `make_estimator`; the field `estimator_params`) | `components.py` |
| search spaces | 1059–1191 | `ComponentSearchSpace` (`prefix`, `suggest_feature_transformer_name`, `suggest_lgbm_hyperparams`), `CountSearchSpace`, `ProbabilitySearchSpace` (`CALIBRATION_METHODS` ClassVar, `suggest_calibration_method`) | `search_spaces.py` |
| builders | 1194–1377 | `build_independent_cohorts_model`, `suggest_independent_cohorts_model`, `build_total_times_probability_model`, `suggest_total_times_probability_model` | `builders.py` |
| cross-validation | 1380–1466 | `fold_score`, `cv_score`, `tune_model` | `cross_validation.py` |
| `fit_and_score` | 1479–1508 | a top-level `@app.function` | `cross_validation.py` |
| aggregation | `src/age_group_prediction/aggregation.py`, `tests/unit/test_aggregation.py` | `Aggregator`, `Mean`, `WeightedMean`, `UpperBound`, `corrected_std_error` | `aggregation.py` (moved back, T4b.2) |

**Stays in the notebook:** the section-4 candidate dicts `COUNT_FEATURE_TRANSFORMERS` and
`PROBABILITY_FEATURE_TRANSFORMERS` (lines 602–636, the data the package's mechanism takes), the
settings, the metrics, the buttons, the 8 + 16 run cells (1511–2259, 2354–4031), the tables cells
(2261–2325, 4033–4071 and the Model 2 tables cell before 4033), `best_df`, the inspection cells
(4074–4441).

**Dependencies (D21, one-way):** `params` ← `components` ← `search_spaces`; `builders` uses those
and `modeling`; `cross_validation` uses `aggregation` only (it takes `suggest_model` as a
callable); `aggregation` stands alone. No runner, no results table, no MLflow, no transformer
registry in the package.

**T4b.1** (records and search spaces): `params.py`, `components.py`, `search_spaces.py`, typed
for mypy strict, `__all__` per module, docstrings kept to what matters; `pyproject.toml` mypy
`files` and overrides list the package again; unit tests `tests/unit/test_hyperparameter_tuning_{params,components,search_spaces}.py`:
`kw_only`, a typo in the params raises `TypeError`, an unknown family raises in `make_estimator`
and `from_params`, the trial names `<name>__<parameter>`, `from_params` on a trial's params
equals the objective's draws (the notebook test's check, D25.4), `KeyError` names a missing
field. The notebook keeps its own cells until T4b.3.

*Result (2026-10-29), awaiting the user's commit:* the three modules and `__init__.py`; an AST
comparison with the notebook's classes (annotations and docstrings stripped) finds every class
identical except the approved changes: the families as `type` statements; `calibration_method`
and `CALIBRATION_METHODS` typed `CalibrationMethod | None` (precision only: mypy accepts
`str | None`, since `CohortProbabilityModel`'s scikit-learn base is untyped); a `cast` in
`suggest_calibration_method` (Optuna types a mixed list's draw as any choice type, with either
annotation); local `estimator_params:` declarations in `from_params`; at the user's review,
`type EstimatorParams` (the four records' union, `params.py`) bounds the generic
`set_estimator_params[P: EstimatorParams]` (it was typeshed's private `DataclassInstance`, any
dataclass) and types the abstract `suggest_hyperparameters`. mypy no issues in 26 files;
48 new tests under `-W error`; the non-slow suite 1272 passed before the review's two added tests
(1 skipped, 1 xfailed, 19 deselected); eight mutations (the `shared` filter, the raising `else`,
the prefix, the LightGBM losses, the calibration and transformer read-backs, a defaulted field)
each fail the tests. The independent review found no bug; it added the two `KeyError` cases and
the docstring's "LightGBM" (the scikit-learn estimators and NB2 raise on a typo themselves); the
bounds stay unpinned by tests (one source of truth; Check C catches a change). Pickling: the
components and search spaces round-trip through pickle, cloudpickle and loky workers.

**T4b.2** (builders, loop, aggregation): `builders.py`, `cross_validation.py`, `aggregation.py`
moved from the root (update `tests/unit/test_aggregation.py`'s import, or move it to
`test_hyperparameter_tuning_aggregation.py`; remove the root module and its mypy entries); tests:
the builders' models (the three sharing cases, both models), `cv_score` in parallel equal to one
after another, `tune_model`'s `catch=(RuntimeError,)`, `n_startup_trials` reaching the sampler.
`_config.py`'s docstring names the package again. The notebook's import of `Mean` moves to the
package path in T4b.3 (or here, if the root module goes here: then T4b.2 touches the one
notebook import line and Checks A/B/C run).

**T4b.3** (the notebook imports the package): the class and function cells (params, components,
search spaces, builders, cross-validation, `fit_and_score`) go; the import cell imports them from
`age_group_prediction.hyperparameter_tuning`; the families cell keeps `COHORTS` and
`PREDICTION_FLOOR`; the header's prose unchanged unless it names a cell that left.
`tests/characterization/test_experiments_notebook.py`: the `tune_model` rule exempts the import
cell instead of the `def tune_model(` cell; its `notebook_defs` fixture still reads the classes
from `app.run()`'s defs (an imported name is a def). Ruff's baseline is re-counted (the 7 marimo
`return` findings may change with the removed cells). Check B names: `CountComponent`,
`ProbabilityComponent`, `CountSearchSpace`, `ProbabilitySearchSpace`, the four builders,
`tune_model`, `fit_and_score`, `Mean`, `COHORTS`, `PREDICTION_FLOOR`, `MODEL1_RUNS`,
`MODEL2_RUNS`, `model1_df`, `model2_df`, `best_df`, `intervals_df`, `importance_df`. Check C
must reproduce the 24 baselines.

**T5** (docs, §6): a new `docs/HYPERPARAMETER_TUNING.md` for the new package; `MODULE_REFERENCE.md`
rows for the six modules; `README.md` and `docs/README.md` entries; the handoff rewritten for the
next session (§1 history, §2 state, §3 decisions, §6 the prompt for the items after: E11's final
evaluation on test, MLflow runs from the run records); `EXPERIMENT_NOTEBOOK_PLAN.md` E4, E5, E11,
E12, D2/D3 and its status line; the dataclass-vs-TypedDict table into the handoff.

## 4. The checks (NOTEBOOK_TUNING_PLAN.md §1.1, with today's values)

**Check A**
```sh
.venv/bin/pytest tests/characterization/test_experiments_notebook.py -q    # 12 passed
.venv/bin/ruff check notebooks/02_experiments.py | tail -2                   # 14 findings (re-count after T4b.3)
.venv/bin/ruff format --diff notebooks/02_experiments.py | grep '^@@'        # -158 and -213 only
.venv/bin/marimo check notebooks/02_experiments.py                           # the two known empty-cell issues
```

**Check B** (which names a script run defines): the heredoc of §1.1 with the sub-task's `NAMES`.

**Check C** (the clicked run on a copy, `n_trials = 3`, under `PYTHONWARNINGS=error`, in the
background): build `<scratch>/project/notebooks/02_experiments.py` from the notebook with
`n_trials = 400` replaced by `n_trials = 3` (exactly one occurrence), and
`<scratch>/project/configs` a symlink to the repo's `configs`; then
```sh
cd /Users/galkampel/Desktop/Projects/age-group-prediction
PYTHONWARNINGS=error PYTHONPATH=src nohup .venv/bin/python -u - > <scratch>/check_c.log 2>&1 < /dev/null <<'EOF' &
import importlib.util, types
from unittest import mock
path = "<scratch>/project/notebooks/02_experiments.py"
spec = importlib.util.spec_from_file_location("nb", path)
nb = importlib.util.module_from_spec(spec)
with mock.patch("marimo.ui.run_button", return_value=types.SimpleNamespace(value=True)):
    spec.loader.exec_module(nb)
    _, defs = nb.app.run()
print(defs["model1_df"][defs["model1_df"]["cohort"] == "composite"][["family", "use_exposure", "cv_best"]].to_string(index=False))
print(defs["model2_df"][defs["model2_df"]["cohort"] == "composite"][["family", "use_exposure", "cv_best"]].to_string(index=False))
EOF
```
It takes 47–67 s unloaded; the run-button patch also fires the inspection button (the SHAP of the
best runs, one to two minutes more). Optuna's WARNING lines for failed NB2 trials are expected.
A warning in a fold process fails the check only through `PYTHONWARNINGS`, not `-W` (D15).
Never run two `cv_score` calls at once from threads (T2.4: wrong scores).

## 5. The 24 Check C baselines at `n_trials = 3` (`cv_best`, both sharing flags `False`)

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

The notebook's settings are `share_hyperparams = True`, `share_feature_transformer = False`
(D24.4), so a Check C of the notebook as it is gives Model 1's "shared hyperparameters" arm
(handoff §2's second table: `lgbm_poisson` 2.1812 / 2.2193, `lgbm_gaussian` 2.2177 / 2.2805,
`poisson_glm` 2.4158 / 2.8685, `nb2` 2.2762 / 2.8707); Model 2 does not depend on the flags.
State which arm was run. The shared probe (D23–D25): a 3-trial `lgbm_poisson` study with
`share_hyperparams=True` gives 2.1812 with 13 parameters.

## 6. Traps recorded by the earlier sessions (handoff §4)

- marimo's editor trims each cell's `return` to the names other cells read when the user saves;
  ruff then jumps to ~31 findings and Check B loses names. The user has HEAD's returns restored
  after seeing the diff.
- A function sent to joblib must not capture the notebook's `defs` (fails to pickle,
  `_thread.lock`); `partial(suggest_independent_cohorts_model, …)` pickles. The `Literal`
  families are plain assignments in a cell because cloudpickle cannot pickle a class annotated
  with a cell-defined `type` alias; in a package module either form pickles (D3).
- pandas 3.0.5: `None` in a str column becomes NaN and `value_counts()` drops it; the tables
  count by label after `fillna("None")` (D16).
- `json.dumps(record, default=str)` writes a `NaN` token; use `trials.to_json(orient="records")`
  when records are saved.
- The characterization test requires every cell containing the string `tune_model` (except the
  `def tune_model(` cell, the import cell after T4b.3) to hold `mo.stop(not `; the name must not
  appear in the markdown cell or comments outside run cells.
- The user is not familiar with marimo's UI terms ("button"): explain them.
