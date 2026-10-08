# Handoff: rebuild Model 2 as `TotalTimesProbabilityModel`

**For:** the implementing session (Claude Opus 5.5). **Written:** 2026-10-07 at the end of
the planning session; **updated 2026-10-08 at the end of the third implementing session:
sub-tasks 0–4 are done (4 awaiting the user's commit); sub-task 5 (`TotalTimesProbabilityModel`)
is next.**
**Plan (source of truth):** [TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md).
Read it in full before the first edit (its §8 records say what each sub-task did and why,
including sub-task 3's four revisions); this file only orients you.

## 1. What the task is

Model 2 (the torch build of PR #11) predicted each cohort as **predicted total × predicted
cohort probability** with hand-written torch objectives. The user wants it rebuilt on
**library estimators**, no torch:

1. **Total (done, sub-tasks 1–2):** `DirectCohortModel`, **renamed `CountModel`**, with any
   regressor. The exposure is applied by named cases read from the estimator's `fit`
   signature: an estimator taking `exposure` gets the raw exposure (**NB2**:
   `NegativeBinomialRegressor`, statsmodels' `NegativeBinomial(loglike_method="nb2",
   exposure=…)`); one taking `sample_weight` gets the weighted rate; any other raises.
   `TotalChildrenModel` is deleted in sub-task 5.
2. **Probabilities (sub-task 3 done; sub-task 4 next):** `CohortProbabilityModel(estimator:
   Classifier, replication: ReplicationType = "weighted", feature_transformer=None)` in
   `modeling/cohort_probability.py`. `fit` checks `len(X) == len(y)` and that every cohort has
   a child, fits the feature transformer on the samples (rows of `X`), and builds the
   categorical rows with `multinomial_to_categorical(y, replication)`: `"weighted"` = one row
   per (sample, cohort) with a child, `sample_weight` = the count; `"per_child"` = one row per
   child, no `sample_weight` keyword. The classifier is fitted on the cohorts' **column
   positions** (0..K−1), so its `classes_` are `y`'s order and `predict` names the columns with
   `cohorts_`; `predict` raises if a row of probabilities does not sum to 1 (LightGBM's
   `"multiclassova"`). **Sub-task 4 added** `calibration_method` (`None` default; temperature,
   sigmoid, isotonic) and `calibration_cv` (5): `CalibratedClassifierCV(ensemble=False)` on
   folds of **samples**, split round-robin before the categorical rows
   (`PredefinedSplit(sample_folds[sample_positions])`), so both replications share them (plan
   P9–P11, P10 revised). The torch `IndependentTotalProbabilityModel` and
   `TemperatureCalibrator` were deleted in sub-task 3.
3. **Combined (sub-task 5):** `TotalTimesProbabilityModel(total_model, probability_model)`
   (plan P12); delete `total_children.py`, `optimization.py` and their tests.
4. **Smoke run and docs (sub-tasks 6–7)** explaining every decision (plan §9).

## 2. State at handoff

| Item | State |
|---|---|
| Branch | `feat/total-times-probability-model`, HEAD **`b525954`**, pushed, tree clean |
| Commits | `02b7725` plan + handoff (0); `9f9fb58` the rename (1); `732e465` NB2 and `CountModel`'s exposure cases (2); `1e2bc66` handoff; **`b525954` `CohortProbabilityModel` on a classifier, the torch Model 2 and `TemperatureCalibrator` deleted (3, with its four revisions)** |
| PR | Draft **#13** into `feat/hyperparameter-tuning`; its body has the sub-task checklist (the user ticks it) |
| Uncommitted | **Sub-task 4** (calibration): `modeling/cohort_probability.py`, `modeling/__init__.py`, `tests/unit/test_modeling_cohort_probability.py`, the plan, this file; the user commits |
| Suite | **1238 passed, 1 skipped, 1 xfailed** after sub-task 4 (`uv run pytest -m "not slow"`; pytest collects only `tests/`) |
| Next | **Sub-task 5** (`TotalTimesProbabilityModel`, deletions; §5 below). Then 6–7, one per stop |

## 3. Decisions (settled; do not reopen)

From planning:
- **No torch** in `modeling/`; `total_children.py` and `optimization.py` (and their tests,
  `tests/validation/test_total_children.py`) are deleted in sub-task 5.
- **Rename `DirectCohortModel` → `CountModel`**; `DIRECT_COHORT_MODEL.md` keeps its file name.
- **The categorical rows are built inside `CohortProbabilityModel.fit`**, each child counting
  once. *The planning-time "no `replication` setting" was **overridden by the user in sub-task
  3*** (below). The docs must still carry the five-variant RF table, the unbiasedness argument
  (`E[m_i] = 1`), the observed-vs-unobserved building effect note and the XGBoost note.
- **Calibration:** `CalibratedClassifierCV` in the model, `ensemble=False` (cross-fitting),
  5 folds grouped by sample, temperature/sigmoid/isotonic (isotonic documented as unsuitable at
  this size), `None` default.
- **Name:** `TotalTimesProbabilityModel`, settings `total_model`, `probability_model`.
- Everything else: plan §5 P1–P15 (with the sub-task 3 revision notes in P5–P8), evidence in §6.

From sub-tasks 1–2 (the user's answers; plan §8 records):
- `docs/validation/pr12/` deleted; `testpaths = ["tests"]`.
- **NB2 takes the raw `exposure`** (statsmodels' meaning), **exposure only, no `offset`**;
  statsmodels is a main dependency.
- **`NegativeBinomialRegressor`:** BFGS for the main and the preliminary fit;
  `add_constant(has_constant="add")`; `skip_hessian=True`; only statsmodels'
  `ConvergenceWarning` silenced, non-convergence raises `RuntimeError`; no validation of `y`.
- **`CountModel`:** the cases inline as `if / elif / elif / else: raise TypeError`; `TypeGuard`s
  over `has_fit_parameter`, no `cast`; a `**fit_params` wrapper is refused under `use_exposure`.

From sub-task 3 (the user's answers; plan §8, the sub-task 3 record and revisions 1–4):
- **Order:** the torch baseline was measured first (`COHORT_LOG_LOSS` 1.0819; the new model
  with `LogisticRegression(C=inf)` 1.0783), then the torch class's dependents were deleted.
- **`replication: ReplicationType = "weighted"`** (`Literal["per_child", "weighted"]`):
  measured identical for LR, within noise for the trees; `"per_child"` 2–3× slower, admits
  classifiers without `sample_weight` (KNN, LDA, QDA, Gaussian process, `OneVsRestClassifier`
  without routing), needs an integer dtype numpy casts safely to `int64`.
- **Labels are the cohort positions**, not the names (names are sorted by every classifier,
  and mixed-type names fail).
- **No one-vs-rest detection or wrap** (every common classifier fits K classes; binary-only
  ones raise); the caller wraps. **`predict` raises if a row does not sum to 1.**
- **Checks kept:** `check_consistent_length(X, y)` (the classifier's rows come from `y`, so a
  shorter `y` would fit silently), every cohort observed (by position), rows sum to 1,
  `check_is_fitted`. **Removed as not essential:** `validate_data` (matters only without a
  feature transformer and with LightGBM), a classes check (guarded a `FrozenEstimator` misuse).
  The checks are private static methods (`_check_every_cohort_observed`,
  `_check_rows_sum_to_one`).
- **Not checked in the model:** count values (validated where the data is prepared). A classifier
  without `sample_weight` under `"weighted"` fails in the library (measured: `TypeError` or
  `ValueError`; Bagging uses the weights), left so.
- **Follow-ups in plan §10, not this PR's sub-tasks:** a counts validator in `preprocessing.py`;
  reordered columns pass silently for `CountModel` and `CohortProbabilityModel` with LightGBM and
  no feature transformer.

From sub-task 4 (the user's answers; plan §8, the sub-task 4 record):
- **A calibration fold lacking a cohort: documented only** (`decision_function` classifiers
  raise; `predict_proba`-only ones warn and see the cohort at 0; never on the simulator).
- **Folds of samples, before the categorical rows, round-robin** (`i mod calibration_cv`): folds
  of rows differed between the replications. No check on `calibration_cv` (bad values raise in
  sklearn; more folds than samples gives one per sample).
- **`"weighted"` + a classifier without `sample_weight` under calibration** only warns and fits
  unweighted: a docstring note (KNN goes under `"per_child"`).

## 4. Working rules (the user's; plan §3, plus what the sessions learned)

- Plan mode at the start of every sub-task (and of every revision the user asks for). **Right
  before `ExitPlanMode`, write a short numbered summary of the plan in the chat.** Open choices
  go to `AskUserQuestion` with the measured evidence; when the user answers with a question,
  answer it with evidence and ask again; give the full list of options they ask for.
- Baseline `uv run pytest -m "not slow"` before the first edit. ruff, ruff format, mypy (both
  runs: `uv run mypy` and `uv run mypy src/age_group_prediction/modeling <changed tests>`), the
  changed tests with `-W error`; **one mutation check per claimed behavior**: the scratchpad
  `mutate.py` pattern (replace a unique string, run the tests, restore by copy, compare md5;
  never `git checkout`/`stash`); make sure each mutation really changes behavior.
- An independent review subagent (say it takes 6–8 minutes; update the plan doc and run the
  suite meanwhile); reproduce each finding before fixing it.
- At each stop: file-by-file summary, check results, commit commands. **The user commits and
  pushes**; no `Co-Authored-By`, no "Generated with". **`git add` only paths that exist:** a
  deleted file's pathspec makes `git add` abort entirely (this happened: only the staged
  deletions were committed, and the user amended).
- **The user edits files between turns** (e.g. renamed variables): re-read a file before editing
  it, keep their names unless they ask otherwise, and say whether you agree with their change.
- After editing `pyproject.toml`, run uv with `--frozen` (the user runs `uv lock`). Probes:
  `PYTHONPATH=src .venv/bin/python …`; quote globs in zsh.
- **Code style:** no `else` that silently handles a case (an `else` that only raises is fine;
  name every case); no `cast`; checks as private static methods; **keep only essential checks**:
  each must guard a silent or late failure in the repo's real flow (say under which conditions it
  matters); never add a check for a misuse the plan already rules out; **general names**, not
  today's domain unit (scikit-learn's "sample", not "building"); informative type aliases
  (`ReplicationType`); the library's and the repo's vocabulary; comments concise, keeping the "why".
- Correct your own earlier wrong claims explicitly in the next message.

## 5. Sub-task 5 (`TotalTimesProbabilityModel`): files, what to re-verify

Files: new `modeling/total_times_probability.py` and `tests/unit/test_modeling_total_times_probability.py`;
delete `modeling/total_children.py`, `modeling/optimization.py`, `tests/unit/test_modeling_total_children.py`,
`tests/unit/test_modeling_optimization.py`, `tests/validation/test_total_children.py`; update
`modeling/__init__.py` (drop `Solver`, `TotalChildrenModel`; add `TotalTimesProbabilityModel`), the
contract test's `EXAMPLES`, `test_modeling_feature_transformer.py`. Plan P12 and the sub-task 5 bullets.
Re-verify before relying on them: every path above exists; who imports `total_children` /
`optimization` (the plan's `grep`); `tests/validation/helpers.py` and other validation files that
may name the deleted classes; the contract test's single-thread LightGBM note (P14 keeps it).
At the stop, `git add` the new files and stage the deletions with `git rm` (or `git add -A` on
existing directories only): a deleted path given to `git add` aborts it.

Memory to update at each stop: `multi-cohort-models-plan.md` in the Claude memory directory.
