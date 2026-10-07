# Handoff: rebuild Model 2 as `TotalTimesProbabilityModel`

**For:** the implementing session (Claude Opus 5.5). **Written:** 2026-10-07 at the end of
the planning session; **updated 2026-10-07 at the end of the first implementing session:
sub-tasks 0–2 are done and committed.**
**Plan (source of truth):** [TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md).
Read it in full before the first edit (its §8 records say what each sub-task did and why);
this file only orients you.

## 1. What the task is

Model 2 (`modeling/independent_total_probability.py`, PR #11) predicts each cohort as
**predicted total × predicted cohort probability**. Its halves are torch objectives
(`TotalChildrenModel`, a Dirichlet `CohortProbabilityModel`, `TemperatureCalibrator`,
`optimization.py`). The user wants them replaced by **library estimators**, no torch:

1. **Total (done, sub-tasks 1–2):** `DirectCohortModel`, **renamed `CountModel`**, with any
   regressor. The exposure is applied by named cases read from the estimator's `fit`
   signature: an estimator taking `exposure` gets the raw exposure (**NB2**:
   `NegativeBinomialRegressor`, statsmodels' `NegativeBinomial(loglike_method="nb2",
   exposure=…)`); one taking `sample_weight` gets the weighted rate; any other raises.
   `TotalChildrenModel` is deleted in sub-task 5.
2. **Probabilities (next, sub-tasks 3–4):** `CohortProbabilityModel(estimator: Classifier,
   calibration_method=None, calibration_cv=5, feature_transformer)`. `fit` converts the counts
   to one weighted row per building and cohort (weight = count) **inside the model**, after
   fitting the feature transformer on the buildings; optional
   `CalibratedClassifierCV(ensemble=False)` with `GroupKFold` splits by building; `predict`
   maps `predict_proba` by `classes_` into `y`'s column order (plan P5–P11).
   `TemperatureCalibrator` is deleted.
3. **Combined (sub-task 5):** `TotalTimesProbabilityModel(total_model, probability_model)`
   replaces `IndependentTotalProbabilityModel` (plan P12).
4. **Smoke run and docs (sub-tasks 6–7)** explaining every decision (plan §9): NB2
   representation, data replication, bagging under replicated rows, calibration.

## 2. State at handoff

| Item | State |
|---|---|
| Branch | `feat/total-times-probability-model`, HEAD `732e465`, pushed, tree clean |
| Commits | `02b7725` plan + handoff (sub-task 0); `9f9fb58` the rename (1); `732e465` NB2 and `CountModel`'s exposure cases (2, with its three revisions) |
| PR | Draft **#13** into `feat/hyperparameter-tuning`; its body has the sub-task checklist (the user ticks it) |
| Suite | **1247 passed, 1 skipped, 1 xfailed** (`uv run pytest -m "not slow"`; pytest collects only `tests/`) |
| Next | **Sub-task 3** (`CohortProbabilityModel` on a classifier), after the open question in §5. Then 4–7, one per stop |

## 3. Decisions (settled; do not reopen)

From planning:
- **No torch** in `modeling/`; delete `total_children.py`, `calibration.py`,
  `optimization.py` and their tests (sub-task 5, unless §5 moves it).
- **Rename `DirectCohortModel` → `CountModel`**; `DIRECT_COHORT_MODEL.md` keeps its file name.
- **Replication inside `CohortProbabilityModel.fit`**, weight = count (per child). Weighted rows
  and one row per child are equivalent for any weighted-likelihood estimator (1.8e-15); for
  RandomForest's bootstrap they differ as procedures but are **equal within noise** on held-out
  cross-entropy (five variants, 0.708–0.717 against a seed spread of 0.57–0.86). So **no
  `replication` setting and no grouped bagging** ("leave it, since only RF needs the
  adjustment"). The docs must carry the five-variant table, the unbiasedness argument
  (`E[m_i] = 1`), the observed-vs-unobserved building effect note and the XGBoost note.
- **Calibration:** `CalibratedClassifierCV` in the model, `ensemble=False` (cross-fitting),
  5 folds grouped by building, temperature/sigmoid/isotonic (isotonic documented as unsuitable
  at this size), `None` default.
- **Name:** `TotalTimesProbabilityModel`, settings `total_model`, `probability_model`.
- Everything else: plan §5 P1–P15, with the measured evidence in §6.

From implementing sub-tasks 1–2 (the user's answers; plan §8 records):
- **`docs/validation/pr12/` deleted** (kept in `c966edb`); `testpaths = ["tests"]`: tests live
  only in `tests/`.
- **NB2 takes the raw `exposure`** (statsmodels' meaning; identical parameters to
  `offset=log(exposure)`), **exposure only, no `offset`** argument. statsmodels is a main
  dependency (the `validation` group is gone).
- **`NegativeBinomialRegressor`:** BFGS for the main **and** the preliminary fit
  (`optim_kwds_prelim`; Newton fails on an all-zero dummy); `add_constant(has_constant="add")`;
  `skip_hessian=True` (standard errors unused); only statsmodels' `ConvergenceWarning` is
  silenced, and non-convergence raises `RuntimeError`; no validation of `y` (counts by
  construction); the exposure's shape is checked at `predict`.
- **`CountModel`:** the cases inline as `if / elif / elif / else: raise TypeError` in `fit` and
  `predict`; the checks are `TypeGuard`s over `has_fit_parameter` (`TypeIs` does not narrow
  these protocols), so no `cast`; a wrapper whose `fit` takes `**fit_params` (`Pipeline`,
  `TransformedTargetRegressor`, `GridSearchCV`) is refused under `use_exposure=True`.

## 4. Working rules (the user's; plan §3, plus what this session learned)

- Plan mode at the start of every sub-task. **Right before `ExitPlanMode`, write a short
  summary of the plan in the chat** (numbered, a few lines); the user rejected an approval
  request once for lacking it. Open choices go to `AskUserQuestion` with the measured evidence.
- Baseline `uv run pytest -m "not slow"` before the first edit. ruff, ruff format, mypy (both
  runs), the changed tests with `-W error`; **one mutation check per claimed behavior**: a
  scratchpad `mutate.py` (replace a unique string), the test run, restore by copy and compare
  md5; never `git checkout`/`stash`. Pass several test paths unquoted, and make sure a mutation
  really changes behavior (one malformed mutation "passed" this session).
- An independent review subagent (say it takes 6–8 minutes; update the plan doc and run the
  suite meanwhile); reproduce each finding before fixing it.
- At each stop: file-by-file summary, check results, commit commands. **The user commits and
  pushes**; no `Co-Authored-By`, no "Generated with".
- After editing `pyproject.toml`, run uv with **`--frozen`** (the user runs `uv lock`). Probes:
  `PYTHONPATH=src .venv/bin/python …`; quote globs in zsh.
- **Code style the user asked for:** no `else` that silently handles a case (an `else` that only
  raises is fine; name every case); no `cast` to placate mypy; silence a warning only if it does
  not affect results; don't raise unless necessary (validate what would pass silently); when the
  user says "best practice", decide and justify with measured evidence; use the library's and the
  repo's vocabulary (e.g. the raw `exposure`); comments concise, keeping the "why".
- Correct your own earlier wrong claims explicitly in the next message.

## 5. Sub-task 3: files, and an open question to raise first

Files: `src/age_group_prediction/modeling/cohort_probability.py` (rewritten on a classifier),
`tests/unit/test_modeling_cohort_probability.py` (rewritten), the contract test's `EXAMPLES`.

**Open question (found 2026-10-07; re-verify with `grep -rn predict_logits src tests`):** the
rewrite removes `CohortProbabilityModel.predict_logits`, which the torch build still uses:
`modeling/independent_total_probability.py:89`, `tests/unit/test_modeling_independent_total_probability.py:158`,
`tests/unit/test_modeling_calibration.py:236,242`, `tests/unit/test_modeling_feature_transformer.py:132-143`,
`tests/unit/test_modeling_cohort_probability.py:183` (rewritten anyway), plus the contract
`EXAMPLES` of the torch classes. The plan's sub-task 3 "Verify" probe (the new model's
`COHORT_LOG_LOSS` against the torch Dirichlet build on the same split) assumed the torch build
stays importable until sub-task 5. Plan sub-task 3 must settle the order before editing, with
the options and evidence, and ask the user. For example: run the torch baseline probe first
and record its number, then delete the old Model 2 stack in sub-task 3; or merge sub-task 3
with sub-task 5's deletions.

Memory to update at each stop: `multi-cohort-models-plan.md` in the Claude memory directory.
