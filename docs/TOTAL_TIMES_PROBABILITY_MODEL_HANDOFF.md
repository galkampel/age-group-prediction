# Handoff: rebuild Model 2 as `TotalTimesProbabilityModel`

**For:** the implementing session (Claude Opus 5.5). **Written:** 2026-10-07, at the
end of the planning session; nothing is implemented yet.
**Plan (source of truth):** [TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md).
Read it in full before the first edit; this file only orients you.

## 1. What the task is

Model 2 (`modeling/independent_total_probability.py`, PR #11) predicts each cohort as
**predicted total × predicted cohort probability**. Its halves are torch objectives
(`TotalChildrenModel`, a Dirichlet `CohortProbabilityModel`, `TemperatureCalibrator`,
`optimization.py`). The user wants them replaced by **library estimators**, no torch:

1. **Total:** `DirectCohortModel`, **renamed `CountModel`**, with any regressor
   (`PoissonRegressor`, `LGBMRegressor`, `Ridge`), with or without the exposure as today
   (the weighted rate), **plus NB2** through a new scikit-learn regressor
   `NegativeBinomialRegressor` (statsmodels `NegativeBinomial`, `loglike_method="nb2"`)
   whose `fit`/`predict` take `offset`. `CountModel` detects an offset estimator with
   `sklearn.utils.validation.has_fit_parameter(estimator, "offset")` and passes
   `offset=log(exposure)` instead of the rate (plan P3–P4). `TotalChildrenModel` is deleted.
2. **Probabilities:** `CohortProbabilityModel(estimator: Classifier, calibration_method=None,
   calibration_cv=5, feature_transformer)`. `fit` converts the counts to one weighted row
   per building and cohort (weight = count) **inside the model**, after fitting the feature
   transformer on the buildings; optional `CalibratedClassifierCV(ensemble=False)` with
   `GroupKFold` splits by building; `predict` maps `predict_proba` by `classes_` into `y`'s
   column order (plan P5–P11). `TemperatureCalibrator` is deleted.
3. **Combined:** `TotalTimesProbabilityModel(total_model, probability_model)` replaces
   `IndependentTotalProbabilityModel` (plan P12).
4. **Docs** explaining every decision (plan §9): NB2 representation, data replication,
   bagging under replicated rows, calibration.

## 2. State at handoff

| Item | State |
|---|---|
| Branch | Planning happened on `feat/hyperparameter-tuning` at `c966edb` (clean tree except the two new docs). Sub-task 0 creates `feat/total-times-probability-model` from it and a draft PR into it, like PRs #10–#12 |
| Uncommitted | The plan and this handoff were committed with sub-task 0 (`02b7725`); each later sub-task is committed by the user at its stop |
| Suite | after sub-task 1 (2026-10-07): **1230 passed, 1 skipped, 1 xfailed** (`uv run pytest -m "not slow"`; `testpaths = ["tests"]`) |
| Next | Sub-tasks 0–1 are done (plan §8 records). **Sub-task 2** (`NegativeBinomialRegressor` and `CountModel`'s offset branch) next. Then 3–7, one per stop |

## 3. Decisions the user made during planning (settled; do not reopen)

- **No torch** in `modeling/`; delete `total_children.py`, `calibration.py`,
  `optimization.py` and their tests (sub-task 5).
- **NB2 via statsmodels, as an offset regressor inside `CountModel`**, not as its own
  `BaseAgeGroupModel` and not through the weighted rate (measured: the rate form fits a
  different NB2, coefficients differ by 0.021; exact for Poisson, 2e-13). Unpenalized;
  statsmodels moves to the main dependencies. "Two-step" GLM-with-fixed-α is noted as a
  later option only.
- **Rename `DirectCohortModel` → `CountModel`** (chosen over `CountRegressionModel`), in its
  own sub-task before any behavior change. `DIRECT_COHORT_MODEL.md` keeps its file name.
- **Replication inside `CohortProbabilityModel.fit`**, weight = count (per child). The
  user asked whether weighted rows and one row per child are equivalent: yes for any
  weighted-likelihood estimator (1.8e-15); for RandomForest's bootstrap they are different
  procedures but **equal within noise** on held-out cross-entropy (five variants incl. a
  hand-made bootstrap by building: 0.708–0.717 against a seed spread of 0.57–0.86). So
  **no `replication` setting and no grouped bagging**: "leave it, since only RF needs the
  adjustment". The docs must carry the five-variant table, the unbiasedness argument
  (`E[m_i] = 1` for every unit), the note that observed building features (type, year) do
  not change the case while an *unobserved* building effect or a need for intervals does,
  and the XGBoost note (all rows by default, `subsample=1.0`; not a dependency).
- **Calibration:** `CalibratedClassifierCV` in the model, `ensemble=False` (cross-fitting,
  the user's preference), 5 folds grouped by building, temperature/sigmoid/isotonic with
  isotonic documented as unsuitable at this size, `None` default.
- **Name:** `TotalTimesProbabilityModel`, settings `total_model`, `probability_model`.
- Everything else: plan §5 P1–P15, with the measured evidence in §6.

## 4. Working rules (the user's; plan §3)

Plan mode at the start of every sub-task with a short chat summary before asking approval;
baseline before the first edit; ruff/mypy/`-W error` tests; one mutation check per claimed
behavior (edit `src` in place, restore by copy with md5, never `git checkout`/`stash`); an
independent review subagent (say it takes 6–8 minutes; update the plan doc meanwhile);
file-by-file summary and commit commands at each stop; **the user commits**; no
`Co-Authored-By`, no "Generated with". Justify every class, field and check with measured
evidence, or drop it. Probes: `PYTHONPATH=src .venv/bin/python -c "..."` (do not let
`uv run` re-sync the venv); quote globs in zsh.

## 5. Files you will touch first

Sub-tasks 0–1 are done (the rename: `modeling/count_model.py`, `CountModel`). Sub-task 2:
- `pyproject.toml` (`statsmodels>=0.14.5` into `dependencies`).
- New `src/age_group_prediction/modeling/negative_binomial.py`; `modeling/count_model.py`
  (`OffsetRegressor`, the `has_fit_parameter` branch); `modeling/__init__.py`.
- New `tests/unit/test_modeling_negative_binomial.py`; `tests/unit/test_modeling_count_model.py`.
- Memory to update at the end: `multi-cohort-models-plan.md` in the Claude memory directory.
