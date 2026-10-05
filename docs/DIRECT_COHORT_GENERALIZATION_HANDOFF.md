# Handoff: generalize `DirectCohortModel` and move the feature transformer into the models

**For:** the implementing session (Sonnet or Opus). **Written:** 2026-10-05.
**Plan (source of truth):** [DIRECT_COHORT_GENERALIZATION_PLAN.md](DIRECT_COHORT_GENERALIZATION_PLAN.md). Read it in full before the first edit; this file only orients you.

## 1. What the task is

Two changes to the scikit-learn-style `modeling/` package, plus the consequences:

1. **`DirectCohortModel` takes any regressor** (`estimator=`) with a Poisson or Gaussian
   loss, instead of a hard-wired `LGBMRegressor` with ten mirrored hyperparameters. The
   exposure offset becomes a **weighted regression of the per-apartment rate**
   (`fit(X, y / exposure, sample_weight=exposure)`, `predict(X) * exposure`), which is the
   same Poisson likelihood as LightGBM's `init_score` offset and works for every estimator
   that accepts `sample_weight`. The derivation is plan §2b; it goes into the model doc.
2. **`feature_transformer` becomes a parameter of each leaf model** (`DirectCohortModel`,
   `TotalChildrenModel`, `CohortProbabilityModel`), with the fit/transform logic as helpers
   in `BaseAgeGroupModel`. `ModelPipeline` is deleted. The tuning evaluator then fits the
   model on the raw rows and loses its own `feature_transformer` field.

Two user questions about `IndependentCohortModels` are answered in plan §2 (I1, I2) and
need **no code change**: keep `dict` (not `OrderedDict`); both `cohort_models` (template)
and `cohort_models_` (fitted copies) are required by the sklearn contract.

## 2. State at handoff

| Item | State |
|---|---|
| Branch | `feat/estimator-and-feature-transformer`, cut from `feat/hyperparameter-tuning` (where `modeling/` lives; `main` does not have it). PR goes into `feat/hyperparameter-tuning` |
| Done | Step 0: branch, the plan doc, its `docs/README.md` row, baseline recorded |
| Baseline | `uv run pytest -m "not slow"`: 1221 passed, 1 skipped, 1 xfailed |
| Uncommitted | `docs/DIRECT_COHORT_GENERALIZATION_PLAN.md` (new), `docs/README.md`, this file. The user commits |
| Still to do in Step 0 | the first commit and the **draft PR** (title and body in plan §7) |
| Next | Step 1 (base-class helpers), then Steps 2–7, one at a time |

## 3. How the user works (non-negotiable)

Plan §3 restates `docs/MULTI_COHORT_MODELS_PLAN.md` §3. The short form:

- **One step per approval.** Enter plan mode, re-verify the step's facts (paths, line
  numbers, library behaviour), post a short summary of the step in chat, wait for approval,
  implement, run the routine, update the plan doc's checkboxes and status, **stop**.
- **Routine per step:** `uv run ruff check` and `uv run ruff format` on the changed files →
  `uv run mypy` and `uv run mypy src/age_group_prediction/modeling <changed tests>` → changed
  tests with `-W error` → one mutation check per claimed behaviour → an independent review
  subagent, each finding reproduced before fixed or rejected → `uv run pytest -m "not slow"`.
- **At each stop:** file-by-file summary of the `.py` changes, the check results, suggested
  commit commands. **The user commits and pushes.** No `Co-Authored-By`, no "Generated with"
  footer, in commits or the PR.
- **Justify every class, field and check**, or drop it. Validate only what would otherwise
  pass silently; leave to the library what it already raises.
- **Style** as `modeling/direct_cohort.py`: keyword-only constructors storing arguments
  verbatim, validation in `fit`, fitted state in trailing-underscore attributes assigned
  together after success, `predict` follows the fitted state, docstrings that say *why*,
  precise types, lower-case error messages that say what to do. Shared logic in `modeling`
  goes under a class; there is no utils file.
- **Tests:** each test's name and comment state the mistake it catches; no test that only
  checks a library.
- **Docs in the same PR**; run every code block you put in a doc.
- **If a plan assumption breaks, stop** and show the options with evidence.
- Probes: `PYTHONPATH=src uv run --group test python -c "..."`. Quote globs in zsh.

## 4. Facts already verified (do not re-derive, do re-check line numbers)

- `sklearn.utils.estimator_checks.check_parameters_default_constructible` allows only
  `None`, scalars, tuples, types and callables as defaults, but accepts a required
  argument with no default. So `estimator` is required (plan G2, revised 2026-10-05);
  there is no `default_estimator()`.
- `clone(model).set_params(estimator__n_estimators=5)` works on a `BaseEstimator` with an
  estimator parameter and leaves the template untouched.
- Weighted rate vs `init_score` on LightGBM (2000 rows, 50 trees, Poisson): predictions
  agree to 1.5e-8 relative. `HistGradientBoostingRegressor(loss="poisson")` and
  `PoissonRegressor` fit the same formulation without changes. `xgboost` is not installed.
- `has_fit_parameter(LGBMRegressor(), "init_score")` is True; False for sklearn estimators.
- Installed: scikit-learn ≥ 1.9, lightgbm 4.x, torch. mypy strict on `modeling.*`; sklearn
  and lightgbm have no stubs (`ignore_missing_imports`), so a `Protocol` carries the type.

## 5. Where things are

| What | Where |
|---|---|
| Base class, `_check_exposure` | `src/age_group_prediction/modeling/base.py` |
| Model to generalize | `src/age_group_prediction/modeling/direct_cohort.py` |
| Leaf models that also gain `feature_transformer` | `modeling/total_children.py`, `modeling/cohort_probability.py` |
| Composites (docstrings only) | `modeling/independent_cohorts.py`, `modeling/independent_total_probability.py` |
| To delete | `modeling/pipeline.py`, `tests/unit/test_modeling_pipeline.py` |
| Evaluator (Step 5) | `src/age_group_prediction/hyperparameter_tuning/evaluator.py`, `tests/unit/test_hyperparameter_tuning_evaluator.py` |
| Contract test (discovers every model; needs an `EXAMPLES` entry each) | `tests/unit/test_modeling_contract.py` |
| Feature transformer | `src/age_group_prediction/feature_engineering/transformer.py` (`FeatureTransformer`) |
| Docs to update (Step 6) | `docs/DIRECT_COHORT_MODEL.md` §0, `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0, `docs/MODULE_REFERENCE.md`, `docs/HYPERPARAMETER_TUNING_PLAN.md` (§5, D13, §6 task), `docs/FEATURE_TRANSFORMATIONS.md`, `docs/MULTI_COHORT_MODELS_PLAN.md`, `docs/MODEL_REIMPLEMENTATION_PLAN.md` |
| Old stack (never import from `modeling`) | `src/age_group_prediction/models/`; the contract test checks this |

## 6. Pitfalls

- `test_every_shipped_model_has_an_example` fails the moment a model is added or removed
  without touching `EXAMPLES` in the contract test (Step 4 removes `ModelPipeline`).
- `test_hyperparameter_tuning_evaluator.py` tunes `n_estimators` on a `DirectCohortModel`;
  after Step 2 that is `estimator__n_estimators` on an explicit `LGBMRegressor`.
- `CohortProbabilityModel` uses `validate_data(self, X, reset=...)`; after Step 3 it must see
  the design matrix, not the raw table, on both paths.
- LightGBM ignores `subsample` unless `subsample_freq >= 1`; the old model derived it, the
  new one leaves it to the estimator. The tuning plan's §5 example must set it.
- Fitted state is assigned together after success: `_fit_features` returns the fitted copy;
  do not assign `feature_transformer_` inside the helper.
- The simulator also has a `pipeline.py` (`src/student_simulator/pipeline.py`, row 52 of
  `docs/MODULE_REFERENCE.md`); it is unrelated and stays. Delete only `modeling/pipeline.py`
  and its row (row 104).
