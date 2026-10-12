# Handoff: generalize `DirectCohortModel` and move the feature transformer into the models

**For:** the implementing session (Sonnet or Opus). **Written:** 2026-10-05; **updated
2026-10-05 after Step 7**: the plan is complete.
**Plan (source of truth):** [DIRECT_COHORT_GENERALIZATION_PLAN.md](DIRECT_COHORT_GENERALIZATION_PLAN.md). Read it in full before the first edit; this file only orients you.

## 1. What the task is

Two changes to the scikit-learn-style `modeling/` package, plus the consequences:

1. **`DirectCohortModel` takes any regressor** with a Poisson or Gaussian loss, through a
   **required** `estimator=` argument (no default, no `default_estimator()`; typed by the
   `Regressor` Protocol), instead of a hard-wired `LGBMRegressor` with ten mirrored
   hyperparameters. The exposure offset is a **weighted regression of the per-apartment
   rate** (`fit(X, y / exposure, sample_weight=exposure)`, `predict(X) * exposure`): for a
   Poisson loss the same likelihood as LightGBM's `init_score` offset, for a Gaussian loss
   least squares of the count with variance ∝ exposure. Both derivations are plan §2b
   ((A)–(C) and (D1)–(D4)); Step 6 copies them into the model doc.
2. **`feature_transformer` is a parameter of each leaf model** (`DirectCohortModel`,
   `TotalChildrenModel`, `CohortProbabilityModel`), with the fit/transform logic as helpers
   in `BaseAgeGroupModel`. Still to do: delete `ModelPipeline` (Step 4); the tuning
   evaluator fits the model on the raw rows and loses its own `feature_transformer` field
   (Step 5); docs (Step 6); final checks (Step 7).

Two user questions about `IndependentCohortModels` are answered in plan §2 (I1, I2) and
need **no code change**: keep `dict` (not `OrderedDict`); both `cohort_models` (template)
and `cohort_models_` (fitted copies) are required by the sklearn contract. Step 4 adds the
I1 sentence to the docstring.

## 2. State at handoff

| Item | State |
|---|---|
| Branch | `feat/estimator-and-feature-transformer`, cut from `feat/hyperparameter-tuning` (where `modeling/` lives; `main` does not have it) |
| PR | **#12, draft**, into `feat/hyperparameter-tuning`. The user applies body updates; give them the ticked Steps checklist at each stop |
| Done and committed | Steps 0–6: `e09c58a`, `8fd73f7`, `ea67cd7`, `229c22d`, `adff045`/`6334a90` (Step 4), `1ef8da5` (Step 5), `e8e654d` (Step 6); Step 7 (final checks, PR body) done 2026-10-05 |
| Suite | `uv run pytest -m "not slow"`: **1230 passed, 1 skipped, 1 xfailed** (1221 before the PR). The 6 warnings come from mlflow in `test_gate8_tracking` |
| Working tree | clean once Step 7's doc updates are committed |
| Next | **Nothing: the plan is complete.** The PR awaits the user's review; the follow-up (failed-refit mixed state, §2b) is a separate PR |

### 2a. Decisions made in Steps 1–3 (settled; do not reopen)

- **The estimator is always given explicitly.** Every caller builds it, e.g.
  `LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1)`; `n_jobs=1` because more
  OpenMP threads crash alongside torch on macOS. Tests and doc examples follow this.
- **The exposure is the weighted rate for every estimator**; `use_exposure=True` is allowed
  with a Gaussian loss (no `objective` guard: the model cannot read an arbitrary estimator's
  loss). A penalized scikit-learn GLM normalizes `sample_weight`, so its `alpha` acts as
  `alpha * mean(exposure)` in the offset model: documented, not corrected.
- **`y` is divided as given** (`Series / ndarray` divides by position): no numpy conversion,
  no `ndim` check.
- **In `fit`, `X` is rebound to the design matrix**: `feature_transformer, X =
  self._fit_features(X, y)`. The user's naming choice; do not introduce `X_design`.
- **`CohortProbabilityModel.predict` does not transform**; only `predict_logits` does (it is
  called by `predict`; a second transform would pass silently, since the design matrix keeps
  the raw column names).
- **`estimator_`, `use_exposure_` and the `Regressor` Protocol stay** (plan G1b, G6).
- **Estimator instance, not kwargs or a dict** (plan G1a): `clone` drops `**kwargs`, and
  `set_params` cannot reach inside a dict.

### 2b. Answered: the failed-refit mixed state is deferred to a separate PR (the user, 2026-10-05)

Pre-existing since PR #11, found by the Step 3 review: in `TotalChildrenModel` and
`CohortProbabilityModel`, a refit whose final `validate_data(self, X, reset=True)` raises
(e.g. a non-string column name) leaves the new `coef_` with the old `feature_names_in_`
(probed: `coef_` shape (3,), names `['a', 'b']`); `feature_transformer_` now joins that
mixed state. The comment "Recorded after success, so a failed refit leaves the previous fit
whole" is therefore not quite true. Fix: check the names before any fitted state is
assigned. The user chose a separate PR: this PR does not touch those two `fit` methods.

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
  commit commands, the ticked Steps checklist for the PR body. **The user commits and
  pushes.** No `Co-Authored-By`, no "Generated with" footer, in commits or the PR.
- **Justify every class, field and check**, or drop it. Validate only what would otherwise
  pass silently; leave to the library what it already raises. The user prefers less code:
  no default they did not ask for, no conversion or check without a silent failure behind
  it. When they ask "why X?" or "is X necessary?", answer each point with probe evidence and
  offer to remove what is not needed; they often ask for exactly that.
- **Style** as `modeling/direct_cohort.py`: keyword-only constructors storing arguments
  verbatim, validation in `fit`, fitted state in trailing-underscore attributes assigned
  together after success, `predict` follows the fitted state, docstrings that say *why*,
  precise types, lower-case error messages that say what to do. Shared logic in `modeling`
  goes under a class; there is no utils file.
- **Tests:** each test's name and comment state the mistake it catches; no test that only
  checks a library. Drop a planned test when another already catches its mistake, or when
  the mistake would fail loudly anyway, and say so.
- **Docs in the same PR**; run every code block you put in a doc.
- **If a plan assumption breaks, stop** and show the options with evidence.
- **The review subagent takes 6–8 minutes.** Tell the user when you start it, run the suite
  and update the plan doc meanwhile; the user has asked "what is taking so long?" twice.
- Probes: `PYTHONPATH=src uv run --group test python -c "..."`. Quote globs in zsh.

## 4. Facts already verified (do not re-derive, do re-check line numbers)

- `check_parameters_default_constructible` allows only `None`, scalars, tuples, types and
  callables as defaults, but accepts a required argument with no default (hence the
  required `estimator`; `IndependentCohortModels(cohort_models)` is another). The contract
  test builds every model from an explicit example in `EXAMPLES`.
- `clone(model).set_params(estimator__n_estimators=5)` changes only the copy; the
  evaluator builds each trial with `clone(self.model).set_params(**params)`.
- Weighted rate vs `init_score` on LightGBM: predictions agree to 1.7e-8 relative.
  `HistGradientBoostingRegressor(loss="poisson")` and `PoissonRegressor` fit the same
  formulation. `xgboost` is not installed.
- `FeatureTransformer` output keeps the table's index and the raw column names (a Center
  plan on `x` gives the column `x`). Its `fit` ignores `y`.
- A single centred column cannot reveal a skipped transformer: trees ignore a shift and an
  intercept absorbs it. Tests of the transformer path need columns the transformer drops
  (see `tests/unit/test_modeling_feature_transformer.py`).
- Installed: scikit-learn ≥ 1.9, lightgbm 4.x, torch. mypy (project config) on
  `modeling.*`; sklearn and lightgbm have no stubs, so a `Protocol` carries the type.
  `mypy --strict` is not the project's mode (other `modeling` files already fail it).

## 5. Where things are

| What | Where |
|---|---|
| Base class, `_check_exposure`, `_fit_features`, `_transform_features` | `src/age_group_prediction/modeling/base.py` |
| Leaf models (done) | `modeling/direct_cohort.py`, `modeling/total_children.py`, `modeling/cohort_probability.py` |
| Composites (docstrings only, Step 4) | `modeling/independent_cohorts.py`, `modeling/independent_total_probability.py` |
| To delete (Step 4) | `modeling/pipeline.py`, `tests/unit/test_modeling_pipeline.py` |
| `ModelPipeline` users to rewrite (Step 4) | `modeling/__init__.py`, `tests/unit/test_modeling_contract.py`, `test_modeling_independent_cohorts.py`, `test_modeling_independent_total_probability.py`, `test_modeling_calibration.py` (around line 225) |
| Transformer-path tests (Step 3) | `tests/unit/test_modeling_feature_transformer.py` |
| Evaluator (Step 5) | `src/age_group_prediction/hyperparameter_tuning/evaluator.py`, `tests/unit/test_hyperparameter_tuning_evaluator.py` |
| Contract test (discovers every model; needs an `EXAMPLES` entry each) | `tests/unit/test_modeling_contract.py` |
| Feature transformer | `src/age_group_prediction/feature_engineering/transformer.py` (`FeatureTransformer`) |
| Docs to update (Step 6) | `docs/DIRECT_COHORT_MODEL.md` §0, `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0, `docs/MODULE_REFERENCE.md` (row 100 still lists `Objective` and `ModelPipeline`), `docs/HYPERPARAMETER_TUNING_PLAN.md` (§5, D13, §6 task), `docs/FEATURE_TRANSFORMATIONS.md`, `docs/MULTI_COHORT_MODELS_PLAN.md`, `docs/MODEL_REIMPLEMENTATION_PLAN.md` |
| Old stack (never import from `modeling`) | `src/age_group_prediction/models/`; the contract test checks this |

## 6. Pitfalls

- **The plan's Step 4–6 text was written before Steps 1–3.** Re-verify every claim against
  the code. Known spots: Step 4's grep `"ModelPipeline\|pipeline"` will also hit
  unrelated lower-case "pipeline" (comments in `test_modeling_calibration.py`, e.g. "a
  pipeline's fold"), so judge hits rather than demand an empty result; Step 5's
  `_RecordingTransformer` goes into `DirectCohortModel(estimator=LGBMRegressor(...),
  feature_transformer=...)`; Step 6's §0.2 list has no `default_estimator()`.
- `test_every_shipped_model_has_an_example` fails the moment a model is added or removed
  without touching `EXAMPLES` in the contract test (Step 4 removes `ModelPipeline`).
- **Mutation checks must edit the repo's `src` in place** (back up the file, mutate, run,
  restore, `git status` to confirm). `pyproject.toml` sets pytest's `pythonpath = ["src"]`,
  so a mutated copy of `src` on `PYTHONPATH` is silently ignored.
- **The review subagent's prompt must forbid `git stash`, `git checkout` and `git reset`**
  (one reviewer tried a stash; the permission system blocked it).
- LightGBM ignores `subsample` unless `subsample_freq >= 1`; the model leaves it to the
  estimator. The tuning plan's §5 example (Step 5) must set it.
- Fitted state is assigned together after success: `_fit_features` returns the fitted copy;
  do not assign `feature_transformer_` inside the helper.
- The simulator also has a `pipeline.py` (`src/student_simulator/pipeline.py`, row 52 of
  `docs/MODULE_REFERENCE.md`); it is unrelated and stays. Delete only `modeling/pipeline.py`
  and its row (row 104).
