# Plan: complete Model 1 (independent cohorts) and rebuild Model 2 (total × probability)

**Written for:** the implementing model (Claude Opus 5.5) and the user who validates each step.
This file is self-contained: it assumes no memory of the planning conversation.
This file is the source of truth: update its status line and checkboxes as steps finish.

**Status (2026-09-29):** A0 done (draft PR #10). A1 done, awaiting the user's
commit. Non-slow suite: **1082 passed** (1 skipped, 1 xfailed). Next: A2.

## Contents
1. Context and goal
2. Branches and PRs
3. How to work (rules for the implementer)
4. What exists today
5. Decisions
6. Measured evidence
7. Target code shape
8. Steps, PR A (Model 1)
9. Steps, PR B (Model 2)
10. Old tests: carried over or dropped
11. Risks and pitfalls
12. Verification

---

## 1. Context and goal

The repo predicts children per building for three cohorts (`n_kindergarten`,
`n_elementary`, `n_highschool`; total `n_children_total`; exposure
`n_apartments`; split key `neighborhood_id`). The old models (`models/`,
`modeling_config.py`) cannot take fixed hyperparameters, so they are being
rebuilt as small scikit-learn-style classes in `src/age_group_prediction/modeling/`.
Model 1 and Model 2 here are the other docs' **Model A** (direct cohort) and
**Model B** (independent total and probability).
Model 1's single-cohort class exists; two things are missing:

- **Model 1 completion.** Three independent single-cohort models combined into
  one object that predicts every cohort. Each cohort has its **own model and its
  own feature transformer**.
- **Model 2 rebuild.** Two independent models whose predictions are multiplied:
  1. a **total-children** model: a count regression for the building's total;
  2. a **cohort-probability** model: a multinomial model giving (calibrated)
     probabilities of each cohort, from building and neighborhood features.

  Final prediction per cohort = total mean × cohort probability.
  It must import **nothing** from the old stack.

## 2. Branches and PRs

- **Base branch: `feat/hyperparameter-tuning`.** `modeling/`, `scoring.py` and
  `hyperparameter_tuning/` exist only there (33 commits ahead of `main`; draft
  PR #5 `main ← feat/hyperparameter-tuning` is still open). PRs #6–#9 used the
  same base.
- **Two PRs, in order:**

| PR | Branch | Content | Why separate |
|---|---|---|---|
| A | `feat/independent-cohort-models` | Model 1 completion (steps A0–A4) | Small. It settles what B builds on: the widened base contract, the contract test, the raw-table helpers, the DataFrame output |
| B | `feat/independent-total-probability-model` | Model 2 (steps B0–B9) | Large. Branched from `feat/hyperparameter-tuning` **after A is merged** |

- One PR would mix a ~150-line change with a ~10-step rebuild, and a change
  requested in A's conventions would then be reworked inside B's code.
- PR #5's diff will now also carry these models; note it in PR #5's description.

## 3. How to work (rules for the implementer)

These are the user's standing rules. Follow them exactly.

1. **One step at a time.** Each step in §8–§9 is a validation stop. Start a
   step in plan mode: re-verify this plan's facts for that step (paths, line
   numbers, library behavior), write a **short summary of the step in the chat**,
   then ask for approval. Implement only after approval.
2. **Baseline first.** Before a step's first edit run `uv run pytest -m "not slow"`.
   If it fails, stop and report. Record the passing count in the doc.
3. **If an assumption in this plan breaks, stop** and show the options with
   evidence. Do not work around it or change course silently.
4. **Routine per step:** implement → `uv run ruff check <files>` and
   `uv run ruff format <files>` (changed files only, paths listed explicitly) →
   `uv run mypy` plus `uv run mypy src/age_group_prediction/modeling <changed test files>` →
   the changed tests with `-W error` → **one mutation check per claimed
   behavior** (break the code, see the named test fail, restore) → an
   **independent review subagent**, each finding reproduced before it is fixed
   or rejected → the non-slow suite → update the plan doc → stop.
5. **At each stop** give: a file-by-file summary of the `.py` changes, the check
   results, and suggested commit commands. **The user commits and pushes.**
   Never commit without explicit approval. No `Co-Authored-By` line and no
   "Generated with" footer.
6. **Justify every class, field and check**, or drop it. Validate only what
   would otherwise pass silently; leave to the library what it already raises.
7. **Code style** (match `modeling/direct_cohort.py`): one-line module
   docstring, `from __future__ import annotations`, explicit `__all__`;
   keyword-only constructors that store arguments verbatim; validation in
   `fit`; fitted state in trailing-underscore attributes assigned together only
   after success; `predict` follows the fitted state, not current settings;
   short reST docstrings that say *why*; sparse comments for non-obvious
   reasons; precise types with named aliases; lower-case error messages that
   say what to do.
8. **Names:** `feature_transformer`, `exposure_*` (never `n`/`N` as a public
   name), `train`/`val`. Shared helpers go in a **public** `utils.py`.
9. **Tests:** each test's name and comment state the mistake it catches. No
   test that only checks a library.
10. **Docs** are updated in the same PR. Run every code block you put in a doc.
11. **Probes** (read-only, never write files in the repo):
    `PYTHONPATH=src uv run --group test python -c "..."`. zsh does not split
    `$VAR`, and an unquoted `--include=*.py` fails: quote globs.

## 4. What exists today

| File | What it gives |
|---|---|
| `src/age_group_prediction/modeling/base.py` | `BaseAgeGroupModel(BaseEstimator, ABC)`: abstract `fit(X, y: pd.Series, exposure=None) -> Self`, abstract `predict(X, exposure=None) -> np.ndarray`, concrete `evaluate(y_true, y_pred, metric) -> float`. *Since A1: `y: pd.Series \| pd.DataFrame`, `predict -> np.ndarray \| pd.DataFrame`* |
| `src/age_group_prediction/modeling/direct_cohort.py` | `DirectCohortModel`: LightGBM for **one** cohort, on a finished design matrix; `use_exposure`; `_check_exposure` (lines 70–91) |
| `src/age_group_prediction/scoring.py` | `Metric(name, function, greater_is_better=False)`, `POISSON_DEVIANCE`, `RMSE`, `MAE` |
| `src/age_group_prediction/feature_engineering/transformer.py` | `FeatureTransformer(plans, *, interactions, exposure_column, remainder)`; `fit`, `transform -> DataFrame`, `log_exposure` |
| `src/age_group_prediction/hyperparameter_tuning/evaluator.py` | `CVHyperparameterEvaluator`; `build_feature_transformer_and_model(params) -> (FeatureTransformer, BaseAgeGroupModel)`; `evaluate(..., y: pd.Series \| np.ndarray, ...)` at line 83 |
| `src/age_group_prediction/utils.py` | `DesignMatrix`, `Target`, `Groups`, `Exposure`, `take_rows` |
| `tests/unit/test_modeling_contract.py` | Discovers every concrete model in `modeling`, builds it with `model_class()`, runs 4 sklearn checks and a refit test that calls `fit(X, y)` with a Series and no exposure. *Since A1: built from an `EXAMPLES` factory per class* |
| `docs/FEATURE_TRANSFORMATIONS.md` §8.1–§8.3 | The transformer declarations `tree`, `total_base`, `composition_base` |
| `docs/MODEL_REIMPLEMENTATION_PLAN.md` | Decisions M1–M12 for Model A; §5 roadmap (step 3 = this rebuild) |

**Old Model 2** (read for the math only; import nothing from it):
`models/independent_total_probability.py`, `count_regression.py`,
`grouped_multinomial.py`, `probability_calibration.py`, `fold_scoring.py`;
reference doc `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md`.

**Forbidden imports in `modeling/`:** `models`, `modeling_config`,
`fitted_features`, `distributions`, `results`, `tuning`, `metrics` (old),
`predictive`, `resampling`, `state_bundle`, `data_splitting`, `evaluation`.

## 5. Decisions

| # | Decision | Why |
|---|---|---|
| N1 | One class hierarchy. `BaseAgeGroupModel` is widened to `y: pd.Series \| pd.DataFrame` and `predict -> np.ndarray \| pd.DataFrame`; every new class subclasses it | The cohort-probability model needs a DataFrame `y` anyway. One contract keeps `evaluate`, `clone`/`set_params` and the contract test for all models. A second base or generics adds code that mypy cannot check (pandas is untyped here) |
| N2 | Multi-cohort classes take the **raw table**, hold the feature transformers, and return a **DataFrame**: one column per cohort, named from `y`'s columns at fit, indexed like `X` | User's choice. Named columns cannot be mixed up by position. M9 still holds for single models: they take a finished design matrix |
| N3 | **The exposure is declared in preprocessing**: a `FeatureTransformer` with `exposure_column="n_apartments"` means "this model gets the exposure". The multi-cohort class reads the raw `X[exposure_column]` and passes it to that model; with no `exposure_column` it passes none. Passing `exposure=` to a multi-cohort class raises | User's idea. It follows statistical practice, where the offset is part of the model's specification (R's `offset(log(n))` in the formula, statsmodels' `exposure=`). Cohorts can differ, and a transformer and model that disagree raise in both directions, because each model already raises on an unexpected or missing exposure |
| N4 | No scoring override. Per-cohort scores are a caller loop: `model.evaluate(Y[c], predictions[c], metric)` | `mean_poisson_deviance` rejects several columns, and how to average cohorts is the caller's choice |
| N5 | Names: `IndependentCohortModels` (Model 1), `TotalChildrenModel`, `CohortProbabilityModel`, `IndependentTotalProbabilityModel` (Model 2), `TemperatureCalibrator` | User: the first model is named for total children, the second for cohort probabilities. The combined class keeps the old name, as `DirectCohortModel` did; new classes are imported from `age_group_prediction.modeling` only |
| N6 | Fitting uses `scipy.optimize.minimize(method="L-BFGS-B")` on objectives built from **predefined library functions**: `scipy.stats.poisson.logpmf`, `scipy.stats.nbinom.logpmf`, `scipy.special.log_softmax` and `xlogy`. Gradients are analytic | User: scipy is fine if the objective is predefined or easy to validate. Every objective is pinned to a library fit in tests (N7), and every gradient to `scipy.optimize.check_grad` |
| N7 | Test oracles: sklearn `PoissonRegressor` and `LogisticRegression` in `tests/unit`; statsmodels in `tests/validation` with `pytest.importorskip` | statsmodels is only in the `validation` dependency group |
| N8 | One penalty meaning in both models: `l2_penalty` multiplies `½‖coefficients‖²` added to the **mean** negative log-likelihood (per building for totals, per child for probabilities). Intercepts are not penalized | Comparable across folds of different size. Conversions for the oracles are in §7 |
| N9 | No clipping of the linear predictor and no floor on the mean. The optimizer runs under `np.errstate(over="raise", invalid="raise")`; a failure or non-finite result raises `RuntimeError` naming feature scale as the likely cause | Measured: clipping hid a failed fit (it returned `success=True` at a wrong point). `exp(·) > 0` already |
| N10 | `TotalChildrenModel`: `family: Literal["poisson", "nb2"] = "poisson"`, `use_exposure: bool = True`. **Poisson is built first (B2); NB2 is its own step (B8)** | User's choice. The offset is Model 2's specification, so a forgotten exposure raises. NB2 showed no gain in the means (§6), so it must be droppable |
| N11 | `CohortProbabilityModel` works for any number of cohorts ≥ 2, taken from `y`'s columns. Symmetric parameterization (one coefficient row per cohort), as sklearn uses | The old code hard-coded 3. With `l2_penalty=0` coefficients are not unique but probabilities are; tests compare probabilities |
| N12 | Calibration is **temperature scaling**, fitted by a separate `TemperatureCalibrator` on **out-of-fold** logits. `CohortProbabilityModel` has an ordinary setting `temperature: float = 1.0`, applied in `predict` as `softmax(logits / temperature)`. No folds inside any model | User's choice. Best practice: it is sklearn's own multiclass method (`CalibratedClassifierCV(method="temperature")`, since 1.8), has one parameter, and measured best here (§6). Isotonic is not advised below ~1000 calibration rows |
| N13 | **No likelihood-ratio gate** on the temperature: the fitted value is always used | Best practice and simpler: sklearn's implementation has none. On well-calibrated data the fitted temperature lands near 1 and changes little. The old gate guarded a threshold rule that no longer exists |
| N14 | Out-of-fold logits come from a short documented loop (in the doc and one integration test), not a helper | One caller today. Promote it to a helper when a second caller exists |
| N15 | Dropped from Model 2, as M12 did for Model A: tuning inside `fit`, bootstrap draws and intervals, pointwise log-probabilities, `PredictionResult`, state bundles, metadata, seed records | Means only. Tuning lives in `hyperparameter_tuning` |
| N16 | Shared helpers live in a public `modeling/utils.py` | The top-level `utils.py` is imported by the tuner and should not pull in scipy optimizers |

**Two decisions that change settled behavior. Ask the user explicitly at the step named:**
- **Exposure length (B1).** A shared `check_exposure` adds a length check. `DirectCohortModel.predict` would then reject a length-1 exposure, which `MODEL_REIMPLEMENTATION_PLAN.md` Step 2.2 accepted as "all buildings have this n". In the GLMs the check is needed: numpy broadcasts a length-1 exposure silently at fit.
- **Index alignment (A3).** Multi-cohort classes check `X.index.equals(y.index)`; otherwise a misaligned `y` is used by position, silently.

## 6. Measured evidence (2026-09-29; sklearn 1.9.0, scipy 1.18.0, statsmodels 0.14.6)

Re-run any figure you rely on. Simulated tables: `StudentPopulationSimulator(load_simulation_config("configs/simulation.toml")).run(rng=np.random.default_rng(seed))`,
then `ShareTransformer(("3_rooms","4_rooms","5_rooms","6_rooms"), reference_column="3_rooms")`;
245 buildings, 60 neighborhoods; grouped 80/20 split; seeds 0–9.

| Question | Result |
|---|---|
| Poisson, scipy + analytic gradient vs sklearn `PoissonRegressor` | max coefficient difference 2.7e-07 (12 function evaluations). Numerical gradients: 1.4e-05, 110 evaluations |
| Poisson and NB2 vs statsmodels (`GLM`, `NegativeBinomial(exposure=n)`), unpenalized | ≤ 1e-5 on coefficients and dispersion |
| NB2 vs Poisson, held-out mean Poisson deviance of the total | 4.34 vs 4.40; difference −0.06 ± 0.15; NB2 better in 5 of 10. Fitted dispersion ≈ 0.105 |
| Calibration, held-out child-weighted log loss | uniform 1.0986, marginal 1.0912, raw 1.0866, **temperature 1.0850** (better in 8 of 10, T ≈ 1.1–1.25), Dirichlet 1.0855 (7 of 10) |
| Softmax on grouped counts (scipy) vs `LogisticRegression` on one row per child | 2e-16, for C in {0.01, 1.5, 100, inf}; 2 and 4 cohorts work unchanged |
| A never-observed cohort | Neither scipy nor sklearn raises. We must check |
| `clone` of `{cohort: (FeatureTransformer, model)}` | Unfitted deep copies, key order and settings kept |
| sklearn's 4 contract checks on a class with required nested estimators | Pass, when given an instance |
| LightGBM with different columns per cohort | No warning under `-W error` |
| Tuner with a DataFrame `y` and a composition metric | Already runs; only the annotation at `evaluator.py:83` is narrow |
| mypy and pandas | pandas is untyped here, so `pd.Series` vs `pd.DataFrame` is not checked. Dropping the `exposure` parameter from an override does fail |

## 7. Target code shape

```python
# modeling/base.py  (A1: annotations and docstring only)
def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame,
        exposure: ArrayLike | None = None) -> Self: ...
def predict(self, X: pd.DataFrame,
            exposure: ArrayLike | None = None) -> np.ndarray | pd.DataFrame: ...

# modeling/utils.py  (A2, B1)
def fit_feature_transformer_and_model(feature_transformer, model, X, y)
        -> tuple[FeatureTransformer, BaseAgeGroupModel]:   # clones, then fits both
def predict_from_raw_table(feature_transformer, model, X) -> np.ndarray | pd.DataFrame
def check_exposure(exposure, *, expected: bool, n_rows: int) -> np.ndarray | None
def minimize_lbfgs(objective_and_gradient, initial, *, max_iter, tol, bounds=None) -> np.ndarray

# modeling/independent_cohorts.py  (A3)
CohortModels = Mapping[str, tuple[FeatureTransformer, BaseAgeGroupModel]]
class IndependentCohortModels(BaseAgeGroupModel):
    def __init__(self, cohort_models: CohortModels) -> None: ...
    # fit(X_raw, y: DataFrame) ; predict(X_raw) -> DataFrame ; fitted: cohort_models_

# modeling/total_children.py  (B2, B8)
class TotalChildrenModel(BaseAgeGroupModel):
    def __init__(self, *, family="poisson", use_exposure=True, l2_penalty=0.0,
                 max_iter=500, tol=...) -> None: ...
    # fitted: intercept_, coef_, feature_names_in_, uses_exposure_ (, dispersion_ for nb2)

# modeling/cohort_probability.py  (B4)
class CohortProbabilityModel(BaseAgeGroupModel):
    def __init__(self, *, l2_penalty=0.0, temperature=1.0, max_iter=500, tol=...) -> None: ...
    # fit(X, y: DataFrame of cohort counts) ; predict_logits(X) -> DataFrame
    # predict(X) -> DataFrame of probabilities, rows sum to 1
    # fitted: intercept_, coef_, cohorts_, feature_names_in_

# modeling/calibration.py  (B6)
class TemperatureCalibrator(BaseEstimator):
    def fit(self, logits, counts) -> Self: ...      # fitted: temperature_

# modeling/independent_total_probability.py  (B5)
class IndependentTotalProbabilityModel(BaseAgeGroupModel):
    def __init__(self, *, total_children_feature_transformer, total_children_model,
                 cohort_probability_feature_transformer, cohort_probability_model) -> None: ...
    # fit(X_raw, y: DataFrame of cohort counts): total target = y.sum(axis=1)
    # predict(X_raw) -> DataFrame = total_mean[:, None] * probabilities
```

**The math.** `D = [1, X]`, `N` buildings, `M = Σ n_bk` children, `λ = l2_penalty`.

| Model | Objective (minimized) | Gradient | Oracle conversion |
|---|---|---|---|
| Total, Poisson | `−mean(poisson.logpmf(y, μ)) + ½λ‖β[1:]‖²`, `μ = exposure·exp(Dβ)` | `Dᵀ(μ − y)/N`, plus `λβ` on non-intercepts | `PoissonRegressor(alpha=λ / mean(exposure)).fit(X, y/exposure, sample_weight=exposure)`; mean = `exposure · predict(X)` |
| Total, NB2 | `−mean(nbinom.logpmf(y, 1/α, 1/(1+αμ))) + ½λ‖β[1:]‖²`, over `(β, log α)` | `−Dᵀ((y−μ)/(1+αμ))/N`; for `log α` see the probe in §6 | statsmodels `NegativeBinomial(y, D, exposure=exposure)`, λ = 0 |
| Cohort probability | `−Σ xlogy(n_bk, p_bk)/M + ½λ‖W‖²`, `p = softmax(a + XW)` | `Xᵀ(n_b·p_bk − n_bk)/M`, plus `λW`; intercepts unpenalized | `LogisticRegression(C=1/(λM))` on one row per (building, cohort) with `sample_weight = n_bk > 0` |
| Temperature | `COHORT_LOG_LOSS(counts, softmax(logits/T))` over `log(1/T)` in (−10, 10), `minimize_scalar(method="bounded")` | — | `sklearn.calibration` `_TemperatureScaling` uses the same form |

Start values: total intercept `log(Σy / Σexposure)`, everything else 0.

**Usage, end to end:**

```python
# Model 1
model_1 = IndependentCohortModels({
    "n_kindergarten": (tree_with_exposure, DirectCohortModel(use_exposure=True)),
    "n_elementary":   (tree,               DirectCohortModel()),
    "n_highschool":   (tree_with_exposure, DirectCohortModel(use_exposure=True)),
}).fit(train_df, Y_train)
predictions = model_1.predict(test_df)            # DataFrame, 3 columns

# Model 2, with calibration
probability_model = CohortProbabilityModel(l2_penalty=1e-3)
logits_val, counts_val = [], []
for train_index, val_index in cv.split(train_df, Y_train, groups_train):
    feature_transformer, model = fit_feature_transformer_and_model(
        cohort_probability_base, probability_model,
        take_rows(train_df, train_index), take_rows(Y_train, train_index))
    logits_val.append(model.predict_logits(feature_transformer.transform(take_rows(train_df, val_index))))
    counts_val.append(take_rows(Y_train, val_index))
temperature = TemperatureCalibrator().fit(pd.concat(logits_val), pd.concat(counts_val)).temperature_

model_2 = IndependentTotalProbabilityModel(
    total_children_feature_transformer=total_base,          # exposure_column="n_apartments"
    total_children_model=TotalChildrenModel(l2_penalty=0.1),
    cohort_probability_feature_transformer=cohort_probability_base,   # no exposure
    cohort_probability_model=CohortProbabilityModel(l2_penalty=1e-3, temperature=temperature),
).fit(train_df, Y_train)
predictions = model_2.predict(test_df)
```

**Hazard to document:** multi-cohort classes clone in `fit`, so
`set_params(cohort_probability_model__temperature=T)` after `fit` does not reach the
fitted copy. Set it before `fit`.

## 8. Steps, PR A: Model 1 completion

### A0. Plan doc and draft PR
1. ✓ `feat/independent-cohort-models` created from `feat/hyperparameter-tuning` (`65d6dbe`).
2. ✓ Link this file from `docs/README.md` and `docs/MODEL_REIMPLEMENTATION_PLAN.md` §5.
3. After the user approves the doc, the user commits the docs and pushes. Then
   a **draft** PR is opened into `feat/hyperparameter-tuning`.

Done when:
- [x] The user has approved the doc, names included (N5): approved 2026-09-29.
  The user also decided that the §8.3 transformer is renamed at B9.
- [x] The draft PR is open: [#10](https://github.com/galkampel/age-group-prediction/pull/10).

### A1. Base contract and contract test
- **Files:** `modeling/base.py`, `tests/unit/test_modeling_contract.py`.
- **Build:** widen the two annotations and the class docstring ("one count
  target, or one per cohort"). In the contract test replace `model_class()` with
  an `EXAMPLES` mapping: class → factory returning `(model, X, y)`.
- **Tests:** (1) a discovered model without an example **fails** rather than
  being skipped; (2) the four checks and refit-equals-fresh run on every example.

Done when:
- [x] `DirectCohortModel` passes with no change to its code.
- [x] Mutation: deleting its example fails the suite.
- [x] mypy, ruff and the non-slow suite pass (1082).

**Record (2026-09-29).**
- **Verified first:**
  - mypy: pandas has no stubs here, so the widened base type-checks, and so do
    subclasses that keep `y: pd.Series`.
  - `check_parameters_default_constructible` rebuilds the class from its
    required arguments only. So an example may set non-default settings or
    need a nested estimator.
- **Mutation checks**, each failing the named test:
  - example deleted: the coverage test, plus a `KeyError` in all 5 parametrized cases;
  - an attribute set in `__init__`: `check_no_attributes_set_in_init`;
  - a refit that differs from a fresh fit: `test_a_refit_equals_a_fresh_fit`;
  - an example that builds another class, and an example with no discovered
    model: the coverage test.
- **Review:**
  - *Fixed.* An example whose factory built another class (a copied factory)
    let that class skip every check. The coverage test now asserts each
    example's class.
  - *Fixed.* An extra example key failed with the message "add an example for []".
    The message now lists both sides.
  - *Not changed.* The refit test compares values, not DataFrame labels. It
    targets leftover fitted state, and A3's test 5 covers columns and index.

### A2. Raw-table helpers
- **Files:** new `modeling/utils.py`, new `tests/unit/test_modeling_utils.py`.
- **Build:** `fit_feature_transformer_and_model` and `predict_from_raw_table`
  (N3): clone both, fit the transformer on the raw table, transform, read
  `X[exposure_column]` when declared, fit the model.
- **Tests, each naming its mistake:**
  1. the caller's templates stay unfitted (fitting in place);
  2. doubling the `n_apartments` column doubles the prediction (exposure dropped at predict);
  3. validation rows are transformed with the training rows' statistics (refit at predict);
  4. a transformer that declares an exposure with a model that uses none raises, and the reverse;
  5. no declared exposure means none is passed.

Done when:
- [ ] One mutation check per test fails as expected.

### A3. `IndependentCohortModels`
- **Files:** new `modeling/independent_cohorts.py`, `modeling/__init__.py`, new
  `tests/unit/test_modeling_independent_cohorts.py`, an example in the contract test.
- **Build:** `fit` checks that the mapping's keys equal `y`'s columns, that
  `exposure` is `None`, and (ask first, §5) index alignment. `cohort_models_` is
  assigned once every cohort has succeeded. `predict` returns the DataFrame of N2.
- **Tests:**
  1. each column equals that cohort's model fitted alone (a cohort fitted on the wrong target);
  2. keys that differ from `y`'s columns raise;
  3. cohorts with different columns and different exposure use fit under `-W error`;
  4. a failing cohort leaves the previous fitted state intact;
  5. columns and index are right on a non-default index;
  6. `predict` works on a table without the target columns (targets leaking into features).

Done when:
- [ ] The contract test discovers the class.
- [ ] Mutation checks, review, non-slow suite.

### A4. Smoke run and docs
- **Smoke run** (scratchpad script, outside the repo): 10 simulated
  populations; per-cohort held-out Poisson deviance of `IndependentCohortModels`,
  and the deviance of the summed prediction against `n_children_total`. Put the
  table in the plan doc.
- **Docs:** `DIRECT_COHORT_MODEL.md` new §0.6; `FEATURE_TRANSFORMATIONS.md` §8.1
  (declare `exposure_column` when the model uses the exposure);
  `MODULE_REFERENCE.md`; `MODEL_REIMPLEMENTATION_PLAN.md` §5.

Done when:
- [ ] Every edited code block has been run.
- [ ] The user has seen the numbers and reviewed the docs; PR A is ready to merge.

## 9. Steps, PR B: Model 2

### B0. Branch and draft PR
Branch `feat/independent-total-probability-model` from `feat/hyperparameter-tuning`
after PR A is merged. Re-verify Part B against the merged code, update the doc,
open the draft PR.

### B1. Shared numerics
- **Files:** `modeling/utils.py`, `modeling/direct_cohort.py`, tests.
- **Build:** `check_exposure` (ask first about the length check, §5) with
  `DirectCohortModel` switched to it; `minimize_lbfgs` (N9).
- **Tests:** Model A's exposure tests pass unchanged; an iteration limit
  raises; an overflow raises instead of returning the start point.

Done when:
- [ ] The default `tol` reproduces the sklearn oracle to 1e-6 (measured: scipy
  defaults gave 9e-6; `ftol=1e-12, gtol=1e-8` gave 1e-7). Record how `tol` maps
  to scipy's options.

### B2. `TotalChildrenModel`, Poisson
- **Files:** new `modeling/total_children.py`, `__init__.py`, new
  `tests/unit/test_modeling_total_children.py`, contract example; a statsmodels
  check in `tests/validation/`.
- **Build:** N6, N8–N10. Input checks, each silent otherwise:
  `sklearn.utils.validation.validate_data` for `X` (reordered columns at
  predict), finite non-negative counts, at least one child (an all-zero `y`
  returns `success=True` with intercept `−inf`).
- **Tests:**
  1. matches `PoissonRegressor` at penalties 0, 0.01 and 1 (wrong gradient or penalty scale);
  2. `check_grad` on the objective;
  3. a huge penalty gives `exposure · Σy/Σexposure` (penalized intercept);
  4. doubling the exposure doubles the mean;
  5. `predict` follows the fitted state after `set_params(use_exposure=False)`;
  6. all-zero `y`, negative counts, reordered columns, unknown family raise;
  7. beats the constant-rate baseline on informative data.

### B3. Cohort log loss and tuner typing
- **Files:** `scoring.py`, `hyperparameter_tuning/evaluator.py` (line 83: `y: Target`), their tests.
- **Build:** `COHORT_LOG_LOSS = Metric("cohort_log_loss", ...)`:
  `−Σ xlogy(n_bk, p_bk) / Σ n_bk` (a plain `0·log 0` gives nan).
- **Tests:** equals a hand computation; a zero-total building adds nothing; the
  evaluator scores a DataFrame `y`.
- **Note in the doc:** `WeightedMean` weights folds by rows; this metric is per child.

### B4. `CohortProbabilityModel`
- **Files:** new `modeling/cohort_probability.py`, `__init__.py`, tests, contract example.
- **Build:** N6, N8, N11, N12. Raises when a cohort is never observed, when
  there are no children, when `temperature ≤ 0`, when an exposure is passed.
- **Tests:**
  1. equals `LogisticRegression` on one row per child with `C = 1/(λM)`, across the penalty range, with a zero-total row;
  2. `check_grad`;
  3. adding a zero-total building leaves the fit unchanged;
  4. doubling every count leaves the fit unchanged (summed instead of mean loss);
  5. 2 and 4 cohorts work (hard-coded 3);
  6. an unobserved cohort raises;
  7. rows sum to 1; `temperature=1` is the plain softmax; a temperature set after `fit` is applied;
  8. non-convergence raises;
  9. beats the marginal proportions on informative data.

### B5. `IndependentTotalProbabilityModel`
- **Files:** new `modeling/independent_total_probability.py`, `__init__.py`, tests, contract example.
- **Tests:**
  1. the output equals total × probabilities of the two models fitted separately, and rows sum to the total mean;
  2. the total target is `y.sum(axis=1)`; tables without target columns work;
  3. each model uses its own transformer (swapped transformers);
  4. doubling `n_apartments` doubles every cohort;
  5. nested `set_params` reaches the next fit; a temperature set before `fit` is applied;
  6. a cohort-probability transformer that declares an exposure raises.

### B6. `TemperatureCalibrator`
- **Files:** new `modeling/calibration.py`, tests.
- **Build:** N12–N14; no settings.
- **Tests:** logits scaled by a known factor recover it; calibrated data gives a
  temperature near 1; uninformative logits flatten instead of failing; the
  out-of-fold loop of §7 as an integration test.

### B7. Smoke run
Ten populations. Per cohort: Model 2 (raw and calibrated) against Model 1; plus
the total's deviance and the cohort log loss. Table in the plan doc.

Done when:
- [ ] The user has seen the numbers.

### B8. NB2 family (droppable)
- **Build:** `family="nb2"`, `dispersion_`; `log α` bounded (old bounds
  `(1e-4, 5)`; justify or change by probe).
- **Tests:** `check_grad`; statsmodels `NegativeBinomial` agreement in
  `tests/validation`; `dispersion_` exists only for NB2; near-Poisson data does
  not fail at the bound.
- **Before starting**, show the user §6's NB2 row and the B7 table, and confirm
  the step is still wanted: means-only output has no consumer for `dispersion_`.

### B9. Docs and close
- `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` new §0 (the rebuilt model: equations,
  settings, calibration flow, the clone hazard);
  `FEATURE_TRANSFORMATIONS.md` §8.2 (its `log_exposure` snippet is outdated: the
  model takes the raw exposure), §8.3 (rename `composition_base` to
  `cohort_probability_base`, as §7 uses, to match `CohortProbabilityModel`;
  user's decision, 2026-09-29), §8.7 item 7 (penalty ranges restated
  under N8: the old `C` range [0.01, 100] is about `l2_penalty` [2e-6, 2e-2] at
  ~4,500 training children); `MODULE_REFERENCE.md`; `docs/README.md`;
  `MODEL_REIMPLEMENTATION_PLAN.md` §5 (step 3 done).
- Decide whether `FeatureTransformer.log_exposure` still has a caller (Model C).

Done when:
- [ ] The user has reviewed the docs; PR B is ready to merge.

## 10. Old tests: carried over or dropped

Source: `tests/unit/test_independent_total_probability.py`.

| Carried over, in new form | Step |
|---|---|
| Exposure has a fixed unit coefficient (doubling doubles) | B2, B5 |
| Grouped fit equals literal per-child expansion | B4 |
| Zero totals add nothing | B3, B4 |
| An unobserved cohort raises | B4 |
| Probabilities sum to 1 and cohorts sum to the total | B4, B5 |
| Targets never enter a feature matrix | A3, B5 |
| Both models beat their constant baselines | B2, B4 |
| Optimizers report non-convergence | B1, B4 |
| The family is explicit | B2, B8 |
| Temperature 1 is the plain softmax | B4 |

**Dropped with their features (N13, N15):** tuning inside `fit`, pointwise
log-probabilities, bootstrap draws and failures, the likelihood-ratio gate,
calibration metadata, seed provenance, state bundles.

## 11. Risks and pitfalls

- **Old-stack imports.** The package root imports the old stack, so
  independence holds per module. Add an AST test over `modeling/*.py` against
  the forbidden list in §4 (pattern: `tests/unit/test_tracking.py`, near line 939). Step A2.
- **Same names, two stacks.** `from age_group_prediction import DirectCohortModel`
  gives the **old** class. Never export new classes from the root.
  `tests/unit/test_model_contracts.py` is the old contract test.
- **Types.** mypy cannot tell a Series from a DataFrame here; the tests must.
- **Zero-total buildings.** The simulated table may have none: use synthetic rows.
- **Tuning.** A convergence error inside a fold ends an Optuna study. Note it
  in `HYPERPARAMETER_TUNING_PLAN.md` §6 for Phase 3.
- **pydantic/ruff.** The repo has no `[tool.ruff]`; do not add one.
- **Stale editor buffers.** If a file "looks unchanged" to the user: "File: Revert File".

## 12. Verification

| Purpose | Command |
|---|---|
| New tests | `uv run pytest tests/unit/test_modeling_*.py tests/unit/test_scoring.py -q -W error` |
| Suite | `uv run pytest -m "not slow"` (record the count at each stop) |
| Oracles needing statsmodels | `uv run --group validation pytest tests/validation -k "total_children or nb2" -q` |
| Types | `uv run mypy` and `uv run mypy src/age_group_prediction/modeling <changed test files>` |
| Lint | `uv run ruff check <files>`; `uv run ruff format <files>` |
| End to end | The smoke runs of A4 and B7, and the §7 usage block run as written |

Each PR is done when its steps are checked, the suite passes, the independent
review's findings are fixed, and the user has approved the final stop.
