# Plan: complete Model 1 (independent cohorts) and rebuild Model 2 (total × probability)

**Written for:** the implementing model (Claude Opus 5.5) and the user who validates each step.
This file is self-contained: it assumes no memory of the planning conversation.
This file is the source of truth: update its status line and checkboxes as steps finish.

**Status (2026-09-29):** A0 done (draft PR #10). A1 committed (`8fed394`).
A2 committed (`2c38010`, `5b5ea4d`, `7204ec2`). A3 committed (`6b234f0`,
`3f46fbc`). The handoff edit to this doc follows them. **Next: A4** (§8),
starting with its "Facts" and "Decisions to ask".
Non-slow suite: **1104 passed** (1 skipped, 1 xfailed).

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
| A | `feat/independent-cohort-models` | Model 1 completion (steps A0–A4) | Small. It settles what B builds on: the widened base contract, the contract test, `ModelPipeline`, the DataFrame output |
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
   Data values are validated where the data is prepared (`preprocessing.py`),
   not inside each model (N3).
7. **Code style** (match `modeling/direct_cohort.py`): one-line module
   docstring, `from __future__ import annotations`, explicit `__all__`;
   keyword-only constructors that store arguments verbatim; validation in
   `fit`; fitted state in trailing-underscore attributes assigned together only
   after success; `predict` follows the fitted state, not current settings;
   short reST docstrings that say *why*; sparse comments for non-obvious
   reasons; precise types with named aliases; lower-case error messages that
   say what to do.
8. **Names:** `feature_transformer`, `exposure_*` (never `n`/`N` as a public
   name), `train`/`val`. In `modeling`, shared logic goes **under a class**;
   there is no utils file (N16).
9. **Tests:** each test's name and comment state the mistake it catches. No
   test that only checks a library.
10. **Docs** are updated in the same PR. Run every code block you put in a doc.
11. **Probes** (read-only, never write files in the repo):
    `PYTHONPATH=src uv run --group test python -c "..."`. zsh does not split
    `$VAR`, and an unquoted `--include=*.py` fails: quote globs.

## 4. What exists today

| File | What it gives |
|---|---|
| `src/age_group_prediction/modeling/base.py` | `BaseAgeGroupModel(BaseEstimator, ABC)`: abstract `fit(X, y: pd.Series, exposure=None) -> Self`, abstract `predict(X, exposure=None) -> np.ndarray`, concrete `evaluate(y_true, y_pred, metric) -> float`. *Since A1: `y: pd.Series \| pd.DataFrame`, `predict -> np.ndarray \| pd.DataFrame`. A2 added the property `uses_exposure`; A3 removed it (N3)* |
| `src/age_group_prediction/modeling/direct_cohort.py` | `DirectCohortModel`: LightGBM for **one** cohort, on a finished design matrix; `use_exposure`; `_check_exposure`; fitted `regressor_` and `base_log_rate_` |
| `src/age_group_prediction/preprocessing.py` | `ShareTransformer`: fit-free, row-wise, run on the full table before splitting. *A2 adds `ExposureTransformer`* |
| `src/age_group_prediction/scoring.py` | `Metric(name, function, greater_is_better=False)`, `POISSON_DEVIANCE`, `RMSE`, `MAE` |
| `src/age_group_prediction/feature_engineering/transformer.py` | `FeatureTransformer(plans, *, interactions, remainder)`; `fit`, `transform -> DataFrame`. *`exposure_column` and `log_exposure` were removed in A2 (N3)* |
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
| N2 | Multi-cohort classes take the **raw table** and return a **DataFrame**: one column per cohort, named from `y`'s columns at fit, indexed like `X`. Each of their models is a `ModelPipeline`, which holds the feature transformer (N17) | User's choice. Named columns cannot be mixed up by position. M9 still holds for single models: they take a finished design matrix |
| N3 | *Revised four times on 2026-09-29, each time by the user.* **The exposure is the model's, and its values are validated in preprocessing.** (a) Whether a model has an offset is its own constructor setting (`use_exposure`), on the models that can have one; the base declares no such setting or property. (b) Which column holds it is a schema fact: `exposure_column="n_apartments"`. (c) `preprocessing.ExposureTransformer` reads that column and returns it as floats, or raises if a value is not strictly positive and finite. (d) The caller builds the exposure with it, on the full table before splitting, and passes it as `exposure=` to every model, `ModelPipeline` (N17) included. (e) Every model takes the same exposure: one with an offset raises if it is missing; any other model **ignores** it (A3). (f) The models take the **raw** exposure, not its log. (g) `FeatureTransformer.exposure_column` and `log_exposure` were removed | (a, b) statsmodels, R, glum, LightGBM and sklearn's examples all pass the exposure to the model, outside the feature matrix. (c) LightGBM accepts a zero, infinite or NaN exposure silently (§6), so the check is needed once. (e) A missing exposure would drop the offset silently; an ignored one lets a caller pass one exposure to every model, and a tuner compare `use_exposure` with one fixed exposure (the evaluator passes the same exposure to every trial). As sklearn's metadata routing does for metadata a consumer declares not requested. Before A3 an unexpected exposure raised, and `uses_exposure` told an aggregator where to route it. (f) `DirectCohortModel` needs `Σ exposure` for its intercept, and `exposure=` is settled (M10), as in statsmodels. History: first a transformer's `exposure_column` declared the exposure; then helper functions read it; then the user asked for the validation in preprocessing and the logic under a class; then for an explicit exposure argument on `ModelPipeline` (d), which gives up the guarantee that the pipeline always validates it |
| N4 | No scoring override. Per-cohort scores are a caller loop: `model.evaluate(Y[c], predictions[c], metric)` | `mean_poisson_deviance` rejects several columns, and how to average cohorts is the caller's choice |
| N5 | Names: `IndependentCohortModels` (Model 1), `TotalChildrenModel`, `CohortProbabilityModel`, `IndependentTotalProbabilityModel` (Model 2), `TemperatureCalibrator` | User: the first model is named for total children, the second for cohort probabilities. The combined class keeps the old name, as `DirectCohortModel` did; new classes are imported from `age_group_prediction.modeling` only |
| N6 | Fitting uses `scipy.optimize.minimize(method="L-BFGS-B")` on objectives built from **predefined library functions**: `scipy.stats.poisson.logpmf`, `scipy.stats.nbinom.logpmf`, `scipy.special.log_softmax` and `xlogy`. Gradients are analytic | User: scipy is fine if the objective is predefined or easy to validate. Every objective is pinned to a library fit in tests (N7), and every gradient to `scipy.optimize.check_grad` |
| N7 | Test oracles: sklearn `PoissonRegressor` and `LogisticRegression` in `tests/unit`; statsmodels in `tests/validation` with `pytest.importorskip` | statsmodels is only in the `validation` dependency group |
| N8 | One penalty meaning in both models: `l2_penalty` multiplies `½‖coefficients‖²` added to the **mean** negative log-likelihood (per building for totals, per child for probabilities). Intercepts are not penalized | Comparable across folds of different size. Conversions for the oracles are in §7 |
| N9 | No clipping of the linear predictor and no floor on the mean. The optimizer runs under `np.errstate(over="raise", invalid="raise")`; a failure or non-finite result raises `RuntimeError` naming feature scale as the likely cause | Measured: clipping hid a failed fit (it returned `success=True` at a wrong point). `exp(·) > 0` already |
| N10 | `TotalChildrenModel`: `family: Literal["poisson", "nb2"] = "poisson"`, `use_exposure: bool = True` (N3). **Poisson is built first (B2); NB2 is its own step (B8)** | User's choice. The offset is Model 2's specification, so a forgotten exposure raises. NB2 showed no gain in the means (§6), so it must be droppable |
| N11 | `CohortProbabilityModel` works for any number of cohorts ≥ 2, taken from `y`'s columns. Symmetric parameterization (one coefficient row per cohort), as sklearn uses | The old code hard-coded 3. With `l2_penalty=0` coefficients are not unique but probabilities are; tests compare probabilities |
| N12 | Calibration is **temperature scaling**, fitted by a separate `TemperatureCalibrator` on **out-of-fold** logits. `CohortProbabilityModel` has an ordinary setting `temperature: float = 1.0`, applied in `predict` as `softmax(logits / temperature)`. No folds inside any model | User's choice. Best practice: it is sklearn's own multiclass method (`CalibratedClassifierCV(method="temperature")`, since 1.8), has one parameter, and measured best here (§6). Isotonic is not advised below ~1000 calibration rows |
| N13 | **No likelihood-ratio gate** on the temperature: the fitted value is always used | Best practice and simpler: sklearn's implementation has none. On well-calibrated data the fitted temperature lands near 1 and changes little. The old gate guarded a threshold rule that no longer exists |
| N14 | Out-of-fold logits come from a short documented loop (in the doc and one integration test), not a helper | One caller today. Promote it to a helper when a second caller exists |
| N15 | Dropped from Model 2, as M12 did for Model A: tuning inside `fit`, bootstrap draws and intervals, pointwise log-probabilities, `PredictionResult`, state bundles, metadata, seed records | Means only. Tuning lives in `hyperparameter_tuning` |
| N16 | *Withdrawn by the user, 2026-09-29.* **No utils file in `modeling`; logic lives under a class.** The home of `minimize_lbfgs` is decided at B1 | User's rule. First version: shared helpers in a public `modeling/utils.py` |
| N17 | *Revised by the user on 2026-09-29: the exposure is an explicit argument.* `ModelPipeline(feature_transformer, model)`, a `BaseAgeGroupModel`: a feature transformer, then a model, fitted and used on the raw table. `fit(X, y, exposure=None)` clones both; `predict(X, exposure=None)` uses the fitted copies `feature_transformer_` and `model_`. `exposure` is passed through to the model, whose own check decides whether one is needed. Rows of `X`, `y` and the exposure are paired by position (N19). The aggregators (`IndependentCohortModels`, `IndependentTotalProbabilityModel`) hold models that take the raw table, and only loop and combine | User's choice. scikit-learn's `Pipeline` pattern, by composition: each class does one thing. sklearn's own `Pipeline` was already rejected (`HYPERPARAMETER_TUNING_PLAN.md` D13): its `fit` and `predict` name the exposure differently, and it has no `evaluate`. The explicit exposure keeps the base contract `fit(X, y, exposure)` for every model, and lets the tuner's `exposure=` path take a pipeline. First version: the pipeline read `X[exposure_column]` and rejected an `exposure` argument |
| N18 | `base_log_rate_` stays. It is the intercept `b` in `exposure × exp(b + F(x))`. No intercept option is added for a model without an exposure | User's decision after the evidence in §6. Given an `init_score`, LightGBM switches off its own starting average, so the model supplies `b`. Without an exposure LightGBM starts from `mean(y)` itself |
| N19 | *Reversed by the user at the end of A3, 2026-09-29.* **No index check.** `X`, `y` and the exposure are paired by position, as in scikit-learn; the docstrings say so, and to take the exposure's rows as `exposure.loc[X_train.index]` | The splitter (`Splitter.train_test_split`, `cv`) splits `X`, `y` and `groups` by the same positions, so they cannot be misaligned; sklearn pairs by position too (`PoissonRegressor().fit(X, y_shuffled)` runs silently); the check raised on position-correct data whose labels differ (e.g. `X` after `reset_index`). History: the check was added to the multi-cohort classes, then to `ModelPipeline` (`7204ec2`), then shared in the base, then removed |
| N20 | The tuner path is **documented, not changed**. `CVHyperparameterEvaluator.evaluate` passes its `exposure` straight to `model.fit`, so the documented way to build that argument is `ExposureTransformer(...).fit_transform(table)`. Switching the evaluator to a `ModelPipeline` is recorded for the tuning work | User's decision. It keeps PR A out of the tuning package, and the exposure still comes from the validating class |
| N21 | *User, 2026-09-29 (A3).* `IndependentCohortModels` takes a `Mapping[str, BaseAgeGroupModel]`. Nested `set_params` names do not reach into it: each cohort is tuned on its own, and the tuned models are assembled. Replace the mapping with `set_params(cohort_models=...)` | Measured: `get_params(deep=True)` lists only `cohort_models`, and a nested name raises `AttributeError`. The cohorts are independent, so no study tunes them together |

**One decision still to ask, at the step named:**
- **Exposure length (B1).** Should a model reject an exposure whose length
  differs from `X`'s? `DirectCohortModel.predict` accepts a length-1 exposure,
  which `MODEL_REIMPLEMENTATION_PLAN.md` Step 2.2 accepted as "all buildings
  have this n". In the GLMs numpy broadcasts a length-1 exposure silently at
  fit. Rows are paired by position (N19). Re-examine at B0.

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
| A bad exposure given to LightGBM (2026-09-29) | A negative one raises. **Zero, inf and NaN pass silently**, in the `init_score` form and in the rate form |
| Rate form (`y / exposure`, `sample_weight=exposure`) vs `init_score` + `base_log_rate_` | The same model: relative difference 3e-8 over 4 hyperparameter settings, the same held-out deviance to 6 decimals. Not adopted (N18) |
| LightGBM's start without an exposure | With almost no learning every prediction is `mean(y)` (4.7733). With `boost_from_average=False` it is 1.0 |
| `ModelPipeline` and `ExposureTransformer`, defined inline in a probe | sklearn's 4 checks pass; nested names such as `model__learning_rate` work; doubling the exposure column gives a ratio of exactly 2; a refit equals a fresh fit; `set_params(model__use_exposure=False)` after `fit` leaves predictions unchanged; a model without an exposure works on a table without the column; a zero, negative, infinite or NaN exposure raises at fit and at predict, naming the row; a missing column gives `KeyError`; pickling works |
| mypy and pandas | pandas is untyped here, so `pd.Series` vs `pd.DataFrame` is not checked. Dropping the `exposure` parameter from an override does fail |

## 7. Target code shape

```python
# modeling/base.py  (A1: annotations and docstring only)
def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame,
        exposure: ArrayLike | None = None) -> Self: ...
def predict(self, X: pd.DataFrame,
            exposure: ArrayLike | None = None) -> np.ndarray | pd.DataFrame: ...

# preprocessing.py  (A2)
class ExposureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, exposure_column: str = "n_apartments") -> None: ...
    def fit(self, X, y=None) -> Self: ...          # learns nothing
    def transform(self, X) -> pd.Series: ...       # the raw exposure as floats, or raises

# modeling/pipeline.py  (A2)
class ModelPipeline(BaseAgeGroupModel):
    def __init__(self, feature_transformer: FeatureTransformer, model: BaseAgeGroupModel) -> None: ...
    # fit(X_raw, y, exposure=None) ; predict(X_raw, exposure=None)
    # fitted: feature_transformer_, model_

# modeling/independent_cohorts.py  (A3)
CohortModels = Mapping[str, BaseAgeGroupModel]   # each takes the raw table: a ModelPipeline
class IndependentCohortModels(BaseAgeGroupModel):
    def __init__(self, cohort_models: CohortModels) -> None: ...
    # fit(X_raw, y: DataFrame, exposure=None) ; predict(X_raw, exposure=None) -> DataFrame
    # the same exposure goes to every cohort; a model without an offset ignores it ; fitted: cohort_models_

# modeling/total_children.py  (B2, B8)
class TotalChildrenModel(BaseAgeGroupModel):
    def __init__(self, *, family="poisson", use_exposure=True, l2_penalty=0.0,
                 max_iter=500, tol=...) -> None: ...
    # fitted: intercept_, coef_, feature_names_in_ (, dispersion_ for nb2)

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
    def __init__(self, *, total_children_model: BaseAgeGroupModel,
                 cohort_probability_model: BaseAgeGroupModel) -> None: ...   # two ModelPipelines
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
# The exposure, validated once on the full table, then split with it
exposure = ExposureTransformer("n_apartments").fit_transform(table)

# Model 1
model_1 = IndependentCohortModels({
    "n_kindergarten": ModelPipeline(tree, DirectCohortModel(use_exposure=True)),   # applies the exposure
    "n_elementary":   ModelPipeline(tree, DirectCohortModel()),                    # ignores it
    "n_highschool":   ModelPipeline(tree, DirectCohortModel(use_exposure=True)),
}).fit(train_df, Y_train, exposure=exposure_train)
predictions = model_1.predict(test_df, exposure=exposure_test)   # DataFrame, 3 columns

# Model 2, with calibration
probability_pipeline = ModelPipeline(cohort_probability_base, CohortProbabilityModel(l2_penalty=1e-3))
logits_val, counts_val = [], []
for train_index, val_index in cv.split(train_df, Y_train, groups_train):
    fold = clone(probability_pipeline).fit(
        take_rows(train_df, train_index), take_rows(Y_train, train_index))
    X_val = fold.feature_transformer_.transform(take_rows(train_df, val_index))
    logits_val.append(fold.model_.predict_logits(X_val))
    counts_val.append(take_rows(Y_train, val_index))
temperature = TemperatureCalibrator().fit(pd.concat(logits_val), pd.concat(counts_val)).temperature_

model_2 = IndependentTotalProbabilityModel(
    total_children_model=ModelPipeline(total_base, TotalChildrenModel(l2_penalty=0.1)),   # use_exposure=True
    cohort_probability_model=ModelPipeline(                                               # no exposure
        cohort_probability_base,
        CohortProbabilityModel(l2_penalty=1e-3, temperature=temperature)),
).fit(train_df, Y_train, exposure=exposure_train)
predictions = model_2.predict(test_df, exposure=exposure_test)
```

**Hazard to document:** `ModelPipeline` and the multi-cohort classes clone in
`fit`, so `set_params(cohort_probability_model__model__temperature=T)` after
`fit` does not reach the fitted copy. Set it before `fit`.

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

### A2. The exposure moves to the model; `ExposureTransformer` and `ModelPipeline`
*Revised by the user after the first version (N3, N16, N17, N18, N20). The step
starts from the uncommitted working tree below. Two commits.*

**The working tree today** (first version; suite 1083 passed):

| Change in the working tree | What to do |
|---|---|
| **A2a.** `exposure_column` and `log_exposure` removed from `FeatureTransformer` (`feature_engineering/transformer.py`); their 6 tests removed (`test_feature_transformer.py`, `test_transformer_spec.py`); `FEATURE_TRANSFORMATIONS.md` §8.1, §8.2, §8.3, §8.6 edited | **Keep unchanged.** It is commit 1 |
| `BaseAgeGroupModel.uses_exposure`, default `False` (`modeling/base.py`) | Keep; reword the docstring |
| `DirectCohortModel.uses_exposure` with an "as fitted" branch | Simplify to `return self.use_exposure` |
| `test_modeling_never_imports_the_old_stack` and `OLD_STACK` (`test_modeling_contract.py`) | Keep unchanged |
| `modeling/utils.py` and `tests/unit/test_modeling_utils.py` (untracked) | **Delete.** The tests move to `test_modeling_pipeline.py` |
| `MODULE_REFERENCE.md`: a `modeling/utils.py` row | Replace with `pipeline.py` |

**Build:**
1. **`preprocessing.py`: `ExposureTransformer`** (§7), added to `__all__`.
   - `fit` learns nothing and returns `self`. `transform` needs no prior `fit`.
   - `transform` returns `X[exposure_column]` as floats, with `X`'s index. It
     raises `ValueError` if a value is not strictly positive and finite. The
     message names the column, the number of invalid rows and the first 5
     index labels.
   - A missing column is left to pandas (`KeyError`), and a non-numeric one to
     `astype` (`ValueError`).
   - The docstring says why (LightGBM accepts a zero, infinite or NaN exposure
     silently) and that running it on the full table before splitting makes a
     bad test row fail early.
2. **`modeling/pipeline.py` (new): `ModelPipeline`** (N17, revision 2), exported
   from `modeling/__init__.py` and never from the package root.
   ```python
   def fit(self, X, y, exposure=None) -> Self:
       self._check_exposure_index(X, exposure)
       feature_transformer = clone(self.feature_transformer).fit(X, y)
       model = clone(self.model).fit(feature_transformer.transform(X), y, exposure=exposure)
       self.feature_transformer_ = feature_transformer   # together, after success
       self.model_ = model
       return self

   def predict(self, X, exposure=None) -> np.ndarray | pd.DataFrame:
       check_is_fitted(self)
       self._check_exposure_index(X, exposure)
       return self.model_.predict(self.feature_transformer_.transform(X), exposure=exposure)
   ```
   - Its one check: a `pd.Series` exposure whose index differs from `X`'s
     raises `ValueError`, because the model reads it by position. Whether an
     exposure is given at all is the model's own check.
   - `uses_exposure` returns `self.model.uses_exposure`, so an aggregator can
     route one exposure to the models that use one.
3. **`modeling/direct_cohort.py`:**
   - `_check_exposure` keeps the presence check and the float conversion. The
     "strictly positive and finite" check is removed; the docstring points to
     `preprocessing.ExposureTransformer`.
   - `base_log_rate_`, `init_score` and `predict`'s arithmetic do not change (N18).
4. **The tuner path (N20):** no code change in `hyperparameter_tuning`. Update
   the evaluator's class docstring example to build the exposure with
   `ExposureTransformer`, and add a task to `HYPERPARAMETER_TUNING_PLAN.md` §6:
   switch the evaluator to a `ModelPipeline`.

**Tests, each naming its mistake, with one mutation each:**

| File | Test | Mistake it catches | Mutation |
|---|---|---|---|
| `test_preprocessing.py` | an invalid exposure is rejected, parametrized: zero, negative, inf, NaN | a bad exposure reaches LightGBM, which accepts it | `> 0` → `>= 0`; drop `np.isfinite` |
| | the error names the invalid rows | an error the user cannot act on | drop the index labels |
| | the exposure is returned as given, with the table's index | a transformed or re-indexed exposure | return `np.log(exposure)`; reset the index |
| `test_modeling_pipeline.py` (new) | the templates stay unfitted | fitting the caller's objects in place | drop the model's `clone` |
| | doubling the exposure doubles the prediction | the exposure lost at predict | `predict` passes `np.ones(len(X))` |
| | rows are transformed with the training statistics (row by row equals the batch) | refitting the transformer at predict | refit a clone in `predict` |
| | predict follows the fitted copies after `set_params(model__use_exposure=False)` | reading the template at predict | use `self.model` in `predict` |
| | `uses_exposure` follows the model (`True`, `False`) | the base default withholding the exposure | delete the override |
| | an exposure for other rows (a shuffled Series) raises, at fit and at predict | an exposure applied to the wrong buildings | drop the check |
| `test_modeling_direct_cohort.py` | `test_exposure_misuse_raises` keeps `missing-while-on` and `given-while-off`. Its zero, negative and infinite cases are removed; the transformer's tests cover them | | |
| `test_modeling_contract.py` | an `EXAMPLES` entry for `ModelPipeline`, without an exposure (the refit test calls `fit(X, y)`). Discovery finds the class, so without it `test_every_shipped_model_has_an_example` fails | | delete the entry |

The pipeline tests can reuse the data of `test_modeling_utils.py`: a
`FeatureTransformer` with one `Center()` plan on `x`; a table with `x` and
`n_apartments`; a Poisson `y` proportional to `n_apartments`; and
`DirectCohortModel`. `n_apartments` is not a feature, so doubling is exact
(`rtol=1e-12`).

**Docs:**
- `MODULE_REFERENCE.md`: `ExposureTransformer` in the `preprocessing.py` row;
  `pipeline.py` replaces the `utils.py` row; `ModelPipeline` in the
  `modeling/__init__.py` row.
- `DIRECT_COHORT_MODEL.md` §0.3: the value check moved to `ExposureTransformer`.
- `MODEL_REIMPLEMENTATION_PLAN.md` M10: a one-line note pointing to N3.
- `HYPERPARAMETER_TUNING_PLAN.md` §6 (N20).
- Memory file `api-design-preferences.md`: already updated on 2026-09-29.
- Run every code block that is edited or added.

**Pitfalls met in the first version:**
- In tests, write `ColumnPlan(..., columns=("x",))`. A bare string passes at
  runtime but fails mypy.
- `tests/unit/test_transformer_spec.py` has 4 mypy errors that predate this
  step (near lines 86, 96, 105, 210). They are out of scope.
- The simulator is a top-level package: `from student_simulator.pipeline import
  StudentPopulationSimulator`, `from student_simulator.config import load_simulation_config`.
- To mutate the import guard, put the forbidden import under
  `if TYPE_CHECKING:`, so test collection still runs.
- To run a doc's code blocks: extract the section's `python` blocks and `exec`
  them in order in one namespace. Set `raw_table` from the simulator first,
  and `fit_df`, `train_df`, `valid_df` after the block that builds `table`.
- Filter LightGBM's log lines from a probe's output with `grep -v "^\[LightGBM\]"`.

**Suggested commits:**
1. A2a: `transformer.py`, `test_feature_transformer.py`,
   `test_transformer_spec.py`, `docs/FEATURE_TRANSFORMATIONS.md`.
   `refactor(feature_engineering): the exposure is the model's, not the transformer's`
2. Everything else.
   `feat(modeling): ModelPipeline fits a feature transformer and a model on the raw table; preprocessing validates the exposure`

Done when:
- [x] `modeling/utils.py` and `test_modeling_utils.py` are gone.
- [x] One mutation check per test fails as expected.
- [x] The review's findings are reproduced, then fixed or rejected.
- [x] mypy, ruff and the non-slow suite pass: 1093 passed (1 skipped, 1 xfailed).

**Record of the revised A2 (2026-09-29).**
- **Baseline** before the first edit: 1083 passed. After: 1093 (+6 exposure
  tests, +8 pipeline tests, +5 contract cases for `ModelPipeline`; −6 tests of
  the deleted `utils.py`, −3 value cases of `test_exposure_misuse_raises`).
- **Verified first** (sklearn 1.9.0): the 4 contract checks pass on a class
  whose nested estimators are required arguments; nested `set_params` reaches
  only the copy; LightGBM accepts a zero exposure's `-inf` offset silently.
- **Mutation checks**, each failing its named test: `> 0` → `>= 0` (zero);
  `np.isfinite` dropped (inf; NaN is still caught by `> 0`); index labels
  dropped; the `[:5]` limit dropped; `np.log` returned; index reset; the model's
  `clone` dropped; `np.ones` at predict; the transformer refitted at predict;
  the column always read; `"n_apartments"` hard-coded; the template read at
  predict; the `exposure=` check dropped; the column read unvalidated; the
  `ModelPipeline` example deleted (the coverage test and its 5 cases).
- **Ran:** the evaluator's docstring example, as written, under `-W error` (3
  trials on a simulated table).
- **Review** (independent subagent; 16 mutations of its own, pickling and
  nullable dtypes checked):
  - *Fixed.* Nothing tested the "first 5 labels" limit; the test now has 6
    invalid rows. *Fixed.* The message read "1 rows are not"; now "1 invalid,
    first at rows [...]".
  - *Not changed.* Two columns with the exposure's name raise pandas'
    "truth value is ambiguous": it raises, so nothing passes silently.
  - *For A3.* `ModelPipeline.fit` fits a `y` with another index by position,
    silently. N19 covers only the multi-cohort classes.
  - *Open, for the user.* `astype(float)` does not reject every non-numeric
    column, as assumed in Build item 1: a numpy **bool** column passes as all
    1.0, and digit strings are parsed. A nullable `boolean` column with a
    `False`, and a non-digit string, do raise.
- **Revision 2 (user, 2026-09-29).** `ModelPipeline` takes the exposure as an
  argument (N17, N3 d) and drops `exposure_column`; `DirectCohortModel`'s local
  `n` is `exposure_values`, and the tests' `N` is `EXPOSURE` (§3 rule 8).
  - Pipeline tests: 4 dropped with the behavior they covered (no column
    needed, `exposure_column`, an `exposure` argument raises, an invalid
    exposure raises), 3 added (`uses_exposure` ×2, the index check).
  - Mutations, each failing its test: the model's `clone` dropped; `np.ones`
    at predict; the transformer refitted at predict; the template used at
    predict; `uses_exposure` returning `False`; the index check dropped.
  - Review (subagent), each finding reproduced:
    - *Fixed.* A 2-D exposure (a one-column DataFrame) broadcast into an
      (n, n) prediction, silently; older than A2. `DirectCohortModel` now
      raises unless the exposure is 1-D (new case `two-dimensional`; its
      mutation fails it).
    - *Fixed.* `base.py`'s `uses_exposure` docstring still named the column.
    - *Documented.* `ModelPipeline.uses_exposure` reads the template, so after
      `set_params` it describes the next fit; `predict` then raises, never
      misuses the exposure. A3's aggregator routes by it at fit and predict.
    - *Resolved before A3.* `ModelPipeline.fit` paired a `y` with another index
      by position. It now raises (N19): `test_a_y_for_other_rows_raises_at_fit`,
      whose mutation (skip the check) fails it and only it. Suite: 1094.
  - Suite: 1093 passed (1 skipped, 1 xfailed): 1092 after the redesign, +1 case.
- **Also left:** `HYPERPARAMETER_TUNING_PLAN.md` §5 still builds
  `exposure_train = train_df["n_apartments"]`; its block cannot run until
  `HyperparameterStudy` exists (Phase 3). The new §6 task covers it.

**Record of the first version (2026-09-29).** Facts that still hold:
- `clone` keeps the transformer's settings, `fit` accepts `y`, and `transform`
  keeps the index.
- A refit at predict changes the row-by-row predictions (by up to 9.7).
- All 12 §8 code blocks of `FEATURE_TRANSFORMATIONS.md` ran on a simulated
  table under `-W error`, after the A2a edits.
- The import guard's 3 mutations each failed it: `from ..models import base`,
  `from .. import models`, `from age_group_prediction.metrics import Metric`.
  The guard sees static imports only, as its comment says.
- Review finding, fixed: a section header in `test_transformer_spec.py` had
  been deleted along with the removed test.

### A3. `IndependentCohortModels`
*Revised by the user at the start of the step (N3 e, N21). Committed:
`6b234f0` (the exposure rule; rows paired by position), `3f46fbc`
(`IndependentCohortModels`).*

**Decisions at the start (user, 2026-09-29):**
- **Every model gets the same exposure.** A model applies it only if its
  setting says so (`use_exposure=True`); any other model ignores it. A model
  fitted with an offset still raises if it is missing. So the aggregator passes
  the exposure to every cohort and routes nothing, and an exposure passed when
  no cohort uses one is ignored too.
- **`uses_exposure` is removed** (base, `DirectCohortModel`, `ModelPipeline`):
  nothing reads it any more. No base `use_exposure` field either: sklearn's
  `get_params` reads each model's own `__init__`, and a model without an offset
  (`CohortProbabilityModel`) has no such setting.
- **A mapping of cohort models; each cohort tuned on its own** (N21). The plan
  had assumed nested `set_params` reaches the templates; it does not.
- **No index check** (N19 reversed at the end of the step): rows are paired by
  position, as in scikit-learn; `_check_aligned` removed from `ModelPipeline` too.

**A3a. A model ignores an exposure it does not use.**
- `modeling/direct_cohort.py`: `_check_exposure` returns `None` when no
  exposure is expected; it raises only when one is expected and missing, and
  checks that a used one is 1-D. `predict` still follows the fitted state.
- `modeling/base.py`, `modeling/pipeline.py`: `uses_exposure` removed; the
  `fit` docstring states the rule.
- Tests: `test_exposure_misuse_raises` loses `given-while-off` and matches each
  message; new `test_an_unused_exposure_is_ignored`; the 2 pipeline
  `uses_exposure` tests removed; the evaluator test matches the new message.
- Docs: `DIRECT_COHORT_MODEL.md` §0.2, §0.3; M10 note; `MODULE_REFERENCE.md`.

**A3b. `IndependentCohortModels`** (`modeling/independent_cohorts.py`, exported
from `modeling` only; a contract `EXAMPLES` entry).
- `fit(X, y, exposure=None)`: the mapping's keys equal `y`'s columns,
  duplicates counted (`Counter`); then
  `clone(model).fit(X, y[cohort], exposure=exposure)` per column,
  in `y`'s order; `cohort_models_` set once all succeeded.
- `predict(X, exposure=None)`: a DataFrame with `y`'s columns and `X`'s
  index; each prediction taken as an array.
- `ModelPipeline`'s index check and its 2 tests removed (N19); its docstring
  says rows are paired by position.

**Tests** (`tests/unit/test_modeling_independent_cohorts.py`), each with its mutation:

| Test | Mistake it catches | Mutation |
|---|---|---|
| each column equals that cohort's model fitted alone (under `filterwarnings("error")`) | a cohort fitted on another target | every cohort fitted on the first column; exposure dropped at predict |
| models that do not match `y`'s columns raise: missing, extra, duplicated | a cohort dropped silently; a duplicate reaching its model as a DataFrame | check dropped; a set check |
| a failing cohort leaves the previous fit intact | new and old cohorts paired | assigned cohort by cohort |
| columns follow `y`, rows follow `X` | misnamed cohorts, misaligned rows | mapping order; `index=None` |
| predict needs no target columns | reading a target from `X` | targets dropped from `X` at predict |
| doubling the exposure doubles only the cohorts with an offset | the exposure lost, or applied without an offset | exposure dropped at predict |
| predictions are placed by position, not by their own index | a Series prediction realigned into NaN | `np.asarray` dropped |

Done when:
- [x] The contract test discovers the class (mutation: its example deleted fails 6 cases).
- [x] Mutation checks, review, non-slow suite: 1104 passed (1 skipped, 1 xfailed).

**Record (2026-09-29).**
- **Baseline** before the first edit: 1094 passed (1 skipped, 1 xfailed).
- **Verified first** (sklearn 1.9.0): the 4 contract checks pass on an
  aggregator whose parameter is a dict of `ModelPipeline`s; `clone` deep-copies
  it (keys, order, settings); `get_params(deep=True)` lists only
  `cohort_models`, and a nested name raises `AttributeError`; the tuner passes
  one fixed exposure to every trial (`evaluator.py:104-122`), so before A3a a
  trial with `use_exposure=False` raised.
- **A3a mutations:** the exposure applied while off fails
  `test_an_unused_exposure_is_ignored`; the missing-exposure check dropped fails
  `missing-while-on` and the evaluator's test (without the `match`, the 1-D
  check would have caught `None` with another message).
- **Review** (independent subagent, 9 mutations of its own), each reproduced:
  - *Fixed, then withdrawn.* An exposure Series for other rows was used by
    position by cohort models that are not pipelines. A shared
    `_check_aligned` checked it; the user then dropped every index check (N19).
  - *Fixed.* `sorted` raised `TypeError` for mixed-type cohort names; now `Counter`.
  - *Fixed.* A cohort model returning a Series with its own index gave 100
    NaNs per column. Now taken as an array.
  - *Fixed.* `MODULE_REFERENCE.md` lacked the new module and names.
  - *Not changed.* `MODEL_REIMPLEMENTATION_PLAN.md` Steps 2.2 and 2.3 describe
    the finished Phase 2 as built; M10 carries the dated note.
  - *Not changed.* A nested `set_params` name raises `AttributeError`, not
    sklearn's "Invalid parameter". It raises, and the docstring says to tune
    each cohort on its own (N21).
- **User's questions at the stop**, answered with probes:
  - *Index check:* dropped (N19). −4 tests; suite 1104.
  - *`use_exposure=True` with `"regression"` still raises:* sklearn raises for
    settings that cannot be honored together (`LogisticRegression(penalty="l1",
    solver="lbfgs")`) and ignores merely irrelevant ones (`SVC(kernel="linear",
    gamma=...)`, silently; `l1_ratio`, with a warning). Ignoring it would fit
    without the offset while `get_params` says `use_exposure=True`; `objective`
    is not tuned, so the raise cannot end a study.
  - *`_check_exposure` stays:* at fit it reads `use_exposure`, at predict the
    fitted state (M10), so `expected` is an argument. Without its `None` check
    an offset model's predict gives all-NaN (`np.asarray(None)` is `nan`) and
    its fit a LightGBM `TypeError` about `init_score`.
- **Ran:** no doc code block was edited.

### A4. Smoke run and docs

**Facts from A3 that A4 builds on** (re-verify in plan mode):
- The API: `IndependentCohortModels({cohort: ModelPipeline(transformer,
  DirectCohortModel(...))})`, then `fit(table, Y, exposure=...)` and
  `predict(table, exposure=...)`, which returns a DataFrame with `Y`'s columns
  and the table's index. Import from `age_group_prediction.modeling`: the
  package root exports the **old** classes (§11).
- Every model gets the same exposure. `use_exposure=False` ignores it; a model
  fitted with an offset raises without one (N3 e). `use_exposure=True` with
  `"regression"` raises (kept by the user at the end of A3).
- Rows are paired by position; there is no index check (N19).
  `Splitter.train_test_split` splits `X`, `y` and `groups`, but **not the
  exposure**: take it as `exposure.loc[X_train.index]`, or by the same positions.
- Nested `set_params` names do not reach into the mapping (N21); each cohort is
  tuned on its own.
- `FEATURE_TRANSFORMATIONS.md` §8.1: `tree` keeps `n_apartments` as a feature
  *and* uses it as the offset, on purpose. Its second code block passes the raw
  `fit_df["n_apartments"]` as the exposure, bypassing `ExposureTransformer`;
  update it to build the exposure with `ExposureTransformer` (N3 d), and run it.
- `MODULE_REFERENCE.md` already has the `independent_cohorts.py` row and the
  new names (A3); A4 only checks it.
- For running doc blocks and the simulator, see A2's "Pitfalls met".

**Decisions to ask the user at the start of A4:**
- Smoke-run cohorts: `tree` for every cohort, with `use_exposure=True` for all,
  or both with and without the offset? (Recommended: both. It feeds §5 step 2
  of `MODEL_REIMPLEMENTATION_PLAN.md`, "re-check the exposure offset".)
- A baseline in the table? (Recommended: the constant rate
  `Σy / Σexposure × exposure` per cohort, so the deviances can be read.)
- Split and settings: grouped 80/20 by `neighborhood_id`, seeds 0–9, default
  (untuned) hyperparameters, as in §6?
- After A4: mark PR #10 ready for review and draft its description, with the
  PR #5 note from §2?

- **Smoke run** (scratchpad script, outside the repo): 10 simulated
  populations; per-cohort held-out Poisson deviance of `IndependentCohortModels`,
  and the deviance of the summed prediction against `n_children_total`. Put the
  table in the plan doc.
- **Docs:** `DIRECT_COHORT_MODEL.md` new §0.6 (`ModelPipeline`,
  `IndependentCohortModels`, and the data flow: `ShareTransformer` and
  `ExposureTransformer` on the full table, then the split);
  `FEATURE_TRANSFORMATIONS.md` §8.1 (combining the cohorts);
  `MODULE_REFERENCE.md` (check); `MODEL_REIMPLEMENTATION_PLAN.md` §5 step 3
  (Model A completed).

Done when:
- [ ] Every edited code block has been run.
- [ ] The user has seen the numbers and reviewed the docs; PR A is ready to merge.

## 9. Steps, PR B: Model 2

### B0. Branch and draft PR
Branch `feat/independent-total-probability-model` from `feat/hyperparameter-tuning`
after PR A is merged. Re-verify Part B against the merged code, update the doc,
open the draft PR.

### B1. Shared numerics
*Revised by N3 and N16: there is no shared `check_exposure` and no utils file.*
- **Build:** `minimize_lbfgs` (N9), **under a class**. Propose its home at the
  start of B1 (for example a shared parent of the two GLMs) and ask the user.
  Each GLM keeps its own presence check for the exposure; the values are
  validated by `ExposureTransformer`. Ask about the length check (§5).
- **Tests:** an iteration limit raises; an overflow raises instead of
  returning the start point.

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
  there are no children, when `temperature ≤ 0`. A passed exposure is ignored (N3 e).
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
- **Build:** an aggregator of two models that take the raw table
  (`ModelPipeline`s, N17). Rows are paired by position (N19).
- **Tests:**
  1. the output equals total × probabilities of the two models fitted separately, and rows sum to the total mean;
  2. the total target is `y.sum(axis=1)`; tables without target columns work;
  3. each pipeline uses its own transformer (swapped transformers);
  4. doubling `n_apartments` doubles every cohort;
  5. nested `set_params` reaches the next fit; a temperature set before `fit` is applied;
  6. the exposure changes the total model only (the probability model ignores it).

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
  `FEATURE_TRANSFORMATIONS.md` §8.3 (rename `composition_base` to
  `cohort_probability_base`, as §7 uses, to match `CohortProbabilityModel`;
  user's decision, 2026-09-29), §8.7 item 7 (penalty ranges restated
  under N8: the old `C` range [0.01, 100] is about `l2_penalty` [2e-6, 2e-2] at
  ~4,500 training children); `MODULE_REFERENCE.md`; `docs/README.md`;
  `MODEL_REIMPLEMENTATION_PLAN.md` §5 (step 3 done).

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
  independence holds per module. Done in A2:
  `test_modeling_never_imports_the_old_stack` checks `modeling/**/*.py` against
  the forbidden list in §4 (static imports only).
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
| New tests | `uv run pytest tests/unit/test_modeling_*.py tests/unit/test_preprocessing.py tests/unit/test_scoring.py -q -W error` |
| Suite | `uv run pytest -m "not slow"` (record the count at each stop) |
| Oracles needing statsmodels | `uv run --group validation pytest tests/validation -k "total_children or nb2" -q` |
| Types | `uv run mypy` and `uv run mypy src/age_group_prediction/modeling src/age_group_prediction/preprocessing.py <changed test files>` |
| Lint | `uv run ruff check <files>`; `uv run ruff format <files>` |
| End to end | The smoke runs of A4 and B7, and the §7 usage block run as written |

Each PR is done when its steps are checked, the suite passes, the independent
review's findings are fixed, and the user has approved the final stop.
