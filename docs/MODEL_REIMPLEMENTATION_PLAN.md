# Plan: Re-implement the Models, Starting with the Base Class and DirectCohortModel

**Branch:** `fix/direct-cohort-fixed-hyperparameters`, rebased onto
`feat/hyperparameter-tuning`. Draft PR #6 merges into `feat/hyperparameter-tuning` (PR #5's branch).
**Status (2026-09-24):** Phases 0, 1 and 2 are done, including the docs cleanup in
Step 2.6. The non-slow suite gives **956 passed**. PR #6 was squash-merged into `feat/hyperparameter-tuning` as `aee3e6a`. Next: roadmap
step 2 (§5).

**Workflow**
- Every step in §4 is a validation stop.
- An independent review subagent checks each step, and its findings are fixed before the stop.
- The only commit made without a further ask is this doc, in Step 0.2, which the draft PR needs.
- Every later commit waits for your explicit approval.

## Contents
1. Context and goal
2. The exposure offset in LightGBM
3. Decisions
4. Steps
5. Roadmap after this PR
6. Files
7. Verification

---

## 1. Context and goal

The current models, `models/base.py` (499 lines) and `models/direct_cohort.py`
(605 lines), mix many concerns:
- `modeling_config` schemas and configs
- Optuna tuning inside `fit`
- NB2 custom gradients
- bootstrap uncertainty
- state bundles
- metadata

They also accept no fixed hyperparameters, so the `hyperparameter_tuning`
package (PR #5, paused at its §9.7) cannot drive them.

**Goal:** small scikit-learn-style models that use nothing from
`modeling_config`. This plan covers the base class and Model A
(`DirectCohortModel`). `modeling_config.py` is deleted only after all three
models are rebuilt.

---

## 2. The exposure offset in LightGBM

**Terms.** The *exposure* $n_i$ (apartments) is the size the expected count
scales with. That's the standard Poisson-GLM term, as in person-years at risk.
Its log, $\log n_i$, is the *offset*: it enters with a coefficient fixed at 1.
statsmodels calls these `exposure=` and `offset=`, R writes `offset(log(n))`,
and LightGBM takes the offset as `init_score`.

With the exposure, the trees learn the cohort's **rate per apartment** rather
than per building. `predict` multiplies that rate back by $n_i$ to give a
count. `base_log_rate_` is the model's **intercept in log space**, the average
log rate per apartment (see "In equations" below).

- **Turning it on.** Construct the model with `use_exposure=True` (Poisson
  only). Pass the exposure to both methods:
  `fit(X, y, exposure=n)` and `predict(X, exposure=n)`.
- **The input.** `exposure` is the raw count `n` (number of apartments), one
  strictly positive value per row of `X`, in the same order, e.g.
  `exposure=df["n_apartments"]`. It is the raw count, not `log n`, because the
  model needs `Σn` for the starting rate below. statsmodels follows the same
  convention: `exposure=` is raw and logged internally, while `offset=` is
  already on the log scale.
- **As a feature too.** To also keep `n_apartments` as an ordinary feature,
  include the column in `X`. The trees can then learn departures from
  proportionality ([FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md)
  §3.3). That is the caller's choice.
- **At fit**, the model passes `log n + base_log_rate_` as `init_score`.
- **Why the starting rate.** Given an `init_score`, LightGBM skips
  `boost_from_average`. The trees would then start from `log n` alone, that is
  1 child per apartment, and would have to learn the real rate (about 0.13)
  through slow gradient steps. `base_log_rate_ = log(Σy / Σn)` restores that
  start in rate space.

  Probe on 300 simulated rows with 50 trees:

  | Start | Training mean prediction (true 6.6) | Test Poisson deviance |
  |---|---|---|
  | `log n` only | 9.95 | 2.30 |
  | `log n + base_log_rate_` | 6.60 | 1.13 |
  | No offset | — | 1.22 |

  Without the starting rate, the offset model is worse than no offset. With
  200 trees the two starts converge.
- **At predict**, LightGBM does **not** add the offset back. A probe with
  lightgbm 4.7.0 gave a mean `predict()` of **0.66** against a true mean of
  **29.9**, while `exp(predict(raw_score=True) + log n)` gave 29.9. So the
  model computes `exp(raw + log n + base_log_rate_)` itself.
- **Regression** has no log link, so an exposure is refused.

### In equations

The equations, the derivation of $b$, and the correct inputs now live in
[DIRECT_COHORT_MODEL.md §0.1](DIRECT_COHORT_MODEL.md#01-the-model), the
reference for the rebuilt model. They were moved there in Step 2.6a so that
there is one copy.

---

## 3. Decisions (agreed 2026-09-24)

| # | Decision | Why |
|---|---|---|
| M1 | A new package, `src/age_group_prediction/modeling/`, alongside the old code | The same approach as `splitting` and `feature_engineering`. The old `models/`, `experiment/` and `tracking/` code and their tests keep working until all three models are rebuilt. They are then deleted together with `modeling_config.py` |
| M2 | `BaseAgeGroupModel(BaseEstimator, ABC)`. Settings go in `__init__`, stored verbatim. Data are method arguments only. Fitted state lives in trailing-underscore attributes | scikit-learn then provides `get_params`, `set_params` and `clone`, which the tuning evaluator needs (PR #5, D11) |
| M3 | Base API: abstract `fit(X, y)` and `predict(X)` (*`exposure=None` added to both in PR #5, D14*), and a concrete `evaluate(y_true, y_pred, metric) -> float` that does not call `predict` | A metric is applied to the targets and the predictions. `evaluate` stays a method so a later model (B or C) can override how it scores. `RegressorMixin` is not used, because its `score` (R²) would be a second scoring path |
| M4 | A metric is a `Metric(name, function, greater_is_better=False)` in `modeling/metrics.py` (*moved to the top-level `scoring.py` in PR #5, D16*), a frozen standard-library dataclass checked in `__post_init__`, like `splitting.Splitter`. It is not callable: `model.evaluate` is the one named way to score. `function(y_true, y_pred) -> float` may be any callable, from sklearn or custom. Ready-made: `POISSON_DEVIANCE`, `RMSE` and `MAE` | Custom metrics can't be assumed to be lower-is-better, so the direction is stored, as sklearn's `make_scorer` does, and the tuner's sign rule (D12) reads it. The package-local module (now the top-level `scoring.py`, D16) avoids the old top-level `metrics.py`, whose `Metric` Protocol needs a `PredictionResult` and is deleted with the old stack. Not pydantic: a metric is built only in code (a function can't come from a config), so parsing adds nothing, and a dataclass already rejects a misspelled keyword |
| M5 | `get_metadata` is **not added** (decided in Step 2.5). Configuration comes from `get_params()`, and fitted state from the public trailing-underscore attributes (`regressor_`, `base_log_rate_`). Revisit when the new tracking integration exists, and define the method then on `BaseAgeGroupModel` with the fields the tracker actually reads | No caller exists in the new code. A method without a consumer would lock in a guessed schema, which is how the old one grew nine placeholder keys (`models/base.py:460-472`) |
| M6 | One `DirectCohortModel` instance per cohort. `y` is a Series, and `predict` returns a 1-D array of means | Cohorts are independent, so each gets its own features, target and tuned hyperparameters |
| M7 | Only the built-in objectives `"poisson"` and `"regression"`. No NB2, no `custom_nb2_gradient` and no dispersion | Requirement. The old `nb2_gradient_hessian` stays in `distributions.py` until the old stack is deleted (M1) |
| M8 | Fixed hyperparameters are explicit keyword arguments with LightGBM's defaults: `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `min_split_gain`, `subsample` and `colsample_bytree`, plus `random_state=42` and `n_jobs=1`. No Optuna runs inside `fit` | `set_params(**trial_params)` needs explicit arguments. `n_jobs=1` avoids the OpenMP crash alongside torch on macOS. See the note below for the fixed internals |
| M9 | The model takes a finished design matrix `X`. Preprocessing (e.g. a `FeatureTransformer` fitted per fold) happens before the model, which holds no transformer | Your requirement. It keeps the model to one job, fitting trees, and whoever builds the folds decides how the features are made |
| M10 | `use_exposure: bool = False` in the constructor, with the raw exposure passed as `fit(X, y, exposure=n)` and `predict(X, exposure=n)`. When on, `fit` learns `base_log_rate_ = log(Σy / Σn)` and passes `log n + base_log_rate_` as `init_score`, and `predict` returns `exp(raw + log n + base_log_rate_)`. `ValueError`, only for what would otherwise pass silently: `use_exposure` with `"regression"`; `exposure` not passed exactly when `use_exposure` is on (a forgotten one would drop the offset, an unexpected one would be ignored); an exposure that is not strictly positive and finite (LightGBM accepts a `-inf`/`nan` `init_score`). LightGBM already raises for an unknown objective, a wrong-length exposure and an all-zero `y`, so the model doesn't repeat those checks | See §2. The indicator makes "with or without the offset" a declared setting, so it survives `clone`/`set_params` and a tuner can compare both. It also turns a forgotten `exposure` at predict time into an error rather than silent per-apartment rates. The starting rate replaces the `boost_from_average` that LightGBM switches off when given an `init_score`. `use_exposure` with `"regression"` raises rather than being ignored: like sklearn, a pair of settings that can't be honored together is an error, whereas a merely irrelevant setting is only ignored with a warning. Ignoring it would silently fit a model without the offset and discard the exposure passed in |
| M11 | Validation happens in `fit`, not in `__init__` | `set_params` bypasses `__init__` |
| M12 | Dropped from Model A: bootstrap draws, intervals, pointwise log probabilities, `PredictionResult`, state bundles, `configuration_record`, seed records, timers and `minimum_mean` clipping | Out of scope ("means only for now") or not needed. Poisson means are `exp(·) > 0`, and regression output is returned as is. A regression model can predict negative values, and scoring it with `POISSON_DEVIANCE` then raises. That is correct, because the metric is undefined there |

**Fixed internals (M8):**
- `deterministic=True`, so repeated fits give identical trees.
- `force_col_wise=True`, which LightGBM's docs recommend alongside `deterministic`.
- `verbosity=-1`, which silences LightGBM's messages.
- `subsample_freq=1`, set only when `subsample < 1`. Without it, LightGBM
  ignores `subsample`: a probe with `subsample=0.5` gave identical predictions.

### Target code shape

*As planned in this PR. Since then (PR #5), `Metric` lives in the top-level
`scoring.py` (D16), and `fit`/`predict` take `exposure=None` (D14).*

```python
# modeling/metrics.py
@dataclass(frozen=True)
class Metric:
    name: str
    function: Callable[[ArrayLike, ArrayLike], float | np.floating]
    greater_is_better: bool = False


POISSON_DEVIANCE = Metric("poisson_deviance", mean_poisson_deviance)
RMSE = Metric("rmse", root_mean_squared_error)
MAE = Metric("mae", mean_absolute_error)


# modeling/base.py
class BaseAgeGroupModel(BaseEstimator, ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def evaluate(self, y_true, y_pred, metric: Metric) -> float: ...


# modeling/direct_cohort.py
Objective = Literal["poisson", "regression"]


class DirectCohortModel(BaseAgeGroupModel):
    def __init__(self, *, objective: Objective = "poisson", use_exposure: bool = False,
                 n_estimators=100, ..., random_state=42, n_jobs=1): ...

    def fit(self, X, y, exposure: ArrayLike | None = None) -> Self: ...

    def predict(self, X, exposure: ArrayLike | None = None) -> np.ndarray: ...

    # fitted: regressor_, base_log_rate_ (None without an exposure)
```

Usage: transform first, then one instance per cohort.

```python
features = clone(tree).fit(train_df)
X_train, X_test = features.transform(train_df), features.transform(test_df)
models = {cohort: DirectCohortModel(use_exposure=True) for cohort in COHORTS}
for cohort, model in models.items():
    model.fit(X_train, train_df[cohort], exposure=train_df["n_apartments"])
predictions = {
    cohort: model.predict(X_test, exposure=test_df["n_apartments"])
    for cohort, model in models.items()
}
```

---

## 4. Steps

Each step ends with a stop for your validation.

### Phase 0: plan and draft PR

**Step 0.1: write this plan doc.**
- [x] You have read and approved this doc, and any requested changes are applied.

**Step 0.2: open the draft PR.**
1. Commit only this doc, as `docs: plan the model re-implementation`.
2. Push the branch: `git push -u origin fix/direct-cohort-fixed-hyperparameters`.
3. Open the PR against `feat/hyperparameter-tuning`, titled "Re-implement the
   models: base class and DirectCohortModel". The body gives a summary, a
   checklist of these steps and a link to this doc.

Done when:
- [x] Draft PR #6 is open: https://github.com/galkampel/age-group-prediction/pull/6

### Phase 1: base class and metrics

The base class holds only what every model shares: the `fit` / `predict` /
`evaluate` contract.

It deliberately has none of the following:
- **An `__init__`.** There are no shared settings, and sklearn reads each
  subclass's own constructor.
- **`get_metadata`.** Deferred (M5).
- **Fitted-state helpers.**

Two tests are left out because they would test libraries rather than our code:
- **"The base cannot be instantiated"** tests Python's `ABC`.
- **`clone` / `set_params`** tests sklearn. It runs in Step 2.3 instead, on the
  real `DirectCohortModel` constructor.

*Revised 2026-09-24.* The first version of Step 1.1 put a name-keyed `METRICS`
table in `base.py`, and `evaluate(X, y, metric)` called `predict` itself. After
review, a metric became a class in `modeling/metrics.py`, and `evaluate` scores
`(y_true, y_pred)` (M3, M4).

**Step 1.1: `modeling/metrics.py`.** This file comes first because `base.py`
imports it. It contains:
- `Metric(name, function, greater_is_better=False)`, a frozen standard-library
  dataclass whose `__post_init__` checks for a non-empty name, a callable
  function and a bool direction;
- three ready-made metrics: `POISSON_DEVIANCE`, `RMSE` and `MAE`.

Checklist:
- [x] The file contains only these items.
- [x] Every docstring states the "why".

**Step 1.2: revise `modeling/base.py`.**
- Remove `MetricName` and `METRICS`.
- `evaluate(y_true, y_pred, metric)` returns
  `float(metric.function(y_true, y_pred))`. The cast turns a NumPy scalar from a
  custom metric into a plain float.
- `fit` and `predict` are unchanged.

Checklist:
- [x] The file contains only the class.
- [x] The docstrings are updated.

**Step 1.3: `modeling/__init__.py`.** A short docstring saying what the package
is, and that it replaces `models/` and `modeling_config` (M1). It exports
`BaseAgeGroupModel`, `Metric`, `POISSON_DEVIANCE`, `RMSE` and `MAE`.
- [x] Importing these names from `age_group_prediction.modeling` works.

**Step 1.4: register `modeling/` with mypy.** In `pyproject.toml`, add the
package to `[tool.mypy].files` and to the strict override, as every other new
package is. Without this, mypy silently skips it.
- [x] `uv run mypy` passes.
- [x] `uv run ruff check` passes on the new files.

**Step 1.5: tests.**

`tests/unit/test_modeling_metrics.py`:
- The ready-made metrics match sklearn, parametrized over the three. This checks
  that each name is wired to its function and that `greater_is_better` is
  `False`.
- Invalid construction is rejected: an empty name, a function that isn't
  callable, and a `greater_is_better` that isn't a bool.

`tests/unit/test_modeling_base.py`:
- `evaluate` applies a custom (non-sklearn) metric to `(y_true, y_pred)` and
  returns a Python `float`. The test uses a minimal subclass, because the base
  class is abstract.

Done when:
- [x] The new tests pass.
- [x] `uv run pytest -m "not slow"` passes: **940** (932 + 8 new).
- [x] An independent review of Phase 1 is done and its findings are fixed.

**Review record.** No bugs were found. Fixed:
- **Unknown keywords.** A misspelled `greater_is_better` now raises instead of
  silently keeping the default direction. It has a test case. (This was first
  fixed with pydantic's `extra="forbid"`; after the later switch to a standard
  dataclass, the dataclass constructor rejects it natively.)
- **`function` type.** Now `Callable[[ArrayLike, ArrayLike], float | np.floating]`,
  which states the `(y_true, y_pred)` contract.
- **Wiring test.** It checks identity (`is`), not values.
- **The stand-in model in the `evaluate` test.** It no longer has fit/predict
  logic of its own.
- **Invalid-construction cases.** Built with lambdas, so the ignores are
  narrowed.

**Design follow-up (your questions).** `Metric` was switched from a pydantic
dataclass to a standard frozen dataclass (M4), and it is deliberately not
callable (M3).

Not changed:
- **Whitespace-only names** are accepted. Low value.
- **`Metric` equality** compares function identity. This is noted for a tuner
  that might deduplicate metrics.

### Phase 2: DirectCohortModel

*Revised 2026-09-24, after Phase 1.*
- **Starting rate.** §2 adds `base_log_rate_`, found by probing LightGBM.
- **`fit` and `predict` merged.** They form one step, because the offset is a
  single contract across both methods, and a fit can only be checked through
  its predictions.
- **Tests reworked.** Each test now catches a mistake in our own code. Tests
  that only checked a library (sklearn's `get_params` and `check_is_fitted`,
  LightGBM's seeding) or restated the implementation were dropped.

*Revised again after Step 2.1.*
- **Transformer removed.** The model takes a finished design matrix (M9).
- **`use_exposure` indicator added.** The raw exposure is passed to `fit` and
  `predict` (M10, §2).

**Step 2.1: constructor and validation.**
- `__init__` stores every argument verbatim (M8). `objective` is typed
  `Objective = Literal["poisson", "regression"]`, and `use_exposure: bool = False`
  (M10). There is no transformer (M9).
- A private check, `_check_exposure`, rejects `use_exposure` combined with
  `"regression"`. An unknown objective is left to LightGBM, which raises
  `Unknown objective type name`; the `Objective` Literal documents the two
  supported values.

Done when:
- [x] The code reads cleanly. Its behavior is tested in Step 2.3.

**Step 2.2: `fit` and `predict` (M10).**

Both methods first call `_check_exposure`, which holds the three checks for
what would otherwise pass silently:
- regression combined with `use_exposure`;
- `exposure` passed exactly when `use_exposure` is on;
- the exposure is strictly positive and finite.

A wrong-length exposure and an all-zero `y` are left to LightGBM, which raises
for both.

`fit`:
1. Run `_check_exposure`.
2. With the exposure on, set `base_log_rate_ = log(Σy / Σn)` and fit the
   `LGBMRegressor` with `init_score = log n + base_log_rate_`. Otherwise set
   `base_log_rate_ = None` and pass no `init_score`.
3. Store `regressor_` and `base_log_rate_`.

`predict`:
1. Call `check_is_fitted`, then run the exposure check.
2. Without the exposure, return `regressor_.predict(X)`.
3. With it, return
   `exp(regressor_.predict(X, raw_score=True) + log n + base_log_rate_)`.

Done when:
- [x] On a small frame, the mean training prediction is close to the mean of `y`.
  With 300 rows and 20 trees, the mean of `y` is 6.677. Poisson with the
  exposure gives 6.684, Poisson without it 6.687, and regression 6.677.
  Doubling the exposure doubles every prediction exactly.

`DirectCohortModel` and `Objective` are exported from `modeling`.

*Trimmed after review.* The checks went from 7 to 3. Four were removed because
LightGBM or NumPy already raises for them, as a probe confirmed: an unknown
objective, a wrong-length exposure at fit, and an all-zero `y`. At predict, a
wrong length above 1 raises a NumPy broadcast error. A scalar or length-1
exposure broadcasts to every row, meaning "all buildings have this `n`". The two presence checks were merged into one. Test
6 pins the errors LightGBM raises. The class docstring and a comment in `fit`
now say that the exposure gives a rate per apartment, and that
`base_log_rate_` is the intercept.

**Step 2.3: `tests/unit/test_modeling_direct_cohort.py`.** Seven tests (16
cases) on a small synthetic frame. Each one names the mistake it catches:
1. **Constructor stores arguments verbatim**, which the tuner relies on:
   `clone(model).set_params(n_estimators=5)` changes only the copy.
2. **Exposure misuse raises**, parametrized: missing while `use_exposure` is on;
   given while it's off; zero, negative or infinite.
3. **Calibration:** the mean training prediction is close to the mean of `y`
   with 20 trees. It is parametrized over Poisson with an exposure, Poisson
   without one, and regression. It catches a missing starting rate (21.75
   against 6.68) or a broken offset. The two cases without the exposure are the
   only cover of the plain predict path.
4. **Proportionality:** doubling `exposure` at predict time doubles the
   prediction to 1e-12. This is exact because the exposure is not a feature
   unless the caller adds it. It catches an offset ignored at predict time.
5. **Bagging is active:** `subsample=0.5` gives different predictions from
   `subsample=1.0`. It catches a missing `subsample_freq`.
6. **Invalid input surfaces an error**, parametrized:
   - regression with `use_exposure` (our check);
   - an unknown objective, a wrong-length exposure at fit, and an all-zero `y`
     with the exposure on. These are LightGBM's errors. The test pins the
     behavior that the removed checks now rely on.
7. **Predict follows the fitted model** (added after review): after
   `set_params(use_exposure=False)` on a model fitted with the exposure,
   `predict(X)` raises instead of returning rates per apartment.

Done when:
- [x] The new tests pass: 16 cases in about 3 s.
- [x] `uv run pytest -m "not slow"` passes: **956** (940 + 16), after the review
  fixes.
- [x] mypy and ruff pass, run on the changed files only.
- [x] An independent review is done and its findings are fixed.

**Mutation checks.** Each one broke the code on purpose, and the matching test
failed:
- no starting rate: calibration (Poisson with exposure) fails;
- offset dropped at predict: calibration and proportionality fail;
- `subsample_freq` always 0: the bagging test fails;
- predict following the current `use_exposure`: test 7 fails.

**Review record.**

Fixed:
- **Bug:** `predict` read the current `use_exposure`, so a
  `set_params(use_exposure=False)` after fitting silently returned rates per
  apartment (mean 0.99 against 6.68). It now follows the fitted state
  (`base_log_rate_ is not None`), and the regression rule moved into `fit`.
- `regressor_` and `base_log_rate_` are now assigned together, only after
  LightGBM's fit succeeds. A failed refit could otherwise pair new trees with
  an old intercept.
- Tests: an infinite exposure case; the calibration comment now quotes the
  20-tree figure; `objective: Objective` replaces a type ignore.
- Doc: the status line, and the predict-time wrong-length wording.

Noted, not changed:
- A NaN in `y` passes silently: `np.sum` on a Series skips it, and LightGBM
  accepts NaN labels. This is outside the three chosen checks.

**Step 2.4: smoke run on simulated data.** A scratchpad script that:
1. builds the table with `StudentPopulationSimulator` and `ShareTransformer`;
2. splits it with `Splitter("grouped")`, and fits the Model A
   `FeatureTransformer` (FEATURE_TRANSFORMATIONS §8.1) on the training rows,
   outside the model;
3. fits three Poisson models, one per cohort, each with and without the
   exposure (`exposure=df["n_apartments"]`);
4. reports `poisson_deviance` per cohort, with the table added to this doc.

Done when:
- [x] You have seen the numbers. They are evidence for the "+ size offset"
  variation in FEATURE_TRANSFORMATIONS §5.

**Results (2026-09-24).**
- **Data:** 10 simulated populations (RNG seeds 0–9) of 245 buildings each,
  with a grouped split by neighborhood: 196 training and 49 test buildings.
- **Models:** Poisson with default hyperparameters (100 trees, untuned), and
  `n_apartments` kept in `X` in both variants.
- **Score:** held-out mean Poisson deviance, lower is better, as mean ± SD over
  the 10 populations. "Diff" is with the exposure minus without it, paired by
  population.

| Cohort | Without exposure | With exposure | Diff (mean ± SD) | Exposure wins |
|---|---|---|---|---|
| `n_kindergarten` | 2.732 ± 0.911 | 2.628 ± 0.877 | −0.105 ± 0.169 | 7 / 10 |
| `n_elementary` | 2.124 ± 0.515 | 2.128 ± 0.499 | +0.004 ± 0.200 | 6 / 10 |
| `n_highschool` | 2.658 ± 0.703 | 2.572 ± 0.710 | −0.086 ± 0.207 | 7 / 10 |

**Reading.**
- The exposure lowers deviance by about 4% for kindergarten and high school,
  and changes nothing for elementary.
- The paired differences are about as large as their spread. For kindergarten
  the mean difference is about 2 standard errors from zero; for high school it
  is about 1.3.
- So this is weak evidence for the offset, not the "largest gain" §5 expected.
  With `n_apartments` already a feature, the trees learn much of the size
  effect themselves.
- Calibration holds in all six cases: the mean test prediction over the mean
  test target is 1.00–1.06 on average across populations.
- The models are untuned, and 196 training rows is small. Tuning (#5) is where
  the offset should be re-checked.

The script is `smoke_exposure.py` in the session scratchpad, outside the repo.

**Step 2.5: decide `get_metadata` (M5).** Propose keeping or dropping it, with
a reason based on what exists by then.
- [x] You have decided, and this doc records the decision.

**Decision (2026-09-24): not added** (M5).
- Nothing in `modeling/`, `hyperparameter_tuning/`, `splitting/` or
  `feature_engineering/` calls `get_metadata`. Only the old `experiment/` code
  does, and it is deleted with `modeling_config`.
- `get_params()` is already plain JSON (checked with `json.dumps`), and the
  fitted state is public. `mlflow.log_params(model.get_params())` plus the
  model artifact covers logging.

**Step 2.6: docs, and a docs cleanup.** You widened this step: document the
rebuilt model, fix outdated docs, and remove completed plans. A doc was deleted
only if it held no information for future steps (or that information was moved
first), and no kept doc or code depended on its content. Every file was read in
full before deciding.

| Sub-step | What was done |
|---|---|
| 2.6a | `DIRECT_COHORT_MODEL.md` §0: the rebuilt model, with the equations moved in from §2 of this doc |
| 2.6b | `FEATURE_TRANSFORMATIONS.md`: intro, Model A passages, §8.0 room shares (4/5/6 via `ShareTransformer`), §8.7 items 1 and 3 done. Every §8 code block was run on a simulated table |
| 2.6c | `MODULE_REFERENCE.md`: the `modeling` section, plus `preprocessing.py` and `hyperparameter_tuning/` |
| 2.6d | Deleted the three completed plans below, and fixed their links |
| 2.6e, then R | Deleted `IMPLEMENTATION_PLAN.md`, **then restored it** (`767a47a`). A full read found future-stage validation content (§13–§16, §19) that exists nowhere else. The first check had looked only at model content |
| E | Moved the two open items of the deleted EDA plan into `SIMPLIFIED_MODEL_PLAN.md` §9 Stage 6 and `TODO.md` |
| 2.6f | Moved the only copies of the Gate 8 canonical run record (into `GATE_VALIDATION_FINDINGS.md`) and of the composition-first rationale (into `CROSS_VALIDATION_AND_SELECTION.md` §9). Checked them byte for byte, then deleted `docs/archive/` |
| 2.6g | Rewrote `docs/README.md` as a grouped index of every remaining doc |
| 2.6h | This record, the §2 link, §5, §6 and the status line |

| Doc | Decision | Why |
|---|---|---|
| `COMPACT_SIMULATOR_MIGRATION_PLAN.md`, `MARIMO_MIGRATION_PLAN.md` | Deleted (in git) | Completed; no open items |
| `EDA_AND_PREDICTIVE_MODELING_PLAN.md` | Deleted (in git) | Completed; its two open items moved (E) |
| `docs/archive/` (19 files) | Deleted (never committed, so gone for good) | Finished handoffs; the two unique pieces moved (2.6f) |
| `MODELING_REBUILD_PLAN.md`, `GATE_VALIDATION_FINDINGS.md`, `GATE_9_INDEPENDENT_VALIDATION_REPORT.md` | Kept until the old stack is deleted (§5) | The running old code cites the rebuild plan, and the three link to each other |
| `DATA_GENERATION_PLAN.md`, `PARAMETER_REFERENCE.md`, `PROBLEM_DEFINITION.md`, `IMPLEMENTATION_PLAN.md`, `legacy/*` | Kept | They extend the simplified model (time, projects, building types, dynamics, future validation) beyond `SIMPLIFIED_MODEL_PLAN.md` §9 |

Done when:
- [ ] You have reviewed the docs.

---

## 5. Roadmap after this PR

In order. Each step has its own plan and gated phases.

1. ✓ **Merge PR #6 into `feat/hyperparameter-tuning`**: squash-merged as
   `aee3e6a` (2026-09-24).
2. **Resume the tuning package** at
   [HYPERPARAMETER_TUNING_PLAN.md §9](HYPERPARAMETER_TUNING_PLAN.md): switch
   the evaluator to `BaseAgeGroupModel`, then tune `DirectCohortModel` per
   cohort. Re-check the exposure offset once tuned; Step 2.4 found only weak
   evidence, untuned.
3. **Rebuild Model B** (`IndependentTotalProbabilityModel`) in `modeling/`,
   with its own plan doc. Its feature declarations are already in
   [FEATURE_TRANSFORMATIONS.md §8.2–8.5](FEATURE_TRANSFORMATIONS.md).
4. **Rebuild Model C** (`BayesianConditionalModel`). *Needs step 3*: C reuses
   B's frozen feature forms.
5. **Delete the old stack:** `modeling_config.py`, `models/`,
   `nb2_gradient_hessian` and `tuning.py`; rewire `experiment/` and `tracking/`
   onto `modeling/`; delete `MODELING_REBUILD_PLAN.md`,
   `GATE_VALIDATION_FINDINGS.md` and `GATE_9_INDEPENDENT_VALIDATION_REPORT.md`,
   which only the old code cites. *Needs steps 3 and 4.*

---

## 6. Files

- **New:**
  - `src/age_group_prediction/modeling/__init__.py`
  - `src/age_group_prediction/modeling/base.py`
  - `src/age_group_prediction/modeling/metrics.py`
  - `src/age_group_prediction/modeling/direct_cohort.py`
  - `tests/unit/test_modeling_base.py`
  - `tests/unit/test_modeling_metrics.py`
  - `tests/unit/test_modeling_direct_cohort.py`
- **Reused:**
  - `feature_engineering.FeatureTransformer`: used by callers and the smoke run, not by the model (M9)
  - `splitting.Splitter`
  - `preprocessing.ShareTransformer`
  - `student_simulator.pipeline.StudentPopulationSimulator`
- **Docs edited (Step 2.6):** `DIRECT_COHORT_MODEL.md`, `FEATURE_TRANSFORMATIONS.md`,
  `MODULE_REFERENCE.md`, `README.md` (root and `docs/`), `SIMPLIFIED_MODEL_PLAN.md`,
  `TODO.md`, `CROSS_VALIDATION_AND_SELECTION.md`, `GATE_VALIDATION_FINDINGS.md`,
  `MODELING_REBUILD_PLAN.md`, the two `legacy/` files.
- **Docs deleted (Step 2.6):** see the table in Step 2.6.
- **Not touched:** `modeling_config.py`, `models/*`, `experiment/*`, `tracking/*`.

---

## 7. Verification

- `uv run pytest tests/unit/test_modeling_base.py tests/unit/test_modeling_direct_cohort.py`
- `uv run pytest -m "not slow"`: the existing suite is unchanged.
- The Step 2.4 smoke run, with its results recorded in §4.
- An independent review subagent before each stop.
