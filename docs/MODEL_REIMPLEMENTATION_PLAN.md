# Plan: Re-implement the Models, Starting with the Base Class and DirectCohortModel

**Branch:** `fix/direct-cohort-fixed-hyperparameters`, rebased onto
`feat/hyperparameter-tuning`. Draft PR #6 merges into `feat/hyperparameter-tuning` (PR #5's branch).
**Status (2026-09-24):** Phase 0 is done. The non-slow suite on the rebased branch
gives **932 passed**. Phase 1 is done (940 passed) and awaiting your validation (uncommitted). Next: Phase 2, Step 2.1.

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
5. Out of scope
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

LightGBM's `init_score` is the offset.

- **Declaring it.** Set `exposure_column="n_apartments"` on the
  `FeatureTransformer`. Also keep `n_apartments` in a plan as an ordinary
  feature, so the trees can learn departures from proportionality
  ([FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md) §3.3 and §8.1).
- **At fit**, the model passes `transformer.log_exposure(X)` as `init_score`.
- **At predict**, LightGBM does **not** add the offset back. A probe with
  lightgbm 4.7.0 gave a mean `predict()` of **0.66** against a true mean of
  **29.9**, while `exp(predict(raw_score=True) + log n)` gave 29.9. So the
  model computes that sum itself.
- **Regression** has no log link, so an exposure is refused.

---

## 3. Decisions (agreed 2026-09-24)

| # | Decision | Why |
|---|---|---|
| M1 | A new package, `src/age_group_prediction/modeling/`, alongside the old code | The same approach as `splitting` and `feature_engineering`. The old `models/`, `experiment/` and `tracking/` code and their tests keep working until all three models are rebuilt. They are then deleted together with `modeling_config.py` |
| M2 | `BaseAgeGroupModel(BaseEstimator, ABC)`. Settings go in `__init__`, stored verbatim. Data are method arguments only. Fitted state lives in trailing-underscore attributes | scikit-learn then provides `get_params`, `set_params` and `clone`, which the tuning evaluator needs (PR #5, D11) |
| M3 | Base API: abstract `fit(X, y)` and `predict(X)`, and a concrete `evaluate(y_true, y_pred, metric) -> float` that does not call `predict` | A metric is applied to the targets and the predictions. `evaluate` stays a method so a later model (B or C) can override how it scores. `RegressorMixin` is not used, because its `score` (R²) would be a second scoring path |
| M4 | A metric is a `Metric(name, function, greater_is_better=False)` in `modeling/metrics.py`, a frozen standard-library dataclass checked in `__post_init__`, like `splitting.Splitter`. It is not callable: `model.evaluate` is the one named way to score. `function(y_true, y_pred) -> float` may be any callable, from sklearn or custom. Ready-made: `POISSON_DEVIANCE`, `RMSE` and `MAE` | Custom metrics can't be assumed to be lower-is-better, so the direction is stored, as sklearn's `make_scorer` does, and the tuner's sign rule (D12) reads it. The package-local module avoids the old top-level `metrics.py`, whose `Metric` Protocol needs a `PredictionResult` and is deleted with the old stack. Not pydantic: a metric is built only in code (a function can't come from a config), so parsing adds nothing, and a dataclass already rejects a misspelled keyword |
| M5 | `get_metadata` is deferred and decided in Step 2.6 | It may be needed. Until then, `get_params()` and the public fitted attributes cover it |
| M6 | One `DirectCohortModel` instance per cohort. `y` is a Series, and `predict` returns a 1-D array of means | Cohorts are independent, so each gets its own features, target and tuned hyperparameters |
| M7 | Only the built-in objectives `"poisson"` and `"regression"`. No NB2, no `custom_nb2_gradient` and no dispersion | Requirement. The old `nb2_gradient_hessian` stays in `distributions.py` until the old stack is deleted (M1) |
| M8 | Fixed hyperparameters are explicit keyword arguments with LightGBM's defaults: `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `min_split_gain`, `subsample` and `colsample_bytree`, plus `random_state=42` and `n_jobs=1`. No Optuna runs inside `fit` | `set_params(**trial_params)` needs explicit arguments. `n_jobs=1` avoids the OpenMP crash alongside torch on macOS. See the note below for the fixed internals |
| M9 | The constructor takes `transformer: FeatureTransformer`, and `fit` uses `clone(transformer).fit(X)` | Each fold learns its own statistics, and the caller's declaration is never mutated |
| M10 | When `transformer_.exposure_column` is set, `fit` passes `log n` as `init_score` and `predict` returns `exp(raw + log n)`. `"regression"` with an exposure raises | See §2 |
| M11 | Validation happens in `fit`, not in `__init__` | `set_params` bypasses `__init__` |
| M12 | Dropped from Model A: bootstrap draws, intervals, pointwise log probabilities, `PredictionResult`, state bundles, `configuration_record`, seed records, timers and `minimum_mean` clipping | Out of scope ("means only for now") or not needed. Poisson means are `exp(·) > 0`, and regression output is returned as is |

**Fixed internals (M8):**
- `deterministic=True`
- `force_col_wise=True`
- `verbosity=-1`
- `subsample_freq=1`, set only when `subsample < 1`. Without it, LightGBM ignores `subsample`.

### Target code shape

```python
# modeling/metrics.py
@dataclass(frozen=True)
class Metric:
    name: str
    function: Callable[..., float]  # function(y_true, y_pred) -> float
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
class DirectCohortModel(BaseAgeGroupModel):
    def __init__(self, transformer, *, objective="poisson", n_estimators=100, ...,
                 random_state=42, n_jobs=1): ...
```

Usage: one instance per cohort.

```python
models = {cohort: DirectCohortModel(tree, objective="poisson") for cohort in COHORTS}
for cohort, model in models.items():
    model.fit(X_train, y_train[cohort])
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
- **`clone` / `set_params`** tests sklearn. It runs in Step 2.4 instead, on the
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

**Step 2.1: constructor and validation.** Add `__init__` (M8, M9), which stores
every argument verbatim. Add a check, called from `fit`, that rejects an
unknown objective and `"regression"` combined with an exposure.
- [ ] `get_params()` lists exactly the constructor arguments.

**Step 2.2: `fit`.**
1. Clone and fit the transformer.
2. Build the `LGBMRegressor` from the hyperparameters and the fixed internals (M8).
3. Fit it with `init_score=log n` when an exposure is declared (M10).
4. Store `transformer_` and `regressor_`.

Done when:
- [ ] Fitting a small simulated frame works for both objectives.

**Step 2.3: `predict`.**
1. Call `check_is_fitted`, then transform `X`.
2. Without an exposure, return `regressor_.predict(features)`.
3. With an exposure, return
   `exp(regressor_.predict(features, raw_score=True) + log n)`.

Done when:
- [ ] A Poisson model with an exposure has a mean training-data prediction close
  to the mean of `y`. The naive path gets this wrong (§2).

**Step 2.4: `tests/unit/test_modeling_direct_cohort.py`.** The tests cover:
- `clone` and `set_params` round-trip;
- the caller's transformer stays unfitted after `fit`;
- `NotFittedError` is raised before `fit`;
- Poisson with an exposure: `predict == exp(raw + log n)`;
- with `n_apartments` as the exposure only, doubling it doubles the prediction exactly;
- Poisson without an exposure works;
- `"regression"` with an exposure raises, and so does an unknown objective;
- a fixed `random_state` with `subsample < 1` is reproducible;
- `evaluate` matches sklearn.

Done when:
- [ ] The new tests pass.
- [ ] The full non-slow suite passes.
- [ ] The review findings are fixed.

**Step 2.5: smoke run on simulated data.** A scratchpad script that:
1. builds the table with `StudentPopulationSimulator` and `ShareTransformer`;
2. splits it with `Splitter("grouped")`;
3. fits three Poisson instances, one per cohort, both with and without the
   exposure;
4. reports `poisson_deviance` per cohort, with the table added to this doc.

Done when:
- [ ] You have seen the numbers. They are evidence for the "+ size offset"
  variation in FEATURE_TRANSFORMATIONS §5.

**Step 2.6: decide `get_metadata` (M5).** Propose keeping or dropping it, with
a reason based on what exists by then.
- [ ] You have decided, and this doc records the decision.

**Step 2.7: docs.**
- `DIRECT_COHORT_MODEL.md`: add a new-model section and mark the old one superseded.
- `FEATURE_TRANSFORMATIONS.md`: drop NB2 from Model A, add the `raw_score`
  detail to §3.3 and §8.1, and mark §8.7 item 3 done.
- `MODULE_REFERENCE.md`: add the `modeling` package.
- This doc: update the status line.

Done when:
- [ ] You have reviewed the docs.
- [ ] The PR checklist is ticked for each approved commit.

---

## 5. Out of scope (later plans)

- Rebuilding Model B (`IndependentTotalProbabilityModel`) and Model C
  (`BayesianConditionalModel`).
- Switching the tuning evaluator to `BaseAgeGroupModel` (PR #5 §9.7). This
  plan supplies what it needs: `clone`, `set_params` and
  `evaluate(y_true, y_pred, metric)` with a `Metric` that carries its direction.
- Deleting `modeling_config.py`, the old `models/`, `nb2_gradient_hessian` and
  `tuning.py`, and rewiring `experiment/` and `tracking/`.

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
  - `feature_engineering.FeatureTransformer` (`transform`, `log_exposure`, `exposure_column`)
  - `splitting.Splitter`
  - `preprocessing.ShareTransformer`
  - `student_simulator.pipeline.StudentPopulationSimulator`
- **Not touched:** `modeling_config.py`, `models/*`, `experiment/*`, `tracking/*`.

---

## 7. Verification

- `uv run pytest tests/unit/test_modeling_base.py tests/unit/test_modeling_direct_cohort.py`
- `uv run pytest -m "not slow"`: the existing suite is unchanged.
- The Step 2.5 smoke run, with its results recorded in §4.
- An independent review subagent before each stop.
