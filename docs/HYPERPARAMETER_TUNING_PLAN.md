# Plan: modular Optuna hyperparameter tuning

**Branch:** `feat/hyperparameter-tuning` · draft PR #5 · **Status:** Phase 1 done:
reviewed, full suite 917 passed (880 on `main` + 37 new), 0 failed. Waiting for
your Phase 1 approval and commit. Next: Phase 2 (objective).
**Workflow:** each phase ends with an independent review, then stops for your
approval. Nothing is committed without your approval.

## Contents
1. Context and goal
2. Key decisions
3. Package layout
4. Design: parameters, objective, study
5. Worked example: `DirectCohortModel`'s LightGBM tuning
6. Implementation phases and sub-tasks
7. Verification
8. Draft PR

---

## 1. Context and goal

Every Optuna objective in the repo today is a closure inside a model
(`models/direct_cohort.py:128-148` with `_suggest_parameters`, and
`models/independent_total_probability.py:178-200`). Each closure hard-codes the
search space, the study settings and the fold loop together, so none of it can
be reused, configured or tested on its own.

**Goal:** a new standalone package with three independent classes:

| class | answers |
|---|---|
| **Parameter** (`FloatParameter`, `IntParameter`, `CategoricalParameter`); a search space is a plain list of them | Which values may each hyperparameter take? |
| **`CrossValidationObjective`** | How is one trial scored? The parameters are set once, then every CV fold is fitted and scored, and the fold scores are combined into one number. |
| **`HyperparameterStudy`** | How does Optuna search, and what was the best result? |

**Out of scope:** connecting this to the models, and changing the old
`tuning.py`. This follows the same approach as the `splitting` package.

---

## 2. Key decisions

| # | decision | why | details |
|---|---|---|---|
| D1 | One frozen **pydantic dataclass** per parameter type, under an abstract `Parameter` base. Types are checked strictly by pydantic; value rules are delegated to Optuna's own distribution classes. | Each class holds only the options its type accepts. Pydantic rejects wrong types, which plain dataclasses let through (§4.1.1). Optuna already checks almost every value rule, so we add only the three it lets through. There is no dict/TOML loading in v1: the existing TOML table stores only `[low, high]` pairs without `log`, so a migration would need a new format anyway. | §4.1 |
| D2 | The model is an sklearn estimator or `Pipeline` used as a template: `clone(template).set_params(**params)` | This is the sklearn standard. Preprocessing inside a `Pipeline` is refitted on every fold, so validation rows can't leak into it. | §4.2 |
| D3 | The folds are built once, when the objective is created | Every trial is then compared on identical folds. | §4.2 |
| D4 | The model is fitted inside the objective; everything that doesn't depend on the trial's parameters happens outside it | The parameters change on every trial. The folds (built before the study) and the final refit (after it) do not. | §4.2 |
| D5 | Fold aggregation is a setting: `"weighted_mean"` (default), `"mean"` or `"lower_bound"` | The size-weighted mean is the pooled per-row loss, the same measure the test set uses, and the least noisy combination. The other two are available on request. | §4.2.1 |
| D6 | Scoring uses sklearn scorers, where greater is better, and the study always maximizes | One sign convention, so there's no sign to get wrong. | §4.2 |
| D7 | The study takes Optuna sampler and pruner objects directly, with seeded defaults | Wrapping them would duplicate Optuna's API. | §4.3 |
| D8 | Only COMPLETE trials can win | A pruned trial's value comes from fewer folds. | §4.3 |
| D9 | No conditional or derived parameters in v1 | The LightGBM example (§5) didn't need them. | §5 |

---

## 3. Package layout

The layout mirrors `splitting/` and `feature_engineering/`:

```
src/age_group_prediction/hyperparameter_tuning/
    __init__.py      # docstring (how the pieces fit) and the public API
    parameters.py    # ParamValue, Parameter, FloatParameter, IntParameter, CategoricalParameter
    objective.py     # CrossValidationObjective (fold loop, aggregation, corrected SE)
    study.py         # HyperparameterStudy, TuningResult, TrialRecord

tests/unit/
    test_hyperparameter_tuning_parameters.py
    test_hyperparameter_tuning_objective.py
    test_hyperparameter_tuning_study.py
```

- **Imports go one way:** `objective.py` imports `parameters.py`, and
  `study.py` imports nothing from the package. The study accepts any
  `trial -> float` callable.
- **Dependencies:** optuna, scikit-learn, numpy and pydantic (already a project
  dependency, used by `feature_engineering`). No imports from
  `tuning.py`, `modeling_config.py` or `data_splitting.py`.
- **Not re-exported** from `age_group_prediction/__init__.py`, the same as
  `splitting` and `feature_engineering`.
- **Type-checked strictly:** added to `[tool.mypy].files` in `pyproject.toml`.

---

## 4. Design

### 4.1 Parameters (`parameters.py`)

Five names, each doing one thing:

```python
from pydantic.dataclasses import dataclass as pydantic_dataclass

type ParamValue = None | bool | int | float | str      # what Optuna can store
_STRICT = ConfigDict(strict=True, allow_inf_nan=False)  # no silent conversion; no NaN/inf

@pydantic_dataclass(frozen=True, config=_STRICT)
class Parameter(ABC):
    """One hyperparameter: its set_params name and the values it may take."""
    name: str                                           # e.g. "model__learning_rate"; inherited
    def __post_init__(self) -> None: ...                # builds the Optuna distribution (value rules)
    @abstractmethod
    def suggest(self, trial: BaseTrial) -> ParamValue: ...

@pydantic_dataclass(frozen=True, config=_STRICT)
class FloatParameter(Parameter):                        # -> trial.suggest_float
    low: float; high: float
    _: KW_ONLY                                          # log and step keyword-only, as in Optuna
    log: bool = False
    step: float | None = None

@pydantic_dataclass(frozen=True, config=_STRICT)
class IntParameter(Parameter):                          # -> trial.suggest_int
    low: int; high: int
    _: KW_ONLY
    log: bool = False
    step: int = 1

@pydantic_dataclass(frozen=True, config=_STRICT)
class CategoricalParameter(Parameter):                  # -> trial.suggest_categorical
    # A list is accepted as a tuple; numpy integers and bools are rejected.
    choices: Annotated[tuple[ParamValue, ...], BeforeValidator(_as_choice_tuple)]
```

**A search space is a plain list of parameters**, not a class of its own:
`[IntParameter("model__max_depth", 3, 8), FloatParameter(...)]`. A
`SearchSpace` class was built in 1.5 and then removed (§4.1.2).

The arguments mirror Optuna's own calls. The name, low and high are
positional; `log` and `step` are keyword-only, the same as in
`trial.suggest_float(name, low, high, *, step, log)`:
`FloatParameter("model__learning_rate", 0.01, 0.2, log=True)`.

**Validation happens in two layers when the object is built:**
1. **Types (pydantic, strict):** for example `IntParameter("p", 1.5, 9)`,
   `IntParameter("p", True, 9)`, `log="yes"` and `name=None` are rejected. An
   int is accepted where a float is expected, as Optuna itself allows.
2. **Values (Optuna):** see below.

Both layers raise `pydantic.ValidationError`, which is a subclass of
`ValueError`. A value error names the parameter (`parameter 'model__alpha': ...`).
A type error names the class and the argument: its position if passed
positionally, e.g. `1 validation error for IntParameter` / `1` for `low`.

**Value rules.** Measured against Optuna 4.9, Optuna's
distribution classes already raise a clear `ValueError` for every basic rule:
`low > high`, `log` with `step`, `log` with `low <= 0` (float) or `low < 1`
(int), `step <= 0`, an int `log` with `step != 1`, and empty choices. So each
parameter's `__post_init__` builds its Optuna distribution (`FloatDistribution`,
`IntDistribution`, `CategoricalDistribution`), and those errors appear when
the parameter is created instead of at the first trial. The rules aren't
written a second time.

Only three cases pass Optuna silently. These are the only rules this package
adds:

| case | what Optuna does | here |
|---|---|---|
| the step doesn't divide `high - low` (e.g. `0..1` with step `0.3`) | a warning, then it **quietly lowers `high`** to 0.9 | error. Optuna's warnings are escalated to errors while the distribution is built, which also covers non-scalar categorical choices. |
| duplicate categorical choices | accepted silently, so the sampler gives that choice double weight | error |
| two parameters with the same name | the second call returns the first value, or a warning if the ranges differ | error in `CrossValidationObjective`'s constructor |

The objective also rejects an empty list.

**Deliberately left out** (can be added without breaking anything):
- **the `kind` field, a type union and dict/TOML loading:** nothing loads a
  search space from a file yet (see D1).
- **a "fixed" parameter type:** fixed values go on the estimator template (§5).
- **conditional or derived ranges:** see D9.

#### 4.1.1 Why pydantic dataclasses, not plain dataclasses or `BaseModel`

I measured this against the Phase 1 code as first written with plain dataclasses:

| input | plain `dataclass` | pydantic `dataclass` (strict) |
|---|---|---|
| `IntParameter("p", 1.5, 9)` | **accepted**: stores `low=1.5` | rejected: "Input should be a valid integer" |
| `IntParameter("p", True, 9)` | **accepted** | rejected |
| `FloatParameter("p", 0.1, 1.0, log="yes")` | **accepted**: `log="yes"` | rejected: "Input should be a valid boolean" |
| `FloatParameter(None, 0.1, 1.0)` | **accepted**: `name=None` | rejected |
| `CategoricalParameter("p", ["a", "b"])` | **accepted as a list, and the list can still be changed after creation**, so the "frozen" object isn't frozen | converted to the tuple `("a", "b")` |
| `FloatParameter("p", 0.1, 1.0, log=True)` (positional, like Optuna) | ✓ | ✓ |
| `KW_ONLY`, `frozen`, the abstract base, `__post_init__` | ✓ | ✓ (all checked) |
| mypy `--strict` | ✓ | ✓ (checked) |

| option | verdict |
|---|---|
| plain `dataclasses.dataclass` | Rejected: wrong types pass silently (table above). |
| **`pydantic.dataclasses.dataclass`** (chosen) | The same syntax and positional calls as a dataclass, plus type validation, which is what the table shows. The change is one import and a `config=`. |
| `pydantic.BaseModel` (as in `feature_engineering`) | Rejected: it only accepts keyword arguments (`FloatParameter(name=..., low=..., high=...)`), so the calls would no longer mirror Optuna's `suggest_float(name, low, high)`. `BaseModel`'s extra features (`model_dump`, `model_validate`) aren't needed in v1, and `TypeAdapter` gives the same for pydantic dataclasses if loading is ever added. |

Strict mode is the right default: lax mode would silently turn `"0.1"` into
`0.1` and `1` into `True`. The single exception is `choices`, where accepting
a list is what users expect.

**The rule used across this package** (best practice):

| tool | use it for | here |
|---|---|---|
| `pydantic.BaseModel` | data that crosses a boundary (config files, JSON, APIs): parsed, validated and serialized | not needed in v1; `feature_engineering`'s specs are this case |
| `pydantic.dataclasses.dataclass` | small value objects that code builds, needing validation plus stdlib dataclass behavior (positional arguments, `KW_ONLY`, `dataclasses.asdict/replace`) | **the parameter classes**, the one place where user-typed values enter |
| stdlib `dataclasses.dataclass` | internal data whose types the code already guarantees | **`TrialRecord` and `TuningResult`** (Phase 3), which are built from Optuna's own records |

This choice doesn't block loading from a file later:
`TypeAdapter(FloatParameter).validate_python({...})` works on pydantic
dataclasses.

**Readability:** the decorator is imported as
`from pydantic.dataclasses import dataclass as pydantic_dataclass`, so every
class shows it's pydantic (`@pydantic_dataclass(frozen=True, config=_STRICT)`).
Imported under its bare name, it looked exactly like the standard library.

#### 4.1.2 Why a list, not a `SearchSpace` class

| what `SearchSpace` did | needs a class? |
|---|---|
| `suggest(trial)` → `{name: value}` | no: one dict comprehension, in the objective |
| reject an empty space | no: one check |
| check each item is a `Parameter` | no: mypy checks it statically |
| **reject duplicate names** | the only real content; runs once in the objective's constructor, where the parameters are used |

It had one consumer, and it added a type users must learn and wrap lists in.
If a second consumer ever needs the same checks, a module-level function can
hold them.

### 4.2 Objective (`objective.py`)

```python
class CrossValidationObjective:
    def __init__(self, estimator: BaseEstimator, parameters: Sequence[Parameter],
                 X, y, *, cv: BaseCrossValidator, scoring: str | Scorer, groups=None,
                 aggregation: Literal["weighted_mean", "mean", "lower_bound"] = "weighted_mean",
                 z: float | None = None): ...     # "lower_bound" only; None -> 1.0
        # folds = list(cv.split(X, y, groups))   -- once (D3)
        # rejects: no parameters; duplicate parameter names (Optuna would silently
        #          reuse the first one's value); zero folds; an empty validation fold;
        #          "lower_bound" with one fold;
        #          z <= 0; z given with any other aggregation

    def __call__(self, trial: optuna.Trial) -> float:
        # params = {p.name: p.suggest(trial) for p in parameters} -- once, before any fold
        # for each fold i:
        #     model = build_estimator(params).fit(fit rows)
        #     score = scorer(model, validation rows)              -- NaN or inf -> the trial fails
        #     trial.report(running mean, step=i); prune if the pruner says so
        # record user attrs: fold_scores, fold_sizes, std_error
        # return the aggregated value (§4.2.1)

    def build_estimator(self, params) -> BaseEstimator:
        # clone(template).set_params(**params); also used for the final refit
```

- Rows are selected by position: `.iloc` for pandas, plain indexing for numpy.
- A holdout is just a cross-validator with one split
  (`ShuffleSplit(n_splits=1)` or `StratifiedHoldout`).

#### 4.2.1 How the fold scores are combined

| `aggregation=` | trial value | when to use it |
|---|---|---|
| `"weighted_mean"` (**default**) | `Σ n_k · score_k / Σ n_k` | Almost always. It is the pooled per-row score, the same measure the test set reports. When folds differ in size (e.g. under `grouped`) it is the least noisy combination. On equal folds it equals `"mean"`. |
| `"mean"` | `mean(score_k)` | To match sklearn's `cross_val_score(...).mean()` or the models' current tuning |
| `"lower_bound"` | `weighted_mean − z · SE` (default `z=1.0`, needs at least 2 folds) | Only when you explicitly want stable settings over the best average. The ranking gets noisier: the SE is estimated from K numbers, and most of the spread between folds is difficulty every trial shares. |

- **SE:** the Nadeau–Bengio corrected formula, `s · sqrt(1/K + n_val/n_fit)`.
  The naive `s/√K` is too small, because the folds share most of their fit
  rows. Every trial records its SE, whatever the aggregation setting.
- **Pruning** always uses a running mean (weighted, or plain for `"mean"`). An
  SE from 2-3 folds is too unstable to prune on.
- **Rejected weightings:** by each fold's own variance (this leans toward easy
  folds), and by neighborhoods per fold (this measures loss per neighborhood,
  a different target from the test-set metric).
- **Caveat:** for a metric that isn't an average over rows (RMSE, R²), size
  weights still reduce noise but don't equal the pooled metric exactly.

### 4.3 Study (`study.py`)

```python
class HyperparameterStudy:
    def __init__(self, *, seed: int, n_trials: int | None = 50,
                 timeout_seconds: float | None = None, n_jobs: int = 1,
                 sampler: BaseSampler | None = None,     # None -> TPESampler(seed=seed, multivariate=True)
                 pruner: BasePruner | None = None,       # None -> NopPruner()
                 show_progress_bar: bool = False, study_name: str | None = None,
                 storage: str | None = None,             # e.g. "sqlite:///tuning.db" to persist or resume
                 initial_params: Sequence[dict[str, ParamValue]] = ()): ...  # run first
        # rejects: n_trials and timeout_seconds both None; n_trials < 1; n_jobs == 0

    def optimize(self, objective: Callable[[optuna.Trial], float]) -> TuningResult:
        # create_study(direction="maximize", ...); enqueue initial_params; study.optimize(...)
        # best = max over COMPLETE trials; ties -> lowest trial number
        # RuntimeError if no trial completed
        # self.optuna_study_ is kept for optuna.visualization

@dataclass(frozen=True)
class TrialRecord:  number, state, value, params, fold_scores, fold_sizes, std_error

@dataclass(frozen=True)
class TuningResult: best_params, best_value, best_trial_number, trials, is_reproducible
    def to_dict(self) -> dict        # JSON-safe, e.g. for MLflow logging
```

`is_reproducible` is `n_jobs == 1 and timeout_seconds is None`. Timeouts and
parallel jobs are allowed; the result records which kind of run it was.

---

## 5. Worked example: `DirectCohortModel`'s LightGBM tuning

```python
parameters = [                              # same bounds as [direct_cohort_search_space]
    IntParameter("model__max_depth", 3, 8),
    IntParameter("model__num_leaves", 7, 63),
    IntParameter("model__min_child_samples", 5, 40),
    FloatParameter("model__learning_rate", 0.01, 0.2, log=True),
    IntParameter("model__n_estimators", 50, 400),
    FloatParameter("model__reg_alpha", 1e-8, 10.0, log=True),
    FloatParameter("model__reg_lambda", 1e-8, 10.0, log=True),
    FloatParameter("model__min_split_gain", 0.0, 1.0),
    FloatParameter("model__subsample", 0.7, 1.0),
    FloatParameter("model__colsample_bytree", 0.7, 1.0),
]
pipeline = Pipeline([                         # fixed settings live on the template
    ("features", FeatureTransformer(...)),
    ("model", LGBMRegressor(objective="poisson", subsample_freq=1, random_state=seed,
                            n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)),
])

splitter = Splitter("stratified_by_group")
X_train, X_test, y_train, y_test, g_train, g_test = splitter.train_test_split(
    X, Y, groups, test_size=0.2, random_state=42)

for cohort in cohort_columns:                 # one study per cohort, as today
    objective = CrossValidationObjective(
        pipeline, parameters, X_train, y_train[cohort], groups=g_train,
        cv=splitter.cv(n_splits=5, random_state=42), scoring="neg_mean_poisson_deviance")
    result = HyperparameterStudy(seed=seeds[cohort], n_trials=30).optimize(objective)
    model = objective.build_estimator(result.best_params).fit(X_train, y_train[cohort])
```

**What the example showed:**

| in `direct_cohort.py` today | in the new design | outcome |
|---|---|---|
| 11 `trial.suggest_*` calls | 10 `Parameter`s, with the same bounds as the TOML table | covered |
| `num_leaves <= 2**max_depth`, a range that depends on another parameter | Both are sampled independently. LightGBM already caps the leaves by depth. A range that changes between trials is an Optuna "dynamic search space", which multivariate TPE can't model jointly with the other parameters. | no conditional component (D9); slightly changes how the search behaves |
| `subsample_freq` derived from `subsample` | `subsample_freq=1` fixed. A bagging fraction of 1.0 means no bagging, so the result is the same. | no derived-parameter component (D9) |
| fixed LightGBM settings | on the template | no "fixed parameter" class |
| feature transformer cached per fold | a `Pipeline` step, refitted per fold | no cache component; the cost is trivial at about 1.2k rows |
| mean Poisson NLL | `neg_mean_poisson_deviance`, which ranks trials in the same order | sklearn scorer (D6) |
| a different seed for each trial and fold | the template's `random_state` | simpler, and trial comparisons are less noisy |
| NB2 `dispersion` (used by the objective **and** the score) | can't be set through `set_params` | **known gap:** it needs an sklearn wrapper when the model is migrated |
| `_cross_fitted_normal_scale` needs out-of-fold predictions | not part of tuning | possible later addition: `objective.out_of_fold_predict(params)` |

---

## 6. Implementation phases and sub-tasks

Each sub-task has a checkbox and a **done when** line. At the end of each phase:
an independent review, fix its findings, run the full suite, then **stop for
your approval**. With your approval, I commit and push the phase to the draft
PR.

### Phase 0: setup
- [x] **0.1 Plan doc.** Copy this plan to `docs/HYPERPARAMETER_TUNING_PLAN.md`.
  *Done when:* the doc matches this plan.
- [x] **0.2 Draft PR.** (PR #5) *(Needs your approval: it is the first commit.)* Commit
  the plan doc, push, and run `gh pr create --draft` with the text in §8.
  *Done when:* the draft PR URL is shared with you.

**Stop: you approve the plan and the PR.**

### Phase 1: parameters (`parameters.py`)
Every class test uses the same method. Ask a real study for a trial, call
`suggest`, then compare `trial.distributions[name]` with the expected Optuna
distribution. That one comparison proves every option (`low`, `high`, `log`,
`step`, `choices`) reached Optuna unchanged.

- [x] **1.1 Package skeleton.** Create `hyperparameter_tuning/__init__.py` (the
  docstring and an empty `__all__`) and add the package to `[tool.mypy].files`.
  *Done when:* `import age_group_prediction.hyperparameter_tuning` works and
  `uv run mypy` passes.
- [x] **1.2 `Parameter` base and `FloatParameter`.** *Done when:*
  - plain, `log=True` and `step=` settings each reach Optuna exactly
  - the value drawn lies in `[low, high]`
  - an invalid setting raises when the parameter is created, not at the first
    trial (Optuna's own error). One representative case: `log=True` with `low=0`.
  - a step that doesn't divide the range raises (this package's rule)
  - the object is frozen: assigning an attribute raises
- [x] **1.3 `IntParameter`.** *Done when:* the same checks as 1.2 pass, and the
  drawn value is an `int`.
- [x] **1.4 `CategoricalParameter`.** *Done when:*
  - the choices reach Optuna in order
  - a mix of `None`, `bool`, `int`, `float` and `str` is accepted
  - duplicate choices raise (this package's rule)
  - a non-scalar choice such as `[1]` raises (since 1.4b, pydantic's type check
    catches it before Optuna's warning)
- [x] **1.4b Switch to pydantic dataclasses** (§4.1.1). In `parameters.py`,
  import `dataclass` from `pydantic.dataclasses`, add `config=_STRICT` to all
  four classes, and type `choices` as
  `Annotated[tuple[ParamValue, ...], Strict(False)]`. Nothing else in the
  classes changes. In the plan doc, update D1, the dependencies and §4.1 to
  match this plan. *Done when:*
  - all 19 existing tests still pass. One test changes: passing `log`
    positionally now raises pydantic's `ValidationError` (a `ValueError`)
    instead of `TypeError`, so `test_log_and_step_are_keyword_only` expects
    `ValueError`. A non-scalar choice is now caught by pydantic's type check
    first; that test's `match` changes accordingly.
  - new, parametrized test: the wrong types from §4.1.1's table are rejected
    (`1.5` and `True` for an int, `log="yes"`, `name=None`)
  - new test: a list of choices is stored as a tuple, and the parameter can't
    be changed afterwards
  - mypy `--strict` and ruff pass
- [x] **1.4c Make pydantic visible.** Import the decorator as
  `pydantic_dataclass` and use `@pydantic_dataclass(frozen=True, config=_STRICT)`
  on all four classes. Copy the "rule used across this package" table (§4.1.1)
  into the plan doc. *Done when:* no bare `@dataclass` is left in
  `parameters.py`, and the 26 tests, mypy and ruff pass. There is no
  behavior change.
- [x] **1.5 `SearchSpace` and exports.** Built and tested, then **superseded
  by 1.5b**: a class was not needed (§4.1.2).
- [x] **1.5b Replace `SearchSpace` with a plain list.** Delete the class and its
  export; the duplicate-name and empty checks move to Phase 2.1 (the objective's
  constructor). *Done when:*
  - `parameters.py` has five public names: `ParamValue`, `Parameter`,
    `FloatParameter`, `IntParameter`, `CategoricalParameter`; `__all__` in the
    package lists the same five
  - the `SearchSpace` tests are removed; the LightGBM test keeps checking that
    the §5 parameter list builds and that a 5-trial study drawing every
    parameter runs
  - ruff, mypy and the full suite pass (the four `SearchSpace` tests go; after
    the review fixes below, 880 on `main` + 37 = 917)

**Phase 1 independent review** (after 1.5b). Each finding was reproduced before
it was fixed:

| # | finding | verdict | fix |
|---|---|---|---|
| 1 | `choices` silently turned `np.int64(7)` into `7.0` and `np.True_` into `1.0` | **bug**, confirmed | `choices` now uses a `BeforeValidator` that turns a list into a tuple and rejects numpy integers and bools. The reviewer's suggestion (strict items) was tested and **did not** fix it: pydantic's float validator still converts them. |
| 2 | NaN or inf bounds passed construction and failed at the first trial with `OverflowError` / `decimal.InvalidOperation` | **bug**, confirmed | `ConfigDict(strict=True, allow_inf_nan=False)` |
| 3 | two distinct NaN choices passed the duplicate check | covered by fix 2 | none needed |
| 4 | warning escalation swallowed unrelated warnings (e.g. a `DeprecationWarning`) | should-fix, confirmed | `simplefilter("error", UserWarning)` only; other warnings pass through |
| 5 | numpy integer bounds are rejected by `IntParameter` | documented | module docstring: pass Python numbers (`int(x)`) |
| 6 | no test that a float step which divides the range is accepted | added | `0.1..1.0/0.1` and `0.7..1.0/0.05` |
| 7 | package docstring names modules not yet written | accepted | resolved in Phases 2 and 3 |
| 8 | `[direct_cohort_search_space]` is "not in any config" | **rejected** | it is at `configs/modeling.toml:53` |

New tests: 9, so 37 parameter tests in all. Final full suite: **917 passed**,
0 failed; the 15 warnings all come from existing tests.

**Stop: review and your approval of Phase 1.**

### Phase 2: objective (`objective.py`)
- [ ] **2.1 Constructor.** Build the folds once, compute the fold sizes and
  weights, set up the scorer, and add `build_estimator`.
  *Done when:* tests show that `KFold(shuffle=True, random_state=None)` gives
  identical folds on repeated calls; that an empty parameter list, duplicate
  parameter names, zero folds, an empty validation fold
  and every bad `aggregation`/`z` combination are rejected; and that
  `build_estimator` returns an unfitted clone with the given parameters and
  leaves the template unchanged.
- [ ] **2.2 Fold loop.** Suggest once, then fit and score every fold, then
  report and prune.
  *Done when:* tests show that the parameters are suggested once and every fold
  sees the same values (a test estimator records them); that a `Pipeline`
  transformer is fitted only on fit rows; that a NaN score raises; and that
  `trial.report` gets one value per fold.
- [ ] **2.3 Aggregation and SE.**
  *Done when:* on deliberately unequal folds, each of the three options matches
  a hand-calculated value; `"weighted_mean"` equals `"mean"` on equal folds
  and equals the pooled score for a mean-per-row scorer; the recorded SE
  matches the corrected formula.
- [ ] **2.4 Split methods.** *Done when:* the objective runs with
  `Splitter("random" | "stratified_by_group" | "grouped").cv(...)` plus `groups`,
  and with a one-split holdout.

**Stop: review and your approval of Phase 2.**

### Phase 3: study (`study.py`)
- [ ] **3.1 Constructor and defaults.** *Done when:* the default sampler is a
  seeded multivariate TPE and the default pruner is `NopPruner`; invalid
  `n_trials`, timeout or `n_jobs` combinations are rejected.
- [ ] **3.2 `optimize` and selecting the best trial.** *Done when:* tests show
  that the same seed gives an identical `TuningResult`; a pruned trial never
  wins; ties go to the lowest trial number; a study with no completed trial
  raises `RuntimeError`; `initial_params` run first.
- [ ] **3.3 Records and results.** *Done when:* each `TrialRecord` carries the
  fold scores, sizes and SE from the objective; `to_dict()` round-trips through
  `json`; `is_reproducible` is False when a timeout is set or `n_jobs > 1`.
- [ ] **3.4 LightGBM integration test** (the §5 example) on a small simulated
  table, for a few trials. *Done when:* the best parameters refit and predict.

**Stop: review and your approval of Phase 3.**

### Phase 4: docs
- [ ] **4.1 `docs/HYPERPARAMETER_TUNING.md`:** the three classes, the decisions
  (§2), aggregation guidance (§4.2.1), the worked example and what it showed,
  hazards, and what remains (wrapping the models, migrating them, NB2
  dispersion, out-of-fold predictions).
- [ ] **4.2 Links** from `docs/README.md`, `docs/MODULE_REFERENCE.md` and
  `docs/SPLITTING.md` §7. Mark the plan doc as implemented.
- [ ] **4.3 Final check:** the full verification in §7, and the PR checklist is updated.

**Stop: review and your approval. Then the PR can move from draft to ready.**

---

## 7. Verification

- `uv run pytest tests/unit/test_hyperparameter_tuning_*.py -q`
- `uv run pytest -q`: the 880 tests on `main` pass, plus the new tests, with no warnings from the new tests (the full run takes about 10 minutes)
- `uv run mypy` and `uv run ruff check src tests`
- `git status` shows only the package, its tests, `pyproject.toml` and the docs changed

---

## 8. Draft PR

**Title:** `feat: add modular Optuna hyperparameter tuning`

```markdown
## Summary
Adds the `hyperparameter_tuning` package: Optuna tuning in three independent classes.

- **Parameters** (`parameters.py`): `FloatParameter`, `IntParameter` and
  `CategoricalParameter` cover every option of Optuna's `suggest_*` calls and
  are validated when built, using Optuna's own checks plus the three cases
  Optuna lets through silently. A search space is a plain list of them.
- **`CrossValidationObjective`** (`objective.py`): builds the folds once and
  suggests one parameter set per trial. It fits and scores every fold and
  combines the fold scores as set by `aggregation`: the fold-size-weighted mean
  (default), the plain mean, or a lower confidence bound. It records the fold
  scores, fold sizes and a corrected standard error.
- **`HyperparameterStudy`** (`study.py`): creates and runs the study (sampler,
  pruner, trials, timeout, storage, initial trials) and records the best result
  and every trial. Only completed trials can win.

Tested end to end on the `DirectCohortModel` LightGBM search space.

## Scope
Not connected to any model yet. `tuning.py` and the models' built-in
objectives are unchanged. The design and its decisions are in
`docs/HYPERPARAMETER_TUNING_PLAN.md`.

## Checklist
- [ ] Phase 1: parameters
- [ ] Phase 2: objective
- [ ] Phase 3: study, including the LightGBM integration test
- [ ] Phase 4: docs
- [ ] Full suite passes: the 880 tests on `main` plus the new ones

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
