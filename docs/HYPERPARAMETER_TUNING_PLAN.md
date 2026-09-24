# Plan: modular Optuna hyperparameter tuning

**Branch:** `feat/hyperparameter-tuning` · **Status:** plan approved · Phase 0.1 done · no code written yet
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
| **Parameter** (`FloatParameter`, `IntParameter`, `CategoricalParameter`) + `SearchSpace` | Which values may each hyperparameter take? |
| **`CrossValidationObjective`** | How is one trial scored? The parameters are set once, then every CV fold is fitted and scored, and the fold scores are combined into one number. |
| **`HyperparameterStudy`** | How does Optuna search, and what was the best result? |

**Out of scope:** connecting this to the models, and changing the old
`tuning.py`. This follows the same approach as the `splitting` package.

---

## 2. Key decisions

| # | decision | why | details |
|---|---|---|---|
| D1 | One class per parameter type, built as frozen pydantic models with a `kind` field that tells them apart | Each class holds only the options its type accepts, so a bad combination fails when the object is built. Specs can be loaded from TOML. Same pattern as `feature_engineering/transforms.py`. | §4.1 |
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
    parameters.py    # FloatParameter, IntParameter, CategoricalParameter, Parameter, SearchSpace
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
- **Dependencies:** optuna, scikit-learn, numpy and pydantic. No imports from
  `tuning.py`, `modeling_config.py` or `data_splitting.py`.
- **Not re-exported** from `age_group_prediction/__init__.py`, the same as
  `splitting` and `feature_engineering`.
- **Type-checked strictly:** added to `[tool.mypy].files` in `pyproject.toml`.

---

## 4. Design

### 4.1 Parameters (`parameters.py`)

```python
class _ParameterBase(BaseModel):              # frozen, extra="forbid"
    name: str                                 # the set_params key, e.g. "model__learning_rate"
    def suggest(self, trial: optuna.Trial) -> ParamValue: ...

class FloatParameter(_ParameterBase):         # -> trial.suggest_float(name, low, high, step=, log=)
    kind: Literal["float"] = "float"
    low: float; high: float; log: bool = False; step: float | None = None

class IntParameter(_ParameterBase):           # -> trial.suggest_int(name, low, high, step=, log=)
    kind: Literal["int"] = "int"
    low: int; high: int; log: bool = False; step: int = 1

class CategoricalParameter(_ParameterBase):   # -> trial.suggest_categorical(name, choices)
    kind: Literal["categorical"] = "categorical"
    choices: tuple[None | bool | int | float | str, ...]

Parameter = Annotated[FloatParameter | IntParameter | CategoricalParameter,
                      Field(discriminator="kind")]

class SearchSpace(BaseModel):
    parameters: tuple[Parameter, ...]
    def suggest(self, trial) -> dict[str, ParamValue]: ...
```

**Validation when the object is built.** These rules match Optuna's own but
fail earlier, with a clearer message:

| type | rules |
|---|---|
| all | `name` is not empty; `low <= high` |
| float | `log` and `step` cannot both be set; `log` needs `low > 0`; `step > 0`; `high - low` must be a multiple of `step` (otherwise Optuna quietly lowers `high`) |
| int | `step >= 1`; `log` needs `step == 1` and `low >= 1`; `high - low` must be a multiple of `step` |
| categorical | at least one choice; choices are unique |
| SearchSpace | at least one parameter; no duplicate names |

### 4.2 Objective (`objective.py`)

```python
class CrossValidationObjective:
    def __init__(self, estimator: BaseEstimator, search_space: SearchSpace,
                 X, y, *, cv: BaseCrossValidator, scoring: str | Scorer, groups=None,
                 aggregation: Literal["weighted_mean", "mean", "lower_bound"] = "weighted_mean",
                 z: float | None = None): ...     # "lower_bound" only; None -> 1.0
        # folds = list(cv.split(X, y, groups))   -- once (D3)
        # rejects: zero folds; an empty validation fold; "lower_bound" with one fold;
        #          z <= 0; z given with any other aggregation

    def __call__(self, trial: optuna.Trial) -> float:
        # params = search_space.suggest(trial)                   -- once, before any fold
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
space = SearchSpace(parameters=(            # same bounds as [direct_cohort_search_space]
    IntParameter(name="model__max_depth", low=3, high=8),
    IntParameter(name="model__num_leaves", low=7, high=63),
    IntParameter(name="model__min_child_samples", low=5, high=40),
    FloatParameter(name="model__learning_rate", low=0.01, high=0.2, log=True),
    IntParameter(name="model__n_estimators", low=50, high=400),
    FloatParameter(name="model__reg_alpha", low=1e-8, high=10.0, log=True),
    FloatParameter(name="model__reg_lambda", low=1e-8, high=10.0, log=True),
    FloatParameter(name="model__min_split_gain", low=0.0, high=1.0),
    FloatParameter(name="model__subsample", low=0.7, high=1.0),
    FloatParameter(name="model__colsample_bytree", low=0.7, high=1.0),
))
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
        pipeline, space, X_train, y_train[cohort], groups=g_train,
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
- [ ] **0.2 Draft PR.** *(Needs your approval: it is the first commit.)* Commit
  the plan doc, push, and run `gh pr create --draft` with the text in §8.
  *Done when:* the draft PR URL is shared with you.

**Stop: you approve the plan and the PR.**

### Phase 1: parameters (`parameters.py`)
- [ ] **1.1 Package skeleton.** Create `hyperparameter_tuning/__init__.py` (the
  docstring and an empty `__all__`) and add the package to `[tool.mypy].files`.
  *Done when:* `import age_group_prediction.hyperparameter_tuning` works and
  `uv run mypy` passes.
- [ ] **1.2 `FloatParameter`.** *Done when:* tests show that `suggest` passes
  `low`, `high`, `log` and `step` through to `trial.suggest_float` (checked
  with `FixedTrial` and a real study), and that every float rule in §4.1
  rejects bad input.
- [ ] **1.3 `IntParameter`.** *Done when:* the same checks pass for `suggest_int`
  and the int rules.
- [ ] **1.4 `CategoricalParameter`.** *Done when:* `suggest_categorical` is
  called with the choices; empty or duplicate choices are rejected; `None`,
  `bool`, `int`, `float` and `str` choices are accepted.
- [ ] **1.5 `SearchSpace` and the `Parameter` union.** *Done when:* it builds
  from a list of dicts; an unknown `kind` names the valid options in the
  error; duplicate names and an empty space are rejected; `suggest` returns
  one value per parameter; the §5 LightGBM space builds.
- [ ] **1.6 Exports and docstrings.** Fill in `__all__`.
  *Done when:* ruff and mypy pass, and the suite passes (696 plus the new tests).

**Stop: review and your approval of Phase 1.**

### Phase 2: objective (`objective.py`)
- [ ] **2.1 Constructor.** Build the folds once, compute the fold sizes and
  weights, set up the scorer, and add `build_estimator`.
  *Done when:* tests show that `KFold(shuffle=True, random_state=None)` gives
  identical folds on repeated calls; that zero folds, an empty validation fold
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
- `uv run pytest -q`: the 696 baseline tests pass, plus the new tests, with no new warnings
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
  are validated when built. `SearchSpace` groups them and can be loaded from TOML.
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
- [ ] Full suite passes: the 696 existing tests plus the new ones

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
