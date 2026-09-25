# Plan: modular Optuna hyperparameter tuning

**Branch:** `feat/hyperparameter-tuning` · draft PR #5.
**Status:** Phases 1 (parameters) and 2 (evaluator, aggregations) are done;
2.3c is committed (`e756667`), and 2.4 with the Phase 2 review fixes awaits
commit. Non-slow suite: 1002 passed, 8 warnings. **Next: Phase 3** (study),
then Phase 4 (docs).
The history of each step is in the git log and the commit messages.
**To resume:** read this doc (§2 decisions, §4 design, §6 remaining work, §7
working notes), then the package code and its tests; start the next sub-task
in plan mode.

## 1. Goal and scope

Every Optuna objective in the old stack is a closure inside a model
(`models/direct_cohort.py`, `models/independent_total_probability.py`) that
hard-codes the search space, the study settings and the fold loop. This
package splits them into independent, tested parts:

| part | answers |
|---|---|
| `Parameter`s (a search space is a plain list of them) | which values may each hyperparameter take? |
| `CVHyperparameterEvaluator` + an `Aggregation` | how is one trial scored? |
| `HyperparameterStudy` | how does Optuna search, and which trial won? |

The package tunes the repo's own `modeling.BaseAgeGroupModel`s. **Out of
scope:** the old `tuning.py` and `models/` (unchanged), and wiring into the
experiment runner.

## 2. Decisions

| # | decision | why |
|---|---|---|
| D1 | One frozen, strict pydantic dataclass per parameter type; Optuna's own distribution classes check the values | pydantic rejects wrong types that a plain dataclass stores; Optuna already checks almost every value rule (§4.1) |
| D3 | Every trial splits with a `Splitter.cv` validator, which requires an int seed | trials are compared on identical folds; `Splitter.cv` guarantees at least 2 non-empty folds |
| D4 | Only the trial's parameters are fitted inside the objective | the folds and the final refit don't depend on them |
| D5 | Fold aggregation is an `Aggregation` object: `WeightedMean()` (default), `Mean()`, `LowerBound(z)` or a custom subclass, with one method `aggregate(scores, fold_sizes)` | it takes plain lists, so it knows nothing about the data; each class carries its own settings (`z`) |
| D7 | The study takes Optuna sampler and pruner objects, with seeded defaults | wrapping them would duplicate Optuna's API |
| D8 | Only COMPLETE trials can win | a pruned trial's value comes from fewer folds |
| D9 | No conditional or derived parameters in v1 | the LightGBM example (§5) doesn't need them |
| D10 | Settings in the constructor; data (`X, y, groups, exposure`) as arguments of the named method `evaluate` | the sklearn convention; one evaluator scores many targets; Optuna's one-argument objective is a lambda |
| D11 | The evaluator tunes a `BaseAgeGroupModel`: per trial, `build_feature_transformer_and_model(params)` makes copies (`clone`, `set_params`); each fold refits them, predicts, and scores with `model.evaluate(y, y_pred, metric)` | the repo's models get `clone`/`set_params` from `BaseEstimator`; a misspelled parameter raises at `set_params` |
| D12 | One direction: the score is the metric's value if `greater_is_better`, else its negation; the study always maximizes | no sign to get wrong |
| D13 | A required `feature_transformer: FeatureTransformer`, fitted per fold on the training rows and targets | a transformer fitted on all rows would leak validation statistics; a `Pipeline` was rejected (its `fit` and `predict` name the exposure differently, and it has no `evaluate`) |
| D14 | `exposure=None` is part of `BaseAgeGroupModel.fit`/`predict`; the evaluator slices it per fold and always passes it | typed calls (mypy); the model decides whether it needs one |
| D15 | `evaluate(trial, X, y, groups=None, *, exposure=None)`; `X` is the raw table | the exposure is data (D10) |
| D16 | `Metric` and the ready-made metrics live in the top-level `scoring.py` | models and tuning both use them |
| D17 | Only `modeling.BaseAgeGroupModel`s are tunable; a contract test runs sklearn's parameter checks over every one | the old models can't be cloned; a rebuilt model gets `clone`/`set_params` if it stores its settings verbatim and validates in `fit` |

D2 (sklearn estimators) and D6 (sklearn scorers) were superseded by D11–D13.

## 3. Package layout

```
src/age_group_prediction/hyperparameter_tuning/
    __init__.py      # how the parts fit; the public API
    _config.py       # _STRICT: the pydantic config shared by parameters and aggregation
    parameters.py    # Parameter, FloatParameter, IntParameter, CategoricalParameter
    aggregation.py   # Aggregation, WeightedMean, Mean, LowerBound, corrected_std_error
    evaluator.py     # CVHyperparameterEvaluator
    study.py         # HyperparameterStudy, TrialRecord, TuningResult (Phase 3)
```

- Shared helpers: `utils.take_rows` and the `DesignMatrix`, `Target`,
  `Groups` and `Exposure` aliases.
- Imports go one way: `evaluator` → `parameters`, `aggregation`, `modeling`,
  `scoring`, `feature_engineering`; `study` imports only
  `aggregation.corrected_std_error`, and accepts any `trial -> float` objective.
- The package is not re-exported from `age_group_prediction`, and mypy checks
  it strictly.
- Tests: `tests/unit/test_hyperparameter_tuning_{parameters,aggregation,evaluator,study}.py`,
  `test_scoring.py`, `test_modeling_contract.py`.

## 4. Design

**Pydantic or dataclass** (the package's rule): a frozen pydantic dataclass
for objects that users build and whose values need checking (parameters,
aggregations, the evaluator's settings); a stdlib dataclass for data the code
builds itself (`TrialRecord`, `TuningResult`); a plain `ABC` for an interface
that holds no data (`Aggregation`).

### 4.1 Parameters (`parameters.py`)

```python
FloatParameter("learning_rate", 0.01, 0.2, log=True)   # -> trial.suggest_float
IntParameter("max_depth", 3, 8)                        # -> trial.suggest_int
CategoricalParameter("layers", [[32], [64, 32]])       # -> trial.suggest_categorical
```
- Arguments mirror Optuna: name, low and high are positional; `log` and
  `step` are keyword-only.
- Validated when created: pydantic checks the types strictly (`1.5` or
  `True` for an int is rejected); Optuna's distribution classes check the
  values. Both raise `ValueError`: a value error is prefixed
  `parameter '<name>'`; a type error names the class and the argument (its
  position if passed positionally).
- Three rules Optuna lets through, added here: a step that doesn't divide the
  range (Optuna quietly lowers `high`), duplicate choices (double weight),
  and duplicate parameter names (in the evaluator).
- Choices may be any objects and are passed unchanged. A list is stored as a
  tuple; a set is rejected, since its order varies between runs.

### 4.2 Evaluator (`evaluator.py`)

```python
@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class CVHyperparameterEvaluator:
    model: BaseAgeGroupModel
    parameters: Annotated[Sequence[Parameter], Field(min_length=1),
                          AfterValidator(_unique_names)]   # stored as a tuple
    _: KW_ONLY
    cv: BaseCrossValidator                     # from Splitter.cv(...)
    metric: InstanceOf[Metric]
    feature_transformer: FeatureTransformer
    aggregation: Aggregation = field(default_factory=WeightedMean)

    def evaluate(self, trial, X, y, groups=None, *, exposure=None) -> float: ...
    def build_feature_transformer_and_model(self, params) -> tuple[FeatureTransformer, BaseAgeGroupModel]: ...
```

Per trial: suggest the parameters once, build one copy of the transformer and
the model, then for each fold (`train_index`, `val_index`):
1. slice `X`, `y` and `exposure` by position (`take_rows`);
2. fit the transformer on the training rows and transform both parts;
3. fit the model, predict the validation rows, and score them (signed, D12);
4. report the fold's own score to the pruner, and prune if it says so.

After the last fold: record the user attrs `fold_scores` and `fold_sizes`,
and return `aggregation.aggregate(scores, fold_sizes)`.

- **Settings are validated when the evaluator is built** (pydantic). Each
  check prevents a silent or late failure (measured): a string metric failed
  after 1 fold was fitted, a string aggregation after every fold; any sklearn
  transformer ran; an empty search space gave identical trials; duplicate
  names made Optuna reuse the first value; a set of parameters would make a
  seeded study's order vary between runs. `model` (D17) and `cv` (always a
  `Splitter.cv` validator) are type-checked too. `InstanceOf[Metric]`: lax
  pydantic would otherwise build a `Metric` from a dict.
- **Data checks in `evaluate`:** an exposure that isn't one value per row of
  `X` (a longer one would be sliced to size silently); a NaN or inf fold score
  raises `ValueError` before `report` (Optuna stores NaN silently), which
  marks the trial FAIL and stops the study.
- **Copies once per trial, refits per fold:** `fit` replaces all fitted state
  (a contract test checks it), so one pair serves every fold; copies keep the
  templates unfitted and parallel trials (`n_jobs > 1`) independent.
- **Pruning** sees each fold's own score (`step` = fold number), the contract
  of Optuna's `WilcoxonPruner`, which pairs fold k across trials. Measured,
  on 10 folds of unequal difficulty with a bad trial that ties on fold 0:
  `WilcoxonPruner` pruned it after 5 folds and `SuccessiveHalvingPruner`
  after 2; `MedianPruner`/`PercentilePruner` never did, whatever was reported,
  because they compare a trial's best value over its steps (a learning-curve
  assumption).
- **User attrs** are written only for a completed trial, as plain lists (what
  sqlite storage accepts and returns). A pruned trial's scores are its
  intermediate values.
- `groups=None` for the `random` method: `KFold` warns when given groups.
- `exposure` is the raw count the model expects, not its log.
- Leakage is still possible if `X` was already fitted on all rows and a
  pass-through transformer is used.

### 4.3 Aggregation (`aggregation.py`)

| class | trial value | use it |
|---|---|---|
| `WeightedMean()` (default) | `Σ n_k·score_k / Σ n_k` | almost always; for a per-row mean metric (MAE, Poisson deviance) it is the pooled score the test set reports |
| `Mean()` | `mean(score_k)` | to match `cross_val_score(...).mean()` |
| `LowerBound(z=1.0)` | `weighted mean − z·SE` | to prefer stable settings; `z=1` is the one-SE rule, `z=1.645` roughly a one-sided 95% bound. Its ranking is noisier: the SE comes from K scores |

- `aggregate(scores, fold_sizes)` rejects no folds and unequal lengths with a
  clear `ValueError` (numpy's own errors are unclear), and returns a float.
- For a metric that isn't a mean over rows (RMSE, R²), size weights reduce
  noise but don't equal the pooled metric. Rejected weightings: by each
  fold's variance (leans toward easy folds) and by neighborhoods (a different
  target from the test-set metric).
- **SE** (`corrected_std_error(scores)`): Nadeau–Bengio in its K-fold form,
  `s·sqrt(1/K + 1/(K−1))`. Exact when every row is validated once (`random`,
  `grouped`); under `stratified_by_group`, rows alone in their stratum are
  never validated, so it is slightly conservative (measured: 7% at 20%
  singletons, small next to the SE's own ~35% sampling error at K=5). Under
  `LowerBound` it is the unweighted mean's SE, an approximation on unequal
  folds. It needs at least 2 folds. It is not stored per trial; compute it
  from `fold_scores`.

### 4.4 Study (`study.py`, Phase 3)

```python
class HyperparameterStudy:
    def __init__(self, *, seed: int, n_trials: int | None = 50,
                 timeout_seconds: float | None = None, n_jobs: int = 1,
                 sampler: BaseSampler | None = None,   # None -> TPESampler(seed=seed, multivariate=True)
                 pruner: BasePruner | None = None,     # None -> NopPruner()
                 show_progress_bar: bool = False, study_name: str | None = None,
                 storage: str | None = None,           # e.g. "sqlite:///tuning.db"
                 initial_params: Sequence[dict[str, Any]] = ()): ...
        # rejects: n_trials and timeout_seconds both None; n_trials < 1; n_jobs == 0

    def optimize(self, objective: Callable[[optuna.Trial], float]) -> TuningResult:
        # direction="maximize"; enqueue initial_params; best = max over COMPLETE
        # trials, ties -> lowest number; RuntimeError if none completed

@dataclass(frozen=True)
class TrialRecord: number, state, value, params, fold_scores, fold_sizes, std_error
    # the last three None unless COMPLETE; std_error = corrected_std_error(fold_scores)

@dataclass(frozen=True)
class TuningResult: best_params, best_value, best_trial_number, trials, is_reproducible
    def to_dict(self) -> dict   # JSON-safe, e.g. for MLflow
```
`is_reproducible` is `n_jobs == 1 and timeout_seconds is None`: timeouts and
parallel jobs are allowed, and the result records which kind of run it was.
The Optuna study is kept as `optuna_study_` for `optuna.visualization`. No
`catch=` option in v1: a NaN fold score stops the study (§4.2).

## 5. Worked example: `DirectCohortModel`

```python
parameters = [                                  # same bounds as [direct_cohort_search_space]
    IntParameter("max_depth", 3, 8),
    IntParameter("num_leaves", 7, 63),
    IntParameter("min_child_samples", 5, 40),
    FloatParameter("learning_rate", 0.01, 0.2, log=True),
    IntParameter("n_estimators", 50, 400),
    FloatParameter("reg_alpha", 1e-8, 10.0, log=True),
    FloatParameter("reg_lambda", 1e-8, 10.0, log=True),
    FloatParameter("min_split_gain", 0.0, 1.0),
    FloatParameter("subsample", 0.7, 1.0),
    FloatParameter("colsample_bytree", 0.7, 1.0),
]
splitter = Splitter("stratified_by_group")
train_df, test_df, Y_train, Y_test, g_train, g_test = splitter.train_test_split(
    df, df[cohort_columns], groups, test_size=0.2, random_state=42)
evaluator = CVHyperparameterEvaluator(
    DirectCohortModel(use_exposure=True), parameters,
    cv=splitter.cv(n_splits=5, random_state=42),
    metric=POISSON_DEVIANCE, feature_transformer=tree,
)
exposure_train = train_df["n_apartments"]
for cohort in cohort_columns:                    # one study per cohort
    result = HyperparameterStudy(seed=seeds[cohort], n_trials=30).optimize(
        lambda trial: evaluator.evaluate(
            trial, train_df, Y_train[cohort], g_train, exposure=exposure_train))
    transformer, model = evaluator.build_feature_transformer_and_model(result.best_params)
    transformer.fit(train_df, Y_train[cohort])
    models[cohort] = model.fit(
        transformer.transform(train_df), Y_train[cohort], exposure=exposure_train)
```

Compared with the old closure: `num_leaves` and `max_depth` are sampled
independently (D9; LightGBM caps the leaves by depth); `subsample_freq` is
derived inside the model; fixed settings live on the template; the model's
`random_state` replaces a seed per trial and fold. Out-of-fold predictions
(for `_cross_fitted_normal_scale`) are a possible later
`evaluator.out_of_fold_predict(params)`.

## 6. Remaining work

Every sub-task: plan mode first (probe, a plan with a "done when" list, open
choices asked), then §7's routine, then a stop for approval.

- [x] **2.4 Split methods:** `test_each_split_method_scores_its_own_folds`
  runs the evaluator with each `Splitter` method under
  `filterwarnings("error")` and compares each fold's size and score with the
  validator's folds (`grouped`: 20/5/3). The trial value alone can't tell the
  methods apart here: over folds that cover every row, the weighted mean of
  fold means is the overall mean. **Phase 2 review:** no correctness bugs.
  Fixed: `evaluate`'s `y` is typed as one target (`pd.Series | np.ndarray`;
  a two-column `y` failed inside LightGBM), the package docstring, a numpy
  test on a path nothing uses, and a misplaced comment.
- [ ] **3.1 Study constructor and defaults.** *Done when:* the default sampler is a seeded multivariate TPE and the default pruner `NopPruner`; invalid `n_trials`, timeout or `n_jobs` combinations are rejected.
- [ ] **3.2 `optimize` and the best trial.** *Done when:* the same seed gives an identical `TuningResult`; a pruned trial never wins; ties go to the lowest number; no completed trial raises `RuntimeError`; `initial_params` run first.
- [ ] **3.3 Records and results.** *Done when:* each `TrialRecord` carries the fold scores and sizes and the derived SE; `to_dict()` round-trips through `json`; `is_reproducible` is False with a timeout or `n_jobs > 1`.
- [ ] **3.4 Integration test** (§5 on a small simulated table, a few trials). *Done when:* the best parameters refit through `build_feature_transformer_and_model` and predict.
- [ ] **3.5 Pruners in the docs:** `WilcoxonPruner` or `SuccessiveHalvingPruner` for CV, not `MedianPruner` (§4.2). Wilcoxon's docs also advise shuffling the evaluation order per trial; decide whether that applies with fixed folds.
  **Then the Phase 3 review** and your approval.
- [ ] **4.1 `docs/HYPERPARAMETER_TUNING.md`:** the three parts, the decisions, aggregation and pruning guidance, the worked example, hazards, how a rebuilt model becomes tunable (D17), what remains.
- [ ] **4.2 Links** from `docs/README.md`, `docs/MODULE_REFERENCE.md` and `docs/SPLITTING.md` §7; mark this plan implemented.
- [ ] **4.3 Final check** (§7; `git status` shows only the package, `modeling/`, `scoring.py`, `utils.py`, their tests, `pyproject.toml` and the docs) and the PR checklist. **Then the Phase 4 review** and your approval; the PR moves from draft to ready.

## 7. Working notes

**How the user works:** one sub-task at a time, each ending with a stop.
Every class, field and mechanism is justified with measured evidence (probe
Optuna/sklearn/pydantic), or dropped; generic and simple over specific. No
checks or docs that aren't essential; comments keep only the non-obvious
"why". sklearn conventions and informative names (`train`/`val`,
`feature_transformer`, `exposure_*`). The user commits and pushes: give the
commands and a file-by-file .py summary (ask before a `Co-Authored-By`
line).

**Routine per sub-task:** implement → ruff, mypy (with the changed test
files), the changed tests under `-W error` → a mutation check per claimed
behavior (break it, see a test fail, restore, `cmp`) → an independent review
subagent → reproduce each finding before fixing or rejecting it → the
non-slow suite in the background, with nothing editing src → update this
doc → stop with the commands.

| purpose | command |
|---|---|
| package tests | `uv run pytest tests/unit/test_hyperparameter_tuning_*.py -q -W error` |
| suite | `uv run pytest -m "not slow"` (1002 passed, 8 warnings, ~85 s) |
| types | `uv run mypy`, plus `uv run mypy src/age_group_prediction/hyperparameter_tuning <changed test files>`. The package path is needed: without it, mypy doesn't check the tests' calls into the package. The parameters test has 4 deliberate type errors |
| lint | `uv run ruff check <files>` and `uv run ruff format <files>`, changed files only, paths listed explicitly (zsh doesn't split `$FILES`) |
| probes | `PYTHONPATH=src uv run --group test python -c "..."` |

**Pitfalls:**
- Pydantic wraps only `ValueError` in `ValidationError`: validators raise
  `ValueError`. In strict mode a float field turns `np.int64` into a float,
  and an int field rejects it; lax tuples accept sets and generators (a
  `Sequence` annotation rejects them); lax mode builds a stdlib dataclass
  from a dict (use `InstanceOf`).
- Your ruff settings (user-level; the repo has no `[tool.ruff]`, and adding
  one would replace them) enable B008 and RUF009: use
  `field(default_factory=...)` for a dataclass default.
- Optuna only warns for some bad settings (a step that doesn't divide the
  range lowers `high`), and its errors don't name the parameter. It warns at
  each `suggest` for non-scalar categorical choices (mind
  `filterwarnings("error")`), and a repeated `suggest` of a name returns the
  cached value, so a test must count calls.
- Optuna's sqlite storage returns tuples as lists and rejects `np.int64`
  attrs: store plain Python numbers.
- mypy's blind spots: pandas has no stubs, and `clone` returns `Any`.
- `BaseAgeGroupModel.__subclasses__()` includes test stand-ins (filter by
  package), and an empty `parametrize` list skips instead of failing.
- `if not x` raises on a numpy array; use `len(x) == 0`.
- A .py file that "looks unchanged" in the IDE is usually a stale buffer:
  "File: Revert File", and never save the old tab.

## 8. Draft PR text

**Title:** `feat: add modular Optuna hyperparameter tuning`. At each phase
stop, update the status line and checklist, and give the user the lines to
paste. No "Generated with" footer: the user removed it.

```markdown
**Status:** draft. Phases 1 (parameters) and 2 (the evaluator and fold
aggregations) are in. Next: the study.

## Summary
Adds the `hyperparameter_tuning` package: Optuna tuning in independent parts.

- **Parameters** (`parameters.py`): `FloatParameter`, `IntParameter` and
  `CategoricalParameter`, strict pydantic dataclasses validated when created
  (Optuna's checks plus the cases Optuna lets through). A search space is a
  plain list of them.
- **`CVHyperparameterEvaluator`** (`evaluator.py`): a frozen pydantic
  dataclass of settings (a `BaseAgeGroupModel`, parameters, a `Splitter.cv`
  validator, a `Metric`, a `FeatureTransformer`, an `Aggregation`), checked
  when built; data in `evaluate(trial, X, y, groups, *, exposure=None)`. Each
  trial copies the transformer and model once, and each fold refits them on
  its training rows. A lower-is-better metric is negated. Each fold's score
  is reported for pruning; a NaN or inf score raises. A completed trial
  records its fold scores and sizes.
- **Aggregations** (`aggregation.py`): `WeightedMean` (default), `Mean` and
  `LowerBound(z)`, via `aggregate(scores, fold_sizes)`; `corrected_std_error`
  is the K-fold Nadeau–Bengio SE.
- **`HyperparameterStudy`** (`study.py`, not started).

Supporting changes: `Metric` moved to `scoring.py`; `BaseAgeGroupModel.fit`/
`predict` take `exposure=None`, with a contract test over every model;
`Splitter.cv` requires an int `random_state`; a public `utils.py` with
`take_rows` and the table type aliases.

## Scope
Not wired into the experiment runner; `tuning.py` and the old `models/` are
unchanged. Design and decisions: `docs/HYPERPARAMETER_TUNING_PLAN.md`.

## Checklist
- [x] Phase 1: parameters
- [x] Phase 2: evaluator and aggregations
- [ ] Phase 3: study, including the integration test
- [ ] Phase 4: docs
- [ ] Non-slow suite passes
```

## 9. The switch to `BaseAgeGroupModel` (settled 2026-09-24)

The evaluator was first built for generic sklearn estimators. PR #6 gave
`modeling.BaseAgeGroupModel` `clone`/`set_params` (via `BaseEstimator`) and
fixed hyperparameters, so the evaluator tunes it instead (D11–D17). Measured
then: `clone(DirectCohortModel(...)).set_params(**all 10 §5 parameters)` fits
and leaves the template unchanged; the old `IndependentTotalProbabilityModel`
and `BayesianConditionalModel` can't be cloned; a `Pipeline` needs
`model__exposure=` in `fit` but `exposure=` in `predict`.
