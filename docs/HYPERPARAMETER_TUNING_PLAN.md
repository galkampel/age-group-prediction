# Plan: modular Optuna hyperparameter tuning

**Branch:** `feat/hyperparameter-tuning` · draft PR #5.
**Status (2026-10-11):** Phases 1 (parameters) and 2 (evaluator, aggregations)
are done; 3.1 (the study's constructor and defaults, `study.py`) is committed
(`1f4ed67`); 3.2 (`optimize`, the best trial) is committed (`81f8896`); 3.3
(`TrialRecord`, the full `TuningResult`; the study minimizes, D12 reversed) is
committed (`b0d436c`); 3.4 (the slow integration test) and 3.5 (pruner
guidance) are done and await the user's commit. The sub-tasks are driven from
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §6 (A1–A6).
**Next:** **Phase 4** (4.1 the user doc `docs/HYPERPARAMETER_TUNING.md`, 4.2 links, 4.3 final check). The §6 task "Evaluator on the raw table" was done by [DIRECT_COHORT_GENERALIZATION_PLAN.md](DIRECT_COHORT_GENERALIZATION_PLAN.md) Step 5 (PR #12).
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
| D5 | Fold aggregation is an `Aggregation` object: `WeightedMean()` (default), `Mean()`, `UpperBound(z)` or a custom subclass, with one method `aggregate(scores, fold_sizes)` | it takes plain lists, so it knows nothing about the data; each class carries its own settings (`z`) |
| D7 | The study takes Optuna sampler and pruner objects, with seeded defaults | wrapping them would duplicate Optuna's API |
| D8 | Only COMPLETE trials can win | a pruned trial's value comes from fewer folds |
| D9 | No conditional or derived parameters in v1 | the LightGBM example (§5) doesn't need them |
| D10 | Settings in the constructor; data (`X, y, groups, exposure`) as arguments of the named method `evaluate` | the sklearn convention; one evaluator scores many targets; Optuna's one-argument objective is a lambda |
| D11 | The evaluator tunes a `BaseAgeGroupModel`: per trial, `build_model(params)` makes a copy (`clone`, `set_params`); each fold refits it on the raw rows, predicts, and scores with `model.evaluate(y, y_pred, metric)` | the repo's models get `clone`/`set_params` from `BaseEstimator`; a misspelled parameter raises at `set_params` |
| D12 | One direction: the score is the metric's value, or its negation if `greater_is_better`; the study always **minimizes** | no sign to get wrong; every metric in the repo is a loss (`POISSON_DEVIANCE`, `RMSE`, `MAE`, `COHORT_LOG_LOSS`), so for all of them the score *is* the metric, in Optuna's plots too; Optuna's `create_study()` minimizes by default, hyperopt only minimizes. `UpperBound`'s mean + z·SE and the tie rule need no sign. Each `TrialRecord` and the `TuningResult` still carry both the score and the value (the metric in its own sign, from the `greater_is_better` the evaluator records on that trial), which differ only for a greater-is-better metric. **Reversed 2026-10-11** (user's choice, folded into 3.3): it was "always maximize" (scikit-learn's `neg_*` convention), which negated every repo metric |
| D13 | A required `feature_transformer: FeatureTransformer`, fitted per fold on the training rows and targets | a transformer fitted on all rows would leak validation statistics; a `Pipeline` was rejected (its `fit` and `predict` name the exposure differently, and it has no `evaluate`). **Superseded** by [DIRECT_COHORT_GENERALIZATION_PLAN.md](DIRECT_COHORT_GENERALIZATION_PLAN.md) F1/F3: the model owns its `feature_transformer` and fits it inside its own `fit`, on each fold's training rows; the evaluator has no transformer |
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
    aggregation.py   # Aggregation, WeightedMean, Mean, UpperBound, corrected_std_error
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
that holds no data (`Aggregation`). Every user-built class takes the shared
`_config._STRICT` (`strict=True, allow_inf_nan=False, extra="forbid"`; a class
with Optuna or sklearn objects adds `arbitrary_types_allowed=True`). Pydantic
dataclasses inherit no config, so the shared constant is what keeps them
consistent; `extra="forbid"` (added in 3.1) rejects a misspelt keyword, which
pydantic's default ignores, leaving the intended setting at its default.

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
    aggregation: Aggregation = field(default_factory=WeightedMean)

    def evaluate(self, trial, X, y, groups=None, *, exposure=None) -> float: ...
    def build_model(self, params) -> BaseAgeGroupModel: ...
```

Per trial: suggest the parameters once, build one copy of the model, then for
each fold (`train_index`, `val_index`):
1. slice the raw `X`, `y` and `exposure` by position (`take_rows`);
2. fit the model (which fits its own `feature_transformer` on the training
   rows), predict the validation rows, and score them (signed, D12);
3. report the fold's own score to the pruner, and prune if it says so.

After the last fold: record the user attrs `fold_scores` and `fold_sizes`,
and return `aggregation.aggregate(scores, fold_sizes)`.

- **Settings are validated when the evaluator is built** (pydantic). Each
  check prevents a silent or late failure (measured): a string metric failed
  after 1 fold was fitted, a string aggregation after every fold; an empty
  search space gave identical trials; duplicate
  names made Optuna reuse the first value; a set of parameters would make a
  seeded study's order vary between runs. `model` (D17) and `cv` (always a
  `Splitter.cv` validator) are type-checked too. `InstanceOf[Metric]`: lax
  pydantic would otherwise build a `Metric` from a dict.
- **Data checks in `evaluate`:** an exposure that isn't one value per row of
  `X` (a longer one would be sliced to size silently); a NaN or inf fold score
  raises `ValueError` before `report` (Optuna stores NaN silently), which
  marks the trial FAIL and stops the study. The fold count is not checked
  here: the Splitter owns that rule (D3; `Splitter(m).cv(n_splits=1)` raises
  for all three methods, measured). A non-Splitter `cv` with one fold, outside
  that contract, fails loudly in `corrected_std_error` after the search (the
  3.3 review; a check here was added, then removed by the user's choice).
- **A copy once per trial, refits per fold:** `fit` replaces all fitted state
  (a contract test checks it), so one copy serves every fold; the copy keeps
  the template unfitted and parallel trials (`n_jobs > 1`) independent.
- **Pruning** sees each fold's own score (`step` = fold number), the contract
  of Optuna's `WilcoxonPruner`, which pairs fold k across trials. Measured,
  on 10 folds of unequal difficulty with a bad trial that ties on fold 0:
  `WilcoxonPruner` pruned it after 5 folds and `SuccessiveHalvingPruner`
  after 2; `MedianPruner`/`PercentilePruner` never did, whatever was reported,
  because they compare a trial's best value over its steps (a learning-curve
  assumption). **Guidance (3.5):** for CV, use `WilcoxonPruner` or
  `SuccessiveHalvingPruner`, never `MedianPruner`/`PercentilePruner`; the
  default is no pruning (`NopPruner`), and the notebook uses none.
- **Folds run in a fixed order (3.5).** Optuna's `WilcoxonPruner` docs advise
  shuffling the evaluation order per trial "so that the optimization doesn't
  overfit to the instances in the beginning". Measured on 8 folds of unequal
  difficulty, 80 trials, 4 seeds: under Wilcoxon, shuffled and fixed order
  gave about the same results (identical in our run; the review's own probe
  found small differences both ways, e.g. best 1.110 vs 1.148); under
  successive halving, shuffling made the best **worse** in 3 of 4 seeds in
  our run (1.163 vs 1.016, 1.048 vs 0.989, 1.186 vs 1.004) and 4 of 4 in the
  review's, because it compares trials at the
  same step, which must then be the same fold in every trial. So the order is
  kept fixed (user's rule: change only what affects the results). Without a
  pruner the order changes nothing: every fold is fitted, each fit starts
  afresh, and the aggregations ignore order.
- **User attrs** are written only for a completed trial, as plain Python
  values (what sqlite storage accepts and returns): `fold_scores`,
  `fold_sizes`, and (3.3) `metric_name` and `greater_is_better`, from which the
  study reports each trial's metric in its own sign (D12). A pruned trial's
  scores are its intermediate values.
- `groups=None` for the `random` method: `KFold` warns when given groups.
- `exposure` is the raw count the model expects, not its log. Build it with
  `ExposureTransformer` on the full table: the evaluator checks only its shape,
  and only `ExposureTransformer` checks its values.
- Leakage is still possible if `X` was already fitted on all rows and the
  model has `feature_transformer=None`.

### 4.3 Aggregation (`aggregation.py`)

| class | trial value | use it |
|---|---|---|
| `WeightedMean()` (default) | `Σ n_k·score_k / Σ n_k` | almost always; for a per-row mean metric (MAE, Poisson deviance) it is the pooled score the test set reports |
| `Mean()` | `mean(score_k)` | to match `cross_val_score(...).mean()` |
| `UpperBound(z=1.0)` | `weighted mean + z·SE` | a loss's pessimistic bound, to prefer stable settings; `z=1` is the one-SE rule, `z=1.645` roughly a one-sided 95% bound. Its ranking is noisier: the SE comes from K scores |

- `aggregate(scores, fold_sizes)` rejects no folds and unequal lengths with a
  clear `ValueError` (numpy's own errors are unclear), and returns a float.
- For a metric that isn't a mean over rows (RMSE, R²), size weights reduce
  noise but don't equal the pooled metric. `COHORT_LOG_LOSS` is a mean per
  **child**, not per row: the pooled loss would weight each fold by its
  children, so the size-weighted mean is close but not equal (B3,
  `MULTI_COHORT_MODELS_PLAN.md`). Rejected weightings: by each
  fold's variance (leans toward easy folds) and by neighborhoods (a different
  target from the test-set metric).
- **SE** (`corrected_std_error(scores)`): Nadeau–Bengio in its K-fold form,
  `s·sqrt(1/K + 1/(K−1))`. Exact when every row is validated once (`random`,
  `grouped`); under `stratified_by_group`, rows alone in their stratum are
  never validated, so it is slightly conservative (measured: 7% at 20%
  singletons, small next to the SE's own ~35% sampling error at K=5). Under
  `UpperBound` it is the unweighted mean's SE, an approximation on unequal
  folds. It needs at least 2 folds (`Splitter.cv` guarantees them, D3). The study stores
  it per trial, as `TrialRecord.std_error` (3.3).

### 4.4 Study (`study.py`, Phase 3)

Settled 2026-10-11 (sub-tasks 3.1, 3.2), measured on optuna 4.9.0 / pydantic 2.13.4.

```python
# A strict pydantic dataclass (the package's _STRICT config, plus
# arbitrary_types_allowed for the Optuna objects); not frozen, since optimize
# sets optuna_study_. kw_only=True: no field is positional.
@pydantic_dataclass(kw_only=True, config=ConfigDict(**_STRICT, arbitrary_types_allowed=True))
class HyperparameterStudy:
    seed: int                                          # seeds the default sampler only
    n_trials: int | None = 50
    timeout_seconds: float | None = None
    n_jobs: int = 1
    sampler: BaseSampler | None = None                 # None -> TPESampler(seed=seed, multivariate=True)
    pruner: BasePruner | None = None                   # None -> NopPruner()
    initial_params: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        # ValueError: "n_trials and timeout_seconds are both None: the study would never stop"
        # ValueError: "n_trials must be at least 1, got {n_trials}"
        # ValueError: "timeout_seconds must be positive, got {timeout_seconds}"
        # ValueError: "initial_params holds {k} trials but n_trials is {n}: the rest would never run"

    def _create_optuna_study(self) -> optuna.Study:
        # optuna.create_study(direction="minimize", sampler=..., pruner=...), the
        # defaults resolved here; called by each optimize

    def optimize(self, objective: Callable[[optuna.Trial], float]) -> TuningResult:
        study = self._create_optuna_study()
        for params in self.initial_params:
            study.enqueue_trial(dict(params))
        self.optuna_study_ = study     # before running: a failed trial stays inspectable
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_seconds,
                       n_jobs=self.n_jobs, callbacks=[self._check_initial_params])
        complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if len(complete) == 0:
            raise RuntimeError("no trial completed: every trial was pruned or the timeout came first")
        best = min(complete, key=lambda trial: trial.values[0])  # number order: a tie keeps the lowest
        ran = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))
        return TuningResult(
            best_params=dict(best.params),                  # a copy: see below
            best_score=best.values[0], best_value=_sign(best) * best.values[0],
            best_trial_number=best.number,
            metric_name=best.user_attrs.get("metric_name"),
            trials=tuple(_trial_record(trial) for trial in ran),
            is_reproducible=self.n_jobs == 1 and self.timeout_seconds is None)

    def _check_initial_params(self, study: optuna.Study, trial: FrozenTrial) -> None:
        # An Optuna callback, run after each trial. COMPLETE only: a pruned
        # objective may stop before suggesting every name.
        # ValueError: "initial_params[{i}]: no trial suggested {names}; check the names"

@dataclass(frozen=True)
class TrialRecord:                          # 3.3; built by _trial_record(trial)
    number: int
    state: TrialState                       # COMPLETE, PRUNED or FAIL
    params: dict[str, Any]                  # a copy
    metric_name: str | None                 # this trial's attr; None for a plain objective
    score: float | None                     # the search value (lower is better)
    value: float | None                     # the metric in its own sign: _sign(trial) * score
    fold_scores: tuple[float, ...] | None   # the evaluator's attrs
    fold_values: tuple[float, ...] | None   # in the metric's own sign
    fold_sizes: tuple[int, ...] | None
    std_error: float | None                 # corrected_std_error(fold_scores); sign-free
    # all but number, state and params None unless COMPLETE

def _sign(trial: FrozenTrial) -> float:
    # -1.0 when the trial's attr greater_is_better is True (its score was negated), else 1.0; 1.0 when absent

@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]             # 3.2
    best_score: float                       # 3.2 (named best_value there); the search value
    best_value: float                       # 3.3, the metric in its own sign
    best_trial_number: int                  # 3.2
    metric_name: str | None                 # 3.3, the best trial's attr
    trials: tuple[TrialRecord, ...]         # 3.3
    is_reproducible: bool                   # 3.3

    def to_dict(self) -> dict[str, Any]:
        # JSON-safe, for MLflow: the fields as they are, the state by name,
        # tuples as lists. No sign logic: the records already carry both.
```

**The constructor (3.1), each item with its evidence:**
- The two checks catch what Optuna swallows: with `n_trials` and `timeout`
  both `None` the study never stops; `n_trials` ≤ 0 and `timeout` ≤ 0 run 0
  trials silently. Nothing else is checked. `n_jobs` ≤ 0 (other than −1): Optuna
  raises `ValueError` at the start of `optimize`, before any trial runs; a
  negative seed: `TPESampler` raises a clear `ValueError`.
- Strict pydantic rejects `seed=True`, `n_trials=2.0`, `timeout_seconds=inf`
  and a string or dict sampler (`arbitrary_types_allowed` checks with
  `isinstance`; no `InstanceOf` needed, unlike the evaluator's `Metric`, which
  is itself a pydantic dataclass), and stores a given sampler or pruner as
  given (`is`), not a copy. An int timeout is stored as a float.
- `seed` is required and seeds only the default sampler; with a given sampler it
  is unused but still recorded (every study's seed is logged, E12 of
  `EXPERIMENT_NOTEBOOK_PLAN.md`).
- The defaults are built at each `optimize`, in `_create_optuna_study`, so a
  second `optimize` on the same object starts from the same seed and draws the
  same trials. The default `TPESampler(seed=seed, multivariate=True)` emits one
  `ExperimentalWarning` at construction (none during `optimize`); the trials
  are identical with and without it, so it is silenced narrowly, around that
  one construction, with `warnings.filterwarnings("ignore",
  message="Argument ``multivariate``")` (the pattern of `parameters.py`).
- Dropped from the first draft, unused by any planned client: `storage`,
  `study_name`, `show_progress_bar` (`storage` would also need a
  `load_if_exists` rule). Add them with their first use.
- `initial_params` is a `Sequence` (a set would vary the order). More of them
  than `n_trials` is rejected: the extra enqueued trials never run and stay
  WAITING, silently. (Only checkable with `n_trials`; with a timeout alone,
  the timeout decides.)

**`optimize` (3.2), each item with its evidence:**
- **The best trial is chosen here, not by `study.best_trial`.** Optuna updates
  its best with a strict `<` as trials *complete*, so a tie goes to the first
  to finish: with `n_jobs=2`, trial 1 beat trial 0 on a tie. `get_trials()` is
  in number order and `min` returns the first minimum, so a tie keeps the
  lowest number whatever the completion order.
- **COMPLETE trials only.** A PRUNED trial keeps a `value`, its last reported
  intermediate (100.0 in the 3.2 probe; the test reports −100.0, below every
  complete trial), which is not a full-CV score (D8).
  `TrialRecord` (3.3) must not report it as one.
- **No completed trial → `RuntimeError`** with a clear message (every trial
  pruned, or the timeout came before the first finished). Optuna's own
  `best_trial` says `ValueError: No trials are completed yet.`; our `min`
  over an empty list would say "min() arg is an empty sequence".
- **A misspelt `initial_params` key** is dropped by Optuna silently (no
  warning; the trial runs with the suggested parameters). The callback
  compares `initial_params[trial.number]` with `trial.params`: enqueued trials
  are numbered 0..k−1 when enqueued, so no Optuna internal (`fixed_params`) is
  needed. A raising callback stops `optimize` at once, so a typo costs one
  trial per job (measured: 1, 2, 4 trials with `n_jobs` = 1, 2, 4; no trial
  left RUNNING), not the study. The check is per trial: a name the objective
  suggests only on some branches would be flagged, which D9 (no conditional
  parameters) makes moot.
- **`best_params` is a copy** of the trial's params: `get_trials(deepcopy=False)`
  returns Optuna's stored dict, and editing the result rewrote the kept
  study's trial (measured).
- **Left to Optuna:** an objective error propagates at once (trial FAIL; no
  `catch=`); an out-of-range enqueued value is *warned* about ("Fixed
  parameter x … out of range for distribution").
- **`optuna_study_` is set before the search runs**, so a study that raised
  (a failed trial, a misspelt key) can still be inspected.

**Records and results (3.3), each item with its evidence:**
- **Which trials are recorded:** COMPLETE, PRUNED and FAIL. A NaN or `None`
  from a plain objective marks the trial FAIL and the study **continues**,
  with only a log line (no Python warning), so leaving FAIL out would hide it
  (the evaluator raises on NaN itself). WAITING trials, leftover
  `initial_params` after a timeout, never ran and have no params: not recorded.
- **A record's score is `None` unless COMPLETE:** a PRUNED trial's Optuna
  `value` is its last intermediate (measured: 100.0), not a full-CV score (D8).
  Its fold scores are in `optuna_study_` (its intermediate values).
- **Copies:** `params` is a `dict` copy and the folds are tuples, because
  `get_trials(deepcopy=False)` returns Optuna's own objects (the 3.2 finding).
- **Score and value, both (D12; user's choice 2026-10-11):** the score is
  what the search compared (lower is better); the value is the metric as
  measured. They differ only for a greater-is-better metric (an R² of 0.7:
  score −0.7, value 0.7); for every repo metric, a loss, they are equal.
  Each record takes its sign from *its own* trial's
  `greater_is_better` attr, so an objective that switches metric between
  trials is still reported right (the first design flipped every trial by the
  best trial's sign; the review showed a mixed objective flipped wrongly). The
  sign comes from the evaluator, so it cannot be forgotten; a plain objective
  without it is taken as a score (value = score). `std_error` is sign-free.
  `best_value` is the *aggregation's* value in the metric's sign, not
  necessarily the plain CV mean: under `UpperBound` it is the loss's mean +
  z·SE (for R², mean − z·SE), so log it under a name that says which.
  `UpperBound` gives a loss's mean + z·SE, its pessimistic bound, directly
  (under the former "maximize" it took −(−mean − z·SE); the review matched it
  by hand).
- **JSON:** the evaluator's attrs and the params are plain Python and
  round-trip through `json` exactly (measured, also with `allow_nan=False`);
  a NumPy int in a user's categorical choices raises `TypeError` in
  `json.dumps`, loud. `to_dict` converts only the state (an `IntEnum`: `json`
  would write `1`) and tuples to lists. An `inf` from a plain objective
  reaches `best_score` (the evaluator rejects it), and `json.dumps` with
  `allow_nan=False` would raise: left as is.
- **`is_reproducible`** is `n_jobs == 1 and timeout_seconds is None`; it
  assumes a given sampler is seeded (Optuna exposes no seed to check).

**Tests** (`tests/unit/test_hyperparameter_tuning_study.py`; each name is the
mistake it catches, each with the mutation that must fail it):

| test | mutation |
|---|---|
| `test_the_default_sampler_is_seeded_by_the_study`: two `_create_optuna_study()` calls on one study draw identical params over 12 trials of a toy objective; `isinstance(sampler, TPESampler)` | drop `seed=` |
| `test_the_default_sampler_is_multivariate`: `sampler._multivariate is True` (Optuna has no public getter) | drop `multivariate=True` |
| `test_the_sampler_searches_toward_lower_scores`: `direction == MINIMIZE` (D12; the best trial is our own `min`, so only the sampler and pruner see it) | `direction="maximize"` |
| `test_the_default_pruner_never_prunes`: `isinstance(pruner, NopPruner)` | another pruner |
| `test_a_given_sampler_and_pruner_are_used_as_given`: `is` | wrap or copy them |
| `test_the_default_sampler_does_not_warn`: the helper under `@pytest.mark.filterwarnings("error")` | remove the filter |
| `test_a_study_with_no_stopping_rule_is_rejected` | drop the check |
| `test_fewer_than_one_trial_is_rejected` (0, −1) | `< 0` |
| `test_a_non_positive_timeout_is_rejected` (0, −1.0) | `< 0` |
| `test_a_misspelt_setting_is_rejected` (`n_trial=10`) | drop `extra="forbid"` |
| `test_a_bool_seed_is_rejected` (`seed=True`, the one case lax mode accepts) | drop `_STRICT` |
| `test_more_initial_params_than_trials_are_rejected` (3.2) | drop the check |
| `test_the_same_seed_gives_the_same_result`: two objects, and two `optimize` calls on one (3.2) | an unseeded default sampler |
| `test_the_best_params_are_the_best_trials`: trial 2 is the only minimum (3.4 review) | `best_params` from another trial |
| `test_a_pruned_trial_never_wins`: a pruned trial reports −100.0, below every other (3.2) | include PRUNED states |
| `test_a_tie_goes_to_the_lowest_trial_number`: `n_jobs=2`, trial 0 finishes last, all return 1.0 (3.2) | return `study.best_trial` |
| `test_no_completed_trial_raises` (3.2) | drop the check |
| `test_initial_params_run_first_in_order` (3.2) | enqueue them reversed |
| `test_a_misspelt_initial_param_stops_the_study`: stops after 1 trial (3.2) | drop the callback |
| `test_the_optuna_study_is_kept`: `len(optuna_study_.trials) == n_trials` (3.2) | don't set it |
| `test_editing_the_result_leaves_the_study_intact` (3.2, from the review) | `best_params=best.params` |
| `test_a_complete_record_carries_its_folds_and_their_std_error` (3.3) | the naive `s/sqrt(K)` |
| `test_a_pruned_record_reports_no_score`: it reported 100.0 (3.3) | `score=trial.value` |
| `test_a_failed_trial_is_recorded`: a NaN objective (3.3) | record COMPLETE and PRUNED only |
| `test_trials_that_never_ran_are_not_recorded`: a timeout leaves WAITING (3.3) | record every state |
| `test_editing_a_record_leaves_the_study_intact` (3.3) | `params=trial.params` |
| `test_to_dict_round_trips_through_json` (3.3) | a NumPy value in the dict |
| `test_to_dict_names_the_state` (3.3): `TrialState` is an `IntEnum`, so `json` writes `1` and the round trip still compares equal (measured) | keep the `TrialState` in the dict |
| `test_a_loss_is_reported_as_it_is`: score = value, per record and best (3.3) | the evaluator's sign condition reversed (via `_sign` default) |
| `test_a_greater_is_better_metric_is_reported_in_its_own_sign`: score −0.7, value 0.7 (3.3) | `value=trial.values[0]`; `best_value=best.values[0]` |
| `test_each_trial_is_reported_in_its_own_metric_sign` (3.3, from the review) | one sign for every trial |
| `test_a_plain_objective_is_taken_as_a_score` (3.3) | default `greater_is_better=False` |
| `test_is_reproducible_only_without_a_timeout_or_parallel_jobs` (3 cases, 3.3) | drop the `n_jobs` term |

In `tests/unit/test_hyperparameter_tuning_evaluator.py`:
`test_a_completed_trial_records_its_folds`, now for both signs of
`greater_is_better`, expects the two new attrs (mutation: hard-code `True`;
the review found the single-sign version let it pass).

Timeouts and parallel jobs are allowed, and `is_reproducible` records which
kind of run it was. The Optuna study is kept as `optuna_study_` for
`optuna.visualization`. No `catch=` option in v1: through the evaluator, a NaN
fold score stops the study (§4.2); a plain objective's NaN is a FAIL record.

## 5. Worked example: `CountModel`

```python
parameters = [                                  # same bounds as [direct_cohort_search_space]
    IntParameter("estimator__max_depth", 3, 8),
    IntParameter("estimator__num_leaves", 7, 63),
    IntParameter("estimator__min_child_samples", 5, 40),
    FloatParameter("estimator__learning_rate", 0.01, 0.2, log=True),
    IntParameter("estimator__n_estimators", 50, 400),
    FloatParameter("estimator__reg_alpha", 1e-8, 10.0, log=True),
    FloatParameter("estimator__reg_lambda", 1e-8, 10.0, log=True),
    FloatParameter("estimator__min_split_gain", 0.0, 1.0),
    FloatParameter("estimator__subsample", 0.7, 1.0),
    FloatParameter("estimator__colsample_bytree", 0.7, 1.0),
]
# On the full table, before the split; then taken by the split's positions.
exposure = ExposureTransformer("n_apartments").fit_transform(df)
splitter = Splitter("stratified_by_group")
train_index, test_index = splitter.train_test_indices(
    df, groups, test_size=0.2, random_state=42)
train_df, Y_train = take_rows(df, train_index), take_rows(df[cohort_columns], train_index)
g_train, exposure_train = take_rows(groups, train_index), take_rows(exposure, train_index)
evaluator = CVHyperparameterEvaluator(
    CountModel(
        # subsample_freq=1: LightGBM ignores subsample without it.
        estimator=LGBMRegressor(objective="poisson", subsample_freq=1, n_jobs=1, verbosity=-1,
                                random_state=0),
        use_exposure=True, feature_transformer=tree),
    parameters,
    cv=splitter.cv(n_splits=5, random_state=42),
    metric=POISSON_DEVIANCE,
)
for cohort in cohort_columns:                    # one study per cohort
    result = HyperparameterStudy(seed=seeds[cohort], n_trials=30).optimize(
        lambda trial: evaluator.evaluate(
            trial, train_df, Y_train[cohort], g_train, exposure=exposure_train))
    models[cohort] = evaluator.build_model(result.best_params).fit(
        train_df, Y_train[cohort], exposure=exposure_train)
```

The runnable version is `tests/unit/test_hyperparameter_tuning_integration.py`
(3.4, marked `slow`): the simulator at 20 neighborhoods (79 buildings), `tree`
copied from `FEATURE_TRANSFORMATIONS.md` §8.1, 3 trials per cohort, then each
best parameter set rebuilt, refitted and used to predict the held-out rows
(about 0.2 s per cohort). Compared with the old closure: the model
takes the raw table and fits its own `feature_transformer` per fold; the
LightGBM settings live on the template estimator and are tuned by nested
names (`estimator__…`); `num_leaves` and `max_depth` are sampled
independently (D9; LightGBM caps the leaves by depth); `subsample_freq` is set
on the estimator, since the model no longer derives it; the estimator's
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
- [x] **Evaluator on the raw table** (from [MULTI_COHORT_MODELS_PLAN.md](MULTI_COHORT_MODELS_PLAN.md) N20). Switch `CVHyperparameterEvaluator` from a separate `feature_transformer` and model to one `modeling.ModelPipeline`, cloned per fold; its `exposure` argument is kept and passed through. Build that argument with `preprocessing.ExposureTransformer(...).fit_transform(table)`. *Done when:* the evaluator takes a pipeline, and a test shows it fits one per fold; §5's example builds `exposure = ExposureTransformer("n_apartments").fit_transform(df)` before the split and passes `take_rows(exposure, train_index)` (the split's positions), not `train_df["n_apartments"]`. **Done 2026-10-05** by [DIRECT_COHORT_GENERALIZATION_PLAN.md](DIRECT_COHORT_GENERALIZATION_PLAN.md) Step 5, with the model's own `feature_transformer` instead of `ModelPipeline` (removed there in Step 4): the evaluator has no transformer, fits the model on each fold's raw rows, and `build_feature_transformer_and_model` became `build_model`.
- [x] **3.1 Study constructor and defaults** (§4.4). *Done when:* the default sampler is a seeded multivariate TPE and the default pruner `NopPruner`, built without a warning; `n_trials=None` with `timeout_seconds=None`, `n_trials < 1` and `timeout_seconds <= 0` raise `ValueError`; a given sampler or pruner is used as given.
- [x] **3.2 `optimize` and the best trial.** *Done when:* the same seed gives an identical `TuningResult`; a pruned trial never wins; ties go to the lowest number; no completed trial raises `RuntimeError`; `initial_params` run first, in order, and a key that no trial suggested raises (Optuna drops it silently); more `initial_params` than `n_trials` raises; the Optuna study is kept as `optuna_study_`.
- [x] **3.3 Records and results.** *Done when:* each `TrialRecord` carries the fold scores and sizes and the derived SE; a PRUNED trial's record does not report its last intermediate as a score (§4.4: Optuna keeps it as the trial's `value`); `to_dict()` round-trips through `json`; `is_reproducible` is False with a timeout or `n_jobs > 1`; the best value is reported in the metric's own sign wherever it is shown or logged, the score kept as the search value (D12, reversed to "always minimize" in 3.3).
- [x] **3.4 Integration test** (§5 on a small simulated table, a few trials). *Done when:* the best parameters refit through `build_model` and predict.
- [x] **3.5 Pruners in the docs:** `WilcoxonPruner` or `SuccessiveHalvingPruner` for CV, not `MedianPruner` (§4.2). Wilcoxon's docs also advise shuffling the evaluation order per trial; decide whether that applies with fixed folds. *Decided:* no, measured (§4.2, "Folds run in a fixed order").
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

**Docs policy:** this doc holds only the current decisions, design and
remaining work; update those sections in place, with no step records or
review tables (the commit messages keep the history). Code comments keep only
the non-obvious "why".

**Before designing:** check what the library recommends (e.g. Optuna's
`WilcoxonPruner` docs for CV pruning), and measure the alternatives before
recommending one. When the user asks "why X?", answer with the evidence; ask
before changing a settled decision.

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
**Status:** draft. Phase 1 (parameters) and Phase 2 (the evaluator and fold
aggregations) are in. Next: the study (Phase 3), then the docs (Phase 4).

## Summary
Adds the `hyperparameter_tuning` package: Optuna tuning split into independent,
tested parts that tune the repo's own `modeling.BaseAgeGroupModel`s.

- **Parameters** (`parameters.py`): `FloatParameter`, `IntParameter` and
  `CategoricalParameter`, strict pydantic dataclasses validated when created.
  They run Optuna's own checks, plus three that Optuna lets through silently:
  a step that doesn't divide the range, duplicate choices, and NaN/inf
  bounds. A search space is a plain list of them.
- **`CVHyperparameterEvaluator`** (`evaluator.py`): a frozen pydantic
  dataclass of settings, checked when built: a model, parameters, a
  `Splitter.cv` validator, a `Metric` and an `Aggregation`. The data go to `evaluate(trial, X, y, groups, *, exposure=None)`.
  - Each trial copies the model once; each fold refits it (and its own
    `feature_transformer`) on its training rows only, so no validation rows
    leak in.
  - A greater-is-better metric is negated, so the study always minimizes.
  - Each fold's own score is reported for pruning, the contract of Optuna's
    `WilcoxonPruner`. A NaN or inf score raises.
  - A completed trial records its `fold_scores` and `fold_sizes`.
- **Aggregations** (`aggregation.py`): `WeightedMean` (the default), `Mean`
  and `UpperBound(z)`, each through `aggregate(scores, fold_sizes)`.
  `corrected_std_error` is the K-fold Nadeau–Bengio standard error.
- **`HyperparameterStudy`** (`study.py`): constructor, `optimize`, `TrialRecord` and `TuningResult` (3.1–3.3).

Supporting changes:
- `Metric` and the ready-made metrics moved to the top-level `scoring.py`.
- `BaseAgeGroupModel.fit`/`predict` take `exposure=None`, and a contract test
  runs sklearn's parameter checks over every model.
- `Splitter.cv` requires an int `random_state`, so every trial sees identical
  folds.
- A public `utils.py` holds `take_rows` and the table type aliases.

## Scope
Not wired into the experiment runner. `tuning.py` and the old `models/` are
unchanged. The design, the decisions and the remaining work are in
`docs/HYPERPARAMETER_TUNING_PLAN.md`.

## Checklist
- [x] Phase 1: parameters
- [x] Phase 2: evaluator and aggregations
- [ ] Phase 3: study, including the integration test
- [ ] Phase 4: docs
- [x] Non-slow suite passes (1002 passed)
```

## 9. The switch to `BaseAgeGroupModel` (settled 2026-09-24)

The evaluator was first built for generic sklearn estimators. PR #6 gave
`modeling.BaseAgeGroupModel` `clone`/`set_params` (via `BaseEstimator`) and
fixed hyperparameters, so the evaluator tunes it instead (D11–D17). Measured
then: `clone(DirectCohortModel(...)).set_params(**all 10 §5 parameters)` fits
and leaves the template unchanged; the old `IndependentTotalProbabilityModel`
and `BayesianConditionalModel` can't be cloned; a `Pipeline` needs
`model__exposure=` in `fit` but `exposure=` in `predict`.
