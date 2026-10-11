# Hyperparameter Tuning

> The `age_group_prediction.hyperparameter_tuning` package: Optuna tuning of the
> rebuilt `modeling` models, split into three independent parts. Parameters say
> which values a hyperparameter may take; `CVHyperparameterEvaluator` scores one
> trial by cross-validation; `HyperparameterStudy` runs the search and reports the
> winner. Built and tested on PR #5; the experiment notebook
> ([plan](EXPERIMENT_NOTEBOOK_PLAN.md)) is its first client. The decisions
> (D1–D20), the design with its measured evidence and the test table are in the
> [tuning plan](HYPERPARAMETER_TUNING_PLAN.md). The old stack's tuning
> (`tuning.py`, inside `models/`) is described in
> [MODELING_GUIDE.md](MODELING_GUIDE.md#hyperparameter-tuning) until it is deleted.

Every code block below was run as written on 2026-10-11 (Optuna 4.9.0); the
numbers are its output. The blocks build on each other in order.

## 1. The Parts

| part | answers | class |
|---|---|---|
| parameters | which values may each hyperparameter take? | `FloatParameter`, `IntParameter`, `CategoricalParameter` |
| evaluator | how is one trial scored? | `CVHyperparameterEvaluator` with an `Aggregator` |
| study | how does Optuna search, and which trial won? | `HyperparameterStudy` → `TuningResult` |

One trial flows through them like this: the study asks Optuna for a trial; the
evaluator suggests every parameter once, builds one copy of the model with them
(`clone` + `set_params`), fits and scores it on each fold of a seeded
`Splitter.cv` validator, and returns the aggregated score; the study keeps the
best COMPLETE trial and a record of every trial. Imports go one way,
`evaluator → parameters, aggregation, modeling, scoring, splitting`; the study
knows only the aggregation's standard error and accepts any `trial -> float`
objective. Nothing here is re-exported from `age_group_prediction`'s root:
import from `age_group_prediction.hyperparameter_tuning`.

## 2. The Worked Example

Model 1 ([Independent cohort models](INDEPENDENT_COHORT_MODELS.md)): one
`CountModel` with a LightGBM Poisson regressor per cohort, each tuned by its own
study, then assembled and refit. The data are the simulator at 20 neighborhoods
(79 buildings); `tree` is the declaration of
[Feature transformations §8.1](FEATURE_TRANSFORMATIONS.md), copied until it
lives in `src` (the notebook plan's D1).

```python
import optuna
from lightgbm import LGBMRegressor

from age_group_prediction.feature_engineering import ColumnPlan, FeatureTransformer, OneHot
from age_group_prediction.hyperparameter_tuning import (
    CVHyperparameterEvaluator,
    FloatParameter,
    HyperparameterStudy,
    IntParameter,
)
from age_group_prediction.modeling import CountModel, IndependentCohortModels
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from age_group_prediction.scoring import POISSON_DEVIANCE
from age_group_prediction.splitting import Splitter
from age_group_prediction.utils import take_rows
from student_simulator import StudentPopulationSimulator, load_simulation_config

optuna.logging.set_verbosity(optuna.logging.WARNING)

config = load_simulation_config("configs/simulation.toml")
config = config.model_copy(
    update={"simulation": config.simulation.model_copy(update={"n_neighborhoods": 20})}
)
table = ShareTransformer(
    ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
).fit_transform(StudentPopulationSimulator(config).run())
cohorts = ["n_kindergarten", "n_elementary", "n_highschool"]

tree = FeatureTransformer(
    plans=(
        ColumnPlan(
            name="numeric",
            columns=("4_rooms_share", "5_rooms_share", "6_rooms_share", "ses",
                     "avg_household_size", "median_age", "n_daycares_500m", "n_apartments"),
        ),
        ColumnPlan(
            name="school",
            columns=("school_status",),
            transforms=(OneHot(categories=("none", "existing", "planned"), reference_category="none"),),
        ),
    )
)

# The exposure on the full table; then every array is taken by the split's positions.
exposure = ExposureTransformer("n_apartments").fit_transform(table)
groups = table["neighborhood_id"]
splitter = Splitter("stratified_by_group")
train_index, test_index = splitter.train_test_indices(table, groups, test_size=0.2, random_state=42)
train, test = take_rows(table, train_index), take_rows(table, test_index)
groups_train = take_rows(groups, train_index)
exposure_train, exposure_test = take_rows(exposure, train_index), take_rows(exposure, test_index)

parameters = [
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
evaluator = CVHyperparameterEvaluator(
    CountModel(
        # subsample_freq=1: LightGBM ignores subsample without it. n_jobs=1: one thread.
        estimator=LGBMRegressor(objective="poisson", subsample_freq=1, n_jobs=1, verbosity=-1, random_state=0),
        use_exposure=True,
        feature_transformer=tree,
    ),
    parameters,
    cv=splitter.cv(n_splits=5, random_state=42),
    metric=POISSON_DEVIANCE,
)

results, models = {}, {}
for i, cohort in enumerate(cohorts):
    y_train = train[cohort]
    results[cohort] = HyperparameterStudy(seed=100 + i, n_trials=30).optimize(
        lambda trial: evaluator.evaluate(trial, train, y_train, groups_train, exposure=exposure_train)
    )
    models[cohort] = evaluator.build_model(results[cohort].best_params)

model = IndependentCohortModels(models).fit(train, train[cohorts], exposure=exposure_train)
prediction = model.predict(test, exposure=exposure_test)

result = results["n_elementary"]
best = result.trials[result.best_trial_number]
print(result.metric_name, round(result.best_value, 3), round(best.std_error, 3), result.best_trial_number)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.best_params.items()})
print(prediction.shape)
```

```text
poisson_deviance 1.812 0.225 13
{'estimator__max_depth': 8, 'estimator__num_leaves': 21, 'estimator__min_child_samples': 11, 'estimator__learning_rate': 0.0308, 'estimator__n_estimators': 134, 'estimator__reg_alpha': 1.3506, 'estimator__reg_lambda': 0.0, 'estimator__min_split_gain': 0.3634, 'estimator__subsample': 0.8965, 'estimator__colsample_bytree': 0.9262}
(20, 3)
```

The three studies (90 trials, 450 LightGBM fits) and the refit took 9 s. The
names are `set_params` keys: `estimator__…` reaches the `CountModel`'s
estimator; `num_leaves` and `max_depth` are sampled independently (LightGBM caps
the leaves by depth). The same loop, as a test, is
`tests/unit/test_hyperparameter_tuning_integration.py`.

## 3. Parameters

A search space is a plain list of parameters. Each is a frozen, strict pydantic
dataclass, validated when created rather than at the first trial: pydantic checks
the types (`IntParameter("p", 1.5, 9)` fails; so does a numpy bound, use
`int(x)`), then Optuna's own distribution class checks the values, and three
rules Optuna lets through silently are added: a step that does not divide the
range (Optuna would quietly lower `high`), duplicate choices (double weight),
and, in the evaluator, duplicate names (Optuna would reuse the first value).

| class | Optuna call | arguments |
|---|---|---|
| `FloatParameter(name, low, high, *, log=False, step=None)` | `suggest_float` | `log` needs `low > 0`; `step` must divide `high − low`; not both |
| `IntParameter(name, low, high, *, log=False, step=1)` | `suggest_int` | `log` needs `low ≥ 1` and `step == 1` |
| `CategoricalParameter(name, choices)` | `suggest_categorical` | choices are `None`, `bool`, `int`, `float` or `str`; a list is stored as a tuple; a set is rejected (its order varies between runs) |

Choices are restricted to Optuna's own scalar types (D18) because Optuna hands a
choice back by reference: a list choice edited through a result changed the kept
study and the parameter itself, and a tuple or an object cannot be written to
JSON. A `np.float64` is a `float` and is kept as given; a `np.int64` is rejected.

```python
import numpy as np

from age_group_prediction.hyperparameter_tuning import CategoricalParameter

for build in (
    lambda: FloatParameter("estimator__learning_rate", 0.0, 1.0, step=0.3),
    lambda: CategoricalParameter("calibration_method", [None, "temperature", "temperature"]),
    lambda: CategoricalParameter("estimator__max_depth", [np.int64(3), np.int64(5)]),
    lambda: IntParameter("estimator__max_depth", np.int64(3), 8),
):
    try:
        build()
    except ValueError as error:
        print(error.errors()[0]["msg"])
```

```text
Value error, parameter 'estimator__learning_rate': The distribution is specified by [0.0, 1.0] and step=0.3, but the range is not divisible by `step`. It will be replaced with [0.0, 0.9].
Value error, parameter 'calibration_method': choices must be unique, got (None, 'temperature', 'temperature')
Value error, choices must be None, bool, int, float or str, got [np.int64(3), np.int64(5)]
Input should be a valid integer
```

Conditional or derived parameters (a parameter suggested only on some branches)
are out of scope (D9).

## 4. The Evaluator

`CVHyperparameterEvaluator(model, parameters, *, cv, metric, aggregation=WeightedMean())`
holds the settings (scikit-learn style: settings in the constructor, data as
method arguments) and checks them when built: `model` is a
`modeling.BaseAgeGroupModel`, `cv` a `Splitter.cv(n_splits=..., random_state=int)`
validator (so every trial sees identical folds, D3), `metric` a `scoring.Metric`,
`aggregation` an `Aggregator` (§5). Each check prevents a failure that would
otherwise come after a fold was fitted.

`evaluate(trial, X, y, groups=None, *, exposure=None) -> float` does, per trial:

1. suggest every parameter once and build one copy of the model
   (`build_model(params)`: `clone`, then `set_params`, which raises on a
   misspelt name);
2. for each fold, slice `X`, `y` and `exposure` by position (`take_rows`), fit the
   copy on the training rows (the model fits its own `feature_transformer` there,
   so no validation statistic leaks), predict and score the validation rows with
   `model.evaluate(y_val, y_val_pred, metric)`;
3. negate the score if the metric is greater-is-better, so the study always
   minimizes (D12); raise on a NaN or infinite score (Optuna would store it
   silently); report the fold's own score to the pruner (§7);
4. after the last fold, record `fold_scores`, `fold_sizes`, `metric_name` and
   `greater_is_better` on the trial (the study reads them) and return
   `aggregation.aggregate(scores, fold_sizes)`.

`X` is the raw table (a numpy array works only with a model that takes the
design matrix, i.e. `feature_transformer=None`). `y` is one target: a Series or
array for `CountModel`, the (buildings × cohorts) count DataFrame for
`CohortProbabilityModel` with `COHORT_LOG_LOSS`. `groups` are the neighborhood
ids for the grouped and stratified methods and `None` for `random` (`KFold`
warns otherwise). `exposure` is the raw count from `ExposureTransformer`, one
value per row of `X` (checked; a longer one would be sliced to size silently).

Two uses beyond a study. `optuna.trial.FixedTrial(params)` scores one setting
without a search, with the same folds; and an **empty search space** scores a
model that has no hyperparameters (D20), as the notebook's NB2 run does, in a
study of one trial (more trials would repeat the same fit):

```python
from optuna.trial import FixedTrial

from age_group_prediction.modeling import NegativeBinomialRegressor

one = evaluator.evaluate(FixedTrial(result.best_params), train, train["n_elementary"], groups_train, exposure=exposure_train)
print(round(one, 3), one == result.best_value)

nb2 = CVHyperparameterEvaluator(
    CountModel(estimator=NegativeBinomialRegressor(), use_exposure=True, feature_transformer=tree),
    [],
    cv=splitter.cv(n_splits=5, random_state=42),
    metric=POISSON_DEVIANCE,
)
nb2_result = HyperparameterStudy(seed=0, n_trials=1).optimize(
    lambda trial: nb2.evaluate(trial, train, train["n_elementary"], groups_train, exposure=exposure_train)
)
print(nb2_result.best_params, round(nb2_result.best_value, 3), round(nb2_result.trials[0].std_error, 3), len(nb2_result.trials))
```

```text
1.812 True
{} 2.247 0.454 1
```

## 5. Aggregation

An aggregation turns a trial's fold scores into the one value the study
minimizes. It takes plain lists and knows nothing about the data.

| class | trial value | use it |
|---|---|---|
| `WeightedMean()` (default) | `Σ nₖ·scoreₖ / Σ nₖ` | almost always: for a per-row mean metric (MAE, Poisson deviance) it is the pooled score a test set would report |
| `Mean()` | `mean(scoreₖ)` | to match `cross_val_score(...).mean()` |
| `UpperBound(z=1.0)` | `weighted mean + z·SE` | a loss's pessimistic bound, to prefer stable settings over the best average |

`UpperBound` is a penalty on each candidate's **own** uncertainty, not the
one-SE rule (the simplest model within one SE of the best): a noisy candidate
loses even when its mean lies inside the best one's interval. `z=1.645` is
roughly a one-sided 95% bound. Its ranking is noisier than the mean's, because
the SE is estimated from only K scores; and its `best_value` is not the CV mean,
so log the aggregation beside it.

The SE is `corrected_std_error(scores)`: Nadeau–Bengio's correction in its K-fold
form, `s · sqrt(1/K + n_val/n_train)` with `n_val/n_train = 1/(K−1)`, `s` the
scores' standard deviation (ddof=1). The naive `s/sqrt(K)` ignores that the folds
share most of their training rows, so it understates how much the whole CV
estimate would vary on new data. The correction is an approximation of that
variance, not an exact formula: the `1/(K−1)` identity holds when every row is
validated once (`random`, `grouped`; under `stratified_by_group` rows alone in
their stratum are never validated, so the true ratio is a little smaller), and
for a predictor that does not depend on its training rows there is nothing to
correct and it overstates (measured, a fixed predictor on 5 folds: true SE 0.061,
corrected 0.090, naive 0.060). It needs at least 2 folds, which `Splitter.cv`
guarantees. The study stores it per trial as `TrialRecord.std_error`.

```python
from collections.abc import Sequence

from age_group_prediction.hyperparameter_tuning import Aggregator, Mean, UpperBound, WeightedMean, corrected_std_error

scores, sizes = [3.0, 5.0, 4.5], [20, 5, 3]
print(WeightedMean().aggregate(scores, sizes), Mean().aggregate(scores, sizes), round(corrected_std_error(scores), 4), round(UpperBound().aggregate(scores, sizes), 4))

stable = [1.0 - 0.05 / np.sqrt(3), 1.0 + 0.05 / np.sqrt(3)]  # mean 1.00, SE 0.05
simple = [1.03 - 0.2 / np.sqrt(3), 1.03 + 0.2 / np.sqrt(3)]  # mean 1.03, SE 0.20: within one SE of the stable one
print(round(UpperBound().aggregate(stable, [1, 1]), 2), round(UpperBound().aggregate(simple, [1, 1]), 2))


class Median(Aggregator):
    def aggregate(self, scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
        return float(np.median(scores))


print(Median().aggregate(scores, sizes))
```

```text
3.517857142857143 4.166666666666667 0.9501 4.468
1.05 1.23
4.5
```

A custom rule subclasses `Aggregator` with the one method; for a rule with a
validated setting, decorate it as `UpperBound` is (a frozen pydantic dataclass
with the package's strict config).

## 6. The Study

`HyperparameterStudy(*, seed, n_trials=50, sampler=None, pruner=None)` holds
the search settings; `optimize(objective) -> TuningResult` runs it. The default
sampler is `TPESampler(seed=seed, multivariate=True)`, built afresh for each
`optimize`, so two calls on one study draw the same trials; a given `sampler` is
used as is and the seed is then unused (but still recorded: every study's seed
is logged). The default pruner is `NopPruner`. Trials run one after the other;
a timeout, parallel jobs and warm-start trials were dropped because no client
uses them (D19). The study always minimizes; `n_trials < 1` and a misspelt
keyword are rejected.

`optimize` returns the best **COMPLETE** trial (a pruned trial's value comes
from fewer folds, D8), chosen here rather than by Optuna's `best_trial`, with a
tie going to the lowest trial number; it raises `RuntimeError` when no trial
completed. The Optuna study stays on the object as `optuna_study_`, also when
the search raised, for `optuna.visualization` or inspection.

`TuningResult`: `best_params` (a copy), `best_value`, `best_trial_number`,
`metric_name` (the evaluator's; `None` for a plain objective) and `trials`, one
`TrialRecord` per COMPLETE, PRUNED or FAIL trial (`number`, `state`, `params`,
`value`, `fold_values`, `fold_sizes`, `std_error`). Values are the metric in its
**own sign** (an R² searched as −0.7 is reported as 0.7; a loss is reported as it
is); the search-side numbers stay in `optuna_study_`. All but `number`, `state`
and `params` are `None` unless the trial is COMPLETE: a pruned trial's Optuna
value is its last intermediate, not a CV score. A plain objective's trial has
`fold_values` and `std_error` `None`. `to_dict()` is JSON-safe (the state by
name, tuples as lists), for MLflow artifacts.

```python
import json

payload = result.to_dict()
print(sorted(payload), sorted(payload["trials"][0]))
print(json.loads(json.dumps(payload)) == payload)

plain = HyperparameterStudy(seed=0, n_trials=3).optimize(lambda t: (t.suggest_float("x", -1, 1) - 0.2) ** 2)
print(plain.metric_name, [round(t.value, 3) for t in plain.trials], plain.trials[0].fold_values)
```

```text
['best_params', 'best_trial_number', 'best_value', 'metric_name', 'trials'] ['fold_sizes', 'fold_values', 'number', 'params', 'state', 'std_error', 'value']
True
None [0.01, 0.053, 0.0] None
```

## 7. Pruning

The evaluator reports each fold's own score at `step = fold`, the contract of
Optuna's `WilcoxonPruner`, which pairs fold k across trials. Use
`WilcoxonPruner` or `SuccessiveHalvingPruner` for cross-validation, never
`MedianPruner` or `PercentilePruner`: those compare a trial's *best* value over
its steps, a learning-curve assumption, and on folds of unequal difficulty they
prune by fold luck (measured in the plan, §4.2). The folds run in a fixed order
in every trial, although Optuna's Wilcoxon docs suggest shuffling: measured,
shuffling changed little under Wilcoxon and made successive halving worse,
because it compares trials at the same step, which must then be the same fold.
Pruning pays with many folds; on this 5-fold example `WilcoxonPruner()` at its
default `p_threshold=0.1` pruned nothing, and `SuccessiveHalvingPruner()` at its
defaults completed only 2 of 30 trials. Both pruners emit Optuna's
`ExperimentalWarning` once, when constructed.

```python
from optuna.pruners import WilcoxonPruner

y_train = train["n_elementary"]
pruned_study = HyperparameterStudy(seed=101, n_trials=30, pruner=WilcoxonPruner(p_threshold=0.2))
pruned = pruned_study.optimize(
    lambda trial: evaluator.evaluate(trial, train, y_train, groups_train, exposure=exposure_train)
)
states = [t.state.name for t in pruned.trials]
fits = sum(len(t.intermediate_values) for t in pruned_study.optuna_study_.trials)
print(states.count("COMPLETE"), states.count("PRUNED"), fits, round(pruned.best_value, 3), round(result.best_value, 3))
```

```text
18 12 130 1.694 1.812
```

Twelve trials were pruned, saving 20 of 150 fits. The best value differs from
the unpruned run of the same seed because pruned trials change what the sampler
has seen; the folds themselves are unchanged.

## 8. Reproducibility

The study's `seed` covers the default sampler only. What it does not cover must
be seeded on its own: the estimator (`random_state` on `LGBMRegressor`, with
`n_jobs=1`: more threads change the fit), a sampler you pass in, the split
(`Splitter.train_test_indices(..., random_state=...)`) and the folds
(`Splitter.cv(..., random_state=...)`, required). With those fixed, the same
seed gives an identical result, trial for trial, because the trials run one
after the other with a fixed count:

```python
again = HyperparameterStudy(seed=101, n_trials=30).optimize(
    lambda trial: evaluator.evaluate(trial, train, y_train, groups_train, exposure=exposure_train)
)
print(again == result)
```

```text
True
```

The notebook plan names four seeds (E12): `split_seed`, `cv_seed`, `study_seed`
(one per study, `study_seed + i` for the i-th cohort or component) and
`estimator_seed`, all logged.

## 9. Making A Model Tunable

The evaluator tunes any `modeling.BaseAgeGroupModel` (D17). A model is tunable
when it stores its constructor settings verbatim and validates in `fit`: it then
gets `get_params`, `set_params` and `clone` from scikit-learn's `BaseEstimator`,
and a contract test (`tests/unit/test_modeling_contract.py`) runs scikit-learn's
parameter checks over every model in the package. Nested names reach inside:
`estimator__alpha` for a `CountModel`'s regressor, `total_model__estimator__alpha`
and `probability_model__calibration_method` for `TotalTimesProbabilityModel`.

Two composites are tuned per component, not as one object. `IndependentCohortModels`
does not forward `set_params` into its mapping, so Model 1 is tuned as one study
per cohort `CountModel` (§2). Model 2 is tuned as two studies (the notebook plan's
E4): the total `CountModel` on `POISSON_DEVIANCE` of `y.sum(axis=1)` and the
`CohortProbabilityModel` on `COHORT_LOG_LOSS` of `y`; the tuned components are
then assembled into one `TotalTimesProbabilityModel` and refit.

## 10. Hazards

- **Leakage.** The model fits its features on each fold's training rows only
  because it owns its `feature_transformer`. A table transformed on all rows and
  a model with `feature_transformer=None` leak validation statistics into every
  fold.
- **The exposure.** Build it on the full table with `ExposureTransformer` (it
  rejects what LightGBM would accept silently) and take its rows by the split's
  positions, never `train["n_apartments"]`; the evaluator checks only its shape.
- **One column per metric.** `POISSON_DEVIANCE`, `RMSE` and `MAE` score one
  target column; a (n, K) matrix raises "Multioutput not supported". The matrix
  metric `COHORT_LOG_LOSS` normalizes rows, so it ignores the total. The
  cross-setting count metric on the whole matrix comes with the notebook plan's
  PR B.
- **`groups=None` for `random`.** `KFold` warns on every split that receives
  groups.
- **An empty search space with many trials** repeats one fit `n_trials` times;
  pair it with `n_trials=1`.
- **numpy numbers.** A `np.int64` bound or choice is rejected (`int(x)`); a
  `np.float64` choice is a float and passes.
- **`UpperBound`'s `best_value`** is the mean plus `z` SE, not the CV mean: log
  the aggregation beside it.
- **Warnings.** Building parameters escalates Optuna's `UserWarning`s to errors
  process-wide for the duration of the check, so it is not thread-safe; the
  default sampler's `ExperimentalWarning` is silenced around its construction
  only; a pruner's is not.
- **A NaN score** from the evaluator raises and stops the study (the trial
  FAILs); a plain objective's NaN only fails its trial, and the search goes on
  with a FAIL record.

## 11. What Remains

- The notebook plan's PR B adds `COHORT_POISSON_DEVIANCE` (the cross-setting
  metric); PR D logs a `TuningResult` to MLflow (`log_tuning_result`), the
  first user of `to_dict()`, and runs the NB2 one-trial studies.
- Only on demand: out-of-fold predictions from the evaluator, Optuna
  visualizations as artifacts, conditional parameters.
- The old stack's `tuning.py` and `models/` are deleted with roadmap step 5 of
  the [re-implementation plan](MODEL_REIMPLEMENTATION_PLAN.md).
