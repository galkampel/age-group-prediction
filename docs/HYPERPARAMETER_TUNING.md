# Hyperparameter Tuning

> The `age_group_prediction.hyperparameter_tuning` package tunes the two rebuilt composites,
> the independent cohorts model (Model 1) and the total times probability model (Model 2),
> with plain Optuna and a cross-validation loop written out in full. A trial draws each
> component's choices, the components are set as complete records, and the composite is
> built from them with constructor calls: there are no `set_params` paths. The code was
> rebuilt from section 5 of the experiment notebook (`notebooks/02_experiments.py`), which
> is its first client. The decisions and their measured evidence are D1–D26 of the
> [notebook tuning plan](NOTEBOOK_TUNING_PLAN.md), above all D21 (what lives in the package),
> D25 (components read back from `best_params`) and D26 (the tuner).

Every code block below was run as written on 2026-10-29 (optuna 4.9.0, joblib 1.5.3,
LightGBM 4.7.0, scikit-learn 1.9.0); the numbers are its output. The blocks build on each
other in order.

## 1. The Parts

| module | answers | main names |
|---|---|---|
| `params` | which hyperparameters an estimator family searches | `LGBMParams`, `PoissonGLMParams`, `NB2Params`, `LogisticParams`, `EstimatorParams` |
| `components` | what one part of a composite is: family, transformer, estimator params | `CountComponent`, `ProbabilityComponent` (`from_params`, `make_estimator`, `to_dict`), `CountFamily`, `ProbabilityFamily` |
| `search_spaces` | how a trial samples one component's choices | `CountSearchSpace`, `ProbabilitySearchSpace` |
| `builders` | which components a trial draws for a composite, and the composite built from components | `suggest_independent_cohorts_components`, `suggest_total_times_probability_components`, `build_independent_cohorts_model`, `build_total_times_probability_model` |
| `cross_validation` | a model's cross-validated score, and the study of each composite | `CrossValidationTuner` |
| `evaluation` | the refit of the best model, scored on training and validation | `fit_and_score` |
| `aggregation` | how a trial's fold scores become its value | `Mean`, `WeightedMean`, `UpperBound`, `Aggregator`, `corrected_std_error` |

The dependencies run one way: `params` ← `components` ← `search_spaces` ← `builders` ←
`cross_validation`, which also uses `aggregation`; within the package, `evaluation` and
`aggregation` stand alone. Import from `age_group_prediction.hyperparameter_tuning`; nothing is re-exported from
the project's root.

What lives here is the mechanism: records a test can pin down and another notebook could
reuse. What stays in the notebook is the reading of one experiment (D21): the candidate
feature transformers of each family, the metrics, the settings, one cell per run, the tables
and the inspection. The package holds no runner over runs, no results table, no MLflow and
no transformer registry; the candidates are injected, as scikit-learn's
`GridSearchCV(param_grid=...)` takes its grid.

## 2. The Worked Example

The notebook's data (sections 1–3: the simulator at 150 neighborhoods × 9 buildings, the
`random` split of [Splitting §3](SPLITTING.md#3-train-validation-and-test), seed 42), two
candidate transformers per family, and one short study of each composite. The notebook runs
the same calls at 400 trials, one cell per family and exposure.

```python
import json

import optuna
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.feature_engineering import FeatureTransformer, OneHot, Plan, Quadratic, Standardize, CenterByReferencePoint
from age_group_prediction.hyperparameter_tuning import (
    CountComponent, CrossValidationTuner, ProbabilityComponent,
    build_independent_cohorts_model, build_total_times_probability_model, fit_and_score,
)
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from age_group_prediction.scoring import Metric, cohort_poisson_deviance
from age_group_prediction.splitting import Splitter
from age_group_prediction.utils import AgeGroupDataset
from student_simulator import (
    ApartmentSimulator, BuildingSimulator, CohortCompositionSimulator, NeighborhoodSimulator,
    StudentPopulationSimulator, TotalChildrenSimulator, load_simulation_config,
)

# Optuna logs every trial at INFO.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# The data of the notebook's sections 1-3: 150 neighborhoods, the random split, seed 42.
config = load_simulation_config("configs/simulation.yml").with_overrides({
    "simulation": {"n_neighborhoods": 150},
    "building": {"buildings_per_neighborhood_rate": 9.0},
})
simulator = StudentPopulationSimulator(
    NeighborhoodSimulator(config.simulation, config.neighborhood),
    BuildingSimulator(config.simulation, config.building, config.transforms),
    ApartmentSimulator(config.simulation, config.building.room_mix),
    TotalChildrenSimulator(config.simulation, config.total_children, config.transforms),
    CohortCompositionSimulator(
        config.simulation, config.cohort, config.transforms, config.total_children
    ),
    n_neighborhoods=config.simulation.n_neighborhoods,
    seed=config.simulation.seed,
)
df = ShareTransformer(
    ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
).fit_transform(simulator.simulate().dataset)
cohorts = ["n_kindergarten", "n_elementary", "n_highschool"]
X = df.drop(columns=["building_id", "neighborhood_id", *cohorts])
Y = df[cohorts]
exposure = ExposureTransformer("n_apartments").fit_transform(df)

splitter = Splitter("random")
train_val_index, test_index = splitter.train_test_indices(df, None, test_size=0.2, random_state=42)
train_sub, val_sub = splitter.train_test_indices(
    df.iloc[train_val_index], None, test_size=0.25, random_state=42
)
train, val = (
    AgeGroupDataset(
        X.iloc[index], Y.iloc[index], exposure.iloc[index],
        df["neighborhood_id"].iloc[index], df["building_id"].iloc[index],
    )
    for index in (train_val_index[train_sub], train_val_index[val_sub])
)
print(len(train.X), len(val.X), len(test_index))
```

```text
917 306 306
```

The candidates are the experiment's data. The tuner holds the study's design, which every
run shares; each call then names a composite, its settings and its seed.

```python
# Each family's candidate transformers, by name: the experiment's data, not the package's.
room_shares = ["4_rooms_share", "5_rooms_share", "6_rooms_share"]
numeric = [*room_shares, "ses", "avg_household_size", "median_age", "n_daycares_500m"]
school = (
    Plan(name="one_hot", transforms=(
        OneHot(categories=("none", "existing", "planned"), reference_category="none"),
    )),
    ["school_status"],
)
ses_squared = (
    Plan(name="square_around_0", transforms=(CenterByReferencePoint(reference_point=0.0), Quadratic())),
    ["ses"],
)
standardize = Plan(name="standardize", transforms=(Standardize(),))
# The trees take n_apartments as a feature too; the GLMs take it as the exposure only.
tree_candidates = {
    "tree": FeatureTransformer(
        column_plans=((Plan(name="passthrough"), [*numeric, "n_apartments"]), school)
    ),
    "tree__ses_quadratic": FeatureTransformer(
        column_plans=((Plan(name="passthrough"), [*numeric, "n_apartments"]), ses_squared, school)
    ),
}
glm_candidates = {
    "glm_base": FeatureTransformer(column_plans=((standardize, numeric), school)),
    "glm__ses_quadratic": FeatureTransformer(
        column_plans=((standardize, numeric), school, ses_squared)
    ),
}

# The study's design, shared by every run: the metric scores each fold and
# gives the direction; 5 folds fitted in 5 processes.
COHORT_POISSON_DEVIANCE = Metric("cohort_poisson_deviance", cohort_poisson_deviance)
tuner = CrossValidationTuner(
    cv=splitter.cv(n_splits=5, random_state=42),
    metric=COHORT_POISSON_DEVIANCE,
    n_trials=20,
    n_startup_trials=10,
    n_jobs=5,
)
print(tuner.cv, tuner.aggregator, tuner.n_trials, tuner.n_startup_trials, tuner.n_jobs)
```

```text
KFold(n_splits=5, random_state=42, shuffle=True) Mean() 20 10 5
```

```python
# The independent cohorts model: LightGBM's Poisson loss, the exposure on, one
# set of hyperparameters for every cohort, each cohort its own transformer.
study_1, seconds_1 = tuner.tune_independent_cohorts_model(
    train.X, train.Y, train.exposure,
    family="lgbm_poisson",
    use_exposure=True,
    cohorts=cohorts,
    feature_transformer_dict=tree_candidates,
    share_hyperparams=True,
    share_feature_transformer=False,
    y_min=1e-9,
    random_state=42,
    study_seed=42,
)
print(f"{study_1.best_value:.4f}", len(study_1.trials), len(study_1.best_params))
print(sorted({name.split("__")[0] for name in study_1.best_params}))
print({name: value for name, value in study_1.best_params.items() if name.endswith("transformer")})
```

```text
2.1569 20 13
['n_elementary', 'n_highschool', 'n_kindergarten', 'shared']
{'n_kindergarten__feature_transformer': 'tree', 'n_elementary__feature_transformer': 'tree__ses_quadratic', 'n_highschool__feature_transformer': 'tree'}
```

With `share_hyperparams=True` the ten LightGBM hyperparameters are drawn once, under
`shared__`, and each cohort still draws its transformer: 13 parameters instead of 33.

```python
# The total times probability model: a Poisson GLM total with the exposure,
# logistic cohort probabilities with a searched calibration.
study_2, seconds_2 = tuner.tune_total_times_probability_model(
    train.X, train.Y, train.exposure,
    total_family="poisson_glm",
    probability_family="logistic",
    use_exposure=True,
    total_feature_transformer_dict=glm_candidates,
    probability_feature_transformer_dict=glm_candidates,
    y_min=1e-9,
    random_state=42,
    study_seed=43,
)
print(f"{study_2.best_value:.4f}")
print({name: round(value, 4) if isinstance(value, float) else value
       for name, value in study_2.best_params.items()})
```

```text
2.2520
{'total__alpha': 0.0018, 'total__feature_transformer': 'glm__ses_quadratic', 'probability__C': 24.8263, 'probability__calibration_method': 'sigmoid', 'probability__feature_transformer': 'glm__ses_quadratic'}
```

A study keeps every trial; only the best is refit. Its components are read back from
`best_params` by the component types (§3), built the same way the objective built them,
refit on the whole training part and scored on both parts:

```python
# The best components, read back from best_params by their type, then built,
# refit on the whole training part and scored on training and validation.
best_components_1 = tuple(
    CountComponent.from_params(
        study_1.best_params, name=cohort, family="lgbm_poisson",
        use_exposure=True, feature_transformer_dict=tree_candidates,
    )
    for cohort in cohorts
)
best_total = CountComponent.from_params(
    study_2.best_params, name="total", family="poisson_glm",
    use_exposure=True, feature_transformer_dict=glm_candidates,
)
best_probability = ProbabilityComponent.from_params(
    study_2.best_params, name="probability", family="logistic",
    feature_transformer_dict=glm_candidates,
)
POISSON_DEVIANCE = Metric("poisson_deviance", mean_poisson_deviance)
for label, model in [
    ("model 1", build_independent_cohorts_model(best_components_1, y_min=1e-9, random_state=42)),
    ("model 2", build_total_times_probability_model(
        best_total, best_probability, y_min=1e-9, random_state=42
    )),
]:
    _, score_table = fit_and_score(
        model, train, val, cohort_metric=POISSON_DEVIANCE, metric=COHORT_POISSON_DEVIANCE
    )
    print(label)
    print(pd.DataFrame(score_table).round(4).to_string(index=False))
```

```text
model 1
        cohort  train    val
n_kindergarten 1.4953 2.2851
  n_elementary 1.2335 1.9329
  n_highschool 1.4479 2.1662
     composite 1.3922 2.1281
model 2
        cohort  train    val
n_kindergarten 2.3719 2.4573
  n_elementary 1.9832 2.0303
  n_highschool 2.1978 2.0501
     composite 2.1843 2.1792
```

The refit's training scores are in-sample; the validation column is the one that compares
runs (the notebook's `best_df`). What a run keeps:

```python
# What a run keeps: every trial, and each best component as plain data.
record = json.loads(json.dumps(best_total.to_dict()))
print(sorted(record))
print(record["estimator_params"], record["feature_transformer"]["column_plans"][0]["plan"]["name"])
trials = study_2.trials_dataframe()
print(len(trials), sorted(column for column in trials.columns if column.startswith("params_")))
```

```text
['estimator_params', 'family', 'feature_transformer', 'name', 'use_exposure']
{'alpha': 0.0017885780393720311} standardize
20 ['params_probability__C', 'params_probability__calibration_method', 'params_probability__feature_transformer', 'params_total__alpha', 'params_total__feature_transformer']
```

## 3. The Records

**Params records** (`params`) hold an estimator family's searched hyperparameters and
nothing else. They are frozen and keyword-only, and their field names are the estimator's
keywords, so a component makes its estimator as `Estimator(<fixed settings>,
**asdict(params))`. A typo raises when the record is built; LightGBM would accept
`max_dept=3` and fit silently (D11). `LGBMParams` serves the regressor and the classifier.
`EstimatorParams` is the union of the four.

**Components** (`components`) are complete records of one part of a run: `name` (a cohort,
or `"total"` / `"probability"`), `family`, `feature_transformer` (the object itself) and
`estimator_params`; `CountComponent` adds `use_exposure`, `ProbabilityComponent`
`calibration_method`. A component is set in one of two ways: by its constructor from a
trial's draws (the objective, §5), or by `from_params(best_params, *, name, family, ...,
feature_transformer_dict)` from a study's parameters (the refit). `from_params` takes the
entries under `<name>__` and under `shared__` (a choice is shared or the component's own,
never both), fills the family's record field by field, so a missing entry raises `KeyError`
naming it, and resolves the transformer's name through the dict.

`make_estimator(*, random_state=None)` is an `if`/`elif` on the family with a raising
`else`. Each family adds fixed settings to the record's values:

| family | estimator | fixed settings |
|---|---|---|
| `lgbm_poisson` | `LGBMRegressor` | `objective="poisson"`, `subsample_freq=1`, `n_jobs=1`, `verbosity=-1`, `random_state` |
| `lgbm_gaussian` | `LGBMRegressor` | `objective="regression"`, the rest as above |
| `poisson_glm` | `PoissonRegressor` | `max_iter=1000` |
| `nb2` | `NegativeBinomialRegressor` | — |
| `logistic` | `LogisticRegression` | `max_iter=1000` |
| `lgbm` (probability) | `LGBMClassifier` | `objective="multiclass"`, `subsample_freq=1`, `n_jobs=1`, `verbosity=-1`, `random_state` |

LightGBM's two losses are separate families (D6): TPE samples each parameter on its own,
so it cannot learn that good hyperparameters differ per loss. `subsample_freq=1` is needed
for `subsample` to act; `n_jobs=1` keeps the parallel folds from oversubscribing the cores
(D19). `to_dict()` gives the record as plain data, the transformer by its declaration
(`FeatureTransformer.to_dict()`: each plan with its columns, the interactions, the
remainder).

```python
from age_group_prediction.hyperparameter_tuning import NB2Params

# A typo is caught when the record is built (LightGBM would accept it silently).
try:
    NB2Params(dispersion=0.15, alpah=0.01)
except TypeError as error:
    print(error)
# A missing entry is named when a component is read back.
try:
    CountComponent.from_params(
        {"total__dispersion": 0.15, "total__feature_transformer": "glm_base"},
        name="total", family="nb2", use_exposure=True, feature_transformer_dict=glm_candidates,
    )
except KeyError as error:
    print(repr(error))
print(best_total.make_estimator(random_state=42))
print(best_components_1[0].make_estimator(random_state=42).get_params()["objective"])
```

```text
NB2Params.__init__() got an unexpected keyword argument 'alpah'. Did you mean 'alpha'?
KeyError('alpha')
PoissonRegressor(alpha=0.0017885780393720311, max_iter=1000)
poisson
```

## 4. The Search Spaces

A search space holds every Optuna call for one component and builds nothing (D10, D11).
Given the component's `name`, its `family` and its candidate transformers' names, it samples
under `<name>__<parameter>`: `suggest_hyperparameters(trial)` dispatches to the family's
explicit `suggest_<family>_hyperparams`, `suggest_feature_transformer_name(trial)` draws a
name (Optuna's choices must be primitives; the caller resolves the object through its dict,
D25.1), and `ProbabilitySearchSpace.suggest_calibration_method(trial)` draws from
`CALIBRATION_METHODS`. The separator is `__` because cohort names hold `_`.

| family | parameter | range |
|---|---|---|
| LightGBM (both losses, and the classifier) | `max_depth` | 3–8 |
| | `num_leaves` | 7–63 |
| | `min_child_samples` | 5–40 |
| | `learning_rate` | 0.01–0.2, log |
| | `n_estimators` | 50–400 |
| | `reg_alpha`, `reg_lambda` | 1e-8–10, log |
| | `min_split_gain` | 0–1 |
| | `subsample`, `colsample_bytree` | 0.7–1 |
| `poisson_glm` | `alpha` | 1e-4–10, log |
| `nb2` | `dispersion` | 0.01–1, log (the estimated dispersions are 0.12–0.18) |
| | `alpha` | 1e-3–1, log (from the unpenalized fit to all-zero coefficients) |
| `logistic` | `C` | 1e-3–100, log |
| probability, any family | `calibration_method` | `None`, `"temperature"`, `"sigmoid"` |

Isotonic calibration is not a candidate: its lowest score block can hold no child of a
cohort and then maps a building there to probability 0, an infinite log loss (measured with
the LightGBM classifier). The search spaces read nothing back; a study's `best_params` is
read by the components (§3).

## 5. The Builders

`suggest_independent_cohorts_components(trial, *, family, use_exposure, cohorts,
feature_transformer_dict, share_hyperparams, share_feature_transformer)` returns one
`CountComponent` per cohort; `suggest_total_times_probability_components(trial, *,
total_family, probability_family, use_exposure, total_feature_transformer_dict,
probability_feature_transformer_dict)` returns the total's component and the probabilities'.
They draw and set components only; `y_min` and `random_state` are build settings (D26.1).

The two sharing flags of the independent cohorts model (D23–D24): a shared choice is drawn
once, before the cohorts, under `shared__`, and every cohort takes it; otherwise each cohort
draws its own under `<cohort>__`. With LightGBM and three cohorts a trial draws 33
parameters with neither flag, 13 with `share_hyperparams`, 11 with both. Partial sharing
keeps the per-cohort feature choice (daycares matter for kindergarten, not for high school)
while TPE searches ten hyperparameters instead of thirty.

**The draw order is part of the result.** A seeded study draws the same values only in the
same order: the cohorts in the given order, each its hyperparameters then its transformer;
for Model 2 the total's hyperparameters and transformer, then the probabilities'
hyperparameters, calibration and transformer. The 24 baselines of §9 pin it.

`build_independent_cohorts_model(components, *, y_min, random_state)` makes one
`CountModel` per component, keyed by its name; `build_total_times_probability_model(total,
probability, *, y_min, random_state)` makes the total's `CountModel` and a
`CohortProbabilityModel` whose replication is fixed at `"weighted"` (D24.1: for the logistic
classifier the two replications are the same fit; for LightGBM they differed within noise
and `"per_child"` is 2–3× slower). `y_min` floors the predicted mean (a Gaussian loss can
predict a negative count; the Poisson deviance is infinite at 0). The objective and the
refit share these two functions, so the model scored in a trial is the model refit.

## 6. The Tuner

`CrossValidationTuner(cv, metric, n_trials, n_startup_trials, aggregator=Mean(), n_jobs=1)`
is a frozen dataclass holding what every run shares: the folds, the metric (it scores each
fold and gives the study its direction), the budget, TPE's random start, how fold scores
combine (§7) and how many processes fit a trial's folds. Each call takes the data, the
groups, the composite's settings and the study's seed (D26):

- `cv_score(model, X, Y, exposure, groups=None)` refits `model` on each fold's training rows
  (`fit` replaces all state) and aggregates the validation scores. The folds are fitted in
  `n_jobs` processes; the scores are the same as one after another, so a seeded study does
  not depend on `n_jobs` (D13: ×2.3–3.3 per trial with 5 processes).
- `tune_independent_cohorts_model(X, Y, exposure, groups=None, *, <the builder's
  settings>, y_min, random_state, study_seed)` and `tune_total_times_probability_model(...)`
  each run one study whose objective draws the components, builds the composite and
  returns its `cv_score`; they return the study and its seconds. The method names the
  model, so no callable is passed in.

The study is `optuna.create_study(direction=metric.direction,
sampler=TPESampler(seed=study_seed, n_startup_trials=n_startup_trials))`. The notebook runs
400 trials with a random start of 20 (D20: the LightGBM runs still improved after 200, and a
start of 10–20% of the budget is the usual rule). `study.optimize(..., catch=(RuntimeError,))`
marks a trial FAIL when its fit raises `RuntimeError` (NB2 that does not converge, raised
from a fold's process as such) and goes on; any other error ends the study, so a bug is not
hidden (D7, D19). There is no pruning: `WilcoxonPruner` needs the folds one after another,
which costs more than it saves (D19).

## 7. Aggregation

An aggregator turns a trial's fold scores into its value; it takes plain lists and knows
nothing about the data. It decides the selection (D9: on two Model 1 models the weighted
mean differed from the mean by 0.0006, while the median flipped which family won).

| class | trial value | use it |
|---|---|---|
| `Mean()` (the default) | `mean(scoreₖ)` | as `cross_val_score(...).mean()`; the folds differ by one row at most |
| `WeightedMean()` | `Σ nₖ·scoreₖ / Σ nₖ` | for unequal folds: for a per-row mean metric it is the pooled score a test set would report |
| `UpperBound(z=1.0)` | weighted mean + `z`·SE | a loss's pessimistic bound, to prefer stable settings over the best average |

`UpperBound` penalizes each candidate's own uncertainty; it is not the one-SE rule. Its
ranking is noisier than the mean's, because the SE comes from only K scores, and its
`best_value` is not the CV mean. The SE is `corrected_std_error(scores)`, Nadeau–Bengio's
correction in its K-fold form, `s · sqrt(1/K + 1/(K−1))`: the naive `s/sqrt(K)` ignores
that the folds share most of their training rows. A custom rule subclasses `Aggregator`
with the one method `aggregate_scores(scores, fold_sizes)`.

```python
import dataclasses
from collections.abc import Sequence

import numpy as np

from age_group_prediction.hyperparameter_tuning import (
    Aggregator, Mean, UpperBound, WeightedMean, corrected_std_error,
)

scores, sizes = [3.0, 5.0, 4.5], [20, 5, 3]
print(
    Mean().aggregate_scores(scores, sizes),
    WeightedMean().aggregate_scores(scores, sizes),
    round(corrected_std_error(scores), 4),
    round(UpperBound().aggregate_scores(scores, sizes), 4),
)


class Median(Aggregator):
    def aggregate_scores(self, scores: Sequence[float], fold_sizes: Sequence[int]) -> float:
        return float(np.median(scores))


print(Median().aggregate_scores(scores, sizes))
print(dataclasses.replace(tuner, aggregator=UpperBound(z=1.645)).aggregator)
```

```text
4.166666666666667 3.517857142857143 0.9501 4.468
4.5
UpperBound(z=1.645)
```

## 8. The Refit And The Run Record

`fit_and_score(model, train_age_group, val_age_group, *, cohort_metric, metric)` fits the
model on the training part and returns it with a score table: one row per cohort column,
scored by `cohort_metric`, then the composite, scored by `metric`, each on training and
validation. The best model is rebuilt from `best_params` (§3), so no fitted model or Optuna
study needs to be kept. A notebook run keeps a record: its settings,
`study.trials_dataframe()` (every trial, about 1.7 KiB each), `[c.to_dict() for c in
best_components]`, `best_params`, `best_value`, the score table, the seconds, the seed and
`y_min`. To save it as JSON use `trials.to_json(orient="records")`: `json.dumps(record,
default=str)` writes a bare `NaN`, which strict readers reject.

## 9. Reproducibility

- `study_seed` seeds TPE's draws; `random_state` seeds the estimators (LightGBM's bagging
  and column sampling). The notebook seeds run *i* with `seed + i`, in a fixed order of
  cells, and passes `random_state=seed`.
- The folds come from the seeded `cv`; fitting them in parallel changes nothing (tested).
- The draw order (§5) is fixed; a change to it, to a range, or to a candidate list changes
  every seeded result.
- **The regression check:** the notebook's 24 runs at 3 trials, both sharing flags off,
  reproduce to the fourth decimal the baselines recorded in the
  [handoff](NOTEBOOK_T4_HANDOFF.md) §5 (Check C). Every change to this package since it was
  rebuilt has passed it.

## 10. Hazards

- **Two `Parallel` calls at once from threads in one process return wrong scores**
  (measured: joblib's shared pool mixes the calls' inputs). Never run two tuner calls
  concurrently from threads, nor `study.optimize(n_jobs>1)`; marimo runs cells one at a
  time.
- **A warning in a fold process does not raise under `-W error`**: the processes do not
  inherit a script's `-W` filters. Set `PYTHONWARNINGS=error` in the environment for a
  strict check (D15).
- **A cohort name must not contain `__` nor be `shared`**: `from_params` splits the
  parameter names at the first `__`, and `shared__` is the shared choices' prefix. Nothing
  checks it.
- **A function sent to joblib must pickle**: in a notebook, a callable that captures the
  notebook's globals fails (`_thread.lock`). The tuner sends its bound `fold_score`, the
  model and the data, all of which pickle.
- **pandas turns a `None` calibration into NaN** in the trials table, and `value_counts()`
  drops it: count by label after `fillna("None")` (D16).
- **`num_leaves` above `2**max_depth` changes nothing in LightGBM**; TPE spends some trials
  on equivalent settings (T1's F7, left as a known inefficiency).
- **The logistic model needs scaled inputs**: it does not converge on the `tree`
  declaration, so its candidates are the GLM declarations.
- **NB2 can fail to converge**: its trial is marked FAIL and counted in the notebook's
  tables; a run whose trials all fail raises in its own cell.

## 11. What Remains

- MLflow runs from the notebook's run records.
- The final evaluation on the test part, after the refit on training + validation
  ([experiment notebook plan](EXPERIMENT_NOTEBOOK_PLAN.md) E11).
- Measured and rejected, not planned: pruning, LightGBM threads, the cohort fits in
  threads (D19).
