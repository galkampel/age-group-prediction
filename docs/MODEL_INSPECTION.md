# Model Inspection

> The `age_group_prediction.inspection` package: what a **fitted** model does
> and how sure its score is, in three independent parts. `BootstrapEvaluator`
> gives a metric's mean and percentile interval on an evaluation set;
> `PermutationImportance` scores each raw input column by the metric's drop
> when it is shuffled; `ShapExplainer` gives Shapley values of each row's
> prediction per raw column. Built on PRs #14 and C of the
> [experiment notebook plan](EXPERIMENT_NOTEBOOK_PLAN.md), whose notebook is
> the first client. The decisions and measured evidence are in the
> [bootstrap plan](MODEL_INSPECTION_PLAN.md) (I1–I2, I7, I12–I13) and the
> [importance plan](FEATURE_IMPORTANCE_PLAN.md) (I3–I6, I8–I11).

Every code block below was run as written on 2026-10-13 (shap 0.53.0,
scikit-learn 1.9.0, LightGBM 4.7.0); the numbers are its output. The blocks
build on each other in order.

## 1. The Parts

| part | answers | class |
|---|---|---|
| bootstrap | how sure is the validation score? | `BootstrapEvaluator(n_resamples, alpha=0.05, random_state=None).evaluate(..., metrics=[...])` → a DataFrame, one row per metric: `score`, `mean`, `lower`, `upper` |
| permutation importance | which columns does the score depend on? | `PermutationImportance(metric, n_repeats, random_state=None)` → a DataFrame per column |
| SHAP | how does each column move each row's prediction? | `ShapExplainer(method, max_evals=None, random_state=None, n_jobs=1)` → a `shap.Explanation` |

All three take a fitted `BaseAgeGroupModel` and the **raw** table the model
was fitted on: the model transforms it inside `predict` through its own
`feature_transformer`, so a one-hot or interaction column is credited to its
source column. Data are paired by position, never by index label. Settings
are checked when the object is built (a misspelt keyword, a bool or string
where an int is expected, and numpy integers are rejected). Nothing is
re-exported from `age_group_prediction`'s root: import from
`age_group_prediction.inspection`.

## 2. The Worked Example

The simulator at 150 neighborhoods × 9 buildings, the three-way split of
[Splitting §3](SPLITTING.md#3-train-validation-and-test) (`grouped`, seed 42),
and the two rebuilt composites at their class defaults: Model 1
([Independent cohort models](INDEPENDENT_COHORT_MODELS.md)), one LightGBM
Poisson `CountModel` per cohort on the `tree` declaration of
[Feature transformations §8.1](FEATURE_TRANSFORMATIONS.md), and Model 2
([Total times probability model](TOTAL_TIMES_PROBABILITY_MODEL.md)), a Poisson
total times a logistic-regression composition on standardized columns. Both
are fitted on the training part and inspected on the validation part.

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.feature_engineering import FeatureTransformer, OneHot, Plan, Standardize
from age_group_prediction.inspection import BootstrapEvaluator, PermutationImportance, ShapExplainer
from age_group_prediction.modeling import (
    CohortProbabilityModel, CountModel, IndependentCohortModels, TotalTimesProbabilityModel,
)
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from age_group_prediction.scoring import Metric, cohort_log_loss, cohort_poisson_deviance
from age_group_prediction.splitting import Splitter
from age_group_prediction.utils import take_rows
from student_simulator import StudentPopulationSimulator, load_simulation_config

# The metrics, built here: the notebook does the same (the scoring module holds no constants).
COHORT_POISSON_DEVIANCE = Metric("cohort_poisson_deviance", cohort_poisson_deviance)
COHORT_LOG_LOSS = Metric("cohort_log_loss", cohort_log_loss)
POISSON_DEVIANCE = Metric("poisson_deviance", mean_poisson_deviance)

base = load_simulation_config("configs/simulation.yml")
config = base.with_overrides({
    "simulation": {"n_neighborhoods": 150},
    "building": {"buildings_per_neighborhood_rate": 9.0},
})
table = ShareTransformer(
    ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
).fit_transform(StudentPopulationSimulator(config).run())
cohorts = ["n_kindergarten", "n_elementary", "n_highschool"]
numeric = ("4_rooms_share", "5_rooms_share", "6_rooms_share", "ses",
           "avg_household_size", "median_age", "n_daycares_500m", "n_apartments")
school = (
    Plan(
        name="one_hot",
        transforms=(OneHot(categories=("none", "existing", "planned"), reference_category="none"),),
    ),
    ["school_status"],
)
tree = FeatureTransformer(column_plans=((Plan(name="passthrough"), list(numeric)), school))
linear = FeatureTransformer(
    column_plans=((Plan(name="standardize", transforms=(Standardize(),)), list(numeric)), school)
)
X, Y = table[[*numeric, "school_status"]], table[cohorts]
exposure = ExposureTransformer("n_apartments").fit_transform(table)
groups = table["neighborhood_id"]

splitter = Splitter("grouped")
train_val_index, test_index = splitter.train_test_indices(table, groups, test_size=0.2, random_state=42)
train_sub, val_sub = splitter.train_test_indices(
    take_rows(table, train_val_index), take_rows(groups, train_val_index), test_size=0.25, random_state=42
)
train_index, val_index = train_val_index[train_sub], train_val_index[val_sub]
X_train, X_val = take_rows(X, train_index), take_rows(X, val_index)
Y_train, Y_val = take_rows(Y, train_index), take_rows(Y, val_index)
exposure_train, exposure_val = take_rows(exposure, train_index), take_rows(exposure, val_index)
groups_val = take_rows(groups, val_index)

model_1 = IndependentCohortModels({
    cohort: CountModel(
        estimator=LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1, random_state=0),
        use_exposure=True, feature_transformer=tree,
    )
    for cohort in cohorts
}).fit(X_train, Y_train, exposure=exposure_train)
model_2 = TotalTimesProbabilityModel(
    total_model=CountModel(estimator=PoissonRegressor(), use_exposure=True, feature_transformer=linear),
    probability_model=CohortProbabilityModel(
        estimator=LogisticRegression(max_iter=1000), feature_transformer=linear
    ),
).fit(X_train, Y_train, exposure=exposure_train)
models = {"Model 1": model_1, "Model 2": model_2}
print(len(X_train), len(X_val), groups_val.nunique())
```

```text
908 311 30
```

### The bootstrap

```python
bootstrap = BootstrapEvaluator(500, random_state=0)
metrics = [COHORT_POISSON_DEVIANCE]
for name, model in models.items():
    rows = bootstrap.evaluate(model, X_val, Y_val, exposure=exposure_val, metrics=metrics)
    by_group = bootstrap.evaluate(
        model, X_val, Y_val, groups_val, exposure=exposure_val, metrics=metrics
    )
    rows, by_group = rows.iloc[0], by_group.iloc[0]
    print(f"{name}  rows: {rows['mean']:.3f} [{rows['lower']:.3f}, {rows['upper']:.3f}]"
          f"  neighborhoods: {by_group['mean']:.3f} [{by_group['lower']:.3f}, {by_group['upper']:.3f}]")
```

```text
Model 1  rows: 2.589 [2.330, 2.842]  neighborhoods: 2.598 [2.200, 3.031]
Model 2  rows: 2.187 [1.967, 2.410]  neighborhoods: 2.193 [1.961, 2.472]
```

### Permutation importance

On the validation rows, the decision table; on the training rows, the overfit
diagnostic; and, for Model 2, per component.

```python
importance = PermutationImportance(COHORT_POISSON_DEVIANCE, 10, random_state=0)
permutation = {}
for name, model in models.items():
    on_train = importance.compute_importances(model, X_train, Y_train, exposure=exposure_train)
    on_val = importance.compute_importances(model, X_val, Y_val, exposure=exposure_val)
    permutation[name] = on_val["importance_mean"]
    print(name)
    print(pd.DataFrame({"train": on_train["importance_mean"], "val": on_val["importance_mean"],
                        "val_std": on_val["importance_std"]}).sort_values("val", ascending=False).round(3))
component = pd.DataFrame({
    "total (poisson_deviance)": PermutationImportance(POISSON_DEVIANCE, 10, random_state=0)
        .compute_importances(model_2.total_model_, X_val, Y_val.sum(axis=1), exposure=exposure_val)["importance_mean"],
    "probability (cohort_log_loss)": PermutationImportance(COHORT_LOG_LOSS, 10, random_state=0)
        .compute_importances(model_2.probability_model_, X_val, Y_val)["importance_mean"],
})
print(component.sort_values("total (poisson_deviance)", ascending=False).round(4))
```

```text
Model 1
                    train    val  val_std
avg_household_size  1.870  0.438    0.073
5_rooms_share       0.441  0.066    0.034
6_rooms_share       0.581  0.062    0.032
4_rooms_share       0.359  0.051    0.019
school_status       0.138  0.042    0.021
median_age          0.479 -0.004    0.024
ses                 0.360 -0.010    0.027
n_apartments        0.340 -0.035    0.023
n_daycares_500m     0.184 -0.070    0.021
Model 2
                    train    val  val_std
avg_household_size  0.566  0.337    0.033
median_age          0.241  0.119    0.029
6_rooms_share       0.118  0.076    0.014
5_rooms_share       0.058  0.038    0.018
school_status       0.029  0.019    0.006
4_rooms_share       0.003  0.005    0.002
ses                 0.000 -0.002    0.001
n_apartments        0.003 -0.002    0.002
n_daycares_500m     0.032 -0.003    0.010
                    total (poisson_deviance)  probability (cohort_log_loss)
avg_household_size                    0.6859                         0.0072
6_rooms_share                         0.0598                         0.0038
5_rooms_share                         0.0598                         0.0012
school_status                         0.0527                         0.0001
median_age                            0.0198                         0.0074
4_rooms_share                         0.0090                         0.0002
n_apartments                          0.0033                        -0.0002
ses                                  -0.0043                        -0.0000
n_daycares_500m                      -0.0242                         0.0003
```

### SHAP

200 seeded validation rows against 50 seeded training rows, with the
validation exposure, so the values explain the predicted **counts** and sum
to them. The global importance is the mean |SHAP| per column and cohort.

```python
rng = np.random.default_rng(0)
explained = rng.choice(len(X_val), size=200, replace=False)
background = X_train.iloc[rng.choice(len(X_train), size=50, replace=False)]
X_shap, exposure_shap = X_val.iloc[explained], np.asarray(exposure_val)[explained]
explanations, mean_abs = {}, {}
for name, model in models.items():
    # Nine columns: 512 evaluations per row; Model 1 runs three LightGBM
    # models per evaluation, so its rows go to four processes.
    explainer = ShapExplainer("exact", n_jobs=4 if name == "Model 1" else 1)
    explanations[name] = explainer.explain(model, X_shap, background, exposure=exposure_shap)
    values = explanations[name]
    mean_abs[name] = pd.DataFrame(
        np.abs(values.values).mean(axis=0), index=values.feature_names, columns=values.output_names
    )
    mean_abs[name]["all cohorts"] = mean_abs[name].sum(axis=1)
    y_pred = np.asarray(model.predict(X_shap, exposure=exposure_shap), dtype=float)
    print(name, "additivity", f"{np.abs(values.base_values + values.values.sum(axis=1) - y_pred).max():.1e}")
    print(mean_abs[name].sort_values("all cohorts", ascending=False).round(3))
ranks = pd.DataFrame({
    f"{name} {kind}": series.rank(ascending=False).astype(int)
    for name in models
    for kind, series in (("permutation", permutation[name]), ("mean |SHAP|", mean_abs[name]["all cohorts"]))
})
print(ranks.sort_values("Model 2 permutation").to_string())
```

```text
Model 1 additivity 4.3e-14
                    n_kindergarten  n_elementary  n_highschool  all cohorts
avg_household_size           1.410         1.709         1.748        4.867
ses                          0.894         0.531         0.679        2.104
6_rooms_share                0.464         0.618         0.954        2.035
median_age                   1.089         0.216         0.358        1.663
5_rooms_share                0.303         0.516         0.610        1.430
4_rooms_share                0.344         0.352         0.313        1.009
school_status                0.283         0.340         0.375        0.998
n_daycares_500m              0.531         0.309         0.136        0.976
n_apartments                 0.426         0.260         0.231        0.918
Model 2 additivity 3.4e-14
                    n_kindergarten  n_elementary  n_highschool  all cohorts
avg_household_size           0.128         0.720         1.063        1.911
median_age                   1.024         0.150         0.534        1.708
6_rooms_share                0.183         0.280         0.723        1.186
5_rooms_share                0.197         0.319         0.403        0.920
n_daycares_500m              0.394         0.168         0.140        0.702
school_status                0.137         0.131         0.247        0.515
4_rooms_share                0.074         0.093         0.048        0.215
n_apartments                 0.043         0.101         0.052        0.195
ses                          0.028         0.056         0.010        0.094
                    Model 1 permutation  Model 1 mean |SHAP|  Model 2 permutation  Model 2 mean |SHAP|
avg_household_size                    1                    1                    1                    1
median_age                            6                    4                    2                    2
6_rooms_share                         3                    3                    3                    3
5_rooms_share                         2                    5                    4                    4
school_status                         5                    7                    5                    6
4_rooms_share                         4                    6                    6                    7
ses                                   7                    2                    7                    9
n_apartments                          8                    9                    8                    8
n_daycares_500m                       9                    8                    9                    5
```

What the tables say, and what the sections below explain:

- The cluster bootstrap (whole neighborhoods) is wider than the row
  bootstrap, 0.83 against 0.51 for Model 1: buildings in a neighborhood share
  their simulator draws, so the rows are not independent.
- Model 1 at LightGBM's defaults overfits: its training importances are 4 to
  100 times its validation ones (`median_age` 0.48 on train, 0.00 on
  validation). Model 2's are close. The validation column is the decision
  table; the training column is the diagnostic.
- Both methods agree on the top of Model 2 (`avg_household_size`,
  `median_age`, the room shares). For Model 1, `ses` is 2nd by mean |SHAP| and
  7th by permutation: the trees use `ses`, but shuffling it costs nothing
  because correlated columns carry the same signal.
- Model 2's total depends on `avg_household_size` (0.69 of the deviance); its
  composition depends on little, in log-loss units.

The blocks above took 44 s on an 8-core laptop: the fits and bootstraps
under a second each, the permutation tables 6 s, SHAP 31 s for both models
(4 s for Model 2; Model 1 alone takes 90 s sequentially, 27 s on four
processes).

## 3. The Bootstrap

`BootstrapEvaluator.evaluate(model, X, y, groups=None, *, exposure=None, metrics)`
predicts `X` **once**, then draws `n_resamples` replicates of the evaluation
rows with replacement and recomputes every metric on each. The evaluator
holds the resampling design only; the metrics are named per call, as
`cross_validate`'s `scoring`, and all of them score the same replicates, so
their intervals are paired. The result has one row per metric name: `score`
is the metric on the unresampled rows, `mean` the mean of the replicate
scores, `lower` and `upper` their `alpha/2` and `1 - alpha/2` quantiles
(`np.quantile`, so the 95 % interval at the default `alpha=0.05`). The interval is **conditional on the fitted model**: nothing is
refit, so it measures how much the score depends on which rows were held out,
not how much the model would change on other training data.

**Whole neighborhoods.** With `groups` (one label per row), each replicate
draws as many groups as there are and keeps all their rows. Buildings in one
neighborhood share the neighborhood's draws, so resampling rows pretends to
have more independent evidence than there is and gives an interval that is
too narrow; the example shows it. Pass the validation rows' `neighborhood_id`.
With `groups=None` rows are drawn.

**Few groups.** The cluster bootstrap resamples groups, so with few of them
the interval is wide and unstable from seed to seed; with 30 validation
neighborhoods, as above, it is usable. With one group every replicate would
be the whole set and the interval would have zero width, so fewer than two
distinct groups raise `ValueError`.

**Ratio metrics.** The cohort log loss is a ratio of sums recomputed on each
replicate, as it should be; a replicate with no children at all is 0/0, and a
non-finite replicate score raises `ValueError` naming the metric and how many, rather than
returning NaN bounds silently.

## 4. Permutation Importance

`PermutationImportance(metric, n_repeats, random_state).compute_importances(model, X, y, *, exposure=None)`
shuffles one column of `X` at a time, `n_repeats` times, and scores the model
on `metric` each time (scikit-learn's `permutation_importance` with a scorer
built from the `Metric`). The importance of a column is the score drop,
averaged over the repeats:

    importance = score(X) - score(X with the column shuffled)

with a loss entering negated, so a **positive** importance means the column
helps, whichever way the metric points, and the units are the metric's.
`importance_std` is the spread over the repeats; a column whose mean is
within a standard deviation of zero is noise. The exposure stays paired with
its row (scikit-learn shuffles a copy of one column with the rows in place).

**Held-out rows** answer "what matters for the prediction we ship"; the
**training rows** show what the model fitted on, and a column important on
train and not on validation is an overfit signal (Model 1 above).

**Correlated columns** share or hide importance: shuffling one of two columns
that carry the same signal costs little, because the other still carries it,
so both can look unimportant while the model needs at least one. The
simulator's columns are correlated by construction. There are no column
blocks; compare with mean |SHAP|, which credits a used column regardless.

**Per component.** The fitted children of Model 2 are public, so the recipe
above runs the total's `CountModel` on the Poisson deviance of `y.sum(axis=1)`
with the exposure, and the `CohortProbabilityModel` on the cohort log loss of
the count matrix (it normalizes the rows, so the total does not enter). For
Model 1 the composite restricted to one output column is the cohort's own
model, so there is nothing more to split.

## 5. SHAP

`ShapExplainer(method, max_evals=None, random_state=None, n_jobs=1).explain(model, X, background, *, exposure=None)`
returns a `shap.Explanation` whose `values[i, j]` is the Shapley value of
column `j` for row `i`: how much knowing that column's value moves the row's
prediction away from the average prediction over `background`
(`base_values`). The values are **additive**: `base_values + values.sum(axis=1)`
equals `model.predict(X, exposure)` to floating-point precision (4e-14 above).
For a cohort model `values` has shape `(rows, columns, cohorts)` and
`output_names` holds the cohort columns, so `explanation[:, :, "n_elementary"]`
slices one cohort.

**What is explained.** Without `exposure` the values explain the **rate**,
`predict(X, exposure=1)`; with it, the **count** each row is predicted to
have. The rows are explained one at a time: shap evaluates the model on
synthetic rows that mix one row of `X` with the background rows, and hands
them over without saying which row they came from, so each row's own exposure
can only reach `predict` row by row. The cost is the same as shap's batched
call, which loops over the rows itself.

**The background and the masker.** A Shapley value needs the prediction with
some columns "absent". `shap.maskers.Independent(background)` defines absent
as: replace the absent columns of the row by each background row's values in
turn and average the predictions. The background is therefore the reference
the values are relative to; a few dozen **training** rows are enough, and
every background row is used (shap's own default would keep 100 silently).

**Codes and dtypes.** shap's masker needs one float matrix, so non-numeric
columns reach it as integer codes from a scikit-learn `OrdinalEncoder` fitted
on `X` and `background` together, and every column travels as float64; the
prediction function decodes them and restores `X`'s dtypes before `predict`,
so the model sees `school_status` as the strings it was fitted on.
`Explanation.data` holds the codes and `display_data` the original values,
which is what the plots label with.

**Exact or permutation.** `"exact"` (`shap.ExactExplainer`) evaluates every
coalition of the columns: `2 ** n_columns` model evaluations per row, exact
Shapley values, no seed. `"permutation"` (`shap.PermutationExplainer`) samples
column orders within `max_evals` evaluations per row (shap's default 500; at
least `2 * n_columns + 1`): approximate, seeded by `random_state`. Measured at
5 columns on Model 2, exact took 0.2 s per 20 rows and permutation 5.3 s, and
they agreed to 2e-5; exact is cheaper up to about 9 columns (512
evaluations) and explodes past 12 (4096). Each evaluation runs
`len(background)` rows through `predict`, transformer included, which is why
Model 1's three LightGBM models cost 90 s for 200 rows at 9 columns while
Model 2 cost 4 s. `n_jobs` explains rows in parallel processes
(`-1` for all cores): worth it for a run of minutes, slower than `1` for a run
of seconds, because the processes take seconds to start.

**Global importance** is `np.abs(values).mean(axis=0)` per column and
cohort, as in the example. It credits a column the model uses even when
permutation importance cannot see it behind a correlated neighbor.

**Plots.** The `Explanation` keeps shap's plots; one cohort at a time, with
`display_data` labelling the string levels:

```python
import tempfile
from pathlib import Path

import matplotlib
import shap

matplotlib.use("Agg")
out = Path(tempfile.mkdtemp())
explanation = explanations["Model 2"]
for cohort in explanation.output_names:
    shap.plots.bar(explanation[:, :, cohort], show=False)
    matplotlib.pyplot.savefig(out / f"shap_bar_{cohort}.png", bbox_inches="tight")
    matplotlib.pyplot.close()
    shap.plots.beeswarm(explanation[:, :, cohort], show=False)
    matplotlib.pyplot.savefig(out / f"shap_beeswarm_{cohort}.png", bbox_inches="tight")
    matplotlib.pyplot.close()
print(sorted(path.name for path in out.iterdir()))
```

```text
['shap_bar_n_elementary.png', 'shap_bar_n_highschool.png', 'shap_bar_n_kindergarten.png', 'shap_beeswarm_n_elementary.png', 'shap_beeswarm_n_highschool.png', 'shap_beeswarm_n_kindergarten.png']
```

## 6. Hazards

- **The exposure.** Pass the same exposure the model would get at prediction
  time, taken by the split's positions; the bootstrap and permutation
  importance score `predict(X, exposure)`, and SHAP explains it. A model
  fitted without an exposure ignores the argument.
- **Column order.** `background` is aligned to `X`'s columns by name (shap
  itself pairs columns by position), and a missing column raises `KeyError`.
- **Categorical columns.** A pandas `category` column is restored with the
  union of `X`'s and `background`'s levels, so a background-only level reaches
  the model instead of becoming NaN. A missing value in a string column
  reaches the model as NaN.
- **Large integers.** Values travel as float64 through shap, so an integer
  column above 2^53 loses precision; counts are far below.
- **Cost.** `2 ** n_columns` evaluations per row for `"exact"`, each over the
  whole background: subsample `X` (a few hundred rows) and `background` (a few
  dozen), count the raw columns before choosing the method, and use `n_jobs`
  only when a run is long.
- **One column per metric.** scikit-learn's `mean_poisson_deviance`,
  `root_mean_squared_error` and `mean_absolute_error` score one target column;
  `cohort_log_loss` ignores the total; `cohort_poisson_deviance` scores the
  whole count matrix.
- **Correlated columns** split permutation importance between them and can
  hide a column the model needs; read permutation and mean |SHAP| together.

## 7. What Remains

- The notebook plan's PR D logs the three tables and the plots per run
  (`log_importance`), on a seeded validation subsample whose sizes it chooses
  from the costs above.
- Importance of engineered features (one-hot pieces, interactions) is out of
  scope: raw columns only.
