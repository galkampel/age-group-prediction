# Direct Cohort Model (Model A)

> **Two implementations.** §0 describes the rebuilt
> `age_group_prediction.modeling.CountModel`, and §0.6 the classes that
> fit every cohort from the raw table. They are built and tested, but nothing
> calls them yet. §1–§10 describe the original
> `models/direct_cohort.py`, which `experiment/` and `tracking/` still run.
> That code is deleted once all three models are rebuilt
> ([plan](MODEL_REIMPLEMENTATION_PLAN.md)).

## 0. The Rebuilt Model (`modeling/count_model.py`)

> **2026-10-07:** the class is `CountModel`, renamed from `DirectCohortModel` because it fits
> any one count column: a cohort's here, and Model 2's total
> ([TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md) P2).
> This file keeps its name; "Model A" stays the name of the direct approach.

One regressor for **one** cohort's child count. Cohorts are independent, so
each gets its own instance, features and hyperparameters.
- **Any regressor** with a Poisson or Gaussian loss, given as `estimator`:
  `LGBMRegressor(objective="poisson")`, `HistGradientBoostingRegressor(loss="poisson")`,
  `PoissonRegressor()`, `LGBMRegressor(objective="regression")`, … The loss and
  the hyperparameters live on the estimator; a tuner reaches them by nested
  names (`estimator__n_estimators`) through `set_params`. Nothing is searched in `fit`.
- **`X` is the raw table when `feature_transformer` is given**, otherwise the
  finished design matrix. The model fits a copy of the transformer on the rows
  it is fitted on and transforms new rows with it
  ([Feature transformations §8.1](FEATURE_TRANSFORMATIONS.md#81-model-a--directcohortmodel-lightgbm)).
- **The exposure is optional** (`use_exposure=True`). The raw apartment count
  is passed to `fit` and `predict`; the estimator learns the cohort's rate
  **per apartment**, as a weighted regression of `y / exposure`, and `predict`
  multiplies it back into a count (§0.1). For a Poisson loss this is the
  offset model $\log \mu = \log E + F(x)$; for a Gaussian loss it is least
  squares of the count with a variance proportional to the exposure.

### 0.1 The model: the exposure as a weighted regression of the per-apartment rate

**Setup.** Building $b$ has $E_b > 0$ apartments (the exposure) and $y_b \in \{0, 1, 2, \dots\}$
children of one cohort, with features $x_b$. The quantity the model learns is the
**average number of children of the cohort per apartment**, the rate

$$\lambda_b = f(x_b) > 0, \qquad \text{so that the building's expected count is } \mu_b = E_b\,\lambda_b .$$

For a log-link (Poisson) model $f(x) = e^{F(x)}$, where $F$ is the raw score (the sum of
the trees in LightGBM, $\beta_0 + x\beta$ in a GLM).

**(A) The offset formulation (what LightGBM's `init_score` implements).**
$y_b \sim \text{Poisson}(\mu_b)$ with $\log \mu_b = \log E_b + F(x_b)$. Dropping the
$\log y_b!$ term, which does not depend on $F$, the negative log-likelihood is

$$\mathcal{L}_{\text{off}}(F) = \sum_b \Big[\mu_b - y_b \log \mu_b\Big]
= \sum_b \Big[E_b e^{F(x_b)} - y_b F(x_b)\Big] \;-\; \sum_b y_b \log E_b .$$

**(B) The weighted-rate formulation (what the generalized model does).**
Regress the observed rate $r_b = y_b / E_b$ on $x_b$ with the estimator's own Poisson
loss and `sample_weight` $w_b = E_b$. The Poisson loss of a non-negative real target $r$
with mean $\lambda$ is $\lambda - r \log \lambda$ (the Poisson deviance up to terms free
of $\lambda$; scikit-learn and LightGBM define it for any real $r \ge 0$, so a fractional
rate is a valid target). The weighted objective is

$$\mathcal{L}_{\text{rate}}(F) = \sum_b w_b \Big[\lambda_b - r_b \log \lambda_b\Big]
= \sum_b E_b \Big[e^{F(x_b)} - \frac{y_b}{E_b} F(x_b)\Big]
= \sum_b \Big[E_b e^{F(x_b)} - y_b F(x_b)\Big].$$

**(C) Equivalence.** Comparing the two,

$$\mathcal{L}_{\text{off}}(F) = \mathcal{L}_{\text{rate}}(F) - \sum_b y_b \log E_b ,$$

and the last term is a constant in $F$. The two objectives therefore have the same
minimizer, and more strongly the same per-row gradient and Hessian in the raw score,

$$\frac{\partial \mathcal{L}}{\partial F_b} = E_b e^{F_b} - y_b = \mu_b - y_b,
\qquad
\frac{\partial^2 \mathcal{L}}{\partial F_b^2} = E_b e^{F_b} = \mu_b ,$$

which is all a gradient-boosting step uses to build a tree. The starting point also
agrees: with weights, LightGBM's `boost_from_average` starts at the weighted mean rate
$\log\big(\sum_b w_b r_b / \sum_b w_b\big) = \log\big(\sum_b y_b / \sum_b E_b\big)$, exactly
the intercept $F_0 = \log(\sum y / \sum E)$ the old code put into `init_score`. So the
trees are the same, and the prediction is

$$\hat\mu_b = E_b \,\hat\lambda_b = E_b \cdot \texttt{estimator\_.predict}(x_b),$$

which is why `predict` multiplies by the exposure instead of adding $\log E_b$ to a raw
score. The equivalence is of the likelihoods. An estimator that also penalizes and
*normalizes* `sample_weight` scales its penalty differently: scikit-learn's
`PoissonRegressor(alpha=…)` divides the weighted loss by $\sum_b w_b = \sum_b E_b$, so its
`alpha` acts as `alpha` $\times \bar E$ in the offset model (measured 2026-10-05: with
`alpha=1` the weighted-rate coefficients match the offset model's at `alpha` $= \bar E = 47.6$
to $10^{-4}$, and at `alpha=0` the two agree). LightGBM and `HistGradientBoostingRegressor`
use the weights as given, so their penalties are unchanged. No single rescaling of the
weights fixes both: dividing by $\bar E$ would change LightGBM's hessians and so its
`reg_lambda` and `min_child_weight`. (Measured on LightGBM, 2000 rows, 50 trees: the two predictions agree to
$1.5 \times 10^{-8}$ relative; `min_child_samples` counts rows in both, so the only
difference is floating point.)

**(D) The same transformation under a Gaussian likelihood.** The derivation runs as for
Poisson, with the squared-error loss in place of the Poisson one; $f$ now enters the
mean directly (identity link), and $f_b$ is short for $f(x_b)$.

*(D1) The model.* The building's count is Gaussian with a mean proportional to the
exposure and a variance that grows with it, as a count's does:

$$y_b \sim \mathcal{N}\big(\mu_b,\ \sigma^2 E_b\big), \qquad \mu_b = E_b\, f_b .$$

Its negative log-likelihood is

$$\mathcal{L}_{\text{Gauss}}(f, \sigma^2) = \sum_b \Big[\frac{(y_b - E_b f_b)^2}{2\sigma^2 E_b}
+ \tfrac12 \log\big(2\pi\sigma^2 E_b\big)\Big].$$

*(D2) The weighted-rate formulation.* Regress the rate $r_b = y_b / E_b$ on $x_b$ with
the estimator's squared-error loss $\tfrac12 (r - f)^2$ and `sample_weight` $w_b = E_b$:

$$\mathcal{L}_{\text{rate}}(f) = \tfrac12 \sum_b w_b \big(r_b - f_b\big)^2
= \tfrac12 \sum_b E_b \Big(\frac{y_b}{E_b} - f_b\Big)^2
= \tfrac12 \sum_b \frac{\big(y_b - E_b f_b\big)^2}{E_b}.$$

*(D3) Equivalence.* Comparing the two,

$$\mathcal{L}_{\text{Gauss}}(f, \sigma^2) = \frac{1}{\sigma^2}\,\mathcal{L}_{\text{rate}}(f)
+ \tfrac12 \sum_b \log\big(2\pi\sigma^2 E_b\big),$$

and the last term does not depend on $f$. For every $\sigma^2$ the two objectives
therefore have the same minimizer in $f$, so the prediction never needs $\sigma^2$. Per
row, in $f_b$,

$$\frac{\partial \mathcal{L}_{\text{rate}}}{\partial f_b} = E_b f_b - y_b = \mu_b - y_b,
\qquad
\frac{\partial^2 \mathcal{L}_{\text{rate}}}{\partial f_b^2} = E_b :$$

the gradient has the Poisson form of (C), and the Hessian is the exposure instead of
$\mu_b$. The starting point agrees too: a constant $f$ minimizes
$\tfrac12\sum_b (y_b - E_b f)^2 / E_b$ at $f = \sum_b y_b / \sum_b E_b$, the weighted
mean rate a booster starts from. The prediction is again
$\hat\mu_b = E_b \cdot \texttt{estimator\_.predict}(x_b)$. (Checked numerically: the
weighted-rate fit equals the weighted least-squares fit of $y$ on $[E, E x]$ with
weights $1/E$ to $10^{-16}$, and the full maximum-likelihood fit of (D1), $\sigma^2$
included, to $10^{-8}$; pinned by
`test_the_gaussian_weighted_rate_is_least_squares_of_the_count`.)

*(D4) Why not the additive offset, and why the weight.* With an identity link, the
additive offset of (A) gives the mean $f(x_b) + \log E_b$, which is not proportional to
the apartments; that is why the old model rejected `use_exposure` with a Gaussian loss.
The weighted rate scales the mean instead, so the mean structure is the intended one
(expected count proportional to apartments, $f$ the average children per apartment) and
`use_exposure=True` is meaningful for a Gaussian loss too. Without the weight (plain
regression of $r_b$) the mean would be the same, but each building would count equally,
a $\operatorname{Var}(y_b) \propto E_b^2$ assumption. The one Gaussian-specific caveat
is that $f(x_b)$ can be negative, as it already could without an exposure.

**(E) Why not a residual.** Fitting $y_b - \mu^{(0)}_b$ (the count minus an initial
prediction) is only meaningful for a squared-error loss, where the residual is again a
Gaussian target; for a Poisson loss the residual is not a count and can be negative, so
a residual-based path would fit a different model than the LightGBM one. The model uses
(B) for every estimator instead.

**Consequences.**
- **If the exposure is not a column of `X`,** doubling it doubles the
  prediction exactly: $\hat\mu(x, 2E) = 2\,\hat\mu(x, E)$.
- **If `X` includes `n_apartments`,** $F$ can learn departures from
  proportionality.
- **Without the exposure,** the estimator fits `y` directly: $\log \mu_b = F(x_b)$
  for a Poisson loss, $\mu_b = f(x_b)$ for a Gaussian one.

**Correct inputs.**

| Argument | Pass | Not |
|---|---|---|
| `X` | The raw table with `feature_transformer`, otherwise transformed features, one row per building (either may include `n_apartments`) | IDs or targets as features |
| `y` | The raw cohort count $y_b$ | The rate $y_b / E_b$, or $\log y_b$: the model divides by the exposure itself |
| `exposure` | The raw $E_b$, in the same row order as `X` | $\log E_b$, which would be divided into `y` as if it were apartments |

The output $\hat\mu_b$ is an expected **count**, not a rate.

### 0.2 API

| Member | What it takes or gives |
|---|---|
| `CountModel(*, estimator, use_exposure=False, feature_transformer=None)` | Settings only, stored verbatim. `estimator` is required: any object with `fit(X, y, sample_weight=None)` and `predict(X)` (the `Regressor` protocol). Build it with `n_jobs=1` for LightGBM, since more OpenMP threads crash alongside torch on macOS. LightGBM ignores `subsample` unless `subsample_freq >= 1` is set on the estimator too |
| `fit(X, y, exposure=None)` | `y` is the raw cohort count; `exposure` the raw $E$. With `use_exposure=False` a passed exposure is ignored, so one exposure can go to every model and a tuner can compare with and without |
| `predict(X, exposure=None)` | Expected counts, a 1-D array. It follows how the model was fitted: an exposure is required if it was fitted with one, and ignored otherwise |
| `evaluate(y_true, y_pred, metric)` | One `float`. `metric` is a `Metric`: `POISSON_DEVIANCE`, `RMSE`, `MAE`, or a custom `Metric(name, function, greater_is_better=False)` |
| `estimator_`, `use_exposure_`, `feature_transformer_` | Fitted state: the fitted copy of `estimator`, whether it was fitted with the exposure, and the fitted copy of `feature_transformer` (`None` without one) |

```python
from lightgbm import LGBMRegressor
from age_group_prediction.modeling import CountModel
from age_group_prediction.preprocessing import ExposureTransformer
from age_group_prediction.scoring import POISSON_DEVIANCE
from age_group_prediction.utils import take_rows

# table and tree: Feature transformations §8.0 and §8.1; train_index and
# test_index: Splitter.train_test_indices on that table. The exposure is built
# before the split and taken by the same positions (§0.6).
exposure = ExposureTransformer("n_apartments").fit_transform(table)
train_df, test_df = take_rows(table, train_index), take_rows(table, test_index)
model = CountModel(
    estimator=LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1),
    use_exposure=True,
    feature_transformer=tree,  # fitted on train_df inside fit
).fit(train_df, train_df["n_kindergarten"], exposure=take_rows(exposure, train_index))
predictions = model.predict(test_df, exposure=take_rows(exposure, test_index))
score = model.evaluate(test_df["n_kindergarten"], predictions, POISSON_DEVIANCE)
```

### 0.3 Errors

The model checks only what would otherwise pass silently. Each of these raises
`ValueError`:
- `exposure` missing when the model uses one, which would drop the offset
  silently (one passed to a model without the offset is ignored);
- an exposure that is not one-dimensional, which would broadcast into an
  $(n, n)$ prediction;
- an exposure whose length differs from `X`'s, at fit and at predict: numpy
  would broadcast a length-1 exposure to every row at predict silently.

The exposure checks are `BaseAgeGroupModel._check_exposure`, shared by every
model with an offset.

The exposure's values are checked where the data is prepared, not here:
`preprocessing.ExposureTransformer` returns the column as floats and raises
unless every value is strictly positive and finite. LightGBM fits a zero, NaN
or infinite exposure silently (a non-finite rate or weight;
`HistGradientBoostingRegressor` raises), so build the exposure with it.

Everything else is left to the estimator, with its own error:
- a regressor whose `fit` takes no `sample_weight` fails at `fit` with
  `use_exposure=True` (Python's `TypeError` for a plain `fit(X, y)`; a
  scikit-learn `Pipeline` raises `ValueError`); there is no pre-check;
- an unknown objective, and an all-zero `y` under a Poisson loss (LightGBM
  raises; not every estimator does).

### 0.4 What Changed From §1–§10

| Original | Rebuilt |
|---|---|
| One instance, three cohorts, and Optuna tuning inside `fit` | One instance per cohort; hyperparameters fixed on the estimator and tuned from outside (`estimator__…`) |
| LightGBM only, families `poisson`, `nb2` (custom gradient), `normal` | Any regressor with a Poisson or Gaussian loss, given as `estimator`; NB2 removed |
| A `FeatureSpec` and fitted preprocessing owned by the model | An optional `feature_transformer`, fitted inside `fit` on the training rows |
| No exposure offset | Optional `use_exposure`, as a weighted regression of the per-apartment rate (§0.1), for Poisson and Gaussian losses |
| Bootstrap draws, intervals, pointwise log probabilities, `PredictionResult` | Means only |
| State bundles, `get_metadata`, `configuration_record` | `get_params()` and the public fitted attributes |

The first rebuilt version (PR #6) mirrored ten LightGBM hyperparameters and
an `objective` in its constructor and put the offset into LightGBM's
`init_score`; the generalization
([plan](DIRECT_COHORT_GENERALIZATION_PLAN.md), PR #12) replaced both with the
`estimator` and the weighted rate, which give LightGBM the same trees (§0.5).

### 0.5 Evidence For The Exposure

**The weighted rate is the offset model.** On LightGBM (2000 rows, 50 trees)
the weighted-rate prediction and LightGBM's own `init_score` offset agree to
$1.5 \times 10^{-8}$ relative. `test_the_weighted_rate_equals_lightgbms_offset`
pins it (`rtol=1e-6`), and `test_the_gaussian_weighted_rate_is_least_squares_of_the_count`
pins §0.1 (D) at `rtol=1e-8` (measured: the weighted least-squares fit of $y$ on
$[E, Ex]$ with weights $1/E$, to $10^{-16}$).

**Does the exposure help?** We ran 10 simulated populations, split by
neighborhood, with untuned defaults and `n_apartments` kept as a feature. The
exposure lowered held-out Poisson deviance by about 4% for kindergarten and
high school, and not at all for elementary. That is weak evidence
([plan, Step 2.4](MODEL_REIMPLEMENTATION_PLAN.md)). Re-check it once the models
are tuned.

The same run through `IndependentCohortModels` (§0.6) gives identical
per-cohort numbers. The summed prediction's deviance against
`n_children_total` is 4% lower with the exposure, in 8 of 10 populations
([multi-cohort plan, A4](MULTI_COHORT_MODELS_PLAN.md#a4-smoke-run-and-docs)).

### 0.6 Every Cohort From The Raw Table: `IndependentCohortModels`

`IndependentCohortModels(cohort_models)` maps each cohort (a column of `y`) to
its model, usually a `CountModel` with its own `feature_transformer`,
so each cohort keeps its own features and hyperparameters. `fit` fits a copy
of each on the raw table and its column of `y`, stored in `cohort_models_`;
`predict` returns a DataFrame with one column per cohort, in `y`'s order,
indexed like `X`.

Import it from `age_group_prediction.modeling`: the package root still
exports the **original** classes of §1–§10.

**The data flow.**
1. **Row-wise preprocessing, on the full table, before the split.**
   `ShareTransformer` turns the room counts into shares.
   `ExposureTransformer` returns the exposure as floats. It raises for a zero,
   negative, infinite or NaN value, which the model would otherwise accept
   silently (§0.3).
   Neither learns anything, so nothing leaks from the test rows, and a bad
   test row fails before any model is fitted.
2. **Split by neighborhood** with `Splitter.train_test_indices`. It returns
   row positions; take the table, the targets and the exposure (and the
   groups, if a `cv` follows) by the same positions with `take_rows`.
3. **Fit on the training rows.** Each model fits its feature transformer
   inside its own `fit`, on those rows only.

```python
from lightgbm import LGBMRegressor
from age_group_prediction.modeling import CountModel, IndependentCohortModels
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from age_group_prediction.scoring import POISSON_DEVIANCE
from age_group_prediction.splitting import Splitter
from age_group_prediction.utils import take_rows

COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]

# 1. On the full table: raw_table is the simulator's or the real data
table = ShareTransformer(
    ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
).fit_transform(raw_table)
exposure = ExposureTransformer("n_apartments").fit_transform(table)

# 2. The split: row positions, applied to every array alike
train_index, test_index = Splitter("grouped").train_test_indices(
    table, table["neighborhood_id"], test_size=0.2, random_state=0
)
X_train, X_test = take_rows(table, train_index), take_rows(table, test_index)
Y_train, Y_test = take_rows(table[COHORTS], train_index), take_rows(table[COHORTS], test_index)
exposure_train, exposure_test = take_rows(exposure, train_index), take_rows(exposure, test_index)

# 3. One model per cohort, each with its own transformer; tree is Feature transformations §8.1
def lightgbm():
    return LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1)

model = IndependentCohortModels({
    "n_kindergarten": CountModel(estimator=lightgbm(), use_exposure=True, feature_transformer=tree),
    "n_elementary": CountModel(estimator=lightgbm(), feature_transformer=tree),  # ignores the exposure
    "n_highschool": CountModel(estimator=lightgbm(), use_exposure=True, feature_transformer=tree),
}).fit(X_train, Y_train, exposure=exposure_train)
predictions = model.predict(X_test, exposure=exposure_test)  # a DataFrame
scores = {
    cohort: model.evaluate(Y_test[cohort], predictions[cohort], POISSON_DEVIANCE)
    for cohort in COHORTS
}
```

**Rules.**

| Rule | Why |
|---|---|
| **The same exposure goes to every cohort.** A model with `use_exposure=False` ignores it; one fitted with the offset raises without it | One exposure serves every model, and a tuner can compare with and without the offset on one fixed exposure. A forgotten exposure would drop the offset silently |
| **Rows are paired by position**, not by index, as in scikit-learn. Nothing checks the index | The splitter returns row positions, and every array is taken by them, the exposure included, as above |
| **The keys of `cohort_models` equal `y`'s columns**, each once, or `fit` raises `ValueError` | A cohort would otherwise be dropped silently, or a duplicated column would reach its model as a DataFrame |
| **The targets come from `y` only.** With the default `remainder="drop"`, as in `tree`, `X` may keep the target columns: each transformer reads only its plans' columns, and `predict` needs none. With `remainder="passthrough"`, drop the targets from `X` first | Otherwise the targets become features, and `predict` fails on a table without them |
| **Each cohort is tuned on its own.** A nested name such as `cohort_models__n_kindergarten__estimator__learning_rate` raises `AttributeError`; tune each cohort's `CountModel` (whose nested names, e.g. `estimator__learning_rate`, work), then assemble the mapping, or replace it with `set_params(cohort_models=...)` | The cohorts are independent, so no study tunes them together |
| **Set before `fit`.** `IndependentCohortModels` and each `CountModel` fit copies (of the cohort models, the estimator and the transformer), so a setting changed with `set_params` after `fit` reaches only the next `fit`; `predict` follows the fitted copies | The templates stay unfitted and can be reused |
| **`cohort_models` is the template, `cohort_models_` the fitted copies**, as `estimator` and `estimator_`. A plain `dict` suffices: the output follows `y`'s column order at fit, whatever the mapping's order | scikit-learn's contract: `get_params`, `set_params` and `clone` read the template stored verbatim; fitting copies keeps it unfitted, so a refit equals a fresh fit. `OrderedDict` adds only order-sensitive `==` and `move_to_end`, neither used |
| **Per-cohort scores are a loop**, as above; how to combine the cohorts is the caller's choice | `mean_poisson_deviance` takes one column |

## 1. Statistical Model

For building $b$ with transformed features $x_b$ and cohorts
$k \in \{\text{kindergarten}, \text{elementary}, \text{highschool}\}$, each cohort
count $C_{b,k}$ gets its own tree ensemble $g_k$ and a family chosen for the
whole model instance:

| `family` | Cohort distribution | LightGBM objective | Mean link |
|---|---|---|---|
| `poisson` (default) | $C_{b,k} \sim \operatorname{Poisson}(\mu_{b,k})$ | built-in `poisson` | log, applied by LightGBM |
| `nb2` | $C_{b,k} \sim \operatorname{NB2}(\mu_{b,k}, \alpha_k)$, $\operatorname{Var}=\mu+\alpha_k\mu^2$ | custom gradient and Hessian (`nb2_gradient_hessian`) | log; raw scores clipped to $[-30, 30]$ and exponentiated |
| `normal` | $C_{b,k} \sim \mathcal N(\mu_{b,k}, \sigma_k^2)$ | built-in L2 `regression` | identity |

The cohorts are modeled **independently**: nothing ties their errors together,
and the total is not modeled separately. Predictions are

$$
\widehat\mu_{b,k} = \max\bigl(g_k(x_b), \text{minimum\_mean}\bigr),
\qquad
\widehat\mu_b = \sum_k \widehat\mu_{b,k},
\qquad
\widehat p_{b,k} = \frac{\widehat\mu_{b,k}}{\widehat\mu_b}.
$$

Cohort means are clipped at `minimum_mean` (default $10^{-8}$) for every
family, including Normal. Because the total is defined as the sum of cohort
means, reconciliation is exact by construction. If all three means were zero,
the probabilities would fall back to uniform.

## 2. Features

Model A requires a `tree` feature spec and refuses an exposure offset. The
default `DEFAULT_TREE_FEATURE_SPEC` passes the numeric features unscaled,
includes `n_apartments` as an ordinary predictor (trees learn size effects
themselves), and one-hot encodes `school_status`. See
[FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md).

## 3. Fitting

`fit(train_df, feature_spec=..., rng=...)` runs, for the training partition only:

1. **Tuning folds.** `make_validation_folds(train_df,
   config=direct_cohort_config.tuning_folds)` builds known-neighborhood folds
   inside the training data (package default: 5 folds; `configs/modeling.toml`
   does not set these). A fresh feature transformer is fitted on each fold's fit
   rows, and the same fold matrices are reused for all cohorts.
2. **One Optuna study per cohort.** Each uses a seeded TPE sampler and
   `n_trials` (30 in `[tuning]`), `n_jobs=1`, and no timeout. The objective is
   the mean held-out score across folds:
   - Poisson and NB2: mean negative log probability;
   - Normal: RMSE.

   Pruning is off by default. When enabled, only completed trials can win, and a
   study with no completed trial is an error.
3. **Final refit.** Each cohort is refitted on the full training data with its
   best parameters and a purpose-derived seed; the fitted LightGBM `Booster` is
   kept.
4. **Ancillary parameters.**
   - NB2: the tuned `dispersion` ($\alpha_k$) is kept per cohort.
   - Normal: $\sigma_k$ is the RMSE of cross-fitted out-of-fold residuals using the
     selected parameters, floored at `minimum_scale`.
5. The training frame is retained in memory for bootstrap uncertainty.

### Search space (`[direct_cohort_search_space]`)

| Hyperparameter | Range | Scale |
|---|---|---|
| `max_depth` | 3–8 | integer |
| `num_leaves` | 7–63, capped at $2^{\text{max\_depth}}$ | integer |
| `min_child_samples` | 5–40 | integer |
| `learning_rate` | 0.01–0.2 | log |
| `n_estimators` | 50–400 | integer |
| `reg_alpha`, `reg_lambda` | $10^{-8}$–10 | log |
| `min_split_gain` | 0–1 | linear |
| `subsample`, `colsample_bytree` | 0.7–1.0 | linear (bagging enabled only when `subsample` < 1) |
| `dispersion` (NB2 only) | $10^{-4}$–5 | log |

Estimators run with `deterministic=True`, `force_col_wise=True`, and
`lightgbm_n_jobs` threads (default 1) for reproducibility.

### Hyperparameter tuning in the wider workflow

- **Nested in cross-validation.** The experiment runner calls `fit` on each
  outer fold's fit rows, so the studies above run separately inside every fold
  on training-only inner folds. The outer validation fold never influences the
  chosen hyperparameters.
- **Re-run for the final refit.** The final refit calls `fit` on the whole
  training partition, so tuning repeats there with seeds derived from the
  frozen master seed.
- **Reproducibility.** Studies use a seeded TPE sampler, `n_jobs=1`, and no
  timeout; only completed trials can be selected. With pruning enabled,
  `trial.report` is called after each inner fold.
- **Evidence.** `metadata["diagnostics"]["tuning"][cohort]` holds every trial's
  parameters, state, value, and per-fold intermediate values, next to
  `diagnostics["search_space"]`. MLflow logs `tuning/{cohort}/best_value` and
  `tuning/fold_{i}/{cohort}_trials.csv` for each candidate run.
- **Not tuned.** The family, the feature spec, `minimum_mean`, and
  `bootstrap_replicates` are configuration or candidate identity, not
  hyperparameters.

## 4. Prediction

`predict(eval_df, prediction_config=..., rng=...)` returns a `PredictionResult`
with cohort and total means and probabilities, plus optional items:

### Pointwise log probabilities (`include_pointwise_log_probabilities`)

Scope is **`marginal`**. Requires observed targets in `eval_df`.

| Key | Poisson | Normal | NB2 |
|---|---|---|---|
| each cohort | Poisson log mass | Normal log density | NB2 log mass with $\alpha_k$ |
| `total` | Poisson log mass at $\widehat\mu_b$ (a sum of independent Poissons is Poisson) | Normal log density with scale $\sqrt{\sum_k \sigma_k^2}$ | exact finite convolution of the three independent NB2 cohort distributions |

The keys double-count: the total is scored from the same cohort predictions, so
the entries do **not** sum to a joint score. Never compare these values with
the conditional models' `sequential_joint` scores. Normal densities are also not
comparable with count log masses.

### Parametric distributions

Poisson declares Poisson for cohorts and total; Normal declares Normal with the
scales above; NB2 declares NB2 per cohort and **no** total entry, because a sum
of independent NB2 variables is not NB2.

### Predictive draws and intervals (`n_predictive_draws`, `interval_levels`)

Uncertainty comes from a **neighborhood-cluster bootstrap**:

1. Draw $R=\max(\text{bootstrap\_replicates}, \text{n\_predictive\_draws})$
   resamples of whole neighborhoods from the training frame
   (`bootstrap_replicates` = 200).
2. For each resample, refit the feature transformer and each cohort's trees with
   the **already selected** hyperparameters (no re-tuning).
3. Draw one outcome per building and cohort from the family (Poisson, NB2, or
   Normal). Total draws are the sums of the cohort draws.

The first `n_predictive_draws` columns are reported; central intervals use all
$R$ draws. Normal draws are continuous and can be negative.

## 5. Selection And Comparison

The canonical candidate is `direct-poisson` (family from `[direct_cohort]`,
default `poisson`). Within the direct-cohort approach,
`direct_cohort_selection_policy` ranks candidates by
`independent_cohort_joint_nll`: the sum of the three per-cohort parametric
NLLs, with the total excluded to avoid double counting. Ties are broken by mean
cohort RMSE. Selection refuses to rank continuous (Normal) and discrete
(Poisson/NB2) candidates together.

Across families, `select_cross_family_winner` compares the direct winner with
the conditional winner by composition log loss, then mean cohort RMSE, then
mean cohort MAE; it never compares Model A's likelihoods with the conditional
models' joint likelihood. Details:
[CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md) and
[EVALUATION_AND_METRICS.md](EVALUATION_AND_METRICS.md#5-likelihood-comparability);
the one-time refit and holdout scoring are in
[FINAL_EVALUATION.md](FINAL_EVALUATION.md).

## 6. Persistence And Serving

`to_state_bundle()` stores the configuration, each cohort's trees as
LightGBM's text model format (no pickle), selected and ancillary parameters,
tuning evidence, and the feature transformer state. It stores no training rows.

- Point predictions, pointwise scores, and parametric distributions need only
  the bundle.
- Draws and intervals refit on bootstrap resamples, so
  `DirectCohortModel.from_state_bundle(bundle, train_df=train_df)` must receive
  the original training frame; its hashes are checked.
- `age_group_prediction.models` imports LightGBM before torch because LightGBM
  can crash restoring boosters after torch's OpenMP runtime loads. The canonical
  `direct-poisson` MLflow model predates that fix and needs `import lightgbm`
  before `mlflow.pyfunc.load_model(...)` in a fresh process.

## 7. Configuration

| TOML section / field | Default | Effect |
|---|---|---|
| `[direct_cohort] family` | `poisson` | `poisson`, `nb2`, or `normal` |
| `[direct_cohort] bootstrap_replicates` | 200 | Minimum bootstrap refits for draws/intervals |
| `[direct_cohort] lightgbm_n_jobs` | 1 | LightGBM thread count |
| `[direct_cohort] minimum_mean`, `minimum_scale` | $10^{-8}$ | Mean clip and Normal scale floor |
| `[tuning] n_trials`, `n_jobs`, `enable_pruning` | 30, 1, false | Optuna study settings (shared) |
| `[direct_cohort_search_space]` | see Section 3 | Hyperparameter bounds |
| `tuning_folds` (code only) | 5 folds | Inner tuning folds; not set by the TOML loader |

## 8. Metadata

`model.metadata` reports the implementation version, likelihood and
parameterization text, selected hyperparameters per cohort, dependency
versions, uncertainty method, and diagnostics: `family`, `lightgbm_objective`,
ancillary parameters, full tuning trials, the search space, and
`normal_mean_policy` (`clip to positive minimum`).

## 9. Pitfalls

- **Do not compare marginal scores with joint scores.** Use the selection
  policies and cross-family rule, which respect likelihood comparability.
- **Probabilities are derived, not modeled.** They are ratios of independent
  cohort means, so they are not calibrated composition probabilities.
- **Normal is a diagnostic comparator.** Its draws are not counts, and it is
  not ranked against count families.
- **Bootstrap intervals are slow.** $R \ge 200$ refits of three tree ensembles;
  request draws only when needed.
- **Reloaded models need `train_df` for uncertainty.** Without it only point
  predictions and pointwise scores are available.

## 10. Tests And Related Documents

- Tests: `tests/unit/test_direct_cohort.py`, `tests/unit/test_model_state_bundles.py`,
  `tests/unit/test_tuning.py`, and real-model checks in
  `tests/validation/test_experiment_real_models.py`.
- [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md),
  [MODELING_GUIDE.md](MODELING_GUIDE.md) Sections 6–11,
  [INDEPENDENT_TOTAL_PROBABILITY_MODEL.md](INDEPENDENT_TOTAL_PROBABILITY_MODEL.md),
  [BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md](BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md),
  [EVALUATION_AND_METRICS.md](EVALUATION_AND_METRICS.md),
  [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md).
