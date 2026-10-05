# Independent Total And Probability Model (Model B)

> **Two implementations.** §0 describes the rebuilt
> `age_group_prediction.modeling.IndependentTotalProbabilityModel` and the
> three classes it is made of. They are built and tested, but nothing calls
> them yet. §1–§12 describe the original
> `models/independent_total_probability.py`, which `experiment/` and
> `tracking/` still run. That code is deleted once all three models are
> rebuilt ([plan](MODEL_REIMPLEMENTATION_PLAN.md)).

## 0. The Rebuilt Model (`modeling/independent_total_probability.py`)

Two independent models, multiplied: a count regression for a building's
**total** children and a Dirichlet regression for its **cohort shares**, each
with its own `feature_transformer`, fitted on the raw table.
The shares can be calibrated afterwards by a temperature fitted on
out-of-fold logits. Means only; nothing is searched inside `fit`.
Decisions and evidence: [MULTI_COHORT_MODELS_PLAN.md](MULTI_COHORT_MODELS_PLAN.md)
(§5, §7 and §9 B1–B9).

### 0.1 The Model

For building $b$ with apartment count $A_b$, feature vectors $x_b$ (total
stage) and $w_b$ (composition stage), cohort counts $C_{b,k}$,
$k = 1,\dots,K$, and total $Y_b = \sum_k C_{b,k}$:

**Total** (`TotalChildrenModel`): the GLM of §1 without clipping or a floor,

$$
\log\mu_b = \log A_b + \beta_0 + x_b^\top\beta,
\qquad
Y_b \sim \operatorname{Poisson}(\mu_b)
\;\;\text{or}\;\;
Y_b \sim \operatorname{NB2}(\mu_b, \alpha),\ \operatorname{Var}(Y_b) = \mu_b + \alpha\mu_b^2 .
$$

$\log A_b$ is the offset (`use_exposure=True`, the default; without it
there is no offset). For NB2, $\log\alpha$ is fitted with $\beta_0$ and
$\beta$ and kept at $\alpha \ge 10^{-6}$, the Poisson limit in practice
(below it the likelihood of data without overdispersion has no maximum).

**Composition** (`CohortProbabilityModel`): a Dirichlet regression of the
observed shares $s_{b,k} = C_{b,k} / Y_b$,

$$
s_b \sim \operatorname{Dirichlet}(\alpha_b),
\qquad
\alpha_{b,k} = \exp\!\left(a_k + w_b^\top\gamma_k\right),
\qquad
p_{b,k} = \frac{\alpha_{b,k}}{\sum_j \alpha_{b,j}} = \operatorname{softmax}(\log\alpha_b)_k ,
$$

one intercept and one coefficient vector per cohort (the common
parameterization: a shift of all intercepts changes the precision
$\sum_k \alpha_{b,k}$, not the mean, so the coefficients are identified).
A share of 0 has no density, so the shares are compressed toward the centre
before the fit, $s' = (s\,(N-1) + 1/K)/N$ over the $N$ training buildings
(Smithson & Verkuilen 2006, as R's `DirichletReg`). Every building needs a
child; the fit weighs buildings equally (one composition each).

**Objectives.** Each model minimizes the **mean** negative log-likelihood per
building plus $\tfrac{\lambda}{2}\lVert\beta\rVert^2$ on the feature
coefficients only ($\lambda$ = `l2_penalty`; intercepts and $\alpha$
unpenalized), with scipy's L-BFGS-B or BFGS on torch's likelihood and
gradient. A fit that does not converge, overflows or ends non-finite raises.

**Calibration** (`TemperatureCalibrator`, post-hoc): one temperature $T$
fitted on **out-of-fold** logits $\ell_{b,k} = \log\alpha_{b,k}$ of the
composition model and the same buildings' counts,

$$
p_{b,k}(T) = \frac{\exp(\ell_{b,k}/T)}{\sum_j \exp(\ell_{b,j}/T)},
\qquad
\hat T = \arg\min_{\log(1/T)\in(-10,\,10)}
\;-\frac{1}{N}\sum_{b}\sum_{k} s_{b,k}\,\log p_{b,k}(T),
$$

the cross-entropy of each building's observed composition under the scaled
shares, averaged over buildings (per building, as the Dirichlet fit), in
log space so that a sharp logit never underflows to $\log 0$. $T = 1$ is
the model's own prediction; $T > 1$ flattens it. The fitted value is always
used: no likelihood-ratio gate. The precision $\sum_k\alpha_{b,k}$ is a row
constant of $\ell_b$ and cancels, so the calibration is a power transform
of the mean shares alone.

**Combined prediction:**

$$
\widehat C_{b,k} = \widehat\mu_b\,\widehat p_{b,k}(\hat T),
\qquad
\sum_k \widehat C_{b,k} = \widehat\mu_b .
$$

### 0.2 API

Import from `age_group_prediction.modeling`; the package root exports the
**original** class of the same name (§1–§12).

| Class | Settings (defaults) | `fit` takes | Fitted state | `predict` returns |
|---|---|---|---|---|
| `TotalChildrenModel` | `family="poisson"` or `"nb2"`; `solver="lbfgs"` or `"bfgs"` (`nb2`: `lbfgs` only, the solver with bounds); `use_exposure=True`; `l2_penalty=0.0`; `max_iter=500`; `tol=1e-6` (bounds the gradient); `feature_transformer=None` | the raw table with `feature_transformer`, otherwise the design matrix; the totals; `exposure=` (raw apartments) | `intercept_`, `coef_`, `use_exposure_`, `dispersion_` (NB2 only), `feature_transformer_`, `feature_names_in_`, `n_features_in_` (of the design matrix) | the mean total, an array |
| `CohortProbabilityModel` | `solver`, `l2_penalty`, `max_iter`, `tol`, `feature_transformer` as above | `X` as above, a DataFrame of cohort counts (≥ 2 columns, every building with a child); an exposure is ignored | `intercept_` (K), `coef_` (d × K), `cohorts_`, `feature_transformer_`, the feature names | the mean shares, a DataFrame with `y`'s columns, rows summing to 1; `predict_logits` gives $\log\alpha$ |
| `TemperatureCalibrator` | none | out-of-fold logits and the counts of the same rows (DataFrames or arrays, paired by position) | `temperature_` | `softmax(logits / T)`, a DataFrame with the logits' columns and index |
| `IndependentTotalProbabilityModel` | `total_children_model`, `cohort_probability_model` (each with its own `feature_transformer`), `temperature_calibrator=None` (a fitted calibrator, or a `FrozenEstimator` of one) | the raw table, the cohort counts, `exposure=` for both models | `total_children_model_`, `cohort_probability_model_`, `temperature_calibrator_`, `cohorts_` | `total × shares`, a DataFrame with `y`'s columns at fit, indexed like `X` |

`CohortProbabilityModel.predict_logits(X)` transforms the raw table with its
fitted `feature_transformer_` and returns the logits: what a calibrator is fitted on
and what Model 2 feeds it at `predict`. Every model keeps the base signature
`fit(X, y, exposure=None)` and `predict(X, exposure=None)`, and `evaluate`.

### 0.3 Data Flow

1. **Row-wise preprocessing, on the full table, before the split.**
   `ShareTransformer` turns the room counts into shares; `ExposureTransformer`
   returns the exposure as floats and raises for a zero, negative, infinite
   or NaN value. Neither learns anything, so nothing leaks from the test rows.
2. **Split by neighborhood** with `Splitter.train_test_indices`: row
   positions, applied to the table, the targets, the groups and the exposure
   alike with `take_rows`.
3. **Calibrate on the training rows only.** Grouped folds on the training
   rows; each fold's copy of the composition model gives the held-out
   rows' logits; the calibrator is fitted on all of them.
4. **Fit Model 2 on the training rows** with the fitted calibrator, and
   predict the test rows.

```python
from sklearn.base import clone

from age_group_prediction.modeling import (
    CohortProbabilityModel, IndependentTotalProbabilityModel, TemperatureCalibrator,
    TotalChildrenModel,
)
from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
from age_group_prediction.scoring import COHORT_LOG_LOSS, POISSON_DEVIANCE
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
groups_train = take_rows(table["neighborhood_id"], train_index)
exposure_train, exposure_test = take_rows(exposure, train_index), take_rows(exposure, test_index)

# 3. The calibrator, on out-of-fold logits of the composition model;
#    total_base and cohort_probability_base are Feature transformations §8.2–§8.3
probability_model = CohortProbabilityModel(feature_transformer=cohort_probability_base)
logits_val, counts_val = [], []
for fit_index, val_index in Splitter("grouped").cv(n_splits=5, random_state=0).split(
    X_train, Y_train, groups_train
):
    fold = clone(probability_model).fit(take_rows(X_train, fit_index), take_rows(Y_train, fit_index))
    logits_val.append(fold.predict_logits(take_rows(X_train, val_index)))
    counts_val.append(take_rows(Y_train, val_index))
calibrator = TemperatureCalibrator().fit(pd.concat(logits_val), pd.concat(counts_val))

# 4. Model 2: the exposure goes to both models; the composition one ignores it
model = IndependentTotalProbabilityModel(
    total_children_model=TotalChildrenModel(feature_transformer=total_base),  # or family="nb2"
    cohort_probability_model=probability_model,
    temperature_calibrator=calibrator,  # FrozenEstimator(calibrator) if the model is cloned
).fit(X_train, Y_train, exposure=exposure_train)
predictions = model.predict(X_test, exposure=exposure_test)  # a DataFrame, total × shares
scores = {
    cohort: model.evaluate(Y_test[cohort], predictions[cohort], POISSON_DEVIANCE)
    for cohort in COHORTS
}
scores["total"] = model.evaluate(Y_test.sum(axis=1), predictions.sum(axis=1), POISSON_DEVIANCE)
scores["composition"] = model.evaluate(Y_test, predictions, COHORT_LOG_LOSS)  # per child
```

### 0.4 Rules

| Rule | Why |
|---|---|
| **The same exposure goes to both models.** The total model raises without it (`use_exposure=True`); the composition model ignores it | A forgotten exposure would drop the offset silently; one exposure serves every model (Model 1's rule too) |
| **Rows are paired by position**, not by index; nothing checks the index | The splitter returns positions and every array is taken by them |
| **Every building has a child, and `y` has at least two cohorts.** Counts are validated where the data is prepared, not in the models; a cohort with no child at all raises | The observation is a building's composition; an unobserved cohort would be fitted to the compressed floor silently |
| **The calibrator is fitted by the caller on out-of-fold logits and used as given.** `sklearn.base.clone` drops its fit, so wrap it in `sklearn.frozen.FrozenEstimator` when the model is cloned (a tuner clones per trial) | sklearn's own idiom for a prefit estimator inside a meta-estimator; the model has no folds inside |
| **Set before `fit`.** The two models are cloned in `fit`, and `predict` follows the fitted copies and the calibrator given at `fit`. Nested names reach both models, e.g. `set_params(cohort_probability_model__l2_penalty=1e-3)` (unlike Model 1's mapping) | Two named estimators are ordinary parameters; the templates stay unfitted |
| **The composition model's columns must equal `y`'s at fit**, or `predict` raises | Shares under other names or in another order would be multiplied in by position, silently |
| **`nb2` takes `solver="lbfgs"` only** | The floor on $\alpha$ is an L-BFGS-B bound; scipy would only warn and drop it under BFGS |
| **One penalty meaning**: per building, intercepts unpenalized | Comparable across folds of different size; the old `probability_c` was per child ([Feature transformations §8.7 item 7](FEATURE_TRANSFORMATIONS.md#87-still-outstanding)) |
| **$T$ is fitted per building; `COHORT_LOG_LOSS` scores per child** | The calibrator weighs buildings as the Dirichlet fit does; the metric weighs children, so the fitted $T$ is not the metric's optimum |
| **The target columns come from `y` only**; with `remainder="drop"` (the §8 transformers) `X` may keep them | `predict` needs no target column |

### 0.5 Errors

| Raises | When |
|---|---|
| `ValueError` "pass `exposure`" / "inconsistent numbers of samples" / "one-dimensional" | the total model fitted or used with the offset but no exposure, or one of another length or shape |
| `ValueError` "no child is observed in cohorts" | a cohort column of `y` sums to 0 |
| `ValueError` "feature names" | columns reordered, renamed or missing at `predict` (scikit-learn's check) |
| `ValueError` "unknown family" / "takes no bounds" / "l2_penalty must be at least 0" / "unknown solver" | a setting that would otherwise fit something else silently |
| `RuntimeError` "did not converge" / "overflowed" / "non-finite objective" | the optimizer stopped early, or features on too large a scale (standardize them) |
| `RuntimeError` "temperature search failed" | nan logits in the calibrator's `fit`; a building without children too, after numpy's divide warning (an error under `-W error`) |
| `NotFittedError` | `predict` before `fit`; a calibrator that was never fitted or was dropped by `clone` |
| `ValueError` "predicts cohorts … but y's columns at fit were" | the composition model's columns differ from `y`'s |

Left to the libraries or to preprocessing: a negative count or an empty
building (torch's argument validation raises; `ExposureTransformer` and the
data preparation validate values), a Series `y` for the composition model,
NaN in `X` (scikit-learn's input check).

### 0.6 What Changed From §1–§12

- **The composition model is a Dirichlet regression of the shares** (§0.1),
  not a grouped multinomial on the counts: it models the composition and its
  precision, weighs buildings equally, and is identified. Zero shares are
  compressed, not dropped.
- **Calibration is post-hoc** (`TemperatureCalibrator`), fitted by the caller
  on out-of-fold logits, per building, with **no likelihood-ratio gate**: the
  fitted $T$ is always used (§4's gate and `calibration_significance_level`
  are gone).
- **No clipping and no floor** on the mean (§3's $[-30, 30]$ and
  `minimum_mean`); a failed fit raises. NB2's $\alpha$ has a floor at
  $10^{-6}$ and no ceiling (§3's $[10^{-4}, 5]$).
- **The exposure is an argument** (`exposure=`), built once with
  `ExposureTransformer`, not a column the model reads.
- **Hyperparameters are fixed in the constructors** and tuned from outside
  (`set_params`); nothing is searched in `fit` (§5).
- **Dropped** (means only): pointwise and joint log probabilities, predictive
  draws and intervals, bootstrap refits, state bundles, persistence and
  metadata (§6–§10).
- **Evidence**: the plan's B7 smoke run over ten simulated populations,
  untuned: Model 2 equals Model 1 on held-out deviance (within the noise) and
  beats it on composition (cohort log loss, under 1% per child); calibration
  adds a small, inconsistent gain ([plan §9 B7](MULTI_COHORT_MODELS_PLAN.md)).

`IndependentTotalProbabilityModel` splits the prediction into two separately
fitted parts: a count regression for a building's **total** children and a
grouped multinomial model for the **age composition** of those children. Its
cohort means reconcile exactly with the total, and its likelihood is a joint
score directly comparable with the Bayesian model. Everything here is derived
from `src/age_group_prediction/models/independent_total_probability.py` and its
helpers (`count_regression.py`, `grouped_multinomial.py`,
`probability_calibration.py`, `fold_scoring.py`); the design rationale is in
[MODELING_REBUILD_PLAN.md](MODELING_REBUILD_PLAN.md) Section 7.

## 1. Statistical Model

For building $b$ with apartment count $A_b$, total-feature vector $x_b$, and
probability-feature vector $w_b$:

**Total component** (family fixed per model instance):

$$
Y_b \sim \operatorname{NB2}(\mu_b, \alpha)
\;\;\text{or}\;\;
Y_b \sim \operatorname{Poisson}(\mu_b),
\qquad
\log\mu_b = \log A_b + \beta_0 + x_b^\top\beta,
$$

with $\operatorname{Var}(Y_b)=\mu_b+\alpha\mu_b^2$ for NB2 ($\alpha$ is the
project's dispersion; $\phi = 1/\alpha$ in the Bayesian guide's notation).
$\log A_b$ is an **exposure offset** with coefficient fixed at 1, so the model
estimates children per apartment and expected children scale proportionally
with building size.

**Composition component:**

$$
(C_{b,1}, C_{b,2}, C_{b,3}) \mid Y_b \sim \operatorname{Multinomial}(Y_b, p_b),
\qquad
p_b = \operatorname{softmax}\!\left(\frac{z_b}{T}\right),
\qquad
z_{b,k} = \gamma_{0,k} + w_b^\top\gamma_k,
$$

where $T$ is a calibration temperature (Section 4).

**Combined predictions:**

$$
\widehat C_{b,k} = \widehat\mu_b\,\widehat p_{b,k},
\qquad
\sum_k \widehat C_{b,k} = \widehat\mu_b.
$$

The two components are fitted independently: the composition model conditions
on observed totals during training, and the total model never sees the cohort
split.

## 2. Features

- **Total:** the `total_count` spec passed to `fit` (default
  `DEFAULT_TOTAL_FEATURE_SPEC`: scaled numeric features, one-hot
  `school_status`, `log(n_apartments)` offset). `fit` refuses a spec without an
  exposure.
- **Composition:** `probability_feature_spec` passed to the constructor
  (default `DEFAULT_PROBABILITY_FEATURE_SPEC`: the same features, no exposure).

Each component has its own fitted transformer. See
[FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md).

## 3. Total Component: Penalized Count Regression

The design matrix is the transformed features plus an intercept. The objective
minimized with SciPy L-BFGS-B is

$$
-\frac{1}{n}\sum_b \log f(y_b \mid \mu_b) \;+\; \frac{\lambda}{2}\,\lVert\beta\rVert^2,
$$

the mean negative log likelihood plus an L2 penalty on the **non-intercept**
coefficients ($\lambda$ = `total_l2_penalty`).

- **NB2** jointly estimates $\beta_0$, $\beta$, and $\log\alpha$, with
  $\alpha$ bounded by `dispersion_bounds` = $[10^{-4}, 5]$. The intercept starts
  at the pooled rate $\log(\sum_b y_b / \sum_b A_b)$ and $\log\alpha$ at the
  geometric mean of its bounds.
- **Poisson** estimates $\beta_0$ and $\beta$ from the same intercept start.
- Log means are clipped to $[-30, 30]$ during optimization, and predicted means
  are floored at `minimum_mean`.
- A failed or non-finite optimization raises instead of returning a fit.

## 4. Composition Component

### Grouped multinomial fit

Each building becomes at most three weighted rows (features, cohort label,
weight = that cohort's child count); zero-weight rows are dropped. This
optimizes exactly the same multinomial log likelihood as one row per child,
which `test_grouped_and_literal_expansion_are_equivalent` verifies. A scikit-learn
`LogisticRegression` (`lbfgs`, inverse regularization `C` = `probability_c`)
fits the weighted rows. Fitting refuses data with no children or with any age
class unobserved.

Zero-total buildings still contribute to the total model and to preprocessing
state, but carry zero composition weight.

### Temperature calibration

*Rebuilt (2026-10-01, `modeling/calibration.py`, `TemperatureCalibrator`):
the calibrator is post-hoc, fitted by the caller on out-of-fold logits, with no
likelihood-ratio gate and a per-building objective; see §0.1. The paragraph
below describes the old model.*

After `probability_c` is selected, the model fits the composition component on
each tuning fold's fit rows and collects out-of-fold logits for the validation
rows. A single temperature $T \in [0.25, 4]$ minimizes the child-weighted
multinomial NLL of $\operatorname{softmax}(z/T)$ on those out-of-fold logits.
The fitted $T$ is **retained only if** a one-degree-of-freedom likelihood-ratio
test rejects $T = 1$ at `calibration_significance_level` (0.05); otherwise
$T = 1$. The statistic is $2 \cdot n_{\text{children}} \cdot
(\text{NLL}_{T=1} - \text{NLL}_{\hat T})$. Diagnostics record the raw and fitted
temperatures, the decision, the test statistic and p-value, weighted NLL, and
child-weighted share error.

## 5. Fitting Procedure

`fit(train_df, feature_spec=..., rng=...)` on the training partition only:

1. Build internal known-neighborhood tuning folds from `train_df`
   (`tuning_folds`, taken from the `[folds]` TOML section; 5 folds canonically).
2. **Tune the total** with a seeded single-job Optuna TPE study (`n_trials` =
   30): `total_l2_penalty` in $[0, 1]$ (linear scale), objective = mean held-out
   total NLL across folds.
3. **Tune the composition** with its own study: `probability_c` in
   $[0.01, 100]$ (log scale), objective = held-out child-weighted composition
   NLL. Only completed trials can win.
4. **Calibrate** the temperature from cross-fitted logits (Section 4).
5. **Refit both components** on the full training data with the selected
   regularization; the probability transformer is refitted on all training
   buildings.
6. Retain the training frame in memory for bootstrap uncertainty.

Feature forms and the total family are **not** chosen inside a fit. They are
explicit candidate identities compared by the experiment runner on identical
folds.

### Hyperparameter tuning in the wider workflow

| Hyperparameter | Search range | Scale | Objective | Fixed afterwards |
|---|---|---|---|---|
| `total_l2_penalty` ($\lambda$) | $[0, 1]$ | linear | mean held-out total NLL across inner folds | yes |
| `probability_c` (`C`) | $[0.01, 100]$ | log | held-out child-weighted composition NLL | yes |
| NB2 dispersion $\alpha$ | $[10^{-4}, 5]$ | estimated by maximum likelihood in each fit, not by Optuna | — | refitted per fit |
| Temperature $T$ | $[0.25, 4]$ | fitted on out-of-fold logits after `C` is fixed, retained only by the likelihood-ratio test | — | yes |

- **Separate studies.** The two components are tuned in separate seeded TPE
  studies, each with `[tuning] n_trials` (30), `n_jobs=1`, and no timeout.
  Only completed trials can be selected.
- **Nested in cross-validation.** The experiment runner calls `fit` on each
  outer fold's fit rows, so tuning and calibration run inside every fold on
  training-only inner folds (`[folds]`, 5 canonically). The final refit repeats
  them on the full training partition.
- **Evidence.** `metadata["diagnostics"]["selection"]` records the
  `total_tuning` and `probability_tuning` trial histories, the feature specs,
  and the search space; calibration diagnostics are recorded separately. MLflow
  logs `tuning/total/best_value`, `tuning/probability/best_value`, and
  `tuning/fold_{i}/{total,probability}_trials.csv`.
- **Bootstrap reuse.** Bootstrap uncertainty refits reuse the selected
  $\lambda$, `C`, and $T$ rather than re-tuning, so intervals reflect sampling
  variability at fixed hyperparameters.

## 6. Prediction

`predict(eval_df, prediction_config=..., rng=...)`:

- **Means.** $\widehat\mu_b$ from the total regression and offset;
  $\widehat p_b = \operatorname{softmax}(z_b/T)$; cohort means
  $\widehat\mu_b \widehat p_b$.
- **Parametric distributions.** Only `total` is declared (NB2 with $\alpha$, or
  Poisson). Cohorts have no marginal family declaration; their uncertainty comes
  from joint draws.

### Pointwise log probabilities

Scope is **`sequential_joint`** (observed targets required):

- `total`: $\log f(Y_b \mid \widehat\mu_b)$ under NB2 or Poisson.
- each cohort: successive differences of multinomial prefix log masses
  (`multinomial_prefix_log_masses`). The three cohort entries sum to the full
  multinomial log mass, **including** the multinomial coefficient.

So `total` plus the cohort entries equals the joint log mass
$\log f(Y_b) + \log \operatorname{Mult}(C_b \mid Y_b, \widehat p_b)$. These
values are directly comparable with the Bayesian model's `sequential_joint`
scores, and never with Model A's `marginal` scores. The kernel does not clip: a
positive count in a cohort with zero predicted probability has log mass
$-\infty$, which `PredictionResult` refuses as non-finite, so `predict` raises
instead of returning a clipped score.

### Predictive draws and intervals

Neighborhood-cluster bootstrap with
$R=\max(\text{bootstrap\_replicates}, \text{n\_predictive\_draws})$ replicates
(default 200). For each replicate:

1. resample whole neighborhoods from the training frame;
2. refit both transformers, the total regression, and the grouped multinomial
   with the **selected** regularization and the **fitted** temperature (no
   re-tuning or re-calibration);
3. draw an integer total from NB2 or Poisson, then cohort counts from
   $\operatorname{Multinomial}(\text{total}, p_b)$.

Every draw reconciles exactly: its cohort counts sum to its total. A replicate
whose resample contains no child of some cohort cannot fit the composition
model; it is skipped and counted. If more than `bootstrap_max_failed_fraction`
(0.25) of replicates fail, prediction raises. Successful and failed replicate
counts are recorded in metadata. Intervals therefore condition on resamples
that contain every cohort.

## 7. Selection And Comparison

The canonical candidate is `independent-nb2` (`total_family` from
`[independent_total_probability]`, default `nb2`). Poisson and NB2 would be
separate sibling candidates, each tuned independently on the same folds.
`sequential_joint_selection_policy` ranks conditional candidates by:

1. `joint_predictive_nll` (the sequential joint log mass above),
2. composition log loss,
3. total RMSE.

Across families, `select_cross_family_winner` first compares Model B's winner
with the Bayesian winner by `joint_predictive_nll`, then compares the better of
the two with Model A's winner by composition log loss, mean cohort RMSE, and
mean cohort MAE. When both conditional approaches are selected, the Bayesian
candidate must use Model B's frozen total and probability feature specs.
Details: [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md);
the one-time refit and holdout scoring are in
[FINAL_EVALUATION.md](FINAL_EVALUATION.md).

## 8. Persistence And Serving

`to_state_bundle()` stores the configuration, the total fit (family,
coefficients, NB2 dispersion, objective value), the multinomial coefficients,
intercepts, and classes, both transformer states, the selected specs and
regularization, the temperature, tuning evidence, and selection and calibration
diagnostics. Neither optimizer re-runs on load, and no training rows or fold
building IDs are stored. A bundle whose stored total family disagrees with its
configuration is refused.

- Point predictions, pointwise scores, and the total distribution need only the
  bundle.
- Draws and intervals need
  `IndependentTotalProbabilityModel.from_state_bundle(bundle, train_df=train_df)`
  with the hash-matching training frame.

## 9. Configuration

| TOML section / field | Default | Effect |
|---|---|---|
| `[independent_total_probability] total_family` | `nb2` | `nb2` or `poisson` |
| `optimizer_max_iterations`, `optimizer_tolerance` | 500, $10^{-8}$ | Count-regression, logistic, and temperature optimizers |
| `dispersion_bounds` | $[10^{-4}, 5]$ | NB2 $\alpha$ bounds |
| `minimum_mean` | $10^{-8}$ | Predicted total mean floor |
| `calibration_temperature_bounds` | $[0.25, 4]$ | Temperature search range |
| `calibration_significance_level` | 0.05 | Likelihood-ratio retention test level |
| `bootstrap_replicates` | 200 | Minimum bootstrap refits |
| `bootstrap_max_failed_fraction` | 0.25 | Maximum skipped replicates before raising |
| `[independent_total_probability_search_space] total_l2_penalty` | $[0, 1]$ | Total L2 search range (linear) |
| `probability_c` | $[0.01, 100]$ | Multinomial `C` search range (log) |
| `[tuning]` and `[folds]` | 30 trials; 5 folds | Optuna study and internal tuning folds |

## 10. Metadata

`model.metadata` reports the likelihood (`nb2 total with conditional grouped
multinomial composition`), parameterization, hyperparameters
(`total_family`, `total_l2_penalty`, `probability_c`, `dispersion_alpha`),
calibration diagnostics, dependency versions, uncertainty method, and
diagnostics: tuning evidence and search space, selected probability spec,
probability preprocessing, `zero_total_policy`, bootstrap replicate counts, and
the cohort distribution declaration.

## 11. Pitfalls

- **Exposure is mandatory.** A total spec without `exposure_column` is refused;
  do not add `n_apartments` as an ordinary predictor.
- **Every cohort must appear in the training data.** Otherwise the composition
  fit raises; bootstrap replicates without a cohort are skipped and counted.
- **Temperature 1 is common and correct.** A rejected temperature means there
  was no significant evidence of miscalibration, not a failure.
- **Compare likelihoods only with conditional models.** Joint scores are
  comparable with the Bayesian model, not with Model A.
- **Family and feature forms are candidate identities.** To compare Poisson
  with NB2 or other feature forms, register separate candidates; do not change
  them inside a fit.
- **Reloaded models need `train_df` for uncertainty.**

## 12. Tests And Related Documents

- Tests of the rebuilt model (§0): `tests/unit/test_modeling_total_children.py`,
  `test_modeling_cohort_probability.py`, `test_modeling_calibration.py`,
  `test_modeling_independent_total_probability.py`, `test_modeling_feature_transformer.py`,
  `test_modeling_optimization.py`, the contract test `test_modeling_contract.py`;
  statsmodels oracles in `tests/validation/test_total_children.py`.
- Tests: `tests/unit/test_independent_total_probability.py` (including the
  grouped/literal expansion oracle), `tests/unit/test_distributions.py`,
  `tests/unit/test_composition_kernels.py`,
  `tests/unit/test_model_state_bundles.py`, and real-model checks in
  `tests/validation/test_experiment_real_models.py`.
- [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md),
  [MODELING_GUIDE.md](MODELING_GUIDE.md) Sections 6–11,
  [DIRECT_COHORT_MODEL.md](DIRECT_COHORT_MODEL.md),
  [BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md](BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md),
  [EVALUATION_AND_METRICS.md](EVALUATION_AND_METRICS.md),
  [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md).
