# Direct Cohort Model (Model A)

> **Two implementations.** §0 describes the rebuilt
> `age_group_prediction.modeling.DirectCohortModel`. It is built and tested,
> but nothing calls it yet. §1–§10 describe the original
> `models/direct_cohort.py`, which `experiment/` and `tracking/` still run.
> That code is deleted once all three models are rebuilt
> ([plan](MODEL_REIMPLEMENTATION_PLAN.md)).

## 0. The Rebuilt Model (`modeling/direct_cohort.py`)

One LightGBM regressor for **one** cohort's child count. Cohorts are
independent, so each gets its own instance, features and hyperparameters.
- **Hyperparameters are fixed** in the constructor. They are tuned from
  outside through `set_params`; nothing is searched in `fit`.
- **`X` is the finished design matrix.** Features are transformed before they
  reach the model, e.g. with a `FeatureTransformer` fitted per fold
  ([Feature transformations §8.1](FEATURE_TRANSFORMATIONS.md#81-model-a--directcohortmodel-lightgbm)).
- **The objective** is `"poisson"` (default) or `"regression"` (squared
  error).
- **The exposure is optional** (`use_exposure=True`, Poisson only). The raw
  apartment count $n$ is passed to `fit` and `predict`, and enters as the
  offset $\log n$. The trees then learn the cohort's rate **per apartment**
  rather than per building.

### 0.1 The model

**Notation.** For building $i$ and one cohort:

- $\mathbf{x}_i \in \mathbb{R}^p$ is the transformed feature row (a row of `X`).
- $y_i \in \{0, 1, \dots\}$ is the cohort count (`y`).
- $n_i > 0$ is the number of apartments (`exposure`).

**Poisson with `use_exposure=True`:**

$$
y_i \sim \operatorname{Poisson}(\mu_i), \qquad
\log \mu_i = \underbrace{\log n_i}_{\text{offset, coefficient } 1}
+ \underbrace{b + F(\mathbf{x}_i)}_{\text{log rate per apartment}}
$$

- $F(\mathbf{x}) = \sum_{m=1}^{M} \eta\, h_m(\mathbf{x})$ is the sum of trees,
  with learning rate $\eta$.
- $b = \log\big(\sum_i y_i / \sum_i n_i\big)$ is `base_log_rate_`.
- LightGBM receives `init_score` $s_i = \log n_i + b$. It fits $F$ starting
  from $F \equiv 0$, by minimizing the Poisson loss
  $\sum_i \big(e^{s_i + F(\mathbf{x}_i)} - y_i\,(s_i + F(\mathbf{x}_i))\big)$.

**Why this $b$.** It is the maximum-likelihood intercept when $F \equiv 0$:

$$
\frac{\partial}{\partial b} \sum_i \big(y_i(\log n_i + b) - n_i e^{b}\big)
= \sum_i y_i - e^{b} \sum_i n_i = 0
\;\Rightarrow\; b = \log \frac{\sum_i y_i}{\sum_i n_i}
$$

**Prediction.**

$$
\hat\mu_i = \exp\big(\log n_i + b + \hat F(\mathbf{x}_i)\big) = n_i\, e^{\,b + \hat F(\mathbf{x}_i)}
$$

LightGBM's `raw_score` returns only $\hat F$, so the model adds $\log n_i + b$
itself.

- **If $n$ is not a column of `X`,** then
  $\hat\mu(\mathbf{x}, 2n) = 2\,\hat\mu(\mathbf{x}, n)$.
- **If `X` includes `n_apartments`,** $F$ can learn departures from
  proportionality.

**Poisson without exposure.** $\log \mu_i = F(\mathbf{x}_i)$, and $F$ starts at
$\log \bar y$ (LightGBM's `boost_from_average`).

**Regression.** $\mu_i = F(\mathbf{x}_i)$, and $F$ starts at $\bar y$. The
loss is $\sum_i (y_i - \mu_i)^2$.

**Correct inputs.**

| Argument | Pass | Not |
|---|---|---|
| `X` | Transformed features, one row per building (may include `n_apartments`) | The raw table with targets or IDs |
| `y` | The raw cohort count $y_i$ | The rate $y_i / n_i$, or $\log y_i$ |
| `exposure` | The raw $n_i$, in the same row order as `X` | $\log n_i$: the model takes the log itself, so it would be applied twice |

The output $\hat\mu_i$ is an expected **count**, not a rate.

### 0.2 API

| Member | What it takes or gives |
|---|---|
| `DirectCohortModel(*, objective="poisson", use_exposure=False, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_alpha, reg_lambda, min_split_gain, subsample, colsample_bytree, random_state=42, n_jobs=1)` | Settings only, stored verbatim. The 10 hyperparameters default to LightGBM's own. `n_jobs=1` avoids an OpenMP crash alongside torch on macOS |
| `fit(X, y, exposure=None)` | `y` is the raw cohort count; `exposure` is the raw $n$ |
| `predict(X, exposure=None)` | Expected counts, a 1-D array. It follows how the model was fitted: pass `exposure` exactly when it was fitted with one |
| `evaluate(y_true, y_pred, metric)` | One `float`. `metric` is a `Metric`: `POISSON_DEVIANCE`, `RMSE`, `MAE`, or a custom `Metric(name, function, greater_is_better=False)` |
| `regressor_`, `base_log_rate_` | Fitted state: the LightGBM model, and the intercept $b$ (`None` without the exposure) |

```python
from sklearn.base import clone
from age_group_prediction.modeling import POISSON_DEVIANCE, DirectCohortModel

features = clone(tree).fit(train_df)  # tree: a FeatureTransformer (§8.1)
X_train, X_test = features.transform(train_df), features.transform(test_df)
model = DirectCohortModel(use_exposure=True).fit(
    X_train, train_df["n_kindergarten"], exposure=train_df["n_apartments"]
)
predictions = model.predict(X_test, exposure=test_df["n_apartments"])
score = model.evaluate(test_df["n_kindergarten"], predictions, POISSON_DEVIANCE)
```

### 0.3 Errors

The model checks only what would otherwise pass silently. Each of these raises
`ValueError`:
- `use_exposure=True` with `"regression"`, which has no log link;
- `exposure` not passed exactly when the model uses one, whether it's missing
  or unexpected;
- an exposure that isn't strictly positive and finite. LightGBM would accept
  the resulting `-inf` or `nan` offset.

LightGBM raises its own error for an unknown objective, a wrong-length exposure
at fit, and an all-zero `y`. The unit tests pin these.

### 0.4 What Changed From §1–§10

| Original | Rebuilt |
|---|---|
| One instance, three cohorts, and Optuna tuning inside `fit` | One instance per cohort; hyperparameters fixed and tuned from outside |
| Families `poisson`, `nb2` (custom gradient), `normal` | Objectives `poisson` and `regression`; NB2 removed |
| A `FeatureSpec` and fitted preprocessing owned by the model | `X` is transformed before `fit` |
| No exposure offset | Optional `use_exposure`, with starting rate $b$ |
| Bootstrap draws, intervals, pointwise log probabilities, `PredictionResult` | Means only |
| State bundles, `get_metadata`, `configuration_record` | `get_params()` and the public fitted attributes |

### 0.5 Evidence For The Exposure

We ran 10 simulated populations, split by neighborhood, with untuned defaults
and `n_apartments` kept as a feature. The exposure lowered held-out Poisson
deviance by about 4% for kindergarten and high school, and not at all for
elementary. That is weak evidence
([plan, Step 2.4](MODEL_REIMPLEMENTATION_PLAN.md)). Re-check it once the models
are tuned.

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
