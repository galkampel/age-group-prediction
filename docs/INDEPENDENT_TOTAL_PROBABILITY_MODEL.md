# Independent Total And Probability Model (Model B)

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
