# Bayesian Conditional Model (Model C): Overview

`BayesianConditionalModel` predicts each building's total number of children
with a hierarchical negative-binomial model, then divides that total among the
age cohorts with a Dirichlet-multinomial composition model. Both stages are
fitted with Pyro NUTS. This page covers the model, fitting, prediction,
selection, persistence, and configuration in the same shape as the Model A and
Model B descriptions. The full technical guide, covering Pyro syntax, named
sites, tensor shapes, the PyTorch parameter mapping, numerical workarounds, and
implementation cautions, is
[BAYESIAN_CONDITIONAL_MODEL.md](BAYESIAN_CONDITIONAL_MODEL.md).

Source: `src/age_group_prediction/models/bayesian_conditional.py`
(lifecycle and prediction), `bayesian_components.py` (Pyro models and prior
predictive checks), and `bayesian_inference.py` (NUTS and diagnostics).

## 1. Statistical Model

For building $b$ in neighborhood $j[b]$, with apartment count $A_b$,
transformed total features $x_b$, and probability features $w_b$:

**Total stage** (hierarchical NB2 with an exposure offset):

$$
\log\mu_b = \log A_b + \beta_0 + x_b^\top\beta + \sigma_u\,r_{j[b]},
\qquad
Y_b \mid \mu_b, \phi \sim \operatorname{NB2}(\mu_b, \phi),
\qquad
\operatorname{Var}(Y_b) = \mu_b + \frac{\mu_b^2}{\phi}.
$$

Neighborhood effects use a non-centered parameterization,
$u_j = \sigma_u r_j$ with $r_j \sim \mathcal N(0, 1)$.

**Composition stage** (conditional on the observed total during fitting):

$$
\eta_{b,k} = a_k + w_b^\top\gamma_k \;\;(k = 1, \dots, K-1),
\qquad
\eta_{b,K} = 0,
\qquad
p_b = \operatorname{softmax}(\eta_b),
$$

$$
C_b \mid Y_b, p_b, \kappa \sim \operatorname{DirichletMultinomial}(Y_b,\; \kappa\, p_b).
$$

The last cohort (`n_highschool`) is the softmax reference. The concentration
$\kappa$ controls extra-multinomial variation between buildings with the same
expected composition.

**Priors** (`[bayesian_priors]` defaults):

| Parameter | Prior |
|---|---|
| $\beta_0$ | $\mathcal N(-2, 1)$ |
| $\beta$ | $\mathcal N(0, 0.5^2)$ per coefficient |
| $\log\phi$ | $\mathcal N(0, 1)$ (keys `dispersion_log_loc`, `dispersion_log_scale`) |
| $\sigma_u$ | $\operatorname{HalfNormal}(0.5)$ |
| $a_k$ | $\mathcal N(0, 1)$ |
| $\gamma_k$ | $\mathcal N(0, 0.5^2)$ per coefficient |
| $\log\kappa$ | $\mathcal N(2, 1)$ |

The two stages are fitted as **separate posteriors**. Total and composition
parameters are independent a posteriori because the composition likelihood
conditions on observed totals.

## 2. Features

- **Total:** the `total_count` spec passed to `fit`, which must define the
  `n_apartments` exposure; the fit refuses it otherwise.
- **Composition:** the `probability_feature_spec` constructor argument, with
  its own fitted transformer.

In cross-validation both must equal Model B's selected specs, or the selection
freeze is refused. See [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md).

## 3. Fitting

`fit(train_df, feature_spec=..., rng=...)` on the training partition:

1. Fit both feature transformers and compute the log-exposure offset; map
   neighborhood IDs to integer indices. Rows whose cohorts do not sum to the
   total are refused.
2. **Prior-predictive checks** (`[bayesian_prior_predictive]`, 200 draws):
   children per apartment, expected cohort shares, dominant share, and
   concentration are compared with plausibility bounds. Violations warn by
   default (`action = "warn"`) or raise under `"error"`.
3. **NUTS for the total stage**, then **NUTS for the composition stage**,
   using the active profile. Chains run one at a time, with seeds derived from
   the model's generator.
4. **Diagnostics** are computed per stage and recorded **before** the policy is
   applied, so a failing fit still carries its evidence:
   - worst split R-hat;
   - minimum effective sample size;
   - divergences;
   - mean acceptance probability range;
   - post-warmup tree-depth saturation.
5. **Policy:** under `warn`, a failure emits `RuntimeWarning`; under `error`, it
   raises `RuntimeError` with `stage`, `diagnostics`, and `failures` attached,
   and fitted state is cleared.

### Hyperparameters and tuning

There is **no Optuna search**. Priors, NUTS profiles, diagnostic thresholds,
prior-predictive bounds, and stabilization are fixed configuration. NUTS
adapts its step size and mass matrix during warmup; that is sampler
adaptation, not model selection. Changing priors or profiles defines a new
candidate. See
[Hyperparameter Tuning](MODELING_GUIDE.md#hyperparameter-tuning).

## 4. Prediction

`predict(eval_df, prediction_config=..., rng=...)` uses the stored posterior
samples and never re-runs NUTS.

- **Point means.** $\widehat\mu_b$ is the posterior mean of $\mu_b$ and
  $\widehat p_b$ the posterior mean of $p_b$; cohort means are
  $\widehat\mu_b\,\widehat p_b$, which reconcile exactly with the total.
- **Unseen neighborhoods.** A building whose neighborhood was not in training
  gets a fresh effect $u \sim \mathcal N(0, \sigma_u)$ per posterior sample and
  row, so its prediction depends on the random generator. The count is recorded
  as `last_prediction_unseen_neighborhood_count`. The project's claims cover
  known neighborhoods only.
- **Predictive draws.** Each draw picks a total posterior sample and a
  composition posterior sample independently. It then draws
  $Y \sim \operatorname{NB2}(\mu, \phi)$, shares from
  $\operatorname{Dirichlet}(\kappa p)$, and cohorts from
  $\operatorname{Multinomial}(Y, \text{shares})$. Cohort draws sum exactly to
  the total draw; this is checked on every call.
- **Intervals.** Central intervals from at least 500 internal draws when
  `interval_levels` is set; `n_predictive_draws` controls how many draws are
  reported.
- **Stabilization.** Optional clipping of total log means and composition
  logits (`[bayesian_stabilization]`, off by default).

### Pointwise log probabilities

Scope is **`sequential_joint`**:

- `total`: posterior-integrated NB2 log mass, $\log \frac{1}{S}\sum_s
  f_{\text{NB2}}(Y_b \mid \mu_b^{(s)}, \phi^{(s)})$.
- each cohort: successive differences of posterior-integrated
  Dirichlet-multinomial prefix log masses (the multinomial coefficient is
  included). The last cohort is determined by the total and scores about zero.

`total` plus the cohort entries is the joint score used for selection. It is
comparable with Model B's `sequential_joint` scores and never with Model A's
`marginal` scores.

## 5. Selection And Comparison

| Stage | Candidate and profile | Diagnostic action |
|---|---|---|
| Cross-validation | `bayesian-reduced` with `active_profile = "reduced"` (2 chains, 150 warmup, 150 samples) | `warn` |
| Final full-training refit | final-refit factory forced to `full` (4 chains, 1000 warmup, 1000 samples) | `error` |

- **Within the approach,** `sequential_joint_selection_policy` ranks by
  `joint_predictive_nll`, then composition log loss, then total RMSE. By
  default (`require_convergence=True`), candidates with failed diagnostic
  policies are excluded.
- **Across families,** the Bayesian and Model B winners are compared by
  `joint_predictive_nll`; the better one is compared with Model A by
  composition log loss, mean cohort RMSE, and mean cohort MAE.
- **Final refit guard:** the refit refuses a model whose recorded diagnostics
  do not show the full profile with thresholds at least as strict as the full
  defaults.
- In the canonical run this model was the selected cross-family winner.

Details: [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md)
and [FINAL_EVALUATION.md](FINAL_EVALUATION.md#4-refitting-the-frozen-winners).

## 6. Persistence And Serving

`to_state_bundle()` stores the configuration, both transformer states, the
posterior samples for the sites prediction reads, the neighborhood lookup,
per-stage diagnostics, and the prior-predictive summary. It stores no training
rows.

- Reloading never re-runs NUTS, and **draws and intervals do not need the
  training frame** (unlike Models A and B).
- The bundle contains posterior summaries and training neighborhood IDs; store
  it like the data. A full-profile bundle holds 4,000 samples per site.
- MLflow pyfunc serving returns deterministic point predictions only.

## 7. Configuration

| Section | Defaults | Role |
|---|---|---|
| `[bayesian_conditional]` | `active_profile = "reduced"` | Profile and matching diagnostic policy |
| `[bayesian_priors]` | see Section 1 | Priors |
| `[bayesian_reduced_profile]` | 2 chains, 150 warmup, 150 samples, `target_acceptance = 0.9`, `max_tree_depth = 10` | Cross-validation sampling |
| `[bayesian_full_profile]` | 4 chains, 1000 warmup, 1000 samples, same sampler settings | Final refit sampling |
| `[bayesian_reduced_diagnostics]` | `warn`; R-hat ≤ 1.05; ESS ≥ 50; 0 divergences; acceptance 0.6–0.98; tree-depth saturation ≤ 0.05 | Reduced-profile policy |
| `[bayesian_full_diagnostics]` | `error`; ESS ≥ 100; other thresholds as above | Full-profile policy |
| `[bayesian_prior_predictive]` | 200 draws, `warn`, plausibility bounds | Checks before inference |
| `[bayesian_stabilization]` | clipping off; bounds ±20 | Optional numerical clipping |

Feature forms and the random seed are not configured here.

## 8. Metadata

`model.metadata` includes the likelihood and parameterization, the
pointwise-score scope, the full configuration and priors, torch and pyro
versions, per-stage diagnostics with `policy_passed` and `policy_failures`,
the prior-predictive summary, the known-neighborhood count, the last
prediction's unseen-neighborhood count, and probability preprocessing. MLflow
logs `diag/{stage}/max_rhat`, `min_ess_clamped`, `ess_valid`, and
`policy_passed`.

## 9. Pitfalls

- **The full profile is slow.** It runs 8,000 NUTS iterations per stage across
  chains, one chain at a time; cross-validation uses the reduced profile for
  this reason.
- **A reduced-profile warning is evidence, not noise.** Check
  `policy_failures` before trusting a candidate, or rely on the default
  convergence requirement.
- **Unseen neighborhoods are a fallback.** Their predictions include random
  neighborhood effects and are outside the project's validated claim.
- **Keep feature specs tied to Model B.** A different spec is refused in
  selection.
- **Compare likelihoods only with Model B.**

## 10. Tests And Related Documents

- Tests: `tests/unit/test_bayesian_conditional.py` (mocked NUTS contracts),
  `tests/validation/test_bayesian_recovery.py` (real NUTS, marked `slow`),
  `tests/unit/test_distributions.py`, and
  `tests/unit/test_model_state_bundles.py`.
- [BAYESIAN_CONDITIONAL_MODEL.md](BAYESIAN_CONDITIONAL_MODEL.md) (full
  technical guide), [DIRECT_COHORT_MODEL.md](DIRECT_COHORT_MODEL.md) (Model A),
  [INDEPENDENT_TOTAL_PROBABILITY_MODEL.md](INDEPENDENT_TOTAL_PROBABILITY_MODEL.md)
  (Model B), [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md),
  [MODELING_GUIDE.md](MODELING_GUIDE.md),
  [EVALUATION_AND_METRICS.md](EVALUATION_AND_METRICS.md),
  [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md),
  [FINAL_EVALUATION.md](FINAL_EVALUATION.md).
