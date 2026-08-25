# Model Fitting, Evaluation, And Feature Importance Plan

> **Status: EDA handoff passed; ready for iterative implementation.** This is the follow-on to [EDA_AND_PREDICTIVE_MODELING_PLAN.md](EDA_AND_PREDICTIVE_MODELING_PLAN.md). Implement one agreed modeling slice at a time and evaluate its evidence before expanding scope.

The eventual orchestration target is `notebooks/02_model_fitting.py`. Create it
only after the EDA handoff gate passes, following the package boundaries,
dependency sequencing, and model routing in
[MARIMO_MIGRATION_PLAN.md](MARIMO_MIGRATION_PLAN.md).

## 1. Goal

Fit and compare three building-level approaches for predicting resident child cohorts:

1. Three direct cohort-count models.
2. Model B: two independently fitted frequentist components, one for total
   children and one for child-level age-group probabilities, whose expected
   outputs are multiplied to obtain cohort-count predictions.
3. Model C: a Bayesian conditional two-stage model in NumPyro, with an NB2
   total-count stage followed by a Dirichlet-multinomial age-group count stage
   conditioned on the total.

The work must evaluate point accuracy, discrete predictive distributions, cohort composition, calibration, uncertainty, and feature importance. Track all experiments locally with MLflow after the core models are stable.

## 2. Preconditions From EDA

Complete these decisions in the EDA findings table before modeling:

1. Confirm the final building-level schema, target accounting identity, missingness, and candidate prediction-time features.
2. Choose a non-collinear room representation for linear or probabilistic models:
   - omit one room-count category; or
   - use `n_apartments` plus three room shares, omitting the fourth share.
3. Confirm that targets have overdispersion evidence from variance-to-mean ratios and distributional diagnostics.
4. Document feature-support limits, correlated contextual features, and any nonlinearities or interactions worth modeling.
5. Select the deployment question and matching split before model fitting.

The final DataFrame has these prediction-time features:

| Type | Columns |
|---|---|
| Numeric | `ses`, `avg_household_size`, `median_age`, `n_daycares_500m`, room composition, `n_apartments` |
| Categorical | `school_status` |
| IDs excluded from features | `building_id`, `neighborhood_id` |
| Cohort targets | `n_kindergarten`, `n_elementary`, `n_highschool` |
| Total target | `n_children_total` |

The mandatory identity is:

$$
n_{b,\mathrm{total}} = n_{b,\mathrm{kindergarten}} + n_{b,\mathrm{elementary}} + n_{b,\mathrm{highschool}}.
$$

Do not use `n_children_total` to directly predict an individual cohort. It is a post-outcome aggregate, not a prediction-time feature. Do not use IDs or simulator latent effects as model features.

### 2.1 Base Features And Deferred Candidate Terms

The observed base feature set is complete for the current simulator output. Do
not add another raw predictor before the first baselines: every exported
prediction-time variable is already represented, and no latent effect, ID, or
target-derived value is eligible to fill a perceived gap.

Keep model-specific functional forms out of the shared modeling table. Create
them inside each training fold or fitted preprocessing pipeline, and compare
them with the corresponding simpler form on held-out validation data.

Use this priority order:

1. **Baseline forms:** linear terms for the continuous context features,
   `school_status` indicators, `n_apartments` as size or exposure where the
   model supports it, and the agreed three room shares.
2. **EDA-priority candidates:** compare a linear `ses` term with a quadratic or
   low-degree spline; test `ses` by `avg_household_size` and
   `n_daycares_500m` by `median_age` interactions. Retain a term only when its
   improvement is stable across validation resamples.
3. **Secondary generator-informed sensitivity candidates:** room composition by
   `avg_household_size` for total counts and room composition by `median_age`
   for age-group probabilities. The simplified simulator defines these
   mechanisms using each apartment's discrete room count, whereas prediction
   occurs on aggregated building shares. Test parsimonious building-level
   translations such as mean rooms or a large-room excess share as alternatives
   to interactions with all three shares; do not include redundant room
   summaries together. These are hypotheses to validate, not oracle features.
4. **Low-priority daycare shape:** the EDA did not visually support the
   generator's fixed saturation form. Revisit daycare with a monotone transform
   or low-degree spline only if residual diagnostics and held-out validation
   justify it; do not copy the simulator's saturation scale into a predictive
   model by default.

Standardization, centering, one-hot encoding, and `log(n_apartments)` when used
as an exposure offset are preprocessing choices rather than new predictors.
Fit or define them from the training partition only. The direct tree baseline
can learn nonlinearities without explicit polynomial columns, while the
frequentist and Bayesian statistical models require candidate terms to be
declared explicitly.

The simplified simulator also imposes structural dependencies that affect
interpretation and validation:

- Its room mix is generated partly from neighborhood `ses` and
   `avg_household_size`. Room shares remain valid observed predictors, but their
   coefficients and individual importance values are not effects independent of
   those contextual variables. Inspect conditioning and collinearity, and use a
   joint room/context importance sensitivity analysis where appropriate.
- Apartment-level room effects are nonlinear and flatten across room categories.
   Three shares preserve that composition without rank deficiency. A scalar
   `mean_rooms` or `large_room_excess_share` may be tested later as a more
   parsimonious alternative, especially for interactions, but not as an
   additional simultaneous representation.
- Shared building and neighborhood random effects create residual clustering
   and overdispersion. They are private simulator state and must not become
   predictors. For Models A and B, assess residual clustering and use
   neighborhood-aware resampling or uncertainty estimates. Model C may represent
   supported neighborhood heterogeneity through partial pooling; unexplained
   new-building variation remains predictive uncertainty rather than a feature.

## 3. Data Preparation And Split Strategy

Perform feature transformations inside the training partition or cross-validation fold only. One-hot encode `school_status` with `none` as the reference for statistical models. Tree models can use a suitable categorical encoder or one-hot representation within their preprocessing pipeline.

Always split at the building level before constructing Model B's child-level
categorical training view. Expand cohort counts only within the training
partition so children from one building cannot cross partitions and test target
counts cannot influence model fitting.

Use a split that matches deployment:

| Deployment question | Recommended split | Claim supported |
|---|---|---|
| New building in a known neighborhood | Random building-level holdout, preferably randomized within each neighborhood | Conditional within-neighborhood accuracy |
| New building in an unseen neighborhood | Leave-one-neighborhood-out or repeated grouped folds | New-neighborhood generalization |
| Future building after a time boundary | Temporal holdout | Future-time generalization |
| New building in a new geographic area | Spatial block holdout | New-area generalization |

For a known-neighborhood split, stratify within a neighborhood by coarse total-count bins only when the group has enough buildings. Otherwise use a reproducible random split that preserves every neighborhood's representation where possible. Do not label this as unseen-neighborhood performance: the shared contextual features occur in both partitions.

For the planned later evolution to 9-15 neighborhoods, grouped estimates will be noisy. Use leave-one-neighborhood-out or repeated grouped splits and report the full distribution of scores rather than one aggregate point estimate.

Every experiment must log the split strategy, random seed, building counts, neighborhood counts, target distributions, and feature schema for each partition.

### 3.1 Incremental Model Improvement Protocol

Freeze the final test partition before model development. Use only resamples or
a validation subset drawn from `train_df` to select preprocessing, functional
forms, interactions, regularization, dispersion, calibration, and
hyperparameters. Evaluate the untouched test partition once after those choices
are fixed.

Develop each model in explicit stages:

1. **Fit the base specification.** Use only the agreed observed predictors and
   the simplest model-appropriate preprocessing. Record fold-level point and
   distributional metrics, calibration, residual diagnostics, and runtime.
2. **Diagnose a limitation.** Propose another feature form only when EDA,
   validation residuals, or the documented candidate list identifies a concrete
   deficiency. Do not search the final test partition for improvements.
3. **Add one candidate block at a time.** In order, compare the SES shape, the
   EDA-priority interactions, the generator-informed room interactions, and
   finally a daycare shape if diagnostics support it. A block may contain the
   terms required to preserve hierarchy, such as both main effects plus their
   interaction.
4. **Compare on identical training resamples.** Retain a block only when its
   improvement is stable across resamples and meaningful for the model's
   primary metrics, without unacceptable calibration, conditioning, or
   complexity costs. Record rejected blocks as well as retained ones.
5. **Refit the selected specification.** After model and calibration choices
   are frozen, refit on the full training partition and produce the single final
   test report. Any change prompted by that report starts a new experiment with
   a new untouched test partition; it is not part of the original confirmatory
   result.

Feature-engineering priority differs by model:

| Model component | Initial feature engineering | Incremental emphasis |
|---|---|---|
| Model A direct tree regressors | Low. Use the base columns; trees can discover thresholds and interactions without explicit polynomial or product columns. | Tune capacity and regularization first. Add an engineered term only when it encodes useful structure the trees cannot learn reliably at the available sample size, then require a resampled validation gain. |
| Model B NB2 total-count component | High enough to test explicitly. Its linear predictor cannot discover curvature or interactions by itself. | Test SES shape first, then total-specific EDA interactions, then a parsimonious room-composition by household-size block. Test daycare shape last. |
| Model B categorical probability component | High enough to test explicitly, independently of the total component. | Test composition-specific interactions, especially a parsimonious large-room by older-neighborhood term. Do not assume a term selected for the total component belongs here. |
| Model C total and conditional-composition components | Start from the corresponding selected Model B forms rather than reopening feature search during Bayesian fitting. | Use priors and partial pooling to regularize supported terms. Add Bayesian-only complexity only when posterior predictive diagnostics identify a remaining deficiency. |

The components may therefore finish with different feature specifications.
Model A is primarily a flexible predictive benchmark; Models B and C carry the
greater burden of explicit functional-form engineering and interpretable term
selection.

## 4. Model A: Direct Cohort Baseline

Fit three independent non-negative count regressors:

$$
\widehat C_{b,k} = f_k(x_b), \qquad k \in \{\mathrm{kindergarten}, \mathrm{elementary}, \mathrm{highschool}\}.
$$

Use regularized LightGBM with `objective="poisson"` when LightGBM is introduced. Until then, use `HistGradientBoostingRegressor(loss="poisson")`. Keep tuning conservative: small leaf counts or depths, nonzero regularization, minimum leaf sizes that respect the effective sample size, and early stopping only with a validation partition that matches the selected split strategy.

This baseline optimizes each cohort independently. Its summed cohort prediction need not equal a separately predicted total. Report the discrepancy:

$$
d_b = \sum_k \widehat C_{b,k} - \widehat C_{b,\mathrm{total}},
$$

when a direct total model is included as a diagnostic. Do not force reconciliation in the first direct baseline; use it to compare direct predictions with Models B and C.

**Gate:** Predictions are finite and non-negative; no target or target-derived feature enters any cohort model; all preprocessing is fit only on training data.

## 5. Model B: Independent Total And Age-Group Probability Models

Model B consists of two separately fitted components. The total-count model and
the age-group probability model use the same prediction-time environment and
building features, but neither fitted output is an input to the other model.
Combine their predictions only after both components have been fit.

### 5.1 Step 1: Total Child Count

Model total children with an exposure offset for apartment count:

$$
Y_b \sim \operatorname{NB2}(\mu_b, \phi),
$$

$$
\log \mu_b = \log A_b + \beta_0 + x_b^\top\beta,
$$

where $Y_b=n_{b,\mathrm{total}}$ and $A_b=n_{b,\mathrm{apartments}}$. The NB2 variance is:

$$
\operatorname{Var}(Y_b \mid x_b) = \mu_b + \frac{\mu_b^2}{\phi}.
$$

Fit a Poisson model only as a benchmark. The first NB2 model may use one global dispersion parameter. Do not treat the existing coefficient-recovery helper as a production estimator: it supplies a known fixed dispersion for test recovery. The modeling implementation must estimate, profile, or otherwise select the dispersion using training data only.

Use `statsmodels` for the initial interpretable frequentist implementation. Compare held-out Poisson and NB2 likelihood, deviance, residual dispersion, and calibration. Retain NB2 when it materially improves held-out distributional fit; the generator's apartment-level NB2 process and shared random effects make this likely but it must be checked.

### 5.2 Independent Child-Level Age-Group Probability Model

Construct a modeling view from the training partition by expanding each
building's observed cohort counts into one categorical age-group label per
observed child. Repeat only the building's prediction-time environment and
building features on those rows. This is a modeling transformation of the
audited building-level targets; it does not require changes to the simulator or
exposure of apartment-level outcomes.

Zero-child buildings contribute no rows when fitting this categorical model,
but they remain valid inputs at prediction time. The categorical predictors
must exclude `n_children_total`, every cohort target, IDs, and simulator latent
effects. In particular, neither the observed total nor the total model's
prediction is an input to the probability model.

Use a single lifecycle index initially, preserving the intended age progression:

$$
\theta_b = \alpha + x_b^\top\gamma,
$$

$$
\mathbf p_b = \operatorname{softmax}
\begin{pmatrix}
\log(0.40) - 1.0\theta_b \\
\log(0.35) \\
\log(0.25) + 1.1\theta_b
\end{pmatrix}.
$$

The first row is kindergarten, the second elementary, and the third high school. This is a categorical probability model driven by an ordered lifecycle index; it is not a generic ordered-logit regression.

The normalized output is the independently estimated categorical probability
vector $\widehat{\mathbf p}_b$. At prediction time, combine it with the
independently estimated total-count mean:

$$
\widehat C_{b,k} = \widehat\mu_b \widehat p_{b,k}.
$$

These are generally fractional expected cohort counts. Because the probabilities
sum to one, they reconcile with Model B's own expected total:

$$
\sum_k \widehat C_{b,k} = \widehat\mu_b.
$$

Model B does not fit a conditional cohort-count likelihood and does not use a
Dirichlet-multinomial distribution. A later simulation that draws an NB2 total
and allocates it categorically may be reported as a derived approximation, but
it is not part of Model B's defining fit and must not be presented as its joint
predictive distribution.

### 5.3 Probability Calibration And Combined Predictions

Evaluate the categorical component independently with child-weighted multiclass
log loss, count-weighted reliability curves, and a count-weighted Brier score.
Restrict observed shares to buildings with positive observed total. Report that
the child-level fit weights buildings in proportion to their observed child
count. A building-balanced sensitivity analysis may be added later, but it does
not replace the primary child-level estimand.

Evaluate the combined expected cohort counts with per-cohort point metrics and
verify that they sum to the predicted total mean. If calibration is needed,
hold out a calibration subset from training neighborhoods, fit multiclass
temperature scaling to logits, then evaluate once on the untouched test data.
Do not tune calibration on the final test partition.

**Gate:** Both components are fit using training data only; the categorical
model consumes no total or target-derived predictor; probabilities are finite,
non-negative, and sum to one; and multiplied cohort means sum to Model B's
predicted total mean.

## 6. Model C: Bayesian Conditional NB2 And Dirichlet-Multinomial Model

Use NumPyro as the planned Bayesian framework. Keep it in an optional dependency group so ordinary EDA and frequentist work remain lightweight.

### 6.1 Total-Count Component

Start with a building-level hierarchical NB2 model:

$$
Y_b \mid \mu_b, \phi \sim \operatorname{NB2}(\mu_b, \phi),
$$

$$
\log\mu_b = \log A_b + \beta_0 + x_b^\top\beta + u_{g[b]},
$$

$$
u_g \sim \mathcal N(0, \sigma_u^2), \qquad \beta_j \sim \mathcal N(0,1), \qquad \sigma_u \sim \operatorname{HalfNormal}(0.5), \qquad \phi \sim \operatorname{Exponential}(1).
$$

Standardize continuous features with training statistics and run prior-predictive simulation before posterior inference. A direct building-level NB2 is a predictive approximation: the simulator itself generates apartment-level NB2 counts and aggregates them. Document that distinction.

### 6.2 Conditional Age-Group Count Stage

Use the same lifecycle-index probability construction as Model B, with partial pooling when supported by the selected split and sample size:

$$
\theta_b = \alpha + x_b^\top\gamma + v_{g[b]}, \qquad v_g \sim \mathcal N(0, \sigma_v^2).
$$

$$
\mathbf C_b \mid Y_b, \mathbf p_b, \kappa \sim \operatorname{DirichletMultinomial}(Y_b, \kappa\mathbf p_b),
$$

with a prior on concentration $\kappa$. This is Model C's defining second-stage
likelihood: it predicts an age-group count vector conditional on total children
and allows extra-multinomial variation. A plain multinomial is a nested
diagnostic or ablation for whether the fitted concentration is effectively
large; it is not the default Model C definition. Use a centered or sum-to-zero
identification convention appropriate to the implementation.

During training, condition this stage on each building's observed total. At
prediction time, draw or otherwise represent total children from the first-stage
posterior predictive distribution, then draw the age-group count vector from
the Dirichlet-multinomial conditional on that total. Every integer cohort draw
must sum exactly to its corresponding total draw.

Run NUTS with at least four chains for the first reference model. Accept a posterior only after checking divergent transitions, $\hat R < 1.01$, effective sample sizes, energy diagnostics, and posterior-predictive fit. Use SVI only after the NUTS reference establishes a well-behaved model; compare SVI predictive results against the NUTS benchmark.

**Gate:** Prior-predictive samples are plausible, MCMC diagnostics meet
thresholds, posterior-predictive summaries reproduce held-out count and cohort
properties, and every conditional Dirichlet-multinomial draw reconciles with its
total draw.

## 7. Evaluation

Evaluate all candidate models on the same held-out partitions. Report fold-level values and their mean and spread. Reserve a final untouched test set for a single final report after model and calibration choices are fixed.

### 7.1 Point Forecast Metrics

- MAE for average child-count error.
- RMSE for larger-building error sensitivity.
- Mean bias: $\operatorname{mean}(\widehat y-y)$.
- $R^2$ only as secondary context; never as the sole count-model metric.

Report total and every cohort separately.

### 7.2 Discrete Count Metrics

- Negative log likelihood under each model's stated predictive distribution.
- Poisson or NB2 deviance, consistent with the fitted likelihood.
- Predictive interval coverage and average interval width when a model produces intervals.
- PIT histograms or randomized quantile residuals for count-distribution calibration.
- Observed-versus-predicted mean and variance by predicted-count bin.

MAE and RMSE assess point estimates. Likelihood and deviance assess whether a model assigns realistic probability to discrete outcomes. Report both.

### 7.3 Cohort And Composition Metrics

- Report per-cohort MAE, RMSE, and bias for all models on the same held-out
   buildings. These compare Model A's direct counts, Model B's multiplied means,
   and Model C's posterior predictive means.
- For Model B, evaluate the independently fitted categorical probabilities with
   child-weighted multiclass log loss, count-weighted Brier score, and reliability
   plots among buildings with positive totals. Do not report these as count
   likelihood metrics.
- For Model C, report the Dirichlet-multinomial joint log score, posterior
   predictive intervals, coverage, and conditional count calibration.
- Report model-specific reconciliation: Model B's expected cohort predictions
   must sum to its expected total, while every Model C integer cohort draw must
   sum to its conditioning total draw.

## 8. Feature Importance And Interpretation

Run importance analysis only after selecting a validation protocol and a candidate model.

1. Compute permutation importance on held-out validation data using negative log likelihood or deviance, not training error.
2. Repeat permutation runs and report uncertainty in rankings.
3. Permute correlated feature blocks together where appropriate, especially SES, household size, median age, and daycare availability.
4. Treat room counts and apartment count as a constrained block due to their deterministic relation.
5. Use SHAP only as an optional local-explanation supplement; do not rely on split gain or SHAP alone for a global causal story.
6. Reuse PD, ICE, and ALE visualizations from EDA to explain conditional mean patterns and interactions.
7. For Model B, report importance separately for the total-count and categorical
   probability components. For NB2 models, report coefficient intervals and
   incidence-rate ratios. For Model C, report posterior intervals and
   posterior-predictive scenario curves separately for total-count and
   conditional age-group allocation effects.

All results describe predictive association. They do not identify causal effects.

## 9. MLflow Experiment Tracking

Add MLflow only after the first core model executes reproducibly. Start with a local file store:

```python
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("building-age-cohort-prediction")
```

Make the tracking URI configurable through an environment variable so a later server migration does not require model-code changes.

Use one parent run for an experiment and nested runs for each model, hyperparameter candidate, or cross-validation fold. Log:

- simulator configuration contents and hash, simulation seed, package versions, and final-table hash;
- split type, split seed, partition counts, feature schema, and preprocessing choices;
- model name, likelihood, hyperparameters, learned dispersion, and calibration settings;
- all point, distributional, calibration, and per-cohort metrics;
- fitted artifacts, predictions, audit tables, diagnostics, importance outputs, and plots;
- Bayesian priors, sampler settings, diagnostics, posterior summary, and posterior-predictive artifacts.

Never log private simulator latent effects as model inputs or target artifacts.

## 10. Dependency Plan

Do not add modeling dependencies during EDA. The separate marimo migration may
add only its `notebook` tooling group and package build metadata. For this
modeling plan, add optional groups rather than inflating core simulator
dependencies:

| Dependency group | Packages | Purpose |
|---|---|---|
| `modeling` | `lightgbm` and the existing `statsmodels` validation dependency | Boosted count baselines and frequentist NB2 work |
| `bayesian` | `numpyro`, `jax`, and compatible JAX runtime packages | Hierarchical NB2 and conditional Dirichlet-multinomial inference |
| `tracking` | `mlflow` | Local experiment tracking and future remote tracking URI |
| `explainability` | optional `shap` | Supplemental local tree explanations |

Keep exact versions compatible with Python 3.13 and verify each optional group independently before expanding its use. The present `statsmodels` group is sufficient for the first frequentist prototype; decide whether to move or duplicate it only when the modeling dependency policy is implemented.

## 11. Implementation Gates

### Gate 1: EDA Handoff

- All EDA completion checks pass.
- The feature representation and deployment-matched split are documented.
- Overdispersion and zero-count handling are documented.

### Gate 2: Data And Split Tests

- Tests prove no target, target-derived column, ID, or latent effect enters features.
- Split generation is deterministic with a fixed seed.
- Tests prove the selected split matches its stated no-overlap guarantees.

### Gate 3: Baseline Models

- Fit a constant mean or constant cohort-share benchmark.
- Fit direct cohort count models and a Poisson total-count benchmark.
- Verify finite non-negative outputs and reproducible metrics.

### Gate 4: Model B Independent Components

- Fit NB2 and compare to Poisson on held-out distributional metrics.
- Fit the child-level age-group probability model without total or target-derived predictors.
- Verify probability normalization and that multiplied cohort means reconcile with the predicted total mean.

### Gate 5: Model C Conditional Stages

- Fit the Bayesian hierarchical NB2 total stage and conditional Dirichlet-multinomial age-group count stage.
- Verify prior and posterior diagnostics and exact reconciliation of every conditional integer draw.

### Gate 6: Evaluation And Importance

- Calculate the agreed metric suite on held-out data only.
- Produce calibration and residual diagnostics.
- Reproduce permutation importance and preserve correlated-feature caveats.

### Gate 7: Tracking

- MLflow records a clean reproducible run with all necessary metadata and artifacts.

## 12. Agent Model Guide

Use these assignments after the marimo plan's EDA handoff gate. The migration
plan controls model assignments for packaging, conversion, reactive refactoring,
parity, and Jupyter removal.

| Work item | Recommended agent model | Why |
|---|---|---|
| Implement data splits, preprocessing, and baseline HGB models | GPT-5.6 Sol or Sonnet | Bounded implementation with moderate ML pipeline complexity |
| Implement Model B NB2 total and child-level categorical probability fits | GPT-5.6 Sol or Terra | Requires careful separation of estimands, weighting, and leakage controls |
| Review split validity, calibration, metric comparability, and leakage | GPT-5.6 Terra or Opus | Strong independent statistical review |
| Implement Model C NumPyro NB2 and conditional Dirichlet-multinomial stages | GPT-5.6 Terra or Opus | Highest Bayesian inference and numerical-risk work |
| Add MLflow, artifacts, and dependency groups | GPT-5.5 or Sonnet | Conventional integration work once contracts are fixed |
| Broad code review before merge | Codex plus GPT-5.6 Opus | Codex for repository implementation review; Opus for statistical/design review |