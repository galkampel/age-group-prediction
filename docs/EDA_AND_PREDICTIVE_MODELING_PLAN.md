# Building-Level EDA And Predictive Modeling Plan

> **Status: EDA Steps 1-6 implemented; modeling handoff design recorded in Step 7.** Formal model fitting and selection, held-out evaluation, feature importance, Bayesian inference, and MLflow are deferred until the modeling notebook begins. This document does not change simulator behavior.

The compact simulator contract remains governed by [SIMPLIFIED_MODEL_PLAN.md](SIMPLIFIED_MODEL_PLAN.md).
The notebook format, conversion sequence, dependency changes, and per-step
agent assignments are governed by
[MARIMO_MIGRATION_PLAN.md](MARIMO_MIGRATION_PLAN.md).

## 1. Goal And Scope

Use the building-level DataFrame returned by `StudentPopulationSimulator.run()` to audit data quality and investigate relationships between prediction-time features, total child counts, and age-cohort counts.

The target EDA deliverable is `notebooks/01_eda.py`, a marimo notebook that can
run from a clean project environment and ends with a short findings table.
The marimo cutover gate has passed and `research.ipynb` has been removed. Do
not add model packages, MLflow, a production data loader, or reusable modeling
modules in this phase.

## 2. EDA Population And Reproducibility

Generate one expanded synthetic population by updating only the existing configuration values in `configs/stage1.toml`:

```toml
[simulation]
n_neighborhoods = 150

[building]
buildings_per_neighborhood_rate = 9.0
```

Keep `buildings_per_neighborhood_min = 1`. The resulting mean is about ten buildings per neighborhood, or roughly 1,500 buildings overall:

$$
E[B] = 150 \times (1 + 9) = 1{,}500.
$$

The Poisson building-count draw retains useful variation between neighborhoods. Increasing the neighborhood count provides 150 independent context values for SES, household size, median age, daycare availability, and school status. Increasing buildings per neighborhood improves precision of within-neighborhood building summaries; it does not create additional distinct context values.

At the start of the notebook, record the seed, configuration path and contents, a configuration hash, the generated row count, unique neighborhood count, and the distribution of buildings per neighborhood. Use the configured seed for the reproducible reference run. Do not generate repeated seeds until the first EDA pass is complete. Keep these canonical values in code rather than interactive controls so clean script execution and model-assisted review reproduce the same reference analysis.

## 3. Data Contract And Leakage Register

The final table has one row per building:

| Role | Columns |
|---|---|
| IDs | `building_id`, `neighborhood_id` |
| Numeric prediction-time features | `ses`, `avg_household_size`, `median_age`, `n_daycares_500m`, `3_rooms`, `4_rooms`, `5_rooms`, `6_rooms`, `n_apartments` |
| Categorical prediction-time feature | `school_status` |
| Cohort targets | `n_kindergarten`, `n_elementary`, `n_highschool` |
| Total target | `n_children_total` |

The following accounting identity must hold exactly:

$$
n_{b,\mathrm{total}} = n_{b,\mathrm{kindergarten}} + n_{b,\mathrm{elementary}} + n_{b,\mathrm{highschool}}.
$$

Create a notebook table that classifies each column as identifier, prediction-time feature, target, target-derived value, or excluded internal value. Apply these rules:

1. Never use `building_id` or `neighborhood_id` as a predictive feature.
2. Never use any cohort target or `n_children_total` to predict an individual cohort directly. The total is a post-outcome aggregate.
3. Keep simulator random effects excluded. They are not exported prediction features and would leak hidden outcome information.
4. Treat only the listed static building and neighborhood values as candidate prediction-time features.

## 4. EDA Implementation Checklist

Perform the following in `notebooks/01_eda.py`, in order. Use the full generated
population; do not create a train/test split during this EDA phase. Follow
marimo's reactive constraints: define each global name once, use distinct names
for successive DataFrame transformations, keep cell-local temporaries private,
and do not mutate a DataFrame owned by another cell.

### Step 1: Structural And Quality Audit

1. Display shape, column names, dtypes, and a sample of final building rows.
2. Verify unique `building_id`, non-null `neighborhood_id`, non-negative integer targets, positive `n_apartments`, and the exact cohort-total identity.
3. Report missing count and missing rate for every column, including an explicit statement when all values are complete.
4. Report buildings per neighborhood and identify the minimum, maximum, quantiles, and any unexpectedly small groups.
5. Report the number of unique values for each feature. Flag that all neighborhood-level features repeat within neighborhoods.

**Completion check:** Every structural assertion passes and the notebook shows one compact quality-summary table.

### Step 2: Target And Feature Distributions

1. For `n_children_total` and each cohort target, report count, mean, standard deviation, quartiles, maximum, zero share, and variance-to-mean ratio.
2. Plot discrete histograms or probability bars and empirical CDFs for total children and each cohort.
3. Plot cohort shares only for buildings where `n_children_total > 0`; report zero-total buildings separately because their cohort proportions are undefined.
4. Plot distributions for numeric features and frequency bars for `school_status`.
5. Produce target summaries by neighborhood, school status, apartment-count quantile, and room-composition bin.

The variance-to-mean ratio is the first overdispersion diagnostic. The simulator uses NB2 apartment counts and shared effects, so a ratio meaningfully above one at building level is expected.

**Completion check:** All four targets and all candidate features have a readable numerical summary and at least one appropriate distribution plot.

### Step 3: Correlation And Collinearity

1. Plot both Spearman and Pearson correlation matrices among numeric features and between numeric features and each target.
2. Treat correlations as global association summaries only. A U-shaped SES relationship can have correlation near zero, and a saturating daycare relationship can have a weak linear correlation despite a visible effect.
3. Display and explain the exact accounting relation:

$$
n_{b,\mathrm{apartments}} = n_{b,3\mathrm{rooms}} + n_{b,4\mathrm{rooms}} + n_{b,5\mathrm{rooms}} + n_{b,6\mathrm{rooms}}.
$$

4. Do not calculate VIF using all room counts and `n_apartments`. For later linear models, choose either one dropped-reference room count or `n_apartments` with three room shares.
5. Discuss correlations among SES, household size, median age, and daycares as generated contextual dependence, not causal evidence.

**Completion check:** The notebook identifies deterministic feature relations and cautions against interpreting a small correlation as proof of no nonlinear relationship.

### Step 4: Visual Marginal Nonlinearities

For SES and `n_daycares_500m`, and then for household size, median age, and apartment count:

1. Plot raw scatter or hexbin views against each target.
2. Create quantile-binned plots showing the bin mean or median target, a confidence interval, the bin count, and the feature range.
3. Overlay a LOWESS or spline smoother only when its local support is shown.
4. Repeat the plots for total children and each cohort. Do not infer cohort shares for zero-total buildings.

These plots show marginal descriptive shape directly. Partial dependence is not the only way to see nonlinearities: it is a later conditional, model-dependent view. Expect the configured SES-square and daycare-saturation effects to be more visible in binned or smoothed curves than in one-number correlations.

**Completion check:** SES and daycare have raw, binned, and smoothed views for the total and cohort targets, with sparse regions visibly identified.

### Step 5: Exploratory Conditional Nonlinearities

Fit an exploratory tree ensemble only to create conditional visualizations. It is not a selected prediction model, a distributional model, or a source of uncertainty intervals.

1. Use the existing dependency `HistGradientBoostingRegressor(loss="poisson")`; do not add LightGBM during this EDA milestone.
2. Fit one exploratory conditional-mean model per total/cohort target using prediction-time features only. Encode `school_status` without changing the source EDA table.
3. Do not use Random Forest as the primary exploratory count model. Its piecewise-constant predictions, lack of a native scikit-learn Poisson objective, and poor extrapolation make it less useful here.
4. Produce one-dimensional partial-dependence and sampled ICE plots for SES, daycare, household size, median age, and apartment count.
5. Keep two-dimensional empirical interaction views in Step 6. Defer model-based 2D partial dependence and ALE until a later diagnostic pass establishes that they add information beyond the supported-data views.
6. State clearly that partial dependence can evaluate feature combinations with weak support because contextual features are correlated. Limit plotted grids to observed central ranges and show the marginal observations behind each conditional curve.

Interpret all conditional plots as predictive or descriptive patterns, not causal effects. Restrict attention to supported feature ranges.

**Completion check:** Conditional plots are accompanied by raw/binned views and a note describing correlation and support limitations.

### Step 6: Interaction Views And Findings

1. Create faceted binned target curves for daycare by median-age quantile and SES by household-size quantile.
2. Create two-dimensional binned heatmaps for the same pairs, including the observation count in each cell or a mask for sparse cells.
3. Examine room composition jointly with `n_apartments`; do not interpret room counts without building size.
4. Finish with an EDA findings table listing distributional properties, missingness, collinearity constraints, candidate nonlinearities, plausible interactions, unsupported regions, and implications for the deferred modeling stage.

**Completion check:** Every suggested interaction has a visual inspection and the findings table distinguishes evidence from untested hypotheses.

## 5. Count-Distribution Decision For Later Modeling

Use an NB2, equivalently Gamma-Poisson, likelihood as the planned probabilistic model for total building children. A Poisson likelihood assumes:

$$
\operatorname{Var}(Y \mid x) = \mu,
$$

whereas NB2 allows the expected overdispersion:

$$
\operatorname{Var}(Y \mid x) = \mu + \frac{\mu^2}{\phi}.
$$

The EDA tree model may use a Poisson objective to estimate a flexible conditional mean, but it does not model NB2 dispersion and must not be used to produce calibrated count intervals. In the deferred modeling phase, compare Poisson and NB2 using held-out count likelihood, deviance, and residual-dispersion diagnostics. Prefer NB2 when it materially improves held-out distributional fit.

## 6. Next Plan

After the EDA completion checks pass, implement the deferred modeling, evaluation, feature-importance, Bayesian, and MLflow stages described in [MODEL_FITTING_EVALUATION_AND_FEATURE_IMPORTANCE_PLAN.md](MODEL_FITTING_EVALUATION_AND_FEATURE_IMPORTANCE_PLAN.md). The deployment-matched split strategy is defined there.

## 7. Agent Model Guide

Select the strongest model only where it reduces genuine risk. The listed names are capability tiers, so use the closest available variant in the agent picker.

| Work item | Recommended agent model | Why |
|---|---|---|
| Steps 1-3 maintenance; straightforward plots and summaries | Sonnet | Clear local changes with common pandas, seaborn, and matplotlib patterns |
| Step 4 marginal nonlinearities | Sol | Requires careful statistical presentation, support-aware binning, and notebook ergonomics |
| Step 5 HGB Poisson, PD, and ICE | Sol | Strong technical reasoning for model diagnostics without turning the exploratory model into formal evaluation |
| Step 6 empirical interactions and findings implementation | Sol | Requires sparse-cell handling and consistent comparisons across total/cohort targets |
| Review EDA conclusions, leakage register, and collinearity interpretation | Terra or Opus | Best for skeptical review of statistical claims, support limitations, and hidden evaluation errors |
| Mechanical configuration or documentation edits | Sonnet | Low-risk, bounded changes |
| Implement later independent NB2 total and child-level categorical probability models | Sol or Terra | Needs disciplined separation of estimands, split, weighting, and calibration reasoning |
| Implement later Bayesian conditional NB2 and Dirichlet-multinomial model | Terra or Opus | Needs careful conditional likelihood, priors, NUTS diagnostics, and posterior-predictive reasoning |
| Broad independent code review before a merge | Codex or Opus | Use Codex for repository-wide implementation review; use Opus for design and statistical review |

Use Sonnet for bounded maintenance of Steps 1-3. Use Sol to implement Steps 4-6, including the HGB conditional plots, then ask Terra or Opus to review conclusions before moving from EDA to modeling. Use Codex when an independent repository-wide implementation review is needed. The model names can vary by provider configuration; choose by the stated role, not the label alone.

For the conversion, packaging, parity, and Jupyter-removal steps that precede
this EDA implementation, use the primary/review pairings in
[MARIMO_MIGRATION_PLAN.md](MARIMO_MIGRATION_PLAN.md#4-migration-steps-and-model-routing).

## 8. Expected Limitation

This data is static and synthetic. EDA confirms behavior under the configured generator, not real-world predictive validity or causal effects. Before using real data, repeat the availability audit, define the decision time and deployment split, validate target construction, and reassess missingness, distribution shift, fairness, and privacy.
