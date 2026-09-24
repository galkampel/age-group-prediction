# Synthetic Student Population Simulator: Problem Definition

> **Status: advanced target reference.** The current implementation milestone
> is the static apartment-level model in
> [SIMPLIFIED_MODEL_PLAN.md](SIMPLIFIED_MODEL_PLAN.md).

## 1. Purpose

Build a configurable synthetic data generator for residential buildings in
Tel Aviv-Yafo that estimates how many resident children live in each building
three years after first occupancy.

The target has three reported cohorts:

| Cohort | Ages | Birth-year width |
|---|---:|---:|
| Kindergarten | 3-6 | 4 |
| Primary | 7-12 | 6 |
| Secondary | 13-18 | 6 |

The cohort widths are unequal. Per-birth-year comparisons must therefore use
$p_k / w_k$, where $w=(4,6,6)$, rather than comparing the raw cohort shares.

The simulator is intended to:

- exercise the future forecasting pipeline before real data is available;
- quantify error introduced by observable proxies;
- test whether known generating coefficients can be recovered;
- expose assumptions and sensitivity to configurable parameters.

It is not a real forecast. Population levels and most coefficients are
calibrated assumptions rather than measured area-level values.

## 2. Unit, Prediction Time, and Measurement Time

The unit of observation is a **building**, not a project or apartment.
Projects remain important because buildings in one project share zoning,
timing, location, and an unobserved project effect.

For building $b$:

- $t^b_0$ is its construction-start year;
- $t^b_{occ}$ is its first-occupancy year;
- the target is measured three years after $t^b_{occ}$;
- every model-visible environmental feature must be available at $t^b_0$.

The temporal feature contract is:

$$
\operatorname{environment}(b)
= f\left(\operatorname{location}(b), t^b_0\right)
$$

Using the project start, the current environment, or information observed
after construction starts is future leakage.

## 3. Scope Boundary

The generator models the increment associated with new construction. A
separate cohort-progression model is expected to age the existing population
forward. Returning residents therefore reduce the exposed new-unit count;
they are not generated as a second household population.

The primary target contains ages 3-18. The source specification also calls
for a latent ages 0-2 cohort because it feeds future kindergarten demand.
However, it does not define a generating distribution or maturation equation
for that cohort. It is consequently an explicit extension point, not a value
to invent silently during implementation.

## 4. Core Statistical Model

The process has two stages with separate coefficients.

### 4.1 Total children

For net exposed units $U_b^{new}$:

$$
N_b \sim \operatorname{NB2}(\mu_b,\phi_b)
$$

$$
\log \mu_b
= \log U_b^{new}
+ \beta_0
+ \mathbf{x}_b^T\boldsymbol{\beta}
+ \operatorname{interactions}_b
+ u_{project(b)}
+ w_{area(b)}
$$

The offset $\log U_b^{new}$ makes the model learn children per exposed unit.
The project and area effects preserve dependence between related buildings.

In this equation, $N_b$ is the integer total for ages 3-18, $\mu_b$ is its
expected value, $\phi_b$ is NB2 precision, $\mathbf{x}_b$ is the predictor
vector, $\boldsymbol\beta$ is its coefficient vector, $\beta_0$ is the
baseline log rate, and $u$ and $w$ are shared project and area effects. The
NB2 variance is $\mu_b+\mu_b^2/\phi_b$, so larger $\phi_b$ means less extra
variation around the mean.

### 4.2 Cohort composition

Conditional on $N_b$:

$$
\mathbf{q}_b \sim
\operatorname{Dirichlet}(\tau_b\mathbf{p}_b),
\qquad
\mathbf{n}_b \sim
\operatorname{Multinomial}(N_b,\mathbf{q}_b)
$$

A one-dimensional youth index controls the expected split. This enforces the
intended monotone movement between kindergarten and secondary instead of
allowing an unconstrained multinomial model to learn contradictory signs.

Here $\mathbf p_b$ is the expected three-cohort probability vector,
$\mathbf q_b$ is the building's realized probability vector, $\tau_b$ is the
Dirichlet concentration controlling how close $\mathbf q_b$ stays to
$\mathbf p_b$, and $\mathbf n_b$ is the vector of integer cohort counts that
sums to $N_b$.

## 5. Causal Generation Order

The order is part of the model and must not be rearranged:

1. Statistical-area baselines and persistent area characteristics.
2. Projects, including zoning, target size, location, and first occupancy.
3. Buildings, including phasing, form, room mix, apartment areas, and units.
4. Exposure, including realization, lag, fill duration, and net new units.
5. Population panel, including jumps caused by occupancy events.
6. Environment at each building's own construction-start year.
7. Total children.
8. Cohort composition.
9. Returning, intra-city, outside-city, budget, and registration decomposition.

Buildings are generated before the population panel because their occupancy
events create population jumps. Building room mix uses baseline area values,
not the later year-specific panel, which breaks the apparent circularity.

## 6. Truth and Observability

Every table may contain both latent truth and model-visible values. Latent
columns must be clearly named with `_true` or `latent_`. Model input must be
selected with an explicit allowlist; it must never be formed by dropping a
few known latent columns.

Examples of proxy pairs include:

| Latent truth | Model-visible proxy |
|---|---|
| True room shares | Bayesian shares inferred from area categories |
| True floor count and form | Plate-based estimated floors and form |
| True occupancy lag | Expected lag from form and planned units |
| True population values | Observed or interpolated panel values |

Raw values are stored in tables. Standardization occurs only when constructing
the stage-1 and stage-2 model matrices.

## 7. Output Tables

The simulation returns six related tables:

| Table | Grain | Main responsibility |
|---|---|---|
| `areas` | Statistical area | Hierarchy, baselines, persistent character, room prior |
| `projects` | Project | Zoning, timing, target size, building membership |
| `buildings` | Building | Form, room/area mix, units, exposure, visible proxies |
| `population` | Area-year | True and observed demographics, stock, population |
| `environment` | Building | Contemporaneous stock, intensity, SPILL, institutions |
| `results` | Building | Counts, probabilities, concentration, decomposition |

IDs and join cardinalities are part of each table's contract. A building must
belong to exactly one project and area, and results and environment must each
have exactly one row per building.

## 8. Real-Data Replacement Boundary

Population data is synthetic now, but the population provider must support two
implementations with the same schema:

- a synthetic provider driven by area baselines and occupancy events;
- a future CBS/municipal provider that reads manually downloaded files.

The real-data provider must fail with a clear list of missing inputs. It must
never silently fall back to synthetic values.

## 9. Decisions Fixed by This Plan

The implementation plan adopts the following choices where the source
documents are ambiguous:

1. **Project assignment:** sample areas in proportion to baseline housing
   stock by default; expose the weighting rule in configuration.
2. **Project size:** retain both sampled `target_units` and generated
   `planned_units_total`; use bounded project-level resampling when generated
   totals fall outside 100-2,500 units.
3. **Planned-unit rounding:** use nearest-integer rounding, matching the
   executable data-generation guide.
4. **Interpolation horizon:** internally generate 1995-2022 when interpolation
   is enabled, then export 2000-2022. Keep 2017 onward observed.
5. **Institutions:** the first version uses building-time synthetic counts and
   status. A dated institution panel is a future provider with the same output
   contract.
6. **Ages 0-2:** reserve a latent output field, but do not generate it until a
   distribution and its relationship to the reported cohorts are specified.

## 10. Success Criteria

The simulator is complete when it:

- is deterministic for a fixed seed and configuration;
- allows every stage to run and validate independently;
- prevents latent columns from entering model-visible matrices;
- preserves building-specific time and contemporaneous stock joins;
- satisfies structural invariants on every run;
- satisfies calibration checks for the canonical full-size configuration;
- exports enough latent truth to measure proxy and missing-information gaps;
- documents every calibrated, published, structural, and unresolved assumption.

The generation formulas and their execution order are defined in
[DATA_GENERATION_PLAN.md](DATA_GENERATION_PLAN.md). Definitions and intuition
for every default parameter are in
[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md).