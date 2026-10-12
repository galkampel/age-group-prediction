# Simplified Student Simulation Plan

## 1. What the Simulator Must Produce

Build a first synthetic pipeline that predicts resident children for each
**apartment**, splits them into three age cohorts, and aggregates the result to
the **building** level.

The first version is static and has three levels:

```text
neighborhood -> building -> apartment -> building aggregation
```

For each apartment, simulate the total number of resident children and divide
that total into three age cohorts. Then sum the apartment counts to produce the
building-level prediction target.

Reported cohorts are:

| Cohort | Ages | Apartment target |
|---|---:|---|
| Kindergarten | 3-6 | `n_kindergarten` |
| Elementary | 7-12 | `n_elementary` |
| High school | 13-18 | `n_highschool` |

This simulator is for pipeline testing, coefficient recovery, and sensitivity
analysis. Its coefficients describe household sorting and association; they
must not be interpreted as causal effects of changing an apartment or
neighborhood.

## 2. Observable Inputs

| Predictor | Level | Total children | Cohort composition | Reason to keep |
|---|---|---|---|---|
| Rooms | Apartment | Strong positive | More rooms shifts older | Main measure of family capacity and lifecycle |
| SES | Neighborhood | Quadratic | None in Stage 1 | Captures socioeconomic sorting |
| Average household size | Neighborhood | Strong positive | Larger households shift older | Strongest direct demographic signal |
| Daycares within 500 m | Neighborhood | Positive, saturating | Shifts younger | Low-cost family-infrastructure signal |
| School status | Neighborhood | Existing/planned positive | None in Stage 1 | Availability signal without load complexity |
| Median age | Neighborhood | Weak negative | Shifts older | Neighborhood lifecycle signal |

Only variables expected to be available at prediction time are predictors.

## 3. Configuration

All numerical values in this document are configurable defaults, including
distribution parameters, coefficients, random-effect scales, and validation
tolerances. The scale defaults below create about 240
buildings and 9,600 apartments, enough to test recovery without making a run
unnecessarily large.

| Parameter | Default | Meaning and intuition |
|---|---:|---|
| `seed` | `42` | Makes a run reproducible; changing it creates another valid synthetic sample |
| `n_neighborhoods` | `60` | Provides enough neighborhood variation to estimate contextual effects |
| `buildings_per_neighborhood_min` | `1` | Every simulated neighborhood contributes at least one building |
| `buildings_per_neighborhood_rate` | `3.0` | Add a Poisson draw with mean 3, giving 4 buildings per neighborhood on average |
| `apartments_per_building_mean` | `40` | Represents a medium-sized residential building |
| `apartments_per_building_sd` | `12` | Allows meaningful variation in building size |
| `apartments_per_building_min` | `12` | Avoids unrealistically tiny buildings for this use case |
| `apartments_per_building_max` | `80` | Prevents rare normal draws from creating implausibly large buildings |
| `nb_dispersion_phi` | `3.0` | Allows household child counts to vary more than a Poisson model |
| `target_cohort_shares` | `(0.40, 0.35, 0.25)` | Baseline shares for kindergarten, elementary, and high school |

The optional `buildings_per_neighborhood` list contains one exact count per
neighborhood. The optional `apartments_per_building` list contains one exact
count per generated building. When supplied, these lists override their
corresponding distributions. This supports small tests and scenarios based on
known project sizes.

## 4. Output Tables

### Neighborhood

One row per neighborhood:

`neighborhood_id`, `ses`, `avg_household_size`, `median_age`,
`n_daycares_500m`, `school_status`.

School status is categorical with values `existing`, `planned`, and `none`.
Derive the mutually exclusive `existing_school` and `planned_school` dummy
variables only when constructing the model matrix; `none` is the reference.
Storing both the category and its dummies in the canonical table would be
redundant and could create inconsistent rows.

### Building

One row per building:

`building_id`, `neighborhood_id`.

Keep a long room-composition table:

`building_id`, `rooms`, `n_apartments`.

Derive total apartments as:

$$A_b=\sum_{r\in\{3,4,5,6\}}A_{br}$$

The room-composition table is the canonical source of apartment counts. A
derived building view may include `n_apartments` for convenience, but it should
not be stored independently in both tables. Validate that every building has
all required room categories, including explicit zero counts when a category is
absent.

### Apartment

One row per apartment:

`apartment_id`, `building_id`, `rooms`,
`n_kindergarten`, `n_elementary`, `n_highschool`, `n_children_total`.

Derive `neighborhood_id` by joining apartment to building. This normalized
representation avoids storing the same relationship twice. Modeling and export
views may denormalize it after validating the join.

## 5. Transformations and Shared Rules

- Use one `numpy.random.Generator`, seeded once and passed to every stage.
- Use fixed, named transformations whose references and scales are read from
  the `[features]` configuration section:
  $HH_{0.5}=(HH-2.6)/0.5$, $Age_{10}=(Age-37)/10$, and
  $Room_c=R-4$. Keep $SES$ in its original scale because it is already centered
  at zero.
- Therefore, one unit means one SES unit, 0.5 additional household members, 10
  additional years of neighborhood median age, or one additional room. The
  reference apartment has 4 rooms in a neighborhood with household size 2.6
  and median age 37.
- $\mathcal N(m,s^2)$ is normal with mean $m$ and variance $s^2$.
- NB2 uses $E[N]=\mu$ and
  $\operatorname{Var}(N)=\mu+\mu^2/\phi$.
- Store raw features and calculate the named transformations in model-matrix
  construction.
- Draw each neighborhood and building random effect once and reuse it for all
  descendant apartments.
- Keep random effects in private run state. They create clustered outcomes but
  are not model inputs, exported tables, or proxy truth.

## 6. Data Generation

### Step 1: Neighborhoods

Generate `n_neighborhoods` neighborhoods; the default is 60:

$$
SES_j\sim\operatorname{clip}(\mathcal N(0,1),-2.5,2.5)
$$

$$
Age_j\sim\operatorname{clip}(\mathcal N(37,5^2),25,52)
$$

$$
HH_j=\operatorname{clip}
\left(2.60-0.020(Age_j-37)+\epsilon_j,1.8,4.2\right),
\qquad \epsilon_j\sim\mathcal N(0,0.25^2)
$$

$$
D_j\sim\operatorname{Poisson}
\left(\exp[1.0+0.15SES_j-0.030(Age_j-37)]\right)
$$

$$
School_j\sim\operatorname{Categorical}(0.55,0.20,0.25)
$$

where the school probabilities correspond to existing, planned, and none.

| Parameter | Value | Intuition |
|---|---:|---|
| SES mean and standard deviation | 0 and 1 | Uses the familiar socioeconomic index scale centered at 0 |
| SES limits | -2.5 to 2.5 | Retains broad variation but removes extremely rare synthetic values |
| Median-age mean and standard deviation | 37 and 5 years | Represents a mostly family-age population with meaningful neighborhood variation |
| Median-age limits | 25 to 52 years | Avoids implausibly young or old neighborhood medians |
| Household-size baseline | 2.60 people | Approximate household size in a typical neighborhood |
| Age slope in household size | -0.020 per year | A neighborhood 10 years older averages 0.20 fewer people per household |
| Household-size residual SD | 0.25 people | Allows neighborhoods of similar age to differ materially |
| Household-size limits | 1.8 to 4.2 people | Keeps simulated neighborhood averages in a plausible range |
| Daycare log intercept | 1.0 | Gives $e^1\approx2.7$ expected facilities at reference SES and age |
| Daycare SES coefficient | 0.15 | One SES unit multiplies expected facilities by $e^{0.15}=1.16$ |
| Daycare age coefficient | -0.030 per year | A 10-year older neighborhood has about 26% fewer expected daycares |
| School probabilities | 55%, 20%, 25% | Existing schools are common; planned and no-school contexts remain well represented |

These defaults make older neighborhoods slightly smaller in household size and
lower in daycare supply, while higher-SES neighborhoods have somewhat more
daycares. Other correlations are intentionally weak so Stage 1 can verify
coefficient recovery. Correlated neighborhood proxies can be added later.

### Step 2: Buildings

For each neighborhood, draw the configured number of buildings:

$$
B_j=B_{min}+\operatorname{Poisson}(\lambda_B)
$$

The defaults $B_{min}=1$ and $\lambda_B=3$ give an average of four buildings
per neighborhood. For each building, draw its apartment count:

$$
A_b=\operatorname{clip}
\left(\operatorname{round}(\mathcal N(\mu_A,\sigma_A^2)),A_{min},A_{max}\right)
$$

Here, $\mu_A$, $\sigma_A$, $A_{min}$, and $A_{max}$ are the four corresponding
`apartments_per_building_*` configuration values. Round to an integer after
drawing and then apply the limits. A mean of 40 and standard deviation of 12
create mostly 20-60 apartment buildings while still allowing 12-80. When exact
counts are supplied in configuration, skip both random draws.

### Step 3: Room composition

For room counts $k\in\{3,4,5,6\}$, start from:

$$\boldsymbol\pi^0=(0.30,0.38,0.22,0.10)$$

Define the neighborhood tilt:

$$q_j=0.35HH_{0.5,j}+0.20SES_j$$

$q_j$ compresses two neighborhood signals into one direction for the room mix.
If $q_j>0$, larger households or higher SES shift weight toward larger
apartments; if $q_j<0$, weight shifts toward smaller apartments; and if
$q_j=0$, the baseline mix is unchanged. Household size receives more weight
than SES because it is more directly related to room demand.

$$
\widetilde w_{bk}=\pi_k^0
\exp\left(q_j\frac{k-4}{2}\right),
\qquad
w_{bk}=\frac{\widetilde w_{bk}}{\sum_h\widetilde w_{bh}}
$$

For $k=(3,4,5,6)$, the factor $(k-4)/2$ equals
$(-0.5,0,0.5,1)$. Four rooms is therefore the pivot: its multiplier is always
1 before normalization. A positive $q_j$ decreases the relative weight of
3-room apartments and increases the weights of 5- and 6-room apartments, with
the largest change at 6 rooms. Dividing by 2 keeps this rotation gradual. For
example, when $q_j=1$, the four unnormalized multipliers are approximately
$(0.61,1.00,1.65,2.72)$ before they are converted back to probabilities.

Draw a building-specific mix:

$$
\mathbf r_b\sim\operatorname{Dirichlet}(30\mathbf w_b)
$$

Then draw integer room counts that sum to the building total:

$$
(A_{b3},A_{b4},A_{b5},A_{b6})
\sim\operatorname{Multinomial}(A_b,\mathbf r_b)
$$

Assign those room labels to apartment IDs in random order.

| Parameter | Value | Intuition |
|---|---:|---|
| Base shares for 3/4/5/6 rooms | 30%/38%/22%/10% | Four-room apartments are most common; very large apartments are uncommon |
| Household-size tilt | 0.35 per 0.5 people | Larger-household neighborhoods shift the available mix toward larger apartments |
| SES tilt | 0.20 per SES unit | Higher-SES areas have a modest shift toward larger apartments |
| Room distance divisor | 2 | Keeps the tilt gradual rather than moving nearly all mass to an extreme room count |
| Dirichlet concentration | 30 | Allows building mixes to differ, while keeping them near their neighborhood expectation |

Increasing concentration makes buildings more alike; decreasing it creates
more specialized buildings. The multinomial guarantees that room counts sum
exactly to the configured apartment total.

### Step 4: Total children per apartment

Let $R_i$ be the room count for apartment $i$ and $Room_{c,i}=R_i-4$.

Daycare saturation is:

$$S(D_j)=1-\exp(-D_j/3)$$

Draw latent shared effects once:

$$u_b\sim\mathcal N(0,0.25^2),
\qquad w_j\sim\mathcal N(0,0.20^2)$$

The expected total count is:

$$
\begin{aligned}
\log\mu_i={}&-1.30
+0.45\mathbb{1}\{R_i=4\}
+0.80\mathbb{1}\{R_i=5\}
+0.95\mathbb{1}\{R_i=6\}\\
&+0.05SES_j^2
+0.30HH_{0.5,j}
+0.12S(D_j)\\
&+0.10Existing_j+0.04Planned_j
-0.06Age_{10,j}\\
&+0.10Room_{c,i}HH_{0.5,j}+u_b+w_j.
\end{aligned}
$$

Draw:

$$N_i\sim\operatorname{NB2}(\mu_i,\phi=3.0)$$

The coefficients are on the log expected-count scale. Holding other terms
fixed, a coefficient $\beta$ multiplies expected children by $e^\beta$.

| Term | $\beta$ | Practical interpretation |
|---|---:|---|
| Intercept | -1.30 | A reference 3-room apartment has $e^{-1.30}=0.27$ expected children before school, daycare, and random effects |
| 4 rooms | 0.45 | Multiplies the 3-room expectation by 1.57, a 57% increase |
| 5 rooms | 0.80 | Multiplies it by 2.23, a 123% increase |
| 6 rooms | 0.95 | Multiplies it by 2.59, a 159% increase; the gain flattens after 5 rooms |
| SES squared | 0.05 | Creates a mild symmetric U-shape: SES $\pm2$ adds 0.20 to log mean, or about 22% more expected children than SES 0 |
| Household size | 0.30 | Each additional 0.5 people in neighborhood average multiplies expected children by 1.35 |
| Daycare saturation | 0.12 | Moving from no daycares toward full saturation can raise the expectation by at most 13% |
| Daycare saturation scale | 3 facilities | The first few facilities matter most; at 3 facilities the saturation function has reached 63% of its maximum |
| Existing school | 0.10 | Multiplies the no-school reference by 1.11 |
| Planned school | 0.04 | Multiplies the no-school reference by 1.04; weaker because it is not yet operating |
| Median age | -0.06 | A 10-year older neighborhood multiplies expected children by 0.94, after holding the other neighborhood covariates fixed |
| Rooms by household size | 0.10 | Modifies the effect of an extra 0.5 neighborhood household members: the expected-count multipliers are 1.22, 1.35, 1.49, and 1.65 for 3, 4, 5, and 6 rooms respectively |
| Building-effect SD | 0.25 | A typical one-SD building difference changes the mean by a factor of about 1.28 up or 0.78 down |
| Neighborhood-effect SD | 0.20 | A typical one-SD neighborhood difference changes the mean by about 1.22 up or 0.82 down |
| NB2 dispersion $\phi$ | 3.0 | Adds variance $\mu^2/3$ beyond Poisson; smaller values create more household-to-household variation |

Rooms have a nonlinear, flattening effect. Household size is the strongest
contextual driver. Daycare has diminishing returns. The interaction says large
apartments produce more children where local household demand is also high.
The negative median-age coefficient encodes a neighborhood lifecycle signal:
areas with older resident populations are expected to contain fewer children
than otherwise similar younger areas. This is a modeled population-level
association, not a causal claim that an older individual household has fewer
children.
The household-size main effect remains positive for every supported room count.
For a 3-room apartment, one additional 0.5 people in neighborhood average
household size changes the log mean by $0.30-0.10=0.20$, or multiplies expected
children by $e^{0.20}=1.22$. Small apartments in large-household neighborhoods
can therefore still be more crowded; the interaction only makes their increase
smaller than the increase in apartments with more capacity.

`avg_household_size` is a neighborhood-level context variable, not the actual
number of people occupying apartment $i$. The interaction must therefore be
interpreted as **apartment capacity by local household demand**, not as direct
evidence that a particular apartment is crowded. At high household size, the
main effect raises expected children for every apartment, while the interaction
allows larger apartments to accommodate more of that demand.
The positive SES-square coefficient represents the limited hypothesis that
both low- and high-SES contexts may have somewhat larger families than the
middle. It is deliberately weak because SES can proxy cultural, religious, and
demographic differences that the simulator does not observe. Calibrate or
remove this term if empirical validation does not support the U-shape.

$u_b$ and $w_j$ create unexplained clustering within buildings and
neighborhoods. They represent unobserved finish, marketing, accessibility,
local preferences, and similar omitted factors. Because they are generated
independently of observed features, they are **unobserved heterogeneity**, not
confounders in Stage 1. Real omitted factors may correlate with SES, rooms, or
school access and then become genuine confounders.

### Step 5: Cohort composition

Create one ordered lifecycle index. Draw one composition effect per building:

$$v_b\sim\mathcal N(0,0.20^2)$$

$$
\theta_i=
0.30Room_{c,i}
-0.20S(D_j)
+0.25Age_{10,j}
+0.15HH_{0.5,j}
+0.12\max(Room_{c,i},0)\max(Age_{10,j},0)
+v_b
$$

Start from the configured baseline cohort shares and apply the lifecycle index:

$$a_{kg,i}=0.40\exp(-1.0\theta_i)$$

$$a_{el,i}=0.35$$

$$a_{hs,i}=0.25\exp(1.1\theta_i)$$

$$
p_{k,i}=\frac{a_{k,i}}{a_{kg,i}+a_{el,i}+a_{hs,i}}
$$

$$
(N_{kg,i},N_{el,i},N_{hs,i})
\sim\operatorname{Multinomial}(N_i,\mathbf p_i)
$$

When $\theta=0$, probabilities are exactly 40%, 35%, and 25%. Positive
$\theta$ means an older family lifecycle: it lowers kindergarten and raises
high school while elementary stays between them.

| Term | Coefficient | Practical interpretation |
|---|---:|---|
| Additional room | 0.30 | Each room above 4 moves the lifecycle index 0.30 toward older cohorts |
| Daycare saturation | -0.20 | Strong daycare availability moves the index up to 0.20 toward younger cohorts |
| Median age | 0.25 | A 10-year older neighborhood moves the index 0.25 toward older cohorts |
| Household size | 0.15 | An additional 0.5 people moves the index 0.15 toward older cohorts |
| Large rooms in older neighborhoods | 0.12 | Each room above 4 interacts with each positive 10-year age unit; the interaction is zero for small apartments or neighborhoods younger than 37 |
| Building composition SD | 0.20 | Allows otherwise similar buildings to differ modestly in lifecycle |
| Kindergarten lifecycle slope | -1.0 | A one-unit increase in $\theta$ multiplies kindergarten-to-elementary odds by 0.37 |
| High-school lifecycle slope | 1.1 | A one-unit increase in $\theta$ multiplies high-school-to-elementary odds by 3.00 |

The one-sided interaction represents move-up families: a large apartment in an
established neighborhood attracts older children more strongly than the same
apartment in a young neighborhood. Using positive parts is intentional. The
ordinary product $Room_cAge_{10}$ would also be positive for a small apartment
in a young neighborhood because both centered values are negative. That would
partly cancel their two younger-cohort main effects without a clear mechanism.
The one-sided form leaves the other three room-age combinations to their main
effects and adds a bonus only for large apartments in older neighborhoods.

Generic school status does not alter cohort composition in Stage 1. Knowing
that a school exists does not identify whether it serves kindergarten,
elementary, or high-school ages. Stage-specific school access can be added when
the school stage is observed.

The shared $v_b$ makes apartments in one building similar in cohort age. A
plain multinomial is sufficient for this first version; add a
Dirichlet-multinomial only if composition remains overdispersed after accounting
for $v_b$.

The 40%/35%/25% values are **reference shares**, not a requirement that the
whole generated sample have exactly those shares. The sample average can differ
because its apartments and neighborhoods do not all have $\theta=0$. This
removes the need for a separate, hard-to-interpret intercept-calibration loop.

### Step 6: Aggregate

For each building and cohort:

$$N_{b,k}=\sum_{i\in b}N_{i,k}$$

Also store total building children and apartment count. Keep apartment targets
wide for modeling and aggregation; create a derived long view only for plots.

## 7. Observed Features, Proxies, and Internal Variation

Stage 1 has no **measurement proxy** for rooms: true room count is directly
available. That does not mean every feature is a causal quantity.

| Variable | Interpretation |
|---|---|
| Rooms | Directly observed apartment attribute; proxy for household capacity and lifecycle |
| SES | Contextual proxy for economic and social conditions that are not individually observed |
| Household size | Area-level proxy for the family demand facing a particular apartment |
| Daycares | Proxy for family infrastructure and neighborhood family orientation |
| School status | Proxy for education access; availability is not the same as quality or capacity |
| Median age | Proxy for neighborhood lifecycle and household turnover |
| $u_b,w_j,v_b$ | Internal shared heterogeneity creating within-group dependence |

The synthetic random effects are independent of observed predictors, so Stage 1
can test coefficient recovery. In real data, unobserved household preferences
and building characteristics can correlate with observed proxies, so a fitted
coefficient should still be interpreted as predictive association rather than
causation.

Random effects are not hidden truths for proxy evaluation. A future proxy
experiment must generate an explicit truth and its prediction-time proxy, such
as true room count and room count inferred from apartment size. Approved
truth/proxy pairs may be exposed in research tables; random effects remain
private implementation state.

## 8. Validation

### Structural checks

- Same seed and configuration produce identical tables.
- IDs and joins have the expected one-to-many relationships.
- Generated building and apartment counts respect all configured limits.
- Exact configured building and apartment lists are reproduced without random
  changes.
- Room counts sum to building apartment totals.
- Every building has one room-composition row for each supported room count;
  absent categories are represented by zero.
- Derived school dummies are mutually exclusive and agree with `school_status`.
- Apartment-to-building-to-neighborhood joins are complete and unique.
- Cohort counts are non-negative integers and sum to apartment total children.
- Building totals equal grouped apartment totals exactly.

### Statistical checks

| Check | Initial expectation |
|---|---|
| Mean children per apartment | 0.40-0.60 |
| Apartments with zero children | 60-75% |
| Apartment variance/mean | Clearly above 1 |
| Reference composition | At $\theta=0$, exactly 40%/35%/25% |
| Simulated mean composition | Report and monitor; do not force it to equal the reference shares |
| Mean children rises from 3 to 5 rooms | Required; 6-room increment may flatten |
| Building variance/mean | Clearly above 1 |
| Room effect strengthens with household size | Required for the total-count interaction |
| Household-size effect remains positive for 3, 4, 5, and 6 rooms | Required; verify $0.30+0.10(R-4)>0$ for every supported room count |
| Room by household-size scenario grid | Report expected means for every room count across household sizes 1.8, 2.6, 3.1, 3.6, and 4.2 |
| Extreme 6-room/high-household prediction | Inspect during calibration so the positive interaction does not create implausibly large counts |
| Large-room effect shifts composition older more strongly only in neighborhoods older than 37 | Required for the one-sided composition interaction |
| Small rooms in young neighborhoods receive no room-age interaction | Required; their younger shift comes only from the main effects |

Do not freeze the proposed exact room means or variance ratio until a canonical
run verifies that all interactions and random effects behave as intended.

### Model recovery

A plain NB2 GLM that ignores $u_b$ and $w_j$ does not reproduce the generating
likelihood and will understate uncertainty. Use one of these explicit tests:

1. **Oracle recovery:** include known generated random effects as offsets and
   verify fixed coefficients.
2. **Hierarchical recovery:** fit an NB mixed model and verify fixed effects and
   random-effect scales.
3. **Naive benchmark:** fit a plain NB2 GLM, but label deviations and narrow
   standard errors as the expected cost of ignored clustering.

The implemented slow recovery gate runs the oracle and naive NB2 fits. The
naive fit is diagnostic, not a generator acceptance criterion.

For zero inflation, compare the observed share with the analytic expected NB2
zero probability using apartment-specific generating means. Add a
hurdle or zero-inflated model only if observed zeros exceed the NB2 prediction
materially and consistently across seeds.

## 9. Complexity Roadmap

Add complexity only after the current stage passes its structural, distribution,
and recovery checks. Each extension should have a measurable reason to exist.

### Stage 2: Add the remaining observable baseline features

Add only variables that are expected to be available when the prediction is
made:

- Add population density and its change over recent years.
- Add planned-versus-realized apartment counts when both fields are observable.
- Add school load only as utilization multiplied by existing-school status.
- Add state-education share only when the target is restricted to that stream.
- Add room-mix contrast: the building's mean room count minus the surrounding
  neighborhood's mean. Positive contrast identifies local move-up housing.

Each proxy needs an availability audit, a documented reference date, and an
ablation test showing that it improves validation beyond SES and the existing
neighborhood variables.

For the high-school cohort, retain the signals already present in the older
design rather than inventing new facility categories:

- mean household size;
- large-apartment share and room-mix contrast;
- neighborhood median age;
- state-education share when predicting that education stream.

Do not use the neighborhood count or share of children aged 13-18 when it is
measured at or after the prediction date: it directly overlaps the target and
would cause leakage. Add any candidate only when it is known at prediction
time and improves held-out validation.

### Stage 3: Add time, projects, and building type

Time is a generator of the conditions seen by a building, not an additional
raw coefficient in the child-count equation. Generate dates in causal order:

$$
t_p^{permit}\le t_b^{start}<t_b^{occupancy}<t_b^{measurement}
$$

- $t_p^{permit}$ is the project permit date.
- $t_b^{start}$ is the building's construction-start and prediction date.
- $t_b^{occupancy}$ is its first-population or first-occupancy date.
- $t_b^{measurement}=t_b^{occupancy}+\Delta_{measure}$ is when resident
  children are counted.

Use a configurable measurement lag with a default of three years and a
two-to-three-year scenario range. Three years places measurement after most
initial filling. If the lag varies within a run or becomes shorter, add an
explicit occupancy/fill curve rather than treating every building as full.

Introduce a project table because one project can contain multiple phased
buildings. Draw one project type and inherit it at the building level:

- `new_neighborhood`: generally more buildings, more high-rise construction,
  weaker reliance on the existing neighborhood population, and mostly net-new
  units;
- `urban_renewal`: generally fewer buildings, stronger connection to existing
  population, and an explicit distinction between gross units, net-new units,
  and returning residents.

Project type controls the distribution of project size, number of buildings,
planned units per building, construction phasing, and building type. Generate
it once per project so all buildings in that project share the same development
context while retaining their own start and occupancy dates.

#### Net-new student exposure

Stage 1 predicts all resident children. In this future stage, add a separate
exposure decomposition that converts resident counts into **net-new students**
for a fixed geographic planning boundary, normally the building's statistical
area. A move from another statistical area counts as new local demand even if
the household remains in the same city.

Assign one origin to each apartment household, so siblings move together:

- `returning`: returns to a previous dwelling or the urban-renewal project
  site;
- `within_area`: moves from elsewhere inside the same statistical area;
- `outside_area`: arrives from outside the statistical area.

Only `outside_area` households contribute to the net-new target:

$$
N^{new}_{i,k}=N^{resident}_{i,k}
\mathbb{1}\{Origin_i=outside\_area\}.
$$

Returning households are already part of the local baseline, while within-area
movers relocate existing local demand. Keep resident and net-new counts side by
side so the subtraction is auditable and the resident-population simulator can
still be validated independently.

Origin probabilities must be configurable and depend on project type.
Urban-renewal projects should generally have a higher returning share.
New-neighborhood projects should generally have a lower returning share and a
higher outside-area arrival share. Room-mix contrast may increase the
within-area share because locally larger apartments attract move-up households
from the surrounding area.

Allow origin shares to change over calendar time only when a defensible trend
is configured; do not infer that trend from the generated student counts. At
minimum, validate shares and net-new/resident ratios by year, project type, and
cohort. The shares must sum to one, net-new counts must never exceed resident
counts, and returning plus within-area households must contribute zero to the
net-new target.

#### Dynamic building types

The future synthetic generator may retain a true floor count internally so it
can define the true building category:

- `low_building`: 10 floors or fewer;
- `high_building`: more than 10 floors.

The deployable model must not use the true floor count because it is unknown at
prediction time. The known input is each building's `planned_units`. Apply a
fixed, configurable mapping from planned units to an estimated floor count, and
then classify that estimate with the same 10-floor threshold:

```text
planned_units
    -> fixed floor-estimation rule
    -> estimated_floors
    -> estimated_building_type: low or high
```

The exact estimation function is intentionally deferred. Before implementing
this stage, define it from planning knowledge or historical projects, freeze
its parameters in configuration, and ensure it uses only information available
at construction start. Use `estimated_building_type`, rather than
`estimated_floors`, as the normal model feature: the category captures the
intended building-form distinction without implying that the estimated floor
count is exact.

Building form must change over time. Early periods should contain relatively
few high buildings, while later periods contain more. Project type modifies
that trend because new-neighborhood projects are generally more likely to use
high buildings than urban-renewal projects. A future truth generator can use:

$$
\operatorname{logit}P(HighRise_b=1)=
\alpha_0+\alpha_t(t_b^{start}-t_{ref})
+\alpha_p\mathbb{1}\{NewNeighborhood_p\}+c_j,
$$

where $\alpha_t>0$ makes high-rise construction more common over time,
$\alpha_p>0$ makes it more common in new-neighborhood projects, and $c_j$ is a
persistent neighborhood tendency toward high- or low-rise construction.

Time may also change planned units and the number of buildings per project.
Consequently, the estimated building-type distribution can become more
high-rise over time through both the changing construction regime and later
project configurations. Check the trend separately within `new_neighborhood`
and `urban_renewal`; otherwise a changing mix of project types could be
mistaken for a building-type trend.

Test a weak negative direct high-rise coefficient on expected children per
apartment. It represents household sorting, not a causal effect of height.
Retain it only if it adds predictive value after controlling for room mix,
project type, SES, household size, intensity, and construction period; otherwise
let those observed mechanisms explain the difference.

Keep these fields conceptually separate during simulation and validation:

| Field | Role |
|---|---|
| `planned_units` | Observed prediction-time input |
| `true_building_type` | Synthetic truth derived from internally generated true floors; never used by the deployable model |
| `estimated_building_type` | Prediction-time proxy derived from planned units; used by the deployable model |

Validate this extension by reporting the high-building share by year and
project type, proxy classification accuracy near the 10-floor threshold, and
model performance under three specifications: planned units alone, planned
units plus `estimated_building_type`, and an oracle using
`true_building_type`. This shows whether the category contributes useful form
information beyond planned units and quantifies the cost of not knowing true
floors.

Buildings within one project receive separate start and occupancy dates. A
later building must see the population, housing stock, and nearby construction
environment at its own $t_b^{start}$, not the first building's date.

### Stage 4: Add population and environment dynamics

Create a neighborhood-year panel and evaluate it at each building's prediction
date. Restore the dynamics from the advanced design:

- average household size changes gradually over time and carries the fertility
  trend;
- SES follows a neighborhood-specific trend and can jump when newly occupied
  construction changes local composition;
- median age changes gradually and can shift after occupancy events;
- housing stock and population update when buildings become occupied;
- occupancy events create explicit population and demographic jumps.

Do not also add an unrestricted calendar-year coefficient to the child model.
Household size and the other time-varying predictors already carry the relevant
period changes, so a direct year term could double-count them.

Restore only the environment features from the older design, all evaluated at
$t_b^{start}$:

- daycare count, allowed to evolve with housing stock and calendar time;
- school status and school load, with load active only when a school exists;
- contemporaneous housing stock as a denominator and diagnostic;
- project/building intensity relative to contemporaneous stock;
- `SPILL`, the recently occupied units near the building within a configurable
  distance and lookback window;
- infrastructure lag, so residential growth can temporarily outpace daycare
  and school provision.

Do not add parks, clinics, community facilities, or generic public-transport
features to this roadmap. They were not part of the prior design and should
require a separate data-availability and mechanism review.

### Stage 5: Connect the cohort-progression model

- Keep this simulator responsible for the initial short- to medium-term
  population of a newly populated building.
- Define an output contract containing the cohort counts and a reference date.
- Pass that output to a separate cohort-progression model that ages the existing
  population forward.
- Add ages 0-2 only if the progression model needs them as an input cohort.

Do not mix aging transitions into this initial-population model. The two models
answer different questions and can be validated independently.

### Stage 6: Deployment and decision quality

- Add cold-start shrinkage for neighborhoods with no usable history.
- Tune forecasts for asymmetric under- versus over-capacity cost.
- Monitor calibration by neighborhood, building size, and cohort.
- Stress-test distribution shift and uncertain proxy mappings.
- Before using real data, repeat the availability audit, define the decision
  time and deployment split, validate target construction, and reassess
  missingness, fairness, and privacy.

## 10. Exclusions and Extension Criteria

The baseline contains the minimum useful signals: apartment capacity, local
household demand, lifecycle, socioeconomic context, family infrastructure, and
shared building/neighborhood heterogeneity.

The following variables and mechanisms are intentionally excluded:

| Exclusion | Reason |
|---|---|
| Observed floor count | Unavailable at prediction time and excluded from the deployable model; roadmap Stage 3 may generate true floors internally only to evaluate the planned-units building-type proxy |
| Apartments per floor | Unavailable and not needed; total apartments are modeled directly |
| Apartment price | Unavailable and will not be part of later extensions; SES, density, and facilities are observable contextual proxies |
| Tenure | Ownership versus rental status is unavailable and is not expected to improve the short- to medium-term initial-population prediction enough to justify a separate mechanism |
| Apartment area | Not required while room count is available; do not add it unless it becomes observable and improves validation |
| Historical fertility by generation | Current neighborhood median age measures neighborhood lifecycle, not how many births an older generation had; modeling historical fertility requires birth cohort and calendar time |
| Internal aging | Belongs to the separate cohort-progression model |
| Ages 0-2 | Not one of the current prediction targets; add only if required by the progression-model interface |
| Project phasing and renewal exposure | Requires dates and exposure boundaries; add in roadmap Stages 3-4 after the static model is validated |
| Registered-pupil leakage | Resident-to-registration conversion is separate from household-origin exposure; add it only when the target is registered pupils rather than resident children |
| Zero-inflation component | Add only if NB2 consistently underpredicts observed zeros across seeds |

Excluded variables should not remain as unused columns in the generated tables.
An extension should add a variable only when it is observable at prediction
time, has a clear mechanism, and improves out-of-sample validation.

## 11. Recommended Document Status

Use this document as the canonical implementation plan for the next milestone.
Keep the existing complex plan, but relabel it as an **advanced target design**,
not the current implementation specification.

Do not delete the complex documents. They preserve formulas and rationale for
Stages 2-5 and prevent those choices from being reinvented. Keeping both is
safe only if the documentation index clearly identifies:

- this simplified plan as **current**;
- the complex plan as **future reference**;
- the original source specifications as **legacy input**.