# Synthetic Data Generation Plan

> **Status: advanced target reference.** Use
> [SIMPLIFIED_MODEL_PLAN.md](SIMPLIFIED_MODEL_PLAN.md) for the next
> implementation milestone. This document preserves later-stage complexity.

## 1. Reading This Guide

This document is the canonical, implementation-ready plan for generating the
synthetic data. It defines logic and formulas only; it does not implement them.

Each stage has five parts:

- **Requires:** tables or values that must already exist.
- **Produces:** the stage's new table or columns.
- **Logic:** formulas and sampling order.
- **Visibility:** which outputs are latent or model-visible.
- **Validation gate:** checks that must pass before the next stage.

The governing scope and data contracts are in
[PROBLEM_DEFINITION.md](PROBLEM_DEFINITION.md). Planned software boundaries are
in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Definitions, units, and
intuition for every configured value are collected in
[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md).

### 1.1 Notation and distribution conventions

Subscripts identify the level at which a value varies:

- $d$ is a district, $a$ is a statistical area, $p$ is a project, $b$ is a
   building, $t$ is a calendar year, $k$ is a room or cohort category, and $j$
   is a floor-area category.
- $a(b)$ and $p(b)$ mean the area and project containing building $b$.
- A bar, as in $\bar H$ or $\bar A$, denotes a mean. A hat denotes an
   observable estimate. A `true` superscript denotes latent generating truth.
- $\mathbb{1}\{condition\}$ is 1 when the condition is true and 0 otherwise.
- `clip(x, lower, upper)` truncates a realized value to the stated bounds.

Distribution parameters use these conventions throughout:

- $\mathcal{N}(m,s^2)$ is normal with mean $m$ and variance $s^2$.
- $U(l,u)$ is continuous uniform on $[l,u]$; `UniformInteger(l,u)` includes
   both integer endpoints.
- $\operatorname{Beta}(\alpha,\beta)$ uses positive shape parameters, with
   mean $\alpha/(\alpha+\beta)$.
- $\operatorname{LogNormal}(m,s^2)$ means the natural logarithm is
   $\mathcal{N}(m,s^2)$.
- $\operatorname{Gamma}(shape,scale)$ has mean `shape * scale`.
- $\operatorname{Dirichlet}(\boldsymbol\alpha)$ has mean
   $\boldsymbol\alpha/\sum_k\alpha_k$; the sum of its parameters controls
   concentration.
- $\Phi$ is the standard-normal cumulative distribution function.
- $\operatorname{sigmoid}(x)=1/(1+e^{-x})$.
- `softmax` maps logits $\boldsymbol\eta$ to probabilities
   $p_k=e^{\eta_k}/\sum_h e^{\eta_h}$.

Unless a formula explicitly says otherwise, logarithms are natural logarithms,
time coefficients are per calendar year, coordinates are kilometres, areas are
square metres, and unit counts refer to dwelling units.

## 2. Global Rules

1. Create one `numpy.random.Generator` from the run seed and pass it through
   every stochastic stage. Never reseed inside a stage.
2. Put every numeric assumption in typed configuration. Keep canonical default
   values in one searchable location.
3. Store raw, interpretable values. Standardize only while constructing model
   terms.
4. Keep latent truth beside its visible proxy and use explicit visible-column
   allowlists.
5. Raise on failed invariants. Do not warn and continue.
6. Use kilometres for all coordinates and spatial radii.
7. Key temporal features on each building's own construction start.

Default simulation scale:

$$N_{areas}=180,\qquad N_{districts}=9,\qquad N_{projects}=320$$

Default exported years are 2000-2022. If interpolation is enabled, internal
population anchors begin in 1995.

## 3. Pipeline Order

```text
areas
  -> projects
  -> buildings
  -> exposure
  -> occupancy events
  -> population panel
  -> building-time environment
  -> stage 1 total children
  -> stage 2 cohort split
  -> output decomposition
```

No later stage may modify the meaning of an earlier table. It may only append
derived columns or create a new table.

## 4. Stage 1: Statistical Areas

### Requires

- Run configuration.
- Shared random generator.

### Produces

One row per statistical area with district, centroid, baseline housing and
population, demographics, persistent built character, and room prior.

### Logic

Assign each of 180 areas independently and uniformly to one of 9 districts,
then assert that all configured districts are represented. Draw district
effects once and reuse them for all areas in that district:

$$
SES_d \sim \mathcal{N}(0,1),
\qquad
gentr_d \sim \operatorname{Uniform}(0,0.045)
$$

For area $a$ in district $d(a)$:

$$
SES_a = SES_{d(a)} + \epsilon_a,
\qquad
\epsilon_a \sim \mathcal{N}(0,0.55^2)
$$

$$
gentr_a = gentr_{d(a)} + \zeta_a,
\qquad
\zeta_a \sim \mathcal{N}(0,0.008^2)
$$

$$
\bar H_a \sim \operatorname{clip}
\left(\mathcal{N}(2.55,0.30^2),1.8,4.2\right)
$$

$$
medage_a \sim \operatorname{clip}
\left(\mathcal{N}(36.5,4.0^2),26,50\right)
$$

$$
stock_a(0) \sim \operatorname{UniformInteger}(900,6500)
$$

Draw unscaled population from
$\operatorname{UniformInteger}(2500,9000)$, then rescale and integer-balance
the values so their exact sum is 468,000.

Draw coordinates uniformly over an approximately 9 km by 12 km city box, and
draw persistent built character once:

$$x_a\sim U(0,9),\quad y_a\sim U(0,12),\quad c_a\sim\mathcal{N}(0,1)$$

Generate state-education share as a spatially heterogeneous mixture:

$$
state_a \sim
\begin{cases}
\operatorname{Beta}(8.0,2.5), & \text{with probability }0.85\\
\operatorname{Beta}(3.0,5.5), & \text{with probability }0.15
\end{cases}
$$

Clip to $[0.08,0.99]$, then apply a bounded calibration that preserves order
while making the city mean 0.70.

Define rooms and baseline weights:

$$
\mathbf{k}=(2,3,4,5,6),
\qquad
\boldsymbol{\pi}^0=(0.10,0.28,0.34,0.20,0.08)
$$

Standardize area baseline SES and household size across areas. The
area-conditional room prior is:

$$
\tilde\pi_{ak}
= \pi^0_k
\exp\left[
\left(0.25z_{SES,a}+0.45z_{\bar H,a}\right)\frac{k-4}{2}
\right]
$$

$$
\pi_{ak}=\frac{\tilde\pi_{ak}}{\sum_j\tilde\pi_{aj}}
$$

The area's mean room profile is
$\bar k_a=\sum_k k\pi_{ak}$.

Here and below, $z_{v,a}$ means the within-run z-score of variable $v$ at the
relevant table grain. Area-prior z-scores are fitted across areas. Stage-1 and
stage-2 z-scores are fitted across buildings after all required tables are
joined; the two standardization contexts must not be mixed.

### Visibility

Area baselines, room priors, and built character belong in the area lookup.
Built character is known to the generator but is not a downstream building
feature unless a future model explicitly permits it.

### Validation gate

- Exactly 180 unique area IDs and 9 represented district IDs.
- Every area belongs to exactly one district.
- Population sums exactly to 468,000.
- All clipped variables lie in their configured ranges.
- State share has mean 0.70 within numerical tolerance and remains bounded.
- Every room-prior row is positive and sums to one.
- Area centroids use kilometre-scale bounds.

## 5. Stage 2: Projects

### Requires

- Validated areas table.

### Produces

One row per project with area, zoning, target units, first occupancy, centroid,
building count, returning share, and net-budget share.

### Logic

Select an area with default probability proportional to baseline housing stock:

$$
\Pr(a_p=a)=\frac{stock_a(0)}{\sum_j stock_j(0)}
$$

This weighting is calibrated and configurable.

Draw target size and first occupancy:

$$
T_p=\operatorname{clip}
\left(\operatorname{LogNormal}(\log 450,0.75^2),100,2500\right)
$$

$$t^{first}_p\sim\operatorname{UniformInteger}(2000,2022)$$

Let $s_p=t^{first}_p-2000$. New-neighbourhood probability is:

$$
P_{new}(s_p)=\operatorname{sigmoid}(1.20-0.13s_p)
$$

If the project is not a new neighbourhood, split zoning probabilities as:

$$
(P_{clearance},P_{38/2},P_{38/1})=(0.35,0.40,0.25)
$$

Use zoning-specific expected units and minimum building counts:

| Zoning | Units/building | Minimum buildings | $r_{ret}$ | $s_{net}$ |
|---|---:|---:|---:|---:|
| `new_hood` | 70 | 4 | 0.00 | 1.00 |
| `clearance` | 85 | 2 | 0.25 | 0.65 |
| `tama38_2` | 60 | 1 | 0.55 | 0.45 |
| `tama38_1` | 40 | 1 | 0.85 | 0.15 |

$$
n_p=\operatorname{clip}
\left(
\operatorname{round}\left(\frac{T_p}{m_{z_p}}\right),
n^{min}_{z_p},14
\right)
$$

Project coordinates are offset from the selected area centroid:

$$
(x_p,y_p)=(x_a,y_a)+\mathcal{N}(0,0.25^2I)
$$

### Visibility

All project fields are structural or derived. Random effects used later are
latent model-stage values and should be stored with a `_true` suffix if kept.

### Validation gate

- Exactly the configured number of unique projects.
- Target size is in $[100,2500]$.
- Building count is in $[1,14]$ and respects zoning minima.
- No single-building new-neighbourhood or clearance project.
- First occupancy is in 2000-2022.
- Every project references one valid area.

## 6. Stage 3: Buildings and Observable Proxies

### Requires

- Validated projects and areas.

### Produces

One row per building with occupancy timing, geometry, true form, room mix,
floor-area categories, planned units, estimated room shares, and form proxy.

### Logic

#### 6.1 Phasing and form

For each of the project's $n_p$ buildings:

$$
t^{occ}_b=\operatorname{clip}
\left(
t^{first}_p
+ U\{0,\ldots,n_p-1\}
+ U\{0,1\},
2000,2022
\right)
$$

Let $s_b=t^{occ}_b-2000$. Draw a provisional form indicator using the area's
persistent built character:

$$
P(high_b=1)=\operatorname{sigmoid}
\left(-1.50+0.11s_b+1.2c_{a(b)}\right)
$$

Floors are the primitive:

$$
F_b^{true}=\begin{cases}
12+\operatorname{Poisson}(10), & high_b=1\\
4+\operatorname{Poisson}(3), & high_b=0
\end{cases}
$$

Clip floors to $[3,45]$, then redefine true form as
$\mathbb{1}\{F_b^{true}>10\}$.

Building coordinates are:

$$
(x_b,y_b)=(x_p,y_p)+\mathcal{N}(0,0.06^2I)
$$

#### 6.2 True room mix

Starting from the area prior, define the building tilt:

$$tilt_b=-0.35high_b-0.020s_b$$

$$
\tilde w_{bk}=\pi_{a(b),k}
\exp\left(tilt_b\frac{k-4}{2}\right),
\qquad
w_{bk}=\frac{\tilde w_{bk}}{\sum_j\tilde w_{bj}}
$$

Draw true building room shares:

$$
\mathbf{r}_b^{true}\sim\operatorname{Dirichlet}(45\mathbf{w}_b)
$$

The true mean room count is
$\bar k_b^{true}=\sum_k kr_{bk}^{true}$.

#### 6.3 Floor area conditional on rooms

For room count $k$:

$$
\mu_{bk}=8+22k+4z_{SES,a}-3high_b-0.15s_b-2.5z_{\bar H,a}
$$

$$
A\mid k,a,b\sim
\operatorname{clip}\left(\mathcal{N}(\mu_{bk},9^2),22,260\right)
$$

Obtain each building's floor-area distribution as the room-mixture distribution,
then aggregate it into categories:

$$[0,50),\quad[50,85),\quad[85,100),\quad[100,\infty)$$

The preferred implementation computes category probabilities from normal CDF
differences rather than simulating every apartment. This preserves the model
while avoiding an unnecessary apartment-level table.

For category $j=[l_j,u_j)$:

$$
c_{bj}=\sum_k r_{bk}^{true}
\left[
\Phi\left(\frac{u_j-\mu_{bk}}{9}\right)
-\Phi\left(\frac{l_j-\mu_{bk}}{9}\right)
\right]
$$

Normalize only to correct floating-point error. Then:

$$HHI_b=\sum_jc_{bj}^2$$

#### 6.4 Unit count

Use expected mean apartment area under the true room mix:

$$
\bar A_b=\sum_k r_{bk}^{true}\mu_{bk}
$$

Draw a varying floor plate:

$$
P_b^{true}=450\exp(\epsilon_b),
\qquad
\epsilon_b\sim\mathcal{N}(0,0.18^2)
$$

$$
U_b^{planned}=\operatorname{clip}
\left(
\operatorname{round}\left(\frac{F_b^{true}P_b^{true}}{\bar A_b}\right),
8,400
\right)
$$

After all buildings in a project are generated, compute
$\sum_bU_b^{planned}$. If the total lies outside $[100,2500]$, resample that
project's building-level realization up to a configured maximum attempt count.
Failure after the maximum attempts is an error containing the project ID and
its sampled parameters.

#### 6.5 Bayesian room inversion

Downstream users observe area categories, not true rooms. For each building,
category, and room count:

$$
P(k\mid l_j\le A<u_j,a,b)
\propto
\pi_{a(b),k}
\left[
\Phi\left(\frac{u_j-\mu_{bk}}{9}\right)
-\Phi\left(\frac{l_j-\mu_{bk}}{9}\right)
\right]
$$

Normalize over $k$, then aggregate:

$$
\widehat r_{bk}=\sum_jc_{bj}P(k\mid j,a,b)
$$

Store visible shares for 3, 4, and 5-or-more rooms. The 2-room share is the
omitted compositional baseline.

Compute visible mean room count and mix contrast:

$$
\widehat{\bar k}_b=\sum_k k\widehat r_{bk},
\qquad
\Delta k_b=\widehat{\bar k}_b-\bar k_{a(b)}
$$

Estimate floors and form using the standard plate:

$$
\widehat F_b=\frac{U_b^{planned}\bar A_b}{450},
\qquad
\widehat{form}_b=\mathbb{1}\{\widehat F_b>10\}
$$

### Visibility

Latent fields include true floors, true form, true room shares, true mean
rooms, true plate, and the generating area mixture. Category shares, estimated
room shares, HHI, estimated form, planned units, and mix contrast are visible.

### Validation gate

- One or more buildings per project and unique building IDs.
- Occupancy is no earlier than project first occupancy and no later than 2022.
- True and estimated room shares each sum to one.
- Floor-area category shares sum to one and HHI lies in $[0.25,1]$.
- Planned units lie in $[8,400]$ per building.
- Aggregate planned project units lie in $[100,2500]$.
- True form equals the thresholded true floor count.
- Estimated room probabilities are finite and non-negative.
- Mix contrast has both positive and negative observations at canonical scale.

## 7. Stage 4: Exposure

### Requires

- Validated buildings and project zoning.

### Produces

Realization, construction lag, fill duration, construction start, expected lag,
and net exposed units.

### Logic

For planned units $U_b^{planned}$:

$$\nu_b=4+0.05U_b^{planned}$$

$$
R_b^{true}=0.7+0.5B_b,
\qquad
B_b\sim\operatorname{Beta}(0.6\nu_b,0.4\nu_b)
$$

For buildings below 150 planned units, replace $R_b^{true}$ with a draw from
$U(1.5,2.0)$ with probability 0.03.

$$
L_b^{true}=2+2\operatorname{Beta}(2,2)
$$

$$
D_b^{fill,true}=0.5+2\operatorname{Beta}(1.5,3.5)
$$

$$
t^b_0=\operatorname{round}(t_b^{occ}-L_b^{true})
$$

Do not clip construction start to 2000: a building first occupied in 2000 may
correctly have started before the exported study period.

The visible lag estimate is:

$$
\widehat L_b=2.0+0.9\widehat{form}_b+0.002U_b^{planned}
$$

Net exposed units are:

$$
U_b^{new}=U_b^{planned}R_b^{true}(1-r_{ret,z_b})
$$

### Visibility

True realization, true lag, fill duration, and exposed units are latent.
Construction start and estimated lag are visible. `U_new` remains available to
the generator and validation but is an offset unavailable to a real predictor
unless an exposure estimate is explicitly introduced later.

### Validation gate

- Realization and fill duration are in their configured ranges.
- Construction start precedes or equals first occupancy.
- Estimated lag is positive.
- Exposed units are non-negative and follow the zoning exposure identity.
- Buildings with $r_{ret}=1$ would have zero exposure; defaults stay below one.

## 8. Stage 5: Population Panel

### Requires

- Validated areas and exposed building occupancy events.

### Produces

One row per area-year with true and observed household size, SES, median age,
and state share, plus stock, population, and cumulative occupied units.

### Logic

Aggregate occupancy events by `(stat_area_id, t_occupancy)` using $U_b^{new}$.
For area $a$ and year $t$, define cumulative exposed units:

$$
G_a(t)=\sum_{b:a(b)=a}U_b^{new}\mathbb{1}\{t_b^{occ}\le t\}
$$

For each building event, let
$frac_b=U_b^{new}/stock_a(0)$. Cumulative jumps through year $t$ are:

$$
J_{hh,a}(t)=\sum_b-0.55frac_b\mathbb{1}\{t_b^{occ}\le t\}
$$

$$
J_{ses,a}(t)=\sum_b2.20frac_b\mathbb{1}\{t_b^{occ}\le t\}
$$

$$
J_{age,a}(t)=\sum_b-9.00frac_b\mathbb{1}\{t_b^{occ}\le t\}
$$

With $s=t-2000$:

$$hh_a^{true}(t)=\bar H_a-0.011s+J_{hh,a}(t)$$

$$ses_a^{true}(t)=SES_a+gentr_as+J_{ses,a}(t)$$

$$medage_a^{true}(t)=medage_a+0.05s+J_{age,a}(t)$$

$$state_a^{true}(t)=state_a-0.002s$$

$$stock_a(t)=stock_a(0)+G_a(t)$$

$$pop_a(t)=pop_a(0)+2.40G_a(t)$$

Clip only variables whose domain requires it, especially state share to
$[0,1]$. Do not clip SES, which is a continuous standardized index.

### Optional construction-aware observation model

When disabled, set every observed value equal to its true value and mark the
run metadata accordingly.

When enabled, internally generate 1995-2022. Treat 1995 and 2008 as pre-annual
anchors and 2017 onward as observed annually. For a missing year $t$ between
anchors $c_0$ and $c_1$:

$$
\widehat x_a(t)=x_a(c_0)+[x_a(c_1)-x_a(c_0)]
\left[
(1-\theta)\frac{t-c_0}{c_1-c_0}
+\theta\frac{G_a(t)-G_a(c_0)}{G_a(c_1)-G_a(c_0)}
\right]
$$

with $\theta=0.50$. If the construction denominator is zero, fall back to the
linear component for that interval. Export only 2000-2022.

### Visibility

`*_true` values are latent. `*_obs` values are visible population features.
Stock, population, and cumulative units are denominators and diagnostics, not
automatic model predictors.

### Validation gate

- Exactly one row per area-year over the exported range.
- Stock and cumulative units are non-decreasing within area.
- Occupancy events affect only their own area from occupancy year onward.
- Observed values equal truth at anchors and for all years from 2017 onward.
- Disabled interpolation gives exact observed/true equality everywhere.
- Enabled interpolation produces finite values and uses the linear fallback
  when no construction occurs in an interval.

## 9. Stage 6: Building-Time Environment

### Requires

- Validated buildings, exposure, and population panel.

### Produces

One row per building with contemporaneous stock, intensity, SPILL, daycare,
school status, and load.

### Logic

Lookup stock in the building's area at its own $t^b_0$. For starts before the
exported panel, use the internally generated value for that year or baseline
stock if interpolation is disabled.

$$
intensity_b=1000\frac{U_b^{new}}{stock_{a(b)}(t^b_0)}
$$

Define distance in kilometres. SPILL is:

$$
SPILL_b=\frac{1000}{stock_{a(b)}(t^b_0)}
\sum_{b'\ne b}U_{b'}^{new}
\mathbb{1}\{d(b,b')\le1\}
\mathbb{1}\{t^b_0-5\le t_{b'}^{occ}<t^b_0\}
$$

Let $s_0=t^b_0-2000$:

$$
daycare_b\sim\operatorname{Poisson}
\left(0.9+0.0016stock_{a(b)}(t^b_0)+0.05s_0\right)
$$

The Poisson rate must be floored at zero for starts early enough to make the
linear time term negative under alternative configurations.

$$
p_b^{school}=\operatorname{sigmoid}
\left(-0.4+0.0004stock_{a(b)}(t^b_0)
-1.3\mathbb{1}\{z_b=new\_hood\}\right)
$$

For $u\sim U(0,1)$:

$$
status_b=\begin{cases}
existing, & u<p_b^{school}\\
planned, & p_b^{school}\le u<\min(p_b^{school}+0.25,1)\\
none, & \text{otherwise}
\end{cases}
$$

$$
util_b^{true}=\operatorname{clip}
\left(\mathcal{N}(0.92,0.16^2),0.45,1.6\right)
$$

$$
load_b=util_b^{true}\mathbb{1}\{status_b=existing\}
$$

### Visibility

Raw utilization is latent. Pre-multiplied load, status, daycare, intensity,
SPILL, construction start, and contemporaneous stock are visible or supporting
denominators according to the explicit schema.

### Validation gate

- Exactly one environment row per building.
- Every stock lookup uses building area and building construction start.
- Intensity and SPILL are non-negative and finite.
- A building never contributes to its own SPILL.
- SPILL contributors satisfy both distance and time-window predicates.
- Load is zero unless status is `existing`.
- School status is one of the three configured categories.

## 10. Shared Model Transformations

Before stages 7 and 8, join visible building, environment, and population
features by building ID and `(area, t_start)`. Record standardization means and
standard deviations in run metadata.

For every continuous feature $v$:

$$z(v)=\frac{v-\bar v}{s_v}$$

If $s_v=0$, fail with the feature name rather than substituting zero silently.

Fixed transforms:

$$g_{daycare}(m)=1-e^{-m/3}$$

$$g_{SES}(z)=0.05z-0.07z^2$$

The SES expression is the complete stage-1 SES contribution, not a transformed
feature that receives another coefficient. Daycare is different:
$g_{daycare}(m)$ is first computed as a feature and then multiplied by its
stage-specific coefficient, +0.06 in stage 1 and +0.15 in stage 2.

No standardized columns are written back to source tables.

## 11. Stage 7: Total Children

### Requires

- Validated model-visible building, population, and environment features.
- Latent exposed units for the generating offset.

### Produces

Expected count, dispersion, latent rate, project/area effects, and integer total
children ages 3-18.

### Logic

Default main-effect coefficients:

| Term | Coefficient |
|---|---:|
| Intercept | -0.79 |
| 3-room share | +0.14 |
| 4-room share | +0.21 |
| 5+-room share | +0.26 |
| Mix contrast | +0.04 |
| High-rise proxy | -0.13 |
| Intensity | +0.05 |
| New neighbourhood | +0.10 |
| TAMA 38/2 | -0.05 |
| TAMA 38/1 | -0.25 |
| Saturated daycare | +0.06 |
| Existing school | +0.10 |
| Planned school | +0.04 |
| Load | -0.07 |
| SPILL | +0.05 |
| Household size | +0.30 |
| SES linear | +0.05 |
| SES quadratic | -0.07 |
| Median age | -0.05 |
| State share | -0.20 |

Clearance and no school are reference categories. The 2-room share is the
omitted room baseline.

Interactions:

$$I_1=0.09z_{5+}z_{hh}$$

$$I_2=0.07\widehat{form}\,z_{5+}$$

$$I_3=-0.06z_{SPILL}z_{load}$$

$$I_4=-0.05z_{intensity}z_{hh}$$

Draw one random effect per project and area:

$$u_p\sim\mathcal{N}(0,0.10^2),\qquad w_a\sim\mathcal{N}(0,0.13^2)$$

Construct the linear predictor using standardized continuous main effects:

$$
\log\mu_b=\log U_b^{new}+\beta_0
+\mathbf{x}_b^T\boldsymbol\beta
+I_{1b}+I_{2b}+I_{3b}+I_{4b}
+u_{p(b)}+w_{a(b)}
$$

Dispersion rises with exposed units:

$$\phi_b=\min(2+0.06U_b^{new},25)$$

Use the Gamma-Poisson representation of NB2:

$$
\lambda_b\sim\operatorname{Gamma}
\left(shape=\phi_b,scale=\frac{\mu_b}{\phi_b}\right)
$$

$$N_b\sim\operatorname{Poisson}(\lambda_b)$$

This is the NB2 parameterization with

$$
\mathbb{E}[N_b]=\mu_b,
\qquad
\operatorname{Var}(N_b)=\mu_b+\frac{\mu_b^2}{\phi_b}.
$$

Thus larger $\phi_b$ means less overdispersion and a count closer to its mean.

### Visibility

`N_total` is an output. Expected mean, dispersion, rate, and random effects are
latent diagnostics. Preserve the model matrix separately for coefficient
recovery, but do not merge standardized columns into the building table.

### Validation gate

- Means, dispersion, and rates are finite and positive for positive exposure.
- Total counts are non-negative integers.
- Buildings sharing a project or area reuse the same corresponding effect.
- No period coefficient is present.
- Canonical full-run children per exposed unit is in 0.45-0.50.

## 12. Stage 8: Cohort Composition

### Requires

- Validated total counts and visible composition predictors.
- Latent fill duration for concentration noise.

### Produces

Expected cohort probabilities, youth index, concentration, realized cohort
probabilities, and integer counts.

### Logic

Youth-index coefficients:

| Term | Coefficient |
|---|---:|
| 3-room share | +0.38 |
| 4-room share | +0.10 |
| 5+-room share | -0.28 |
| New neighbourhood | +0.35 |
| Saturated daycare | +0.15 |
| Median age | -0.28 |
| Household size | -0.20 |
| SES | -0.10 |
| State share | -0.12 |
| Mix contrast | -0.22 |

$$\theta_b=\mathbf{z}_b^T\boldsymbol\delta$$

With target mean composition
$\mathbf{p}_0=(0.42,0.36,0.22)$:

$$
\eta_{kg,b}=\gamma_{kg}+1.0\theta_b
+0.18\mathbb{1}\{status_b=none\}
$$

$$\eta_{primary,b}=0$$

$$
\eta_{secondary,b}=\gamma_{secondary}-1.2\theta_b+0.45z_{hh,b}
$$

The expected cohort probabilities used below are the row-wise softmax:

$$
\mathbf p_b=\operatorname{softmax}
\left(\eta_{kg,b},\eta_{primary,b},\eta_{secondary,b}\right).
$$

Calibrate intercepts on the complete predictor matrix. Initialize:

$$
\gamma_{kg}^{(0)}=\log\frac{0.42}{0.36},
\qquad
\gamma_{secondary}^{(0)}=\log\frac{0.22}{0.36}
$$

For 60 iterations, compute row-wise softmax probabilities and their column
means $\bar p_k$, then update:

$$
\gamma_k\leftarrow\gamma_k+0.7
\left[
\log\frac{p_{0k}}{p_{0,primary}}
-\log\frac{\bar p_k}{\bar p_{primary}}
\right]
$$

Recompute probabilities after the final update.

Concentration is:

$$
\log\tau_b=2.5+0.5z_{HHI,b}
+0.3\log\left(\frac{U_b^{new}}{50}\right)
-0.4z_{D^{fill},b}
$$

$$\tau_b=\operatorname{clip}(e^{\log\tau_b},4,80)$$

Draw realized probabilities and counts:

$$
\mathbf{q}_b\sim\operatorname{Dirichlet}(\tau_b\mathbf{p}_b)
$$

$$
(n_{kg},n_{primary},n_{secondary})_b
\sim\operatorname{Multinomial}(N_b,\mathbf{q}_b)
$$

If $N_b=0$, counts are all zero while $\mathbf{p}_b$ and $\tau_b$ remain
defined; no multinomial draw is needed.

### Visibility

Counts, expected probabilities, youth index, and concentration belong in
results. Realized Dirichlet probabilities may be retained as latent diagnostics.

### Validation gate

- Every expected and realized probability row sums to one.
- Cohort counts are non-negative integers and sum exactly to `N_total`.
- Concentration lies in $[4,80]$.
- Mean expected composition is within 0.02 of each target component.
- Increasing a test youth index raises kindergarten and lowers secondary when
  non-monotone terms are held fixed.

## 13. Stage 9: Output Decomposition

### Requires

- Validated cohort counts, zoning, and standardized mix contrast.

### Produces

Returning, intra-city, outside-city, budget, and registered outputs by cohort.

### Logic

$$
\rho_b=\operatorname{clip}
\left(0.60+0.10z_{\Delta k,b},0.35,0.85\right)
$$

For each cohort $k$:

$$n_{returning,bk}=n_{bk}r_{ret,z_b}$$

$$
n_{intra,bk}=n_{bk}(1-r_{ret,z_b})\rho_b
$$

$$
n_{outside,bk}=n_{bk}(1-r_{ret,z_b})(1-\rho_b)
$$

$$n_{net,bk}=n_{bk}s_{net,z_b}$$

With leakage $\boldsymbol\ell=(0.02,0.05,0.15)$:

$$n_{registered,bk}=n_{bk}(1-\ell_k)$$

Decomposition values are expected fractional allocations of integer resident
counts. They should remain numeric, not be independently rounded, because
rounding each component would break accounting identities.

### Visibility

All decomposition outputs and relocation share belong in results. They are
derived outputs, not predictors of the generating stages.

### Validation gate

- Relocation share lies in $[0.35,0.85]$ and varies at canonical scale.
- Returning plus intra-city plus outside-city equals the cohort count.
- Registered counts do not exceed resident counts.
- Leakage is largest for secondary and smallest for kindergarten.
- No independently rounded decomposition columns are exported.

## 14. Final Table Contracts

### Areas

Keys, district, centroid, baseline stock and population, demographic baselines,
gentrification, state share, built character, room priors, and area mean rooms.

### Projects

Project key, area key, zoning, first occupancy, centroid, target units,
building count, returning share, net share, and generated planned-unit total.

### Buildings

Keys, coordinates, start and occupancy years, planned units, estimated lag,
area-category shares, visible room shares, HHI, form proxy, mix contrast, and
zoning. Latent truth remains in clearly marked columns in the same table.

### Population

Area, district, year, observed and true demographic values, stock, population,
and cumulative occupied units.

### Environment

Building key, construction start, contemporaneous stock, intensity, SPILL,
daycare, school status, load, and latent raw utilization.

### Results

Building key, total and cohort counts, expected cohort probabilities, youth
index, concentration, relocation share, decompositions, and measurement year.

## 15. Canonical Calibration Checks

Run these on the full default configuration and canonical seed. They are
statistical acceptance checks, not exact invariants for tiny test samples.

| Check | Expected target |
|---|---|
| Children per exposed unit | 0.45-0.50 |
| Expected composition | Within 0.02 of $(0.42,0.36,0.22)$ |
| Project planned totals | Every project in 100-2,500 |
| Estimated vs true 4-room share | Correlation near 0.75 |
| Estimated vs true floors | Correlation near 0.93 |
| Form classification | Accuracy near 94% |
| Area form persistence | Clearly positive |
| Area m2/room vs SES | Positive gradient |
| Area m2/room vs household size | Negative gradient |
| Buildings per project | Median above one |
| Single-building projects | TAMA projects only |
| Mix contrast | Centered near zero with material spread |
| Intra-city share | Variable and bounded |
| Interpolation error at observed years | Zero |
| Interpolation error vs construction | Clearly non-zero correlation |
| Stage-1 coefficient recovery | About 90% within two standard errors |

Calibration tolerances belong in validation configuration so changes are
reviewable and sensitivity runs can use different expectations.

## 16. Assumptions Register

| Assumption | Default | Status |
|---|---:|---|
| City population | 468,000 | Published calibration anchor |
| State-education city mean | 0.70 | Published calibration anchor |
| Baseline children per exposed unit | $e^{-0.79}$ | Calibrated |
| Age-three composition | $(0.42,0.36,0.22)$ | Calibrated |
| Standard floor plate | 450 m2 | Calibrated |
| Measurement after occupancy | 3 years | Structural |
| Building as unit of observation | Fixed | Structural |
| Returning residents reduce exposure | Fixed | Structural |
| Single youth composition index | Fixed | Structural |
| Population inputs synthetic | Fixed for current project | Constraint |
| Ages 0-2 generating process | Undefined | Unresolved extension |
| Institution opening/closing panel | Not in first version | Planned extension |

Every configuration field added later must be classified in this register or
an equivalent machine-readable assumptions catalogue.