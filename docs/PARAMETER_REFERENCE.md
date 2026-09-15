# Parameter and Logic Reference

> **Status: advanced target reference.** These parameters describe the full
> future model. Current simplified parameters are defined inline in
> [SIMPLIFIED_MODEL_PLAN.md](SIMPLIFIED_MODEL_PLAN.md).

## 1. Purpose

This document explains every default parameter in the canonical synthetic data
generation plan. For each component it states:

- what the parameter controls;
- its units or scale;
- why it exists in the generating process;
- what generally happens when its value increases.

The formulas and execution order remain in
[DATA_GENERATION_PLAN.md](DATA_GENERATION_PLAN.md). This reference explains
their meaning rather than repeating the pipeline contract.

All values other than the few published city-level anchors are synthetic
calibration choices. They encode plausible structure, not empirical estimates.

### 1.1 Domain terms and acronyms

| Term | Meaning |
|---|---|
| CBS | Israel Central Bureau of Statistics, the eventual source for real statistical-area data. |
| SES | Continuous standardized socio-economic index, measured in national standard deviations. It is not the CBS 1-10 cluster. |
| NB2 | Negative-binomial type 2 count model with variance $\mu+\mu^2/\phi$. |
| HHI | Herfindahl-Hirschman concentration index, here the sum of squared floor-area-category shares. |
| SPILL | Recent nearby occupied construction, normalized by the focal area's contemporaneous stock. It is a named feature rather than a separate fitted spatial model. |
| TAMA 38/1 | Reinforcement and extension of an existing building; usually one building with many returning residents. |
| TAMA 38/2 | Demolition and reconstruction of an existing building; usually one or two buildings. |
| Clearance/rebuild | Block-scale demolition and replacement, used as the reference zoning category in stage 1. |
| New neighborhood | Multi-building construction on newly developed or comprehensively redeveloped land. |

## 2. How to Interpret Parameters

### 2.1 Indices and mappings

| Symbol | Definition |
|---|---|
| $d$ | District index. Districts create shared area-level context. |
| $a$ | Statistical-area index. An area contains projects and buildings. |
| $p$ | Project index. A project contains one or more phased buildings. |
| $b$ | Building index, the unit of observation and prediction. |
| $t$ | Calendar year. |
| $k$ | Room category in building stages or child cohort in result stages. |
| $j$ | Floor-area category. |
| $a(b)$ | Statistical area containing building $b$. |
| $p(b)$ | Project containing building $b$. |
| $z_b$ | Zoning category of building $b$, inherited from its project. |

### 2.2 Marks on variables

| Mark | Meaning |
|---|---|
| $\bar v$ | Mean value, such as mean household size or apartment area. |
| $\widehat v$ | Observable estimate or proxy rather than generating truth. |
| $v^{true}$ | Latent value known to the simulator but unavailable to the model. |
| $z_v$ | Within-run standardized value $(v-\bar v)/s_v$. |
| $\mathbb{1}\{C\}$ | Indicator equal to 1 when condition $C$ is true. |
| $\tilde v$ | Unnormalized positive weight that must be divided by its sum. |

### 2.3 Statistical roles

Not every number is the same kind of assumption:

| Role | Meaning |
|---|---|
| Level | Sets a typical value, such as 2.55 people per household. |
| Spread | Controls heterogeneity around a level. |
| Trend | Changes a value per year. |
| Slope | Changes one outcome as another variable changes. |
| Threshold | Converts a continuous value into a category. |
| Concentration | Controls how closely a random composition follows its mean. |
| Bound | Prevents impossible or extreme simulated values. |
| Calibration target | A behavior checked after generation, not a direct draw. |

Positive regression coefficients increase a log mean, logit, or latent index;
negative coefficients reduce it. For a standardized predictor, a coefficient
is the change produced by a one-standard-deviation increase. A coefficient
$c$ on a log mean multiplies the expected count by $e^c$.

## 3. Global Scale and Time

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Number of districts | 9 | districts | Hierarchical groups sharing SES and gentrification context. More districts create more independent regional regimes. |
| Number of areas | 180 | areas | Statistical areas forming the population panel and stock denominator. More areas make local effects finer and projects sparser per area. |
| Number of projects | 320 | projects | Residential developments generated over the study period. More projects increase buildings, occupancy events, and calibration precision. |
| Export start | 2000 | year | First year shown to consumers. Earlier internal years may exist for interpolation and construction starts. |
| Export end | 2022 | year | Last occupancy and population year in the planned dataset. |
| Internal anchor start | 1995 | year | Earlier census-like anchor used only when observation interpolation is enabled. |
| Annual-observation start | 2017 | year | From this year onward population values are treated as observed rather than reconstructed. |
| Measurement lag | 3 | years after occupancy | Fixes when children are counted. It places measurement after typical filling and avoids modeling an occupancy curve. |
| Cohort widths | $(4,6,6)$ | birth years | Kindergarten, primary, and secondary span unequal numbers of ages. Divide cohort share by width for per-birth-year comparisons. |

The seed is not a demographic parameter. It selects one reproducible realization
of all configured distributions. Changing only the seed should change sampled
rows but not the simulator's long-run behavior.

## 4. Statistical Areas

### 4.1 District hierarchy and SES

$$SES_d\sim\mathcal N(0,1),\qquad SES_a=SES_{d(a)}+\epsilon_a$$

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| District SES mean | 0 | national SES standard deviations | Centers district effects on the national index mean. |
| District SES SD | 1.00 | SES SD | Produces meaningful differences between districts. Increasing it makes broad geographic inequality stronger. |
| Area SES residual SD | 0.55 | SES SD | Local deviation around the district value. Increasing it makes areas within the same district less alike. |
| District assignment probability | $1/9$ per district | probability | Areas are assigned independently and uniformly. This avoids tying district size to SES or stock before any data supports such a relationship. |

`SES` is the continuous standardized socio-economic index, not the 1-10 CBS
cluster. The district component creates partial pooling structure: two areas in
one district share a baseline even when their local residuals differ.

### 4.2 Gentrification

$$gentr_d\sim U(0,0.045),\qquad gentr_a=gentr_{d(a)}+\zeta_a$$

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| District gentrification lower bound | 0 | SES SD/year | Prevents a systematic district-level SES decline in the default process. |
| District gentrification upper bound | 0.045 | SES SD/year | Maximum shared annual SES increase. Larger values create stronger long-term neighborhood change. |
| Area gentrification residual SD | 0.008 | SES SD/year | Allows nearby areas to gentrify at slightly different rates. |

Gentrification is a slope, not an SES level. An area with slope 0.03 gains
about 0.66 SES standard deviations over 22 years before occupancy jumps.

### 4.3 Household size and median age

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Household-size mean | 2.55 | persons/household | Area baseline around which household size varies. It is above the citywide anchor because weighting and later time decline affect the realized panel. |
| Household-size SD | 0.30 | persons/household | Cross-area heterogeneity. Larger spread strengthens room-prior and child-count contrasts. |
| Household-size lower bound | 1.8 | persons/household | Avoids implausibly small area averages. |
| Household-size upper bound | 4.2 | persons/household | Avoids extreme area averages. |
| Median-age mean | 36.5 | years of age | Typical baseline age of area residents. |
| Median-age SD | 4.0 | years | Cross-area demographic heterogeneity. |
| Median-age lower bound | 26 | years | Prevents unrealistically young area medians. |
| Median-age upper bound | 50 | years | Prevents unrealistically old area medians. |

Household size is both a demographic predictor and part of room-proxy
correction. Median age represents lifecycle stage: older areas generally
generate fewer incoming children in stage 1 and an older cohort mix in stage 2.

### 4.4 Baseline stock, population, and city geometry

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Baseline stock range | 900-6,500 | dwelling units/area | Existing housing denominator before generated construction. Wider ranges create stronger intensity differences for equally sized buildings. |
| Initial population draw | 2,500-9,000 | persons/area | Unscaled area populations before exact city calibration. |
| City population target | 468,000 | persons | Published calibration anchor. Area draws are balanced to this exact total. |
| City x extent | 9 | km | Synthetic east-west coordinate span. |
| City y extent | 12 | km | Synthetic north-south coordinate span. |
| Built-character mean | 0 | standardized latent units | Centers persistent high-rise propensity. |
| Built-character SD | 1 | standardized latent units | Creates low-rise and high-rise areas that persist through time. Larger spread makes area form more segregated. |

Coordinates are synthetic and carry no real geography. Their purpose is to
create realistic distance relationships for SPILL.

### 4.5 State-education share mixture

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Main-component probability | 0.85 | Most areas belong to a high state-school participation regime. Increasing it reduces the number of low-share enclaves. |
| Main Beta shapes | $(8.0,2.5)$ | Mean $8/(8+2.5)\approx0.762$. The large shape sum keeps most main-regime areas near that level. |
| Low-component probability | 0.15 | Minority of areas with spatially concentrated non-state schooling. |
| Low Beta shapes | $(3.0,5.5)$ | Mean $3/(3+5.5)\approx0.353$, creating a distinct low-share mode rather than symmetric noise. |
| Lower clip | 0.08 | Avoids an area with essentially no state-school participation. |
| Upper clip | 0.99 | Avoids a mathematically perfect share. |
| City mean target | 0.70 | Published calibration anchor applied after the mixture draw. |

The mixture matters because the same city mean could otherwise be generated by
a narrow, uninformative distribution. The minority component creates signal
that is clustered by area.

### 4.6 Area room prior

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Room categories $\mathbf k$ | $(2,3,4,5,6)$ | Supported true room counts. One-room units are outside the planned family-housing profile. |
| Baseline prior $\boldsymbol\pi^0$ | $(0.10,0.28,0.34,0.20,0.08)$ | Citywide room mix at neutral SES and household size. Four-room units are most common. |
| SES prior tilt | 0.25 | One SD higher SES shifts prior weight toward larger apartments. It reflects ability to buy more rooms. |
| Household-size prior tilt | 0.45 | One SD larger household size shifts the prior more strongly toward larger apartments. It is larger than the SES tilt because room demand is more directly tied to household size. |
| Center room | 4 | rooms | Makes neutral tilting leave the 4-room weight unchanged and moves smaller/larger rooms in opposite directions. |
| Tilt divisor | 2 | rooms | Controls how quickly tilt grows away from four rooms. A smaller divisor would exaggerate differences at 2 and 6 rooms. |

The area prior answers: before seeing a building's floor-area categories, what
room mix is plausible in this area? It is also used by Bayesian inversion, so
SES and household size correct the floor-area proxy rather than acting only as
direct predictors.

## 5. Projects

### 5.1 Area assignment and size

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Area assignment weight | baseline stock | relative weight | Areas with more existing housing receive more projects. Uniform weighting would overrepresent tiny areas. |
| Log target median | 450 | dwelling units/project | Because log-location is $\log 450$, the untruncated median target is 450 units. |
| Log target SD | 0.75 | log units | Controls right-skew. Larger values produce more very small and very large projects. |
| Target lower bound | 100 | units/project | Project-scope minimum. |
| Target upper bound | 2,500 | units/project | Project-scope maximum. |
| First-occupancy range | 2000-2022 | year | Distributes project beginnings across the study period. |

`target_units` drives project structure; it is not guaranteed to equal the sum
of generated building units. Both are retained so the difference is visible.

### 5.2 Zoning transition

$$P(new\_hood)=\operatorname{sigmoid}(1.20-0.13s_p)$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| New-neighborhood intercept | 1.20 | Log-odds in year 2000. It gives probability about 0.77 at the start. |
| New-neighborhood yearly slope | -0.13 | Annual log-odds change. Negative value shifts construction toward renewal; probability falls to about 0.16 by 2022. |
| Renewal split | $(0.35,0.40,0.25)$ | Conditional probabilities for clearance, TAMA 38/2, and TAMA 38/1 after a project is classified as renewal. |

The intercept determines the initial construction regime. The slope determines
how quickly that regime changes, without adding calendar year directly to the
child-count model.

### 5.3 Building count and zoning exposure

| Zoning | Expected units/building $m_z$ | Minimum buildings $n_z^{min}$ | Returning share $r_{ret}$ | Net-budget share $s_{net}$ |
|---|---:|---:|---:|---:|
| New neighborhood | 70 | 4 | 0.00 | 1.00 |
| Clearance/rebuild | 85 | 2 | 0.25 | 0.65 |
| TAMA 38/2 | 60 | 1 | 0.55 | 0.45 |
| TAMA 38/1 | 40 | 1 | 0.85 | 0.15 |

- $m_z$ converts target project size into a plausible building count. Larger
  $m_z$ means fewer buildings for the same target.
- $n_z^{min}$ encodes physical planning regimes. A new neighborhood cannot be
  represented by one building, while TAMA may be.
- $r_{ret}$ is the share associated with residents already tied to the site.
  It reduces net exposure and prevents double counting against the existing
  population model.
- $s_{net}$ is the separate policy/budget increment used only in output
  decomposition. It need not equal $1-r_{ret}$ because it answers a different
  accounting question.

The maximum of 14 buildings prevents the lognormal tail from creating projects
with implausibly many separately phased buildings.

### 5.4 Project location

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Project centroid SD | 0.25 | km per coordinate | Places projects near their area's synthetic centroid. Larger values make area membership less spatially compact and increase cross-area proximity. |

The covariance $0.25^2I$ is isotropic: x and y offsets are independent with the
same standard deviation. $I$ is the two-dimensional identity matrix.

## 6. Buildings and Observable Proxies

### 6.1 Occupancy phasing

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Main phase offset | integer 0 to $n_p-1$ | years | Spreads project buildings over a horizon that grows with project size. |
| Extra phase offset | integer 0 or 1 | years | Prevents perfectly regular one-building-per-year phasing. |
| Occupancy bounds | 2000-2022 | year | Keeps exported buildings in the study period; clipping may bunch late projects at 2022 and should be monitored. |

Phasing is signal: later buildings in one project see updated stock, SPILL, and
institutions.

### 6.2 High-rise probability and floors

$$P(high_b=1)=\operatorname{sigmoid}(-1.50+0.11s_b+1.2c_{a(b)})$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Form intercept | -1.50 | Baseline high-rise log-odds in 2000 for an average area, giving probability about 0.18. |
| Form yearly slope | +0.11 | Annual rise in high-rise log-odds. Positive value captures the citywide tower trend. |
| Built-character weight | +1.20 | Effect of one SD higher persistent area character. It is large enough to keep form spatially persistent. |
| High-rise floor base | 12 | floors | Minimum component before Poisson variation for provisionally high-rise buildings. |
| High-rise Poisson mean | 10 | floors | Adds an average of ten floors, producing a typical value around 22. |
| Low-rise floor base | 4 | floors | Base for walk-up and mid-rise buildings. |
| Low-rise Poisson mean | 3 | floors | Produces a typical low-rise value around seven floors. |
| Floor lower/upper bounds | 3/45 | floors | Prevents physically implausible outcomes. |
| Final form threshold | $F>10$ | floors | Defines high-rise from generated floors. Floors remain primitive; form is not an independent final draw. |

The provisional high-rise draw selects a floor distribution. The final form is
recomputed from floors so every stored label is consistent with the threshold.

### 6.3 Building location

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Building centroid SD | 0.06 | km per coordinate | Keeps sibling buildings close to the project centroid. Larger values weaken within-project spatial clustering. |

### 6.4 Building room mix

$$tilt_b=-0.35high_b-0.020s_b$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| High-rise room tilt | -0.35 | Shifts a high-rise toward smaller room counts, reflecting compact tower layouts. More negative values create a stronger form/mix relationship. |
| Annual room tilt | -0.020 | Each later occupancy year shifts room mix slightly smaller. It represents apartment shrinkage over time. |
| Room Dirichlet concentration | 45 | Controls building-to-building variation around the tilted area prior. Higher values make buildings closely resemble their expected mix; lower values produce specialized buildings. |

The tilt acts symmetrically around four rooms. A negative value raises relative
weight on 2-3 rooms and lowers relative weight on 5-6 rooms.

### 6.5 Floor area given rooms

$$
\mu_{bk}=8+22k+4z_{SES,a}-3high_b-0.15s_b-2.5z_{\bar H,a}
$$

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Area intercept | 8 | m2 | Base before room count. Combined with the room slope it yields neutral 3-room area near 74 m2. |
| Per-room slope | 22 | m2/room | Average additional area per room. Increasing it separates room categories more clearly and improves proxy inversion. |
| SES area effect | +4 | m2 per SES SD | Wealthier areas build more generous apartments for the same room count. |
| High-rise area effect | -3 | m2 | Towers use slightly more compact layouts at fixed room count. |
| Annual area trend | -0.15 | m2/year | Later apartments shrink at fixed room count. Over 22 years this removes about 3.3 m2. |
| Household-size area effect | -2.5 | m2 per household-size SD | Large-household areas fit the same room count into a smaller envelope. This creates systematic proxy bias in the opposite direction from SES. |
| Residual area SD | 9 | m2 | Overlap between room categories. Larger values make area-to-room inversion less certain. |
| Area lower/upper clips | 22/260 | m2 | Physical bounds on apartment area. |

The SES and household-size effects are deliberately load-bearing. Without them,
the same area category would imply nearly the same room mix everywhere, turning
proxy error into harmless random noise.

### 6.6 Floor-area categories and HHI

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Category 1 | $[0,50)$ | m2 | Very small apartments. |
| Category 2 | $[50,85)$ | m2 | Small-to-medium apartments. |
| Category 3 | $[85,100)$ | m2 | Transitional band where 3-room and 4-room interpretations overlap strongly. |
| Category 4 | $[100,\infty)$ | m2 | Large apartments. |

$c_{bj}$ is the share of building $b$ in category $j$. The Herfindahl-Hirschman
index $HHI_b=\sum_jc_{bj}^2$ measures concentration of the area mix. With four
categories it ranges from 0.25 for an equal split to 1 for a single category.
HHI affects composition certainty, not expected child count.

### 6.7 Floor plate and planned units

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Standard plate | 450 | m2/floor | Typical residential floor area used both as the latent plate median and the predictor's fixed inversion assumption. |
| Plate log SD | 0.18 | log m2 | True plate variation. Larger values make estimated floors and form less accurate, preventing a perfect inverse. |
| Planned-unit lower/upper bounds | 8/400 | units/building | Keeps building size within plausible limits. |
| Project planned-unit bounds | 100/2,500 | units/project | Acceptance bounds after building generation. |
| Maximum resampling attempts | configurable, not yet fixed | attempts/project | Safety limit for projects whose generated building totals violate bounds. This must receive a reviewed default before implementation. |

The formula divides total residential floor area, floors times plate, by mean
apartment area. Plate variation is the main reason a visible fixed-plate
inversion cannot recover true floors exactly.

### 6.8 Bayesian room inversion

The inversion has no new fitted coefficient. It combines:

- the area room prior $\pi_{ak}$;
- the conditional area mean $\mu_{bk}$;
- residual SD 9 m2;
- observed category boundaries.

$\Phi((u-\mu)/9)-\Phi((l-\mu)/9)$ is the chance that a normal apartment area
falls inside category $[l,u)$. Multiplying this likelihood by the area prior and
normalizing over room counts is Bayes' rule. A deterministic inverse would hide
the intended uncertainty.

Mix contrast $\Delta k_b$ is estimated mean rooms in the building minus the
area's prior mean rooms. Positive contrast means the building offers relatively
large units for its local market, which is interpreted as a move-up signal.

## 7. Exposure

### 7.1 Realization rate

$$\nu_b=4+0.05U_b^{planned}$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Base precision | 4 | Minimum Beta concentration before size. Small buildings remain variable. |
| Precision per planned unit | 0.05 | Makes realization more stable as buildings grow. A 100-unit building has $\nu=9$; a 300-unit building has $\nu=19$. |
| Beta mean share | 0.60/0.40 | Shapes are $0.6\nu$ and $0.4\nu$, so the Beta mean is 0.6. |
| Realization offset | 0.7 | Lower endpoint of ordinary realization. |
| Realization scale | 0.5 | Converts Beta support $[0,1]$ to ordinary support $[0.7,1.2]$. Together with mean 0.6 this gives $E[R]=1.0$. |
| Outlier size threshold | 150 | units | Only smaller buildings can receive major upward revisions. |
| Outlier probability | 0.03 | probability | Rare exceptional realization event. Increasing it thickens the upper tail. |
| Outlier range | 1.5-2.0 | realized/planned ratio | Represents small projects that deliver far more units than initially recorded. |

Realization rate $R_b^{true}$ multiplies planned units. It is latent because
actual delivery is not known at construction start.

### 7.2 Lag and fill duration

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| True lag offset | 2 | years | Minimum construction-to-occupancy lag. |
| True lag scale | 2 | years | Maps Beta(2,2) to 2-4 years, centered at 3. |
| True lag Beta shapes | $(2,2)$ | shapes | Symmetric lag variation, with fewer values near endpoints. |
| Fill offset | 0.5 | years | Minimum first-occupancy-to-full duration. |
| Fill scale | 2 | years | Produces support 0.5-2.5 years. |
| Fill Beta shapes | $(1.5,3.5)$ | shapes | Right-skewed toward faster filling; mean fill is about 1.1 years. |
| Visible lag intercept | 2.0 | years | Predictor's baseline expected lag. |
| Visible high-rise lag | +0.9 | years | Towers are expected to take longer. |
| Visible units lag | +0.002 | years/planned unit | Larger buildings add expected duration; 100 units add 0.2 years. |

Fill duration does not change expected cohort probabilities. It lowers their
concentration because slowly filled buildings combine households arriving over
a wider time window.

### 7.3 Net exposure

$$U_b^{new}=U_b^{planned}R_b^{true}(1-r_{ret,z_b})$$

`U_new` is the equivalent number of newly exposed dwellings after actual
realization and returning-resident reduction. It may be fractional because it
is an exposure measure, not a literal observed apartment count.

## 8. Population Panel

### 8.1 Background trends

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Household-size trend | -0.011 | persons/household/year | Encodes gradual fertility and household-size decline. More negative values lower stage-1 child demand in later years through the population feature. |
| SES trend | $gentr_a$ | SES SD/year | Area-specific gentrification drawn in stage 1. |
| Median-age trend | +0.05 | age years/calendar year | Slowly ages the background population. |
| State-share trend | -0.002 | share/year | Gradual decline in state-education participation. Over 22 years the change is -0.044 before clipping. |

There is no separate calendar-period coefficient in stage 1. Time already
changes the predictors, so another period coefficient would double count the
same mechanism.

### 8.2 Occupancy jumps

For $frac_b=U_b^{new}/stock_a(0)$:

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Household-size jump | -0.55 | persons/household per stock fraction | New construction brings somewhat smaller households than the area baseline. A building equal to 10% of baseline stock changes the area mean by -0.055. |
| SES jump | +2.20 | SES SD per stock fraction | New construction brings more affluent households. A 10% stock event raises SES by 0.22. |
| Median-age jump | -9.00 | age years per stock fraction | New residents are younger. A 10% event lowers median age by 0.9 years. |
| Population per exposed unit | +2.40 | persons/unit | Adds residents to area population for every net exposed dwelling. |
| Stock per exposed unit | +1.00 | dwelling/unit | Adds exposure-equivalent units to the stock denominator. |

Jumps are cumulative and begin at occupancy, not construction start. Using
baseline stock in `frac` makes each event's demographic shock comparable to the
original area size; contemporaneous stock is still used later for intensity.

### 8.3 Observation interpolation

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Interpolation weight $\theta$ | 0.50 | Half of reconstructed progress follows elapsed time and half follows construction timing. |
| Linear component weight | $1-\theta=0.50$ | Represents background change unrelated to generated construction. |
| Construction component weight | $\theta=0.50$ | Places change near years when units were occupied. |
| Pre-annual anchors | 1995 and 2008 | Mimic sparse historical census measurements. |
| Annual observed period | 2017 onward | Mimics later annual area-level publication. |

$G_a(t)$ is cumulative exposed units through year $t$. If no units were added
between anchors, the construction fraction has a zero denominator, so the model
must use pure linear interpolation. The interpolation layer is optional because
the synthetic truth is already known.

## 9. Building-Time Environment

### 9.1 Intensity and SPILL

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Intensity multiplier | 1,000 | exposed units per 1,000 existing units | Makes ratios readable. It changes scale, not information. |
| SPILL radius | 1.0 | km | Defines the local construction market around a building. Larger radius includes more neighboring projects. |
| SPILL lookback | 5 | years | Includes recently occupied units that may still affect infrastructure and absorption. Longer windows retain older construction pressure. |
| SPILL multiplier | 1,000 | prior units per 1,000 existing units | Uses the same interpretable scale as intensity. |

Intensity measures the focal building relative to contemporaneous area stock.
SPILL measures other nearby units occupied before its construction start. The
strict upper bound excludes future occupancy and prevents self-contribution.

### 9.2 Daycare count

$$\lambda_{daycare}=0.9+0.0016stock_a(t_0)+0.05(t_0-2000)$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Daycare intercept | 0.9 | Expected count before stock and time contributions. |
| Daycare stock slope | 0.0016 | Additional expected daycares per existing dwelling; 1,000 units add 1.6 expected facilities. |
| Daycare yearly slope | 0.05 | Later years have more facilities; 20 years add one expected daycare. |
| Rate floor | 0 | Poisson rates cannot be negative under alternative year configurations. |

The Poisson distribution creates integer counts around this expected local
capacity. Stage 1 and stage 2 use a saturated transform because the first few
facilities matter more than additions to an already dense network.

### 9.3 School status

$$p^{school}=\operatorname{sigmoid}(-0.4+0.0004stock-1.3I_{new})$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| School intercept | -0.4 | Baseline log-odds of an existing school before stock and zoning terms. |
| School stock slope | +0.0004 | Existing-school probability rises with area size; 1,000 units add 0.4 log-odds. |
| New-neighborhood penalty | -1.3 | New neighborhoods are less likely to have an existing school when construction begins. |
| Planned-status probability width | 0.25 | After the existing-school interval, up to 25 percentage points are allocated to planned status. Remaining probability is no school. |

The planned interval is capped at 1. If existing probability is 0.9, planned
probability is 0.1 rather than 0.25.

### 9.4 School load

| Parameter | Default | Units | Definition and intuition |
|---|---:|---|---|
| Utilization mean | 0.92 | occupied/capacity ratio | Typical existing school is near capacity. |
| Utilization SD | 0.16 | ratio | Creates meaningful under- and over-capacity variation. |
| Utilization lower/upper clips | 0.45/1.60 | ratio | Prevents impossible extremes while allowing overcrowding above one. |

`load` equals utilization only for an existing school and zero otherwise. This
pre-multiplication prevents a meaningless utilization draw for planned or absent
schools from contaminating model coefficients.

## 10. Shared Model Transforms

### 10.1 Standardization

$$z(v)=\frac{v-\bar v}{s_v}$$

$\bar v$ and $s_v$ are the within-run mean and sample standard deviation at the
relevant grain. Area-prior standardization is fitted over areas. Outcome-model
standardization is fitted over buildings after joining their prediction-time
features. Standardization makes coefficients comparable and lets a coefficient
represent a one-SD change.

### 10.2 Daycare saturation

$$g_{daycare}(m)=1-e^{-m/3}$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Saturation scale | 3 daycares | At three facilities the transform reaches $1-e^{-1}\approx0.63$. Larger scales make saturation slower. |

The transform is 0 at no daycare, about 0.28 at one, 0.63 at three, and 0.86 at
six. It encodes diminishing returns.

### 10.3 SES quadratic

$$g_{SES}(z)=0.05z-0.07z^2$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| SES linear term | +0.05 | Slightly raises total-child log mean as SES moves above average near the center. |
| SES quadratic term | -0.07 | Bends the relationship downward away from its maximum, preventing a simple monotone SES effect. |

The maximum occurs at $z=0.05/(2\times0.07)\approx0.36$. This is technically an
inverted U, not “high fertility at both ends.” If the intended mechanism is high
at both SES extremes, the quadratic sign must be positive instead. This semantic
choice should be reviewed before implementation.

The expression $0.05z-0.07z^2$ is inserted directly into the stage-1 linear
predictor. It is not multiplied by a second generic SES coefficient. In
contrast, the daycare saturation function creates a transformed feature that is
then multiplied by +0.06 in stage 1 or +0.15 in stage 2.

## 11. Stage 1: Total Children

Stage 1 coefficients act on $\log\mu_b$. For a one-unit change in a binary
feature or a one-SD change in a standardized continuous feature, $e^\beta$ is
the multiplicative effect on expected total children, holding other terms fixed.

### 11.1 Main effects

| Parameter | Default | Approximate multiplier | Logic |
|---|---:|---:|---|
| Intercept $\beta_0$ | -0.79 | 0.454 | Baseline children ages 3-18 per net exposed unit for reference categories and average continuous predictors. It sets overall output scale. |
| 3-room share | +0.14 | 1.15 | More 3-room units support young/small families relative to the omitted 2-room baseline. |
| 4-room share | +0.21 | 1.23 | Family-sized units increase total child capacity more strongly. |
| 5+-room share | +0.26 | 1.30 | Largest apartments have the strongest direct total-child association. Composition may still shift their children older. |
| Mix contrast | +0.04 | 1.04 | A building larger-grained than its area holds slightly more children beyond its absolute mix. Kept small to avoid double counting room shares. |
| High-rise proxy | -0.13 | 0.88 | Towers attract fewer children after controlling for room mix, reflecting access, cost, and household selection. |
| Intensity | +0.05 | 1.05 | A larger intervention relative to local stock attracts somewhat more child-bearing households per exposed unit. |
| New neighborhood | +0.10 | 1.11 | New neighborhoods are relatively family-oriented compared with clearance, the reference zoning. |
| TAMA 38/2 | -0.05 | 0.95 | Demolition/rebuild attracts slightly fewer children per exposed unit than clearance. |
| TAMA 38/1 | -0.25 | 0.78 | Reinforcement projects have high returning shares and weaker new-family inflow. |
| Saturated daycare | +0.06 | 1.06 | Childcare availability modestly attracts families, with diminishing returns already handled by the transform. |
| Existing school | +0.10 | 1.11 | Existing education capacity makes the building more attractive to families than no school. |
| Planned school | +0.04 | 1.04 | A plan is helpful but weaker than an operating school. |
| Load | -0.07 | 0.93 | Higher school utilization or overcrowding deters families. |
| SPILL | +0.05 | 1.05 | Recent nearby construction signals an active, attractive residential market before crowding interaction is applied. |
| Household size | +0.30 | 1.35 | One SD larger local households strongly raises children per exposed unit. |
| SES linear | +0.05 | varies | Local positive slope near the center of the quadratic relation. |
| SES quadratic | -0.07 | varies | Downward curvature; interpretation depends on squared standardized SES. |
| Median age | -0.05 | 0.95 | Older local populations predict fewer children in incoming households. |
| State share | -0.20 | 0.82 | Higher state-school share is treated as a proxy for areas with somewhat lower total resident-child inflow after other demographics. Its sign is calibrated and should not be read causally. |

Room-share coefficients apply to shares from 0 to 1, not automatically to a
one-SD change unless the model matrix explicitly standardizes them. The planned
model standardizes continuous predictors, including room shares, so the table's
practical interpretation is one SD. The 2-room share remains omitted to avoid a
singular compositional design.

### 11.2 Interactions

| Interaction | Default | Logic |
|---|---:|---|
| $z_{5+}z_{hh}$ | +0.09 | Large apartments produce more children where household demand exists. A 5-room-heavy building in a small-household area is more likely a space upgrade than a family expansion. |
| $form\times z_{5+}$ | +0.07 | Offsets part of the high-rise penalty when a tower actually contains large family units. This prevents penalizing it through both form and mix for the same reason. |
| $z_{SPILL}z_{load}$ | -0.06 | Recent construction becomes deterrent when schools are already crowded. Either condition alone is less harmful. |
| $z_{intensity}z_{hh}$ | -0.05 | Existing household size becomes less informative when a project is large enough to replace the local population composition. |

An interaction coefficient applies to the product. For example, if both SPILL
and load are one SD above average, the third interaction changes log mean by
-0.06 in addition to their main effects.

### 11.3 Random effects and NB2 dispersion

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Project-effect SD | 0.10 log points | Shared unobserved developer, marketing, finish, and buyer-target variation. Larger values make sibling buildings more correlated. |
| Area-effect SD | 0.13 log points | Shared unmeasured neighborhood variation. It exceeds project SD because more area context remains unobserved. |
| Dispersion intercept | 2 | NB2 precision | Minimum concentration for a near-zero-exposure building. |
| Dispersion units slope | 0.06 | precision per exposed unit | Larger buildings become more predictable in relative terms. |
| Dispersion cap | 25 | precision | Prevents very large buildings from becoming nearly Poisson with negligible extra heterogeneity. |

NB2 variance is $\mu+\mu^2/\phi$. Increasing $\phi$ reduces overdispersion. The
Gamma shape is $\phi$ and scale is $\mu/\phi$, giving latent rate mean $\mu$.

## 12. Stage 2: Cohort Composition

### 12.1 Youth-index coefficients

Positive $\delta$ increases the youth index $\theta$, which raises kindergarten
logit by $+\theta$ and lowers secondary logit by $-1.2\theta$. Negative $\delta$
does the reverse.

| Parameter | Default | Logic |
|---|---:|---|
| 3-room share | +0.38 | Smaller family apartments are associated with younger households and first children. |
| 4-room share | +0.10 | Mildly younger composition while still supporting primary-age families. |
| 5+-room share | -0.28 | Large apartments attract established families with older children. |
| New neighborhood | +0.35 | New neighborhoods attract younger incoming households. |
| Saturated daycare | +0.15 | Childcare availability shifts composition toward kindergarten age. |
| Median age | -0.28 | Older surrounding populations predict older incoming family lifecycle. |
| Household size | -0.20 | Through the youth axis, larger households are treated as more established and older. A separate secondary term captures large families across ages. |
| SES | -0.10 | Higher SES mildly shifts expected composition older in this calibration. |
| State share | -0.12 | Higher state share mildly shifts composition older; this is a proxy assumption, not a causal claim. |
| Mix contrast | -0.22 | Move-up households trading for locally larger units tend to have older children. |

### 12.2 Cohort logits

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Kindergarten youth loading | +1.0 | One unit higher $\theta$ adds one kindergarten logit point. It fixes the scale of $\theta$. |
| Secondary youth loading | -1.2 | Secondary responds 20% more strongly in the opposite direction. |
| Primary logit | 0 | Reference category needed to identify the softmax. Adding the same constant to all logits would not change probabilities. |
| No-school kindergarten effect | +0.18 | Non-monotone exception: areas without a school may contain very young developments whose children have not reached primary age. |
| Secondary household-size effect | +0.45 | Large households can contain older children even when the youth index points younger. This restores a family-size mechanism outside the single youth axis. |

### 12.3 Target composition and intercept calibration

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Target $P_0$ | $(0.42,0.36,0.22)$ | Desired average kindergarten, primary, secondary composition for buildings measured three years after occupancy. It is younger than a stationary age-width profile. |
| Calibration iterations | 60 | Repeated intercept corrections. More iterations improve convergence but should not change a converged result. |
| Calibration learning rate | 0.7 | Fraction of each log-ratio error applied per iteration. Values near one converge faster but can oscillate; smaller values are steadier. |
| Composition tolerance | 0.02 | Maximum accepted absolute difference between each mean expected share and its target. |

$\gamma_{kg}$ and $\gamma_{secondary}$ are calibrated intercepts, not fixed
input parameters. Their initial values are log target ratios to primary. They
are adjusted because nonlinear softmax averaging otherwise moves the population
mean away from $P_0$.

### 12.4 Concentration $\tau$

$$\log\tau=2.5+0.5z_{HHI}+0.3\log(U^{new}/50)-0.4z_{fill}$$

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Concentration intercept | 2.5 | Baseline log concentration; $e^{2.5}\approx12.2$ before other terms. |
| HHI slope | +0.5 | Buildings with concentrated apartment types attract more homogeneous households, so realized composition stays closer to expectation. |
| Exposure slope | +0.3 | Larger exposed buildings average over more households and have a more stable composition. |
| Exposure reference | 50 | units | Makes the size term zero at 50 exposed units. Changing it shifts the effective intercept. |
| Fill-duration slope | -0.4 | Slowly filled buildings mix arrival cohorts and have less predictable composition. |
| Tau lower/upper bounds | 4/80 | Prevent compositions from becoming either nearly unconstrained or unrealistically deterministic. |

Higher $\tau$ reduces Dirichlet variation around expected probabilities. It
does not alter the mean $\mathbf p_b$.

## 13. Output Decomposition

### 13.1 Intra-city relocation

| Parameter | Default | Definition and intuition |
|---|---:|---|
| Base intra-city share | 0.60 | At average mix contrast, 60% of non-returning arrivals are assumed to come from elsewhere in the city. |
| Contrast slope | +0.10 | One SD higher mix contrast raises intra-city relocation by ten percentage points because move-up moves are often local. |
| Relocation lower/upper bounds | 0.35/0.85 | Prevent implausibly low or near-total intra-city assignment. |

The share applies only after returning residents are removed. It addresses
double-counting risk against the citywide cohort-progression model.

### 13.2 Registration leakage

| Cohort | Leakage | Registered share | Logic |
|---|---:|---:|---|
| Kindergarten | 0.02 | 0.98 | Very little loss from resident children to the registration target. |
| Primary | 0.05 | 0.95 | Some private or out-of-city enrollment. |
| Secondary | 0.15 | 0.85 | Larger private, boarding, and out-of-city leakage. |

Leakage converts resident-child counts to expected registered-pupil counts. It
does not change the resident target and should only be used when the downstream
target is registration.

### 13.3 Accounting outputs

- `returning` uses zoning-specific $r_{ret}$.
- `intra_city` and `outside_city` partition the non-returning share using
  $\rho_b$.
- `net_budget` uses zoning-specific $s_{net}$ and answers a separate planning
  question.
- `registered` applies cohort-specific leakage.

These are expected fractional allocations, even though resident cohort counts
are integers. Independent rounding would break identities.

## 14. Validation Targets

Validation targets describe desired aggregate behavior. They are not generation
coefficients and should not be tuned independently without understanding which
upstream assumptions produce them.

| Target | Default | What it tests |
|---|---:|---|
| Children per exposed unit | 0.45-0.50 | Overall stage-1 scale, mainly intercept and random/nonlinear effects. |
| Mean cohort tolerance | 0.02 each | Softmax intercept calibration. |
| Project total range | 100-2,500 | Project/building size coherence. |
| 4-room recovery correlation | about 0.75 | Difficulty of area-category room inversion. |
| Floor recovery correlation | about 0.93 | Information loss caused by plate variation. |
| Form accuracy | about 94% | Observable high-rise proxy quality. |
| Mix-contrast center | near zero | Buildings deviate around, rather than systematically above, area profiles. |
| Coefficient recovery | about 90% within two SE | Whether sample size and generating design recover configured stage-1 effects. |

“About” targets need explicit numeric tolerances in `ValidationConfig` before
implementation. They should initially be treated as calibration goals, then
frozen only after a canonical run shows that the full parameter set is mutually
consistent.

## 15. Parameters Still Requiring a Decision

The following are intentionally not given silent defaults:

| Parameter | Why unresolved | Required decision |
|---|---|---|
| Ages 0-2 distribution | Source documents require the cohort but provide no probability or maturation equation. | Define its total-rate and composition relationship, or formally remove it from version 1. |
| Maximum project resampling attempts | Added by the new bounded-resampling design. | Choose a default after measuring rejection frequency in a prototype. |
| Exact bounded state-share calibration method | A direct shift can violate bounds; order-preserving calibration needs a specified numerical method. | Select bounded-logit intercept calibration or another explicit algorithm. |
| Calibration tolerances for “near” correlations | Legacy targets are approximate. | Run canonical simulations and approve numeric intervals. |
| SES quadratic mechanism | Current negative quadratic is an inverted U, while legacy prose says fertility is high at both ends. | Approve the formula or reverse the curvature to match the prose. |
| Canonical interpolation mode | The observation layer is optional. | Decide whether default outputs emulate missing historical observations. |
| Observable mean area for floor inversion | The current floor proxy uses $\bar A_b$, but the planned visible building table retains only area-category shares, not exact mean area. | Either expose a prediction-time mean-area field, or define an estimate from category shares and representative category values. Do not use latent true mean area in a visible proxy. |

These are specification decisions, not coding details. Implementation should
not proceed by inventing values for them inside methods.