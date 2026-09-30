# Data generation guide — student forecasting simulator, Tel Aviv

> **Status: legacy full-model source.** This document is preserved for its
> original assumptions and derivations and may conflict with the current
> simplified model. Use the [documentation index](../README.md) and the
> [current model specification](../SIMPLIFIED_MODEL_PLAN.md) for active work.

A step-by-step build guide for the synthetic dataset. Read Sections 0–2 before writing any code; the rest is executable in order.

**No real CBS data is available.** Every population-level variable must be generated. See Section 3.

---

## 0. What you are building

Five tables describing residential buildings in Tel Aviv-Yafo occupied between 2000 and 2022, plus the number of children living in each building three years after first occupancy, in three cohorts.

| Cohort | Ages | Birth-years wide |
|---|---|---|
| kindergarten | 3–6 | 4 |
| primary | 7–12 | 6 |
| secondary | 13–18 | 6 |

Cohort widths `w = (4, 6, 6)` are unequal. Any per-birth-year comparison needs `p_k / w_k`. Comparing `p_k` directly across cohorts is wrong.

The **building** is the unit of observation, not the project. A project holds 100–2,500 units across 1–14 buildings that are occupied at different times, and that staggering is signal: a late building faces a different environment from an early one in the same project.

Two stages:

```
N_b                        ~ NB2(mu_b, phi_b)              # total children 3-18
(n_kg, n_pr, n_sec) | N_b  ~ DirMult(N_b, tau_b * p_b)     # split into cohorts
```

Stage 1 is about **how many**; stage 2 is about **which ages**. They use different coefficient sets because they are driven by different things — exposure and a child coefficient versus the life-cycle stage of the incoming household.

### Deliverables

```
cbs_population_data.py    # population panel: simulate() now, ingest_cbs() later
simulate_students.py      # buildings, environment, both model stages
validate_simulation.py    # the checks in Section 10
```

Dependencies: `numpy`, `pandas`, `scipy`, `openpyxl`; `statsmodels` for validation only.

---

## 1. Non-negotiable implementation rules

**One random generator.** Create `rng = np.random.default_rng(SEED)` once in `main()` and pass it into every function. Never call `np.random.*` directly and never seed inside a function — both silently destroy reproducibility.

**Constants live at module level.** Every number in this guide goes into a module-level dict or constant. No magic numbers inside function bodies. When someone asks "where does 0.30 come from", the answer must be one grep away.

**Latent variables live inside their table, clearly marked.** Do not put them in a separate file — validation should be a column comparison, not a join. Suffix them `_true` or prefix them `latent_`, and keep a single explicit list of model-visible columns so nothing leaks by accident.

**Standardise inside the model step, not in the stored table.** Store raw values; z-score at the point of use. Storing z-scores makes the tables uninterpretable and couples them to the sample.

**Assert, do not warn.** The checks in Section 10 should raise. A warning in a log nobody reads is not a check.

---

## 2. Sampling order

Later steps depend on earlier ones. Do not reorder.

```
1. statistical areas (baselines)
2. projects (zoning, size, timing)
3. buildings (phasing, floors, form, room mix, areas, units)
4. exposure (realisation rate, lag, fill duration)
5. population panel  <- needs step 3's occupancy events for the jump terms
6. environment       <- needs step 5's housing stock as denominator
7. stage 1 (total children)
8. stage 2 (cohort split)
9. output decomposition
```

Steps 3 and 5 are mutually dependent in appearance only. Resolve it by generating buildings first, collecting their occupancy events, then building the panel:

```python
areas    = build_areas(rng)
projects = build_projects(rng, areas)
builds   = build_buildings(rng, projects, areas)   # uses baseline SES, not the panel
events   = builds.groupby(["stat_area_id", "t_occupancy"]).U_new.sum().to_dict()
panel    = population_panel(rng, areas, occupancy_events=events)
env      = build_environment(rng, builds, panel)
```

Buildings use each area's **baseline** SES for the room-mix tilt, not the year-specific panel value. This breaks the circularity without materially changing the result.

---

## 3. The CBS problem, and what to do about it

CBS statistical-area data was not obtained and cannot be fetched programmatically. **All population-level inputs are synthetic.** For a simulator this is correct rather than a compromise — the panel is properly an output of the process — but it bounds what the results mean:

- Structure is faithful: hierarchy, correlations, trends, jumps at occupancy.
- Absolute levels are calibrated from a few published city-wide figures, not measured per area.
- Valid for testing the pipeline, pricing proxy error, and checking coefficient recovery. **Not valid as a forecast.**

### Calibration anchors — the only real numbers here

| Quantity | Value | Source |
|---|---|---|
| City population | ~468,000 | Statistical Abstract of Israel, CBS |
| Mean household size | ~2.3 (Israel 3.2) | Statistical Abstract of Israel, CBS |
| Households with children ≤17 | ~52,700 (2022) | Statistical Yearbook of Tel Aviv-Yafo |
| Share in state education | ~0.70 | Municipal / CBS education statistics |

### A structural fact to design around

CBS publishes area-level demographics **annually from about 2017**, and before that only at census anchors (1995, 2008, 2022). So interpolation is needed for **2000–2016 only**, not the whole period.

Reproduce this asymmetry in the simulator: mask pre-2017 non-anchor years, leave 2017+ observed. It makes error quality vary systematically by building vintage, which is a real property of the eventual dataset and something people are otherwise surprised by later.

### Share in state education is not published per area by anyone

The real-data path runs through municipal pupil-registration records — resident pupils in state schools over all registered pupils, per area and year. Until then it must be simulated, and specifically as a **mixture** (Section 4).

### Structure the module for a clean swap

```python
def population_panel(rng, areas, occupancy_events):
    """Synthetic panel. One row per (stat_area_id, year), 2000-2022.
    Columns: hh, ses, medage, state -- each in _true and _obs form."""

def ingest_cbs(paths):
    """Same schema from downloaded files. Raise a clear error naming
    exactly which files are missing -- never silently fall back."""
```

Sources to document in the module docstring (all require manual download):

- Statistical-area layers with demographic attributes, annual from ~2017: `https://www.cbs.gov.il/he/Pages/geo-layers.aspx` — zipped shapefiles; the attribute table carries population and age counts. **Attribute names change between vintages**, so the rename map is the one thing to re-check when adding a year.
- Socio-economic index of statistical areas, published by CBS per index round.
- Census anchors 1995, 2008, 2022 for the pre-2017 stretch.
- Tel Aviv-Yafo statistical yearbook: `https://www.tel-aviv.gov.il/Transparency/Pages/Statistics.aspx`.

---

## 4. Step 1 — Statistical areas

180 areas nested in 9 districts. The district layer is real, not decorative: it is a level in the generating hierarchy, the target of partial pooling during estimation, and the fallback when a value is missing. The first two agree by construction, which is the point.

```python
N_AREAS, N_DISTRICTS = 180, 9

for each area a:
    district(a)     ~ Uniform{1..9}
    SES_district    ~ Normal(0, 1)             # drawn ONCE per district
    gentr_district  ~ Uniform(0, 0.045)        # gentrification slope, per district
    centroid (x, y) ~ Uniform over ~9 x 12 km, in km
    stock_a(0)      ~ Uniform{900, 6500}       # existing housing units
    pop_a(0)        ~ Uniform{2500, 9000}      # rescale so the sum is 468,000
    SES_a           = SES_district + Normal(0, 0.55)
    gentr_a         = gentr_district + Normal(0, 0.008)
    Hbar_a          ~ Normal(2.55, 0.30) clipped [1.8, 4.2]
    medage_a        ~ Normal(36.5, 4.0) clipped [26, 50]
    state_a         ~ mixture, below
    built_char_a    ~ Normal(0, 1)             # persistent high-rise propensity
```

### Built character — draw it once, reuse it forever

`built_char_a` is drawn once per area and reused for **every building ever built there**. It feeds the high-rise probability in Section 6.

Without it, building form is drawn from the year alone, so a low-rise district sprouts towers at random and a tower district produces walk-ups. That does not happen in reality: planning regimes, plot geometry, conservation status and height limits all operate at neighbourhood level and change slowly. With `KAPPA = 1.2`, roughly a third of areas stay predominantly low-rise even in 2022 while about a quarter become predominantly high-rise, and the city-wide trend is unchanged.

This also makes mix contrast (Section 6.5) meaningful — an area needs a stable character for a building to depart from it.

### The socio-economic index: use the right number

CBS publishes a **continuous standardised index** (mean 0, SD 1 nationally, range roughly −2.9 to +2.6) and an integer **cluster 1–10** derived from it afterwards. Use the continuous index — every formula here expects a z-score, and a 1–10 cluster inflates each SES term about threefold and shifts its centre by roughly six standard deviations. If only the cluster is available, map back with `z ≈ (cluster - 5.5) / 2.5`. Tel Aviv sits above the national mean, so choose national or within-city centring explicitly and stay consistent.

---

## 5. Step 2 — Projects

```python
N_PROJECTS = 320

UNITS_PER_BUILDING = {"new_hood": 70, "clearance": 85, "tama38_2": 60, "tama38_1": 40}
MIN_BUILDINGS      = {"new_hood":  4, "clearance":  2, "tama38_2":  1, "tama38_1":  1}

for each project p:
    target_units = clip(LogNormal(log(450), 0.75), 100, 2500)
    t_first      ~ Uniform{2000..2022}
    s            = t_first - 2000

    P(new_neighbourhood) = sigmoid(1.20 - 0.13 * s)     # 0.77 -> 0.16
    else split (clearance, tama38_2, tama38_1) as (0.35, 0.40, 0.25)

    n_buildings  = clip(round(target_units / UNITS_PER_BUILDING[zoning]),
                        MIN_BUILDINGS[zoning], 14)

    centroid = area centroid + Normal(0, 0.25) km
```

**Derive the building count from a target project size**, never the reverse. Drawing buildings first and letting their sizes accumulate produces projects outside the 100–2,500 range, and you will only notice when the acceptance check fails.

**Multi-building projects are the norm, and the exceptions are systematic rather than random.** TAMA 38/1 reinforces one existing building and is genuinely single-building. TAMA 38/2 demolishes and rebuilds one or two. Clearance-and-rebuild covers a block. A new neighbourhood is several buildings by definition. A uniform draw over 1–14 would put single-building new neighbourhoods and fourteen-building 38/1 projects into the data, neither of which exists.

Zoning determines two derived quantities:

```python
R_RET = {"new_hood": 0.00, "clearance": 0.25, "tama38_2": 0.55, "tama38_1": 0.85}
S_NET = {"new_hood": 1.00, "clearance": 0.65, "tama38_2": 0.45, "tama38_1": 0.15}
```

`R_RET` is the returning-resident share and `S_NET` is the net new-unit share. There is deliberately **no project-size feature** — it is a derived sum, and SPILL already carries the information, since a building in a large project sees its own siblings inside the radius and time window.

---

## 6. Step 3 — Buildings

### 6.1 Phasing and floors

```python
t_occupancy = clip(t_first + Uniform{0, n_buildings-1} + Uniform{0,1}, 2000, 2022)
s = t_occupancy - 2000

KAPPA = 1.2
P(high_rise | s, a) = sigmoid(-1.50 + 0.11*s + KAPPA * built_char_a)
F = 12 + Poisson(10) if high_rise else 4 + Poisson(3)
F = clip(F, 3, 45)
high_rise = 1 if F > 10 else 0
building centroid = project centroid + Normal(0, 0.06) km
```

**Floors are the primitive; form is derived from them.** This ordering resolves the "mixed" question: at building level there is no mixed category, only above or below 10 floors. "Mixed" is a property of a *project* that contains buildings on both sides. Treating it as a third building-level category creates a phantom category and wastes a parameter.

Zoning and form are drawn **from the year**, not independently of it. Drawing them independently produces combinations that never occurred — 2003 high-rise renewal projects, for instance. Form is additionally drawn from the **area's** built character, so that form is persistent in space as well as trending in time.

### 6.2 Room mix

The room mix has **two** area-dependent layers, and both matter. The first is the prior over room counts in that area; the second is the building's own deviation from it.

```python
ROOMS = [2, 3, 4, 5, 6]
PRIOR = [0.10, 0.28, 0.34, 0.20, 0.08]
GAMMA_SES, GAMMA_HH = 0.25, 0.45

# (a) area-conditional prior: what this AREA typically builds
prior_a_k = PRIOR[k] * exp((GAMMA_SES*z_SES + GAMMA_HH*z_Hbar) * (k - 4) / 2)
prior_a   = normalise(prior_a_k)

# (b) building tilt on top of the area prior
tilt = -0.35 * high_rise - 0.020 * s
w_k  = prior_a[k] * exp(tilt * (k - 4) / 2)
mix  ~ Dirichlet(45 * normalise(w))
```

**Why the prior itself is area-dependent.** An area with large households contains more 5- and 6-room apartments before anything else is known, and an area of small households more 2- and 3-room ones. Modelling only the *conditional mean* of area given rooms (Section 6.3) while leaving the prior city-wide is a half-measure: it captures that rooms are **sized** differently across the city but not that they are **distributed** differently. Both channels are needed for the inversion in Section 14 to give genuinely different answers in different areas.

The tilt is centred on `k = 4` so a neutral area reproduces `PRIOR` exactly. Household size carries the larger weight because it bears more directly on room demand than income does. Across the range of `z_Hbar` this shifts mean room count by roughly 0.8 rooms.

The Dirichlet concentration 45 controls how much buildings differ from each other within an area. Lower it and every building looks like the area average.

### 6.3 Floor areas — the central proxy

Households choose **rooms**; the data records **area**. So generate rooms and derive areas, never the reverse.

```python
ALPHA0, ALPHA1, ALPHA2, ALPHA3, ALPHA4, ALPHA5, SIGMA_A = 8, 22, 4, -3, -0.15, -2.5, 9

mu_k = (ALPHA0 + ALPHA1*k + ALPHA2*z_SES + ALPHA3*high_rise
        + ALPHA4*s + ALPHA5*z_Hbar)
A | k ~ Normal(mu_k, SIGMA_A^2), clipped [22, 260]
```

Baseline: 3 rooms ≈ 74 m², 4 rooms ≈ 96 m², 5 rooms ≈ 118 m².

**Why each term.** `ALPHA2 > 0`: wealthier areas build more generous rooms for the same count. **`ALPHA5 < 0`**: areas with large households carve more rooms out of the same envelope, so a 4-room flat there is physically smaller. `ALPHA3 < 0`: high-rise floor plates favour compact layouts. `ALPHA4 < 0`: apartments shrank over the period as prices rose.

**`ALPHA2` and `ALPHA5` are the load-bearing terms and must not be dropped.** Together with the area-conditional prior above, they encode that the same floor area means a different room count in different parts of the city — a spacious 3-room flat in a wealthy area of small households, a cramped 4-room one in a dense area of large ones. Without them, proxy error is harmless noise. With them, it becomes systematic bias correlated with area characteristics, which is the realistic case and the thing worth measuring.

Discretise areas into four categories and keep **only the shares**:

```python
CATEGORIES = [(0, 50), (50, 85), (85, 100), (100, inf)]
HHI = sum(share_j ** 2 for j in categories)
```

### 6.4 Unit count and the observable form inversion

```python
mean_k    = mix . ROOMS
mean_area = ALPHA0 + ALPHA1*mean_k + ALPHA2*z_SES + ALPHA3*high_rise + ALPHA4*s
plate     = 450 * exp(Normal(0, 0.18))       # the TRUE plate varies per building
U_planned = clip(round(F * plate / mean_area), 8, 400)

# what a predictor can actually compute:
F_hat     = U_planned * mean_area / 450
form_hat  = 1 if F_hat > 10 else 0
```

**The lognormal plate is load-bearing.** If you treat 450 as a fixed constant during generation, `F_hat` becomes an exact inverse of `F` and the simulator will report near-perfect recovery of building form — a falsely optimistic result that propagates into everything downstream.

### 6.5 Mix contrast — move-up buyers and internal migration

A building whose apartments are **larger than the norm for its statistical area** does not draw the same population as one that matches the norm. It attracts move-up buyers, and in Tel Aviv those come disproportionately from within the same area or its immediate neighbours — a family trading a 3-room flat for a 5-room one a few streets away.

Nothing else in the feature set captures this, because every mix feature is absolute. A building with 40% five-room apartments means something completely different in an area where that is typical and in an area where nothing above three rooms exists.

```python
mean_rooms_building = mix . ROOMS                     # from the estimated shares
mean_rooms_area     = prior_a . ROOMS                 # the area's existing profile
mix_contrast_b      = mean_rooms_building - mean_rooms_area
```

Available at prediction time, since both terms come from estimated room shares and the area's housing profile. It enters in **three** places, and the third matters most:

| Where | Coefficient | Mechanism |
|---|---|---|
| Stage 1 mean | +0.04 | Relatively large flats hold somewhat more children. Small — the absolute mix features already carry most of this. |
| Stage 2 index | −0.22 | Move-up families have older children *by construction*: they trade up because the household grew. Shifts composition toward primary and secondary. |
| Relocation share | +0.10 per SD | A high-contrast building draws more of its households from within the city, because the move-up move is short. |

The third is what turns the output decomposition (Section 12) from a flat assumption into something that varies across buildings — and since intra-city relocation is the main double-counting risk against the cohort-progression model, that is where it pays off.

A second-order consequence worth noting: high mix contrast means the building is **atypical for its area**, so the area's population characteristics are less informative about who moves in. Same logic as the intensity interaction in Section 10, arriving from a different direction.

---

## 7. Step 4 — Exposure

```python
nu = 4 + 0.05 * U_planned
R  = 0.7 + 0.5 * Beta(0.6*nu, 0.4*nu)                 # E[R] = 1 by construction
# outliers:
if U_planned < 150 and rng.random() < 0.03:
    R = Uniform(1.5, 2.0)

L        = 2 + 2 * Beta(2, 2)                # construction start -> first occupancy
D_fill   = 0.5 + 2 * Beta(1.5, 3.5)          # first occupancy -> full
t_start  = round(t_occupancy - L)
L_hat    = 2.0 + 0.9*high_rise + 0.002*U_planned      # the OBSERVABLE estimate

U_new = U_planned * R * (1 - R_RET[zoning])
```

`nu` grows with size, so large buildings converge to R = 1 while small ones carry wide variance. That is the realistic pattern and the reason small buildings should be harder to predict.

**Returning residents reduce exposure here.** They are not a second sub-population with their own coefficients. This is what prevents double counting against the separate cohort-progression model that ages the existing population forward, and it removes an entire parameter set.

`R`, `L`, `D_fill` are **latent** — keep them, never feed them to the model. `L_hat` is the visible substitute.

---

## 8. Step 5 — Population panel

One row per `(stat_area_id, year)` for 2000–2022.

```python
for area a, for year t:
    s = t - 2000
    cumulative jumps: for every building in a occupied by year t,
        frac = U_new / stock_a(0)
        hh     += -0.55 * frac
        ses    += +2.20 * frac
        medage += -9.00 * frac
        pop    += +2.40 * U_new

    hh_true     = Hbar_a     - 0.011*s + cum_hh
    ses_true    = SES_a      + gentr_a*s + cum_ses
    medage_true = medage_a   + 0.05*s  + cum_medage
    state_true  = state_a    - 0.002*s
    stock       = stock_a(0) + cumulative U_new
```

Then **hide what could not have been observed**:

```python
visible = year in {1995, 2008, 2022} or year >= 2017
mask everything else, then interpolate the gaps
```

### Construction-aware interpolation — optional, and only to emulate a real constraint

**This step is not needed to produce the data and can be skipped.** The panel is fully synthetic, so the true series is known at every year; nothing has to be reconstructed.

Its purpose is to **emulate a limitation that will exist once real data replaces the synthetic panel**. CBS publishes area-level demographics annually only from about 2017. Before that, values exist at census anchors (1995, 2008, 2022) and the years between must be reconstructed. "Construction-aware" means the reconstruction uses the record of *when units were actually occupied* — from the building table here, from building-completion records in the real system — to decide where inside the interval the change happened, instead of spreading it evenly.

Linear interpolation errs where change is concentrated, and worse: the change it misses is itself caused by construction, so **the error correlates with the explanatory variable**. It produces bias, not attenuation.

```
x_hat(t) = x(c0) + [x(c1) - x(c0)] *
           [ (1 - THETA)*(t - c0)/(c1 - c0) + THETA*G(t)/G(c1) ]

THETA = 0.50
G(t)  = cumulative units occupied in the area since c0
```

`THETA = 0` is plain linear interpolation; `THETA = 1` attributes all change to construction timing. Keep both `x_true` and `x_obs` — the gap between them is one of the three quantities this exercise exists to price.

**If you skip this step**, set `x_obs = x_true` everywhere and say so when reporting, because the simulator will then understate error for buildings from 2000–2016 and the correlation check in Section 15 will come out at zero by construction.

---

## 9. Step 6 — Environment

**Everything here describes the state of the surroundings at the moment this building started construction** — not today, and not when the project began. This is the single most damaging thing to get wrong in the whole codebase, and it produces no error message, so it gets stated before the formulas rather than after.

Three requirements follow:

1. **Institutions need opening and closing dates, not just locations.** A kindergarten opened in 2015 must not be counted for a building that started in 2008. The institutions layer is therefore a *panel*, not a snapshot — and any GIS layer that exists only as a current snapshot cannot be used at all, because using today's value for a 2004 building is leakage from the future.
2. **The stock denominator must be contemporaneous.** Both intensity and SPILL divide by the area's stock at `t_start`.
3. **The key is the building's own `t_start`, never the project's.**

```python
stock_a(t)  = stock_a(0) + sum of U_new for buildings in a occupied by t
intensity_b = 1000 * U_new_b / stock_a(b)(t_start_b)

SPILL_b = (1000 / stock_a(b)(t_start_b)) * sum over b' != b of U_new_b'
          where distance(b, b') <= 1 km
          and   t_start_b - 5 <= t_occupancy_b' < t_start_b

daycare_b ~ Poisson(0.9 + 0.0016*stock_a(t_start) + 0.05*(t_start - 2000))

p_exists = sigmoid(-0.4 + 0.0004*stock_a(t_start) - 1.3*1{new_neighbourhood})
u ~ Uniform(0, 1)
school_status = "existing" if u < p_exists
                "planned"  if u < p_exists + 0.25
                "none"     otherwise

utilisation ~ Normal(0.92, 0.16) clipped [0.45, 1.6]
load_b = utilisation if school_status == "existing" else 0.0
```

Three things that are easy to get wrong here:

**Key on the building's `t_start`, never the project's.** Using the project start makes the fourth building inherit the environment the first one saw five years earlier, destroying exactly the phasing effect the design exists to capture. This is the single most damaging bug available in this codebase and it produces no error message.

**The stock denominator must be contemporaneous.** A fixed denominator inflates the intensity of the second project in an area, because it never sees the stock the first project added.

**Store `load` pre-multiplied.** Utilisation is undefined when no school exists. Storing raw utilisation with zeros or nulls for "no school" contaminates the coefficient. One column, already multiplied by the existence indicator.

`SPILL` solves three problems in one sum with no new parameters: phased projects, two projects in one area, and infrastructure lagging construction.

---

## 10. Step 7 — Stage 1, total children

```
log mu_b = log(U_new_b) + beta_0 + x_b . beta + interactions + u_project + w_area

u_project ~ Normal(0, 0.10^2)      # one draw per project
w_area    ~ Normal(0, 0.13^2)      # one draw per statistical area
phi_b     = min(2 + 0.06 * U_new_b, 25)
lambda_b  ~ Gamma(shape=phi_b, scale=mu_b/phi_b)
N_b       ~ Poisson(lambda_b)
```

```python
BETA = {
    "intercept":       -0.79,   # exp(-0.79) = 0.45 children 3-18 per new unit
    "share_3room":     +0.14,
    "share_4room":     +0.21,
    "share_5room":     +0.26,
    "mix_contrast":    +0.04,
    "high_rise":       -0.13,
    "intensity":       +0.05,
    "new_hood":        +0.10,   # clearance-and-rebuild is the reference level
    "tama38_2":        -0.05,
    "tama38_1":        -0.25,
    "daycare":         +0.06,   # applied AFTER the N2 saturation transform
    "school_existing": +0.10,
    "school_planned":  +0.04,   # "none" is the reference level
    "load":            -0.07,
    "spill":           +0.05,
    "hh_size":         +0.30,
    "ses_linear":      +0.05,
    "ses_quadratic":   -0.07,
    "median_age":      -0.05,
    "state_share":     -0.20,
}
INTERACTIONS = {"I1": +0.09, "I2": +0.07, "I3": -0.06, "I4": -0.05}
```

**Room shares are compositional.** The three shares plus an omitted "2 rooms or fewer" baseline sum to 1. Include all four and the design matrix is singular.

**`phi` rises with building size.** A fixed dispersion would make every building equally predictable in relative terms, which is wrong — small buildings genuinely deviate more.

**There is no period coefficient.** Household size already carries the fertility decline, and there is an opposing crowding channel (prices rose, so the same room mix absorbs more people). The two roughly cancel conditional on room mix. Adding a period term double counts.

### Interactions

| ID | Formula | Coefficient | Why |
|---|---|---|---|
| I1 | `z_5room * z_hh_size` | +0.09 | Large apartments produce children only where a population exists to fill them. In an area of small households a 5-room flat is a space upgrade, not a family home. |
| I2 | `high_rise * z_5room` | +0.07 | The negative high-rise coefficient rests on high-rises having small units. A high-rise of 5-room flats is different; without this the model penalises it twice for one reason. |
| I3 | `z_spill * z_load` | −0.06 | Rapid absorption **combined with** full school utilisation deters. Neither alone does — a crowded area with no recent wave is already at equilibrium. |
| I4 | `z_intensity * z_hh_size` | −0.05 | When a project is huge relative to its area, the incoming population *replaces* the existing one, so existing-population features lose predictive power. |

### Non-linear transforms

```python
N1: 0.05 * z_ses - 0.07 * z_ses**2     # inverted U: fertility high at both ends
N2: 1 - exp(-daycare_count / 3)        # 0->0, 1->0.28, 3->0.63, 6->0.86
```

N2 is a **fixed transform, not an estimated parameter**. Going from zero daycares to one changes the picture; eight to nine does not. On a linear scale an area with twelve daycares dominates the estimate and drags the coefficient down for everyone.

---

## 11. Step 8 — Stage 2, cohort split

**Do not write a full multinomial logit.** Most features act on composition through one dimension — how young the incoming household is. A single latent index costs about a third of the parameters and, more importantly, **enforces the correct monotone ordering**: anything that makes a building younger must move mass from secondary toward kindergarten, with primary in between. An unconstrained logit can violate that while still fitting.

```python
DELTA = {
    "share_3room":  +0.38,
    "share_4room":  +0.10,
    "share_5room":  -0.28,
    "new_hood":     +0.35,
    "daycare":      +0.15,     # after N2
    "median_age":   -0.28,
    "hh_size":      -0.20,
    "ses":          -0.10,
    "state_share":  -0.12,
    "mix_contrast": -0.22,
}
A_KG, C_SEC = 1.0, 1.2
P0 = (0.42, 0.36, 0.22)        # calibrated to a building at age 3

theta   = z . DELTA
eta_kg  = gamma_kg  + A_KG * theta  + 0.18 * 1{school_status == "none"}
eta_pr  = 0                                      # primary is the reference
eta_sec = gamma_sec - C_SEC * theta + 0.45 * z_hh_size
```

The two extra terms are effects that are **not** monotone on the youth axis. The household-size term captures the part of household size that is not "youth": large families hold children in all three cohorts at once, so they raise secondary even when the index points young. Forcing it through `theta` would produce the wrong sign somewhere.

### Intercept calibration — do not skip

Softmax convexity pushes mass toward the extremes and compresses the middle category, so `gamma_kg` and `gamma_sec` are **not** `log(P0_k / P0_pr)`. Solve iteratively:

```python
gamma_kg  = log(P0[0] / P0[1])
gamma_sec = log(P0[2] / P0[1])
for _ in range(60):
    p = softmax(stack([gamma_kg + base_kg, zeros, gamma_sec + base_sec]))
    m = p.mean(axis=0)
    gamma_kg  += 0.7 * (log(P0[0]/P0[1]) - log(m[0]/m[1]))
    gamma_sec += 0.7 * (log(P0[2]/P0[1]) - log(m[2]/m[1]))
assert allclose(p.mean(axis=0), P0, atol=0.02)
```

Skipping this pulls the middle cohort down by roughly a quarter of its mass, silently. Uncalibrated output looks like `(0.468, 0.269, 0.263)` instead of `P0`.

### Concentration and draw

```python
log_tau = 2.5 + 0.5*z_HHI + 0.3*log(U_new / 50) - 0.4*z_D_fill
tau     = clip(exp(log_tau), 4, 80)

q_b ~ Dirichlet(tau_b * p_b)
n_b ~ Multinomial(N_b, q_b)
```

**HHI belongs here, not in stage 1.** It barely affects how many children there are; it affects how *concentrated* the composition is. A building entirely of 3-room units draws a homogeneous population with a predictable split; one mixing 2, 4 and 6 rooms draws three household types and is volatile. Putting HHI in the mean is the natural mistake and yields a near-zero coefficient plus a badly calibrated variance.

`D_fill` enters here too: a building that filled in six months is measured with an older, tighter composition than one that took two and a half years.

`P0` is far younger than the stationary profile implied by `w = (4,6,6)`, which would be `(0.25, 0.375, 0.375)`. That is deliberate — it is calibrated to a three-year-old building, and this is where the demographic maturation effects are absorbed.

---

## 12. Step 9 — Output decomposition

The main double-counting risk against the cohort-progression model is not returning residents but **intra-city relocation**: in Tel Aviv most buyers of new apartments already live in the city. Expose the split rather than guessing where the boundary lies.

```python
LEAK = (0.02, 0.05, 0.15)      # resident -> registered, per cohort

# the relocation share is NOT a constant -- see Section 6.5
rho_b = clip(0.60 + 0.10 * z_mix_contrast_b, 0.35, 0.85)

n_returning    = n * R_RET[zoning]
n_intra_city   = n * (1 - R_RET[zoning]) * rho_b
n_from_outside = n * (1 - R_RET[zoning]) * (1 - rho_b)
n_net_budget   = n * S_NET[zoning]
n_registered   = n * (1 - LEAK[k])       # only if the target is registered pupils
```

Whoever maintains the cohort-progression model subtracts what they already count. Leakage — boarding schools, private, out-of-city — is largest in the secondary cohort.

**Why `rho` varies rather than sitting at a constant 0.60.** Holding it fixed assigns the same double-counting correction to a TAMA 38/1 reinforcement and to a new-neighbourhood tower, which is clearly wrong. A building that is much larger-grained than its surroundings pulls families who are already in the city; one that matches its surroundings, or a new neighbourhood on former industrial land, draws more from outside.

---

## 13. Table layouts

Mark every column as model-visible or latent. Keep one explicit list of visible columns.

**building** — `building_id`, `project_id`, `stat_area_id`, `bx`, `by`, `t_start`, `t_occupancy`, `U_planned`, `L_hat`, four area-category shares, `share_3room/4room/5room` (estimated), `hhi`, `form_hat`, `intensity`, `mix_contrast`, `zoning`
*latent:* `F_true`, `rooms_true_mean`, true room shares, `R_true`, `L_true`, `D_fill_true`, `U_new`, `plate_true`, hidden 0–2 cohort

**project** — `project_id`, building list, `zoning`, `n_buildings`, `t_first`, centroid, derived `r_ret`, `s_net`, aggregated totals

**environment** — `building_id`, `t_start`, `stock_t0`, `intensity`, `spill`, `daycare_n`, `school_status`, `load`
*latent:* `utilisation_raw`

**population** — `stat_area_id`, `district_id`, `year`, `hh_obs`, `ses_obs`, `medage_obs`, `state_obs`, `stock`, `pop`, `cum_units`
*latent:* `hh_true`, `ses_true`, `medage_true`, `state_true`

**areas** — `stat_area_id`, `district_id`, centroid, `stock0`, `pop0`, baselines, `built_char`, and the area's existing room profile used for mix contrast

**results** — `building_id`, `n_kg`, `n_primary`, `n_secondary`, `N_total`, `p_kg/pr/sec`, `theta`, `tau`, `rho_internal`, the decomposition columns, `year_measured`

---

## 14. Estimating rooms from area — for downstream consumers

The generator knows the true room counts. Anyone consuming the tables sees only area categories and must estimate. Ship this as a standalone reusable function.

**Do not use a deterministic inversion.** `k_hat = (A - 8)/22` returns one number per flat and erases the uncertainty that carries the information. It makes the estimate look certain when it is not.

Invert with Bayes over a category `[l, u)`:

```
Pr(k | l <= A < u, a)  ∝  prior_a[k] * [ Phi((u - mu_k)/sigma) - Phi((l - mu_k)/sigma) ]

mu_k     = 8 + 22k + 4*z_SES - 3*high_rise - 0.15*s - 2.5*z_Hbar
sigma    = 9
prior_a  = the AREA-conditional room prior from Section 6.2
```

Both the prior and the conditional mean are area-dependent, so the same floor-area category yields different room probabilities in different areas. That is the point — a 90 m² flat is genuinely more likely to be 4 rooms in a dense area of large households and 3 rooms in a wealthy area of small ones.

Then aggregate to the building:

```
share_hat_k(b) = sum over categories j of  share_j(b) * Pr(k | category j)
```

This yields **shares**, propagates uncertainty, and lets a downstream model learn an appropriately attenuated coefficient. Point-estimate fallback only if something demands a scalar: `clip(round((A - 8)/22), 1, 6)`.

**Consequence:** because the mapping depends on both SES and household size, **both must stay in the model even if their direct predictive effects are weak**. They are doing proxy correction, not prediction. This is the least obvious dependency in the design and the one most likely to be pruned by someone selecting features on predictive power alone.

---

## 15. Validation

Assert these; do not warn.

| Check | Expected |
|---|---|
| Children per new unit, overall | 0.45–0.50 |
| Realised composition vs `P0` | within 0.02 per cohort |
| Project totals | every project within 100–2,500 units |
| Estimated vs true 4-room share | correlation ≈ 0.75 |
| Estimated vs true floor count | correlation ≈ 0.93 |
| Building-form classification | ≈ 94% correct |
| m² per room across SES terciles | visible gradient; correlation with SES ≈ +0.7 |
| m² per room across household-size terciles | visible gradient in the opposite direction |
| Form persistence within area | correlation of form across buildings in one area clearly positive |
| Buildings per project | median above 1; single-building projects confined to TAMA 38 |
| Mix contrast | roughly centred on zero, with meaningful spread |
| Intra-city relocation share | varies across buildings, inside [0.35, 0.85] |
| Interpolation error vs distance from anchor | monotonically increasing, zero at an anchor |
| Corr(interpolation error, cumulative units) | clearly non-zero — proves bias, not noise |
| Coefficient recovery | ≈ 90% of coefficients within ±2 standard errors |

The recovery test fits `statsmodels` NB2 GLM with `offset=log(U_new)` against `BETA`. Coefficients for features deliberately omitted from the recovery regression will be biased — that is the intended demonstration of omitted-variable bias, not a defect.

If the interpolation error comes out at exactly zero, check that occupancy events are actually being passed into the panel builder. With no construction the true series is linear in time and interpolation is exact, which hides the whole effect.

---

## 16. Pitfall checklist

Ordered by how easy each is to commit and how much damage it does.

1. **Keying environment or population on the project's start date instead of the building's.** Silently destroys the phasing effect the design exists to capture.
2. **Feeding the 1–10 socio-economic cluster into a formula expecting the standardised index.** Inflates every SES term threefold and shifts its centre by about six standard deviations.
3. **Skipping softmax intercept calibration.** Compresses the middle cohort by ~25% of its mass, with no error.
4. **Drawing building form from the year alone.** Scatters towers through low-rise districts. Use the area's built character.
5. **A city-wide room prior.** Captures that rooms are sized differently across the city but not that they are distributed differently — a half-measure.
6. **Fixed floor plate during generation.** Makes the form inversion exact and produces falsely optimistic recovery.
7. **Deterministic area-to-rooms inversion.** Hides the uncertainty that is the point.
8. **Dropping SES or household size because their direct effects are weak.** Both do proxy correction; removing them converts a correctable bias into an uncorrectable one.
9. **A constant intra-city relocation share.** Assigns the same double-counting correction to a 38/1 reinforcement and a new-neighbourhood tower.
10. **A uniform draw for buildings per project.** Produces single-building new neighbourhoods and fourteen-building 38/1 projects, neither of which exists.
11. **Including all four room-share categories.** Singular design matrix.
12. **HHI in the mean instead of the concentration parameter.** Near-zero coefficient, badly calibrated variance.
13. **Raw utilisation with zeros where no school exists.** Contaminates the coefficient.
14. **Fixed housing-stock denominator.** Inflates the second project's intensity.
15. **A period coefficient on top of the household-size trend.** Double counts the fertility decline.
16. **Returning residents as a second sub-population.** Double counts against the cohort-progression model.
17. **Drawing building counts before project sizes.** Produces projects outside 100–2,500.
18. **Seeding inside functions or calling `np.random` directly.** Destroys reproducibility.
19. **Reporting simulator output as a forecast.** Absolute levels come from assumptions, not data.

---
