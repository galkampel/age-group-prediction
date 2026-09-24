---
title: "Student Forecasting Simulator for New Residential Projects in Tel Aviv"
subtitle: "How to build the synthetic data-generating process — transformation functions, assumptions, and pitfalls"
---

> **Status: legacy full-model source.** This document is preserved for its
> original rationale and assumptions and may conflict with the current
> simplified model. Use the [documentation index](../README.md) and the
> [current model specification](../SIMPLIFIED_MODEL_PLAN.md) for active work.

# 1. Purpose and scope

This document specifies **how to construct** a synthetic dataset for a model that predicts, at the moment construction begins on a **building** in a Tel Aviv-Yafo residential project, how many children will live in it three years after first occupancy, split into three cohorts.

It is a construction manual, not a report. No data has been generated here. Section 7 is a catalogue of every transformation function the generator needs — each with its formula, its constants, and the reason it takes the form it does. Section 11 lists all assumptions with their status and their risk.

| Cohort | Ages | Birth-years wide |
|---|---|---|
| Kindergarten | 3–6 | 4 |
| Primary | 7–12 | 6 |
| Secondary | 13–18 | 6 |

Cohort widths **w = (4, 6, 6)** are unequal, so any comparison of per-birth-year rates requires p̃_k = p_k / w_k. Comparing p_k directly across cohorts is a mistake.

**The unit of observation is the building.** Project-level figures come from aggregating over buildings. A project holds 100–2,500 units across **typically several buildings** occupied at different times, and that staggering is signal rather than noise: a late building faces a different environment from an early one in the same project.

**Prediction time** is t₀, the start of construction. Anything revealed afterwards is unavailable: realisation rate, actual room counts, buyer identity, sale price. **Measurement time** is t₀ + L̂ + 3.

**Scope boundary.** The model accounts for the increment from new construction only. The existing population is aged forward by a separate cohort-progression model, so returning residents must reduce exposure rather than form a second sub-population.

Two stages, driven by different things and therefore carrying different coefficient sets:

$$N_b \sim \mathrm{NB2}(\mu_b, \phi_b), \qquad (n_{\mathrm{kg}}, n_{\mathrm{pr}}, n_{\mathrm{sec}}) \mid N_b \sim \mathrm{DirMult}(N_b,\ \tau_b\,\mathbf{p}_b)$$

**In words:** the first stage decides **how many** children the building holds — an integer drawn from a negative binomial with mean μ and dispersion φ. The second stage splits that number across the three cohorts using a probability vector p, with τ controlling how tightly the realised split hugs the expected one.

# 2. Why the population data must be simulated

**CBS statistical-area data was not obtained and could not be retrieved programmatically.** Every population-level input must therefore be generated synthetically. For a simulator this is correct rather than a compromise — the population panel is properly an *output* of the generating process — but it bounds what the results mean:

- The **structure** will be faithful: district-to-area hierarchy, correlations between variables, time trends, and discrete jumps at occupancy events.
- The **absolute levels** are calibrated from a handful of published city-wide figures, not measured per area. They are plausible, not accurate.
- The simulator is valid for testing the pipeline, quantifying proxy error, and checking whether coefficients are recoverable. It is **not** valid for producing an actual forecast.

**Calibration anchors**, the only figures taken from published sources: Tel Aviv-Yafo population of roughly 468,000; mean household size near 2.3 persons against 3.2 nationally; about 52,700 households with children up to age 17 as of 2022; roughly 70% of pupils registered in state education.

Write the population module behind one interface with two implementations, so a later swap to real data touches nothing downstream: a `simulate` path used now, and an `ingest` path that reads downloaded files and raises a clear error naming exactly what is missing — never silently falling back.

**The socio-economic index — which number was chosen.** CBS publishes two quantities: a continuous standardised index value (mean 0, SD 1 nationally, ranging roughly −2.9 to +2.6) and an integer cluster 1–10 derived from it. **The continuous index was chosen**, because every formula here expects a z-score; substituting a 1–10 cluster inflates each SES term roughly threefold and shifts its centre by about six standard deviations. If only the cluster is available, map it back with z ≈ (cluster − 5.5) / 2.5. Tel Aviv sits above the national mean, so decide explicitly between national and within-city centring and stay consistent.

# 3. The time dimension

## 3.1 Time is a generator, not a feature

Time is not a variable in the prediction model. It is the latent driver that produces everything else. The simulator draws the occupancy year **first** and conditions every other draw on it. Let s = occupancy year − 2000, ranging over 0–22.

Over the period, construction shifts from new neighbourhoods to urban renewal and from low-rise to high-rise. Zoning type and building form are therefore **not** independent of the year — they must be drawn from it, or the synthetic data will contain combinations that never occurred.

Time also drives the room mix (apartments shrink), household size (falls), the socio-economic index (rises in renewal areas), and the stock of institutions in the surroundings (accumulates, lagging construction).

**Watch out for double counting the fertility trend.** Household size already falls over time in the generator and carries the fertility decline. Adding a separate negative period coefficient to the child-count model counts the same effect twice. There is also a genuine counter-channel — prices rose, so the same room mix absorbs more people today — pushing the opposite way. The two roughly cancel *conditional on room mix*, which is why the specification carries **no residual period coefficient at all**.

## 3.2 Why measurement is fixed at three years

A single building fills quickly: apartments are sold off-plan and handover is concentrated, so the 90th percentile of fill duration is about two years. The commonly cited 2–5 years applies at **project** level and reflects phasing, not fill speed.

Fixing measurement at three years places it safely after fill completion, and this is what makes the model tractable: occupancy is close to 1, so **the occupancy curve is not a model variable**; years-since-occupancy is constant, so **it is not a variable**; and demographic maturation coefficients are constant, so **they are absorbed into the composition intercept**.

What remains is the variance around that constant, which enters only through the concentration parameter τ. **Shortening to two years would reinstate the occupancy curve and every parameter attached to it.**

## 3.3 The hidden 0–2 cohort

Ages 0–2 fall outside compulsory education, are not reliably measured, and are excluded from the target — but they must exist in the simulator as a fourth, unreported cohort. Families buy when the child is one year old or before birth, so the kindergarten layer is fed from below rather than only draining upward.

**Omitting this is a modelling error, not a simplification.** Without it the simulator drains kindergarten far too fast in the years after occupancy, and any downstream model calibrated on such data will under-predict kindergarten demand.

# 4. The environment is time-dependent — and this is where most errors happen

Every environment variable describes the **state of the surroundings at the moment the building started construction**, not their state today and not their state when the project began. This deserves its own section because it is the single most damaging mistake available in this design, and it produces no error message.

The join specification, stated precisely:

$$\mathrm{env}(b) = f\big(\text{geometry}(b),\ t_0^b\big), \qquad \text{institution counted} \iff \text{opened} \le t_0^b < \text{closed}$$

**In words:** the environment features of building b are computed from its location, evaluated at **its own** construction-start year. A kindergarten or school is counted only if it had already opened and had not yet closed in that year — one opened in 2015 does not count for a building started in 2008.

$$\mathrm{pop}(b) = \hat{x}_{a(b)}\big(t_0^b\big), \qquad \mathrm{stock}(b) = \mathrm{stock}_{a(b)}\big(t_0^b\big)$$

**In words:** the population and housing-stock values attached to a building are those of its statistical area in that same year, not today's values.

Three practical requirements follow:

1. **Institutions need opening and closing dates, not just locations.** The institutions layer is a panel, not a snapshot. A GIS layer that exists only as a current snapshot is unusable — today's value for a 2004 building is leakage from the future.
2. **The housing-stock denominator must be contemporaneous.** A fixed denominator inflates the second project in an area, because it never sees the stock the first one added.
3. **The key is the building's own t₀, never the project's.** Otherwise the fourth building inherits the surroundings the first one saw, erasing exactly the phasing effect — and the output still looks reasonable.

# 5. Estimating room count from floor area

This is the central proxy in the design and the largest quantifiable source of error.

## 5.1 The mapping depends on the area, not just on the apartment

A household chooses a **room count**; the data records **floor area**. So the simulator generates rooms first and derives areas. But the relationship between the two is **not a city-wide constant** — it depends on who lives in the area, through two separate channels that must both be modelled:

**The conditional mean of area given rooms.** The same 4-room apartment is larger in a wealthy area and smaller in a dense one with large households. So μ_k depends on the socio-economic index and on mean household size, alongside building form and period.

**The prior over room counts itself.** Before observing any area figure, an area with large households is more likely to contain 5- and 6-room apartments, and an area of small households more likely to contain 2- and 3-room ones. Modelling only the conditional mean and leaving the prior city-wide is a half-measure: it captures that rooms are *sized* differently across the city but not that they are *distributed* differently.

Both channels appear as functions F3 and F4 in Section 7. Together they mean the inversion in Section 5.2 gives different answers for the same floor area in different areas — which is the realistic case, and the reason proxy error here is **systematic bias correlated with area characteristics** rather than harmless noise.

## 5.2 The inversion — what to feed the model

**Do not use a deterministic inversion.** Computing a single room count per apartment erases the very uncertainty that carries the information, and makes the estimate look certain when it is not.

Invert with Bayes over a floor-area category [l, u), which is what the data actually contains, using the area-conditional prior and the area-conditional mean:

$$\Pr(k \mid l \le A < u,\ a) \;=\; \frac{\pi_k(a)\Big[\Phi\big(\tfrac{u-\mu_k(a)}{\sigma_A}\big) - \Phi\big(\tfrac{l-\mu_k(a)}{\sigma_A}\big)\Big]}{\sum_{k'}\pi_{k'}(a)\Big[\Phi\big(\tfrac{u-\mu_{k'}(a)}{\sigma_A}\big) - \Phi\big(\tfrac{l-\mu_{k'}(a)}{\sigma_A}\big)\Big]}$$

Then aggregate to a building-level share:

$$\widehat{\mathrm{share}}_k(b) \;=\; \sum_{j} \mathrm{share}_j(b)\cdot \Pr(k \mid \text{category } j,\ a(b))$$

This yields **shares** rather than counts, propagates uncertainty forward, and lets a downstream model learn an appropriately attenuated coefficient.

## 5.3 Consequence for feature selection

Because the mapping depends on the socio-economic index and on household size, **both must stay in the model even if their direct predictive effect is weak**. They are doing proxy correction, not prediction. This is the least obvious dependency in the specification and the one most likely to be removed by someone pruning features on predictive power alone.

# 6. Move-up buyers and internal migration

A building whose apartments are **larger than the norm for its statistical area** does not draw the same population as one that matches the norm. It attracts move-up buyers, and in Tel Aviv those come disproportionately from within the same area or its immediate neighbours — a family trading a 3-room flat for a 5-room one a few streets away.

Nothing in the feature set captures this, because every mix feature is absolute. A building of 40% five-room apartments means something completely different in an area where that is typical and in an area where nothing above three rooms exists. The fix is a single derived feature, **mix contrast**:

$$\Delta k_b \;=\; \bar{k}_b \;-\; \bar{k}_{a(b)}$$

**In words:** mix contrast is the building's mean room count minus the area's mean room count. Positive means the building is coarser-grained than its surroundings. Both terms are available at prediction time — the first from estimated room shares, the second from the area's existing housing profile.

the building's mean room count minus the area's. It is available at prediction time, since both terms come from the estimated room shares and the area's existing housing profile. It enters in three places, and the third is the one that matters most:

**Stage 1, weakly positive.** Relatively large apartments hold somewhat more children. Small effect — the absolute mix features already carry most of this.

**Stage 2, negative on the youth index.** Move-up families have older children by construction: they are trading up *because* the household grew. Positive contrast therefore shifts composition toward primary and secondary, and this is a distinct effect from the absolute share of large apartments.

**The output decomposition.** The intra-city relocation share is not a constant. A building with high mix contrast draws more of its households from within the city, because the move-up move is short. Making the relocation share a function of contrast is what turns the output decomposition from a flat assumption into something that varies meaningfully across buildings — and since intra-city relocation is the main double-counting risk against the cohort-progression model, this is where it pays off.

A second-order consequence worth noting: high mix contrast means the building is **atypical for its area**, so the area's population characteristics are less informative about who will move in. This is the same logic as the intensity interaction in Section 8, arriving from a different direction.

# 7. Transformation functions

Every function the generator needs, with its formula, its constants, and why it takes that form. Sampling order is F1 → F17 for inputs, then F18 → F26 for the model. Use a single random generator seeded once and passed into every function; seeding inside a function silently destroys reproducibility.

## F1 — Area baselines

For each of 180 statistical areas nested in 9 districts:

$$\mathrm{SES}_a = \mathrm{SES}_{d(a)} + \varepsilon,\quad \mathrm{SES}_d \sim \mathcal{N}(0,1),\quad \varepsilon \sim \mathcal{N}(0, 0.55^2)$$

**In words:** an area's socio-economic level is its district's level plus a local deviation. The district value is drawn once and shared by all its areas — that is what creates spatial correlation, so neighbouring areas resemble each other.

$$\bar{H}_a \sim \mathcal{N}(2.55,\ 0.30^2) \in [1.8, 4.2], \qquad \mathrm{age}_a \sim \mathcal{N}(36.5,\ 4.0^2) \in [26, 50]$$

$$\mathrm{stock}_a(0) \sim \mathrm{Unif}\{900, 6500\}, \qquad \mathrm{pop}_a(0) \sim \mathrm{Unif}\{2500, 9000\}\ \text{rescaled to } 468{,}000$$

$$\mathrm{gentr}_a = \mathrm{gentr}_{d(a)} + \mathcal{N}(0, 0.008^2), \qquad \mathrm{gentr}_d \sim \mathrm{Unif}(0,\ 0.045)$$

The district layer is drawn once and shared, which is what creates spatial correlation. Coordinates in kilometres over roughly a 9 × 12 km box — the SPILL radius is 1 km and units are the one place a mistake produces plausible-looking nonsense.

## F2 — Built character of the area

Each area draws **once** a value c_a ~ N(0, 1) expressing a persistent leaning toward high-rise construction, and it enters the high-rise probability alongside the time trend:

$$\Pr(\text{high-rise} \mid s, a) = \sigma\big(-1.50 + 0.11 s + 1.2\, c_a\big)$$

**In words:** the chance a building is high-rise rises over the years (0.11 per year) and is additionally shifted permanently up or down by the character of its area. On its own c_a is just a standard normal draw; what makes it matter is that it is **not redrawn per building** — it is shared by everything built in that area across the whole period.

| Year | Low-rise area (c = −1) | Average (c = 0) | High-rise area (c = +1) |
|---|---|---|---|
| 2000 | 6% | 18% | 43% |
| 2011 | 18% | 43% | 71% |
| 2022 | 43% | 72% | 89% |

Without c_a, form is drawn from the year alone and a low-rise district sprouts towers at random. In reality planning regimes, plot geometry, conservation status and height limits operate at neighbourhood level and change slowly. It also gives mix contrast (Section 6) something to be measured against: an area needs a stable character for a building to depart from it.

Floors follow: high-rise → 12 + Poisson(10), low-rise → 4 + Poisson(3), clipped to [3, 45]. **Floors are the primitive and form is derived from them** at a 10-storey threshold, which settles the "mixed" question — mixed is a property of a project containing buildings on both sides, never a third building-level category.

## F3 — Area-conditional room prior

$$\pi_k(a) \;\propto\; \pi^0_k \exp\!\Big[\big(\gamma_1 z_{\mathrm{SES},a} + \gamma_2 z_{\bar H, a}\big)\cdot\tfrac{k-4}{2}\Big], \qquad k \in \{2,3,4,5,6\}$$

**In words:** start from a city-wide baseline room mix and tilt it toward the area. The factor (k−4)/2 is positive for large apartments and negative for small ones, so an area with high SES or large households gets more weight on 5–6 rooms and less on 2–3. The tilt is centred on k = 4, so a neutral area reproduces the baseline exactly.

$$\pi^0 = (0.10,\ 0.28,\ 0.34,\ 0.20,\ 0.08), \qquad \gamma_1 = 0.25,\quad \gamma_2 = 0.45$$

**Why this exists.** The prior over room counts is not city-wide. An area with large households contains more large apartments before anything else is known. The tilt is centred on k = 4 so that a neutral area reproduces π⁰ exactly. Household size gets the larger weight because it bears more directly on room demand than income does. Across the range of z_H̄ this shifts mean room count by roughly 0.8 rooms, which is the right order of magnitude.

## F4 — Floor area given room count

$$A \mid k,\ a,\ b \ \sim\ \mathcal{N}\big(\mu_k,\ \sigma_A^2\big) \in [22, 260]$$

$$\mu_k = \alpha_0 + \alpha_1 k + \alpha_2 z_{\mathrm{SES}} + \alpha_3\,\mathbb{1}\{\text{high-rise}\} + \alpha_4 s + \alpha_5 z_{\bar H}$$

**In words:** apartment area is drawn around a mean that depends on its room count and its context. Each extra room adds about 22 m²; a wealthy area adds space (more generous rooms for the same count); high-rise construction subtracts (compact layouts); time subtracts (apartments shrank as prices rose); and **an area with large households subtracts** — more rooms are carved from the same envelope, so a 4-room flat there is physically smaller. The 9 m² standard deviation is exactly the overlap that makes 90 m² ambiguous between 3 large rooms and 4 small ones.

$$\alpha_0 = 8,\quad \alpha_1 = 22,\quad \alpha_2 = 4,\quad \alpha_3 = -3,\quad \alpha_4 = -0.15,\quad \alpha_5 = -2.5,\quad \sigma_A = 9$$

At neutral values: 3 rooms ≈ 74 m², 4 rooms ≈ 96 m², 5 rooms ≈ 118 m².

**Why each term.** α₂ > 0: wealthier areas build more generous rooms for the same count. **α₅ < 0**: areas with large households carve more rooms out of the same envelope, so a 4-room apartment there is physically smaller — this is the term that was missing, and it is the one that makes "rooms per m²" genuinely area-dependent. α₃ < 0: high-rise floor plates favour compact layouts. α₄ < 0: apartments shrank over the period as prices rose.

**α₂ and α₅ are the load-bearing terms.** Without them, proxy error is harmless noise. With them, it is systematic bias correlated with area characteristics — the realistic case, and the thing worth measuring.

## F5 — Room count given floor-area category (the inversion)

As specified in Section 5.2, using π_k(a) from F3 and μ_k from F4. This is the function a downstream consumer calls; the generator itself never needs it.

## F6 — Project size and building count

$$\text{target}_p \sim \mathrm{clip}\big(\mathrm{LogNormal}(\log 450,\ 0.75^2),\ 100,\ 2500\big)$$

$$n^{\mathrm{bld}}_p = \mathrm{clip}\Big(\mathrm{round}\big(\text{target}_p / m_z\big),\ n^{\min}_z,\ 14\Big)$$

with the average building size m_z and the minimum count both depending on zoning:

| Zoning | Units per building m_z | Minimum buildings |
|---|---|---|
| New neighbourhood | 70 | 4 |
| Clearance-and-rebuild | 85 | 2 |
| TAMA 38/2 | 60 | 1 |
| TAMA 38/1 | 40 | 1 |

**Why derive the count from the size.** Drawing buildings first and letting their sizes accumulate produces projects outside the 100–2,500 range, and the violation only surfaces at the acceptance check.

**Why the count depends on zoning.** Multi-building projects are the norm, not the exception, but the exceptions are systematic rather than random. TAMA 38/1 is reinforcement of a single existing building and is genuinely single-building. TAMA 38/2 is demolition and reconstruction of one or two. Clearance-and-rebuild covers a block. A new neighbourhood is several buildings by definition. A uniform draw over 1–14 would put single-building new neighbourhoods and fourteen-building 38/1 projects into the data, neither of which exists.

## F7 — Zoning probability

$$\Pr(\text{new neighbourhood} \mid s) = \sigma(1.20 - 0.13 s)$$

Falling from about 0.77 in 2000 to 0.16 in 2022. Renewal projects split (0.35, 0.40, 0.25) across clearance-and-rebuild, TAMA 38/2 and TAMA 38/1.

## F8 — High-rise probability

$$\Pr(\text{high-rise} \mid s, a) = \sigma\big(-1.50 + 0.11 s + \kappa\, c_a\big), \qquad \kappa = 1.2$$

The period term reproduces the city-wide shift from low-rise to high-rise; the c_a term from F2 makes it persistent within an area. Floors then follow:

$$F_b = \begin{cases} 12 + \mathrm{Poisson}(10) & \text{high-rise} \\ 4 + \mathrm{Poisson}(3) & \text{otherwise}\end{cases}, \qquad F_b \in [3, 45], \qquad \text{high-rise} = \mathbb{1}\{F_b > 10\}$$

**Floors are the primitive and form is derived from them.** This settles the "mixed" question: at building level there is no mixed category, only above or below ten floors. Mixed is a property of a *project* containing buildings on both sides. Treating it as a third building-level category creates a phantom category and wastes a parameter.

## F9 — Floors and mix to unit count

$$P_b = 450\,e^{\varepsilon},\ \varepsilon\sim\mathcal{N}(0, 0.18^2), \qquad U_b^{\mathrm{plan}} = \mathrm{clip}\left(\left\lceil \frac{F_b P_b}{\bar{A}_b} \right\rceil,\ 8,\ 400\right)$$

**In words:** unit count is total building floor area — storeys times floor plate — divided by mean apartment size. The true plate is not fixed; it varies lognormally around 450 m².

The observable inversion available to a predictor is F̂ = U_plan · Ā / 450.

**The lognormal plate is load-bearing.** If 450 is treated as a fixed constant during generation, the inversion becomes exact and the simulator reports near-perfect recovery of building form — a falsely optimistic result that propagates into everything downstream.

## F10 — Realisation rate

$$R_b = 0.7 + 0.5\,B_b,\quad B_b \sim \mathrm{Beta}(0.6\nu_b,\ 0.4\nu_b),\quad \nu_b = 4 + 0.05\,U_b^{\mathrm{plan}}$$

**In words:** the realisation rate runs between 0.7 and 1.2 with mean 1, and ν controls how tightly it clings to 1. ν grows with building size, so large buildings are built almost exactly as planned while small ones vary widely — which is why small buildings should be harder to predict.

with E[R] = 1 by construction, and with probability 0.03 when U_plan < 150, R ~ Unif(1.5, 2.0) to generate outliers.

**Why ν grows with size.** Large buildings converge to R = 1 while small ones carry wide variance. That is the realistic pattern and the reason small buildings should be harder to predict.

## F11 — Lag, fill duration, and exposure

$$L_b \sim 2 + 2\,\mathrm{Beta}(2,2), \qquad D^{\mathrm{fill}}_b \sim 0.5 + 2\,\mathrm{Beta}(1.5, 3.5)$$

$$\hat{L}_b = 2.0 + 0.9\,\mathbb{1}\{\text{high-rise}\} + 0.002\,U^{\mathrm{plan}}_b, \qquad U_b^{\mathrm{new}} = U_b^{\mathrm{plan}}\,R_b\,(1 - r_{\mathrm{ret}})$$

L and D_fill are latent; L̂ is the visible substitute. **Returning residents reduce exposure** rather than forming a second sub-population with its own coefficients — this is what prevents double counting against the cohort-progression model, and it removes an entire parameter set.

$$r_{\mathrm{ret}} = (0.00,\ 0.25,\ 0.55,\ 0.85), \qquad s_{\mathrm{net}} = (1.00,\ 0.65,\ 0.45,\ 0.15)$$

across new neighbourhood, clearance-and-rebuild, TAMA 38/2 and TAMA 38/1.

## F12 — Mix contrast

$$\Delta k_b \;=\; \bar{k}_b - \bar{k}_{a(b)}$$

where the building's mean room count comes from its estimated room shares and the area's from its existing housing profile. Standardise before use. See Section 6 for the mechanism; it enters F19, F22 and F26.

## F13 — Population trend and occupancy jumps

$$x_a(t) = \text{baseline}_a + \text{trend}\cdot s + \sum_{b'\in a}\Delta x\,\mathbb{1}\{t^{\mathrm{occ}}_{b'} \le t\}$$

**In words:** an area's value in a given year is its baseline, plus slow annual drift, plus the accumulated jumps from every building already occupied by that year. The indicator ensures a building contributes only from its occupancy year onward.

Per-year trends: household size −0.011, median age +0.05, SES +gentr_a, state share −0.002. Occupancy jumps with frac = U_new / stock_a(0):

$$\Delta \bar{H} = -0.55\,\mathrm{frac}, \quad \Delta \mathrm{SES} = +2.20\,\mathrm{frac}, \quad \Delta \mathrm{age} = -9.00\,\mathrm{frac}, \quad \Delta \mathrm{pop} = +2.4\,U^{\mathrm{new}}$$

New construction brings younger, smaller, better-off households, so the three jumps move in the directions shown. The trends carry the slower background drift.

## F14 — State-education share

$$\mathrm{state}_a \sim \begin{cases} \mathrm{Beta}(8.0,\ 2.5) & \text{prob. } 0.85 \quad (\text{mean} \approx 0.762)\\ \mathrm{Beta}(3.0,\ 5.5) & \text{prob. } 0.15 \quad (\text{mean} \approx 0.353)\end{cases}$$

clipped to [0.08, 0.99] and rescaled to a mean of exactly 0.70. Expect median ≈ 0.75 and 10th percentile ≈ 0.40.

**Why a mixture.** A city mean of 0.70 does not come from a symmetric spread around 0.70. Most areas sit near 0.80 and a concentrated minority far lower, because non-state schooling is spatially clustered. A single Beta fitted to the mean would produce a variable with the right average and no signal — it would erase exactly the clustering that makes this a usable sector proxy. If the realised median comes out close to the mean, the mixture is not working.

## F15 — Stock, intensity and SPILL

$$\mathrm{stock}_a(t) = \mathrm{stock}_a(0) + \sum_{b'\in a} U^{\mathrm{new}}_{b'}\mathbb{1}\{t^{\mathrm{occ}}_{b'} \le t\}, \qquad \mathrm{intensity}_b = \frac{1000\,U_b^{\mathrm{new}}}{\mathrm{stock}_{a(b)}(t_0^b)}$$

**In words:** intensity is how many new units the building adds per thousand existing units in its area, at its own start year. 400 units in an area holding 8,000 is marginal; the same 400 in an area holding 900 is population replacement.

$$\mathrm{SPILL}_b = \frac{1000}{\mathrm{stock}_{a(b)}(t_0^b)}\sum_{b'\neq b} U^{\mathrm{new}}_{b'}\,\mathbb{1}\{d(b,b')\le 1\ \mathrm{km}\}\,\mathbb{1}\{t_0^b-5 \le t^{\mathrm{occ}}_{b'} < t_0^b\}$$

**In words:** the units occupied within one kilometre in the five years before this building broke ground, normalised per thousand existing units. The two indicators filter by distance and by time window. It is a plain sum over the same building table — no spatial model, no new parameters.

SPILL solves three problems in one sum with no new parameters: phased projects, two projects in one area, and infrastructure lagging construction. It also makes a separate project-size feature unnecessary, since a building in a large project sees its own siblings inside the radius and window.

## F16 — Institutions in the surroundings

$$m_b \sim \mathrm{Poisson}(\lambda_b), \qquad \lambda_b = 0.9 + 0.0016\,\mathrm{stock}_a(t_0) + 0.05\,s$$

$$\Pr(\text{school exists}) = \sigma\big(-0.4 + 0.0004\,\mathrm{stock}_a(t_0) - 1.3\,\mathbb{1}\{\text{new neighbourhood}\}\big)$$

Given a uniform draw u: status is *existing* if u < p, *planned* if u < p + 0.25, otherwise *none*. Both institution counts grow with the area's stock and with time, which reproduces infrastructure lagging construction.

## F17 — Load

$$\mathrm{load}_b = \mathbb{1}\{\text{existing}\}\cdot\mathrm{clip}\big(\mathcal{N}(0.92,\ 0.16^2),\ 0.45,\ 1.6\big)$$

**In words:** load is the school utilisation ratio, but **only where a school exists**; otherwise zero. It is stored pre-multiplied as a single column, because utilisation is undefined without a school and storing a raw value with zeros or nulls contaminates the coefficient.

Stored pre-multiplied as one column. A utilisation ratio is undefined when no school exists, and storing raw utilisation with zeros or nulls for "no school" contaminates the coefficient.

## F18 — Saturation and quadratic transforms

$$g_{\mathrm{daycare}}(m) = 1 - e^{-m/3}, \qquad g_{\mathrm{SES}}(z) = 0.05z - 0.07z^2$$

The saturation maps 0 → 0, 1 → 0.28, 3 → 0.63, 6 → 0.86. Going from zero daycares to one changes the picture; eight to nine does not. On a linear scale an area with twelve daycares dominates the estimate and drags the coefficient down for everyone. This is a fixed transform, not an estimated parameter.

The quadratic encodes an inverted U: fertility is high at both ends of the socio-economic scale and low in the middle. A linear term alone would arbitrarily favour one end and bias the other systematically.

## F19 — Interactions

$$\mathrm{I}_1 = +0.09\, z_{5+} z_{\bar H}, \quad \mathrm{I}_2 = +0.07\,\mathbb{1}\{\text{high-rise}\} z_{5+}, \quad \mathrm{I}_3 = -0.06\, z_{\mathrm{SPILL}} z_{\mathrm{load}}, \quad \mathrm{I}_4 = -0.05\, z_{\mathrm{int}} z_{\bar H}$$

Rationale for each is in Section 8.

## F20 — Standardisation

$$z(v) = (v - \bar v)/s_v$$

Applied to every continuous predictor at the point of use, not in the stored table. Storing z-scores makes the tables uninterpretable and couples them to the sample.

## F21 — Stage-1 mean and dispersion

$$\log\mu_b = \log U_b^{\mathrm{new}} + \beta_0 + \mathbf{x}_b^\top\boldsymbol\beta + \textstyle\sum_j \mathrm{I}_j + u_{p(b)} + w_{a(b)}$$

**In words:** the log mean has four parts. `log U_new` is an **offset**, not an estimated coefficient — it forces the model to learn children **per unit** rather than per building, which is what makes a coefficient comparable across a 20-unit walk-up and a 300-unit tower. Then an intercept and the feature coefficients, then the interactions, then two random effects: u_p at project level, absorbing traits shared by buildings from one developer under one approval (finish, marketing, target buyer), and w_a at area level, absorbing everything about a neighbourhood the four measured population variables miss.

$$u_p \sim \mathcal{N}(0,\ 0.10^2),\quad w_a \sim \mathcal{N}(0,\ 0.13^2),\quad \phi_b = \min(2 + 0.06\,U_b^{\mathrm{new}},\ 25)$$

u_p is a **project random effect**: buildings raised by one developer under a single approval share unobserved traits — finish level, marketing, target buyer — and ignoring that would understate uncertainty for multi-building projects. w_a is an **area random effect** absorbing everything about a neighbourhood that the four measured population variables miss. The offset log(U_new) forces the model to learn children *per unit* rather than per building, which is what makes coefficients comparable across a 20-unit walk-up and a 300-unit tower.

$$\lambda_b \sim \mathrm{Gamma}(\phi_b,\ \phi_b/\mu_b), \qquad N_b \sim \mathrm{Poisson}(\lambda_b)$$

**In words:** φ is the dispersion — the larger it is, the closer the building sits to its expectation. It rises with size, so small buildings genuinely deviate more. Drawing in two steps (Gamma then Poisson) is how a negative binomial is produced: each building has its own latent rate λ, and averaging over it creates the excess variance.

φ rising with size is what makes small buildings noisier; a fixed dispersion would make every building equally predictable in relative terms, which is wrong.

| Term | β | | Term | β |
|---|---|---|---|---|
| Intercept | −0.79 | | Daycares, after F18 | +0.06 |
| Share 3-room | +0.14 | | School existing / planned | +0.10 / +0.04 |
| Share 4-room | +0.21 | | Load | −0.07 |
| Share 5+-room | +0.26 | | SPILL | +0.05 |
| Mix contrast | +0.04 | | Mean household size | +0.30 |
| High-rise | −0.13 | | SES, linear | +0.05 |
| Project intensity | +0.05 | | SES, quadratic | −0.07 |
| New neighbourhood | +0.10 | | Median age | −0.05 |
| TAMA 38/2 / 38/1 | −0.05 / −0.25 | | State-education share | −0.20 |

exp(−0.79) = 0.45 is the baseline child coefficient: children aged 3–18 per new unit in a low-rise clearance-and-rebuild project with an average mix. Clearance-and-rebuild is the reference zoning and "no school" the reference status. Room shares are compositional; the three shown plus an omitted "2 rooms or fewer" baseline sum to 1, and including all four makes the design matrix singular.

## F22 — The composition index

$$\theta_b = \mathbf{z}_b^\top\boldsymbol\delta, \qquad \eta_{\mathrm{kg}} = \gamma_{\mathrm{kg}} + a\theta_b, \qquad \eta_{\mathrm{pr}} \equiv 0, \qquad \eta_{\mathrm{sec}} = \gamma_{\mathrm{sec}} - c\theta_b$$

**In words:** instead of a full multinomial logit (28 coefficients), every feature funnels into a **single number** θ meaning "how young is the incoming household". Positive θ raises kindergarten and lowers secondary simultaneously, with primary as the reference fixed at zero. The 1.2 exceeds 1.0 because secondary responds more sharply. Beyond saving parameters, this structure **enforces the correct monotone ordering**, which a free logit could violate while still fitting.

with a = 1.0 and c = 1.2, secondary being the more responsive end.

| Component of θ | δ | | Component of θ | δ |
|---|---|---|---|---|
| Share 3-room | +0.38 | | Median age | −0.28 |
| Share 4-room | +0.10 | | Mean household size | −0.20 |
| Share 5+-room | −0.28 | | Mix contrast | −0.22 |
| New neighbourhood | +0.35 | | SES | −0.10 |
| Daycares, after F18 | +0.15 | | State-education share | −0.12 |

**Why a single index rather than a full multinomial logit.** Most features act on composition through one dimension — how young the incoming household is. A single index costs about a third of the parameters and, more importantly, **enforces the correct monotone ordering**: anything making a building younger must move mass from secondary toward kindergarten, with primary in between. An unconstrained logit can violate that while still fitting the data.

**Mix contrast is negative here** because move-up families have older children by construction — they trade up *because* the household grew. This is distinct from the absolute share of large apartments, which is already in the index separately.

Two effects are **not** monotone on this axis and take a free coefficient in one cohort only:

$$\eta_{\mathrm{kg}} \mathrel{+}= 0.18\cdot\mathbb{1}\{\text{no school}\}, \qquad \eta_{\mathrm{sec}} \mathrel{+}= 0.45\cdot z_{\bar H}$$

The second captures the part of household size that is not youth: large families hold children in all three cohorts at once, so they raise secondary even when the index points young.

## F23 — Intercept calibration

$$\gamma^{(i+1)}_k = \gamma^{(i)}_k + 0.7\left[\log\frac{p_{0k}}{p_{0,\mathrm{pr}}} - \log\frac{\overline{p}_k^{(i)}}{\overline{p}_{\mathrm{pr}}^{(i)}}\right]$$

Iterate about 60 times from a start of log(p₀ₖ / p₀,pr), then assert the mean composition is within 0.02 of p₀.

**Why this is necessary.** Softmax convexity pushes mass toward the extremes and compresses the middle category, so the intercepts are *not* the log-ratios of the target. Skipping this pulls the middle cohort down by roughly a quarter of its mass, with no error message.

$$\mathbf{p}_0 = (0.42,\ 0.36,\ 0.22)$$

far younger than the stationary profile implied by w = (4, 6, 6), which would be (0.25, 0.375, 0.375). That is deliberate: p₀ is calibrated to a building at age three, and this is where the maturation coefficients are absorbed.

## F24 — Concentration

$$\log\tau_b = 2.5 + 0.5 z_{\mathrm{HHI}} + 0.3\log(U_b^{\mathrm{new}}/50) - 0.4 z_{D^{\mathrm{fill}}}, \qquad \tau \in [4, 80]$$

$$\mathrm{HHI}_b = \sum_{j} \mathrm{share}_{bj}^2 \quad \text{over the four floor-area categories}$$

**HHI belongs here, not in the mean.** It barely affects how many children there are; it affects how *concentrated* the composition is. A building entirely of 3-room units draws a homogeneous population with a predictable split; one mixing 2, 4 and 6 rooms draws three household types and is volatile. Putting HHI in the mean is the natural mistake and yields a near-zero coefficient plus a badly calibrated variance.

## F25 — The composition draw

$$\mathbf{q}_b \sim \mathrm{Dirichlet}(\tau_b\mathbf{p}_b), \qquad \mathbf{n}_b \sim \mathrm{Multinomial}(N_b,\ \mathbf{q}_b)$$

**In words:** p is the expected composition, q is the composition this particular building actually drew, and n is the resulting count in each of the three cohorts. τ governs how far q can stray from p.

## F26 — Output decomposition

$$\rho_b = \mathrm{clip}\big(0.60 + 0.10\, z_{\Delta k, b},\ 0.35,\ 0.85\big)$$

**In words:** the intra-city relocation share — of the arrivals who are not returning residents, what fraction came from elsewhere in the city rather than from outside it. It starts at 0.60 and rises with mix contrast, because a building coarser than its surroundings attracts move-up buyers making a short move.

$$\mathbf{n}^{\mathrm{ret}} = \mathbf{n}\,r_{\mathrm{ret}}, \qquad \mathbf{n}^{\mathrm{intra}} = \mathbf{n}(1-r_{\mathrm{ret}})\rho_b, \qquad \mathbf{n}^{\mathrm{outside}} = \mathbf{n}(1-r_{\mathrm{ret}})(1-\rho_b)$$

$$\mathbf{n}^{\mathrm{net}} = \mathbf{n}\,s_{\mathrm{net}}, \qquad \mathbf{n}^{\mathrm{registered}}_k = \mathbf{n}_k(1-\ell_k), \quad \boldsymbol\ell = (0.02, 0.05, 0.15)$$

**Why ρ varies rather than being constant.** The main double-counting risk against the cohort-progression model is intra-city relocation, and it is not uniform across buildings. A building with high mix contrast attracts move-up buyers making a short move, so more of its inflow is already counted elsewhere in the city. Holding ρ fixed at 0.60 assigns the same double-counting correction to a 38/1 reinforcement and to a new-neighbourhood tower, which is clearly wrong.

The registered-pupil line applies only if the target is registered pupils rather than resident children; leakage to boarding, private and out-of-city schooling is largest in the secondary cohort.

## F27 — Interpolation (optional, and only to emulate a real constraint)

**This function is not part of the core generator and can be omitted.** Since the entire population panel is synthetic, the true series is known at every year and no interpolation is needed to produce the data.

Its purpose is to **emulate a limitation that will exist when real data replaces the synthetic panel**. CBS publishes area-level demographics annually only from about 2017; before that, values exist at census anchors (1995, 2008, 2022) and must be reconstructed for the years between. "Construction-aware" means the reconstruction uses the record of *when units were actually occupied* — from the building table here, and from building-completion records in the real system — to decide where within the interval the change occurred, rather than spreading it evenly.

$$\hat{x}_a(t) = x_a(c_0) + \big[x_a(c_1)-x_a(c_0)\big]\left[(1-\theta)\frac{t-c_0}{c_1-c_0} + \theta\,\frac{G_a(t)}{G_a(c_1)}\right], \qquad \theta = 0.50$$

where G(t) is cumulative units occupied since c₀. θ = 0 is plain linear interpolation; θ = 1 attributes all change to construction timing.

**If you include it**, mask every year that is neither an anchor nor 2017-or-later, reconstruct with the formula, and keep both the true and reconstructed series. The gap between them measures a real bias — the change linear interpolation misses is itself caused by construction, which is the explanatory variable of interest, so the error correlates with it rather than being noise. **If you omit it**, say so when reporting, because the simulator will then understate error for buildings from 2000–2016.

# 8. Interaction rationale

**I1 — large apartments × household size.** Large apartments produce children only where a population exists to fill them. In an area of small households a 5-room flat is bought as a space upgrade, not to raise a family. Without the interaction the model assigns large apartments a uniform effect that does not exist in the field.

**I2 — high-rise × large apartments.** The negative high-rise coefficient rests on high-rises being built with small units. A high-rise of 5-room apartments is a different animal; without this the model penalises it twice for one underlying reason, once through the mix and once through the form.

**I3 — SPILL × load.** Rapid absorption **combined with** full school utilisation produces deterrence; neither alone does. A crowded area with no recent construction wave has already reached equilibrium and its families are placed.

**I4 — intensity × household size.** When a project is large relative to its area, the incoming population *replaces* the existing one rather than joining it, so the predictive power of existing-population features decays. Conceptually the most important interaction, and it also defines when to fall back to district-level shrinkage. Note that mix contrast (Section 6) reaches the same conclusion by a different route: a building atypical for its area is less well described by that area's population.

# 9. Feature contribution by cohort

The effect on a single cohort is the sum of two channels — the effect on the **total** and the effect on the **composition**:

$$\frac{\partial \log \mathbb{E}[n_k]}{\partial z_j} \;=\; \beta_j \;+\; \frac{\partial \log p_k}{\partial \theta}\cdot\delta_j$$

**In words:** a feature can raise the total number of children while simultaneously shifting the composition away from a particular cohort, in which case the two channels cancel. That is why the table below reports the **total** effect rather than the stage-1 coefficient alone.

At p₀ = (0.42, 0.36, 0.22) with a = 1.0 and c = 1.2, the composition sensitivities are +0.844 for kindergarten, −0.156 for primary, and −1.356 for secondary. Values below are **total** effects per standard deviation.

Composition sensitivities at p₀ are +0.844 for kindergarten, −0.156 for primary and −1.356 for secondary. The table gives the **total** effect per standard deviation.

| Feature | Kindergarten | Primary | Secondary |
|---|---|---|---|
| Mean household size | +0.032 | +0.232 | **+0.922** |
| Share 5+-room | +0.024 | +0.304 | **+0.640** |
| Share 3-room | **+0.461** | +0.081 | −0.375 |
| New neighbourhood | **+0.395** | +0.045 | −0.375 |
| Share 4-room | +0.294 | **+0.194** | +0.074 |
| Mix contrast | −0.146 | +0.074 | +0.338 |
| Median age | −0.286 | −0.006 | +0.330 |
| Daycares | +0.187 | 0 | −0.143 |
| No primary school | +0.104 | −0.076 | −0.076 |
| School existing | +0.100 | +0.100 | +0.100 |
| SPILL, intensity | +0.050 | +0.050 | +0.050 |
| Socio-economic index | −0.034 | +0.066 | +0.186 |
| Load | −0.070 | −0.070 | −0.070 |
| High-rise | −0.130 | −0.130 | −0.130 |
| TAMA 38/1 | −0.250 | −0.250 | −0.250 |
| State-education share | −0.301 | −0.181 | −0.037 |

Three readings stand out. **Share of 5+-room apartments contributes almost nothing to kindergarten** (+0.024) — large apartments do not produce kindergarten children, because the effect on the total is cancelled by the shift in composition. **Household size is the strongest effect in the model for secondary** (+0.922) yet negligible for kindergarten, the same cancellation running the other way. **Share of 4-room apartments is the only distinctive signal primary has**, which is why it must not be merged with the 3-room share.

Secondary is the hardest cohort to predict: it has no direct signals and leans on household size, large apartments and mix contrast. Primary is the reference cohort, so most of its effects are moderate and move through the total — a structural property of the parameterisation, not a finding about the world.

# 10. Table structure

Four tables plus a lookup. **Latent variables live inside the relevant table**, clearly marked, so validation is a column comparison rather than a join. Keep one explicit list of model-visible columns so nothing leaks by accident.

**Building** — keys and geometry; planned units; construction start and expected lag; floor-area category shares; estimated 3 / 4 / 5+ room shares; HHI; estimated form; intensity; **mix contrast**. Latent: true floor count, true room counts, true plate, realisation rate, true lag and fill duration, hidden 0–2 cohort.

**Project** — keys and building list; zoning; derived returning-resident and net-new-unit shares; aggregates. Latent: none beyond the derived relocation share.

**Environment** — keyed on building and its t₀: daycare count, school status, load, SPILL, contemporaneous stock.

**Population** — (area, year) panel: household size, socio-economic index, median age, state-education share, each in observed and true form; stock and population as denominators, **not** features.

**Areas lookup** — area to district, centroid, baselines, **built character c_a**, and the area's existing housing profile used for mix contrast.

The district layer exists for three distinct reasons that are easy to conflate: it is a real level in the generating hierarchy, the target of partial pooling during estimation, and the fallback when a value is missing or boundaries changed. The first two agree by construction, which is the point.

# 11. Assumptions register

**Published** means taken from a source and citable. **Calibrated** means chosen to produce plausible behaviour and replaceable with a measured value. **Structural** means a modelling decision rather than a number.

| Assumption | Value | Status | Risk if wrong |
|---|---|---|---|
| City population | 468,000 | Published | Low — scaling only |
| Mean household size | 2.3 | Published | Low |
| State-education share, city | 0.70 | Published | Medium |
| Households with children ≤17 | 52,700 | Published | Low — cross-check |
| SES index is standardised, not 1–10 | mean 0, SD 1 | Published | **High** — wrong scale breaks every SES term |
| Baseline child coefficient | 0.45 per new unit | Calibrated | **High** — scales every output |
| Composition at age three | (0.42, 0.36, 0.22) | Calibrated | **High** — sets the cohort split |
| Area-to-rooms map | α = (8, 22, 4, −3, −0.15, −2.5), σ = 9 | Calibrated | **High** — the central proxy |
| Room-prior tilt | γ = (0.25, 0.45) | Calibrated | Medium |
| Built-character weight κ | 1.2 | Calibrated | Medium — controls spatial persistence of form |
| Standard floor plate | 450 m² | Calibrated | Medium — biases estimated floors and form |
| Units per building by zoning | 40–85 | Calibrated | Medium |
| Realisation rate | mean 1, Beta-shaped | Calibrated | Medium |
| Fill duration | 0.5 + 2·Beta(1.5, 3.5) | Calibrated | Low, given measurement at three years |
| Returning-resident shares | 0 / 0.25 / 0.55 / 0.85 | Calibrated | **High** for renewal projects |
| Relocation share base and slope | 0.60, +0.10 per SD of contrast | Calibrated | **High** for the net increment |
| Registration leakage | (0.02, 0.05, 0.15) | Calibrated | Medium, secondary only |
| Interpolation weight θ | 0.50 | Calibrated | Low — the function is optional |
| Measurement at three years | fixed | **Structural** | Changing it reinstates the occupancy curve |
| Building as the unit | fixed | **Structural** | Project-level modelling loses phasing |
| Returning residents as exposure | fixed | **Structural** | The alternative double-counts |
| Single latent composition index | fixed | **Structural** | A free logit can break monotonicity |
| Hidden 0–2 cohort | fixed | **Structural** | Omitting it drains kindergarten too fast |
| Floors as primitive, form derived | fixed | **Structural** | Otherwise "mixed" becomes a phantom category |
| All population data synthetic | forced | **Constraint** | Absolute levels are not real |

# 12. Pitfalls

Ordered roughly by how easy each is to commit and how much damage it does.

1. **Keying environment or population on the project's start date instead of the building's.** Silently destroys the phasing effect the whole design exists to capture.
2. **Feeding the 1–10 socio-economic cluster into a formula expecting the standardised index.** Inflates every SES term threefold and shifts its centre by about six standard deviations.
3. **Skipping the softmax intercept calibration.** Compresses the middle cohort by roughly a quarter of its mass, with no error message.
4. **Drawing building form from the year alone.** Produces towers scattered through low-rise districts. Use the area's built character.
5. **Treating the floor plate as a constant in generation.** Makes the floor-count inversion exact and produces a falsely optimistic view of form recovery.
6. **A city-wide room prior.** Captures that rooms are sized differently across the city but not that they are distributed differently — a half-measure.
7. **Deterministic area-to-rooms inversion.** Hides the uncertainty that is the point of the exercise.
8. **Dropping the socio-economic index or household size because their direct effects are weak.** Both do proxy correction; removing them converts a correctable bias into an uncorrectable one.
9. **Including all four room-share categories.** Singular design matrix. Omit "2 or fewer" as the baseline.
10. **Putting HHI into the mean instead of the concentration parameter.** Near-zero coefficient and badly calibrated variance.
11. **Storing raw utilisation with zeros where no school exists.** Contaminates the coefficient.
12. **A fixed housing-stock denominator.** Inflates the intensity of the second project in an area.
13. **A constant intra-city relocation share.** Assigns the same double-counting correction to a 38/1 reinforcement and a new-neighbourhood tower.
14. **A uniform draw for buildings per project.** Produces single-building new neighbourhoods and fourteen-building 38/1 projects, neither of which exists.
15. **Adding a period coefficient on top of the household-size trend.** Double counts the fertility decline.
16. **Modelling returning residents as a second sub-population.** Double counts against the cohort-progression model.
17. **Reporting simulator output as a forecast.** The absolute levels come from assumptions, not data.

# 13. Acceptance checks

The generator should assert these and fail loudly rather than warn.

| Check | Expected |
|---|---|
| Children per new unit, overall | 0.45–0.50 |
| Realised composition vs p₀ | within 0.02 per cohort |
| Project totals | every project within 100–2,500 units |
| Buildings per project | median above 1; single-building projects confined to TAMA 38 |
| Estimated vs true 4-room share | correlation roughly 0.75 |
| Estimated vs true floor count | correlation roughly 0.93 |
| Building-form classification | roughly 94% correct |
| Form persistence within area | correlation of form across buildings in the same area clearly positive |
| m² per room across SES and household-size terciles | a visible gradient on both |
| Mix contrast | roughly centred on zero, with meaningful spread |
| Intra-city share | varies across buildings, within [0.35, 0.85] |
| Coefficient recovery, NB2 with log-exposure offset | roughly 90% of coefficients within ±2 standard errors |

Coefficients for features deliberately omitted from the recovery regression will be biased. That is the intended demonstration of omitted-variable bias, not a defect.


**Two kinds of gap when reporting.** *Proxy gaps* are precisely quantifiable, because the generator holds both the true and the observed value in the same table: room counts through area categories, floor counts and form through the plate inversion, sector through the state-education share, occupancy lag through its estimate. *Missing-information gaps* are irreducible — realisation rate, fill duration, buyer identity — and they define the error floor of any model trained on this data.

**Sources for the calibration anchors.** Tel Aviv-Yafo population and household size: Statistical Abstract of Israel, CBS. Households with children: Statistical Yearbook of Tel Aviv-Yafo. Socio-economic index of statistical areas and its published scale: CBS, *Characterization and Classification of Geographical Units by the Socio-Economic Level of the Population*. Statistical-area layers with demographic attributes: cbs.gov.il/he/Pages/geo-layers.aspx. Every other numeric value in this document is a working assumption for the simulation and is not an estimate from real data.
