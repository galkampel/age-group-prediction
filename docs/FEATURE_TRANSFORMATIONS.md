# Feature Transformations: Decisions And Rationale

The transformation each feature gets in each model, what the resulting
coefficient means, and why the alternatives were rejected.
- **Quick answer:** the per-feature × per-model table is in §7.
- **Implementation:** every transformation here is implemented in
  [`feature_engineering/`](../src/age_group_prediction/feature_engineering/).
- **Declarations:** §8 gives each model's declaration, built at the call site.
- **The old models** still use the older `FeatureSpec` machinery, described in
  [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md).

| Label | Class | Stages and feature use |
|---|---|---|
| **A** | `DirectCohortModel` | One LightGBM regressor per cohort (Poisson or regression objective); raw features; optional exposure offset |
| **B** | `IndependentTotalProbabilityModel` | Penalized Poisson/NB2 total with a log-exposure offset, plus a grouped multinomial composition stage |
| **C** | `BayesianConditionalModel` | Hierarchical NB2 total plus a composition stage; uses B's frozen specs; `Normal(0, 0.5)` coefficient priors |

**Reference statistics.** All numbers below come from simulated populations
(`configs/simulation.toml`), pooling 10 or 20 runs of about 250 buildings in 60
neighborhoods each.

| Feature | Mean | SD | Range | Notes |
|---|---|---|---|---|
| `ses` | 0.07 | 0.84 | −2.5 to 2.1 | Neighborhood-level: about **60 distinct values** per run |
| `avg_household_size` | 2.62 | 0.27 | 1.8 to 3.4 | Neighborhood-level |
| `median_age` | 36.0 | 3.8 | 25 to 45 | Neighborhood-level |
| `n_daycares_500m` | 2.9 | 1.7 | 0 to 12 | Neighborhood-level; 99th percentile **8**; 6.3% zeros |
| `n_apartments` | 39.8 | 11.8 | 12 to 80 | Building-level; always ≥ 12 |
| `3_rooms_share` | 0.29 | 0.12 | 0 to 0.67 | Zero in **0.1%** of buildings |
| `4_rooms_share` | 0.38 | 0.12 | 0 to 0.72 | Zero in 0.0% |
| `5_rooms_share` | 0.22 | 0.10 | 0 to 0.58 | Zero in 0.8% |
| `6_rooms_share` | 0.11 | 0.08 | 0 to 0.39 | Zero in **9.9%** |

Moderator correlations, which matter for the interactions in Section 6:

| | `ses` | `avg_household_size` | `median_age` | `n_daycares_500m` |
|---|---|---|---|---|
| `ses` | 1.00 | −0.13 | −0.01 | 0.29 |
| `avg_household_size` | −0.13 | 1.00 | **−0.44** | 0.08 |
| `median_age` | −0.01 | −0.44 | 1.00 | −0.23 |
| `n_daycares_500m` | 0.29 | 0.08 | −0.23 | 1.00 |

---

## 1. Principles

**Order of operations:** nonlinear transform → centering and scaling →
interactions. Scaling before a nonlinear transform changes the shape of the
effect, not just its unit.

**Centering vs. scaling.** They do different things, and the difference decides
most of the choices below.

| | What it changes | What it does not change |
|---|---|---|
| Centering ($x - c$) | The intercept, and every main effect that appears inside an interaction | The slope; and in B, nothing about the fit, since the intercept is unpenalized |
| Scaling ($x/s$) | The coefficient's unit, and hence how hard B's L2 penalty and C's priors shrink it | The shape of the fitted relationship |

**Consequence:** a scale learned per fold changes the unit from fold to fold,
while a center learned per fold is harmless. This is the main argument in
Sections 4.1 and 4.2.

**Trees see only ordering.** For Model A, any strictly monotone per-feature
transform (scaling, `log1p`) leaves the splits unchanged. Only these matter:
non-monotone transforms, transforms that replace or combine columns, and
offsets, which act on the target side.

**Prefer fixed anchors over learned scales.** $\tilde x = (x-c)/s$, with $c$ and
$s$ chosen from domain knowledge, keeps units the same across folds, cannot
leak, and reads naturally ("per decade", "from no daycare to 8"). Choosing $s$
near 2 SD follows Gelman (2008): a one-unit change is then comparable to
switching a binary indicator, which is the scale the `Normal(0, 0.5)` priors
were set for.

---

## 2. Room-Share Reference: 3 Rooms

**Decision: the 3-room share becomes the omitted reference**, and the modeled
shares are `4_rooms_share`, `5_rooms_share`, `6_rooms_share`.

The reference choice does not change the fitted relationship in an unpenalized
model; it changes what each coefficient means, and it slightly changes the fit
through B's penalty and C's priors, which are not invariant to
reparameterization.

**Why 3 rooms:**

- **A reference should be well populated.** The 3-room share is zero in 0.1% of
  buildings, against 9.9% for the current 6-room reference. Contrasts against a
  category that is often absent describe a substitution that frequently cannot
  happen.
- **Ordered sizes give a ladder.** Each coefficient becomes "larger apartments
  vs. the smallest ones", and the three coefficients should increase with size,
  which is a check you can read at a glance.
- **It matches the data-generating convention.** The simulator's
  `room_log_mean_effects` are defined relative to 3 rooms, so recovered
  coefficients are directly comparable with the true values.

---

## 3. Model A (LightGBM)

### 3.1 Features and transforms

All 8 numeric columns (`ses`, `avg_household_size`, `median_age`,
`n_daycares_500m`, `n_apartments`, and the 4-, 5- and 6-room shares) plus
`school_status`.

| Transform | Effect on the trees | Decision |
|---|---|---|
| Z-scoring, min-max, fixed anchors | None: monotone, identical splits | **Not used.** Keep raw |
| One-hot `school_status` (reference `none` dropped) | Required, because LightGBM needs numeric input. `none` is the row with both dummies at 0, so isolating it takes two splits. Dropping a reference is unnecessary for trees but harmless. | **Keep.** This is the only required transform. Alternatives: an ordinal code `none`=0 < `planned`=1 < `existing`=2, or native categorical support; with 3 levels the difference is negligible |
| `log1p` daycares | None: monotone, so the partition of the training rows is identical. Only the thresholds move, which can change predictions for counts falling between observed values. | **Drop the `tree__daycare_log1p` candidate:** it can't be distinguished from the linear form |
| `ses` squared | Adds a **non-monotone** column while keeping `ses`: $(\text{ses}-c)^2$ orders buildings by their distance from $c$, so one split isolates both tails where `ses` alone needs two. Useless if $c$ lies outside the data range, since $x^2$ is then monotone | **Keep as a candidate.** Center at the mean for robustness; with this SES it barely matters (Section 3.2) |
| SES B-spline | **Removes `ses`** and replaces it with 5 overlapping basis columns ([fitted_features.py:285-290](../src/age_group_prediction/fitted_features.py#L285-L290)), so no column orders buildings by SES and a simple SES threshold has to be approximated across columns | **Rejected** |
| Interactions | Trees build them through successive splits | Not needed; not offered for `tree` |
| $\log n$ as `init_score` | Changes the target scale, not a column | **Add** (Section 3.3) |

### 3.2 SES candidates

Two, matching the GLM candidates:

- **linear:** raw `ses`;
- **quadratic:** $[\,\text{ses}-m,\ (\text{ses}-m)^2\,]$, equivalent to
  `PolynomialFeatures(degree=2, include_bias=False)` on centered SES.

**Why centering, and how much it matters.** A tree gains nothing from a
monotone column, so the squared column earns its place only by being
non-monotone: $(\text{ses}-c)^2$ ranks buildings by **distance from $c$**, and
one split on it isolates both tails at once, where `ses` alone would need two
splits in separate branches. The center $c$ therefore defines what the column
means.

- If $c$ lies inside the data range, the column reads as "how far from $c$".
- If $c$ lies outside it, $x^2$ is monotone over the observed values and the
  trees can extract nothing from it. Squaring `median_age` (25–45) raw would be
  the clear example.
- **For this SES it is nearly a no-op:** SES is simulated as $N(0,1)$ with mean
  0.07, so 0 already sits mid-range. Centering moves the vertex by 0.07 SD.

So centering here is a robustness choice for real SES data, whose zero point may
be arbitrary or outside the observed range, not a correction of a present
problem. If a substantive turning point is hypothesized, that value is a better
$c$ than the mean.

This is unlike the GLM case (Section 4.2), where centering before squaring
changes what the coefficients mean: it makes $\beta_1$ the slope at average SES
and decorrelates $z$ from $z^2$. Neither concern applies to trees.

### 3.3 Exposure: add the offset **and** keep `n_apartments`

For the Poisson objective, `DirectCohortModel(use_exposure=True)` passes
$\log n_i + b_c$ as LightGBM's `init_score`:

$$
\log \mu_{i,c} = \log n_i + b_c + f_c(\mathbf{x}_i),
\qquad b_c = \log \frac{\sum_i y_{i,c}}{\sum_i n_i},
$$

so each cohort's trees learn a **per-apartment rate**.
- **Why $b_c$:** it is the intercept, the average log rate per apartment.
  LightGBM skips its `boost_from_average` once given an `init_score`, so
  without $b_c$ the trees would start at one child per apartment.
- **Why the model adds the offset itself:** LightGBM's `predict` doesn't add
  the `init_score` back, so the model returns
  $\exp(\text{raw score} + \log n_i + b_c)$.
- **Details:** the equations and the correct inputs are in
  [DIRECT_COHORT_MODEL.md §0](DIRECT_COHORT_MODEL.md#0-the-rebuilt-model-modelingdirect_cohortpy).

- **Why the offset:** child counts grow roughly in proportion to $n$. Trees
  approximate that with step functions, need many splits to do so, and cannot
  extrapolate beyond the sizes seen in training, so a 100-apartment building is
  predicted like the largest training building.
- **Why keep `n_apartments` as a feature too:** the offset asserts exact
  proportionality. Keeping the feature lets the trees learn departures from it,
  for example larger buildings having fewer children per apartment.
- **Not for `regression`:** it has no log link, so a log-scale offset does not
  apply, and the model refuses the combination. (An alternative there is to
  model $y/n$ with weights $n$.)

---

## 4. Models B and C (GLMs): Final Per-Feature Decisions

Both models share these specs; C uses B's selected specs frozen.

### 4.1 `n_daycares_500m`: two candidates

**Candidate 1: linear with fixed domain bounds.**

$$
\tilde d = \frac{d - 0}{8 - 0} = \frac{d}{8}
$$

- The bounds are domain constants, not learned: $d \ge 0$ by definition, and 8
  is the 99th percentile. Only 0.7% of buildings exceed it (maximum 12).
- **Do not clip.** Values above 1 are fine in a GLM; clipping would discard
  real variation.
- **Reading:** $\beta$ is the log-rate change from no daycares to 8, and
  $\tilde d = 0$ means "no daycare", the baseline you wanted.
- **Why not a learned min-max:** zeros occur in only about 6% of buildings, so
  a fold can contain none, and its learned minimum becomes 1, at which point 0
  no longer means "no daycare" and the baseline differs across folds. The
  learned maximum is also a single-neighborhood outlier that moves between
  folds. A fixed bound keeps the interpretation without that instability.

**Candidate 2: centered `log1p` (option ii).**

$$
\tilde d = \log\frac{1+d}{1+\bar d} = \log(1+d) - \log(1+\bar d)
$$

This is preferred over option i, $\log(1+d)/\log(1+\bar d)$:

- **The unit is fixed even when $\bar d$ is learned.** Option ii only *centers*,
  and centering does not change the slope, so every fold's coefficient is in
  the same unit. Option i *scales* by a learned quantity, so the unit changes
  from fold to fold and coefficients are no longer comparable.
- **The coefficient is an elasticity.** With a log link,
  $\mu \propto (1+d)^{\beta}$: a 1% increase in $(1+d)$ multiplies the rate by
  about $\beta$%. Use $\log_2$ instead if you prefer "per doubling of $(1+d)$".
- **The baseline is an average building**, which is the right reference when
  the term also appears in an interaction.
- **Option i is acceptable only with a fixed $\bar d$** (e.g. 3, the
  simulator's saturation scale). Then it is a fixed anchor with a 0-daycare
  baseline, and the coefficient is the effect of going from 0 to 3 daycares.

**Caveat when the daycare term is 0-anchored** (candidate 1): in
`daycare_x_median_age`, the median-age main effect then describes a building
with **no daycares**, a case covered by only 6% of buildings. The centered
candidate 2 avoids this, which is another reason to prefer it when interactions
are active.

### 4.2 `ses`: two candidates (no spline)

Both start from $z = (\text{ses} - m)/s$, with $m$ and $s$ from the fit
partition.

| Candidate | Columns | Reading |
|---|---|---|
| linear | $z$ | Log-rate change per SD of SES |
| quadratic | $z,\ z^2$ | $\beta_1$: the slope at average SES; $\beta_2$: curvature, i.e. how the slope changes away from the mean. $z^2$ is the squared distance from the mean in SD units |

**Scale once, before squaring.** Squaring the already scaled column is the
right order, and it is what the code does today
([fitted_features.py:283-284](../src/age_group_prediction/fitted_features.py#L283-L284)).
`PolynomialFeatures(degree=2, include_bias=False)` on $z$ is equivalent.
Rescaling $z^2$ afterwards would only re-center it (a shift of about 1, which
moves the intercept) and change its penalty weight, at the cost of a coefficient
you can no longer read as curvature. Squaring *before* scaling is worse still:
with uncentered SES, $\text{ses}^2$ correlates strongly with $\text{ses}$, and
the linear coefficient becomes the slope at SES = 0 in raw units.

**Why no spline candidate:**

- SES is neighborhood-level, so a run has only about **60 distinct values**. A
  spline with 5 basis columns spends 4 more parameters than the quadratic on
  those 60 points, which is where overfitting is most likely.
- Its individual coefficients cannot be interpreted; you need a
  partial-dependence plot to read the fit at all.
- It is incompatible with `ses_x_household_size`, because the spline replaces
  the linear SES column the interaction needs.
- The quadratic already answers the interpretable question: is there curvature
  or a U-shape in the SES gradient?

**The spline remains a diagnostic.** Plot partial residuals against SES for the
quadratic fit. Only if that shows a shape the quadratic cannot represent (a
plateau, or more than one turning point) is a spline worth adding. For the
record, if one is ever added: scaling SES beforehand makes no difference,
because the knots move with the data and the basis is identical, and the basis
columns should **not** be z-scored afterwards, since they lie in [0, 1] and sum
to at most 1 (a partition of unity) which scaling would destroy.

### 4.3 Room shares (4, 5, 6 rooms; reference 3 rooms)

$$
\tilde s_k = \frac{s_k - \bar s_k}{0.10}, \qquad k \in \{4,5,6\}
$$

**How to read $\beta_k$:** the change in log-rate when **10 percentage points**
of a building's apartments are $k$-room instead of 3-room, with the other
modeled shares held fixed.

Two corrections to a natural misreading:

- **A unit is 10 pp, not a 10% relative change.** A 4-room share of 38% going
  to 48% is one unit; 38% → 41.8% would be a 10% relative increase.
- **The coefficient is not "deviation from the mean".** The model is linear, so
  $\beta_k$ applies to any 10 pp shift, wherever it starts. Centering changes
  no slope; it only moves the intercept (and the main effects inside
  interactions) to a building with the **average room mix**, which is a
  building that actually exists.

**Why one common unit for all three shares.** The shares are compositional,
$s_3+s_4+s_5+s_6 = 1$, so each coefficient is a substitution effect against the
3-room reference. That reading requires the same unit everywhere: with per-share
z-scores or per-share min-max, "+1" would move a different number of apartments
in each column, and the three coefficients would no longer be comparable or
addable.

**Rejected alternatives:**

- **Raw shares:** a one-unit change means moving 100% of apartments into size
  $k$, roughly 8 SD away, and the intercept describes an all-3-room building.
- **Per-share z-score or min-max:** loses the substitution reading, as above;
  min-max additionally inherits the fold instability from Section 4.1.
- **Log-ratio transforms (ALR/ILR):** the standard compositional approach, but
  zeros occur (0.8% for 5 rooms, 9.9% for 6 rooms) and zero replacement would
  be arbitrary.

**Side note.** In the simulator a building's expected total is
$\mu_i = n_i \sum_k s_{ik} e^{\gamma_k + \dots}$, which is linear in the shares
on the *rate* scale. A log link with linear shares is therefore an
approximation, though a good one over the observed range.

### 4.4 Exposure: `log_n_apartments`

**Confirmed: `n_apartments` is not a feature in B or C.** The spec refuses an
exposure column that is also listed as an ordinary numeric feature, so building
size enters only as the offset

$$
\log \mu_i = \log n_i + \beta_0 + \mathbf{x}_i^\top\boldsymbol\beta
\quad\Longleftrightarrow\quad
\frac{\mu_i}{n_i} = \exp(\beta_0 + \mathbf{x}_i^\top\boldsymbol\beta),
$$

with the coefficient of $\log n_i$ **fixed at 1**. The linear predictor
therefore models children per apartment, and every other coefficient is a
multiplicative effect on that rate.

Used by: B's total stage and C's total stage only. The composition stages use
no exposure, because age-group shares do not depend on building size, and
Model A uses none today (Section 3.3 proposes adding one).

- **Never scale the offset.** Replacing $\log n$ by $(\log n - c)/s$ with the
  coefficient still fixed at 1 would assert $\mu \propto n^{1/s}$, a different
  model. Centering it would only shift the intercept and gains nothing.
- **`log`, not `log1p`.** `log1p` treats every building as having one extra
  apartment:
  $\log(1+n) - \log n = \log(1 + 1/n) \approx 1/n$, about **8%** at $n = 12$
  and **1%** at $n = 80$. Because the distortion depends on $n$, the intercept
  cannot absorb it and small buildings get biased rates. `log1p` is only useful
  when zeros are legitimate; here $n \ge 12$, and a building with no apartments
  has no children with certainty.
- **Optional check:** fit $\log n$ as an ordinary covariate with a free
  coefficient $\gamma$. $\hat\gamma \approx 1$ supports the offset; a clearly
  smaller value would say larger buildings have lower per-apartment rates.

### 4.5 `avg_household_size` and `median_age`: standard scaling

**Yes, z-scoring is the right default**, and it assumes nothing about the
relationship: the linear *shape* is the modeling assumption, while the scale
only sets the unit.

- It matches the units C's `Normal(0, 0.5)` priors were set for, so each
  coefficient is shrunk by a comparable amount.
- **Report natural units too**: $\beta_{\text{natural}} = \beta / s$, giving
  "per person" and "per year" alongside "per SD".
- Fixed anchors — $(x-2.6)/0.5$ persons and $(x-37)/10$ years — remain a nicer
  option for a final reported model, because the units don't move between
  folds and both are close to 2 SD. Either choice is defensible; z-scoring is
  the pragmatic default.

### 4.6 `school_status`

One-hot with the schema reference `none` dropped, unscaled. The dummies are
already interpretable ("existing vs. no school"), and scaling them would only
obscure that.

### 4.7 Why the composition stage carries no offset

The total stage takes $\log n$ as an offset; the composition stage takes none.
That is not an omission, and the second reason below makes it impossible for it
to be one.

**1. Shares are scale-free.** The stage models
$P(\text{cohort}=c \mid \text{building})$, three numbers summing to 1. Doubling
a building's size doubles all three cohort counts and leaves the shares
unchanged, so there is no scale for an offset to correct.

**2. A common offset cancels exactly in the softmax.** Adding the same
$\log n_i$ to every class's linear predictor gives

$$
P(c) = \frac{e^{\eta_c + \log n}}{\sum_k e^{\eta_k + \log n}}
     = \frac{n\,e^{\eta_c}}{n \sum_k e^{\eta_k}}
     = \frac{e^{\eta_c}}{\sum_k e^{\eta_k}} .
$$

The factor $n$ cancels between numerator and denominator. A class-constant
offset in a multinomial logit is algebraically a no-op: it cannot change a
single predicted probability, however large the building.

**3. Size does enter — through the weights, not the mean.** In
`_fit_grouped_multinomial` each building is expanded into up to three rows, one
per cohort, with `sample_weight` equal to that cohort's observed child count.
Buildings with more children therefore carry proportionally more weight in the
likelihood. That is the correct channel for exposure in a composition model: it
acts on **precision** (how much a building's observed mix is trusted), not on
the location of the mean.

**The resulting decomposition** is what makes the two-stage model coherent:

$$
\text{cohort count}_{ic} \;=\; \underbrace{\mu_i}_{\text{total stage: offset lives here}}
\times \underbrace{p_{ic}}_{\text{composition stage: no offset}}
$$

Building size belongs entirely to the first factor. `FeatureSpec` enforces this
("Only total-count features may define an exposure"), so a composition spec
carrying an exposure is refused at construction.

---

## 5. Model And Variation Catalogue

### Model A: `DirectCohortModel`

One LightGBM regressor per cohort (`n_kindergarten`, `n_elementary`,
`n_highschool`). The hyperparameters are fixed per instance and tuned from
outside, not inside `fit`.

| Variation | What changes | Question it answers |
|---|---|---|
| **Base** | All 8 numeric columns raw, one-hot `school_status`, 3-room reference | Reference point for everything below |
| SES quadratic | Adds $(\text{ses}-m)^2$, keeps `ses` | Does an explicit non-monotone column beat the splits the trees would make anyway? |
| **+ size offset** | `use_exposure=True`: $\log n + b$ as `init_score` (Poisson only); `n_apartments` stays a feature | Does modeling the per-apartment rate beat letting the trees learn size from scratch? It was expected to be the largest gain. An untuned smoke run over 10 simulated populations found only weak evidence: about 4% lower deviance for kindergarten and high school, and none for elementary ([plan, Step 2.4](MODEL_REIMPLEMENTATION_PLAN.md)) |
| Objective: Poisson / regression | Poisson log-likelihood vs squared error | Does a count likelihood beat squared error? |

### Model B: `IndependentTotalProbabilityModel`

Two independent stages, each with its own transformer and its own tuned L2
penalty. One Model B **candidate is a pair** of specs,
`component_feature_specs=(total_spec, probability_spec)`
([candidate_registry.py:198](../src/age_group_prediction/experiment/candidate_registry.py#L198)),
so the two stages are declared together even though they are fitted
separately.

#### Which columns each stage gets, and why

Both stages take the **same 7 non-exposure numeric columns plus
`school_status`**. What differs is forced, not chosen:

| | Total stage | Composition stage |
|---|---|---|
| Numeric columns | the 7 non-exposure columns | **the same 7** |
| `school_status` | yes | yes |
| Exposure | $\log n$ offset | **none** (Section 4.7) |
| Room-share interaction | `x_household_size` only | `x_median_age` only |

- **Why `n_apartments` is not a feature in either stage:** it *is* the
  exposure. `FeatureSpec` refuses a column that is both, and including it as a
  predictor would let the model partly undo the offset by re-estimating the
  size elasticity — which is the optional diagnostic of Section 4.4, not the
  default specification.
- **Why the room shares are in:** the apartment-size mix is the main
  building-level driver of how many children live per apartment, and of which
  ages they are.
- **Why `school_status` is in:** schools attract and retain families, which
  shifts both the rate and the age mix.
- **Why the composition stage keeps the same columns as the total stage:**
  every one of them is a plausible shifter of the *age mix*, not only of the
  *count*, so there is no a-priori basis for dropping any. Aligned matrices
  also keep the two stages' coefficients comparable.

#### Total stage — penalized Poisson/NB2, offset $\log n$, target `n_children_total`

| Variation | Columns | Reading | Why this stage |
|---|---|---|---|
| **Base** | $z_{\text{ses}}$, $z_{\text{hh}}$, $z_{\text{age}}$, $d/8$, 3 centered shares, 2 school dummies | Log children-per-apartment rate at an average building with no daycares | The parsimonious specification everything else is measured against |
| **SES quadratic** | + $z_{\text{ses}}^2$ | $\beta_1$ is the slope at average SES, $\beta_2$ the curvature | Fertility–SES gradients are commonly non-monotone, so one extra parameter is cheap insurance when modeling **how many** children |
| **Daycare centered log1p** | $d/8 \rightarrow \log\frac{1+d}{1+\bar d}$ | Elasticity: $\mu \propto (1+d)^\beta$ | Daycare count measures *access*, and access saturates — the first facility matters far more than the eighth |
| **`room_share_x_household_size`** | + 3 columns | How the room-mix effect shifts per SD of household size | Capacity meets demand: the child-count payoff of a larger apartment depends on how large local households are (Section 6.2, first choice) |
| `room_share_x_ses` | + 3 columns | How the room-mix effect shifts per SD of SES | Does extra space become more children, or more space per person? Section 6.2's second choice — needs the new interaction of Section 8 |
| Combined | the forms that won above | — | Only after the single-term candidates have established which ones earn their place |

#### Composition stage — grouped multinomial over the three cohorts, no exposure

Buildings are weighted by their observed child counts (Section 4.7).

| Variation | Columns | Reading | Why this stage |
|---|---|---|---|
| **Base** | The same design matrix, no offset | Log-odds of each cohort against the reference cohort, for an average building | As above |
| **Daycare centered log1p** | $d/8 \rightarrow \log\frac{1+d}{1+\bar d}$ | Elasticity on the cohort log-odds | Same saturating-access argument as the total stage — it is a property of the covariate, so it applies wherever daycare enters |
| **`room_share_x_median_age`** | + 3 columns | How the room-mix effect on the age mix shifts per SD of neighborhood age | Apartment size and neighborhood age jointly mark family lifecycle stage (Section 6.3, first choice) |
| `room_share_x_daycare` | + 3 columns | How the room-mix effect on the age mix shifts per unit of daycare access | Daycare marks young children specifically, targeting the kindergarten cohort. Section 6.3's second choice — needs the new interaction of Section 8 |
| **SES quadratic** | — | — | **Not a candidate here.** Curvature has a mechanism for *how many* children but none for *which ages*; keep SES linear |
| Combined | the forms that won above | — | As above |

#### The two asymmetries between the stages, and why they are principled

- **Daycare log1p is offered in *both* stages** because saturation is a
  property of the **covariate**, not of the outcome. If access to childcare
  saturates, it saturates whether it is shifting how many children live in a
  building or which ages they are.
- **SES quadratic is offered in the *total stage only*** because curvature
  there has a mechanism — non-monotone fertility–SES gradients — while the age
  mix is driven by family lifecycle (apartment size, neighborhood age,
  daycare) rather than by affluence. Spending a parameter on SES curvature in
  the composition stage buys a hypothesis nobody has a reason to hold.
- **Parsimony should bind harder in the composition stage** in any case: a
  three-category weighted multinomial carries less information per parameter
  than the count stage does.

### Model C: `BayesianConditionalModel`

Same two stages, and its specs **must equal** the selected Model B candidate's
frozen specs, so it has no feature variations of its own. What differs:

| Aspect | Consequence for transformations |
|---|---|
| **Feature forms are frozen to Model B's winner** | `_validate_bayesian_feature_freeze` ([selection.py:271-295](../src/age_group_prediction/experiment/selection.py#L271-L295)) raises unless the selected Bayesian candidate's `total_count` and `composition` specs **equal** the selected independent candidate's. See the note below |
| Hierarchical neighborhood effects (non-centered, `HalfNormal` scale) | Absorbs neighborhood-level variation the neighborhood-level features do not explain; the interpretation of building-level features (the room shares) is the most robust |
| `Normal(0, 0.5)` coefficient priors | The feature's unit decides how strongly each coefficient is shrunk. A unit near 2 SD, as recommended, keeps that prior sensible; a unit spanning the whole range shrinks the per-daycare effect harder |
| `Normal(-2, 1)` total-intercept prior | Assumes a standardized design where 0 is an average building. Note the observed average log rate is about $\log(23.9/39.8) \approx -0.5$, 1.5 prior SDs above the prior's center; worth revisiting independently of any rescaling |
| Prior-predictive checks | Must be rerun after any change of unit or baseline, since they test rates per apartment and cohort shares |

**Run the form comparison once, on Model B.** Model C then consumes the frozen
winner. Registering independent form-variants for C is not merely redundant, it
is unsafe: the guard compares the two approaches' *selected* candidates, so if
B and C happened to pick different winners, selection would fail outright. The
design intent is a single sweep whose result both models share, which also
keeps the Bayesian model's advantage attributable to its inference rather than
to a different feature form.

### The candidate list

Seven predeclared candidates, each changing **one stage at a time**:

| # | Candidate | Total spec | Composition spec |
|---|---|---|---|
| 1 | base | base | base |
| 2 | `total__ses_quadratic` | SES quadratic | base |
| 3 | `total__daycare_log1p` | daycare log1p | base |
| 4 | `total__room_share_x_household_size` | + interaction | base |
| 5 | `composition__daycare_log1p` | base | daycare log1p |
| 6 | `composition__room_share_x_median_age` | base | + interaction |
| 7 | combined | whichever total forms won | whichever composition forms won |

**Why one stage at a time is enough, and a cross-product is not needed.** The
two stages are fitted independently, with separately tuned penalties, so
changing the total spec leaves the composition fit bit-identical, and vice
versa. Attribution therefore comes from the **stage-level metrics** rather than
from the joint one:

- `composition_log_loss` and `composition_brier`
  ([metrics.py:646-699](../src/age_group_prediction/metrics.py#L646-L699)) score
  the cohort probabilities alone, independently of predicted totals, so they
  isolate the composition stage;
- the `n_children_total` metrics isolate the total stage.

A full cross-product of the two stages' forms would multiply the candidate
count for no extra information, and would inflate the optimism of whichever
candidate won. Candidate 7 is the only one with more than one change, and it is
assembled from what candidates 2–6 established rather than guessed in advance.

---

## 6. Interactions

This section asks the blank-slate question: **ignoring what the current code
allows, which interactions are worth considering at all, and what does each one
buy?** The answer is driven less by domain intuition than by one structural
fact about the data.

### 6.1 What can be interacted, and why that is the whole question

Features live at two different levels, and only one of them has many units
(counts below from one simulated population, seed 0):

| Level | Features | Units available | Varies within a neighborhood? |
|---|---|---|---|
| **Building** | `3_rooms_share`, `4_rooms_share`, `5_rooms_share`, `n_apartments` | **245** | **Yes** — 57–76% of room-share variance is within-neighborhood |
| **Neighborhood** | `ses`, `avg_household_size`, `median_age`, `n_daycares_500m`, `school_status` | **60** | No — constant for every building in the neighborhood |

Three consequences follow, and together they prune most of the candidate space
before any domain reasoning starts:

1. **Any interaction not involving room shares is neighborhood × neighborhood**,
   and rests on 60 points however many buildings the data holds. `ses ×
   avg_household_size` has the same 60 units behind it whether the sample is
   245 buildings or 2,450.
2. **In Model C they are weaker still**: the hierarchical neighborhood effect
   absorbs neighborhood-level variation, so a neighborhood × neighborhood term
   competes with the random effect for exactly the same signal.
3. **Room shares are the only genuinely building-level predictor**, because
   `n_apartments` becomes the offset rather than a feature.

So the interactions worth considering are essentially **room shares × one
neighborhood feature**, and the real question is *which moderator* — asked
separately for each phase, because the two phases ask different questions.

### 6.2 Total stage — "how many children per apartment"

The mechanism here is **occupancy**: how many children an apartment holds
depends on who lives in it.

| Option | Verdict | What it buys you |
|---|---|---|
| room share × `avg_household_size` | **Add — first choice** | Capacity meets demand. A 5-room apartment in an area where households average 3.4 people holds more children than the same apartment where they average 1.8. This is the most direct mechanism available for *how many* |
| room share × `ses` | **Add — second choice** | Tests whether extra space turns into more children or into more space per person. The sign is genuinely unknown beforehand — crowding in poorer areas pushes one way, affordability in richer ones the other — which is exactly what makes it worth estimating rather than assuming. Nearly orthogonal to household size (r = −0.13), so it asks a separate question rather than restating the first choice |
| room share × `median_age` | Optional — but not alongside the first choice | Older neighborhoods may hold empty-nesters even in large apartments. But the moderator correlates −0.44 with household size, and the two interaction blocks correlate 0.52, so they partly restate each other. For *how many* children, occupancy is the better-motivated mechanism |
| room share × `n_daycares_500m` | Skip here | Daycare mostly explains whether families with young children are present at all, which is a main effect. That large apartments benefit *more* from it is second-order. It earns its place in the composition stage instead |
| room share × `school_status` | Skip | 2 dummies × 3 shares = 6 columns for a weak prior |
| `ses` × `avg_household_size`, `n_daycares_500m` × `median_age`, or any other neighborhood pair | Skip | 60 effective units, and in Model C they compete with the neighborhood random effect |
| `log n` × anything | Not an interaction question | This asks "is the offset coefficient really 1?", which the single free-$\gamma$ diagnostic of Section 4.4 already answers more directly |

### 6.3 Composition stage — "which ages, given there are children"

The mechanism here is **family lifecycle**: children age in place, so a
building's age mix reflects when its families formed and moved in.

One structural note before the options. In a multinomial logit **every feature
already gets its own coefficient per cohort**, so a feature × cohort
interaction is automatic and free. The consequence is that this stage is
roughly twice as parameter-hungry as the total stage for the same design
matrix, and parsimony should bind harder here.

| Option | Verdict | What it buys you |
|---|---|---|
| room share × `median_age` | **Add — first choice** | The lifecycle term. A large apartment in an old neighborhood holds teenagers; the same apartment in a young neighborhood holds toddlers. This is the sharpest statement the data can make about *which* ages |
| room share × `n_daycares_500m` | **Add — second choice** | Daycare presence marks **young** children specifically, so this term targets the kindergarten cohort directly rather than shifting the whole mix. Distinct enough from neighborhood age to sit beside it (moderators r = −0.23, blocks r = 0.28) |
| room share × `avg_household_size` | Skip | Household size says how many people live in an apartment, not how old its children are — and it is collinear with `median_age`, the better-motivated moderator here |
| room share × `ses` | Skip | There is no mechanism by which affluence shifts the age *mix*, as distinct from the number of children. Its place is the total stage |
| room share × `school_status` | Skip the interaction, keep the main effect | The composition story is already carried by the `school_status` **main effect**: an existing school attracts families with school-age children, a planned one attracts families anticipating. Interacting it with room shares is second-order and costs 6 columns |
| Any neighborhood pair | Skip | As in 6.2 |

### 6.4 The two phases side by side

**The recommended moderator differs by phase by design, not by convention.**
Household size answers *how many*; neighborhood age answers *which ages*. Each
phase gets the moderator matching the question it is asking, and the collinear
alternative is the one dropped in both cases.

|  | First choice | Second choice | Deliberately dropped |
|---|---|---|---|
| Total | room share × `avg_household_size` | room share × `ses` | `median_age` (collinear with the first choice) |
| Composition | room share × `median_age` | room share × `n_daycares_500m` | `avg_household_size` (collinear, and the wrong question) |

Three further points:

- **This reproduces the split the current code enforces**, which refuses
  `room_share_x_household_size` on `composition` and
  `room_share_x_median_age` on `total_count`. The conclusion is the same, but
  reached by mechanism and collinearity rather than by rule. The code is
  **missing both second choices**: neither `room_share_x_ses` nor
  `room_share_x_daycare` exists in `_VALID_INTERACTIONS`.
- **Daycare `log1p` belongs in both phases**, but that is a functional form,
  not an interaction: saturation is a property of the covariate, so it applies
  wherever daycare enters.
- **If one shared interaction were wanted across both phases** for simplicity,
  `room share × median_age` is the only defensible choice, since neighborhood
  age plausibly shifts both the count and the mix. It is not recommended: it
  costs the total stage its better moderator.

### 6.5 If two interactions are used in one model

Rules for a combination that makes sense:

1. every operand is present as a main effect;
2. the two terms answer different substantive questions rather than restating
   one;
3. their **moderators are not strongly correlated**, since the interaction
   blocks inherit that correlation;
4. never mix the 3-column room-share form with the 1-column index of Section
   6.6 for the same variable.

Measured collinearity between room-share interaction blocks (10 simulated
populations, centered and scaled as recommended; largest absolute correlation
between the two 3-column blocks, and the condition number of the 6 columns):

| Combination | Max block correlation | Condition number | Verdict |
|---|---|---|---|
| rooms × household size **+** rooms × SES | 0.15 | 2.1 | **The total-stage pair** |
| rooms × median age **+** rooms × daycare | 0.28 | 2.6 | **The composition-stage pair** |
| rooms × SES + rooms × median age | 0.08 | 2.2 | Safe, but mixes the two phases' questions |
| rooms × household size + rooms × daycare | 0.12 | 2.2 | Safe, but likewise |
| rooms × household size + rooms × median age | **0.52** | 3.5 | **Avoid** — the moderators correlate −0.44, so the blocks partly restate each other |

The two recommended pairs are among the safest measured combinations, which is
a useful confirmation: the choices made on mechanism in 6.2 and 6.3 turn out
not to fight each other statistically.

### 6.6 A cheaper form of any room-share interaction

Each room-share interaction costs 3 columns. The scalar mean-apartment-size
index

$$
\bar r_i = \sum_k k\,s_{ik}
$$

gives a 1-column version of the same term, read as "per extra room on average".
The cost is an assumption that the size effect is linear in room count, which
is itself checkable: compare the scalar against the 3-column form as a **main
effect** before using it in an interaction. Never carry both forms of the same
variable at once.

---

## 7. Summary

| Feature | A (trees) | B and C (GLMs) | Motivation |
|---|---|---|---|
| `ses` | Raw; candidate: centered then squared. No spline | $z$; quadratic candidate in the **total stage only**, scaled once before squaring. No spline | Quadratic is nested and interpretable, and curvature has a mechanism for *how many* children but none for *which ages*; only about 60 distinct SES values, so a 5-column spline overfits and can't be read |
| `avg_household_size` | Raw | $z$ (report $\beta/s$; fixed $(x-2.6)/0.5$ optional) | Scale sets only the unit; matches C's priors |
| `median_age` | Raw | $z$ (report $\beta/s$; fixed $(x-37)/10$ optional) | Same |
| `n_daycares_500m` | Raw; drop the `log1p` candidate | Candidate 1: $d/8$ (0 = none; 8 = 99th percentile, no clipping). Candidate 2: $\log\frac{1+d}{1+\bar d}$ | Fixed bounds are fold-stable where a learned min-max is not; the centered log1p keeps its unit across folds and reads as an elasticity |
| Room shares (4, 5, 6; reference 3) | Raw | $(s_k-\bar s_k)/0.10$, one common unit | 10 pp substituted out of the 3-room reference; centering puts the baseline at the average mix |
| `n_apartments` | Raw feature, and optionally the exposure (`use_exposure=True`, Poisson) | Not a feature | Trees cannot extrapolate proportional growth; the feature still captures departures from proportionality |
| Exposure | Raw $n$ passed as `fit(..., exposure=n)` / `predict(..., exposure=n)`; the model uses $\log n + b$ as `init_score` | $\log n$ offset, coefficient 1, unscaled | Models the per-apartment rate; `log1p` adds a size-dependent bias of about $1/n$ |
| `school_status` | One-hot (the only required transform) | One-hot, reference `none`, unscaled | Already interpretable |
| Interactions | None needed — splits represent them | **Total:** room share × household size, then × SES. **Composition:** room share × median age, then × daycare. One per candidate | Only room shares vary within a neighborhood, so every well-powered interaction is room share × a neighborhood feature. Each phase takes the moderator matching its question — household size for *how many*, neighborhood age for *which ages* (Section 6) |

---

## 8. Building Each Model's Transformer

Every declaration below is **complete and runnable**: a candidate is written out
in full rather than derived from a base, so what a fold actually fits is on the
page. They build on
[`age_group_prediction.feature_engineering`](../src/age_group_prediction/feature_engineering/),
which knows nothing about this project — columns, categories and bounds are
supplied here, at the call site.

```python
from age_group_prediction.feature_engineering import (
    Center, ColumnPlan, DomainMinMax, DomainScale, FeatureTransformer,
    Interaction, OneHot, Quadratic, RelativeSaturation, Standardize,
)
```

Three rules that the declarations depend on:

- **A plan's chain is applied in order**, and only the last renaming step
  changes the column's name. `Center() → DomainScale(0.1)` leaves
  `4_rooms_share` called `4_rooms_share`; `RelativeSaturation()` renames
  `n_daycares_500m` to `n_daycares_500m_sat`.
- **`Interaction` names design-matrix columns**, not raw inputs — what a plan
  *emits*. Under the daycare-saturation form the operand is
  `n_daycares_500m_sat`, and an interaction still written against
  `n_daycares_500m` fails at fit rather than quietly meaning something else.
- **The same column may feed several plans.** That is how `ses` and
  `ses_squared` sit side by side.

### 8.0 Shared column groups

```python
from age_group_prediction.preprocessing import ShareTransformer

# Section 2: the 3-room share is the omitted reference.
shares = ShareTransformer(
    ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
)
table = shares.fit_transform(raw_table)  # adds 4_, 5_ and 6_rooms_share

ROOM_SHARES = ("4_rooms_share", "5_rooms_share", "6_rooms_share")
SCHOOL = OneHot(
    categories=("none", "existing", "planned"), reference_category="none"
)
```

### 8.1 Model A — `DirectCohortModel` (LightGBM)

Trees gain nothing from a monotone rescale (§3.1), so the numerics are passed
through untouched and only the categorical is encoded.

```python
tree = FeatureTransformer(
    plans=(
        ColumnPlan(
            name="numeric",
            columns=(
                *ROOM_SHARES,
                "ses",
                "avg_household_size",
                "median_age",
                "n_daycares_500m",
                "n_apartments",
            ),
        ),
        ColumnPlan(name="school", columns="school_status", transforms=(SCHOOL,)),
    ),
)
```

**`n_apartments` can be both a feature and the offset, on purpose.** Section
3.3 keeps it as a column *and* uses $\log n$ as the offset. The offset asserts
exact proportionality, and the feature lets the trees learn departures from
it. The model takes the raw exposure itself, so the transformer declares no
`exposure_column`:

```python
from sklearn.base import clone
from age_group_prediction.modeling import DirectCohortModel

features = clone(tree).fit(fit_df)
model = DirectCohortModel(use_exposure=True).fit(
    features.transform(fit_df), fit_df["n_kindergarten"],
    exposure=fit_df["n_apartments"],
)
```

For `objective="regression"` there is no log link, and the model refuses
`use_exposure=True` (§3.3).

**SES-quadratic variant.** Add one plan; `ses` itself stays, because the squared
column earns its place only by being non-monotone (§3.2):

```python
ColumnPlan(name="ses_sq", columns="ses", transforms=(Center(), Quadratic()))
```

Not offered for Model A: the `log1p` daycare form (indistinguishable from linear
under trees), SES splines (rejected, §3.1), and interactions (trees build them
through successive splits).

### 8.2 Model B — total stage

The offset makes the linear predictor a per-apartment rate, so
`n_apartments` is the exposure and never a feature (§4.4).

```python
total_base = FeatureTransformer(
    plans=(
        ColumnPlan(name="ses", columns="ses", transforms=(Standardize(),)),
        ColumnPlan(
            name="household", columns="avg_household_size", transforms=(Standardize(),)
        ),
        ColumnPlan(
            name="median_age", columns="median_age", transforms=(Standardize(),)
        ),
        ColumnPlan(
            name="room_share",
            columns=ROOM_SHARES,
            transforms=(Center(), DomainScale(scale=0.1)),
        ),
        ColumnPlan(
            name="daycare",
            columns="n_daycares_500m",
            transforms=(DomainMinMax(minimum=0.0, maximum=8.0),),
        ),
        ColumnPlan(name="school", columns="school_status", transforms=(SCHOOL,)),
    ),
    exposure_column="n_apartments",
)
```

Reading the coefficients: per SD for the z-scored neighborhood numerics (§4.5);
per 10 percentage points of share for the room shares (§4.3); from no daycares
to eight for the daycare term (§4.1); contrasts against `none` for the school
dummies (§4.6).

Unlike Model A, B and C do not list `n_apartments` among the plans, so building
size enters **only** as the offset (§4.4). The package does not enforce that;
the declaration is what expresses it. Hand the offset to the model separately,
where it cannot be mistaken for a predictor:

```python
offset = total_base.fit(fit_df).log_exposure(fit_df)   # a Series, log n
```

### 8.3 Model B — composition stage

**The same plans with no exposure** (§4.7): age-group shares do not depend on
building size. In a multinomial logit every feature already gets a coefficient
per cohort, so this stage is roughly twice as parameter-hungry for the same
design matrix and parsimony binds harder (§6.3).

```python
composition_base = FeatureTransformer(
    plans=total_base.plans,      # identical; only the exposure differs
)
```

### 8.4 Model C — `BayesianConditionalModel`

Model C has **no candidates of its own**: it consumes Model B's frozen winner
(§5), so it is fitted with whichever declarations won there, unchanged. Its
`Normal(0, 0.5)` priors were set for z-scored units, which is one reason §4.5
keeps standard scaling as the default — changing a unit changes how hard the
prior shrinks.

### 8.5 The seven candidates of Section 5, as declarations

Each candidate changes **one stage at a time**, so only the changed stage is
written out; the other is the base declaration above, unchanged.

| # | Candidate | Total | Composition |
|---|---|---|---|
| 1 | base | `total_base` | `composition_base` |
| 2 | `total__ses_quadratic` | + the plan in 8.5.1 | base |
| 3 | `total__daycare_log1p` | daycare plan replaced, 8.5.2 | base |
| 4 | `total__room_share_x_household_size` | + the interactions in 8.5.3 | base |
| 5 | `composition__daycare_log1p` | base | daycare plan replaced, 8.5.2 |
| 6 | `composition__room_share_x_median_age` | base | + the interactions in 8.5.4 |
| 7 | combined | whichever total forms won | whichever composition forms won |

#### 8.5.1 SES quadratic (candidate 2)

Scale once, then square: $\beta_1$ is then the slope at average SES and
$\beta_2$ the curvature (§4.2). `ses` keeps its own plan; this one is added
beside it.

```python
ColumnPlan(name="ses_sq", columns="ses", transforms=(Standardize(), Quadratic()))
```

#### 8.5.2 Daycare, centered `log1p` (candidates 3 and 5)

Replaces the daycare plan rather than joining it. `RelativeSaturation` emits
$\log(1+d) - \log(1 + \bar d)$ — `log1p` of the mean, not the mean of `log1p` —
so the reference stays in the original counts and the coefficient is an
elasticity (§4.1, candidate 2).

```python
ColumnPlan(
    name="daycare", columns="n_daycares_500m", transforms=(RelativeSaturation(),)
)
```

**The column is now `n_daycares_500m_sat`.** Any interaction naming it must use
the new name.

#### 8.5.3 Total-stage interaction (candidate 4)

Room shares × `avg_household_size` — capacity meets demand, the first choice for
*how many* (§6.2). One `Interaction` per pair; never a within-side pair, because
room shares sum to at most 1 and their pairwise products are structurally
constrained (§6.1).

```python
interactions=tuple(
    Interaction(left=share, right="avg_household_size") for share in ROOM_SHARES
)
```

The second choice is room shares × `ses`; §6.5 measures the two together at a
block correlation of 0.15 and a condition number of 2.1, making them the
recommended total-stage pair.

#### 8.5.4 Composition-stage interaction (candidate 6)

Room shares × `median_age` — the lifecycle term, the first choice for *which
ages* (§6.3).

```python
interactions=tuple(
    Interaction(left=share, right="median_age") for share in ROOM_SHARES
)
```

The second choice is room shares × `n_daycares_500m`, which targets the
kindergarten cohort directly; §6.5 measures that pair at 0.28 and 2.6. Under
candidate 5 the operand is `n_daycares_500m_sat`.

**Do not combine** room shares × `avg_household_size` with room shares ×
`median_age`: the moderators correlate −0.44 and the blocks 0.52 (§6.5).

### 8.6 Per-fold use

One declaration is fitted once per fold. `sklearn.base.clone` gives a fresh
unfitted copy, so a validation fold is transformed by the training fold's
statistics rather than its own:

```python
from sklearn.base import clone

fold = clone(total_base).fit(train_df)
X_train, X_valid = fold.transform(train_df), fold.transform(valid_df)
offset_train = fold.log_exposure(train_df)
```

### 8.7 Still Outstanding

What this section replaced was an implementation checklist. Some of its items
are now satisfied by the package — the daycare, room-share and centering forms of
its item 2 are expressed by `DomainMinMax`, `RelativeSaturation`,
`Center`+`DomainScale` and `Standardize`+`Quadratic`; and its item 4, adding
entries to a closed `_VALID_INTERACTIONS` literal, is replaced by open
user-named `Interaction`s. **The rest still stand:**

1. **Room-share reference:** done for new code.
   `ShareTransformer(..., reference_column="3_rooms")` derives the 4-, 5- and
   6-room shares (§8.0). The old `build_modeling_table` keeps the 6-room
   reference until it is deleted with `modeling_config`.
2. **Candidate enumeration:** drop the spline candidates and
   `tree__daycare_log1p`; offer the two daycare forms; keep the composition
   stage's SES form **linear** (no `composition__ses_quadratic`); emit the
   seven paired candidates of Section 5. This machinery still runs on
   `FeatureSpec` in
   [fitted_features.py](../src/age_group_prediction/fitted_features.py) rather
   than on the declarations above, and moves when that module is retired.
3. **`DirectCohortModel`:** done in the rebuilt
   `modeling.DirectCohortModel`, which takes `use_exposure=True` and uses
   $\log n + b$ as `init_score`
   ([DIRECT_COHORT_MODEL.md §0](DIRECT_COHORT_MODEL.md#0-the-rebuilt-model-modelingdirect_cohortpy)).
4. **Model C:** no candidate changes of its own. Rerun the prior-predictive
   checks after any unit or baseline change, and revisit
   `total_intercept_loc = -2` against the observed log rate of about −0.5.
5. **Provenance:** every change above alters feature specs and therefore
   candidate fingerprints. Compare old and new forms as separate candidates on
   identical folds.

## 9. Related Documents

- [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md): the `FittedFeatureTransformer`
  implementation the models still use, in `fitted_features.py`.
- [DIRECT_COHORT_MODEL.md](DIRECT_COHORT_MODEL.md): Model A.
- [INDEPENDENT_TOTAL_PROBABILITY_MODEL.md](INDEPENDENT_TOTAL_PROBABILITY_MODEL.md):
  Model B's two stages.
- [BAYESIAN_CONDITIONAL_MODEL.md](BAYESIAN_CONDITIONAL_MODEL.md): Model C's
  priors and prior-predictive checks.
- [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md): how
  candidates are compared on identical folds.
