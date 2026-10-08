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
| **A** | `CountModel` | One regressor per cohort (any estimator with a Poisson or Gaussian loss); raw features; optional exposure as a weighted rate |
| **B** | `TotalTimesProbabilityModel` | A Poisson/NB2 total with the exposure as an offset (`CountModel`), times a classifier of the cohort probabilities (`CohortProbabilityModel`) |
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
| Centering ($x - c$) | The intercept, every main effect that appears inside an interaction, and, before squaring, where the vertex lies | The slope of a linear column; and in B, nothing about a linear column's fit, since the intercept is unpenalized |
| Scaling ($x/s$) | The coefficient's unit, and hence how hard B's L2 penalty and C's priors shrink it | The shape of the fitted relationship |

**Consequence:** a scale learned per fold changes the unit from fold to fold,
while a center learned per fold is harmless for a linear column. This is the
main argument in Sections 4.1 and 4.2. Before squaring, the center fixes the
vertex, so a declared reference point is better there (Section 3.2).

**Trees see only ordering.** For Model A, any strictly monotone per-feature
transform (scaling, `log1p`) leaves the splits unchanged. Only these matter:
non-monotone transforms, transforms that replace or combine columns, and
offsets, which act on the target side.

**Prefer fixed anchors over learned scales.** $\tilde x = (x-c)/s$, with $c$ and
$s$ chosen from domain knowledge, keeps units the same across folds, cannot
leak, and reads naturally ("per decade", "per 2 daycares"). In the package,
`CenterByReferencePoint` fixes $c$, `DomainScale` fixes $s$, and `DomainMinMax`
fixes both ends of a declared range.

**One scale for every penalized column: $s$ near 1 SD.** The z-scored columns
have SD 1 by construction, so a fixed $s$ is chosen near the column's SD:
0.10 for the room shares, 2 for the daycare count. A unit far from 1 SD acts
as a hidden per-feature penalty: a column with standard deviation $\sigma$
has the same real effect penalized $1/\sigma^2$ times as hard as a z-scored
one. Section 4.8 measures every column against this rule.

- **The school dummies are the known exception.** A 0-to-1 switch is about
  2 SD, so they are shrunk 4 to 6 times harder than the z-scored columns.
- **The alternative is Gelman's (2008) 2-SD convention:** halve every
  continuous column so that all of them sit near SD 0.5 and match the dummies.
  It is not used here, because every coefficient would then read "per 2 SD"
  or "per 20 pp".

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
| `ses` squared | Adds a **non-monotone** column while keeping `ses`: $(\text{ses}-c)^2$ orders buildings by their distance from $c$, so one split isolates both tails where `ses` alone needs two. Useless if $c$ lies outside the data range, since $x^2$ is then monotone | **Keep as a candidate.** Center at the SES reference point 0.0 (Section 3.2) |
| SES B-spline | **Removes `ses`** and replaces it with 5 overlapping basis columns ([fitted_features.py:285-290](../src/age_group_prediction/fitted_features.py#L285-L290)), so no column orders buildings by SES and a simple SES threshold has to be approximated across columns | **Rejected** |
| Interactions | Trees build them through successive splits | Not needed; not offered for `tree` |
| The exposure $n$, as a weighted regression of the rate $y/n$ with weight $n$ (§3.3) | Changes the target scale, not a column | **Add** (Section 3.3) |

### 3.2 SES candidates

Two, matching the GLM candidates:

- **linear:** raw `ses`;
- **quadratic:** $[\,\text{ses},\ (\text{ses}-c)^2\,]$, raw `ses` plus its
  squared deviation from the SES reference point $c = 0.0$.

**Why the center matters.** A tree gains nothing from a
monotone column, so the squared column earns its place only by being
non-monotone: $(\text{ses}-c)^2$ ranks buildings by **distance from $c$**, and
one split on it isolates both tails at once, where `ses` alone would need two
splits in separate branches. The center $c$ therefore defines what the column
means.

- If $c$ lies inside the data range, the column reads as "how far from $c$".
- If $c$ lies outside it, $x^2$ is monotone over the observed values and the
  trees can extract nothing from it. Squaring `median_age` (25–45) raw would be
  the clear example.

**The SES reference point: 0.0.** A substantive turning point is a better $c$
than the mean, and here one is named:
- **What it is.** 0.0 is the simulated population mean (`ses_mean` in
  `configs/simulation.toml`, with `ses_sd = 1.0`): average SES, neither poor
  nor rich. The squared column then reads as the squared distance from average
  SES, in population SDs.
- **Why not the fold mean.** `Center` would use the fold's mean, which moves
  from sample to sample: 0.07 in the reference table above, but from −0.25 to
  +0.26 across single runs of 60 neighborhoods (20 runs). A fixed reference gives the column the
  same meaning in every fold, and it is the point the hypothesis below names.
- **How it is declared.** `CenterByReferencePoint(reference_point=0.0)`
  (§8.1). In this data, subtracting 0 changes no value, so the column is
  $\text{ses}^2$. The declaration still matters: it states the reference, which
  real SES data needs, since its zero point may be arbitrary or outside the
  observed range.

**The hypothesis, and what the simulator does.** The motivating hypothesis is
that strong- and weak-SES populations have larger households than the middle:
an effect in both tails, which is why the distance from average SES is the
quantity to model. The simulator does not generate that. There, the SES
U-shape is in the **number of children**: the total-children log-mean has a
$0.05\,\text{ses}^2$ term with its vertex at 0 and no linear SES term
(`ses_quadratic_coef` in `configs/simulation.toml`). `avg_household_size` is
drawn from median age plus noise, independently of SES, so the r = −0.13 in the
table above is sampling variation, not an effect. In this data, then, the
squared deviation can pick up a U-shape in child counts; it says nothing about
household size.

The GLMs use the same reference for their squared term (Section 4.2), where the
center also changes what the coefficients mean.

### 3.3 Exposure: add the offset **and** keep `n_apartments`

For a Poisson loss, `CountModel(use_exposure=True)` fits the offset model

$$
\log \mu_{i,c} = \log n_i + f_c(\mathbf{x}_i),
$$

so each cohort's estimator learns a **per-apartment rate**. It does so as a
weighted regression of the rate: the estimator is fitted on $y_{i,c} / n_i$
with `sample_weight` $n_i$, and `predict` multiplies by $n_i$. That has the
same likelihood as the offset, works for any regressor that takes
`sample_weight`, and starts from the average rate $\sum_i y_{i,c} / \sum_i n_i$.
The derivation, and the Gaussian case, are in
[Independent cohort models §2.1](INDEPENDENT_COHORT_MODELS.md#21-the-exposure-as-a-weighted-rate).

- **Why the offset:** child counts grow roughly in proportion to $n$. Trees
  approximate that with step functions, need many splits to do so, and cannot
  extrapolate beyond the sizes seen in training, so a 100-apartment building is
  predicted like the largest training building.
- **Why keep `n_apartments` as a feature too:** the offset asserts exact
  proportionality. Keeping the feature lets the trees learn departures from it,
  for example larger buildings having fewer children per apartment.
- **With a Gaussian loss too:** the weighted rate is then least squares of the
  count with a mean proportional to $n$ and a variance proportional to $n$, as
  a count's variance grows
  ([Independent cohort models §2.1 (D)](INDEPENDENT_COHORT_MODELS.md#21-the-exposure-as-a-weighted-rate)).

---

## 4. Models B and C (GLMs): Final Per-Feature Decisions

Both models share these specs; C uses B's selected specs frozen.

### 4.1 `n_daycares_500m`: two candidates

**Candidate 1: linear with a fixed domain unit.**

$$
\tilde d = \frac{d}{2}
$$

- The unit is a domain constant, not learned: 2 daycares, the round number
  nearest the SD of 1.7 (1.4 to 2.2 across single runs). The column then has
  SD 0.86, close to the z-scored columns and the room shares (Section 4.8).
- **Reading:** $\beta$ is the log-rate change per 2 daycares, and
  $\tilde d = 0$ means "no daycare", the baseline you wanted. The effect from
  no daycares to 8, the 99th percentile, is $4\beta$.
- **Why not $d/8$:** 8 is the *range*, about 4.7 SD, while every other column
  is divided by about 1 SD. $d/8$ has SD 0.22, so the same real effect would
  be penalized about 22 times harder than on a z-scored column, and harder
  than candidate 2, which makes the comparison between the two candidates
  unfair.
- **Why not a learned min-max:** zeros occur in only about 6% of buildings, so
  a fold can contain none, and its learned minimum becomes 1, at which point 0
  no longer means "no daycare" and the baseline differs across folds. The
  learned maximum is also a single-neighborhood outlier that moves between
  folds. A fixed unit keeps the interpretation without that instability.

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
- **The column has SD 0.50**, so it is penalized about 4 times harder than a
  z-scored column. That is kept on purpose. Dividing by $\ln 2$ ("per
  doubling") raises the SD to 0.72 but did not improve held-out fit in either
  stage (Section 4.8), and it gives up the elasticity reading.

**Caveat when the daycare term is 0-anchored** (candidate 1): in
`daycare_x_median_age`, the median-age main effect then describes a building
with **no daycares**, a case covered by only 6% of buildings. The centered
candidate 2 avoids this, which is another reason to prefer it when interactions
are active.

### 4.2 `ses`: two candidates (no spline)

The linear column is $z = (\text{ses} - m)/s$, with $m$ and $s$ from the fit
partition. The squared column is $(\text{ses} - 0)^2$, the squared deviation
from the SES reference point 0.0 (Section 3.2).

| Candidate | Columns | Reading |
|---|---|---|
| linear | $z$ | Log-rate change per SD of SES |
| quadratic | $z,\ (\text{ses}-0)^2$ | $\beta_1$: the slope at the reference point, per SD, because the squared term is flat there, and standardizing the linear column only moves the intercept; $\beta_2$: the curvature around the reference, per population SD², directly comparable with the simulator's $0.05$ |

**Square the deviation from the reference, not $z$.** Both give the same
fitted curve in an unpenalized model; they differ in what $\beta_2$ means and
whether that meaning holds across folds.

- $z^2$ puts the vertex at the fold mean and measures it in fold SDs, so both
  move from fold to fold: a single run's mean ranges from −0.25 to +0.26 and
  its SD from 0.82 to 1.14 (20 runs of 60 neighborhoods).
- $(\text{ses}-0)^2$ fixes both. The vertex is average SES, the point the
  hypothesis names, and the unit is the population SD (`ses_sd = 1.0`).
- **Neither choice decorrelates the two columns** in a sample this small. The
  sample's skew sets $\text{corr}(x, x^2)$: across the same 20 runs it ranges
  from −0.55 to +0.27 for $\text{ses}$ and from −0.39 to +0.35 for $z$ (median
  −0.09 for both). The correlation does not change the fitted curve, only how
  precisely $\beta_1$ and $\beta_2$ are separated, and B's penalty and C's
  priors keep them stable. The choice therefore rests on meaning.

The old code squares $z$
([fitted_features.py:283-284](../src/age_group_prediction/fitted_features.py#L283-L284)).
Squaring works here without rescaling only because 0 lies near the centre of
the SES data.
Squaring a column whose centre is far from zero, relative to its spread, makes
the two columns nearly collinear: raw `median_age` would be the clear example.

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

**Why 0.10 and not 0.01.** The shares have SD 0.08 to 0.12, so dividing by
0.10 gives columns with SD 0.77 to 1.18, the same scale as the z-scored
columns. A unit of 1 pp would give SD 8 to 12: the coefficient becomes 10
times smaller and its penalty 100 times smaller, so the shares would be almost
unregularized beside everything else. To report per percentage point, divide
the fitted coefficient: $\beta_{1\,\text{pp}} = \beta_{10\,\text{pp}} / 10$.

**Rejected alternatives:**

- **Raw shares:** a one-unit change means moving 100% of apartments into size
  $k$, roughly 8 SD away, and the intercept describes an all-3-room building.
  The same real effect would also be penalized about 100 times harder than on
  a z-scored column.
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
exposure column that is also listed as an ordinary numeric feature (the old
`FeatureSpec`; in the rebuilt code nothing enforces it, and the declaration
and the model's `use_exposure` express it, §8.2), so building
size enters only as the offset

$$
\log \mu_i = \log n_i + \beta_0 + \mathbf{x}_i^\top\boldsymbol\beta
\quad\Longleftrightarrow\quad
\frac{\mu_i}{n_i} = \exp(\beta_0 + \mathbf{x}_i^\top\boldsymbol\beta),
$$

with the coefficient of $\log n_i$ **fixed at 1**. The linear predictor
therefore models children per apartment, and every other coefficient is a
multiplicative effect on that rate.

Used by: B's total stage and C's total stage, and optionally Model A
(`use_exposure=True`, Section 3.3). The composition stages use no exposure,
because age-group shares do not depend on building size.

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
  folds. Both are close to 2 SD, so under the 1-SD convention of Section 1
  they are for reporting only; a fitted column would use about 1 SD,
  $(x-2.6)/0.25$ and $(x-37)/4$. Either choice is defensible; z-scoring is
  the pragmatic default.

### 4.6 `school_status`

One-hot with the schema reference `none` dropped, unscaled. The dummies are
already interpretable ("existing vs. no school"), and scaling them would only
obscure that. The cost is that they are shrunk 4 to 6 times harder than the
z-scored columns (Section 4.8), which is accepted.

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
carrying an exposure is refused at construction. That is the old `FeatureSpec`;
in the rebuild (planned: [multi-cohort plan](MULTI_COHORT_MODELS_PLAN.md) N3 e,
B4) the probability model will have no exposure setting and will ignore a
passed exposure.

### 4.8 Scale audit

**The rule.** B's penalty and C's priors act on the coefficient, so a column
with standard deviation $\sigma$ has the same real (per-SD) effect penalized
$1/\sigma^2$ times as hard as a z-scored column.

**Measured scales.** The declarations of Section 8, fitted on 20 simulated
populations of 240 buildings; the range is across single runs.

| Column | SD | Penalty vs. z-scored | Verdict |
|---|---|---|---|
| `ses`, `avg_household_size`, `median_age` ($z$) | 1.00 | 1.0× | Reference |
| `4_rooms_share` | 1.18 (1.08 to 1.25) | 0.7× | Fine |
| `5_rooms_share` | 1.04 (0.96 to 1.20) | 0.9× | Fine |
| `6_rooms_share` | 0.77 (0.69 to 0.87) | 1.7× | Fine |
| `ses_squared` | 1.27 (0.86 to 1.78) | 0.6× | Fine; it varies by run because a run has about 60 neighborhoods |
| Room share × household size, SES or median age | 0.77 to 1.18 | 0.7× to 1.7× | Fine |
| `n_daycares_500m` as $d/2$ | 0.86 (0.67 to 1.09) | 1.4× | Fine |
| `n_daycares_500m_sat` | 0.50 (0.43 to 0.58) | 4× | Accepted (Section 4.1) |
| `school_status_existing` | 0.49 | 4× | Accepted (Section 4.6) |
| `school_status_planned` | 0.41 (0.34 to 0.49) | 6× | Accepted (Section 4.6) |
| Former $d/8$, for comparison | 0.22 (0.17 to 0.27) | 22× | Replaced |

**What rescaling buys.** Both of B's stages were fitted with each daycare
form on 30 pairs of simulated populations, trained on one and scored on the
other, across the tuned penalty range.

- **It equalizes shrinkage.** Share of the unpenalized effect kept at the
  strongest tuned penalty:

  | Column | Total stage (`total_l2_penalty` = 1.0) | Composition stage (`probability_c` = 0.01) |
  |---|---|---|
  | $z$-scored columns | 97% | 91% |
  | $d/8$ | 51% | 39% |
  | $d/2$ | 100% | 94% |
  | Saturation, natural log | 85% | 78% |
  | Saturation, per doubling | 94% | 89% |

- **It does not improve held-out fit.** At each form's best penalty, $d/8$
  against $d/2$ differs by 0.027 in total-stage deviance (SE 0.018) and by
  0.00013 in composition log loss (SE 0.00010), and the two saturation forms
  by 0.008 (SE 0.004) and 0.00008 (SE 0.00003). The harder-shrunk form is the
  slightly better one each time, because the daycare effect is small in this
  data: about 1% of the rate per SD in the total stage.

**Conclusion.** The common scale is a consistency rule, not an accuracy gain:
it keeps a column's unit from acting as a hidden penalty, and it makes the two
daycare candidates comparable. How hard to shrink is the tuned penalty's job.

---

## 5. Model And Variation Catalogue

### Model A: `CountModel`

One regressor per cohort (`n_kindergarten`, `n_elementary`,
`n_highschool`). The hyperparameters are fixed per instance and tuned from
outside, not inside `fit`.

| Variation | What changes | Question it answers |
|---|---|---|
| **Base** | All 8 numeric columns raw, one-hot `school_status`, 3-room reference | Reference point for everything below |
| SES quadratic | Adds $(\text{ses}-0)^2$, the squared deviation from the SES reference point; keeps `ses` | Does an explicit non-monotone column beat the splits the trees would make anyway? |
| **+ size offset** | `use_exposure=True`: a weighted regression of the rate $y/n$ with weight $n$ (§3.3); `n_apartments` stays a feature | Does modeling the per-apartment rate beat letting the trees learn size from scratch? It was expected to be the largest gain. An untuned smoke run over 10 simulated populations found only weak evidence: about 4% lower deviance for kindergarten and high school, and none for elementary ([plan, Step 2.4](MODEL_REIMPLEMENTATION_PLAN.md)) |
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
  exposure. The old `FeatureSpec` refuses a column that is both (the rebuilt
  `FeatureTransformer` does not check it: §8.2), and including it as a
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
| **Base** | $z_{\text{ses}}$, $z_{\text{hh}}$, $z_{\text{age}}$, $d/2$, 3 centered shares, 2 school dummies | Log children-per-apartment rate at an average building with no daycares | The parsimonious specification everything else is measured against |
| **SES quadratic** | + $(\text{ses}-0)^2$ | $\beta_1$ is the slope at the reference point (average SES), $\beta_2$ the curvature around it | Fertility–SES gradients are commonly non-monotone, so one extra parameter is cheap insurance when modeling **how many** children |
| **Daycare centered log1p** | $d/2 \rightarrow \log\frac{1+d}{1+\bar d}$ | Elasticity: $\mu \propto (1+d)^\beta$ | Daycare count measures *access*, and access saturates — the first facility matters far more than the eighth |
| **`room_share_x_household_size`** | + 3 columns | How the room-mix effect shifts per SD of household size | Capacity meets demand: the child-count payoff of a larger apartment depends on how large local households are (Section 6.2, first choice) |
| `room_share_x_ses` | + 3 columns | How the room-mix effect shifts per SD of SES | Does extra space become more children, or more space per person? Section 6.2's second choice — needs the new interaction of Section 8 |
| Combined | the forms that won above | — | Only after the single-term candidates have established which ones earn their place |

#### Composition stage — grouped multinomial over the three cohorts, no exposure

Buildings are weighted by their observed child counts (Section 4.7).

| Variation | Columns | Reading | Why this stage |
|---|---|---|---|
| **Base** | The same design matrix, no offset | Log-odds of each cohort against the reference cohort, for an average building | As above |
| **Daycare centered log1p** | $d/2 \rightarrow \log\frac{1+d}{1+\bar d}$ | Elasticity on the cohort log-odds | Same saturating-access argument as the total stage — it is a property of the covariate, so it applies wherever daycare enters |
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
| `Normal(0, 0.5)` coefficient priors | The feature's unit decides how strongly each coefficient is shrunk. A unit near 1 SD, as recommended (Section 1), shrinks every column about equally; a unit spanning the whole range, such as $d/8$, shrinks the daycare effect about 22 times harder (Section 4.8) |
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

One structural note before the options. In the composition model **every
feature already gets its own coefficient per cohort**, so a feature × cohort
interaction is automatic and free. The consequence is that this stage is
$K$ times as parameter-hungry as the total stage for the same design matrix
(one intercept and coefficient column per cohort in a multinomial logistic
regression, the rebuilt `CohortProbabilityModel` with `LogisticRegression`,
§8.3; the old multinomial logit with a reference cohort had $K-1$, "roughly
twice" for three cohorts), and parsimony should bind harder here.

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
| `ses` | Raw; candidate: the squared deviation from the SES reference point 0.0. No spline | $z$; quadratic candidate in the **total stage only**: $(\text{ses}-0)^2$ beside $z$. No spline | 0.0 is the population mean, average SES; a fixed vertex means the same in every fold, where the fold mean moves. Quadratic is nested and interpretable, and curvature has a mechanism for *how many* children but none for *which ages*; only about 60 distinct SES values, so a 5-column spline overfits and can't be read |
| `avg_household_size` | Raw | $z$ (report $\beta/s$; fixed $(x-2.6)/0.5$ optional) | Scale sets only the unit; matches C's priors |
| `median_age` | Raw | $z$ (report $\beta/s$; fixed $(x-37)/10$ optional) | Same |
| `n_daycares_500m` | Raw; drop the `log1p` candidate | Candidate 1: $d/2$ (0 = none; per 2 daycares, about 1 SD). Candidate 2: $\log\frac{1+d}{1+\bar d}$ | A fixed unit is fold-stable where a learned min-max is not, and a unit near 1 SD is shrunk like the z-scored columns; the centered log1p keeps its unit across folds and reads as an elasticity |
| Room shares (4, 5, 6; reference 3) | Raw | $(s_k-\bar s_k)/0.10$, one common unit | 10 pp substituted out of the 3-room reference; centering puts the baseline at the average mix |
| `n_apartments` | Raw feature, and optionally the exposure (`use_exposure=True`) | Not a feature | Trees cannot extrapolate proportional growth; the feature still captures departures from proportionality |
| Exposure | Raw $n$ passed as `fit(..., exposure=n)` / `predict(..., exposure=n)`; the model fits a weighted regression of the rate $y/n$ with weight $n$ (§3.3) | $\log n$ offset, coefficient 1, unscaled | Models the per-apartment rate; `log1p` adds a size-dependent bias of about $1/n$ |
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
    Center, CenterByReferencePoint, ColumnPlan, DomainMinMax, DomainScale,
    FeatureTransformer, Interaction, OneHot, Quadratic, RelativeSaturation,
    Standardize,
)
```

**The vocabulary.** What each transform emits, what it learns at fit, and
when to choose it over its neighbour:

| Transform | Emits | Learned at fit | Choose it when |
|---|---|---|---|
| `Standardize()` | $(x - m)/s$ | $m$, $s$ | A unit per SD is wanted, and the priors were set for it (§4.5) |
| `Center()` | $x - m$ | $m$ | A linear column in its own units; the center only moves the intercept |
| `CenterByReferencePoint(reference_point=c)` | $x - c$ | Nothing | The column will be squared, or the reference is a named substantive point (SES 0.0, §3.2); it means the same in every fold |
| `DomainScale(scale=s)` | $x / s$ | Nothing | The unit comes from domain knowledge, near 1 SD: per 10 pp of share (§4.3), per 2 daycares (§4.1) |
| `DomainMinMax(minimum, maximum)` | $(x - \min)/(\max - \min)$ | Nothing | Declared bounds, no clipping. No declaration below uses it: a range is several SDs wide, so the column is shrunk harder than the rest (§4.8) |
| `Quadratic()` | $x^2$, as `<col>_squared` | Nothing | After a center; never on a raw column whose centre is far from zero |
| `Log()`, `Log1p()` | $\log x$, $\log(1 + x)$ | Nothing | $x > 0$, or $x > -1$ for `Log1p`; never after a center |
| `RelativeSaturation()` | $\log(1+x) - \log(1+\bar x)$, as `<col>_sat` | $\bar x$ | Diminishing returns on a count, read as an elasticity (§4.1); rather than `Center`, when the effect saturates |
| `OneHot(categories, reference_category)` | Dummies, the reference dropped | Nothing | Declared levels keep the columns identical across folds |

`Log1pRatioScaler` is the fitted scikit-learn estimator `RelativeSaturation`
builds. Import it directly for a pipeline of your own or an `isinstance` check;
inside a `ColumnPlan`, declare `RelativeSaturation()`.

Four rules that the declarations depend on:

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
- **A log never follows centering, standardizing or relative saturation.**
  `Log`, `Log1p` and `RelativeSaturation` may not come after `Center`,
  `Standardize` or `RelativeSaturation` in a plan's chain: a column measured
  from its mean is negative somewhere, and a log of it is a modeling mistake
  (for `Log` it is `-inf` or `nan` outright). The plan is rejected when
  declared; take the log first (`Log() → Center()`). After a step whose sign
  depends on the data (`CenterByReferencePoint`, `DomainMinMax`, `Log`) or that
  clears negatives (`Quadratic`) the plan is accepted, and a bad value is
  caught at fit or transform, by the step's own input check
  (`RelativeSaturation`) or the finite-output check.

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

### 8.1 Model A — `CountModel` (LightGBM)

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
it. The exposure belongs to the model, not the transformer: `use_exposure=True`,
with the raw count passed to `fit` and `predict`. Build it with
`ExposureTransformer` on the full table, before splitting: it rejects a zero,
negative, infinite or NaN count, which LightGBM would otherwise fit silently.
The model takes `tree` as its `feature_transformer` and fits a copy on the
rows it is fitted on.

```python
from lightgbm import LGBMRegressor
from age_group_prediction.modeling import CountModel
from age_group_prediction.preprocessing import ExposureTransformer

exposure = ExposureTransformer("n_apartments").fit_transform(table)
model = CountModel(
    estimator=LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1),
    use_exposure=True,
    feature_transformer=tree,
).fit(fit_df, fit_df["n_kindergarten"], exposure=exposure.loc[fit_df.index])
```

With a Gaussian loss (`objective="regression"`) the exposure is allowed too
(§3.3).

**Combining the cohorts.** Each cohort gets its own model, with its own copy
of the transformer, held by `IndependentCohortModels`, which fits and predicts
on the raw table and returns one column per cohort. The same exposure goes to
every cohort; a model without the offset ignores it. The data flow and the
rules are in
[Independent cohort models §3](INDEPENDENT_COHORT_MODELS.md#3-independentcohortmodels).

```python
from age_group_prediction.modeling import IndependentCohortModels

COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]
cohort_models = IndependentCohortModels({
    cohort: CountModel(
        estimator=LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1),
        use_exposure=True,
        feature_transformer=tree,
    )
    for cohort in COHORTS
}).fit(fit_df, fit_df[COHORTS], exposure=exposure.loc[fit_df.index])
predictions = cohort_models.predict(valid_df, exposure=exposure.loc[valid_df.index])
```

**SES-quadratic variant.** Add one plan; `ses` itself stays, because the squared
column earns its place only by being non-monotone (§3.2). It squares the
deviation from the SES reference point 0.0, average SES, not from the fold
mean:

```python
ColumnPlan(
    name="ses_sq",
    columns="ses",
    transforms=(CenterByReferencePoint(reference_point=0.0), Quadratic()),
)
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
            transforms=(DomainScale(scale=2.0),),
        ),
        ColumnPlan(name="school", columns="school_status", transforms=(SCHOOL,)),
    ),
)
```

Reading the coefficients: per SD for the z-scored neighborhood numerics (§4.5);
per 10 percentage points of share for the room shares (§4.3); per 2 daycares
for the daycare term (§4.1); contrasts against `none` for the school dummies
(§4.6).

Unlike Model A, B and C do not list `n_apartments` among the plans, so building
size enters **only** as the offset (§4.4). The package does not enforce that;
the declaration is what expresses it. The exposure is not the transformer's:
the model takes it separately, as `exposure=`, where it cannot be mistaken for
a predictor, and forms the offset $\log n$ itself. Build it with
`ExposureTransformer` on the full table before splitting, as in §8.1.

```python
from age_group_prediction.modeling import CountModel, NegativeBinomialRegressor
from age_group_prediction.preprocessing import ExposureTransformer

exposure = ExposureTransformer("n_apartments").fit_transform(table)
total_model = CountModel(
    estimator=NegativeBinomialRegressor(),  # or PoissonRegressor(alpha=...): the same CountModel
    use_exposure=True,
    feature_transformer=total_base,
).fit(fit_df, fit_df["n_children_total"], exposure=exposure.loc[fit_df.index])
total_mean = total_model.predict(valid_df, exposure=exposure.loc[valid_df.index])
```

The total model is the same `CountModel` as Model A's, given the total column.
An estimator whose `fit` takes `exposure` (`NegativeBinomialRegressor`) gets the
raw exposure and forms the offset itself; one taking `sample_weight`
(`PoissonRegressor`, LightGBM) gets the weighted per-apartment rate, which is the
Poisson offset model exactly ([Independent cohort models §2.1](INDEPENDENT_COHORT_MODELS.md#21-the-exposure-as-a-weighted-rate)).

### 8.3 Model B — composition stage

**The same plans, and a model with no exposure** (§4.7): age-group shares do
not depend on building size. The model is a multi-class classifier of the
cohort probabilities (`CohortProbabilityModel`), fitted on the children as
categorical rows, one weighted row per (building, cohort); with
`LogisticRegression` it has one intercept and one coefficient per feature
**for every cohort**, so this stage has $K$ times the total stage's
parameters for the same design matrix and parsimony binds harder (§6.3).
*Renamed from `composition_base` on 2026-10-01, after the model it feeds.*

```python
from sklearn.linear_model import LogisticRegression
from age_group_prediction.modeling import CohortProbabilityModel

cohort_probability_base = FeatureTransformer(
    plans=total_base.plans,      # identical; only the model differs, taking no exposure
)
probability_model = CohortProbabilityModel(
    estimator=LogisticRegression(),  # or LGBMClassifier(...), RandomForestClassifier(...)
    calibration_method="temperature",  # None: the classifier's own probabilities
    feature_transformer=cohort_probability_base,
).fit(fit_df, fit_df[["n_kindergarten", "n_elementary", "n_highschool"]])
shares = probability_model.predict(valid_df)  # a DataFrame; rows sum to 1
```

**Combining the two.** `TotalTimesProbabilityModel` holds both models, fits
them on the raw table (the total on the row sum of `y`) and predicts
`total × probabilities`, one column per cohort. The same exposure goes to
both; the probability model ignores it. The derivations (the categorical
rows, the calibration folds, the exposure cases), the data flow and the
rules are in
[Total times probability model](TOTAL_TIMES_PROBABILITY_MODEL.md).

```python
from age_group_prediction.modeling import TotalTimesProbabilityModel

COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]
model_2 = TotalTimesProbabilityModel(
    total_model=CountModel(
        estimator=NegativeBinomialRegressor(), use_exposure=True, feature_transformer=total_base
    ),
    probability_model=CohortProbabilityModel(
        estimator=LogisticRegression(), feature_transformer=cohort_probability_base
    ),
).fit(fit_df, fit_df[COHORTS], exposure=exposure.loc[fit_df.index])
predictions = model_2.predict(valid_df, exposure=exposure.loc[valid_df.index])
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
| 1 | base | `total_base` | `cohort_probability_base` |
| 2 | `total__ses_quadratic` | + the plan in 8.5.1 | base |
| 3 | `total__daycare_log1p` | daycare plan replaced, 8.5.2 | base |
| 4 | `total__room_share_x_household_size` | + the interactions in 8.5.3 | base |
| 5 | `composition__daycare_log1p` | base | daycare plan replaced, 8.5.2 |
| 6 | `composition__room_share_x_median_age` | base | + the interactions in 8.5.4 |
| 7 | combined | whichever total forms won | whichever composition forms won |

#### 8.5.1 SES quadratic (candidate 2)

Square the deviation from the SES reference point 0.0, as in Model A: $\beta_1$,
on the base plan's $z$, is then the slope at the reference point, and
$\beta_2$ the curvature around it, per population SD² (§4.2). `ses` keeps its
own plan; this one is added beside it.

```python
ColumnPlan(
    name="ses_sq",
    columns="ses",
    transforms=(CenterByReferencePoint(reference_point=0.0), Quadratic()),
)
```

#### 8.5.2 Daycare, centered `log1p` (candidates 3 and 5)

Replaces the daycare plan rather than joining it. `RelativeSaturation` emits
$\log(1+d) - \log(1 + \bar d)$ — `log1p` of the mean, not the mean of `log1p` —
so the reference stays in the original counts and the coefficient is an
elasticity (§4.1, candidate 2). Its input must be greater than −1, where
`log1p` is defined; anything else is rejected at fit. The fitted estimator is
the public `Log1pRatioScaler`.

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

**Center the linear daycare column before using it as an operand.** $d/2$ has
mean 1.4, so its product with a room share correlates 0.85 with that share's
main effect and has SD 1.3 to 1.9. With `Center() → DomainScale(scale=2.0)`
the correlation is about 0 and the SD 0.66 to 0.99. The cost is the
"no daycare" baseline (§4.1). `n_daycares_500m_sat` is already centered.

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
```

### 8.7 Still Outstanding

What this section replaced was an implementation checklist. Some of its items
are now satisfied by the package — the daycare, room-share and centering forms of
its item 2 are expressed by `DomainMinMax`, `RelativeSaturation`,
`Center`+`DomainScale` and `CenterByReferencePoint`+`Quadratic`; and its item
4, adding entries to a closed `_VALID_INTERACTIONS` literal, is replaced by
open user-named `Interaction`s. **The rest still stand:**

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
3. **`CountModel`:** done in the rebuilt
   `modeling.CountModel`, which takes `use_exposure=True` and fits
   a weighted regression of the rate $y/n$ with weight $n$ (§3.3)
   ([INDEPENDENT_COHORT_MODELS.md](INDEPENDENT_COHORT_MODELS.md#2-countmodel)).
4. **Model C:** no candidate changes of its own. Rerun the prior-predictive
   checks after any unit or baseline change, and revisit
   `total_intercept_loc = -2` against the observed log rate of about −0.5.
5. **Provenance:** every change above alters feature specs and therefore
   candidate fingerprints. Compare old and new forms as separate candidates on
   identical folds.
6. **Non-finite constants in the fixed-anchor transforms:**
   `DomainScale(scale=inf)` and `DomainMinMax(minimum=nan, maximum=nan)` are
   accepted at declaration and fail at fit. `CenterByReferencePoint` already
   rejects a non-finite reference (`Field(allow_inf_nan=False)`); apply the
   same to both siblings.
7. **Penalty ranges:** in the scale audit (§4.8) held-out fit was best at the
   edge of both tuned ranges, `total_l2_penalty = 1.0` and
   `probability_c = 0.01`, and total-stage deviance fell from 4.09 unpenalized
   to about 4.00 there, more than any scaling choice moved it. The audit
   scored on independent populations, not on the project's cross-validation,
   so this is a prompt to check the ranges in the tuning work, not a
   conclusion. *In the rebuilt models (2026-10-08) the penalties are the
   estimators' own: `PoissonRegressor(alpha)` on the total, where under the
   weighted rate `alpha` acts as `alpha × mean(exposure)` of the offset model
   (scikit-learn normalizes `sample_weight`;
   [Independent cohort models §2.1](INDEPENDENT_COHORT_MODELS.md#21-the-exposure-as-a-weighted-rate));
   `NegativeBinomialRegressor` is unpenalized; `LogisticRegression(C)` on the
   probability model is per **child**, as the old `probability_c` was (its
   range $[0.01, 100]$ applies again). History: PR #11's torch build had one
   `l2_penalty` per building in both stages, and its B7 smoke run found the
   unpenalized Poisson total overfitting one population of ten (deviance 6.90
   against the constant rate's 4.69; `l2_penalty = 1.0` pulled it back to
   5.29). With sklearn's `PoissonRegressor()` default `alpha=1` that population
   gives 3.60 ([plan, sub-task 6](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md)).*

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
