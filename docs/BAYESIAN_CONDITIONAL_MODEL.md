# Bayesian Conditional Model (Model C)

For a shorter summary without Pyro syntax, tensor shapes, and implementation
details, read
[BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md](BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md).
This page is the full technical guide.

Companion model descriptions:
[DIRECT_COHORT_MODEL.md](DIRECT_COHORT_MODEL.md) (Model A) and
[INDEPENDENT_TOTAL_PROBABILITY_MODEL.md](INDEPENDENT_TOTAL_PROBABILITY_MODEL.md)
(Model B). Operational sections for this model start at
[Selection And Comparison](#selection-and-comparison).

## Purpose

`BayesianConditionalModel` predicts the total number of children in each
building and divides that total among the configured age cohorts. It is one
public model with two separate Bayesian fits:

1. a hierarchical negative-binomial model for the total count; and
2. a Dirichlet-multinomial model for cohort composition conditional on the
   observed total.

Both stages use PyTorch tensors, Pyro model primitives, and NUTS inference. The
public estimator lifecycle is in
[`src/age_group_prediction/models/bayesian_conditional.py`](../src/age_group_prediction/models/bayesian_conditional.py);
the Pyro model definitions and prior-predictive simulation are in
[`src/age_group_prediction/models/bayesian_components.py`](../src/age_group_prediction/models/bayesian_components.py);
and NUTS execution and sampler diagnostics are in
[`src/age_group_prediction/models/bayesian_inference.py`](../src/age_group_prediction/models/bayesian_inference.py).
Runtime settings and priors are defined by `BayesianConditionalConfig` and the
`bayesian_*` sections of [`configs/modeling.toml`](../configs/modeling.toml).
Feature forms are not configured there. The total spec is the `feature_spec`
passed to `fit`, and the composition spec is the `probability_feature_spec`
constructor argument. The plan requires both to be Model B's frozen forms, but
nothing in this model enforces that locally. Gate 6's
`run_cross_model_validation(...)` rejects the selection freeze unless the
selected `BayesianConditionalModel` and
`IndependentTotalProbabilityModel` candidates declare identical total and
probability feature specifications. Likewise, `[randomness] default_seed`
remains outside the estimator: Gate 6 resolves it as the experiment master
seed and passes purpose-specific generators to the public model lifecycle.

## Statistical Model

Let $b$ identify a building, $g[b]$ its neighborhood, $x_b$ its total-count
features, and $z_b$ its composition features.

### Total stage

The total number of children follows the NB2 parameterization

$$
Y_b \mid \mu_b, \phi \sim \operatorname{NB2}(\mu_b, \phi),
\qquad
\operatorname{Var}(Y_b)=\mu_b+\frac{\mu_b^2}{\phi}.
$$

Its log mean is

$$
\log \mu_b
= \log n_{b,\mathrm{apartments}}
+ \beta_0 + x_b^\mathsf{T}\beta + u_{g[b]}.
$$

The coefficient-one `log(n_apartments)` term is an exposure offset rather than
an estimated feature coefficient. Neighborhood effects use a non-centered
parameterization:

$$
u_g = \sigma_u r_g,
\qquad
r_g \sim \mathcal N(0,1),
\qquad
\sigma_u \sim \operatorname{HalfNormal}(s_u).
$$

Sampling a standardized raw effect separately from its scale often gives NUTS
a more favorable posterior geometry than sampling $u_g$ directly.

The code calls the NB2 concentration `phi`. Shared SciPy utilities instead use
the reciprocal dispersion

$$
\alpha=\frac{1}{\phi}.
$$

The equivalent SciPy and NumPy parameters are

$$
r=\phi,
\qquad
p=\frac{\phi}{\phi+\mu}.
$$

### PyTorch `NegativeBinomial` parameters

In the PyTorch call below, `total_count` does **not** mean the observed total
number of children. It is PyTorch's name for the negative-binomial shape
parameter $r$. In this model, that shape parameter is exactly the NB2
concentration:

```python
dist.NegativeBinomial(
  total_count=concentration,
  logits=log_mean - log_concentration,
)
```

Therefore,

$$
\texttt{total\_count}=r=\phi
=\texttt{concentration}.
$$

The observed building total is instead the random outcome $Y_b$. When the
model is generating data, that outcome is named `total_count` at the Pyro
sample site. The similar names refer to different things:

| Code name | Meaning |
|---|---|
| Distribution argument `total_count` | NB shape/concentration parameter $r=\phi$ |
| Pyro site `"total_count"` | Random or observed building child total $Y_b$ |

PyTorch's `logits` parameter is the log odds of its `probs` parameter,
$\texttt{probs}=\operatorname{sigmoid}(\texttt{logits})$. Writing
$p=1-\texttt{probs}$ for the complementary (SciPy-style) success probability,

$$
\ell=\log\left(\frac{1-p}{p}\right).
$$

For this parameterization, the mean is

$$
E[Y]=r\exp(\texttt{logits}).
$$

The Bayesian NB2 + Dirichlet-Multinomial model needs that mean to be $\mu$, so
it chooses

$$
\ell=\log\mu-\log\phi
=\log\left(\frac{\mu}{\phi}\right).
$$

Substituting $r=\phi$ gives

$$
E[Y]=\phi\exp\left(\log\mu-\log\phi\right)=\mu.
$$

It also recovers the desired success probability

$$
p=\operatorname{sigmoid}(-\texttt{logits})
=\frac{\phi}{\phi+\mu},
$$

and thus the desired NB2 variance $\mu+\mu^2/\phi$. Passing `logits` avoids
forming this probability directly and is numerically stable when $\mu$ or
$\phi$ varies substantially.

### Composition stage

For $K$ cohorts, the final cohort is the reference category. Only $K-1$ logits
are estimated:

$$
\eta_{b,k}=a_k+z_b^\mathsf{T}\gamma_k
\quad (k=1,\ldots,K-1),
\qquad
\eta_{b,K}=0,
$$

$$
p_{b,k}=\operatorname{softmax}(\eta_b)_k.
$$

Fixing the final logit at zero makes the parameterization identifiable. With
the default schema order (`n_kindergarten`, `n_elementary`, `n_highschool`),
the final and therefore reference cohort is `n_highschool`. The specification
requires an identification convention but does not prescribe a particular
cohort, so this choice is aligned with it. Cohort counts then follow

$$
\mathbf C_b \mid Y_b, \mathbf p_b, \kappa
\sim \operatorname{DirichletMultinomial}
\left(Y_b,\kappa\mathbf p_b\right).
$$

The scalar $\kappa>0$ controls variation around the mean composition. Larger
values approach ordinary multinomial variation; smaller values permit more
building-to-building variation. This stage deliberately has no neighborhood
random effect.

#### Composition stage summary

1. The regression and reference-category softmax produce the expected cohort
   proportions $\mathbf p_b$, which are positive and sum to one.
2. The model infers one positive concentration parameter
   $\kappa=\exp(\log\kappa)$ from the observed cohort compositions.
3. Pyro receives the concentration vector $\kappa\mathbf p_b$. Its components
   sum to $\kappa$, its expected proportions remain $\mathbf p_b$, and
   $\kappa$ controls how tightly building compositions follow those expected
   proportions.
4. The Dirichlet-multinomial allocates the building total $Y_b$ among the
   cohorts, guaranteeing that the resulting cohort counts sum to $Y_b$.

During training, the composition model conditions on observed cohort vectors,
each of which determines its own total by construction. Conditioning on the
observed total is therefore carried entirely by the observed cohort vector's
sum, not by the `total_count` argument passed to
`dist.DirichletMultinomial(...)`: Pyro's `DirichletMultinomial.log_prob`
ignores `total_count` entirely and only uses it to build the distribution's
`sample()` path. This was verified directly — inflating `total_count` by a
constant factor while keeping the observed cohort vector fixed left the
log-probability unchanged. `_fit_model` in `bayesian_conditional.py` now
asserts that every observed cohort vector sums to its building's observed
total before fitting, so this equivalence cannot silently fail to hold for the
training data. The two stages are fitted separately, so their joint posterior
is represented as the product of the two fitted posteriors. Under that
design,

$$
E[\mu_b p_{b,k}]
=E[\mu_b]E[p_{b,k}],
$$

which is why point cohort means are calculated as the product of the two
posterior means. Posterior predictive simulation independently selects a total
posterior state and a composition posterior state for the same reason.

## Pyro Syntax Used Here

### `pyro.sample`

`pyro.sample(name, distribution)` creates a named random variable, called a
sample site. NUTS treats unobserved continuous sample sites as parameters to
infer. For example:

```python
intercept = pyro.sample(
    "total_intercept",
    dist.Normal(priors.total_intercept_loc, priors.total_intercept_scale),
)
```

Passing `obs=value` tells Pyro that this value was observed rather than being
an unknown value to sample. Pyro evaluates the distribution's `log_prob` at
that value and adds it to the model's joint log probability. For an outcome
site, this is the likelihood contribution:

```python
pyro.sample("cohort_count", distribution, obs=observed)
```

Intuitively, without `obs`, the model says "generate a cohort count from this
distribution." With `obs`, it says "score the cohort count that was actually
observed under this distribution." NUTS then favors parameter values that give
the observed data higher probability. In the Bayesian NB2 +
Dirichlet-Multinomial model, `cohort_count` uses this standard observed-site
likelihood. The total stage uses `pyro.factor` instead because its NB2
likelihood is evaluated with a custom but equivalent formula.

In practice, `cohort_count` is observed-only for this model's buildings.
Pyro's `DirichletMultinomial.sample()` raises `NotImplementedError` for
heterogeneous (per-row) `total_count`, and building totals vary by
construction, so the generative (no-`obs`) path is not usable here even
though it is syntactically available. Prior- and posterior-predictive cohort
generation instead use the explicit Dirichlet-then-multinomial construction
described under [Prior-predictive simulation](#prior-predictive-simulation)
and [Posterior prediction](#posterior-prediction), which is mathematically
equivalent and does not depend on that sampler path. The total stage's
`total_count` site does not have this limitation because PyTorch's
`NegativeBinomial.sample()` supports heterogeneous `total_count`.

### `to_event`

PyTorch distributions distinguish batch dimensions from event dimensions.
`to_event(n)` reinterprets the rightmost $n$ batch dimensions as one dependent
event for log-probability reduction.

```python
dist.Normal(torch.zeros(feature_count), scale).to_event(1)
```

An intuitive way to read the dimensions is:

- a **batch dimension** means "several distributions evaluated side by side";
- an **event dimension** means "several values that together form one draw at
  this named sample site."

Suppose a regression has three coefficients. Before `to_event`,
`dist.Normal(torch.zeros(3), 1)` behaves like a batch of three scalar Normal
distributions. Its value has shape `(3,)`, and `log_prob(value)` also has shape
`(3,)`, with one log probability per coefficient. After `to_event(1)`, the
rightmost one dimension becomes the event:

```text
value shape:             (3,)
event shape:             (3,)
log_prob(value) shape:   ()
```

The three component log probabilities are summed, so Pyro sees one named
coefficient-vector draw.

Now suppose the composition coefficients form a matrix with four features and
two modeled cohorts, shape `(4, 2)`. Using `to_event(2)` turns both rightmost
dimensions into one matrix-valued event:

```text
value shape:             (4, 2)
event shape:             (4, 2)
log_prob(value) shape:   ()
```

With `to_event(1)`, only the final cohort dimension would be an event. The
feature dimension would remain a batch dimension, and `log_prob(value)` would
have shape `(4,)`. The composition stage uses `to_event(2)` because the
complete coefficient matrix is one latent parameter at the
`composition_coefficients` site.

`to_event` does **not** make independent Normal components statistically
correlated. It tells Pyro how to interpret dimensions, sum log probabilities,
and validate model shapes. Dependence would need to come from the distribution
itself, such as a multivariate Normal with a non-diagonal covariance matrix.

### `pyro.deterministic`

`pyro.deterministic(name, value)` records a derived value in the execution
trace without adding probability mass. Yes: the recorded value is a
deterministic transformation of values already in the model. For example,
`total_concentration = exp(log_total_concentration)` is completely determined
once `log_total_concentration` has been sampled.

A deterministic site is therefore not another uncertain parameter and does
not add a prior or likelihood term. It can still vary across posterior or prior
draws because its sampled inputs vary. The model records total log means,
concentration, and composition probabilities so `Predictive` can return these
useful derived quantities by name.

### `pyro.factor`

`pyro.factor(name, log_factor)` adds an already-computed log-probability term to
the model. The observed total stage uses an exact closed-form NB2 log mass:

$$
\begin{aligned}
\log p(y\mid\mu,\phi)
={}&\log\Gamma(y+\phi)-\log\Gamma(\phi)-\log\Gamma(y+1)\\
&+\phi\left[\log\phi-\log(\phi+\mu)\right]
+y\left[\log\mu-\log(\phi+\mu)\right].
\end{aligned}
$$

Using `pyro.factor` here is statistically equivalent to observing a Torch
`NegativeBinomial` with `total_count=phi` and
`logits=log(mu)-log(phi)`. It avoids a native runtime failure observed for the
distribution's `log_prob` path on the target macOS ARM/Python 3.13 environment.

Intuitively, NUTS compares candidate parameter values using the model's total
log probability. `pyro.factor` says "add this score to that total." A candidate
that predicts the observed totals well receives a less negative NB2 log score;
a candidate that predicts them poorly receives a more negative score and is
less likely to be retained.

Unlike `pyro.sample(..., obs=value)`, a factor has no distribution object and
no observed or generated value. The implementation is responsible for
computing the correct log-probability expression. This makes `factor` useful
for custom likelihoods, corrections, constraints, or other terms that are
awkward to express as a standard Pyro distribution. Here it represents the
ordinary observed-total likelihood, not an additional penalty or an
approximation.

### `Predictive`

`pyro.infer.Predictive` executes a model repeatedly and returns selected named
sites. `_run_prior_predictive` calls it without posterior samples, so every
latent site is drawn from its prior. The deterministic sites provide the
parameters needed by the explicit outcome samplers.

### `NUTS` and `MCMC`

NUTS is an adaptive Hamiltonian Monte Carlo kernel. It explores a continuous
posterior using gradients and automatically chooses trajectory lengths. Pyro's
`MCMC` driver handles warmup, adaptation, and retained posterior samples.

The Bayesian NB2 + Dirichlet-Multinomial model runs each configured chain
sequentially. The outer loop runs `profile.chains` separate Pyro `MCMC`
instances, each with `num_chains=1`, and then stacks their samples. Therefore,
the effective chain count is the TOML-configured value: two for the reduced
profile or four for the full profile, not one. Sequential execution permits a
small `InstrumentedNUTS` subclass to observe recursive `_build_tree` calls and
record realized tree depth for every chain. The stacked, chain-aware tensors
are used for R-hat and effective sample size calculations, then flattened for
ordinary prediction.

## Named Sites

| Site | Stage | Kind | Meaning |
|---|---|---|---|
| `total_intercept` | Total | Latent scalar | Baseline log rate after the exposure offset |
| `total_coefficients` | Total | Latent vector | Fixed effects for transformed total features |
| `log_total_concentration` | Total | Latent scalar | Log NB2 concentration $\log\phi$ |
| `neighborhood_scale` | Total | Latent scalar | Population scale $\sigma_u$ for neighborhood effects |
| `neighborhood_raw` | Total | Latent vector | Standard-normal non-centered effects $r_g$ |
| `total_log_mean` | Total | Deterministic vector | Building log means $\log\mu_b$ |
| `total_concentration` | Total | Deterministic scalar | Positive concentration $\phi$ |
| `total_count_log_likelihood` | Total | Factor | Summed observed NB2 log mass |
| `total_count` | Total | Outcome vector | Generated totals when no observations are supplied |
| `composition_intercepts` | Composition | Latent vector | Intercepts for the $K-1$ modeled cohorts |
| `composition_coefficients` | Composition | Latent matrix | Feature effects for the $K-1$ modeled cohorts |
| `log_composition_concentration` | Composition | Latent scalar | Log Dirichlet concentration $\log\kappa$ |
| `composition_probabilities` | Composition | Deterministic matrix | Reference-softmax cohort probabilities |
| `cohort_count` | Composition | Observed-only matrix | Cohort vector conditioned on each total |

## Tensor Shapes

Let $N$ be the number of buildings, $P_t$ the number of total features, $P_c$
the number of composition features, $G$ the number of known neighborhoods,
$K$ the number of cohorts, and $S$ the number of flattened posterior samples.

### Model inputs

| Value | Shape |
|---|---|
| Total features | `(N, P_t)` |
| Composition features | `(N, P_c)` |
| Log exposure | `(N,)` |
| Neighborhood index | `(N,)` |
| Observed totals | `(N,)` |
| Observed cohorts | `(N, K)` |

### Flattened posterior arrays

| Value | Shape |
|---|---|
| Total intercept | `(S,)` |
| Total coefficients | `(S, P_t)` |
| Log total concentration | `(S,)` |
| Neighborhood scale | `(S,)` |
| Raw neighborhood effects | `(S, G)` |
| Composition intercepts | `(S, K - 1)` |
| Composition coefficients | `(S, P_c, K - 1)` |
| Log composition concentration | `(S,)` |

Prediction computes total means with shape `(S, N)` and composition
probabilities with shape `(S, N, K)`. Public predictive payloads use
`(N, D)`, where $D$ is the requested number of draws. Prediction intervals use
`(N, L, 2)` for $L$ interval levels and lower/upper endpoints.

## Execution Flow

### Hyperparameters and tuning

Unlike Models A and B, this model runs **no Optuna hyperparameter search**.
The quantities that play that role are fixed, reviewed configuration:
- priors: `[bayesian_priors]`;
- NUTS settings (`[bayesian_reduced_profile]`, `[bayesian_full_profile]`):
  chains, warmup, samples, target acceptance, and maximum tree depth;
- diagnostic thresholds: `[bayesian_*_diagnostics]`;
- prior-predictive checks and numerical stabilization:
  `[bayesian_prior_predictive]`, `[bayesian_stabilization]`.

NUTS adapts its step size and mass matrix during warmup, but that is sampler
adaptation, not model selection. Feature forms are not searched either:
cross-validation requires this model to reuse the independent model's selected
total and probability feature specs. Cross-validation uses the reduced profile;
the final refit is forced to the full profile. Persistent diagnostic failures
call for changing priors, profiles, or parameterization in the configuration,
which creates a new candidate rather than a tuned variant.

### Fitting

1. The base model fits the total feature transformer and produces the exposure
   offset.
2. The Bayesian NB2 + Dirichlet-Multinomial model fits a separate composition
  feature transformer.
3. Neighborhood labels are mapped to integer random-effect indices.
4. Prior-predictive checks run before observed-likelihood inference.
5. NUTS fits the total stage.
6. NUTS fits the composition stage using observed totals.
7. Both stages' diagnostics and the prior-predictive summary are recorded
   before any diagnostic policy is enforced. The prior-predictive policy is
   applied earlier, before NUTS; under its `action = "error"` a violation
   raises before the summary is stored.
8. Each stage's diagnostics are checked against the active policy. Under
   `action = "error"` a failing stage raises a `RuntimeError` that carries
   `stage`, `diagnostics`, and `failures` attributes, because the base class
   then clears fitted state and the evidence would otherwise be lost.
9. Posterior tensors become fitted state only after both stages pass the
   required policy.

### Prior-predictive simulation

The prior total path uses the exact Gamma-Poisson representation of NB2:

$$
\lambda\sim\operatorname{Gamma}
\left(\text{shape}=\phi,\text{scale}=\mu/\phi\right),
\qquad
Y\sim\operatorname{Poisson}(\lambda).
$$

The installed Pyro version cannot sample heterogeneous row-wise totals from
`DirichletMultinomial.sample()`. The implementation therefore uses the exact
generative decomposition

$$
q_b\sim\operatorname{Dirichlet}(\kappa p_b),
\qquad
C_b\sim\operatorname{Multinomial}(Y_b,q_b).
$$

Zero-total rows receive an all-zero cohort vector and are excluded from cohort
share quantiles because their shares are undefined. The checks require finite,
nonnegative totals and cohorts, positive concentrations, exact reconciliation,
at least one positive total, and plausible children-per-apartment rates.

The plausibility check does not threshold realized cohort shares. Their
quantiles are legitimately dispersed by the Dirichlet concentration prior, so
thresholding them would conflate that modeled dispersion with a genuinely
degenerate prior. Realized `cohort_share_quantiles` are reported in the
summary for inspection only.

Four statistics can raise a violation. Only the middle two have power over a
degenerate *composition* prior:

| Statistic | Violation when | Detects |
|---|---|---|
| 95th percentile of `children_per_apartment_quantiles` | above `maximum_children_per_apartment` | Total-stage priors implying implausible child densities |
| `expected_dominant_share` | above `maximum_expected_dominant_share` | Diffuse composition logit priors that push individual buildings toward a single cohort |
| `concentration_quantile` | below `minimum_concentration_quantile` | A $\kappa$ prior with enough mass near zero to make individual buildings effectively single-cohort |
| `expected_cohort_shares` as a multiple of $1/K$ | outside `[minimum_expected_share_ratio, maximum_expected_share_ratio]` | Only a prior that systematically favors or excludes one cohort; see below |

`expected_dominant_share` is $E[\max_k p_{b,k}]$, averaged over prior draws
and rows. `concentration_quantile` is the 5th percentile of the prior $\kappa$
draws.

The expected-share ratio check is kept, but it is not a test of prior
degeneracy and, with the current priors and three cohorts, it cannot fire. It
averages $p_{b,k}$ over draws and rows, and because the composition logit
priors are zero-mean, that average stays near uniform however diffuse the
priors are. As the logit scale grows, the reference cohort's expected share
tends to $2^{-(K-1)}$, so for $K=3$ the ratios tend to about 0.75 and 1.125,
well inside $[0.25, 2.0]$. The lower bound becomes reachable only from $K=6$
cohorts, where $K\,2^{-(K-1)}<0.25$. $\kappa$ does not enter the statistic at
all. Measured through `_run_prior_predictive`: with the composition intercept
and coefficient prior scales raised to 100 and 20, the ratios stayed within
0.69-1.19 and raised no expected-share violation, while
`expected_dominant_share` rose to 0.994 and fired. A $\log\kappa$ prior of
$\operatorname{Normal}(-3,1)$ left every share statistic unchanged and was
caught only by `concentration_quantile` (0.012).
`test_prior_predictive_checks_detect_degenerate_priors` pins one pathological
prior per active statistic.

The `[bayesian_prior_predictive]` config section sets `draws`, `action`,
`maximum_children_per_apartment`, `minimum_expected_share_ratio`,
`maximum_expected_share_ratio`, `maximum_expected_dominant_share`, and
`minimum_concentration_quantile`. The current configuration uses 200 draws, a
maximum 95th-percentile rate of 10 children per apartment, expected-share
ratios of 0.25 and 2.0, a maximum expected dominant share of 0.85, and a
minimum $\kappa$ 5th percentile of 1.0. On the full-profile acceptance data
(see [Diagnostics and Profiles](#diagnostics-and-profiles)), the configured
priors gave an expected dominant share of 0.661, a $\kappa$ 5th percentile of
1.521, children-per-apartment quantiles of $[0.0, 0.077, 2.468]$, and no
violations.

Prior-predictive violations use their own `action` setting
(`BayesianPriorPredictiveConfig.action`, default `warn`) rather than the
per-profile convergence `action`. Prior plausibility depends on the priors and
design matrix, not on the sampling budget of the active inference profile, so
the two are configured and evaluated independently.

### Posterior prediction

For each requested predictive draw:

1. choose one total posterior sample;
2. choose one composition posterior sample independently;
3. sample an NB2 total using its posterior mean and concentration;
4. sample latent cohort shares from a Dirichlet distribution;
5. sample the cohort vector with a multinomial conditional on that total; and
6. assert that cohort counts sum exactly to the sampled total.

There is no rounding, rescaling, or reconciliation repair. NumPy stores the
result arrays as floating-point values, but every generated value is
integer-valued by construction.

Known neighborhoods use their fitted posterior effects. An unseen neighborhood
draws from the fitted population distribution
$\mathcal N(0,\sigma_u)$, and fallback use is recorded in metadata.

## Diagnostics and Profiles

The active profile selects both NUTS settings and a diagnostic action.

The profile's `chains` value is the total number of independent chains. The
fixed `num_chains=1` passed to each Pyro `MCMC` object only controls one
sequential invocation and is deliberately not a separate TOML setting.

| Setting | Reduced profile | Full profile |
|---|---:|---:|
| Chains | 2 | 4 |
| Warmup steps per chain | 150 | 1,000 |
| Posterior samples per chain | 150 | 1,000 |
| Target acceptance | 0.9 | 0.9 |
| Maximum tree depth | 10 | 10 |
| Failure action | Warn | Error |

The reduced profile is intended for cheaper comparisons and smoke-scale work;
its 150 warmup and 150 retained draws per chain are not the preferred final-fit
budget. The full profile's four chains, 1,000 warmup steps, and 1,000 retained
draws per chain are reasonable starting values for final inference, subject to
the diagnostics below. A target acceptance of 0.9 is conservative for this
hierarchical model, maximum tree depth 10 is a standard starting cap, and a
diagonal mass matrix (`full_mass = false`) avoids the adaptation cost of a dense
matrix. These settings are not guarantees: divergences, poor R-hat or ESS, or
repeated tree-depth saturation should trigger retuning or reparameterization.

Both policies require no more than 0 divergences, R-hat no greater than 1.05,
a minimum per-chain mean accept probability of at least 0.6, a maximum
per-chain mean accept probability of at most 0.98, and tree-depth saturation
no greater than 0.05. Minimum effective sample size is 50 for the reduced
profile and 100 for the full profile. A diagnostic that is unavailable
(`None` or non-finite) fails the policy rather than passing it.

| Diagnostic | Short explanation |
|---|---|
| Worst R-hat | The largest split-chain R-hat across sampled parameters. Values near 1 indicate that chains have mixed to similar distributions; larger values suggest non-convergence. |
| Minimum effective sample size (ESS) | The smallest estimated number of effectively independent draws across sampled parameters after accounting for autocorrelation. Larger values mean more reliable posterior estimates. |
| Divergences | The number of NUTS transitions that could not accurately follow the posterior geometry. Any divergence can indicate biased exploration and requires investigation. |
| Minimum mean accept probability | The smallest per-chain running mean of the Metropolis accept probability that `target_accept_prob` adapts step size against. This is distinct from Pyro's `"acceptance rate"` diagnostic, which counts multinomial-sampler moves and sits at approximately 1.0 for essentially every NUTS run regardless of sampler health, so it cannot detect a badly behaved sampler. A low mean accept probability instead suggests the step size is too large for the local posterior geometry. |
| Maximum mean accept probability | The largest per-chain mean accept probability. With a target of 0.9, the 0.6 floor is almost never reached by a real failure; the failure mode seen in hierarchical models is the opposite, a chain whose step size collapses in the neck of a funnel and then accepts nearly everything. A chain above 0.98 is treated as that signature. |
| Maximum observed tree depth | The deepest NUTS trajectory reached during retained sampling. Reaching the configured maximum frequently can mean that the sampler needs longer trajectories or that the posterior is difficult to explore. |
| Tree-depth saturation | The fraction of retained iterations whose trajectory reached the configured maximum tree depth. Values near zero are preferred. If every recorded depth is zero, the recursive `_build_tree` hook never fired, for example because a Pyro upgrade changed how NUTS builds trajectories. Both tree-depth fields are then reported as `None`, which fails the policy instead of passing as a healthy zero. |

R-hat and ESS come from `pyro.infer.mcmc.util.diagnostics` in Pyro 1.9.1,
which uses `split_gelman_rubin` and the classic autocorrelation-based
`effective_sample_size`. These are the pre-2021 definitions, not the
rank-normalized split-R-hat or bulk and tail ESS of Vehtari et al. (2021). The
older R-hat compares means and variances of untransformed draws, so it is less
sensitive to chains that disagree in scale or in the tails. That is precisely
how a poorly explored `neighborhood_scale` funnel tends to show up, so an
R-hat of at most 1.05 on this statistic is weaker evidence than the same
number on the rank-normalized version. Pyro also does not compute E-BFMI, the
usual companion to divergences for funnels, and this implementation does not
add it.

"Worst" R-hat and "minimum" ESS or mean accept probability deliberately report
the least favorable result, so a healthy aggregate cannot hide a problematic
parameter or chain. The target acceptance and maximum tree depth in the profile
are sampler settings used to interpret these diagnostics, not convergence
diagnostics themselves.

The mean accept probability is captured inside `InstrumentedNUTS.sample()`,
per chain, immediately after each iteration; it cannot be read after the run
because `MCMC.cleanup()` resets it.

Metadata records the worst R-hat, minimum ESS, divergence count, minimum and
maximum per-chain mean accept probability, configured and observed tree depth,
saturation, runtime, seed, chain count, warmup, retained samples, and target
acceptance.

### Full-profile acceptance run

The full profile has been run once on simulator data and both stages passed
the strict policy. The dataset was built with
`StudentPopulationSimulator(load_simulation_config("configs/simulation.toml")).run()`
(simulator seed 42, 60 neighborhoods), then `build_modeling_table(...)`, then
`split_known_neighborhood_buildings(..., rng=np.random.default_rng(42))`. That
gives 251 buildings split into 192 for training and 59 for testing, with a
modeling table whose SHA-256 begins `51c46a6467d19892`. The run took 6.7
minutes with 4 chains, 1,000 warmup and 1,000 retained draws per chain, target
acceptance 0.9, maximum tree depth 10, and a diagonal mass matrix.

| Stage | Worst R-hat | Min ESS | Divergences | Mean accept | Max depth | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| Total | 1.0029 | 1456 | 0 | 0.907 | 7 / 10 | 0.000 |
| Composition | 1.0017 | 1953 | 0 | 0.923 | 6 / 10 | 0.000 |

On the 59 held-out buildings, reconciliation was exact (maximum error 1.7e-13
in expected counts), every draw was an integer and nonnegative, the joint NLL
was 8.1008 per building, total MAE was 7.793, and empirical coverage was 0.763
for the 80% interval and 0.915 for the 95% interval. The run deliberately used
`action = "warn"` and applied the strict full-profile policy afterward, so a
single threshold miss could not discard the evidence. The joint NLL is the
per-building mean of the summed pointwise keys, which is the quantity
`JointPredictiveNegativeLogLikelihood` computes; see
[Pointwise Posterior Log Probabilities](#pointwise-posterior-log-probabilities).

## Numerical and Runtime Workarounds

The implementation contains deliberate environment-specific paths:

- the observed NB2 likelihood uses an exact closed-form `pyro.factor`;
- prior totals use the exact Gamma-Poisson representation;
- prior and posterior cohort outcomes use the exact
  Dirichlet-then-multinomial representation;
- prior softmax probabilities use `exp(logits - logsumexp(logits))`;
- chains run sequentially to preserve tree-depth instrumentation; and
- fitting calls `torch.set_num_threads(1)` because multithreaded Torch/Pyro was
  unstable on the target macOS ARM/Python 3.13 runtime.

These substitutions preserve the stated probability model. They are not
approximations to the corresponding distributions.

Torch and Pyro are currently top-level imports and core project dependencies.
Consequently, importing `age_group_prediction` requires both packages; the
Bayesian NB2 + Dirichlet-Multinomial model is not lazily loaded as an optional
feature.

## Implementation Cautions

These items were raised during two rounds of Gate 5 independent review. All
but two are resolved; the remaining two are explicitly accepted.

- **Resolved.** `torch.set_num_threads(1)` and `torch.set_default_dtype(torch.float64)`
  modify process-global Torch state. `_scoped_torch_runtime_settings()` in
  `bayesian_inference.py` now wraps every fit in a context manager that
  restores both previous values, including when the wrapped block raises.
- **Explicitly accepted.** The context manager is safe to nest but not
  thread-safe, and `torch.set_default_dtype` and `pyro.set_rng_seed` remain
  process-global. Run one fit per process. The current runner is
  single-threaded, so this cannot happen today.
- **Resolved.** A diagnostic failure under `action = "error"` used to discard
  both the posterior and the diagnostics explaining the failure. Diagnostics
  are now recorded before the policy runs and are attached to the raised
  `RuntimeError`.
- **Resolved.** The mean-accept-probability floor of 0.6 almost never fires
  for NUTS at a target of 0.9. A ceiling, `maximum_mean_accept_prob = 0.98`,
  was added and compared against the largest per-chain value.
- **Resolved.** If Pyro stopped calling `_build_tree` recursively, the
  tree-depth guard would have recorded all-zero depths and passed. All-zero
  depths now report `None`, which fails the policy.
- **Explicitly accepted.** Repeated rows for the same unseen neighborhood
  still receive separate fallback effect draws, and the fallback metadata
  still counts rows rather than unique unseen neighborhoods. This is
  reasonable to accept as-is: the behavior is documented and visible in
  metadata (`last_prediction_unseen_neighborhood_count`), it only affects the
  fallback path for neighborhoods absent from training, and the model's
  neighborhood-effect claims are already limited to known neighborhoods — an
  unseen neighborhood is, by construction, outside what the fitted random
  effects can inform.
- **Resolved.** Tree-depth collection overrides Pyro's protected
  `NUTS._build_tree` method. `_assert_supported_nuts_internals()` in
  `bayesian_inference.py` now pins the expected parameter tuple of
  `NUTS._build_tree` and raises a clear `RuntimeError` before fitting if a
  Pyro upgrade changes that signature, instead of silently recording the
  wrong positional argument as tree depth.
- **Resolved.** Pointwise posterior log probabilities now cover both the
  total NB2 outcome and the joint Dirichlet-multinomial composition score;
  see [Pointwise Posterior Log Probabilities](#pointwise-posterior-log-probabilities) below.

An earlier version of this guide said no further caution-driven changes were
expected. The second review round then produced several, so treat this list as
the current record, not a closed one. Residual risks carried forward:

- All evidence, including the full-profile run, comes from simulator data. The
  effect of misspecification on divergences and ESS for real exported data has
  not been measured.
- Recovery is checked by `tests/validation/test_bayesian_recovery.py`, not by
  simulation-based calibration, which was deliberately not done.
- `_predict_model` holds a `(samples, rows, cohorts)` probability array, and
  `_predictive_draws` loops over draws and rows in Python with one
  `rng.dirichlet` call per row. This is fine at 251 buildings but would take
  minutes to hours and hundreds of megabytes at thousands.

`tests/validation/test_bayesian_recovery.py` fits real NUTS to data simulated
from the model's own design matrices (600 buildings, 30 neighborhoods), with
generating values placed away from their prior means. It asserts that the
posterior standard deviation of each of the four scalar parameters
(`total_intercept`, `log_total_concentration`, `neighborhood_scale`,
`log_composition_concentration`) is below 0.4 of its prior standard deviation.
It also asserts that at least three of those four 95% intervals cover the
truth, and that at least 70% of total-coefficient intervals and 80% of
composition-coefficient intervals do. Contraction is the assertion with
power: an earlier version checked coverage only and passed with the NB2
likelihood deleted, because a posterior that simply reproduces a prior centred
near the truth still covers it. Rerunning the current test against a copy of
the model with the NB2 factor multiplied by zero fails it, as intended:
`total_intercept` stays at 0.96 of its prior standard deviation. The test
shows that one reduced-profile fit at
one seed learns every scalar from the data and centres it correctly. It does
not show calibrated coverage across seeds, it does not exercise the
coefficient vectors' contraction, and it does not assert convergence: at the
reduced profile its total stage currently warns (worst R-hat about 1.09,
minimum ESS about 26).

## Pointwise Posterior Log Probabilities

`_pointwise_log_probabilities` in `bayesian_conditional.py` returns one dict
per prediction with keys `{"total", *cohort_names}`, not only `"total"`.

- `"total"` is the posterior-integrated NB2 log mass, as before.
- Each cohort key is a **conditional** Dirichlet-multinomial log mass given
  the preceding cohorts in schema order. It is obtained by computing
  posterior-integrated log masses for the cumulative cohort *prefixes*
  (`dirichlet_multinomial_prefix_log_masses` in `distributions.py`, integrated
  over posterior samples before differencing — integrating each increment
  separately would not recover the joint score) and then taking successive
  differences (`np.diff(..., prepend=0.0)`).
- The cohort entries therefore sum to the posterior-integrated joint
  Dirichlet-multinomial composition score, up to floating-point error. The
  joint log mass agrees with `scipy.stats.dirichlet_multinomial.logpmf`, but
  not to machine precision: the worst absolute error over 500 random cases is
  about 1e-13 at this project's scale (totals up to 120), about 1e-11 to 4e-11
  for totals up to 1,000 (an independent reviewer measured 4.4e-11), and about
  1e-9 for totals up to 5,000 with very large concentrations. All of these are
  negligible next to NLL differences between models.
- `get_metadata()` reports this contract explicitly under
  `pointwise_log_probability_scope`, including the key list, per-key
  semantics, cohort order, and `includes_multinomial_coefficient: true`.
- `parametric_distributions` remains `None` for this model, unchanged.

The same per-cohort conditional decomposition (via the
`multinomial_prefix_log_masses` helper in `distributions.py`) was applied to
the independent total/probability model's (Model B's) multinomial scores, so
Model B and this model can be compared key by key. That comparability depends
on three properties that any consumer of these scores must respect:

- **The per-cohort values depend on schema order.** Each value is conditional
  on the cohorts before it, so reordering `cohort_target_columns` moves
  probability mass between keys. Only the sum over all keys is invariant.
- **The final cohort's value is 0.0, up to rounding (about 1e-13).** Once the total and the
  preceding cohorts are known, the last cohort count is determined.
- **Model A's cohort keys are not comparable.** The direct cohort model's
  per-cohort keys are *marginal* log masses of independent Poisson, NB2, or
  Normal cohort distributions, and its `"total"` key is a convolution of those
  same cohort distributions. Comparing Model A to either other model key by
  key is meaningless. In particular, the per-target `predictive_nll` that
  `PredictiveNegativeLogLikelihood` reports for the last cohort would show
  Model B and this model "beating" Model A by exactly Model A's
  `n_highschool` NLL, for free. Summing Model A's keys does not give a joint
  score either, because its total is already implied by the cohort terms and
  would be counted twice.

For Model B and this model, the sum over all keys is the joint predictive log
score $\log p(Y_b, \mathbf C_b)$. For this model that holds because the two
stages are fitted separately, so the posterior is a product and
$\log E[p(Y_b)] + \log E[p(\mathbf C_b \mid Y_b)]
= \log E[p(Y_b)\,p(\mathbf C_b \mid Y_b)]$ (see
[Composition stage](#composition-stage)). This is the score that section 9.3 of the plan names as the
primary distributional selection criterion. It is computed by
`JointPredictiveNegativeLogLikelihood` in `metrics.py` (metric name
`joint_predictive_nll`, target `joint`), which `default_metric_set` adds when
given a `joint_nll_interpretation`. The metric relies on each prediction
declaring what its keys mean, through
`PredictionResult.pointwise_log_probability_scope`:

- `"sequential_joint"`, set by this model and Model B: the keys sum to the
  joint score. Only this scope provides the
  `joint_pointwise_log_probabilities` capability.
- `"marginal"`, set by Model A: the keys are marginal scores and cannot be
  summed.
- `None`, for no declaration: treated as not joint.

Model A therefore fails the capability check. Evaluation raises a clear error,
or skips the metric and records the skip under
`on_missing_capability = "skip"`, instead of reporting a double-counted
number. If entries carry a draw axis, they are summed within each draw before
the log-mean-exp. The per-target `predictive_nll` values are unchanged and
still carry the order dependence described above, so rank models by
`joint_predictive_nll`, not by the per-cohort values.

Gate 6 applies that rule through `sequential_joint_selection_policy(...)` for
the independent total/probability and Bayesian conditional approaches. It does
not force a global likelihood winner across all three approaches. The direct
cohort policy sums only its independent cohort NLLs and excludes the total
convolution, while Normal direct-cohort runs are point/interval diagnostic
comparators. Cross-approach tables retain common point and composition metrics
alongside the explicit likelihood interpretation.

## Selection And Comparison

### Candidate, profiles, and feature forms

| Stage | Candidate | Profile | Diagnostic action |
|---|---|---|---|
| Cross-validation | `bayesian-reduced` (`build_canonical_candidate_registry`) | `[bayesian_conditional] active_profile` (`reduced`: 2 chains, 150 warmup, 150 samples) | `warn` |
| Final full-training refit | same candidate, final-refit factory | forced to `full` (4 chains, 1000 warmup, 1000 samples) | `error` |

The final refit refuses a Bayesian model that did not actually run the full
profile with thresholds at least as strict as the full defaults. In
cross-validation, the selected Bayesian candidate must declare the same total
and probability feature specs as the selected Model B candidate, or the
selection freeze is refused.

### Within-approach selection

`sequential_joint_selection_policy("BayesianConditionalModel")` ranks Bayesian
candidates by:

1. `joint_predictive_nll` (posterior-integrated total plus composition log
   mass, `sequential_joint` scope);
2. composition log loss;
3. total RMSE.

By default, `run_cross_model_validation` (`require_convergence=True`) also
excludes candidates whose recorded diagnostic policy failed; pass
`require_convergence=False` to rank unconverged candidates anyway.

### Cross-family comparison

`select_cross_family_winner` compares the Bayesian winner with the Model B
winner by `joint_predictive_nll`, which is comparable because both report
`sequential_joint` log masses that include the multinomial coefficient. The
better conditional model is then compared with Model A's winner by composition
log loss, mean cohort RMSE, and mean cohort MAE. Model A's `marginal` scores
are never ranked against this model's joint score. In the canonical run the
Bayesian conditional model was selected, with Models A and B logged as
predeclared comparators.

Details: [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md)
for the selection rules and freezes, and
[FINAL_EVALUATION.md](FINAL_EVALUATION.md#4-refitting-the-frozen-winners) for
the full-profile refit guard and the one-time lockbox evaluation.

## Persistence And Serving

`to_state_bundle()` stores, as plain JSON:

- the base bundle header, total feature-transformer state, and training
  hashes;
- `probability_feature_spec` and the full `BayesianConditionalConfig`
  (priors, both profiles, both diagnostic policies, active profile);
- the probability feature-transformer state;
- posterior samples for the named sites prediction reads, for both stages;
- the neighborhood lookup as a list of `[neighborhood_id, index]` pairs, so
  numeric IDs survive JSON;
- per-stage diagnostics and the prior-predictive summary.

Reloading never re-runs NUTS. Unlike Models A and B, **predictive draws and
intervals do not need the training frame**: they come from the stored
posterior samples. The bundle does contain posterior summaries and the
neighborhood IDs seen in training, so store it with the same care as the data.
A full-profile bundle holds 4,000 posterior samples per site and is much larger
than a reduced-profile one.

Prediction details that matter when serving:

- **Point means** average the total mean over posterior samples, average the
  composition probabilities over samples, and multiply them, so cohort means
  reconcile exactly with the total mean.
- **Unseen neighborhoods** get a fresh effect $u \sim \mathcal N(0, \sigma_u)$
  per posterior sample and row, so their point predictions depend on the
  random generator (the model's default seed when none is passed). The count is
  recorded as `last_prediction_unseen_neighborhood_count`.
- **Draws** pair independently sampled total and composition posterior indices;
  intervals use at least 500 internal draws, and every draw is checked for
  exact reconciliation.
- **MLflow pyfunc** serving returns deterministic point predictions only; see
  [MLFLOW_EXPERIMENTS_GUIDE.md](MLFLOW_EXPERIMENTS_GUIDE.md#9-load-a-final-model).

## Configuration

All sections live in [`configs/modeling.toml`](../configs/modeling.toml) and
must be present. Values below are the shipped defaults.

| Section | Keys (defaults) | Role |
|---|---|---|
| `[bayesian_conditional]` | `active_profile = "reduced"` | Selects the NUTS profile and its diagnostic policy |
| `[bayesian_priors]` | `total_intercept_loc = -2.0`, `total_intercept_scale = 1.0`, `total_coefficient_scale = 0.5`, `dispersion_log_loc = 0.0`, `dispersion_log_scale = 1.0`, `neighborhood_scale = 0.5`, `composition_intercept_scale = 1.0`, `composition_coefficient_scale = 0.5`, `kappa_log_loc = 2.0`, `kappa_log_scale = 1.0` | Priors for both stages |
| `[bayesian_reduced_profile]` | `chains = 2`, `warmup_steps = 150`, `posterior_samples = 150`, `target_acceptance = 0.9`, `max_tree_depth = 10`, `full_mass = false`, `jit_compile = false` | Cross-validation sampling |
| `[bayesian_full_profile]` | `chains = 4`, `warmup_steps = 1000`, `posterior_samples = 1000`, same sampler settings | Final refit sampling |
| `[bayesian_reduced_diagnostics]` | `action = "warn"`, `maximum_rhat = 1.05`, `minimum_effective_sample_size = 50.0`, `maximum_divergences = 0`, mean acceptance in `[0.6, 0.98]`, `maximum_tree_depth_saturation = 0.05` | Convergence policy for the reduced profile |
| `[bayesian_full_diagnostics]` | `action = "error"`, `minimum_effective_sample_size = 100.0`, other thresholds as above | Convergence policy for the full profile |
| `[bayesian_prior_predictive]` | `draws = 200`, `action = "warn"`, `maximum_children_per_apartment = 10.0`, `minimum_expected_share_ratio = 0.25`, `maximum_expected_share_ratio = 2.0`, `maximum_expected_dominant_share = 0.85`, `minimum_concentration_quantile = 1.0` | Prior plausibility checks run before inference |
| `[bayesian_stabilization]` | `clip_total_log_mean = false`, `total_log_mean_bounds = [-20, 20]`, `clip_composition_logits = false`, `composition_logit_bounds = [-20, 20]` | Optional, recorded numerical clipping at prediction |

Profiles require at least two chains, so R-hat is always defined. Feature
forms and the random seed are not configured here (see [Purpose](#purpose)).

## Metadata

`model.metadata` reports:

- `likelihood` and `parameterization`;
- `pointwise_log_probability_scope`: keys, `sequential_joint` scope,
  log-mean-exp posterior integration, cohort order, and the multinomial
  coefficient flag;
- `hyperparameters` (the full configuration) and `priors`;
- `dependency_versions` (torch, pyro) and `uncertainty_method`;
- `diagnostics`: per stage, R-hat, ESS, divergences, acceptance, tree depth,
  the active profile, `policy_passed`, `policy_failures`, and thresholds;
- `prior_predictive`, `known_neighborhood_count`,
  `last_prediction_unseen_neighborhood_count`, and
  `probability_preprocessing`.

MLflow logs the stage diagnostics as `diag/{stage}/max_rhat`,
`diag/{stage}/min_ess_clamped`, `diag/{stage}/ess_valid`, and
`diag/{stage}/policy_passed`.

## Related Files

- [`src/age_group_prediction/models/bayesian_conditional.py`](../src/age_group_prediction/models/bayesian_conditional.py): public estimator lifecycle, tensor preparation, posterior prediction, and metadata.
- [`src/age_group_prediction/models/bayesian_components.py`](../src/age_group_prediction/models/bayesian_components.py): named Pyro total/composition models and prior-predictive plausibility checks.
- [`src/age_group_prediction/models/bayesian_inference.py`](../src/age_group_prediction/models/bayesian_inference.py): sequential per-chain NUTS execution and convergence diagnostics.
- [`src/age_group_prediction/modeling_config.py`](../src/age_group_prediction/modeling_config.py): typed priors, profiles, and policies.
- [`src/age_group_prediction/distributions.py`](../src/age_group_prediction/distributions.py): NB2 parameter conversions and prefix log masses for the Dirichlet-multinomial and multinomial distributions.
- [`configs/modeling.toml`](../configs/modeling.toml): runtime defaults.
- [`tests/unit/test_bayesian_conditional.py`](../tests/unit/test_bayesian_conditional.py): focused Bayesian NB2 + Dirichlet-Multinomial contracts.
- [`tests/unit/test_distributions.py`](../tests/unit/test_distributions.py): analytic NB2 checks and SciPy oracles for the composition log masses.
- [`tests/unit/test_model_state_bundles.py`](../tests/unit/test_model_state_bundles.py): state-bundle reload, header refusal, and exact reproduction for all three models.
- [`tests/validation/test_bayesian_recovery.py`](../tests/validation/test_bayesian_recovery.py): the only unmocked NUTS tests, marked `slow`: end-to-end invariants and parameter recovery.
- [`MODELING_REBUILD_PLAN.md`](MODELING_REBUILD_PLAN.md): Gate 5 specification and acceptance criteria.
- [`EVALUATION_AND_METRICS.md`](EVALUATION_AND_METRICS.md), [`CROSS_VALIDATION_AND_SELECTION.md`](CROSS_VALIDATION_AND_SELECTION.md), and [`FINAL_EVALUATION.md`](FINAL_EVALUATION.md): scoring, selection, and the final evaluation this model participates in.
