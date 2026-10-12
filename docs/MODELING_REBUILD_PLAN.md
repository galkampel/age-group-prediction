# Modeling Rebuild And Experiment Tracking Plan

> **Status: Gates 1 through 9 complete; the rebuild was approved by the user on
> 2026-09-15.** See the
> post-completion audit in `docs/GATE_9_INDEPENDENT_VALIDATION_REPORT.md`. Gate 8
> (cross-family selection and the
> one-time lockbox) completed on 2026-09-14, after its independent validation,
> two remediation passes, and a fresh-session remediation review; see its
> implementation record below. Gates 1-6
> were implemented, accepted and independently validated (below). Gate 6's
> statistical acceptance found two blocking defects and several evidence gaps,
> all fixed and verified (see the Gate 6 section). Gate 5 had three independent
> reviews (REJECT, ACCEPT WITH GAPS, and a final ACCEPT WITH GAPS with no
> blockers); every finding was fixed and verified, and the full-profile
> acceptance run passed (see the Gate 5 section). Gate 7 (MLflow tracking) was
> implemented in phases, each with an independent review, and its acceptance
> review's conditions were discharged; a behavior-preserving module split
> followed (see the Gate 7 implementation record). This plan
> superseded the pre-rebuild model-fitting roadmap, which Gate 9 removes.
> Implement one gate
> at a time, run its verification checks, and obtain the named independent
> review before continuing.
>
> **Independent validation status (separate from implementation status above).**
> A second, independent validation pass, one gate at a time, with verdicts and
> remediation recorded in `docs/GATE_VALIDATION_FINDINGS.md` (the current status
> record): **Gates 1, 2A, 2B, 2C, 3, 4, 5 and 6 are validated and remediated**,
> under the protocol in `docs/GATE_5_6_VALIDATION_INSTRUCTION.md` for the last
> two. Gate 7 had per-phase independent reviews rather than a separate
> validation pass. Gate 8 was independently validated, remediated, and its
> remediation independently reviewed. Gate 9 Phase 4 was independently
> accepted and approved by the user on 2026-09-15; its post-completion
> independent validation returned **ACCEPT**. Current suite baseline:
> **696 passing**, with 13 documented warnings: twelve MLflow
> integer-schema hints and one Bayesian recovery warning. Note that a gate being "complete
> and accepted" refers to its implementation review, not to this validation
> pass — Gates 3 and 4 each carried defects through the former that the latter
> found.

This work follows the EDA handoff in
`EDA_AND_PREDICTIVE_MODELING_PLAN.md`. Simulation remains owned by
`student_simulator`; the modeling package consumes only exported,
prediction-time building data.

## 1. Objective

Predict, for each building:

1. the expected total number of resident children;
2. the probability that a resident child belongs to each age group; and
3. the expected or predictive count in each age group.

The age-group probability estimand is

$$
P(K_b = k \mid x_b),
\qquad
k \in \{\text{kindergarten},\text{elementary},\text{highschool}\},
$$

where $x_b$ contains only information available for building $b$ at prediction
time. The mandatory observed accounting identity is

$$
n_{b,\mathrm{total}}
= n_{b,\mathrm{kindergarten}}
+ n_{b,\mathrm{elementary}}
+ n_{b,\mathrm{highschool}}.
$$

Fit and compare three approaches:

1. **Model A: direct cohort distributional-tree baseline.** Three independent
   LightGBM regressors, one per age group, compared under Normal, Poisson, and
   NB2 objectives. One fitted model instance uses one family for all cohorts.
2. **Model B: independent total and age-group probability models.** An NB2
   total-count model and a grouped multinomial logistic probability model,
   fitted independently and combined into reconciled expected cohort counts.
3. **Bayesian NB2 + Dirichlet-Multinomial model.** A hierarchical NB2 total-count
   stage and a conditional Dirichlet-multinomial age-group stage implemented in
   PyTorch Pyro.

All models use the same split manifests, cross-validation folds, result
contracts, and metric registry. MLflow is the final experiment entry point but
must remain outside the model implementations.

## 2. Non-Negotiable Modeling Rules

1. Split before fitting transformations, deriving target-dependent rows, tuning,
   or calibrating.
2. Use cross-validation drawn only from the training partition for feature,
   model, prior, regularization, and calibration decisions.
3. Persist the final test building IDs and do not evaluate them until every
   modeling choice is frozen.
4. Do not use identifiers, targets, target-derived columns, or private simulator
   latent effects as predictors.
5. Fit preprocessing independently inside every training fold.
6. Give every prediction a declared probabilistic interpretation. Do not label a
   plug-in Poisson score as an NB2 or posterior-predictive score.
7. Use neighborhood-aware resampling and neighborhood-cluster bootstrap
   intervals because buildings within a neighborhood are not independent.
8. Never claim unseen-neighborhood performance from the primary split. It
   supports prediction for a new building in a known neighborhood.
9. Keep the notebook as a thin experiment client. Reusable fitting, prediction,
   evaluation, bootstrapping, and tracking logic belongs in tested source code.

## 3. Target Package Structure

```text
src/age_group_prediction/
    __init__.py
    modeling_config.py
    dataset_builder.py
    data_splitting.py
    feature_engineering.py
    results.py
    metrics.py
    evaluation.py
    predictive.py
    resampling.py
    distributions.py
    experiment_config.py
    hashing.py
    state_bundle.py
    tuning.py
    experiment/
        __init__.py
        aggregation.py
        artifacts.py
        candidate_registry.py
        candidates.py
        contracts.py
        evidence.py
        final_evaluation.py
        final_selection.py
        importance.py
        partitions.py
        policies.py
        runner.py
        seeds.py
        selection.py
    tracking/
        __init__.py
        _files.py
        settings.py
        _metadata.py
        runs.py
        candidates.py
        evidence.py
        final.py
        pyfunc_model.py
    models/
        __init__.py
        base.py
        direct_cohort.py
        count_regression.py
        grouped_multinomial.py
        probability_calibration.py
        fold_scoring.py
        independent_total_probability.py
        bayesian_components.py
        bayesian_inference.py
        bayesian_conditional.py
```

Ownership boundaries:

| Module | Responsibility |
|---|---|
| `modeling_config.py` | Typed, immutable schemas and configurable defaults for columns, features, splitting, evaluation, and randomness |
| `results.py` | Model-independent `PredictionResult`, `EvaluationResult`, and `ParametricDistributionSpec` contracts shared by `metrics.py`, `evaluation.py`, and `models/base.py` |
| `dataset_builder.py` | Raw modeling-table assembly and deterministic derivations such as room shares |
| `data_splitting.py` | Outer train/test split, train-only cross-validation folds, and serializable manifests |
| `feature_engineering.py` | Fold-fitted encoders, scalers, splines, transformations, and interactions |
| `models/base.py` | Abstract model lifecycle, fitted preprocessing ownership, and metadata contracts |
| `predictive.py` | Model-agnostic central prediction intervals from predictive draws |
| `resampling.py` | Validated neighborhood-cluster row-index resampling reused by evaluation and model refits |
| `metrics.py` | Metric protocol/classes, array-level composition kernels, prediction transformations, and capability requirements |
| `evaluation.py` | Metric orchestration and fixed-prediction neighborhood-bootstrap evaluation; no model or Pyro dependencies |
| `models/direct_cohort.py` | Cohesive LightGBM cohort fitting, likelihood assembly, and bootstrap-refit implementation |
| `models/count_regression.py` | Poisson/NB2 total-count fit and mean-prediction component used by the independent model |
| `models/grouped_multinomial.py` | Weighted grouped multinomial fit and logits component used by the independent model |
| `models/probability_calibration.py` | Child-weighted composition NLL and share-error kernels, and the evidence-gated temperature fit (bounded minimization, 1-df likelihood-ratio retention, diagnostics) used by the independent model |
| `models/fold_scoring.py` | Held-out fold losses for the independent model's total and probability components, with seeds supplied by the model |
| `models/independent_total_probability.py` | Independent-model lifecycle, tuning coordination and seed recording, calibration cross-fitting, prediction assembly, bootstrap refits, and metadata |
| `models/bayesian_components.py` | Pyro total/composition model definitions and prior-predictive simulation |
| `models/bayesian_inference.py` | Sequential NUTS execution, sampler instrumentation, and convergence-diagnostic policy helpers |
| `models/bayesian_conditional.py` | Bayesian estimator lifecycle, posterior prediction, policy application, and metadata |
| `experiment/` | MLflow-free CV, selection, refit, and final evaluation workflow, split into `contracts`, `policies`, `partitions`, `seeds`, `evidence`, `aggregation`, `selection`, `importance`, `runner`, `artifacts` (captured fold bundles and reload checks), `candidates` (predeclared feature-spec enumeration), `candidate_registry` (the canonical candidates, which imports the concrete model classes), `final_selection` (cross-family rule), and `final_evaluation` (full refit and one-time lockbox evaluation) |
| `tracking/` | Optional MLflow adapter; no statistical or model logic. Public API re-exported from `__init__`; internal modules, with one-way imports: `_files` (param and file writers), `settings` (environment resolution), `_metadata` (result and model-metadata readers), `runs` (context, parent run, run endings, final-run guards), `candidates` (child runs), `evidence` (entry points, preflight checks, parent evidence), `final` (guarded final run and final children), `pyfunc_model` (models-from-code pyfunc) |
| `distributions.py`, `experiment_config.py`, `hashing.py`, `state_bundle.py`, `tuning.py` | Shared distribution kernels, the strict TOML runtime loader, canonical hashes, JSON state-bundle helpers, and bounded Optuna tuning |

### 3.1 Existing Code To Retain

Retain and harden these useful foundations rather than rewriting working logic:

- `ModelingDatasetBuilder` in `dataset_builder.py`;
- the `split_known_neighborhood_buildings` algorithm currently in `split.py`;
   and
- the train-only fold-generation convention currently in `resampling.py`.

Move both partitioning implementations into `data_splitting.py`. They operate
on the same concepts, validation rules, identifiers, RNG stream, and manifest
types, so keeping them together makes the outer split and inner CV relationship
explicit and reduces duplicated validation. Keep separate public functions and
typed manifest classes inside the module; consolidation does not mean one
function should perform both operations.

`ModelingDatasetBuilder` should copy supplied column lists so callers cannot
mutate its internal schema accidentally. Derived-feature registration should be
idempotent or raise an explicit duplicate-feature error. It remains a raw table
builder, not a fitted preprocessing pipeline.

### 3.2 Existing Code To Remove After Replacement

Delete the following only after replacement tests and the end-to-end workflow
pass:

- `benchmarks.py`;
- `direct_cohort_models.py`;
- `poisson_total_benchmark.py`;
- `gate3.py`;
- their obsolete exports from `__init__.py`;
- obsolete Gate 3 tests; and
- Gate 3-specific cells in `notebooks/02_model_fitting.py`.

Replace the narrow contents of `metrics.py`; do not remove the metric boundary.
Do not modify or remove simulator implementation as part of this migration.
The former `split.py` and `resampling.py` modules were consolidated into
`data_splitting.py` during Gate 1 because that gate owns the partition API.

## 4. Typed Configuration And Shared Interfaces

In this plan, a **contract** means an explicit promise between components: for
example, which columns a model receives, what a prediction result contains, or
which invariants a split must satisfy. This is a useful engineering practice
because tests can verify those promises across all three models. It does not
require a file named `contract.py` or force values to be hard-coded.

Use the clearer module name `modeling_config.py`. Define frozen dataclasses such
as `ModelingSchema`, `SplitConfig`, `FeatureSpec`, `EvaluationConfig`, and
`PredictionConfig`. `FeatureSpec` is immutable experiment configuration; fitted
means, scales, knots, and category state belong to a separate transformer
object. Provide project defaults as named constants, including
`DEFAULT_SEED = 42`, but allow callers and experiment configurations to replace
all research choices explicitly. Separate:

- **invariants**, which are not configurable within this problem, such as the
  target accounting identity and the exclusion of target columns from
  predictors; and
- **experiment parameters**, which are configurable and logged, such as column
  selections, reference category, test fraction, fold count, feature blocks,
  interval levels, bootstrap replicates, and random seed.

No model should import an unexplained global feature list when the corresponding
typed schema or feature specification can be passed explicitly. Defaults make
the canonical experiment concise; typed overrides make alternative experiments
auditable. Configuration precedence is explicit constructor/function arguments,
then the loaded experiment configuration, then documented project defaults.
Model implementations must not contain unreported statistical magic constants.

### 4.1 Canonical Columns

The `ModelingSchema` must distinguish identifiers, raw numeric features,
categorical features, deterministic room-share features, cohort targets, and
the total target. Validation must reject:

- duplicate building IDs;
- missing required columns;
- non-finite features;
- non-integer or negative targets;
- non-positive `n_apartments`;
- unknown `school_status` values;
- violations of total-equals-cohort-sum; and
- private latent-effect columns in the selected feature list.

### 4.2 Data Splitting And Manifests

Continue using the deterministic within-neighborhood building split. Every
non-singleton neighborhood contributes buildings to both train and test;
singletons remain in training. Store a serializable manifest for the outer
test lockbox containing:

- split strategy and version;
- requested and realized test fraction, which differ whenever the
  at-least-one-holdout-per-neighborhood floor binds;
- ordered train and test building IDs;
- source-table and column-schema hashes;
- row and neighborhood counts;
- whether the split's randomness came from a caller-supplied generator or fell
  back to the project default, recorded as a source rather than a seed value
  because a supplied generator may already have been advanced; and
- the supported deployment claim.

Validation folds are deterministic, short-lived fit/validation partitions of
the outer training table. They contain no manifests and may contain only outer
training IDs. Persisting one outer manifest is sufficient to protect the test
lockbox; regenerate CV folds from the same input, configuration, and RNG for a
repeated experiment.

Folds rotate each neighborhood's buildings across `n_folds` rather than drawing
repeated independent holdouts, so every non-singleton training building is
validated exactly once and every neighborhood keeps at least one building in
every fit partition. The validation share is therefore `1 / n_folds` by
construction and there is no separate validation fraction to disagree with it.
A building alone in its neighborhood cannot be validated without leaving that
neighborhood unrepresented at fit time, which is outside the supported
deployment claim; the fold generator returns those building IDs so the
exclusion is recorded evidence rather than an emergent property of the draw.

Both the outer split and inner fold generator accept
`rng: numpy.random.Generator | None = None`. When omitted, create a fresh
generator with `numpy.random.default_rng(DEFAULT_SEED)`. Use the resolved
generator directly for the partition permutations; a caller-provided generator
therefore advances as its requested split or folds are created. Do not also
expose a competing `seed` argument on the same API. Exact outer replay uses the
persisted building IDs, after verifying the source-table and column-schema
hashes, that the recorded strategy matches the active configuration, and that
the recorded row count, neighborhood count, and realized holdout fraction match
the table the manifest is replayed against.

### 4.3 Base Model API

Define `BaseAgeGroupModel` as an abstract base class with:

```python
fit(train_df, *, feature_spec, rng=None) -> BaseAgeGroupModel
predict(eval_df, *, prediction_config=None, rng=None) -> PredictionResult
evaluate(eval_df, *, metrics, rng=None) -> EvaluationResult
get_metadata() -> dict[str, object]
```

The precise signatures may use typed configuration dataclasses, but all three
models must satisfy the same behavioral contract.

Use `numpy.random.Generator` wherever the implementation accepts one. Libraries
such as scikit-learn, statsmodels, Torch, or Pyro may require integer seeds or
framework generators instead. In those cases, derive deterministic child seeds
from the supplied NumPy generator, initialize the library-specific RNG locally,
and record every derived seed and its purpose in metadata. If `rng` is omitted,
resolve a fresh generator from `DEFAULT_SEED`; never use module-level mutable
RNG state. Tests must cover omitted-RNG reproducibility, caller-provided
generator behavior, operation replay from recorded child seeds, and stable
child-seed derivation.

`PredictionResult` carries building IDs, total and cohort means, normalized
age-group probabilities, optional predictive draws/intervals, optional
pointwise log probabilities, and reconciliation diagnostics.

`get_metadata()` must return JSON-serializable information covering model and
implementation version, likelihood and parameterization, feature specification,
preprocessing summary, hyperparameters, priors, calibration, seeds, dependency
versions, training data/schema hashes, fit duration, uncertainty method, and
model-specific diagnostics. The experiment runner adds split, fold, selection,
and test-lockbox metadata.

## 5. Feature Engineering

Create a dedicated `feature_engineering.py`. This separation is best practice
because raw data assembly and fitted transformations have different lifecycles:
the former is deterministic table construction, while the latter learns state
from each training fold and can differ by model component.

Feature engineering is implemented and accepted in **Gate 2A**, before the
shared model contract, metrics, or any concrete model. Gate 1 owns deterministic
modeling-table construction, including the configured room shares. Gate 2A owns
all transformations that learn state or vary by model component.

### 5.1 Feature Specification And Fitted State

Define one frozen, JSON-serializable `FeatureSpec` with:

- component: `tree`, `total_count`, or `age_probability`;
- selected base numeric and categorical columns;
- categorical reference levels and unknown-category policy;
- SES form: linear, quadratic, or low-degree spline;
- enabled interaction blocks;
- exposure policy;
- centering/scaling policy; and
- feature-specification version.

Use one value type with component-aware validation rather than model-specific
subclasses. Keep learned state in a separate `FittedFeatureTransformer` with:

```python
fit(fit_df) -> FittedFeatureTransformer
transform(eval_df) -> pandas.DataFrame
fit_transform(fit_df) -> pandas.DataFrame
get_feature_names_out() -> tuple[str, ...]
get_metadata() -> dict[str, object]
```

Transformation before fitting must fail. Output must be finite, numeric, and
have deterministic column names and order. Metadata must be JSON serializable
and record the feature-spec version, component, selected blocks, fitted feature
names, reference categories, and fit-row count. The experiment runner owns the
canonical training-table hash used for artifact provenance.

### 5.2 Shared Base Representation

- Keep continuous prediction-time variables numeric.
- Encode categorical features from `ModelingSchema.categorical_feature_specs`,
   using each configured reference category.
- Represent building size by `n_apartments` or
  $\log(n_\mathrm{apartments})$ as an exposure offset where supported.
- Represent room composition by 3-, 4-, and 5-room shares, with the 6-room
  share omitted as reference.
- Fit centering, scaling, knots, and category state from the fit fold only.
- Reject identifiers, targets, and private simulator effects as transformer
  inputs.
- Fit Model B probability preprocessing on every building in the fit fold
  before excluding zero-total buildings from the grouped multinomial
  likelihood. The composition target must not determine which rows establish
  preprocessing state.

Define named defaults:

- `DEFAULT_TREE_FEATURE_SPEC`: base features, no scaling, no explicit
  nonlinearities or interactions;
- `DEFAULT_TOTAL_FEATURE_SPEC`: linear base form with log exposure offset; and
- `DEFAULT_PROBABILITY_FEATURE_SPEC`: linear base form without exposure as an
  ordinary probability predictor.

The Bayesian NB2 + Dirichlet-Multinomial model later receives the frozen selected total and probability specs from
Model B. It refits preprocessing state on its own training data but does not
reopen feature-form search.

### 5.3 Candidate Nonlinearities And Interactions

Compare a small, predeclared sequence instead of an unrestricted feature search:

1. linear SES versus quadratic SES versus a low-degree spline;
2. SES by average household size;
3. daycare count by median age;
4. one parsimonious room-composition by household-size term for total counts;
5. room composition by median age for age-group probabilities; and
6. a monotone or low-degree daycare saturation curve only if held-out residuals
   justify it.

Every interaction includes its main effects. Do not include room counts, room
shares, mean rooms, and a large-room summary simultaneously. Retain candidate
blocks only when improvement is stable across identical training folds and does
not cause unacceptable conditioning or calibration.

Represent these candidates with a closed, validated vocabulary. `FeatureSpec`
validation must reject mutually exclusive SES forms, interactions without their
main effects, total-only blocks in probability specifications,
probability-only blocks in total specifications, and redundant room
representations. Candidate factories enumerate an ordered set of specs; they do
not fit models or select a winner.

| Component | Feature policy |
|---|---|
| Model A trees | Start with base features; tune capacity before adding explicit nonlinearities or interactions |
| Model B total | Compare named structural specifications in separate runs; tune each run's regularization independently |
| Model B probabilities | Compare named composition specifications independently from the total model and tune each run's regularization |
| Bayesian NB2 + Dirichlet-Multinomial | Start from frozen Model B forms; do not reopen broad feature search during Bayesian fitting |

### 5.4 Feature Selection Ownership

Gate 2A defines, validates, and transforms every allowed candidate. It does not
choose a winning form. Gate 3 evaluates optional tree candidates only after
tuning tree capacity. For Gate 4, one Model B instance represents one fixed
pair of total and probability specifications and one explicit total family.
The experiment runner compares named structural candidates and Poisson/NB2
siblings on identical train-only folds; each candidate uses seeded Optuna to
tune its numerical regularization. Gate 5 consumes the selected frozen Model B
specs. Gate 6 records the selection evidence and prevents further changes
before the outer test lockbox is opened.

## 6. Model A: Direct Cohort Baseline

Implement `DirectCohortModel` in `models/direct_cohort.py`.

1. Fit one `LGBMRegressor` per cohort with a model-level family selection:
   built-in L2 for Normal, built-in Poisson, or a tested custom NB2 objective.
2. Keep raw numeric tree features unchanged and fit categorical encoding only
   on each training fold. Raw `n_apartments` is an ordinary tree predictor;
   Model A does not use a log-exposure offset.
3. Tune a bounded capacity and regularization search space with seeded Optuna
   TPE over fixed train-only known-neighborhood folds. Reproducible runs use a
   fixed trial count, `n_jobs=1`, and no timeout stopping. Select the best
   trial from completed trials only; a pruned trial's partial-fold score is not
   comparable with a fully evaluated trial's.
4. Return cohort means directly and define total mean as their sum.
5. Derive probabilities by normalizing cohort means; specify a deterministic
   fallback when all three means are numerically zero.
6. Dispatch predictive scoring and draws according to the fitted family.
   Poisson and independent Normal cohort sums have exact Poisson and Normal
   total declarations. A general sum of independent NB2 variables is not NB2;
   score its observed total by finite convolution and form intervals from
   summed cohort draws rather than declaring a false total family.

For uncertainty, use neighborhood-cluster bootstrap refits. If predictive
intervals are required, combine refit uncertainty with family-specific outcome
draws. Estimate Normal scales from train-only cross-fitted residuals and select
NB2 dispersion within each cohort's train-only tuning study.

## 7. Model B: Independent Total And Age-Group Probabilities

Implement `IndependentTotalProbabilityModel` in
`models/independent_total_probability.py`.

### 7.1 Total-Count Component

Fit either a Poisson or NB2 regression as an explicit run configuration. The
NB2 candidate is:

$$
Y_b \sim \operatorname{NB2}(\mu_b, \phi),
\qquad
\log \mu_b
= \log n_{b,\mathrm{apartments}} + \beta_0 + f(x_b),
$$

with $\operatorname{Var}(Y_b)=\mu_b+\mu_b^2/\phi$. Estimate dispersion from
training data. Run Poisson and NB2 as separate, independently tuned sibling
candidates on identical folds, then compare them using held-out NLL and
deviance. Do not choose the family inside a model fit or reuse hyperparameters
selected for the other family.

#### Apartment Count As An Exposure Offset

The $\log n_{b,\mathrm{apartments}}$ term is an exposure offset, not a
building fixed effect. It fixes the apartment-count coefficient at one, so the
model estimates a child rate per apartment and then scales it to the building:

$$
\mu_b
= n_{b,\mathrm{apartments}}
   \exp\left(\beta_0 + f(x_b)\right).
$$

Consequently, two otherwise identical buildings with twice as many apartments
have twice the expected number of children. A building fixed effect would be a
separate building-specific intercept, $u_b$, in
$\log\mu_b=\log n_{b,\mathrm{apartments}}+\beta_0+f(x_b)+u_b$. It is not
appropriate for prediction of a new building and is not usefully identifiable
from one aggregated outcome per building. Estimating a coefficient on
$\log n_{b,\mathrm{apartments}}$ instead would relax proportionality, giving
$\mu_b\propto n_{b,\mathrm{apartments}}^{\beta_{\mathrm{size}}}$; that is a
different size-effect model, not an exposure-offset model. Treat fixed
proportionality as a train-only CV-checkable assumption rather than a claim
that apartment count captures unobserved building heterogeneity.

#### NB2 Optimizer Initialization

Initialization supplies a numerically sensible starting point to the optimizer;
it does not fix either fitted parameter. With all non-intercept coefficients
initially zero, the model has $\mu_b=A_b\exp(\beta_0)$, where
$A_b=n_{b,\mathrm{apartments}}$. Initialize the intercept from the pooled
child-per-apartment rate:

$$
r_{\mathrm{initial}}
= \frac{\sum_b y_b}{\sum_b A_b},
\qquad
\beta_{0,\mathrm{initial}}
= \log r_{\mathrm{initial}}.
$$

Using the ratio of sums makes the initial total predicted count match the
observed total when all other effects are zero. It is exposure-weighted and is
therefore intentionally different from the unweighted average of per-building
rates, $B^{-1}\sum_b y_b/A_b$.

The implementation uses the project NB2 dispersion $\alpha$,
$\operatorname{Var}(Y_b)=\mu_b+\alpha\mu_b^2$, and optimizes $\log\alpha$.
Given configured positive bounds $[\alpha_{\min},\alpha_{\max}]$, initialize
at their geometric mean:

$$
\log\alpha_{\mathrm{initial}}
= \frac{\log\alpha_{\min}+\log\alpha_{\max}}{2},
\qquad
\alpha_{\mathrm{initial}}
= \sqrt{\alpha_{\min}\alpha_{\max}}.
$$

This is the midpoint on the optimization's log scale, which is preferable to
the arithmetic mean when the allowed dispersions span orders of magnitude. The
optimizer subsequently estimates the intercept, remaining coefficients, and
dispersion jointly within the configured bounds.

### 7.2 Grouped Multinomial Probability Component

Fit building-conditioned probabilities with grouped or weighted multinomial
logistic regression. For each building, construct at most three training rows:

| Building features | Class label | Sample weight |
|---|---|---|
| $x_b$ | kindergarten | $n_{b,\mathrm{kindergarten}}$ |
| $x_b$ | elementary | $n_{b,\mathrm{elementary}}$ |
| $x_b$ | highschool | $n_{b,\mathrm{highschool}}$ |

Drop zero-weight rows. A zero-total building contributes no composition
likelihood but still contributes to the independent total model.

This is preferred over literal one-row-per-child expansion. Children in one
building have identical predictors, so weighted rows optimize the same
multinomial log-likelihood:

$$
\ell_b = \sum_k n_{b,k}\log p_{b,k}.
$$

The grouped representation estimates the requested $P(K_b=k\mid x_b)$ while
using at most three rows per building, avoiding memory growth and accidental
leakage during expansion. A small fixture must verify coefficient, objective,
and prediction equivalence with literal expansion to solver tolerance.

Total or cohort counts may appear only as likelihood responses/weights during
training, never as predictor columns.

#### Probability Calibration

Multinomial `predict_proba` outputs are nonnegative and normalized, but this
simplex validity does not guarantee empirical calibration. After the
probability feature specification and regularization are frozen, generate
decision logits from neighborhood-aware out-of-fold fits using only the outer
training partition. Fit one positive temperature $T$ by minimizing the grouped,
child-count-weighted multinomial NLL:

$$
-\sum_b\sum_k n_{b,k}\log\left[
\operatorname{softmax}\left(\frac{z_b}{T}\right)_k
\right].
$$

Use $T=1$ as the uncalibrated baseline and retain the fitted temperature only
when it improves out-of-fold NLL beyond a configured numerical tolerance.
Otherwise keep $T=1$. A single shared temperature preserves the multiclass
probability simplex and is preferred here to separate one-versus-rest sigmoid
or isotonic calibrators, which can overfit and require incoherent
renormalization. Fit probability preprocessing on every fold-training
building, but give zero-total buildings zero likelihood and calibration weight.
Record raw and calibrated weighted NLL, weighted multiclass Brier score, the
temperature, and the retain/reject decision. Never fit or select calibration
against the outer test lockbox.

### 7.3 Combined Predictions And Uncertainty

Combine components as

$$
\widehat C_{b,k}=\widehat\mu_b\widehat p_{b,k}.
$$

Expected cohort counts must sum to expected total within numerical tolerance.
Generate predictive draws by sampling an NB2 total and then multinomial cohort
counts. Use neighborhood-cluster bootstrap refits for confidence intervals on
parameters, aggregate metrics, or mean predictions.

Because the bootstrap resamples whole neighborhoods, a replicate can omit every
child of one cohort, which leaves the composition model unidentified for that
replicate. Skip and count such replicates rather than failing the prediction,
and raise only when the failed share exceeds `bootstrap_max_failed_fraction`.
Record `bootstrap_successful_replicates` and `bootstrap_failed_replicates`.
Intervals then condition on resamples that span every cohort, which is the
honest description of what is computed; a cohort concentrated in a small number
of neighborhoods will show a nonzero failed count and its interval should be
read accordingly.

Tune components independently with seeded single-job Optuna studies: total
NLL/deviance for the first and composition log loss/Brier score for the second.
Select the best trial from **completed** trials only: a pruned trial is scored
on a subset of folds, so its value is not comparable with a fully evaluated
trial's, and a study with no completed trial is an error rather than a result.
Keep structural feature specifications fixed within a run so SES forms and
interaction blocks remain explicit experiment identities. Tune total L2 and
multinomial $C$ numerically; estimate NB2 dispersion in the likelihood fit and
fit calibration temperature only after the probability hyperparameters are
frozen.

## 8. Bayesian NB2 + Dirichlet-Multinomial

Implement `BayesianConditionalModel` in `models/bayesian_conditional.py` using
PyTorch Pyro (`torch` and `pyro-ppl`), not NumPyro/JAX.

### 8.1 Hierarchical Total Stage

Use

$$
Y_b \mid \mu_b,\phi \sim \operatorname{NB2}(\mu_b,\phi),
$$

$$
\log\mu_b
= \log n_{b,\mathrm{apartments}}
+ \beta_0 + f(x_b)^\top\beta + u_{g[b]},
$$

with regularizing fixed-effect, dispersion, neighborhood-scale, and
non-centered neighborhood random-effect priors. Unit-test conversion between
the project NB2 definition and Pyro/Torch parameters against analytic moments.

### 8.2 Conditional Composition Stage

Use

$$
\mathbf C_b \mid Y_b,\mathbf p_b,\kappa
\sim \operatorname{DirichletMultinomial}
\left(Y_b,\kappa\mathbf p_b\right),
$$

where $\mathbf p_b$ is a softmax predictor and $\kappa$ controls
extra-multinomial variation. Estimate $\kappa$ with a documented positive
prior. Add a composition neighborhood effect only if prior checks and
train-only validation support it.

During training, condition composition on observed totals. During prediction:

1. sample total from the posterior predictive distribution;
2. sample the cohort vector conditional on that sampled total; and
3. assert every cohort draw is nonnegative integer-valued and sums exactly to
   the sampled total.

Use posterior predictive intervals for the Bayesian NB2 + Dirichlet-Multinomial model; do not bootstrap Bayesian fits.

### 8.3 Inference And Diagnostics

1. Run prior-predictive checks before fitting outcomes.
2. Use NUTS/MCMC for the accepted implementation.
3. Record chains, warmup, posterior samples, target acceptance, tree depth,
   divergences, R-hat, ESS, runtime, and seeds.
4. Use reduced but diagnostic-valid settings for predefined CV comparisons and
   full settings for final training.
5. Do not accept SVI as final unless a separate comparison shows adequate
   agreement with NUTS for predictions and uncertainty.
6. Define known-neighborhood prediction and an explicit prior fallback for an
   unseen neighborhood, while limiting the primary claim to known neighborhoods.

## 9. Evaluation Metric Classes And Registry

Use a list of metric objects instead of hard-coding evaluation inside models.
Define a `Metric` protocol or abstract base class whose immutable implementations
declare name, required prediction capability, target scope, optimization
direction, and aggregation. A metric receives observed values plus the common
`PredictionResult` and may transform model output before scoring.

This transformation belongs to the metric when it is part of the metric's
definition. For example, a Bayesian NLL metric may convert posterior log
likelihood draws into a posterior-integrated log score with a numerically stable
log-mean-exp; an interval metric may compute quantiles from posterior predictive
draws; and randomized PIT requires both a predictive distribution and random
jitter. The model exposes distribution parameters or draws, while the metric
owns the common scoring transformation. Metrics must not refit models or mutate
predictions.

Stateless metrics such as MAE can still be small frozen classes. Stochastic
metrics accept an optional NumPy generator and use the same `DEFAULT_SEED`
fallback convention. Return a typed `MetricResult` containing value, target,
aggregation level, sample count, and diagnostic metadata.

### 9.1 Metrics Shared By All Models

Report by total and each cohort wherever defined:

- MAE;
- RMSE;
- mean bias $\operatorname{mean}(\widehat y-y)$;
- $R^2$ as secondary context only;
- mean Poisson deviance;
- predictive NLL/log score under the declared distribution;
- building/neighborhood counts; and
- mean/max reconciliation error.

Reconciliation error is an internal accounting diagnostic, not a forecast-loss
metric. For building $b$,

$$
r_b=\left|\sum_c\widehat\mu_{bc}-\widehat\mu_{b,\mathrm{total}}\right|.
$$

The mean of $r_b$ reports typical accounting drift, while the max catches a
single severe inconsistency that a small mean can hide. Report both and also a
count above a declared tolerance. Do not use reconciliation error alone for
model ranking.

Report composition quality after converting predictions to age-group
probabilities:

- child-count-weighted multiclass log loss;
- count-weighted multiclass Brier score; and
- reliability tables for buildings with positive observed totals.

NLL interpretation differs and must be stored in metadata:

| Model | Distribution used for scoring |
|---|---|
| Model A | Independent Normal, Poisson, or NB2 cohort likelihoods, declared per fitted run; NB2 total uses convolution |
| Model B | NB2 total plus multinomial conditional composition |
| Bayesian NB2 + Dirichlet-Multinomial | Posterior-integrated NB2 total plus Dirichlet-multinomial composition |

Do not rank unlike NLL definitions without displaying this distinction. Point
and composition metrics remain directly comparable.

`Pointwise` log probabilities mean one predictive log probability (or log
density) per held-out observation. If a model provides one value per posterior
draw, aggregate with a numerically stable log-mean-exp so the metric is the
posterior predictive log score:

$$
\log p(y_b\mid\mathcal D)
=\log\left(\frac{1}{S}\sum_{s=1}^{S}\exp\{\ell_{bs}\}\right),
$$

where $\ell_{bs}$ is the draw-specific log probability for observation $b$.

### 9.2 Distributional And Calibration Metrics

When predictive draws or intervals are available, add:

- 80% and 95% empirical coverage;
- mean interval width;
- weighted interval score;
- randomized PIT or quantile-residual summaries;
- observed-versus-predicted mean and variance by prediction bin; and
- residual/calibration summaries by neighborhood.

#### Probability Integral Transform (PIT)

PIT checks calibration of the entire predictive distribution, not only its
mean. For a continuous predictive CDF $F_b$ and observed value $y_b$,

$$
u_b = F_b(y_b).
$$

If the predictive distributions are calibrated on held-out data, the $u_b$
values are approximately uniform on $[0,1]$. Child counts are discrete, so use
randomized PIT to avoid placing all observations at CDF jump boundaries:

$$
u_b
= F_b(y_b-1)
+ v_b\left[F_b(y_b)-F_b(y_b-1)\right],
\qquad
v_b\sim\operatorname{Uniform}(0,1).
$$

Implement this as a metric class that can obtain $F_b$ from explicit Poisson or
NB2 parameters, from a Normal approximation for continuous outcomes, or estimate
the randomized rank from posterior predictive draws for the Bayesian NB2 + Dirichlet-Multinomial model. For predictive
draws $\tilde y_{bs}$, define

$$
L_b=\sum_s\mathbf 1\{\tilde y_{bs}<y_b\},
\qquad
E_b=\sum_s\mathbf 1\{\tilde y_{bs}=y_b\},
\qquad
u_b=\frac{L_b+v_bE_b}{S},
\quad v_b\sim\operatorname{Uniform}(0,1).
$$

Its RNG follows the shared default convention. Interpret the held-out PIT
histogram as follows:

- approximately flat: broadly calibrated predictive distributions;
- U-shaped: predictive distributions are too narrow or under-dispersed;
- center-heavy: predictive distributions are too wide or over-dispersed;
- mass shifted toward zero: observations tend to be lower than predictions;
- mass shifted toward one: observations tend to be higher than predictions.

PIT is a diagnostic, not a standalone selection score. Report it with NLL,
coverage, interval width, and residual checks. For Model A it diagnoses the
declared plug-in Poisson approximation; for Model B use the NB2 total CDF and
conditional composition diagnostics; for the Bayesian NB2 + Dirichlet-Multinomial model use posterior predictive
draws so parameter uncertainty is represented.

Use neighborhood-cluster bootstrap intervals over fixed out-of-fold or test
predictions for aggregate metrics. Store unit, replicate count, seed,
confidence level, failed replicates, and interval method.

### 9.3 Selection

Use identical folds for all candidates. Predeclare joint predictive NLL as the
primary distributional criterion; cohort RMSE/MAE and composition log loss are
secondary. Calibration, finite/nonnegative predictions, and reconciliation are
hard diagnostics. Include fold values, means, standard deviations, and
cluster-bootstrap confidence intervals. Runtime and convergence are explicit
selection constraints.

**This section governs selection *within* a model family.** Joint predictive
NLL is the primary criterion for the two conditional families, which score the
same object. `DirectCohortModel` cannot supply it — it declares a `marginal`
pointwise scope and is excluded by the capability gate rather than given a
double-counted score — so its policy sums the three independent cohort NLLs and
excludes the separately scored total. Those two quantities are not
interchangeable, and **no single likelihood ranks all three families**.
Choosing between families is Gate 8's, and the rule is stated there.

Convergence is enforced as written: `run_cross_model_validation` reads each
fitted model's `policy_passed` per stage and excludes a candidate that failed,
recording its failures as the rejection reason. Runtime is recorded by
`_operation_scope` but is not yet a constraint, because no budget is declared
anywhere; see Gate 7.

## 10. Feature Importance And Interpretation

After selecting candidates on training folds:

1. compute repeated permutation importance on held-out validation predictions
   using NLL or deviance;
2. permute correlated contextual and room-composition blocks together;
3. report Model B total and probability importance separately;
4. report NB2 coefficient intervals and incidence-rate ratios;
5. report Bayesian NB2 + Dirichlet-Multinomial posterior intervals and posterior-predictive scenario curves
   separately by stage; and
6. use SHAP only as an optional local supplement.

All interpretation is predictive association, not causal effect.

## 11. MLflow Experiment Design

Add MLflow after model/result and experiment-runner contracts are stable. The
runner must produce the same `CrossValidationExperimentResult` with tracking
enabled or disabled; the adapter logs that result afterward.

Read tracking URI and experiment name from environment variables with a
documented local default. A parent run may group a complete comparison that
shares one dataset hash, split manifest, feature specification, and master
seed. Each complete model configuration is a distinct nested child run; this
includes a separate `DirectCohortModel` run for each `family=poisson`,
`family=nb2`, and `family=normal`. A child run owns its train-only fold
construction, Optuna studies, cohort refits, predictive draws, and evaluation.
It must never silently select a distribution family or use outer-test evidence
to select one.

For Model A, tag every family run with `model_class`, `family`,
`objective_family`, data/schema/split-manifest hashes, test-lock status, master
seed, and source revision. Log the model metadata verbatim, including each
cohort's Optuna trial history, selected parameters, derived backend seeds, and
Normal scale or NB2 dispersion. Store trial tables as artifacts by default;
use nested per-trial or per-fold MLflow runs only when their operational value
outweighs the additional run volume.

Compare Poisson and NB2 with the same held-out cohort count NLL, alongside
point accuracy and calibration. Normal is a requested continuous comparator:
compare it using MAE, RMSE, interval coverage, and residual diagnostics, not
by ranking its log density against discrete Poisson/NB2 log masses. Choose a
family only from training/CV evidence, record that decision before opening the
outer lockbox, and report outer-test results for pre-declared candidate runs
without feeding them back into selection.

Log:

- simulator configuration/data/schema hashes and package versions;
- split/fold manifests, seeds, partition summaries, and test-lock status;
- likelihood, features, hyperparameters, priors, preprocessing, calibration,
  and uncertainty settings;
- scalar and tidy fold/aggregate/final metrics;
- predictions, bootstrap summaries, calibration, diagnostics, plots, and
  importance outputs;
- fitted artifacts and load/predict smoke-test results; and
- Pyro sampler settings, posterior/convergence summaries, and posterior
  predictive artifacts.

For Pyro, log a reconstructable state bundle, schema, priors, code/dependency
metadata, and tested loader rather than opaque pickling. Tag interrupted or
failed runs explicitly. Never log private simulator latent effects.

Rewrite `notebooks/02_model_fitting.py` as a thin client that builds/loads data,
displays manifests, launches package experiment APIs, and displays MLflow
comparisons. Expensive Bayesian and final-test actions require explicit runs.

## 12. Dependency Groups

This section originally planned a separate `bayesian` optional dependency
group. That plan was not followed: `torch` and `pyro-ppl` were instead added
as **core** `[project.dependencies]` in `pyproject.toml`, so importing
`age_group_prediction` always requires both packages, and there is no
`bayesian` group to install. The table below reflects the actual
`pyproject.toml` groups as of the Gate 5 remediation, not the original plan.

| Group | Actual packages | Purpose |
|---|---|---|
| (core, not a group) | `torch`, `pyro-ppl`, plus the rest of `[project.dependencies]` | Frequentist and Bayesian models alike; always installed |
| `notebook` | `marimo`, `nbformat`, `ruff` | Marimo research notebooks and linting |
| `test` | `pytest` | Fast test suite |
| `validation` | `statsmodels` | Slow calibration/coefficient-recovery checks |
| `tracking` | `mlflow>=3.16,<4` | Optional MLflow tracking adapter (Gate 7); tracking tests skip without it |

There is no `explainability` group; add one to `pyproject.toml` when it is
implemented, rather than assuming it exists. Verify
Python 3.13 compatibility before pinning versions. Each optional group must
install/test independently and must not break simulator or EDA imports.

## 13. Implementation Gates And Exact Agent Routing

Use exact model labels already established in repository routing documents:

- `GPT-5.6 Terra`: statistically sensitive likelihood/Bayesian acceptance;
- `GPT-5.6 Sol`: cross-module implementation and integration;
- `Sonnet 5`: bounded application code and notebook integration;
- `GPT-5.5`: conventional MLflow integration;
- `Codex`: focused implementation and repository verification; and
- `GPT-5.6 Opus`: independent high-risk statistical/design review.

Do not assign unversioned Haiku in this plan: the repository does not identify
which Haiku version is available. Add it after recording the exact picker
label; suitable work is mechanical documentation, fixtures, or repetitive test
scaffolding, not statistical acceptance.

### Gate 1: Modeling Configuration And Locked Data Splits

**Status: revised model implementation complete; orchestration remains in Gates 6-7.**

**Implementation: GPT-5.6 Sol**
**Verification: Codex**
**Statistical review: GPT-5.6 Terra**

Implement typed `modeling_config.py`, build the canonical modeling table,
move the outer split and train-only fold generation into `data_splitting.py`,
serialize the outer lockbox manifest, implement the shared RNG convention, and
prove final-test IDs cannot enter CV.

Acceptance: schema/accounting/leakage tests pass; split is deterministic and
row-order invariant; singleton behavior is correct for both the outer split and
the fold generator; every non-singleton training building is validated exactly
once and any excluded building is reported; manifests replay exact assignments
from serialized form; test targets are not evaluated or summarized during
tuning.

Evidence: `tests/unit/test_modeling_data.py` covers typed defaults and schema
overrides, table invariants, default and caller-provided RNG behavior (including
generator advancement across calls), row-order invariance for the outer split
and for folds, JSON outer-manifest round trip and replay, changed-source,
changed-provenance and swapped-partition rejection, requested-versus-realized
holdout fractions, neighborhood guarantees, uniform validation exposure with the
unvalidated set named explicitly, and proof that validation-fold IDs are drawn
only from the outer training partition.

Independent validation: see `docs/GATE_VALIDATION_FINDINGS.md`.

### Gate 2: Shared Foundations

Gate 2 is completed in order as Gates 2A, 2B, and 2C. All three must pass before
Gate 3 begins.

#### Gate 2A: Feature Engineering Foundation

**Status: complete.**

**Implementation: GPT-5.6 Sol**
**Verification: Codex**
**Statistical review: GPT-5.6 Opus**

Implement immutable `FeatureSpec`, named component defaults, validated candidate
factories, and `FittedFeatureTransformer` in `feature_engineering.py`. Fit all
learned preprocessing state on the fit fold only. Keep deterministic room-share
construction in Gate 1 and keep candidate evaluation out of the transformer.

Acceptance: transformation before fit fails; output feature names and order are
stable; learned means, scales, knots, and category state use only fit-fold data;
changing validation values cannot change fitted metadata or fit matrices;
configured category references and unknown-category behavior are honored;
identifiers, targets, and private effects never enter transformed matrices;
interaction hierarchy and component restrictions are enforced; redundant room
representations are rejected; zero-total row filtering occurs only after
probability preprocessing is fitted; specs and metadata are JSON serializable.

Evidence belongs in `tests/unit/test_feature_engineering.py`, including exact
output contracts for every default and candidate specification.

Implemented evidence: immutable component-aware `FeatureSpec`; named tree,
total-count, and age-probability defaults; ordered candidate factories
(`experiment/candidates.py`, added during Gate 6 validation);
schema-driven `FittedFeatureTransformer`; separate log-exposure output;
fit-fold scaling and spline state; deterministic categorical encoding and
interaction names; target-free row-order-invariant fit hashes; serializable
scaling, category-reference, and spline-knot metadata; and 24 focused Gate 2A
tests covering leakage isolation, invalid specifications, custom schema names,
all candidate families, zero-total preprocessing order, and stable outputs.

#### Gate 2B: Shared Model And Result Contracts

**Status: complete.**

**Implementation: GPT-5.6 Sol**
**Verification: Codex**

Implement `BaseAgeGroupModel`, `PredictionResult`, and `EvaluationResult`
against the Gate 2A feature contract. Models own fitted preprocessing or fitted
pipelines; callers pass immutable feature specs rather than globally
preprocessed matrices.

Acceptance: synthetic interface models pass; unfitted operations fail clearly;
prediction shapes and invariants hold; feature specifications and preprocessing
summaries appear in JSON-serializable metadata; default and caller-provided RNG
behavior is reproducible.

Implemented evidence: `PredictionResult`, `EvaluationResult`, and
`ParametricDistributionSpec` moved unchanged into model-independent
`results.py`, breaking the import cycle between `models/base.py`, `metrics.py`,
and `evaluation.py`, with every legacy import path (`age_group_prediction`,
`age_group_prediction.models`, `age_group_prediction.models.base`) preserved as
an alias to the same class objects. `BaseAgeGroupModel.evaluate` is concrete:
it predicts through the fitted model, then delegates to the shared
`evaluate_predictions` orchestrator instead of an abstract per-model hook, so
evaluation never sees feature matrices. `fit`, `predict`, and `evaluate` each
run inside a shared `_operation_scope` context manager that tracks the active
operation and commits `fit_duration_seconds`, `predict_duration_seconds`, and
`evaluate_duration_seconds` in JSON-serializable metadata only when the
operation succeeds; a failed fit clears all recorded durations. 34 focused
contract tests cover lifecycle, RNG reproducibility, import-path identity, and
success-only duration commits.

#### Gate 2C: Metric And Evaluation Interfaces

**Status: complete.**

**Implementation: GPT-5.6 Sol**
**Verification: Codex**
**Statistical review: GPT-5.6 Opus**

Implement metric protocols/classes, predictive NLL, randomized-PIT
transformations, and the neighborhood-cluster-bootstrap evaluator. Evaluators
consume prediction results, not feature matrices, and remain independent of
feature fitting.

Acceptance: metric golden tests cover zeros, invalid predictions, posterior
integration, and randomized PIT; bootstrap and stochastic transformations are
reproducible for omitted and caller-provided RNGs.

**Reopened during Gate 6 acceptance.** `PredictiveNegativeLogLikelihood` and
`RandomizedPIT` took a `source` field annotated as a `Literal` but never
validated at runtime, and `__post_init__` and `compute` each interpreted it
independently. An unrecognized string therefore left the declared
`required_capability` on the parametric branch while `compute` silently took
the pointwise branch, so the metric misreported what payload it needed and
scored by a different path than it advertised. Nothing raised.

Both are now discriminated unions rather than flag-carrying classes:
`ParametricPredictiveNegativeLogLikelihood` and
`PointwisePredictiveNegativeLogLikelihood`, and `ParametricRandomizedPIT` and
`DrawsRandomizedPIT`. Each variant declares its own `source` and
`required_capability` as class constants and implements only its own scoring
path, so the two can no longer disagree. The former names are retained as
abstract bases that carry the shared identity, validation, and reduction, which
keeps `isinstance` checks and the metric-table key (name, target, aggregation
level) identical across variants — the key identifies the score, not how the
model supplied it. Constructing a base directly raises. `default_metric_set`
is now the single place a `nll_source` or `pit_source` string is parsed into a
class, and it rejects unknown values, which is the boundary that matters if
these are ever driven from configuration.

Observed impact was confined to the declared capability and the code path: the
three models derive their pointwise values from the same parametric families,
so the Gate 6 acceptance run produced identical metric values either way. A
model whose pointwise values are not derived from its declared families, or
which supplies only one of the two payloads, would have been mis-scored or
failed with a misleading error.

Implemented evidence: typed `MetricResult`/`Metric` protocol; point (MAE, RMSE,
mean bias, $R^2$, mean Poisson deviance), predictive-NLL, composition
(log loss, Brier), reconciliation, interval (coverage, width, WIS), and
randomized-PIT metric classes; a config-driven `default_metric_set(...)`
factory that sources Poisson clipping, PIT bins, and reconciliation tolerance
from `EvaluationConfig`/`PredictionValidationConfig` rather than literals and
keeps identical `(metric_name, target)` shape across Model A/Model B/Bayesian
NB2 + Dirichlet-Multinomial NLL interpretations; `available_prediction_capabilities` plus target-aware
capability enforcement in `evaluate_predictions` and
`neighborhood_cluster_bootstrap`, which reject unsupported metrics by default
with one clear error naming the metric, target, and missing capability, and
expose an explicit `on_missing_capability="skip"` opt-in with skipped
definitions recorded in evaluation metadata. Closed-form point/composition
metrics are computed directly in NumPy (sklearn is retained only as a tested
oracle) and the bootstrap precomputes per-neighborhood row indices for
performance. 54 focused metric/evaluation tests plus the full fast suite
back this gate.

### Gate 3: Model A

**Status: complete.**

**Implementation: Sonnet 5**
**Verification: Codex**
**Statistical review: GPT-5.6 Terra**

Implement direct LightGBM cohort trees under Normal, Poisson, and custom NB2
objectives, deterministic Optuna CV tuning, normalized probabilities, and
optional family-specific bootstrap predictive intervals.

Start from `DEFAULT_TREE_FEATURE_SPEC`. Tune tree capacity before evaluating
any explicit nonlinear or interaction candidate from Gate 2A.

Acceptance: shared contracts pass; outputs are finite/nonnegative;
probabilities normalize with tested zero fallback; preprocessing has no fold
leakage; objective derivatives and total-distribution semantics are tested;
bootstrap, tuning, and point results are reproducible within tolerance.

**Independently validated 2026-09-12 — ACCEPT WITH CONDITIONS**, conditions
since discharged. All six acceptance clauses hold, re-derived rather than taken
from the suite: NB2 derivatives match an independent hand derivation exactly and
finite differences to 9.1e-6; the NB2 total convolution matches an independent
`np.convolve` implementation to 4.4e-15 and normalizes to 1.0. Two defects
fixed: a pruned Optuna trial could win a study (shared `tuning.py`; see §7.3),
and the family dispatch could launder a Normal scale into an NB2 dispersion.
Four suspicions were withdrawn on evidence. The suite could not distinguish the
model from a constant predictor — a recovery test now closes that. See
`docs/GATE_VALIDATION_FINDINGS.md`.

### Gate 4: Model B

**Status: complete.**

**Implementation: GPT-5.6 Sol**
**Verification: Codex**
**Statistical review: GPT-5.6 Terra**

Implement explicit Poisson/NB2 total regression candidates, grouped weighted
multinomial probabilities, independent Optuna tuning, evidence-gated train-only
temperature calibration, reconciled means, predictive simulation, and cluster
bootstrap uncertainty.

Treat each fixed total/probability feature-spec pair as a named experiment
candidate. Tune total and probability regularization independently on identical
train-only folds. Compare structural forms and total families between sibling
runs, then freeze both selected specs after train-only CV.

Acceptance: NB2 analytic tests pass; grouped and expanded fits agree on a
fixture; zero totals work; targets never enter predictor matrices;
probabilities normalize; calibration uses grouped out-of-fold predictions and
cannot worsen its selection NLL; expected cohort counts reconcile to totals.

**Independently validated 2026-09-12 — ACCEPT WITH CONDITIONS.** All eight
acceptance clauses hold. The central claim verifies exactly: the four pointwise
keys sum to the true joint log mass against an independent scipy oracle
(7.1e-15 NB2, 1.4e-14 Poisson), so the total is genuinely not double-counted
against cohort scores conditional on it. Both §7.1 initialization formulas are
exact, the exposure offset is a true offset, and grouped/literal equivalence
holds to 4.2e-17 across the full C range. Fixed: a bootstrap replicate omitting
a cohort killed the whole prediction (see §7.3); the search space never reached
metadata; `selection_improvement_tolerance` was dead configuration. Two
suspicions withdrawn — notably that the duplicate NB2 formula should be
deduplicated, when in fact the project's implementation is *more* accurate than
scipy's at small means. **One condition remains open: the calibration retention
rule (G4-1)**, whose retain/reject decision is seed-driven; it is carried into
the Gate 5–6 review. See `docs/GATE_VALIDATION_FINDINGS.md`.

Implemented evidence: `IndependentTotalProbabilityModel` fits an explicitly
configured Poisson or NB2 total regression with a fixed log-apartment exposure
offset and a grouped weighted multinomial probability model without literal
child expansion. One instance owns fixed total/probability feature specs as its
run identity. Seeded Optuna studies tune total L2 and multinomial $C$
independently over identical train-only known-neighborhood folds and retain
trial and fold-level evidence. Family comparison is intentionally owned by
later experiment orchestration rather than performed silently inside the
model. Probability calibration uses
evidence-gated scalar temperature scaling on grouped cross-fitted logits.
Predictions reconcile expected cohorts by construction; uncertainty combines
neighborhood-cluster refits with family-specific totals followed by
multinomial-cohort simulation, and only the total receives a marginal
parametric declaration.

**Reopened during Gate 5 remediation.** `_pointwise_log_probabilities` in
`models/independent_total_probability.py` previously emitted
`n_bk * log p_bk` per cohort — a multinomial log *kernel* missing the
coefficient $\log(y! / \prod_k c_k!)$. This was not a problem in isolation,
but once Gate 5 added an analogous Dirichlet-multinomial composition score
that does include its coefficient, a future Gate 6 joint-NLL ranking would
have been dominated by that missing-coefficient artifact rather than by a
genuine likelihood difference (measured at 3.8 nats at mean total 6, rising
to 93 nats at mean total 90). Model B now uses a new
`multinomial_prefix_log_masses` helper in `distributions.py` and emits
conditional multinomial log masses that sum exactly to
`scipy.stats.multinomial.logpmf` (0.0 measured error), decomposing
composition into per-cohort conditional scores the same way the Bayesian
model does, so Model B and the Bayesian model are comparable key by key.

That comparability does **not** extend to Model A. Model A's per-cohort keys
are *marginal* log masses of independent cohort distributions, while Model B's
and the Bayesian model's are *conditional* on the preceding cohorts in schema
order. The conditional values move between keys if the cohort order changes,
and the last cohort's value is 0.0 up to rounding (about 1e-13). A per-target NLL table would
therefore show Model B and the Bayesian model beating Model A on
`n_highschool` by exactly Model A's `n_highschool` NLL, for no modeling reason.
Only the sum over all keys is order-invariant, and that sum is a joint score
only for Model B and the Bayesian model: Model A's `"total"` key is a
convolution of its cohort distributions, so adding it to the cohort keys
counts the same information twice.

`multinomial_prefix_log_masses` used to clip probabilities to
`np.finfo(float).tiny` and did not check that they summed to one. A positive
count against a zero probability then scored about -708 instead of $-\infty$,
which capped the penalty for an impossible outcome at a finite value. It now
rejects probabilities that do not sum to one (tolerance 1e-9) and uses
`xlogy`, so that case scores $-\infty$, matching
`scipy.stats.multinomial.logpmf`. In Model B's per-cohort differences, the
first impossible prefix gives $-\infty$ and every later key becomes NaN,
because $-\infty-(-\infty)$ is undefined. `PredictionResult` rejects any
non-finite pointwise value, so the prediction fails loudly at that finiteness
check rather than entering an NLL average. Model B feeds it a float64 softmax, which underflows to an exact
zero only for logit gaps above about 745, so this is unreachable in practice.

### Gate 5: Bayesian NB2 + Dirichlet-Multinomial In Pyro

**Implementation: GPT-5.6 Terra**
**Verification: Codex**
**Independent statistical review: GPT-5.6 Opus**

**Status: accepted. A first independent statistical review returned REJECT and
a second adversarial review returned ACCEPT WITH GAPS. A final review of the
remediation returned ACCEPT WITH GAPS with no blockers. The full-profile
acceptance run is done and passed. Every finding is fixed and verified; the
items under "Open items" below belong to Gate 6.**

The final review was run by Claude Fable 5.1, a different model from the
implementer, rather than by the GPT-5.6 Opus named in the routing. It
confirmed that the joint NLL is the posterior-predictive joint log score for
both Model B and the Bayesian model, reproduced the prior-predictive and
SciPy-agreement figures, and mutation-tested the new tests. Its findings were
one untested pass-through (the scope on `BaseAgeGroupModel.predict`'s
validation-config rebuild, now covered by
`test_validation_config_rebuild_preserves_pointwise_scope`) and wording: the
last cohort's value is zero only up to rounding, the F9 failure surfaces as a
NaN caught by the finiteness check, and a stale Gate 2C test count. All are
fixed.

The independent review found that the prior-predictive plausibility check was
blocking: it thresholded the 5th/95th percentiles of *realized* cohort shares
against a boundary value, which fired on every run under the configured
priors regardless of data scale, and because prior-predictive violations
reused the full inference profile's `action="error"`, the full profile
aborted before NUTS ever ran. The review also found that the acceptance-rate
diagnostic could never fire (Pyro's `"acceptance rate"` measures
multinomial-sampler moves, not the Metropolis accept probability, and sits at
~1.0 regardless of sampler health), that pointwise log probabilities omitted
the composition likelihood, and that the resulting composition scores would
not have been comparable to Model B's multinomial scores in a future Gate 6
ranking. All of these are now fixed:

- The prior-predictive check gained its own `action` field
  (`BayesianPriorPredictiveConfig.action`), independent of the per-profile
  convergence action, and no longer thresholds realized cohort shares. The
  first replacement compared the prior *expected* composition $E[p_k]$ with
  `[minimum_expected_share_ratio, maximum_expected_share_ratio]` multiples of
  the uniform share. The second review found that this check **could not
  fire for any configurable prior**. The composition logit priors are
  zero-mean, so averaging $p_k$ over draws keeps it near uniform however
  diffuse they are, and for three cohorts the ratio tends to about 0.75 and
  1.125 in the limit. $\kappa$ never enters it. An earlier version of this
  section claimed that "a degenerate prior still trips it"; that was wrong. The
  bounds remain but are not load-bearing. The statistics with power are
  `expected_dominant_share` ($E[\max_k p_{b,k}]$, limit 0.85), which catches
  diffuse logit priors, and `concentration_quantile` (5th percentile of prior
  $\kappa$, limit 1.0), which catches a degenerate concentration prior.
  Raising the composition prior scales to 100 and 20 kept the expected-share
  ratios within 0.69-1.19 while the dominant share reached 0.994 and fired.
  `test_prior_predictive_checks_detect_degenerate_priors` pins one
  pathological prior per active statistic.
- The convergence diagnostic and config field are renamed
  `minimum_acceptance_rate` -> `minimum_mean_accept_prob` and now capture the
  kernel's running mean accept probability inside the `InstrumentedNUTS`
  `sample()` override, because `MCMC.cleanup()` resets it after the run. A
  real fit reports ~0.906 and ~0.892 against a `target_acceptance` of 0.9.
  Because a floor of 0.6 almost never fires at that target, the second review
  added a ceiling, `maximum_mean_accept_prob = 0.98`, compared with the
  largest per-chain value. A chain that accepts nearly everything is the
  signature of a step size collapsed in a funnel.
- `models/bayesian_inference.py` pins the expected `NUTS._build_tree`
  parameter tuple and raises before fitting if a Pyro upgrade changes it. If
  the recursive hook never fires, every recorded depth is zero; that case
  used to pass silently as zero saturation. Both tree-depth fields are now
  reported as `None`, which fails the policy.
- Stage diagnostics and the prior-predictive summary are recorded before the
  policy is enforced. A policy failure under `action = "error"` raises a
  `RuntimeError` carrying `stage`, `diagnostics`, and `failures`, so a failed
  expensive run no longer destroys the evidence of why it failed.
- Pointwise scores now include the composition likelihood: keys are
  `{"total", *cohort_names}`, where each cohort entry is a conditional
  Dirichlet-multinomial log mass (via the new
  `dirichlet_multinomial_prefix_log_masses` helper in `distributions.py`)
  that sums to the posterior-integrated joint composition score. The joint
  log mass matches `scipy.stats.dirichlet_multinomial.logpmf` to about 1e-13
  at this project's scale; an earlier "8.9e-16" figure came from a single
  case. The error grows with scale: the reviewer measured a worst case of
  4.4e-11 over 500 random cases, and totals up to 5,000 with very large
  concentrations reach about 1e-9. `get_metadata()` declares this contract
  under `pointwise_log_probability_scope`.
- Model B's pointwise composition scores were changed to match (see the Gate
  4 note above), so a future Gate 6 joint-NLL ranking of Model B against the
  Bayesian model is not dominated by a missing multinomial coefficient. The
  per-cohort values are conditional, so they depend on schema order, the last
  cohort's value is 0.0 up to rounding, and they are **not** comparable to Model
  A's marginal per-cohort values.
- `_fit_model` now asserts observed cohort vectors sum to the observed total,
  documenting that Pyro's `DirichletMultinomial.log_prob` conditions through
  that sum rather than through its `total_count` argument (which it ignores).
- `torch.set_num_threads(1)` / `torch.set_default_dtype(torch.float64)` are
  now scoped to one fit via `_scoped_torch_runtime_settings()` and restored
  afterward, including on exception.

See [Implementation Cautions](BAYESIAN_CONDITIONAL_MODEL.md#implementation-cautions)
for the full before/after accounting, including the one caution (unseen-
neighborhood fallback draws) that was explicitly accepted rather than
changed.

See [Bayesian conditional model guide](BAYESIAN_CONDITIONAL_MODEL.md) for the
model equations, Pyro syntax, named sites, tensor shapes, inference flow, and
runtime workarounds.

Implement hierarchical NB2 and conditional Dirichlet-multinomial stages,
prior/posterior prediction, NUTS diagnostics, and exact draw reconciliation.

Consume the frozen Model B total and probability feature forms. Refit their
preprocessing state on the Bayesian training data, but do not introduce or
search new forms during Pyro fitting.

Acceptance: Torch/Pyro NB2 moments match project definitions; priors are
plausible; tiny-data MCMC completes; diagnostics are extracted/thresholded;
posterior predictive shapes are stable; every draw reconciles exactly.

Local evidence: Python 3.13/macOS ARM resolves Torch 2.14.0 and Pyro 1.9.1;
the focused Bayesian/configuration/distribution suite passes; a real two-chain
tiny-data fit completes both NUTS stages and produces strictly JSON-serializable,
exactly reconciled integer draws including unseen-neighborhood fallback; and
`PYTHONPATH=src uv run pytest -q -m "not calibration and not slow"` passes
repository-wide. This section deliberately does not cite a test count, which
drifts as coverage grows; run the command for the current number. The repo
sets no pytest `addopts`, so `pytest tests/unit` alone does **not** deselect
slow tests.

`tests/validation/test_bayesian_recovery.py` holds the only unmocked NUTS
tests, both marked `slow`; run them with
`PYTHONPATH=src uv run pytest -q -m slow tests/validation/test_bayesian_recovery.py`.
One checks end-to-end invariants. The other fits simulated data (600
buildings, 30 neighborhoods) whose generating values sit away from their
prior means. It asserts that each of the four scalar parameters has a
posterior standard deviation below 0.4 of its prior standard deviation, that
at least three of those four 95% intervals cover the truth, and that at least
70% of total-coefficient and 80% of composition-coefficient intervals do.
An earlier version asserted coverage only and **passed with the NB2
likelihood deleted**, because a posterior equal to a prior centred near the
truth still covers it. The contraction assertions are what give the test
power: with the NB2 factor multiplied by zero, the current test fails because
`total_intercept` stays at 0.96 of its prior standard deviation. It demonstrates that one reduced-profile fit at one seed learns every
scalar and centres it correctly. It is not simulation-based calibration, it
does not show calibrated coverage across seeds, and it does not assert
convergence: its total stage currently warns under the reduced policy
(worst R-hat about 1.09, minimum ESS about 26).

**Full-profile acceptance run (done, passed).** Data: `configs/stage1.toml`
simulator, seed 42, 60 neighborhoods, 251 buildings split 192 train / 59 test
by `split_known_neighborhood_buildings(..., rng=np.random.default_rng(42))`,
modeling table SHA-256 prefix `51c46a6467d19892`. Profile: 4 chains, 1,000
warmup and 1,000 retained draws per chain, target acceptance 0.9, maximum
tree depth 10, diagonal mass matrix; 6.7 minutes. Both stages pass the strict
full-profile policy:

| Stage | Worst R-hat | Min ESS | Divergences | Mean accept | Max depth | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| Total | 1.0029 | 1456 | 0 | 0.907 | 7 / 10 | 0.000 |
| Composition | 1.0017 | 1953 | 0 | 0.923 | 6 / 10 | 0.000 |

The prior-predictive summary gave an expected dominant share of 0.661 (limit
0.85), a $\kappa$ 5th percentile of 1.521 (limit 1.0), children-per-apartment
quantiles of $[0.0, 0.077, 2.468]$, and no violations. On the 59 held-out
buildings, reconciliation was exact (maximum error 1.7e-13), every draw was an
integer and nonnegative, the joint NLL (the sum of the pointwise keys) was
8.1008 per building, total MAE was 7.793, and coverage was 0.763 at 80% and
0.915 at 95%. The run deliberately used `action = "warn"` and applied the
strict policy afterward, so a single threshold miss could not discard the
evidence.

R-hat and ESS are Pyro 1.9.1's `split_gelman_rubin` and classic
autocorrelation ESS, not rank-normalized split-R-hat or bulk and tail ESS. An
R-hat of at most 1.05 on the older statistic is less sensitive to
disagreement in scale and in the tails, which is exactly how the
`neighborhood_scale` funnel would present. E-BFMI is not computed.

Items carried into Gate 6 and their implementation status:

- **Joint predictive NLL is now used by the Gate 6 selection policy.**
  Section 9.3 names it the primary distributional criterion. During Gate 5
  remediation, `JointPredictiveNegativeLogLikelihood` (`joint_predictive_nll`,
  target `joint`) was added to the Gate 2C metric registry as a purely
  additive change: existing metrics and `default_metric_set` defaults are
  unchanged, and the metric is added only when a `joint_nll_interpretation`
  is passed. `PredictionResult` gained an optional
  `pointwise_log_probability_scope`. Model B and the Bayesian model declare
  `"sequential_joint"`, Model A declares `"marginal"`, and only the former
  provides the `joint_pointwise_log_probabilities` capability. Model A is
  therefore excluded by the existing capability gate instead of receiving a
  double-counted score. Neighborhood-bootstrap resampling preserves the
   scope. `sequential_joint_selection_policy(...)` uses it as the primary
   criterion for the independent total/probability and Bayesian conditional
   approaches. `direct_cohort_selection_policy(...)` instead sums only the
   three independent cohort NLLs and explicitly excludes the separately
   scored total, avoiding double counting. Normal direct-cohort runs remain
   diagnostic comparators unless a caller declares a point-metric policy.
- **"Consume the frozen Model B feature forms" is enforced at selection
   freeze.** The
  Bayesian model takes its total spec from `fit(feature_spec=...)` and its
  probability spec from the constructor, so using Model B's frozen forms is
   caller discipline at the estimator boundary. Gate 6 rejects a freeze unless
   the selected independent and Bayesian candidates declare identical
   `total_count` and `age_probability` feature specifications. The
  `[bayesian_conditional] total_feature_spec` and `probability_feature_spec`
  config keys, which were validated but never read and so implied
  configurability that did not exist, have been removed. A TOML that still
  sets them is now rejected at load time.
- **`[randomness] default_seed` is wired at the experiment boundary.**
   `run_cross_model_validation` accepts an `ExperimentConfig` and resolves the
   master seed in the precedence declared in section 4: an explicit
   `master_seed` argument, then `experiment_config.randomness.default_seed`. A
   call supplying neither is rejected rather than silently seeded from a
   default, and the freeze artifact records both the resolved seed and its
   `master_seed_source`. The same config supplies `evaluation_config` when one
   is not passed explicitly, so bootstrap evidence is not silently skipped. The
   runner derives order-stable child seeds by candidate, fold, and operation and
   passes fresh NumPy generators into model fit, prediction, evaluation,
   bootstrap, and permutation work. The neighborhood bootstrap receives its seed
   rather than a prebuilt generator, so its metadata records the seed instead of
   `None`.
- All evidence is from simulator data; misspecification effects on real
  exported data are unmeasured. Posterior prediction loops over draws and
  rows in Python and will not scale to thousands of buildings without
  vectorization.

### Gate 6: Cross-Model CV, Selection, And Interpretation

**Implementation: GPT-5.6 Sol**
**Verification: Codex**
**Statistical acceptance: GPT-5.6 Opus**

**Status: implementation complete; independent statistical acceptance ran and
required remediation, which is now applied and verified.**

Run identical training-only folds, freeze selected features/hyperparameters/
priors, and produce out-of-fold comparisons and validation-only importance.

Acceptance: all candidates share folds; metric definitions are visible;
accepted/rejected feature blocks are recorded; no test metrics exist; Opus
approves likelihood comparability, calibration, and selection logic.

Implemented evidence: the `experiment/` package defines an explicit typed candidate
registry, validates the outer-training frame and every fixed fold against the
lockbox manifest before constructing a model, fits fresh public model instances
with fold-local preprocessing, and stores fold-scoped predictions because the
validation folds are repeated splits rather than a disjoint K-fold partition.
It records visible metric definitions, fixed-prediction neighborhood-bootstrap
intervals, model metadata, calibration decisions, and purpose-specific seeds.
Selection is deterministic and produces one frozen candidate per public
approach with rejected alternatives and reasons. Bayesian selection is rejected
unless its total and probability feature forms equal the selected independent
model forms.

Repeated permutation importance runs only after selection, only through each
fold's fitted public model, and only on that fold's validation rows. Declared
raw feature blocks are schema-checked; one permutation is shared across every
column in a block. Evidence identifies the candidate, component, metric,
target, block, validation IDs, repeat, and seed, and reports repeat-level and
aggregate degradation. The runner has no test-frame argument, records
`test_metrics = null` in the freeze artifact, does not perform the Gate 8
full-training refit, and has no MLflow dependency.

Verification evidence: 22 focused Gate 6 tests pass; the adjacent split,
contract, metric, and evaluation suite passes; the Gate 6 plus three
model-focused suites pass; and
`PYTHONPATH=src uv run pytest -q tests/unit tests/validation tests/characterization`
passes 420 tests. The full run emits the already documented reduced-profile
Bayesian recovery warning (worst R-hat 1.086 and minimum ESS 25.66) but has no
failures, skips, or deselections. Test paths must be spelled in full: there is
no `testpaths` setting, so a bare `pytest -q test_experiment.py` resolves
nothing. `pythonpath = ["src"]` is already configured, so the `PYTHONPATH=src`
prefix is redundant for pytest and required only for bare `python`.

Independent statistical acceptance found, and remediation fixed:

- `sequential_joint_selection_policy` referenced `composition_log_loss` at
  aggregation level `building`, while `CompositionLogLoss` declares `child` and
  fixes it with `init=False`. Every real conditional selection therefore failed
  before a model was fitted, and no test caught it because all Gate 6 tests use
  a spy model and a synthetic policy. `MetricReference.from_metric(...)` now
  derives name, target, and aggregation level from the metric object so the
  three keys cannot drift apart, and a test asserts both shipped policies key
  only metrics the registry actually declares.
- Nothing prevented ranking a continuous Normal direct-cohort candidate against
  discrete Poisson or NB2 candidates, comparing log densities to log masses;
  the guard existed only as policy prose. `_validate_likelihood_comparability`
  now rejects that combination under any distributional criterion and directs
  the caller to `selection_role="diagnostic_comparator"`. Candidates sharing a
  measure, such as Poisson against NB2, remain comparable.
- `SelectionCriterion.optimization_direction` was never checked against the
  metric's own declared direction, so declaring `maximize` for `rmse` silently
  selected the worst candidate. Mismatches, including `target_zero` metrics
  such as `mean_bias`, are now rejected before any model factory runs.
- The `scope="cross_validation"` bootstrap band averages replicates paired by
  index across folds, which treats the folds as independent. Validation folds
  are repeated overlapping splits, so that band is narrower than the true
  variability of the fold mean: measured widths were 1.08 to 4.00 per fold
  against 1.41 across folds. The rows now carry `interval_basis` and
  `assumes_independent_folds` so the assumption is visible, and fold-scope
  intervals remain the primary evidence. A correlation-preserving cross-fold
  bootstrap is deferred.
- `fold_coverage_df` reports how often each outer-training building is
  validated. Because `make_validation_folds` draws repeated overlapping splits
  and never validates singleton neighborhoods, some buildings contribute no
  out-of-fold evidence at all: 20 of 50 training buildings in the acceptance
  run. Fold means and standard deviations are over correlated, unbalanced
  samples, and the reported standard deviation is not a standard error.
- Calibration evidence now exposes `temperature`, `fitted_temperature`,
  `retained`, and the raw and selected scores as columns rather than only
  inside a serialized payload, so the retention decision is readable
  downstream. The redundant `calibration_attempted` flag was removed.
- Bootstrap interval evidence now keys on `aggregation_level`, which it
  previously ignored in both grouping and point-estimate lookup.

Confirmed correct under adversarial probing with all three real models: no
holdout ID reaches any fold, prediction, importance payload, or freeze;
`test_metrics` is null and no MLflow dependency exists; results are
reproducible, candidate-order invariant, and genuinely seed-sensitive;
`joint_predictive_nll` equals the negative mean of the summed sequential keys
to ten decimal places; `DirectCohortModel` is excluded from the joint score by
the capability gate rather than double-counted; importance degradation is
signed correctly for both minimize and maximize metrics; and orchestration
touches no private model state.

`experiment.py` was then split into the `experiment/` package (`contracts`,
`policies`, `partitions`, `seeds`, `evidence`, `aggregation`, `selection`,
`importance`, `runner`) with the public surface re-exported unchanged. The
split is behaviour-preserving: the same real three-model run produces
byte-identical freeze, fold-metric, importance, and bootstrap artifacts before
and after.

Known remaining limitation: selection ranks criteria strictly
lexicographically on floats, so exact ties are required before a secondary
criterion can matter, and there is no tolerance band or one-standard-error rule
accounting for fold-to-fold variance.

### Gate 7: MLflow Tracking

**Phase 0 dependency spike: GPT-5.6 Sol**
**Phase 1 frequentist state bundles: GPT-5.6 Opus**
**Phase 2 artifact evidence and MLflow adapter: GPT-5.6 Opus**
**Phase 3 tests, docs, and regression: Codex**
**Integration review: Sonnet 5**
**Final acceptance review: GPT-5.6 Opus**

The phase routing is intentional. Dependency resolution is a bounded packaging
task suited to Sol. State reconstruction and post-run artifact ownership cross
LightGBM, SciPy, scikit-learn, immutable result contracts, and MLflow run
lifecycle, so Opus owns the two high-risk phases. Codex owns the execution-heavy
acceptance phase, while Sonnet reviews whether the adapter remains a simple
optional integration. Terra is needed only if Gate 7 is explicitly expanded to
new statistical outputs from §10.4 or §10.5; those outputs are otherwise out of
scope.

Implement optional tracking, a comparison parent with one sibling Model A run
per distribution family, artifact schemas, failure tagging, and model
reconstruction checks. Tracking is orchestration only: it must not add
family-selection behavior to the model implementations.

Acceptance: tracking-on/off results are equivalent; required runs/tags exist;
metrics/artifacts are logged; each loss-family run records independent
train/CV-only tuning evidence; cross-family metric comparability is explicit;
interrupted runs cannot appear complete; frequentist and Pyro artifacts pass
reload/predict smoke tests.

**Implementation sequence.** Phase 0 adds the PEP 735 `tracking` group and must
prove the chosen MLflow distribution resolves against Python 3.13, pandas 3,
and NumPy 2.5 before any adapter code is written. Prefer `mlflow-skinny` when it
provides all required tracking APIs. Phase 1 adds hash-verified, JSON-safe state
bundles and tested loaders for `DirectCohortModel` and
`IndependentTotalProbabilityModel`, mirroring the existing Bayesian contract
without native MLflow flavors or pickling. Phase 2 captures exact fitted-fold
bundle and reload-smoke evidence in an MLflow-neutral experiment component,
then adds the optional `tracking.py` adapter: one parent per comparison and one
nested child per complete candidate configuration. Phase 3 proves the adapter
does not mutate `CrossValidationExperimentResult`, completes isolated-store
tracking tests, updates documentation, and runs the full regression suite.

Tracking configuration is environment-only, with documented local defaults;
the closed-world modeling TOML loader is not extended. Failed comparisons remain
all-or-nothing and receive parent-level failure/interruption records. Notebook
work, outer-test evaluation, runtime selection constraints, and the currently
unimplemented §10.4/§10.5 statistical artifacts remain outside Gate 7.

**Starting position, established by the Gate 6 validation.** These are
measured facts about the code Gate 7 will wrap, not predictions.

- **The reload requirement is already half-built, and it is the half nobody
  expects.** Section 11 asks for a reconstructable state bundle with a tested
  loader "for Pyro". That exists: `BayesianConditionalModel.to_state_bundle()`
  / `from_state_bundle(bundle, *, train_df)`, with five covering tests and no
  pickling — the bundle carries no training data and the loader refits the
  feature transformers from a frame verified against the recorded
  `training_data_hash` and `training_schema_hash`. **Neither frequentist model
  has any equivalent.** Gate 7's "frequentist and Pyro artifacts pass
  reload/predict smoke tests" is therefore new work for Models A and B and a
  wiring exercise for the Bayesian model.
- **Failure tagging has no partial run to tag.** `run_cross_model_validation`
  aborts the whole experiment when any candidate fails after fitting; the
  failure is identified by candidate, fold and operation, but the remaining
  candidates' completed work is discarded. Gate 7 must either depend on the
  runner gaining per-candidate containment first, or state that a failed
  comparison is all-or-nothing and tag it as such. It cannot log a partially
  completed comparison today.
- **Run identity must be injected.** No timestamp, run id, git commit or
  experiment name exists anywhere in `src/`, deliberately: identity is
  content-hash based (`manifest_fingerprint`, `FoldIdentity.fingerprint`,
  `training_data_hash`, `master_seed` plus `master_seed_source`). A git SHA is
  genuinely unavailable from inside the package and must be supplied by the
  tracking layer.
- **Log the candidate set's provenance, not just its outcome.**
  `SelectionFreeze.to_dict()` already carries each policy in full, so the
  declared criteria are recoverable. What it does not carry is where the
  candidate list came from; `experiment/candidates.py` now enumerates section
  5.3's predeclared specs in a stable order, which makes that provenance
  loggable.
- **`validation_building_ids` is an artifact, not a column.** It is a complete
  JSON ID list repeated on every importance row — candidates times folds times
  blocks times repeats.
- **Do not log convergence diagnostics unconditionally.** Pyro 1.9.1's classic
  autocorrelation ESS is not constrained to be positive: a short profile
  produced `minimum_effective_sample_size = -3554.19` during validation. It
  correctly fails the threshold, but a negative ESS is not a quantity worth
  charting. Clamp it, or move to rank-normalized bulk and tail ESS.
- **Runtime is recorded but unused.** `_operation_scope` commits
  `fit_duration_seconds`, `predict_duration_seconds` and
  `evaluate_duration_seconds`, and nothing reads them. Section 9.3 names
  runtime a selection constraint; convergence was implemented during Gate 6
  remediation and runtime was not, because no document declares a budget.
  Gate 7 is the natural place to log durations; making them a constraint needs
  that budget declared first.
- **Sections 10.4 and 10.5 are unimplemented.** NB2 coefficient intervals,
  incidence-rate ratios, and per-stage Bayesian posterior intervals and
  scenario curves are produced nowhere in `src/`. If they are wanted as logged
  artifacts, Gate 7 owns building them, not merely logging them.

**Implementation record.** Gate 7 ran as a single session that implemented all
phases and stopped for approval at each boundary, with an independent review
subagent per phase. Decisions taken with the user override the routing and the
`mlflow-skinny` preference above where they differ, and are recorded here.

- **Phase 0, dependency (complete).** `tracking = ["mlflow>=3.16,<4"]` in
  `[dependency-groups]`. It resolved to mlflow 3.16.0 with pandas 3.0.5, NumPy
  2.5.1 and Python 3.13 unchanged, adding pyarrow 25.0.1, SQLAlchemy 2.0.52 and
  Alembic 1.19.2 and changing no installed version. Full `mlflow` was chosen
  over `mlflow-skinny`: MLflow deprecated the filesystem tracking store in
  3.6-3.7 in favor of a database store, and `mlflow-skinny` ships no SQLAlchemy
  or Alembic, so it cannot use one. The documented local default is therefore
  `sqlite:///mlflow.db`; `mlflow.db`, `mlruns/` and `mlartifacts/` are
  gitignored. Measured behavior the adapter must handle: `KeyboardInterrupt`
  ends a run `FAILED`, not `KILLED`; `start_run(nested=True)` without an active
  parent silently creates a top-level run; param values over 6000 characters
  are truncated with only a warning; client-side artifacts default to
  `./mlruns/<experiment_id>` relative to the working directory.
- **Phase 1, state bundles (complete).** All three families now share one
  bundle contract, format `"2"`, documented in
  `src/age_group_prediction/state_bundle.py`.
  `BaseAgeGroupModel.to_state_bundle()` and
  `from_state_bundle(bundle, *, train_df=None)` own the lifecycle as a template
  method; each model implements `_export_model_state` / `_from_model_state`.
  Deliberate changes from the Gate 5 Bayesian contract (format `"1"`):
  - **Bundles are self-contained.** `FittedFeatureTransformer.to_state()` /
    `from_state()` store fitted preprocessing (means, scales, spline base knots;
    one-hot categories come from the schema), so point predictions, pointwise
    log probabilities and parametric distributions reload without the training
    frame. `train_df` is optional and hash-verified; Models A and B require it
    only for predictive draws and intervals, which refit on neighborhood-cluster
    bootstrap resamples by construction. The Bayesian model never needs it.
    No format-1 bundle was ever persisted, so no migration reader was kept.
  - **The bundle header also pins `implementation_version`**, so fitted state
    from a different implementation of a model is refused rather than reused.
  - **Model A** stores each cohort's trees in LightGBM's text model format
    (`Booster.model_to_string()`), reloaded with `lightgbm.Booster(model_str=)`.
    The model now keeps the fitted `Booster` itself, and `_predict_mean` is the
    single prediction path for tuning, bootstrap refits, fitted and reloaded
    models. It passes `num_threads` explicitly: a bare `Booster` otherwise uses
    every OpenMP thread, which segfaulted alongside torch's OpenMP runtime on
    macOS. `_unvalidated_building_ids` is now cleared on reset.
  - **Model B** stores count-regression coefficients (plus NB2 dispersion), the
    multinomial's `coef_`/`intercept_`/`classes_`, temperature and calibration
    evidence; neither optimizer re-runs on load. Selection folds and the
    retained training frame are never serialized.
  - **Bayesian** neighborhood lookup keys are native Python values, so numeric
    neighborhood IDs serialize (they previously would have failed `json.dumps`).
  - **"No training rows" is precise.** No per-building record or building ID
    is stored. Fitted summary statistics are model state and do appear:
    scaling moments, spline knots spanning the data range, fit row counts,
    LightGBM split thresholds, per-feature `[min:max]` bounds and per-node
    sample counts, Bayesian neighborhood IDs, and per-fold tuning and
    calibration counts. The full list is in `state_bundle.py`.
  - **Restoration is strict.** `restore_config` refuses a payload that does
    not name exactly the config's fields (a missing field would otherwise take
    its default), accepts both `asdict` output and its JSON round trip, and
    per-column statistics are rebuilt in the spec's column order, so a writer
    that sorts keys cannot reorder features. Bundles are checked with
    `json.dumps(..., allow_nan=False)`.
  - **Pre-existing defect fixed in passing.** The one-hot encoder listed
    categories in schema order while receiving columns in spec order; with two
    categorical columns declared in different orders it failed to fit or, under
    `treat_as_reference`, silently mislabelled levels. Categories now follow
    the spec's column order. Default specs have one categorical column and were
    unaffected.

  Evidence: `tests/unit/test_model_state_bundles.py` holds all six model
  configurations (direct Poisson/Normal/NB2, independent NB2/Poisson,
  Bayesian) to one contract: JSON reload with fitting disabled reproduces
  point predictions, pointwise scores and parametric distributions exactly,
  and the fit-describing metadata (per-call durations, seeds and prediction
  counters are not bundle state and are excluded by name); reloads from
  in-memory objects and sorted-key JSON agree; draws and intervals reproduce
  exactly, and bootstrap models refuse draws without the frame; changed rows
  or dtypes and incompatible headers are refused by name; no building ID or
  interior covariate value appears; exported bundles share no state with the
  model. `tests/unit/test_state_bundles.py` covers transformer state for every
  predeclared candidate feature spec and for key order, categorical column
  order, strict config restoration, header and training-frame checks, and
  `TuningResult.from_dict`. The superseded format-1 tests in
  `test_bayesian_conditional.py` were removed. An independent review found no
  blockers; its should-fix findings are the strictness items above. Suite:
  524 passed, ruff clean, one documented Bayesian warning.
- **Phase 2, artifact evidence and tracking adapter (complete).** Run in a
  fresh session briefed by
  `GATE_7_SESSION_HANDOFF.md` (archived).
  - **Artifact capture stays MLflow-free** (`experiment/artifacts.py`).
    `run_cross_model_validation(..., capture_artifacts=False)`; when enabled,
    directly after `get_metadata()` and inside
    `_fold_failure_context(..., "capture_artifacts")`, every candidate×fold
    model is exported, reloaded from its JSON text **without** `train_df`, and
    point-predicted by both models with `PredictionConfig()`, each model with
    its own identically seeded generator (never a runner stream).
    `ReloadCheck` records the largest absolute difference in `total_mean`,
    `cohort_means`, `age_group_probabilities`, and every parametric
    distribution's dispersion and scale, against tolerance 0.0; any difference
    aborts the run. Comparing distribution parameters goes beyond the
    handoff's means-only wording, because the review showed a Model B bundle
    with 5× its NB2 dispersion passing a means-only check. The always-true
    `reloaded` field was dropped.
  - **`ExperimentProvenance` on the result**: outer-training
    `table_hash`/`column_schema_hash`, manifest fingerprint, master seed and
    its source, per-candidate per-component feature-spec fingerprints, package
    versions, and `run_settings` (evaluation config, `require_convergence`,
    `required_approaches`, `capture_artifacts`, fold count, building-ID
    column). The parent run's params therefore derive from the result alone.
  - **`src/age_group_prediction/tracking/`** (written as a single
    `tracking.py` in Gate 7 and split into a package afterwards, without
    behavior change; see the follow-up note at the end of this record), not
    re-exported from the package: `resolve_tracking_settings`, `TrackingContext`,
    `tracked_comparison`, `log_experiment_result`,
    `log_cross_validation_experiment`. One parent run per comparison and one
    nested child per candidate, with the tags, params, metrics and artifacts
    listed in the handoff (§5) and the README. Decisions taken in this session:
    the artifact location comes from `AGE_GROUP_MLFLOW_ARTIFACT_LOCATION`
    (MLflow defines no such variable; default `mlartifacts/` beside a SQLite
    file), and an existing experiment with a different artifact location is
    refused. Every run carries the caller's context tags, and MLflow's inferred
    `mlflow.source.git.*` tags are deleted. A failed or interrupted run is
    tagged `evidence_complete=false` even if logging had finished, so
    completeness means `FINISHED` plus `evidence_complete=true`.
  - **`DirectCohortModel` diagnostics gain `lightgbm_objective`** (`poisson`,
    `regression`, `custom_nb2_gradient`), which the `objective_family` tag
    reports; fitted state and `implementation_version` are unchanged.
  - **Independent review:** one blocker (child runs landed in MLflow's
    `Default` experiment and `./mlruns`, hidden because the boundary script set
    the experiment name) and five should-fix items (a space in the artifact
    path refused the second run; a run failing after logging read complete;
    children lacked context tags while MLflow inferred Git tags from HEAD; the
    means-only reload check; double logging into one parent). All were fixed
    and re-verified, and the nits taken.
  - Evidence: a real five-candidate comparison (direct Poisson and NB2, direct
    Normal as a diagnostic comparator, independent NB2, reduced Bayesian)
    logged to a temporary store with `MLFLOW_EXPERIMENT_NAME` unset; every
    downloaded bundle reloaded and reproduced the runner's predictions exactly.
- **Phase 3, tests, documentation and acceptance (complete).**
  - `tests/unit/test_tracking.py`: an isolated SQLite store per module or
    test, a bundle-capable spy experiment (`tests/unit/bundle_spy.py`) with
    Poisson, NB2 and Normal direct candidates plus Model B and Bayesian
    stand-ins. Covers hierarchy and tags, artifact paths and CSV/JSON schemas,
    metric keys and steps, negative-ESS clamping, settings resolution, `FAILED`
    and `KILLED` parents, failure midway through children, refusals (no
    artifacts, mismatched `train_df`, active run, double logging, logging
    outside the block), an unchanged deep snapshot of the result after logging,
    and an AST scan proving `experiment/` and `models/` never import MLflow.
  - `tests/unit/test_experiment_artifacts.py` (MLflow-free, so it runs
    without the tracking group): capture coverage, capture on/off
    equivalence with a model that consumes its generator for point
    predictions, drifted reloads and lost dispersion aborting the run, and
    provenance.
  - `tests/validation/test_tracking_real_models.py` (`slow`): the real
    five-candidate comparison, with every bundle downloaded from MLflow,
    reloaded without `train_df`, and matched exactly to the runner's
    predictions; per-fold tuning trial tables for each loss-family run.
  - **Defect found and fixed in Phase 3.** Restoring MLflow's process-global
    tracking URI with `mlflow.set_tracking_uri` also rewrote
    `MLFLOW_TRACKING_URI`, which silently pinned later comparisons to the first
    store. The adapter now clears MLflow's setting and restores the variable
    exactly; a test pins it.
  - **Acceptance review: ACCEPT WITH CONDITIONS, conditions discharged.** No
    blocker and no code defect. Its real-model probes confirmed capture on/off
    equivalence, the family and objective tags, and that no latent effect or
    training row reaches the artifacts. Conditions and their discharge:
    - The tuning tests compared only trial numbers, which are identical across
      families. The spy's direct families now have distinct best values, and
      the unit and real tests compare trial values and per-fold
      `tuning/*/best_value` against each run's own metadata.
    - No test proved that a failure after logging overwrites
      `evidence_complete=true` (a mutant survived). The double-log test now
      asserts `false`, and a new test interrupts after logging finished.
    - The findings doc claimed the freeze logs `source_table_hash`; it did not.
      `ExperimentProvenance.split_summary` (strategy, source table and schema
      hashes, row and neighborhood counts, holdout fractions, split sizes, seed
      source; no building IDs) is now logged in `provenance.json` and as
      `split.*` params.
    - The comparability record gave the Bayesian candidate no measure kind and
      omitted likelihood scope. It now records each candidate's pointwise
      log-probability scopes, states that `predictive_nll` is never ranked
      across approaches, and states that measure kinds come from declared
      parametric families.
    - The `family` tag trusted the caller's label. A declared family that
      disagrees with the fitted model's is now refused before any run starts.
    - Docstring coverage was overstated. 49 docstrings were added to
      `models/{direct_cohort,base,independent_total_probability,
      bayesian_conditional}.py` and the Phase 1 test files.
    - Nits taken: runner seeds are logged as `folds/fold_k/seeds.json`; the real
      reload test also compares probabilities; a slow real-model capture on/off
      equivalence test was added; the README states that a complete child never
      implies a complete comparison.
  - Every function and method in every file Gate 7 touched has a docstring.
  - Suite: 567 passing, ruff clean, one documented Bayesian warning. Residuals
    are recorded in `GATE_VALIDATION_FINDINGS.md`, Gate 7.
  - Follow-up outside Gate 7: splitting `tracking.py` into a package and
    extracting the calibration math from `independent_total_probability.py`,
    both behavior-preserving, briefed by
    `MODULE_SPLIT_HANDOFF.md` (archived).
    - **Phase 1 (complete): `tracking.py` became the `tracking/` package.**
      Every definition moved verbatim into `_files`, `settings`, `_metadata`,
      `runs`, `candidates` and `evidence`; `__init__` keeps the module
      docstring and re-exports the unchanged public API and `__all__`.
      Imports run one way (`_files`, `settings`, `_metadata` import nothing
      internal; `runs` uses `settings` and `_files`; `candidates` uses `runs`,
      `_metadata` and `_files`; `evidence` uses all of them). Two departures
      from the handoff's table: the preflight checks (`_require_loggable`,
      `_require_consistent_families`) live in `evidence`, because they read
      model metadata and would otherwise make `runs` and `evidence` import each
      other; and child runs and metadata readers are their own modules, so no
      module holds most of the old file. Tests changed only where they reach
      private names: the midway-failure test patches
      `tracking.candidates._write_candidate_artifacts` (confirmed to fail when
      the injected failure is disabled), the serialization tests use
      `tracking._files`, and the boundary test's package root moved up one
      level. Evidence: a spy comparison logged before and after the split gave
      identical tags, params, metrics, inputs and artifact contents (IDs and
      durations masked); `test_tracking.py`, `test_experiment_artifacts.py`,
      the slow real-model tracking tests and the full suite pass; tracking
      tests still skip without MLflow.
    - **Phase 2 (complete): Model B's pure numerics left the class.**
      `models/probability_calibration.py` holds the two weighted composition
      kernels and `_fit_calibration_temperature`, which returns the applied
      temperature and the diagnostics dict (same keys, order and values; the
      likelihood-ratio comment block moved with it verbatim).
      `models/fold_scoring.py` holds `_total_fold_loss` and
      `_probability_fold_loss`, which `_score_fold` now calls. The class keeps
      everything that records seeds or writes model state: the calibration
      cross-fitting loop, Optuna tuning, bootstrap refits, state export and
      import, and metadata. `implementation_version` stays `"2"`, and the file
      drops from 823 to 710 lines (`wc -l`). The fold-scoring module goes
      beyond the handoff, which scoped only calibration; the user asked for a
      readable, modular Model B. In `_score_fold` the probability seed is now
      drawn before the fold's features are built rather than after;
      preprocessing consumes no randomness, so every seed of a successful fit
      is unchanged. If preprocessing raises on a probability fold, the
      caller's generator has advanced one draw further than before; the fit
      fails either way and its recorded seeds are cleared. Tests changed only
      where they reach moved names: `test_composition_kernels.py` imports the
      kernels from `probability_calibration`, and both `minimize_scalar`
      patches in `test_independent_total_probability.py` target that module
      (confirmed to fail when patched on the old module). Short docstrings
      were added to the two test files' functions that lacked them. Evidence:
      fifteen fits (unit frame with NB2 and Poisson totals, narrow and wide
      searches; well-specified frame, six seeds) produced byte-identical
      canonical records before and after: temperature and calibration
      diagnostics, full metadata including derived seeds and tuning trials,
      state bundles, predictions with draws, intervals and pointwise scores,
      and reloaded predictions. Nine cases retained a temperature and six did
      not, so both retention branches are covered. The focused Model B,
      kernel, state-bundle and calibration-validation tests pass, and the
      full suite is at 567 passing with the one documented Bayesian warning.

### Gate 8: Notebook And Final Lockbox Evaluation

**Implementation: Sonnet 5**
**Verification: Codex**
**Final statistical acceptance: GPT-5.6 Opus**

Rewrite the notebook as a thin client, refit frozen specifications on all
training data, and execute explicit one-time test evaluation through MLflow.

Acceptance: marimo checks/execution pass; final action records frozen selection;
all models receive identical test IDs; final metrics/predictions/intervals/
artifacts are logged; no test-driven change enters the same experiment.

**Gate 8 owns cross-family selection.** Gate 6 freezes one winner *per*
approach and nothing more: `FrozenApproachSelection` is within-approach by
construction, so a validated Gate 6 run ends with three winners, not one.
Section 11 requires the family decision to be made from training and CV
evidence and recorded *before* the lockbox opens, so it cannot be deferred to
whichever model happens to score best on the test set. Gate 8 must therefore
record an explicit cross-family choice, derived from Gate 6 evidence, before
any test prediction runs.

**Not every metric can carry that choice, and the constraint is measured, not
assumed.** On a real three-model run:

| Metric | Model A | Model B | Bayesian | Cross-family comparable |
|---|---|---|---|---|
| `joint_predictive_nll` | not produced | 3.546663 | 3.463581 | **No** — Model A declares `marginal` scope and is excluded by the capability gate |
| `predictive_nll` (per target) | 1.242692 | 0.886666 | 0.865895 | **No** — Model A's keys are marginal, the others conditional on preceding cohorts |
| `composition_log_loss` | 0.970440 | 0.958103 | 0.955918 | **Yes** |
| `rmse` (cohort) | 1.153610 | 1.364436 | 1.230392 | **Yes** |

Model A wins on point accuracy while losing on every likelihood-shaped metric,
so the declared rule determines the winner. Declare it before looking at it.

The rule Gate 8 implements:

1. Rank `IndependentTotalProbabilityModel` against
   `BayesianConditionalModel` on `joint_predictive_nll`. These two score the
   same object: their per-key decomposition was verified against independent
   scipy oracles to 1.8e-15 and is pinned by
   `test_bayesian_and_independent_keys_are_the_same_conditional_object`.
2. Admit `DirectCohortModel` only on metrics that are the same quantity for all
   three — composition log loss and cohort RMSE/MAE.
3. **Never rank `predictive_nll` across families.** Model A's per-cohort keys
   are marginal log masses; Model B's and the Bayesian model's are conditional,
   so their last cohort key is ~0 by construction. A per-target table hands the
   conditional models that key as free winnings.
4. Record the rule, its inputs, and the resulting choice in the freeze artifact
   before opening the lockbox.

Report the non-selected families' test results as pre-declared comparator runs
per section 11, without feeding them back into selection.

`SelectionFreeze.to_dict()["test_metrics"]` is the reserved slot for Gate 8's
results; it is confirmed `None` through a full real-model Gate 6 run and
nothing else writes it.

Gate 8's refit on all training data is also the first time the frozen
specifications run at the **full** Bayesian profile, whose diagnostic policy
action is `error` rather than `warn`. A configuration that only ever passed
under the reduced profile can therefore fail here for the first time.

**Loadable model artifacts belong to Gate 8.** Gate 7 logs cross-validation
fold models only as plain JSON state-bundle artifacts, with no MLflow
LoggedModel, pyfunc wrapper, or model flavor: they are evaluation evidence.
The full-training refit is the model worth loading. Wrap it as a
models-from-code pyfunc whose `load_context` calls `from_state_bundle`, never
cloudpickle, and supply `train_df` where the frequentist models' predictive
draws need it.

**Implementation record.** Gate 8 ran as one session, split into five phases
(canonical registry and manifest persistence; cross-family freeze; full
refit and lockbox evaluation; final MLflow tracking and models-from-code;
the thin marimo client), each stopped for approval with an independent
review subagent. `src/age_group_prediction/experiment/candidate_registry.py`,
`final_selection.py` and `final_evaluation.py`, and
`src/age_group_prediction/tracking/final.py`, `pyfunc_model.py` and the
additions to `runs.py` are new; `experiment/evidence.py`'s `SelectionFreeze`
gained the additive `cross_family_selection`/`test_metrics` fields and
`with_cross_family_selection`/`with_test_metrics` transitions Gate 6
continues to leave `None`. `experiment/` and `models/` still import no
MLflow (`tests/unit/test_tracking.py::test_experiment_and_model_code_never_import_mlflow`
still passes over every new file).

- **Phase 4 (final tracking and models-from-code) review** found two issues,
  both fixed: `tracking/final.py::_log_final_model`'s reload-equality check
  originally compared a pyfunc-loaded prediction against
  `evaluation.predictions_df`, which was computed with a different,
  purpose-scoped seed than the pyfunc's own (seedless) `predict` call —
  harmless only because the project's split strategy never hands the
  Bayesian model an unseen-neighborhood holdout row, the one case where a
  point mean itself draws randomness. It now compares two predictions
  computed the same way (no explicit seed on either side, mirroring
  `experiment.artifacts.capture_fold_artifact`'s identically-seeded
  comparison), via a shared `pyfunc_model.prediction_to_frame` helper so the
  two code paths cannot drift apart. Second, the in-process reload/equality
  check does not itself prove `code_paths` makes the model loadable from a
  fresh interpreter (the already-imported package is reused from
  `sys.modules`); this is now documented explicitly in the function's
  docstring rather than left implied.
- **Phase 5 (thin marimo client) review** found one real defect: the
  pre-split modeling-table preview cell displayed `.head(5)` of the full,
  unpartitioned table, which still carries target columns, before the
  outer split (and therefore the lockbox concept) existed — a ~67% chance,
  at this project's ~20% holdout fraction, of showing a real target value
  for a building the very next cell assigns to the holdout partition, on a
  plain setup run with no button clicked. Fixed by withholding target
  columns from that preview, with a new characterization test
  (`test_notebook_never_previews_the_full_modeling_table_with_targets`)
  guarding the specific class of leak the existing `split.test_df`-only
  check could not catch.
- A pre-existing environment defect was found and worked around, not
  introduced by Gate 8: the editable install of `age_group_prediction`/
  `student_simulator` does not resolve for a plain `python <script>.py`
  invocation in this environment (reproduced identically against the
  untouched `notebooks/01_eda.py`); `pytest` only works because of
  `pyproject.toml`'s `pythonpath = ["src"]`. Script-mode verification
  (`uv run notebooks/02_model_fitting.py`, and the notebook's own
  characterization test) sets `PYTHONPATH=src` explicitly.
- **Canonical run (authorized and complete).** The persisted lockbox
  manifest (`artifacts/lockbox/split_manifest.json`, 1529 rows / 150
  neighborhoods / 1222 train / 307 holdout) was replayed once. Cross-family
  selection chose `bayesian-reduced` (`BayesianConditionalModel`) on
  `composition_log_loss`, no tie-break needed; `direct-poisson` and
  `independent-nb2` are logged as predeclared comparators. All three
  full-training refits completed, the Bayesian refit's strict
  (`action="error"`) full-profile diagnostics passed, all three models
  predicted the same 307 ordered holdout IDs, and all three reloaded as
  models-from-code pyfuncs reproducing their point predictions exactly. The
  final MLflow parent (`run_role=final_evaluation`, linked by
  `source_cv_run_id` to the Gate 6/7 comparison parent) and every child
  finished `FINISHED`/`evidence_complete=true`; verified directly against
  the MLflow store, not only from script output.
- Suite: 654 passing (baseline 624 plus 20 fast `test_gate8_tracking.py`
  tests, 3 slow real-model tracking tests, 6 fast and 1 slow notebook
  characterization tests), ruff clean on every touched `src/`/`tests/` file,
  one documented Bayesian recovery warning unchanged.
- **Independent validation (2026-09-14, Opus 5): accepted with conditions.**
  The full record is the Gate 8 "Independent validation" subsection of
  `GATE_VALIDATION_FINDINGS.md`. The canonical run was executed by a scratch
  script mirroring the notebook's cells, not by the notebook's buttons; the
  script is reproduced in the Gate 8 "Canonical run record" section of
  `GATE_VALIDATION_FINDINGS.md`.
  Approved remediation:
  - the pretest freeze is bound to the rule and to its CV result;
  - once the lockbox has opened, a later run is accepted only as a retry of
    the identical decision;
  - the evaluator checks scored IDs against the manifest's holdout;
  - LightGBM is imported before torch, so logged models load in a fresh
    process;
  - final children are tagged `test_lock_status=opened`;
  - the test gaps found by sabotage are closed, with each new test proven to
    fail on its defect;
  - the notebook setup test no longer touches the canonical manifest.

  A second pass fixed F7: the full-profile check now enforces the recorded
  draw counts and thresholds, not just the profile label. It also fixed F9:
  leakage is checked against the schema the evaluation used. F6, F8 and F10
  are accepted as deferred.

  The independent review of this remediation
  (`GATE_8_REMEDIATION_REVIEW_HANDOFF.md`) reproduced every power proof. It
  found that a retry after an opened attempt could still change the refit
  models or seeds (R1), and that several guarantees had no test (R2–R5). Its
  fixes, applied with approval:
  - a retry must also match a `final_attempt_fingerprint`, computed after the
    refit and before the run opens, over the pretest freeze, each refit's
    seeds, training hashes and constructor configuration
    (`BaseAgeGroupModel.configuration_record`), prediction and evaluation
    settings, and the schema;
  - the refit requires the freeze's master seed to equal the CV provenance
    seed;
  - new tests cover deleted opened attempts, mismatched selected tags,
    failures before opening, every freeze field, and the draw-count boundary.

  The details are in the "Remediation review" subsections of
  `GATE_VALIDATION_FINDINGS.md`. The user accepted that remediation, and
  **Gate 8 is complete** (2026-09-14; suite 705 passing).

  Accepted deferrals:
  - F6, F8 and F10;
  - the retry guard's single-experiment scope;
  - the public lockbox functions without the F2 check;
  - unhashed model source code.

  The canonical evidence predates both remediation passes and was verified
  independently, not re-run.

### Gate 9: Remove Old Modeling Code

**Implementation: Codex**
**Verification: GPT-5.6 Sol**
**Architecture review: GPT-5.6 Terra**

Delete superseded modules, tests, exports, and notebook cells from Section 3.2,
then update active documentation.

Acceptance: no stale Gate 3, old benchmark, NumPyro, or child-expansion API
references remain; simulator tests stay green; optional dependency matrices
pass; package build and clean-environment import checks pass.

**`predictive.py` is not legacy and must not be removed.** It is absent from
section 3.2's list above, which is correct: all three model families import
`central_prediction_intervals` from it and it has its own tests. An earlier
revision of the Gate 3–6 validation brief listed it for deletion; that entry is
corrected. Section 3.2's list is the authority.

Gate 9 must also leave `src/student_simulator/` untouched. It is known not to
be lint-clean, and a tree-wide `ruff --fix` reformats roughly twelve unrelated
simulator files.

**Implementation record (2026-09-15): complete, pending user acceptance of the
Phase 4 report.** Phases 0-3 removed exactly the four superseded modules, the
ten-test `test_baselines.py`, the two narrow metric helpers, and the ten package
root exports; added the modeling and MLflow guides; and updated active workflow
documentation. Phase 4 established the following acceptance evidence:

- the authoritative command passed **695 tests with 13 warnings in 736.28
  seconds** (743.83 seconds wall time), exactly accounting for the ten
  intentionally removed tests;
- frozen dependency synchronization succeeded, and an external build produced
  both the wheel and source distribution in 1.48 seconds;
- archive inspection found both current packages and none of the deleted
  modules or obsolete tests;
- a fresh core-only environment loaded the installed wheel, not the source
  tree, and proved the four modules and ten exports unavailable; its bounded
  slice passed 479 tests except for seven repository-relative configuration
  path cases, after which all 36 tests in the owning file passed from the
  documented project working directory while still importing the wheel;
- without MLflow, two tracking modules skipped through `importorskip` while 11
  adjacent tests passed; with MLflow installed, 68 focused tracking tests
  passed against a unique scratch SQLite store and artifact root, with six
  expected integer-schema warnings;
- both notebooks passed `marimo check`; notebook characterization was included
  in the authoritative suite; active relative links and stale references
  passed after every residual was classified; and
- before any command, all protected evidence matched the Phase 0 baseline:
  `mlflow.db` SHA-256 `5b1d0f34f7d6c9be68ecf59deeb803f9ebebffb00c3652a7998af4fe800bac09`,
  lockbox SHA-256 `1c6fe0be481c9cdb0675faabbbdd591a403dc89860eece84390fd284ff62f5d4`,
  336 artifact files, artifact-manifest digest
  `e379f50379755c730c67472dc35b9001d969ddb978a9451d64dba005004f0cbc`,
  and combined 338-file digest
  `553cc8c5bfc9804ca29e369cdcb628c76c25f78d2afa35aaa5be478fbef2e52c`.

No modeling behavior changed in Phase 4. The canonical final evaluation was
not called, the canonical split was not replayed or persisted, and no client
was initialized against the canonical MLflow store.

## 14. Verification Matrix

| Area | Required verification | Gate 9 acceptance evidence | Verdict |
|---|---|---|---|
| Data/configuration | Typed defaults and overrides, schema, accounting identity, feature exclusion, hash stability | `test_modeling_config.py` and `test_modeling_data.py`, included in the 695-test authoritative suite | Pass |
| Data splitting | Default/caller RNG behavior, row-order invariance, disjoint IDs, manifest replay, no test-in-CV | `test_modeling_data.py` and `test_resampling.py`; manifest replay remained test-only and never used the canonical manifest | Pass |
| Features | Gate 2A passes before models; fit-fold-only state; stable names/order; schema-driven references; interaction hierarchy; component restrictions; no redundant room representation; zero-total filtering after preprocessing fit | `test_feature_engineering.py` and model predictor-matrix tests | Pass |
| Shared API | Unfitted errors, result shapes, finite means, normalized probabilities, serializable metadata | `test_model_contracts.py` and all three model suites | Pass |
| Model A | Deterministic pipeline, bounded tuning, bootstrap reproducibility | `test_direct_cohort.py`, `test_tuning.py`, and `test_resampling.py` | Pass |
| Model B | NB2 moments, grouped/expanded equivalence, zero totals, exact expected reconciliation | `test_independent_total_probability.py`, `test_distributions.py`, and `test_composition_kernels.py` | Pass |
| Bayesian NB2 + Dirichlet-Multinomial | Prior checks, Pyro parameterization, MCMC smoke, diagnostics, exact draw reconciliation | `test_bayesian_conditional.py` and the real NUTS checks in `test_bayesian_recovery.py` | Pass |
| Metrics | Class/protocol conformance, point/NLL/composition/coverage/WIS/PIT golden tests, posterior transformations, RNG and bootstrap reproducibility | `test_metrics.py` and `test_evaluation.py` | Pass |
| Experiments | Identical folds, no lockbox access, deterministic result schemas | `test_experiment.py`, `test_final_selection.py`, and experiment artifact tests | Pass |
| MLflow | Comparison parent, one sibling run per loss family, required metadata/artifacts, failure tags, artifact reload | Authoritative tracking tests plus 68 wheel-only focused tests against an isolated scratch store | Pass |
| Notebook | `marimo check`, clean execution, explicit expensive/final actions | Both notebooks passed `marimo check`; `test_model_fitting_notebook.py` passed in the authoritative suite; static inspection confirmed both explicit controls | Pass |
| Migration | Full suite, package build/import, stale-reference search | 695-test suite; external wheel/sdist build; clean core/tracking wheel environments; deleted-API assertions; classified stale-reference and active-link audits | Pass |

Use focused tests after each gate, then the full non-slow suite. Mark full
Bayesian inference and tracked end-to-end experiments separately so ordinary
simulator/unit testing remains fast.

## 15. Completion Definition

The rebuild is complete when:

1. all models satisfy the shared API and metric registry;
2. Model B estimates building-conditioned child age-group probabilities without
   literal child expansion;
3. The Bayesian NB2 + Dirichlet-Multinomial model uses Pyro and produces exactly reconciled posterior predictions;
4. selection uses only train/CV evidence;
5. final test evaluation runs once from a persisted lockbox;
6. MLflow records reproducible family-specific candidate runs with train/CV
   selection evidence, a comparison parent where useful, and reconstructable
   artifacts;
7. the notebook is only a client of tested package code; and
8. obsolete modeling code is removed without changing simulator behavior.

### Gate 9 Completion Evidence

| # | Completion requirement | Evidence and qualification | Verdict |
|---:|---|---|---|
| 1 | All models satisfy the shared API and metric registry | Shared contract, result, metric, and all three model test suites passed in the authoritative run and installed-wheel slice. | Satisfied |
| 2 | Model B estimates building-conditioned probabilities without literal child expansion | The implementation uses grouped weighted multinomial rows; `test_grouped_and_literal_expansion_are_equivalent` retains literal expansion only as an oracle. | Satisfied |
| 3 | The Bayesian model uses Pyro and exactly reconciles posterior predictions | Pyro model ownership, parameterization, prior checks, diagnostics, MCMC recovery, and draw-by-draw reconciliation are covered by the Bayesian unit and validation suites. | Satisfied |
| 4 | Selection uses only train/CV evidence | Experiment partition, identical-fold, selection-freeze, and final-selection tests passed; no Phase 4 command accessed holdout values. | Satisfied |
| 5 | Final test evaluation runs once from a persisted lockbox | The accepted Gate 8 record and byte-identical canonical evidence prove the completed run. Phase 4 did not replay the manifest or call any final-evaluation API. | Satisfied, historical evidence only |
| 6 | MLflow records reproducible candidate/comparison evidence and reconstructable artifacts | Tracking and real-model tests passed in the authoritative suite; 68 focused tests passed from the wheel against a unique scratch SQLite store; archive reload and pyfunc contracts remain covered. | Satisfied |
| 7 | The notebook is only a client of tested package code | Marimo checks, characterization tests, and static control inspection passed; expensive and final actions remain explicit. | Satisfied |
| 8 | Obsolete code is removed without changing simulator behavior | The suite moved from 705 to exactly 695 by deleting ten approved tests; deleted modules/exports are absent from source and wheel; simulator tests passed; `src/student_simulator/` was untouched. | Satisfied |

All eight clauses are satisfied, the Phase 4 independent review returned
**ACCEPT**, and protected evidence remained byte-identical. The user approved
Gate 9 and the modeling rebuild on 2026-09-15.

**Post-completion independent validation (2026-09-15).** A separate Opus 5 audit
re-derived all eight clauses and returned **ACCEPT**. It remediated three
Medium findings — notebook-setup replay wording, the direct-cohort policy
description, and a missing `PredictionResult` reconciliation regression case —
plus twelve Low documentation findings, without changing modeling behavior.
The suite baseline is now **696 passing**. See
[GATE_9_INDEPENDENT_VALIDATION_REPORT.md](GATE_9_INDEPENDENT_VALIDATION_REPORT.md).