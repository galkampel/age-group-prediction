# Evaluation And Metrics

This document describes how fixed predictions are scored: the shared
`PredictionResult` contract, the metric protocol and capabilities, every
metric class with its formula, the metric sets used by the canonical
candidates, the likelihood-comparability rules, `evaluate_predictions`, and
the neighborhood-cluster bootstrap. The evaluation layer never fits a model.
Everything here is derived from `src/age_group_prediction/results.py`,
`metrics.py`, `evaluation.py`, `resampling.py`, `distributions.py`,
`modeling_config.py`, and, for selection use, `experiment/candidate_registry.py`,
`experiment/policies.py`, `experiment/selection.py`,
`experiment/final_selection.py`, and `tracking/evidence.py`.

## 1. The Prediction Contract

Every model's `predict` returns an immutable `PredictionResult`. Arrays are
copied and made read-only.

| Field | Shape / type | Required |
|---|---|---|
| `building_ids` | $(n,)$, unique | yes |
| `cohort_names` | non-empty unique tuple | yes |
| `total_mean` | $(n,)$ | yes |
| `cohort_means` | $(n, K)$ | yes |
| `age_group_probabilities` | $(n, K)$ | yes |
| `reconciliation_error` | $(n,)$ | yes |
| `predictive_draws` | mapping target → array | optional |
| `prediction_intervals` | mapping target → $(n, L, 2)$ | optional |
| `interval_levels` | tuple of $L$ levels | default `()` |
| `pointwise_log_probabilities` | mapping target → $(n,)$ or $(n, S)$ | optional |
| `parametric_distributions` | mapping target → `ParametricDistributionSpec` | optional |
| `validation_config` | `PredictionValidationConfig` | default |
| `pointwise_log_probability_scope` | `marginal`, `sequential_joint`, or `None` | optional |

Construction raises `ValueError` unless: the four core arrays are finite and
nonnegative; probabilities are at most 1 and each row sums to 1 within
`probability_sum_tolerance`; `reconciliation_error` equals
$\lvert\sum_k \text{cohort mean} - \text{total mean}\rvert$ within
`reported_error_tolerance` and that error is at most
`reconciliation_tolerance`; optional mapping keys are `"total"` or a cohort
name and have one finite row per building; interval levels are unique, lie in
$(0,1)$, match the interval array, and every lower bound is at most its upper
bound. Mapping keys use `"total"`; metrics translate `n_children_total` to it.

**Non-finite log probabilities are refused.** A kernel that returns
$-\infty$ (for example a positive cohort count at zero predicted probability)
fails `PredictionResult` construction, so the model's `predict` raises rather
than produce an infinite score.

`ParametricDistributionSpec(family, dispersion=None, scale=None)` accepts
`poisson` (no dispersion or scale), `nb2` (dispersion required), and `normal`
(scale required); parameters broadcast to positive per-building vectors.
`PredictionResult.from_means(...)` derives probabilities and reconciliation
error from means; zero-total rows get uniform probabilities.

### Pointwise log-probability scopes

- **`marginal`** — each key scores its own target independently; keys must not
  be summed. Set by `DirectCohortModel`.
- **`sequential_joint`** — `total` is the marginal total score and each cohort
  key is conditional on the total and the preceding cohorts, so the keys sum
  to the joint log mass. It requires keys for the total and every cohort with
  equal shapes. Set by `IndependentTotalProbabilityModel` and
  `BayesianConditionalModel`.

## 2. Metric Protocol, Results, And Capabilities

A `Metric` (runtime-checkable protocol) exposes `name`, `target`,
`required_capability`, `optimization_direction` (`minimize`, `maximize`, or
`target_zero`), `aggregation_level`, and
`compute(observed, prediction, *, rng=None) -> MetricResult`.

`MetricResult(metric_name, value, target, aggregation_level, sample_count,
metadata={})` requires a non-empty name, a finite value, a nonnegative sample
count, and JSON-safe metadata (deep-copied and frozen). `to_record()` returns
the five table columns without metadata.

Capabilities are string literals. `available_prediction_capabilities(prediction)`
always includes `means`, `probabilities`, and `reconciliation`; adds
`predictive_draws`, `prediction_intervals`, `pointwise_log_probabilities`, and
`parametric_distributions` when those payloads are present; and adds
`joint_pointwise_log_probabilities` only for the `sequential_joint` scope.
Payload capabilities are checked per target: the metric's target key must be
present in the mapping.

**Missing-capability policy.** `on_missing_capability="error"` (default)
raises `ValueError` naming the metric, target, and capability; `"skip"` omits
the metric and records it under `metadata["skipped_metric_definitions"]`. Any
other value is refused.

## 3. Metric Catalogue

Notation: $y_b$ observed, $\mu_b$ predicted mean, $n$ buildings. Point metrics
require observed values that are finite and nonnegative with matching row
count. Unless noted, `aggregation_level` is `building`.

### Point metrics (capability `means`)

| Class | `metric_name` | Direction | Value |
|---|---|---|---|
| `MeanAbsoluteError(target)` | `mae` | minimize | $\frac1n\sum_b\lvert\mu_b-y_b\rvert$ |
| `RootMeanSquaredError(target)` | `rmse` | minimize | $\sqrt{\frac1n\sum_b(\mu_b-y_b)^2}$ |
| `MeanBias(target)` | `mean_bias` | target zero | $\frac1n\sum_b(\mu_b-y_b)$ (predicted minus observed) |
| `R2(target)` | `r2` | maximize | scikit-learn `r2_score` |
| `MeanPoissonDeviance(target, minimum_mean)` | `mean_poisson_deviance` | minimize | scikit-learn `mean_poisson_deviance` with $\mu$ clipped below at `minimum_mean` |

`target` is a cohort column or `n_children_total`.

### Likelihood metrics (`metric_name` `predictive_nll`, minimize)

`PredictiveNegativeLogLikelihood(target, interpretation)` is an abstract base;
instantiate a variant.

- **`ParametricPredictiveNegativeLogLikelihood`** — capability
  `parametric_distributions`. Value $-\frac1n\sum_b \log f(y_b)$ using
  `distributions.pointwise_log_probability`: Poisson log mass, NB2 log mass
  with size $1/\alpha$ and $p = \text{size}/(\text{size}+\mu)$, or Normal log
  **density**. Poisson and NB2 require integer observations; an unknown family
  raises.
- **`PointwisePredictiveNegativeLogLikelihood`** — capability
  `pointwise_log_probabilities`. For $(n,)$ input the value is
  $-\frac1n\sum_b \ell_b$. For $(n,S)$ posterior draws it is the
  posterior-integrated score
  $-\frac1n\sum_b\bigl[\operatorname{logsumexp}_s \ell_{bs} - \log S\bigr]$.
- **`JointPredictiveNegativeLogLikelihood(interpretation)`** — `metric_name`
  `joint_predictive_nll`, target `joint`, capability
  `joint_pointwise_log_probabilities`. It requires the `sequential_joint`
  scope, sums the keys `("total", *cohort_names)` per building (per draw,
  before the log-mean-exp), and applies the same reduction. Metadata records
  the scope and summed keys.

### Composition metrics (target `composition`, capability `probabilities`, aggregation `child`)

With cohort counts $c_{bk}$, building totals $N_b=\sum_k c_{bk}$, and
probabilities $p_{bk}$:

$$
\text{composition\_log\_loss} = -\frac{\sum_{b,k:\,c_{bk}>0} c_{bk}\log p_{bk}}{\sum_b N_b},
\qquad
\text{composition\_brier} = \frac{\sum_{b,k}\bigl[c_{bk}(1-p_{bk})^2 + (N_b-c_{bk})\,p_{bk}^2\bigr]}{\sum_b N_b}.
$$

Classes `CompositionLogLoss` and `CompositionBrierScore`. Log loss raises when
there are no children or a positive count has $p \le 0$. Brier is per child on
a $[0,2]$ scale. Both report `sample_count` as the number of **buildings**.
The kernels `composition_log_loss`, `composition_log_loss_from_log_probabilities`
(used by Model B's calibration), and `composition_brier_score` are module
functions.

### Accounting metric (capability `reconciliation`, target `all`)

`ReconciliationError(statistic, tolerance=0.0)` reads
`prediction.reconciliation_error` and ignores observations. `statistic` is
`mean` (`mean_reconciliation_error`), `max` (`max_reconciliation_error`), or
`count_above_tolerance` (`reconciliation_count_above_tolerance`, the count
with error strictly greater than `tolerance`).

### Interval metrics (capability `prediction_intervals`)

Each takes `(target, level)`; the level must match an entry of
`interval_levels` within $10^{-12}$. With bounds $[l_b, u_b]$ and
$\alpha = 1-\text{level}$:

| Class | `metric_name` | Direction | Value |
|---|---|---|---|
| `IntervalCoverage` | `interval_coverage` | maximize | $\frac1n\sum_b 1\{l_b\le y_b\le u_b\}$ |
| `MeanIntervalWidth` | `mean_interval_width` | minimize | $\frac1n\sum_b(u_b-l_b)$ |
| `WeightedIntervalScore` | `weighted_interval_score` | minimize | $\frac1n\sum_b \frac{\alpha}{2}\,\mathrm{IS}_b$ |

$$
\mathrm{IS}_b = (u_b-l_b) + \tfrac{2}{\alpha}(l_b-y_b)\,1\{y_b<l_b\} + \tfrac{2}{\alpha}(y_b-u_b)\,1\{y_b>u_b\}.
$$

This is a **single-interval** score (equal to the two pinball losses at
$\alpha/2$ and $1-\alpha/2$); there is no median term or multi-level average.
The level is recorded only in metadata (see Pitfalls).

### Calibration metrics (`metric_name` `randomized_pit_ks`, minimize)

`RandomizedPIT(target, default_seed, histogram_bins)` is an abstract base.
With $U_b \sim \operatorname{Uniform}(0,1)$ drawn from the caller's `rng` or
`default_rng(default_seed)`:

- **`ParametricRandomizedPIT`** (capability `parametric_distributions`):
  Poisson/NB2 use $F(y_b-1) + U_b\,[F(y_b)-F(y_b-1)]$; Normal uses $F(y_b)$
  without randomization.
- **`DrawsRandomizedPIT`** (capability `predictive_draws`, 2-D draws):
  $\bigl(\#\{s: d_{bs}<y_b\} + U_b\,\#\{s: d_{bs}=y_b\}\bigr)/S$.

PIT values are clipped to $[0,1]$. The **reported value is the
Kolmogorov–Smirnov statistic** against the uniform distribution; the p-value,
PIT values, and histogram (`histogram_bins` bins) are metadata.

## 4. Default Metric Sets And The Canonical Candidates

There is no public name-to-class registry. `default_metric_set` builds a
metric tuple:

```text
default_metric_set(
    targets, *, nll_source, nll_interpretation, evaluation_config,
    prediction_validation_config, interval_levels=(), pit_source=None,
    pit_default_seed=None, joint_nll_interpretation=None,
) -> tuple[Metric, ...]
```

Order: for each target, MAE, RMSE, mean bias, R², Poisson deviance
(`minimum_mean = poisson_minimum_mean`), and the `nll_source` NLL variant;
then joint NLL if `joint_nll_interpretation` is given; composition log loss
and Brier; the three reconciliation statistics with
`reconciliation_tolerance`; then, per target, coverage, width, and WIS for
each level followed by that target's PIT if `pit_source` is given (a seed is
required). Targets and
levels must be unique.

| Candidate | Targets | NLL | Joint NLL |
|---|---|---|---|
| `direct-poisson` | three cohorts (no total) | parametric | no |
| `independent-nb2` | total and three cohorts | pointwise | yes |
| `bayesian-reduced` | total and three cohorts | pointwise | yes |

No canonical candidate requests interval or PIT metrics.

## 5. Likelihood Comparability

Three rules keep incomparable likelihoods apart:

1. **Marginal vs joint.** `joint_predictive_nll` requires the
   `joint_pointwise_log_probabilities` capability, which only the
   `sequential_joint` scope provides. Per-target `predictive_nll` is marginal
   for Model A and conditional for Models B and C, so it is **not** one
   quantity across approaches.
2. **Log density vs log mass.** Within an approach, selection refuses a policy
   whose criteria use a distributional capability when eligible candidates
   declare both discrete (`poisson`, `nb2`) and continuous (`normal`)
   parametric families; the continuous candidate must be a
   `diagnostic_comparator`. Measure kinds are read only from
   `parametric_distributions`, so a candidate scored purely through pointwise
   log probabilities is not classified.
3. **Across approaches.** `select_cross_family_winner` compares
   `joint_predictive_nll` only between the two conditional approaches, then
   compares that winner with Model A on `composition_log_loss`, mean cohort
   RMSE, and mean cohort MAE. It never reads `predictive_nll`.

The tracking layer logs these rules as `metric_comparability.json`. Details:
[CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md).

## 6. Evaluating Fixed Predictions

```text
evaluate_predictions(observed, prediction, metrics, *, rng=None,
                     on_missing_capability="error") -> EvaluationResult
```

`EvaluationResult.metrics_df` has exactly the columns `metric_name`, `value`,
`aggregation_level`, `sample_count`, and `target`. `diagnostics` maps
`"{metric_name}:{target}"` to the metadata of each result that has any. `metadata` records
`metric_count`, `metric_definitions`, `missing_capability_policy`, and
`skipped_metric_definitions`. `BaseAgeGroupModel.evaluate(eval_df, *, metrics,
rng=None)` is `predict` followed by `evaluate_predictions`.

## 7. Neighborhood-Cluster Bootstrap

```text
neighborhood_cluster_bootstrap(observed, prediction, metrics, *, config,
    default_seed, rng=None, on_missing_capability="error") -> BootstrapEvaluationResult
```

- **Fixed predictions.** No model is refitted. Each replicate subsets the
  observed rows and every prediction payload (including per-building
  dispersion and scale) to the resampled rows, keeping the scope and levels.
- **Resampling.** `NeighborhoodClusterResampler.from_frame(frame, column)`
  requires at least two neighborhoods; each replicate draws $G$ neighborhoods
  with replacement from the $G$ present and concatenates all their rows.
- **RNG.** `rng` if given, otherwise `default_rng(default_seed)`. The same
  generator drives point evaluation, resampling, and metric randomness
  (PIT). `metadata["default_seed"]` is `None` when `rng` was passed.
- **Failures.** A `ValueError` or `FloatingPointError` in a replicate drops
  the whole replicate and increments `failed_replicates`; other exceptions
  propagate. If `failed / replicates > max_failed_fraction`, it raises
  `RuntimeError`.
- **Intervals.** For each point metric, replicate values matched on
  `metric_name`, `target`, and `aggregation_level` give
  `np.quantile(values, [α/2, 1-α/2])` with $\alpha = 1 -$ `confidence_level`.
  Metrics with no successful replicate are omitted.

`BootstrapEvaluationResult` fields: `point_evaluation`, `intervals_df`
(`metric_name`, `target`, `aggregation_level`, `value`, `confidence_lower`,
`confidence_upper`, `successful_replicates`), `replicate_metrics_df`
(`replicate` plus the five metric columns), `failed_replicates`, and
`metadata` (`bootstrap_unit="neighborhood"`, `replicate_count`,
`confidence_level`, `interval_method`, `failed_replicates`, `default_seed`,
`neighborhood_id_column`). How cross-validation combines fold bootstraps is
described in [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md#5-aggregation-bootstrap-evidence-and-fold-coverage).

## 8. Configuration

`[evaluation]` has **no dataclass defaults**: every key must be present.
Missing or unknown keys fail when `EvaluationConfig` is constructed.

| TOML section / field | Value | Effect |
|---|---|---|
| `[evaluation] bootstrap_replicates` | 200 | Replicates; at least 1 |
| `confidence_level` | 0.95 | Percentile interval level; in $(0,1)$ |
| `neighborhood_id_column` | `neighborhood_id` | Cluster column |
| `interval_method` | `percentile` | Only allowed value; recorded in metadata |
| `max_failed_fraction` | 0.5 | Failure threshold; in $[0,1)$ |
| `poisson_minimum_mean` | $10^{-8}$ | Deviance clip, applied through `default_metric_set` |
| `pit_histogram_bins` | 10 | PIT histogram bins through `default_metric_set`; at least 2 |
| `[prediction_validation] probability_sum_tolerance` | $10^{-12}$ | Row-sum tolerance |
| `reported_error_tolerance` | $10^{-12}$ | Reported vs actual reconciliation error |
| `reconciliation_tolerance` | $10^{-9}$ | Maximum reconciliation error; also the reconciliation metric tolerance |
| `zero_total_probability_policy` | `uniform` | Only allowed value; used by `from_means` |

The `[prediction_validation]` tolerances must be finite and positive and have
the listed values as dataclass defaults. The PIT seed is an argument
(`pit_default_seed`), not a configuration key.

## 9. Pitfalls

- **Do not rank `predictive_nll` across approaches.** Use
  `joint_predictive_nll` between conditional approaches only.
- **Composition metrics are per child, point metrics per building.** Their
  `sample_count` is buildings in both cases.
- **Interval levels are not part of a metric's key.** Two interval metrics
  for one target at different levels would share `(metric_name, target,
  aggregation_level)` and collide in `metrics_df`, bootstrap pooling, fold
  aggregation, and diagnostics. `CandidateDefinition` already refuses such
  duplicate keys, so only direct `evaluate_predictions` or bootstrap callers
  can hit this; use one level per target and metric.
- **`MeanIntervalWidth` does not validate its level** at construction; a bad
  level fails only when matched against the prediction.
- **Bootstrap intervals are conditional on the fitted model.** They reflect
  evaluation-set sampling variability, not refit uncertainty.
- **`max_failed_fraction` here (0.5) is not Model B's
  `bootstrap_max_failed_fraction` (0.25)**, which governs predictive refits.

## 10. Tests And Related Documents

- Tests: `tests/unit/test_metrics.py` (formulas, WIS oracle, PIT, capability
  sets, `default_metric_set`), `tests/unit/test_evaluation.py` (capability
  policy, bootstrap reproducibility, whole-cluster resampling, scope
  preservation), `tests/unit/test_composition_kernels.py`,
  `tests/unit/test_distributions.py`, `tests/unit/test_resampling.py`, and
  `tests/unit/test_final_selection.py` (no cross-approach `predictive_nll`).
- [MODELING_GUIDE.md](MODELING_GUIDE.md) Sections 6–7,
  [DIRECT_COHORT_MODEL.md](DIRECT_COHORT_MODEL.md),
  [INDEPENDENT_TOTAL_PROBABILITY_MODEL.md](INDEPENDENT_TOTAL_PROBABILITY_MODEL.md),
  [BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md](BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md),
  [CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md).
