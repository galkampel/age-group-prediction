# TODO: Known Code Issues

Issues found while writing the component documents (2026-09-15). None affects
the canonical results: the canonical candidates declare no interval metrics,
and the rest are comments, docstrings, or test gaps, plus one deferred analysis
item at the end. Each item names the code to change, the fix, and how to
verify it. Fixing any `src/` item requires the normal gated workflow (plan,
independent review, full test suite) and must not touch `mlflow.db`,
`mlartifacts/`, or `artifacts/lockbox/`.

## Correctness

- [ ] **Interval level is not part of a metric's identity.**
  `IntervalCoverage`, `MeanIntervalWidth`, and `WeightedIntervalScore` record
  `interval_level` only in metadata (`src/age_group_prediction/metrics.py`).
  Two levels for one target share `(metric_name, target, aggregation_level)`,
  so they collide in `EvaluationResult.metrics_df`, bootstrap replicate
  pooling (`evaluation.py`, `_percentile_intervals`), fold aggregation
  (`experiment/aggregation.py`, `_aggregate_fold_metrics`), and the
  `"{metric_name}:{target}"` diagnostics key. `CandidateDefinition` already
  refuses duplicate keys, so only direct `evaluate_predictions` and bootstrap
  callers are exposed.
  *Fix:* encode the level in the metric identity (for example
  `interval_coverage_0.9`, or a level-qualified `aggregation_level`), or have
  `evaluate_predictions` refuse duplicate keys. *Verify:* a unit test that
  evaluates two levels for one target and checks distinct rows and intervals.
- [ ] **`MeanIntervalWidth` does not validate its level.** Unlike
  `IntervalCoverage` and `WeightedIntervalScore`, it has no `__post_init__`
  check that `level` lies in $(0, 1)$. *Fix:* add the same check. *Verify:*
  extend the metric construction tests in `tests/unit/test_metrics.py`.

## Robustness

- [ ] **Bayesian final-refit guard errors are unwrapped.**
  `_require_full_bayesian_policy` runs outside `_final_failure_context` in
  `experiment/final_evaluation.py`, so its errors lack the
  `Candidate '{id}' failed during final ...` context, and a missing
  diagnostics block raises `TypeError` instead of `ValueError`.
  *Fix:* run the guard inside a failure context and refuse missing
  diagnostics with `ValueError`. *Verify:* extend the guard tests in
  `tests/unit/test_final_evaluation.py`.

## Misleading comments and docstrings

- [ ] **"Repeated overlapping splits" comments.** `experiment/partitions.py`
  (`_fold_coverage` docstring) and `experiment/aggregation.py` (the
  cross-validation bootstrap comment) describe the validation folds as
  repeated overlapping splits. `make_validation_folds` validates every
  non-singleton building exactly once. *Fix:* reword both; restate the
  `assumes_independent_folds` caveat without the overlap premise.
- [ ] **Comparability record says approaches are never ranked.**
  `tracking/evidence.py` writes `"selection_scope": "one selection per
  approach; approaches are never ranked against each other"` into
  `metric_comparability.json`, but `experiment/final_selection.py` ranks
  approaches on composition log loss, RMSE, and MAE. *Fix:* describe the
  within-approach scope and reference the cross-family rule. This changes
  logged evidence text for future runs only; update
  `tests/unit/test_tracking.py` expectations if they assert the string.
- [ ] **sklearn comment in `metrics.py`.** The module comment says
  closed-form metrics avoid scikit-learn, but `R2` and `MeanPoissonDeviance`
  call it. *Fix:* reword the comment.
- [ ] **Pyfunc column order docstring.** The `tracking/pyfunc_model.py` module
  docstring reads as if all cohort means precede all probabilities;
  `prediction_to_frame` interleaves `{cohort}_mean` and
  `{cohort}_probability` per cohort. *Fix:* state the interleaved order.

## Test gaps

- [ ] **Empty validation fold.** No test covers the `ValueError` from
  `make_validation_folds` when a fold has no validation rows
  (`data_splitting.py`). *Verify:* a table with fewer non-singleton buildings
  than `n_folds`.
- [ ] **Manifest payload keys.** No test covers `SplitManifest.from_dict`
  refusing unknown or missing keys, or the `deployment_claim` default.
- [ ] **Hashing functions.** `table_hash` and `column_schema_hash`
  (`hashing.py`) are only tested indirectly. *Verify:* row-order invariance,
  and sensitivity to column order and dtype.

## Deferred analysis

- [ ] **Model-based 2D partial dependence and ALE.** The EDA notebook
  (`notebooks/01_eda.py`) shows 1D partial dependence and ICE, plus
  two-dimensional *empirical* interaction views. Model-based 2D partial
  dependence and ALE were deferred until a diagnostic pass establishes that
  they add information beyond the supported-data views. (Carried over from the
  removed EDA plan.)

## Related Documents

[DATA_AND_SPLITTING.md](DATA_AND_SPLITTING.md),
[EVALUATION_AND_METRICS.md](EVALUATION_AND_METRICS.md),
[CROSS_VALIDATION_AND_SELECTION.md](CROSS_VALIDATION_AND_SELECTION.md),
[FINAL_EVALUATION.md](FINAL_EVALUATION.md).
