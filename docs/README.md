# Simulator Documentation

This index defines document authority for the compact component simulator.

## Current Stage 1 Documents

| Document | Authority |
|---|---|
| [Simplified model specification](SIMPLIFIED_MODEL_PLAN.md) | Canonical behavior, formulas, tables, observability rules, and Section 9 complexity roadmap |
| [Compact simulator migration plan](COMPACT_SIMULATOR_MIGRATION_PLAN.md) | Completed implementation record, validation gates, deletion sequence, and model assignments |
| [Marimo research workflow migration plan](MARIMO_MIGRATION_PLAN.md) | Completed Jupyter-to-marimo cutover, package setup, reactive notebook structure, validation gates, and per-step model routing |
| [EDA and predictive modeling plan](EDA_AND_PREDICTIVE_MODELING_PLAN.md) | Completed EDA roadmap and modeling handoff record |
| [Modeling rebuild and experiment tracking plan](MODELING_REBUILD_PLAN.md) | Implementation and acceptance record for shared model contracts, three model families, evaluation, strict test isolation, MLflow, and Gate 9 cleanup |
| [Modeling guide](MODELING_GUIDE.md) | User guide to the simulator-to-model pipeline, public APIs, configuration, evidence, and invariants |
| [MLflow experiments guide](MLFLOW_EXPERIMENTS_GUIDE.md) | User guide to scratch tracking, run layout, metrics, search, final-run semantics, and model loading |
| [Direct cohort model](DIRECT_COHORT_MODEL.md) | Model A: per-cohort LightGBM families, tuning, marginal scoring, bootstrap uncertainty, persistence |
| [Independent total and probability model](INDEPENDENT_TOTAL_PROBABILITY_MODEL.md) | Model B: NB2/Poisson total with exposure offset, grouped multinomial composition, temperature calibration, joint scoring, bootstrap uncertainty |
| [Feature engineering](FEATURE_ENGINEERING.md) | Feature specs, validation rules, fitted transformations, output columns, per-model use, and candidate forms |
| [Data and splitting](DATA_AND_SPLITTING.md) | Modeling table and schema, known-neighborhood outer split, write-once split manifest and replay, training-only folds, partition checks, the canonical lockbox |
| [Evaluation and metrics](EVALUATION_AND_METRICS.md) | Prediction contract, metric protocol and capabilities, metric formulas, canonical metric sets, likelihood comparability, neighborhood-cluster bootstrap |
| [Cross-validation and selection](CROSS_VALIDATION_AND_SELECTION.md) | Candidate registry, cross-validation runner, fold artifacts and reload checks, permutation importance, within-approach selection, selection freeze, cross-family rule, seeds and provenance |
| [Known code issues (TODO)](TODO.md) | Open correctness, robustness, comment, and test-gap items found while documenting the components |
| [Final evaluation](FINAL_EVALUATION.md) | Pretest-freeze verification, guarded full-training refit, attempt fingerprint, one-time lockbox evaluation, tracked-run guards, final evidence, pyfunc serving contract |
| [Module reference](MODULE_REFERENCE.md) | One entry per source module: responsibility, main public API, internal dependencies, and where to read more |
| [Bayesian conditional model overview](BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md) | Model C in brief: statistical model, fitting and diagnostics, prediction, selection, persistence, configuration |
| [Bayesian NB2 + Dirichlet-Multinomial guide](BAYESIAN_CONDITIONAL_MODEL.md) | Model C: selection and comparison, persistence, configuration, metadata, and a technical explanation of the model's statistics, Pyro syntax, tensor shapes, estimator/component/inference ownership, prediction, diagnostics, and runtime workarounds |
| [Gate validation findings](GATE_VALIDATION_FINDINGS.md) | Authoritative record of the independent validation pass: per-gate verdicts, findings, withdrawn findings, and remediation |
| [Gate 9 independent validation report](GATE_9_INDEPENDENT_VALIDATION_REPORT.md) | Post-completion audit verdict, findings, remediation, command ledger, and protected-evidence comparison |

Completed session handoffs (Gates 3-9, the module split) and the superseded
stage/state implementation plan were moved on 2026-09-15 to a local,
git-ignored `docs/archive/`. Their outcomes are recorded in the findings
record, the rebuild plan, and the Gate 9 report.

If implementation details conflict, the model specification controls behavior
and the compact migration plan controls software boundaries and delivery order.

## Advanced Target References

These documents contain unique future-stage material. They are not Stage 1
implementation specifications.

1. [Problem definition](PROBLEM_DEFINITION.md) defines the eventual
   building-level target and temporal boundary.
2. [Data generation plan](DATA_GENERATION_PLAN.md) contains the full causal
   process with time, proxies, exposure, environment, and decomposition.
3. [Parameter reference](PARAMETER_REFERENCE.md) explains full-model
   parameters, units, and unresolved choices.
4. [Advanced implementation plan](IMPLEMENTATION_PLAN.md) proposes future
   provider, schema, pipeline, and validation architecture.

## Legacy Sources

Files under [`legacy/`](legacy/) preserve historical assumptions and
derivations. They may conflict with the current simplified model and must not
be used as implementation specifications:

- [Original data-generation guide](legacy/DATA_GENERATION_GUIDE.md)
- [Original English source specification](legacy/specification_source_en.md)

The superseded root-level Stage 1 proposal was removed. Its resolved behavior
is represented in the canonical simplified model specification.

## Current Status

- Simplified model and formulas: specified.
- Simplified implementation: compact generation behavior complete.
- Compact architecture migration: complete through the Step 13 final cleanup
   gate.
- Marimo research workflow migration: complete through Step 10;
   `research.ipynb` and `ipykernel` have been removed, and both marimo
   notebooks are active.
- Building-level EDA: complete in `notebooks/01_eda.py`; the accepted findings
   feed the modeling workflow in `notebooks/02_model_fitting.py`.
- Complexity roadmap: specified but deferred.
- Advanced target model: retained as future reference.
- Simplified canonical calibration and oracle recovery: implemented.
- Modeling rebuild: Gates 1-8 complete (Gates 1-6 and 8 accepted and
   independently validated). Gate 9 is complete: its Phase 4 acceptance
   matrix passed, its review returned **ACCEPT**, and the user approved it on
   2026-09-15. The post-completion independent validation returned
   **ACCEPT**; see [its report](GATE_9_INDEPENDENT_VALIDATION_REPORT.md),
   the Phase 4 record (archived: `GATE_9_PHASE_4_ACCEPTANCE_HANDOFF.md`)
   and [the rebuild plan](MODELING_REBUILD_PLAN.md). Gate 5 (Bayesian NB2 +
  Dirichlet-Multinomial) is accepted: three independent reviews (REJECT,
  ACCEPT WITH GAPS, and a final ACCEPT WITH GAPS with no blockers) have had
  every finding fixed and verified, and the full-profile acceptance run
   passed. Gate 6 now provides fixed-fold cross-model validation, deterministic
   within-approach selection freezes, visible likelihood-comparability and
   calibration evidence, and validation-only block permutation importance.
   It enforces the Bayesian model's use of the selected independent model's
   feature forms and wires the configured master seed into purpose-specific
   model and evaluation RNGs.
- Gate 7 (MLflow tracking): complete. Optional `tracking` group with a SQLite
  default; self-contained state bundles for all three models; reload-checked
  fold artifacts and run provenance captured by the runner without MLflow; and
  `age_group_prediction.tracking`, which logs a finished comparison as one
  parent run with one child run per candidate. Records: the plan's Gate 7
  implementation record and the Gate 7 section of
  [GATE_VALIDATION_FINDINGS.md](GATE_VALIDATION_FINDINGS.md). Suite: 567 passing, ruff clean, one documented Bayesian
  warning. A behavior-preserving follow-up is briefed in
  `MODULE_SPLIT_HANDOFF.md` (archived): Phase 1 (`tracking.py`
  split into the `tracking/` package) and Phase 2 (Model B's calibration math
  and fold losses moved to `models/probability_calibration.py` and
  `models/fold_scoring.py`) are complete.
- Gate 8 (cross-family selection and the final lockbox): **complete**
  (2026-09-14). Implemented, and independently accepted with conditions. The must-fix conditions
  and findings F7 and F9 are remediated; F6, F8 and F10 are accepted as
  deferred. The independent review of that remediation, briefed by
  `GATE_8_REMEDIATION_REVIEW_HANDOFF.md` (archived),
  found one High defect: a retry after an opened lockbox could re-tune or
  re-seed the refit. That defect and four test gaps are now fixed; a retry must
  also match a `final_attempt_fingerprint`. The user accepted that fix,
  completing Gate 8. See
  [MODELING_REBUILD_PLAN.md](MODELING_REBUILD_PLAN.md) and the Gate 8 section
  of [GATE_VALIDATION_FINDINGS.md](GATE_VALIDATION_FINDINGS.md).
  `experiment.select_cross_family_winner` picks one candidate across
  approaches from training/CV evidence only, before any holdout row is read;
  `experiment.final_evaluation` refits the three frozen winners on the full
  training partition (the Bayesian winner forced to the full NUTS profile)
  and evaluates them once on the persisted lockbox holdout;
  `tracking.run_final_evaluation` logs that as a new, linked MLflow parent
  run and each refit as a loadable, pickle-free models-from-code pyfunc;
  `notebooks/02_model_fitting.py` is the thin marimo client, gated behind
  explicit run buttons. The canonical run is authorized and complete: the
  Bayesian conditional model was selected, with `DirectCohortModel` and
  `IndependentTotalProbabilityModel` logged as predeclared comparators.
  Implementation delivered across five reviewed phases; two independent
  review findings (a fragile same-process reload proof and a pre-split
  target-column leak in the notebook's table preview) were fixed and
   regression-tested. Historical Gate 8 suite: 705 passing after the validation remediation and the remediation
  review (654 before), ruff clean on touched files
  (notebooks carry the same pre-existing marimo-required-boilerplate lint
   category `01_eda.py` already has), one documented Bayesian warning. Gate 9
   then removed ten obsolete tests (695), and its independent validation added
   one regression case; the current authoritative suite is 696 passing with 13
   documented warnings.
- Independent validation of the modeling rebuild (a separate pass from the
   implementation reviews above): **Gates 1, 2A, 2B, 2C, 3, 4, 5 and 6 validated
   and remediated**, and **Gate 8 validated, remediated and complete**. Gate 9
   Phase 4 was independently accepted and approved by the user on 2026-09-15;
   its post-completion independent validation returned **ACCEPT**.
   Gate 7 had per-phase independent reviews instead of a separate
  pass. Gate 8 had per-phase reviews, a separate independent validation, and a
  fresh-session review of its remediation. Verdicts, findings, withdrawn findings
  and remediation are recorded per gate in
  [GATE_VALIDATION_FINDINGS.md](GATE_VALIDATION_FINDINGS.md), whose header is
  the current status record. The protocol used for Gates 5 and 6 is kept in
  `GATE_5_6_VALIDATION_INSTRUCTION.md` (archived),
  with reviewer context in
   `GATE_3_6_VALIDATION_BRIEF.md` (archived). Suite baseline:
   696 passing after the Gate 9 deletion and validation, with twelve MLflow
   integer-schema hints and one documented Bayesian warning. The Gate 4
   condition on the composition calibration retention rule was closed in the
   Gate 5 remediation (a 1-df likelihood-ratio test).

Full-model review gates in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md#19-review-gates-before-coding)
apply only when their corresponding complexity stage is reached.