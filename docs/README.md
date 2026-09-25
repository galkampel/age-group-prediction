# Documentation

An index of the project's documents and what each one is authoritative for. If
implementation details conflict, the model specification controls behavior.

## Current Status

- **Simulator:** the simplified model is implemented and calibrated. The
  complexity roadmap in its Section 9 is specified but deferred.
- **Modeling (the stack still in use: `models/`, `experiment/`, `tracking/`):** the
  rebuild's Gates 1–9 were completed and accepted on 2026-09-15. The canonical
  final run is complete: the Bayesian conditional model was selected, with the
  other two families logged as comparators. The historical records below hold
  the full evidence.
- **In progress:**
  - the scikit-learn-style re-implementation of the models in `modeling/`
    (Model A done; see the [plan](MODEL_REIMPLEMENTATION_PLAN.md));
  - the `hyperparameter_tuning` package (the evaluator is done; see its
    [plan](HYPERPARAMETER_TUNING_PLAN.md));
  - the `splitting` package (built and tested, not yet wired).

## Simulator

| Document | Authority |
|---|---|
| [Simplified model specification](SIMPLIFIED_MODEL_PLAN.md) | Canonical behavior, formulas, tables, observability rules, and Section 9 complexity roadmap |

## User Guides

| Document | Authority |
|---|---|
| [Modeling guide](MODELING_GUIDE.md) | User guide to the simulator-to-model pipeline, public APIs, configuration, evidence, and invariants |
| [MLflow experiments guide](MLFLOW_EXPERIMENTS_GUIDE.md) | User guide to scratch tracking, run layout, metrics, search, final-run semantics, and model loading |

## Model And Component References

| Document | Authority |
|---|---|
| [Direct cohort model](DIRECT_COHORT_MODEL.md) | Model A. §0: the rebuilt `modeling.DirectCohortModel` (equations, exposure offset, API). §1–§10: the current per-cohort LightGBM families, tuning, marginal scoring, bootstrap uncertainty, persistence |
| [Independent total and probability model](INDEPENDENT_TOTAL_PROBABILITY_MODEL.md) | Model B: NB2/Poisson total with exposure offset, grouped multinomial composition, temperature calibration, joint scoring, bootstrap uncertainty |
| [Bayesian conditional model overview](BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md) | Model C in brief: statistical model, fitting and diagnostics, prediction, selection, persistence, configuration |
| [Bayesian NB2 + Dirichlet-Multinomial guide](BAYESIAN_CONDITIONAL_MODEL.md) | Model C: selection and comparison, persistence, configuration, metadata, and a technical explanation of the model's statistics, Pyro syntax, tensor shapes, estimator/component/inference ownership, prediction, diagnostics, and runtime workarounds |
| [Feature transformations](FEATURE_TRANSFORMATIONS.md) | Which transformation each feature gets in each model and why; the per-model declarations built with the `feature_engineering` package |
| [Feature engineering](FEATURE_ENGINEERING.md) | Feature specs, validation rules, fitted transformations, output columns, per-model use, and candidate forms of the current `fitted_features.py` |
| [Data and splitting](DATA_AND_SPLITTING.md) | Modeling table and schema, known-neighborhood outer split, write-once split manifest and replay, training-only folds, partition checks, the canonical lockbox |
| [Splitting](SPLITTING.md) | The `splitting` package that replaces the above: three split methods each pairing a train/test split with its cross-validator, per-group-size coverage, how to choose a method, hazards |
| [Evaluation and metrics](EVALUATION_AND_METRICS.md) | Prediction contract, metric protocol and capabilities, metric formulas, canonical metric sets, likelihood comparability, neighborhood-cluster bootstrap |
| [Cross-validation and selection](CROSS_VALIDATION_AND_SELECTION.md) | Candidate registry, cross-validation runner, fold artifacts and reload checks, permutation importance, within-approach selection, selection freeze, cross-family rule, seeds and provenance |
| [Final evaluation](FINAL_EVALUATION.md) | Pretest-freeze verification, guarded full-training refit, attempt fingerprint, one-time lockbox evaluation, tracked-run guards, final evidence, pyfunc serving contract |
| [Module reference](MODULE_REFERENCE.md) | One entry per source module: responsibility, main public API, internal dependencies, and where to read more |
| [Known code issues (TODO)](TODO.md) | Open correctness, robustness, comment, and test-gap items, plus one deferred analysis item |

## Active Plans

| Document | Authority |
|---|---|
| [Model re-implementation plan](MODEL_REIMPLEMENTATION_PLAN.md) | Decisions and step-by-step record for the scikit-learn-style `modeling` package |
| [Hyperparameter tuning plan](HYPERPARAMETER_TUNING_PLAN.md) | Decisions, design and remaining work for the `hyperparameter_tuning` package |

## Extensions Beyond The Simplified Model (Not Implemented)

These documents contain unique future-stage material: time, projects, building
types, population dynamics, and the software and validation boundaries for
them. They are not implementation specifications for the current simulator.

1. [Problem definition](PROBLEM_DEFINITION.md) defines the eventual
   building-level target and temporal boundary.
2. [Data generation plan](DATA_GENERATION_PLAN.md) contains the full causal
   process with time, proxies, exposure, environment, and decomposition.
3. [Parameter reference](PARAMETER_REFERENCE.md) explains full-model
   parameters, units, and unresolved choices.
4. [Advanced implementation plan](IMPLEMENTATION_PLAN.md) proposes future
   provider, schema, pipeline, and validation architecture. Its
   [review gates](IMPLEMENTATION_PLAN.md#19-review-gates-before-coding) apply
   only when their corresponding complexity stage is reached.

Files under [`legacy/`](legacy/) preserve the original assumptions and
derivations. They may conflict with the current simplified model and must not
be used as implementation specifications:

- [Original data-generation guide](legacy/DATA_GENERATION_GUIDE.md)
- [Original English source specification](legacy/specification_source_en.md)

## Historical Records

The acceptance records of the current modeling stack. The code in `models/`,
`experiment/` and `tracking/` still cites them, so they are removed together
with that code.

| Document | Authority |
|---|---|
| [Modeling rebuild and experiment tracking plan](MODELING_REBUILD_PLAN.md) | Implementation and acceptance record for shared model contracts, three model families, evaluation, strict test isolation, MLflow, and Gate 9 cleanup |
| [Gate validation findings](GATE_VALIDATION_FINDINGS.md) | Independent validation record: per-gate verdicts, findings, withdrawn findings, remediation, and the Gate 8 canonical run record |
| [Gate 9 independent validation report](GATE_9_INDEPENDENT_VALIDATION_REPORT.md) | Post-completion audit verdict, findings, remediation, command ledger, and protected-evidence comparison |
