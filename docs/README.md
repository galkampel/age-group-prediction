# Simulator Documentation

This index defines document authority for the compact component simulator.

## Current Stage 1 Documents

| Document | Authority |
|---|---|
| [Simplified model specification](SIMPLIFIED_MODEL_PLAN.md) | Canonical behavior, formulas, tables, observability rules, and Section 9 complexity roadmap |
| [Compact simulator migration plan](COMPACT_SIMULATOR_MIGRATION_PLAN.md) | Completed implementation record, validation gates, deletion sequence, and model assignments |
| [Marimo research workflow migration plan](MARIMO_MIGRATION_PLAN.md) | Completed Jupyter-to-marimo cutover, package setup, reactive notebook structure, validation gates, and per-step model routing |
| [EDA and predictive modeling plan](EDA_AND_PREDICTIVE_MODELING_PLAN.md) | Active EDA-first research roadmap; predictive modeling is explicitly deferred |
| [Model fitting, evaluation, and feature importance plan](MODEL_FITTING_EVALUATION_AND_FEATURE_IMPORTANCE_PLAN.md) | Active iterative roadmap for direct baselines, independent total/probability modeling, conditional Bayesian modeling, evaluation, importance analysis, and MLflow |
| [Superseded implementation plan](SIMPLIFIED_MODEL_IMPLEMENTATION_PLAN.md) | Historical description of the removed stage/state architecture |

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
- Marimo research workflow migration: complete through Step 8 cutover;
   `research.ipynb` and `ipykernel` have been removed.
- Building-level EDA: planned in `notebooks/01_eda.py`; modeling remains gated
   on the completed EDA findings table.
- Complexity roadmap: specified but deferred.
- Advanced target model: retained as future reference.
- Simplified canonical calibration and oracle recovery: implemented.

Full-model review gates in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md#19-review-gates-before-coding)
apply only when their corresponding complexity stage is reached.