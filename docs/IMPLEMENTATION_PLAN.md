# Simulator Architecture and Validation Plan

> **Status: advanced architecture reference.** Implement the model in
> [SIMPLIFIED_MODEL_PLAN.md](SIMPLIFIED_MODEL_PLAN.md) first, then adopt these
> boundaries as later stages require them.

## 1. Goal

Refactor the current prototype into a configurable, class-based pipeline in
which each generation stage can be run, inspected, and validated separately.

This document describes the intended design only. Implementation begins after
the problem and formula plans are reviewed.

Related documents:

- [PROBLEM_DEFINITION.md](PROBLEM_DEFINITION.md): scope and data contracts.
- [DATA_GENERATION_PLAN.md](DATA_GENERATION_PLAN.md): formulas and stage order.
- [PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md): parameter definitions,
  units, direction, and generation intuition.

## 2. Design Principles

1. **Pipeline classes own orchestration.** They declare dependencies and update
   the simulation state in causal order.
2. **Pure functions own mathematics.** Sigmoid, softmax, room inversion, CDF
   category probabilities, and standardization should not depend on mutable
   pipeline state.
3. **Configuration owns assumptions.** No calibrated number should be hidden
   inside a method body.
4. **Schemas own observability.** Explicit allowlists define model-visible
   columns and prevent latent leakage.
5. **Validators own assertions.** Generation methods should enforce local
   preconditions; dedicated validators should enforce table and calibration
   contracts.
6. **One RNG belongs to one run.** The pipeline creates or receives a single
   generator and passes it to stochastic stages.
7. **Tables remain the integration boundary.** Classes organize behavior, but
   outputs remain ordinary pandas DataFrames that can be inspected and saved.

## 3. Proposed Package Layout

```text
src/
  student_simulator/
    __init__.py
    config.py
    schemas.py
    state.py
    pipeline.py
    metadata.py
    providers/
      population.py
      institutions.py
    stages/
      base.py
      areas.py
      projects.py
      buildings.py
      exposure.py
      population.py
      environment.py
      total_children.py
      cohort_composition.py
      decomposition.py
    transforms/
      probabilities.py
      room_model.py
      standardization.py
    validation/
      invariants.py
      calibration.py
      coefficient_recovery.py
    export.py
tests/
  unit/
  integration/
  calibration/
```

The old `SyntheticDataSimulator` should be removed or isolated as a legacy
example once replacement behavior is covered. It should not be adapted one
method at a time because its unit of observation and causal model are different.

## 4. Configuration Model

Use frozen dataclasses so a run's assumptions cannot change halfway through.
Nested configuration keeps related constants together without creating one
constructor with dozens of unrelated arguments.

Planned groups:

| Configuration | Responsibility |
|---|---|
| `SimulationConfig` | Seed-independent top-level composition and run options |
| `AreaConfig` | Counts, city bounds, baselines, mixture parameters |
| `ProjectConfig` | Project count, size distribution, zoning probabilities |
| `BuildingConfig` | Form, room prior, area model, plates, category boundaries |
| `ExposureConfig` | Realization, outliers, lag, fill, zoning exposure shares |
| `PopulationConfig` | Years, trends, jumps, interpolation mode and weight |
| `EnvironmentConfig` | Radius, lookback, institution and load parameters |
| `Stage1Config` | Main coefficients, interactions, random effects, dispersion |
| `Stage2Config` | Youth coefficients, target composition, calibration, tau |
| `DecompositionConfig` | Relocation and registration leakage parameters |
| `ValidationConfig` | Structural and stochastic acceptance tolerances |
| `ExportConfig` | Output path, format, latent inclusion, metadata behavior |

Canonical defaults remain module-level constants, then populate one
`DEFAULT_CONFIG`. This provides both grep-friendly provenance and convenient
run-level replacement.

Configuration validation occurs before any random draw. Examples:

- probability vectors sum to one;
- room values and prior lengths agree;
- category boundaries are sorted and cover the full positive range;
- year ranges and anchor years are coherent;
- minimum project buildings do not exceed the maximum;
- Dirichlet concentrations and standard deviations are positive;
- every coefficient dictionary has exactly the schema's expected keys.

## 5. Simulation State

Use one state container holding optional tables and run metadata:

```text
SimulationState
  areas
  projects
  buildings
  population
  environment
  results
  model_matrices
  metadata
```

`SimulationState` is not a second copy of the data. It is the typed container
through which stages exchange their DataFrames.

Run metadata should include:

- seed and a serialized configuration snapshot;
- simulator version;
- interpolation mode;
- stage completion and validation status;
- standardization means and standard deviations;
- calibrated composition intercepts;
- generation timestamp and output schema version.

## 6. Stage Interface

Every stage follows one conceptual contract:

```text
name
requires
produces
run(state, rng, config) -> state
validate(state, config) -> validation report or raise
```

The stage receives the RNG rather than constructing one. It should refuse to
run when required tables or columns are absent.

Planned stages:

| Stage class | Requires | Produces or appends |
|---|---|---|
| `AreaStage` | Configuration | `areas` |
| `ProjectStage` | `areas` | `projects` |
| `BuildingStage` | `areas`, `projects` | `buildings`, project totals |
| `ExposureStage` | `buildings`, `projects` | exposure columns |
| `PopulationStage` | `areas`, exposed buildings | `population` |
| `EnvironmentStage` | buildings, population | `environment` |
| `TotalChildrenStage` | visible joined features | stage-1 result columns |
| `CohortCompositionStage` | totals and predictors | cohort result columns |
| `DecompositionStage` | cohorts, zoning, contrast | decomposition columns |

The pipeline validates immediately after each stage. A later stage never runs
on state that failed an earlier stage's invariants.

## 7. Pipeline Interface

The top-level `StudentPopulationSimulator` should support:

```text
run()                          full validated simulation
run_until(stage_name)         stop after validating one stage
run_stage(stage_name, state)  advanced use with explicit prerequisite state
validate(state)               rerun all applicable validation
visible_dataset(state)        return only approved model-visible columns
```

`run_until` is important for analysis notebooks and debugging. It makes it
possible to inspect room inversion before population outcomes or inspect stock
history before environment generation.

Stage order is declared once in the pipeline. Users may disable optional
observation layers such as interpolation, but may not reorder causal stages.

## 8. Provider Boundaries

### 8.1 Population provider

Define one population interface returning the population table contract:

- `SyntheticPopulationProvider` uses area baselines and occupancy events.
- `CBSMunicipalPopulationProvider` will later ingest manually downloaded data.

The real provider must enumerate missing files and schema mismatches. It must
never call the synthetic provider as an implicit fallback.

### 8.2 Institution provider

The first provider samples daycare and school state directly at building time,
as specified by the formulas.

A future dated provider will accept institution IDs, coordinates, opening
years, and closing years, then evaluate:

$$opened_i\le t^b_0<closed_i$$

Both providers must produce the same building-level environment columns.

## 9. Schema and Visibility Contracts

Define a schema object for every table containing:

- key columns and expected uniqueness;
- required columns and dtypes;
- nullable columns;
- latent columns;
- model-visible columns;
- allowed categorical values;
- units and short descriptions.

Model input is created by selecting the allowlist:

$$X_b=building[VISIBLE\_BUILDING]
\Join environment[VISIBLE\_ENV]
\Join population[VISIBLE\_POP]$$

It must never be created as "all columns except known latent columns." New
latent diagnostics would otherwise leak automatically.

Derived denominators such as population and stock should be classified
explicitly. Their presence in a visible output table does not automatically
make them predictors.

## 10. Mathematical Transform Boundaries

The following operations should be pure, independently tested functions:

- stable sigmoid and row-wise softmax;
- normalization of positive weights;
- z-score fitting and application;
- normal-CDF interval probability;
- area-conditional room prior;
- Bayesian room inversion by area category;
- daycare saturation;
- composition intercept calibration;
- Euclidean distance in kilometres.

Pure functions make numerical edge cases testable without constructing the
full city. They also keep stage classes focused on table dependencies.

## 11. Validation Layers

Validation has three layers with different purposes.

### 11.1 Configuration validation

Runs before simulation and has no stochastic tolerance. It catches impossible
or inconsistent assumptions.

Examples:

- malformed probability vectors;
- missing coefficient names;
- overlapping or incomplete area categories;
- invalid year windows;
- non-positive scale parameters.

### 11.2 Stage invariants

Run after every stage and must pass for every seed and supported scale.

Examples:

- keys are unique and foreign keys resolve;
- probability vectors sum to one;
- values remain in mathematical domains;
- temporal ordering is valid;
- table cardinalities are correct;
- accounting identities close;
- model matrices contain no latent columns.

Invariant failures should identify the stage, condition, and a small sample of
offending IDs and values.

### 11.3 Statistical calibration

Run on a canonical seed and full-size configuration. These checks evaluate the
behavior of distributions and therefore use tolerances.

Examples:

- children per exposed unit;
- mean cohort composition;
- room-share and floor recovery correlations;
- form classification accuracy and within-area persistence;
- interpolation error behavior;
- coefficient recovery.

Calibration checks should be separated from fast unit tests so small samples
do not create random failures.

## 12. Validation by Stage

| Stage | Fast invariant | Canonical calibration |
|---|---|---|
| Areas | IDs, ranges, population total, prior sums | State-share shape and spatial hierarchy |
| Projects | Valid area, zoning, timing, count bounds | Zoning trend, median buildings above one |
| Buildings | Share sums, form threshold, unit bounds | Room/floor recovery and form persistence |
| Exposure | Timing and exposure identities | Realization variance decreases with size |
| Population | Full panel, monotone stock, correct jumps | Interpolation error correlates with construction |
| Environment | Building-time key, SPILL predicates, load rule | Plausible intensity and institution distributions |
| Stage 1 | Finite means, integer totals, shared effects | 0.45-0.50 children per exposed unit |
| Stage 2 | Probability/count identities, tau bounds | Composition within 0.02 of target |
| Decomposition | Accounting identities and bounds | Relocation share has meaningful spread |

## 13. Test Plan

### Unit tests

- Every pure transform at ordinary and boundary inputs.
- Same seed and configuration produce identical tables.
- Different seeds change stochastic outputs.
- Stage refuses missing prerequisites.
- Visible allowlists exclude every latent schema field.
- Zero-construction interpolation uses linear fallback.
- Zero-child composition returns zero counts safely.
- School probability above 0.75 does not create an invalid planned interval.

### Integration tests

- Small pipeline run creates all six tables.
- Every join has the expected cardinality.
- `run_until` produces only tables through the requested stage.
- Full run accounting closes from total children through decomposition.
- Saving and reloading preserves table schema and metadata.

### Temporal leakage tests

Use hand-built miniature tables rather than random data:

- a future occupancy event cannot affect earlier stock;
- a building outside the five-year SPILL window is excluded;
- a nearby future building is excluded;
- a distant recent building is excluded;
- two buildings in one project use their own construction starts;
- population joins use building area and start, never project first occupancy.

### Calibration tests

Use one named canonical seed and default scale. Report measured values alongside
targets so a failure is diagnosable. Store tolerances in `ValidationConfig`,
not as unexplained literals in tests.

### Coefficient-recovery test

Fit an NB2 GLM with $\log U^{new}$ as offset using the exact generating design
matrix. Confirm approximately 90% of included coefficients fall within two
estimated standard errors of the configured values.

Run recovery as a slower test because it depends on `statsmodels` and full-size
data. Deliberately omitted features should be documented before interpreting
their resulting bias.

## 14. Error Handling

Use exceptions with domain context rather than generic assertion messages at
public boundaries. Internal validators may use assertions when they attach
useful diagnostics.

Expected messages include:

- stage name and missing prerequisite table;
- project ID and generated total after resampling exhaustion;
- feature name with zero standard deviation;
- area and interval with malformed interpolation anchors;
- missing real-data file paths and expected schemas;
- sample IDs violating a join, range, or accounting identity.

No validation path should consume RNG state. Running validation must not change
later simulation output.

## 15. Documentation and Comments

Code documentation should explain decisions that are not obvious from syntax:

- why room priors and area means both depend on area characteristics;
- why environment joins use building start;
- why returning residents reduce exposure;
- why floor plate variation is required;
- why HHI affects concentration rather than the mean;
- why softmax intercepts require calibration;
- why decomposition remains fractional.

Avoid comments that merely restate assignments or formulas. Each public config,
stage, provider, table schema, and exported transform should have a concise
docstring stating its contract and units.

The generated dataset should ship with a data dictionary derived from schema
definitions so documentation and exports cannot drift independently.

## 16. Export Plan

Support separate export profiles:

1. **Research export:** complete tables including clearly named latent truth.
2. **Model export:** explicit visible features and targets only.
3. **Validation report:** invariant results, calibration metrics, configuration,
   and seed.

CSV or Parquet should be the primary tabular format. Excel may be offered for
inspection, but it should not be the canonical storage format because dtype and
schema fidelity are weaker.

Every export directory should include run metadata and schema version. A model
export must be reproducible from the corresponding research export and schema
allowlists.

## 17. Planned Dependencies

Core generation:

- `numpy`
- `pandas`
- `scipy`

Optional export:

- `openpyxl` for Excel
- a Parquet engine if Parquet is selected

Validation only:

- `pytest`
- `statsmodels` for NB2 coefficient recovery

Dependencies should be grouped so users running only the generator do not need
the coefficient-recovery stack.

## 18. Implementation Milestones

No implementation is part of the current documentation task. When coding is
approved, proceed in these independently reviewable milestones.

### Milestone 1: Contracts and foundation

- Package skeleton, frozen configuration, schemas, state, and metadata.
- RNG ownership and deterministic-run test.
- Configuration and schema validators.

### Milestone 2: Physical city generation

- Areas, projects, buildings, room transforms, and exposure.
- Stage-level structural tests and project resampling behavior.

### Milestone 3: Time-dependent context

- Population provider, optional interpolation, and environment.
- Hand-built temporal leakage and SPILL tests.

### Milestone 4: Outcome model

- Model-matrix construction, stage 1, calibrated stage 2, and decomposition.
- Accounting and monotonicity tests.

### Milestone 5: Calibration and exports

- Canonical simulation report and coefficient recovery.
- Research/model exports, metadata, and generated data dictionary.
- CLI wiring and concise README usage.

Each milestone ends with focused executable validation before the next one
begins.

## 19. Review Gates Before Coding

The following decisions should be approved before implementation:

1. Whether stock-weighted project-to-area assignment is acceptable.
2. Whether bounded project-level resampling is preferred to proportional unit
   reconciliation.
3. Whether interpolation is enabled in the canonical default run.
4. Whether the first version may reserve but not generate ages 0-2.
5. Whether full latent research exports are acceptable from a privacy and
   workflow perspective, even though all data is synthetic.
6. Whether Parquet, CSV, or both are required as canonical outputs.

Once these are fixed, configuration defaults and acceptance tolerances become
the reviewed simulation contract rather than incidental implementation details.