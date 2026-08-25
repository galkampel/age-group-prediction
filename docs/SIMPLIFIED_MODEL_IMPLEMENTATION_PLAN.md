# Simplified Model Implementation Plan (Superseded)

> **Status: superseded architecture reference.** This document describes the
> stage/state implementation that is being removed. Use
> [COMPACT_SIMULATOR_MIGRATION_PLAN.md](COMPACT_SIMULATOR_MIGRATION_PLAN.md) for
> current implementation order, software boundaries, validation gates, and
> deletion decisions. [SIMPLIFIED_MODEL_PLAN.md](SIMPLIFIED_MODEL_PLAN.md)
> remains authoritative for model behavior and formulas.

> **Implemented through Phase 7.** The data-generation and validation pipeline
> is complete. Phase 8 covers export, CLI, and prototype replacement rather
> than additional data-generating behavior.

## 1. Goal and Completion Criteria

Replace the current prototype with a configurable, class-based simulator that:

- generates the canonical neighborhood, building, room-composition,
  apartment, and building-target tables;
- can stop after any generation stage so intermediate tables can be inspected;
- validates configuration before drawing data and validates every table before
  a dependent stage runs;
- separates structural invariants from seed-sensitive statistical checks;
- uses explicit model-visible column lists so latent values cannot leak into a
  fitted model;
- reproduces every generated table from the same configuration and seed; and
- preserves stable orchestration, state, validation, and export contracts for
  the complexity additions in Section 9.

Stage 1 is complete only when the fast test suite, canonical calibration run,
coefficient-recovery check, table round-trip checks, and documentation checks
all pass.

## 2. Sources of Truth

Use the documents in this order:

1. [SIMPLIFIED_MODEL_PLAN.md](SIMPLIFIED_MODEL_PLAN.md), Sections 1-8, defines
   current model behavior, tables, formulas, observability, and validation.
2. This document defines implementation order, software boundaries, tests,
   and delivery gates.
3. Section 9 of the simplified model plan defines the compatibility target for
   later stages; it is not part of the current implementation.
4. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) is an advanced architecture
   reference. Reuse its durable stage, schema, state, and validation patterns,
   but do not introduce its future model behavior into Stage 1.
5. Files under `docs/legacy/` preserve historical assumptions and derivations.
   They are not implementation specifications.

When documents disagree, the first applicable item in this list wins. In
particular, Stage 1 does not generate apartment floor, project type, dates,
planned units, building-type proxies, or zero inflation.

## 3. Design Rules

### 3.1 Configuration owns assumptions

All adjustable numeric and categorical values belong in `configs/stage1.toml`.
Frozen dataclasses provide typed access after loading. Generation methods must
not contain unexplained calibrated literals.

Configuration groups:

| Group | Responsibility |
|---|---|
| `simulation` | Seed, schema version, neighborhood count, run options |
| `transforms` | Named centering references and scaling divisors |
| `neighborhood` | SES, age, household-size, daycare, and school distributions |
| `building` | Building-count and apartment-count distributions and overrides |
| `room_mix` | Supported rooms, base shares, tilt, and Dirichlet concentration |
| `total_children` | NB2 coefficients, dispersion, and shared-effect scales |
| `cohort` | Reference shares, lifecycle coefficients, and building-effect scale |
| `validation` | Calibration ranges, severities, and recovery criteria |
| `export` | Output profile, format, and destination defaults |

Use Python 3.13 `tomllib` to avoid a runtime configuration dependency. Reject
unknown keys and validate the complete object before creating the random
generator. A run receives an immutable configuration snapshot; experiments
create a new configuration rather than mutating one in progress.

The `[transforms]` values make every centering/scaling reference explicit.
`total_children.room_log_mean_effects` names room-category offsets in their
actual units; they are coefficients on the log expected-count scale, not
probabilities. `cohort.reference_shares` remains named as shares because the
values are exactly the cohort probabilities when the lifecycle index is zero.

### 3.2 Classes own orchestration, functions own mathematics

The pipeline and stages are classes. Scaling, saturation, room probabilities,
positive-part interactions, normalization, and NB2 parameter conversion are
pure functions. This keeps random state and table dependencies visible while
making mathematical boundaries easy to test.

### 3.3 Tables are the integration boundary

Stages exchange ordinary pandas DataFrames through a typed state container.
The state stores tables by registered name rather than one hard-coded property
per possible future table. Section 9 can then add projects or neighborhood-year
panels without changing the state abstraction.

Canonical Stage 1 tables:

| Table | Key | Produced by |
|---|---|---|
| `neighborhoods` | `neighborhood_id` | `NeighborhoodStage` |
| `buildings` | `building_id` | `BuildingStage` |
| `building_room_composition` | `building_id`, `rooms` | `BuildingStage` |
| `apartments` | `apartment_id` | `ApartmentStage`, then outcome stages |
| `building_targets` | `building_id` | `BuildingAggregationStage` |

The canonical apartment table stores `building_id`, not a duplicate
`neighborhood_id`. Modeling and inspection views may add neighborhood fields
through a validated many-to-one join.

### 3.4 Schemas own visibility

Every table schema declares keys, required columns, dtypes, nullability,
categories, units, targets, research-truth fields, and model-visible fields. Construct
model input by selecting an allowlist, never by dropping known latent columns.

Store `school_status` canonically. Derive `existing_school` and
`planned_school` in the design matrix and test their mutual exclusivity.
Random effects belong in private `SimulationState` storage keyed by owner ID.
They support generation and oracle recovery but cannot appear in model or
research tables. Future proxy evaluation may add explicitly approved
truth/proxy research tables; those are distinct from random effects.

### 3.5 One random generator belongs to one run

`StudentPopulationSimulator` creates one `numpy.random.Generator` after config
validation and passes it to every stochastic stage. Stages and validators must
not create private generators. Validation must not consume random state.

## 4. Target Package Layout

```text
configs/
  stage1.toml
src/
  student_simulator/
    __init__.py
    config.py
    schemas.py
    state.py
    metadata.py
    design.py
    transforms.py
    pipeline.py
    export.py
    stages/
      __init__.py
      base.py
      neighborhoods.py
      buildings.py
      apartments.py
      total_children.py
      cohort_composition.py
      aggregation.py
    validation/
      __init__.py
      models.py
      invariants.py
      calibration.py
      recovery.py
tests/
  unit/
  integration/
  validation/
```

The existing `SyntheticDataSimulator` has a different schema and probability
model. Isolate or remove it only after replacement tests cover structure and
reproducibility; do not adapt its Poisson implementation incrementally.

## 5. Core Contracts

### 5.1 Stage contract

Each stage implements this conceptual interface:

```python
class SimulationStage(ABC):
    name: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]

    @abstractmethod
    def run(
        self,
        state: SimulationState,
        rng: np.random.Generator,
        config: SimulationConfig,
    ) -> None: ...
```

The pipeline checks prerequisites, runs the stage, and applies all hard
validators for its output before proceeding. Stage order is declared once in
a registry. Later stages may be inserted without modifying existing stages.

### 5.2 Pipeline contract

`StudentPopulationSimulator` exposes:

```text
run()                          run all registered stages
run_until(stage_name)         stop after a validated stage
run_stage(stage_name, state)  run against explicit prerequisite state
validate(state)               rerun applicable checks without using RNG
model_tables(state)           return schema-approved observed tables
research_tables(state)        include generated and approved proxy-truth tables
```

### 5.3 Validation contract

A validation result contains:

- check name and producing stage;
- table and columns involved;
- severity: `error`, `warning`, or `info`;
- observed value and configured target or tolerance;
- pass status; and
- a bounded sample of offending IDs and values.

Hard invariants raise before a dependent stage runs. Calibration checks return
an inspectable report and use configurable severity. A failure must explain
what was measured rather than expose only a bare assertion.

## 6. Implementation Phases

Each phase ends with focused executable validation. Do not begin the next
phase while the current phase's hard gate is failing.

### Phase 1: Configuration, schemas, and state

**Build**

- Add `configs/stage1.toml` containing every Stage 1 default from the canonical
  model plan.
- Add frozen `SimulationConfig`, `TransformConfig`, `NeighborhoodConfig`,
  `BuildingConfig`, `RoomMixConfig`, `TotalChildrenConfig`, `CohortConfig`,
  `ValidationConfig`, and `ExportConfig` dataclasses.
- Add a strict TOML loader and a JSON-serializable configuration snapshot.
- Add `TableSchema`, the Stage 1 schema registry, `SimulationState`, private
  internal-effect storage, and `RunMetadata`.
- Add the stage base class and pipeline registry with a no-op test stage.

**Check**

- Probabilities sum to one and have the expected labels.
- Supported room values and prior/coefficient lengths agree.
- Standard deviations, concentrations, and NB2 dispersion are positive.
- Minimum and maximum counts are coherent.
- Exact override lists have valid values and expected lengths.
- Unknown and missing configuration keys fail with their full path.
- Frozen configurations cannot change after construction.
- Visible schema columns exclude every research-truth field and internal effect.
- Stage prerequisites, stage order, and `run_until` behave as declared.

**Gate:** `tests/unit/test_config.py`, `test_schemas.py`, and the pipeline
contract test pass without generating a full dataset.

### Phase 2: Mathematical transforms and design builders

**Build**

- Implement household-size and age scaling, centered rooms, daycare
  saturation, positive-part interaction, positive-weight normalization, and
  stable row normalization.
- Implement room-mix expected probabilities separately from Dirichlet and
  multinomial draws.
- Implement NB2 sampling using the documented parameterization
  $E[N]=\mu$ and $Var(N)=\mu+\mu^2/\phi$.
- Add `TotalChildrenDesignBuilder` and `CohortDesignBuilder`. Builders join
  observed tables, derive transformations and school dummies, and return
  columns keyed exactly to configured coefficient names.

Each outcome stage performs one validated apartment-context join. A shared
feature helper then applies the configured transformations in a visible order:
raw input, centering/scaling, named feature, and model term. Do not cache this
derived context in state; recomputing one join is simpler than cache invalidation
at the current scale.

Outcome samplers consume named design terms rather than embedding every
feature in branching code. Adding a Section 9 feature should require a config
term and design-builder extension, not a rewrite of NB2 or multinomial logic.

**Check**

- Named scales equal zero at their documented references.
- Saturation starts at zero, is monotone, and approaches one.
- Room probabilities are finite, positive, sum to one, and tilt toward larger
  apartments as household size or SES increases.
- The room-age interaction is active only for rooms above four and age above
  37.
- Design columns have a deterministic order and exactly match config keys.
- No latent column enters an observed design matrix.
- NB2 draws have the expected empirical mean and variance at several values of
  $\mu$ and $\phi$ using generous, deterministic tolerances.

**Gate:** transform and design unit tests pass independently of pipeline size.

### Phase 3: Neighborhoods

**Build**

Implement `NeighborhoodStage` using Section 6, Step 1 exactly. Generate raw
observable features and canonical `school_status`. Draw the neighborhood total
count random effect once into private run state.

**Check every run**

- Row count equals configured `n_neighborhoods`.
- IDs are unique and stable.
- SES, median age, and household size obey configured bounds.
- Daycare counts are nonnegative integers.
- School status uses only `existing`, `planned`, and `none`.
- No required value is null or non-finite.
- Same config and seed produce identical tables; another seed changes data.

**Report at canonical scale**

- Means, standard deviations, clipping shares, school shares, and correlations
  implied by the age-household and age-daycare relationships.

**Gate:** the neighborhood table can be generated, inspected, and validated in
isolation with `run_until("neighborhoods")`.

### Phase 4: Buildings, room composition, and apartments

**Build**

- Implement configured building counts per neighborhood, including exact
  override support.
- Implement configured apartment totals per building, including exact
  overrides.
- Draw each building's room mix and multinomial room counts.
- Emit all four room rows for every building, including explicit zeros.
- Draw building total-count and cohort-composition effects once into private
  run state.
- Expand room counts to apartments and shuffle labels with the shared RNG.

Stage 1 does not assign floor. Apartment records initially contain only
`apartment_id`, `building_id`, and `rooms`.

**Check every run**

- Building and apartment IDs are unique and all foreign keys resolve.
- Generated and overridden counts obey their contracts exactly.
- Every building has one row for every supported room value.
- Room counts are nonnegative integers and sum to the building apartment total.
- Expanded apartment room counts reproduce the long room table exactly.
- Total apartment rows equal total room-composition counts.
- Internal effects have one value per owning neighborhood or building and are
  absent from all generated tables.

**Report at canonical scale**

- Building and apartment count distributions.
- Overall room shares and building-to-building variation.
- Mean room count against household size and SES to confirm tilt direction.

**Gate:** physical tables pass accounting and join checks before outcomes are
generated.

### Phase 5: Total children

**Build**

Implement `TotalChildrenStage` using Section 6, Step 4:

- categorical room effects;
- positive SES-square U-shape;
- household-size, saturated-daycare, school-status, and median-age effects;
- the rooms-by-household-size interaction;
- shared building and neighborhood effects; and
- NB2 dispersion.

Map internal effects by owner ID after constructing the observed design. Store
only `n_children_total` on the canonical apartment table.

**Check every run**

- The linear predictor and $\mu$ are finite and $\mu>0$.
- Counts are nonnegative integers.
- Hand-built states isolate each coefficient and reproduce exact expected
  log-mean differences.
- Household size remains positive for every supported room count.
- Shared effects are constant for descendants of the same owner.

**Report at canonical scale**

- Mean children per apartment and zero share.
- Apartment variance-to-mean ratio.
- Mean count by room value.
- Room-by-household-size expected-mean grid at household sizes 1.8, 2.6, 3.1,
  3.6, and 4.2.
- Extreme six-room/high-household predictions.

Do not add a hurdle or zero-inflation component. First compare observed zero
share with simulations from the fitted NB2 model as specified in Section 8.

**Gate:** structural outcome checks pass and canonical metrics are reported
against configurable tolerances.

### Phase 6: Cohort composition and building aggregation

**Build**

- Implement `CohortCompositionStage` with the ordered lifecycle index,
  one-sided room-age interaction, shared building effect, configured reference
  shares, and one multinomial draw per apartment.
- Implement `BuildingAggregationStage` as a deterministic group-by producing
  apartment count, all three cohort totals, and total children.
- Add derived apartment and long cohort views for modeling and plotting without
  duplicating canonical relationships.

Reference shares are exactly 40%/35%/25% at $\theta=0$. Report the generated
sample mean but do not force it to those shares with an intercept-calibration
loop.

**Check every run**

- Cohort values are nonnegative integers.
- Cohorts sum exactly to `n_children_total` for every apartment.
- Zero-child apartments receive three zeros.
- Lifecycle probabilities are finite, nonnegative, and sum to one.
- Positive lifecycle shifts mass from kindergarten toward high school.
- The room-age interaction is zero in its three inactive quadrants.
- Every building appears once in `building_targets`.
- Building targets equal grouped apartment targets exactly.

**Report at canonical scale**

- Reference probabilities at $\theta=0$ and observed sample shares.
- Composition by room, age, daycare, and household-size scenarios.
- Building total variance-to-mean ratio.

**Gate:** all cross-table accounting identities close exactly.

### Phase 7: Validation and coefficient recovery

**Build**

- Implement structured `ValidationResult` and `ValidationReport` records.
- Implement schema, key, foreign-key, cardinality, range, integer, accounting,
  and visibility invariants.
- Implement canonical statistical checks from Section 8 with thresholds and
  severity in configuration.
- Implement the expected zero share under the generating NB2 means.
- Implement oracle recovery using generated random effects as offsets and a
  clearly labeled naive NB2 benchmark that ignores clustering.
- Add hierarchical recovery if the selected validation library supports the
  required NB2 mixed model reliably.

Add `statsmodels` to a separate validation dependency group. Core generation
must continue to require only NumPy, pandas, and SciPy.

Hard invariants run after every stage before a dependent stage starts.
Calibration runs automatically after a complete pipeline execution. The zero
share diagnostic uses the analytic probability
$P(N=0)=(\phi/(\phi+\mu))^\phi$ for each apartment's generating mean; it does
not estimate means from observed counts. Recovery remains in the marked slow
suite and does not expose random effects as tables.

**Test organization**

- Fast unit and integration tests cover deterministic contracts.
- Hand-built invalid tables prove that each invariant catches its intended
  defect and reports useful IDs.
- Tests marked `calibration` use one named full-size seed and config.
- Tests marked `slow` run coefficient recovery.
- Additional seeds produce reports or scheduled checks instead of flaky
  per-commit assertions.

**Gate:** fast validation, canonical calibration, and oracle recovery all pass;
naive recovery limitations are visible and documented rather than mistaken for
generator defects.

### Phase 8: Export, CLI, and replacement

**Build**

- Add model, research, and validation-report export profiles.
- Guarantee CSV initially. Add Parquet only after deliberately selecting an
  engine and adding its dependency.
- Write configuration snapshot, schema version, run metadata, and validation
  report beside exported tables.
- Replace `main.py` with a thin CLI accepting config path, output directory,
  export profile, optional seed override, `run_until`, and validation mode.
- Export the supported public API from `student_simulator.__init__`.
- Remove or isolate the old prototype only after new end-to-end coverage passes.

**Check**

- CSV round trips preserve keys, dtypes, row counts, and accounting.
- Model exports contain no latent columns.
- Research exports include only explicitly approved latent truth.
- Two runs with the same config and seed have identical canonical DataFrames.
- Another seed changes stochastic values without changing schemas.
- CLI smoke execution creates the expected tables and report.

**Gate:** the generated data can be inspected both in memory and after export,
and the old prototype is no longer a competing public implementation.

## 7. Validation Matrix by Table

| Table | Hard checks | Statistical or inspection checks |
|---|---|---|
| `neighborhoods` | Schema, key, bounds, category, finite values | Moments, clipping, category shares, expected correlations |
| `buildings` | Schema, key, neighborhood FK, configured counts | Building-size distribution |
| `building_room_composition` | Composite key, four rows/building, integer counts, exact sums | Room shares, tilt, concentration behavior |
| `apartments` | Key, building FK, supported rooms, integer targets, cohort sum | Child mean, zeros, dispersion, room and lifecycle trends |
| `building_targets` | Key, all buildings present, exact grouped totals | Building dispersion and cohort shares |
| internal effects | One finite value per owner; never exported | Configured standard deviations |
| model view | Join cardinality, allowlist, no research truth | Design-matrix rank and scenario grids |

## 8. Commands and Delivery Checks

Add pytest markers in `pyproject.toml`, then use these gates:

```bash
pytest -m "not calibration and not slow"
pytest -m calibration -v
pytest -m slow tests/validation/test_recovery.py -v
```

The delivery check also runs the CLI twice with the canonical config and seed,
compares canonical tables with strict DataFrame equality, then runs once with a
different seed and confirms that stochastic tables differ.

After any documentation move, search all Markdown files for removed paths and
run a Markdown link checker if one is configured. No current document may link
to a deleted proposal or an old root legacy path.

## 9. Compatibility with Future Complexity

Future-proofing means preserving useful contracts, not adding inactive flags or
empty future columns. Before declaring Stage 1 complete, review these changes:

| Section 9 extension | Expected Stage 1-compatible change |
|---|---|
| Observable baseline feature | Add config, schema metadata, design term, and ablation check |
| Project and date generation | Register new project/time schemas and stages before existing context stages |
| Neighborhood-year panel | Register a panel table and prediction-time join stage |
| Room or building proxy | Add truth to research schema, proxy to visible schema, and a proxy stage |
| Net-new exposure | Add household origin/exposure stage and target columns without replacing resident counts |
| Population/environment dynamics | Add providers and dated tables while retaining stage/state contracts |
| Cohort progression handoff | Add an export contract with cohort counts and reference date |
| Deployment checks | Add validators and decision reports without changing generation tables |

The compatibility review passes only if these additions do not require
replacing `SimulationState`, pipeline orchestration, NB2 sampling, cohort
multinomial sampling, validation result types, or export profile semantics.

## 10. Explicit Scope Boundaries

Included now:

- all generation rules and tables in Sections 1-8;
- external configuration and immutable typed access;
- independently runnable stages;
- structural, statistical, zero-share, and recovery validation;
- derived inspection views, exports, metadata, and a CLI; and
- tests that exercise formulas, stages, tables, and the full pipeline.

Deferred to Section 9:

- projects, time, planned units, building-type proxies, and floor estimation;
- population and environment panels, spillovers, intensity, and infrastructure
  lag;
- net-new exposure, returning residents, and registration leakage;
- ages 0-2 and cohort progression;
- cold-start shrinkage and asymmetric decision loss; and
- real-data providers or production forecasting.

No deferred variable should remain as an unused Stage 1 column or config flag.