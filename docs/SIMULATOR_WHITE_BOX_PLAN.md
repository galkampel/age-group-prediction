# Plan: the white-box simulator

**Written for:** the model that implements each stop (Opus), and the user
who verifies it. Self-contained: re-verify every fact against the code of
the day before relying on it.

**Status (2026-10-14):** approved by the user; **Stop 1 done (uncommitted,
awaiting the user's review), next Stop 2**. Branch `feat/experiment-notebook`
(PR D's D1a + D1b committed in 2b13970). This refactor lands before PR D's
D1c, and Stop 3 updates the notebook's data cell. A whole-file `ruff format`
of the simulator and its tests (about 340 lines of drift and 17 pre-existing
lint errors) is a separate pass, once the user is satisfied with the
refactor (user's call).

**How to work:** the standing rules of
[EXPERIMENT_NOTEBOOK_PLAN.md §3](EXPERIMENT_NOTEBOOK_PLAN.md): plan mode
first with a short summary in the chat and approval before implementing;
one stop per approval; baseline `uv run pytest -m "not slow"` before the
first edit (expected 1296 passed, 1 skipped, 1 xfailed); no commit, no
`Co-Authored-By`, never `git checkout/stash/reset/mv`; `uv run ...` for every
tool; probes read-only (`PYTHONPATH=src .venv/bin/python -W error -c "..."`);
every class, field, method and check justified by measured evidence or
dropped; one mutation check per claimed behaviour (edit `src` in place, see
the test fail, restore by copy and `cmp`); an independent review subagent
per stop; at each stop a file-by-file summary of the `.py` changes, the
check results (ruff, mypy, the changed tests under `-W error`, the
mutations, the non-slow suite) and suggested commit commands. The user
commits.

## 1. Context

The user: `StudentPopulationSimulator` is a black box. `run(rng)` builds the
five components from the config inside itself, returns only the building
table, stashes everything else in `_last_result`, and the latent effects
travel through a private dict on one component (`latent_effects()`) while
the cohort stage's building effect is thrown away. The user wants:

1. the independent components (`NeighborhoodSimulator`, `BuildingSimulator`,
   `ApartmentSimulator`, `TotalChildrenSimulator`,
   `CohortCompositionSimulator`) defined **outside** and handed in;
2. one method, `create_dataset`, whose body **is** the flow, returning the
   dataset **and** the latent effects;
3. more informative names, and notes where the flow is not obvious.

The user's model of the RNG is the right one and stays: one generator is
created once per dataset and passed to every component in order.

### Three calls settled (the user asked)

- **Who owns the seed (user's call, 2026-10-14, replacing "components take
  only their own settings").** Components keep `SimulationSettings` in their
  constructors and `rng` stays **optional** on every sampling method: when
  omitted, the component creates `np.random.default_rng(simulation.seed)`
  (and `NeighborhoodSimulator` defaults `n_neighborhoods` from it), as
  before the refactor. The orchestrator still owns the dataset's seed: it
  calls `np.random.default_rng(seed)` once in `create_dataset` and passes
  the generator to every stage in order.
- **Latent effects: no flag.** A boolean that changes what a method returns
  makes two return types for one name and forces every caller to remember
  the flag; a result object with a `latent_effects` field costs nothing when
  unused and is the pattern the project already has (`SimulationResult`,
  `BootstrapResult`, `TuningResult`). Each component **returns** its
  effects beside its table; the orchestrator collects them. **Format:** one
  `pd.Series` per effect, indexed by the owner id, in a `dict[str, pd.Series]`
  keyed by the effect's name (as today, plus the cohort effect). A single
  DataFrame would mix two index levels (building ids and neighborhood ids)
  and NaN-pad them; a Series per effect is what the recovery test's
  `.map(effect)` wants.
- **`from_config` classmethod: yes.** The explicit five-line construction
  is the white box and the notebook writes it out; the classmethod is the
  one place that maps TOML settings to components, so tests and doc
  examples that only need a table do not repeat and drift. Pandas,
  pydantic and this project's `load_simulation_config` follow the same
  `from_*` convention.

### Verified facts (read-only, 2026-10-14)

- Components and their sampling draws: `NeighborhoodSimulator.generate(n, rng)`
  (SES, median age, household size, daycares, school status);
  `BuildingSimulator.generate(neighborhoods, rng)` (building counts,
  apartment totals, Dirichlet-multinomial room counts);
  `ApartmentSimulator.generate(buildings, rng)` (one permutation per
  building) plus static `with_context` and `to_cohort_long`;
  `TotalChildrenSimulator.generate(context, rng)` (two random effects, NB2
  counts) plus `compute_log_mu`, `build_design_matrix`, `latent_effects()`;
  `CohortCompositionSimulator.generate(context, rng)` (one building effect,
  multinomial per apartment) plus `compute_theta`, `cohort_probabilities`.
- `pipeline.py`: `SimulationResult(neighborhoods, buildings, apartments,
  final_df, latent_effects)`, `_to_building_level` (aggregation plus three
  accounting checks), `StudentPopulationSimulator(config).run(rng)`,
  `last_result` (deep copies).
- Clients of `run`/`last_result`: `tests/unit/test_pipeline.py` (8 tests),
  `test_apartment_stage.py`, `test_cohort_aggregation_stage.py`,
  `test_total_children_stage.py`, `test_rng_convention.py`,
  `tests/validation/{helpers,test_recovery,test_calibration}.py`,
  `tests/characterization/test_current_behavior.py`, `notebooks/01_eda.py`
  (cell at line 131), `notebooks/03_experiments.py` (data cell), and, found
  at Stop 1, eight more that call `StudentPopulationSimulator(config).run()`
  for a table: `tests/unit/test_{fitted_features,modeling_data,state_bundles,hyperparameter_tuning_integration,inspection_permutation,inspection_shap}.py`,
  `tests/characterization/{eda_baseline,test_experiments_notebook}.py`. Component
  factories live in `tests/unit/helpers.py` (`make_*_simulator(config)`).
  Doc blocks run as written: `docs/HYPERPARAMETER_TUNING.md` §2,
  `docs/MODEL_INSPECTION.md` §2, `docs/SPLITTING.md` §3, `README.md:65`,
  `docs/BAYESIAN_CONDITIONAL_MODEL.md:673`, `docs/MODULE_REFERENCE.md`
  (simulator rows), `docs/EXPERIMENT_NOTEBOOK_PLAN.md` §2 "Synthetic data".
  `src/age_group_prediction` never imports the simulator.
- `config.ApartmentSettings` is an empty "reserved" class, unused anywhere:
  out of scope, noted for later deletion.
- Baseline: non-slow suite 1296 passed, 1 skipped, 1 xfailed.

## Target API (the whole design on one page)

```python
# student_simulator/neighborhood.py
class NeighborhoodSimulator:
    def __init__(self, simulation: SimulationSettings, settings: NeighborhoodSettings) -> None
    def sample_neighborhoods(self, n_neighborhoods: int | None = None, rng: np.random.Generator | None = None) -> pd.DataFrame

# student_simulator/building.py
class BuildingSimulator:
    def __init__(self, simulation: SimulationSettings, settings: BuildingSettings, transforms: TransformSettings) -> None
    def sample_buildings(self, neighborhoods: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame

# student_simulator/apartment.py
class ApartmentSimulator:
    def __init__(self, simulation: SimulationSettings, room_mix: RoomMixSettings) -> None
    def expand_apartments(self, buildings: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame
    @staticmethod
    def join_context(apartments, buildings, neighborhoods) -> pd.DataFrame      # was with_context
    @staticmethod
    def to_cohort_long(apartments) -> pd.DataFrame                              # unchanged

# student_simulator/outcomes.py
LatentEffects = dict[str, pd.Series]   # effect name -> one value per owner id (the Series index)

class TotalChildrenSimulator:
    def __init__(self, simulation: SimulationSettings, settings: TotalChildrenSettings, transforms: TransformSettings) -> None
    def sample_total_children(self, apartment_context, rng=None) -> tuple[pd.DataFrame, LatentEffects]
        # effects: "building_total_count_effect", "neighborhood_total_count_effect"
    def compute_log_mu(...), def build_design_matrix(...)                        # unchanged
    # latent_effects() and the private stash are deleted

class CohortCompositionSimulator:
    def __init__(self, simulation: SimulationSettings, settings: CohortSettings, transforms: TransformSettings,
                 total_children: TotalChildrenSettings) -> None
    def sample_cohorts(self, apartment_context, rng=None) -> tuple[pd.DataFrame, LatentEffects]
        # effects: "building_cohort_effect" (today discarded)
    def compute_theta(...), def cohort_probabilities(...)                        # unchanged

# student_simulator/pipeline.py
@dataclass(frozen=True)
class SimulatedPopulation:                      # was SimulationResult
    """The tables of one dataset and the latent effects that generated it.

    ``building_table`` is the modeling table (one row per building, context
    plus the four counts); the other tables are the stages it was built
    from. ``latent_effects`` are the random effects per owner id; they
    exist for coefficient-recovery checks and never become columns.
    """
    neighborhoods: pd.DataFrame
    buildings: pd.DataFrame
    apartments: pd.DataFrame
    building_table: pd.DataFrame                # was final_df
    latent_effects: LatentEffects = field(repr=False)

def aggregate_to_buildings(apartments, buildings, neighborhoods) -> pd.DataFrame   # was _to_building_level, public

class StudentPopulationSimulator:
    """The flow from neighborhoods to the building table, one generator for every draw."""

    def __init__(
        self,
        neighborhood_simulator: NeighborhoodSimulator,
        building_simulator: BuildingSimulator,
        apartment_simulator: ApartmentSimulator,
        total_children_simulator: TotalChildrenSimulator,
        cohort_composition_simulator: CohortCompositionSimulator,
        *,
        n_neighborhoods: int,
        seed: int,
    ) -> None

    @classmethod
    def from_config(cls, config: Config) -> StudentPopulationSimulator
        # the five constructions from config.simulation (first argument of each), .neighborhood,
        # .building, .transforms, .building.room_mix, .total_children, .cohort;
        # n_neighborhoods and seed from config.simulation

    def create_dataset(self) -> SimulatedPopulation:
        rng = np.random.default_rng(self.seed)
        # 1. Context: one row per neighborhood, then buildings inside them, then
        #    one row per apartment with its room count.
        neighborhoods = self.neighborhood_simulator.sample_neighborhoods(self.n_neighborhoods, rng)
        buildings = self.building_simulator.sample_buildings(neighborhoods, rng)
        apartments = self.apartment_simulator.expand_apartments(buildings, rng)
        # 2. Outcomes per apartment: the total first, then its split into cohorts.
        #    Each stage reads the apartment joined to its building and neighborhood.
        apartment_context = ApartmentSimulator.join_context(apartments, buildings, neighborhoods)
        apartments_with_totals, total_effects = self.total_children_simulator.sample_total_children(apartment_context, rng)
        apartments_with_cohorts, cohort_effects = self.cohort_composition_simulator.sample_cohorts(apartments_with_totals, rng)
        # 3. The modeling table: apartments summed into their buildings, with the
        #    neighborhood context and three accounting identities checked.
        building_table = aggregate_to_buildings(apartments_with_cohorts, buildings, neighborhoods)
        return SimulatedPopulation(neighborhoods, buildings, apartments_with_cohorts, building_table,
                                   latent_effects={**total_effects, **cohort_effects})
```

Notes for the implementer:
- `sample_total_children` returns the context **with** `n_children_total`
  (it does today), so the cohort stage takes that frame directly: the second
  `with_context` join and the two `copy(deep=True)`/column-copy dances in
  `run` disappear. The cohort stage's output keeps the context columns; the
  `apartments` field of the result is that frame (ids, rooms, context, the
  four counts), which is what `to_cohort_long` and the recovery helper
  read. Check `test_cohort_aggregation_stage.py` and `tests/validation/helpers.py`
  for the columns they expect before deciding whether to drop the context
  columns from `apartments`; keep whatever a client reads.
- `last_result`, `_copy_result`, `run` and `latent_effects()` are deleted.
  No defensive copies: the result is a new object per call; the user's rule
  is "validate only what would otherwise pass silently".
- Each stage's docstring says what it draws and from which columns, in a
  few lines; the numbered comments in `create_dataset` are the notes the
  user asked for. The existing per-draw comments in the components stay.
- `n_neighborhoods` and `seed` are validated by nothing new: `default_rng`
  rejects a negative seed; `sample_neighborhoods` already raises on `n <= 0`
  (kept: with `size=0` numpy returns empty arrays silently).
- `__init__.py`: export `SimulatedPopulation`, `LatentEffects`,
  `aggregate_to_buildings`; drop `SimulationResult`.

## Stops

### [x] Stop 1. The five components (files: neighborhood.py, building.py, apartment.py, outcomes.py, pipeline.py `run()` adapter, tests/unit/test_{neighborhood,building,apartment_stage,total_children_stage,cohort_aggregation_stage,rng_convention,design}.py, tests/validation/helpers.py)

Done 2026-10-14, uncommitted.
- Methods renamed as in the target API (`sample_neighborhoods`,
  `sample_buildings`, `expand_apartments`, `join_context`,
  `sample_total_children`, `sample_cohorts`); constructors and the optional
  `rng` unchanged (user's call). The two outcome stages return
  `(frame, LatentEffects)`; `building_cohort_effect` is now returned;
  `latent_effects()` and `_latent_effects` deleted. Each sampling docstring
  says what it draws, in order, and from which columns.
- `run()` adapted (new names, tuples unpacked, same draw order, same two
  effects) so the suite stays green; Stop 2 replaces it. Proof: the default
  config's building table, apartments and both effects hash identically
  before and after (building table `f385be22b88aebf7`, 251 rows).
- Tests: renamed to the new methods; `_state_for_log_mu_checks` returns the
  `Config`; new `test_cohort_effect_is_returned_per_building_and_not_as_a_column`
  (hand-built context, independent of `run()`). `tests/unit/helpers.py`
  unchanged.
- Mutations, all caught: c1–c5 (each sampling method ignores the passed
  `rng`) → its rng-convention test; (a) neighborhood effect under the
  building key → the owner-set test; (b) cohort effects `{}` → the new test.
  Lesson: clear `__pycache__` and set `PYTHONDONTWRITEBYTECODE=1` per
  mutation; two same-size mutations written in the same second reuse a stale
  `.pyc` and look uncaught.
- Checks: ruff no new error (17 pre-existing, one fewer than HEAD); mypy
  `src/student_simulator` 4 errors, all in config.py; the touched tests under
  `-W error`; the slow recovery + calibration tests 2 passed (3.5 s); the
  independent review found no defects (two naming nits, one applied).

### Stop 2. The orchestrator (files: pipeline.py, __init__.py, tests/unit/test_pipeline.py, the pipeline-level tests of test_apartment_stage.py / test_cohort_aggregation_stage.py / test_total_children_stage.py / test_rng_convention.py, tests/characterization/{test_current_behavior,eda_baseline,test_experiments_notebook}.py, tests/unit/test_{fitted_features,modeling_data,state_bundles,hyperparameter_tuning_integration,inspection_permutation,inspection_shap}.py, tests/validation/{helpers,test_recovery,test_calibration}.py)

- `SimulatedPopulation`, `aggregate_to_buildings`, the new
  `StudentPopulationSimulator`, `from_config`, `create_dataset` as above.
- `test_pipeline.py` rewritten: `test_create_dataset_returns_the_building_table_with_the_canonical_columns`,
  `test_the_building_table_equals_the_aggregated_apartments` (the three
  accounting identities, by recomputing from `apartments`),
  `test_latent_effects_hold_the_three_named_effects_per_owner`,
  `test_the_same_seed_reproduces_the_dataset_and_another_seed_changes_it`,
  `test_from_config_builds_the_same_dataset_as_explicit_construction`
  (the white-box construction and the classmethod give equal tables for the
  same config), `test_components_see_one_shared_generator` (spy: wrap a
  component's sampling method and assert the `rng` object is the same
  instance in every call). Delete the `last_result` tests.
- Validation helpers: `result.final_df` → `result.building_table` (the
  constructor and `join_context` changed at Stop 1). `test_current_behavior.py`:
  `run()` + `last_result` → `create_dataset()`. The eight table-only clients:
  `StudentPopulationSimulator(config).run()` →
  `StudentPopulationSimulator.from_config(config).create_dataset().building_table`.
- The stage tests that still reach `run()` for their input
  (`test_rng_convention.py`'s `_result_before_*`, `test_apartment_stage.py`'s
  `_small_result`, `test_cohort_aggregation_stage.py`'s `_result`, and
  `test_total_children_stage.py`'s first test) move to `create_dataset()`.
- Mutations: (d) drop the `apartment_rows != n_apartments` check in
  `aggregate_to_buildings` and feed a building with one apartment removed →
  the accounting test fails; (e) create two generators in `create_dataset`
  (one per stage) → the shared-generator test fails; (f) in `from_config`
  pass `config.cohort` settings to the wrong component → the equality test
  fails.
- Checks: ruff, mypy, the changed tests under `-W error`, the non-slow
  suite (expect the same counts minus the two `last_result` tests plus the
  new ones; report the numbers), and the slow validation tests that touch
  the simulator: `uv run pytest tests/validation/test_recovery.py tests/validation/test_calibration.py -q`
  (report the wall time; the recovery test is the one consumer of the
  latent effects' numeric values).

### Stop 3. The notebooks (files: notebooks/03_experiments.py data cell, notebooks/01_eda.py cell at line 131, tests/characterization/test_experiments_notebook.py)

- `03_experiments.py` data cell becomes the white box the user asked for:
  ```python
  simulator = StudentPopulationSimulator(
      NeighborhoodSimulator(config.simulation, config.neighborhood),
      BuildingSimulator(config.simulation, config.building, config.transforms),
      ApartmentSimulator(config.simulation, config.building.room_mix),
      TotalChildrenSimulator(config.simulation, config.total_children, config.transforms),
      CohortCompositionSimulator(config.simulation, config.cohort, config.transforms, config.total_children),
      n_neighborhoods=n_neighborhoods,
      seed=simulation_seed if simulation_seed is not None else base_config.simulation.seed,
  )
  population = simulator.create_dataset()
  table = ShareTransformer(...).fit_transform(population.building_table)
  ```
  with `config` still the overridden config (the building-rate override
  lives in `config.building`), and `DATA_SETTINGS` reading `simulator.n_neighborhoods`
  and `simulator.seed`. The `simulation.model_copy(update={"n_neighborhoods": ...})`
  override is no longer needed for the simulator (it is a constructor
  argument now); keep `config_params` as the flattened config and add
  `n_neighborhoods`/`seed` from the simulator to `DATA_SETTINGS` as today.
  The printed `data` line is unchanged, so the hash stays `c92d4213...` if
  the draw order is unchanged: **the test's recomputed hash is the check**;
  if it changes, find out why before accepting (the draw order must be the
  same as `run`'s: neighborhoods, buildings, apartments, totals, cohorts).
- `01_eda.py`: `eda_population = StudentPopulationSimulator.from_config(eda_config).create_dataset()`;
  `eda_final_df` → `eda_population.building_table`; `eda_result` → `eda_population`
  (grep the notebook for every later use of `eda_result.` and `eda_final_df`).
- `test_experiments_notebook.py`: `_canonical_hash` uses `from_config(...).create_dataset().building_table`.
- Checks: `marimo check` on both notebooks, the characterization tests,
  the script run of `03_experiments.py`, and `uv run --env-file .env
  notebooks/01_eda.py` (the EDA notebook as a script; report the time).

### Stop 4. Docs, run as written (files: README.md:65 area, docs/SPLITTING.md §3, docs/HYPERPARAMETER_TUNING.md §2, docs/MODEL_INSPECTION.md §2, docs/BAYESIAN_CONDITIONAL_MODEL.md:673, docs/MODULE_REFERENCE.md simulator rows, docs/EXPERIMENT_NOTEBOOK_PLAN.md §2 "Synthetic data", docs/README.md if it names `run`)

- Each code block: `StudentPopulationSimulator(config).run()` →
  `StudentPopulationSimulator.from_config(config).create_dataset().building_table`.
  Re-run each block as written (the tuning example 9 s, the inspection
  example 44 s, the split block seconds) and confirm the printed numbers
  are unchanged; if any differ, stop and report (it means the draw order
  changed).
- `MODULE_REFERENCE.md`: the `pipeline.py` row describes `SimulatedPopulation`,
  `aggregate_to_buildings`, `from_config`, `create_dataset`.
- A short section in whichever doc describes the simulator (check
  `docs/README.md` for the simulator's guide; if none, `MODULE_REFERENCE.md`
  is enough): the flow in five lines, what `latent_effects` holds and why
  it is returned rather than stored.
- Historical docs (GATE_*, archive/) untouched.

## Verification (end to end)

- `uv run pytest -m "not slow"`: all green; report counts against the
  baseline 1296 / 1 / 1.
- `uv run pytest tests/validation/test_recovery.py -q`: the oracle fit with
  the returned effects as offset still recovers the coefficients.
- `uv run --env-file .env notebooks/03_experiments.py` prints
  `data rows=1529 neighborhoods=150 data_hash=c92d4213...` (same population
  as before the refactor).
- `grep -rn 'last_result\|final_df\|SimulationResult\|with_context\|\.run(' src tests notebooks docs/*.md README.md`
  shows only historical docs.

## Suggested commits (one per stop)

```
git add src/student_simulator tests/unit tests/validation/helpers.py docs/SIMULATOR_WHITE_BOX_PLAN.md
git commit -m "refactor(simulator): name the sampling methods and return the latent effects"
git add src/student_simulator tests
git commit -m "refactor(simulator): StudentPopulationSimulator takes its components; create_dataset returns the population and its latent effects"
git add notebooks tests/characterization
git commit -m "feat: notebooks build the simulator from its components"
git add README.md docs
git commit -m "docs: simulator examples use from_config and create_dataset"
```

---

## Appendix: PR D stops D1c–D1e (unchanged spec, resumed after this refactor)

See `docs/EXPERIMENT_NOTEBOOK_PLAN.md` §9 for the done-when of each. Cell
code and tests:

```python
# C5 split (section "3) Train, validation and test split"): SPLITTING.md §3 with the settings
splitter = Splitter(split_method)
split_groups = None if split_method == "random" else groups
train_val_index, test_index = splitter.train_test_indices(table, split_groups, test_size=test_size, random_state=seed)
groups_train_val = None if split_groups is None else take_rows(split_groups, train_val_index)
train_sub, val_sub = splitter.train_test_indices(take_rows(table, train_val_index), groups_train_val,
                                                 test_size=val_size / (1 - test_size), random_state=seed)
train_index, val_index = train_val_index[train_sub], train_val_index[val_sub]
overlap = len(set(train_index) & set(val_index)) + len(set(train_index) & set(test_index)) + len(set(val_index) & set(test_index))
X_train, X_val, X_test = (take_rows(X, i) for i in (train_index, val_index, test_index))   # same for Y, exposure, groups
cv = splitter.cv(n_splits=n_splits, random_state=seed)
print(f"split n_train={len(train_index)} n_val={len(val_index)} n_test={len(test_index)} overlap={overlap}")
# test: test_a_dropped_position_mapping_in_the_split_is_caught (908/311/310, overlap 0); mutation: train_index, val_index = train_sub, val_sub

# C6 feature variants (section "4) Feature variants"): FEATURE_TRANSFORMATIONS.md §8 with tuple columns;
# FEATURE_VARIANTS keyed tree, total_base, cohort_probability_base, total__ses_quadratic, total__daycare_log1p,
# total__room_share_x_household_size, composition__daycare_log1p, composition__room_share_x_median_age;
# each clone(variant).fit(X_train) once; print "variants tree=10 total_base=9 ..." (counts 10/9/9/10/9/12/9/12)
# test: test_a_wrong_interaction_operand_is_caught; mutation: Interaction(left="4_rooms_share", right="n_daycares_500m") under the saturation variant

# C7 MLflow (section "5) MLflow"): refuse when MLFLOW_TRACKING_URI is unset or resolves to <repo>/mlflow.db
# (probe sqlite:///mlflow.db relative to cwd and sqlite:////abs/path first); set_tracking_uri, set_experiment; print "mlflow experiment=... tracking_uri=..."
# tests: ..._opens_no_run (search_runs empty, root mlflow.db mtime unchanged), ..._refuses_the_canonical_store (3 cases),
# test_no_model_is_fitted_outside_a_run_button (source check); mutations: delete the raise; add mlflow.start_run(); add an unguarded .fit(
```
