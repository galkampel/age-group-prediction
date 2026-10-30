# Plan: the white-box simulator

> **2026-10-29:** the `hyperparameter_tuning` package this document names was removed with its docs ([NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) D17, T4a); T4b rebuilds it from the experiment notebook's section 5.

**Written for:** the model that implements each stop (Opus), and the user
who verifies it. Self-contained: re-verify every fact against the code of
the day before relying on it.

**Status (2026-10-14):** approved by the user; Stops 1–3 committed
(`6b90805`, `59d15d6`, `e66b97d`). **Next: Stop 4** (docs) and then the
cleanup stop (dead code, then a whole-file `ruff format` of the simulator and
its tests, once the user is satisfied). The experiment notebook continues in
parallel from [EXPERIMENT_NOTEBOOK_HANDOFF.md](EXPERIMENT_NOTEBOOK_HANDOFF.md).

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
2. one method, `simulate`, whose body **is** the flow, returning the
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
  calls `np.random.default_rng(seed)` once in `simulate` and passes
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
  (cell at line 131), `notebooks/02_experiments.py` (data cell), and, found
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
class SimulationResult:                      # the old name, new fields
    """The modeling table of one dataset and the latent effects that generated it."""
    dataset: pd.DataFrame                       # was final_df; one row per building
    building_effects: pd.DataFrame              # index building_id; columns total_count, cohort
    neighborhood_effects: pd.DataFrame          # index neighborhood_id; column total_count
    # User's calls (Stop 2 review): no apartment or stage tables (predictions are
    # building-level); effects as one wide frame per owner level (lme4's ranef).

def aggregate_to_buildings(apartments_df, buildings_df, neighborhoods_df) -> pd.DataFrame   # was _to_building_level, public
    # sums per building reindexed with fill_value=0; two merges in one chain
    # (neighborhoods how="left", then the sums validate="one_to_one"); then the column
    # order taken from the frames (frozen by the EDA baseline); no checks: the
    # identities are the components' and are tested there (user's call)

# student_simulator/outcomes.py: COHORT_COLUMNS = ("n_kindergarten", "n_elementary", "n_highschool")

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

    def simulate(self, rng: np.random.Generator | None = None) -> SimulationResult:
        generator = np.random.default_rng(self.seed) if rng is None else rng
        # 1. Context: one row per neighborhood, then buildings inside them, then
        #    one row per apartment with its room count.
        neighborhoods_df = self.neighborhood_simulator.sample_neighborhoods(self.n_neighborhoods, generator)
        buildings_df = self.building_simulator.sample_buildings(neighborhoods_df, generator)
        apartments_df = self.apartment_simulator.expand_apartments(buildings_df, generator)
        # 2. Outcomes per apartment: the total first, then its split into cohorts.
        apartment_context = ApartmentSimulator.join_context(apartments_df, buildings_df, neighborhoods_df)
        apartments_with_totals, total_effects = self.total_children_simulator.sample_total_children(apartment_context, generator)
        apartments_with_cohorts, cohort_effects = self.cohort_composition_simulator.sample_cohorts(apartments_with_totals, generator)
        # 3. The modeling table: apartments summed into their buildings.
        return SimulationResult(
            dataset=aggregate_to_buildings(apartments_with_cohorts, buildings_df, neighborhoods_df),
            building_effects=pd.DataFrame({"total_count": total_effects["building_total_count_effect"],
                                           "cohort": cohort_effects["building_cohort_effect"]}),
            neighborhood_effects=total_effects["neighborhood_total_count_effect"].to_frame("total_count"),
        )
```

Notes for the implementer:
- `sample_total_children` returns the context **with** `n_children_total`
  (it does today), so the cohort stage takes that frame directly: the second
  `with_context` join and the two `copy(deep=True)`/column-copy dances in
  `run` disappear. Tests of the apartment-level stages take their rows from
  the test helper `tests/unit/helpers.py::simulate_stages`, which runs the
  stages in `simulate`'s order.
- `last_result`, `_copy_result`, `run` and `latent_effects()` are deleted.
  No defensive copies: the result is a new object per call; the user's rule
  is "validate only what would otherwise pass silently".
- Each stage's docstring says what it draws and from which columns, in a
  few lines; the numbered comments in `simulate` are the notes the
  user asked for. The existing per-draw comments in the components stay.
- `n_neighborhoods` and `seed` are validated by nothing new: `default_rng`
  rejects a negative seed; `sample_neighborhoods` already raises on `n <= 0`
  (kept: with `size=0` numpy returns empty arrays silently).
- `__init__.py`: export `SimulationResult`, `LatentEffects` (the
  components' return type), `aggregate_to_buildings`, `COHORT_COLUMNS`; drop
  `SimulationResult`.

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

### [x] Stop 2. The orchestrator (files: pipeline.py, outcomes.py `COHORT_COLUMNS`, apartment.py `join_context`, __init__.py; tests/unit/test_{pipeline,rng_convention,apartment_stage,total_children_stage,cohort_aggregation_stage,fitted_features,modeling_data,state_bundles,hyperparameter_tuning_integration,inspection_permutation,inspection_shap}.py, tests/characterization/{test_current_behavior,eda_baseline,test_experiments_notebook}.py, tests/validation/{helpers,test_recovery,test_calibration}.py, notebooks/{01_eda,02_experiments}.py one line each)

Done 2026-10-14, uncommitted; revised after the user's review of the first
version. The user's questions, the evidence and the decisions:
- *Is the result object necessary? Why apartments in it?* Predictions are
  building-level; only the simulator's own validation tests read apartment
  rows, and they test the total-children stage, which they now run directly
  (`simulate_stages`). → `SimulationResult(dataset, building_effects,
  neighborhood_effects)`.
- *Latent-effects format:* the owners differ (251 buildings, 60
  neighborhoods). lme4's `ranef()` gives one wide frame per grouping level
  (joins by id); broom.mixed's `tidy()` one long frame (plots). The
  consumers join, so: wide, one frame per level. The components keep
  returning `dict[str, pd.Series]` (`LatentEffects`).
- *Column constants / preprocessing:* not to preprocessing (the simulator
  never imports `age_group_prediction`; `preprocessing.py` takes column
  names as arguments). Four of five deleted; `COHORT_COLUMNS` stays in
  outcomes.py, where the columns are written. `join_context` joins all
  neighborhood columns.
- *`aggregate_to_buildings`:* `apartment_rows` and the fill-and-cast step
  are gone (`reindex(..., fill_value=0)` keeps int64 zeros for a building
  without apartments); one merge plus an explicit column order replaces the
  double join (the order is frozen by the EDA baseline fixture; probed); no
  checks: each identity is pinned by a component test (test_building.py:69,
  `test_expand_apartments_matches_building_room_counts`,
  `test_cohort_counts_nonnegative_and_sum_to_total_per_apartment`).
- `simulate(rng=None)`: a passed generator replaces the seed.
- Naming pass (user's call): the stage frames carry a `_df` suffix across the
  simulator (`neighborhoods_df`, `buildings_df`, `apartments_df`: component
  parameters, `aggregate_to_buildings`, `simulate` locals, the
  `Stages` helper); the modeling table is `population.dataset`; the
  aggregation is two merges in one chain, then the column order. Mutation j
  (column order reversed) is caught by the canonical-columns test.
- Proof: the default `dataset` hashes `f385be22b88aebf7` (251 rows); the
  three effects equal the previous Series; the calibration and recovery
  numbers equal Stop 1's (mean 0.600120, zero share 0.609331, oracle max
  error 0.150019).
- `test_pipeline.py`: 7 tests (canonical columns, effect frames per owner,
  seed reproducibility, a passed generator, `from_config` = explicit
  construction, one shared generator, a building without apartments).
  `test_apartments_exactly_expand_wide_building_room_counts` was deleted as
  a duplicate of the component test.
- Mutations, all caught: e (second generator), f (`n_neighborhoods - 1` in
  `from_config`), g (`cohort` column dropped), h (passed `rng` ignored),
  i (no zero fill).

### [x] Stop 3. The experiment notebook's white box (files: notebooks/02_experiments.py data cell)

Done 2026-10-14, uncommitted. The data cell builds the simulator from its
five components (each with `config.simulation` first; `n_neighborhoods` and
`seed` from the overridden `config.simulation`), then
`simulation_res = simulator.simulate()` and `df = simulation_res.dataset`, then a named
`share_transformer = ShareTransformer(...)` and
`df = share_transformer.fit_transform(df)` (user's calls: `df`, not
`population`; a named transformer in separate steps). `table` became `df` in
the cell and its return. The overrides stay on `config`, so the logged
`config_params` match the data. `data_hash` stays (user's call): it is the
MLflow provenance of every run (E9) and the notebook test's proof that the
overrides and the seed reach the simulator.
- Proof: the script still prints `data rows=1529 neighborhoods=150
  data_hash=c92d4213ac419d96…`; the test's independent `from_config`
  recomputation gives the same hash.
- Mutation: `n_neighborhoods=60` in the constructor → the override test
  fails (604 rows, another hash).
- `marimo check` on both notebooks: identical to HEAD (one pre-existing
  empty cell in `01_eda.py`). `01_eda.py` needed no change (Stop 2).
- Follow-up (user's calls, same stop): the notebook is now
  `notebooks/02_experiments.py`; `create_dataset` is `simulate()` and
  `SimulatedPopulation` is `SimulationResult` (fields unchanged); the
  notebook keeps `simulation_res` and returns it beside `df`; the first cell
  puts `src` on `sys.path` (Python 3.13 on macOS skips the venv's hidden
  editable-install `.pth`, so the VS Code kernel could not import the
  project).
- Second follow-up (user's calls): `simulate().dataset` no longer has
  `n_children_total`. It is the sum of the three cohorts, and as a column it
  leaked into the notebook's `X`. The EDA notebook, its frozen fixture's
  builder and the old stack's three test fixtures add it back right after
  `simulate()`; it was the last column, so the EDA fixture is unchanged.
  The notebook keeps `ExposureTransformer` (the one check of a zero or NaN
  exposure) and its two printed lines, which the notebook test reads. The
  notebook's data hash is now `e1dd694935d0…` (one column fewer); the test's
  recomputation matches it.

### Stop 4. Docs, run as written (files: README.md:65 area, docs/SPLITTING.md §3, docs/HYPERPARAMETER_TUNING.md §2, docs/MODEL_INSPECTION.md §2, docs/BAYESIAN_CONDITIONAL_MODEL.md:673, docs/MODULE_REFERENCE.md simulator rows, docs/EXPERIMENT_NOTEBOOK_PLAN.md §2 "Synthetic data", docs/README.md if it names `run`)

- Each code block: `StudentPopulationSimulator(config).run()` →
  `StudentPopulationSimulator.from_config(config).simulate().dataset`.
  Re-run each block as written (the tuning example 9 s, the inspection
  example 44 s, the split block seconds) and confirm the printed numbers
  are unchanged; if any differ, stop and report (it means the draw order
  changed).
  *Done for `docs/MODEL_INSPECTION.md` §2 in sub-task 1.2 of
  [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)
  (2026-10-30): every block reran with the same numbers.*
- `MODULE_REFERENCE.md`: the `pipeline.py` row describes `SimulationResult`,
  `aggregate_to_buildings`, `from_config`, `simulate`.
- A short section in whichever doc describes the simulator (check
  `docs/README.md` for the simulator's guide; if none, `MODULE_REFERENCE.md`
  is enough): the flow in five lines, what the two effect frames hold and why
  it is returned rather than stored.
- Historical docs (GATE_*, archive/) untouched.

### Cleanup stop (after Stop 4; user's call)

- Delete dead code with its tests and exports: `positive_part`,
  `positive_part_interaction`, `normalize_positive_weights` (features.py; no
  source caller), `ApartmentSimulator.to_cohort_long` (only its own tests),
  `config.ApartmentSettings` (unused).
- Then a format-only commit: `ruff format` and `ruff check --fix` on
  `src/student_simulator` and its tests (drift and import-order errors that
  predate the refactor).

## Verification (end to end)

- `uv run pytest -m "not slow"`: all green; report counts against the
  baseline 1296 / 1 / 1.
- `uv run pytest tests/validation/test_recovery.py -q`: the oracle fit with
  the returned effects as offset still recovers the coefficients.
- `uv run --env-file .env notebooks/02_experiments.py` prints
  `data rows=1529 neighborhoods=150 data_hash=e1dd694935d0...` (the population
  of Stop 3, after the dataset dropped `n_children_total`).
- `grep -rn 'last_result\|final_df\|SimulationResult\|with_context\|\.run(' src tests notebooks docs/*.md README.md`
  shows only historical docs.

## Suggested commits (one per stop)

```
git add src/student_simulator tests/unit tests/validation/helpers.py docs/SIMULATOR_WHITE_BOX_PLAN.md
git commit -m "refactor(simulator): name the sampling methods and return the latent effects"
git add src/student_simulator tests notebooks docs/SIMULATOR_WHITE_BOX_PLAN.md
git commit -m "refactor(simulator): StudentPopulationSimulator takes its components; simulate returns the dataset and the latent effects per owner"
git add notebooks tests/characterization docs/SIMULATOR_WHITE_BOX_PLAN.md
git commit -m "feat: the experiment notebook builds the simulator from its components"
git add README.md docs
git commit -m "docs: simulator examples use from_config and simulate"
```

---

## Appendix: PR D stops D1c–D1e (unchanged spec, resumed after this refactor)

See `docs/EXPERIMENT_NOTEBOOK_PLAN.md` §9 for the done-when of each. Cell
code and tests:

```python
# C5 split (section "3) Train, validation and test split"): SPLITTING.md §3 with the settings
splitter = Splitter(split_method)
split_groups = None if split_method == "random" else groups
train_val_index, test_index = splitter.train_test_indices(df, split_groups, test_size=test_size, random_state=seed)
groups_train_val = None if split_groups is None else take_rows(split_groups, train_val_index)
train_sub, val_sub = splitter.train_test_indices(take_rows(df, train_val_index), groups_train_val,
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
