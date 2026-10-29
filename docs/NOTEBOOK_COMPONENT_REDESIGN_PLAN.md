# Notebook section 5: search spaces return objects, components hold the transformer, named builders

**Status (2026-10-25):** approved by the user (plan reviewed in chat; the four design questions
settled, see §0). S1–S5 done 2026-10-25: `FeatureTransformer.to_dict()` (`model_dump(mode="json")`,
so tuples become lists; 25 transformer tests, mypy clean); the four cells of §1.1–§1.3 and the
import cell; the 24 run cells, the Model 1 table, the inspection rebuilds, the header; the test
of both flag values (7/7, in-process `app.run()` works under pytest); D23 in the tuning plan, the
handoff's state and style bullet. Checks: A (8 ruff findings, the two format hunks, marimo's two
known issues), B, C (59 s at 3 trials under `PYTHONWARNINGS=error`, the 24 baselines reproduced,
every component record JSON), the shared probe (13 parameters, ten `shared__*`, a 3-trial study
2.1812 in 6 s). Notebook md5 `7477a8c9819c29d54f1747b0ef4c6b60` after S5. Then the user decided D23.9–D23.11
(§6) and edited the notebook; the session was interrupted before applying them. **S6 done
2026-10-26** (§6): the names settled, D23.9 and D23.10 applied, HEAD's returns restored; Check C 56 s,
the 24 baselines reproduced; notebook md5 `df5d373e868e79a9c7323dfe61ffb829`. Awaiting the user's
commit with T3. **D25 (2026-10-29)** then revised D24.3 and D24.5 at the user's review: the search spaces read nothing back, `estimator_params`, `from_params` per component type, the one-loop objectives (NOTEBOOK_TUNING_PLAN.md §0 D25; this file's code blocks are history). T4a, T4b and T5 done 2026-10-29 (T4b.2b D26); **next:** [EXPERIMENT_NOTEBOOK_HANDOFF.md](EXPERIMENT_NOTEBOOK_HANDOFF.md) §6. The draft PR for the branch could not be
opened from the session (permission); the command is in §7.

Companion of [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) (its §1 rules and checks A/B/C
apply unchanged) and [EXPERIMENT_NOTEBOOK_HANDOFF.md](EXPERIMENT_NOTEBOOK_HANDOFF.md) §4. This
redesign precedes T4a; T4b.1–T4b.2 move the code it produces into the package.

## 0. Context and decisions

`notebooks/02_experiments.py` section 5 tunes Model 1 (`IndependentCohortModels`) and Model 2
(`TotalTimesProbabilityModel`) with Optuna. Today a search space draws a feature transformer
*name*, the component stores that name, and a builder resolves it through
`feature_transformer_dict`; two generic helpers (`suggest_count_component`, `make_count_model`)
sit between the search spaces and the models. The user asked (2026-10-25) for:

- the search space to return the chosen object (the `FeatureTransformer`), with explicit public
  `suggest_<family>_hyperparams` methods;
- the component to hold the `FeatureTransformer` itself, serializable through its plans, so the
  component choice is kept apart from the tuned hyperparameters;
- the Model 1 / Model 2 functions named after the models, with `CountModel` and
  `CohortProbabilityModel` constructed explicitly in each build, no intermediate helpers;
- a `shared_hyperparams` flag for Model 1.

**Decisions (the user, 2026-10-25):**

- D23.1 **Sharing is a boolean and partially shared:** `shared_hyperparams=True` draws one set of
  the estimator's hyperparameters under the prefix `shared__` and copies it into every cohort's
  component; each cohort still draws its own feature transformer. Parameters a trial draws, three
  cohorts: LightGBM 33 → 13, GLMs 6–9 → 4–5; a completely shared design would save two more
  categoricals only and lose the per-cohort feature choice (daycares matter for kindergarten, not
  high school). TPE samples each parameter independently, so the saving is in evidence per
  parameter, not in feasibility.
- D23.2 **Argument only, both values tested:** the 8 Model 1 run cells pass
  `shared_hyperparams=False` explicitly and log it; no new run cells (they would double Model 1's
  click). A test checks both values without fitting.
- D23.3 **`FeatureTransformer.to_dict()` lives in `src`** (`feature_engineering/transformer.py`),
  with a unit test; the notebook's `Component.to_dict()` calls it.
- D23.4 **Only the suggest/build functions are renamed.** Run records (`model1_*`), `MODEL1_RUNS`,
  `model1_df`, `best_model1`, the buttons, captions and the docs keep "Model 1" / "Model 2".
- D23.5 **The `Literal` checks in both `__post_init__`s go.** They resolved the subclass's
  `family` annotation to its `Literal` (`get_type_hints` + `get_args`) and rejected a value
  outside it, since dataclasses do not validate types. In the search space the next line
  (`family not in FEATURE_TRANSFORMER_NAMES`, whose keys are the `Literal`'s values) already
  rejects it; in the component `make_estimator`'s raising `else` does, and every flow builds right
  after constructing. `get_args`, `get_type_hints` leave the import cell.
- D23.6 **`seed` → `random_state`, default `None`,** in `make_estimator` and the builders:
  scikit-learn's term and the repo's (`Splitter`, `BootstrapEvaluator`, `PermutationImportance`,
  `ShapExplainer`); `None` is LightGBM's and scikit-learn's own default (D22's precedent). The
  notebook's experiment setting stays `seed`, passed as `random_state=seed`.
- D23.7 **The registry is injected:** `ComponentSearchSpace` holds `feature_transformer_dict`
  as a field, so the class reads no notebook global and can move to the package (D21: no
  transformer registry in the package). `FEATURE_TRANSFORMER_NAMES` stays the ClassVar the
  tables read. A candidate name missing from the registry is rejected at construction: it would
  otherwise raise `KeyError` in the trial that draws it, minutes into a study, outside
  `catch=(RuntimeError,)`.
- D23.8 **Naming, open for the user's review:** the explicit methods use the user's spelling
  `suggest_<family>_hyperparams`; the dispatcher keeps `suggest_hyperparameters` (the field and
  the record key are `hyperparameters`).

**Why the transformer serializes.** A `FeatureTransformer` is its declaration: `column_plans`
(pairs of a pydantic `Plan` and its columns), `interactions` (pydantic `Interaction`) and
`remainder`. `Plan.model_dump()` gives `{"name": ..., "transforms": [{"kind": "standardize"},
{"kind": "domain_scale", "scale": 0.1}, ...]}`, plain JSON. The registry name survives in
`best_params` and the trials table (`<component>__feature_transformer`). `dataclasses.asdict`
alone would deep-copy the object into the record, not JSON; hence `Component.to_dict()`.

**Draw order.** Keyword arguments evaluate left to right, so `hyperparameters=` stays before
`feature_transformer=` in every constructor call. With `shared_hyperparams=False` the order is
today's and Check C must reproduce all 24 baselines: Model 1 (exposure on / off) `lgbm_poisson`
2.190 / 2.281, `lgbm_gaussian` 2.191 / 2.315, `poisson_glm` 2.570 / 2.878, `nb2` 2.283 / 2.890;
Model 2's 16 in NOTEBOOK_TUNING_PLAN.md T2.1.

## 1. The design (notebook cells, section 5)

### 1.1 Search spaces cell: `ComponentSearchSpace`

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class ComponentSearchSpace(abc.ABC):
    """The search of one component's choices, given its name and family; returns the chosen objects."""

    name: str
    family: str
    # The registry the drawn name resolves to; injected, so the class reads no notebook global.
    feature_transformer_dict: Mapping[str, FeatureTransformer]
    FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]]

    def __post_init__(self):
        if self.family not in self.FEATURE_TRANSFORMER_NAMES:
            raise ValueError(f"no feature transformers for family {self.family!r}")
        # A candidate missing from the registry would raise KeyError in the trial that
        # draws it, minutes into a study, and `catch=(RuntimeError,)` would not hold it.
        missing = [
            name
            for name in self.FEATURE_TRANSFORMER_NAMES[self.family]
            if name not in self.feature_transformer_dict
        ]
        if missing:
            raise ValueError(f"feature transformers {missing} are not in feature_transformer_dict")

    @property
    def prefix(self):
        """The trial's parameter names are ``<name>__<parameter>``."""
        return f"{self.name}__"

    def suggest_feature_transformer(self, trial):
        """The chosen transformer; the trial records its name (Optuna's choices must be primitives)."""
        name = trial.suggest_categorical(
            f"{self.prefix}feature_transformer",
            list(self.FEATURE_TRANSFORMER_NAMES[self.family]),
        )
        return self.feature_transformer_dict[name]

    @abc.abstractmethod
    def suggest_hyperparameters(self, trial):
        """The family's params record: the family's ``suggest_<family>_hyperparams``."""

    def suggest_lgbm_hyperparams(self, trial):
        ...  # today's _suggest_lgbm, unchanged bounds
```

- `CountSearchSpace`: `FEATURE_TRANSFORMER_NAMES` unchanged; `suggest_poisson_glm_hyperparams`,
  `suggest_nb2_hyperparams` (today's branches with their comments); `suggest_hyperparameters` is
  the `if/elif/raise` dispatch to the three methods.
- `ProbabilitySearchSpace`: `suggest_logistic_hyperparams`; `suggest_hyperparameters` dispatches;
  `suggest_calibration_method` and `CALIBRATION_METHODS` unchanged.
- The cell's inputs gain `FeatureTransformer` and `Mapping`; it loses `get_args`,
  `get_type_hints`.

### 1.2 Components cell

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class Component(abc.ABC):
    """A named part of a composite: its family, feature transformer and hyperparameters."""

    name: str
    family: str
    # CountModel and CohortProbabilityModel clone it at fit, so the shared declaration is safe.
    feature_transformer: FeatureTransformer

    @abc.abstractmethod
    def make_estimator(self, *, random_state=None):
        """The unfitted estimator: the family's fixed settings and ``hyperparameters``."""

    def to_dict(self):
        """The record as plain data: the transformer by its declaration (plans, interactions)."""
        record = dataclasses.asdict(self)
        record["feature_transformer"] = self.feature_transformer.to_dict()
        return record
```

No `__post_init__`. `CountComponent` / `ProbabilityComponent`: `make_estimator(self, *,
random_state=None)`, `random_state=random_state` in the two LightGBM constructors; otherwise
unchanged. The cell's inputs gain `FeatureTransformer`, lose `get_args`, `get_type_hints`.

### 1.3 Builders cell

Removed: `suggest_count_component`, `suggest_probability_component`, `make_count_model`,
`suggest_model1(_components)`, `build_model1`, `suggest_model2(_components)`, `build_model2`.

```python
def suggest_independent_cohorts_components(
    trial, *, family, use_exposure, cohorts, feature_transformer_dict, shared_hyperparams
):
    """Model 1's components ``trial`` draws: one per cohort, all of ``family``.

    ``shared_hyperparams``: one draw of the hyperparameters, under ``shared__``, in every
    component; each cohort still draws its own transformer.
    """
    # The draws, in order: the shared hyperparameters if any, then per cohort its own
    # hyperparameters (when not shared) and its transformer. A seeded study's result
    # depends on it; with shared_hyperparams=False it is the order before D23.
    shared_hyperparameters = (
        CountSearchSpace(
            name="shared", family=family, feature_transformer_dict=feature_transformer_dict
        ).suggest_hyperparameters(trial)
        if shared_hyperparams
        else None
    )
    components = []
    for cohort in cohorts:
        space = CountSearchSpace(
            name=cohort, family=family, feature_transformer_dict=feature_transformer_dict
        )
        hyperparameters = (
            shared_hyperparameters if shared_hyperparams else space.suggest_hyperparameters(trial)
        )
        components.append(
            CountComponent(
                name=cohort,
                family=family,
                use_exposure=use_exposure,
                hyperparameters=hyperparameters,
                feature_transformer=space.suggest_feature_transformer(trial),
            )
        )
    return tuple(components)


def build_independent_cohorts(components, *, y_min, random_state):
    """Model 1: one CountModel per component, keyed by the component's name, a cohort."""
    return IndependentCohortModels(
        {
            component.name: CountModel(
                estimator=component.make_estimator(random_state=random_state),
                use_exposure=component.use_exposure,
                feature_transformer=component.feature_transformer,
                y_min=y_min,
            )
            for component in components
        }
    )


def suggest_independent_cohorts(
    trial, *, family, use_exposure, cohorts, feature_transformer_dict, y_min, random_state,
    shared_hyperparams,
):
    """Model 1 of the components ``trial`` draws for one family and exposure."""
    components = suggest_independent_cohorts_components(
        trial, family=family, use_exposure=use_exposure, cohorts=cohorts,
        feature_transformer_dict=feature_transformer_dict, shared_hyperparams=shared_hyperparams,
    )
    return build_independent_cohorts(components, y_min=y_min, random_state=random_state)


def suggest_total_times_probability_components(
    trial, *, total_family, probability_family, use_exposure, feature_transformer_dict
):
    """Model 2's total and probability components ``trial`` draws, the total first."""
    # The draws, in order: the total's hyperparameters and transformer, then the
    # probabilities' hyperparameters, calibration and transformer.
    total_space = CountSearchSpace(
        name="total", family=total_family, feature_transformer_dict=feature_transformer_dict
    )
    total = CountComponent(
        name=total_space.name,
        family=total_family,
        use_exposure=use_exposure,
        hyperparameters=total_space.suggest_hyperparameters(trial),
        feature_transformer=total_space.suggest_feature_transformer(trial),
    )
    probability_space = ProbabilitySearchSpace(
        name="probability",
        family=probability_family,
        feature_transformer_dict=feature_transformer_dict,
    )
    probability = ProbabilityComponent(
        name=probability_space.name,
        family=probability_family,
        hyperparameters=probability_space.suggest_hyperparameters(trial),
        calibration_method=probability_space.suggest_calibration_method(trial),
        feature_transformer=probability_space.suggest_feature_transformer(trial),
    )
    return total, probability


def build_total_times_probability(total, probability, *, y_min, random_state):
    """Model 2: the total's CountModel and the cohort probabilities' classifier."""
    return TotalTimesProbabilityModel(
        total_model=CountModel(
            estimator=total.make_estimator(random_state=random_state),
            use_exposure=total.use_exposure,
            feature_transformer=total.feature_transformer,
            y_min=y_min,
        ),
        probability_model=CohortProbabilityModel(
            estimator=probability.make_estimator(random_state=random_state),
            calibration_method=probability.calibration_method,
            feature_transformer=probability.feature_transformer,
        ),
    )


def suggest_total_times_probability(
    trial, *, total_family, probability_family, use_exposure, feature_transformer_dict, y_min,
    random_state,
):
    """Model 2 of the components ``trial`` draws for one family pair and exposure."""
    total, probability = suggest_total_times_probability_components(...)
    return build_total_times_probability(
        total, probability, y_min=y_min, random_state=random_state
    )
```

### 1.4 The 24 run cells (uniform edits)

Model 1 (8 cells, the first at line 1435):
- after `_family, _use_exposure = ...` add `_shared_hyperparams = False`;
- `partial(suggest_independent_cohorts, ..., shared_hyperparams=_shared_hyperparams,
  random_state=seed)`;
- `_best = suggest_independent_cohorts_components(FixedTrial(_study.best_params), family=_family,
  use_exposure=_use_exposure, cohorts=COHORTS, feature_transformer_dict=FEATURE_TRANSFORMERS,
  shared_hyperparams=_shared_hyperparams)`;
- `build_independent_cohorts(_best, y_min=PREDICTION_FLOOR, random_state=seed)`;
- the record: `"shared_hyperparams": _shared_hyperparams` after `"use_exposure"`, and
  `"components": [component.to_dict() for component in _best]`;
- the cell's inputs: the new function names; `dataclasses` dropped.

Model 2 (16 cells, the first at line 2210): the same without the shared flag;
`build_total_times_probability(*_best, y_min=PREDICTION_FLOOR, random_state=seed)`.

Format each edited cell by hand to `ruff format`'s output; `ruff format --diff` must still show
only the two known hunks (lines 158 and 213).

### 1.5 Other cells
- **Model 1 tables cell** (line 2136): `_settings` gains
  `"shared_hyperparams": _run["shared_hyperparams"]`. `best_df` keeps its columns.
- **Inspection rebuilds** (lines 3805, 3835): the new names; the components call gets
  `feature_transformer_dict=FEATURE_TRANSFORMERS` (Model 1 also
  `shared_hyperparams=best_run1["shared_hyperparams"]`); the build gets
  `random_state=best_run1["seed"]`, no dict.
- **Import cell** (line 674): drop `get_args`, `get_type_hints`; add
  `from collections.abc import Mapping`.
- **Section 5 header** (line 619): the draws sentence becomes "the estimator's hyperparameters
  and the feature transformer, drawn by name and returned as the object; they make a component
  ..."; one sentence on `shared_hyperparams` (one draw of the hyperparameters for all cohorts,
  each cohort its own transformer; every run here passes `False`). Nothing else.

## 2. The `src` change: `FeatureTransformer.to_dict()`

`src/age_group_prediction/feature_engineering/transformer.py`:

```python
def to_dict(self) -> dict[str, Any]:
    """The declaration as plain data: each plan's ``model_dump`` with its columns, the interactions', the remainder."""
    return {
        "column_plans": [
            {"plan": plan.model_dump(), "columns": list(columns)}
            for plan, columns in self.column_plans
        ],
        "interactions": [interaction.model_dump() for interaction in self.interactions],
        "remainder": self.remainder,
    }
```

Test in `tests/unit/test_feature_transformer.py`, in its style: a transformer with two plans (the
identity, and `Center` + `DomainScale(scale=0.1)`) and an interaction; `json.dumps` of the result
succeeds; the kinds, the scale, the columns and the interaction's `left`/`right` appear; the
identity plan has `transforms == []`. One line in `docs/MODULE_REFERENCE.md`'s `FeatureTransformer`
entry.

## 3. The test of both `shared_hyperparams` values (D23.2)

In `tests/characterization/test_experiments_notebook.py`: one test that loads the notebook
in-process as Check B does (`importlib` + `nb.app.run()`, fits nothing, about 5 s), then for
`shared_hyperparams in (False, True)`:
- `trial = optuna.create_study(sampler=RandomSampler(seed=0)).ask()`;
- `components = defs["suggest_independent_cohorts_components"](trial, family="lgbm_poisson",
  use_exposure=True, cohorts=defs["COHORTS"], feature_transformer_dict=defs["FEATURE_TRANSFORMERS"],
  shared_hyperparams=shared_hyperparams)`;
- `False`: 33 parameters, every name starts with a cohort's prefix, no `shared__`;
  `True`: 13 parameters, ten `shared__*` and one `<cohort>__feature_transformer` per cohort,
  the three components' `hyperparameters` equal;
- each component's `to_dict()` passes `json.dumps`.

If `app.run()` misbehaves under pytest, fall back to the file's subprocess pattern with a small
driver script printing the same facts. The test is temporary: T4b.1–T4b.2 move the code to the
package with unit tests.

## 4. Sub-tasks

### [x] S1 `FeatureTransformer.to_dict()` (src, test, MODULE_REFERENCE.md)
**Verify:** `uv run pytest tests/unit/test_feature_transformer.py -q`; `uv run mypy` on the
package; the notebook unchanged (md5 `1e3a9eeb6a3d9f807bdf3a8287212158`).

### [x] S2 The three class/function cells (§1.1–§1.3) and the import cell
**Verify:** Check A; Check B with the names of §5 step 3.

### [x] S3 The 24 run cells, the Model 1 table, the inspection rebuilds, the header (§1.4–§1.5)
**Verify:** Check A (`ruff format --diff`: the two known hunks only); Check B; Check C
(§5 step 4) reproducing the 24 baselines; the shared probe (§5 step 5).

### [x] S4 The test of both flag values (§3)
**Verify:** `uv run pytest tests/characterization -q`.

### [x] S5 Docs: D23 and the status line in NOTEBOOK_TUNING_PLAN.md, its T4b.1/T4b.2 module tables
(lines ~232–235: `ComponentSearchSpace`, the new builder names, the removed helpers), the
handoff's §2 state, md5 and §4 style bullet ("the trial records the name; the component holds the
object"); this file's status line.
**Verify:** `uv run pytest -m "not slow"` once; the file-by-file report and the suggested commit
message in the chat (the user commits; no `Co-Authored-By`).

## 5. Checks

1. Before the first edit: `git status` (T3's uncommitted changes only) and the notebook's md5.
2. Check A: `uv run pytest tests/characterization/test_experiments_notebook.py -q`;
   `ruff check notebooks/02_experiments.py` (8 findings, none new); `uv run marimo check
   notebooks/02_experiments.py` (the two known warnings); `ruff format --diff` (the two hunks).
3. Check B: defined `ComponentSearchSpace`, `CountSearchSpace`, `ProbabilitySearchSpace`,
   `Component`, `CountComponent`, `ProbabilityComponent`, `suggest_independent_cohorts`,
   `suggest_independent_cohorts_components`, `build_independent_cohorts`,
   `suggest_total_times_probability`, `suggest_total_times_probability_components`,
   `build_total_times_probability`; absent `SearchSpace`, `suggest_model1`, `suggest_model2`,
   `build_model1`, `build_model2`, `suggest_count_component`, `suggest_probability_component`,
   `make_count_model`, `get_args`.
4. Check C under `PYTHONWARNINGS=error` at 3 trials, the three buttons patched (inspection at
   `n_shap_rows = 5`, `n_repeats = 2`): the 24 `cv_best` equal the baselines; every record's
   `components` passes `json.dumps`; `shared_hyperparams` in each Model 1 record and in
   `model1_df`.
5. The shared probe (read-only, scratchpad): one 3-trial study of `suggest_independent_cohorts`
   with `shared_hyperparams=True`, `lgbm_poisson`, exposure on; report its best value and that
   `best_params` holds ten `shared__*` and three `__feature_transformer` names.
6. `uv run pytest -m "not slow"`; `uv run mypy` on the package.

## 6. Continuation (interrupted 2026-10-25, after S5; the user then edited the notebook)

**Decided after S5 (the user, 2026-10-25):**
- D23.9 **Candidates injected, no names in the class (option A).** `ComponentSearchSpace`
  takes `feature_transformers: Mapping[str, FeatureTransformer]`, one family's candidates,
  name to object, and draws from its keys; no `FEATURE_TRANSFORMER_NAMES`, no
  `feature_transformer_dict`, no `__post_init__` (an unknown family raises at the first
  `suggest_hyperparameters`). The notebook's section-4 cell keeps the data as
  `COUNT_FEATURE_TRANSFORMERS` and `PROBABILITY_FEATURE_TRANSFORMERS` (family → `{name:
  object}`), replacing `FEATURE_TRANSFORMERS`; the candidate order must stay today's, so Check C
  reproduces the baselines. Rationale: the package holds the mechanism, the notebook the data
  (scikit-learn's `GridSearchCV(param_grid=...)` split); a ClassVar registry would be a notebook
  global read by the class (rejected). The builders take the candidates for the component:
  `suggest_independent_cohorts(_components)(..., feature_transformers=COUNT_FEATURE_TRANSFORMERS[_family])`,
  `suggest_total_times_probability(_components)(..., total_feature_transformers=COUNT_FEATURE_TRANSFORMERS[_total_family],
  probability_feature_transformers=PROBABILITY_FEATURE_TRANSFORMERS[_probability_family])`.
- D23.10 **The shared draw inside the loop, by prefix:** the hyperparameter space is
  `CountSearchSpace(name="shared", ...)` when sharing, else the cohort's space; the first cohort
  draws the ten values and Optuna returns the stored values to the next cohorts (probed:
  `RandomSampler`, `TPESampler` and `FixedTrial` return the stored value and record the
  parameter once). No draw before the loop, no `None`.
- D23.11 **The two arms to compare:** no sharing versus partial sharing (one difference, the big
  search-space effect). Complete sharing needs no run: the partial run's `best_params` shows
  whether the cohorts chose the same transformer.

**The notebook when interrupted** (md5 `1ce2db86efa3ffe83f0eb85f6859c772`): S1–S5 as built
(`feature_transformer_dict`, `FEATURE_TRANSFORMER_NAMES`, the draw before the loop), plus the
user's edits, saved from marimo:
- renames in the builders cell: the argument `shared_hyperparams` → `share_hyperparams`
  (the local became `shared_hyperparams`; `suggest_independent_cohorts` and the 8 run cells
  still pass `shared_hyperparams=` → a `TypeError` at the first trial until the names agree),
  `components` → `cohort_components`, `space` → `count_search_space`, `hyperparameters` →
  `hyperparams`, `name` → `feature_transformer_name`, `total`/`probability` →
  `total_component`/`probability_component` (one typo `total_copmonent`),
  `total_search_space`/`probability_search_space`; `suggest_independent_cohorts` binds `model`
  then returns it; `build_total_times_probability(total_component,
  probability_component=probability_component, ...)`;
- `fit_and_score` moved to a top-level `@app.function` (the 24 run cells no longer list it);
- marimo trimmed the returns (`config_hash`, `Component`, `ComponentSearchSpace`, `cv_score`,
  `fold_score`, `best_df`, `intervals_df`, `importance_df` and three bare `return`s), so ruff
  is at 14 findings (baseline 8) and Check B loses those names; the user previously chose to
  have HEAD's returns restored after seeing the diff (handoff §4).

**[x] S6, done 2026-10-26** (the results follow the steps). **The steps, in order:**
1. Settle the names in the builders cell with the user (`share_hyperparams` vs
   `shared_hyperparams` in the argument and the 8 run cells; fix `total_copmonent`), and
   whether `fit_and_score` stays an `@app.function` (then the test's `tune_model` string rule
   is unaffected; `ruff` must not flag it).
2. Apply D23.9: the section-4 cell (two dicts, drop `FEATURE_TRANSFORMERS` and its return),
   the search-space cell (field `feature_transformers`, no ClassVar names, no `__post_init__`,
   `MappingProxyType` out of the cell and of the import cell if unused elsewhere), the builders
   (D23.10 loop; the new arguments), the 24 run cells (`feature_transformers=COUNT_FEATURE_TRANSFORMERS[_family]`
   / the two Model 2 arguments; cell inputs `COUNT_FEATURE_TRANSFORMERS`,
   `PROBABILITY_FEATURE_TRANSFORMERS` instead of `FEATURE_TRANSFORMERS`), the two tables
   (`list(COUNT_FEATURE_TRANSFORMERS[_run["family"]])` and the Model 2 analogue;
   `ProbabilitySearchSpace.CALIBRATION_METHODS` stays), the two inspection rebuilds, and the
   test (`feature_transformers=notebook_defs["COUNT_FEATURE_TRANSFORMERS"]["lgbm_poisson"]`).
3. Restore the trimmed returns as at HEAD (after showing the diff), or keep the trims if the
   user now prefers them; then Check A (ruff 8 or the agreed count, the two format hunks),
   Check B, Check C (the 24 baselines), the shared probe, `pytest -m "not slow"`.
4. Docs: D23.9–D23.11 are here; update D23 and the T4b table in NOTEBOOK_TUNING_PLAN.md
   ("candidates injected as `feature_transformers`"), the handoff's md5, this status line.

**S6 as built (2026-10-26):**
- **Names (the user, step 1):** `share_hyperparams` everywhere: the argument, `_share_hyperparams`
  in the 8 run cells, the record key, the Model 1 table column, the test. `fit_and_score` stays a
  top-level `@app.function` (in `app.run()`'s defs; the test's `tune_model` rule unaffected).
  `build_total_times_probability(total_component, probability_component, ...)` takes both by
  position. The typo is fixed.
- **D23.9:** section 4 holds `COUNT_FEATURE_TRANSFORMERS` and `PROBABILITY_FEATURE_TRANSFORMERS` as
  plain dicts, today's order. `ComponentSearchSpace.feature_transformers` replaces the registry,
  the ClassVar names and `__post_init__`. `MappingProxyType` left the notebook. The tables read
  `list(COUNT_FEATURE_TRANSFORMERS[...])` and `list(PROBABILITY_FEATURE_TRANSFORMERS[...])`.
- **D23.10:** each cohort's hyperparameters come from the `shared` space when sharing, otherwise
  from its own. Probed: a second suggest under the same name returns the stored value and records
  one parameter (`RandomSampler`, `TPESampler`, `FixedTrial`).
- **Returns:** HEAD's restored (`config_hash, config_params`, `Component`, `ComponentSearchSpace`,
  `cv_score, fold_score`, `best_df`, `intervals_df`, `importance_df`; no bare `return` in the two
  display cells; the importance cell's one-line signature). Measured: the trimmed returns had cost
  ruff findings only; `app.run()`'s defs held the names either way.
- **Checks:**
  - Check A: notebook test 7/7; ruff 8; `ruff format --diff` 158 and 213 only; `marimo check`
    the two known issues.
  - Check B: all names present; the old names and `FEATURE_TRANSFORMER_NAMES` absent.
  - Check C under `PYTHONWARNINGS=error` (55.7 s, no warning): the 24 baselines, every
    `components` record JSON, `share_hyperparams` in every Model 1 record and in `model1_df`.
  - The shared probe: 2.1812, 13 parameters (ten `shared__*`, three `__feature_transformer`),
    3.0 s.
  - `pytest -m "not slow"`: 1359 passed. mypy clean.

**D24 (2026-10-27, the user's notes on the S6 notebook; decided in NOTEBOOK_TUNING_PLAN.md §0 D24):**
the user's hand-written `share_feature_transformer` recorded the shared draws under the first
cohort's name and required a keyword the run cells did not pass; replaced: a shared choice is
drawn once before the loop under `shared__`, each cohort takes it or draws its own (a generator
of per-cohort spaces was tried first and rejected by the user). Both flags became experiment settings (`True` /
`False`, the other arm a click). The best components are read from `best_params` through the
search spaces' `*_from_params` methods (explicit, per family), no `FixedTrial`. `feature_transformers` →
`feature_transformer_dict` (D23.9's naming only). The builders are `build_independent_cohorts_model`
and `build_total_times_probability_model`, the objectives `suggest_*_model`.
`CohortProbabilityModel.replication_method` (renamed) is fixed at `"weighted"` in the builder.
Check B's names: the defined list of §5 plus `independent_cohorts_components_from_params`, `total_times_probability_search_spaces`,
`total_times_probability_components_from_params`, `build_*_model`, `suggest_*_model`; absent
`FixedTrial`, `build_independent_cohorts`, `build_total_times_probability`,
`suggest_independent_cohorts`, `suggest_total_times_probability`.

## 7. The pull request

The branch (42 commits over `main`, pushed to `origin` at `de94ac8`) had no pull request on
2026-10-25. To open the draft PR from the repository root:

```sh
gh pr create --draft --base main --head feat/experiment-notebook \
  --title "feat(notebook): experiment notebook with Optuna tuning, inspection and the component redesign" \
  --body-file docs/NOTEBOOK_COMPONENT_REDESIGN_PLAN.md
```

(or a shorter body naming the three plan documents and the checks of §5).
