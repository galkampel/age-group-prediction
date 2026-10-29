# Re-implement the hyperparameter tuning inside the notebook

**Status (2026-10-21):** approved; T1.1 committed (`0416267`); T1.2 committed (`bee781a`) after the user's review (estimator dataclasses hold hyperparameters only; LightGBM's loss is a family); T1.3 committed (`7ae6314`) after three user reviews (components are a study's fixed settings, a `Literal` family and `use_exposure`; the family → transformer-names mapping is an argument of `suggest_feature_transformer`; no dataclass builds a model; the exposure a study setting, both arms; feature variants renamed feature transformers; D2, D3, D6, T1.4–T2.2 rewritten); T1.4 done 2026-10-19, then reworked at two rounds of the user's review on 2026-10-20 (D8: a named component per part of the composite under an abstract `Component`; the family checked from its `Literal`; each subclass holds its families' transformer names; `make_estimator(hyperparameters, *, seed)`; the T1.3 cells changed too; T1.4–T2.2 rewritten), committed (`8cc231d`); T1.5 done 2026-10-21 (`cv_score`, `tune_model`; at the user's review the fold variables named and an aggregator added, D9: `src/age_group_prediction/aggregation.py` and its tests, default `Mean`), committed (`a5ae06b`); T1.6 done 2026-10-21 (Model 1's button and run cell; Check C 14.7 s at 3 trials), then the redesign of D10 at its review, revised by D11 and D12 at the next reviews (search spaces with explicit suggest methods, the builder assembles components, the transformer by name; a study is a family combination and exposure; `feature_transformer_dict`; Check C reproduces T1.6's best values), committed (`6316d91`); T1.7 committed (`59acae6`); its 200-trial run took 380 s with no failed trial (D15); T1.8 done 2026-10-21 (independent review: no correctness bug, nine findings reproduced and settled, D15), committed (`9eca96e`); notebook md5 `9d49fe78f99d7ff4d96e17742ea7d277`; handoff rewritten 2026-10-21 for a new session; T2.1 done 2026-10-21 (Model 2's button and 16 run cells; no `MODEL2_TOTALS`, the run cells present are the runs tuned; Check C 53.3 s, Model 2's baseline recorded), awaiting the user's commit; notebook md5 `1405425ae7bf3da901cf0eba8860bf71`; T2.2 done 2026-10-21 (Model 2's tables, display and best-per-model cells; D16), 200-trial run 1827 s (Model 1 366 s, Model 2 1447 s, no failed trial); T2.1–T2.2 committed (`a2b8693`); T2.3 done 2026-10-21 (independent review: no correctness bug; findings F1–F8 settled), awaiting the user's commit; notebook md5 `eec46ac5cbec902ffdd98cce6f4c9cbd`; T2.4 (pruning and parallelization check) added at the user's request; D17: T4 replaces the package (T4a delete, T4b rebuild from the notebook). D18: T2.5 (the bootstrap interval and feature importance of the best runs) added. The `n_startup_trials` decision postponed by the user until after T2.4, with `n_trials`; all 24 runs kept (D6 stands). T2.3 committed (`7580304`), handoff committed (`1811551`). T2.4 done 2026-10-22 (measured: no pruning or parallelization place beats today's design, two are unsafe; D19: today's design kept, `catch` unchanged; docs only, notebook md5 unchanged `eec46ac5cbec902ffdd98cce6f4c9cbd`), committed (`9075dd7`). The budget sub-task done 2026-10-22 (D20: `n_trials = 400`, `n_startup_trials = 20`, a new setting; `tune_model` keyword; the 24 run cells; the notebook test; the 400-trial click 3708 s, no failed trial; Check C baselines unchanged; D21 the T4 design and the inspection recipe's home, planned ahead), notebook md5 `d919a7838019a1494d92768f0bafd483`, committed (`10d6e83`); handoff rewritten for a new session. T2.5 done 2026-10-22 (D22: `BootstrapEvaluator` holds the resampling design and `evaluate(..., metrics=[...])` returns a frame; the evaluation and the importances are two separate components, no `inspect_model`; the seven metrics, the three settings, `best_params` in the 24 records, a button, three cells per model, two tables; exact SHAP 100 × 50 on all cores; Check C 66.8 s at 3 trials, the 24 baselines unchanged), notebook md5 `1e3a9eeb6a3d9f807bdf3a8287212158`, committed (`de94ac8`). T3 done 2026-10-22 (the five constants left `scoring.py`; `Metric` and the two functions stay, the user's scope; eight test sites and three doc blocks build their metric inline; mypy clean, the non-slow suite green, the notebook unchanged), awaiting the user's commit. The component redesign done 2026-10-25 (D23, [NOTEBOOK_COMPONENT_REDESIGN_PLAN.md](NOTEBOOK_COMPONENT_REDESIGN_PLAN.md): `ComponentSearchSpace` returns objects, components hold the transformer, `FeatureTransformer.to_dict()`, the builders named after the models, `shared_hyperparams`; Check C 59 s at 3 trials, the 24 baselines reproduced; the notebook test 7/7), then S6 done 2026-10-26 (D23.9–D23.11: the candidates injected as `feature_transformers`, the shared draw by prefix inside the loop, `share_hyperparams`; Check C 56 s at 3 trials, the 24 baselines reproduced; the shared probe 2.1812), notebook md5 `df5d373e868e79a9c7323dfe61ffb829`; then D24 done 2026-10-27 (the user's notes: `replication_method` fixed at `"weighted"`, `share_feature_transformer` with the sharing rule written once, the best components read from `best_params` with no `FixedTrial`, the two flags as settings, `feature_transformer_dict`, the builders `build_*_model`; the notebook test 11/11, Check C the 24 baselines at both flags `False` and the new arms recorded), notebook md5 `c42c578ec48f5da3d6e1ae9ff1ba5e1f`, awaiting the user's commit with T3 and D23; then D25 done 2026-10-29 (the user's review of D24: no readers on the search spaces, `estimator_params`, `from_params` per component type, the one-loop objectives, `_best_components` in the run cells; Check C the 24 baselines, the shared probe 2.1812, the notebook test 12/12, the non-slow suite green), notebook md5 `425b6cf3ff396a478ad08756c2211338`, awaiting the user's commit (T3, D23, D24, D25 in one; the docs in a second). Committed as `15932d7` (2026-10-29, one commit). T4a–T5 planned 2026-10-29 in [NOTEBOOK_T4_HANDOFF.md](NOTEBOOK_T4_HANDOFF.md) (T4a as three sub-tasks: T4a.1 the code, T4a.2 the current guides, T4a.3 the finished plans; the verified state, the checks and the 24 baselines). T4a done 2026-10-29 on `refactor/hyperparameter-tuning-package` (draft PR #17): T4a.1 the code (`5c0908a`), T4a.2 the current guides (`ed1a770`), T4a.3 the finished plans (`6935416`); the non-slow suite 1226 passed, mypy 22 files, the notebook unchanged, Check C the 24 baselines. T4b.1 committed (`2765196`; `params.py`, `components.py`, `search_spaces.py` and 48 tests; mypy 26 files). T4b.2 done 2026-10-29 (`builders.py`, `cross_validation.py`, `aggregation.py` moved into the package and the root module removed; the notebook imports `Mean` from the package; `Aggregator.aggregate` renamed `aggregate_scores` at the user's note; md5 `e8a0942fff75f66e6173d6b47ba162a4`; mypy 28 files; the package's tests 95; Check C the 24 baselines; NOTEBOOK_T4_HANDOFF.md §3), committed (`2bdc808`). T4b.2b done 2026-10-29 (D26, the user's notes on T4b.2: the suggest functions return components, `CrossValidationTuner` with a named tune method per model, `fit_and_score` in `evaluation.py`; the package only, the notebook unchanged; mypy 29 files; the package's tests 98, the non-slow suite 1299 passed; the 24 baselines through the tuner on the notebook's data; NOTEBOOK_T4_HANDOFF.md §3), awaiting the user's commit. Next: T4b.3 → T5. The implementing session updates this line and ticks the sub-tasks.

Written for the model that implements it, one sub-task per approval. Every sub-task has the
code to write, a "done when" list and the commands that verify it. The user reads the diff in
the IDE, runs the checks and commits; the implementer never commits.

## 0. Context and decisions

Section 5 of `notebooks/02_experiments.py` (lines 598–1222 at HEAD `8ee091a`, md5
`55f81ffb7e6dd3f85aed71f1a45118ce`) tunes Model 1 (`IndependentCohortModels`) and Model 2
(`TotalTimesProbabilityModel`) through the package `src/age_group_prediction/hyperparameter_tuning`
(`Parameter` classes, `KeyedParameter`, `CVHyperparameterEvaluator` driving `set_params` paths
such as `cohort_models__<cohort>__…`, `HyperparameterStudy`, `Aggregator`). The user rejected
that package: too much machinery between the notebook and Optuna, and the composites are
assembled by `set_params` strings instead of being built explicitly.

**Decisions (2026-10-18, the user's; do not re-ask):**

- D1. Section 5 is rewritten with plain Optuna and a hand-written cross-validation loop. None of
  the package's classes or functions is imported. The package is deleted afterwards (T4).
- D2. **Frozen dataclasses** (`dataclasses.dataclass(frozen=True)`), JSON-serializable through
  `dataclasses.asdict`, for two things: an estimator's searched hyperparameters
  (`LGBMRegressorParams`, …), and a component's settings fixed for one study
  (`CountComponent(family, use_exposure)`, `ProbabilityComponent(family)`). Every field is a
  JSON scalar; a feature transformer is stored as its key in `FEATURE_TRANSFORMERS`, and the
  family is a `Literal` (`CountFamily`, `ProbabilityFamily`). A study's record is
  `{"settings": asdict(component), "params": study.best_params}`; `best_params` holds the
  chosen transformer's name. (The comparison with
  TypedDict: §6.) Since D8 a component also holds its `name`, and Model 1's
  record holds the list of its components' settings.
- D3. An estimator dataclass has a classmethod `suggest(trial, prefix, …)` calling
  `trial.suggest_*` inline. A component looks up its params class from its family, and
  samples with `suggest_params` / `suggest_feature_transformer(trial, prefix,
  family_transformers)` (and
  `suggest_calibration_method`). **No dataclass builds a model** (the user, 2026-10-18, at
  the reviews of T1.2 and T1.3). The objective builds it: `suggest_model1` / `suggest_model2`
  sample from the trial and call the constructors explicitly (estimator, then
  `CountModel` / `CohortProbabilityModel`, then the composite; no `set_params`). The best run's
  refit calls the same function on `FixedTrial(study.best_params)`. (Since D8: a component
  builds its estimator, never the model; the methods take no prefix, the component's name
  gives it; a subclass holds its families' transformer names, and the family → names
  argument is gone.)
  **Package-ready (the user, 2026-10-18, at T1.3's second and third reviews):** a dataclass
  reads no notebook global. A setting fixed for the study (the family, the exposure) is a
  field. The family's candidate feature transformers come in as an argument of
  `suggest_feature_transformer`: a mapping from each family to the names that suit it
  (`COUNT_FAMILY_TRANSFORMERS`, `PROBABILITY_FAMILY_TRANSFORMERS`), and the component reads
  its own entry. The reason is the user's: one place pairs a family with its transformers, so
  a component cannot be given another family's (a field of names, the second review's
  version, could). Whether a transformer suits a family is the experiment's knowledge, which
  no code can check; the mapping is where it is reviewed. The builder resolves a name to its
  object (`feature_transformers[name]`). An unknown family raises at construction, a family
  missing from the mapping when sampling, and `suggest_params`'s `else` raises. The families are
plain `Literal` assignments, not the `type` statement: cloudpickle cannot pickle a class
  annotated with a `type` alias defined in a cell (measured), and the joblib workers receive
  the components by cloudpickle. In a package module either form pickles.
- D4. The metric is built **in the notebook**: `Metric("cohort_poisson_deviance",
  cohort_poisson_deviance, direction="minimize")`; `optuna.create_study(direction=metric.direction)`;
  the objective returns the metric's value unchanged. The module-level constants leave
  `scoring.py` (T3); the `Metric` class and the two functions stay there.
- D5. Pruning and parallelization beyond today's parallel studies are **not decided now**; a
  later performance pass (§7 holds the facts for it).
- D6. Still in force from 2026-10-15: one joint study per family on the composite; a trial
  chooses per component its feature transformer; `n_trials = 200` (since D20, 2026-10-22: 400, with `n_startup_trials = 20`); studies of
  one model in parallel with joblib, study i seeded `seed + i`; only the best run is refit;
  Model 2's totals come from the edited list `MODEL2_TOTALS` (superseded 2026-10-21, below:
  16 run cells, no list); `split_method = "random"`. (The parallel studies: superseded by
  D13.)
  **Changed 2026-10-18 (the user, at T1.2's review):** LightGBM's loss is no longer a sampled
  categorical. `lgbm_poisson` and `lgbm_gaussian` are separate families, and
  `LGBMRegressorParams.objective` is fixed by the family. Why: TPE here is univariate (D7),
  so it cannot learn that good hyperparameters differ per loss; a 30-trial study that sampled
  the loss split 10/20 and did worse than either loss alone; and the family table compares
  the losses directly. The cost: one run cannot mix the losses across cohorts, and there is
  one more study per exposure setting.
  **Changed 2026-10-21 (the user, at T2.1's plan):** no `MODEL2_TOTALS` list. With one cell
  per run (D14), the run cells present are the runs tuned: all four totals, 16 cells. To drop
  a total, delete its four cells and its four names in `MODEL2_RUNS`.
  **Changed 2026-10-18 (the user, at T1.3's review):** the exposure is no longer a sampled
  categorical either. It is a study setting (`CountComponent.use_exposure`), and every count
  family runs with it on and with it off, as the older plan's E5 argued (the exposure changes
  the target, a count or a weighted rate, so one TPE budget would under-tune the losing arm).
  Measured on validation (one fit per cohort, fixed hyperparameters): the GLMs lose 30–50%
  with the exposure off, because their variants hold no `n_apartments`; LightGBM loses about
  3%. The user kept both arms for every family. Model 1 runs 8 studies; Model 2 runs the four
  totals × exposure × the two probability families, 16 run cells (since 2026-10-21, D14).
- D7. Settled by probe in this session (Optuna 4.9.0, joblib 1.5.3): the trial value is the
  plain mean of the fold scores (KFold on 917 rows: folds of 184/183 rows; since D9, the
  default of an aggregator); the sampler is
  `TPESampler(seed=study_seed)` (the `multivariate=True` option raises `ExperimentalWarning`
  under `-W error`); the best run is rebuilt with `suggest(optuna.trial.FixedTrial(best_params))`
  (exact for `None`/`bool`/`str`, raises `ValueError` on a missing name or a foreign value);
  `study.optimize(..., catch=(RuntimeError,))` keeps a study alive when an NB2 fit does not
  converge (without it the study stops and `Parallel` discards the sibling studies; returning
  `inf` would hide the traceback); an in-memory `optuna.Study` pickles back whole from a loky
  worker (`trials`, `best_trial.distributions`, `best_params` all present).
- D8. **Changed 2026-10-20 (the user, at two rounds of T1.4's review; supersedes the parts of
  D2, D3 it names):**
  - **A component is a named part of the composite.** It holds `name`, so
    `CountComponent(name, family, use_exposure)` and `ProbabilityComponent(name, family)`.
    Model 1 has one component per cohort, named after it, because each cohort is a separate
    prediction problem; a study is the tuple of its three components. Model 2 has `"total"`
    and `"probability"`.
  - **The name replaces the prefix argument.** A trial's names are `<name>__<parameter>`,
    and the component adds the separator (`prefix` property). The separator stays `__`
    because cohort names hold `_` and the tables split on the first `__`. The trial names
    are as before.
  - **An abstract `Component(abc.ABC)` base.** It holds `name`, `family`, the checks of
    `__post_init__` and the shared `suggest_feature_transformer`; `suggest_params` and
    `make_estimator` are abstract.
    - The family is checked against the subclass's `Literal` annotation:
      `get_args(get_type_hints(type(self))["family"])`. This needs no list of families
      beside the `Literal`, and it also resolves string annotations
      (`from __future__ import annotations`) in a package module (measured).
    - No `type` field: the first round had one; the second removed it, since no code read
      it and the subclass states the kind.
  - **Each subclass holds its families' candidate feature transformers.** The ClassVar
    `FEATURE_TRANSFORMER_NAMES` maps family → names, as a `MappingProxyType`.
    - It is fixed on the class, so no component can carry another family's: the guarantee of
      T1.3's third review, without the mapping argument.
    - A family without an entry raises at construction.
    - The trial samples a name, because Optuna's categorical choices must be primitives:
      objects raise a `UserWarning`, an error under `-W error` (measured).
    - `suggest_feature_transformer(trial, feature_transformers)` returns the object
      `feature_transformers[name]`.
    - Read-only, because ruff's RUF012 flags a mutable class attribute (in a cell it cannot
      see the `ClassVar` parameter). It pickles through cloudpickle and runs under
      `Parallel` (measured).
  - **Sampling and construction are separate steps, both on the component.**
    `suggest_params(trial)` returns the params record. `make_estimator(hyperparameters, *,
    seed)` builds the unfitted estimator from a mapping of the estimator's keywords (a dict
    works; the builder passes `asdict(params)`): an `if`/`elif` on the family with a raising
    `else`.
    - `make_`: scikit-learn's word for its factories (`make_pipeline`,
      `BaseEnsemble._make_estimator`).
    - Tracking does not depend on where the estimator is built: Optuna records every
      sampled value by name.
    - D3's "no dataclass builds a model" becomes: a component builds its estimator; the
      `CountModel` / `CohortProbabilityModel` and the composite are built in
      `suggest_model1` / `suggest_model2`.

- D9. **Added 2026-10-21 (the user, at T1.5's review):** the fold scores are combined by an
  aggregator, the user's `Aggregator` classes (`Mean`, `WeightedMean`, `UpperBound`).
  - **Where the classes live:** they moved out of the package that D1 keeps out of the
    notebook into `src/age_group_prediction/aggregation.py`, a copy unchanged but for the
    `_config` import and one docstring word. Their tests went to
    `tests/unit/test_aggregation.py`, and mypy's strict list gained the module. T4 deletes
    the package's copies.
  - **The default:** `cv_score(..., aggregator=None)` and `tune_model(..., aggregator=None)`,
    where None means `Mean()`, resolved inside `cv_score`. This is scikit-learn's pattern
    (`cv=None`, `scoring=None`). A `Mean()` default in the signature trips ruff's B008, and
    a missing required argument needs no hand-written raise: Python raises `TypeError`.
  - **Why the aggregator is an argument:** it decides the selection. Measured on two Model 1
    models, the weighted mean differs from the plain mean by 0.0006, while the median flips
    which family wins (by mean 2.262 vs 2.328, by median 2.313 vs 2.228).
  - **Measured at seed 42 (2-trial `poisson_glm` study):**

    | Aggregator | Best value |
    |---|---|
    | default | 2.48456 |
    | `Mean()` | 2.48456 |
    | `WeightedMean()` | 2.48390 |
    | `UpperBound()` | 2.70768 |
  - `cv_score` names the fold variables (`X_train`, `Y_train`, `exposure_train`, `X_val`, …,
    `Y_val_pred`); the user's edit.

- D10. **Changed 2026-10-21 (the user, at T1.6's review; supersedes D2's "a component holds
  a study's fixed settings", D3's `suggest` on the params classes, and D8's sampling methods on
  the component):** a component is a complete record of one part of a run, and every Optuna
  call lives in a search space.
  - **Records, no Optuna, `kw_only=True`:**
    - The params dataclasses `LGBMParams`, `PoissonGLMParams`, `NB2Params` and
      `LogisticParams` hold fields only.
    - `Component(abc.ABC)` holds `name`, `family`, `feature_transformer` (a name in
      `FEATURE_TRANSFORMERS`) and `hyperparameters` (the family's params class, checked by
      a `PARAMS` ClassVar).
    - `CountComponent` adds `use_exposure`; `ProbabilityComponent` adds
      `calibration_method`.
    - `make_estimator(*, seed)` is an if/elif on the family with a raising `else`.
    - A component is built the same way by hand or from a trial, e.g.
      `CountComponent(name="n_elementary", family="nb2", use_exposure=True,
      feature_transformer="glm_base", hyperparameters=NB2Params(dispersion=0.15,
      alpha=0.01))`.
  - **Search spaces, every `trial.suggest_*`:**
    - `SearchSpace(abc.ABC)` holds `name` and `family`, plus the per-subclass
      `FEATURE_TRANSFORMER_NAMES` ClassVar and the shared feature-transformer and LightGBM
      sampling.
    - `CountSearchSpace` adds `use_exposure`; `ProbabilitySearchSpace` adds nothing.
    - `suggest(trial)` returns a complete component. The draws keep their old order, so the
      seeded studies are unchanged: T1.6's Check C reproduced to the fourth decimal on all
      8 studies.
  - **Builders without Optuna:** `make_count_model`, `build_model1(components, …)` and
    `build_model2(total, probability, …)`. `suggest_model1(trial, spaces, …)` and
    `suggest_model2(trial, total_space, probability_space, …)` sample, then build.
  - **The best run:** the best components are `space.suggest(FixedTrial(best_params))` per
    space, and `asdict(component)` is the saved record, its hyperparameters nested.
  - **LightGBM's objective** is set by `make_estimator` from the family: one `LGBMParams` for
    the regressor and the classifier, so the record states the loss once.
  - **Why, the user's words:** a component should hold everything needed to create a run, so
    the best one can be saved and set by hand; Optuna's suggest is kept apart. Best practice
    agrees: a configuration separate from its search space (scikit-learn's estimator params
    vs `param_distributions`; Optuna's objective building a plain config). A component that
    sampled its own fields would be half-built until a trial filled it.

- D11. **Changed 2026-10-21 (the user, at D10's review; revises D10):**
  - **Search spaces have explicit methods and build nothing.** A search space holds only
    `name` and `family` (no exposure). Given the family, it offers:
    - `suggest_hyperparameters(trial)`, returning the family's params record;
    - `suggest_feature_transformer_name(trial)`;
    - `suggest_calibration_method(trial)`, probability only.
  - **The builder assembles the components.** `suggest_count_component(trial, space, *,
    use_exposure)` and `suggest_probability_component(trial, space)` call those methods and
    feed the draws into the component.
    - The exposure is the study's setting: a Model 1 study is `(spaces, use_exposure)`.
    - Keyword arguments evaluate left to right, which keeps the old order of the draws:
      Check C reproduced T1.6's 8 best values exactly.
  - **The component holds `feature_transformer_name`, resolved by the builder** from
    `FEATURE_TRANSFORMERS`. That is best practice: a record refers to a registered object by
    name (scikit-learn's `scoring="…"` strings, config systems). It also fits the
    measurements: a `FeatureTransformer` hashes by identity, is not equal to its clone, and
    is not JSON. So `json.dumps(asdict(component))` is the saved best component.
  - **No `PARAMS` check, no `object` field.** Each subclass annotates its own
    `hyperparameters` union.
  - **The seed is an argument of `make_estimator`:** the run decides it, and a stability
    check can refit one component under several seeds.
  - **The params classes stay separate records** (what to build; the search space says how
    to sample). Measured: LightGBM accepts and fits a typo silently (`max_dept=3`), and
    `LGBMParams(max_dept=3)` raises `TypeError` when the component is built.

- D12. **Changed 2026-10-21 (the user, at D11's review):** a study is one family combination
  and exposure, so each combination's results can be seen with its own best parameters.
  - **Model 1:** a study is `(family, use_exposure)`, with one family for every cohort.
  - **Model 2:** a study is `(total_family, probability_family, use_exposure)`.
  - **Keys:** `MODEL1_STUDY_SETTINGS` and `MODEL2_STUDY_SETTINGS` list these plain tuples,
    and the studies are keyed by them. (Superseded by D14: one cell per run, so neither list
    exists; a run's record holds its settings.)
  - **Search spaces are built inside.** `suggest_model1_components(trial, *, family,
    use_exposure, cohorts)` and `suggest_model2_components(trial, *, total_family,
    probability_family, use_exposure)` build them from the family. `suggest_model1/2` call
    them, and the refit calls them on `FixedTrial(best_params)` for the best components.
  - **Rename:** the builders' keyword `feature_transformers` is now
    `feature_transformer_dict`; the constant `FEATURE_TRANSFORMERS` keeps its name.
  - **`y_min` stays a builder argument** (the user's choice). The run record holds `seed`
    and `y_min` once.
  - **Unchanged:** the search itself. Check C reproduced T1.6's 8 best values again.

- D13. **Changed 2026-10-21 (the user, at T1.7's review; supersedes D6's "studies of one model
  in parallel with joblib"):**
  - **A run is one `(family, use_exposure)` combination, and the runs go one after
    another.** Each trial fits its 5 folds in parallel (`fold_score`, `cv_score(…,
    n_jobs)`, the run cell passing `n_jobs=n_splits`).
  - **Measured, one trial,** one fold after another vs in parallel, identical scores:

    | Family | One after another | In parallel | Speed-up |
    |---|---|---|---|
    | `lgbm_poisson` | 0.53 s | 0.21 s | ×2.5 |
    | `lgbm_gaussian` | 0.40 s | 0.18 s | ×2.3 |
    | `nb2` | 1.50 s | 0.45 s | ×3.3 |
    | `poisson_glm` | 0.33 s | 0.13 s | ×2.6 |

    - A new `Parallel` per trial costs about 12 ms, since joblib reuses the workers.
    - A worker's `RuntimeError` comes back as one, so `catch` still marks the trial FAIL.
    - Check C at 3 trials reproduced T1.6's 8 best values: 13.6 s against 22.6 s with the
      studies in parallel.
  - **No pruning.** `WilcoxonPruner` needs the folds one at a time and saves at most 20% of
    a pruned trial at its default threshold. Parallel folds save 55–70% of every trial.
  - **A run keeps only a summary** (`summarize_study`): the best parameters, `cv_best`,
    `n_complete`, `n_failed`, the seconds, and the completed trials per feature
    transformer. The Optuna study is dropped.
  - **Only the best parameters are evaluated.** The score cell rebuilds the best components,
    fits and scores them on training and validation, and keeps one JSON record per run in
    `MODEL1_RUNS`. The fitted models and the best-choices table go: the components rebuild
    the first and hold the second.
  - **`evaluate` stays `metric(y_true, y_pred)`.** Each metric decides how to treat the
    columns, as scikit-learn's `multioutput` does. `cohort_log_loss` needs the matrix and
    raises `AxisError` when flattened, and both models predict the target's column order.

- D14. **Changed 2026-10-21 (the user, at D13's review; supersedes D13's "a run keeps a
  summary"):**
  - **One cell per run.** Model 1 has eight explicit run cells, one per `(family,
    use_exposure)`, seeded `seed + 0` … `seed + 7` in the old order, each behind the shared
    "Tune Model 1" button. A run cell tunes, rebuilds the best components from
    `FixedTrial(best_params)`, and scores them on training and validation with
    `fit_and_score`.
  - **The run is kept as a record** `model1_<family>_[no_]exposure`: `family`,
    `use_exposure`, `trials` (`study.trials_dataframe()`, every trial), `components`,
    `cv_best`, `score_table`, `seconds`, `seed` and `y_min`. The study object is not kept.
  - **No function that is not needed.** `summarize_study`, `trial_counts`,
    `categorical_names` and `arm_counts` went. The tables cell reads the counts from the
    kept trials: completed trials per feature transformer by `value_counts`, and
    `n_complete` / `n_failed` from `state`.
  - **Names:** `score_table` (it was `rows`), and `y_pred_train` / `y_pred_val`.
  - **Why the full history.** Best practice is to keep every trial and evaluate only the
    best: the history is what is analyzed later (optimization history, importance, starved
    choices), and it is small. Measured: about 1.7 KiB per trial, so roughly 335 KiB per
    200-trial run.
  - **Why the per-transformer counts.** TPE concentrates on what looked good early. In a
    100-trial Poisson GLM run, `n_elementary`'s transformers got 48, 24, 9, 7, 7 and 5
    trials, and `glm_base` took 22 of the last 30. A transformer tried 5 times has not been
    judged. Only categorical choices are counted, since a float takes a new value in every
    trial.
  - **A run whose trials all fail** raises in its own cell ("No trials are completed yet").
    The other runs are untouched, and the tables wait for it.
  - **Measured:** Check C at 3 trials reproduced T1.6's 8 best values in 13.6 s.

- D15. **Added 2026-10-21 (T1.8's review, the user's choices):**
  - **Strict checks.** Fold processes do not inherit a script's `-W` filters. Measured: a
    warning in a worker under `python -W error` is printed and the call returns; run
    sequentially, it raises. Check C therefore runs with `PYTHONWARNINGS=error`, which the
    processes inherit, so a warning in any fold fails the check as it would in a single
    process. `cv_score`'s docstring says so. The notebook's behaviour in marimo is
    unchanged. Check C under it at 3 trials: no warning, T1.6's 8 best values.
  - **Untried transformers show 0.** The trials-per-transformer table is reindexed on the
    family's candidates (`CountSearchSpace.FEATURE_TRANSFORMER_NAMES`), so a transformer
    with no completed trial shows 0 instead of vanishing (finding F1).
  - **`num_leaves` above `2**max_depth` changes nothing in LightGBM** (measured: at
    `max_depth` 3, 96% of the 7–63 range; at 5, 54%). It is left as a known inefficiency:
    the search spaces stay fixed, and TPE only spends some trials on equivalent settings.
    Revisit in a search-space pass.
  - **The 200-trial run of Model 1** (the clicked path, button patched): 380 s, no failed
    trial. Every feature transformer was tried in every run and cohort, the least-tried 6–9
    times in the GLM families (6 candidates) and 17 or more in LightGBM (2 candidates).
    Best validation composite:

    | Family | Validation composite |
    |---|---|
    | `lgbm_poisson`, exposure on | 2.1604 |
    | `lgbm_gaussian`, exposure on | 2.1646 |
    | `poisson_glm`, exposure on | 2.1882 |
    | `nb2`, exposure on | 2.1962 |

    Every family is worse with the exposure off, the GLMs by about 0.85.
  - **Pending, to decide after T2.2:** TPE's random start, `n_startup_trials` 40 or 10 (the
    default).
    - Measured at 200 trials, exposure on:

      | Run | Random start | Best | Least-tried transformer per cohort |
      |---|---|---|---|
      | Poisson GLM | 10 | 2.2490 | 6–9 |
      | Poisson GLM | 40 | 2.2490 | 9–11 |
      | NB2 | 10 | 2.2428 | 9 each |
      | NB2 | 40 | 2.2426 | 7–11 |

      400 trials gave no gain.
    - A change re-runs both models and resets the Check C baseline (handoff §3).
    - **Postponed (the user, 2026-10-21, after T2.2):** decided after T2.4, together with
      `n_trials`, since pruning and parallelization change what a trial costs. The evidence
      at 10, from T2.2's 200-trial run: every choice was tried at least 7 times (the GLM
      totals' transformer 7–9, logistic's 8–12, every LightGBM choice 16 or more, the
      calibration 12–34).
    - **All 24 runs kept (the user, 2026-10-21):** D6 stands. The GLM totals with the
      exposure off are about 0.6 worse on CV in both models (Model 1: Poisson GLM 2.8652
      against 2.2490, NB2 2.8572 against 2.2428; Model 2: 2.844–2.869 against 2.235–2.258),
      but no run is dropped.

- D16. **Added 2026-10-21 (the user, at T2.2's plan and implementation):**
  - **The calibration candidates are a ClassVar of the search space**,
    `ProbabilitySearchSpace.CALIBRATION_METHODS = (None, "temperature", "sigmoid")`. It is read
    by `suggest_calibration_method` and by Model 2's tables cell, so there is one source of
    truth, as with `FEATURE_TRANSFORMER_NAMES`.
    - Best practice: the search space declares its choices, and the analysis reads them from
      there. Optuna's own analysis reads the choices from each trial's recorded distribution,
      which the trials table does not keep.
    - Measured: `list(CALIBRATION_METHODS)` gives identical seeded draws (30 TPE trials), and
      Check C reproduced all 24 baselines.
  - **The trials per choice are counted by label.** Measured with pandas 3.0.5:
    - `value_counts()` drops the `None` calibration.
    - With `dropna=False`, `reindex([None, …])` counts a mixed column (str dtype, NaN) right,
      but gives 0 for a column where every trial drew `None` (object dtype). Check C at 3
      trials hit this: the counts did not add up to `n_complete`.
    - A DataFrame also turns `None` into NaN.
    - So the column is counted after `fillna("None")` and reindexed on `str(candidate)`.
      Every column shape counts right, and the table prints `None`.
  - **A family pair reads as one family**, `"<total>+<probability>"` (e.g.
    `"lgbm_gaussian+logistic"`). Model 2's tables then have Model 1's settings columns
    (`family`, `use_exposure`), and the best-per-model cell treats both models alike.
    - The run records keep `total_family` and `probability_family`.
    - The separator is `+` because the family names contain `_`.
  - **`use_exposure` is its own column** in every table, not part of the family name:
    - D6 made the exposure a run setting, and in Model 2 it applies to the total only.
    - The best-per-model table gains it; the old cell, written before the exposure was a run
      setting, dropped it.

- D17. **Added 2026-10-21 (the user):** the old package is replaced, not only deleted.
  - Once the notebook's tuning code is final (after T2.4), it moves into a new
    `hyperparameter_tuning/` under the same name, and the notebook imports it.
  - The code was written package-ready for this (D3): it reads no notebook global.
  - Why: unit tests, mypy strict, and reuse outside the notebook.
  - D1's "none of the package's classes or functions is imported" applies to the old package
    only.
  - T4 becomes two stops: T4a deletes the old package, and T4b builds the new one.

- D18. **Added 2026-10-21 (the user):** each model's best run is evaluated with the
  inspection package (T2.5). This brings the older plan's E3, E7 and E8
  (`EXPERIMENT_NOTEBOOK_PLAN.md`) into the new notebook.
  - **No wrapper for the two importance methods.** `PermutationImportance` and
    `ShapExplainer` are each called as they are, and one cell puts their rankings side by
    side. Why:
    - They answer different questions with different inputs.
    - Permutation importance is global and performance-based: it needs `y` and a metric,
      and gives one mean and std per column in the metric's units.
    - SHAP attributes the prediction: it needs a background, takes no `y`, and gives a
      value per row, column and cohort.
    - A wrapper would need a union signature (`y` for one, the background for the other)
      and would collapse SHAP to mean |SHAP|.
    - scikit-learn's `permutation_importance` and the shap library are separate APIs; the
      practice is to compare the rankings, which also shows where they disagree.
  - **Placement: T2.5**, after T2.4 and before T3, so the notebook is complete before T4b
    moves code into the package. It replaces the handoff's older item, "the bootstrap and
    importance check".

- D19. **Added 2026-10-22 (T2.4, measured; the user's choice):** today's design stays as
  it is: the folds of a trial in parallel (`n_jobs=n_splits`), no pruning, LightGBM
  `n_jobs=1`, `catch=(RuntimeError,)`. T2.4 holds the measurements. In short:
  - **Where the time goes:** in the slow pairs the LightGBM classifier's fit is the trial,
    and the calibration multiplies it by about 5 (`CalibratedClassifierCV(cv=5,
    ensemble=False)` fits the classifier 6 times per fold, 30 per trial). The totals,
    prediction and scoring are a few percent.
  - **Pruning loses:** `WilcoxonPruner` saves 2–22% of the fold fits but needs the folds one
    after another, which costs the 65–72% the parallel folds save; the slowest pair ran
    2.8–5× longer, the fastest family 3.3×. A pruned trial's value is a partial mean, so the
    seeded search diverges. The pruner is experimental and its constructor raises
    `ExperimentalWarning` under `-W error`. Early stopping inside LightGBM does not fit the
    composites (no `eval_set` path; the final refit would have no stopping set).
  - **Parallelization elsewhere loses:** LightGBM threads are 3× slower at 2 threads on 917
    rows; Model 1's cohort fits in threads are slower; the calibration's inner folds in
    parallel gain nothing inside a fold process (nested joblib runs sequentially).
  - **Two places are unsafe:** concurrent `Parallel` calls from threads in one process
    (runs at once in threads, or `study.optimize(n_jobs>1)` with fold processes) return
    wrong scores: joblib's shared pool mixes the calls' inputs (measured: four fixed models,
    different wrong values on every repeat). marimo also runs cells one at a time.
  - **`catch` stays `(RuntimeError,)`:** NB2's non-convergence arrives as one from both fit
    paths; a cohort unobserved in a fold is the same for every trial and should end the
    run; a `ConvergenceWarning` raises only under strict warnings, which D15 wants visible,
    and none occurred in 200 trials.
  - What would shorten a run is a modeling choice, not asked for here: fewer calibration
    inner folds, a lower `n_estimators` ceiling, or fewer trials.

- D20. **Added 2026-10-22 (the user, after T2.4's measurements):** `n_trials = 400` and
  `n_startup_trials = 20` (a new experiment setting, logged with the others; `tune_model`
  takes it as a keyword and gives it to `TPESampler`). The user's rule: never decrease
  `n_trials`; decide by best practice for the search space.
  - **The search space:** 23–33 dimensions for the runs with LightGBM in every part (11 per
    LightGBM component), 14–15 with one LightGBM part, 5–9 for the GLM runs.
  - **Best practice:** Optuna's random start of 10 is a default for small spaces; Hyperopt's
    TPE starts with 20, and the usual rule is 10–20% of the budget. For the budget, about 10
    trials per dimension, and empirically a plateau of the best-so-far curve with agreement
    between seeds.
  - **Measured at 400 trials** (the folds in 5 processes, `PYTHONWARNINGS=error`):

    | Run | Start | Best at 200 / 300 / 400 | Best found at | Wall |
    |---|---|---|---|---|
    | Model 2 `lgbm_gaussian+lgbm`, exposure on | 10 | 2.1066 / 2.1020 / 2.0947 | 399 | 177 s |
    | same | 20 | 2.0962 / 2.0962 / 2.0962 | 177 | 269 s |
    | same | 40 | 2.1127 / 2.1080 / 2.1080 | 283 | 356 s |
    | Model 1 `lgbm_poisson`, exposure on | 10 | 2.1459 / 2.1429 / 2.1429 | 299 | 111 s |
    | same | 40 | 2.1330 / 2.1253 / 2.1253 | 231 | 88 s |

    Every run improved after 200, by 0.003–0.012, and T2.4's nine search paths of the same
    pair at 200 spread over 0.02, as wide as the gap between the two LightGBM totals: at
    200 the LightGBM runs were not converged. The random start's effect is within that
    noise (40 won on Model 1 by 0.018 and lost on Model 2 by 0.013); its cost is not: the
    random trials draw large models, so Model 2's run took 1.5–2× longer with 20 or 40.
    Chosen: 400 and 20, the best practice at the lower cost.
  - **Elimination considered and declined (the user):** by a racing rule (an arm more than
    0.2, 10× the noise, behind its model's best at 200 trials is dropped) the exposure-off
    GLM arms would go in both models (2 of Model 1's 8, 4 of Model 2's 16; about 15 of 70
    minutes per click). The user kept all 24 runs, so the tables stay complete. Trimming
    Model 2's totals from Model 1's ranking is not the practice: the total is its own
    target; Model 2's own 200-trial results are the evidence, and they agree with Model 1's.
  - **Check C is unchanged:** a 3-trial study draws the same points with a start of 10 or
    20 (measured on two runs), so the 24 baselines at 3 trials stand.
  - **The 400-trial click** (both buttons patched, `PYTHONWARNINGS=error`, 2026-10-22):
    3708 s in all, Model 1 821 s, Model 2 2871 s; no failed trial, no warning; every
    choice tried at least 11 times (Model 1) and 14 times (Model 2).
    `lgbm_gaussian+lgbm` (exposure on) reproduced the probe's 2.0962.

    | Model 1 run | Exposure on: cv_best / train / val / s | Exposure off: cv_best / train / val / s |
    |---|---|---|---|
    | `lgbm_poisson` | 2.1280 / 1.3350 / 2.1254 / 98 | 2.1654 / 1.2024 / 2.1919 / 90 |
    | `lgbm_gaussian` | 2.1332 / 1.1452 / 2.1438 / 93 | 2.2085 / 1.2526 / 2.3043 / 97 |
    | `poisson_glm` | 2.2490 / 2.1828 / 2.1895 / 39 | 2.8644 / 2.7846 / 3.0352 / 36 |
    | `nb2` | 2.2436 / 2.1802 / 2.1993 / 194 | 2.8561 / 2.7991 / 3.0340 / 174 |

    | Model 2 run | Exposure on: cv_best / train / val / s | Exposure off: cv_best / train / val / s |
    |---|---|---|---|
    | `lgbm_poisson+logistic` | 2.0489 / 1.2706 / 2.1285 / 67 | 2.0878 / 1.4289 / 2.1777 / 59 |
    | `lgbm_poisson+lgbm` | 2.0809 / 1.1324 / 2.1775 / 224 | 2.0976 / 1.1519 / 2.1679 / 442 |
    | `lgbm_gaussian+logistic` | 2.0715 / 1.4267 / 2.1256 / 56 | 2.1336 / 1.5228 / 2.2040 / 56 |
    | `lgbm_gaussian+lgbm` | 2.0962 / 1.3288 / 2.1202 / 247 | 2.1483 / 1.3811 / 2.2304 / 306 |
    | `poisson_glm+logistic` | 2.2418 / 2.1783 / 2.1749 / 58 | 2.8511 / 2.7754 / 3.0380 / 48 |
    | `poisson_glm+lgbm` | 2.2562 / 2.0562 / 2.2019 / 231 | 2.8656 / 2.6479 / 3.0672 / 244 |
    | `nb2+logistic` | 2.2289 / 2.1779 / 2.1848 / 87 | 2.8457 / 2.7792 / 3.0489 / 89 |
    | `nb2+lgbm` | 2.2479 / 2.0261 / 2.2132 / 371 | 2.8633 / 2.6248 / 3.0780 / 286 |

    `best_df`, by validation: Model 1 `lgbm_poisson`, exposure on, cv 2.1280, val 2.1254
    (2.1604 at 200 trials); Model 2 `lgbm_gaussian+lgbm`, exposure on, cv 2.0962, val
    2.1202 (at 200 trials `lgbm_gaussian+logistic` won with 2.0950; the four LightGBM pairs
    with the exposure on sit within 0.06 on validation, inside the search noise, so the
    best pair is not settled by this table). Model 2's LightGBM-classifier runs took 224–442
    s, its logistic runs 48–89 s.

- D21. **Added 2026-10-22 (the user, at the T4 design review, planned ahead):**
  - **What moves into the package, what stays.** Code moves when it is a mechanism or a
    record a test can pin down and another notebook could reuse: the params records, the
    families and components, the search spaces (the bounds and candidates stay the one
    source of truth the tables read), the builders, `fold_score` / `cv_score` /
    `tune_model` / `fit_and_score`, and the aggregators. Code stays when it is the reading
    of this experiment: the settings, the metrics (D4), the constants, the buttons, the 24
    run cells, the tables, `best_df`, the inspection cells.
  - **Six modules, one-way dependencies:** `params` ← `components` ← `search_spaces`;
    `builders` uses those and `modeling`; `cross_validation` uses `aggregation` only (it
    takes `suggest_model` as a callable); `aggregation` stands alone. No runner over
    runs, no results-table class, no MLflow, no transformer registry in the package.
  - **Evaluation stays outside the model.** No bootstrap or importance on
    `BaseAgeGroupModel` (scikit-learn's split: estimators fit and predict; `inspection`
    judges them; a model would otherwise carry settings that are not its own, and Model C
    would inherit them). The three inspection classes keep their settings. T2.5's recipe,
    "a fitted model, the splits, every metric's interval, the importances", becomes
    `inspect_model` in `age_group_prediction.inspection`, typed and tested, so the same
    evaluation runs on training, validation and later test.
  - T4b is split into T4b.1 (records and search spaces), T4b.2 (builders, loop,
    aggregation) and T4b.3 (the notebook imports the package).

- D22. **Added 2026-10-22 (the user, at T2.5's plan; revises D21's recipe):**
  - **The bootstrap evaluator holds the resampling design only:**
    `BootstrapEvaluator(n_resamples, alpha=0.05, random_state=None)`, and
    `evaluate(model, X, y, groups=None, *, exposure=None, metrics)` takes a sequence of
    `Metric` and always returns a DataFrame, one row per metric name: `score` (the
    unresampled rows), `mean`, `lower`, `upper`. It predicts once and draws the replicates
    once, so every metric scores the same resampled rows and the intervals are paired.
    `BootstrapResult` went. The practice: scikit-learn's `cross_validate(scoring=[...])`;
    no return type that depends on the input's shape (a Series for one metric, a frame for
    several, was considered and declined). The seed stays a field, as in `KFold` and
    `permutation_importance`; `None` means fresh entropy, NumPy's and scikit-learn's default.
  - **The bootstrap stays outside the model** (D21 confirmed): the model's `evaluate` is a
    pure scoring hook; the resampling settings are the evaluation's, not the model's.
  - **Two separate components, not one recipe.** The evaluation (the evaluator over the
    splits) and the importances (`PermutationImportance` and `ShapExplainer` called as they
    are, D18). No `inspect_model`: with the evaluator taking the metrics, the over-splits
    recipe is two calls and a `concat` in the notebook, so nothing else was added to the
    package.
  - **SHAP, measured** (the largest LightGBM sizes, 400 trees at depth 8,
    `PYTHONWARNINGS=error`): exact at 100 rows × 100 background 376 / 476 s per model
    single-threaded (Model 1 / Model 2), 89 / 141 s on all cores; at 100 × 50 on all cores
    46 / 93 s; at 50 × 50 33 / 46 s. shap's own auto rule picks the exact explainer at ≤ 10
    columns (9 here); the permutation explainer at its minimum 19 evaluations took 21 s and
    was within 6% of exact per column with one rank swap. Chosen: exact, `n_shap_rows =
    100` validation rows against `n_shap_background = 50` training rows, `n_jobs=-1` (the
    explainer's existing setting). The bootstrap of 7 metrics × 1000 resamples costs 2–3.4
    s per split and the permutation importance at `n_repeats = 10` 3 s (validation) to 9 s
    (training), so SHAP decides the cost.

- D23. **Added 2026-10-25 (the user; the decisions and the code are in
  [NOTEBOOK_COMPONENT_REDESIGN_PLAN.md](NOTEBOOK_COMPONENT_REDESIGN_PLAN.md) §0):** the search
  space (`ComponentSearchSpace`) holds the injected registry and returns the chosen
  `FeatureTransformer`; explicit public `suggest_<family>_hyperparams` methods; the component
  holds the transformer object and `to_dict()` writes it as its declaration
  (`FeatureTransformer.to_dict()` in `src`); the `Literal` checks removed; `random_state=None`
  replaces `seed`; the builders named after the models with `CountModel` and
  `CohortProbabilityModel` constructed inline; `shared_hyperparams` for Model 1 (one draw of the
  hyperparameters under `shared__`, each cohort its own transformer; the 8 run cells pass
  `False`, a test checks both values). Check C reproduced the 24 baselines. Supersedes D10's
  "the transformer by name" in the record and D11's `suggest_feature_transformer_name`.
  **Revised 2026-10-26 (S6, the user's D23.9–D23.11):** a search space takes
  `feature_transformers`, one family's candidates (name → object), and draws from its keys; no
  `FEATURE_TRANSFORMER_NAMES`, no registry, no `__post_init__`. The candidates are data in the
  notebook's section 4, `COUNT_FEATURE_TRANSFORMERS` and `PROBABILITY_FEATURE_TRANSFORMERS` (family →
  `{name: object}`, today's order), the one source of truth the run cells and the tables read
  (superseding D15's and D16's ClassVar for the transformers; `CALIBRATION_METHODS` stays a
  ClassVar). The package holds the mechanism and the notebook the data, as in scikit-learn's
  `GridSearchCV(param_grid=...)`. The shared hyperparameters are drawn inside the loop: every
  cohort asks the `shared` space, the first draws, and the trial returns the stored values to the
  next (probed for `RandomSampler`, `TPESampler`, `FixedTrial`). The flag is `share_hyperparams`
  everywhere. The two arms to compare are no sharing and partial sharing; complete sharing needs
  no run. Check C reproduced the 24 baselines again; the shared probe gave 2.1812, as before.
- D24. **Added 2026-10-27 (the user's notes on the S6 notebook):**
  - D24.1 `CohortProbabilityModel`'s `replication` is renamed `replication_method` (the alias
    `ReplicationType` → `ReplicationMethod`) and is **fixed at `"weighted"`** in the builder, not
    searched and not a run setting: for the logistic classifier the two replications are the same
    fit (the likelihood identity, TOTAL_TIMES_PROBABILITY_MODEL.md §3.1), for LightGBM they
    differed within noise and `"per_child"` is 2–3× slower; a categorical that cannot change the
    objective for one family only splits TPE's trials (D6's lesson). The §3.3 remark about tuning
    it concerns row-resampling forests, which the notebook does not tune.
  - D24.2 **`share_feature_transformer` beside `share_hyperparams`; a shared choice is drawn once,
    before the loop, under `shared__`** (the `shared` `CountSearchSpace`), and each cohort takes
    it or draws its own under `<cohort>__`; no `None`, no index test, no second suggest. The
    user's hand-written version recorded the shared draws under the first cohort's name and
    required a keyword the run cells did not pass; a first replacement (a generator of per-cohort
    spaces relying on Optuna's repeated suggest) was rejected as hard to read. With both flags
    `False` the draw order is HEAD's, so the 8 baselines reproduce; the shared arms' values are
    the handoff's tables.
  - D24.3 **The best components are read from `best_params` through the search spaces, no
    `FixedTrial`:** `hyperparameters_from_params` dispatches by family, as `suggest_hyperparameters`
    does, to an explicit `<family>_hyperparams_from_params` that builds the family's params class
    field by field (`params[f"{prefix}max_depth"]`, …); `feature_transformer_from_params` reads the
    candidate dict by the recorded name; `calibration_method_from_params`; the builders cell's
    `independent_cohorts_components_from_params`
    and `total_times_probability_components_from_params` take a study's `best_params` and the
    run's fixed settings (family, exposure, cohorts, candidate dict, share flags: what
    `best_params` cannot hold).
  - D24.4 **The two sharing flags are experiment settings** (revises D23.2 and D23.11): section 2
    holds `share_hyperparams = True` and `share_feature_transformer = False`, logged with the
    others, in every Model 1 record and in `model1_df`; the other arm (both `True`) is a click with
    the setting changed. The test covers the three cases (33 / 13 / 11 parameters).
  - D24.5 **Names:** the search-space field and the builders' keyword `feature_transformer_dict`
    (`total_…`, `probability_…` for Model 2; reverses D23.9's naming only); the builders named
    after the model they return, `build_independent_cohorts_model` and
    `build_total_times_probability_model`, the objectives `suggest_independent_cohorts_model` and
    `suggest_total_times_probability_model`; prose names the models "the independent cohorts
    model (Model 1)" and "the total times probability model (Model 2)", variable names unchanged
    (D23.4).
- D25. **Added 2026-10-29 (the user's review of the D24 notebook; revises D24.3 and D24.5):**
  - D25.1 **The search spaces read nothing back:** no `*_from_params` methods. One family's
    candidates are injected as names, `feature_transformer_names: tuple[str, ...]` (the keys of
    the notebook's `COUNT_FEATURE_TRANSFORMERS[family]` / `PROBABILITY_FEATURE_TRANSFORMERS[family]`),
    and `suggest_feature_transformer_name` returns the drawn name; the caller resolves the
    object through the dict.
  - D25.2 **The component keeps its record as an object,** the field `estimator_params`
    (`LGBMParams`, `PoissonGLMParams`, `NB2Params`, `LogisticParams`: the record `make_estimator`
    unpacks into the estimator's constructor; scikit-learn's term). Two ways to set a component:
    the plain constructor from a trial's draws (the objective), and one classmethod per
    component type, `CountComponent.from_params(best_params, *, name, family, use_exposure,
    feature_transformer_dict)` / `ProbabilityComponent.from_params(best_params, *, name, family,
    feature_transformer_dict)`: one comprehension selects the entries under `<name>__` and
    `shared__` with the prefix stripped (a choice is shared or the cohort's own, never both;
    without the filter the cohorts' entries would collide), `Component.set_estimator_params(params,
    params_class)` fills the family's record by its fields (a missing one raises `KeyError`
    naming it, the check), the transformer through the dict, the calibration from its entry.
    No reader per model (`independent_cohorts_components_from_params` and
    `total_times_probability_components_from_params` removed), no stored prefix, no `FixedTrial`.
  - D25.3 **The objectives are one loop each:** `suggest_independent_cohorts_model` draws the
    shared choices once before the loop under `shared__`, then per cohort its own choices or the
    shared ones, and sets its component with a constructor call; `suggest_total_times_probability_model`
    the same for its two parts, the search spaces constructed inline
    (`total_times_probability_search_spaces` removed). The draw order is D24's: Check C reproduced
    the 24 baselines at 3 trials; the shared probe gave 2.1812 with 13 parameters.
  - D25.4 **The run cells set the best components explicitly:** `_best_components = tuple(
    CountComponent.from_params(_study.best_params, name=cohort, ...) for cohort in COHORTS)` in
    Model 1's 8 cells, the two `from_params` calls in Model 2's 16; the inspection cells the same
    on the records' `best_params`; the build calls unchanged. The notebook test: the objective's
    model equals the one rebuilt from `from_params` on the trial's params, both models, the three
    sharing cases; `KeyError` on an empty mapping (12 tests). ruff's baseline on the notebook is
    14 since the user's D24 edits (none new).

- D26. **Added 2026-10-29 (the user's notes on T4b.2; T4b.2b; revises D9's default, D21's
  dependency line and D25.3's objectives in the package):**
  - D26.1 **The suggest functions draw the components only:**
    `suggest_independent_cohorts_components` and `suggest_total_times_probability_components`
    return the components (a tuple per cohort; the total's and the probabilities') and take no
    `y_min` or `random_state`, which are build settings; `build_*_model` builds the model, for
    the objective and the refit alike. The draw order is D24's.
  - D26.2 **`fold_score`, `cv_score` and `tune_model` become `CrossValidationTuner`**, a frozen
    dataclass holding the study's design, what all 24 runs share: `cv`, `metric`, `n_trials`,
    `n_startup_trials`, `aggregator` (`default_factory=Mean`; D9's `None` existed only to avoid
    ruff's B008 in a function default) and `n_jobs`. Each call takes the data, `groups`
    (scikit-learn's name, `cv_groups` before), the model's settings and `study_seed`, as
    `GridSearchCV(cv=, scoring=, n_jobs=)` does with `fit(X, y, groups)`. Probed: the bound
    `fold_score` runs in loky's processes under `PYTHONWARNINGS=error`, scores as before.
  - D26.3 **The model is named, not passed as a callable (the user's choice of two named
    methods over one `tune` with a `Literal` and a settings record per model):**
    `tune_independent_cohorts_model` and `tune_total_times_probability_model`; each objective
    draws the components, builds the model and returns `cv_score`; a private `_optimize` holds
    the study (`TPESampler(seed, n_startup_trials)`, `catch=(RuntimeError,)`, the seconds).
    `cross_validation` now uses `builders` and `aggregation`; the graph stays one-way
    (`params` ← `components` ← `search_spaces` ← `builders` ← `cross_validation`).
  - D26.4 **`fit_and_score` moves to `evaluation.py`** (the user's choice over `refit.py`),
    unchanged.
  - The notebook is unchanged by T4b.2b; T4b.3 adopts the API (one tuner in a cell, the run
    cells call its methods, `partial` and `tune_model` leave).

**What does not change:** notebook sections 1–4 and the `cv`/`cv_groups` cell; the models in
`src/age_group_prediction/modeling/` (the S1 `get_params`/`set_params` forwarding in
`IndependentCohortModels` stays: tested, `clone`-safe, simply unused by the notebook);
`scoring.Metric`, `cohort_poisson_deviance`, `cohort_log_loss`; the search spaces, variant lists,
`PREDICTION_FLOOR` as `y_min`, the LightGBM constants, no isotonic calibration, NB2 with
`dispersion` and `alpha` searched; the outputs (per-cohort and composite train/validation
deviance of each family's refit best run, best choices per component, trials per arm, best
family per model).

## 1. Rules for the implementer

- Plan mode summary in chat before each sub-task; one sub-task per approval. Never commit,
  never add a `Co-Authored-By` line, never `git checkout`/`stash`/`reset`/`mv`.
- Before the first edit: `git status` clean except what the previous sub-task left; run the
  baseline `uv run pytest -m "not slow"` once at T1.1 (expect green; stop and report otherwise).
- The user edits the notebook between turns: record its md5 (`md5 -q notebooks/02_experiments.py`)
  after every edit and `cmp` before the next; if it changed, show the diff and ask.
- marimo's editor trims a cell's `return` to the names other cells read; a new cell that reads
  a name re-adds it to the producing cell's return. An undefined name in a cell fails silently
  in a script run: verify every new cell ran with the `app.run()` check (§1.1).
- Format new cells by hand to `ruff format`'s output; do not run whole-file `ruff format`
  (three known hunks in the user's cells). `ruff check notebooks/02_experiments.py` has a
  baseline of 9 findings (7 marimo `return`s PLR1711, F401 `clone`, RUF059 `test_age_group`);
  add none. The notebook is outside mypy's `files`.
- The characterization test `tests/characterization/test_experiments_notebook.py` requires:
  the printed lines of sections 1–3 unchanged; every cell whose source contains the string
  `tune_model` (except the cell with `def tune_model(`) to contain `mo.stop(not `, and at
  least one such cell. So the tuning function keeps the name `tune_model`, and that name must
  not appear in the markdown cell or in comments outside the run cells.
- Probes are read-only: `PYTHONPATH=src .venv/bin/python -W error -c "..."`. Scratch files go
  to the session scratchpad, never into `notebooks/`.
- Review subagents write nothing into the repository.

### 1.1 The two checks used by every notebook sub-task

**Check A, script run (fits nothing, ≈ 5 s):**
```sh
cd /Users/galkampel/Desktop/Projects/age-group-prediction
uv run pytest tests/characterization/test_experiments_notebook.py -q
ruff check notebooks/02_experiments.py | tail -3     # 9 findings before T1.1, 8 after (F401 clone gone)
uv run marimo check notebooks/02_experiments.py       # only the two known empty-cell warnings
```

**Check B, the defs of a full run** (which cells ran). `NAMES` is the sub-task's list:
```sh
PYTHONPATH=src .venv/bin/python -W error - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("nb", "notebooks/02_experiments.py")
nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)
_, defs = nb.app.run()
NAMES = [...]
missing = [n for n in NAMES if n not in defs]
print("missing:", missing)
EOF
```

**Check C, the clicked run on a copy** (T1.6 onward). Build a scratch project: `<scratch>/project/notebooks/02_experiments.py`
= the notebook with `n_trials = 400` (200 before D20) replaced by `n_trials = 3`, and `<scratch>/project/configs`
a symlink to the repo's `configs` (the notebook reads the config relative to its own directory;
`src` comes from `PYTHONPATH`). Then:
```sh
PYTHONPATH=src .venv/bin/python -W error - <<'EOF'
import importlib.util, types
from unittest import mock
path = "<scratch>/project/notebooks/02_experiments.py"
spec = importlib.util.spec_from_file_location("nb", path)
nb = importlib.util.module_from_spec(spec)
with mock.patch("marimo.ui.run_button", return_value=types.SimpleNamespace(value=True)):
    spec.loader.exec_module(nb)
    _, defs = nb.app.run()
for name in [...]:        # the sub-task's names, e.g. "MODEL1_RUNS", "model1_df"
    print(name, name in defs)
print(defs["model1_df"].to_string(index=False))
EOF
```
(Verified on 2026-10-15 on a scratch app: a button's value is `False` in a script run, and the
patch makes `mo.stop` pass.) `optuna` logs failed trials at WARNING to stderr; that is expected
for NB2 and is not a failure of the check.

## 2. T1: section 5 for Model 1 (notebook only)

The old section 5 (every cell from the `## 5) Hyperparameter tuning` markdown cell to the
best-per-model cell, lines 598–1218) is replaced. T1.1 removes all of it and writes the first
cells; the later sub-tasks append cells. Between sub-tasks the notebook always runs.

### [x] T1.1 Header, imports, metric and constants

Delete the old section-5 cells (keep the user's empty cell at lines 593–595 and the
`if __name__` block). Write four cells:

**Cell 5.1, header** (`hide_code=True`):
```python
mo.md(r"""
## 5) Hyperparameter tuning

One Optuna study per model family and exposure (on or off, fixed for the
study), on the composite: the model that is scored is the model that is
tuned, by the cohort Poisson deviance of its cross-validation folds on the
training part, the plain mean over folds. A study's fixed settings are a
frozen dataclass, its component; a trial samples per cohort (per component
for Model 2) the feature transformer by name and the estimator's
hyperparameters, under the names `<component>__<parameter>`, and builds
the estimator, its `CountModel` and the composite with constructor calls.
LightGBM's two losses, Poisson and Gaussian, are separate families, and
the exposure a study setting: a study that sampled the loss split 30
trials 10/20 and did worse than either loss alone, and the exposure
changes the target (a count or a weighted rate). `n_trials = 200`, so
that no categorical arm starves; the table of trials per arm shows it.
The i-th study gets `seed + i`; the studies of one model run
in parallel with joblib. A trial whose fit raises (an NB2 fit that does not
converge) is marked failed and the study goes on; the tables count them.
Only each study's best run is rebuilt, refit on the training part and
scored on training and validation. A Gaussian loss can predict a negative
count, which `CountModel` floors at `y_min` (eps). Model 2 is tuned the
same way, one study per total family (the list `MODEL2_TOTALS`, edited
after Model 1's table), the total's exposure and a classifier family,
under `total__…` and `probability__…`; the probability model ignores the
exposure, and its calibration is searched. The last table shows each model's best family on
validation. The run cells sit behind buttons: opening the notebook fits
nothing.
""")
```

**Cell 5.2, imports:**
```python
import dataclasses
import time
from collections import Counter
from functools import partial

import numpy as np
import optuna
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from optuna.distributions import CategoricalDistribution
from optuna.samplers import TPESampler
from optuna.trial import FixedTrial, TrialState
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.modeling import (
    CohortProbabilityModel,
    CountModel,
    IndependentCohortModels,
    NegativeBinomialRegressor,
    TotalTimesProbabilityModel,
)
from age_group_prediction.scoring import Metric, cohort_poisson_deviance

# Optuna logs every trial at INFO; the tuning function repeats this in
# each joblib worker.
optuna.logging.set_verbosity(optuna.logging.WARNING)
```
Return every imported name (marimo trims the list to what is read; re-add as cells arrive).
Also remove `from sklearn.base import clone` and `clone` from the section-1 imports cell
(lines 47 and 74–98): nothing reads it (ruff F401 today).

**Cell 5.3, the metrics:**
```python
# The study's objective and the tables' per-cohort score, with the direction
# Optuna is given; a loss, so "minimize".
COHORT_POISSON_DEVIANCE = Metric(
    "cohort_poisson_deviance", cohort_poisson_deviance, direction="minimize"
)
POISSON_DEVIANCE = Metric("poisson_deviance", mean_poisson_deviance, direction="minimize")
```

**Cell 5.4, constants:**
```python
COHORTS = list(Y.columns)
# The smallest mean a CountModel predicts: a Gaussian loss can predict a
# negative count, the Poisson deviance raises at 0 and stays finite at eps.
PREDICTION_FLOOR = float(np.finfo(float).eps)
# The trees take n_apartments as a feature and the offset; the GLMs take it
# as the exposure only (docs/FEATURE_TRANSFORMATIONS.md §3.3, §4.4), and the
# logistic model needs the scaled plans (it does not converge on `tree`).
TREE_TRANSFORMERS = ("tree", "tree__ses_quadratic")
COUNT_GLM_TRANSFORMERS = (
    "glm_base",
    "count__ses_quadratic",
    "count__daycare_log1p",
    "count__room_share_x_household_size",
    "count__room_share_x_ses",
    "count__room_share_x_household_size_and_ses",
)
PROBABILITY_GLM_TRANSFORMERS = (
    "glm_base",
    "probability__daycare_log1p",
    "probability__room_share_x_median_age",
    "probability__room_share_x_daycare",
    "probability__room_share_x_median_age_and_daycare",
)
# Plain assignments, not the `type` statement: cloudpickle cannot pickle a
# class annotated with a `type` alias defined in a cell, and joblib's
# workers receive the components by cloudpickle.
CountFamily = Literal["lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2"]
ProbabilityFamily = Literal["logistic", "lgbm"]
COUNT_FAMILIES = get_args(CountFamily)
PROBABILITY_FAMILIES = get_args(ProbabilityFamily)
# The candidate feature transformers of each family, given to its components.
COUNT_FAMILY_TRANSFORMERS = {
    "lgbm_poisson": TREE_TRANSFORMERS,
    "lgbm_gaussian": TREE_TRANSFORMERS,
    "poisson_glm": COUNT_GLM_TRANSFORMERS,
    "nb2": COUNT_GLM_TRANSFORMERS,
}
PROBABILITY_FAMILY_TRANSFORMERS = {
    "logistic": PROBABILITY_GLM_TRANSFORMERS,
    "lgbm": TREE_TRANSFORMERS,
}
```

**Done when:** no `hyperparameter_tuning` string in the notebook; Check A passes (test green,
ruff 8 findings, marimo check clean; the `tune_model` test fails as noted at T1.5); Check B with `NAMES = ["COHORT_POISSON_DEVIANCE",
"POISSON_DEVIANCE", "COHORTS", "PREDICTION_FLOOR", "TREE_TRANSFORMERS", "COUNT_FAMILIES",
"PROBABILITY_FAMILIES"]` prints `missing: []`; md5 recorded.

### [x] T1.2 The estimator dataclasses

(Superseded by D10 on 2026-10-21; the current code is in T1.4.)
One cell. An estimator dataclass holds only its estimator's searched hyperparameters, and its
fields are exactly the estimator's keyword names, so T1.3 builds an estimator as
`Estimator(<fixed settings>, **dataclasses.asdict(params))`. The bounds are the old section 5's
(`8ee091a`, lines 703–725). The regressor and the classifier share `LGBMParams`'s ten boosting
fields. `LGBMRegressorParams` adds `objective`, which the family fixes (D6), passed through
`suggest`'s `**fixed`: `LGBMRegressorParams.suggest(trial, prefix, objective="poisson")`. The
classifier's `multiclass` is not searched. (Reworked at the user's review: the first version
had one `LGBMParams` with `build_regressor`/`build_classifier`, and a `build` on each class.
The subclass does not override `suggest`: an override narrowing `**fixed` to `objective` is a
Liskov violation the IDE flagged.)
```python
@dataclasses.dataclass(frozen=True)
class LGBMParams:
    """LightGBM's searched hyperparameters, shared by the regressor and the classifier."""

    max_depth: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    n_estimators: int
    reg_alpha: float
    reg_lambda: float
    min_split_gain: float
    subsample: float
    colsample_bytree: float

    @classmethod
    def suggest(cls, trial, prefix, **fixed):
        return cls(
            max_depth=trial.suggest_int(f"{prefix}max_depth", 3, 8),
            num_leaves=trial.suggest_int(f"{prefix}num_leaves", 7, 63),
            min_child_samples=trial.suggest_int(
                f"{prefix}min_child_samples", 5, 40
            ),
            learning_rate=trial.suggest_float(
                f"{prefix}learning_rate", 0.01, 0.2, log=True
            ),
            n_estimators=trial.suggest_int(f"{prefix}n_estimators", 50, 400),
            reg_alpha=trial.suggest_float(
                f"{prefix}reg_alpha", 1e-8, 10.0, log=True
            ),
            reg_lambda=trial.suggest_float(
                f"{prefix}reg_lambda", 1e-8, 10.0, log=True
            ),
            min_split_gain=trial.suggest_float(f"{prefix}min_split_gain", 0.0, 1.0),
            subsample=trial.suggest_float(f"{prefix}subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                f"{prefix}colsample_bytree", 0.7, 1.0
            ),
            **fixed,
        )

@dataclasses.dataclass(frozen=True)
class LGBMRegressorParams(LGBMParams):
    # LightGBM's name for the loss, "poisson" or "regression" (Gaussian):
    # fixed by the family and recorded, not sampled; `suggest` takes it as
    # a keyword, `suggest(trial, prefix, objective="poisson")`.
    objective: str

@dataclasses.dataclass(frozen=True)
class LGBMClassifierParams(LGBMParams):
    """The classifier's; its objective, multiclass, is not searched."""

@dataclasses.dataclass(frozen=True)
class PoissonGLMParams:
    alpha: float

    @classmethod
    def suggest(cls, trial, prefix):
        return cls(
            alpha=trial.suggest_float(f"{prefix}alpha", 1e-4, 10.0, log=True)
        )

@dataclasses.dataclass(frozen=True)
class NB2Params:
    # The estimated dispersions are 0.12 to 0.18; the penalty's path runs
    # from the unpenalized fit at 1e-3 to all-zero coefficients at 1.
    dispersion: float
    alpha: float

    @classmethod
    def suggest(cls, trial, prefix):
        return cls(
            dispersion=trial.suggest_float(
                f"{prefix}dispersion", 0.01, 1.0, log=True
            ),
            alpha=trial.suggest_float(f"{prefix}alpha", 1e-3, 1.0, log=True),
        )

@dataclasses.dataclass(frozen=True)
class LogisticParams:
    C: float

    @classmethod
    def suggest(cls, trial, prefix):
        return cls(C=trial.suggest_float(f"{prefix}C", 1e-3, 100.0, log=True))
```
The IDE's checker reports `object.__init__` errors on these classes. That is an artifact: in a
marimo cell `dataclasses` is a parameter, so the checker cannot see the decorator.

**Done when:** Check A; Check B with the six class names and `COUNT_FAMILIES`; a probe builds
each class from `suggest(FixedTrial({...}), "x__")`, prints `json.dumps(dataclasses.asdict(p))`
(hyperparameters only), checks that `LGBMRegressor(**asdict(r))` takes `objective`, that a
rebuild is equal, that the classifier's params are not `LGBMRegressorParams`, and that a
missing `objective` raises `TypeError`.

### [x] T1.3 The component dataclasses

(Superseded by D10 on 2026-10-21; the current code is in T1.4.)
(Reworked at T1.4's review, 2026-10-20, D8: the cell's current code is in T1.4.)
One cell. (Redesigned 2026-10-18 at the user's review. The first version held the sampled
choices and built the model; see D3 and D6.) A component holds only what is fixed for one
study: the family (a `Literal`) and, for a count component, the exposure. Given the family it
samples its params class under a prefix, and its feature transformer from its own entry of
the family → names mapping passed in (D3: the classes read no notebook global). An unknown
family raises at construction, a family missing from the mapping when sampling. A frozen
component is hashable, so it keys the studies; it pickles through cloudpickle to the joblib
workers.
```python
# The components read no notebook global: the family's candidate feature
# transformers come in as an argument, so the classes can move to a package.
@dataclasses.dataclass(frozen=True)
class CountComponent:
    """A CountModel's settings fixed for one study; a trial samples the rest under a prefix."""

    family: CountFamily
    use_exposure: bool

    def __post_init__(self):
        # At construction, before the parallel studies start.
        if self.family not in get_args(CountFamily):
            raise ValueError(f"unknown count family {self.family!r}")

    def suggest_feature_transformer(self, trial, prefix, family_transformers):
        """The name of a feature transformer, from ``family_transformers[self.family]``.

        ``family_transformers`` maps each family to the names of the feature
        transformers that suit it, so a family is never given another's.
        """
        if self.family not in family_transformers:
            raise ValueError(f"no feature transformers for family {self.family!r}")
        return trial.suggest_categorical(
            f"{prefix}feature_transformer", list(family_transformers[self.family])
        )

    def suggest_params(self, trial, prefix):
        if self.family == "lgbm_poisson":
            return LGBMRegressorParams.suggest(trial, prefix, objective="poisson")
        elif self.family == "lgbm_gaussian":
            return LGBMRegressorParams.suggest(
                trial, prefix, objective="regression"
            )
        elif self.family == "poisson_glm":
            return PoissonGLMParams.suggest(trial, prefix)
        elif self.family == "nb2":
            return NB2Params.suggest(trial, prefix)
        else:
            raise ValueError(f"unknown count family {self.family!r}")

@dataclasses.dataclass(frozen=True)
class ProbabilityComponent:
    """Model 2's CohortProbabilityModel settings fixed for one study; the exposure does not apply."""

    family: ProbabilityFamily

    def __post_init__(self):
        if self.family not in get_args(ProbabilityFamily):
            raise ValueError(f"unknown probability family {self.family!r}")

    def suggest_feature_transformer(self, trial, prefix, family_transformers):
        """As CountComponent's."""
        if self.family not in family_transformers:
            raise ValueError(f"no feature transformers for family {self.family!r}")
        return trial.suggest_categorical(
            f"{prefix}feature_transformer", list(family_transformers[self.family])
        )

    def suggest_params(self, trial, prefix):
        if self.family == "lgbm":
            return LGBMClassifierParams.suggest(trial, prefix)
        elif self.family == "logistic":
            return LogisticParams.suggest(trial, prefix)
        else:
            raise ValueError(f"unknown probability family {self.family!r}")

    def suggest_calibration_method(self, trial, prefix):
        # No isotonic: its lowest score block can hold no child of a cohort and
        # then maps a building there to probability 0, an infinite log loss
        # (measured 2026-10-15 with the LightGBM classifier).
        return trial.suggest_categorical(
            f"{prefix}calibration_method", [None, "temperature", "sigmoid"]
        )
```

**Done when:** Check A; Check B with both class names, the two `Literal`s and the two mappings; a probe, for every
count family × exposure and both probability families, runs a 2-trial study whose objective
calls the `suggest_*` methods under `"a__"`. It checks four things: the sampled names are
exactly the family's; the transformer is in `family_transformers[family]`; a `FixedTrial(trial.params)`
rebuild is equal; and `json.dumps(asdict(component))` works. Unknown families, and a mapping
without the family, raise `ValueError`; components are equal and hash by value; a cloudpickle round trip
and a `Parallel(n_jobs=2)` pass work.

### [x] T1.4 The models a trial builds

Done 2026-10-19, reworked 2026-10-20 (D8), redesigned 2026-10-21 (D10, revised by D11 and D12):
- the params dataclasses are fields only;
- a component is a complete record that makes its estimator;
- a search space offers explicit suggest methods;
- the builders assemble the components from a trial's draws and turn them into models.

This section holds the current code of the four cells.

**Params:**
```python
# Hyperparameters only, no Optuna: the field names are the estimator's
# keywords, so a component makes its estimator as
# Estimator(<fixed settings>, **asdict(params)).
@dataclasses.dataclass(frozen=True, kw_only=True)
class LGBMParams:
    """LightGBM's searched hyperparameters, for the regressor and the classifier alike."""

    max_depth: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    n_estimators: int
    reg_alpha: float
    reg_lambda: float
    min_split_gain: float
    subsample: float
    colsample_bytree: float

@dataclasses.dataclass(frozen=True, kw_only=True)
class PoissonGLMParams:
    alpha: float

@dataclasses.dataclass(frozen=True, kw_only=True)
class NB2Params:
    dispersion: float
    alpha: float

@dataclasses.dataclass(frozen=True, kw_only=True)
class LogisticParams:
    C: float
```

**Components:**
```python
# Complete records of one part of a run, built by hand or from a trial's
# draws; no Optuna here (D10, D11). They read no notebook global, so the
# classes can move to a package.
@dataclasses.dataclass(frozen=True, kw_only=True)
class Component(abc.ABC):
    """A named part of a composite: its family, feature transformer and hyperparameters."""

    name: str
    family: str
    # A key of FEATURE_TRANSFORMERS, resolved by the builder, so the record
    # stays JSON.
    feature_transformer_name: str

    def __post_init__(self):
        # The subclass annotates `family` with its Literal, which holds the
        # valid names.
        families = get_args(get_type_hints(type(self))["family"])
        if self.family not in families:
            raise ValueError(
                f"unknown {type(self).__name__} family {self.family!r}"
            )

    @abc.abstractmethod
    def make_estimator(self, *, seed):
        """The unfitted estimator: the family's fixed settings and ``hyperparameters``."""

@dataclasses.dataclass(frozen=True, kw_only=True)
class CountComponent(Component):
    """A CountModel's part of a run; ``use_exposure`` makes the target a weighted rate."""

    family: CountFamily
    use_exposure: bool
    hyperparameters: LGBMParams | PoissonGLMParams | NB2Params

    def make_estimator(self, *, seed):
        hyperparameters = dataclasses.asdict(self.hyperparameters)
        if self.family in ("lgbm_poisson", "lgbm_gaussian"):
            # LightGBM's name for the loss, which the family fixes (D6).
            objective = "poisson" if self.family == "lgbm_poisson" else "regression"
            # subsample_freq=1: LightGBM ignores subsample without it; n_jobs=1: one
            # thread per fit, so the parallel studies do not oversubscribe the cores.
            return LGBMRegressor(
                objective=objective,
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
                **hyperparameters,
            )
        elif self.family == "poisson_glm":
            return PoissonRegressor(max_iter=1000, **hyperparameters)
        elif self.family == "nb2":
            return NegativeBinomialRegressor(**hyperparameters)
        else:
            raise ValueError(f"unknown count family {self.family!r}")

@dataclasses.dataclass(frozen=True, kw_only=True)
class ProbabilityComponent(Component):
    """Model 2's CohortProbabilityModel part of a run; the exposure does not apply."""

    family: ProbabilityFamily
    calibration_method: str | None
    hyperparameters: LGBMParams | LogisticParams

    def make_estimator(self, *, seed):
        hyperparameters = dataclasses.asdict(self.hyperparameters)
        if self.family == "lgbm":
            return LGBMClassifier(
                objective="multiclass",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
                **hyperparameters,
            )
        elif self.family == "logistic":
            return LogisticRegression(max_iter=1000, **hyperparameters)
        else:
            raise ValueError(f"unknown probability family {self.family!r}")
```

**Search spaces:**
```python
# Every Optuna call (D10, D11): a search space, given a component's name and
# family, samples its choices from a trial under `<name>__<parameter>`; the
# caller feeds them into the component.
@dataclasses.dataclass(frozen=True, kw_only=True)
class SearchSpace(abc.ABC):
    """The search of one component's choices, given its name and family."""

    name: str
    family: str
    # Set by each subclass: each family's candidate feature transformers, by
    # name. Fixed on the class, so no search space carries another family's.
    FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]]

    def __post_init__(self):
        # At construction, before the parallel studies start.
        families = get_args(get_type_hints(type(self))["family"])
        if self.family not in families:
            raise ValueError(
                f"unknown {type(self).__name__} family {self.family!r}"
            )
        if self.family not in self.FEATURE_TRANSFORMER_NAMES:
            raise ValueError(f"no feature transformers for family {self.family!r}")

    @property
    def prefix(self):
        """The trial's parameter names are ``<name>__<parameter>``."""
        return f"{self.name}__"

    def suggest_feature_transformer_name(self, trial):
        """A name from the family's candidates; Optuna's choices must be primitives."""
        return trial.suggest_categorical(
            f"{self.prefix}feature_transformer",
            list(self.FEATURE_TRANSFORMER_NAMES[self.family]),
        )

    @abc.abstractmethod
    def suggest_hyperparameters(self, trial):
        """The family's params record, sampled under ``prefix``."""

    def _suggest_lgbm(self, trial):
        prefix = self.prefix
        return LGBMParams(
            max_depth=trial.suggest_int(f"{prefix}max_depth", 3, 8),
            num_leaves=trial.suggest_int(f"{prefix}num_leaves", 7, 63),
            min_child_samples=trial.suggest_int(
                f"{prefix}min_child_samples", 5, 40
            ),
            learning_rate=trial.suggest_float(
                f"{prefix}learning_rate", 0.01, 0.2, log=True
            ),
            n_estimators=trial.suggest_int(f"{prefix}n_estimators", 50, 400),
            reg_alpha=trial.suggest_float(
                f"{prefix}reg_alpha", 1e-8, 10.0, log=True
            ),
            reg_lambda=trial.suggest_float(
                f"{prefix}reg_lambda", 1e-8, 10.0, log=True
            ),
            min_split_gain=trial.suggest_float(f"{prefix}min_split_gain", 0.0, 1.0),
            subsample=trial.suggest_float(f"{prefix}subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                f"{prefix}colsample_bytree", 0.7, 1.0
            ),
        )

@dataclasses.dataclass(frozen=True, kw_only=True)
class CountSearchSpace(SearchSpace):
    """A count component's search: its hyperparameters and feature transformer."""

    family: CountFamily
    # The trees take n_apartments as a feature and the offset; the GLMs take
    # it as the exposure only (docs/FEATURE_TRANSFORMATIONS.md §3.3, §4.4).
    FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]] = (
        MappingProxyType(
            {
                **dict.fromkeys(
                    ("lgbm_poisson", "lgbm_gaussian"),
                    ("tree", "tree__ses_quadratic"),
                ),
                **dict.fromkeys(
                    ("poisson_glm", "nb2"),
                    (
                        "glm_base",
                        "count__ses_quadratic",
                        "count__daycare_log1p",
                        "count__room_share_x_household_size",
                        "count__room_share_x_ses",
                        "count__room_share_x_household_size_and_ses",
                    ),
                ),
            }
        )
    )

    def suggest_hyperparameters(self, trial):
        prefix = self.prefix
        if self.family in ("lgbm_poisson", "lgbm_gaussian"):
            return self._suggest_lgbm(trial)
        elif self.family == "poisson_glm":
            return PoissonGLMParams(
                alpha=trial.suggest_float(f"{prefix}alpha", 1e-4, 10.0, log=True)
            )
        elif self.family == "nb2":
            # The estimated dispersions are 0.12 to 0.18; the penalty's path runs
            # from the unpenalized fit at 1e-3 to all-zero coefficients at 1.
            return NB2Params(
                dispersion=trial.suggest_float(
                    f"{prefix}dispersion", 0.01, 1.0, log=True
                ),
                alpha=trial.suggest_float(f"{prefix}alpha", 1e-3, 1.0, log=True),
            )
        else:
            raise ValueError(f"unknown count family {self.family!r}")

@dataclasses.dataclass(frozen=True, kw_only=True)
class ProbabilitySearchSpace(SearchSpace):
    """A probability component's search: its hyperparameters, calibration and feature transformer."""

    family: ProbabilityFamily
    # The logistic model needs the scaled plans (it does not converge on `tree`).
    FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]] = (
        MappingProxyType(
            {
                "logistic": (
                    "glm_base",
                    "probability__daycare_log1p",
                    "probability__room_share_x_median_age",
                    "probability__room_share_x_daycare",
                    "probability__room_share_x_median_age_and_daycare",
                ),
                "lgbm": ("tree", "tree__ses_quadratic"),
            }
        )
    )

    def suggest_hyperparameters(self, trial):
        if self.family == "lgbm":
            return self._suggest_lgbm(trial)
        elif self.family == "logistic":
            return LogisticParams(
                C=trial.suggest_float(f"{self.prefix}C", 1e-3, 100.0, log=True)
            )
        else:
            raise ValueError(f"unknown probability family {self.family!r}")

    def suggest_calibration_method(self, trial):
        # No isotonic: its lowest score block can hold no child of a cohort and
        # then maps a building there to probability 0, an infinite log loss
        # (measured 2026-10-15 with the LightGBM classifier).
        return trial.suggest_categorical(
            f"{self.prefix}calibration_method", [None, "temperature", "sigmoid"]
        )
```

**Builders** (D12 since 2026-10-21):
```python
# The suggest functions call a search space's methods explicitly and feed the
# draws into a component; the build functions turn components into models,
# with no Optuna. A study is one family combination and exposure (D12): the
# objective calls suggest_model1/2, and the best run's refit rebuilds the best
# components with suggest_model1/2_components on FixedTrial(best_params).
def suggest_count_component(trial, space, *, use_exposure):
    """The count component of ``trial``'s draws in ``space``; the exposure is the study's."""
    # Keyword arguments are evaluated left to right, so this is the order of
    # the draws: the hyperparameters, then the transformer. A seeded study's
    # result depends on it.
    return CountComponent(
        name=space.name,
        family=space.family,
        use_exposure=use_exposure,
        hyperparameters=space.suggest_hyperparameters(trial),
        feature_transformer_name=space.suggest_feature_transformer_name(trial),
    )

def suggest_probability_component(trial, space):
    """The probability component of ``trial``'s draws in ``space``."""
    # The draws, in order: the hyperparameters, the calibration, the transformer.
    return ProbabilityComponent(
        name=space.name,
        family=space.family,
        hyperparameters=space.suggest_hyperparameters(trial),
        calibration_method=space.suggest_calibration_method(trial),
        feature_transformer_name=space.suggest_feature_transformer_name(trial),
    )

def suggest_model1_components(trial, *, family, use_exposure, cohorts):
    """Model 1's components ``trial`` draws: one per cohort, all of ``family``."""
    return tuple(
        suggest_count_component(
            trial,
            CountSearchSpace(name=cohort, family=family),
            use_exposure=use_exposure,
        )
        for cohort in cohorts
    )

def suggest_model2_components(
    trial, *, total_family, probability_family, use_exposure
):
    """Model 2's total and probability components ``trial`` draws, the total first."""
    total = suggest_count_component(
        trial,
        CountSearchSpace(name="total", family=total_family),
        use_exposure=use_exposure,
    )
    probability = suggest_probability_component(
        trial, ProbabilitySearchSpace(name="probability", family=probability_family)
    )
    return total, probability

def make_count_model(component, *, feature_transformer_dict, y_min, seed):
    """The CountModel of a count component."""
    # CountModel clones the transformer at fit, so the shared declaration is safe.
    return CountModel(
        estimator=component.make_estimator(seed=seed),
        use_exposure=component.use_exposure,
        feature_transformer=feature_transformer_dict[
            component.feature_transformer_name
        ],
        y_min=y_min,
    )

def build_model1(components, *, feature_transformer_dict, y_min, seed):
    """Model 1: one CountModel per component, keyed by the component's name, a cohort."""
    return IndependentCohortModels(
        {
            component.name: make_count_model(
                component,
                feature_transformer_dict=feature_transformer_dict,
                y_min=y_min,
                seed=seed,
            )
            for component in components
        }
    )

def build_model2(total, probability, *, feature_transformer_dict, y_min, seed):
    """Model 2: the total's CountModel and the cohort probabilities' classifier."""
    return TotalTimesProbabilityModel(
        total_model=make_count_model(
            total,
            feature_transformer_dict=feature_transformer_dict,
            y_min=y_min,
            seed=seed,
        ),
        probability_model=CohortProbabilityModel(
            estimator=probability.make_estimator(seed=seed),
            calibration_method=probability.calibration_method,
            feature_transformer=feature_transformer_dict[
                probability.feature_transformer_name
            ],
        ),
    )

def suggest_model1(
    trial, *, family, use_exposure, cohorts, feature_transformer_dict, y_min, seed
):
    """Model 1 of the components ``trial`` draws for one family and exposure."""
    components = suggest_model1_components(
        trial, family=family, use_exposure=use_exposure, cohorts=cohorts
    )
    return build_model1(
        components,
        feature_transformer_dict=feature_transformer_dict,
        y_min=y_min,
        seed=seed,
    )

def suggest_model2(
    trial,
    *,
    total_family,
    probability_family,
    use_exposure,
    feature_transformer_dict,
    y_min,
    seed,
):
    """Model 2 of the components ``trial`` draws for one family pair and exposure."""
    total, probability = suggest_model2_components(
        trial,
        total_family=total_family,
        probability_family=probability_family,
        use_exposure=use_exposure,
    )
    return build_model2(
        total,
        probability,
        feature_transformer_dict=feature_transformer_dict,
        y_min=y_min,
        seed=seed,
    )
```

**Done when** (D11, met 2026-10-21):
- Check A green (5/5).
- Check B: the new names are defined. Absent: `MODEL1_SEARCH_SPACES`, D10's `suggest`
  methods and `PARAMS`. A search space's fields are `name` and `family`.
- Per family, the trial names are as before (11, 11, 2, 3, 12, 3).
- The best components of a real NB2 study rebuild from `FixedTrial`, serialize to JSON and
  fit (best 2.2716, unchanged).
- A component built by hand makes its estimator.
- Failures raise:
  - an unknown family, in a component or a search space (`ValueError`);
  - a LightGBM typo in `LGBMParams`, or `use_exposure` given to a search space
    (`TypeError`).
- Hashing, cloudpickle and `Parallel` work.
- A Model 2 pair builds and fits (14 trial names).
- Check C at 3 trials reproduces T1.6's 8 best values exactly (13.8 s).

### [x] T1.5 `cv_score` and `tune_model`

One cell: the user's fold variables, the aggregator of D9 and the parallel folds of D13
(2026-10-21):
```python
def fold_score(model, X, Y, exposure, train_index, val_index, metric):
    """The score on one fold's validation rows of ``model`` fitted on its training rows."""
    X_train, Y_train, exposure_train = (
        X.iloc[train_index],
        Y.iloc[train_index],
        exposure.iloc[train_index],
    )
    X_val, Y_val, exposure_val = (
        X.iloc[val_index],
        Y.iloc[val_index],
        exposure.iloc[val_index],
    )
    model.fit(X_train, Y_train, exposure=exposure_train)
    Y_val_pred = model.predict(X_val, exposure=exposure_val)
    return model.evaluate(Y_val, Y_val_pred, metric)

def cv_score(
    model, X, Y, exposure, cv, cv_groups, metric, aggregator=None, n_jobs=1
):
    """The aggregated fold score of ``model`` refit on each fold's training rows (``fit`` replaces all state).

    ``aggregator`` combines the fold scores; None means the plain mean.
    ``n_jobs`` processes fit the folds in parallel; the scores are the
    same as one after another, so a seeded study is unchanged.
    """
    # The aggregator decides the selection (D9); the plain mean by default,
    # since the folds differ by one row at most.
    aggregator = Mean() if aggregator is None else aggregator
    folds = list(cv.split(X, Y, cv_groups))
    scores = Parallel(n_jobs=n_jobs)(
        delayed(fold_score)(model, X, Y, exposure, train_index, val_index, metric)
        for train_index, val_index in folds
    )
    fold_sizes = [len(val_index) for _, val_index in folds]
    return aggregator.aggregate(scores, fold_sizes)

def tune_model(
    suggest_model,
    X,
    Y,
    exposure,
    cv,
    cv_groups,
    metric,
    *,
    study_seed,
    n_trials,
    aggregator=None,
    n_jobs=1,
):
    """One study of the model ``suggest_model(trial)`` builds, on the rows of ``X``; the study and its seconds.

    A trial whose fit raises RuntimeError (an NB2 fit that does not
    converge, re-raised as such from a fold's process) is marked FAIL and
    the study goes on; without ``catch`` one such trial would end the
    study. ``aggregator`` and ``n_jobs`` go to ``cv_score``.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Optuna passes the trial alone; the data is bound here, not to a cell.
        return cv_score(
            suggest_model(trial),
            X,
            Y,
            exposure,
            cv,
            cv_groups,
            metric,
            aggregator,
            n_jobs,
        )

    study = optuna.create_study(
        direction=metric.direction, sampler=TPESampler(seed=study_seed)
    )
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
    return study, time.perf_counter() - start
```
**Done when:** Check A (the test `test_a_tuning_cell_outside_a_run_button_is_caught` fails from
T1.1 until T1.6, since no cell names `tune_model` behind a button yet; the other four tests of
the file must pass; say so in the report); Check B; a probe calls `tune_model` directly with `n_trials=2` for
`partial(suggest_model1, family="poisson_glm", use_exposure=True, cohorts=COHORTS, …)` (D12's names) on the notebook's training data and prints `study.best_value` and the seconds (2.48456 by default and with `Mean()`; met 2026-10-21, with the other aggregators of D9).

### [x] T1.6 Model 1's button and run cells

The button cell, then eight run cells (D14 since 2026-10-21), one per `(family,
use_exposure)` in the order `lgbm_poisson`, `lgbm_gaussian`, `poisson_glm`, `nb2` × exposure
on, off. They are seeded `seed + 0` … `seed + 7`, and each stops on the button. The button:
```python
run_model1 = mo.ui.run_button(label="Tune Model 1")
mo.md(
    f"{run_model1} 8 runs, one cell each (a count family × exposure on or off); "
    "each trial fits its folds in parallel."
)
```
Run 1 of 8; the other seven differ only in the comment, `_family, _use_exposure`, the seed
offset and the variable name:
```python
mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
# Model 1, run 1 of 8: LightGBM with the Poisson loss, exposure on.
_family, _use_exposure = "lgbm_poisson", True
_study, _seconds = tune_model(
    partial(
        suggest_model1,
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
        feature_transformer_dict=FEATURE_TRANSFORMERS,
        y_min=PREDICTION_FLOOR,
        seed=seed,
    ),
    train_age_group.X,
    train_age_group.Y,
    train_age_group.exposure,
    cv,
    cv_groups,
    COHORT_POISSON_DEVIANCE,
    study_seed=seed + 0,
    n_trials=n_trials,
    n_jobs=n_splits,
)
# The best components, rebuilt from the best trial's parameters, are the
# model scored and the record kept (D10-D12, D14).
_best = suggest_model1_components(
    FixedTrial(_study.best_params),
    family=_family,
    use_exposure=_use_exposure,
    cohorts=COHORTS,
)
_, _score_table = fit_and_score(
    build_model1(
        _best,
        feature_transformer_dict=FEATURE_TRANSFORMERS,
        y_min=PREDICTION_FLOOR,
        seed=seed,
    ),
    train_age_group,
    val_age_group,
    cohort_metric=POISSON_DEVIANCE,
    metric=COHORT_POISSON_DEVIANCE,
)
# Every trial and the best (D14); the study object is not kept.
model1_lgbm_poisson_exposure = {
    "family": _family,
    "use_exposure": _use_exposure,
    "trials": _study.trials_dataframe(),
    "components": [dataclasses.asdict(c) for c in _best],
    "cv_best": _study.best_value,
    "score_table": _score_table,
    "seconds": _seconds,
    "seed": seed,
    "y_min": PREDICTION_FLOOR,
}
```

**Done when:** Check A fully green (the `tune_model` rule holds again); Check C with
`n_trials = 3` prints `MODEL1_STUDIES True` (since D13 `MODEL1_TUNING`) and, for each component, `len(study.trials) == 3`;
record the wall time of Check C.

Met 2026-10-21:
- Check A green (the notebook test 5/5).
- Check B: `MODEL1_COMPONENTS` (since D11 `MODEL1_STUDY_SETTINGS`) and `run_model1` are defined, and `MODEL1_STUDIES` is not,
  because the script run stops at the button.
- Check C: all 8 studies at 3 trials, no FAIL. The whole notebook run takes 14.7 s; the
  studies take 1.7 s (`poisson_glm`, exposure on) to 7.1 s (`nb2`, exposure off). The best
  values equal the planning probe's, so the seeded run reproduces:

  | Family | Exposure on | Exposure off |
  |---|---|---|
  | `lgbm_poisson` | 2.190 | 2.281 |
  | `lgbm_gaussian` | 2.191 | 2.315 |
  | `poisson_glm` | 2.570 | 2.878 |
  | `nb2` | 2.283 | 2.890 |

- At NB2's pace of about 2 s a trial, a 200-trial run should take roughly 7 minutes; the
  user's click (T1.7) measures it.

### [x] T1.7 Refit, tables and the Model 1 views

The refit lives in each run cell (D14). This sub-task's cells are the function, the tables
cell and the display cell.
- **History:** implemented 2026-10-21. The reviews that day gave D13 (the runs one after
  another, the folds in parallel) and D14 (one cell per run, every trial kept).
- **Kept from the first version:** the best components, rebuilt from
  `FixedTrial(best_params)`, are both the model scored and the record.
```python
def fit_and_score(model, train_age_group, val_age_group, *, cohort_metric, metric):
    """Fit ``model`` on the training part; it and its score table on training and validation.

    A row per cohort column, scored by ``cohort_metric``, then the
    composite, scored by ``metric``.
    """
    model.fit(
        train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure
    )
    y_pred_train = model.predict(
        train_age_group.X, exposure=train_age_group.exposure
    )
    y_pred_val = model.predict(val_age_group.X, exposure=val_age_group.exposure)
    score_table = [
        {
            "cohort": cohort,
            "train": model.evaluate(
                train_age_group.Y[cohort], y_pred_train[cohort], cohort_metric
            ),
            "val": model.evaluate(
                val_age_group.Y[cohort], y_pred_val[cohort], cohort_metric
            ),
        }
        for cohort in train_age_group.Y.columns
    ]
    score_table.append(
        {
            "cohort": "composite",
            "train": model.evaluate(train_age_group.Y, y_pred_train, metric),
            "val": model.evaluate(val_age_group.Y, y_pred_val, metric),
        }
    )
    return model, score_table
```
Tables cell:
```python
MODEL1_RUNS = [
    model1_lgbm_poisson_exposure,
    model1_lgbm_poisson_no_exposure,
    model1_lgbm_gaussian_exposure,
    model1_lgbm_gaussian_no_exposure,
    model1_poisson_glm_exposure,
    model1_poisson_glm_no_exposure,
    model1_nb2_exposure,
    model1_nb2_no_exposure,
]
_score_rows, _arm_rows = [], []
for _run in MODEL1_RUNS:
    _settings = {"family": _run["family"], "use_exposure": _run["use_exposure"]}
    _trials = _run["trials"]
    _complete = _trials[_trials["state"] == "COMPLETE"]
    _score_rows.extend({**_settings, **row} for row in _run["score_table"][:-1])
    _score_rows.append(
        {
            **_settings,
            **_run["score_table"][-1],
            "cv_best": _run["cv_best"],
            "n_complete": len(_complete),
            "n_failed": int((_trials["state"] == "FAIL").sum()),
            "seconds": _run["seconds"],
        }
    )
    # The completed trials per feature transformer, Model 1's only categorical
    # choice: TPE concentrates on what looked good early, so a transformer
    # tried a few times has not been judged (D14).
    for _column in _trials.columns:
        if _column.endswith("__feature_transformer"):
            for _value, _n in _complete[_column].value_counts().items():
                _arm_rows.append(
                    {
                        **_settings,
                        "parameter": _column.removeprefix("params_"),
                        "value": _value,
                        "n_trials": int(_n),
                    }
                )
model1_df, model1_arms_df = pd.DataFrame(_score_rows), pd.DataFrame(_arm_rows)
```
Display cell:
```python
mo.vstack(
    [
        mo.md(
            "**Model 1: cohort Poisson deviance** (cv_best: the run's value; "
            "n_complete / n_failed: its trials)"
        ),
        mo.md(f"```text\n{model1_df.round(4).to_string(index=False)}\n```"),
        mo.md("**Trials per feature transformer**"),
        mo.md(f"```text\n{model1_arms_df.to_string(index=False)}\n```"),
    ]
)
```

**Done when:** Check A; Check C (met 2026-10-21, 13.6 s, D14):
- `MODEL1_RUNS` holds the 8 run records, each with a 3-row `trials` frame.
- `model1_df` has 32 rows with `family` and `use_exposure`.
- Every `cv_best` equals T1.6's.
- The per-transformer counts add up to each run's completed trials.
- The records serialize with `trials` as `to_dict("records")` and `json.dumps(…,
  default=str)` (the trial timestamps).

The 200-trial run: 380 s, no failed trial (D15). Then the user clicks the button in `marimo edit` at
`n_trials = 200` and reports the wall time (today's package version: ≈ 5 min).

### [x] T1.8 Independent review of T1

A review subagent (writes nothing into the repository) reads the new section against this
plan and the decisions; each finding is reproduced before it is acted on. Then the file-by-file
summary and the suggested commit command for the user.

Done 2026-10-21. The reviewer found no correctness bugs. It checked:
- the `FixedTrial` rebuild equals the scored value;
- parallel folds equal sequential ones;
- no leakage;
- the metric pairing, the exposure handling and Model 2's builders.

Its nine findings were all reproduced and settled:
- **F1:** an untried transformer now shows 0 (D15).
- **F2:** strict checks with `PYTHONWARNINGS=error`, and a note in the docstring (D15).
- **F3:** the section-5 header rewritten for D10–D14.
- **F4:** three stale "parallel studies" comments reworded.
- **F5:** the redundant `set_verbosity` removed from `tune_model`.
- **F6:** the unused `COUNT_FAMILIES` removed.
- **F7:** `num_leaves`, left as a known inefficiency (D15).
- **F8:** this document's status, ticks, a stray line, and T2.1's settings list.
- **F9:** the stale comment in the notebook test (the comment only).

## 3. T2: Model 2 and the best-per-model table (notebook only)

### [x] T2.1 Model 2's button and run cells
No `MODEL2_TOTALS` list (D6, changed 2026-10-21): the run cells present are the runs tuned.
`PROBABILITY_FAMILIES` went with it, since nothing read it.
Button:
```python
run_model2 = mo.ui.run_button(label="Tune Model 2")
mo.md(
    f"{run_model2} 16 runs, one cell each (a total family × exposure on or off "
    "× a probability family); each trial fits its folds in parallel."
)
```
Run cells: as T1.6 (D14), 16 of them, in the order total (`lgbm_poisson`, `lgbm_gaussian`,
`poisson_glm`, `nb2`) → exposure (on, off) → probability (`logistic`, `lgbm`), seeded
`seed + 0` … `seed + 15`. Each tunes `partial(suggest_model2, total_family=…,
probability_family=…, use_exposure=…, feature_transformer_dict=FEATURE_TRANSFORMERS,
y_min=PREDICTION_FLOOR, seed=seed)` with `n_jobs=n_splits`, rebuilds the best components with
`suggest_model2_components(FixedTrial(best_params), …)`, refits and scores them with
`build_model2(*_best, …)` and `fit_and_score`, and keeps the record
`model2_<total>_<probability>_[no_]exposure`: `total_family`, `probability_family`,
`use_exposure`, `trials`, `components` (`total`, `probability`), `cv_best`, `score_table`,
`seconds`, `seed`, `y_min`.
**Done when:** Check A; Check C gives 16 run records with 3 trials each.

Met 2026-10-21:
- Check A green (notebook test 5/5, ruff 8, the two known format hunks, `marimo check` the
  two known issues). Check B: `run_model2` defined, no `model2_*` record (the button).
- Planning probe (3 trials, exposure on, folds one after another): logistic pairs
  0.30–0.56 s a trial, LightGBM-classifier pairs 4.1–6.5 s.
- Check C under `PYTHONWARNINGS=error`, both buttons, 3 trials: 53.3 s for the whole
  notebook, no warning, no failed trial. Model 1's 8 `cv_best` equal the baseline. 16 Model 2
  records, each with 3 trials, a 4-row score table and the two components, each
  JSON-serializable. Model 2's Check C baseline (`cv_best`):

  | Total | Probability | Exposure on | Exposure off |
  |---|---|---|---|
  | `lgbm_poisson` | `logistic` | 2.1543 | 2.1929 |
  | `lgbm_poisson` | `lgbm` | 2.2212 | 2.2505 |
  | `lgbm_gaussian` | `logistic` | 2.1318 | 2.1685 |
  | `lgbm_gaussian` | `lgbm` | 2.1706 | 2.2239 |
  | `poisson_glm` | `logistic` | 2.2646 | 2.8660 |
  | `poisson_glm` | `lgbm` | 2.2794 | 2.8828 |
  | `nb2` | `logistic` | 2.2487 | 2.8540 |
  | `nb2` | `lgbm` | 2.3039 | 2.9222 |

- Run times at 3 trials with parallel folds: logistic runs 0.3–1.0 s, LightGBM-classifier
  runs 1.1–5.4 s (0.4–1.8 s a trial), so the 200-trial click should take roughly 20–30 min
  (the user's click at T2.2 measures it).

### [x] T2.2 Model 2's tables and display cells, the best-per-model cell
The refit lives in each run cell (T2.1, D14). The cells:
- **The search space:** `ProbabilitySearchSpace` gains the ClassVar `CALIBRATION_METHODS`,
  read by `suggest_calibration_method` (D16).
- **The tables cell** mirrors T1.7's. It gives:
  - `MODEL2_RUNS`, the 16 run records in run order;
  - `model2_df`, with the settings columns `family` (`"<total>+<probability>"`) and
    `use_exposure` (D16);
  - `model2_arms_df`, the completed trials per choice, every candidate listed and counted by
    label, for `total__feature_transformer`, `probability__feature_transformer` and
    `probability__calibration_method` (D15, D16).
- **The display cell** is as Model 1's: "Model 2: cohort Poisson deviance", then "Trials per
  choice".
- **The best-per-model cell** is the old one (lines 1197–1218 at `8ee091a`) with
  `use_exposure` added. `best_df` is each model's composite row with the lowest validation
  score, with the columns `model`, `family`, `use_exposure`, `cv_best`, `train` and `val`.
- **The header** now says "best run".
```python
for _column, _values in _candidates.items():
    # Counted by label: the trials table holds the None calibration as NaN
    # in a str column, or as None in an object column when every trial
    # drew it, and reindex matches None in neither reliably (D16).
    _labels = [str(_value) for _value in _values]
    _counts = (
        _complete[_column]
        .fillna("None")
        .value_counts()
        .reindex(_labels, fill_value=0)
    )
    for _value, _n in _counts.items():
        _arm_rows.append({**_settings, "parameter": …, "value": _value, "n_trials": int(_n)})
```
**Done when:** Check A; Check C prints `model2_df` with 64 rows (16 × 4), `MODEL2_RUNS` with
16 records, `best_df` with 2 rows; the user clicks both buttons at 200 and reports the times.
Then decide TPE's random start, `n_startup_trials` 40 or 10, with the evidence of D15 and the
handoff's §3 plus Model 2's coverage.

Met 2026-10-21:
- **Check A** green. **Check B:** `CALIBRATION_METHODS` is defined; `MODEL2_RUNS` and
  `best_df` are not, since the buttons stop the script run.
- **Check C** under `PYTHONWARNINGS=error`, 3 trials, both buttons: 47.0 s.
  - All 24 `cv_best` values equal T2.1's baselines, so the ClassVar changed no draw.
  - `MODEL2_RUNS` holds 16 records, and both models' runs serialize to JSON.
  - `model2_df` has 64 rows.
  - The counts per parameter add up to each run's `n_complete`, and the calibration shows
    `None`, `temperature` and `sigmoid`.
  - `best_df` has 2 rows: Model 1 `lgbm_gaussian`, exposure on, val 2.0842; Model 2
    `lgbm_gaussian+logistic`, exposure on, val 2.0872.
- **The first Check C** ran the approved `value_counts(dropna=False)` version, and its sums
  did not add up. The fix was put to the user and approved (D16).
- **The 200-trial run** (both buttons patched, `PYTHONWARNINGS=error`, run 2026-10-21;
  history since D20, whose 400-trial results are current):
  1827 s for the whole notebook.
  - Model 1 took 366 s and Model 2 1447 s, with no failed trial and no warning.
  - Model 1 reproduced D15's 200-trial values exactly.
  - In Model 2, the eight LightGBM-classifier runs took 81–220 s each and the logistic runs
    23–45 s.

  | Model 2 run | Exposure on: cv_best / val | Exposure off: cv_best / val |
  |---|---|---|
  | `lgbm_poisson+logistic` | 2.0617 / 2.1283 | 2.0802 / 2.1713 |
  | `lgbm_poisson+lgbm` | 2.0820 / 2.1307 | 2.1033 / 2.1480 |
  | `lgbm_gaussian+logistic` | 2.0728 / 2.0950 | 2.1253 / 2.1390 |
  | `lgbm_gaussian+lgbm` | 2.1066 / 2.1273 | 2.1526 / 2.1804 |
  | `poisson_glm+logistic` | 2.2418 / 2.1749 | 2.8511 / 3.0383 |
  | `poisson_glm+lgbm` | 2.2582 / 2.2093 | 2.8667 / 3.0603 |
  | `nb2+logistic` | 2.2348 / 2.1811 | 2.8441 / 3.0418 |
  | `nb2+lgbm` | 2.2533 / 2.2193 | 2.8686 / 3.1080 |

  - `best_df`, by validation:
    - Model 1: `lgbm_poisson`, exposure on, cv 2.1459, val 2.1604;
    - Model 2: `lgbm_gaussian+logistic`, exposure on, cv 2.0728, val 2.0950.
  - **Coverage, the least-tried choice per run** (the `n_startup_trials` evidence):
    - Model 1's transformers: 6–9 for the GLMs, 17–49 for LightGBM.
    - Model 2's total transformer: 7–9 for the GLM totals, 16–75 for LightGBM.
    - Model 2's probability transformer: 8–12 for logistic (5 candidates), 16–57 for the
      LightGBM classifier (2 candidates).
    - The calibration: 12–34.

### [x] T2.3 Independent review of T2, summary, commit suggestion.

Done 2026-10-21. A review subagent read the diff `58c3bcb..a2b8693` (writing nothing into the
repository) and found no correctness bug. It checked:
- the 16 run cells, parsed with `ast`: comment, settings, seed offset and record name agree;
  the order is right; `MODEL2_RUNS` is complete;
- the `FixedTrial` rebuild against `best_value`, on three pairs;
- no leakage; the exposure reaches the total only; the metric pairing;
- the counts by label in every column shape;
- `best_df`, JSON, and the draws unchanged;
- Check C (all 16 Model 2 baselines) and Check A.

Its eight findings were reproduced and settled:
- **F1:** stale `MODEL2_TOTALS` / `PROBABILITY_FAMILIES` / `*_STUDY_SETTINGS` text in D6 and
  D12, now marked superseded.
- **F2:** the counting comment said reindex fails "in neither" case; it fails only on the
  all-None column. Corrected.
- **F3:** strict JSON and the `NaN` token, noted in T2.5 (`to_json` when records are saved).
- **F4:** the header's seed wording and "trials per choice", corrected.
- **F5:** the two tables cells count with different idioms. Left as is: Model 1 has no None
  choice.
- **F6:** the interval would be measured on the selection rows. Noted in T2.5's open points.
- **F7:** `catch` covers `RuntimeError` only. Noted in T2.4.
- **F8:** the status line and Model 1's CV gap (0.616 and 0.614), corrected.

### [x] T2.4 Check whether pruning and parallelization help, and where (asked by the user, 2026-10-21)
After T2.3, before T3. It revisits D13 (folds in parallel, no pruning), which was measured on
Model 1 alone, with Model 2's timings. Done 2026-10-22: measured by read-only probes on the
notebook's own functions under `PYTHONWARNINGS=error`, nothing added to the notebook (D19).
The user's instruction: best practices, no over-complication.

**Where the time goes** (Apple M1 Pro, 8 cores, 6 performance; the training part's 917 rows,
folds of 184/183; one fold, sampled trials):

| Pair (fold 0) | Total fit | Probability fit | Same, calibration off | Predict + score |
|---|---|---|---|---|
| `nb2+lgbm` (5 trials) | 0.02–0.17 s | 0.46–1.67 s | 0.08–0.32 s | 0.01–0.04 s |
| `poisson_glm+lgbm` (5 trials) | 0.01–0.07 s | 0.10–1.76 s | 0.04–0.31 s | ≤ 0.015 s |
| `lgbm_gaussian+logistic` (3 trials) | 0.02–0.04 s | 0.03–0.06 s | 0.02–0.03 s | ≤ 0.013 s |
| Model 1 `nb2` (5 trials) | 3 cohort fits, 0.15–0.49 s in all | – | – | ≤ 0.02 s |

- In the slow pairs the LightGBM classifier's fit is the trial, and the calibration
  multiplies it by about 5: `CalibratedClassifierCV(cv=5, ensemble=False)` fits the
  classifier 5 times on inner folds plus once on all rows, 30 fits per trial; two of the
  three calibration choices calibrate. The one-row-per-child expansion costs under 1 ms.
- A whole trial of the slow pairs: 8–10 s with the folds one after another, 2.3–2.7 s in
  parallel (×3.5); `lgbm_gaussian+logistic` 0.36 vs 0.16 s; Model 1 NB2 0.88 vs 0.32 s.
- At 200 trials of `lgbm_gaussian+lgbm` (exposure on, seed 47) a trial takes 0.49 s on
  average (median 0.35, p90 0.85, max 3.07): 97 s for the run, `cv_best` 2.1066 as T2.2.
- joblib: the worker start is paid once per process (about 3 s); then a 5-fold trial of the
  fast families costs 0.16–0.19 s.

**Parallelization, each against today** (5 fold processes, LightGBM `n_jobs=1`):

| # | Place | Measured | Reproducible | Verdict |
|---|---|---|---|---|
| 1 | The folds of a trial (today) | ×2.3–3.6 per trial, identical scores | yes | keep |
| 2 | LightGBM threads | slower everywhere on 917 rows: 2 threads ×3.2, 4 ×5–6, 8 ×11–14 (sequential folds); 2 threads with 5 fold processes ×4–9 | yes | reject |
| 3a | Model 1's 3 cohort fits in threads | slower: 0.15→0.27 s, 0.39→0.74 s per fold (statsmodels holds the GIL); a trial 1.4–3.6 s vs 0.4–0.6 s today | yes | reject |
| 3b | Model 2's total and probability in two threads (measured at the user's question) | 0.78–1.31× of one after the other: the total is 1–10% of a slow fit; only LightGBM releases the GIL, so the GLM and logistic pairs are 1.1–1.3× slower; the one gain (`lgbm_gaussian+lgbm`, 22%) is in an idle process, not inside 5 fold processes; a `src` change | yes | reject |
| 3c | The calibration's inner folds in parallel (`CalibratedClassifierCV(n_jobs=5)`, a `src` change) | a probability fit 1.89→0.71 s and a sequential-fold trial 10.2→3.8 s, but with today's fold processes 2.60→2.42 s: a nested `Parallel` in a loky worker runs sequentially | yes | reject |
| 4 | Several runs at once | marimo runs cells one at a time (`cell_runner` pops its queue "one at a time"; `mo.Thread` runs code, not cells), so it conflicts with D14. As threads over two studies with fold processes: wall 356 s vs 97 + 145 s alone, and wrong values (1.45, 1.17 against 2.1) | no | reject, unsafe |
| 5a | Trials in parallel, `study.optimize(n_jobs=2)` with fold processes | 177 s and 100 s on two repeats, different bests (2.0920, 2.1013); values unreliable | no | reject, unsafe |
| 5b | Trials in parallel, `n_jobs=4` threads with the folds in-process | 227 s and 155 s, bests 2.0958 / 2.0956 at trials 186 / 197; 1.6–2.3× today | no | reject |

Concurrent `Parallel` calls from threads in one process give wrong scores (four fixed
models: sequential 2.1981 / 2.2445 / 2.2471 / 2.3495, concurrent 2.09–2.20 and different
on every repeat; with in-process folds the threads agree exactly): joblib's shared loky
pool mixes the calls' inputs. That is why 4 and 5a are unsafe, not only slow.

**Pruning** (200 trials, `lgbm_gaussian+lgbm`, exposure on, seed 47). `WilcoxonPruner`
(Optuna 4.9) is the pruner for a mean over fixed instances; Optuna's documented usage
returns the partial mean on `should_prune()` instead of raising, and shuffles the fold
order per trial (seeded here by `study_seed + trial.number`). Its constructor raises
`ExperimentalWarning` under `-W error`, so a strict Check C would need it suppressed, as
D7 found for `multivariate=True`. With 5 folds the one-sided exact test gives p = 1/2^n
after n folds all worse: 0.125 after 3, 0.0625 after 4, so at threshold 0.1 a trial is
pruned after fold 4 at the earliest, at 0.2 after fold 3; a 3-then-2 batch hybrid cannot
prune at 0.1.

| Run | Wall | Best (trial) | Pruned | Fold fits | Same search as today? |
|---|---|---|---|---|---|
| R0 today: 5 fold processes, no pruning | 97 s | 2.1066 (148) | 0 | 1000 | yes (T2.2's value) |
| R1 Wilcoxon 0.1, folds sequential | 480 s | 2.1010 (154) | 56 | 944 | no |
| R2 Wilcoxon 0.2, folds sequential | 408 s | 2.1053 (173) | 132 | 782 | no |
| R3 hybrid 3 + 2 processes, 0.2 | 326 s | 2.0909 (164) | 68 | 864 | no |
| R3b hybrid 3 + 2 processes, 0.1 | 169 s | 2.1066 (148) | 0 | 1000 | yes: prunes nothing, the pure cost of two batches (×1.7) |
| R4 Wilcoxon 0.1 sequential + calibration `n_jobs=5` (`src` change) | 276 s | 2.1010 (154) | 56 | 944 | no (R1's search) |
| R8a Model 1 `poisson_glm` today | 30 s | 2.2490 (199) | 0 | 1000 | yes (the handoff's value) |
| R8b the same, Wilcoxon 0.1 sequential | 99 s | 2.2490 (199) | 21 | 979 | yes |

- Pruning saves 2–22% of the fold fits but costs the parallel folds, which save 65–72%:
  the slowest pair runs 2.8–5× longer, the fastest family 3.3× longer.
- A pruned trial's value is the mean of fewer folds, so the sampler is fed other values and
  the search diverges: R1–R4 found other bests, better or worse by chance.
- Inside a LightGBM fit (early stopping, `LightGBMPruningCallback`): needs an `eval_set` at
  every fit. `CountModel.fit` and `CohortProbabilityModel.fit` take none,
  `CalibratedClassifierCV` forwards none to its inner fits, and the final refit on the whole
  training part would have no stopping set, so `n_estimators` would be undefined there; it
  is already searched (50–400). Not applicable.

**Which errors fail one trial (T2.3's F7),** by forced failures through `cv_score`:

| Failure | Arrives as | Trial-specific? | Today |
|---|---|---|---|
| NB2 IRLS or penalized fit does not converge (`max_iter=1`; `_PENALIZED_MAX_ITER=1`) | `RuntimeError`, from 1 or 5 processes | yes (dispersion, alpha) | caught |
| sklearn `ConvergenceWarning` under `PYTHONWARNINGS=error` (`max_iter=1`) | `sklearn.exceptions.ConvergenceWarning` from a worker; `catch=(ConvergenceWarning,)` would mark FAIL | yes (C, alpha) | ends the run, strict runs only |
| the same in marimo (no `PYTHONWARNINGS`) | printed by the worker, the trial completes | – | nothing to catch |
| a cohort unobserved in a fold's training rows | `ValueError` | no: the folds are fixed, every trial fails alike | ends the run, rightly |
| `_check_rows_sum_to_one` | unreachable: the objective is fixed to `multiclass` | – | – |

`catch` stays `(RuntimeError,)` (D19).

**The `n_trials` / `n_startup_trials` evidence** (decided next):

| Run | Best found at trial | Best so far at 100 / 150 / 200 | Least-tried choice | Wall |
|---|---|---|---|---|
| `lgbm_gaussian+lgbm`, start 10 (R0) | 148 | 2.1068 / 2.1066 / 2.1066 | 16 | 97 s |
| the same, start 40 (R7) | 192 | 2.1228 / 2.1201 / 2.1127 | 18 | 310 s |
| `lgbm_poisson+lgbm`, start 10 (R6a) | 130 | 2.1102 / 2.0820 / 2.0820 | 16 | 145 s |
| Model 1 `poisson_glm`, start 10 (R8a) | 199 | 2.2519 / 2.2496 / 2.2490 | 6 | 30 s |

A random start of 40 draws the large slow models at random (3× the time) and ends worse.
At 200 the LightGBM runs stop improving by 150, and the GLM's last 50 trials gain 0.0006;
400 trials gave the GLM nothing (handoff §3). Every choice was tried at least 6 times.
Recommendation: keep 200 and 10.

**Done when** (met 2026-10-22): the measurements above; the notebook unchanged (md5
`eec46ac5cbec902ffdd98cce6f4c9cbd`); Check A green; the 200-trial values of R0, R6a and R8a
equal T2.2's and the handoff's.

### [x] T2.5 The best runs: bootstrap interval and feature importance (D18)

After T2.4. For each model's best run in `best_df`:
- **The fitted model:** run records keep no fitted model (D13), so the best run is refit from
  its record's components.
- **Bootstrap (E3):** `BootstrapEvaluator(COHORT_POISSON_DEVIANCE, n_resamples,
  random_state=seed)` on validation, with `groups=val_age_group.groups` (`neighborhood_id`)
  and the exposure. The table shows `mean`, `lower` and `upper` beside each best run's
  point score.
- **Permutation importance (E7):** `PermutationImportance(COHORT_POISSON_DEVIANCE,
  n_repeats, random_state=seed)` on validation, and on training as the overfit check (a
  column important on training and not on validation).
- **SHAP (E8):** `ShapExplainer(method="exact")` explains a seeded validation subsample
  against a seeded training background. Global importance is mean |SHAP| per column, and the
  per-cohort values are available from the `Explanation`.
- **The comparison cell (D18):** per model and raw column, `perm_mean`, `perm_std` and
  `perm_rank` beside `mean_abs_shap` and `shap_rank`.

The API, read 2026-10-21:
- `BootstrapEvaluator(...).evaluate(model, X, y, groups, *, exposure)` returns
  `BootstrapResult(mean, lower, upper)`.
- `PermutationImportance(...).compute_importances(model, X, y, *, exposure)` returns
  `importance_mean` and `importance_std` per column.
- `ShapExplainer(...).explain(model, X, background, *, exposure)` returns a
  `shap.Explanation`.
- `n_resamples = 1000` is already in the notebook's settings.

**Settled 2026-10-22 (the user, at T2.5's plan):** the best model is refit on the whole
training part and bootstrapped on the training rows and on the validation rows (the test
part waits for the refit on training + validation, E11); the bootstrap covers seven
metrics, each explained in the header (Poisson deviance, cohort log loss, D² Poisson,
MAE, RMSE, WMAPE, bias; scikit-learn's functions on the flattened cells, the two without a
library form as one-line notebook functions); the run records gain `best_params`, so the
best model is rebuilt with the existing `FixedTrial` path; one button.

**Done 2026-10-22, as built (D22 revised the recipe):**
- `src/age_group_prediction/inspection/bootstrap.py`: the evaluator of D22, its tests
  rewritten (22, four new: shared replicates and one predict, the `score` column and the
  row order, empty and duplicate metrics raise), `MODEL_INSPECTION.md` and
  `MODULE_REFERENCE.md` updated; no `inspect_model`.
- The notebook: the settings `n_repeats = 10`, `n_shap_rows = 100`, `n_shap_background =
  50` (logged; the notebook test's expected dict gained them); the header paragraph; the
  imports; the metrics cell (`on_cells`, `wmape`, `bias`, `COHORT_LOG_LOSS`,
  `INSPECTION_METRICS`); `"best_params"` in the 24 records; the button
  `run_inspection`; a seeded cell drawing `shap_rows` (validation) and `shap_background`
  (training) once, so both models are explained on the same rows; per model three cells
  behind the button: `best_run1` / `best_model1` (the lowest validation composite, rebuilt
  from `best_params`, refit on training), `model1_intervals` (two `evaluate` calls, train
  and val, concatenated), `model1_permutation` / `model1_shap`; `intervals_df` and
  `importance_df` with their display.
- Check C at 3 trials (`n_shap_rows = 5`, `n_repeats = 2`, three buttons patched,
  `PYTHONWARNINGS=error`): 66.8 s, the 24 baselines reproduced, both tables rendered,
  `best_params` in every record and JSON-serializable.

**Open points, settled at T2.5's own plan:**
- How the fitted best model is obtained: rebuild the components from the record's dicts, or
  keep `best_params` in the record and use `FixedTrial`.
- Whether the bootstrap covers only the primary metric or E3's full list.
- `n_repeats`, the SHAP sample sizes and `n_jobs`, from measured cost.
- A button, since this costs minutes.
- Which split gets the interval (T2.3's F6). `best_df` selects on validation, so an
  interval on the same rows is optimistic; the test part (`test_age_group`) is unused, and
  E11 refits on training + validation and scores test once.
- How a run record is saved (T2.3's F3). `json.dumps(…, default=str)` writes a `NaN` token
  (the None calibration in a str column, a failed trial's value), which strict JSON readers
  reject. `trials.to_json(orient="records")` writes `null`. Use it when records are first
  saved (here or in T4b).

**Done when:**
- Check A and Check B.
- Check C (3 trials, with the bootstrap and the importances at reduced sizes) prints the
  interval table and the comparison table for both models.
- The baselines are unchanged.

## 4. [x] T3: the metric constants leave `scoring.py` (src, tests, docs)

**Done 2026-10-22.** Scope decided by the user at the plan summary: the constants only; `Metric`, `cohort_log_loss` and `cohort_poisson_deviance` stay. The two tuning tests that T4a deletes got a module-level inline constant each and nothing else. `docs/HYPERPARAMETER_TUNING.md` was left as is (deleted in T4a).

- `src/age_group_prediction/scoring.py`: delete `POISSON_DEVIANCE`, `RMSE`, `MAE`,
  `COHORT_LOG_LOSS`, `COHORT_POISSON_DEVIANCE`, their `__all__` entries, the now-unused
  imports (`mean_absolute_error`, `root_mean_squared_error`; `mean_poisson_deviance` stays
  for `cohort_poisson_deviance`); the module docstring's mention of `hyperparameter_tuning`
  becomes "the notebook".
- Tests: `tests/unit/test_scoring.py` (the parametrized `test_ready_made_metrics_wrap_their_function`
  goes; the `Metric` tests stay), `test_modeling_count_model.py:28` and `:341`,
  `test_inspection_permutation.py:29` and `:216`, `test_inspection_bootstrap.py:17`, `:223`,
  `:225`: each builds its metric inline, e.g. `Metric("poisson_deviance", mean_poisson_deviance)`.
  The tuning tests that import the constants are deleted in T4 (do T3 and T4 in this order, or
  T4 first; T4 first is simpler because it removes 8 of the usages).
- Doc code blocks (run as written): `docs/MODEL_INSPECTION.md:59`,
  `docs/INDEPENDENT_COHORT_MODELS.md:341`, `docs/TOTAL_TIMES_PROBABILITY_MODEL.md:376`: the
  import line becomes `from age_group_prediction.scoring import Metric` plus the sklearn
  function, and one line builds the metric. Prose in the plan/validation docs stays (history).
- **Open point for the user (ask at T3's plan summary):** "remove the metrics from scoring"
  is read as the constants only; `Metric`, `cohort_poisson_deviance`, `cohort_log_loss` stay
  because `inspection` and the model docs use them through a `Metric`.

**Done when:** `grep -rn "COHORT_POISSON_DEVIANCE\|COHORT_LOG_LOSS\|POISSON_DEVIANCE\|RMSE\b\|MAE\b" src tests`
returns only prose in the old stack (`models/`, `experiment/`, `bundle_spy.py`,
`test_final_selection.py`, `test_direct_cohort.py`, `test_final_evaluation.py`); `uv run mypy`;
the changed tests under `-W error`; the non-slow suite; the three doc blocks executed by probe.

## 5. T4: replace the `hyperparameter_tuning` package (D17)

After T2.4 and T3. Two stops, each with its own approval.

### [x] T4a Delete the old package

Split on 2026-10-29 into T4a.1 (the code), T4a.2 (the current guides) and T4a.3 (the finished
plans, the status lines, the notebook checks), each with its steps, done-when list and commit
message in [NOTEBOOK_T4_HANDOFF.md](NOTEBOOK_T4_HANDOFF.md) §2; the list below is the scope.

- Delete `src/age_group_prediction/hyperparameter_tuning/` (4 modules + `__init__`),
  `tests/unit/test_hyperparameter_tuning_{aggregation,evaluator,integration,parameters,study}.py`,
  `docs/HYPERPARAMETER_TUNING.md`, `docs/HYPERPARAMETER_TUNING_PLAN.md`,
  `docs/HYPERPARAMETER_TUNING_REVIEW.md`, `docs/HYPERPARAMETER_TUNING_VALIDATION.md` (plain
  `rm`; the user commits the deletions). Keep `src/age_group_prediction/aggregation.py` and
  `tests/unit/test_aggregation.py`, the aggregator's interim home (D9); T4b moves them into
  the new package.
- Edit: `pyproject.toml` (the package's entries in `[tool.mypy] files` and overrides, then
  `uv run --frozen`), `src/age_group_prediction/_config.py` docstring, `README.md`,
  `docs/README.md`, `docs/MODULE_REFERENCE.md`, `docs/INDEPENDENT_COHORT_MODELS.md` §3/§4/§6
  (the sentences pointing at the package; the forwarding stays documented as a `set_params`
  feature), `docs/SPLITTING.md` and the others listed by
  `grep -rln "hyperparameter_tuning\|HYPERPARAMETER_TUNING" docs README.md` where the text
  states current API (plan docs of finished work: a one-line note that the package was removed
  on this date, nothing more). `optuna` stays in `pyproject.toml`.

**Done when:** the grep above returns nothing in `src`, `tests`, `README.md` and the current
guides; `uv run mypy`; the non-slow suite (expect the count to drop by the five files' tests);
the notebook test green.
(The directory is re-created under the same name in T4b.)

### [ ] T4b The new package, from the notebook's code

Under `src/age_group_prediction/hyperparameter_tuning/`:

| Module | From notebook section 5 |
|---|---|
| `params.py` | `LGBMParams`, `PoissonGLMParams`, `NB2Params`, `LogisticParams` |
| `components.py` | `CountFamily`, `ProbabilityFamily` (as `type` statements, which pickle from a module, D3), `Component` (with `set_estimator_params`), `CountComponent`, `ProbabilityComponent` (each with `from_params`, D25.2; the field `estimator_params`) |
| `search_spaces.py` | `ComponentSearchSpace`, `CountSearchSpace`, `ProbabilitySearchSpace` (with `CALIBRATION_METHODS`; one family's candidate names injected as `feature_transformer_names`, D25.1; the `suggest_*` methods, `suggest_feature_transformer_name` among them; no readers; the candidate dicts stay in the notebook) |
| `builders.py` | `build_independent_cohorts_model`, `suggest_independent_cohorts_components`, `build_total_times_probability_model`, `suggest_total_times_probability_components` (D25.3: one loop each, the components set with constructor calls; D26.1: the suggest functions return the components, the build functions build; no readers, no `FixedTrial`) |
| `cross_validation.py` | `CrossValidationTuner` (D26.2–D26.3: `fold_score`, `cv_score`, `tune_independent_cohorts_model`, `tune_total_times_probability_model`) |
| `evaluation.py` | `fit_and_score` (D26.4) |
| `aggregation.py` | `Aggregator`, `Mean`, `WeightedMean`, `UpperBound`, moved back from `age_group_prediction/aggregation.py` (D9's interim home), with `tests/unit/test_aggregation.py` |

The rule for what moves and what stays, the one-way dependencies and the split into T4b.1
(records and search spaces), T4b.2 (builders, loop, aggregation) and T4b.3 (the notebook
imports the package) are D21 (2026-10-22).

- **Code:** typed for mypy strict (the notebook code is untyped), docstrings kept to what
  matters, and an `__all__` per module. `pyproject.toml`'s mypy `files` and overrides list
  the package.
- **Unit tests per module**, in new files `tests/unit/test_hyperparameter_tuning_*.py`:
  - the records' validation: an unknown family raises, `kw_only`, a typo in the params
    raises;
  - the search spaces' trial names, and the `from_params` read-back of a trial's params equal to the objective's model (the notebook test's check, D25.4);
  - the builders' models;
  - `cv_score` with the folds in parallel equal to one after another;
  - `tune_model`'s `catch`.
- **T2.5's cells** (the bootstrap and importance) stay in the notebook unless T4b's plan
  says otherwise.
- **The notebook:** section 5 keeps the header, the imports from the package, the metric and
  the constants (`COHORTS`, `PREDICTION_FLOOR`), the buttons, the run cells and the tables.
  The class and function cells go.
- **`tests/characterization/test_experiments_notebook.py`:** its rule (every cell containing
  `tune_model` holds `mo.stop(not `, except the `def tune_model(` cell) now exempts the import
  cell instead.

**Done when:**
- Check A, with ruff's baseline re-counted and reported, since cells are removed;
- Check B;
- Check C reproduces every baseline: Model 1's 8 and Model 2's 16, or T2.4's new ones;
- `uv run mypy`;
- the new tests and the non-slow suite pass under `-W error`.

## 6. T5: docs for the new design

- A new `docs/HYPERPARAMETER_TUNING.md`, the guide to the new package (T4b); the old one is
  deleted in T4a.
- `docs/EXPERIMENT_NOTEBOOK_HANDOFF.md`: §1 item 3 rewritten (the S1–S5 block becomes one
  paragraph of history; the new design and its dataclasses described), §2 state, §3 the
  decisions D1–D7 of this plan, §6 the prompt for the performance pass.
- `docs/EXPERIMENT_NOTEBOOK_PLAN.md`: E4, E5, E11 (`evaluator.build_model(best_params)` →
  `suggest_model1(FixedTrial(best_params), …)`), E12, D2/D3 and the status line.
- The dataclass-vs-TypedDict comparison (below) into the handoff's decisions.

**Dataclass vs TypedDict, for the record:**

| | frozen dataclass (chosen) | TypedDict |
|---|---|---|
| JSON out | `json.dumps(asdict(component))`; an estimator's params likewise (probed) | the dict itself |
| JSON in | a constructor per class, or re-running `suggest` on a `FixedTrial` (what the notebook does) | `json.loads`; nothing checks the keys |
| Immutability | frozen and hashable; a component keys its study | a plain dict, not hashable |
| Equality | by value (a `FixedTrial` rebuild of the params is equal: the T1.3 check) | by value |
| Dispatch on the params | `isinstance` in `count_estimator` / `probability_estimator` | `isinstance` does not work on TypedDicts; dispatch needs a `family` key |
| Behaviour | `suggest` (params) and `suggest_*` (components) on the class; no `build` (D3) | functions only |

## 7. The performance pass (decided 2026-10-21 by D13: folds in parallel, no pruning; the facts weighed)

Revisited by T2.4 (the user's request, 2026-10-21) with Model 2's timings; the facts below were
its starting point, and T2.4 measured them on 2026-10-22: D13's choice stands (D19).

- **Pruning mid-CV.** Optuna's `WilcoxonPruner` (4.9: `p_threshold=0.1`, `n_startup_steps=2`)
  is the pruner designed for a mean over fixed instances such as CV folds: it pairs fold k of
  the trial with fold k of the best trial by `step`, and needs `trial.report(score, step=fold)`
  and the objective returning the partial mean on `should_prune()`. `MedianPruner`,
  `SuccessiveHalving`, `Hyperband` assume a converging learning curve, which fold scores are
  not. With 5 folds the one-sided exact test gives p = 1/2^n after n folds all worse: 0.125
  after 3, 0.0625 after 4, so at 0.1 a trial is pruned after fold 4 at the earliest (≤ 20% of
  its cost saved), at 0.2 after fold 3 (≤ 40%). Worth it only where trials are slow (the
  LightGBM classifier pairs); decide from T2's timings. `cv_score` already keeps the per-fold
  list, so the change is local.
- **Parallelization.** Today one process per study (joblib; seeded studies identical in
  parallel). Keeps the seeded search reproducible: folds of one trial in parallel (trials stay
  sequential), or the three cohort fits of Model 1 inside the composite (a src change). Loses
  it: `study.optimize(n_jobs>1)` (threads; order not reproducible) and multi-process trials on
  a shared storage. The machine has 8 cores (6 performance); Model 2 already runs up to 16 studies at
  once, so per-fold parallelism competes with it.

## 8. End-to-end verification (after T2, again after T4)

1. `uv run pytest -m "not slow"` green; the notebook's script run fits nothing (≈ 5 s).
2. Check C with both buttons patched at `n_trials = 3`: every table renders;
   `json.dumps` of `MODEL1_RUNS` and `MODEL2_RUNS` works (D13).
3. The user clicks both buttons at 200 in `marimo edit`; wall times recorded in the handoff.
