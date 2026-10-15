# Handoff: the experiment notebook, from tuning to MLflow

**For:** the session that continues `notebooks/02_experiments.py`. **Written:** 2026-10-15,
after step 3's declarations (section 5 of the notebook) were committed as `2c0859c`.
**Plan (source of truth):** [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md): §3 (how
to work), §4 (decisions E1–E12) and §9 (stops D2–D5). Read those three sections in full before
the first edit. This file orients you and records the user's order of work, the decisions of
2026-10-15 and their evidence. Re-verify every fact here against the code of the day.

## 1. What the task is, in the user's order

Each step is one or more plan stops. Plan each in plan mode, one step per approval.

1. **Train, validation and test split** (D1c). **Done:** `537158a`.
2. **Feature-engineering configurations** (D1d). **Done:** `98640ba`, `efbfeb8`; renamed in
   `2c0859c` (§3).
3. **Hyperparameter tuning of the two models, by CV on the training part** (the studies of plan
   stops D2/D3, without MLflow). Its stops:
   - **3a, done** (`958cdea`): `CountModel.predict` floors the mean at `np.finfo(float).eps`,
     so a Gaussian loss's negative count scores a large deviance instead of stopping a study.
   - **3a2, done** (`a0e361c`): `NegativeBinomialRegressor(dispersion=None, alpha=0.0,
     max_iter=500)`: a fixed dispersion and an L1 penalty as hyperparameters, one GLM stage.
   - **3b, done** (`2c0859c`): section 5's declarations (§2 below).
   - **3c, rejected 2026-10-15 (evening):** the run cells as the morning's handoff designed them
     (one study per component, a 32/7/224 settings grid, every setting refit and scored on validation)
     were built, checked and reviewed, then rejected by the user: **the composite is what
     is scored**, and Model 2's components interact through the product, so component
     studies do not optimize the final metric. The notebook and its test were restored to
     HEAD `2c0859c` (cmp verified). Nothing of that build remains in the tree.
   - **3c, the new design (user's decisions, do not re-ask):**
     - **One Optuna study per model family on the composite.** Model 1
       (`IndependentCohortModels`): `lgbm`, `poisson_glm`, `nb2`. Model 2
       (`TotalTimesProbabilityModel`): total family × classifier family (`logistic`,
       `lgbm`), six studies. Objective: `COHORT_POISSON_DEVIANCE` on the training folds
       (`cv`, `cv_groups`), through `CVHyperparameterEvaluator` as it is.
     - **A trial chooses per cohort** (per component for Model 2) its `use_exposure`
       (categorical `[True, False]`), its feature variant (a `FeatureTransformer`, S2
       below), its LightGBM loss (`estimator__objective` in `poisson` / `regression`) and
       the estimator's hyperparameters, under the names
       `cohort_models__<cohort>__<param>` (S1 below) or `total_model__<param>` /
       `probability_model__<param>` (already reachable).
     - **`n_trials = 200`** for the joint studies (a LightGBM trial sets about 39 values
       for three cohorts; 30 trials would be a near-random search). The studies of one
       model run **in parallel** with joblib (measured: two seeded studies give identical
       results in parallel; a loky start costs ≈ 2 s).
     - **Only the best trial's combination is refit** on the training part and scored on
       training and on validation, with per-cohort (per-component) deviances shown.
     - **Model 2's total choices are narrowed** by a list the user edits after Model 1's
       run (a settings cell; the pairs are that list × the classifier families).
     - **Stops**, one approval each (S1 and S2 may share one if the user says so):
       - **S1 (src):** `IndependentCohortModels` overrides `get_params` / `set_params` so
         `cohort_models__<cohort>__<param>` reaches each cohort model, `cohort_models__<cohort>`
         replaces one model and `cohort_models` the mapping; `clone` keeps working.
         Measured: today `set_params(cohort_models__a__estimator__alpha=0.5)` raises
         `AttributeError: 'dict' object has no attribute 'set_params'`; a prototype override
         (about 20 lines) gives deep keys, leaves the template untouched under `clone`
         (sklearn clones a dict parameter value by value) and a misspelt sub-name raises
         sklearn's `Invalid parameter` naming the estimator. sklearn's
         `_BaseComposition` was rejected: it takes a list of pairs (a dict fails to unpack),
         would change the public constructor and the `fit` Counter check, and its names are
         flat. `set_params` pops `cohort_models` first (sklearn's order: nested names set
         on the old models would otherwise be overwritten silently). No new checks: a
         misspelt cohort raises `KeyError` (loud); a `__` inside a cohort name is
         documented, not checked. Tests (one mutation each):
         `test_nested_names_reach_each_cohorts_model`,
         `test_set_params_on_a_clone_leaves_the_template_untouched`,
         `test_the_whole_mapping_is_replaced_before_nested_names`,
         `test_a_whole_cohort_model_can_be_replaced`; the contract test
         (`check_set_params`, shallow params) needs no change. Docs:
         INDEPENDENT_COHORT_MODELS.md §3 (the "tuned on its own" bullet), §4, §6;
         HYPERPARAMETER_TUNING.md §9 (the composites paragraph).
       - **S2 (src):** `KeyedParameter(name, choices: Mapping[str, Any])` in
         `parameters.py`: `suggest` returns `trial.suggest_categorical(name, tuple(choices))`,
         the key; `resolve(key)` returns the object (`KeyError` on an unknown key);
         `Parameter.resolve` is the identity for the other classes; `choices` stored as a
         `MappingProxyType` (a later item assignment would change what trials resolve while
         `best_params` still says the key). `CVHyperparameterEvaluator.build_model`
         resolves `{name: by_name[name].resolve(value) if name in by_name else value}`; a
         name without a Parameter passes through as today. `evaluate`,
         `build_model(best_params)`, `FixedTrial(best_params)` (checks key membership) and
         `to_dict()` stay as they are (measured). Tests: parameters
         (`test_keyed_choices_reach_optuna_as_keys_in_order`,
         `test_resolve_returns_the_object_for_its_key`,
         `test_keyed_choices_cannot_be_edited_after_build`,
         `test_non_string_keys_are_rejected`, the exports test); evaluator
         (`test_build_model_resolves_a_keyed_parameter`,
         `test_a_name_without_a_parameter_passes_through`,
         `test_best_params_of_a_keyed_study_are_json_and_refit_via_fixed_trial`). Docs:
         HYPERPARAMETER_TUNING.md §3 (table row, the D18 paragraph: keys, not objects), §4
         step 1. The per-cohort prefix is a 3-line notebook helper over
         `dataclasses.replace(p, name=f"cohort_models__{cohort}__{p.name}")` (measured: it
         re-validates and keeps the class, the bounds and the choices; the result is frozen).
       - **S3 (notebook):** section 5 rewritten: per-family joint search spaces with the
         per-cohort prefixes, `n_trials = 200` in the settings cell (the test's
         `experiment_settings` line changes with it), the Model 1 button cell and run cell
         (three studies through joblib), the refit of the best combination and a table of
         train and validation deviances per cohort; per-arm trial counts of each categorical
         in the table (§5's risk). The split cell returns `X_train, X_val, Y_train, Y_val,
         exposure_train, exposure_val` and the imports cell `clone` when read. The test
         `test_a_tuning_cell_outside_a_run_button_is_caught` (every cell calling
         `tune_component(` holds `mo.stop(not `; mutation: delete one `mo.stop`). Verified
         2026-10-15: a button's value is `False` in a script run, `mo.stop` halts the cell
         and its descendants, and `app.run()`'s defs hold unreturned names, so the return
         trimming is a file-format matter only. A "buttons clicked" check runs `app.run()`
         with `marimo.ui.run_button` patched to return an object whose `value` is `True`
         (verified on a scratch app); the user clicks in `marimo edit` to confirm.
       - **S4 (notebook):** Model 2: the narrowed total choices cell, the six studies, the
         refit and the table, the best-per-model view.
       - **S5 (docs):** the plan's E4/E5 and D2/D3 rewritten (they still record the
         two-study form and the settings grid), HYPERPARAMETER_TUNING.md §9,
         INDEPENDENT_COHORT_MODELS.md, this file.
     - **Budget facts (2026-10-15, 917 training rows, 5 folds, 30 trials, measured while
       the suite ran, so upper bounds):** one study ≈ lgbm 7.4 s, lgbm_gaussian 5.3 s,
       poisson_glm 3.3 s, nb2 11.4 s, logistic 7.5 s, LightGBM classifier 111 s
       (calibration searched). Optuna's own cost is 0.03 s per 30-trial study; one NB2 fit
       0.02 s at the maximum-likelihood default, 0.08–0.14 s with a fixed dispersion and
       an L1 penalty, against 0.03 s for the Poisson GLM; a composite refit + validation
       score 0.05–0.16 s. Nothing in the package runs in parallel (trials, folds and
       `n_jobs=1` are sequential), but BLAS threads under the GLMs give ≈ 2 cores.
4. **Check that bootstrapping and feature importance work** (plan stop D2 steps 3–4, as a
   working check before any logging).
   - Guide: [MODEL_INSPECTION.md](MODEL_INSPECTION.md) (§2 worked example, §3 bootstrap, §4
     permutation, §5 SHAP).
   - `BootstrapEvaluator` (`n_resamples=1000`, `groups=` the validation groups).
   - `PermutationImportance(metric=COHORT_POISSON_DEVIANCE, …)` on validation and on train
     (E7).
   - `ShapExplainer` on a seeded validation subsample (E8).
   - Run them on the refit tuned models. Report the outputs and the wall times.
5. **MLflow runs** (plan stop D1e's guard, then D2/D3's runs).
   - The guard first: the scratch store only, with a refusal on the canonical `mlflow.db`. See
     [MLFLOW_EXPERIMENTS_GUIDE.md](MLFLOW_EXPERIMENTS_GUIDE.md) §2.
   - Then one parent run per setting with nested runs per study, logging what E9 lists.
   - The E10 helper module (`mlflow_runs.py`) only once two cells repeat the same logging.
   - Each run cell sits behind `mo.ui.run_button()`, so opening the notebook fits nothing.

## 2. State of the branch (2026-10-15)

- **Branch:** `feat/experiment-notebook`, HEAD `2c0859c`; the tree holds only this file
  (the notebook and its test were restored to HEAD on 2026-10-15; notebook md5
  `723c40d98f9faf64078f5ecc51338010`).
- **Non-slow suite:** 1305 passed, 1 skipped, 1 xfailed (≈ 90 s).
- **The notebook test** (`tests/characterization/test_experiments_notebook.py`) runs the
  notebook as a script and reads its printed lines. Four tests: the two settings lines, the
  data overrides, the split sizes, and the seed override. Keep those prints. The script run
  takes ≈ 4 s and fits nothing.
- **`notebooks/02_experiments.py`:**
  - **Section 1, data:** `df`, `X` with 9 columns including `n_apartments`, `Y` (3 cohort
    columns), `groups`, `exposure` from `ExposureTransformer`, `id_col`.
  - **Section 2, settings:** `seed=42`, `split_method="random"` (the user's call, "for now"),
    `test_size=0.2`, `val_size=0.2`, `n_splits=5`, `n_trials=30`, `n_resamples=1000`. The
    cell returns `n_splits, n_trials, seed, split_method, test_size, val_size`.
  - **Section 3, split:** by position with `.iloc`; prints `split n_train=917 n_val=306
    n_test=306`; defines the 15 parts (`X_`, `Y_`, `exposure_`, `groups_`, `building_id_` for
    `train`, `val`, `test`) but returns only `split_groups, splitter, train_index`. The cv
    cell returns `cv, cv_groups` (`cv_groups` is `None` under `random`: `KFold` raises under
    `-W error` when given groups).
  - **Section 4, feature variants:** a shared-plans cell and a variants cell returning
    `FEATURE_VARIANTS`, **12** declarations: `tree`, `tree__ses_quadratic`; `glm_base` (the
    base of every GLM, count or probability: one declaration replaces the identical
    `total_base` and `cohort_probability_base` of FEATURE_TRANSFORMATIONS.md §8.2–8.3);
    `count__ses_quadratic`, `count__daycare_log1p`, `count__room_share_x_household_size`,
    `count__room_share_x_ses`, `count__room_share_x_household_size_and_ses`;
    `probability__daycare_log1p`, `probability__room_share_x_median_age`,
    `probability__room_share_x_daycare`, `probability__room_share_x_median_age_and_daycare`.
    Only the two tree variants list `n_apartments`. Fitted column counts on the training part:
    10/11/9/10/9/12/12/15/9/12/12/15. No check cell and no test cover the variants.
  - **Section 5, tuning declarations** (six cells after the user's empty trailing cell):
    - a header (`hide_code=True`) stating the design in prose;
    - imports: `time`, `optuna` (verbosity WARNING), `LGBMClassifier`, `LGBMRegressor`,
      `LogisticRegression`, `PoissonRegressor`, the five tuning classes, the five models, the
      three metrics; all returned;
    - `SEARCH_SPACES` (`lgbm`: the guide's ten; `poisson_glm`: `estimator__alpha` 1e-4..10
      log; `nb2`: `estimator__dispersion` 0.01..1 log and `estimator__alpha` 1e-3..1 log;
      `logistic`: `estimator__C` 1e-3..100 log + `CALIBRATION`; `lgbm_classifier`: the ten +
      `CALIBRATION`), `CALIBRATION = CategoricalParameter("calibration_method", [None,
      "temperature", "sigmoid"])`;
    - `REGRESSORS` and `CLASSIFIERS`: name → `(estimator, search space, variant keys)`.
      `lgbm_poisson` / `lgbm_gaussian` on `TREE_VARIANTS`, `poisson_glm` / `nb2` on
      `COUNT_GLM_VARIANTS` (`glm_base` + the five `count__*`), `logistic` on
      `PROBABILITY_GLM_VARIANTS` (`glm_base` + the four `probability__*`), `lgbm` (the
      classifier) on `TREE_VARIANTS`. Every LightGBM: `subsample_freq=1, n_jobs=1,
      verbosity=-1, random_state=seed`; GLM and logistic `max_iter=1000`;
    - `MODEL1_SETTINGS` (32: name → `(regressor, variant, use_exposure)`, keys like
      `lgbm_poisson__tree__exposure`), `PROBABILITY_SETTINGS` (7: name → `(classifier,
      variant)`), `MODEL2_SETTINGS` (224: `f"{model1 key}__{probability key}"` → the pair);
    - `tune_component(model, parameters, X, y, *, exposure, metric, study_seed, cv,
      cv_groups, n_trials) -> (TuningResult, seconds)`: evaluator + study; one trial when
      `parameters` is empty.
    - Nothing in section 5 fits a model; the variant keys are resolved only in 3c's run cells
      (a misspelt key raises `KeyError` at the button click, not silently).
- **`src` since the previous handoff:** `CountModel.predict` floors the mean (3a);
  `NegativeBinomialRegressor` has `dispersion` and `alpha` (3a2; docs
  INDEPENDENT_COHORT_MODELS.md §2.3). The `hyperparameter_tuning` package is unchanged and is
  used as is (measured 2026-10-15: every estimator family ran through it under `-W error`).

## 3. Decisions already made (do not re-ask)

- **The plan's E1–E12** (§4 of the plan). The ones that matter now: E3 (the primary metric is
  `COHORT_POISSON_DEVIANCE` on validation), E4 (one study per cohort; two studies for Model
  2), E5 (exposure on/off is a setting), E7/E8, E9, E10, E11, E12 (one seed; study i of a run
  gets `seed + i`).
- **The exposure rule:** `n_apartments` is a feature and the offset for the trees, the
  exposure only for the GLMs. `ExposureTransformer` stays.
- **The split by position** with `.iloc`, and `split_method="random"` for now.
- **Decisions of 2026-10-15 (morning), with their evidence** (the measurements are on the
  notebook's own data: 917 training rows, 5 seeded folds, 30 trials, `-W error`). The
  evening block below supersedes the items that make the loss, the variant and the
  exposure settings; the measurements stay as evidence:
  - **The LightGBM loss is a setting, not a `CategoricalParameter`.** On `n_elementary`,
    Poisson alone reached CV deviance 1.953 and Gaussian alone 1.958; one study with the
    objective as a categorical split its 30 trials 10/20 and reached 1.966, worse than either.
    Without the exposure the Gaussian arm predicted a negative count in trial 4, sklearn's
    deviance raised and the study stopped (the joint study too). Hence 3a's floor: at 0 the
    deviance still raises, at `np.finfo(float).tiny` it overflows, at eps it is 216.85 for a
    count of 3, finite; with the floor the Gaussian × no-exposure study completed (best 2.030).
  - **The feature variant is a setting axis** (regressor × variant × exposure), never a
    categorical: six GLM variants would get five trials each.
  - **NB2 tunes `dispersion` and `alpha` in one GLM stage.** A two-stage fit (dispersion by
    maximum likelihood, then a penalized fit at it) was rejected by the user. statsmodels'
    discrete L1 solver (SLSQP) never reached the penalized optimum (kept 9 of 9 coefficients
    up to alpha 3 where the optimum is all zeros); cvxopt is not installed. The GLM
    coordinate descent is monotone and smooth: deviance 4.007 (MLE) → 3.979 at alpha 0.03 →
    4.87 at 1; the estimated dispersions are 0.12–0.18. `dispersion=None` keeps the
    maximum-likelihood fit; a penalty needs a fixed dispersion.
  - **No isotonic calibration.** With the LightGBM classifier it mapped a validation building
    with 17 kindergarten children (raw probability 0.20) to exactly 0: the calibrator's lowest
    score block held no kindergarten weight. An infinite log loss stops the study; a
    sample-size effect of isotonic's tail, by the user's rule dropped.
  - **The LightGBM classifier's calibration is searched** ({None, temperature, sigmoid}, ≈ 3.7
    s per trial on average, 1.8 min per study); the logistic model's too.
  - **The GLMs also run without the exposure**, as size-blind baselines (no GLM variant lists
    `n_apartments`).
  - **Model 2's totals are Model 1's 32 settings on the row sums**, each tuned once; the 224
    pairs are assembled from the tuned components (E4: the components are independent).
  - **`tune_component` takes `cv`, `cv_groups` and `n_trials` explicitly** (no closure over
    the cell's inputs).
  - **The parameter classes stay** (`FloatParameter` etc.): the search space is data,
    loggable, validated at build through Optuna's distributions (`_distribution`).
  - **Names:** `glm_base` / `count__*` / `probability__*` for the GLM variants;
    `COUNT_GLM_VARIANTS`, `PROBABILITY_GLM_VARIANTS`, `TREE_VARIANTS`; `CALIBRATION` is one
    parameter. FEATURE_TRANSFORMATIONS.md keeps its own stage vocabulary (`total_base`,
    `cohort_probability_base`), which the notebook comments cite by section.
- **Decisions of 2026-10-15 (evening), reversing the morning's where they conflict:**
  - **The composite is tuned, not its components** (§1, 3c): one study per model family;
    the loss, the feature variant and the exposure are **per-cohort categoricals inside
    the study**, no longer settings. The morning's measurement (a categorical loss split 30
    trials 10/20 and did worse than either loss alone) is the reason for `n_trials = 200`
    and for reporting per-arm trial counts, not a reason to re-ask.
  - `n_trials = 200`; the studies of one model run in parallel; only the best trial's
    combination is refit, scored on training and validation; Model 2's total choices are a
    list the user edits after Model 1. The plan's E4 and E5 are superseded (S5 rewrites them).
- **The user's style:**
  - names: `df` for frames, the `_df` suffix for stage tables, `simulation_res` for a result;
  - explicit, named steps rather than nested calls; grouped tuple assignments are fine;
  - no validation check without measured evidence that a failure would pass silently;
  - fewer, smaller cells and `mo.md` text views of tables; no print-only check cells;
  - a function's inputs in its signature, not closed over; every class, field and default
    justified by measured evidence; a rejected design is discarded by restoring from HEAD.

## 4. How to work

The standing rules are in [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3. In
short:
- **Before editing:**
  - Plan mode first: re-verify the plan's facts for the step, summarize the step in the chat
    as sub-steps the user can verify, and ask for approval.
  - One step per approval. Any change to `src` is its own stop.
  - Run the baseline `uv run pytest -m "not slow"` before the first edit.
- **Git:** the user commits and pushes. No `Co-Authored-By` line. Never `git checkout`,
  `stash`, `reset` or `mv`; move files with plain `mv`; restore a file from HEAD with
  `git show HEAD:<path> > <path>` and `cmp`.
- **Evidence:**
  - Probes are read-only: `.venv/bin/python -W error -c "..."` with `PYTHONPATH=src`.
  - Justify every class, field, method and check with measured evidence, or drop it.
  - One mutation per claimed behaviour: edit, see the test fail, then restore by copy and
    `cmp`. Clear `__pycache__` and set `PYTHONDONTWRITEBYTECODE=1`.
- **The notebook:**
  - **The user edits the notebook between turns** in the marimo editor. Record its md5 after
    every edit of yours and `cmp` before the next; if it changed, show the diff and ask. HEAD's
    notebook: md5 `723c40d98f9faf64078f5ecc51338010`.
  - **marimo's editor trims each cell's `return`** to the names other cells read. A new cell
    that reads a name must re-add it to the producing cell's return.
  - **An undefined name in a cell fails silently** in a script run. Verify every new cell
    ran: import the notebook with `importlib`, call `app.run()` under `-W error`, and check
    the returned defs hold the new names (used throughout 3b).
  - A new printed line for the test needs the user's approval.
  - `ruff format --diff` on the notebook shows three hunks in the user's cells above line 640;
    leave them. Format your own cells by hand to the formatter's output.
  - Tell review subagents to write nothing into the repository: one wrote a scratch copy of
    the notebook into `notebooks/` for a comparison (and removed it). The scratchpad is for
    that.
- **At each step:**
  - ruff, then `uv run mypy` and `uv run mypy src/age_group_prediction/<package>
    <changed tests>`;
  - `marimo check` on the notebook (two known empty-cell warnings);
  - the changed tests under `-W error`, and the non-slow suite;
  - an independent review subagent, with each finding reproduced before acting;
  - a file-by-file summary and the suggested commit commands.
- **No new ruff errors and no new format drift.** The notebook has 24 known ruff errors (the
  user's bare `return`s, F841 on the split parts not yet returned, an unused `clone` import,
  one B018). Whole-file `ruff format` is a separate pass the user will ask for.
- **After editing `pyproject.toml`,** use `uv run --frozen`. `uv lock` and `uv sync` are the
  user's.
- **On this machine:**
  - A plain `.venv/bin/python` does not see `src`, because Python 3.13 skips the hidden
    `.pth`. The notebook adds `src` itself, `.env` sets `PYTHONPATH=src`, and the notebook
    test passes `PYTHONPATH=src` to its subprocess.
  - `timeout` does not exist on macOS: run long commands in the background.

## 5. Open items outside this flow

- **The notebook:** two empty cells (`marimo check` warns) and seven bare `return`s (ruff
  `PLR1711`) that marimo writes. Both are the user's; ask before changing them. The section-4
  header still says "the check below prints it" and "fitted once on the training part as a
  check": the check cell is gone (user's text; ask).
- **The plan doc's D3** still names the old variant keys (`total_*`, `composition__*`); 3d
  rewrites D2/D3.
- **The simulator plan's Stop 4:** the doc examples, SPLITTING.md §3 among them, still show
  `StudentPopulationSimulator(config).run()`. They become `from_config(config).simulate().dataset`.
- **SPLITTING.md §3** says the group-aware methods raise when the second call gets the whole
  table. That holds only if the groups are not re-passed too.
- **The simulator plan's cleanup stop:** delete the dead code (`positive_part`,
  `positive_part_interaction`, `normalize_positive_weights`, `to_cohort_long`,
  `ApartmentSettings`), then a format-only commit.
- **The notebook test** still sets `PYTHONPATH=src`, so it does not cover the notebook's own
  `sys.path` line. Offered to the user, not decided.
- **A categorical arm can starve** (measured at 30 trials: a joint loss study split 10/20
  and lost to either loss alone). S3's table shows per-arm trial counts of every
  categorical, so an under-tuned arm is visible before a family is declared worse.
- **A `log(n_apartments)` plan for the GLMs** (FEATURE_TRANSFORMATIONS.md §4.4) would turn the
  size-blind no-exposure GLM runs into a real comparison. Not asked for.

## 6. The prompt for the next session

```text
Read docs/EXPERIMENT_NOTEBOOK_HANDOFF.md in full (§1 item 3 is the new design and its
stops S1–S5; §3 the decisions; §4 how to work), then docs/EXPERIMENT_NOTEBOOK_PLAN.md
§3 and §4 and docs/HYPERPARAMETER_TUNING.md §3, §4, §9. Style to match:
src/age_group_prediction/modeling/independent_cohorts.py and
src/age_group_prediction/hyperparameter_tuning/evaluator.py for src, their tests for
tests.

Expected state: branch feat/experiment-notebook, HEAD 2c0859c or one commit later (the
handoff commit), tree clean. Verify HEAD, git status and the branch first; if anything
the handoff does not name has changes, stop and show me the diff. I commit, push and
merge myself: never commit, never add a Co-Authored-By line, never git checkout/stash/
reset/mv.

Task: stop S1 of the handoff (§1, item 3c): IndependentCohortModels exposes
cohort_models__<cohort>__<param> to get_params/set_params. Work in plan mode: re-verify
the handoff's facts for it with read-only probes (PYTHONPATH=src .venv/bin/python
-W error -c "..."), run the baseline uv run pytest -m "not slow" (expect 1305 passed,
1 skipped, 1 xfailed), and ask for approval before editing. One stop per approval; say
if S1 and S2 can share one. Do not re-ask the decisions of §3.

At each stop give me: a file-by-file summary; the check results (ruff and format on the
changed files, uv run mypy and uv run mypy src/age_group_prediction/<package> <changed
tests>, the changed tests under -W error, one mutation per claimed behaviour restored by
copy and cmp, the non-slow suite); an independent review subagent's findings reproduced
(tell it to write nothing into the repository); and suggested commit commands.
```
