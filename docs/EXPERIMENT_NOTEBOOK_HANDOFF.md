# Handoff: the experiment notebook, from tuning to MLflow

**For:** the session that continues `notebooks/02_experiments.py`. **Written:** 2026-10-15,
after the feature variants (plan stop D1d) were committed.
**Plan (source of truth):** [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md): §3 (how
to work), §4 (decisions E1–E12) and §9 (stops D2–D5). Read those three sections in full before
the first edit. This file orients you and records the user's order of work and their notes.
Re-verify every fact here against the code of the day.

## 1. What the task is, in the user's order

Each step is one or more plan stops. Plan each in plan mode, one step per approval.

1. **Train, validation and test split** (D1c). **Done:** `537158a`.
2. **Feature-engineering configurations** (D1d). **Done:** `98640ba`, `efbfeb8`.
3. **Hyperparameter tuning of the two models, by CV on the training part** (the studies of plan
   stops D2/D3, without MLflow at first). **Next.**
   - **The user's notes for this step (2026-10-15):**
     1. **The plan starts with how to run the Optuna tuning efficiently for both models**:
        the number of studies and fits, what runs once and what per trial, the measured wall
        time of one trial, and what that implies for the budget.
     2. **Compare `use_exposure=True` and `False`, and LightGBM's Gaussian and Poisson
        losses** (`objective="regression"` and `objective="poisson"`).
        - E5 already makes exposure on/off a **setting** (two runs with the same seeds and
          folds), not a parameter. Do not re-ask it.
        - The objective is **not decided**. Decide with evidence whether it is a setting or a
          `CategoricalParameter`, and ask the user.
        - `CountModel` with a Gaussian LightGBM and `use_exposure=True` fits the weighted
          rate; [FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md) §3.3 and §8.1 allow
          it.
     3. **The implementation should rely on the `hyperparameter_tuning` package.** Answer with
        evidence: can it be used as is, does it overcomplicate the notebook, can it be
        simplified?
        - The package: 4 modules, 677 lines (`parameters.py` 170, `evaluator.py` 155,
          `aggregation.py` 125, `study.py` 178).
        - Its parts: `FloatParameter`, `IntParameter`, `CategoricalParameter`;
          `CVHyperparameterEvaluator(model, parameters, *, cv, metric, aggregation)` with
          `evaluate(trial, X, y, groups, *, exposure)` and `build_model(params)`; the
          aggregations `WeightedMean`, `Mean`, `UpperBound`; `HyperparameterStudy(seed,
          n_trials)` with `optimize(objective) -> TuningResult`.
        - Any change to `src` is its own stop, approved by the user, with tests and mutations.
   - **Guide:** [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) (§2 worked example,
     §3 parameters, §4 evaluator, §5 aggregation, §6 study, §7 pruning, §8 reproducibility,
     §11 what remains).
   - **Model 1:** `IndependentCohortModels`, one `CountModel` per cohort. One study per cohort
     on `POISSON_DEVIANCE` of that cohort's column (E4).
   - **Model 2:** `TotalTimesProbabilityModel`. Two studies (E4):
     - the total `CountModel` on `POISSON_DEVIANCE` of `Y.sum(axis=1)` (the dataset has no
       total column; the sum is the total);
     - the `CohortProbabilityModel` on `COHORT_LOG_LOSS` of `Y`.
   - Study i gets `seed + i` (E12). The budget is `n_trials=30` and `n_splits=5`; record the
     wall time.
   - The feature variants per model: `tree` / `tree__ses_quadratic` for LightGBM,
     `glm_base` / `count__*` for the count models (Model 1's cohorts, Model 2's total),
     `glm_base` / `probability__*` for the probability model (§2 below; renamed
     2026-10-15 from `total_*` / `cohort_probability_base` / `composition__*`).
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

- **Branch:** `feat/experiment-notebook`, HEAD `efbfeb8`, tree clean.
- **Non-slow suite:** 1295 passed, 1 skipped, 1 xfailed.
- **The notebook test** (`tests/characterization/test_experiments_notebook.py`) runs the
  notebook as a script and reads its printed lines. Four tests: the two settings lines, the
  data overrides, the split sizes, and the seed override. Keep those prints.
- **`notebooks/02_experiments.py`, as the user left it:**
  - **Section 1, data:** as before (`df`, `X` with 9 columns including `n_apartments`, `Y`,
    `groups`, `exposure` from `ExposureTransformer`, `id_col`).
  - **Section 2, settings:** `seed=42`, `split_method="random"` (the user's call, "for now"),
    `test_size=0.2`, `val_size=0.2`, `n_splits=5`, `n_trials=30`, `n_resamples=1000`. The
    cell returns only `n_splits, seed, split_method, test_size, val_size`; add `n_trials` and
    `n_resamples` when a cell reads them.
  - **Section 3, split:**
    - The split cell selects by **position** with `.iloc`. The second call's positions are
      mapped with `train_val_index[...]`. It prints `split n_train=917 n_val=306 n_test=306`.
    - It defines 15 parts: `X_`, `Y_`, `exposure_`, `groups_`, `building_id_` for `train`,
      `val` and `test`. It returns only `split_groups, splitter, train_index` today.
    - A separate cell defines `cv` and `cv_groups`. `cv_groups` is `None` under `random`:
      `KFold` raises under `-W error` when given groups. The cell returns nothing today.
  - **Section 4, feature variants:**
    - A shared-plans cell names each plan once, and a variants cell builds
      `FEATURE_VARIANTS`, 12 declarations (renamed 2026-10-15: one `glm_base` replaces the
      identical `total_base` and `cohort_probability_base`; `total__*` → `count__*`,
      `composition__*` → `probability__*`):
      - tree: `tree`, `tree__ses_quadratic`;
      - count models: `glm_base`, `count__ses_quadratic`, `count__daycare_log1p`,
        `count__room_share_x_household_size`, `count__room_share_x_ses`,
        `count__room_share_x_household_size_and_ses`;
      - probability model: `glm_base`, `probability__daycare_log1p`,
        `probability__room_share_x_median_age`, `probability__room_share_x_daycare`,
        `probability__room_share_x_median_age_and_daycare`.
    - `n_apartments` is a plan only in the tree variants (verified by probe 2026-10-15). The
      daycare operand of the two daycare interactions is centered (FEATURE_TRANSFORMATIONS.md
      §8.5.4).
    - The user removed the check cell, so no test covers the variants. A GLM variant listing
      `n_apartments` would fit unnoticed until a model is fitted on it.
    - The cell returns nothing today; return `FEATURE_VARIANTS` when a cell reads it.

## 3. Decisions already made (do not re-ask)

- **The plan's E1–E12** (§4 of the plan). The ones that matter now:
  - E3: the primary metric is `COHORT_POISSON_DEVIANCE` on validation;
  - E4: how each model is tuned (one study per cohort; two studies for Model 2);
  - E5: exposure on/off is a setting;
  - E7/E8: importance and SHAP;
  - E9: what a run logs;
  - E10: the helper only after repetition;
  - E11: the final refit on train + validation;
  - E12: one seed.
- **The exposure rule:** `n_apartments` is a feature and the offset for the trees, the
  exposure only for the GLM stages. `ExposureTransformer` stays: it is the one check that
  catches a zero or NaN exposure, which LightGBM would accept silently.
- **The split by position** with `.iloc`, and `split_method="random"` for now.
- **The 13 feature variants**, and no check or print-only cell for them.
- **The user's style:**
  - names: `df` for frames, the `_df` suffix for stage tables, `simulation_res` for a result;
  - explicit, named steps rather than nested calls; grouped tuple assignments are fine;
  - no validation check without measured evidence that a failure would pass silently;
  - fewer, smaller cells and `mo.md` text views of tables; no print-only check cells.

## 4. How to work

The standing rules are in [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3. In
short:
- **Before editing:**
  - Plan mode first: re-verify the plan's facts for the step, summarize the step in the chat
    as sub-steps the user can verify, and ask for approval.
  - One step per approval.
  - Run the baseline `uv run pytest -m "not slow"` before the first edit.
- **Git:** the user commits and pushes. No `Co-Authored-By` line. Never `git checkout`,
  `stash`, `reset` or `mv`; move files with plain `mv`.
- **Evidence:**
  - Probes are read-only: `.venv/bin/python -W error -c "..."` with `PYTHONPATH=src`.
  - Justify every class, field, method and check with measured evidence, or drop it.
  - One mutation per claimed behaviour: edit, see the test fail, then restore by copy and
    `cmp`. Clear `__pycache__` and set `PYTHONDONTWRITEBYTECODE=1`.
- **The notebook, lessons of 2026-10-15:**
  - **The user edits the notebook between turns** in the marimo editor. Before every edit,
    `cmp` it against your last copy; if it changed, show the diff and ask.
  - **marimo's editor trims each cell's `return`** to the names other cells read. A new cell
    that reads a name must re-add it to the producing cell's return. Ruff's F841 on the
    unread parts is expected until then.
  - **An undefined name in a cell fails silently:** the script run exits 0 and `marimo check`
    passes, but the cell does not run. Read every new cell's inputs by hand.
  - A new printed line for the test needs the user's approval: they removed a check cell
    twice.
- **At each step:**
  - ruff, then `uv run mypy` and `uv run mypy src/age_group_prediction/<package>
    <changed tests>`;
  - `marimo check` on the notebook;
  - the changed tests under `-W error`, and the non-slow suite;
  - an independent review subagent, with each finding reproduced before acting;
  - a file-by-file summary and the suggested commit commands.
- **No new ruff errors and no new format drift.** Whole-file `ruff format` is a separate pass
  the user will ask for; don't reformat code you didn't touch.
- **After editing `pyproject.toml`,** use `uv run --frozen`. `uv lock` and `uv sync` are the
  user's.
- **On this machine:**
  - A plain `.venv/bin/python` does not see `src`, because Python 3.13 skips the hidden
    `.pth`. The notebook adds `src` itself, `.env` sets `PYTHONPATH=src`, and the notebook
    test passes `PYTHONPATH=src` to its subprocess.
  - `timeout` does not exist on macOS: run long commands in the background.

## 5. Open items outside this flow

- **The notebook:** two empty cells (`marimo check` warns) and eight bare `return`s (ruff
  `PLR1711`) that marimo writes. Both are the user's; ask before changing them.
- **The simulator plan's Stop 4:** the doc examples, SPLITTING.md §3 among them, still show
  `StudentPopulationSimulator(config).run()`. They become `from_config(config).simulate().dataset`.
- **SPLITTING.md §3** says the group-aware methods raise when the second call gets the whole
  table. That holds only if the groups are not re-passed too.
- **The simulator plan's cleanup stop:** delete the dead code (`positive_part`,
  `positive_part_interaction`, `normalize_positive_weights`, `to_cohort_long`,
  `ApartmentSettings`), then a format-only commit.
- **The notebook test** still sets `PYTHONPATH=src`, so it does not cover the notebook's own
  `sys.path` line. Offered to the user, not decided.
