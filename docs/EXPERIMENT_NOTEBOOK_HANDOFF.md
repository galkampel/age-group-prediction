# Handoff: the experiment notebook, from the split to MLflow runs

**For:** the session that continues `notebooks/02_experiments.py`. **Written:** 2026-10-14,
after the simulator refactor's Stop 3 was committed.
**Plan (source of truth):** [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md): §3 (how
to work), §4 (decisions E1–E12) and §9 (stops D1c–D5). Read those three sections in full before
the first edit. This file orients you and records the user's order of work. Re-verify every
fact here against the code of the day.

## 1. What the task is, in the user's order

Each step is one or more plan stops. Plan each in plan mode, one stop per approval.

1. **Train, validation and test split** (plan stop D1c).
   - The pattern: [SPLITTING.md](SPLITTING.md) §3, two `Splitter.train_test_indices` calls with
     the sub-positions mapped back to table positions, then `splitter.cv(n_splits=…,
     random_state=seed)` for the training part.
   - `Splitter` is in `age_group_prediction.splitting`; `take_rows` is in
     `age_group_prediction.utils`.
   - Re-measured 2026-10-14 on today's data, `split_method="grouped"`, seed 42: n_train=908,
     n_val=311, n_test=310.
   - The plan's appendix C5 in [SIMULATOR_WHITE_BOX_PLAN.md](SIMULATOR_WHITE_BOX_PLAN.md) has
     the cell code, test and mutation (it uses `df`, the notebook's table).
2. **Feature-engineering configurations** (plan stop D1d), including whether `n_apartments`
   stays a feature when it is the exposure.
   - The declarations are [FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md) §8: `tree`
     (§8.1), `total_base` (§8.2), `cohort_probability_base` (§8.3), and the §8.5 candidates.
   - The exposure question is already decided per model family. Verify it, don't re-decide it:
     - **LightGBM (`tree`):** `n_apartments` is a feature *and* the offset, on purpose (§3.3;
       the note under §8.1). The offset asserts proportionality and the feature lets the trees
       learn departures from it.
     - **GLM total stage and composition stage (`total_base`, `cohort_probability_base`):**
       `n_apartments` is the exposure and never a feature (§4.4, §5, §8.2). As a feature, it
       would let the model partly undo the offset.
     - **Exposure off** (E5) is a separate setting and run, not a parameter.
   - **The catch:** the rebuilt `FeatureTransformer` does not enforce any of this. Only the
     declaration does (§4.4, §8.2). The notebook's `X` currently keeps `n_apartments` for
     every model, so check, with a probe, that each declaration's output has or lacks
     `n_apartments` as above.
   - Plan stop D1d expects column counts per variant: 10/9/9/10/9/12/9/12. Re-measure them; the
     dataset lost `n_children_total` since they were written, but `X` never held it.
3. **Hyperparameter tuning of the two models, by CV on the training part** (the studies of plan
   stops D2/D3, without MLflow at first).
   - Guide: [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) (§2 worked example, §3
     parameters, §4 evaluator, §6 study, §8 reproducibility).
   - Classes: `CVHyperparameterEvaluator`, `HyperparameterStudy` and the `*Parameter` classes,
     from `age_group_prediction.hyperparameter_tuning`.
   - **Model 1:** `IndependentCohortModels`, one `CountModel` per cohort. One study per cohort
     on `POISSON_DEVIANCE` of that cohort's column (E4).
   - **Model 2:** `TotalTimesProbabilityModel`. Two studies (E4):
     - the total `CountModel` on `POISSON_DEVIANCE` of `Y.sum(axis=1)` (the dataset has no
       total column; the sum is the total);
     - the `CohortProbabilityModel` on `COHORT_LOG_LOSS` of `Y`.
   - Study i gets `seed + i` (E12). The budget is `n_trials=30` and `n_splits=5`; record the
     wall time.
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

## 2. State of the branch (2026-10-14)

- **Branch:** `feat/experiment-notebook`, HEAD `e66b97d`, tree clean. The commits since `main`:
  - D1a/D1b (`2b13970`);
  - the simulator refactor, Stops 1–3 (`6b90805`, `59d15d6`, `e66b97d`).
- **Non-slow suite:** 1294 passed, 1 skipped, 1 xfailed.
- **The simulator** (see [SIMULATOR_WHITE_BOX_PLAN.md](SIMULATOR_WHITE_BOX_PLAN.md)):
  - `StudentPopulationSimulator(<five components>, n_neighborhoods=…, seed=…)`, or
    `.from_config(config)`;
  - `.simulate(rng=None)` returns a `SimulationResult` with three fields:
    - `dataset`: one row per building, 15 columns, ending with the three cohort counts. It has
      **no `n_children_total`**; the total is their sum;
    - `building_effects` and `neighborhood_effects`: the latent effects, one frame per owner
      level.
- **`notebooks/02_experiments.py`**, as the user restructured it:
  - **Cell 1:** imports `mo` and `pd`, and puts `<repo>/src` on `sys.path`. Python 3.13 on macOS
    skips the venv's hidden editable-install `.pth`; see §4.
  - **Imports cell:** the simulator components, `ShareTransformer`, `ExposureTransformer`, and
    `hashlib`.
  - **Config cell:** the overrides (150 neighborhoods, building rate 9.0, the seed) and
    `DATA_SETTINGS`.
  - **Simulator cell:** the explicit construction,
    `simulation_res = simulator.simulate()`, `df = simulation_res.dataset`, then a named
    `share_transformer` (3 rooms is the omitted share). It returns `df` only.
  - **Hash cell:**
    - `data_hash` (now `e1dd694935d0…`);
    - `X = df.drop(columns=[id, group, targets])`, 9 columns that **still include
      `n_apartments`**;
    - `Y` (the 3 cohorts) and `groups`;
    - `exposure = ExposureTransformer("n_apartments").fit_transform(df)`.

    It returns only `X`, `data_hash` and `groups`. Return `Y` and `exposure` when a later cell
    needs them.
  - **Display cell:** `config_params` (the flattened config), the two printed lines and the
    `mo.md` views.
  - **Experiment-settings cell:** `seed=42`, `split_method="grouped"`, `test_size=0.2`,
    `val_size=0.2`, `n_splits=5`, `n_trials=30`, `n_resamples=1000`, and its printed line.
- **The notebook test** (`tests/characterization/test_experiments_notebook.py`) runs the
  notebook as a script and reads three printed lines: `data_settings`, `data rows=…
  neighborhoods=… data_hash=…` and `experiment_settings`. Keep those prints. The plan grows the
  test by one test per stop.

## 3. Decisions already made (do not re-ask)

- **The plan's E1–E12** (§4 of the plan). The ones that matter now:
  - E3: the primary metric is `COHORT_POISSON_DEVIANCE` on validation;
  - E4: how each model is tuned;
  - E5: exposure on/off is a setting;
  - E6: the split;
  - E7/E8: importance and SHAP;
  - E9: what a run logs;
  - E10: the helper only after repetition;
  - E11: the final refit on train + validation;
  - E12: one seed.
- **The exposure rules** in §1 step 2 above. `ExposureTransformer` stays: it is the one check
  that catches a zero or NaN exposure, which LightGBM would accept silently.
- **The user's style, from the simulator refactor:**
  - names: `df` for frames, the `_df` suffix for stage tables, `simulation_res` for a result;
  - explicit, named steps rather than nested calls; a named transformer rather than an inline
    one;
  - no validation check without measured evidence that a failure would pass silently;
    invariants are tested where they are produced;
  - one wide frame per level for owner-level data.
- **Notebook layout:** the user prefers small, separate cells and `mo.md` text views of
  tables.

## 4. How to work

The standing rules are in [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §3. In
short:
- **Before editing:**
  - Plan mode first: re-verify the plan's facts for the stop, summarize the stop in the chat
    as sub-steps the user can verify, and ask for approval.
  - One stop per approval.
  - Run the baseline `uv run pytest -m "not slow"` before the first edit.
- **Git:** the user commits and pushes. No `Co-Authored-By` line. Never `git checkout`,
  `stash`, `reset` or `mv`; move files with plain `mv`.
- **Evidence:**
  - Probes are read-only: `.venv/bin/python -W error -c "..."` with `PYTHONPATH=src`.
  - Justify every class, field, method and check with measured evidence, or drop it.
  - One mutation per claimed behaviour: edit, see the test fail, then restore by copy and
    `cmp`. Clear `__pycache__` and set `PYTHONDONTWRITEBYTECODE=1`; same-size mutations within
    one second otherwise reuse a stale `.pyc`.
- **At each stop:**
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
- **Imports on this machine:** a plain `.venv/bin/python` does not see `src`, because Python
  3.13 skips the hidden `.pth`.
  - The notebook adds `src` itself, and `.env` sets `PYTHONPATH=src`.
  - The notebook test passes `PYTHONPATH=src` to its subprocess.
  - The durable fix is the user's:
    `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`.

## 5. Open items outside this flow

- **The simulator plan's Stop 4:** the doc examples still show `StudentPopulationSimulator(config).run()`
  or `create_dataset`. They become `from_config(config).simulate().dataset`, each re-run as
  written.
- **The simulator plan's cleanup stop:** delete the dead code (`positive_part`,
  `positive_part_interaction`, `normalize_positive_weights`, `to_cohort_long`,
  `ApartmentSettings`), then a format-only commit.
- **The notebook:** an empty cell (`marimo check` warns), and five bare `return`s (ruff
  `PLR1711`) that marimo 0.25.1 writes. Both are the user's edits; ask before changing them.
- **The notebook test:** it still sets `PYTHONPATH=src`, so it does not cover the notebook's
  own `sys.path` line. Dropping it, plus a `src` link in the seed-override test's scratch copy,
  would cover it. This was offered to the user, who has not decided.
