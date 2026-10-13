# Plan: the experiment notebook (tuning, splitting, importance, MLflow)

**Written for:** the model that implements each PR, and the user who validates
it. Self-contained: everything an implementer needs is here or linked. Re-verify
every fact against the code of the day before relying on it (file paths, line
counts, test counts drift).

**Status (2026-10-12):** PR A done: the tuning study (A0–A5), an independent
validation ([HYPERPARAMETER_TUNING_VALIDATION.md](HYPERPARAMETER_TUNING_VALIDATION.md):
ACCEPT WITH NOTES, findings processed), PR #5 squash-merged into `main` as
`ba2aa11` (A6). PR B: B1 committed (`2d4f0c5`: `COHORT_POISSON_DEVIANCE`,
non-slow suite 1252 passed); B2 done (the split pattern tested for every method
and documented in `SPLITTING.md` §3; no new code), awaiting the user's commit.
**Next:** open PR B on GitHub, then PR C.

**Branch state:** `feat/split-and-cohort-deviance` (PR B), from `main` at `ba2aa11`.

## Contents

1. [Goal](#1-goal)
2. [What exists today](#2-what-exists-today)
3. [How to work](#3-how-to-work)
4. [Decisions](#4-decisions)
5. [The PRs in order, and why this order](#5-the-prs-in-order-and-why-this-order)
6. [PR A: finish the tuning study](#6-pr-a-finish-the-tuning-study)
7. [PR B: the three-way split and the cohort count metric](#7-pr-b-the-three-way-split-and-the-cohort-count-metric)
8. [PR C: feature importance](#8-pr-c-feature-importance)
9. [PR D: the notebook, the data and MLflow](#9-pr-d-the-notebook-the-data-and-mlflow)
10. [Out of scope and later](#10-out-of-scope-and-later)
11. [Commands](#11-commands)

---

## 1. Goal

A marimo notebook, `notebooks/03_experiments.py`, that runs one MLflow
experiment (`age-group-prediction`, or the scratch name from `.env`) on a
synthetic population. **Each cell is one run, and one run is one setting:**

| axis | values today | values later |
|---|---|---|
| composite model | `IndependentCohortModels` (Model 1), `TotalTimesProbabilityModel` (Model 2) | the Bayesian NB2 × Dirichlet-multinomial model (Model C) |
| model family per component | LightGBM (Poisson objective), `PoissonRegressor`, `NegativeBinomialRegressor` (NB2); classifiers `LogisticRegression`, `LGBMClassifier` | any scikit-learn regressor / classifier |
| feature engineering | the declarations of `docs/FEATURE_TRANSFORMATIONS.md` §8 (`tree`, `total_base`, `cohort_probability_base`, the §8.5 candidates) | |
| exposure | `use_exposure=True` / `False` on each `CountModel` | |

In each run: tune the setting's hyperparameters by k-fold cross-validation on
the **training** part, refit the tuned model on the training part, score it on
the **validation** part (with bootstrap intervals), compute feature importance,
and log parameters, metrics and artifacts. The validation scores of all runs are
compared in one table. At the very end, **once**, the best setting is refit on
train + validation and scored on the **test** part, which stands for production.

Example run: Model 1 with a LightGBM `CountModel` per cohort, features `tree`,
exposure on; tuned per cohort; compared with the same run with exposure off and
with the §8.5 feature candidates.

---

## 2. What exists today

Verified 2026-10-11 on `feat/hyperparameter-tuning`. Paths are under
`src/age_group_prediction/` unless stated.

**Models (`modeling/`), documented in
[INDEPENDENT_COHORT_MODELS.md](INDEPENDENT_COHORT_MODELS.md) and
[TOTAL_TIMES_PROBABILITY_MODEL.md](TOTAL_TIMES_PROBABILITY_MODEL.md).**

- `BaseAgeGroupModel` (`modeling/base.py`): `fit(X, y, exposure=None)`,
  `predict(X, exposure=None)`, `evaluate(y_true, y_pred, metric) -> float`.
  Rows are paired by **position**. `get_params`/`set_params`/`clone` work
  (scikit-learn `BaseEstimator`).
- `CountModel(*, estimator, use_exposure=False, feature_transformer=None)`:
  one count target. An estimator taking `exposure` (NB2) gets it raw; one taking
  `sample_weight` (Poisson GLM, LightGBM) fits the weighted rate.
- `CohortProbabilityModel(*, estimator, replication="weighted",
  calibration_method=None, calibration_cv=5, feature_transformer=None)`:
  `y` is the (buildings × cohorts) count DataFrame; `predict` returns
  probabilities, rows summing to 1.
- `IndependentCohortModels(cohort_models: Mapping[str, BaseAgeGroupModel])`:
  **`set_params` does not reach into the mapping** (`independent_cohorts.py`,
  class docstring). Fitted copies in `cohort_models_`.
- `TotalTimesProbabilityModel(*, total_model, probability_model)`: nested names
  work (`total_model__estimator__alpha`, `probability_model__calibration_method`).
  Fitted copies in `total_model_`, `probability_model_`. `predict` = total ×
  probabilities, a DataFrame with `y`'s columns.
- Each leaf model owns its `feature_transformer` and fits a copy inside `fit`;
  composites pass the raw table down.

**Tuning (`hyperparameter_tuning/`), plan
[HYPERPARAMETER_TUNING_PLAN.md](HYPERPARAMETER_TUNING_PLAN.md).**

- Done: `FloatParameter`, `IntParameter`, `CategoricalParameter` (`name` is the
  `set_params` key), `CVHyperparameterEvaluator(model, parameters, *, cv, metric,
  aggregation=WeightedMean())` with `evaluate(trial, X, y, groups=None, *,
  exposure=None) -> float` (negated for a greater-is-better metric, so a
  study always minimizes; since A3, see the tuning plan's D12) and
  `build_model(params)`; aggregations `WeightedMean`, `Mean`, `UpperBound(z)`
  (`LowerBound` before A3), `corrected_std_error`.
- **Missing:** `study.py` with `HyperparameterStudy`, `TrialRecord`,
  `TuningResult` (that plan's §4.4; its §6 sub-tasks 3.1–3.5 and 4.1–4.3).

**Splitting (`splitting/`), documented in [SPLITTING.md](SPLITTING.md).**
`Splitter(method)` with `method` in `random`, `stratified_by_group`,
`grouped`; `train_test_indices(X, groups, *, test_size, random_state) ->
(train_index, test_index)` positions; `cv(*, n_splits, random_state)` the
matching cross-validator, to be given `groups_train`. `utils.take_rows(array,
index)` takes any row container by positions. **No three-way split exists.**

**Scoring (`scoring.py`).** `Metric(name, function, greater_is_better=False)`;
`POISSON_DEVIANCE` (scikit-learn's `mean_poisson_deviance`, **one column
only**: a (n, K) matrix raises "Multioutput not supported"), `RMSE`, `MAE`,
`COHORT_LOG_LOSS` (takes the (n, K) count matrix; per child; rows of `y_pred`
are normalized, so it ignores the total).

**Inspection (`inspection/`), plan
[MODEL_INSPECTION_PLAN.md](MODEL_INSPECTION_PLAN.md).** `BootstrapEvaluator(metric,
n_resamples, alpha=0.05, random_state=None).evaluate(model, X, y, groups=None,
*, exposure=None) -> BootstrapResult(mean, lower, upper)`: predicts once,
resamples rows or whole groups, no refit. Its sub-task 2 (smoke run +
`docs/MODEL_INSPECTION.md`) is open, and its status line still says sub-task 1
"awaits the user's commit" (stale: merged as `7d865ab`). Permutation importance
and SHAP are planned, not built:
[FEATURE_IMPORTANCE_PLAN.md](FEATURE_IMPORTANCE_PLAN.md) (decisions I3–I11,
sub-tasks 1–3, code shapes in its §4).

**Synthetic data (`src/student_simulator/`).**
`StudentPopulationSimulator(config).run(rng=None) -> DataFrame`, one row per
building: `building_id`, `neighborhood_id`, `ses`, `avg_household_size`,
`median_age`, `n_daycares_500m`, `school_status`, `3_rooms`…`6_rooms`,
`n_apartments`, `n_kindergarten`, `n_elementary`, `n_highschool`,
`n_children_total`. `Config` is a frozen pydantic model loaded by
`load_simulation_config("configs/simulation.toml")` (seed 42, 60
neighborhoods); `notebooks/02_model_fitting.py` lines ~128–142 show the
`model_copy(update=...)` override to 150 neighborhoods × 9 buildings.
`preprocessing.ExposureTransformer("n_apartments").fit_transform(table)` gives
the validated exposure; `preprocessing.ShareTransformer` the room shares.

**Feature declarations.** `tree` (§8.1), `total_base` (§8.2),
`cohort_probability_base` (§8.3) and the §8.5 candidates exist **only as code
blocks** in [FEATURE_TRANSFORMATIONS.md](FEATURE_TRANSFORMATIONS.md) §8; nothing
in `src/` defines them.

**MLflow.** `mlflow>=3.16,<4` is the optional dependency group `tracking`.
`tracking/` and `experiment/` are the **old stack**: they import `models/` and
`modeling_config.py` and cannot log the new models;
[MODEL_REIMPLEMENTATION_PLAN.md §5](MODEL_REIMPLEMENTATION_PLAN.md) step 5
deletes them. `hashing.py` is old-stack too. The repo root's `mlflow.db` and
`mlartifacts/` are the canonical store and **must never be written**; the
scratch store is `.mlflow-local/` via `.env` (template `.env.example`;
[MLFLOW_EXPERIMENTS_GUIDE.md](MLFLOW_EXPERIMENTS_GUIDE.md) §1–§3). Importing the
package root `age_group_prediction` loads the old stack (torch, pyro): **the
notebook and the new modules import submodules** (`from
age_group_prediction.modeling import ...`).

**Notebooks** are marimo `.py` files (`notebooks/01_eda.py`,
`notebooks/02_model_fitting.py`; the latter is the old stack's client). Run
with `uv run --env-file .env --group notebook --group tracking marimo edit
notebooks/<name>.py` from the repo root.

---

## 3. How to work

The user's standing rules, as [MODEL_INSPECTION_PLAN.md §3](MODEL_INSPECTION_PLAN.md)
states them; they apply to every sub-task below.

1. **One sub-task per stop.** Start each in plan mode: re-verify this plan's
   facts for it, write a short summary of the sub-task **in the chat**, and ask
   for approval. Implement only after it. The user says "plan A1" to start one.
2. **Baseline first:** `uv run pytest -m "not slow"` before the first edit. If
   it fails, stop and report; never work around it.
3. **If an assumption breaks, stop** and show the options with evidence.
4. **Routine:** implement → `uv run ruff check` and `uv run ruff format` on the
   changed files (paths listed explicitly; zsh does not split `$FILES`) →
   `uv run mypy`, then `uv run mypy src/age_group_prediction/<package> <changed
   tests>` (without the package path mypy skips the tests' calls into it) → the
   changed tests under `-W error` → one mutation check per claimed behaviour
   (edit `src` in place, see a test fail, restore by copy and `cmp`/md5; never
   `git checkout`/`stash`/`reset`) → an independent review subagent (6–8
   minutes: say so, run the suite and update this doc meanwhile; reproduce each
   finding before fixing or rejecting it) → the non-slow suite → update this
   doc's status line and the sub-task's checkbox → stop.
5. **At each stop:** a file-by-file summary of the `.py` changes, the check
   results, and suggested commit commands. **The user commits and pushes.**
   Never commit without explicit approval; no `Co-Authored-By` line.
6. **Justify every class, field, method and check**, or drop it. Validate only
   what would otherwise pass silently; leave to the library what it already
   raises. Add no default the user did not ask for. When the user asks "is X
   necessary?" or "why X over Y?", answer with measured evidence (probe the
   library: `PYTHONPATH=src .venv/bin/python -c "..."`, read-only) and size
   the change.
7. **API style** (scikit-learn): settings in the constructor, data as method
   arguments, never stored; a named method, not `__call__`; names `train`/`val`,
   `feature_transformer`, `exposure_*`; precise types, a named alias when a
   type repeats; in `modeling`, logic under a class (no utils module); dispatch
   as `if/elif/else: raise`, never an `else` that silently handles a case.
   Match `hyperparameter_tuning/evaluator.py` and `modeling/count_model.py`.
8. **Docs style:** a docstring says what the code does, inputs/outputs and the
   non-obvious constraint, in a few lines; a comment only for a "why" the code
   cannot show; derivations, figures and library quirks go in the docs. Every
   doc code block is run as written before the doc claims it. No placeholder
   figures.
9. **Tests:** each test's name states the mistake it catches; no test may only
   check a library. Tests live flat in `tests/unit/`.
10. **Docs policy for plan docs:** current decisions and remaining work only,
    updated in place; the commit messages keep the history.
11. After editing `pyproject.toml`, run `uv` with `--frozen`; `uv lock` and
    `uv sync` are the user's.

**Working notes (2026-10-12):**
- Files in the working tree were overwritten twice by stale editor buffers:
  docs, then `parameters.py`, reverted to older texts mid-session. Start with
  `git status`. If a file you did not touch is modified, stop and show the user
  the diff.
- When the tree cannot be trusted, validate from a `git archive HEAD` copy in
  the scratchpad. Assert the import path first: the `.venv` has an editable
  `.pth` pointing at the repo's `src`, so a scratch copy is ignored unless it
  comes first on the path (pytest: `-c <copy>/pyproject.toml --rootdir <copy>`).
- A test of a shape check needs non-square data: a transposed square matrix
  has the same shape (B1's first try).

---

## 4. Decisions

Numbered E1–E12. The user chose E3's primary metric, E4's two-study form and
E11's final refit on 2026-10-11; do not re-ask them.

**E1. Order: A (tuning study) → B (split + count metric) → C (importance) → D
(notebook: data, runs, MLflow).** The user proposed tuning → importance →
split → data → MLflow. One change: the split moves before importance. Why: B
is small and needs no new class; the importance smoke run must compare training
with validation rows, so "validation" has to be defined first; and D's first
cell needs B anyway. Data generation and MLflow stay last, inside the notebook
PR, as the user proposed: they only consume A–C. C imports nothing from
`hyperparameter_tuning`, so it may run as a branch parallel to A if the user
wants; the default is sequential.

**E2. One cohort-matrix count metric.** `cohort_poisson_deviance(y_true,
y_pred)` in `scoring.py`: scikit-learn's `mean_poisson_deviance` on the
flattened (buildings × cohorts) cells, i.e. the mean over all cells (each
cohort weighted exactly as its own `CountModel` is). Shapes must match, as in
`cohort_log_loss`. Exported as `COHORT_POISSON_DEVIANCE =
Metric("cohort_poisson_deviance", cohort_poisson_deviance)`. Why: the two
composites predict a (n, K) matrix and before B1 only `COHORT_LOG_LOSS` accepted it,
and that one ignores the total. It lands in **PR B**, not A: the study does not
need it and PR #5 is large enough to merge as is.

**E3. Metrics.**
- *Component studies* score what the component predicts: `CountModel` (a
  cohort, or the total) → `POISSON_DEVIANCE`; `CohortProbabilityModel` →
  `COHORT_LOG_LOSS`.
- *Across settings, on validation:* **`COHORT_POISSON_DEVIANCE` is the primary
  metric** (the deployment target is cohort counts; comparable across both
  composites). `COHORT_LOG_LOSS` is secondary and never selects (it normalizes
  rows, so a model with wrong totals and right proportions looks good).
- *Logged per refit model on validation:* `COHORT_POISSON_DEVIANCE`,
  `COHORT_LOG_LOSS`, per-cohort `POISSON_DEVIANCE`, `MAE` and `RMSE`, each as
  `BootstrapEvaluator` mean / lower / upper with `groups=neighborhood_id`.
- Deviance is compared only on the same `y` layout (cells against cells, never
  a total against cells). Scoring an NB2 total by Poisson deviance is fine: it
  is a proper scoring rule for the mean; say so in the docs.

**E3b. The same metrics must score the future Bayesian model (Model C).** They
do, by construction: E3's metrics are point-prediction metrics on `predict`'s
cohort matrix, and Model C's prediction is its posterior mean (total mean ×
composition mean, [BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md §4](BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md)),
so the deviance, log loss, MAE/RMSE, the bootstrap CI and permutation
importance apply unchanged once C is a `BaseAgeGroupModel`. What does **not**
carry over is the old stack's `joint_predictive_nll` (a posterior-integrated
log mass needing predictive draws); `Metric(y_true, y_pred)` has no place for
draws. Decision: the cross-setting comparison stays on point metrics, the only
ones all three models share (the old selection also compared across families
on composition log loss, RMSE and MAE). Distributional metrics (predictive NLL,
interval coverage / PIT) are added **with Model C's PR** as a second kind of
metric that takes predictive samples, logged only for models that produce
them. Rule for PRs A–D: never assume `y_pred` is only a mean (no metric
registry keyed on "the" prediction; keep `Metric` as is).

**E4. How each composite is tuned inside a run.**
- Model 2: **two studies**, one per component (user's choice): the total
  `CountModel` on `POISSON_DEVIANCE` of `y.sum(axis=1)`, the
  `CohortProbabilityModel` on `COHORT_LOG_LOSS` of `y`. The components are
  fitted independently, so their optima are independent and each search space
  is smaller. The tuned components are then assembled into one
  `TotalTimesProbabilityModel`, refit on the training part.
- Model 1: **one study per cohort** `CountModel`, on `POISSON_DEVIANCE` of that
  cohort's column (the tuning plan's §5 example), then
  `IndependentCohortModels({cohort: tuned_model})`, refit on the training part.
- In MLflow (E9): one **parent run per setting**, one **nested child run per
  study** (per cohort, or per component).

**E5. Exposure on/off is a setting (two runs), not a `CategoricalParameter`.**
A boolean parameter would be legal (no conditional parameters, D9 of the tuning
plan), but: `use_exposure` changes the target (weighted rate vs count), so the
other hyperparameters' good regions differ and one TPE budget would under-tune
the losing arm; "no exposure" normally re-adds `n_apartments` as a feature,
which is a feature-variant change anyway; and Model 1 cannot express it as a
parameter at all (E2 of §2). Two runs with the same seeds and folds, compared on
validation.

**E6. The three-way split needs no new class.** Positions, as everything else:

```python
splitter = Splitter(split_method)                    # a setting: see B2
train_val_index, test_index = splitter.train_test_indices(
    table, groups, test_size=0.2, random_state=split_seed)
groups_train_val = take_rows(groups, train_val_index)
train_sub, val_sub = splitter.train_test_indices(
    take_rows(table, train_val_index), groups_train_val,
    test_size=0.25, random_state=split_seed)         # 0.25 of 80 % = 20 % of all
train_index, val_index = train_val_index[train_sub], train_val_index[val_sub]
cv = splitter.cv(n_splits=5, random_state=cv_seed)   # given take_rows(groups, train_index)
```

The second call runs on the remainder, with the remainder's groups, and the
sub-positions are mapped back to table positions. PR B tests this for every
`Method` (parts disjoint, union = all rows, groups never straddle a boundary
for `grouped`, the small-group behaviour of `stratified_by_group` applied
twice). A `Splitter.train_val_test_indices` method is added **only if** those
tests show the two calls need care an implementer would get wrong; otherwise
`SPLITTING.md` documents the pattern and the notebook uses it inline.

**E7. Feature importance: per raw input column, the whole composite first.**
Permutation importance of the composite with the primary metric
(`COHORT_POISSON_DEVIANCE`) on the **validation** rows is the decision table:
it answers "what matters for the prediction we ship". The same on the
**training** rows is logged as an overfit diagnostic (a column important on
train and not on validation). Per-component importance (`total_model_` on
`POISSON_DEVIANCE` of the total, `probability_model_` on `COHORT_LOG_LOSS`) is
logged for Model 2 only; for Model 1 it equals the composite restricted to one
output column, so it is skipped. Raw columns (I4), not transformed ones:
transformed columns are not comparable across feature variants, which is an
experiment axis, and one-hot pieces or interactions cannot be permuted
independently. The simulator's columns are correlated by construction, so
permutation importance splits credit between them: stated in the docs, no
column blocks (I6).

**E8. SHAP on a seeded validation subsample.** `ShapExplainer` (I9–I11)
explains the rate, `predict(X, exposure=1)` (I5), per cohort output; global
importance is mean |SHAP|. Cost measured at ~0.45 s per row for Model 2, so:
`n_rows ≈ 200` validation rows, `n_background ≈ 50` **training** rows, both
seeded and logged as parameters. Leak-free: the feature transformer is fitted
inside `fit`, never on the explained rows.

**E9. What one run logs.**

| kind | content |
|---|---|
| params | setting: `composite`, `family_total` / `family_cohort` / `family_probability`, `feature_variant`, `use_exposure`; the simulator `Config` flattened from `model_dump()` (dotted keys) and a data hash (SHA-256 of the table's CSV bytes, own 3-line helper; `hashing.py` is old-stack); `split_method`, `split_seed`, `cv_seed`, `study_seed`, `estimator_seed`, `test_size`, `val_size`, `n_train` / `n_val` / `n_test`; the search space (parameter name → bounds); `n_trials`, `n_splits`, `aggregation` |
| metrics | E3's validation set (`val/<metric>` mean, `val/<metric>/lower`, `.../upper`); per child study `cv/best_value`, `cv/best_std_error` (nested runs), from `TuningResult.best_value` and the best record's `std_error`, in the metric's own sign (A3); `best_value` is the aggregation's value (the CV mean under `WeightedMean`, mean + z·SE under `UpperBound`), so `aggregation` is logged as a param beside it |
| artifacts | `trials.csv` (one row per `TrialRecord`), `best_params.json`, `model.joblib` (the refit model), `importance/permutation_val.csv`, `importance/permutation_train.csv`, `importance/shap_values.csv` (mean |SHAP| per column and cohort), plots `importance/permutation_val.png` (bar with error bars), `importance/shap_bar.png`, `importance/shap_beeswarm_<cohort>.png`, `features.txt` (`repr` of the feature transformers) |
| tags | `setting` (a short name, the cell's title), `final=true` on the one test run |

Test metrics (`test/<metric>`) appear **only** in the final run (E11).

**E10. MLflow helper: start inline, extract functions, no class.** The first
run cell uses `mlflow.start_run` and plain `log_params` / `log_metrics` /
`log_artifact` calls. When the second cell repeats them, the repeated parts
become functions (`log_tuning_result(result)`, `log_validation_metrics(...)`,
`log_importance(...)`) in **one new module**, `src/age_group_prediction/mlflow_runs.py`
(rename to `tracking.py` when the old package is deleted). Why no class:
nothing persists across the calls (the experiment and URI are process-global
through `mlflow.set_experiment`), so a class would carry no settings, and the
user's rule is that every class is justified. The module is **not** imported
from the package root. The old `tracking/` is left untouched.

**E11. The final evaluation refits on train + validation** (user's choice).
After the comparison table, the winning setting's `best_params` are rebuilt
with `evaluator.build_model(best_params)`, fitted on `train_index ∪ val_index`,
and scored once on `test_index` with the E3 bootstrap set; logged in a run
tagged `final=true`. The validation scores remain the selection evidence; the
test score is reported, never used to choose.

**E12. Seeds.** Four named seeds, all logged: `split_seed` (both
`train_test_indices` calls), `cv_seed` (`Splitter.cv`), `study_seed` (one per
study: `study_seed + i` for the i-th cohort / component), `estimator_seed`
(`random_state` of every estimator that takes one). The study runs its trials
one after the other with a fixed `n_trials`, so a seeded sampler reproduces
it (A4b dropped `n_jobs`, timeouts and `is_reproducible`: no client used them).

---

## 5. The PRs in order, and why this order

| PR | branch (from) | delivers | depends on |
|---|---|---|---|
| A | `feat/hyperparameter-tuning` (existing draft PR #5) | `HyperparameterStudy`, `TrialRecord`, `TuningResult`; `docs/HYPERPARAMETER_TUNING.md`; **PR #5 merged into `main`** | — |
| B | `feat/split-and-cohort-deviance` (from `main` after A) | `COHORT_POISSON_DEVIANCE`; the three-way split pattern, tested and documented | A merged (so B branches from a `main` that has the models) |
| C | `feat/feature-importance` (from `main` after B) | `PermutationImportance`, `ShapExplainer`; `docs/MODEL_INSPECTION.md`; closes the inspection plan | B (validation rows for the smoke run; the primary metric) |
| D | `feat/experiment-notebook` (from `main` after C) | feature declarations as code; `notebooks/03_experiments.py`; `mlflow_runs.py`; `docs/EXPERIMENTS.md` | A, B, C |

Each PR is small enough to review in one sitting except A, which is the tail of
an existing plan. Every PR ends with its docs; no handoff doc is kept at a
close.

---

## 6. PR A: finish the tuning study

Source of truth for the design: [HYPERPARAMETER_TUNING_PLAN.md](HYPERPARAMETER_TUNING_PLAN.md)
§4.4 (the code shape), §5 (the worked example), §6 (the remaining work, with
"done when" lists), §7 (working notes, commands, pitfalls). This section only
orders that work and adds what this plan needs from it. Update that plan's
status line as its sub-tasks close.

### [x] A0. Settle the branch

- The uncommitted edit in `inspection/bootstrap.py` added comments that restate
  the code; the user chose to revert it (done by copy from `HEAD`).
- Fix the stale status line of `MODEL_INSPECTION_PLAN.md` ("awaits the user's
  commit" → merged as `7d865ab`; sub-task 2 moves to this plan's PR C).
- Baseline: `uv run pytest -m "not slow"` (1216 passed on `7d865ab`).

**Done when:** `git status` is clean after the user's commit; the baseline
passes.

### [x] A1. `HyperparameterStudy` constructor and defaults (tuning plan 3.1)

File `hyperparameter_tuning/study.py`; tests
`tests/unit/test_hyperparameter_tuning_study.py`. Signature as §4.4 of the
tuning plan. Before writing it, probe Optuna's `TPESampler(seed=,
multivariate=True)` warning (experimental) and decide with the user whether to
silence it (the user's rule: silence a warning only if it does not affect
results).

**Done when:** default sampler is a seeded multivariate TPE and the default
pruner `NopPruner`; `n_trials=None` with `timeout_seconds=None`, `n_trials < 1`
and `n_jobs == 0` raise `ValueError`; a user-given sampler/pruner is used as
given (not cloned).

### [x] A2. `optimize` and the best trial (tuning plan 3.2)

**Done when:** the same seed gives an identical `best_params` twice; a pruned
trial never wins; ties go to the lowest trial number; no completed trial
raises `RuntimeError`; `initial_params` run first, in order, and a key no trial
suggested raises (measured in A1: Optuna drops it silently, without a warning);
more `initial_params` than `n_trials` raises; the Optuna study is kept as
`optuna_study_`. `optimize` returns a minimal `TuningResult` (`best_params`,
`best_value`, `best_trial_number`; user's choice), which A3 completes.

### [x] A3. `TrialRecord` and `TuningResult` (tuning plan 3.3)

**Done when:** each `TrialRecord` of a COMPLETE trial carries `fold_scores`,
`fold_sizes` (from the trial's user attrs set by the evaluator) and
`std_error = corrected_std_error(fold_scores)`, `None` otherwise; a PRUNED
trial's record does not report its last intermediate as a score (Optuna keeps
it as the trial's `value`; measured in A2);
`TuningResult.to_dict()` round-trips through `json.dumps`/`loads` (plain
Python numbers only: Optuna's sqlite storage rejects `np.int64`);
`is_reproducible` is `False` with a timeout or `n_jobs > 1`; a lower-is-better
metric's best value is reported in its natural sign wherever it is shown or
logged, the score kept as the search value (D12 of the tuning plan kept,
user's choice 2026-10-11; design decided in A3's plan mode).

### [x] A4. Integration test, pruner guidance, penalty ranges (tuning plan 3.4–3.5)

- The §5 example as a test marked `slow`, with `n_trials=3`, on the simulator
  at 20 neighborhoods.
- Pruner guidance in the docs: `WilcoxonPruner` or `SuccessiveHalvingPruner`,
  not `MedianPruner` (reason in the tuning plan).
- Re-check the penalty ranges flagged in
  [FEATURE_TRANSFORMATIONS.md §8.7](FEATURE_TRANSFORMATIONS.md) item 7
  (the held-out optimum sat at the edge of `total_l2_penalty=1.0` and
  `probability_c=0.01`): widen the ranges used in D4 to
  `alpha ∈ [1e-4, 10]` (log) and `C ∈ [1e-3, 100]` (log), and record it there.

**Done when:** the slow test passes; the ranges are written in this plan's D4
and the feature doc's §8.7.

### [x] A4b. External review and simplification

The review of `80fd068`, its evidence, qualifications and continuation prompt
are in [HYPERPARAMETER_TUNING_REVIEW.md](HYPERPARAMETER_TUNING_REVIEW.md).
It is review input, not an approved replacement for the decisions in §4 or
the tuning plan. No package changes have been chosen or applied.

- Verify branch, HEAD and worktree before assuming the A4b plan is committed
  or the tree is clean. Preserve unrelated user changes.
- Reproduce every finding A1–E2 against current code before accepting or
  rejecting it. Use read-only probes or isolated scratch mutations; verify
  the scratch import path and restore scratch files by copy and `cmp`.
  Do not mutate repository `src` during this review. Temporary scripts are
  optional: the review document includes durable reproduction recipes.
- Address the empty search space required by D4's one-trial NB2 run and the
  test gaps: hard-coding the sampler seed or returning the wrong `best_score`
  both survived 141 package tests. Reuse existing tests; do not add a test for
  a field removed by a chosen simplification.
- Set the report's simplifications beside these candidates, found before the
  review (estimates, not applied); the user chooses which to apply:
  - **S1. Drop `n_jobs`, `timeout_seconds`, `initial_params`.** No planned
    client uses them (E12: "the notebook uses neither"); the rule that
    dropped `storage` and `study_name` in A1 (add a field with its first
    use). About 27 lines of `study.py` (two checks, the misspelt-name
    callback, the WAITING-trial handling, `is_reproducible`) and ~8 tests.
    With sequential trials, Optuna's `study.best_trial` already gives a tie
    to the lowest number and skips pruned trials (measured), so the study's
    own `min` selection can go too, after re-verifying those guarantees.
    The review's C1 additionally considers sampler/pruner configuration;
    evaluate that extension separately. Its 42-line/16-case footprint is not
    S1's net saving. Preserve guarantees for objectives still supported.
  - **S2. One sign: keep `value`, `fold_values`, `best_value`; drop `score`,
    `fold_scores`, `best_score`.** Since D12's reversal the score equals the
    value for every repo metric (all losses); a greater-is-better score is
    `-value`. Changes the user's explicit "keep both" choice recorded with
    A3's minimize convention, so requires approval. Keep D12's evaluator
    negation and any retained natural-sign reporting. Removing a result field
    does not automatically remove its evaluator user attr or `_sign`.
  - **S3. Drop `metric_name` from each `TrialRecord`**, keeping it on
    `TuningResult`. Minor.
  - Compare the review's additional candidates with the existing preference
    to retain `Mean` and useful state-name/tuple conversions. E9 mentions
    `UpperBound`: establish planned use before removing policies or moving
    serialization. Couple mutable-choice aliasing and JSON safety with the
    choice of whether to retain arbitrary categorical objects. Wrapper/test
    removals need evidence of preserved behavior, not just smaller footprints.
  - Nit: `aggregation.py`'s module docstring has an over-long line.
- Before A5, adjudicate the statistical/JSON documentation findings: the
  corrected SE is not generally exact sampling uncertainty; mean plus each
  candidate's SE is not the classical one-SE simplicity rule; accepted params
  are not universally JSON-safe. The SE finding does not request a new formula.
- Rank actual net changes and tests affected; footprints overlap and do not
  justify deleting tests for retained guarantees. Ask the user to choose and
  stop before implementation.
- Implement the chosen changes with the §3 routine; update the tuning plan's
  §2, §4.4 and its test table to match.

**Done when:** every finding has a reproduced verdict, recorded in the
session's stop summary; the user has chosen the changes; those changes pass
the routine; the tuning plan's §4 and test table match the code. Stop before
A5 rather than writing the user guide around an unsettled API.

### [x] A5. Docs (tuning plan 4.1–4.3)

`docs/HYPERPARAMETER_TUNING.md` (user doc: the three classes, the worked
example run as written, aggregation choice, pruners, reproducibility,
pitfalls), links from `docs/README.md`, `MODULE_REFERENCE.md`, the root
`README.md`; roadmap step 2 ticked in `MODEL_REIMPLEMENTATION_PLAN.md` §5.

**Done when:** every code block in the new doc runs as written; the tuning
plan's status line says "complete".

### [x] A6. Merge PR #5 into `main`

The user marks PR #5 ready and squash-merges it (their convention). Later PRs
branch from `main`.

**Done when:** `git log --oneline -1 main` shows the squash commit; the
non-slow suite passes on `main`. *Done 2026-10-12:* `ba2aa11`; 1247 passed,
1 skipped, 1 xfailed on `main`.

---

## 7. PR B: the three-way split and the cohort count metric

Branch `feat/split-and-cohort-deviance` from `main`. A small PR; one plan-mode
stop per sub-task.

### [x] B1. `COHORT_POISSON_DEVIANCE` (E2)

- `scoring.py`: `cohort_poisson_deviance(y_true, y_pred) -> float`, the shape
  check as in `cohort_log_loss`, then `mean_poisson_deviance(counts.ravel(),
  predictions.ravel())`; the constant `COHORT_POISSON_DEVIANCE`; both in
  `__all__`.
- Tests in `tests/unit/test_scoring.py`: the ready-made row (name,
  function, a loss); equals the mean of the per-column deviances on a complete
  matrix; raises on a transposed or flattened prediction (measured: as many
  cells, so after `ravel` it is scored mispaired with no error: 2.809 instead
  of 0.386 on the test's fixtures; a (n, 1) prediction is not silent, scikit-learn raises on the
  lengths); DataFrames are paired by position. The "equals `POISSON_DEVIANCE`
  on one column" test was dropped: a special case of the mean.
- `EVALUATION_AND_METRICS.md` is the old stack's; add the metric to the new
  metric list in `MODULE_REFERENCE.md`'s `scoring` entry only.

**Done when:** the four tests pass under `-W error`; a mutation that drops the
shape check fails a test. *Done 2026-10-12:* both mutations (no shape check;
the prediction raveled column-major) fail their tests.

### [x] B2. The split pattern (E6)

- **Method: not fixed (user, 2026-10-13: "check on each setup").**
  `split_method` is a setting, set at the top of the notebook (D2) and logged
  (E9); PR D decides which method the comparison reports. Evidence for that
  choice is in `SPLITTING.md` §3 (sizes per method) and §5 (how to choose).
- Tests in `tests/unit/test_splitters.py` ("The three-way split"), on groups of
  sizes 1, 2, 3 and 10, 20 seeds each: the three parts are disjoint and cover
  every row (every method); `grouped` never puts a group in two parts;
  `stratified_by_group` gives each group size its exact counts (train / val /
  test): 1 → 1/0/0, 2 → 1/0/1, 3 → 1/1/1, 10 → 6/2/2, the same for every seed.
  A pair is a singleton after the holdout, so it has no validation row; a group
  of 3 has one in each part.
- No `Splitter.train_val_test_indices`: of the mistakes probed, two are silent
  and the doc states both: the missing position mapping (up to 4 of 16 rows
  overlap the test part) and, under `random` only, the whole table given to the
  second call (it re-splits every row; the group-aware methods raise on the
  length mismatch). One client (D2), which displays the three sizes.
- `SPLITTING.md` §3 "Train, validation and test": the block run as written for
  each method, and its sizes on the canonical table (`split_seed = 42`):
  `random` 917 / 306 / 306, `stratified_by_group` 915 / 307 / 307, `grouped`
  908 / 311 / 310 (30 test neighborhoods; `test_size` counts groups).

**Done when:** the tests pass for all three methods; the doc block runs as
written; the method choice and its reason are recorded here. *Done 2026-10-13:*
three mutations (no position mapping; `grouped` holding out with
`ShuffleSplit`; `_held_out_count` without its floor of one) each fail a test.

---

## 8. PR C: feature importance

Branch `feat/feature-importance` from `main`. Source of truth for the design:
[FEATURE_IMPORTANCE_PLAN.md](FEATURE_IMPORTANCE_PLAN.md): its "To start" list
(re-add `shap>=0.53,<1`, then the user runs `uv lock` and `uv sync`), decisions
I3–I11, code shapes §4, sub-tasks 1–3 with their test lists, risks §6. Move
its status line to "in progress" at C1 and to "complete" at C3.

### [ ] C1. `PermutationImportance` (importance plan sub-task 1)

`inspection/permutation.py`, pydantic dataclass like `BootstrapEvaluator`,
`compute(model, X, y, *, exposure=None) -> DataFrame` (one row per column of
`X`: `importance_mean`, `importance_std`; sign from `greater_is_better`, so
positive = helps). Built on `sklearn.inspection.permutation_importance` with a
scorer that passes `exposure` row-aligned. Tests as the importance plan lists.
Its pydantic config, and `BootstrapEvaluator`'s, should take `extra="forbid"`
(a misspelt keyword is ignored silently today), as the tuning package does
since A1 (`hyperparameter_tuning/_config.py`); either share that constant or
add an `inspection/_config.py`.

**Done when:** that plan's four tests pass under `-W error`; each mutation
fails a test; works on `TotalTimesProbabilityModel` with the string column
`school_status`.

### [ ] C2. `ShapExplainer` (importance plan sub-task 2)

`inspection/shap_values.py`, `explain(model, X, background) -> shap.Explanation`
with cohort output names; `shap.PermutationExplainer`, `max_evals=500`
default, codes for non-numeric columns (I9). Tests as the importance plan
lists. Add `inspection` mypy entries if missing.

**Done when:** additivity against `predict(exposure=1)` holds for `CountModel`;
the multi-output shape and names are right; the same seed gives identical
values.

### [ ] C3. Per-component importance, smoke run, docs (importance plan sub-task 3 + inspection plan sub-task 2)

- No new class for per-component importance: the fitted children are public
  (`total_model_`, `probability_model_`), so the notebook calls
  `PermutationImportance(metric=POISSON_DEVIANCE, ...).compute(model.total_model_,
  X, y.sum(axis=1), exposure=...)` and the same with `COHORT_LOG_LOSS` on
  `probability_model_`. The smoke run shows the three tables side by side.
- Smoke run script in the session scratchpad (recipe recorded in the doc):
  simulator at 150 × 9, the B2 split, `CountModel` (LightGBM) and
  `TotalTimesProbabilityModel` (Poisson × LR) at class defaults; permutation on
  train and on validation; SHAP on 200 validation rows with 50 training
  background rows; `BootstrapEvaluator` on validation grouped by neighborhood.
  Compare the rankings of permutation and mean |SHAP|.
- `docs/MODEL_INSPECTION.md`: bootstrap (definition, groups, the zero-width
  interval with one group: decide with the user whether `len(unique(groups)) < 2`
  raises, the inspection plan's open item), permutation (definition, held-out
  rows, correlated columns, the per-component recipe), SHAP (the rate, codes,
  cost, subsampling, mean |SHAP|), one plot recipe each (matplotlib bar with
  error bars; `shap.plots.bar` and `shap.plots.beeswarm` per cohort). Every
  block run as written.
- `MODULE_REFERENCE.md`, `docs/README.md`, root `README.md` entries; the
  inspection plan's status line "complete"; roadmap note.

**Done when:** the doc's blocks run as written; the smoke tables are in the
doc; the non-slow suite passes; the importance plan is marked complete.

---

## 9. PR D: the notebook, the data and MLflow

Branch `feat/experiment-notebook` from `main`. The notebook is a marimo file,
`notebooks/03_experiments.py`, run as in §2 ("Notebooks"). Each run cell is
guarded by a `mo.ui.run_button()` like `02_model_fitting.py`, so opening the
notebook fits nothing. Docs in `docs/EXPERIMENTS.md`.

### [ ] D1. The feature declarations as code

- New module `src/age_group_prediction/feature_engineering/declarations.py`
  (decide the name with the user; alternative: `modeling/features.py`)
  exporting `ROOM_SHARES`, `SCHOOL`, `tree`, `total_base`,
  `cohort_probability_base`, and the §8.5 candidates as functions returning a
  fresh `FeatureTransformer` each (`total_ses_quadratic()`,
  `total_daycare_log1p()`, ...), copied verbatim from
  [FEATURE_TRANSFORMATIONS.md §8](FEATURE_TRANSFORMATIONS.md). Functions, not
  module constants, for the variants with state: a `FeatureTransformer` is
  cloned by the models anyway, so constants would also be safe; choose
  constants for the three bases and functions for the variants, and say why in
  the module docstring. Add the module to the mypy `files` list. Expect one
  typing gap (found in A4): the doc writes `ColumnPlan(columns="school_status")`,
  which runs (a "before" validator wraps a string) but `ColumnPlan.columns` is
  typed `tuple[str, ...]`, so mypy rejects it. Either write tuples or widen the
  annotation (`str | tuple[str, ...]` with the validator); decide with the user.
- Tests: each declaration fits and transforms the simulator table at 20
  neighborhoods without error and yields finite columns; the doc blocks of §8
  now `import` from the module instead of redefining (edit the doc).

**Done when:** `FEATURE_TRANSFORMATIONS.md` §8 imports the declarations and its
blocks run as written; the tests pass.

### [ ] D2. The data cell and the split cell

- Data cell: `load_simulation_config`, the 150 × 9 override as in
  `02_model_fitting.py`, `StudentPopulationSimulator(config).run()`;
  `exposure = ExposureTransformer("n_apartments").fit_transform(table)`;
  `groups = table["neighborhood_id"]`; `Y = table[cohort_columns]`;
  `data_hash`, `config_params = flatten(config.model_dump())` (a 5-line helper
  in the notebook; move to `mlflow_runs.py` in D4 if a second user appears).
- Split cell: the E6 block with `split_method` (B2); the four seeds (E12) in one
  cell at the top, displayed; `n_train` / `n_val` / `n_test` displayed.
- MLflow cell: `mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])`
  from `.env`; refuse to run when `MLFLOW_TRACKING_URI` is unset or points at
  the repo root's `mlflow.db` (one `if` with a clear message: the canonical
  store must never be written).

**Done when:** the notebook runs as a script (`uv run --env-file .env
notebooks/03_experiments.py`) without fitting anything and prints the sizes
and the hash.

### [ ] D3. The first run: Model 1 × LightGBM × `tree` × exposure on

One cell, inline MLflow calls (E10):

1. `with mlflow.start_run(run_name="model1_lgbm_tree_exposure")`: log the E9
   params.
2. For each cohort, `with mlflow.start_run(nested=True, run_name=cohort)`:
   `CVHyperparameterEvaluator(CountModel(estimator=LGBMRegressor(objective="poisson",
   subsample_freq=1, n_jobs=1, verbosity=-1, random_state=estimator_seed),
   use_exposure=True, feature_transformer=tree), parameters, cv=cv,
   metric=POISSON_DEVIANCE)`, parameters as the tuning plan's §5 list;
   `HyperparameterStudy(seed=study_seed + i, n_trials=30).optimize(lambda
   trial: evaluator.evaluate(trial, X_train, Y_train[cohort], groups_train,
   exposure=exposure_train))`; log `cv/best_value`, `cv/best_std_error`,
   `trials.csv`, `best_params.json`.
3. Assemble `IndependentCohortModels(...)`, fit on the training part, predict
   the validation part, log the E3 metrics with `BootstrapEvaluator`
   (`n_resamples=1000`, `groups=groups_val`), `model.joblib`.
4. Importance: `PermutationImportance(metric=COHORT_POISSON_DEVIANCE,
   n_repeats=10, random_state=estimator_seed)` on validation and on train;
   `ShapExplainer` on the E8 subsample; log the CSVs and the plots.

**Done when:** the run appears in the scratch MLflow UI with every E9 item;
the cell's wall time is recorded in the doc (budget: 3 cohorts × 30 trials × 5
folds LightGBM fits ≈ a few minutes).

### [ ] D4. The remaining runs and the helper module

- Write the second cell (Model 2: Poisson total × LR probability, `total_base`
  / `cohort_probability_base`, exposure on), then extract what the two cells
  repeat into `src/age_group_prediction/mlflow_runs.py` (E10): expected
  `log_tuning_result(result: TuningResult) -> None` (params, metrics, the two
  artifacts), `log_validation_metrics(model, X, y, groups, *, exposure,
  metrics: Sequence[Metric], n_resamples, random_state) -> dict[str, float]`,
  `log_importance(model, X_train, y_train, X_val, y_val, *, exposure_train,
  exposure_val, metric, ...) -> None`. Justify each; drop any that one cell
  alone uses. Tests with a temporary `mlflow` file store in `tmp_path`
  (`pytest.importorskip("mlflow")`, as the old tracking tests do).
- Then one cell per remaining setting, in this order (each a parent run with
  nested studies):
  1. Model 1 × LightGBM × `tree` × exposure **off** (E5).
  2. Model 1 × `PoissonRegressor(alpha)` × `total_base` (+ `n_apartments`
     dropped, exposure on): `alpha ∈ [1e-4, 10]` log (checked in A4).
  3. Model 1 × NB2 (`NegativeBinomialRegressor`): no hyperparameters; the
     "study" is one trial, logged the same way.
  4. Model 2 × {Poisson, NB2, LightGBM} total × {LR (`C ∈ [1e-3, 100]` log, checked in A4;
     `calibration_method ∈ {None, "temperature", "sigmoid"}`), `LGBMClassifier`
     (the §5 LightGBM ranges)} probability: six cells, or one cell with a
     dropdown for the pair (prefer six explicit cells: "each run is a cell").
  5. The §8.5 feature candidates on the best Model 2 pair, one cell each
     (`total__ses_quadratic`, `total__daycare_log1p`,
     `total__room_share_x_household_size`, `composition__daycare_log1p`,
     `composition__room_share_x_median_age`, combined).
- Search spaces are written once in a cell near the top (`SEARCH_SPACES: dict[str,
  list[Parameter]]`) and logged as the `search_space` param.

**Done when:** every cell runs on the scratch store; the helper's tests pass;
no two cells repeat a block of more than ~5 lines of logging.

### [ ] D5. The comparison table and the final run (E11)

- Comparison cell: `mlflow.search_runs(experiment_names=[...],
  filter_string="tags.mlflow.parentRunId = ''")` → a table of
  `val/cohort_poisson_deviance` (mean, lower, upper) and `val/cohort_log_loss`
  per setting, sorted by the primary metric; a one-line statement of the
  winner. No automatic selection in code beyond the sort: the user reads the
  table (the intervals overlap for close settings).
- Final cell, behind its own button and a confirmation checkbox like
  `02_model_fitting.py`: rebuild the winner from its `best_params.json`
  artifact(s), fit on `train_index ∪ val_index`, score on `test_index` with
  the E3 bootstrap set, log as a run tagged `final=true` with the winner's
  parent run id as a param. Refuse to run twice in the same experiment (search
  for an existing `final=true` run and stop with a message), because the test
  part stands for production and is evaluated once.

**Done when:** the table shows every setting; the final run exists once and
holds `test/*` metrics only there.

### [ ] D6. Docs and close

- `docs/EXPERIMENTS.md`: how to run (env, commands, time budget), the three-way
  split and seeds, what a run logs (the E9 table), how to add a setting (a cell
  template), how the final run is guarded, the Bayesian model's future place
  (E3b).
- `docs/README.md` (user guides + this plan under Active Plans → mark
  complete), `MODULE_REFERENCE.md` (`mlflow_runs.py`, `declarations.py`), root
  `README.md` (run the new notebook), roadmap in
  `MODEL_REIMPLEMENTATION_PLAN.md` §5 (a line for the notebook; step 5 can now
  also delete `02_model_fitting.py`'s dependency once the old stack goes).
- This plan's status line → "complete".

**Done when:** the doc's commands run as written; links resolve.

---

## 10. Out of scope and later

- **Model C (Bayesian).** Its PR adds it as a `BaseAgeGroupModel` with a mean
  `predict`, plus the sample-based metrics of E3b; it then enters the notebook
  as settings (a setting = one prior/profile configuration; the old stack chose
  fixed candidates over a search because NUTS per fold is slow). Nothing in
  A–D blocks it; see E3b's rule.
- **Roadmap step 5** (delete the old stack, rewire or delete `experiment/`,
  `tracking/`, `02_model_fitting.py`). The new notebook never imports the old
  stack, so the deletion can follow D independently.
- **Out-of-fold predictions** from the evaluator, column-block permutation
  importance, Optuna visualizations as artifacts: only if a run shows the need.

---

## 11. Commands

| purpose | command |
|---|---|
| baseline / suite | `uv run pytest -m "not slow"` |
| one package's tests | `uv run pytest tests/unit/test_<package>_*.py -q -W error` |
| types | `uv run mypy`, then `uv run mypy src/age_group_prediction/<package> <changed tests>` |
| lint | `uv run ruff check <files>` and `uv run ruff format <files>` |
| probes (read-only) | `PYTHONPATH=src .venv/bin/python -c "..."` |
| the notebook | `uv run --env-file .env --group notebook --group tracking marimo edit notebooks/03_experiments.py` |
| the MLflow UI (scratch) | `uv run --env-file .env --group tracking mlflow ui --backend-store-uri sqlite:///.mlflow-local/mlflow.db --port 5000` |
| after a `pyproject.toml` edit | `uv run --frozen ...`; the user runs `uv lock` and `uv sync` |
