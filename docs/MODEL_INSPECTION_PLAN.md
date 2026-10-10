# Plan: model inspection: bootstrap CI, permutation importance, SHAP

**Written for:** the implementing model and the user who validates each
sub-task. Self-contained: it assumes no memory of the planning conversation.
This file is the source of truth: update its status line and the sub-task
checkboxes in §8 as work finishes.

**Status (2026-10-10):** approved by the user. Sub-task 0 is committed and
pushed as `0785aca`: this doc, `shap>=0.53,<1` and `uv.lock`. The branch is
`feat/model-inspection`, and draft
[PR #14](https://github.com/galkampel/age-group-prediction/pull/14) targets
`feat/hyperparameter-tuning`. No code exists yet. **Next:** sub-task 1
(`BootstrapEvaluator`). After this plan closes, tuning resumes at
[HYPERPARAMETER_TUNING_PLAN.md](HYPERPARAMETER_TUNING_PLAN.md) 3.1.

**To resume (new session):**
1. Run `uv sync`. shap is in `uv.lock` but was not yet installed in `.venv`
   (`import shap` raised `ModuleNotFoundError` at the handoff).
2. Run the baseline, `uv run pytest -m "not slow"`. Expect 1200 passed,
   1 skipped, 1 xfailed; if it differs, stop and report (§3 rule 2).
3. Read §3 (how to work), §5 (decisions; do not re-ask them), §6 (evidence),
   §7 (code shape) and §8 sub-task 1.
4. Read the patterns to match: `hyperparameter_tuning/evaluator.py` (a
   pydantic dataclass with `InstanceOf[Metric]`, settings vs method data),
   `modeling/base.py`, `scoring.py`, `utils.py` (`take_rows`, aliases), and a
   test such as `tests/unit/test_hyperparameter_tuning_evaluator.py`.
5. Start sub-task 1 in plan mode: re-verify the facts, ask the open choices,
   and summarize the step in chat before asking for approval.

## Contents

1. Context and goal
2. Branch and PR
3. How to work
4. What exists today
5. Decisions
6. Measured evidence
7. Target code shape
8. Sub-tasks
9. Docs: what the model-inspection doc must say
10. Risks
11. Verification

## 1. Context and goal

Two analysis tools for a **fitted** `BaseAgeGroupModel`. In practice that is the
tuned model refit on the full training data,
`evaluator.build_model(best_params).fit(X_train, y_train, exposure=...)`:

1. **Bootstrap:** the mean and the (1 − α) percentile interval of a
   `scoring.Metric` on an evaluation set.
2. **Feature importance**, in two variants: permutation importance and SHAP.

Out of scope:
- refitting inside the bootstrap (model uncertainty);
- importance of engineered features (this plan works on the raw columns only);
- plots (shap's own plotting works on the returned `Explanation`).

## 2. Branch and PR

The branch `feat/model-inspection` is cut from `feat/hyperparameter-tuning` at
`3c4c028`. Its draft PR is
[#14](https://github.com/galkampel/age-group-prediction/pull/14), which targets
`feat/hyperparameter-tuning`. The user commits and pushes; commit messages carry
no `Co-Authored-By` line.

## 3. How to work

The user's standing rules, as
[TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md](TOTAL_TIMES_PROBABILITY_MODEL_PLAN.md)
§3 states them:

1. **One sub-task per stop.** Start each in plan mode: re-verify this plan's
   facts for it, write a short summary of the sub-task in the chat, and ask for
   approval. Implement only after it.
2. **Baseline first:** run `uv run pytest -m "not slow"` before the first edit.
   If it fails, stop and report.
3. **If an assumption breaks, stop** and show the options with evidence.
4. **Routine:**
   1. implement;
   2. `uv run ruff check` and `uv run ruff format` on the changed files;
   3. `uv run mypy`, then `uv run mypy src/age_group_prediction/inspection <changed tests>`;
   4. the changed tests with `-W error`;
   5. one mutation check per claimed behavior: edit `src` in place, then
      restore by copy with md5 (never `git checkout` or `stash`);
   6. an independent review subagent, which takes 6–8 minutes (say so, and
      update the plan doc and run the suite meanwhile); reproduce each finding
      before fixing it;
   7. the non-slow suite;
   8. update the plan doc, then stop.
5. **At each stop:** a file-by-file summary of the `.py` changes, the check
   results, and suggested commit commands.
6. **Justify every class, field and check**, or drop it. Validate only what
   would otherwise pass silently, and leave to the library what it already
   raises. Add no default the user did not ask for.
7. **Code style:** match `hyperparameter_tuning/evaluator.py` and
   `modeling/count_model.py`.
   - A docstring says what the code does, its inputs and outputs, and the
     non-obvious constraint, in a few lines.
   - A comment states only a "why" the code cannot show.
   - Derivations, measured figures and library quirks go in the docs (§9).
8. **Probes** are read-only: `PYTHONPATH=src .venv/bin/python -c "..."`
   (`uv run` may re-sync the venv). Quote globs in zsh.
9. **Tests:** each test's name and comment state the mistake it catches. No
   test may only check a library.

## 4. What exists today

- `modeling/base.py`:
  - `BaseAgeGroupModel.predict(X, exposure=None)` returns an array for one
    target, or a DataFrame with `y`'s columns.
  - `evaluate(y_true, y_pred, metric)` is `metric.function(y_true, y_pred)`.
  - `_check_exposure` returns `None` when the model expects no exposure, so
    passing one to such a model is harmless.
- `scoring.py`: `Metric(name, function, greater_is_better)`, and
  `POISSON_DEVIANCE`, `MAE`, `RMSE`, `COHORT_LOG_LOSS`.
- `utils.py`: `take_rows`, and the aliases `DesignMatrix`, `Target`, `Groups`,
  `Exposure`.
- Groups are neighborhoods. One row is one building. `Splitter` takes
  `groups=None` for a plain random split.
- The old stack has the same tools, bound to its own types and deleted with it
  (roadmap §5 step 5), so they are not reused:
  - `evaluation.neighborhood_cluster_bootstrap` with
    `resampling.NeighborhoodClusterResampler`, which uses fixed predictions,
    whole neighborhoods and a percentile CI;
  - `experiment/importance.py` (per-block permutation).

  This plan keeps their method.

## 5. Decisions

| # | Decision | Why |
|---|---|---|
| I1 | The bootstrap takes a **fitted** model and an evaluation set. It predicts **once**, resamples the evaluation rows `n_resamples` times, and recomputes the metric each time. It returns `mean`, `lower` and `upper`, the `alpha/2` and `1 − alpha/2` percentiles of the replicates. There is no refit. | The user's spec. The interval is conditional on the fitted model, as the old stack's was. |
| I2 | The resampling unit is whole groups when `groups` is given, and rows otherwise. | Buildings in one neighborhood are correlated. This mirrors `Splitter`'s optional groups. |
| I3 | Importance runs on whatever `X, y` the caller passes. | The user's choice. The doc recommends held-out data for permutation importance. |
| I4 | Importance is computed per **raw input column** of `X`. | It covers every `BaseAgeGroupModel`, composites included, and one-hot or interaction expansions map back to their source column. |
| I5 | SHAP explains the **rate per unit exposure**, `predict(X, exposure=1)`. | SHAP predicts on synthetic masked rows, so a per-row exposure cannot follow them. For an exposure model, `predict = rate × exposure`, and the features drive the rate. Models without exposure ignore the argument (§4). |
| I6 | Permutation importance runs per raw column through `sklearn.inspection.permutation_importance`, with a scorer built from our `Metric`. | Probed (§6): it works with a cohort-table `y`, string columns and a closed-over exposure. There are no column blocks. |
| I7 | The code goes in a new package `age_group_prediction/inspection/`, as `bootstrap.py`, `permutation.py` and `shap_values.py`. | "inspection" is scikit-learn's term, and the package is independent of the old `evaluation.py`/`resampling.py`. |
| I8 | `shap` is a main dependency (`shap>=0.53,<1`). | It is a core step after tuning, and the tests need no extra group. |
| I9 | SHAP sees non-numeric columns as integer codes. The prediction function decodes them back before `predict`. | Probed (§6): shap's `Independent` masker fails on string columns with `TypeError: unsupported operand type(s) for -: 'str' and 'str'`. The codes are the categories observed in `background` and `X`. |
| I10 | `ShapExplainer` uses `shap.PermutationExplainer`, and `background` is a required argument of `explain`. `max_evals` defaults to shap's own default, 500. | The explainer is model-agnostic and exactly additive (§6). There is no sensible default background: the caller chooses its rows and size, which set the cost (§6). |
| I11 | `explain` returns the `shap.Explanation` with the model's output names set (`y`'s cohort columns). Global importance is `mean(|values|)` per feature, shown in the doc. | shap drops the output names and labels them `Output 0..K-1` (§6). The `Explanation` keeps shap's plots. |

## 6. Measured evidence

Measured 2026-10-10 with shap 0.53.0, sklearn 1.9.0 and pandas 3. The probe
script is in that session's scratchpad (`probes/shap_probe.py`). Its setup:
- 400 rows, with columns `x`, `z` (float) and `kind` (str: a/b/c);
- an exposure;
- `CountModel(PoissonRegressor, use_exposure=True)` and
  `TotalTimesProbabilityModel` (with `CohortProbabilityModel(LogisticRegression)`),
  each with a `FeatureTransformer` (Standardize, then OneHot on `kind`);
- 100 background rows and 50 explained rows.

| Probe | Result |
|---|---|
| sklearn `permutation_importance`, DataFrame `y` (2 columns), a string column, exposure closed over in the scorer | Works. The importances are `[921.6, 0, 0]` for the used column, the constant column and the string column. |
| `shap.maskers.Independent` on a DataFrame with a str column | `TypeError: unsupported operand type(s) for -: 'str' and 'str'`. |
| The same with the str column encoded as integer codes, decoded inside `f` | Works. `f` receives a DataFrame. |
| Additivity, `base + Σ values` against `predict(exposure=1)` | Max error 6.7e-16 (`CountModel`) and 1.1e-15 (`TotalTimesProbabilityModel`). |
| Multi-output shapes | `values` is (50, 3 features, 3 cohorts) and `base_values` is (50, 3). The output names come back as `Output 0..2`, even with `output_names=` passed. |
| Same `seed`, run twice | Identical values. |
| `shap.Explainer(f, masker)` default algorithm | `ExactExplainer`, which is exponential in the number of features. Hence I10 calls `PermutationExplainer` directly. |
| Cost (`max_evals=500`, 100 background rows) | `CountModel`: 15.7 s per 50 rows. `TotalTimesProbabilityModel`: 32.6 s per 50 rows, and 222 s per 500 rows. Each `predict` call re-runs the feature transformer, so the cost grows with the background size × `max_evals`. |

## 7. Target code shape

Settings go in the constructor, as a pydantic dataclass with
`InstanceOf[Metric]` like `CVHyperparameterEvaluator`. Data are method
arguments.

```python
# inspection/bootstrap.py                                   (sub-task 1)
@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    lower: float
    upper: float

class BootstrapEvaluator:
    metric: InstanceOf[Metric]
    n_resamples: int                  # >= 1, required
    alpha: float = 0.05               # in (0, 1)
    random_state: int | None = None

    def evaluate(self, model, X, y, groups=None, *, exposure=None) -> BootstrapResult:
        # predict once; per replicate draw rows (or all rows of drawn groups) with
        # replacement; score with model.evaluate on take_rows(y), take_rows(y_pred);
        # percentiles via np.quantile(scores, [alpha/2, 1 - alpha/2])
```

```python
# inspection/permutation.py                                 (sub-task 2)
class PermutationImportance:
    metric: InstanceOf[Metric]
    n_repeats: int
    random_state: int | None = None

    def compute(self, model, X, y, *, exposure=None) -> pd.DataFrame:
        # scorer = ±metric(y, est.predict(X, exposure=exposure)), the sign from
        # greater_is_better, so a positive importance means the column helps;
        # one row per column of X: importance_mean, importance_std
```

```python
# inspection/shap_values.py                                 (sub-task 3)
class ShapExplainer:
    max_evals: int = 500
    random_state: int | None = None

    def explain(self, model, X, background) -> shap.Explanation:
        # encode non-numeric columns as codes (I9); f decodes and returns
        # model.predict(frame, exposure=np.ones(len(frame))) as an array (I5);
        # shap.PermutationExplainer(f, Independent(background), seed=random_state);
        # set output names (I11)
```

## 8. Sub-tasks

Each sub-task ends at a stop (§3). **Verify** lists what the user can check.

### [x] 0. Plan doc, branch, dependency, probes

- Write this doc.
- Create the branch `feat/model-inspection`.
- Add `shap>=0.53,<1` to `[project] dependencies`.
- Run the probes in §6.

**Verify:** the doc renders; the baseline passes; the probe results are in §6.

**Record (2026-10-10):**
- Baseline: 1200 passed, 1 skipped, 1 xfailed.
- The probes changed the design from the approved plan: the str-column
  failure gives I9, the lost output names give I11, and `background` became
  required with shap's `max_evals` default (I10).
- The mypy entries moved to sub-task 1: mypy fails on the missing package
  path.
- Committed and pushed as `0785aca` (with `uv.lock`). Draft PR #14 was
  opened against `main` and retargeted to `feat/hyperparameter-tuning` at the
  handoff.

### [ ] 1. `BootstrapEvaluator`

- Write `inspection/__init__.py`, `inspection/bootstrap.py` and
  `tests/unit/test_inspection_bootstrap.py`.
- Add `src/age_group_prediction/inspection` to the mypy `files` and the
  `disallow_untyped_defs` override. This waits for this sub-task because mypy
  fails on a missing path (`Cannot read file`), as checked in sub-task 0.
- Tests:
  - With groups, each replicate holds whole groups (a mutation to row
    resampling fails).
  - The same seed gives identical results.
  - The bounds equal `np.quantile` of the replicates, and `lower ≤ mean ≤ upper`.
  - `predict` is called once (a spy).
  - A `greater_is_better` metric is reported raw, not negated.

**Verify:** the tests pass under `-W error`; each mutation fails a test.

### [ ] 2. `PermutationImportance`

- Write `inspection/permutation.py` and its tests.
- Tests:
  - A column the model ignores scores 0, and a used one scores > 0.
  - A positive score means "important" under both metric directions.
  - Exposure stays row-aligned while columns are permuted.
  - It works on `TotalTimesProbabilityModel` with a string column.

**Verify:** the same.

### [ ] 3. `ShapExplainer`

- Write `inspection/shap_values.py` and its tests.
- Tests:
  - Additivity against `predict(exposure=1)` for `CountModel` with exposure.
  - The multi-output shape and cohort output names.
  - A string column round-trips through the codes.
  - The same seed gives identical values.

**Verify:** the same.

### [ ] 4. Smoke run, docs and close

- Run on the simulator data:
  - fit `CountModel` and `TotalTimesProbabilityModel` on train;
  - bootstrap the test metric, with and without neighborhood groups;
  - compute both importances, then compare the rankings of permutation and
    mean |SHAP|.
- Write `docs/MODEL_INSPECTION.md` (§9), and update `MODULE_REFERENCE.md`, the
  README, the roadmap in `MODEL_REIMPLEMENTATION_PLAN.md`, and the status line of
  `HYPERPARAMETER_TUNING_PLAN.md`.

**Verify:** every code block in the doc runs as written; the non-slow suite
passes.

## 9. Docs: what the model-inspection doc must say

- **Bootstrap:**
  - the percentile interval;
  - why the cluster bootstrap (whole neighborhoods);
  - that the interval is conditional on the fitted model (no refit);
  - that few groups give wide, unstable intervals;
  - that `COHORT_LOG_LOSS` is a ratio of sums recomputed per replicate.
- **Permutation:**
  - the definition (score drop when one column is shuffled, averaged over
    repeats);
  - why to use held-out data;
  - that correlated columns share or hide importance.
- **SHAP:**
  - Shapley values of the rate (I5) and their additivity;
  - the code encoding (I9);
  - the cost (§6) and how to subsample `background` and `X`;
  - mean |SHAP| as the global importance.

## 10. Risks

- **SHAP cost.** It is minutes for a few hundred rows of a composite model
  (§6). The doc gives the subsampling recipe, and the smoke run measures the
  cost at simulator scale.
- **A category in `X` but not in `background`.** The codes come from both
  frames (I9), so it decodes correctly.
- **Few neighborhoods** give a degenerate cluster bootstrap. This is
  documented, not checked.

## 11. Verification

- `uv run pytest -m "not slow"` before and after each sub-task.
- `uv run mypy`, plus `uv run mypy src/age_group_prediction/inspection <tests>`.
- ruff, the new tests under `-W error`, the mutation checks, and the smoke run
  in sub-task 4.
