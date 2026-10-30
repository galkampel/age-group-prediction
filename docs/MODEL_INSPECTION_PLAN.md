# Plan: model inspection: bootstrap CI

> **2026-10-29:** the `hyperparameter_tuning` package this document names was removed with its docs ([NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) D17, T4a); T4b rebuilds it from the experiment notebook's section 5.

**Written for:** the implementing model and the user who validates each
sub-task. Self-contained: it assumes no memory of the planning conversation.
This file is the source of truth: update its status line and the sub-task
checkboxes in §8 as work finishes.

**Status (2026-10-13): complete.** Sub-task 1 (`BootstrapEvaluator`) was
merged as [PR #14](https://github.com/galkampel/age-group-prediction/pull/14)
(squash `7d865ab`). Sub-task 2 (the smoke run and the user guide
[MODEL_INSPECTION.md](MODEL_INSPECTION.md)) was done as sub-task C3 of
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) (PR C), together
with feature importance, whose decisions, evidence and records live in
[FEATURE_IMPORTANCE_PLAN.md](FEATURE_IMPORTANCE_PLAN.md). The open item of §10
is closed: fewer than two groups raise. Nothing to resume; §5 stays the
reference for the bootstrap's decisions.

## Contents

1. Context and goal
2. Branch and PR
3. How to work
4. What exists today
5. Decisions
6. Measured evidence
7. Code shape
8. Sub-tasks
9. Docs: what the model-inspection doc must say
10. Risks
11. Verification

## 1. Context and goal

The bootstrap of a **fitted** `BaseAgeGroupModel`: the mean and the (1 − α)
percentile interval of a `scoring.Metric` on an evaluation set. In practice the
model is the tuned model refit on the full training data,
`evaluator.build_model(best_params).fit(X_train, y_train, exposure=...)`.

Out of scope:
- refitting inside the bootstrap (model uncertainty);
- feature importance, deferred to
  [FEATURE_IMPORTANCE_PLAN.md](FEATURE_IMPORTANCE_PLAN.md);
- plots.

## 2. Branch and PR

The branch `feat/model-inspection` was cut from `feat/hyperparameter-tuning` at
`3c4c028`; its PR [#14](https://github.com/galkampel/age-group-prediction/pull/14)
was squash-merged into `feat/hyperparameter-tuning` as `7d865ab`. The user
commits and pushes; commit messages carry no `Co-Authored-By` line.

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

- `inspection/bootstrap.py`: `BootstrapEvaluator` and `BootstrapResult`
  (sub-task 1).
- `modeling/base.py`:
  - `BaseAgeGroupModel.predict(X, exposure=None)` returns an array for one
    target, or a DataFrame with `y`'s columns.
  - `evaluate(y_true, y_pred, metric)` is `metric.function(y_true, y_pred)`.
- `scoring.py`: `Metric(name, function, greater_is_better)`, and
  `POISSON_DEVIANCE`, `MAE`, `RMSE`, `COHORT_LOG_LOSS`.
- `utils.py`: `take_rows`, and the aliases `DesignMatrix`, `Target`, `Groups`,
  `Exposure`.
- Groups are neighborhoods. One row is one building. `Splitter` takes
  `groups=None` for a plain random split.
- The old stack has the same tool, bound to its own types and deleted with it
  (roadmap §5 step 5), so it is not reused:
  `evaluation.neighborhood_cluster_bootstrap` with
  `resampling.NeighborhoodClusterResampler`, which uses fixed predictions,
  whole neighborhoods and a percentile CI. This plan keeps its method.

## 5. Decisions

I3–I6 and I8–I11 (importance) moved to
[FEATURE_IMPORTANCE_PLAN.md](FEATURE_IMPORTANCE_PLAN.md) §2, with their numbers.

| # | Decision | Why |
|---|---|---|
| I1 | The bootstrap takes a **fitted** model and an evaluation set. It predicts **once**, resamples the evaluation rows `n_resamples` times, and recomputes the metric each time. It returns `mean`, `lower` and `upper`: the mean of the replicates, and their `alpha/2` and `1 − alpha/2` percentiles. There is no refit. | The user's spec. The interval is conditional on the fitted model, as the old stack's was. |
| I2 | The resampling unit is whole groups when `groups` is given, and rows otherwise. | Buildings in one neighborhood are correlated. This mirrors `Splitter`'s optional groups. |
| I7 | The code goes in a new package `age_group_prediction/inspection/`, as `bootstrap.py`. | "inspection" is scikit-learn's term, and the package is independent of the old `evaluation.py`/`resampling.py`. The importance modules join it later. |
| I12 | A non-finite replicate score raises `ValueError`, naming how many. | The user's choice (2026-10-11). Otherwise `np.quantile` returns nan bounds silently. Example: `COHORT_LOG_LOSS` is 0/0 on a replicate with no children. |
| I13 | No shap dependency in this PR. | The user's choice (2026-10-11): it is unused until the importance PR, which re-adds it. |

## 6. Measured evidence

- Sub-task 1, mutation checks: 13 mutations of `bootstrap.py`, each failed by
  a test (§8, sub-task 1 record).
- The shap and permutation probes of 2026-10-10 are in
  [FEATURE_IMPORTANCE_PLAN.md](FEATURE_IMPORTANCE_PLAN.md) §3.

## 7. Code shape

Settings go in the constructor, as a pydantic dataclass with
`InstanceOf[Metric]` like `CVHyperparameterEvaluator`. Data are method
arguments.

```python
# inspection/bootstrap.py
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
        # check_consistent_length(X, y, groups); predict once; per replicate
        # draw rows (or as many groups as there are, with all their rows) with
        # replacement; score with model.evaluate on take_rows(y), take_rows(y_pred);
        # raise on a non-finite score (I12);
        # percentiles via np.quantile(scores, [alpha/2, 1 - alpha/2])
```

## 8. Sub-tasks

Each sub-task ends at a stop (§3). **Verify** lists what the user can check.

### [x] 0. Plan doc, branch, dependency, probes

- Write this doc.
- Create the branch `feat/model-inspection`.
- Add `shap>=0.53,<1` to `[project] dependencies`.
- Run the probes (now in FEATURE_IMPORTANCE_PLAN.md §3).

**Record (2026-10-10):**
- Baseline: 1200 passed, 1 skipped, 1 xfailed.
- The probes changed the importance design (I9–I11).
- The mypy entries moved to sub-task 1: mypy fails on the missing package
  path.
- Committed and pushed as `0785aca` (with `uv.lock`). Draft PR #14 was
  opened against `main` and retargeted to `feat/hyperparameter-tuning`.

### [x] 1. `BootstrapEvaluator`

- Write `inspection/__init__.py`, `inspection/bootstrap.py` and
  `tests/unit/test_inspection_bootstrap.py`.
- Add `src/age_group_prediction/inspection` to the mypy `files` and the
  `disallow_untyped_defs` override.
- Remove shap (I13), and run `uv lock`.
- Move the importance material to FEATURE_IMPORTANCE_PLAN.md.

**Verify:** the tests pass under `-W error`; each mutation fails a test.

**Record (2026-10-11):**
- Baseline: 1200 passed, 1 skipped, 1 xfailed.
- `uv.lock` after `uv lock` is identical to `3c4c028`'s: shap and slicer
  removed.
- 16 tests, passing under `-W error`. They cover:
  - whole groups drawn, as many as there are;
  - rows drawn with replacement;
  - `y` and the predictions paired by position;
  - seed reproducibility;
  - the bounds equal to `np.quantile` of the replicates;
  - `predict` called once, with the exposure;
  - a raw `greater_is_better` score;
  - the non-finite raise;
  - the length checks;
  - the settings validation;
  - `CohortProbabilityModel` with `COHORT_LOG_LOSS` on a shuffled index.
- Mutations, each failed by a test:
  - row resampling with groups;
  - a fixed seed;
  - quantiles at α instead of α/2;
  - predict per replicate;
  - a negated score;
  - no finite check;
  - no length check;
  - unpaired predictions;
  - no replacement;
  - label-based row selection;
  - and, from the review, contiguous groups assumed (no `argsort`), a raise
    only when every replicate is non-finite, and label indexing on both `y`
    and the predictions.
- Review (independent subagent): no correctness bug. It probed position
  pairing for every pandas/numpy mix, NaN and categorical groups, and the
  exposure. It found three test gaps, each reproduced as a surviving mutant,
  then fixed:
  - the groups are now interleaved and unsorted;
  - one NaN among finite replicate scores now raises;
  - the cohort frames have a non-range index.
  Its fourth point is left as the user's choice (§10): a single group gives a
  zero-width interval silently.

### [x] 2. Smoke run, docs and close

Done as [EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) C3, with
the importance smoke run; the list below is its bootstrap part.

- Run on the simulator data:
  - fit `CountModel` and `TotalTimesProbabilityModel` on train;
  - bootstrap the test metric, with and without neighborhood groups.
- Write `docs/MODEL_INSPECTION.md` (§9), and update `MODULE_REFERENCE.md`, the
  README, the roadmap in `MODEL_REIMPLEMENTATION_PLAN.md`, and the status line of
  `HYPERPARAMETER_TUNING_PLAN.md`.

**Verify:** every code block in the doc runs as written; the non-slow suite
passes.

**Record (2026-10-13):** the simulator at 150 × 9, a `grouped` three-way
split (908 / 311 / 310 rows, 30 validation neighborhoods), Model 1 (LightGBM
Poisson per cohort) and Model 2 (Poisson × LR) at class defaults;
`COHORT_POISSON_DEVIANCE` on validation, 500 resamples: Model 1 by rows
2.589 [2.330, 2.842], by neighborhood 2.598 [2.200, 3.031]; Model 2 2.187
[1.967, 2.410] and 2.193 [1.961, 2.472]. The cluster interval is wider, as §9
says. The guide's five blocks run as written (44 s). The §10 raise was added
(one check, one test, one mutation killed).

## 9. Docs: what the model-inspection doc must say

- the percentile interval;
- why the cluster bootstrap (whole neighborhoods);
- that the interval is conditional on the fitted model (no refit);
- that few groups give wide, unstable intervals;
- that `COHORT_LOG_LOSS` is a ratio of sums recomputed per replicate, and that
  a replicate with no children raises (I12).

## 10. Risks

- **Few neighborhoods** give a degenerate cluster bootstrap: wide, unstable
  intervals, documented in the guide. With one group, every replicate would
  be the full set and the interval would have zero width while looking
  certain, so fewer than two distinct groups raise `ValueError` (decided
  2026-10-13; the check lives in the bootstrap, not the splitter, because the
  bootstrap takes its own `groups` and `random` / `stratified_by_group`
  splits accept one group).

## 11. Verification

- `uv run pytest -m "not slow"` before and after each sub-task.
- `uv run mypy`, plus `uv run mypy src/age_group_prediction/inspection <tests>`.
- ruff, the new tests under `-W error`, the mutation checks, and the smoke run
  in sub-task 2.
