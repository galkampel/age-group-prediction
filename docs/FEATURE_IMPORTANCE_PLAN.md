# Plan: feature importance (permutation and SHAP), deferred

**Written for:** the model that picks this up in a later PR, and the user who
validates it. Self-contained.

**Status (2026-10-13):** in progress as PR C of
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §8, branch
`feat/feature-importance`. Sub-task 1 (`PermutationImportance`, C1) is done;
sub-tasks 2–3 (C2, C3) are open. This material was planned as sub-tasks 2–3 of
[MODEL_INSPECTION_PLAN.md](MODEL_INSPECTION_PLAN.md) (PR #14), which delivered
only the bootstrap. The decisions below were approved on 2026-10-10; the
evidence was measured then and re-verified on 2026-10-13 (§3). Re-verify both
against the code of the day before relying on them.

**To start:**
1. Re-add `shap>=0.53,<1` to `[project] dependencies`, then run `uv lock` and
   `uv sync`. PR #14 removed it, unused.
2. Follow the working rules of [MODEL_INSPECTION_PLAN.md](MODEL_INSPECTION_PLAN.md) §3.
3. Write the code in the existing `age_group_prediction/inspection/` package
   (I7), next to `bootstrap.py`.

## 1. Goal

Feature importance of a **fitted** `BaseAgeGroupModel`, in two variants:
permutation importance and SHAP. In practice the model is the tuned model
refit on the full training data.

Out of scope:
- importance of engineered features (raw columns only);
- plots (shap's own plotting works on the returned `Explanation`).

## 2. Decisions

The numbers are those of MODEL_INSPECTION_PLAN.md, where I1, I2 and I7 remain.

| # | Decision | Why |
|---|---|---|
| I3 | Importance runs on whatever `X, y` the caller passes. | The user's choice. The doc recommends held-out data for permutation importance. |
| I4 | Importance is computed per **raw input column** of `X`. | It covers every `BaseAgeGroupModel`, composites included, and one-hot or interaction expansions map back to their source column. |
| I5 | SHAP explains the **rate per unit exposure**, `predict(X, exposure=1)`. | SHAP predicts on synthetic masked rows, so a per-row exposure cannot follow them. For an exposure model, `predict = rate × exposure`, and the features drive the rate. Models without exposure ignore the argument (`_check_exposure` returns `None`). |
| I6 | Permutation importance runs per raw column through `sklearn.inspection.permutation_importance`, with a scorer built from our `Metric`. | Probed (§3): it works with a cohort-table `y`, string columns and a closed-over exposure. sklearn shuffles one column of a copy with the rows in place, so an exposure taken by position stays aligned. There are no column blocks. |
| I7 | `inspection/permutation.py` and `inspection/shap_values.py`. | The package exists (PR #14). |
| I8 | `shap` is a main dependency (`shap>=0.53,<1`). | It is a core step after tuning, and the tests need no extra group. |
| I9 | SHAP sees non-numeric columns as integer codes. The prediction function decodes them back before `predict`. | Probed (§3): shap's `Independent` masker fails on string columns with `TypeError: unsupported operand type(s) for -: 'str' and 'str'`. The codes are the categories observed in `background` and `X`. |
| I10 | `ShapExplainer` uses `shap.PermutationExplainer`, and `background` is a required argument of `explain`. `max_evals` defaults to shap's own default, 500. | The explainer is model-agnostic and exactly additive (§3). There is no sensible default background: the caller chooses its rows and size, which set the cost (§3). |
| I11 | `explain` returns the `shap.Explanation` with the model's output names set (`y`'s cohort columns). Global importance is `mean(|values|)` per feature, shown in the doc. | shap drops the output names and labels them `Output 0..K-1` (§3). The `Explanation` keeps shap's plots. |

## 3. Measured evidence

Measured 2026-10-10 with shap 0.53.0, sklearn 1.9.0 and pandas 3. The setup:
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
| Re-verified 2026-10-13 (sklearn 1.9.0, pandas 3.0.5, pydantic 2.13.4), under `-W error` | The permutation row is unchanged. On the simulator (150 neighborhoods, 624 rows), `TotalTimesProbabilityModel` (Poisson × LR, OneHot on the string column `school_status`) with `COHORT_POISSON_DEVIANCE`: `avg_household_size` 0.38, `median_age` 0.20, `n_daycares_500m` 0.08, `ses` 0.02, `school_status` 0.009; 0.13 s for 5 columns × 5 repeats. A NaN score gives NaN importances with no error (hence the finite check). Every length mismatch already raises in the model or the metric. |
| Cost (`max_evals=500`, 100 background rows) | `CountModel`: 15.7 s per 50 rows. `TotalTimesProbabilityModel`: 32.6 s per 50 rows, and 222 s per 500 rows. Each `predict` call re-runs the feature transformer, so the cost grows with the background size × `max_evals`. |

## 4. Target code shape

Settings go in the constructor, as a pydantic dataclass like
`BootstrapEvaluator`. Data are method arguments.

```python
# inspection/permutation.py (as built in C1)
@pydantic_dataclass(frozen=True, config=STRICT)   # age_group_prediction._config
class PermutationImportance:
    metric: Metric
    n_repeats: int
    random_state: int | None = None

    def compute_importances(self, model, X: pd.DataFrame, y, *, exposure=None) -> pd.DataFrame:
        # scorer = ±metric(y, est.predict(X, exposure=exposure)), the sign from
        # greater_is_better, so a positive importance means the column helps;
        # one row per column of X: importance_mean, importance_std; raises on
        # a non-finite importance
```

`STRICT` rejects a misspelt keyword, a bool or string where an int is
expected, and numpy integers (`np.int64`): pass Python ints, as the tuning
classes require.

```python
# inspection/shap_values.py
class ShapExplainer:
    max_evals: int = 500
    random_state: int | None = None

    def explain(self, model, X, background) -> shap.Explanation:
        # encode non-numeric columns as codes (I9); f decodes and returns
        # model.predict(frame, exposure=np.ones(len(frame))) as an array (I5);
        # shap.PermutationExplainer(f, Independent(background), seed=random_state);
        # set output names (I11)
```

## 5. Sub-tasks

### [x] 1. `PermutationImportance`

- Write `inspection/permutation.py` and its tests.
- Tests:
  - A column the model ignores scores 0, and a used one scores > 0.
  - A positive score means "important" under both metric directions.
  - Exposure stays row-aligned while columns are permuted.
  - It works on `TotalTimesProbabilityModel` with a string column.

**Verify:** the tests pass under `-W error`; each mutation fails a test.
*Done 2026-10-13:* eight tests (11 cases) in
`tests/unit/test_inspection_permutation.py`; seven mutations (sign, exposure
reversed, mean/std swapped, seed not passed, finite check dropped,
`extra="forbid"` dropped, `strict` dropped) each fail a test.

### [ ] 2. `ShapExplainer`

- Re-add shap (see "To start").
- Write `inspection/shap_values.py` and its tests.
- Tests:
  - Additivity against `predict(exposure=1)` for `CountModel` with exposure.
  - The multi-output shape and cohort output names.
  - A string column round-trips through the codes.
  - The same seed gives identical values.

**Verify:** the same.

### [ ] 3. Smoke run and docs

- On the simulator data, compute both importances for `CountModel` and
  `TotalTimesProbabilityModel`, then compare the rankings of permutation and
  mean |SHAP|.
- Add an importance section to `docs/MODEL_INSPECTION.md`:
  - **Permutation:** the definition (the score drop when one column is
    shuffled, averaged over repeats); why to use held-out data; that correlated
    columns share or hide importance.
  - **SHAP:** Shapley values of the rate (I5) and their additivity; the code
    encoding (I9); the cost (§3) and how to subsample `background` and `X`;
    mean |SHAP| as the global importance.

**Verify:** every code block in the doc runs as written; the non-slow suite
passes.

## 6. Risks

- **SHAP cost.** It is minutes for a few hundred rows of a composite model
  (§3). The doc gives the subsampling recipe, and the smoke run measures the
  cost at simulator scale.
- **A category in `X` but not in `background`.** The codes come from both
  frames (I9), so it decodes correctly.
