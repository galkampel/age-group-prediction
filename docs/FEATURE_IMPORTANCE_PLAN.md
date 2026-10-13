# Plan: feature importance (permutation and SHAP), deferred

**Written for:** the model that picks this up in a later PR, and the user who
validates it. Self-contained.

**Status (2026-10-13): complete** as PR C of
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) §8, branch
`feat/feature-importance`: sub-tasks 1 (`PermutationImportance`, C1), 2
(`ShapExplainer`, C2) and 3 (the smoke run and the user guide
[MODEL_INSPECTION.md](MODEL_INSPECTION.md), C3). This material was planned as
sub-tasks 2–3 of [MODEL_INSPECTION_PLAN.md](MODEL_INSPECTION_PLAN.md)
(PR #14), which delivered only the bootstrap. The decisions below were
approved on 2026-10-10 and I5, I9 and I10 amended on 2026-10-13 while
building C2; the evidence is in §3. The code lives in
`age_group_prediction/inspection/` (I7) and `shap>=0.53,<1` is a main
dependency (I8).

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
| I5 | SHAP explains `predict(X, exposure)`: the **rate** when `explain` gets no exposure, the **count** when it gets the rows' exposure. The rows are explained one at a time, each with its own exposure value. | shap hands the prediction function batches of synthetic variants of **one** row of `X` without saying which row (§3), so an exposure of `len(X)` values has nothing to align with inside one shap call; driving shap one row at a time lets `predict(rows, exposure=np.full(len(rows), exposure[i]))` take the row's exposure literally, at the same cost (§3). shap's own pass-through masker (`maskers.Fixed`) fails inside shap 0.53. Models without exposure ignore the argument (`_check_exposure` returns `None`). |
| I6 | Permutation importance runs per raw column through `sklearn.inspection.permutation_importance`, with a scorer built from our `Metric`. | Probed (§3): it works with a cohort-table `y`, string columns and a closed-over exposure. sklearn shuffles one column of a copy with the rows in place, so an exposure taken by position stays aligned. There are no column blocks. |
| I7 | `inspection/permutation.py` and `inspection/shap_values.py`. | The package exists (PR #14). |
| I8 | `shap` is a main dependency (`shap>=0.53,<1`). | It is a core step after tuning, and the tests need no extra group. |
| I9 | Non-numeric columns reach shap as integer codes from a scikit-learn `OrdinalEncoder` fitted on `X ∪ background`; every column is cast to float for the masker; the prediction function decodes and restores `X`'s dtypes (a categorical dtype over the union of levels) before `predict`. NaN passes through. `Explanation.data` holds the codes and `display_data` the original values. `background` is aligned to `X`'s columns by name. | Probed (§3): shap's `Independent` masker fails on a string column (`TypeError: unsupported operand type(s) for -: 'str' and 'str'`) and on a bool column mixed with floats (`isfinite`). An encoder fitted on `background` alone raises on a level seen only in `X`. Casting back to `X`'s categorical dtype would turn a background-only level into NaN. shap pairs the masker's columns with `X`'s by position, so a reordered background gave wrong values silently (the review's finding). |
| I10 | `ShapExplainer(method, max_evals=None, random_state=None, n_jobs=1)`: `method` is `"exact"` (`shap.ExactExplainer`, `2^p` model evaluations per row, exact) or `"permutation"` (`shap.PermutationExplainer`, `max_evals` evaluations per row, approximate, seeded by `random_state`); `max_evals=None` is shap's default for the method; `n_jobs` explains rows in joblib processes. `background` is a required argument of `explain`. | Both explainers are model-agnostic and additive (§3). Exact is cheaper than Permutation's 500 evaluations up to about 9 columns and 25× faster at 5 (§3); Permutation covers wider tables. `random_state` with `"exact"` raises: Exact takes no seed. Processes give 3–4× on long runs and lose on short ones (§3), so the caller chooses. There is no sensible default background: the caller chooses its rows and size, which set the cost. |
| I11 | `explain` returns a `shap.Explanation` with the model's output names set (the cohort columns `predict` returns). Global importance is `mean(|values|)` per feature, shown in the doc. | shap drops the output names and labels them `Output 0..K-1` (§3). The `Explanation` keeps shap's plots, labelled with the original values through `display_data`. |

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
| shap re-verified 2026-10-13 (shap 0.53.0, numpy 2.5.1), under `-W error` | `import shap` and explaining emit no warning. A bool column mixed with floats fails too (`isfinite` on an object array). shap hands the prediction function a **fresh DataFrame with `X`'s columns** on every call; its rows are batches of synthetic variants of **one** explained row (3 rows → calls of 1600, 1600, 300, 1600, 300 rows), with no row index. `shap.maskers.Independent(data)` keeps 100 rows by default and drops the rest with a printed line, not a warning; `max_samples=len(data)` keeps all. `maskers.Fixed` + `Composite` (shap's pass-through input) fail inside shap for both explainers. |
| Exact vs Permutation, composite, 5 columns, 50 background rows, 20 rows | `ExactExplainer` 0.2 s (39 `predict` calls), `PermutationExplainer(max_evals=500)` 5.3 s (919 calls); values agree to 2.1e-5 on values of size 0.03 (Permutation is the approximation). Exact needs `2^p` evaluations per row: 32 at 5 columns, 512 at 9, 32768 at 15. `ExactExplainer` takes no seed and raises when `max_evals < 2^p`; `PermutationExplainer` raises below `2p+1`. |
| Rows explained one at a time (the row's exposure inside `predict`) | Same cost as shap's batched call (shap loops over rows itself): Exact 0.04 s vs 0.05 s for 50 rows, Permutation 0.19 s vs 0.18 s for 10 rows. Additivity to `predict(X, exposure)`: ≤ 6e-14 for `CountModel` and the composite, both methods. joblib processes over rows (`n_jobs=4`, 8 CPUs): composite Permutation 13.1 s → 4.5 s per 50 rows, Exact 0.3 s → 3.5 s (process start-up); threads give nothing. |
| Review findings (reproduced) | A `background` with its columns in another order gave wrong values silently (`predict = x + y`: `[[-149, 10], ...]` instead of `[[1, -140], ...]`; additivity still held). NaN in a string column crashed inside `OrdinalEncoder.inverse_transform` on pandas 3's read-only slice. A `category`-dtype column with a background-only level became NaN on the cast back (a pandas 4 deprecation warning). All three fixed with tests. An int64 value above 2**53 loses precision through the float codes (documented, not fixed). |
| Cost as built (50 background rows, 50 rows, composite) | `"exact"`: 0.7 s. `"permutation"` at `max_evals=500`: 13 s. `CountModel`: 0.4 s / 8.5 s. Each `predict` call re-runs the feature transformer, so the cost grows with the background size × evaluations per row. |

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
# inspection/shap_values.py (as built in C2)
@pydantic_dataclass(frozen=True, config=STRICT)
class ShapExplainer:
    method: Literal["exact", "permutation"]
    max_evals: int | None = None      # None: shap's default for the method
    random_state: int | None = None   # permutation only; raises with exact
    n_jobs: int = 1                   # rows in joblib processes

    def explain(self, model, X, background, *, exposure=None) -> shap.Explanation:
        # OrdinalEncoder on X ∪ background for the non-numeric columns; encode
        # casts every column to float (I9); predict(X_encoded, exposure_value)
        # decodes, restores X's dtypes and calls
        # model.predict(rows, exposure=np.full(len(rows), exposure_value)) (I5);
        # Independent(encode(background[X.columns]), max_samples=len(background));
        # one Exact/Permutation explainer call per row (I10), the parts
        # assembled into one Explanation with feature_names, display_data
        # and the cohort output names (I11)
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

### [x] 2. `ShapExplainer`

- Re-add shap (see "To start").
- Write `inspection/shap_values.py` and its tests.
- Tests:
  - Additivity against `predict(exposure=1)` for `CountModel` with exposure.
  - The multi-output shape and cohort output names.
  - A string column round-trips through the codes.
  - The same seed gives identical values.

**Verify:** the same. *Done 2026-10-13:* 12 tests (28 cases) in
`tests/unit/test_inspection_shap.py`, both methods where they apply:
additivity to the rate and to the count, the exact values of an additive
model with a reordered background, cohort names and shape, str/int/bool
round-trip with a level only in `X`, NaN and a categorical column, the
whole background used, the seed, parallel rows identical and in worker
processes, the exposure length, eleven invalid settings, and the composite
on the simulator. 21 mutations each fail a test; the independent review's
findings are in §3.

### [x] 3. Smoke run and docs

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
passes. *Done 2026-10-13:* the guide's worked example (simulator 150 × 9,
`grouped` split, Model 1 = LightGBM Poisson per cohort and Model 2 =
Poisson × LR at class defaults) runs as written. Permutation on validation
and on train, per component for Model 2; SHAP exact on 200 validation rows
against 50 training rows with the validation exposure. Findings: the two
methods agree on Model 2's top four (`avg_household_size`, `median_age`, the
room shares); on Model 1 `ses` is 2nd by mean |SHAP| and 7th by permutation
(correlated columns); Model 1 at defaults overfits (training importances 4
to 100 times the validation ones); Model 2's total depends on
`avg_household_size` (0.69 of the deviance). Costs at 9 raw columns: exact
SHAP 4 s per 200 rows for Model 2, 90 s for Model 1 (27 s on four
processes).

## 6. Risks

- **SHAP cost.** With `"exact"` it is seconds for a few hundred rows of a
  composite model at a handful of raw columns, but `2^columns` grows fast:
  past about ten columns use `"permutation"`, which is minutes (§3). The
  doc gives the subsampling recipe, and the smoke run measures the cost at
  simulator scale.
- **A category in `X` but not in `background`, or the reverse.** The codes
  come from both frames (I9), and a categorical dtype is restored over the
  union, so both decode correctly.
- **Column order.** shap pairs the masker's columns by position; `explain`
  aligns `background` to `X`'s columns by name, so a reordered background is
  safe and a missing column raises `KeyError`.
