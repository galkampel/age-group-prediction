# Validation report: PR #12 (any regressor in `DirectCohortModel`; the feature transformer inside each model)

**Validated:** branch `feat/estimator-and-feature-transformer` at `bb3eb48` (= `origin`), PR #12 (draft)
into `feat/hyperparameter-tuning` at `1ea99e9`. **Date:** 2026-10-05. **Brief:**
[DIRECT_COHORT_GENERALIZATION_VALIDATION.md](DIRECT_COHORT_GENERALIZATION_VALIDATION.md).
**Scripts and tests:** `docs/validation/pr12/` (§5). Nothing under validation was changed.

> **2026-10-07:** the scripts and tests of `docs/validation/pr12/` were removed (the code they
> checked is renamed and later rebuilt; the behaviors of ours they pinned are in `tests/unit/`, except that `estimator` has
> no default, which the signature enforces).
> They are kept in commit `c966edb`: `git show c966edb:docs/validation/pr12/<file>`.

## 1. Verdict: ACCEPT WITH NOTES

The code implements every decision of the plan's §2 as stated, and each one is backed by a probe
or test I ran (§2, V1). The derivation in §2b follows step by step, and the two numerical
identities it rests on hold: the weighted rate reproduces LightGBM's own `init_score` offset to
1.5e-8 relative, an unpenalized `PoissonRegressor` through the model reproduces statsmodels'
offset GLM to 1e-8, and the Gaussian path equals weighted least squares of the count. The tests
catch what they claim: all 16 mutations I ran (6 recorded, 10 new) were killed by the named
tests, with no survivor. All 13 code blocks of the edited doc sections, the tuning plan's §5
example and the plan's §6 probe run under `-W error`; 138 markdown anchors resolve; the routine
(ruff, mypy, 203 targeted tests, the 1230-test suite, the slow statsmodels oracles) is clean.
The findings are Low: the live PR body is still the Step 0 draft and describes an API that no
longer exists (`estimator=None`, `default_estimator()`), so it must be replaced with the plan's
§7 before merge; and one test comment quotes a different measurement than the docs.

## 2. Phase table

| Phase | Result | Evidence |
|---|---|---|
| V0 Orientation | pass | `git diff --stat feat/hyperparameter-tuning...HEAD`: **30** files (the brief's 29 plus the brief itself), all under `modeling/`, `hyperparameter_tuning/`, their tests and `docs/`; `pipeline.py` and `test_modeling_pipeline.py` deleted. Baseline `uv run pytest -m "not slow"`: **1230 passed, 1 skipped, 1 xfailed** (98 s). statsmodels 0.14.6, scikit-learn 1.9.0, lightgbm 4.7.0, optuna 4.9.0 |
| V1 Decisions | pass | `test_decisions.py`: 14 tests, one or more per decision (table below) |
| V2 Math | pass | Re-derived (A)–(E) by hand, every step follows (§2.2). `probe_doc_identity.py`: §0.1 equals plan §2b, **IDENTICAL: 6858 characters, 118 lines** |
| V3 Tests | pass | 16 mutations, 0 survivors (§4). Test review in §2.3 |
| V4 Docs | pass | `probe_doc_blocks.py`: 13 blocks ran (10 of `FEATURE_TRANSFORMATIONS.md` §8.0–§8.3, 2 of `DIRECT_COHORT_MODEL.md` §0, 1 of `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0) in one namespace under `-W error`, on the simulator's table (245 rows); `probe_tuning_plan_s5.py`: §5 ran with a seeded-TPE stand-in; `probe_end_to_end.py`: `{'n_kindergarten': 0.806, 'n_elementary': 0.792}`; `test_doc_claims.py`: 10 tests; `probe_anchors.py`: **138 links checked, 0 broken**; old-API grep (§2.4) |
| V5 Rule 7 | pass | Every added comment and docstring read (§2.5); no derivation, example or motivation left; no silent-failure constraint lost |
| V6 Routine | pass | `ruff check`: All checks passed; `ruff format --check`: 22 files already formatted; `uv run mypy`: no issues in 23 files; targeted mypy: no issues in 22 files; `pytest -W error` on the modeling and evaluator tests: **203 passed**; non-slow suite **1230 passed, 1 skipped, 1 xfailed**; `tests/validation/test_total_children.py`: **2 passed** |
| V7 PR body | **finding L1** | The live body is the Step 0 draft; `diff` against plan §7 in §3 |

### 2.1 V1, decision by decision (`docs/validation/pr12/test_decisions.py`)

| Decision | Test | What it showed |
|---|---|---|
| G1, G1a | `test_g1_constructor_signature_and_nested_names` | `__init__(self, *, estimator, use_exposure, feature_transformer)`, all keyword-only; `get_params(deep=True)` has `estimator__n_estimators`; no LightGBM name or `objective` among the shallow params; `clone(...).set_params(estimator__n_estimators=5)` leaves the template at 50 |
| G1b | `test_g1b_regressor_is_an_exported_protocol` | `Regressor` in `modeling.__all__`, `typing.is_protocol`, `fit(self, X, y, sample_weight=None)`, `predict(self, X)`; mypy clean (V6) |
| G2 | `test_g2_estimator_is_required` | no default; `DirectCohortModel()` raises `TypeError`; `check_parameters_default_constructible` passes in the contract test (V6) |
| G3 (a) | `test_g3a_weighted_rate_equals_lightgbm_init_score_offset` | 2000 rows, 50 trees: max relative difference **1.538e-08** (the plan: 1.5e-8) |
| G3 (b) | `test_g3b_unpenalized_glm_equals_statsmodels_offset_glm` | `PoissonRegressor(alpha=0)` coefficients `[-1.99731226, 0.39340554, -0.00671332]` equal statsmodels `GLM(Poisson, offset=log E)` to the printed digits; fitted values within 1e-5 |
| G3 (c) | `test_g3c_penalized_glm_alpha_acts_as_alpha_times_mean_exposure` | at `alpha=1`, the model's coefficients `[-1.9299, 0.04877, -0.001161]` equal the offset objective minimized by hand at `alpha·mean(E)`, to 4e-8 |
| G3 (d) | `test_g3d_hist_gradient_boosting_fits_with_the_exposure` | fits and predicts; positive; training mean within 1% of the target mean |
| G4 | `test_g4_gaussian_weighted_rate_is_wls_of_the_count` | `LinearRegression` through the model equals `sm.WLS(y, [E, E·x], weights=1/E)` to 1e-10; the Gaussian LightGBM doubles with the exposure |
| G5 | `test_g5_no_pre_check_of_sample_weight` | Python's own `TypeError: ... unexpected keyword argument 'sample_weight'` at `fit`; the source of `fit` has no other mention of `sample_weight`; the same estimator fits without an exposure |
| G6 | `test_g6_fitted_state_and_predict_follows_it` | fitted attributes are exactly `{estimator_, use_exposure_, feature_transformer_}`; no `regressor_`, `base_log_rate_`; after `set_params(use_exposure=False)`, `predict(X)` raises "pass `exposure`" and `predict(X, exposure=E)` is unchanged |
| F1 | `test_f1_feature_transformer_on_leaf_models_only_and_helpers` | in the three leaf `__init__`s, not the composites; `_fit_features` returns a fitted copy, the template unfitted; `_transform_features` centres new rows on the training mean and drops the unplanned column |
| F2 | `test_f2_model_pipeline_is_gone` | not in `__all__` or the package; `modeling.pipeline` raises `ModuleNotFoundError`. `grep -rn ModelPipeline src tests`: empty |
| F3 | `test_f3_evaluator_has_build_model_and_fits_the_model_per_fold_on_raw_rows` | no `feature_transformer` field; `build_model` exists, the old name does not. A recording `FeatureTransformer` inside a `DirectCohortModel` inside the evaluator, `FixedTrial`, 3 grouped folds: each fit saw exactly that fold's training rows; each transform (training design at fit, validation rows at predict) used that fold's fit; the template model and its transformer stay unfitted |
| I1, I2 | `test_i1_i2_prediction_follows_y_order_and_the_template_stays_unfitted` | mapping given as `{b, a}`, `y` as `[a, b]`: `predict` columns `['a', 'b']`, `cohort_models_` keyed in `y`'s order; `cohort_models` is the very object given, unfitted; `cohort_models_` holds fitted copies |

### 2.2 V2, the derivation

Re-derived independently from the setup:

- (A) $\sum_b[\mu_b - y_b\log\mu_b]$ with $\log\mu_b = \log E_b + F_b$ gives $\sum_b[E_b e^{F_b} - y_b F_b] - \sum_b y_b\log E_b$. ✓
- (B) $\sum_b E_b[e^{F_b} - (y_b/E_b)F_b] = \sum_b[E_b e^{F_b} - y_b F_b]$. ✓
- (C) The difference is $-\sum_b y_b \log E_b$, free of $F$; $\partial/\partial F_b = E_b e^{F_b} - y_b = \mu_b - y_b$, $\partial^2/\partial F_b^2 = \mu_b$. The weighted start $\log(\sum w_b r_b/\sum w_b) = \log(\sum y_b/\sum E_b)$. ✓ (G3a confirms the trees are the same to floating point.)
- (D1)–(D3) $\mathcal{L}_{\text{Gauss}} = \mathcal{L}_{\text{rate}}/\sigma^2 + \tfrac12\sum_b\log(2\pi\sigma^2E_b)$; $\partial\mathcal{L}_{\text{rate}}/\partial f_b = -E_b(y_b/E_b - f_b) = \mu_b - y_b$; Hessian $E_b$; the constant minimizer solves $\sum_b(y_b - E_b f) = 0$, so $f = \sum y/\sum E$. ✓ (G4 confirms the WLS identity.)
- (D4), (E) are arguments, consistent with (A)–(D3). ✓
- Claim in (B) that scikit-learn and LightGBM accept any real $r \ge 0$ as a Poisson target: used by every test with a fractional rate; holds.

The doc copy in `DIRECT_COHORT_MODEL.md` §0.1 is byte-identical to plan §2b from "**Setup.**" to the end of (E).

### 2.3 V3, the tests as read

Each test in the eight files names a mistake in our code, with these notes:

- `test_modeling_base.py`: `test_fit_features_fits_a_copy_and_leaves_the_template_unfitted` and
  `test_transform_features_uses_the_statistics_learned_at_fit` are subsumed by the per-leaf
  tests in `test_modeling_feature_transformer.py` (mutations R1 and N4 fail both sets together).
  I agree with the Step 7 note: redundant, harmless; `test_without_a_transformer_the_helpers_return_x_itself` is the one check of the `is X` path and stays.
- `test_modeling_direct_cohort.py`: `test_a_regressor_without_sample_weight_surfaces_the_library_error` and `test_an_all_zero_target_surfaces_lightgbms_error` pin library behaviour, on purpose, to document G5 and the deliberate absence of a check. Acceptable as documented. See finding L2 for a comment.
- `test_hyperparameter_tuning_evaluator.py`: the hand-written fold loop is the end-to-end oracle; with the recording-transformer test deleted in Step 5, per-fold fitting of the transformer is covered only through `DirectCohortModel`'s own behaviour plus `test_every_trial_sees_identical_folds` (the rows `fit` sees). My `test_f3_...` adds the direct observation of the transformer per fold; mutations N5/N6/N8/N10 show the existing tests catch each anyway.
- No test tests nothing; no two tests are exact duplicates.

### 2.4 V4, the claims

Checked by `test_doc_claims.py` and the block runs:

- `DIRECT_COHORT_MODEL.md` §0.3: LightGBM fits a zero, NaN or infinite exposure without an error; `HistGradientBoostingRegressor(loss="poisson")` raises `ValueError` on each. ✓ A scikit-learn `Pipeline` as the estimator raises `ValueError` with the exposure. ✓ An unknown objective and an all-zero `y` raise `LightGBMError`. ✓
- §0.6: `cohort_models__a__estimator__learning_rate` raises `AttributeError`; `estimator__learning_rate` on the leaf model works. ✓ Targets may stay in `X` with the default `remainder="drop"`: `predict(table)` equals `predict(table[["x"]])`. ✓
- `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0.2: the fitted-state lists for both sub-models (`feature_transformer_`, `feature_names_in_`, `n_features_in_`, no `dispersion_` under Poisson, `coef_` of shape (d, K)); §0.4: `cohort_probability_model__l2_penalty` reaches the sub-model. ✓
- `MODULE_REFERENCE.md`: the `__init__.py` row lists exactly `modeling.__all__`, sorted; no `pipeline.py` row, no `ModelPipeline`; the `direct_cohort.py`, `base.py`, `independent_cohorts.py`, `total_children.py`, `cohort_probability.py` and `evaluator.py` rows match the code (`feature_engineering` added to the three leaf models' dependencies). ✓
- `FEATURE_TRANSFORMATIONS.md` §3.3 and §8.1–§8.3: the text matches the code; the blocks run with `feature_transformer=` on each model and `exposure.loc[...]` on the raw table. ✓
- Old API in live docs: `grep` over the live docs hits only `objective=` on an estimator, `init_score` as the mechanism being compared against (§0.1 (A), §0.4, §0.5), the tuning plan's ticked task with its dated "Done" note, and `model__exposure=` in the tuning plan's §9 history about scikit-learn's `Pipeline`. No live text describes the removed API as current. ✓

### 2.5 V5, rule 7 over `git diff feat/hyperparameter-tuning...HEAD -- src`

Every added or changed comment and docstring states what the code does or a constraint the
caller must know. The only motivational clauses are two short ones already listed as optional
by Step 7: `base.py` "`# y as sklearn's Pipeline passes it.`" and "`Never refitted, so a row's
prediction does not depend on the rows predicted with it.`". Not counted. Trimmed text that
was a silent-failure constraint: the LightGBM `subsample_freq` note, now in `DIRECT_COHORT_MODEL.md`
§0.2 and the PR body only, by the user's decision (out of scope). The `n_jobs=1` macOS note
moved to §0.2 and to every example and test; fine, it is an environment constraint, not a
model one.

## 3. Findings

### L1 (Low) The live PR body is the Step 0 draft, and names an API the PR does not have

`gh pr view 12 --json body` differs from plan §7 (the final form) in: item 1 says
`estimator=None` and `DirectCohortModel.default_estimator()`, both removed in Step 2's revision
(G2); item 3 says the composites are "unchanged in code"; there is no item 5 (Docs), no
"Follow-up" section; the migration lines lack `build_model`, the `estimator__` prefix and the
`subsample_freq` note; the Steps checklist is unticked (0 of 7); "Checks" says "mypy strict"
and has no counts. Reproduction:

```
gh pr view 12 --json body -q .body | sed 's/\r$//' > live.md
awk '/^\*\*Body:\*\*/{p=1;next} p&&/^```markdown/{q=1;next} q&&/^```$/{exit} q' \
    docs/DIRECT_COHORT_GENERALIZATION_PLAN.md > plan.md
diff live.md plan.md        # 6,8c6,8  21c21,22  24a26,29  33c38,42  34a44,49  38,44c53,59  48,50c63,67
```

Suggested fix (the user's action): paste plan §7's body into the PR before marking it ready.
Not applied.

### L2 (Low) A test comment quotes a measurement the docs do not

`tests/unit/test_modeling_direct_cohort.py:216`: "measured agreement 1.7e-8" (on the test's
300 rows, 20 trees), while plan §2b, `DIRECT_COHORT_MODEL.md` §0.1 and §0.5 say 1.5e-8 (2000
rows, 50 trees). Both are true on their own data (my 2000-row run: 1.538e-8), and the test
asserts `rtol=1e-6`. Suggested fix: cite one figure or the tolerance only. Not applied.

### Notes, not findings

- The brief says the diff is 29 files; it is 30, the 30th being the brief.
- `test_modeling_base.py`'s two subsumed helper tests (§2.3), already listed by Step 7 as optional.

## 4. Mutation results

Each mutation was applied in place to the repo's `src`, the file backed up to the scratchpad,
the targeted tests run (`-W error`; the 10 modeling and evaluator files plus
`docs/validation/pr12/`, 204 tests), the file restored by copy and its md5 compared with the
backup (all equal; `git status` clean afterwards). Runner:
`/private/tmp/claude-501/…/scratchpad/mutate.py`.

| # | Mutation | Failed | Tests that failed (ours; mine in italics) |
|---|---|---|---|
| R1 | `base._fit_features` fits the template (no `clone`) | 5 | `test_fit_features_fits_a_copy_and_leaves_the_template_unfitted`, `test_the_template_transformer_stays_unfitted` ×3, *test_f1* |
| R2 | `direct_cohort.fit` drops `sample_weight` | 10 | `test_fit_receives_the_rate_with_the_exposure_as_weight`, `test_the_weighted_rate_equals_lightgbms_offset`, `test_the_gaussian_weighted_rate_is_least_squares_of_the_count`, `test_a_regressor_without_sample_weight_surfaces_the_library_error`, `test_training_mean_prediction_matches_the_target_mean[lightgbm-exposure]`, *g3a, g3b, g3c, g4, g5* |
| R3 | `predict` drops `* exposure` | 16 | `test_doubling_the_exposure_doubles_the_prediction` ×4, `test_predict_follows_how_the_model_was_fitted`, the offset, Gaussian and mean tests, `test_doubling_the_exposure_doubles_only_the_cohorts_with_an_offset`, *g3a, g3b, g3d, g4, g6* |
| R4 | `CohortProbabilityModel.predict` transforms too | 1 | `test_rows_are_transformed_with_the_training_statistics[CohortProbabilityModel]` |
| R5 | `_transform_features` checks `feature_transformer` (the template) | 3 | `test_predict_follows_the_transformer_fitted_at_fit` ×3 |
| R6 | `build_model` returns `self.model` (no `clone`) | 3 | `test_build_model_returns_an_unfitted_copy`, `test_the_model_template_is_left_unfitted`, *f3* |
| N1 | `predict` divides by the exposure | 16 | as R3 |
| N2 | target `y * exposure` | 12 | recording, offset, Gaussian, follows-fit, mean ×3, *g3a–g3d, g4* |
| N3 | `sample_weight=y` | 13 | as N2 plus *g5* |
| N4 | `_fit_features` returns the raw `X` | 24 | `test_transform_features_uses_the_statistics_learned_at_fit`, the statistics test ×3, the logits test, 5 of `independent_cohorts`, 12 of `independent_total_probability` (feature-name mismatch, loud), `test_matches_a_hand_written_fold_loop`, *f3* |
| N5 | evaluator predicts on `X_train` | 3 | `test_exposure_is_sliced_to_each_fold`, `test_matches_a_hand_written_fold_loop`, *f3* |
| N6 | evaluator fits on all rows | 5 | `test_every_trial_sees_identical_folds`, `test_exposure_is_sliced_to_each_fold`, `test_matches_a_hand_written_fold_loop`, `test_a_dataframe_target_is_split_and_scored_per_fold`, *f3* |
| N7 | `IndependentCohortModels` iterates the mapping | 2 | `test_columns_follow_y_and_rows_follow_X`, *i1_i2* |
| N8 | `DirectCohortModel.predict` skips the transform | 8 | the statistics test [Direct], 5 of `independent_cohorts`, the hand loop, *f3* |
| N9 | `TotalChildrenModel.predict` skips the transform | 11 | the statistics test [Total], 10 of `independent_total_probability` (loud) |
| N10 | `DirectCohortModel.fit` hands the estimator the raw `X` | 8 | as N8 |

**Survivors: none.** No gap test was needed. Every recorded mutation (R1–R6) fails the test the
plan names for it.

## 5. Scripts and tests written (`docs/validation/pr12/`)

| File | Claim | Result |
|---|---|---|
| `test_decisions.py` | G1–G6, F1–F3, I1–I2 against the code (§2.1) | 14 passed |
| `test_doc_claims.py` | §0.3 bad exposures, `Pipeline` error, LightGBM errors; §0.6 nested names and targets in `X`; Model 2 §0.2/§0.4 state and nested names; `MODULE_REFERENCE.md` rows | 10 passed |
| `probe_doc_identity.py` | §0.1 equals plan §2b word for word | IDENTICAL, 6858 characters |
| `probe_doc_blocks.py` | the 13 python blocks of the three edited sections, one namespace, `-W error` | 13 blocks ran |
| `probe_tuning_plan_s5.py` | the tuning plan's §5 block with a seeded-TPE `HyperparameterStudy` stand-in (`n_trials` 30→3 for time) | ran; three fitted models with `estimator__…` best params |
| `probe_end_to_end.py` | plan §6, verbatim | `{'n_kindergarten': 0.806, 'n_elementary': 0.792}` |
| `probe_anchors.py` | every `#slug` / `file.md#slug` / `file.md` link in the 11 touched docs | 138 checked, 0 broken |

Re-run everything:

```
uv run pytest -q -c pyproject.toml --rootdir . -W error docs/validation/pr12/
for p in probe_doc_identity probe_doc_blocks probe_tuning_plan_s5 probe_end_to_end probe_anchors; do
  PYTHONPATH=src uv run --group test python docs/validation/pr12/$p.py | grep -v "^\[LightGBM\]"
done
```

## 6. Not verified

- **"1221 passed before the PR."** Checking it needs the base branch checked out, which the brief
  forbids; taken from the plan's Step 0 record.
- **"More OpenMP threads crash alongside torch on macOS"** (the `n_jobs=1` rule): not provoked.
- **(D3)'s "full maximum-likelihood fit of (D1), σ² included, to 1e-8"**: I checked the
  weighted-least-squares identity (1e-10), not the joint fit with σ².
- **§0.5's smoke-run numbers** (4% deviance gain, 8 of 10 populations): historical records from
  earlier PRs, outside this diff.
- The mutation runner ran the targeted 204 tests per mutation, not the full suite; the
  mutations touch only `modeling/` and `evaluator.py`, whose tests are those files.
