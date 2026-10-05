# Plan: Any regressor in `DirectCohortModel`, and the feature transformer inside each model

**Branch:** `feat/estimator-and-feature-transformer`, from `feat/hyperparameter-tuning` (where `modeling/` lives; PRs #6–#11 used the same base). PR into `feat/hyperparameter-tuning`.
**Status (2026-10-05):** Step 0 done except the first commit and the draft PR (§7); next Step 1. Handoff for the implementing session: `DIRECT_COHORT_GENERALIZATION_HANDOFF.md`. Baseline `uv run pytest -m "not slow"`: **1221 passed, 1 skipped, 1 xfailed**.
**Source of truth:** this file. Update its status and checkboxes at every stop.

## Contents

1. Context
2. Decisions (including the answers to the four questions)
3. How to work
4. Target code shape
5. Steps 0–7, each with files, changes, tests and a "done when"
6. Verification
7. Draft PR title and body

## 1. Context

Model 1 (`IndependentCohortModels`) fits one `DirectCohortModel` per cohort. Today
`DirectCohortModel` is hard-wired to `lightgbm.LGBMRegressor`: it mirrors ten LightGBM
hyperparameters as constructor arguments, and its exposure offset is LightGBM-specific
(`init_score` at fit, `raw_score=True` plus the offset at predict). Feature
transformation reaches a model only through `ModelPipeline(feature_transformer, model)`,
a wrapper every cohort and every Model 2 sub-model has to be put inside.

The user wants:

1. `DirectCohortModel` to take **any regressor** with a Poisson or Gaussian objective
   (LightGBM, scikit-learn's `HistGradientBoostingRegressor`, `PoissonRegressor`, …),
   with the exposure offset applied in a way that works for all of them.
2. The **feature transformer as a parameter of the model itself**, so `ModelPipeline`
   goes away and a model is configured in one place.
3. Answers on `IndependentCohortModels`: `dict` vs `OrderedDict`, and why both
   `cohort_models` and `cohort_models_` exist.

The tuning evaluator (`hyperparameter_tuning/evaluator.py`) is the one consumer of
`DirectCohortModel`; its own plan already has an open task to stop transforming
features itself ("Evaluator on the raw table"). This plan closes that task too.

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| G1 | `DirectCohortModel(*, estimator=None, use_exposure=False, feature_transformer=None)`. The ten LightGBM hyperparameters and `objective` are removed; they live on the estimator | The estimator carries its own loss and hyperparameters; mirroring them would have to be redone for every library. Tuning reaches them by nested names (`estimator__n_estimators`), which `BaseEstimator.set_params` already supports (verified) |
| G2 | `estimator=None` builds LightGBM in `fit`: `LGBMRegressor(objective="poisson", n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1, random_state=42)`, returned by the static method `DirectCohortModel.default_estimator()` | sklearn's `check_parameters_default_constructible` (in `test_modeling_contract.py`) only allows `None`, scalars, tuples, types and callables as defaults, so an estimator instance cannot be the default. A tuner needs an instance to reach `estimator__…`, so the default is also available as a factory |
| G3 | **The exposure enters as a weighted rate, for every estimator:** `fit(X, y / exposure, sample_weight=exposure)`; `predict` returns `estimator_.predict(X) * exposure` | For a Poisson loss this is the *same* likelihood as the offset `log(exposure)`: `Σ E_i (μ_i − r_i log μ_i)` with `r_i = y_i/E_i` has the same gradient and hessian per row as `Σ (E_i μ_i − y_i log(E_i μ_i))`. Verified on LightGBM: weighted-rate vs `init_score` predictions agree to 1.5e-8 relative. It needs only `sample_weight`, which LightGBM, `HistGradientBoostingRegressor`, `PoissonRegressor` and most regressors accept; `init_score` exists only in LightGBM. One code path, no `raw_score`, no stored intercept. The user chose this over an `init_score` branch |
| G4 | **`use_exposure` is allowed with a Gaussian loss.** The old guard ("an exposure offset needs a log link") is removed with `objective` | Under G3 a Gaussian estimator minimizes `Σ E_i (y_i/E_i − f(x_i))² = Σ (y_i − E_i f(x_i))² / E_i`: weighted least squares of the count with mean `E_i f(x_i)` and variance proportional to `E_i`, the variance a count has. The mean structure is the one we want (count ∝ exposure) and the weights are the right ones, so it models the right thing; it is the Gaussian analogue of the offset, not an abuse of it. The one caveat is inherited from the Gaussian loss itself: a rate can come out negative, as it already could without an exposure. *(This answers the user's question; if they still prefer the restriction, keep an `objective: Literal["poisson","regression"]` declaration and the guard: a one-line addition to Step 2.)* |
| G5 | A regressor whose `fit` lacks `sample_weight` is **not** pre-checked | Python raises `TypeError: fit() got an unexpected keyword argument 'sample_weight'` itself. Rule: validate only what would pass silently |
| G6 | Fitted state of `DirectCohortModel`: `estimator_`, `use_exposure_: bool`, `feature_transformer_`. `regressor_` and `base_log_rate_` go | `estimator_` is sklearn's name for a fitted inner estimator; `use_exposure_` is what `TotalChildrenModel` already records, and `predict` follows it rather than the current setting |
| F1 | **The `feature_transformer` parameter is declared by each leaf model** (`DirectCohortModel`, `TotalChildrenModel`, `CohortProbabilityModel`), default `None` = "X is already the design matrix". **The base class owns the logic**: `_fit_features(X, y) -> (fitted transformer or None, design matrix)` and `_transform_features(X)`, plus the class-level annotation `feature_transformer: FeatureTransformer \| None` | `BaseEstimator.get_params` reads each concrete class's `__init__` signature, so a parameter the base "declares" would be invisible to `get_params`, `set_params` and `clone`; it must appear in every leaf `__init__`. The *behaviour* (clone, fit on the training rows, transform with the training statistics, None = identity) is shared, so it lives in `BaseAgeGroupModel`. Composites (`IndependentCohortModels`, `IndependentTotalProbabilityModel`) take none: their children do |
| F2 | `ModelPipeline` is deleted (`modeling/pipeline.py`, its tests, its export) | With F1 it is a second way to do the same thing |
| F3 | `CVHyperparameterEvaluator` drops its `feature_transformer` field; `build_feature_transformer_and_model` becomes `build_model`; each fold calls `model.fit(raw rows)` | Otherwise the package keeps two ways to transform. This is the tuning plan's open task "Evaluator on the raw table", done with the model instead of `ModelPipeline` |
| I1 | **`dict`, not `OrderedDict`** in `IndependentCohortModels` | Since Python 3.7 `dict` preserves insertion order by language guarantee; `OrderedDict` adds only order-sensitive `==` and `move_to_end`, neither used. The output column order is already fixed by contract: `cohort_models_` is built by iterating `y.columns`, so `predict` follows `y`'s order at fit regardless of the mapping's order (`test_columns_follow_y_and_rows_follow_X` pins this). The template is typed `Mapping`, which says "read-only, any order". No code change; one sentence added to the docstring (Step 4) |
| I2 | **Both `cohort_models` and `cohort_models_` are necessary** | sklearn's contract: `cohort_models` is the unfitted template stored verbatim, so `get_params`, `set_params` and `clone` work and a tuner can refit the same object; `cohort_models_` holds the fitted *copies*, so the template is never mutated, a refit equals a fresh fit, and `predict` follows the fitted state even after `set_params`. It is the same split as `estimator` / `estimator_` in `DirectCohortModel` and `feature_transformer` / `feature_transformer_` in `ModelPipeline` today. No code change |

## 2b. The derivation: the offset model as a weighted regression of the per-apartment rate

This section is written into `docs/DIRECT_COHORT_MODEL.md` §0.1 in Step 6, verbatim, and
its one-line summary goes into the `DirectCohortModel` docstring in Step 2. The symbols
follow §0.1's existing notation.

**Setup.** Building $b$ has $E_b > 0$ apartments (the exposure) and $y_b \in \{0, 1, 2, \dots\}$
children of one cohort, with features $x_b$. The quantity the model learns is the
**average number of children of the cohort per apartment**, the rate

$$\lambda_b = f(x_b) > 0, \qquad \text{so that the building's expected count is } \mu_b = E_b\,\lambda_b .$$

For a log-link (Poisson) model $f(x) = e^{F(x)}$, where $F$ is the raw score (the sum of
the trees in LightGBM, $\beta_0 + x\beta$ in a GLM).

**(A) The offset formulation (what LightGBM's `init_score` implements).**
$y_b \sim \text{Poisson}(\mu_b)$ with $\log \mu_b = \log E_b + F(x_b)$. Dropping the
$\log y_b!$ term, which does not depend on $F$, the negative log-likelihood is

$$\mathcal{L}_{\text{off}}(F) = \sum_b \Big[\mu_b - y_b \log \mu_b\Big]
= \sum_b \Big[E_b e^{F(x_b)} - y_b F(x_b)\Big] \;-\; \sum_b y_b \log E_b .$$

**(B) The weighted-rate formulation (what the generalized model does).**
Regress the observed rate $r_b = y_b / E_b$ on $x_b$ with the estimator's own Poisson
loss and `sample_weight` $w_b = E_b$. The Poisson loss of a non-negative real target $r$
with mean $\lambda$ is $\lambda - r \log \lambda$ (the Poisson deviance up to terms free
of $\lambda$; scikit-learn and LightGBM define it for any real $r \ge 0$, so a fractional
rate is a valid target). The weighted objective is

$$\mathcal{L}_{\text{rate}}(F) = \sum_b w_b \Big[\lambda_b - r_b \log \lambda_b\Big]
= \sum_b E_b \Big[e^{F(x_b)} - \frac{y_b}{E_b} F(x_b)\Big]
= \sum_b \Big[E_b e^{F(x_b)} - y_b F(x_b)\Big].$$

**(C) Equivalence.** Comparing the two,

$$\mathcal{L}_{\text{off}}(F) = \mathcal{L}_{\text{rate}}(F) - \sum_b y_b \log E_b ,$$

and the last term is a constant in $F$. The two objectives therefore have the same
minimizer, and more strongly the same per-row gradient and Hessian in the raw score,

$$\frac{\partial \mathcal{L}}{\partial F_b} = E_b e^{F_b} - y_b = \mu_b - y_b,
\qquad
\frac{\partial^2 \mathcal{L}}{\partial F_b^2} = E_b e^{F_b} = \mu_b ,$$

which is all a gradient-boosting step uses to build a tree. The starting point also
agrees: with weights, LightGBM's `boost_from_average` starts at the weighted mean rate
$\log\big(\sum_b w_b r_b / \sum_b w_b\big) = \log\big(\sum_b y_b / \sum_b E_b\big)$, exactly
the intercept $b = \log(\sum y / \sum E)$ the old code put into `init_score`. So the
trees are the same, and the prediction is

$$\hat\mu_b = E_b \,\hat\lambda_b = E_b \cdot \texttt{estimator\_.predict}(x_b),$$

which is why `predict` multiplies by the exposure instead of adding $\log E_b$ to a raw
score. (Measured on LightGBM, 2000 rows, 50 trees: the two predictions agree to
$1.5 \times 10^{-8}$ relative; `min_child_samples` counts rows in both, so the only
difference is floating point.)

**(D) What a Gaussian estimator fits under the same transformation.** With squared
error and the same target and weights,

$$\sum_b w_b \big(r_b - f(x_b)\big)^2 = \sum_b E_b \Big(\frac{y_b}{E_b} - f(x_b)\Big)^2
= \sum_b \frac{\big(y_b - E_b f(x_b)\big)^2}{E_b},$$

weighted least squares of the **count** with mean $E_b f(x_b)$ and weight $1/E_b$,
i.e. the estimator for a model whose variance grows with the exposure,
$\operatorname{Var}(y_b) \propto E_b$, as a count's does. The mean structure is the
intended one (expected count proportional to apartments, $f$ the average children per
apartment), so `use_exposure=True` is meaningful for a Gaussian loss too (G4); the only
Gaussian-specific caveat is that $f(x_b)$ can be negative, as it already could without
an exposure. Without weights (plain regression of $r_b$) the mean structure would be the
same but each building would count equally, a variance $\propto E_b^2$ assumption.

**(E) Why not a residual.** Fitting $y_b - \mu^{(0)}_b$ (the count minus an initial
prediction) is only meaningful for a squared-error loss, where the residual is again a
Gaussian target; for a Poisson loss the residual is not a count and can be negative, so
the fallback would fit a different model than the LightGBM path. Decision G3 uses (B)
for every estimator instead.

## 3. How to work

The standing rules of `docs/MULTI_COHORT_MODELS_PLAN.md` §3 apply verbatim. In short:

1. One step at a time. Start a step in plan mode, re-verify this plan's facts for it,
   summarize the step in the chat, ask for approval, then implement.
2. Baseline `uv run pytest -m "not slow"` before a step's first edit.
3. Per step: implement → `uv run ruff check <files>` and `uv run ruff format <files>` →
   `uv run mypy` and `uv run mypy src/age_group_prediction/modeling <changed tests>` →
   changed tests with `-W error` → one mutation check per claimed behaviour →
   an independent review subagent → the non-slow suite → update this doc → stop.
4. At each stop: a file-by-file summary, check results, suggested commit commands.
   **The user commits.** No `Co-Authored-By`, no "Generated with" footer.
5. Code style as `modeling/direct_cohort.py`: keyword-only constructors that store
   arguments verbatim; validation in `fit`; fitted state assigned together after
   success; `predict` follows the fitted state; each test's name and comment state
   the mistake it catches; no test that only checks a library.
6. Probes: `PYTHONPATH=src uv run --group test python -c "..."`; quote globs in zsh.

## 4. Target code shape

```python
# modeling/base.py  (additions)
class BaseAgeGroupModel(BaseEstimator, ABC):
    # Declared here for typing only; the models that take one list it in __init__
    # (get_params reads the signature), store it verbatim, and fit a copy.
    feature_transformer: FeatureTransformer | None

    def _fit_features(
        self, X: pd.DataFrame, y: pd.Series | pd.DataFrame
    ) -> tuple[FeatureTransformer | None, pd.DataFrame]:
        """A fitted copy of ``feature_transformer`` and the design matrix; ``(None, X)`` without one.

        The caller assigns the copy to ``feature_transformer_`` with its other
        fitted state, once everything succeeded.
        """
        if self.feature_transformer is None:
            return None, X
        feature_transformer = clone(self.feature_transformer).fit(X, y)  # y as sklearn's Pipeline passes it
        return feature_transformer, feature_transformer.transform(X)

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """``X`` through the copy fitted at ``fit``, never refitted; ``X`` itself without one."""
        if self.feature_transformer_ is None:
            return X
        return self.feature_transformer_.transform(X)
```

```python
# modeling/direct_cohort.py
class Regressor(Protocol):
    """What ``estimator`` must do: scikit-learn's regressor API with ``sample_weight``."""
    def fit(self, X, y, sample_weight=None) -> Self: ...
    def predict(self, X) -> ArrayLike: ...

class DirectCohortModel(BaseAgeGroupModel):
    def __init__(self, *, estimator: Regressor | None = None, use_exposure: bool = False,
                 feature_transformer: FeatureTransformer | None = None) -> None: ...

    @staticmethod
    def default_estimator() -> LGBMRegressor:
        return LGBMRegressor(objective="poisson", n_jobs=1, deterministic=True,
                             force_col_wise=True, verbosity=-1, random_state=42)

    def fit(self, X, y: pd.Series, exposure=None) -> Self:
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure)
        feature_transformer, X_design = self._fit_features(X, y)
        estimator = clone(self.estimator) if self.estimator is not None else self.default_estimator()
        if exposure_values is None:
            estimator.fit(X_design, y)
        else:
            # The offset as a weighted rate: the same Poisson likelihood as
            # log(exposure), for any regressor that takes sample_weight (G3).
            estimator.fit(X_design, np.asarray(y, dtype=float) / exposure_values,
                          sample_weight=exposure_values)
        self.estimator_ = estimator
        self.use_exposure_: bool = exposure_values is not None
        self.feature_transformer_ = feature_transformer
        return self

    def predict(self, X, exposure=None) -> np.ndarray:
        check_is_fitted(self)
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure_)
        prediction = np.asarray(self.estimator_.predict(self._transform_features(X)), dtype=float)
        return prediction if exposure_values is None else prediction * exposure_values
```

`TotalChildrenModel` and `CohortProbabilityModel` gain the same `feature_transformer`
parameter and the two helper calls (`_fit_features` at the top of `fit`, assigning
`feature_transformer_` with the other fitted state; `_transform_features` at the top of
`predict` / `predict_logits`). `IndependentCohortModels` and
`IndependentTotalProbabilityModel` are unchanged in code.

```python
# hyperparameter_tuning/evaluator.py  (shape after Step 5)
evaluator = CVHyperparameterEvaluator(
    DirectCohortModel(estimator=DirectCohortModel.default_estimator(),
                      use_exposure=True, feature_transformer=tree),
    parameters,            # names such as "estimator__n_estimators"
    cv=..., metric=POISSON_DEVIANCE)
model = evaluator.build_model(params)          # was build_feature_transformer_and_model
model.fit(X_train, y_train, exposure=exposure_train)   # raw rows; the model transforms
```

## 5. Steps

Each step is a validation stop. Line numbers are as of 2026-10-05; refer to names.

### Step 0 — Branch, plan doc, baseline
- [x] `git switch -c feat/estimator-and-feature-transformer` from `feat/hyperparameter-tuning`.
- [x] Copy this file to `docs/DIRECT_COHORT_GENERALIZATION_PLAN.md`; add a row to the
  "Active Plans" table in `docs/README.md`.
- [x] Run `uv run pytest -m "not slow"`: **1221 passed, 1 skipped, 1 xfailed** (2026-10-05, on the new branch).
- [ ] First commit (the plan doc, the handoff `docs/DIRECT_COHORT_GENERALIZATION_HANDOFF.md`, the README row), push, and open a **draft PR** into `feat/hyperparameter-tuning` with the title and body of §7. Later steps update the PR body's checklist.
- **Done when:** the branch exists, the doc is in the repo, the baseline matches. Stop.

### Step 1 — Feature helpers in the base class
Files: `src/age_group_prediction/modeling/base.py`, `tests/unit/test_modeling_base.py`.
- [ ] Add the class-level annotation, `_fit_features` and `_transform_features` as in §4.
  Import `FeatureTransformer` from `..feature_engineering` and `clone` from `sklearn.base`
  (both already imported by `pipeline.py`, so no new dependency direction).
- [ ] Docstring of `BaseAgeGroupModel`: one paragraph on `feature_transformer`: a model
  that takes one lists it in `__init__`, fits a copy on the training rows inside `fit`,
  and transforms with that copy at predict; `None` means `X` is already the design matrix.
- [ ] Tests, with a small stand-in `_FeatureModel(BaseAgeGroupModel)` that has
  `__init__(self, *, feature_transformer=None)` and uses both helpers:
  - `test_without_a_transformer_the_helpers_return_x_itself` — catches a copy or
    conversion of `X` when there is nothing to do (`is X`).
  - `test_fit_features_fits_a_copy_and_leaves_the_template_unfitted` — catches fitting
    the caller's object in place (`check_is_fitted(template)` raises `NotFittedError`).
  - `test_transform_features_uses_the_statistics_learned_at_fit` — centred feature on
    new rows equals `rows − training mean`, catches a refit at predict. (Move the logic of
    `test_rows_are_transformed_with_the_training_statistics` from `test_modeling_pipeline.py`.)
- **Done when:** the three tests pass with `-W error`; mypy strict passes on `modeling`;
  nothing else changed. Stop.

### Step 2 — `DirectCohortModel` takes any regressor; the offset as a weighted rate
Files: `modeling/direct_cohort.py`, `modeling/__init__.py`, `tests/unit/test_modeling_direct_cohort.py`,
`tests/unit/test_modeling_contract.py` (example factory only), `tests/unit/test_hyperparameter_tuning_evaluator.py`
(parameter names only), `hyperparameter_tuning/evaluator.py` (docstring example only).
- [ ] Rewrite the class as in §4 (G1–G6). Remove `Objective` from the module and from
  `modeling/__init__.py` (`__all__` too); export `Regressor` instead.
- [ ] Module and class docstrings: *why* a weighted rate (G3; the one-line identity
  $\mathcal{L}_{\text{off}} = \mathcal{L}_{\text{rate}} - \sum y_b \log E_b$ from §2b (C),
  with a pointer to the doc section for the full derivation), that it holds for Gaussian
  too (G4, §2b (D)), and that the estimator's own
  objective and hyperparameters are set on the estimator (`estimator__…` names in a tuner).
  Note that LightGBM ignores `subsample` unless `subsample_freq ≥ 1` is set on the estimator
  (the model no longer derives it).
- [ ] Tests (replace the file's LightGBM-specific ones; keep the exposure-misuse ones):
  - Parametrize a module-level `ESTIMATORS` list with ids:
    `DirectCohortModel.default_estimator().set_params(n_estimators=20)`,
    `HistGradientBoostingRegressor(loss="poisson", max_iter=20, random_state=0)`,
    `PoissonRegressor()`, and the Gaussian `LGBMRegressor(objective="regression", n_estimators=20, n_jobs=1, verbosity=-1)`.
  - `test_doubling_the_exposure_doubles_the_prediction[estimator]` — catches the exposure
    lost at predict, for every estimator including the Gaussian one (G4).
  - `test_training_mean_prediction_matches_the_target_mean[estimator]` (Poisson ones, with
    and without exposure) — catches a wrong rate (e.g. `y * exposure`) or a missing weight.
  - `test_the_weighted_rate_equals_lightgbm_offset` — fit the default LightGBM through the
    model and, by hand, `LGBMRegressor(...).fit(X, y, init_score=log(E)+b)` with
    `exp(raw + log(E) + b)`; `np.testing.assert_allclose(rtol=1e-6)`. Catches a
    reformulation that is *not* the offset model (the probe gave 1.5e-8).
  - `test_predict_follows_how_the_model_was_fitted` — keep, now on `use_exposure_`.
  - `test_clone_and_set_params_change_only_the_copy` — keep; add
    `set_params(estimator__n_estimators=5)` and assert the template's estimator is untouched.
  - `test_a_regressor_without_sample_weight_surfaces_the_library_error` — a stand-in
    regressor whose `fit(X, y)` has no `sample_weight`, with `use_exposure=True`:
    `pytest.raises(TypeError, match="sample_weight")`. Documents G5.
  - `test_the_template_estimator_stays_unfitted` — catches fitting `self.estimator` in place.
  - `test_invalid_input_surfaces_an_error`: drop the `regression + use_exposure` case (G4)
    and the `not_an_objective` case; keep `y * 0` with exposure (LightGBM raises).
  - Delete `test_subsample_below_one_changes_the_model` (it now tests LightGBM, not us).
- [ ] `test_modeling_contract.py`: `_direct_cohort_example` unchanged in shape (`DirectCohortModel()`
  still default-constructible). `check_parameters_default_constructible` must pass with
  `estimator=None` — this is why G2 exists.
- [ ] `test_hyperparameter_tuning_evaluator.py` and `evaluator.py`'s docstring: wherever a
  `DirectCohortModel` is tuned on `n_estimators`, build it as
  `DirectCohortModel(estimator=DirectCohortModel.default_estimator(), use_exposure=True)`
  and name the parameter `estimator__n_estimators`. Minimal edits; Step 5 reworks this file.
- [ ] Mutation checks: (a) replace `sample_weight=exposure_values` by nothing → the
  mean-prediction and offset-equality tests fail; (b) drop `* exposure_values` in predict →
  the doubling test fails; (c) fit `self.estimator` instead of a clone → the template test fails.
- **Done when:** the non-slow suite passes; mypy strict passes; a probe shows
  `DirectCohortModel(estimator=HistGradientBoostingRegressor(loss="poisson"), use_exposure=True)`
  fitting and predicting on `test_modeling_direct_cohort.py`'s data. Stop.

### Step 3 — `feature_transformer` on the three leaf models
Files: `modeling/direct_cohort.py`, `modeling/total_children.py`, `modeling/cohort_probability.py`,
new `tests/unit/test_modeling_feature_transformer.py`.
- [ ] Add `feature_transformer: FeatureTransformer | None = None` (keyword-only, last) to each
  `__init__`, stored verbatim. In each `fit`: `feature_transformer, X_design = self._fit_features(X, y)`
  first, then the existing logic on `X_design`; assign `self.feature_transformer_ = feature_transformer`
  **together with the other fitted state, after success**. In `predict` (and
  `CohortProbabilityModel.predict_logits`): `X_design = self._transform_features(X)` first.
  `CohortProbabilityModel`'s `validate_data(self, X_design, ...)` then records the design
  matrix's columns, which is what it must compare at predict.
- [ ] Docstrings: replace "``X`` is the finished design matrix" with "``X`` is the raw table
  when ``feature_transformer`` is given, otherwise the finished design matrix", in all three.
- [ ] New test file, parametrized over factories for the three models (ids = class names),
  on a raw table with a `Center()` plan (as `test_modeling_pipeline.py` builds it), moving the
  `ModelPipeline` tests here:
  - `test_the_template_transformer_stays_unfitted` (from `test_the_templates_stay_unfitted`).
  - `test_rows_are_transformed_with_the_training_statistics` — fit on half, predict the
    other half; equals a hand-built `clone(transformer).fit(train).transform(test)` fed to the
    same model with `feature_transformer=None`. Catches a refit at predict.
  - `test_predict_follows_the_fitted_copy_after_set_params` — `set_params(feature_transformer=other)`
    after fit changes nothing until the next fit.
  - `test_predict_logits_transforms_the_table_like_predict` — `CohortProbabilityModel` only.
  - `test_a_model_without_a_transformer_takes_a_design_matrix` — `feature_transformer=None`
    fitted on `transformer.fit_transform(table)` equals the model with the transformer fitted
    on `table`. Catches the helper being skipped on one path.
- [ ] `test_modeling_contract.py`: nothing to add (defaults are `None`), but run it: the
  four sklearn checks and the refit test must pass for all three.
- [ ] Mutation checks: use `self.feature_transformer.fit` in place → template test fails;
  skip `_transform_features` in `predict_logits` → the logits test fails.
- **Done when:** the new file passes with `-W error`; suite and mypy pass. Stop.

### Step 4 — Remove `ModelPipeline`; docstrings of the composites
Files: delete `modeling/pipeline.py` and `tests/unit/test_modeling_pipeline.py`;
edit `modeling/__init__.py`, `modeling/independent_cohorts.py`, `modeling/independent_total_probability.py`,
`tests/unit/test_modeling_contract.py`, `tests/unit/test_modeling_independent_cohorts.py`,
`tests/unit/test_modeling_independent_total_probability.py`, `tests/unit/test_modeling_calibration.py`.
- [ ] Delete the module, its export in `__init__.py` and `__all__`, and its `EXAMPLES` entry
  and import in the contract test (`test_every_shipped_model_has_an_example` will otherwise fail).
- [ ] Replace every `ModelPipeline(features, Model(...))` in the tests with
  `Model(..., feature_transformer=features)`. Ensure `test_modeling_calibration.py`'s use
  (around line 225) and the `_RenamedShares` / `_SeriesTotal` stand-ins still fit.
- [ ] `independent_cohorts.py`: the comment "usually a ModelPipeline" → "each with its own
  `feature_transformer`"; docstring: "each has its own feature transformer, model and
  hyperparameters" stays true. Add the I1 sentence: "A plain `dict` suffices: the output
  order is `y`'s column order at fit, whatever the mapping's order."
  `independent_total_probability.py`: "Both are usually a ModelPipeline, each with its own
  features; their settings are reached by nested names (`cohort_probability_model__l2_penalty`)"
  — one level shorter now that there is no `model__`.
- [ ] Grep check: `grep -rn "ModelPipeline\|pipeline" src/age_group_prediction/modeling tests/unit/test_modeling_*`
  returns nothing (`src/student_simulator/pipeline.py` is a different package and stays).
- **Done when:** suite and mypy pass; the grep is empty. Stop.

### Step 5 — The evaluator fits the model on the raw rows
Files: `hyperparameter_tuning/evaluator.py`, `tests/unit/test_hyperparameter_tuning_evaluator.py`,
`docs/HYPERPARAMETER_TUNING_PLAN.md` (§5 example, D13, the §6 task).
- [ ] Remove the `feature_transformer` field and its `InstanceOf` import if unused; rename
  `build_feature_transformer_and_model(params) -> (FeatureTransformer, BaseAgeGroupModel)`
  to `build_model(params) -> BaseAgeGroupModel`. In `evaluate`, delete the per-fold
  `feature_transformer.fit/transform` block; call `model.fit(X_train, y_train, exposure=exposure_train)`
  and `model.predict(X_val, exposure=exposure_val)` on the raw rows.
- [ ] Docstrings: "`X` is the raw table; a numpy array works only with a model whose
  `feature_transformer` is `None`"; the class example as in §4 (`estimator=…`,
  `feature_transformer=tree`, `estimator__…` names). Module docstring of `parameters.py`
  line 3: the example name becomes `"estimator__learning_rate"`.
- [ ] Tests: drop the `feature_transformer` construction/validation cases (lines ~151, 184,
  214–222); rename the two `build_feature_transformer_and_model` tests to `build_model`;
  `_RecordingTransformer` now goes in as `DirectCohortModel(feature_transformer=_RecordingTransformer(), …)`
  and the test asserts it saw only training rows per fold; the "templates stay unfitted"
  test checks `evaluator.model` and `evaluator.model.feature_transformer`. Add
  `test_the_model_transforms_each_fold_on_its_training_rows` if `_RecordingTransformer`
  does not already prove it.
- [ ] `HYPERPARAMETER_TUNING_PLAN.md`: §5 example rewritten (every parameter name prefixed
  `estimator__`; `subsample_freq=1` set on the template estimator since the model no longer
  derives it; `exposure = ExposureTransformer("n_apartments").fit_transform(df)` before the
  split and `take_rows(exposure, train_index)`, as the §6 task required; refit through
  `build_model`); D13 marked superseded by this plan; the §6 task ticked with a pointer here;
  3.4's "refit through `build_feature_transformer_and_model`" → `build_model`.
- **Done when:** the evaluator tests pass with `-W error`; the suite passes; the §5 code
  block runs (rule: run every code block you put in a doc). Stop.

### Step 6 — Documentation
Files: `docs/DIRECT_COHORT_MODEL.md` §0.1–§0.6, `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0,
`docs/MODULE_REFERENCE.md`, `docs/FEATURE_TRANSFORMATIONS.md` (ModelPipeline mentions),
`docs/MULTI_COHORT_MODELS_PLAN.md` (a dated note that `ModelPipeline` was folded into the
models by this plan; N20 pointer), `docs/MODEL_REIMPLEMENTATION_PLAN.md` (same note at §2),
`docs/README.md`.
- [ ] `DIRECT_COHORT_MODEL.md` §0.1: copy §2b of this plan **in full** (setup, (A)–(E),
  LaTeX as written) as a new subsection "The exposure as a weighted regression of the
  per-apartment rate", replacing the current `init_score` description; keep §0.1's symbols
  consistent with it ($E_b$, $\lambda_b$, $\mu_b$, $F$). §0.5 (evidence) gets the measured
  $1.5 \times 10^{-8}$ agreement and the test that pins it. §0.2 API table: the new constructor,
  `default_estimator()`, `estimator_`, `use_exposure_`, `feature_transformer_`; the code
  example passes `feature_transformer=tree` and the raw rows. §0.3 errors: remove the
  log-link error, add the `TypeError` of G5. §0.4: note the generalization. §0.6: the title
  and table lose `ModelPipeline`; the example builds
  `DirectCohortModel(use_exposure=True, feature_transformer=tree)` per cohort; the rules table:
  the nested-names rule becomes `estimator__learning_rate`, and the I1/I2 answers get a row.
- [ ] `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0: sub-models carry their own
  `feature_transformer`; nested names drop `model__`.
- [ ] `MODULE_REFERENCE.md`: delete the `modeling/pipeline.py` row; update `direct_cohort.py`
  ("any scikit-learn-style regressor with a Poisson or Gaussian loss; the exposure as a
  weighted rate; optional `feature_transformer`"), `independent_cohorts.py`, `total_children.py`,
  `cohort_probability.py`, `base.py` (the helpers) and `hyperparameter_tuning/evaluator.py` rows.
- [ ] Run every code block changed in the docs.
- **Done when:** `grep -rn "ModelPipeline" docs` hits only historical notes that say it was
  removed; all changed blocks ran. Stop.

### Step 7 — Final check and PR
- [ ] Full routine on the whole diff: ruff, mypy, `uv run pytest -m "not slow"`, then the
  slow tests touching `modeling` or `hyperparameter_tuning` if any.
- [ ] `git status` shows only `modeling/`, `hyperparameter_tuning/`, their tests and `docs/`.
- [ ] Draft PR body (user applies it): the two generalizations, the decisions table, the
  evaluator change, the removed classes/parameters (`ModelPipeline`, `Objective`, the ten
  LightGBM arguments, `regressor_`, `base_log_rate_`), and the migration line
  `DirectCohortModel(n_estimators=…)` → `DirectCohortModel(estimator=LGBMRegressor(…))`.
- [ ] Update this doc's status; update the memory files for the tuning plan (its §6 task is
  done here) and the multi-cohort plan.

## 6. Verification

End to end, after Step 6, run as a probe (and keep as the §0.6 doc example):

```python
# PYTHONPATH=src uv run --group test python - <<'EOF'
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from age_group_prediction.feature_engineering import Center, ColumnPlan, FeatureTransformer
from age_group_prediction.modeling import DirectCohortModel, IndependentCohortModels
from age_group_prediction.scoring import POISSON_DEVIANCE

rng = np.random.default_rng(0); n = 400
table = pd.DataFrame({"x": rng.normal(size=n), "n_apartments": rng.integers(5, 40, n).astype(float)})
table["n_kindergarten"] = rng.poisson(0.1 * table["n_apartments"] * np.exp(0.5 * table["x"]))
table["n_elementary"] = rng.poisson(np.exp(1 - 0.5 * table["x"]))
tree = FeatureTransformer((ColumnPlan(name="x", columns=("x",), transforms=(Center(),)),))
y = table[["n_kindergarten", "n_elementary"]]; exposure = table["n_apartments"].to_numpy()

model = IndependentCohortModels({
    "n_kindergarten": DirectCohortModel(use_exposure=True, feature_transformer=tree),           # default LightGBM
    "n_elementary": DirectCohortModel(estimator=HistGradientBoostingRegressor(loss="poisson"),
                                      feature_transformer=tree),                              # sklearn, no offset
}).fit(table, y, exposure=exposure)
pred = model.predict(table, exposure=exposure)
assert list(pred.columns) == list(y.columns) and pred.index.equals(table.index)
print({c: round(model.evaluate(y[c], pred[c], POISSON_DEVIANCE), 3) for c in y})
EOF
```

Checks per step are in the steps; the whole-PR checks are the §3 routine over the final
diff, the contract test discovering exactly the five remaining models, and the hyperparameter
tuning §5 example running unchanged from the doc.

## 7. Draft PR title and body

Opened after Step 0's first commit; ticked as steps land. No "Generated with" footer.

**Title:** `Any regressor in DirectCohortModel; the feature transformer inside each model`

**Body:**

```markdown
Into `feat/hyperparameter-tuning`, where `modeling/` lives. Plan and decisions:
`docs/DIRECT_COHORT_GENERALIZATION_PLAN.md`.

## What changes

1. **`DirectCohortModel(*, estimator=None, use_exposure=False, feature_transformer=None)`.**
   Any scikit-learn-style regressor with a Poisson or Gaussian loss (`estimator=None` builds
   the configured LightGBM, also available as `DirectCohortModel.default_estimator()`). The
   ten mirrored LightGBM hyperparameters and `objective` are removed; a tuner reaches the
   estimator's own by nested names (`estimator__n_estimators`).
2. **The exposure as a weighted regression of the per-apartment rate**:
   `fit(X, y / exposure, sample_weight=exposure)` and `predict(X) * exposure`. For a Poisson
   loss this is the same likelihood as the `log(exposure)` offset (derivation in
   `docs/DIRECT_COHORT_MODEL.md` §0.1; LightGBM agrees with its own `init_score` path to
   1.5e-8), and it works for every regressor that takes `sample_weight`. With a Gaussian
   loss it is weighted least squares of the count with variance ∝ exposure, so
   `use_exposure=True` is allowed there too.
3. **`feature_transformer` is a parameter of each leaf model** (`DirectCohortModel`,
   `TotalChildrenModel`, `CohortProbabilityModel`), with the fit/transform helpers in
   `BaseAgeGroupModel`. **`ModelPipeline` is removed.** `IndependentCohortModels` and
   `IndependentTotalProbabilityModel` are unchanged in code.
4. **`CVHyperparameterEvaluator` fits the model on the raw rows**: its `feature_transformer`
   field goes, `build_feature_transformer_and_model` becomes `build_model`. This closes the
   tuning plan's "Evaluator on the raw table" task.

## Removed or renamed

`ModelPipeline`, `Objective`, the ten LightGBM constructor arguments of `DirectCohortModel`,
its `regressor_` and `base_log_rate_` (now `estimator_`, `use_exposure_`,
`feature_transformer_`), `CVHyperparameterEvaluator.feature_transformer`.
Migration: `DirectCohortModel(n_estimators=100, ...)` →
`DirectCohortModel(estimator=LGBMRegressor(objective="poisson", n_estimators=100, ...))`;
`ModelPipeline(tree, model)` → `model.set_params(feature_transformer=tree)`.

## Steps

- [x] 0 Branch, plan doc, baseline (1221 passed)
- [ ] 1 Feature helpers in `BaseAgeGroupModel`
- [ ] 2 `DirectCohortModel`: any regressor, weighted-rate exposure
- [ ] 3 `feature_transformer` on the three leaf models
- [ ] 4 Remove `ModelPipeline`
- [ ] 5 Evaluator on the raw rows
- [ ] 6 Docs (including the derivation)
- [ ] 7 Final checks

## Checks

Per step: ruff, mypy strict on `modeling`, changed tests with `-W error`, one mutation check
per claimed behaviour, independent review, `uv run pytest -m "not slow"`.
```
