# Plan: Any regressor in `DirectCohortModel`, and the feature transformer inside each model

**Branch:** `feat/estimator-and-feature-transformer`, from `feat/hyperparameter-tuning` (where `modeling/` lives; PRs #6–#11 used the same base). PR into `feat/hyperparameter-tuning`.
**Status (2026-10-05):** **All steps done** (draft PR #12; Steps 0–6 committed, Step 7's doc updates await the user's commit). The PR awaits review. Follow-up, by the user's decision a separate PR: the mixed fitted state after a failed refit in `TotalChildrenModel` and `CohortProbabilityModel` (Step 3's open finding). Handoff for the implementing session: `DIRECT_COHORT_GENERALIZATION_HANDOFF.md`. Baseline `uv run pytest -m "not slow"`: **1221 passed, 1 skipped, 1 xfailed**; after Step 1: **1224 passed**; after Step 2: **1232 passed** (1233 before the revision removed one test). After Step 3: **1239 passed**. After Step 4: **1232 passed** (the 5 `ModelPipeline` tests and its 5 contract cases removed, 3 added). After Step 5: **1230 passed** (2 evaluator tests removed).
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
| G1 | `DirectCohortModel(*, estimator, use_exposure=False, feature_transformer=None)`, `estimator` required. The ten LightGBM hyperparameters and `objective` are removed; they live on the estimator | The estimator carries its own loss and hyperparameters; mirroring them would have to be redone for every library. Tuning reaches them by nested names (`estimator__n_estimators`), which `BaseEstimator.set_params` already supports (verified) |
| G1a | **An estimator instance, not `**kwargs` or a parameter dict** | scikit-learn's meta-estimator practice (`BaggingRegressor(estimator=…)`, `TransformedTargetRegressor(regressor=…)`). Probed 2026-10-05: `**kwargs` are omitted by `get_params` and **dropped by `clone`**, so every trial would run on defaults; a dict cannot be reached by `set_params(estimator_params__n_estimators=…)` (`AttributeError`); an instance exposes `estimator__n_estimators` in `get_params(deep=True)`, and the evaluator's `clone(model).set_params(**params)` changes only the trial copy. Tuning needs only the nested names |
| G1b | **`estimator` is typed by a `Protocol` `Regressor`** (`fit(X, y, sample_weight=None)`, `predict(X)`), exported from `modeling` | No library type states this contract: scikit-learn's `RegressorMixin`, which LightGBM, `HistGradientBoostingRegressor` and `PoissonRegressor` subclass, declares only `score` (no `fit`, `predict` or `sample_weight`), and with no stubs mypy reads it as `Any`. The Protocol states what the model calls; any library meets it structurally, without inheriting |
| G2 | **`estimator` is required, with no default** (revised 2026-10-05 at the user's request; first planned as `estimator=None` plus a `default_estimator()` factory). The caller always builds the estimator, e.g. `LGBMRegressor(objective="poisson", n_jobs=1)` | The user always sets it explicitly, and a tuner needs the instance anyway to reach `estimator__…`. The first plan's reason was wrong: `check_parameters_default_constructible` only constrains defaults that exist (`None`, scalars, tuples, types, callables) and accepts a required argument (probed; `IndependentCohortModels(cohort_models)` already has one, and the contract test builds each model from an example instance). Settings the factory carried now belong to whoever builds the estimator: `n_jobs=1`, since more OpenMP threads crash alongside torch on macOS |
| G3 | **The exposure enters as a weighted rate, for every estimator:** `fit(X, y / exposure, sample_weight=exposure)`; `predict` returns `estimator_.predict(X) * exposure` | For a Poisson loss this is the *same* likelihood as the offset `log(exposure)`: `Σ E_i (μ_i − r_i log μ_i)` with `r_i = y_i/E_i` has the same gradient and hessian per row as `Σ (E_i μ_i − y_i log(E_i μ_i))`. Verified on LightGBM: weighted-rate vs `init_score` predictions agree to 1.5e-8 relative. It needs only `sample_weight`, which LightGBM, `HistGradientBoostingRegressor`, `PoissonRegressor` and most regressors accept; `init_score` exists only in LightGBM. One code path, no `raw_score`, no stored intercept. The user chose this over an `init_score` branch |
| G4 | **`use_exposure` is allowed with a Gaussian loss.** The old guard ("an exposure offset needs a log link") is removed with `objective` | Under G3 a Gaussian estimator minimizes `Σ E_i (y_i/E_i − f(x_i))² = Σ (y_i − E_i f(x_i))² / E_i`: weighted least squares of the count with mean `E_i f(x_i)` and variance proportional to `E_i`, the variance a count has. The mean structure is the one we want (count ∝ exposure) and the weights are the right ones, so it models the right thing; it is the Gaussian analogue of the offset, not an abuse of it. The one caveat is inherited from the Gaussian loss itself: a rate can come out negative, as it already could without an exposure. Full derivation in §2b (D), pinned by `test_the_gaussian_weighted_rate_is_least_squares_of_the_count`. A guard could not be reliable anyway: the model cannot read an arbitrary estimator's loss (`objective` in LightGBM, `loss` in HistGradientBoosting, the class for `PoissonRegressor`), so it would need a second `objective=` declaration that can disagree silently. The user agreed on 2026-10-05 |
| G5 | A regressor whose `fit` lacks `sample_weight` is **not** pre-checked | The estimator fails with its own error: Python's `TypeError: fit() got an unexpected keyword argument 'sample_weight'` for a plain `fit(X, y)`, `ValueError` for a scikit-learn `Pipeline`. Rule: validate only what would pass silently |
| G6 | Fitted state of `DirectCohortModel`: `estimator_`, `use_exposure_: bool`, `feature_transformer_`. `regressor_` and `base_log_rate_` go | `estimator_` is sklearn's name for a fitted inner estimator; `use_exposure_` is what `TotalChildrenModel` already records, and `predict` follows it rather than the current setting. Both are needed: `estimator_` holds the fitted copy, so the template stays unfitted across folds and trials; without `use_exposure_`, `set_params(use_exposure=False)` after a fit would make `predict` return rates instead of counts, silently, and nothing in the fitted estimator records the choice |
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
the intercept $F_0 = \log(\sum y / \sum E)$ the old code put into `init_score`. So the
trees are the same, and the prediction is

$$\hat\mu_b = E_b \,\hat\lambda_b = E_b \cdot \texttt{estimator\_.predict}(x_b),$$

which is why `predict` multiplies by the exposure instead of adding $\log E_b$ to a raw
score. The equivalence is of the likelihoods. An estimator that also penalizes and
*normalizes* `sample_weight` scales its penalty differently: scikit-learn's
`PoissonRegressor(alpha=…)` divides the weighted loss by $\sum_b w_b = \sum_b E_b$, so its
`alpha` acts as `alpha` $\times \bar E$ in the offset model (measured 2026-10-05: with
`alpha=1` the weighted-rate coefficients match the offset model's at `alpha` $= \bar E = 47.6$
to $10^{-4}$, and at `alpha=0` the two agree). LightGBM and `HistGradientBoostingRegressor`
use the weights as given, so their penalties are unchanged. No single rescaling of the
weights fixes both: dividing by $\bar E$ would change LightGBM's hessians and so its
`reg_lambda` and `min_child_weight`. (Measured on LightGBM, 2000 rows, 50 trees: the two predictions agree to
$1.5 \times 10^{-8}$ relative; `min_child_samples` counts rows in both, so the only
difference is floating point.)

**(D) The same transformation under a Gaussian likelihood.** The derivation runs as for
Poisson, with the squared-error loss in place of the Poisson one; $f$ now enters the
mean directly (identity link), and $f_b$ is short for $f(x_b)$.

*(D1) The model.* The building's count is Gaussian with a mean proportional to the
exposure and a variance that grows with it, as a count's does:

$$y_b \sim \mathcal{N}\big(\mu_b,\ \sigma^2 E_b\big), \qquad \mu_b = E_b\, f_b .$$

Its negative log-likelihood is

$$\mathcal{L}_{\text{Gauss}}(f, \sigma^2) = \sum_b \Big[\frac{(y_b - E_b f_b)^2}{2\sigma^2 E_b}
+ \tfrac12 \log\big(2\pi\sigma^2 E_b\big)\Big].$$

*(D2) The weighted-rate formulation.* Regress the rate $r_b = y_b / E_b$ on $x_b$ with
the estimator's squared-error loss $\tfrac12 (r - f)^2$ and `sample_weight` $w_b = E_b$:

$$\mathcal{L}_{\text{rate}}(f) = \tfrac12 \sum_b w_b \big(r_b - f_b\big)^2
= \tfrac12 \sum_b E_b \Big(\frac{y_b}{E_b} - f_b\Big)^2
= \tfrac12 \sum_b \frac{\big(y_b - E_b f_b\big)^2}{E_b}.$$

*(D3) Equivalence.* Comparing the two,

$$\mathcal{L}_{\text{Gauss}}(f, \sigma^2) = \frac{1}{\sigma^2}\,\mathcal{L}_{\text{rate}}(f)
+ \tfrac12 \sum_b \log\big(2\pi\sigma^2 E_b\big),$$

and the last term does not depend on $f$. For every $\sigma^2$ the two objectives
therefore have the same minimizer in $f$, so the prediction never needs $\sigma^2$. Per
row, in $f_b$,

$$\frac{\partial \mathcal{L}_{\text{rate}}}{\partial f_b} = E_b f_b - y_b = \mu_b - y_b,
\qquad
\frac{\partial^2 \mathcal{L}_{\text{rate}}}{\partial f_b^2} = E_b :$$

the gradient has the Poisson form of (C), and the Hessian is the exposure instead of
$\mu_b$. The starting point agrees too: a constant $f$ minimizes
$\tfrac12\sum_b (y_b - E_b f)^2 / E_b$ at $f = \sum_b y_b / \sum_b E_b$, the weighted
mean rate a booster starts from. The prediction is again
$\hat\mu_b = E_b \cdot \texttt{estimator\_.predict}(x_b)$. (Checked numerically: the
weighted-rate fit equals the weighted least-squares fit of $y$ on $[E, E x]$ with
weights $1/E$ to $10^{-16}$, and the full maximum-likelihood fit of (D1), $\sigma^2$
included, to $10^{-8}$; pinned by
`test_the_gaussian_weighted_rate_is_least_squares_of_the_count`.)

*(D4) Why not the additive offset, and why the weight.* With an identity link, the
additive offset of (A) gives the mean $f(x_b) + \log E_b$, which is not proportional to
the apartments; that is why the old model rejected `use_exposure` with a Gaussian loss.
The weighted rate scales the mean instead, so the mean structure is the intended one
(expected count proportional to apartments, $f$ the average children per apartment) and
`use_exposure=True` is meaningful for a Gaussian loss too. Without the weight (plain
regression of $r_b$) the mean would be the same, but each building would count equally,
a $\operatorname{Var}(y_b) \propto E_b^2$ assumption. The one Gaussian-specific caveat
is that $f(x_b)$ can be negative, as it already could without an exposure.

**(E) Why not a residual.** Fitting $y_b - \mu^{(0)}_b$ (the count minus an initial
prediction) is only meaningful for a squared-error loss, where the residual is again a
Gaussian target; for a Poisson loss the residual is not a count and can be negative, so
a residual-based path would fit a different model than the LightGBM one. The model uses
(B) for every estimator instead.

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
7. **Comments and docstrings keep only what is relevant and important** (the user, 2026-10-05):
   what the code does and the non-obvious constraint (what would otherwise pass silently, what
   the caller must do). Derivations, examples and motivation or intuition go in the docs, with a
   pointer at most.

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
    def __init__(self, *, estimator: Regressor, use_exposure: bool = False,
                 feature_transformer: FeatureTransformer | None = None) -> None: ...

    def fit(self, X, y: pd.Series, exposure=None) -> Self:
        exposure_values = self._check_exposure(X, exposure, expected=self.use_exposure)
        feature_transformer, X_design = self._fit_features(X, y)
        estimator: Regressor = clone(self.estimator)
        if exposure_values is None:
            estimator.fit(X_design, y)
        else:
            # The offset as a weighted rate: the same Poisson likelihood as
            # log(exposure), for any regressor that takes sample_weight (G3).
            estimator.fit(X_design, y / exposure_values, sample_weight=exposure_values)
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
    DirectCohortModel(estimator=LGBMRegressor(objective="poisson", n_jobs=1),
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
- [x] First commit (the plan doc, the handoff `docs/DIRECT_COHORT_GENERALIZATION_HANDOFF.md`, the README row), push, and open a **draft PR** into `feat/hyperparameter-tuning` with the title and body of §7. Later steps update the PR body's checklist.
- **Done when:** the branch exists, the doc is in the repo, the baseline matches. Stop.

### Step 1 — Feature helpers in the base class
Files: `src/age_group_prediction/modeling/base.py`, `tests/unit/test_modeling_base.py`.
- [x] Add the class-level annotation, `_fit_features` and `_transform_features` as in §4.
  Also annotated `feature_transformer_: FeatureTransformer | None`, which mypy strict needs for
  `_transform_features`; with no value it creates no attribute, so `get_params` and
  `check_is_fitted` are unaffected (probed).
  Import `FeatureTransformer` from `..feature_engineering` and `clone` from `sklearn.base`
  (both already imported by `pipeline.py`, so no new dependency direction).
- [x] Docstring of `BaseAgeGroupModel`: one paragraph on `feature_transformer`: a model
  that takes one lists it in `__init__`, fits a copy on the training rows inside `fit`,
  and transforms with that copy at predict; `None` means `X` is already the design matrix.
- [x] Tests, with a small stand-in `_FeatureModel(BaseAgeGroupModel)` that has
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
- **Result (2026-10-05):** ruff and mypy clean; tests pass with `-W error`; four mutations
  (no clone, refit at transform, `X.copy()` without a transformer, untransformed design
  matrix at fit) each fail their test; review: three low notes applied (comment that only
  leaf models set the attributes, `y` comment reworded since `FeatureTransformer` ignores
  it, test comment covers the design-matrix check); non-slow suite 1224 passed, 1 skipped,
  1 xfailed.

### Step 2 — `DirectCohortModel` takes any regressor; the offset as a weighted rate
Files: `modeling/direct_cohort.py`, `modeling/__init__.py`, `tests/unit/test_modeling_direct_cohort.py`,
`tests/unit/test_modeling_contract.py` (example factory only), `tests/unit/test_hyperparameter_tuning_evaluator.py`
(parameter names only), `hyperparameter_tuning/evaluator.py` (docstring example only).
- [x] Rewrite the class as in §4 (G1–G6). Remove `Objective` from the module and from
  `modeling/__init__.py` (`__all__` too); export `Regressor` instead.
- [x] Module and class docstrings: *why* a weighted rate (G3; the one-line identity
  $\mathcal{L}_{\text{off}} = \mathcal{L}_{\text{rate}} - \sum y_b \log E_b$ from §2b (C),
  with a pointer to the doc section for the full derivation), that it holds for Gaussian
  too (G4, §2b (D)), and that the estimator's own
  objective and hyperparameters are set on the estimator (`estimator__…` names in a tuner).
  Note that LightGBM ignores `subsample` unless `subsample_freq ≥ 1` is set on the estimator
  (the model no longer derives it).
- [x] Tests (replace the file's LightGBM-specific ones; keep the exposure-misuse ones):
  - Parametrize a module-level `ESTIMATORS` list with ids:
    the configured `LGBMRegressor(objective="poisson", n_estimators=20, n_jobs=1, …)`,
    `HistGradientBoostingRegressor(loss="poisson", max_iter=20, random_state=0)`,
    `PoissonRegressor(alpha=0.0)` (a separate `POISSON_ESTIMATORS` for the Poisson-only tests), and the Gaussian `LGBMRegressor(objective="regression", n_estimators=20, n_jobs=1, verbosity=-1)`.
  - `test_doubling_the_exposure_doubles_the_prediction[estimator]` — catches the exposure
    lost at predict, for every estimator including the Gaussian one (G4).
  - `test_training_mean_prediction_matches_the_target_mean[estimator]` (Poisson ones, with
    and without exposure) — catches a prediction on the wrong scale (e.g. `y * exposure`); a
    missing weight is caught by the recording test, see mutation (a).
  - `test_the_weighted_rate_equals_lightgbms_offset` — fit the configured LightGBM through the
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
  - Added during the step: `test_fit_receives_the_rate_with_the_exposure_as_weight` (a
    recording stand-in pins `fit`'s target and weight exactly, for any estimator) and
    `test_the_gaussian_weighted_rate_is_least_squares_of_the_count` (§2b (D)). The single
    remaining invalid-input case became `test_an_all_zero_target_surfaces_lightgbms_error`,
    without the divide-by-zero warning filter (our `log(sum y / sum E)` is gone).
- [x] `test_modeling_contract.py`: every example passes an explicit
  `LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1)`; the four sklearn checks pass
  with `estimator` required (G2, revised).
- [x] `test_hyperparameter_tuning_evaluator.py` and `evaluator.py`'s docstring: wherever a
  `DirectCohortModel` is tuned on `n_estimators`, build it as
  `DirectCohortModel(estimator=LGBMRegressor(objective="poisson", n_jobs=1, …), use_exposure=True)`
  and name the parameter `estimator__n_estimators`. The same explicit estimator goes into
  `test_modeling_independent_cohorts.py` and `test_modeling_pipeline.py`. Minimal edits; Step 5 reworks this file.
- [x] Mutation checks: (a) drop `sample_weight=exposure_values` → the recording test, the
  offset-equality test and the Gaussian least-squares test fail (the mean test alone
  misses it for HistGradientBoosting: 0.48% at rel=0.01, hence the recording test);
  (b) drop `* exposure_values` in predict → the doubling test fails; (c) fit `self.estimator`
  instead of a clone → the template test fails; (d) predict on `self.use_exposure` → the
  follows-fit test fails; (e) target `y * exposure` → the recording and mean tests fail.
- **Done when:** the non-slow suite passes; mypy strict passes; a probe shows
  `DirectCohortModel(estimator=HistGradientBoostingRegressor(loss="poisson"), use_exposure=True)`
  fitting and predicting on `test_modeling_direct_cohort.py`'s data. Stop.
- **Result (2026-10-05):** ruff and mypy clean; the changed tests pass with `-W error`;
  the five mutations above fail their tests; the HistGradientBoosting probe predicts a
  training mean of 6.677 against 6.677. Review findings, each reproduced: a penalized
  scikit-learn GLM normalizes `sample_weight`, so its `alpha` acts as `alpha * mean(E)` in
  the offset model (documented in the docstring and §2b; the test GLM is now
  `PoissonRegressor(alpha=0.0)`, which at `alpha=1` had shrunk `ses` to 0.05 from 0.4); a
  one-column `y` broadcast silently into an (n, n) target with `LinearRegression`, so
  `fit` now rejects a `y` that is not one-dimensional (`test_a_two_dimensional_target_raises`,
  mutation-checked); docstrings no longer claim `TypeError` or that every estimator rejects
  an all-zero `y`. Rejected: an unclear error for `estimator__…` on `estimator=None` (it is
  scikit-learn's own). Non-slow suite after the fixes: 1233 passed, 1 skipped, 1 xfailed.
- **Revision (2026-10-05, the user's review):** `estimator` is required and
  `default_estimator()` is gone (G2 revised; G1b, G6 justified). `y` is divided as given
  (`Series / ndarray` divides by position and keeps `y`'s index), so the numpy conversion
  and the `np.ndim(y)` check it needed are removed with `test_a_two_dimensional_target_raises`:
  without the conversion pandas itself raises for a one-column frame with an exposure
  (`Unable to coerce to Series`), and a frame `y` otherwise contradicts the declared
  `pd.Series`. Mutations (a)–(e) re-run, each fails its tests. Review: no correctness
  findings (positional division confirmed with mismatched indexes; the short
  `LGBMRegressor(n_jobs=1)` refits bit-identically); stale plan lines and the `fit` comment
  (Poisson-only wording) fixed. Non-slow suite: 1232 passed, 1 skipped, 1 xfailed.

### Step 3 — `feature_transformer` on the three leaf models
Files: `modeling/direct_cohort.py`, `modeling/total_children.py`, `modeling/cohort_probability.py`,
new `tests/unit/test_modeling_feature_transformer.py`.
- [x] Add `feature_transformer: FeatureTransformer | None = None` (keyword-only, last) to each
  `__init__`, stored verbatim. In each `fit`, after the configuration checks:
  `feature_transformer, X = self._fit_features(X, y)` (the user's choice: `X` is rebound to
  the design matrix, no separate name), then the existing logic on it, including
  `check_X_y` and `validate_data(self, X, reset=True)` in `TotalChildrenModel` and
  `CohortProbabilityModel`; `self.feature_transformer_ = feature_transformer` is assigned with
  the other fitted state, after success. At predict, `_transform_features(X)` feeds
  `DirectCohortModel.predict`, `TotalChildrenModel.predict` and
  `CohortProbabilityModel.predict_logits`. **`CohortProbabilityModel.predict` does not
  transform**: it goes through `predict_logits`, and a second transform would pass silently,
  since the design keeps the raw column names (probed: `['x']`).
- [x] Docstrings: "``X`` is the raw table when ``feature_transformer`` is given, otherwise the
  finished design matrix", in all three.
- [x] New test file, parametrized over the three models (ids = class names). The raw table
  has columns the transformer drops (`z`, `n_apartments`) and `x` with mean 2: on a centred
  `x` alone a skipped transformer is invisible (probed: trees ignore a shift, an intercept
  absorbs it).
  - `test_rows_are_transformed_with_the_training_statistics[model]` — fit on half, predict
    the other half; equals the same model with `feature_transformer=None` on a hand-built
    design matrix. Catches the transformer skipped at fit, at predict or both, refitted at
    predict, or applied twice.
  - `test_the_template_transformer_stays_unfitted[model]`.
  - `test_predict_logits_transforms_the_table_like_predict` — `CohortProbabilityModel` only.
  - Dropped from the original list: the `set_params` test (the template at predict is
    unfitted, so a misuse raises `NotFittedError`, loudly) and
    `test_a_model_without_a_transformer_takes_a_design_matrix` (the first test compares with
    exactly that model, on held-out rows).
- [x] `test_modeling_contract.py`: unchanged; the four sklearn checks and the refit test pass
  for all three.
- [x] Mutation checks, each fails its test: (a) the base helper fits the template → template
  test ×3; (b) each model fits on raw `X` → statistics test; (c) Direct/Total skip the
  transform at predict → statistics test; (d) `CohortProbabilityModel.predict` transforms too
  → statistics test (double shift); (e) `predict_logits` skips it → logits test. Total and
  Cohort mistakes also trip their own feature-name check (loud); Direct's would be silent.
- **Done when:** the new file passes with `-W error`; suite and mypy pass. Stop.
- **Result (2026-10-05):** ruff and mypy clean; 115 tests (new file, the three models' files,
  contract) pass with `-W error`; non-slow suite 1239 passed, 1 skipped, 1 xfailed.
  Review: no correctness findings. Applied: `TotalChildrenModel.fit` checks the exposure
  before fitting the transformer (as `DirectCohortModel` does); the logits test's comment
  now names the silent mistake it catches (a double transform in `predict_logits`, verified
  by mutation). Open, pre-existing since PR #11: in `TotalChildrenModel` and
  `CohortProbabilityModel` a refit whose `validate_data(reset=True)` raises (e.g. a
  non-string column name) leaves the new `coef_` with the old `feature_names_in_` (probed:
  `coef_` (3,), names `['a', 'b']`); `feature_transformer_` now joins that state. Not fixed
  here; for the user to decide.

### Step 4 — Remove `ModelPipeline`; docstrings of the composites
Files: delete `modeling/pipeline.py` and `tests/unit/test_modeling_pipeline.py`;
edit `modeling/__init__.py`, `modeling/independent_cohorts.py`, `modeling/independent_total_probability.py`,
`tests/unit/test_modeling_contract.py`, `tests/unit/test_modeling_independent_cohorts.py`,
`tests/unit/test_modeling_independent_total_probability.py`, `tests/unit/test_modeling_calibration.py`.
- [x] Delete the module, its export in `__init__.py` and `__all__`, and its `EXAMPLES` entry
  and import in the contract test (`test_every_shipped_model_has_an_example` will otherwise fail).
- [x] Replace every `ModelPipeline(features, Model(...))` in the tests with
  `Model(..., feature_transformer=features)`. Ensure `test_modeling_calibration.py`'s use
  (around line 225) and the `_RenamedShares` / `_SeriesTotal` stand-ins still fit.
- [x] `independent_cohorts.py`: the comment "usually a ModelPipeline" → "each with its own
  `feature_transformer`"; docstring: "each has its own feature transformer, model and
  hyperparameters" stays true. Add the I1 sentence: "A plain `dict` suffices: the output
  order is `y`'s column order at fit, whatever the mapping's order."
  `independent_total_probability.py`: "Both are usually a ModelPipeline, each with its own
  features; their settings are reached by nested names (`cohort_probability_model__l2_penalty`)"
  — one level shorter now that there is no `model__`.
- [x] Grep check: `grep -rn "ModelPipeline\|pipeline" src/age_group_prediction/modeling tests/unit/test_modeling_*`
  returns nothing (`src/student_simulator/pipeline.py` is a different package and stays).
  Result: one unrelated hit remains, `test_the_preprocessing_pipeline_reproduces_build_modeling_table_exactly`
  in `test_modeling_data.py`.
- [x] Added during the step: `ModelPipeline`'s docstring was the only place stating that the rows
  of `X`, `y` and the exposure are paired by position (a misaligned split passes silently); that
  sentence moved to `BaseAgeGroupModel.fit`, and `IndependentCohortModels` points there.
  `test_modeling_pipeline.py` was deleted without moving a test: each of its five has a
  counterpart (template tests in `test_modeling_feature_transformer.py` and
  `test_modeling_direct_cohort.py`; Direct's doubling and follows-fit tests; the statistics test,
  which also catches a refit at predict; the logits test).
- **Done when:** suite and mypy pass; the grep is empty. Stop.
- **Result (2026-10-05):** ruff and mypy clean; the four changed test files (62 tests) pass with
  `-W error`; mutation: `pipeline.py` restored with its export but no `EXAMPLES` entry fails
  `test_every_shipped_model_has_an_example` (it discovers six models against five examples);
  review: no correctness findings. One test gap, reproduced: `_transform_features` checking the
  template (`self.feature_transformer is None`) instead of the fitted copy passed every test, and
  after `set_params(feature_transformer=None)` all three models predicted on the raw table
  silently (40–74% off, no error); the deleted `ModelPipeline` follows-fit test had guarded the
  analogous mistake. Added `test_predict_follows_the_transformer_fitted_at_fit[model]` to
  `test_modeling_feature_transformer.py`; that mutation fails it ×3. Nits applied: the composites'
  docstrings reflowed and de-duplicated, the cross-reference fully qualified, "usually" restored in
  the `CohortModels` comment, a stale `type: ignore` dropped. Non-slow suite 1232 passed,
  1 skipped, 1 xfailed.
  **Deferred by the user to a separate PR:** the mixed fitted state after a failed refit in
  `TotalChildrenModel` and `CohortProbabilityModel` (Step 3's open finding); this PR leaves
  both `fit` methods as they are.

### Step 5 — The evaluator fits the model on the raw rows
Files: `hyperparameter_tuning/evaluator.py`, `tests/unit/test_hyperparameter_tuning_evaluator.py`,
`docs/HYPERPARAMETER_TUNING_PLAN.md` (§5 example, D13, the §6 task).
- [x] Remove the `feature_transformer` field and its `InstanceOf` import if unused (`InstanceOf` stays: `metric` uses it); rename
  `build_feature_transformer_and_model(params) -> (FeatureTransformer, BaseAgeGroupModel)`
  to `build_model(params) -> BaseAgeGroupModel`. In `evaluate`, delete the per-fold
  `feature_transformer.fit/transform` block; call `model.fit(X_train, y_train, exposure=exposure_train)`
  and `model.predict(X_val, exposure=exposure_val)` on the raw rows.
- [x] Docstrings: "`X` is the raw table; a numpy array works only with a model whose
  `feature_transformer` is `None`"; the class example as in §4 (`estimator=…`,
  `feature_transformer=tree`, `estimator__…` names). Module docstring of `parameters.py`
  line 3: the example name becomes `"estimator__learning_rate"`.
- [x] Tests: drop the `feature_transformer` construction/validation cases (lines ~151, 184,
  214–222); rename the two `build_feature_transformer_and_model` tests to `build_model`;
  `_RecordingTransformer` now goes in as `DirectCohortModel(feature_transformer=_RecordingTransformer(), …)`
  and the test asserts it saw only training rows per fold; the "templates stay unfitted"
  test checks `evaluator.model` and `evaluator.model.feature_transformer`. Add
  `test_the_model_transforms_each_fold_on_its_training_rows` if `_RecordingTransformer`
  does not already prove it.
- [x] `HYPERPARAMETER_TUNING_PLAN.md`: §5 example rewritten (every parameter name prefixed
  `estimator__`; `subsample_freq=1` set on the template estimator since the model no longer
  derives it; `exposure = ExposureTransformer("n_apartments").fit_transform(df)` before the
  split and `take_rows(exposure, train_index)`, as the §6 task required; refit through
  `build_model`); D13 marked superseded by this plan; the §6 task ticked with a pointer here;
  3.4's "refit through `build_feature_transformer_and_model`" → `build_model`.
- **Done when:** the evaluator tests pass with `-W error`; the suite passes; the §5 code
  block runs (rule: run every code block you put in a doc). Stop.
- **Changed during the step (approved):** `_RecordingTransformer` and
  `test_the_transformer_is_fitted_per_fold_on_training_rows_only` were deleted, not moved into
  the model. The evaluator no longer fits a transformer; that `model.fit` gets exactly each
  fold's training rows is pinned by `test_every_trial_sees_identical_folds`, and the model's own
  transformer behaviour by `test_modeling_feature_transformer.py`. The transformer-template test
  became `test_the_model_template_is_left_unfitted` (`evaluate` fitting `self.model` is silent:
  equal scores). The hand-written fold loop keeps an independent by-hand transform on a model
  with `feature_transformer=None`. In the tuning plan, §4.2 and D11 were updated too.
  `HyperparameterStudy` is Phase 3, so §5 ran with a seeded-Optuna stand-in (the user's choice).
- **Result (2026-10-05):** ruff and mypy clean; the evaluator tests (38) pass with `-W error`;
  mutations, each restored: (a) `evaluate` fits `self.model` → the template, suggested-once and
  hand-loop tests fail; (b) the model fitted on all rows → identical-folds, exposure-slicing,
  hand-loop and DataFrame-target tests fail; (c) `build_model` without `set_params` → both build
  tests, suggested-once and hand-loop fail. The §5 block (extracted from the doc) and the class
  docstring example ran on simulated tables. Non-slow suite 1230 passed, 1 skipped, 1 xfailed.
  Review: no correctness findings; the deletion of the recording-transformer test confirmed (no
  evaluator mutation is caught only by it), and the hand loop does tell per-fold features from
  features fitted on all rows (-1.41335 vs -1.40813). Applied: the tuning plan's §8 PR text and
  §4.2 no longer list a `FeatureTransformer` setting, its status line marks the 1002 count as
  Phase-2 history, and `evaluate`'s numpy note covers models without a `feature_transformer`.
  Rejected: re-adding a type check on `feature_transformer` (the evaluator's pydantic field had
  one). Probed: a `StandardScaler` given as a model's `feature_transformer` fits a copy and
  predicts with the training statistics, equal to scaling by hand; nothing passes silently, so
  there is no check to add.

### Step 6 — Documentation
Files: `docs/DIRECT_COHORT_MODEL.md` §0.1–§0.6, `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0,
`docs/MODULE_REFERENCE.md`, `docs/FEATURE_TRANSFORMATIONS.md` (ModelPipeline mentions),
`docs/MULTI_COHORT_MODELS_PLAN.md` (a dated note that `ModelPipeline` was folded into the
models by this plan; N20 pointer), `docs/MODEL_REIMPLEMENTATION_PLAN.md` (same note at §2),
`docs/README.md`.
- [x] `DIRECT_COHORT_MODEL.md` §0.1: copy §2b of this plan **in full** (setup, (A)–(E),
  LaTeX as written) as a new subsection "The exposure as a weighted regression of the
  per-apartment rate", replacing the current `init_score` description; keep §0.1's symbols
  consistent with it ($E_b$, $\lambda_b$, $\mu_b$, $F$). §0.5 (evidence) gets the measured
  $1.5 \times 10^{-8}$ agreement and the test that pins it. §0.2 API table: the new constructor,
  `estimator_`, `use_exposure_`, `feature_transformer_`; the code
  example passes `feature_transformer=tree` and the raw rows. §0.3 errors: remove the
  log-link error, add the estimator's own error of G5 (no pre-check). §0.4: note the generalization. §0.6: the title
  and table lose `ModelPipeline`; the example builds
  `DirectCohortModel(estimator=LGBMRegressor(…), use_exposure=True, feature_transformer=tree)`
  per cohort; the rules table:
  the nested-names rule becomes `estimator__learning_rate`, and the I1/I2 answers get a row.
- [x] `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0: sub-models carry their own
  `feature_transformer`; nested names drop `model__`.
- [x] `MODULE_REFERENCE.md`: delete the `modeling/pipeline.py` row; update `direct_cohort.py`
  ("any scikit-learn-style regressor with a Poisson or Gaussian loss; the exposure as a
  weighted rate; optional `feature_transformer`"), `independent_cohorts.py`, `total_children.py`,
  `cohort_probability.py`, `base.py` (the helpers) and `hyperparameter_tuning/evaluator.py` rows.
- [x] Trim the docstrings and comments of the files this PR touched to §3 rule 7 (the user's
  scope); the moved derivations, examples and motivation land in `DIRECT_COHORT_MODEL.md` §0
  first (and the evaluator's example in the tuning plan's §5).
- [x] Run every code block changed in the docs.
- **Done when:** `grep -rn "ModelPipeline" docs` hits only historical notes that say it was
  removed; all changed blocks ran. Stop.
- **Result (2026-10-05):** docs updated: `DIRECT_COHORT_MODEL.md` §0 (§0.1 is §2b verbatim, the
  rest in its notation; §0.3 records, probed, that LightGBM fits a zero, NaN or infinite exposure
  silently while `HistGradientBoostingRegressor` raises; §0.6 retitled, anchors updated),
  `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0, `FEATURE_TRANSFORMATIONS.md` §3.3 and §8.1–§8.3,
  `MODULE_REFERENCE.md`, `README.md`, and dated notes in the two historical plans (N17, N18, N20
  pointers). Trim (§3 rule 7): `DirectCohortModel`'s class docstring keeps the contract, the
  §0.1 pointer and the GLM `alpha` line (the examples, `subsample_freq`, the likelihood identity,
  the `init_score` comparison and the missing-`sample_weight` error moved to §0.2–§0.3);
  `BaseAgeGroupModel` lost four motivation clauses; the evaluator's example became a pointer to
  the tuning plan's §5; `IndependentCohortModels` lost one. Ran, under `-W error`, in one
  namespace with `raw_table` from the simulator (seed 0): the 10 blocks of
  `FEATURE_TRANSFORMATIONS.md` §8.0–§8.3, both of `DIRECT_COHORT_MODEL.md` §0 and the one of
  `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0; the §6 end-to-end probe; the §0.6 nested-name rule
  (`cohort_models__a__estimator__…` raises `AttributeError`). The `ModelPipeline` grep hits only
  plans, the handoff and the README's plan row. ruff and mypy clean; non-slow suite 1230 passed,
  1 skipped, 1 xfailed. Review: no code findings; every probed doc claim held (bad exposures,
  the `sample_weight` errors, all-zero `y`, nested names, the `alpha` scaling, the anchors).
  Applied: four links in `MODEL_REIMPLEMENTATION_PLAN.md` that the heading renames had broken;
  six places in `FEATURE_TRANSFORMATIONS.md` (§1, §3.1, §5, §7, §8.7) that still described
  `init_score` or "Poisson only"; in §0.1 and §2b alike, the intercept renamed $F_0$ (it clashed
  with the building index $b$), and "(G4)", "Decision G3" and "the fallback" reworded; §0.3's
  parenthetical ("a non-finite rate or weight"); §0.5 states the pinned tolerance and the measured
  one; `MODULE_REFERENCE.md`'s "Depends on" adds `feature_engineering` to the three leaf models.
  Blocks re-run, all pass; §0.1 still equals §2b. Left for the user: rule-7 leftovers in
  `total_children.py` (the lgamma-floor evidence, the penalty's motivation) and
  `cohort_probability.py` (the Model 2 context, the citation), pre-existing PR #11 text outside
  the agreed scope; and whether the LightGBM `subsample_freq` note, now only in the docs, should
  return to `DirectCohortModel`'s docstring as a silent-failure constraint.

### Step 7 — Final check and PR
- [x] The whole diff's comments and docstrings follow §3 rule 7.
- [x] Full routine on the whole diff: ruff, mypy, `uv run pytest -m "not slow"`, then the
  slow tests touching `modeling` or `hyperparameter_tuning` if any.
- [x] `git status` shows only `modeling/`, `hyperparameter_tuning/`, their tests and `docs/`.
- [x] Draft PR body (user applies it): the two generalizations, the decisions table, the
  evaluator change, the removed classes/parameters (`ModelPipeline`, `Objective`, the ten
  LightGBM arguments, `regressor_`, `base_log_rate_`), and the migration line
  `DirectCohortModel(n_estimators=…)` → `DirectCohortModel(estimator=LGBMRegressor(…))`.
- [x] Update this doc's status; update the memory files for the tuning plan (its §6 task is
  done here) and the multi-cohort plan.
- **Result (2026-10-05):** ruff, format and mypy clean on the whole diff; non-slow suite 1230
  passed, 1 skipped, 1 xfailed; the slow statsmodels oracles (`tests/validation/test_total_children.py`)
  pass; the contract test discovers exactly the five models; the tuning plan's §5 block and the
  §6 probe run. The PR diff is 29 files, all under `modeling/`, `hyperparameter_tuning/`, their
  tests and `docs/`. The `subsample_freq` note stays docs-only (the user asked for best practice:
  scikit-learn's meta-estimators leave inner-library quirks to the library; the model no longer
  touches `subsample`). Whole-diff review: no correctness findings; code, tests and docs agree
  (exports, parameters, fitted state, nested names probed). Rejected, by probe: §0.3's
  "(n, n) prediction" is right (with a Series `y`, pandas raises at fit; the silent broadcast is at
  predict). Left for the user, optional: a few short rule-7 phrases in `base.py` and
  `independent_cohorts.py`; two helper tests in `test_modeling_base.py` that the per-leaf tests
  now subsume; a stale comment in `test_hyperparameter_tuning_parameters.py` (outside the diff:
  bare LightGBM names called "the DirectCohortModel bounds"). §7 holds the final PR body; the
  handoff and memory are updated.

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
    "n_kindergarten": DirectCohortModel(estimator=LGBMRegressor(objective="poisson", n_jobs=1),
                                        use_exposure=True, feature_transformer=tree),         # LightGBM
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

Opened after Step 0's first commit; final form after Step 7. No "Generated with" footer.

**Title:** `Any regressor in DirectCohortModel; the feature transformer inside each model`

**Body:**

```markdown
Into `feat/hyperparameter-tuning`, where `modeling/` lives. Plan and decisions:
`docs/DIRECT_COHORT_GENERALIZATION_PLAN.md`.

## What changes

1. **`DirectCohortModel(*, estimator, use_exposure=False, feature_transformer=None)`.**
   Any scikit-learn-style regressor with a Poisson or Gaussian loss, always given
   explicitly (`estimator` is required). The
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
   `IndependentTotalProbabilityModel` are unchanged in behaviour; their sub-models carry
   their own transformers.
4. **`CVHyperparameterEvaluator` fits the model on the raw rows**: its `feature_transformer`
   field goes, `build_feature_transformer_and_model` becomes `build_model`. This closes the
   tuning plan's "Evaluator on the raw table" task.
5. **Docs:** `docs/DIRECT_COHORT_MODEL.md` §0 (the derivation of the weighted rate for Poisson
   and Gaussian losses, the API, the errors), Model 2's §0, `FEATURE_TRANSFORMATIONS.md`,
   `MODULE_REFERENCE.md`, the tuning plan. Docstrings keep only what the code does and its
   non-obvious constraints; derivations, examples and motivation moved to the docs.

## Removed or renamed

`ModelPipeline`, `Objective`, the ten LightGBM constructor arguments of `DirectCohortModel`,
its `regressor_` and `base_log_rate_` (now `estimator_`, `use_exposure_`,
`feature_transformer_`), `CVHyperparameterEvaluator.feature_transformer`.
Migration: `DirectCohortModel(n_estimators=100, ...)` →
`DirectCohortModel(estimator=LGBMRegressor(objective="poisson", n_estimators=100, ...))`;
`ModelPipeline(tree, model)` → the leaf model with `feature_transformer=tree`;
`evaluator.build_feature_transformer_and_model(params)` → `evaluator.build_model(params)`,
fitted on the raw rows; tuned names gain the prefix `estimator__` (`estimator__learning_rate`).
LightGBM ignores `subsample` unless `subsample_freq >= 1` is set on the estimator; the old
model set it itself.

## Follow-up

Pre-existing since PR #11, deferred to a separate PR: in `TotalChildrenModel` and
`CohortProbabilityModel`, a refit whose final `validate_data(reset=True)` raises leaves the new
`coef_` with the old `feature_names_in_`.

## Steps

- [x] 0 Branch, plan doc, baseline (1221 passed)
- [x] 1 Feature helpers in `BaseAgeGroupModel`
- [x] 2 `DirectCohortModel`: any regressor, weighted-rate exposure
- [x] 3 `feature_transformer` on the three leaf models
- [x] 4 Remove `ModelPipeline`
- [x] 5 Evaluator on the raw rows
- [x] 6 Docs (including the derivation)
- [x] 7 Final checks

## Checks

Per step: ruff, mypy, changed tests with `-W error`, one mutation check per claimed
behaviour, an independent review, `uv run pytest -m "not slow"`. Every changed doc code block
was run. Final: the non-slow suite gives 1230 passed, 1 skipped, 1 xfailed (1221 before the
PR); the slow statsmodels oracles for `TotalChildrenModel` pass; the contract test discovers
the five remaining models; an independent review of the whole diff.
```
