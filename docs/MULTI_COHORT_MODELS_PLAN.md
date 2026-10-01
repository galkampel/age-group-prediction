# Plan: complete Model 1 (independent cohorts) and rebuild Model 2 (total × probability)

**Written for:** the implementing model (Claude Opus 5.5) and the user who validates each step.
This file is self-contained: it assumes no memory of the planning conversation.
This file is the source of truth: update its status line and checkboxes as steps finish.

**Status (2026-09-30):** **PR A is merged** (PR #10, merge commit `27459eb`
into `feat/hyperparameter-tuning`; steps A0–A4 and the close-out `01837d4`).
PR B's branch `feat/independent-total-probability-model` was created from
`27459eb`, and its first commit is this handoff. **B0 is done** (2026-09-30),
except the user's PR #5 note (§2):
the branch is pushed, draft PR [#11](https://github.com/galkampel/age-group-prediction/pull/11)
is open, and the facts from PR A are re-verified. **B0a** (added by the user at
the start of B1): `Splitter.train_test_indices` replaces `train_test_split`, so
the exposure is split by the same positions as every other array; committed
`6970c8b`. **B1 is done** (`e2a4955`): `LBFGSMinimizer` in
`modeling/optimization.py`, and the exposure rule, with a length check, in
`BaseAgeGroupModel._check_exposure`. **B2 is done** (2026-09-30, revised
2026-10-01 on the user's notes; the user commits it): `TotalChildrenModel`,
Poisson, without a `family` setting (B8 adds it), with a `solver` setting over
`Minimizer`; committed `31ebdd1`. **B3 is done** (2026-10-01; the user commits
it; committed `316aabf`): `COHORT_LOG_LOSS` in `scoring.py` and the
evaluator's `y: Target`. **B4 is half done** (2026-10-01): the user specified the cohort model as a
**Dirichlet regression of the cohort shares** (after a hand-coded multinomial
logit and a scikit-learn `LogisticRegression` build were rejected);
`CohortProbabilityModel` is implemented (uncommitted; its likelihood and
gradient from torch's `Dirichlet.log_prob` with autograd, the user's choice
over written formulas) and the docs updated; **its tests, checks, mutations,
review and the stop are left to the next session** (B4 §5–§7).
`TotalChildrenModel` moved to torch's `Poisson.log_prob` the same way (its
tests pass) and lost its all-zero-`y` check. The solver of the scipy GLM
moved into `Minimizer` (`Minimizer(solver, ...)`; its own commit). **Calibration is post-hoc** (N12
revised): the model has no temperature; **B6 (`TemperatureCalibrator`) comes
before B5**, whose Model 2 takes an optional calibrator. **B8 is done**
(2026-10-01; the user commits it): `family="nb2"` on `TotalChildrenModel`
(torch's `NegativeBinomial`, a floor on `α`), and `Minimizer` runs torch
single-threaded, because LightGBM's and torch's OpenMP runtimes crash together
(B8's record; committed `5a27727`, `68b72c2`). **B4 is done** (2026-10-01;
committed `4c02f8b`): `CohortProbabilityModel`'s tests rewritten for the
Dirichlet build, its `fit` single-threaded like B8's. **B6 is done**
(2026-10-01; the user commits it): `TemperatureCalibrator` in
`modeling/calibration.py`, post-hoc temperature scaling in log space, pinned
to scikit-learn's own temperature scaling (sklearn 1.9). **Next: B5**
(`IndependentTotalProbabilityModel`), then B7, B9, one step per stop; the
handoff block before B6 in §9 still holds (B5's decisions: clone the two
models, use the calibrator as given; ask once about `FrozenEstimator`).
**B5 is done** (2026-10-01; the user commits it):
`IndependentTotalProbabilityModel` in `modeling/independent_total_probability.py`,
and `ModelPipeline.predict_logits`; the §7 usage block runs end to end.
**Next: B7** (the smoke run), then B9. Non-slow suite after B5: **1221
passed** (1 skipped, 1 xfailed).

## Contents
1. Context and goal
2. Branches and PRs
3. How to work (rules for the implementer)
4. What exists today
5. Decisions
6. Measured evidence
7. Target code shape
8. Steps, PR A (Model 1)
9. Steps, PR B (Model 2)
10. Old tests: carried over or dropped
11. Risks and pitfalls
12. Verification

---

## 1. Context and goal

The repo predicts children per building for three cohorts (`n_kindergarten`,
`n_elementary`, `n_highschool`; total `n_children_total`; exposure
`n_apartments`; split key `neighborhood_id`). The old models (`models/`,
`modeling_config.py`) cannot take fixed hyperparameters, so they are being
rebuilt as small scikit-learn-style classes in `src/age_group_prediction/modeling/`.
Model 1 and Model 2 here are the other docs' **Model A** (direct cohort) and
**Model B** (independent total and probability).
Model 1's single-cohort class exists; two things are missing:

- **Model 1 completion.** Three independent single-cohort models combined into
  one object that predicts every cohort. Each cohort has its **own model and its
  own feature transformer**.
- **Model 2 rebuild.** Two independent models whose predictions are multiplied:
  1. a **total-children** model: a count regression for the building's total;
  2. a **cohort-probability** model: a multinomial model giving (calibrated)
     probabilities of each cohort, from building and neighborhood features.

  Final prediction per cohort = total mean × cohort probability.
  It must import **nothing** from the old stack.

## 2. Branches and PRs

- **Base branch: `feat/hyperparameter-tuning`.** `modeling/`, `scoring.py` and
  `hyperparameter_tuning/` exist only there (33 commits ahead of `main`; draft
  PR #5 `main ← feat/hyperparameter-tuning` is still open). PRs #6–#9 used the
  same base.
- **Two PRs, in order:**

| PR | Branch | Content | Why separate |
|---|---|---|---|
| A | `feat/independent-cohort-models` | Model 1 completion (steps A0–A4) | Small. It settles what B builds on: the widened base contract, the contract test, `ModelPipeline`, the DataFrame output |
| B | `feat/independent-total-probability-model` | Model 2 (steps B0–B9) | Large. Branched from `feat/hyperparameter-tuning` **after A is merged** |

- One PR would mix a ~150-line change with a ~10-step rebuild, and a change
  requested in A's conventions would then be reworked inside B's code.
- PR #5's diff will now also carry these models; note it in PR #5's description.
  *PR A merged 2026-09-30 (`27459eb`); PR #5's note is still to be added (B0).
  PR B is draft [#11](https://github.com/galkampel/age-group-prediction/pull/11).*

## 3. How to work (rules for the implementer)

These are the user's standing rules. Follow them exactly.

1. **One step at a time.** Each step in §8–§9 is a validation stop. Start a
   step in plan mode: re-verify this plan's facts for that step (paths, line
   numbers, library behavior), write a **short summary of the step in the chat**,
   then ask for approval. Implement only after approval.
2. **Baseline first.** Before a step's first edit run `uv run pytest -m "not slow"`.
   If it fails, stop and report. Record the passing count in the doc.
3. **If an assumption in this plan breaks, stop** and show the options with
   evidence. Do not work around it or change course silently.
4. **Routine per step:** implement → `uv run ruff check <files>` and
   `uv run ruff format <files>` (changed files only, paths listed explicitly) →
   `uv run mypy` plus `uv run mypy src/age_group_prediction/modeling <changed test files>` →
   the changed tests with `-W error` → **one mutation check per claimed
   behavior** (break the code, see the named test fail, restore) → an
   **independent review subagent**, each finding reproduced before it is fixed
   or rejected → the non-slow suite → update the plan doc → stop.
5. **At each stop** give: a file-by-file summary of the `.py` changes, the check
   results, and suggested commit commands. **The user commits and pushes.**
   Never commit without explicit approval. No `Co-Authored-By` line and no
   "Generated with" footer.
6. **Justify every class, field and check**, or drop it. Validate only what
   would otherwise pass silently; leave to the library what it already raises.
   Data values are validated where the data is prepared (`preprocessing.py`),
   not inside each model (N3).
7. **Code style** (match `modeling/direct_cohort.py`): one-line module
   docstring, `from __future__ import annotations`, explicit `__all__`;
   keyword-only constructors that store arguments verbatim; validation in
   `fit`; fitted state in trailing-underscore attributes assigned together only
   after success; `predict` follows the fitted state, not current settings;
   short reST docstrings that say *why*; sparse comments for non-obvious
   reasons; precise types with named aliases; lower-case error messages that
   say what to do.
8. **Names:** `feature_transformer`, `exposure_*` (never `n`/`N` as a public
   name), `train`/`val`. In `modeling`, shared logic goes **under a class**;
   there is no utils file (N16).
9. **Tests:** each test's name and comment state the mistake it catches. No
   test that only checks a library.
10. **Docs** are updated in the same PR. Run every code block you put in a doc.
11. **Probes** (read-only, never write files in the repo):
    `PYTHONPATH=src uv run --group test python -c "..."`. zsh does not split
    `$VAR`, and an unquoted `--include=*.py` fails: quote globs.

## 4. What exists today

| File | What it gives |
|---|---|
| `src/age_group_prediction/modeling/base.py` | `BaseAgeGroupModel(BaseEstimator, ABC)`: abstract `fit(X, y: pd.Series, exposure=None) -> Self`, abstract `predict(X, exposure=None) -> np.ndarray`, concrete `evaluate(y_true, y_pred, metric) -> float`. *Since A1: `y: pd.Series \| pd.DataFrame`, `predict -> np.ndarray \| pd.DataFrame`. A2 added the property `uses_exposure`; A3 removed it (N3)* |
| `src/age_group_prediction/modeling/direct_cohort.py` | `DirectCohortModel`: LightGBM for **one** cohort, on a finished design matrix; `use_exposure`; `_check_exposure`; fitted `regressor_` and `base_log_rate_` |
| `src/age_group_prediction/preprocessing.py` | `ShareTransformer`: fit-free, row-wise, run on the full table before splitting. *A2 adds `ExposureTransformer`* |
| `src/age_group_prediction/scoring.py` | `Metric(name, function, greater_is_better=False)`, `POISSON_DEVIANCE`, `RMSE`, `MAE` |
| `src/age_group_prediction/feature_engineering/transformer.py` | `FeatureTransformer(plans, *, interactions, remainder)`; `fit`, `transform -> DataFrame`. *`exposure_column` and `log_exposure` were removed in A2 (N3)* |
| `src/age_group_prediction/hyperparameter_tuning/evaluator.py` | `CVHyperparameterEvaluator`; `build_feature_transformer_and_model(params) -> (FeatureTransformer, BaseAgeGroupModel)`; `evaluate(..., y: pd.Series \| np.ndarray, ...)` (at line 83 when written; refer to it by name, lines move) |
| `src/age_group_prediction/utils.py` | `DesignMatrix`, `Target`, `Groups`, `Exposure`, `take_rows` |
| `tests/unit/test_modeling_contract.py` | Discovers every concrete model in `modeling`, builds it with `model_class()`, runs 4 sklearn checks and a refit test that calls `fit(X, y)` with a Series and no exposure. *Since A1: built from an `EXAMPLES` factory per class* |
| `docs/FEATURE_TRANSFORMATIONS.md` §8.1–§8.3 | The transformer declarations `tree`, `total_base`, `composition_base` |
| `docs/MODEL_REIMPLEMENTATION_PLAN.md` | Decisions M1–M12 for Model A; §5 roadmap (step 3 = this rebuild) |

**Old Model 2** (read for the math only; import nothing from it):
`models/independent_total_probability.py`, `count_regression.py`,
`grouped_multinomial.py`, `probability_calibration.py`, `fold_scoring.py`;
reference doc `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md`.

**Forbidden imports in `modeling/`:** `models`, `modeling_config`,
`fitted_features`, `distributions`, `results`, `tuning`, `metrics` (old),
`predictive`, `resampling`, `state_bundle`, `data_splitting`, `evaluation`.

## 5. Decisions

| # | Decision | Why |
|---|---|---|
| N1 | One class hierarchy. `BaseAgeGroupModel` is widened to `y: pd.Series \| pd.DataFrame` and `predict -> np.ndarray \| pd.DataFrame`; every new class subclasses it | The cohort-probability model needs a DataFrame `y` anyway. One contract keeps `evaluate`, `clone`/`set_params` and the contract test for all models. A second base or generics adds code that mypy cannot check (pandas is untyped here) |
| N2 | Multi-cohort classes take the **raw table** and return a **DataFrame**: one column per cohort, named from `y`'s columns at fit, indexed like `X`. Each of their models is a `ModelPipeline`, which holds the feature transformer (N17) | User's choice. Named columns cannot be mixed up by position. M9 still holds for single models: they take a finished design matrix |
| N3 | *Revised four times on 2026-09-29, each time by the user.* **The exposure is the model's, and its values are validated in preprocessing.** (a) Whether a model has an offset is its own constructor setting (`use_exposure`), on the models that can have one; the base declares no such setting or property. (b) Which column holds it is a schema fact: `exposure_column="n_apartments"`. (c) `preprocessing.ExposureTransformer` reads that column and returns it as floats, or raises if a value is not strictly positive and finite. (d) The caller builds the exposure with it, on the full table before splitting, and passes it as `exposure=` to every model, `ModelPipeline` (N17) included. (e) Every model takes the same exposure: one with an offset raises if it is missing; any other model **ignores** it (A3). (f) The models take the **raw** exposure, not its log. (g) `FeatureTransformer.exposure_column` and `log_exposure` were removed | (a, b) statsmodels, R, glum, LightGBM and sklearn's examples all pass the exposure to the model, outside the feature matrix. (c) LightGBM accepts a zero, infinite or NaN exposure silently (§6), so the check is needed once. (e) A missing exposure would drop the offset silently; an ignored one lets a caller pass one exposure to every model, and a tuner compare `use_exposure` with one fixed exposure (the evaluator passes the same exposure to every trial). As sklearn's metadata routing does for metadata a consumer declares not requested. Before A3 an unexpected exposure raised, and `uses_exposure` told an aggregator where to route it. (f) `DirectCohortModel` needs `Σ exposure` for its intercept, and `exposure=` is settled (M10), as in statsmodels. History: first a transformer's `exposure_column` declared the exposure; then helper functions read it; then the user asked for the validation in preprocessing and the logic under a class; then for an explicit exposure argument on `ModelPipeline` (d), which gives up the guarantee that the pipeline always validates it |
| N4 | No scoring override. Per-cohort scores are a caller loop: `model.evaluate(Y[c], predictions[c], metric)` | `mean_poisson_deviance` rejects several columns, and how to average cohorts is the caller's choice |
| N5 | Names: `IndependentCohortModels` (Model 1), `TotalChildrenModel`, `CohortProbabilityModel`, `IndependentTotalProbabilityModel` (Model 2), `TemperatureCalibrator` | User: the first model is named for total children, the second for cohort probabilities. The combined class keeps the old name, as `DirectCohortModel` did; new classes are imported from `age_group_prediction.modeling` only |
| N6 | Fitting uses `scipy.optimize.minimize(method="L-BFGS-B")` on objectives built from **predefined library functions**: `scipy.stats.poisson.logpmf`, `scipy.stats.nbinom.logpmf`, `scipy.special.log_softmax` and `xlogy`. Gradients are analytic. *B2 (user, 2026-10-01): the method is the model's `solver` setting, `"lbfgs"` (default) or `"bfgs"`, the two gradient-only methods that reach the oracle; see B2's revision. B4 (user): the solver is the optimizer's own setting, `Minimizer(solver, max_iter, tol)`, which maps it to scipy's method. **B4 (user, 2026-10-01): the cohort model is a Dirichlet regression of the shares**, its objective **torch's `Dirichlet(α).log_prob(shares)` with autograd** (torch is a project dependency): vectorized over buildings and an exact gradient, no written density or gradient (the user's choice; measured equal to the `gammaln`/`digamma` formulas to 1e-15, 13–44 ms per fit; scipy's `dirichlet.logpdf` takes one `α` and has no gradient, so a row loop with finite differences cost 1.8–31 s per fit), minimized by `Minimizer` (scipy L-BFGS-B/BFGS: the standard for a smooth 12–30-parameter problem, as `DirichletReg`). **The total model's objective was moved to torch the same way** (`Poisson(μ).log_prob(y)`, autograd; the `PoissonRegressor` oracles and the statsmodels check pass unchanged; ~4 ms per fit at 400 rows). B8's NB2 is then `NegativeBinomial` from torch (B8: in its logits form, `log α + log μ`, with the floor `α ≥ 1e-6`; a model's `fit` runs its torch calls under `optimization.single_threaded_torch()`, since LightGBM's, scikit-learn's and torch's own OpenMP runtimes share one process and torch's threaded kernels then crash, see B8's record). No library offers the regression itself (not sklearn, not statsmodels). A scikit-learn `LogisticRegression` build (multinomial on the counts) was rejected because it models the counts, not the composition* | User: scipy is fine if the objective is predefined or easy to validate. Every objective is pinned to a library fit in tests (N7), and every gradient to `scipy.optimize.check_grad` |
| N7 | Test oracles: sklearn `PoissonRegressor` and `LogisticRegression` in `tests/unit`; statsmodels in `tests/validation` with `pytest.importorskip`. *B4: the Dirichlet regression has no library oracle; its tests pin the objective to `scipy.stats.dirichlet.logpdf`, the gradient to `check_grad`, and recover known parameters from Dirichlet-drawn data* | statsmodels is only in the `validation` dependency group |
| N8 | One penalty meaning in both models: `l2_penalty` multiplies `½‖coefficients‖²` added to the **mean** negative log-likelihood, per building for both models (*B4: the Dirichlet observation is a building's composition; before B4 the probability model's was per child*). Intercepts are not penalized. *B6 (user): the temperature objective is per building too* | Comparable across folds of different size. Conversions for the oracles are in §7 |
| N9 | No clipping of the linear predictor and no floor on the mean. The optimizer runs under `np.errstate(over="raise", invalid="raise")`; a failure or non-finite result raises `RuntimeError` naming feature scale as the likely cause | Measured: clipping hid a failed fit (it returned `success=True` at a wrong point). `exp(·) > 0` already |
| N10 | `TotalChildrenModel`: `family: Literal["poisson", "nb2"] = "poisson"`, `use_exposure: bool = True` (N3). **Poisson is built first (B2); NB2 is its own step (B8)**. *User, 2026-10-01: NB2 wanted; B8 moved before B4's tests. B8: NB2 is torch's `NegativeBinomial`, its dispersion `α` fitted as `log α` from `log 0.1` with a floor `α ≥ 1e-6` (the Poisson limit; without it `log α` runs to −16…−23 on data without overdispersion and BFGS fails), so `nb2` takes `solver="lbfgs"` only; fitted `dispersion_`* | User's choice. The offset is Model 2's specification, so a forgotten exposure raises. NB2 showed no gain in the means (§6), so it must be droppable |
| N11 | `CohortProbabilityModel` works for any number of cohorts ≥ 2, taken from `y`'s columns. *B4 (user): a **Dirichlet regression** of the shares `y_b / Σ_k y_bk` in the common parameterization, `α_bk = exp(a_k + x_b β_k)` (one intercept and coefficient column per cohort, as `DirichletReg`); the prediction is the Dirichlet mean `α/Σα`; shares with a zero are compressed toward the centre (Smithson & Verkuilen); every building needs a child. B4's tests: a one-column `y` fits the share 1, trivially right, so it is not checked* | The old code hard-coded 3. The common parameterization is identified: a shift of all intercepts changes the precision `Σα`, not the mean, so the coefficients are unique and tests can recover them. *Before B4: a softmax of the counts (symmetric, coefficients not unique)* |
| N12 | Calibration is **temperature scaling**, fitted by a separate `TemperatureCalibrator` on **out-of-fold** logits. ~~`CohortProbabilityModel` has an ordinary setting `temperature: float = 1.0`, applied in `predict` as `softmax(logits / temperature)`.~~ *Revised by the user at B4 (2026-10-01): calibration is post-hoc, as `CalibratedClassifierCV(method="temperature")` and Guo et al. (2017) do it: the model is fitted and left as is, and the fitted calibrator maps its logits to `softmax(logits / T)`. `CohortProbabilityModel` has no temperature (`predict_logits`, `predict` = softmax); `IndependentTotalProbabilityModel` takes an optional `temperature_calibrator=None`, fitted by the caller. The setting had also broken rule 7 (predict read a current setting)* No folds inside any model | User's choice. Best practice: it is sklearn's own multiclass method (`CalibratedClassifierCV(method="temperature")`, since 1.8), has one parameter, and measured best here (§6). Isotonic is not advised below ~1000 calibration rows |
| N13 | **No likelihood-ratio gate** on the temperature: the fitted value is always used | Best practice and simpler: sklearn's implementation has none. On well-calibrated data the fitted temperature lands near 1 and changes little. The old gate guarded a threshold rule that no longer exists |
| N14 | Out-of-fold logits come from a short documented loop (in the doc and one integration test), not a helper | One caller today. Promote it to a helper when a second caller exists |
| N15 | Dropped from Model 2, as M12 did for Model A: tuning inside `fit`, bootstrap draws and intervals, pointwise log-probabilities, `PredictionResult`, state bundles, metadata, seed records | Means only. Tuning lives in `hyperparameter_tuning` |
| N16 | *Withdrawn by the user, 2026-09-29.* **No utils file in `modeling`; logic lives under a class.** The home of `minimize_lbfgs` is decided at B1. *Decided (user, B1): a component class, `LBFGSMinimizer` in `modeling/optimization.py`. B2: `Minimizer`; B4: it owns the solver names (`Solver`). No GLM base (user, B4): the GLMs repeat only scikit-learn's per-estimator calls* | User's rule. First version: shared helpers in a public `modeling/utils.py` |
| N17 | *Revised by the user on 2026-09-29: the exposure is an explicit argument.* `ModelPipeline(feature_transformer, model)`, a `BaseAgeGroupModel`: a feature transformer, then a model, fitted and used on the raw table. `fit(X, y, exposure=None)` clones both; `predict(X, exposure=None)` uses the fitted copies `feature_transformer_` and `model_`. `exposure` is passed through to the model, whose own check decides whether one is needed. Rows of `X`, `y` and the exposure are paired by position (N19). The aggregators (`IndependentCohortModels`, `IndependentTotalProbabilityModel`) hold models that take the raw table, and only loop and combine | User's choice. scikit-learn's `Pipeline` pattern, by composition: each class does one thing. sklearn's own `Pipeline` was already rejected (`HYPERPARAMETER_TUNING_PLAN.md` D13): its `fit` and `predict` name the exposure differently, and it has no `evaluate`. The explicit exposure keeps the base contract `fit(X, y, exposure)` for every model, and lets the tuner's `exposure=` path take a pipeline. First version: the pipeline read `X[exposure_column]` and rejected an `exposure` argument |
| N18 | `base_log_rate_` stays. It is the intercept `b` in `exposure × exp(b + F(x))`. No intercept option is added for a model without an exposure | User's decision after the evidence in §6. Given an `init_score`, LightGBM switches off its own starting average, so the model supplies `b`. Without an exposure LightGBM starts from `mean(y)` itself |
| N19 | *Reversed by the user at the end of A3, 2026-09-29.* **No index check.** `X`, `y` and the exposure are paired by position, as in scikit-learn; the docstrings say so, and to take the exposure's rows as `exposure.loc[X_train.index]`. *Since B0a (2026-09-30): the splitter returns row positions (`Splitter.train_test_indices`), and every array, the exposure included, is taken by them with `take_rows`* | The splitter (then `Splitter.train_test_split`; since B0a `train_test_indices`, and `cv`) splits `X`, `y` and `groups` by the same positions, so they cannot be misaligned; sklearn pairs by position too (`PoissonRegressor().fit(X, y_shuffled)` runs silently); the check raised on position-correct data whose labels differ (e.g. `X` after `reset_index`). History: the check was added to the multi-cohort classes, then to `ModelPipeline` (`7204ec2`), then shared in the base, then removed |
| N20 | The tuner path is **documented, not changed**. `CVHyperparameterEvaluator.evaluate` passes its `exposure` straight to `model.fit`, so the documented way to build that argument is `ExposureTransformer(...).fit_transform(table)`. Switching the evaluator to a `ModelPipeline` is recorded for the tuning work | User's decision. It keeps PR A out of the tuning package, and the exposure still comes from the validating class |
| N21 | *User, 2026-09-29 (A3).* `IndependentCohortModels` takes a `Mapping[str, BaseAgeGroupModel]`. Nested `set_params` names do not reach into it: each cohort is tuned on its own, and the tuned models are assembled. Replace the mapping with `set_params(cohort_models=...)` | Measured: `get_params(deep=True)` lists only `cohort_models`, and a nested name raises `AttributeError`. The cohorts are independent, so no study tunes them together |

**Decided at the start of B1 (user, 2026-09-30):** see §9 B1. A model with an
offset rejects an exposure whose length differs from `X`'s, with sklearn's
`check_consistent_length`. The question as it was asked:
- **Exposure length (B1).** Should a model reject an exposure whose length
  differs from `X`'s? `DirectCohortModel.predict` accepts a length-1 exposure,
  which `MODEL_REIMPLEMENTATION_PLAN.md` Step 2.2 accepted as "all buildings
  have this n". In the GLMs numpy broadcasts a length-1 exposure silently at
  fit. Rows are paired by position (N19). Re-examine at B0.
  *Measured after PR A (2026-09-30):* `DirectCohortModel` with a length-1
  exposure: `predict` broadcasts it silently, `fit` fails in LightGBM
  ("Initial score size doesn't match data size"); another wrong length fails
  in numpy at predict. The tuner already rejects any exposure whose shape is
  not `(len(X),)` (`CVHyperparameterEvaluator._check_exposure`).

## 6. Measured evidence (2026-09-29; sklearn 1.9.0, scipy 1.18.0, statsmodels 0.14.6)

Re-run any figure you rely on. Simulated tables: `StudentPopulationSimulator(load_simulation_config("configs/simulation.toml")).run(rng=np.random.default_rng(seed))`,
then `ShareTransformer(("3_rooms","4_rooms","5_rooms","6_rooms"), reference_column="3_rooms")`;
60 neighborhoods and 215–251 buildings (seed 0: 245); grouped 80/20 split; seeds 0–9.
*Corrected in A4: this line first gave seed 0's 245 buildings for every seed.*

| Question | Result |
|---|---|
| Poisson, scipy + analytic gradient vs sklearn `PoissonRegressor` | max coefficient difference 2.7e-07 (12 function evaluations). Numerical gradients: 1.4e-05, 110 evaluations |
| Poisson and NB2 vs statsmodels (`GLM`, `NegativeBinomial(exposure=n)`), unpenalized | ≤ 1e-5 on coefficients and dispersion |
| NB2 vs Poisson, held-out mean Poisson deviance of the total | 4.34 vs 4.40; difference −0.06 ± 0.15; NB2 better in 5 of 10. Fitted dispersion ≈ 0.105 |
| Calibration, held-out child-weighted log loss | uniform 1.0986, marginal 1.0912, raw 1.0866, **temperature 1.0850** (better in 8 of 10, T ≈ 1.1–1.25), Dirichlet 1.0855 (7 of 10) |
| Softmax on grouped counts (scipy) vs `LogisticRegression` on one row per child | 2e-16, for C in {0.01, 1.5, 100, inf}; 2 and 4 cohorts work unchanged |
| A never-observed cohort | Neither scipy nor sklearn raises. We must check |
| `clone` of `{cohort: (FeatureTransformer, model)}` | Unfitted deep copies, key order and settings kept |
| sklearn's 4 contract checks on a class with required nested estimators | Pass, when given an instance |
| LightGBM with different columns per cohort | No warning under `-W error` |
| Tuner with a DataFrame `y` and a composition metric | Already runs; only the `y` annotation of `CVHyperparameterEvaluator.evaluate` is narrow |
| A bad exposure given to LightGBM (2026-09-29) | A negative one raises. **Zero, inf and NaN pass silently**, in the `init_score` form and in the rate form |
| Rate form (`y / exposure`, `sample_weight=exposure`) vs `init_score` + `base_log_rate_` | The same model: relative difference 3e-8 over 4 hyperparameter settings, the same held-out deviance to 6 decimals. Not adopted (N18) |
| LightGBM's start without an exposure | With almost no learning every prediction is `mean(y)` (4.7733). With `boost_from_average=False` it is 1.0 |
| `ModelPipeline` and `ExposureTransformer`, defined inline in a probe. *Historical: the first A2 version, whose pipeline read and validated the exposure column itself; the shipped pipeline takes `exposure=` and only `ExposureTransformer` validates* | sklearn's 4 checks pass; nested names such as `model__learning_rate` work; doubling the exposure column gives a ratio of exactly 2; a refit equals a fresh fit; `set_params(model__use_exposure=False)` after `fit` leaves predictions unchanged; a model without an exposure works on a table without the column; a zero, negative, infinite or NaN exposure raises at fit and at predict, naming the row; a missing column gives `KeyError`; pickling works |
| mypy and pandas | pandas is untyped here, so `pd.Series` vs `pd.DataFrame` is not checked. Dropping the `exposure` parameter from an override does fail |

## 7. Target code shape

```python
# modeling/base.py  (A1: annotations and docstring only)
def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame,
        exposure: ArrayLike | None = None) -> Self: ...
def predict(self, X: pd.DataFrame,
            exposure: ArrayLike | None = None) -> np.ndarray | pd.DataFrame: ...

# preprocessing.py  (A2)
class ExposureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, exposure_column: str = "n_apartments") -> None: ...
    def fit(self, X, y=None) -> Self: ...          # learns nothing
    def transform(self, X) -> pd.Series: ...       # the raw exposure as floats, or raises

# modeling/pipeline.py  (A2; predict_logits since B5)
class ModelPipeline(BaseAgeGroupModel):
    def __init__(self, feature_transformer: FeatureTransformer, model: BaseAgeGroupModel) -> None: ...
    # fit(X_raw, y, exposure=None) ; predict(X_raw, exposure=None)
    # predict_logits(X_raw): model_.predict_logits on the transformed table (a model that has them)
    # fitted: feature_transformer_, model_

# modeling/independent_cohorts.py  (A3)
CohortModels = Mapping[str, BaseAgeGroupModel]   # each takes the raw table: usually a ModelPipeline
class IndependentCohortModels(BaseAgeGroupModel):
    def __init__(self, cohort_models: CohortModels) -> None: ...
    # fit(X_raw, y: DataFrame, exposure=None) ; predict(X_raw, exposure=None) -> DataFrame
    # the same exposure goes to every cohort; a model without an offset ignores it ; fitted: cohort_models_

# modeling/total_children.py  (B2, B8)
class TotalChildrenModel(BaseAgeGroupModel):
    def __init__(self, *, family="poisson", solver="lbfgs", use_exposure=True,
                 l2_penalty=0.0, max_iter=500, tol=1e-6) -> None: ...   # family: "poisson" | "nb2" (B8)
    # fit(X, y: Series, exposure=None) ; predict(X, exposure=None) -> ndarray
    # exposure rule: BaseAgeGroupModel._check_exposure (B1): raises if used and
    # missing, ignored if not used, must be 1-D and len(X) long; predict follows
    # the fitted state (A3); fitting via Minimizer (B1; solver added in B2)
    # fitted: intercept_, coef_, use_exposure_, feature_names_in_,
    # n_features_in_, and dispersion_ (α) for nb2 only; α ≥ 1e-6 by an L-BFGS-B
    # bound, so nb2 takes solver="lbfgs" only

# modeling/optimization.py  (B1; solver names since B4)
Solver = Literal["lbfgs", "bfgs"]
@dataclass(frozen=True)
class Minimizer:                                     # a component, not an estimator
    solver: Solver; max_iter: int; tol: float
    def minimize(self, objective, start, bounds=None) -> np.ndarray: ...

@contextmanager
def single_threaded_torch() -> Iterator[None]: ...   # B8: a fit wraps its torch calls in it

# modeling/cohort_probability.py  (B4: a Dirichlet regression of the shares)
class CohortProbabilityModel(BaseAgeGroupModel):
    def __init__(self, *, solver="lbfgs", l2_penalty=0.0,      # solver: Minimizer's ("lbfgs" | "bfgs")
                 max_iter=500, tol=1e-6) -> None: ...          # no temperature (N12, B4)
    # fit(X, y: DataFrame of cohort counts, exposure=None): shares y_b / Σy_b, compressed
    #   (Smithson–Verkuilen), Dirichlet(α_b) with α_b = exp(a + x_b W), fitted by Minimizer
    #   under single_threaded_torch() (B8); a building without children raises
    # predict_logits(X) -> DataFrame of log α (what a calibrator is fitted on)
    # predict(X, exposure=None) -> DataFrame of the Dirichlet mean α/Σα = softmax(log α), rows sum to 1
    # the base signature is kept; a passed exposure is ignored (N3 e)
    # fitted: intercept_ (K), coef_ (d×K), cohorts_, feature_names_in_, n_features_in_

# modeling/calibration.py  (B6, built before B5)
class TemperatureCalibrator(BaseEstimator):
    def fit(self, logits, counts) -> Self: ...      # fitted: temperature_
    def predict(self, logits) -> DataFrame: ...     # softmax(logits / temperature_)

# modeling/independent_total_probability.py  (B5)
class IndependentTotalProbabilityModel(BaseAgeGroupModel):
    def __init__(self, *, total_children_model: BaseAgeGroupModel,
                 cohort_probability_model: BaseAgeGroupModel,   # two ModelPipelines
                 temperature_calibrator: TemperatureCalibrator | FrozenEstimator | None = None) -> None: ...
    # fit(X_raw, y: DataFrame of cohort counts, exposure=None): clones both; total target = y.sum(axis=1)
    # predict(X_raw, exposure=None) -> DataFrame = total_mean[:, None] * shares, the shares
    #   calibrator.predict(cohort_probability_model_.predict_logits(X)) when a calibrator is given
    #   (used as given, stored at fit as temperature_calibrator_; its own predict raises if unfitted;
    #   clone drops a fit, so wrap it in FrozenEstimator for a tuner),
    #   else the probability model's own; their columns must equal y's at fit
    # the same exposure goes to both models; the probability model ignores it
    # fitted: total_children_model_, cohort_probability_model_, temperature_calibrator_, cohorts_
```

**The math.** `D = [1, X]`, `N` buildings, `M = Σ n_bk` children, `λ = l2_penalty`.

| Model | Objective (minimized) | Gradient | Oracle conversion |
|---|---|---|---|
| Total, Poisson | `−mean(log Poisson(y \| μ)) + ½λ‖β‖²`, `μ = exposure·exp(b + Xβ)`; in code torch's `Poisson(μ).log_prob` (since B4; B2 wrote `poisson.logpmf` and the gradient) | torch autograd (exact); intercept unpenalized | `PoissonRegressor(alpha=λ / mean(exposure)).fit(X, y/exposure, sample_weight=exposure)`; mean = `exposure · predict(X)` |
| Total, NB2 (B8) | `−mean log NB2(y \| μ, α) + ½λ‖β‖²` over `(b, β, log α)`, `μ = exposure·exp(b + Xβ)`, variance `μ(1+αμ)`; in code torch's `NegativeBinomial(total_count=1/α, logits=log α + log μ).log_prob` (equals `nbinom.logpmf(y, 1/α, 1/(1+αμ))` to 1e-13; the logits form never saturates, `probs` rounds to 1 at αμ ≳ 1e16); `log α ≥ log 1e-6`, start `log 0.1` | torch autograd (exact); `b` and `α` unpenalized | statsmodels `NegativeBinomial(y, D, exposure=exposure, loglike_method="nb2")`, λ = 0, within 1e-7 |
| Cohort probability (B4: Dirichlet regression) | `−mean_b log Dirichlet(s'_b \| α_b) + ½λ‖W‖²`, `α_b = exp(a + x_b W)`, `s'_b` the building's shares compressed `(s(N−1) + 1/K)/N`; in code, torch's `Dirichlet(α).log_prob`. Prediction: the mean `α/Σα = softmax(a + XW)`. `COHORT_LOG_LOSS` (B3) scores it against the counts, per child | torch autograd (exact); intercepts unpenalized | none in a library: `scipy.stats.dirichlet.logpdf` row by row pins the objective, `check_grad` the gradient, and Dirichlet-drawn data with known `(a, W)` the fit. *The earlier softmax-on-counts builds' oracle was `LogisticRegression(C=1/(λM))`, `2/(λM)` for 2 cohorts* |
| Temperature (B6) | `−mean_b Σ_k s_bk · log softmax(logits_b / T)_k`, `s_bk = n_bk / n_b` the building's observed composition (per building, as the Dirichlet fit; user's decision at B6), in log space (`log_softmax`), over `log(1/T)` in (−10, 10), `minimize_scalar(method="bounded", xatol=64·eps)`. *Before the stop: `cohort_log_loss(counts, …)`, per child* | — | `sklearn.calibration._TemperatureScaling` on one row per (building, cohort) with `sample_weight = s_bk`, the same form, T to 0.0 |

Start values: total intercept `log(Σy / Σexposure)`, everything else 0.

**The calibration objective (B6, per building; the user's decision at B6's
stop).** For building $b = 1,\dots,N$ and cohort $k = 1,\dots,K$, with counts
$n_{bk}$, total $n_b = \sum_k n_{bk}$ and the probability model's logits
$\ell_{bk} = \log \alpha_{bk}$ (`predict_logits`), the calibrated distribution at
temperature $T$ is

$$
p_{bk}(T) = \frac{\exp(\ell_{bk}/T)}{\sum_{j=1}^{K} \exp(\ell_{bj}/T)},
$$

the model's own prediction at $T = 1$ (the Dirichlet mean). With the observed
composition $s_{bk} = n_{bk}/n_b$, `TemperatureCalibrator.fit` minimizes the
cross-entropy of each building's composition under $p_b(T)$, averaged over
buildings:

$$
L(T) = -\frac{1}{N} \sum_{b=1}^{N} \sum_{k=1}^{K} s_{bk} \log p_{bk}(T),
\qquad
\hat T = \arg\min_{\log(1/T) \in (-10,\,10)} L(T).
$$

Every building counts once, as in the Dirichlet fit. $\log p_{bk}(T)$ is
computed as `log_softmax`, so a sharp logit never underflows to $\log 0$.
The earlier per-child form weighted each term by $n_{bk}$ instead of
$s_{bk}$ (one row per child, `cohort_log_loss`'s weighting).

**Usage, end to end:**

```python
# The exposure, validated once on the full table; the split's row positions
# take every array alike (N19, B0a)
COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]
exposure = ExposureTransformer("n_apartments").fit_transform(table)
train_index, test_index = Splitter("grouped").train_test_indices(
    table, table["neighborhood_id"], test_size=0.2, random_state=0)
train_df, test_df = take_rows(table, train_index), take_rows(table, test_index)
Y_train, Y_test = take_rows(table[COHORTS], train_index), take_rows(table[COHORTS], test_index)
groups_train = take_rows(table["neighborhood_id"], train_index)
exposure_train, exposure_test = take_rows(exposure, train_index), take_rows(exposure, test_index)
cv = Splitter("grouped").cv(n_splits=5, random_state=0)

# Model 1
model_1 = IndependentCohortModels({
    "n_kindergarten": ModelPipeline(tree, DirectCohortModel(use_exposure=True)),   # applies the exposure
    "n_elementary":   ModelPipeline(tree, DirectCohortModel()),                    # ignores it
    "n_highschool":   ModelPipeline(tree, DirectCohortModel(use_exposure=True)),
}).fit(train_df, Y_train, exposure=exposure_train)
predictions = model_1.predict(test_df, exposure=exposure_test)   # DataFrame, 3 columns

# Model 2, with calibration
probability_pipeline = ModelPipeline(cohort_probability_base, CohortProbabilityModel(l2_penalty=1e-3))
logits_val, counts_val = [], []
for fit_index, val_index in cv.split(train_df, Y_train, groups_train):
    fold = clone(probability_pipeline).fit(
        take_rows(train_df, fit_index), take_rows(Y_train, fit_index))
    logits_val.append(fold.predict_logits(take_rows(train_df, val_index)))
    counts_val.append(take_rows(Y_train, val_index))
calibrator = TemperatureCalibrator().fit(pd.concat(logits_val), pd.concat(counts_val))

model_2 = IndependentTotalProbabilityModel(
    total_children_model=ModelPipeline(total_base, TotalChildrenModel(l2_penalty=0.1)),   # use_exposure=True
    cohort_probability_model=ModelPipeline(                                               # no exposure
        cohort_probability_base, CohortProbabilityModel(l2_penalty=1e-3)),
    temperature_calibrator=calibrator,   # fitted above; FrozenEstimator(calibrator) if model_2 will be cloned
).fit(train_df, Y_train, exposure=exposure_train)
predictions = model_2.predict(test_df, exposure=exposure_test)
```

**Hazard to document:** `ModelPipeline` and the multi-cohort classes clone in
`fit`, so a nested `set_params(...)` after `fit` does not reach the fitted
copy. Set it before `fit`. *(Since B4 the temperature is the calibrator's,
not a setting. Decided for B5: Model 2 does not clone the calibrator, see the handoff before B6.)*

## 8. Steps, PR A: Model 1 completion

### A0. Plan doc and draft PR
1. ✓ `feat/independent-cohort-models` created from `feat/hyperparameter-tuning` (`65d6dbe`).
2. ✓ Link this file from `docs/README.md` and `docs/MODEL_REIMPLEMENTATION_PLAN.md` §5.
3. After the user approves the doc, the user commits the docs and pushes. Then
   a **draft** PR is opened into `feat/hyperparameter-tuning`.

Done when:
- [x] The user has approved the doc, names included (N5): approved 2026-09-29.
  The user also decided that the §8.3 transformer is renamed at B9.
- [x] The draft PR is open: [#10](https://github.com/galkampel/age-group-prediction/pull/10).

### A1. Base contract and contract test
- **Files:** `modeling/base.py`, `tests/unit/test_modeling_contract.py`.
- **Build:** widen the two annotations and the class docstring ("one count
  target, or one per cohort"). In the contract test replace `model_class()` with
  an `EXAMPLES` mapping: class → factory returning `(model, X, y)`.
- **Tests:** (1) a discovered model without an example **fails** rather than
  being skipped; (2) the four checks and refit-equals-fresh run on every example.

Done when:
- [x] `DirectCohortModel` passes with no change to its code.
- [x] Mutation: deleting its example fails the suite.
- [x] mypy, ruff and the non-slow suite pass (1082).

**Record (2026-09-29).**
- **Verified first:**
  - mypy: pandas has no stubs here, so the widened base type-checks, and so do
    subclasses that keep `y: pd.Series`.
  - `check_parameters_default_constructible` rebuilds the class from its
    required arguments only. So an example may set non-default settings or
    need a nested estimator.
- **Mutation checks**, each failing the named test:
  - example deleted: the coverage test, plus a `KeyError` in all 5 parametrized cases;
  - an attribute set in `__init__`: `check_no_attributes_set_in_init`;
  - a refit that differs from a fresh fit: `test_a_refit_equals_a_fresh_fit`;
  - an example that builds another class, and an example with no discovered
    model: the coverage test.
- **Review:**
  - *Fixed.* An example whose factory built another class (a copied factory)
    let that class skip every check. The coverage test now asserts each
    example's class.
  - *Fixed.* An extra example key failed with the message "add an example for []".
    The message now lists both sides.
  - *Not changed.* The refit test compares values, not DataFrame labels. It
    targets leftover fitted state, and A3's test 5 covers columns and index.

### A2. The exposure moves to the model; `ExposureTransformer` and `ModelPipeline`
*Revised by the user after the first version (N3, N16, N17, N18, N20). The step
starts from the uncommitted working tree below. Two commits.*

**The working tree today** (first version; suite 1083 passed):

| Change in the working tree | What to do |
|---|---|
| **A2a.** `exposure_column` and `log_exposure` removed from `FeatureTransformer` (`feature_engineering/transformer.py`); their 6 tests removed (`test_feature_transformer.py`, `test_transformer_spec.py`); `FEATURE_TRANSFORMATIONS.md` §8.1, §8.2, §8.3, §8.6 edited | **Keep unchanged.** It is commit 1 |
| `BaseAgeGroupModel.uses_exposure`, default `False` (`modeling/base.py`) | Keep; reword the docstring |
| `DirectCohortModel.uses_exposure` with an "as fitted" branch | Simplify to `return self.use_exposure` |
| `test_modeling_never_imports_the_old_stack` and `OLD_STACK` (`test_modeling_contract.py`) | Keep unchanged |
| `modeling/utils.py` and `tests/unit/test_modeling_utils.py` (untracked) | **Delete.** The tests move to `test_modeling_pipeline.py` |
| `MODULE_REFERENCE.md`: a `modeling/utils.py` row | Replace with `pipeline.py` |

**Build:**
1. **`preprocessing.py`: `ExposureTransformer`** (§7), added to `__all__`.
   - `fit` learns nothing and returns `self`. `transform` needs no prior `fit`.
   - `transform` returns `X[exposure_column]` as floats, with `X`'s index. It
     raises `ValueError` if a value is not strictly positive and finite. The
     message names the column, the number of invalid rows and the first 5
     index labels.
   - A missing column is left to pandas (`KeyError`), and a non-numeric one to
     `astype` (`ValueError`).
   - The docstring says why (LightGBM accepts a zero, infinite or NaN exposure
     silently) and that running it on the full table before splitting makes a
     bad test row fail early.
2. **`modeling/pipeline.py` (new): `ModelPipeline`** (N17, revision 2), exported
   from `modeling/__init__.py` and never from the package root.
   ```python
   def fit(self, X, y, exposure=None) -> Self:
       self._check_exposure_index(X, exposure)
       feature_transformer = clone(self.feature_transformer).fit(X, y)
       model = clone(self.model).fit(feature_transformer.transform(X), y, exposure=exposure)
       self.feature_transformer_ = feature_transformer   # together, after success
       self.model_ = model
       return self

   def predict(self, X, exposure=None) -> np.ndarray | pd.DataFrame:
       check_is_fitted(self)
       self._check_exposure_index(X, exposure)
       return self.model_.predict(self.feature_transformer_.transform(X), exposure=exposure)
   ```
   - Its one check: a `pd.Series` exposure whose index differs from `X`'s
     raises `ValueError`, because the model reads it by position. Whether an
     exposure is given at all is the model's own check.
   - `uses_exposure` returns `self.model.uses_exposure`, so an aggregator can
     route one exposure to the models that use one.
3. **`modeling/direct_cohort.py`:**
   - `_check_exposure` keeps the presence check and the float conversion. The
     "strictly positive and finite" check is removed; the docstring points to
     `preprocessing.ExposureTransformer`.
   - `base_log_rate_`, `init_score` and `predict`'s arithmetic do not change (N18).
4. **The tuner path (N20):** no code change in `hyperparameter_tuning`. Update
   the evaluator's class docstring example to build the exposure with
   `ExposureTransformer`, and add a task to `HYPERPARAMETER_TUNING_PLAN.md` §6:
   switch the evaluator to a `ModelPipeline`.

**Tests, each naming its mistake, with one mutation each:**

| File | Test | Mistake it catches | Mutation |
|---|---|---|---|
| `test_preprocessing.py` | an invalid exposure is rejected, parametrized: zero, negative, inf, NaN | a bad exposure reaches LightGBM, which accepts it | `> 0` → `>= 0`; drop `np.isfinite` |
| | the error names the invalid rows | an error the user cannot act on | drop the index labels |
| | the exposure is returned as given, with the table's index | a transformed or re-indexed exposure | return `np.log(exposure)`; reset the index |
| `test_modeling_pipeline.py` (new) | the templates stay unfitted | fitting the caller's objects in place | drop the model's `clone` |
| | doubling the exposure doubles the prediction | the exposure lost at predict | `predict` passes `np.ones(len(X))` |
| | rows are transformed with the training statistics (row by row equals the batch) | refitting the transformer at predict | refit a clone in `predict` |
| | predict follows the fitted copies after `set_params(model__use_exposure=False)` | reading the template at predict | use `self.model` in `predict` |
| | `uses_exposure` follows the model (`True`, `False`) | the base default withholding the exposure | delete the override |
| | an exposure for other rows (a shuffled Series) raises, at fit and at predict | an exposure applied to the wrong buildings | drop the check |
| `test_modeling_direct_cohort.py` | `test_exposure_misuse_raises` keeps `missing-while-on` and `given-while-off`. Its zero, negative and infinite cases are removed; the transformer's tests cover them | | |
| `test_modeling_contract.py` | an `EXAMPLES` entry for `ModelPipeline`, without an exposure (the refit test calls `fit(X, y)`). Discovery finds the class, so without it `test_every_shipped_model_has_an_example` fails | | delete the entry |

The pipeline tests can reuse the data of `test_modeling_utils.py`: a
`FeatureTransformer` with one `Center()` plan on `x`; a table with `x` and
`n_apartments`; a Poisson `y` proportional to `n_apartments`; and
`DirectCohortModel`. `n_apartments` is not a feature, so doubling is exact
(`rtol=1e-12`).

**Docs:**
- `MODULE_REFERENCE.md`: `ExposureTransformer` in the `preprocessing.py` row;
  `pipeline.py` replaces the `utils.py` row; `ModelPipeline` in the
  `modeling/__init__.py` row.
- `DIRECT_COHORT_MODEL.md` §0.3: the value check moved to `ExposureTransformer`.
- `MODEL_REIMPLEMENTATION_PLAN.md` M10: a one-line note pointing to N3.
- `HYPERPARAMETER_TUNING_PLAN.md` §6 (N20).
- Memory file `api-design-preferences.md`: already updated on 2026-09-29.
- Run every code block that is edited or added.

**Pitfalls met in the first version:**
- In tests, write `ColumnPlan(..., columns=("x",))`. A bare string passes at
  runtime but fails mypy.
- `tests/unit/test_transformer_spec.py` has 4 mypy errors that predate this
  step (near lines 86, 96, 105, 210). They are out of scope.
- The simulator is a top-level package: `from student_simulator.pipeline import
  StudentPopulationSimulator`, `from student_simulator.config import load_simulation_config`.
- To mutate the import guard, put the forbidden import under
  `if TYPE_CHECKING:`, so test collection still runs.
- To run a doc's code blocks: extract the section's `python` blocks and `exec`
  them in order in one namespace. Set `raw_table` from the simulator first,
  and `fit_df`, `train_df`, `valid_df` after the block that builds `table`.
- Filter LightGBM's log lines from a probe's output with `grep -v "^\[LightGBM\]"`.

**Suggested commits:**
1. A2a: `transformer.py`, `test_feature_transformer.py`,
   `test_transformer_spec.py`, `docs/FEATURE_TRANSFORMATIONS.md`.
   `refactor(feature_engineering): the exposure is the model's, not the transformer's`
2. Everything else.
   `feat(modeling): ModelPipeline fits a feature transformer and a model on the raw table; preprocessing validates the exposure`

Done when:
- [x] `modeling/utils.py` and `test_modeling_utils.py` are gone.
- [x] One mutation check per test fails as expected.
- [x] The review's findings are reproduced, then fixed or rejected.
- [x] mypy, ruff and the non-slow suite pass: 1093 passed (1 skipped, 1 xfailed).

**Record of the revised A2 (2026-09-29).**
- **Baseline** before the first edit: 1083 passed. After: 1093 (+6 exposure
  tests, +8 pipeline tests, +5 contract cases for `ModelPipeline`; −6 tests of
  the deleted `utils.py`, −3 value cases of `test_exposure_misuse_raises`).
- **Verified first** (sklearn 1.9.0): the 4 contract checks pass on a class
  whose nested estimators are required arguments; nested `set_params` reaches
  only the copy; LightGBM accepts a zero exposure's `-inf` offset silently.
- **Mutation checks**, each failing its named test: `> 0` → `>= 0` (zero);
  `np.isfinite` dropped (inf; NaN is still caught by `> 0`); index labels
  dropped; the `[:5]` limit dropped; `np.log` returned; index reset; the model's
  `clone` dropped; `np.ones` at predict; the transformer refitted at predict;
  the column always read; `"n_apartments"` hard-coded; the template read at
  predict; the `exposure=` check dropped; the column read unvalidated; the
  `ModelPipeline` example deleted (the coverage test and its 5 cases).
- **Ran:** the evaluator's docstring example, as written, under `-W error` (3
  trials on a simulated table).
- **Review** (independent subagent; 16 mutations of its own, pickling and
  nullable dtypes checked):
  - *Fixed.* Nothing tested the "first 5 labels" limit; the test now has 6
    invalid rows. *Fixed.* The message read "1 rows are not"; now "1 invalid,
    first at rows [...]".
  - *Not changed.* Two columns with the exposure's name raise pandas'
    "truth value is ambiguous": it raises, so nothing passes silently.
  - *For A3.* `ModelPipeline.fit` fits a `y` with another index by position,
    silently. N19 covers only the multi-cohort classes.
  - *Open, for the user.* `astype(float)` does not reject every non-numeric
    column, as assumed in Build item 1: a numpy **bool** column passes as all
    1.0, and digit strings are parsed. A nullable `boolean` column with a
    `False`, and a non-digit string, do raise.
- **Revision 2 (user, 2026-09-29).** `ModelPipeline` takes the exposure as an
  argument (N17, N3 d) and drops `exposure_column`; `DirectCohortModel`'s local
  `n` is `exposure_values`, and the tests' `N` is `EXPOSURE` (§3 rule 8).
  - Pipeline tests: 4 dropped with the behavior they covered (no column
    needed, `exposure_column`, an `exposure` argument raises, an invalid
    exposure raises), 3 added (`uses_exposure` ×2, the index check).
  - Mutations, each failing its test: the model's `clone` dropped; `np.ones`
    at predict; the transformer refitted at predict; the template used at
    predict; `uses_exposure` returning `False`; the index check dropped.
  - Review (subagent), each finding reproduced:
    - *Fixed.* A 2-D exposure (a one-column DataFrame) broadcast into an
      (n, n) prediction, silently; older than A2. `DirectCohortModel` now
      raises unless the exposure is 1-D (new case `two-dimensional`; its
      mutation fails it).
    - *Fixed.* `base.py`'s `uses_exposure` docstring still named the column.
    - *Documented.* `ModelPipeline.uses_exposure` reads the template, so after
      `set_params` it describes the next fit; `predict` then raises, never
      misuses the exposure. A3's aggregator routes by it at fit and predict.
    - *Resolved before A3.* `ModelPipeline.fit` paired a `y` with another index
      by position. It now raises (N19): `test_a_y_for_other_rows_raises_at_fit`,
      whose mutation (skip the check) fails it and only it. Suite: 1094.
  - Suite: 1093 passed (1 skipped, 1 xfailed): 1092 after the redesign, +1 case.
- **Also left:** `HYPERPARAMETER_TUNING_PLAN.md` §5 still builds
  `exposure_train = train_df["n_apartments"]`; its block cannot run until
  `HyperparameterStudy` exists (Phase 3). The new §6 task covers it.

**Record of the first version (2026-09-29).** Facts that still hold:
- `clone` keeps the transformer's settings, `fit` accepts `y`, and `transform`
  keeps the index.
- A refit at predict changes the row-by-row predictions (by up to 9.7).
- All 12 §8 code blocks of `FEATURE_TRANSFORMATIONS.md` ran on a simulated
  table under `-W error`, after the A2a edits.
- The import guard's 3 mutations each failed it: `from ..models import base`,
  `from .. import models`, `from age_group_prediction.metrics import Metric`.
  The guard sees static imports only, as its comment says.
- Review finding, fixed: a section header in `test_transformer_spec.py` had
  been deleted along with the removed test.

### A3. `IndependentCohortModels`
*Revised by the user at the start of the step (N3 e, N21). Committed:
`6b234f0` (the exposure rule; rows paired by position), `3f46fbc`
(`IndependentCohortModels`).*

**Decisions at the start (user, 2026-09-29):**
- **Every model gets the same exposure.** A model applies it only if its
  setting says so (`use_exposure=True`); any other model ignores it. A model
  fitted with an offset still raises if it is missing. So the aggregator passes
  the exposure to every cohort and routes nothing, and an exposure passed when
  no cohort uses one is ignored too.
- **`uses_exposure` is removed** (base, `DirectCohortModel`, `ModelPipeline`):
  nothing reads it any more. No base `use_exposure` field either: sklearn's
  `get_params` reads each model's own `__init__`, and a model without an offset
  (`CohortProbabilityModel`) has no such setting.
- **A mapping of cohort models; each cohort tuned on its own** (N21). The plan
  had assumed nested `set_params` reaches the templates; it does not.
- **No index check** (N19 reversed at the end of the step): rows are paired by
  position, as in scikit-learn; `_check_aligned` removed from `ModelPipeline` too.

**A3a. A model ignores an exposure it does not use.**
- `modeling/direct_cohort.py`: `_check_exposure` returns `None` when no
  exposure is expected; it raises only when one is expected and missing, and
  checks that a used one is 1-D. `predict` still follows the fitted state.
- `modeling/base.py`, `modeling/pipeline.py`: `uses_exposure` removed; the
  `fit` docstring states the rule.
- Tests: `test_exposure_misuse_raises` loses `given-while-off` and matches each
  message; new `test_an_unused_exposure_is_ignored`; the 2 pipeline
  `uses_exposure` tests removed; the evaluator test matches the new message.
- Docs: `DIRECT_COHORT_MODEL.md` §0.2, §0.3; M10 note; `MODULE_REFERENCE.md`.

**A3b. `IndependentCohortModels`** (`modeling/independent_cohorts.py`, exported
from `modeling` only; a contract `EXAMPLES` entry).
- `fit(X, y, exposure=None)`: the mapping's keys equal `y`'s columns,
  duplicates counted (`Counter`); then
  `clone(model).fit(X, y[cohort], exposure=exposure)` per column,
  in `y`'s order; `cohort_models_` set once all succeeded.
- `predict(X, exposure=None)`: a DataFrame with `y`'s columns and `X`'s
  index; each prediction taken as an array.
- `ModelPipeline`'s index check and its 2 tests removed (N19); its docstring
  says rows are paired by position.

**Tests** (`tests/unit/test_modeling_independent_cohorts.py`), each with its mutation:

| Test | Mistake it catches | Mutation |
|---|---|---|
| each column equals that cohort's model fitted alone (under `filterwarnings("error")`) | a cohort fitted on another target | every cohort fitted on the first column; exposure dropped at predict |
| models that do not match `y`'s columns raise: missing, extra, duplicated | a cohort dropped silently; a duplicate reaching its model as a DataFrame | check dropped; a set check |
| a failing cohort leaves the previous fit intact | new and old cohorts paired | assigned cohort by cohort |
| columns follow `y`, rows follow `X` | misnamed cohorts, misaligned rows | mapping order; `index=None` |
| predict needs no target columns | reading a target from `X` | targets dropped from `X` at predict |
| doubling the exposure doubles only the cohorts with an offset | the exposure lost, or applied without an offset | exposure dropped at predict |
| predictions are placed by position, not by their own index | a Series prediction realigned into NaN | `np.asarray` dropped |

Done when:
- [x] The contract test discovers the class (mutation: its example deleted fails 6 cases).
- [x] Mutation checks, review, non-slow suite: 1104 passed (1 skipped, 1 xfailed).

**Record (2026-09-29).**
- **Baseline** before the first edit: 1094 passed (1 skipped, 1 xfailed).
- **Verified first** (sklearn 1.9.0): the 4 contract checks pass on an
  aggregator whose parameter is a dict of `ModelPipeline`s; `clone` deep-copies
  it (keys, order, settings); `get_params(deep=True)` lists only
  `cohort_models`, and a nested name raises `AttributeError`; the tuner passes
  one fixed exposure to every trial (`evaluator.py:104-122`), so before A3a a
  trial with `use_exposure=False` raised.
- **A3a mutations:** the exposure applied while off fails
  `test_an_unused_exposure_is_ignored`; the missing-exposure check dropped fails
  `missing-while-on` and the evaluator's test (without the `match`, the 1-D
  check would have caught `None` with another message).
- **Review** (independent subagent, 9 mutations of its own), each reproduced:
  - *Fixed, then withdrawn.* An exposure Series for other rows was used by
    position by cohort models that are not pipelines. A shared
    `_check_aligned` checked it; the user then dropped every index check (N19).
  - *Fixed.* `sorted` raised `TypeError` for mixed-type cohort names; now `Counter`.
  - *Fixed.* A cohort model returning a Series with its own index gave 100
    NaNs per column. Now taken as an array.
  - *Fixed.* `MODULE_REFERENCE.md` lacked the new module and names.
  - *Not changed.* `MODEL_REIMPLEMENTATION_PLAN.md` Steps 2.2 and 2.3 describe
    the finished Phase 2 as built; M10 carries the dated note.
  - *Not changed.* A nested `set_params` name raises `AttributeError`, not
    sklearn's "Invalid parameter". It raises, and the docstring says to tune
    each cohort on its own (N21).
- **User's questions at the stop**, answered with probes:
  - *Index check:* dropped (N19). −4 tests; suite 1104.
  - *`use_exposure=True` with `"regression"` still raises:* sklearn raises for
    settings that cannot be honored together (`LogisticRegression(penalty="l1",
    solver="lbfgs")`) and ignores merely irrelevant ones (`SVC(kernel="linear",
    gamma=...)`, silently; `l1_ratio`, with a warning). Ignoring it would fit
    without the offset while `get_params` says `use_exposure=True`; `objective`
    is not tuned, so the raise cannot end a study.
  - *`_check_exposure` stays:* at fit it reads `use_exposure`, at predict the
    fitted state (M10), so `expected` is an argument. Without its `None` check
    an offset model's predict gives all-NaN (`np.asarray(None)` is `nan`) and
    its fit a LightGBM `TypeError` about `init_score`.
- **Ran:** no doc code block was edited.

### A4. Smoke run and docs
*Committed: `4f083c2`. The close-out that follows it is recorded at the end of this step.*

**Facts from A3 that A4 builds on** (re-verify in plan mode):
- The API: `IndependentCohortModels({cohort: ModelPipeline(transformer,
  DirectCohortModel(...))})`, then `fit(table, Y, exposure=...)` and
  `predict(table, exposure=...)`, which returns a DataFrame with `Y`'s columns
  and the table's index. Import from `age_group_prediction.modeling`: the
  package root exports the **old** classes (§11).
- Every model gets the same exposure. `use_exposure=False` ignores it; a model
  fitted with an offset raises without one (N3 e). `use_exposure=True` with
  `"regression"` raises (kept by the user at the end of A3).
- Rows are paired by position; there is no index check (N19).
  `Splitter.train_test_split` splits `X`, `y` and `groups`, but **not the
  exposure**: take it as `exposure.loc[X_train.index]`, or by the same positions.
- Nested `set_params` names do not reach into the mapping (N21); each cohort is
  tuned on its own.
- `FEATURE_TRANSFORMATIONS.md` §8.1: `tree` keeps `n_apartments` as a feature
  *and* uses it as the offset, on purpose. Its second code block passes the raw
  `fit_df["n_apartments"]` as the exposure, bypassing `ExposureTransformer`;
  update it to build the exposure with `ExposureTransformer` (N3 d), and run it.
- `MODULE_REFERENCE.md` already has the `independent_cohorts.py` row and the
  new names (A3); A4 only checks it.
- For running doc blocks and the simulator, see A2's "Pitfalls met".

**Decisions to ask the user at the start of A4:**
- Smoke-run cohorts: `tree` for every cohort, with `use_exposure=True` for all,
  or both with and without the offset? (Recommended: both. It feeds §5 step 2
  of `MODEL_REIMPLEMENTATION_PLAN.md`, "re-check the exposure offset".)
- A baseline in the table? (Recommended: the constant rate
  `Σy / Σexposure × exposure` per cohort, so the deviances can be read.)
- Split and settings: grouped 80/20 by `neighborhood_id`, seeds 0–9, default
  (untuned) hyperparameters, as in §6?
- After A4: mark PR #10 ready for review and draft its description, with the
  PR #5 note from §2?

- **Smoke run** (scratchpad script, outside the repo): 10 simulated
  populations; per-cohort held-out Poisson deviance of `IndependentCohortModels`,
  and the deviance of the summed prediction against `n_children_total`. Put the
  table in the plan doc.
- **Docs:** `DIRECT_COHORT_MODEL.md` new §0.6 (`ModelPipeline`,
  `IndependentCohortModels`, and the data flow: `ShareTransformer` and
  `ExposureTransformer` on the full table, then the split);
  `FEATURE_TRANSFORMATIONS.md` §8.1 (combining the cohorts);
  `MODULE_REFERENCE.md` (check); `MODEL_REIMPLEMENTATION_PLAN.md` §5 step 3
  (Model A completed).

Done when:
- [x] Every edited code block has been run.
- [x] The user has seen the numbers and reviewed the docs; PR A is ready to merge.
  Merged 2026-09-30 as `27459eb` (PR #10, a merge commit; branch deleted).

**Decisions at the start (user, 2026-09-30):**
- Both variants: `tree` for every cohort, with `use_exposure=True` for all and
  with `False` for all.
- Each variant is read against **its own start**, the prediction before any
  tree: the constant mean `mean(y_train)` without the offset (LightGBM's
  `boost_from_average`), and the constant rate
  `Σy_train / Σexposure_train × exposure_test` with it (`exp(base_log_rate_)`).
  The total row has no baseline.
- Grouped 80/20 split by `neighborhood_id`, `random_state=seed`, seeds 0–9,
  default (untuned) hyperparameters.
- PR #10: the description is drafted in the scratchpad; the user applies it and
  marks the PR ready.

**Smoke run (2026-09-30).** `IndependentCohortModels` of three
`ModelPipeline(tree, DirectCohortModel(use_exposure=...))`, `tree` as in
`FEATURE_TRANSFORMATIONS.md` §8.1 (`n_apartments` kept as a feature). Per
population: simulate → `ShareTransformer` and `ExposureTransformer` on the full
table → `Splitter("grouped")` → `exposure.loc[X.index]` → fit and predict.
10 populations of 60 neighborhoods and 215–251 buildings; 179–206 training and
33–53 test buildings. Held-out mean Poisson deviance, lower is better, mean ± SD
over the populations. "Diff" is offset minus none, paired by population; "wins"
counts populations where the offset is lower. Calibration is the mean
prediction over the mean target, averaged over populations. The total row
scores the summed prediction against `n_children_total`.

| Target | No offset: constant mean | No offset: model | Offset: constant rate | Offset: model | Diff | Offset wins | Calibration none / offset |
|---|---|---|---|---|---|---|---|
| `n_kindergarten` | 3.394 ± 0.906 | 2.732 ± 0.911 | 2.739 ± 0.680 | 2.628 ± 0.877 | −0.105 ± 0.169 | 7 / 10 | 1.06 / 1.04 |
| `n_elementary` | 3.039 ± 0.387 | 2.124 ± 0.515 | 2.532 ± 0.492 | 2.128 ± 0.499 | +0.004 ± 0.200 | 6 / 10 | 1.00 / 1.03 |
| `n_highschool` | 3.349 ± 0.822 | 2.658 ± 0.703 | 2.936 ± 0.909 | 2.572 ± 0.710 | −0.086 ± 0.207 | 7 / 10 | 1.03 / 1.02 |
| total (sum) | — | 4.382 ± 1.360 | — | 4.215 ± 1.450 | −0.166 ± 0.315 | 8 / 10 | 1.03 / 1.02 |

**Reading.**
- The per-cohort models equal `MODEL_REIMPLEMENTATION_PLAN.md` Step 2.4's to
  the third decimal, diffs and wins included. The raw-table API
  (`ModelPipeline`, `IndependentCohortModels`, `ExposureTransformer`) fits the
  same models as the Phase 2 path, where the caller transformed the features.
- On average each variant beats its own start in every cohort: by 20–30%
  without the offset, and by 4–16% with it. The constant rate alone is already close to the
  model for kindergarten (2.739 vs 2.628): building size carries most of what
  the untuned trees find.
- The offset lowers the total's deviance by 4% (8 of 10), about 1.7 standard
  errors. Still weak evidence, and still untuned: §5 step 2 of
  `MODEL_REIMPLEMENTATION_PLAN.md` re-checks it after tuning.
- Calibration holds in both variants (1.00–1.06).
- The script was `smoke_a4.py` in that session's scratchpad, outside the repo
  (per-session, so it may be gone; the recipe above rebuilds it). It
  asserts the output's columns and index and that `exp(base_log_rate_)` equals
  the constant rate. It ran under `-W error`, silently.

**Record (2026-09-30).**
- **Baseline** before the first edit: 1104 passed (1 skipped, 1 xfailed).
- **Verified first:** the facts above hold. `train_test_split` returns 6
  pieces and not the exposure. One seed end to end under `-W error` gave a
  DataFrame with `Y`'s columns and the test index. A nested name
  `cohort_models__a__model__learning_rate` raises `AttributeError` ('dict'
  object has no attribute 'set_params'); `model__learning_rate` on a
  `ModelPipeline` works.
- **Found beyond the handoff, fixed:** `DIRECT_COHORT_MODEL.md` §0.2's block
  also passed the raw `n_apartments` column; `MODULE_REFERENCE.md` said every
  `modeling` class takes a transformed design matrix; the population size was
  wrong in §6 and in Step 2.4's record (seed 0's for every seed); PR #10's
  description still described `exposure_column` and `utils.py`.
- **Docs:** `DIRECT_COHORT_MODEL.md` (top note, §0.2 block, §0.5 paragraph, new
  §0.6); `FEATURE_TRANSFORMATIONS.md` §8.1 (the exposure block, and "Combining
  the cohorts" with a block); `MODULE_REFERENCE.md` (the section intro, the
  §0.6 link; rows checked, unchanged); `MODEL_REIMPLEMENTATION_PLAN.md` §5 steps
  2 and 3, Step 2.4's data line.
- **Ran:** all 13 python blocks of `FEATURE_TRANSFORMATIONS.md` §8 and both of
  `DIRECT_COHORT_MODEL.md` §0, in order in one namespace, under `-W error`
  (scratchpad `run_doc_blocks_a4.py`; `raw_table` from seed 0, `fit_df`/`train_df`
  and `valid_df`/`test_df` from a grouped split).
- **No `.py` file changed**, so no ruff, mypy or mutation checks.
- **Review** (independent subagent; the smoke table, its reading and every §0.6
  API claim reproduced by its own probes), each finding reproduced:
  - *Fixed.* Step 2.4's reading still said "196 training rows".
  - *Fixed.* §0.6's rule "`X` may keep the target columns" holds only with
    `remainder="drop"`: with `"passthrough"` the targets became features, and
    `predict` on a table without them raised "columns are missing". The rule
    now says so.
  - *Fixed.* §0.2's comment "§8.0 and §8.1" pointed at this doc's own §8
    (Metadata); it now names Feature transformations, and says `train_df` and
    `test_df` are a split of that table.
  - *Fixed.* `MODEL_REIMPLEMENTATION_PLAN.md` M10's input line and the §3
    usage block taught the raw column as the exposure; each now has a dated
    pointer to §0.6. `HYPERPARAMETER_TUNING_PLAN.md` §5 is left: its §6 task
    covers it (A2).
  - *Fixed.* Wording: "Next" and "✓ complete" no longer read as if PR #10 were
    merged.
  - *Open, outside A4 (no `.py` change in a docs step).* `FeatureTransformer`'s
    class docstring (`feature_engineering/transformer.py:124-125`) says the
    model "takes the raw column from the table itself", stale since A2. The
    evaluator's docstring example builds the exposure on `train_df`, not the
    full table: harmless (row-wise), but unlike the documented flow.

**Close-out of PR A (2026-09-30, user's request before marking PR #10 ready).**
- **Stale statements fixed**, found by two read-only sweeps (code; docs), each spot-checked:
  - Code (comments, docstrings and one message; no behavior change):
    `FeatureTransformer`'s class docstring (the model "takes the raw column");
    the evaluator's docstring example (the exposure built on the full table,
    then `.loc[train_df.index]`; run as written, 3 trials, `-W error`) and its
    stale `y` comment; `DirectCohortModel`'s missing-exposure message now says
    `use_exposure=True` *at fit*, true at predict too; `IndependentCohortModels`'
    alias comment says "usually a ModelPipeline". Leftover `.pyc` files of the
    deleted `modeling/utils.py` and `modeling/metrics.py` removed.
  - Docs: `FEATURE_TRANSFORMATIONS.md` §4.4, §4.7, §5 (the old `FeatureSpec`'s
    checks marked as old), §8.1 ("negative"; measured through
    `DirectCohortModel`, zero, negative and inf are accepted with only a numpy
    warning and NaN silently, so "silently" became "at most a numpy warning",
    also in `DIRECT_COHORT_MODEL.md` §0.6), §8.2 (the exposure built with
    `ExposureTransformer`); `HYPERPARAMETER_TUNING_PLAN.md` (D13, §4.2, status,
    the §6 task now also rewrites §5's raw-column example); `SPLITTING.md` §2;
    `docs/README.md`; `MODULE_REFERENCE.md`; `MODEL_REIMPLEMENTATION_PLAN.md`
    (the base-signature note).
- **Part B brought up to date** with what PR A shipped: §5's length question
  (measured evidence), §6 (the first-version probe marked historical), §7 (the
  base signature on every Model 2 class; the usage block splits and takes the
  exposure by labels; its Model 1 part run under `-W error`), B0 (what to
  re-verify), B1 (the exposure rule to reuse), B2/B4/B5 (exposure tests,
  contract examples with `use_exposure=False`, A3's aggregator lessons), B3
  (no line numbers), B7 (A4's script and reference), B9 (§8.2/§8.3 blocks,
  the §0.6 pattern), §11 (three pitfalls).
- **Review** (independent subagent; the new message, the evaluator docstring
  run, §5's evidence, nested `set_params` on two named estimators and the §7
  Model 1 block reproduced), each finding reproduced, all fixed: a duplicated
  sentence in `SPLITTING.md`; §4's "line 83"; §4.7 describing the not-yet-built
  probability model in the present tense; "only" in §4.4; the per-session
  scratchpad scripts named as if they persist (B7, B9, §11 now give the
  recipe); an over-broad claim and "New classes" in the PR body; "silently"
  (measured above); the evaluator's "the splitter"; the alias comment; an
  indent; a leftover `metrics` `.pyc`.
- **Checks:** ruff and mypy on the 4 changed `.py` files; their tests and the
  `modeling` tests under `-W error` (178 passed); the evaluator docstring, the
  §8/§0 doc blocks and §7's Model 1 part run under `-W error`; non-slow suite
  1104 passed (1 skipped, 1 xfailed).

## 9. Steps, PR B: Model 2

### B0. Branch and draft PR
*Handoff written 2026-09-30, at the end of the PR A session.*

**Already done:** the branch `feat/independent-total-probability-model`, from
`feat/hyperparameter-tuning` at `27459eb` (PR A merged); its first commit is
this handoff (status lines in this doc, `docs/README.md` and
`MODEL_REIMPLEMENTATION_PLAN.md` §5).

**Left for B0** (a docs-only step; no code):
1. ✓ Confirm the branch, that its base is `27459eb`, that at most the handoff
   commit follows it, and a clean tree. Run the non-slow baseline (1104 passed,
   1 skipped, 1 xfailed).
2. ✓ Re-verify the facts below against the code, and fix any that are stale here.
3. ✓ The user pushes; then a **draft** PR `feat/independent-total-probability-model`
   → `feat/hyperparameter-tuning`. Its body lists B0–B9 (unchecked):
   [#11](https://github.com/galkampel/age-group-prediction/pull/11).
4. The user adds the PR #10 note to PR #5's description (§2), if not done yet.
   *Not done at B0's close (PR #5's body does not mention #10).*

Done when:
- [x] Branch, baseline and the facts below verified; draft PR #11 open.
- [ ] PR #5's note (the user's).

**Facts from PR A that Part B builds on** (re-verify; §8 A2–A4 hold the detail):
- **The base contract** (`modeling/base.py`): abstract `fit(X, y, exposure=None)`
  with `y: pd.Series | pd.DataFrame`, and `predict(X, exposure=None) -> np.ndarray
  | pd.DataFrame`; concrete `evaluate(y_true, y_pred, metric)`. Every Model 2
  class keeps the signature, even one that never uses an exposure.
- **The exposure** (N3): a model's own setting (`use_exposure`); the caller builds
  it with `preprocessing.ExposureTransformer("n_apartments").fit_transform(table)`
  on the full table before splitting and passes `exposure=` to every model. One
  that does not use it ignores it; one fitted with the offset raises without it.
  The rule is `DirectCohortModel._check_exposure(exposure, *, expected)` (B1).
  *Since B1: `BaseAgeGroupModel._check_exposure(X, exposure, *, expected)`,
  which also rejects a length other than `X`'s.*
- **Rows are paired by position** (N19); no index check anywhere.
  `Splitter.train_test_split` splits `X`, `y` and `groups` only: take the
  exposure as `exposure.loc[X_train.index]`. *Superseded by B0a:
  `train_test_indices` returns positions, and every array is taken by them.*
- **Classes and fitted state:** `DirectCohortModel` (`regressor_`,
  `base_log_rate_`); `ModelPipeline(feature_transformer, model)`
  (`feature_transformer_`, `model_`; fits clones on the raw table);
  `IndependentCohortModels(cohort_models)` (`cohort_models_`; a mapping, nested
  `set_params` does not reach into it, N21). Exported from
  `age_group_prediction.modeling` only; the package root exports the **old**
  classes (§11).
- **The contract test** (`tests/unit/test_modeling_contract.py`) discovers every
  concrete `BaseAgeGroupModel` in `modeling` and fails without an `EXAMPLES`
  entry (a factory returning `(model, X, y)`). It runs 4 sklearn checks and a
  refit test that calls `fit(X, y)` **without an exposure**, slicing `y` with
  `.iloc`. So `TotalChildrenModel`'s and Model 2's examples set
  `use_exposure=False`. `test_modeling_never_imports_the_old_stack` checks the
  forbidden imports (§4).
- **Aggregator lessons (A3):** clone in `fit`; assign fitted copies together
  only after all succeed; take sub-model predictions as arrays (a Series with
  its own index realigns into NaN); output columns from `y` at fit, index from `X`.
- **Docs pattern:** `DIRECT_COHORT_MODEL.md` §0.6 (data flow, one runnable block,
  a table of rules with reasons). Doc blocks are run by extracting a section's
  `python` blocks and `exec`-ing them in order in one namespace, with `raw_table`
  from the simulator and the split frames set by the runner (A2's "Pitfalls
  met", A4's record). Scratchpads are per session: rebuild scripts from the
  recipes, never write them in the repo.
- **Smoke-run recipe and reference** (B7): A4's "Smoke run"; Model 1's offset
  variant is the reference (total 4.215 ± 1.450).
- **Open in the tuning package, not PR B's:** `HYPERPARAMETER_TUNING_PLAN.md` §6
  "Evaluator on the raw table" (also rewrites its §5 example).

**Decisions to ask the user at the start of B1** (none are needed for B0):
- **Home of `minimize_lbfgs`** (N9, under a class; no utils file, N16), e.g. a
  shared abstract parent of the two GLMs.
- **Home of the exposure rule** for `TotalChildrenModel`: a copy of
  `DirectCohortModel._check_exposure`, or moved to a shared parent (which would
  touch `DirectCohortModel`).
- **Exposure length** (§5): reject an exposure whose length differs from `X`'s?
  With §5's measured evidence.

**Later:** before B8, confirm NB2 is still wanted (B8's note).

**Record (2026-09-30).**
- **Branch:** `27459eb` is an ancestor; only `0fd0fa1` (the handoff) follows;
  the tree was clean at the start; the remote head is `0fd0fa1`.
- **Baseline:** 1104 passed (1 skipped, 1 xfailed). Versions as in §6
  (sklearn 1.9.0, scipy 1.18.0, statsmodels 0.14.6), and LightGBM 4.7.0.
- **Verified**, every fact above, against the code (the docs facts against the docs): the base signatures
  (`base.py`); `DirectCohortModel._check_exposure(exposure, *, expected)`;
  `ModelPipeline` and `IndependentCohortModels` (clones, fitted copies, arrays,
  `y`'s columns and `X`'s index, no index check); `ExposureTransformer`;
  `Splitter.train_test_split` (6 pieces, no exposure); the contract test
  (`EXAMPLES`, the refit test's `fit(X, y)` with `.iloc`, the old-stack guard);
  the evaluator (`_check_exposure` rejects a given exposure whose shape is not `(len(X),)`;
  `evaluate`'s `y` is `pd.Series | np.ndarray`); `utils.Target` exists for B3.
- **§5's length evidence, re-measured:** `DirectCohortModel.predict` broadcasts
  a length-1 exposure silently; a wrong length at fit fails in LightGBM
  ("Initial score size doesn't match data size"), at predict in numpy; a GLM's
  `exposure * exp(D @ b)` broadcasts a length-1 exposure silently.
- **Also checked, unchanged:** §4's `total_base` and `composition_base`
  (`FEATURE_TRANSFORMATIONS.md` §8.2, §8.3; no model and no pipeline block
  yet, B9's work); §8.7 item 7 exists (B9); B9's conversion (C in
  [0.01, 100] is `l2_penalty` in about [2e-6, 2e-2] at ~4,500 training
  children); the
  old total model's penalty is N8's (mean negative log-likelihood, intercept
  unpenalized).
- **For B2:** the statsmodels precedent in `tests/validation` is
  `test_recovery.py`: a module-level `pytest.importorskip("statsmodels")`, its
  tests marked `slow`.
- **Stale here, fixed:** the status line and §2 (the draft PR is #11).

### B0a. The splitter returns row positions
*Added by the user at the start of B1 (2026-09-30).*

**Decision.** The exposure was the one array taken by label
(`exposure.loc[X_train.index]`), against N19's by-position rule, because
`Splitter.train_test_split` split only `X`, `y` and `groups`. The user chose,
over an `exposure=` argument (8 return pieces) and keeping `.loc`: a method
that returns the positions, as scikit-learn's splitters and `cv` do; then
`train_test_split` is **removed**. Callers take every array with `take_rows`.

**Build.** `Splitter.train_test_indices(X, groups, *, test_size, random_state)
-> (train_index, test_index)`: the holdout logic `train_test_split` had,
returning positions. `train_test_split` removed; no module in `src` called it.

**Tests** (`tests/unit/test_splitters.py`): every `train_test_split` test moved
to the indices; the two tests of the 6-tuple became:

| Test | Mistake it catches | Mutation |
|---|---|---|
| every row lands on exactly one side | a row lost, or in both halves | `train_index[1:]` returned |
| the indices are positions whatever the index (labels from 100 give the same indices as after `reset_index`) | labels returned as positions, which take other rows silently once in range | `X.index[...]` returned |

**Docs.** `SPLITTING.md` §2 and §5; `DIRECT_COHORT_MODEL.md` §0.2 and §0.6
(data flow, block, rules table); `HYPERPARAMETER_TUNING_PLAN.md` §5's block and
the §6 task; this doc (status, N19, §7, B0, B7). Docstrings: `Splitter`,
`ModelPipeline`, and the evaluator's (docstring only, since it named the
removed method). Left: `FEATURE_TRANSFORMATIONS.md` §8.1 (it does not split;
`.loc` by the reader's `fit_df` labels is right there); `SPLITTING_FIX_PLAN.md`
(a finished plan, history).

Done when:
- [x] Mutations, review, doc blocks, non-slow suite: 1104 passed (1 skipped,
  1 xfailed); the tests were replaced one for one.

**Record (2026-09-30).**
- **Baseline** before the first edit: 1104 passed (1 skipped, 1 xfailed).
- **Mutations**, each failing its named test and restored: `train_index[1:]`
  fails `every row lands on exactly one side` (3) and 3 others; `X.index[...]`
  returned fails `the indices are positions whatever the index` (3) on its
  assertion (113 vs 13).
- **Checks:** ruff and ruff format on the 4 changed `.py` files; `uv run mypy`
  clean. `test_splitters.py` has 13 mypy errors, all older kinds
  (`method: str` for a `Literal`, an `np.int64` seed). HEAD's version had 25;
  the new annotations removed 12. The changed tests pass under `-W error`.
- **Ran** under `-W error`, one namespace, seed 0 (scratchpad
  `run_doc_blocks_b0a.py`: `table` and `tree` from `FEATURE_TRANSFORMATIONS.md`
  §8.0–§8.1). These were `DIRECT_COHORT_MODEL.md` §0.2 and §0.6, `SPLITTING.md`
  §2 (a `DummyRegressor`), §7's Model 1 part, and the evaluator's docstring
  example (3 trials). `HYPERPARAMETER_TUNING_PLAN.md` §5 cannot run until
  `HyperparameterStudy` exists.
- **Review** (independent subagent), each finding reproduced:
  - *Verified, no change.* The splits are unchanged: HEAD's `train_test_split`
    equals `take_rows` with the new indices, for 3 methods, 25 seeds and 2 test
    sizes, on a shuffled index (0 mismatches).
  - *Fixed.* `the indices are positions` caught the labels mutant only because
    `take_rows` raised `IndexError` on 24 rows. Once the labels fall in range,
    labels used as positions take other rows silently: reproduced on 180 rows,
    where rows 200–202 were taken for positions 0–2. The test now compares the
    indices with those after `reset_index`.
  - *Fixed.* `SPLITTING.md` §2 took `take_rows(groups, …)` right after saying
    `random` takes `groups=None`, but `take_rows(None, …)` raises `TypeError`
    (reproduced). A new hazard in §5: a fold's exposure comes from
    `exposure_train`.
  - *Fixed.* N19's "Why" cell still named `train_test_split`. §0.6's step 2
    listed groups that its block does not take. Wording in the `ModelPipeline`
    and evaluator docstrings; three lines too long. The same-seed test now
    compares both halves.

### B1. Shared numerics
*Handoff written 2026-09-30, at the end of the B0/B0a session. The decisions
below were made by the user at the start of B1; the facts were measured then.
Re-verify them in plan mode, write a short summary, ask for approval.*

**Decisions at the start (user, 2026-09-30):**
- **The minimizer is a component, not a model**, so it is its own class:
  `LBFGSMinimizer`, not a model parent. (User: a base class if the shared thing
  is a model; a new class if it is a component.) A GLM model base is added only
  if B2/B4 show model logic both GLMs share.
- **The exposure rule moves to the base**, with only the relevant checks:
  `BaseAgeGroupModel._check_exposure`. `DirectCohortModel` drops its own copy.
- **The length check** (the user asked for the best practice): a model with an
  offset rejects an exposure whose length differs from `X`'s. sklearn checks
  every per-row array's length at runtime in `fit` (`check_consistent_length`,
  `_check_sample_weight`); a test only shows the check exists. Positions from
  `train_test_indices` (B0a) remove the usual source of a mismatch, not a
  caller's mistake. `X` against `y` needs no new check: LightGBM checks it for
  `DirectCohortModel`, and `validate_data(X, y)` for the GLMs (B2, B4).
- The splitter change came first, as B0a.

**Facts measured at the start** (read-only probes; re-verify):
- sklearn's own L-BFGS-B mapping, in `PoissonRegressor` (`sklearn/linear_model/_glm/glm.py`,
  the `minimize` call) and `LogisticRegression` (`_logistic.py`):
  `maxiter=max_iter`, `gtol=tol`, `ftol=64 * np.finfo(float).eps`, `maxls=50`.
- Against a tight `PoissonRegressor(tol=1e-12)` (simulated seeds 0–9; 7
  standardized features: `ses`, `avg_household_size`, `median_age`,
  `n_daycares_500m`, the 4/5/6-room shares; `y = n_children_total`,
  exposure `n_apartments`; λ in {0, 0.01, 1}; oracle `alpha = λ / mean(exposure)`,
  `fit(X, y / exposure, sample_weight=exposure)`), worst max coefficient
  difference: scipy defaults 1.5e-5 (the old 9e-6: the same order); sklearn's
  mapping with `gtol` 1e-4: 5.8e-6, 1e-5: 4.7e-7, **1e-6: 6.1e-8**. So the
  GLMs' default is `tol=1e-6`. `check_grad` on the Poisson objective: 4e-7.
- An iteration limit: `success=False`, status 1,
  "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT".
- An overflow (features × 1e3): without `errstate`, scipy returns
  `success=False`, "ABNORMAL", `fun=nan` and `x` = the start point. Under
  `np.errstate(over="raise", invalid="raise")` a `FloatingPointError` escapes
  `minimize`.
- `sklearn.utils.validation.check_consistent_length(X, exposure)` ignores
  `None` and rejects a length-1 or any other wrong-length exposure
  ("Found input variables with inconsistent numbers of samples: [5, 1]").
  `validate_data(estimator, X, y)` rejects a wrong-length `y` the same way.
- Elsewhere: sklearn rejects a wrong-length `sample_weight` array (length 1
  included; a scalar is accepted); statsmodels `GLM` rejects a wrong-length
  exposure at fit and broadcasts a length-1 one at predict.

**Build:**
1. **`modeling/optimization.py` (new): `LBFGSMinimizer`**, a frozen dataclass
   `LBFGSMinimizer(max_iter: int, tol: float)`, not an estimator.
   - `minimize(objective, start, bounds=None) -> np.ndarray`; `objective`
     returns `(value, gradient)` (`jac=True`); `bounds` for B8's `log α`.
   - Options: sklearn's mapping above, with the reason in a comment.
   - Under `np.errstate(over="raise", invalid="raise")`; a `FloatingPointError`
     becomes `RuntimeError` naming feature scale as the likely cause (N9).
   - `RuntimeError` when `not result.success` (quote scipy's message, suggest a
     larger `max_iter`), and when the value or `x` is not finite. No clipping.
   - No defaults: the GLMs own `max_iter=500`, `tol=1e-6` and build a minimizer
     in `fit`. Not exported from `modeling/__init__.py` (an internal component);
     `__all__` in the module; a row in `MODULE_REFERENCE.md`.
2. **`BaseAgeGroupModel._check_exposure(X, exposure, *, expected) -> np.ndarray | None`**
   (`modeling/base.py`), a protected static method; the rule PR A settled, plus
   the length:
   - `None` when not `expected` (a passed exposure is ignored, N3 e);
   - `ValueError` when expected and missing (the offset would drop silently);
   - floats; `ValueError` unless 1-D (an (n, 1) broadcasts to (n, n));
   - `check_consistent_length(X, exposure_values)`.
   `DirectCohortModel` (`modeling/direct_cohort.py`) deletes its
   `_check_exposure` and calls the base's with `X`, at fit (`expected` =
   `use_exposure`) and at predict (`expected` = fitted state). Its docstring's
   "LightGBM itself rejects ... a wrong-length exposure at fit" is updated.
   **Behavior change:** `predict` rejects a length-1 exposure.

**Tests** (each names its mistake; one mutation each):

| File | Test | Mistake it catches | Mutation |
|---|---|---|---|
| `test_modeling_optimization.py` (new) | an iteration limit raises `RuntimeError` | a non-converged fit used silently | drop the `success` check |
| | an overflow raises `RuntimeError` naming feature scale | the start point returned as a fit (N9) | drop `errstate` |
| | a non-finite objective raises | a nan fit returned | drop the `isfinite` check |
| `test_modeling_direct_cohort.py` | `test_exposure_misuse_raises` + `wrong-length` cases: length 1 and n−1, at fit and at predict | an exposure broadcast, or paired wrongly | drop the length check |
| | the existing `missing-while-on`, `two-dimensional` and `test_an_unused_exposure_is_ignored` pass unchanged | the move changed the rule | (existing) |

The contract test is unchanged: no new model class, and an abstract or
non-model class is not discovered.

**Docs:** this doc (B1 record, the `tol` mapping, Done-when); §7's
`TotalChildrenModel` comment (the base rule); `DIRECT_COHORT_MODEL.md` §0.3 if it
states the length behavior; `MODULE_REFERENCE.md`. Run any edited block.

Done when:
- [x] The default `tol` reproduces the sklearn oracle to 1e-6, and how `tol` maps
  to scipy's options is recorded (measured above: `gtol=tol`, `ftol=64·eps`,
  `maxls=50`; `tol=1e-6` gives 6.1e-8). B2's oracle test pins it.
- [x] Mutations, review, non-slow suite: 1115 passed (1 skipped, 1 xfailed).

**Record (2026-09-30).**
- **Baseline** before the first edit: 1104 passed (1 skipped, 1 xfailed).
  Versions: sklearn 1.9.0, scipy 1.18.0, LightGBM 4.7.0, numpy 2.5.1.
- **Verified first**, every fact above: sklearn's options (`_glm/glm.py:282-291`,
  `_logistic.py:587-594`); the oracle figures on seeds 0–9 exactly (1.5e-5,
  5.8e-6, 4.7e-7, 6.1e-8); `check_grad` 6.4e-7 (4e-7 above: the same order);
  the iteration-limit and overflow behavior; `check_consistent_length`.
  **New:** a NaN or ±inf objective with a zero gradient returns `success=True`
  ("CONVERGENCE"), so the finite-objective check is needed on its own.
- **Built** as planned. `LBFGSMinimizer` has the aliases
  `ObjectiveWithGradient` (not `Objective`, which `direct_cohort` exports) and
  `Bounds`. `DirectCohortModel.fit`'s docstring keeps what LightGBM still
  rejects (an unknown objective, an all-zero `y`).
- **Tests.** `test_modeling_optimization.py` (8): the minimum is returned;
  bounds are honored; the gradient at the minimum is within `tol`; an
  iteration limit, a failed line search (without "raise max_iter"), an
  overflow, an invalid value and a non-finite objective each raise.
  `test_modeling_direct_cohort.py`: `test_exposure_misuse_raises` gains
  `length-one` and `one-row-short`; new `test_a_wrong_length_exposure_raises_at_predict`
  (2); the `wrong-length` case of `test_invalid_input_surfaces_an_error` moved
  out (it is our check now, not LightGBM's).
- **Mutations**, each failing its named test and restored: the `success` check
  dropped; `errstate` dropped; `invalid="raise"` dropped; the finite check
  dropped; the start returned; `bounds` dropped; `gtol` dropped and `ftol`
  dropped (each stops at a gradient of 6.4e-6); the advice always "raise
  max_iter"; `check_consistent_length` dropped (4 cases, fit and predict).
- **Checks:** ruff and ruff format on the 5 changed `.py` files; `uv run mypy`
  and mypy on `modeling` plus both test files clean; the `modeling` tests
  under `-W error` (126 passed).
- **Review** (independent subagent, its own mutations and probes), each
  finding reproduced:
  - *Fixed.* Nothing pinned `tol → gtol`: dropping `gtol` or `ftol` passed.
    New test: the gradient at the returned point is within `tol`.
  - *Fixed.* An abnormal stop (status 2, e.g. a NaN gradient) advised "raise
    max_iter"; the advice now depends on the status.
  - *Fixed.* An invalid value was reported as an overflow, and was untested;
    the message names both, and a test covers it.
  - *Fixed (dropped).* The `isfinite(x)` half was unreachable: an all-zero
    Poisson `y` stops at an intercept of −14 with `success=True`, not −inf.
    B2's own check on `y` covers that case.
  - *Fixed.* The test comments claimed a wrong length passes silently at fit;
    LightGBM rejects it there. They now give the reason (scikit-learn checks
    every per-row array; numpy broadcasts a length-1 exposure at predict).
  - *Fixed.* `MODEL_REIMPLEMENTATION_PLAN.md` Step 2.2, its review note and
    test 6 left the wrong length to LightGBM: dated notes added.
  - *Not changed.* The length error is sklearn's own message ("Found input
    variables with inconsistent numbers of samples: [50, 1]"), capitalized and
    without the word exposure: `check_consistent_length` was the user's choice.
  - *Noted for B2/B4.* `scipy.special` functions (`xlogy`, `gammaln`) ignore
    `np.errstate`, so a NaN from them is not raised; a NaN objective at the end
    is still caught. A wrong analytic gradient can return `success=True` at a
    wrong point: only the `check_grad` tests (N6) catch it.
- **Docs:** `DIRECT_COHORT_MODEL.md` §0.3; `MODULE_REFERENCE.md` (`base.py`
  row, new `optimization.py` row); `MODEL_REIMPLEMENTATION_PLAN.md` (notes).
  No code block was edited.

### B2. `TotalChildrenModel`, Poisson
- **Files:** new `modeling/total_children.py`, `__init__.py`, new
  `tests/unit/test_modeling_total_children.py`, contract example; a statsmodels
  check in `tests/validation/`.
- **Build:** N6, N8–N10. Input checks, each silent otherwise:
  `sklearn.utils.validation.validate_data` for `X` (reordered columns at
  predict), finite non-negative counts, at least one child (an all-zero `y`
  returns `success=True` with intercept `−inf`).
  *Changed at the start of B2 (user): no count check and no `family`; see the
  record.*
- **Tests:**
  1. matches `PoissonRegressor` at penalties 0, 0.01 and 1 (wrong gradient or penalty scale);
  2. `check_grad` on the objective;
  3. a huge penalty gives `exposure · Σy/Σexposure` (penalized intercept);
  4. doubling the exposure doubles the mean;
  5. `predict` follows the fitted state after `set_params(use_exposure=False)`;
  6. all-zero `y`, negative counts, reordered columns, unknown family raise;
  7. beats the constant-rate baseline on informative data;
  8. the exposure rule (A3, B1): a missing exposure raises at fit and at
     predict, a 2-D or wrong-length one raises, and one passed with
     `use_exposure=False` is ignored (the same predictions as without it).
- **Contract example:** `TotalChildrenModel(use_exposure=False)`. The refit
  test calls `fit(X, y)` without an exposure, so the default `True` would
  raise there.

**Decisions at the start (user, 2026-09-30):**
- **No `family` setting yet.** B8 adds `family: Literal["poisson", "nb2"]`
  and its unknown-family test; if NB2 is dropped, no one-valued setting is left.
- **No count check on `y`.** Measured: a NaN or infinite `y` is rejected by
  scikit-learn's input check; a negative or non-integer count makes the
  objective infinite, and `LBFGSMinimizer` raises ("non-finite objective").
  Nothing passes silently, and data values are validated in preprocessing.
  Only the all-zero `y` is checked: it "converges" silently to an intercept of
  −17.6 (not `−inf`, as written above). *Dropped at B4 (user, 2026-10-01):
  without the check an all-zero `y` predicts 0, the right limit; the test
  case went with it, and the failed-refit test uses `max_iter=1`. Also at
  B4: the objective moved from `poisson.logpmf` + a written gradient to
  torch's `Poisson.log_prob` with autograd (N6); `_objective` takes tensors,
  and the gradient test builds them.*
- **`maxfun` fixed here** (open from B1): scipy's evaluation cap also ends with
  status 1, where "raise max_iter" would mislead.

**Revision before the commit (user's notes, 2026-10-01):**
- **`solver` is a model setting**, as scikit-learn's: `solver: Literal["lbfgs",
  "bfgs"] = "lbfgs"`, mapped to scipy's method. `LBFGSMinimizer` became
  `Minimizer(method, max_iter, tol)` over a table of methods, each mapping
  `tol` to its own gradient tolerance. Measured (seeds 0–9, λ ∈ {0, 0.01, 1},
  worst coefficient difference from `PoissonRegressor(tol=1e-12)`): L-BFGS-B
  with sklearn's options 6.1e-8 (15 evaluations, 1.1 ms); with scipy's
  one-size `tol=1e-6` **2.7e-4** (it also sets `ftol`); BFGS 6.2e-8 (20, 1.6 ms);
  CG 1.2e-7; TNC 8.7e-5; trust-constr 6.6e-8 but 19 ms; Newton-CG with a
  Hessian failed 2 of 30; trust-exact with a Hessian 6.2e-8 in 5 evaluations.
  So only L-BFGS-B (the one with bounds, for B8) and BFGS are allowed; no
  Hessian methods (nothing to gain at ~1 ms per fit; NB2's Hessian is work).
  An unknown method raises; bounds with BFGS raise (scipy would only warn).
- **No stacked `[1, X]` matrix**: the objective takes `X` with the intercept a
  separate parameter, `μ = exp(offset + b + Xβ)`, gradient
  `[mean(μ − y), Xᵀ(μ − y)/N + λβ]`.
- **Without an exposure** the offset is 0 and the start intercept `log(mean y)`,
  stated explicitly. "Offset" is the GLM term for `log(exposure)` (statsmodels
  uses both words the same way), so the name stays.
- **Input checks, the fewest** (user's question: are they necessary, and in
  every model?): measured without them, a NaN in `X` or `y` ends with
  `success=False` (the minimizer raises) and every other bad input raises a
  numpy `TypeError`; nothing is silent, but a bool column would raise instead
  of being converted. So `check_X_y` stays at the top of `fit` for the float
  conversion (it records nothing); the column names are **fitted state**,
  recorded after success with `coef_` (`feature_names_in_`, `n_features_in_`)
  by `validate_data(self, X, reset=True, skip_check_array=True)`, which
  `validate_data(reset=False)` reads at predict (reordered columns would
  otherwise be silent). No base-class helper: `DirectCohortModel` gets both
  from LightGBM; B4 repeats the three lines, and a GLM base is proposed then
  if the repetition warrants it.
- Tests: the oracle test runs for both solvers; new: an unknown method, bounds
  with BFGS, an unknown solver, the gradient-within-`tol` test for both
  methods. 25 mutations, each failing its named test, among them: the BFGS
  entry without `gtol`; `"bfgs"` mapped to an unknown method; the names not
  recorded (reordered columns then pass with only a warning).
- **Review of the revision** (independent subagent; `check_grad` at 30 points
  with and without the offset, 5.2e-8; both solvers on the simulated table
  unstandardized, seeds 0–9: within 1e-5 of the oracle, BFGS 2–4× faster
  there), each finding reproduced:
  - *Fixed.* The names were recorded by hand (`np.asarray(X.columns)`), which
    is only half of sklearn's rule: it records names only when all are
    strings, and none for an ndarray. Integer column names then warned at
    predict on the training frame itself, and an ndarray `X` fitted fully and
    then failed on `.columns`. Now sklearn's own call records them, after
    success.
  - *Fixed.* No test told the two solvers apart (both mapped to L-BFGS-B
    passed everything). New test: `solver="bfgs"` passes `method="BFGS"` to
    scipy (recorded by monkeypatching). A test comment claiming the oracle
    test catches an unmapped tolerance was wrong for BFGS; reworded.
  - *Fixed.* An unused `type: ignore` in the `maxfun` test; the two tables are
    keyed by `Solver` and `Method`.
  - *Not changed.* The `arg-type` ignores in the tests are dead under the
    project's mypy (the package resolves to `Any` from tests) but needed with
    `MYPYPATH=src`; same as `test_modeling_direct_cohort.py`'s.

Done when:
- [x] Matches `PoissonRegressor` (1e-6, both solvers) and statsmodels (1e-6);
  mutations, review, non-slow suite: 1147 passed (1 skipped, 1 xfailed).

**Record (2026-09-30).**
- **Baseline** before the first edit: 1115 passed (1 skipped, 1 xfailed).
- **Verified first:** `validate_data` rejects NaN in `X`, a wrong-length, 2-D
  or NaN `y`, and at predict reordered, renamed or extra columns (an ndarray
  only warns); the oracle conversion (6.1e-8, B1) and statsmodels `GLM` with
  `exposure=` (1.3e-9); a negative `l2_penalty` passes silently (the fit moves;
  scikit-learn rejects `alpha < 0`), so it is checked; `max_iter ≤ 0` already
  raises and `tol ≤ 0` is harmless, so neither is.
- **Built:** `TotalChildrenModel(*, use_exposure=True, l2_penalty=0.0,
  max_iter=500, tol=1e-6)`; the objective is the protected static method
  `_objective(parameters, design, y, offset, l2_penalty)`; fitted `intercept_`,
  `coef_`, `use_exposure_` (what `predict` follows), and the feature names and
  count. `LBFGSMinimizer`: `maxfun = (max_iter + 1)(2·maxls + 1)`.
- **Tests:** `test_modeling_total_children.py` (17): the two oracle tests (λ in
  {0, 0.01, 1} with the exposure, 0.01 without), `check_grad`, the huge
  penalty, doubling, the fitted state, the three silent inputs (all-zero `y`,
  negative penalty, reordered columns), a failed refit, the constant-rate
  baseline (deviance below 0.8 of it on new buildings), the exposure rule (3
  cases at fit and at predict) and an ignored exposure. The `maxfun` test in
  `test_modeling_optimization.py`; the contract example (5 cases);
  `tests/validation/test_total_children.py` (slow; statsmodels, seed 0).
- **Mutations**, each failing its named test and restored: the penalty doubled;
  the gradient without `/N`; the penalty's gradient dropped; the intercept
  penalized; the offset dropped at predict; the template read at predict; each
  of the three checks dropped; the names recorded before the fit; the start
  returned; the exposure always expected at fit; `maxfun` for one line search
  per iteration; the contract example deleted.
- **Checks:** ruff and ruff format on the changed `.py` files; `uv run mypy`
  and mypy on `modeling` plus the 4 test files; the `modeling` tests under
  `-W error` (149 passed); the statsmodels test under `-W error`.
- **Review** (independent subagent; its own mutations and probes), each
  finding reproduced:
  - *Fixed.* A failed refit left mixed state: `validate_data` recorded the new
    column names before the fit could fail, so `predict` then ran the old
    coefficients on the new columns, silently (and after a failed first fit
    raised `AttributeError`, not `NotFittedError`). This plan had assumed it
    would raise. Now `check_X_y` checks the data without touching the model,
    and `validate_data` records the names only after success. New test: a
    failed refit leaves the previous fit intact.
  - *Fixed.* The `maxfun` bound was not strict: after a failed line search
    L-BFGS-B resets and searches again, so an iteration can use about
    2·maxls evaluations. Now `(max_iter + 1)(2·maxls + 1)`.
  - *Fixed.* The docstring said `tol` means what it means in
    `PoissonRegressor`; with the exposure our per-building gradient is on
    another scale. It now says `tol` bounds the gradient of our objective.
  - *Noted.* Unstandardized simulated features converge (coefficients within
    3.7e-5 of a tight `PoissonRegressor`); features ×1e3 raise "standardize
    them". `predict` has no `errstate`: features ×1e4 give `inf` with only a
    numpy warning (N9 covers the fit).
- **Docs:** `MODULE_REFERENCE.md` (`__init__.py`, `optimization.py`, new
  `total_children.py` row); this doc (status, §7, B2, B8, §10). No code block
  was edited.

### B3. Cohort log loss and tuner typing
*Handoff written 2026-10-01, at the end of the B2 session. Re-verify the facts
in plan mode, write a short summary, ask for approval.*

- **Files:** `scoring.py`, `hyperparameter_tuning/evaluator.py` (`evaluate`'s `y` annotation becomes `Target`), their tests.
- **Build:** `COHORT_LOG_LOSS = Metric("cohort_log_loss", ...)`:
  `−Σ xlogy(n_bk, p_bk) / Σ n_bk` (a plain `0·log 0` gives nan).
- **Tests:** equals a hand computation; a zero-total building adds nothing; the
  evaluator scores a DataFrame `y`.
- **Note in the doc:** `WeightedMean` weights folds by rows; this metric is per child.

Done when:
- [x] The metric equals sklearn's `log_loss` on one row per child; mutations,
  review, non-slow suite: 1155 passed (1 skipped, 1 xfailed).

**Record (2026-10-01).**
- **Baseline** before the first edit: 1147 passed (1 skipped, 1 xfailed).
  B2 is `31ebdd1`, on `e2a4955`; the tree was clean.
- **Verified first**, every fact below (sklearn 1.9.0, scipy 1.18.0):
  `xlogy(0, p) = 0` where `0 · log 0` is nan; `−Σ xlogy / Σ n` equals sklearn's
  `log_loss` on one row per (building, cohort) weighted by its count
  (0.98147…); `xlogy` on two DataFrames aligns by label (a disjoint index sums
  to 0, silently), so the arrays are taken positionally; a count matrix as
  `y_pred` scores silently, and a `(n, 1)` `y_pred` broadcasts over the cohorts
  silently; sklearn's `log_loss` raises for a value above 1 and only warns
  when rows do not sum to 1; the evaluator already runs with a DataFrame `y`
  and a DataFrame-returning model for all 3 split methods under `-W error`.
- **Decision (user):** each row of `y_pred` is divided by its sum, so expected
  counts per cohort (Model 1's or Model 2's output) and probabilities both
  score; `y_true` stays as counts, which weights buildings by their children.
  The user's note: this is the one place a prediction is normalized.
- **Built:** `cohort_log_loss(y_true, y_pred)` and `COHORT_LOG_LOSS`
  (`scoring.py`). One check, for what passes silently: both arrays 2-D of the
  same shape (two 1-D arrays already fail in numpy, so `ndim` is not
  checked). A zero-sum or negative `y_pred` row gives nan or inf, which the
  evaluator's finite check rejects (not silent, so not checked). A negative
  count in `y_true` scores silently: counts are validated in preprocessing
  (N3), not here. The
  evaluator's `y: Target`; its unused `pandas` import removed.
- **Tests.** `test_scoring.py` (+6 cases): the mean per child, pinned to a
  hand value and sklearn's `log_loss` on expanded rows; a building without
  children adds nothing (and a zero probability for a zero count is finite);
  predicted counts score as their proportions; DataFrames pair by position;
  a one-column or 1-D `y_pred` raises; the ready-made row. Not tested: a
  `y_pred` with fewer rows, which numpy itself rejects.
  `test_hyperparameter_tuning_evaluator.py` (+1): a DataFrame `y` is split
  and scored per fold with `COHORT_LOG_LOSS`, against a hand fold loop.
- **Mutations**, each failing its named test and restored: `/ len(counts)`;
  `np.log` for `xlogy`; the normalization dropped; the shape check dropped
  (both cases); `greater_is_better=True`. The annotation has no mutation:
  pandas is untyped here (§6), so mypy accepts a DataFrame `y` under the old
  annotation too (probed with `MYPYPATH=src`); the widening is documentation,
  and the evaluator test exercises the behavior.
- **Checks:** ruff and ruff format on the 4 changed `.py` files; `uv run mypy`
  and mypy on `modeling` plus the 2 test files clean; the 2 test files under
  `-W error` (54 passed).
- **Review** (independent subagent; its own probes and 5 mutations, each
  caught by the pinned test), each finding reproduced:
  - *Fixed.* The `ndim != 2` half of the shape check was untested and guarded
    nothing silent (two 1-D arrays fail in numpy's `sum(axis=1)`); dropped.
  - *Fixed.* The docstring's "adds nothing" holds only for a positive
    prediction row (a zero count with a zero-sum row is nan); now says so.
  - *Fixed.* This record said a negative entry is never silent; a negative
    *count* is. §7's Temperature row called `COHORT_LOG_LOSS(...)`, but a
    `Metric` is not callable: now `cohort_log_loss(...)`. The "Facts from B2"
    list named the renamed test.
  - *Noted for B6.* `softmax(logits / T)` underflows to an exact 0 at a small
    `T` (logits `[800, 0, 0]`), where a positive count gives `inf`; sklearn's
    `_TemperatureScaling` works from `log_softmax`. Decide at B6.
- **Docs:** `MODULE_REFERENCE.md` (`scoring.py` row); `HYPERPARAMETER_TUNING_PLAN.md`
  §4.3 (per child vs per row); this doc (status, §7, B3, B6). No code block was edited.

**Before the first edit:** confirm B2 is committed on top of `e2a4955`
(`git log` shows a `feat(modeling): TotalChildrenModel ...` commit) and the
tree is clean; otherwise stop and report. Baseline: 1147 passed (1 skipped,
1 xfailed).

**Facts from B2 that B3–B5 build on** (verified 2026-10-01; re-verify):
- `scoring.py`: `Metric(name, function, greater_is_better=False)`, a frozen
  dataclass whose `__post_init__` checks the name, the callable and the bool;
  ready-made `POISSON_DEVIANCE`, `RMSE`, `MAE` wrap sklearn; `__all__` lists
  them; tests in `tests/unit/test_scoring.py` (`test_ready_made_metrics_wrap_their_function`,
  `test_invalid_metric_is_rejected`).
- `hyperparameter_tuning/evaluator.py`: `evaluate`'s `y` is annotated
  `pd.Series | np.ndarray` (line 91 when written; find it by name); it imports
  `DesignMatrix, Exposure, Groups, take_rows` from `..utils`, where `Target =
  pd.Series | pd.DataFrame | np.ndarray` already exists (`utils.py:13`). The
  fold aggregation is `hyperparameter_tuning/aggregation.py` (`WeightedMean`,
  the size-weighted mean by rows). B3 changes nothing else in that package.
- `modeling/total_children.py`: `TotalChildrenModel(*, solver="lbfgs",
  use_exposure=True, l2_penalty=0.0, max_iter=500, tol=1e-6)`; fitted
  `intercept_`, `coef_`, `use_exposure_`, `feature_names_in_`, `n_features_in_`;
  `_objective(parameters, X, y, offset, l2_penalty)` static. `check_X_y` at the
  top of `fit`; the names recorded after success by `validate_data(self, X,
  reset=True, skip_check_array=True)`; `validate_data(reset=False)` at predict.
  **B4 repeats these lines** for `CohortProbabilityModel` (with a DataFrame `y`:
  `check_X_y(..., multi_output=True)`); propose a shared GLM base then only if
  the repetition warrants it (user, B2).
- `modeling/optimization.py`: `Minimizer(method, max_iter, tol)`, `Method =
  Literal["L-BFGS-B", "BFGS"]`, `_OPTIONS` table; bounds only with L-BFGS-B.
  B4's `CohortProbabilityModel` gets the same `solver: Solver = "lbfgs"` setting
  (`Solver` and `_METHODS` live in `total_children.py`; move them to
  `optimization.py` if B4 needs them, rather than importing one model from another).
- The user kept `use_exposure_` (predict follows the fitted state, M10) after
  asking whether it is practical: consistency with `DirectCohortModel` and
  sklearn's principle that predict reads fitted state only. Do not re-ask.
- `__init__.py` exports `Solver` and `TotalChildrenModel`; nothing from the
  package root. `MODULE_REFERENCE.md` has the `optimization.py` and
  `total_children.py` rows.

### B8. NB2 family (moved before B4's tests; user, 2026-10-01)
*Handoff written 2026-10-01. The user wants NB2 now: "like Poisson, with the
NB2 adjustments". Start in plan mode, re-verify the facts, summarize, ask for
approval. Order of the remaining steps: **B8 → finish B4 (§5–§7) → B6 → B5 →
B7 → B9.** `cohort_probability.py` and its stale tests stay uncommitted in
the tree during B8; B8's baseline is the non-slow suite with
`--ignore=tests/unit/test_modeling_cohort_probability.py` (1159 passed,
1 skipped, 1 xfailed).*

- **Files:** `modeling/total_children.py`, `tests/unit/test_modeling_total_children.py`,
  `tests/validation/test_total_children.py`, docs.
- **Build** (N10): a setting `family: Literal["poisson", "nb2"] = "poisson"`;
  an unknown family raises at `fit` (a `set_params` value is read only there).
  `_objective` takes the family. For `"nb2"` the point is `[b, β, log α]` and
  the likelihood is torch's
  `NegativeBinomial(total_count=1/α, probs=αμ/(1+αμ)).log_prob(y)`, `μ =
  exp(offset + b + Xβ)` — verified against
  `scipy.stats.nbinom.logpmf(y, 1/α, 1/(1+αμ))` (3e-8) and torch's own mean
  and variance (`μ`, `μ(1+αμ)`): the NB2 of §6. For `"poisson"` the objective
  is unchanged. Start `log α = log 0.1` (§6's fitted dispersion ≈ 0.105).
  **Bounds on `log α`: decide by probe.** The old code bounded `α` in
  `(1e-4, 5)`. Fit near-Poisson data (simulated totals with Poisson draws)
  without bounds: if `log α` runs away or the fit fails, pass `Minimizer`'s
  `bounds` on `log α` (only `lbfgs` takes them; `Minimizer` raises for
  `bfgs`, so say so in the docstring); otherwise no bounds. Fitted
  `dispersion_` (α) only for `"nb2"`; `predict` unchanged (the mean). The
  docstring gains one paragraph on the family.
- **Tests** (one mutation each): `family="poisson"` leaves every existing
  test untouched; NB2 agrees with statsmodels `NegativeBinomial(y, D,
  exposure=exposure)` at λ = 0 in `tests/validation/test_total_children.py`
  (coefficients and α within 1e-5; §6 measured ≤ 1e-5); the gradient test
  (autograd vs `check_grad`) runs for both families; `dispersion_` exists
  only for NB2 (`hasattr`); near-Poisson data converges (and, with bounds,
  does not sit at one); an unknown family raises; NB2's mean doubles with
  the exposure; the contract example stays Poisson.
- **Docs:** §7 shape (`family`, `dispersion_`) and the math row (torch
  `NegativeBinomial`); N10; `MODULE_REFERENCE.md` row; the B8 record (the
  bounds decision with its probe).
- **Commit** (after the solver-move commit from B4 §7):
  `feat(modeling): TotalChildrenModel on torch, with an NB2 family` —
  `total_children.py`, its two test files, the docs. The cohort model is
  committed when B4 finishes.

Done when:
- [x] NB2 matches statsmodels (1.0e-7); mutations, review, suite (with the
  `--ignore`): 1169 passed (1 skipped, 1 xfailed).

**Record (2026-10-01).**
- **Baseline** before the first edit: 1159 passed (1 skipped, 1 xfailed), with
  the `--ignore`; HEAD `316aabf`, the tree as B4 §2.
- **Verified first** (torch 2.14, scipy 1.18, statsmodels 0.14.6; 10 simulated
  tables, seeds 0–9): torch's `NegativeBinomial(total_count=1/α, logits=log α
  + log μ).log_prob` equals `nbinom.logpmf(y, 1/α, 1/(1+αμ))` to ≤ 3.5e-13
  (relative) for α in 1e-4…50, 2.4e-7 at α = 1e-8, mean μ and variance
  μ(1+αμ); the `probs` form is equal but raises torch's own `ValueError` once
  αμ rounds `probs` to 1 (≳ 1e16; the largest αμ any of 60 fits evaluated was
  3.1e2). The objective over `[b, β, log α]` from `log α = log 0.1`, fitted by
  `Minimizer`: within 2.6e-7 of statsmodels' NB2 on every table (both
  solvers; 17–24 evaluations, ~10 ms; α 0.09–0.13).
- **Bounds, decided by probe** (the open question): on near-Poisson draws
  (Poisson draws from the fitted Poisson means, var/mean 0.8–1.06) unbounded
  `log α` runs to −16…−23: lbfgs still converges (13/13, 57–150 evaluations),
  bfgs raises "precision loss" in 10/13, statsmodels warns in 5/10. With a
  floor `α ≥ 1e-6` (lbfgs): 10/10 converge in 27–51 evaluations at the floor,
  coefficients equal to the Poisson's (≤ 2.8e-4); on the real, overdispersed
  totals the floor changes nothing (7.6e-13). Nothing ran upward, so no upper
  bound (the old stack had (1e-4, 5)). The user asked for the rule of thumb:
  α → 0 is the Poisson boundary, where the likelihood has no finite maximum
  on such data; a floor where NB2 is numerically Poisson is the practice.
- **Built:** `family: Family = "poisson"`; `_objective(..., family)` with
  NB2's point `[b, β, log α]`; the one new check (an unknown family would fit
  the Poisson silently); `bounds` on `log α` for nb2; `dispersion_` for nb2
  only (a Poisson refit removes it). `predict` unchanged.
- **The OpenMP clash** (found by the validation test, which segfaulted):
  Homebrew's `libomp` (LightGBM 4.7), scikit-learn's bundled one and torch's
  own all load; with LightGBM first (the package's order: the root imports
  the old stack's LightGBM model before its torch model) torch's threaded
  kernels crash (`logsigmoid` at 10 elements; `exp`, `lgamma`, `sum`, `xlogy`
  at 100k), torch-first after the first LightGBM fit. The unit tests had
  passed only because the test module imports torch first; the Poisson
  objective survived by size (its kernels parallelize above ~32k elements).
  Measured: `threadpoolctl.threadpool_limits(1, user_api="openmp")` lists all
  three runtimes but leaves torch's count at 6 (it crashes);
  `torch.set_num_threads(1)` before the tensors are built and during the fit
  is safe in both import orders, interleaved with LightGBM fits, up to 100k
  rows. Speed, one thread against torch's six (5 fits, 7 features): 245
  rows Poisson 3.0 vs 2.7 ms, NB2 12 vs 17 ms; 2,000 rows 3.5 vs 4.4 and
  29 vs 31 ms; 20,000 rows 13 vs 10 and 134 vs 76 ms. So no cost at this
  data's sizes; from ~20k rows the threads would win. The projects' guidance (LightGBM FAQ, PyTorch forums)
  is one runtime per process, by environment (symlinks; `KMP_DUPLICATE_LIB_OK`
  is called unsafe), which is neither portable nor the library's to do.
  **Decided (user: "use best practices"): the fit runs torch at one thread
  and restores the count in `finally`**; the fits are 12–30 parameters on
  hundreds of rows, where threads gain nothing. First built inside
  `Minimizer.minimize`; the review showed that scope misses the tensors
  built in `fit` (a tensor above torch's grain size, ~32k elements, 4,700 ×
  7, crashes on construction), so it is now `optimization.single_threaded_torch()`,
  a context manager that `fit` wraps its torch calls in, tensors included,
  and `Minimizer` is torch-free again. A module-level function, not a
  class (§3 rule 8): a context manager is what the `with` needs, and it has
  no state; the user may ask for another home. **B4's cohort `fit` must
  wrap its torch calls the same way.** The full suite then still crashed, in the
  NB2 gradient test: it calls `_objective` directly, outside `Minimizer`,
  after an earlier module loaded LightGBM (the file alone passed because it
  imports torch first). So `test_modeling_total_children.py` has a
  module-level autouse fixture that sets one thread and restores it;
  **B4's cohort tests need the same fixture.** The subprocess test is
  outside the fixture's reach, so it still exercises the crash path, on
  40,000 rows, above the grain size, so a scope that misses the tensors
  fails it too.
- **Tests** (`test_modeling_total_children.py`, 29; one mutation each, every
  one failing its named test and restored by copy, md5 confirmed): the
  gradient test over both families (penalty added after `backward`); the
  NB2 objective pinned to `nbinom.logpmf` (1/α and α swapped); a known
  dispersion 0.5 recovered (nb2 fitted as Poisson); Poisson draws end at the
  floor with the Poisson's coefficients (bounds dropped: α lands at ~1e-8);
  `dispersion_` only for nb2 (set for Poisson too); an unknown family and
  `nb2` with `bfgs` raise (check dropped); doubling for both families
  (offset dropped); an NB2 fit survives LightGBM loaded first, in a
  subprocess with `PYTHONPATH=src` (`set_num_threads(1)` dropped, and the
  scope moved to cover only the minimize: segfault, each); the thread count
  is restored after a fit and after a failed one (restore not in
  `finally`). `tests/validation/test_total_children.py`: NB2 vs statsmodels
  `NegativeBinomial(..., loglike_method="nb2").fit(method="newton",
  tol=1e-12)` (converges without a warning; `bfgs` reports non-convergence on
  seed 0) within 1e-5, measured 1.0e-7 (nb2 branch dropped: fails).
- **Checks:** ruff and ruff format on the 5 changed `.py` files; `uv run mypy`
  and mypy on `modeling` plus the 3 test files; the modeling tests under
  `-W error` (72 passed); the validation tests under `-W error` (2 passed);
  the non-slow suite with the `--ignore`: 1169 passed (+10: 9 cases in
  `test_modeling_total_children.py`, 1 in `test_modeling_optimization.py`).
- **Review** (independent subagent; its own NB2 density by `gammaln` and a
  40-digit `mpmath` derivative of `log α` down to the floor: value within
  1.4e-11, gradient 1.1e-9 at α = 1e-6; 8 mutations of its own), each
  finding reproduced:
  - *Fixed.* The single-thread scope in `Minimizer` did not cover the
    tensors `fit` builds: above ~32k elements (4,700 × 7 rows) the
    construction itself segfaulted with LightGBM loaded first, and this
    record's "safe up to 100k rows" had been measured with the thread set
    before the tensors. Now `single_threaded_torch()` around the whole fit;
    the subprocess test runs 40,000 rows.
  - *Fixed.* The restore under an exception was untested (a restore after
    the `try` survived every test). The thread test now also fits features
    on a huge scale, which raises, and checks the count after.
  - *Noted.* A non-integer or negative `y` is rejected by torch's own
    argument validation (`IntegerGreaterThan(0)`), which is off under
    `python -O`; data values are preprocessing's (N3), so no check is added.
    An all-zero `y` ends in "non-finite objective" with the scale hint
    (since B2). Under torch an overflow is not numpy's `FloatingPointError`
    (torch ignores `np.errstate`; `inf` ends as "did not converge:
    ABNORMAL"), so N9's wording describes the scipy version: it still
    raises, with the other message (since B4's torch move).
  - *Fine.* The parameterization, the floor, the start (not load-bearing:
    fits converge from it for α in 1e-6…20), `coef_`'s slicing, the
    `dispersion_` handling across `set_params` refits, the penalty on `β`
    only, the subprocess test's path and cleared environment, the
    tolerances.
- **Docs:** `MODULE_REFERENCE.md` (`optimization.py`, `total_children.py`
  rows); this doc (status, N6, N10, §7 shape and math row, §11, this record).
  No code block was edited.

### B4. `CohortProbabilityModel`: a Dirichlet regression of the cohort shares
*Finish after B8 (user, 2026-10-01): the B8 session's commits leave
`cohort_probability.py` and its stale tests uncommitted.*
*Specified by the user on 2026-10-01 after two builds were rejected. Handoff
written at the end of that session: the model is implemented (§4 below);
**the tests, checks, mutations, review, suite and the stop are left to the
next session** (§5–§7). Re-verify §2–§3 in plan mode, summarize, ask for
approval, then do §5–§7.*

- **Files:** `modeling/cohort_probability.py` (implemented, untracked),
  `__init__.py` (exports it; done), `tests/unit/test_modeling_cohort_probability.py`
  (**the previous build's tests; must be rewritten per §5**), the contract
  example (present; see §5's last line).
- **Build:** N6, N8, N11 (all revised at B4). A passed exposure is ignored (N3 e).

Done when:
- [x] The objective equals `scipy.stats.dirichlet.logpdf`; known parameters
  are recovered; mutations, review, non-slow suite pass: 1189 passed
  (1 skipped, 1 xfailed).

**Record of the finish (2026-10-01; B8 came first).**
- **Baseline** before the first edit: 1169 passed (1 skipped, 1 xfailed) with
  the `--ignore`; HEAD `68b72c2`.
- **Verified first** (probes with torch imported first): `_objective` equals
  `−mean_b dirichlet.logpdf(s'_b, α_b) + ½λ‖W‖²` to 4e-17 and its gradient
  `approx_fprime` to 1.9e-7; on Dirichlet-drawn data (N = 2000, totals 1000)
  `coef_` is within 0.062 (K = 2), 0.032 (K = 3), 0.044 (K = 4) of `W0`,
  intercepts within 0.043, 35–46 ms per fit; scaling the counts by 10 leaves
  the fit unchanged (0.0); λ = 1e8 equals the fit on `X·0` (1.5e-7);
  `predict == softmax(predict_logits)`; the three checks, non-convergence,
  the ignored exposure and the numbered cohorts behave as §4 says.
  Suite count below: 1189 (1169 + 20).
  **Correction to §4 and §8:** a negative count and an empty building raise
  from torch's argument validation (`within the support (Simplex())`), the
  empty building after numpy's "invalid value" warning on the division, not
  from the minimizer (only under `python -O` would they reach it). Both
  raise, so no check is added (N3). A one-column `y` fits the share 1, as §4
  says (this session first wrote that it raises: it did in a probe only
  because that column had zeros). The §5 margin
  "< 0.95 × the marginal" fails on some seeds at 300 rows (ratios
  0.92–0.98); at 1000 fit rows the ratio is 0.77–0.91 over 6 seeds, and the
  test asserts 0.95 on a seed at 0.77.
- **Built:** `fit` wraps the tensors and the minimize in
  `single_threaded_torch()` (B8); the docstring says at least two cohorts.
- **Tests** (`test_modeling_cohort_probability.py`, 20; the stale sklearn
  build's tests deleted; a module-level single-thread fixture as B8's): the
  table in §5, with the recovery test over K = 2, 3, 4 (the "any number of
  cohorts" test folded into it), the marginal test at 1000/1000 rows, the
  thread count restored after a fit and a failed one, and (from the review)
  the pinned objective stationary at `fit`'s point, which ties `fit`'s
  compression to the formula the objective test uses. No subprocess
  test: the guard is B8's `single_threaded_torch()`, whose LightGBM-first
  test covers it.
- **Mutations** (15), each failing its named test and restored by copy with
  md5 confirmed: `.sum()` for `.mean()`; the penalty added after `backward`;
  the counts fitted unnormalized (the recovery test, all K); the compression
  dropped; the intercepts penalized; `index=None`; `columns=None`; a literal
  3 in the reshape (K = 2 and 4); `list(y.columns)` on an array; the
  unobserved and the penalty checks dropped; the names recorded before the
  fit; both solvers mapped to L-BFGS-B; the start returned; the contract
  example deleted (6 cases). *Pitfall met:* restoring
  `test_modeling_contract.py` with `git checkout` reverted the uncommitted
  example too; it was re-added from the recorded diff and checked line by
  line. Restore by copy only.
- **Checks:** ruff and ruff format on the 3 changed `.py` files; `uv run
  mypy` and mypy on `modeling` plus the 2 test files; the modeling tests
  under `-W error`.
- **Review** (independent subagent; its own `gammaln`/`digamma` density and
  gradient, equal to 1e-13 at K = 2/3/4; the fit equal to the MLE started
  from the truth to 1e-5; 6 mutations, all caught), each finding reproduced:
  - *Fixed (test).* A scale-invariant wrong compression in `fit` (e.g.
    `(s(N−2) + 2/K)/N`) passed every test: the objective test used its own
    shares. New test: the pinned objective is stationary at `fit`'s point
    (gradient 2.7e-7; mutation: the compression's floor changed).
  - *Fixed (docs).* N8 still said "per child for probabilities"; this
    record's one-column claim (above); §4/§8's "raises from the minimizer".
  - *Not changed.* A Series `y` raises an `IndexError` without a message
    (it raises; §4 records it). The margin is 0.95 as planned.
  - *Fine.* Compression = `DirichletReg::DR_data`'s; prediction = the mean to
    1e-12; the thread guard covers the tensors; edge inputs (float shares,
    constant total, duplicated names, constant column, totals 1e7, extra
    columns, `set_params` after fit, `clone`) behave or raise.
- **Docs:** `MODULE_REFERENCE.md` row; this doc (status, N8, N11, §7, this
  record). No code block was edited.

**§1. The user's specification and decisions (2026-10-01).**
- Model 2 predicts `μ_b · p_bk`: `μ_b` from `TotalChildrenModel`, `p_b` the
  building's cohort **shares**. **The shares come from a Dirichlet
  distribution whose concentration depends on the features**: a Dirichlet
  regression of the observed composition `n_bk / n_b` ("normalize the cohort
  counts by the building's total to get a probability distribution"),
  prediction = the Dirichlet mean. Not Dirichlet-multinomial: that is the
  third, Bayesian model. "There should be no building without children."
  Boundary treatment: best practice, explained, not overcomplicated.
- Rejected builds, for the record: (1) a hand-coded multinomial logit
  (softmax, `log_softmax` objective, analytic gradient, `Minimizer`), verified
  against `LogisticRegression` to 1e-6 — the user asked why not a library;
  (2) sklearn's `LogisticRegression` on one row per (building, cohort) with
  `sample_weight = count`, `C = 1/(λM)` (`2/(λM)` for 2 cohorts), verified to
  5e-6 — the user then specified the Dirichlet. Both predicted
  `softmax(a + xW)` fitted on the **counts**; the Dirichlet fits the
  **composition** and also models its precision.
- Kept from earlier in B4 (user): the solver is `Minimizer`'s own setting
  (`optimization.py`: `Solver = Literal["lbfgs", "bfgs"]`,
  `Minimizer(solver, max_iter, tol)`; `total_children.py` uses it; its own
  table and check removed); no GLM base (the GLMs repeat only
  scikit-learn's per-estimator calls); calibration is post-hoc (N12
  revised): no `temperature` on the model, `predict_logits` for the
  calibrator, B6 before B5.

**§2. State of the tree at the handoff** (verify; stop and report if different):
branch `feat/independent-total-probability-model`, HEAD `316aabf` (B3).
Modified: `modeling/{__init__,optimization,total_children}.py`,
`tests/unit/test_modeling_{contract,optimization,total_children}.py`,
`docs/MODULE_REFERENCE.md`, this doc. Untracked: `modeling/cohort_probability.py`
(the Dirichlet model) and `tests/unit/test_modeling_cohort_probability.py`
(the sklearn build's tests, stale). The non-slow suite fails in that test
file until it is rewritten (everything else: 1159 passed with
`--ignore=tests/unit/test_modeling_cohort_probability.py`); at HEAD it gives
1155 passed (1 skipped, 1 xfailed). `total_children.py` and its tests are
also modified (torch objective; the all-zero check dropped; the failed-refit
test uses `max_iter=1`; the gradient test builds tensors), and all pass.

**§3. Measured facts** (scipy 1.18.0; re-run any you rely on; the probes
lived in the session scratchpad, so rebuild them from these recipes):
- Simulated tables (`StudentPopulationSimulator(load_simulation_config(
  "configs/simulation.toml")).run(rng=np.random.default_rng(seed))`, seeds
  0–9): **no building without children** (min total 2, median 19); 16 of 245
  buildings (seed 0) have a zero count in one cohort. So the boundary
  matters; empty buildings do not occur and are rejected.
- `scipy.stats.dirichlet.logpdf` of a share with a zero is `-inf`.
- Dirichlet regression with `α = exp(a + XW)`, objective
  `−mean_b log Dirichlet(s'_b | α_b) + ½λ‖W‖²` (measured first with the
  written `gammaln`/`digamma` form, `check_grad` 5e-8; the shipped torch
  objective equals it to 1e-15), fitted by `Minimizer("lbfgs", 500, 1e-6)` from zeros:
  5–17 ms; on Dirichlet-drawn shares (N = 2000, 3 features, K = 3,
  `a0 = [1, 1.5, 0.5]`, `W0 ~ N(0, 0.5²)`) recovers `W` within 0.035 and the
  mean within 0.019; the `gammaln` form equals `dirichlet.logpdf` exactly (0.0).
- **Boundary transforms**, synthetic data at this data's totals (shares
  Dirichlet-drawn, totals `2 + Poisson(17)`, counts multinomial, N = 200,
  K = 3, 5 seeds; ~66 of 200 buildings with a zero share), mean absolute
  error of the predicted mean on 200 new buildings:

  | Transform | Error |
  |---|---|
  | Smithson–Verkuilen dataset compression `s' = (s·(N−1) + 1/K)/N` | **0.019 ± 0.002** |
  | per-building pseudo-count 1/K: `(n_bk + 1/K)/(n_b + 1)` | 0.026 ± 0.003 |
  | Jeffreys 0.5: `(n_bk + ½)/(n_b + K/2)` | 0.031 ± 0.004 |
  | per-building S&V with `n_b` | 0.027 ± 0.003 |
  | multinomial logit on the counts (reference, not a Dirichlet) | 0.015 ± 0.002 |

  **Chosen: S&V dataset compression.** It is the standard in Dirichlet
  regression (Smithson & Verkuilen 2006; R's `DirichletReg::DR_data` applies
  it), parameter-free, one line, vanishes as N grows, and measured best of
  the Dirichlet options.
- The "common" parameterization (one `α_k = exp(a_k + x β_k)` per cohort,
  `DirichletReg`'s default) is identified: shifting every intercept changes
  the precision `Σα`, not the mean, so the coefficients are unique (unlike
  the softmax of the rejected builds). The fit is unweighted by `n_b`: the
  observation is a building's composition (`DirichletReg`'s default).

**§4. Build** (`modeling/cohort_probability.py`, implemented; simplified
on the user's notes, 2026-10-01: fewer comments, fewer checks, a short
implementation over the explicit formula):
`CohortProbabilityModel(*, solver: Solver = "lbfgs", l2_penalty=0.0,
max_iter=500, tol=1e-6)`.
- `_objective(parameters, X: Tensor, shares: Tensor, l2_penalty)` (static):
  `parameters = [a (K), W (d×K, row-major)]` as a numpy point from scipy;
  `α = exp(a + XW)` in torch; value `−Dirichlet(α).log_prob(shares).mean()
  + ½λ‖W‖²`; `backward()`; returns `(value.item(), params.grad.numpy())`.
  No written density or gradient (user): torch's density is vectorized
  over buildings and autograd exact. `fit` builds the two float64 tensors
  once.
- `fit(X, y, exposure=None)`: `l2_penalty < 0` raises; `check_X_y(X, y,
  multi_output=True, y_numeric=True)`; a cohort with no child raises (it
  would fit the compressed floor silently); `shares = counts / totals`
  compressed `(s(N−1) + 1/K)/N` inline; start zeros (`α = 1`, `W = 0`);
  `Minimizer(self.solver, self.max_iter, self.tol)`; fitted `intercept_`
  (K), `coef_` (d×K), `cohorts_` (`y`'s columns, or 0..K−1 for an array),
  then `validate_data(self, X, reset=True, skip_check_array=True)`.
  **Dropped checks** (measured without them): a one-column `y` fits a share
  of 1 (trivially right); a Series `y` raises on its own; a negative count
  and a building without children each raise (torch's argument validation;
  see the record). The docstring states both requirements (N3: data values are
  preprocessing's).
- `predict_logits(X) -> DataFrame`: `log α = a + XW` (columns `cohorts_`,
  `X`'s index; `validate_data(reset=False)` first). `predict(X,
  exposure=None) -> DataFrame`: `softmax(log α)` = the Dirichlet mean
  `α/Σα`, rows sum to 1.

**Smoke of the implementation (this session, not the tests):** ruff and
`uv run mypy` clean (the IDE's own checker flags the `y: pd.DataFrame`
override, as it would `total_children.py`'s `y: pd.Series`; mypy, the
project's check, does not). On Dirichlet-drawn data (N = 2000, totals 1000)
`coef_` is within 0.025 of `W0` and the mean within 0.017, intercepts
`[1.01, 1.51, 0.49]` for `[1, 1.5, 0.5]`; on totals `2 + Poisson(17)` (583
rows with a zero share) the fit runs and its `cohort_log_loss` is 0.957
against 1.042 for the marginal shares; `check_grad` 7e-8 on the formula version; the torch version equals the
looped `dirichlet.logpdf` to 6e-17 and the formula gradient to 2e-15, fits
in 13 ms at 245 buildings and 44 ms at 2,000 (coefficients within 0.029 of
`W0`); the
checks raise their messages (unobserved cohort, `max_iter=1`, reordered
columns); an array `y` gives `cohorts_ == [0, 1, 2]`.

**§5. Tests** (`tests/unit/test_modeling_cohort_probability.py`, rewrite;
each names its mistake; one mutation each). Data: `_data(rows, n_cohorts,
seed)`: 3 features (`ses`, `size`, `noise`); true `α = exp(a0 + X W0)`;
shares `rng.dirichlet(α_b)`; totals `2 + rng.poisson(17)`; counts
`rng.multinomial(total, share)` → DataFrame named `n_kindergarten`,
`n_elementary`, `n_highschool` (or `c0..`). A large-count variant (totals
1000) for parameter recovery.

| Test | Mistake it catches | Mutation |
|---|---|---|
| the objective is the Dirichlet log-density: `_objective` at random parameters (tensors in) equals `−mean_b scipy.stats.dirichlet.logpdf(s'_b, α_b) + ½λ‖W‖²` (loop over rows) | the wrong distribution, or a sum for the mean | `.sum()` for `.mean()` |
| *(optional)* the autograd gradient matches `scipy.optimize.approx_fprime` (1e-5) | a value that is not what is differentiated (e.g. the penalty added after `backward`) | penalty added after `backward()` |
| known parameters are recovered (totals 1000, N = 2000: `coef_` within 0.1 of `W0`, the mean within 0.02) | shares not normalized, or `exp` missing | fit on the counts |
| shares on the boundary are compressed (data with zero counts fits; a raw `log 0` would end in `RuntimeError`) | the compression dropped | the compression line removed |
| scaling every count leaves the fit unchanged (`10·Y`, 1e-6) | counts where shares should be | fit on the counts |
| a huge penalty leaves the intercept-only fit (λ = 1e8 equals the fit on `X·0`, rtol 1e-6) | intercepts penalized | penalize `a` |
| probabilities are named like `y`, indexed like `X`, rows sum to 1, `predict == softmax(predict_logits)` | misnamed/misaligned; logits out of step | `index=None`; `columns=None` |
| any number of cohorts works (K = 2, 4 recover `W0` within 0.15 on totals 1000) | 3 hard-coded | reshape with a literal 3 |
| cohorts from an array are numbered (`cohorts_ == [0, 1, 2]`) | `.columns` on an array | `list(y.columns)` |
| input that would fit silently raises: unobserved cohort (`cohorts \['n_highschool'\]`), a negative penalty, reordered columns at predict ("feature names") | each silent | each check → `if False:` |
| non-convergence raises (`max_iter=1` → `RuntimeError` "did not converge") | a non-converged fit used | (Minimizer's; one case) |
| a failed refit leaves the previous fit intact (`set_params(max_iter=1)` on renamed columns raises `RuntimeError`; old predictions unchanged; renamed columns then raise) | mixed fitted state | names recorded before the fit |
| the solver setting picks the scipy method (monkeypatch `optimization.minimize`; `solver="bfgs"` → `"BFGS"`; as in `test_modeling_total_children.py`) | solvers mapped alike | map both to L-BFGS-B |
| the model beats the marginal shares on new buildings (`cohort_log_loss` < 0.95 × the marginal) | a fit stuck at its start | return the start |
| a passed exposure is ignored (same fit and predictions) | — | — |

Contract example (`test_modeling_contract.py`): done, `1 + rng.poisson(...)`
for one cohort so that no row is empty.

**§6. Docs** (done in this session unless marked): `MODULE_REFERENCE.md`
`cohort_probability.py` row; this doc: status, N6, N7, N11, §7 shape and
math row, §10, this section. *Left to the next session:* the B4 record
(baseline, checks, mutations, review, suite count) and the status line's
count. No doc code block runs yet (§7's block needs B5/B6).

**§7. Checks and the stop.** Baseline first (`uv run pytest -m "not slow"`:
expect failures only in the stale test file; record the count after the
rewrite); `uv run ruff check <files>` and `uv run ruff format <files>`
(explicit paths; zsh does not split `$VAR`); `uv run mypy`; `uv run mypy
src/age_group_prediction/modeling tests/unit/test_modeling_cohort_probability.py
tests/unit/test_modeling_contract.py`; `uv run pytest tests/unit/test_modeling_*.py
-q -W error -p no:cacheprovider`; the mutations (back the file up to the
scratchpad, restore by copy, confirm with md5 and `git diff --stat`); an
independent review subagent (verify the log-density and gradient
independently, the compression, K = 2/4, edge inputs, 4–6 mutations of its
own); reproduce each finding before fixing or rejecting it; the non-slow
suite; this doc. Stop with a file-by-file summary, the check results, and
two suggested commits:
1. `refactor(modeling): the solver is Minimizer's own setting` —
   `optimization.py`, `total_children.py`, `test_modeling_optimization.py`,
   `test_modeling_total_children.py`.
2. `feat(modeling): CohortProbabilityModel, a Dirichlet regression of the cohort shares` —
   `cohort_probability.py`, `__init__.py`, its tests, the contract test, both docs.

**§8. Pitfalls.** `_objective` takes torch tensors (float64), not arrays:
tests build them with `torch.tensor(..., dtype=torch.float64)`; `fit` does
it once. Scratch scripts only in the scratchpad, run as
`PYTHONPATH=src uv run --group test python <script>`. Import the model from
`age_group_prediction.modeling` (the root exports the old stack).
`check_X_y` keeps integer dtypes; the division to shares is true division.
A negative count or an empty building raises from torch's argument
validation (not checked: data values are preprocessing's, N3). The import guard forbids `models`, `metrics`, `tuning`, …
in `modeling/`.

**Facts for B6 and B5** (verify in plan mode):
- `CohortProbabilityModel.predict_logits(X)` returns `log α` as a DataFrame
  (columns `cohorts_`, `X`'s index); `predict` is its row softmax, the
  Dirichlet mean. Temperature scaling acts on `log α` like on any logits
  (`softmax(log α / T)`). A fold's logits come from
  `fold.model_.predict_logits(fold.feature_transformer_.transform(X_val))`
  (§7 loop).
- B3's review: `softmax(logits / T)` underflows to an exact 0 at a small `T`
  (logits `[800, 0, 0]`), where a positive count makes `cohort_log_loss`
  infinite; a log-space objective (`log_softmax`) avoids it.
- `cohort_log_loss` normalizes `y_pred`'s rows, so it scores probabilities and
  counts alike; it is per child, so it weighs buildings by their children
  even though the Dirichlet fit does not.

### Handoff for the next session (written 2026-10-01, after B4)
*Read this, then B6, B5, B7, B9; re-verify every fact in plan mode before
relying on it. Order: **B6 → B5 → B7 → B9**, one step per stop.*

- **State:** HEAD `4c02f8b` (B4), on `68b72c2` (B8) and `5a27727` (the
  solver move); the tree is clean; non-slow suite 1189 passed (1 skipped,
  1 xfailed), no `--ignore` needed. `modeling` exports `BaseAgeGroupModel`,
  `CohortModels`, `CohortProbabilityModel`, `DirectCohortModel`,
  `IndependentCohortModels`, `ModelPipeline`, `Objective`, `Solver`,
  `TotalChildrenModel`; `optimization.py` has `Minimizer`, `Solver`,
  `single_threaded_torch`; `total_children.py` has `Family`.
- **What the last two steps settled, and every later step inherits:**
  - LightGBM, scikit-learn and torch each load an OpenMP runtime; importing
    the package loads LightGBM first, and torch's threaded kernels then
    segfault. A torch model's `fit` wraps its torch calls, tensors included,
    in `single_threaded_torch()`. A test module that calls a torch objective
    directly has a module-level autouse fixture setting one thread
    (`test_modeling_total_children.py`); a probe imports torch first or sets
    one thread; a test that needs LightGBM loaded first runs in a subprocess
    (`test_an_nb2_fit_survives_lightgbm_loaded_first`). B6 needs none of
    this (no torch); B5 and B7 only inherit it through the models.
  - Restore a mutated file by copying the backup, never `git checkout`
    (it reverted uncommitted work once); confirm with `md5 -q`.
  - The statsmodels tests run with `uv run --group validation pytest
    tests/validation/... -W error`.
- **Facts measured for B6 (probe, 2026-10-01):** on miscalibrated logits
  (`2.5 · log p`, 400 rows, multinomial counts of 20) the plain objective
  `cohort_log_loss(counts, softmax(logits/T))` and the log-space one
  `−Σ n·log_softmax(logits/T) / Σn` both give T = 2.506 (16–17 evaluations,
  `minimize_scalar(bounds=(−10, 10), method="bounded")` over `log(1/T)`);
  at logits `[800, 0, 0]` and T = 0.5 the plain one is `inf`, the log-space
  one 1600. **Decided: the log-space objective** (sklearn's
  `_TemperatureScaling` form). It needs only scipy (`log_softmax`), no torch.
  `cohort_log_loss` still scores the result in tests (it normalizes rows).
  *At B6's stop the user made it per building (each building's observed
  composition, averaged over buildings), see B6's record.*
- **Facts measured for B5:** `sklearn.base.clone` of a fitted calibrator
  drops `temperature_`; `sklearn.frozen.FrozenEstimator` (sklearn 1.9) keeps
  it through `clone`. **Decided: Model 2 clones its two models in `fit`
  (A3's pattern) and uses the calibrator as given**, fitted by the caller;
  `predict` checks it is fitted (`check_is_fitted`). Ask the user once, at
  B5's plan, whether a `FrozenEstimator` wrapper is wanted instead. One
  check in B5's `fit`: the fitted probability model's `cohorts_` equal
  `y`'s columns (a positional mismatch would be silent).
- **B7:** the smoke run fits Model 1 (LightGBM) and Model 2 (torch) in one
  process: the OpenMP case above; the models' guard covers it. Model 1's
  reference is A4's table; reuse A4's recipe (B0a's `train_test_indices`).
- **Rules that bit this session:** the §5 margins in a test must be measured
  on the test's own data and seed before they are written (B4's "< 0.95 ×"
  failed at 300 rows, held at 1000); every "raises" claim is probed, since
  torch's argument validation raises before the minimizer does; `-W error`
  turns numpy's divide warning into the raise.

### B6. `TemperatureCalibrator` (built before B5, since B4)
- **Files:** new `modeling/calibration.py`, tests.
- **Build:** N12–N14; no settings. `fit(logits, counts)` sets `temperature_`;
  `predict(logits)` returns `softmax(logits / temperature_)` as a DataFrame
  (the logits' columns and index). B3's review: `softmax(logits / T)` can
  underflow to an exact 0 at a small `T`, where a positive count makes
  `cohort_log_loss` infinite; sklearn's `_TemperatureScaling` works from
  `log_softmax`. *Decided in the handoff above: the objective is
  `−Σ n·log_softmax(logits / T) / Σn` over `log(1/T)` in (−10, 10),
  `minimize_scalar(method="bounded")`; no torch. Revised by the user at the
  stop: per building, `−mean_b Σ_k s_bk · log_softmax(logits_b / T)_k` with
  `s_b` the observed composition (the record).*
- **Tests:** logits scaled by a known factor recover it; calibrated data gives a
  temperature near 1; uninformative logits flatten instead of failing;
  `temperature_ = 1` is the plain softmax (moved from B4); the out-of-fold
  loop of §7 as an integration test (with B5, or a `ModelPipeline` alone).

Done when:
- [x] The fit equals scikit-learn's temperature scaling (`_TemperatureScaling`
  on one row per (building, cohort), weighted by the building's observed
  share): T to 0.0 with sklearn's `xatol`.
- [x] Mutations, review, non-slow suite: 1203 passed (1 skipped, 1 xfailed).

**Record (2026-10-01).**
- **Baseline** before the first edit: 1189 passed (1 skipped, 1 xfailed).
  HEAD was `86749d6`, one docs-only commit (this doc's B6–B9 handoff) on
  `4c02f8b`; no code differed from the handoff's state; the tree was clean.
- **Verified first** (sklearn 1.9.0, scipy 1.18.0; probes with torch
  imported first): the plain objective `cohort_log_loss(counts,
  softmax(logits/T))` and the log-space one give the same T (2.484 on
  `2.5 · log p`, 400 rows; 15 evaluations); calibrated logits give T in
  0.986–1.023 over seeds; at logits `[800, 0, 0]` the plain loss is `inf`
  and scipy's bounded search then multiplies `inf − inf`, a
  `RuntimeWarning` (an error under `-W error`), where the log-space fit
  succeeds; sklearn's `_TemperatureScaling` (`sklearn/calibration.py`,
  class at line 1068) is the decided form: `minimize_scalar(log_loss,
  bounds=(-10, 10), options={"xatol": 64·eps})` over `log β`, a log-space
  multinomial loss, `RuntimeError` on `not success`, `predict =
  softmax(β · logits)`; fitted on one row per (building, cohort) with
  `sample_weight = count` it reproduces our T to 1.4e-6 with scipy's
  default `xatol` and to **0.0 with sklearn's** (17 evaluations), so that
  option is kept. Noise logits land at a large T (373, or the bound 2.2e4)
  and the probabilities flatten to within 7e-3 of uniform; all-zero logits
  give a flat objective (any T; the search ends at a bound). `clone` of a
  fitted calibrator drops `temperature_` (B5 uses it as given). The §7
  out-of-fold loop with `CohortProbabilityModel` alone: T = 1.019, held-out
  loss 0.9411 raw and calibrated.
- **Built:** `TemperatureCalibrator(BaseEstimator)`, no settings;
  `fit(logits, counts)`: arrays as floats, paired by position; the two
  checks for what passes silently (equal shapes, since numpy broadcasts a
  column; equal columns when both are DataFrames, since a cohort order
  mismatch is silent); the objective `−mean_b Σ_k s_bk · log_softmax(logits_b · β)_k`
  over `log β` (`s_b` the building's observed composition; per child until
  the user's decision at the stop, below); one `RuntimeError` when the search did not succeed or ended
  (scipy reports a nan objective the same way: nan logits, and all-zero
  counts, whose 0/0 is numpy's warning under `-W error`, else a nan
  objective); `temperature_ = 1/β`. `predict(logits)`:
  `check_is_fitted`, `softmax(logits / T)` with the logits' columns and
  index (`getattr`, as `predict_logits`). Exported from `modeling` only; not
  a `BaseAgeGroupModel` (its `fit` takes logits, not `X`), so the contract
  test does not discover it.
- **Tests** (`test_modeling_calibration.py`, 14; data: Dirichlet(2, 2, 2)
  shares, totals `2 + Poisson(17)`, multinomial counts, logits `factor ·
  log p`): the sklearn oracle, weighted by the shares (T within 1e-7,
  measured 0.0; predictions 1e-6); a factor of 2.5 recovered within 0.1
  (seeds 0–4: 2.47–2.59); calibrated logits within 0.05 of 1 (0.99–1.04); extreme logits fitted in log space under
  `filterwarnings("error")`; noise logits flatten (T > 10, within 0.05 of
  uniform; measured 373 and 7e-3); `temperature_ = 1` is the plain softmax;
  columns, index and row sums; frames and arrays agree (positional); a
  one-column `counts` and reordered cohorts raise; predict before fit
  raises `NotFittedError`; a failed search (monkeypatched `success=False`)
  and nan logits raise; the §7 out-of-fold loop with a
  `ModelPipeline(FeatureTransformer, CohortProbabilityModel)` and `KFold`
  runs as documented under `filterwarnings("error")`.
- **Mutations** (17), each failing its named test and restored by copy with
  md5 confirmed: the weight per child (the counts summed over buildings,
  where each building's composition belongs; T moves 0.014; the oracle
  test), `xatol` dropped (T moves 9.4e-7; the oracle
  test at 1e-7), `β` stored as T, the start returned, the normalization
  dropped, `log(softmax)` for `log_softmax`, `log β` bounded at 0 (T ≤ 1)
  and the range narrowed to (−5, 5), softmax over the buildings, T
  multiplied at predict, `columns=None`, `index=None`, the frames multiplied
  by label, the shape check and the columns check each dropped, the search
  check dropped (2), `check_is_fitted` dropped.
  *Pitfall:* a first weighting mutant, `/ len(counts)` for `/ Σ counts`,
  survived rightly: a constant factor on the objective cannot move its
  minimum. And the real one survived until the test data's totals varied:
  with 20 children in every building the two weights coincide.
- **Checks:** ruff and ruff format on the 3 changed `.py` files; `uv run
  mypy` and mypy on `modeling` plus the test file clean; the new tests and
  the contract test under `-W error` (14 and 27 passed).
- **Review** (independent subagent; its own oracle run at K = 2 and 3 with
  totals 1–60, T equal to 4e-16; 14 edge inputs; 6 mutations, all caught),
  each finding reproduced:
  - *Fixed.* The `isfinite(result.fun)` half of the search check was dead:
    scipy's bounded search already returns `success=False` ("NaN result
    encountered") on a nan objective, so dropping that half survived every
    test. Now `if not result.success` alone, and the comment says scipy
    reports a nan this way.
  - *Fixed.* The flatten test asserted only `T > 10` and 0.05 of uniform; a
    range narrowed to (−5, 5) survived (T = 148). It now pins the bound
    (`T == e^10`, rel 1e-3) and 1e-3 of uniform (measured 1.2e-4).
  - *Fixed.* The oracle test's `abs=1e-6` let a dropped `xatol` survive
    (9.4e-7); now 1e-7 (the gap on the test data is 3.4e-8, not the
    reviewer's 0.0 on its own data, so not the suggested 1e-9).
  - *Fixed.* The comment "(no child at all)" described a path that under
    `-W error` is numpy's warning; reworded. `test_predict_before_fit_raises`
    gained its comment (without the call it is an `AttributeError`, not the
    error callers test for), and its mutation.
  - *Documented.* Negative counts fit silently (a negative share just
    reweights); the docstring now says the counts are nonnegative, with at
    least one child per building, validated where the data is prepared (N3). The columns check runs only when both inputs are
    DataFrames (by design; the shape check always runs).
  - *Fine.* Integer counts, permuted indexes (positional), a single
    building, `predict` on an ndarray, Series inputs (numpy's `AxisError`),
    `clone` unfitted. A zero-count row among others added nothing in the
    per-child form; with the shares it is 0/0 and raises (numpy's warning
    under `-W error`, else the search's nan), never silent.
  - *Re-review of the per-building delta* (its own argmin on a 20k grid and
    sklearn's class weighted by the shares, K = 2 and 3, mixed totals: gaps
    ≤ 8e-8 and ≤ 5e-9, 0.0 on the test data; the per-child form differs by
    0.014–0.23): no bug; the three wording nits above and in the docstring
    fixed.
- **Decision at the stop (user, 2026-10-01): the objective is per building.**
  The user asked what is minimized and why the counts: the answer was the
  negative log probability of the observed children (one row per child, as
  `cohort_log_loss` weighs it), and the user chose instead each building's
  observed composition `s_b = n_b / Σ_k n_bk`, its cross-entropy averaged
  over buildings: the calibrator returns a probability distribution per
  building, so it is calibrated per building, as the Dirichlet fit weighs
  its observations. One line in `fit` (`shares`, then `.sum(axis=1).mean()`);
  the oracle test weights sklearn's rows by the shares (T to 0.0; the
  per-child form differs by 0.014 on the test data and is now the mutant);
  the margins re-measured on the test's own data (above); the 17 mutants
  rerun, all caught; the delta reviewed; suite 1203. The docs here, §7's
  math row, N8 and the handoff note updated.
- **Docs:** `MODULE_REFERENCE.md` (`__init__.py` row; new `calibration.py`
  row); this doc (status, N8, §7's math row and the LaTeX block "The
  calibration objective" after it, the handoff note, B6);
  `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` (a dated pointer at "Temperature
  calibration" to that block; its §0 is B9's). No code block was edited (§7's block needs
  B5).

### B5. `IndependentTotalProbabilityModel`
- **Files:** new `modeling/independent_total_probability.py`, `__init__.py`, tests, contract example.
- **Build:** an aggregator of two models that take the raw table
  (`ModelPipeline`s, N17). Rows are paired by position (N19). Follow
  `IndependentCohortModels` (A3): clone both in `fit`, pass the same exposure
  to both, assign the fitted copies together only after both succeed, and take
  each prediction as an array before combining (a Series with its own index
  would realign into NaN). The output has `y`'s columns at fit and `X`'s index;
  the probability model's cohorts must equal `y`'s columns (checked: a
  positional mismatch would be silent). Unlike N21's
  mapping, two named estimators are ordinary parameters, so nested `set_params`
  (`cohort_probability_model__model__l2_penalty`) reaches them. *From the
  handoff: the calibrator is used as given, not cloned (`clone` would drop
  its `temperature_`); `predict` checks it is fitted.*
- **Tests:**
  1. the output equals total × probabilities of the two models fitted separately, and rows sum to the total mean;
  2. the total target is `y.sum(axis=1)`; tables without target columns work;
  3. each pipeline uses its own transformer (swapped transformers);
  4. doubling the exposure doubles every cohort (`n_apartments` is not a feature of `total_base`);
  5. nested `set_params` reaches the next fit; a given `temperature_calibrator` is applied to the probabilities (and without one they are the model's own);
  6. the exposure changes the total model only (the probability model ignores it);
  7. predictions returned with their own index are placed by position (A3's NaN finding).
- **Contract example:** the total model with `use_exposure=False` (the refit
  test passes no exposure).

Done when:
- [x] The output equals total × shares of the two pipelines fitted alone, and
  the §7 usage block runs as written under `-W error`.
- [x] Mutations, review, non-slow suite: 1221 passed (1 skipped, 1 xfailed).

**Record (2026-10-01).**
- **Baseline** before the first edit: 1203 passed (1 skipped, 1 xfailed);
  HEAD `42ebb2f` (B6), the tree clean.
- **Verified first** (probes with torch first): `clone` of a fitted
  `TemperatureCalibrator` drops `temperature_`, so **`clone(model_2)` with a
  plain fitted calibrator holds an unfitted one** (the tuner clones per
  trial; the contract refit test clones); `FrozenEstimator(calibrator)`
  keeps the fit through `clone`, its `predict` forwards and its `fit` is a
  no-op. sklearn's 4 contract checks pass on a class with two required
  estimator parameters and an optional `None` one; `get_params(deep=True)`
  lists `total_children_model__model__l2_penalty` and
  `cohort_probability_model__model__l2_penalty`, and nested `set_params`
  reaches them. `ModelPipeline` had no `predict_logits`. Calibrating
  `log(probabilities)` equals calibrating the logits (0.0; the row constant
  cancels), an alternative not taken.
- **Decisions (user: "best practices, do not overcomplicate; the settings
  easy to change"):** the calibrator is **used as given** (not cloned, not
  wrapped); the docstring and §7 say to wrap it in `FrozenEstimator` when
  the model is cloned, sklearn's own idiom for a prefit estimator inside a
  meta-estimator; no wrapping code. **`ModelPipeline.predict_logits(X)`**
  (the fitted transformer, then `model_.predict_logits`), so Model 2 never
  reaches into the pipeline and the §7 loop is `fold.predict_logits(X_val)`;
  the settings stay on the models inside the pipelines.
- **Built:** `IndependentTotalProbabilityModel(*, total_children_model,
  cohort_probability_model, temperature_calibrator=None)`; `fit` clones
  both, the total on `y.sum(axis=1)`, the probability model on `y`, the
  same exposure to both, `total_children_model_`,
  `cohort_probability_model_`, `cohorts_` and `temperature_calibrator_`
  (the calibrator as given) set together after both succeed. `predict`: the shares from `calibrator.predict(pipeline.predict_logits(X))`
  with a calibrator, else the probability model's own `predict` (wrapped in
  a DataFrame, so an array-returning model has numbered columns); **one
  check**, the shares' columns equal `y`'s at fit (other cohorts, or another
  order, would be multiplied in by position); the total as an array;
  `total[:, None] × shares` with `y`'s columns and `X`'s index. A
  `check_is_fitted` on the calibrator was built and then dropped: the
  calibrator's own `predict` raises `NotFittedError` already (its mutant
  survived). The annotation admits `FrozenEstimator`. Exported from
  `modeling` only.
- **Tests** (`test_modeling_independent_total_probability.py`, 12; data:
  features `x` and `z`, the exposure in no feature, Dirichlet-multinomial
  counts with a child in every building, non-default index): the product
  of the two pipelines fitted alone and rows summing to the total (under
  `filterwarnings("error")`; it also catches a probability model given the
  total's features, so the planned "own transformer" test was dropped as
  redundant); the total is `y`'s row sum and predict needs no target
  columns; doubling the exposure doubles every cohort exactly (so it
  reaches the total only); nested `set_params` reaches the next fit; a
  given calibrator is applied to the pipeline's logits; an unfitted or
  cloned plain calibrator raises `NotFittedError` and a `FrozenEstimator`
  survives `clone`; predict follows the calibrator given at fit; the
  templates stay unfitted; a failing second fit
  leaves the previous fit intact; columns follow `y`, rows follow `X`; a
  probability model with other cohort names raises; a total returned as a
  Series with its own index is placed by position. `test_modeling_pipeline.py`:
  `predict_logits` equals the inner model's logits on the transformed
  table. The contract example (6 cases): two pipelines, no exposure, no
  calibrator.
- **Ran:** §7's usage block (Model 1 and Model 2 parts) as written, under
  `-W error`, in one namespace after `FEATURE_TRANSFORMATIONS.md` §8.0, the
  `tree` block of §8.1, §8.2 and §8.3 (`cohort_probability_base` aliased to
  `composition_base` until B9 renames it), on seed 0: 49 test buildings,
  the DataFrame named and indexed as documented, T = 1.26, total deviance
  3.52, cohort log loss 1.086 (scratchpad `run_usage_b5.py`).
- **Mutations** (13 behaviors + the contract example), each failing its
  named test and restored by copy with md5 confirmed: the total fitted on
  the first cohort (2 tests); `+` for `×`; the probability model given the
  total's transformer (the product test); the exposure replaced at
  predict; the calibrator skipped; the calibrator setting read at predict;
  `clone` dropped; the total assigned before the second fit; `index=None`;
  the columns check dropped; `np.asarray` dropped on the total; the
  transform skipped, and the transformer refitted, in `predict_logits`;
  the contract example deleted (6 cases). Three mutants
  survived rightly, not being behaviors: reordering the assignments after
  both fits; the columns taken from the shares (equal by the check); and
  the calibrator's `check_is_fitted` (dropped, above).
- **Checks:** ruff and ruff format on the 6 changed `.py` files; `uv run
  mypy` and mypy on `modeling` plus the 3 test files clean; the 3 test
  files under `-W error` (49 passed).
- **Review** (independent subagent; its own end-to-end product, exposure,
  calibrator, clone and `FrozenEstimator` checks to 1e-14; 12 edge inputs;
  one mutation), each finding reproduced:
  - *Fixed.* `predict` read the calibrator from the setting, so
    `set_params(temperature_calibrator=...)` after `fit` changed a fitted
    model's output without a refit, unlike every other setting (§3 rule 7).
    Now `fit` stores it as `temperature_calibrator_` (still used as given)
    and `predict` reads that; new test `predict follows the calibrator given
    at fit` (mutation: the setting read at predict, both the branch and the
    call).
  - *Fixed.* The pipeline's `predict_logits` test ran on the training table,
    where a refitted `Center` gives the same values, so the "refitted
    transformer" mistake its comment named survived; it now runs on a subset
    (the refit mutant then fails it).
  - *Fixed.* `cohorts` is read from `y` before the two fits, so an ndarray
    `y` fails at once rather than after two fits (no check added: the
    signature says DataFrame).
  - *Fixed.* §7's shape said the calibrator is "checked fitted"; now "its own
    predict raises if unfitted". The never-fitted-calibrator case was
    dropped from the clone test (it tested the calibrator's own raise).
  - *Accepted.* The two `type: ignore[attr-defined]` over a `Protocol` (a
    class for the type checker only); the columns check at predict; the
    comments; `cohorts_: list[object]`.
  - *Fine.* A 1-column `y`; integer cohort names; a duplicated index at
    predict; a wrong-length or missing exposure; nested
    `set_params(total_children_model__model__family="nb2")`; pickling a
    calibrated fitted model; `evaluate` with `COHORT_LOG_LOSS`; a calibrator
    fitted on logits in another cohort order (T is a scalar).
- **Docs:** `MODULE_REFERENCE.md` (`__init__.py`, `pipeline.py`, new
  `independent_total_probability.py` rows); this doc (status, §7 shape and
  usage block, B5). `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 is B9's.

### B7. Smoke run
Ten populations. Per cohort: Model 2 (raw and calibrated) against Model 1; plus
the total's deviance and the cohort log loss. Table in the plan doc. Reuse A4's
recipe (A4's "Smoke run"; its script lived in a per-session scratchpad and
may be gone): `ShareTransformer` and `ExposureTransformer` on the full table, `Splitter("grouped")` with
`random_state=seed`, every array taken by `train_test_indices`'s positions
(B0a; A4 used `exposure.loc[X.index]`), each model against its own start).
Model 1's reference is A4's table (offset variant; total 4.215 ± 1.450).

Done when:
- [ ] The user has seen the numbers.

### B9. Docs and close
- `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` new §0 (the rebuilt model: equations,
  settings, calibration flow, the clone hazard);
  `FEATURE_TRANSFORMATIONS.md` §8.3 (rename `composition_base` to
  `cohort_probability_base`, as §7 uses, to match `CohortProbabilityModel`;
  user's decision, 2026-09-29), §8.7 item 7 (penalty ranges restated
  under N8: the old `C` range [0.01, 100] is about `l2_penalty` [2e-6, 2e-2] at
  ~4,500 training children); `FEATURE_TRANSFORMATIONS.md` §8.2 and §8.3 get
  runnable `ModelPipeline` blocks with `ExposureTransformer`, as §8.1 has;
  `MODULE_REFERENCE.md`; `docs/README.md`;
  `MODEL_REIMPLEMENTATION_PLAN.md` §5 (step 3 done). Pattern:
  `DIRECT_COHORT_MODEL.md` §0.6 (data flow, one runnable block, a table of
  rules with their reasons). Run every block with a runner as in A4's record
  (extract each section's `python` blocks, `exec` them in order in one
  namespace, set `raw_table` and the split frames), extended to the new section.

Done when:
- [ ] The user has reviewed the docs; PR B is ready to merge.

## 10. Old tests: carried over or dropped

Source: `tests/unit/test_independent_total_probability.py`.

| Carried over, in new form | Step |
|---|---|
| Exposure has a fixed unit coefficient (doubling doubles) | B2, B5 |
| Grouped fit equals literal per-child expansion | *dropped at B4: the Dirichlet fits the composition, not the children* |
| Zero totals add nothing | B3 (*B4: a building without children is rejected; the shares need one*) |
| An unobserved cohort raises | B4 |
| Probabilities sum to 1 and cohorts sum to the total | B4, B5 |
| Targets never enter a feature matrix | A3, B5 |
| Both models beat their constant baselines | B2, B4 |
| Optimizers report non-convergence | B1, B4 |
| The family is explicit | B8 |
| Temperature 1 is the plain softmax | B6 |

**Dropped with their features (N13, N15):** tuning inside `fit`, pointwise
log-probabilities, bootstrap draws and failures, the likelihood-ratio gate,
calibration metadata, seed provenance, state bundles.

## 11. Risks and pitfalls

- **Old-stack imports.** The package root imports the old stack, so
  independence holds per module. Done in A2:
  `test_modeling_never_imports_the_old_stack` checks `modeling/**/*.py` against
  the forbidden list in §4 (static imports only).
- **Same names, two stacks.** `from age_group_prediction import DirectCohortModel`
  gives the **old** class. Never export new classes from the root.
  `tests/unit/test_model_contracts.py` is the old contract test.
- **Types.** mypy cannot tell a Series from a DataFrame here; the tests must.
- **Zero-total buildings.** The simulated table may have none: use synthetic rows.
- **Tuning.** A convergence error inside a fold ends an Optuna study. Note it
  in `HYPERPARAMETER_TUNING_PLAN.md` §6 for Phase 3.
- **pydantic/ruff.** The repo has no `[tool.ruff]`; do not add one.
- **Three OpenMP runtimes (B8).** LightGBM (Homebrew's `libomp`), scikit-learn
  (its bundled copy) and torch (its own) each load one; `import
  age_group_prediction` loads LightGBM before torch, and torch's threaded
  kernels then segfault (`logsigmoid` even on 10 numbers; building a tensor
  above ~32k elements). A model's `fit` wraps every torch call, the tensors'
  construction included, in `single_threaded_torch()`; `threadpoolctl` cannot
  reach torch's setting. The documented remedy for a machine is one runtime per
  process (the LightGBM FAQ: symlink every copy to one `libomp`; never
  `KMP_DUPLICATE_LIB_OK`, which can give wrong results silently). In a probe,
  import `torch` first or set one thread; a test that needs LightGBM loaded
  first runs in a subprocess.
- **Contract test and the exposure.** The refit check calls `fit(X, y)`
  without an exposure, so an example must not need one (`use_exposure=False`).
- **Position, not index.** Rows are paired by position (N19). A sub-model
  returning a Series with its own index realigns into NaN when combined: take
  predictions as arrays (A3).
- **Doc blocks.** `fit_df`, `train_df`, `valid_df`, `test_df` are left to the
  reader; a doc-block runner sets them (A2's "Pitfalls met", A4's record).
- **Stale editor buffers.** If a file "looks unchanged" to the user: "File: Revert File".

## 12. Verification

| Purpose | Command |
|---|---|
| New tests | `uv run pytest tests/unit/test_modeling_*.py tests/unit/test_preprocessing.py tests/unit/test_scoring.py -q -W error` |
| Suite | `uv run pytest -m "not slow"` (record the count at each stop) |
| Oracles needing statsmodels | `uv run --group validation pytest tests/validation -k "total_children or nb2" -q` |
| Types | `uv run mypy` and `uv run mypy src/age_group_prediction/modeling src/age_group_prediction/preprocessing.py <changed test files>` |
| Lint | `uv run ruff check <files>`; `uv run ruff format <files>` |
| End to end | The smoke runs of A4 and B7, and the §7 usage block run as written |

Each PR is done when its steps are checked, the suite passes, the independent
review's findings are fixed, and the user has approved the final stop.
