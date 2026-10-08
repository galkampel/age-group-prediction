# Plan: rebuild Model 2 as `TotalTimesProbabilityModel` (scikit-learn classifiers, statsmodels NB2)

**Written for:** the implementing model (Claude Opus 5.5) and the user who
validates each sub-task. Self-contained: it assumes no memory of the planning
conversation. This file is the source of truth: update its status line and the
sub-task checkboxes in §8 as work finishes. Orientation for a new session:
[TOTAL_TIMES_PROBABILITY_MODEL_HANDOFF.md](TOTAL_TIMES_PROBABILITY_MODEL_HANDOFF.md).

**Status (2026-10-08):** approved by the user. **Sub-tasks 0–4 are done and committed**
(HEAD `fd843bc`, pushed; branch `feat/total-times-probability-model`; `DirectCohortModel` is now `CountModel`, with
an exposure branch and NB2 as `NegativeBinomialRegressor`; `CohortProbabilityModel` is a
classifier on weighted or per-child rows, optionally calibrated (`calibration_method`,
`calibration_cv`), and the torch Model 2 and `TemperatureCalibrator` are deleted;
records in §8). **Sub-task 5 is committed (`99d4dd0`):** `TotalTimesProbabilityModel` added,
`total_children.py` and `optimization.py` deleted (no torch left in `modeling/`). **Sub-task 6 (the
smoke run) is committed (`c9002b1`);** its tables and reading are in §8. **Sub-task 7 is split
in two stops (the user, 2026-10-08): 7a, the docs with every derivation, done and awaiting the
user's commit; 7b, the docstring trimming pass and the close, next.** During planning the user revised it twice: NB2 goes through
`CountModel` (the renamed `DirectCohortModel`) as an estimator taking the exposure (an offset in the first plan; raw `exposure` since sub-task 2); a
`replication` setting for row-resampling classifiers was considered and dropped
on measurement (P6, P8).

## Contents
1. Context and goal
2. Branch and PR
3. How to work
4. What exists today, and what goes
5. Decisions (P1–P15, with the answers to the questions asked in the request)
6. Measured evidence (2026-10-07)
7. Target code shape
8. Sub-tasks
9. Docs: what each decision's explanation must say
10. Risks
11. Verification

---

## 1. Context and goal

Model 2 (`modeling/independent_total_probability.py`, PR #11) predicts each
cohort's count as **predicted total × predicted cohort probability**. Its two
halves are hand-written torch objectives: `TotalChildrenModel` (Poisson/NB2
GLM via scipy on torch's log-likelihood) and `CohortProbabilityModel` (a
Dirichlet regression of the shares), plus a post-hoc `TemperatureCalibrator`
and `optimization.py` (`Minimizer`, `single_threaded_torch`, needed because
torch's and LightGBM's OpenMP runtimes crash together).

The user wants Model 2 re-implemented on **library estimators**, in the shape
PR #12 gave `DirectCohortModel` (any scikit-learn regressor as `estimator`):

- **Total:** any Poisson or Gaussian regressor, with or without the exposure,
  exactly as `DirectCohortModel` does it; plus **NB2** through a
  scikit-learn wrapper of statsmodels' `NegativeBinomial` that `DirectCohortModel`
  can take as its `estimator`, treating it differently (the exposure as an
  offset). The total model and the direct model are the same thing (a count
  regression of one column), so they are **combined**: `TotalChildrenModel`
  is deleted and `DirectCohortModel`, **renamed `CountModel`**, serves both.
- **Cohort probabilities:** any scikit-learn **multi-class classifier**
  (`LogisticRegression`, `HistGradientBoostingClassifier`, `LGBMClassifier`,
  `RandomForestClassifier`, or a `OneVsRestClassifier`), fitted on a
  **categorical representation** of the counts (weighted rows, or one row per
  child for estimators that resample rows), optionally **calibrated** with
  `CalibratedClassifierCV`.
- **Combined model:** renamed `TotalTimesProbabilityModel`; `fit` gives the
  row sums of `y` to the total model and `y` to the probability model;
  `predict` multiplies.
- **No torch** in `modeling/`. `TotalChildrenModel`, `TemperatureCalibrator`
  and `optimization.py` are deleted with their tests.
- **Docs** explain every decision: the NB2 representation, the data
  replication, bagging under replicated rows, and calibration.

Inputs stay as the base contract says (`modeling/base.py`): `X` the raw
table (or the design matrix when there is no `feature_transformer`), `y` a
DataFrame with one count column per cohort (≥ 2), `exposure` optional and
raw (not its log), rows paired by position.

## 2. Branch and PR

Branch `feat/total-times-probability-model` from `feat/hyperparameter-tuning`
(`c966edb`), draft PR into `feat/hyperparameter-tuning`, like PRs #10–#12.
The user commits and pushes; no `Co-Authored-By` line, no "Generated with"
footer.

## 3. How to work

The user's standing rules, as `docs/MULTI_COHORT_MODELS_PLAN.md` §3 states
them. In short:

1. **One sub-task per stop.** Start each in plan mode: re-verify this plan's
   facts for it, write a short summary of the sub-task in the chat, ask for
   approval, implement only after it.
2. **Baseline first:** `uv run pytest -m "not slow"` before the first edit;
   stop and report if it fails.
3. **If an assumption breaks, stop** and show the options with evidence.
4. **Routine:** implement → `uv run ruff check` / `uv run ruff format` on the
   changed files → `uv run mypy` and `uv run mypy src/age_group_prediction/modeling <changed tests>`
   → the changed tests with `-W error` → one mutation check per claimed
   behavior (edit `src` in place, restore by copy with md5, never `git
   checkout`/`stash`) → an independent review subagent (6–8 minutes: say so,
   update the plan doc and run the suite meanwhile), each finding reproduced
   before it is fixed → the non-slow suite → update the plan doc → stop.
5. **At each stop:** file-by-file summary of the `.py` changes, the check
   results, suggested commit commands.
6. **Justify every class, field and check**, or drop it. Validate only what
   would otherwise pass silently; leave to the library what it already
   raises. No default the user did not ask for: `estimator` is required.
7. **Code style:** match `modeling/count_model.py`. **Docstrings and comments hold only
   relevant information: concise but informative** (the user, 2026-10-08):
   - a docstring says what the code does, its inputs and outputs, and the non-obvious
     constraint a caller must know, in a few lines;
   - a comment states only a "why" the code cannot show;
   - derivations, measured figures, alternatives, history and library quirks go in the
     docs (§9), with at most a pointer from the code;
   - never restate the code or record revisions in it.

   **Every decision's derivation is documented** in the model docs (§9, sub-task 7).
8. **Probes** are read-only: `PYTHONPATH=src .venv/bin/python -c "..."`
   (`uv run` may re-sync the venv; `--no-sync` or the venv's python). Quote
   globs in zsh (`--include='*.py'`).
9. **Tests:** each test's name and comment state the mistake it catches; no
   test that only checks a library.

## 4. What exists today, and what goes

| File | Today | After this PR |
|---|---|---|
| `modeling/base.py` | `BaseAgeGroupModel`: `fit(X, y, exposure=None)`, `predict(X, exposure=None)`, `evaluate`, `_fit_features`, `_transform_features`, `_check_exposure` | unchanged |
| `modeling/direct_cohort.py` | `DirectCohortModel(estimator: Regressor, use_exposure, feature_transformer)`; exposure as the weighted rate (`y / exposure`, `sample_weight=exposure`, `predict × exposure`) | **renamed `modeling/count_model.py`, `CountModel`**; an estimator whose `fit` takes `offset` gets `log(exposure)` as the offset instead of the rate (P3) |
| `modeling/independent_cohorts.py` | Model 1, `IndependentCohortModels` (and its type alias `CohortModels`) | unchanged (it never names `DirectCohortModel`; only its test does) |
| `modeling/total_children.py` | torch Poisson/NB2 GLM | **deleted** |
| `modeling/negative_binomial.py` | — | **new:** `NegativeBinomialRegressor`, a scikit-learn regressor around statsmodels NB2 with `fit(X, y, exposure=None)` and `predict(X, exposure=None)` |
| `modeling/cohort_probability.py` | torch Dirichlet regression, `predict_logits` | **rewritten:** a classifier on the categorical representation (one weighted row per building and cohort), optional `CalibratedClassifierCV` |
| `modeling/calibration.py` | `TemperatureCalibrator` | **deleted in sub-task 3** (`CalibratedClassifierCV(method="temperature")` replaces it in sub-task 4) |
| `modeling/optimization.py` | `Minimizer`, `Solver`, `single_threaded_torch` | **deleted** (only the two torch models use it; the old stack's `models/count_regression.py` has its own) |
| `modeling/independent_total_probability.py` | `IndependentTotalProbabilityModel(total_children_model, cohort_probability_model, temperature_calibrator)` | **deleted in sub-task 3** (it needed the torch `CohortProbabilityModel`); replaced in sub-task 5 by `modeling/total_times_probability.py`, `TotalTimesProbabilityModel(total_model, probability_model)` |
| `modeling/__init__.py` | exports the above | `CountModel`, `Regressor`, `ExposureRegressor`, `NegativeBinomialRegressor`, `Classifier`, `CalibrationMethod`, `CohortProbabilityModel`, `TotalTimesProbabilityModel`, Model 1's names; gone: `DirectCohortModel`, `Solver`, `TemperatureCalibrator`, `TotalChildrenModel`, `IndependentTotalProbabilityModel` |
| `pyproject.toml` | `statsmodels` only in the `validation` group | `statsmodels>=0.14.5` in `dependencies` |
| `tests/unit/test_modeling_total_children.py`, `test_modeling_calibration.py`, `test_modeling_optimization.py`, `tests/validation/test_total_children.py` | tests of the deleted code | **deleted** (`test_modeling_calibration.py` in sub-task 3, the rest in sub-task 5) |
| `tests/unit/test_modeling_direct_cohort.py` | `DirectCohortModel` | renamed `test_modeling_count_model.py`; the exposure branch added |
| `tests/unit/test_modeling_cohort_probability.py`, `test_modeling_independent_total_probability.py` | tests of the torch builds | the first **rewritten** (sub-task 3); the second **deleted** in sub-task 3, `test_modeling_total_times_probability.py` new in sub-task 5 |
| `tests/unit/test_modeling_negative_binomial.py` | — | **new** |
| `tests/unit/test_modeling_contract.py` | `EXAMPLES` per concrete model (line 136 after sub-task 5) | entries renamed/added; a model without an example fails `test_every_shipped_model_has_an_example` |
| `tests/unit/test_modeling_feature_transformer.py`, `test_modeling_independent_cohorts.py`, `tests/unit/test_hyperparameter_tuning_evaluator.py` (builds a `DirectCohortModel`: lines 40, 296, 435) | use the old names | updated |
| `docs/DIRECT_COHORT_MODEL.md` §0 | `DirectCohortModel` | file name kept; §0 says the class is `CountModel`, used for a cohort and for the total, with the exposure rule. *7a revision 1 (the user): §0 is a pointer; the rebuilt model has its own doc, `docs/INDEPENDENT_COHORT_MODELS.md`* |
| `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 | the torch build | §0 rewritten (the file keeps §1–§12 for the old stack until roadmap step 5 deletes it). *7a revision 1 (the user): §0 is a pointer; the rebuilt model has its own doc, `docs/TOTAL_TIMES_PROBABILITY_MODEL.md`, 400–500 lines* |
| `docs/FEATURE_TRANSFORMATIONS.md` §8.1–8.3, `docs/MODULE_REFERENCE.md`, `docs/README.md`, `docs/MODEL_REIMPLEMENTATION_PLAN.md` §5, `docs/HYPERPARAMETER_TUNING_PLAN.md`, `docs/DIRECT_COHORT_GENERALIZATION_PLAN.md` (a top note only) | name the old classes | updated |

Reused as is: `BaseAgeGroupModel._fit_features` / `_transform_features` /
`_check_exposure` (`modeling/base.py:76-133`), `preprocessing.ExposureTransformer`,
`scoring.COHORT_LOG_LOSS` (per child; it "equals scikit-learn's `log_loss` on
one row per child", which is exactly the representation below),
`splitting.Splitter`, `utils.take_rows`, sklearn's
`sklearn.utils.validation.has_fit_parameter`.

The old stack (`models/`, `modeling_config.py`, `experiment/`, `tracking/`)
imports its own `IndependentTotalProbabilityModel` from
`age_group_prediction.models`; nothing there imports `modeling`, so it is
untouched (deleted by roadmap step 5).

## 5. Decisions

| # | Decision | Why |
|---|---|---|
| P1 | **Inputs:** `X` raw table, `y` DataFrame of cohort counts (≥ 2 columns), `exposure` optional raw. Unchanged base contract | The user's specification; nothing else in the repo changes |
| P2 | **`DirectCohortModel` is renamed `CountModel`** (`modeling/count_model.py`): a regression of one count column, a cohort's or the total, with any estimator and an optional exposure. **It is Model 2's total model**; `TotalChildrenModel` is deleted | The user: "this model and the direct model should be similar, can combine the two", and asked whether to rename once it serves the total too; `CountModel` chosen over `CountRegressionModel`. `DirectCohortModel` already gives Poisson and Gaussian losses with and without the exposure (the weighted rate is the Poisson offset model exactly; measured again, coefficient difference 2e-13). Its torch Poisson with `l2_penalty` is `PoissonRegressor(alpha=…)` (pinned to 3e-7 in PR #11's N6). "Model A" stays the docs' name for the direct approach |
| P3 | **`CountModel` takes two kinds of estimator and tells them apart with sklearn's `has_fit_parameter(estimator, "exposure")`.** An estimator whose `fit` has an `exposure` parameter (`ExposureRegressor` protocol: `fit(X, y, exposure=None)`, `predict(X, exposure=None)`) is given the raw exposure at `fit` and `predict`; any other (`Regressor`, as today) gets the weighted rate. Without `use_exposure` both are fitted plainly. The fitted copy decides at `predict` (the same test on `estimator_`). *Revised by the user in sub-task 2 (2026-10-07): the argument is statsmodels' own raw `exposure`, not `offset=log(exposure)`* | The user: "DirectCohortModel should be able to use NB2 as well: identify the NB2 model and treat it differently". **Why not the rate for NB2:** the weighted-rate form is exact for Poisson only; for NB2 it fits a different model (measured: coefficients differ by 0.021), and statsmodels' NB accepts the non-integer `y / exposure` silently. **Why a signature test, not `isinstance`:** it is sklearn's own idiom for `sample_weight` (`has_fit_parameter`), it names the capability rather than one class, so any future estimator taking the exposure gets the same treatment, and it is measured to say `False` for `PoissonRegressor`, `LGBMRegressor`, `LinearRegression` and HGB. **Why the raw exposure, not an offset** (the user): it is what statsmodels calls `exposure` (the log is taken inside, coefficient 1) and what every model here takes ("raw, not its log"); measured identical to `offset=log(exposure)` (difference 0.0) |
| P4 | **NB2 is `NegativeBinomialRegressor`** in `modeling/negative_binomial.py`: a scikit-learn regressor (`BaseEstimator`, `RegressorMixin`) around statsmodels' `discrete_model.NegativeBinomial(loglike_method="nb2", exposure=…)`, `fit(method="bfgs", maxiter=max_iter, disp=0)` (sub-task 2 adds the BFGS preliminary fit, `has_constant="add"`, `skip_hessian=True` and the warning handling: §8 record); fitted `intercept_`, `coef_`, `dispersion_` (α), `n_features_in_`/`feature_names_in_` (`validate_data`); `predict(X, exposure=None)` = `exposure · exp(b + Xβ)` from the stored coefficients (what statsmodels' `predict(exposure=)` computes), the exposure's shape checked. **Exposure only, no `offset`** (the user asked whether to allow both): no model here has a second fixed log-scale term, and two ways of giving the same thing add up silently if both are passed; an `offset` is one argument to add if ever needed. Unpenalized; `max_iter=500` is its one setting. A fit that does not converge raises. statsmodels moves to the main dependencies | The user asked for a scikit-learn wrapper of statsmodels' NB2 with an optional exposure and no torch. As a plain regressor it plugs into `CountModel` (P3), so Model 1's cohorts and Model 2's total get NB2 the same way. **"Can I predict using the exposure component?"** Yes: `results.predict(exog, exposure=e)` (equivalently `offset=log e`) returns `e · exp(b + Xβ)`. **Alternatives** (table in §9): statsmodels `GLM(family=NegativeBinomial(alpha))` (α must be fixed; L2 via `fit_regularized(L1_wt=0)`: a penalized variant for later if the smoke run needs one), the current torch objective (dropped: no torch), `glum` (not installed; θ fixed), LightGBM/XGBoost (no NB objective; a custom objective would re-create the torch code), scikit-learn (none: `TweedieRegressor` is not NB) |
| P5 | *Revision 3 (the user):* the labels are the cohorts' **column positions** in `y` (0..K−1), not their names; `multinomial_to_categorical` returns `(sample_positions, cohort_positions, weights | None)`. *Revised by the user in sub-task 3, revision 1 (2026-10-07):* the static method is `multinomial_to_categorical(y, replication) -> (positions, labels, weights | None)` (the user: "`to_categorical` is misleading"), with both representations of P6. As first planned: **The categorical representation is built inside `CohortProbabilityModel.fit`,** by a static method `CohortProbabilityModel.to_categorical(y) -> (positions, labels, weights)`: one row per `(building, cohort)` with a positive count, `labels` the cohort name, `weights` the count (`np.nonzero(counts)`). `X` is the design matrix's rows at `positions`. The feature transformer is fitted on the **original** rows first | The user: inside only if both the raw and the categorical targets are needed there. They are: (a) the feature transformer must see one row per building, as in every other model (a replicated fit would weight the training statistics by children); (b) the calibration folds must keep a building's rows together (P10), which needs the original row ids; (c) `fit(X, y: counts)` keeps the base contract, so the probability model is fitted, scored (`COHORT_LOG_LOSS` on counts) and tuned on its own, and `TotalTimesProbabilityModel` only passes `y` on |
| P6 | *Revised by the user in sub-task 3, revision 1 (2026-10-07):* **both replications, a setting `replication: ReplicationType = "weighted"`** (`Literal["per_child", "weighted"]`; named in revision 2): "weighted" one row per (building, cohort) with `sample_weight` = the count, "per_child" one row per child and no `sample_weight` passed (the user: "set two categorical representations; the idea is to check both"). Measured on the simulator (§6): identical for LR; for the trees within noise per seed (replicated − weighted −0.002 to +0.010; `min_samples_leaf`/`min_child_samples` count rows, not weights); "per_child" fits 2–3× slower and admits classifiers without `sample_weight` (KNN, LDA, QDA, Gaussian process, `OneVsRestClassifier` without routing); it needs integer counts (`np.repeat` raises on floats). Never `count / total`. As first planned: **Weighted replication only, weight = count (per child).** No `replication` setting; not one row per child; never `count / total` | The user's choice of weighting, and the evidence against a setting. For an estimator that fits a weighted likelihood the two replications are **equivalent:** a `sample_weight` of `c` multiplies that row's log-likelihood term by `c`, exactly what `c` identical rows contribute; measured on `LogisticRegression`: coefficient difference 1.8e-15 (442 weighted rows vs 1,325 per-child rows). For an estimator that **resamples rows** (RF) they are not identical fits, but the held-out quality is the same within noise (P8's table: 0.712 vs 0.708 against a seed spread of 0.57–0.86), so a setting would serve no measured purpose; it is a one-argument extension of `to_categorical` if real data ever shows one. The weighted rows are the multinomial likelihood of the counts given the total, the quantity `COHORT_LOG_LOSS` scores; `count / total` (one unit per building, the Dirichlet build's weighting) is a different estimator (coefficients move by 0.06) and not that likelihood |
| P7 | *Revision 3 (the user):* with position labels, `classes_` is `0..K−1` in `y`'s order (every cohort observed), so `predict` names the columns with `cohorts_` directly; no by-name mapping, and cohort names of mixed types work (as names they fail sklearn's sort). *Revised by the user in sub-task 3, revision 1 (2026-10-07):* **no one-vs-rest detection or wrap** (the user, after the survey in §6: no common classifier lacks a multiclass fit; binary-only ones raise; scikit-learn's `multi_class` tag is `True` for every classifier, so it cannot tell); the caller wraps explicitly. **`predict` raises if a row of probabilities does not sum to 1** (atol 1e-6; softmax classifiers measured within 2.2e-16): LightGBM's `objective="multiclassova"` returns rows summing to 0.67–1.28 silently, and Model 2 would then predict cohorts that miss its total; the valid objectives are LightGBM's `"multiclass"` (its default above 2 classes), CatBoost's `"MultiClass"` and XGBoost's `multi:softprob` (both from the docs, not installed). This replaces "the model asserts nothing" below. As first planned: **`estimator: Classifier`**, a `Protocol`: `fit(X, y, sample_weight=None)`, `predict_proba(X)`, `classes_`. Multinomial classifiers (`LogisticRegression`, whose lbfgs is multinomial in sklearn 1.9, `multi_class` is gone; `HistGradientBoostingClassifier`; `LGBMClassifier`, `objective_ = "multiclass"` when ≥ 3 labels; `RandomForestClassifier`) and `OneVsRestClassifier(binary)` alike. `predict` maps `predict_proba`'s columns **by `classes_`**, never by position, into `y`'s column order at fit (until revision 3: now the labels are positions) | `classes_` are sorted labels (`['el', 'hs', 'kg']` for `['kg', 'el', 'hs']`): a positional mapping would permute cohorts silently. **Rows sum to 1** for all of them: `OneVsRestClassifier.predict_proba` normalizes in the multiclass case (measured 2e-16), as do HGB, LightGBM, RF and `CalibratedClassifierCV` (sigmoid and isotonic are per-class, then normalized; temperature is a softmax). The model asserts nothing about it: it is what the libraries do, and `COHORT_LOG_LOSS` renormalizes anyway. `OneVsRestClassifier.fit` takes `sample_weight` only through metadata routing (`set_fit_request(sample_weight=True)` on the inner estimator, `sklearn.set_config(enable_metadata_routing=True)`); documented, not special-cased |
| P8 | *Revised by the user in sub-task 3, revision 1 (2026-10-07):* the per-child representation now exists as `replication="per_child"` (P6), so "no representation switch" below no longer holds; grouped bagging is still not built. As first planned: **Bagging and bootstrap under replication: nothing is built; the estimators' own resampling is left as the user sets it.** The docs explain the units and give the measured table | **The user's question:** the cohorts of a building are connected (they share `x_b`, sum to `Y_b`, and their proportions sum to one), so should a resample keep the building's rows together? **The answer in three parts.** (1) The sum-to-one constraint is on the model's output `p_b`, which every classifier's `predict_proba` (and `CalibratedClassifierCV`) enforces by softmax or normalization; it is not a dependence between rows. (2) Under the conditional multinomial model `C_b \| Y_b ~ Mult(Y_b, p_b)` is `Y_b` independent categorical draws, so the exchangeable unit is the **child**; the weighted representation merely compresses identical child rows into one cell per cohort. A row resampler then draws **cells** (a bag can hold building b's kindergarten cell and drop its elementary cell) instead of children (a bag thins each building's composition at random); neither keeps a building whole; only a bootstrap **by building** does, which scikit-learn and LightGBM do not offer. (3) **Measured** (RF, 300 trees, 5 seeds, held-out cross-entropy against the true `p`): weighted + bootstrap 0.712, per-child + bootstrap 0.708, weighted without bootstrap 0.716, per-child without bootstrap 0.717, a hand-made bootstrap by building 0.717, against a seed-to-seed spread of 0.57–0.86: **all the same within noise**. So no representation switch and no grouped bagging is justified. **What the libraries do** (the user's "permutation" is not it): LightGBM's bagging is subsampling **without replacement** (`subsample`, active only with `subsample_freq > 0`; off by default); `HistGradientBoostingClassifier` has **no** row subsampling and its early stopping is off under 10,000 rows; `LogisticRegression` has none; `RandomForestClassifier` bootstraps **with replacement**, on by default (with `sample_weight`, sklearn multiplies the weight by the draw count); XGBoost (not a dependency; from its parameter docs) uses **all rows** by default, `subsample=1.0`, and subsamples without replacement per round only when set below 1. So RF is the only common estimator that resamples by default, and the user chose to leave it as is. **Observed building features** (type, year) change nothing: every row of a building shares its whole `x_b` already, and conditioning on more of it makes the conditional independence of its children more plausible, not less; the grouped unit becomes the right one only for an **unobserved** building effect (a random building intercept in the generator, or a generated characteristic withheld from the model) or for uncertainty intervals. **If real data shows extra-multinomial variation between buildings** (the case where the building is the right unit): `bootstrap=False` on RF removes the resampling (randomness then comes from `max_features`), or a bootstrap-by-building aggregator over `CohortProbabilityModel` (the ten-line probe above), added only then. The folds that must be grouped are the calibration folds (by building, P10) and the tuner's (by neighborhood, `Splitter`) |
| P9 | **Calibration is `CalibratedClassifierCV` inside `CohortProbabilityModel`:** `calibration_method: CalibrationMethod \| None = None` (`Literal["temperature", "sigmoid", "isotonic"]`, `None` = the estimator's own probabilities), `calibration_cv: int = 5`. `fit` wraps the cloned estimator as `CalibratedClassifierCV(estimator, method=…, cv=<grouped splits>, ensemble=False)` and fits it with the weights. **No `TemperatureCalibrator`** | The user's specification. `ensemble=False` is **cross-fitting**: the estimator is fitted on each of `k` folds, its out-of-fold probabilities for **every** row are collected, **one** calibrator is fitted on them, then the estimator is refitted on all rows; `predict` is that one model through that one map. `ensemble=True` (sklearn's default for a non-frozen estimator) averages `k` calibrated fold models and never fits on all rows. Cross-fitting is the user's preference and what the previous build did by hand (the B6 loop). `TemperatureCalibrator` duplicated sklearn's `_TemperatureScaling`, whose objective it was pinned to; the calibration now sits in the model, so clones and tuning (`probability_model__calibration_method`) carry it, and no `FrozenEstimator` idiom is needed |
| P10 | *Revised by the user in sub-task 4 (2026-10-08): the samples are split **before** the categorical rows, round-robin (sample `i` in fold `i mod calibration_cv`), and each row goes to its sample's fold (`PredefinedSplit(sample_folds[sample_positions])`), so both replications calibrate on the same folds (folds of rows balanced cells under `"weighted"` and children under `"per_child"`: different partitions, `beta_` 0.963 vs 0.970 for the same unpenalized LR; now equal to 2e-13).* As first planned: **Calibration folds: `GroupKFold(n_splits=calibration_cv)` over the categorical rows with `groups=sample_positions`** (the building; under `"per_child"` repeated by count), unshuffled, passed as a list of splits | A building's rows carry its known composition; with plain `KFold` the same building sits in a fit fold and its calibration fold, and the calibrator sees in-sample confidence. Grouping by building removes that; grouping by **neighborhood** (the repo's evaluation unit) would need `groups` at `fit`, which the base contract does not carry: documented as the limitation (the tuner's outer folds are by neighborhood regardless). Unshuffled `GroupKFold` is deterministic, so no `random_state` setting (round-robin likewise, and each fold spans the table even if it is sorted) |
| P11 | **Folds: 5. Isotonic is allowed but documented as inappropriate here** | **How many:** the calibrator must map the **final** model's probabilities, but it is fitted on fold models trained on `(k−1)/k` of the rows, which are less confident than the final one; `k = 2` (50/50) calibrates a model fitted on half the data and biases the temperature toward sharpening; larger `k` approaches the final model at the cost of `k` fits; with ~245 buildings per training set, 5 (sklearn's default) leaves ~200 buildings per fold fit and uses every row for calibration *(stale, found in 7a's review: 245 is the whole table; the grouped 80/20 training sets hold 179–206 buildings, 143–165 per fold fit)*. **How much is enough:** temperature fits 1 parameter and sigmoid 2 per class, so every row of a 5-fold cross-fit (~1,300 child-weighted rows) is ample; isotonic is non-parametric per class and sklearn advises it only well above ~1,000 samples per class, so it overfits here (the previous plan's N12 said the same). Measured: all three run on the categorical rows; on the simulator the fitted inverse temperature (`beta_`, `softmax(beta_ · logits)`) is 0.88 for unpenalized LR and 0.47 for default LightGBM (sub-task 4 table) |
| P12 | **`TotalTimesProbabilityModel(total_model, probability_model)`** in `modeling/total_times_probability.py`: `fit` clones and fits `total_model` on `y.sum(axis=1)` and `probability_model` on `y`, both with `exposure`; `predict` = `total[:, None] × probabilities`, a DataFrame with `y`'s columns at fit, indexed like `X`; raises if the probability model's columns differ from `cohorts_`. No calibrator argument. *Revised in sub-task 5 (2026-10-08; the user: "compare the two, choose by best practice, do not overcomplicate"):* **no column check, no `cohorts_`**: `probability_model: CohortProbabilityModel` and `predict` = `probabilities * total[:, None]`, so the columns and index are the probability model's own DataFrame's (`y`'s columns at fit, `X`'s index) and nothing can be mislabelled; the check could fire only for another probability model reordering its columns, which does not exist (widening the type is one line if one is ever written) | The user's name (chosen over `TotalSplitModel`, `TotalCompositionModel`): it names the prediction. Settings named for their role (`total_model`, `probability_model`); nested names reach both (`total_model__estimator__alpha`, `probability_model__estimator__C`, `probability_model__calibration_method`) |
| P13 | **A building with no children contributes to the total model only** (it has no row in the categorical representation). **A cohort with no child in `y` raises at `fit`** | The classifier cannot learn an absent class, and `predict` would lack its column: raise early with the cohort's name, as today. The Dirichlet build's "every building needs a child" rule disappears |
| P14 | **No torch in `modeling/`;** `torch` stays a dependency of the old stack (`pyro`). The contract test's single-thread LightGBM note stays until the old stack goes | The user: "Do not use torch". Deleting `optimization.py` removes the OpenMP guard with the code that needed it; LightGBM alone did not crash |
| P15 | **Penalty settings are the estimators' own** (`PoissonRegressor(alpha)`, `LogisticRegression(C)`, LightGBM's); `NegativeBinomialRegressor` has none | `DirectCohortModel`'s rule since PR #12. The docs note that sklearn's penalized GLMs normalize `sample_weight`, so `alpha` acts as `alpha × mean(exposure)` under the rate form, and that `C` on the replicated rows is per **child** |

## 6. Measured evidence (2026-10-07; sklearn 1.9.0, lightgbm 4.7.0, statsmodels 0.14.6, scipy 1.18.0)

Re-run any figure you rely on. Synthetic data: 240–245 buildings, 3 features,
3 cohorts, totals `Poisson(4.8) + 1`, multinomial counts; the NB2 data drawn
with α = 0.1 and exposure 5–40.

| Question | Result |
|---|---|
| Weighted replication (442 rows, weight = count) vs one row per child (1,325 rows), unpenalized `LogisticRegression` | max coefficient difference **1.8e-15** |
| Weight = `count / total` (per building) vs per child | coefficients differ by **0.059** |
| `RandomForestClassifier(300, min_samples_leaf=5)` on weighted rows vs per-child rows | `predict_proba` differs pointwise by up to 0.26 with bootstrap on (two different forests), but **held-out cross-entropy against the true `p`** (5 seeds, 2,000 test buildings): weighted + bootstrap **0.712**, per-child + bootstrap **0.708**, weighted `bootstrap=False` 0.716, per-child `bootstrap=False` 0.717, hand-made bootstrap by building (300 trees on resampled buildings) 0.717; seed spread 0.57–0.86. No difference beyond noise |
| `has_fit_parameter(estimator, "offset")` | `True` for a class with `fit(X, y, offset=None)`; `False` for `PoissonRegressor`, `LGBMRegressor` |
| `OneVsRestClassifier(LogisticRegression)`, HGB, `LGBMClassifier` row sums of `predict_proba` | all within 2.2e-16 of 1; `LGBMClassifier.objective_ == "multiclass"` |
| `CalibratedClassifierCV(…, cv=list(GroupKFold(5, shuffle=True).split(X_rep, labels, groups=building)), ensemble=False).fit(X_rep, labels, sample_weight=weights)` (planning; name labels, shuffled folds: superseded by the sub-task 4 rows) | runs for temperature, sigmoid and isotonic; one calibrated model (`len(calibrated_classifiers_) == 1`); rows sum to 1; `classes_ == ['el', 'hs', 'kg']`; fitted `beta_ = 1.00006` (an inverse temperature) |
| `CalibratedClassifierCV` signature | `(estimator=None, *, method='sigmoid', cv=None, n_jobs=None, ensemble='auto')`; `fit(X, y, sample_weight=None, **fit_params)`; `_TemperatureScaling.fit(X, y, sample_weight=None)` |
| `OneVsRestClassifier.fit` | `(X, y, **fit_params)`: `sample_weight` needs metadata routing (P7) |
| LightGBM bagging defaults | `subsample=1.0`, `subsample_freq=0`; `bagging_by_query` is not in the Python parameter list |
| `HistGradientBoostingClassifier` | no `subsample` parameter; `early_stopping="auto"` (off below 10,000 rows) |
| `RandomForestClassifier().bootstrap` | `True` |
| `GroupKFold` | `(n_splits=5, *, shuffle=False, random_state=None)` |
| statsmodels `NegativeBinomial(y, add_constant(X), exposure=e).fit(disp=0, maxiter=500)` | converged, **10 ms**, α 0.121 (true 0.1); no warning under `-W error` (the `ConvergenceWarning` seen once came from `fit_regularized`) |
| `results.predict(exog, exposure=e)` vs `exp(b + Xβ) · e` | 1.1e-14 |
| statsmodels NB2 vs the repo's torch `TotalChildrenModel(family="nb2")` | coefficients 7.7e-7, α 7.8e-8 |
| **NB2 offset vs weighted rate** (`GLM(NegativeBinomial(alpha))`, `offset=log e` vs `y/e, var_weights=e`) | coefficients differ by **0.021**: the rate form is not the offset model for NB2 |
| Poisson offset vs weighted rate, the same way | 2.0e-13: identical |
| statsmodels NB penalties | `NegativeBinomial.fit_regularized` is `method='l1'` only (`L1_wt` is swallowed by `**kwargs`); `GLM.fit_regularized(alpha, L1_wt=0)` gives L2 with α fixed |
| statsmodels NB with non-integer `y` | fits silently (so a rate form would not fail) |
| `glum` | not installed |
| *Sub-task 2 probes:* NB2 offset fit on the simulator's table (245 rows, 7 features), BFGS | converged, 6 ms, α 0.1002, no warning under an "error" filter; BFGS vs Newton (`tol=1e-12`) 8.5e-8 |
| A fit that does not converge (`max_iter=1`) | statsmodels **warns** (`HessianInversionWarning`, then `ConvergenceWarning`), `converged=False`; under an "error" filter the first is raised before any check of ours. statsmodels' import sets `simplefilter("always", ConvergenceWarning)`, which overrides `python -W error` but not pytest's per-test filter |
| An all-zero or all-ones column, default preliminary fit (Newton) | `LinAlgError: Singular matrix`; with `optim_kwds_prelim={"method": "bfgs"}` both converge, the all-ones column's predictions equal the fit without it to 1.3e-5; the normal case unchanged |
| `add_constant(X)` (default `has_constant="skip"`) on a design with an all-ones column | no intercept added (4 columns, not 5): the parameters shift by one, so `params[0]` would be a coefficient. Hence `has_constant="add"` |
| Non-integer `y` (y + 0.5) | fits silently (α 0.038 instead of 0.096); negative `y` ends in non-convergence |
| `has_fit_parameter(…, "offset")` | also `False` for `LinearRegression`, `HistGradientBoostingRegressor` |
| *Sub-task 2 revision:* statsmodels NB2 `exposure=e` vs `offset=log e` | identical parameters (difference 0.0); statsmodels accepts both at once (they add up); an exposure of 0, negative or NaN ends in non-convergence |
| `has_fit_parameter(…, "exposure")` | `False` for `PoissonRegressor`, `LinearRegression`, HGB, `LGBMRegressor` |
| mypy 2.3.1, `TypeIs[ExposureRegressor]` on `Regressor \| ExposureRegressor` (`OffsetRegressor` then) | narrows the `True` branch only; the `False` branch stays the union (hence one `cast`) |
| mypy 2.3.1, revision 2: two named checks on `Regressor \| ExposureRegressor` | `TypeIs` treats the protocols as overlapping (the first case stays the union, the second is marked unreachable); `TypeGuard` narrows each case to exactly its type: no `cast` |
| `has_fit_parameter(…, "sample_weight")` | `True` for `LGBMRegressor`, HGB, `PoissonRegressor`, `LinearRegression`; `False` for `NegativeBinomialRegressor`, `Pipeline`, `TransformedTargetRegressor` (sklearn's `BaggingRegressor` uses the same test) |
| *Sub-task 3 probes:* a negative `sample_weight` | fitted silently by `LogisticRegression`, HGB and `LGBMClassifier`; `RandomForestClassifier` raises. A NaN weight: `LGBMClassifier` silent, `LogisticRegression` raises; `np.nonzero` selects NaN cells |
| Reordered DataFrame columns at `predict_proba` | `LGBMClassifier` silent; LR, HGB, RF raise "feature names should match". `CountModel(LGBMRegressor)` without a transformer: predictions differ by 2.41, no error (§10 follow-up) |
| Unpenalized `LogisticRegression` under `-W error` | `C=np.inf` clean; `penalty=None` raises a `FutureWarning` (deprecated in 1.8) |
| `OneVsRestClassifier` with routing on | `set_fit_request(sample_weight=True)` on the inner LR: the weights reach it (the fit moves by 0.065) |
| The new class, weighted rows vs one row per child (`LogisticRegression(C=np.inf)`) | coefficients 2.8e-16 |
| *Sub-task 3, revision 1:* weighted vs replicated rows, 5 simulated populations (all buildings; 189 training buildings, 571 weighted rows vs 4,543 per-child rows), held-out `COHORT_LOG_LOSS` (replicated / weighted; replicated − weighted per seed; fit seconds) | LR (`C=inf`) 1.0783 / 1.0783, 0; 0.005 / 0.003. HGB 1.0899 / 1.0869, −0.001 to +0.010; 3.7 / 1.3. LGBM 1.0884 / 1.0850, −0.002 to +0.006; 0.12 / 0.04. RF (300 trees) 1.0822 / 1.0817, −0.001 to +0.005; 0.38 / 0.19 |
| Every scikit-learn 1.9 classifier fitted on 3 classes | all fit except `FixedThresholdClassifier`, `TunedThresholdClassifierCV` ("Only binary classification"); no `predict_proba`: `LinearSVC`, `NuSVC`/`SVC` (without `probability=True`), `Perceptron`, `PassiveAggressive`, `RidgeClassifier(CV)`, `SGDClassifier` (default hinge loss), `OneVsOne`, `OutputCode`; no `sample_weight` in `fit`: KNN, `RadiusNeighbors`, `NearestCentroid`, LDA, QDA, `GaussianProcessClassifier`, `LabelPropagation`/`LabelSpreading`, `OneVsRestClassifier`. `get_tags(...).classifier_tags.multi_class` is `True` for every one, `LGBMClassifier(objective="binary")` too |
| LightGBM objectives on 3 classes | `"multiclass"`: rows sum to 1; `"multiclassova"`: rows sum to 0.67–1.28, no error; `"binary"`, `"cross_entropy"`: raise "Number of classes must be 1" |
| `OneVsRestClassifier(...).fit(X, y, sample_weight=None)` without routing | raises (any extra keyword needs metadata routing): so `"per_child"` passes no keyword |
| Held-out `COHORT_LOG_LOSS`, 5 simulated populations (seeds 0–4; buildings with a child; 4 standardized features; grouped 80/20 by neighborhood) | torch Dirichlet build **1.0819** (measured before any edit); the new model with `LogisticRegression(C=np.inf)` **1.0783** (1.0895, 1.0712, 1.0864, 1.0632, 1.0811) |
| *Sub-task 4 probes (2026-10-08):* `CalibratedClassifierCV(…, cv=list(GroupKFold(5).split(…, groups=sample_positions)), ensemble=False)` on position labels, LR, LGBM, RF × temperature, sigmoid, isotonic | one calibrated model; `classes_ == [0, 1, 2]` (int64); rows sum to 1 within 2.2e-16; no sample on both sides of a split. An invalid `method` raises `InvalidParameterError`; `GroupKFold(1)` and more splits than groups raise. With a list as `cv` sklearn's "fewer than n_folds examples per class" pre-check is skipped; `cv=int` would be stratified, not grouped |
| Folds of categorical rows vs folds of samples (P10 revision; unpenalized LR, temperature, the unit tests' data) | `GroupKFold` over the rows: `beta_` weighted 0.962802, per-child 0.969523 (different partitions). Samples first, round-robin: 0.968229 both (2.2e-13); `KFold` blocks: 0.972192 both. 57–60 samples per calibration fold. Round-robin fold counts: 1, 0 and negative raise (sklearn: "Found array with 0 sample(s)", "only works for partitions"); more folds than samples give one fold per sample (1000 → 296 splits), slower, not wrong |
| With calibration (review) | `"weighted"` + a classifier without `sample_weight` (KNN, OvR without routing) only warns and fits it unweighted (without calibration it raises); with metadata routing on, every calibrated classifier needs `set_fit_request(sample_weight=True)` (`UnsetMetadataPassedError`, loud); LightGBM `"multiclassova"` is renormalized by every method (rows 1 ± 2e-16), so the row-sum check does not fire |
| `CalibratedClassifierCV`'s response method | `decision_function` when the estimator has it, else `predict_proba`: the calibrator sees logits for LR, LGBM, HGB, probabilities for RF, KNN |
| Weights through the calibration (`sample_weight` = counts vs none, LR, temperature) | `beta_` 0.9655 vs 0.9615: the weights reach the classifier's fits and the calibrator. An estimator without `sample_weight` given weights only **warns** ("sample weights will only be used for the calibration itself"), hence no keyword under `"per_child"`; KNN and OvR without routing fit then |
| A grouped training fold lacking a cohort (one cohort in one sample) | `decision_function` estimators (LR, LGBM, HGB) raise "Only 2 class/es in training fold, but 3 in overall dataset", every method; `predict_proba`-only ones (RF, KNN) fit with a `RuntimeWarning` ("Number of classes in training fold (2) does not match…"), the cohort's out-of-fold column 0, rows still summing to 1. On the simulator's training sets (seeds 0–9, grouped 80/20) every cohort is in 174+ of 179–206 samples; 0 of 150 training folds (k = 2, 5, 10) lack one |

## 7. Target code shape

```python
# modeling/count_model.py  (sub-task 1: the rename; sub-task 2: the exposure branch)
class Regressor(Protocol):        # as today
    def fit(self, X, y, sample_weight=None) -> Self: ...
    def predict(self, X) -> ArrayLike: ...

class ExposureRegressor(Protocol):  # takes the raw exposure itself, e.g. NegativeBinomialRegressor
    def fit(self, X, y, exposure=None) -> Self: ...
    def predict(self, X, exposure=None) -> ArrayLike: ...

class CountModel(BaseAgeGroupModel):
    """One regressor for one count column, a cohort's or the total; the exposure as a weighted rate or passed to the estimator."""
    def __init__(self, *, estimator: Regressor | ExposureRegressor, use_exposure: bool = False,
                 feature_transformer=None) -> None: ...
    def fit(self, X, y, exposure=None) -> Self:
        # exposure_values = self._check_exposure(...); feature_transformer, X = self._fit_features(X, y)
        # estimator = clone(self.estimator)
        # if exposure_values is None: estimator.fit(X, y)
        # elif has_fit_parameter(estimator, "exposure"): estimator.fit(X, y, exposure=exposure_values)
        # else: estimator.fit(X, y / exposure_values, sample_weight=exposure_values)
    def predict(self, X, exposure=None) -> np.ndarray:
        # the same test on estimator_: predict(X, exposure=e), or predict(X) * e, or predict(X)

# modeling/negative_binomial.py  (sub-task 2)
class NegativeBinomialRegressor(RegressorMixin, BaseEstimator):
    """NB2 regression, log μ = log(exposure) + b + Xβ, Var = μ(1 + αμ), α by maximum likelihood; statsmodels inside."""
    def __init__(self, *, max_iter: int = 500) -> None: ...
    def fit(self, X, y, exposure=None) -> Self:
        # X = validate_data(self, X, y, reset=True); NegativeBinomial(y, add_constant(X), exposure=exposure)
        #   .fit(method="bfgs", maxiter=self.max_iter, disp=0); raise RuntimeError unless mle_retvals["converged"]
        # intercept_, coef_, dispersion_
    def predict(self, X, exposure=None) -> np.ndarray:
        # validate_data(reset=False); exp(intercept_ + X @ coef_) * (exposure or 1)

# modeling/cohort_probability.py  (sub-tasks 3–4; as built, see the code for the checks)
type ReplicationType = Literal["per_child", "weighted"]
type CalibrationMethod = Literal["temperature", "sigmoid", "isotonic"]

class Classifier(Protocol):
    classes_: np.ndarray
    def fit(self, X, y, sample_weight=None) -> Self: ...
    def predict_proba(self, X) -> ArrayLike: ...

class CohortProbabilityModel(BaseAgeGroupModel):
    """Each cohort's probability for a building: a classifier on one weighted row per (building, cohort)."""
    def __init__(self, *, estimator: Classifier, replication: ReplicationType = "weighted",
                 calibration_method: CalibrationMethod | None = None, calibration_cv: int = 5,
                 feature_transformer=None) -> None: ...
    @staticmethod
    def multinomial_to_categorical(y: pd.DataFrame, replication: ReplicationType) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Sample positions, cohort positions (the labels) and counts of the categorical rows."""
    def fit(self, X, y: pd.DataFrame, exposure=None) -> Self:
        # check_consistent_length(X, y); unobserved cohort -> ValueError; feature_transformer, X = self._fit_features(X, y)
        # sample_positions, cohort_positions, weights = self.multinomial_to_categorical(y, self.replication)
        # X_categorical = take_rows(X, sample_positions); estimator = clone(self.estimator)
        # if self.calibration_method is not None:
        #     sample_folds = np.arange(len(y)) % self.calibration_cv   # samples first, round-robin
        #     estimator = CalibratedClassifierCV(estimator, method=..., ensemble=False,
        #                                        cv=PredefinedSplit(sample_folds[sample_positions]))
        # "weighted": estimator.fit(X_categorical, cohort_positions, sample_weight=weights)
        # "per_child": estimator.fit(X_categorical, cohort_positions)
        # estimator_, cohorts_ (list(y.columns)), feature_transformer_
    def predict(self, X, exposure=None) -> pd.DataFrame:
        # probabilities = estimator_.predict_proba(self._transform_features(X)); rows must sum to 1
        # DataFrame(probabilities, columns=self.cohorts_, index=X.index)  (classes_ are 0..K−1)

# modeling/total_times_probability.py  (sub-task 5)
class TotalTimesProbabilityModel(BaseAgeGroupModel):
    def __init__(self, *, total_model: BaseAgeGroupModel, probability_model: CohortProbabilityModel) -> None: ...
    def fit(self, X, y: pd.DataFrame, exposure=None) -> Self:
        # clone(total_model).fit(X, y.sum(axis=1), exposure=exposure); clone(probability_model).fit(X, y, exposure=exposure)
        # total_model_, probability_model_ (set together, once both succeeded)
    def predict(self, X, exposure=None) -> pd.DataFrame:
        # total = np.asarray(total_model_.predict(X, exposure=exposure), dtype=float)
        # probability_model_.predict(X, exposure=exposure) * total[:, None]   (P12 revision: no column check)
```

Usage (the doc's §0.3 block, to be run):

```python
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from age_group_prediction.modeling import (
    CohortProbabilityModel, CountModel, NegativeBinomialRegressor, TotalTimesProbabilityModel,
)

model = TotalTimesProbabilityModel(
    total_model=CountModel(estimator=NegativeBinomialRegressor(), use_exposure=True, feature_transformer=total_base),
    # or estimator=PoissonRegressor(alpha=1e-3), or LGBMRegressor(objective="poisson"): the same CountModel
    probability_model=CohortProbabilityModel(
        estimator=LogisticRegression(C=1.0, max_iter=1000),  # or LGBMClassifier(...), RandomForestClassifier()
        calibration_method="temperature",
        feature_transformer=cohort_probability_base,
    ),
).fit(X_train, Y_train, exposure=exposure_train)
predictions = model.predict(X_test, exposure=exposure_test)
```

## 8. Sub-tasks

Each ends at a stop (§3). "Verify" lists what the user can check.

### [x] 0. Plan doc, branch, draft PR, baseline
- This plan and its handoff are already in `docs/` (written 2026-10-07, on
  `feat/hyperparameter-tuning`, uncommitted). Add a one-line pointer at the top
  of `docs/MULTI_COHORT_MODELS_PLAN.md` (its §9 B-steps are superseded) and in
  `docs/MODEL_REIMPLEMENTATION_PLAN.md` §5 step 3.
- Branch `feat/total-times-probability-model`; draft PR with the plan as body.
- Baseline `uv run pytest -m "not slow"`; record the count. `grep` for
  `DirectCohortModel` across `src tests docs` to size sub-task 1.
- **Verify:** the plan doc renders; baseline passes; the branch and PR exist.
- **Record (2026-10-07):** branch `feat/total-times-probability-model` created from
  `c966edb`; the draft PR is opened by the user with the commands of the stop.
  Pointers added to `MULTI_COHORT_MODELS_PLAN.md` (top) and
  `MODEL_REIMPLEMENTATION_PLAN.md` §5 step 3. **Baseline:** 1254 passed, 1 skipped,
  1 xfailed (`uv run pytest -m "not slow"`). The handoff's 1230 is PR #12's count
  before its validation suite was committed: pytest has no `testpaths`, so the 24
  tests of `docs/validation/pr12/` are collected too (1230 + 24). Facts re-verified:
  versions as §6; `base.py:76-133`; `EXAMPLES` at line 155; statsmodels only in the
  `validation` group; every file to delete exists; `hyperparameter_tuning/` names no
  Model 2 class; no old-stack module imports `modeling` (`tracking/pyfunc_model.py`
  imports `modeling_config`). Fixed stale facts: §4 (Model 1's class name; the
  evaluator test does build a `DirectCohortModel`), sub-task 1's grep (below).
  **Rename sizing** (the `modeling` class only; the old stack has its own
  `models.DirectCohortModel`, exported from the top-level package and used by
  `experiment/`, `tracking/` and their tests, which stays): src
  `modeling/__init__.py`, `modeling/direct_cohort.py`; tests
  `test_modeling_direct_cohort.py`, `test_modeling_contract.py`,
  `test_modeling_feature_transformer.py`, `test_modeling_independent_cohorts.py`,
  `test_hyperparameter_tuning_evaluator.py`; docs `DIRECT_COHORT_MODEL.md`,
  `FEATURE_TRANSFORMATIONS.md`, `MODULE_REFERENCE.md` (the rebuilt-model rows; its
  `models/` rows are the old stack), `README.md`, `HYPERPARAMETER_TUNING_PLAN.md`,
  `MODEL_REIMPLEMENTATION_PLAN.md` §5 step 2; plus `docs/validation/pr12/` (§10).

### [x] 1. Rename `DirectCohortModel` → `CountModel`
- `modeling/direct_cohort.py` → `modeling/count_model.py`; class, docstring
  ("one count column, a cohort's or the total"), exports, every test and doc
  that names it (`DIRECT_COHORT_MODEL.md` keeps its file name with a note at
  §0; `FEATURE_TRANSFORMATIONS.md` §8.1; `HYPERPARAMETER_TUNING_PLAN.md`;
  `MODULE_REFERENCE.md`; `docs/README.md`; `MODEL_REIMPLEMENTATION_PLAN.md` §5 step 2;
  the memory file). No behavior change.
- **Verify:** `grep -rn DirectCohortModel src/age_group_prediction/modeling tests/unit/test_modeling_*.py tests/unit/test_hyperparameter_tuning_evaluator.py`
  hits nothing; in `docs/*.md` it hits only historical plan records and the old
  stack's `DirectCohortModel` (which keeps its name: `MODELING_GUIDE.md`,
  `CROSS_VALIDATION_AND_SELECTION.md` and the like, and in code
  `src/age_group_prediction/{__init__.py,models,experiment,tracking}` and their
  tests). So `count_model.py` carries no "formerly `DirectCohortModel`" note; that
  belongs in the docs. The suite passes at 1230 passed, 1 skipped, 1 xfailed
  (the baseline 1254 less the 24 removed `docs/validation/pr12/` tests).
- **Record (2026-10-07):** baseline 1254 passed, 1 skipped, 1 xfailed. Renamed
  `modeling/direct_cohort.py` → `count_model.py` (class `CountModel`, docstrings "one
  count column: a cohort's or the total") and `test_modeling_direct_cohort.py` →
  `test_modeling_count_model.py`; the name updated in `modeling/__init__.py` and the
  contract (`_count_model_example`), feature-transformer, independent-cohorts and
  evaluator tests. No logic changed. Docs: `DIRECT_COHORT_MODEL.md` (top note, §0
  heading with a rename note, §0.2, §0.6; its anchor changed, so the links in
  `FEATURE_TRANSFORMATIONS.md` and `MODULE_REFERENCE.md` follow),
  `FEATURE_TRANSFORMATIONS.md`, `MODULE_REFERENCE.md` (rebuilt-model rows),
  `README.md`, `HYPERPARAMETER_TUNING_PLAN.md` §5, `MODEL_REIMPLEMENTATION_PLAN.md`
  §5 step 2; dated records unchanged. **§10's open question, decided** (the user:
  "best practices; in the end stale files and tests are removed"): the behaviors of
  our code that `docs/validation/pr12/` pinned are in `tests/unit/` (constructor and
  nested names, weighted rate = LightGBM offset, Gaussian WLS, predict follows the fit,
  column order, evaluator on raw rows), except G2 (`estimator` has no default), which
  only the signature now enforces; the rest checked library facts (G3b: sklearn's GLM
  equals statsmodels' offset GLM; G3c: sklearn normalizes `sample_weight`, so `alpha`
  acts as `alpha × mean(exposure)`) and PR #12's doc text. So the folder is deleted (kept in `c966edb`, noted at the top of
  both PR #12 validation docs) and `testpaths = ["tests"]` is set, so nothing under
  `docs/` is collected again. Checks: ruff, format, mypy (23 and 14 files) clean;
  the changed tests pass under `-W error` (109); mutation: dropping `CountModel` from
  the contract's `EXAMPLES` fails `test_every_shipped_model_has_an_example` (restored,
  md5 equal).

### [x] 2. `NegativeBinomialRegressor` and `CountModel`'s exposure branch
- `pyproject.toml`: `statsmodels>=0.14.5` into `dependencies` (`uv lock`/`uv sync`
  by the user).
- `modeling/negative_binomial.py` as in §7; `ExposureRegressor` and the
  `has_fit_parameter` branch in `count_model.py`; exports.
- Tests, `test_modeling_negative_binomial.py`: recovers known parameters and α
  on NB2 draws (slow-marked if > 1 s); the exposure enters `predict`
  (`predict(X, exposure=2e) == 2 · predict(X, exposure=e)`); a
  non-converged fit raises (`max_iter=1`); columns in another order raise at
  `predict`; `dispersion_` positive; sklearn's `check_estimator` subset the
  contract test already runs, applied here directly.
  `test_modeling_count_model.py`: with an exposure estimator and `use_exposure`,
  `fit` passes the raw exposure and no `sample_weight`, and `predict`
  passes the same exposure (a spy estimator); a `Regressor` still gets the rate;
  the fitted model follows `estimator_`, not a later `set_params(estimator=…)`;
  NB2 through `CountModel` equals statsmodels with `exposure=` directly
  (the two statsmodels forms, 1e-14). Contract `EXAMPLES` entry for the NB2 variant
  if `CountModel`'s example should stay Poisson (one entry per class: keep
  Poisson, test NB2 in the model's own file).
- Probe to record: statsmodels under `-W error` on the simulated table; the
  convergence check (`mle_retvals["converged"]`) vs its `ConvergenceWarning`.
- **Verify:** mutation checks for each test; mypy clean; a `CountModel` with
  `NegativeBinomialRegressor` fits the simulated totals and its α is reported.
- **Record (2026-10-07):** baseline 1230 passed. Decided with the user after the probes
  (§6, "use best practices"): (1) inside `fit`, statsmodels' `ConvergenceWarning` is ignored and
  non-convergence is raised as our `RuntimeError` (same information, one signal); the
  Hessian, which only gives standard errors (unused, no effect on the result), is not
  computed (`skip_hessian=True`, after the review found its `HessianInversionWarning`
  on converged fits with an all-zero column too: identical parameters, no warning);
  other warnings surface. `predict` raises on an offset whose shape is not `(n,)` (a
  column `(n, 1)` broadcast silently to `(n, n)`; review finding). (2) BFGS only: the preliminary
  Poisson fit too (`optim_kwds_prelim`), so an all-zero dummy (absent from a fold) or a
  constant column fits instead of raising `LinAlgError`. (3) No validation of `y`:
  targets are integer counts, `CountModel` never passes a rate, sklearn's
  `PoissonRegressor` does not check integers either. Also `add_constant(…,
  has_constant="add")` (§10). Files: new `modeling/negative_binomial.py`;
  `count_model.py` (`OffsetRegressor`, `_takes_offset` as `TypeIs` over
  `has_fit_parameter`, the offset branch in `fit` and `predict`, decided by
  `estimator_`); exports; `pyproject.toml` (statsmodels in `dependencies`, the
  `validation` group removed; the user runs `uv lock`); stale guards removed
  (`README.md`, `tests/validation/helpers.py`, `test_recovery.py`,
  `test_total_children.py`). Tests: `test_modeling_negative_binomial.py` (11:
  recovery, offset at predict, offset shape at predict, non-convergence under `-W
  error`, column order, all-zero/all-ones column under an "error" filter, the four
  parameter checks); `test_modeling_count_model.py`
  (offset spy, NB2 = statsmodels' `exposure=` to 1e-5, predict follows `estimator_`,
  NB2 in the doubling and training-mean tests). Recovery on 5,000 draws: b −2.011
  (true −2), β (0.302, −0.193, 0.099), α 0.1014 (0.1). Mutations, all caught: parameter
  layout; offset dropped at predict; convergence check removed; warning filter removed;
  `reset=True` at predict; Newton preliminary fit; `has_constant="skip"`; the rate for an
  offset estimator; the template tested at predict; the offset prediction times the
  exposure; after the review, `skip_hessian` removed, the `ConvergenceWarning` filter
  removed, the shape check removed. Review: no correctness bug; its points fixed (the
  two above, P4's wording, `CountModel`'s docstring summary naming NB2, a blank line). ruff, format, mypy (24 and 12 files) clean; changed tests 68 passed under
  `-W error` (69 after the review); the slow validation files pass. **End to end:** `CountModel(estimator=
  NegativeBinomialRegressor(), use_exposure=True)` on the simulated totals: α 0.1002,
  mean prediction / mean y 1.005, Poisson deviance 3.10. Suite 1247 passed, 1 skipped,
  1 xfailed.
- **Revision (2026-10-07, before the commit; the user):** "the exposure input is the same
  exposure in statsmodels, it is not offset", and "should the input allow both offset and
  exposure?". `NegativeBinomialRegressor.fit/predict` take the raw `exposure` (statsmodels'
  `exposure=`; `predict` = `exposure · exp(b + Xβ)`, its shape checked); `CountModel` detects
  `has_fit_parameter(estimator, "exposure")` (`_takes_exposure`) and passes the raw exposure;
  `OffsetRegressor` → `ExposureRegressor`. **Exposure only** (P4's reasons). Probes: `exposure=e`
  and `offset=log e` give identical parameters; statsmodels accepts both at once (they add up).
  Tests switched to `exposure=` (the spy asserts the raw exposure at fit and predict; the NB
  predict test doubles the exposure). Mutations, all caught: `log(exposure)` passed at
  `CountModel.fit` and at `predict`; the exposure applied twice; the rate for an exposure
  estimator; the template tested at predict; the exposure ignored at NB `predict`; the
  exposure passed to statsmodels as `offset=`; the shape check removed; and the earlier ten
  still valid (layout, convergence, `ConvergenceWarning` filter, `skip_hessian`, Newton,
  `has_constant`, `reset=True`). Suite 1247 passed, 1 skipped, 1 xfailed.
  Review of the revision: no correctness bug (parameters equal statsmodels' `exposure=` fit to
  2.5e-6; `predict` equals its `predict(exposure=)` to 3.4e-6); its points were wording, fixed:
  a docstring line re-wrapped; the exposure's precondition (positive, one per row, validated
  where the data is prepared) stated rather than checked at `fit` (a column exposure already
  raises in statsmodels; a zero one ends in non-convergence); one test comment narrowed to what
  it compares; `ExposureRegressor`'s docstring says it must be the `estimator` itself (inside a
  `Pipeline` it is not detected, and `Pipeline.fit` rejects the `sample_weight`: loud, checked; since revision 2, `CountModel`'s own `TypeError`).
- **Revision 2 (2026-10-07, before the commit; the user):** "Using else does not follow best
  practices. What if I want to add another regressor family... change to elif and throw an
  exception if nothing is satisfied"; "why cast(Regressor, estimator)?"; "Do the same for
  predict... make sure the cases are clear". The cast existed because mypy does not narrow a
  protocol union on a `TypeIs`'s False branch. Now `CountModel._fit_estimator` and
  `_predict_estimator` name each case with one `return` (no exposure; the estimator takes
  `exposure`; it takes `sample_weight`: the weighted rate) and end in `TypeError`; the checks
  are `_takes_exposure` / `_takes_sample_weight`, `TypeGuard`s over `has_fit_parameter` (§6:
  `TypeIs` does not narrow these protocols), so plain `estimator.fit(...)` type-checks, no
  `cast`. **Behavior change:** under `use_exposure=True` an estimator taking neither argument
  (e.g. a `Pipeline`) raises our `TypeError` naming both, not Python's or the `Pipeline`'s own
  error; every estimator used here falls in exactly one case. The old
  `test_a_regressor_without_sample_weight_surfaces_the_library_error` became
  `test_an_estimator_taking_neither_exposure_nor_sample_weight_raises`. Mutations, all caught:
  the final raise removed; the `sample_weight` check replaced by `True` (the old `else`); each
  case dropped at fit and at predict; no multiplication in the rate case; the template at
  predict. Swapping the two cases passes (disjoint, as measured). Suite 1247 passed.
  Review: no correctness bug; one consequence documented in the class docstring: the cases
  are read from the estimator's own `fit` signature, so a wrapper whose `fit` takes
  `**fit_params` (`Pipeline`, `TransformedTargetRegressor`, `VotingRegressor`, `GridSearchCV`)
  is refused under `use_exposure=True` even if it would forward `sample_weight` (measured:
  `TransformedTargetRegressor` used to fit on the rate). None is used as an `estimator` here;
  kept as an explicit refusal (pass the regressor itself), per the user's "no silent case".
- **Revision 3 (2026-10-07, before the commit; the user: "isn't it better to use if-elif-else
  with raise an exception?"):** yes: the objection was to an `else` that silently *handles* a
  case, not to one that only raises. The helpers `_fit_estimator` / `_predict_estimator` (which
  existed only to avoid `else`) are gone: `fit` and `predict` hold the cases inline as
  `if / elif / elif / else: raise TypeError`, every handled case named, the `TypeGuard`s
  narrowing each branch, no `cast`. Behavior unchanged. Mutations re-run, all caught (the
  `raise` replaced by `pass`; the `sample_weight` check by `True`; the exposure case dropped at
  fit and at predict; no multiplication in the rate case; the template at predict). ruff, mypy
  clean; suite 1247 passed, 1 skipped, 1 xfailed; review: no code problem.

### [x] 3. `CohortProbabilityModel` on a classifier (no calibration yet)
- Rewrite `modeling/cohort_probability.py`: `Classifier`, `to_categorical`,
  `fit`, `predict` (P5–P8, P13); delete `predict_logits`.
- Tests (`test_modeling_cohort_probability.py`, rewritten): `to_categorical`
  drops zero cells and keeps counts as weights; the weighted fit equals a fit on one row per child
  (`LogisticRegression`, unpenalized: the P6 identity, pinned so a change of
  the representation is noticed); `predict` columns follow `y`'s order although
  `classes_` are sorted (cohort names chosen so the two orders differ); rows
  sum to 1 for `LogisticRegression`, HGB, `LGBMClassifier`,
  `RandomForestClassifier` and `OneVsRestClassifier`; an unobserved cohort
  raises naming it; a building with no children is fitted around and
  predicted; `exposure` ignored; the feature transformer is fitted on the
  buildings, not the replicated rows (a standardizing transformer: compare its
  fitted mean with the buildings'); refit equals fresh fit; contract `EXAMPLES`.
- **Verify:** `COHORT_LOG_LOSS` of the new model on the simulated split vs the
  torch Dirichlet build on the same split: the torch number measured before the
  first edit (the rewrite removes the torch class; record below).
- **Record (2026-10-07):** baseline 1247 passed, 1 skipped, 1 xfailed. **Order (the user,
  handoff §5's open question):** the rewrite changes the constructor (`estimator` required;
  `solver`, `l2_penalty`, `max_iter`, `tol` gone), so every user of the torch class broke, not
  only `predict_logits`'s; options shown (delete the dependents now; keep the torch class alive
  under another name until sub-task 5; merge sub-tasks 3 and 5). Chosen: delete the dependents
  now: `modeling/independent_total_probability.py`, `modeling/calibration.py`, their unit
  tests, the `predict_logits` test of `test_modeling_feature_transformer.py`, the two names in
  `modeling/__init__.py` and the contract `EXAMPLES`. `total_children.py` and `optimization.py`
  (which never use the class) stay until sub-task 5. The torch baseline was measured first
  (§6: 1.0819; the new model 1.0783 on the same splits, per seed identical to the prototype).
  **Counts are not validated in the model (the user):** data values are validated where the
  data is prepared (`preprocessing.py`, as `ExposureTransformer`); not in
  `TotalTimesProbabilityModel` either, since the probability model is also fitted and tuned
  alone and Model 1's cohorts take the same `y`. The docstring states the precondition
  (non-negative, finite); measured why it matters (§6: a negative weight fits silently in LR,
  HGB, LGBM). Follow-up in §10. **Column order:** `validate_data` kept at fit and predict, as in
  the torch build: `LGBMClassifier` accepts reordered columns silently (§6). The same gap in
  `CountModel` is a §10 follow-up. `y` must be a DataFrame (P1): the torch build's numbered
  cohorts from an array are dropped. Files: `cohort_probability.py` rewritten (`Classifier`,
  `to_categorical`, `fit`, `predict`); `Classifier` exported; the contract example on
  `LogisticRegression()` (buildings without children allowed now); the feature-transformer
  entry on `LogisticRegression()` and its fixture's `+ 1` child per building removed (only the
  torch model needed it). Tests (`test_modeling_cohort_probability.py`, 19 after the review):
  `to_categorical`'s cells; weighted = per child (2.8e-16); names like `y`, index like `X`;
  five classifier kinds mapped by name (OvR with metadata routing); unobserved cohort; a building
  without children; exposure ignored; the transformer fitted on the buildings; reordered columns
  with `LGBMClassifier`; a failed refit (a NaN column) keeps the previous fit; the template stays
  unfitted; beats the marginal shares. ruff, format, mypy (22 and 11 files) clean; changed tests
  44 passed under `-W error`. Mutations (scratchpad `mutate.py`, md5-restored), all 11 caught:
  labels out of step with positions; `sample_weight` dropped; weight = count/total; columns by
  position; the transformer fitted on the replicated rows; `validate_data` dropped at predict;
  the unobserved check dropped; state set before the classifier's fit; `clone` dropped; zero
  cells kept; `X`'s index dropped.
  **Review** (independent subagent; each finding reproduced before the fix): (1) a bug: the
  feature names were recorded (`validate_data`, which can raise: column names of mixed types,
  which LightGBM fits) after the state was set, so a failed refit left the new classifier beside
  the old names; now recorded right after the classifier's fit, before the state.
  (2) A bug: a `y` shorter than `X` fitted silently on `X`'s first rows (the rows are taken from
  `y`'s cells); `check_consistent_length(X, y)` added. (3) A test gap: a transformer assigned
  before the classifier's fit went unseen; a test with a classifier failing at `fit` added.
  (4) My mistake: removing the `predict_logits` test also removed the generic
  `test_predict_follows_the_transformer_fitted_at_fit` after it (a truncation); restored.
  (5) Docstring: the count precondition no longer reads as if validated somewhere. New tests: a
  refit rejected for its feature names; the transformer kept after a failed refit; `X` and `y`
  of different lengths. Mutations re-run, all 14 caught (the 11 above, plus: the length check
  dropped; the names recorded after the state; the transformer assigned before the classifier's
  fit). ruff, format, mypy clean; changed tests 50 passed under `-W error`. **Suite 1214 passed,
  1 skipped, 1 xfailed** (1247 less the deleted torch tests, plus the new ones).

- **Revision 1 (2026-10-07, before the commit; the user):** "check if the estimator has a native
  categorical loss, or we need to convert a binary classifier into multiclass (one-vs-rest)?";
  "set two categorical representations... check both: replicated (one child per row) and weighted
  replication (one row per age group in a building, with the sample weight)"; "`to_categorical` is
  misleading: `multinomial_to_categorical`". Probes (§6): no detection is possible (the tag says
  `multi_class=True` everywhere) or needed (only threshold meta-estimators and LightGBM's binary
  objectives cannot fit 3 classes, and they raise); but LightGBM's `"multiclassova"` returns rows
  not summing to 1, silently. Decided (the user): **`representation: Representation =
  "weighted"`** (`Literal["weighted", "replicated"]`, exported); **no one-vs-rest wrap** (the caller
  wraps); **`predict` raises if a row does not sum to 1** (atol 1e-6; the user asked whether both
  boosting libraries have a valid softmax objective: yes, LightGBM `"multiclass"`, CatBoost
  `"MultiClass"`, their defaults above 2 classes). Code: `multinomial_to_categorical(y,
  representation)` with named cases (else raises); `fit` names both cases and passes no keyword
  under `"replicated"`, so classifiers without `sample_weight` fit; the row-sum check before the
  mapping. Tests (+8 functions, 3 parametrized over both representations; 50 → 63 changed tests):
  the replicated rows; an unknown representation; the two representations fit the same LR; KNN and
  OvR without routing under `"replicated"`; `"multiclassova"` raises. Mutations, all 19 caught: the
  14 earlier ones (re-pointed at the new code) and the replicated rows not repeated,
  `sample_weight` passed under `"replicated"`, the unknown representation not raised, the two cases
  swapped, the row-sum check removed. ruff, format, mypy (22 and 11 files) clean.
  Review: no correctness bug; two wordings fixed: the replicated rows need a numpy signed integer
  dtype (`np.repeat` also rejects `uint64` and pandas' nullable `Int64`, loudly), and the row-sum
  error shows the row sums' range, so NaN probabilities are not mistaken for a one-vs-all
  objective. Suite **1227 passed, 1 skipped, 1 xfailed**.

- **Revision 2 (2026-10-07, before the commit; the user's review of the code, seven questions):**
  (1) "do all models have `sample_weight`?": no (§6: KNN, LDA, QDA, Gaussian process, nearest
  centroid, label propagation, `OneVsRestClassifier` without routing; `sample_weight=None` fails
  for them too), so the weighted case passes the counts and the per-child case no keyword.
  (2) The user renamed the variables in `multinomial_to_categorical`; kept with general names (not
  "building": the granularity may change): `sample_positions, cohort_positions` (scikit-learn's
  "sample"; the repo's "positions"); the comments that named other variables removed. (3) **Renamed**
  (the user): `Representation` → `ReplicationType`, `representation` → `replication`, `"replicated"`
  → `"per_child"`. (4) `predict` uses `X.index` after the transform (the transformer keeps the
  index, measured); the `getattr` served only a numpy `X`. (5) The unobserved-cohort and row-sum
  checks are private static methods (`_check_every_cohort_observed`, `_check_rows_sum_to_one`), as
  `_check_exposure`. (6) "Use best practice; remove checks that are not necessary": `validate_data`
  stays (scikit-learn's API records and checks the feature names; it guards LightGBM without a
  transformer, measured; redundant with one: identical predictions); every other check guards a
  silent or late failure, none removed. (7) `cohorts_` stays: `y`'s order, lost by the sorted
  `classes_`; `predict` selects by it. No behavior change; the unknown-value test now uses the stale
  `"replicated"`. All 19 mutations re-pointed and caught; ruff, format, mypy clean; changed tests 64
  passed under `-W error`. Review: no behavior change; two points fixed: the per-child dtype rule
  (any integer dtype numpy casts safely to `int64`, e.g. `int32`, `uint8`, `bool`; not floats,
  `uint64` or pandas' `Int64`), and the index test now also runs with a feature transformer (the
  index is read after the transform). Suite **1228 passed, 1 skipped, 1 xfailed**.

- **Revision 3 (2026-10-07, before the commit; the user):** "why does `multinomial_to_categorical`
  return labels? we are interested in the label index, not the value"; "what if an estimator with no
  sample weights uses `"weighted"`?"; "the difference between the classes order and `cohorts_`?".
  (1) **Positions as labels** (the user's choice): names are sorted by every classifier into
  `classes_` (`['el', 'hs', 'kg']` for `['kg', 'el', 'hs']`), which `predict` had to map back, and
  mixed-type names fail sklearn's sort (`TypeError`); positions give `classes_ = [0, 1, 2]` (LR,
  HGB, LGBM measured), `y`'s order, so `predict` is `DataFrame(probabilities, columns=cohorts_,
  index=X.index)`. (2) **Left to the library** (the user's choice): KNN, LDA, Gaussian process raise
  `TypeError`, OvR without routing and `Pipeline` raise `ValueError`, `BaggingClassifier` uses the
  weights as sampling probabilities (P(class 0) 0.45 → 0.84); an up-front `has_fit_parameter` check
  would refuse OvR with routing; one docstring line. (3) `classes_` is the classifier's sorted labels,
  `cohorts_` is `y`'s order: with positions they coincide. Tests: expected labels as positions; the
  order tests assert `classes_ == [0, 1, 2]` and compare with `predict_proba` directly; new: mixed-type
  cohort names fit. Mutations, all 19 caught (new: the names as labels again, 17 tests fail; the
  by-position-mapping mutation is gone with the mapping). ruff, format, mypy clean; changed tests 65
  passed under `-W error`. **Review:** `classes_ == 0..K−1` (int64, `y`'s order) holds for LR, HGB,
  LGBM, RF, OvR (with and without routing), KNN, `CalibratedClassifierCV` (`ensemble` False and True),
  K = 2 too, and integer names colliding with positions (`[2, 0, 1]`). Fixed: (a) a classifier that
  keeps its own labels (`FrozenEstimator`, measured: its `['x', 'y', 'z']` silently named `a, b, c`)
  now raises: `_check_classes_are_cohort_positions` after the fit; (b) the observed check works by
  position (`(y.to_numpy() > 0).any(axis=0)`), as duplicate names made the by-name lookup fail with
  pandas' "truth value is ambiguous"; (c) a test with unsorted integer names (`[2, 0, 1]`), the one
  case where a mapping by name would permute silently. Left: MultiIndex cohort names come back as
  tuples (`list(y.columns)`); the per-child dtype rule is the library's. Mutations, all 21 caught
  (+ the classes check removed, the observed check by name); changed tests 68. Suite **1232 passed, 1 skipped, 1 xfailed**.

- **Revision 4 (2026-10-07, before the commit; the user):** "why do we need
  `_check_classes_are_cohort_positions`?", "how can cohort names be duplicated?", "is `validate_data`
  essential?", "is `check_consistent_length` safe to remove?"; the rule: remove what is not essential.
  (1) **Classes check removed:** the classifier is always a clone fitted on the positions, so
  `classes_` are `0..K−1`; the check only caught a `FrozenEstimator`, a misuse P9 already rules out
  (revision 3's "silent misassignment" overstated it). (2) Duplicate names cannot arise here (`y` is
  `table[COHORTS]`); the observed check stays position-based as code, its duplicate comment and test
  removed. (3) **`validate_data` removed** (both calls): it mattered only without a transformer and with
  LightGBM; nothing reads `feature_names_in_`; the gap joins `CountModel`'s in §10. (4)
  **`check_consistent_length` kept:** `CountModel` gets the length check from its regressor; here
  the classifier's rows come from `y`'s cells, so a shorter `y` fits silently (measured). Tests
  removed: frozen classifier, duplicate names, reordered columns, refit rejected for mixed-type
  names; the failed-refit test narrowed (a failing fit with renamed cohorts leaves `predict`
  unchanged). Mutations, all 17 caught (new: the state set before the classifier's fit); ruff,
  format, mypy clean; changed tests 64 passed under `-W error`.

### [x] 4. Calibration
- *From sub-task 3's review:* with `CalibratedClassifierCV(ensemble=False)` and grouped folds, a
  training fold lacking a cohort raises for an estimator with `decision_function` (LR, LGBM):
  "Only 2 class/es in training fold, but 3 in overall dataset" (loud; P10's risk). The groups are
  the sample positions (repeated by count under `"per_child"`); `cv=int` would be stratified, not
  grouped, and leak across samples.
- *Since sub-task 3's revision 1:* under `replication="per_child"` the calibrated
  estimator is fitted without `sample_weight`; the grouped folds still use the row positions.
- `calibration_method`, `calibration_cv`, the `CalibratedClassifierCV` wrap
  with grouped splits (P9–P11). Invalid method names: let
  `CalibratedClassifierCV` raise (it validates `method`), unless it is silent.
- Tests: `None` leaves the estimator's own probabilities (`estimator_` is the
  classifier, not a `CalibratedClassifierCV`); `"temperature"` yields one
  calibrated model (`ensemble=False`); the folds keep a building's rows
  together (inspect the `cv` list: no building on both sides of any split);
  the weights reach the calibrator (dropping `sample_weight` changes the fitted
  temperature); `calibration_cv` controls the number of splits.
- **Verify:** on the simulated split, `COHORT_LOG_LOSS` for `None`,
  `"temperature"`, `"sigmoid"`, `"isotonic"` over seeds 0–9 (the B7 recipe):
  a table in the plan doc; expected: temperature ≈ none ± noise, isotonic worse.
- **Record (2026-10-08):** baseline 1228 passed, 1 skipped, 1 xfailed. Re-verified (§6, sub-task
  4 rows): the signature, `method` validated by the library, one calibrated model, `classes_ ==
  [0, 1, 2]`, rows summing to 1, the weights reaching the calibrator. **Open point (handoff §5),
  decided by the user: documented only.** A grouped training fold lacking a cohort raises for a
  `decision_function` classifier and only warns for a `predict_proba`-only one (zero-filled); it
  never occurred on the simulator (0 of 150 folds), so no check (§10). Code
  (`cohort_probability.py`): `CalibrationMethod` (exported), `calibration_method=None`,
  `calibration_cv=5`; `fit` wraps the clone in `CalibratedClassifierCV(estimator,
  method=calibration_method, cv=PredefinedSplit(sample_folds[sample_positions]),
  ensemble=False)` when a method is set (`sample_folds = arange(len(y)) % calibration_cv`, since
  the review: first `GroupKFold` over the rows), then the existing weighted / per-child cases
  fit it; `predict` unchanged. No check of ours: the method and the fold count
  raise in the library. Tests (+4 functions, 6 cases; the per-child test parametrized over `None`
  and `"temperature"` under an "error" warning filter): the classifier used as is without a
  method; one model on all rows with the given method, `predict` = its probabilities with `y`'s
  columns; the folds keep each sample on one side, `calibration_cv` splits, both replications;
  the weights reach the calibrated classifier (`beta_` equal to a direct weighted fit, unequal
  to an unweighted one). ruff, format, mypy (22 and 9 files) clean; changed tests 73 passed
  under `-W error`. Mutations (scratchpad `mutate.py`, md5-restored), all 8 caught: wrapped
  when the method is None; `ensemble=True`; the method hard-coded; `cv=int`; grouped by cohort;
  the fold count fixed at 5; `sample_weight` dropped in the weighted case; `sample_weight`
  passed under per-child.
  **Table** (held-out `COHORT_LOG_LOSS`; seeds 0–9; the simulator, grouped 80/20 by
  neighborhood, all training buildings, test buildings with a child; 4 standardized features;
  5 grouped folds; mean ± sd over seeds; fitted `beta_` for temperature; fit seconds):

  | Estimator | None | temperature | sigmoid | isotonic |
  |---|---|---|---|---|
  | `LogisticRegression(C=inf)` | 1.0831 ± 0.0098 | 1.0827 ± 0.0090 (`beta_` 0.89) | 1.0829 ± 0.0092 | 1.0849 ± 0.0090 |
  | `LGBMClassifier()` defaults | 1.0888 ± 0.0104 | 1.0876 ± 0.0062 (`beta_` 0.48) | 1.0868 ± 0.0085 | 1.0874 ± 0.0082 |

  (Final folds, samples first; the first run on `GroupKFold` row folds read the same within
  0.0008.) Per seed, method − None: LR temperature −0.0022 to +0.0010 (mean −0.0005), sigmoid
  −0.0028 to +0.0034 (−0.0002), isotonic −0.0019 to +0.0059 (+0.0018); LGBM temperature
  −0.0093 to +0.0066 (−0.0012), sigmoid −0.0118 to +0.0046 (−0.0020), isotonic −0.0110 to
  +0.0039 (−0.0014). Fit time ×1.6 (LR, 0.018 → 0.028 s) and ×4 (LGBM, 0.06 → 0.23 s).
  **Reading:** every method is within noise of `None` (seed sd ≈ 0.01). Temperature is the
  steadiest (narrowest per-seed range for LR). Isotonic is the worst for LR, as expected, but
  not for LGBM. LightGBM's defaults are overconfident (`beta_` 0.48, i.e. T ≈ 2.1), and
  temperature narrows its seed spread (sd 0.0104 → 0.0062), yet its mean gains only 0.001.
  The unpenalized LR is slightly overconfident too (`beta_` 0.89). So on this data the setting
  matters little; the smoke run (sub-task 6) compares None and temperature per variant.
  **Review** (independent subagent; each finding reproduced): no correctness bug in the wiring
  (row alignment, weights in the out-of-fold and final fits, feature names, `get_params` /
  nested `set_params`, failed refit). (1) Under a calibration method, `"weighted"` with a
  classifier lacking `sample_weight` only warns and fits it unweighted: **the user: a note**
  (KNN is to be used under `"per_child"`). (3) Folds of categorical rows differ between the
  replications (beta 0.963 vs 0.970): **the user: split the samples before the categorical
  rows, round-robin** (P10 revision; new test `test_both_replications_calibrate_on_the_same_folds`).
  Wording fixed: (2) routing on + calibration needs `set_fit_request(sample_weight=True)`;
  (4) `"multiclassova"` is renormalized under calibration, so `predict` raises only without a
  method; (5) the per-child test's comment (weights through the calibration warn; `None` is
  silent: the mutation was caught because it passed real weights); (6) a docstring "it".
  Mutations re-run, all 10 caught (the 8 above re-pointed, plus folds of rows ungrouped and the
  previous `GroupKFold` row folds). Changed tests 74 passed under `-W error`; ruff, format,
  mypy (22 and 9 files) clean. **Suite 1238 passed, 1 skipped, 1 xfailed.**

### [x] 5. `TotalTimesProbabilityModel`, deletions, exports
- `modeling/total_times_probability.py` (P12); delete `total_children.py`,
  `optimization.py`, their tests and `tests/validation/test_total_children.py`
  (`independent_total_probability.py` and `calibration.py` went in sub-task 3);
  update `modeling/__init__.py`, the contract test's `EXAMPLES`,
  `test_modeling_feature_transformer.py`;
  `grep -rn "TotalChildrenModel\|TemperatureCalibrator\|IndependentTotalProbabilityModel\|single_threaded_torch\|modeling.optimization" src tests --include='*.py'`
  must hit only the old stack's own `IndependentTotalProbabilityModel` (`models/`, `experiment/`,
  `tracking/`, `modeling_config.py`, the package root `__init__.py`, and their tests, e.g.
  `bundle_spy.py`, `test_experiment.py`, `test_final_*.py`, `test_gate8_tracking.py`).
- Tests (`test_modeling_total_times_probability.py`): `fit` gives the row sum
  to the total model and `y` to the probability model (spies); `predict` =
  total × probabilities, columns in `y`'s order, index of `X`; rows of
  `predict` sum to the total's prediction; ~~a probability model with other
  columns raises~~ (dropped with the check, P12 revision); exposure reaches both; templates stay unfitted (clones);
  nested `set_params` names (`probability_model__calibration_method`); refit
  equals fresh fit; works with each total variant (`CountModel` Poisson with
  and without exposure, `CountModel` with `NegativeBinomialRegressor`).
- Docstrings and comments of the new class follow §3 rule 7 from the start (the older
  ones are trimmed in sub-task 7).
- **Verify:** `uv run pytest -m "not slow"` passes; the `grep` above; mypy
  clean; no `import torch` under `modeling/`.
- **Record (2026-10-08):** baseline 1238 passed, 1 skipped, 1 xfailed. Re-verified: the five
  paths to delete exist; `total_children` / `optimization` are imported only by
  `modeling/__init__.py`, each other, their tests, and the contract and feature-transformer
  tests; `tests/validation/helpers.py` names only the simulator's `TotalChildrenSimulator`
  (unrelated); old-stack tests still import torch, so the single-thread LightGBM notes stay (P14).
  Stale facts fixed: §4's `EXAMPLES` line; the grep's expected hits (above). **P12 revised (the
  user: "compare the two, choose by best practice, do not overcomplicate"):** no column check,
  no `cohorts_` (P12 note). Code: new `modeling/total_times_probability.py`
  (`TotalTimesProbabilityModel`, Model 1's shape: clones, state set together, `check_is_fitted`;
  the total through `np.asarray`, since a total with its own index would realign `mul` into NaN);
  exports (`Solver`, `TotalChildrenModel` out; `TotalTimesProbabilityModel` in); the contract's
  `_total_times_probability_example` (`CountModel` on LightGBM + `CohortProbabilityModel` on LR;
  `_cohort_counts` shared with the probability example); the feature-transformer entry removed
  (the total is a `CountModel`, already there). Deleted (`git rm`): `total_children.py`,
  `optimization.py`, `test_modeling_total_children.py`, `test_modeling_optimization.py`,
  `tests/validation/test_total_children.py`. Tests (`test_modeling_total_times_probability.py`,
  7 functions, 9 cases): row sums to the total and `y` to the probabilities, exposure to both
  (spies); exposure at predict (spies); `predict` = the halves fitted alone, `y`'s unsorted columns,
  `X`'s non-default index; rows sum to the total for Poisson with and without exposure and NB2;
  templates unfitted; a failing half keeps the previous fit; nested `set_params` reach the fit.
  ruff, format, mypy (21 and 10 files) clean; changed tests 37 passed under `-W error`. Mutations
  (scratchpad `mutate.py`, md5-restored), all 11 caught: total from one column; exposure dropped
  at fit (each half) and at predict (each half); the total not broadcast along rows; index
  dropped; `clone` dropped (each half); the total's state set before the probability fit; the
  contract entry removed. **Review** (independent subagent; findings reproduced): no correctness
  bug (row alignment by position with duplicated, string and reordered indexes; NaN counts raise
  in the probability model's fit; contract and nested names). Fixed: the `predict` comment moved to
  the `np.asarray` line (its real "why", reproduced: a Series total gives all NaN); the docstring's
  "(e.g. a `CountModel`)" placed on `total_model`; the dropped column-check test struck above.
  Noted for sub-task 7: `MODULE_REFERENCE.md` and `FEATURE_TRANSFORMATIONS.md` still name
  `TotalChildrenModel`, `Solver`, `optimization.py`. **Suite 1200 passed, 1 skipped, 1 xfailed**
  (1238 − 44 deleted unit tests − 3 feature-transformer cases + 9 new; 2 slow validation tests gone).
- **Revision (2026-10-08, before the commit; the user asked whether `mul(total, axis=0)` is faster
  than `probabilities * total`):** bare `* total` aligns a 1-D array with the **columns** (60 rows:
  `ValueError`; rows == cohorts: silent, rows no longer sum to the total); `mul(total, axis=0)`
  55.5 µs and `* total[:, None]` 30.5 µs on 245 rows, both correct. The user chose
  `probabilities * total[:, None]`.

### [x] 6. Smoke run (as PR #11's B7)
- Script in the scratchpad, recipe in the plan doc: ten simulated populations,
  grouped 80/20 split, class defaults (untuned). Compare Model 1 (PR #12's
  `IndependentCohortModels` with LightGBM) against Model 2 variants:
  total ∈ {`PoissonRegressor` with exposure, `NegativeBinomialRegressor` with
  exposure, `LGBMRegressor(objective="poisson")` with exposure} × probability ∈
  {`LogisticRegression`, `LGBMClassifier`, HGB, `RandomForestClassifier`} ×
  calibration ∈ {None, temperature}. Metrics: per-cohort and total
  Poisson deviance, `COHORT_LOG_LOSS`. Record the table and the fitted NB2 α.
- **Verify:** the table; a one-paragraph reading (what won, by how much, noise).
- **Record (2026-10-08):** HEAD `99d4dd0`, tree clean. Re-verified: the simulator is
  `src/student_simulator` (`PYTHONPATH=src`; the 245-row table's `n_children_total` equals the cohort
  sum); `FEATURE_TRANSFORMATIONS.md` §8.0–8.3's `tree`, `total_base`, `cohort_probability_base`;
  `Splitter("grouped")`, `ExposureTransformer`, `take_rows`, `POISSON_DEVIANCE`, `COHORT_LOG_LOSS`;
  every variant below runs at the class defaults under `-W error` (`LogisticRegression()` converges at
  its default `max_iter`). **Recipe:** per seed 0–9,
  `StudentPopulationSimulator(load_simulation_config("configs/simulation.toml")).run(rng=np.random.default_rng(seed))`;
  §8.0 `ShareTransformer`; `ExposureTransformer("n_apartments")` on the full table;
  `Splitter("grouped").train_test_indices(table, table["neighborhood_id"], test_size=0.2,
  random_state=seed)`, every array by `take_rows`; fit on all training buildings (179–206), score on
  all test buildings (33–53; a building without children adds nothing to the log loss). Baseline:
  the constant rate per cohort `Σy_train / Σexposure_train × exposure_test`, its sum for the total,
  its rows the training marginal shares. Model 1: `IndependentCohortModels` of
  `CountModel(LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1), use_exposure=True,
  feature_transformer=tree)`. Model 2: `TotalTimesProbabilityModel(total_model=CountModel(T,
  use_exposure=True, feature_transformer=total_base), probability_model=CohortProbabilityModel(P,
  calibration_method=M, feature_transformer=cohort_probability_base))`, T ∈ {`PoissonRegressor()`,
  `NegativeBinomialRegressor()`, `LGBMRegressor(objective="poisson", n_jobs=1, verbosity=-1)`}
  (labelled poisson / nb2 / lgbm), P ∈ {`LogisticRegression()`, `LGBMClassifier(n_jobs=1,
  verbosity=-1)`, `HistGradientBoostingClassifier()`, `RandomForestClassifier(n_jobs=1,
  random_state=seed)`}, M ∈ {None, "temperature"}; everything else at the class defaults (untuned,
  as B7 decided). Each composite's `predict` rows were asserted to sum to the total model's
  prediction (§11). Scratchpad `smoke_run.py` (per-session; rebuild from this recipe), ≈ 6 min.

  **Mean ± SD over the 10 populations; lower is better.** The total's deviance depends on T only.

  | Model | `n_kindergarten` | `n_elementary` | `n_highschool` | total | cohort log loss |
  |---|---|---|---|---|---|
  | baseline | 2.739 ± 0.680 | 2.532 ± 0.492 | 2.936 ± 0.909 | 5.277 ± 1.722 | 1.089 ± 0.007 |
  | Model 1 | 2.628 ± 0.877 | 2.128 ± 0.499 | 2.572 ± 0.710 | 4.215 ± 1.450 | 1.093 ± 0.010 |
  | poisson/LR/none | 2.485 ± 0.531 | 2.159 ± 0.453 | 2.324 ± 0.618 | 4.318 ± 1.272 | 1.082 ± 0.010 |
  | poisson/LR/temperature | 2.428 ± 0.543 | 2.160 ± 0.446 | 2.344 ± 0.629 | 4.318 ± 1.272 | 1.082 ± 0.008 |
  | poisson/LGBM/none | 2.825 ± 0.641 | 2.344 ± 0.487 | 2.660 ± 0.548 | 4.318 ± 1.272 | 1.101 ± 0.014 |
  | poisson/LGBM/temperature | 2.512 ± 0.620 | 2.191 ± 0.432 | 2.612 ± 0.681 | 4.318 ± 1.272 | 1.090 ± 0.006 |
  | poisson/HGB/none | 3.013 ± 0.662 | 2.372 ± 0.575 | 2.640 ± 0.567 | 4.318 ± 1.272 | 1.106 ± 0.013 |
  | poisson/HGB/temperature | 2.547 ± 0.633 | 2.186 ± 0.443 | 2.594 ± 0.682 | 4.318 ± 1.272 | 1.091 ± 0.005 |
  | poisson/RF/none | 2.634 ± 0.571 | 2.276 ± 0.488 | 2.524 ± 0.633 | 4.318 ± 1.272 | 1.093 ± 0.011 |
  | poisson/RF/temperature | 2.495 ± 0.605 | 2.205 ± 0.449 | 2.573 ± 0.661 | 4.318 ± 1.272 | 1.089 ± 0.006 |
  | nb2/LR/none | 2.554 ± 0.498 | 2.159 ± 0.568 | 2.285 ± 0.553 | 4.347 ± 1.306 | 1.082 ± 0.010 |
  | nb2/LR/temperature | 2.519 ± 0.506 | 2.148 ± 0.565 | 2.294 ± 0.559 | 4.347 ± 1.306 | 1.082 ± 0.008 |
  | nb2/LGBM/none | 2.828 ± 0.495 | 2.374 ± 0.543 | 2.657 ± 0.655 | 4.347 ± 1.306 | 1.101 ± 0.014 |
  | nb2/LGBM/temperature | 2.608 ± 0.535 | 2.168 ± 0.523 | 2.567 ± 0.718 | 4.347 ± 1.306 | 1.090 ± 0.006 |
  | nb2/HGB/none | 3.025 ± 0.656 | 2.438 ± 0.594 | 2.591 ± 0.669 | 4.347 ± 1.306 | 1.106 ± 0.013 |
  | nb2/HGB/temperature | 2.654 ± 0.608 | 2.176 ± 0.524 | 2.526 ± 0.710 | 4.347 ± 1.306 | 1.091 ± 0.005 |
  | nb2/RF/none | 2.735 ± 0.451 | 2.289 ± 0.621 | 2.438 ± 0.607 | 4.347 ± 1.306 | 1.093 ± 0.011 |
  | nb2/RF/temperature | 2.627 ± 0.535 | 2.192 ± 0.591 | 2.483 ± 0.656 | 4.347 ± 1.306 | 1.089 ± 0.006 |
  | lgbm/LR/none | 2.614 ± 0.679 | 2.173 ± 0.594 | 2.375 ± 0.609 | 4.512 ± 1.603 | 1.082 ± 0.010 |
  | lgbm/LR/temperature | 2.592 ± 0.687 | 2.166 ± 0.594 | 2.368 ± 0.623 | 4.512 ± 1.603 | 1.082 ± 0.008 |
  | lgbm/LGBM/none | 2.887 ± 0.825 | 2.363 ± 0.599 | 2.774 ± 0.593 | 4.512 ± 1.603 | 1.101 ± 0.014 |
  | lgbm/LGBM/temperature | 2.698 ± 0.768 | 2.180 ± 0.575 | 2.631 ± 0.694 | 4.512 ± 1.603 | 1.090 ± 0.006 |
  | lgbm/HGB/none | 3.076 ± 0.885 | 2.404 ± 0.657 | 2.739 ± 0.618 | 4.512 ± 1.603 | 1.106 ± 0.013 |
  | lgbm/HGB/temperature | 2.744 ± 0.769 | 2.178 ± 0.582 | 2.598 ± 0.693 | 4.512 ± 1.603 | 1.091 ± 0.005 |
  | lgbm/RF/none | 2.823 ± 0.741 | 2.289 ± 0.651 | 2.515 ± 0.642 | 4.512 ± 1.603 | 1.093 ± 0.011 |
  | lgbm/RF/temperature | 2.734 ± 0.746 | 2.203 ± 0.623 | 2.530 ± 0.692 | 4.512 ± 1.603 | 1.089 ± 0.006 |

  **Model 2 minus Model 1, paired by population** (mean ± SD; t = mean / (SD/√10); populations
  where Model 2 is lower), the Poisson total; the nb2 and lgbm rows read the same on the cohorts
  and the log loss (the probability half is independent of T).

  | Variant | `n_kindergarten` | `n_elementary` | `n_highschool` | cohort log loss |
  |---|---|---|---|---|
  | poisson/LR/none | −0.142 ± 0.421 (t −1.1; 7/10) | +0.031 ± 0.245 (t +0.4; 4/10) | −0.249 ± 0.398 (t −2.0; 7/10) | −0.010 ± 0.006 (t −5.4; 9/10) |
  | poisson/LR/temperature | −0.199 ± 0.402 (t −1.6; 8/10) | +0.032 ± 0.243 (t +0.4; 4/10) | −0.228 ± 0.408 (t −1.8; 6/10) | −0.011 ± 0.006 (t −5.8; 9/10) |
  | poisson/LGBM/none | +0.198 ± 0.439 (t +1.4; 3/10) | +0.216 ± 0.331 (t +2.1; 2/10) | +0.088 ± 0.319 (t +0.9; 5/10) | +0.009 ± 0.008 (t +3.3; 1/10) |
  | poisson/LGBM/temperature | −0.116 ± 0.392 (t −0.9; 6/10) | +0.063 ± 0.256 (t +0.8; 3/10) | +0.040 ± 0.474 (t +0.3; 4/10) | −0.002 ± 0.007 (t −1.0; 6/10) |
  | poisson/HGB/none | +0.385 ± 0.325 (t +3.8; 1/10) | +0.245 ± 0.377 (t +2.1; 3/10) | +0.068 ± 0.259 (t +0.8; 3/10) | +0.013 ± 0.007 (t +6.2; 0/10) |
  | poisson/HGB/temperature | −0.081 ± 0.416 (t −0.6; 6/10) | +0.058 ± 0.246 (t +0.7; 5/10) | +0.022 ± 0.463 (t +0.2; 5/10) | −0.002 ± 0.008 (t −0.8; 7/10) |
  | poisson/RF/none | +0.006 ± 0.478 (t +0.0; 5/10) | +0.148 ± 0.203 (t +2.3; 3/10) | −0.048 ± 0.369 (t −0.4; 5/10) | −0.000 ± 0.006 (t −0.0; 6/10) |
  | poisson/RF/temperature | −0.132 ± 0.435 (t −1.0; 7/10) | +0.077 ± 0.193 (t +1.3; 4/10) | +0.001 ± 0.460 (t +0.0; 4/10) | −0.003 ± 0.006 (t −1.8; 7/10) |

  Total minus Model 1's: poisson +0.103 ± 0.600 (t +0.5; 5/10), nb2 +0.132 ± 1.066 (t +0.4; 4/10),
  lgbm +0.297 ± 0.470 (t +2.0; 3/10). Total by population (baseline / Model 1 / poisson / nb2 /
  lgbm): seed 1 4.69 / 3.58 / 3.60 / 6.05 / 4.13; seed 5 9.71 / 7.31 / 7.29 / 5.54 / 7.81; seed 2
  4.60 / 4.86 / 4.45 / 5.36 / 5.85; seed 7 4.22 / 5.04 / 4.15 / 4.83 / 4.52.

  **Temperature minus None, paired** (the Poisson total; the other totals within 0.001 of these on
  the log loss): LR −0.0007 ± 0.0014 (t −1.5; 7/10), LGBM −0.0109 ± 0.0091 (t −3.8; 9/10), HGB
  −0.0148 ± 0.0098 (t −4.8; 9/10), RF −0.0032 ± 0.0053 (t −1.9; 8/10); on kindergarten's deviance
  LR −0.057 (t −2.3), LGBM −0.314 (t −3.0), HGB −0.466 (t −5.8), RF −0.139 (t −2.3); on elementary's
  LGBM −0.153 (t −3.1), HGB −0.186 (t −2.5), RF −0.071 (t −2.5), LR 0; on high school's none (|t| ≤
  0.6). **Against the marginal shares** (log loss, Poisson total): LR −0.0067 ± 0.0058 (t −3.6; 9/10)
  raw and −0.0074 ± 0.0050 (t −4.6; 10/10) calibrated; LGBM +0.0123 (t +3.4; 0/10) raw, +0.0014
  (t +0.6) calibrated; HGB +0.0165 (t +4.6; 0/10) raw, +0.0016 (t +0.8) calibrated; RF +0.0036
  (t +1.5) raw, +0.0003 (t +0.2) calibrated. **Fitted:** NB2 α 0.103 ± 0.014 (per seed 0.081–0.121);
  `beta_` (inverse temperature) LR 0.82 ± 0.05, RF 0.55 ± 0.11, LGBM 0.43 ± 0.10, HGB 0.39 ± 0.09.
  Fit seconds per composite, none / temperature: LR 0.03 / 0.05, LGBM 0.08 / 0.26, RF 0.12 / 0.53,
  HGB 1.4 / 6.9.

  **Reading.** (1) *The recipe is B7's:* Model 1 and the baseline reproduce it to the third
  decimal (2.628 / 2.128 / 2.572, total 4.215 ± 1.450; 2.739 / 2.532 / 2.936, 5.277 ± 1.722, log
  loss 1.089 ± 0.007), and so does NB2 (α 0.103 ± 0.014 as the torch build's; total 4.347 ± 1.306
  against 4.348 ± 1.306): statsmodels' NB2 through `CountModel` is the torch NB2. (2) *On the total
  Model 2 and Model 1 are level* for the two GLM totals (t +0.5 and +0.4, lower in 4–5 of 10); the
  LightGBM total is worse (t +2.0, lower in 3 of 10: at the defaults, on `total_base`'s nine
  scaled columns (six plans), it is a weaker total than Model 1's three LightGBMs on `tree`, which keep
  `n_apartments` as a feature). sklearn's `PoissonRegressor()` default `alpha=1` is a penalized fit,
  unlike B7's unpenalized torch Poisson: the seed-1 overfit B7 saw (6.90) is gone (3.60), the mean
  moved from 4.448 to 4.318, and its SD from 1.434 to 1.272. NB2 is unpenalized and keeps the
  seed-1 outlier (6.05). (3) *On the composition `LogisticRegression()` wins, by a small amount:*
  its cohort log loss is below Model 1's by 0.010–0.011 (t −5.4 raw, −5.8 calibrated, 9 of 10) and
  below the marginal shares by 0.007 (t −3.6 raw, −4.6 calibrated, 9–10 of 10); 1.082 against the
  torch Dirichlet's 1.086 raw and 1.083 calibrated. It also lowers kindergarten's and high school's
  deviance (t −1.1 to −2.0; −2.0 to −2.4 under NB2) and is level on elementary's. The gain stays
  under 1 % per child: the shares are close to unpredictable from these features. (4) *The tree
  classifiers at their defaults are overconfident:* `beta_` 0.39–0.55 (T ≈ 1.8–2.6; LR 0.82), and
  raw they are worse than the marginal shares (LGBM +0.012, HGB +0.017, t > 3, better in 0 of 10)
  and than Model 1 (t +3.3 and +6.2), with kindergarten's deviance up by 0.2–0.4. (5) *Temperature
  calibration repairs most of that* (LGBM −0.011, HGB −0.015, RF −0.003 on the log loss, t −1.9 to
  −4.8, lower in 8–9 of 10; kindergarten −0.14 to −0.47) and narrows the seed spread (SD 0.013–0.014
  → 0.005–0.006), bringing the trees back to the marginal shares (within +0.002, |t| < 1) and to
  Model 1 (−0.002, |t| ≤ 1), not beyond. For LR it is within noise (−0.0007, t −1.5), as the
  sub-task 4 table found. The calibrated LR and the calibrated trees differ by 0.007–0.009 in LR's
  favour. (6) *Nothing here calls for a code change:* every variant fits and predicts as the class
  contracts say, the rows sum to the total, and the losses are the untuned defaults' (tuning is the
  next PR's job: the trees' depth and learning rate, LR's `C`, the GLM penalties). For the docs
  (sub-task 7): the default-overconfidence of the tree classifiers and what temperature does to it;
  that sklearn's Poisson default is penalized; the LightGBM total's feature base.

### [ ] 7. Docs and close (7a done; 7b next)
- `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 rewritten for the new build
  (§9 lists what it must explain); `docs/DIRECT_COHORT_MODEL.md` §0 (the exposure
  rule, NB2); `docs/FEATURE_TRANSFORMATIONS.md` §8.1–8.3 usage blocks;
  `docs/MODULE_REFERENCE.md`; `docs/README.md`; `docs/MODEL_REIMPLEMENTATION_PLAN.md`
  §5; ~~`docs/HYPERPARAMETER_TUNING_PLAN.md` where it names Model 2's tunables~~ (re-verified in
  7a: it names none; L405–413 is a historical note; no edit);
  `docs/MULTI_COHORT_MODELS_PLAN.md` top note. Every code block run.
- **Derivations (the user, 2026-10-08):** each §9 item is written as a derivation: the model
  and its assumptions, each step, then the result the code relies on, with the formulas. Each
  one names the code it justifies (class, method, setting) and ends with its measured check
  (§6 or the §8 records).
- **Trimming pass (the user, 2026-10-08):** every docstring and comment written in this PR is
  cut to §3 rule 7: `count_model.py`, `negative_binomial.py`, `cohort_probability.py`,
  `total_times_probability.py` and their tests' comments. Whatever is cut and still matters
  moves to the docs. The main case is `CohortProbabilityModel`'s class docstring, about 45
  lines today: replication, dtype rules, routing, calibration and the fold gap. No behavior
  change: the suite and mypy pass unchanged.
- Memory: update `multi-cohort-models-plan.md` (the state), note the decisions.
- PR body drafted in the scratchpad; the user applies it and marks the PR ready.
- **Record, 7a (2026-10-08):** HEAD `c9002b1`. The user split sub-task 7 into two stops: **7a** the
  docs, **7b** the trimming pass, the PR body and the close. Re-verified by a survey of `docs/`
  (line numbers at `c9002b1`): the Model 2 doc's §0 (L11–238) was the torch build throughout, with
  stale notes at L3–9, L343–347, L535–539; `DIRECT_COHORT_MODEL.md` §0 lacked NB2 and said "NB2
  removed" (L238), its L28 anchor was broken; `FEATURE_TRANSFORMATIONS.md` L15, L740, the §8.2–8.3
  blocks and the §8.7 penalty note; `MODULE_REFERENCE.md` L87–113; `README.md` L16–20, L42–43 and
  no row for this plan; `MODEL_REIMPLEMENTATION_PLAN.md` L9–11, L562–565; the §0 anchor of the
  Model 2 doc is linked from three docs (fixed with the heading). Figures re-run 2026-10-08:
  Poisson offset vs rate 1.5e-12; NB2 offset vs rate 0.012 (another draw than §6's 0.021);
  `exposure=` vs `offset=log e` 0.0; `NegativeBinomialRegressor` vs statsmodels 0.0; weights scaled
  by a constant 1.4e-17; weighted vs per-child LR 6.1e-16. **Written:**
  `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 rewritten as §0.1–§0.13 (the model; the total as
  `CountModel`, with the Poisson identity, the NB2 non-identity derived from the log-likelihood,
  statsmodels' exposure, the `has_fit_parameter` cases, `alpha × mean(exposure)`; the NB2
  representation with the alternatives table; data replication with the likelihood identity,
  count vs count/total, the two replications and their table, positions as labels, the edge cases;
  bagging and bootstrap, P8 in full with `E[m_i] = 1` and the five-variant table; calibration:
  cross-fitting, folds of buildings split before the rows, five folds, the three methods, the
  sub-task 4 table and the smoke run's `beta_`, the documented-only cases; the classifiers survey
  and the row-sum rule; the API, data flow (the §7 usage block, run on seed 0: 1.855 / 1.958 /
  2.250, total 3.587, composition 1.084, α 0.0876), rules and errors tables; what changed; the
  smoke-run evidence), heading and the three links renamed to
  `#0-the-rebuilt-model-modelingtotal_times_probabilitypy`; the two later notes updated.
  `DIRECT_COHORT_MODEL.md` §0: the exposure case as §0.1 (F) (the NB2 derivation, statsmodels'
  exposure, the signature test, wrappers refused), the API rows (`ExposureRegressor`,
  `NegativeBinomialRegressor`; the `n_jobs=1` rationale), the errors (`TypeError` neither;
  `RuntimeError` NB2), §0.4 (NB2 back), §0.5 (the NB2 evidence), the L28 anchor; §0.1's heading kept
  (four links). `FEATURE_TRANSFORMATIONS.md`: row B, L740, the §8.2 / §8.3 / combining blocks on
  the new classes with a prose note on the two exposure cases, the §8.7 penalty note (the
  estimators' own penalties; `C` per child; the `l2_penalty` figures as history).
  `MODULE_REFERENCE.md`: the modeling intro and table (seven current modules), the link, the date.
  `README.md`: the in-progress bullet, both model rows, an Active Plans row for this plan.
  `MODEL_REIMPLEMENTATION_PLAN.md`: the status note and §5 step 3 (done, PR #13).
  `MULTI_COHORT_MODELS_PLAN.md` L9: "done 2026-10-08, PR #13". **Checks:** every rewritten code
  block executed as written, pulled from the files, under `-W error` (scratchpad `run_blocks.py`:
  the §8.0–8.3 blocks, the combining block, the §0.9 block, the direct model's §0.2 and §0.6
  blocks); the rows of every combined prediction equal the total model's; the stale-name grep over
  the five reference docs hits only history sentences and old-stack rows; every markdown anchor in
  `docs/` resolves (the sub-task 1 anchor script; the reports left are source-line links and two
  older plan docs' relative paths). No `.py` change: no suite run. **Review** (independent
  subagent; every finding reproduced): no error in the derivations' mathematics or in the traced
  figures (it re-traced every number to §6 / §8 and re-ran the §0.9 block); fixed: (1) §0.2 (e)'s
  intermediate sentence compared the weighted objective with an *unnormalized* offset objective
  (that gives `alpha × ΣE`); it is scikit-learn's (1/n)-normalized one that gives `alpha × mean(E)`,
  the stated conclusion; (2) two cited test names; (3) §0.6 (d)'s "~1,300 rows, ~450 per class"
  was §6's synthetic set, now the simulator's (~4,200 children in ~570 cells per training set,
  1,200–1,600 per cohort: at isotonic's threshold, not far below it); (4) P11's "~245 buildings"
  (the whole table) noted stale, the doc cites the smoke run's 179–206; (5) "the tree classifiers
  raw worse than the marginal shares (t > 3)" overgeneralized to RF (t +1.5): LightGBM and HGB
  named; (6) `total_base` has nine columns from six plans, not "six standardized features" (here
  too); (7) the sub-task 4 table and the smoke run are on different features and test sets: said.
  Kept by design: the NB2 non-identity and the signature-test paragraph appear in both model docs
  (§9: the total model's part in both), each pointing at the other.
- **Revision 1 of 7a (2026-10-08, before the commit; the user: "create a doc for each model,
  describing each component along the way (formula, derivation, motivation)"; "one doc per model;
  since the count model is shared, you do not need to explain it twice"; the Model 2 doc "up to
  400–500 lines"; "the handoff should be removed once completed"):** the 7a §0 content moved into
  two new docs, compacted and restructured by component (what, formula, derivation, why, code,
  check): `docs/INDEPENDENT_COHORT_MODELS.md` (Model 1: the model; `CountModel` with the weighted
  rate (A)–(E) in short, the penalty under normalized weights, the exposure passed to the
  estimator with the NB2 non-identity, `NegativeBinomialRegressor` with its fit choices and the
  alternatives; `IndependentCohortModels` with its rules as text; API, data flow, errors,
  evidence) and `docs/TOTAL_TIMES_PROBABILITY_MODEL.md` (Model 2: the model; the total in one
  paragraph pointing at the Model 1 doc; `CohortProbabilityModel` by component: the categorical
  rows, the classifier, bagging under replication, calibration; `TotalTimesProbabilityModel`; API,
  data flow, errors, what changed, evidence). `DIRECT_COHORT_MODEL.md` §0 and
  `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 are pointers (their §1+ untouched); every link into
  the old §0 anchors re-pointed (`FEATURE_TRANSFORMATIONS.md`, `MODEL_REIMPLEMENTATION_PLAN.md`,
  `MODULE_REFERENCE.md`, `README.md`, which also gets rows for the two docs). The handoff is
  deleted in 7b (its pointer at the top of this plan with it). **Checks:** both docs' code blocks
  executed as written under `-W error` (the Model 1 block's seed-0 scores 1.903 / 2.011 / 2.404,
  the Model 2 block's as before); every markdown anchor resolves; the old names appear only in
  history sentences; no link into the old §0 anchors remains. **Review** (independent subagent;
  every finding reproduced): the derivations and every figure correct after compaction; nothing
  explained twice; fixed: two torch-era rules still valid (rows by position; targets from `y`
  only) restored in the Model 2 doc §6; the Model 1 doc gained its "what changed" list (§7), §3's
  check, the Gaussian "can be negative" caveat and a `Code` label; "reproducible trees" (an
  unmeasured rationale) dropped; "`y` is not validated" → "not checked to be integer"; link
  texts in three docs still read "DIRECT_COHORT_MODEL.md §0.1"; the cuts it suggested to bring
  the Model 2 doc under 500 lines (the §3.4 restatement of §9, the data-flow steps 1–2 and the
  three `CountModel` error rows as links to the Model 1 doc, the classifier survey and the
  library list condensed, a repeated routing sentence). `count_model.py`'s docstring citation of
  `DIRECT_COHORT_MODEL.md` §0.1 goes to 7b with the trimming.
- **Revision 2 of 7a (2026-10-08; the user: "you didn't separate between Poisson and Gaussian (I
  think NB2 is OK)"):** `INDEPENDENT_COHORT_MODELS.md` §2.1 rewritten as (a) the Poisson loss (the
  model, (A)–(C), its own check) and (b) the Gaussian loss ((D1)–(D4), its own check, the full-ML
  figure restored), then (c) why not a residual, the Why and the Code; §2.2 (NB2) unchanged. No
  figure or heading changed.
- **Verify:** `grep -rn "TotalChildrenModel\|TemperatureCalibrator\|Dirichlet\|DirectCohortModel" docs --include='*.md'`
  hits only historical records (plan docs' step records, the old stack's §1–§12). Every §9
  item has its derivation. A review subagent checks the docstrings and comments against §3
  rule 7, and that nothing was lost: each cut fact is in the docs.

## 9. Docs: what each decision's explanation must say

In `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 (the model doc), one subsection
each, **derived step by step** (assumptions, steps, result, the code it
justifies), with the formulas and the measured figures of §6; the total model's
part also in `DIRECT_COHORT_MODEL.md` §0. *7a revision 1 (the user, 2026-10-08): one doc
per rebuilt model instead, `docs/INDEPENDENT_COHORT_MODELS.md` (Model 1, where `CountModel`
and NB2 are explained once) and `docs/TOTAL_TIMES_PROBABILITY_MODEL.md` (Model 2), each
component with its formula, derivation and motivation; the two §0s are pointers.*
Derivations expected at least:

- the weighted rate equals the Poisson offset model, but not NB2;
- statsmodels' `exposure` equals `offset = log(exposure)` with coefficient 1;
- `alpha` acts as `alpha × mean(exposure)` under normalized `sample_weight`;
- the multinomial likelihood equals the categorical rows' likelihood (weight = count, not
  `count / total`);
- `E[m_i] = 1` for every resampling unit;
- cross-fitting (`ensemble=False`) vs the ensemble, and the (k−1)/k bias;
- temperature vs sigmoid vs isotonic, and their parameter counts;
- `Σ_k Ĉ_{b,k} = μ̂_b`.

The sections:

1. **The model.** `log μ_b = log A_b + b + x_bᵀβ` (Poisson, NB2, or any
   regressor's mean × exposure); `p_{b,k} = classifier(w_b)`;
   `Ĉ_{b,k} = μ̂_b p̂_{b,k}`, `Σ_k Ĉ_{b,k} = μ̂_b`.
2. **The total model is `CountModel`.** Why one class serves a cohort and the
   total; the two exposure forms: the weighted rate (the Poisson offset model
   exactly; for a Gaussian or tree loss it is "mean = exposure × f(x)", not an
   offset) and the exposure passed as is (statsmodels' `exposure`: the offset
   `log(exposure)` with coefficient 1, taken inside, for any estimator whose
   `fit` takes `exposure`, detected with `has_fit_parameter`); why NB2 needs it
   (the 0.021 difference; non-integer `y` accepted silently); why exposure only,
   no `offset` (P4); `alpha × mean(exposure)`.
3. **NB2 representation.** NB2 = `Var = μ(1 + αμ)`, α fitted jointly by
   maximum likelihood (statsmodels' `loglike_method="nb2"`); what
   `predict(exposure=)` computes and that it equals `offset=log(exposure)`; the
   alternatives table (statsmodels discrete NB, statsmodels GLM with fixed α
   and L2, torch, glum, LightGBM/XGBoost custom objective, scikit-learn: none)
   and why the first.
4. **Data replication.** Multinomial → categorical: `Mult(Y_b, p_b)` ∝
   `Π_k p_{b,k}^{C_{b,k}}`, so one weighted row per positive cell with weight
   `C_{b,k}` is the same log-likelihood as one row per child; the 1.8e-15
   check; why the weight is the count and not `count / total`; the two
   replications (`replication="weighted"` / `"per_child"`, P6): equal for a
   weighted-likelihood estimator, different procedures for trees (rows counted by
   `min_samples_leaf`, the bootstrap), the revision-1 table, the cost (rows ×
   children per building), the integer counts the replicated rows need, and the
   classifiers without `sample_weight` that only `"per_child"` admits; where the
   conversion (`multinomial_to_categorical`) is done and why
   (the transformer on buildings, grouped folds, the contract); a building
   without children; an unobserved cohort.
5. **Bagging and bootstrap under replication.** How a building's cohorts are
   connected (shared features, the total, proportions summing to one) and why
   only the first two concern the rows (the sum to one is the classifier's
   output); the exchangeable unit (the child, given the building); what a row
   resampler draws under weighted rows (cells) and per-child rows (children),
   and that only a bootstrap by building keeps a building whole; what LightGBM
   (subsampling without replacement, off by default), HGB (none),
   `LogisticRegression` (none), RF (bootstrap with replacement, on by
   default) and XGBoost (all rows by default, `subsample=1.0`; a note, since
   it is not a dependency) do; P8's measured five-variant table (all equal
   within noise) with what each bag draws; the unbiasedness argument
   (`E[m_i] = 1` for every unit, so no variant misrepresents the
   composition, and a bag by building drops buildings too); why observed
   building features do not change the case and an unobserved building
   effect or a need for intervals does; the two options then
   (`bootstrap=False`, a `GroupedBaggingClassifier` taking `groups`, passed
   by `has_fit_parameter` like the NB2 exposure); what to tune.
6. **Calibration.** `CalibratedClassifierCV`: cross-fitting (`ensemble=False`)
   vs the ensemble; why folds are grouped by building (and the neighborhood
   limitation); 5 folds (the (k−1)/k argument, the 50/50 bias); temperature
   (1 parameter, softmax) vs sigmoid (OvR, 2 per class, renormalized) vs
   isotonic (non-parametric; sklearn's ≥ 1,000-samples guidance; overfits
   here); every method returns rows summing to 1; the measured table from
   sub-task 4; `C` and the weights are per child.
7. **Classifiers.** The `Classifier` protocol; multinomial vs OvR; `classes_`
   mapping; `OneVsRestClassifier` and `sample_weight` through metadata routing
   (none needed under `"per_child"`). The survey of §6: which classifiers fit a
   softmax loss (LR, HGB, `GradientBoosting`, MLP, LightGBM `"multiclass"`,
   CatBoost `"MultiClass"`, XGBoost `multi:softprob`), which are multiclass by
   construction (trees, naive Bayes, neighbors, LDA/QDA), which reduce to binary
   internally (`SGDClassifier`, `GaussianProcessClassifier` OvR; `SVC` OvO), which
   are binary-only (the threshold meta-estimators; LightGBM `"binary"`); why the
   model does not wrap (the tag cannot tell); and the row-sum rule (`predict`
   raises; LightGBM `"multiclassova"`).
8. **API, data flow, rules, errors** tables as the present §0.2–§0.5, updated.

## 10. Risks

- **statsmodels under `-W error`:** resolved in sub-task 2. A converged fit is clean;
  a non-converged one warns before our check, so `fit` ignores the
  `ConvergenceWarning` and raises, skips the Hessian, and uses BFGS for the
  preliminary fit (§8 record).
- **`add_constant` skips a constant column** (`has_constant="skip"`): corrected in
  sub-task 2. It is not harmless: no intercept is added, the parameters shift by
  one, and `params[0]` would be read as the intercept. `has_constant="add"` is used;
  with the BFGS preliminary fit a constant column then fits (its share of the
  intercept is arbitrary, the predictions are not).
- **`has_fit_parameter` on a `Pipeline` or meta-estimator** inspects the outer
  `fit`, so an exposure estimator inside a `Pipeline` is not detected: documented
  (the feature transformer belongs to `CountModel`, so no `Pipeline` is needed).
- **`OneVsRestClassifier` + `sample_weight`** needs metadata routing; if the
  `Classifier` protocol cannot express it cleanly, document OvR as "enable
  routing" rather than special-case it.
- **`CalibratedClassifierCV` with a training fold lacking a cohort** (re-measured in sub-task
  4, §6): a classifier with `decision_function` (LR, LGBM, HGB) raises; one with
  `predict_proba` alone (RF, KNN) only warns, and the calibrator sees that cohort at 0 out of
  fold. It needs a cohort seen in a few samples only; on the simulator every cohort is in
  174+ of ~180–206 training samples and no fold lacked one. **Decided (the user, sub-task 4):
  documented only** (the class docstring, the model doc's calibration section), no check.
- **The rename touches many files:** done first and alone (sub-task 1) so the
  behavior changes afterwards are reviewable.
- **Tuning package:** `hyperparameter_tuning` names nothing of Model 2 in code
  (PR #12's evaluator takes any `BaseAgeGroupModel`); only its plan doc does.
  Re-verify with `grep` in sub-task 0.
- **Old stack imports:** `models/count_regression.py` has its own NB2; nothing
  in `experiment/` or `tracking/` imports `modeling`. Re-verify in sub-task 5.

- **The PR #12 validation suite is collected by the non-slow run** (found in
  sub-task 0): `docs/validation/pr12/test_decisions.py` and `test_doc_claims.py`
  (24 tests) import `DirectCohortModel` from `modeling`, and 11 lines name
  `TotalChildrenModel`, the torch `CohortProbabilityModel` and
  `IndependentTotalProbabilityModel`, which sub-tasks 3 and 5 rewrite or delete.
  Sub-tasks 1 and 5 break them unless decided first. The two uncollected scripts
  `probe_end_to_end.py` and `probe_tuning_plan_s5.py` also name `DirectCohortModel`. **Decided in sub-task 1:** the folder is deleted and `testpaths = ["tests"]`
  set (sub-task 1 record).

- **Sub-task 3 removes `CohortProbabilityModel.predict_logits`** (found at the end of the
  first implementing session). **Resolved in sub-task 3:** its dependents were deleted there,
  after the torch baseline was measured (sub-task 3 record).

- **Follow-up: a counts validator in `preprocessing.py`** (found in sub-task 3). Negative or
  NaN counts become negative or NaN weights in `CohortProbabilityModel`, which most classifiers
  fit silently (§6). By the repo's rule the values are validated where the data is prepared,
  like `ExposureTransformer`, once on the table before splitting; it serves every model. Not
  in this PR's sub-tasks; the user decides when.
- **Follow-up: `CountModel` and `CohortProbabilityModel` accept reordered columns silently** with
  an estimator that does not check feature names (LightGBM; `CountModel` measured: 2.41
  difference), but only **without a feature transformer**: the transformer selects columns by name,
  and this repo's flow always uses one. `CohortProbabilityModel` checked it with `validate_data`
  until sub-task 3's revision 4, which removed it as not essential (the user). If the design-matrix
  path ever matters, one check in `BaseAgeGroupModel._fit_features` / `_transform_features` closes it
  for every model.

## 11. Verification

- Per sub-task: the changed tests with `-W error`, the mutation checks, the
  review subagent, `uv run pytest -m "not slow"`, `uv run mypy`.
- End to end: the §7 usage block in the doc runs on the simulator's table
  (`StudentPopulationSimulator(load_simulation_config("configs/simulation.toml")).run(rng=np.random.default_rng(0))`,
  `ShareTransformer`, `ExposureTransformer`, `Splitter("grouped").train_test_indices`,
  `take_rows`), predicts a DataFrame whose rows sum to the total model's
  prediction, and scores with `POISSON_DEVIANCE` and `COHORT_LOG_LOSS`.
- The smoke-run table (sub-task 6) against PR #11's B7 numbers: Model 2 at or
  near Model 1 on deviance, better on composition.
- `grep` for the deleted and renamed names and for `torch` under
  `src/age_group_prediction/modeling`.
- The docs' derivations (§9) and the docstring and comment trimming pass (§3 rule 7, sub-task 7).
