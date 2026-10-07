# Plan: rebuild Model 2 as `TotalTimesProbabilityModel` (scikit-learn classifiers, statsmodels NB2)

**Written for:** the implementing model (Claude Opus 5.5) and the user who
validates each sub-task. Self-contained: it assumes no memory of the planning
conversation. This file is the source of truth: update its status line and the
sub-task checkboxes in §8 as work finishes. Orientation for a new session:
[TOTAL_TIMES_PROBABILITY_MODEL_HANDOFF.md](TOTAL_TIMES_PROBABILITY_MODEL_HANDOFF.md).

**Status (2026-10-07):** approved by the user. **Sub-task 0 is done** (branch
`feat/total-times-probability-model`, baseline 1254 passed, 1 skipped, 1 xfailed;
record in §8); sub-task 1 is next, after the open question in §10 (the PR #12
validation tests). Nothing in `src` is implemented yet. During planning the user revised it twice: NB2 goes through
`CountModel` (the renamed `DirectCohortModel`) as an offset estimator; a
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
7. **Code style:** match `modeling/direct_cohort.py`. Docstrings keep what the
   code does and its non-obvious constraint; derivations and motivation go in
   the docs with a pointer at most. Inner-library quirks go in the docs.
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
| `modeling/negative_binomial.py` | — | **new:** `NegativeBinomialRegressor`, a scikit-learn regressor around statsmodels NB2 with `fit(X, y, offset=None)` and `predict(X, offset=None)` |
| `modeling/cohort_probability.py` | torch Dirichlet regression, `predict_logits` | **rewritten:** a classifier on the categorical representation (one weighted row per building and cohort), optional `CalibratedClassifierCV` |
| `modeling/calibration.py` | `TemperatureCalibrator` | **deleted** (`CalibratedClassifierCV(method="temperature")` replaces it) |
| `modeling/optimization.py` | `Minimizer`, `Solver`, `single_threaded_torch` | **deleted** (only the two torch models use it; the old stack's `models/count_regression.py` has its own) |
| `modeling/independent_total_probability.py` | `IndependentTotalProbabilityModel(total_children_model, cohort_probability_model, temperature_calibrator)` | **renamed** `modeling/total_times_probability.py`, `TotalTimesProbabilityModel(total_model, probability_model)` |
| `modeling/__init__.py` | exports the above | `CountModel`, `Regressor`, `OffsetRegressor`, `NegativeBinomialRegressor`, `Classifier`, `CalibrationMethod`, `CohortProbabilityModel`, `TotalTimesProbabilityModel`, Model 1's names; gone: `DirectCohortModel`, `Solver`, `TemperatureCalibrator`, `TotalChildrenModel`, `IndependentTotalProbabilityModel` |
| `pyproject.toml` | `statsmodels` only in the `validation` group | `statsmodels>=0.14.5` in `dependencies` |
| `tests/unit/test_modeling_total_children.py`, `test_modeling_calibration.py`, `test_modeling_optimization.py`, `tests/validation/test_total_children.py` | tests of the deleted code | **deleted** |
| `tests/unit/test_modeling_direct_cohort.py` | `DirectCohortModel` | renamed `test_modeling_count_model.py`; the offset branch added |
| `tests/unit/test_modeling_cohort_probability.py`, `test_modeling_independent_total_probability.py` | tests of the torch builds | **rewritten** (`test_modeling_total_times_probability.py`) |
| `tests/unit/test_modeling_negative_binomial.py` | — | **new** |
| `tests/unit/test_modeling_contract.py` | `EXAMPLES` per concrete model (line 155) | entries renamed/added; a model without an example fails `test_every_shipped_model_has_an_example` |
| `tests/unit/test_modeling_feature_transformer.py`, `test_modeling_independent_cohorts.py`, `tests/unit/test_hyperparameter_tuning_evaluator.py` (builds a `DirectCohortModel`: lines 40, 296, 435) | use the old names | updated |
| `docs/DIRECT_COHORT_MODEL.md` §0 | `DirectCohortModel` | file name kept; §0 says the class is `CountModel`, used for a cohort and for the total, with the offset rule |
| `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 | the torch build | §0 rewritten (the file keeps §1–§12 for the old stack until roadmap step 5 deletes it) |
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
| P3 | **`CountModel` takes two kinds of estimator and tells them apart with sklearn's `has_fit_parameter(estimator, "offset")`.** An estimator whose `fit` has an `offset` parameter (`OffsetRegressor` protocol: `fit(X, y, offset=None)`, `predict(X, offset=None)`) is fitted with `offset=log(exposure)` and predicts with the same offset; any other (`Regressor`, as today) gets the weighted rate. Without `use_exposure` both are fitted plainly. The fitted copy decides at `predict` (the same test on `estimator_`) | The user: "DirectCohortModel should be able to use NB2 as well: identify the NB2 model and treat it differently". **Why not the rate for NB2:** the weighted-rate form is exact for Poisson only; for NB2 it fits a different model (measured: coefficients differ by 0.021), and statsmodels' NB accepts the non-integer `y / exposure` silently. **Why a signature test, not `isinstance`:** it is sklearn's own idiom for `sample_weight` (`has_fit_parameter`), it names the capability rather than one class, so any future offset estimator (a statsmodels GLM wrapper, say) gets the same treatment, and it is measured to say `False` for `PoissonRegressor` and `LGBMRegressor`. The offset is `log(exposure)`, the GLM offset with coefficient 1, so `predict` needs no multiplication |
| P4 | **NB2 is `NegativeBinomialRegressor`** in `modeling/negative_binomial.py`: a scikit-learn regressor (`BaseEstimator`, `RegressorMixin`) around statsmodels' `discrete_model.NegativeBinomial(loglike_method="nb2", offset=…)`, `fit(method="bfgs", maxiter=max_iter, disp=0)`; fitted `intercept_`, `coef_`, `dispersion_` (α), `n_features_in_`/`feature_names_in_` (`validate_data`); `predict(X, offset=None)` = `exp(offset + b + Xβ)` (statsmodels' own `predict(offset=)`; measured equal to its exposure form to 1e-14). Unpenalized; `max_iter=500` is its one setting. A fit that does not converge raises. statsmodels moves to the main dependencies | The user asked for a scikit-learn wrapper of statsmodels' NB2 with an optional exposure offset and no torch. As a plain regressor it plugs into `CountModel` (P3), so Model 1's cohorts and Model 2's total get NB2 the same way. **"Can I predict using the exposure component?"** Yes: `results.predict(exog, offset=log e)` (equivalently `exposure=e`) returns `e · exp(b + Xβ)`. **Alternatives** (table in §9): statsmodels `GLM(family=NegativeBinomial(alpha))` (α must be fixed; L2 via `fit_regularized(L1_wt=0)`: a penalized variant for later if the smoke run needs one), the current torch objective (dropped: no torch), `glum` (not installed; θ fixed), LightGBM/XGBoost (no NB objective; a custom objective would re-create the torch code), scikit-learn (none: `TweedieRegressor` is not NB) |
| P5 | **The categorical representation is built inside `CohortProbabilityModel.fit`,** by a static method `CohortProbabilityModel.to_categorical(y) -> (positions, labels, weights)`: one row per `(building, cohort)` with a positive count, `labels` the cohort name, `weights` the count (`np.nonzero(counts)`). `X` is the design matrix's rows at `positions`. The feature transformer is fitted on the **original** rows first | The user: inside only if both the raw and the categorical targets are needed there. They are: (a) the feature transformer must see one row per building, as in every other model (a replicated fit would weight the training statistics by children); (b) the calibration folds must keep a building's rows together (P10), which needs the original row ids; (c) `fit(X, y: counts)` keeps the base contract, so the probability model is fitted, scored (`COHORT_LOG_LOSS` on counts) and tuned on its own, and `TotalTimesProbabilityModel` only passes `y` on |
| P6 | **Weighted replication only, weight = count (per child).** No `replication` setting; not one row per child; never `count / total` | The user's choice of weighting, and the evidence against a setting. For an estimator that fits a weighted likelihood the two replications are **equivalent:** a `sample_weight` of `c` multiplies that row's log-likelihood term by `c`, exactly what `c` identical rows contribute; measured on `LogisticRegression`: coefficient difference 1.8e-15 (442 weighted rows vs 1,325 per-child rows). For an estimator that **resamples rows** (RF) they are not identical fits, but the held-out quality is the same within noise (P8's table: 0.712 vs 0.708 against a seed spread of 0.57–0.86), so a setting would serve no measured purpose; it is a one-argument extension of `to_categorical` if real data ever shows one. The weighted rows are the multinomial likelihood of the counts given the total, the quantity `COHORT_LOG_LOSS` scores; `count / total` (one unit per building, the Dirichlet build's weighting) is a different estimator (coefficients move by 0.06) and not that likelihood |
| P7 | **`estimator: Classifier`**, a `Protocol`: `fit(X, y, sample_weight=None)`, `predict_proba(X)`, `classes_`. Multinomial classifiers (`LogisticRegression`, whose lbfgs is multinomial in sklearn 1.9, `multi_class` is gone; `HistGradientBoostingClassifier`; `LGBMClassifier`, `objective_ = "multiclass"` when ≥ 3 labels; `RandomForestClassifier`) and `OneVsRestClassifier(binary)` alike. `predict` maps `predict_proba`'s columns **by `classes_`**, never by position, into `y`'s column order at fit | `classes_` are sorted labels (`['el', 'hs', 'kg']` for `['kg', 'el', 'hs']`): a positional mapping would permute cohorts silently. **Rows sum to 1** for all of them: `OneVsRestClassifier.predict_proba` normalizes in the multiclass case (measured 2e-16), as do HGB, LightGBM, RF and `CalibratedClassifierCV` (sigmoid and isotonic are per-class, then normalized; temperature is a softmax). The model asserts nothing about it: it is what the libraries do, and `COHORT_LOG_LOSS` renormalizes anyway. `OneVsRestClassifier.fit` takes `sample_weight` only through metadata routing (`set_fit_request(sample_weight=True)` on the inner estimator, `sklearn.set_config(enable_metadata_routing=True)`); documented, not special-cased |
| P8 | **Bagging and bootstrap under replication: nothing is built; the estimators' own resampling is left as the user sets it.** The docs explain the units and give the measured table | **The user's question:** the cohorts of a building are connected (they share `x_b`, sum to `Y_b`, and their proportions sum to one), so should a resample keep the building's rows together? **The answer in three parts.** (1) The sum-to-one constraint is on the model's output `p_b`, which every classifier's `predict_proba` (and `CalibratedClassifierCV`) enforces by softmax or normalization; it is not a dependence between rows. (2) Under the conditional multinomial model `C_b \| Y_b ~ Mult(Y_b, p_b)` is `Y_b` independent categorical draws, so the exchangeable unit is the **child**; the weighted representation merely compresses identical child rows into one cell per cohort. A row resampler then draws **cells** (a bag can hold building b's kindergarten cell and drop its elementary cell) instead of children (a bag thins each building's composition at random); neither keeps a building whole; only a bootstrap **by building** does, which scikit-learn and LightGBM do not offer. (3) **Measured** (RF, 300 trees, 5 seeds, held-out cross-entropy against the true `p`): weighted + bootstrap 0.712, per-child + bootstrap 0.708, weighted without bootstrap 0.716, per-child without bootstrap 0.717, a hand-made bootstrap by building 0.717, against a seed-to-seed spread of 0.57–0.86: **all the same within noise**. So no representation switch and no grouped bagging is justified. **What the libraries do** (the user's "permutation" is not it): LightGBM's bagging is subsampling **without replacement** (`subsample`, active only with `subsample_freq > 0`; off by default); `HistGradientBoostingClassifier` has **no** row subsampling and its early stopping is off under 10,000 rows; `LogisticRegression` has none; `RandomForestClassifier` bootstraps **with replacement**, on by default (with `sample_weight`, sklearn multiplies the weight by the draw count); XGBoost (not a dependency; from its parameter docs) uses **all rows** by default, `subsample=1.0`, and subsamples without replacement per round only when set below 1. So RF is the only common estimator that resamples by default, and the user chose to leave it as is. **Observed building features** (type, year) change nothing: every row of a building shares its whole `x_b` already, and conditioning on more of it makes the conditional independence of its children more plausible, not less; the grouped unit becomes the right one only for an **unobserved** building effect (a random building intercept in the generator, or a generated characteristic withheld from the model) or for uncertainty intervals. **If real data shows extra-multinomial variation between buildings** (the case where the building is the right unit): `bootstrap=False` on RF removes the resampling (randomness then comes from `max_features`), or a bootstrap-by-building aggregator over `CohortProbabilityModel` (the ten-line probe above), added only then. The folds that must be grouped are the calibration folds (by building, P10) and the tuner's (by neighborhood, `Splitter`) |
| P9 | **Calibration is `CalibratedClassifierCV` inside `CohortProbabilityModel`:** `calibration_method: CalibrationMethod \| None = None` (`Literal["temperature", "sigmoid", "isotonic"]`, `None` = the estimator's own probabilities), `calibration_cv: int = 5`. `fit` wraps the cloned estimator as `CalibratedClassifierCV(estimator, method=…, cv=<grouped splits>, ensemble=False)` and fits it with the weights. **No `TemperatureCalibrator`** | The user's specification. `ensemble=False` is **cross-fitting**: the estimator is fitted on each of `k` folds, its out-of-fold probabilities for **every** row are collected, **one** calibrator is fitted on them, then the estimator is refitted on all rows; `predict` is that one model through that one map. `ensemble=True` (sklearn's default for a non-frozen estimator) averages `k` calibrated fold models and never fits on all rows. Cross-fitting is the user's preference and what the previous build did by hand (the B6 loop). `TemperatureCalibrator` duplicated sklearn's `_TemperatureScaling`, whose objective it was pinned to; the calibration now sits in the model, so clones and tuning (`probability_model__calibration_method`) carry it, and no `FrozenEstimator` idiom is needed |
| P10 | **Calibration folds: `GroupKFold(n_splits=calibration_cv)` over the replicated rows with `groups=positions`** (the building), unshuffled, passed as a list of splits | A building's rows carry its known composition; with plain `KFold` the same building sits in a fit fold and its calibration fold, and the calibrator sees in-sample confidence. Grouping by building removes that; grouping by **neighborhood** (the repo's evaluation unit) would need `groups` at `fit`, which the base contract does not carry: documented as the limitation (the tuner's outer folds are by neighborhood regardless). Unshuffled `GroupKFold` is deterministic, so no `random_state` setting |
| P11 | **Folds: 5. Isotonic is allowed but documented as inappropriate here** | **How many:** the calibrator must map the **final** model's probabilities, but it is fitted on fold models trained on `(k−1)/k` of the rows, which are less confident than the final one; `k = 2` (50/50) calibrates a model fitted on half the data and biases the temperature toward sharpening; larger `k` approaches the final model at the cost of `k` fits; with ~245 buildings per training set, 5 (sklearn's default) leaves ~200 buildings per fold fit and uses every row for calibration. **How much is enough:** temperature fits 1 parameter and sigmoid 2 per class, so every row of a 5-fold cross-fit (~1,300 child-weighted rows) is ample; isotonic is non-parametric per class and sklearn advises it only well above ~1,000 samples per class, so it overfits here (the previous plan's N12 said the same). Measured: all three run on the replicated data; temperature fits `T ≈ 1.00` on well-specified simulated data |
| P12 | **`TotalTimesProbabilityModel(total_model, probability_model)`** in `modeling/total_times_probability.py`: `fit` clones and fits `total_model` on `y.sum(axis=1)` and `probability_model` on `y`, both with `exposure`; `predict` = `total[:, None] × probabilities`, a DataFrame with `y`'s columns at fit, indexed like `X`; raises if the probability model's columns differ from `cohorts_`. No calibrator argument | The user's name (chosen over `TotalSplitModel`, `TotalCompositionModel`): it names the prediction. Settings named for their role (`total_model`, `probability_model`); nested names reach both (`total_model__estimator__alpha`, `probability_model__estimator__C`, `probability_model__calibration_method`) |
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
| `CalibratedClassifierCV(…, cv=list(GroupKFold(5, shuffle=True).split(X_rep, labels, groups=building)), ensemble=False).fit(X_rep, labels, sample_weight=weights)` | runs for temperature, sigmoid and isotonic; one calibrated model (`len(calibrated_classifiers_) == 1`); rows sum to 1; `classes_ == ['el', 'hs', 'kg']`; fitted `T = 1.00006` |
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

## 7. Target code shape

```python
# modeling/count_model.py  (sub-task 1: the rename; sub-task 2: the offset branch)
class Regressor(Protocol):        # as today
    def fit(self, X, y, sample_weight=None) -> Self: ...
    def predict(self, X) -> ArrayLike: ...

class OffsetRegressor(Protocol):  # an estimator with a GLM offset, e.g. NegativeBinomialRegressor
    def fit(self, X, y, offset=None) -> Self: ...
    def predict(self, X, offset=None) -> ArrayLike: ...

class CountModel(BaseAgeGroupModel):
    """One regressor for one count column, a cohort's or the total; the exposure as a weighted rate or an offset."""
    def __init__(self, *, estimator: Regressor | OffsetRegressor, use_exposure: bool = False,
                 feature_transformer=None) -> None: ...
    def fit(self, X, y, exposure=None) -> Self:
        # exposure_values = self._check_exposure(...); feature_transformer, X = self._fit_features(X, y)
        # estimator = clone(self.estimator)
        # if exposure_values is None: estimator.fit(X, y)
        # elif has_fit_parameter(estimator, "offset"): estimator.fit(X, y, offset=np.log(exposure_values))
        # else: estimator.fit(X, y / exposure_values, sample_weight=exposure_values)
    def predict(self, X, exposure=None) -> np.ndarray:
        # the same test on estimator_: predict(X, offset=log e), or predict(X) * e, or predict(X)

# modeling/negative_binomial.py  (sub-task 2)
class NegativeBinomialRegressor(RegressorMixin, BaseEstimator):
    """NB2 regression, log μ = offset + b + Xβ, Var = μ(1 + αμ), α by maximum likelihood; statsmodels inside."""
    def __init__(self, *, max_iter: int = 500) -> None: ...
    def fit(self, X, y, offset=None) -> Self:
        # X = validate_data(self, X, y, reset=True); NegativeBinomial(y, add_constant(X), offset=offset)
        #   .fit(method="bfgs", maxiter=self.max_iter, disp=0); raise RuntimeError unless mle_retvals["converged"]
        # intercept_, coef_, dispersion_
    def predict(self, X, offset=None) -> np.ndarray:
        # validate_data(reset=False); exp((offset or 0) + intercept_ + X @ coef_)

# modeling/cohort_probability.py  (sub-tasks 3–4)
type CalibrationMethod = Literal["temperature", "sigmoid", "isotonic"]

class Classifier(Protocol):
    classes_: np.ndarray
    def fit(self, X, y, sample_weight=None) -> Self: ...
    def predict_proba(self, X) -> ArrayLike: ...

class CohortProbabilityModel(BaseAgeGroupModel):
    """Each cohort's probability for a building: a classifier on one weighted row per (building, cohort)."""
    def __init__(self, *, estimator: Classifier, calibration_method: CalibrationMethod | None = None,
                 calibration_cv: int = 5, feature_transformer=None) -> None: ...
    @staticmethod
    def to_categorical(y: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row positions, cohort labels and counts of every positive (building, cohort) cell."""
    def fit(self, X, y: pd.DataFrame, exposure=None) -> Self:
        # unobserved cohort -> ValueError; feature_transformer, X = self._fit_features(X, y)
        # positions, labels, weights = self.to_categorical(y); X_rep = X.iloc[positions]
        # estimator = clone(self.estimator)
        # if self.calibration_method is not None:
        #     splits = list(GroupKFold(self.calibration_cv).split(X_rep, labels, groups=positions))
        #     estimator = CalibratedClassifierCV(estimator, method=..., cv=splits, ensemble=False)
        # estimator.fit(X_rep, labels, sample_weight=weights)
        # estimator_, cohorts_ (list(y.columns)), feature_transformer_
    def predict(self, X, exposure=None) -> pd.DataFrame:
        # probabilities = estimator_.predict_proba(self._transform_features(X));
        # DataFrame(probabilities, columns=estimator_.classes_)[self.cohorts_], index=X.index

# modeling/total_times_probability.py  (sub-task 5)
class TotalTimesProbabilityModel(BaseAgeGroupModel):
    def __init__(self, *, total_model: BaseAgeGroupModel, probability_model: BaseAgeGroupModel) -> None: ...
    def fit(self, X, y: pd.DataFrame, exposure=None) -> Self:
        # clone(total_model).fit(X, y.sum(axis=1), exposure=exposure); clone(probability_model).fit(X, y, exposure=exposure)
        # total_model_, probability_model_, cohorts_
    def predict(self, X, exposure=None) -> pd.DataFrame:
        # probabilities = probability_model_.predict(X, exposure=exposure); columns must equal cohorts_
        # total = np.asarray(total_model_.predict(X, exposure=exposure)); total[:, None] * probabilities
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

### [ ] 1. Rename `DirectCohortModel` → `CountModel`
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
  belongs in the docs. The suite passes at the baseline count (1254, or 1230 if the
  user excludes `docs/validation/pr12/`, §10).

### [ ] 2. `NegativeBinomialRegressor` and `CountModel`'s offset branch
- `pyproject.toml`: `statsmodels>=0.14.5` into `dependencies` (`uv lock`/`uv sync`
  by the user).
- `modeling/negative_binomial.py` as in §7; `OffsetRegressor` and the
  `has_fit_parameter` branch in `count_model.py`; exports.
- Tests, `test_modeling_negative_binomial.py`: recovers known parameters and α
  on NB2 draws (slow-marked if > 1 s); the offset enters `predict`
  (`predict(X, offset=log 2e) == 2 · predict(X, offset=log e)`); a
  non-converged fit raises (`max_iter=1`); columns in another order raise at
  `predict`; `dispersion_` positive; sklearn's `check_estimator` subset the
  contract test already runs, applied here directly.
  `test_modeling_count_model.py`: with an offset estimator and `use_exposure`,
  `fit` passes `offset=log(exposure)` and no `sample_weight`, and `predict`
  passes the same offset (a spy estimator); a `Regressor` still gets the rate;
  the fitted model follows `estimator_`, not a later `set_params(estimator=…)`;
  NB2 through `CountModel` equals statsmodels with `exposure=` directly
  (the two statsmodels forms, 1e-14). Contract `EXAMPLES` entry for the NB2 variant
  if `CountModel`'s example should stay Poisson (one entry per class: keep
  Poisson, test NB2 in the model's own file).
- Probe to record: statsmodels under `-W error` on the simulated table; the
  convergence check (`mle_retvals["converged"]`) vs its `ConvergenceWarning`.
- **Verify:** mutation checks for each test; mypy clean; a `CountModel` with
  `NegativeBinomialRegressor` fits the simulated totals and its α is reported.

### [ ] 3. `CohortProbabilityModel` on a classifier (no calibration yet)
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
  torch Dirichlet build on the same split (a probe; both importable until
  sub-task 5): report both numbers.

### [ ] 4. Calibration
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

### [ ] 5. `TotalTimesProbabilityModel`, deletions, exports
- `modeling/total_times_probability.py` (P12); delete
  `independent_total_probability.py`, `total_children.py`, `calibration.py`,
  `optimization.py`, their tests and `tests/validation/test_total_children.py`;
  update `modeling/__init__.py`, the contract test's `EXAMPLES`,
  `test_modeling_feature_transformer.py`;
  `grep -rn "TotalChildrenModel\|TemperatureCalibrator\|IndependentTotalProbabilityModel\|single_threaded_torch\|modeling.optimization" src tests --include='*.py'`
  must hit only the old stack (`models/`, `experiment/`, `tracking/`, their tests).
- Tests (`test_modeling_total_times_probability.py`): `fit` gives the row sum
  to the total model and `y` to the probability model (spies); `predict` =
  total × probabilities, columns in `y`'s order, index of `X`; rows of
  `predict` sum to the total's prediction; a probability model with other
  columns raises; exposure reaches both; templates stay unfitted (clones);
  nested `set_params` names (`probability_model__calibration_method`); refit
  equals fresh fit; works with each total variant (`CountModel` Poisson with
  and without exposure, `CountModel` with `NegativeBinomialRegressor`).
- **Verify:** `uv run pytest -m "not slow"` passes; the `grep` above; mypy
  clean; no `import torch` under `modeling/`.

### [ ] 6. Smoke run (as PR #11's B7)
- Script in the scratchpad, recipe in the plan doc: ten simulated populations,
  grouped 80/20 split, class defaults (untuned). Compare Model 1 (PR #12's
  `IndependentCohortModels` with LightGBM) against Model 2 variants:
  total ∈ {`PoissonRegressor` with exposure, `NegativeBinomialRegressor` with
  exposure, `LGBMRegressor(objective="poisson")` with exposure} × probability ∈
  {`LogisticRegression`, `LGBMClassifier`, HGB, `RandomForestClassifier`} ×
  calibration ∈ {None, temperature}. Metrics: per-cohort and total
  Poisson deviance, `COHORT_LOG_LOSS`. Record the table and the fitted NB2 α.
- **Verify:** the table; a one-paragraph reading (what won, by how much, noise).

### [ ] 7. Docs and close
- `docs/INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 rewritten for the new build
  (§9 lists what it must explain); `docs/DIRECT_COHORT_MODEL.md` §0 (the offset
  rule, NB2); `docs/FEATURE_TRANSFORMATIONS.md` §8.1–8.3 usage blocks;
  `docs/MODULE_REFERENCE.md`; `docs/README.md`; `docs/MODEL_REIMPLEMENTATION_PLAN.md`
  §5; `docs/HYPERPARAMETER_TUNING_PLAN.md` where it names Model 2's tunables;
  `docs/MULTI_COHORT_MODELS_PLAN.md` top note. Every code block run.
- Memory: update `multi-cohort-models-plan.md` (the state), note the decisions.
- PR body drafted in the scratchpad; the user applies it and marks the PR ready.
- **Verify:** `grep -rn "TotalChildrenModel\|TemperatureCalibrator\|Dirichlet\|DirectCohortModel" docs --include='*.md'`
  hits only historical records (plan docs' step records, the old stack's §1–§12).

## 9. Docs: what each decision's explanation must say

In `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 (the model doc), one subsection
each, with the formulas and the measured figures of §6; the total model's
part also in `DIRECT_COHORT_MODEL.md` §0:

1. **The model.** `log μ_b = log A_b + b + x_bᵀβ` (Poisson, NB2, or any
   regressor's mean × exposure); `p_{b,k} = classifier(w_b)`;
   `Ĉ_{b,k} = μ̂_b p̂_{b,k}`, `Σ_k Ĉ_{b,k} = μ̂_b`.
2. **The total model is `CountModel`.** Why one class serves a cohort and the
   total; the two exposure forms: the weighted rate (the Poisson offset model
   exactly; for a Gaussian or tree loss it is "mean = exposure × f(x)", not an
   offset) and the offset (`log(exposure)` with coefficient 1, for any
   estimator whose `fit` takes `offset`, detected with `has_fit_parameter`);
   why NB2 needs the offset (the 0.021 difference; non-integer `y` accepted
   silently); `alpha × mean(exposure)`.
3. **NB2 representation.** NB2 = `Var = μ(1 + αμ)`, α fitted jointly by
   maximum likelihood (statsmodels' `loglike_method="nb2"`); what
   `predict(offset=)` computes and that it equals `exposure=`; the
   alternatives table (statsmodels discrete NB, statsmodels GLM with fixed α
   and L2, torch, glum, LightGBM/XGBoost custom objective, scikit-learn: none)
   and why the first.
4. **Data replication.** Multinomial → categorical: `Mult(Y_b, p_b)` ∝
   `Π_k p_{b,k}^{C_{b,k}}`, so one weighted row per positive cell with weight
   `C_{b,k}` is the same log-likelihood as one row per child; the 1.8e-15
   check; why the weight is the count and not `count / total`; why one
   representation suffices (P8's table); where the conversion is done and why
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
   by `has_fit_parameter` like the NB2 offset); what to tune.
6. **Calibration.** `CalibratedClassifierCV`: cross-fitting (`ensemble=False`)
   vs the ensemble; why folds are grouped by building (and the neighborhood
   limitation); 5 folds (the (k−1)/k argument, the 50/50 bias); temperature
   (1 parameter, softmax) vs sigmoid (OvR, 2 per class, renormalized) vs
   isotonic (non-parametric; sklearn's ≥ 1,000-samples guidance; overfits
   here); every method returns rows summing to 1; the measured table from
   sub-task 4; `C` and the weights are per child.
7. **Classifiers.** The `Classifier` protocol; multinomial vs OvR; `classes_`
   mapping; `OneVsRestClassifier` and `sample_weight` through metadata routing.
8. **API, data flow, rules, errors** tables as the present §0.2–§0.5, updated.

## 10. Risks

- **statsmodels under `-W error`:** `NegativeBinomial.fit` runs a preliminary
  Poisson fit; a `ConvergenceWarning` would become an error in the suite
  before the model's own check. Measured clean on the probes; sub-task 2 probes
  the simulated table and, if needed, sets `optim_kwds_prelim`.
- **`add_constant` skips a constant column** (`has_constant="skip"`): a design
  with an all-ones column would use it as the intercept silently (harmless);
  two constant columns fail statsmodels' rank check. Document.
- **`has_fit_parameter` on a `Pipeline` or meta-estimator** inspects the outer
  `fit`, so an offset estimator inside a `Pipeline` is not detected: documented
  (the feature transformer belongs to `CountModel`, so no `Pipeline` is needed).
- **`OneVsRestClassifier` + `sample_weight`** needs metadata routing; if the
  `Classifier` protocol cannot express it cleanly, document OvR as "enable
  routing" rather than special-case it.
- **`CalibratedClassifierCV` with a fold lacking a class:** grouped folds over
  ~245 buildings always contain every cohort in practice; a rare cohort could
  break a fold, and sklearn raises. Left to the library.
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
  `probe_end_to_end.py` and `probe_tuning_plan_s5.py` also name `DirectCohortModel`. **Open, for the user at sub-task 1.**

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
