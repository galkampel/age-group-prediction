# Validation: the hyperparameter tuning package (PR #5)

**Validated:** `feat/hyperparameter-tuning` at `f219cb8` (= `origin`), draft PR #5, not merged.
**Date:** 2026-10-12. **Versions:** Python 3.13.9, optuna 4.9.0, lightgbm 4.7.0, scikit-learn 1.9.0,
pydantic 2.13.4, numpy 2.5.1, pandas 3.0.5.
**Under validation:** `src/age_group_prediction/hyperparameter_tuning/` and its tests, against
[HYPERPARAMETER_TUNING_PLAN.md](HYPERPARAMETER_TUNING_PLAN.md) §2 (D1–D20) and §4, and the user guide
[HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md).

**How it was run:**
- Nothing in the repository was changed except this file.
- Every check ran on a `git archive f219cb8` copy in a scratch directory. The working tree was not
  usable: two plan docs were overwritten with older texts during the session, and `parameters.py` has
  an uncommitted diff (§6).
- Docs were read with `git show f219cb8:<path>`.
- Before every run, a guard asserted that `age_group_prediction.hyperparameter_tuning` was imported
  from the scratch copy. This matters because the repository's `.venv` has an editable `.pth` pointing
  at the repository's own `src`.
- The scripts are in §8.

## 1. Verdict: ACCEPT WITH NOTES

The code implements every decision, D1 and D3–D20, as stated. D2, D6 and D13 are superseded and
gone. Each decision is backed by a probe.

- **The user guide:** all seven code blocks print exactly the documented output. Block 1 took 9.3 s;
  the guide says 9 s.
- **The tests:** they catch what plan §4.4's test table claims. All 30 mutations were killed by the
  named tests, with no survivor.
- **The worked example:** the trial value equals a hand-written fold loop bit for bit. The NB2
  one-trial client gives the guide's 2.247 and 0.454.
- **The corrected SE:** the plan's statements hold on a repeated-sampling reconstruction. A fixed
  predictor's naive SE ≈ true, and the correction overstates it by about 1.4×. Singleton strata make it
  7.4% conservative (the plan says 7%). The SE's own error at K=5 is 36% (the plan says ~35%).
- **Reproducibility:** holds as documented.
- **The routine is clean:**
  - ruff;
  - mypy (only the four deliberate errors in the parameters tests);
  - 131 package tests under `-W error`;
  - the non-slow suite: 1244 passed, 1 skipped, 1 xfailed;
  - the slow integration test: 3 passed.

**The findings (§5):**
- **F1, code, Low, your decision:** HEAD accepts NaN and ±inf categorical choices. A NaN reaches
  `best_params`, and `to_dict()` no longer round-trips through JSON. This contradicts "D18 makes every
  categorical value JSON-native". The uncommitted `parameters.py` diff closes it, but no test covers
  that check and no D-decision records it (§6).
- **Six documentation discrepancies, all Low**, each with a proposed edit:
  - a pruner warning that does not exist;
  - a LightGBM thread claim that did not reproduce;
  - the `cv` check described as stronger than it is;
  - the import graph;
  - the Median/Percentile "never pruned";
  - an error-prefix rule with one exception.

## 2. Phase table

| # | Group | Result | Evidence |
|---|---|---|---|
| 0 | Setup and routine | pass | Guard on every run. `ruff check`: all checks passed; `ruff format --check`: 11 files already formatted. `mypy`: no issues in 24 source files. `mypy <package> <tests>`: the 4 deliberate `arg-type` errors of `test_hyperparameter_tuning_parameters.py:157–161` (plan §7). Package tests under `-W error`: **131 passed**. Non-slow suite: **1244 passed, 1 skipped, 1 xfailed, 23 deselected** (99 s). Slow integration test: **3 passed** |
| 1 | D-decisions | pass | 20 probes, §3.1 |
| 2 | Guide and plan §4 claims | pass, 6 doc discrepancies | 23 probes plus reading, §3.2; discrepancies in §5 |
| 3 | The guide's 7 blocks as written | pass | 7 of 7 outputs **IDENTICAL**, §3.3 |
| 4 | Mutations (§4.4 test table) | pass | **30 killed, 0 survived**, §4 |
| 5 | Hand-written fold loop, NB2 | pass | difference **0.00e+00**, §3.4 |
| 6 | Corrected SE vs repeated sampling | pass | §3.5 |
| 7 | Pruning guidance | pass, 1 doc discrepancy (D5) | §3.6 |
| 8 | Reproducibility | pass, 1 doc discrepancy (D2) | §3.7 |
| 9 | The uncommitted `parameters.py` diff | described, not validated as HEAD | §6 |

## 3. Claims

Each row gives the claim, how it was checked, the observed output and a verdict. Command keys:
- **P** = `probe_claims.py`
- **W** = `probe_worked_example.py`
- **T** = `probe_threads.py`
- **E** = `probe_se.py`
- **R** = `probe_pruning.py`
- **G** = `run_guide_blocks.py`

All are run from the scratch copy with `PYTHONPATH=src` (§8).

### 3.1 Decisions (P)

| Decision | Claim | Observed | Verdict |
|---|---|---|---|
| D1 | One frozen, strict pydantic dataclass per parameter type; Optuna checks the values | frozen; pydantic dataclasses; `IntParameter("p", 1.5, 9)` and `("p", True, 9)`: "Input should be a valid integer"; `FloatParameter("p", 1.0, 0.0)`: "parameter 'p': \`low <= high\` must hold" | pass |
| D2, D6 | Superseded: no sklearn estimators/scorers | a scorer string as `metric` and a `Ridge` as `model` are rejected at construction | pass |
| D3 | Every trial splits with a `Splitter.cv` validator with an int seed, ≥ 2 folds | for all 3 methods: `n_splits=1` raises, `random_state=None` raises, two `split()` calls give identical folds | pass. The evaluator itself checks only the type (D3 in §5) |
| D4 | Only the trial's parameters are fitted inside the objective | folds identical before and after a trial; 3 fits per trial on 3 folds | pass |
| D5 | `Aggregator` with one method; `WeightedMean` default; `Mean`, `UpperBound(z)` | `Aggregator` abstract; default `WeightedMean()`; `z` ∈ {0, −1, nan, inf} rejected; `UpperBound(2)` 3.825742 = hand value | pass |
| D7 | Sampler and pruner objects with seeded defaults | default `TPESampler`, `_multivariate=True`; default `NopPruner`; given ones used by identity | pass |
| D8 | Only COMPLETE trials can win | a pruned trial reporting −100 does not win; its record's `value` is `None` | pass |
| D9 | No conditional or derived parameters | `Parameter` subclasses: exactly the three | pass |
| D10, D15 | Settings in the constructor; `evaluate(trial, X, y, groups=None, *, exposure=None)` | signature as stated; evaluator frozen; no data fields | pass |
| D11 | `build_model` = `clone` + `set_params`; a misspelt name raises | a copy (template `alpha` stays 1.0, copy 0.5); `estimator__alpah`: "Invalid parameter 'alpah'" | pass |
| D12 | The study minimizes; a greater-is-better metric is negated; the result is in the metric's own sign only | direction `MINIMIZE`; Optuna value −0.7, `best_value` 0.7, fold values 0.7; `TrialRecord` has no `score`, `fold_scores`, `metric_name` | pass |
| D13 | Superseded: the evaluator has no `feature_transformer` | fields `model, parameters, cv, metric, aggregation` | pass |
| D14 | The exposure is sliced per fold and always passed | `None` arrives as `None`; with an exposure, lengths `[26, 14, 27, 13, 27, 13]` = the folds' train/val sizes | pass |
| D16 | `Metric` lives in `scoring.py` | `Metric.__module__ == "age_group_prediction.scoring"` | pass |
| D17 | Only `BaseAgeGroupModel`s are tunable; a contract test | `Ridge()` as `model` rejected; `test_modeling_contract.py` passes in the suite | pass |
| D18 | Choices are `None`, `bool`, `int`, `float`, `str` | list, tuple, `object()`, `np.int64` rejected; `np.float64` kept by identity; a set rejected; a list stored as a tuple | pass. NaN/inf floats are accepted: F1 |
| D19 | Sequential trials, fixed `n_trials`; no timeout, `n_jobs`, `initial_params`, `is_reproducible` | fields `seed, n_trials, sampler, pruner`; the three keywords rejected as extras; `TuningResult` has no `is_reproducible` | pass |
| D20 | An empty search space is allowed and scores the model as given | `parameters=[]` builds; 3 trials give 3 equal values `0.86008937…`; `best_params == {}` | pass |

### 3.2 Guide and plan §4 claims (P, unless stated)

| Source | Claim | Observed | Verdict |
|---|---|---|---|
| guide §1 | Not re-exported from `age_group_prediction`'s root | the root `__init__` does not mention it | pass |
| guide §1, plan §3 | The import graph | grep: `evaluator` imports `modeling`, `scoring`, `utils`, `_config`, `aggregation`, `parameters`; `study` imports `_config`, `aggregation` | **D4** |
| guide §3, plan §4.1 | The parameter table: `log` needs `low > 0` (float) / `≥ 1` and `step == 1` (int); step must divide; `log` and `step` keyword-only | all 7 bad cases rejected; int bounds accepted by `FloatParameter` | pass |
| guide §3, plan §4.1 | Three rules Optuna lets through | Optuna: `step=0.3` silently makes `high` 0.9 (a `UserWarning` only); duplicate choices: no warning; a repeated name returns the cached value. All three rejected here | pass |
| plan §4.1 | A value error is prefixed `parameter '<name>'`; a type error names the class and the argument | Optuna's errors and the duplicate-choice error are prefixed; the **choice-type error is not** (guide §3's own output shows it). Type error: title `FloatParameter`, `loc=(1,)` | **D6** |
| guide §4 | The evaluator checks its settings when built | string or dict `metric`, string `aggregation`, a set of parameters and a misspelt keyword are all rejected at construction; parameters stored as a tuple | pass |
| guide §4 | `cv` is "a `Splitter.cv(n_splits=..., random_state=int)` validator" | checked by type only: `KFold(3, shuffle=True)` without a seed is accepted, and its folds differ between `split()` calls | **D3** |
| guide §4, plan §4.2 | The exposure is checked: one value per row | `N+1` values: "exposure must hold one value per row of X (40), got shape (41,)"; `(N, 1)` rejected | pass |
| guide §4 | `groups=None` for `random`; `KFold` warns otherwise | 1 warning with groups ("The groups parameter is ignored by KFold"), 0 without | pass |
| guide §4 | A numpy `X` works only with `feature_transformer=None` | with a transformer: "Specifying the columns using strings is only supported for dataframe"; without one: scored | pass |
| guide §4, plan §4.2 | The user attrs `fold_scores`, `fold_sizes`, `metric_name`, `greater_is_better`, plain Python, sqlite-safe | exactly these four; types `list, list, bool, str`; an in-memory sqlite study stores them; exact JSON round trip with `allow_nan=False` | pass |
| guide §4 | `FixedTrial(best_params)` scores one setting with the same folds | equal to `best_value`, bit for bit (also W: `1.812 True`) | pass |
| plan §4.2 | A one-fold non-Splitter `cv` fails loudly in `corrected_std_error` after the search | `evaluate` alone returns 0.5594; the study raises "the standard error needs at least 2 folds, got 1" after both trials ran | pass |
| guide §5, plan §4.3 | `WeightedMean` is the pooled per-row score for a per-row mean metric | fold-weighted mean 0.833251059784241 = pooled 0.833251059784241 | pass |
| plan §4.3 | `COHORT_LOG_LOSS` (a per-child mean): the size-weighted mean is close to the pooled loss, not equal | 1.11721 vs 1.11767 | pass |
| plan §4.3 | `aggregate` rejects no folds and unequal lengths; the SE needs 2 folds; field-less aggregators compare by value | all three classes raise both errors; `corrected_std_error([1.0])` raises; `WeightedMean() == WeightedMean()`, repr `WeightedMean()` | pass |
| plan §4 | `_STRICT = strict, allow_inf_nan=False, extra="forbid"`; `TrialRecord`/`TuningResult` frozen stdlib dataclasses; `Aggregator` a plain ABC; the study not frozen | as stated | pass |
| guide §6, plan §4.4 | The constructor: `n_trials < 1`, `seed=True`, `n_trials=2.0`, a string or dict sampler, a misspelt keyword and a positional seed are rejected; Optuna runs 0 trials silently; a negative seed fails clearly | all 7 rejected; Optuna `n_trials=0`: 0 trials, no error; `seed=-1`: "Seed must be between 0 and 2**32 - 1" | pass |
| plan §4.4 | The default sampler's `ExperimentalWarning` is silenced at construction only | raw `TPESampler(multivariate=True)`: 1 `ExperimentalWarning`; `optimize`: 0 warnings | pass |
| guide §7 | "Both pruners emit Optuna's `ExperimentalWarning` once, when constructed" | `WilcoxonPruner`: `ExperimentalWarning`; `SuccessiveHalvingPruner`: **none** (`MedianPruner`: none) | **D1** |
| guide §6, plan §4.4 | `optimize`: an objective error propagates and the study is kept; no completed trial → `RuntimeError`; a tie goes to the lowest number; `best_params` is a copy | `ZeroDivisionError` propagates, `optuna_study_` holds 1 FAIL trial; all pruned: "no trial completed: every trial was pruned"; tie between trials 2 and 4 → 2 (Optuna's `best_trial` → 2, as the plan says); editing `best_params` leaves the study intact | pass |
| guide §6, plan §4.4 | Records: COMPLETE, PRUNED, FAIL; all but number/state/params `None` unless COMPLETE; a pruned trial's Optuna value is its last intermediate; `to_dict()` JSON-safe with state names; a plain objective has no `metric_name`/folds; an `inf` reaches `best_value` and strict JSON raises | states `COMPLETE, PRUNED, FAIL, COMPLETE`; pruned record `value=None` while Optuna's is 100.0; round trip equal; `metric_name None`, folds `None`; `inf` → `json.dumps(allow_nan=False)` raises | pass. JSON-safety fails for a NaN choice: F1 |
| guide §10, plan §4.4 | An evaluator NaN stops the study (trial FAIL); a plain objective's NaN or `None` fails only its trial, with a log line and no Python warning | evaluator: `ValueError` "trial 0, fold 0: score is nan", states `[FAIL]`; plain: `FAIL, COMPLETE, FAIL`, 0 Python warnings | pass |
| guide §10 | One column per metric | `POISSON_DEVIANCE` on `(4, 2)`: "Multioutput not supported in mean_tweedie_deviance" | pass |
| guide §10 | `ExposureTransformer` rejects what LightGBM would accept | zero and NaN exposures rejected | pass |
| guide §9 | `IndependentCohortModels` does not forward `set_params`; nested names reach into `TotalTimesProbabilityModel` | deep params only `cohort_models`; `cohort_models__a__estimator__alpha` raises `AttributeError`; `total_model__estimator__alpha=0.3` and `probability_model__calibration_method` set | pass |
| guide §10 | Building parameters changes the warning filter process-wide (not thread-safe) | by reading `parameters.py:58–63` (`warnings.catch_warnings` + `simplefilter`, process-global) | pass (reading) |
| guide §2 | The same loop is the integration test | `test_hyperparameter_tuning_integration.py`, 3 passed (slow) | pass |

### 3.3 The guide's code blocks, as written (G)

`run_guide_blocks.py` extracts the seven ```` ```python ```` blocks of the guide at `f219cb8` in order. It
runs them in one namespace with `cwd` at the scratch copy, captures each block's stdout and compares it
with the ```` ```text ```` block that follows.

| Block | Section | Time | Output | Warnings |
|---|---|---|---|---|
| 1 | §2 worked example | 9.3 s (guide: 9 s) | IDENTICAL | none |
| 2 | §3 parameter errors | 0.0 s | IDENTICAL | none |
| 3 | §4 `FixedTrial`, NB2 | 0.1 s | IDENTICAL | none |
| 4 | §5 aggregation | 0.0 s | IDENTICAL | none |
| 5 | §6 `to_dict`, plain objective | 0.0 s | IDENTICAL | none |
| 6 | §7 pruning (`18 12 130 1.694 1.812`) | 1.7 s | IDENTICAL | `ExperimentalWarning: WilcoxonPruner is experimental` |
| 7 | §8 reproducibility (`True`) | 2.1 s | IDENTICAL | none |

### 3.4 The fold loop by hand, and NB2 (W)

| Claim | Check | Observed | Verdict |
|---|---|---|---|
| The evaluator's trial value is the size-weighted mean of the fold deviances | Guide block 1 run as written, then, for the best `n_elementary` trial (13), for each fold of `evaluator.cv.split(train, y, groups_train)`: `clone`, `set_params(**best_params)`, `fit` with the fold's exposure, `predict`, `mean_poisson_deviance`, weighted by fold size | by hand **1.811570047925884** = `best_value` 1.811570047925884, difference 0.00e+00; fold scores `[2.07484, 1.561176, 2.212915, 1.765381, 1.419605]` and sizes `[12, 11, 11, 11, 11]` equal the record's; SE by hand 0.224685 = `std_error` | pass |
| `stratified_by_group` never validates a row alone in its stratum | the fold sizes above | they sum to 56 of 59 training rows | pass (as documented) |
| NB2, an empty search space, one trial (guide §4: `2.247 0.454`) | the same loop with `NegativeBinomialRegressor` and `{}` | by hand 2.246929 = `best_value`; SE by hand 0.453802 = record; `best_params {}`; 1 trial. Three trials: `2.246928873` three times | pass |

### 3.5 The corrected SE against repeated sampling (E)

The plan does not record the setup behind its numbers, so this is a reconstruction:
- **Data:** a Poisson DGP, `log μ = 0.5 + 0.3x₁ − 0.2x₂ + 0.15x₃`, 5 features, n = 200 rows, K = 5
  `random` folds, seed `r` per dataset.
- **The true SD:** the SD, over R fresh datasets, of the CV estimate.
- **Compared with:** the mean naive `s/√K` and the mean corrected SE computed on each dataset.
- **(b) and (b2)** run through the package's evaluator (`FixedTrial`) and `corrected_std_error`.
- **The verdicts** judge the plan's statements and ratios, not its digits.

| Case | Claim (plan §4.3, guide §5) | Observed | Verdict |
|---|---|---|---|
| (a) fixed predictor (the true mean), R = 2000 | "nothing to correct and it overstates": true 0.061, corrected 0.090, naive 0.060 (ratios 1.48×, 0.98×) | true 0.1000; naive 0.0927 (0.93×); corrected 0.1391 (1.39×) | pass |
| (b) `PoissonRegressor(alpha=1e-4)`, R = 1000 | the naive SE understates; the correction is an approximation | true 0.1069; naive 0.0963 (0.90×); corrected 0.1444 (1.35×) | pass (note N1) |
| (b2) LightGBM Poisson, 100 trees, R = 400 | as (b) | true 0.1618; naive 0.1177 (0.73×); corrected 0.1766 (1.09×) | pass |
| (c) `stratified_by_group`, 40 singletons in 200 rows (20%) | "slightly conservative: 7% at 20% singletons" | 160 of 200 rows validated; mean `n_val/n_train` 0.1905 vs 1/(K−1) 0.25; overstated by **7.4%** | pass |
| (d) the SE's own sampling error at K = 5 | "~35%" | relative SD of `s` from normal scores: **0.364**; the corrected SE's own coefficient of variation in (a), (b), (b2): 0.37, 0.38, 0.41 | pass |

### 3.6 Pruning guidance (R)

These are synthetic objectives that report each fold's score at `step = fold`, exactly as the evaluator
does. The pruners are passed to `HyperparameterStudy`.

| Claim | Check | Observed | Verdict |
|---|---|---|---|
| Plan §4.2: on 10 folds of unequal difficulty, a bad trial that ties on fold 0 is pruned by `WilcoxonPruner` after 5 folds and by `SuccessiveHalvingPruner` after 2; `MedianPruner`/`PercentilePruner` "never did, whatever was reported" | Difficulties `linspace(0.5, 2, 10)` in three orders; 10 good trials (noise SD 0.05), then a bad one (+0.5 on folds 1–9); defaults (`Wilcoxon p=0.1`, `PercentilePruner(25)`) | **Random order:** Wilcoxon after 4, SH after 2, **Median after 2**, **Percentile after 1**. **Fold 0 easiest:** Wilcoxon after 5, SH after 2, **Median never**, Percentile after 1. **Fold 0 hardest:** Wilcoxon after 4, SH after 2, Median after 2, Percentile after 2. Good trials pruned (of 10, all equally good): Wilcoxon 1–2, SH 6, Median 1–5, Percentile 2–5 | Wilcoxon and SH pass; **"never" holds only when fold 0 is the easiest**: D5. The guide's wording ("they prune by fold luck") holds |
| Plan §4.2: fixed vs shuffled fold order, 8 folds, 80 trials, 4 seeds. Wilcoxon about the same; SH shuffled worse (3 of 4 in the plan's run, 4 of 4 in the review's) | Shuffled per trial; Wilcoxon reports the fold id as the step (as Optuna's docs do), SH the position | Wilcoxon: equal best values in all 4 seeds (1.2008, 1.2012, 1.2008, 1.2007); SH: shuffled worse in **4 of 4** (1.2182 vs 1.2056, 1.2027 vs 1.2019, 1.3186 vs 1.2019, 1.2196 vs 1.2025) | pass |
| Guide §7: on the worked example (seed 101, 30 trials), `WilcoxonPruner()` at its defaults pruned nothing and `SuccessiveHalvingPruner()` completed only 2 of 30 trials | Guide block 1, then each pruner | Wilcoxon: 30 complete, 0 pruned, best 1.812; SH: **2 complete**, 28 pruned, best 1.839 | pass |

### 3.7 Reproducibility (W, T)

| Claim | Observed | Verdict |
|---|---|---|
| Guide §6, §8: the same seed gives an identical result, trial for trial; two `optimize` calls on one study draw the same trials | worked example, seed 101, 30 trials: the same object twice `True`; a new object, same seed `True`; equal to the guide's `result` `True` | pass |
| A different seed differs (the mutation M10 test) | seed 102: different trial-0 params, best 1.8035 vs 1.8116 | pass |
| Guide §8: LightGBM with `n_jobs=1`, "more threads change the fit" | Each in its own subprocess: the worked example's 30 trial values, `best_value` and 60 predictions are **bit-identical** for `n_jobs` 1, 4 and 8; a 100 000 × 30 Poisson fit (300 trees, 63 leaves, bagging) is bit-identical for 1 vs 4 threads, row-wise and column-wise; `n_jobs=1` and `n_jobs=4` each repeat exactly | **not reproduced**: D2 |

## 4. Mutations (plan §4.4's test table)

`mutate.py`:
- Each mutation is one or more string replacements that must match exactly once in the scratch copy's
  `src`.
- It then runs the package's non-slow tests under `-W error`, through the guard, and records the failed
  tests.
- Finally it restores `src` from the pristine archive and compares byte for byte.
- Baseline: 128 passed, 3 deselected (the slow ones), 0 failed.

The table's "drop `seed=`" row and its "an unseeded default sampler" row are one edit (M01). Every test
the table names was among the failures.

| # | Mutation | Named test | Failed tests | Result |
|---|---|---|---|---|
| M01 | drop `seed=` (an unseeded default sampler) | `…default_sampler_is_seeded_by_the_study`, `…same_seed_gives_the_same_result` | those 2 | killed |
| M02 | drop `multivariate=True` | `…default_sampler_is_multivariate` | 1 | killed |
| M03 | `direction="maximize"` | `…sampler_searches_toward_lower_scores` | 1 | killed |
| M04 | `MedianPruner()` as the default | `…default_pruner_never_prunes` | 1 | killed |
| M05 | deep-copy the given sampler and pruner | `…given_sampler_and_pruner_are_used_as_given` | 1 | killed |
| M06 | remove the warning filter | `…default_sampler_does_not_warn` | 21 (every study test under `-W error`) | killed |
| M07 | `n_trials < 0` | `…fewer_than_one_trial_is_rejected` | 1 | killed |
| M08 | drop `extra="forbid"` | `…misspelt_setting_is_rejected` | 1 | killed |
| M09 | drop `_STRICT` from the study's config | `…bool_seed_is_rejected` | 2 (and the misspelt setting) | killed |
| M10 | `seed=0` hard-coded (review D1) | `…same_seed_gives_the_same_result` | 1 | killed |
| M11 | `best_params` from the first complete trial (review D2) | `…best_params_are_the_best_trials` | 1 | killed |
| M12 | `best_value` from the first complete trial (review D2) | `…best_params_are_the_best_trials` | 1 | killed |
| M13 | PRUNED among the candidates | `…pruned_trial_never_wins` | 2 | killed |
| M14 | drop the no-complete check | `…no_completed_trial_raises` | 1 | killed |
| M15 | don't keep `optuna_study_` | `…optuna_study_is_kept` | 5 | killed |
| M16 | `best_params=best.params` | `…editing_the_result_leaves_the_study_intact` | 1 | killed |
| M17 | the naive `s/√K` | `…complete_record_carries_its_folds_and_their_std_error` | 1 | killed |
| M18 | a pruned record's `value=trial.value` | `…pruned_record_reports_no_value` | 1 | killed |
| M19 | record COMPLETE and PRUNED only | `…failed_trial_is_recorded` | 1 | killed |
| M20 | a record's `params=trial.params` | `…editing_a_record_leaves_the_study_intact` | 1 | killed |
| M21 | a NumPy `int64` in `to_dict` | `…to_dict_round_trips_through_json` | 1 | killed |
| M22 | keep the `TrialState` in `to_dict` | `…to_dict_names_the_state` | 1 | killed |
| M23 | the sign condition reversed | `…loss_is_reported_as_it_is` | 6 | killed |
| M24 | `value=trial.values[0]`; `best_value=best.values[0]` | `…greater_is_better_metric_is_reported_in_its_own_sign` | 2 | killed |
| M25 | one sign for every trial (the best trial's) | `…each_trial_is_reported_in_its_own_metric_sign` | 1 | killed |
| M26 | default `greater_is_better=True` | `…plain_objective_is_reported_unsigned` | 2 | killed |
| M27 | the evaluator's `greater_is_better` attr hard-coded `True` | `…completed_trial_records_its_folds` | 1 (the `False` case) | killed |
| M28 | `Field(min_length=1)` back on `parameters` | `…empty_search_space_scores_the_model_as_given` | 1 | killed |
| M29 | drop the choice validator | `…choice_outside_optunas_types_is_rejected` | 4 parametrized cases | killed |
| M30 | convert float choices with `float(c)` | `…choices_are_stored_as_given` | 1 | killed |

**30 killed, 0 survived.** After the last restore, `diff -r` of the scratch `src` against the pristine
archive is empty.

## 5. Findings and proposed fixes

**Processed 2026-10-12** (after this report, on the same branch):
- **F1:** `_scalar_choices` now rejects a NaN or infinite float choice. The duplicate check is
  `len(set(choices)) != len(choices)`, without the `TypeError` fallback. That is your choice: an
  unhashable float subclass now raises a raw `TypeError`, and a subclass whose hash disagrees with `==`
  is not caught. Both break Python's hash/`==` contract.
- **The new test:** `test_a_non_finite_float_choice_is_rejected`. Mutation checks: dropping the
  finite check fails its 3 cases, and dropping or loosening the duplicate check fails the duplicate
  tests. D18 records the rule.
- **D1, D3–D6 and N2:** fixed as proposed in the guide and the plan, plus the "four rules" and JSON
  sentences that F1 touched.
- **D2:** re-measured before rewording.
  - LightGBM does run multithreaded here: 8 threads use 2.4× CPU per wall-second, at half the wall
    time.
  - Seven settings were bit-identical across 1, 2, 4 and 8 threads, and repeatable at 8: 60 to
    400 000 rows, up to 500 columns, Poisson with weights and bagging, row- and column-wise
    histograms, and `deterministic=True`.
  - LightGBM's parameter docs promise stability across `num_threads` only with `deterministic=true`.
  - Guide §8 now says both, and keeps `n_jobs=1` as the safe default.
- The table below is kept as reported.

| # | Severity, kind | Where | What | Reproduction | Proposed fix |
|---|---|---|---|---|---|
| F1 | Low, **code** (your decision) | `parameters.py:122–132` (`_scalar_choices`) | NaN and ±inf float choices pass D18's type check. Strict pydantic's `allow_inf_nan=False` does not reach a `tuple[Any, ...]` field. A NaN choice reaches `best_params`: `json.loads(json.dumps(to_dict())) == to_dict()` is `False`, `json.dumps(..., allow_nan=False)` raises, and the default writes `NaN`. This contradicts plan §4.4 ("D18 makes every categorical value JSON-native") and guide §6 ("`to_dict()` is JSON-safe") for this input. `[nan, nan]` (one object) also passes the duplicate check, since `nan != nan`. No planned client uses a float choice | `probe_diff.py` on HEAD: `[nan, 1.0]`, `[inf, 1.0]`, `[-inf, 1.0]` accepted; `best_params={'p': nan}` | The uncommitted diff (§6) does this. If you keep it: a D18 note in the plan, and a test (a NaN/inf choice raises), because the check survives its own mutation today |
| D1 | Low, doc | guide §7, last sentence | "Both pruners emit Optuna's `ExperimentalWarning` once, when constructed": `SuccessiveHalvingPruner` emits none | P, `G7-both-pruners-warn` | "`WilcoxonPruner` emits Optuna's `ExperimentalWarning` once, when constructed." |
| D2 | Low, doc | guide §8 ("with `n_jobs=1`: more threads change the fit") | Not reproduced on LightGBM 4.7.0: 1, 4 and 8 threads give bit-identical fits on the worked example and on a 100k-row problem | T, §3.7 | Keep `n_jobs=1` and change the reason to what was measured: "`n_jobs=1` (the examples' setting; measured on LightGBM 4.7.0, 1, 4 and 8 threads gave identical fits, so the thread count is not part of the seed)". Or drop the parenthesis. Your choice; the macOS OpenMP-with-torch crash noted in the PR #12 validation is a separate reason |
| D3 | Low, doc | guide §4, first paragraph | "checks them when built: … `cv` a `Splitter.cv(n_splits=..., random_state=int)` validator": only the type is checked. A seedless scikit-learn validator is accepted and gives each trial different folds | P, `D3-cv-type-only` | "`cv` a scikit-learn cross-validator, from `Splitter.cv(n_splits=..., random_state=int)` so every trial sees identical folds (D3; only the type is checked)" |
| D4 | Low, doc | guide §1; plan §3 | Guide: "`evaluator → parameters, aggregation, modeling, scoring, splitting`"; the evaluator imports `utils`, not `splitting`. Plan §3: "`evaluator` → … `feature_engineering`" (it imports `utils`); "`study` imports only `aggregation.corrected_std_error`" (and `_config`); "`_config.py` … shared by parameters and aggregation" (by all four modules); the test list omits `test_hyperparameter_tuning_integration.py` | grep of the imports, §3.2 | Guide: "`evaluator → parameters, aggregation, modeling, scoring, utils`". Plan §3: the same list; "`study` imports only `aggregation.corrected_std_error` (and `_config`)"; "`_STRICT`: the pydantic config shared by the package's modules"; add `integration` to the test list |
| D5 | Low, doc | plan §4.2, "Pruning" | "`MedianPruner`/`PercentilePruner` never did, whatever was reported": on the reconstruction they pruned the bad trial after 1–2 folds unless fold 0 was the easiest. The mechanism stated (a trial's best value over its steps) is right, and the guidance is unchanged | R, §3.6 | "…`MedianPruner`/`PercentilePruner` pruned it or not depending on the fold order (never when the tied fold 0 was the easiest, after 1–2 folds otherwise), because they compare a trial's best value over its steps…" |
| D6 | Low, doc | plan §4.1, second bullet | "a value error is prefixed `parameter '<name>'`": the choice-type error is not ("Value error, choices must be None, bool, int, float or str, got …", as guide §3's own output shows). It is a field validator, which runs before the dataclass exists | P, `P4.1-…`; guide §3 block 2 output | "Optuna's value errors and the duplicate-choice error are prefixed `parameter '<name>'` (the choice-type check, a field validator, is not)" |

**Notes (no change proposed):**
- **N1.** The corrected SE overstates for a stable learner too: 1.35× for a near-unpenalized
  `PoissonRegressor`, against 1.39× for a fixed predictor. It is close for LightGBM (1.09×). The docs
  call it an approximation and name only the fixed-predictor case. Not a contradiction.
- **N2.** Plan §7's command table quotes the suite as "1002 passed" (today: 1244 passed, 1 skipped,
  1 xfailed). Historical counts are not invariants, but the line could drop its count.
- **N3.** The evaluator's class docstring points its example at the plan's §5 rather than the user guide.
  Both are correct.

## 6. Appendix: the uncommitted `parameters.py` diff (not part of HEAD)

The working tree's `parameters.py` (+9/−1 against `f219cb8`):
- adds `if any(isinstance(choice, float) and not isfinite(choice) …): raise ValueError("float choices must be finite")`
  to `_scalar_choices`;
- replaces the pairwise duplicate check with `len(set(c)) != len(c)`, falling back to the pairwise `==`
  on `TypeError`.

It was checked in a second scratch copy (`git archive f219cb8` with the working-tree file copied in).

| Check | Observed |
|---|---|
| NaN, inf, −inf choices | rejected: "Value error, float choices must be finite" (no `parameter 'p':` prefix, like the choice-type error, D6). HEAD accepts all three (F1) |
| Duplicate detection vs HEAD's pairwise `==` | the same verdict on `[True, 1]`, `[1, 1.0]`, `[0.0, -0.0]`, `[None, None]`, `['a', 'b', 'a']` (all rejected) and `[None, 'a', 2, 2.5]` (accepted); `[nan, nan]` (one object) is now rejected, by the finite check, before the set |
| The `TypeError` fallback | reachable only by an unhashable `float`/`int`/`str` subclass, which passes `_scalar_choices` (`isinstance`): `class F(float): __hash__ = None`, `[F(1.0), F(1.0)]` is rejected via the fallback. No planned client builds one |
| Package tests on the diff copy | 128 passed, 3 deselected (non-slow, `-W error`); ruff and `ruff format --check` clean; mypy: no issues in 6 source files |
| Test coverage, by mutation | dropping the finite check: **128 passed** (no test covers it); replacing the set check with `False`: 2 failed (the duplicate-choice tests cover it) |
| Decision record | none: D18 does not mention non-finite floats |

**Reading:**
- The finite check closes F1, at the cost of one line and a test still to be written.
- The set-based check gives the same answers as the pairwise one on every scalar case above. It differs
  only in cost: O(K) instead of O(K²) for K choices, which is immaterial at K < 20.
- The `TypeError` fallback guards a case no client has. Under the rule "justify every check", it would
  be dropped.

## 7. Not verified

- **The plan's exact numbers for the SE and the pruning experiments.** Their setups are not recorded in
  the docs, the commit messages or the review. §3.5 and §3.6 are reconstructions, judged on the stated
  ratios and directions.
- **LightGBM thread effects on other platforms, compilers or LightGBM versions.** Only this machine
  (macOS, LightGBM 4.7.0) was measured.
- **The guide's §11 ("What remains")** is forward-looking and has nothing to check.
- **The PR #5 body** was not compared with the diff; that was not in this validation's scope.

## 8. Scripts

All scripts lived in the session's scratch directory (not committed).

**Layout:**
- `head/` is the `git archive f219cb8` copy of `src tests configs notebooks docs pyproject.toml uv.lock
  README.md .env.example`.
- `pristine/src` is the restore source.
- `diff/` is the appendix copy.

**Re-running:**
- **pytest:** go through `pt.sh <copy> <args>`. It runs the repository's `.venv` pytest with
  `-c <copy>/pyproject.toml --rootdir <copy> -p guard_plugin -p no:cacheprovider`.
- **Probes:**
  `cd <copy> && PYTHONPATH=<copy>/src /path/to/repo/.venv/bin/python <script> [args] | grep -v '^\[LightGBM\]'`.

<details><summary><code>guard_plugin.py</code></summary>

```python
"""Pytest plugin: abort unless the tuning package is imported from the scratch copy under test."""
import os


def pytest_sessionstart(session):
    import age_group_prediction.hyperparameter_tuning as m

    root = os.environ["VAL_ROOT"]
    assert m.__file__.startswith(root), f"wrong import path: {m.__file__} (want {root})"
    print(f"\n[guard] {m.__file__}")
```

</details>

<details><summary><code>pt.sh</code></summary>

```zsh
#!/bin/zsh
# Usage: pt.sh <copy dir> <pytest args...>; runs the repo's venv pytest inside the copy, guarded.
ROOT=$1; shift
cd $ROOT && VAL_ROOT=$ROOT PYTHONPATH=<scratch>/val/scripts \
  <repo>/.venv/bin/python -m pytest -c $ROOT/pyproject.toml --rootdir $ROOT -p guard_plugin -p no:cacheprovider "$@"
```

</details>

<details><summary><code>run_guide_blocks.py</code></summary>

```python
"""Run docs/HYPERPARAMETER_TUNING.md's python blocks in order, in one namespace, as written.

Each block's stdout is diffed against the ```text block that follows it; warnings are
recorded per block (not escalated), and each block is timed.
Usage: python run_guide_blocks.py <path to the guide>
"""

import contextlib
import difflib
import io
import re
import sys
import time
import warnings

text = open(sys.argv[1]).read()
blocks = re.findall(r"```(python|text)\n(.*?)```", text, flags=re.S)
pairs, pending = [], None
for kind, body in blocks:
    if kind == "python":
        if pending is not None:
            pairs.append((pending, None))
        pending = body
    else:
        pairs.append((pending, body))
        pending = None
if pending is not None:
    pairs.append((pending, None))

namespace: dict = {"__name__": "__guide__"}
all_ok = True
for i, (code, expected) in enumerate(pairs, 1):
    buffer = io.StringIO()
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(buffer):
        warnings.simplefilter("default")
        exec(compile(code, f"<block {i}>", "exec"), namespace)
    elapsed = time.perf_counter() - start
    out = buffer.getvalue()
    same = expected is not None and out == expected
    all_ok &= same or expected is None
    print(f"=== block {i}: {elapsed:.1f} s, output {'IDENTICAL' if same else 'DIFFERS' if expected else '(no text block)'}")
    if expected is not None and not same:
        sys.stdout.writelines(
            difflib.unified_diff(expected.splitlines(True), out.splitlines(True), "doc", "run")
        )
    for w in caught:
        print(f"    warning: {w.category.__name__}: {str(w.message)[:110]}")
print("ALL IDENTICAL" if all_ok else "SOME DIFFER", f"({len(pairs)} python blocks)")
```

</details>

<details><summary><code>probe_claims.py</code></summary>

```python
"""Probe each D-decision and each claim of the user guide / plan §4 against the code under test.

Prints one line per check: id | observed | PASS/FAIL. Read-only: nothing is written.
Usage (cwd = the scratch copy): PYTHONPATH=src python probe_claims.py
"""

from __future__ import annotations

import dataclasses
import inspect
import io
import json
import logging
import math
import warnings
from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from optuna.pruners import MedianPruner, NopPruner, SuccessiveHalvingPruner, WilcoxonPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.trial import FixedTrial, TrialState
from pydantic import ValidationError
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.model_selection import BaseCrossValidator, KFold

import age_group_prediction.hyperparameter_tuning as ht
from age_group_prediction.hyperparameter_tuning import (
    Aggregator,
    CategoricalParameter,
    CVHyperparameterEvaluator,
    FloatParameter,
    HyperparameterStudy,
    IntParameter,
    Mean,
    TrialRecord,
    TuningResult,
    UpperBound,
    WeightedMean,
    corrected_std_error,
)
from age_group_prediction.hyperparameter_tuning._config import _STRICT
from age_group_prediction.modeling import (
    BaseAgeGroupModel,
    CohortProbabilityModel,
    CountModel,
    IndependentCohortModels,
    TotalTimesProbabilityModel,
)
from age_group_prediction.preprocessing import ExposureTransformer
from age_group_prediction.scoring import COHORT_LOG_LOSS, POISSON_DEVIANCE, Metric, cohort_log_loss
from age_group_prediction.splitting import Splitter

optuna.logging.set_verbosity(optuna.logging.WARNING)
results: list[tuple[str, str, bool]] = []


def check(cid: str, fn) -> None:
    try:
        observed, ok = fn()
    except Exception as error:  # a probe that crashes is a FAIL with its error shown
        observed, ok = f"probe raised {type(error).__name__}: {error}", False
    observed = " ".join(str(observed).split())
    results.append((cid, observed, ok))
    print(f"{cid} | {observed[:300]} | {'PASS' if ok else 'FAIL'}")


def raises(fn, exc=Exception) -> tuple[bool, str]:
    try:
        fn()
    except exc as error:
        return True, f"{type(error).__name__}: {str(error).splitlines()[0][:140]}"
    return False, "no error"


def msg(fn) -> str:
    try:
        fn()
    except ValidationError as error:
        return error.errors()[0]["msg"]
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return "no error"


# --- toy data: rows in 4 strata, a positive count target
rng = np.random.default_rng(0)
GROUPS = np.repeat(np.arange(6), [3, 5, 8, 12, 6, 6])
N = len(GROUPS)
X = pd.DataFrame({"x": rng.normal(size=N), "z": rng.normal(size=N)})
Y = rng.poisson(np.exp(0.5 + 0.3 * X["x"].to_numpy())).astype(float)
EXPOSURE = rng.integers(1, 9, size=N).astype(float)
CV = Splitter("stratified_by_group").cv(n_splits=3, random_state=0)
GLM = CountModel(estimator=PoissonRegressor(alpha=1.0))


class Recorder(BaseAgeGroupModel):
    seen: list = []

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any, exposure: Any = None) -> Self:
        Recorder.seen.append(("fit", None if exposure is None else len(exposure)))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any, exposure: Any = None) -> np.ndarray:
        Recorder.seen.append(("predict", None if exposure is None else len(exposure)))
        return np.full(len(X), self.mean_)


def evaluator(**kw: Any) -> CVHyperparameterEvaluator:
    args: dict[str, Any] = {"cv": CV, "metric": POISSON_DEVIANCE} | kw
    model = args.pop("model", GLM)
    params = args.pop("parameters", [FloatParameter("estimator__alpha", 0.01, 10.0, log=True)])
    return CVHyperparameterEvaluator(model, params, **args)


# ===================== D-decisions =====================
def d1():
    frozen, _ = raises(lambda: setattr(FloatParameter("p", 0.0, 1.0), "low", 0.5))
    is_pyd = all(dataclasses.is_dataclass(c) and hasattr(c, "__pydantic_fields__")
                 for c in (FloatParameter, IntParameter, CategoricalParameter))
    strict = [msg(lambda: IntParameter("p", 1.5, 9)), msg(lambda: IntParameter("p", True, 9))]
    value = msg(lambda: FloatParameter("p", 1.0, 0.0))
    ok = frozen and is_pyd and all("valid integer" in s for s in strict) and "parameter 'p':" in value
    return f"frozen={frozen}; pydantic dataclasses={is_pyd}; 1.5/True for int: {strict}; low>high: {value!r}", ok


check("D1", d1)


def d3():
    out = []
    for m in ("random", "stratified_by_group", "grouped"):
        ok1, s1 = raises(lambda: Splitter(m).cv(n_splits=1, random_state=0))
        ok2, s2 = raises(lambda: Splitter(m).cv(n_splits=3, random_state=None))
        cv = Splitter(m).cv(n_splits=3, random_state=7)
        g = None if m == "random" else GROUPS
        same = all(np.array_equal(a[1], b[1]) for a, b in zip(cv.split(X, Y, g), cv.split(X, Y, g)))
        out.append((m, ok1, ok2, same))
    return f"(method, n_splits=1 raises, random_state=None raises, two split() calls identical): {out}", all(
        a and b and c for _, a, b, c in out)


check("D3", d3)


def d3_cv_check():
    # Guide §4: the evaluator checks that `cv` is "a Splitter.cv(n_splits=..., random_state=int) validator".
    ev = evaluator(cv=KFold(3, shuffle=True))  # seedless: folds differ per split() call
    a = [v.tolist() for _, v in ev.cv.split(X)]
    b = [v.tolist() for _, v in ev.cv.split(X)]
    rejected_str, s = raises(lambda: evaluator(cv="kfold"))
    return (f"KFold(3, shuffle=True) (no seed) accepted; its folds differ between calls: {a != b}; "
            f"a string cv rejected: {rejected_str} ({s[:60]})"), True  # observation, judged in the doc


check("D3-cv-type-only", d3_cv_check)


def d4():
    Recorder.seen.clear()
    ev = evaluator(model=Recorder(), parameters=[FloatParameter("alpha", 0.1, 1.0)], metric=Metric("mse", lambda a, b: float(np.mean((np.asarray(a) - b) ** 2))))
    folds1 = [v.tolist() for _, v in ev.cv.split(X, Y, GROUPS)]
    ev.evaluate(FixedTrial({"alpha": 0.2}), X, Y, GROUPS)
    folds2 = [v.tolist() for _, v in ev.cv.split(X, Y, GROUPS)]
    return f"folds identical before/after a trial: {folds1 == folds2}; fits per trial = {sum(1 for k, _ in Recorder.seen if k == 'fit')} (= n folds 3)", folds1 == folds2


check("D4", d4)


def d5():
    abstract = inspect.isabstract(Aggregator)
    default = evaluator().aggregation
    zbad = [raises(lambda z=z: UpperBound(z=z))[0] for z in (0.0, -1.0, math.nan, math.inf)]
    ub = UpperBound(z=2.0).aggregate([1.0, 2.0, 3.0], [1, 1, 1])
    hand = 2.0 + 2.0 * float(np.std([1, 2, 3], ddof=1)) * math.sqrt(1 / 3 + 1 / 2)
    return f"Aggregator abstract={abstract}; evaluator default={default!r}; z in (0,-1,nan,inf) rejected={zbad}; UpperBound(2) {ub:.6f} vs hand {hand:.6f}", (
        abstract and default == WeightedMean() and all(zbad) and math.isclose(ub, hand))


check("D5", d5)


def d7():
    st = HyperparameterStudy(seed=3)
    s = st._create_optuna_study()
    sampler, pruner = s.sampler, s.pruner
    given_s, given_p = RandomSampler(seed=1), MedianPruner()
    s2 = HyperparameterStudy(seed=3, sampler=given_s, pruner=given_p)._create_optuna_study()
    return (f"default sampler {type(sampler).__name__} multivariate={sampler._multivariate}; default pruner {type(pruner).__name__}; "
            f"given used by identity: {s2.sampler is given_s and s2.pruner is given_p}"), (
        isinstance(sampler, TPESampler) and sampler._multivariate and isinstance(pruner, NopPruner)
        and s2.sampler is given_s and s2.pruner is given_p)


check("D7", d7)


def d8():
    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        if trial.number == 1:
            trial.report(-100.0, step=0)
            raise optuna.TrialPruned()
        return x

    r = HyperparameterStudy(seed=0, n_trials=4).optimize(objective)
    pruned = [t for t in r.trials if t.state == TrialState.PRUNED]
    return f"best_trial_number={r.best_trial_number}; pruned record value={pruned[0].value}", (
        r.best_trial_number != 1 and pruned[0].value is None)


check("D8", d8)


def d9():
    subclasses = sorted(c.__name__ for c in ht.Parameter.__subclasses__())
    return f"Parameter subclasses: {subclasses}; no conditional API", subclasses == ["CategoricalParameter", "FloatParameter", "IntParameter"]


check("D9", d9)


def d10():
    sig = str(inspect.signature(CVHyperparameterEvaluator.evaluate))
    frozen, _ = raises(lambda: setattr(evaluator(), "metric", POISSON_DEVIANCE))
    fields = [f.name for f in dataclasses.fields(CVHyperparameterEvaluator)]
    return f"evaluate{sig}; frozen={frozen}; fields={fields}", (
        "(self, trial: 'BaseTrial', X: 'DesignMatrix', y: 'Target', groups: 'Groups | None' = None, *, exposure: 'Exposure | None' = None) -> 'float'" == sig
        and frozen and "X" not in fields)


check("D10/D15", d10)


def d11():
    ev = evaluator()
    m = ev.build_model({"estimator__alpha": 0.5})
    copy = m is not ev.model and m.estimator is not ev.model.estimator
    unfitted = not hasattr(ev.model, "estimator_")
    bad, s = raises(lambda: ev.build_model({"estimator__alpah": 0.5}))
    return f"copy={copy}; alpha={m.estimator.alpha}; template alpha={ev.model.estimator.alpha}; misspelt raises: {s}", (
        copy and m.estimator.alpha == 0.5 and ev.model.estimator.alpha == 1.0 and unfitted and bad)


check("D11", d11)


def d12():
    r2 = Metric("r2ish", lambda a, b: 0.7, greater_is_better=True)
    ev = evaluator(metric=r2)
    st = HyperparameterStudy(seed=0, n_trials=2)
    res = st.optimize(lambda t: ev.evaluate(t, X, Y, GROUPS))
    direction = st.optuna_study_.direction.name
    return (f"direction={direction}; optuna value={st.optuna_study_.best_value}; best_value={res.best_value}; "
            f"record values={[t.value for t in res.trials]}; fold_values={res.trials[0].fold_values}; fields of TrialRecord={[f.name for f in dataclasses.fields(TrialRecord)]}"), (
        direction == "MINIMIZE" and math.isclose(st.optuna_study_.best_value, -0.7) and math.isclose(res.best_value, 0.7)
        and all(math.isclose(v, 0.7) for v in res.trials[0].fold_values)
        and not any(f.name in ("score", "fold_scores", "metric_name") for f in dataclasses.fields(TrialRecord)))


check("D12", d12)


def d13():
    fields = [f.name for f in dataclasses.fields(CVHyperparameterEvaluator)]
    return f"evaluator fields={fields}", "feature_transformer" not in fields


check("D13 (superseded)", d13)


def d14():
    Recorder.seen.clear()
    ev = evaluator(model=Recorder(), parameters=[], metric=Metric("mse", lambda a, b: 0.0))
    ev.evaluate(FixedTrial({}), X, Y, GROUPS)
    none_seen = {e for _, e in Recorder.seen}
    Recorder.seen.clear()
    ev.evaluate(FixedTrial({}), X, Y, GROUPS, exposure=EXPOSURE)
    sizes = [e for _, e in Recorder.seen]
    folds = [(len(tr), len(va)) for tr, va in CV.split(X, Y, GROUPS)]
    expected = [n for tr, va in folds for n in (tr, va)]
    return f"no exposure -> model got {none_seen}; with exposure, lengths fit/predict {sizes} vs folds {expected}", (
        none_seen == {None} and sizes == expected)


check("D14", d14)


def d16():
    import age_group_prediction.scoring as sc
    return f"Metric module={Metric.__module__}; POISSON_DEVIANCE in scoring: {hasattr(sc, 'POISSON_DEVIANCE')}", Metric.__module__ == "age_group_prediction.scoring"


check("D16", d16)


def d17():
    bad, s = raises(lambda: CVHyperparameterEvaluator(Ridge(), [], cv=CV, metric=POISSON_DEVIANCE))
    return f"an sklearn Ridge as model rejected: {bad} ({s[:90]})", bad


check("D17", d17)


def d2_d6():
    bad, s = raises(lambda: evaluator(metric="neg_mean_poisson_deviance"))
    return f"an sklearn scorer string as metric rejected: {bad} ({s[:80]})", bad


check("D2/D6 (superseded)", d2_d6)


def d18():
    rejected = {}
    for name, choices in {"list": [[1, 2]], "tuple": [(1, 2)], "object": [object()], "np.int64": [np.int64(3)]}.items():
        rejected[name] = raises(lambda c=choices: CategoricalParameter("p", c))[0]
    f = np.float64(0.5)
    p = CategoricalParameter("p", [f, 1.0])
    kept = p.choices[0] is f
    set_rejected = raises(lambda: CategoricalParameter("p", {1, 2}))[0]
    as_tuple = isinstance(CategoricalParameter("p", [1, 2]).choices, tuple)
    return f"rejected={rejected}; np.float64 kept by identity={kept}; set rejected={set_rejected}; list stored as tuple={as_tuple}", (
        all(rejected.values()) and kept and set_rejected and as_tuple)


check("D18", d18)


def d19():
    fields = [f.name for f in dataclasses.fields(HyperparameterStudy)]
    rej = {k: raises(lambda k=k: HyperparameterStudy(seed=0, **{k: 1}))[0] for k in ("timeout_seconds", "n_jobs", "initial_params")}
    no_attr = not any(hasattr(TuningResult, a) for a in ("is_reproducible",)) and "is_reproducible" not in [f.name for f in dataclasses.fields(TuningResult)]
    return f"fields={fields}; rejected as extra={rej}; TuningResult has no is_reproducible={no_attr}", (
        fields == ["seed", "n_trials", "sampler", "pruner"] and all(rej.values()) and no_attr)


check("D19", d19)


def d20():
    ev = evaluator(parameters=[])
    st = HyperparameterStudy(seed=0, n_trials=3)
    res = st.optimize(lambda t: ev.evaluate(t, X, Y, GROUPS))
    values = [t.value for t in res.trials]
    return f"parameters=[] built; 3 trials values={values}; best_params={res.best_params}", (
        len(set(values)) == 1 and res.best_params == {})


check("D20", d20)


# ===================== guide / plan claims =====================
def g_root_export():
    import age_group_prediction
    src = inspect.getsource(age_group_prediction)
    return f"'hyperparameter_tuning' in the root __init__: {'hyperparameter_tuning' in src}", "hyperparameter_tuning" not in src


check("G1-not-reexported", g_root_export)


def g_param_table():
    cases = {
        "Float log low=0": lambda: FloatParameter("p", 0.0, 1.0, log=True),
        "Float log+step": lambda: FloatParameter("p", 0.1, 1.0, log=True, step=0.1),
        "Int log low=0": lambda: IntParameter("p", 0, 8, log=True),
        "Int log step=2": lambda: IntParameter("p", 1, 9, log=True, step=2),
        "Int step not dividing": lambda: IntParameter("p", 0, 9, step=2),
        "Float positional log": lambda: FloatParameter("p", 0.1, 1.0, True),
        "empty choices": lambda: CategoricalParameter("p", []),
    }
    out = {k: raises(v)[0] for k, v in cases.items()}
    ok_float_int_bounds = FloatParameter("p", 1, 2).low == 1
    return f"rejected={out}; int bounds accepted by Float={ok_float_int_bounds}", all(out.values()) and ok_float_int_bounds


check("G3-table", g_param_table)


def g_optuna_lets_through():
    # The three rules are claimed to be things Optuna itself lets through silently.
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = FloatDistribution(0.0, 1.0, step=0.3)
    step_silent = f"Optuna: high became {d.high}, warned={[str(x.category.__name__) for x in w]}"
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        CategoricalDistribution(("a", "a"))
    s = optuna.create_study(sampler=RandomSampler(seed=0))
    t = s.ask()
    a, b = t.suggest_float("dup", 0, 1), t.suggest_float("dup", 0, 1)
    dup_names = raises(lambda: evaluator(parameters=[FloatParameter("q", 0, 1), FloatParameter("q", 0, 2)]))
    return f"{step_silent}; duplicate choices warned={len(w2)}; repeated name returns cached value={a == b}; evaluator names repeated rejected: {dup_names[1]}", (
        math.isclose(d.high, 0.9) and len(w2) == 0 and a == b and dup_names[0])


check("G3-three-rules", g_optuna_lets_through)


def g_type_error_names_argument():
    m1 = msg(lambda: FloatParameter("p", "a", 1.0))
    try:
        FloatParameter("p", "a", 1.0)
    except ValidationError as e:
        loc = e.errors()[0]["loc"]
        title = e.title
    return f"error title={title}, loc={loc}, msg={m1}", title == "FloatParameter" and loc in ((1,), ("low",))


check("P4.1-type-error-names-class-and-arg", g_type_error_names_argument)


def g_evaluator_settings():
    cases = {
        "string metric": lambda: evaluator(metric="poisson_deviance"),
        "dict metric": lambda: evaluator(metric={"name": "m", "function": len}),
        "string aggregation": lambda: evaluator(aggregation="mean"),
        "set of parameters": lambda: evaluator(parameters={FloatParameter("a", 0, 1)}),
        "misspelt keyword": lambda: CVHyperparameterEvaluator(GLM, [], cv=CV, metric=POISSON_DEVIANCE, aggregaton=Mean()),
    }
    out = {k: raises(v)[0] for k, v in cases.items()}
    stored = evaluator(parameters=[FloatParameter("estimator__alpha", 0.1, 1.0)]).parameters
    return f"rejected at construction={out}; parameters stored as {type(stored).__name__}", all(out.values()) and isinstance(stored, tuple)


check("G4/P4.2-settings-checked", g_evaluator_settings)


def g_exposure_shape():
    ev = evaluator()
    longer = raises(lambda: ev.evaluate(FixedTrial({"estimator__alpha": 1.0}), X, Y, GROUPS, exposure=np.ones(N + 1)))
    two_d = raises(lambda: ev.evaluate(FixedTrial({"estimator__alpha": 1.0}), X, Y, GROUPS, exposure=np.ones((N, 1))))
    return f"longer: {longer[1][:90]}; (n,1): {two_d[0]}", longer[0] and two_d[0]


check("G4-exposure-shape", g_exposure_shape)


def g_groups_random():
    ev = evaluator(cv=Splitter("random").cv(n_splits=3, random_state=0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ev.evaluate(FixedTrial({"estimator__alpha": 1.0}), X, Y, GROUPS)
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        ev.evaluate(FixedTrial({"estimator__alpha": 1.0}), X, Y, None)
    return f"random with groups: {len(w)} warnings ({str(w[0].message)[:70] if w else ''}); with None: {len(w2)}", len(w) > 0 and len(w2) == 0


check("G4-groups-None-for-random", g_groups_random)


def g_numpy_X():
    from age_group_prediction.feature_engineering import ColumnPlan, FeatureTransformer
    tree = FeatureTransformer(plans=(ColumnPlan(name="n", columns=("x", "z")),))
    ev = evaluator(model=CountModel(estimator=PoissonRegressor(), feature_transformer=tree), parameters=[])
    a = raises(lambda: ev.evaluate(FixedTrial({}), X.to_numpy(), Y, GROUPS))
    b = evaluator(parameters=[]).evaluate(FixedTrial({}), X.to_numpy(), Y, GROUPS)
    return f"numpy X with a feature_transformer: {a[1][:80]}; without one: {b:.4f}", a[0] and math.isfinite(b)


check("G4-numpy-X", g_numpy_X)


def g_user_attrs_and_sqlite():
    ev = evaluator()
    s = optuna.create_study(storage="sqlite:///:memory:", sampler=RandomSampler(seed=0))
    s.optimize(lambda t: ev.evaluate(t, X, Y, GROUPS), n_trials=1)
    attrs = s.trials[0].user_attrs
    types = {k: type(v).__name__ for k, v in attrs.items()}
    js = json.loads(json.dumps(attrs, allow_nan=False)) == attrs
    return f"attrs={sorted(attrs)}; types={types}; sqlite round-trip ok; json exact={js}", (
        sorted(attrs) == ["fold_scores", "fold_sizes", "greater_is_better", "metric_name"] and js)


check("G4/P4.2-user-attrs", g_user_attrs_and_sqlite)


def g_nan_score():
    nan_metric = Metric("nan", lambda a, b: math.nan)
    ev = evaluator(metric=nan_metric, parameters=[])
    st = HyperparameterStudy(seed=0, n_trials=3)
    stopped = raises(lambda: st.optimize(lambda t: ev.evaluate(t, X, Y, GROUPS)), ValueError)
    states = [t.state.name for t in st.optuna_study_.trials]
    st2 = HyperparameterStudy(seed=0, n_trials=3)
    calls = iter([math.nan, 1.0, None])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = st2.optimize(lambda t: (t.suggest_float("x", 0, 1), next(calls))[1])
    return (f"evaluator NaN: stops={stopped[0]} ({stopped[1][:70]}), states={states}; plain NaN/None: states={[t.state.name for t in r.trials]}, "
            f"python warnings={len(w)}"), (stopped[0] and states == ["FAIL"] and [t.state.name for t in r.trials] == ["FAIL", "COMPLETE", "FAIL"] and len(w) == 0)


check("G10/P4.4-NaN", g_nan_score)


def g_poisson_multioutput():
    a = raises(lambda: POISSON_DEVIANCE.function(np.ones((4, 2)), np.ones((4, 2))))
    return a[1][:80], a[0] and "Multioutput" in a[1]


check("G10-one-column-metric", g_poisson_multioutput)


def g_exposure_transformer():
    t = pd.DataFrame({"n_apartments": [1.0, 0.0, 3.0]})
    z = raises(lambda: ExposureTransformer("n_apartments").fit_transform(t))
    t2 = pd.DataFrame({"n_apartments": [1.0, np.nan, 3.0]})
    n = raises(lambda: ExposureTransformer("n_apartments").fit_transform(t2))
    return f"zero rejected={z[0]}; NaN rejected={n[0]}", z[0] and n[0]


check("G10-exposure-transformer", g_exposure_transformer)


def g_study_settings():
    cases = {
        "n_trials=0": lambda: HyperparameterStudy(seed=0, n_trials=0),
        "seed=True": lambda: HyperparameterStudy(seed=True),
        "n_trials=2.0": lambda: HyperparameterStudy(seed=0, n_trials=2.0),
        "string sampler": lambda: HyperparameterStudy(seed=0, sampler="tpe"),
        "dict sampler": lambda: HyperparameterStudy(seed=0, sampler={}),
        "misspelt n_trial": lambda: HyperparameterStudy(seed=0, n_trial=10),
        "positional seed": lambda: HyperparameterStudy(0),
    }
    out = {k: raises(v)[0] for k, v in cases.items()}
    s = optuna.create_study()
    s.optimize(lambda t: 0.0, n_trials=0)
    neg = raises(lambda: HyperparameterStudy(seed=-1, n_trials=1).optimize(lambda t: t.suggest_float("x", 0, 1)))
    return f"rejected={out}; optuna n_trials=0 ran {len(s.trials)} trials silently; negative seed: {neg[1][:80]}", (
        all(out.values()) and len(s.trials) == 0 and neg[0])


check("G6/P4.4-constructor", g_study_settings)


def g_default_sampler_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TPESampler(seed=0, multivariate=True)
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        HyperparameterStudy(seed=0, n_trials=3).optimize(lambda t: t.suggest_float("x", 0, 1))
    return f"raw TPESampler(multivariate) warns {[x.category.__name__ for x in w]}; study.optimize warns {len(w2)}", len(w) == 1 and len(w2) == 0


check("P4.4-sampler-warning-silenced", g_default_sampler_warning)


def g_pruners_experimental():
    out = {}
    for cls in (WilcoxonPruner, SuccessiveHalvingPruner, MedianPruner):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls()
        out[cls.__name__] = [x.category.__name__ for x in w]
    return f"warnings at construction: {out}", bool(out["WilcoxonPruner"]) and bool(out["SuccessiveHalvingPruner"])


check("G7-both-pruners-warn", g_pruners_experimental)


def g_optimize_behaviour():
    st = HyperparameterStudy(seed=0, n_trials=5)
    boom = raises(lambda: st.optimize(lambda t: (t.suggest_float("x", 0, 1), 1 / 0)[1]), ZeroDivisionError)
    kept = hasattr(st, "optuna_study_") and len(st.optuna_study_.trials) == 1 and st.optuna_study_.trials[0].state == TrialState.FAIL
    st2 = HyperparameterStudy(seed=0, n_trials=3)

    def all_pruned(t):
        t.suggest_float("x", 0, 1)
        raise optuna.TrialPruned()

    none = raises(lambda: st2.optimize(all_pruned), RuntimeError)
    st3 = HyperparameterStudy(seed=0, n_trials=6)
    r = st3.optimize(lambda t: round(t.suggest_float("x", 0, 1)) * 0 + (1.0 if t.number in (2, 4) else 5.0))
    tie = r.best_trial_number
    optuna_tie = st3.optuna_study_.best_trial.number
    r.best_params["x"] = 99
    intact = st3.optuna_study_.trials[2].params["x"] != 99
    return (f"error propagates={boom[0]} and study kept with 1 FAIL trial={kept}; all pruned: {none[1][:60]}; tie (trials 2,4) -> {tie}, "
            f"Optuna's best_trial -> {optuna_tie}; editing best_params leaves study intact={intact}"), (
        boom[0] and kept and none[0] and tie == 2 and optuna_tie == 2 and intact)


check("G6/P4.4-optimize", g_optimize_behaviour)


def g_record_states_and_json():
    def objective(t):
        x = t.suggest_float("x", 0, 1)
        if t.number == 1:
            t.report(100.0, 0)
            raise optuna.TrialPruned()
        if t.number == 2:
            return math.nan
        return x

    st = HyperparameterStudy(seed=0, n_trials=4)
    r = st.optimize(objective)
    pruned_optuna_value = st.optuna_study_.trials[1].value
    recs = [(t.number, t.state.name, t.value, t.fold_values, t.std_error) for t in r.trials]
    d = r.to_dict()
    rt = json.loads(json.dumps(d)) == d
    state_names = [t["state"] for t in d["trials"]]
    plain_none = r.metric_name is None and all(t.fold_values is None for t in r.trials)
    inf = HyperparameterStudy(seed=0, n_trials=1).optimize(lambda t: (t.suggest_float("x", 0, 1), math.inf)[1])
    inf_json = raises(lambda: json.dumps(inf.to_dict(), allow_nan=False))
    return (f"records={recs}; Optuna value of the pruned trial={pruned_optuna_value}; to_dict round-trips={rt}; states={state_names}; "
            f"plain objective metric_name None & folds None={plain_none}; inf best_value={inf.best_value}, json allow_nan=False raises={inf_json[0]}"), (
        rt and state_names == ["COMPLETE", "PRUNED", "FAIL", "COMPLETE"] and pruned_optuna_value == 100.0
        and recs[1][2] is None and plain_none and inf_json[0])


check("G6/P4.4-records", g_record_states_and_json)


def g_fixed_trial_same_folds():
    ev = evaluator()
    st = HyperparameterStudy(seed=0, n_trials=3)
    r = st.optimize(lambda t: ev.evaluate(t, X, Y, GROUPS))
    again = ev.evaluate(FixedTrial(r.best_params), X, Y, GROUPS)
    return f"FixedTrial(best_params)={again!r} vs best_value={r.best_value!r}", again == r.best_value


check("G4-FixedTrial", g_fixed_trial_same_folds)


def g_one_fold_custom_cv():
    class OneFold(BaseCrossValidator):
        def _iter_test_indices(self, X=None, y=None, groups=None):
            yield np.arange(5)

        def get_n_splits(self, X=None, y=None, groups=None):
            return 1

    ev = evaluator(cv=OneFold())
    value = ev.evaluate(FixedTrial({"estimator__alpha": 1.0}), X, Y, GROUPS)
    st = HyperparameterStudy(seed=0, n_trials=2)
    after = raises(lambda: st.optimize(lambda t: ev.evaluate(t, X, Y, GROUPS)))
    n_ran = len(st.optuna_study_.trials)
    return f"evaluate alone returns {value:.4f}; study: {after[1][:80]} after {n_ran} trials ran", after[0] and n_ran == 2


check("P4.2-one-fold-cv-fails-after-search", g_one_fold_custom_cv)


def g_composites():
    icm = IndependentCohortModels({"a": GLM})
    keys = sorted(icm.get_params(deep=True))
    fwd = raises(lambda: icm.set_params(**{"cohort_models__a__estimator__alpha": 2.0}))
    ttp = TotalTimesProbabilityModel(total_model=GLM, probability_model=CohortProbabilityModel(estimator=__import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression()))
    ttp.set_params(total_model__estimator__alpha=0.3, probability_model__calibration_method=None)
    return f"IndependentCohortModels deep params={keys}; nested set raises={fwd[0]} ({fwd[1][:60]}); TTP nested set ok, alpha={ttp.total_model.estimator.alpha}", (
        fwd[0] and ttp.total_model.estimator.alpha == 0.3)


check("G9-composites", g_composites)


def g_pydantic_or_dataclass():
    std = [not hasattr(c, "__pydantic_fields__") and dataclasses.is_dataclass(c) for c in (TrialRecord, TuningResult)]
    frozen = [c.__dataclass_params__.frozen for c in (TrialRecord, TuningResult)]
    agg_plain = not dataclasses.is_dataclass(Aggregator)
    study_frozen = HyperparameterStudy.__dataclass_params__.frozen
    return f"_STRICT={dict(_STRICT)}; TrialRecord/TuningResult stdlib dataclasses={std}, frozen={frozen}; Aggregator plain ABC={agg_plain}; HyperparameterStudy frozen={study_frozen}", (
        dict(_STRICT) == {"strict": True, "allow_inf_nan": False, "extra": "forbid"} and all(std) and all(frozen) and agg_plain and not study_frozen)


check("P4.0-pydantic-or-dataclass", g_pydantic_or_dataclass)


def g_aggregation_errors():
    out = {}
    for agg in (WeightedMean(), Mean(), UpperBound()):
        out[type(agg).__name__] = (raises(lambda a=agg: a.aggregate([], []))[1][:40], raises(lambda a=agg: a.aggregate([1.0], [1, 2]))[1][:40])
    se1 = raises(lambda: corrected_std_error([1.0]))
    eq = WeightedMean() == WeightedMean() and repr(WeightedMean()) == "WeightedMean()"
    return f"{out}; corrected_std_error([1.0]): {se1[1][:60]}; compare by value / repr={eq}", se1[0] and eq


check("P4.3-aggregate-errors", g_aggregation_errors)


def g_cohort_log_loss_weighting():
    # P4.3: COHORT_LOG_LOSS is a mean per child, so the size-weighted mean of fold losses is close to,
    # but not equal to, the pooled loss.
    y = np.array([[1, 0, 2], [0, 3, 0], [4, 1, 1], [0, 0, 1], [2, 2, 2], [1, 0, 0]], dtype=float)
    p = np.tile([[0.3, 0.3, 0.4]], (6, 1))
    folds = [np.arange(0, 2), np.arange(2, 6)]
    fold_losses = [cohort_log_loss(y[f], p[f]) for f in folds]
    weighted = WeightedMean().aggregate(fold_losses, [len(f) for f in folds])
    pooled = cohort_log_loss(y, p)
    return f"weighted mean of fold losses {weighted:.5f} vs pooled {pooled:.5f}", not math.isclose(weighted, pooled) and abs(weighted - pooled) / pooled < 0.1


check("P4.3-cohort-log-loss-weighting", g_cohort_log_loss_weighting)


def g_weighted_mean_pooled():
    folds = list(CV.split(X, Y, GROUPS))
    pred = np.full(N, Y.mean())
    fold = [POISSON_DEVIANCE.function(Y[v], pred[v]) for _, v in folds]
    w = WeightedMean().aggregate(fold, [len(v) for _, v in folds])
    idx = np.concatenate([v for _, v in folds])
    pooled = POISSON_DEVIANCE.function(Y[idx], pred[idx])
    return f"weighted mean of fold deviances {w!r} vs pooled {pooled!r}", math.isclose(w, pooled, rel_tol=1e-12)


check("G5/P4.3-weighted-mean-is-pooled", g_weighted_mean_pooled)

print()
print(f"{sum(ok for _, _, ok in results)} PASS, {sum(not ok for _, _, ok in results)} FAIL of {len(results)}")
```

</details>

<details><summary><code>mutate.py</code></summary>

```python
"""One scratch mutation per guarantee of the tuning plan's §4.4 test table.

For each mutation: every edit must match exactly once in the scratch copy; run the package tests
(non-slow, -W error) through pt.sh (guarded import path); record the failed tests; require the
named test(s) among them; restore src from the pristine copy and compare (filecmp, byte for byte).
Usage: python mutate.py <scratch val dir>
"""

import filecmp
import re
import shutil
import subprocess
import sys
from pathlib import Path

VAL = Path(sys.argv[1])
HEAD, PRISTINE = VAL / "head", VAL / "pristine"
PKG = "src/age_group_prediction/hyperparameter_tuning"
S, E, P, C = f"{PKG}/study.py", f"{PKG}/evaluator.py", f"{PKG}/parameters.py", f"{PKG}/_config.py"

# (id, description, [(file, old, new)], named tests)
MUTATIONS = [
    ("M01", "drop seed= (an unseeded default sampler)", [(S, "TPESampler(seed=self.seed, multivariate=True)", "TPESampler(multivariate=True)")],
     ["test_the_default_sampler_is_seeded_by_the_study", "test_the_same_seed_gives_the_same_result"]),
    ("M02", "drop multivariate=True", [(S, "TPESampler(seed=self.seed, multivariate=True)", "TPESampler(seed=self.seed)")],
     ["test_the_default_sampler_is_multivariate"]),
    ("M03", 'direction="maximize"', [(S, 'direction="minimize"', 'direction="maximize"')],
     ["test_the_sampler_searches_toward_lower_scores"]),
    ("M04", "another default pruner", [(S, "NopPruner() if self.pruner is None", "optuna.pruners.MedianPruner() if self.pruner is None")],
     ["test_the_default_pruner_never_prunes"]),
    ("M05", "copy the given sampler and pruner", [(S, "sampler = self.sampler", "sampler = __import__('copy').deepcopy(self.sampler)"),
                                                  (S, "pruner = NopPruner() if self.pruner is None else self.pruner", "pruner = NopPruner() if self.pruner is None else __import__('copy').deepcopy(self.pruner)")],
     ["test_a_given_sampler_and_pruner_are_used_as_given"]),
    ("M06", "remove the warning filter", [(S, 'warnings.filterwarnings("ignore", message="Argument ``multivariate``")', "pass")],
     ["test_the_default_sampler_does_not_warn"]),
    ("M07", "n_trials check < 0", [(S, "if self.n_trials < 1:", "if self.n_trials < 0:")],
     ["test_fewer_than_one_trial_is_rejected"]),
    ("M08", 'drop extra="forbid"', [(C, ', extra="forbid")', ")")],
     ["test_a_misspelt_setting_is_rejected"]),
    ("M09", "drop _STRICT from the study", [(S, "config=ConfigDict(**_STRICT, arbitrary_types_allowed=True)", "config=ConfigDict(arbitrary_types_allowed=True)")],
     ["test_a_bool_seed_is_rejected"]),
    ("M10", "seed=0 hard-coded (review D1)", [(S, "TPESampler(seed=self.seed, multivariate=True)", "TPESampler(seed=0, multivariate=True)")],
     ["test_the_same_seed_gives_the_same_result"]),
    ("M11", "best_params from another trial (review D2)", [(S, "best_params=dict(best.params)", "best_params=dict(complete[0].params)")],
     ["test_the_best_params_are_the_best_trials"]),
    ("M12", "best_value from another trial (review D2)", [(S, "best_value=_sign(best) * best.values[0]", "best_value=_sign(best) * complete[0].values[0]")],
     ["test_the_best_params_are_the_best_trials"]),
    ("M13", "include PRUNED among the candidates", [(S, "states=(TrialState.COMPLETE,))", "states=(TrialState.COMPLETE, TrialState.PRUNED))")],
     ["test_a_pruned_trial_never_wins"]),
    ("M14", "drop the no-complete check", [(S, "if len(complete) == 0:", "if False:")],
     ["test_no_completed_trial_raises"]),
    ("M15", "don't keep optuna_study_", [(S, "self.optuna_study_ = study", "pass")],
     ["test_the_optuna_study_is_kept"]),
    ("M16", "best_params=best.params (no copy)", [(S, "best_params=dict(best.params)", "best_params=best.params")],
     ["test_editing_the_result_leaves_the_study_intact"]),
    ("M17", "the naive s/sqrt(K)", [(S, "else corrected_std_error(fold_scores)", "else float(__import__('numpy').std(fold_scores, ddof=1)) / len(fold_scores) ** 0.5")],
     ["test_a_complete_record_carries_its_folds_and_their_std_error"]),
    ("M18", "value=trial.value for a pruned record", [(S, "dict(trial.params), *(None,) * 4)", "dict(trial.params), trial.value, None, None, None)")],
     ["test_a_pruned_record_reports_no_value"]),
    ("M19", "record COMPLETE and PRUNED only", [(S, "states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),", "states=(TrialState.COMPLETE, TrialState.PRUNED),")],
     ["test_a_failed_trial_is_recorded"]),
    ("M20", "params=trial.params (no copy)", [(S, "params=dict(trial.params),", "params=trial.params,")],
     ["test_editing_a_record_leaves_the_study_intact"]),
    ("M21", "a NumPy value in the dict", [(S, '"best_trial_number": self.best_trial_number,', '"best_trial_number": __import__("numpy").int64(self.best_trial_number),')],
     ["test_to_dict_round_trips_through_json"]),
    ("M22", "keep the TrialState in the dict", [(S, 'as_dict["state"] = record.state.name', 'as_dict["state"] = record.state')],
     ["test_to_dict_names_the_state"]),
    ("M23", "the sign condition reversed", [(S, 'return -1.0 if trial.user_attrs.get("greater_is_better", False) else 1.0', 'return -1.0 if not trial.user_attrs.get("greater_is_better", False) else 1.0')],
     ["test_a_loss_is_reported_as_it_is"]),
    ("M24", "value=trial.values[0]; best_value=best.values[0]", [(S, "value=sign * trial.values[0],", "value=trial.values[0],"),
                                                                 (S, "best_value=_sign(best) * best.values[0],", "best_value=best.values[0],")],
     ["test_a_greater_is_better_metric_is_reported_in_its_own_sign"]),
    ("M25", "one sign for every trial (the best's)", [(S, "trials=tuple(_trial_record(trial) for trial in ran),", "trials=tuple(_trial_record(trial, best) for trial in ran),"),
                                                      (S, "def _trial_record(trial: FrozenTrial) -> TrialRecord:", "def _trial_record(trial: FrozenTrial, best: FrozenTrial | None = None) -> TrialRecord:"),
                                                      (S, "sign = _sign(trial)", "sign = _sign(best or trial)")],
     ["test_each_trial_is_reported_in_its_own_metric_sign"]),
    ("M26", "default greater_is_better=True", [(S, 'trial.user_attrs.get("greater_is_better", False)', 'trial.user_attrs.get("greater_is_better", True)')],
     ["test_a_plain_objective_is_reported_unsigned"]),
    ("M27", "hard-code the greater_is_better attr True", [(E, 'trial.set_user_attr("greater_is_better", self.metric.greater_is_better)', 'trial.set_user_attr("greater_is_better", True)')],
     ["test_a_completed_trial_records_its_folds"]),
    ("M28", "Field(min_length=1) back", [(E, "parameters: Annotated[Sequence[Parameter], AfterValidator(_unique_names)]", "parameters: Annotated[Sequence[Parameter], __import__('pydantic').Field(min_length=1), AfterValidator(_unique_names)]")],
     ["test_an_empty_search_space_scores_the_model_as_given"]),
    ("M29", "drop the choice validator", [(P, "        AfterValidator(_scalar_choices),\n", "")],
     ["test_a_choice_outside_optunas_types_is_rejected"]),
    ("M30", "convert float choices instead of keeping them", [(P, "    return choices\n", "    return tuple(float(c) if isinstance(c, float) else c for c in choices)\n")],
     ["test_choices_are_stored_as_given"]),
]

TESTS = sorted(str(p.relative_to(HEAD)) for p in (HEAD / "tests/unit").glob("test_hyperparameter_tuning_*.py"))


def restore() -> None:
    shutil.rmtree(HEAD / "src")
    shutil.copytree(PRISTINE / "src", HEAD / "src")
    cmp = filecmp.dircmp(PRISTINE / "src", HEAD / "src")
    assert not cmp.diff_files and not cmp.left_only and not cmp.right_only, "restore failed"
    for f in (S, E, P, C):
        assert filecmp.cmp(PRISTINE / f, HEAD / f, shallow=False), f"{f} differs after restore"


def run_tests() -> tuple[list[str], str]:
    out = subprocess.run([str(VAL / "scripts/pt.sh"), str(HEAD), *TESTS, "-q", "-W", "error", "-m", "not slow", "-rf", "-p", "no:randomly"],
                         capture_output=True, text=True).stdout
    assert "[guard] " + str(HEAD) in out, "guard line missing"
    failed = sorted({re.sub(r"\[.*", "", m) for m in re.findall(r"^FAILED \S+::(\S+)", out, flags=re.M)})
    return failed, out.strip().splitlines()[-1]


restore()
failed, summary = run_tests()
print(f"BASELINE | {summary} | failed={failed}")
assert not failed
rows = []
for mid, desc, edits, named in MUTATIONS:
    for file, old, new in edits:
        path = HEAD / file
        text = path.read_text()
        assert text.count(old) == 1, f"{mid}: {old!r} matches {text.count(old)} times in {file}"
        path.write_text(text.replace(old, new))
    failed, summary = run_tests()
    killed = all(n in failed for n in named)
    restore()
    rows.append((mid, desc, named, failed, killed))
    print(f"{mid} | {desc} | named={named} | {'KILLED' if killed else 'SURVIVED'} | {len(failed)} failed: {failed} | {summary}")
print(f"\n{sum(r[4] for r in rows)} killed, {sum(not r[4] for r in rows)} survived of {len(rows)}; src restored and compared after each")
```

</details>

<details><summary><code>probe_worked_example.py</code></summary>

```python
"""The worked example (guide block 1, run as written) checked against a hand-written fold loop,
the NB2 one-trial client by hand, and reproducibility (seeds, LightGBM threads).

Usage (cwd = the scratch copy): PYTHONPATH=src python probe_worked_example.py docs/HYPERPARAMETER_TUNING.md
"""

import contextlib
import io
import math
import re
import subprocess
import sys

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.utils import take_rows

guide = open(sys.argv[1]).read()
block1 = re.findall(r"```python\n(.*?)```", guide, flags=re.S)[0]
ns: dict = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(block1, ns)
train, groups_train, exposure_train = ns["train"], ns["groups_train"], ns["exposure_train"]
evaluator, result = ns["evaluator"], ns["results"]["n_elementary"]
y = train["n_elementary"]


def by_hand(template, params):
    """The fold loop written out: clone, set_params, fit, predict, deviance, size-weighted mean."""
    scores, sizes = [], []
    for train_index, val_index in evaluator.cv.split(train, y, groups_train):
        model = clone(template).set_params(**params)
        model.fit(train.iloc[train_index], y.iloc[train_index], exposure=take_rows(exposure_train, train_index))
        prediction = model.predict(train.iloc[val_index], exposure=take_rows(exposure_train, val_index))
        scores.append(mean_poisson_deviance(y.iloc[val_index], prediction))
        sizes.append(len(val_index))
    return float(np.dot(scores, sizes) / np.sum(sizes)), scores, sizes


def nb_se(scores):
    k = len(scores)
    return float(np.std(scores, ddof=1)) * math.sqrt(1 / k + 1 / (k - 1))


hand, scores, sizes = by_hand(evaluator.model, result.best_params)
record = result.trials[result.best_trial_number]
print(f"LGBM best trial {result.best_trial_number}: by hand {hand!r}; best_value {result.best_value!r}; "
      f"diff {abs(hand - result.best_value):.2e}")
print(f"  fold scores by hand {np.round(scores, 6).tolist()} vs record {np.round(record.fold_values, 6).tolist()}; "
      f"sizes {sizes} vs {list(record.fold_sizes)}; SE by hand {nb_se(scores):.6f} vs record {record.std_error:.6f}")
print(f"  sizes cover the training rows once: sum {sum(sizes)} = n_train {len(train)}")

# --- NB2, empty search space, one trial (guide §4's second block, rebuilt here)
from optuna.trial import FixedTrial

from age_group_prediction.hyperparameter_tuning import CVHyperparameterEvaluator, HyperparameterStudy
from age_group_prediction.modeling import CountModel, NegativeBinomialRegressor
from age_group_prediction.scoring import POISSON_DEVIANCE

nb2_template = CountModel(estimator=NegativeBinomialRegressor(), use_exposure=True, feature_transformer=ns["tree"])
nb2 = CVHyperparameterEvaluator(nb2_template, [], cv=ns["splitter"].cv(n_splits=5, random_state=42), metric=POISSON_DEVIANCE)
nb2_result = HyperparameterStudy(seed=0, n_trials=1).optimize(
    lambda trial: nb2.evaluate(trial, train, y, groups_train, exposure=exposure_train))
nb2_hand, nb2_scores, _ = by_hand(nb2_template, {})
print(f"NB2 one trial: by hand {nb2_hand:.6f} (guide 2.247) vs best_value {nb2_result.best_value:.6f}; "
      f"SE by hand {nb_se(nb2_scores):.6f} (guide 0.454) vs record {nb2_result.trials[0].std_error:.6f}; "
      f"best_params {nb2_result.best_params}; trials {len(nb2_result.trials)}")
three = HyperparameterStudy(seed=0, n_trials=3).optimize(
    lambda trial: nb2.evaluate(trial, train, y, groups_train, exposure=exposure_train))
print(f"  NB2 with 3 trials: values {[round(t.value, 9) for t in three.trials]} (D20: all equal)")

# --- reproducibility
cohort_y = train["n_elementary"]
objective = lambda trial: evaluator.evaluate(trial, train, cohort_y, groups_train, exposure=exposure_train)
study = HyperparameterStudy(seed=101, n_trials=30)
a = study.optimize(objective)
b = study.optimize(objective)
c = HyperparameterStudy(seed=101, n_trials=30).optimize(objective)
d = HyperparameterStudy(seed=102, n_trials=30).optimize(objective)
print(f"REPRO same object twice equal={a == b}; new object same seed equal={a == c}; equal to the guide's result={a == result}; "
      f"seed 102 differs={d != a} (best {a.best_value:.4f} vs {d.best_value:.4f}; params of trial 0 equal: {a.trials[0].params == d.trials[0].params})")

# --- LightGBM threads, in a subprocess each (OpenMP)
code = r"""
import numpy as np, sys
from lightgbm import LGBMRegressor
rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 8)); y = rng.poisson(np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.5))
m = LGBMRegressor(objective="poisson", n_estimators=200, num_leaves=31, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=0, verbosity=-1, n_jobs=int(sys.argv[1])).fit(X, y)
np.save(sys.argv[2], m.predict(X))
"""
import os
import tempfile

tmp = tempfile.mkdtemp(dir=os.environ.get("VAL_TMP"))
preds = {}
for label, jobs in (("1a", 1), ("1b", 1), ("4a", 4), ("4b", 4)):
    path = f"{tmp}/{label}.npy"
    subprocess.run([sys.executable, "-c", code, str(jobs), path], check=True)
    preds[label] = np.load(path)
print(f"LGBM n_jobs=1 twice identical={np.array_equal(preds['1a'], preds['1b'])}; n_jobs=4 twice identical={np.array_equal(preds['4a'], preds['4b'])}; "
      f"n_jobs=1 vs 4 identical={np.array_equal(preds['1a'], preds['4a'])}, max |diff|={np.max(np.abs(preds['1a'] - preds['4a'])):.3e}")
```

</details>

<details><summary><code>probe_threads.py</code></summary>

```python
"""Does LightGBM's thread count change the fit? (guide §8: "n_jobs=1: more threads change the fit")

Mode "example": the guide's worked example (block 1, as written), one study per n_jobs, compared.
Mode "big": a larger raw LightGBM fit, row-wise and col-wise histograms.
Usage: python probe_threads.py example <guide> <n_jobs> <out.npy> | big <n_jobs> <force> <out.npy>
"""
import contextlib, io, re, sys
import numpy as np

if sys.argv[1] == "example":
    guide, jobs, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    code = re.findall(r"```python\n(.*?)```", open(guide).read(), flags=re.S)[0]
    code = code.replace("n_jobs=1,", f"n_jobs={jobs},")
    assert f"n_jobs={jobs}," in code
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, ns)
    r = ns["results"]["n_elementary"]
    np.save(out, np.array([t.value for t in r.trials] + [r.best_value] + ns["prediction"].to_numpy().ravel().tolist()))
else:
    from lightgbm import LGBMRegressor
    jobs, force, out = int(sys.argv[2]), sys.argv[3], sys.argv[4]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100_000, 30)); y = rng.poisson(np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2] * X[:, 3] + 0.5))
    m = LGBMRegressor(objective="poisson", n_estimators=300, num_leaves=63, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=0, verbosity=-1, n_jobs=jobs, **{force: True}).fit(X, y)
    np.save(out, m.predict(X))
```

</details>

<details><summary><code>probe_se.py</code></summary>

```python
"""The corrected SE against repeated sampling (tuning plan §4.3, guide §5).

The true sampling SD of the whole K-fold CV estimate is the SD, over R fresh datasets drawn from one
Poisson DGP, of the CV estimate; it is compared with the mean naive s/sqrt(K) and the mean corrected
s*sqrt(1/K + 1/(K-1)) computed on each dataset. The CV estimates come from the package's evaluator
(FixedTrial) and corrected_std_error, except for the fixed predictor, which needs no fit.
Usage (cwd = the scratch copy): PYTHONPATH=src python probe_se.py
"""

import math
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from optuna.trial import FixedTrial
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.hyperparameter_tuning import CVHyperparameterEvaluator, corrected_std_error
from age_group_prediction.modeling import CountModel
from age_group_prediction.scoring import POISSON_DEVIANCE
from age_group_prediction.splitting import Splitter

K, N = 5, 200
BETA = np.array([0.3, -0.2, 0.15, 0.0, 0.0])


def dataset(rng):
    X = rng.normal(size=(N, len(BETA)))
    mu = np.exp(0.5 + X @ BETA)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(len(BETA))]), rng.poisson(mu).astype(float), mu


def naive(scores):
    return float(np.std(scores, ddof=1)) / math.sqrt(len(scores))


def summarize(label, estimates, naives, correcteds, plan=None):
    true = float(np.std(estimates, ddof=1))
    n, c = float(np.mean(naives)), float(np.mean(correcteds))
    cv_c = float(np.std(correcteds, ddof=1) / np.mean(correcteds))
    extra = f"   (plan: {plan})" if plan else ""
    print(f"{label:<44} true SD {true:.4f} | mean naive {n:.4f} ({n / true:.2f}x) | mean corrected {c:.4f} ({c / true:.2f}x) | "
          f"corrected SE's own CV {cv_c:.2f}{extra}")


def fixed_predictor(R, rng):
    est, nv, cr = [], [], []
    for r in range(R):
        X, y, mu = dataset(rng)
        folds = [v for _, v in Splitter("random").cv(n_splits=K, random_state=r).split(X)]
        scores = [mean_poisson_deviance(y[v], mu[v]) for v in folds]
        est.append(np.average(scores, weights=[len(v) for v in folds]))
        nv.append(naive(scores))
        cr.append(corrected_std_error(scores))
    summarize("(a) fixed predictor (the true mean)", est, nv, cr, "true 0.061, corrected 0.090, naive 0.060")


def learned(label, estimator, R, rng):
    est, nv, cr = [], [], []
    for r in range(R):
        X, y, _ = dataset(rng)
        evaluator = CVHyperparameterEvaluator(CountModel(estimator=estimator), [], cv=Splitter("random").cv(n_splits=K, random_state=r),
                                              metric=POISSON_DEVIANCE)
        captured = {}

        class Trial(FixedTrial):
            def set_user_attr(self, key, value):
                captured[key] = value

        est.append(evaluator.evaluate(Trial({}), X, y))
        scores = captured["fold_scores"]
        nv.append(naive(scores))
        cr.append(corrected_std_error(scores))
    summarize(label, est, nv, cr)


rng = np.random.default_rng(2026)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fixed_predictor(2000, rng)
    learned("(b) PoissonRegressor(alpha=1e-4), evaluator", PoissonRegressor(alpha=1e-4), 1000, rng)
    learned("(b2) LGBMRegressor(poisson, 100 trees)", LGBMRegressor(objective="poisson", n_estimators=100, learning_rate=0.1,
                                                                 num_leaves=15, min_child_samples=10, n_jobs=1, verbosity=-1, random_state=0),
            400, rng)

# (c) stratified_by_group with ~20% of the rows alone in their stratum: the true n_val/n_train
# against the 1/(K-1) the formula assumes.
groups = np.concatenate([np.arange(40), np.repeat(np.arange(40, 60), 8)])  # 40 singletons of 200 rows = 20%
X = pd.DataFrame({"x": np.zeros(len(groups))})
folds = list(Splitter("stratified_by_group").cv(n_splits=K, random_state=0).split(X, None, groups))
validated = sum(len(v) for _, v in folds)
ratio = float(np.mean([len(v) / len(t) for t, v in folds]))
assumed, actual = math.sqrt(1 / K + 1 / (K - 1)), math.sqrt(1 / K + ratio)
print(f"(c) stratified_by_group, {40 / len(groups):.0%} singletons: validated {validated} of {len(groups)} rows; "
      f"mean n_val/n_train {ratio:.4f} vs 1/(K-1) {1 / (K - 1):.4f}; corrected SE overstated by {assumed / actual - 1:.1%}   (plan: 7%)")

# (d) the SE's own sampling error at K=5: the relative SD of s, from normal fold scores.
s = np.std(np.random.default_rng(0).normal(size=(200_000, K)), axis=1, ddof=1)
print(f"(d) relative SD of s at K={K}: {np.std(s) / np.mean(s):.3f}   (plan: ~35%)")
```

</details>

<details><summary><code>probe_pruning.py</code></summary>

```python
"""Pruning guidance (tuning plan §4.2, guide §7), on synthetic objectives that report per fold as the
evaluator does (trial.report(fold score, step=fold)), and on the guide's worked example.

Usage (cwd = the scratch copy): PYTHONPATH=src python probe_pruning.py docs/HYPERPARAMETER_TUNING.md
"""

import contextlib
import io
import re
import sys
import warnings

import numpy as np
import optuna
from optuna.pruners import MedianPruner, PercentilePruner, SuccessiveHalvingPruner, WilcoxonPruner

from age_group_prediction.hyperparameter_tuning import HyperparameterStudy

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
PRUNERS = {
    "Wilcoxon": lambda: WilcoxonPruner(p_threshold=0.1),
    "SuccessiveHalving": lambda: SuccessiveHalvingPruner(),
    "Median": lambda: MedianPruner(),
    "Percentile(25)": lambda: PercentilePruner(25.0),
}


# ---------- P1: 10 folds of unequal difficulty; 10 good trials, then a bad one that ties on fold 0
def p1(difficulty, label):
    out = {}
    for name, make in PRUNERS.items():
        def objective(trial):
            trial.suggest_float("x", 0, 1)
            noise = np.random.default_rng(trial.number).normal(0, 0.05, size=len(difficulty))
            scores = difficulty + noise if trial.number < 10 else difficulty + np.r_[0.0, np.full(len(difficulty) - 1, 0.5)]
            for fold, score in enumerate(scores):
                trial.report(float(score), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study = HyperparameterStudy(seed=0, n_trials=11, pruner=make())
        study.optimize(objective)
        bad = study.optuna_study_.trials[10]
        good_pruned = sum(t.state.name == "PRUNED" for t in study.optuna_study_.trials[:10])
        out[name] = (f"pruned after {len(bad.intermediate_values)} folds" if bad.state.name == "PRUNED" else "never pruned") + f" (good trials pruned: {good_pruned})"
    print(f"P1 {label}: " + "; ".join(f"{k}: {v}" for k, v in out.items()))


rng = np.random.default_rng(3)
d = rng.permutation(np.linspace(0.5, 2.0, 10))
p1(d, f"difficulties in random order {np.round(d, 2).tolist()}")
p1(np.sort(d), "fold 0 the easiest (increasing)")
p1(np.sort(d)[::-1], "fold 0 the hardest (decreasing)")


# ---------- P2: fixed vs shuffled fold order, 8 folds, 80 trials, 4 seeds
D8 = np.random.default_rng(11).permutation(np.linspace(0.6, 1.8, 8))


def fold_score(x, y, fold):
    # The true per-fold loss of a setting; a small deterministic per-(setting, fold) wobble.
    wobble = 0.03 * np.sin(37.0 * x + 11.0 * y + 3.0 * fold)
    return float(D8[fold] * (1.0 + (x - 0.3) ** 2 + 0.5 * (y - 0.7) ** 2) + wobble)


def p2(pruner_name, shuffled, seed):
    make = {"Wilcoxon": lambda: WilcoxonPruner(p_threshold=0.1), "SuccessiveHalving": lambda: SuccessiveHalvingPruner()}[pruner_name]

    def objective(trial):
        x, y = trial.suggest_float("x", 0, 1), trial.suggest_float("y", 0, 1)
        order = np.random.default_rng(1000 * seed + trial.number).permutation(8) if shuffled else np.arange(8)
        scores = []
        for position, fold in enumerate(order):
            score = fold_score(x, y, fold)
            scores.append(score)
            # Wilcoxon: Optuna's docs report the instance id as the step; successive halving's rungs
            # count steps, so there the step is the position.
            trial.report(score, step=int(fold) if pruner_name == "Wilcoxon" else position)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    result = HyperparameterStudy(seed=seed, n_trials=80, pruner=make()).optimize(objective)
    n_complete = sum(t.state.name == "COMPLETE" for t in result.trials)
    return result.best_value, n_complete


for name in ("Wilcoxon", "SuccessiveHalving"):
    rows = []
    for seed in range(4):
        (fixed, nf), (shuf, ns) = p2(name, False, seed), p2(name, True, seed)
        rows.append(f"seed {seed}: fixed {fixed:.4f} ({nf} complete) vs shuffled {shuf:.4f} ({ns} complete)")
    worse = sum(float(r.split("shuffled ")[1].split()[0]) > float(r.split("fixed ")[1].split()[0]) for r in rows)
    print(f"P2 {name}: " + "; ".join(rows) + f"  -> shuffled worse in {worse} of 4")

# ---------- P3: the guide §7 claims on the worked example (seed 101, 30 trials, default pruner settings)
block1 = re.findall(r"```python\n(.*?)```", open(sys.argv[1]).read(), flags=re.S)[0]
ns: dict = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(block1, ns)
train, groups_train, exposure_train, evaluator = ns["train"], ns["groups_train"], ns["exposure_train"], ns["evaluator"]
y_train = train["n_elementary"]
for name, pruner in (("WilcoxonPruner()", WilcoxonPruner()), ("SuccessiveHalvingPruner()", SuccessiveHalvingPruner())):
    r = HyperparameterStudy(seed=101, n_trials=30, pruner=pruner).optimize(
        lambda trial: evaluator.evaluate(trial, train, y_train, groups_train, exposure=exposure_train))
    states = [t.state.name for t in r.trials]
    print(f"P3 {name} on the worked example: {states.count('COMPLETE')} complete, {states.count('PRUNED')} pruned of 30; best {r.best_value:.3f}")
```

</details>

<details><summary><code>probe_diff.py</code></summary>

```python
"""The uncommitted parameters.py diff: what it changes, against HEAD's behaviour.

Run once with PYTHONPATH=<head>/src and once with <diff>/src; prints the observed behaviour.
"""
import json, math, sys
import age_group_prediction.hyperparameter_tuning as ht
from age_group_prediction.hyperparameter_tuning import CategoricalParameter, HyperparameterStudy
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
print("module:", ht.__file__.split("/val/")[1])

def built(choices):
    try:
        CategoricalParameter("p", choices); return "accepted"
    except ValueError as e:
        return "rejected: " + str(e).splitlines()[-1].strip()[:70]

for label, c in [("[nan, 1.0]", [math.nan, 1.0]), ("[inf, 1.0]", [math.inf, 1.0]), ("[-inf, 1.0]", [-math.inf, 1.0]),
                 ("[nan, nan] same object", [math.nan] * 2), ("[True, 1]", [True, 1]), ("[1, 1.0]", [1, 1.0]),
                 ("[0.0, -0.0]", [0.0, -0.0]), ("[None, None]", [None, None]), (["a", "b", "a"], ["a", "b", "a"]), ("[None, 'a', 2, 2.5]", [None, "a", 2, 2.5])]:
    print(f"  {str(label):<26} {built(c)}")

class F(float):
    __hash__ = None  # an unhashable float subclass: the only way into the TypeError fallback
print(f"  [F(1.0), F(1.0)] (unhashable float subclass) {built([F(1.0), F(1.0)])}")
print(f"  [F(1.0), F(2.0)]                              {built([F(1.0), F(2.0)])}")

# HEAD: does a NaN choice reach the result and break strict JSON?
try:
    p = CategoricalParameter("p", [math.nan])
    r = HyperparameterStudy(seed=0, n_trials=1).optimize(lambda t: 1.0 if math.isnan(p.suggest(t)) else 0.0)
    try:
        json.dumps(r.to_dict(), allow_nan=False); strict = "ok"
    except ValueError as e:
        strict = f"raises ({e})"
    print(f"  a NaN choice in a study: best_params={r.best_params}; json.dumps(allow_nan=False) {strict}; json.dumps default writes {json.dumps(r.best_params)}")
except ValueError as e:
    print(f"  a NaN choice in a study: not buildable ({str(e).splitlines()[-1].strip()[:60]})")
```

</details>
