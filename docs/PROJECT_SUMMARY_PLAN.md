# Project summary document: the plan

> **Status (2026-11-23):** plan approved; S0 done (branch `docs/project-summary`, this document,
> the environment checks). S1a, S1b done. **Paused (2026-11-24) for the simulator revision (P12,
> `docs/SIMULATOR_REALISM_PLAN.md`, its own PR from `main`).** After it merges: **S1c** (section 2
> for the revised simulator), then S2. The implementation record is §9.

The project is documented in 62 files under `docs/`, most of them plans and work records. No
single document tells a new reader what the task is, how the data is generated, how it is
preprocessed, what the two models are, how they were tuned and what the results say. This plan
produces that document, `docs/PROJECT_SUMMARY.md`, in its own PR.

## 1. Decisions

| # | Decision |
|---|---|
| P1 | Reader: a technical reader new to the repo. Self-contained, full formulas, short motivation per choice, about 600–900 lines, links to the detailed docs. |
| P2 | Structure: 7 sections (§2), not the 4 first proposed. |
| P3 | Results as markdown tables plus about 6–8 static PNGs committed under `docs/figures/`. |
| P4 | New analyses are first computed by git-ignored scripts under `.mlflow-check/summary/`. Those the user finds valuable are then promoted to notebook 04 (S9, planned separately). |
| P5 | Model 3 (Dirichlet-multinomial) gets a marked placeholder section; the user adds it later. |
| P6 | A new PR: branch `docs/project-summary` from `main`, a draft PR opened after the first commit. |
| P7 | Small sub-tasks, one stop each, so each can be verified alone. |
| P8 | Each sub-task names its implementing model. Sonnet 5.5 or Opus 5.5 by default; Fable 5.2 only for the hard ones (S5d, S7a, S7b, the S8b review). |
| P9 | The PNGs are drawn with matplotlib (and shap's own matplotlib plots). `kaleido` is not installed and is not added. The PNGs therefore differ in look from notebook 04's Plotly figures. |
| P10 | Depth where it matters (user, 2026-11-23): sections 2 (data generation), 3 (preprocessing) and 4 (models) are specific and in LaTeX: every generated variable, every transform and every model has its formula, with its symbols defined, in the order the code applies them. Section 1 stays lean (no seeds, config defaults or overrides there). Cross-validation belongs to section 5 (tuning), not section 1. |
| P12 | The realistic simulator first (user, 2026-11-24). Section 2 questioned the neighborhood model; the user chose to make the simulator realistic, calibrated to Israel, step by step (neighborhoods first), in separate PRs from `main` (`docs/SIMULATOR_REALISM_PLAN.md`). This branch merges `main` after each simulator PR. The order becomes: simulator step(s) → S1c and S2–S4b (the documentation) → the reruns (notebooks 03 and 04 on the new data) → S5a onward. The results of the run at `da6afc9` (§3) are superseded then. |
| P11 | The reader's view (user, 2026-11-24), for the whole summary: no implementation details (no Python names or calls, file paths, line numbers, config keys or notebook names; dataset column names and links to the detailed docs stay). Formulas carry their actual values inline, with no notation or parameter tables: distributions labeled, e.g. $\mathcal N(\mu=0, \sigma^2=1^2)$, linear formulas with plain coefficients explained in words. Each section opens by describing and motivating its subject. Code lines go only into §9, the record. |

The summary describes **the path the notebooks run**: the `student_simulator` package and the
new `preprocessing.py`, `feature_engineering`, `modeling`, `hyperparameter_tuning` and
`inspection` packages. The old stack (`models/`, `modeling_config.py`, the 6-room reference) is
not described.

## 2. The summary's outline

1. **Task and settings**: 2–5 lines on the task, then a setup table (unit = building, 3 cohort
   targets, 1,529 buildings / 150 neighborhoods / 61,317 apartments, seeds, 917/306/306 split).
2. **Data generation process**: neighborhood → building → apartment → outcomes → aggregation, in
   draw order, one formula per variable with its values inline (P11), a column table of the dataset.
3. **Preprocessing**: room counts → shares (3 rooms the reference), the exposure, one table of
   feature × model family (transform, formula, motivation), the candidate variants, the checks.
4. **Models**: 4.1 the shared exposure treatment (weighted rate vs log offset, the Poisson
   equivalence); 4.2 Model 1, independent cohorts; 4.3 Model 2, total × probability (the
   multinomial likelihood identity, calibration); 4.4 Model 3, a placeholder (the planned
   formulation, marked "not implemented in the new stack").
5. **Tuning and evaluation protocol**: split roles, 5-fold CV, TPE, the composite RMSE objective,
   the 24 studies, the search-space table, the six metrics, the grouped bootstrap, selection and
   the single test evaluation.
6. **Results and model inspection**: the 24 runs, the two best runs and their hyperparameters,
   validation and test scores with CIs and the paired difference, permutation importance and SHAP
   (composite and per cohort), best and worst buildings, biases.
7. **Discussion**: 7.1 which model when; 7.2 why the models fail where they do; 7.3 why
   `n_apartments` has low importance; 7.4 future steps.

## 3. Sources

| Section | Code | Existing doc to adapt |
|---|---|---|
| 2 | `src/student_simulator/{neighborhood,building,apartment,outcomes,features,pipeline}.py`, `configs/simulation.yml`, the overrides at `notebooks/03_hyperparameter_tuning.py:107-124` | `docs/SIMPLIFIED_MODEL_PLAN.md` §6 (LaTeX already) |
| 3 | `src/age_group_prediction/preprocessing.py`, `feature_engineering/{transforms,transformer}.py`, the declarations at `notebooks/03_hyperparameter_tuning.py:475-703` | `docs/FEATURE_TRANSFORMATIONS.md` |
| 4 | `modeling/{base,count_model,negative_binomial,independent_cohorts,cohort_probability,total_times_probability}.py`; Model 3: `models/bayesian_components.py:29-155` | `docs/INDEPENDENT_COHORT_MODELS.md`, `docs/TOTAL_TIMES_PROBABILITY_MODEL.md`, `docs/BAYESIAN_CONDITIONAL_MODEL_OVERVIEW.md` |
| 5 | `hyperparameter_tuning/{cross_validation,search_spaces,builders,components,aggregation}.py`, `scoring.py`, `inspection/bootstrap.py`, `splitting/splitters.py` | `docs/HYPERPARAMETER_TUNING.md`, `docs/MLFLOW_RUNS.md` |
| 6–7 | the final run's tables in `.mlflow-check/`, `inspection/{permutation,shap_values}.py` | `docs/EXPERIMENT_NOTEBOOK_HANDOFF.md` status block, `docs/MODEL_INSPECTION.md` |

**The results' store.** `.mlflow-check/mlflow.db` with `.mlflow-check/mlartifacts/` (git-ignored),
experiment `age-group-check`, 26 runs of 2026-11-22 at code `da6afc9`: 24 tuning runs, the best
runs' run `bf9bc0f5ff9d44928ec251490d7e72ad`, the final (test) run
`fff76eb93148408c93a7064ffe498664`. Model 1's best tuning run is
`17ee770ff6ff46b69058a2158e8365cf`, Model 2's is `9ecae8a3b21346ea8ebf9c374dbe62ed`. The final
run's tables: `test_scores`, `test_difference`, `val_against_test`, `test_importance`,
`test_buildings`, `test_building_scores`, `test_shap_values`.

**Doc/code disagreements the summary must not copy.** Code wins; fixing these docs is out of scope.

- The root README's `StudentPopulationSimulator(config).run()` example; the code is
  `StudentPopulationSimulator.from_config(config).simulate()`.
- `FEATURE_ENGINEERING.md` (6-room reference, shares over `n_apartments`) and
  `DATA_AND_SPLITTING.md` (the known-neighborhood split) describe the old stack. The notebooks
  use the 3-room reference and `Splitter("random")`.
- Superseded tuning decisions: `n_trials` 200 and 400 (now 100, 10 startup, D31), the deviance
  objective (now RMSE, D42), coordinate descent for Model 2 (now joint, D40).
- `SIMPLIFIED_MODEL_PLAN.md` mentions a long room table and a stored building total; the dataset
  is wide (`3_rooms … 6_rooms`) and has no total column.
- `FEATURE_TRANSFORMATIONS.md`'s reference statistics come from about 250 buildings; the
  notebooks use 1,529.

## 4. Facts found in the exploration (2026-11-23)

Each session re-verifies what it uses; these are leads, not sources.

**Simulator** (`src/student_simulator/`, parameters in `configs/simulation.yml`, seed 42).
- Notebooks override to 150 neighborhoods and a building rate of 9: 1,529 buildings, 61,317
  apartments. Targets `n_kindergarten`, `n_elementary`, `n_highschool`; no total column.
- Neighborhood: SES, median age, household size (depends on age), daycares (Poisson, log-linear
  in SES and age), school status (categorical).
- Building: $B_j = 1 + \text{Poisson}(\text{rate})$; apartments
  $A_b = \text{clip}(\text{round}(\mathcal N(40, 12^2)), 12, 80)$, **independent of every feature**;
  room mix: a tilt of base shares by household size and SES, then Dirichlet, then multinomial.
- Apartment total: NB2 with $\log\mu_i$ linear in room dummies (3 rooms the reference), SES²,
  scaled household size, daycare saturation $1 - e^{-D/3}$, school dummies, scaled median age, a
  rooms × household-size interaction, plus a building and a neighborhood random effect.
  **No building-size term and no linear SES term.**
- Cohort split: one lifecycle score $\theta_i$ (rooms, daycare saturation, age, household size, a
  positive-part interaction, a building random effect), a softmax over the three cohorts, a
  multinomial of the apartment's total.
- The building's counts are the sums over its apartments, so the expected count is proportional
  to `n_apartments`. The random effects are returned by the simulator but are not dataset columns.

**Preprocessing.**
- `ShareTransformer(("3_rooms","4_rooms","5_rooms","6_rooms"), reference_column="3_rooms")`: shares
  over the row sum, the 3-room share dropped, the counts dropped.
- `ExposureTransformer("n_apartments")`: the raw count as the exposure.
- Trees: every feature passed through, `school_status` one-hot, `n_apartments` kept as a feature.
  GLMs: standardize `ses`, `avg_household_size`, `median_age`; shares centered then divided by
  0.1; daycares divided by 2; `n_apartments` not a feature (exposure only). Candidate variants add
  SES² around 0, a relative-saturation daycare transform, and share interactions.

**Models.**
- `CountModel`: $\hat\mu_b = E_b \hat f(x_b)$. LightGBM and `PoissonRegressor` fit the rate
  `y / exposure` with `sample_weight=exposure`; NB2 takes the raw exposure as a log offset.
- Model 1 (`IndependentCohortModels`): one `CountModel` per cohort. Families `lgbm_poisson`,
  `lgbm_gaussian`, `poisson_glm`, `nb2`.
- Model 2 (`TotalTimesProbabilityModel`): $\hat C_{b,k} = \hat\mu_b \hat p_{b,k}$; the total is a
  `CountModel` on the cohorts' sum; the probabilities are a classifier (`logistic` or `lgbm`) on
  weighted categorical rows, with optional calibration on building folds; no exposure.
- Model 3 exists only in the old stack (`models/bayesian_components.py`): NB2 total with a
  neighborhood random effect, Dirichlet-multinomial composition, Pyro NUTS.

**Tuning.** Optuna TPE, no pruning, 100 trials (10 startup), 5-fold CV inside the training part,
objective the pooled (composite) RMSE over all building × cohort cells, mean over folds. 8 Model 1
studies and 16 Model 2 studies (family × exposure on/off). Selection on the validation part, then
a refit on train + validation and one test evaluation. Bootstrap: 1,000 resamples of
neighborhoods, percentile 95% CI, paired differences.

**Results.**
- Best runs: Model 1 `lgbm_gaussian` with exposure (validation RMSE 4.5858); Model 2
  `lgbm_poisson` + `logistic` with exposure (4.4941).
- Test composite RMSE 4.2118 against 4.1603; the difference −0.0516, CI [−0.130, 0.027]. Only the
  composite log loss's interval excludes 0.
- Both under-predict the test total: 7,062 and 7,044 against 7,399. Kindergarten bias is largest.
- Importance (composite, both models): `avg_household_size` first, `median_age` second, then
  `6_rooms_share` and `ses`. `n_apartments` ranks 5–8 of 9 by permutation and 7–9 by SHAP.
- GLM runs without the exposure: validation RMSE 5.40–5.44 against 4.61–4.66 with it. LightGBM
  without it: 4.656 against 4.586.

**The `n_apartments` hypothesis (S5c tests it).** Permutation importance and SHAP keep each row's
exposure fixed while the `n_apartments` column is shuffled. The size effect is carried by the
exposure, so the measurement shows only departures from proportionality, which the simulator does
not have. In `test_shap_values.csv` the `base_value` correlates 1.0 with `n_apartments`.

## 5. Workflow

- The user commits and pushes; a draft PR after S0's commit. Commit messages in a `text` block,
  no `Co-Authored-By` line, no "Generated with" line.
- One sub-task per session, on the model §6 names. At each stop the next session's prompt is
  printed in the chat as one block (the opening of §8, a blank line, the sub-task's text), with
  the model to select named above it.
- At each stop: the diff, the check's output, the commit message, and §9 updated. On the
  sub-tasks marked "review", an independent review subagent runs before the stop and its findings
  are reproduced before being fixed.
- Scripts and their outputs live in `.mlflow-check/summary/` (git-ignored), run with
  `~/.venvs/age-group-prediction/bin/python`. One script per sub-task. Each check runs once.
- If a plan assumption breaks, the session stops and reports with the evidence.
- No file under `src/` or `notebooks/` changes in S0–S8b, so the test suite is not rerun.

## 6. Sub-tasks

| # | Sub-task | Model | Review | Done |
|---|---|---|---|---|
| S0 | Setup: branch, plan doc, environment checks | Sonnet 5.5 (done by the planning session) | no | ✅ |
| S1a | Skeleton + section 1 (task and settings) | Sonnet 5.5 | no | ✅ |
| S1b | Section 2 (data generation process) | Opus 5.5 | yes | ✅ |
| S1c | Section 2 rewritten for the revised simulator (after P12's PR merges into this branch) | Opus 5.5 | yes | |
| S2 | Section 3 (preprocessing) | Opus 5.5 | yes | |
| S3a | Section 4.1–4.2 (exposure treatment, Model 1) | Opus 5.5 | yes | |
| S3b | Section 4.3–4.4 (Model 2, Model 3 placeholder) | Opus 5.5 | yes | |
| S4a | Section 5, part 1 (split, CV, tuner, objective, studies, search space) | Sonnet 5.5 | no | |
| S4b | Section 5, part 2 (metrics, bootstrap, selection, test, MLflow) | Sonnet 5.5 | yes | |
| S5a | Analysis: rebuild the two best models, score tables, best/worst buildings | Opus 5.5 | no | |
| S5b | Analysis: error and bias by size, cohort and feature bins | Sonnet 5.5 | no | |
| S5c | Analysis: the `n_apartments` test | Opus 5.5 | yes | |
| S5d | Analysis: the noise floor from the simulator | **Fable 5.2** | yes | |
| S5e | The PNGs | Sonnet 5.5 | no | |
| S6a | Section 6, part 1 (runs, best runs, scores, the difference) | Sonnet 5.5 | no | |
| S6b | Section 6, part 2 (permutation, SHAP, best/worst, biases) | Opus 5.5 | yes | |
| S7a | Section 7.1–7.2 (which model when, why they fail) | **Fable 5.2** | yes | |
| S7b | Section 7.3–7.4 (`n_apartments`, future steps) | **Fable 5.2** | yes | |
| S8a | Links, README entries, render check | Sonnet 5.5 | no | |
| S8b | Whole-document review against the code | **Fable 5.2** | is the review | |
| S9 | Promote the valued analyses to notebook 04 | planned separately | — | |

Why Fable on four: S5d derives expected building counts from the simulator's formulas, and a
mistake there silently misstates what is achievable; S7a and S7b are the interpretation, where a
wrong causal claim is the main risk; S8b is the one read not anchored on an implementing session.

## 7. The sub-tasks in detail

### S0. Setup
Done when:
- [x] Branch `docs/project-summary` created from `main`.
- [x] This document written.
- [x] `find src notebooks configs docs .mlflow-check -flags +dataless`: 0 files.
- [x] `kaleido` not installed; the user chose matplotlib (P9).

### S1a. Skeleton + section 1 (Sonnet 5.5)
Create `docs/PROJECT_SUMMARY.md` with the 7 headings and their sub-headings (§2), a short "what
this document covers" note (the notebooks' path; results from the run at `da6afc9`), and section 1.
Done when:
- [x] `check_s1a.py` runs the simulator with the notebooks' overrides and the notebooks' split,
      and prints the building, neighborhood and apartment counts, the column list and the split
      sizes.
- [x] The setup table matches that output (expected 1,529 / 150 / 61,317 and 917 / 306 / 306).
- [x] Sections 2–7 hold headings only.

### S1b. Section 2, data generation (Opus 5.5, review)
Neighborhood; building (count, apartments, room mix: tilt → Dirichlet → multinomial); apartment;
total children (the NB2 log-linear mean with random effects); cohort split (lifecycle score,
softmax, multinomial); aggregation to the building. Values inline (P11), the dataset's column
table. Adapt `docs/SIMPLIFIED_MODEL_PLAN.md` §6; follow the code where they differ.
LaTeX per P10: no variable, transform or model is described in words only.
Done when:
- [x] Every generated variable has its formula, in draw order.
- [x] `check_s1b.py` reads every value written into section 2 and compares it with
      `configs/simulation.yml` plus the notebooks' overrides: zero mismatches, every value covered.
- [x] The section states what is hidden from the models (the random effects, apartment-level rows).
- [x] Review findings reproduced and fixed.

### S1c. Section 2 for the revised simulator (Opus 5.5, review)
Starts after the simulator PR (P12) merges into `main` and `main` is merged into this branch.
Rewrite sections 2.1–2.6 for the new model in `docs/SIMULATOR_REALISM_PLAN.md`, keeping the
opening as revised in S1b. The user's notes (2026-11-24):
- The explanations must be better throughout; each variable says what it represents.
- SES: the neighborhood's low or high socioeconomic class, on a national index (relative to the
  whole population, not to the city: a global feature).
- Median age: the median age of the residents of neighborhood $j$.
- No explanation of what clipping does.
- Each relation is motivated (why household size falls with median age, why daycares follow
  demand and purchasing power, why school status follows the demographics).
Done when:
- [ ] `check_s1c.py` (adapted from `check_s1b.py`) matches every inline value with the new config
      plus the notebooks' overrides, the columns with the dataset, and finds no implementation
      detail; section 1's counts are re-checked (they change with the new draws).
- [ ] Review findings reproduced and fixed.

### S2. Section 3, preprocessing (Opus 5.5, review)
`ShareTransformer` (the formula, the 3-room reference and why); `ExposureTransformer`; the
transform vocabulary in use; the feature × family table (trees, count GLMs, logistic) with formula
and motivation per feature; the candidate variants per family; where statistics are learned (the
training fold only); the validation checks.
LaTeX per P10: no variable, transform or model is described in words only.
Done when:
- [ ] `check_s2.py` builds each candidate transformer from the notebook's declarations, fits it on
      the training part and prints its output columns; the doc's tables agree row by row.
- [ ] Every transform has a one-line motivation traced to `FEATURE_TRANSFORMATIONS.md` or a
      docstring.
- [ ] Review findings reproduced and fixed.

### S3a. Section 4.1–4.2 (Opus 5.5, review)
4.1 `CountModel`: $\hat\mu_b = E_b \hat f(x_b)$; the weighted-rate form and its Poisson
equivalence to the log offset (the derivation); the Gaussian case; NB2 taking the raw exposure.
4.2 Model 1: the formulation, the four families (loss, link, penalty), motivation, trade-offs.
LaTeX per P10: no variable, transform or model is described in words only.
Done when:
- [ ] `check_s3a.py` fits a Poisson GLM on the training part both ways (rate with weights, and
      offset) and prints the largest prediction difference.
- [ ] Each formula's code line is listed in §9.
- [ ] Review findings reproduced and fixed.

### S3b. Section 4.3–4.4 (Opus 5.5, review)
4.3 Model 2: total × probability; the multinomial likelihood identity (weighted rows); the
logistic and LightGBM classifiers; calibration on building folds; exact reconciliation;
motivation and trade-offs. 4.4 Model 3: the planned NB2 × Dirichlet-multinomial formulation from
the old stack, marked not implemented in the new stack, with a "to be added" note.
LaTeX per P10: no variable, transform or model is described in words only.
Done when:
- [ ] `check_s3b.py` rebuilds Model 2's best run and prints the largest
      $|\sum_k \hat C_{b,k} - \hat\mu_b|$ on the validation part.
- [ ] Each formula's code line is listed in §9.
- [ ] Section 4.4 is clearly a placeholder.
- [ ] Review findings reproduced and fixed.

### S4a. Section 5, part 1 (Sonnet 5.5)
The split and each part's role; 5-fold CV inside the training part; TPE (seeds, 100 trials, 10
startup, no pruning and why); the composite RMSE objective (the formula, why not a weighted sum,
why RMSE over the deviance); the 8 + 16 studies; the per-cohort transformer categoricals; the
search-space table.
Done when:
- [ ] `check_s4a.py` prints the search space drawn from `search_spaces.py` (parameter, range, log)
      and the settings from the final run's `settings.json`; the doc's tables agree.
- [ ] No superseded decision is stated as current (§3).

### S4b. Section 5, part 2 (Sonnet 5.5, review)
The six metrics with formulas; the grouped bootstrap (neighborhoods, 1,000 resamples, percentile
CI, the paired difference); run selection on validation; the refit on train + validation and the
single test evaluation; one paragraph on what MLflow holds.
Done when:
- [ ] Each metric's `scoring.py` line is listed in §9.
- [ ] The bootstrap settings are read from the final run's `settings.json`.
- [ ] Review findings reproduced and fixed.

### S5a. Rebuild, score tables, best/worst buildings (Opus 5.5)
`s5a.py` reads the final run's tables, rebuilds the two best models by the notebook's path (the
`read_tuning_run` / `rebuild_best_run` logic in `notebooks/04_model_inspection.py`, the builders
in `hyperparameter_tuning/builders.py`), refits on train + validation, and writes CSVs: the 24
runs' validation table; the best runs' validation and test scores with CIs; the difference table;
the best and worst 5 test buildings per model with features and true and predicted counts; the
test predictions for S5b–S5e.
Done when:
- [ ] The rebuilt models' test predictions equal `test_buildings.csv`.
- [ ] The score table equals `test_scores.csv` to the last digit.
- [ ] The tables are shown at the stop.

### S5b. Error and bias by size, cohort, feature bins (Sonnet 5.5)
`s5b.py`, from S5a's saved predictions: RMSE and bias by `n_apartments` quintile, by cohort, by
bins of household size and median age, by true-count bins; the children-per-apartment rate in the
train, validation and test parts (does the split explain the under-prediction?).
Done when:
- [ ] Each binning's rows sum to 306 buildings.
- [ ] The pooled bias equals `test_scores.csv`.
- [ ] The tables are shown at the stop.

### S5c. The `n_apartments` test (Opus 5.5, review)
`s5c.py`: permutation importance on the test part, with notebook 04's seed and repeats, in three
arms: (a) the feature shuffled, the exposure fixed (the existing measurement); (b) the exposure
shuffled, the feature fixed; (c) both shuffled together. Plus the exposure on/off pairs from the
24 runs and the correlation of SHAP `base_value` with `n_apartments`.
Done when:
- [ ] Arm (a) equals `test_importance.csv` for `n_apartments`.
- [ ] The three arms are shown at the stop, for both models, composite and per cohort.
- [ ] Review findings reproduced and fixed.

### S5d. The noise floor (Fable 5.2, review)
`s5d.py`: from the simulator's formulas and its returned random effects, the expected cohort
counts per test building (i) given the features and the true random effects, (ii) given the
features only. Their composite and per-cohort RMSE on the test part are the floor the models'
scores are read against. The derivation goes into §9.
Done when:
- [ ] Over many redraws of the outcomes for fixed buildings, the mean count converges to
      expectation (i).
- [ ] The oracle's RMSE and bias on test are shown next to both models' at the stop.
- [ ] Review findings reproduced and fixed.

### S5e. The PNGs (Sonnet 5.5)
`s5e.py`, matplotlib (P9), into `docs/figures/`: permutation importance (both models); a SHAP
beeswarm per model (total); predicted against true; error by size; the three `n_apartments` arms;
one worst-case waterfall. The user chooses which stay.
Done when:
- [ ] Each file opens and is under 300 KB.
- [ ] Every plotted value comes from S5a–S5d's output files or the final run's tables.
- [ ] The figures are shown at the stop.

### S6a. Section 6, part 1 (Sonnet 5.5)
The 24 runs table; the two best runs and their hyperparameters; validation and test scores with
CIs; the paired difference; validation against test.
Done when:
- [ ] `check_s6a.py` extracts every number in the part and matches it to S5a's files.

### S6b. Section 6, part 2 (Opus 5.5, review)
Permutation importance and SHAP (composite and per cohort, with the figures); train against
validation importance; best and worst buildings (what they have in common); biases by cohort,
size and bins. Descriptive only; the interpretation is section 7.
Done when:
- [ ] `check_s6b.py` matches every number to S5a–S5c's files and `test_importance.csv`.
- [ ] No causal claim appears in section 6.
- [ ] Review findings reproduced and fixed.

### S7a. Section 7.1–7.2 (Fable 5.2, review)
7.1 Which model when: what the intervals do and do not support; per-cohort differences;
reconciliation; cost; exposure sensitivity by family. 7.2 Why the models fail: the error against
the noise floor; variance growing with size; the hidden random effects; the under-prediction and
its cause as S5b measured it.
Done when:
- [ ] Every claim cites a table, a figure or an S5 number.
- [ ] Claims the numbers do not support are dropped or stated as open questions.
- [ ] Review findings reproduced and fixed.

### S7b. Section 7.3–7.4 (Fable 5.2, review)
7.3 Why `n_apartments` has low importance: the generating process, the exposure, what permutation
and SHAP hold fixed, S5c's three arms. 7.4 Future steps, each tied to a finding. Candidates:
Model 3 for uncertainty and neighborhood effects; a split grouped by neighborhood; importance
that moves the feature and the exposure together; distributional metrics; the total's bias.
Done when:
- [ ] As S7a.
- [ ] Each future step names the finding that motivates it.
- [ ] Review findings reproduced and fixed.

### S8a. Links and render (Sonnet 5.5)
Add the summary to `docs/README.md` and the root `README.md`. Check that every relative link and
image path resolves. Check the LaTeX in the VS Code preview and on the GitHub branch view (blank
lines around `$$`, no `_` clashes inside inline math).
Done when:
- [ ] `check_s8a.py` reports zero broken links and image paths.
- [ ] The render problems found are listed in §9 and fixed.

### S8b. Whole-document review (Fable 5.2)
A fresh session reads the summary against the code as a new reader would: wrong formulas, numbers
without a source, claims the results do not support, gaps. Findings reproduced, then fixed. Then
the PR description.
Done when:
- [ ] The findings and each one's resolution are listed in §9.
- [ ] The PR description is printed in the chat.

### S9. Promote the valued analyses to notebook 04
Planned separately after the user's review of S5: it changes the notebook and replaces the final
MLflow run.

## 8. Session prompts

Each session gets the opening, a blank line, then its sub-task's text.

**The opening:**

```text
We are writing docs/PROJECT_SUMMARY.md on branch docs/project-summary. Read
docs/PROJECT_SUMMARY_PLAN.md first: §1 decisions, §2 the outline, §3 sources and the doc/code
disagreements, §4 facts (leads to re-verify, not sources), §5 workflow, §7 your sub-task, §9 the
record. Rules: do only this sub-task; code wins over docs; write for a technical reader new to
the repo, with no implementation details in the summary (P11); scripts and outputs go in .mlflow-check/summary/ and run with
~/.venvs/age-group-prediction/bin/python; run each check once; change nothing under src/ or
notebooks/; do not commit. If a plan assumption breaks, stop and show the evidence. At the stop:
show the diff and the check's output, tick the sub-task's "done when" boxes and add its entry to
§9, write the commit message in a text block (no Co-Authored-By line), and print the next
sub-task's prompt (this opening, a blank line, its text) in one code block, naming the model to
select.
```

**Per sub-task:**

| Sub-task | Text after the opening |
|---|---|
| S1a | `Implement S1a: the skeleton and section 1 (task and settings), with check_s1a.py.` |
| S1b | `Implement S1b: section 2 (the data generation process), with check_s1b.py comparing the parameter table with configs/simulation.yml. Run an independent review subagent before the stop.` |
| S1c | `Implement S1c: section 2 rewritten for the revised simulator (docs/SIMULATOR_REALISM_PLAN.md), with check_s1c.py. Run an independent review subagent before the stop.` |
| S2 | `Implement S2: section 3 (preprocessing), with check_s2.py printing each candidate transformer's output columns. Run an independent review subagent before the stop.` |
| S3a | `Implement S3a: sections 4.1–4.2 (the exposure treatment and Model 1), with check_s3a.py for the Poisson rate/offset equivalence. Run an independent review subagent before the stop.` |
| S3b | `Implement S3b: sections 4.3–4.4 (Model 2 and the Model 3 placeholder), with check_s3b.py for the reconciliation. Run an independent review subagent before the stop.` |
| S4a | `Implement S4a: section 5, part 1 (split, CV, tuner, objective, studies, search space), with check_s4a.py.` |
| S4b | `Implement S4b: section 5, part 2 (metrics, bootstrap, selection, test evaluation, MLflow). Run an independent review subagent before the stop.` |
| S5a | `Implement S5a: s5a.py only, no summary text. Rebuild the two best models, write the score tables and the best and worst buildings, and show the tables.` |
| S5b | `Implement S5b: s5b.py only, no summary text. Error and bias by size, cohort and feature bins, and the rate per part; show the tables.` |
| S5c | `Implement S5c: s5c.py only, no summary text. The three-arm n_apartments permutation test; show the arms. Run an independent review subagent before the stop.` |
| S5d | `Implement S5d: s5d.py only, no summary text. The noise floor from the simulator's expected counts, with the derivation in §9. Run an independent review subagent before the stop.` |
| S5e | `Implement S5e: s5e.py, the matplotlib PNGs into docs/figures/; show them so I can choose which stay.` |
| S6a | `Implement S6a: section 6, part 1 (runs, best runs, scores, the difference), with check_s6a.py.` |
| S6b | `Implement S6b: section 6, part 2 (permutation, SHAP, best and worst buildings, biases), descriptive only, with check_s6b.py. Run an independent review subagent before the stop.` |
| S7a | `Implement S7a: sections 7.1–7.2 (which model when, why the models fail). Every claim cites a number. Run an independent review subagent before the stop.` |
| S7b | `Implement S7b: sections 7.3–7.4 (n_apartments, future steps). Every claim cites a number. Run an independent review subagent before the stop.` |
| S8a | `Implement S8a: the README entries, the link check (check_s8a.py) and the render check.` |
| S8b | `Implement S8b: review the whole summary against the code as a new reader; reproduce each finding before fixing it; then write the PR description.` |

## 9. Implementation record

### S0 (2026-11-23, the planning session)
- Branch `docs/project-summary` created from `main` at `bbe232b`.
- Offloaded files under `src`, `notebooks`, `configs`, `docs`, `.mlflow-check`: 0.
- Env: `kaleido` not installed; `plotly` 7.1.0, `matplotlib` 3.11.1, `shap` 0.53.0. Decision P9.
- This session ran on Fable 5.2, not Sonnet 5.5: it held the exploration's findings that §3 and
  §4 record.

### S1a (2026-11-23)
- Created `docs/PROJECT_SUMMARY.md`: title, scope note, section 1, headings only for sections 2-7
  (4.1-4.4, 7.1-7.4).
- `.mlflow-check/summary/check_s1a.py` (output in `check_s1a.out`) mirrors notebook 03's overrides
  (150 neighborhoods, rate 9.0), simulator, `ShareTransformer` and `Splitter("random")`. Output: 1,529
  buildings, 150 neighborhoods, 61,317 apartments, seed 42, split 917 / 306 / 306; raw columns
  `building_id, neighborhood_id, ses, avg_household_size, median_age, n_daycares_500m,
  school_status, 3_rooms..6_rooms, n_apartments, n_kindergarten, n_elementary, n_highschool`; after
  shares `4_rooms_share, 5_rooms_share, 6_rooms_share` replace the room counts. All as §4 expected.

### S1b (2026-11-23)
- Wrote section 2 of `docs/PROJECT_SUMMARY.md` with sub-sections 2.1 notation, 2.2 neighborhoods,
  2.3 buildings and room mix, 2.4 apartments and transforms, 2.5 total children, 2.6 cohort split,
  2.7 aggregation (the conditional expectation, proportional to `n_apartments` at fixed shares),
  2.8 what is hidden, 2.9 parameters (67 rows), 2.10 the dataset's columns. Formulas are in
  symbols; the values are in 2.9.
- Code facts not in §4: household size and daycares center age at `neighborhood.median_age_mean`,
  the outcome models at `features.median_age_reference`. The room-tilt pivot 4 is a literal
  (`features.py:107`), not `room_reference`. $A_b$ is rounded with `np.rint` before clipping.
  Notebook 01 uses `from_config`; notebooks 02–04 build the five components by hand (same result).
- Each formula's code: $S_j$ `neighborhood.py:86-96`; $M_j$ `:98-113`; $H_j$ `:115-136`;
  $D_j$ `:138-155`; $Z_j$ `:157-178`; $B_j$ `building.py:102-105`; $A_b$ `:154-165`;
  $q_j$, $\omega_{jr}$ `features.py:84-109`; Dirichlet and multinomial `building.py:210-220`;
  permutation `apartment.py:50`; transforms `features.py:8-43`, `outcomes.py:180-201`;
  $u_b$, $w_j$ `outcomes.py:90-99`; $\log\mu_i$ `outcomes.py:123-178`; NB2 `outcomes.py:224-234`;
  $v_b$ `outcomes.py:278-282`; $\theta_i$ `outcomes.py:306-324, 344-387`; $p_{ik}$
  `outcomes.py:326-342`; cohort multinomial `outcomes.py:295-300`; $C_{bk}$ `pipeline.py:39-71`.
- `.mlflow-check/summary/check_s1b.py` (output `check_s1b.out`), one run:
  (1) the parameter table against `configs/simulation.yml`, with the overrides read from notebook
  03's source: 67 rows, 0 mismatches. Every YAML leaf has a row; the two override lists are
  asserted empty and excluded. (2) The column table against the simulated dataset: equal names and
  order, 1,529 rows. (3) Beyond the plan: a numpy redraw of the dataset from section 2's formulas,
  in its stated order, with the values parsed from the doc's table. It is exactly equal to the
  simulator's dataset in every column and in $u_b$, $w_j$, $v_b$. (4) Beyond the plan: 2.7's
  closed form against $\sum_i \mu_i p_{ik}$: max abs diff 3.2e-14. The mean expected building
  total is 24.169 against 24.153 observed.
- Review (independent subagent): no errors in the formulas, values, draw order, 2.7 or 2.8. It
  reported 13 points; each one was reproduced against the code or the doc text, then fixed:
  (1) not all notebooks call `from_config`: reworded; (2) SES also acts through the daycares and
  the room mix; (3) clip caps values, it does not remove them: meanings and the notation row;
  (4) the reference neighborhood behind $\beta_0$ stated, and "per unit of its term";
  (5) $B_j$ added to the notation table, and "every other symbol is a constant" corrected;
  (6) the $s$ and $c$ clashes in $\mathcal N$ and clip; (7) the code name for $(x)_+$ is
  `np.maximum`; (8) $\tilde H_j$ defined once, and $h_0$ vs $H_{\mathrm{ref}}$ noted;
  (9) $\bar\eta_{jr}$, $\bar\theta_{jr}$ for the effect-free terms; (10) $x_b$ defined, and the
  effects' independence of $x_b$ stated; (11) the cohort expectation given the features
  written out, with only the integral over $v$ open; (12) the noise statement made precise;
  (13) `\{` in inline math replaced by `\lbrace`, and the inline formula with `\,` moved into a
  display block. The fixes changed only prose and the meaning cells of 2.9, which the check does
  not parse, so the check was not rerun.
- Length: section 2 is about 400 lines, 67 of them the parameter table. The whole document may
  pass P1's 600–900 lines; S8b decides.
- **Revision (2026-11-24, after the user's review; decision P11):**
  - The opening paragraph was replaced by a description and motivation of the levels, in draw
    order: neighborhood (the context); building (count, size, room mix); apartment (where the
    children are generated); building sums (the prediction level). It also says why the
    simulation runs per apartment and then sums.
  - The notation table and the 67-row parameter table were removed. Every formula now carries its
    values inline. Distributions are labeled ($\mathcal N(\mu=0, \sigma^2=1^2)$); linear formulas
    use plain coefficients explained in words.
  - Every implementation detail was removed (code calls, file and line references, config keys,
    notebook names), as was the link to `SIMPLIFIED_MODEL_PLAN.md`. Section 2 is now 2.1
    neighborhoods, 2.2 buildings, 2.3 apartments, 2.4 building sums and expectations, 2.5 what
    is hidden, 2.6 the dataset's columns, about 345 lines. The column table keeps its Symbol
    column, which maps each column to its formula.
  - `check_s1b.py` was rewritten and run once:
    - (1) 78 numbers in 27 formulas, read where the doc writes them, match the config plus the
      notebook's overrides, read from notebook 03's source: 0 mismatches. 67 of 69 config keys
      are covered; the 2 empty override lists are asserted empty.
    - (2) The column table equals the dataset's columns.
    - (3) 0 implementation-detail hits, and no backticked token other than a dataset column.
    - The first run's redraw was not repeated: the formulas are unchanged.
  - Second review (independent subagent): no errors in the formulas, values, draw order,
    expectations or hidden-variable claims. It raised 16 points:
    - Reproduced and fixed (13):
      - (1) three hidden effects, not two kinds.
      - (2) the 1.35 holds at 4 rooms and the room multipliers at household size 2.6; the
        interaction's 1.22 and 1.65 were added.
      - (3) the tilt also lowers the 4-room share.
      - (4) the $q_j = 1$ example lay outside the data (realized −0.57 to 0.60); it is now
        $q_j = 0.5$ (multipliers 0.78/1/1.28/1.65, shares 22/36/27/16%).
      - (5) $A_b$ is independent of the neighborhood features, not of the room counts.
      - (6) SES reaches the score indirectly.
      - (7) $w_j$ belongs to the neighborhood.
      - (8) $x_b$ defined.
      - (9) NB2 explained.
      - (10) the averaged factor is the constant 1.053.
      - (11) "building by building".
      - (12) a motivation for school status and for daycares in the score.
      - (14) the link to an outdated doc dropped.
    - Not reproduced (1): (13) "section 7.3 points nowhere". The heading exists; the forward
      references go to S8b's link check.
    - Kept by choice (2): (15) the Symbol column, flagged for the user; (16) `\mathbb 1`
      renders as a plain 1 in MathJax, which is acceptable, and the check's templates read it.
  - None of these fixes touched a formula block that the check reads, so the check was not rerun.
- **Paused (2026-11-24).** The user's notes on the revised section 2:
  - Opening: the two building bullets merged into one ("Building: the level we predict"); the
    "why simulate apartments" paragraph replaced by its hidden-effects sentence. Applied.
  - Section 2.1: the explanations are not good enough, and the neighborhood model itself was
    questioned:
    - household size against median age and SES;
    - daycares against age and SES;
    - the order of the draws.
  - Two Fable consults:
    - The negative age slope of household size is right: the median age counts the children.
    - Household size should follow a reverse-J in SES.
    - Daycares should follow demand (household size, young age) and purchasing power (SES).
    - Demographics come before facilities.
  - The user chose a realistic, Israel-calibrated simulator first (P12). The design, the values
    and the sub-tasks are in `docs/SIMULATOR_REALISM_PLAN.md`. Sections 2.1–2.6 are rewritten in
    S1c; the notes are in §7 S1c.
