# Project summary document: the plan

> **Status (2026-11-23):** plan approved; S0 done (branch `docs/project-summary`, this document,
> the environment checks). S1a, S1b done. Paused (2026-11-24) for the simulator revision (P12);
> its step 1 (neighborhoods) is merged (PR #29) and `main` is merged into this branch.
> **The order is now P13: section 2 part by part, S1c.1–S1c.5, each part analyzed for realism
> before it is written.** S1c.1 done (2.1 written). The simulator's step 2 (buildings and the room
> mix) is merged (PR #30) and `main` is merged into this branch. S1c.2 done (2.2 written; the
> children count gets a simulator step: a stronger household-size relation and a cap). The
> simulator's step 3 (children per apartment, the building-size scales) is merged (PR #31) and
> `main` is merged into this branch. S1c.3 done (2.3's first half written).
> The cohort split gets a
> simulator step: Israel's cohort levels, a planned-school term, the negligible bonus dropped.
> **Next: the simulator's step 4 (planned from `main`), then S1c.4.** The implementation record
> is §9.

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
| P12 | The realistic simulator first (user, 2026-11-24). Section 2 questioned the neighborhood model; the user chose to make the simulator realistic, calibrated to Israel, step by step (neighborhoods first), in separate PRs from `main` (`docs/SIMULATOR_REALISM_PLAN.md`). This branch merges `main` after each simulator PR. The order becomes: simulator step(s) → S1c and S2–S4b (the documentation) → the reruns (notebooks 03 and 04 on the new data) → S5a onward. The results of the run at `da6afc9` (§3) are superseded then. **The order of the simulator steps and S1c is replaced by P13;** the rest holds. |
| P11 | The reader's view (user, 2026-11-24), for the whole summary: no implementation details (no Python names or calls, file paths, line numbers, config keys or notebook names; dataset column names and links to the detailed docs stay). Formulas carry their actual values inline, with no notation or parameter tables: distributions labeled, e.g. $\mathcal N(\mu=0, \sigma^2=1^2)$, linear formulas with plain coefficients explained in words. Each section opens by describing and motivating its subject. Code lines go only into §9, the record. |
| P13 | Section 2 part by part (user, 2026-11-24); replaces P12's order. Section 2 is written one part at a time, in draw order: 2.1 neighborhoods, 2.2 buildings, 2.3 total children, 2.3 the cohort split, then 2.4–2.6 and section 1's counts (S1c.1–S1c.5). Each part is written for the simulator as it stands when its sub-task starts. Before a part is written it is analyzed with the user: is this model realistic for Israel? The user decides: **keep**, or **a simulator step** (planned in `docs/SIMULATOR_REALISM_PLAN.md`, its own PR from `main`; this branch merges `main` after it). A part is written only after its decision has taken effect, so no part is written twice. 2.1 needs no analysis: step 1 was its step. After section 2: S2–S4b, the reruns, S5a onward, as in P12. |

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
These facts predate the simulator's step 1 (PR #29). The neighborhood model and five downstream
values have changed since (`docs/SIMULATOR_REALISM_PLAN.md` §2–§3), and the counts are now 1,523
buildings and a 913 / 305 / 305 split. Later steps may change more.
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
- No file under `src/` or `notebooks/` changes in S0–S8b, so the test suite is not rerun. A
  simulator step is never done on this branch.
- **The section 2 loop (P13), per sub-task S1c.k:** write part k, run the check, run the review,
  then analyze part k+1 with the user. The user decides: keep, or a simulator step.
  - **Keep:** the decision is recorded in §9, and the session prints S1c.k+1's prompt.
  - **A step:** the findings are recorded in §9, and the session prints the step's planning
    prompt (§8) with the findings in it. The step is planned and built from `main`, in its own
    PR. After it merges, the user merges `main` into this branch, and S1c.k+1 starts.

## 6. Sub-tasks

| # | Sub-task | Model | Review | Done |
|---|---|---|---|---|
| S0 | Setup: branch, plan doc, environment checks | Sonnet 5.5 (done by the planning session) | no | ✅ |
| S1a | Skeleton + section 1 (task and settings) | Sonnet 5.5 | no | ✅ |
| S1b | Section 2 (data generation process) | Opus 5.5 | yes | ✅ |
| S1c.1 | Section 2.1 (neighborhoods, the step-1 model); then the analysis of 2.2 (buildings per neighborhood, apartments per building, the room mix) | **Fable 5.2** | yes |✅ |
| S1c.2 | Section 2.2 (buildings); then the analysis of 2.3's first half (the hidden effects, the children count per apartment) | **Fable 5.2** | yes | ✅ |
| S1c.3 | Section 2.3, first half (the scaled values, the hidden effects, total children); then the analysis of the cohort split | **Fable 5.2** | yes | ✅ |
| S1c.4 | Section 2.3, second half (the cohort split); no analysis follows | Opus 5.5 | yes | |
| S1c.5 | Sections 2.4–2.6, the section's opening, section 1's counts, every seed-42 number | Opus 5.5 | yes | |
| S2 | Section 3 (preprocessing); starts after S1c.5 | Opus 5.5 | yes | |
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

Why Fable on S1c.1–S1c.3 (P13): each holds a realism analysis, which is the Fable consult the
realism plan requires per step (its R2), with real-world targets from Fable's memory (its R10). A
wrong verdict costs a whole simulator PR, or leaves an unrealistic model in. One Fable session
also avoids relaying the discussion with the user through a subagent. S1c.4 and S1c.5 only write
against the code, with a check and a review. The review subagent is Opus 5.5 in all five.

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

### S1c. Section 2, part by part (P13)
Five sub-tasks, S1c.1–S1c.5, in the loop of §5. The rules common to all five:
- **The user's notes (2026-11-24)** apply to every part:
  - The explanations must be better throughout; each variable says what it represents.
  - Each relation is motivated.
  - No explanation of what clipping does.
  - P10 and P11 hold.
- **One check script, `check_s1c.py`,** adapted from `check_s1b.py`. It grows by one part per
  sub-task and always checks every part written so far: each inline value against the config plus
  the notebooks' overrides, and no implementation detail. A later simulator step that changes an
  earlier part's value (as step 1 changed five downstream values) therefore fails the check, and
  that part is corrected in the same sub-task.
- **Population statements** (a mean, a correlation, a profile over bins) are checked on a large
  draw. **Numbers that depend on the seed-42 dataset** (building and apartment totals, realized
  ranges) are not written before S1c.5, when the simulator is final.
- Each sub-task updates its level's bullet in section 2's opening.
- **The analysis of the next part** reads that part's code and its current section text, and
  takes its row in `docs/SIMULATOR_REALISM_PLAN.md` §7 as leads. Per component it shows what the
  simulator does, what Israel looks like (with a confidence), what a change would do to the
  benchmark, and a recommended verdict. The user decides.
- One commit per sub-task, after the decision is recorded in §9.

### S1c.1. Section 2.1, neighborhoods; then the analysis of 2.2 (Fable 5.2, review)
Write 2.1 for the step-1 model: the five draws in the order of the realism plan's §2. Notes for
this part:
- SES: the neighborhood's low or high socioeconomic class, on a national index (relative to the
  whole population, not to the city: a global feature).
- Median age: the median age of the residents of neighborhood $j$.
- Why household size falls with median age and follows a reverse-J in SES, why daycares follow
  demand and purchasing power, why school status follows the demographics.

Leads for the analysis of 2.2: large-household, low-SES areas get larger apartments, where in
Israel they are crowded; building size may depend on the area.
Done when:
- [x] Each of the five draws has its formula with its values inline and its motivation.
- [x] `check_s1c.py` passes on 2.1.
- [x] Review findings reproduced and fixed.
- [x] The analysis of 2.2 is shown, and the user's decision is recorded in §9.

### S1c.2. Section 2.2, buildings; then the analysis of the children count (Fable 5.2, review)
Starts after 2.2's decision has taken effect (if a step: its PR merged into this branch). Write
2.2 for the simulator as it then is: buildings per neighborhood, apartments per building, the
room mix.
Leads for the analysis of 2.3's first half: the negative binomial's tail, the zero share (0.52
against about 0.60 real), the count too flat at small households.
Done when:
- [x] Each draw of 2.2 has its formula with its values inline and its motivation.
- [x] `check_s1c.py` passes on 2.1–2.2.
- [x] Review findings reproduced and fixed.
- [x] The analysis of the children count is shown, and the user's decision is recorded in §9.

### S1c.3. Section 2.3, total children; then the analysis of the cohort split (Fable 5.2, review)
Starts after the children count's decision has taken effect. Write the first half of 2.3: the
scaled values, the hidden effects of the building and the neighborhood, the expected count and
its draw.
Leads for the analysis of the cohort split: reference shares 40/35/25% against a real 27/38/35%
for ages 3–6 / 7–12 / 13–18; the household-size coefficient pushes large-household areas toward
high school, the opposite of Israel.
Done when:
- [x] Each term of the expected count is explained and motivated.
- [x] `check_s1c.py` passes on 2.1 through the first half of 2.3.
- [x] Review findings reproduced and fixed.
- [x] The analysis of the cohort split is shown, and the user's decision is recorded in §9.

### S1c.4. Section 2.3, the cohort split (Opus 5.5, review)
Starts after the cohort split's decision has taken effect. Write the second half of 2.3: the
building's second hidden effect, the lifecycle score, the cohort probabilities and their draw. No
analysis follows: 2.4–2.6 are derived from the parts above, not modelled.
Done when:
- [ ] Each term of the score is explained and motivated.
- [ ] `check_s1c.py` passes on 2.1–2.3.
- [ ] Review findings reproduced and fixed.

### S1c.5. Sections 2.4–2.6, the opening, section 1's counts (Opus 5.5, review)
Rewrite 2.4 (the sums and the expectations, their constants re-derived), 2.5 (what is hidden) and
2.6 (the columns). Re-read section 2's opening against the finished parts. Add the seed-42
numbers left out of S1c.1–S1c.4, and correct section 1's counts and split sizes (stale at
1,529 / 61,317 and 917 / 306 / 306).
Done when:
- [ ] `check_s1c.py` passes on all of section 2, and now also checks: the column table against
      the dataset, that every config key is covered, section 1's counts and split sizes, and
      every seed-42 number.
- [ ] 2.4's closed form is checked against the sum over the apartments' expected counts.
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
| S1c.1 | `Implement S1c.1: write section 2.1 (neighborhoods) for the step-1 model in docs/SIMULATOR_REALISM_PLAN.md §2–§3, with check_s1c.py (adapted from check_s1b.py) covering 2.1. Read §7 "S1c" first for the rules common to S1c.1–S1c.5. Run an independent review subagent (Opus 5.5). Then analyze section 2.2 with me (buildings per neighborhood, apartments per building, the room mix): is it realistic for Israel? Record my decision in §9 and print the prompt it leads to.` |
| S1c.2 | `Implement S1c.2: write section 2.2 (buildings) for the simulator as it now is, with check_s1c.py extended to 2.2. Read §7 "S1c" first for the common rules and §9 for 2.2's decision. Run an independent review subagent (Opus 5.5). Then analyze the first half of section 2.3 with me (the hidden effects, the children count per apartment): is it realistic for Israel? Record my decision in §9 and print the prompt it leads to.` |
| S1c.3 | `Implement S1c.3: write the first half of section 2.3 (the scaled values, the hidden effects, total children per apartment) for the simulator as it now is, with check_s1c.py extended to it. Read §7 "S1c" first for the common rules and §9 for the decision. Run an independent review subagent (Opus 5.5). Then analyze the cohort split with me: is it realistic for Israel? Record my decision in §9 and print the prompt it leads to.` |
| S1c.4 | `Implement S1c.4: write the second half of section 2.3 (the cohort split) for the simulator as it now is, with check_s1c.py extended to it. Read §7 "S1c" first for the common rules and §9 for the decision. Run an independent review subagent before the stop.` |
| S1c.5 | `Implement S1c.5: rewrite sections 2.4–2.6, re-read section 2's opening, add the seed-42 numbers and correct section 1's counts, with check_s1c.py extended to the columns, the config-key coverage and the counts. Read §7 "S1c" first for the common rules. Run an independent review subagent before the stop.` |
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

**A simulator step's planning session (P13).** Printed by an S1c session when the user decides a
part needs a step. It is a prompt of its own, without the opening above. Model: Fable 5.2. The
session fills in the three bracketed fields.

```text
Plan step <n> of the realistic simulator (<component>) in docs/SIMULATOR_REALISM_PLAN.md, on a
new branch from main (R9). Read its §1 (decisions), §7 (this step's known issues) and §8–§10
(how step 1 was split, prompted and recorded). The analysis that decided this step, from the
project summary's session: <findings>. Run a Fable design consult on the design and its values,
then write the step's plan in step 1's form: the model, the downstream consistency, the config,
the tests, the sub-tasks with a model each, the session prompts and a record entry. Do not
implement and do not commit; stop for my review of the plan.
```

After the step's PR is merged: merge `main` into `docs/project-summary`, then start the next
S1c sub-task with its prompt above.

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

### Re-plan (2026-11-24, Fable 5.2; decision P13)
- The simulator's step 1 (neighborhoods) is merged as PR #29 (`876f045`), and `main` is merged
  into this branch.
- The user changed the order: section 2 is written part by part, and each part is analyzed for
  realism in Israel before it is written (P13). P12's "all simulator steps, then S1c" is replaced.
- S1c is split into S1c.1–S1c.5 (§6, §7), with a prompt each and a prompt for a simulator step's
  planning session (§8). The loop is in §5.
- `docs/SIMULATOR_REALISM_PLAN.md`: its status line and a note above its §7 now point to P13.
- No summary text was written.

### S1c.1 (2026-11-24, Fable 5.2; review Opus 5.5)
- **Section 2.1 written** for the step-1 model, and the neighborhood bullet of section 2's
  opening. The five draws in order, each with what it represents, its formula with the values
  inline and its motivation. The reference neighborhood (SES 0, median age 31, household size
  3.0) is introduced there.
- **The user's choices** (asked before writing):
  - formulas only: no population statistics (means, bands, correlations) in 2.1;
  - Israel named, and the lowest-SES areas as largely Haredi and Arab communities;
  - no real-world figures.
- **Symbols:** the noises are $\varepsilon^{M}_j$ and $\varepsilon^{H}_j$; the school weights
  are $\psi_{jz}$ ($\omega$ is taken by the room shares of 2.2).
- **Each formula's code:** $S_j$ `neighborhood.py:93-108`; $M_j$ `:110-126`; $H_j$ `:128-150`;
  $D_j$ `:152-169`; $Z_j$ `:171-204`.
- **`.mlflow-check/summary/check_s1c.py`** (output `check_s1c.out`), covering the opening and 2.1:
  - 35 numbers in 10 formulas against the config plus notebook 03's overrides: 0 mismatches;
    25 of 25 `neighborhood` keys covered.
  - Derived numbers: the six $e^x$ multipliers; the combined SES effect on household size
    ($-0.41$, $0.18$, minimum 1.139, written 1.1); the direct minimum (0.556, written 0.6); the
    daycare log-rate shift at SES −2.5 over all paths (+0.033, "roughly cancel"). 0 problems.
  - 0 implementation-detail hits, no backticked token, no explanation of clipping.
  - It ran twice: the review's fixes changed checked sentences and added two derived numbers.
- **Review (independent subagent, Opus 5.5):** no error in a formula, a value or the draw order.
  - Reproduced and fixed (3):
    - "it cannot fully separate their effects" overreached: the three features are correlated
      (about 0.6–0.75), not collinear. Now "harder to estimate than if they were independent".
    - "grow slightly again at the top" was said of the direct SES terms, where the rise is 40% of
      the fall. The direct minimum (0.6) and the combined reverse-J (minimum 1.1) are now stated
      separately.
    - "the daycares follow the population" ignored purchasing power, which at the lowest SES
      cancels the demand. The bullet, the daycare paragraph and the closing say so now.
  - Clarity points applied (4): the multipliers hold with the other values fixed; why a young
    neighborhood gets a planned school; why school status has no SES term; "the formulas of this
    section".
  - Expected, left for S1c.2–S1c.3: 2.2 and 2.3 still center on 2.6 and 37.
- **The analysis of 2.2** (`s1c1_analysis_2_2.py`, 400,000 neighborhoods; real-world figures from
  Fable's memory, not looked up):

  | Component | The simulator | Israel (confidence) |
  |---|---|---|
  | Buildings per neighborhood | $1 + \mathrm{Poisson}(9)$ | a sample-size setting, not a model |
  | Apartments per building | normal, mean 40.0, SD 11.9, skew 0.06, the same everywhere | right-skewed: many buildings of 8–24 apartments, few towers (medium); alike within a neighborhood (high); larger in younger, newer neighborhoods (low) |
  | Room mix | mean rooms 4.27 / 4.16 / 4.13 / 4.17 / 4.28 over five SES bands; correlation with household size 0.73, with SES 0.04; people per room 1.04 / 0.85 / 0.73 / 0.67 / 0.67 | apartment size rises with SES, and the poorest areas do not have the largest apartments (medium); about 1.3 people per room in Haredi and Arab areas (medium) |
  | Base shares | 30 / 38 / 22 / 10% for 3–6 rooms | 10–13% of homes have 1–2 rooms; 6 rooms nearer 3–5% in apartment buildings (medium) |
  | Between buildings | a share's SD 0.05–0.09 | probably larger (low) |

- **The user's decision (2026-11-24): a simulator step for 2.2**, covering the room mix and the
  building size. The room categories stay 3–6, with a lower 6-room share. Buildings per
  neighborhood and the between-building variation are kept. S1c.2 waits for that step's PR to be
  merged into this branch.

### S1c.2 (2026-11-24, Fable 5.2; review Opus 5.5)
- **Before the start:** the merge of `main` (`4a818da`, PR #30) was open, with one conflict in
  the status line of `docs/SIMULATOR_REALISM_PLAN.md`. It was resolved with `main`'s line (the P13
  pointer stays in the note above that plan's §7) and committed as `5c1cde1`.
- **Section 2.2 written** for the step-2 model, with the building bullet and the hidden-values
  paragraph of section 2's opening. In draw order: buildings per neighborhood, the two hidden
  neighborhood values, the hidden building type, apartments per building, the room tilt, the
  expected shares, the Dirichlet and the multinomial.
- **The user's choices** (asked before writing): the same as 2.1 (formulas only, Israel named, no
  real-world figures).
- **Symbols:** $\rho_j$ the tower propensity, $\tau_j$ the texture, $T_b$ the type, $m_b$ the
  type's median, $\varepsilon^{A}_b$ the building's deviation, $q_b$ the tilt (now per building),
  $\omega_{br}$ the expected shares (now per building). $\tilde M_j$ and $\tilde H_j$ are defined
  here on the new references (31 and 10; 3.0 and 0.75).
- **Each formula's code:** $B_j$ `building.py:109-128`; $\rho_j$, $\tau_j$ `:185-194`; $T_b$
  `:195-208`; $A_b$ `:209-224`; $q_b$ `:241-259` and `features.py:113-117`; $\omega_{br}$
  `features.py:118-120`; the Dirichlet and the multinomial `building.py:265-275`.
- **`check_s1c.py`** extended to 2.2 and run once, after the review's fixes (output
  `check_s1c.out`):
  - 62 numbers in 22 formulas against the config plus notebook 03's overrides: 0 mismatches; 50
    of 50 `neighborhood` and `building` keys covered (the two override lists asserted empty).
  - Derived numbers: 12 $e^x$ statements; the reference tower chance (0.0871, written 9%); the
    tilt change of a doubled building (0.2218, written 0.22) and of a tower (0.35); 12 statements
    in words; the worked example at SES 1 (multipliers 0.84 / 1 / 1.19 / 1.42, shares
    25 / 40 / 28 / 7%, computed with the simulator's own function). 0 problems.
  - 0 implementation-detail hits, no backticked token, no explanation of clipping.
  - New part 4: the 12 qualitative claims of 2.2 as signs on a large draw (20,000 neighborhoods,
    199,155 buildings): all hold. Size mean 40.7, median 32, skew 2.19; correlation with SES
    +0.19, with median age +0.14; intraclass correlation of log size 0.28; mean rooms 3.87 where
    SES is under −1.5 against 4.04 elsewhere. None of these numbers is in the text.
- **Review (independent subagent, Opus 5.5; its script in `review_s1c2/`):** no error in a
  formula, a value, a symbol or the draw order.
  - Reproduced and fixed (9):
    - "the strongest term, 0.35 per SES unit" compared coefficients in different units. Now the
      term's range over the SES scale (−0.875 to 0.875), and a tower's size term (0.35) against it.
    - "more than any single feature does" (the propensity) fails over SES's whole range. Now
      "more than one SES unit or a planned school does".
    - "Large poor households live in small apartments" claimed a household-level match that the
      model does not make. Now a statement about neighborhoods.
    - The worked example's tilt of 0.35 needs household size 3.0, which SES 1 does not give on
      average (2.77). The example now states both values.
    - $\tau_j$ was called a size; it is a log shift. Its factor per standard deviation (1.22) and
      the building deviation's (1.25) were added.
    - "Dividing by 2 keeps the shift gradual" replaced by the reading of $e^{q_b}$ and
      $e^{-q_b/2}$; the size term got a motivation.
    - The size term takes back part of the SES term at high SES: one sentence added.
    - The opening's bullet: the type sets the typical size, not the size; household size added.
    - "a long tail" is now "a smaller group of towers around 90".
  - Kept by choice (2): "6 to 10 floors" is the user's definition of a texture building (realism
    plan R12), not a statistic; `\mathbb 1` as in S1b.
  - Left for S1c.3: section 2.3 still defines $\tilde M_j$ on 37 and says $\tilde H_j$ uses 2.6,
    which now contradicts 2.2.
- **The analysis of 2.3's first half** (`s1c2_analysis_2_3.py`, the full simulator on 1,000
  neighborhoods at rate 9, seed 42: 410,300 apartments; it ran twice, the first output had a
  truncated table; real-world figures from Fable's memory, not looked up):

  | Component | The simulator | Israel (confidence) |
  |---|---|---|
  | Level | 0.89 children aged 3–18 per apartment | about 0.9–0.95 per household (medium) |
  | Zero share | 0.50 | about 0.60 (medium) |
  | Family size | 1.79 per apartment with children; 57% of them hold one child | about 2.4; one-child households nearer a third (medium) |
  | Tail | maximum 40; 1.0% hold 6 or more, 0.3% 8 or more, 0.11% 10 or more | 6 or more at about 1% is plausible (low); above 10–12 does not occur (high) |
  | Household size | 0.45 / 0.61 / 0.72 / 0.89 / 1.09 / 1.39 / 2.09 children at household size 2.05 / 2.43 / 2.82 / 3.23 / 3.70 / 4.21 / 4.98; slope 0.51 per person | size minus the adults: about 0.1–0.2 at 2.05, 1.0 at 3.2, 2.5–2.7 at 5.0; slope about 0.8–0.9 (high for the direction, medium for the levels) |
  | Rooms | 0.54 / 0.87 / 1.27 / 1.54 at 3 / 4 / 5 / 6 rooms | a 2–3 times gap is plausible (low to medium) |
  | Rooms × household size | where household size is over 4.5: 0.97 at 3 rooms, 5.78 at 6 rooms | an average near 6 is not plausible (medium) |
  | Hidden effects | SD 0.25 and 0.20; 35% of the variance of the log expected count; a building's total against its expectation: relative RMSE 0.40 from the features alone, 0.23 with the effects known | they exist (medium); their size cannot be calibrated from memory (low) |
  | Daycares, school status, median age | each at most a factor of 1.11–1.12 across its range | plausible as small associations (low) |

  - With the same expected counts, a larger dispersion lowers the zero share (0.48 at $\phi = 6$,
    0.46 for a Poisson draw), so the tail and the zero share need different fixes.
- **The user's decision (2026-11-24): a simulator step for the children count**, kept at the
  apartment level (the user asked whether to model the building directly; children belong to
  households, the room effect is an apartment's, and the sums give the building's behaviour).
  Its scope, kept simple at the user's request: **a stronger household-size relation** (children
  follow household size minus the adults; the rooms × household-size interaction is dropped if
  the stronger term makes it unnecessary) **and a cap on the count**. Not in the step: the zero
  share and family size (invisible in building sums; a two-part draw would be one more model
  part), the hidden effects' sizes, the main room effects, the daycare, school-status and
  median-age terms. S1c.3 waits for that step's PR to be merged into this branch.
- **A second decision, on 2.2's building size (the user's note, 2026-11-24):** the texture shift
  (SD 0.20, the neighborhood) and the building's deviation (SD 0.22) have about the same scale,
  so only 45% of the size variation of texture buildings is between neighborhoods, and texture
  buildings inside one neighborhood range from about 21 to 43 apartments. The building's
  deviation should be the smaller one: a neighborhood's buildings share an era, a plan and plot
  sizes, and the towers already give the contrast inside a neighborhood. **Decided: changed
  inside step 3**, to about 0.25 (neighborhood) and 0.15 (building), which keeps the combined
  log spread (0.29 against 0.30) and puts 74% of it between neighborhoods (a range of 23–38
  inside one). The figures are from the formula; the step's consult confirms the values on a
  draw. Section 2.2 is committed for the simulator as it is (0.20 with 1.22, 0.22 with 1.25);
  S1c.3's check fails on those two values and corrects them.

### S1c.3 (2026-11-25, Fable 5.2; review Opus 5.5)
- **Before the start:** the simulator's step 3 is merged (PR #31, `3ab0360`) and `main` is merged
  into this branch (`f14c3da`). Two stale status notes were corrected: this plan's header and the
  step-3 note of `docs/SIMULATOR_REALISM_PLAN.md`.
- **Section 2.3's first half written** for the step-3 model, from the heading to "The cohort
  split", with the apartment bullet of section 2's opening. In draw order: the values used, the
  two hidden effects, the expected count, the draw and its cap, one bullet per term, the two
  absent terms. The cohort-split half keeps its S1b text until S1c.4.
- **Section 2.2 corrected** (the second decision of S1c.2): $\tau_j$ has SD 0.25 (factor 1.28)
  and $\varepsilon^{A}_b$ SD 0.15 (factor 1.16), with one sentence on what the order means.
- **The writing choices** of 2.1 and 2.2 were assumed, not asked again (stated in the session's
  plan, which the user approved): formulas only, Israel named, no real-world figures.
- **Symbols:** $L_j$ the household term (the realism plan's $x_j$; $x_b$ is taken by 2.4), and
  $N^{*}_i$ the count before the cap. The room effects stay as indicators with their values. The
  SES² term (coefficient 0) is not in the formula; the text says there is no direct SES term.
- **Each formula's code:** $L_j$ `features.py:18-42`; $c_i$, sat `features.py:55-70`; $u_b$,
  $w_j$ `outcomes.py:92-101`; $\log\mu_i$ `outcomes.py:128-227`; the draw `outcomes.py:229-239`;
  the cap `outcomes.py:119-121`; $\tau_j$ `building.py:189-194`; $\varepsilon^{A}_b$ `:214-216`.
- **`check_s1c.py`** extended to 2.3's first half and run once, after the review's fixes (output
  `check_s1c.out`):
  - 82 numbers in 30 formulas against the config plus notebook 03's overrides: 0 mismatches; 67
    of 67 `neighborhood`, `building` and `total_children` keys covered (the two override lists
    asserted empty, the SES² coefficient asserted 0).
  - Derived numbers: 22 $e^x$ statements; 18 statements in words; the powers by rooms
    (1.15 / 1.0 / 0.85 / 0.70); the 6-to-3-room ratios (2.5857, 3.9748, 1.8308, written 2.59,
    4.0, 1.8); the doubling and halving household sizes; the saturation example. 0 problems.
  - 0 implementation-detail hits, no backticked token, no explanation of clipping.
  - Part 4 (2.2's 12 claims, 20,000 neighborhoods): all hold. The intraclass correlation of log
    size is now 0.377 (0.279 before step 3).
  - New part 5: 8 qualitative claims of 2.3's first half on one full draw (1,000 neighborhoods,
    411,814 apartments): all hold. Children ÷ ($H$ − 1.7) at 4 rooms is 0.92–1.08 of the middle
    band; the 6-to-3-room ratio is 3.65 where household size is under 2.6 and 1.81 over 4.5; zero
    share 0.510 against 0.469 for a Poisson draw; maximum 10; the household term's SD over
    apartments is 0.465 against 0.321 for rooms. None of these numbers is in the text.
- **Review (independent subagent, Opus 5.5; its script and output in `review_s1c3/`):** no error
  in a formula, a value, a symbol, a derived number or the draw order.
  - Reproduced and fixed (13):
    - "$H_j - 1.7$ is the part of the household that children fill" overreached: children aged
      3–18 are about 60% of it (0.92 children at an excess of 1.52, the realism plan's 12.2). It
      is now the excess over a household without such children.
    - "the household members beyond the adults" contradicted the definition of 1.7, which holds
      younger and older children too. Reworded in the bullet and in the opening.
    - The crowding example at household size 5.0 lay outside the benchmark's data (realized 1.80
      to 4.57). It is now 4.5, with a ratio of 1.8.
    - "the few families with children live in the large apartments": now "are concentrated in",
      and the small ones "mostly" hold singles and couples.
    - The opening's "the small apartments are full too": now "hold many children too".
    - "many hold no children" tied the zeros to the overdispersion, which adds only 4 points
      (0.510 against 0.469). Now "somewhat more of them".
    - "Building size acts only through the room mix" holds inside a neighborhood. Across
      neighborhoods large buildings stand where households are smaller; both are now said.
    - The paths of SES: "mainly through the household size, and also through the other
      neighborhood values and the room mix".
    - Which step uses $\tilde M_j$ and which $\tilde H_j$.
    - 0.41 is the baseline, not a typical apartment.
    - "exactly proportional" holds with the other terms fixed.
    - 2.2: "smaller than the neighborhood's texture" compared two draws; now their standard
      deviations (0.15 against 0.25), and $\tau_j$ is called the shift.
    - 2.2: "every building there is $e^{\tau_j}$ times larger": now its size is multiplied.
  - Kept by choice (1): $\mu$ in $\mathcal N(\mu=0, \ldots)$ next to $\mu_i$; the label is P11's
    form.
  - Left for S1c.5: section 1's counts are stale (the draw now gives 1,523 buildings and 61,450
    apartments); the apartment order's permutation draw is not mentioned in section 2.
- **The analysis of the cohort split** (`s1c3_analysis_cohort.py`, the full simulator on 1,000
  neighborhoods at rate 9, seed 42: 411,814 apartments, 361,885 children; it ran three times,
  the first two stopped on a wrong attribute and a missing column; real-world figures from
  Fable's memory, not looked up):

  | Component | The simulator | Israel (confidence) |
  |---|---|---|
  | Level | 40 / 34 / 26% kindergarten / elementary / high school | about 27 / 38 / 35% for ages 3–6 / 7–12 / 13–18 (medium-high): the bands are 4, 6 and 6 years wide, so a stable population gives 25 / 37.5 / 37.5, and 2% yearly growth 28 / 38 / 34 |
  | Household size (+0.15) | flat: kindergarten 36–41% in every household-size band, 39% over 4.5 | a mild tilt to the young where households are large: about 31 / 38 / 30 at 4% growth (medium) |
  | Median age (+0.25) | kindergarten 43% to 30%, high school 24% to 37%, youngest to oldest band | right direction (high), plausible size (medium) |
  | Rooms (+0.30) | the strongest term: kindergarten 53 / 43 / 32 / 22% at 3 / 4 / 5 / 6 rooms | right direction (medium); a 30-point gap is probably too wide (low) |
  | Daycares (−0.20) | kindergarten 35% with none, 42% with 5–7 | plausible, small (medium) |
  | Rooms × age bonus (+0.12) | mean +0.01, 95th percentile +0.07 of a score whose SD is 0.36 | negligible |
  | Building effect (SD 0.20) | SD of a building's expected kindergarten share 0.082, from the features alone 0.050; observed 0.113 (buildings with 20 or more children) | buildings fill with one generation of families (medium-high); the size cannot be calibrated (low) |
  | Planned school, tower-sized buildings | no effect: kindergarten 41% against 39–40%, and 40.3% against 39.9% | new neighborhoods and new buildings hold younger children (medium) |

  - **The household-size lead is overstated.** The coefficient only cancels the median-age term
    in large-household areas (median age 21). At 0 the kindergarten share would be 52% where
    household size is over 4.5 against 32% under 2.2, and at −0.15 it would be 64% against 26%.
    About +0.10 gives the mild real tilt.
  - With reference shares of 27 / 38 / 35% the realized shares are 28 / 37 / 36%. The totals do
    not change; only the split between the three targets moves.
- **The user's decision (2026-11-25):** asked to choose between a values-only step, a fuller step
  and keeping the model, the user answered: it does not have to be exact; add relations where
  they make the model more realistic, without overcomplicating. **A simulator step for the cohort
  split**, with the scope chosen by this session under that instruction:
  - the reference shares move to about 27 / 38 / 35%;
  - one new relation: a planned school shifts the split toward kindergarten (a new, growing
    neighborhood);
  - the rooms × age bonus is dropped, so the score keeps five terms;
  - the household-size coefficient is re-checked by the step's consult (about +0.10), and the
    room coefficient's size with it.
  - Not in the step: a building-size term (the consult may propose it, the default is no), the
    building effect's SD, the daycare and median-age terms.
  S1c.4 waits for that step's PR to be merged into this branch.
