# Next phase plan: simulator realism, the rerun, failure analysis, stakeholder deck

Approved by the user on 2026-11-28. Committed on branch `docs/next-phase-plan` (from `main`) and
merged to `main` by its own small PR; `main` is then merged into `docs/project-summary`.
The planning session implemented nothing and ran no notebook.

## 0. Context

The project predicts children per building (total and three cohorts) on simulated Israeli
buildings. Three things are open, and they depend on each other:

- **The simulator** (steps 1–4 merged, PRs #29–#32) still has one driver, SES, behind almost every
  feature, and several choices the user questions (household size, room mix, the apartment draws).
- **The results** in `.mlflow-check/mlflow.db` (26 runs at `da6afc9`) are on the old data
  (1,529 buildings) and are superseded; the summary's sections 4–7 are empty.
- **No failure analysis and no stakeholder deck exist.**

The simulator changes the data, so the order is fixed: **A** realism steps → **B** one rerun →
**C** failure analysis (ends with the single test look) → **D** the deck. The user's three-task list
is the scope; the rerun (B) links tasks 1 and 2.

State at planning (2026-11-28): `docs/project-summary` at `d60e0d9`; `main` at `81c8cc5`.
`docs/PROJECT_SUMMARY.md` and `docs/PROJECT_SUMMARY_PLAN.md` exist only on `docs/project-summary`.
Many files are iCloud-offloaded, including 32 under `src/age_group_prediction` and
`configs/simulation.yml`.

## 1. Decisions

| # | Decision | Source |
|---|---|---|
| Q1 | Order A → B → C → D; all realism steps before the single rerun. | constraint |
| Q2 | Realism analyses: **four verdict tables decided together** (exogenous variables, household size, room mix, apartments); the user decides keep or step per topic. Decided changes are grouped into **at most three steps in draw order** (neighborhood, building, apartment), one PR each from `main`, each with a Fable consult (R2, R10). | user, 2026-11-28 |
| Q3 | Each step's detailed plan goes in `docs/SIMULATOR_REALISM_PLAN.md` as §14, §15, §16, in the form of §11–§13. This document holds the order, the topics and the record. | simplification |
| Q4 | The summary (section 2, and the data numbers of sections 1 and 3) is updated **once, after the last step**. | user |
| Q5 | The rerun keeps the **random 60/20/20 split**. No grouped view; the random-split caveat goes in section 7.4 and on the deck's limits slide. | user |
| Q6 | P15 as decided: `z(log(H − 1))` is a swap candidate for `z(H)` in the count GLMs, declared identically in notebooks 02, 03 and 04. Notebook 02 is edited, not rerun: D40 and D42 stand. | P15 |
| Q7 | **The test part is evaluated once, at the end of C** (C5). | user |
| Q8 | The failure analysis runs on **validation**, models fitted on train, as git-ignored scripts (P4). The views the user keeps are added to notebook 04 **before** the test click (replaces S9). The best-runs run and the final run are both logged at C5, at one commit. | user + review |
| Q9 | Views of the **total** (row sum) are added to notebook 04 in any case: today no metric, beeswarm or permutation figure covers it. | user's list |
| Q10 | The deck: content first in `docs/STAKEHOLDER_DECK.md`; converted to **.pptx** only when the content is approved. At most four word formulas, no symbols, no code. | user |
| Q11 | One set of PNGs in `docs/figures/` (matplotlib, P9) serves the summary and the deck. | simplification |
| Q12 | Accepted simplifications: fewer rerun checks; merged analysis specs; one independent review per part. Record: sub-tasks on `main`'s line record in this plan's §9; sub-tasks on `docs/project-summary` record in `PROJECT_SUMMARY_PLAN.md` §9 (two branches appending to one section would conflict at every merge). | user + review |
| Q13 | **Models: Fable 5.2 only for the complex sub-tasks** (realism verdicts, each step's design consult, the noise floor, section 7, the final review). Opus 5.5 for implementation and analysis; Sonnet 5.5 for mechanical work. Each sub-task names its model and has one stop the user can verify. | user, 2026-11-28 |
| Q14 | The rerun is **two PRs**: PR-B (the candidate, after B2's check) and PR-C (notebook 04's views and the test look). `check_store.py` applies `git_commit == HEAD` per stage, so `tuning` is checked at B1's commit and `best_runs final` at C4's. | review |
| Q15 | Everything runs inside the project in git-ignored folders with `~/.venvs/age-group-prediction/bin/python`; each check runs once, static checks before long runs; the user commits and pushes; commit and PR texts carry no Co-Authored-By or "Generated with" line; every choice goes in the question dialog, recommendation first. | workflow rules |

## 2. What was simplified against the existing plans

1. Four analysis sessions (P13) → the topics are measured once and decided together (Q2).
2. Section 2 edited after every step → once (Q4).
3. After the rerun: `check_store.py` and the `MLFLOW_RUNS.md` §9 recipe only. Dropped: the skip
   re-clicks, the UI look, the separate review of recorded numbers.
4. No best-runs run in B2, so nothing is deleted and re-logged later: opening notebook 04 without
   a click already reads the store, runs its guard and shows validation scores with intervals.
5. S5a + S5b share one validation table; S5c shrinks to the three arms; S5d extends
   `review_s2_candidates/analysis.py:oracle()`; S6a + S6b → C6; S7a + S7b → C7.
6. S9 disappears (Q8). S5e's PNGs and the deck's figures are one script, one folder (Q11).

## 3. Part A: simulator realism

### 3.1 How it runs

- **A0.1** measures the four topics on the current simulator; **A0.2** adds the Israel column, the
  effect on the benchmark, the recorded decisions each change would reopen, and a recommended
  verdict; the four decisions go in one question dialog; decided changes are assigned to steps 5–7.
- **Steps 5–7** (only those decided), each on its own branch from `main`, in §11–§13's form, five
  sub-tasks: `x.0` plan + Fable consult → `x.1` model, config, unit tests → `x.2` intercept and
  level on 31 seeds (R24), the validation statistics tests → `x.3` fixtures, pins, full suite, docs
  → `x.4` independent review with a mutation check, PR text. A later step's `x.0` re-measures its
  topic on the simulator as it then stands.
- **AS1, AS2**: the summary update (Q4), after the last step, before B.

Files a step changes (as steps 2–4): `configs/simulation.yml`,
`src/student_simulator/{config,neighborhood,building,features,outcomes}.py`, the component's unit
tests, a `tests/validation/test_<component>_statistics.py`, the ranges in
`tests/validation/helpers.py`, the EDA fixture, the three notebook characterization pins.

### 3.2 The four topics, with the preliminary read

From the code and recorded draws; A0.1 measures them. Starting points, not decisions.

**T1. Exogenous variables (step 5, neighborhood).**
- SES enters five draws directly (median age +3.5, household size −0.20·S + 0.18·S², daycares
  +0.35, tower log-odds +0.7, room tilt +0.35) and the rest indirectly. Independent draws exist only
  as noise: median-age noise (SD 4.5), household noise (SD 0.30), the hidden tower propensity
  (SD 1.0) and texture (SD 0.25).
- Share explained by SES: median age about 37% (already mostly its own); household size about 76%
  from SES and median age together (the heavily dependent one).
- Candidate drivers: (1) **SES**, kept; (2) **neighborhood newness**: would drive median age,
  planned school and tower propensity together, replacing three separate noises (lead 12.7.3);
  (3) **family orientation** (religious/traditional vs secular-urban): would drive household size
  and median age and replace most of the SES and SES² terms (lead 13.7.4).
- SES terms that could go: daycares, towers. To keep: room size, part of median age.
- **Recorded decisions these would reopen** (A0.2 names them per row): R3 (draw order), R5 (SES in
  daycares), R13 (SES in the tower odds, the user's own relation), R7 (the U-shape comes through
  household size). A driver independent of SES also moves step 1's calibrated targets: household
  size 4.88 in the lowest SES bin and corr(SES, median age) +0.6 to +0.75; in Israel family
  orientation and low SES are largely the same areas, so the driver would need a negative relation
  with SES or it loses that target.
- To decide: are new drivers **observed columns** (new features: the schema, preprocessing,
  notebooks and SHAP cost all change) or **hidden** (features become less SES-dependent, nothing
  downstream changes). Starting recommendation: hidden, entering only through observed features.

**T2. Household size (step 5).**
- *Is SES² significant?* In the generator its SD is 0.23, about 14% of the variance of H (the
  linear SES path about 43%, median-age noise about 19%, own noise about 24%). The shape is not a
  symmetric U: mean H by SES bin runs 4.88 → 2.77 (lowest at S = 1.14) → 3.01. It steepens the low
  end and gives a small rise at the top. For the models, an SES² feature gained only under the
  random split.
- *Why subtract 3?* 3.0 is the household size of the **reference neighborhood** (SES 0, median age
  31), not the mean; the mean is 3.18 because the squared term adds 0.18·E[S²]. Subtracting the
  reference only moves intercepts (daycares, school weights, the room tilt, the cohort score, and
  the count's log((H − 1.7)/(3.0 − 1.7))); R8 requires one reference neighborhood for all models.
- Options: keep and say so in the summary (recommended start); or centre the squared term
  (S² − 1). The second is **not free**: the SES-0 neighborhood would then have 2.82, the mean falls
  to 3.00, below step 1's target of 3.1–3.3, unless the baseline moves to 3.18 and every reference
  and intercept with it.
- If T1 adopts family orientation the squared term is its stand-in; decide T1 first.

**T3. Room mix (step 6, building).**
- *Towers have more small apartments*: already in, through −0.32·log(A/30): a 90-apartment tower
  gets −0.35, one SD of SES in the other direction.
- *Texture and high SES → large apartments*: already in (+0.35·S; the size term is zero at 30).
- *Low SES, texture, high median age → renewal moves families to larger apartments*: new, and it
  runs against R16 and R21 (large poor households are crowded). Renewal is already modelled as
  towers in older areas (`tower_median_age_coef` 0.3), and towers get smaller apartments. Lead
  12.7.4 found no age effect at fixed SES. Confidence in the proposed effect: low.
- *Examples of tilts*: A0.1 prints the four shares for extreme but sensible cases (S = −2, H = 5,
  30 apartments; S = +2, 30; S = +2, a 90-apartment tower; S = −1, a 120-apartment tower).
- *Dirichlet-multinomial or multinomial?* Room shares are observed features, so this is not target
  noise. A 4-room share's SD: multinomial only 0.090 at 30 apartments and 0.052 at 90; with
  concentration 30, 0.125 and 0.102. The layer gives a building a mix of its own (a developer picks
  a mix) and keeps the shares from being a near-function of the neighborhood, which is T1's
  concern. The value 30 was inherited, never calibrated. A0.1 measures, at 10, 30, 100 and ∞: the
  share SDs and the R² of each share on the neighborhood features. Starting verdict: keep the
  layer; the value follows the measurement.

**T4. Apartments (step 7).**
- *Too much randomness?* The cohort multinomial is the minimum: each child has one age and
  "probability × N" is not an integer; real data has more (siblings' ages are linked). The count is
  where it matters. A0.1 prints the noise budget of building sums on the current simulator:
  features, neighborhood effect (SD 0.20), building effect (0.25), the NB2 draw (φ = 3), the cohort
  draw, the cohort effect (0.20). (The recorded 0.40 / 0.23 relative RMSE and 35% are from before
  steps 3–4 and are stale.) Options: raise φ or shrink the hidden effects; both were out of scope
  under R18, the user's question reopens them.
- *A building-type factor?* The type is hidden (R12); leads 12.7.2 and 13.7.1 and R30 left it out.
  If wanted it enters through the observed `n_apartments` (a log-size term). Reality is ambiguous.
  It would make `n_apartments` informative beyond exposure (today γ = −0.025), which bears on
  section 7.3.

## 4. Part B: the rerun

**B1 (branch `feat/rerun-candidate` from `main`, after the last step).**
- The candidate, as built in `.mlflow-check/summary/check_s2.py:274-276`:
  `Plan(name="log_excess_household", transforms=(CenterByReferencePoint(reference_point=1.0), Log(),
  Standardize()))` on `avg_household_size`, added to `_glm_count_candidates`
  (`notebooks/03_hyperparameter_tuning.py:676-685`; plans at `:475-512`), copied identically to 02
  and 04. Pins: `test_hyperparameter_tuning_notebook.py:948-1063` (02 = 03),
  `test_model_inspection_notebook.py:217-279` (03 = 04, as cell text).
- Check that the minimum of H stays above 1.0 after steps 5–7.
- Search-space notes fitted to old data (`search_spaces.py:103-104, 116-118`,
  `HYPERPARAMETER_TUNING.md:427-435`, `negative_binomial.py:20-23`) are not re-measured before the
  run; B2 checks that no best value sits on a range edge and the notes are updated after it.
- Notebook settings and `check_store.py` do not change. ruff, mypy (`--cache-dir` in the
  scratchpad), the affected tests once.

**B2 (the user clicks).** Pre-checks: no offloaded file under
`src tests notebooks configs checks studies`; clean tree; notebooks restored from HEAD before one
that logs is opened (the `git_dirty` trap); `studies/` holds no folder for the new config hash.
Store: move `.mlflow-check/mlflow.db` and `.mlflow-check/mlartifacts/` into
`.mlflow-check/backup_da6afc9/` (the db stores absolute artifact paths, so the backup is readable
by MLflow only if moved back); `summary/`, `realism/`, `e/` stay; `.env` unchanged. Never
`rm -rf .mlflow-check`.
1. Notebook 03: both Tune buttons (8 + 16 runs).
2. Notebook 04: open only, **no click**: the guard passes; record the best runs and their
   validation scores with intervals.
3. `check_store.py` on a copy, stage `tuning`; the §9 recipe. Each once, before any commit.
4. The range-edge check. Tag B1's commit (`rerun-tuning`) so the runs' `git_commit` survives the
   squash. Then PR-B; merge `main` into `docs/project-summary`; there: point
   `PROJECT_SUMMARY_PLAN.md` §3 at the new store, re-run `check_s2.py` (the candidate is now
   declared), then S3a and S3b can run.

## 5. Part C: failure analysis

Branch `feat/inspection-total-views` from `main` after PR-B. Scripts and CSVs in
`.mlflow-check/summary/c*/`. On validation, models fitted on train.

- **C1a. The validation table.** Rebuild both best models by notebook 04's path (`read_tuning_run`,
  `rebuild_best_run`, as `.mlflow-check/e/recipe_s9.py`); one row per validation building: features,
  true and predicted counts (three cohorts, total, both models), joined with the hidden effects
  (`SimulationResult.building_effects`, `.neighborhood_effects`). Check: scores equal the two best
  tuning runs' `scores.csv` exactly.
- **C1b. Distributions and bias.** Per cohort and total, for train, validation and test: mean, SD,
  quantiles, maximum, zero share, children per apartment (labels only; no model touches test).
  Bias mean(predicted − true) per cohort and total, both models, with neighborhood-bootstrap
  intervals (`BootstrapEvaluator`, `scoring.py`).
- **C1c. Size groups and ranges.** Four groups by the true value, per cohort and total: RMSE, MAE,
  bias, wMAPE, and true vs predicted minimum, maximum and quantiles. Grouping by the true value
  always shows under-prediction at the top (regression to the mean), so the same table is also
  grouped by the **predicted** value. Plus error by `n_apartments` quintile, household-size and
  median-age bins; the best and worst five buildings per model and what they share.
- **C2a. The noise floor (S5d).** Extend `oracle()` to the cohorts and to the hidden effects:
  expected counts per validation building (i) given features and the true effects, (ii) given
  features only. Coefficients are read from the config or the simulator's own functions (the
  script's hard-coded values go stale after steps 5–7); `simulate()` must keep the effects; the cap
  at 10 is handled or its size shown. RMSE and bias beside both models; C1c's size-group table for
  the oracle.
- **C2b. Overdispersion and residuals.** Residual variance against the predicted mean beside the
  simulator's own variance at that mean; how much of each model's residual the hidden effects
  explain.
- **C3a. SHAP answers and coverage.** Written answer with numbers: the waterfall's "mean
  prediction" is the building's `n_apartments` × one fixed background rate per cohort
  (`inspection/shap_values.py:140-155`) when the best run uses the exposure, so it changes per
  building by design; all nine features are drawn, nothing is folded into an "other" bar; what the
  base absorbs is the exposure role of `n_apartments`. Coverage table: figure type × (total, three
  cohorts) × (validation, test) × (logged, menu only).
- **C3b. Waterfall prototype and the three arms.** A waterfall with a **fixed base** plus one
  "building size" bar, beside a children-per-apartment version; the three arms of `n_apartments`
  on validation. Then the **C-stop dialog**: which views go into notebook 04 (the total's metrics,
  beeswarm and permutation are in by Q9), and whether a finding calls for a model change before the
  test look (default none; a re-tune first deletes the affected study files and their MLflow runs,
  because the study cache does not detect code changes).
- **C4a. The total in notebook 04.** Only in 04's own cells after the read: the copied cells,
  `METRICS`, `COHORTS` and `compute_metrics`' existing outputs stay identical, or the guard refuses
  the B2 store. Acceptance: a script run of 04 without a click passes the read cell on the B2 store.
- **C4b. The chosen views, `check_store.py` counts, review.** Counts (54 metrics, 18/33 figures,
  3/7 tables) updated; tests; static checks. The user commits.
- **C5. The test look (the user clicks).** Clean tree, notebooks from HEAD. "Inspect the best
  runs", then "Evaluate on the test part", once. `check_store.py` on a copy, stages
  `best_runs final`, before any commit. Test scores into §9. No model, setting or selection change
  after this. PR-C; merge `main` into `docs/project-summary`; S4a and S4b can run.
- **F1. Result figures** (before C6): PNGs from C's CSVs and the final run's tables.
- **C6. Summary section 6** (S6a + S6b specs). Every number marked validation or test.
- **C7. Summary section 7** (S7a + S7b specs): which model when; why they fail (error against the
  floor, size, hidden effects); why `n_apartments` ranks low; future steps (Model 3, a split by
  neighborhood with the random-split caveat, importance moving feature and exposure together).
- **C8a, C8b.** Links and render; the whole-document review and the summary's PR (after the deck).

For the deck: the bias chart, the size-group chart, model against the floor, one waterfall, the
importance bar.

## 6. Part D: the stakeholder deck

On `docs/project-summary`, after C7.

- **D1.** `docs/STAKEHOLDER_DECK.md`: per slide a title stating the takeaway, at most four bullets,
  the figure's file name, a speaker note. The user reviews and approves the content.
- **D2.** EDA figures from the benchmark dataset, added to `.mlflow-check/summary/figures.py`
  (notebook 01 is not rerun); each PNG under 300 KB.
- **D3.** The approved markdown → `docs/stakeholder_deck.pptx` with the pptx skill (or the user
  converts the same markdown in Claude Desktop). Review: each results number against the summary,
  each EDA number against `figures.py`'s output, validation and test labelled.

Outline (13 slides): 1 title · 2 the question: how many children, of which school age, will live
in a building · 3 why simulated data: no fine-grained real counts; a simulator calibrated to Israel
gives a known truth · 4 how a building is generated, one picture · 5 the data at a glance · 6 EDA:
what moves the number of children · 7 EDA: how the features relate · 8 two models in words: "each
cohort on its own" and "total × cohort share" · 9 how we chose and tested · 10 results: the best
model's error in children per building, against the best possible · 11 where it fails: by size and
cohort, over- or under-estimation · 12 what drives a prediction · 13 limits and next steps.

## 7. What happens to `docs/PROJECT_SUMMARY_PLAN.md`

- Stay, with their prompts: **S3a, S3b** (after PR-B is merged in), **S4a, S4b** (after C5).
- Replaced here: S5a–S5d → C1a–C3b; S5e → F1, D2; S6a–S6b → C6; S7a–S7b → C7; S8a–S8b → C8a–C8b;
  S9 → C4a–C4b. The "306" in S5b is void.
- Note at the top of its §6, added on `docs/project-summary` after P0's merge: "S5a–S9 are
  replaced by `docs/NEXT_PHASE_PLAN.md` (2026-11-28). S3a–S4b stay here and run after that plan's
  B2 and C5; this section's §9 also records that plan's sub-tasks done on this branch."
- `docs/SIMULATOR_REALISM_PLAN.md` §7 (on `main`, in P0): the row "5 | The reruns" becomes steps
  5–7 plus the rerun, pointing here; R9 and P15 named.

## 8. Sub-tasks: one stop each, the model, how the user verifies

| Id | What | Model | Branch | The user verifies |
|---|---|---|---|---|
| P0 | Commit this plan and the realism-plan pointer; PR; merge `main` into `docs/project-summary`; add the summary-plan note there | user | `docs/next-phase-plan` | the two notes read right |
| A0.1 | Measure T1–T4 (scripts `.mlflow-check/realism/a0_*.py`) | Opus 5.5 | `main`, nothing tracked | four output files; the tilt examples |
| A0.2 | Verdict tables, the four decisions, steps assigned | **Fable 5.2** | `main`; the §9 entry is carried to the first decided branch | the tables; the dialog |
| x.0 | Step plan + design consult (x = 5, 6, 7) | **Fable 5.2** | `feat/realistic-…` | the plan section |
| x.1 | Model, config, unit tests | Opus 5.5 | same | the diff; unit tests pass |
| x.2 | Calibration on 31 seeds, validation statistics tests | Opus 5.5 | same | the calibration table |
| x.3 | Fixtures, pins, full suite, docs | Sonnet 5.5 | same | suite result |
| x.4 | Independent review with a mutation check; PR text | Opus 5.5 | same | findings; the PR |
| AS1 | Summary section 2 + `check_s1c.py` entries | Opus 5.5 | `docs/project-summary` | `check_s1c.py` green |
| AS2 | Sections 1 and 3 numbers + `check_s2.py` | Sonnet 5.5 | same | `check_s2.py` green |
| B1 | The candidate in 02/03/04; review by an Opus subagent | Opus 5.5 | `feat/rerun-candidate` | the diff; pin tests pass |
| B2 | The run and its checks; PR-B | user clicks; Sonnet 5.5 | same | `check_store.py` 0 failures; scores |
| C1a | The validation table | Opus 5.5 | `feat/inspection-total-views` | scores equal `scores.csv` |
| C1b | Distributions, bias | Sonnet 5.5 | same | two tables |
| C1c | Size groups, ranges, bins, best/worst | Sonnet 5.5 | same | the tables; groups sum to n |
| C2a | The noise floor | **Fable 5.2** | same | the redraw check; floor beside models |
| C2b | Overdispersion, residuals | Opus 5.5 | same | one table, one chart |
| C3a | SHAP answers, coverage table | Opus 5.5 | same | base ÷ `n_apartments` constant |
| C3b | Waterfall prototype, three arms; the C-stop dialog | Opus 5.5 | same | two figures; the dialog |
| C4a | The total in notebook 04 | Opus 5.5 | same | 04 reads the B2 store, no click |
| C4b | Chosen views, `check_store.py` counts; review by an Opus subagent | Opus 5.5 | same | tests pass; findings |
| C5 | The test look; PR-C | user clicks; Sonnet 5.5 | same | `check_store.py` 0 failures |
| F1 | Result figures | Sonnet 5.5 | `docs/project-summary` | the PNGs |
| C6 | Summary section 6 | Opus 5.5 | same | numbers against the CSVs |
| C7 | Summary section 7; review by an Opus subagent | **Fable 5.2** | same | each claim cites a number |
| D1 | Deck content in markdown | Opus 5.5 | same | the markdown |
| D2 | EDA figures | Sonnet 5.5 | same | the PNGs |
| D3 | The .pptx; review by an Opus subagent | Sonnet 5.5 | same | the file opens; numbers match |
| C8a | Links, render | Sonnet 5.5 | same | zero broken links |
| C8b | Whole-document review; the summary's PR | **Fable 5.2** | same | findings; the PR |

At each stop the session prints: what was done, the commit message in a code block, the next
sub-task's prompt in one code block with its model, and at a PR the title, body, squash text, branch
deletion and the commands for the next branch.

### Session prompts

Opening (every session; joined with the sub-task text into one block):

```
Read docs/NEXT_PHASE_PLAN.md (§1 decisions, §8 sub-tasks, §9 record) and the files your sub-task
names. First: find src tests notebooks configs checks docs -flags +dataless | wc -l ; if not 0,
tell me and wait. Rules: run only inside the project, scripts and outputs in git-ignored folders
(.mlflow-check/...), with ~/.venvs/age-group-prediction/bin/python; static checks before long
runs; each check runs once; do not commit or push: give me the commit message in a code block with
no Co-Authored-By line; every choice for me goes in the question dialog, recommendation first. Do
exactly one sub-task, add its entry to the record, then stop and print the next sub-task's prompt
in one code block with its model.
```

A0.1 (Opus 5.5):

```
Sub-task A0.1. On main. Read configs/simulation.yml, src/student_simulator/, this plan's §3.2, and
section 2 of the summary with: git show docs/project-summary:docs/PROJECT_SUMMARY.md . Write and
run scripts .mlflow-check/realism/a0_t1.py … a0_t4.py on the current simulator (large draws, as
.mlflow-check/summary/s1c2_analysis_2_3.py does). Measure only; no verdicts, no simulator change.
T1: each feature's R² on SES and the feature correlation matrix. T2: the variance of household
size by component, mean H by SES bin, the mean against the reference 3.0. T3: the four room shares
for the tilt examples in §3.2; at Dirichlet concentration 10, 30, 100 and none: each share's SD at
30 and 90 apartments and its R² on the neighborhood features. T4: the noise budget of building
totals and cohort counts by component; children per apartment by building type and size at fixed
features. Print one table per topic to a0_tK.out and show them to me. The record entry is not
written by this sub-task (nothing tracked changes); A0.2 writes it.
```

Later prompts are printed by the session before them; step `x.0`–`x.4` prompts follow
`SIMULATOR_REALISM_PLAN.md` §13.9.

## 9. Record

- **Planning (2026-11-28, Fable 5.2).** Three explorations; the user's decisions Q2, Q4, Q5, Q7,
  Q10, Q12, Q13 and the plan's location. Independent review (Fable 5.2): 20 findings, each
  reproduced against the files, all folded in: the commit rule is per stage (Q14); no best-runs run
  in B2; C4 must keep the read path identical; C1a compares with `scores.csv`; the study-cache
  pre-check; the summary files are not on `main`; two record sections; the two-PR rerun; scoped
  offload checks; AS split with `check_s2.py`; the SES² share 14%; the cost of centring S²; the
  recorded decisions T1 and T3 reopen; stale noise numbers; `oracle()`'s hard-coded coefficients;
  the deck's EDA numbers checked against the figure script.

## 10. Verification

- A steps: each validation test fails under its own mutation; the full suite passes; calibration
  ranges hold on 31 seeds. AS: `check_s1c.py` and `check_s2.py` green.
- B: `check_store.py` 0 failures for `tuning`; the §9 recipe exact; no best value on a range edge
  (or the edge is recorded and judged).
- C: C1a's scores equal `scores.csv` exactly; every grouping sums to the validation count; C2a's
  redraw mean converges; C4a's script run passes the read; C5: `check_store.py` 0 failures for
  `best_runs final`.
- D: every results number found in the summary, every EDA number in the figure script's output; the
  .pptx opens and each figure renders.
