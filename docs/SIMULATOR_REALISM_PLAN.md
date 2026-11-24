# Simulator realism: the plan

> **Status (2026-11-24):** step 1 complete and merged (PR #29). The record is §10.

The simulator's neighborhood model is stylized: household size ignores SES, daycares ignore
demand, school status is independent of everything, and the levels (household size 2.6, median
age 37) are European rather than Israeli. The project summary (`docs/PROJECT_SUMMARY.md`, branch
`docs/project-summary`, paused under its decision P12) will describe a realistic simulator
instead. This plan revises the simulator **step by step, one PR per step, neighborhoods first**.

> **Step 2 (2026-11-24):** buildings and the room mix, planned in §11 on
> `feat/realistic-buildings`; decisions R11–R17. B0–B3 committed (the intercept is −0.88, not
> −0.95); **B4 done** (the review and the PR description), awaiting the user's commit and the PR.
> Next: step 3 of §7, after the project summary's S1c.2. The record is 11.9.

> **Step 3 (2026-11-24):** children per apartment and the building-size scales, planned in §12
> on `feat/realistic-children`; decisions R18–R24. Next: C0 (the user commits this plan), then C1.
> The record is 12.10.

## 1. Decisions

| # | Decision |
|---|---|
| R1 | Goal: data close to real behaviour, **calibrated to Israel** at the level of urban neighborhoods (statistical areas). SES is a **national** standardized index: 0 is the national average, units of SD. |
| R2 | Step by step: **step 1 = neighborhoods** (this plan, in detail). Steps 2–4 (buildings and room mix; children per apartment; the cohort split) are outlined in §7 and each gets its own Fable design consult and plan section before it starts. |
| R3 | Draw order: **demographics first, facilities last** (demand drives supply): SES → median age → household size → daycares → school status. |
| R4 | School status depends on the demographics, kept minimal: two slopes on two drivers. |
| R5 | Daycares follow demand (household size, young age) and purchasing power (SES). The direct age term is in (Fable's final answer, 2026-11-24). |
| R6 | In step 1, downstream values change only where the new neighborhoods would make the existing model inconsistent: the feature references and scale, the children intercept, the SES² coefficient (§3). Everything else downstream waits for its step. |
| R7 | The SES² term of the children count goes to **0**. The U-shape now comes through household size, and keeping 0.05 would double count it. |
| R8 | The reference neighborhood of the outcome model equals the neighborhood baselines. A cross-section config check enforces it. |
| R9 | One PR per step, from `main`: step 1 on `feat/realistic-neighborhoods`. The runs (notebooks 03 and 04, the MLflow store) are redone only after the documentation is written (summary P12). |
| R10 | The real-world targets come from memory (Fable's), not from a lookup; each carries a confidence. The tests assert shapes and signs, not exact numbers. |
| R11 | Step 2, building size: the typical size stays about 40 apartments (user, 2026-11-24), so the targets keep their scale. It reads as newer, larger residential buildings, not the whole national stock. |
| R12 | Step 2: two building types (user). A **texture** building keeps the neighborhood's texture (6–10 floors); a **tower** is higher. The type is hidden in step 2: it shapes `n_apartments` and the room mix and is not a dataset column. It becomes a column only if step 3's analysis finds that the type should affect the children directly. |
| R13 | Step 2: the chance of a tower rises with SES, with an older median age and with a planned school (user's suggestion, adjusted). The daycares are left out: they follow the population and mostly repeat SES. |
| R14 | Step 2: a hidden neighborhood effect on the tower log-odds, so towers cluster within neighborhoods, and a hidden neighborhood texture in the size, so buildings of one neighborhood are alike. |
| R15 | Step 2: larger buildings hold smaller apartments (user): the room tilt gets a term in the building's own size. |
| R16 | Step 2, room mix: apartment size rises with SES and depends only weakly on household size, so large poor households are crowded. The room categories stay 3–6 (no new dataset column), with a lower 6-room share (user). |
| R17 | Step 2 changes downstream only the children intercept (R6). The fall of the children count in large-household neighborhoods is left to step 3 (user). |
| R18 | Step 3: the children model stays at the apartment level (user). Scope: the household-size relation, one interaction, a cap, the intercept. Not in the step: the zero share and family size (no two-part draw), the hidden effects' SDs, the main room effects, the daycare, school-status and median-age terms, $\phi$. |
| R19 | Step 3: expected children are **proportional to household size minus 1.7**, the household size of an area without children aged 3–18. The exponent stays 1. |
| R20 | Step 3: the top target is revised. Children aged 3–18 per household equal household size times the population share aged 3–18, which gives about 2.0–2.4 at household size 5.0 (Haredi areas 2.4–2.6, Arab towns 1.7–1.9) and a slope of 0.6–0.7 per person, not 2.5–2.7 and 0.8–0.9 (medium). |
| R21 | Step 3: the model keeps one interaction (user: at least one, it may be non-linear). **Crowding:** rooms × the log household-size term, with a negative sign: where households are large, small apartments also hold many children. |
| R22 | Step 3: the count is capped at **10** by clipping the draw (user). |
| R23 | Step 3: the building-size log SDs become **0.25** (neighborhood texture) and **0.15** (the building's own), from 0.20 and 0.22 (user). |
| R24 | Step 3: the intercept is calibrated on the average of 31 seeds, not on seed 42 alone (B2's lesson). |

## 2. Step 1: the neighborhood model

In draw order, per neighborhood $j$. $s$ is SES, $a$ median age, $H$ average household size, $D$
daycares within 500 m.

| # | Variable | Formula | Why |
|---|---|---|---|
| 1 | SES | $s = \mathrm{clip}(\mathcal N(0, 1^2), -2.5, 2.5)$ (unchanged) | national standardized index |
| 2 | Median age | $a = \mathrm{clip}(31 + 3.5\,s + \mathcal N(0, 4.5^2),\ 16,\ 50)$ | the lowest-SES areas (Haredi, Arab) are very young; high-SES areas are older |
| 3 | Household size | $H = \mathrm{clip}(3.0 - 0.06\,(a - 31) - 0.20\,s + 0.18\,s^2 + \mathcal N(0, 0.30^2),\ 1.8,\ 6.5)$ | falls with age (median age counts the children; old areas are empty nesters); reverse-J in SES |
| 4 | Daycares | $D \sim \mathrm{Poisson}(\exp[1.02 + 0.30\,(H - 3.0) + 0.35\,s - 0.03\,(a - 31)])$ | demand (large households, young areas) and purchasing power (SES) |
| 5 | School status | $P \propto (0.55\,e^{0.8 (H - 3.0)},\ 0.20\,e^{-0.08 (a - 31)},\ 0.25)$ over (existing, planned, none) | existing schools follow child demand; planned schools follow young, growing areas |

- The SES draws stay bit-identical to today for a given seed (same first draw).
- RNG order: the SES normal, the age noise, the household noise, the Poisson, then **one uniform
  per row** for school status (inverse CDF over the row's probabilities; `rng.choice` cannot take
  per-row probabilities).
- The school shares 0.55 / 0.20 / 0.25 are exact at the reference neighborhood ($H = 3.0$,
  $a = 31$), the pattern of `cohort.reference_shares`.
- Through age, the total SES effect on household size is $-0.41\,s + 0.18\,s^2$, minimal near
  $s \approx 1.1$.

### Calibration (Fable, 200,000–400,000 draws)

| Variable | Mean | SD | Range | Clipped | Target (confidence) |
|---|---|---|---|---|---|
| SES | 0.00 | 0.99 | −2.5 to 2.5 | 1.3% | by definition |
| Median age | 31.0 | 5.6 | 16 to 50 | 0.4% | national about 30 (high); Bnei Brak/Rahat about 17–20, Tel Aviv/Haifa about 35–38 (medium) |
| Household size | 3.18 | 0.61 | 1.8 to 6.3 | 0.2% | national 3.1–3.3 (high); SD across areas 0.6–0.8 (low) |
| Daycares | 3.03 | — | 0 up | — | about 3 within 500 m (low); zeros 6.4%, variance/mean 1.24 |
| School status | 57.6 / 18.6 / 23.8% | | | | base shares kept from the current config (low) |

Mean by SES bin:

| SES bin | ≤ −2 | −2…−1.5 | −1.5…−1 | −1…−0.5 | −0.5…0 | 0…0.5 | 0.5…1 | 1…1.5 | 1.5…2 | > 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Median age | 23.1 | 25.0 | 26.7 | 28.5 | 30.1 | 31.8 | 33.5 | 35.3 | 37.0 | 38.9 |
| Household size | 4.88 | 4.23 | 3.77 | 3.40 | 3.12 | 2.92 | 2.80 | 2.77 | 2.83 | 3.01 |

- **Daycares by SES**, in five bins (below −1.5, −1.5…−0.5, −0.5…0.5, 0.5…1.5, above 1.5): 2.76,
  2.68, 2.86, 3.36, 4.42.
- **Daycares by median age**, in six bins from under 24 to over 40: 3.65, 3.23, 3.01, 2.86, 2.77,
  2.66.
- **Household size targets** (medium confidence): about 5–6 in the lowest clusters, about 2.6–2.9
  in clusters 8–9, and a small rise at the top.

Correlations:

| Pair | Value | Target |
|---|---|---|
| SES, median age | +0.61 | +0.6 to +0.75 |
| SES, household size | −0.66 | |
| Median age, household size | −0.74 | −0.7 to −0.8 |
| Daycares, median age | −0.14 | |
| Daycares, SES | +0.21 | |
| Daycares, household size | +0.13 | |
| Daycares, median age, at fixed SES (partial) | −0.34 | |

The targets carry medium confidence.

**At 150 neighborhoods, seed 42** (before the daycare age term):
- SES SD 0.85 (sample luck, the same draws as today).
- Median age 30.7 ± 5.5 (18.3–45.5).
- Household size 3.17 ± 0.60 (1.8–4.57).
- No Haredi-like tail above $H = 4.6$ in this draw.

## 3. Step 1: downstream consistency

| Field | Now | New | Why |
|---|---|---|---|
| `features.household_size_reference` | 2.6 | 3.0 | must equal the reference neighborhood (R8) |
| `features.median_age_reference` | 37.0 | 31.0 | the same |
| `features.household_size_scale` | 0.5 | 0.75 | household-size SD doubled (0.29 → 0.61). At 0.5, neighborhoods with $H > 5$ average 4.96 children per apartment, more than $H - 1$; at 0.75 they average 2.47 (real about 2.65). This also damps the room-mix tilt and the cohort term consistently. |
| `total_children.intercept` | −1.30 | −1.00 | about 0.9 children aged 3–18 per household (about 2.8M children / 3M households, medium) |
| `total_children.ses_quadratic_coef` | 0.05 | 0.0 | R7 |

Full-run results, seed 42, with the earlier daycare formula (no age term):

| Variant | Children per apartment, mean / SD | Zero share | Max | Per building, mean / SD | Kindergarten / elementary / high school |
|---|---|---|---|---|---|
| Today | 0.602 / 0.962 | .609 | 14 | 24.2 / 15.1 | .390 / .331 / .279 |
| New neighborhoods, the five values | 0.872 / 1.279 | .515 | 37 | 35.1 / 23.4 | .375 / .336 / .289 |

Seeds 7 and 11 give 0.85–0.86 per apartment and 34.2–34.5 per building. **N2 re-verifies these
with the final daycare formula.** If the mean per apartment leaves 0.85–0.90, N2 recalibrates only
the intercept and records it.

## 4. Step 1: the config schema (`neighborhood` section)

```yaml
neighborhood:
  ses_mean: 0.0
  ses_sd: 1.0
  ses_min: -2.5
  ses_max: 2.5
  median_age_baseline: 31.0                # at SES 0 (replaces median_age_mean)
  median_age_ses_slope: 3.5                # years per SES SD (new)
  median_age_noise_sd: 4.5                 # replaces median_age_sd
  median_age_min: 16.0
  median_age_max: 50.0
  household_size_baseline: 3.0             # at SES 0 and the baseline age
  household_size_age_slope: -0.06          # persons per year
  household_size_ses_slope: -0.20          # new
  household_size_ses_quadratic_coef: 0.18  # new
  household_size_noise_sd: 0.30
  household_size_min: 1.8
  household_size_max: 6.5
  daycare_log_intercept: 1.02
  daycare_household_size_coef: 0.30        # demand, per person (new)
  daycare_ses_coef: 0.35                   # purchasing power
  daycare_age_coef: -0.03                  # young areas, per year (kept)
  school_status_reference_shares:          # replaces school_status_probs
    existing: 0.55
    planned: 0.20
    none: 0.25
  school_existing_household_size_coef: 0.8 # log-odds vs none, per person (new)
  school_planned_age_coef: -0.08           # log-odds vs none, per year (new)
```

- **Removed:** `median_age_mean`, `median_age_sd`, `school_status_probs`.
- **Centering:** household size, daycares and school status all center age at
  `median_age_baseline`, and household size at `household_size_baseline`.
- **Validation kept:**
  - the min ≤ max checks;
  - the noise SDs above 0;
  - the reference shares as an exact-key probability map.
- **Validation added:**
  - every reference share above 0 (its log is taken);
  - each baseline inside its bounds;
  - `household_size_min` above 1;
  - a cross-section check in `Config` that `features.household_size_reference` equals
    `neighborhood.household_size_baseline`, and `features.median_age_reference` equals
    `neighborhood.median_age_baseline` (R8).
- **No sign constraints** on the coefficients; the signs belong to the statistical tests.

## 5. Step 1: the tests

**New statistical tests.** These run on a large draw (e.g. 20,000 neighborhoods, a fixed seed)
and assert shapes and signs only:
- **SES:** mean about 0, SD about 1.
- **Median age:** correlation with SES above 0.4; the bottom SES decile's mean at least 8 years
  below the top decile's.
- **Household size:**
  - OLS on age, SES and SES² gives a negative age coefficient;
  - the bin means follow the reverse-J: the lowest SES bin is above the middle, the middle is
    above the minimum bin, and the top bin is above the minimum bin;
  - the overall mean is in 3.0–3.4.
- **Daycares:**
  - non-negative integers;
  - a Poisson regression on household size, SES and age gives positive, positive, negative
    coefficients;
  - the top SES quintile's mean is above the bottom quintile's;
  - the zero share is in 2–15%.
- **School status:**
  - labels exactly the three values;
  - P(existing) rises across household-size terciles;
  - P(planned) falls across age terciles;
  - the shares are near the reference shares in a window around $H = 3.0$, $a = 31$.
- **Bounds and reproducibility:** all values within bounds; the clipped share under 2% per
  variable; the same seed gives the same frame.
- **Consistency invariant** (full run): in at least 95% of neighborhoods, mean children per
  apartment is at most $H - 1.5$.
  - **Added in N4:** on a 1000-neighborhood run, at least 80% of the neighborhoods with
    $H > 4.5$ satisfy the same rule. The 95% rule alone barely detects a household-size scale
    that is too small (§10, N2 and N4).

**Config tests:**
- a removed field is rejected (`extra="forbid"`);
- a zero reference share is rejected;
- a baseline outside its bounds is rejected;
- the reference-equality check fires.

**Existing tests to update:**
- `tests/unit/test_neighborhood.py`, `tests/unit/test_config.py` and `tests/unit/helpers.py`
  (the removed fields).
- `tests/unit/test_pipeline.py` and `tests/unit/test_rng_convention.py`, if they pin the draw
  order or counts.
- `tests/validation/helpers.py` pins mean children 0.40–0.61 and zero share 0.60–0.75. These
  become about 0.75–1.00 and 0.45–0.60.
- `tests/characterization/fixtures/` (e.g. `eda_jupyter_baseline.json`) and the notebook
  characterization tests pin today's outputs. They are regenerated in N3, after the values are
  final.

## 6. Step 1: effects on the benchmark

- **Collinearity:**
  - VIFs across three seeds: SES 1.9–2.5, household size 2.4–3.4, median age 2.4–2.8, daycares
    about 1.1, school dummies 1.3–1.6 (today all are at most 1.6).
  - This is moderate and realistic.
  - Importance (permutation, SHAP) will be shared among SES, age and household size.
  - The coefficient-recovery tolerances for household size and age loosen.
- **Model families:** linear and tree models were not separated before and will not be after. In
  an untuned check, a Poisson GLM beat Poisson boosting in every run. The GLM's D² for the building
  total rises from about 0.3 to about 0.45; the hidden effects still leave more than half
  unexplained.
- **Targets:** building totals are more spread out (SD 23 against 15, maximum about 260), so a few
  large-household neighborhoods weigh more in the metrics.
- **School status** is now weakly informative of household size and age, so its coefficients
  (0.10, 0.04) are partly confounded.

## 7. Later steps (outlined; each gets a Fable consult and its own plan section)

| Step | Component | Known issues (from Fable's step-1 run) |
|---|---|---|
| 2 | Buildings and room mix | Large-household, low-SES areas get larger apartments (mean rooms 4.39 against 4.04); in Israel they are crowded. Building size may depend on the area. |
| 3 | Children per apartment (**planned in §12**) | The NB tail: a maximum of 37 children in one apartment (needs a cap or a larger $\phi$). Zero share 0.52 against about 0.60 real. Too flat at small households (0.45 children at $H = 2.2$; real about 0.1–0.3). |
| 4 | The cohort split | Reference shares 0.40/0.35/0.25 against a real 27/38/35% for ages 3–6/7–12/13–18. The household-size coefficient (+0.15) pushes large-household areas toward high school, the opposite of Israel. |
| 5 | The reruns | After the summary's documentation (summary P12): notebooks 03 and 04 rerun on the new data, a fresh MLflow store, then the summary's S5a onward. |

## 8. Step 1 sub-tasks

| # | Sub-task | Model | Review | Done |
|---|---|---|---|---|
| N0 | The branch `feat/realistic-neighborhoods` from `main`; this document committed | user | no | |
| N1 | The config schema and validation (§4) and the new neighborhood draws (§2), with their unit and statistical tests (§5) | Opus 5.5 | yes | 2026-11-24 |
| N2 | The downstream values (§3), the cross-section check, the validation-helper ranges, the consistency invariant; a full run confirming §3's targets with the final daycare formula | Opus 5.5 | yes | 2026-11-24 |
| N3 | Regenerate the characterization fixtures; the full test suite; update the docs that name the old fields (grep for every removed and renamed field) | Sonnet 5.5 | no | 2026-11-24 |
| N4 | An independent review of the whole step against this plan, then the PR description | Fable 5.2 | is the review | 2026-11-24 |

### N0. Setup (the user)
1. On `docs/project-summary`, commit the summary's changes (not this document).
2. `git switch -c feat/realistic-neighborhoods main`. This document is untracked, so it comes
   along.
3. Commit it: `docs(simulator): the plan for a realistic, Israel-calibrated simulator`.

### N1. Config and the neighborhood draws (Opus 5.5, review)
Done when:
- [x] The `neighborhood` section and its settings class follow §4, with the validation rules.
- [x] The neighborhood component draws §2 in its order, with the school status from one uniform
      per row.
- [x] The unit tests (config, neighborhood) and the statistical tests of §5 pass; the other
      neighborhood-dependent unit tests are updated.
- [x] A check script prints the marginals, the SES-bin means and the correlations on a large
      draw, next to §2's numbers.
- [x] Review findings reproduced and fixed.

### N2. Downstream consistency (Opus 5.5, review)
Done when:
- [x] The five values of §3 are in the config, and the cross-section check is in `Config`.
- [x] A full run (150 neighborhoods, rate 9, seeds 42, 7, 11) prints children per apartment and
      per building, the zero share, the max and the cohort shares, next to §3's table. The
      intercept is recalibrated only if the mean leaves 0.85–0.90.
- [x] The validation helpers' ranges are updated, and the consistency invariant is a test.
- [x] Review findings reproduced and fixed.

### N3. Fixtures, the suite, the docs (Sonnet 5.5)
Done when:
- [x] The characterization fixtures are regenerated, and the diff of each is explained in §10.
- [x] The full test suite passes (output shown).
- [x] Every doc naming a removed or renamed field is updated. Historical plans and records under
      `docs/archive/` stay as they are.

### N4. Review and the PR (Fable 5.2)
Done when:
- [x] The step is reviewed against §1–§5 as a new reader; each finding is reproduced, then fixed.
- [x] The PR description is printed in the chat.

## 9. Session prompts

**The opening:**

```text
We are making the simulator realistic, step 1 (neighborhoods), on branch
feat/realistic-neighborhoods. Read docs/SIMULATOR_REALISM_PLAN.md first: §1 decisions, §2 the
model, §3 downstream, §4 the config, §5 the tests, §8 your sub-task, §10 the record. Rules: do
only this sub-task; the plan's values are decided (do not re-derive them; if one fails a check,
stop and show the evidence); scripts and their outputs go in .mlflow-check/realism/ (git-ignored)
and run with ~/.venvs/age-group-prediction/bin/python; static checks (ruff, mypy with
--cache-dir in that folder) before the tests; run each check once; do not commit. At the stop:
show the diff and the test output, tick the sub-task's "done when" boxes and add its entry to
§10, write the commit message in a text block (no Co-Authored-By line), and print the next
sub-task's prompt (this opening, a blank line, its text) in one code block, naming the model to
select.
```

| Sub-task | Text after the opening |
|---|---|
| N1 | `Implement N1: the neighborhood config schema and validation (§4) and the new neighborhood draws (§2), with their unit and statistical tests (§5) and a check script against §2's numbers. Run an independent review subagent before the stop.` |
| N2 | `Implement N2: the downstream consistency values (§3), the cross-section check, the validation-helper ranges and the consistency invariant, with a full run against §3's table. Run an independent review subagent before the stop.` |
| N3 | `Implement N3: regenerate the characterization fixtures, run the full test suite, and update the docs that name the removed or renamed fields.` |
| N4 | `Implement N4: review the whole step against the plan as a new reader; reproduce each finding before fixing it; then write the PR description.` |

## 10. Record

### Planning (2026-11-24)
- The user's review of the summary's section 2 questioned the neighborhood model. That led to
  this plan (summary decision P12).
- **First Fable consult:**
  - the age slope of household size is right in sign: the median age counts the children;
  - household size should follow a reverse-J in SES;
  - the daycares already had age and SES, but no demand;
  - demographics come before facilities;
  - the daycare and school terms are small against the hidden effects.
- **Second Fable consult:** the Israel-calibrated design of §2–§6.
- **Follow-up on daycares:** Fable recommended adding the direct age term, which needed SES 0.35
  and intercept 1.02 to keep the SES profile. The user had left this choice to Fable, so it was
  taken.
- The user chose:
  - Israel, national;
  - step by step, neighborhoods first;
  - a separate PR from `main`;
  - school status dependent, kept simple;
  - the runs after the documentation.

### N1 (2026-11-24, Opus 5.5; review Fable 5.2)
- **Config:** `NeighborhoodSettings` and `configs/simulation.yml` follow §4.
  - Removed: `median_age_mean`, `median_age_sd`, `school_status_probs`. `extra="forbid"`
    rejects each one.
  - Added rules:
    - each reference share must be above 0;
    - both baselines must be inside their bounds;
    - `household_size_min` must be above 1.
  - **The R8 cross-section check is left to N2**, as §8 places it. With today's `features`
    references (2.6, 37) it would fail until N2 sets them to 3.0 and 31.
- **Draws** (`NeighborhoodSimulator`): §2 in its RNG order.
  - The order is SES, age noise, household noise, the Poisson, then one uniform per row.
  - School status inverts the uniform through the row's cumulative probabilities, using only the
    first two edges, so a rounding error at 1.0 cannot index past the labels.
  - SES is bit-identical to the previous model.
- **Check script:** `.mlflow-check/realism/n1_neighborhood_check.py`, 400,000 draws, seed 2026.
  - Every §2 number matches to rounding: the marginals, clipped shares, daycare zeros and
    dispersion, school shares, the SES-bin means, the daycare bins and all seven correlations.
  - The 150-neighborhood, seed-42 summary matches too. The one exception is the median-age SD,
    which is 5.547: 5.5 rounded, where §2 has 5.6. The mean and range are identical, so this is
    rounding, not a different draw.
- **Tests:**
  - Unit:
    - an oracle test rebuilds every column in the documented RNG order;
    - SES is pinned to the previous model's values;
    - a reference-neighborhood test checks the exact shares and the rate at H = 3.0, a = 31;
    - config tests cover the removed fields, a zero share, baselines out of bounds on both
      sides, and `household_size_min` = 1.
  - `tests/validation/test_neighborhood_statistics.py` holds §5's statistical tests (20,000
    rows, a fixed seed).
  - Result: `tests/unit` plus the new file give 1397 passed, 1 skipped, 1 xfailed.
- **Expected failures until N2/N3:**
  - `tests/validation/test_calibration.py`: mean children 1.03 and zero share 0.50, against the
    old ranges.
  - The characterization and notebook tests that pin today's row counts and hashes.
- **Fable review:** no bugs, no spec deviations.
  - Fixed:
    - the SES test now pins literal values from HEAD;
    - the lower bound of each baseline is now tested;
    - the household-minimum test matches its message;
    - an over-long line is wrapped;
    - a needless `size=` change is reverted.
  - Kept:
    - **P(planned) across the age terciles:** the margin is the thinnest in the file, but no
      seed failed in 300.
    - **A NaN `household_size_min`** is reported under the baseline's name. This comes from the
      existing min/max check.
    - **`ses_mean` is not bounds-checked:** §4 names only the baselines.
    - **Pre-existing lint and mypy findings** on lines this change did not touch are left as they
      are.

### N2 (2026-11-24, Opus 5.5; review Fable 5.2)
- **Config:** §3's five values are in `configs/simulation.yml`:
  - `features.household_size_reference` 3.0, `features.median_age_reference` 31.0;
  - `features.household_size_scale` 0.75;
  - `total_children.intercept` −1.00;
  - `total_children.ses_quadratic_coef` 0.0.
- **R8 check:** `Config.validate_cross_section_contracts` now requires each `features` reference
  to equal its `neighborhood` baseline. Either side changed alone raises
  `features.<x>_reference must equal neighborhood.<x>_baseline`.
- **Full run** (`.mlflow-check/realism/n2_full_run_check.py`; 150 neighborhoods, rate 9, the
  final daycare formula):

  | Run | Per apartment, mean / SD | Zero share | Max | Per building, mean / SD | Kindergarten / elementary / high school | Invariant share |
  |---|---|---|---|---|---|---|
  | §3, seed 42 | 0.872 / 1.279 | .515 | 37 | 35.1 / 23.4 | .375 / .336 / .289 | |
  | seed 42 | 0.888 / 1.282 | .509 | 16 | 35.6 / 23.3 | .373 / .339 / .288 | 0.987 |
  | seed 7 | 0.871 / 1.261 | .512 | 19 | 34.9 / 22.3 | .371 / .336 / .293 | 1.000 |
  | seed 11 | 0.842 / 1.203 | .516 | 19 | 33.9 / 21.4 | .371 / .338 / .291 | 0.993 |

  - Seed 42's mean is inside 0.85–0.90 (0.012 from the top), so **the intercept is not
    recalibrated**.
  - Seed 11 (0.842) is just below §3's 0.85–0.86. The rule looks at seed 42 only.
  - The maximum is 16–19, where §3 had 37. That was one extreme draw, made with the earlier
    daycare formula.
- **Canonical config** (60 neighborhoods, rate 3, seed 42):
  - mean children 0.829 and zero share 0.520, inside the new ranges;
  - oracle recovery's maximum error is 0.084 against the tolerance of 0.20.
- **Validation helpers:** mean children 0.75–1.00 and zero share 0.45–0.60 (§5); the recovery
  tolerance stays at 0.20.
- **Tests:**
  - `test_calibration.py` gains the consistency invariant, on the benchmark's setup at seed 42.
  - `test_config.py` covers R8 from either side, plus the accepted case where both sides move
    together.
  - Two unit frames sat at the old reference point and are moved to H 3.0 and age 31:
    - the `test_design.py` reference row;
    - the room-effect test in `test_total_children_stage.py`. There the rooms × household
      interaction made its differences non-zero; the first test run caught it.
  - Result:
    - `tests/unit` plus the non-slow `tests/validation`: 1404 passed, 1 failed (that room-effect
      frame), 1 skipped, 1 xfailed;
    - after the fix, that file plus `test_recovery.py` (slow): 8 passed;
    - after the review, the three touched files: 39 passed.
- **Fable review:** no bugs, no spec deviations.
  - Fixed:
    - the R8 accepted case is now tested;
    - the positive-quadrant test's ages are widened (30 → 25), since it passed by one year
      against the new reference;
    - the invariant's docstring no longer overclaims.
  - **Open for N4 or the user: the invariant is weak.**
    - At the old scale of 0.5, seed 42 fails by one neighborhood (0.947), and 7 of 9 seeds pass.
    - Seed 42 has no H > 5 neighborhood.
    - It implements §5's 95% rule literally and catches gross errors (intercept −0.5 gives 0.760;
      scale 0.3 gives 0.847). A stronger test would change §5.
  - Recorded:
    - over 31 seeds of the canonical config, the mean spans 0.757–1.023 (seed 24 lies outside
      0.75–1.00); the `calibration` marker already calls it seed-sensitive;
    - the zero share spans 0.481–0.550;
    - recovery error spans 0.041–0.139;
    - `test_transforms.py` keeps the literals 2.6 and 0.5. It is self-contained, so they are
      stale but harmless.

### N3 (2026-11-24, Sonnet 5.5)
- **EDA fixture** (`tests/characterization/fixtures/eda_jupyter_baseline.json`), regenerated
  by `capture_eda_baseline()` after checking that the writer reproduces the old file byte for
  byte. 9 fields changed, 42 did not:
  - `config.content_sha256`: `configs/simulation.yml` changed (N1, N2);
  - `final_dataframe.shape`: 1529 to 1523 rows; `final_dataframe.sha256` and the two summary
    hashes (`cohort_share_describe_sha256`, `feature_target_correlations_sha256`): every draw
    changed;
  - `buildings_per_neighborhood`: mean 10.19 to 10.15, SD 2.96 to 2.97, minimum 4 to 3;
  - `targets`: total children per building 24.2 to 35.6 (SD 15.1 to 23.3), maximum 118 to 187;
    kindergarten, elementary and high-school means 9.4 / 8.0 / 6.7 to 13.3 / 12.1 / 10.2.
    Zero shares are still at most 1.3%.
  - All nine quality checks are still true.
- **Pinned counts**, measured (`.mlflow-check/realism/n3_measure_pins.py`), in
  `test_objective_eda_notebook.py`, `test_hyperparameter_tuning_notebook.py` and
  `test_model_inspection_notebook.py`:

  | Pin | Old | New |
  |---|---|---|
  | rows, seed 42 | 1529 | 1523 |
  | rows, seed 7 | 1561 | 1564 |
  | split train / val / test | 917 / 306 / 306 | 913 / 305 / 305 |
  | `n_train_val` | 1223 | 1218 |
  | wrong second share (0.2) | 978 / 245 / 306 | 974 / 244 / 305 |
  | whole table to the second call | 1146 / 383 / 306 | 1142 / 381 / 305 |

  The wrong wirings still differ from the correct split, so each test still catches its bug.
- **Full suite**, no marker filter (slow, calibration and notebook tests included): 1494
  passed, 1 skipped, 1 xfailed, in 10 min 55 s.
  - The slow NUTS recovery test emits a `RuntimeWarning` (worst R-hat 1.086 against 1.05,
    effective sample size 25.7 against 50) and still passes. I did not run it on HEAD, so I do
    not know whether the warning is new. N4 can check.
- **Docs:** README and `FEATURE_TRANSFORMATIONS.md` give 1523 buildings. The §3.2 paragraph
  "The hypothesis, and what the simulator does" now says household size depends on SES and age
  and the children count has no direct SES term.
  - A grep of `README.md`, `docs/`, `src/`, `notebooks/`, `checks/`, `tests/` and `configs/`
    finds `median_age_mean`, `median_age_sd` and `school_status_probs` only in this plan and the
    config rejection test.
- **Stale until the reruns (R9), left as they are:**
  - the measured-result docs: `HYPERPARAMETER_TUNING.md` (917/306/306),
    `INDEPENDENT_COHORT_MODELS.md`, `MLFLOW_RUNS.md`, and the VIF, correlation and coefficient
    tables of `FEATURE_TRANSFORMATIONS.md` (including its "r = −0.13" mentions);
  - the handoffs, plans and records that name 1529, 917 or 1223;
  - `PARAMETER_REFERENCE.md` and `DATA_GENERATION_PLAN.md`, which describe an earlier panel
    design and not this simulator.

### N4 (2026-11-24, Fable 5.2)
The whole step (`git diff main..HEAD`, 17 files) read against §1–§5.
- **No defect in the model or the config.** The formulas, centering, clips and RNG order match
  §2; the fields, removed fields and validation rules match §4; the five values match §3.
- **Fixed:**
  - **The invariant** (N2's open item). Reproduced on 1000 neighborhoods, seeds 42, 7 and 11
    (`.mlflow-check/realism/n4_invariant_tail.py`):

    | Household-size scale | Share of all neighborhoods within $H - 1.5$ | Share of the $H > 4.5$ neighborhoods | Their mean children per apartment |
    |---|---|---|---|
    | 0.75 (now) | 0.976–0.985 | 0.949–0.975 | 1.96–2.27 |
    | 0.5 (before) | 0.946–0.956 | 0.237–0.487 | 3.66–4.73 |

    The 95% rule moves by three points; the tail moves by fifty or more. The user chose to add a
    second test, `test_large_household_neighborhoods_leave_room_for_the_adults`: at least 80% of
    the $H > 4.5$ neighborhoods (about 39 of 1000) within $H - 1.5$. §5 records the added rule.
  - **`docs/README.md`** had no row for this plan; added.
  - **§2:** the median-age SD at 150 neighborhoods is 5.5 (5.547), not 5.6.
- **Checked, not caused by this step:**
  - The NUTS `RuntimeWarning` in N3's full run: `tests/validation/test_bayesian_recovery.py`
    builds its own data and imports nothing from `student_simulator`.
  - `docs/MODULE_REFERENCE.md` names `NeighborhoodSimulator.generate`; the method is
    `sample_neighborhoods`. The line is the same on `main`.
- **Left:** `tests/unit/test_transforms.py` passes its own literals (2.6, 0.5, 37) to the
  transforms and does not read the config.
- **Tests:** `tests/validation/test_calibration.py`: 3 passed. The full suite was not rerun:
  N4 changed no source or config, and N3's run gave 1494 passed.

## 11. Step 2: buildings and the room mix

Decided on 2026-11-24 in the project summary's session (its decision P13): the analysis of the
summary's section 2.2 found two problems, and the user chose a step for both.

- **Room mix.** The poorest neighborhoods got apartments as large as the richest (mean rooms 4.26
  against 4.28), and the room mix followed household size (correlation 0.73) and not SES (0.04).
- **Building size.** A symmetric normal around 40 apartments, the same in every neighborhood.

Decisions R11–R17 (§1) hold the user's choices. Buildings per neighborhood and the Dirichlet
concentration are kept as they are.

### 11.1 The model

In draw order. $j$ is a neighborhood, $b$ a building; $S$, $M$, $H$, $Z$ are SES, median age,
household size and school status. $\tilde M = (M - 31)/10$ and $\tilde H = (H - 3.0)/0.75$ are the
existing scaled features.

| # | Variable | Formula | Why |
|---|---|---|---|
| 1 | Buildings per neighborhood | $1 + \mathrm{Poisson}(\text{rate})$ (unchanged) | a sample-size setting |
| 2 | Tower propensity (hidden) | $r_j \sim \mathcal N(0, 1.0^2)$ | towers come by plan, so they cluster: a neighborhood has a renewal or a new-construction plan, or it does not |
| 3 | Texture (hidden) | $t_j \sim \mathcal N(0, 0.20^2)$ | buildings of one neighborhood were built in one era and are alike |
| 4 | Building type (hidden) | $P(\text{tower}) = \mathrm{logistic}(-2.35 + 0.7\,S_j + 0.3\,\tilde M_j + 0.7\,\mathbb 1\lbrace Z_j = \text{planned}\rbrace + r_j)$ | SES: towers need high land value. Older age: the established neighborhoods that renewal targets. Planned school: new, growing neighborhoods |
| 5 | Apartments | $A_b = \mathrm{clip}(\mathrm{round}(m \cdot e^{t_j + e_b}),\ 8,\ 160)$, $e_b \sim \mathcal N(0, 0.22^2)$, $m = 30$ (texture) or $90$ (tower) | a texture building is mid-rise (6–10 floors); a tower is about three times larger; lognormal, so sizes are right-skewed |
| 6 | Room tilt | $q_b = 0.35\,S_j + 0.10\,\tilde H_j - 0.32\,\log(A_b / 30)$ | apartment size rises with SES; large households weigh little, so large poor households are crowded; larger buildings hold smaller apartments |
| 7 | Room mix | base shares $(0.30, 0.41, 0.24, 0.05)$ for 3–6 rooms; the tilt, the Dirichlet (30) and the multinomial as today | 6-room apartments are rare in apartment buildings |

- **RNG order:** the building counts (unchanged, first), then $r_j$ and $t_j$ (one normal each per
  neighborhood), then one uniform per building for the type, then one normal per building for
  $e_b$, then the Dirichlet and multinomial loop as today.
- **The number of buildings does not change** for a given seed: the count draw is still the first
  draw after the neighborhoods (1,523 at 150 neighborhoods, rate 9, seed 42). B1 verifies it.
- **The size reference of the tilt is the texture median (30):** a typical texture building has no
  size term. A building twice that size has about 0.10 fewer rooms per apartment.
- **The type, $r_j$ and $t_j$ are not returned** in any frame in this step (R12). The tests read the
  types through the sizes.
- **The per-building override** (`apartments_per_building`) stays: it replaces draws 2–5.
- **`rooms - 4.0` stays a literal** in the tilt, as today.

### 11.2 Calibration (20,000 neighborhoods × 10 buildings, seed 2026)

Scripts: `.mlflow-check/realism/s2_design_grid.py`, `s2_design_types.py`,
`s2_design_types_cluster.out`. The real-world targets are from Fable's memory (R10).

**Building size.**

| Quantity | Value | Target (confidence) |
|---|---|---|
| Tower share | 15.3% | a minority of buildings (low) |
| Tower share by SES band (below −1.5, −1.5…−0.5, −0.5…0.5, 0.5…1.5, above 1.5) | 3.6 / 7.8 / 13.8 / 22.9 / 35.4% | few towers in the poorest areas, which are low-rise (medium) |
| Tower share by median age (under 24 to over 40, six bands) | 6.7 / 10.0 / 13.8 / 17.8 / 22.6 / 29.4% | renewal in established neighborhoods (low to medium) |
| Tower share with a planned school / without | 21.4% / 13.9% | new neighborhoods (low to medium) |
| Neighborhoods with no tower among 10 buildings | 38% | towers cluster (medium) |
| Texture buildings: mean, 5th–95th percentile | 31.3, 18–49 | 6–10 floors, about 24–40 apartments (the user) |
| Towers: mean, 5th–95th percentile | 93.0, 55–146 | |
| All buildings: mean, SD, skew | 40.7, 26.1, 2.17 | the mean stays about 40 (R11); today's SD is 11.9, skew 0.06 |
| Percentiles 5 / 25 / 50 / 75 / 95 / 99 | 19 / 25 / 32 / 43 / 102 / 140 | |
| Clipped at 8 / at 160 | 0.00% / 0.39% | |
| Intraclass correlation of log size | 0.28 | alike within a neighborhood (high) |
| Mean size by SES band | 33.4 / 36.0 / 39.9 / 45.4 / 52.7 | |
| Correlation of size with SES / median age / household size | +0.19 / +0.14 / −0.13 | |

**Room mix** (apartment-weighted, with the size term).

| Quantity | Today | New | Target (confidence) |
|---|---|---|---|
| Mean rooms by SES band | 4.26 / 4.15 / 4.12 / 4.16 / 4.28 | 3.85 / 3.92 / 3.99 / 4.08 / 4.20 | rises with SES (medium) |
| Mean rooms, all | 4.16 | 4.01 | about 4 (medium) |
| People per room where $H > 4.5$ | 1.12 | 1.26 | about 1.3 in Haredi and Arab areas (medium) |
| 6-room share | 10.9% | 4.8% | 3–5% in apartment buildings (medium) |
| Mean rooms, texture / tower | — | 4.03 / 3.97 | larger buildings hold smaller apartments (the user) |
| Correlation of the expected mean rooms with SES / household size, without the size term | +0.04 / +0.73 | +0.97 / −0.47 | |

The expected mix is now close to a function of SES. The building's own Dirichlet and multinomial
draws keep the observed shares noisy (a share's SD between buildings is 0.05–0.09).

### 11.3 Downstream consistency

| Field | Now | New | Why |
|---|---|---|---|
| `total_children.intercept` | −1.00 | −0.95 | smaller apartments lower the children count through the room effects; the target stays 0.85–0.90 children per apartment (step 1, §3) |

Full runs with the room-mix values only (150 neighborhoods, rate 9;
`.mlflow-check/realism/s2_design_full_run.py`):

| Run | Per apartment, seed 42 / 7 / 11 | Zero share | Kindergarten share | Neighborhoods with $H > 4.5$, seed 42 |
|---|---|---|---|---|
| Today | 0.888 / 0.871 / 0.842 | .509–.516 | .371–.373 | 2.02 |
| New room mix, intercept −1.00 | 0.828 / 0.811 / 0.794 | .523–.527 | .387–.396 | 1.57 |
| New room mix, intercept −0.95 | 0.873 / 0.855 / 0.833 | .506–.514 | .388–.398 | 1.63 |

- **B2 re-verifies with the final model** (the types and the size term were not in these runs). The
  intercept is changed from −0.95 only if seed 42's mean leaves 0.85–0.90.
  **B2's outcome: −0.88** (seed 42 gave 0.818 at −0.95; 11.9).
- **Nothing else downstream changes in this step (R6).** Two effects are recorded as leads for the
  later steps, not fixed here (R17):
  - **Step 3:** children per apartment in the large-household neighborhoods fall from about 2.0 to
    about 1.6; the realistic level is about 2.7. The children count leans on rooms more than on
    household size. **B2's measure:** those two numbers rest on 3 neighborhoods; on 1000
    neighborhoods the fall is about 11% (2.27 / 2.21 / 1.96 to 2.02 / 1.89 / 1.80 at seeds 42 /
    7 / 11; 11.9).
  - **Step 4:** the kindergarten share rises from .37 to about .39, because rooms push the cohort
    score toward the older cohorts.

### 11.4 The config schema (`building` section)

```yaml
building:
  buildings_per_neighborhood_min: 1
  buildings_per_neighborhood_rate: 3.0
  tower_log_odds_intercept: -2.35       # at the reference neighborhood, no planned school (new)
  tower_ses_coef: 0.7                   # per SES SD (new)
  tower_median_age_coef: 0.3            # per scaled decade of median age (new)
  tower_planned_school_coef: 0.7        # log-odds, planned against the other two (new)
  tower_propensity_sd: 1.0              # hidden neighborhood effect on the log-odds (new)
  texture_apartments_median: 30         # replaces apartments_per_building_mean
  tower_apartments_median: 90           # new
  apartments_neighborhood_log_sd: 0.20  # the hidden texture (new)
  apartments_building_log_sd: 0.22      # replaces apartments_per_building_sd
  apartments_per_building_min: 8        # was 12
  apartments_per_building_max: 160      # was 80
  buildings_per_neighborhood: []
  apartments_per_building: []
  room_mix:
    supported_rooms: [3, 4, 5, 6]
    household_size_tilt: 0.10           # was 0.35
    ses_tilt: 0.35                      # was 0.20
    building_size_tilt: -0.32           # per log of size over the texture median (new)
    room_distance_divisor: 2.0
    dirichlet_concentration: 30.0
    base_shares: {"3": 0.30, "4": 0.41, "5": 0.24, "6": 0.05}
```

- **Removed:** `apartments_per_building_mean`, `apartments_per_building_sd`.
- **The median age is scaled** with the existing `features.median_age_reference` and
  `features.median_age_scale`; no new reference.
- **Validation added:** both medians inside the bounds; `apartments_per_building_min` above 0
  (the tilt takes the log of each size); the three SDs above 0. No ordering rule between the two
  medians: nothing fails when they are swapped, and the statistical tests catch it (B1, the
  user's decision).
- **Validation kept:** min ≤ max; the overrides inside the bounds; the base shares as an exact-key
  probability map.
- **No sign constraints** on the coefficients; the signs belong to the statistical tests.

### 11.5 The tests

**New statistical tests** (a large draw, a fixed seed; shapes and signs only). A building counts as
tower-sized at 52 apartments or more (between the texture's 95th percentile and the tower's 5th).

- **Size:**
  - all sizes are whole numbers inside the bounds; under 2% are clipped;
  - the skew is above 1; the mean is in 36–45; the median is in 28–36;
  - the intraclass correlation of log size is above 0.15.
- **Towers:**
  - the tower-sized share is in 8–22%;
  - it rises across the SES terciles, and across the median-age terciles inside the middle SES
    tercile (median age rises with SES, so a plain age split misses a flipped age sign);
  - it is higher with a planned school than without, inside the middle SES tercile;
  - **moved to a unit test (B1):** with the hidden effects and the size noise near zero, sizes are
    exactly 30 or 90, and the tower share of four kinds of neighborhood (the reference, a planned
    school, SES +1, age +10) matches the logistic formula. The propensity's own effect is pinned
    by the oracle test.
- **Room mix:**
  - mean rooms rise across the SES quintiles;
  - people per room in the neighborhoods with $H > 4.5$ is above 1.15;
  - the 6-room share is under 8%;
  - inside each SES tercile, tower-sized buildings have fewer mean rooms than the others;
- **The draw order:** an oracle unit test replays 11.1 with numpy and equals the simulator
  exactly; it also covers the building counts (drawn first) and the room-count sums. The building
  counts at 150 neighborhoods, rate 9, seeds 42 and 7 equal the previous model's (the check
  script).
- **Unchanged and still required:** the calibration ranges (mean children 0.75–1.00, zero share
  0.45–0.60) and both consistency invariants of §5.

**Config tests:** a removed field is rejected; a median outside the bounds is rejected; a zero SD
is rejected; a zero apartment minimum is rejected.

**Existing tests to update:** `tests/unit/test_building.py`, `tests/unit/test_config.py`,
`tests/unit/helpers.py`, `tests/unit/test_transforms.py` (the room function's new argument),
`tests/characterization/` (fixtures and pins, regenerated in B3). The pinned row counts and split
sizes should not change; B3 measures them. `tests/unit/helpers.py` needed no change: it names no
building field (B4).

### 11.6 Effects on the benchmark

- **`n_apartments` is no longer independent of the features:** its correlation with SES is about
  +0.19 and with median age +0.14. Given the features, the children rate still does not depend
  on it, so it remains the exposure. Tree models, which see it as a feature, will share some
  importance between it, SES and median age.
- **Targets:** the spread of the building counts grows with the size spread (size SD 26 against
  12), and a tower in a large-household neighborhood can hold several hundred children. A few
  towers will weigh heavily in RMSE.
- **Room shares** now carry SES information (the expected mix is close to a function of SES), so
  the importance of SES and of the room shares will be shared.
- **The 6-room share** halves, so `6_rooms_share` becomes a noisier feature.
- **Room shares now move with building size.** `notebooks/01_eda.py` states as a fixed finding
  that mean room shares vary little across the building-size bins. The range is 0.054 (0.039
  without the size term), and the 3-room share now rises with size, from .286 to .328 (B4,
  11.9). The sentence is to be checked at the reruns (R9).

### 11.7 Step 2 sub-tasks

| # | Sub-task | Model | Review | Done |
|---|---|---|---|---|
| B0 | This plan committed on `feat/realistic-buildings` | user | no | 2026-11-24 |
| B1 | The config schema and validation (11.4), the type and size draws and the room tilt's size term (11.1), with their unit and statistical tests (11.5) and a check script against 11.2 | Opus 5.5 | yes | 2026-11-24 |
| B2 | The intercept (11.3), a full run on three seeds next to 11.3's table, the calibration ranges and the invariants | Opus 5.5 | yes | 2026-11-24 |
| B3 | Regenerate the characterization fixtures; the full test suite; update the docs that name the removed fields | Sonnet 5.5 | no | 2026-11-24 |
| B4 | An independent review of the whole step against this section, then the PR description | Fable 5.2 | is the review | 2026-11-24 |

**B1. Done when:**
- [x] The `building` section and its settings classes follow 11.4, with the validation rules.
- [x] The building component draws 11.1 in its RNG order; the building counts for seeds 42 and 7
      equal the previous model's.
- [x] The unit tests and the statistical tests of 11.5 pass.
- [x] A check script prints 11.2's quantities on a large draw, next to 11.2's numbers.
- [x] Review findings reproduced and fixed.

**B2. Done when:**
- [x] The intercept is in the config.
- [x] A full run (150 neighborhoods, rate 9, seeds 42, 7, 11) prints children per apartment and per
      building, the zero share, the maximum, the cohort shares and the children in the
      neighborhoods with $H > 4.5$, next to 11.3's table. The intercept is recalibrated only if
      seed 42's mean leaves 0.85–0.90.
- [x] The calibration ranges and both invariants pass, or the failure is shown with its numbers.
- [x] Review findings reproduced and fixed.

**B3. Done when:**
- [x] The characterization fixtures are regenerated, and the diff of each is explained in 11.9.
- [x] The full test suite passes (output shown).
- [x] Every doc naming a removed field is updated; `docs/archive/` stays as it is.

**B4. Done when:**
- [x] The step is reviewed against 11.1–11.5 as a new reader; each finding is reproduced, then
      fixed.
- [x] The PR description is printed in the chat.

### 11.8 Session prompts

**The opening:**

```text
We are making the simulator realistic, step 2 (buildings and the room mix), on branch
feat/realistic-buildings. Read docs/SIMULATOR_REALISM_PLAN.md first: §1 decisions (R11–R17 are
this step's), §11 the step (11.1 the model, 11.3 downstream, 11.4 the config, 11.5 the tests,
11.7 your sub-task, 11.9 the record), and §10 for how step 1's sub-tasks were recorded. Rules: do
only this sub-task; the plan's values are decided (do not re-derive them; if one fails a check,
stop and show the evidence); scripts and their outputs go in .mlflow-check/realism/ (git-ignored)
and run with ~/.venvs/age-group-prediction/bin/python; static checks (ruff, mypy with
--cache-dir in that folder) before the tests; run each check once; do not commit. At the stop:
show the diff and the test output, tick the sub-task's "done when" boxes and add its entry to
11.9, write the commit message in a text block (no Co-Authored-By line), and print the next
sub-task's prompt (this opening, a blank line, its text) in one code block, naming the model to
select.
```

| Sub-task | Text after the opening |
|---|---|
| B1 | `Implement B1: the building config schema and validation (11.4), the type and size draws and the room tilt's size term (11.1), with their unit and statistical tests (11.5) and a check script against 11.2's numbers. Run an independent review subagent before the stop.` |
| B2 | `Implement B2: the children intercept (11.3), a full run on three seeds against 11.3's table, the calibration ranges and the invariants. Run an independent review subagent before the stop.` |
| B3 | `Implement B3: regenerate the characterization fixtures, run the full test suite, and update the docs that name the removed fields.` |
| B4 | `Implement B4: review the whole step against §11 as a new reader; reproduce each finding before fixing it; then write the PR description.` |

After the PR is merged: merge `main` into `docs/project-summary` and start the summary's S1c.2.

### 11.9 Record

#### Planning (2026-11-24, Fable 5.2, in the project summary's session)
- **The analysis** (`.mlflow-check/summary/s1c1_analysis_2_2.py`, recorded in
  `docs/PROJECT_SUMMARY_PLAN.md` §9 on `docs/project-summary`) found the room mix and the
  building size unrealistic; the user chose a step for both.
- **The user's choices,** each asked with options:
  - the typical size stays about 40 (R11);
  - two building types, hidden in this step and revisited at step 3 (R12);
  - towers follow SES, an older median age and a planned school; the user's own suggestion also
    had the daycares, which were dropped because they follow the population and repeat SES (R13);
  - larger buildings hold smaller apartments (R15);
  - the room categories stay 3–6, with a lower 6-room share (R16);
  - the fall in children in large-household neighborhoods is left to step 3 (R17).
- **Design choices made in the session:**
  - the room tilt's values (0.35 on SES, 0.10 on household size), from eight candidates;
  - the hidden tower propensity (R14): without it the intraclass correlation of log size was 0.22
    and 30% of neighborhoods had no tower; with an SD of 1.0 it is 0.28 and 38%;
  - the bounds 8 and 160: 0.4% of sizes are clipped at the top.
- **Not run:** a full simulator run with the types and the size term; they need the code. B2 does
  it.

#### B1 (2026-11-24, Opus 5.5; reviews Opus 5.5 and Fable 5.2)
- **Config:** `building` follows 11.4. Removed `apartments_per_building_mean` and
  `apartments_per_building_sd` (`extra="forbid"` rejects them); bounds 8 and 160; the room tilt's
  new values and `building_size_tilt`.
- **Draws** (`BuildingSimulator`): 11.1 in its order. All the neighborhoods' propensities, then
  all their textures; one uniform per building for the type; one normal per building for the
  size; then the unchanged Dirichlet and multinomial loop.
  - The neighborhood draws are aligned to the buildings by repeating each by its building count:
    the rows run neighborhood by neighborhood.
  - The tower probability uses `scipy.special.expit` (scipy is already a dependency).
  - `expected_room_probabilities` takes `building_size_tilt` and the log of the size over the
    texture median, as a required argument.
  - The building context (SES, household size, median age, school status) is merged once and used
    by both stages.
- **Check script:** `.mlflow-check/realism/b1_building_check.py`, 20,000 neighborhoods, rate 9.
  The type is read exactly by replaying the draws; the replayed sizes equal the simulator's.
  - Every 11.2 quantity matches to sampling noise: tower share 0.151 (plan 0.153); by SES band
    0.035–0.355; texture 31.4 (18–49), towers 93.3 (55–147); mean 40.7, SD 26.0, skew 2.19; ICC
    0.29; mean rooms by SES band 3.85–4.20, exactly as planned; 1.27 people per room where
    $H > 4.5$; 6-room share 0.048; texture / tower rooms 4.03 / 3.97.
  - The building counts at 150 neighborhoods, rate 9: 1,523 (seed 42) and 1,564 (seed 7), equal
    to the previous model's.
  - It ran before the reviews' cleanup and was not rerun: the oracle test, which replays the
    draws with a hand-written logistic, still passes, so the draws are unchanged.
- **Tests:** `tests/unit` plus both statistical files: 1429 passed before the reviews, 1426 passed
  after (three duplicate tests removed), 1 skipped, 1 xfailed. Not run, expected to fail until
  B2/B3: `tests/validation/test_calibration.py` and the characterization and notebook tests.
  - Static checks: the 8 ruff and 4 mypy findings in the touched files are on untouched lines and
    exist at HEAD.
- **Correctness review (Opus 5.5):** no deviation from 11.1 or 11.4; the RNG order is pinned
  (mutations of the order fail the oracle); the statistical tests pass at five seeds with stable
  margins.
  - Fixed:
    - **A building with zero apartments now crashed** (`log 0` in the tilt; reproduced: numpy's
      "pvals contains NaNs"). `apartments_per_building_min` must now be above 0, with a test.
    - **The age half of the tower test could not catch a flipped age sign** (median age rises
      with SES). Now taken inside the middle SES tercile; a mutation with the age coefficient
      at −0.3 now fails it.
    - Four stale comments (the draw order's wording, the room tilt's comments, the base shares'
      comment).
  - Kept: the exact-type test switches the propensity off, so the propensity's marginal effect is
    pinned by the oracle only. The household tilt's sign is too weak to test statistically.
- **Simplicity review (Fable 5.2):**
  - Fixed: the neighborhood-to-building lookup replaced by the repeat (one argument fewer); the
    hand-written logistic replaced by `expit`; `cfg` renamed `settings`; two needless
    conversions removed; the method docstring cut to what the public one lacks; the room
    function's docstring cut to its one constraint; the `gt=0` on the medians removed (implied by
    the bounds once the minimum is above 0).
  - Tests: the building-count test and the same-seed test removed (the oracle and the existing
    RNG test cover them); the bounds test reduced to its one new assertion; the replay calls the
    room function instead of re-deriving it; short names replaced; the exact-type helper inlined.
  - **The user's decisions:** the median-order rule dropped (nothing fails without it; it forbade
    equal medians, the way to switch towers off); the SDs keep `gt=0`, the file's convention, so
    the exact-type test uses 1e-9 with a comment.

#### B2 (2026-11-24, Opus 5.5; review Opus 5.5)
- **Config:** `total_children.intercept` −1.00 → **−0.88**, not the planned −0.95.
- **Full run** (`.mlflow-check/realism/b2_full_run_check.py`; 150 neighborhoods, rate 9, the
  final model; output `b2_full_run_output.txt`, and `b2_full_run_output_at_-0.95.txt` for the
  first run):

  | Run | Per apartment, mean / SD | Zero share | Max | Per building, mean / SD / max | Kindergarten / elementary / high school | Invariant share | $H > 4.5$ |
  |---|---|---|---|---|---|---|---|
  | seed 42, −0.95 | 0.818 / 1.171 | .524 | 17 | 33.1 / 26.3 / 233 | .397 / .335 / .268 | 0.973 | 1.53 |
  | seed 42, −0.88 | 0.872 / 1.229 | .507 | 19 | 35.3 / 28.0 / 250 | .398 / .336 / .266 | 0.973 | 1.64 |
  | seed 7, −0.88 | 0.947 / 1.288 | .479 | 18 | 37.3 / 29.1 / 312 | .400 / .339 / .261 | 0.987 | 1.90 |
  | seed 11, −0.88 | 0.865 / 1.207 | .504 | 18 | 37.1 / 29.0 / 270 | .389 / .337 / .274 | 0.987 | 1.78 |

  - **The rule:** seed 42 at −0.95 gave 0.818, outside 0.85–0.90. The intercept moved by
    $\log(0.875/0.818) = +0.067$, rounded to +0.07; the second run gives 0.872.
  - **Why −0.95 was low** (the review, 31 seeds): the building types and the size term lower the
    mean by about 4% in log terms, and seed 42 was a high draw in the design runs (the room mix
    alone averages 0.855 at −0.95, against seed 42's 0.873). About 0.04 of the +0.07 is the
    model, 0.03 the seed.
  - **The change is centred** (the review, seeds 42 and 0–29): at −0.88 the mean averages 0.882
    (SD 0.038, range 0.822–0.979); seed 42 is the median; the intercept that centres the average
    on 0.875 is −0.888.
- **Canonical config** (60 neighborhoods, rate 3, seed 42): mean children 0.833 and zero share
  0.509, inside the ranges; oracle recovery's maximum error 0.070 against 0.20. Over 31 seeds
  (the review): mean 0.769–0.963, none outside 0.75–1.00 (step 1: 0.757–1.023, one seed
  outside); zero share 0.481–0.543; recovery error 0.043–0.159.
- **Tests:** `tests/unit` and `tests/validation -m "not slow"` (the calibration file with both
  invariants, both statistical files): 1429 passed, 1 skipped, 1 xfailed.
  `tests/validation/test_recovery.py` (slow): 1 passed. Not run: the characterization and
  notebook tests (B3). No source or test file changed, so no static check ran on them; ruff
  passes on the script.
- **Review (independent subagent, Opus 5.5):** no real problem. The script computes 11.3's
  quantities; the recalibration follows the rule; no file pins the intercept's value.
  - Observations recorded, not fixed:
    - **The $H > 4.5$ column rests on 3 neighborhoods at seed 42** (about 5 per run). 11.3's
      "2.0 → 1.6" came from the same noise. On 1000 neighborhoods (seeds 42 / 7 / 11) the tail
      averages 2.02 / 1.89 / 1.80 against step 1's 2.27 / 2.21 / 1.96: **step 2 lowers it by
      about 11%**, not by a fifth. The realistic level stays about 2.7 (lead for step 3).
    - **The tail invariant catches the old household-size scale (0.5) with less margin:** its
      pass share at 0.5 is 0.58 / 0.75 / 0.69 against the 0.80 threshold (step 1: 0.24–0.49).
      It still fails at 0.5, as it should. For B4 and step 3.
    - **The kindergarten share** is .389–.400 (31-seed average .395, the same at −0.95 and
      −0.88); step 1 had .371–.373. The new room mix causes nearly all of it, smaller apartments
      pushing toward the younger cohorts (lead for step 4).
    - **Building totals spread more:** SD 28–29 and maximum 250–312 (31 seeds: average maximum
      284, worst 525), against 18.7–21.0 and 150–177 with the room mix alone. Towers in
      large-household areas, as 11.6 expected.
    - **One seed is a noisy rule:** a single seed's mean has a 95% band of about ±0.075, wider
      than the 0.85–0.90 window. It worked here because seed 42 is the median. A later step's
      recalibration could use a multi-seed average instead.
    - Stale, older than this step: `docs/FEATURE_TRANSFORMATIONS.md` quotes an observed log
      rate of about −0.5 (now about −0.13); `docs/DATA_GENERATION_PLAN.md` has −0.79. For B3's
      doc pass.

#### B3 (2026-11-24, Sonnet 5.5)
- **EDA fixture** (`tests/characterization/fixtures/eda_jupyter_baseline.json`), regenerated by
  `.mlflow-check/realism/b3_regenerate_eda_fixture.py` after it reproduced the old file byte
  for byte. 5 fields changed, 46 did not (`b3_fixture_diff.txt`):
  - `config.content_sha256`: `configs/simulation.yml` changed (B1, B2);
  - `final_dataframe.sha256` and the two summary hashes (`cohort_share_describe_sha256`,
    `feature_target_correlations_sha256`): every draw changed;
  - `summaries.targets`: total children per building mean 35.60 to 35.29, SD 23.3 to 28.0,
    maximum 187 to 250; kindergarten / elementary / high-school means 13.3 / 12.1 / 10.2 to
    14.0 / 11.9 / 9.4; the kindergarten share rises (B2's lead for step 4). Zero shares are
    still at most 1.8%.
  - **Not changed:** `final_dataframe.shape` (1523 rows) and the building-count statistics,
    because the building counts are drawn first and B1 kept them.
  - All nine quality checks are still true.
- **Pinned counts**, measured (`b3_measure_pins.py`): **none changed**. Rows 1523 (seed 42) and
  1564 (seed 7), split 913/305/305, `n_train_val` 1218, wrong second share 974/244/305, whole
  table to the second call 1142/381/305. No test file needed an edit.
- **Full suite**, no marker filter (slow, calibration and notebook tests included): 1517
  passed, 1 skipped, 1 xfailed, in 11 min 22 s (`b3_pytest_output.txt`). No source or test file
  changed, so no static check ran.
  - The slow NUTS recovery test emits the same `RuntimeWarning` as in N3 (worst R-hat 1.086,
    effective sample size 25.7) and passes. The numbers equal N3's, so step 2 did not change
    them; whether they are older than step 1 is still unchecked (a HEAD run was not made).
- **Docs:** `docs/SIMPLIFIED_MODEL_PLAN.md`: the building parameter table, Step 2 (types, hidden
  effects, lognormal sizes) and Step 3 (base shares, the building tilt with its size term, the
  parameter table) now follow 11.1 and 11.4; a note says its neighborhood and outcome sections
  are stale until the reruns. A grep of `README.md`, `docs/`, `src/`, `notebooks/`, `checks/`,
  `tests/` and `configs/` finds `apartments_per_building_mean` and `_sd` only in this plan and
  the config rejection test.
- **Stale until the reruns (R9), left as they are:** `FEATURE_TRANSFORMATIONS.md` (the observed
  log rate of about −0.5, the VIF and correlation tables), `HYPERPARAMETER_TUNING.md`,
  `INDEPENDENT_COHORT_MODELS.md`, `MLFLOW_RUNS.md`, `PARAMETER_REFERENCE.md` and
  `DATA_GENERATION_PLAN.md` (earlier panel design, −0.79).

#### B4 (2026-11-24, Fable 5.2)
The whole step (`git diff main...HEAD`, 11 files) read against 11.1–11.5.
- **No defect in the model or the config.** The draws and their RNG order match 11.1; the
  fields, the removed fields and the validation rules match 11.4; the intercept is B2's −0.88;
  each test of 11.5 exists; the dataset columns are unchanged (R12).
- **New check: does each statistical test fail when its own sign or value is wrong?** 11.4
  leaves the signs and the order of the medians to the statistical tests; B1 had verified the
  RNG order and the age sign only. `.mlflow-check/realism/b4_review_check.py` applies each
  mutation as a config override on the tests' own draw (20,000 neighborhoods, rate 9) and calls
  the test functions (output `b4_review_check_output.txt`):

  | Mutation | Tests that fail |
  |---|---|
  | none (the config) | 0 |
  | `tower_ses_coef` −0.7 | towers rise with SES |
  | `tower_median_age_coef` −0.3 | towers rise with age at fixed SES |
  | `tower_planned_school_coef` −0.7 | a planned school raises the tower share |
  | the two medians swapped | 6: clipping, the skew and the mean, the tower share, and the three tower tests |
  | no hidden texture or propensity (SDs 1e-9) | buildings of one neighborhood are alike |
  | `building_size_tilt` +0.32, and 0 | larger buildings hold smaller apartments |
  | `ses_tilt` −0.35 | apartments grow with SES; large households are crowded |
  | the old tilts (household 0.35, SES 0.20) | apartments grow with SES |
  | the old base shares (6 rooms 0.10) | 6-room apartments are rare |

  Every mutation fails at least one test, so no test is changed.
- **Fixed** (docs and one blank line):
  - `docs/README.md`: the row for this plan described step 1 only.
  - This plan's status header said B2 done, next B3.
  - 11.3's lead for step 3 now carries B2's 1000-neighborhood measure (about 11%, not a fifth).
  - `docs/SIMPLIFIED_MODEL_PLAN.md`: the Dirichlet row said the mixes stay near their
    neighborhood expectation; with the size term it is the building's.
  - 11.5 notes that `tests/unit/helpers.py` needed no change.
  - `tests/unit/test_building.py`: a third blank line before `_replay_buildings`.
- **Measured, left for the reruns (R9):** `notebooks/01_eda.py` has a fixed sentence, "Mean room
  shares vary little across building-size bins", with the advice to treat building size as
  distinct from the room mix. On the benchmark data (1,523 buildings, seed 42) the notebook's
  number, the largest range of a mean share across its 8 size bins, is 0.054; with the size
  term off it is 0.039. The level is still small, but the 3-room share now rises with size
  (.286 in the smallest bin, .328 in the largest) where it fell before (.311 to .277). The
  notebook is not edited in this PR; 11.6 records it.
- **Checked, left:**
  - **The tail invariant's margin** (B2's item): at the old household-size scale (0.5) its pass
    share is 0.58–0.75 against the 0.80 threshold, so it still fails as it should. The
    threshold is kept; step 3 changes the children model and can revisit it.
  - **The NUTS `RuntimeWarning`** (B3's item) is not this plan's: N4 found that
    `tests/validation/test_bayesian_recovery.py` imports nothing from `student_simulator`; its
    imports are unchanged.
  - `docs/MODULE_REFERENCE.md` names `BuildingSimulator.generate`; the method is
    `sample_buildings`. The line is the same on `main`.
- **Static checks:** on the step's source and test files, ruff reports 6 findings (import order
  and one unused import) and mypy the same 4 as B1, in `config.py`; all are on lines this step
  did not touch.
- **Tests:** `tests/unit/test_building.py`: 6 passed. The full suite was not rerun: B4
  changed no source or config, and B3's run gave 1517 passed.

## 12. Step 3: children per apartment, and the building-size scales

Decided on 2026-11-24 from the project summary's S1c.2 analysis
(`docs/PROJECT_SUMMARY_PLAN.md` §9 on `docs/project-summary`). Decisions R18–R24 (§1).

### 12.1 The model

$H_j$ is the neighborhood's household size, $r_i$ the apartment's rooms.

| # | Variable | Formula | Why |
|---|---|---|---|
| 1 | Household-size term | $x_j = \log\dfrac{H_j - 1.7}{3.0 - 1.7}$ | children are the household members beyond the adults; 1.7 is the household size of an area with no children aged 3–18; 3.0 is the reference neighborhood, where the term is 0 |
| 2 | Log expected count | $\log\mu_i = -0.89 + \rho_{r_i} + 1.0\,x_j - 0.15\,(r_i - 4)\,x_j + \text{daycare, school and median-age terms} + u_b + w_j$ | $\rho$: the room effects (0, 0.45, 0.80, 0.95), unchanged. The interaction is crowding: the room gap shrinks where households are large and widens where they are small |
| 3 | Count | $n_i = \min(\mathrm{NB2}(\mu_i,\ \phi = 3),\ 10)$ | ten children inside a 16-year age span is about the limit |
| 4 | Building size (step 2's draw 5) | $A_b = \mathrm{clip}(\mathrm{round}(m \cdot e^{t_j + e_b}), 8, 160)$ with $t_j \sim \mathcal N(0, 0.25^2)$, $e_b \sim \mathcal N(0, 0.15^2)$ | a neighborhood's buildings share an era, a plan and plot sizes; the towers give the contrast inside a neighborhood |

- **Replaced:** `0.30 · H̃` and `+0.10 · (rooms − 4) · H̃`, with $\tilde H = (H - 3.0)/0.75$. The
  SES² term (coefficient 0 since step 1) is left as it is.
- **RNG order unchanged:** the building effects, the neighborhood effects, one NB2 draw per
  apartment. The clip uses no draw. The building counts and the hidden types are unchanged for a
  seed; the sizes change, so every later draw changes.
- **The scaled household size stays** in the room tilt and the cohort score, with
  `features.household_size_scale`.
- **Near the floor:** household size is clipped at 1.8, where the term is $\log(0.1/1.3)$ (a factor
  of 0.077). 3 of 1,000 neighborhoods sit there. The config must keep 1.7 below the minimum (12.4).

### 12.2 Calibration

All runs in memory (plan mode), the new model replayed on the simulator's own draws: 1,000
neighborhoods, rate 9, seeds 42 / 7 / 11 (about 410,000 apartments each). The targets are from
memory (R10).

**Household size** (mean children per apartment by the neighborhood's household-size band):

| Band's mean household size | 2.05 | 2.43 | 2.82 | 3.22 | 3.70 | 4.22 | 4.98 | Slope per person |
|---|---|---|---|---|---|---|---|---|
| Today | 0.45 | 0.61 | 0.72 | 0.88 | 1.09 | 1.39 | 2.09 | 0.44–0.51 |
| New (seed 42) | 0.20 | 0.42 | 0.66 | 0.92 | 1.20 | 1.54 | 2.03 | 0.61–0.63 |
| New, seeds 7 / 11 | 0.20 / 0.20 | 0.43 / 0.45 | 0.65 / 0.66 | 0.92 / 0.95 | 1.24 / 1.25 | 1.61 / 1.52 | 1.93 / 1.85 | |
| Target (confidence) | 0.15–0.25 (medium-high) | | | 0.90–0.95 (high) | | | 2.0–2.4 (medium) | 0.6–0.7 |

Children divided by (household size − 1.7), each band against the middle band: 0.95–1.06 (today
2.2–2.6 in the lowest band).

**Rooms and crowding** (children in a 6-room apartment over a 3-room one):

| Household size | under 2.6 | 2.6–3.5 | 3.5–4.5 | over 4.5 | Children at 3 / 6 rooms, over 4.5 |
|---|---|---|---|---|---|
| Today | 2.0–2.1 | | | 5.0–5.9 | 0.97 / 5.78 |
| New | 3.4–3.7 | 2.6 | 2.1 | 1.7–1.8 | 1.63 / 2.76 |

**Level, tail, cap:**

| Quantity | Today | New | Target (confidence) |
|---|---|---|---|
| Mean per apartment, 1,000 neighborhoods | 0.87–0.89 | 0.87–0.88 | about 0.9 (medium) |
| Zero share | 0.50 | 0.51–0.52 | about 0.60; not in the step (R18) |
| Maximum | 25–40 | 10 | above 10–12 does not occur (high) |
| Clipped draws; mean lost | — | 0.03–0.04%; 0.1% | |
| Apartments at exactly 10 | 0.04% | 0.07% | a small pile-up, invisible in building sums |
| Both invariants of §5 | 0.98; 1.00 | 0.995–1.00; 1.00 | |

**The intercept** (150 neighborhoods, rate 9, seeds 42 and 0–29): at −0.89 the mean is 0.879 (SD
0.042, range 0.820–0.991); the value centring 0.875 is −0.894. **Seed 42 is a low draw: 0.845**
(0.872 today). Seeds 7 / 11: 0.958 / 0.868.

**The canonical config** (60 neighborhoods, rate 3; 31 seeds): mean 0.777–0.961, zero share
0.490–0.547, all inside the ranges. Oracle recovery's largest error: 0.17 over 31 seeds (2 above
0.15), 0.05 at seed 42; the tolerance stays 0.20.

**Building size** (step 2's statistical draw: 20,000 neighborhoods, seed 20261125):

| Quantity | 0.20 / 0.22 | 0.25 / 0.15 |
|---|---|---|
| Mean, SD, skew | 40.7, 26.1, 2.19 | 40.6, 25.9, 2.18 |
| Percentiles 5 / 25 / 50 / 75 / 95 / 99 | 19 / 26 / 32 / 43 / 102 / 141 | 19 / 26 / 32 / 42 / 102 / 139 |
| Clipped at 160 | 0.41% | 0.37% |
| Intraclass correlation of log size | 0.28 | 0.375 |
| Texture-size variance between neighborhoods (formula) | 45% | 74% |
| 10th–90th percentile inside one neighborhood, texture buildings | 23–40 | 25–36 |
| Correlation with SES / median age / household size | +0.20 / +0.15 / −0.13 | the same |
| The 11 tests of 11.5 | pass | pass |

### 12.3 Downstream consistency

| Field | Now | New | Why |
|---|---|---|---|
| `total_children.intercept` | −0.88 | −0.89 | keeps the level: 0.879 on 31 seeds (R24) |

- **C3 re-verifies with the code.** The intercept changes from −0.89 only if the 31-seed mean
  (150 neighborhoods, rate 9, seeds 42 and 0–29) leaves 0.865–0.885.
- Nothing else downstream changes (R6). The cohort split is step 4.

### 12.4 The config schema

```yaml
building:
  apartments_neighborhood_log_sd: 0.25   # was 0.20
  apartments_building_log_sd: 0.15       # was 0.22

total_children:
  intercept: -0.89                        # was -0.88
  childless_household_size: 1.7           # new: household size with no children aged 3-18
  household_size_coef: 1.0                # was 0.30; now on log((H - 1.7) / (3.0 - 1.7))
  rooms_household_interaction_coef: -0.15 # was +0.10; now on (rooms - 4) x that log term
  max_children: 10                        # new: the count is clipped here
```

- **No field removed.** The two coefficients keep their names and change their feature.
- **Validation added:** `max_children` a whole number of at least 1; a cross-section check in
  `Config` that `childless_household_size` is below `neighborhood.household_size_min` (the log is
  taken).
- **A runtime guard** in the transform: a household size at or below the childless size raises a
  clear error (hand-built contexts bypass the clip).
- **Code:** a public `log_household_size_excess(values, reference, childless)` in
  `src/student_simulator/features.py`; the design column `household_size_scaled` becomes
  `log_household_size_excess`; `rooms_household_interaction` keeps its name; the clip follows the
  draw in `sample_total_children`.

### 12.5 The tests

**New statistical tests** (`tests/validation/test_children_statistics.py`: 1,000 neighborhoods,
rate 9, a fixed seed, the stages up to the totals; shapes and signs only). Each fails on today's
model:

- **Small households hold few children:** the mean where household size is under 2.3 is below
  0.35 (new 0.23–0.26; today 0.47–0.53).
- **Proportionality:** children ÷ (household size − 1.7) in each of the seven bands is within 25%
  of the middle band (new 0.95–1.06; today 2.2–2.6 in the lowest).
- **Crowding:** the 6-to-3-room ratio where household size is over 4.5 is below the ratio where
  it is under 2.6, and below 2.6 (new 1.7–1.8 against 3.4–3.7; today 5.0–5.9 against 2.0–2.1).
- **The cap:** no count above 10; under 0.3% of apartments at the cap (today's maximum 25–40).
- **The level:** the mean is in 0.80–0.95.

**Unit tests** (`tests/unit/test_total_children_stage.py`):
- the log term is 0 at the reference and adds $\log 2$ when $H - 1.7$ doubles;
- the room difference in log $\mu$ at household size $H$ equals the room effects plus
  $-0.15\,(r - 4)\,x$;
- the counts equal the minimum of the replayed NB2 draw and the cap; with a very large intercept
  every count equals the cap;
- a household size at or below 1.7 raises;
- `test_household_size_main_effect_positive_for_supported_rooms` is rewritten for the new terms
  (the slope in $x$ is positive for every room count);
- the hand-built contexts move to the current references (3.0 and 31; they use 2.6 and 37).

**Config tests:** `childless_household_size` at or above the household-size minimum is rejected;
`max_children` of 0 is rejected.

**Changed:** the intraclass-correlation test of `test_building_statistics.py` asks for more than
0.30 (was 0.15): 0.28 fails, 0.375 passes.

**Existing tests to update:** `tests/unit/test_design.py` (the column list),
`tests/validation/helpers.py` (the recovery's expected coefficients),
`tests/characterization/` (fixtures and pins, regenerated in C4).

**Unchanged and still required:** the calibration ranges, both invariants of §5 (they pass at
0.995 and 1.00 and now guard only against a gross error) and the other ten tests of 11.5.

### 12.6 Effects on the benchmark

- **Household size is no longer log-linear.** A log-link model with a linear household-size term
  loses about 6% of the explained deviance against the true form (the consult's measure). Trees
  and splines can follow it; the modeling side has a spline option for SES only.
- **The rooms × household-size feature of the models keeps a ground truth, with the opposite
  sign.** `docs/FEATURE_TRANSFORMATIONS.md` (the total-stage pair) is revised at the reruns (R9).
- **Buildings in small-household areas are nearly empty of children** (0.2 per apartment), and
  the spread between neighborhoods grows (slope 0.63 against 0.48).
- **Building totals** (150 neighborhoods, rate 9, seeds 42 / 7 / 11): mean 34–38, SD 30–32,
  maximum 265–305 (step 2: SD 28–29, maximum 250–312).
- **The benchmark's seed gives a lower level:** 0.845 children per apartment at seed 42 (0.872
  today), because the intercept is centred on 31 seeds.
- **Sizes inside a neighborhood are more alike,** so `n_apartments` carries more neighborhood
  information (intraclass correlation 0.375), which matters for grouped splits.

### 12.7 Leads, not in this step

- **Affordability** (the user's relation: high-SES households afford large apartments): measured
  as rooms × SES above 0 at −0.10. It is recovered, but at SES over 1.5 a 6-room apartment then
  holds fewer children than a 5-room one (0.88 against 0.95), which has no support. Not taken;
  the room mix already gives high-SES areas larger apartments (step 2).
- **Families with many children avoid towers** (the user's relation; plausible: cost, upkeep
  fees, smaller apartments). It already holds through two paths, measured on seed 42: towers are
  rare where households are large (5.6% of buildings where household size is over 4, against
  19.5% in the middle band), and tower apartments are smaller. Tower-sized buildings hold 0.77
  children per apartment against 0.94, and 28% of the apartments with four or more children
  against 38% of all apartments. Inside large-household neighborhoods the gap is small (1.55
  against 1.71). A direct effect needs the hidden type (R12); the type stays hidden.
- **New buildings attract young families** (the user's relation): the data has no building age.
  For step 4: a higher kindergarten share where a school is planned or the building is a tower.
- **Median age in the room mix:** at a fixed SES the mean rooms do not move with age
  (3.88 / 3.89 / 3.85). The poorest areas already have the smallest apartments (3.88 against
  4.17).
- **The tail under the cap:** $\mu$ still reaches about 8.5 through the hidden effects; the cap
  hides it. The effects' SDs are out of scope (R18).
- **The zero share** stays 0.51 against about 0.60 (R18).

### 12.8 Step 3 sub-tasks

| # | Sub-task | Model | Review | Done |
|---|---|---|---|---|
| C0 | This plan committed on `feat/realistic-children` | user | no | |
| C1 | The two building SDs (12.4), the ICC threshold (12.5), a check script against 12.2's building table | Sonnet 5.5 | no | x |
| C2 | The children model: the config and validation (12.4), the transform, the design, the cap (12.1), the unit, config and statistical tests (12.5), a check script against 12.2 | Opus 5.5 | yes | x |
| C3 | The intercept on 31 seeds (12.3), a full run on seeds 42 / 7 / 11, the calibration ranges, the invariants, the recovery check | Opus 5.5 | yes | x |
| C4 | Regenerate the characterization fixtures; the full test suite; update the docs that describe the changed terms | Sonnet 5.5 | no | |
| C5 | An independent review of the whole step against §12, with a mutation check of the new tests (B4's form), then the PR description | Fable 5.2 | is the review | |

**C1. Done when:**
- [x] The two SDs are in the config and the ICC test asks for more than 0.30.
- [x] A check script prints 12.2's building table on step 2's draw; the building counts at 150
      neighborhoods, rate 9, seeds 42 and 7 are still 1,523 and 1,564.
- [x] `tests/unit/test_building.py` and `tests/validation/test_building_statistics.py` pass.

**C2. Done when:**
- [x] `total_children` follows 12.4, with the validation rules and the runtime guard.
- [x] The design and the draw follow 12.1; the RNG order is unchanged.
- [x] The unit, config and statistical tests of 12.5 pass.
- [x] A check script prints 12.2's household-size, rooms and tail tables on seeds 42 / 7 / 11.
- [x] Review findings reproduced and fixed.

**C3. Done when:**
- [x] The 31-seed mean is printed; the intercept changes only by 12.3's rule.
- [x] A full run (150 neighborhoods, rate 9, seeds 42, 7, 11) prints children per apartment and
      per building, the zero share, the maximum, the cohort shares and both invariants.
- [x] The calibration ranges, the invariants and the recovery test pass, or the failure is shown
      with its numbers.
- [x] Review findings reproduced and fixed.

**C4. Done when:**
- [ ] The characterization fixtures are regenerated, and each diff is explained in 12.10.
- [ ] The full test suite passes (output shown).
- [ ] `docs/SIMPLIFIED_MODEL_PLAN.md` follows 12.1 and 12.4; `docs/archive/` stays as it is; the
      docs that wait for the reruns (R9) are listed.

**C5. Done when:**
- [ ] The step is reviewed against 12.1–12.5 as a new reader; each finding is reproduced, then
      fixed.
- [ ] Each new test fails under its own mutation (the old coefficients, the interaction's sign
      flipped, no cap, the old SDs).
- [ ] The PR description is printed in the chat.

### 12.9 Session prompts

**The opening:**

```text
We are making the simulator realistic, step 3 (children per apartment and the building-size
scales), on branch feat/realistic-children. Read docs/SIMULATOR_REALISM_PLAN.md first: §1
decisions (R18–R24 are this step's), §12 the step (12.1 the model, 12.3 downstream, 12.4 the
config, 12.5 the tests, 12.8 your sub-task, 12.10 the record), and 11.9 for how step 2's
sub-tasks were recorded. Rules: do only this sub-task; the plan's values are decided (do not
re-derive them; if one fails a check, stop and show the evidence); scripts and their outputs go
in .mlflow-check/realism/ (git-ignored) and run with ~/.venvs/age-group-prediction/bin/python;
static checks (ruff, mypy with --cache-dir in that folder) before the tests; run each check
once; do not commit. At the stop: show the diff and the test output, tick the sub-task's "done
when" boxes and add its entry to 12.10, write the commit message in a text block (no
Co-Authored-By line), and print the next sub-task's prompt (this opening, a blank line, its
text) in one code block, naming the model to select.
```

| Sub-task | Text after the opening |
|---|---|
| C1 | `Implement C1: the two building-size SDs (12.4), the ICC threshold (12.5) and a check script against 12.2's building table.` |
| C2 | `Implement C2: the children model's config and validation (12.4), the log household-size transform, the design and the cap (12.1), with the unit, config and statistical tests (12.5) and a check script against 12.2's tables. Run an independent review subagent before the stop.` |
| C3 | `Implement C3: the intercept on 31 seeds (12.3), a full run on three seeds, the calibration ranges, the invariants and the recovery check. Run an independent review subagent before the stop.` |
| C4 | `Implement C4: regenerate the characterization fixtures, run the full test suite, and update the docs that describe the changed terms.` |
| C5 | `Implement C5: review the whole step against §12 as a new reader; reproduce each finding before fixing it; run a mutation check of the new tests; then write the PR description.` |

After the PR is merged: merge `main` into `docs/project-summary` and start the summary's S1c.3.

### 12.10 Record

#### Planning (2026-11-24, Fable 5.2)
- **Source:** the S1c.2 analysis and the user's two decisions (the children step; the
  building-size scales).
- **Runs:** in memory only (plan mode); no script was saved. C1 and C2 write the check scripts.
- **Candidates measured for the household-size relation:**
  - log-linear at 0.40 / 0.50 / 0.60: the lowest band stays at 0.29–0.42, and at 0.60 the top
    reaches 3.35;
  - proportional with exponent 1.15: 0.15 / 0.93 / 2.23; it grows faster than proportionally
    and adds a value;
  - childless size 1.5: the lowest band is 0.27.
- **The Fable consult** (independent, with its own draws):
  - its targets: 0.15–0.25, 0.90–0.95 and 2.0–2.4 children at household size 2.05, 3.2 and 5.0;
    slope 0.6–0.7 (R20);
  - the proportional form, 1.7 and exponent 1; a clip at 10; 0.25 / 0.15 with the ICC threshold
    at 0.30;
  - it advised dropping the interaction and noted that crowding argues for a negative one.
- **The user's choices:** the cap of 10; at least one interaction, possibly non-linear; the
  relations of 12.7 recorded as leads.
- **Chosen in the session:** crowding alone. The user left "both" (crowding and affordability)
  to the session's judgment of realism; affordability gave an unsupported room profile (12.7).
- **Not run:** the code itself; the cohort shares under the new counts (C3).

#### C1 (2026-11-24, Sonnet 5.5)
- **Config:** `configs/simulation.yml`: `apartments_neighborhood_log_sd` 0.20 to 0.25,
  `apartments_building_log_sd` 0.22 to 0.15. No source file changed.
- **Test:** the intraclass-correlation test of `test_building_statistics.py` asks for more than
  0.30 (was 0.15).
- **Check** (`.mlflow-check/realism/c1_building_check.py`, output `c1_building_check_output.txt`;
  20,000 neighborhoods, rate 9, seed 20261125). New column against 12.2: mean / SD / skew
  40.6 / 25.9 / 2.18 (same); percentiles 19 26 32 42 102 139 (same); clipped at 160 0.37%
  (same); ICC 0.379 (plan 0.375); texture variance between neighborhoods 0.74 (same);
  10th–90th inside one neighborhood 25–36 (same); correlations +0.20 / +0.15 / −0.13 (same).
  Old column: 40.7 / 26.1 / 2.19; 19 26 32 43 102 141; 0.41%; ICC 0.283; 0.45; 23–39.
  Small differences from 12.2 (ICC 0.379, old 10th–90th 23–39) are the estimators' rounding.
- **Building counts** at 150 neighborhoods, rate 9: seed 42 gives 1,523 and seed 7 gives 1,564,
  both unchanged.
- **Tests:** `tests/unit/test_building.py` and `tests/validation/test_building_statistics.py`:
  17 passed (`c1_pytest_output.txt`). `ruff check` and `mypy` clean. `ruff format --check`
  flags the test file for formatting that was already there (not the edited line); left as it is.

#### C2 (2026-11-24, Opus 5.5; review Fable 5.2)
- **Config** (`configs/simulation.yml`, `TotalChildrenSettings`): intercept −0.89,
  `childless_household_size` 1.7 (new), `household_size_coef` 1.0, `rooms_household_interaction_coef`
  −0.15, `max_children` 10 (new, `Field(ge=1)`; 0 and 2.5 are rejected). `Config` rejects a
  childless size at or above `neighborhood.household_size_min`.
- **Code:** a public `log_household_size_excess(values, reference, childless)` in `features.py`
  (exported from the package). It raises when a household size, or the reference, is at or below
  the childless size. The total-children design column `household_size_scaled` became
  `log_household_size_excess`, and the interaction is `rooms_centered` × that term. The count is
  `np.minimum(NB2 draw, max_children)`. The cohort stage and the room tilt keep
  `scale_household_size`. The recovery's expected key in `tests/validation/helpers.py` is renamed;
  the recovery test itself runs in C3.
- **Check** (`.mlflow-check/realism/c2_children_check.py`, output `c2_children_check_output.txt`;
  1,000 neighborhoods, rate 9, the totals replayed from the generator state before the stage).
  - On all three seeds the capped replay equals the simulator's counts, so the RNG order is
    unchanged.
  - The "old" column is today's children model on C1's buildings.

  | Seed 42 | Old | New | 12.2 |
  |---|---|---|---|
  | Band children (band H 2.08 … 4.98) | 0.50 0.58 0.71 0.87 1.08 1.36 1.95 | 0.23 0.43 0.66 0.90 1.19 1.50 2.03 | 0.20 0.42 0.66 0.92 1.20 1.54 2.03 |
  | Slope per person (bands / neighborhoods) | 0.50 / 0.48 | 0.62 / 0.61 | 0.61–0.63 |
  | Children ÷ (H − 1.7), relative to the middle band | 0.95–2.28 | 0.95–1.03 | 0.95–1.06 |
  | 6 / 3-room ratio by H band | 1.95 2.71 3.66 5.50 | 3.65 2.50 2.04 1.81 | 3.4–3.7, 2.6, 2.1, 1.7–1.8 |
  | Children at 3 / 6 rooms, H > 4.5 | 0.90 / 4.95 | 1.55 / 2.82 | 1.63 / 2.76 |
  | Mean; zero share; maximum | 0.880; 0.505; 25 | 0.879; 0.510; 10 | 0.87–0.88; 0.51–0.52; 10 |
  | Clipped draws; mean lost; at 10 | —; —; 0.04% | 0.04%; 0.11%; 0.07% | 0.03–0.04%; 0.1%; 0.07% |
  | Invariants (all / H > 4.5) | 0.984 / 1.00 | 1.000 / 1.00 | 0.995–1.00 / 1.00 |

  - Seeds 7 / 11, new band children: 0.23 / 0.21, 0.45 / 0.46, 0.66 / 0.66, 0.91 / 0.94,
    1.20 / 1.20, 1.57 / 1.52, 1.93 / 1.85.
  - Seeds 7 / 11, other values: mean 0.871 / 0.871; invariants 0.996 / 0.995 and 1.00.
  - The lowest band reads 0.21–0.23 against 12.2's 0.20, inside the 0.15–0.25 target. Its mean
    household size is 2.05–2.09 here.
- **Band edges** (12.5 gives none): 2.25, 2.625, 3.0, 3.45, 3.95, 4.5, chosen to reproduce
  12.2's band means.
- **Tests:**
  - `tests/validation/test_children_statistics.py` (new, seed 42) holds the five tests of 12.5.
  - On today's model (the check's "old" column) four of them fail. The level test does not
    fail, since the level is unchanged: it guards the intercept, and 12.5's "each fails" is too
    strong for it.
  - In `test_total_children_stage.py`, the contexts moved to 3.0 and 31, and these tests were
    added: the log term, the room differences at H = 4.0, the replayed draw clipped at the cap,
    every count at the cap under an intercept of 8, and the guards.
  - In `test_config.py`, a childless size of 1.8 or 2.0 and `max_children` of 0 or 2.5 are
    rejected. In `test_design.py`, the column list is renamed.
  - Results: 96 passed on the eight affected files (`c2_pytest_output.txt`); 25 passed after the
    review (`c2_pytest_after_review.txt`).
- **Static checks:** ruff and mypy show no new findings. Pre-existing findings were left as they
  are: the import sorting in `config.py`, `features.py` and `__init__.py`, the format of the
  older test files, and four mypy errors in `config.py`.
- **Review** (independent, Fable 5.2):
  - It ran 20 mutated copies of the tree and found no bugs; every source-level mutant was killed
    except for the four test gaps below.
  - Each gap was reproduced (`.mlflow-check/realism/c2_repro/repro.py`), then fixed:
    - The cap test read the cap from the config. With a cap of 1,000 it passed (maximum 24). It
      now uses the literal 10, and fails at 1,000.
    - The household-size slope test was arithmetic on two config numbers and never called the
      simulator. It now takes `compute_log_mu` at H = 3.0 and 4.3 and expects 1.15 / 1.0 /
      0.85 / 0.70.
    - The proportionality test read the childless size from the config. It now uses the literal
      1.7.
    - The `reference <= childless` guard had no test. One was added.
  - Nits: `log_household_size_excess` is now exported, and the module docstring is corrected.
    Left as they are: lax int coercion for `max_children` (as for the other fields), and no bound
    on `childless_household_size` beyond the cross-section check (NaN still ends in the
    non-finite error).
- **Not run:** the calibration, invariant and recovery tests, and the 31-seed intercept check
  (C3); the characterization fixtures (C4).

#### C3 (2026-11-24, Opus 5.5; review Fable 5.2)
- **No file changed but this plan:** the intercept stays **−0.89**.
- **The intercept on 31 seeds** (`.mlflow-check/realism/c3_intercept_seeds.py`, output
  `c3_intercept_seeds.out`; 150 neighborhoods, rate 9, seeds 42 and 0–29):
  - The mean averages **0.8788** (SD 0.0429, range 0.820–0.991). Seeds 42 / 7 / 11 give
    0.845 / 0.958 / 0.868. All equal 12.2.
  - The centring intercept is −0.8943, which rounds to −0.89. The average is inside 0.865–0.885,
    so 12.3's rule keeps the value.
  - The review reran the 31 seeds at −0.894 and got 0.8742; the cap does not disturb the
    centring.
- **The canonical config on the same 31 seeds:**
  - Mean 0.777–0.961 and zero share 0.490–0.547, none outside the ranges.
  - Oracle recovery's maximum error 0.021–0.172 (average 0.086). Seeds 11 and 24 are above 0.15,
    none above 0.20. Seed 42 is 0.033 (12.2 said 0.05, from the plan-mode replay).
  - The worst case is `log_household_size_excess` at seed 11 (0.172): the margin across seeds is
    thin, but the test runs seed 42 only (for C5).
- **Full run** (`c3_full_run_check.py`, output `c3_full_run_output.txt`; 150 neighborhoods,
  rate 9):

  | Run | Per apartment, mean / SD | Zero share | Max | Per building, mean / SD / max | Kindergarten / elementary / high school | Invariant share | $H > 4.5$ |
  |---|---|---|---|---|---|---|---|
  | B2, seed 42 (−0.88) | 0.872 / 1.229 | .507 | 19 | 35.3 / 28.0 / 250 | .398 / .336 / .266 | 0.973 | 1.64 |
  | seed 42 | 0.845 / 1.211 | .524 | 10 | 34.1 / 30.0 / 265 | .398 / .337 / .266 | 0.993 | 1.77 |
  | seed 7 | 0.958 / 1.297 | .481 | 10 | 37.5 / 31.8 / 305 | .409 / .338 / .253 | 1.000 | 1.47 |
  | seed 11 | 0.868 / 1.222 | .513 | 10 | 36.8 / 30.9 / 276 | .395 / .334 / .271 | 0.993 | 1.92 |

  - **Building totals:** they equal 12.6 (mean 34–38, SD 30–32, maximum 265–305). Over 31 seeds
    (the review): mean 32.9–38.8, SD 26.8–34.2, maximum averaging 309 with a worst of 571 (B2:
    284 and 525).
  - **Cohort shares** (not run in the planning): over 31 seeds the review measured .400 / .336 /
    .264, with kindergarten .388–.413 (B2 averaged .395). A small upward drift, the lead for
    step 4.
  - **The tail invariant on its test's setup** (1,000 neighborhoods, rate 3): seeds 42 / 7 / 11
    have 38 / 40 / 39 tail neighborhoods averaging 2.11 / 1.98 / 1.98 children. The pass share is
    0.974 / 1.000 / 0.974 against the threshold of 0.80. Over all neighborhoods it is 0.989–0.991.
    (12.2's 0.995–1.00 and 1.00 were measured at rate 9; one neighborhood fails at rate 3.)
  - **Apartments at the cap** (the review): 0.046% / 0.072% / 0.043%.
- **Canonical config at seed 42:** mean 0.849, zero share 0.506, oracle recovery 0.033; all
  pass.
- **Tests:**
  - `tests/unit` and `tests/validation -m "not slow"` (the calibration ranges, both invariants,
    the three statistical files): 1445 passed, 1 skipped, 1 xfailed, 19 deselected, in 1 min 42 s
    (`c3_pytest_output.txt`). The 6 warnings come from mlflow in `test_gate8_tracking.py`.
  - `tests/validation/test_recovery.py` (slow): 1 passed (`c3_pytest_recovery.txt`).
  - Not run: the characterization and notebook tests (C4).
- **Static checks:** no source or test file changed. Ruff shows 5 style findings on the two
  git-ignored scripts: implicit string concatenation copied from B2's script, and unused `noqa`.
  They were left as they are.
- **Review** (independent, Fable 5.2):
  - It reproduced the numbers with its own code through the public pipeline: the 31-seed block,
    the centring, seed 42's full row, both invariants and the canonical block.
  - It found no bug, no gap, and no pin of −0.88 or of the old level or maximum. Tracked files in
    `src/`, `tests/` (without characterization), `checks/` and `configs/` were searched.
  - Its observations are recorded above: the invariants at rate 3, the recovery margin, the
    cohort drift, and the 31-seed building totals.
