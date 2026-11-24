# Simulator realism: the plan

> **Status (2026-11-24):** N0, N1 committed; **N2 done** (reviewed by Fable), awaiting the user's
> commit. Next: **N3** (Sonnet 5.5). The record is §10.

The simulator's neighborhood model is stylized: household size ignores SES, daycares ignore
demand, school status is independent of everything, and the levels (household size 2.6, median
age 37) are European rather than Israeli. The project summary (`docs/PROJECT_SUMMARY.md`, branch
`docs/project-summary`, paused under its decision P12) will describe a realistic simulator
instead. This plan revises the simulator **step by step, one PR per step, neighborhoods first**.

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
- Median age 30.7 ± 5.6 (18.3–45.5).
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
| 3 | Children per apartment | The NB tail: a maximum of 37 children in one apartment (needs a cap or a larger $\phi$). Zero share 0.52 against about 0.60 real. Too flat at small households (0.45 children at $H = 2.2$; real about 0.1–0.3). |
| 4 | The cohort split | Reference shares 0.40/0.35/0.25 against a real 27/38/35% for ages 3–6/7–12/13–18. The household-size coefficient (+0.15) pushes large-household areas toward high school, the opposite of Israel. |
| 5 | The reruns | After the summary's documentation (summary P12): notebooks 03 and 04 rerun on the new data, a fresh MLflow store, then the summary's S5a onward. |

## 8. Step 1 sub-tasks

| # | Sub-task | Model | Review | Done |
|---|---|---|---|---|
| N0 | The branch `feat/realistic-neighborhoods` from `main`; this document committed | user | no | |
| N1 | The config schema and validation (§4) and the new neighborhood draws (§2), with their unit and statistical tests (§5) | Opus 5.5 | yes | 2026-11-24 |
| N2 | The downstream values (§3), the cross-section check, the validation-helper ranges, the consistency invariant; a full run confirming §3's targets with the final daycare formula | Opus 5.5 | yes | 2026-11-24 |
| N3 | Regenerate the characterization fixtures; the full test suite; update the docs that name the old fields (grep for every removed and renamed field) | Sonnet 5.5 | no | |
| N4 | An independent review of the whole step against this plan, then the PR description | Fable 5.2 | is the review | |

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
- [ ] The characterization fixtures are regenerated, and the diff of each is explained in §10.
- [ ] The full test suite passes (output shown).
- [ ] Every doc naming a removed or renamed field is updated. Historical plans and records under
      `docs/archive/` stay as they are.

### N4. Review and the PR (Fable 5.2)
Done when:
- [ ] The step is reviewed against §1–§5 as a new reader; each finding is reproduced, then fixed.
- [ ] The PR description is printed in the chat.

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
