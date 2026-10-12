# Gate Validation Findings

## State of play

| | |
|---|---|
| **Validated and remediated** | Gates 1, 2A, 2B, 2C, 3, 4, 5, 6, and **Gate 8, complete** (independent validation 2026-09-14: accepted with conditions; must-fix conditions, F7 and F9 remediated; F6, F8 and F10 accepted as deferred; the fresh-session remediation review found R1 (High) and test gaps R2–R5, all remediated, independently reviewed, and accepted by the user on 2026-09-14) |
| **Implemented with per-phase independent reviews** | **Gate 7** (MLflow tracking); see its section below |
| **Pending** | User approval of the Gate 9 independent post-completion validation verdict |
| **Independently accepted** | **Gate 9**: Phase 4 removal, guides, authoritative suite, dependency/build/wheel, notebook, stale-reference, Section 14, Section 15, and protected-evidence checks passed with review verdict **ACCEPT**, approved by the user on 2026-09-15. The separate post-completion Opus 5 validation (2026-09-15) returned **ACCEPT** after remediating three Medium and twelve Low findings; see [GATE_9_INDEPENDENT_VALIDATION_REPORT.md](GATE_9_INDEPENDENT_VALIDATION_REPORT.md) |
| **Suite baseline** | **696 passing, 13 warnings** after the Gate 9 validation added one regression case (695 after the intentional removal of ten obsolete tests); warnings are twelve MLflow integer-schema hints and one documented Bayesian recovery warning (worst R-hat 1.086056330170683, minimum ESS 25.660557049567544) |
| **Progression** | 346 → 358 (Gate 1) → 363 (2A) → 376 (2B/2C) → 382 (Gate 3) → 392 (Gate 4) → 407 (Gate 5) → 420 (Gate 6) → 524 (Gate 7 Phases 0-1) → 567 (Gate 7) → 624 (Gate 8 Phases 0-3) → 654 (Gate 8) → 676 (Gate 8 remediation) → 689 (Gate 8 remediation, second pass) → 705 (Gate 8 remediation review) → 695 (Gate 9 removal) → 696 (Gate 9 independent validation) |

**Open items carried forward:**

1. ~~**G4-1 — Model B's calibration retention rule.**~~ **CLOSED in Gate 5
   remediation.** Re-derived independently, confirmed, and replaced with a
   likelihood-ratio test at 1 df. `calibration_improvement_tolerance` is gone
   and `calibration_significance_level` (0.05) replaces it. See the Gate 5
   section, finding G5-2 and remediation R2.
2. ~~**§5.3 candidate factories do not exist.**~~ **CLOSED in Gate 6
   remediation.** `experiment/candidates.py` now enumerates section 5.3's
   predeclared specs in declared order, per component. See the Gate 6 section,
   remediation R10.
3. ~~**Stale plan evidence paragraphs** for Gates 2A/2B/2C.~~ **CLOSED in Gate 6
   remediation.** Test counts re-measured, 2A's candidate-factory claim now
   true, the Gate 6 suite count corrected from 342, and
   `GATE_5_6_VALIDATION_INSTRUCTION.md` brought up to date. See Gate 6
   remediation R10.

**Contract changes later gates inherit** (details in the gate sections):
`run_optuna_study` selects completed trials only and raises when none completed;
`selection_improvement_tolerance` removed and `bootstrap_max_failed_fraction`
added to `IndependentTotalProbabilityConfig`; Model B's bootstrap skips and
counts replicates that omit a cohort; both models record
`diagnostics[...]["search_space"]` under the same key; Model B's calibration
keys renamed to `*_weighted_share_error`.

**One correction that matters outside this document:** `predictive.py` is **not**
a legacy module and must not be removed — all three model families import
`central_prediction_intervals` from it. The Gate 3–6 brief previously listed it
for deletion; that entry is corrected.

---

Independent validation records, one section per gate. Each section states a
verdict against that gate's own acceptance criteria, findings ranked by
severity, statistical judgement calls kept separate from defects, code-quality
findings, documentation-accuracy notes, and cross-references for other gates.

Probe scripts are named per finding, but they live in the reviewing session's
scratchpad and **do not survive that session**. Treat a probe name as a record of
what was run, not as a re-runnable artifact. Any probe that encodes a contract is
promoted to a test in the repository during that gate's remediation, and those
tests are the durable evidence — each is named in the gate's "Remediation
applied" block.

---

## Gate 1 — Modeling Configuration And Locked Data Splits

**Reviewed:** 2026-09-12 · `modeling_config.py`, `dataset_builder.py`,
`data_splitting.py` · against plan §902, §4, §4.1, §4.2.

### Verdict: **ACCEPT WITH CONDITIONS** — conditions since discharged

> **Remediation applied (same session).** F1–F4 are fixed, F5 resolved by
> amending §4.1, F6 (found later, while tracing the seed wiring) fixed, and all
> four code-quality findings addressed. Suite: **358 passed** (was 346), ruff
> clean. Per-finding status is recorded inline below
> and summarized under "Remediation applied" at the end of this section. The
> findings themselves are left as written so a later session can audit the
> reasoning, not just the outcome.

Every clause of Gate 1's stated acceptance sentence is met, verified
independently rather than by trusting the suite:

| Clause | Result |
|---|---|
| schema/accounting/leakage tests pass | 53 Gate 1 tests pass; 346 pass overall, matching the documented baseline exactly, with the one documented Bayesian warning. `ruff` clean. |
| split is deterministic | Confirmed, omitted-RNG and fresh-equal-generator paths. |
| row-order invariant | Confirmed for the outer split **and** for `make_validation_folds` (the latter was untested). |
| singleton behavior is correct | Correct for the outer split, exactly as §4.2 specifies. See F1 for the fold generator. |
| manifests replay exact assignments | Confirmed, including a real `json.dumps`→`json.loads` round trip. |
| test targets not evaluated/summarized during tuning | Confirmed structurally; `test_df` never leaves `data_splitting.py`; no MLflow import anywhere. |

All eight §4.1 rejections are enforced. The §4.2 RNG convention is implemented
exactly as written, including the caller-provided-generator advancement
sentence that no test covers. The conditions are F1 and F2 below.

### Findings by severity

**F1 — `make_validation_folds` gives no coverage guarantee; 29.2% of training
buildings are never validated, and some never can be.**
`data_splitting.py:221-244`, with the cause at `:124-125`. **Probe-confirmed**
(`p1_p2_folds_and_fraction.py`, `p10_bias_and_order.py`). **Non-blocking for
Gate 1's stated criteria** — §4.2 specifies fold determinism and train-only
membership, both of which hold, and says nothing about coverage — but it must
be resolved before any fold-based uncertainty or model comparison is trusted.

At project defaults on `configs/stage1.toml` (251 buildings, 60 neighborhoods,
192 in outer train):

- **56/192 (29.2%) appear in zero validation sets.**
- Per-building validation exposure ranges 0–5: `{0: 56, 1: 58, 2: 47, 3: 28,
  4: 1, 5: 2}`. A true K-fold would give every building exactly 1.
- **10 buildings are structurally unvalidatable at any fold count** — they sit
  in outer-training singleton neighborhoods, which `_partition_known_neighborhoods`
  skips by design (`:124-125`). Raising `n_folds` from 5 to 100 reduces the
  never-validated count from 56 only to 10, and no further.
- **7 of those 10 singletons were created by the outer split itself.** The
  population has 3 singleton neighborhoods; after the outer split there are 10,
  because every size-2 neighborhood contributes one building to test and leaves
  a singleton behind. The outer split manufactures the exclusion.
- Realized validation fraction is 0.26 per fold against a requested 0.20.

Scope of impact is wider than cross-model CV: `models/direct_cohort.py:88` and
`models/independent_total_probability.py:136` call `make_validation_folds`
**inside `fit`** for hyperparameter tuning and specification selection. On those
paths the same 29.2% of rows never influence the selected hyperparameters, and
no coverage diagnostic is recorded at all.

Credit where due: Gate 6 already mitigates observability on its own path —
`experiment/partitions.py:89-108` computes `_fold_coverage` and its docstring
describes this exact problem accurately. The gap is that Gate 1's own API
provides no guarantee, warning, or diagnostic, and the model-internal tuning
callers have no equivalent.

*Recommended remediation:* give the fold generator an explicit, documented
coverage contract. Either (a) a true grouped K-fold that rotates every
non-singleton neighborhood's buildings so exposure is exactly 1, or (b) keep
repeated splits but return a coverage record alongside the folds and raise when
a caller-set minimum exposure is unmet. Separately, decide deliberately what
outer-train singletons are *for*: they can train but never validate, so either
document that as intended or route size-2 neighborhoods differently in the
outer split.

**F2 — `SplitManifest.holdout_fraction` records the requested fraction, not the
realized one, and the gap reaches 2.27×.** `data_splitting.py:67, 153` with the
cause at `:126-129`. **Probe-confirmed** (`p1_p2_folds_and_fraction.py`,
`p10_bias_and_order.py`). **Non-blocking** — the ID lists are authoritative and
the truth is recoverable from them — but the field as named and typed is
misleading in a manifest whose stated purpose is lockbox evidence.

`max(1, min(round(n * holdout_fraction), n - 1))` forces at least one holdout
building per non-singleton neighborhood. On a population of small
neighborhoods this systematically inflates the realized fraction:

| requested | recorded in manifest | realized |
|---|---|---|
| 0.1 | 0.1 | **0.227 (2.27×)** |
| 0.2 | 0.2 | 0.235 (1.18×) |
| 0.3 | 0.3 | 0.323 (1.08×) |

A reader of the manifest is told 0.1 when 22.7% of the data was held out.
`len(holdout_building_ids) / n_rows` gives the true value, so this is a
reporting defect, not a correctness one.

Two related notes on the same expression: `round()` is banker's rounding, so
`n=10, f=0.25` holds out 2 rather than 3 — defensible but unpinned by any test;
and the `max(1, ...)` floor means `test_fraction` is a lower bound in
small neighborhoods, not a target, which is undocumented.

*Recommended remediation:* record both — keep `holdout_fraction` as
`requested_holdout_fraction` and add `realized_holdout_fraction` computed from
the partition. Rename per the brief's "a name that lies" standard.

**F3 — the manifest's self-describing fields are never verified on replay.**
`data_splitting.py:191-218`. **Probe-confirmed** (`p3_p4_p6_manifest_rng.py`).
**Non-blocking**, low severity.

Replay verifies the two hashes and that the IDs form an exact partition — those
defences are strong (changed values, changed column order, changed dtype, an
added column, an ID in both partitions, and a dropped building are all
rejected). But `strategy`, `holdout_fraction`, `n_rows`, and `n_neighborhoods`
are carried and never checked. A manifest claiming
`strategy="totally-different-strategy-v9"`, `holdout_fraction=0.99`, and
`n_rows=999999` replays without complaint against a 251-row table. A manifest
with train and holdout **swapped** also replays silently, since it is still a
valid partition — the lockbox would then be 192 buildings instead of 59.

*Recommended remediation:* validate `n_rows`, `n_neighborhoods`, and
`strategy` against the table and active config in `replay_split_manifest`.
These are cheap checks over data already in hand.

**F4 — `EvaluationConfig.interval_method` is a `Literal` with no runtime
check.** `modeling_config.py:106`. **Probe-confirmed** (`p5_p7_schema_and_literals.py`).
**Non-blocking**, low severity — but it is precisely the defect class the brief
records escaping once already in `metrics.py`.

A full audit of all 12 `Literal`-annotated dataclass fields in
`modeling_config.py` found exactly one unvalidated: `interval_method`. Every
other field has a hand-validated `_VALID_*` counterpart actually checked in
`__post_init__`, including `FeatureSpec.interactions`, which is validated
element-wise at `:562-564`. `interval_method` is only ever recorded into
evidence metadata (`evaluation.py:168`) and never selects a code path, so an
unrecognized value corrupts the evidence record rather than silently changing
behaviour. That is why this is low and not blocking.

*Recommended remediation:* add `_VALID_INTERVAL_METHODS` and check it, matching
the module's own convention for the other eleven.

**F5 — `ModelingSchema` has no unknown-category policy, which §4.1 requires.**
`modeling_config.py:789-797`. **Judgement call / spec gap.** **Non-blocking.**

§4.1 requires rejecting "unknown `school_status` values **unless an explicit
unknown policy is selected**." `validate_table` always rejects; there is no
policy field on `ModelingSchema` and no policy reference in `validate_table`.
`UnknownCategoryPolicy` exists only on `FeatureSpec`, which is Gate 2A's
transform-time concern, not table validation.

The implemented behaviour (always strict) is the *safe* half of the spec, so
this is unimplemented optionality rather than a hole. Either implement the
policy or amend §4.1 to drop the clause — the current state has the plan
promising a knob that does not exist.

### Statistical judgement calls (not defects)

- **The never-validated subset is not measurably unrepresentative on this
  population.** Worth stating plainly because it is the obvious next inference
  from F1 and the probe does not support it. Mean covariates for validated vs
  never-validated buildings: neighborhood size 4.02 vs 4.25, `n_children_total`
  23.25 vs 23.30, `n_apartments` 39.43 vs 39.25, `ses` 0.04 vs 0.13. Two
  opposing mechanisms roughly cancel — size-1 neighborhoods are excluded 100%
  of the time, while *larger* neighborhoods have higher exclusion rates (37.5%
  at size 6) because more buildings compete for a fixed 20% share. So F1's real
  cost is **unequal and uncontrolled exposure**, not demonstrated covariate
  bias. The distinction matters for how it gets fixed.
- **Fold means remain correlated, unequally-weighted samples.** With exposure
  ranging 0–5, a plain standard deviation across folds is not a standard error.
  This restates the brief's §6 concern; F1 quantifies it. Any Gate 6 uncertainty
  derived from across-fold spread inherits this.
- **Room shares are not range-validated.** `validate_table` accepts a
  `3_rooms_share` of 5.0 or −0.3. §4.1 does not require it and the builder
  derives shares from counts it does not itself validate as ≤ `n_apartments`.
  Defensible as specified; noted because a hand-built table bypasses the only
  guard.

### Code-quality findings

- **Dependency inversion between the config and algorithm modules.**
  `modeling_config.py:11` imports `SplitConfig` and `DEFAULT_SEED` *from*
  `data_splitting.py`. Plan §4 names `SplitConfig` among the dataclasses
  `modeling_config.py` defines, and the brief's §4 architecture table assigns
  "split/evaluation/prediction configs" to it. The configuration module
  depending on the algorithm module is backwards and makes
  `modeling_config.py` non-self-contained. *Recommend* moving `SplitConfig` and
  `DEFAULT_SEED` into `modeling_config.py` and having `data_splitting.py`
  import them.
- **`SplitConfig` conflates two unrelated jobs.** It carries outer-split
  settings (`test_fraction`, `strategy_version`) and fold settings
  (`validation_fraction`, `n_folds`) in one frozen dataclass.
  `DirectCohortConfig.tuning_split` (`:194`) and
  `IndependentTotalProbabilityConfig.tuning_split` (`:240`) both reuse it in
  contexts where `test_fraction` and `strategy_version` are meaningless but
  still settable — an illegal state made representable, which the brief's §7
  explicitly asks to avoid. *Recommend* splitting into `OuterSplitConfig` and
  `FoldConfig`.
- **A shared helper that obscures a semantic difference.**
  `_partition_known_neighborhoods` serves both the outer split and the fold
  generator, differing only in `holdout_fraction`. Good reuse on its face, but
  it silently gives the fold generator the outer split's singleton-skip
  semantics, which are correct for one caller and are the direct cause of F1
  for the other. This is the brief's "shared helpers so general they obscure
  intent" pattern. *Recommend* making the singleton policy an explicit
  parameter so each caller states its intent.
- **`modeling_config.py` at 846 lines holds configuration for every gate** —
  Optuna (`:127`), LightGBM search spaces (`:145`), NB2 optimizer settings
  (`:234`), Bayesian priors (`:281`), NUTS profiles (`:317`), prior-predictive
  bounds (`:387`), numerical clipping (`:449`). A reader looking for
  `ModelingSchema`, the module's Gate 1 subject, finds it at line 659 after
  ~500 lines of Bayesian and tree-tuning policy. The size looks inherent to
  "all typed config in one place" but is not: the same pressure already caused
  `experiment.py` to be decomposed into `experiment/`. *Recommend* the same
  treatment — a `config/` package split by concern (`schema`, `features`,
  `splits`, `evaluation`, `models/*`), which would also resolve the inversion
  above.
- **Efficiency: no algorithmic problems found.** `_partition_known_neighborhoods`
  makes two full-frame copies per call and runs once per fold, so `n_folds=5`
  costs 10 copies of a 192-row frame — irrelevant at any plausible size, and
  not worth changing. `_table_hash` sorts and hashes the whole table on each
  split and replay, but is never called inside a loop. Stated explicitly so the
  absence of efficiency findings is not read as an absence of review.
- **Naming:** `SplitManifest.holdout_fraction` (F2) is the one name that lies.
  Others read accurately.

### Documentation accuracy

| Claim | Status |
|---|---|
| Plan §902 acceptance sentence, all five clauses | **Supported** (see verdict table). |
| Plan §902 evidence: "JSON outer-manifest replay" | **Overstated.** `test_modeling_data.py:316-329` calls `json.dumps` but then replays from the **in-memory dataclass**, never from deserialized output. I verified the round trip does work — `building_id` is a `str`, so the list-vs-tuple change after `json.loads` is harmless to `isin`. The claim is true; the test does not establish it. Note there is no `SplitManifest.from_dict`, and naive `SplitManifest(**json.loads(...))` produces a manifest that compares **unequal** to the original because lists ≠ tuples. |
| Plan §902 evidence: remaining items (typed defaults, schema overrides, table invariants, RNG behaviour, row-order invariance, changed-source rejection, neighborhood guarantees, deterministic folds, train-only fold IDs) | **Supported.** |
| Plan §4.1: the eight rejections | **Supported**, all eight enforced and probe-confirmed. |
| Plan §4.1: "unless an explicit unknown policy is selected" | **Unsupported** — no such policy exists (F5). |
| Plan §4.2: manifest contents, all seven bullets | **Supported** as to presence; see F2 on what "test fraction" means. |
| Plan §4.2: RNG convention, including caller-generator advancement and no competing `seed` argument | **Supported** and probe-confirmed — omitting `rng` equals `default_rng(DEFAULT_SEED)`, one generator across two calls yields different splits, and the 5 default folds are mutually distinct. |
| Brief §5 table: `tests/unit/test_rng_convention.py` is a Gate 1 primary test | **Unsupported.** That file tests `student_simulator` components exclusively (neighborhood, building, apartment, total-children, cohort stages). It contains no `age_group_prediction` import. Gate 1's RNG evidence lives in `test_modeling_data.py`. |
| Brief §7: "`__all__` ordered as sorted constants followed by sorted names" is a codebase convention | **Overstated as stated.** 12 of 20 top-level modules lack `__all__` entirely, including `data_splitting.py`, `modeling_config.py`, `dataset_builder.py`, `metrics.py`, `evaluation.py`, `results.py`, and `feature_engineering.py`. It is consistently applied in `models/` and `experiment/`, not package-wide. Gate 1's modules follow the top-level majority, so this is **not** a Gate 1 deviation — the brief's characterization is what needs correcting. |
| Brief §4: `modeling_config.py` holds "split/evaluation/prediction configs" | **Overstated.** It holds evaluation and prediction configs; `SplitConfig` lives in `data_splitting.py` and is only re-exported. |
| Brief §3: 346 tests, ruff clean, one documented Bayesian warning | **Supported**, reproduced exactly (346 passed, 92.38s, worst R-hat 1.086, min ESS 25.66). |
| `configs/modeling.toml:9-14` comment on `[randomness] default_seed` | **Supported.** The comment accurately describes limited wiring: `RandomnessConfig` is loaded by `experiment_config.py:170` and consumed at the experiment boundary, while Gate 1's `_resolve_rng` (`:52-57`) independently uses the module constant. Both are 42, so no divergence is observable today. The previously-recorded documentation defect in this area appears genuinely fixed. |

### Cross-references for other gates

- **Gates 3 and 4 (models):** `models/direct_cohort.py:88` and
  `models/independent_total_probability.py:136` call `make_validation_folds`
  inside `fit` for tuning and selection, inheriting F1 with no coverage
  diagnostic. Whether their hyperparameter choices are stable under the 29%
  exclusion is a Gate 3/4 question.
- **Gates 2B/3/4/5 (RNG convention):** models expose a `default_rng_seed`
  constructor argument alongside `rng` (`models/base.py:50`,
  `direct_cohort.py:48`, `independent_total_probability.py:85`,
  `bayesian_conditional.py:60`). §4.2's "do not also expose a competing `seed`
  argument on the same API" is written about the split/fold API, and these are
  constructor defaults rather than per-operation overrides, so Gate 1's wording
  does **not** forbid them. Flagging so the owning gate can rule on it rather
  than assuming Gate 1 blessed it.
- **Gate 6:** `experiment/partitions.py:89-108` `_fold_coverage` is the correct
  mitigation for F1's observability and its docstring is accurate. If F1 is
  fixed in `data_splitting.py`, check whether this diagnostic should stay
  (probably yes, as a regression guard).
- **Gate 8 (lockbox):** F3's unverified manifest fields matter most here, since
  Gate 8 is where a persisted manifest is replayed for real. The swapped-
  partition case is the one to guard.
- **Gate 9 (removal):** `gate3.py:21` still imports `SplitConfig` and
  `make_validation_folds`, so Gate 1 symbols remain reachable from code slated
  for removal. Noted only for reachability; not reviewed.

### Remediation applied

| Finding | Fix | Evidence |
|---|---|---|
| F1 | `make_validation_folds` now deals each neighborhood's buildings round-robin across folds instead of drawing repeated independent holdouts, and returns a `FoldPlan` naming the buildings no fold validates. A cursor carries the starting fold between neighborhoods so remainders do not pile onto fold 0. | Validation exposure went from `{0:56, 1:58, 2:47, 3:28, 4:1, 5:2}` to `{0:10, 1:182}` — exactly one for every non-singleton building, with the 10 outer-train singletons named in `unvalidated_building_ids`. Fold sizes `[58,50,33,24,17]` → `[37,37,36,36,36]`. Every neighborhood is present in every fit fold. |
| F2 | `SplitManifest.holdout_fraction` split into `requested_holdout_fraction` and `realized_holdout_fraction`. | `test_manifest_records_requested_and_realized_holdout_fractions`. |
| F3 | `replay_split_manifest` now verifies `strategy`, `n_rows`, `n_neighborhoods`, and the realized fraction against the table. | `test_split_manifest_rejects_mismatched_provenance`, `test_split_manifest_rejects_swapped_partitions` — the swapped-partition manifest is now caught. |
| F4 | Added the `IntervalMethod` alias and `_VALID_INTERVAL_METHODS`, checked in `EvaluationConfig.__post_init__`. All 12 literal-annotated fields are now runtime-validated. | `ruff` clean; loader tests exercise the section. |
| F5 | Resolved by amending plan §4.1 to drop the unknown-policy clause, per the "no knob with one value" argument in brief §7. Always-strict behavior is unchanged. | §4.1 edited. |

**F6 (found while answering a follow-up, fixed)** — `SplitManifest` recorded no
provenance for its randomness, so a lockbox could not say whether it came from a
deliberate generator or the implicit `DEFAULT_SEED` fallback. The experiment path
already records `master_seed_source` (`experiment/runner.py:54-68`, which raises
rather than falling back); the split path had no equivalent. Added
`SplitManifest.seed_source` (`"caller_generator"` / `"project_default"`), a
validated `SeedSource` literal defined in `data_splitting.py` because it is
evidence about a split rather than a configuration choice. Deliberately *not*
done: recording a seed integer, which would be a name that lies for an advanced
generator; verifying `seed_source` on replay, since it is history rather than a
property of the table; and asserting that the TOML `default_seed` equals
`DEFAULT_SEED`, which would forbid a legitimate seed-robustness run.

Structural changes made alongside:

- `DEFAULT_SEED`, `OuterSplitConfig`, and `FoldConfig` now live in
  `modeling_config.py`; `data_splitting.py` imports from it. The dependency
  inversion is gone and §4's placement of these types is now accurate.
- `SplitConfig` split into `OuterSplitConfig` (test fraction, strategy) and
  `FoldConfig` (fold count). `tuning_split` → `tuning_folds`, so a meaningless
  `test_fraction` is no longer settable on a tuning config.
- `validation_fraction` **removed**: the rotation makes the validation share
  `1 / n_folds` by construction, which is what eliminated the 0.20-requested /
  0.26-realized discrepancy rather than documenting it.
- `n_folds = 1` **now rejected** — a single fold leaves each neighborhood
  unrepresented in its own fit partition. Four `test_experiment.py` cases used
  it and now use 2.
- `SplitManifest.from_dict` added, so a JSON round trip compares equal to the
  manifest it came from.
- `__all__` added to `data_splitting.py`.
- `configs/modeling.toml` `[split]` → `[outer_split]` + `[folds]`.
- The shared-helper concern dissolved rather than being parameterized: folds no
  longer call the outer-split partitioner, so `_partition_outer_holdout` and
  `_assign_fold_indices` are separate functions named for what they do.

### Note for future split strategies

The split axis is neighborhood today and is expected to become time-based or
multi-axis later. Two things already support that: the grouping column is
configurable on both config types, and `replay_split_manifest` now verifies
`strategy` against the active config (F3), so a future `time-blocked-v1` config
cannot silently replay a `known-neighborhood-v1` manifest. What is *not* yet
generalized: the `neighborhood_id_column` naming, the singleton rule (which is
group-shaped, not time-shaped — a time split needs an ordering rule, not a
group-size rule), and `_assign_fold_indices`' rotation, which assumes
exchangeable units within a group and is wrong for time-ordered folds.
Adding a strategy should mean adding a partition function plus a config type
and a new `strategy_version`, not editing the existing ones.

### Verification of this review

- Probe scripts: `p1_p2_folds_and_fraction.py`, `p3_p4_p6_manifest_rng.py`,
  `p5_p7_schema_and_literals.py`, `p10_bias_and_order.py`.
- One planned finding was **withdrawn on evidence**: the room-count/room-share
  redundancy branch at `modeling_config.py:593-594` was suspected dead code. It
  is reachable — it is checked before the "not modeled" test, so it fires first
  and with the better message when a caller names a raw room count.
- One P7 result was a **false positive of the audit heuristic**:
  `FeatureSpec.interactions` is validated at `:562-564` by set difference
  rather than the `not in` idiom the scan looked for.
- `git status --porcelain` unchanged from the session baseline apart from this
  file.

---

## Gate 2A — Feature Engineering Foundation

**Reviewed:** 2026-09-12 · `feature_engineering.py` · against plan §930, §5,
§5.3, §5.4. Probes: `g2a_probes.py`.

### Verdict: **ACCEPT WITH CONDITIONS** — conditions since discharged

> **Remediation applied (same session).** G2A-1 through G2A-5 and the
> code-quality items are fixed; see "Remediation applied" below. The candidate
> factory gap (documentation section) is **deferred to Gate 6 validation**,
> since whether the factory is still wanted depends on how Gate 6 consumes
> candidates. Suite: **363 passed**, ruff clean, Gate 2A tests 19 → 24.

Most of the acceptance sentence is satisfied and was re-derived independently,
not taken from the suite:

| Criterion | Result |
|---|---|
| transformation before fit fails | Holds; `transform`, `get_metadata`, `get_feature_names_out` all raise. |
| output names and order are stable | Holds across partitions, shuffled rows, and refitting on shuffled rows. |
| learned state uses fit-fold data only | Holds. Poisoning eval rows (`ses × 1000 + 500`) changed neither metadata nor the fit matrix; including eval rows in `fit` does move the means, so the probe has power. |
| identifiers/targets/private effects never enter matrices | Holds for all three named defaults. |
| redundant room representations rejected | Holds. |
| zero-total filtering only after probability preprocessing is fitted | Holds; the transformer does no row filtering. |
| specs and metadata JSON serializable | Holds for all defaults and for spline state. |
| interaction hierarchy and component restrictions enforced | **Enforced on the spec, violated in the output matrix — see G2A-2.** |
| category references and unknown-category behavior honored | **References yes; unknown behavior no — see G2A-1.** |

### Findings by severity

**G2A-1 — `unknown_category_policy="ignore"` silently encodes an unknown
category as the reference category.** `feature_engineering.py:62-71`.
**Probe-confirmed.** Blocking for any run that selects `"ignore"`; the default
is `"error"`, so the default path is safe.

The encoder is built with `drop="first"` and `handle_unknown` taken straight
from the policy. Under `"ignore"`, sklearn encodes an unrecognized level as all
zeros — which is exactly the encoding of the dropped reference level. Probed on
real data: rows with `school_status="demolished"` and rows with the true
reference `"none"` both produce `school_status_existing=0,
school_status_planned=0`. They are numerically indistinguishable to the model.

So `"ignore"` does not mean "ignore"; it means "treat as the reference
category", which is a modeling decision no one declared. sklearn emits a
`UserWarning` ("unknown categories ... will be encoded as all zeros"), but
nothing catches it, and `get_metadata()` records no unknown-category
information at all — the metadata keys are `category_references`,
`feature_names`, `feature_spec`, `fit_row_count`, `numeric_means`,
`numeric_scales`, `spline_knots`.

*Remediation:* either forbid `"ignore"` together with `drop="first"`, or add an
explicit indicator column for unknowns, or keep the behavior and rename the
policy to what it does (`"treat_as_reference"`). In every case, record the
unknown count in metadata rather than dropping a warning on the floor.

**G2A-2 — interaction hierarchy is validated on the spec but broken in the
matrix whenever a nonlinear form replaces the main effect.**
`feature_engineering.py:183-198` (drops) vs `:206-227` (interactions).
**Probe-confirmed.** Non-blocking but statistically material.

`FeatureSpec.validate_for_schema` enforces §5.3's "every interaction includes
its main effects" by requiring the column in `numeric_features`. But
`_transform_fitted` then *drops* that column when a nonlinear form is selected,
while `_add_interactions` builds the interaction from the untransformed
`numeric_df`. The check and the construction read the same requirement at two
different levels — the brief's "two code paths independently interpreting one
discriminator" pattern.

Two probed instances:

- `daycare_form="log1p"` with `daycare_x_median_age` →
  columns `[..., n_daycares_500m_log1p, n_daycares_500m_x_median_age]`. The
  interaction equals **raw** daycare × age (`allclose` True), not log1p × age
  (False). The model carries `log1p(x)` and `x·age` but no `x`.
- `ses_form="spline"` with `ses_x_household_size` → columns
  `[avg_household_size, ses_spline_0..4, ses_x_avg_household_size]`. The linear
  `ses` main effect is absent while its interaction is present.

The spline case is arguably defensible (the basis spans the linear term); the
`log1p` case is not — the functional form differs between a variable's main
effect and its interaction with no declaration anywhere.

*Remediation:* build interactions from the transformed representation, or state
explicitly in `FeatureSpec` which representation an interaction multiplies, so
the choice is declared rather than incidental.

**G2A-3 — category validation error names an empty list when the cause is a
missing value.** `feature_engineering.py:163-168`. **Probe-confirmed.** Low.

The condition is `values.isna().any() or (unknown and policy == "error")`, but
the message only ever formats `unknown`. A `None` in `school_status` raises
`Invalid school_status values: []`, which tells the reader nothing. Split the
two conditions and give the missing-value case its own message.

**G2A-4 — metadata reports scaling state that was never applied.**
`feature_engineering.py:49-51, 104-120`. Low.

`_numeric_means`/`_numeric_scales` are computed unconditionally, but
`_transform_fitted` applies them only when `scale_numeric` is true.
`DEFAULT_TREE_FEATURE_SPEC` has `scale_numeric=False`, yet its metadata still
reports a full `numeric_means`/`numeric_scales` map; the probe confirms the
output is in fact unscaled. A reader auditing metadata would conclude the tree
matrices were standardized. Either skip computing them or record
`scale_numeric` alongside so the fields are unambiguous.

**G2A-5 (efficiency) — `fit_transform` does all the work twice.**
`feature_engineering.py:73, 88-90`. **Probe-confirmed.** Low severity, but it
is in the hot path.

`fit` calls `_transform_fitted` to learn output names, then `fit_transform`
calls `transform`, which validates and transforms again. Instrumented counts
for one `fit_transform` call: `_transform_fitted` 2, `_validate_input` 2,
`get_log_exposure` 2. `models/direct_cohort.py` calls `fit_transform` once per
fold, so every fold's fit partition is built twice, per candidate, per trial.
Return the already-computed frame from `fit_transform` after the name check.

### Statistical judgement calls (not defects)

- **Spline input is unscaled while its neighbors are scaled.** With
  `scale_numeric=True` and `ses_form="spline"`, the spline is fit and applied on
  raw SES while other numerics are standardized. Internally consistent (fit and
  transform agree) and knot placement is scale-free, so this is defensible —
  noted because the mixed scaling of one matrix is easy to misread later.
- **The `ses_form="spline"` half of G2A-2** is a reasonable modeling choice if
  deliberate. It is listed under G2A-2 because nothing declares it as one.

### Code-quality findings

- **Project literals passed verbatim into a third-party API.**
  `handle_unknown=self.feature_spec.unknown_category_policy` (`:69`) works only
  because `UnknownCategoryPolicy`'s two values coincide with sklearn's accepted
  strings. The coupling is invisible and unasserted; a rename on either side
  breaks it silently. Map explicitly.
- ~~**Private sklearn attribute in serialized metadata.**~~ **Withdrawn on
  evidence.** I flagged `self._spline.bsplines_[0].t` as reaching into sklearn
  internals. It does not: `bsplines_` is a documented public *fitted attribute*
  of `SplineTransformer` (trailing underscore is sklearn's public fitted-state
  convention), and `.t` is scipy's public `BSpline` knot array. The code is
  correct as written; no change made.
- **Comments that restate the code**, exactly the pattern brief §7 asks to flag:
  `:53` and `:185` both say "Apply spline transformation to the SES column",
  `:183` "Apply quadratic transformation...", `:188` "Drop the original SES
  column before adding spline features". The code says all of this. The one
  comment worth keeping is `:249`, which explains *why* the reference is placed
  first.
- **Validation by discarded side effect.** `_validate_input` ends with
  `self.get_log_exposure(df)` (`:169`) purely to trigger its checks, computing
  and throwing away a full log column. A `_validate_exposure` helper would say
  what it means and cost nothing.
- **Good, worth preserving:** `get_log_exposure` is a genuinely well-documented
  decision (`log` not `log1p`, with the reason); the transformer never mutates
  its input; and `_ordered_categories` makes the reference-first invariant
  explicit rather than relying on sort order.

### Documentation accuracy

| Claim | Status |
|---|---|
| Gate 2A acceptance sentence | **Supported except two clauses** — unknown-category behavior (G2A-1) and interaction hierarchy in the matrix (G2A-2). |
| "20 focused Gate 2A tests" | **Unsupported.** 19 collected and passing in `tests/unit/test_feature_engineering.py` (15 functions, one parametrized 5 ways). |
| "validated candidate factories" / "ordered candidate factories" | **Unsupported — they do not exist.** No factory, enumeration, or ordered spec sequence appears anywhere in the package; `grep` for candidate enumeration in `src/` returns nothing. The only candidate set is a hand-written `replace(...)` list inside `test_candidate_transforms_are_finite_and_have_stable_names`. §5.3 explicitly requires "Candidate factories enumerate an ordered set of specs", and §5.4 assigns that job to Gate 2A. The validation half of §5.3's "closed, validated vocabulary" is built; the enumeration half is not. |
| §930 "Implement immutable `FeatureSpec` ... in `feature_engineering.py`" | **Contradicted by the code and by §4**, which places `FeatureSpec` in `modeling_config.py`. The code follows §4. §930's wording should be corrected. |
| "Keep deterministic room-share construction in Gate 1" | **Supported.** |
| "keep candidate evaluation out of the transformer" | **Supported.** The transformer neither fits models nor ranks specs. |

### Cross-references for other gates

- **Gate 6 / §5.3:** because no candidate factory exists, nothing guarantees the
  six predeclared candidate families are the ones actually compared, nor that
  they are enumerated in a stable order. Gate 6 builds its `CandidateDefinition`
  list by hand. Whoever owns §5.3 should decide whether the factory is still
  wanted or whether §5.3 should be amended to match the hand-built approach.
- **Gates 3/4:** G2A-5's double transform is paid inside their tuning loops.
- **Gate 4:** G2A-1 matters most for the composition model, whose categorical
  handling is the one most likely to meet an unseen level in deployment.

### Note on Gate 1 interaction

Gate 1's remediation moved `FeatureSpec` nowhere but did change
`modeling_config.py` around it. Gate 2A's suite passes unchanged (19/19), and
`FeatureSpec`/`validate_for_schema` were not modified, so no Gate 1 regression
reaches Gate 2A.


### Remediation applied

| Finding | Fix | Verified |
|---|---|---|
| G2A-1 | Policy value renamed `"ignore"` → `"treat_as_reference"`, so the name states the modeling consequence. Added an explicit `_SKLEARN_HANDLE_UNKNOWN` map so our vocabulary no longer rides on sklearn's happening to match, and documented the behavior on `FeatureSpec`. | `"ignore"` is now refused; a test asserts an unknown level encodes *identically* to the reference, so the behavior is pinned rather than merely renamed. |
| G2A-2 | `_transform_fitted` builds an `operands` map holding each variable in the representation the matrix actually carries, and `_add_interactions` multiplies those. `FeatureSpec` now rejects `ses_form="spline"` with `ses_x_household_size`, whose correct form needs a tensor-product basis — a modeling extension, not a bug fix. | Interaction now equals `log1p(daycare) × age` (was raw × age); the spline pair raises at construction. |
| G2A-3 | Missing values and unknown levels are separate checks with separate messages. | `school_status must not be missing` instead of `Invalid school_status values: []`. |
| G2A-4 | Scaling state is learned only when `scale_numeric` is set; metadata reports `scale_numeric` plus `None` rather than unused numbers. | Tree spec now reports `scale_numeric=False, numeric_means=None`. |
| G2A-5 | `fit_transform` returns a copy of the frame `fit` already built. | `_transform_fitted` calls per `fit_transform`: 2 → 1. |
| Code quality | Removed four comments that restated the code; replaced the discarded-side-effect validation with a named `_validate_exposure`. | ruff clean. |

Deliberately **not** done: adding an unknown-category indicator column. It
would be constant-zero in every fit fold (unknowns cannot appear at fit time by
definition), so its coefficient is never estimated and the column has no
effect — it would give the appearance of handling unknowns without doing so.

**Deferred to Gate 6 validation:** the missing candidate factories (§5.3/§5.4).
The decision — build the factory or amend §5.3 to match the hand-built approach —
depends on how Gate 6 actually assembles and orders `CandidateDefinition`s,
which is out of scope until that gate is reviewed.

**Documentation note now stale:** the "20 focused Gate 2A tests" claim was
already wrong (19); the suite is now 24. The plan's Gate 2A evidence paragraph
still needs updating for the test count, the candidate-factory claim, and the
`FeatureSpec` location — all left for the deferred decision so the paragraph is
rewritten once rather than twice.

---

## Gate 2B — Shared Model And Result Contracts

**Reviewed:** 2026-09-12 · `models/base.py`, `results.py` · against plan §964,
§4.3. Probes: `g2b_probes.py`.

### Verdict: **REJECT** → **ACCEPT** after remediation

> **Remediation applied (same session).** G2B-1 through G2B-5 fixed and
> verified against the original probe numbers; see "Remediation applied" below.
> Suite: **376 passed** (was 363), ruff clean. Gate 2B tests 26 → 34.

One blocking defect (G2B-1), which is shared with Gate 2C and is the exact
failure class the brief records as having escaped once already. Everything else
in the acceptance sentence holds and was re-derived independently.

| Criterion | Result |
|---|---|
| synthetic interface models pass | Holds; a real `BaseAgeGroupModel` subclass drives the full lifecycle. |
| unfitted operations fail clearly | Holds. `predict`/`evaluate`/`feature_transformer` each raise `RuntimeError: Model must be fitted before …`. |
| prediction shapes and invariants hold | Holds. |
| feature specs and preprocessing summaries in JSON-serializable metadata | Holds; `get_metadata` asserts with `json.dumps`. **But see G2B-2 for a missing §4.3 field.** |
| default and caller-provided RNG reproducible | Holds. Omitted RNG equals `default_rng(DEFAULT_SEED)`, and one generator advances across successive fits. |

### Findings by severity

**G2B-1 — an unrecognized predictive family is silently reclassified as
`normal`.** `results.py:100-151` (`_copy_parametric_distribution_mapping`), with
no validation on the dataclass at `:89-95`. **Probe-confirmed. Blocking.**

`ParametricDistributionSpec.family` is annotated `PredictiveFamily =
Literal["poisson", "nb2", "normal"]` and has **no `__post_init__` and no
`_VALID_*` set**. The validator is an if/elif chain whose final branch is an
unguarded `else` meaning "normal". Probed:

| requested | stored |
|---|---|
| `"poisson"` / `"nb2"` / `"normal"` | unchanged |
| `"gamma"` | **`"normal"`** |
| `"NORMAL"` | **`"normal"`** |
| `"student_t"` | **`"normal"`** |
| `""` | **`"normal"`** |

`ParametricDistributionSpec(family="not_a_family")` also constructs without
complaint.

This is the brief's "a `Literal` annotation mistaken for validation" and "two
code paths independently interpreting one discriminator" patterns at once —
here **three** chains read the same discriminator with an implicit
`else = normal`: `results.py:120`, `metrics.py:450`, and `metrics.py:978`.

The guard that would catch it is unreachable.
`distributions.pointwise_log_probability` ends with
`raise ValueError(f"Unsupported predictive family: {family}")` — correct
defensive design — but `results.py` has already rewritten the value to
`"normal"` before any kernel sees it. The validation exists *downstream of the
coercion*, so it can never fire.

Statistical consequence, and why this is blocking rather than cosmetic: a count
model whose family is mistyped is scored with `norm.logpdf`, a log **density**,
and then ranked against log **masses** from Poisson/NB2 candidates. Brief §6
names exactly this as an invalid comparison. Worse, `metrics.py:466` and `:995`
return `spec.family` as the reported label, so the evidence would read
`"normal"` and look deliberate.

*Remediation:* validate `family` in `ParametricDistributionSpec.__post_init__`
against a `_VALID_PREDICTIVE_FAMILIES` set, and make the trailing `else`
branches in `results.py` and `metrics.py` explicit `elif family == "normal"`
with a raise, so no chain can absorb an unknown value.

**G2B-2 — §4.3's required training-data and schema hashes are absent from model
metadata.** `models/base.py:185-207`. **Probe-confirmed.** Non-blocking against
the Gate 2B acceptance sentence, which does not enumerate them, but it is an
unmet §4.3 requirement.

§4.3 requires `get_metadata()` to cover "model and implementation version,
likelihood and parameterization, feature specification, preprocessing summary,
hyperparameters, priors, calibration, seeds, dependency versions, **training
data/schema hashes**, fit duration, uncertainty method, and model-specific
diagnostics." Thirteen of fourteen are present; the hashes are the exception,
and no key containing "hash" appears anywhere in a model's metadata. The only
hashes in the package are Gate 1's manifest hashes over the *whole* modeling
table, which are a different object from the training partition a given model
was fitted on.

Consequence: a serialized fitted model cannot be tied back to the rows it
learned from, so a later audit cannot prove which partition produced it.

**G2B-3 — `derived_seeds` accumulates duplicate records across predict calls.**
`models/base.py:218-236`, never reset in `predict`. **Probe-confirmed.** Low.

Recorded seed counts after 1–4 `predict` calls: 2, 3, 4, 5, with purposes
`['spy/fit', 'spy/predict', 'spy/predict', 'spy/predict', 'spy/predict']`. With
`rng` omitted each `predict` resolves a *fresh* `default_rng(default_rng_seed)`,
so the appended records are identical, and metadata grows without bound while
saying nothing new. A reader would reasonably conclude four distinct seeds were
drawn. Record seeds per operation, or replace on re-run rather than append.

**G2B-4 — `_derive_child_seed` is dead code.** `models/base.py:209-216`.
**Probe-confirmed.** Trivial.

Documented as a "compatibility alias" for `_derive_backend_seed`. A repo-wide
grep finds exactly one hit — its own definition. No caller, no test, no
notebook. Delete it, or state which external compatibility it preserves.

### Statistical judgement calls (not defects)

- **`evaluate_duration_seconds` includes the nested `predict` time.** `evaluate`
  calls `self.predict` inside its own `_record_duration` block, so the two
  reported durations overlap rather than partition. Probe confirms
  `evaluate ≥ predict` always. Defensible (it is the wall time of `evaluate`),
  but a reader summing durations will double-count. Worth one sentence in the
  docstring.
- **`predict` silently rebuilds a `PredictionResult` whose `validation_config`
  differs from the model's** (`base.py:152-167`). Reconstruction re-runs
  `__post_init__`, so invariants are re-checked under the model's config and the
  behavior is safe. Noted because it quietly overrides a subclass's declared
  policy instead of rejecting the disagreement; if a subclass ever means it, the
  intent is lost.

### Code-quality findings

- **Three parallel dispatch chains on one discriminator** (G2B-1). Beyond the
  correctness bug, the duplication itself is the design flaw: `results.py` and
  `metrics.py` each re-derive what a family means. A single validated accessor
  returning the family plus its parameters would leave one place to change.
- **Good, worth preserving:** `results.py` makes arrays read-only and wraps
  mappings in `MappingProxyType`, so a returned `PredictionResult` genuinely
  cannot be mutated by a metric — this is what lets brief §4's "metrics never
  mutate predictions" invariant hold structurally rather than by convention.
  `get_metadata`'s `json.dumps` self-assertion is likewise the right pattern.
- The `_record_duration` contextmanager correctly commits only on success: the
  assignment sits *after* the `try/finally`, so an exception skips it. The
  docstring says so and the probe confirms it (failed fit →
  `fit_duration=None`, `seeds=[]`, `is_fitted=False`).

### Documentation accuracy

| Claim | Status |
|---|---|
| Gate 2B acceptance sentence | **Supported** except as qualified by G2B-2. |
| "every legacy import path preserved as an alias to the same class objects" | **Supported**, probe-confirmed: `PredictionResult`, `EvaluationResult`, and `ParametricDistributionSpec` are the identical object via `age_group_prediction`, `.models`, `.models.base`, and `.results`. |
| "`evaluate` … delegates to the shared `evaluate_predictions` … so evaluation never sees feature matrices" | **Supported.** `evaluate` never references `features`, and `evaluation.py` imports no model or Pyro code. |
| "commits durations … only when the operation succeeds; a failed fit clears all recorded durations" | **Supported**, probe-confirmed. |
| "24 focused contract tests" | **Unsupported**; 26 collected. |

### Cross-references for other gates

- **Gate 2C owns half of G2B-1.** `metrics.py:444-466`
  (`PredictiveNegativeLogLikelihood._log_probabilities`) and `:970-995`
  (`RandomizedPIT`) both end in an unguarded `else` that assumes normal, and
  both *trust* that `results.py` already normalized the value. Fixing only
  `results.py` would leave two latent chains; fixing only `metrics.py` would
  leave the coercion. Both must change together — see the Gate 2C section.
- **Gates 3/4/5:** G2B-2's missing hashes are a base-class gap, so all three
  models inherit it; whichever gate owns metadata completeness should decide
  whether the hash covers `train_df` rows, the schema, or both.


### Remediation applied (Gates 2B and 2C)

Fixed together because G2B-1 spans both gates.

| Finding | Fix | Verified |
|---|---|---|
| **G2B-1 / G2C-1** (blocking) | `_VALID_PREDICTIVE_FAMILIES` added beside the `PredictiveFamily` alias, and `ParametricDistributionSpec.__post_init__` now validates `family`. All three dispatch chains closed: `results.py` and both `metrics.py` sites end in an explicit `elif family == "normal"` followed by a `raise`, so no chain can absorb an unknown value even if construction is bypassed. | `"gamma"`, `"NORMAL"`, `"student_t"`, `""` all raise at construction (each previously scored NLL 35.449586 as Normal). A test that corrupts a spec *after* construction proves the metric chains refuse independently of the dataclass guard. |
| **G2B-2** | Gate 1's `_table_hash`/`_column_schema_hash` extracted verbatim into a new `hashing.py` with public names; `data_splitting.py` and `models/base.py` both import from it. `fit` records `training_data_hash` and `training_schema_hash`; `_clear_fitted_state` clears them. | Both present in metadata. Hash values unchanged — Gate 1's manifest replay tests pass untouched, which is the specific proof. A test asserts the hash is stable under row reordering but **differs** for changed rows and changed dtypes, so it cannot pass on a constant. |
| **G2B-3** | `_record_duration` renamed `_operation_scope` and now drops seed records for the same operation on entry. | Seed counts across 4 `predict` calls: `[2, 3, 4, 5]` → `[2, 2, 2, 2]`, with `fit`'s seeds preserved. |
| **G2B-4** | `_derive_child_seed` deleted. | Repo-wide grep returns nothing. |
| **G2B-5** | The 13-field manual `PredictionResult` rebuild replaced with `dataclasses.replace`, which re-runs `__post_init__` and cannot drop a future field. | Suite green. |
| **G2B judgement call** | `evaluate`'s docstring now states that `evaluate_duration_seconds` contains `predict_duration_seconds` rather than partitioning with it. | — |

Why `hashing.py` exists rather than importing from `data_splitting.py`: a model
has no business depending on split machinery. The helpers were used only inside
`data_splitting.py`, so extraction was free, and Gates 7–8 will want the same
functions.

Deliberately **not** done, all recorded above as judgement calls rather than
defects: decomposing `metrics.py` (a refactor of a module whose correctness was
just verified belongs in its own change); altering the bootstrap's
`default_seed`/`rng` pair (precedence is explicit and provenance is recorded);
and changing failed-replicate handling (bounded by `max_failed_fraction`, with
`successful_replicates` recorded).

---

## Gate 2C — Metric And Evaluation Interfaces

**Reviewed:** 2026-09-12 · `metrics.py`, `evaluation.py`, `resampling.py`,
`distributions.py` · against plan §997, §9. Probes: `g2c_probes.py`.

### Verdict: **ACCEPT** (after the shared G2B-1/G2C-1 fix above)

Gate 2C's only defect was its half of the shared family-dispatch bug — two
`else` branches that assumed Normal and trusted `results.py` to have normalized
the value. Everything else was re-derived independently and holds.

| Criterion | Result |
|---|---|
| metric golden tests cover zeros, invalid predictions, posterior integration, randomized PIT | Holds; 79 tests across the four modules. |
| bootstrap reproducible for omitted and caller-provided RNGs | Holds. Omitted reproducible; equal fresh generators reproducible; one shared generator advances across calls. |
| evaluators consume predictions, not feature matrices | Holds; `evaluation.py` imports no model or Pyro code. |

### Independently verified correct (not re-derived from the suite)

- **Randomized PIT is numerically right.** An independent from-scratch
  implementation — `lower + u·(upper − lower)` with `lower = P(X ≤ k−1)` —
  reproduced the metric's KS statistic **exactly** (0.0230) on identical data.
  Across 15 seeds at n=4000 the KS mean was 0.0146 with 2/15 exceedances of the
  5% critical value 0.0215, consistent with correct uniformity. It also
  discriminates: a 1.6× biased mean gives KS 0.6222.
- **The Gate 6 `source`-literal fix held.** Both abstract bases raise
  (`"… is abstract; construct a concrete variant"`), `source` is a frozen
  `init=False` class constant per variant so declared capability and scoring
  path cannot disagree, and `default_metric_set` rejects unknown `nll_source`
  and `pit_source` values by name.
- **Every other `Literal` in the 2C surface is runtime-validated:**
  `ReconciliationError.statistic` (`metrics.py:714`) and
  `on_missing_capability` (`evaluation.py:249`). The only unvalidated one in the
  package was `PredictiveFamily`, now fixed.
- **Metrics never mutate predictions.** After running the full default metric
  set, `total_mean`, `age_group_probabilities`, and `cohort_means` are
  unchanged, and the arrays are genuinely read-only (`assignment destination is
  read-only`), so the invariant holds structurally rather than by convention.
- **The bootstrap resamples whole neighborhood clusters**, records
  `bootstrap_unit="neighborhood"`, and records seed provenance
  (`default_seed if rng is None else None`).

### Statistical judgement calls (not defects)

- **`neighborhood_cluster_bootstrap` takes both `default_seed` and `rng`**
  (`evaluation.py`), a mild deviation from brief §4's one-RNG convention.
  Precedence is explicit, provenance is recorded, and probes confirm correct
  behavior. Noted, not changed.
- **Percentile intervals are computed from surviving replicates.** Failed
  replicates are counted and skipped, so the interval conditions on success.
  `max_failed_fraction` bounds it and `successful_replicates` is recorded, but
  a partially-failing run yields an interval over a non-random subset.

### Code-quality findings

- **`metrics.py` is 1154 lines**, the largest module in the package. Unlike
  `modeling_config.py` its size looks closer to inherent — it is one coherent
  registry of metric classes — but the point/predictive/composition/interval/PIT
  families are independent enough to split if it grows further. Not changed;
  refactoring a module whose correctness was just verified belongs in its own
  change.

### Documentation accuracy

| Claim | Status |
|---|---|
| Gate 2C acceptance sentence | **Supported.** |
| The reopened-gate narrative (discriminated unions, `default_metric_set` as the single parse boundary) | **Supported**, re-verified rather than assumed. |
| "50+ focused metric/evaluation tests" | **Supported**: 79 (metrics 36, distributions 29, evaluation 10, resampling 4). |
| "sklearn retained only as a tested oracle" | **Supported.** |

---

## Gate 3 — Model A: `DirectCohortModel`

**Reviewed:** 2026-09-12 · `models/direct_cohort.py`, `tuning.py` · against plan
§1085 (Gate 3), §6, §4.3, §11. Probes: `g3_p2_nb2_derivatives.py` (+`g3_p2b`),
`g3_p3_total_convolution.py`, `g3_p4_tuning_direction.py` (+`g3_p4b`, `g3_p4c`),
`g3_p5_recovery.py`, `g3_p6_9_10.py`, `g3_p7_family_dispatch.py` (+`g3_p7b`,
`g3_p7c`), `g3_p8_coverage.py`, `g3_p11_misc.py`.

**Baseline reproduced exactly:** 376 passed in 88.91s, ruff clean, one documented
Bayesian warning with identical numbers (worst R-hat 1.086056330170683, minimum
ESS 25.660557049567544). Gate 3's own files collect 22 tests (20 + 2), all passing.

### Verdict: **ACCEPT WITH CONDITIONS**

The numerical core of this gate is sound, and that was established independently
rather than taken from the suite. The conditions are G3-1 (a shared `tuning.py`
selection defect) and G3-2 (a dispatch fall-through that reopens the Gate 2B
class). Neither fires under the default configuration, which is why this is not
a reject.

| Acceptance clause | Result |
|---|---|
| shared contracts pass | **Holds.** Lifecycle re-derived against the real model, not the spy: unfitted `predict`/`evaluate` raise, a failed fit clears all state, metadata is JSON-serializable, `training_data_hash`/`training_schema_hash` present. |
| outputs finite / nonnegative | **Holds for means, probabilities and intervals**, all three families. Normal *draws* go negative (2.96% of total draws, min −6.295) — a statistical judgement call, recorded below, not a violation of this clause as written. |
| probabilities normalize, tested zero fallback | **Holds.** Uniform fallback fires on zero predicted *means* and is covered by an existing test. |
| preprocessing has no fold leakage | **Holds, probe-confirmed.** Instrumenting `FittedFeatureTransformer` shows each fold's transformer learns from a strict subset of the training rows (sizes 133/133/134 of 200) with **zero** fit/validation ID overlap per fold. The final estimator refits on the full training partition, which is correct. |
| objective derivatives tested | **Holds and independently re-derived.** I derived the NB2 NLL derivatives w.r.t. the log-mean raw score by hand as `(μ−y)/(1+αμ)` and `μ(1+αy)/(1+αμ)²`; `distributions.py:289-290` matches both exactly (`atol=0`). Over 125 (y, η, α) cases including y=0 and α=1e-4, worst relative gradient error **1.3e-6** and, at a step size free of cancellation, worst relative Hessian error **9.1e-6**, Hessian strictly positive throughout. |
| total-distribution semantics tested | **Holds and independently re-implemented.** An independent pmf-space `np.convolve` reproduces `nb2_total_log_probability` to **4.4e-15**, and the NB2 total mass sums to 1.0000000000 over 0..399. Poisson total = `poisson.logpmf(Σμ)` to 0.0; Normal total scale = √Σσ². NB2 correctly declares **no** total family, per plan §6 item 6. |
| bootstrap, tuning, point results reproducible | **Holds** across all three families, same-seed and fresh-process. |

### Findings by severity

**G3-1 — a pruned, partially evaluated Optuna trial can win the study and have
its hyperparameters selected for the final refit.** `tuning.py:73-76`.
**Probe-confirmed** (`g3_p4c.py`). **Blocking whenever `enable_pruning=True`;
latent at the default `False`.** **Shared module — also affects Gate 4.**

`best = min((t for t in study.trials if t.value is not None), ...)`. The filter
was evidently intended to exclude pruned trials. It does not: Optuna assigns a
**pruned trial the value of its last intermediate report**, so all 11 pruned
trials in `g3_p4b.py` carried a non-`None` value.

`direct_cohort.py:122` reports the *running mean over folds so far*. When later
folds are harder — which is the norm, since Gate 1's rotation equalizes fold
*size* but not fold *difficulty* — a trial pruned early is scored on its easy
folds only, and its recorded value is systematically lower than any completed
trial's full-fold mean. Decisive probe, 40 trials over 5 folds of increasing
difficulty:

```
study winner: trial 38 state=PRUNED value=0.2323 v=0.0808 folds_evaluated=1/5
best COMPLETE: trial 14 value=1.0002 v=0.0001   (v=0.0001 is the true optimum)
```

A configuration evaluated on **1 of 5 folds** beat the genuinely better
configuration evaluated on all 5, and `best_params` returned the loser's
parameters for the final refit. This is a partial-fold score compared against
full-fold scores — the brief's "silent selection bias" pattern.

*Recommended remediation:* select over `state == "COMPLETE"` trials only (this is
exactly what `study.best_trial` does), and raise a named error when no trial
completed. Keep pruned trials in `TuningResult.trials` as evidence.

**G3-2 — a corrupted family discriminator is laundered into NB2 on the predict
path, with a Normal standard deviation reused as an NB2 dispersion.**
`direct_cohort.py:353-358` (`_parametric_distributions`), with the same
fall-through shape at `:195` and `:236`. **Probe-confirmed** (`g3_p7c.py`).
**Non-blocking** — not reachable through the public API — but it is the Gate 2B
defect class left open, and Gate 2B's own remediation standard was explicitly
*"no chain can absorb an unknown value even if construction is bypassed."*

`DirectCohortConfig.__post_init__` **does** validate `family`, and all four of
`"gamma"`, `"NORMAL"`, `"student_t"`, `""` are rejected at construction. Credit
where due: the Gate 2B fix is present and working. But seven downstream sites
re-read the discriminator, and three end in an implicit branch meaning NB2.
Bypassing the frozen dataclass with `object.__setattr__` after a legitimate fit:

| fitted as | plain `predict` with a corrupted discriminator |
|---|---|
| `normal` | **SUCCEEDS.** Declares `nb2` with `dispersion = 1.7414550792212247` — the fitted Normal *standard deviation*, silently reinterpreted as a count dispersion. The `total` key is dropped. |
| `nb2` | Succeeds, declares `nb2` (benign). |
| `poisson` | `KeyError: 'n_kindergarten'` — `_ancillary_parameters` is empty. |

The Normal row is the damaging one: a continuous scale becomes a count
dispersion, the result is a well-formed discrete predictive distribution, and
the evidence label reads `"nb2"` as though deliberate. The `pointwise` path does
raise `ValueError: Unsupported predictive family: gamma`, but only when
`include_pointwise_log_probabilities=True`; the plain predict path has no guard.

*Recommended remediation:* close the three chains with an explicit
`elif family == "nb2": ... else: raise ValueError(...)`, matching what Gate 2B
did in `results.py` and `metrics.py`. Replace the bare `assert dispersion is not
None` at `:196` with a real error — it currently surfaces as a message-less
`AssertionError` and vanishes entirely under `python -O`.

**G3-3 — `_unvalidated_building_ids` is write-only: Gate 1's fold-coverage
diagnostic is discarded on this path.** `direct_cohort.py:92`.
**Probe-confirmed** (`g3_p11_misc.py`). Low severity, but it silently undoes a
Gate 1 remediation.

The attribute is assigned in `_fit_model` and **never read anywhere** — grep
finds one write and no reads. It is absent from `get_metadata()`
(`diagnostics` keys are `ancillary_parameters`, `family`, `normal_mean_policy`,
`search_space`, `tuning`), and it is not cleared in `_reset_model_state`, so it
survives a later failed fit as stale state. Gate 4's model, from the identical
`FoldPlan` API, records `unvalidated_building_count` in its diagnostics
(`independent_total_probability.py:160`). Gate 1's F1 remediation made this
information available precisely so a consumer could report it; Gate 3 takes it
and drops it.

*Recommended remediation:* report `unvalidated_building_count` in diagnostics,
matching Gate 4 exactly so the key means the same thing across model families,
and add the attribute to `_reset_model_state`.

**G3-4 (efficiency) — `predict` performs 600 LightGBM refits and 200 feature-
transformer fits at default configuration, and the refit uncertainty it buys is
~0.3% of interval width.** `direct_cohort.py:403-452`. **Probe-confirmed**
(`g3_p8_coverage.py`, `g3_p6_9_10.py`).

Every `predict` that requests draws or intervals runs `bootstrap_replicates`
(default **200**) × 3 cohorts LightGBM fits, and refits *and* re-applies a
`FittedFeatureTransformer` on each of the 200 replicates. `evaluate` calls
`predict`, so Gate 6 pays this per fold, per candidate.

What it buys, measured against the declared outcome-only parametric interval on
identical rows at 100 replicates:

| family | target | draw-interval width | declared (outcome-only) width |
|---|---|---|---|
| poisson | n_kindergarten | 3.57 | 3.56 |
| poisson | total | 6.66 | 6.69 |
| nb2 | n_elementary | 4.45 | 4.44 |
| normal | total | 7.28 | 7.23 |

Outcome noise dominates; the refit component is within noise of zero at this
signal-to-noise ratio. That does not make the bootstrap wrong — it is the
correct construction and the ratio would differ on smaller training sets — but
the default of 200 replicates should be a deliberate, documented choice rather
than an incidental one, and the per-replicate transformer refit is avoidable
work regardless.

*Recommended remediation:* hoist nothing that changes semantics. Do report the
refit-vs-outcome variance decomposition in diagnostics so the replicate count
can be justified, and consider lowering the default. The transformer refit per
replicate is required for correctness (it must learn from the resampled rows),
so it stays.

**G3-5 (MLflow / metadata volume) — seed records scale with the tuning grid and
the bootstrap.** `base.py:215-233` called from `direct_cohort.py:113`, `:430`,
`:438`. **Probe-confirmed** (`g3_p6_9_10.py`).

`_derive_backend_seed` records one entry per trial × fold × cohort during `fit`,
and two per replicate × cohort during `predict`. Measured at 10 trials / 3 folds
/ 2 replicates: **96 seed records after `fit`, 108 after `predict`, 28 KB of
metadata JSON**. At `DirectCohortConfig()` defaults (30 trials, 5 folds, 200
replicates) that is **~456 fit seeds and ~1,200 predict seeds**. Gate 2B's G2B-3
fix stops *duplication* across repeated `predict` calls but not growth within
one. `get_metadata()` deep-copies and `json.dumps`-es all of it.

This is not a correctness problem — every seed is genuinely distinct and
reproducibility depends on them — but it is the wrong shape for a tracked run.
See the MLflow section for the recommendation.

### Statistical judgement calls (not defects)

- **Normal predictive draws are negative 2.96% of the time** (min −6.295 on a
  total whose true mean is ~7). Counts cannot be negative, and nothing in
  `PredictionValidationConfig` rejects a negative *draw*. Plan §11 says Normal
  is compared by "MAE, RMSE, interval coverage, and residual diagnostics", so
  the negative mass lands directly in the interval-coverage metric that Normal
  is *supposed* to be judged on. Defensible as the honest consequence of a
  continuous comparator on count data — but it should be a declared, recorded
  property of the Normal runs, not a silent one.
- **The Normal total scale assumes cross-cohort independence.**
  `direct_cohort.py:344` sets the total scale to √Σσ². Plan §6 item 6 authorizes
  exactly this ("independent Normal cohort sums have exact Normal total
  declarations"), so it is per spec. It is nonetheless an *assumption*: if
  cohorts are positively correlated within a building — likely, since a large
  building has more of every cohort — the declared total variance is understated
  and total intervals under-cover. The same independence assumption underlies
  the Poisson total and the NB2 convolution. Worth one measured diagnostic
  rather than a silent premise.
- **Cross-family log scores are the same order of magnitude, which makes an
  invalid comparison harder to spot than in Gate 2B.** On identical eval rows:
  Poisson `n_kindergarten` NLL 1.6948, NB2 1.7070, **Normal 1.8180**. Gate 2B's
  laundering was caught partly because 35.45 vs 12.78 was glaring; here a log
  density and a log mass differ by ~7%, so a ranking would look entirely
  plausible while being meaningless. Gate 3 itself does not rank them and
  correctly sets `pointwise_log_probability_scope="marginal"` — this is recorded
  as a cross-reference for Gate 6, which owns the comparison.
- **Interval coverage over-covers, as expected for discrete outcomes.** At a
  nominal 80%: Poisson 0.863–0.950, NB2 0.903–0.930, Normal 0.827–0.860.
  Discreteness inflates coverage for count families; Normal sits closest to
  nominal. Nothing here is miscalibrated, but no test asserts any of it.
- **NB2 dispersion is one constant per cohort, selected by the tuning study.**
  Plan §6 specifies exactly this. It means dispersion cannot vary with the
  covariates, so heteroscedasticity beyond the NB2 mean-variance link is not
  represented. On truly-Poisson simulated data the study correctly selected
  dispersion 0.0023–0.048, i.e. it found its way to the Poisson limit — good
  evidence the NB2 tuning objective works.
- **Returned draws are a prefix of the sample the interval was computed from**
  when `n_predictive_draws < bootstrap_replicates` (`direct_cohort.py:302-312`).
  The replicates are exchangeable, so the prefix is a valid subsample and the
  wider-sample interval is the better estimate. Noted for the record; no change
  recommended.

### Findings withdrawn on evidence

Four suspicions from the Phase 1 plan did not survive checking, and the record is
better for saying so:

1. **"The NB2 custom objective has no `init_score`, so boosting starts at μ=1
   and may never reach the data mean."** **Withdrawn.** On a known signal
   (`g3_p5_recovery.py`), NB2 reached mean 2.103 against a true 2.206 and
   correlation 0.899 with the truth. All three families beat an intercept-only
   baseline on every cohort — RMSE ratios 0.39–0.69, correlations 0.825–0.923.
   The model learns.
2. **"The all-pruned study crashes on `min()` of an empty sequence."**
   **Withdrawn** — it returns normally, because pruned trials carry values. That
   investigation is what produced G3-1, which is the real and worse defect.
3. **"`num_leaves`' conditional upper bound `min(hi, 2**max_depth)` can invert
   and raise."** **Withdrawn.** `LightGBMSearchSpaceConfig.__post_init__:227`
   guards `num_leaves[0] > 2 ** max_depth[0]`, and the sampled depth is always
   ≥ the lower bound, so the range cannot invert.
4. **"The declared parametric uncertainty and the bootstrap-draw uncertainty
   measure different things, so metrics reading both are inconsistent."**
   **Withdrawn as a defect, retained as an observation.** Structurally true, but
   quantitatively the two interval widths agree to ~0.3% (table under G3-4). The
   real consequence is the efficiency finding, not an inconsistency.

Also worth stating plainly, since absence of a finding should not read as absence
of review: my first P2 run reported a worst relative Hessian error of 0.99 and
printed `REJECT`. That was **my probe's** finite-difference step size (h=1e-5
against an NLL of magnitude 347 — pure cancellation), not the code. At h=1e-2 the
worst error over all 125 cases is 9.1e-6. The code is correct.

### MLflow readiness

Assessed against brief §6 and plan §11. No MLflow code written, no dependency
added. Confirmed absent: `grep -rn mlflow src/` returns nothing, and
`DirectCohortModel` has no `test_df`/`holdout_df` parameter on any path.

**Confirmed present — the brief's claim holds.** `diagnostics["search_space"]`
is recorded (`direct_cohort.py:481`) as the full `LightGBMSearchSpaceConfig`,
and per-cohort Optuna trial history is recorded in full, pruned trials included,
with JSON-safe params and `intermediate_values` as `(step, value)` pairs. This is
the asymmetry the brief flags against Gate 4 — noted as a cross-reference, not
investigated here.

**M1 — the recorded search space is not the distribution actually sampled.**
`direct_cohort.py:155-159` samples `num_leaves` with upper bound
`min(search.num_leaves[1], 2**max_depth)`, i.e. **conditional on another sampled
parameter**. The flat record says `num_leaves: (7, 31)`; a trial that drew
`max_depth=3` was actually sampled from `(7, 8)`. A reader reproducing the study
from its own record would use the wrong space. Since the `optuna.Study` is
discarded, the declared distributions are not recoverable anywhere else.
*Recommendation:* record the effective per-trial bounds alongside each trial's
params, or state the conditional rule explicitly in the search-space record.

**M2 — nothing in the tuning payload can be a `log_param`.**
`hyperparameters` is cohort → dict, and `diagnostics["tuning"]` is cohort → full
trial table. Neither flattens. *Recommendation:* per model run, log flat params
`selected.<cohort>.<param>`; log the trial tables as a single tidy CSV **artifact**
(columns `cohort, trial, state, value, param_*`), which is also what plan §11
asks for ("store trial tables as artifacts by default"); and log each trial's
`intermediate_values` as stepped metrics keyed `tuning.<cohort>.objective` with
`step = fold_index` — the data is already in exactly that shape.

**M3 — metric-key stability across families needs the family in the key space,
not in the metric name.** This model emits a `total` parametric spec for Poisson
and Normal but **not** for NB2 (correctly — §6 item 6). A comparison table keyed
`(metric_name, target, aggregation_level)` therefore has a legitimately missing
`total` NLL row for NB2, which must read as "not declared", not as a failure.
Separately, Normal's log density must never share a column with Poisson/NB2 log
masses (plan §11 is explicit, and the ~7% gap measured above makes a wrong
ranking look plausible). *Recommendation:* tag runs with `family` and
`objective_family`, keep the metric name identical, and add a
`likelihood_kind ∈ {density, mass}` tag that any ranking query must filter on.
The `pointwise_log_probability_scope="marginal"` declaration already present is
the right mechanism for the double-counting half and should be surfaced as a tag.

**M4 — seed records belong in an artifact, not in the run record.** ~456 + ~1,200
entries at defaults (G3-5). *Recommendation:* log `derived_seeds` as a JSON
artifact, and promote only the reproducibility-relevant scalars to params —
`master_seed`, `default_rng_seed`, and the per-cohort `optuna-sampler/<cohort>`
seeds, which are the ones that actually replay a study.

**M5 — run identity is content-hash-based and complete except for provenance.**
Present and correct for a Model A child run: `training_data_hash`,
`training_schema_hash`, `implementation_version`, `dependency_versions`
(lightgbm/optuna/numpy), `contract_version`, and per-purpose derived seeds. Still
absent anywhere in `src/`: timestamp, run id, git SHA, experiment name. A git SHA
is genuinely unavailable from inside the package. *Recommendation:* the tracking
adapter injects `source_revision`, `run_started_at` and experiment name at the
boundary; the package should not learn about git. Per plan §11, Model A needs one
child run per family tagged `model_class`/`family`/`objective_family` — nothing in
the current metadata blocks that, since `diagnostics["family"]` is recorded.

**M6 — `normal_mean_policy: "clip to positive minimum"` is a free-text string
describing a real modeling decision.** `direct_cohort.py:482`. It is the only
record that Normal cohort means are clipped at `minimum_mean` (1e-8), which is a
genuine deviation from an unconstrained Normal. *Recommendation:* log the numeric
`minimum_mean` as a param and the clip *incidence* (how many predictions were
actually clipped) as a metric. A policy that never fires and one that fires on
40% of rows are very different runs and currently record identically.

### Code-quality findings

- **`assert dispersion is not None` (`:196`) is load-bearing control flow.** It
  is the only thing standing between a non-NB2 family and the NB2 branch, it
  raises with no message, and `python -O` removes it. Make it a `ValueError`.
- **The family discriminator is re-read at seven sites** (`:139`, `:188`,
  `:223`, `:233`, `:333`, `:374`, `:422`, `:455`). Beyond G3-2, this is the
  duplication itself: adding a family means finding all seven. A single
  validated accessor returning the family together with its ancillary parameter
  and its draw/score kernels would leave one place to change.
- **`self._train_df = train_df.copy(deep=True)` (`:146`)** retains a full deep
  copy of the training partition for the lifetime of the model, solely so
  `_bootstrap_predictive_draws` can resample it. It is also **not** cleared by
  `_reset_model_state`... it is, via `_train_df = None` — correct, noted as
  verified rather than as a finding. The deep copy is defensible (the caller
  could mutate the frame); worth a one-line comment saying that is why.
- **`test_tuning_folds_come_only_from_supplied_training_data` does not test its
  name.** `tests/unit/test_direct_cohort.py:204-212` asserts trial counts and
  that `best_value` is finite. It makes no statement about fold membership or
  about training data. The property is in fact true — I confirmed it in P6 — but
  nothing in the suite establishes it. Rename to match what it asserts and add
  the membership assertion the old name promised.
- **The Gate 3 suite cannot detect a model that does not learn.** Every test
  drives a 24-row frame whose targets are `rng.integers(0, 5)`, independent of
  every feature, with `n_trials=2` and `n_estimators=5–12`. A constant predictor
  passes all 20. This is structurally the same gap as the Gate 6 spy-model
  episode in brief §4. *Recommendation:* add one signal-recovery test on the
  `g3_p5` pattern, asserting each family beats an intercept-only baseline. It
  runs in ~1.5s per family.
- **Good, worth preserving:** the comment at `:325-327` explaining why the
  pointwise keys are marginal and must not be summed is exactly the kind of
  *why* comment the brief asks for, and it is load-bearing for Gate 6. The
  refusal to declare an NB2 total family, scoring it by convolution instead, is
  the statistically honest choice and is implemented correctly.

### Documentation accuracy

| Claim | Status |
|---|---|
| Gate 3 acceptance sentence, all six clauses | **Supported** (verdict table), subject to G3-1 and G3-2, neither of which the sentence speaks to. |
| §6.1 "one `LGBMRegressor` per cohort with model-level family selection: built-in L2 for Normal, built-in Poisson, or a tested custom NB2 objective" | **Supported.** The NB2 objective is genuinely tested — `test_nb2_derivatives_match_finite_differences` in `test_distributions.py` is a real FD test, and my independent derivation agrees. |
| §6.2 "raw `n_apartments` is an ordinary tree predictor; Model A does not use a log-exposure offset" | **Supported and enforced** — `_fit_model:80` and `_predict_model:275` both raise if an exposure offset is supplied. |
| §6.3 "fixed trial count, `n_jobs=1`, and no timeout stopping" | **Supported and enforced at construction** by `OptunaTuningConfig.__post_init__`, which rejects `n_jobs != 1` and any timeout. |
| §6.3 "seeded Optuna TPE over fixed train-only known-neighborhood folds" | **Supported**, probe-confirmed for the train-only half. |
| §6.4 "total mean as their sum" / §6.5 normalized probabilities with zero fallback | **Supported.** |
| §6.6 "a general sum of independent NB2 variables is not NB2; score by finite convolution … rather than declaring a false total family" | **Supported**, and the implementation is exact to 4.4e-15. |
| §6 "Estimate Normal scales from train-only cross-fitted residuals" | **Supported** — `_cross_fitted_normal_scale` refits per fold and pools validation residuals. |
| §6 "select NB2 dispersion within each cohort's train-only tuning study" | **Supported**, and probe evidence shows the selection is meaningful (it finds the Poisson limit on Poisson data). |
| Gate 3 "Tune tree capacity before evaluating any explicit nonlinear or interaction candidate from Gate 2A" | **Supported as an ordering constraint.** `DirectCohortModel` does not override `_select_feature_spec`, so it never evaluates nonlinear/interaction candidates at all — capacity tuning is all it does. Candidate evaluation is Gate 6's. |
| §11 "each cohort's Optuna trial history, selected parameters, derived backend seeds, and Normal scale or NB2 dispersion" logged | **Supported as to availability** — all four are in `get_metadata()`. See M1–M4 on shape. |
| Brief §8: `predictive.py` is a legacy module Gate 9 will remove | **Unsupported — the brief is wrong.** Plan §3.2's removal list is `benchmarks.py`, `direct_cohort_models.py`, `poisson_total_benchmark.py`, `gate3.py` and does **not** include `predictive.py`. All three current model families import `central_prediction_intervals` from it (`direct_cohort.py:31`, `independent_total_probability.py:36`, `bayesian_conditional.py:30`), and it has its own test file. It is live shared code. The brief's §8 line should be corrected so a later gate does not delete it. |
| Brief §5: `tuning.py` "hardcodes direction `minimize` — verify every objective actually is" | **Verified: every objective is a minimand.** Normal returns RMSE, Poisson/NB2 return mean NLL. On identical data, true means score 1.7772 / 1.8652 / 1.9707 (RMSE / Poisson NLL / NB2 NLL) against 2.3798 / 2.2918 / 2.2026 for a constant and 4.2595 / 2.9942 / 2.3781 for a 2× bias — the true-mean row is smallest in every column. The `direction` is right; the *selection rule* under it is not (G3-1). |
| Brief §6: `DirectCohortModel` writes `diagnostics["search_space"]` and per-cohort trial history | **Supported**, confirmed directly. |

### Cross-references for other gates

- **Gate 4 — inherits G3-1 directly.** `tuning.py` is shared;
  `IndependentTotalProbabilityModel` calls `run_optuna_study` too. If Gate 4
  enables pruning anywhere, the same partial-fold selection applies. Fixing
  `tuning.py` in Gate 3's remediation fixes both; Gate 4 should re-verify rather
  than assume.
- **Gate 4 — the `unvalidated_building_count` key (G3-3).** Gate 4 already
  reports it. Whatever key Gate 3 adopts must match Gate 4's spelling exactly,
  or the MLflow comparison gains two names for one quantity.
- **Gate 6 — owns the cross-family comparison that Gate 3 makes possible and
  dangerous.** Gate 3 emits `pointwise_log_probability_scope="marginal"` and
  per-family pointwise keys whose `sum(cohorts) ≠ total` (5.1880 vs 2.3094 for
  Poisson). Gate 6 must not sum them, and must not rank Normal's 1.8180 log
  density against Poisson's 1.6948 log mass. The brief records a
  likelihood-comparability guard already fixed in Gate 6 — the measured
  magnitudes above are the concrete test case it should be checked against.
- **Gate 6 — `predict` cost (G3-4).** 600 LightGBM refits per `predict` at
  defaults, paid per fold per candidate. Gate 6 should know what it is buying
  before it sets `bootstrap_replicates` for a full comparison run.
- **Gate 9 — do not delete `predictive.py`.** See the documentation table. Three
  live models import it.

### Recommended remediation (proposed, not applied)

1. **G3-1** — in `tuning.py`, select the best trial over completed trials only,
   and raise a named error if none completed. Add a regression test built on the
   `g3_p4c` construction (unequal fold difficulty + pruning), which fails on the
   current code.
2. **G3-2** — close the three fall-through chains with an explicit NB2 branch
   plus `raise`; replace the bare `assert` at `:196` with a `ValueError`. Add a
   test that corrupts the discriminator after construction and asserts every
   path refuses, mirroring the Gate 2B test.
3. **G3-3** — record `unvalidated_building_count` in diagnostics using Gate 4's
   exact key, and add `_unvalidated_building_ids` to `_reset_model_state`.
4. **Code quality** — rename `test_tuning_folds_come_only_from_supplied_training_data`
   and give it the membership assertion its name promises; add one
   signal-recovery test asserting each family beats an intercept-only baseline.
5. **Deliberately not proposed:** collapsing the seven family-dispatch sites into
   one accessor (a refactor of code whose correctness was just verified belongs
   in its own change); changing `bootstrap_replicates`' default (a modeling
   decision for the owner, not a review finding); and anything about negative
   Normal draws beyond recording them, since they are the honest consequence of
   a comparator the plan deliberately requested.

### Remediation applied

Suite: **382 passed** (was 376), ruff clean, the one documented Bayesian warning
unchanged to the digit (worst R-hat 1.086056330170683, minimum ESS
25.660557049567544). Gate 3 tests 22 → 28.

| Finding | Fix | Verified against the original probe numbers |
|---|---|---|
| **G3-1** (blocking when pruning is on) | `tuning.py` now selects over `TrialState.COMPLETE` trials only and raises a named `RuntimeError` when no trial completed. Pruned trials are still retained in `TuningResult.trials` as evidence, with their partial `intermediate_values`. Tie-breaking on trial number is kept, so the winner stays order-invariant. | `test_partially_evaluated_pruned_trial_never_wins` reproduces `g3_p4c`'s construction (five folds of increasing difficulty, pruning on, seed 5). On the pre-fix code it fails — the winner was trial 38, `PRUNED`, scored on 1 of 5 folds, value 0.2323 against the best complete trial's 1.0002. On the fixed code the winner is `COMPLETE` and equals the best fully-scored trial. I verified the failure by temporarily reverting the fix, not by assuming it. |
| **G3-2** (defense-in-depth) | The two laundering chains now end in an explicit `elif family == "nb2"` followed by `raise ValueError(f"Unsupported direct-cohort family: {family}")`. The load-bearing `assert dispersion is not None` at `:196` is now a real `ValueError`. | `g3_p7c.py` re-run unchanged. Previously a Normal-fitted model with a corrupted discriminator declared `nb2` with `dispersion=1.7414550792212247` — its own Normal standard deviation. Now all three starting families raise `ValueError: Unsupported direct-cohort family: gamma`. |
| **Code quality** — duplicated ancillary arguments | The `dispersion=... if nb2 else None, scale=... if normal else None` pair, written out at three sites, is replaced by one `_ancillary_kwargs(cohort)` helper. The two kernel call sites are now identical lines. Strict `[cohort]` lookup is preserved, so a missing ancillary parameter still raises `KeyError` rather than silently passing `None`. | Suite green; `g3_p8_coverage.py` re-run gives unchanged coverage and interval widths. |
| **Code quality** — a test that did not test its name | `test_tuning_folds_come_only_from_supplied_training_data` asserted trial counts. It now records which rows each fold's `FittedFeatureTransformer` is fitted on and asserts every one is a subset of the training partition, with the fold-level fits strict subsets. The old assertions kept their own test under an accurate name, `test_every_cohort_records_a_completed_tuning_study`. | The property was probe-confirmed in P6 and is now pinned by the suite. |
| **Test coverage** — the suite could not detect a model that does not learn | Added `test_fitted_model_beats_an_intercept_only_baseline`, parametrized over all three families, on a 160-row table whose cohort counts genuinely depend on the features. Asserts each fitted model beats an intercept-only baseline by RMSE and correlates above 0.5 with the outcome. | **This is the decisive number.** Patching `_predict_mean` to return a constant `2.0` — ignoring every feature, every fitted tree and every tuned hyperparameter — previously gave **20 passed in 0.54s**, against 3.59s unsabotaged; the suite got 6× faster and noticed nothing. Against the new suite the same sabotage gives **3 failed, 21 passed**. A supporting signature visible in the sabotaged log: every Optuna trial returned the identical objective value 1.5299374343949992, because the hyperparameters could not matter. |

**Re-checked on request, and the two answers differed.** The recovery test was
confirmed essential by the sabotage run above. **G3-2 was confirmed *not*
essential**, and that correction belongs on the record: the only path by which
`family` enters from outside the code is `experiment_config.py:182`,
`DirectCohortConfig(**direct_cohort_raw)` from user-editable TOML, and it is
validated — a `configs/modeling.toml` carrying `family = "gamma"` is rejected
with `ValueError: family must be one of ['nb2', 'normal', 'poisson']`
(`g3_p12_toml.py`). G3-2 was therefore unreachable, unlike the Gate 2B defect it
resembles. It was fixed as a low-cost side effect of the dispatch cleanup and on
the consistency argument, not because it was exploitable. My original write-up
over-weighted it by framing it as "the Gate 2B class left open"; the Gate 2B
defect was reachable and this one was not.

**Deliberately not done:**

- **The full family-strategy refactor** (one class per family owning objective,
  link, score, ancillary estimation, declaration and draws). It is the textbook
  design for a variant with this many decision points, but it restructures a
  486-line module whose numerical correctness was just verified, and it buys
  structure rather than behaviour. Same reasoning Gate 2B used for declining to
  decompose `metrics.py`. The three fall-throughs being explicit removes the
  defect risk that motivated it.
- **G3-3** (`_unvalidated_building_ids` is write-only) — **still open.** The fix
  is small, but it adds a metadata key that must match Gate 4's
  `unvalidated_building_count` exactly or the MLflow comparison gains two names
  for one quantity. Worth deciding with Gate 4 in view rather than unilaterally.
- **G3-4/G3-5** (bootstrap cost, seed-record volume) — design findings for the
  tracking layer, not defects.
- Anything about negative Normal draws beyond recording them.

### Contract changes later gates must know

1. **`run_optuna_study` now raises `RuntimeError` when no trial completes.**
   Previously it raised an opaque `ValueError` from `min()` on an empty
   sequence, or — the real defect — returned a pruned trial's partial score.
   **Gate 4 calls this same function** and inherits the fix; it should re-verify
   rather than assume, and must not treat "no completed trial" as recoverable.
2. **`TuningResult.best_*` now always describes a fully evaluated trial.** Any
   consumer comparing `best_value` across cohorts or models is now comparing
   like with like. `trials` is unchanged and still includes pruned records.
3. **`DirectCohortModel` raises on an unrecognized family at every dispatch
   site**, not only where a distribution kernel happens to be called.


---

## Gate 4 — Model B: `IndependentTotalProbabilityModel`

**Reviewed:** 2026-09-12 · `models/independent_total_probability.py`,
`count_regression.py`, `grouped_multinomial.py` · against plan §1105 (Gate 4),
§7, §7.1–§7.3. Probes: `g4_p1_joint_identity.py`, `g4_p2_3_7_10.py` (+`g4_p2b`),
`g4_p4_recovery.py`, `g4_p5_p6.py` (+`g4_p6b`), `g4_p8_p9_meta.py`,
`g4_p11_claims.py`, `g4_p12_bootstrap.py`, `sabotage_b_total.py`,
`sabotage_b_probs.py`.

**Baseline:** 382 passed, ruff clean, Bayesian warning unchanged. Gate 4's files
collect 23 tests (15 + 8). No MLflow import and no `test_df`/`holdout_df` in any
Gate 4 module; `run_cross_model_validation` takes `outer_train_df` only.

### Verdict: **ACCEPT WITH CONDITIONS**

**All eight acceptance clauses hold**, each re-derived independently rather than
taken from the suite. The two conditions (G4-1, G4-2) are *outside* the
acceptance sentence — it does not speak to either — but both should be settled
before Gate 6 runs this model at scale.

| Clause | Result |
|---|---|
| NB2 analytic tests pass | **Holds**, and better than the shipped test shows — see the withdrawal note below. |
| grouped and expanded fits agree | **Holds.** Extended beyond the shipped 4-row fixture to 60 rows with categoricals and zero-total buildings: max coefficient difference **2.8e-16** at C=1 and **7.9e-12** at C=100 (387 literal rows vs 134 grouped). |
| zero totals work | **Holds.** A zero-total building's composition increments are exactly `[0. 0. 0.]`, so it contributes nothing to composition while still contributing to the total likelihood, exactly as §7.2 requires. |
| targets never enter predictor matrices | **Holds** for both the total and probability matrices. |
| probabilities normalize | **Holds** (softmax, structural). |
| calibration uses grouped out-of-fold predictions and cannot worsen its selection NLL | **Holds literally** — `T=1` is inside the bounds, so the fitted NLL is ≤ the raw NLL by construction. See G4-1 for why the guarantee this *implies* is not delivered. |
| expected cohort counts reconcile to totals | **Holds by construction**; `cohort_means = total_mean[:,None] * probabilities`. |
| independent Optuna tuning on identical folds | **Holds.** Both components consume the same `_selection_folds` (`[0,1,2]`), and each tunes only its own parameter. |

**The gate's central statistical claim verifies exactly.** The four pointwise
keys sum to the true joint log mass — against an independent scipy oracle
(`nbinom.logpmf(total) + multinomial.logpmf(cohorts | total)`), max error
**7.1e-15** for NB2 and **1.4e-14** for Poisson over 120 rows. This is *not*
double counting: the total is a separate component and the cohort keys are
conditional on the observed total, so `total + composition = joint`. It is the
structural opposite of Model A, whose `total` key is a convolution of its own
cohort distributions. The shipped test asserted only that the joint metric
equals the sum of the same arrays it was built from, which is circular; this is
the first external check of the identity.

**The model recovers known parameters.** On 400 simulated buildings generated
through the model's own exposure-offset form: total mean correlation **0.9952**,
RMSE ratio 0.103 against an intercept-only baseline; composition mean
|p̂ − p| **0.0161** against a constant-share baseline of 0.0878; and on truly
Poisson data the NB2 dispersion went to **1.0e-4**, its configured lower bound —
the right answer, and evidence that dispersion estimation is live rather than
decorative.

### Findings by severity

**G4-1 — the calibration "evidence gate" is structurally almost incapable of
rejecting, and the temperatures it retains can make held-out calibration worse.**
`independent_total_probability.py:322-339`. **Probe-confirmed** (`g4_p6b.py`).
Not a violation of the acceptance clause as worded; a violation of what it
implies.

`T=1` lies strictly inside `calibration_temperature_bounds=(0.25, 4.0)`, so the
minimiser's value is ≤ the value at `T=1` by construction. `retained` then
compares that minimum against `T=1` **on the same rows the temperature was fitted
on**, with a tolerance of `1e-6`. So the gate fires whenever the optimiser finds
any improvement at all, which on continuous data is essentially always.

On **perfectly calibrated** synthetic logits — where the correct answer is
`T=1` — the gate retained a temperature in **5 of 5** cases:

| rows | fitted T | in-sample gain | retained | out-of-sample gain |
|---|---|---|---|---|
| 100 | 1.0351 | 1.76e-04 | True | **−2.50e-03** |
| 400 | 1.0098 | 1.49e-05 | True | **−8.43e-04** |
| 1600 | 1.0184 | 5.29e-05 | True | **−8.36e-05** |
| 6400 | 1.0174 | 4.66e-05 | True | +6.19e-05 |
| 25600 | 1.0038 | 2.24e-06 | True | +2.11e-06 |

At realistic sample sizes the retained temperature *degraded* held-out
calibration. The only rejection I could produce was the degenerate case of
symmetric, unidentified logits, where the optimiser returned the bound 4.0 with
zero improvement. The tolerance `1e-6` sits two orders of magnitude below the
noise scale (~1e-4) at plausible n.

The magnitudes are small (T ≈ 1.02, losses ~1e-4 nats) and shrink with n, which
is why this is a condition and not a blocker. But §7.2's "retain the fitted
temperature only when it improves out-of-fold NLL beyond a configured numerical
tolerance" reads as a genuine evidence gate, and it is not one.

*Recommended remediation:* gate on data not used to fit `T` — split the
cross-fitted logits, fit on one part and test the improvement on the other — or
apply a 1-degree-of-freedom penalty (an AIC-style `+1` nat, or a likelihood-ratio
test) instead of a bare numerical tolerance. Failing either, amend §7.2 to say
the gate guarantees only in-sample non-worsening.

**G4-2 — `predict` crashes mid-bootstrap when a cluster resample drops an age
class, after `fit` has already succeeded.** `grouped_multinomial.py:39-40`,
reached from `independent_total_probability.py:582`. **Probe-confirmed**
(`g4_p12_bootstrap.py`).

`_fit_grouped_multinomial` raises `ValueError: Probability fitting requires
every age class to be observed`. The neighborhood-cluster bootstrap resamples
whole neighborhoods, so any cohort concentrated in a minority of clusters has a
real per-replicate probability of vanishing. Probed on 60 buildings where
highschool children live in 1 of 6 neighborhoods: `fit` succeeds, then
`predict(interval_levels=(0.8,))` with 40 refits raises. With the default
**200** replicates the chance of at least one failure is high.

Consequence: the whole prediction fails — no partial result, no diagnostic, and
an error message naming a condition the caller cannot act on. Inside a Gate 6 CV
run this kills a fold or a candidate outright. A single high-school catchment in
real data is exactly this shape.

*Recommended remediation:* skip and count failed replicates rather than
propagating, recording `successful_replicates` the way
`neighborhood_cluster_bootstrap` already does in `evaluation.py` — that
convention exists in the codebase and should be reused. Bound the failures with
a configured maximum fraction and raise only past it.

**G4-3 (MLflow) — the search space never reaches metadata, so a run records its
chosen regularization without the bounds it was chosen from.**
`independent_total_probability.py:158-165`. **Probe-confirmed** — the brief's
suspicion is correct.

`diagnostics["selection"]` carries `fold_count`, `unvalidated_building_count`,
both feature specs, and both `TuningResult`s. A repo-wide search of the
serialized metadata for `search_space` returns **False**. Measured on a live fit:
recorded `total_l2_penalty=0.893145` and `probability_c=0.0337089`, with the
configured bounds `(1e-06, 2.0)` and `(0.01, 100.0)` recorded **nowhere**.

This is the asymmetry with Gate 3, which does write
`diagnostics["search_space"]`. A tracked run is not reproducible from its own
record: 0.893 means something very different drawn from `(1e-6, 2.0)` than from
`(0.5, 1.0)`. *Recommendation:* record `asdict(self.independent_config.search_space)`
under the same `diagnostics["search_space"]` key Model A uses, so the column
means the same thing across families.

**G4-4 (test coverage) — the suite cannot detect a composition component that
ignores every feature.** Same class as Gate 3's, confirmed the same way.

| sabotage | result |
|---|---|
| composition logits forced to zero (uniform probabilities) | **15 passed, 0 failed** |
| total mean forced to a constant 5.0 | 1 failed, 14 passed |

The single catch in the second row is `test_log_exposure_has_fixed_unit_coefficient`,
a unit test of the helper the sabotage replaced — not an end-to-end assertion.
Every model-level test passed with a constant total.

Contributing cause: **every test collapses the search space to a point** —
`_config` sets `total_l2_penalty=(0.01, 0.01)` and `probability_c=(1.0, 1.0)`
(`tests/unit/test_independent_total_probability.py:41-45`), so no test exercises
selection at all. With real bounds the tuner does explore and select
(`total_l2=0.893`, `probability_c=0.0337` on my probe).

*Recommended remediation:* add a recovery test on the `g4_p4` pattern asserting
both components beat their respective constant baselines, and one tuning test
with non-degenerate bounds.

**G4-5 (documentation/metadata) — §7.2 requires recording raw *and* calibrated
Brier; only the calibrated one is recorded.** `:341-355` records
`raw_weighted_nll` and `selected_weighted_nll` but only
`selected_weighted_brier`. Trivial to add and needed for the "did calibration
help" comparison the section asks for.

### Statistical judgement calls (not defects)

- **The temperature is fitted on the same folds that selected `probability_c`.**
  §7.2 requires only that the logits be out-of-fold, which they are, and the
  spec's ordering ("after the probability feature specification and
  regularization are frozen") is honored. But `C` was chosen to minimize NLL on
  those very validation rows, so the logits `T` calibrates are mildly optimistic.
  This compounds G4-1 rather than causing it.
- **The last cohort key is ~0 by construction** (max 1.42e-14). This is correct
  for a conditional decomposition — the final cohort is determined once the
  total and the preceding cohorts are known — but it means a per-target NLL table
  is not comparable across models. Recorded as a Gate 6 cross-reference below.
- **Poisson and NB2 are sibling runs, not an in-fit choice.** `total_family` is
  configuration; nothing selects it inside `fit`. This is §7.1 exactly, and it
  is the right call — but it means nothing in Model B itself ever compares the
  two families. That comparison exists only if Gate 6 enumerates both.

### Findings withdrawn on evidence

1. **"`count_regression._nb2_log_probability` duplicates `distributions.pointwise_log_probability`
   and the two could drift."** The duplication is real, and the two *do* disagree
   — worst 1.99e-05 at α=1e-4, μ=1e-6. **But the project's implementation is the
   accurate one.** Against a stable `log1p`-based reference, worst error:
   `count_regression` **1.7e-11**, scipy **2.0e-5**. scipy's `nbinom.logpmf`
   loses accuracy at tiny means; the project's closed form does not. Within any
   reachable range (μ ≥ 0.01) both agree to ≤1.4e-13.
   **This matters practically:** the obvious cleanup — delete the local formula
   and call the shared one — would make the fitting objective *less* accurate.
   Recommend the opposite: keep both, and add a comment recording why the local
   implementation exists.
2. **"The L2 penalty is applied to the mean NLL, so the penalty selected on folds
   is applied ~25% stronger in the full-data refit."** **Withdrawn — wrong.**
   Mean-scaling makes the fit exactly invariant to sample size: duplicating the
   dataset gives `max|coef(n) − coef(2n)| = 0.000e+00` and identical dispersion
   (0.017552). The penalty per observation is constant, which is precisely what
   makes λ transfer correctly from folds to the final refit. This is correct
   design, not a defect.

### Plan claims verified

Every checkable number in the Gate 4 "Reopened during Gate 5 remediation"
paragraphs holds:

| Claim | Measured |
|---|---|
| conditional masses "sum exactly to `scipy.stats.multinomial.logpmf` (0.0 measured error)" | **0.000e+00** over 500 rows. Supported. |
| "the last cohort's value is 0.0 up to rounding (about 1e-13)" | max 1.42e-14, mean 3.02e-15. Supported (slightly conservative). |
| "rejects probabilities that do not sum to one (tolerance 1e-9)" | 1e-10 accepted, 1e-7 rejected. Supported. |
| "scores −∞ … every later key becomes NaN" | prefixes `[-inf -inf -inf]`, increments `[-inf nan nan]`, scipy oracle `-inf`. Supported. |
| "float64 softmax … underflows to an exact zero only for logit gaps above about 745" | 700 → 4.93e-305; **745 → exactly 0.0**. Supported. |
| "3.8 nats at mean total 6, rising to 93 nats at mean total 90" | **Unverifiable** — describes code that no longer exists. Marked as historical, not re-derived. |
| §7.1 initialization: `β₀ = log(Σy/ΣA)`, `α = √(α_min·α_max)` | Both exact to machine precision; the unweighted per-building rate (−1.5627) is correctly *not* used against the exposure-weighted −1.5949. Supported. |
| §7.1 exposure offset fixes the apartment coefficient at 1 | Doubling apartments doubles the mean exactly (ratio `[2. 2. 2.]`); `n_apartments` is absent from the design matrix and present only as `exposure_column`. Supported. |

### Code-quality findings

- **`_select_feature_spec` does not select a feature spec.** `:125-167` assigns
  `self.probability_feature_spec` unchanged and tunes two regularization
  scalars. The name is inherited from the base hook, but in this subclass it
  misdescribes the method by the codebase's own "a name that lies" standard.
  It also **fits the calibration temperature as a side effect** (`:166`), which
  is a third unrelated job in a method named for a fourth.
- **`selected_weighted_brier` is not a Brier score.**
  `_weighted_proportion_squared_error`'s own docstring says it "differs from the
  per-child Brier score in `metrics.composition_brier_score`". The docstring is
  honest; the metadata key is not, and it is the key that reaches a tracked run.
- **The model retains two copies of the training data**: `_train_df`
  (deep copy, `:397`) and `_selection_folds`, which hold further frame
  references per fold. Gate 3 retains one. Worth a comment on why, or clearing
  `_selection_folds` after calibration.
- **Good, worth preserving:** the comment at `:511-515` explaining why the
  cohort entries are conditional masses *including* the coefficient is exactly
  the kind of *why* comment that prevents a regression — it documents a bug that
  was actually shipped once. `_grouped_multinomial_rows` dropping zero-weight
  rows before fitting is both correct and clearly the right place for it.

### Cross-references for other gates

- **Gate 5 — the comparability claim is half-verified.** I verified Model B's
  side: its cohort keys are conditional multinomial masses summing exactly to
  the scipy oracle. Whether the Bayesian model's Dirichlet-multinomial keys
  decompose the same way, and are therefore comparable key-by-key, is Gate 5's
  to establish. Do not inherit it from this section.
- **Gate 6 — a per-target NLL table is not model-comparable.** Model B's last
  cohort key is ~1e-14 by construction, Model A's is a genuine marginal mass.
  A per-target comparison would show Model B beating Model A on `n_highschool`
  by exactly Model A's value, for no modeling reason. Only the **sum** over keys
  is order-invariant, and it is a joint score only for Model B (and, pending
  Gate 5, the Bayesian model).
- **Gate 6 — G4-2's bootstrap crash is a fold-killer.** A candidate that fits
  and then dies in `predict` needs a defined behaviour in the runner.
- **Gate 3 — G3-3 key naming is now decidable.** Gate 4 records
  `unvalidated_building_count` inside `diagnostics["selection"]`. Gate 3 should
  use the same key name; the nesting differs (Model B has a `selection` block,
  Model A does not), which is itself worth settling for the MLflow schema.
- **The `tuning.py` fix from Gate 3 is confirmed in situ.** Both of Model B's
  studies selected `COMPLETE` winners scored on all 3 folds.

### Recommended remediation (proposed, not applied)

1. **G4-2** (robustness) — skip-and-count failed bootstrap replicates, reusing
   the `successful_replicates` / `max_failed_fraction` convention already in
   `evaluation.py`, rather than letting one resample kill the prediction.
2. **G4-1** (evidence gate) — gate the temperature on held-out rows or a
   1-df penalty; or amend §7.2 to claim only in-sample non-worsening.
3. **G4-3** (MLflow) — record `search_space` under the same key Model A uses.
4. **G4-4** (coverage) — a recovery test for both components, plus one tuning
   test with non-degenerate bounds.
5. **G4-5** — record `raw_weighted_brier` alongside the calibrated one.
6. **Naming** — rename or split `_select_feature_spec`; rename
   `selected_weighted_brier` to what it measures.
7. **Not proposed:** deduplicating the NB2 log-pmf (it would reduce accuracy —
   see withdrawal 1); any change to the mean-scaled penalty (withdrawal 2).

### Phase 2b — gaps closed after the first report

The first Gate 4 report left six things unverified. They were closed before
remediation; five passed, one produced a new finding, and one of my own earlier
claims was withdrawn.

**G4-6 — `selection_improvement_tolerance` is dead configuration.**
`modeling_config.py:293` (as was), exposed to operators at
`configs/modeling.toml:72`. **Probe-confirmed.** Defined, validated in
`__post_init__`, loaded from TOML — and **never read by any code**. A repo-wide
search finds only its own definition, its own validation, and the TOML line. The
only improvement gate that exists is `calibration_improvement_tolerance`. An
operator tuning this knob today changes nothing, which is the brief's "a knob
with one value" pattern in its most literal form, and it is user-facing.

**Verified correct (no findings):**

| Gap | Result |
|---|---|
| Leakage containment, instrumented | **Clean.** 12 transformer fits during `fit` — 10 fold-level, 2 full-partition. All inside the training partition; every fold-level fit a strict subset (90 of 180 rows at 2 folds). Model B builds fold transformers in three places (`_score_fold`, `_fit_temperature`, the bootstrap) and all three are contained. |
| Optimizer non-convergence | **Correct.** `maxiter=1` and `2` both raise `RuntimeError: NB2 optimization failed: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT` (and the Poisson equivalent); `maxiter=300` converges. The branch works; it was simply untested. |
| Grouped equivalence at the low C bound | **Holds.** C=0.01 → max coefficient difference **4.2e-17**; C=0.1 → 1.7e-16; C=1 → 8.9e-16. Equivalence holds across the full default range (0.01–100), not just the two values first checked. |
| Bootstrap cost | **Not a concern.** 26 ms/replicate, flat across 10 and 50 replicates → ~5.2 s at the default 200. Model B fits two components per replicate but both are cheap convex fits; contrast Model A's 600 LightGBM refits. No efficiency finding. |
| Same-seed reproducibility | **Holds**, independently re-derived: identical total means, probabilities, draws, temperature, and selected parameters across two fits at seed 44. |

**Withdrawn: "a different fit seed produces identical predictive draws."** My
check compared only the `total` draws. Those are correctly deterministic given a
fixed `l2` (scipy L-BFGS-B has no randomness) and a fixed predict seed; the fit
seed reaches the cohort split through the temperature. Cohort draws do differ
across fit seeds. The probe was too narrow, not the code.

**G4-1 refined, and partly corrected.** The seed sweep that resolved the above
gives a sharper picture than the synthetic result, and corrects it. On real
fitted logits over six fit seeds on identical data:

| fit seed | fitted T | retained | applied T | in-sample gain |
|---|---|---|---|---|
| 44 | 1.0039 | **False** | 1.0000 | 0.0 |
| 45 | 1.0615 | True | 1.0615 | 2.25e-04 |
| 46 | 0.9998 | **False** | 1.0000 | 0.0 |
| 47 | 0.9829 | True | 0.9829 | 1.67e-05 |
| 48 | 1.0258 | True | 1.0258 | 3.96e-05 |
| 49 | 1.0551 | True | 1.0551 | 1.75e-04 |

So the gate **does** reject on real data (2 of 6), and my original phrasing
"structurally almost incapable of rejecting" was overstated for that case — it
holds for the synthetic well-calibrated data (0 of 5), not here. The sharper
statement is that **the retain/reject decision is driven by the fold draw rather
than by evidence**: identical data and a different seed flips it, with accepted
"gains" of 1.7e-05 to 2.3e-04, i.e. noise scale. That is a stronger objection
than the original one, and it is the reason the remediation below leaves the
decision to the owner.

A concrete, checkable alternative: `_weighted_composition_nll` returns a
**per-child mean**, so with `W` children the likelihood-ratio statistic is
`2·W·(raw − fitted)` against `chi2(1) = 3.841` at 95%. On these numbers that
gives ≈0.89 for seed 45 (reject, correctly — it is noise) and ≈2170 for the
deliberately overconfident case in `g4_p5_p6.py` (retain, correctly). One line,
standard, and it separates the two cases the current tolerance cannot.

### Remediation applied

Suite: **392 passed** (was 382), ruff clean, Bayesian warning unchanged.
Gate 4's own file: 15 → 23 tests.

| Finding | Fix | Verified against the original probe numbers |
|---|---|---|
| **G4-2** (bootstrap crash) | `_bootstrap_predictive_draws` now catches the unfittable-composition `ValueError`, skips and counts that replicate, and raises only past a new `bootstrap_max_failed_fraction` (default 0.25) — mirroring `EvaluationConfig.max_failed_fraction`. Draw columns are accumulated and stacked, so the matrix width is the number of *successful* replicates. | The original probe (highschool children in 1 of 6 neighborhoods) previously raised the opaque `ValueError: Probability fitting requires every age class to be observed`; it now raises `RuntimeError: 15 of 40 bootstrap replicates could not fit the composition model because a resample omitted an age cohort entirely…`, naming cause and counts. The **skip** path is separately confirmed at 2 of 6 neighborhoods: **9 of 40 failed, prediction succeeded**, draw width 31, cohort draws still summing exactly to total draws, intervals finite and ordered. Previously that case crashed. |
| **G4-3** (MLflow) | `diagnostics["selection"]["search_space"]` now records `asdict(independent_config.search_space)`, under the same key name `DirectCohortModel` uses. | `'search_space' anywhere in metadata`: **False → True**. Bounds `(1e-06, 2.0)` and `(0.01, 100.0)` now travel with the chosen `total_l2_penalty=0.893145` / `probability_c=0.0337089`. |
| **G4-5** | `raw_weighted_share_error` recorded alongside the calibrated one, so "did calibration help" is answerable from the record on both scores. | Both keys present in the calibration block. |
| **G4-6** (dead config) | `selection_improvement_tolerance` removed from `IndependentTotalProbabilityConfig`, its validation, and `configs/modeling.toml`. Removal rather than wiring, following Gate 1's F5 precedent: the behaviour it implied does not exist and nothing wanted it. | Config tests pass; the TOML loader (which passes `**raw`) would have raised on a stale key, so the removal is verified by the loader itself. |
| **Naming** | `selected_weighted_brier` → `selected_weighted_share_error`. The function's own docstring already said it is *not* the per-child Brier score; the metadata key was the part that lied, and it is the part that reaches a tracked run. | Key renamed; a test now asserts both share-error keys are present. |
| **Structure** | `_fit_temperature` moved out of `_select_feature_spec` into `_fit_model`. Semantically identical — the base calls `_select_feature_spec` then `_fit_model`, and calibration needs only the selected `C` and the selection folds — but each method now does one job, and the method named for selecting a specification no longer fits an estimator as a side effect. | Suite green; temperature and all downstream draws unchanged at a fixed seed. |

**Probes promoted to tests** (these are the durable artifacts; the scripts are not):

- `test_pointwise_keys_sum_to_the_true_joint_log_mass` — the scipy oracle from
  `g4_p1`. The shipped test was circular; this one would have caught the
  missing-coefficient bug the plan records as having shipped once.
- `test_both_components_beat_their_constant_baselines` — the recovery check.
- `test_tuning_selects_from_a_recorded_search_space` — the first test with
  non-degenerate bounds, pinning both that selection happens and that the space
  is recorded.
- `test_calibration_applies_exactly_one_when_it_is_not_retained` — pins the
  mechanism without blessing the threshold, since G4-1 is unresolved.
- `test_bootstrap_skips_replicates_that_omit_a_cohort` and
  `test_bootstrap_raises_when_too_many_replicates_fail` — both sides of the new
  behaviour, with an assertion message that fires if the fixture ever stops
  provoking a failure.
- `test_total_optimizers_report_non_convergence`, and
  `test_grouped_and_literal_expansion_are_equivalent` parametrized over
  C ∈ {0.01, 1.5, 100.0} with a zero-total building added.

**Sabotage re-run — and my first attempt at the recovery test was too weak.**

| sabotage | before | after |
|---|---|---|
| composition logits forced to zero (uniform probabilities) | 15 passed, 0 failed | **1 failed**, 22 passed |
| total mean forced to a constant | 1 failed, 14 passed (incidental, via a helper unit test) | **2 failed**, 21 passed |

Worth recording because it nearly slipped through: my first version of the
recovery test asserted only that the composition beats a *constant-share*
baseline, and the uniform sabotage **still passed all 23 tests**. The fixture's
true composition averages to roughly uniform, so 1/3-everywhere scores about as
well as the observed overall share. The assertion that has power is that
predicted probabilities must *vary* across buildings and correlate with the truth
per cohort (σ > 0.01, r > 0.5). The sabotage harness caught a weak test that
would otherwise have been recorded as adequate coverage.

**Deliberately not done — G4-1 remains open.** Changing the retention rule
changes which model ships, which is an owner's decision, not a reviewer's. The
two options are stated above (likelihood-ratio test at 1 df, or a held-out gate);
a third is to amend §7.2 to claim only in-sample non-worsening, which is what the
code actually delivers. Until one is chosen, the calibration temperature applied
to a given run depends on its fold draw.


---

## Gate 5 — Bayesian NB2 + Dirichlet-Multinomial In Pyro

**Reviewed:** 2026-09-12 · `models/bayesian_conditional.py`,
`models/bayesian_components.py`, `models/bayesian_inference.py` · against the
plan's Gate 5 section, §8, §11, and `docs/BAYESIAN_CONDITIONAL_MODEL.md`.

### Verdict: **ACCEPT WITH CONDITIONS**

The statistical core is correct, and the two claims that most needed independent
proof both hold at better than the documented precision. The conditions are
about **what the model records** and **what its tests can detect**, not about
what it computes.

**Baseline reproduced exactly:** 392 passed, 1 warning, in 100.8s; the warning
carries `worst_rhat=1.086056330170683` and
`minimum_effective_sample_size=25.660557049567544`, matching the documented
values to the digit. `ruff check src/age_group_prediction/` clean.
`git status --porcelain` unchanged from session start.

#### Clause-by-clause

| # | Acceptance clause | Result |
|---|---|---|
| A1 | Torch/Pyro NB2 moments match **project** definitions | **MET.** Four-way comparison against an independent `log1p` reference over 432 (μ, φ, y) cases: the `pyro.factor` closed form worst error **2.3e-11**, `torch.distributions.NegativeBinomial` **2.8e-11**, `nb2_torch_parameters` **2.8e-11**, scipy **9.9e-6** (at μ=1e-6). Sampler moments match NB2 exactly (μ=6, φ=2.7 → mean 6.00078/6, var 19.362/19.333). |
| A2 | Priors plausible; bounds meaningful not decorative | **MET.** All three load-bearing statistics fire on a targeted prior: children/apartment (`total_intercept_loc=4.0`), `expected_dominant_share` (logit scales 100/20), `concentration_quantile` (log κ ~ N(−3,1)). The prior-predictive `action` is confirmed independent of the convergence `action`. |
| A3 | Tiny-data MCMC completes | **MET.** Both `slow` NUTS tests pass in the control run. |
| A4 | Diagnostics extracted **and thresholded** | **MET for `action="error"`; PARTIAL for `"warn"`.** Under `error` a failing stage raises, carries `stage`/`failures` (3)/`diagnostics`, and leaves `_total_posterior is None`, `is_fitted False`. Under `warn` the failure is unrecorded — see **G5-1**. |
| A5 | Posterior-predictive shapes stable | **MET.** All eight declared posterior shapes match the model guide. Public payloads correct across `n_draws=12/levels=2`, `n_draws=0/levels=1` (the internal 500-draw path), and `n_draws=5/levels=()`. |
| A6 | Every draw reconciles exactly | **MET.** Reconciliation, integrality, and nonnegativity hold on every configuration probed. |
| A7 | Does not reopen feature search | **MET.** `BayesianConditionalModel` does not override `_select_feature_spec`; the probability transformer's metadata is byte-identical before and after `predict`; an unseen categorical level is *rejected*, not silently relearned. |
| A8 | **Comparability — proven here, not inherited** | **MET, at better than documented precision.** Against an independent `scipy.stats.dirichlet_multinomial.logpmf` oracle integrated over posterior samples by hand, on a deliberately **non-degenerate** stubbed posterior: (i) cohort keys sum to the posterior-integrated joint DM score, worst error **1.78e-15**; (ii) the last cohort key is **8.9e-16**; (iii) the `"total"` key matches an independent `log1p` posterior-integrated NB2 to **1.78e-15**. Fitting Model B and the Bayesian model on one frame gives identical key sets, both declaring `sequential_joint`, both with last-cohort key ~0 (1.8e-15 / 3.6e-15), and Model B's sum matches `scipy.stats.multinomial.logpmf` to 7.1e-15. **Gate 6 may rely on key-by-key comparability between these two models.** |
| A9 | Reduced profile cannot silently pass as full | **MET, weakly.** `active_profile` is recorded (nested in `hyperparameters`) and `_run_nuts` writes `chains`/`warmup_steps`/`posterior_samples`/`target_acceptance`/`maximum_tree_depth` into each stage's diagnostics. A reduced run is therefore distinguishable from a full one. What is *not* distinguishable is a reduced run that **failed** its policy from one that passed — see **G5-1**. |

### Findings, ranked

**G5-1 — Diagnostic policy failures are not recorded anywhere.** *Medium-high;
probe-confirmed; blocks Gate 6's §9.3 convergence constraint.*
`bayesian_conditional.py:190-196` stores raw diagnostic numbers, then
`_apply_diagnostic_policy` (`:201-220`) computes `failures` and, under
`action="warn"`, emits them as a `RuntimeWarning` and **discards them**. The
recorded `diagnostics` dict has no `failures`, no pass/fail flag, no applied
thresholds, and no profile name. The default `active_profile` is `"reduced"`,
whose action **is** `"warn"`.
*Failure scenario:* a reduced-profile candidate fits with worst R-hat 1.9 and 37
divergences. `fit` succeeds, `is_fitted` is True, two warnings scroll past, and
`get_metadata()` is structurally identical to a clean run — same key set, only
different numbers. Plan §9.3 names convergence an explicit selection constraint,
so Gate 6 must re-run `evaluate_stage_diagnostics` with the correct policy
itself, or silently select a non-converged model.
*Probe:* `p5_gating.py` — under `error`, raise + 3 failures + cleared state; under
`warn`, `fit` succeeded and every one of `failures`/`policy_failures`/`converged`/
`policy_passed`/`action`/`active_profile` was absent from the recorded payload.

**G5-2 — G4-1 re-derived, confirmed, and strengthened.** *Medium; probe-confirmed;
owned by this gate per instruction §4(a).*
The structural claim holds exactly: `T=1` is interior to `(0.25, 4.0)`, so
`fitted_nll <= raw_nll` by construction, and `retained` compares them on the same
cross-fitted rows `T` was fitted on at a `1e-6` tolerance
(`independent_total_probability.py:326-344`).
*New, decisive measurement:* on **perfectly calibrated** logits generated from
the true model, where `T=1` is correct by construction, the configured rule
retained **143 of 150** replicates across n = 100…1600, while a 1-df
likelihood-ratio test retained **8 of 150** — 5.3%, essentially the nominal 5%.
The configured `1e-6` tolerance is ~1280× below the LRT threshold `3.841/(2W)`
at W≈1500. The gate has effectively no power.
*Counter-check that the LRT is not merely conservative:* on a fixture with
genuine overconfidence, six fit seeds produced fitted `T` of 1.41–2.01 with gains
1.2e-3…4.0e-3, and the LRT retained **all six** (p = 0.0001…0.0343).
**Recommendation: remedy 1, the likelihood-ratio test at 1 df.** It is correctly
sized on calibrated data and retains genuine overconfidence. This changes which
temperature ships for most runs.

**G5-3 — The suite has no power over the prediction path.** *Medium; probe-confirmed;
test adequacy, not a code defect.*
Sabotage harness over the 46-test Gate 5 suite (`test_bayesian_conditional.py`,
`test_distributions.py`, `test_bayesian_recovery.py`):

| Sabotage | Result |
|---|---|
| `_composition_probabilities` → uniform, ignoring every feature | **46 / 46 pass** |
| `_total_posterior_means` → constant 3.0, ignoring features, exposure, and neighborhood | **46 / 46 pass** |
| NB2 `pyro.factor` × 0 | 1 genuine failure — `test_real_nuts_posterior_recovers_known_generating_parameters` |

The third variant's second reported failure was a **probe artifact** (my
replacement function's `__name__` broke the unit test's `model.__name__`
dispatch, giving `KeyError: 'total_intercept'`), so the honest tally is **1 of 46**.
That one detection is exactly the documented claim about the recovery test's
contraction assertions, which is therefore **supported**.
*Root cause:* the only unit fit/predict test monkeypatches `_run_nuts` **and**
`_run_prior_predictive` (`test_bayesian_conditional.py:150-161`), and its stub
sets composition intercepts and coefficients to **zeros** — the fixture is
already the uniform-composition sabotage, so it cannot detect it. No Pyro model
definition is executed anywhere in `tests/unit`.

**G5-4 — Three NB2 implementations; the reported score uses the least accurate.**
*Low; probe-confirmed; consistency rather than correctness.*
`bayesian_conditional.py:391-401` calls `scipy.stats.nbinom.logpmf` directly,
bypassing both the project's own `nb2_scipy_parameters` helper and the closed
form the Pyro model actually scores with. Measured worst error in the reachable
range (μ ≥ 0.01): scipy **5.99e-9** vs the project closed form **2.3e-11**.
**I am withdrawing the correctness half of this lead:** 6e-9 nats is negligible
against any NLL difference between models, so this is a duplication and
parameterization-drift finding, not a defect. Consistent with Gate 4's result
that scipy is the less accurate side, the *recommendation is not* "deduplicate
toward scipy".

**G5-5 — Reconciliation tolerance asymmetry.** *Low; probe-confirmed.*
`_fit_model:129-130` uses `np.isclose`; `_pointwise_log_probabilities:385` uses
exact `np.array_equal` for the same invariant. Cohorts summing to 7.0000000001
against a total of 7.0 pass at fit and are rejected at scoring. Unreachable while
targets are integers, but the two lines should agree.

**G5-6 — Scaling, confirmed not rediscovered.** *Low; already on record.*
`_predictive_draws:449-454` loops draws × rows in Python with one
`rng.dirichlet` per row. Two cheap adjacent items worth fixing while nearby:
`_predictive_draws:437` re-calls `_posterior_arrays()`, redoing the full
torch→numpy conversion already done at `:248`; and `_pointwise_log_probabilities:392-401`
loops over **every posterior sample** in Python for a trivially vectorizable
scipy call (4,000 iterations at the full profile). Memory: `alpha` is
`(S, 512, K)` ≈ 49 MB at the full profile, with several same-sized temporaries
inside the prefix helper.

### Statistical judgement calls (not defects)

1. **The two stages are a cut model.** The composition stage conditions on
   *observed* totals and is fitted separately, so "the joint posterior is the
   product of the two" is exact for what is computed but is an approximation to
   the true joint posterior — composition parameters never inform the total
   stage. This is coherent and documented; the joint NLL is a genuine
   posterior-predictive joint score for the model as defined. Worth stating
   plainly in the guide rather than changing.
2. **κ's prior sits close to its own violation threshold.** On a plausible
   synthetic design the configured prior gave `concentration_quantile` **1.076**
   against a limit of **1.0** — a 7.6% margin. The documented full-profile run on
   real simulator data gave 1.521. The check is meaningful, but a modest design
   change could trip it spuriously; worth knowing before it fires mid-comparison.
3. **Pre-2021 diagnostics.** Pyro 1.9.1's `split_gelman_rubin` and classic ESS,
   no rank-normalization, no bulk/tail split, no E-BFMI — precisely the blind
   spot for a `neighborhood_scale` funnel. Documented and accepted; I agree it is
   a real limitation rather than a defect.
4. **`expected_cohort_shares` is provably inert at K=3.** Confirmed numerically:
   as the logit scale grows to 1000 the ratios converge to 1.174 / 1.165 / 0.661,
   strictly inside `[0.25, 2.0]`. The documentation already says this. Keeping an
   inert check is defensible for K ≥ 6; it should stay labelled as such.

### MLflow readiness (design review; no code written, no dependency added)

- **§11's "reconstructable state bundle … and tested loader rather than opaque
  pickling" does not exist.** There is no `save`, `load`, `to_dict`, `from_dict`,
  `state_dict`, or any other export in the three Bayesian modules. Posterior
  tensors live only on the instance. A fitted Bayesian model cannot currently be
  persisted or reloaded at all. **This is the largest Gate 5 MLflow gap.**
  Recommend a bundle of the two posterior dicts + the two feature specs +
  neighborhood lookup + config + schema, with a loader test.
- **Posterior summaries are absent.** Convergence summaries are recorded per
  stage; per-parameter posterior mean/sd/quantiles are not, anywhere. §11 asks
  for both. Recommend adding a compact per-site summary.
- **The Gate 3/4 `search_space` asymmetry is discharged for Gate 5.** There is no
  Optuna study here; the analogue is priors plus sampler settings, and both are
  recorded (`priors`, `hyperparameters`, and per-stage sampler fields).
- **`hyperparameters` carries both profiles and both diagnostic policies**
  (`reduced_profile`, `full_profile`, `reduced_diagnostics`, `full_diagnostics`)
  when only one of each is active, nested ~5 levels. Flattened into MLflow params
  this is noise that differs between runs for no reason. Recommend logging the
  *resolved* active profile and policy as params, the rest as an artifact.
- **`prior_predictive.cohort_share_quantiles`** is a list-of-lists — artifact
  material sitting in what will be flattened into params.
- **`dependency_versions`** records torch and pyro only; numpy and scipy both
  materially affect these numbers.
- **`last_prediction_unseen_neighborhood_count`** is last-call state mutated
  during `predict` (`:335`) and read by `get_metadata()`. Logged after a
  different `predict` it silently describes the wrong call.
- **G5-1 is also an MLflow finding:** with no pass/fail flag, a tracked
  comparison cannot filter or tag non-converged runs.
- Confirmed absent and correct: no MLflow import anywhere in `src`/`tests`, no
  `test_df`/`holdout` path in any Bayesian module.

### Documentation accuracy

| Claim | Status |
|---|---|
| "Torch/Pyro NB2 moments match project definitions" | **Supported** (2.3e-11) — but the cited unit test covers `nb2_torch_parameters`, a helper the Bayesian model **never imports**. The claim is true; the *test evidence for it* is **overstated**. |
| Joint DM agrees with scipy "to about 1e-13 at this project's scale" | **Supported**, conservatively — measured 1.8e-15 here. |
| Last cohort value "0.0 up to rounding (about 1e-13)" | **Supported** — 8.9e-16. |
| Recovery test has power; fails with the NB2 factor zeroed | **Supported** — reproduced under sabotage. |
| `_mean_accept_prob` must be read inside `sample()` because `cleanup()` resets it; its denominator restarts after warmup | **Supported** — verified against Pyro 1.9.1 `HMC.sample` (`n = self._t - self._warmup_steps`). |
| Divergences are retained-only | **Supported** — `self._divergences.append(self._t - self._warmup_steps)` guarded by `self._t >= self._warmup_steps`. |
| `_build_tree` tree-depth argument position | **Supported** — `tree_depth` is the 6th bound-positional argument, matching `_TREE_DEPTH_ARGUMENT_POSITION = 5`; the top-level loop runs `0 .. max_tree_depth-1`, so recording `tree_depth + 1` makes `>= max_tree_depth` the correct saturation test. |
| `expected_cohort_shares` "cannot fire" at K=3 | **Supported** numerically. |
| Prior-predictive `action` independent of convergence `action` | **Supported**. |
| Posterior prediction "will not scale" | **Supported** — confirmed, not rediscovered. |

### Cross-references for Gate 6

1. **Key-by-key comparability between Model B and the Bayesian model is proven**
   (A8). Gate 6 may rely on it.
2. **The per-target trap, quantified.** Fitting all three models on one frame:
   on `n_highschool`, Model A's mean key is **−2.51030** (marginal, informative)
   while Model B's and the Bayesian model's are **−0.00000** (conditional, ~0 by
   construction). A per-target NLL table hands the latter two a **2.51-nat** win
   for no modeling reason. Gate 6 must rank on `joint_predictive_nll` only.
3. **Key order differs across models.** Bayesian and Model B emit
   `['total', …cohorts]`; Model A emits `[…cohorts, 'total']`. Harmless for dict
   access; a hazard for any positional handling.
4. **G5-1 blocks plan §9.3's convergence-as-selection-constraint** until
   diagnostics carry a policy outcome.
5. Model A declares `marginal` and correctly fails the joint capability gate.

### Recommended remediation (proposed, not applied)

| # | Fix | Priority |
|---|---|---|
| R1 | Record the policy outcome: add `failures`, a boolean pass flag, the applied thresholds, and `active_profile` to each stage's diagnostics, written **before** the warn/raise branch so both paths record it. Addresses G5-1 and the §9.3 cross-reference. | High |
| R2 | Replace Model B's retention rule with the 1-df likelihood-ratio test: retain iff `2·W·(raw − fitted) > chi2.ppf(0.95, 1)`. Record the statistic and p-value in the calibration diagnostics. Addresses G5-2; **changes which model ships**. | High |
| R3 | Close the G5-3 gap with tests that have demonstrated power: an oracle test for the pointwise decomposition on a **non-degenerate** posterior, a cross-model key-identity test, and a direct oracle test for the `pyro.factor` NB2 formula. Each must be shown to fail under the corresponding sabotage. | High |
| R4 | Route the pointwise total through the project's own NB2 closed form rather than scipy, and vectorize that loop; drop the redundant `_posterior_arrays()` call. G5-4, G5-6. | Medium |
| R5 | Align the two reconciliation checks on one tolerance. G5-5. | Low |
| R6 | MLflow: add a posterior state bundle with a tested loader, and per-site posterior summaries. Design recommendation only — **no MLflow code**. | Medium (Gate 7 work) |

Probe scripts (session scratchpad, not re-runnable): `p2_nb2_oracle.py`,
`p3_dm_oracle.py`, `p4_cross_model.py`, `p5_gating.py`, `p6_prior.py`,
`p7_shapes.py`, `g41_calibration.py`, `g41_power.py`, `sabotage_{comp,total,factor}.py`.
Contract-bearing probes are promoted to repository tests in Phase 3.

### Remediation applied (same session)

R1–R6 applied. R6 was initially deferred to Gate 7 and was then completed in
this session on the user's instruction to finish the gate. It adds no MLflow
code and no tracking dependency, so instruction §9 is respected: the state
bundle is a model capability that plan §11 requires, not a tracking adapter.

**Suite: 407 passed** (was 392), ruff clean over `src/age_group_prediction/` and
both touched test files, with the same single documented warning
(`worst_rhat=1.086056330170683`, `minimum_effective_sample_size=25.660557049567544`).
`git status --porcelain` is **identical to session start** — every file touched
was already modified or untracked.

Each fix is verified below **against the original probe numbers**, not merely
against a green suite.

| Fix | Finding | Verified against the original probe |
|---|---|---|
| **R1** — `_evaluate_stage_policy` records `policy_passed`, `policy_failures`, `policy_thresholds` (full `asdict(policy)`), and `active_profile` on **both** paths and **before** the warn-or-raise branch (`bayesian_conditional.py:201-224`) | G5-1 | `p5_gating.py` re-run: the four keys that were all **absent** are now all **present**. Under `action="error"` the raise still carries `stage`/`failures`/`diagnostics` and still clears fitted state. |
| **R2** — retention is now a 1-df likelihood-ratio test; `calibration_improvement_tolerance` **removed**, `calibration_significance_level` (0.05) added, both in `modeling_config.py` and `configs/modeling.toml`; the statistic, critical value, p-value, significance level and child count are recorded | G5-2 | `g41_calibration.py` re-run: all six seeds on the genuinely overconfident fixture **still retain** (statistics 4.48–15.13 against 3.841), so real calibration gains are not discarded. A real fit's recorded `likelihood_ratio_statistic` is **14.435066619586976**, matching the probe's independent `2·W·(raw−fitted)` of **14.4351** for that seed. The original power measurement stands: **143/150** false retentions under the old rule vs **8/150** under this one. |
| **R3** — seven new Bayesian tests and three new Model B tests | G5-3 | Sabotage harness re-run; see the table below. |
| **R4** — pointwise total routed through `distributions.pointwise_log_probability`, the same shared NB2 entry point Models A and B use, and evaluated for all posterior samples in one call instead of a per-sample Python loop; `_predictive_draws` now receives the already-converted posteriors instead of re-calling `_posterior_arrays()` | G5-4, G5-6 | `p3_dm_oracle.py` and `p2_nb2_oracle.py` re-run: **every number unchanged** — joint 1.776e-15, last key 8.882e-16, total 1.776e-15; `pyro.factor` 2.261e-11, scipy 5.986e-09. The change is numerically exact. |
| **R5** — `_pointwise_log_probabilities` now uses `np.isclose(...).all()`, matching `_fit_model` | G5-5 | A frame accepted at fit can no longer be rejected at scoring. |

**Sabotage harness, before and after** (the measurement that drove R3):

| Sabotage | Before | After |
|---|---|---|
| `_composition_probabilities` → uniform | **46 / 46 passed** | **detected** — `test_pointwise_cohort_keys_sum_to_an_independent_joint_dirichlet_multinomial` |
| `_total_posterior_means` → constant, ignoring every input | **46 / 46 passed** | **detected** — `test_total_posterior_mean_responds_to_exposure_features_and_neighborhood` |
| NB2 `pyro.factor` × 0 | detected by 1 test | detected |

**A circular test was caught and fixed during this phase, not after it.** The
first version of `test_pointwise_total_key_matches_an_independent_nb2_reference`
built its own reference by calling `_total_posterior_means` — the very helper the
total sabotage replaces — so it agreed with a constant-mean model and the total
sabotage still passed 52/52. That test is retained for what it genuinely covers
(the NB2 parameterization of the total key), and
`test_total_posterior_mean_responds_to_exposure_features_and_neighborhood` was
added alongside it, built only from the public `predict` payload, to carry the
power. This is precisely the failure mode instruction §8 warns about.

**Tests promoted from probes** (the durable evidence; scratchpad probes do not
survive the session):

| Test | Probe it replaces | Contract |
|---|---|---|
| `test_pointwise_cohort_keys_sum_to_an_independent_joint_dirichlet_multinomial` | `p3_dm_oracle.py` | Cohort keys sum to the posterior-integrated joint DM score against a `scipy.stats.dirichlet_multinomial` oracle; last key is zero. Carries an explicit fixture guard rejecting a near-uniform composition, so it cannot silently lose power. |
| `test_pointwise_total_key_matches_an_independent_nb2_reference` | `p3_dm_oracle.py` (iii) | The total key is the posterior-integrated NB2 log mass under an independent `log1p` reference. |
| `test_total_model_factor_matches_an_independent_nb2_log_mass` | `p2_nb2_oracle.py` | The Pyro `pyro.factor` closed form is the NB2 log mass across a μ/φ grid. **Nothing previously executed this expression.** |
| `test_total_posterior_mean_responds_to_exposure_features_and_neighborhood` | sabotage (ii) | Predicted totals depend on exposure, features, and neighborhood. |
| `test_bayesian_and_independent_keys_are_the_same_conditional_object` | `p4_cross_model.py` | Model B and the Bayesian model emit identical key sets, both `sequential_joint`, both with a zero final cohort key; Model A declares `marginal` and its final key is emphatically non-zero. **This is the contract Gate 6 is built on.** |
| `test_diagnostic_policy_outcome_is_recorded_even_when_it_only_warns` / `..._records_a_clean_pass` | `p5_gating.py` | The policy outcome survives into metadata on both paths. |
| `test_calibration_retention_uses_a_likelihood_ratio_test` | `g41_calibration.py` | The recorded decision is the LRT and matches its own recorded statistic. |
| `test_noise_scale_calibration_gain_is_rejected` | `g41_power.py` | A 1e-5 per-child gain — above the retired 1e-6 tolerance, far below the chi-square threshold — is **rejected**. Fails under the old rule. |
| `test_large_calibration_gain_is_retained` | `g41_calibration.py` | A genuine gain is still retained, so the previous test is not vacuous. |

**Contract changes later gates inherit from Gate 5:**

- `IndependentTotalProbabilityConfig.calibration_improvement_tolerance` is
  **removed**; `calibration_significance_level` (default 0.05) replaces it.
  `configs/modeling.toml` updated. **This changes which temperature ships**:
  retention now requires `2·W·(raw − fitted) > chi2.ppf(0.95, 1)`.
- Model B's `calibration` metadata gains `retention_rule`,
  `likelihood_ratio_statistic`, `likelihood_ratio_critical_value`,
  `likelihood_ratio_p_value`, `calibration_significance_level`, and
  `weighted_child_count`.
- Each Bayesian stage's `diagnostics` gains `policy_passed`, `policy_failures`,
  `policy_thresholds`, and `active_profile`. **Gate 6 should select on
  `policy_passed` rather than re-deriving the policy** — this is what discharges
  plan §9.3's convergence-as-selection-constraint clause.
- `BayesianConditionalModel._predictive_draws` now takes `total_posterior` and
  `composition_posterior` keyword arguments.

### R6 — reconstructable state bundle and tested loader (plan §11)

`BayesianConditionalModel.to_state_bundle()` returns a JSON-safe description of
a fitted model — posteriors as nested lists, every configuration object as its
own `asdict` form, plus the neighborhood lookup, both stages' diagnostics, the
prior-predictive summary, the training hashes, and dependency versions now
including numpy and scipy. `from_state_bundle(bundle, *, train_df)` rebuilds a
model that predicts without re-running NUTS. Nothing is pickled.

Two design decisions worth recording, because they bound what the loader
promises:

- **The bundle contains no training data.** The two feature transformers hold
  fitted scikit-learn state (`SplineTransformer`, `OneHotEncoder`). Rather than
  forking that state into a second serialization format inside Gate 5 — Gate 2A
  owns it — the loader refits both transformers from a caller-supplied
  `train_df` whose `training_data_hash` and `training_schema_hash` must match
  the bundle. Transformer fitting is deterministic, so refitting on the verified
  rows reproduces the original preprocessing exactly. A test asserts the
  reloaded model reproduces every predictive payload element-wise.
- **The cost of that choice:** the bundle is not self-sufficient. Reloading in a
  fresh environment requires the original training partition. This is a genuine
  limitation, stated rather than hidden. Making the bundle standalone means
  giving `FittedFeatureTransformer` its own `to_dict`/`from_dict`, which is a
  Gate 2A change and belongs to whoever reopens that gate. The provenance check
  is a real benefit in the meantime: a wrong frame is rejected loudly instead of
  silently yielding a different model.

Config objects are rebuilt through `_restore_config`, which walks an already-
valid template instance to restore tuples and nested dataclasses (including
tuples *of* dataclasses, such as `FeatureSpec.categorical_feature_specs`) and
then calls each class's validating constructor, so a hand-edited bundle is
rejected by the same `__post_init__` rules as a hand-written config.

Five tests cover it: a full JSON round trip reproducing every predictive payload
with `_run_nuts` and `_run_prior_predictive` monkeypatched to raise if touched;
rejection of a mismatched training frame; rejection of an unknown
`bundle_format` and of another model class's bundle; and an assertion that no
training row or raw covariate value appears in the serialized bundle.

**Remaining MLflow gap after R6:** per-parameter posterior summaries (mean, sd,
quantiles) are still not recorded anywhere, and the `hyperparameters` payload
still carries both profiles and both diagnostic policies when only one of each
is active. Both are recommendations for Gate 7, not defects.

**Still open after Gate 5** (both belong to Gate 6, unchanged): §5.3 candidate
factories do not exist; the stale plan evidence paragraphs for Gates 2A/2B/2C,
to which the Gate 5 section's now-stale 392 count should be added.

---

## Gate 6 — Cross-Model CV, Selection, And Interpretation

Reviewed 2026-09-12. `src/age_group_prediction/experiment/` (10 modules, 1,515
lines) against plan §§9–10, the plan's Gate 6 section, and brief §6.
Tests: `tests/unit/test_experiment.py` (1,212 lines, 22 tests).

**Baseline reproduced exactly:** 407 passed, 1 warning, in 102.25s; the warning
carries `worst_rhat=1.086056330170683` and
`minimum_effective_sample_size=25.660557049567544`, matching the documented
values to the digit. `ruff check src/age_group_prediction/` clean.
`git status --porcelain` unchanged from session start. The instruction's §1
baseline of 392 is stale and is **not** reported as a finding.

### Verdict: **ACCEPT WITH CONDITIONS**

The statistical core is sound. Every REJECT trigger declared in the Phase 1 plan
was tested and **none fired**: no holdout ID reaches any output, ranking uses
`joint_predictive_nll` only, the capability gate excludes Model A from joint
scoring, the joint-NLL identity is exact, and results are deterministic,
candidate-order invariant and genuinely seed-sensitive. The conditions concern
**one shared-code defect that makes a claimed output unobtainable in a real
run**, **two plan clauses the implementation does not satisfy**, and **a test
suite that still exercises no real model**.

This is the first time in the repository's history that Gate 6 orchestration has
been run against all three real model families. Every finding below marked
"probe-confirmed" comes from that run.

#### Acceptance clauses (plan:1471–1473)

| # | Clause | Verdict |
|---|---|---|
| A1 | All candidates share folds | **MET.** Real five-candidate run; all candidates receive identical fold row sets, fold identities sorted by index. |
| A2 | Metric definitions are visible | **MET.** Each candidate descriptor in the freeze records all 23 metrics with `name`, `target`, `aggregation_level`, `required_capability`, `optimization_direction`. |
| A3 | Accepted **and rejected** feature blocks are recorded | **MET.** All five candidate descriptors — selected and rejected alike — carry `component_feature_specs`, `fit_feature_spec` and `importance.feature_blocks`. |
| A4 | No test metrics exist | **MET.** `freeze["test_metrics"] is None`; `run_cross_model_validation` has no `test_df`/`holdout_df` parameter; measured intersection of holdout IDs with predictions, importance payloads and the freeze is empty in all three. No module in `src/` or `tests/` references MLflow. |
| A5 | Likelihood comparability, calibration, selection logic | **MET with conditions.** Comparability and selection verified (G6-A, G6-B below); calibration evidence is recorded but is not a gate, and convergence is not a constraint at all (G6-2). |

#### Plan §9.3 / §10 clauses

| Clause | Verdict |
|---|---|
| §9.3 identical folds; joint NLL primary; RMSE/composition secondary | **Supported.** `sequential_joint_selection_policy` ranks `joint_predictive_nll` → `composition_log_loss` → `total_rmse`. |
| §9.3 fold values, means, standard deviations, cluster-bootstrap CIs | **Partly supported.** `aggregate_metrics_df` carries `mean`, `standard_deviation`, `fold_count`. The cluster-bootstrap CIs are unobtainable for two of three families — G6-1. |
| §9.3 calibration / finite / nonnegative / reconciliation are **hard diagnostics** | **Supported for finite, nonnegative and reconciliation** — all raise in `results.py:226-245`. **Not supported for calibration**, which is recorded (`calibration_df`) but never gates. |
| §9.3 **"Runtime and convergence are explicit selection constraints"** | **UNSUPPORTED.** Neither is implemented — G6-2. |
| §10.1 importance on held-out validation predictions | **Supported.** Verified with real models. |
| §10.1 "using NLL or deviance" | **Judgement call.** The importance metric is caller-supplied and unconstrained; nothing requires NLL or deviance. |
| §10.3 Model B total and probability importance separately | **Supported** via `PermutationImportanceSpec.component`. |
| §10.4 NB2 coefficient intervals and incidence-rate ratios | **UNSUPPORTED.** `grep -rn 'incidence_rate_ratio|coefficient_interval|irr'` over `src/` returns nothing. |
| §10.5 Bayesian posterior intervals and scenario curves separately by stage | **UNSUPPORTED.** Not produced by `experiment/`; consistent with Gate 5's recorded "per-parameter posterior summaries are still not recorded anywhere". |

### Findings, ranked

**G6-1 — `_subset_prediction` does not subset `parametric_distributions`, so
neighborhood-bootstrap intervals are unobtainable for Models A and B.
(High / blocking for real use. Probe-confirmed.)**

`evaluation.py:176-204` rebuilds a resampled `PredictionResult` and subsets
`total_mean`, `cohort_means`, `age_group_probabilities`,
`reconciliation_error`, draws, intervals and `pointwise_log_probabilities` — but
passes `parametric_distributions=prediction.parametric_distributions` through
**unsubset**. Both non-Bayesian models emit a fitted scalar dispersion
*broadcast to a per-building array of the original length*, so every replicate
whose cluster resample has a different row count fails
`results.py:86` ("NB2 dispersion must be scalar or one value per building").

Measured on a real 340-row training frame, 170 validation rows, fold 0:

| Candidate | Replicates failed | Cause |
|---|---|---|
| `A-nb2` (DirectCohortModel) | **21/25 = 84%** | `NB2 dispersion must be scalar or one value per building` |
| `B-nb2` (IndependentTotalProbabilityModel) | **21/25 = 84%** | same |
| `Bayes-reduced` | 0/25 | emits no `parametric_distributions` |

Over 300 replicates: 248 differently-sized (all raised), 52 same-sized.
`max_failed_fraction` is 0.5 in `configs/modeling.toml`, so
`neighborhood_cluster_bootstrap` raises `RuntimeError("Too many
neighborhood-bootstrap replicates failed")` and **aborts the whole experiment
before selection is ever reached**. Reproducing the Gate 6 run at all required
disabling the bootstrap.

This falsifies the Gate 6 evidence claim that the package "records … fixed-
prediction neighborhood-bootstrap intervals", and §9.3's requirement to include
cluster-bootstrap confidence intervals.

*Why no test caught it:* the only spy that emits parametric distributions,
`_ParametricSpyModel` (`tests/unit/test_experiment.py:1092-1114`), uses a
**scalar** `scale=1.0`, and its poisson variant carries no parameters at all.
Scalars broadcast at any row count, so the suite is structurally blind to the
array case that every real model produces.

*Scope note:* the defect is in `evaluation.py` (Gate 2C code), but Gate 6 is its
only real consumer and the only place it is reachable. Recorded here.

*Second, currently latent consequence:* when a resample coincidentally has the
same row count, the replicate is **accepted** with the original rows'
distribution parameters. Measured mismatch today is **0.000** across 52 such
replicates, because the dispersion array is constant (one unique value per
cohort: 0.01127437200037459, 0.02382735965298137, 0.5044339230775182). Any
future model emitting genuinely per-building dispersion or scale turns this into
a silent scoring error. Reported as latent, not live.

**G6-2 — Convergence is not a selection constraint, so a non-converged Bayesian
candidate is selected and frozen. (Medium-high. Probe-confirmed.)**

Plan §9.3 states "Runtime and convergence are explicit selection constraints",
and Gate 5's R1 added `policy_passed`, `policy_failures`, `policy_thresholds`
and `active_profile` to each stage's diagnostics specifically so Gate 6 could
gate on them (this findings file, "Gate 6 should select on `policy_passed`").

`grep -rniE 'policy_passed|r_hat|rhat|ess|divergen|converg'` over
`experiment/` returns **zero hits**. `model.get_metadata()` is stored verbatim
on `FoldRunEvidence.model_metadata` (`runner.py:181,190`) and never inspected;
the only metadata consumer is `_calibration_record` (`evidence.py:161-194`),
which reads `metadata["model"]["calibration"]` alone.

Measured on the real run: both Bayesian stages reported
`policy_passed=False` — total with
`worst_rhat=1.1606239230768491`, `minimum_effective_sample_size=11.537417785316615`
and `maximum_mean_accept_prob=0.9842466112002666` violations; composition with
`minimum_effective_sample_size=48.089310983021385` — and the candidate was
nevertheless **selected and written into the freeze** as
`BayesianConditionalModel: Bayes-reduced`. `is_fitted` is `True` throughout.

Runtime is the other half of the same clause and is equally unimplemented:
durations are recorded by `_operation_scope` but never read.

**G6-3 — A candidate that fails after fitting aborts the entire experiment, and
the error identifies neither the candidate nor the fold. (Medium-high.
Probe-confirmed.)**

`runner.py` contains **no `try`/`except`**; the only `try` in the package is
`partitions.py:125-130`. Probe: a `DirectCohortModel` subclass that fits
normally and raises in `_predict_model`, run alongside a healthy candidate.

```
run aborted with RuntimeError: candidate exploded during predict
  error names the failing candidate ('A-broken'): False
  error names the fold index                    : False
```

The healthy candidate's completed folds are discarded. This is reachable
without contrivance: Gate 4 recorded it as "a fold-killer" for Gate 6, Model B
raises past `bootstrap_max_failed_fraction`, the Bayesian model raises under
`action="error"`, and G6-1 above raises today. During this review the same
class of abort occurred three separate times from ordinary causes, each time
requiring the run to be re-driven manually to discover which candidate failed.

**G6-4 — No Gate 6 test drives a real model; the seeding mechanism and
importance magnitudes have no test power. (Medium-high. Probe-confirmed.)**

`tests/unit/test_experiment.py` imports only `models.base` (line 47). **0 of 22
tests** instantiate `DirectCohortModel`, `IndependentTotalProbabilityModel` or
`BayesianConditionalModel`; the three approach names appear only as `approach=`
string labels. `_SpyModel._fit_model` (line 73) discards features entirely and
`_predict_model` (line 77) returns `clip(eval_df["ses"] + offset)` with a fixed
`[0.2, 0.5, 0.3]` composition. `test_experiment.py` is also the **only** file in
the repository that imports `experiment/`, and `CandidateDefinition` is
constructed nowhere in `src/` — the package has no production caller.

This is the same failure brief §4 records from Gate 6's own acceptance review
("every Gate 6 test drove a spy model"), and the plan's Gate 6 section records a
blocking defect that shipped for exactly this reason.

Sabotage harness (§8's first technique), all 22 tests re-run per variant:

| Sabotage | Result |
|---|---|
| Selection freezes the **worst** candidate instead of the best | 3 failed — **detected** |
| Likelihood-comparability guard never fires | 1 failed — **detected** |
| Bayesian feature-freeze check never fires | 1 failed — **detected** |
| Every purpose-scoped child seed collapses to the constant 12345 | **22/22 passed — NOT detected** |
| Importance degradation forced to 0.0 for every block | **22/22 passed — NOT detected** |
| `DirectCohortModel._predict_model` raises on entry | 22/22 passed — never reached |

Each undetected sabotage was instrumented to prove it executed rather than
silently no-opping: the seed patch was invoked **233 times across 197 distinct
purposes**; the importance patch ran 13 times and overwrote genuine degradations
of `{-0.04, 0.0, 0.1}` with zero. So the entire purpose-scoped seed derivation —
the mechanism that decorrelates fit, predict, evaluate, bootstrap and
permutation randomness — can be replaced by one constant with no test noticing,
and importance magnitudes are never asserted at all.

Gate 6's suite is nonetheless **stronger than Gates 3–5 were**: it has genuine
power over selection ranking, the comparability guard and the feature freeze.

**G6-5 — `test_fold_coverage_reports_unvalidated_and_repeated_buildings` cannot
observe either condition on its own fixture. (Low-medium. Probe-confirmed.)**

`tests/unit/test_experiment.py:1007`. Forcing every `validation_fold_count` to 1
leaves all 22 tests passing. Instrumentation shows why: the observed counts on
the 12-row fixture are already `{1}` — no building is unvalidated (0) and none
is repeated (≥2), so the sabotage is indistinguishable from correct behaviour.
The test asserts a hand-recomputed count that is constant by construction. This
is the Gate 4 precedent in §8 (a test that passes under sabotage because its
fixture is degenerate), not a suite-wide blind spot.

**G6-6 — Selection has no tolerance band, and `rejection_reasons` records no
margin. (Low. Judgement call, probe-confirmed behaviour.)**

`selection.py:37-48` ranks lexicographically on raw floats with `candidate_id`
as the terminal tie-break. Probe: two candidates whose summed criterion differs
by 3e-12 (`3.0` vs `3.0000000000030003`) — the first is selected outright and
the second recorded as rejected with the boilerplate
"Ranked below 'cand-a' under the declared ordered training-only selection
criteria". A candidate losing by 3e-12 and one losing by 3 nats produce
byte-identical rejection text; the margin is recoverable only from the parallel
`criterion_values`. The plan acknowledges the missing tolerance band at
:1565-1568; the uninformative reason string is brief §6's fourth MLflow lead and
is confirmed.

**G6-7 — Every fitted model for every candidate and fold is retained, though
only selected candidates' models are used. (Low. Code quality.)**

`runner.py:194` populates `fitted_models[(candidate_id, fold_index)]` for all
candidates, but `_compute_selected_importance` (`importance.py:66-67`) consumes
only `{selection.selected_candidate_id}`. Memory grows as candidates × folds,
including Bayesian posterior arrays that Gate 5 measured at ~49 MB each at full
profile. On the six predeclared candidate families across five folds this
retains twenty-plus unused fitted models.

**G6-8 — Fold-scope and CV-scope bootstrap rows have different schemas.
(Low. Probe-confirmed as latent.)**

`_percentile_intervals` (`evaluation.py:207-231`) masks replicates on
`metric_name` and `target` only and emits no `aggregation_level` column, while
the `scope="cross_validation"` path in `aggregation.py:44-46` *does* group by it.
The two are stacked into one `bootstrap_intervals_df`, so fold rows carry a null
`aggregation_level` and CV rows populate it.

Brief §6 asks whether any shipped metric set actually collides. **It does not:**
enumerating the registry gives 12 distinct `(name, target)` keys and **none maps
to more than one `aggregation_level`**, because `name` and `aggregation_level`
are both `init=False` constants per metric class. Reported as a schema
inconsistency and a latent collision risk, **not** a live defect. The Gate 6
remediation claim that bootstrap evidence "now keys on `aggregation_level`" is
**supported for the CV scope** — `replicate_metrics_df` does carry the column via
`result.to_record()` (`metrics.py:110`, `evaluation.py:144-149`) — and
**unsupported for the fold scope**.

**G6-9 — `test_importance_rejects_forbidden_or_out_of_component_columns` has
dead code and does not test what it is named for. (Low. Code quality.)**

`tests/unit/test_experiment.py:479`. The `ValueError` is raised at
`CandidateDefinition(...)` construction on line 482, so the
`run_cross_model_validation` call on lines 498–509 inside the same
`pytest.raises` block is unreachable. The test also covers only *forbidden*
(target) columns; the "out-of-component" branch (`importance.py:46-51`) is never
exercised.

**G6-10 — `aggregation.py:57-61` takes `point.iloc[0]` on an unguarded mask.
(Low. Code quality.)**

The point-estimate lookup masks `aggregate_metrics` on whichever key columns
exist and takes the first row without asserting exactly one match, unlike
`selection._criterion_value` (`selection.py:159-164`), which raises when a
candidate does not provide exactly one aggregate. An ambiguous mask silently
picks a row.

### Findings raised during planning and **withdrawn**

- **The `None == None` hole in `_validate_bayesian_feature_freeze`.**
  `selection.py:188` compares `independent_specs.get(component) !=
  bayesian_specs.get(component)`, which would pass if neither candidate declared
  a component. **Unreachable:** `_validate_candidate_registry`
  (`runner.py:297-307`) already requires every non-`DirectCohortModel` candidate
  to declare exactly `{total_count, age_probability}`, and raises otherwise.
  Withdrawn.
- **"The freeze check compares declared rather than actual feature specs" as a
  documentation defect.** It does compare declarations, but the plan states this
  openly ("caller discipline at the estimator boundary", :1430-1440). A design
  gap and a legitimate judgement call, not an inaccuracy. Reclassified below.

### Statistical judgement calls (not defects)

1. **Gate 6 never selects across families.** `FrozenApproachSelection` is
   explicitly "within-approach" (`evidence.py:43`), and the run produces one
   winner per approach. Yet the joint scores *are* comparable — measured
   `joint_predictive_nll` of 3.546663 (`B-nb2`), 3.576361 (`B-poisson`) and
   3.463581 (`Bayes-reduced`), with key-by-key comparability proven in Gate 5.
   Deferring the cross-family choice to Gate 8 is defensible, but no document
   states that this is where it happens, and §9.3 reads as though a single
   primary criterion picks one model.
2. **The Bayesian feature freeze compares declarations, not fitted state.** A
   candidate could declare matching specs while the model fits something else;
   the guard would not notice. The plan concedes this. Enforcing it on fitted
   metadata would close it.
3. **`assumes_independent_folds=True`** on the CV-scope band remains honest in
   the narrow sense that the flag is published, but folds now rotate and Gate 1
   recorded that across-fold spread is not a standard error. The flag documents
   an assumption known to be false rather than avoiding it.
4. **Importance pools over correlated folds.** `importance_summary_df` averages
   across folds that share training rows; the summary's spread is not a standard
   error, same caveat as Gate 1's.

### Confirmed correct (probe-verified, with real models)

These re-establish the Gate 6 section's closing paragraph, whose evidence had
been lost with the scratch probes that produced it:

- **Leakage containment.** Holdout-ID intersection with `predictions_df`,
  `importance_df.validation_building_ids` and the freeze is empty in all three.
- **`test_metrics`** is `None` in the freeze; nothing fills it.
- **The joint-NLL identity is exact.** `joint_predictive_nll` versus an
  independent recomputation as the negative mean of summed sequential keys:
  `3.6041761338475227` against `3.6041761338475227`, absolute difference
  **0.000e+00**. The plan claims "ten decimal places"; it is exact.
- **The capability gate excludes Model A from joint scoring.**
  `joint_predictive_nll` rows exist for `B-nb2`, `B-poisson` and
  `Bayes-reduced` only, and never for either Model A candidate — by capability,
  not by a special case in `selection.py`.
- **Ranking uses `joint_predictive_nll` only** for the conditional approaches,
  and a cohorts-only sum excluding the total for `DirectCohortModel`.
- **The per-target trap is real and was avoided.** On this frame the mean
  `n_highschool` key is **0.9517434** for `A-nb2` (marginal, informative) against
  **-7.18e-18** for `B-nb2` and **1.18e-17** for `Bayes-reduced` (conditional,
  ~0 by construction). A per-target table would hand the conditional models a
  0.95-nat win for no modeling reason. Gate 5 measured 2.51 nats on its frame;
  the magnitude is frame-dependent, the phenomenon is not.
- **Key order differs across families**, as Gate 5 recorded: Model B emits
  `['total', 'n_kindergarten', 'n_elementary', 'n_highschool']`.
- **Determinism, order-invariance and seed-sensitivity all hold.** Two runs at
  `master_seed=23` are identical; reversing candidate order is identical;
  `master_seed=1009` differs (largest deltas 0.058473 on
  `mean_poisson_deviance`/`n_elementary`, -0.044382 on `r2`). The freeze records
  `master_seed=23`, `master_seed_source=explicit_argument`.
- **Importance runs only after selection**, only on selected candidates
  (6 rows = 3 selected × 2 folds), through each fold's fitted model, on that
  fold's validation rows, with real degradations of 0.31–0.82 from permuting
  `ses`.
- **`SelectionFreeze.to_dict()` is JSON-serializable.**

### MLflow readiness (design review only; no MLflow code written)

Brief §6's four leads, all four checked:

1. `_percentile_intervals` masking — **real but latent**, see G6-8. No shipped
   metric set collides. The fold/CV schema split should still be fixed before a
   tracking layer flattens both scopes into one table.
2. `bootstrap_intervals_df` stacking two grains with
   `assumes_independent_folds` — **flag is published and honest**, judgement
   call 3 above. Not a defect.
3. `SelectionFreeze.to_dict()["test_metrics"] = None` — **confirmed clean**.
   Hardcoded at `evidence.py:100`; no other occurrence in the package; still
   `None` after a full real run.
4. `rejection_reasons` boilerplate — **confirmed inadequate** for a tracked
   comparison, see G6-6.

Additional readiness notes:

- **Params/metrics/artifacts separate cleanly.** `aggregate_metrics_df` is tidy
  with `(candidate_id, approach, metric_name, target, aggregation_level, mean,
  standard_deviation, fold_count)` — directly loggable. Candidate descriptors are
  JSON-safe params.
- **Comparability across models holds at the schema level**: the metric key
  `(metric_name, target, aggregation_level)` is stable across families, and
  family-specific meaning lives in `interpretation` metadata rather than a
  renamed metric.
- **No Gate 6 equivalent of `diagnostics["search_space"]`.** Gates 3 and 4
  record the search space beside chosen params; Gate 6 records neither a
  candidate-set provenance nor the policy criteria as a loggable artifact
  (`SelectionPolicy` is not in the freeze — only its *outcome* is). A tracked run
  cannot recover which criteria were declared without the code.
- **Run identity is still absent**: no timestamp, run id, git SHA or experiment
  name anywhere in `src/`, by design. A tracking layer must inject them.
- **`validation_building_ids` is a full JSON ID list repeated on every
  importance row** — 6 rows here, but candidates × folds × blocks × repeats in a
  real comparison. Belongs in an artifact keyed by fold, not a column.
- **G6-1 blocks interval logging outright** for two of three families.

### Documentation accuracy

| Claim | Verdict |
|---|---|
| "defines an explicit typed candidate registry" | Supported. |
| "validates the outer-training frame and every fixed fold against the lockbox manifest before constructing a model" | Supported; probe-confirmed (factory never called on a rejected frame). |
| "records … fixed-prediction neighborhood-bootstrap intervals" | **Overstated.** Unobtainable for Models A and B under the shipped config — G6-1. |
| "Selection is deterministic and produces one frozen candidate per public approach with rejected alternatives and reasons" | Supported, with G6-6 on reason quality. |
| "Bayesian selection is rejected unless its total and probability feature forms equal the selected independent model forms" | Supported for *declared* forms; judgement call 2. |
| "importance runs only after selection, only through each fold's fitted public model, only on that fold's validation rows" | Supported; probe-confirmed. |
| "The runner has no test-frame argument, records `test_metrics = null` … and has no MLflow dependency" | Supported; all three verified. |
| "passes **342 tests**" (:1500) | **Stale.** The suite is 407. |
| "Confirmed correct under adversarial probing with all three real models" (:1548-1556) | **True but unreproducible when written** — no repository test exercised a real model. Now re-established by this review's probes, which Phase 3 should promote to tests. |
| §9.3 "Runtime and convergence are explicit selection constraints" | **Unsupported** — G6-2. |
| §10.4 / §10.5 coefficient intervals, IRR, per-stage posterior intervals | **Unsupported** — not produced. |

### Carried-forward items

**§4(b) — candidate factories.** Confirmed absent on the strongest evidence:
`CandidateDefinition` is constructed nowhere in `src/`, and `experiment/` has no
production caller. Note the runner already guarantees stable order
(`sorted(candidate_by_id)`, `sorted(policies)`) and full approach coverage
(`_validate_candidate_registry` requires eligible candidates to exactly cover
the required approaches), so the *ordering* half of §5.3 is already delivered.
The unmet half is that nothing guarantees the six predeclared candidate families
are what is compared.

**Recommendation: build the factory.** Assembling the real candidate set by hand
for this review took five corrections to get right — a metric set over-broad for
Model A's declared capabilities, `include_pointwise_log_probabilities` left off
for the conditional models, an under-sized frame, a chains<2 profile, and the
bootstrap failure of G6-1. Every one of those is a caller-discipline error that a
factory would make once and encode. The factory is also the natural home for the
real-model integration tests G6-4 requires, and would give `experiment/` its
first production caller. Decision deferred to the user's approval of Phase 3.

**§4(c) — stale documentation.** To be corrected once, in Phase 3, now including
Gate 6's own count:

| File:line | Wrong | Correct |
|---|---|---|
| `GATE_VALIDATION_FINDINGS.md:25` | "392 → 402" | 407 |
| `MODELING_REBUILD_PLAN.md:1005` | "ordered candidate factories" | none exist |
| `MODELING_REBUILD_PLAN.md:987-988` | "validated candidate factories" (requirement) | depends on §4(b) |
| `MODELING_REBUILD_PLAN.md:1009` | "20 focused Gate 2A tests" | was 19, now 24 |
| `MODELING_REBUILD_PLAN.md:1042` | "24 focused contract tests" (2B) | re-measure |
| `MODELING_REBUILD_PLAN.md:1107-1108` | "50+ focused metric/evaluation tests" (2C) | re-measure |
| `MODELING_REBUILD_PLAN.md:1500` | "passes 342 tests" | 407 |
| `GATE_5_6_VALIDATION_INSTRUCTION.md` | §1 baseline 392; §3 "9 modules, ~1,350 lines"; §4(a) open; §5 contract list predates Gate 5 | 407; 10 modules, 1,515 lines; G4-1 closed; add Gate 5's four contracts |

### Recommended remediation — proposed, not applied

| # | Fix | Severity |
|---|---|---|
| **R1** | Subset `parametric_distributions` in `_subset_prediction` (`evaluation.py:176-204`), indexing array-valued `dispersion`/`scale` by `row_indices` and passing scalars through. Add a test with an **array-valued** dispersion that fails before the fix. Closes G6-1. | High |
| **R2** | Make convergence an explicit selection constraint: read `policy_passed` from each stage's diagnostics and either exclude a failing candidate or record an explicit override in the freeze. Discharges §9.3 and consumes Gate 5's R1. Decide whether runtime joins it or §9.3 is amended. | Medium-high |
| **R3** | Define candidate-failure behaviour in the runner: wrap the per-(candidate, fold) body so a failure is recorded as evidence and the remaining candidates complete, or — if aborting is intended — re-raise with the candidate id and fold index attached. Closes G6-3. | Medium-high |
| **R4** | Add real-model integration tests: one run covering all three families on a small frame at the reduced Bayesian profile, asserting leakage containment, the joint-NLL identity, the capability gate, seed-sensitivity, and that importance is nonzero. Promote this review's probes. Each must be shown to fail under the corresponding sabotage. Closes G6-4. | Medium-high |
| **R5** | Give the seed mechanism and importance magnitudes test power — assert distinct purposes yield distinct seeds, and that a block that matters has nonzero degradation. Both currently survive a constant. | Medium |
| **R6** | Fix the degenerate fold-coverage fixture so an unvalidated and a repeated building both occur (G6-5); remove the dead code in `test_experiment.py:479` and add the out-of-component case (G6-9). | Low-medium |
| **R7** | Emit `aggregation_level` from `_percentile_intervals` so both bootstrap scopes share one schema (G6-8); guard the `point.iloc[0]` lookup in `aggregation.py:57-61` (G6-10). | Low |
| **R8** | Record the losing criterion and margin in `rejection_reasons`, or state in the policy that `criterion_values` is the rationale of record (G6-6). | Low |
| **R9** | Drop non-selected fitted models once selection is frozen (G6-7). | Low |
| **R10** | §4(b) decision and the §4(c) documentation pass, including `GATE_5_6_VALIDATION_INSTRUCTION.md`. | — |

### Cross-references

- **Gate 2C / `evaluation.py`** — G6-1 is a defect in shared evaluation code
  reachable only through Gate 6. If Gate 2C is ever re-opened, its
  `_subset_prediction` tests need an array-valued parametric parameter.
- **Gate 5** — G6-2 consumes R1's `policy_passed`; this review confirms Gate 5's
  prediction that Gate 6 "does not do it yet". Gate 5's proven key-by-key
  comparability was relied on and not re-derived.
- **Gate 4** — its "a candidate that fits and then dies in `predict` needs a
  defined behaviour in the runner" cross-reference is confirmed as G6-3, and its
  per-target-NLL warning is confirmed with fresh magnitudes.
- **Gate 3** — its `predict`-cost warning (600 LightGBM refits at defaults, paid
  per fold per candidate) is real; this review ran at
  `bootstrap_replicates=5-8` and `n_trials=3` to stay tractable.
- **Gate 1** — `_fold_coverage` remains the right mitigation for F1, but G6-5
  shows its test cannot observe the condition it reports.
- **Gates 7–9** — confirmed not built: no MLflow reference in `src/` or
  `tests/`, no final-test prediction, `test_metrics` null.

### Remediation applied — Gate 6

R1–R10 applied. Each fix is verified below **against the original probe
numbers**, not merely against a green suite.

**Suite: 425 passed** (was 407), ruff clean on `src/age_group_prediction/` and
the three test files touched. The single documented Bayesian warning is
unchanged to the digit — `worst_rhat=1.086056330170683`,
`minimum_effective_sample_size=25.660557049567544` — because the new real-model
test suppresses only its own short-profile warnings, which are asserted rather
than ignored. `git status --porcelain` matches session start apart from the
approved files.

Test progression: 407 → 420 (R1, R4, R5, R6) → 425 (R10's factory tests).

| # | Fix | Verification against the original numbers |
|---|---|---|
| **R1** | `_subset_prediction` now subsets `parametric_distributions`, indexing array-valued `dispersion`/`scale` and re-broadcasting scalars (`evaluation.py:181-212`) | Replicate failure rate **84% → 0%** for both `A-nb2` and `B-nb2` (was 21/25 each); Bayesian unchanged at 0/25. Three new tests in `test_evaluation.py`, all three demonstrated failing with the fix reverted |
| **R2** | Convergence is an explicit selection constraint: `_convergence_failures` reads each stage's `policy_passed`, and `_select_candidates` excludes failing candidates before ranking, recording their failures as the rejection reason. `run_cross_model_validation(..., require_convergence=True)` by default | The probe candidate that was **selected and frozen** with `policy_passed=False` on both stages (R-hat 1.1606239230768491, ESS 11.537417785316615) is now excluded, and an approach whose candidates all fail raises naming them. Plan §9.3 discharged |
| **R3** | Per-operation `_fold_failure_context` names the candidate, fold and operation on any failure | Probe message went from `RuntimeError: candidate exploded during predict` (candidate named: **False**, fold named: **False**) to `Candidate 'A-broken' failed during predict on fold 0: ...` (**True**, **True**) |
| **R4** | `tests/validation/test_experiment_real_models.py` — 7 tests running all three real families through the runner, reduced Bayesian profile, 24s | First real-model Gate 6 tests in the repository. They pin the leakage containment, the capability gate (`joint_predictive_nll` present for the two conditional candidates and never for Model A), the joint-score identity to 1e-12, the per-target trap (conditional last-cohort keys < 1e-9 against Model A's > 0.1), and nonzero importance |
| **R5** | Two tests give the seed mechanism and importance magnitudes power | Constant-seed sabotage **22/22 passed → detected**; constant-importance sabotage **22/22 passed → detected** |
| **R6** | Fold-coverage fixture gains a singleton neighborhood; the importance-guard test split in two | Fold-coverage sabotage **22/22 passed → detected**. Coverage counts are now `{0, 1}` rather than the constant `{1}` that made the assertion vacuous |
| **R7** | `_percentile_intervals` masks on and emits `aggregation_level`; `aggregation.py` raises on an ambiguous point-estimate mask instead of `iloc[0]` | Both bootstrap scopes now carry the same key columns; no shipped metric set collides (12 distinct `(name, target)` keys, none spanning two levels) |
| **R8** | Rejection reasons name the deciding criterion and margin (`_margin_summary`) | The 3e-12 near-tie is now recorded as `independent_cohort_joint_nll 3 versus 3.0000000000030003 (margin 3e-12, minimize)` instead of text identical to a 3-nat loss |
| **R9** | Non-selected fitted models are released once the freeze is computed | Retained models drop from candidates × folds to selected × folds — on the five-candidate probe, from 10 to 6 |
| **R10** | `experiment/candidates.py` enumerates §5.3's predeclared specs; the eight stale documentation claims corrected in one pass | Five factory tests; documentation table below |

**Full sabotage harness, before and after.** Every variant now fails at least
one test:

| Sabotage | Before | After |
|---|---|---|
| Selection freezes the worst candidate | 3 failed — detected | **16 failed** |
| Likelihood-comparability guard off | 1 failed — detected | 1 failed |
| Bayesian feature-freeze off | 1 failed — detected | 1 failed |
| Every child seed collapses to a constant | **22/22 passed** | **detected** — `test_purpose_scoped_seeds_are_distinct_per_candidate_fold_and_operation` |
| Importance degradation forced to zero | **22/22 passed** | **detected** — `test_importance_degradation_is_nonzero_for_a_block_that_matters` |
| Fold coverage forced to a constant | **22/22 passed** | **detected** — `test_fold_coverage_reports_unvalidated_and_repeated_buildings` |

#### R10 — the §4(b) decision, and what it revealed

**The factory was built rather than §5.3 amended.** Reading §5.3 closely during
remediation corrected the premise the finding rested on: the "ordered set of
specs" it requires is an ordered set of **`FeatureSpec`s** — the six predeclared
families are SES-form and interaction variants — not an ordered set of model
classes. That is a narrower and clearly in-scope object, which settled the
decision.

`enumerate_feature_specs(component, ...)` returns the three SES forms followed
by the component's declared interactions, then §5.3's opt-in daycare saturation
curve. The component split is load-bearing and previously existed only as a
`FeatureSpec.__post_init__` rejection: `room_share_x_household_size` is
total-count only and `room_share_x_median_age` is age-probability only, so the
enumeration never offers the pairing the vocabulary forbids. Trees get the SES
forms only, because a gradient-boosted model represents interactions itself.

Documentation corrected in the same pass:

| File | Was | Now |
|---|---|---|
| `GATE_VALIDATION_FINDINGS.md` header | "392 → 402"; Gate 6 pending | 420; Gate 6 validated; items 2 and 3 closed |
| `MODELING_REBUILD_PLAN.md:1005` | "ordered candidate factories" (false) | true, with the module named |
| `MODELING_REBUILD_PLAN.md:1009` | "20 focused Gate 2A tests" | **24** (measured) |
| `MODELING_REBUILD_PLAN.md:1042` | "24 focused contract tests" | **34** (measured) |
| `MODELING_REBUILD_PLAN.md:1042` | "`_record_duration` context manager" | `_operation_scope` (renamed in Gate 2B/2C) |
| `MODELING_REBUILD_PLAN.md:1107` | "50+ … (run the fast-suite command for the current count)" | **54** (measured) |
| `MODELING_REBUILD_PLAN.md:1500` | "passes 342 tests" | **420** |
| `GATE_5_6_VALIDATION_INSTRUCTION.md` | §1 baseline 392, Gates 5–6 open; §3 "9 modules, ~1,350 lines"; §4(a)(b)(c) open; §5 predates Gate 5 | 420 and both gates done; 11 modules ~1,660 lines; all three carried items marked closed; Gate 5's four contracts added |

Plan §5.3's *requirement* sentence ("validated candidate factories", :987-988)
was left untouched: it is now satisfied rather than stale.

#### Findings withdrawn or corrected during remediation

Both were the reviewer's own, and the record is better for saying so.

- **G6-9's dead-code claim is WITHDRAWN.** It asserted that the
  `run_cross_model_validation` call inside
  `test_importance_rejects_forbidden_or_out_of_component_columns` was
  unreachable because the `ValueError` fired at `CandidateDefinition`
  construction. That is wrong: `CandidateDefinition.__post_init__` does **not**
  validate importance specs — `_validate_candidate_registry` does, during the
  run (`runner.py:305`). The call was reachable and necessary, and rewriting the
  test without it made both cases fail. Only the second half of the finding
  stood: the test never exercised the out-of-component branch despite its name.
  It is now two tests, both driving a real run.
- **The MLflow note that `SelectionPolicy` is absent from the freeze is
  CORRECTED.** `SelectionFreeze.to_dict()` does emit `"policy":
  asdict(selection.policy)` (`evidence.py:94`), so a tracked run *can* recover
  the declared criteria. The surviving gap is narrower: there is no Gate 6
  equivalent of `diagnostics["search_space"]` recording the *candidate set's*
  provenance, which `experiment/candidates.py` now makes expressible.

#### Two measurements worth carrying forward

- **The spy fixture's importance was identically zero by construction, not
  merely unasserted.** The spy predicts 0.3–0.5 against observations of 2–4, so
  every residual carries the same sign and mean absolute error collapses to
  `mean(observed) - mean(predicted)` — invariant to any permutation. No
  permutation importance over that fixture could ever be nonzero, whatever the
  code did. The replacement test uses squared error and an offset that places
  predictions among the observations, and was confirmed to fail under the
  constant-importance sabotage. This is the Gate 4 precedent (§8, "verify that
  your own new tests have power") recurring in a new form: the first version of
  the test was written with MAE and passed while measuring nothing.
- **A 30-sample Bayesian profile reported `minimum_effective_sample_size =
  -3554.19`.** Pyro 1.9.1's classic autocorrelation ESS is not constrained to be
  positive and degenerates badly at very short chains. It fails the threshold, so
  the convergence constraint behaves correctly here, but a negative ESS is not a
  meaningful quantity to record or log. **Cross-reference for Gate 5 and Gate
  7:** consider clamping at zero or switching to rank-normalized bulk/tail ESS,
  which the Gate 5 section already notes is not what is computed.

#### Remaining Gate 6 conditions, not addressed here

Recorded so they are carried rather than lost:

1. **Runtime is still not a selection constraint.** §9.3 names runtime alongside
   convergence; durations are recorded by `_operation_scope` and never read.
   Convergence was implemented because Gate 5 built the evidence for it;
   runtime needs a declared budget first, which no document states.
2. **Candidate failure still aborts the run.** R3 made the failure legible; it
   did not contain it. Containing it means deciding what selection compares when
   one candidate has no evidence, which is a contract change rather than a fix.
3. **§10.4 and §10.5 remain unimplemented** — NB2 coefficient intervals,
   incidence-rate ratios, and per-stage Bayesian posterior intervals are
   produced nowhere.
4. **The Bayesian feature freeze still compares declarations, not fitted
   state**, as the plan concedes.
5. **Gate 6 still never selects across families.** One winner per approach is
   frozen; the cross-family choice is deferred to Gate 8, which no document
   states explicitly.

---

## Gate 7 — MLflow Tracking

Gate 7 was implemented in phases, each stopped for user approval after an
independent review subagent checked it; it has not had a separate validation
pass like Gates 1-6. The design, decisions and evidence are in the plan's Gate 7
"Implementation record"; this section records the acceptance mapping, what the
reviews found, and what remains.

After acceptance, the adapter `tracking.py` was split without behavior change
into the `tracking/` package (plan, Gate 7 follow-up note); the test names cited
below are unchanged.

### Verdict: **ACCEPT WITH CONDITIONS** — conditions discharged

The independent acceptance review found the implementation satisfies §11 and
every clause of the Gate 7 acceptance line, with no blocker and no code defect.
Its conditions concerned test power and records: two tests could not detect
the failure they claimed to guard, and one residual misstated what was logged.
All conditions were discharged, as listed under "Findings fixed" below.

#### Acceptance clauses (plan, Gate 7)

| Clause | Evidence |
|---|---|
| Tracking-on/off results are equivalent | `experiment/` and `models/` import no MLflow (`test_experiment_and_model_code_never_import_mlflow`, AST scan); capture on and off give identical frames, selections, freeze and fold evidence (`test_default_run_captures_nothing_and_capture_changes_no_result`), including for real models (`test_real_capture_changes_no_comparison_result`); logging leaves a deep snapshot of every result field unchanged (`test_logging_leaves_every_result_field_unchanged`) |
| Required runs and tags exist | One parent plus one nested child per candidate, three `DirectCohortModel` family siblings, identity and context tags on every run (`test_one_parent_and_one_nested_child_per_candidate`, `test_every_run_carries_identity_and_context_tags`, `test_real_comparison_logs_one_complete_run_per_family`) |
| Metrics and artifacts are logged | Parent and child artifact trees and file schemas, metric keys and fold steps (`test_parent_logs_params_training_input_and_comparison_artifacts`, `test_candidate_artifacts_hold_bundles_checks_tuning_and_slices`, `test_metrics_are_keyed_by_metric_target_level_and_stepped_by_fold`) |
| Each loss-family run records independent train/CV-only tuning evidence | Per-fold Optuna trial tables and `tuning/{component}/best_value` in each family's own run, with trial values and best values matched to that run's own metadata and distinct across families (`test_each_family_run_logs_its_own_tuning_evidence`, `test_each_loss_family_records_its_own_train_only_tuning`). That the studies use training folds only rests on the earlier gates' runner and model tests |
| Cross-family metric comparability is explicit | `metric_comparability.json`, derived from the family sets and capabilities that selection enforces (`test_metric_comparability_is_derived_from_the_selection_rules`, `test_real_metric_comparability_separates_densities_from_masses`) |
| Interrupted runs cannot appear complete | `FAILED`/`KILLED` with `evidence_complete=false` for a runner failure, an interrupt, a failure midway through children, and a failure after logging had finished, which overwrites `true` (`test_a_comparison_that_fails_inside_the_block_is_recorded`, `test_a_failure_midway_through_children_never_looks_complete`, `test_an_interrupt_after_logging_finished_still_reads_incomplete`, `test_one_comparison_run_accepts_one_result`) |
| Frequentist and Pyro artifacts pass reload/predict smoke tests | Every fold is reload-checked inside the runner, and every bundle downloaded from MLflow reproduces the runner's predictions exactly for all real families (`test_every_stored_bundle_reloads_to_the_runner_predictions`) |

### Findings fixed during Gate 7

- **Phase 2 review, blocker:** child runs were started without the parent's
  experiment ID, so they landed in MLflow's `Default` experiment with
  artifacts in `./mlruns`. The Phase 2 boundary script hid it by setting
  `MLFLOW_EXPERIMENT_NAME`; the script and tests now run with it unset.
- **Phase 2 review, should-fix:** `file:` URIs were not percent-decoded, so an
  artifact path containing a space was refused on the second run; a run
  failing after logging finished still read `evidence_complete=true`; child
  runs lacked the caller's context tags while MLflow added Git tags inferred
  from HEAD; the reload check compared only means, so a lost NB2 dispersion
  passed; one parent accepted a second result.
- **Phase 2 review, nits taken:** `objective_family` now reports the LightGBM
  objective actually used; MLflow's dataset warnings are filtered by module
  only; param flattening keeps empty mappings and refuses name collisions; a
  missing selection no longer raises while tagging.
- **Phase 3:** restoring the global tracking URI through
  `mlflow.set_tracking_uri` rewrote `MLFLOW_TRACKING_URI` and pinned later
  comparisons to the first store
  (`test_a_comparison_leaves_the_tracking_uri_as_the_environment_says`).
- **Acceptance review, conditions:**
  - the tuning tests compared only trial numbers, identical across families,
    so evidence logged into the wrong family's run would have passed; they now
    compare values against each run's own metadata, with distinct spy values;
  - no test proved a failure after logging overwrites `evidence_complete=true`
    (a mutant survived); two tests now do;
  - residual 5 wrongly said the freeze logs `source_table_hash`; a split
    summary is now logged in provenance and as `split.*` params;
  - the comparability record omitted likelihood scope and gave the Bayesian
    candidate no measure kind without saying why; it now records pointwise
    scopes, the no-cross-approach `predictive_nll` rule, and the basis of
    measure kinds;
  - the `family` tag trusted the caller's label; a family that disagrees with
    the fitted model is now refused before any run starts
    (`test_a_candidate_whose_fitted_family_disagrees_is_refused`);
  - docstring coverage of files Gate 7 touched was incomplete; 49 were added.
- **Acceptance review, nits taken:** runner seeds logged per fold
  (`seeds.json`); the real reload test also compares probabilities; a slow
  real-model capture on/off test; README completion wording.

### Residuals, carried forward

1. **No per-candidate failure containment.** A failed comparison is
   all-or-nothing, recorded on the parent as `FAILED` or `KILLED` (settled
   decision).
2. **Runtime is logged, not constrained.** `duration/fit_seconds` and
   `duration/predict_seconds` are logged per fold; no runtime budget is
   declared. `duration/evaluate_seconds` never appears, because the runner
   scores predictions directly rather than calling `model.evaluate`.
3. **§10.4 and §10.5 are still not built**, so they are not logged.
4. **ESS is clamped, not rank-normalized.** Negative classic ESS charts as 0
   with `ess_valid=0`; bulk and tail ESS are not computed.
5. **Section 11 items not logged:** a simulator-configuration hash (the
   modeling table carries none; the split summary logs the manifest's
   `source_table_hash` of the table the split was drawn from), plots, and
   posterior-predictive artifacts beyond the stored posterior and point
   predictions.
6. **Fold configuration beyond the fold count** is not available to the
   runner, which receives folds rather than a `FoldConfig`; fold identities and
   fingerprints are logged instead.
7. **The Bayesian child has no `family` tag**, because no candidate
   configuration declares one for its fixed NB2 + Dirichlet-multinomial
   likelihood.
8. **Bundles in the artifact store contain fitted summary statistics** (see
   `state_bundle.py`); store them with the same care as the training data.
9. **Test-lock status and source revision are asserted by the caller**, not
   verified: `TrackingContext` records what it is given, because the package
   never runs Git and has no lockbox state to check against.
10. **Predictive measure kinds come from declared parametric families only.** A
    candidate scored through pointwise log probabilities (Model B's joint
    score, the Bayesian model) may show no kind; its pointwise scope is
    recorded instead, and the record states this basis.

## Gate 8 — Cross-Family Selection And The Final Lockbox

Gate 8 was implemented in five phases (canonical registry and manifest
persistence; cross-family freeze; full refit and lockbox evaluation; final
MLflow tracking and models-from-code; the thin marimo client), each stopped
for user approval after an independent review subagent checked it; like
Gate 7, it has no separate validation pass. The design, decisions and
evidence are in the plan's Gate 8 "Implementation record"; this section
records what the phase reviews found and the canonical run's result.

### Findings fixed during Gate 8

- **Phase 4 review (final tracking and models-from-code):**
  - `tracking/final.py::_log_final_model`'s reload-equality check compared a
    freshly loaded pyfunc's prediction against `evaluation.predictions_df`,
    which had been computed with a distinct, purpose-scoped seed
    (`final/{candidate_id}/predict`) rather than the seedless call the
    pyfunc `predict` API actually uses. This coincidentally passed only
    because the project's split strategy never hands the Bayesian model an
    unseen-neighborhood holdout row (the one case where a point mean itself
    draws a random effect); it would have diverged, not silently passed,
    had that ever happened. Fixed by comparing two predictions computed the
    same way (no explicit seed on either side), via a new shared
    `pyfunc_model.prediction_to_frame` helper so the wrapper's own
    `predict` and the reload check cannot drift apart — mirroring the
    identically-seeded comparison `experiment.artifacts.capture_fold_artifact`
    already uses for the same reason.
  - The in-process reload/equality check does not itself prove `code_paths`
    makes the model loadable from a fresh interpreter (the already-imported
    package is reused from `sys.modules` rather than the code copied into
    the artifact); this limitation is now stated explicitly in
    `_log_final_model`'s docstring rather than left implied by the passing
    test.
- **Phase 5 review (thin marimo client):**
  - `notebooks/02_model_fitting.py`'s pre-split modeling-table preview cell
    displayed `.head(5)` of the full, unpartitioned table — which still
    carries target columns — before the outer split (and therefore the
    lockbox concept) existed. At this project's ~20% holdout fraction, a
    plain setup run with no button clicked had roughly a two-in-three
    chance of showing a real target value for a building the very next
    cell assigns to the one-time holdout. Fixed by withholding target
    columns from that preview, with a new characterization test
    (`test_notebook_never_previews_the_full_modeling_table_with_targets`)
    guarding this specific class of leak, which the existing
    `split.test_df`-only check could not have caught.
  - No other finding: no direct model calls, correct AND-gating on both run
    buttons plus the confirmation checkbox, no reactive-rerun bypass, correct
    package-API argument passing, correct MLflow tag/query syntax in the
    final display cell, and no vacuous test coverage.

### Canonical run

Authorized and executed once, after both phases and their reviews passed.
The persisted lockbox manifest (`artifacts/lockbox/split_manifest.json`,
1,529 rows / 150 neighborhoods / 1,222 training / 307 holdout buildings) was
replayed, never re-split. `experiment.select_cross_family_winner` chose
`bayesian-reduced` (`BayesianConditionalModel`) on `composition_log_loss`
with no tie-break needed; `direct-poisson` (`DirectCohortModel`) and
`independent-nb2` (`IndependentTotalProbabilityModel`) are logged as
predeclared comparators, not rejected candidates. Verified directly against
the MLflow store (not only from script output):

- all three full-training refits completed, and the Bayesian refit's strict
  (`action="error"`) full-profile diagnostic policy passed;
- all three models predicted the same 307 ordered holdout building IDs;
- all three reloaded as models-from-code pyfuncs and reproduced their point
  predictions exactly;
- the final MLflow parent (`run_role=final_evaluation`, linked to the Gate
  6/7 comparison parent by `source_cv_run_id`, `test_lock_status=opened`
  only after the pretest freeze was logged) and every one of its three
  children finished `FINISHED` with `evidence_complete=true`.

### Residuals, carried forward

1. **Bootstrap replicate counts and diagnostic thresholds are the values in
   `configs/modeling.toml`**, not independently re-derived for the canonical
   population; they were carried unchanged from the Gate 6/7 configuration.
2. **Runtime and cost of the full-profile Bayesian refit are not budgeted or
   compared against the reduced-profile CV runs** — Gate 7 residual 2 applies
   here too; the final refit's duration is recorded in its metadata but not
   constrained.
3. **The comparator families' test metrics are descriptive only**, as
   designed (plan section 11): nothing in the codebase can feed them back
   into a selection, but no automated check re-verifies that on every run
   beyond the existing unit tests for `evaluate_frozen_models_on_lockbox`
   and `log_final_evaluation_result`.

### Gate 8 — Independent validation

Independent statistical acceptance and implementation verification per
`GATE_8_SESSION_HANDOFF.md` §12, run on 2026-09-14 (Opus 5) under
`GATE_8_VALIDATION_HANDOFF.md`. Phase 2 (validate and report) only. No fix
applied; this subsection is the only repository change.

**Method and safety.** The canonical lockbox was never re-run and nothing was
re-split. `mlflow.db`, `mlartifacts/` and `artifacts/lockbox/` were read only
(`sqlite3 -readonly` and a scratch copy). SHA-256 over all 338 evidence files
was identical before and after the full suite. The persisted manifest was
replayed read-only once, to recompute logged metrics. Nothing derived from it
touched a selection, configuration, freeze or model. Sabotage ran on a scratch
copy of `src/`/`tests/`; a canary proved the copy's code was the code under
test. Probes used `bundle_spy` and temporary SQLite stores. Worktree status at
start and end: 11 modified, 41 untracked (the Phase 1 plan's "38" was a
miscount).

**Baseline reproduced.** 654 passed in 586.8 s. One Bayesian recovery warning,
exactly as documented (worst R-hat 1.086056330170683, minimum ESS
25.660557049567544). The slow real-model tracking tests are included. The Gate
8 focused suite gives 99 passed. `marimo check` and ruff on the Gate 8 files
are clean.

### Verdict: **ACCEPT WITH CONDITIONS**

The canonical evidence is statistically sound and independently reproduced.
The conditions are code guarantees that do not hold on paths the canonical run
did not take (F1, F2), a loadability defect the in-process check cannot see
(F3), and evidence-labelling and test-power gaps. The most serious test-power
gap is F12: no test would notice the lockbox evaluator scoring training rows.
Gate 8 should not be marked complete until F1–F3 and F12 are remediated or
explicitly accepted by the user.

#### §12 acceptance items

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Cross-family metric objects comparable where claimed | **Holds** | `composition_log_loss` is one child-weighted function for all families (`metrics.py:614-671`). CV `mean` is an unweighted mean over folds for every candidate (`experiment/aggregation.py:17-18`). `DirectCohortModel` probabilities are cohort mean ÷ summed total (`results.py` `from_means`); logged predictions confirm it to 1.7e-16. `DirectCohortModel` declares no total RMSE/MAE and no joint NLL (0 rows in `aggregate_metrics.csv`). Joint NLL sums `total` plus cohort keys only under `sequential_joint` scope (`metrics.py:546-556`). Independent recomputation of the whole decision from the CV parent's `aggregate_metrics.csv` (pandas, no package code) matches `pretest_freeze.json` exactly (max \|Δ\| = 0.0). |
| 2 | Rule cannot read test evidence | **Holds** | `select_cross_family_winner(cv_result)` reads only `selections`, `freeze.candidate_descriptors`, `freeze.manifest_fingerprint` and `aggregate_metrics_df`. CV partitions refuse holdout IDs (`partitions.py`, `validate_experiment_partitions`). Probe P1: poisoning every other frame with winning "test-like" rows leaves the decision byte-identical, while perturbing the aggregate flips it (power). |
| 3 | Decision artifact predates manifest replay | **Holds in code; corroborated, not provable, from the store** | Probe P2 instrumented a real `run_final_evaluation`: the order is `pretest_freeze` logged, then `test_lock_status=opened`, then `replay_split_manifest`. A failure while logging the freeze stops before replay (parent FAILED, `locked`). The store corroborates: file birth times are `pretest_freeze.json` 418 s, `finalized_freeze.json` and `test_metrics.csv` 420 s, then children's `predictions.csv` 422–425 s. The first child metric is at 420.41 s, and the 734 s refit ends before the parent opens. Pretest and finalized decisions are identical. The store cannot prove tag order: tags have no history, artifacts have no MLflow timestamps, and filesystem times are mutable. No test detects the order either (see F5, M5). |
| 4 | Full-profile Bayesian diagnostics enforced | **Holds for the canonical run; check is label-only (F7)** | `refit_metadata.json`, both stages: 4 chains, 1000 warmup, 1000 samples, `action=error`, R-hat ≤ 1.0066, ESS ≥ 1397, 0 divergences, policy passed. These equal the `full_profile`/`full_diagnostics` defaults. The refit runs before any MLflow run opens (`tracking/final.py:321-339`). |
| 5 | Identical ordered test IDs | **Holds** | Code: per-model alignment plus a cross-model check (`final_evaluation.py:619-627`). Store: all three `predictions.csv` have identical order, 307 unique IDs, and a set equal to the manifest holdout. The logged `split_manifest.json` equals `artifacts/lockbox/split_manifest.json`. |
| 6 | Final NLL scopes visibly distinct | **Partially** | Distinct in `comparability.json` and in the joint metric's definition (`sequential joint …`). Not distinct in `test_metrics.csv`, the children's `metrics.csv`, or MLflow metric keys: no scope column, and the per-target definitions omit "conditional". A reader can sort `predictive_nll:n_highschool` (1.4e-17 / 4.9e-17 / 2.867) straight from the table (F6). |
| 7 | Test results cannot mutate selection or configuration | **Holds for the canonical evidence; fails as a code guarantee (F1, F2)** | Store: `finalized_freeze.json` equals `pretest_freeze.json` except `test_metrics`. The pretest freeze's selections, descriptors, seed and fold identities equal the CV parent's `freeze.json` and `selections.json`. Parent tags, params and child `role` tags agree. There are no FAILED, KILLED or deleted runs (8 runs, all `FINISHED`/`active`). |

Independent recomputation of test metrics (R2): composition log loss and
per-target RMSE/MAE, recomputed from each child's `predictions.csv` plus
replayed targets, match every logged value to ≤ 4.4e-16. The replay also
re-verified the table and schema hashes, the manifest fingerprint `52b4f7be…`,
and that 0 holdout buildings fall in neighborhoods absent from training.
Joint and per-target NLL were not recomputed: that would require re-predicting
the holdout.

### Findings, ranked

**F1 — High (probe-confirmed). A failed final attempt can be retried with a
different selected family, and it leaves test evidence behind.**
`tracking/runs.py:229-247` refuses only *complete* prior final runs for the
same source CV run and manifest.
Scenario (probe P4, temporary store):
1. Attempt 1 fails while logging the third model, after
   `test_lock_status=opened`.
2. It leaves the parent FAILED holding `test_metrics.csv`. Two FINISHED
   children and one FAILED child hold `predictions.csv` and `test/*` metrics.
3. A pretest freeze naming a *different* candidate is built from the same Gate
   6 freeze and passed to `run_final_evaluation`. It is accepted and completes.
   Nothing links it to attempt 1.
4. A third attempt is then refused.

A person who saw attempt 1's test numbers could re-select through the
sanctioned API. This violates boundary 2 in substance, though boundary 9's
letter covers only complete runs. It is not realized in the canonical store,
which has no failed attempt.

**F2 — High (probe-confirmed). The lockbox path does not bind the pretest
freeze to the rule or to the CV result it came from.** The following never
compare `pretest_freeze.cross_family_selection` with
`select_cross_family_winner(cv_result)`, or the pretest freeze's Gate 6 fields
with `cv_result.freeze`:
- `run_final_evaluation` (`tracking/final.py:316-339`)
- `evaluate_frozen_models_on_lockbox` (`final_evaluation.py:541-570`)
- `with_cross_family_selection` (`evidence.py:136-163`)

Probe P5 shows three consequences:
- a pretest freeze with `master_seed` changed (32 vs 31) is accepted, and the
  evaluator's seeds then diverge from the refit's;
- a pretest freeze with rewritten `selections` evidence is accepted;
- every Gate 8 tracking and evaluation test passes a hand-built stub decision
  (rule `test-stub`, empty `criterion_values`), and the pipeline accepts it.

This is what makes F1 exploitable.

**F3 — Medium (probe-confirmed). The canonical `direct-poisson` model cannot be
loaded in a fresh process in this project's environment.** Loading
`models:/m-fcacaf40…` (from a copy) in a new interpreter segfaults inside
LightGBM `model_from_string`. It also segfaults without MLflow
(`DirectCohortModel.from_state_bundle`) and with `PYTHONPATH=src`. It works
when `lightgbm` is imported before `torch`, and output then matches the logged
`predictions.csv` to 1.8e-15.

Cause:
- `models/__init__.py:5-6` and `tracking/pyfunc_model.py:25-33` import
  `bayesian_conditional` (torch, which bundles its own `libomp`) before
  `direct_cohort`.
- The venv has two `libomp` copies (torch and scikit-learn).
- `KMP_DUPLICATE_LIB_OK=TRUE` does not help.

The in-process reload in `_log_final_model` cannot detect this, because
LightGBM is already initialized there. Outcome item 9 ("loadable") therefore
holds only in-process for this model. The other two models load and reproduce
in fresh interpreters.

**F4 — Medium (store-confirmed). Final child runs carry
`test_lock_status=locked` while holding test predictions and metrics.** All
three canonical children show it. `_terminating_run` stamps the caller's
context tag, `locked`, on children (`runs.py:292-300`, `tracking/final.py:168-175`).
Only the parent is flipped. A filter such as `tags.test_lock_status='locked'`,
meaning "safe, pre-test evidence", returns test evidence. No test asserts a
child's lock tag.

**F5 — Medium (sabotage-confirmed). Several Gate 8 guarantees have no
detecting test.** See the harness table below.
- M5: setting `opened` before the pretest freeze is logged passes all 99
  focused tests. `test_pretest_freeze_is_logged_before_the_lockbox_opens`
  (`test_gate8_tracking.py:224-236`) inspects state only after the block
  opens.
- M2: removing the total-exclusion filter (`final_selection.py:258`) passes.
  The decoy total rows sit only on conditional candidates
  (`test_final_selection.py:251-259`), while the cohort list is read from the
  `DirectCohortModel` descriptor, so the decoys can never reach it. The
  canonical Model A declares no total metric, so this is latent.
- M8 passes, but is equivalent: per-model alignment already implies identical
  IDs.
- M20: logging the Gate 6 freeze, which carries no decision, as
  `pretest_freeze.json` passes all 99 tests. The ordering test checks that the
  artifact exists, never its contents, so a pretest artifact missing the very
  decision it exists to record would go unnoticed.
- M25: removing the refit's training-data hash check
  (`final_evaluation.py:360-363`) passes. The schema-hash and outer-ID checks
  remain, but a refit on altered training values with the same IDs and
  columns would go unnoticed.

**F6 — Low/Medium (store-confirmed). NLL scope is not carried in the test
tables.** `test_metrics.csv` columns are `metric_name, value,
aggregation_level, sample_count, target, candidate_id, approach`. The
per-target `predictive_nll` definitions for the conditional models say
"posterior-integrated NB2 total plus Dirichlet-multinomial composition", not
conditional. The scope survives only in `comparability.json`.

**F7 — Low (probe-confirmed). `_require_full_bayesian_policy` checks labels,
not what ran** (`final_evaluation.py:309-328`; the label comes from
`bayesian_conditional.py:249`). Probe P3 found:
- a `BayesianConditionalConfig` with `active_profile="full"`,
  `full_profile=reduced_profile` and `maximum_rhat=10` constructs;
- diagnostics reporting 1 chain, 10 samples and R-hat 3.0 pass.

The canonical registry uses defaults, and the logged draws are genuinely full
(item 4), so this is not realized.

**F8 — Low (confirmed). The logged unseen-neighborhood count describes the
reload smoke frame, not the holdout.** `model_metadata` is captured at refit
(`final_evaluation.py:456`) and logged as-is (`:665`). The refit's own
reload-check prediction on training rows set
`last_prediction_unseen_neighborhood_count=0`. Nothing asserts, at evaluation
time, that the Bayesian point means drew no random effect. Replay shows the
canonical holdout has 0 such buildings, so exact reproducibility held.

**F9 — Low (probe-confirmed).** `FinalEvaluationResult.__post_init__`
(`final_evaluation.py:197-203`) checks leakage against
`DEFAULT_MODELING_SCHEMA`. Probe P7: a frame carrying a custom schema's target
column is accepted, while a default target is refused. Canonical calls use the
default schema.

**F10 — Low (judgement, store-confirmed). The final Bayesian child reads as a
reduced-profile model.** Its run name, `candidate_id` and frozen descriptor
(`configuration={"profile": "reduced"}`, in `finalized_freeze.json`) say
reduced. Only the child's `refit_metadata.json` shows `active_profile=full`.
No parent or child tag or param records the refit profile.

**F11 — Low (test hygiene; code-confirmed). The notebook's setup test can
create the canonical lockbox manifest.**
`test_default_script_execution_performs_setup_only`
(`tests/characterization/test_model_fitting_notebook.py:90-130`) runs the
notebook with `cwd=PROJECT_ROOT`. It redirects MLflow but not
`project_root / "artifacts/lockbox/split_manifest.json"`. On a checkout without
the file, running the suite persists the canonical manifest, which is how it
was first created. The code path and seed are identical to the notebook's, and
replay verifies the hashes, so the canonical provenance is sound. The side
effect is not.

**F12 — Medium-High (sabotage-confirmed, test power). No test would notice the
lockbox evaluator scoring the wrong rows.** Mutation M11 replaces every
`split.test_df` in `final_evaluation.py` with `split.train_df`, so all three
models are scored on the training partition. All 99 focused tests pass.

The cause is circular: every holdout-ID check, including the slow real-model
`test_gate8_final_predictions_share_identical_holdout_ids`
(`tests/validation/test_tracking_real_models.py:508-517`), compares prediction
IDs with `evaluation_result.holdout_building_ids`. That value is computed from
the same frame the evaluator scored (`final_evaluation.py:598`). Nothing
compares against `split_manifest.holdout_building_ids`.

This is not realized: the canonical `predictions.csv` sets equal the manifest
holdout (item 5), and R2 reproduces the logged metrics from holdout targets.

**F13 — Medium (sabotage-confirmed, test power). Nothing tests that the pyfunc
reproduces predictions or that the leakage guard fires.**
- **M14.** Making `_log_final_model`'s equality check vacuous passes every
  test. `test_logged_pyfunc_reloads_and_reproduces_predictions_exactly`
  (`test_gate8_tracking.py:588-602`) asserts only that a `final_model_uri` tag
  exists. M7 was caught only because the in-run check then fired.
- **The slow test's name overstates it.**
  `test_gate8_final_models_are_loadable_pyfuncs_with_matching_predictions`
  (`test_tracking_real_models.py:519-527`) asserts only `loaded is not None`,
  in-process.
- **M13.** Removing the `FinalEvaluationResult` leakage guard passes too.
  `test_holdout_input_frame_carries_no_target_column` checks the produced
  frame, never the guard.

### Gate 8 — Canonical run record

Moved verbatim from the local handoff `GATE_8_VALIDATION_HANDOFF.md` (its §3 and
Appendix A) when `docs/archive/` was deleted on 2026-09-24; that file no longer
exists. Headings are demoted one level; the text is unchanged.

#### Where it lives

| Item | Value |
|---|---|
| Tracking store | `sqlite:///mlflow.db` (repository root), artifacts in `mlartifacts/` |
| Experiment | `age-group-prediction` |
| CV comparison parent (Gate 6/7) | `9cf84b8e1e9f4ed1b75e61511449b50f` (3 candidate children) |
| Final evaluation parent (Gate 8) | `9450596574494ea188a05f7988e28650` (3 final children) |
| Run status | All 8 runs `FINISHED` with `evidence_complete=true` |
| Split manifest | `artifacts/lockbox/split_manifest.json` |
| Manifest fingerprint | `52b4f7be54c61d36fc45256b3523ba5ebb4b0fad4ad7a1672e383bd48f74eeeb` |
| Table | 1,529 buildings, 150 neighborhoods; 1,222 training, 307 holdout (realized fraction 0.20078) |
| Master seed | 42, source `experiment_config.randomness.default_seed` |

`mlflow.db`, `mlartifacts/` and `artifacts/lockbox/` are git-ignored.

**Final parent artifacts:** `pretest_freeze.json`, `finalized_freeze.json`,
`cross_family_rule.json`, `comparability.json`, `provenance.json`,
`test_metrics.csv`, `split_manifest.json`. Tags include
`test_lock_status=opened`, `source_cv_run_id`, `manifest_fingerprint`,
`cross_family_rule_version=1.0`, `selected_candidate_id`, `selected_approach`.

**Each final child:** tags `candidate_id`, `approach`, `role`
(`selected`/`comparator`), `final_model_uri`; artifacts `refit_metadata.json`,
`seeds.json`, `state_bundle.json.gz`, `reload_check.json`,
`evaluation_metadata.json`, `predictions.csv`, `metrics.csv`, `intervals.csv`,
and the logged pyfunc model.

#### How it ran

- The manifest file was **first created by a Phase 5 script-mode test** of the
  notebook's setup cells (`split_known_neighborhood_buildings` seeded with
  `default_seed`, then `persist_split_manifest`). No CV or final action ran then.
- The canonical run itself was executed by a **scratch script that mirrors the
  notebook's cells**, not by clicking the notebook's buttons. It is reproduced
  verbatim in Appendix A, because the original scratchpad does not survive the
  session. It loaded and replayed the persisted manifest, ran and tracked CV,
  computed the cross-family decision, and called `run_final_evaluation` once.
- The canonical registry declares **one candidate per approach**, so every
  within-approach selection was trivial (`rejected_candidate_ids` empty).

#### Cross-family decision (from `pretest_freeze.json`)

| Criterion (CV aggregate mean) | `bayesian-reduced` | `independent-nb2` | `direct-poisson` |
|---|---|---|---|
| `joint_predictive_nll` | **7.8363** | 8.0107 | not produced |
| `composition_log_loss` | **1.08386** | — | 1.08843 |
| `mean_cohort_rmse` | 4.1272 | — | 4.3990 |
| `mean_cohort_mae` | 3.0660 | — | 3.2687 |

Conditional winner `bayesian-reduced` (no tie-break). Selected
`bayesian-reduced` (`BayesianConditionalModel`); decisive criterion
`composition_log_loss` (margin 0.0046); no final tie-break.
`direct-poisson` and `independent-nb2` are comparators.

#### Test metrics (descriptive only; from `test_metrics.csv`)

| Metric | `bayesian-reduced` | `direct-poisson` | `independent-nb2` |
|---|---|---|---|
| `composition_log_loss` (child) | 1.07522 | 1.07928 | 1.07541 |
| `joint_predictive_nll` (building) | 7.8021 | — | 7.9258 |
| RMSE kindergarten / elementary / highschool | 4.680 / 3.718 / 3.850 | 4.720 / 3.848 / 4.146 | 5.126 / 3.957 / 3.933 |
| MAE kindergarten / elementary / highschool | 3.363 / 2.623 / 2.854 | 3.434 / 2.705 / 3.045 | 3.637 / 2.793 / 2.941 |
| `predictive_nll` n_highschool | 1.4e-17 | 2.867 | 4.9e-17 |

The near-zero last-cohort `predictive_nll` for the conditional models is the
documented artifact of their schema-ordered conditional decomposition; it is
exactly why per-target `predictive_nll` must never be ranked across families.

#### Bayesian final refit (from its `refit_metadata.json`)

| Stage | `active_profile` | Chains / warmup / samples | Worst R-hat | Min ESS | Divergences | Policy |
|---|---|---|---|---|---|---|
| total | full | 4 / 1000 / 1000 | 1.0066 | 1397 | 0 | passed, `action=error` |
| composition | full | 4 / 1000 / 1000 | 1.0048 | 1796 | 0 | passed, `action=error` |

Fit duration 734 s. The reduced-profile Bayesian fits during CV logged no
diagnostic-failure warnings.

#### Appendix A: canonical run script (verbatim)

Executed once, on 2026-09-14, from the repository root with
`MLFLOW_DISABLE_AGENT_HINT=1 uv run --group tracking python run_gate8_canonical.py`.
**Do not run it again.** It is kept as provenance only.

```python
"""One-time canonical Gate 8 final evaluation.

Mirrors notebooks/02_model_fitting.py's guarded cells exactly, using the
real (already-persisted) lockbox manifest and the default local MLflow
store. This is the authorized, one-time action: do not rerun against the
same manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/galkampel/Desktop/Projects/age-group-prediction")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from age_group_prediction import (
    DEFAULT_MODELING_SCHEMA,
    build_modeling_table,
    load_experiment_config,
    load_split_manifest,
    make_validation_folds,
    replay_split_manifest,
)
from age_group_prediction.experiment import (
    build_canonical_candidate_registry,
    run_cross_model_validation,
    select_cross_family_winner,
)
from age_group_prediction.tracking import (
    TrackingContext,
    log_cross_validation_experiment,
    resolve_tracking_settings,
    run_final_evaluation,
)
from student_simulator import StudentPopulationSimulator, load_simulation_config

print("=== 1) Canonical population ===", flush=True)
simulation_config_path = PROJECT_ROOT / "configs" / "stage1.toml"
base_simulation_config = load_simulation_config(simulation_config_path)
canonical_simulation_config = base_simulation_config.model_copy(
    update={
        "simulation": base_simulation_config.simulation.model_copy(
            update={"n_neighborhoods": 150}
        ),
        "building": base_simulation_config.building.model_copy(
            update={"buildings_per_neighborhood_rate": 9.0}
        ),
    }
)
population_df = StudentPopulationSimulator(canonical_simulation_config).run()
print(
    f"population: {len(population_df):,} buildings, "
    f"{population_df['neighborhood_id'].nunique()} neighborhoods",
    flush=True,
)

print("=== 2) Modeling table ===", flush=True)
modeling_df = build_modeling_table(population_df)
print(f"modeling_df: {len(modeling_df):,} rows", flush=True)

print("=== 3) Experiment configuration ===", flush=True)
experiment_config_path = PROJECT_ROOT / "configs" / "modeling.toml"
experiment_config = load_experiment_config(experiment_config_path)

print("=== 4) Replay the persisted lockbox manifest ===", flush=True)
lockbox_manifest_path = PROJECT_ROOT / "artifacts" / "lockbox" / "split_manifest.json"
assert lockbox_manifest_path.exists(), "canonical manifest must already be persisted"
split_manifest = load_split_manifest(lockbox_manifest_path)
split = replay_split_manifest(
    modeling_df, split_manifest, config=experiment_config.outer_split
)
print(
    f"train={len(split.train_df):,} holdout={len(split.manifest.holdout_building_ids):,} "
    f"(holdout rows never printed)",
    flush=True,
)

print("=== 5) Candidate registry and validation folds ===", flush=True)
candidate_registry = build_canonical_candidate_registry(experiment_config)
validation_fold_plan = make_validation_folds(
    split.train_df,
    config=experiment_config.folds,
    rng=np.random.default_rng(experiment_config.randomness.default_seed),
)
print(
    f"candidates={len(candidate_registry.candidates)} "
    f"folds={len(validation_fold_plan.folds)}",
    flush=True,
)

print("=== 6) Run and track cross-validation (Gate 6/7) ===", flush=True)
cv_result = run_cross_model_validation(
    split.train_df,
    split_manifest=split.manifest,
    validation_folds=validation_fold_plan.folds,
    candidates=candidate_registry.candidates,
    selection_policies=candidate_registry.selection_policies,
    experiment_config=experiment_config,
    capture_artifacts=True,
)
source_cv_run_id = log_cross_validation_experiment(
    cv_result,
    TrackingContext(run_name="gate8-cv-canonical"),
    train_df=split.train_df,
)
print(f"source_cv_run_id={source_cv_run_id}", flush=True)
for selection in cv_result.selections:
    print(f"  selected[{selection.approach}] = {selection.selected_candidate_id}", flush=True)

print("=== 7) Cross-family decision (Gate 8) ===", flush=True)
cross_family_selection = select_cross_family_winner(cv_result)
pretest_freeze = cv_result.freeze.with_cross_family_selection(cross_family_selection)
print(
    f"cross_family_winner={cross_family_selection.selected_candidate_id} "
    f"({cross_family_selection.selected_approach})",
    flush=True,
)
print(
    f"decisive_final_criterion={cross_family_selection.decisive_final_criterion} "
    f"tie_break={cross_family_selection.final_tie_break_used}",
    flush=True,
)

print("=== 8) FINAL EVALUATION (one-time, opens the lockbox) ===", flush=True)
final_evaluation_result = run_final_evaluation(
    split.train_df,
    modeling_df,
    split_manifest=split.manifest,
    cv_result=cv_result,
    pretest_freeze=pretest_freeze,
    candidates=candidate_registry.candidates,
    final_refit_factories=candidate_registry.final_refit_factories,
    context=TrackingContext(run_name="gate8-final-canonical"),
    source_cv_run_id=source_cv_run_id,
    evaluation_config=experiment_config.evaluation,
)
print("Final evaluation complete.", flush=True)

tracking_settings = resolve_tracking_settings()
print(f"tracking_uri={tracking_settings.tracking_uri}", flush=True)
print(f"experiment_name={tracking_settings.experiment_name}", flush=True)
print(f"manifest_fingerprint={final_evaluation_result.manifest_fingerprint}", flush=True)
print(f"holdout_building_count={len(final_evaluation_result.holdout_building_ids)}", flush=True)
for evaluation in final_evaluation_result.evaluations:
    print(
        f"  {evaluation.candidate_id} ({evaluation.approach}, role={evaluation.role})",
        flush=True,
    )

print("=== DONE ===", flush=True)
```

Key output lines (Optuna and MLflow logging omitted):

```text
population: 1,529 buildings, 150 neighborhoods
modeling_df: 1,529 rows
train=1,222 holdout=307 (holdout rows never printed)
candidates=3 folds=5
  selected[BayesianConditionalModel] = bayesian-reduced
  selected[DirectCohortModel] = direct-poisson
  selected[IndependentTotalProbabilityModel] = independent-nb2
cross_family_winner=bayesian-reduced (BayesianConditionalModel)
decisive_final_criterion=composition_log_loss tie_break=False
Final evaluation complete.
tracking_uri=sqlite:///mlflow.db
experiment_name=age-group-prediction
manifest_fingerprint=52b4f7be54c61d36fc45256b3523ba5ebb4b0fad4ad7a1672e383bd48f74eeeb
holdout_building_count=307
  bayesian-reduced (BayesianConditionalModel, role=selected)
  direct-poisson (DirectCohortModel, role=comparator)
  independent-nb2 (IndependentTotalProbabilityModel, role=comparator)
=== DONE ===
```

### Sabotage harness (focused Gate 8 suite, 99 tests; each mutation alone, then restored)

| Mutation | Caught by |
|---|---|
| M1 swap composition/RMSE order | `test_composition_log_loss_decides_the_final_stage_first` |
| M2 total enters cohort reduction | **none** (F5) |
| M3 stage 2 reads per-target `predictive_nll` | 17 tests in `test_final_selection.py` |
| M4a refit fingerprint checks removed | `test_refit_rejects_a_manifest_fingerprint_mismatch` |
| M4b evaluator fingerprint checks removed | `test_lockbox_evaluation_rejects_a_manifest_mismatch` |
| M4c logger fingerprint check removed | `test_a_manifest_mismatch_is_refused_before_any_child_logs` |
| M5 `opened` set before pretest freeze logged | **none** (F5) |
| M6 `_require_full_bayesian_policy` call removed | `test_refit_enforces_the_full_bayesian_profile`, `…_strict_bayesian_diagnostic_policy` |
| M7 pyfunc returns constant total | `test_logged_pyfunc_reloads_and_reproduces_predictions_exactly`, `test_pyfunc_model_class_dispatches_and_predicts_from_a_bundle` |
| M8 identical-ID assertion dropped | none (equivalent mutant) |
| M9 duplicate-final-run refusal disabled | `test_a_second_complete_final_run_is_refused` |
| M10a `with_test_metrics` second-call refusal removed | `test_freeze_rejects_a_second_test_result` |
| M10b evaluator already-finalized refusal removed | `test_lockbox_evaluation_refuses_a_second_evaluation` |
| M11 evaluator scores the **training** partition instead of the holdout | **none** (F12) |
| M13 `FinalEvaluationResult` leakage guard removed | **none** (F13) |
| M14 in-run pyfunc reload check compares expected to itself | **none** (F13) |
| M15 `with_cross_family_selection` second-call refusal removed | `test_freeze_records_a_cross_family_decision_once` |
| M16 source-CV completeness check removed | `test_incomplete_source_cv_run_is_refused`, `test_unknown_source_cv_run_is_refused` |
| M17 selected/comparator roles inverted | `test_lockbox_evaluation_does_not_rank_across_families`, `test_lockbox_evaluation_scores_all_three_winners_once`, `test_selected_versus_comparator_roles_are_tagged` |
| M19 stage 1 picks the *higher* joint NLL | 8 tests in `test_final_selection.py` |
| M20 logged `pretest_freeze.json` is the Gate 6 freeze (no decision) | **none** (F5) |
| M24 refit descriptor-match check removed | `test_refit_rejects_a_candidate_descriptor_mismatch` |
| M25 refit training-data hash check removed | **none** (F5) |

Summary: 23 mutations, 15 caught and 8 not. Of the 8, M8 is an equivalent
mutant; the other 7 (M2, M5, M11, M13, M14, M20, M25) are real gaps (F5, F12,
F13). When the harness finished, the scratch copy's `src/` was diffed against
the repository's `src/` and showed no differences.

### Open points from the handoff, resolved

- **`code_paths` loadability out of process — withdrawn as a `code_paths`
  defect.** In a fresh interpreter with no `PYTHONPATH`, `age_group_prediction`
  is not importable before loading; after loading it resolves inside each
  copied model's `code/` directory for all three models. The separate
  fresh-process crash is F3.
- **Unseen-neighborhood reproducibility.** Confirmed as a gap in assertion and
  labelling (F8), not realized: 0 unseen holdout buildings.
- **Default-schema leakage check.** Confirmed, low (F9).
- **`bayesian-reduced` naming on a full refit.** Confirmed, low (F10).
- **Retry semantics.** Confirmed, high (F1, with F2).
- **Canonical provenance.** Acceptable. Appendix A of the validation handoff
  (reproduced in the Canonical run record above) matches the notebook's setup,
  CV, decision and final cells argument for argument, apart from run names.
  The manifest was created by that same code path, with seed 42 and
  `seed_source=caller_generator`, and replay verifies it. The plan's
  implementation record does not mention that the canonical run came from a
  script mirroring the notebook rather than the notebook's buttons. ###
  Findings withdrawn
- The `code_paths` concern (above).
- "Retry could already have contaminated the canonical run": the store has 8
  runs, all `FINISHED`/`active`, with artifact directories matching.
- "Manifest provenance is unacceptable because a test created it": same code
  path, seed and hashes (kept only as the hygiene finding F11).
- The pyfunc reload check comparing a seedless default-config prediction
  rather than `predictions.csv` (`tracking/final.py:255-278`): this does not
  affect canonical correctness. Fresh-process pyfunc output equals the logged
  `predictions.csv` rows (≤ 3.6e-15, CSV round-trip) for all three models. It
  remains a documentation point below.

### Statistical judgement calls (not defects)

- **Composition margin 0.00457 (≈ 0.4 % of the loss).**
  - Paired by fold, Bayesian minus Direct is negative in **5/5** folds (mean
    −0.00457, SD 0.00218).
  - Mean cohort RMSE (−0.272, SD 0.076) and MAE (−0.203, SD 0.051) are also
    5/5 in Bayesian's favour, so the outcome does not depend on the order of
    the final criteria.
  - The folds are 5 overlapping, correlated draws (`partitions.py:95-98`), so
    a naive paired t (≈ −4.7) overstates precision.
  - The unpaired CV-level bootstrap intervals overlap (1.0810–1.0865 vs
    1.0855–1.0917).

  Report it as a small but directionally consistent CV advantage, not a
  resolved difference. This is never grounds to re-select.
- **Composition-first applied as declared (§3.1).** Stage 1 compares only the
  two conditional winners, on joint NLL (margin 0.174). Stage 2 applies
  composition, then RMSE, then MAE, then candidate ID, with exact-equality
  ties. No interval, runtime or test input is used.
- **Composition near-tie between the conditional models, not reached by the
  rule.** In CV, `independent-nb2`'s composition log loss (1.083831) is
  marginally *below* Bayesian's (1.083861). The rule never compares them on
  composition, by design.
- **Validated at reduced, deployed at full.** The CV evidence behind the
  selection is from the reduced NUTS profile; the delivered model is the
  full-profile refit. This is by design, but worth stating wherever the
  selection is reported (compare F10).
- **Test results (descriptive only).** Composition log loss is Bayesian
  1.07522, Direct 1.07928, Independent 1.07541. Nothing here was used for any
  decision.

### Documentation accuracy

| Claim | Location | Assessment |
|---|---|---|
| Gate 8 "complete" | `docs/README.md:90` | **Unsupported.** Brief §13 forbids the claim before independent statistical acceptance. |
| Phases "each stopped for approval with an independent review subagent" | plan `MODELING_REBUILD_PLAN.md:2027`, findings `:2786` | **Overstated.** The subagents were launched by the implementer. |
| Header "Final statistical acceptance: GPT-5.6 Opus" | plan `:1957` | Inaccurate model label (acceptance performed by Opus 5). |
| Joint NLL "verified against independent scipy oracles … pinned by `test_bayesian_and_independent_keys_are_the_same_conditional_object`" | plan `:1990-1994` | **Supported, with a nuance.** Value-level oracle tests exist (`test_bayesian_conditional.py:503`, `test_independent_total_probability.py:437`); the named test pins structure only. |
| Manifest must equal "byte-for-byte" | `README.md:246-248` | **Overstated.** Dataclass equality (`data_splitting.py:137-144`); probe P8 shows a re-indented file is accepted. The semantic guarantee (no relabelling) holds. |
| "A second complete final run … is refused" | `README.md:271-272` | Supported (M9 caught). |
| Lockbox opens "only once that evidence is durably recorded" | `README.md:273-275` | Supported in code (P2); **untested** (M5). |
| "a caller can never route test evidence anywhere except through it" | `README.md:275-277` | **Overstated** (F1: a failed attempt leaves test evidence; the returned result carries test metrics). |
| Pyfunc required "to reproduce the evaluator's point predictions exactly" | `README.md:296-301` | **Overstated in mechanism.** It compares a fresh seedless, default-config prediction, not the evaluator's output. The outcome holds for the canonical models (probe P6). |
| Pyfunc models "loadable" | `README.md:282-301`, plan `:2081-2082`, findings `:2846-2847` | **Overstated.** Proven in-process only; `direct-poisson` segfaults in a fresh process in this environment (F3). |
| Final action "cannot be undone for a given manifest" | notebook checkbox label | **Overstated** (F1). |
| `opened` "only after the pretest freeze was logged … verified directly against the MLflow store" | findings `:2848-2851` | **Overstated.** The store cannot evidence tag order; supported by code (P2) and corroborated by file birth times only. |
| Canonical numbers (decision table, test metrics, Bayesian diagnostics, 654 tests) | validation handoff §3, plan `:2073-2090` | Supported; all independently recomputed or re-read. |
| Canonical run "replayed once" via the notebook workflow | plan `:2073-2086` | Incomplete: it omits that a script mirroring the notebook ran it (accurately recorded in the validation handoff, Appendix A). |

### Proposed remediation (not applied)

1. **F1/F2.**
   - In `run_final_evaluation`, require that `pretest_freeze` minus the
     decision and test fields equals `cv_result.freeze`, and that its decision
     equals `select_cross_family_winner(cv_result)`.
   - In `tracked_final_evaluation`, refuse to open when any prior final run for
     the same source run and manifest (any status) reached
     `test_lock_status=opened` with a different `selected_candidate_id` (or a
     different decision hash tag). Otherwise tag the retry with the prior
     attempt's run ID.
   - Tests: the P4 and P5 scenarios.
2. **F3.**
   - Import `direct_cohort` (LightGBM) before `bayesian_conditional` in
     `tracking/pyfunc_model.py`, and consider the same order in
     `models/__init__.py`. Or isolate the OpenMP runtimes.
   - Add a slow test that loads a logged model in a `subprocess` without the
     package pre-imported.
   - Verify against the P6 probe for all three canonical model copies.
3. **F4.** Tag final children `test_lock_status=opened`, and assert it.
4. **F5.** Record event order in the lockbox-ordering test (for example, patch
   `mlflow.set_tag` and `log_artifacts`), and put decoy total rows on the
   `DirectCohortModel` descriptor. Assert the logged `pretest_freeze.json`
   carries the cross-family decision, and add a refit test with altered
   training values but the same IDs. Prove each fails on M5, M2, M20 and M25
   respectively.
5. **F6.** Add an NLL scope column (`marginal` / `sequential_conditional` /
   `sequential_joint`) to the metrics tables, or omit conditional per-target
   NLL rows from cross-family tables.
6. **F7.** Also assert chains, warmup, samples and thresholds equal the
   configured full profile and diagnostics.
7. **F8.** Record, and require zero, unseen-neighborhood rows after the holdout
   prediction.
8. **F9.** Check leakage against the schema actually passed.
9. **F10.** Add a `refit_profile=full` param/tag on the Bayesian final child.
10. **F11.** Point the notebook test's project root or manifest path at
    `tmp_path`.
11. **F12.** In the unit and slow lockbox tests, assert that prediction IDs
    and `holdout_building_ids` equal `split_manifest.holdout_building_ids`
    (an independent oracle). Prove it fails on M11. Optionally add the same
    runtime assertion inside `evaluate_frozen_models_on_lockbox`.
12. **F13.** Make the pyfunc round-trip tests compare loaded output with an
    independently computed prediction frame (prove it fails on M14). Rename or
    strengthen the slow `…_with_matching_predictions` test. Add a test that
    constructs `FinalEvaluationResult` with a target column and expects
    refusal (prove it fails on M13).
13. **Documentation.** Correct the rows marked overstated or unsupported
    above, and set Gate 8 status to "implemented; independent acceptance: with
    conditions".

### Remediation applied — Gate 8 independent validation

Approved scope: F1, F2, F3, F4, F11, F12, the test gaps (M2, M5, M13, M14,
M20, M25), and documentation. F6–F10 are deferred (low severity; not realized
in the canonical run). The canonical store was never modified: evidence
checksums are unchanged, and nothing was re-run against the lockbox.

| Finding | Fix | Guarding test(s) |
|---|---|---|
| F1 | `tracking/runs.py`: `tracked_final_evaluation` tags every final parent `cross_family_decision_hash`. `_retryable_opened_attempts` searches all final runs on the manifest, deleted ones included, that reached `test_lock_status=opened`. It refuses a different source CV run or decision, and allows an identical retry tagged `retry_of_run_ids`. `_refuse_duplicate_final_run` now also counts deleted complete runs (independent review). | `test_a_different_decision_after_the_lockbox_opened_is_refused`, `test_a_different_source_cv_run_after_the_lockbox_opened_is_refused`, `test_an_identical_retry_after_the_lockbox_opened_is_allowed_and_linked`, `test_a_deleted_complete_final_run_still_refuses_a_second_evaluation` |
| F2 | `experiment/final_evaluation.py::verify_pretest_freeze` requires the pretest freeze to equal `cv_result.freeze` plus a decision equal to `select_cross_family_winner(cv_result)`. `run_final_evaluation` calls it before refitting. `tracked_final_evaluation` refuses a pretest freeze without a decision or with mismatched selected tags. | `test_verify_pretest_freeze_*` (6 tests), `test_run_final_evaluation_refuses_a_decision_the_rule_did_not_produce`, `test_a_pretest_freeze_without_a_decision_never_opens_a_final_run` |
| F3 | `models/__init__.py` and `tracking/pyfunc_model.py` import `direct_cohort` (LightGBM) before `bayesian_conditional` (torch). | `test_logged_pyfunc_loads_and_predicts_in_a_fresh_process` (subprocess, no `PYTHONPATH`; asserts the package resolves inside the model's `code/` and predictions match exactly) |
| F4 | Final children run under `TrackingContext.with_test_lock_status("opened")`. | `test_final_children_are_tagged_as_opened_test_evidence` |
| F11 | The notebook setup test runs a copy of the notebook in a temporary project and asserts the repository manifest's mtime is unchanged. | `test_default_script_execution_performs_setup_only` |
| F12 | `evaluate_frozen_models_on_lockbox` refuses a replay whose holdout IDs differ from `split_manifest.holdout_building_ids`. The unit and slow tests compare against the manifest. | `test_lockbox_evaluation_scores_exactly_the_manifest_holdout`, `test_lockbox_evaluation_refuses_a_replay_whose_holdout_is_not_the_manifest`, slow `test_gate8_final_predictions_share_identical_holdout_ids` |
| M2 | Test only. | `test_total_metrics_declared_by_direct_cohort_never_enter_the_cohort_reduction` |
| M5, M20 | Test only; event order and logged content are both observed. | `test_run_final_evaluation_logs_the_decision_before_the_lockbox_opens` |
| M13 | Test only. | `test_final_evaluation_result_refuses_a_target_column` |
| M14 | Test only; the slow pyfunc test also now compares reloaded predictions with the logged ones. | `test_log_final_model_refuses_a_pyfunc_that_does_not_reproduce`, slow `test_gate8_final_models_are_loadable_pyfuncs_with_matching_predictions` |
| M25 | Test only. | `test_refit_rejects_altered_training_values_with_the_same_ids` |

**Power proofs.** Each fix was reverted, or its defect re-applied, in a
scratch copy of the remediated repository, and the focused unit suite (120
tests) or notebook suite was run. All 17 were caught:

| Revert / defect | Caught by |
|---|---|
| F1 retry check disabled | both "different … after the lockbox opened" tests |
| F2 freeze-equality check disabled | `…refuses_a_freeze_that_is_not_this_cv_results[master_seed]`, `[selections]` |
| F2 rule-equality check disabled | `…refuses_a_decision_the_rule_did_not_produce` (evaluation and tracking) |
| F2 call removed from `run_final_evaluation` | `test_run_final_evaluation_refuses_a_decision_the_rule_did_not_produce` |
| F3 original torch-first import order | `test_logged_pyfunc_loads_and_predicts_in_a_fresh_process` |
| F4 children inherit `locked` | `test_final_children_are_tagged_as_opened_test_evidence` |
| M11 train partition scored (runtime check present) | 32 tests (runtime check fires) |
| M11 with the runtime check also disabled | `test_lockbox_evaluation_scores_exactly_the_manifest_holdout`, and the replay test |
| F12 runtime check alone disabled | `test_lockbox_evaluation_refuses_a_replay_whose_holdout_is_not_the_manifest` |
| M2, M5, M13, M14, M20, M25 | each by its new test (table above) |
| M20b decision-less refusal removed | `test_a_pretest_freeze_without_a_decision_never_opens_a_final_run` |
| F11 notebook test from the repository root | `test_default_script_execution_performs_setup_only` |
| Deleted-run duplicate check without `ViewType.ALL` | `test_a_deleted_complete_final_run_still_refuses_a_second_evaluation` (separate copy) |

**Independent review** (read-only subagent) found no High issues. Four Low
items:
1. The lockbox-retry guarantee is scoped to one MLflow experiment and store.
   This is a residual; the README now says so.
2. The duplicate check ignored deleted runs. Fixed and tested (above).
3. `tracked_final_evaluation` and `evaluate_frozen_models_on_lockbox` stay
   public, and calling them directly bypasses the F2 check. This is a residual,
   consistent with the single-guarded-entry design.
4. A stale suite count in `docs/README.md`. Updated.

**Documentation corrected.**
- `README.md` Gate 8:
  - manifest matching (semantic, not byte-level);
  - F2 binding, the F1 retry rule and its scope, and the manifest holdout check;
  - the child lock tag;
  - what the in-process reload proves, the fresh-process test, and the import
    order;
  - that the canonical `direct-poisson` model needs `import lightgbm` first in
    a fresh process, because its bundled code predates the fix.
- `docs/README.md` status: "accepted with conditions; must-fix remediated; F6–F10
  deferred".
- `MODELING_REBUILD_PLAN.md` Gate 8 record: a validation bullet, including the
  scratch-script provenance of the canonical run.
- The `tracked_final_evaluation` and `_log_final_model` docstrings.

**Canonical evidence, recorded rather than changed.**
- The three canonical final children keep `test_lock_status=locked` (F4
  predates the fix; `mlflow.db` is not modified).
- The canonical final parent has no `cross_family_decision_hash` tag. It is
  complete, so the duplicate check refuses any repeat first.
- The canonical `direct-poisson` model's bundled code keeps the torch-first
  import order.

**Suite (first remediation pass).**
- Full suite: **676 passed** (654 + 22 new tests) in 605 s. One documented
  Bayesian recovery warning is unchanged (worst R-hat 1.086056330170683,
  minimum ESS 25.660557049567544).
- Focused Gate 8 unit suite: 121 tests.
- Slow real-model tracking: 9 passed.
- `marimo check` clean; ruff clean on every touched file.
- Evidence checksums unchanged, and the worktree still shows the same 11
  modified and 41 untracked paths.

### Second remediation pass — F7, F9, and accepted deferrals

Approved 2026-09-14 as the first step to closing Gate 8.

| Finding | Fix | Guarding test(s) | Power proof (scratch copy) |
|---|---|---|---|
| F7 | `_require_full_bayesian_policy` now also requires each stage's recorded `chains`, `warmup_steps` and `posterior_samples` to be at least the package's default full profile (4 / 1000 / 1000). Every recorded threshold must also be at least as strict as the default full diagnostics. The package defaults are the reference, never the model's own config, which a weakened config would also weaken. A stricter configuration passes. The `_full_bayesian_diagnostics` test fixture now carries realistic draw counts and thresholds. | `test_refit_refuses_a_full_label_with_fewer_draws_than_the_full_profile` (4 cases, including a missing count), `test_refit_refuses_a_full_label_with_looser_thresholds` (6 cases), `test_refit_accepts_a_stricter_than_default_full_profile` | Draw-count floor disabled: all 4 caught. Threshold floor disabled: all 6 caught. |
| F9 | `FinalEvaluationResult` gains a `schema` field (the default schema unless given) and checks leakage against it. `evaluate_frozen_models_on_lockbox` passes its schema. | `test_final_evaluation_result_checks_leakage_against_the_schema_it_is_given`, `test_lockbox_evaluation_result_carries_the_schema_it_evaluated_with` | Reverting to the default schema: caught. Evaluator not passing its schema: uncaught on the first run; the second test was added and then caught it. |

The canonical Bayesian refit satisfies F7's floor (4 chains, 1000 warmup,
1000 samples, default thresholds; see `refit_metadata.json`), so the stricter
check would have accepted it.

**Accepted as deferred (not fixed):**
- **F6 — no NLL scope column in the test tables.** Scope is recorded in
  `comparability.json` and in each candidate's comparability text. It affects
  how a table is read, not any decision. Do not build cross-family rankings of
  per-target `predictive_nll`.
- **F8 — no evaluation-time unseen-neighborhood assertion.** The
  known-neighborhood split cannot hold out a building in an unseen
  neighborhood, and manifest replay confirmed 0 for the canonical holdout.
- **F10 — no refit-profile tag.** `refit_metadata.json` records
  `active_profile=full` and the draw counts, and F7 now enforces them.
- **Review residual 1.** The F1 retry guard sees one MLflow experiment and
  store; `README.md` says so.
- **Review residual 3.** `tracked_final_evaluation` and
  `evaluate_frozen_models_on_lockbox` stay public, and only
  `run_final_evaluation` applies the F2 check. It is the single guarded entry
  point, and the notebook uses it.

**Suite (second pass).**
- Full suite: **689 passed** (676 + 13 new tests) in 594 s. The one documented
  Bayesian recovery warning is unchanged.
- Focused Gate 8 unit suite: 134 tests.
- Ruff clean on touched files.
- Evidence checksums unchanged.

**Status.** Gate 8 is independently accepted with conditions. All must-fix
conditions, plus F7 and F9, are remediated, and the remaining items are
accepted as deferred. Because this session wrote the fixes for its own
findings, Gate 8 is complete only after an independent review of the
remediation in a fresh session, briefed by
`GATE_8_REMEDIATION_REVIEW_HANDOFF.md`.

### Gate 8 — Remediation review

An independent review of the remediation code and tests, run on 2026-09-14
(Opus 5) under `GATE_8_REMEDIATION_REVIEW_HANDOFF.md`, Phase 2. It covers
scope only; the canonical statistical result was not re-validated. No fix was
applied, and this subsection is the only repository change.

**Method and safety.**
- The canonical lockbox was never run and nothing was re-split.
- Mutations and probes ran in a scratch copy of `src/`, `tests/`, `configs/`,
  `notebooks/`, `pyproject.toml` and `uv.lock`, with no `artifacts/`. An
  import-time canary proved the copy's code was the code under test. After
  every mutation the file was restored and verified byte-identical; at the end
  the copy was diffed clean against the repository.
- Probes used `bundle_spy` fixtures and temporary SQLite stores. One probe
  called the guard helpers only (no replay) on a scratch copy of `mlflow.db`.
- The 338 evidence checksums and `git status --porcelain` were compared at
  start, after the baseline, and at the end (see "End state").

**Baseline reproduced.**
- Full suite: **689 passed** in 609 s, matching the record.
- The Bayesian recovery warning is the documented one (worst R-hat
  1.086056330170683, minimum ESS 25.660557049567544). The other 12 warnings in
  the run are MLflow integer-schema hints, so "one documented warning" means
  one *Bayesian* warning.
- Focused suite: 134 tests.
- Ruff on the touched files and `marimo check` are clean.

### Verdict: **NOT YET ACCEPTED** (one High defect, one claimed guarantee untested)

Every remediation except F1 does what its finding required. Every handoff
power proof reproduces. F1 is only partly closed: after an attempt has opened
the lockbox and failed, the guarded entry point still accepts a "retry" that
changes the refit models or seeds (R1 below). Gate 8 should not be marked
complete until R1 and R2 are fixed, or the user explicitly accepts them.

#### Verdict per finding

| Finding | Verdict | Basis |
|---|---|---|
| F1 retry after an opened lockbox | **Partially met** | A different decision or source CV run is refused after an opened attempt that ended FAILED, KILLED, RUNNING (hard crash) or deleted (probes P1–P3). An identical retry is allowed and linked, an attempt that never opened does not block, and the first run in a fresh store is not a retry (P2, P4, P5). On a copy of the canonical store, every new attempt is refused (P6). **Open: R1, R2.** |
| F2 decision bound to the rule | Met (test gaps R5) | `verify_pretest_freeze` compares the full `to_dict()` (all nine `SelectionFreeze` fields) and runs before the refit and before any run opens. The lifecycle and unit refusal tests patch `select_cross_family_winner`. Binding to the real rule is proven by the identity test, and the slow `gate8_final_run` fixture runs the real rule and the real check. |
| F3 fresh-process LightGBM crash | Met | In fresh interpreters, `lightgbm` enters `sys.modules` before `torch` for `age_group_prediction`, `models.bayesian_conditional` and `models.bayesian_inference` imported first, `experiment`, `tracking`, `tracking.pyfunc_model` and `tracking.final`. The fresh-process test has power: removing `code_paths` fails it, and so does removing `code_paths` while putting the repository's `src` back on `PYTHONPATH`, through its `is_relative_to` assertion (E17a, E17b). |
| F4 children tagged `locked` | Met | Context tags override the supplied tags in `_terminating_run`, and `_log_final_candidate_run` is the only child path. |
| F7 label-only full-profile check | Met (test gap R5) | `_THRESHOLD_STRICTER_DIRECTION` covers all six `BayesianDiagnosticConfig` threshold fields, and each direction matches `evaluate_stage_diagnostics` (`bayesian_inference.py:72-98`). The draw counts are the config's Python ints, so no numpy-type false refusal occurs. The canonical `refit_metadata.json` (4/1000/1000, thresholds equal to the defaults) passes the strict `<`/`>` floors. |
| F9 default-schema leakage check | Met | The only construction in `src/` (`final_evaluation.py:757`) passes `schema=schema`. |
| F11 notebook test could create the manifest | Met | The notebook runs from a temporary project copy. Mutation row 22 is caught. |
| F12 no manifest oracle | Met | The runtime check and the unit and slow tests compare against `split_manifest.holdout_building_ids`. |
| M2, M5, M13, M14, M20, M25 | Met | Rows 10, 11, 12, 13, 14 and 16 are caught by their named tests. The M5/M20 test wraps the real functions and asserts both order and logged content. The M2 decoy sits on the `DirectCohortModel` descriptor. |
| Deferrals F6, F8, F10, residuals 1 and 3 | Stand | No concrete harm found. R1 is distinct from residual 3: it goes through the guarded entry point itself. |

### Defects, ranked

**R1 — High (probe-confirmed). An "identical retry" can change the refit models
and seeds after test evidence exists.**

`tracking/runs.py:298-337` (`_retryable_opened_attempts`) compares only
`source_cv_run_id` and `cross_family_decision_hash`. The decision hash covers
the cross-family decision alone. Nothing in the retry check covers what
`run_final_evaluation` (`tracking/final.py:291-357`) actually refits and
scores.

Probes (bundle_spy, temporary store; attempt 1 opened the lockbox, then
failed):
- **P7a/P7c.** The same pretest freeze and source run, with
  `final_refit_factories` giving the selected candidate a different model,
  complete as a linked retry. The selected candidate's test metrics change
  (1.70 → 1.10 and 2.728 → 1.823).
- **P7b.** A `cv_result` whose freeze carries `master_seed=32` (provenance
  records 31), with the same decision, is accepted by `verify_pretest_freeze`
  (it is self-consistent) and completes. Every refit and evaluation seed
  differs.

Why nothing refuses it:
- `_verify_refit_preconditions` checks candidate *descriptors*. The canonical
  descriptor records only `family` or `profile` (`candidate_registry.py:190,
  200, 212`), never hyperparameters, priors or tuning, and never inspects the
  factory's model.
- No check compares `cv_result.freeze.master_seed` with
  `cv_result.provenance.master_seed`, or `cv_result` with the logged
  `source_cv_run_id`.

A person who saw attempt 1's partial test evidence could therefore re-tune,
re-prior or re-seed the selected family through the sanctioned API. That
violates boundary 2 ("numerical hyperparameters, priors … prediction
settings") and §4.7, and contradicts "identical retry" in the docstring,
README, plan and record. It is not realized in the canonical store, which
refuses every new attempt (P6).

**R2 — Medium (sabotage-confirmed, test power). Deleted opened attempts are a
claimed but untested guarantee.** Mutation E3 drops
`run_view_type=ViewType.ALL` from `_retryable_opened_attempts`, so a deleted
FAILED or KILLED opened attempt no longer blocks a different decision. All 134
focused tests pass. The docstring (`runs.py:312-313`), the record's F1 row and
the handoff table all claim it. The code is correct today (probe P3); only
`_refuse_duplicate_final_run`'s deleted-run search is tested (row 17).

**R3 — Low (sabotage-confirmed, test power). The selected-tag vs decision check
is untested.** Mutation E7 disables the check in `tracked_final_evaluation`
(`runs.py:189-196`) and all tests pass. `run_final_evaluation` derives the tags
from the freeze, so only direct callers of `tracked_final_evaluation` could
mislabel a run.

**R4 — Low (sabotage-confirmed, test power). The semantics for an attempt that
never opened are untested.** Mutation E2 drops the `test_lock_status='opened'`
filter, which would make a pre-open failure block a different decision, and all
tests pass. The current behavior, allowing it (probe P4), is correct: no test
evidence exists yet.

**R5 — Low (sabotage-confirmed, test power). Field coverage and boundaries.**
- E10 and E10b: excluding `candidate_descriptors` or `fold_identities` from
  `verify_pretest_freeze`'s comparison passes, because only `master_seed` and
  `selections` are tampered with.
- E11: accepting one draw below the floor passes, because the refusal cases
  use 2 and 150.

**R6 — Low (note; not realized).** Within-approach `criterion_values` have no
finiteness check (`selection.py:246-268`). A NaN there would make
`verify_pretest_freeze`'s dict comparison falsely refuse, but only for a freeze
rebuilt from JSON: identical in-memory float objects compare equal. The
notebook and the slow fixture use in-memory objects.

### Proof results (scratch copy, focused suite unless noted)

| Row | Result |
|---|---|
| 1 | Caught: both "…after the lockbox opened is refused" tests |
| 2 | Caught: `…refuses_a_freeze_that_is_not_this_cv_results[master_seed]`, `[selections]` |
| 3 | Caught: both `…refuses_a_decision_the_rule_did_not_produce` |
| 4 | Caught: `test_run_final_evaluation_refuses_a_decision_the_rule_did_not_produce` |
| 5 | Caught: `test_logged_pyfunc_loads_and_predicts_in_a_fresh_process` |
| 6 | Caught: `test_final_children_are_tagged_as_opened_test_evidence` |
| 7 | Caught: 11 failed and 23 errors (the runtime manifest check fires) |
| 8 | Caught: `…scores_exactly_the_manifest_holdout`, `…refuses_a_replay_whose_holdout_is_not_the_manifest` |
| 9 | Caught: `…refuses_a_replay_whose_holdout_is_not_the_manifest` |
| 10 | Caught: `test_total_metrics_declared_by_direct_cohort_never_enter_the_cohort_reduction` |
| 11 | Caught: `test_run_final_evaluation_logs_the_decision_before_the_lockbox_opens` |
| 12 | Caught: `…refuses_a_target_column`, and also `…checks_leakage_against_the_schema_it_is_given` |
| 13 | Caught: `test_log_final_model_refuses_a_pyfunc_that_does_not_reproduce` |
| 14 | Caught: the event-order test |
| 15 | Caught: `test_a_pretest_freeze_without_a_decision_never_opens_a_final_run` |
| 16 | Caught: `test_refit_rejects_altered_training_values_with_the_same_ids` |
| 17 | Caught: `test_a_deleted_complete_final_run_still_refuses_a_second_evaluation` |
| 18 | Caught: all 4 fewer-draws cases |
| 19 | Caught: all 6 looser-threshold cases |
| 20 | Caught: `…checks_leakage_against_the_schema_it_is_given` |
| 21 | Caught: `…carries_the_schema_it_evaluated_with` |
| 22 | Caught: `test_default_script_execution_performs_setup_only`, run in a copy with no `artifacts/lockbox/`; the manifest it created stayed inside the copy and was deleted |

Extra mutations:

| # | Mutation | Result |
|---|---|---|
| 5b | Swap the imports in `tracking/pyfunc_model.py` only | Uncaught; **equivalent**, because `models.base` loads the package `__init__` first |
| E1 | `_retryable_opened_attempts` returns `[]` | Caught: identical-retry test |
| E2 | Drop the `opened` filter | **Uncaught** (R4) |
| E3 | Drop `ViewType.ALL` in the retry search | **Uncaught** (R2) |
| E4 | Drop the decision-hash comparison | Caught: different-decision test |
| E5 | Constant decision hash | Caught: different-decision test |
| E6 | Omit the `cross_family_decision_hash` tag | Caught: identical-retry test |
| E7 | Disable the selected-tag mismatch check | **Uncaught** (R3) |
| E9 | Remove the `test_metrics` refusal in `verify_pretest_freeze` | Caught: `…refuses_a_finalized_freeze` |
| E10, E10b | Exclude descriptors, or fold identities, from the freeze comparison | **Uncaught** (R5) |
| E11 | Draw floor off by one | **Uncaught** (R5) |
| E12a–f | Flip each threshold direction | Each caught by its own looser-threshold case; rhat and ESS also by the stricter-accept test |
| E13 | Threshold reference taken from the recorded value itself | Caught: all 6 looser cases |
| E14 | `>` → `>=` on the thresholds (the defaults would be refused) | Caught: 7 failed, 39 errors |
| E15 | Manifest check made tautological (IDs from `split.test_df`) | Caught: replay-mismatch test |
| E17a | `code_paths` removed | Caught: fresh-process test |
| E17b | `code_paths` removed and the repository `src` put on `PYTHONPATH` | Caught: `is_relative_to` assertion |
| E18 | `check_exact=False` in `_log_final_model` | Caught: M14 drift test |

Probes: P1 KILLED, P2 RUNNING (a store run left open after a crash), P3
deleted FAILED, P4 never opened, P5 first run, P6 canonical store copy, P7a/b/c
retry variations (R1).

### Withdrawn concerns

- **F3 test isolation.** The venv's `.pth` files pointing at `src` carry the
  macOS `hidden` flag, which Python 3.13 skips. Even if they were active,
  E17b shows the `is_relative_to` assertion would fail rather than pass
  falsely.
- **F7.** Missing threshold keys, wrong directions, numpy-int draw counts,
  `None` default thresholds, and refusal of the canonical refit: all ruled out
  (see F7 above).
- **E8 (retry search seeing the run just opening).** Not a hazard. The search
  runs before `mlflow.start_run`, and a new run is tagged `opened` only after
  its pretest freeze is logged.
- **Tag-filter quoting.** The IDs are hex, and the fingerprint is a SHA-256.
- **Legacy canonical parent without a decision hash.** Its duplicate check
  refuses first; any other attempt is refused as a mismatch (P6).

### Documentation notes

- The "identical retry / identical decision" wording overstates the guarantee
  while R1 stands. It appears in:
  - `README.md:279-281`;
  - the `tracked_final_evaluation` docstring;
  - `MODELING_REBUILD_PLAN.md:2098-2099`;
  - the record's F1 row.
- The record's F1 row and the handoff table describe deleted opened attempts as
  searched. That is true, but untested (R2).
- `README.md:283` has an unwrapped over-long line (cosmetic).
- The suite's "one documented warning" should read "one documented Bayesian
  warning (plus MLflow integer-schema hints)".
- `docs/README.md:94` correctly says completion awaits this review. Its rows
  19–20 call the historical handoffs "complete", which refers to the documents,
  not to the gate.

### Proposed fixes (not applied)

1. **R1: bind retries to what is actually refit and scored.**
   - In `run_final_evaluation`, after the refit and before
     `tracked_final_evaluation`, compute a `final_attempt_fingerprint`. It is
     the SHA-256 of the canonical JSON of:
     - `pretest_freeze.to_dict()`;
     - for each refit artifact: candidate ID, approach, seeds,
       `training_data_hash`, `training_schema_hash`, and the model's full
       configuration (verify `get_metadata()` carries it, or add it);
     - each selected candidate's descriptor and `prediction_config`;
     - `evaluation_config`;
     - the schema.
   - Tag every final parent with the fingerprint. In
     `_retryable_opened_attempts`, require it to match (as well as the
     decision hash and source run).
   - In `_verify_refit_preconditions`, also require
     `cv_result.freeze.master_seed == cv_result.provenance.master_seed`.
     Ideally, also compare `cv_result`'s manifest fingerprint, training hash
     and seed with the tags on the `source_cv_run_id` run.
   - Tests: P7a (different factories) and P7b (different seed) are refused
     after an opened attempt; an identical retry is still allowed and linked.
     Prove each fails without the fix.
2. **R2.** Add a test in which a deleted opened FAILED attempt still blocks a
   different decision (probe P3); prove it fails on E3.
3. **R3–R5.** Add tests for:
   - a selected-tag mismatch refused before any run opens (E7);
   - a pre-open failure not blocking a new decision (E2);
   - `verify_pretest_freeze` tampering parametrized over every freeze field
     (E10, E10b);
   - a draw count exactly one below the floor refused (E11).
4. **R6 (optional).** Refuse non-finite criterion values in
   `selection._criterion_value`.
5. **Documentation.**
   - After R1, state what a retry must match.
   - Record R2's test.
   - Wrap `README.md:283`.
   - Clarify the warning count.

### End state

- All 338 evidence checksums (`mlflow.db`, `artifacts/lockbox/split_manifest.json`,
  `mlartifacts/`) are identical to the start of the review.
- `git status --porcelain` is identical: 11 modified and 42 untracked. This
  file was already untracked, so its edit does not change the listing.
- The scratch copy's `src/` diffs clean against the repository.
- `src/student_simulator/` is untouched. No commits were made.

### Remediation applied — Gate 8 remediation review

The user approved the recommendation: fix R1 and R2, add the R3–R5 tests,
and correct the documentation. R6 stays a note. The reviewer applied the fixes,
so the independence gap this review was meant to close reopens for this pass
alone. That is why the fixes were independently reviewed by a read-only
subagent (below), and Gate 8 is marked complete only on the user's acceptance.

| Defect | Fix | Guarding test(s) |
|---|---|---|
| R1 | `experiment/final_evaluation.py::final_attempt_fingerprint` hashes everything an attempt refits and scores: the pretest freeze; each refit's seeds, training hashes, constructor configuration and recorded dependency versions; the selected candidates' descriptors and prediction settings; the evaluation settings; and the schema. The configuration comes from the new `BaseAgeGroupModel.configuration_record()`, which never includes fitted values, with `_model_configuration` implemented by the three real models and `BundleSpyModel`. `run_final_evaluation` computes it after the refit and before any run opens. `tracked_final_evaluation` requires it, tags `final_attempt_fingerprint` (a reserved tag), and `_retryable_opened_attempts` requires it to match. `_verify_refit_preconditions` also requires the freeze's master seed and seed source to equal the CV provenance record. | `test_a_retry_that_refits_different_models_after_the_lockbox_opened_is_refused`, `test_a_retry_with_a_different_master_seed_after_the_lockbox_opened_is_refused`, `test_an_identical_guarded_retry_after_the_lockbox_opened_is_allowed_and_linked`, `test_a_different_attempt_fingerprint_after_the_lockbox_opened_is_refused`, `test_final_attempt_fingerprint_is_stable_and_covers_what_is_refit_and_scored`, `test_real_models_record_their_constructor_configuration`, `test_refit_rejects_a_frozen_seed_that_is_not_the_cv_provenance_seed[*]` |
| R2 | Test only. | `test_a_deleted_opened_attempt_still_blocks_a_different_decision` |
| R3 | Test only. | `test_selected_tags_that_do_not_match_the_decision_never_open_a_final_run` |
| R4 | Test only; the intended semantics are that a failure before opening blocks nothing. | `test_an_attempt_that_failed_before_opening_does_not_block_a_new_decision` |
| R5 | Test only. | `test_verify_pretest_freeze_refuses_a_freeze_that_is_not_this_cv_results`, now over all six Gate 6 fields; `test_refit_refuses_a_full_label_with_fewer_draws_than_the_full_profile[chains=3]` |

**Power proofs** (scratch copy, focused suite, each mutation applied alone and
then restored):

| # | Mutation | Result |
|---|---|---|
| N1 | Retry guard ignores the fingerprint | Caught: re-tuned, re-seeded, and different-fingerprint tests |
| N2 | Fingerprint omits the model configuration | Caught: fingerprint unit test, re-tuned retry test |
| N3 | Fingerprint omits the evaluation settings | Caught: fingerprint unit test |
| N4 | Fingerprint omits the separate prediction settings | Uncaught; **equivalent**, because the hashed descriptor already carries `prediction_config` |
| N5 | Fingerprint omits the schema | Caught: fingerprint unit test |
| N6 | Fingerprint omits both the pretest freeze and the refit seeds | Caught: fingerprint unit test, re-seeded retry test |
| N6a, N6b | Omits only one of the two | Uncaught; **equivalent**. Refit seeds derive from the master seed in the freeze, and every other freeze field is pinned before opening by `verify_pretest_freeze` and the decision hash. |
| N7 | Seed-provenance check disabled | Caught: both `…frozen_seed_that_is_not_the_cv_provenance_seed` cases |
| N8 | `configuration_record` omits `default_rng_seed` | Caught: real-model configuration test |
| N9a–c | One real model's `_model_configuration` loses its config | Caught: real-model configuration test (each) |
| N10 | `run_final_evaluation` passes a constant fingerprint | Caught: re-tuned and re-seeded retry tests |
| N11 | Fingerprint includes a per-process nonce | Caught: fingerprint stability and identical-retry tests (proves they are not vacuous) |
| N12 | Fingerprint omits dependency versions | Caught: fingerprint unit test ("dependency versions" case) |
| E2, E3, E7, E10, E10b, E11 | The previously uncaught review mutants | All caught now, each by its new test or case |

**Independent review of the fixes** (read-only subagent; no pytest, no MLflow,
no repository writes): accept with conditions, nothing blocking.
- It confirmed R1 is closed through `run_final_evaluation`. Every other input
  that changes test numbers is covered by the fingerprint or by an existing
  hash check.
- The canonical registry's fingerprint inputs hash identically under three
  `PYTHONHASHSEED` values, and the real models' records serialize with
  `allow_nan=False`.
- Every caller passes the fingerprint.
- The retry refusals come from the real guard, not from a monkeypatch.

Findings and their disposition:
- **M1 (Medium), fixed.** Library versions were not fingerprinted. The
  dependency versions recorded in each state bundle are now included, guarded
  by the fingerprint unit test's "dependency versions" case. Model source code
  is still not hashed; `README.md` and the docstring state that a retry
  assumes the same package code.
- **L1, documented.** A final run that opened the lockbox before the
  fingerprint existed has no fingerprint tag, so every later attempt on its
  manifest is refused (fails closed). The canonical final parent is complete,
  so the duplicate check already refuses any repeat.
- **L2, note.** The guard trusts a model's own `configuration_record`. A
  deliberately forged model class is outside the threat model.
- **L3, note.** Any non-empty fingerprint string is accepted by
  `tracked_final_evaluation`. Only `run_final_evaluation` computes a real one.
- **L4, note.** The identical-retry lifecycle test runs within one process. The
  reviewer checked cross-process stability by hand.

**Documentation.** Corrected:
- `README.md` Gate 8: what a retry must match, the source-code assumption,
  legacy runs without a fingerprint, the seed-provenance check, and the
  over-long line.
- The docstrings of `tracked_final_evaluation`, `_retryable_opened_attempts`
  and `final_attempt_fingerprint`.
- `MODELING_REBUILD_PLAN.md` Gate 8 record.

**Suite (after the review's remediation).**
- Full suite: **705 passed** (689 + 16 new tests and cases) in 648 s.
- The documented Bayesian recovery warning is unchanged (worst R-hat
  1.086056330170683). The remaining 12 warnings are MLflow integer-schema
  hints.
- Focused Gate 8 unit suite: 150 tests. The slow real-model Gate 8 tests pass,
  so the notebook-equivalent `run_final_evaluation` path works with real
  models.
- Ruff is clean on every touched file, and `marimo check` is clean.
- All 338 evidence checksums are identical to the start of the review, and
  `git status --porcelain` is unchanged (11 modified, 42 untracked).
  `src/student_simulator/` is untouched, and no commits were made.
- The canonical evidence predates both remediation passes. It was verified
  independently, never re-run.

**Status.** R1–R5 are remediated and independently reviewed; R6 and L2–L4
remain notes. The user accepted this remediation on 2026-09-14, and **Gate 8 is
complete**.

The canonical evidence predates both remediation passes and was verified
independently, never re-run.

Accepted deferrals:
- F6, F8 and F10;
- the retry guard's single-experiment, single-store scope;
- `tracked_final_evaluation` and `evaluate_frozen_models_on_lockbox` staying
  public without the F2 check;
- model source code not being part of the final attempt fingerprint.

## Gate 9 — Removal, Guides, And Final Acceptance

**Reviewed:** 2026-09-15 against the Gate 9 implementation and Phase 4
acceptance handoffs, plan Sections 3.2, 14, and 15.

### Verdict: **ACCEPT**, pending user approval

Gate 9 removed exactly the approved obsolete surface and did not change current
modeling or simulator behavior. The authoritative suite passed **695 tests with
13 warnings in 736.28 seconds**, exactly ten fewer than the pre-deletion 705
baseline. Both package archives and clean installed-wheel environments exclude
the deleted modules and exports. The optional tracking boundary skips cleanly
without MLflow and passes against an isolated scratch store with MLflow.

No implementation defect was found during Phase 4. One wheel-test harness
issue was withdrawn: seven registry tests initially failed because they load
`configs/modeling.toml` relative to the current working directory and the first
probe ran from the scratch directory. The owning 36-test file passed from the
documented project working directory while `age_group_prediction` still
resolved from the installed wheel. This was not a package or behavior defect.

Every Section 14 row and Section 15 clause has an explicit evidence mapping in
the rebuild plan and Phase 4 acceptance record. Clause 5 relies only on the
accepted Gate 8 record and checksum-protected canonical evidence; Phase 4 did
not rerun or replay the final path.

The read-only independent Phase 4 review returned **ACCEPT** with no findings,
unsupported claims, unresolved risks, or required changes. The exact Phase 0
manifest procedure was then rerun: both manifest files were byte-identical,
all 336 artifact tuples matched, and the database, lockbox, artifact-manifest,
and combined-manifest hashes remained unchanged. Only explicit user approval
remains; the separate post-completion Opus 5 audit has not begun.

> **Update (2026-09-15).** The user approved Phase 4. The post-completion
> audit then ran; its record follows. The paragraph above is kept as written.

### Gate 9 — Independent post-completion validation

**Reviewed:** 2026-09-15 by Opus 5 under
`GATE_9_INDEPENDENT_VALIDATION_HANDOFF.md` (archived).
Full record, command ledger, and evidence:
[GATE_9_INDEPENDENT_VALIDATION_REPORT.md](GATE_9_INDEPENDENT_VALIDATION_REPORT.md).

#### Verdict: **ACCEPT**

The deletion delta was re-derived from the Phase 0 backup and matches the
approved removals exactly; `src/student_simulator/` is unchanged; collection
reconciled 705 → 695 at entry. The rebuilt wheel is byte-identical to Phase 4.
A clean core wheel passed 600 non-slow tests with tracking skipping cleanly,
and the tracking wheel passed 269 tests on a scratch store. Independent probes
confirmed result validation, grouped/literal-expansion equivalence, holdout
poisoning invariance, freeze refusals, state-bundle refusals and exact reload,
exact Pyro posterior reconciliation, and the guide's scratch MLflow example.
All eight Section 15 clauses are independently supported. Protected evidence
was byte-identical at entry; exit comparison is recorded in the report.

#### Findings, ranked

- **M1 (Medium):** the notebook header, README, and both guides said plain
  notebook setup never replays the holdout, but setup always loads (or first
  creates) the persisted manifest and replays it in memory. No holdout row is
  displayed or evaluated. *Documentation remediated; notebook behavior
  unchanged; changing setup behavior is a user decision.*
- **M2 (Medium):** the modeling guide misdescribed the direct-cohort selection
  policy (it ranks summed per-cohort parametric NLL, then mean cohort RMSE).
  *Remediated.*
- **M3 (Medium):** no test detected disabling `PredictionResult`'s
  means-reconciliation refusal (663 non-slow unit tests still passed).
  *Remediated with one parametrized regression case, proven to fail under the
  mutation; suite 695 → 696.*
- **L1-L12 (Low):** documentation inaccuracies in the modeling guide (loader
  strictness, holdout-fraction cause, categorical state, fold validation,
  checkbox label), MLflow guide (full-profile attribution, fold count),
  Bayesian deep dive (PyTorch `probs` naming, Pyro `DirichletMultinomial`,
  prior-predictive ordering), plan ownership table, and README. *Remediated.*
- **L13 (Low):** stale `_fold_coverage` docstring in `experiment/partitions.py`.
  *Not remediated (source-only wording); recommended for a later code change.*
- **L14 (Low):** the Phase 4 scratch retains a superseded first final
  comparison (non-Phase-0 serialization, `overall_pass=FAIL`); the Phase 4
  record reports only the exact, fully matching rerun. Integrity unaffected.
- **L15 (Low):** status lines still said Phase 4 awaited user approval.
  *Records updated.*

#### Withdrawn concerns

Child-expansion suspicion in `grouped_multinomial.py`, untested holdout guards
(mutually backstopping), "deferred" wording in completed historical plans,
Phase 4's seven core-wheel cwd failures (not reproduced), and the tracking
wheel's extra warnings (harness-induced marker warnings). Evidence is in the
report's Section 4.
