# Gate 9 Independent Validation Report

> **Status: complete — verdict ACCEPT, pending user approval (2026-09-15).**
> Controlled by
> `GATE_9_INDEPENDENT_VALIDATION_HANDOFF.md` (archived).

## 1. Validator, Scope, And Environment

| Item | Value |
|---|---|
| Validator | Opus 5 (`claude-opus-5[1m]`), separate post-completion audit |
| Date | 2026-09-15 |
| Phase 4 status at entry | Complete, independently accepted, and approved by the user on 2026-09-15 |
| HEAD | `b712cafd0efc360c2ac3079a30c66a25a797f2fd` (worktree intentionally dirty; rebuild untracked) |
| Entry status | `git status --porcelain=v1 -uall` SHA-256 `0dc72b8e…7e89eed`, byte-identical to the Phase 4 final status |
| Platform | macOS Darwin 25.6.0 ARM64; project Python 3.13.9; `uv` 0.11.26 |
| Scratch root | `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/24468d8b-9c60-407c-8b26-01f37e0a20a4/scratchpad/gate9-opus5-validation-20260915T121921Z` |
| Scratch MLflow stores | `tracking-store/` (wheel tracking tests) and `guide-store/` (guide example); no client touched the canonical store |
| Clean environments | `core-venv/` (wheel + pytest 9.1.1, no MLflow/statsmodels), `tracking-venv/` (same wheel + MLflow 3.16.0) |

An earlier Opus 5 attempt left only an entry snapshot at
`/var/folders/x3/z1qrvczj2x93k66m6qksg1yr0000gn/T/gate9-opus5-validationy5qk4efb`
(its hashes match the baseline). It was recorded, not reused, and not modified.

Safety: no final-evaluation function or notebook final action was called; the
canonical manifest was never replayed or persisted by this audit; no MLflow
client was opened on `mlflow.db`; `mlflow.db`, `mlartifacts/`,
`artifacts/lockbox/`, and `src/student_simulator/` were not modified; no commit
or branch was created; one heavy process ran at a time.

## 2. Verdict

**ACCEPT.** Every Section 15 clause is independently supported, the Gate 9
deletion delta reconciles exactly, and protected evidence is byte-identical at
entry and exit. No Blocker or High finding exists. The three Medium findings
(M1-M3) and twelve Low documentation findings were remediated without any
modeling-behavior or source change. The added regression case moved the
authoritative suite to **696 passed, 13 warnings**. The independent review of
the audit and remediation recommended ACCEPT WITH CONDITIONS; its conditions
were the then-unfinished suite, exit comparison, placeholders, and four wording
corrections, all now discharged (Section 12). Unresolved items are one Low
source-docstring inaccuracy (L13) and an optional notebook-behavior decision
(Section 13), neither material to completion.

## 3. Findings (Ordered By Severity)

No Blocker or High finding.

### Medium

**M1 — Notebook setup replays the canonical manifest, contrary to the notebook
header and README and left implicit by both guides.**
- *Contract:* documented plain-script and tutorial safety semantics.
- *Location:* `notebooks/02_model_fitting.py` header (former lines 36-39: "no
  holdout row is replayed"); `README.md` notebook controls ("never … replays
  the holdout"); `docs/MLFLOW_EXPERIMENTS_GUIDE.md` Section 4 (scratch variables
  imply isolation; only the final controls said to replay);
  `docs/MODELING_GUIDE.md` Section 15.
- *Inspection:* the setup cell (`02_model_fitting.py`, manifest cell) always
  loads `project_root/artifacts/lockbox/split_manifest.json`, calls
  `persist_split_manifest` when it is absent, and then calls
  `replay_split_manifest(modeling_df, split_manifest, …)`, which materializes
  `split.test_df` in memory. The scratch MLflow variables do not redirect this
  path. No pre-final cell displays or evaluates holdout rows (static
  inspection; `test_notebook_never_displays_the_holdout_frame`), and
  `test_default_script_execution_performs_setup_only` runs in a temporary
  project copy.
- *Impact:* no leakage or lifecycle breach, but a reader following the guide
  replays, and on a fresh checkout creates, the canonical manifest while
  believing it is isolated.
- *Remediation:* documentation-only corrections in all four places, including
  the notebook's markdown header text; notebook logic unchanged. **Remediated.**
- *Revalidation:* `marimo check` clean on the edited notebook; notebook
  characterization 7 passed after the edit (temporary project copy); link
  check 0 missing; stale-text scan clean; included in the 696-test suite.
- *User decision (optional):* whether setup should stop replaying the holdout
  in memory (for example, derive the training partition from manifest IDs).
  That is a notebook behavior change and was not made.

**M2 — Modeling guide misstates the direct-cohort selection policy.**
- *Location:* `docs/MODELING_GUIDE.md` Section 10 ("common composition and
  cohort errors").
- *Inspection:* `experiment/policies.py` `direct_cohort_selection_policy` ranks
  by `independent_cohort_joint_nll` (sum of per-cohort
  `ParametricPredictiveNegativeLogLikelihood`, total excluded), then
  `mean_cohort_rmse`; no composition metric. Composition log loss belongs only
  to the cross-family rule.
- *Remediation:* corrected. **Remediated.**

**M3 — No non-slow unit test detected removal of the means-reconciliation
refusal in `PredictionResult`.**
- *Contract:* shared-API invariant "cohort means reconcile to the total"
  (Section 14 Shared API; Section 15 clause 1).
- *Location:* `src/age_group_prediction/results.py`
  (`if (actual_error > …reconciliation_tolerance).any(): raise`).
- *Reproducer:* scratch-copy sabotage replacing the check with `if False:`:
  `tests/unit/test_model_contracts.py` 34/34 passed and the whole non-slow unit
  suite 663/663 passed. The adversarial probe proved current code refuses
  truthfully reported, unreconciled means ("Cohort and total means do not
  reconcile").
- *Remediation:* one parametrized case added to
  `test_prediction_result_rejects_invalid_values` (total 3.5, cohorts 1+1+1,
  reported error 0.5). Test-only; no source change. Collection 695 → 696.
- *Revalidation:* under the scratch-copy mutation the new case fails
  (`[changes4-do not reconcile]`: 1 failed, 4 passed); on current code the owning
  file passes 35/35 and scoped ruff is clean; collection 696; authoritative
  suite 696 passed, 13 warnings. **Remediated.**

### Low

| ID | Location | Defect (verified in owning code) | Status |
|---|---|---|---|
| L1 | `MODELING_GUIDE.md` intro | Claimed missing TOML keys fail. Probe: removing `[outer_split].test_fraction` loads with default `0.2`; unknown keys are refused. Only top-level sections and Bayesian subsections are exact. | Doc remediated |
| L2 | `MODELING_GUIDE.md` Section 3 | Attributed holdout-fraction overshoot to singletons; singletons lower it, the per-neighborhood `max(1, min(round(n·f), n-1))` clamp raises it. | Doc remediated |
| L3 | `MODELING_GUIDE.md` Section 5 | Claimed fitted categorical state; `FittedFeatureTransformer` takes one-hot levels from the schema. | Doc remediated |
| L4 | `MODELING_GUIDE.md` Section 9 | "Overlapping folds" is imprecise; validation checks within-fold fit/validation disjointness, exact partition, unique fold indices, and manifest/holdout consistency. | Doc remediated |
| L5 | `MODELING_GUIDE.md` Section 15 | Wrong checkbox label; actual: "I understand this opens the one-time test holdout and cannot be undone for this manifest." | Doc remediated |
| L6 | `MLFLOW_EXPERIMENTS_GUIDE.md` Section 8 step 3 | Said `run_final_evaluation` forces the full Bayesian profile; the registry's final factory forces it and the refit refuses a non-full run. | Doc remediated |
| L7 | `MLFLOW_EXPERIMENTS_GUIDE.md` Section 4 | "Two folds" applied only to outer CV folds; direct-cohort inner tuning folds stay at `DEFAULT_FOLD_CONFIG` (five), which the TOML loader never sets (`experiment_config.py` passes no `tuning_folds` to `DirectCohortConfig`). The first remediation wording ("keep their configured count") was itself wrong and was corrected after independent review. | Doc remediated (revised) |
| L8 | `BAYESIAN_CONDITIONAL_MODEL.md` NB parameters | Named the complementary probability as PyTorch's; PyTorch `probs = sigmoid(logits) = μ/(φ+μ)`. Mathematics unchanged. | Doc remediated |
| L9 | `BAYESIAN_CONDITIONAL_MODEL.md` Pyro syntax | `DirichletMultinomial` is Pyro's (`pyro/distributions/conjugate.py`), not PyTorch's. | Doc remediated |
| L10 | `BAYESIAN_CONDITIONAL_MODEL.md` fitting step 7 | Prior-predictive policy is applied before NUTS; under `action="error"` the summary is not stored. | Doc remediated |
| L11 | `MODELING_REBUILD_PLAN.md` Section 3 ownership table | Omitted five `experiment/` modules, `tracking/final` and `pyfunc_model`, and five top-level helpers. | Doc remediated |
| L12 | `README.md` notebook controls | "Calls only package APIs": its display cell uses the MLflow client for run links. | Doc remediated (with M1) |
| L13 | `experiment/partitions.py` `_fold_coverage` docstring | Says folds are "repeated overlapping splits"; `make_validation_folds` rotates, validating each non-singleton building once. | **Not remediated**: source docstring edit would change package bytes for no behavioral gain; recommended for a later code change |
| L14 | `GATE_9_PHASE_4_ACCEPTANCE_HANDOFF.md` Section 15 | The retained Phase 4 scratch shows a first final comparison (12:09, `overall_pass=FAIL`, non-Phase-0 serialization without the `mlartifacts/` prefix) superseded by the exact comparison (12:53, all byte matches). The record reports only the latter. Evidence integrity unaffected. | Recorded here and in the findings record |
| L15 | Plan, findings header, docs index, Phase 4 handoff | Still say "pending user approval" after the 2026-09-15 approval. | Records updated |

### Observations

- O1: `marimo check` exits 0 but reports one `empty-cells` warning at
  `notebooks/01_eda.py:2498`; identical in the Phase 0 backup (pre-existing).
- O2: Scoped ruff: `__init__.py` and `metrics.py` clean; the notebooks' 32 and
  16 findings are identical in the Phase 0 backup (marimo boilerplate).
- O3: Rebuilt wheel is byte-identical to Phase 4
  (`cf67d2eaa0f5…35fd9`); the sdist differs only in six Gate 9 record documents
  edited after the Phase 4 build.
- O4: Guide Sections 7 and 9 snippets resolve settings from the environment;
  without Section 2's scratch variables they would open the canonical store
  read-only. Section 2 already requires scratch variables.
- O5: The notebook's final display cell takes `final_parent_runs[0]`; after an
  allowed retry the displayed parent is arbitrary. Display-only, post-final.
- O6: `run_final_evaluation` checks for an already-active MLflow run only after
  the expensive refit (refusal is still before any run or holdout access).
- O7: `SplitManifest` does not itself require training/holdout disjointness.
  `validate_experiment_partitions` refuses a tampered overlapping manifest.
  The fold holdout guard (`partitions.py`, "Validation folds contain
  split-manifest holdout IDs") is logically redundant while the manifest-ID
  equality, outer-holdout, and exact-partition checks exist; the outer guard
  is live only for tampered manifests, and when it is disabled the fold guard
  refuses instead (probe). The outer-guard mutation passed all 663 non-slow
  unit tests; the fold-guard mutation was run only against its owning test.
  Neither is a durable coverage defect because leakage is still refused.
- O8: Handoff trees omit `configs/validation.toml`; handoff log-mean equations
  omit the intercept (labelled simplified); Model B temperature calibration
  and Poisson total option are not in the handoff summary.
- O9: The adversarial probe's "bundle contains no training rows" assertion was
  weak; that contract rests on `test_bundle_contains_no_training_rows`.
- O10: `MLFLOW_EXPERIMENTS_GUIDE.md` wording "a nonlocal tracking server
  decides" (Section 1) and "evaluation includes prediction time" (Section 6) are
  imprecise but not misleading in practice.

## 4. Withdrawn Findings

| Suspicion | Evidence that cleared it |
|---|---|
| `np.repeat` in `models/grouped_multinomial.py` is child expansion | It emits at most three weighted rows per building (`labels = tile(0..2)`, weights = counts); the literal-expansion oracle agreed to ≤1.3e-15 in logits and 0 in NLL at C ∈ {0.05, 1, 20} on independent data (27 grouped vs 76 literal rows). |
| The two holdout guards in `partitions.py` are untested | Single-guard mutations pass the suite because each guard backstops the other; the tampered-manifest probe is refused under either single mutation. Recorded as O7. |
| "Deferred" wording in `MARIMO_MIGRATION_PLAN.md`/`EDA_AND_PREDICTIVE_MODELING_PLAN.md` | Body text of completed historical steps; both status headers state completion. |
| Phase 4's seven core-wheel failures indicate a package defect | Not reproduced: from the project working directory the installed wheel passed 600 non-slow core tests (imports resolved to site-packages). |
| The Bayesian probe crash is a model defect | Probe bug (draws are requested with `n_predictive_draws`); the corrected probe passed. |
| Plain notebook execution can fit or open the lockbox | Both `mo.ui.run_button`s default to unclicked and final also needs the checkbox; `mo.stop` gates both; characterization tests cover it. |
| 28 tracking-wheel warnings exceed the profile | 12 MLflow integer-schema hints (the established category) plus 16 `PytestUnknownMarkWarning` caused by `-c /dev/null` dropping marker registration. |

## 5. Gate 9 Deletion Delta And 705 → 695 Reconciliation

Derived by comparing the Phase 0 backup
(`/tmp/gate9-phase0-20260915T062606Z-55442/backup`) with the current tree:

| Item | Evidence | Result |
|---|---|---|
| `benchmarks.py`, `direct_cohort_models.py`, `poisson_total_benchmark.py`, `gate3.py` | `diff -rq`: only in backup; absent from wheel (61 entries); `ModuleNotFoundError` from source and clean wheel | Removed |
| `tests/unit/test_baselines.py` | Only in backup | Removed |
| `compute_point_metrics`, `compute_poisson_deviance` | Only deletions in the `metrics.py` diff; absent from module and root | Removed |
| Ten root exports | `__init__.py` diff removes exactly the ten imports/`__all__` entries plus the obsolete statsmodels comment; `__all__` now 101 names, all resolvable | Removed |
| Superseded fitting plan | Only in backup; no active link (137 links, 0 missing) | Removed |
| Other source changes | None under `src/` besides the above; `src/student_simulator/` identical; `02_model_fitting.py` identical at entry | None |
| Other Gate 9 changes | `pyproject.toml` slow-marker description; `01_eda.py` intro text; documentation | Non-behavioral |
| Retained | `predictive.py`, 46 current modules (incl. `tracking/`), statsmodels validation group (`tests/validation/helpers.py`, `test_recovery.py`), EDA-only `HistGradientBoostingRegressor` | Present |
| Collection | 695 collected at entry; none of the ten Phase 0 `test_baselines.py` node IDs collected; Phase 0 recorded 705 | 705 − 10 = 695 |

After M3 remediation the collection is 696: the single added parametrized case,
with no other collection change.

## 6. Section 14 Verification Matrix

| Area | Focused tests | Independent evidence this audit | Verdict |
|---|---|---|---|
| Data/configuration | `test_modeling_config.py`, `test_modeling_data.py` | Config-strictness probe (unknown refused, missing defaulted: L1) | Pass |
| Data splitting | `test_modeling_data.py`, `test_resampling.py` | In-memory split probe: folds hold no holdout IDs (10 holdout, 31 fold IDs); rotation code inspection | Pass |
| Features | `test_feature_engineering.py` | Transformer state inspection (L3); 600-test core wheel slice | Pass |
| Shared API | `test_model_contracts.py` | Five `PredictionResult` refusals (dimension, non-finite, unnormalized, unreconciled, misreported error); M3 gap found and closed | Pass after remediation |
| Model A | `test_direct_cohort.py`, `test_tuning.py`, `test_resampling.py` | Direct-only CV in probe and guide example; marginal scope, no joint capability | Pass |
| Model B | `test_independent_total_probability.py`, `test_distributions.py`, `test_composition_kernels.py` | Independent grouped/literal oracle; α = 1/φ convention traced through `distributions.py`, `count_regression.py` | Pass |
| Bayesian | `test_bayesian_conditional.py`, `test_bayesian_recovery.py` | Real reduced-profile Pyro NUTS fit (2×150/150): 15×64 integer draws reconcile exactly; `sequential_joint` scope; NumPyro absent; equations re-derived (L8-L10) | Pass |
| Metrics | `test_metrics.py`, `test_evaluation.py` | Comparability guard traced (`metrics.py`, `evaluation.py`, `final_selection.py`) | Pass |
| Experiments | `test_experiment.py`, `test_final_selection.py`, `test_final_evaluation.py`, `test_experiment_artifacts.py` | Holdout-target poisoning (30 holdout cells) left freeze and fold metrics identical; freeze refuses test metrics before a decision; tampered manifest refused; freeze-replacement and bundle-version mutations detected | Pass |
| MLflow | `test_tracking.py`, `test_gate8_tracking.py`, `test_tracking_real_models.py` | 269 wheel tests on a scratch store (includes active-run refusal, location mismatch, failure/interrupt, deleted-run duplicate, retry fingerprint, locked→opened); guide example: parent and child `FINISHED` + `evidence_complete=true`, parent link, both reload checks `passed` at tolerance 0.0 | Pass |
| Notebook | `test_model_fitting_notebook.py` | `marimo check`; static gating inspection; preview drops targets; M1 remediated | Pass (characterization 7 passed after M1 edit) |
| Migration | Collection, archives, imports | Backup diff; clean core/tracking wheels; stale-reference and link audits | Pass |

## 7. Section 15 Clause Verdicts

| # | Clause | Independent evidence | Verdict |
|---:|---|---|---|
| 1 | Shared API and metric registry | Three model classes subclass `BaseAgeGroupModel` from source and wheel; result refusals probed; M3 regression added | Satisfied |
| 2 | Model B without literal child expansion | Grouped weighted rows only; no expansion API in `src/`; oracle equivalence on new data | Satisfied |
| 3 | Pyro Bayesian model, exact reconciliation | Real NUTS draws reconcile exactly; Pyro only; diagnostics policy traced | Satisfied |
| 4 | Selection uses only train/CV evidence | Partition refusals; holdout poisoning invariance; freeze transition refusals; `run_cross_model_validation` accepts only training data and manifest | Satisfied |
| 5 | One final evaluation from persisted lockbox | Accepted Gate 8 record plus byte-identical canonical evidence at entry and exit (Section 11); not replayed or rerun | Satisfied (historical evidence only) |
| 6 | Reproducible MLflow evidence, reconstructable artifacts | Wheel tracking matrix; scratch guide run with exact reload checks; bundle reload reproduces point predictions exactly | Satisfied |
| 7 | Notebook as tested client | No model fit/predict/evaluate calls; gated actions; documentation corrected (M1, L12) | Satisfied |
| 8 | Obsolete code removed without simulator change | Backup diff; simulator tests 73 passed; `src/student_simulator/` identical | Satisfied |

## 8. Documentation, Formula, Structure, And MLflow Audit

Three read-only audit agents compared every section of the modeling guide (15),
MLflow guide (10), and Bayesian deep dive, plus project trees and Mermaid flows,
against owning source. Every reported inaccuracy was independently confirmed in
code before being recorded (Section 3). Confirmed accurate, among others: all
environment variables and defaults; tag keys, metric prefixes, artifact paths,
completeness semantics, search filters, refusal messages; final-run ordering
(freeze logged while `locked`, `opened` before holdout replay, duplicates
counted across deleted runs, identical-retry rule); pyfunc input/output
contract and LightGBM-first import note; NB2 and Dirichlet-multinomial
parameterizations, posterior allocation, pointwise joint log mass; the plan's
50-module tree; no MLflow imports in `experiment/` or `models/`; guide Mermaid
flows. No guide contains a runnable canonical final-evaluation path.

## 9. Package, Dependency, And Clean-Import Results

| Check | Result |
|---|---|
| `uv build --out-dir <scratch>/dist` | Pass, 1.20 s; wheel SHA-256 identical to Phase 4 |
| Wheel archive | 61 entries; no obsolete module; `predictive.py`, models, experiment, tracking, simulator present |
| Core venv import (no `PYTHONPATH`) | `age_group_prediction` and `student_simulator` from site-packages; MLflow neither installed nor loaded; `tracking` import fails on `mlflow` as designed; four modules and ten names absent |
| Core venv tests (`-c /dev/null`, non-slow, tracking files excluded) | 600 passed in 25.38 s |
| Core optional boundary | `test_tracking.py`, `test_gate8_tracking.py`: 2 skipped via `importorskip("mlflow")` |
| Tracking venv (wheel + MLflow 3.16.0, scratch store) | 269 passed, 28 warnings in 511.08 s |
| Source-tree import probe | Removed surface absent; 46 current modules import; no MLflow loaded by core/experiment/models |

## 10. Command And Result Ledger

Retained Phase 4 evidence consumed (verified present and consistent):
`/var/folders/x3/z1qrvczj2x93k66m6qksg1yr0000gn/T/gate9-phase4-twrlmza7`
— `pytest_summary.txt` "695 passed, 13 warnings in 736.28s", exit 0, warnings
12 integer-schema hints + 1 Bayesian R-hat/ESS warning; build hashes;
68-test tracking log; core skip logs; `exact_final_comparison.txt` all true.
Phase 0 record `/tmp/gate9-phase0-20260915T062606Z-55442`: `step5.log` ten IDs,
`step6.log` 705 passed/13 warnings, manifest script. **Conditional
full-suite trigger at entry: inactive.** It later fired because of M3's test
addition (Section 12).

| Step | Command / method | Result | Duration |
|---|---|---|---|
| Entry manifests | Verbatim Phase 0 step-2 Python (SHA-256 of extracted script `3af68872…`) + combined = target lines 3-4 + artifact manifest | Five files byte-identical to Phase 0 | <5 s |
| Source import probe | `PYTHONPATH=src uv run --frozen --group tracking python v1_source_import_probe.py` | FAIL_COUNT 0 | 4.8 s |
| Collection | `uv run --frozen --group tracking --group validation --group notebook pytest --collect-only -q -p no:cacheprovider tests/unit tests/validation tests/characterization` | 695; removed IDs 0 | 5.3 s |
| Simulator slice | Characterization current/EDA, ten simulator unit files, calibration, recovery | 73 passed | 5.5 s |
| Backup diff | `diff -rq` Phase 0 backup vs tree | Section 5 | — |
| Links | Relative Markdown checker, 28 files | 137 links, 0 missing (entry and after edits) | <2 s |
| Stale references | `rg` over `src tests notebooks configs pyproject.toml README.md` and active docs | All hits classified (Sections 3-5) | — |
| Build | `uv build --out-dir <scratch>/dist` | Pass | 1.2 s |
| Marimo | `uv run --frozen --group notebook marimo check notebooks/01_eda.py notebooks/02_model_fitting.py` | Exit 0; O1 warning; clean after M1 edit | 1.3 s |
| Ruff (scoped) | `ruff check` on Gate 9 Python files and backup copies | O2 | — |
| Core wheel | See Section 9 | 600 passed; 2 skipped | 26.5 s |
| Tracking wheel | `tracking-venv/bin/python -m pytest -c /dev/null … test_tracking, test_gate8_tracking, test_final_evaluation, test_final_selection, test_experiment_artifacts, test_state_bundles, test_model_state_bundles, test_tracking_real_models` | 269 passed, 28 warnings | 511.08 s |
| Config probe | Load edited TOML copies | L1 | <10 s |
| Adversarial probe | `v3_adversarial_probe.py` (bounded simulator data, in memory) | 27 PASS; Bayesian section probe bug | 67 s |
| Bayesian probe | `v3_bayesian_probe.py` real reduced NUTS | 5 PASS | 50 s |
| Sabotage v1 | Five single mutations on scratch copies vs owning tests | Freeze-replace and bundle-version detected; reconciliation and both holdout guards not detected | ~20 s |
| Sabotage v2 | Reconciliation and outer-holdout mutations vs 663 non-slow unit tests; tampered-manifest probe | Reconciliation undetected (M3); holdout guards mutually backstopped (O7) | 390 s |
| Guide example | Guide Section 4 snippet verbatim with Section 2 scratch variables; scratch-store inspection | Pass; canonical `mlflow.db` mtime/size unchanged | 7.4 s |
| Independent review | Read-only review agent over report, remediation, and owning code | ACCEPT WITH CONDITIONS; conditions discharged (Section 12) | ~6 min |
| M3 proof chain | Scratch-copy mutation vs `test_prediction_result_rejects_invalid_values`; owning file; scoped ruff; `test_model_fitting_notebook.py`; `marimo check`; collection | New case fails under mutation (1 failed, 4 passed); 35 passed; ruff clean; notebook 7 passed; marimo exit 0 (O1 only); 696 collected | ~30 s |
| Authoritative suite (trigger fired by M3 test addition) | `MLFLOW_DISABLE_AGENT_HINT=1 uv run --frozen --group tracking --group validation --group notebook pytest -q -p no:cacheprovider -rw tests/unit tests/validation tests/characterization` | **696 passed, 13 warnings**, exit 0; twelve MLflow integer-schema hints (6 `test_gate8_tracking.py`, 6 `test_tracking_real_models.py`) and one Bayesian recovery warning, worst R-hat `1.086056330170683`, minimum ESS `25.660557049567544` | 656.95 s (665 s wall) |
| Exit manifests | Same verbatim Phase 0 Python into `evidence/exit/`; `cmp` against entry and Phase 0; exit status diff | All five files byte-identical; status delta is the new report only | <5 s |

## 11. Protected Evidence Before And After

| Evidence | Baseline | Entry | Exit | Verdict |
|---|---|---|---|---|
| `mlflow.db` SHA-256 | `5b1d0f34f7d6c9be68ecf59deeb803f9ebebffb00c3652a7998af4fe800bac09` | Same | Same | Match |
| Lockbox manifest SHA-256 | `1c6fe0be481c9cdb0675faabbbdd591a403dc89860eece84390fd284ff62f5d4` | Same | Same | Match |
| `mlartifacts/` regular files | 336 | 336 | 336 | Match |
| Artifact manifest SHA-256 | `e379f50379755c730c67472dc35b9001d969ddb978a9451d64dba005004f0cbc` | Same | Same | Match |
| Combined manifest SHA-256 | `553cc8c5bfc9804ca29e369cdcb628c76c25f78d2afa35aaa5be478fbef2e52c` | Same | Same | Match |

Both entry and exit used the verbatim Phase 0 step-2 Python serialization
(extracted from `run_gate9_phase0.sh`) and the proven combined composition.
All five files (`target_status.txt`, `mlartifacts_manifest.txt`,
`mlartifacts_full_digest.txt`, `mlartifacts_count.txt`,
`combined_evidence_manifest.txt`) are byte-identical entry-to-exit and
exit-to-Phase-0 (`cmp`). HEAD is unchanged. Exit `git status --porcelain=v1
-uall` differs from entry by exactly one line,
`?? docs/GATE_9_INDEPENDENT_VALIDATION_REPORT.md`; all other edits were to files
already listed as modified or untracked. No protected path or
`src/student_simulator/` appears in status, and `src/student_simulator/` is
identical to the Phase 0 backup.

## 12. Remediation And Independent Revalidation

| Change | Files | Type | Revalidation |
|---|---|---|---|
| M1, L12 notebook-setup replay wording | `notebooks/02_model_fitting.py` (markdown header only), `README.md`, `docs/MODELING_GUIDE.md` Section 15, `docs/MLFLOW_EXPERIMENTS_GUIDE.md` Section 4 | Documentation | `marimo check`; notebook characterization 7 passed; links; suite |
| M2, L1-L5 | `docs/MODELING_GUIDE.md` | Documentation | Owning-code inspection; config probe (L1); links |
| L6, L7 | `docs/MLFLOW_EXPERIMENTS_GUIDE.md` Sections 4 and 8 | Documentation | Owning-code inspection; L7 wording revised after review |
| L8-L10 | `docs/BAYESIAN_CONDITIONAL_MODEL.md` | Documentation | torch/Pyro source and `bayesian_conditional.py` inspection |
| L11 | `docs/MODELING_REBUILD_PLAN.md` Section 3 table | Documentation | Module listing |
| M3 | `tests/unit/test_model_contracts.py` (one parametrized case) | Test only | Fails under mutation; 35/35; collection 696; full suite (trigger fired) |
| L14, L15 and closure | Findings record, plan status and Section 15 note, docs index, Phase 4 handoff status, this handoff (historical) | Records | Links (0 missing); prior history preserved with "kept as written" notes |

No file under `src/` changed (empty diff against the Phase 0 backup for
`src/`), so the wheel and public API are unchanged; the conditional suite
trigger fired only because a test was added. The authoritative command ran
exactly once after remediation:
`MLFLOW_DISABLE_AGENT_HINT=1 uv run --frozen --group tracking --group validation --group notebook pytest -q -p no:cacheprovider -rw tests/unit tests/validation tests/characterization`
→ **696 passed, 13 warnings in 656.95 s** (665 s wall), exit 0; warnings were
the twelve MLflow integer-schema hints and the Bayesian recovery warning.

**Independent review.** A separate read-only review agent (no pytest, no
writes) re-verified M1-M3, every edited documentation claim against owning
source and the installed torch/Pyro sources, the Low, Observation, and
Withdrawn records, Sections 14-15, and boundaries (no `src/` change, notebook
diff limited to the markdown header, protected-file mtimes unchanged). It
recommended **ACCEPT WITH CONDITIONS**:

1. The first L7 fix wrongly said direct-cohort tuning folds "keep their
   configured count" — confirmed (`experiment_config.py` passes no
   `tuning_folds`; `configs/modeling.toml` has none); corrected in the guide
   and L7.
2. Pending suite, exit manifests, Sections 12-13, and record placeholders —
   discharged.
3. O7 imprecise on guard redundancy and mutation scope — rewritten.
4. M3 heading broader than its evidence — rescoped to the non-slow unit suite
   (the slow suite also passes with the check in place; no mutation was run on
   slow tests).
5. Observations: M1 heading overcounted contradicting documents (rescoped);
   Section 8 step 3 now mentions the strict Bayesian diagnostic policy; the
   regression-chain "Found 1 issue." is the pre-existing `01_eda.py` warning
   (O1; `02_model_fitting.py` alone checked clean).

These follow-up corrections are documentation wording, each confirmed against
owning code by the validator; none changes behavior or reopens a finding.

## 13. Residual Risks And User Decisions

1. **Optional notebook behavior (M1).** Setup still replays the canonical
   manifest in memory (and creates it on a fresh checkout). This is now
   documented. Changing it — for example deriving the training partition from
   manifest IDs without materializing holdout rows — would be a notebook
   behavior change requiring user approval and a regression test.
2. **L13.** The `_fold_coverage` docstring in `experiment/partitions.py` still
   describes overlapping splits. Recommended for the next source change.
3. **Suite baseline.** Downstream references should use **696 passing, 13
   warnings**. Historical records citing 695 or 705 are left as written.
4. **Guide read-only snippets (O4).** Running Sections 7 and 9 without the
   Section 2 scratch variables opens the canonical store read-only; users
   must follow Section 2.
5. No residual risk to protected evidence: entry and exit manifests are
   byte-identical (Section 11).
