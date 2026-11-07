# Plan: the study cache, the bootstrap replicates, the notebook split, SHAP per cohort and the test evaluation

**For:** the sessions that close PR 3 (3.3) and then build PRs 4–7, one sub-task per approval.
Kept updated with each sub-task's Done note; tick the sub-task and update the status line.

**Status (2026-11-04):** written after 3.4 (`afcd6f9`) and the handoff (`b7da88d`), with the user's
list of changes and their answers (§0). 3.3's checks already run: the non-slow suite 1387 passed,
1 skipped, 1 xfailed, 19 deselected (7 warnings, the old stack's); mypy 28 files, no issue. The
notebook's md5 is `676212a808f6c1f87f7926a58b3bffe6`. **Updated 2026-11-05:** 3.3 done (§2): the
review found no correctness bug; its F2 fixed in `aa1e267`, F1 and F5 in 3.3b's docs, F3 and F4
left for PR 6. The notebook's md5 is `95261326e5368bd451607d083a5ab366`. PR 3 closed and merged
as #21 (`0c94f16` on `main`); PR 4's branch `feat/study-cache` created from it. **Updated
2026-11-06:** 4.1 done (§3, the budget 100 / 10); the notebook's md5 is
`78b08bd7936dba06d65f18bd52da5cbb`. 4.2 done (the tuner's journal storage: a partial study
started fresh, the seconds the study's run). **Updated 2026-11-06 (paused):** 4.3 implemented, uncommitted; its Check C was running, the clicks remain (the handoff's §2 "PR 4"); the notebook's md5 `3ac16ed8995efa3899a6b37746cc99d4`. **Updated 2026-11-06:** 4.3 done (§3): the user's renames kept, a stored study loaded without writing to its file (a fix in `src`), the two clicks; the notebook's md5 `91d1c96cdd227a137163cd072235c64a`. **Updated 2026-11-06:** PR 4 merged as #22 (`30bd4f7` on `main`); PR 5's branch `feat/bootstrap-replicates` created from it; 5.1 planned and approved. **Updated 2026-11-06:** 5.1 done (§4, the rename); the notebook's md5 `6edc34aa4cc93d05fcf34b106515dd9c`. **Updated 2026-11-06:** 5.1 committed (`571f721`); 5.2 done and committed (`4cfe502`) (§4: the evaluator returns its replicates; two resampling designs kept, the user's answer); the notebook's md5 `d4a361a5535aaefd89b9dccd4f99e282` (with the user's `median` and `sd`). **Updated 2026-11-07:** 5.3 planned, not implemented (§4): the `n_resamples` measurement done (1000 kept); the user changed the plan (each model's best run chosen by one configurable selection metric on validation, only the two best bootstrapped, one forest-plot function); the plan awaits approval in the next session; 5.4 re-scoped. **Updated 2026-11-07:** 5.3 done as planned (§4): `selection_metric` (RMSE by default), the runs' point scores, the two best bootstrapped, `forest_figure`; the notebook's md5 `9a66a347ed3fc928f14a43bde04da1b9`. **Updated 2026-11-07:** at the user's review, the best runs' table renamed `best_runs_val_scores_df` with `model`, `family`, `use_exposure`; the forest plots to be fixed later; PR 5 closes at 5.3, then PR 5b (the EDA: `build_trials_table`, the joint Model 2 method), PR 6 (the split), PR 7 (5.4, SHAP per cohort, the test evaluation). **Updated 2026-11-07:** PR 5 merged as #23 (`c3bdee8` on `main`, identical to `8a90bf9`); PR 5b planned (§4b) on `feat/model2-tuning-methods`. **Updated 2026-11-07:** 5b.1 done (§4b, the rename); the notebook's md5 `e0f2ba6c32ead22ebee081a4f0a27e3f`. **Updated 2026-11-07:** 5b.1 committed (`82ccac2`); 5b.2 done (§4b, the joint method in the EDA); the notebook's md5 `6731d509d2b5298e4d38a99480f700dc`. **Updated 2026-11-07:** 5b.2 committed (`0b25ede`); 5b.3 done (§4b, the reading at 100 trials): D40, section 6 switches Model 2 to the joint method in PR 7 (7.0). **Updated 2026-11-07:** 5b.3 committed (`537ff7c`); PR 5b closed (the suite 1393 passed, mypy 28 files, Check A, B). **Updated 2026-11-07:** PR 5b merged as #24 (`39dd1aa` on `main`); PR 6's branch `refactor/two-notebooks` created from it. **Next: PR 6 (§5), from 6.1.**

The companion docs: [EXPERIMENT_NOTEBOOK_HANDOFF.md](EXPERIMENT_NOTEBOOK_HANDOFF.md) (the state,
how to work, the checks, the traps, iCloud), [NOTEBOOK_METRICS_LOOP_EDA_PLAN.md](NOTEBOOK_METRICS_LOOP_EDA_PLAN.md)
(PRs 1–3, M1–M14), [NOTEBOOK_TUNING_PLAN.md](NOTEBOOK_TUNING_PLAN.md) §0 (D1–D30),
[EXPERIMENT_NOTEBOOK_PLAN.md](EXPERIMENT_NOTEBOOK_PLAN.md) (E11, the final evaluation).

## Contents

- §0 Context and decisions (D31–D39)
- §1 How to work: the rules, the checks, the reference run
- §2 PR 3's close: 3.3
- §3 PR 4: the tuning budget and the study cache (4.1–4.3)
- §4 PR 5: the bootstrap replicates and the comparison metric (5.1–5.3; 5.4 moved to PR 7)
- §4b PR 5b: Model 2's two tuning methods in the EDA (5b.1–5b.3)
- §5 PR 6: two notebooks (6.1–6.3)
- §6 PR 7: Model 2 by the joint method (7.0, D40), SHAP per cohort and the test evaluation (7.1–7.3)
- §7 Facts verified on 2026-11-04

## 0. Context and decisions

The experiment notebook (`notebooks/02_experiments.py`, marimo) has six sections: the data, the
settings, the split, the feature variants, the objective EDA (section 5, PR 3) and the tuning with
the inspection (section 6). Each tuning run keeps its best model with its bootstrap intervals (3.4,
M14). The user listed seven changes on 2026-11-04. They go in four PRs from `main` after PR 3's
merge, in this order (the user's choice): the `src` changes first (PRs 4, 5), the restructure
(PR 6) once the content is final, the new content (PR 7) in the split notebook.

**Decisions (2026-11-04, the user's; do not re-ask):**

- **D31. The budget: `n_trials = 100`, `n_startup_trials = 10`.** Reverses D20's "never
  decrease `n_trials`" (the user). The random start stays an explicit, logged setting (the test
  pins it; the tuner's field has no default) at Optuna's default of 10, 10% of the budget: D20
  measured the random start's effect within the search noise and its cost not (Model 2's run
  1.5–2× longer at 20, the random trials drawing large models). The question "can I remove it"
  is answered: only by giving `CrossValidationTuner.n_startup_trials` a default; one explicit line
  is simpler and logged.
- **D32. Model 2 stays by coordinate descent** (D29; **superseded by D40**, 2026-11-07). At 100 trials a joint study of 23–33
  dimensions gets 3–4 trials per dimension; the two steps get 100 each, and the method is exact
  under the deviance. The joint tune method stays in the package (a valid path).
- **D33. The study cache: Optuna's journal storage, one file per study, loaded when complete.**
  `JournalStorage(JournalFileBackend(path))` writes a JSON-lines file and loads a real `Study`
  (trials, params, user attributes, durations, states), so the EDA's `trials_table(study, …)`
  and the run loops keep reading `study.trials` / `best_params` unchanged; no `ExperimentalWarning`
  under `-W error` (probed, Optuna 4.9.0). A parquet of the trials would lose the study object
  and force those readers to be rewritten. The file is named from the run's settings so `ls`
  answers whether a study exists; the directory is keyed by the data and split, so a changed
  population never serves a stale study. The cache lives under the project, git-ignored.
- **D34. The Poisson deviance stays the tuner's objective; RMSE is the comparison metric.**
  The user leaned to RMSE for both and asked for the decision on the EDA's evidence (3.2, 3.2b):
  on validation the RMSE-tuned models were last or near last on every metric including RMSE
  (Model 1 4.6070 against 4.5393 deviance-tuned; Model 2 4.5613 against 4.5450; one holdout,
  noise about 0.04); on cross-validation a tie within the 0.02 noise (cv RMSE 4.3663 against
  4.3722; 4.2800 against 4.3019); within its study the RMSE best is also the deviance's best
  (percentile 0.0); coordinate descent is exact under the deviance only (M10). Chosen: the
  deviance tunes, RMSE compares (the best-run rule, the forest plots, the paired difference, the
  test scores); all six metrics stay recorded. **Amended 2026-11-07 (5.3's plan):** the
  selection metric is a setting (`selection_metric`, RMSE or the deviance, default RMSE), each
  model's best run chosen by its validation point score.
- **D35. `scoring.evaluate` becomes `compute_metrics`.** The package's precedent is
  `PermutationImportance.compute_importances`; "calc" is an abbreviation. A rename only.
- **D36. The bootstrap evaluator returns its replicates; the statistics are computed outside.**
  `BootstrapEvaluator(n_resamples, random_state)`; `evaluate(y_true, y_pred, groups, *,
  metrics, cohorts)` returns the `n_resamples × (cohort, metric)` frame; `alpha` leaves the
  class. The notebook computes the mean, the SD and the percentile interval, draws forest plots,
  and the paired difference of two models per replicate (valid because one evaluator with one
  seed draws the same rows for every model: M14). `n_resamples` (1000 or 10,000) is decided by a
  measurement in 5.3, not by rule (2026-11-07: measured, 1000 kept; and only each model's best
  run is bootstrapped, the other runs scored by point score: 5.3's plan). The resampling unit (whole neighborhoods today; rows within
  neighborhoods, the user's pandas sketch `y_true.groupby(groups).sample(frac=1.0, replace=True)`)
  is a design question in 5.2, answered with measured interval widths; the NumPy loop stays
  (pandas' `groupby.sample` is no simpler and loses `np.ndarray` targets).
- **D37. Two notebooks:** `02_objective_eda.py` (sections 1–5) and `03_hyperparameter_tuning.py`
  (sections 1–4, 6, then PR 7's sections). How they share sections 1–4 is 6.1's design question.
- **D38. SHAP per cohort, values and plots** ("show values and plots for each cohort
  separately", to see whether the features' order differs per cohort): the explainer's values
  are already per row, column and cohort; the table and the figures stop averaging over cohorts.
- **D40. Section 6 tunes Model 2 by the joint method (PR 7)** (2026-11-07, the user's, after 5b.3's
  reading at 100 trials). The rule the user set: if the two methods' results are the same on the
  deviance and on RMSE, switch to joint. The evidence (the EDA's setting, `lgbm_poisson+logistic`,
  exposure on; joint − coordinate descent; the validation intervals from a paired bootstrap,
  1000 replicates, whole neighborhoods):
  - tuned on the deviance (the tuner's objective): cv deviance 2.0735 against 2.0791 (−0.006),
    validation deviance +0.012 (95% −0.031 to 0.051), cv RMSE −0.009, validation RMSE +0.022
    (−0.035 to 0.073): the same within the noise;
  - tuned on RMSE: cv RMSE +0.039 (coordinate descent better beyond the 0.02 noise), validation
    RMSE −0.113 (−0.242 to 0.007): opposite directions, no consistent winner;
  - tuned on the log loss: uninformative (the total gets no signal, so each method's total is
    arbitrary; validation deviance +0.065, 0.010 to 0.127).

  The cost the user accepted: section 6's Model 2 becomes 16 joint studies (8 totals × 2
  classifier families) instead of 10, half with the LightGBM classifier; the first click is
  roughly 20 min longer (an estimate from 2.3's 400-trial step times), then cached. Coordinate
  descent stays in the package and in the EDA as the comparison. Supersedes D32.
- **D39. The test evaluation follows E11:** both models' best runs (by the comparison metric on
  validation) refit on train + validation, scored on the test part once, behind a button, in the
  tuning notebook; the test scores are reported, never used to choose.

## 1. How to work

The standing rules are the handoff's §4 (plan mode per sub-task; the facts re-verified with
read-only probes; a design question gets evidence and a recommendation; the steps and the commit
message in the chat; a file-by-file summary, the checks, the notebook's md5 and the commit message
at each stop; the user commits; no `Co-Authored-By`, no "Generated with Claude Code"; never
`git checkout`/`stash`/`reset`/`mv`; format new cells with `ruff format` one by one, never the
whole file; the iCloud checks before every long run). The user's style: the smallest change that
works; loops over settings; no function unless reused; names that say what a thing holds; no
private helpers; settings in the constructor, data in the call.

**The checks of every sub-task** (the handoff's §4 has the details and the script locations):
- **Check A:** the notebook test (`tests/characterization/test_experiments_notebook.py`), ruff
  (14 findings today; none new), `ruff format --diff` (the user's two hunks only), `marimo check`.
- **Check B:** `app.run()`'s defs of a script run; nothing fitted.
- **Check C:** the clicked run on a 3-trial copy (`make_copy.py`: `n_trials = 3`, Model 1's
  `share_hyperparams` False; configs and src symlinked; `PYTHONWARNINGS=error`), the 24 `cv_best`
  baselines (the handoff's §2) and every assertion since 3.1c, compared frame by frame with the
  reference pickle of the previous commit's run (`compare.py`, exact, `seconds` dropped). Before
  the first edit of a sub-task that changes `src` or a run cell, run Check C on HEAD and keep the
  pickle as the next reference. The reference for the first change after 3.4 was
  `check_c_3_4b.pkl`; since 3.3 it is `check_c_3_3_fix.pkl` (the run of `aa1e267`, the F2 fix:
  every frame and record equal to `check_c_3_4b.pkl`'s, Model 2's EDA figures one label each).
- **`src` changes:** the unit tests of the module, the non-slow suite (`pytest -m "not slow"`,
  about 2 min) and mypy (`--cache-dir <scratch>/mypy_cache`); two mutations caught per sub-task.
- **The scripts:** copied into the session's scratchpad from
  `/private/tmp/claude-501/-Users-galkampel-Desktop-Projects-age-group-prediction/99f2beed-010f-4e31-a210-84ca1344c999/scratchpad/checks`
  (the 3.3 session's copy, 2026-11-05, with F2's assertion in `check_c.py`; the earlier copies
  are named in the handoff's §4); if gone, rewritten from the handoff's §4.
- **Settings pinned by the notebook test:** `n_trials`, `n_startup_trials`, `n_resamples`,
  `alpha` and the run-settings lists; a sub-task that changes one updates the test in the same
  commit.
- **A marimo session the user has open** holds the code it loaded: tell the user before editing
  the notebook; they rerun the changed cells or restart the kernel; they must not save over an
  edit from an open editor.

## 2. PR 3's close: 3.3 (on `feat/objective-eda`)

No code change (D34). One commit for the docs, one more if the review's findings are fixed.

### [x] 3.3a The checks and the review
- **Done (2026-11-05).** The review covered `main..HEAD` at `a006995` (19 commits, 13 files,
  +2598/−797, this plan's facts included) and found no correctness bug. Five findings, each
  reproduced by the subagent and again in the session:
  - F1: HYPERPARAMETER_TUNING.md:327/330 named the removed `best_df` and said a run keeps every
    trial;
  - F2: Model 2's EDA figures printed the "probability step" label in all six panels;
  - F3: no repo test of `trials_table` or section 6's records and tables;
  - F4: the transformer-name lookup written four times, the hyperparameters read two ways;
  - F5: 3.1k's and 3.1l's commit lines named commits that do not exist.

  The user chose F2 for a fix, F1 and F5 for 3.3b, and F3 and F4 for PR 6 (the handoff's §5).
  F2 was fixed in `aa1e267`: one `add_annotation` in the top panel. Its checks:
  - Check A: the notebook test 13 passed, ruff 14, the two format hunks, `marimo check` no issue;
  - Check B;
  - Check C (400 s): every frame and record equal to `a006995`'s run (`check_c_3_3a.pkl`, itself
    equal to `check_c_3_4b.pkl`), the figures' JSON equal but Model 2's labels (6 → 1);
  - a new assertion (one label, top panel; none in Model 1's) caught both mutations.

  The notebook's md5 is `95261326e5368bd451607d083a5ab366`.
- **Checks already run (2026-11-04):** the non-slow suite 1387 passed, 1 skipped, 1 xfailed, 19
  deselected; mypy 28 files, no issue; notebook md5 `676212a808f6c1f87f7926a58b3bffe6`.
- **The review:** an independent subagent over `git diff main..HEAD` (18 commits, 11 files,
  +2196/−794: the notebook's sections 5–6, `cross_validation.py`'s recorded metrics, the docs),
  given the diff, the metrics plan's §2 M10–M14 and §5, and the handoff's §3, not the session's
  conclusions; correctness first, then reuse and simplification; every finding reproduced with a
  read-only probe before it is reported; no fix without the user's approval. Fixes, if any, in
  their own commit (`fix(notebook): the PR review's findings`), each with Check A–C.
- **Done when:** the findings are listed in the chat with their reproductions; the user has
  chosen what to fix.

### [x] 3.3b The docs and the PR
- **Done (2026-11-05).** The changes below, plus:
  - F1 and F5;
  - the stale sections, with wording by role: where the notebook runs the tuner, "sections 5
    (the objective EDA) and 6 (the tuning)"; where the package was rebuilt from it, "the tuning
    section (then 5, now 6)";
  - two more places found by grep: `docs/MODULE_REFERENCE.md:116` and the package docstring
    `hyperparameter_tuning/__init__.py:20`.

  The PR's text is given in the chat.
- **Change:**
  - this plan copied to `docs/NOTEBOOK_CACHE_EVALUATION_PLAN.md`, listed in `docs/README.md`'s
    plans;
  - `docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md`: 3.3 ticked with its Done note (the deviance kept,
    D34's evidence, the checks' numbers, the review's findings); the status line;
  - `docs/EXPERIMENT_NOTEBOOK_HANDOFF.md`: rewritten for the session that starts PR 4 (§2 the
    state after the merge, §2b PR 3's commits and review, §4 the scripts' location above, §6 PR
    4's 4.1 and the prompt; MLflow stays deferred, E11 is PR 7);
  - `README.md:39`, `docs/README.md:24`, `docs/SPLITTING.md:267`: "section 5" → "section 6" for
    the tuning; `docs/HYPERPARAMETER_TUNING.md:8`: "rebuilt from section 5" → "from the tuning
    section (then 5, now 6)".
- **In the chat:** the PR's title and description, the squash-merge title and description, and
  after the merge `git push -d origin feat/objective-eda`, `git branch -d feat/objective-eda`
  (the user runs them).
- **Done when:** the docs name no stale section; the handoff's prompt starts 4.1.
- **Commit:** `docs(notebook): PR 3 closed: the deviance kept as the objective; the plan for the
  study cache, the bootstrap replicates, the split and the test evaluation`
- **PR title:** `feat(notebook): the objective EDA; each tuned run keeps its best model and its
  bootstrap intervals`

## 3. PR 4: the tuning budget and the study cache (`feat/study-cache`, from `main`)

### [x] 4.1 The budget (notebook, test)
- **Done (2026-11-06).** As below, plus `HYPERPARAMETER_TUNING.md:49` and `:516`, which
  named the notebook's 400 trials and random start of 20. D20's claim re-probed first (Optuna
  4.9.0, seeds 42 and 49: 3 trials draw the same params at a start of 10 or 20); the new pins
  failed on the old notebook. Checks:
  - Check A: the notebook test 13 passed, ruff 14, the two format hunks, `marimo check` no issue;
  - Check B;
  - Check C (410 s): the 24 baselines and every assertion; every frame and record equal to
    `check_c_3_3_fix.pkl`; both figures' JSON files byte-identical.

  The notebook's md5 is `78b08bd7936dba06d65f18bd52da5cbb`; Check C's pickle `check_c_4_1.pkl`
  (in this session's scratchpad) is 4.2's reference.
- **Change:** `notebooks/02_experiments.py` section 2: `n_trials = 100`, `n_startup_trials = 10`
  (the comment: 10% of the budget, Optuna's default); section 6's text ("`n_trials = 400`, the
  first `n_startup_trials = 20`" → the new values); `tests/characterization/test_experiments_notebook.py:124-125`
  pinned to `"100"`, `"10"`; the scratchpad's `make_copy.py` replaces `    n_trials = 100\n`
  (assert the count). D31 and D32 in NOTEBOOK_TUNING_PLAN.md §0 (one line each, pointing here).
- **Checks:** Check A (the test with the new pins); Check B; Check C against `check_c_3_3_fix.pkl`:
  every frame equal (a 3-trial study draws the same points at a random start of 10 or 20: D20).
- **Done when:** the pinned settings pass and Check C's frames equal 3.4's.
- **Commit:** `feat(notebook): 100 trials, 10 at random`

### [x] 4.2 The tuner takes a study storage (`src`, tests, HYPERPARAMETER_TUNING.md)
- **Done (2026-11-06).** As below, with the user's two answers.
  - **Docs:** HYPERPARAMETER_TUNING.md §6 (the signature, "The stored studies") and §10 (a
    stored study found by its name only).
  - **Tests:** the module 21 passed (+4) under `-W error`; the non-slow suite 1391 passed, 1
    skipped, 1 xfailed, 19 deselected (6 warnings, none from the module); mypy 28 files, no
    issue.
  - **Revised at the user's notes** (the explicit branch for a loaded study): every check below
    rerun on the revision, with the numbers given.
  - **Mutations,** each caught by its test (a sentinel first): the partial check `<=`; 0.0
    returned in place of the stored seconds; the delete removed. The partial test first stored 2 of 3 trials and missed
    the third mutation: at a random start of 2, the third trial is TPE's first draw whichever
    way the study goes on. It stores 1 now, so the resumed second trial repeats the first's
    draw.
  - **The load probe** (the real tuner, `lgbm_poisson`, shared hyperparameters, 13
    parameters, 3 recorded metrics, `PYTHONWARNINGS=error`): two 100-trial studies, 438 KiB
    each, loaded in 0.024 s and 0.025 s with their stored seconds (8.2, 6.0) and the same
    trials.
  - **Check A, B; Check C** (460 s, the notebook unchanged, its tuner in memory): every frame
    and record equal to `check_c_4_1.pkl`, both figures' JSON files byte-identical.

  The notebook's md5 is unchanged, `78b08bd7936dba06d65f18bd52da5cbb`.
- **Change:** `src/age_group_prediction/hyperparameter_tuning/cross_validation.py`:
  `CrossValidationTuner` gains `storage_dir: Path | None = None`; each tune method gains a
  keyword `study_name: str | None = None` passed to `_optimize`. In `_optimize`: without
  `storage_dir`, today's in-memory study; with it, `JournalStorage(JournalFileBackend(storage_dir /
  f"{study_name}.log"))` (the directory created), `optuna.create_study(..., storage=,
  study_name=, load_if_exists=True)`; a new study (no trials) runs `n_trials` and stores its
  run time (`set_user_attr("seconds", …)`), a loaded complete study skips `optimize`; the
  stored seconds are returned either way. A partial
  file (fewer than `n_trials` trials) is deleted (`optuna.delete_study`) and the study run again
  from its seed. `study_name` required when `storage_dir` is given (raise `ValueError`
  otherwise).
- **The user's answers (2026-11-06), on probes:** a partial file **starts fresh**, not resumed (the
  plan's first recommendation): a resumed study draws differently from the seeded run (the
  sampler's random state is not stored), and an interrupted trial stays as FAIL, while delete
  and rerun reproduces the seeded run exactly. **The seconds are how long the study took to
  run**, stored with it, not the load time (the plan's 0.0). **At the user's notes on the
  first version:** a loaded study skips `optimize` by an explicit branch (`if not
  study.trials:`), in place of `optimize(n_trials=0)`, which Optuna's loop exits at once but
  only a reader of that loop would know; no save switch: the cache is always used ("to keep
  track and save the runs"), the package keeping `storage_dir=None` as its in-memory default.
- **Tests** (`tests/unit/test_hyperparameter_tuning_cross_validation.py`): a second call with the
  same dir and name returns the stored trials and the first call's seconds without scoring a fold
  (a counting metric), the dir created; a partial file is run again and equals the seeded
  study; `storage_dir` without a name raises; each of the four tune methods stores its study
  under its name; no dir behaves as before (the existing tests).
- **Checks:** the module's tests, the suite, mypy; three mutations (the partial check `<=`; 0.0
  returned in place of the stored seconds; the delete removed, a resume); Check C against 4.1's
  run.
- **Done when:** the tests pass; a probe shows two notebook-sized studies loading in under a
  second.
- **Commit:** `feat(tuning): a study is stored in a journal file and loaded when complete`

### [x] 4.3 The notebook's runs through the cache (notebook, test, docs)
- **Done (2026-11-06).** As below, with two changes after the first Check C:
  - **The user's renames** (made in the editor, kept at their answer): `name_study` →
    `build_study_name`; the steps `model1` → `independent_cohorts`,
    `model2_coordinate_descent_{total,probability}` →
    `total_times_probability_coordinate_descent_{total,probability}`. Two cells' returns were
    trimmed by marimo to a bare `return` (`eda_figures`, `eda_metric_figures` no longer
    exported) and kept: ruff 13 → 15 (PLR1711); Check C's copy restores them.
  - **A fix in `src`:** the first Check C's loading run grew every study file by one line.
    `create_study(load_if_exists=True)` on a stored study appends a create line before Optuna
    4.9's journal storage finds the duplicate (`JournalStorage.create_new_study`). A complete
    stored study is now opened with `load_study`, which only reads, and returned;
    `create_study` only creates. The load test checks the file's bytes; two mutations caught
    (a complete study opened by `create_study(load_if_exists=True)`: the bytes; deleted and
    rerun: the call count).

  Checks:
  - the tuning module 21 passed under `-W error`; the non-slow suite 1392 passed, 1 skipped,
    1 xfailed, 19 deselected; mypy 28 files, no issue;
  - Check A: the notebook test 14 passed, ruff 15 (13 PLR1711, a B018, a RUF059), the user's
    two format hunks only, `marimo check` no issue; Check B, no `studies/` written by a script
    run;
  - Check C twice on one copy: the first (435 s) equal to 4.2's `check_c_4_2b.pkl` (`seconds`
    dropped), figures' JSON identical, 26 files with the names computed independently; the
    second (398 s) equal to the first with `seconds`, every file's size and md5 unchanged.

  **The 100-trial clicks on the real notebook** (`PYTHONWARNINGS=error`):
  - the first tuned and stored 25 studies (8.5 MiB) in 1237 s. The EDA's and section 6's
    first study of each model share a name (Model 1's flags as the EDA's, Model 2's first total),
    so 27 tune calls store 25 studies (the 3-trial copy, its flags off, stores 26). No failed
    trial. The studies' seconds: Model 1 300 s, Model 2's totals 180 s, its probabilities 119 s;
    the EDA's Model 1 40–46 s, Model 2 43–58 s per objective;
  - the second loaded them in 409 s (the refits, the bootstraps, the inspection): every frame
    equal to the first click's, `seconds` included; the 25 files unchanged;
  - best on validation (composite deviance): Model 1 `lgbm_poisson`, exposure on, val 2.1296
    (cv 2.1402), then `lgbm_gaussian` on 2.1361, `lgbm_poisson` off 2.1684; Model 2
    `lgbm_gaussian+logistic`, exposure on, val 2.1031 (cv 2.0776), then
    `lgbm_poisson+logistic` off 2.1217, `lgbm_gaussian+lgbm` on 2.1293;
  - the EDA's `cv_best` (log loss / deviance / RMSE): Model 1 1.0849 / 2.1402 / 4.3828; Model 2
    1.0828 / 2.0791 / 4.2862.

  The notebook's md5 is `91d1c96cdd227a137163cd072235c64a`.
- **Change:** the tuner cell (section 5) builds `studies_dir = project_root / "studies" / <the
  settings that shape every study>`, joined `key=value` by `__`: the data's config hash (8
  characters), `split_method`, `test_size`, `val_size`, `n_splits`, `seed`, `n_trials`,
  `n_startup_trials`. It passes `storage_dir=studies_dir`, defines `build_study_name(step,
  run_settings, objective_name, study_seed)` (the step, the run settings `key=value`, the
  objective, the study's seed, joined by `__`), and shows the directory. Section 1's two cells
  export `project_root` and `config_hash`; `.gitignore` gains `studies/`. Every tune call
  passes `study_name=build_study_name(...)`:
  - the steps are `independent_cohorts`, `total_times_probability_coordinate_descent_total`
    and `total_times_probability_coordinate_descent_probability` (the user's names, renamed
    from `model1` / `model2_coordinate_descent_*` after the first Check C);
  - the EDA's runs use section 6's run-settings keys, so the two studies both sections run (each
    model's first under the Poisson deviance, M10) get one name and are tuned once.

  Section 5's text explains the storage; section 6 has one sentence. The notebook test pins the
  directory's and one study's name.
- **The user's answers (2026-11-06)**, on the evidence that the planned names left out the test
  and validation sizes, `n_splits`, `n_startup_trials` and each study's seed (a changed setting
  would load a stale study):
  - **names from the settings**, as above;
  - **Model 2's two tunings apart in the names:** coordinate descent's steps against a joint
    study's `total_times_probability_joint`;
  - **no study-name column** in the records or the runs tables: a row's settings and seed name
    its file;
  - **no switch:** the cache is always used (4.2).
- **Checks:**
  - Check A and Check B, with no `studies/` written by a script run;
  - Check C twice on one copy (its own `studies/`):
    - the first run equals 4.2's frames and writes one directory of 26 files whose names are
      computed independently;
    - the second loads them: every frame equal, `seconds` included, no file changed;
  - the 100-trial click on the real notebook (all buttons, `PYTHONWARNINGS=error`), then a
    second one: the seconds and the best runs, recorded in this plan's Done note and the
    handoff's §2.
- **Done when:** a second click skips the tuning. It still refits, bootstraps and inspects the
  kept models (the plan's "seconds, not minutes" was wrong: the bootstrap alone is about 6 min,
  48 calls of about 7 s).
- **Commit:** `feat(notebook): every study cached in a journal file named by its run settings`

## 4. PR 5: the bootstrap replicates and the comparison metric (`feat/bootstrap-replicates`)

### [x] 5.1 `scoring.evaluate` → `compute_metrics` (`src`, tests, docs, notebook)
- **Done (2026-11-06).** As planned; every line number held on `466534e`. One addition: the
  two test names `test_evaluate_*` → `test_compute_metrics_*` (the word-bounded grep missed
  them; one signature wrapped to `ruff format`'s hunk). Docstrings rewrapped where the name
  lengthened a line. Checks:
  - the grep finds only the bootstrap's own `evaluate`, the old stack and history;
    `hasattr(scoring, "evaluate")` False;
  - `test_scoring.py` and `test_inspection_bootstrap.py` 110 passed under `-W error`; the
    non-slow suite 1392 passed, 1 skipped, 1 xfailed, 19 deselected; mypy 28 files, no issue;
  - HYPERPARAMETER_TUNING.md's nine output blocks equal;
  - Check A: the notebook test 14 passed, ruff 15, the user's two format hunks, `marimo check`
    no issue; Check B (`compute_metrics` defined, `evaluate` not);
  - Check C (399 s, a fresh copy): every assertion; every frame and record equal to
    `check_c_4_3.pkl` (`seconds` dropped), both figures' JSON identical. Its pickle
    `check_c_5_1.pkl` (this session's scratchpad) is 5.2's reference.

  The notebook's md5 is `6edc34aa4cc93d05fcf34b106515dd9c`.
- **Planned and approved (2026-11-06; the line numbers read on `383ee53` by grep, re-verify
  them).** A rename only: no behaviour, signature or output changes; no alias.
  - `scoring.py`: `def evaluate` (:205) → `compute_metrics`, `__all__` (:41, kept sorted), the
    module docstring's ``evaluate`` (:13);
  - `inspection/bootstrap.py`: the call `scoring.evaluate(...)` (:102) and the docstrings (:37,
    :59, :62); the bootstrap's own `evaluate` method stays (5.2 reshapes it);
  - `tests/unit/test_scoring.py`: the import (:14), the section comment (:367), three calls
    (:384, :406, :408);
  - the notebook: the import cell's import (:708) and return (:731); the four cells' parameter
    lists (:950, :1063, :1793, :2165) re-sorted (`compute_metrics` before `cv_groups`); the four
    calls (:1036, :1200, :1859, :2203); each edited line to `ruff format`'s hunk;
  - the current guides: HYPERPARAMETER_TUNING.md (:35, :64, :271, the block at :300, :598,
    :605), MODEL_INSPECTION.md (:303, :311), MODULE_REFERENCE.md (:69 twice and its export
    list, :75, :120), INDEPENDENT_COHORT_MODELS.md (:325), TOTAL_TIMES_PROBABILITY_MODEL.md
    (:355), the handoff's current-state lines (§1);
  - history stays as written: D30 (NOTEBOOK_TUNING_PLAN.md) gets one note ("renamed
    `compute_metrics` in 5.1, D35"); closed plans and commit tables are not rewritten;
  - the scratchpad's Check C reads `d["compute_metrics"]` (`check_c.py:93`, `:315`); Check B
    asserts `compute_metrics` in the defs and `evaluate` not.

  Its checks: the grep below finds only the history left on purpose, and
  `hasattr(age_group_prediction.scoring, "evaluate")` is False; `test_scoring.py` and
  `test_inspection_bootstrap.py` under `-W error`; the suite; mypy; HYPERPARAMETER_TUNING.md's
  blocks rerun; Check A (the notebook test 14, ruff 15, the two hunks, `marimo check`), B, C
  equal to `check_c_4_3.pkl` exactly. No mutations: a rename changes no behaviour a test could
  tell apart; the grep and the `hasattr` probe stand in.
- **Change:** the rename in `scoring.py`, `inspection/bootstrap.py`, the notebook's four calls,
  `tests/unit/test_scoring.py`, `test_inspection_bootstrap.py`, `test_hyperparameter_tuning_cross_validation.py`
  (if it names it), `docs/HYPERPARAMETER_TUNING.md`, `MODEL_INSPECTION.md`, the handoff, D30's
  line (a note, not a rewrite). `grep -rn "evaluate(" src tests docs notebooks` (excluding
  `__pycache__`) finds no `scoring.evaluate` after.
- **Checks:** the suite, mypy, Check A–C (the frames equal the reference exactly).
- **Commit:** `refactor(scoring): evaluate renamed compute_metrics`

### [x] 5.2 The evaluator returns the replicates (`src`, tests, MODEL_INSPECTION.md)
- **Done (2026-11-06).** `BootstrapEvaluator(n_resamples, random_state=None)`; `evaluate`
  returns `pd.DataFrame(scores, columns=labels)`, one row per replicate, a `(cohort, metric)`
  column MultiIndex; the drawing loop unchanged.
  - **The resampling unit (the user's answer): the two designs stay** (`groups` → whole
    groups, `None` → rows; no `unit` setting). The probe (the 100-trial best models, validation,
    1000 replicates, seed 42; 306 rows in 129 neighborhoods, median 2 rows, 37 with one): the
    composite's 95% width, rows / whole neighborhoods / rows within neighborhoods: Model 1
    deviance 0.507 / 0.542 / 0.380, RMSE 1.006 / 1.095 / 0.582; Model 2 deviance 0.488 / 0.493
    / 0.360, RMSE 0.885 / 1.024 / 0.582; per cohort, within-neighborhood 0.36–0.82× the whole
    neighborhoods' (its 1–2-row neighborhoods barely vary), rows 0.85–1.0×; about 2 s a call.
  - **A correction to the plan below:** the records' `scores` do not hold the raw replicates
    yet (the tables and the best-run rule read `score, mean, ci95_*`): each loop computes
    `mean` and the `alpha / 2`, `1 - alpha / 2` quantiles from the replicates, so every table
    is unchanged; 5.3 moves them to the tables cells.
  - The bounds equal the old `np.quantile`'s exactly; the mean differs in its last bits
    (pandas sums a column in another order: 344 of 384 and 699 of 768 table values, at most
    1.4e-14 relative), so Check C compares `mean` at rtol 1e-12.
  - **Tests** (`test_inspection_bootstrap.py`, 24): the shape and columns, the replicates the
    metric's scores in order, the old implementation's mean and bounds pinned (seed 0, rows
    and groups), `alpha=` rejected; the module 24 passed under `-W error`; the non-slow suite
    1393 passed, 1 skipped, 1 xfailed, 19 deselected; mypy 28 files, no issue. Two mutations
    caught after a sentinel: the columns labeled in reverse (3 tests), one replicate fewer
    (5 tests).
  - **Docs:** MODEL_INSPECTION.md (the summary, the API row, the bootstrap block, §3; its
    blocks rerun, output unchanged), MODULE_REFERENCE.md, HYPERPARAMETER_TUNING.md §8, the
    handoff.
  - **Check A:** the notebook test 14, ruff 15, the two format hunks, `marimo check` no
    issue; **Check B** (no `alpha` on the evaluator); **Check C** (444 s): every assertion
    (the last runs' statistics recomputed from `evaluate`'s replicates exactly), every frame
    and record equal to `check_c_5_1.pkl` (`mean` at 1e-12), both figures' JSON identical;
    its pickle `check_c_5_2.pkl`.
  - **Amended at the user's review** (before the commit): the loops' result named
    `_replicate_scores` (the user's rename `_replicates_scores`, then the package's own name
    at their answer); `median` and `sd` (pandas' `std()`, `ddof=1`: the bootstrap standard
    error) added, the columns `score, mean, median, sd, ci95_lower, ci95_upper` (their
    order); section 6's text. The scores tables 384 × 13 and 768 × 11, the rows unchanged.
    Check A (the notebook test 14, ruff 15, the two hunks, `marimo check`), B; Check C (414 s):
    every assertion, `ci95_lower ≤ median ≤ ci95_upper`, `sd > 0`, the last runs' `median`
    and `sd` recomputed from the replicates (pandas exactly; NumPy's `median` exactly, its
    `std(ddof=1)` at 1e-12); every other column and record equal to `check_c_5_1.pkl`, both
    figures' JSON identical. Its pickle `check_c_5_2b.pkl` is 5.3's reference.

  The notebook's md5 is `d4a361a5535aaefd89b9dccd4f99e282`.
- **Change:** `BootstrapEvaluator(n_resamples, random_state=None)`; `evaluate` returns
  `pd.DataFrame(scores, columns=labels)` (one row per replicate, a `(cohort, metric)` column
  MultiIndex) after the non-finite check; the `mean/lower/upper` block and `alpha` go. The
  notebook's two loops keep the replicates in each record's `scores` (5.3 reshapes the tables).
  The docstrings and the guide's block rerun.
- **Design question, with evidence from a probe** (one fitted model, validation rows): the
  resampling unit as a constructor setting, `unit: Literal["rows", "groups"]`? Today `groups=None`
  gives rows and `groups` given gives whole groups; the user's sketch resamples rows within each
  group (stratified). Measure the three intervals' widths on the composite RMSE and deviance.
  Recommended: keep the two existing designs (the API already chooses by `groups`); add the
  stratified unit only if its width differs materially and the user wants it, as a third value
  of a `unit` setting.
- **Tests:** the frame's shape `(n_resamples, n_cohorts × n_metrics)`, its quantiles equal the
  old bounds (a fixture from the old implementation's numbers), the same seed the same rows, the
  non-finite error unchanged.
- **Checks:** the suite, mypy, Check C: each record's `scores` quantiles at 0.025/0.975 equal
  the reference's `ci95_lower/ci95_upper` exactly, the means equal.
- **Commit:** `refactor(inspection): the bootstrap evaluator returns its replicates`

### [x] 5.3 Each model's best run by a selection metric, bootstrapped; a forest plot of the two
- **Done (2026-11-07), as planned below;** every line number held on `1af184a`. The notebook:
  - `selection_metric = "rmse"` in section 2 (logged, pinned by the test);
    `SELECTION_METRIC` in the metrics cell;
  - the two loops keep point scores (`_dataset_scores.to_frame("score")`; the tables 384 × 8,
    768 × 6, their captions "(the point score; …)");
  - a cell after Model 2's tables: the best run per model by the validation composite on
    `SELECTION_METRIC`, the two bootstrapped on validation (`best_replicate_scores`,
    `best_runs_val_scores_df` 48 × 11: `model, family, use_exposure, cohort, metric, score,
    mean, median, sd, ci95_lower, ci95_upper`, the models `independent_cohorts` and
    `total_times_probability`, `use_exposure` a bool; the same names as the keys of
    `best_replicate_scores`); the inspection's selection cell removed;
  - `forest_figure(scores, metric_name, lower_column, upper_column, rule_at=None)` (one
    scatter per cohort panel, the asymmetric `error_x`, the y axis reversed, the hover with
    the bounds in `customdata`) and the forest-plot cell (`best_forest_figure`);
  - section 6's text: the selection rule and why, the bootstrap of the two best.
  - **Amended at the user's review** (2026-11-07, before the commit): the table's `model1`/`model2`
    labels became the model's name, family and exposure in three columns, and `best_scores_df`
    was renamed `best_runs_val_scores_df` (what it holds: the best runs' validation scores);
    `forest_figure` builds its row labels from the three columns. The forest plots, seen on the
    100-trial runs (a copy of the notebook on the user's `studies/`, a viewer app), "look really
    bad": to be fixed later (the handoff's §5). On those runs the best by RMSE are the independent
    cohorts model's `lgbm_poisson`, exposure on (composite RMSE 4.5586, 95% 4.04–5.13), and the
    total times probability model's `lgbm_poisson+logistic`, exposure off (4.5388, 4.05–5.08);
    the statistics equal 5.2's records of the two runs (`probe_5_2.pkl`) at 1e-12.
  - **A correction to the plan's Check B:** a script run clicks nothing, so `best_scores_df`
    and `best_replicate_scores` are not in its defs; Check B asserts them absent, Check C
    checks them.
  - **Check C's 3-trial runs** pick the same best runs by RMSE as by the deviance (Model 1
    `lgbm_gaussian`, exposure on; Model 2 `lgbm_gaussian+logistic`, exposure on), so
    `importance_df` and the two best runs' statistics equal 5.2's run.

  Checks: Check A (the notebook test 14 passed, ruff 15, the user's two format hunks, `marimo
  check` no issue); Check B (`SELECTION_METRIC` the `rmse` object, `forest_figure` defined, the
  best runs' names absent); Check C (86 s, `check_c_5_3.pkl`, 5.4's reference): every assertion
  of the handoff's §4 "Since 5.3", every shared column and record equal to `check_c_5_2b.pkl`,
  both EDA figures' JSON identical; rerun after the review's amendment (81 s), the same. Two
  mutations caught (on the version before the amendment): the selection read on `train` (the
  best run's model differs), `error_x`'s two arrays swapped. No `src` change: the suite and mypy
  run at 5.4's close. The notebook's md5 is `9a66a347ed3fc928f14a43bde04da1b9`.
- **Planned 2026-11-07** (approved in the next session). The text from "Since 5.2" on is the
  plan before this one, kept as history.
- **The `n_resamples` measurement (done, read-only;** `probes/probe_5_3.py` on `probe_5_2.pkl`,
  the 100-trial runs' models; log `probe_5_3.log`, per-seed bounds `probe_5_3_bounds.csv`):
  - setup: validation, whole neighborhoods, seeds 42–46, at 1000 and 10,000 replicates; the
    best models by deviance and by RMSE, and their paired difference (Model 2 − Model 1); the
    deviance and RMSE, per cohort and the composite;
  - the bounds' SD over seeds as % of the interval's width: at most **4.6%** at 1000 (the RMSE
    difference's composite; most cells 2–3.5%), at most 1.5% at 10,000;
  - one call (6 metrics × 4 cohorts): 6.9 s at 1000, 69.1 s at 10,000 (the deviance and RMSE
    alone at 1000: 2.1 s);
  - **the user kept 1000** (under the 5% rule); the test's pin is unchanged.
- **Other facts from the probe:**
  - `score` lies inside `[ci95_lower, ci95_upper]` in all 1152 rows (384 + 768);
  - the best run by validation composite score:
    - Model 1: the same run by either metric (`lgbm_poisson`, exposure on: RMSE 4.5586,
      deviance 2.1296);
    - Model 2 by deviance: `lgbm_gaussian+logistic`, exposure on (RMSE 4.5393);
    - Model 2 by RMSE: `lgbm_poisson+logistic`, exposure off (RMSE 4.5388); the two are within
      noise of each other (0.0005 against intervals about 1.0 wide), so under RMSE Model 2's
      inspected run changes;
  - the composite paired difference's interval contains 0 under both rules: RMSE −0.16 to
    0.12, deviance −0.12 to 0.06;
  - keeping every run's replicates in its record: 8.8 MiB at 1000 (375 KiB a run), 88 MiB at
    10,000.
- **The user's change of plan (2026-11-07; amends D34 and D36):**
  - each model's best run is chosen by **one selection metric**, configurable between RMSE and
    the Poisson deviance (default RMSE, D34), as a notebook setting;
  - it is chosen on the **validation point score** of the composite. Why (the user asked for
    the best practice): the bootstrap's mean ranks the runs the same as the point score; the
    bootstrap measures uncertainty, it does not rank better; cv is no fairer, since each run was
    tuned on it; the chosen model's validation score is optimistic, and the test part (PR 7)
    gives the fair one;
  - **only the two best models are bootstrapped**, on every metric; the other runs keep point
    scores only (a click about 5 min shorter);
  - the forest plot is **one function with the metric as a parameter** (Model 1's and Model
    2's score tables have the same shape, so one function serves both).
- **Change (the plan):**
  - **section 2:** `selection_metric = "rmse"  # "rmse" or "poisson_deviance"` in
    `EXPERIMENT_SETTINGS` (printed, logged, pinned by the notebook test);
  - **the metrics cell:** `SELECTION_METRIC` resolved from it by `if`/`elif` with a raising
    `else`; the comment: it picks each model's best run on validation and is the forest plots'
    metric; the deviance still tunes;
  - **the two loops** (Model 1's runs, Model 2's pairs): no bootstrap;
    `_scores_by_dataset[_dataset_name] = _dataset_scores.to_frame("score")`; a record's
    `scores` keeps its `(dataset, cohort, metric)` index with the column `score`;
    `CI_BOUND_NAMES`, `alpha` and `bootstrap_evaluator` leave both cells' parameters;
  - **the tables:** the cells' code unchanged; `model1_scores_df` 384 × 8, `model2_scores_df`
    768 × 6; the two captions "(score)";
  - **a new cell, the best runs** (after Model 2's tables, before the "Inspect the best runs"
    button; no button of its own):
    - each model's best is the `min` of its runs' validation composite `score` on
      `SELECTION_METRIC`; `best_model1` and `best_model2` are their stored models;
    - each predicts validation once; `bootstrap_evaluator.evaluate(..., metrics=METRICS,
      cohorts=[*COHORTS, "composite"])` makes 2 calls (about 14 s; validation only, since a
      model's bootstrap on its own training rows says little);
    - `best_replicate_scores = {"model1": …, "model2": …}` for 5.4;
    - `best_scores_df`: a `model` label (e.g. `model1: lgbm_poisson, exposure True`) × cohort
      × metric, with `score, mean, median, sd, ci95_lower, ci95_upper` (5.2's statistics,
      moved here);
    - it shows the sentence naming both runs and `mo.ui.table(best_scores_df.round(4))`;
  - **the inspection's selection cell is removed:** the importance and SHAP cells keep their
    `mo.stop(not run_inspection.value)` and read `best_model1` and `best_model2` from the new
    cell; the importance's score stays the deviance until 5.4;
  - **a new cell, `forest_figure(scores, metric_name, lower_column, upper_column,
    rule_at=None)`**, defined once like `trials_table`, reading no notebook global:
    - `scores` is long, with the columns `model, cohort, metric, score, <lower>, <upper>`;
    - one panel per cohort (`make_subplots(rows=1, cols=n_cohorts, shared_yaxes=True)`), each
      on its own x scale; one row per model, the first on top;
    - the dot at `score`, the bar from lower to upper (`error_x` `array` / `arrayminus`);
      `rule_at` draws a vertical line (5.4: 0);
    - the dataviz palette (series 1 `#2a78d6`, surface `#fcfcfb`, ink `#52514e`) and
      `plotly_white`, as in 3.1j;
  - **a new cell, the forest plot:** `forest_figure(best_scores_df, SELECTION_METRIC.name,
    CI_BOUND_NAMES["lower"], CI_BOUND_NAMES["upper"])` under a caption; it returns
    `(best_forest_figure,)` for Check C;
  - **section 6's text:** point scores for the runs; the selection rule and why; the
    optimism; the two best bootstrapped and plotted; `n_resamples` kept by measurement;
  - **the test:** `"selection_metric": "rmse"` in the `experiment_settings` pin.
- **Checks (the plan):**
  - **Check A:** the notebook test 14, ruff 15 (none new), the two format hunks, `marimo
    check`;
  - **Check B:** the defs hold `SELECTION_METRIC`, `forest_figure`, `best_scores_df` and
    `best_replicate_scores`;
  - **Check C** (about 3 min; 48 bootstrap calls become 2), against `check_c_5_2b.pkl`:
    - `SCORE_COLUMNS = ["score"]` for the records and tables;
    - the best runs are each the `min` of validation composite `rmse`, named in the output;
    - `best_scores_df` 48 rows: `score` equal to the records'; the statistics recomputed
      exactly from `evaluate` on the two best models; `best_replicate_scores` equal;
      `ci95_lower ≤ median ≤ ci95_upper`; `sd > 0`;
    - the figure: one trace per cohort panel, y the two labels, x the scores,
      `error_x.array = upper − score` and `arrayminus = score − lower` exactly; four subplot
      titles;
    - `compare.py` on the shared columns; both EDA figures' JSON identical;
    - `importance_df`: equal where the selected run is unchanged, otherwise recomputed
      independently with `PermutationImportance` (the new pickle is 5.4's reference);
  - **two mutations:** the selection read on `train`; `error_x`'s two arrays swapped;
  - no `src` change: the suite and mypy run at 5.4's close.
- **Commit:** `feat(notebook): each model's best run chosen by a selection metric and bootstrapped; a forest plot of the two`

**The plan before 2026-11-07 (history):**
- **Since 5.2 (the user's note):** each loop already computes `mean`, `median`, `sd` and the
  bounds from the replicate scores; 5.3 keeps the forest plots and the `n_resamples`
  measurement. Whether the records keep the replicates (5.4's paired difference reads them)
  is 5.3's planning question; the text below is the plan before 5.2.
- **Change:** the two loops keep `score` (the point estimate) and the replicates per dataset;
  the tables cells compute `mean`, `sd`, `ci95_lower`, `ci95_upper` from the replicates
  (`alpha` stays a setting; `CI_BOUND_NAMES` from it) into `model1_scores_df` / `model2_scores_df`
  with the same rows as today plus `sd`; a forest plot per model (plotly): each run's comparison
  metric with its interval, per cohort and the composite, validation (3.1h's palette).
- **The `n_resamples` measurement (a read-only probe before the edit):** one model (the best
  `lgbm_poisson` of a 3-trial run), the validation rows, five seeds at 1000 and at 10,000: the
  spread (SD over seeds) of each bound against the interval's width, for RMSE and the deviance;
  the seconds per call. Recommended rule: keep 1000 if the bounds' spread is under 5% of the
  width; else 10,000 for the comparison section only (the best models) and 1000 for the 24 runs.
  The user decides on the numbers; the test's pin follows.
- **Checks:** Check A (the pin), B, C (the tables' `mean`, `ci95_*` equal the reference exactly;
  the forest figures' traces one per run, the x the metric's values).
- **Commit:** `feat(notebook): each run's scores with their bootstrap SD and interval; a forest plot per model`

### [ ] 5.4 The best models' paired difference; the importance on the selection metric
- **Moved to PR 7 (2026-11-07, the user):** PR 5 closes at 5.3; the EDA work (PR 5b) and the
  split (PR 6) come first, and 5.4 is done in `03_hyperparameter_tuning.py`, with the forest
  plots' layout fixed. The difference is `total_times_probability − independent_cohorts` (the
  keys of `best_replicate_scores` since 5.3's review).
- **Re-scoped 2026-11-07** (after 5.3's plan; the best-run rule and `SELECTION_METRIC` move
  to 5.3):
  - a new cell after 5.3's forest plot: `best_replicate_scores["model2"] −
    best_replicate_scores["model1"]` per replicate (the same rows: one evaluator), per cohort
    and the composite, every metric;
  - its point difference (the two `score`s), mean and interval;
  - `forest_figure(..., rule_at=0)` on `SELECTION_METRIC` ("0 inside the interval: no
    difference shown");
  - the permutation importance's score `SELECTION_METRIC` (today `POISSON_DEVIANCE`), and the
    importance table's caption;
  - section 6's text.
  - **Checks:** Check A–C (the difference equal to `scores2 − scores1` per replicate exactly;
    `importance_df` recomputed with the selection metric, a new reference); the suite and mypy
    before closing PR 5.
  - **Commit:** `feat(notebook): the best models' paired bootstrap difference; the importance on the selection metric`
- **The plan before 2026-11-07 (history):**
- **Change:** the metrics cell gains `COMPARISON_METRIC = RMSE` (the comment: compares models,
  in children per building; the deviance tunes); the inspection's best-run rule and the
  permutation importance's score read it (today `POISSON_DEVIANCE` at the notebook's lines
  2234, 2240, 2277, 2320); the forest plots of 5.3 read it; a new cell after the inspection's
  choice: the two best models' paired difference per replicate on validation (`model2 − model1`
  per cohort and the composite, the same replicate rows), its mean and interval, a forest plot of
  the difference with a rule at 0 ("0 inside the interval: not significant"); section 6's text.
- **Checks:** Check A–C (the best runs by RMSE on validation named; the difference's rows equal
  `scores2 − scores1` per replicate exactly; the inspection's `importance_df` recomputed with
  RMSE, a new reference).
- **Commit:** `feat(notebook): RMSE compares the models; the best models' paired bootstrap difference`

## 4b. PR 5b: Model 2's two tuning methods in the EDA (`feat/model2-tuning-methods`)

**Planned 2026-11-07** (the user's notes after 5.3; PR 5 merged as #23, `c3bdee8`). The user
finds it hard to move between the EDA and the tuning in one notebook, so the EDA's work comes
before the split (PR 6), and the tuning's (5.4 and the rest) after it (PR 7).

**The user's two questions, answered:**
- *Why `trials_table`, rather than saving the attributes during the runs?* They are saved:
  since 3.1e (M12) each trial records its cv score on every metric as a user attribute in the
  study's journal file. The function rebuilds each trial's whole model (each component's
  transformer name and hyperparameters, and the component a Model 2 step holds fixed), which the
  flat params do not hold; Optuna's `trials_dataframe()` gives the flat form (3.1f's table,
  replaced by the user in 3.1i). Recording the components in the tuner instead needs a `src`
  change in the four tune methods (a transformer does not know its name) and every stored study
  rebuilt. **The user's answer:** keep the function, under a more informative name.
- *Coordinate descent or the joint method for Model 2, now that each study is faster?* Fewer
  trials favour coordinate descent: at 100 trials a joint study searches 23–33 dimensions with
  3–4 trials each, a coordinate step has 100; coordinate descent runs 10 studies (1000 trials),
  the joint method 16 (1600, half with a LightGBM classifier); and one round is exact under the
  deviance (D29, D32). At 400 trials (the handoff's §2a) coordinate descent's cv was better in
  9 of 16 pairs, the same in 6, worse in 1; for the EDA's setting the joint method had the
  better cv (2.0489 against 2.0622) and the worse validation (2.1285 against 2.0965). **The
  user's answer:** compare the two in the EDA at 100 trials; section 6 stays coordinate descent
  and gains a `model2_tuning_method` setting only if the joint method wins.

### [x] 5b.1 `trials_table` → `build_trials_table` (notebook; a rename only)
- **Done (2026-11-07), as planned;** every line number held on `fef07bc` (895 the def, 939 its
  return; Model 1's EDA cell's parameter 970 and call 1027; Model 2's parameter 1083, calls 1136,
  1189): 7 lines, the new parameter after `build_study_name` and after
  `build_total_times_probability_model` (sorted). No test, `src` module or current guide named it;
  the handoff's §1 and §5 (F3, F4) renamed. Checks: Check A (the notebook test 14 passed, ruff 15,
  the user's two format hunks, `marimo check` no issue); Check B (`build_trials_table` defined,
  `trials_table` absent); Check C (86 s, `check_c_5b_1.pkl`, 5b.2's reference): every assertion,
  every frame and record equal to `check_c_5_3.pkl` exactly (`best_runs_val_scores_df` too, `mean`
  included), both EDA figures' JSON identical. The notebook's md5 is
  `e0f2ba6c32ead22ebee081a4f0a27e3f`. Found for 5b.2: section 5's text says "20 random startup
  trials" (10 since 4.1).
- The name follows `build_study_name`. The def and its return; the two EDA run cells'
  parameter lists (re-sorted) and their three calls. No behaviour change. The current-state
  docs; history stays as written; Check B asserts the new name and not the old.
- **Checks:** Check A, B, C equal to `check_c_5_3.pkl` exactly.
- **Commit:** `refactor(notebook): trials_table renamed build_trials_table`

### [x] 5b.2 The joint method in the EDA (notebook)
- **Done (2026-11-07), as planned;** every line number held on `82ccac2`. The notebook:
  - a run cell after Model 2's EDA cell: per objective, one
    `tune_total_times_probability_model` study (`lgbm_poisson+logistic`, exposure on, seeded
    `seed`, stored as `total_times_probability_joint__total_family=…__probability_family=…__use_exposure=…`);
    each trial's two components read back with `from_params(name="total")` /
    `from_params(name="probability")` for `build_trials_table`; the best trial's components
    (`_components_by_trial[_study.best_trial.number]`) refit and scored on validation →
    `eda_model2_joint_df` (12 rows), `eda_model2_joint_trials_df` (3 × `n_trials`, no `step`);
  - the summary `eda_best_df` 9 rows (`model2_joint`); the two figure cells treat it as Model
    1's branch (one study: no rule, boxes by objective, `overlay`) and show a third figure;
    the tables cell two more tables; Model 2's captions say "by coordinate descent";
  - section 5's text: why the two methods are compared; the stale "20 random startup trials"
    (10 since 4.1) now "random startup trials".

  Checks: Check A (the notebook test 14 passed, ruff 15, the user's two format hunks, `marimo
  check` no issue); Check B (the joint frames absent from a script run); Check C (86 s,
  `check_c_5b_2.pkl`, 5b.3's reference): every assertion since 3.1c, the 3.1j–3.1l ones
  covering the joint model, and a 5b.2 block (the three joint studies recomputed in memory
  equal to the table, the trials table equal to each trial's params and user attributes, the
  summary's joint rows, the three stored files' names); every shared frame and record equal to
  `check_c_5b_1.pkl`, the 6 and 12 shared figures' JSON identical. **The joint EDA's Check C
  values** (3 trials, `cv_best` log loss / deviance / RMSE): 1.0833 / 2.1543 / 4.4258 (the log
  loss equals coordinate descent's: only the probabilities count). Two mutations caught: the
  joint study under `total_times_probability_coordinate_descent_total` (no joint file stored);
  the refit from the last trial (the recomputation differs under the log loss). No `src`
  change. The notebook's md5 is `6731d509d2b5298e4d38a99480f700dc`.
- **A run cell after Model 2's EDA cell** (behind "Run the objective EDA"): per objective in
  `EDA_OBJECTIVES`, one study of `tuner.tune_total_times_probability_model` for
  `lgbm_poisson+logistic`, exposure on, seeded `seed`, stored as
  `build_study_name("total_times_probability_joint", {total_family, probability_family,
  use_exposure}, objective, seed)`; each trial's two components read back with
  `from_params(name="total")` / `from_params(name="probability")`; the best refit and scored
  on validation → `eda_model2_joint_df` (12 rows) and `eda_model2_joint_trials_df`
  (3 × `n_trials`, no `step`).
- **The summary** `eda_best_df` 9 rows (`model2_joint` added); **the figures** (`eda_figures`,
  `eda_metric_figures`) and their displays gain the joint model (Model 1's branch: no step
  rule); **the tables cell**; **section 5's text** (why the two are compared).
- **Cost:** 3 new studies of 100 trials with the logistic classifier on the next EDA click.
- **Checks:** Check A, B; Check C: the joint EDA's three `cv_best` as new baselines, the joint
  studies recomputed in memory equal to the table, 9 trials rows with each objective column's
  minimum its `cv_best`, `eda_best_df` 9 rows, the figures; the rest equal to the reference;
  two mutations (the joint study under the coordinate step's name; the components read with the
  wrong family).
- **Commit:** `feat(notebook): the objective EDA compares Model 2's two tuning methods, coordinate descent and joint`

### [x] 5b.3 The reading at 100 trials (docs)
- **Done (2026-11-07).** The EDA clicked on the real cache (a copy of the notebook with the two
  figure cells' returns restored and `studies` symlinked to the repo's; only "Run the objective
  EDA" clicked; `PYTHONWARNINGS=error`, no warning): 114 s; the 3 joint studies tuned and stored
  (100 trials each, none failed), the 25 stored files byte-identical; the notebook unchanged.
  Each objective's best (cv on its objective; validation composite deviance / RMSE; the study's
  seconds):

  | tuned on | coordinate descent (200 trials) | joint (100 trials) |
  |---|---|---|
  | log loss | 1.0828; 2.0539 / 4.4523; 43.2 s | 1.0828; 2.1191 / 4.5358; 39.9 s |
  | deviance | 2.0791; 2.1373 / 4.5824; 57.8 s | 2.0735; 2.1493 / 4.6040; 32.7 s |
  | RMSE | 4.2862; 2.1573 / 4.6262; 44.3 s | 4.3247; 2.0990 / 4.5132; 31.5 s |

  The paired bootstrap of the two best models per objective (refit from the stored studies,
  equal to the EDA's tables; `BootstrapEvaluator(1000, random_state=seed)`, validation, the
  composite), joint − coordinate descent, 95%: log loss deviance +0.065 (0.010 to 0.127), RMSE
  +0.084 (0.012 to 0.156); deviance deviance +0.012 (−0.031 to 0.051), RMSE +0.022 (−0.035 to
  0.073); RMSE deviance −0.058 (−0.143 to 0.014), RMSE −0.113 (−0.242 to 0.007). The search:
  the joint deviance study at 2.0771 by trial 25, its best at 61; coordinate descent's step 2
  its best at 34. The figures were saved for the user's viewer app (6 render). **The decision:
  D40** (§0), the joint method in section 6 in PR 7 (7.0). No code change.
- The EDA click on the real cache (the 3 joint studies tuned, the rest loaded): per objective,
  each method's `cv_best`, its refit's six validation metrics, its seconds and failed trials;
  the figures read.
- The decision (D40): section 6 stays coordinate descent; or, if the joint method wins beyond
  the noise (cv about 0.02, validation about 0.04 at 400 trials), the `model2_tuning_method`
  setting and the joint loop go into the tuning notebook in PR 7.
- **Commit:** `docs(notebook): Model 2's tuning methods read at 100 trials`

## 5. PR 6: two notebooks (`refactor/two-notebooks`)

### [ ] 6.1 The shared setup: the design
- **Question, probed with evidence before the edit:** sections 1–4 (lines 1–626: the data,
  the settings, the split, the 260-line feature declarations) in both notebooks as (a) a
  duplicate, (b) a plain module beside the notebooks (`notebooks/experiment_setup.py`:
  `simulate_population(config)`, `split_parts(df, …)`, `feature_transformer_candidates()`,
  taking the settings as arguments, the settings cells staying in each notebook), or (c)
  marimo's reusable cells (`@app.function` / `@app.class_definition` imported from one notebook;
  probe that marimo 0.24 imports them from a file named with a leading digit via
  `importlib`, and what `marimo check` says). Measured: the lines duplicated, the names each
  notebook reads, whether the notebook test's script run still prints the settings lines.
  Recommended before the probe: (b), the package rule's spirit (mechanism out of the notebook,
  the settings in it), one source of truth for the declarations, no marimo-version coupling.
- **Done when:** the user has chosen; the chosen module or cells exist with the same outputs
  (the printed `data`, `experiment_settings`, `split` lines equal).
- **Commit:** `refactor(notebook): the experiment's setup shared by the notebooks` (if b or c)

### [ ] 6.2 `02_objective_eda.py`
- **Change:** sections 1–5 (and the shared setup); the EDA's button, runs, tables and figures;
  the notebook test split: `test_objective_eda_notebook.py` (the printed settings, the tuning
  cells behind `mo.stop`). `README.md`, `docs/README.md`, the handoff name the two notebooks.
- **Checks:** Check A per notebook; Check B; Check C's EDA assertions on the new file equal the
  reference's EDA frames exactly.
- **Commit:** `refactor(notebook): the objective EDA in its own notebook`

### [ ] 6.3 `03_hyperparameter_tuning.py`
- **Change:** sections 1–4 (shared), 6 (tuning, inspection, the comparison of 5.4); `02_experiments.py`
  removed; the characterization test renamed/split; `SPLITTING.md`, `HYPERPARAMETER_TUNING.md`,
  `MODEL_INSPECTION.md` references.
- **Checks:** Check A per notebook; Check C's section-6 assertions on the new file equal the
  reference's frames exactly; the 100-trial click loads the cached studies (the cache's
  directory is keyed by the data, not the notebook's name).
- **Commit:** `refactor(notebook): the tuning in its own notebook; 02_experiments.py removed`

## 6. PR 7: SHAP per cohort and the test evaluation (`feat/test-evaluation`)

### [ ] 7.0 Model 2 by the joint method in section 6 (notebook; D40)
- **Change:** the tuning notebook's Model 2 runs become one `tune_total_times_probability_model`
  study per family pair and exposure (16, in a loop over run settings, seeded in the list's
  order, stored as `total_times_probability_joint__…`), each best refit and scored; the three
  coordinate-descent cells and `MODEL2_TOTAL_RUNS` / `MODEL2_PROBABILITY_RUNS` removed;
  `MODEL2_FIXED_PROBABILITY_COMPONENT` stays for the EDA. To plan with the user: the run
  settings' order and seeds, the records' fields, Check C's new Model 2 baselines (16 joint
  `cv_best` at 3 trials), the notebook test's pins, section 6's text.

### [ ] 7.1 SHAP per cohort (notebook)
- **Change:** the importance cell: `mean_abs_shap` per column × cohort (`np.abs(values).mean(axis=0)`,
  a frame columns × cohorts, the composite beside as today) and the rank per cohort; a
  beeswarm-style plotly figure per model and cohort (the SHAP value per explained row and
  column, the color the column's value, the columns ordered by the cohort's mean |SHAP|), shown
  in a dropdown of the cohorts or stacked; section 6's text.
- **Checks:** Check A–C (the per-cohort means average to the reference's `mean_abs_shap`
  exactly; the figures' traces one per column).
- **Commit:** `feat(notebook): SHAP per cohort, the values and the feature order`

### [ ] 7.2 The final refit and the test scores (E11)
- **Change:** a new section 7 "The test evaluation": a button "Evaluate on the test part"; a cell
  that refits each model's best run (by the comparison metric on validation; the stored
  components, `build_*_model`, `random_state=seed`) on `train + val` (`pd.concat` of the two
  parts' `AgeGroupDataset`s; `test_age_group` returned by the split cell), predicts the test
  part once, and keeps `test_predictions` and `test_scores` (every metric per cohort and the
  composite, with the bootstrap's replicates over the test rows' neighborhoods) per model; the
  two models' paired difference on test; the tables and forest plots as in section 6. The
  section's text: the test part stands for production, scored once, never used to choose
  (E11); the notebook test checks the cell is behind `mo.stop(not `.
- **Checks:** Check A (the new guard), B, C (on the copy: the test scores' rows 2 models × 4
  cohorts × 6 metrics; the refit's training rows 1223; the test predictions' length 306).
- **Commit:** `feat(notebook): the best models refit on train and validation, scored once on the test part`

### [ ] 7.3 The reading of the test predictions
- **Change:** cells after 7.2, per model: (a) the residual table, a row per test building
  (`building_id`, `neighborhood_id`, the features, the true and predicted counts per cohort,
  the composite's absolute error), and the best K and worst K (`K = 10`, a setting) with what
  they share: their neighborhoods' counts among the K, their exposure and share quantiles
  (an anomaly: a value outside the training rows' 1st–99th percentile), one table each; (b)
  the count "confusion matrix": per cohort, the true count × the rounded predicted count as a
  heatmap (plotly), the diagonal the exact hits; (c) the total true against predicted (a scatter
  with the identity line, colored by exposure); (d) the cohort shares true against predicted
  (per cohort, the true share `y / total` against the predicted share: Model 2's classifier's
  probabilities, Model 1's `pred / pred.sum`), a scatter per cohort with the identity line and
  a calibration curve by deciles.
- **Design questions, in the sub-task's plan:** the error that ranks the buildings (the
  composite's absolute error, recommended: readable; or the Poisson deviance per row); which
  hover text and which figure per cell.
- **Checks:** Check A–C (the residual table's rows 306, its errors equal `|y − ŷ|.sum(axis=1)`;
  each heatmap's cells sum to 306; the share scatter's x and y from the predictions).
- **Commit:** `feat(notebook): the test predictions read: best and worst buildings, the count heatmaps, the totals and the shares`

## 7. Facts verified on 2026-11-04 (read-only probes)

- Optuna 4.9.0 `JournalStorage(JournalFileBackend(path))`: one JSON-lines file per study;
  `create_study(..., storage, study_name, load_if_exists=True)` and `load_study` return a real
  `Study`; `get_all_study_names`; no warning under `-W error`. sqlalchemy 2.0.52 and pyarrow
  25.0.1 are installed (not needed).
- `CrossValidationTuner._optimize` (cross_validation.py:315–331) is the only `create_study`.
- D20: a 3-trial study draws the same points at a random start of 10 or 20, so 4.1 keeps the 24
  baselines.
- `tests/characterization/test_experiments_notebook.py:118-131` pins the experiment settings;
  `:189-200` requires `mo.stop(not ` in every cell holding `tuner.tune_`.
- `BootstrapEvaluator.evaluate` (bootstrap.py:45–127): cluster bootstrap over `groups` or rows;
  `scores` is `n_resamples × (cohort, metric)` before the quantiles.
- `ShapExplainer.explain` returns values per row, column and cohort (`output_names`); the
  notebook's `importance_df` averages over rows and cohorts (line 2368).
- `PermutationImportance.compute_importances`: the naming precedent for `compute_metrics`.
- `scoring.evaluate`'s callers: scoring.py, inspection/bootstrap.py, the notebook (4 calls),
  tests (test_scoring, test_inspection_bootstrap, test_hyperparameter_tuning_cross_validation),
  the guides (HYPERPARAMETER_TUNING.md, MODEL_INSPECTION.md), the handoff, the metrics plan.
- `POISSON_DEVIANCE`'s readers in the notebook: the tuner cell (794), the best-run rule (2234,
  2240), the permutation importance (2277, 2320).
- E11 (EXPERIMENT_NOTEBOOK_PLAN.md:423–433): the final refit on train + validation, scored once
  on test; the winner rebuilt from its components.
- `notebooks/` holds `01_eda.py` and `02_experiments.py`; sections 1–4 are lines 1–626; the
  split cell returns `train_age_group`, `val_age_group` (not `test_age_group`); `project_root`
  and `config_hash` exist in section 1.
- The 400-trial click: Model 1 821 s, Model 2 752 s (joint 2871 s), the bootstrap about 7 s per
  call (24 models × 2 datasets); a click about 31 min.
- `.gitignore` has no `studies/` entry.
- The non-slow suite at `b7da88d`: 1387 passed, 1 skipped, 1 xfailed, 19 deselected; mypy 28
  files, no issue.
