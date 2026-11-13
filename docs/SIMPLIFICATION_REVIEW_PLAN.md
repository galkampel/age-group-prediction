# Plan: the review for overcomplication of the tuning and the figures

**For:** the user, who accepts or rejects each sub-task, and the session that implements an
accepted one (written so that Claude Opus 5.5 can implement it from this file and
[the handoff](EXPERIMENT_NOTEBOOK_HANDOFF.md) §4). One sub-task per approval. Tick a sub-task
when it is done and update the status line.

**Status (2026-11-13):** S1 committed (`5bf8e48`), S2 committed (`c01483f`); **S3 and S4 done
together (the user's request) and checked, uncommitted**, on HEAD `c01483f`, branch
`feat/mlflow-runs`: every check passed, Check C equal to S2's run with nothing excepted (both
importance tables bit-equal), S3's mutation caught. Next: the user commits S3 + S4, then S6
(Opus 5.5); then S7 and S5 (Fable); then S8. The notebooks' md5: `02_objective_eda.py`
`373c1c39f717935cb08ce14ff6e05a3a` (unchanged), `03_hyperparameter_tuning.py`
`986a753e9e1da986743cda40450f7f2c`. Check C's references for the next sub-task:
`check_c_s34_tuning.py`'s run `check_c_s34_tuning.pkl` and `check_c_s34_eda.pkl` (the handoff's
top status names the scratchpad).

**Status (2026-11-12):** the review is written and **decided** (the user's answers): S1–S8
all accepted (S7 as option (b); S4 without waiting for the manual check); L1–L8 all left as
they are. No sub-task is implemented yet: nothing in the notebooks, `src` or the tests has
changed. Next: S1. HEAD `bd90f66` (on `main`'s `574287e`), branch
`feat/mlflow-runs`. The notebooks' md5: `02_objective_eda.py`
`373c1c39f717935cb08ce14ff6e05a3a`, `03_hyperparameter_tuning.py`
`47c8ce343350475eebe8c2007a00397e`. The MLflow plan follows these decisions (the handoff's §6,
step 2).

## 1. The question and the verdict

The question (the handoff's §6, step 1): are the hyperparameter tuning (the package and the
notebooks' cells) and the visualization overcomplicated, and what can be simplified?

What was read: every file of `src/age_group_prediction/hyperparameter_tuning/`; the tuning
cells and every figure of `notebooks/03_hyperparameter_tuning.py` (from line 627) and of
`notebooks/02_objective_eda.py` (from line 876); the names the notebook tests read.

- **The package (about 1,240 lines) is not overcomplicated.** Every piece is read by a
  notebook or is a recorded decision (D9–D11, D25–D27). One small merge is worth doing (S8).
- **The weight is in `03` after the runs** (lines 1516–3578, about 2,060 lines): the same
  blocks are written two to seven times, and one menu has a side effect (S4).
- **The EDA's coordinate descent** is the largest block that could leave, since D40 settled
  its question (S7).

Each finding is judged against the user's rules: the smallest change that works; a valid path
kept; no function unless it is reused; one source of truth; no field or check that nothing
reads; `if`/`elif` with a raising `else`.

## 2. The sub-tasks at a glance

| # | Sub-task | What you see change | Size | Implement with |
|---|---|---|---|---|
| S1 | The bootstrap statistics and the paired difference as two functions | nothing | small | Opus 5.5 |
| S2 | The inspection as one cell over `best_runs` | nothing | medium | Opus 5.5 |
| S3 | The importance table as one function (after S2) | nothing | small | Opus 5.5 |
| S4 | The permutation importance always on the selection metric | a menu no longer sets it | small | Opus 5.5, after the user's one manual check |
| S5 | One palette and one plotly template for every figure | nothing intended; every figure rebuilt | large | **Fable** |
| S6 | The EDA's first cell reads the best components once | nothing | tiny | Opus 5.5 |
| S7 | Coordinate descent leaves the EDA notebook | the EDA shows two models, not three | large | **Fable** |
| S8 | `from_params`'s filter in one place (`src`) | nothing | tiny | Opus 5.5 |

**Why Fable for S5 and S7:** each touches many cells at once, changes Check C's references and
scripts on purpose, and needs a design choice made from a probe (S5: that the rendering is the
same; S7: which check blocks go). The others are local edits whose outputs must stay equal, so
Check C says plainly whether they are right.

**The order, if all are accepted:** S1, S2, S3, S4, S6, S7, S5, S8. S3 needs S2. The line
numbers below are HEAD `bd90f66`'s; after any earlier sub-task, find the code by the quoted
names, not by the numbers.

## 3. How an accepted sub-task is done

The standing rules are the handoff's §4 ("How to work", "Git and the editor", "Traps", "The
user's style"). The ones that matter most here:

1. **Before anything:** `git log --oneline -1`, `git status --short`, both notebooks' md5
   (`cat <file> | md5`), and `find .git src notebooks tests docs configs -flags +dataless -type f
   | grep -vc __pycache__` (must be 0). If the md5 differs from this file's status line, stop
   and show the user the diff.
2. **Tell the user before editing a notebook** (their marimo session may be open) and remind
   them after to reload it, not to save from it.
3. **Never commit, never `git checkout`, `stash`, `reset` or `mv`.** The user commits. No
   `Co-Authored-By` line.
4. **Formatting:** write a new cell in the scratchpad, run `ruff format` on it alone, splice it
   in. Never format the whole notebook (it carries two format hunks of the user's). A cell
   lists every notebook name it uses as a parameter, sorted. No new ruff finding (`03` has 14,
   `02` has 11).
5. **Functions in a notebook cell** read no notebook global: everything comes in as an
   argument. No private (underscore) helper functions.
6. **The checks** (scripts in the scratchpad the handoff's top status names; copy them to the
   new session's scratchpad first, set each `S=`, compile the pycache):
   - copy `checks/run_close_7.sh` to `checks/run_<sub-task>.sh` and change only the two output
     pickles' names (`check_c_<sub-task>_eda.pkl`, `check_c_<sub-task>_tuning.pkl`) and the
     references, which become `check_c_close_7_eda.pkl` and `check_c_close_7_tuning.pkl` (or
     the previous accepted sub-task's pickles);
   - run it in the background with stdin closed, logged to the scratchpad (about 15 minutes);
   - it runs the non-slow suite (1411 passed at PR 7's close), mypy (28 files), the notebook
     tests (32), ruff and the format hunks, `marimo check`, Check B on both notebooks, and
     Check C on both 3-trial copies with the comparison against the reference.
   - **Pass means:** every frame and record equal to the reference, except what the sub-task's
     "Validate" names.
   - If a sub-task renames a notebook name, the scratchpad scripts that read it
     (`check_b_tuning.py`, `check_c_7_1e_tuning.py`, `make_copy_tuning.py`, and the EDA's three)
     are edited to the new name, and nothing else in them.
7. **A mutation** where logic moved: break the new code on a copy in one small way, run the
   check that should catch it, show that it fails.
8. **Never run the test evaluation on the user's cache** (E11). Check C clicks it on the
   3-trial copy only.
9. **The stop:** a file-by-file summary, the checks' results, both md5, and the commit message
   in a copyable code block. Then tick the sub-task here and update the status lines here and
   at the top of the handoff.

Kept in every sub-task: the study cache's file names and the run records' keys.

## 4. The sub-tasks recommended to accept

Lines are `03_hyperparameter_tuning.py`'s unless marked `02:` or a `src` file. Mark each
**Decision** line `accept` or `reject`.

### [x] S1. The bootstrap statistics become one function, the paired difference another

- **Model:** Opus 5.5.
- **Evidence:** the frame `score, mean, median, sd, ci95_lower, ci95_upper` is built from
  replicates four times (1443–1452, 1846–1859, 2511–2525, 2537–2550). The paired difference
  frame is built twice (1836–1863 on validation, 2531–2554 on test).
- **Trade-off:** two functions against about 55 repeated lines. The MLflow runs will log these
  frames, so one definition helps there too.
- **Recommendation:** accept.
- **Decision:** accept (the user, 2026-11-12).

**Steps:**
1. Add one cell after the bootstrap evaluator's cell (the one defining `bootstrap_evaluator`
   and `CI_BOUND_NAMES`), with parameter `pd`, defining and returning two functions:
   - `summarize_replicates(scores, replicate_scores, alpha, bound_names)`: returns
     `pd.DataFrame` with the columns, in this order, `score` (= `scores`), `mean`, `median`,
     `sd` (`.std()`), `bound_names["lower"]` (`.quantile(alpha / 2)`), `bound_names["upper"]`
     (`.quantile(1 - alpha / 2)`), all from `replicate_scores`. The same expressions as today,
     so the values are bit-equal.
   - `build_paired_difference(scores_df, replicate_scores, alpha, bound_names)`: the body of
     lines 1836–1863. It returns `(difference_replicates, difference_df)`, where the
     replicates are `replicate_scores["total_times_probability"] -
     replicate_scores["independent_cohorts"]`, the point scores come from `scores_df` indexed
     by `["model", "cohort", "metric"]`, the frame is `summarize_replicates(...)` with
     `.rename_axis(["cohort", "metric"]).reset_index()` and the column `model` inserted first
     with the value `"total_times_probability - independent_cohorts"`.
2. Replace the four statistics blocks and the two difference blocks with calls. The cells'
   exported names stay (`best_runs_val_scores_df`, `best_difference_replicates`,
   `best_difference_df`, `test_scores_df`, `test_difference_replicates`, `test_difference_df`).
   `_difference_name` in the two display cells is read from the frame
   (`…_df["model"].iloc[0]`, as line 2605 already does).
3. Add the two function names to each caller's parameters (sorted) and to Check B's expected
   names (`check_b_tuning.py`).

**Validate:** the runner passes; Check C equal to the reference with nothing excepted.
Mutation: `alpha / 2` → `alpha` in `summarize_replicates`; Check C must fail on a bound.

**Done (2026-11-13):** as the steps say; the new cell sits after the evaluator cell; the
validation difference cell no longer reads `pd`; the lines +69 / −69 (the functions' docstrings
weigh what the copies did). `run_S1.sh` (log `run_S1.log`): the non-slow suite 1411 passed, mypy
28 files, the notebook tests 32, ruff `03` 14 and `02` 11, the two format hunks each, `marimo
check` clean, Check B on both (with `CHECK B S1 OK`), Check C on the EDA (3 frames, 9 and 18
figures identical) and on the tuning (every assertion; all 16 shared frames and both records
equal to `check_c_close_7_tuning.pkl`, nothing skipped; `best_runs_val_scores_df`,
`best_difference_df`, `test_scores_df`, `test_difference_df` also bit-equal, `mean` included).
`run_mutant_S1.sh`: Check C fails on `ci95_lower` (100% of the values). Scratchpad scripts
changed: `check_b_tuning.py` (the two names), `make_copy_tuning.py` (its difference cell's anchor
line, now the `build_paired_difference(` call).

**Commit:**
```text
refactor(notebook): the bootstrap statistics and the paired difference each in one function (S1)
```

### [x] S2. The inspection becomes one cell over `best_runs`

- **Model:** Opus 5.5.
- **Evidence:** the two inspection cells (1957–1993, 1996–2028) differ only in the model.
  `best_model1` and `best_model2` (1415) repeat `best_runs[label]["model"]` (1418–1421). The
  test section already keeps its results by label (2704–2721). The readers pair the two models
  by hand (2128–2131, 2218–2221, 2373–2376).
- **Trade-off:** both models are computed in one cell. They are behind the same button today,
  so nothing is shown later than now.
- **Recommendation:** accept.
- **Decision:** accept (the user, 2026-11-12).

**Steps:**
1. Replace the two inspection cells with one. It keeps the `mo.stop(not run_inspection.value,
   …)` line and the first cell's comment, then loops `for _model_name, _run in
   best_runs.items()` and fills two dictionaries keyed by the model's label:
   - `best_runs_permutation[_model_name]` = today's `{cohort: {split: result}}` for
     `_run["model"]`;
   - `best_runs_shap[_model_name]` = today's `ShapExplainer(...).explain(...)` for it.

   The same constructor arguments and seeds as today, so the results are equal.
2. In the best-runs cell, delete `best_model1, best_model2 = …` and both names from its
   `return`. First confirm nothing else reads them: `grep -n 'best_model[12]'` on the notebook
   and the tests.
3. Update the three readers to loop over the dictionaries instead of the hand-written pairs:
   the importance table cell, the permutation figures cell, the beeswarm cell. Replace
   `model1_permutation` / `model2_permutation` / `model1_shap` / `model2_shap` in their
   parameters with the two new names.
4. Update `tests/characterization/test_hyperparameter_tuning_notebook.py` (the list at line
   694 names `model1_permutation`) and the scratchpad scripts that read the six old names
   (`check_c_7_1e_tuning.py` has 13 such lines; `check_b_tuning.py`; `make_copy_tuning.py` if
   it restores a return holding them): `defs["model1_shap"]` becomes
   `defs["best_runs_shap"]["independent_cohorts"]`, and so on.

**Validate:** the runner passes; Check C equal to the reference (the renamed keys read through
the new names). Mutation: use `best_runs["independent_cohorts"]["model"]` for both labels;
Check C must fail on the second model's SHAP.

**Done (2026-11-13, committed by the user as `c01483f`):** as the steps say; the readers loop
over `best_runs_shap` / `best_runs_permutation` (the importance table's loop body kept for S3);
the best-runs `return` kept multi-line, so `make_copy_tuning.py` needed no change; the notebook
test's "not before the click" list names the two new dictionaries. Notebook −74 / +29. `run_S2.sh`
(log `run_S2.log`): the suite 1411, mypy 28, the notebook tests 32, ruff `03` 14 / `02` 11, the
format hunks, `marimo check`, Check B on both (`CHECK B S2 OK`), Check C on both equal to S1's
run with nothing skipped (section 5's beeswarm still equal to 7.1b2's). Check C is
`check_c_s2_tuning.py`, made by `make_check_c_s2.py` (its 15 reads of the six old names by the
model's label, and an S2 assertion). `run_mutant_S2.sh`: Check C fails on the second model's
permutation importance (90 / 90 values). Left: `probes/make_copy_probe.py`, `probes/look_7_1e.py`,
`probes/probe_7_1.py` and `viewer/beeswarm_7_1b.py` still read the old names (S5.0's probe will
need them updated).

**Commit:**
```text
refactor(notebook): the best runs inspected in one cell, the results by model label (S2)
```

### [x] S3. The importance table becomes one function

- **Model:** Opus 5.5. Needs S2.
- **Evidence:** the table is built twice (2127–2152 on train and validation, 2730–2751 on
  test); the SHAP columns are the same code.
- **Trade-off:** one function with the splits as an argument against about 20 repeated lines.
- **Recommendation:** accept.
- **Decision:** accept (the user, 2026-11-12).

**Steps:**
1. Add one cell before the section-5 importance table cell, parameters `np, pd`, defining
   `build_importance_table(permutation_by_split, explanation)`. It returns a frame with the
   column `column` (the explanation's `feature_names`, via `rename_axis("column")
   .reset_index()`), then for each split in the mapping's order `perm_fi_mean_<split>` and
   `perm_fi_std_<split>`, then `mean_abs_shap_<cohort>` per `explanation.output_names`, then
   `mean_abs_shap_composite` (`np.abs(explanation.values).mean(axis=(0, 2))`, kept as it is).
2. Section 5 calls it with `best_runs_permutation[model]["composite"]` (its `train` and `val`
   splits, in that order); section 6 with `{"test": test_permutation[model]["composite"]}`.
   Each caller still inserts `model` first and concatenates into `importance_df` /
   `test_importance_df`.

**Validate:** the runner passes; both frames equal to the reference, columns in the same
order. Mutation: `importances_std` → `importances_mean` for the std column; Check C must fail.

**Done (2026-11-13, with S4 in one stop and one commit, the user's request):** as the steps say;
the new cell (`np, pd`) sits before section 5's importance table cell; both callers drop `np`.
The checks are S4's below (`run_S34.sh`); `importance_df` and `test_importance_df` bit-equal to
S2's run. `run_mutant_S34.sh` (this mutation): Check C fails on `perm_fi_std_train` (9 / 9
values).

**Commit:**
```text
refactor(notebook): the importance table built by one function for validation and test (S3)
```

### [x] S4. The permutation importance scores the selection metric, not a plot's menu

- **Model:** Opus 5.5. The manual check below is no longer a condition (see the decision);
  it is still worth a minute, to record whether the side effect was real.
- **Evidence:** `importance_metric` is read from the best-runs plot's "Metric" menu, with a
  special case for bias (1767–1776). This is why that menu has 3 options and the other two
  metric menus have 6 (1746–1750 against 1811–1815 and 2582–2586).
- **Expected side effect, not yet confirmed:** by marimo's rule, changing that menu re-runs
  the cells behind "Inspect the best runs" and the test explanation with their buttons
  unclicked, so their results disappear.
- **The user's manual check (one minute, on the real cache):** click "Inspect the best runs",
  wait for the tables, then change the best-runs plot's "Metric" from `rmse` to `bias`. If the
  importance tables and figures vanish, the side effect is real.
- **What changes for the user:** the importance on `poisson_deviance` is then reached through
  the `selection_metric` setting (section 2), not the menu. This reverses part of 5.4b.
- **Trade-off:** one way less to switch the importance's metric, against an `if`/`elif`/`raise`
  gone and no lost results.
- **Recommendation:** accept if the manual check shows the results vanish; otherwise the gain
  is only the removed branch, and it is the user's call.
- **Decision:** accept, whatever the manual check shows (2026-11-12; the user: "I am not sure. Use based on best practices"). The practice applied, the assistant's reading: a menu that chooses what is drawn should not set what an expensive computation scores; a setting has one home (`selection_metric`).

**Steps:**
1. In the best-runs figure cell, delete the `importance_metric` block (the `if`/`elif`/`else`
   at 1767–1776) and the cell's `return (importance_metric,)`; drop `METRICS` and
   `SELECTION_METRIC` from its parameters if nothing else in it reads them.
2. Replace every other read of `importance_metric` with `SELECTION_METRIC` (the inspection
   button's text, the `importance_metrics` cell, the two importance table cells, the two
   permutation figure cells) and fix each cell's parameters.
3. The `best_runs_metric` dropdown gets `options=[metric.name for metric in METRICS]` (add
   `METRICS` to its parameters); its comment loses "which the permutation importance also
   scores unless it is bias".
4. Section 5's markdown (lines 641–642) says the menu chooses the importance's metric: propose
   the new sentence to the user before changing it (the text is theirs).
5. Scratchpad scripts: `importance_metric` is no longer a notebook name; where Check C reads
   it, read `defs["SELECTION_METRIC"]`.

**Validate:** the runner passes; Check C equal to the reference at the defaults (the selection
metric is `rmse` in both). The test `test_the_plots_open_on_their_forms_and_the_selection_metric`
still passes; if it pins the menu's three options, update it to six.

**Done (2026-11-13, with S3):** as the steps say. Section 5's sentence (the user's choice): a
"Metric" menu chooses the metric of each plot (`selection_metric` by default); the permutation
importance always scores `selection_metric`. The best-runs figure cell now exports nothing (a
bare `return`: ruff's `PLR1711` 13 → 14, `03` 14 → 15 findings, the same marimo form as the
others). The test pins the six options. The manual check was not run (not a condition).
Scratchpad: `make_copy_tuning.py` restores that cell's return as `return (best_runs_figure,)`
(its old `importance_metric` would raise); `check_c_s34_tuning.py`, made by
`make_check_c_s34.py` from S2's (11 reads → `SELECTION_METRIC`; the menu's six options;
`importance_metric` absent; `build_importance_table` callable); Check B's `CHECK B S3 S4 OK`.
`run_S34.sh` (log `run_S34.log`): the suite 1411, mypy 28, the notebook tests 32, ruff `03` 15 /
`02` 11, the format hunks, `marimo check` clean, Check B on both, Check C on both equal to S2's
run with nothing skipped. Left: `probes/look_7_1e.py` and other probes may read
`importance_metric` (S5.0 updates them).

**Commit:**
```text
refactor(notebook): the permutation importance on the selection metric; the best runs' Metric menu only draws (S4)
```

### [ ] S5. One palette and one plotly template for every figure

- **Model:** **Fable.** Planned in plan mode first; the design below is the starting point,
  not final.
- **Evidence:** the palette tokens are written 7 times in `03` and 2 in `02`; the font
  dictionary 7 times; `template`, `paper_bgcolor`, `plot_bgcolor` 9 times. `forest_figure` and
  `ridge_figure` end with the same 30 lines (1602–1631 and 1702–1731).
- **What it breaks:** every figure's JSON, since the style moves into the template. Check C's
  figure references are regenerated. The figure functions then rely on a registered template,
  which loosens "code that may move to a package reads no notebook global".
- **Trade-off:** the largest saving in the figures (about 120 lines) for the widest diff. The
  plainer alternative is a `style_figure(figure)` function: the JSON still changes, less.
- **Recommendation:** accept, last among the notebook sub-tasks.
- **Decision:** accept, the full version (the user, 2026-11-12); S5.0's probe still chooses the template or the function.

**Steps (each one a stop the user validates):**
1. **S5.0, the probe (read-only):** on the 3-trial copy, build each figure as today and with
   the template; screenshot both (headless Chrome, as `probes/run_look_7_1e.sh`); show the
   pairs. Decide with the user: the template or the `style_figure` function.
2. **S5.1, `03`'s four figure functions** (`forest_figure`, `ridge_figure`,
   `permutation_figure`, `beeswarm_figure`): the palette cell, the shared style, the forest
   and ridge sharing their layout. The notebook tests of these functions are updated only
   where they pin a moved property.
3. **S5.2, `03`'s inline figures** (the waterfall, the two count figures, the shares bars).
4. **S5.3, `02`'s two figure cells.**

**Validate (each of S5.1–S5.3):** the runner passes with every frame equal; each changed
figure's traces (`data`) equal to the reference's and its effective layout equal after the
template is applied (`figure.full_figure_for_development` or the screenshots); screenshots
before and after shown to the user.

**Commit (one per stop, for example):**
```text
refactor(notebook): the figures' palette and style defined once (S5.1)
```

### [ ] S6. The EDA's first cell reads the best components once

- **Model:** Opus 5.5.
- **Evidence:** the independent cohorts cell builds every trial's components (`02:`1012–1024),
  then the best trial's again (`02:`1034–1043). The joint cell indexes the list it already
  has (`02:`1325).
- **Trade-off:** none found.
- **Recommendation:** accept.
- **Decision:** accept (the user, 2026-11-12).

**Steps:**
1. In `02`'s independent cohorts EDA cell, replace the second `tuple(CountComponent.from_params(
   _study.best_params, …) for cohort in COHORTS)` with
   `_best_components = _components_by_trial[_study.best_trial.number]`.

**Validate:** the runner passes; the EDA's frames and 27 figures equal to the reference.

**Commit:**
```text
refactor(notebook): the EDA's best components read from the trials' list (S6)
```

### [ ] S7. Coordinate descent leaves the EDA notebook

- **Model:** **Fable.** Planned in plan mode first.
- **Evidence:** D40 settled the comparison (the joint method). What remains for it:
  - the cell `02:`1061–1225 (165 lines) and the fixed classifier (`02:`878–892);
  - the `step` branches in `eda_best_df` (`02:`1367–1371) and in both figure cells
    (`02:`1464–1477, 1522–1536, 1617–1626), and two tables;
  - in `src`, `tune_total_component` and `tune_probability_component`
    (`cross_validation.py` 245–331), read only by that cell and two unit tests;
  - 15 of the cache's 50 study files.
- **Options:**
  - (a) keep everything;
  - (b) remove it from the EDA notebook and keep the two `src` methods as the valid path;
  - (c) remove both.
- **What (b) changes for the user:** the EDA shows two models, not three (`eda_best_df` 6
  rows; 18 figures, not 27). D40's comparison can no longer be re-run from the notebook; its
  numbers stay in [the cache plan](NOTEBOOK_CACHE_EVALUATION_PLAN.md)'s 5b.3. The 15 cache
  files stay, unread.
- **Trade-off of (b):** about 230 notebook lines and every `step` special case go; the two
  `src` methods are then read by their tests only.
- **Recommendation:** (b).
- **Decision:** (b) (the user, 2026-11-12).

**Steps under (b) (each one a stop):**
1. **S7.1, the notebook:** remove the coordinate descent cell, `MODEL2_FIXED_PROBABILITY_COMPONENT`
   and the `LogisticParams` import if nothing else reads it; the `step` branches (each
   `if`/`elif`/`raise` over the model label loses its coordinate descent arm and, where only
   one arm is left, the branch); the two tables; section 5's text where it describes the
   comparison (proposed to the user first). The label `total_times_probability` stays the
   joint method's.
2. **S7.2, the tests and the checks:** the EDA notebook test and the equality test where they
   read the removed names; `check_b_eda.py`, `check_c_eda.py` (its step blocks) and
   `make_copy_eda.py`; a new EDA reference pickle.
3. **S7.3, the docs:** the handoff's §1 and the guide where they say the EDA compares the two
   methods.

**Validate:** the runner passes; the EDA's remaining frames and 18 figures equal to the
reference's same-named ones; the tuning notebook's Check C equal with nothing excepted; the
cache's 50 files unchanged by a 3-trial run (it uses a copy).

**Commit (S7.1, for example):**
```text
refactor(notebook): coordinate descent removed from the objective EDA; the joint method stays (S7.1)
```

### [ ] S8. `from_params` filters the params in one place (`src`)

- **Model:** Opus 5.5.
- **Evidence:** both components select their own params with the same five lines
  (`components.py` 105–109 and 170–174). The probability copy also accepts `shared__`, which
  no study draws for it.
- **Trade-off:** a small `src` stop for five lines.
- **Recommendation:** accept, last.
- **Decision:** accept (the user, 2026-11-12).

**Steps:**
1. In `components.py`, add a public static method on `Component` beside
   `set_estimator_params`, for example `select_component_params(best_params, name)`, returning
   the dictionary both `from_params` build today (the entries under `<name>__` or `shared__`,
   the prefix removed). Both `from_params` call it. Behaviour unchanged, `shared__` included.
2. Add one unit test of the method in `tests/unit/test_hyperparameter_tuning_components.py`
   (own entries, shared entries, another component's entries left out).
3. If [the guide](HYPERPARAMETER_TUNING.md) quotes the lines, update it.

**Validate:** the unit tests of the package, the non-slow suite, mypy; the runner's Check C on
the tuning equal with nothing excepted. Mutation: drop `"shared"` from the method; the
existing shared-draw test must fail.

**Commit:**
```text
refactor(tuning): the components' from_params select their params through one method (S8)
```

## 5. The findings recommended to leave

**Decided 2026-11-12 (the user): all eight are left as they are.** L1 and L2 were asked one by
one (keep), L3–L7 together (leave all five), L8 together (leave closed).

| # | Finding | Why leave | Decision |
|---|---|---|---|
| L1 | `ridge_figure`: 100 lines, three "Plot" switches, three `if`/`elif`/`raise`, a test, the scipy import. The handoff's §5 probe found the density adds little (the replicates are near-normal). | The user asked for it on 2026-11-09 and the difference opens on it. S5 removes its repeated layout. Drop it only if it is no longer looked at. | |
| L2 | `aggregation.py`: `WeightedMean`, `UpperBound` and `corrected_std_error` are read by no notebook; pydantic dataclasses only here. | D9; isolated and tested; a valid path. Removing them saves about 90 lines and loses the documented option. | |
| L3 | The tune methods return `(study, seconds)` though the seconds are in `study.user_attrs`. | Returning the study alone fits "return the library's object", but six call sites would then read the attribute: not plainer. | |
| L4 | The two runs-table cells (1052–1084, 1330–1363) and the refit-and-score block (1015–1024, 1290–1299) are near copies. | One merged cell would make a model's table wait for the other's click. A shared function needs the settings passed in. Two explicit loops read well (D28). | |
| L5 | The best or worst k ranking is written twice (3012–3027, 3096–3109). | 15 lines; the two rank on different cohorts, by the user's choice. | |
| L6 | The per-model view loops (`_views += …` 7 times; the heading loop 3 times). | Three-line loops; a function would hide little. | |
| L7 | The EDA builds all 9 and 18 figures at each menu change and shows 3. | The two dictionaries are what Check C compares. | |
| L8 | The search spaces as classes built per call; the explicit run-settings lists; `03`'s 15 dropdowns; the two notebooks' duplicated setup. | Recorded decisions (D10–D11; 7.0's answer; 7.3's "each figure its own menus"; D41). Not reopened. | |

## 6. Considered, no finding

- **F4's two functions** (`find_feature_transformer_name`, `read_hyperparameters`): the name
  is found by identity because a component holds the transformer object (the user's rule: a
  record holds the object). Eight lines; a `feature_transformer_name` field would hold the
  same fact twice.
- **`build_trials_table`** takes each trial's components because coordinate descent's rows
  hold a fixed component that is not in the trial's params. It stays as it is under S7 (b):
  `shared__` params still need `from_params`.
- **`CrossValidationTuner`'s fields and public methods** (`score_fold`, `cross_validate`,
  `score_trial`, `storage_dir`, `recorded_metrics`): each is read by a tune method, the
  notebooks or Check C's in-memory studies.
