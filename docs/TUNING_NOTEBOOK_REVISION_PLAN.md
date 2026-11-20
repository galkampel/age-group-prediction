# Tuning notebook revision: SHAP colours, LightGBM search space, MLflow content, notebook split

## Context

Four notes on `notebooks/03_hyperparameter_tuning.py` (4,293 lines, branch `feat/mlflow-runs`):
the beeswarm is not in shap's blue-to-red; the LightGBM search space does not follow the
library's guidance; the notebook mixes tuning with model inspection; MLflow runs carry too many
params and metrics to compare runs in the chart view.

This is the general plan. Each task opens with a short design stop (a table to approve) that
tightens it; then small build sub-tasks, one approval and one commit each.

**Decided in this session (do not re-ask):**
- The inspection notebook reads the tuning runs from MLflow, rebuilds each model's best run
  from `components.json` + `best_params.json`, and refits only those 2 models. This replaces
  the earlier "no read-back cell" decision.
- SHAP: beeswarm in shap's blue-to-red, colours clipped at each feature's 5th–95th percentile,
  waterfall in shap's red/blue. Categories stay gray (as shap does). Plotly stays.

**Assumed (say so if wrong):** the test evaluation (section 6) moves to the inspection notebook
with the rest; the legacy search space (`modeling_config.py`, `configs/modeling.toml`,
`models/direct_cohort.py`) is not touched (old stack, due for deletion in roadmap §5).

## Order and why

`0 → A (SHAP) → B (LightGBM) → C (MLflow content) → D (split) → E (one full run)`

- A is small and independent; it goes first.
- B changes `src` only, so it does not care about the notebook's shape.
- C goes before D: the inspection notebook will read the MLflow runs, so it should be built
  against the final logged content, not the content about to be slimmed.
- B, C and D each invalidate the stored studies or the check store, so the long run (24 studies
  × 100 trials, four clicks) happens **once**, in E. Sub-tasks are checked with the existing
  3-trial runs and the unit/characterization tests.

## Status

2026-11-19: plan approved; Step 0 done except the merge (the user's). Next: A1 (Sonnet).

## Step 0 — clean start (done 2026-11-19)

- `03`'s uncommitted change (marimo's own re-save: trimmed returns, `@app.function`, an empty
  cell; no logic) restored from HEAD `932d43f` (md5 `5a3877b7…`), the user's choice.
- This plan copied to `docs/TUNING_NOTEBOOK_REVISION_PLAN.md`.
- Branch (the user's choice): `feat/mlflow-runs` is merged into main first, as it stands (the
  re-check of the final run on the old search space is skipped; Task E re-runs everything);
  tasks A–E go on a new branch from main, `feat/tuning-notebook-revision`. This file is
  committed on `feat/mlflow-runs` before the PR, so it reaches main with the merge.

## Task A — SHAP colours (Sonnet; one sub-task)

Files: `notebooks/03_hyperparameter_tuning.py` — `PALETTE` (l.1881–1891), `beeswarm_figure`
(l.2532–2624), the two captions (l.2647, l.3374); the waterfall reads `raises`/`lowers` only.

- **A1.** Add a `PALETTE["shap"]` colour scale sampled from shap's own colormap
  (`shap.plots.colors.red_blue`, ~8 stops written as hex literals, so no guess at the
  mid-tones); `beeswarm_figure` uses it instead of `palette["blues"]` (the counts scatter keeps
  the blues). Colour value = the feature clipped to its 5th–95th percentile, then min–max.
  `raises`/`lowers` → `#ff0051` / `#008bfb`. Captions and the two docs lines that say "blue
  ramp" updated.
  - Verify: the beeswarm test (`tests/characterization/test_hyperparameter_tuning_notebook.py`
    l.1276) still passes, plus one new assert for the colour scale and the clipping; a figure
    dump viewed by you.

## Task B — LightGBM search space

Files: `src/age_group_prediction/hyperparameter_tuning/search_spaces.py` (l.57–72, one method
shared by the regressor and the classifier), `params.py` (`LGBMParams`), `components.py`
(`make_estimator`, l.134–203), their unit tests, `docs/HYPERPARAMETER_TUNING.md` §4/§10.

Facts that drive the ranges: ~917 training rows, ~733 per CV fold, ~11 features; no early
stopping (ruled out: the composite's `fit` has no `eval_set`); TPE, 100 trials, 10 startup.

Proposed space (B1 confirms each row with measurements before any code):

| Hyperparameter | Now | Proposed | Reason |
|---|---|---|---|
| `max_depth` | 3–8 | **removed; fixed `-1`** | LightGBM grows leaf-wise; the docs tune `num_leaves`, not depth. Today 54–96% of the `num_leaves` range is dead under the depth cap. |
| `num_leaves` | 7–63 | **about 4–32, log** | Now the main complexity control. With ~733 rows, 63 leaves is ~12 rows a leaf; the upper half only overfits. B1 measures it. |
| `min_child_samples` | 5–40 | keep (maybe 5–50) | The second structure control; with `num_leaves` it bounds the tree. |
| `n_estimators` | 50–400 | **50–200** | Your note. |
| `learning_rate` | 0.01–0.2 log | **0.001–0.2 log** | Your note. One caution: with ≤200 trees and no early stopping, rates under ~0.01 cannot fit (200 trees at 0.001 move the model ~18% of the way). B1 measures how many trials land there; you decide whether to keep 0.001 or raise the upper bound. |
| `reg_alpha`, `reg_lambda` | 1e-8–10 log | **1e-6–10 log** | Your note; below 1e-6 is indistinguishable from 0. |
| `min_split_gain`, `subsample`, `colsample_bytree` | 0–1, 0.7–1, 0.7–1 | keep; B1 checks them | |
| `boosting_type` | default | **default (`gbdt`) only** | `dart` needs more trees, is slower and has no stable stopping point; `goss` is a speed device for large data (here ~900 rows); `rf` is a different model. A categorical here spends the 100-trial budget on a choice that rarely wins. |

- **B1 (Fable) — design stop.** Measure on the current data: leaves actually reached per tree,
  score against `num_leaves` and against `learning_rate × n_estimators` in the stored studies,
  the weighted classifier's `min_child_samples`. Output: the final table, and how the old
  journals in `studies/` are retired (they are matched by name only, so a new space would load
  stale results; `02_objective_eda.py` shares them). No code.
- **B2 (Opus).** `suggest_lgbm_hyperparams`, `LGBMParams` (drop `max_depth`), `make_estimator`
  (`max_depth=-1` fixed beside `subsample_freq`), the unit tests' names and fixtures
  (`test_hyperparameter_tuning_{search_spaces,params,components,builders}.py`, the notebook
  test's `total__max_depth` line).
  - Verify: the touched unit tests; a mutation per range.
- **B3 (Opus).** `docs/HYPERPARAMETER_TUNING.md` §4 table and §10 hazard; retire the old
  studies as B1 decided; re-record the 3-trial baselines (Check C: 8 + 16 runs).
  - Verify: Check C against the new baselines, run once.

**Note for you:** the check store's test scores (E11's one look) were produced with the old
space. A new space gives new best models, so the test set gets a second look. The change is
motivated by the library's docs, not by the test scores, but the handoff should record it.

## Task C — MLflow: fewer params and metrics

Files: `notebooks/03_hyperparameter_tuning.py` — `shared_params` (l.1021–1060),
`log_tuning_run` (l.1063–1168), `log_evaluation_run` (l.2729–2813), the two logging cells;
`docs/MLFLOW_RUNS.md`; the notebook tests at l.339, l.431, l.558; `check_store.py`.

Rule: **a param or metric is something you would put on a chart axis or filter by; everything
else is an artifact.** A value identical in every run cannot separate runs in a chart.

| Run kind | Now | Proposed |
|---|---|---|
| Tuning (24) params | ~28 + `best.*` | the run's settings (family, exposure, sharing), `best.*` (the scatter's x axis), and the few settings that differ between re-runs (`config_hash`, `seed`, `n_trials`, `objective_metric`, `selection_metric`, `split_method`). The other ~17 constants → one `settings.json` artifact. |
| Tuning metrics | 52 | 16: `cv/<objective>`, `{train,val}/composite/<6 metrics>`, `seconds`, `n_complete`, `n_failed`. Per-cohort scores → `scores.csv` in the run. |
| Best runs / final metrics | 108 each | 54: per model and for the difference, `<metric>/{score, ci95_lower, ci95_upper}`. Mean, median, sd stay in the logged table. |
| Best runs / final params | 23 (+1) | the same short list as above. |

**Run names.** A name is optional: without one MLflow generates a random one (like
`bold-cat-123`). The practice is to set a short readable name saying what distinguishes the
run, and to put everything you filter or group by in tags and params, never to parse the name.
Names need not be unique. The current names (`<model> <family> exposure=<bool>`, `best runs`,
`final`) already do this; the plan keeps them.

- **C1 (Fable) — design stop.** The exact kept lists, checked against what Task D's read needs
  (`val/composite/<selection>`, `components.json`, `best_params.json`, the study name) and
  against the chart view (a scatter of `best.shared__num_leaves` against
  `val/composite/rmse` across the LightGBM runs). No code.
- **C2 (Opus).** The tuning runs: `shared_params` split into params and `settings.json`,
  `scores.csv`, the tests at l.339 and l.431.
- **C3 (Opus).** The evaluation runs: `log_evaluation_run`'s metrics, the test at l.558.
- **C4 (Sonnet).** `docs/MLFLOW_RUNS.md` §4–§5 and `check_store.py`'s counts (copied into the
  repo's scratch area for the session; it lives in an old scratchpad).
  - Verify (C2–C4): the touched tests; a 3-trial run into a scratch store, then the UI's chart
    view opened by you with one scatter.

## Task D — split into tuning and inspection notebooks

Cut: after the Model 2 tables (l.1736). `03_hyperparameter_tuning.py` keeps sections 1–5
(setup, studies, refits, point scores, the 24 tuning runs). New `04_model_inspection.py` gets
best-run selection, the validation bootstrap, permutation importance, SHAP, all figures, the
test evaluation, and the two evaluation runs' logging.

- **D1 (Fable) — design stop + probe.** Read-only probe on a scratch store: select each model's
  best run with `mlflow.search_runs` (`stage=tuning`, order by `val/composite/<selection>`),
  rebuild with `from_params`, refit, and show the scores equal the tuning run's logged ones.
  Decide: what `04` repeats from `03` (data, split, feature variants, metrics — duplicated and
  pinned equal by a test, the D41 precedent from the `02`/`03` split); what `04` does when the
  store has fewer than 24 finished runs; whether the `model1_runs`/`model2_runs` tables are
  still logged by the best runs' run or dropped (the tuning runs now hold them); the `notebook`
  tag of the evaluation runs.
- **D2 (Opus).** `04`: setup + the MLflow read + selection + refit, up to `best_runs`. `03`
  unchanged yet. Verify: `best_runs` in `04` equals `03`'s on the same store.
- **D3 (Opus).** Move the validation inspection cells and the best runs' logging to `04`;
  delete them from `03`. Tests re-pointed (they find cells by file).
- **D4 (Opus).** Move the test evaluation and the final run's logging. `03` ends at l.~1736.
- **D5 (Sonnet).** Docs: `MLFLOW_RUNS.md` (which notebook logs what, §10's steps),
  `HYPERPARAMETER_TUNING.md`, the handoff; the three-notebook setup pin.
  - Verify (D3–D4): logged content of the two evaluation runs equal to C's by `run_key`
    (`checks/compare_stores.py`), wall times masked.

## Task E — the one full run (you click; Opus guides)

Fresh check store, new studies: `03` two clicks (24 runs), `04` two clicks (best runs, final).
`check_store.py` 0 failures; the new test scores recorded in the handoff. Then the PR.

## Standing rules for every sub-task

Plan mode per sub-task with a chat summary before approval; static checks (ruff,
`marimo check`, the touched tests) before any long run; each check run once; an independent
review before the stop; commit message in a copyable block, committed by you; no writes to the
repo's `mlflow.db`, `mlartifacts/` or `.mlflow-local/`; `find . -flags +dataless` before a run.

## Session prompts

One session per sub-task, in the plan's order: A1 → B1 → B2 → B3 → C1 → C2 → C3 → C4 → D1 → D2
→ D3 → D4 → D5 → E. At its stop a session prints the **next** sub-task's prompt from this
section, verbatim, in a copyable block, with the model's name above it. It does not write a new
prompt. A design stop (B1, C1, D1) may refine the prompts of its own task's build sub-tasks
here, as part of its output.

**Every prompt starts with this opening** (replace `<ID>`):

```text
Read docs/TUNING_NOTEBOOK_REVISION_PLAN.md: the Status line, the decisions at the top, sub-task <ID>'s task section, the standing rules and the session prompts. Plan sub-task <ID> in plan mode and do only <ID>. Before the first edit: confirm the branch is feat/tuning-notebook-revision, the working tree is clean, and no source file is offloaded (find . -flags +dataless, caches aside). At the stop: a file-by-file summary, the check results, the commit message in a copyable block, the plan's Status line updated, and the next sub-task's prompt printed verbatim from the plan with its model named.
```

**A1 (Sonnet)** — the opening, then:

```text
A1: the SHAP colours in notebooks/03_hyperparameter_tuning.py. PALETTE["shap"] is a plotly colour scale of about 8 stops sampled from shap.plots.colors.red_blue, written as hex literals; show me the sampled values and how you got them. beeswarm_figure uses it instead of palette["blues"] (the counts scatter keeps the blues); the colour value is the feature clipped to its 5th–95th percentile, then min–max; category levels stay gray, one colour bar. PALETTE["raises"] / ["lowers"] become #ff0051 / #008bfb. Update the two beeswarm captions and the docs lines that say "blue ramp". The beeswarm test still passes; add asserts for the colour scale and the clipping. Verify: ruff, marimo check, the touched tests (run once), and a figure dump of one validation beeswarm and one waterfall for me to look at.
```

**B1 (Fable; design stop, no code change)** — the opening, then:

```text
B1: decide the LightGBM search space with measurements. Start from Task B's table; my notes are fixed (max_depth not tuned and set to -1, n_estimators 50–200, learning_rate lower bound 0.001, reg_alpha / reg_lambda lower bound 1e-6, boosting_type the default) unless a measurement shows a problem, which you put to me as a question. Measure on the current data and the stored studies (read-only, scripts in the scratchpad): the leaves a tree actually reaches against num_leaves and min_child_samples at ~733 rows a fold; the score against num_leaves; the score against learning_rate × n_estimators, and how many trials would land where the model cannot fit with at most 200 trees; what min_child_samples means for the weighted classifier; whether min_split_gain, subsample and colsample_bytree earn their place in 100 trials. Decide how the old journals in studies/ are retired (matched by name only; 02_objective_eda.py shares them). Output: the final table with the evidence per row written into Task B, the open choices asked with the question tool, and B2's and B3's prompts refined below.
```

**B2 (Opus)** — the opening, then:

```text
B2: build the search space B1 decided: suggest_lgbm_hyperparams, LGBMParams without max_depth, make_estimator with max_depth=-1 fixed beside subsample_freq, the unit tests' names and fixtures, the notebook test's max_depth line. The legacy space (modeling_config.py, configs/modeling.toml, models/direct_cohort.py) is not touched. Verify: the touched unit tests, a mutation per range.
```

**B3 (Opus)** — the opening, then:

```text
B3: docs/HYPERPARAMETER_TUNING.md §4's table and §10's hazard; retire the old studies as B1 decided (ask me before deleting anything); re-record the 3-trial baselines (Check C, 8 + 16 runs), run once.
```

**C1 (Fable; design stop, no code change)** — the opening, then:

```text
C1: decide exactly which params and metrics each run kind keeps, by the rule in Task C (a chart axis or a filter; the rest an artifact). Check the lists against what the inspection notebook's read will need (val/composite/<selection>, components.json, best_params.json, the study name) and against the chart view: on a scratch store with 3-trial runs, show which scatter plots work across runs (a best.* hyperparameter against a validation metric) and whether the prefixed names (shared__, total__, probability__) compare across families. Output: the kept lists per run kind written into Task C, what moves to settings.json and scores.csv, open choices asked with the question tool, and C2–C4's prompts refined below.
```

**C2 (Opus)** — the opening, then: `C2: the tuning runs as C1 decided: shared_params split into params and settings.json, scores.csv, the tests that pin them.`

**C3 (Opus)** — the opening, then: `C3: the evaluation runs as C1 decided: log_evaluation_run's metrics and params, the test that pins them.`

**C4 (Sonnet)** — the opening, then: `C4: docs/MLFLOW_RUNS.md §4–§5 and check_store.py's counts; a 3-trial run into a scratch store, then guide me to open the chart view with one scatter.`

**D1 (Fable; design stop + read-only probe)** — the opening, then:

```text
D1: design the split into 03 (tuning) and a new 04_model_inspection.py. Probe on a scratch store: select each model's best run with mlflow.search_runs, rebuild it from components.json + best_params.json with from_params, refit, and show the scores equal the tuning run's logged ones. Decide: what 04 repeats from 03 and how the copies are pinned equal (the D41 precedent); what 04 does when the store has fewer than 24 finished runs; whether the model1_runs / model2_runs tables stay in the best runs' run; the evaluation runs' notebook tag; which tests move. Output: the decisions written into Task D, open choices asked with the question tool, and D2–D5's prompts refined below.
```

**D2, D3, D4 (Opus), D5 (Sonnet)** — the opening, then the sub-task's lines from Task D as D1
left them.

**E (Opus guides, the user clicks)** — the opening, then: `E: guide me through the full run on a fresh check store (03 two clicks, 04 two clicks), one step at a time, and remind me to note each message before closing marimo; then check_store.py, the test scores into the handoff, and the PR's title and description.`

## Verification (end to end)

1. `pytest tests/unit -k hyperparameter_tuning` and
   `tests/characterization/test_hyperparameter_tuning_notebook.py` green after each sub-task.
2. Check C (3-trial runs) against the baselines re-recorded in B3.
3. Task E's full run: 26 FINISHED runs, `check_store.py` 0 failures, the chart view showing a
   hyperparameter against a validation metric across runs, the beeswarm blue-to-red.
