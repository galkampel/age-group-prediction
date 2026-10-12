# Validation plan: PR #12 (any regressor in `DirectCohortModel`; the feature transformer inside each model)

> **2026-10-07:** the scripts and tests of `docs/validation/pr12/` were removed (the code they
> checked is renamed and later rebuilt; the behaviors of ours they pinned are in `tests/unit/`, except that `estimator` has
> no default, which the signature enforces).
> They are kept in commit `c966edb`: `git show c966edb:docs/validation/pr12/<file>`.

**For:** an independent validating session with no memory of the implementation. **Written:** 2026-10-05.
**Under validation:** branch `feat/estimator-and-feature-transformer` (PR #12, draft, into
`feat/hyperparameter-tuning`), which implemented
[DIRECT_COHORT_GENERALIZATION_PLAN.md](DIRECT_COHORT_GENERALIZATION_PLAN.md) Steps 0–7.
**Output:** `docs/DIRECT_COHORT_GENERALIZATION_VALIDATION_REPORT.md`, with a verdict, and the
scripts and tests you wrote to check the results. No fixes to the code under validation.

## 1. Purpose and scope

Decide whether PR #12 can be merged: the decisions are implemented as stated, the math holds,
the tests catch the mistakes they claim, the docs are true and runnable, and the PR body matches
the diff.

**Out of scope:** note any of these if seen, but do not count them as findings.
- The known follow-up, deferred by the user to a separate PR: in `TotalChildrenModel` and
  `CohortProbabilityModel`, a refit whose final `validate_data(reset=True)` raises leaves the new
  `coef_` with the old `feature_names_in_`.
- The old stack `src/age_group_prediction/models/`, and `DIRECT_COHORT_MODEL.md` §1–§10, which
  describe it.
- PR #11 text in `total_children.py` and `cohort_probability.py` outside their
  `feature_transformer` lines.
- The `subsample_freq` note being in the docs only, not in a docstring. The user decided this:
  scikit-learn's meta-estimators leave inner-library quirks to the library.
- The settled decisions in [the handoff](DIRECT_COHORT_GENERALIZATION_HANDOFF.md) §2a. Check
  that they are implemented; do not reopen them.
- Historical records in `MULTI_COHORT_MODELS_PLAN.md` and `MODEL_REIMPLEMENTATION_PLAN.md`,
  which keep the old API on purpose under dated notes.

## 2. Rules

- **Check and test, do not fix.** Do not change the code, tests or docs under validation.
  You write two things:
  - the report;
  - your own **probe scripts and test files**, in `docs/validation/pr12/`: `probe_*.py` scripts
    and `test_*.py` pytest files, each with a docstring naming the claim it checks.

  No `git stash`, `checkout`, `reset`, `restore`, `commit` or `push`. The user commits.
- **Test the results, do not only read them.** Every claim you mark "pass" is backed by a probe
  or a test you ran: each V1 decision, each V2 step that can be checked numerically, each V4 doc
  claim. Prefer a pytest test when the claim is a lasting property, for example "the weighted
  rate equals the `init_score` offset". Use a probe script for a one-off measurement. Run tests
  with the repo's settings from the repo root:
  `uv run pytest -q -c pyproject.toml --rootdir . -W error docs/validation/pr12/`. The repo has no
  `conftest.py`, so nothing is lost outside `tests/`, and `pythonpath=["src"]` still applies.
  Run probes with `PYTHONPATH=src uv run --group test python docs/validation/pr12/probe_<name>.py`.
  Your tests must pass against the PR as it is, unless they demonstrate a finding: then mark
  them `@pytest.mark.xfail(strict=True, reason="finding <n>")`, so the suite records the defect.
- **Plan first.** Read §3 below and the files in V0, post a short summary of what you will run,
  and wait for the user's approval before running V1–V7.
- **Mutation checks edit the repo's `src` in place.** Copy the file to the scratchpad, mutate,
  run the tests, copy back, compare md5 with the original, and confirm `git status` is clean.
  `pyproject.toml` sets pytest's `pythonpath = ["src"]`, so a mutated copy on `PYTHONPATH` is
  silently ignored. Never restore with git.
- **Probes:** `PYTHONPATH=src uv run --group test python -c "..."`. Build LightGBM with
  `n_jobs=1` (more OpenMP threads crash alongside torch on macOS) and `verbosity=-1`; filter its
  log lines with `grep -v "^\[LightGBM\]"`. Quote globs in zsh.
- **Every finding is reproduced:** give the exact command and its output. A claim you could not
  check is reported as "not verified", never as passed.
- **Report what you saw**, including passes, with the numbers.

## 3. Phases

### V0 Orientation
- Read the plan in full: §1 context, §2 decisions, §2b derivation, §3 how to work (rule 7),
  §4 target shape, §5 steps with their recorded results, §6 verification, §7 PR body. Then read
  the handoff.
- `git log --oneline origin/feat/hyperparameter-tuning..HEAD`, and
  `git diff --stat origin/feat/hyperparameter-tuning...HEAD` (expect 29 files, all under
  `modeling/`, `hyperparameter_tuning/`, their tests and `docs/`, plus this file).
- Baseline: `uv run pytest -m "not slow"` gives **1230 passed, 1 skipped, 1 xfailed** (1221
  before the PR).

### V1 Decisions, each against the code with a probe
| # | Check |
|---|---|
| G1, G1a | `DirectCohortModel(*, estimator, use_exposure=False, feature_transformer=None)`. `get_params(deep=True)` lists `estimator__n_estimators`; `clone(model).set_params(estimator__n_estimators=5)` leaves the template untouched; no LightGBM hyperparameter or `objective` remains on the model |
| G1b | `Regressor` (a `Protocol`, exported from `modeling`) states `fit(X, y, sample_weight=None)` and `predict(X)`; `uv run mypy` is clean |
| G2 | `estimator` has no default; `sklearn.utils.estimator_checks.check_parameters_default_constructible` passes (the contract test runs it) |
| G3 | (a) Weighted rate vs LightGBM's own offset: fit `LGBMRegressor(objective="poisson", n_jobs=1)` through the model with `use_exposure=True`, and by hand with `init_score=log(E) + log(Σy/ΣE)`, then predict `exp(raw + log E + b)`. The plan records 1.5e-8 relative on 2000 rows and 50 trees. (b) `PoissonRegressor(alpha=0)` through the model equals statsmodels' `GLM(y, X, family=Poisson(), offset=log(E))` (statsmodels 0.14 is installed). (c) With `alpha > 0`, the weighted-rate coefficients match the offset GLM at `alpha × mean(E)` (the plan measured 1e-4 at `alpha=1`). (d) `HistGradientBoostingRegressor(loss="poisson")` fits and predicts with the exposure |
| G4 | `use_exposure=True` with a Gaussian loss: the weighted-rate fit of `LinearRegression` (or `LGBMRegressor(objective="regression")`) equals weighted least squares of `y` on `[E, E·x]` with weights `1/E` |
| G5 | A regressor whose `fit(X, y)` has no `sample_weight` raises its own `TypeError` at `fit` with `use_exposure=True`; there is no pre-check |
| G6 | Fitted state `estimator_`, `use_exposure_`, `feature_transformer_`; after `set_params(use_exposure=False)`, `predict` still requires and applies the exposure |
| F1 | `feature_transformer` is in the `__init__` of `DirectCohortModel`, `TotalChildrenModel` and `CohortProbabilityModel`, not of the composites; `BaseAgeGroupModel._fit_features` returns a fitted **copy** (the template stays unfitted) and `_transform_features` never refits |
| F2 | `grep -rn ModelPipeline src tests` is empty; `modeling.pipeline` does not import |
| F3 | `CVHyperparameterEvaluator` has no `feature_transformer` field and has `build_model(params)`. End to end: put a recording `FeatureTransformer` subclass inside a `DirectCohortModel` inside the evaluator, run `evaluate(FixedTrial(...), ...)`, and confirm each fit saw exactly that fold's training rows and each transform at predict used that fold's fit |
| I1, I2 | `IndependentCohortModels.predict` follows `y`'s column order at fit whatever the mapping's order; `cohort_models` stays the unfitted template and `cohort_models_` holds the fitted copies |

### V2 Math
- Re-derive plan §2b independently:
  - (A)–(C): the two Poisson objectives differ by a constant in $F$; the per-row gradient
    $\mu_b - y_b$ and Hessian $\mu_b$ are equal; the weighted `boost_from_average` start
    $\log(\sum y / \sum E)$.
  - (D1)–(D4): the Gaussian model with variance $\sigma^2 E_b$; the weighted-rate loss is
    $\tfrac12 \sum (y_b - E_b f_b)^2 / E_b$; the Hessian is $E_b$.
  - (E): why a residual is not used.
- Report any step that does not follow.
- Confirm `DIRECT_COHORT_MODEL.md` §0.1 contains plan §2b **word for word**: from "**Setup.**" up
  to the end of (E).

### V3 Tests
- Read the new and changed tests:
  - `tests/unit/test_modeling_base.py`, `test_modeling_direct_cohort.py` and
    `test_modeling_feature_transformer.py`;
  - `test_modeling_contract.py`, `test_modeling_independent_cohorts.py`,
    `test_modeling_independent_total_probability.py` and `test_modeling_calibration.py`;
  - `test_hyperparameter_tuning_evaluator.py`.

  Each test's name and comment must state a real mistake in our code, not only library
  behaviour. Report any test that tests nothing, or duplicates another.
- Re-run at least one recorded mutation per step from plan §5's "Mutation checks" and "Result"
  lines (Steps 1–5), and confirm the named tests fail.
- Then hunt for gaps with new mutations, one at a time. Report each one that no test catches
  (a survivor), and whether it would fail loudly anyway. For a silent survivor, write a test in
  `docs/validation/pr12/` that catches it, and show it failing under the mutation and passing
  without it. The user decides whether to move it into `tests/`.
  - `predict` divides by the exposure, or multiplies the target at `fit`;
  - `sample_weight=y` instead of the exposure;
  - `_transform_features` checks `self.feature_transformer` instead of `self.feature_transformer_`;
  - `_fit_features` fits on `X` but returns the untransformed `X`;
  - `CohortProbabilityModel.predict` transforms as well as `predict_logits`;
  - the evaluator predicts on `X_train`, or fits on all rows;
  - `build_model` returns `self.model` (no clone);
  - `IndependentCohortModels` iterates the mapping instead of `y.columns`.

### V4 Docs
- **Run every python block**, in order in one namespace under `-W error`:
  - `FEATURE_TRANSFORMATIONS.md` §8.0–§8.3 (the section "## 8. Building Each Model's
    Transformer" up to "### 8.4");
  - `DIRECT_COHORT_MODEL.md` §0 (up to "## 1. Statistical Model");
  - `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0 (up to "## 1. ").

  The namespace starts with `pd`, `np` and
  `raw_table = StudentPopulationSimulator(load_simulation_config("configs/simulation.toml")).run(np.random.default_rng(0))`
  (`from student_simulator.pipeline import StudentPopulationSimulator`,
  `from student_simulator.config import load_simulation_config`). After the first block that
  defines `table`, add `train_index, test_index = Splitter("grouped").train_test_indices(table, table["neighborhood_id"], test_size=0.2, random_state=0)`
  and `fit_df = train_df = table.iloc[train_index]`, `valid_df = test_df = table.iloc[test_index]`.
- **The tuning plan's §5 block** (`HYPERPARAMETER_TUNING_PLAN.md`): `HyperparameterStudy` does
  not exist yet (Phase 3), so define a stand-in. It takes `seed` and `n_trials`, and its
  `optimize(objective)` runs a seeded `optuna` TPE study and returns it, so `.best_params` works.
  Run the block on a simulated table with `n_apartments`, a `groups` array, `cohort_columns`,
  `seeds`, `models = {}` and a `tree`.
- **The plan's §6 end-to-end probe.**
- **Claims:** every claim in the edited sections matches the code. These are `DIRECT_COHORT_MODEL.md`
  §0, `INDEPENDENT_TOTAL_PROBABILITY_MODEL.md` §0, `FEATURE_TRANSFORMATIONS.md` §3.3 and §8, and
  the `modeling` and `hyperparameter_tuning` rows of `MODULE_REFERENCE.md`. Among them:
  - DIRECT §0.3's claim that LightGBM fits a zero, NaN or infinite exposure silently while
    `HistGradientBoostingRegressor` raises;
  - the §0.6 rule that `cohort_models__…__estimator__…` raises `AttributeError`.
- **Every markdown anchor** in those files resolves (GitHub slug rules).
- **No old API in live docs:** `ModelPipeline`, `objective=` as a model argument, `init_score` as
  the mechanism, `base_log_rate_`, `regressor_`, `build_feature_transformer_and_model`, `model__`
  nested names, "regression refuses the exposure". Live docs means outside the historical records
  of §1.
- `MODULE_REFERENCE.md`'s `__init__.py` row matches `age_group_prediction.modeling.__all__`.

### V5 Rule 7
The plan's §3 rule 7: comments and docstrings keep only what the code does and the non-obvious
constraint, that is, what would pass silently and what the caller must do. Derivations, examples
and motivation belong in the docs. Apply it to every comment and docstring that
`git diff origin/feat/hyperparameter-tuning...HEAD -- src` adds or changes. Report leftovers
with file:line, and anything trimmed that was an essential silent-failure constraint and is now
in no docstring.

### V6 Routine
- `uv run ruff check` and `uv run ruff format --check` on `src/age_group_prediction/modeling`,
  `src/age_group_prediction/hyperparameter_tuning` and the changed tests;
- `uv run mypy`, and `uv run mypy src/age_group_prediction/modeling src/age_group_prediction/hyperparameter_tuning <changed tests>`;
- `uv run pytest -q -W error 'tests/unit/test_modeling_'*.py tests/unit/test_hyperparameter_tuning_evaluator.py`;
- `uv run pytest -m "not slow"`;
- the slow `uv run pytest tests/validation/test_total_children.py` (statsmodels oracles).

### V7 PR body
The body in the plan's §7 (the user pastes it into PR #12) matches the diff:
- the "What changes" list;
- "Removed or renamed" and the migration lines;
- the follow-up;
- the Steps checklist;
- the counts (1221 → 1230).

Check the live PR with `gh pr view 12`. If it differs from §7, report the difference; do not
edit the PR.

## 4. The report

Write `docs/DIRECT_COHORT_GENERALIZATION_VALIDATION_REPORT.md`:
1. **Verdict:** ACCEPT, ACCEPT WITH NOTES (Low findings only), or REJECT (any High or Medium
   finding), with one paragraph of reasons.
2. **Phase table:** V0–V7, each with pass/fail and its evidence (commands, numbers).
3. **Findings, ranked High / Medium / Low.** Each has file:line, what is wrong, a reproduction
   (command and output), and a suggested fix. Fixes are **not applied**.
4. **Mutation results:** every mutation run, the tests that failed (ours and yours), and the
   survivors. For each survivor, say whether one of your tests now catches it.
5. **Your scripts and tests:** each file in `docs/validation/pr12/`, the claim it checks, and its
   result. Give the command that re-runs them all.
6. **Not verified:** anything you could not check, and why.

Then stop, and give the user a short summary in chat: the verdict, the counts, and the top
findings.
