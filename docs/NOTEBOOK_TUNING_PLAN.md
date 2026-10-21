# Re-implement the hyperparameter tuning inside the notebook

**Status (2026-10-21):** approved; T1.1 committed (`0416267`); T1.2 committed (`bee781a`) after the user's review (estimator dataclasses hold hyperparameters only; LightGBM's loss is a family); T1.3 committed (`7ae6314`) after three user reviews (components are a study's fixed settings, a `Literal` family and `use_exposure`; the family → transformer-names mapping is an argument of `suggest_feature_transformer`; no dataclass builds a model; the exposure a study setting, both arms; feature variants renamed feature transformers; D2, D3, D6, T1.4–T2.2 rewritten); T1.4 done 2026-10-19, then reworked at two rounds of the user's review on 2026-10-20 (D8: a named component per part of the composite under an abstract `Component`; the family checked from its `Literal`; each subclass holds its families' transformer names; `make_estimator(hyperparameters, *, seed)`; the T1.3 cells changed too; T1.4–T2.2 rewritten), committed (`8cc231d`); T1.5 done 2026-10-21 (`cv_score`, `tune_model`; at the user's review the fold variables named and an aggregator added, D9: `src/age_group_prediction/aggregation.py` and its tests, default `Mean`), committed (`a5ae06b`); T1.6 done 2026-10-21 (Model 1's button and run cell; Check C 14.7 s at 3 trials), then the redesign of D10 at its review, revised by D11 and D12 at the next reviews (search spaces with explicit suggest methods, the builder assembles components, the transformer by name; a study is a family combination and exposure; `feature_transformer_dict`; Check C reproduces T1.6's best values), uncommitted; notebook md5 `4d60d6d2dd3cdc07d1150b0c587420cf`; next T1.7. The implementing session updates this line and ticks the sub-tasks.

Written for the model that implements it, one sub-task per approval. Every sub-task has the
code to write, a "done when" list and the commands that verify it. The user reads the diff in
the IDE, runs the checks and commits; the implementer never commits.

## 0. Context and decisions

Section 5 of `notebooks/02_experiments.py` (lines 598–1222 at HEAD `8ee091a`, md5
`55f81ffb7e6dd3f85aed71f1a45118ce`) tunes Model 1 (`IndependentCohortModels`) and Model 2
(`TotalTimesProbabilityModel`) through the package `src/age_group_prediction/hyperparameter_tuning`
(`Parameter` classes, `KeyedParameter`, `CVHyperparameterEvaluator` driving `set_params` paths
such as `cohort_models__<cohort>__…`, `HyperparameterStudy`, `Aggregator`). The user rejected
that package: too much machinery between the notebook and Optuna, and the composites are
assembled by `set_params` strings instead of being built explicitly.

**Decisions (2026-10-18, the user's; do not re-ask):**

- D1. Section 5 is rewritten with plain Optuna and a hand-written cross-validation loop. None of
  the package's classes or functions is imported. The package is deleted afterwards (T4).
- D2. **Frozen dataclasses** (`dataclasses.dataclass(frozen=True)`), JSON-serializable through
  `dataclasses.asdict`, for two things: an estimator's searched hyperparameters
  (`LGBMRegressorParams`, …), and a component's settings fixed for one study
  (`CountComponent(family, use_exposure)`, `ProbabilityComponent(family)`). Every field is a
  JSON scalar; a feature transformer is stored as its key in `FEATURE_TRANSFORMERS`, and the
  family is a `Literal` (`CountFamily`, `ProbabilityFamily`). A study's record is
  `{"settings": asdict(component), "params": study.best_params}`; `best_params` holds the
  chosen transformer's name. (The comparison with
  TypedDict: §6.) Since D8 a component also holds its `name`, and Model 1's
  record holds the list of its components' settings.
- D3. An estimator dataclass has a classmethod `suggest(trial, prefix, …)` calling
  `trial.suggest_*` inline. A component looks up its params class from its family, and
  samples with `suggest_params` / `suggest_feature_transformer(trial, prefix,
  family_transformers)` (and
  `suggest_calibration_method`). **No dataclass builds a model** (the user, 2026-10-18, at
  the reviews of T1.2 and T1.3). The objective builds it: `suggest_model1` / `suggest_model2`
  sample from the trial and call the constructors explicitly (estimator, then
  `CountModel` / `CohortProbabilityModel`, then the composite; no `set_params`). The best run's
  refit calls the same function on `FixedTrial(study.best_params)`. (Since D8: a component
  builds its estimator, never the model; the methods take no prefix, the component's name
  gives it; a subclass holds its families' transformer names, and the family → names
  argument is gone.)
  **Package-ready (the user, 2026-10-18, at T1.3's second and third reviews):** a dataclass
  reads no notebook global. A setting fixed for the study (the family, the exposure) is a
  field. The family's candidate feature transformers come in as an argument of
  `suggest_feature_transformer`: a mapping from each family to the names that suit it
  (`COUNT_FAMILY_TRANSFORMERS`, `PROBABILITY_FAMILY_TRANSFORMERS`), and the component reads
  its own entry. The reason is the user's: one place pairs a family with its transformers, so
  a component cannot be given another family's (a field of names, the second review's
  version, could). Whether a transformer suits a family is the experiment's knowledge, which
  no code can check; the mapping is where it is reviewed. The builder resolves a name to its
  object (`feature_transformers[name]`). An unknown family raises at construction, a family
  missing from the mapping when sampling, and `suggest_params`'s `else` raises. The families are
plain `Literal` assignments, not the `type` statement: cloudpickle cannot pickle a class
  annotated with a `type` alias defined in a cell (measured), and the joblib workers receive
  the components by cloudpickle. In a package module either form pickles.
- D4. The metric is built **in the notebook**: `Metric("cohort_poisson_deviance",
  cohort_poisson_deviance, direction="minimize")`; `optuna.create_study(direction=metric.direction)`;
  the objective returns the metric's value unchanged. The module-level constants leave
  `scoring.py` (T3); the `Metric` class and the two functions stay there.
- D5. Pruning and parallelization beyond today's parallel studies are **not decided now**; a
  later performance pass (§7 holds the facts for it).
- D6. Still in force from 2026-10-15: one joint study per family on the composite; a trial
  chooses per component its feature transformer; `n_trials = 200`; studies of
  one model in parallel with joblib, study i seeded `seed + i`; only the best run is refit;
  Model 2's totals come from the edited list `MODEL2_TOTALS`; `split_method = "random"`.
  **Changed 2026-10-18 (the user, at T1.2's review):** LightGBM's loss is no longer a sampled
  categorical. `lgbm_poisson` and `lgbm_gaussian` are separate families, and
  `LGBMRegressorParams.objective` is fixed by the family. Why: TPE here is univariate (D7),
  so it cannot learn that good hyperparameters differ per loss; a 30-trial study that sampled
  the loss split 10/20 and did worse than either loss alone; and the family table compares
  the losses directly. The cost: one run cannot mix the losses across cohorts, and there is
  one more study per exposure setting.
  **Changed 2026-10-18 (the user, at T1.3's review):** the exposure is no longer a sampled
  categorical either. It is a study setting (`CountComponent.use_exposure`), and every count
  family runs with it on and with it off, as the older plan's E5 argued (the exposure changes
  the target, a count or a weighted rate, so one TPE budget would under-tune the losing arm).
  Measured on validation (one fit per cohort, fixed hyperparameters): the GLMs lose 30–50%
  with the exposure off, because their variants hold no `n_apartments`; LightGBM loses about
  3%. The user kept both arms for every family. Model 1 runs 8 studies; Model 2 runs
  `MODEL2_TOTALS` × exposure × `PROBABILITY_FAMILIES` (16 at four totals).
- D7. Settled by probe in this session (Optuna 4.9.0, joblib 1.5.3): the trial value is the
  plain mean of the fold scores (KFold on 917 rows: folds of 184/183 rows; since D9, the
  default of an aggregator); the sampler is
  `TPESampler(seed=study_seed)` (the `multivariate=True` option raises `ExperimentalWarning`
  under `-W error`); the best run is rebuilt with `suggest(optuna.trial.FixedTrial(best_params))`
  (exact for `None`/`bool`/`str`, raises `ValueError` on a missing name or a foreign value);
  `study.optimize(..., catch=(RuntimeError,))` keeps a study alive when an NB2 fit does not
  converge (without it the study stops and `Parallel` discards the sibling studies; returning
  `inf` would hide the traceback); an in-memory `optuna.Study` pickles back whole from a loky
  worker (`trials`, `best_trial.distributions`, `best_params` all present).
- D8. **Changed 2026-10-20 (the user, at two rounds of T1.4's review; supersedes the parts of
  D2, D3 it names):**
  - **A component is a named part of the composite.** It holds `name`, so
    `CountComponent(name, family, use_exposure)` and `ProbabilityComponent(name, family)`.
    Model 1 has one component per cohort, named after it, because each cohort is a separate
    prediction problem; a study is the tuple of its three components. Model 2 has `"total"`
    and `"probability"`.
  - **The name replaces the prefix argument.** A trial's names are `<name>__<parameter>`,
    and the component adds the separator (`prefix` property). The separator stays `__`
    because cohort names hold `_` and the tables split on the first `__`. The trial names
    are as before.
  - **An abstract `Component(abc.ABC)` base.** It holds `name`, `family`, the checks of
    `__post_init__` and the shared `suggest_feature_transformer`; `suggest_params` and
    `make_estimator` are abstract.
    - The family is checked against the subclass's `Literal` annotation:
      `get_args(get_type_hints(type(self))["family"])`. This needs no list of families
      beside the `Literal`, and it also resolves string annotations
      (`from __future__ import annotations`) in a package module (measured).
    - No `type` field: the first round had one; the second removed it, since no code read
      it and the subclass states the kind.
  - **Each subclass holds its families' candidate feature transformers.** The ClassVar
    `FEATURE_TRANSFORMER_NAMES` maps family → names, as a `MappingProxyType`.
    - It is fixed on the class, so no component can carry another family's: the guarantee of
      T1.3's third review, without the mapping argument.
    - A family without an entry raises at construction.
    - The trial samples a name, because Optuna's categorical choices must be primitives:
      objects raise a `UserWarning`, an error under `-W error` (measured).
    - `suggest_feature_transformer(trial, feature_transformers)` returns the object
      `feature_transformers[name]`.
    - Read-only, because ruff's RUF012 flags a mutable class attribute (in a cell it cannot
      see the `ClassVar` parameter). It pickles through cloudpickle and runs under
      `Parallel` (measured).
  - **Sampling and construction are separate steps, both on the component.**
    `suggest_params(trial)` returns the params record. `make_estimator(hyperparameters, *,
    seed)` builds the unfitted estimator from a mapping of the estimator's keywords (a dict
    works; the builder passes `asdict(params)`): an `if`/`elif` on the family with a raising
    `else`.
    - `make_`: scikit-learn's word for its factories (`make_pipeline`,
      `BaseEnsemble._make_estimator`).
    - Tracking does not depend on where the estimator is built: Optuna records every
      sampled value by name.
    - D3's "no dataclass builds a model" becomes: a component builds its estimator; the
      `CountModel` / `CohortProbabilityModel` and the composite are built in
      `suggest_model1` / `suggest_model2`.

- D9. **Added 2026-10-21 (the user, at T1.5's review):** the fold scores are combined by an
  aggregator, the user's `Aggregator` classes (`Mean`, `WeightedMean`, `UpperBound`).
  - **Where the classes live:** they moved out of the package that D1 keeps out of the
    notebook into `src/age_group_prediction/aggregation.py`, a copy unchanged but for the
    `_config` import and one docstring word. Their tests went to
    `tests/unit/test_aggregation.py`, and mypy's strict list gained the module. T4 deletes
    the package's copies.
  - **The default:** `cv_score(..., aggregator=None)` and `tune_model(..., aggregator=None)`,
    where None means `Mean()`, resolved inside `cv_score`. This is scikit-learn's pattern
    (`cv=None`, `scoring=None`). A `Mean()` default in the signature trips ruff's B008, and
    a missing required argument needs no hand-written raise: Python raises `TypeError`.
  - **Why the aggregator is an argument:** it decides the selection. Measured on two Model 1
    models, the weighted mean differs from the plain mean by 0.0006, while the median flips
    which family wins (by mean 2.262 vs 2.328, by median 2.313 vs 2.228).
  - **Measured at seed 42 (2-trial `poisson_glm` study):**

    | Aggregator | Best value |
    |---|---|
    | default | 2.48456 |
    | `Mean()` | 2.48456 |
    | `WeightedMean()` | 2.48390 |
    | `UpperBound()` | 2.70768 |
  - `cv_score` names the fold variables (`X_train`, `Y_train`, `exposure_train`, `X_val`, …,
    `Y_val_pred`); the user's edit.

- D10. **Changed 2026-10-21 (the user, at T1.6's review; supersedes D2's "a component holds
  a study's fixed settings", D3's `suggest` on the params classes, and D8's sampling methods on
  the component):** a component is a complete record of one part of a run, and every Optuna
  call lives in a search space.
  - **Records, no Optuna, `kw_only=True`:**
    - The params dataclasses `LGBMParams`, `PoissonGLMParams`, `NB2Params` and
      `LogisticParams` hold fields only.
    - `Component(abc.ABC)` holds `name`, `family`, `feature_transformer` (a name in
      `FEATURE_TRANSFORMERS`) and `hyperparameters` (the family's params class, checked by
      a `PARAMS` ClassVar).
    - `CountComponent` adds `use_exposure`; `ProbabilityComponent` adds
      `calibration_method`.
    - `make_estimator(*, seed)` is an if/elif on the family with a raising `else`.
    - A component is built the same way by hand or from a trial, e.g.
      `CountComponent(name="n_elementary", family="nb2", use_exposure=True,
      feature_transformer="glm_base", hyperparameters=NB2Params(dispersion=0.15,
      alpha=0.01))`.
  - **Search spaces, every `trial.suggest_*`:**
    - `SearchSpace(abc.ABC)` holds `name` and `family`, plus the per-subclass
      `FEATURE_TRANSFORMER_NAMES` ClassVar and the shared feature-transformer and LightGBM
      sampling.
    - `CountSearchSpace` adds `use_exposure`; `ProbabilitySearchSpace` adds nothing.
    - `suggest(trial)` returns a complete component. The draws keep their old order, so the
      seeded studies are unchanged: T1.6's Check C reproduced to the fourth decimal on all
      8 studies.
  - **Builders without Optuna:** `make_count_model`, `build_model1(components, …)` and
    `build_model2(total, probability, …)`. `suggest_model1(trial, spaces, …)` and
    `suggest_model2(trial, total_space, probability_space, …)` sample, then build.
  - **The best run:** the best components are `space.suggest(FixedTrial(best_params))` per
    space, and `asdict(component)` is the saved record, its hyperparameters nested.
  - **LightGBM's objective** is set by `make_estimator` from the family: one `LGBMParams` for
    the regressor and the classifier, so the record states the loss once.
  - **Why, the user's words:** a component should hold everything needed to create a run, so
    the best one can be saved and set by hand; Optuna's suggest is kept apart. Best practice
    agrees: a configuration separate from its search space (scikit-learn's estimator params
    vs `param_distributions`; Optuna's objective building a plain config). A component that
    sampled its own fields would be half-built until a trial filled it.

- D11. **Changed 2026-10-21 (the user, at D10's review; revises D10):**
  - **Search spaces have explicit methods and build nothing.** A search space holds only
    `name` and `family` (no exposure). Given the family, it offers:
    - `suggest_hyperparameters(trial)`, returning the family's params record;
    - `suggest_feature_transformer_name(trial)`;
    - `suggest_calibration_method(trial)`, probability only.
  - **The builder assembles the components.** `suggest_count_component(trial, space, *,
    use_exposure)` and `suggest_probability_component(trial, space)` call those methods and
    feed the draws into the component.
    - The exposure is the study's setting: a Model 1 study is `(spaces, use_exposure)`.
    - Keyword arguments evaluate left to right, which keeps the old order of the draws:
      Check C reproduced T1.6's 8 best values exactly.
  - **The component holds `feature_transformer_name`, resolved by the builder** from
    `FEATURE_TRANSFORMERS`. That is best practice: a record refers to a registered object by
    name (scikit-learn's `scoring="…"` strings, config systems). It also fits the
    measurements: a `FeatureTransformer` hashes by identity, is not equal to its clone, and
    is not JSON. So `json.dumps(asdict(component))` is the saved best component.
  - **No `PARAMS` check, no `object` field.** Each subclass annotates its own
    `hyperparameters` union.
  - **The seed is an argument of `make_estimator`:** the run decides it, and a stability
    check can refit one component under several seeds.
  - **The params classes stay separate records** (what to build; the search space says how
    to sample). Measured: LightGBM accepts and fits a typo silently (`max_dept=3`), and
    `LGBMParams(max_dept=3)` raises `TypeError` when the component is built.

- D12. **Changed 2026-10-21 (the user, at D11's review):** a study is one family combination
  and exposure, so each combination's results can be seen with its own best parameters.
  - **Model 1:** a study is `(family, use_exposure)`, with one family for every cohort.
  - **Model 2:** a study is `(total_family, probability_family, use_exposure)`.
  - **Keys:** `MODEL1_STUDY_SETTINGS` and `MODEL2_STUDY_SETTINGS` list these plain tuples,
    and the studies are keyed by them.
  - **Search spaces are built inside.** `suggest_model1_components(trial, *, family,
    use_exposure, cohorts)` and `suggest_model2_components(trial, *, total_family,
    probability_family, use_exposure)` build them from the family. `suggest_model1/2` call
    them, and the refit calls them on `FixedTrial(best_params)` for the best components.
  - **Rename:** the builders' keyword `feature_transformers` is now
    `feature_transformer_dict`; the constant `FEATURE_TRANSFORMERS` keeps its name.
  - **`y_min` stays a builder argument** (the user's choice). The run record holds `seed`
    and `y_min` once.
  - **Unchanged:** the search itself. Check C reproduced T1.6's 8 best values again.

**What does not change:** notebook sections 1–4 and the `cv`/`cv_groups` cell; the models in
`src/age_group_prediction/modeling/` (the S1 `get_params`/`set_params` forwarding in
`IndependentCohortModels` stays: tested, `clone`-safe, simply unused by the notebook);
`scoring.Metric`, `cohort_poisson_deviance`, `cohort_log_loss`; the search spaces, variant lists,
`PREDICTION_FLOOR` as `y_min`, the LightGBM constants, no isotonic calibration, NB2 with
`dispersion` and `alpha` searched; the outputs (per-cohort and composite train/validation
deviance of each family's refit best run, best choices per component, trials per arm, best
family per model).

## 1. Rules for the implementer

- Plan mode summary in chat before each sub-task; one sub-task per approval. Never commit,
  never add a `Co-Authored-By` line, never `git checkout`/`stash`/`reset`/`mv`.
- Before the first edit: `git status` clean except what the previous sub-task left; run the
  baseline `uv run pytest -m "not slow"` once at T1.1 (expect green; stop and report otherwise).
- The user edits the notebook between turns: record its md5 (`md5 -q notebooks/02_experiments.py`)
  after every edit and `cmp` before the next; if it changed, show the diff and ask.
- marimo's editor trims a cell's `return` to the names other cells read; a new cell that reads
  a name re-adds it to the producing cell's return. An undefined name in a cell fails silently
  in a script run: verify every new cell ran with the `app.run()` check (§1.1).
- Format new cells by hand to `ruff format`'s output; do not run whole-file `ruff format`
  (three known hunks in the user's cells). `ruff check notebooks/02_experiments.py` has a
  baseline of 9 findings (7 marimo `return`s PLR1711, F401 `clone`, RUF059 `test_age_group`);
  add none. The notebook is outside mypy's `files`.
- The characterization test `tests/characterization/test_experiments_notebook.py` requires:
  the printed lines of sections 1–3 unchanged; every cell whose source contains the string
  `tune_model` (except the cell with `def tune_model(`) to contain `mo.stop(not `, and at
  least one such cell. So the tuning function keeps the name `tune_model`, and that name must
  not appear in the markdown cell or in comments outside the run cells.
- Probes are read-only: `PYTHONPATH=src .venv/bin/python -W error -c "..."`. Scratch files go
  to the session scratchpad, never into `notebooks/`.
- Review subagents write nothing into the repository.

### 1.1 The two checks used by every notebook sub-task

**Check A, script run (fits nothing, ≈ 5 s):**
```sh
cd /Users/galkampel/Desktop/Projects/age-group-prediction
uv run pytest tests/characterization/test_experiments_notebook.py -q
ruff check notebooks/02_experiments.py | tail -3     # 9 findings before T1.1, 8 after (F401 clone gone)
uv run marimo check notebooks/02_experiments.py       # only the two known empty-cell warnings
```

**Check B, the defs of a full run** (which cells ran). `NAMES` is the sub-task's list:
```sh
PYTHONPATH=src .venv/bin/python -W error - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("nb", "notebooks/02_experiments.py")
nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)
_, defs = nb.app.run()
NAMES = [...]
missing = [n for n in NAMES if n not in defs]
print("missing:", missing)
EOF
```

**Check C, the clicked run on a copy** (T1.6 onward). Build a scratch project: `<scratch>/project/notebooks/02_experiments.py`
= the notebook with `n_trials = 200` replaced by `n_trials = 3`, and `<scratch>/project/configs`
a symlink to the repo's `configs` (the notebook reads the config relative to its own directory;
`src` comes from `PYTHONPATH`). Then:
```sh
PYTHONPATH=src .venv/bin/python -W error - <<'EOF'
import importlib.util, types
from unittest import mock
path = "<scratch>/project/notebooks/02_experiments.py"
spec = importlib.util.spec_from_file_location("nb", path)
nb = importlib.util.module_from_spec(spec)
with mock.patch("marimo.ui.run_button", return_value=types.SimpleNamespace(value=True)):
    spec.loader.exec_module(nb)
    _, defs = nb.app.run()
for name in [...]:        # the sub-task's names, e.g. "MODEL1_STUDIES", "model1_df"
    print(name, name in defs)
print(defs["model1_df"].to_string(index=False))
EOF
```
(Verified on 2026-10-15 on a scratch app: a button's value is `False` in a script run, and the
patch makes `mo.stop` pass.) `optuna` logs failed trials at WARNING to stderr; that is expected
for NB2 and is not a failure of the check.

## 2. T1: section 5 for Model 1 (notebook only)

The old section 5 (every cell from the `## 5) Hyperparameter tuning` markdown cell to the
best-per-model cell, lines 598–1218) is replaced. T1.1 removes all of it and writes the first
cells; the later sub-tasks append cells. Between sub-tasks the notebook always runs.

### [x] T1.1 Header, imports, metric and constants

Delete the old section-5 cells (keep the user's empty cell at lines 593–595 and the
`if __name__` block). Write four cells:

**Cell 5.1, header** (`hide_code=True`):
```python
mo.md(r"""
## 5) Hyperparameter tuning

One Optuna study per model family and exposure (on or off, fixed for the
study), on the composite: the model that is scored is the model that is
tuned, by the cohort Poisson deviance of its cross-validation folds on the
training part, the plain mean over folds. A study's fixed settings are a
frozen dataclass, its component; a trial samples per cohort (per component
for Model 2) the feature transformer by name and the estimator's
hyperparameters, under the names `<component>__<parameter>`, and builds
the estimator, its `CountModel` and the composite with constructor calls.
LightGBM's two losses, Poisson and Gaussian, are separate families, and
the exposure a study setting: a study that sampled the loss split 30
trials 10/20 and did worse than either loss alone, and the exposure
changes the target (a count or a weighted rate). `n_trials = 200`, so
that no categorical arm starves; the table of trials per arm shows it.
The i-th study gets `seed + i`; the studies of one model run
in parallel with joblib. A trial whose fit raises (an NB2 fit that does not
converge) is marked failed and the study goes on; the tables count them.
Only each study's best run is rebuilt, refit on the training part and
scored on training and validation. A Gaussian loss can predict a negative
count, which `CountModel` floors at `y_min` (eps). Model 2 is tuned the
same way, one study per total family (the list `MODEL2_TOTALS`, edited
after Model 1's table), the total's exposure and a classifier family,
under `total__…` and `probability__…`; the probability model ignores the
exposure, and its calibration is searched. The last table shows each model's best family on
validation. The run cells sit behind buttons: opening the notebook fits
nothing.
""")
```

**Cell 5.2, imports:**
```python
import dataclasses
import time
from collections import Counter
from functools import partial

import numpy as np
import optuna
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from optuna.distributions import CategoricalDistribution
from optuna.samplers import TPESampler
from optuna.trial import FixedTrial, TrialState
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from age_group_prediction.modeling import (
    CohortProbabilityModel,
    CountModel,
    IndependentCohortModels,
    NegativeBinomialRegressor,
    TotalTimesProbabilityModel,
)
from age_group_prediction.scoring import Metric, cohort_poisson_deviance

# Optuna logs every trial at INFO; the tuning function repeats this in
# each joblib worker.
optuna.logging.set_verbosity(optuna.logging.WARNING)
```
Return every imported name (marimo trims the list to what is read; re-add as cells arrive).
Also remove `from sklearn.base import clone` and `clone` from the section-1 imports cell
(lines 47 and 74–98): nothing reads it (ruff F401 today).

**Cell 5.3, the metrics:**
```python
# The study's objective and the tables' per-cohort score, with the direction
# Optuna is given; a loss, so "minimize".
COHORT_POISSON_DEVIANCE = Metric(
    "cohort_poisson_deviance", cohort_poisson_deviance, direction="minimize"
)
POISSON_DEVIANCE = Metric("poisson_deviance", mean_poisson_deviance, direction="minimize")
```

**Cell 5.4, constants:**
```python
COHORTS = list(Y.columns)
# The smallest mean a CountModel predicts: a Gaussian loss can predict a
# negative count, the Poisson deviance raises at 0 and stays finite at eps.
PREDICTION_FLOOR = float(np.finfo(float).eps)
# The trees take n_apartments as a feature and the offset; the GLMs take it
# as the exposure only (docs/FEATURE_TRANSFORMATIONS.md §3.3, §4.4), and the
# logistic model needs the scaled plans (it does not converge on `tree`).
TREE_TRANSFORMERS = ("tree", "tree__ses_quadratic")
COUNT_GLM_TRANSFORMERS = (
    "glm_base",
    "count__ses_quadratic",
    "count__daycare_log1p",
    "count__room_share_x_household_size",
    "count__room_share_x_ses",
    "count__room_share_x_household_size_and_ses",
)
PROBABILITY_GLM_TRANSFORMERS = (
    "glm_base",
    "probability__daycare_log1p",
    "probability__room_share_x_median_age",
    "probability__room_share_x_daycare",
    "probability__room_share_x_median_age_and_daycare",
)
# Plain assignments, not the `type` statement: cloudpickle cannot pickle a
# class annotated with a `type` alias defined in a cell, and joblib's
# workers receive the components by cloudpickle.
CountFamily = Literal["lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2"]
ProbabilityFamily = Literal["logistic", "lgbm"]
COUNT_FAMILIES = get_args(CountFamily)
PROBABILITY_FAMILIES = get_args(ProbabilityFamily)
# The candidate feature transformers of each family, given to its components.
COUNT_FAMILY_TRANSFORMERS = {
    "lgbm_poisson": TREE_TRANSFORMERS,
    "lgbm_gaussian": TREE_TRANSFORMERS,
    "poisson_glm": COUNT_GLM_TRANSFORMERS,
    "nb2": COUNT_GLM_TRANSFORMERS,
}
PROBABILITY_FAMILY_TRANSFORMERS = {
    "logistic": PROBABILITY_GLM_TRANSFORMERS,
    "lgbm": TREE_TRANSFORMERS,
}
```

**Done when:** no `hyperparameter_tuning` string in the notebook; Check A passes (test green,
ruff 8 findings, marimo check clean; the `tune_model` test fails as noted at T1.5); Check B with `NAMES = ["COHORT_POISSON_DEVIANCE",
"POISSON_DEVIANCE", "COHORTS", "PREDICTION_FLOOR", "TREE_TRANSFORMERS", "COUNT_FAMILIES",
"PROBABILITY_FAMILIES"]` prints `missing: []`; md5 recorded.

### [x] T1.2 The estimator dataclasses

(Superseded by D10 on 2026-10-21; the current code is in T1.4.)
One cell. An estimator dataclass holds only its estimator's searched hyperparameters, and its
fields are exactly the estimator's keyword names, so T1.3 builds an estimator as
`Estimator(<fixed settings>, **dataclasses.asdict(params))`. The bounds are the old section 5's
(`8ee091a`, lines 703–725). The regressor and the classifier share `LGBMParams`'s ten boosting
fields. `LGBMRegressorParams` adds `objective`, which the family fixes (D6), passed through
`suggest`'s `**fixed`: `LGBMRegressorParams.suggest(trial, prefix, objective="poisson")`. The
classifier's `multiclass` is not searched. (Reworked at the user's review: the first version
had one `LGBMParams` with `build_regressor`/`build_classifier`, and a `build` on each class.
The subclass does not override `suggest`: an override narrowing `**fixed` to `objective` is a
Liskov violation the IDE flagged.)
```python
@dataclasses.dataclass(frozen=True)
class LGBMParams:
    """LightGBM's searched hyperparameters, shared by the regressor and the classifier."""

    max_depth: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    n_estimators: int
    reg_alpha: float
    reg_lambda: float
    min_split_gain: float
    subsample: float
    colsample_bytree: float

    @classmethod
    def suggest(cls, trial, prefix, **fixed):
        return cls(
            max_depth=trial.suggest_int(f"{prefix}max_depth", 3, 8),
            num_leaves=trial.suggest_int(f"{prefix}num_leaves", 7, 63),
            min_child_samples=trial.suggest_int(
                f"{prefix}min_child_samples", 5, 40
            ),
            learning_rate=trial.suggest_float(
                f"{prefix}learning_rate", 0.01, 0.2, log=True
            ),
            n_estimators=trial.suggest_int(f"{prefix}n_estimators", 50, 400),
            reg_alpha=trial.suggest_float(
                f"{prefix}reg_alpha", 1e-8, 10.0, log=True
            ),
            reg_lambda=trial.suggest_float(
                f"{prefix}reg_lambda", 1e-8, 10.0, log=True
            ),
            min_split_gain=trial.suggest_float(f"{prefix}min_split_gain", 0.0, 1.0),
            subsample=trial.suggest_float(f"{prefix}subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                f"{prefix}colsample_bytree", 0.7, 1.0
            ),
            **fixed,
        )

@dataclasses.dataclass(frozen=True)
class LGBMRegressorParams(LGBMParams):
    # LightGBM's name for the loss, "poisson" or "regression" (Gaussian):
    # fixed by the family and recorded, not sampled; `suggest` takes it as
    # a keyword, `suggest(trial, prefix, objective="poisson")`.
    objective: str

@dataclasses.dataclass(frozen=True)
class LGBMClassifierParams(LGBMParams):
    """The classifier's; its objective, multiclass, is not searched."""

@dataclasses.dataclass(frozen=True)
class PoissonGLMParams:
    alpha: float

    @classmethod
    def suggest(cls, trial, prefix):
        return cls(
            alpha=trial.suggest_float(f"{prefix}alpha", 1e-4, 10.0, log=True)
        )

@dataclasses.dataclass(frozen=True)
class NB2Params:
    # The estimated dispersions are 0.12 to 0.18; the penalty's path runs
    # from the unpenalized fit at 1e-3 to all-zero coefficients at 1.
    dispersion: float
    alpha: float

    @classmethod
    def suggest(cls, trial, prefix):
        return cls(
            dispersion=trial.suggest_float(
                f"{prefix}dispersion", 0.01, 1.0, log=True
            ),
            alpha=trial.suggest_float(f"{prefix}alpha", 1e-3, 1.0, log=True),
        )

@dataclasses.dataclass(frozen=True)
class LogisticParams:
    C: float

    @classmethod
    def suggest(cls, trial, prefix):
        return cls(C=trial.suggest_float(f"{prefix}C", 1e-3, 100.0, log=True))
```
The IDE's checker reports `object.__init__` errors on these classes. That is an artifact: in a
marimo cell `dataclasses` is a parameter, so the checker cannot see the decorator.

**Done when:** Check A; Check B with the six class names and `COUNT_FAMILIES`; a probe builds
each class from `suggest(FixedTrial({...}), "x__")`, prints `json.dumps(dataclasses.asdict(p))`
(hyperparameters only), checks that `LGBMRegressor(**asdict(r))` takes `objective`, that a
rebuild is equal, that the classifier's params are not `LGBMRegressorParams`, and that a
missing `objective` raises `TypeError`.

### [x] T1.3 The component dataclasses

(Superseded by D10 on 2026-10-21; the current code is in T1.4.)
(Reworked at T1.4's review, 2026-10-20, D8: the cell's current code is in T1.4.)
One cell. (Redesigned 2026-10-18 at the user's review. The first version held the sampled
choices and built the model; see D3 and D6.) A component holds only what is fixed for one
study: the family (a `Literal`) and, for a count component, the exposure. Given the family it
samples its params class under a prefix, and its feature transformer from its own entry of
the family → names mapping passed in (D3: the classes read no notebook global). An unknown
family raises at construction, a family missing from the mapping when sampling. A frozen
component is hashable, so it keys the studies; it pickles through cloudpickle to the joblib
workers.
```python
# The components read no notebook global: the family's candidate feature
# transformers come in as an argument, so the classes can move to a package.
@dataclasses.dataclass(frozen=True)
class CountComponent:
    """A CountModel's settings fixed for one study; a trial samples the rest under a prefix."""

    family: CountFamily
    use_exposure: bool

    def __post_init__(self):
        # At construction, before the parallel studies start.
        if self.family not in get_args(CountFamily):
            raise ValueError(f"unknown count family {self.family!r}")

    def suggest_feature_transformer(self, trial, prefix, family_transformers):
        """The name of a feature transformer, from ``family_transformers[self.family]``.

        ``family_transformers`` maps each family to the names of the feature
        transformers that suit it, so a family is never given another's.
        """
        if self.family not in family_transformers:
            raise ValueError(f"no feature transformers for family {self.family!r}")
        return trial.suggest_categorical(
            f"{prefix}feature_transformer", list(family_transformers[self.family])
        )

    def suggest_params(self, trial, prefix):
        if self.family == "lgbm_poisson":
            return LGBMRegressorParams.suggest(trial, prefix, objective="poisson")
        elif self.family == "lgbm_gaussian":
            return LGBMRegressorParams.suggest(
                trial, prefix, objective="regression"
            )
        elif self.family == "poisson_glm":
            return PoissonGLMParams.suggest(trial, prefix)
        elif self.family == "nb2":
            return NB2Params.suggest(trial, prefix)
        else:
            raise ValueError(f"unknown count family {self.family!r}")

@dataclasses.dataclass(frozen=True)
class ProbabilityComponent:
    """Model 2's CohortProbabilityModel settings fixed for one study; the exposure does not apply."""

    family: ProbabilityFamily

    def __post_init__(self):
        if self.family not in get_args(ProbabilityFamily):
            raise ValueError(f"unknown probability family {self.family!r}")

    def suggest_feature_transformer(self, trial, prefix, family_transformers):
        """As CountComponent's."""
        if self.family not in family_transformers:
            raise ValueError(f"no feature transformers for family {self.family!r}")
        return trial.suggest_categorical(
            f"{prefix}feature_transformer", list(family_transformers[self.family])
        )

    def suggest_params(self, trial, prefix):
        if self.family == "lgbm":
            return LGBMClassifierParams.suggest(trial, prefix)
        elif self.family == "logistic":
            return LogisticParams.suggest(trial, prefix)
        else:
            raise ValueError(f"unknown probability family {self.family!r}")

    def suggest_calibration_method(self, trial, prefix):
        # No isotonic: its lowest score block can hold no child of a cohort and
        # then maps a building there to probability 0, an infinite log loss
        # (measured 2026-10-15 with the LightGBM classifier).
        return trial.suggest_categorical(
            f"{prefix}calibration_method", [None, "temperature", "sigmoid"]
        )
```

**Done when:** Check A; Check B with both class names, the two `Literal`s and the two mappings; a probe, for every
count family × exposure and both probability families, runs a 2-trial study whose objective
calls the `suggest_*` methods under `"a__"`. It checks four things: the sampled names are
exactly the family's; the transformer is in `family_transformers[family]`; a `FixedTrial(trial.params)`
rebuild is equal; and `json.dumps(asdict(component))` works. Unknown families, and a mapping
without the family, raise `ValueError`; components are equal and hash by value; a cloudpickle round trip
and a `Parallel(n_jobs=2)` pass work.

### [x] T1.4 The models a trial builds

Done 2026-10-19, reworked 2026-10-20 (D8), redesigned 2026-10-21 (D10, revised by D11 and D12):
- the params dataclasses are fields only;
- a component is a complete record that makes its estimator;
- a search space offers explicit suggest methods;
- the builders assemble the components from a trial's draws and turn them into models.

This section holds the current code of the four cells.

**Params:**
```python
# Hyperparameters only, no Optuna: the field names are the estimator's
# keywords, so a component makes its estimator as
# Estimator(<fixed settings>, **asdict(params)).
@dataclasses.dataclass(frozen=True, kw_only=True)
class LGBMParams:
    """LightGBM's searched hyperparameters, for the regressor and the classifier alike."""

    max_depth: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    n_estimators: int
    reg_alpha: float
    reg_lambda: float
    min_split_gain: float
    subsample: float
    colsample_bytree: float

@dataclasses.dataclass(frozen=True, kw_only=True)
class PoissonGLMParams:
    alpha: float

@dataclasses.dataclass(frozen=True, kw_only=True)
class NB2Params:
    dispersion: float
    alpha: float

@dataclasses.dataclass(frozen=True, kw_only=True)
class LogisticParams:
    C: float
```

**Components:**
```python
# Complete records of one part of a run, built by hand or from a trial's
# draws; no Optuna here (D10, D11). They read no notebook global, so the
# classes can move to a package.
@dataclasses.dataclass(frozen=True, kw_only=True)
class Component(abc.ABC):
    """A named part of a composite: its family, feature transformer and hyperparameters."""

    name: str
    family: str
    # A key of FEATURE_TRANSFORMERS, resolved by the builder, so the record
    # stays JSON.
    feature_transformer_name: str

    def __post_init__(self):
        # The subclass annotates `family` with its Literal, which holds the
        # valid names.
        families = get_args(get_type_hints(type(self))["family"])
        if self.family not in families:
            raise ValueError(
                f"unknown {type(self).__name__} family {self.family!r}"
            )

    @abc.abstractmethod
    def make_estimator(self, *, seed):
        """The unfitted estimator: the family's fixed settings and ``hyperparameters``."""

@dataclasses.dataclass(frozen=True, kw_only=True)
class CountComponent(Component):
    """A CountModel's part of a run; ``use_exposure`` makes the target a weighted rate."""

    family: CountFamily
    use_exposure: bool
    hyperparameters: LGBMParams | PoissonGLMParams | NB2Params

    def make_estimator(self, *, seed):
        hyperparameters = dataclasses.asdict(self.hyperparameters)
        if self.family in ("lgbm_poisson", "lgbm_gaussian"):
            # LightGBM's name for the loss, which the family fixes (D6).
            objective = "poisson" if self.family == "lgbm_poisson" else "regression"
            # subsample_freq=1: LightGBM ignores subsample without it; n_jobs=1: one
            # thread per fit, so the parallel studies do not oversubscribe the cores.
            return LGBMRegressor(
                objective=objective,
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
                **hyperparameters,
            )
        elif self.family == "poisson_glm":
            return PoissonRegressor(max_iter=1000, **hyperparameters)
        elif self.family == "nb2":
            return NegativeBinomialRegressor(**hyperparameters)
        else:
            raise ValueError(f"unknown count family {self.family!r}")

@dataclasses.dataclass(frozen=True, kw_only=True)
class ProbabilityComponent(Component):
    """Model 2's CohortProbabilityModel part of a run; the exposure does not apply."""

    family: ProbabilityFamily
    calibration_method: str | None
    hyperparameters: LGBMParams | LogisticParams

    def make_estimator(self, *, seed):
        hyperparameters = dataclasses.asdict(self.hyperparameters)
        if self.family == "lgbm":
            return LGBMClassifier(
                objective="multiclass",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
                **hyperparameters,
            )
        elif self.family == "logistic":
            return LogisticRegression(max_iter=1000, **hyperparameters)
        else:
            raise ValueError(f"unknown probability family {self.family!r}")
```

**Search spaces:**
```python
# Every Optuna call (D10, D11): a search space, given a component's name and
# family, samples its choices from a trial under `<name>__<parameter>`; the
# caller feeds them into the component.
@dataclasses.dataclass(frozen=True, kw_only=True)
class SearchSpace(abc.ABC):
    """The search of one component's choices, given its name and family."""

    name: str
    family: str
    # Set by each subclass: each family's candidate feature transformers, by
    # name. Fixed on the class, so no search space carries another family's.
    FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]]

    def __post_init__(self):
        # At construction, before the parallel studies start.
        families = get_args(get_type_hints(type(self))["family"])
        if self.family not in families:
            raise ValueError(
                f"unknown {type(self).__name__} family {self.family!r}"
            )
        if self.family not in self.FEATURE_TRANSFORMER_NAMES:
            raise ValueError(f"no feature transformers for family {self.family!r}")

    @property
    def prefix(self):
        """The trial's parameter names are ``<name>__<parameter>``."""
        return f"{self.name}__"

    def suggest_feature_transformer_name(self, trial):
        """A name from the family's candidates; Optuna's choices must be primitives."""
        return trial.suggest_categorical(
            f"{self.prefix}feature_transformer",
            list(self.FEATURE_TRANSFORMER_NAMES[self.family]),
        )

    @abc.abstractmethod
    def suggest_hyperparameters(self, trial):
        """The family's params record, sampled under ``prefix``."""

    def _suggest_lgbm(self, trial):
        prefix = self.prefix
        return LGBMParams(
            max_depth=trial.suggest_int(f"{prefix}max_depth", 3, 8),
            num_leaves=trial.suggest_int(f"{prefix}num_leaves", 7, 63),
            min_child_samples=trial.suggest_int(
                f"{prefix}min_child_samples", 5, 40
            ),
            learning_rate=trial.suggest_float(
                f"{prefix}learning_rate", 0.01, 0.2, log=True
            ),
            n_estimators=trial.suggest_int(f"{prefix}n_estimators", 50, 400),
            reg_alpha=trial.suggest_float(
                f"{prefix}reg_alpha", 1e-8, 10.0, log=True
            ),
            reg_lambda=trial.suggest_float(
                f"{prefix}reg_lambda", 1e-8, 10.0, log=True
            ),
            min_split_gain=trial.suggest_float(f"{prefix}min_split_gain", 0.0, 1.0),
            subsample=trial.suggest_float(f"{prefix}subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                f"{prefix}colsample_bytree", 0.7, 1.0
            ),
        )

@dataclasses.dataclass(frozen=True, kw_only=True)
class CountSearchSpace(SearchSpace):
    """A count component's search: its hyperparameters and feature transformer."""

    family: CountFamily
    # The trees take n_apartments as a feature and the offset; the GLMs take
    # it as the exposure only (docs/FEATURE_TRANSFORMATIONS.md §3.3, §4.4).
    FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]] = (
        MappingProxyType(
            {
                **dict.fromkeys(
                    ("lgbm_poisson", "lgbm_gaussian"),
                    ("tree", "tree__ses_quadratic"),
                ),
                **dict.fromkeys(
                    ("poisson_glm", "nb2"),
                    (
                        "glm_base",
                        "count__ses_quadratic",
                        "count__daycare_log1p",
                        "count__room_share_x_household_size",
                        "count__room_share_x_ses",
                        "count__room_share_x_household_size_and_ses",
                    ),
                ),
            }
        )
    )

    def suggest_hyperparameters(self, trial):
        prefix = self.prefix
        if self.family in ("lgbm_poisson", "lgbm_gaussian"):
            return self._suggest_lgbm(trial)
        elif self.family == "poisson_glm":
            return PoissonGLMParams(
                alpha=trial.suggest_float(f"{prefix}alpha", 1e-4, 10.0, log=True)
            )
        elif self.family == "nb2":
            # The estimated dispersions are 0.12 to 0.18; the penalty's path runs
            # from the unpenalized fit at 1e-3 to all-zero coefficients at 1.
            return NB2Params(
                dispersion=trial.suggest_float(
                    f"{prefix}dispersion", 0.01, 1.0, log=True
                ),
                alpha=trial.suggest_float(f"{prefix}alpha", 1e-3, 1.0, log=True),
            )
        else:
            raise ValueError(f"unknown count family {self.family!r}")

@dataclasses.dataclass(frozen=True, kw_only=True)
class ProbabilitySearchSpace(SearchSpace):
    """A probability component's search: its hyperparameters, calibration and feature transformer."""

    family: ProbabilityFamily
    # The logistic model needs the scaled plans (it does not converge on `tree`).
    FEATURE_TRANSFORMER_NAMES: ClassVar[MappingProxyType[str, tuple[str, ...]]] = (
        MappingProxyType(
            {
                "logistic": (
                    "glm_base",
                    "probability__daycare_log1p",
                    "probability__room_share_x_median_age",
                    "probability__room_share_x_daycare",
                    "probability__room_share_x_median_age_and_daycare",
                ),
                "lgbm": ("tree", "tree__ses_quadratic"),
            }
        )
    )

    def suggest_hyperparameters(self, trial):
        if self.family == "lgbm":
            return self._suggest_lgbm(trial)
        elif self.family == "logistic":
            return LogisticParams(
                C=trial.suggest_float(f"{self.prefix}C", 1e-3, 100.0, log=True)
            )
        else:
            raise ValueError(f"unknown probability family {self.family!r}")

    def suggest_calibration_method(self, trial):
        # No isotonic: its lowest score block can hold no child of a cohort and
        # then maps a building there to probability 0, an infinite log loss
        # (measured 2026-10-15 with the LightGBM classifier).
        return trial.suggest_categorical(
            f"{self.prefix}calibration_method", [None, "temperature", "sigmoid"]
        )
```

**Builders** (D12 since 2026-10-21):
```python
# The suggest functions call a search space's methods explicitly and feed the
# draws into a component; the build functions turn components into models,
# with no Optuna. A study is one family combination and exposure (D12): the
# objective calls suggest_model1/2, and the best run's refit rebuilds the best
# components with suggest_model1/2_components on FixedTrial(best_params).
def suggest_count_component(trial, space, *, use_exposure):
    """The count component of ``trial``'s draws in ``space``; the exposure is the study's."""
    # Keyword arguments are evaluated left to right, so this is the order of
    # the draws: the hyperparameters, then the transformer. A seeded study's
    # result depends on it.
    return CountComponent(
        name=space.name,
        family=space.family,
        use_exposure=use_exposure,
        hyperparameters=space.suggest_hyperparameters(trial),
        feature_transformer_name=space.suggest_feature_transformer_name(trial),
    )

def suggest_probability_component(trial, space):
    """The probability component of ``trial``'s draws in ``space``."""
    # The draws, in order: the hyperparameters, the calibration, the transformer.
    return ProbabilityComponent(
        name=space.name,
        family=space.family,
        hyperparameters=space.suggest_hyperparameters(trial),
        calibration_method=space.suggest_calibration_method(trial),
        feature_transformer_name=space.suggest_feature_transformer_name(trial),
    )

def suggest_model1_components(trial, *, family, use_exposure, cohorts):
    """Model 1's components ``trial`` draws: one per cohort, all of ``family``."""
    return tuple(
        suggest_count_component(
            trial,
            CountSearchSpace(name=cohort, family=family),
            use_exposure=use_exposure,
        )
        for cohort in cohorts
    )

def suggest_model2_components(
    trial, *, total_family, probability_family, use_exposure
):
    """Model 2's total and probability components ``trial`` draws, the total first."""
    total = suggest_count_component(
        trial,
        CountSearchSpace(name="total", family=total_family),
        use_exposure=use_exposure,
    )
    probability = suggest_probability_component(
        trial, ProbabilitySearchSpace(name="probability", family=probability_family)
    )
    return total, probability

def make_count_model(component, *, feature_transformer_dict, y_min, seed):
    """The CountModel of a count component."""
    # CountModel clones the transformer at fit, so the shared declaration is safe.
    return CountModel(
        estimator=component.make_estimator(seed=seed),
        use_exposure=component.use_exposure,
        feature_transformer=feature_transformer_dict[
            component.feature_transformer_name
        ],
        y_min=y_min,
    )

def build_model1(components, *, feature_transformer_dict, y_min, seed):
    """Model 1: one CountModel per component, keyed by the component's name, a cohort."""
    return IndependentCohortModels(
        {
            component.name: make_count_model(
                component,
                feature_transformer_dict=feature_transformer_dict,
                y_min=y_min,
                seed=seed,
            )
            for component in components
        }
    )

def build_model2(total, probability, *, feature_transformer_dict, y_min, seed):
    """Model 2: the total's CountModel and the cohort probabilities' classifier."""
    return TotalTimesProbabilityModel(
        total_model=make_count_model(
            total,
            feature_transformer_dict=feature_transformer_dict,
            y_min=y_min,
            seed=seed,
        ),
        probability_model=CohortProbabilityModel(
            estimator=probability.make_estimator(seed=seed),
            calibration_method=probability.calibration_method,
            feature_transformer=feature_transformer_dict[
                probability.feature_transformer_name
            ],
        ),
    )

def suggest_model1(
    trial, *, family, use_exposure, cohorts, feature_transformer_dict, y_min, seed
):
    """Model 1 of the components ``trial`` draws for one family and exposure."""
    components = suggest_model1_components(
        trial, family=family, use_exposure=use_exposure, cohorts=cohorts
    )
    return build_model1(
        components,
        feature_transformer_dict=feature_transformer_dict,
        y_min=y_min,
        seed=seed,
    )

def suggest_model2(
    trial,
    *,
    total_family,
    probability_family,
    use_exposure,
    feature_transformer_dict,
    y_min,
    seed,
):
    """Model 2 of the components ``trial`` draws for one family pair and exposure."""
    total, probability = suggest_model2_components(
        trial,
        total_family=total_family,
        probability_family=probability_family,
        use_exposure=use_exposure,
    )
    return build_model2(
        total,
        probability,
        feature_transformer_dict=feature_transformer_dict,
        y_min=y_min,
        seed=seed,
    )
```

**Done when** (D11, met 2026-10-21):
- Check A green (5/5).
- Check B: the new names are defined. Absent: `MODEL1_SEARCH_SPACES`, D10's `suggest`
  methods and `PARAMS`. A search space's fields are `name` and `family`.
- Per family, the trial names are as before (11, 11, 2, 3, 12, 3).
- The best components of a real NB2 study rebuild from `FixedTrial`, serialize to JSON and
  fit (best 2.2716, unchanged).
- A component built by hand makes its estimator.
- Failures raise:
  - an unknown family, in a component or a search space (`ValueError`);
  - a LightGBM typo in `LGBMParams`, or `use_exposure` given to a search space
    (`TypeError`).
- Hashing, cloudpickle and `Parallel` work.
- A Model 2 pair builds and fits (14 trial names).
- Check C at 3 trials reproduces T1.6's 8 best values exactly (13.8 s).

### [x] T1.5 `cv_score` and `tune_model`

One cell (the user's fold variables and the aggregator of D9, 2026-10-21):
```python
def cv_score(model, X, Y, exposure, cv, cv_groups, metric, aggregator=None):
    """The aggregated fold score of ``model`` refit on each fold's training rows (``fit`` replaces all state).

    ``aggregator`` combines the fold scores; None means the plain mean.
    """
    # The aggregator decides the selection (D9); the plain mean by default,
    # since the folds differ by one row at most.
    aggregator = Mean() if aggregator is None else aggregator
    scores, fold_sizes = [], []
    for train_index, val_index in cv.split(X, Y, cv_groups):
        X_train, Y_train, exposure_train = (
            X.iloc[train_index],
            Y.iloc[train_index],
            exposure.iloc[train_index],
        )
        X_val, Y_val, exposure_val = (
            X.iloc[val_index],
            Y.iloc[val_index],
            exposure.iloc[val_index],
        )
        model.fit(X_train, Y_train, exposure=exposure_train)
        Y_val_pred = model.predict(X_val, exposure=exposure_val)
        scores.append(model.evaluate(Y_val, Y_val_pred, metric))
        fold_sizes.append(len(val_index))
    return aggregator.aggregate(scores, fold_sizes)

def tune_model(
    suggest_model,
    X,
    Y,
    exposure,
    cv,
    cv_groups,
    metric,
    *,
    study_seed,
    n_trials,
    aggregator=None,
):
    """One study of the model ``suggest_model(trial)`` builds, on the rows of ``X``; the study and its seconds.

    Runs in a joblib worker, a fresh process at Optuna's default INFO
    verbosity, so it is set here too. A trial whose fit raises RuntimeError
    (an NB2 fit that does not converge) is marked FAIL and the study goes on;
    without ``catch`` one such trial would end the study. ``aggregator`` goes
    to ``cv_score``; None means the plain mean.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Optuna passes the trial alone; the data is bound here, not to a cell.
        return cv_score(
            suggest_model(trial), X, Y, exposure, cv, cv_groups, metric, aggregator
        )

    study = optuna.create_study(
        direction=metric.direction, sampler=TPESampler(seed=study_seed)
    )
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
    return study, time.perf_counter() - start
```
**Done when:** Check A (the test `test_a_tuning_cell_outside_a_run_button_is_caught` fails from
T1.1 until T1.6, since no cell names `tune_model` behind a button yet; the other four tests of
the file must pass; say so in the report); Check B; a probe calls `tune_model` directly with `n_trials=2` for
`partial(suggest_model1, family="poisson_glm", use_exposure=True, cohorts=COHORTS, …)` (D12's names) on the notebook's training data and prints `study.best_value` and the seconds (2.48456 by default and with `Mean()`; met 2026-10-21, with the other aggregators of D9).

### [x] T1.6 Model 1's button and run cell

Two cells (D12's names since 2026-10-21). The button cell also holds the list of the studies it runs:
```python
# One study per count family and exposure (D6, D12): each combination is
# tuned, and then scored, with its own best parameters.
MODEL1_STUDY_SETTINGS = [
    (family, use_exposure)
    for family in COUNT_FAMILIES
    for use_exposure in (True, False)
]
run_model1 = mo.ui.run_button(label="Tune Model 1")
mo.md(
    f"{run_model1} {len(MODEL1_STUDY_SETTINGS)} studies × `n_trials` trials, in parallel."
)
```
```python
mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
# One process per study; study i gets seed + i.
model1_runs = Parallel(n_jobs=len(MODEL1_STUDY_SETTINGS))(
    delayed(tune_model)(
        partial(
            suggest_model1,
            family=family,
            use_exposure=use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group.X,
        train_age_group.Y,
        train_age_group.exposure,
        cv,
        cv_groups,
        COHORT_POISSON_DEVIANCE,
        study_seed=seed + i,
        n_trials=n_trials,
    )
    for i, (family, use_exposure) in enumerate(MODEL1_STUDY_SETTINGS)
)
MODEL1_STUDIES = dict(zip(MODEL1_STUDY_SETTINGS, model1_runs, strict=True))
```
`MODEL1_STUDIES` maps a study's settings `(family, use_exposure)` (D12) to `(optuna.Study, seconds)`. The machine has 8 cores
(6 performance), so the 8 studies take one process each.

**Done when:** Check A fully green (the `tune_model` rule holds again); Check C with
`n_trials = 3` prints `MODEL1_STUDIES True` and, for each component, `len(study.trials) == 3`;
record the wall time of Check C.

Met 2026-10-21:
- Check A green (the notebook test 5/5).
- Check B: `MODEL1_COMPONENTS` (since D11 `MODEL1_STUDY_SETTINGS`) and `run_model1` are defined, and `MODEL1_STUDIES` is not,
  because the script run stops at the button.
- Check C: all 8 studies at 3 trials, no FAIL. The whole notebook run takes 14.7 s; the
  studies take 1.7 s (`poisson_glm`, exposure on) to 7.1 s (`nb2`, exposure off). The best
  values equal the planning probe's, so the seeded run reproduces:

  | Family | Exposure on | Exposure off |
  |---|---|---|
  | `lgbm_poisson` | 2.190 | 2.281 |
  | `lgbm_gaussian` | 2.191 | 2.315 |
  | `poisson_glm` | 2.570 | 2.878 |
  | `nb2` | 2.283 | 2.890 |

- At NB2's pace of about 2 s a trial, a 200-trial run should take roughly 7 minutes; the
  user's click (T1.7) measures it.

### [ ] T1.7 Refit, tables and the Model 1 views

Three cells: the functions, the Model 1 score cell, the Model 1 display cell.
```python
def refit_best(study, suggest_model, train_age_group, val_age_group, cohort_metric, metric):
    """Rebuild the study's best model, refit it on the training part; the model and its scores.

    The model is rebuilt by ``suggest_model`` on a FixedTrial of the best
    parameters: the same code path as the search, and a missing or foreign
    value raises. The scores: the per-cohort metric on each cohort column and
    the composite metric, on training and validation.
    """
    model = suggest_model(FixedTrial(study.best_params)).fit(
        train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure
    )
    pred_train = model.predict(train_age_group.X, exposure=train_age_group.exposure)
    pred_val = model.predict(val_age_group.X, exposure=val_age_group.exposure)
    rows = [
        {
            "cohort": cohort,
            "train": model.evaluate(train_age_group.Y[cohort], pred_train[cohort], cohort_metric),
            "val": model.evaluate(val_age_group.Y[cohort], pred_val[cohort], cohort_metric),
        }
        for cohort in train_age_group.Y.columns
    ]
    rows.append(
        {
            "cohort": "composite",
            "cv_best": study.best_value,
            "train": model.evaluate(train_age_group.Y, pred_train, metric),
            "val": model.evaluate(val_age_group.Y, pred_val, metric),
            "n_failed": sum(trial.state == TrialState.FAIL for trial in study.trials),
        }
    )
    return model, rows


def categorical_names(study):
    """The names a trial draws from a categorical; every trial has the same names."""
    return [
        name
        for name, distribution in study.best_trial.distributions.items()
        if isinstance(distribution, CategoricalDistribution)
    ]


def best_choices(study):
    """One row per component (the name before the first ``__``) with its best categorical values."""
    rows = {}
    for name in categorical_names(study):
        component, parameter = name.split("__", 1)
        rows.setdefault(component, {"component": component})[parameter] = (
            study.best_trial.params[name]
        )
    return list(rows.values())


def arm_counts(study):
    """Trials per value of every categorical, so a starved arm is visible before a family is declared worse."""
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return [
        {"parameter": name, "value": value, "n_trials": count}
        for name in categorical_names(study)
        for value, count in Counter(trial.params[name] for trial in complete).items()
    ]
```
Model 1 score cell:
```python
MODEL1_MODELS, MODEL1_BEST_RUNS = {}, []
_score_rows, _choice_rows, _arm_rows = [], [], []
for (_family, _use_exposure), (_study, _seconds) in MODEL1_STUDIES.items():
    _model, _rows = refit_best(
        _study,
        partial(
            suggest_model1,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=FEATURE_TRANSFORMERS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        POISSON_DEVIANCE,
        COHORT_POISSON_DEVIANCE,
    )
    # The study's settings columns: its family and exposure (D12).
    _settings = {"family": _family, "use_exposure": _use_exposure}
    MODEL1_MODELS[_family, _use_exposure] = _model
    # The record, JSON for a later logging stop: the best components, complete
    # (D10-D12), rebuilt from the best trial's parameters; seed and y_min once.
    _best = suggest_model1_components(
        FixedTrial(_study.best_params),
        family=_family,
        use_exposure=_use_exposure,
        cohorts=COHORTS,
    )
    MODEL1_BEST_RUNS.append(
        {
            **_settings,
            "components": [dataclasses.asdict(c) for c in _best],
            "cv_best": _study.best_value,
            "seed": seed,
            "y_min": PREDICTION_FLOOR,
        }
    )
    _rows[-1]["seconds"] = _seconds
    _score_rows.extend({**_settings, **row} for row in _rows)
    _choice_rows.extend({**_settings, **row} for row in best_choices(_study))
    _arm_rows.extend({**_settings, **row} for row in arm_counts(_study))
model1_df, model1_choices_df, model1_arms_df = (
    pd.DataFrame(_score_rows),
    pd.DataFrame(_choice_rows),
    pd.DataFrame(_arm_rows),
)
```
(Since D8 a study is a tuple of components. If a later study mixed families across
cohorts, the cohort rows would take their columns from their own component; settle at T1.7's
plan. If the implementer prefers a function `score_studies(studies, suggest_model_for, ...)` over
the loop, the same inputs go in the signature; the user's style accepts either. Say which.)
Display cell: the current one (`mo.vstack` of three `mo.md` code blocks), titles "Model 1:
cohort Poisson deviance (cv_best: the study's value; n_failed: trials whose fit raised)",
"Best combination per component", "Trials per arm".

**Done when:** Check A; Check C prints `model1_df` with 32 rows (8 studies × 4), with the
columns `family` and `use_exposure`; the choices frame with 24 rows (8 × 3 cohorts), whose only
categorical is `feature_transformer`; the arms frame
with the counts summing to the COMPLETE trials per parameter, and
`json.dumps(defs["MODEL1_BEST_RUNS"])` works; the user clicks the button in `marimo edit` at
`n_trials = 200` and reports the wall time (today's package version: ≈ 5 min).

### [ ] T1.8 Independent review of T1

A review subagent (writes nothing into the repository) reads the new section against this
plan and the decisions; each finding is reproduced before it is acted on. Then the file-by-file
summary and the suggested commit command for the user.

## 3. T2: Model 2 and the best-per-model table (notebook only)

### [ ] T2.1 `MODEL2_TOTALS`, the button and the run cell
```python
# Model 2's total families, a list to edit after Model 1's table above. The
# studies are this list × the total's exposure × the probability families.
MODEL2_TOTALS = ["lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2"]
MODEL2_STUDY_SETTINGS = [
    (total_family, probability_family, use_exposure)
    for total_family in MODEL2_TOTALS
    for probability_family in PROBABILITY_FAMILIES
    for use_exposure in (True, False)
]
```
Button: `run_model2 = mo.ui.run_button(label="Tune Model 2")`, text
`f"{run_model2} {len(MODEL2_STUDY_SETTINGS)} studies × `n_trials` trials, in parallel."`.
Run cell: as T1.6 with `Parallel(n_jobs=min(len(MODEL2_STUDY_SETTINGS), 8))` (16 studies at four
totals: two per process on 8 cores),
`partial(suggest_model2, total_family=total_family, probability_family=probability_family,
use_exposure=use_exposure, feature_transformer_dict=FEATURE_TRANSFORMERS,
y_min=PREDICTION_FLOOR, seed=seed)` and `study_seed=seed + i`. The keys are the
`(total_family, probability_family, use_exposure)` settings, giving `MODEL2_STUDIES` (D12).
**Done when:** Check A; Check C prints `MODEL2_STUDIES True` with 16 studies of 3 trials.

### [ ] T2.2 Model 2's score and display cells, the best-per-model cell
The score cell mirrors T1.7 with the `suggest_model2` partials, giving `MODEL2_MODELS`,
`MODEL2_BEST_RUNS` (`{"components": [asdict(best_total), asdict(best_probability)], "cv_best": …}`,
the best components of D10), `model2_df`, `model2_choices_df` and `model2_arms_df`.
The settings columns are `total_family`, `use_exposure` and `probability_family`; the choices
cover the components `total` and `probability`. The display cell is as Model 1's. The
best-per-model cell is today's (lines 1197–1218 at `8ee091a`), adapted to the settings
columns.
**Done when:** Check A; Check C prints `model2_df` with 64 rows (16 × 4), the choices frame with
32 rows, `best_df` with 2 rows; the user clicks both buttons at 200 and reports the times.

### [ ] T2.3 Independent review of T2, summary, commit suggestion.

## 4. T3: the metric constants leave `scoring.py` (src, tests, docs)

- `src/age_group_prediction/scoring.py`: delete `POISSON_DEVIANCE`, `RMSE`, `MAE`,
  `COHORT_LOG_LOSS`, `COHORT_POISSON_DEVIANCE`, their `__all__` entries, the now-unused
  imports (`mean_absolute_error`, `root_mean_squared_error`; `mean_poisson_deviance` stays
  for `cohort_poisson_deviance`); the module docstring's mention of `hyperparameter_tuning`
  becomes "the notebook".
- Tests: `tests/unit/test_scoring.py` (the parametrized `test_ready_made_metrics_wrap_their_function`
  goes; the `Metric` tests stay), `test_modeling_count_model.py:28` and `:341`,
  `test_inspection_permutation.py:29` and `:216`, `test_inspection_bootstrap.py:17`, `:223`,
  `:225`: each builds its metric inline, e.g. `Metric("poisson_deviance", mean_poisson_deviance)`.
  The tuning tests that import the constants are deleted in T4 (do T3 and T4 in this order, or
  T4 first; T4 first is simpler because it removes 8 of the usages).
- Doc code blocks (run as written): `docs/MODEL_INSPECTION.md:59`,
  `docs/INDEPENDENT_COHORT_MODELS.md:341`, `docs/TOTAL_TIMES_PROBABILITY_MODEL.md:376`: the
  import line becomes `from age_group_prediction.scoring import Metric` plus the sklearn
  function, and one line builds the metric. Prose in the plan/validation docs stays (history).
- **Open point for the user (ask at T3's plan summary):** "remove the metrics from scoring"
  is read as the constants only; `Metric`, `cohort_poisson_deviance`, `cohort_log_loss` stay
  because `inspection` and the model docs use them through a `Metric`.

**Done when:** `grep -rn "COHORT_POISSON_DEVIANCE\|COHORT_LOG_LOSS\|POISSON_DEVIANCE\|RMSE\b\|MAE\b" src tests`
returns only prose in the old stack (`models/`, `experiment/`, `bundle_spy.py`,
`test_final_selection.py`, `test_direct_cohort.py`, `test_final_evaluation.py`); `uv run mypy`;
the changed tests under `-W error`; the non-slow suite; the three doc blocks executed by probe.

## 5. T4: delete the `hyperparameter_tuning` package

- Delete `src/age_group_prediction/hyperparameter_tuning/` (4 modules + `__init__`),
  `tests/unit/test_hyperparameter_tuning_{aggregation,evaluator,integration,parameters,study}.py`,
  `docs/HYPERPARAMETER_TUNING.md`, `docs/HYPERPARAMETER_TUNING_PLAN.md`,
  `docs/HYPERPARAMETER_TUNING_REVIEW.md`, `docs/HYPERPARAMETER_TUNING_VALIDATION.md` (plain
  `rm`; the user commits the deletions). Keep `src/age_group_prediction/aggregation.py` and
  `tests/unit/test_aggregation.py`, the aggregator's new home (D9).
- Edit: `pyproject.toml` (the package's entries in `[tool.mypy] files` and overrides, then
  `uv run --frozen`), `src/age_group_prediction/_config.py` docstring, `README.md`,
  `docs/README.md`, `docs/MODULE_REFERENCE.md`, `docs/INDEPENDENT_COHORT_MODELS.md` §3/§4/§6
  (the sentences pointing at the package; the forwarding stays documented as a `set_params`
  feature), `docs/SPLITTING.md` and the others listed by
  `grep -rln "hyperparameter_tuning\|HYPERPARAMETER_TUNING" docs README.md` where the text
  states current API (plan docs of finished work: a one-line note that the package was removed
  on this date, nothing more). `optuna` stays in `pyproject.toml`.

**Done when:** the grep above returns nothing in `src`, `tests`, `README.md` and the current
guides; `uv run mypy`; the non-slow suite (expect the count to drop by the five files' tests);
the notebook test green.

## 6. T5: docs for the new design

- `docs/EXPERIMENT_NOTEBOOK_HANDOFF.md`: §1 item 3 rewritten (the S1–S5 block becomes one
  paragraph of history; the new design and its dataclasses described), §2 state, §3 the
  decisions D1–D7 of this plan, §6 the prompt for the performance pass.
- `docs/EXPERIMENT_NOTEBOOK_PLAN.md`: E4, E5, E11 (`evaluator.build_model(best_params)` →
  `suggest_model1(FixedTrial(best_params), …)`), E12, D2/D3 and the status line.
- The dataclass-vs-TypedDict comparison (below) into the handoff's decisions.

**Dataclass vs TypedDict, for the record:**

| | frozen dataclass (chosen) | TypedDict |
|---|---|---|
| JSON out | `json.dumps(asdict(component))`; an estimator's params likewise (probed) | the dict itself |
| JSON in | a constructor per class, or re-running `suggest` on a `FixedTrial` (what the notebook does) | `json.loads`; nothing checks the keys |
| Immutability | frozen and hashable; a component keys its study | a plain dict, not hashable |
| Equality | by value (a `FixedTrial` rebuild of the params is equal: the T1.3 check) | by value |
| Dispatch on the params | `isinstance` in `count_estimator` / `probability_estimator` | `isinstance` does not work on TypedDicts; dispatch needs a `family` key |
| Behaviour | `suggest` (params) and `suggest_*` (components) on the class; no `build` (D3) | functions only |

## 7. Later: the performance pass (facts, no decision)

- **Pruning mid-CV.** Optuna's `WilcoxonPruner` (4.9: `p_threshold=0.1`, `n_startup_steps=2`)
  is the pruner designed for a mean over fixed instances such as CV folds: it pairs fold k of
  the trial with fold k of the best trial by `step`, and needs `trial.report(score, step=fold)`
  and the objective returning the partial mean on `should_prune()`. `MedianPruner`,
  `SuccessiveHalving`, `Hyperband` assume a converging learning curve, which fold scores are
  not. With 5 folds the one-sided exact test gives p = 1/2^n after n folds all worse: 0.125
  after 3, 0.0625 after 4, so at 0.1 a trial is pruned after fold 4 at the earliest (≤ 20% of
  its cost saved), at 0.2 after fold 3 (≤ 40%). Worth it only where trials are slow (the
  LightGBM classifier pairs); decide from T2's timings. `cv_score` already keeps the per-fold
  list, so the change is local.
- **Parallelization.** Today one process per study (joblib; seeded studies identical in
  parallel). Keeps the seeded search reproducible: folds of one trial in parallel (trials stay
  sequential), or the three cohort fits of Model 1 inside the composite (a src change). Loses
  it: `study.optimize(n_jobs>1)` (threads; order not reproducible) and multi-process trials on
  a shared storage. The machine has 8 cores (6 performance); Model 2 already runs up to 16 studies at
  once, so per-fold parallelism competes with it.

## 8. End-to-end verification (after T2, again after T4)

1. `uv run pytest -m "not slow"` green; the notebook's script run fits nothing (≈ 5 s).
2. Check C with both buttons patched at `n_trials = 3`: every table renders;
   `json.dumps` of `MODEL1_BEST_RUNS` and `MODEL2_BEST_RUNS` works.
3. The user clicks both buttons at 200 in `marimo edit`; wall times recorded in the handoff.
