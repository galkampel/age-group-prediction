# Re-implement the hyperparameter tuning inside the notebook

**Status (2026-10-18):** approved; T1.1 committed (`0416267`); T1.2 done and reworked after the user's review (estimator dataclasses hold hyperparameters only; LightGBM's loss is a family, D6), uncommitted, notebook md5 `99239ad902971970403cc290aa62f266`; next T1.3. The implementing session updates this line and ticks the sub-tasks.

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
- D2. A trial's sampled values are **frozen dataclasses** (`dataclasses.dataclass(frozen=True)`),
  JSON-serializable through `dataclasses.asdict`: every field is a JSON scalar; a feature
  variant is stored as its key in `FEATURE_VARIANTS`. (The comparison with TypedDict: §6.)
- D3. Each dataclass has a classmethod `suggest(trial, prefix, …)` calling `trial.suggest_*`
  inline. The component and run dataclasses have a `build(…)` that calls the constructors
  explicitly (no `set_params`). An estimator dataclass holds only that estimator's searched
  hyperparameters, and has no `build` (the user, 2026-10-18, at T1.2's review).
- D4. The metric is built **in the notebook**: `Metric("cohort_poisson_deviance",
  cohort_poisson_deviance, direction="minimize")`; `optuna.create_study(direction=metric.direction)`;
  the objective returns the metric's value unchanged. The module-level constants leave
  `scoring.py` (T3); the `Metric` class and the two functions stay there.
- D5. Pruning and parallelization beyond today's parallel studies are **not decided now**; a
  later performance pass (§7 holds the facts for it).
- D6. Still in force from 2026-10-15: one joint study per family on the composite; a trial
  chooses per component its exposure and variant; `n_trials = 200`; studies of
  one model in parallel with joblib, study i seeded `seed + i`; only the best run is refit;
  Model 2's totals come from the edited list `MODEL2_TOTALS`; `split_method = "random"`.
  **Changed 2026-10-18 (the user, at T1.2's review):** LightGBM's loss is no longer a sampled
  categorical. `lgbm_poisson` and `lgbm_gaussian` are separate families, and
  `LGBMRegressorParams.objective` is fixed by the family. Why: TPE here is univariate (D7),
  so it cannot learn that good hyperparameters differ per loss; a 30-trial study that sampled
  the loss split 10/20 and did worse than either loss alone; and the family table compares
  the losses directly. The cost: one run cannot mix the losses across cohorts, and there is
  one more study (4 for Model 1, up to 8 pairs for Model 2).
- D7. Settled by probe in this session (Optuna 4.9.0, joblib 1.5.3): the trial value is the
  plain mean of the fold scores (KFold on 917 rows: folds of 184/183 rows); the sampler is
  `TPESampler(seed=study_seed)` (the `multivariate=True` option raises `ExperimentalWarning`
  under `-W error`); the best run is rebuilt with `suggest(optuna.trial.FixedTrial(best_params))`
  (exact for `None`/`bool`/`str`, raises `ValueError` on a missing name or a foreign value);
  `study.optimize(..., catch=(RuntimeError,))` keeps a study alive when an NB2 fit does not
  converge (without it the study stops and `Parallel` discards the sibling studies; returning
  `inf` would hide the traceback); an in-memory `optuna.Study` pickles back whole from a loky
  worker (`trials`, `best_trial.distributions`, `best_params` all present).

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

One Optuna study per model family, on the composite: the model that is
scored is the model that is tuned, by the cohort Poisson deviance of its
cross-validation folds on the training part, the plain mean over folds. A
trial's choices are a frozen dataclass: per cohort (per component for
Model 2) the exposure on or off, the feature variant by name and the
estimator's hyperparameters, under the names `<component>__<parameter>`.
The dataclass builds its model with constructor calls, and
`dataclasses.asdict` is its JSON record. LightGBM's two losses, Poisson
and Gaussian, are separate families: a study that sampled the loss split
30 trials 10/20 and did worse than either loss alone. `n_trials = 200`, so
that no categorical arm starves; the table of trials per arm shows it.
The i-th study gets `seed + i`; the studies of one model run
in parallel with joblib. A trial whose fit raises (an NB2 fit that does not
converge) is marked failed and the study goes on; the tables count them.
Only each family's best run is rebuilt, refit on the training part and
scored on training and validation. A Gaussian loss can predict a negative
count, which `CountModel` floors at `y_min` (eps). Model 2 is tuned the
same way, one study per pair of a total family (the list `MODEL2_TOTALS`,
edited after Model 1's table) and a classifier family, under `total__…`
and `probability__…`; the probability model ignores the exposure, and its
calibration is searched. The last table shows each model's best family on
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
TREE_VARIANTS = ("tree", "tree__ses_quadratic")
COUNT_GLM_VARIANTS = (
    "glm_base",
    "count__ses_quadratic",
    "count__daycare_log1p",
    "count__room_share_x_household_size",
    "count__room_share_x_ses",
    "count__room_share_x_household_size_and_ses",
)
PROBABILITY_GLM_VARIANTS = (
    "glm_base",
    "probability__daycare_log1p",
    "probability__room_share_x_median_age",
    "probability__room_share_x_daycare",
    "probability__room_share_x_median_age_and_daycare",
)
COUNT_FAMILIES = ("lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2")
PROBABILITY_FAMILIES = ("logistic", "lgbm")
```

**Done when:** no `hyperparameter_tuning` string in the notebook; Check A passes (test green,
ruff 8 findings, marimo check clean; the `tune_model` test fails as noted at T1.5); Check B with `NAMES = ["COHORT_POISSON_DEVIANCE",
"POISSON_DEVIANCE", "COHORTS", "PREDICTION_FLOOR", "TREE_VARIANTS", "COUNT_FAMILIES",
"PROBABILITY_FAMILIES"]` prints `missing: []`; md5 recorded.

### [x] T1.2 The estimator dataclasses

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

### [ ] T1.3 The component dataclasses

One cell. (Rewritten 2026-10-18 after T1.2's rework, a sketch to re-verify at T1.3's plan.)
`family` is fixed per study (not sampled) and recorded, so `asdict` says which estimator the
nested dict is. The estimators are built here, from the fixed settings plus
`**dataclasses.asdict(self.params)`; the estimator dataclasses have no `build`. The variant
tuples are passed in, not closed over.
```python
@dataclasses.dataclass(frozen=True)
class CountComponent:
    """One CountModel's choices: a cohort's in Model 1, the total's in Model 2."""

    family: str  # one of COUNT_FAMILIES
    params: LGBMRegressorParams | PoissonGLMParams | NB2Params
    feature_variant: str  # a key of FEATURE_VARIANTS
    use_exposure: bool

    @classmethod
    def suggest(cls, trial, prefix, *, family, tree_variants, glm_variants):
        if family == "lgbm_poisson":
            params = LGBMRegressorParams.suggest(trial, prefix, objective="poisson")
            variants = tree_variants
        elif family == "lgbm_gaussian":
            params = LGBMRegressorParams.suggest(trial, prefix, objective="regression")
            variants = tree_variants
        elif family == "poisson_glm":
            params, variants = PoissonGLMParams.suggest(trial, prefix), glm_variants
        elif family == "nb2":
            params, variants = NB2Params.suggest(trial, prefix), glm_variants
        else:
            raise ValueError(f"unknown count family {family!r}")
        return cls(
            family=family,
            params=params,
            feature_variant=trial.suggest_categorical(
                f"{prefix}feature_variant", list(variants)
            ),
            use_exposure=trial.suggest_categorical(f"{prefix}use_exposure", [True, False]),
        )

    def build(self, *, feature_variants, y_min, seed):
        params = dataclasses.asdict(self.params)
        if isinstance(self.params, LGBMRegressorParams):
            # subsample_freq=1: LightGBM ignores subsample without it; n_jobs=1: one
            # thread per fit, so the parallel studies do not oversubscribe the cores.
            estimator = LGBMRegressor(
                subsample_freq=1, n_jobs=1, verbosity=-1, random_state=seed, **params
            )
        elif isinstance(self.params, PoissonGLMParams):
            estimator = PoissonRegressor(max_iter=1000, **params)
        else:
            estimator = NegativeBinomialRegressor(**params)
        # CountModel clones the transformer at fit, so the shared declaration is safe.
        return CountModel(
            estimator=estimator,
            use_exposure=self.use_exposure,
            feature_transformer=feature_variants[self.feature_variant],
            y_min=y_min,
        )


@dataclasses.dataclass(frozen=True)
class ProbabilityComponent:
    """Model 2's CohortProbabilityModel choices; the exposure does not apply."""

    family: str  # one of PROBABILITY_FAMILIES
    params: LogisticParams | LGBMClassifierParams
    calibration_method: str | None
    feature_variant: str

    @classmethod
    def suggest(cls, trial, prefix, *, family, tree_variants, glm_variants):
        if family == "logistic":
            params, variants = LogisticParams.suggest(trial, prefix), glm_variants
        elif family == "lgbm":
            params, variants = LGBMClassifierParams.suggest(trial, prefix), tree_variants
        else:
            raise ValueError(f"unknown probability family {family!r}")
        # No isotonic: its lowest score block can hold no child of a cohort and
        # then maps a building there to probability 0, an infinite log loss
        # (measured 2026-10-15 with the LightGBM classifier).
        return cls(
            family=family,
            params=params,
            calibration_method=trial.suggest_categorical(
                f"{prefix}calibration_method", [None, "temperature", "sigmoid"]
            ),
            feature_variant=trial.suggest_categorical(
                f"{prefix}feature_variant", list(variants)
            ),
        )

    def build(self, *, feature_variants, seed):
        params = dataclasses.asdict(self.params)
        if isinstance(self.params, LGBMClassifierParams):
            estimator = LGBMClassifier(
                objective="multiclass",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
                **params,
            )
        else:
            estimator = LogisticRegression(max_iter=1000, **params)
        return CohortProbabilityModel(
            estimator=estimator,
            calibration_method=self.calibration_method,
            feature_transformer=feature_variants[self.feature_variant],
        )
```

**Done when:** Check A; Check B with both class names; a probe runs
`CountComponent.suggest(trial, "a__", family=f, tree_variants=..., glm_variants=...)` for the
four families inside a 2-trial `optuna` study with a dummy objective and prints
`json.dumps(dataclasses.asdict(component))` for each, then `build(...)` fits one on
`train_age_group` rows (import the notebook with `app.run()` to get the data, or use
`defs`); the ProbabilityComponent likewise for both families.

### [ ] T1.4 The run dataclasses

One cell.
```python
@dataclasses.dataclass(frozen=True)
class Model1Run:
    """IndependentCohortModels: one CountComponent per cohort, under `<cohort>__`."""

    cohort_components: dict[str, CountComponent]

    @classmethod
    def suggest(cls, trial, *, family, cohorts, tree_variants, glm_variants):
        return cls(
            {
                cohort: CountComponent.suggest(
                    trial,
                    f"{cohort}__",
                    family=family,
                    tree_variants=tree_variants,
                    glm_variants=glm_variants,
                )
                for cohort in cohorts
            }
        )

    def build(self, *, feature_variants, y_min, seed):
        return IndependentCohortModels(
            {
                cohort: component.build(
                    feature_variants=feature_variants, y_min=y_min, seed=seed
                )
                for cohort, component in self.cohort_components.items()
            }
        )


@dataclasses.dataclass(frozen=True)
class Model2Run:
    """TotalTimesProbabilityModel: the total under `total__`, the classifier under `probability__`."""

    total: CountComponent
    probability: ProbabilityComponent

    @classmethod
    def suggest(
        cls, trial, *, total_family, probability_family, tree_variants,
        count_glm_variants, probability_glm_variants,
    ):
        return cls(
            total=CountComponent.suggest(
                trial, "total__", family=total_family,
                tree_variants=tree_variants, glm_variants=count_glm_variants,
            ),
            probability=ProbabilityComponent.suggest(
                trial, "probability__", family=probability_family,
                tree_variants=tree_variants, glm_variants=probability_glm_variants,
            ),
        )

    def build(self, *, feature_variants, y_min, seed):
        return TotalTimesProbabilityModel(
            total_model=self.total.build(
                feature_variants=feature_variants, y_min=y_min, seed=seed
            ),
            probability_model=self.probability.build(
                feature_variants=feature_variants, seed=seed
            ),
        )
```
(Format the long signatures as `ruff format` would.)

**Done when:** Check A; Check B with both names; a probe: for each count family, a 2-trial study
whose objective calls `Model1Run.suggest(...)` and returns 0.0; then
`run == Model1Run.suggest(FixedTrial(study.trials[0].params), ...)` is `True`, and
`json.dumps(dataclasses.asdict(run))` works; the same for one `Model2Run` pair.

### [ ] T1.5 `cv_score` and `tune_model`

One cell.
```python
def cv_score(model, X, Y, exposure, cv, cv_groups, metric):
    """The mean fold score of ``model`` refit on each fold's training rows (``fit`` replaces all state)."""
    scores = []
    for train_index, val_index in cv.split(X, Y, cv_groups):
        model.fit(
            X.iloc[train_index], Y.iloc[train_index], exposure=exposure.iloc[train_index]
        )
        pred = model.predict(X.iloc[val_index], exposure=exposure.iloc[val_index])
        scores.append(model.evaluate(Y.iloc[val_index], pred, metric))
    # The folds differ by one row at most, so the plain mean.
    return float(np.mean(scores))


def tune_model(suggest, build, X, Y, exposure, cv, cv_groups, metric, *, study_seed, n_trials):
    """One study of ``build(suggest(trial))`` on the rows of ``X``; the study and its seconds.

    Runs in a joblib worker, a fresh process at Optuna's default INFO
    verbosity, so it is set here too. A trial whose fit raises RuntimeError
    (an NB2 fit that does not converge) is marked FAIL and the study goes on;
    without ``catch`` one such trial would end the study.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Optuna passes the trial alone; the data is bound here, not to a cell.
        return cv_score(build(suggest(trial)), X, Y, exposure, cv, cv_groups, metric)

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
`poisson_glm` on the notebook's training data and prints `study.best_value` and the seconds.

### [ ] T1.6 Model 1's button and run cell

Two cells, the button unchanged:
```python
run_model1 = mo.ui.run_button(label="Tune Model 1")
mo.md(f"{run_model1} {len(COUNT_FAMILIES)} studies × `n_trials` trials, in parallel.")
```
```python
mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
# One process per family; study i gets seed + i.
model1_runs = Parallel(n_jobs=len(COUNT_FAMILIES))(
    delayed(tune_model)(
        partial(
            Model1Run.suggest,
            family=family,
            cohorts=COHORTS,
            tree_variants=TREE_VARIANTS,
            glm_variants=COUNT_GLM_VARIANTS,
        ),
        partial(
            Model1Run.build,
            feature_variants=FEATURE_VARIANTS,
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
    for i, family in enumerate(COUNT_FAMILIES)
)
MODEL1_STUDIES = dict(zip(COUNT_FAMILIES, model1_runs, strict=True))
```
`MODEL1_STUDIES` maps a family to `(optuna.Study, seconds)`.

**Done when:** Check A fully green (the `tune_model` rule holds again); Check C with
`n_trials = 3` prints `MODEL1_STUDIES True` and, for each family, `len(study.trials) == 3`;
record the wall time of Check C.

### [ ] T1.7 Refit, tables and the Model 1 views

Three cells: the functions, the Model 1 score cell, the Model 1 display cell.
```python
def refit_best(study, suggest, build, train_age_group, val_age_group, cohort_metric, metric):
    """Rebuild the study's best run, refit it on the training part; the model, its record and its scores.

    The run is rebuilt by re-running ``suggest`` on a FixedTrial of the best
    parameters: the same code path as the search, and a missing or foreign
    value raises. The record is ``dataclasses.asdict`` of the run, JSON for a
    later logging stop. The scores: the per-cohort metric on each cohort
    column and the composite metric, on training and validation.
    """
    run = suggest(FixedTrial(study.best_trial.params))
    model = build(run).fit(
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
    return model, dataclasses.asdict(run), rows


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
MODEL1_MODELS, MODEL1_BEST_RUNS = {}, {}
_score_rows, _choice_rows, _arm_rows = [], [], []
for _family, (_study, _seconds) in MODEL1_STUDIES.items():
    _model, _record, _rows = refit_best(
        _study,
        partial(
            Model1Run.suggest,
            family=_family,
            cohorts=COHORTS,
            tree_variants=TREE_VARIANTS,
            glm_variants=COUNT_GLM_VARIANTS,
        ),
        partial(
            Model1Run.build,
            feature_variants=FEATURE_VARIANTS,
            y_min=PREDICTION_FLOOR,
            seed=seed,
        ),
        train_age_group,
        val_age_group,
        POISSON_DEVIANCE,
        COHORT_POISSON_DEVIANCE,
    )
    MODEL1_MODELS[_family], MODEL1_BEST_RUNS[_family] = _model, _record
    _rows[-1]["seconds"] = _seconds
    _score_rows.extend({"family": _family, **row} for row in _rows)
    _choice_rows.extend({"family": _family, **row} for row in best_choices(_study))
    _arm_rows.extend({"family": _family, **row} for row in arm_counts(_study))
model1_df, model1_choices_df, model1_arms_df = (
    pd.DataFrame(_score_rows),
    pd.DataFrame(_choice_rows),
    pd.DataFrame(_arm_rows),
)
```
(If the implementer prefers a function `score_studies(studies, suggest_for, build, ...)` over
the loop, the same inputs go in the signature; the user's style accepts either. Say which.)
Display cell: the current one (`mo.vstack` of three `mo.md` code blocks), titles "Model 1:
cohort Poisson deviance (cv_best: the study's value; n_failed: trials whose fit raised)",
"Best combination per component", "Trials per arm".

**Done when:** Check A; Check C prints `model1_df` with 16 rows (4 families × 4), the choices
frame with 12 rows (4 families × 3 cohorts), no `objective` column, the arms frame
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
# pairs tuned are this list × the probability families.
MODEL2_TOTALS = ["lgbm_poisson", "lgbm_gaussian", "poisson_glm", "nb2"]
MODEL2_PAIRS = [
    (total, probability) for total in MODEL2_TOTALS for probability in PROBABILITY_FAMILIES
]
```
Button: `run_model2 = mo.ui.run_button(label="Tune Model 2")`, text
`f"{run_model2} {len(MODEL2_PAIRS)} studies × `n_trials` trials, in parallel."`.
Run cell: as T1.6 with `Parallel(n_jobs=len(MODEL2_PAIRS))`,
`partial(Model2Run.suggest, total_family=total, probability_family=probability,
tree_variants=TREE_VARIANTS, count_glm_variants=COUNT_GLM_VARIANTS,
probability_glm_variants=PROBABILITY_GLM_VARIANTS)`, `partial(Model2Run.build, ...)`,
`study_seed=seed + i`, keys `f"{total}__{probability}"` → `MODEL2_STUDIES`.
**Done when:** Check A; Check C prints `MODEL2_STUDIES True` with 8 studies of 3 trials.

### [ ] T2.2 Model 2's score and display cells, the best-per-model cell
The score cell mirrors T1.7 with the `Model2Run` partials → `MODEL2_MODELS`,
`MODEL2_BEST_RUNS`, `model2_df`, `model2_choices_df`, `model2_arms_df` (choices: components
`total` and `probability`). Display cell as Model 1's. The best-per-model cell is today's
(lines 1197–1218) unchanged.
**Done when:** Check A; Check C prints `model2_df` with 32 rows (8 × 4), the choices frame with
16 rows, `best_df` with 2 rows; the user clicks both buttons at 200 and reports the times.

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
  `rm`; the user commits the deletions).
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
  `run.build(...)`), E12, D2/D3 and the status line.
- The dataclass-vs-TypedDict comparison (below) into the handoff's decisions.

**Dataclass vs TypedDict, for the record:**

| | frozen dataclass (chosen) | TypedDict |
|---|---|---|
| JSON out | `json.dumps(asdict(run))`, recursive through the dict of components (probed) | the dict itself |
| JSON in | a constructor per class, or re-running `suggest` on a `FixedTrial` (what the notebook does) | `json.loads`; nothing checks the keys |
| Immutability | frozen; a scored run cannot be edited | a plain dict |
| Equality | by value (`run == rebuilt` is the T1.4 check) | by value |
| The union field `params` | a typed attribute; `isinstance` dispatch in `build` | a union of TypedDicts; `isinstance` does not work on them, dispatch needs the `family` key |
| Behaviour | `suggest`/`build` on the class | functions only |

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
  a shared storage. The machine has 8 cores (6 performance); Model 2 already runs up to 8 studies at
  once, so per-fold parallelism competes with it.

## 8. End-to-end verification (after T2, again after T4)

1. `uv run pytest -m "not slow"` green; the notebook's script run fits nothing (≈ 5 s).
2. Check C with both buttons patched at `n_trials = 3`: every table renders;
   `json.dumps` of `MODEL1_BEST_RUNS` and `MODEL2_BEST_RUNS` works.
3. The user clicks both buttons at 200 in `marimo edit`; wall times recorded in the handoff.
