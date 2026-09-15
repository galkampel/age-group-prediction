# Marimo Research Workflow Migration Plan

> **Status: Steps 1-10 complete.** This plan migrated the research workflow from Jupyter
> to marimo without changing simulator behavior. `research.ipynb` and `ipykernel`
> have been removed. The EDA and modeling notebooks are both implemented.

The statistical simulator contract remains governed by
[SIMPLIFIED_MODEL_PLAN.md](SIMPLIFIED_MODEL_PLAN.md). The compact package
boundary remains governed by
[COMPACT_SIMULATOR_MIGRATION_PLAN.md](COMPACT_SIMULATOR_MIGRATION_PLAN.md).
The required EDA and modeling work remains governed by
[EDA_AND_PREDICTIVE_MODELING_PLAN.md](EDA_AND_PREDICTIVE_MODELING_PLAN.md) and
[MODELING_REBUILD_PLAN.md](MODELING_REBUILD_PLAN.md).

## 1. Goal And Decisions

Replace `research.ipynb` with version-controlled marimo Python notebooks that
can be edited reliably by coding models in VS Code and executed reactively in
marimo.

Use these decisions throughout the migration:

1. Make `student_simulator` an installable `src`-layout package. Notebooks must
   not modify `sys.path`, unload modules, or depend on an editor-specific
   working directory.
2. Use two notebooks with different lifecycle gates:
   - `notebooks/01_eda.py` for generation, audit, EDA, and the findings handoff;
   - `notebooks/02_model_fitting.py` only after the EDA gate passes.
3. Treat marimo `.py` files as the only notebook source of truth. Retain
   `research.ipynb` only during parity validation, then delete it.
4. Let Terra, Sol, Codex, Sonnet, or Opus edit normal Python source in VS Code.
   Use marimo to execute cells, inspect values, and render outputs. Built-in
   marimo AI and `marimo pair` are optional, not migration requirements.
5. Keep notebooks thin. Move stable, reusable, or test-worthy data splitting,
   preprocessing, metrics, model fitting, reconciliation, and serialization
   into tested modules under `src/` as complexity is introduced.
6. Add dependencies only at the gate that first needs them. Do not inflate the
   simulator's runtime environment with deferred modeling packages.

## 2. Target Structure

```text
age-group-prediction/
    notebooks/
        01_eda.py
      02_model_fitting.py        # thin modeling and experiment client
    src/
        student_simulator/         # installed compact simulator package
        age_group_prediction/      # add later for reusable modeling code
    configs/
        simulation.toml
        validation.toml
    tests/
        unit/
        validation/
        modeling/                  # add with the modeling phase
    docs/
        MARIMO_MIGRATION_PLAN.md
        EDA_AND_PREDICTIVE_MODELING_PLAN.md
      MODELING_REBUILD_PLAN.md
```

Do not create `src/age_group_prediction/` merely to hold one helper. Introduce
that package only when the first reusable modeling boundary is known. Keep
simulator behavior in `student_simulator`; do not mix prediction code into the
generator package.

## 3. Model Assignment Strategy

The names below describe capability roles. If a provider changes a model name,
select the closest model by role rather than by label.

- **Codex:** bounded implementation, conversion, tests, dependency edits,
  deletion, and repository-wide mechanical checks.
- **Sonnet or GPT-5.5:** straightforward notebook cells, documentation, plots,
  and low-risk integration work.
- **GPT-5.6 Sol:** reactive notebook architecture, multi-file integration,
  nonlinear diagnostics, and moderate statistical implementation.
- **GPT-5.6 Terra:** statistical semantics, leakage review, likelihood design,
  difficult debugging, and acceptance decisions.
- **Opus:** independent design or statistical review where a second strong
  reasoning perspective is valuable.

Use a different model for review whenever a step changes dependencies, deletes
the old notebook, establishes a data contract, or makes statistical claims.
Do not ask multiple models to edit the same notebook concurrently.

## 4. Migration Steps And Model Routing

### Step 1: Freeze The Jupyter Baseline

**Primary model: GPT-5.6 Terra**  
**Review model: Codex**

1. Execute `research.ipynb` once from a clean managed environment without
   changing simulator or notebook behavior.
2. Record the config path and contents hash, seed, final schema and shape,
   unique neighborhood count, deterministic final-table hash, quality checks,
   numerical summary tables, and plot inventory.
3. Run the current fast, calibration, and slow recovery tests.
4. Identify notebook output that is intentionally nondeterministic or only
   visually comparable.

Terra defines which statistical outputs are meaningful parity contracts.
Codex checks that the baseline record is reproducible and does not accidentally
capture private latent effects or obsolete architecture.

**Gate:** The baseline can be reproduced before any dependency or notebook
conversion change.

**Completion record (2026-08-22):** `research.ipynb` executed successfully to
a temporary notebook artifact without modifying the source notebook. The frozen
EDA parity contract is stored in
`tests/characterization/fixtures/eda_jupyter_baseline.json` and captured by
`tests/characterization/eda_baseline.py`. It records the stage-one config hash,
seed-42 EDA overrides, 1,529-by-16 final-table contract, deterministic table
hash, nine quality checks, summary-table hashes, and semantic plot inventory.
The focused parity test, canonical calibration gate, and slow recovery gate
pass in the managed environment.

### Step 2: Package The Project And Add Marimo Tooling

**Primary model: Codex**  
**Review model: GPT-5.6 Sol**

1. Add a minimal Hatchling build backend to `pyproject.toml` and explicitly
   package `src/student_simulator`.
2. Verify the documented public API imports after `uv sync` without path
   manipulation.
3. Add a `notebook` dependency group containing compatible versions of:
   - `marimo` for notebook editing and execution;
   - `nbformat` for conversion and optional interoperability;
   - `ruff` for formatting and static checks.
4. Retain `ipykernel` only until Step 8 completes parity and removes Jupyter.
5. Refresh `uv.lock` and verify the existing runtime, test, and validation
   groups independently.
6. Recommend the `marimo-team.vscode-marimo` VS Code extension. Retain the
   Microsoft Python extension. Configure a consistent workspace notebook root.
7. Ignore generated `__marimo__/` snapshots unless the project later decides
   to publish a specific HTML artifact.

Use project-level `uv` dependencies, not marimo inline or sandbox dependencies.
The notebooks depend on local source and shared project configuration, so the
repository environment is the reproducibility boundary.

**Gate:** `student_simulator` imports from the installed project in Python,
marimo CLI, and the VS Code marimo kernel with no notebook import bootstrap.

**Completion record (2026-08-22):** Added Hatchling packaging metadata and an
explicit wheel target for `src/student_simulator` in `pyproject.toml`, plus a
`notebook` dependency group containing `marimo`, `nbformat`, and `ruff`.
Refreshed `uv.lock` via `uv sync` and verified group-level imports for runtime,
test, validation, and notebook dependencies. The public API imports without
`PYTHONPATH` manipulation, and `uv run marimo --version` succeeds. Added VS
Code workspace recommendations in `.vscode/extensions.json` and a consistent
workspace notebook root in `.vscode/settings.json`.

### Step 3: Mechanically Convert The Existing Notebook

**Primary model: Codex**  
**Review model: GPT-5.6 Sol**

1. Create `notebooks/01_eda.py` with:

   ```bash
   uv run marimo convert research.ipynb -o notebooks/01_eda.py
   ```

2. Preserve the Jupyter notebook unchanged for parity comparison.
3. Remove IPython magic commands, `IPython.display` dependencies, path
   injection, module unloading, and explicit Jupyter-kernel assumptions.
4. Run `marimo check` immediately and list every duplicate global definition,
   cycle, missing dependency, or unsupported construct before refactoring.

This is only a syntax conversion. Do not add EDA features or redesign plots in
the same step.

**Gate:** The converted file parses as a marimo notebook, and all remaining
reactivity problems are explicitly identified.

**Completion record (2026-08-22):** Converted `research.ipynb` to
`notebooks/01_eda.py` with `uv run --group notebook marimo convert`.
`research.ipynb` remained unchanged (source fingerprint preserved), and
`uv run --group notebook marimo check notebooks/01_eda.py` exits successfully.
The converted notebook is syntactically valid but still contains Jupyter-era
patterns that are intentionally deferred to Step 4 refactoring:

1. Local import bootstrap via `Path.cwd()` plus `sys.path` insertion.
2. Explicit module cache clearing via `sys.modules` deletion.
3. `IPython.display.display` usage across reporting cells.
4. Explicit `plt.show()` calls.
5. Legacy notebook-style setup assumptions in one large import/setup cell.

No EDA feature redesign or reactive architecture changes were made in this
step.

### Step 4: Establish The Reactive Notebook Architecture

**Primary model: GPT-5.6 Sol**  
**Review model: Codex**

Refactor `notebooks/01_eda.py` into this dependency order:

1. purpose and methodology markdown;
2. imports and display configuration;
3. portable paths using `mo.notebook_dir()`;
4. base configuration and immutable EDA overrides;
5. compact simulator generation;
6. structural checks and leakage register;
7. distributions and correlations;
8. nonlinear and interaction analyses;
9. final findings table.

Apply these marimo rules:

- define each global variable exactly once;
- prefix cell-local temporary variables with `_`;
- use distinct names for successive DataFrame transformations;
- do not mutate a DataFrame owned by another cell;
- place repeated plotting or transformation logic in small functions;
- return figure objects instead of relying on `plt.show()`;
- display bounded table views rather than full apartment-level tables;
- use `mo.stop()` and run buttons only for optional expensive experiments.

The canonical reference path must remain autorunnable with:

```bash
uv run notebooks/01_eda.py
```

Do not put required generation or completion checks behind an interactive
button. A notebook that needs manual clicks cannot serve as a clean execution
gate.

**Gate:** `marimo check` passes, script execution succeeds, and editing one
cell only invalidates its real downstream dependents.

**Completion record (2026-08-22):** Refactored `notebooks/01_eda.py` into an
autorunnable marimo dependency graph. The notebook now uses the installed
`student_simulator` package, resolves `configs/simulation.toml` from
`mo.notebook_dir()`, keeps immutable EDA overrides in their own cell, and
uses underscore-prefixed cell-local intermediates. Jupyter-era `sys.path`,
module-cache, `IPython.display`, `Path.cwd()`, and `plt.show()` patterns were
removed. Analysis cells now return bounded marimo layouts and Matplotlib
figure objects, and the implemented EDA evidence feeds a final findings
table.

The following gates pass:

```bash
uv run --group notebook ruff check notebooks/01_eda.py
uv run --group notebook marimo check notebooks/01_eda.py
uv run notebooks/01_eda.py
uv run --group test python -m pytest \
   tests/characterization/test_eda_jupyter_baseline.py -q
```

The frozen data-parity suite reports `2 passed`. Static marimo dependency
validation confirms that cell outputs have unique owners and no dependency
cycles. The local macOS environment also required clearing an inherited
filesystem `hidden` flag from Hatchling's generated editable-install `.pth`
file; that was an environment repair rather than a repository path workaround.

### Step 5: Reproduce EDA Steps 1-3

**Primary model: Sonnet or GPT-5.5**  
**Review model: GPT-5.6 Terra**

1. Reproduce the immutable `150`-neighborhood and `9.0` building-rate EDA
   overrides without editing `configs/simulation.toml`.
2. Record seed, config path/content/hash, row count, neighborhood count, and
   buildings-per-neighborhood distribution.
3. Reproduce the structural audit, leakage register, target and feature
   distributions, cohort-share handling, and Pearson/Spearman analysis.
4. Preserve both exact accounting identities and explicitly document the
   room-count collinearity constraint.
5. Compare every deterministic output with the Step 1 baseline.

Sonnet handles the bounded pandas and plotting implementation. Terra reviews
leakage classification, zero-total handling, overdispersion claims, and the
distinction between descriptive association and causal evidence.

**Gate:** Every completion check for EDA Steps 1-3 passes with exact data
parity and semantically equivalent plots.

**Completion record (2026-08-22):** The EDA notebook focuses only on
configuration, population generation, quality audits, distributions, and
correlation analysis. Exact parity with the frozen Jupyter reference remains
an external regression safeguard in
`tests/characterization/test_eda_jupyter_baseline.py`; it recomputes the
config hash/overrides, final-dataframe hash/shape/dtypes, the full nine-check
quality contract, and the buildings-per-neighborhood, target-summary,
cohort-share, and correlation hashes against
`tests/characterization/fixtures/eda_jupyter_baseline.json`. Sections are
numbered `Nonlinear And Interaction Analyses` 7 and `Current Findings` 8.

The following gates pass:

```bash
uv run --group notebook marimo check notebooks/01_eda.py
uv run notebooks/01_eda.py
uv run --group notebook ruff check notebooks/01_eda.py
uv run --group test python -m pytest \
  tests/characterization/test_eda_jupyter_baseline.py -q
```

Canonical script execution completes successfully, while the frozen pytest
suite independently confirms exact data parity (`2 passed`).

### Step 6: Complete EDA Steps 4-6

**Primary model: GPT-5.6 Sol**  
**Review model: GPT-5.6 Terra or Opus**

1. Add raw, quantile-binned, and smoothed marginal views with visible support
   and uncertainty information.
2. Use the existing `HistGradientBoostingRegressor(loss="poisson")` for
   exploratory conditional views. Do not add LightGBM during this migration.
3. Add PD and ICE views, and ALE only when it can be implemented and validated
   without an unnecessary dependency.
4. Add interaction facets and sparse-cell-aware heatmaps.
5. Finish the EDA findings table, including evidence, unresolved hypotheses,
   unsupported regions, room representation, overdispersion, and the selected
   deployment question.

Sol implements the reactive visualization and diagnostic flow. Terra or Opus
reviews support limitations, correlated-feature interpretation, model-dependent
plots, and whether the findings actually satisfy the modeling preconditions.

**Gate:** All six EDA completion checks pass, and an independent reviewer
approves the findings table before modeling begins.

### Step 7: Validate Marimo Parity And Repository Safety

**Primary model: Codex**  
**Review model: GPT-5.6 Terra**

1. Run `marimo check` and clean script execution.
2. Open the notebook through both `uv run marimo edit` and the VS Code marimo
   extension; run all stale cells and inspect outputs.
3. Compare config hash, seed, schema, shape, deterministic table hash, quality
   checks, numerical summaries, and plot inventory with the Jupyter baseline.
4. Require exact parity for deterministic data and semantic parity for plots;
   do not require pixel-identical rendering.
5. Run full simulator tests, calibration, slow recovery, source compilation,
   and public import checks.
6. Search for stale Jupyter imports, path workarounds, manual `df_all` merge
   chains, private latent-effect exposure, and removed compact-framework names.

Codex executes the repository-wide checks. Terra decides whether any numerical
or statistical difference is acceptable; undocumented drift blocks cutover.

**Gate:** All executable checks pass and every parity difference is either
removed or documented and approved.

### Step 8: Cut Over Documentation And Remove Jupyter

**Primary model: Sonnet or GPT-5.5**  
**Review model: Codex**

1. Delete `research.ipynb` only after Step 7 passes.
2. Remove `ipykernel` if no other active Jupyter workflow uses it, then refresh
   `uv.lock`.
3. Update root and documentation README files, the compact migration plan, the
   EDA plan, and the deferred modeling plan to use marimo paths and commands.
4. Search active files for stale `research.ipynb`, Jupyter, `nbconvert`, and
   clean-kernel instructions.
5. Do not retain a synchronized `.ipynb` export. Exported HTML or IPYNB files
   are disposable presentation artifacts, not editable sources.

**Gate:** Active documentation has one research workflow and all linked files
and commands resolve.

**Completion record (2026-08-23):** Deleted `research.ipynb` and removed the
core `ipykernel` dependency from `pyproject.toml` after Step 7's parity gate
passed: `marimo check`, clean script execution, the full fast test suite,
calibration, slow recovery, and the frozen Jupyter-parity regression all
passed. `ipykernel` was unused outside the removed notebook, so no source or
test code required changes. Refreshed `uv.lock` via `uv sync`. Updated the
root `README.md`, `docs/README.md`, and
`EDA_AND_PREDICTIVE_MODELING_PLAN.md` to describe the completed cutover and
removed stale `research.ipynb` retention language. A search of active files
confirmed no remaining stale `research.ipynb`, Jupyter, `nbconvert`, or
clean-kernel instructions outside historical completion records and the
frozen parity-test naming, which intentionally still reference the Jupyter
baseline contract.

### Step 9: Establish The Deferred Modeling Notebook

**Primary model: GPT-5.6 Sol**  
**Review model: GPT-5.6 Terra**

Begin only after Step 6's EDA handoff is approved.

1. Create `notebooks/02_model_fitting.py` around the accepted gates in
   `MODELING_REBUILD_PLAN.md`.
2. Add `src/age_group_prediction/` when the first reusable preprocessing or
   model contract is ready, not before.
3. Put deterministic split generation, leakage-safe preprocessing, metrics,
   count/cohort reconciliation, and model wrappers in tested source modules.
4. Keep experiment selection, narrative, diagnostics, and comparisons in the
   notebook.
5. Add a `modeling` dependency group only when baseline and frequentist model
   implementation starts.

**Gate:** Data and split tests pass before any candidate model is evaluated.

### Step 10: Add High-Complexity Modeling Incrementally

**Primary model: GPT-5.6 Terra**  
**Review model: Opus**

Terra owns the acceptance decision for this high-risk gate, and Opus provides
an independent statistical and design review. Delegate bounded implementation
work to the lower-cost models in the routing table below.

Use the routing below for the deferred implementation gates:

| Modeling work | Primary model | Review model |
|---|---|---|
| Splits, preprocessing, LightGBM direct-cohort model | GPT-5.6 Sol or Sonnet | Codex |
| Model B independent NB2 total and child-level categorical probability fits | GPT-5.6 Sol | GPT-5.6 Terra |
| Evaluation, calibration, and permutation importance | GPT-5.6 Sol | GPT-5.6 Terra or Opus |
| Bayesian NB2 + Dirichlet-Multinomial Pyro inference | GPT-5.6 Terra | Opus |
| MLflow integration and artifact logging | Sonnet or GPT-5.5 | Codex |
| Repository-wide implementation review | Codex | GPT-5.6 Opus |

The completed implementation keeps LightGBM, Pyro, Torch, scikit-learn, and
the numerical stack as core dependencies. MLflow remains optional in the
`tracking` group, statsmodels remains in `validation`, and marimo tooling is in
`notebook`.

**Gate:** Each dependency group installs and validates independently. Bayesian,
tracking, or explainability failures must not break simulator or EDA usage.

## 5. VS Code And Agent Workflow

For every model-assisted notebook edit:

1. Assign one plan step or one bounded notebook section with its completion
   check.
2. Ask the model to read the controlling plan and nearby implementation before
   editing.
3. Edit the marimo `.py` source in VS Code; do not edit a generated IPYNB.
4. Inspect the diff before execution.
5. Run `marimo check`, then execute stale or all cells and inspect rendered
   output.
6. Run the focused tests for any source module changed by the notebook work.
7. Use the designated review model before crossing the step's gate.

Avoid simultaneous unsynchronized edits in a browser editor and VS Code. The
VS Code marimo notebook view may execute and render the same source file, but
only one agent or person should write it at a time.

## 6. Dependency And Extension Plan

| Phase | Dependency group | Packages | Required |
|---|---|---|---|
| Migration and EDA | `notebook` | `marimo`, `nbformat`, `ruff` | Yes |
| Existing tests | `test` | `pytest` | Yes for validation |
| Existing recovery | `validation` | `statsmodels` | Yes for recovery gate |
| Frequentist modeling | core | `lightgbm`, `scikit-learn`, `scipy` | Yes |
| Bayesian modeling | core | `pyro-ppl`, `torch` | Yes |
| Experiment tracking | `tracking` | `mlflow` | Optional |

Do not install `marimo[recommended]` by default. The project already has
pandas, matplotlib, seaborn, and scikit-learn, and it does not currently need
the recommended bundle's SQL, Altair, Polars, or built-in AI dependencies.

Recommended VS Code extensions:

- `marimo-team.vscode-marimo` for notebook execution and rich outputs;
- `ms-python.python` for ordinary package and test development.

The Jupyter extension is not required after Step 8. Marimo's managed language
features should remain enabled initially. Disable them only if a verified need
for Pylance inside rendered marimo cells outweighs the risk of conflicting
diagnostics; Pylance continues to work normally on the `.py` source and
project modules.

## 7. Validation Commands

Use the exact commands supported by the installed marimo version; confirm CLI
help before automating a command in CI.

```bash
uv sync --group notebook --group test --group validation
uv run python -c "from student_simulator import StudentPopulationSimulator, load_simulation_config"
uv run marimo check notebooks/01_eda.py
uv run notebooks/01_eda.py
uv run pytest -m "not calibration and not slow"
uv run pytest -m calibration
uv run pytest -m slow tests/validation/test_recovery.py -v
uv run python -m compileall src/
```

Manual acceptance checks:

- open `notebooks/01_eda.py` in the VS Code marimo view;
- run all stale cells and confirm no hidden working-directory dependency;
- verify bounded DataFrame displays and readable plots;
- confirm optional controls do not trigger accidental expensive reruns;
- confirm the findings table is complete and distinguishes evidence from
  untested hypotheses.

### 7.1 Troubleshooting `student_simulator` Imports On macOS

The notebook must import `student_simulator` as an installed package. Do not
add `sys.path` manipulation to the notebook. An `import sys` statement is also
not part of the fix; `sys` is useful only when diagnosing which interpreter or
import paths a kernel is using.

#### Case 1: First-Time Environment Setup

From the repository root, synchronize the notebook environment:

```bash
uv sync --group notebook
```

Then select this interpreter for the marimo notebook in VS Code:

```text
<repository>/.venv/bin/python
```

Test the installed package with the exact environment interpreter:

```bash
.venv/bin/python -c \
   "import student_simulator; print(student_simulator.__file__)"
```

For a healthy editable installation, the printed path resolves under
`src/student_simulator/`.

#### Case 2: Diagnose `ModuleNotFoundError`

First confirm that the selected environment and editable-install metadata are
the expected ones:

```bash
.venv/bin/python -c "import sys; print(sys.executable); print(sys.prefix)"
uv pip show age-group-prediction
ls -lO \
   .venv/lib/python3.13/site-packages/_editable_impl_age_group_prediction.pth
```

On the affected macOS/Python 3.13 environment, `ls -lO` may show `hidden` on
the generated `.pth` file. Python 3.13 skips hidden `.pth` files during
startup, so the repository's `src/` directory is not added to `sys.path` and
`student_simulator` cannot be imported even though `uv pip show` reports a
valid editable installation.

#### Case 3: Repair The Editable Installation

Clear the macOS hidden flag and verify the import without asking uv to refresh
the environment:

```bash
chflags nohidden \
   .venv/lib/python3.13/site-packages/_editable_impl_age_group_prediction.pth

ls -lO \
   .venv/lib/python3.13/site-packages/_editable_impl_age_group_prediction.pth

.venv/bin/python -c \
   "import student_simulator; print(student_simulator.__file__)"
```

The `ls` output must not contain `hidden`. After this repair, restart the
marimo kernel before rerunning the imports cell: an existing kernel initialized
its `sys.path` before the flag was cleared.

#### Case 4: The Flag Returns When Using uv

In the affected local environment, a later `uv sync`, package refresh, or even
`uv run --no-sync` may recreate or re-hide the editable `.pth` file. If the
quick repair does not remain stable, replace the editable project install with
a regular wheel installation inside the existing virtual environment:

```bash
uv pip install \
   --python .venv/bin/python \
   --force-reinstall \
   --no-deps \
   .
```

This installs `student_simulator` directly under `site-packages` and removes
the runtime dependency on the editable `.pth` file. Verify the exact command
used by uv twice to ensure the fix is stable:

```bash
uv run --no-sync python -c \
   "import student_simulator; print(student_simulator.__file__)"

uv run --no-sync python -c \
   "import student_simulator; print(student_simulator.__file__)"
```

For the regular installation, the printed path resolves under
`.venv/lib/python3.13/site-packages/student_simulator/`.

The tradeoff is that source changes under `src/student_simulator/` are no
longer reflected immediately. Reinstall the local wheel after changing package
source:

```bash
uv pip install \
   --python .venv/bin/python \
   --force-reinstall \
   --no-deps \
   .
```

Avoid `uv sync` after selecting this fallback unless dependency changes require
it, because synchronization may restore the editable installation. If that
happens, either repeat the `chflags nohidden` repair or reinstall the regular
wheel.

#### Case 5: Terminal Import Works But The Notebook Still Fails

A running marimo kernel does not rebuild its startup import paths when the
environment is repaired externally. In VS Code:

1. Restart the marimo kernel.
2. Confirm that `<repository>/.venv/bin/python` remains selected.
3. Rerun the setup/imports cell.

Use this temporary diagnostic in a scratch cell only when the selected kernel
is uncertain:

```python
import sys

print(sys.executable)
print(sys.prefix)
```

Do not keep this diagnostic in the EDA imports cell. The EDA itself does not
need `sys`, and a successful import should not depend on notebook-local path
bootstrap code.

## 8. Completion Criteria

The migration is complete when:

- `student_simulator` imports as an installed package without notebook path
  manipulation;
- `notebooks/01_eda.py` passes marimo checks and clean script execution;
- all six EDA completion checks pass;
- deterministic Jupyter-to-marimo parity is established;
- simulator, calibration, and recovery tests still pass;
- `research.ipynb` and unnecessary `ipykernel` usage are removed;
- active docs and README files reference the marimo workflow;
- model packages remain deferred until the approved modeling gate; and
- Terra or Opus independently approves the EDA handoff findings.
