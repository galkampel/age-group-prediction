# Age Group Prediction

A configurable synthetic-data project for estimating resident children in new
residential buildings by age cohort.

## Current Simulator

The simulator generates static apartment-level outcomes and returns one final
building-level DataFrame:

- [Simplified model specification](docs/SIMPLIFIED_MODEL_PLAN.md) defines the
	current behavior and formulas.
- [Compact simulator migration plan](docs/COMPACT_SIMULATOR_MIGRATION_PLAN.md)
  records the compact component migration and its validation gates.
- [Marimo research workflow migration plan](docs/MARIMO_MIGRATION_PLAN.md)
	records the completed Jupyter cutover, package setup, notebook structure,
	validation gates, and model assignments.
- [EDA and predictive modeling plan](docs/EDA_AND_PREDICTIVE_MODELING_PLAN.md)
	defines the immediate EDA deliverable and modeling handoff.
- [Documentation index](docs/README.md) distinguishes current documents from
	advanced references and legacy source material.

The compact implementation has five generation components:
`NeighborhoodSimulator`, `BuildingSimulator`, `ApartmentSimulator`,
`TotalChildrenSimulator`, and `CohortCompositionSimulator`.
`StudentPopulationSimulator(config).run()` returns the final building-level
table and retains named intermediate tables for research inspection through
`simulator.last_result`. Random effects are not exported as DataFrame columns.

```python
from student_simulator import StudentPopulationSimulator, load_simulation_config

config = load_simulation_config("configs/stage1.toml")
simulator = StudentPopulationSimulator(config)
final_df = simulator.run()
details = simulator.last_result
assert details is not None
```

## Research Workflow

The research source is a pair of version-controlled marimo notebooks:

- `notebooks/01_eda.py` implements the complete EDA checklist and findings
	handoff;
- `notebooks/02_model_fitting.py` is created only after the EDA gate passes.

The Jupyter-to-marimo cutover is complete. `research.ipynb` has been removed;
marimo `.py` notebooks are the sole research source.

Run the fast suite with:

```bash
uv run pytest -m "not calibration and not slow"
```

Run coefficient recovery after installing the validation dependency group:

```bash
uv sync --group validation
uv run pytest -m slow tests/validation/test_recovery.py -v
```