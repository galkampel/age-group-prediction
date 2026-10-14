import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Age Group Prediction
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Data generation

    A synthetic population of buildings from `configs/simulation.toml`, with
    the overrides below. The simulator's own seed fixes the population; leave
    `simulation_seed` as `None` to keep the config's value.
    """)


@app.cell
def _(mo):
    project_root = mo.notebook_dir().parent
    config_path = project_root / "configs" / "simulation.toml"
    n_neighborhoods = 150  # the config has 60
    buildings_per_neighborhood_rate = 9.0
    simulation_seed = None  # None: the config's seed; an int overrides it

    return (
        buildings_per_neighborhood_rate,
        config_path,
        n_neighborhoods,
        project_root,
        simulation_seed,
    )


@app.cell
def _(
    buildings_per_neighborhood_rate,
    config_path,
    mo,
    n_neighborhoods,
    pd,
    project_root,
    simulation_seed,
):
    import hashlib

    from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
    from student_simulator import StudentPopulationSimulator, load_simulation_config

    base_config = load_simulation_config(config_path)
    config = base_config.model_copy(
        update={
            "simulation": base_config.simulation.model_copy(
                update={
                    "n_neighborhoods": n_neighborhoods,
                    **({} if simulation_seed is None else {"seed": simulation_seed}),
                }
            ),
            "building": base_config.building.model_copy(
                update={
                    "buildings_per_neighborhood_rate": buildings_per_neighborhood_rate
                }
            ),
        }
    )
    # The effective values, read back from the config: what every run logs.
    DATA_SETTINGS = {
        "config_path": str(config_path.relative_to(project_root)),
        "n_neighborhoods": config.simulation.n_neighborhoods,
        "buildings_per_neighborhood_rate": config.building.buildings_per_neighborhood_rate,
        "simulation_seed": config.simulation.seed,
    }

    # FEATURE_TRANSFORMATIONS.md §2: why the 3-room share is the omitted one.
    table = ShareTransformer(
        ("3_rooms", "4_rooms", "5_rooms", "6_rooms"), reference_column="3_rooms"
    ).fit_transform(
        StudentPopulationSimulator.from_config(config).create_dataset().dataset
    )
    COHORTS = ["n_kindergarten", "n_elementary", "n_highschool"]
    RAW_COLUMNS = [
        "4_rooms_share",
        "5_rooms_share",
        "6_rooms_share",
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "n_apartments",
        "school_status",
    ]
    X, Y = table[RAW_COLUMNS], table[COHORTS]
    # The exposure on the full table; every part is taken from it by position.
    exposure = ExposureTransformer("n_apartments").fit_transform(table)
    groups = table["neighborhood_id"]
    data_hash = hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()

    def flatten(mapping: dict, prefix: str = "") -> dict:
        """``{"a": {"b": 1}}`` to ``{"a.b": 1}``: MLflow parameters are flat."""
        flat = {}
        for key, value in mapping.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                flat.update(flatten(value, f"{name}."))
            else:
                flat[name] = value
        return flat

    config_params = flatten(config.model_dump())

    print("data_settings " + " ".join(f"{k}={v}" for k, v in DATA_SETTINGS.items()))
    print(
        f"data rows={len(table)} neighborhoods={groups.nunique()} data_hash={data_hash}"
    )
    mo.vstack(
        [
            mo.ui.table(pd.DataFrame([DATA_SETTINGS]), selection=None),
            mo.md(
                f"**{len(table):,} buildings** in {groups.nunique()} neighborhoods; "
                f"data hash `{data_hash}`; {len(config_params)} configuration parameters."
            ),
            mo.ui.table(X.head(), selection=None),
        ]
    )
    return (
        COHORTS,
        DATA_SETTINGS,
        RAW_COLUMNS,
        X,
        Y,
        config,
        config_params,
        data_hash,
        exposure,
        groups,
        table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Experiment settings

    Every value below is logged as a parameter of every run. One `seed` fixes
    every random component: both holdouts, the cross-validation folds, the
    Optuna sampler (the i-th study of a run gets `seed + i`), every estimator
    that takes a `random_state`, and the bootstrap.
    """)


@app.cell
def _(mo, pd):
    seed = 42
    split_method = "grouped"  # "random", "stratified_by_group" or "grouped"
    test_size = 0.2  # share of the whole table
    val_size = 0.2  # share of the whole table, so 0.25 of the remainder
    n_splits = 5
    n_trials = 30
    n_resamples = 1000

    # What every run logs.
    EXPERIMENT_SETTINGS = {
        "seed": seed,
        "split_method": split_method,
        "test_size": test_size,
        "val_size": val_size,
        "n_splits": n_splits,
        "n_trials": n_trials,
        "n_resamples": n_resamples,
    }
    print(
        "experiment_settings "
        + " ".join(f"{k}={v}" for k, v in EXPERIMENT_SETTINGS.items())
    )
    mo.ui.table(pd.DataFrame([EXPERIMENT_SETTINGS]), selection=None)
    return (
        EXPERIMENT_SETTINGS,
        n_resamples,
        n_splits,
        n_trials,
        seed,
        split_method,
        test_size,
        val_size,
    )


if __name__ == "__main__":
    app.run()
