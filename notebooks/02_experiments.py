import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import sys

    import marimo as mo
    import pandas as pd

    # Python 3.13 on macOS skips the venv's hidden .pth file that points at src.
    # Every cell depends on `mo`, so this runs before any project import.
    _src = str(mo.notebook_dir().parent / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Age Group Prediction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Data generation

    A synthetic population of buildings from `configs/simulation.toml`, with
    the overrides below. The simulator's own seed fixes the population; leave
    `simulation_seed` as `None` to keep the config's value.
    """)
    return


@app.cell
def _():
    import hashlib

    from sklearn.base import clone

    from age_group_prediction.feature_engineering import (
        Center,
        CenterByReferencePoint,
        ColumnPlan,
        DomainScale,
        FeatureTransformer,
        Interaction,
        OneHot,
        Quadratic,
        RelativeSaturation,
        Standardize,
    )
    from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
    from age_group_prediction.splitting import Splitter
    from student_simulator import (
        ApartmentSimulator,
        BuildingSimulator,
        CohortCompositionSimulator,
        NeighborhoodSimulator,
        StudentPopulationSimulator,
        TotalChildrenSimulator,
        load_simulation_config,
    )

    return (
        ApartmentSimulator,
        BuildingSimulator,
        Center,
        CenterByReferencePoint,
        CohortCompositionSimulator,
        ColumnPlan,
        DomainScale,
        ExposureTransformer,
        FeatureTransformer,
        Interaction,
        NeighborhoodSimulator,
        OneHot,
        Quadratic,
        RelativeSaturation,
        ShareTransformer,
        Splitter,
        Standardize,
        StudentPopulationSimulator,
        TotalChildrenSimulator,
        hashlib,
        load_simulation_config,
    )


@app.cell
def _(load_simulation_config, mo):
    project_root = mo.notebook_dir().parent
    config_path = project_root / "configs" / "simulation.toml"
    n_neighborhoods = 150  # the config has 60
    buildings_per_neighborhood_rate = 9.0
    simulation_seed = None  # None: the config's seed; an int overrides it

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

    DATA_SETTINGS
    return DATA_SETTINGS, config


@app.cell
def _(
    ApartmentSimulator,
    BuildingSimulator,
    CohortCompositionSimulator,
    NeighborhoodSimulator,
    ShareTransformer,
    StudentPopulationSimulator,
    TotalChildrenSimulator,
    config,
    mo,
):
    simulator = StudentPopulationSimulator(
        NeighborhoodSimulator(config.simulation, config.neighborhood),
        BuildingSimulator(config.simulation, config.building, config.transforms),
        ApartmentSimulator(config.simulation, config.building.room_mix),
        TotalChildrenSimulator(
            config.simulation, config.total_children, config.transforms
        ),
        CohortCompositionSimulator(
            config.simulation, config.cohort, config.transforms, config.total_children
        ),
        n_neighborhoods=config.simulation.n_neighborhoods,
        seed=config.simulation.seed,
    )
    simulation_res = simulator.simulate()
    df = simulation_res.dataset
    share_cols = ("3_rooms", "4_rooms", "5_rooms", "6_rooms")
    share_transformer = ShareTransformer(
        share_cols, reference_column="3_rooms"
    )
    df = share_transformer.fit_transform(df)
    mo.md(f"```text\n{df.head().to_string(index=False)}\n```")
    return (df,)


@app.cell
def _(ExposureTransformer, df, hashlib):
    # Compute a hash of the dataset for reproducibility.
    data_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()

    targets = ["n_kindergarten", "n_elementary", "n_highschool"]
    exposure_col = "n_apartments"
    groups_col = "neighborhood_id"
    id_col = "building_id"
    X, Y, groups = df.drop(columns=[id_col, groups_col, *targets]), df[targets], df[groups_col]
    # Checks every exposure is finite and > 0; the models leave that to it.
    exposure = ExposureTransformer(exposure_col).fit_transform(df)

    print(X.shape, Y.shape, groups.shape, exposure.shape)
    return X, Y, data_hash, exposure, groups, id_col


@app.cell
def _():
    return


@app.cell
def _(DATA_SETTINGS, X, config, data_hash, df, groups, mo, pd):
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
    print(f"data rows={len(df)} neighborhoods={groups.nunique()} data_hash={data_hash}")
    mo.vstack(
        [
            mo.md(f"```text\n{pd.DataFrame([DATA_SETTINGS]).to_string(index=False)}\n```"),
            mo.md(
                f"**{len(df):,} buildings** in {groups.nunique()} neighborhoods; "
                f"{len(config_params)} configuration parameters."
            ),
            mo.md(f"```text\ndata hash: {data_hash}\n```"),
            mo.md(f"```text\n{X.head().to_string(index=False)}\n```"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Experiment settings

    Every value below is logged as a parameter of every run. One `seed` fixes
    every random component: both holdouts, the cross-validation folds, the
    Optuna sampler (the i-th study of a run gets `seed + i`), every estimator
    that takes a `random_state`, and the bootstrap.
    """)
    return


@app.cell
def _(mo, pd):
    seed = 42
    split_method = "random"  # "random", "stratified_by_group" or "grouped"
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
    mo.md(f"```text\n{pd.DataFrame([EXPERIMENT_SETTINGS]).to_string(index=False)}\n```")
    return n_splits, seed, split_method, test_size, val_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Train, validation and test split

    Two holdouts with one `Splitter`: the test part from the table, then the
    validation part from the rest (`val_size / (1 - test_size)` of it, so
    `val_size` of the table). Each call returns positions in the frame it was
    given, so the second call's positions are inside the remainder;
    `train_val_index[...]` maps them back to table positions, and every part
    is taken by position with `.iloc`. The training part's folds come from
    the same splitter, so a run cannot pair one method's split with another's
    folds.
    """)
    return


@app.cell
def _(
    Splitter,
    X,
    Y,
    df,
    exposure,
    groups,
    id_col,
    seed,
    split_method,
    test_size,
    val_size,
):
    splitter = Splitter(split_method)
    split_groups = None if split_method == "random" else groups  # random reads none
    train_val_index, test_index = splitter.train_test_indices(
        df, split_groups, test_size=test_size, random_state=seed
    )
    groups_train_val = (
        None if split_groups is None else split_groups.iloc[train_val_index]
    )
    train_sub, val_sub = splitter.train_test_indices(
        df.iloc[train_val_index],
        groups_train_val,
        test_size=val_size / (1 - test_size),
        random_state=seed,
    )
    # The second call returns positions inside the remainder: map them back
    # to table positions.
    train_index, val_index = train_val_index[train_sub], train_val_index[val_sub]
    # training, validation, and test splits
    X_train, Y_train, exposure_train, groups_train = (
        X.iloc[train_index],
        Y.iloc[train_index],
        exposure.iloc[train_index],
        groups.iloc[train_index],
    )
    X_val, Y_val, exposure_val, groups_val = (
        X.iloc[val_index],
        Y.iloc[val_index],
        exposure.iloc[val_index],
        groups.iloc[val_index],
    )
    X_test, Y_test, exposure_test, groups_test = (
        X.iloc[test_index],
        Y.iloc[test_index],
        exposure.iloc[test_index],
        groups.iloc[test_index],
    )

    building_id_train, building_id_val, building_id_test = (
        df[id_col].iloc[train_index],
        df[id_col].iloc[val_index],
        df[id_col].iloc[test_index],
    )
    print(
        f"split n_train={len(train_index)} n_val={len(val_index)} "
        f"n_test={len(test_index)}"
    )
    return split_groups, splitter, train_index


@app.cell
def _(n_splits, seed, split_groups, splitter, train_index):
    cv = splitter.cv(n_splits=n_splits, random_state=seed)
    # What cv.split takes: the training rows' groups, or None under random,
    # since KFold warns when given groups (an error under -W error).
    cv_groups = None if split_groups is None else split_groups.iloc[train_index]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Feature variants

    The declarations of `docs/FEATURE_TRANSFORMATIONS.md` §8. `n_apartments`
    is a feature *and* the offset for the trees (§3.3: the offset asserts
    proportionality, the feature lets the trees learn departures from it),
    and only the exposure for the GLM stages (§4.4), so no GLM declaration
    lists it. `FeatureTransformer` does not enforce that; the declarations
    do, and the check below prints it. The interactions are §8.5's candidates
    (the first choice per stage, §6.2-§6.3) plus §6's second choice per stage
    and §6.5's recommended pair per stage. Each variant is fitted once on the
    training part as a check; every model fits its own copy.
    """)
    return


@app.cell
def _(
    Center,
    CenterByReferencePoint,
    ColumnPlan,
    DomainScale,
    OneHot,
    Quadratic,
    RelativeSaturation,
    Standardize,
):
    # §8.0: the 3-room share is the omitted reference.
    ROOM_SHARES = ("4_rooms_share", "5_rooms_share", "6_rooms_share")
    school_transformer = OneHot(
        categories=("none", "existing", "planned"), reference_category="none"
    )
    # §8.2: the plans of total_base, named once for the variants below.
    ses_plan = ColumnPlan(name="ses", columns="ses", transforms=(Standardize(),))
    household_plan = ColumnPlan(
        name="household", columns="avg_household_size", transforms=(Standardize(),)
    )
    median_age_plan = ColumnPlan(
        name="median_age", columns="median_age", transforms=(Standardize(),)
    )
    room_share_plan = ColumnPlan(
        name="room_share",
        columns=ROOM_SHARES,
        transforms=(Center(), DomainScale(scale=0.1)),
    )
    daycare_plan = ColumnPlan(
        name="daycare", columns="n_daycares_500m", transforms=(DomainScale(scale=2.0),)
    )
    school_plan = ColumnPlan(
        name="school", columns="school_status", transforms=(school_transformer,)
    )

    ses_squared_plan = ColumnPlan(
        name="ses_sq",
        columns="ses",
        transforms=(CenterByReferencePoint(reference_point=0.0), Quadratic()),
    )
    # §8.5.2: replaces daycare_plan; the column becomes n_daycares_500m_sat.
    daycare_saturation_plan = ColumnPlan(
        name="daycare_sat",
        columns="n_daycares_500m",
        transforms=(RelativeSaturation(),),
    )
    # §8.5.4: replaces daycare_plan when daycare is an interaction operand.
    # Centered, each product barely correlates with its room share (-0.04 to
    # 0.06 on the training part; 0.85 to 0.86 with the uncentered d/2).
    daycare_centered_plan = ColumnPlan(
        name="daycare_centered",
        columns="n_daycares_500m",
        transforms=(Center(), DomainScale(scale=2.0)),
    )
    return (
        ROOM_SHARES,
        daycare_centered_plan,
        daycare_plan,
        daycare_saturation_plan,
        household_plan,
        median_age_plan,
        room_share_plan,
        school_plan,
        ses_plan,
        ses_squared_plan,
    )


@app.cell
def _(
    ColumnPlan,
    FeatureTransformer,
    Interaction,
    ROOM_SHARES,
    daycare_centered_plan,
    daycare_plan,
    daycare_saturation_plan,
    household_plan,
    median_age_plan,
    room_share_plan,
    school_plan,
    ses_plan,
    ses_squared_plan,
):
    # §8.1: trees take the numerics untouched, n_apartments among them.
    tree_numeric_plan = ColumnPlan(
        name="numeric",
        columns=(
            *ROOM_SHARES,
            "ses",
            "avg_household_size",
            "median_age",
            "n_daycares_500m",
            "n_apartments",
        ),
    )
    tree = FeatureTransformer(plans=(tree_numeric_plan, school_plan))
    # §8.1's SES-quadratic variant.
    tree__ses_quadratic = FeatureTransformer(
        plans=(tree_numeric_plan, ses_squared_plan, school_plan)
    )

    # §8.2: n_apartments is the exposure only, never a plan.
    total_base = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_plan,
            school_plan,
        ),
    )
    # §8.5: candidates 2-4, each changing the total stage once.
    total__ses_quadratic = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_plan,
            school_plan,
            ses_squared_plan,
        ),
    )
    total__daycare_log1p = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_saturation_plan,
            school_plan,
        ),
    )
    total__room_share_x_household_size = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_plan,
            school_plan,
        ),
        interactions=tuple(
            Interaction(left=share, right="avg_household_size") for share in ROOM_SHARES
        ),
    )
    # §6.2: the second choice; §6.5: the total-stage pair.
    total__room_share_x_ses = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_plan,
            school_plan,
        ),
        interactions=tuple(
            Interaction(left=share, right="ses") for share in ROOM_SHARES
        ),
    )
    total__room_share_x_household_size_and_ses = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_plan,
            school_plan,
        ),
        interactions=(
            *(
                Interaction(left=share, right="avg_household_size")
                for share in ROOM_SHARES
            ),
            *(Interaction(left=share, right="ses") for share in ROOM_SHARES),
        ),
    )

    # §8.3: the same plans as total_base; the model takes no exposure.
    cohort_probability_base = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_plan,
            school_plan,
        ),
    )
    # §8.5: candidates 5-6, each changing the composition stage once.
    composition__daycare_log1p = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_saturation_plan,
            school_plan,
        ),
    )
    composition__room_share_x_median_age = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_plan,
            school_plan,
        ),
        interactions=tuple(
            Interaction(left=share, right="median_age") for share in ROOM_SHARES
        ),
    )
    # §6.3: the second choice; §6.5: the composition-stage pair. Daycare is
    # centered as an operand (§8.5.4).
    composition__room_share_x_daycare = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_centered_plan,
            school_plan,
        ),
        interactions=tuple(
            Interaction(left=share, right="n_daycares_500m") for share in ROOM_SHARES
        ),
    )
    composition__room_share_x_median_age_and_daycare = FeatureTransformer(
        plans=(
            ses_plan,
            household_plan,
            median_age_plan,
            room_share_plan,
            daycare_centered_plan,
            school_plan,
        ),
        interactions=(
            *(Interaction(left=share, right="median_age") for share in ROOM_SHARES),
            *(
                Interaction(left=share, right="n_daycares_500m")
                for share in ROOM_SHARES
            ),
        ),
    )

    FEATURE_VARIANTS = {
        "tree": tree,
        "tree__ses_quadratic": tree__ses_quadratic,
        "total_base": total_base,
        "total__ses_quadratic": total__ses_quadratic,
        "total__daycare_log1p": total__daycare_log1p,
        "total__room_share_x_household_size": total__room_share_x_household_size,
        "total__room_share_x_ses": total__room_share_x_ses,
        "total__room_share_x_household_size_and_ses": (
            total__room_share_x_household_size_and_ses
        ),
        "cohort_probability_base": cohort_probability_base,
        "composition__daycare_log1p": composition__daycare_log1p,
        "composition__room_share_x_median_age": composition__room_share_x_median_age,
        "composition__room_share_x_daycare": composition__room_share_x_daycare,
        "composition__room_share_x_median_age_and_daycare": (
            composition__room_share_x_median_age_and_daycare
        ),
    }
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
