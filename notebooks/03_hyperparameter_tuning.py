import marimo

__generated_with = "0.25.2"
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
    # Age Group Prediction: hyperparameter tuning

    The 24 tuning runs: one Optuna study per model family and exposure, its
    best model refit, scored and logged to MLflow. `04_model_inspection.py`
    reads them, selects each model's best run, bootstraps and inspects it,
    and evaluates it on the test part.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Data generation

    The synthetic population of `02_objective_eda.py`, where it is described;
    the same code (a test pins the two notebooks' setup equal).
    """)
    return


@app.cell
def _():
    import hashlib
    import json

    from age_group_prediction.feature_engineering import (
        Center,
        CenterByReferencePoint,
        DomainScale,
        FeatureTransformer,
        Interaction,
        OneHot,
        Plan,
        Quadratic,
        RelativeSaturation,
        Standardize,
    )
    from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
    from age_group_prediction.splitting import Splitter
    from age_group_prediction.utils import AgeGroupDataset
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
        AgeGroupDataset,
        ApartmentSimulator,
        BuildingSimulator,
        Center,
        CenterByReferencePoint,
        CohortCompositionSimulator,
        DomainScale,
        ExposureTransformer,
        FeatureTransformer,
        Interaction,
        NeighborhoodSimulator,
        OneHot,
        Plan,
        Quadratic,
        RelativeSaturation,
        ShareTransformer,
        Splitter,
        Standardize,
        StudentPopulationSimulator,
        TotalChildrenSimulator,
        hashlib,
        json,
        load_simulation_config,
    )


@app.cell
def _(load_simulation_config, mo):
    project_root = mo.notebook_dir().parent
    config_path = project_root / "configs" / "simulation.yml"
    n_neighborhoods = 150  # the config has 60
    buildings_per_neighborhood_rate = 9.0
    simulation_seed = None  # None: the config's seed; an int overrides it

    base_config = load_simulation_config(config_path)
    # Validated, unlike model_copy: each value is coerced to its field's type
    # (9 to 9.0), so the config hash does not depend on how it was typed.
    config = base_config.with_overrides(
        {
            "simulation": {
                "n_neighborhoods": n_neighborhoods,
                **({} if simulation_seed is None else {"seed": simulation_seed}),
            },
            "building": {
                "buildings_per_neighborhood_rate": buildings_per_neighborhood_rate
            },
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
    return DATA_SETTINGS, config, project_root


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
def _(ExposureTransformer, df):
    targets = ["n_kindergarten", "n_elementary", "n_highschool"]
    exposure_col = "n_apartments"
    groups_col = "neighborhood_id"
    id_col = "building_id"
    X, Y = df.drop(columns=[id_col, groups_col, *targets]), df[targets]
    groups, ids = df[groups_col], df[id_col]
    # Checks every exposure is finite and > 0; the models leave that to it.
    exposure = ExposureTransformer(exposure_col).fit_transform(df)

    print(X.shape, Y.shape, groups.shape, exposure.shape)
    return X, Y, exposure, groups, ids


@app.cell
def _(DATA_SETTINGS, X, config, df, groups, hashlib, json, mo, pd):
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
    # The data's version: the effective config (overrides included). Keys
    # sorted, so the hash does not depend on the dump's order.
    config_hash = hashlib.sha256(
        json.dumps(config_params, sort_keys=True).encode()
    ).hexdigest()

    print(
        "data_settings "
        + " ".join(f"{name}={value}" for name, value in DATA_SETTINGS.items())
    )
    print(
        f"data rows={len(df)} neighborhoods={groups.nunique()} "
        f"config_hash={config_hash}"
    )
    mo.vstack(
        [
            mo.md(f"```text\n{pd.DataFrame([DATA_SETTINGS]).to_string(index=False)}\n```"),
            mo.md(
                f"**{len(df):,} buildings** in {groups.nunique()} neighborhoods; "
                f"{len(config_params)} configuration parameters."
            ),
            mo.md(f"```text\nconfig hash: {config_hash}\n```"),
            mo.md(f"```text\n{X.head().to_string(index=False)}\n```"),
        ]
    )
    return config_hash, config_params


@app.cell
def _(pd, project_root):
    import os
    import subprocess
    import warnings
    from pathlib import Path

    import mlflow

    # The store: a scratch sqlite file and an artifact folder, from the environment (.env);
    # never the canonical store at the repo root, which an unset URI would open (MLflow's
    # default is sqlite:///mlflow.db in the working directory) (D1e).
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    artifact_location = os.environ.get("AGE_GROUP_MLFLOW_ARTIFACT_LOCATION")
    if tracking_uri is None:
        raise ValueError(
            "MLFLOW_TRACKING_URI is unset: MLflow would use sqlite:///mlflow.db, the canonical store"
        )
    if not tracking_uri.startswith("sqlite:///"):
        raise ValueError(
            f"MLFLOW_TRACKING_URI must be a sqlite:/// URI, got {tracking_uri!r}"
        )
    # A relative path resolves against the working directory, as SQLAlchemy resolves it.
    if (
        Path(tracking_uri.removeprefix("sqlite:///")).resolve()
        == (project_root / "mlflow.db").resolve()
    ):
        raise ValueError(f"MLFLOW_TRACKING_URI {tracking_uri!r} is the canonical store")
    if experiment_name is None:
        raise ValueError("MLFLOW_EXPERIMENT_NAME is unset")
    if artifact_location is None:
        raise ValueError("AGE_GROUP_MLFLOW_ARTIFACT_LOCATION is unset")
    if Path(artifact_location).resolve() == (project_root / "mlartifacts").resolve():
        raise ValueError(
            f"AGE_GROUP_MLFLOW_ARTIFACT_LOCATION {artifact_location!r} is the canonical one"
        )
    # pandas 3 warns inside two MLflow calls (M0): log_table's select_dtypes on a str
    # column, and log_input's schema on an int column. Each filtered by its category,
    # message and module, so any other warning still shows.
    warnings.filterwarnings(
        "ignore",
        category=pd.errors.Pandas4Warning,
        message="For backward compatibility, 'str' dtypes are included by select_dtypes",
        module="mlflow.tracking.client",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Hint: Inferred schema contains integer",
        module="mlflow.types.utils",
    )
    # MLflow reads the two variables checked above at every call, so no global is set
    # (no set_tracking_uri, no set_experiment). The experiment is created here, with its
    # artifact location: MLflow would create it on the first run without one. An
    # experiment that exists keeps the location it was created with, which MLflow uses
    # over the variable: it must be the variable's.
    _experiment = mlflow.get_experiment_by_name(experiment_name)
    if _experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=str(Path(artifact_location).resolve())
        )
    elif _experiment.artifact_location != str(Path(artifact_location).resolve()):
        raise ValueError(
            f"the experiment {experiment_name!r} stores its artifacts in "
            f"{_experiment.artifact_location!r}, not in "
            f"AGE_GROUP_MLFLOW_ARTIFACT_LOCATION {artifact_location!r}"
        )
    # The code's version, tags of every run: the notebook, HEAD, and whether a tracked
    # file differs from it; git runs in src (in Check C's copy a link to the repo).
    code_tags = {
        "notebook": "notebooks/03_hyperparameter_tuning.py",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root / "src",
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "git_dirty": str(
            bool(
                subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=project_root / "src",
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
        ).lower(),
    }
    print(f"experiment={experiment_name} tracking_uri={tracking_uri}")
    return code_tags, mlflow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Experiment settings

    The objective EDA's settings, described in `02_objective_eda.py` (a test
    pins them equal); every tuning run logs them. `04_model_inspection.py`
    adds the settings of the selection, the bootstrap and the inspection.
    """)
    return


@app.cell
def _(mo, pd):
    seed = 42
    split_method = "random"  # "random", "stratified_by_group" or "grouped"
    test_size = 0.2  # share of the whole table
    val_size = 0.2  # share of the whole table, so 0.25 of the remainder
    n_splits = 5
    n_trials = 100
    # TPE's random start: the trials drawn at random before its model guides the
    # draws; 10% of the budget, Optuna's default (D31).
    n_startup_trials = 10
    # The tuner's objective, the composite's cross-validated score every study
    # minimizes: "rmse" or "poisson_deviance" (D42).
    objective_metric = "rmse"

    # What every run logs.
    EXPERIMENT_SETTINGS = {
        "seed": seed,
        "split_method": split_method,
        "test_size": test_size,
        "val_size": val_size,
        "n_splits": n_splits,
        "n_trials": n_trials,
        "n_startup_trials": n_startup_trials,
        "objective_metric": objective_metric,
    }
    print(
        "experiment_settings "
        + " ".join(f"{name}={value}" for name, value in EXPERIMENT_SETTINGS.items())
    )
    mo.md(f"```text\n{pd.DataFrame([EXPERIMENT_SETTINGS]).to_string(index=False)}\n```")
    return (
        EXPERIMENT_SETTINGS,
        n_splits,
        n_startup_trials,
        n_trials,
        objective_metric,
        seed,
        split_method,
        test_size,
        val_size,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Train, validation and test split

    The split and the folds of `02_objective_eda.py`, where they are
    described; the same code (a test pins the two notebooks' setup equal).
    """)
    return


@app.cell
def _(
    AgeGroupDataset,
    Splitter,
    X,
    Y,
    df,
    exposure,
    groups,
    ids,
    seed,
    split_method,
    test_size,
    val_size,
):
    splitter = Splitter(split_method)
    # random reads none
    split_groups = None if split_method == "random" else groups
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
    # One record per part, each component taken at the part's positions.
    train_age_group, val_age_group, test_age_group = (
        AgeGroupDataset(
            X=X.iloc[index],
            Y=Y.iloc[index],
            exposure=exposure.iloc[index],
            groups=groups.iloc[index],
            ids=ids.iloc[index],
        )
        for index in (train_index, val_index, test_index)
    )
    print(
        f"split n_train={len(train_index)} n_val={len(val_index)} "
        f"n_test={len(test_index)}"
    )
    return splitter, test_age_group, train_age_group, val_age_group


@app.cell
def _(n_splits, seed, split_method, splitter, train_age_group):
    cv = splitter.cv(n_splits=n_splits, random_state=seed)
    # What cv.split takes: the training rows' groups, or None under random,
    # since KFold warns when given groups (an error under -W error).
    cv_groups = None if split_method == "random" else train_age_group.groups
    return cv, cv_groups


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Feature variants

    The feature variants of `02_objective_eda.py`, where they are described
    (`docs/FEATURE_TRANSFORMATIONS.md` §8); the same declarations (a test
    pins the two notebooks' setup equal).
    """)
    return


@app.cell
def _(
    Center,
    CenterByReferencePoint,
    DomainScale,
    OneHot,
    Plan,
    Quadratic,
    RelativeSaturation,
    Standardize,
):
    # §8.0: the 3-room share is the omitted reference.
    ROOM_SHARES = ("4_rooms_share", "5_rooms_share", "6_rooms_share")
    # The treatments, named for what they do; each variant below applies them
    # to its columns. scale_by: DomainScale divides by it, so a one-unit
    # change of the output is 0.1 or 2 of the input.
    passthrough_plan = Plan(name="passthrough")  # the identity
    standardize_plan = Plan(name="standardize", transforms=(Standardize(),))
    center_scale_by_tenth_plan = Plan(
        name="center_scale_by_0.1", transforms=(Center(), DomainScale(scale=0.1))
    )
    scale_by_two_plan = Plan(name="scale_by_2", transforms=(DomainScale(scale=2.0),))
    one_hot_plan = Plan(
        name="one_hot",
        transforms=(
            OneHot(
                categories=("none", "existing", "planned"),
                reference_category="none",
            ),
        ),
    )
    square_around_zero_plan = Plan(
        name="square_around_0",
        transforms=(CenterByReferencePoint(reference_point=0.0), Quadratic()),
    )
    # §8.5.2: the column becomes <column>_sat.
    relative_saturation_plan = Plan(
        name="relative_saturation", transforms=(RelativeSaturation(),)
    )
    return (
        ROOM_SHARES,
        center_scale_by_tenth_plan,
        one_hot_plan,
        passthrough_plan,
        relative_saturation_plan,
        scale_by_two_plan,
        square_around_zero_plan,
        standardize_plan,
    )


@app.cell
def _(
    FeatureTransformer,
    Interaction,
    ROOM_SHARES,
    center_scale_by_tenth_plan,
    one_hot_plan,
    passthrough_plan,
    relative_saturation_plan,
    scale_by_two_plan,
    square_around_zero_plan,
    standardize_plan,
):
    # §8.1: trees take the numerics untouched, n_apartments among them.
    tree_numerics = [
        *ROOM_SHARES,
        "ses",
        "avg_household_size",
        "median_age",
        "n_daycares_500m",
        "n_apartments",
    ]
    tree = FeatureTransformer(
        column_plans=(
            (passthrough_plan, tree_numerics),
            (one_hot_plan, ["school_status"]),
        )
    )
    # §8.1's SES-quadratic variant.
    tree__ses_quadratic = FeatureTransformer(
        column_plans=(
            (passthrough_plan, tree_numerics),
            (square_around_zero_plan, ["ses"]),
            (one_hot_plan, ["school_status"]),
        )
    )

    # §8.2: the base of every GLM, count or probability model; n_apartments is
    # the exposure only, never a plan's column.
    glm_base = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (scale_by_two_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
    )
    # §8.5: candidates 2-4, each changing the count model's plans once.
    count__ses_quadratic = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (scale_by_two_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
            (square_around_zero_plan, ["ses"]),
        ),
    )
    count__daycare_log1p = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (relative_saturation_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
    )
    count__room_share_x_household_size = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (scale_by_two_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
        interactions=tuple(
            Interaction(left=share, right="avg_household_size") for share in ROOM_SHARES
        ),
    )
    # §6.2: the second choice; §6.5: the total-stage pair.
    count__room_share_x_ses = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (scale_by_two_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
        interactions=tuple(
            Interaction(left=share, right="ses") for share in ROOM_SHARES
        ),
    )
    count__room_share_x_household_size_and_ses = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (scale_by_two_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
        interactions=(
            *(
                Interaction(left=share, right="avg_household_size")
                for share in ROOM_SHARES
            ),
            *(Interaction(left=share, right="ses") for share in ROOM_SHARES),
        ),
    )

    # §8.3: the probability model takes the same base (§8.2) without an
    # exposure; §8.5's candidates 5-6 each change its plans once.
    probability__daycare_log1p = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (relative_saturation_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
    )
    probability__room_share_x_median_age = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (scale_by_two_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
        interactions=tuple(
            Interaction(left=share, right="median_age") for share in ROOM_SHARES
        ),
    )
    # §6.3: the second choice; §6.5: the composition-stage pair. Daycare enters
    # as d/2, uncentered, so 0 = no daycares and each share's main effect is
    # read there (§8.5.4).
    probability__room_share_x_daycare = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (scale_by_two_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
        interactions=tuple(
            Interaction(left=share, right="n_daycares_500m") for share in ROOM_SHARES
        ),
    )
    probability__room_share_x_median_age_and_daycare = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (scale_by_two_plan, ["n_daycares_500m"]),
            (one_hot_plan, ["school_status"]),
        ),
        interactions=(
            *(Interaction(left=share, right="median_age") for share in ROOM_SHARES),
            *(
                Interaction(left=share, right="n_daycares_500m")
                for share in ROOM_SHARES
            ),
        ),
    )

    # Each family's candidate feature transformers, by name, in the order a trial
    # draws them. The trees take n_apartments as a feature and the offset; the
    # GLMs take it as the exposure only (docs/FEATURE_TRANSFORMATIONS.md §3.3,
    # §4.4); the logistic model needs the scaled plans (it does not converge on
    # `tree`).
    _tree_candidates = {"tree": tree, "tree__ses_quadratic": tree__ses_quadratic}
    _glm_count_candidates = {
        "glm_base": glm_base,
        "count__ses_quadratic": count__ses_quadratic,
        "count__daycare_log1p": count__daycare_log1p,
        "count__room_share_x_household_size": count__room_share_x_household_size,
        "count__room_share_x_ses": count__room_share_x_ses,
        "count__room_share_x_household_size_and_ses": (
            count__room_share_x_household_size_and_ses
        ),
    }
    COUNT_FEATURE_TRANSFORMERS = {
        "lgbm_poisson": _tree_candidates,
        "lgbm_gaussian": _tree_candidates,
        "poisson_glm": _glm_count_candidates,
        "nb2": _glm_count_candidates,
    }
    PROBABILITY_FEATURE_TRANSFORMERS = {
        "logistic": {
            "glm_base": glm_base,
            "probability__daycare_log1p": probability__daycare_log1p,
            "probability__room_share_x_median_age": probability__room_share_x_median_age,
            "probability__room_share_x_daycare": probability__room_share_x_daycare,
            "probability__room_share_x_median_age_and_daycare": (
                probability__room_share_x_median_age_and_daycare
            ),
        },
        "lgbm": _tree_candidates,
    }
    return COUNT_FEATURE_TRANSFORMERS, PROBABILITY_FEATURE_TRANSFORMERS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Hyperparameter tuning

    One Optuna study per model family and exposure, on the composite's
    cross-validated `objective_metric` (RMSE; D42); the total times
    probability model's family is a pair, both components drawn in each trial
    (the joint method, D40; the objective EDA compares it with coordinate
    descent). Each run's best model is refit on the training part, scored on
    training and validation, and logged to MLflow as its study finishes.
    `04_model_inspection.py` reads the 24 runs and selects each model's best
    run by its validation score, then bootstraps it over whole neighborhoods,
    inspects it and evaluates it on the test part. The objective EDA
    (`02_objective_eda.py`) describes the studies' storage and their trials.
    """)
    return


@app.cell
def _():
    import dataclasses

    import numpy as np
    import optuna

    from age_group_prediction.hyperparameter_tuning import (
        CountComponent,
        CrossValidationTuner,
        ProbabilityComponent,
        build_independent_cohorts_model,
        build_total_times_probability_model,
    )
    from age_group_prediction.scoring import (
        Metric,
        compute_metrics,
        mean_absolute_error_loss,
        mean_bias_error,
        mean_poisson_deviance_loss,
        multinomial_log_loss,
        root_mean_squared_error_loss,
        wmape_loss,
    )

    # Optuna logs every trial at INFO; the studies run in this process.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        CountComponent,
        CrossValidationTuner,
        Metric,
        ProbabilityComponent,
        build_independent_cohorts_model,
        build_total_times_probability_model,
        compute_metrics,
        dataclasses,
        mean_absolute_error_loss,
        mean_bias_error,
        mean_poisson_deviance_loss,
        multinomial_log_loss,
        np,
        optuna,
        root_mean_squared_error_loss,
        wmape_loss,
    )


@app.cell
def _(
    Metric,
    mean_absolute_error_loss,
    mean_bias_error,
    mean_poisson_deviance_loss,
    multinomial_log_loss,
    objective_metric,
    root_mean_squared_error_loss,
    wmape_loss,
):
    # Each metric with the direction Optuna is given; a loss, so "minimize".
    POISSON_DEVIANCE = Metric(
        "poisson_deviance", mean_poisson_deviance_loss, direction="minimize"
    )
    LOG_LOSS = Metric("log_loss", multinomial_log_loss, direction="minimize")
    RMSE = Metric("rmse", root_mean_squared_error_loss, direction="minimize")

    # Every metric: what every trial records and each run's score table (each
    # cohort through `cohort=`, then the composite). The count metrics pool the
    # (building, cohort) cells. Bias is signed, so its direction is nominal;
    # nothing in this notebook reads it.
    METRICS = (
        POISSON_DEVIANCE,
        LOG_LOSS,
        Metric("mae", mean_absolute_error_loss, direction="minimize"),
        RMSE,
        Metric("wmape", wmape_loss, direction="minimize"),
        Metric("bias", mean_bias_error, direction="minimize"),
    )
    # The objective metric: the tuner's objective, the score every study
    # minimizes.
    if objective_metric == "rmse":
        OBJECTIVE_METRIC = RMSE
    elif objective_metric == "poisson_deviance":
        OBJECTIVE_METRIC = POISSON_DEVIANCE
    else:
        raise ValueError(f"unknown objective_metric {objective_metric!r}")
    return METRICS, OBJECTIVE_METRIC


@app.cell
def _(Y, np):
    COHORTS = list(Y.columns)
    # The smallest mean a CountModel predicts: a Gaussian loss can predict a
    # negative count, the Poisson deviance raises at 0 and stays finite at eps.
    PREDICTION_FLOOR = float(np.finfo(float).eps)
    return COHORTS, PREDICTION_FLOOR


@app.cell
def _(
    CrossValidationTuner,
    METRICS,
    OBJECTIVE_METRIC,
    config_hash,
    cv,
    mo,
    n_splits,
    n_startup_trials,
    n_trials,
    project_root,
    seed,
    split_method,
    test_size,
    val_size,
):
    # Every study is stored in its own file and loaded, not tuned, when a click
    # finds it complete (D33): its name holds what it tunes, its run settings,
    # its objective and its seed; the directory's name holds what shapes every
    # study (the data, the split and its folds, the seed, the budget), so a
    # changed setting never loads a stale study. Delete a file to tune its
    # study again; a change to the code or the search spaces is not detected.
    studies_dir = (
        project_root
        / "studies"
        / "__".join(
            f"{key}={value}"
            for key, value in {
                "data": config_hash[:8],
                "split_method": split_method,
                "test_size": test_size,
                "val_size": val_size,
                "n_splits": n_splits,
                "seed": seed,
                "n_trials": n_trials,
                "n_startup_trials": n_startup_trials,
            }.items()
        )
    )
    # The study's design, the same for every run: the objective metric of the
    # composite on each fold, the plain mean over them, the first
    # n_startup_trials of n_trials at random; each trial fits its folds in
    # n_splits processes (D13, D20, D26) and records its cross-validated score
    # on every metric, from the same fold predictions.
    tuner = CrossValidationTuner(
        cv=cv,
        objective_metric=OBJECTIVE_METRIC,
        n_trials=n_trials,
        n_startup_trials=n_startup_trials,
        n_jobs=n_splits,
        recorded_metrics=METRICS,
        storage_dir=studies_dir,
    )

    def build_study_name(
        step: str, run_settings: dict, objective_name: str, study_seed: int
    ) -> str:
        """A study's file name: the step it tunes, its run settings, its objective and its seed."""
        return "__".join(
            [
                step,
                *(f"{key}={value}" for key, value in run_settings.items()),
                f"objective={objective_name}",
                f"seed={study_seed}",
            ]
        )

    mo.md(f"The studies are stored in `{studies_dir.relative_to(project_root)}`.")
    return build_study_name, tuner


@app.cell
def _(ProbabilityComponent, dataclasses):
    def find_feature_transformer_name(component, candidates):
        """The name of `component`'s feature transformer among `candidates`
        (a family's candidates by name), found by identity."""
        return next(
            transformer_name
            for transformer_name, transformer in candidates.items()
            if transformer is component.feature_transformer
        )

    def read_hyperparameters(component):
        """`component`'s hyperparameters, a classifier's with its calibration."""
        hyperparameters = dataclasses.asdict(component.estimator_params)
        if isinstance(component, ProbabilityComponent):
            hyperparameters["calibration_method"] = component.calibration_method
        return hyperparameters

    return find_feature_transformer_name, read_hyperparameters


@app.cell
def _(
    DATA_SETTINGS,
    EXPERIMENT_SETTINGS,
    PREDICTION_FLOOR,
    config_hash,
    hashlib,
    mlflow,
    pd,
    test_age_group,
    train_age_group,
    val_age_group,
):
    # Every run's complete settings: the data's version and settings, the
    # notebook's experiment settings (04's are 03's and its own: the selection, the
    # bootstrap and the inspection), the parts' sizes and the prediction floor every
    # model is built with (its y_min). Every run logs them as settings.json, and some
    # of them as parameters (its logger names which).
    shared_params = {
        "config_hash": config_hash,
        **DATA_SETTINGS,
        **EXPERIMENT_SETTINGS,
        "n_train": len(train_age_group.X),
        "n_val": len(val_age_group.X),
        "n_test": len(test_age_group.X),
        "prediction_floor": PREDICTION_FLOOR,
    }

    def build_mlflow_dataset(age_group_dataset, name):
        """`age_group_dataset` (an AgeGroupDataset) as an MLflow dataset named `name`:
        its X and Y side by side, the digest the SHA-256 of its rows (pandas' row
        hashes), its first 36 characters (MLflow's limit)."""
        df = age_group_dataset.X.join(age_group_dataset.Y)
        row_hashes = pd.util.hash_pandas_object(df).to_numpy()
        digest = hashlib.sha256(row_hashes).hexdigest()[:36]
        return mlflow.data.from_pandas(df, name=name, digest=digest)

    tuning_dataset_inputs = {
        "training": build_mlflow_dataset(train_age_group, "train"),
        "validation": build_mlflow_dataset(val_age_group, "val"),
    }
    return build_mlflow_dataset, shared_params, tuning_dataset_inputs


@app.cell
def _(json, mlflow, optuna):
    def log_tuning_run(
        tuning_run,
        model_name,
        run_settings,
        studies_dir,
        objective_metric_name,
        shared_params,
        config_params,
        dataset_inputs,
        code_tags,
    ):
        """One study's run in the active experiment; its id, or None when a FINISHED run
        has its key (a FAILED one is logged again). Tags: the run key, the study folder
        (the key's first part), the stage, the model, the family and `code_tags`.
        Parameters: the run's settings, five of `shared_params`, the study's seed and
        the best parameters; metrics: the cross-validation score, the composite's
        scores, each cohort's validation score under the objective and the study's
        counts, logged last (a run that fails while logging holds none);
        `settings.json` and `scores.csv` hold the rest. Optuna's optimization history,
        slice plot and parameter importances (fANOVA, seeded with the study's seed;
        none when the complete trials hold fewer than two distinct values) as
        figures."""
        study = tuning_run["study"]
        run_key = f"{studies_dir.name}/{study.study_name}"
        if mlflow.search_runs(
            filter_string=f"tags.run_key = '{run_key}' and attributes.status = 'FINISHED'",
            output_format="list",
        ):
            return None
        if model_name == "independent_cohorts":
            family = run_settings["family"]
        elif model_name == "total_times_probability":
            family = (
                f"{run_settings['total_family']}+{run_settings['probability_family']}"
            )
        else:
            raise ValueError(f"unknown model {model_name!r}")
        # The trials: a row per trial, its seconds a number (a timedelta would come back as
        # text); the start and end times are in the journal.
        trials = study.trials_dataframe(
            attrs=("number", "value", "duration", "params", "user_attrs", "state")
        )
        trials["duration"] = trials["duration"].dt.total_seconds()
        trials = trials.rename(columns={"duration": "seconds"})
        # The search space the trials drew: each parameter's distribution, Optuna's JSON
        # as an object (json_to_distribution(json.dumps(value)) reads it back).
        search_space = {}
        for trial in study.trials:
            for name, distribution in trial.distributions.items():
                search_space[name] = json.loads(
                    optuna.distributions.distribution_to_json(distribution)
                )
        with mlflow.start_run(
            run_name=f"{model_name} {family} exposure={run_settings['use_exposure']}"
        ) as mlflow_run:
            mlflow.set_tags(
                {
                    "run_key": run_key,
                    "study_folder": studies_dir.name,
                    "stage": "tuning",
                    "model": model_name,
                    "family": family,
                    **code_tags,
                }
            )
            # settings.json holds every setting; the parameters are the few a chart's
            # axis or a filter uses: the run's settings, five shared ones, the study's
            # seed and the best parameters (which name each component's feature
            # transformer too).
            mlflow.log_params(
                {
                    **run_settings,
                    **{
                        name: shared_params[name]
                        for name in (
                            "config_hash",
                            "seed",
                            "split_method",
                            "n_trials",
                            "objective_metric",
                        )
                    },
                    "study_seed": tuning_run["study_seed"],
                    **{
                        f"best.{name}": value
                        for name, value in study.best_params.items()
                    },
                }
            )
            # The trials at 6 significant digits, as CSV (the UI shows it as a table);
            # log_table would round to 10 decimals. The journal keeps the exact values.
            mlflow.log_text(
                trials.to_csv(index=False, float_format="%.6g"), "trials.csv"
            )
            mlflow.log_dict(
                {
                    component.name: component.to_dict()
                    for component in tuning_run["components"]
                },
                "components.json",
            )
            # Every cohort's scores, written in full (no float_format): the exact
            # record of the scores the metrics leave out.
            mlflow.log_text(
                tuning_run["scores"].reset_index().to_csv(index=False), "scores.csv"
            )
            mlflow.log_dict(
                {**shared_params, "study_seed": tuning_run["study_seed"]},
                "settings.json",
            )
            mlflow.log_dict(study.best_params, "best_params.json")
            mlflow.log_dict(search_space, "search_space.json")
            mlflow.log_dict(config_params, "config.json")
            # A hyperparameter's effect is read within one study: Optuna's figures.
            mlflow.log_figure(
                optuna.visualization.plot_optimization_history(study),
                "figures/optimization_history.html",
            )
            mlflow.log_figure(
                optuna.visualization.plot_slice(study), "figures/slice.html"
            )
            # fANOVA seeded with the study's seed: a rerun draws the same ranking.
            # It cannot rank a single complete trial or equal values (it raises),
            # so such a study logs no importances figure.
            complete_values = {
                trial.value
                for trial in study.get_trials(
                    deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
                )
            }
            if len(complete_values) > 1:
                mlflow.log_figure(
                    optuna.visualization.plot_param_importances(
                        study,
                        evaluator=optuna.importance.FanovaImportanceEvaluator(
                            seed=tuning_run["study_seed"]
                        ),
                    ),
                    "figures/param_importances.html",
                )
            mlflow.log_artifact(
                str(studies_dir / f"{study.study_name}.log"), artifact_path="study"
            )
            for context, mlflow_dataset in dataset_inputs.items():
                mlflow.log_input(mlflow_dataset, context=context)
            # The metrics last: a run that fails while logging holds none, so it
            # never shows in a chart.
            mlflow.log_metrics(
                {
                    f"cv/{objective_metric_name}": tuning_run["cv_best"],
                    # The composite's scores and each cohort's validation score
                    # under the objective (in the run key, so never stale).
                    **{
                        f"{dataset_name}/{cohort}/{metric}": score
                        for (dataset_name, cohort, metric), score in tuning_run[
                            "scores"
                        ]["score"].items()
                        if cohort == "composite"
                        or (dataset_name == "val" and metric == objective_metric_name)
                    },
                    "seconds": tuning_run["seconds"],
                    "n_complete": tuning_run["n_complete"],
                    "n_failed": tuning_run["n_failed"],
                }
            )
        return mlflow_run.info.run_id

    return (log_tuning_run,)


@app.cell
def _():
    # The independent cohorts model's runs, in the order of their seeds (the
    # i-th study gets seed + i): a count family × exposure on or off. In each,
    # the cohorts share one set of hyperparameters and each draws its own
    # feature transformer (D24.4).
    MODEL1_RUN_SETTINGS = [
        {
            "family": "lgbm_poisson",
            "use_exposure": True,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        {
            "family": "lgbm_poisson",
            "use_exposure": False,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        {
            "family": "lgbm_gaussian",
            "use_exposure": True,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        {
            "family": "lgbm_gaussian",
            "use_exposure": False,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        {
            "family": "poisson_glm",
            "use_exposure": True,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        {
            "family": "poisson_glm",
            "use_exposure": False,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        {
            "family": "nb2",
            "use_exposure": True,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
        {
            "family": "nb2",
            "use_exposure": False,
            "share_hyperparams": True,
            "share_feature_transformer": False,
        },
    ]
    return (MODEL1_RUN_SETTINGS,)


@app.cell
def _(mo):
    run_model1 = mo.ui.run_button(label="Tune the independent cohorts model")
    mo.md(
        f"{run_model1} 8 runs in one loop, one per entry of `MODEL1_RUN_SETTINGS` "
        "(a count family × exposure on or off); each trial fits its folds in parallel. "
        "Each run is logged to MLflow as its study finishes; a run logged before is "
        "skipped."
    )
    return (run_model1,)


@app.cell
def _(
    COHORTS,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    METRICS,
    MODEL1_RUN_SETTINGS,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    build_study_name,
    code_tags,
    compute_metrics,
    config_params,
    cv_groups,
    find_feature_transformer_name,
    log_tuning_run,
    mo,
    pd,
    run_model1,
    seed,
    shared_params,
    train_age_group,
    tuner,
    tuning_dataset_inputs,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model's runs, one after another, the i-th seeded
    # seed + i. A failed trial is caught by the study; any other error ends
    # the loop, and the runs made so far are lost (their MLflow runs stay).
    MODEL1_RUNS = []
    _logged = _skipped = 0
    for _i, _run_settings in enumerate(MODEL1_RUN_SETTINGS):
        _family = _run_settings["family"]
        _use_exposure = _run_settings["use_exposure"]
        _study_seed = seed + _i
        _study, _seconds = tuner.tune_independent_cohorts_model(
            train_age_group.X,
            train_age_group.Y,
            train_age_group.exposure,
            cv_groups,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
            share_hyperparams=_run_settings["share_hyperparams"],
            share_feature_transformer=_run_settings["share_feature_transformer"],
            y_min=PREDICTION_FLOOR,
            random_state=seed,
            study_seed=_study_seed,
            study_name=build_study_name(
                "independent_cohorts",
                _run_settings,
                tuner.objective_metric.name,
                _study_seed,
            ),
        )
        # The best components, read from the best trial's parameters by their
        # type, are the model scored and the record kept (D10-D12, D25).
        _best_components = tuple(
            CountComponent.from_params(
                _study.best_params,
                name=cohort,
                family=_family,
                use_exposure=_use_exposure,
                feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
            )
            for cohort in COHORTS
        )
        # The best model refit on the training part; every metric on training
        # and validation, per cohort and for the composite: its point score.
        _model = build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ).fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
        _scores_by_dataset = {}
        for _dataset_name, _dataset in (
            ("train", train_age_group),
            ("val", val_age_group),
        ):
            _y_pred = _model.predict(_dataset.X, exposure=_dataset.exposure)
            _dataset_scores = compute_metrics(
                _dataset.Y, _y_pred, METRICS, cohorts=[*COHORTS, "composite"]
            ).stack()
            _scores_by_dataset[_dataset_name] = _dataset_scores.to_frame("score")
        # The run's settings and its best model: the fitted model, its
        # components and their transformers' names (found by identity among
        # the family's candidates); its study and the study's seed, whose name,
        # trials, best parameters and seed the MLflow logging reads; its value,
        # trial counts and seconds.
        _states = [trial.state.name for trial in _study.trials]
        _tuning_run = {
            **_run_settings,
            "feature_transformers": {
                component.name: find_feature_transformer_name(
                    component, COUNT_FEATURE_TRANSFORMERS[_family]
                )
                for component in _best_components
            },
            "model": _model,
            "components": _best_components,
            "study": _study,
            "study_seed": _study_seed,
            "cv_best": _study.best_value,
            "n_complete": _states.count("COMPLETE"),
            "n_failed": _states.count("FAIL"),
            "seconds": _seconds,
            "scores": pd.concat(_scores_by_dataset, names=["dataset"]),
        }
        MODEL1_RUNS.append(_tuning_run)
        # Its MLflow run, logged as soon as the study is done (a FINISHED run
        # with its key is skipped).
        _run_id = log_tuning_run(
            _tuning_run,
            "independent_cohorts",
            _run_settings,
            tuner.storage_dir,
            tuner.objective_metric.name,
            shared_params,
            config_params,
            tuning_dataset_inputs,
            code_tags,
        )
        if _run_id is None:
            _skipped += 1
        else:
            _logged += 1
    mo.md(f"Logged {_logged} tuning runs, skipped {_skipped} (already logged).")
    return (MODEL1_RUNS,)


@app.cell
def _(MODEL1_RUNS, pd, read_hyperparameters, tuner):
    _run_rows, _score_rows = [], []
    for _run in MODEL1_RUNS:
        _settings = {
            "family": _run["family"],
            "use_exposure": _run["use_exposure"],
            "share_hyperparams": _run["share_hyperparams"],
            "share_feature_transformer": _run["share_feature_transformer"],
        }
        # The kept model: each cohort's feature transformer and hyperparameters,
        # its cross-validated objective, its trials and seconds.
        _run_rows.append(
            {
                **_settings,
                "feature_transformers": _run["feature_transformers"],
                "hyperparameters": {
                    component.name: read_hyperparameters(component)
                    for component in _run["components"]
                },
                f"cv_{tuner.objective_metric.name}": _run["cv_best"],
                "n_complete": _run["n_complete"],
                "n_failed": _run["n_failed"],
                "seconds": _run["seconds"],
            }
        )
        # Its scores: a row per dataset, cohort (and the composite) and metric.
        _score_rows.extend(
            {**_settings, **_row}
            for _row in _run["scores"].reset_index().to_dict("records")
        )
    model1_runs_df = pd.DataFrame(_run_rows)
    model1_scores_df = pd.DataFrame(_score_rows)
    return model1_runs_df, model1_scores_df


@app.cell
def _(mo, model1_runs_df, model1_scores_df):
    mo.vstack(
        [
            mo.md(
                "**The independent cohorts model: each run's best model** (its "
                "feature transformers and hyperparameters per cohort; its "
                "cross-validated objective; its trials and seconds)"
            ),
            mo.ui.table(model1_runs_df.round(4)),
            mo.md(
                "**Its scores, per dataset, cohort and for the composite** (the "
                "point score; `04_model_inspection.py` bootstraps each model's "
                "best run)"
            ),
            mo.ui.table(model1_scores_df.round(4)),
        ]
    )
    return


@app.cell
def _():
    # The total times probability model's runs, in the order of their seeds
    # (the i-th study gets seed + i): a pair of a count family for the total
    # and a classifier family for the probabilities, × exposure on or off.
    # Each trial draws both components (the joint method, D40).
    MODEL2_RUN_SETTINGS = [
        {
            "total_family": "lgbm_poisson",
            "probability_family": "logistic",
            "use_exposure": True,
        },
        {
            "total_family": "lgbm_poisson",
            "probability_family": "logistic",
            "use_exposure": False,
        },
        {
            "total_family": "lgbm_poisson",
            "probability_family": "lgbm",
            "use_exposure": True,
        },
        {
            "total_family": "lgbm_poisson",
            "probability_family": "lgbm",
            "use_exposure": False,
        },
        {
            "total_family": "lgbm_gaussian",
            "probability_family": "logistic",
            "use_exposure": True,
        },
        {
            "total_family": "lgbm_gaussian",
            "probability_family": "logistic",
            "use_exposure": False,
        },
        {
            "total_family": "lgbm_gaussian",
            "probability_family": "lgbm",
            "use_exposure": True,
        },
        {
            "total_family": "lgbm_gaussian",
            "probability_family": "lgbm",
            "use_exposure": False,
        },
        {
            "total_family": "poisson_glm",
            "probability_family": "logistic",
            "use_exposure": True,
        },
        {
            "total_family": "poisson_glm",
            "probability_family": "logistic",
            "use_exposure": False,
        },
        {
            "total_family": "poisson_glm",
            "probability_family": "lgbm",
            "use_exposure": True,
        },
        {
            "total_family": "poisson_glm",
            "probability_family": "lgbm",
            "use_exposure": False,
        },
        {
            "total_family": "nb2",
            "probability_family": "logistic",
            "use_exposure": True,
        },
        {
            "total_family": "nb2",
            "probability_family": "logistic",
            "use_exposure": False,
        },
        {
            "total_family": "nb2",
            "probability_family": "lgbm",
            "use_exposure": True,
        },
        {
            "total_family": "nb2",
            "probability_family": "lgbm",
            "use_exposure": False,
        },
    ]
    return (MODEL2_RUN_SETTINGS,)


@app.cell
def _(mo):
    run_model2 = mo.ui.run_button(label="Tune the total times probability model")
    mo.md(
        f"{run_model2} 16 runs in one loop, one per entry of `MODEL2_RUN_SETTINGS` "
        "(a family pair × exposure on or off); each trial fits its folds in parallel. "
        "Each run is logged to MLflow as its study finishes; a run logged before is "
        "skipped."
    )
    return (run_model2,)


@app.cell
def _(
    COHORTS,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    METRICS,
    MODEL2_RUN_SETTINGS,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_study_name,
    build_total_times_probability_model,
    code_tags,
    compute_metrics,
    config_params,
    cv_groups,
    find_feature_transformer_name,
    log_tuning_run,
    mo,
    pd,
    run_model2,
    seed,
    shared_params,
    train_age_group,
    tuner,
    tuning_dataset_inputs,
    val_age_group,
):
    mo.stop(
        not run_model2.value,
        mo.md("Click the button to tune the total times probability model."),
    )
    # The total times probability model's runs, one after another, the i-th
    # seeded seed + i. A failed trial is caught by the study; any other error
    # ends the loop, and the runs made so far are lost (their MLflow runs stay).
    MODEL2_RUNS = []
    _logged = _skipped = 0
    for _i, _run_settings in enumerate(MODEL2_RUN_SETTINGS):
        _total_family = _run_settings["total_family"]
        _probability_family = _run_settings["probability_family"]
        _use_exposure = _run_settings["use_exposure"]
        _study_seed = seed + _i
        _study, _seconds = tuner.tune_total_times_probability_model(
            train_age_group.X,
            train_age_group.Y,
            train_age_group.exposure,
            cv_groups,
            total_family=_total_family,
            probability_family=_probability_family,
            use_exposure=_use_exposure,
            total_feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_total_family],
            probability_feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                _probability_family
            ],
            y_min=PREDICTION_FLOOR,
            random_state=seed,
            study_seed=_study_seed,
            study_name=build_study_name(
                "total_times_probability_joint",
                _run_settings,
                tuner.objective_metric.name,
                _study_seed,
            ),
        )
        # The best components, read from the best trial's parameters by their
        # type (the two parts share LightGBM's parameter names; each reads its
        # own), are the model scored and the record kept (D25).
        _total_component = CountComponent.from_params(
            _study.best_params,
            name="total",
            family=_total_family,
            use_exposure=_use_exposure,
            feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_total_family],
        )
        _probability_component = ProbabilityComponent.from_params(
            _study.best_params,
            name="probability",
            family=_probability_family,
            feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                _probability_family
            ],
        )
        _best_components = (_total_component, _probability_component)
        # The best model refit on the training part; every metric on training
        # and validation, per cohort and for the composite: its point score.
        _model = build_total_times_probability_model(
            *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ).fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
        _scores_by_dataset = {}
        for _dataset_name, _dataset in (
            ("train", train_age_group),
            ("val", val_age_group),
        ):
            _y_pred = _model.predict(_dataset.X, exposure=_dataset.exposure)
            _dataset_scores = compute_metrics(
                _dataset.Y, _y_pred, METRICS, cohorts=[*COHORTS, "composite"]
            ).stack()
            _scores_by_dataset[_dataset_name] = _dataset_scores.to_frame("score")
        # The run's settings and its best model: the fitted model, its
        # components and their transformers' names (found by identity among
        # each family's candidates); its study and the study's seed, whose name,
        # trials, best parameters and seed the MLflow logging reads; its value,
        # trial counts and seconds.
        _states = [trial.state.name for trial in _study.trials]
        _tuning_run = {
            **_run_settings,
            "feature_transformers": {
                "total": find_feature_transformer_name(
                    _total_component, COUNT_FEATURE_TRANSFORMERS[_total_family]
                ),
                "probability": find_feature_transformer_name(
                    _probability_component,
                    PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
                ),
            },
            "model": _model,
            "components": _best_components,
            "study": _study,
            "study_seed": _study_seed,
            "cv_best": _study.best_value,
            "n_complete": _states.count("COMPLETE"),
            "n_failed": _states.count("FAIL"),
            "seconds": _seconds,
            "scores": pd.concat(_scores_by_dataset, names=["dataset"]),
        }
        MODEL2_RUNS.append(_tuning_run)
        # Its MLflow run, logged as soon as the study is done (a FINISHED run
        # with its key is skipped).
        _run_id = log_tuning_run(
            _tuning_run,
            "total_times_probability",
            _run_settings,
            tuner.storage_dir,
            tuner.objective_metric.name,
            shared_params,
            config_params,
            tuning_dataset_inputs,
            code_tags,
        )
        if _run_id is None:
            _skipped += 1
        else:
            _logged += 1
    mo.md(f"Logged {_logged} tuning runs, skipped {_skipped} (already logged).")
    return (MODEL2_RUNS,)


@app.cell
def _(MODEL2_RUNS, pd, read_hyperparameters, tuner):
    _run_rows, _score_rows = [], []
    for _run in MODEL2_RUNS:
        # A family pair reads as one family, so both models' tables have the same
        # settings columns; the record keeps the two families (D16).
        _settings = {
            "family": f"{_run['total_family']}+{_run['probability_family']}",
            "use_exposure": _run["use_exposure"],
        }
        # The kept model: each component's feature transformer and
        # hyperparameters (the classifier's with its calibration), its
        # cross-validated objective, its trials and seconds.
        _run_rows.append(
            {
                **_settings,
                "feature_transformers": _run["feature_transformers"],
                "hyperparameters": {
                    component.name: read_hyperparameters(component)
                    for component in _run["components"]
                },
                f"cv_{tuner.objective_metric.name}": _run["cv_best"],
                "n_complete": _run["n_complete"],
                "n_failed": _run["n_failed"],
                "seconds": _run["seconds"],
            }
        )
        # Its scores: a row per dataset, cohort (and the composite) and metric.
        _score_rows.extend(
            {**_settings, **_row}
            for _row in _run["scores"].reset_index().to_dict("records")
        )
    model2_runs_df = pd.DataFrame(_run_rows)
    model2_scores_df = pd.DataFrame(_score_rows)
    return model2_runs_df, model2_scores_df


@app.cell
def _(mo, model2_runs_df, model2_scores_df):
    mo.vstack(
        [
            mo.md(
                "**The total times probability model: each run's best model** "
                "(family: the total's + the probabilities'; each component's "
                "feature transformer and hyperparameters; its cross-validated "
                "objective; its trials and seconds)"
            ),
            mo.ui.table(model2_runs_df.round(4)),
            mo.md(
                "**Its scores, per dataset, cohort and for the composite** (the "
                "point score; `04_model_inspection.py` bootstraps each model's "
                "best run)"
            ),
            mo.ui.table(model2_scores_df.round(4)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
