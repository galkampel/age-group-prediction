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
    # Age Group Prediction: model inspection

    The tuning runs `03_hyperparameter_tuning.py` logs to MLflow, read back:
    each model's best run is rebuilt from what it logged and refit on the
    training part. Sections 1-4 are 03's setup, the same code (a test pins
    the two notebooks' setup equal), so the data, the split and the feature
    variants are the ones the runs were tuned on; a guard checks each best
    run against them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Data generation

    The synthetic population of `02_objective_eda.py`, where it is described;
    the same code as `03_hyperparameter_tuning.py`.
    """)
    return


@app.cell
def _():
    import hashlib
    import io
    import json

    import numpy as np

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
    from age_group_prediction.hyperparameter_tuning import (
        CountComponent,
        ProbabilityComponent,
        build_independent_cohorts_model,
        build_total_times_probability_model,
    )
    from age_group_prediction.preprocessing import ExposureTransformer, ShareTransformer
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
        CountComponent,
        DomainScale,
        ExposureTransformer,
        FeatureTransformer,
        Interaction,
        Metric,
        NeighborhoodSimulator,
        OneHot,
        Plan,
        ProbabilityComponent,
        Quadratic,
        RelativeSaturation,
        ShareTransformer,
        Splitter,
        Standardize,
        StudentPopulationSimulator,
        TotalChildrenSimulator,
        build_independent_cohorts_model,
        build_total_times_probability_model,
        compute_metrics,
        hashlib,
        io,
        json,
        load_simulation_config,
        mean_absolute_error_loss,
        mean_bias_error,
        mean_poisson_deviance_loss,
        multinomial_log_loss,
        np,
        root_mean_squared_error_loss,
        wmape_loss,
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
        "notebook": "notebooks/04_model_inspection.py",
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

    The tuning's settings, 03's (a test pins them equal): they name the study
    folder whose runs are read below. Then the inspection's own: the
    selection metric, the bootstrap and the inspection.
    """)
    return


@app.cell
def _(mo, pd):
    seed = 42
    split_method = "random"  # "random", "stratified_by_group" or "grouped"
    test_size = 0.2  # share of the whole table
    val_size = 0.2  # share of the whole table, so 0.25 of the remainder
    # The tuning's budget and folds: nothing is tuned here, they name the folder.
    n_splits = 5
    n_trials = 100
    n_startup_trials = 10
    # The objective every study minimized: "rmse" or "poisson_deviance" (D42).
    objective_metric = "rmse"
    # The metric that picks each model's best run, by its composite score on
    # validation: "rmse" or "poisson_deviance" (D34, D42).
    selection_metric = "rmse"
    n_resamples = 1000
    # The bootstrap interval's level is 1 - alpha: 95%.
    alpha = 0.05
    # The inspection of the best runs: permutation repeats per column, and the
    # validation rows SHAP explains against the training rows it averages over.
    n_repeats = 10
    n_shap_rows = 100
    n_shap_background = 50

    # What every evaluation run logs.
    EXPERIMENT_SETTINGS = {
        "seed": seed,
        "split_method": split_method,
        "test_size": test_size,
        "val_size": val_size,
        "n_splits": n_splits,
        "n_trials": n_trials,
        "n_startup_trials": n_startup_trials,
        "objective_metric": objective_metric,
        "selection_metric": selection_metric,
        "n_resamples": n_resamples,
        "alpha": alpha,
        "n_repeats": n_repeats,
        "n_shap_rows": n_shap_rows,
        "n_shap_background": n_shap_background,
    }
    print(
        "experiment_settings "
        + " ".join(f"{name}={value}" for name, value in EXPERIMENT_SETTINGS.items())
    )
    mo.md(f"```text\n{pd.DataFrame([EXPERIMENT_SETTINGS]).to_string(index=False)}\n```")
    return (
        EXPERIMENT_SETTINGS,
        alpha,
        n_repeats,
        n_resamples,
        n_shap_background,
        n_shap_rows,
        n_splits,
        n_startup_trials,
        n_trials,
        objective_metric,
        seed,
        selection_metric,
        split_method,
        test_size,
        val_size,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Train, validation and test split

    The split of `03_hyperparameter_tuning.py`, the same code; no folds,
    since nothing is tuned here.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Feature variants

    03's candidate feature transformers, the same declarations: a best run's
    parameters name each component's transformer, which is read from these.
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


@app.cell
def _(
    Metric,
    mean_absolute_error_loss,
    mean_bias_error,
    mean_poisson_deviance_loss,
    multinomial_log_loss,
    objective_metric,
    root_mean_squared_error_loss,
    selection_metric,
    wmape_loss,
):
    # Each metric with its direction; a loss, so "minimize".
    POISSON_DEVIANCE = Metric(
        "poisson_deviance", mean_poisson_deviance_loss, direction="minimize"
    )
    LOG_LOSS = Metric("log_loss", multinomial_log_loss, direction="minimize")
    RMSE = Metric("rmse", root_mean_squared_error_loss, direction="minimize")

    # Every metric 03 scores a run on (each cohort through `cohort=`, then the
    # composite): what a rebuilt run is scored on. Bias is signed, so its
    # direction is nominal.
    METRICS = (
        POISSON_DEVIANCE,
        LOG_LOSS,
        Metric("mae", mean_absolute_error_loss, direction="minimize"),
        RMSE,
        Metric("wmape", wmape_loss, direction="minimize"),
        Metric("bias", mean_bias_error, direction="minimize"),
    )
    # The objective metric: the one every study minimized; the runs read are
    # those logged under it.
    if objective_metric == "rmse":
        OBJECTIVE_METRIC = RMSE
    elif objective_metric == "poisson_deviance":
        OBJECTIVE_METRIC = POISSON_DEVIANCE
    else:
        raise ValueError(f"unknown objective_metric {objective_metric!r}")
    # The selection metric: it picks each model's best run (the composite's
    # point score on validation).
    if selection_metric == "rmse":
        SELECTION_METRIC = RMSE
    elif selection_metric == "poisson_deviance":
        SELECTION_METRIC = POISSON_DEVIANCE
    else:
        raise ValueError(f"unknown selection_metric {selection_metric!r}")
    return METRICS, OBJECTIVE_METRIC, SELECTION_METRIC


@app.cell
def _(Y, np):
    COHORTS = list(Y.columns)
    # The smallest mean a CountModel predicts: a Gaussian loss can predict a
    # negative count, the Poisson deviance raises at 0 and stays finite at eps.
    PREDICTION_FLOOR = float(np.finfo(float).eps)
    return COHORTS, PREDICTION_FLOOR


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
    # experiment's settings (the selection metric among them), the parts' sizes and
    # the prediction floor every model is built with (its y_min). Every run logs
    # them as settings.json; a tuning run five of them as parameters, an evaluation
    # run six.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) The best runs

    The FINISHED tuning runs of this setup are read from MLflow: those
    tagged with the study folder below and logged under the objective
    metric. Unless every run is there (8 of the independent cohorts model,
    16 of the total times probability model, each run key once), the
    notebook stops at the read and shows the counts: a partial store is
    never selected from. Each model's best run is the lowest validation
    score of the composite by `selection_metric` (a tie goes to the lower
    study seed). It is rebuilt from what it logged (`components.json`,
    `best_params.json`), refit on the training part and scored; a
    ValueError stops the notebook when the rebuild does not reproduce the
    run: its dataset inputs, its components' declaration or its scores
    (exactly).
    """)
    return


@app.cell
def _(
    config_hash,
    mo,
    n_splits,
    n_startup_trials,
    n_trials,
    seed,
    split_method,
    test_size,
    val_size,
):
    # The tuning runs' study folder: the name 03 gives its studies directory,
    # from what shapes every study (the data, the split and its folds, the
    # seed, the budget). Every run 03 logs carries it as its study_folder tag.
    study_folder = "__".join(
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
    # The runs 03 logs per model, one per entry of its run settings.
    N_TUNING_RUNS = {"independent_cohorts": 8, "total_times_probability": 16}
    mo.md(f"The tuning runs are read from the study folder `{study_folder}`.")
    return N_TUNING_RUNS, study_folder


@app.cell
def _(
    COHORTS,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    METRICS,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_independent_cohorts_model,
    build_total_times_probability_model,
    compute_metrics,
    io,
    json,
    mlflow,
    np,
    pd,
    seed,
    train_age_group,
    val_age_group,
):
    def select_best_runs(tuning_runs_df, selection_metric_name):
        """Each model's best run among `tuning_runs_df` (a search_runs frame), by
        model: the lowest validation score of the composite on
        `selection_metric_name`, a tie to the lower study seed (a parameter, so a
        string, compared as an integer)."""
        ranked = tuning_runs_df.assign(
            study_seed=pd.to_numeric(tuning_runs_df["params.study_seed"])
        ).sort_values([f"metrics.val/composite/{selection_metric_name}", "study_seed"])
        return {
            model_name: model_runs.iloc[0]
            for model_name, model_runs in ranked.groupby("tags.model")
        }

    def read_tuning_run(run_id):
        """What the tuning run `run_id` logged for its rebuild: its components'
        declaration (components.json), its best parameters (best_params.json), its
        scores (scores.csv, read back exactly) and its dataset inputs' digests by
        context."""
        scores_csv = mlflow.artifacts.load_text(f"runs:/{run_id}/scores.csv")
        return {
            "components": mlflow.artifacts.load_dict(f"runs:/{run_id}/components.json"),
            "best_params": mlflow.artifacts.load_dict(
                f"runs:/{run_id}/best_params.json"
            ),
            "scores": pd.read_csv(io.StringIO(scores_csv), float_precision="round_trip"),
            "dataset_digests": {
                {tag.key: tag.value for tag in item.tags}[
                    "mlflow.data.context"
                ]: item.dataset.digest
                for item in mlflow.get_run(run_id).inputs.dataset_inputs
            },
        }

    def rebuild_best_run(run_id, model_name, components_record, best_params):
        """The best model of the tuning run `run_id` of `model_name`, rebuilt from
        what it logged: its components read from `best_params` by `from_params`,
        each with the family (and a count component's exposure)
        `components_record` (components.json) gives it; built as 03 builds it,
        refit on the training part and scored on training and validation, every
        metric per cohort and for the composite. Also the count components'
        common exposure. A name this notebook lacks is a ValueError naming the
        run."""
        # A name the run logged that this notebook lacks (a feature transformer
        # renamed or removed, a cohort or a hyperparameter added) is a KeyError
        # of from_params: refused as the guard refuses, naming the run.
        try:
            if model_name == "independent_cohorts":
                components = tuple(
                    CountComponent.from_params(
                        best_params,
                        name=cohort,
                        family=components_record[cohort]["family"],
                        use_exposure=components_record[cohort]["use_exposure"],
                        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[
                            components_record[cohort]["family"]
                        ],
                    )
                    for cohort in COHORTS
                )
                model = build_independent_cohorts_model(
                    components, y_min=PREDICTION_FLOOR, random_state=seed
                )
            elif model_name == "total_times_probability":
                total_record = components_record["total"]
                probability_record = components_record["probability"]
                components = (
                    CountComponent.from_params(
                        best_params,
                        name="total",
                        family=total_record["family"],
                        use_exposure=total_record["use_exposure"],
                        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[
                            total_record["family"]
                        ],
                    ),
                    ProbabilityComponent.from_params(
                        best_params,
                        name="probability",
                        family=probability_record["family"],
                        feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                            probability_record["family"]
                        ],
                    ),
                )
                model = build_total_times_probability_model(
                    *components, y_min=PREDICTION_FLOOR, random_state=seed
                )
            else:
                raise ValueError(f"unknown model {model_name!r}")
        except KeyError as error:
            raise ValueError(
                f"the tuning run {run_id}'s components cannot be rebuilt: "
                f"this notebook has no {error}"
            ) from error
        model.fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
        scores_by_dataset = {}
        for dataset_name, dataset in (("train", train_age_group), ("val", val_age_group)):
            y_pred = model.predict(dataset.X, exposure=dataset.exposure)
            scores_by_dataset[dataset_name] = (
                compute_metrics(
                    dataset.Y, y_pred, METRICS, cohorts=[*COHORTS, "composite"]
                )
                .stack()
                .to_frame("score")
            )
        # The cohorts' common exposure, or the total's: one value.
        (use_exposure,) = {
            component.use_exposure
            for component in components
            if isinstance(component, CountComponent)
        }
        return {
            "components": components,
            "model": model,
            "scores": pd.concat(scores_by_dataset, names=["dataset"]),
            "use_exposure": use_exposure,
        }

    def check_best_run(run_id, logged, rebuilt, dataset_inputs):
        """A ValueError naming the tuning run `run_id` when its rebuild does not
        reproduce it: its dataset inputs' digests against `dataset_inputs` (this
        notebook's data and split), the rebuilt components' declaration against
        components.json (the feature transformers), the rebuilt scores against
        scores.csv, exactly (the floor, the seed, the libraries). `logged` is
        read_tuning_run's record, `rebuilt` rebuild_best_run's."""
        digests = {
            context: dataset.digest for context, dataset in dataset_inputs.items()
        }
        if logged["dataset_digests"] != digests:
            raise ValueError(
                f"the tuning run {run_id}'s dataset inputs "
                f"{logged['dataset_digests']} differ from this notebook's {digests}"
            )
        # Through JSON, as components.json was written.
        components = {
            component.name: json.loads(json.dumps(component.to_dict()))
            for component in rebuilt["components"]
        }
        if components != logged["components"]:
            differing = sorted(
                name
                for name in components.keys() | logged["components"].keys()
                if components.get(name) != logged["components"].get(name)
            )
            raise ValueError(
                f"the tuning run {run_id}'s components {differing} differ from "
                "their rebuild by this notebook's feature transformers"
            )
        scores = rebuilt["scores"].reset_index()
        keys = ["dataset", "cohort", "metric"]
        if scores[keys].to_numpy().tolist() != logged["scores"][keys].to_numpy().tolist():
            raise ValueError(
                f"the tuning run {run_id}'s scores.csv rows differ from its rebuild's"
            )
        # Exactly; a NaN in both is the same score.
        if not np.array_equal(
            scores["score"].to_numpy(),
            logged["scores"]["score"].to_numpy(),
            equal_nan=True,
        ):
            n_differing = int(
                (scores["score"].to_numpy() != logged["scores"]["score"].to_numpy()).sum()
            )
            raise ValueError(
                f"the tuning run {run_id}'s scores differ from its rebuild's: "
                f"{n_differing} of {len(scores)} in scores.csv"
            )

    return check_best_run, read_tuning_run, rebuild_best_run, select_best_runs


@app.cell
def _(
    METRICS,
    N_TUNING_RUNS,
    SELECTION_METRIC,
    check_best_run,
    mlflow,
    mo,
    objective_metric,
    pd,
    read_tuning_run,
    rebuild_best_run,
    select_best_runs,
    study_folder,
    tuning_dataset_inputs,
):
    # The FINISHED tuning runs of this setup, in the experiment of the
    # environment: its study folder and its objective.
    tuning_runs_df = mlflow.search_runs(
        filter_string=(
            "tags.stage = 'tuning' "
            f"and tags.study_folder = '{study_folder}' "
            f"and params.objective_metric = '{objective_metric}' "
            "and attributes.status = 'FINISHED'"
        )
    )
    # Every run of each model, once: its count and its distinct run keys must
    # both be N_TUNING_RUNS', so a duplicate cannot stand in for a missing run.
    # An empty frame has no tag columns, hence the reindex.
    _keys = tuning_runs_df.reindex(columns=["tags.model", "tags.run_key"])
    _n_runs = {
        model_name: int((_keys["tags.model"] == model_name).sum())
        for model_name in N_TUNING_RUNS
    }
    _n_run_keys = {
        model_name: _keys.loc[_keys["tags.model"] == model_name, "tags.run_key"].nunique()
        for model_name in N_TUNING_RUNS
    }
    print(
        "tuning_runs "
        + " ".join(f"{model_name}={n}" for model_name, n in _n_runs.items())
    )
    _counts_df = pd.DataFrame(
        {
            "expected": N_TUNING_RUNS,
            "finished_runs": _n_runs,
            "distinct_run_keys": _n_run_keys,
        }
    )
    mo.stop(
        _n_runs != N_TUNING_RUNS or _n_run_keys != N_TUNING_RUNS,
        mo.vstack(
            [
                mo.md(
                    f"**The study folder `{study_folder}` does not hold every "
                    f"tuning run** (FINISHED, objective `{objective_metric}`): "
                    "nothing is selected. Tune both models in "
                    "`03_hyperparameter_tuning.py` first."
                ),
                mo.md(f"```text\n{_counts_df.to_string()}\n```"),
            ]
        ),
    )
    # Each model's best run, rebuilt from its logged record and refit on the
    # training part; the guard raises before anything reads a rebuild that
    # does not reproduce its run.
    best_runs = {}
    for _model_name, _run in select_best_runs(
        tuning_runs_df, SELECTION_METRIC.name
    ).items():
        _logged = read_tuning_run(_run["run_id"])
        _rebuilt = rebuild_best_run(
            _run["run_id"], _model_name, _logged["components"], _logged["best_params"]
        )
        check_best_run(_run["run_id"], _logged, _rebuilt, tuning_dataset_inputs)
        best_runs[_model_name] = {
            "run_id": _run["run_id"],
            "study_name": _run["tags.run_key"].split("/", 1)[1],
            "family": _run["tags.family"],
            "use_exposure": _rebuilt["use_exposure"],
            "best_params": _logged["best_params"],
            "components": _rebuilt["components"],
            "model": _rebuilt["model"],
            "scores": _rebuilt["scores"],
        }
    # A row per tuning run: its model, family and exposure, then the
    # composite's validation score per metric; each model's runs from its best.
    _runs_table = (
        tuning_runs_df.rename(
            columns={
                "tags.model": "model",
                "tags.family": "family",
                "params.use_exposure": "use_exposure",
                **{
                    f"metrics.val/composite/{metric.name}": metric.name
                    for metric in METRICS
                },
            }
        )[["model", "family", "use_exposure", *(metric.name for metric in METRICS)]]
        .sort_values(["model", SELECTION_METRIC.name])
        .reset_index(drop=True)
    )
    mo.vstack(
        [
            mo.md(
                f"**The {len(tuning_runs_df)} tuning runs** (the composite's "
                "validation score per metric; each model's runs from the lowest "
                f"{SELECTION_METRIC.name})"
            ),
            mo.ui.table(_runs_table.round(4)),
            mo.md(
                f"**Each model's best run by the {SELECTION_METRIC.name} of the "
                "composite on validation**, rebuilt and refit on the training "
                "part, its scores equal to the logged ones: "
                + "; ".join(
                    f"{model_name} `{run['family']}`, exposure {run['use_exposure']} "
                    f"(run `{run['run_id']}`)"
                    for model_name, run in best_runs.items()
                )
            ),
        ]
    )
    return best_runs, tuning_runs_df


if __name__ == "__main__":
    app.run()
