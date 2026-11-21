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
    run against them. The two best runs are then bootstrapped and inspected
    on validation (section 5) and evaluated once on the test part (section
    6), each section logging its MLflow run.
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
    import functools
    import hashlib
    import io
    import json

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde

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
    from age_group_prediction.inspection import (
        BootstrapEvaluator,
        PermutationImportance,
        ShapExplainer,
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
        BootstrapEvaluator,
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
        PermutationImportance,
        Plan,
        ProbabilityComponent,
        Quadratic,
        RelativeSaturation,
        ShapExplainer,
        ShareTransformer,
        Splitter,
        Standardize,
        StudentPopulationSimulator,
        TotalChildrenSimulator,
        build_independent_cohorts_model,
        build_total_times_probability_model,
        compute_metrics,
        functools,
        gaussian_kde,
        go,
        hashlib,
        io,
        json,
        load_simulation_config,
        make_subplots,
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

    Each best run is then bootstrapped over whole neighborhoods and
    inspected behind its button (the permutation importance and the SHAP
    values per cohort). The two best runs' paired difference, per bootstrap
    replicate, shows whether one is better; a "Metric" menu chooses the
    metric of each plot (`selection_metric` by default); the permutation
    importance always scores `selection_metric`.
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


@app.cell
def _(BootstrapEvaluator, alpha, n_resamples, seed):
    # The bootstrap of each model's best run, over neighborhoods: one seed, so
    # both models score the same replicate rows and the intervals compare; the
    # interval's bounds named by their level (95% at alpha 0.05).
    bootstrap_evaluator = BootstrapEvaluator(n_resamples, random_state=seed)
    _level = round(100 * (1 - alpha))
    CI_BOUND_NAMES = {"lower": f"ci{_level}_lower", "upper": f"ci{_level}_upper"}
    return CI_BOUND_NAMES, bootstrap_evaluator


@app.cell
def _(pd):
    def summarize_replicates(scores, replicate_scores, alpha, bound_names):
        """A row per cohort and metric: `scores`, then the mean, median, SD and
        percentile interval (its bounds named by `bound_names`) of
        `replicate_scores`, a column per cohort and metric."""
        return pd.DataFrame(
            {
                "score": scores,
                "mean": replicate_scores.mean(),
                "median": replicate_scores.median(),
                "sd": replicate_scores.std(),
                bound_names["lower"]: replicate_scores.quantile(alpha / 2),
                bound_names["upper"]: replicate_scores.quantile(1 - alpha / 2),
            }
        )

    def build_paired_difference(scores_df, replicate_scores, alpha, bound_names):
        """The two best runs' paired difference, total times probability minus
        independent cohorts: per replicate (both scored on the same replicate
        rows), and a row per cohort and metric with the point scores'
        difference and the replicates' statistics."""
        difference_replicates = (
            replicate_scores["total_times_probability"]
            - replicate_scores["independent_cohorts"]
        )
        point_scores = scores_df.set_index(["model", "cohort", "metric"])["score"]
        difference_df = (
            summarize_replicates(
                point_scores["total_times_probability"]
                - point_scores["independent_cohorts"],
                difference_replicates,
                alpha,
                bound_names,
            )
            .rename_axis(["cohort", "metric"])
            .reset_index()
        )
        difference_df.insert(
            0, "model", "total_times_probability - independent_cohorts"
        )
        return difference_replicates, difference_df

    return build_paired_difference, summarize_replicates


@app.cell
def _(
    CI_BOUND_NAMES,
    COHORTS,
    METRICS,
    alpha,
    best_runs,
    bootstrap_evaluator,
    build_paired_difference,
    mo,
    pd,
    summarize_replicates,
    val_age_group,
):
    # Each model's best run, read above: rebuilt from its tuning run and refit
    # on the training part. The two bootstrapped on validation, every metric
    # per cohort and for the composite, by one evaluator: both score the same
    # replicate rows, so their replicates pair row by row. Each score with its
    # replicates' mean, median, SD and percentile interval.
    best_replicate_scores, _scores_by_run = {}, {}
    for _model_name, _run in best_runs.items():
        _replicate_scores = bootstrap_evaluator.evaluate(
            val_age_group.Y,
            _run["model"].predict(val_age_group.X, exposure=val_age_group.exposure),
            val_age_group.groups,
            metrics=METRICS,
            cohorts=[*COHORTS, "composite"],
        )
        best_replicate_scores[_model_name] = _replicate_scores
        _scores_by_run[_model_name, _run["family"], _run["use_exposure"]] = (
            summarize_replicates(
                _run["scores"].loc["val", "score"],
                _replicate_scores,
                alpha,
                CI_BOUND_NAMES,
            )
        )
    # A row per best run, cohort and metric: the run's model, family and
    # exposure, then its validation score and the bootstrap's statistics.
    best_runs_val_scores_df = pd.concat(
        _scores_by_run, names=["model", "family", "use_exposure"]
    ).reset_index()
    # The two best runs' paired difference per replicate: both were scored on
    # the same replicate rows (one evaluator), so the difference keeps the
    # pairing, and its interval is far narrower than either run's. Below 0
    # the total times probability model scores lower.
    best_difference_replicates, best_difference_df = build_paired_difference(
        best_runs_val_scores_df, best_replicate_scores, alpha, CI_BOUND_NAMES
    )
    mo.md(
        "**The two best runs, bootstrapped on validation** (each score with its "
        "replicates' mean, median, SD and 95% interval; a table per model below)"
    )
    return (
        best_difference_df,
        best_difference_replicates,
        best_replicate_scores,
        best_runs_val_scores_df,
    )


@app.cell
def _(COHORTS, METRICS, mo):
    # What the best runs' tables show: one cohort (the composite by default)
    # and the chosen metrics (all by default); one choice filters both tables.
    best_runs_cohort = mo.ui.dropdown(
        options=[*COHORTS, "composite"], value="composite", label="Cohort"
    )
    best_runs_metrics = mo.ui.multiselect(
        options=[metric.name for metric in METRICS],
        value=[metric.name for metric in METRICS],
        label="Metrics",
    )
    return best_runs_cohort, best_runs_metrics


@app.cell
def _(best_runs_cohort, best_runs_metrics, best_runs_val_scores_df, mo):
    # Each model's best run in its own table, the chosen cohort and metrics in
    # METRICS' order, so the two tables compare row by row.
    _chosen = best_runs_val_scores_df[
        (best_runs_val_scores_df["cohort"] == best_runs_cohort.value)
        & best_runs_val_scores_df["metric"].isin(best_runs_metrics.value)
    ]
    _views = []
    for _model, _family, _use_exposure in (
        best_runs_val_scores_df[["model", "family", "use_exposure"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        _views.append(mo.md(f"**{_model}:** `{_family}`, exposure {_use_exposure}"))
        # A fresh index: the df's row positions would show as a column.
        _rows = _chosen[_chosen["model"] == _model].reset_index(drop=True)
        _views.append(mo.ui.table(_rows.round(4)))
    mo.vstack(
        [mo.hstack([best_runs_cohort, best_runs_metrics], justify="start"), *_views]
    )
    return


@app.cell
def _():
    # The figures' palette, the dataviz skill's reference: the three
    # categorical colors that stay apart for colorblind readers, the
    # sequential blue ramp (steps 250, 400, 550, 700, so the lightest still
    # shows on the surface), shap's own pair (red raises, blue lowers) and
    # its blue-to-red scale for the beeswarm (8 stops sampled from
    # `shap.plots.colors.red_blue` at linspace(0, 1, 8)), then the surface,
    # the inks, the gridline and the axis line.
    PALETTE = {
        "categorical": ["#2a78d6", "#eb6834", "#1baf7a"],
        "blues": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "shap": [
            [0.0, "#008bfb"],
            [1 / 7, "#0076f1"],
            [2 / 7, "#5d5cdd"],
            [3 / 7, "#8b3bbf"],
            [4 / 7, "#b20ca4"],
            [5 / 7, "#d8008d"],
            [6 / 7, "#f30070"],
            [1.0, "#ff0051"],
        ],
        "raises": "#ff0051",
        "lowers": "#008bfb",
        "surface": "#fcfcfb",
        "ink": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    }
    # Each model's color wherever a figure names it: the forest's, by the
    # models' order.
    MODEL_COLORS = {
        "independent_cohorts": PALETTE["categorical"][0],
        "total_times_probability": PALETTE["categorical"][1],
    }

    def style_figure(figure, palette):
        """Set the figures' shared style on `figure`, from `palette`: the
        surface behind the paper and the plot, the font, and on every axis
        the gridline, no zero line, the axis line and muted ticks. Call it
        once the figure has its axes; what the figure sets after it wins."""
        figure.update_layout(
            template="plotly_white",
            paper_bgcolor=palette["surface"],
            plot_bgcolor=palette["surface"],
            font={
                "family": "system-ui, -apple-system, Segoe UI, sans-serif",
                "color": palette["ink"],
                "size": 12,
            },
        )
        for update_axes in (figure.update_xaxes, figure.update_yaxes):
            update_axes(
                gridcolor=palette["grid"],
                zeroline=False,
                linecolor=palette["axis"],
                tickfont={"color": palette["muted"]},
            )

    return MODEL_COLORS, PALETTE, style_figure


@app.cell
def _(gaussian_kde, go, make_subplots, np, style_figure):
    def style_cohort_rows(figure, n_cohorts, n_models, metric_name, palette):
        """The forest's and the ridge's layout, a row per cohort: the shared
        style, the height, the legend naming the models (none for one model),
        `metric_name` under the last row, no y axis, and each row's title
        (its cohort) at the row's top left."""
        height = 110 * n_cohorts + 130
        style_figure(figure, palette)
        figure.update_layout(
            height=height,
            showlegend=n_models > 1,
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1 - 12 / height,
                "yanchor": "top",
                "yref": "container",
            },
            margin={"l": 30, "r": 20, "t": 80, "b": 50},
            hovermode="closest",
        )
        figure.update_xaxes(title_text=metric_name, row=n_cohorts, col=1)
        figure.update_yaxes(showgrid=False, showline=False, showticklabels=False)
        figure.update_annotations(x=0, xanchor="left")

    def forest_figure(
        scores,
        metric_name,
        lower_column,
        upper_column,
        palette,
        share_x_axis=False,
        rule_at=None,
    ):
        """A forest plot of `metric_name`: a row per cohort, the composite
        first, and in each row a dot per model (the first on top) at its
        score, the bar its interval from `lower_column` to `upper_column`.
        `scores` is long, a row per model, cohort and metric, with the columns
        `model`, `cohort`, `metric`, `score` and the two bounds; the legend
        names the models, each in one of `palette`'s categorical colors. Each
        row has its own x scale unless `share_x_axis`; `rule_at` draws a
        vertical line at that value in every row (0 for a difference)."""
        metric_scores = scores[scores["metric"] == metric_name]
        # The composite first, then the cohorts in their order.
        cohorts = sorted(
            metric_scores["cohort"].unique(), key=lambda cohort: cohort != "composite"
        )
        models = list(metric_scores["model"].unique())
        figure = make_subplots(
            rows=len(cohorts),
            cols=1,
            shared_xaxes=share_x_axis,
            vertical_spacing=0.11,
            subplot_titles=cohorts,
        )
        for row, cohort in enumerate(cohorts, start=1):
            for position, model in enumerate(models):
                color = palette["categorical"][position]
                point = metric_scores[
                    (metric_scores["cohort"] == cohort)
                    & (metric_scores["model"] == model)
                ].iloc[0]
                figure.add_trace(
                    go.Scatter(
                        x=[point["score"]],
                        y=[position],
                        mode="markers",
                        name=model,
                        legendgroup=model,
                        showlegend=row == 1,
                        marker={
                            "size": 10,
                            "color": color,
                            "line": {"width": 2, "color": palette["surface"]},
                        },
                        error_x={
                            "type": "data",
                            "symmetric": False,
                            "array": [point[upper_column] - point["score"]],
                            "arrayminus": [point["score"] - point[lower_column]],
                            "color": color,
                            "thickness": 2,
                            "width": 0,
                        },
                        customdata=[[point[lower_column], point[upper_column]]],
                        hovertemplate=(
                            f"{model}<br>{cohort} {metric_name} %{{x:.4f}}"
                            "<br>interval %{customdata[0]:.4f} to %{customdata[1]:.4f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )
            figure.update_yaxes(range=[len(models) - 0.4, -0.6], row=row, col=1)
            if rule_at is not None:
                figure.add_vline(
                    x=rule_at, line_width=1, line_color=palette["ink"], row=row, col=1
                )
        style_cohort_rows(figure, len(cohorts), len(models), metric_name, palette)
        return figure

    def ridge_figure(
        replicate_scores, metric_name, palette, share_x_axis=False, rule_at=None
    ):
        """A ridge plot of `metric_name`: a row per cohort, the composite
        first, and in each row each model's bootstrap replicates as a density
        (a Gaussian KDE between the replicates' extremes; the row's highest
        density on top). `replicate_scores` is {model: its replicates, a row
        per replicate and a column per cohort and metric}; the legend names
        the models, each in one of `palette`'s categorical colors. Each row
        has its own x scale unless `share_x_axis`; `rule_at` draws a vertical
        line at that value in every row (0 for a difference)."""
        models = list(replicate_scores)
        # The composite first, then the cohorts in their order.
        cohorts = sorted(
            replicate_scores[models[0]].columns.unique(level="cohort"),
            key=lambda cohort: cohort != "composite",
        )
        figure = make_subplots(
            rows=len(cohorts),
            cols=1,
            shared_xaxes=share_x_axis,
            vertical_spacing=0.11,
            subplot_titles=cohorts,
        )
        for row, cohort in enumerate(cohorts, start=1):
            # Each model's density by a Gaussian KDE (Scott's bandwidth) on 200
            # points between its replicates' extremes; the row's curves on one
            # height scale; a 2px line and a 10% wash.
            curves = {}
            for model in models:
                replicates = replicate_scores[model][(cohort, metric_name)]
                x_values = np.linspace(replicates.min(), replicates.max(), 200)
                curves[model] = (x_values, gaussian_kde(replicates)(x_values))
            row_highest = max(density.max() for _, density in curves.values())
            for position, model in enumerate(models):
                color = palette["categorical"][position]
                x_values, density = curves[model]
                figure.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=density / row_highest,
                        mode="lines",
                        fill="tozeroy",
                        fillcolor=color + "1a",
                        line={"width": 2, "color": color},
                        name=model,
                        legendgroup=model,
                        showlegend=row == 1,
                        hovertemplate=(
                            f"{model}<br>{cohort} {metric_name} %{{x:.4f}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )
            figure.update_yaxes(range=[0, 1.1], row=row, col=1)
            if rule_at is not None:
                figure.add_vline(
                    x=rule_at, line_width=1, line_color=palette["ink"], row=row, col=1
                )
        style_cohort_rows(figure, len(cohorts), len(models), metric_name, palette)
        return figure

    return forest_figure, ridge_figure


@app.cell
def _(METRICS, SELECTION_METRIC, mo):
    # The form of the best runs' plot: the forest (each run's score and 95%
    # interval) or the ridge (the density of each run's bootstrap
    # replicates); and its metric, any of the six, the selection metric by
    # default.
    best_runs_plot = mo.ui.radio(
        options=["forest", "ridge"], value="forest", inline=True, label="Plot"
    )
    best_runs_metric = mo.ui.dropdown(
        options=[metric.name for metric in METRICS],
        value=SELECTION_METRIC.name,
        label="Metric",
    )
    return best_runs_metric, best_runs_plot


@app.cell
def _(
    CI_BOUND_NAMES,
    PALETTE,
    best_replicate_scores,
    best_runs_metric,
    best_runs_plot,
    best_runs_val_scores_df,
    forest_figure,
    mo,
    ridge_figure,
):
    if best_runs_plot.value == "forest":
        best_runs_figure = forest_figure(
            best_runs_val_scores_df,
            best_runs_metric.value,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
            PALETTE,
        )
    elif best_runs_plot.value == "ridge":
        best_runs_figure = ridge_figure(
            best_replicate_scores, best_runs_metric.value, PALETTE
        )
    else:
        raise ValueError(f"unknown plot {best_runs_plot.value!r}")
    mo.vstack(
        [
            mo.md(
                "**Each model's best run on validation: the chosen metric per "
                "cohort and for the composite** (a row per cohort, the composite "
                "first; the dot the run's score, the bar its 95% bootstrap interval, "
                "the colors named in the legend; the ridge shows instead the density "
                "of the run's 1000 bootstrap replicates)"
            ),
            mo.hstack([best_runs_plot, best_runs_metric], justify="start"),
            best_runs_figure,
        ]
    )
    return


@app.cell
def _(METRICS, SELECTION_METRIC, mo):
    # The paired difference's own plot, the ridge first, and its own metric,
    # any of the six (the selection metric by default).
    best_difference_plot = mo.ui.radio(
        options=["forest", "ridge"], value="ridge", inline=True, label="Plot"
    )
    best_difference_metric = mo.ui.dropdown(
        options=[metric.name for metric in METRICS],
        value=SELECTION_METRIC.name,
        label="Metric",
    )
    return best_difference_metric, best_difference_plot


@app.cell
def _(
    CI_BOUND_NAMES,
    PALETTE,
    best_difference_df,
    best_difference_metric,
    best_difference_plot,
    best_difference_replicates,
    forest_figure,
    mo,
    ridge_figure,
):
    _difference_name = best_difference_df["model"].iloc[0]
    # The chosen plot and metric, one x axis for every cohort so the rule at
    # 0 lines up.
    if best_difference_plot.value == "forest":
        best_difference_figure = forest_figure(
            best_difference_df,
            best_difference_metric.value,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
            PALETTE,
            share_x_axis=True,
            rule_at=0,
        )
    elif best_difference_plot.value == "ridge":
        best_difference_figure = ridge_figure(
            {_difference_name: best_difference_replicates},
            best_difference_metric.value,
            PALETTE,
            share_x_axis=True,
            rule_at=0,
        )
    else:
        raise ValueError(f"unknown plot {best_difference_plot.value!r}")
    mo.vstack(
        [
            mo.md(
                f"**The paired difference on validation, {_difference_name}: "
                f"{best_difference_metric.value} per cohort and for the composite** "
                "(per bootstrap replicate, so the pairing counts; below 0 the total "
                "times probability model scores lower; 0 inside the 95% interval: no "
                "difference shown; for bias the sign gives the direction, not which "
                "model is better)"
            ),
            mo.hstack([best_difference_plot, best_difference_metric], justify="start"),
            best_difference_figure,
            mo.md(
                "**The difference on every metric** (for bias the sign gives the "
                "direction of the change, not which model is better)"
            ),
            mo.ui.table(best_difference_df.round(4)),
        ]
    )
    return


@app.cell
def _(SELECTION_METRIC, mo):
    run_inspection = mo.ui.run_button(label="Inspect the best runs")
    mo.md(
        f"{run_inspection} each model's best run: the permutation importance on "
        f"{SELECTION_METRIC.name} and the SHAP values per cohort (about two minutes); "
        "then the two best runs' comparison is logged to MLflow as one run (a run "
        "logged before is skipped)."
    )
    return (run_inspection,)


@app.cell
def _(COHORTS, Metric, SELECTION_METRIC, functools):
    # The permutation importance's score: the composite, and each cohort
    # scored alone (the metric's cohort=); a partial, not a lambda, so each
    # metric keeps its own cohort.
    importance_metrics = {
        "composite": SELECTION_METRIC,
        **{
            cohort: Metric(
                f"{SELECTION_METRIC.name}_{cohort}",
                functools.partial(SELECTION_METRIC.function, cohort=cohort),
                SELECTION_METRIC.direction,
            )
            for cohort in COHORTS
        },
    }
    return (importance_metrics,)


@app.cell
def _(
    AgeGroupDataset,
    n_shap_background,
    n_shap_rows,
    np,
    seed,
    train_age_group,
    val_age_group,
):
    # The rows SHAP explains and its background, drawn once, so both models are
    # explained on the same validation rows against the same training rows.
    _rng = np.random.default_rng(seed)
    _rows = _rng.choice(len(val_age_group.X), size=n_shap_rows, replace=False)
    _background = _rng.choice(
        len(train_age_group.X), size=n_shap_background, replace=False
    )
    shap_rows = AgeGroupDataset(*(field.iloc[_rows] for field in val_age_group))
    shap_background = train_age_group.X.iloc[_background]
    return shap_background, shap_rows


@app.cell
def _(
    PermutationImportance,
    ShapExplainer,
    best_runs,
    importance_metrics,
    mo,
    n_repeats,
    run_inspection,
    seed,
    shap_background,
    shap_rows,
    train_age_group,
    val_age_group,
):
    mo.stop(
        not run_inspection.value, mo.md("Click the button to inspect the best runs.")
    )
    # Permutation importance of its metric: on validation the decision
    # table, on training the overfit check. SHAP, exact at nine columns, of the
    # validation subsample against the training background; the values explain
    # the count (the rows' exposure), per row, column and cohort.
    # scikit-learn's results per model, cohort (the composite first) and split,
    # their arrays in X's column order; SHAP's explanation per model.
    best_runs_permutation, best_runs_shap = {}, {}
    for _model_name, _run in best_runs.items():
        best_runs_permutation[_model_name] = {
            cohort: {
                split: PermutationImportance(
                    metric, n_repeats, random_state=seed
                ).compute_importances(
                    _run["model"], dataset.X, dataset.Y, exposure=dataset.exposure
                )
                for split, dataset in (
                    ("train", train_age_group),
                    ("val", val_age_group),
                )
            }
            for cohort, metric in importance_metrics.items()
        }
        best_runs_shap[_model_name] = ShapExplainer(method="exact", n_jobs=-1).explain(
            _run["model"], shap_rows.X, shap_background, exposure=shap_rows.exposure
        )
    return best_runs_permutation, best_runs_shap


@app.cell
def _(go, np, style_figure):
    def permutation_figure(results, feature_names, score_name, palette):
        """The permutation importance of each split in `results` ({split:
        scikit-learn's result}, in display order): per column a box of the
        drops over the repeated shuffles, a box per split, the mean dashed;
        the columns by the last split's mean drop (the largest on top), the
        last split in `palette`'s blue, an earlier one in its muted gray; a
        dotted rule at 0, where shuffling the column changes nothing.
        `score_name` names the score that drops (the composite's or a
        cohort's metric)."""
        splits = list(results)
        # Ascending: a category axis draws its first category at the bottom.
        order = np.argsort(results[splits[-1]].importances_mean, kind="stable")
        columns = [feature_names[position] for position in order]
        figure = go.Figure()
        for split in splits:
            # A row per column, a column per repeat.
            importances = results[split].importances[order]
            figure.add_trace(
                go.Box(
                    x=importances.ravel(),
                    y=np.repeat(columns, importances.shape[1]),
                    name=split,
                    orientation="h",
                    boxmean=True,
                    marker={
                        "color": (
                            palette["categorical"][0]
                            if split == splits[-1]
                            else palette["muted"]
                        )
                    },
                    line={"width": 1.5},
                )
            )
        figure.add_vline(x=0, line={"color": palette["ink"], "width": 1, "dash": "dot"})
        style_figure(figure, palette)
        figure.update_layout(
            boxmode="group",
            height=80 + 28 * len(splits) * len(columns),
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
                "traceorder": "normal",
            },
            margin={"l": 150, "r": 20, "t": 40, "b": 50},
        )
        figure.update_yaxes(
            categoryorder="array",
            categoryarray=columns,
            tickfont={"color": palette["ink"]},
        )
        figure.update_xaxes(title_text=f"drop in the {score_name} when shuffled")
        return figure

    return (permutation_figure,)


@app.cell
def _(np, pd):
    def build_importance_table(permutation_by_split, explanation):
        """A row per raw column, in X's column order (`explanation`'s
        feature_names): the permutation importance's mean and SD over the
        repeats per split of `permutation_by_split` ({split: scikit-learn's
        result}, in its order), then the mean |SHAP| over the explained rows
        per cohort and for the composite (D18)."""
        columns = {}
        for split, result in permutation_by_split.items():
            columns[f"perm_fi_mean_{split}"] = result.importances_mean
            columns[f"perm_fi_std_{split}"] = result.importances_std
        mean_abs_shap = np.abs(explanation.values).mean(axis=0)
        for position, cohort in enumerate(explanation.output_names):
            columns[f"mean_abs_shap_{cohort}"] = mean_abs_shap[:, position]
        # Over the rows and the cohorts at once: the mean over the cohorts, up
        # to the last bit.
        columns["mean_abs_shap_composite"] = np.abs(explanation.values).mean(
            axis=(0, 2)
        )
        table = pd.DataFrame(columns, index=explanation.feature_names)
        return table.rename_axis("column").reset_index()

    return (build_importance_table,)


@app.cell
def _(
    SELECTION_METRIC,
    best_runs_permutation,
    best_runs_shap,
    best_runs_val_scores_df,
    build_importance_table,
    mo,
    pd,
):
    # The two importances side by side, per model and raw column: the permutation
    # importance on validation (the decision) and on training (the overfit
    # check), and the mean |SHAP| over the explained rows per cohort and for the
    # composite (D18). Both hold arrays in X's column order, SHAP's
    # feature_names.
    _rows = []
    for _model, _shap in best_runs_shap.items():
        _table = build_importance_table(
            best_runs_permutation[_model]["composite"], _shap
        )
        _table.insert(0, "model", _model)
        _rows.append(_table)
    importance_df = pd.concat(_rows, ignore_index=True)
    # Each model's table, its rows in X's column order, so the two compare row
    # by row.
    _views = []
    for _model, _family, _use_exposure in (
        best_runs_val_scores_df[["model", "family", "use_exposure"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        _views.append(mo.md(f"**{_model}:** `{_family}`, exposure {_use_exposure}"))
        _views.append(
            mo.ui.table(
                importance_df[importance_df["model"] == _model]
                .drop(columns="model")
                .reset_index(drop=True)
                .round(4)
            )
        )
    mo.vstack(
        [
            mo.md(
                "**Feature importance of each model's best run, per raw column** "
                "(perm_fi_mean_val: the permutation feature importance, the drop of "
                f"the composite {SELECTION_METRIC.name} on validation when the "
                "column is shuffled, the decision; perm_fi_mean_train: the same on "
                "the training rows, what the model fitted, so a column high there "
                "and near 0 on validation is overfit; perm_fi_std_train, "
                "perm_fi_std_val: the spread over the repeated shuffles; "
                "mean_abs_shap_<cohort>: how much the column moves the cohort's "
                "predicted count, the mean |SHAP| over the explained validation "
                "rows, no labels needed; mean_abs_shap_composite: their mean over "
                "the cohorts; the permutation importance per cohort is drawn below)"
            ),
            *_views,
        ]
    )
    return (importance_df,)


@app.cell
def _(COHORTS, mo):
    # The permutation figure's score: the composite or one cohort.
    permutation_cohort = mo.ui.dropdown(
        options=["composite", *COHORTS], value="composite", label="Cohort"
    )
    return (permutation_cohort,)


@app.cell
def _(
    PALETTE,
    SELECTION_METRIC,
    best_runs_permutation,
    mo,
    n_repeats,
    permutation_cohort,
    permutation_figure,
    train_age_group,
):
    _cohort = permutation_cohort.value
    permutation_figures = {
        model: permutation_figure(
            permutation[_cohort],
            list(train_age_group.X.columns),
            f"{_cohort} {SELECTION_METRIC.name}",
            PALETTE,
        )
        for model, permutation in best_runs_permutation.items()
    }
    _views = []
    for _model, _figure in permutation_figures.items():
        _views += [mo.md(f"**{_model}**"), _figure]
    mo.vstack(
        [
            mo.md(
                "**The permutation importance of each model's best run, per "
                f"column: the drop of the {_cohort} {SELECTION_METRIC.name} when "
                f"the column is shuffled** (a box over the {n_repeats} shuffles, "
                "train in gray, validation in blue; the line the median, the "
                "dashed line the mean; the columns by their validation mean)"
            ),
            permutation_cohort,
            *_views,
        ]
    )
    return (permutation_figures,)


@app.cell
def _(COHORTS, mo):
    # The beeswarm's cohort: the SHAP values explain each cohort's predicted
    # count, and each cohort orders the columns its own way.
    shap_cohort = mo.ui.dropdown(options=COHORTS, value=COHORTS[0], label="Cohort")
    return (shap_cohort,)


@app.cell
def _(go, np, pd, style_figure):
    def beeswarm_figure(explanation, X, cohort, random_state, palette):
        """A beeswarm of `cohort`'s SHAP values in `explanation` (rows,
        columns, cohorts), its rows `X`: a row of dots per column, the columns
        by the cohort's mean |SHAP| (the largest on top), each dot a row at its
        SHAP value, colored by the column's value, low to high over `X`
        (clipped to its 5th–95th percentile) on `palette`'s shap scale, blue
        to red; a category's levels (unordered) each in its own
        row, in the muted gray: a row holds one level, so its value has no low
        or high. `random_state` seeds the dots' vertical jitter."""
        # One jitter, so each row of dots spreads the same way in every figure.
        jitter = np.random.default_rng(random_state).uniform(-0.3, 0.3, len(X))
        values = explanation.values[:, :, explanation.output_names.index(cohort)]
        # The columns by the cohort's mean |SHAP|, the largest on top; a
        # numeric column one row, a category a row per level, kept together
        # at the column's place: (label, column, its rows, whether a level).
        order = np.argsort(-np.abs(values).mean(axis=0), kind="stable")
        rows = []
        for column_position in order:
            column = explanation.feature_names[column_position]
            raw = X[column]
            if pd.api.types.is_numeric_dtype(raw):
                rows.append((column, column_position, np.full(len(raw), True), False))
            else:
                for level in sorted(raw.unique()):
                    rows.append(
                        (
                            f"{column} = {level}",
                            column_position,
                            (raw == level).to_numpy(),
                            True,
                        )
                    )
        figure = go.Figure()
        scale_shown = False
        for position, (label, column_position, mask, is_level) in enumerate(rows):
            column = explanation.feature_names[column_position]
            raw = X[column][mask]
            if not is_level:
                # The value clipped to the column's 5th–95th percentile over
                # the rows, then scaled to that range, so an outlier does not
                # wash out the rest; a constant column divides by 1.
                low, high = np.percentile(raw, [5, 95])
                marker = {
                    "color": (
                        (raw.clip(low, high) - low) / ((high - low) or 1)
                    ).to_numpy(),
                    "colorscale": palette["shap"],
                    "cmin": 0,
                    "cmax": 1,
                    "showscale": not scale_shown,
                    "colorbar": {
                        "title": {"text": "value"},
                        "thickness": 12,
                        "tickvals": [0, 1],
                        "ticktext": ["low", "high"],
                    },
                }
                scale_shown = True
                value_text = "%{customdata:.4g}"
            else:
                marker = {"color": palette["muted"]}
                value_text = "%{customdata}"
            figure.add_trace(
                go.Scatter(
                    x=values[mask, column_position],
                    y=len(rows) - 1 - position + jitter[mask],
                    mode="markers",
                    name=label,
                    showlegend=False,
                    marker={"size": 6, "opacity": 0.8, **marker},
                    customdata=raw.to_numpy(),
                    hovertemplate=(
                        f"{column} {value_text}<br>SHAP %{{x:.3f}}<extra></extra>"
                    ),
                )
            )
        style_figure(figure, palette)
        figure.update_layout(
            height=60 + 45 * len(rows),
            margin={"l": 190, "r": 20, "t": 20, "b": 50},
            hovermode="closest",
        )
        figure.update_yaxes(
            tickvals=list(range(len(rows)))[::-1],
            ticktext=[row[0] for row in rows],
            tickfont={"color": palette["ink"]},
        )
        figure.update_xaxes(
            title_text=f"SHAP value (children, {cohort})",
            zeroline=True,
            zerolinecolor=palette["axis"],
        )
        return figure

    return (beeswarm_figure,)


@app.cell
def _(PALETTE, beeswarm_figure, best_runs_shap, mo, seed, shap_cohort, shap_rows):
    _cohort = shap_cohort.value
    shap_beeswarm_figures = {
        model: beeswarm_figure(explanation, shap_rows.X, _cohort, seed, PALETTE)
        for model, explanation in best_runs_shap.items()
    }
    _views = []
    for _model, _figure in shap_beeswarm_figures.items():
        _views += [mo.md(f"**{_model}**"), _figure]
    mo.vstack(
        [
            mo.md(
                f"**The SHAP values of {_cohort} per explained validation row** "
                "(each dot a row: how much the column moves its predicted count "
                "from the mean prediction over the background; the columns ordered "
                "by their mean |SHAP| for the cohort, the largest on top; the color "
                "the column's value, low (blue) to high (red), clipped at its 5th "
                "and 95th percentiles over the explained rows; a category a row "
                "per level, in gray)"
            ),
            shap_cohort,
            *_views,
        ]
    )
    return (shap_beeswarm_figures,)


@app.cell
def _():
    def build_dataset_figures(
        dataset_name,
        scores_df,
        replicate_scores,
        difference_df,
        difference_replicates,
        permutation_results,
        explanations,
        X,
        metric_name,
        bound_names,
        seed,
        palette,
        forest_figure,
        ridge_figure,
        permutation_figure,
        beeswarm_figure,
    ):
        """The 18 figures of one dataset (`dataset_name`: "val" or "test") by artifact
        path: the forest and the ridge of `metric_name` for the two runs and for
        their difference; per model the permutation figure of the composite and each
        cohort and the beeswarm of each cohort. `permutation_results` is {model:
        {cohort (the composite first): {split: result}}}."""
        difference_name = difference_df["model"].iloc[0]
        figures = {
            f"figures/{dataset_name}_forest_runs_{metric_name}.html": forest_figure(
                scores_df,
                metric_name,
                bound_names["lower"],
                bound_names["upper"],
                palette,
            ),
            f"figures/{dataset_name}_ridge_runs_{metric_name}.html": ridge_figure(
                replicate_scores, metric_name, palette
            ),
            f"figures/{dataset_name}_forest_difference_{metric_name}.html": forest_figure(
                difference_df,
                metric_name,
                bound_names["lower"],
                bound_names["upper"],
                palette,
                share_x_axis=True,
                rule_at=0,
            ),
            f"figures/{dataset_name}_ridge_difference_{metric_name}.html": ridge_figure(
                {difference_name: difference_replicates},
                metric_name,
                palette,
                share_x_axis=True,
                rule_at=0,
            ),
        }
        for model, by_cohort in permutation_results.items():
            for cohort, by_split in by_cohort.items():
                figures[f"figures/{dataset_name}_permutation_{model}_{cohort}.html"] = (
                    permutation_figure(
                        by_split, list(X.columns), f"{cohort} {metric_name}", palette
                    )
                )
            for cohort in explanations[model].output_names:
                figures[f"figures/{dataset_name}_beeswarm_{model}_{cohort}.html"] = (
                    beeswarm_figure(explanations[model], X, cohort, seed, palette)
                )
        return figures

    return (build_dataset_figures,)


@app.cell
def _(mlflow):
    def log_evaluation_run(
        run_name,
        tags,
        settings,
        dataset_name,
        scores_df,
        difference_df,
        bound_names,
        tables,
        figures,
        best_runs,
        study_folder,
        dataset_inputs,
    ):
        """One run, in the active experiment, of the two best runs evaluated on
        `dataset_name` ("val" or "test"); its number of metrics. It logs `tags`,
        `study_folder` (the folder of the best runs' studies) as a tag, each best run's
        study name as `best.<model>` and its tuning run's id as `best.<model>.run_id`
        (`best_runs` is the read's record, {model: its study_name, run_id,
        best_params and components}); `settings` (the run's complete settings) as
        `settings.json`, six of them as parameters; as metrics, the
        composite's score and its interval's bounds per model (`scores_df`) and for
        their paired difference (`difference_df`), named
        `<dataset_name>/<model or difference>/<metric>/<statistic>`, every cohort's rows
        and the other statistics being in the tables, logged last (a run that fails
        while logging holds none); each of `tables` ({name: df}) as
        `tables/<name>.csv` at 6 significant digits (the UI shows it as a table);
        each of `figures` ({artifact path: figure}); per best run its best
        parameters (what `from_params` reads) and its components' declaration;
        each of `dataset_inputs` ({context: MLflow dataset}) as an input."""
        # The score and its interval: the mean, median and SD are in the tables.
        statistics = ["score", bound_names["lower"], bound_names["upper"]]
        composite_scores = scores_df[scores_df["cohort"] == "composite"]
        composite_difference = difference_df[difference_df["cohort"] == "composite"]
        metrics = {
            **{
                f"{dataset_name}/{row['model']}/{row['metric']}/{statistic}": row[
                    statistic
                ]
                for row in composite_scores.to_dict("records")
                for statistic in statistics
            },
            **{
                f"{dataset_name}/difference/{row['metric']}/{statistic}": row[statistic]
                for row in composite_difference.to_dict("records")
                for statistic in statistics
            },
        }
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    **tags,
                    "study_folder": study_folder,
                    **{
                        f"best.{model_name}": best_run["study_name"]
                        for model_name, best_run in best_runs.items()
                    },
                    **{
                        f"best.{model_name}.run_id": best_run["run_id"]
                        for model_name, best_run in best_runs.items()
                    },
                }
            )
            # settings.json holds every setting; the parameters are the few a chart's
            # axis or a filter uses.
            mlflow.log_params(
                {
                    name: settings[name]
                    for name in (
                        "config_hash",
                        "seed",
                        "split_method",
                        "n_trials",
                        "objective_metric",
                        "selection_metric",
                    )
                }
            )
            mlflow.log_dict(settings, "settings.json")
            for name, df in tables.items():
                mlflow.log_text(
                    df.to_csv(index=False, float_format="%.6g"), f"tables/{name}.csv"
                )
            # log_figure has no run_id: it needs the active run.
            for path, figure in figures.items():
                mlflow.log_figure(figure, path)
            for model_name, best_run in best_runs.items():
                mlflow.log_dict(
                    best_run["best_params"], f"best_params/{model_name}.json"
                )
                mlflow.log_dict(
                    {
                        component.name: component.to_dict()
                        for component in best_run["components"]
                    },
                    f"components/{model_name}.json",
                )
            for context, mlflow_dataset in dataset_inputs.items():
                mlflow.log_input(mlflow_dataset, context=context)
            # The metrics last: a run that fails while logging holds none, so it
            # never shows in a chart.
            mlflow.log_metrics(metrics)
        return len(metrics)

    return (log_evaluation_run,)


@app.cell
def _(
    CI_BOUND_NAMES,
    OBJECTIVE_METRIC,
    PALETTE,
    SELECTION_METRIC,
    beeswarm_figure,
    best_difference_df,
    best_difference_replicates,
    best_replicate_scores,
    best_runs,
    best_runs_permutation,
    best_runs_shap,
    best_runs_val_scores_df,
    build_dataset_figures,
    code_tags,
    forest_figure,
    importance_df,
    log_evaluation_run,
    mlflow,
    mo,
    permutation_figure,
    ridge_figure,
    seed,
    shap_rows,
    shared_params,
    study_folder,
    tuning_dataset_inputs,
):
    # Runs once "Inspect the best runs" has produced what it reads. One such
    # run per tuning setup, the objective and the selection metric (each
    # changes the best runs): a FINISHED run with the key is kept (a FAILED one is
    # logged again).
    _run_key = (
        f"{study_folder}/best_runs__objective={OBJECTIVE_METRIC.name}"
        f"__selection={SELECTION_METRIC.name}"
    )
    mo.stop(
        bool(
            mlflow.search_runs(
                filter_string=f"tags.run_key = '{_run_key}' and attributes.status = 'FINISHED'",
                output_format="list",
            )
        ),
        mo.md("The best runs' run is logged: skipped."),
    )
    # The 3 tables and the validation part's 18 figures.
    _tables = {
        "best_runs_val_scores": best_runs_val_scores_df,
        "best_difference": best_difference_df,
        "importance": importance_df,
    }
    _figures = build_dataset_figures(
        "val",
        best_runs_val_scores_df,
        best_replicate_scores,
        best_difference_df,
        best_difference_replicates,
        best_runs_permutation,
        best_runs_shap,
        shap_rows.X,
        SELECTION_METRIC.name,
        CI_BOUND_NAMES,
        seed,
        PALETTE,
        forest_figure,
        ridge_figure,
        permutation_figure,
        beeswarm_figure,
    )
    _n_metrics = log_evaluation_run(
        "best runs",
        {"run_key": _run_key, "stage": "best_runs", **code_tags},
        shared_params,
        "val",
        best_runs_val_scores_df,
        best_difference_df,
        CI_BOUND_NAMES,
        _tables,
        _figures,
        best_runs,
        study_folder,
        tuning_dataset_inputs,
    )
    mo.md(
        f"Logged the best runs' run: {_n_metrics} metrics, {len(_tables)} tables, "
        f"{len(_figures)} figures."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) The test evaluation

    The test part stands for production (E11): each model's best run, chosen
    on validation, is refit on the training and validation parts together,
    then the test part is predicted once and scored, with the bootstrap over
    its neighborhoods. Validation stays the selection evidence; the test
    score is reported, never used to choose.

    The test predictions are then read (the menus): each model's best or
    worst test buildings by a metric over each building's own cells (one
    cohort, or the composite), with their true and predicted counts and
    their features; one model's true and predicted counts per building; and,
    for all the test buildings or one neighborhood, the children, the cohort
    shares and the scores, true and of both models.

    The same click explains the refit models on the test part, after the
    test scores: the permutation importance and every test building's SHAP
    values, a table per model, and a beeswarm per cohort. Under the best and
    worst buildings, a waterfall per model shows one building at a chosen
    rank, from its mean prediction over the background to its prediction, a
    step per column. Reported, never used to choose.
    """)
    return


@app.cell
def _(mo):
    run_test_evaluation = mo.ui.run_button(label="Evaluate on the test part")
    mo.md(
        f"{run_test_evaluation} each model's best run: refit on the training and "
        "validation parts, scored once on the test part, and explained there (the "
        "permutation importance on section 5's metric and the SHAP values of every "
        "test building; a minute or two more); then the final run is logged to "
        "MLflow, once per experiment (a second one is refused)."
    )
    return (run_test_evaluation,)


@app.cell
def _(
    AgeGroupDataset,
    CI_BOUND_NAMES,
    COHORTS,
    METRICS,
    PREDICTION_FLOOR,
    alpha,
    best_runs,
    best_runs_val_scores_df,
    bootstrap_evaluator,
    build_independent_cohorts_model,
    build_paired_difference,
    build_total_times_probability_model,
    compute_metrics,
    mo,
    pd,
    run_test_evaluation,
    seed,
    summarize_replicates,
    test_age_group,
    train_age_group,
    val_age_group,
):
    mo.stop(
        not run_test_evaluation.value,
        mo.md("Click the button to evaluate the best runs on the test part."),
    )
    # The training and validation parts together, part by part.
    train_val_age_group = AgeGroupDataset(
        *(
            pd.concat([train_part, val_part])
            for train_part, val_part in zip(train_age_group, val_age_group, strict=True)
        )
    )
    # Each model's best run, chosen on validation, refit on them from its
    # components; the test part predicted once and scored, every metric per
    # cohort and for the composite, with the bootstrap over its
    # neighborhoods (one evaluator: the two models' replicates pair).
    test_models, test_predictions, test_replicate_scores, _scores_by_run = (
        {},
        {},
        {},
        {},
    )
    for _model_name, _run in best_runs.items():
        if _model_name == "independent_cohorts":
            _model = build_independent_cohorts_model(
                _run["components"], y_min=PREDICTION_FLOOR, random_state=seed
            )
        elif _model_name == "total_times_probability":
            _model = build_total_times_probability_model(
                *_run["components"], y_min=PREDICTION_FLOOR, random_state=seed
            )
        else:
            raise ValueError(f"unknown model {_model_name!r}")
        _model.fit(
            train_val_age_group.X,
            train_val_age_group.Y,
            exposure=train_val_age_group.exposure,
        )
        # Kept for the explanation of the test buildings.
        test_models[_model_name] = _model
        test_predictions[_model_name] = _model.predict(
            test_age_group.X, exposure=test_age_group.exposure
        )
        _replicate_scores = bootstrap_evaluator.evaluate(
            test_age_group.Y,
            test_predictions[_model_name],
            test_age_group.groups,
            metrics=METRICS,
            cohorts=[*COHORTS, "composite"],
        )
        test_replicate_scores[_model_name] = _replicate_scores
        _scores_by_run[_model_name, _run["family"], _run["use_exposure"]] = (
            summarize_replicates(
                compute_metrics(
                    test_age_group.Y,
                    test_predictions[_model_name],
                    METRICS,
                    cohorts=[*COHORTS, "composite"],
                ).stack(),
                _replicate_scores,
                alpha,
                CI_BOUND_NAMES,
            )
        )
    # A row per best run, cohort and metric, as on validation.
    test_scores_df = pd.concat(
        _scores_by_run, names=["model", "family", "use_exposure"]
    ).reset_index()
    # The two runs' paired difference per test replicate, as on validation.
    test_difference_replicates, test_difference_df = build_paired_difference(
        test_scores_df, test_replicate_scores, alpha, CI_BOUND_NAMES
    )
    # Each model's composite score per metric on validation (refit on the
    # training part: the selection evidence, optimistic) and on test (refit on
    # the training and validation parts: the fair score).
    val_against_test_df = pd.DataFrame(
        {
            _dataset: _scores[_scores["cohort"] == "composite"].set_index(
                ["model", "metric"]
            )["score"]
            for _dataset, _scores in (
                ("val", best_runs_val_scores_df),
                ("test", test_scores_df),
            )
        }
    )
    val_against_test_df = val_against_test_df.reset_index()
    mo.md(
        "**Each model's best run refit on the training and validation parts "
        f"({len(train_val_age_group.X)} rows) and scored once on the test part "
        f"({len(test_age_group.X)} rows):** "
        + "; ".join(
            f"{model} `{family}`, exposure {use_exposure}"
            for model, family, use_exposure in _scores_by_run
        )
    )
    return (
        test_difference_df,
        test_difference_replicates,
        test_models,
        test_predictions,
        test_replicate_scores,
        test_scores_df,
        train_val_age_group,
        val_against_test_df,
    )


@app.cell
def _(METRICS, SELECTION_METRIC, mo):
    # The test section's plot and metric: the forest first, any of the six
    # metrics (the selection metric by default).
    test_plot = mo.ui.radio(
        options=["forest", "ridge"], value="forest", inline=True, label="Plot"
    )
    test_metric = mo.ui.dropdown(
        options=[metric.name for metric in METRICS],
        value=SELECTION_METRIC.name,
        label="Metric",
    )
    return test_metric, test_plot


@app.cell
def _(
    CI_BOUND_NAMES,
    PALETTE,
    forest_figure,
    mo,
    ridge_figure,
    test_difference_df,
    test_difference_replicates,
    test_metric,
    test_plot,
    test_replicate_scores,
    test_scores_df,
    val_against_test_df,
):
    _difference_name = test_difference_df["model"].iloc[0]
    if test_plot.value == "forest":
        test_runs_figure = forest_figure(
            test_scores_df,
            test_metric.value,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
            PALETTE,
        )
        test_difference_figure = forest_figure(
            test_difference_df,
            test_metric.value,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
            PALETTE,
            share_x_axis=True,
            rule_at=0,
        )
    elif test_plot.value == "ridge":
        test_runs_figure = ridge_figure(
            test_replicate_scores, test_metric.value, PALETTE
        )
        test_difference_figure = ridge_figure(
            {_difference_name: test_difference_replicates},
            test_metric.value,
            PALETTE,
            share_x_axis=True,
            rule_at=0,
        )
    else:
        raise ValueError(f"unknown plot {test_plot.value!r}")
    mo.vstack(
        [
            mo.md(
                "**Each model's best run on the test part: the chosen metric per "
                "cohort and for the composite** (refit on the training and "
                "validation parts; the dot the test score, the bar its 95% "
                "bootstrap interval over the test neighborhoods; the ridge the "
                "replicates' density)"
            ),
            mo.hstack([test_plot, test_metric], justify="start"),
            test_runs_figure,
            mo.md(
                f"**Their paired difference on the test part, {_difference_name}** "
                "(below 0 the total times probability model scores lower; 0 inside "
                "the interval: no difference shown; for bias the sign gives the "
                "direction, not which model is better)"
            ),
            test_difference_figure,
            mo.md(
                "**Validation against test, the composite** (validation: refit on "
                "the training part, the selection evidence; test: refit on the "
                "training and validation parts, scored once)"
            ),
            mo.ui.table(val_against_test_df.round(4)),
            mo.md("**The test scores, per cohort and metric**"),
            mo.ui.table(test_scores_df.round(4)),
            mo.md("**The paired difference on the test part, per cohort and metric**"),
            mo.ui.table(test_difference_df.round(4)),
        ]
    )
    return


@app.cell
def _(
    PermutationImportance,
    ShapExplainer,
    importance_metrics,
    mo,
    n_repeats,
    n_shap_background,
    np,
    run_test_evaluation,
    seed,
    test_age_group,
    test_models,
    train_val_age_group,
):
    mo.stop(not run_test_evaluation.value)
    # The same click explains the refit models on every test building:
    # SHAP against a background drawn from the rows they were fitted on
    # (training and validation), and the permutation importance on the test
    # part. Reported, never used to choose (E11).
    _background = train_val_age_group.X.iloc[
        np.random.default_rng(seed).choice(
            len(train_val_age_group.X), size=n_shap_background, replace=False
        )
    ]
    test_shap, test_permutation = {}, {}
    for _model_name, _model in test_models.items():
        test_shap[_model_name] = ShapExplainer(method="exact", n_jobs=-1).explain(
            _model, test_age_group.X, _background, exposure=test_age_group.exposure
        )
        # scikit-learn's results per cohort (the composite first), their arrays
        # in X's column order.
        test_permutation[_model_name] = {
            cohort: PermutationImportance(
                metric, n_repeats, random_state=seed
            ).compute_importances(
                _model,
                test_age_group.X,
                test_age_group.Y,
                exposure=test_age_group.exposure,
            )
            for cohort, metric in importance_metrics.items()
        }
    return test_permutation, test_shap


@app.cell
def _(
    SELECTION_METRIC,
    build_importance_table,
    mo,
    pd,
    test_permutation,
    test_scores_df,
    test_shap,
):
    # As section 5's tables, on the test part: per model and raw column, the
    # permutation importance's mean and spread, and the mean |SHAP| over the
    # test buildings per cohort and for the composite.
    _tables = []
    for _model, _explanation in test_shap.items():
        _table = build_importance_table(
            {"test": test_permutation[_model]["composite"]}, _explanation
        )
        _table.insert(0, "model", _model)
        _tables.append(_table)
    test_importance_df = pd.concat(_tables, ignore_index=True)
    # Each model's table, headed by its run, its rows in X's column order.
    _views = []
    for _model, _family, _use_exposure in (
        test_scores_df[["model", "family", "use_exposure"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        _views.append(mo.md(f"**{_model}:** `{_family}`, exposure {_use_exposure}"))
        _views.append(
            mo.ui.table(
                test_importance_df[test_importance_df["model"] == _model]
                .drop(columns="model")
                .reset_index(drop=True)
                .round(4)
            )
        )
    mo.vstack(
        [
            mo.md(
                "**Feature importance of each model's refit best run on the test "
                "part, per raw column** (the models refit on the training and "
                "validation parts, not section 5's; perm_fi_mean_test: the drop of "
                f"the composite {SELECTION_METRIC.name} on the test part when the "
                "column is shuffled; perm_fi_std_test: its spread over the "
                "repeated shuffles; mean_abs_shap_<cohort>: the mean |SHAP| over "
                "all the test buildings; mean_abs_shap_composite: their mean over "
                "the cohorts; reported, never used to choose; the permutation "
                "importance per cohort is drawn below)"
            ),
            *_views,
        ]
    )
    return (test_importance_df,)


@app.cell
def _(COHORTS, mo):
    # The test permutation figure's score: the composite or one cohort.
    test_permutation_cohort = mo.ui.dropdown(
        options=["composite", *COHORTS], value="composite", label="Cohort"
    )
    return (test_permutation_cohort,)


@app.cell
def _(
    PALETTE,
    SELECTION_METRIC,
    mo,
    n_repeats,
    permutation_figure,
    test_age_group,
    test_permutation,
    test_permutation_cohort,
):
    _cohort = test_permutation_cohort.value
    test_permutation_figures = {
        model: permutation_figure(
            {"test": permutation[_cohort]},
            list(test_age_group.X.columns),
            f"{_cohort} {SELECTION_METRIC.name}",
            PALETTE,
        )
        for model, permutation in test_permutation.items()
    }
    _views = []
    for _model, _figure in test_permutation_figures.items():
        _views += [mo.md(f"**{_model}**"), _figure]
    mo.vstack(
        [
            mo.md(
                "**The permutation importance of each refit model on the test part, "
                f"per column: the drop of the {_cohort} {SELECTION_METRIC.name} "
                f"when the column is shuffled** (a box over the {n_repeats} "
                "shuffles; the line the median, the dashed line the mean; reported, "
                "never used to choose)"
            ),
            test_permutation_cohort,
            *_views,
        ]
    )
    return (test_permutation_figures,)


@app.cell
def _(COHORTS, mo):
    # The test beeswarm's cohort: SHAP explains each cohort's count.
    test_beeswarm_cohort = mo.ui.dropdown(
        options=COHORTS, value=COHORTS[0], label="Cohort"
    )
    return (test_beeswarm_cohort,)


@app.cell
def _(
    PALETTE,
    beeswarm_figure,
    mo,
    seed,
    test_age_group,
    test_beeswarm_cohort,
    test_shap,
):
    _cohort = test_beeswarm_cohort.value
    test_beeswarm_figures = {
        model: beeswarm_figure(explanation, test_age_group.X, _cohort, seed, PALETTE)
        for model, explanation in test_shap.items()
    }
    _views = []
    for _model, _figure in test_beeswarm_figures.items():
        _views += [mo.md(f"**{_model}**"), _figure]
    mo.vstack(
        [
            mo.md(
                f"**The SHAP values of {_cohort} per test building** (the refit "
                "models; each dot a building: how much the column moves its "
                "predicted count from the mean prediction over the background; "
                "the columns ordered by their mean |SHAP| for the cohort, the "
                "largest on top; the color the column's value, low (blue) to high "
                "(red), clipped at its 5th and 95th percentiles over the test "
                "buildings; a category a row per level, in gray)"
            ),
            test_beeswarm_cohort,
            *_views,
        ]
    )
    return (test_beeswarm_figures,)


@app.cell
def _(COHORTS, SELECTION_METRIC, mo, test_age_group):
    # The reading of the test predictions. The tables: each model's best or
    # worst k buildings by a metric over each building's own cells, one cohort
    # or the composite (its three cohorts); the metric a count loss finite on
    # every building (wmape divides by 0 when a cohort has no child, bias is
    # signed, log loss needs a child). The count figures: the model and the
    # cohort (the composite: the total). The statistics: all the test
    # buildings, or one neighborhood's.
    RANKING_METRIC_NAMES = ["rmse", "mae", "poisson_deviance"]
    test_reading_end = mo.ui.radio(
        options=["best", "worst"], value="worst", inline=True, label="Show"
    )
    test_reading_cohort = mo.ui.dropdown(
        options=[*COHORTS, "composite"], value="composite", label="Cohort"
    )
    test_reading_metric = mo.ui.dropdown(
        options=RANKING_METRIC_NAMES, value=SELECTION_METRIC.name, label="Rank by"
    )
    test_reading_k = mo.ui.number(start=1, stop=50, value=5, label="k")
    test_reading_model = mo.ui.dropdown(
        options=["independent_cohorts", "total_times_probability"],
        value="independent_cohorts",
        label="Model",
    )
    test_figures_cohort = mo.ui.dropdown(
        options=[*COHORTS, "composite"], value="composite", label="Cohort"
    )
    test_neighborhood = mo.ui.dropdown(
        options=["all", *sorted(test_age_group.groups.unique())],
        value="all",
        searchable=True,
        label="Neighborhood",
    )
    return (
        RANKING_METRIC_NAMES,
        test_figures_cohort,
        test_neighborhood,
        test_reading_cohort,
        test_reading_end,
        test_reading_k,
        test_reading_metric,
        test_reading_model,
    )


@app.cell
def _(
    COHORTS,
    METRICS,
    RANKING_METRIC_NAMES,
    compute_metrics,
    pd,
    test_age_group,
    test_predictions,
):
    # Each test building as each model predicts it: its ids and features, the
    # true and the predicted count per cohort, and the totals.
    _ranking_metrics = [
        metric for metric in METRICS if metric.name in RANKING_METRIC_NAMES
    ]
    _buildings, _scores = [], []
    for _model_name, _predictions in test_predictions.items():
        _table = pd.DataFrame(
            {
                "model": _model_name,
                "building_id": test_age_group.ids,
                "neighborhood_id": test_age_group.groups,
            }
        ).join(test_age_group.X)
        for _cohort in COHORTS:
            _table[f"{_cohort}_true"] = test_age_group.Y[_cohort]
            _table[f"{_cohort}_predicted"] = _predictions[_cohort]
        _table["total_true"] = test_age_group.Y.sum(axis=1)
        _table["total_predicted"] = _predictions.sum(axis=1)
        _buildings.append(_table)
        # Each building scored on its own cells: each cohort's, and the
        # composite's (its three cohorts).
        for _position, _building_id in enumerate(test_age_group.ids):
            _scores.append(
                compute_metrics(
                    test_age_group.Y.iloc[[_position]],
                    _predictions.iloc[[_position]],
                    _ranking_metrics,
                    cohorts=[*COHORTS, "composite"],
                )
                .reset_index()
                .assign(model=_model_name, building_id=_building_id)
            )
    test_buildings_df = pd.concat(_buildings, ignore_index=True)
    test_building_scores_df = pd.concat(_scores, ignore_index=True)[
        ["model", "building_id", "cohort", *RANKING_METRIC_NAMES]
    ]
    return test_building_scores_df, test_buildings_df


@app.cell
def _(
    COHORTS,
    RANKING_METRIC_NAMES,
    mo,
    test_building_scores_df,
    test_buildings_df,
    test_reading_cohort,
    test_reading_end,
    test_reading_k,
    test_reading_metric,
):
    _end, _cohort, _metric, _k = (
        test_reading_end.value,
        test_reading_cohort.value,
        test_reading_metric.value,
        test_reading_k.value,
    )
    _counts = [
        "neighborhood_id",
        *[f"{cohort}_{kind}" for cohort in COHORTS for kind in ("true", "predicted")],
    ]
    _cohort_text = (
        "the composite (its three cohorts)" if _cohort == "composite" else _cohort
    )
    # Per model: its buildings in a residual table's order (the ids, the true
    # and the predicted counts, then the scores derived from them, on the
    # chosen cohort), worst first; its best or worst k with their features.
    test_ranked_buildings, _chosen_views, _every_views = {}, [], []
    for _model in ("independent_cohorts", "total_times_probability"):
        _buildings = (
            test_buildings_df[test_buildings_df["model"] == _model]
            .drop(columns="model")
            .set_index("building_id")
        )
        # The features, with the 3-room share (the one the shares leave out:
        # the four sum to 1) before the 4-room share.
        _features = _buildings.drop(columns=[*_counts, "total_true", "total_predicted"])
        _features.insert(
            _features.columns.get_loc("4_rooms_share"),
            "3_rooms_share",
            1
            - _features[["4_rooms_share", "5_rooms_share", "6_rooms_share"]].sum(
                axis=1
            ),
        )
        _ranked = (
            _buildings[_counts]
            .join(
                test_building_scores_df[
                    (test_building_scores_df["model"] == _model)
                    & (test_building_scores_df["cohort"] == _cohort)
                ].set_index("building_id")[RANKING_METRIC_NAMES]
            )
            .sort_values(_metric, ascending=False, kind="stable")
        )
        if _end == "best":
            _chosen = _ranked.nsmallest(_k, _metric)
        elif _end == "worst":
            _chosen = _ranked.nlargest(_k, _metric)
        else:
            raise ValueError(f"unknown end {_end!r}")
        test_ranked_buildings[_model] = (
            _chosen[[*_counts, _metric]].join(_features).reset_index()
        )
        _chosen_views += [
            mo.md(f"**{_model}: its {_end} {_k}**"),
            mo.ui.table(test_ranked_buildings[_model].round(4)),
        ]
        _every_views += [
            mo.md(f"**{_model}: every test building, worst first**"),
            mo.ui.table(_ranked.reset_index().round(4)),
        ]
    mo.vstack(
        [
            mo.md(
                f"**Each model's {_end} {_k} test buildings by their {_metric} on "
                f"{_cohort_text}** (each model ranks its own; the score over the "
                "building's own cells; each row the true and the predicted counts, "
                "the score, then the features)"
            ),
            mo.hstack(
                [
                    test_reading_end,
                    test_reading_cohort,
                    test_reading_metric,
                    test_reading_k,
                ],
                justify="start",
            ),
            *_chosen_views,
            mo.md(
                "**Every test building** (the true and the predicted counts, then "
                f"the three scores on {_cohort_text})"
            ),
            *_every_views,
        ]
    )
    return (test_ranked_buildings,)


@app.cell
def _(go, np, pd, style_figure):
    def waterfall_figure(
        explanation, age_group_dataset, predictions, building_id, cohort, palette
    ):
        """The waterfall of the building `building_id` for `cohort`, or for the
        total when `cohort` is "composite" (the cohorts' values and bases
        summed): from the building's base (the mean prediction over the
        background, at its number of apartments) a step per column to its
        prediction, the largest |SHAP| on top; red raises the prediction, blue
        lowers it. `explanation` explains the rows of `age_group_dataset`, and
        `predictions` holds one model's predicted counts of those rows."""
        cohorts = explanation.output_names if cohort == "composite" else [cohort]
        count_name = "the total" if cohort == "composite" else cohort
        # The building's row, found by its id.
        position = age_group_dataset.ids.tolist().index(building_id)
        outputs = [explanation.output_names.index(name) for name in cohorts]
        values = explanation.values[position][:, outputs].sum(axis=1)
        base = explanation.base_values[position, outputs].sum()
        prediction = predictions.iloc[position][cohorts].sum()
        true = age_group_dataset.Y.iloc[position][cohorts].sum()
        order = np.argsort(-np.abs(values), kind="stable")
        labels = []
        for j in order:
            raw = age_group_dataset.X.iloc[position, j]
            labels.append(
                f"{explanation.feature_names[j]} = "
                + (f"{raw:.3g}" if pd.api.types.is_number(raw) else str(raw))
            )
        figure = go.Figure(
            go.Waterfall(
                orientation="h",
                base=base,
                measure=["relative"] * len(order),
                y=labels,
                x=values[order],
                text=[f"{value:+.2f}" for value in values[order]],
                textposition="outside",
                increasing={"marker": {"color": palette["raises"]}},
                decreasing={"marker": {"color": palette["lowers"]}},
                connector={"line": {"color": palette["axis"], "width": 1}},
                hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
            )
        )
        for x, text, place in (
            (base, f"mean prediction {base:.2f}", "bottom left"),
            (prediction, f"predicted {prediction:.2f}", "top right"),
        ):
            figure.add_vline(
                x=x,
                line={"color": palette["ink"], "dash": "dot", "width": 1},
                annotation_text=text,
                annotation_position=place,
            )
        style_figure(figure, palette)
        figure.update_layout(
            height=60 + 34 * (len(order) + 2),
            margin={"l": 200, "r": 40, "t": 60, "b": 50},
            title={
                "text": (
                    f"{building_id} ({age_group_dataset.groups.iloc[position]}), "
                    f"{count_name}: true {true}, predicted {prediction:.2f}"
                ),
                "x": 0,
            },
        )
        figure.update_yaxes(autorange="reversed", tickfont={"color": palette["ink"]})
        figure.update_xaxes(title_text=f"children ({count_name})")
        return figure

    return (waterfall_figure,)


@app.cell
def _(COHORTS, mo):
    # The waterfalls' own menus: the best or the worst buildings, and the
    # cohort they are ranked on and explained for (the composite: the total).
    test_waterfall_end = mo.ui.radio(
        options=["best", "worst"], value="worst", inline=True, label="Show"
    )
    test_waterfall_cohort = mo.ui.dropdown(
        options=[*COHORTS, "composite"], value=COHORTS[0], label="Cohort"
    )
    return test_waterfall_end, test_waterfall_cohort


@app.cell
def _(
    mo,
    test_building_scores_df,
    test_reading_k,
    test_reading_metric,
    test_waterfall_cohort,
    test_waterfall_end,
):
    # Per model its best or worst k buildings on the waterfalls' cohort, by
    # the tables' metric and k, ranked as the tables rank theirs; a dropdown
    # each, the first by default.
    _end, _cohort = test_waterfall_end.value, test_waterfall_cohort.value
    _metric, _k = test_reading_metric.value, test_reading_k.value
    _menus = {}
    for _model in ("independent_cohorts", "total_times_probability"):
        _scores = (
            test_building_scores_df[
                (test_building_scores_df["model"] == _model)
                & (test_building_scores_df["cohort"] == _cohort)
            ]
            .set_index("building_id")[[_metric]]
            .sort_values(_metric, ascending=False, kind="stable")
        )
        if _end == "best":
            _chosen = _scores.nsmallest(_k, _metric)
        elif _end == "worst":
            _chosen = _scores.nlargest(_k, _metric)
        else:
            raise ValueError(f"unknown end {_end!r}")
        _options = {
            f"{rank}: {building_id}": building_id
            for rank, building_id in enumerate(_chosen.index, start=1)
        }
        _menus[_model] = mo.ui.dropdown(
            options=_options, value=next(iter(_options)), label="Building"
        )
    test_waterfall_buildings = mo.ui.dictionary(_menus)
    return (test_waterfall_buildings,)


@app.cell
def _(
    PALETTE,
    mo,
    test_age_group,
    test_predictions,
    test_reading_k,
    test_reading_metric,
    test_shap,
    test_waterfall_buildings,
    test_waterfall_cohort,
    test_waterfall_end,
    waterfall_figure,
):
    _end, _cohort = test_waterfall_end.value, test_waterfall_cohort.value
    _metric, _k = test_reading_metric.value, test_reading_k.value
    # The composite: the total, so the cohorts' values and bases summed.
    _count = "the total" if _cohort == "composite" else _cohort
    # Each model's waterfall of the building chosen in its dropdown.
    test_waterfall_figures = {
        model: waterfall_figure(
            explanation,
            test_age_group,
            test_predictions[model],
            test_waterfall_buildings.value[model],
            _cohort,
            PALETTE,
        )
        for model, explanation in test_shap.items()
    }
    _views = []
    for _model, _figure in test_waterfall_figures.items():
        _views += [
            mo.md(f"**{_model}**"),
            test_waterfall_buildings[_model],
            _figure,
        ]
    mo.vstack(
        [
            mo.md(
                f"**A test building of each model's {_end} {_k} by {_metric} on "
                f"{_cohort}, explained: {_count}** (each model ranks its own "
                "buildings, by the tables' metric and k; from the building's mean "
                "prediction over the background, at its number of apartments, a "
                "step per column to its prediction, the largest on top; red raises "
                "the prediction, blue lowers it)"
            ),
            mo.hstack([test_waterfall_end, test_waterfall_cohort], justify="start"),
            *_views,
        ]
    )
    return (test_waterfall_figures,)


@app.cell
def _(go, style_figure):
    def counts_histogram_figure(buildings, count_name, model_color, palette):
        """The true counts of `count_name` ("total" or a cohort) per building, in
        `palette`'s muted ink, and the predicted means, in `model_color`, over the
        same bins. `buildings` is one model's rows of the test buildings' table (its
        `<count_name>_true` and `<count_name>_predicted` columns)."""
        figure = go.Figure(
            [
                go.Histogram(
                    x=values,
                    name=name,
                    bingroup="counts",
                    marker={"color": trace_color},
                    opacity=0.6,
                    hovertemplate=f"{name} %{{x}}: %{{y}} buildings<extra></extra>",
                )
                for name, values, trace_color in (
                    ("true", buildings[f"{count_name}_true"], palette["muted"]),
                    ("predicted", buildings[f"{count_name}_predicted"], model_color),
                )
            ]
        )
        figure.update_layout(barmode="overlay")
        figure.update_xaxes(title_text=f"{count_name} count")
        figure.update_yaxes(title_text="buildings")
        style_figure(figure, palette)
        figure.update_layout(
            height=400,
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
            },
            margin={"l": 50, "r": 20, "t": 50, "b": 50},
            hovermode="closest",
        )
        return figure

    def counts_scatter_figure(buildings, count_name, palette):
        """Each building's true count of `count_name` ("total" or a cohort) against
        its prediction, equal on the dotted line, colored by its true total of
        children on `palette`'s blue ramp (on the total, x again): whether the
        errors grow with the building's children. `buildings` is one model's rows
        of the test buildings' table."""
        true = buildings[f"{count_name}_true"]
        predicted = buildings[f"{count_name}_predicted"]
        top = max(true.max(), predicted.max()) * 1.05
        figure = go.Figure(
            [
                go.Scatter(
                    x=[0, top],
                    y=[0, top],
                    mode="lines",
                    line={"color": palette["ink"], "width": 1, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                go.Scatter(
                    x=true,
                    y=predicted,
                    mode="markers",
                    showlegend=False,
                    marker={
                        "size": 8,
                        "color": buildings["total_true"],
                        "colorscale": palette["blues"],
                        "colorbar": {"title": {"text": "true total"}, "thickness": 12},
                        "line": {"width": 1, "color": palette["surface"]},
                    },
                    customdata=buildings[
                        ["building_id", "neighborhood_id", "total_true"]
                    ].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]} (%{customdata[1]}), %{customdata[2]} children"
                        "<br>true %{x}, predicted %{y:.2f}<extra></extra>"
                    ),
                ),
            ]
        )
        figure.update_xaxes(title_text=f"true {count_name}", range=[0, top])
        figure.update_yaxes(title_text=f"predicted {count_name}", range=[0, top])
        style_figure(figure, palette)
        figure.update_layout(
            height=400,
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
            },
            margin={"l": 50, "r": 20, "t": 50, "b": 50},
            hovermode="closest",
        )
        return figure

    return counts_histogram_figure, counts_scatter_figure


@app.cell
def _(
    MODEL_COLORS,
    PALETTE,
    counts_histogram_figure,
    counts_scatter_figure,
    mo,
    test_buildings_df,
    test_figures_cohort,
    test_reading_model,
):
    # The chosen model's buildings, in its color: the chosen cohort's counts per
    # building, or the buildings' totals.
    _model, _cohort = test_reading_model.value, test_figures_cohort.value
    _count = "total" if _cohort == "composite" else _cohort
    _buildings = test_buildings_df[test_buildings_df["model"] == _model]
    test_counts_histogram = counts_histogram_figure(
        _buildings, _count, MODEL_COLORS[_model], PALETTE
    )
    test_counts_scatter = counts_scatter_figure(_buildings, _count, PALETTE)
    _count_text = "the total" if _cohort == "composite" else _cohort
    mo.vstack(
        [
            mo.md(
                f"**The true and the predicted counts of {_model} per test building: "
                f"{_count_text}**"
            ),
            mo.hstack([test_reading_model, test_figures_cohort], justify="start"),
            mo.md(
                "The true counts and the predicted means over the same bins (a "
                "model predicts a mean, so even a perfect model's histogram is "
                "narrower than the counts')"
            ),
            test_counts_histogram,
            mo.md(
                "Each building's true count against its prediction, equal on the "
                "dotted line, the color its true total of children"
            ),
            test_counts_scatter,
        ]
    )
    return test_counts_histogram, test_counts_scatter


@app.cell
def _(go, np, style_figure):
    def selection_shares_bars_figure(
        true_counts, predicted_counts, model_colors, palette
    ):
        """The cohort shares of a selection of buildings as grouped bars: each
        cohort's children over all the children, the truth in `palette`'s muted
        ink, each model in its `model_colors` color. A share of n children varies
        by about sqrt(p (1 - p) / n): the true bars carry that 95% interval, each
        child counted as independent (children of one building share it, so the
        noise is larger); the title gives n and each model's predicted total.
        `true_counts` is the selection's true counts, a column per cohort;
        `predicted_counts` is {model: its predicted counts, the same rows}."""
        cohorts = list(true_counts.columns)
        children = {
            "true": true_counts.sum(),
            **{model: counts.sum() for model, counts in predicted_counts.items()},
        }
        n_children = children["true"].sum()
        true_shares = (children["true"] / n_children).to_numpy()
        true_interval = 1.96 * np.sqrt(true_shares * (1 - true_shares) / n_children)
        figure = go.Figure(
            [
                go.Bar(
                    x=cohorts,
                    y=children[series] / children[series].sum(),
                    name=series,
                    marker={
                        "color": bar_color,
                        "line": {"width": 2, "color": palette["surface"]},
                    },
                    error_y=(
                        {
                            "type": "data",
                            "array": true_interval,
                            "color": palette["ink"],
                            "thickness": 1.5,
                            "width": 6,
                        }
                        if series == "true"
                        else None
                    ),
                    customdata=children[series].to_numpy(),
                    hovertemplate=(
                        f"%{{x}} {series} share %{{y:.3f}}, "
                        "%{customdata:.1f} children<extra></extra>"
                    ),
                )
                for series, bar_color in (
                    ("true", palette["muted"]),
                    *model_colors.items(),
                )
            ]
        )
        predicted_totals = ", ".join(
            f"{model} {children[model].sum():.1f}" for model in predicted_counts
        )
        style_figure(figure, palette)
        figure.update_layout(
            title={
                "text": (
                    f"n = {n_children:.0f} true children (predicted: {predicted_totals})"
                ),
                "x": 0,
                "xref": "paper",
                "xanchor": "left",
                "font": {"size": 13},
            },
            barmode="group",
            bargap=0.3,
            height=380,
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "bottom",
            },
            margin={"l": 50, "r": 20, "t": 80, "b": 50},
        )
        # No gridline through a group of bars.
        figure.update_xaxes(showgrid=False)
        figure.update_yaxes(title_text="share of the children")
        return figure

    return (selection_shares_bars_figure,)


@app.cell
def _(
    COHORTS,
    METRICS,
    MODEL_COLORS,
    PALETTE,
    RANKING_METRIC_NAMES,
    compute_metrics,
    mo,
    pd,
    selection_shares_bars_figure,
    test_age_group,
    test_neighborhood,
    test_predictions,
):
    # The selection: every test building, or one neighborhood's.
    if test_neighborhood.value == "all":
        _mask = pd.Series(True, index=test_age_group.groups.index)
        _selection_text = f"all {len(_mask)} test buildings"
    else:
        _mask = test_age_group.groups == test_neighborhood.value
        _selection_text = (
            f"neighborhood {test_neighborhood.value}: its {_mask.sum()} test buildings"
        )
    _true = test_age_group.Y[_mask]
    _predicted = {
        model: predictions[_mask] for model, predictions in test_predictions.items()
    }
    # Its buildings: each cohort's true count and each model's prediction.
    test_selection_buildings_df = pd.DataFrame(
        {
            "building_id": test_age_group.ids[_mask],
            "neighborhood_id": test_age_group.groups[_mask],
        }
    )
    for _cohort in COHORTS:
        test_selection_buildings_df[f"{_cohort}_true"] = _true[_cohort]
        for _model, _model_predictions in _predicted.items():
            test_selection_buildings_df[f"{_cohort}_{_model}"] = _model_predictions[
                _cohort
            ]
    test_selection_buildings_df = test_selection_buildings_df.reset_index(drop=True)
    # Its children per cohort and in total, and the pooled shares: each
    # cohort's children over all the children, so each child counts once.
    test_selection_counts_df = pd.DataFrame(
        {
            "true": [*_true.sum(), _true.sum().sum()],
            **{
                model: [*predictions.sum(), predictions.sum().sum()]
                for model, predictions in _predicted.items()
            },
        },
        index=pd.Index([*COHORTS, "total"], name="cohort"),
    ).reset_index()
    test_selection_shares_df = (
        pd.DataFrame(
            {
                "true": _true.sum() / _true.sum().sum(),
                **{
                    model: predictions.sum() / predictions.sum().sum()
                    for model, predictions in _predicted.items()
                },
            }
        )
        .rename_axis("cohort")
        .reset_index()
    )
    # Each model's scores over the selection's cells, per cohort and for the
    # composite.
    _ranking_metrics = [
        metric for metric in METRICS if metric.name in RANKING_METRIC_NAMES
    ]
    test_selection_scores_df = pd.DataFrame(
        {
            model: compute_metrics(
                _true, predictions, _ranking_metrics, cohorts=[*COHORTS, "composite"]
            ).stack()
            for model, predictions in _predicted.items()
        }
    ).reset_index()
    # The shares as grouped bars, the truth with its 95% interval.
    test_selection_shares_bars = selection_shares_bars_figure(
        _true, _predicted, MODEL_COLORS, PALETTE
    )
    mo.vstack(
        [
            mo.md(
                f"**The statistics of {_selection_text}, true and each model's "
                "predicted**"
            ),
            test_neighborhood,
            mo.md("**Its buildings** (each cohort's true count, then each model's)"),
            mo.ui.table(test_selection_buildings_df.round(4)),
            mo.md("**Its children**, per cohort and in total"),
            mo.ui.table(test_selection_counts_df.round(4)),
            mo.md(
                "**Its cohort shares** (each cohort's children over all the "
                "children; the bars' n children in the title, the true bars' 95% "
                "interval ±1.96·√(p(1−p)/n): the smaller n, the noisier the true "
                "share, so a gap inside the interval is noise)"
            ),
            mo.ui.table(test_selection_shares_df.round(4)),
            test_selection_shares_bars,
            mo.md(
                "**Each model's scores over its cells**, per cohort and for the "
                "composite"
            ),
            mo.ui.table(test_selection_scores_df.round(4)),
        ]
    )
    return (
        test_selection_buildings_df,
        test_selection_counts_df,
        test_selection_scores_df,
        test_selection_shares_bars,
        test_selection_shares_df,
    )


@app.cell
def _(
    CI_BOUND_NAMES,
    MODEL_COLORS,
    PALETTE,
    SELECTION_METRIC,
    beeswarm_figure,
    best_runs,
    build_dataset_figures,
    build_mlflow_dataset,
    code_tags,
    counts_histogram_figure,
    counts_scatter_figure,
    forest_figure,
    log_evaluation_run,
    mlflow,
    mo,
    pd,
    permutation_figure,
    ridge_figure,
    seed,
    selection_shares_bars_figure,
    shared_params,
    study_folder,
    test_age_group,
    test_building_scores_df,
    test_buildings_df,
    test_difference_df,
    test_difference_replicates,
    test_importance_df,
    test_permutation,
    test_predictions,
    test_replicate_scores,
    test_scores_df,
    test_shap,
    train_val_age_group,
    val_against_test_df,
    waterfall_figure,
):
    # Runs once "Evaluate on the test part" has produced what it reads. One
    # final run per experiment (E11): a FINISHED run tagged final refuses another.
    mo.stop(
        bool(
            mlflow.search_runs(
                filter_string="tags.final = 'true' and attributes.status = 'FINISHED'",
                output_format="list",
            )
        ),
        mo.md("A final run exists in the experiment: nothing logged."),
    )
    # The tables at 6 significant digits, as CSV. The
    # SHAP values a row per model, building and cohort: its base, then a column
    # per feature (the base plus the values is the cohort's prediction; the
    # features' values are in test_buildings, by model and building_id).
    _tables = {
        "test_scores": test_scores_df,
        "test_difference": test_difference_df,
        "val_against_test": val_against_test_df,
        "test_importance": test_importance_df,
        "test_buildings": test_buildings_df,
        "test_building_scores": test_building_scores_df,
        "test_shap_values": pd.concat(
            [
                pd.DataFrame(
                    {
                        "model": model,
                        "building_id": test_age_group.ids.to_numpy(),
                        "cohort": cohort,
                        "base_value": explanation.base_values[:, output],
                        **dict(
                            zip(
                                explanation.feature_names,
                                explanation.values[:, :, output].T,
                                strict=True,
                            )
                        ),
                    }
                )
                for model, explanation in test_shap.items()
                for output, cohort in enumerate(explanation.output_names)
            ],
            ignore_index=True,
        ),
    }
    # The test part's 18 figures (its permutation importance has the test split
    # only); per model the true and predicted totals per test building and the
    # waterfalls of the total for its 5 worst test buildings by the selection
    # metric on the composite (the reading tables' defaults, fixed here: a menu
    # changes nothing logged; rank 1 the worst); the cohort shares of every test
    # building.
    _n_worst_buildings = 5
    _figures = build_dataset_figures(
        "test",
        test_scores_df,
        test_replicate_scores,
        test_difference_df,
        test_difference_replicates,
        {
            model: {cohort: {"test": result} for cohort, result in by_cohort.items()}
            for model, by_cohort in test_permutation.items()
        },
        test_shap,
        test_age_group.X,
        SELECTION_METRIC.name,
        CI_BOUND_NAMES,
        seed,
        PALETTE,
        forest_figure,
        ridge_figure,
        permutation_figure,
        beeswarm_figure,
    )
    for _model in best_runs:
        _buildings = test_buildings_df[test_buildings_df["model"] == _model]
        _figures[f"figures/test_counts_histogram_{_model}_total.html"] = (
            counts_histogram_figure(_buildings, "total", MODEL_COLORS[_model], PALETTE)
        )
        _figures[f"figures/test_counts_scatter_{_model}_total.html"] = (
            counts_scatter_figure(_buildings, "total", PALETTE)
        )
        # Ranked as the waterfalls' menus rank (a tie keeps the earlier building).
        _worst_building_ids = (
            test_building_scores_df[
                (test_building_scores_df["model"] == _model)
                & (test_building_scores_df["cohort"] == "composite")
            ]
            .set_index("building_id")[[SELECTION_METRIC.name]]
            .sort_values(SELECTION_METRIC.name, ascending=False, kind="stable")
            .nlargest(_n_worst_buildings, SELECTION_METRIC.name)
            .index
        )
        for _rank, _building_id in enumerate(_worst_building_ids, start=1):
            _figures[
                f"figures/test_waterfall_{_model}_total_{_rank}_{_building_id}.html"
            ] = waterfall_figure(
                test_shap[_model],
                test_age_group,
                test_predictions[_model],
                _building_id,
                "composite",
                PALETTE,
            )
    _figures["figures/test_selection_shares_bars_all.html"] = (
        selection_shares_bars_figure(
            test_age_group.Y, test_predictions, MODEL_COLORS, PALETTE
        )
    )
    _n_metrics = log_evaluation_run(
        "final",
        {
            "run_key": f"{study_folder}/final",
            "stage": "final",
            "final": "true",
            **code_tags,
        },
        {**shared_params, "n_train_val": len(train_val_age_group.X)},
        "test",
        test_scores_df,
        test_difference_df,
        CI_BOUND_NAMES,
        _tables,
        _figures,
        best_runs,
        study_folder,
        {
            "training": build_mlflow_dataset(train_val_age_group, "train_val"),
            "testing": build_mlflow_dataset(test_age_group, "test"),
        },
    )
    mo.md(
        f"Logged the final run: {_n_metrics} metrics, {len(_tables)} tables, "
        f"{len(_figures)} figures."
    )
    return


if __name__ == "__main__":
    app.run()
