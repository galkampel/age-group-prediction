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

    print("data_settings " + " ".join(f"{k}={v}" for k, v in DATA_SETTINGS.items()))
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
    return (config_hash,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Experiment settings

    The objective EDA's settings (described in `02_objective_eda.py`) and the
    tuning's own: the selection metric, the bootstrap and the inspection;
    every run logs them (a test pins the shared ones equal).
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
        "selection_metric": selection_metric,
        "n_resamples": n_resamples,
        "alpha": alpha,
        "n_repeats": n_repeats,
        "n_shap_rows": n_shap_rows,
        "n_shap_background": n_shap_background,
    }
    print(
        "experiment_settings "
        + " ".join(f"{k}={v}" for k, v in EXPERIMENT_SETTINGS.items())
    )
    mo.md(f"```text\n{pd.DataFrame([EXPERIMENT_SETTINGS]).to_string(index=False)}\n```")
    return (
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
    return splitter, train_age_group, val_age_group


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
    descent). Each run's best model is refit on the
    training part and scored on validation; each model's best run is the
    lowest validation score by `selection_metric`, bootstrapped over whole
    neighborhoods and inspected behind its button. The objective EDA
    (`02_objective_eda.py`) describes the studies' storage and their trials.
    """)
    return


@app.cell
def _():
    import dataclasses

    import numpy as np
    import optuna
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde

    from age_group_prediction.hyperparameter_tuning import (
        CountComponent,
        CrossValidationTuner,
        ProbabilityComponent,
        build_independent_cohorts_model,
        build_total_times_probability_model,
    )
    from age_group_prediction.inspection import (
        BootstrapEvaluator,
        PermutationImportance,
        ShapExplainer,
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
        BootstrapEvaluator,
        CountComponent,
        CrossValidationTuner,
        Metric,
        PermutationImportance,
        ProbabilityComponent,
        ShapExplainer,
        build_independent_cohorts_model,
        build_total_times_probability_model,
        compute_metrics,
        dataclasses,
        gaussian_kde,
        go,
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
    # Each metric with the direction Optuna is given; a loss, so "minimize".
    # The Poisson deviance is also the permutation importance's score.
    POISSON_DEVIANCE = Metric(
        "poisson_deviance", mean_poisson_deviance_loss, direction="minimize"
    )
    LOG_LOSS = Metric("log_loss", multinomial_log_loss, direction="minimize")
    RMSE = Metric("rmse", root_mean_squared_error_loss, direction="minimize")

    # Every metric: what every trial records, each run's score table (each
    # cohort through `cohort=`, then the composite) and the bootstrap of the
    # best runs. The count metrics pool the (building, cohort) cells. Bias is
    # signed, so its direction is nominal: only the permutation importance
    # reads one.
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
    # The selection metric: it picks each model's best run (the composite's
    # point score on validation) and is the forest plots' metric.
    if selection_metric == "rmse":
        SELECTION_METRIC = RMSE
    elif selection_metric == "poisson_deviance":
        SELECTION_METRIC = POISSON_DEVIANCE
    else:
        raise ValueError(f"unknown selection_metric {selection_metric!r}")
    return METRICS, OBJECTIVE_METRIC, POISSON_DEVIANCE, SELECTION_METRIC


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
def _(BootstrapEvaluator, alpha, n_resamples, seed):
    # The bootstrap of each model's best run, over neighborhoods: one seed, so
    # both models score the same replicate rows and the intervals compare; the
    # interval's bounds named by their level (95% at alpha 0.05).
    bootstrap_evaluator = BootstrapEvaluator(n_resamples, random_state=seed)
    _level = round(100 * (1 - alpha))
    CI_BOUND_NAMES = {"lower": f"ci{_level}_lower", "upper": f"ci{_level}_upper"}
    return CI_BOUND_NAMES, bootstrap_evaluator


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
        "(a count family × exposure on or off); each trial fits its folds in parallel."
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
    compute_metrics,
    cv_groups,
    find_feature_transformer_name,
    mo,
    pd,
    run_model1,
    seed,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model1.value,
        mo.md("Click the button to tune the independent cohorts model."),
    )
    # The independent cohorts model's runs, one after another, the i-th seeded
    # seed + i. A failed trial is caught by the study; any other error ends
    # the loop, and the records made so far are lost.
    MODEL1_RUNS = []
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
        # the family's candidates); its value, trial counts and seconds. The
        # trials are not kept: the objective EDA studies them.
        _states = [trial.state.name for trial in _study.trials]
        MODEL1_RUNS.append(
            {
                **_run_settings,
                "feature_transformers": {
                    component.name: find_feature_transformer_name(
                        component, COUNT_FEATURE_TRANSFORMERS[_family]
                    )
                    for component in _best_components
                },
                "model": _model,
                "components": _best_components,
                "cv_best": _study.best_value,
                "n_complete": _states.count("COMPLETE"),
                "n_failed": _states.count("FAIL"),
                "seconds": _seconds,
                "scores": pd.concat(_scores_by_dataset, names=["dataset"]),
            }
        )
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
                "point score; each model's best run is bootstrapped below)"
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
        "(a family pair × exposure on or off); each trial fits its folds in parallel."
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
    compute_metrics,
    cv_groups,
    find_feature_transformer_name,
    mo,
    pd,
    run_model2,
    seed,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(
        not run_model2.value,
        mo.md("Click the button to tune the total times probability model."),
    )
    # The total times probability model's runs, one after another, the i-th
    # seeded seed + i. A failed trial is caught by the study; any other error
    # ends the loop, and the records made so far are lost.
    MODEL2_RUNS = []
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
        # each family's candidates); its value, trial counts and seconds. The
        # trials are not kept: the objective EDA studies them.
        _states = [trial.state.name for trial in _study.trials]
        MODEL2_RUNS.append(
            {
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
                "cv_best": _study.best_value,
                "n_complete": _states.count("COMPLETE"),
                "n_failed": _states.count("FAIL"),
                "seconds": _seconds,
                "scores": pd.concat(_scores_by_dataset, names=["dataset"]),
            }
        )
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
                "point score; each model's best run is bootstrapped below)"
            ),
            mo.ui.table(model2_scores_df.round(4)),
        ]
    )
    return


@app.cell
def _(
    CI_BOUND_NAMES,
    COHORTS,
    METRICS,
    MODEL1_RUNS,
    MODEL2_RUNS,
    SELECTION_METRIC,
    alpha,
    bootstrap_evaluator,
    mo,
    pd,
    val_age_group,
):
    # Each model's best run, by its point score on the selection metric of the
    # composite on validation: its stored model, fitted on the training part.
    _best_run1 = min(
        MODEL1_RUNS,
        key=lambda run: run["scores"].loc[
            ("val", "composite", SELECTION_METRIC.name), "score"
        ],
    )
    _best_run2 = min(
        MODEL2_RUNS,
        key=lambda run: run["scores"].loc[
            ("val", "composite", SELECTION_METRIC.name), "score"
        ],
    )
    best_model1, best_model2 = _best_run1["model"], _best_run2["model"]
    # The two bootstrapped on validation, every metric per cohort and for the
    # composite, by one evaluator: both score the same replicate rows, so their
    # replicates pair row by row. Each score with its replicates' mean, median,
    # SD and percentile interval.
    best_replicate_scores, _scores_by_run = {}, {}
    for _model_name, _run, _family in (
        ("independent_cohorts", _best_run1, _best_run1["family"]),
        (
            "total_times_probability",
            _best_run2,
            f"{_best_run2['total_family']}+{_best_run2['probability_family']}",
        ),
    ):
        _replicate_scores = bootstrap_evaluator.evaluate(
            val_age_group.Y,
            _run["model"].predict(val_age_group.X, exposure=val_age_group.exposure),
            val_age_group.groups,
            metrics=METRICS,
            cohorts=[*COHORTS, "composite"],
        )
        best_replicate_scores[_model_name] = _replicate_scores
        _scores_by_run[_model_name, _family, _run["use_exposure"]] = pd.DataFrame(
            {
                "score": _run["scores"].loc["val", "score"],
                "mean": _replicate_scores.mean(),
                "median": _replicate_scores.median(),
                "sd": _replicate_scores.std(),
                CI_BOUND_NAMES["lower"]: _replicate_scores.quantile(alpha / 2),
                CI_BOUND_NAMES["upper"]: _replicate_scores.quantile(1 - alpha / 2),
            }
        )
    # A row per best run, cohort and metric: the run's model, family and
    # exposure, then its validation score and the bootstrap's statistics.
    best_runs_val_scores_df = pd.concat(
        _scores_by_run, names=["model", "family", "use_exposure"]
    ).reset_index()
    mo.md(
        f"**Each model's best run by the {SELECTION_METRIC.name} of the "
        "composite on validation, bootstrapped on validation:** "
        + "; ".join(
            f"{model} `{family}`, exposure {use_exposure}"
            for model, family, use_exposure in _scores_by_run
        )
        + " (each score with its replicates' mean, median, SD and 95% "
        "interval; a table per model below)"
    )
    return (
        best_model1,
        best_model2,
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
        # A fresh index: the frame's row positions would show as a column.
        _rows = _chosen[_chosen["model"] == _model].reset_index(drop=True)
        _views.append(mo.ui.table(_rows.round(4)))
    mo.vstack(
        [mo.hstack([best_runs_cohort, best_runs_metrics], justify="start"), *_views]
    )
    return


@app.cell
def _(gaussian_kde, go, make_subplots, np):
    def forest_figure(
        scores,
        metric_name,
        lower_column,
        upper_column,
        share_x_axis=False,
        rule_at=None,
    ):
        """A forest plot of `metric_name`: a row per cohort, the composite
        first, and in each row a dot per model (the first on top) at its
        score, the bar its interval from `lower_column` to `upper_column`.
        `scores` is long, a row per model, cohort and metric, with the columns
        `model`, `cohort`, `metric`, `score` and the two bounds; the legend
        names the models. Each row has its own x scale unless `share_x_axis`;
        `rule_at` draws a vertical line at that value in every row (0 for a
        difference)."""
        # The dataviz skill's reference palette: the three categorical colors
        # that stay apart for colorblind readers, then the surface, the inks,
        # the gridline and the axis line.
        colors = ["#2a78d6", "#eb6834", "#1baf7a"]
        surface, ink, muted, grid, axis = (
            "#fcfcfb",
            "#52514e",
            "#898781",
            "#e1e0d9",
            "#c3c2b7",
        )
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
                color = colors[position]
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
                            "line": {"width": 2, "color": surface},
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
                    x=rule_at, line_width=1, line_color=ink, row=row, col=1
                )
        height = 110 * len(cohorts) + 130
        figure.update_layout(
            height=height,
            template="plotly_white",
            paper_bgcolor=surface,
            plot_bgcolor=surface,
            font={
                "family": "system-ui, -apple-system, Segoe UI, sans-serif",
                "color": ink,
                "size": 12,
            },
            showlegend=len(models) > 1,
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
        figure.update_xaxes(
            gridcolor=grid, zeroline=False, linecolor=axis, tickfont={"color": muted}
        )
        figure.update_xaxes(title_text=metric_name, row=len(cohorts), col=1)
        figure.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        # Each row's title (its cohort) at the row's top left.
        figure.update_annotations(x=0, xanchor="left")
        return figure

    def ridge_figure(replicate_scores, metric_name, share_x_axis=False, rule_at=None):
        """A ridge plot of `metric_name`: a row per cohort, the composite
        first, and in each row each model's bootstrap replicates as a density
        (a Gaussian KDE between the replicates' extremes; the row's highest
        density on top). `replicate_scores` is {model: its replicates, a row
        per replicate and a column per cohort and metric}; the legend names
        the models. Each row has its own x scale unless `share_x_axis`;
        `rule_at` draws a vertical line at that value in every row (0 for a
        difference)."""
        # The forest's palette: the three categorical colors, the surface, the
        # inks, the gridline and the axis line.
        colors = ["#2a78d6", "#eb6834", "#1baf7a"]
        surface, ink, muted, grid, axis = (
            "#fcfcfb",
            "#52514e",
            "#898781",
            "#e1e0d9",
            "#c3c2b7",
        )
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
                x_values, density = curves[model]
                figure.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=density / row_highest,
                        mode="lines",
                        fill="tozeroy",
                        fillcolor=colors[position] + "1a",
                        line={"width": 2, "color": colors[position]},
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
                    x=rule_at, line_width=1, line_color=ink, row=row, col=1
                )
        height = 110 * len(cohorts) + 130
        figure.update_layout(
            height=height,
            template="plotly_white",
            paper_bgcolor=surface,
            plot_bgcolor=surface,
            font={
                "family": "system-ui, -apple-system, Segoe UI, sans-serif",
                "color": ink,
                "size": 12,
            },
            showlegend=len(models) > 1,
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
        figure.update_xaxes(
            gridcolor=grid, zeroline=False, linecolor=axis, tickfont={"color": muted}
        )
        figure.update_xaxes(title_text=metric_name, row=len(cohorts), col=1)
        figure.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        # Each row's title (its cohort) at the row's top left.
        figure.update_annotations(x=0, xanchor="left")
        return figure

    return forest_figure, ridge_figure


@app.cell
def _(mo):
    # The form of the best runs' plot: the forest (each run's score and 95%
    # interval) or the ridge (the density of each run's bootstrap
    # replicates).
    best_runs_plot = mo.ui.radio(
        options=["forest", "ridge"], value="forest", inline=True, label="Plot"
    )
    return (best_runs_plot,)


@app.cell
def _(
    CI_BOUND_NAMES,
    SELECTION_METRIC,
    best_replicate_scores,
    best_runs_plot,
    best_runs_val_scores_df,
    forest_figure,
    mo,
    ridge_figure,
):
    metric_name = SELECTION_METRIC.name  # POISSON_DEVIANCE.name

    if best_runs_plot.value == "forest":
        best_runs_figure = forest_figure(
            best_runs_val_scores_df,
            metric_name,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
        )
    elif best_runs_plot.value == "ridge":
        best_runs_figure = ridge_figure(best_replicate_scores, metric_name)
    else:
        raise ValueError(f"unknown plot {best_runs_plot.value!r}")
    mo.vstack(
        [
            mo.md(
                "**Each model's best run on validation: the selection metric per "
                "cohort and for the composite** (a row per cohort, the composite "
                "first; the dot the run's score, the bar its 95% bootstrap interval, "
                "the colors named in the legend; the ridge shows instead the density "
                "of the run's 1000 bootstrap replicates)"
            ),
            best_runs_plot,
            best_runs_figure,
        ]
    )
    return (metric_name,)


@app.cell
def _(metric_name, mo):
    run_inspection = mo.ui.run_button(label="Inspect the best runs")
    mo.md(
        f"{run_inspection} each model's best run: the permutation importance and "
        f"SHAP for the metric {metric_name} (about two minutes)."
    )
    return (run_inspection,)


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
    shap_rows = AgeGroupDataset(*(part.iloc[_rows] for part in val_age_group))
    shap_background = train_age_group.X.iloc[_background]
    return shap_background, shap_rows


@app.cell
def _(
    POISSON_DEVIANCE,
    PermutationImportance,
    ShapExplainer,
    best_model1,
    mo,
    n_repeats,
    pd,
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
    # Permutation importance of the Poisson deviance: on validation the decision
    # table, on training the overfit check. SHAP, exact at nine columns, of the
    # validation subsample against the training background; the values explain
    # the count (the rows' exposure), per row, column and cohort.
    _importance = PermutationImportance(POISSON_DEVIANCE, n_repeats, random_state=seed)
    model1_permutation = pd.concat(
        {
            "train": _importance.compute_importances(
                best_model1,
                train_age_group.X,
                train_age_group.Y,
                exposure=train_age_group.exposure,
            ),
            "val": _importance.compute_importances(
                best_model1,
                val_age_group.X,
                val_age_group.Y,
                exposure=val_age_group.exposure,
            ),
        },
        names=["split", "column"],
    )
    model1_shap = ShapExplainer(method="exact", n_jobs=-1).explain(
        best_model1, shap_rows.X, shap_background, exposure=shap_rows.exposure
    )
    return model1_permutation, model1_shap


@app.cell
def _(
    POISSON_DEVIANCE,
    PermutationImportance,
    ShapExplainer,
    best_model2,
    mo,
    n_repeats,
    pd,
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
    _importance = PermutationImportance(POISSON_DEVIANCE, n_repeats, random_state=seed)
    model2_permutation = pd.concat(
        {
            "train": _importance.compute_importances(
                best_model2,
                train_age_group.X,
                train_age_group.Y,
                exposure=train_age_group.exposure,
            ),
            "val": _importance.compute_importances(
                best_model2,
                val_age_group.X,
                val_age_group.Y,
                exposure=val_age_group.exposure,
            ),
        },
        names=["split", "column"],
    )
    model2_shap = ShapExplainer(method="exact", n_jobs=-1).explain(
        best_model2, shap_rows.X, shap_background, exposure=shap_rows.exposure
    )
    return model2_permutation, model2_shap


@app.cell
def _(
    mo,
    model1_permutation,
    model1_shap,
    model2_permutation,
    model2_shap,
    np,
    pd,
):
    # The two importances side by side, per model and raw column: the permutation
    # importance on validation (its rank decides) and on training, and the mean
    # |SHAP| over the explained rows and the cohorts (D18).
    _rows = []
    for _model, _permutation, _shap in [
        ("independent_cohorts", model1_permutation, model1_shap),
        ("total_times_probability", model2_permutation, model2_shap),
    ]:
        _table = pd.DataFrame(
            {
                "perm_mean_train": _permutation.loc["train", "importance_mean"],
                "perm_mean_val": _permutation.loc["val", "importance_mean"],
                "perm_std_val": _permutation.loc["val", "importance_std"],
                "mean_abs_shap": pd.Series(
                    np.abs(_shap.values).mean(axis=(0, 2)), index=_shap.feature_names
                ),
            }
        )
        _table["perm_rank"] = _table["perm_mean_val"].rank(ascending=False).astype(int)
        _table["shap_rank"] = _table["mean_abs_shap"].rank(ascending=False).astype(int)
        _rows.append(_table.rename_axis("column").reset_index().assign(model=_model))
    importance_df = pd.concat(_rows, ignore_index=True)[
        [
            "model",
            "column",
            "perm_mean_train",
            "perm_mean_val",
            "perm_std_val",
            "perm_rank",
            "mean_abs_shap",
            "shap_rank",
        ]
    ]
    mo.vstack(
        [
            mo.md(
                "**Feature importance of the best runs** (perm: the drop of the cohort "
                "Poisson deviance when the column is shuffled; mean_abs_shap: mean |SHAP| "
                "of the predicted count over the explained rows and cohorts)"
            ),
            mo.md(f"```text\n{importance_df.round(4).to_string(index=False)}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
