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
    descent). Each run's best model is refit on the
    training part and scored on validation; each model's best run is the
    lowest validation score by `selection_metric`, bootstrapped over whole
    neighborhoods and inspected behind its button. The two best runs' paired
    difference, per bootstrap replicate, shows whether one is better; a
    "Metric" menu chooses the metric of the plots and, unless it is bias, of
    the permutation importance (`selection_metric` by default). The
    objective EDA
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
    # The two run records by the model's label: the test evaluation refits
    # them from their components.
    best_runs = {
        "independent_cohorts": _best_run1,
        "total_times_probability": _best_run2,
    }
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
        best_runs,
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
def _(SELECTION_METRIC, mo):
    # The form of the best runs' plot: the forest (each run's score and 95%
    # interval) or the ridge (the density of each run's bootstrap
    # replicates); and its metric, the selection metric by default, which the
    # permutation importance also scores unless it is bias.
    best_runs_plot = mo.ui.radio(
        options=["forest", "ridge"], value="forest", inline=True, label="Plot"
    )
    best_runs_metric = mo.ui.dropdown(
        options=["rmse", "poisson_deviance", "bias"],
        value=SELECTION_METRIC.name,
        label="Metric",
    )
    return best_runs_metric, best_runs_plot


@app.cell
def _(
    CI_BOUND_NAMES,
    METRICS,
    SELECTION_METRIC,
    best_replicate_scores,
    best_runs_metric,
    best_runs_plot,
    best_runs_val_scores_df,
    forest_figure,
    mo,
    ridge_figure,
):
    # The permutation importance's metric: the chosen one when it is a loss;
    # for bias, a signed score, the selection metric.
    if best_runs_metric.value in ("rmse", "poisson_deviance"):
        importance_metric = {metric.name: metric for metric in METRICS}[
            best_runs_metric.value
        ]
    elif best_runs_metric.value == "bias":
        importance_metric = SELECTION_METRIC
    else:
        raise ValueError(f"unknown metric {best_runs_metric.value!r}")
    if best_runs_plot.value == "forest":
        best_runs_figure = forest_figure(
            best_runs_val_scores_df,
            best_runs_metric.value,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
        )
    elif best_runs_plot.value == "ridge":
        best_runs_figure = ridge_figure(best_replicate_scores, best_runs_metric.value)
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
    return (importance_metric,)


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
    alpha,
    best_difference_metric,
    best_difference_plot,
    best_replicate_scores,
    best_runs_val_scores_df,
    forest_figure,
    mo,
    pd,
    ridge_figure,
):
    # The two best runs' paired difference per replicate: both were scored on
    # the same replicate rows (one evaluator), so the difference keeps the
    # pairing, and its interval is far narrower than either run's. Below 0
    # the total times probability model scores lower.
    _difference_name = "total_times_probability - independent_cohorts"
    best_difference_replicates = (
        best_replicate_scores["total_times_probability"]
        - best_replicate_scores["independent_cohorts"]
    )
    _point_scores = best_runs_val_scores_df.set_index(["model", "cohort", "metric"])[
        "score"
    ]
    # A row per cohort and metric: the point scores' difference, then the
    # replicates' statistics, as for each run.
    best_difference_df = (
        pd.DataFrame(
            {
                "score": _point_scores["total_times_probability"]
                - _point_scores["independent_cohorts"],
                "mean": best_difference_replicates.mean(),
                "median": best_difference_replicates.median(),
                "sd": best_difference_replicates.std(),
                CI_BOUND_NAMES["lower"]: best_difference_replicates.quantile(alpha / 2),
                CI_BOUND_NAMES["upper"]: best_difference_replicates.quantile(
                    1 - alpha / 2
                ),
            }
        )
        .rename_axis(["cohort", "metric"])
        .reset_index()
    )
    best_difference_df.insert(0, "model", _difference_name)
    # The chosen plot and metric, one x axis for every cohort so the rule at
    # 0 lines up.
    if best_difference_plot.value == "forest":
        best_difference_figure = forest_figure(
            best_difference_df,
            best_difference_metric.value,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
            share_x_axis=True,
            rule_at=0,
        )
    elif best_difference_plot.value == "ridge":
        best_difference_figure = ridge_figure(
            {_difference_name: best_difference_replicates},
            best_difference_metric.value,
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
def _(importance_metric, mo):
    run_inspection = mo.ui.run_button(label="Inspect the best runs")
    mo.md(
        f"{run_inspection} each model's best run: the permutation importance on "
        f"{importance_metric.name} and SHAP (about two minutes)."
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
    PermutationImportance,
    ShapExplainer,
    best_model1,
    importance_metric,
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
    # Permutation importance of its metric: on validation the decision
    # table, on training the overfit check. SHAP, exact at nine columns, of the
    # validation subsample against the training background; the values explain
    # the count (the rows' exposure), per row, column and cohort.
    _importance = PermutationImportance(importance_metric, n_repeats, random_state=seed)
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
    PermutationImportance,
    ShapExplainer,
    best_model2,
    importance_metric,
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
    _importance = PermutationImportance(importance_metric, n_repeats, random_state=seed)
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
    importance_metric,
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
                f"{importance_metric.name} when the column is shuffled; mean_abs_shap: mean |SHAP| "
                "of the predicted count over the explained rows and cohorts)"
            ),
            mo.md(f"```text\n{importance_df.round(4).to_string(index=False)}\n```"),
        ]
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
    """)
    return


@app.cell
def _(mo):
    run_test_evaluation = mo.ui.run_button(label="Evaluate on the test part")
    mo.md(
        f"{run_test_evaluation} each model's best run: refit on the training and "
        "validation parts, scored once on the test part."
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
    bootstrap_evaluator,
    build_independent_cohorts_model,
    build_total_times_probability_model,
    compute_metrics,
    mo,
    pd,
    run_test_evaluation,
    seed,
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
    test_predictions, test_replicate_scores, _scores_by_run = {}, {}, {}
    for _model_name, _run in best_runs.items():
        if _model_name == "independent_cohorts":
            _model = build_independent_cohorts_model(
                _run["components"], y_min=PREDICTION_FLOOR, random_state=seed
            )
            _family = _run["family"]
        elif _model_name == "total_times_probability":
            _model = build_total_times_probability_model(
                *_run["components"], y_min=PREDICTION_FLOOR, random_state=seed
            )
            _family = f"{_run['total_family']}+{_run['probability_family']}"
        else:
            raise ValueError(f"unknown model {_model_name!r}")
        _model.fit(
            train_val_age_group.X,
            train_val_age_group.Y,
            exposure=train_val_age_group.exposure,
        )
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
        _scores_by_run[_model_name, _family, _run["use_exposure"]] = pd.DataFrame(
            {
                "score": compute_metrics(
                    test_age_group.Y,
                    test_predictions[_model_name],
                    METRICS,
                    cohorts=[*COHORTS, "composite"],
                ).stack(),
                "mean": _replicate_scores.mean(),
                "median": _replicate_scores.median(),
                "sd": _replicate_scores.std(),
                CI_BOUND_NAMES["lower"]: _replicate_scores.quantile(alpha / 2),
                CI_BOUND_NAMES["upper"]: _replicate_scores.quantile(1 - alpha / 2),
            }
        )
    # A row per best run, cohort and metric, as on validation.
    test_scores_df = pd.concat(
        _scores_by_run, names=["model", "family", "use_exposure"]
    ).reset_index()
    # The two runs' paired difference per test replicate, as on validation.
    _difference_name = "total_times_probability - independent_cohorts"
    test_difference_replicates = (
        test_replicate_scores["total_times_probability"]
        - test_replicate_scores["independent_cohorts"]
    )
    _point_scores = test_scores_df.set_index(["model", "cohort", "metric"])["score"]
    test_difference_df = (
        pd.DataFrame(
            {
                "score": _point_scores["total_times_probability"]
                - _point_scores["independent_cohorts"],
                "mean": test_difference_replicates.mean(),
                "median": test_difference_replicates.median(),
                "sd": test_difference_replicates.std(),
                CI_BOUND_NAMES["lower"]: test_difference_replicates.quantile(alpha / 2),
                CI_BOUND_NAMES["upper"]: test_difference_replicates.quantile(
                    1 - alpha / 2
                ),
            }
        )
        .rename_axis(["cohort", "metric"])
        .reset_index()
    )
    test_difference_df.insert(0, "model", _difference_name)
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
        test_predictions,
        test_replicate_scores,
        test_scores_df,
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
    best_runs_val_scores_df,
    forest_figure,
    mo,
    pd,
    ridge_figure,
    test_difference_df,
    test_difference_replicates,
    test_metric,
    test_plot,
    test_replicate_scores,
    test_scores_df,
):
    _difference_name = test_difference_df["model"].iloc[0]
    if test_plot.value == "forest":
        test_runs_figure = forest_figure(
            test_scores_df,
            test_metric.value,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
        )
        test_difference_figure = forest_figure(
            test_difference_df,
            test_metric.value,
            CI_BOUND_NAMES["lower"],
            CI_BOUND_NAMES["upper"],
            share_x_axis=True,
            rule_at=0,
        )
    elif test_plot.value == "ridge":
        test_runs_figure = ridge_figure(test_replicate_scores, test_metric.value)
        test_difference_figure = ridge_figure(
            {_difference_name: test_difference_replicates},
            test_metric.value,
            share_x_axis=True,
            rule_at=0,
        )
    else:
        raise ValueError(f"unknown plot {test_plot.value!r}")
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
def _(go, mo, test_buildings_df, test_figures_cohort, test_reading_model):
    # The forest's palette: each model in its color, the truth in the muted
    # ink; the number of apartments on the sequential blue ramp (steps 250,
    # 400, 550, 700, so the lightest still shows on the surface).
    _model_colors = {
        "independent_cohorts": "#2a78d6",
        "total_times_probability": "#eb6834",
    }
    _surface, _ink, _muted, _grid, _axis = (
        "#fcfcfb",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
    )
    _blues = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    _model, _cohort = test_reading_model.value, test_figures_cohort.value
    _color = _model_colors[_model]
    # The chosen cohort's counts per building, or the buildings' totals.
    _count = "total" if _cohort == "composite" else _cohort
    _buildings = test_buildings_df[test_buildings_df["model"] == _model].set_index(
        "building_id"
    )
    _true, _predicted = _buildings[f"{_count}_true"], _buildings[f"{_count}_predicted"]
    # The true counts and the predicted means over the same bins.
    test_counts_histogram = go.Figure(
        [
            go.Histogram(
                x=_values,
                name=_name,
                bingroup="counts",
                marker={"color": _trace_color},
                opacity=0.6,
                hovertemplate=f"{_name} %{{x}}: %{{y}} buildings<extra></extra>",
            )
            for _name, _values, _trace_color in (
                ("true", _true, _muted),
                ("predicted", _predicted, _color),
            )
        ]
    )
    test_counts_histogram.update_layout(barmode="overlay")
    test_counts_histogram.update_xaxes(title_text=f"{_count} count")
    test_counts_histogram.update_yaxes(title_text="buildings")
    # Each building's true count against its prediction, colored by its
    # number of apartments; on the dotted line they are equal.
    _top = max(_true.max(), _predicted.max()) * 1.05
    test_counts_scatter = go.Figure(
        [
            go.Scatter(
                x=[0, _top],
                y=[0, _top],
                mode="lines",
                line={"color": _ink, "width": 1, "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=_true,
                y=_predicted,
                mode="markers",
                showlegend=False,
                marker={
                    "size": 8,
                    "color": _buildings["n_apartments"],
                    "colorscale": _blues,
                    "colorbar": {"title": {"text": "n_apartments"}, "thickness": 12},
                    "line": {"width": 1, "color": _surface},
                },
                customdata=_buildings.reset_index()[
                    ["building_id", "neighborhood_id", "n_apartments"]
                ].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]} (%{customdata[1]}), %{customdata[2]} apartments"
                    "<br>true %{x}, predicted %{y:.2f}<extra></extra>"
                ),
            ),
        ]
    )
    test_counts_scatter.update_xaxes(title_text=f"true {_count}", range=[0, _top])
    test_counts_scatter.update_yaxes(title_text=f"predicted {_count}", range=[0, _top])
    for _figure in (test_counts_histogram, test_counts_scatter):
        _figure.update_layout(
            height=400,
            template="plotly_white",
            paper_bgcolor=_surface,
            plot_bgcolor=_surface,
            font={
                "family": "system-ui, -apple-system, Segoe UI, sans-serif",
                "color": _ink,
                "size": 12,
            },
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
        _figure.update_xaxes(
            gridcolor=_grid, zeroline=False, linecolor=_axis, tickfont={"color": _muted}
        )
        _figure.update_yaxes(
            gridcolor=_grid, zeroline=False, linecolor=_axis, tickfont={"color": _muted}
        )
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
                "dotted line, the color its number of apartments"
            ),
            test_counts_scatter,
        ]
    )
    return test_counts_histogram, test_counts_scatter


@app.cell
def _(
    COHORTS,
    METRICS,
    RANKING_METRIC_NAMES,
    compute_metrics,
    go,
    mo,
    pd,
    test_age_group,
    test_neighborhood,
    test_predictions,
):
    _model_colors = {
        "independent_cohorts": "#2a78d6",
        "total_times_probability": "#eb6834",
    }
    _surface, _ink, _muted, _grid, _axis = (
        "#fcfcfb",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
    )
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
    # The shares as grouped bars: the truth in the muted ink, each model in
    # its color.
    test_selection_shares_bars = go.Figure(
        [
            go.Bar(
                x=COHORTS,
                y=test_selection_shares_df[_series],
                name=_series,
                marker={"color": _bar_color, "line": {"width": 2, "color": _surface}},
                hovertemplate=f"%{{x}} {_series} share %{{y:.3f}}<extra></extra>",
            )
            for _series, _bar_color in (
                ("true", _muted),
                *_model_colors.items(),
            )
        ]
    )
    test_selection_shares_bars.update_layout(
        barmode="group",
        bargap=0.3,
        height=360,
        template="plotly_white",
        paper_bgcolor=_surface,
        plot_bgcolor=_surface,
        font={
            "family": "system-ui, -apple-system, Segoe UI, sans-serif",
            "color": _ink,
            "size": 12,
        },
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
        },
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    test_selection_shares_bars.update_xaxes(linecolor=_axis, tickfont={"color": _muted})
    test_selection_shares_bars.update_yaxes(
        title_text="share of the children",
        gridcolor=_grid,
        zeroline=False,
        tickfont={"color": _muted},
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
                "**Its cohort shares** (each cohort's children over all the children)"
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


if __name__ == "__main__":
    app.run()
