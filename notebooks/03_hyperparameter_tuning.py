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
    # The metric that picks each model's best run, by its composite score on
    # validation: "rmse" or "poisson_deviance" (D34; the deviance still tunes).
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
    cross-validated Poisson deviance (the total times probability model by
    coordinate descent until PR 7). Each run's best model is refit on the
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

    from age_group_prediction.hyperparameter_tuning import (
        CountComponent,
        CrossValidationTuner,
        LogisticParams,
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
        LogisticParams,
        Metric,
        PermutationImportance,
        ProbabilityComponent,
        ShapExplainer,
        build_independent_cohorts_model,
        build_total_times_probability_model,
        compute_metrics,
        dataclasses,
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
    root_mean_squared_error_loss,
    selection_metric,
    wmape_loss,
):
    # The primary metric: the study's objective and the permutation
    # importance's score, with the direction Optuna is given; a loss, so
    # "minimize".
    POISSON_DEVIANCE = Metric(
        "poisson_deviance", mean_poisson_deviance_loss, direction="minimize"
    )
    LOG_LOSS = Metric("log_loss", multinomial_log_loss, direction="minimize")
    RMSE = Metric("rmse", root_mean_squared_error_loss, direction="minimize")

    # Every metric, the primary first: each run's score table (each cohort
    # through `cohort=`, then the composite) and the bootstrap of the best
    # runs. The count metrics pool the (building, cohort) cells. Bias is
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
    # The selection metric: it picks each model's best run (the composite's
    # point score on validation) and is the forest plots' metric.
    if selection_metric == "rmse":
        SELECTION_METRIC = RMSE
    elif selection_metric == "poisson_deviance":
        SELECTION_METRIC = POISSON_DEVIANCE
    else:
        raise ValueError(f"unknown selection_metric {selection_metric!r}")
    return METRICS, POISSON_DEVIANCE, SELECTION_METRIC


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
    POISSON_DEVIANCE,
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
    # The study's design, the same for every run: the Poisson deviance of the
    # composite on each fold, the plain mean over them, the first
    # n_startup_trials of n_trials at random; each trial fits its folds in
    # n_splits processes (D13, D20, D26) and records its cross-validated score
    # on every metric, from the same fold predictions.
    tuner = CrossValidationTuner(
        cv=cv,
        objective_metric=POISSON_DEVIANCE,
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
def _(LogisticParams, PROBABILITY_FEATURE_TRANSFORMERS, ProbabilityComponent):
    # The total times probability model's fixed probabilities in coordinate
    # descent's step 1, here and in the objective EDA. The composite
    # deviance is the total's deviance plus the probabilities' log loss, so a
    # fixed classifier adds the same constant to every trial and any one finds
    # the same best total: the cheapest.
    MODEL2_FIXED_PROBABILITY_COMPONENT = ProbabilityComponent(
        name="probability",
        family="logistic",
        calibration_method=None,
        estimator_params=LogisticParams(C=1.0),
        feature_transformer=PROBABILITY_FEATURE_TRANSFORMERS["logistic"]["glm_base"],
    )
    return (MODEL2_FIXED_PROBABILITY_COMPONENT,)


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
                    component.name: next(
                        transformer_name
                        for transformer_name, transformer in COUNT_FEATURE_TRANSFORMERS[
                            _family
                        ].items()
                        if transformer is component.feature_transformer
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
def _(MODEL1_RUNS, dataclasses, pd, tuner):
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
                    component.name: dataclasses.asdict(component.estimator_params)
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
    # The total times probability model by coordinate descent (2.3a). Step 1
    # tunes the total, one run per count family × exposure on or off, in the
    # order of their seeds (the i-th study gets seed + i); step 2 tunes the
    # probabilities, one run per classifier family, seeded after step 1's.
    MODEL2_TOTAL_RUN_SETTINGS = [
        {"total_family": "lgbm_poisson", "use_exposure": True},
        {"total_family": "lgbm_poisson", "use_exposure": False},
        {"total_family": "lgbm_gaussian", "use_exposure": True},
        {"total_family": "lgbm_gaussian", "use_exposure": False},
        {"total_family": "poisson_glm", "use_exposure": True},
        {"total_family": "poisson_glm", "use_exposure": False},
        {"total_family": "nb2", "use_exposure": True},
        {"total_family": "nb2", "use_exposure": False},
    ]
    MODEL2_PROBABILITY_RUN_SETTINGS = [
        {"probability_family": "logistic"},
        {"probability_family": "lgbm"},
    ]
    return MODEL2_PROBABILITY_RUN_SETTINGS, MODEL2_TOTAL_RUN_SETTINGS


@app.cell
def _(mo):
    run_model2 = mo.ui.run_button(label="Tune the total times probability model")
    mo.md(
        f"{run_model2} by coordinate descent: 8 runs of the total, one per entry of "
        "`MODEL2_TOTAL_RUN_SETTINGS`, then 2 of the probabilities, one per entry of "
        "`MODEL2_PROBABILITY_RUN_SETTINGS`, with the best total fixed; then the 16 "
        "pairs scored. Each trial fits its folds in parallel."
    )
    return (run_model2,)


@app.cell
def _(
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    MODEL2_FIXED_PROBABILITY_COMPONENT,
    MODEL2_TOTAL_RUN_SETTINGS,
    PREDICTION_FLOOR,
    build_study_name,
    cv_groups,
    mo,
    run_model2,
    seed,
    train_age_group,
    tuner,
):
    mo.stop(
        not run_model2.value,
        mo.md("Click the button to tune the total times probability model."),
    )
    # Step 1: the totals, one after another, the i-th seeded seed + i, the
    # probabilities fixed. A failed trial is caught by the study; any other
    # error ends the loop, and the records made so far are lost.
    MODEL2_TOTAL_RUNS = []
    for _i, _run_settings in enumerate(MODEL2_TOTAL_RUN_SETTINGS):
        _total_family = _run_settings["total_family"]
        _use_exposure = _run_settings["use_exposure"]
        _study_seed = seed + _i
        _study, _seconds = tuner.tune_total_component(
            train_age_group.X,
            train_age_group.Y,
            train_age_group.exposure,
            cv_groups,
            total_family=_total_family,
            use_exposure=_use_exposure,
            total_feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_total_family],
            probability_component=MODEL2_FIXED_PROBABILITY_COMPONENT,
            y_min=PREDICTION_FLOOR,
            random_state=seed,
            study_seed=_study_seed,
            study_name=build_study_name(
                "total_times_probability_coordinate_descent_total",
                _run_settings,
                tuner.objective_metric.name,
                _study_seed,
            ),
        )
        # The best total, read from the best trial's parameters by its type
        # (D25); the run keeps it, its transformer's name, its value, trial
        # counts and seconds. The trials are not kept: the objective EDA studies
        # them.
        _total_component = CountComponent.from_params(
            _study.best_params,
            name="total",
            family=_total_family,
            use_exposure=_use_exposure,
            feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_total_family],
        )
        _states = [trial.state.name for trial in _study.trials]
        MODEL2_TOTAL_RUNS.append(
            {
                **_run_settings,
                "feature_transformers": {
                    "total": next(
                        transformer_name
                        for transformer_name, transformer in COUNT_FEATURE_TRANSFORMERS[
                            _total_family
                        ].items()
                        if transformer is _total_component.feature_transformer
                    )
                },
                "component": _total_component,
                "cv_best": _study.best_value,
                "n_complete": _states.count("COMPLETE"),
                "n_failed": _states.count("FAIL"),
                "seconds": _seconds,
            }
        )
    return (MODEL2_TOTAL_RUNS,)


@app.cell
def _(
    MODEL2_PROBABILITY_RUN_SETTINGS,
    MODEL2_TOTAL_RUNS,
    MODEL2_TOTAL_RUN_SETTINGS,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_study_name,
    cv_groups,
    mo,
    run_model2,
    seed,
    train_age_group,
    tuner,
):
    mo.stop(
        not run_model2.value,
        mo.md("Click the button to tune the total times probability model."),
    )
    # Step 2: the probabilities, one after another, seeded after step 1's runs,
    # step 1's best total by cv fixed. The total adds the same constant to
    # every trial, so each run's best value is the cv of its pair with that
    # total.
    _best_total_component = min(MODEL2_TOTAL_RUNS, key=lambda run: run["cv_best"])[
        "component"
    ]
    MODEL2_PROBABILITY_RUNS = []
    for _j, _run_settings in enumerate(MODEL2_PROBABILITY_RUN_SETTINGS):
        _probability_family = _run_settings["probability_family"]
        _study_seed = seed + len(MODEL2_TOTAL_RUN_SETTINGS) + _j
        _study, _seconds = tuner.tune_probability_component(
            train_age_group.X,
            train_age_group.Y,
            train_age_group.exposure,
            cv_groups,
            total_component=_best_total_component,
            probability_family=_probability_family,
            probability_feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                _probability_family
            ],
            y_min=PREDICTION_FLOOR,
            random_state=seed,
            study_seed=_study_seed,
            study_name=build_study_name(
                "total_times_probability_coordinate_descent_probability",
                _run_settings,
                tuner.objective_metric.name,
                _study_seed,
            ),
        )
        _probability_component = ProbabilityComponent.from_params(
            _study.best_params,
            name="probability",
            family=_probability_family,
            feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                _probability_family
            ],
        )
        _states = [trial.state.name for trial in _study.trials]
        MODEL2_PROBABILITY_RUNS.append(
            {
                **_run_settings,
                "feature_transformers": {
                    "probability": next(
                        transformer_name
                        for transformer_name, transformer in PROBABILITY_FEATURE_TRANSFORMERS[
                            _probability_family
                        ].items()
                        if transformer is _probability_component.feature_transformer
                    )
                },
                "component": _probability_component,
                "cv_best": _study.best_value,
                "n_complete": _states.count("COMPLETE"),
                "n_failed": _states.count("FAIL"),
                "seconds": _seconds,
            }
        )
    return (MODEL2_PROBABILITY_RUNS,)


@app.cell
def _(
    COHORTS,
    METRICS,
    MODEL2_PROBABILITY_RUNS,
    MODEL2_TOTAL_RUNS,
    PREDICTION_FLOOR,
    build_total_times_probability_model,
    compute_metrics,
    cv_groups,
    pd,
    seed,
    train_age_group,
    tuner,
    val_age_group,
):
    # The 16 pairs, each step-1 total with each step-2 classifier, their runs'
    # best components, scored with no search: the pair's cross-validated
    # value, then the refit on the training part scored on training and
    # validation (point scores).
    MODEL2_RUNS = []
    for _total_run in MODEL2_TOTAL_RUNS:
        for _probability_run in MODEL2_PROBABILITY_RUNS:
            _best_components = (
                _total_run["component"],
                _probability_run["component"],
            )
            _model = build_total_times_probability_model(
                *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
            )
            _cv_best = tuner.cross_validate(
                _model,
                train_age_group.X,
                train_age_group.Y,
                train_age_group.exposure,
                cv_groups,
            )[tuner.objective_metric.name]
            _model.fit(
                train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure
            )
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
            # The pair's settings and its best model: the fitted model, its
            # components and their transformers' names; its value, and its two
            # studies' trial counts and seconds summed (a study shared by
            # several pairs counts in each).
            MODEL2_RUNS.append(
                {
                    "total_family": _total_run["total_family"],
                    "probability_family": _probability_run["probability_family"],
                    "use_exposure": _total_run["use_exposure"],
                    "feature_transformers": {
                        **_total_run["feature_transformers"],
                        **_probability_run["feature_transformers"],
                    },
                    "model": _model,
                    "components": _best_components,
                    "cv_best": _cv_best,
                    "n_complete": _total_run["n_complete"]
                    + _probability_run["n_complete"],
                    "n_failed": _total_run["n_failed"] + _probability_run["n_failed"],
                    "seconds": _total_run["seconds"] + _probability_run["seconds"],
                    "scores": pd.concat(_scores_by_dataset, names=["dataset"]),
                }
            )
    return (MODEL2_RUNS,)


@app.cell
def _(MODEL2_RUNS, ProbabilityComponent, dataclasses, pd, tuner):
    _run_rows, _score_rows = [], []
    for _run in MODEL2_RUNS:
        # A family pair reads as one family, so both models' tables have the same
        # settings columns; the record keeps the two families (D16).
        _settings = {
            "family": f"{_run['total_family']}+{_run['probability_family']}",
            "use_exposure": _run["use_exposure"],
        }
        # The kept model, its two studies merged: each component's feature
        # transformer and hyperparameters (the classifier's with its
        # calibration), the pair's cross-validated objective, the two studies'
        # trials and seconds.
        _hyperparameters = {}
        for _component in _run["components"]:
            _hyperparameters[_component.name] = dataclasses.asdict(
                _component.estimator_params
            )
            if isinstance(_component, ProbabilityComponent):
                _hyperparameters[_component.name]["calibration_method"] = (
                    _component.calibration_method
                )
        _run_rows.append(
            {
                **_settings,
                "feature_transformers": _run["feature_transformers"],
                "hyperparameters": _hyperparameters,
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
                "**The total times probability model by coordinate descent: each "
                "pair's model** (family: the total's + the probabilities'; each "
                "component's feature transformer and hyperparameters; the pair's "
                "cross-validated objective; its two studies' trials and seconds, "
                "summed)"
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
    mo.vstack(
        [
            mo.md(
                f"**Each model's best run by the {SELECTION_METRIC.name} of the "
                "composite on validation, bootstrapped on validation:** "
                + "; ".join(
                    f"{model} `{family}`, exposure {use_exposure}"
                    for model, family, use_exposure in _scores_by_run
                )
                + " (each score with its replicates' mean, median, SD and 95% "
                "interval)"
            ),
            mo.ui.table(best_runs_val_scores_df.round(4)),
        ]
    )
    return best_model1, best_model2, best_runs_val_scores_df


@app.cell
def _(go, make_subplots):
    def forest_figure(scores, metric_name, lower_column, upper_column, rule_at=None):
        """A forest plot of `metric_name`: a panel per cohort, each on its own
        scale, and a row per run (the first on top), the dot its score and
        the bar its interval from `lower_column` to `upper_column`. `scores`
        is long, a row per run, cohort and metric, with the columns `model`,
        `family`, `use_exposure`, `cohort`, `metric`, `score` and the two
        bounds. `rule_at` draws a vertical line at that value in every panel
        (0 for a difference)."""
        surface, ink, color = "#fcfcfb", "#52514e", "#2a78d6"
        metric_scores = scores[scores["metric"] == metric_name]
        cohorts = list(metric_scores["cohort"].unique())
        figure = make_subplots(
            rows=1, cols=len(cohorts), shared_yaxes=True, subplot_titles=cohorts
        )
        for column, cohort in enumerate(cohorts, start=1):
            cohort_scores = metric_scores[metric_scores["cohort"] == cohort]
            score = cohort_scores["score"]
            run_labels = (
                cohort_scores["model"]
                + ": "
                + cohort_scores["family"]
                + ", exposure "
                + cohort_scores["use_exposure"].astype(str)
            )
            figure.add_trace(
                go.Scatter(
                    x=score,
                    y=run_labels,
                    mode="markers",
                    marker={"size": 9, "color": color},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": cohort_scores[upper_column] - score,
                        "arrayminus": score - cohort_scores[lower_column],
                        "color": color,
                        "thickness": 2,
                        "width": 6,
                    },
                    customdata=cohort_scores[[lower_column, upper_column]].to_numpy(),
                    hovertemplate=(
                        f"%{{y}}<br>{cohort} {metric_name} %{{x:.4f}}"
                        "<br>interval %{customdata[0]:.4f} to %{customdata[1]:.4f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            if rule_at is not None:
                figure.add_vline(
                    x=rule_at, line_width=1, line_color=ink, row=1, col=column
                )
        figure.update_yaxes(autorange="reversed")
        figure.update_layout(
            title=metric_name,
            height=300,
            template="plotly_white",
            paper_bgcolor=surface,
            plot_bgcolor=surface,
            hovermode="closest",
        )
        return figure

    return (forest_figure,)


@app.cell
def _(
    CI_BOUND_NAMES,
    SELECTION_METRIC,
    best_runs_val_scores_df,
    forest_figure,
    mo,
):
    best_forest_figure = forest_figure(
        best_runs_val_scores_df,
        SELECTION_METRIC.name,
        CI_BOUND_NAMES["lower"],
        CI_BOUND_NAMES["upper"],
    )
    mo.vstack(
        [
            mo.md(
                "**Each model's best run on validation: the selection metric per "
                "cohort and for the composite** (the dot its score, the bar its 95% "
                "bootstrap interval)"
            ),
            best_forest_figure,
        ]
    )
    return


@app.cell
def _(mo):
    run_inspection = mo.ui.run_button(label="Inspect the best runs")
    mo.md(
        f"{run_inspection} each model's best run: the permutation importance and "
        "SHAP (about two minutes)."
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
    # Permutation importance of the primary metric: on validation the decision
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
        ("model1", model1_permutation, model1_shap),
        ("model2", model2_permutation, model2_shap),
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
