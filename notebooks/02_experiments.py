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

    A synthetic population of buildings from `configs/simulation.yml`, with
    the overrides below. The simulator's own seed fixes the population; leave
    `simulation_seed` as `None` to keep the config's value.
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
    n_trials = 400
    # TPE's random start: the trials drawn at random before its model guides the draws.
    n_startup_trials = 20
    n_resamples = 1000
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
        "n_resamples": n_resamples,
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
        n_repeats,
        n_resamples,
        n_shap_background,
        n_shap_rows,
        n_splits,
        n_startup_trials,
        n_trials,
        seed,
        split_method,
        test_size,
        val_size,
    )


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
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) The objective EDA

    Which objective should the tuner minimize? One run per model of section
    6, with the same setting for both (`lgbm_poisson`, exposure on), is
    tuned three times, once on each of `EDA_OBJECTIVES`: the log loss, the
    Poisson deviance and RMSE of the composite. The three studies of a run
    share their seed, so their 20 random startup trials are the same draws
    and only the objective differs. Each study's best model is refit on the
    training part and scored on validation on every metric, per cohort and
    for the composite; `cv_best` is the study's value in its own objective.
    The independent cohorts model keeps the sharing flags of its runs in
    section 6 (one set of hyperparameters, a transformer per cohort). The
    expected caveat: the log loss scores each row's proportions only (the
    row divided by its sum), so under it the independent cohorts model's
    scale gets no signal.
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
        LogisticParams,
        ProbabilityComponent,
        ProbabilitySearchSpace,
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
        evaluate,
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
        ProbabilitySearchSpace,
        ShapExplainer,
        build_independent_cohorts_model,
        build_total_times_probability_model,
        dataclasses,
        evaluate,
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
    wmape_loss,
):
    # The primary metric: the study's objective, the best run's rule and the
    # permutation importance's score, with the direction Optuna is given; a
    # loss, so "minimize".
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
    # The objective EDA's objectives (section 5), the same objects as in METRICS.
    EDA_OBJECTIVES = (LOG_LOSS, POISSON_DEVIANCE, RMSE)
    return EDA_OBJECTIVES, METRICS, POISSON_DEVIANCE


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
    POISSON_DEVIANCE,
    cv,
    n_splits,
    n_startup_trials,
    n_trials,
):
    # The study's design, the same for every run: the Poisson deviance of the
    # composite on each fold, the plain mean over them, the first
    # n_startup_trials of n_trials at random; each trial fits its folds in
    # n_splits processes (D13, D20, D26).
    tuner = CrossValidationTuner(
        cv=cv,
        metric=POISSON_DEVIANCE,
        n_trials=n_trials,
        n_startup_trials=n_startup_trials,
        n_jobs=n_splits,
    )
    return (tuner,)


@app.cell
def _(mo):
    run_eda = mo.ui.run_button(label="Run the objective EDA")
    mo.md(
        f"{run_eda} One run per model, tuned once on each of `EDA_OBJECTIVES`; "
        "each trial fits its folds in parallel."
    )
    return (run_eda,)


@app.cell
def _(
    COHORTS,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    EDA_OBJECTIVES,
    METRICS,
    PREDICTION_FLOOR,
    build_independent_cohorts_model,
    cv_groups,
    dataclasses,
    evaluate,
    mo,
    pd,
    run_eda,
    seed,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(not run_eda.value, mo.md("Click the button to run the objective EDA."))
    # The independent cohorts model's EDA run: one study per objective, each
    # seeded seed, so the three share their startup draws (M8); the tuner's
    # design but its metric. A failed trial is caught by the study.
    _family, _use_exposure = "lgbm_poisson", True
    _scores_by_objective = {}
    for _objective in EDA_OBJECTIVES:
        _tuner = dataclasses.replace(tuner, metric=_objective)
        _study, _seconds = _tuner.tune_independent_cohorts_model(
            train_age_group.X,
            train_age_group.Y,
            train_age_group.exposure,
            cv_groups,
            family=_family,
            use_exposure=_use_exposure,
            cohorts=COHORTS,
            feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
            share_hyperparams=True,
            share_feature_transformer=False,
            y_min=PREDICTION_FLOOR,
            random_state=seed,
            study_seed=seed,
        )
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
        _model = build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ).fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
        # Every metric on validation, per cohort and for the composite; the
        # study's value (in its own objective) and its seconds on each row.
        _scores_by_objective[_objective.name] = evaluate(
            val_age_group.Y,
            _model.predict(val_age_group.X, exposure=val_age_group.exposure),
            METRICS,
            cohorts=[*COHORTS, "composite"],
        ).assign(cv_best=_study.best_value, seconds=_seconds)
    eda_model1_df = pd.concat(_scores_by_objective, names=["objective"]).reset_index()
    return (eda_model1_df,)


@app.cell
def _(eda_model1_df, mo):
    mo.vstack(
        [
            mo.md(
                "**The objective EDA: the independent cohorts model** (each "
                "objective's best model refit; every metric on validation, per "
                "cohort and for the composite; cv_best: the study's value in its "
                "own objective; seconds: the study's)"
            ),
            mo.ui.table(eda_model1_df.round(4)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Hyperparameter tuning

    Two models: the independent cohorts model (Model 1, one count model per
    cohort) and the total times probability model (Model 2, a count model of
    the total times a classifier's cohort probabilities). One run per model
    family and exposure (on or off, fixed for the run):
    one Optuna study on the composite, so the model that is scored is the
    model that is tuned, by the Poisson deviance of the composite on its
    cross-validation folds on the training part, the plain mean over folds.
    A trial draws, per cohort (one part per run for the total times
    probability model), the estimator's hyperparameters and the feature
    transformer,
    under the names `<component>__<parameter>` (the transformer by its name,
    resolved to the object); they make a component, a complete record of
    that part, and the composite is built from the components with
    constructor calls; the best run's components are read back from the
    study's `best_params` by the component type, the same records. Two
    run settings decide what the independent cohorts model's cohorts share,
    each drawn once under `shared__`:
    `share_hyperparams` (one set of hyperparameters for all cohorts) and
    `share_feature_transformer` (one transformer for all cohorts). Each
    entry of `MODEL1_RUN_SETTINGS` sets both: its eight runs take `True` and
    `False`, and a run with other values is one more entry in that list.
    LightGBM's two losses, Poisson and Gaussian, are
    separate families, and the exposure a run setting: a study that sampled
    the loss split 30 trials 10/20 and did worse than either loss alone, and
    the exposure changes the target (a count or a weighted rate).
    `n_trials = 400`, the first `n_startup_trials = 20` drawn at random; the
    tables of trials per choice show whether any was tried too rarely to be
    judged. The runs go one after another, seeded
    `seed + 0`, `seed + 1`, … in order (each model's in loops over its run
    settings), and each trial fits its 5 folds in
    parallel. A trial whose fit raises (an NB2 fit that does not converge) is
    marked failed and the run goes on; the tables count them. Each run keeps
    every trial and its best components, which alone are read back from the
    best trial's parameters; the independent cohorts model's best, and the
    total times probability model's pairs (below), are refit on the training
    part and scored on training and validation on every metric, per cohort
    and for the composite. A Gaussian loss can predict a negative count,
    which `CountModel` floors at `y_min` (eps). The total times probability
    model is tuned by coordinate descent, under `total__…` and
    `probability__…`. First its total: one run per total family and exposure
    (8 runs, `MODEL2_TOTAL_RUN_SETTINGS`), the probabilities fixed at a plain
    logistic model. Then its probabilities: one run per classifier family
    (2 runs, `MODEL2_PROBABILITY_RUN_SETTINGS`), the best total by cv fixed;
    the probability model ignores the exposure, and its calibration is
    searched. Each run scores the whole composite: its cohort Poisson
    deviance is the total's deviance plus the probabilities' log loss, and no
    hyperparameter reaches both, so the fixed part adds the same constant to
    every trial, and one round finds what tuning both at once finds, about 3×
    faster (`docs/NOTEBOOK_METRICS_LOOP_EDA_PLAN.md` 2.3a). The 16 pairs, each
    total run's best with each classifier run's best, are then
    cross-validated and scored, with no search. Its replication of the
    counts into classifier
    rows is fixed at `weighted`, not searched: for the logistic classifier
    the two replications are the same fit, and for LightGBM they differed
    within noise while `per_child` is 2-3× slower
    (`docs/TOTAL_TIMES_PROBABILITY_MODEL.md` §3.1).
    The last table shows each model's best run on validation. The run
    cells sit behind buttons: opening the notebook fits nothing.

    Each model's best run is then inspected, behind its own button: its
    model is rebuilt from the best trial's parameters, refit on the training
    part and judged on the training rows and the validation rows (the test
    part waits for the final refit on training + validation). A bootstrap
    that resamples whole neighborhoods gives the interval of six metrics,
    per cohort and for the composite, all paired on the same replicates:
    the Poisson deviance (the
    primary metric, the tuning's objective); the log loss (the
    proportions between cohorts only, so it never selects); MAE and RMSE per
    (building, cohort) cell; WMAPE (the absolute error as a share of the
    children); and the bias (mean predicted minus true count, signed, so 0
    is best). The permutation importance of the primary metric on validation
    says what the shipped prediction depends on, and on training what the
    model fitted (a column important there only is overfit). Exact SHAP of
    `n_shap_rows` validation rows against `n_shap_background` training rows
    attributes the predicted count per row, column and cohort; mean |SHAP|
    per column is its global importance. The last table puts the two
    importances' rankings side by side.
    """)
    return


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
    cv_groups,
    evaluate,
    mo,
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
            study_seed=seed + _i,
        )
        # The best components, read from the best trial's parameters by their
        # type, are the model scored and the record kept (D10-D12, D14, D25).
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
        # and validation, per cohort and for the composite.
        _model = build_independent_cohorts_model(
            _best_components, y_min=PREDICTION_FLOOR, random_state=seed
        ).fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
        _train_scores = evaluate(
            train_age_group.Y,
            _model.predict(train_age_group.X, exposure=train_age_group.exposure),
            METRICS,
            cohorts=[*COHORTS, "composite"],
        )
        _val_scores = evaluate(
            val_age_group.Y,
            _model.predict(val_age_group.X, exposure=val_age_group.exposure),
            METRICS,
            cohorts=[*COHORTS, "composite"],
        )
        # The run's settings, every trial and the best (D14); the study object
        # is not kept.
        MODEL1_RUNS.append(
            {
                **_run_settings,
                "trials": _study.trials_dataframe(),
                "components": [component.to_dict() for component in _best_components],
                "cv_best": _study.best_value,
                "best_params": _study.best_params,
                "train_scores": _train_scores,
                "val_scores": _val_scores,
                "seconds": _seconds,
                "seed": seed,
                "y_min": PREDICTION_FLOOR,
            }
        )
    return (MODEL1_RUNS,)


@app.cell
def _(COUNT_FEATURE_TRANSFORMERS, MODEL1_RUNS, POISSON_DEVIANCE, pd):
    _score_rows, _arm_rows = [], []
    for _run in MODEL1_RUNS:
        _settings = {
            "family": _run["family"],
            "use_exposure": _run["use_exposure"],
            "share_hyperparams": _run["share_hyperparams"],
            "share_feature_transformer": _run["share_feature_transformer"],
        }
        _trials = _run["trials"]
        _complete = _trials[_trials["state"] == "COMPLETE"]
        # Every metric per cohort and for the composite, a row each with its
        # training and validation scores; the run's value and its trials on its
        # primary metric's composite row.
        _scores = pd.concat(
            {"train": _run["train_scores"].stack(), "val": _run["val_scores"].stack()},
            axis=1,
        ).reset_index()
        for _row in _scores.to_dict("records"):
            if (
                _row["cohort"] == "composite"
                and _row["metric"] == POISSON_DEVIANCE.name
            ):
                _score_rows.append(
                    {
                        **_settings,
                        **_row,
                        "cv_best": _run["cv_best"],
                        "n_complete": len(_complete),
                        "n_failed": int((_trials["state"] == "FAIL").sum()),
                        "seconds": _run["seconds"],
                    }
                )
            else:
                _score_rows.append({**_settings, **_row})
        # The completed trials per feature transformer, the model's categorical
        # choice (one column per cohort, or `shared__`): TPE concentrates on what looked good early, so a transformer
        # tried a few times has not been judged (D14). Every candidate of the
        # family is listed, so one never tried shows 0 rather than nothing.
        _candidates = list(COUNT_FEATURE_TRANSFORMERS[_run["family"]])
        for _column in _trials.columns:
            if _column.endswith("__feature_transformer"):
                _counts = (
                    _complete[_column].value_counts().reindex(_candidates, fill_value=0)
                )
                for _value, _n in _counts.items():
                    _arm_rows.append(
                        {
                            **_settings,
                            "parameter": _column.removeprefix("params_"),
                            "value": _value,
                            "n_trials": int(_n),
                        }
                    )
    model1_df, model1_arms_df = pd.DataFrame(_score_rows), pd.DataFrame(_arm_rows)
    return model1_arms_df, model1_df


@app.cell
def _(mo, model1_arms_df, model1_df):
    mo.vstack(
        [
            mo.md(
                "**The independent cohorts model: every metric, per cohort and for the "
                "composite** (cv_best: the run's value; n_complete / n_failed: its "
                "trials; on the primary metric's composite row)"
            ),
            mo.ui.table(model1_df.round(4)),
            mo.md("**Trials per feature transformer**"),
            mo.md(f"```text\n{model1_arms_df.to_string(index=False)}\n```"),
        ]
    )
    return


@app.cell
def _(LogisticParams, PROBABILITY_FEATURE_TRANSFORMERS, ProbabilityComponent):
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
    # Step 1's fixed probabilities. The composite deviance is the total's
    # deviance plus the probabilities' log loss, so a fixed classifier adds the
    # same constant to every trial and any one finds the same best total: the
    # cheapest.
    MODEL2_FIXED_PROBABILITY_COMPONENT = ProbabilityComponent(
        name="probability",
        family="logistic",
        calibration_method=None,
        estimator_params=LogisticParams(C=1.0),
        feature_transformer=PROBABILITY_FEATURE_TRANSFORMERS["logistic"]["glm_base"],
    )
    return (
        MODEL2_FIXED_PROBABILITY_COMPONENT,
        MODEL2_PROBABILITY_RUN_SETTINGS,
        MODEL2_TOTAL_RUN_SETTINGS,
    )


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
            study_seed=seed + _i,
        )
        # The best total, read from the best trial's parameters by its type
        # (D25); the run keeps every trial and the best (D14), the study object
        # is not kept.
        _total_component = CountComponent.from_params(
            _study.best_params,
            name="total",
            family=_total_family,
            use_exposure=_use_exposure,
            feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_total_family],
        )
        MODEL2_TOTAL_RUNS.append(
            {
                **_run_settings,
                "trials": _study.trials_dataframe(),
                "component": _total_component.to_dict(),
                "fixed_component": MODEL2_FIXED_PROBABILITY_COMPONENT.to_dict(),
                "cv_best": _study.best_value,
                "best_params": _study.best_params,
                "seconds": _seconds,
                "seed": seed,
                "y_min": PREDICTION_FLOOR,
            }
        )
    return (MODEL2_TOTAL_RUNS,)


@app.cell
def _(
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    MODEL2_PROBABILITY_RUN_SETTINGS,
    MODEL2_TOTAL_RUNS,
    MODEL2_TOTAL_RUN_SETTINGS,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
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
    _best_total_run = min(MODEL2_TOTAL_RUNS, key=lambda run: run["cv_best"])
    _best_total_component = CountComponent.from_params(
        _best_total_run["best_params"],
        name="total",
        family=_best_total_run["total_family"],
        use_exposure=_best_total_run["use_exposure"],
        feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[
            _best_total_run["total_family"]
        ],
    )
    MODEL2_PROBABILITY_RUNS = []
    for _j, _run_settings in enumerate(MODEL2_PROBABILITY_RUN_SETTINGS):
        _probability_family = _run_settings["probability_family"]
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
            study_seed=seed + len(MODEL2_TOTAL_RUN_SETTINGS) + _j,
        )
        _probability_component = ProbabilityComponent.from_params(
            _study.best_params,
            name="probability",
            family=_probability_family,
            feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                _probability_family
            ],
        )
        MODEL2_PROBABILITY_RUNS.append(
            {
                **_run_settings,
                "trials": _study.trials_dataframe(),
                "component": _probability_component.to_dict(),
                "fixed_component": _best_total_component.to_dict(),
                "cv_best": _study.best_value,
                "best_params": _study.best_params,
                "seconds": _seconds,
                "seed": seed,
                "y_min": PREDICTION_FLOOR,
            }
        )
    return (MODEL2_PROBABILITY_RUNS,)


@app.cell
def _(
    COHORTS,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    METRICS,
    MODEL2_PROBABILITY_RUNS,
    MODEL2_TOTAL_RUNS,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    cv_groups,
    evaluate,
    seed,
    train_age_group,
    tuner,
    val_age_group,
):
    # The 16 pairs, each step-1 total with each step-2 classifier, read back
    # from their runs' best parameters by the component types (D25) and scored
    # with no search: the pair's cross-validated value, then the refit on the
    # training part scored on training and validation.
    MODEL2_RUNS = []
    for _total_run in MODEL2_TOTAL_RUNS:
        _total_component = CountComponent.from_params(
            _total_run["best_params"],
            name="total",
            family=_total_run["total_family"],
            use_exposure=_total_run["use_exposure"],
            feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[
                _total_run["total_family"]
            ],
        )
        for _probability_run in MODEL2_PROBABILITY_RUNS:
            _probability_component = ProbabilityComponent.from_params(
                _probability_run["best_params"],
                name="probability",
                family=_probability_run["probability_family"],
                feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                    _probability_run["probability_family"]
                ],
            )
            _best_components = (_total_component, _probability_component)
            _model = build_total_times_probability_model(
                *_best_components, y_min=PREDICTION_FLOOR, random_state=seed
            )
            _cv_best = tuner.cv_score(
                _model,
                train_age_group.X,
                train_age_group.Y,
                train_age_group.exposure,
                cv_groups,
            )
            _model.fit(
                train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure
            )
            _train_scores = evaluate(
                train_age_group.Y,
                _model.predict(train_age_group.X, exposure=train_age_group.exposure),
                METRICS,
                cohorts=[*COHORTS, "composite"],
            )
            _val_scores = evaluate(
                val_age_group.Y,
                _model.predict(val_age_group.X, exposure=val_age_group.exposure),
                METRICS,
                cohorts=[*COHORTS, "composite"],
            )
            # The pair's settings, its components and their merged parameters
            # (the `total__` and `probability__` names never clash); its runs'
            # trials are in their step records.
            MODEL2_RUNS.append(
                {
                    "total_family": _total_run["total_family"],
                    "probability_family": _probability_run["probability_family"],
                    "use_exposure": _total_run["use_exposure"],
                    "components": [
                        component.to_dict() for component in _best_components
                    ],
                    "cv_best": _cv_best,
                    "best_params": {
                        **_total_run["best_params"],
                        **_probability_run["best_params"],
                    },
                    "train_scores": _train_scores,
                    "val_scores": _val_scores,
                    "seed": seed,
                    "y_min": PREDICTION_FLOOR,
                }
            )
    return (MODEL2_RUNS,)


@app.cell
def _(
    COUNT_FEATURE_TRANSFORMERS,
    MODEL2_PROBABILITY_RUNS,
    MODEL2_RUNS,
    MODEL2_TOTAL_RUNS,
    POISSON_DEVIANCE,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilitySearchSpace,
    pd,
):
    _score_rows, _step_rows, _arm_rows = [], [], []
    for _run in MODEL2_RUNS:
        # A family pair reads as one family, so both models' tables have the same
        # settings columns; the record keeps the two families (D16).
        _settings = {
            "family": f"{_run['total_family']}+{_run['probability_family']}",
            "use_exposure": _run["use_exposure"],
        }
        # Every metric per cohort and for the composite, a row each with its
        # training and validation scores; the pair's value on its primary
        # metric's composite row.
        _scores = pd.concat(
            {"train": _run["train_scores"].stack(), "val": _run["val_scores"].stack()},
            axis=1,
        ).reset_index()
        for _row in _scores.to_dict("records"):
            if (
                _row["cohort"] == "composite"
                and _row["metric"] == POISSON_DEVIANCE.name
            ):
                _score_rows.append({**_settings, **_row, "cv_best": _run["cv_best"]})
            else:
                _score_rows.append({**_settings, **_row})
    # The 10 studies, each with its categorical choices and their candidates:
    # the total's transformer in step 1; the probabilities' transformer and
    # calibration in step 2. Every candidate is listed, so one never tried
    # shows 0 (D15).
    _studies = [
        (
            {
                "step": "total",
                "family": _run["total_family"],
                "use_exposure": _run["use_exposure"],
            },
            _run,
            {
                "params_total__feature_transformer": list(
                    COUNT_FEATURE_TRANSFORMERS[_run["total_family"]]
                )
            },
        )
        for _run in MODEL2_TOTAL_RUNS
    ] + [
        (
            {
                "step": "probability",
                "family": _run["probability_family"],
                "use_exposure": None,
            },
            _run,
            {
                "params_probability__feature_transformer": list(
                    PROBABILITY_FEATURE_TRANSFORMERS[_run["probability_family"]]
                ),
                "params_probability__calibration_method": (
                    ProbabilitySearchSpace.CALIBRATION_METHODS
                ),
            },
        )
        for _run in MODEL2_PROBABILITY_RUNS
    ]
    for _settings, _run, _candidates in _studies:
        _trials = _run["trials"]
        _complete = _trials[_trials["state"] == "COMPLETE"]
        _step_rows.append(
            {
                **_settings,
                "cv_best": _run["cv_best"],
                "n_complete": len(_complete),
                "n_failed": int((_trials["state"] == "FAIL").sum()),
                "seconds": _run["seconds"],
            }
        )
        for _column, _values in _candidates.items():
            # Counted by label: the trials table holds the None calibration as NaN
            # in a str column, or as None in an object column when every trial
            # drew it, and reindex on None misses the latter (D16).
            _labels = [str(_value) for _value in _values]
            _counts = (
                _complete[_column]
                .fillna("None")
                .value_counts()
                .reindex(_labels, fill_value=0)
            )
            for _value, _n in _counts.items():
                _arm_rows.append(
                    {
                        **_settings,
                        "parameter": _column.removeprefix("params_"),
                        "value": _value,
                        "n_trials": int(_n),
                    }
                )
    model2_df = pd.DataFrame(_score_rows)
    model2_steps_df = pd.DataFrame(_step_rows)
    model2_arms_df = pd.DataFrame(_arm_rows)
    return model2_arms_df, model2_df, model2_steps_df


@app.cell
def _(mo, model2_arms_df, model2_df, model2_steps_df):
    mo.vstack(
        [
            mo.md(
                "**The total times probability model by coordinate descent: its 10 runs** "
                "(step 1 the totals, step 2 the probabilities; cv_best: the run's value; "
                "n_complete / n_failed: its trials)"
            ),
            mo.md(f"```text\n{model2_steps_df.round(4).to_string(index=False)}\n```"),
            mo.md(
                "**Its 16 pairs: every metric, per cohort and for the composite** "
                "(family: the total's + the probabilities'; cv_best: the pair's "
                "cross-validated value, on the primary metric's composite row)"
            ),
            mo.ui.table(model2_df.round(4)),
            mo.md("**Trials per choice**"),
            mo.md(f"```text\n{model2_arms_df.to_string(index=False)}\n```"),
        ]
    )
    return


@app.cell
def _(POISSON_DEVIANCE, mo, model1_df, model2_df, pd):
    # Each model's best run on validation, from the primary metric's composite
    # rows. A fresh index: the two frames' labels overlap, and idxmin + loc
    # would return both.
    _composites = pd.concat(
        [
            _df[
                (_df["cohort"] == "composite")
                & (_df["metric"] == POISSON_DEVIANCE.name)
            ].assign(model=_model)
            for _model, _df in (("model1", model1_df), ("model2", model2_df))
        ],
        ignore_index=True,
    )
    best_df = _composites.loc[
        _composites.groupby("model")["val"].idxmin(),
        ["model", "family", "use_exposure", "cv_best", "train", "val"],
    ]
    mo.vstack(
        [
            mo.md(
                "**Best run per model** (validation Poisson deviance of the composite)"
            ),
            mo.md(f"```text\n{best_df.round(4).to_string(index=False)}\n```"),
        ]
    )
    return


@app.cell
def _(mo):
    run_inspection = mo.ui.run_button(label="Inspect the best runs")
    mo.md(
        f"{run_inspection} each model's best run: refit on the training part, the "
        "bootstrap of six metrics, the permutation importance and SHAP (about two "
        "minutes)."
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
    COHORTS,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    MODEL1_RUNS,
    POISSON_DEVIANCE,
    build_independent_cohorts_model,
    mo,
    run_inspection,
    train_age_group,
):
    mo.stop(
        not run_inspection.value, mo.md("Click the button to inspect the best runs.")
    )
    # The independent cohorts model's best run on validation (best_df's rule),
    # its model rebuilt from the best trial's parameters by the component
    # types and refit on the
    # training part.
    best_run1 = min(
        MODEL1_RUNS,
        key=lambda run: run["val_scores"].loc["composite", POISSON_DEVIANCE.name],
    )
    best_model1 = build_independent_cohorts_model(
        tuple(
            CountComponent.from_params(
                best_run1["best_params"],
                name=cohort,
                family=best_run1["family"],
                use_exposure=best_run1["use_exposure"],
                feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[
                    best_run1["family"]
                ],
            )
            for cohort in COHORTS
        ),
        y_min=best_run1["y_min"],
        random_state=best_run1["seed"],
    ).fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
    return best_model1, best_run1


@app.cell
def _(
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    MODEL2_RUNS,
    POISSON_DEVIANCE,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_total_times_probability_model,
    mo,
    run_inspection,
    train_age_group,
):
    mo.stop(
        not run_inspection.value, mo.md("Click the button to inspect the best runs.")
    )
    # The total times probability model's best run on validation, rebuilt and
    # refit the same way.
    best_run2 = min(
        MODEL2_RUNS,
        key=lambda run: run["val_scores"].loc["composite", POISSON_DEVIANCE.name],
    )
    best_model2 = build_total_times_probability_model(
        CountComponent.from_params(
            best_run2["best_params"],
            name="total",
            family=best_run2["total_family"],
            use_exposure=best_run2["use_exposure"],
            feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[
                best_run2["total_family"]
            ],
        ),
        ProbabilityComponent.from_params(
            best_run2["best_params"],
            name="probability",
            family=best_run2["probability_family"],
            feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                best_run2["probability_family"]
            ],
        ),
        y_min=best_run2["y_min"],
        random_state=best_run2["seed"],
    ).fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
    return best_model2, best_run2


@app.cell
def _(
    BootstrapEvaluator,
    COHORTS,
    METRICS,
    best_model1,
    mo,
    n_resamples,
    pd,
    run_inspection,
    seed,
    train_age_group,
    val_age_group,
):
    mo.stop(
        not run_inspection.value, mo.md("Click the button to inspect the best runs.")
    )
    # The six metrics' intervals per cohort and for the composite, on the
    # training rows and on the validation rows, resampling whole neighborhoods;
    # every cohort and metric scores the same replicates.
    _evaluator = BootstrapEvaluator(n_resamples, random_state=seed)
    model1_intervals = pd.concat(
        {
            "train": _evaluator.evaluate(
                train_age_group.Y,
                best_model1.predict(
                    train_age_group.X, exposure=train_age_group.exposure
                ),
                train_age_group.groups,
                metrics=METRICS,
                cohorts=[*COHORTS, "composite"],
            ),
            "val": _evaluator.evaluate(
                val_age_group.Y,
                best_model1.predict(val_age_group.X, exposure=val_age_group.exposure),
                val_age_group.groups,
                metrics=METRICS,
                cohorts=[*COHORTS, "composite"],
            ),
        },
        names=["split", "cohort", "metric"],
    )
    return (model1_intervals,)


@app.cell
def _(
    BootstrapEvaluator,
    COHORTS,
    METRICS,
    best_model2,
    mo,
    n_resamples,
    pd,
    run_inspection,
    seed,
    train_age_group,
    val_age_group,
):
    mo.stop(
        not run_inspection.value, mo.md("Click the button to inspect the best runs.")
    )
    _evaluator = BootstrapEvaluator(n_resamples, random_state=seed)
    model2_intervals = pd.concat(
        {
            "train": _evaluator.evaluate(
                train_age_group.Y,
                best_model2.predict(
                    train_age_group.X, exposure=train_age_group.exposure
                ),
                train_age_group.groups,
                metrics=METRICS,
                cohorts=[*COHORTS, "composite"],
            ),
            "val": _evaluator.evaluate(
                val_age_group.Y,
                best_model2.predict(val_age_group.X, exposure=val_age_group.exposure),
                val_age_group.groups,
                metrics=METRICS,
                cohorts=[*COHORTS, "composite"],
            ),
        },
        names=["split", "cohort", "metric"],
    )
    return (model2_intervals,)


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
def _(best_run1, best_run2, mo, model1_intervals, model2_intervals, pd):
    # Each metric's interval per model, split and cohort; the caption names
    # the two best runs.
    intervals_df = pd.concat(
        {"model1": model1_intervals, "model2": model2_intervals}, names=["model"]
    ).reset_index()
    mo.vstack(
        [
            mo.md(
                "**The best runs' metrics: their intervals** (mean, lower, upper: "
                "the bootstrap over neighborhoods; the scores themselves are the best "
                "runs' rows of the score tables). model1, the "
                f"independent cohorts model: `{best_run1['family']}`, exposure "
                f"{best_run1['use_exposure']}; model2, the total times probability "
                f"model: `{best_run2['total_family']}+{best_run2['probability_family']}`, "
                f"exposure {best_run2['use_exposure']}."
            ),
            mo.ui.table(intervals_df.round(4)),
        ]
    )
    return


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
