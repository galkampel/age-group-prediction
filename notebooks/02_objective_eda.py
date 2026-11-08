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
    # Age Group Prediction: the objective EDA
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

    Every value below is logged as a parameter of every run. One `seed` fixes
    every random component: both holdouts, the cross-validation folds, the
    Optuna sampler (every study gets `seed`) and every estimator that takes a
    `random_state`.
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
        + " ".join(f"{k}={v}" for k, v in EXPERIMENT_SETTINGS.items())
    )
    mo.md(f"```text\n{pd.DataFrame([EXPERIMENT_SETTINGS]).to_string(index=False)}\n```")
    return (
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
    do. The interactions are §8.5's candidates (the first choice per stage,
    §6.2-§6.3) plus §6's second choice per stage and §6.5's recommended pair
    per stage. Every model fits its own copy of a variant.
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
    ## 5) The objective EDA

    Which objective should the tuner minimize? One run per model of the
    tuning notebook (`03_hyperparameter_tuning.py`), with the same setting
    for both (`lgbm_poisson`, exposure on), is
    tuned three times, once on each of `EDA_OBJECTIVES`: the log loss, the
    Poisson deviance and RMSE of the composite. The three studies of a run
    share their seed, so their random startup trials are the same draws
    and only the objective differs. Each study's best model is refit on the
    training part and scored on validation on every metric, per cohort and
    for the composite; `cv_best` is the study's value in its own objective.
    Every trial is kept too, in the trials tables, as a row of the whole
    model it scored: its id, state, seconds, cross-validated score on the
    six metrics, and each component's feature transformer and
    hyperparameters, to see which are good or bad on each. A summary table
    holds each model's best model per objective: its best trial's id, its
    cross-validated scores and the refit's composite on validation. The
    figures show, per model, every metric per trial of the study of the
    objective chosen in the dropdown, in stacked panels sharing the trial
    axis: the best trial a red star in each, and a rule where the total
    times probability model's probability step starts. The comparison
    figures show, per model, the recorded metric chosen in their dropdown
    for each objective's study: a box with every trial a point and its best
    trial a star, each objective in its own color and marker (a click on
    the legend hides one), the y-axis over the bulk of the trials. The
    independent cohorts model keeps the sharing flags of its runs in the
    tuning notebook (one set of hyperparameters, a transformer per cohort).
    The total times probability model is tuned by coordinate descent: its
    total with the probabilities fixed at a plain logistic model, then its
    logistic probabilities with that best total fixed; its `cv_best` is the
    second study's value, the pair's, and its seconds are the two studies'.
    It is also tuned by the joint method, to compare the two at this
    budget: one study per objective whose every trial draws both
    components (`tune_total_times_probability_model`), its `cv_best` and
    seconds the study's. Coordinate descent gives each part its own
    `n_trials` and is exact under the Poisson deviance only; the joint study
    searches both parts' dimensions at once in `n_trials`. At 100 trials the
    two were the same within the noise, so the tuning notebook tunes it by
    the joint method (D40). The tuner's own objective is `objective_metric`
    (RMSE since D42); each EDA study replaces it with its objective.
    The expected caveat: the log loss scores each row's proportions only
    (the row divided by its sum), so under it the independent cohorts
    model's scale and the total times probability model's total get no
    signal (the total changes nothing in the log loss, so every trial of its
    study scores the same).

    Every study, here and in the tuning notebook, is stored in its own file
    under `studies/` (Optuna's journal file; git-ignored), and a click loads
    a stored study instead of tuning it, with the seconds its tuning took. The
    directory's name holds what shapes every study: the data's config hash,
    the split and its folds, the seed and the budget; the file's name holds
    what the study tunes (the model, or the total times probability model's
    coordinate-descent step), its run settings, its objective and its seed.
    So a changed setting is a new study, and a study run in both notebooks
    (each model's first run under the tuner's objective, the total times
    probability model's by the joint method) is tuned once. Delete
    a study's file to tune it again: a change to the code or the search
    spaces is not detected.
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
        LogisticParams,
        Metric,
        ProbabilityComponent,
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

    # Every metric: what every trial records and the refit best models'
    # validation scores (each cohort through `cohort=`, then the composite).
    # The count metrics pool the (building, cohort) cells. Bias is signed, so
    # its direction is nominal.
    METRICS = (
        POISSON_DEVIANCE,
        LOG_LOSS,
        Metric("mae", mean_absolute_error_loss, direction="minimize"),
        RMSE,
        Metric("wmape", wmape_loss, direction="minimize"),
        Metric("bias", mean_bias_error, direction="minimize"),
    )
    # The objective metric: the tuner's objective, the score every study
    # minimizes (the EDA's studies replace it with each of EDA_OBJECTIVES).
    if objective_metric == "rmse":
        OBJECTIVE_METRIC = RMSE
    elif objective_metric == "poisson_deviance":
        OBJECTIVE_METRIC = POISSON_DEVIANCE
    else:
        raise ValueError(f"unknown objective_metric {objective_metric!r}")
    # The objective EDA's objectives, the same objects as in METRICS.
    EDA_OBJECTIVES = (LOG_LOSS, POISSON_DEVIANCE, RMSE)
    return EDA_OBJECTIVES, METRICS, OBJECTIVE_METRIC


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
def _(LogisticParams, PROBABILITY_FEATURE_TRANSFORMERS, ProbabilityComponent):
    # The total times probability model's fixed probabilities in coordinate
    # descent's step 1. The composite deviance is the total's deviance plus
    # the probabilities' log loss, so a fixed classifier adds the same
    # constant to every trial and any one finds the same best total: the
    # cheapest.
    MODEL2_FIXED_PROBABILITY_COMPONENT = ProbabilityComponent(
        name="probability",
        family="logistic",
        calibration_method=None,
        estimator_params=LogisticParams(C=1.0),
        feature_transformer=PROBABILITY_FEATURE_TRANSFORMERS["logistic"]["glm_base"],
    )
    return (MODEL2_FIXED_PROBABILITY_COMPONENT,)


@app.cell
def _(find_feature_transformer_name, pd, read_hyperparameters):
    def build_trials_table(
        study, objective, metric_names, components_by_trial, feature_transformer_dict
    ):
        """Every trial of `study` as a row of the whole model it scored: its id
        (its number), `objective`, seconds, state, cross-validated score on
        each metric (in `metric_names`' order), and each component's feature
        transformer and hyperparameters (the classifier's with its
        calibration). `components_by_trial` holds the components of each
        trial's model, in the study's order; `feature_transformer_dict` maps a
        component's name to its family's candidates, in which its
        transformer's name is found by identity."""
        trial_records = []
        for trial, components in zip(study.trials, components_by_trial, strict=True):
            feature_transformers = {
                component.name: find_feature_transformer_name(
                    component, feature_transformer_dict[component.name]
                )
                for component in components
            }
            hyperparameters = {
                component.name: read_hyperparameters(component)
                for component in components
            }
            trial_records.append(
                {
                    "id": trial.number,
                    "objective": objective,
                    "seconds": trial.duration.total_seconds(),
                    "state": trial.state.name,
                    # A failed trial records no score: NaN.
                    **{
                        metric_name: trial.user_attrs.get(metric_name)
                        for metric_name in metric_names
                    },
                    "feature_transformers": feature_transformers,
                    "hyperparameters": hyperparameters,
                }
            )
        return pd.DataFrame(trial_records)

    return (build_trials_table,)


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
    build_study_name,
    build_trials_table,
    compute_metrics,
    cv_groups,
    dataclasses,
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
    # design but its metric. A failed trial is caught by the study. The run
    # settings of the tuning notebook's first run: its study on the tuner's
    # objective is that run's.
    _run_settings = {
        "family": "lgbm_poisson",
        "use_exposure": True,
        "share_hyperparams": True,
        "share_feature_transformer": False,
    }
    _family, _use_exposure = _run_settings["family"], _run_settings["use_exposure"]
    _metric_names = [metric.name for metric in METRICS]
    # Every cohort's component draws from the family's candidates.
    _feature_transformer_dict = {
        cohort: COUNT_FEATURE_TRANSFORMERS[_family] for cohort in COHORTS
    }
    _scores_by_objective, _trials = {}, []
    for _objective in EDA_OBJECTIVES:
        _objective_tuner = dataclasses.replace(tuner, objective_metric=_objective)
        _study, _seconds = _objective_tuner.tune_independent_cohorts_model(
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
            study_seed=seed,
            study_name=build_study_name(
                "independent_cohorts", _run_settings, _objective.name, seed
            ),
        )
        # Each trial's model: the cohorts' components its parameters build.
        _components_by_trial = [
            tuple(
                CountComponent.from_params(
                    trial.params,
                    name=cohort,
                    family=_family,
                    use_exposure=_use_exposure,
                    feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_family],
                )
                for cohort in COHORTS
            )
            for trial in _study.trials
        ]
        _trials.append(
            build_trials_table(
                _study,
                _objective.name,
                _metric_names,
                _components_by_trial,
                _feature_transformer_dict,
            )
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
        _scores_by_objective[_objective.name] = compute_metrics(
            val_age_group.Y,
            _model.predict(val_age_group.X, exposure=val_age_group.exposure),
            METRICS,
            cohorts=[*COHORTS, "composite"],
        ).assign(cv_best=_study.best_value, seconds=_seconds)
    eda_model1_df = pd.concat(_scores_by_objective, names=["objective"]).reset_index()
    # Every trial of the three studies, as a row of the whole model it scored.
    eda_model1_trials_df = pd.concat(_trials, ignore_index=True)
    return eda_model1_df, eda_model1_trials_df


@app.cell
def _(
    COHORTS,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    EDA_OBJECTIVES,
    METRICS,
    MODEL2_FIXED_PROBABILITY_COMPONENT,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_study_name,
    build_total_times_probability_model,
    build_trials_table,
    compute_metrics,
    cv_groups,
    dataclasses,
    mo,
    pd,
    run_eda,
    seed,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(not run_eda.value, mo.md("Click the button to run the objective EDA."))
    # The total times probability model's EDA run, by coordinate descent: per
    # objective, step 1 tunes the total with the fixed classifier, step 2 the
    # probabilities with step 1's best total; every study seeded seed (M8).
    # Step 2's value is the pair's cv: its fixed total is the pair's.
    _total_family, _probability_family, _use_exposure = "lgbm_poisson", "logistic", True
    _metric_names = [metric.name for metric in METRICS]
    _feature_transformer_dict = {
        "total": COUNT_FEATURE_TRANSFORMERS[_total_family],
        "probability": PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    }
    _scores_by_objective, _trials = {}, []
    for _objective in EDA_OBJECTIVES:
        _objective_tuner = dataclasses.replace(tuner, objective_metric=_objective)
        _total_study, _total_seconds = _objective_tuner.tune_total_component(
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
            study_seed=seed,
            study_name=build_study_name(
                "total_times_probability_coordinate_descent_total",
                {"total_family": _total_family, "use_exposure": _use_exposure},
                _objective.name,
                seed,
            ),
        )
        # Step 1's trials: each drawn total, with the fixed classifier.
        _total_components_by_trial = [
            (
                CountComponent.from_params(
                    trial.params,
                    name="total",
                    family=_total_family,
                    use_exposure=_use_exposure,
                    feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_total_family],
                ),
                MODEL2_FIXED_PROBABILITY_COMPONENT,
            )
            for trial in _total_study.trials
        ]
        _trials.append(
            build_trials_table(
                _total_study,
                _objective.name,
                _metric_names,
                _total_components_by_trial,
                _feature_transformer_dict,
            ).assign(step="total")
        )
        _total_component = CountComponent.from_params(
            _total_study.best_params,
            name="total",
            family=_total_family,
            use_exposure=_use_exposure,
            feature_transformer_dict=COUNT_FEATURE_TRANSFORMERS[_total_family],
        )
        _probability_study, _probability_seconds = (
            _objective_tuner.tune_probability_component(
                train_age_group.X,
                train_age_group.Y,
                train_age_group.exposure,
                cv_groups,
                total_component=_total_component,
                probability_family=_probability_family,
                probability_feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                    _probability_family
                ],
                y_min=PREDICTION_FLOOR,
                random_state=seed,
                study_seed=seed,
                study_name=build_study_name(
                    "total_times_probability_coordinate_descent_probability",
                    {"probability_family": _probability_family},
                    _objective.name,
                    seed,
                ),
            )
        )
        # Step 2's trials: step 1's best total, with each drawn classifier.
        _probability_components_by_trial = [
            (
                _total_component,
                ProbabilityComponent.from_params(
                    trial.params,
                    name="probability",
                    family=_probability_family,
                    feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                        _probability_family
                    ],
                ),
            )
            for trial in _probability_study.trials
        ]
        _trials.append(
            build_trials_table(
                _probability_study,
                _objective.name,
                _metric_names,
                _probability_components_by_trial,
                _feature_transformer_dict,
            ).assign(step="probability")
        )
        _probability_component = ProbabilityComponent.from_params(
            _probability_study.best_params,
            name="probability",
            family=_probability_family,
            feature_transformer_dict=PROBABILITY_FEATURE_TRANSFORMERS[
                _probability_family
            ],
        )
        _model = build_total_times_probability_model(
            _total_component,
            _probability_component,
            y_min=PREDICTION_FLOOR,
            random_state=seed,
        ).fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
        # Every metric on validation, per cohort and for the composite; the
        # pair's value (in its own objective) and the two studies' seconds on
        # each row.
        _scores_by_objective[_objective.name] = compute_metrics(
            val_age_group.Y,
            _model.predict(val_age_group.X, exposure=val_age_group.exposure),
            METRICS,
            cohorts=[*COHORTS, "composite"],
        ).assign(
            cv_best=_probability_study.best_value,
            seconds=_total_seconds + _probability_seconds,
        )
    eda_model2_df = pd.concat(_scores_by_objective, names=["objective"]).reset_index()
    # Every trial of the six studies (each objective's two steps), as a row of
    # the whole model it scored: its step's drawn component and fixed one.
    eda_model2_trials_df = pd.concat(_trials, ignore_index=True)
    return eda_model2_df, eda_model2_trials_df


@app.cell
def _(
    COHORTS,
    COUNT_FEATURE_TRANSFORMERS,
    CountComponent,
    EDA_OBJECTIVES,
    METRICS,
    PREDICTION_FLOOR,
    PROBABILITY_FEATURE_TRANSFORMERS,
    ProbabilityComponent,
    build_study_name,
    build_total_times_probability_model,
    build_trials_table,
    compute_metrics,
    cv_groups,
    dataclasses,
    mo,
    pd,
    run_eda,
    seed,
    train_age_group,
    tuner,
    val_age_group,
):
    mo.stop(not run_eda.value, mo.md("Click the button to run the objective EDA."))
    # The total times probability model's EDA run by the joint method, to compare
    # with coordinate descent's above on the same setting: per objective, one
    # study whose every trial draws both components, seeded seed (M8). The run
    # settings of the tuning notebook's first run: its study on the tuner's
    # objective is that run's.
    _run_settings = {
        "total_family": "lgbm_poisson",
        "probability_family": "logistic",
        "use_exposure": True,
    }
    _total_family, _probability_family, _use_exposure = (
        _run_settings["total_family"],
        _run_settings["probability_family"],
        _run_settings["use_exposure"],
    )
    _metric_names = [metric.name for metric in METRICS]
    _feature_transformer_dict = {
        "total": COUNT_FEATURE_TRANSFORMERS[_total_family],
        "probability": PROBABILITY_FEATURE_TRANSFORMERS[_probability_family],
    }
    _scores_by_objective, _trials = {}, []
    for _objective in EDA_OBJECTIVES:
        _objective_tuner = dataclasses.replace(tuner, objective_metric=_objective)
        _study, _seconds = _objective_tuner.tune_total_times_probability_model(
            train_age_group.X,
            train_age_group.Y,
            train_age_group.exposure,
            cv_groups,
            total_family=_total_family,
            probability_family=_probability_family,
            use_exposure=_use_exposure,
            total_feature_transformer_dict=_feature_transformer_dict["total"],
            probability_feature_transformer_dict=_feature_transformer_dict[
                "probability"
            ],
            y_min=PREDICTION_FLOOR,
            random_state=seed,
            study_seed=seed,
            study_name=build_study_name(
                "total_times_probability_joint", _run_settings, _objective.name, seed
            ),
        )
        # Each trial's model: the two components its parameters build.
        _components_by_trial = [
            (
                CountComponent.from_params(
                    trial.params,
                    name="total",
                    family=_total_family,
                    use_exposure=_use_exposure,
                    feature_transformer_dict=_feature_transformer_dict["total"],
                ),
                ProbabilityComponent.from_params(
                    trial.params,
                    name="probability",
                    family=_probability_family,
                    feature_transformer_dict=_feature_transformer_dict["probability"],
                ),
            )
            for trial in _study.trials
        ]
        _trials.append(
            build_trials_table(
                _study,
                _objective.name,
                _metric_names,
                _components_by_trial,
                _feature_transformer_dict,
            )
        )
        # The best trial's model, refit on the training part.
        _model = build_total_times_probability_model(
            *_components_by_trial[_study.best_trial.number],
            y_min=PREDICTION_FLOOR,
            random_state=seed,
        ).fit(train_age_group.X, train_age_group.Y, exposure=train_age_group.exposure)
        # Every metric on validation, per cohort and for the composite; the
        # study's value (in its own objective) and its seconds on each row.
        _scores_by_objective[_objective.name] = compute_metrics(
            val_age_group.Y,
            _model.predict(val_age_group.X, exposure=val_age_group.exposure),
            METRICS,
            cohorts=[*COHORTS, "composite"],
        ).assign(cv_best=_study.best_value, seconds=_seconds)
    eda_model2_joint_df = pd.concat(
        _scores_by_objective, names=["objective"]
    ).reset_index()
    # Every trial of the three studies, as a row of the whole model it scored.
    eda_model2_joint_trials_df = pd.concat(_trials, ignore_index=True)
    return eda_model2_joint_df, eda_model2_joint_trials_df


@app.cell
def _(
    METRICS,
    eda_model1_df,
    eda_model1_trials_df,
    eda_model2_df,
    eda_model2_joint_df,
    eda_model2_joint_trials_df,
    eda_model2_trials_df,
    mo,
    pd,
):
    # Each model's best trial per objective: its study's minimum on the
    # objective (the first on ties, as Optuna's best trial; the total times
    # probability model's by coordinate descent in its probability step, the
    # pair the refit holds; by the joint method in its one study), its
    # cross-validated score on every metric, and the refit's composite on
    # validation.
    _metric_names = [metric.name for metric in METRICS]
    _rows = []
    for _model, _trials, _scores in (
        ("independent_cohorts", eda_model1_trials_df, eda_model1_df),
        (
            "total_times_probability_coordinate_descent",
            eda_model2_trials_df[eda_model2_trials_df["step"] == "probability"],
            eda_model2_df,
        ),
        ("total_times_probability", eda_model2_joint_trials_df, eda_model2_joint_df),
    ):
        for _objective, _study_trials in _trials.groupby("objective", sort=False):
            _best = _study_trials.loc[_study_trials[_objective].idxmin()]
            _composite = _scores[
                (_scores["objective"] == _objective)
                & (_scores["cohort"] == "composite")
            ].iloc[0]
            _rows.append(
                {
                    "model": _model,
                    "objective": _objective,
                    "best_id": _best["id"],
                    **{
                        f"cv_{metric_name}": _best[metric_name]
                        for metric_name in _metric_names
                    },
                    **{
                        f"val_{metric_name}": _composite[metric_name]
                        for metric_name in _metric_names
                    },
                }
            )
    eda_best_df = pd.DataFrame(_rows)
    mo.vstack(
        [
            mo.md(
                "**The objective EDA: the best model per objective** (each study's "
                "best trial, the total times probability model's by coordinate "
                "descent (total_times_probability_coordinate_descent) in its "
                "probability step, by the joint method (total_times_probability) "
                "in its one study; cv_: its cross-validated scores; val_: the "
                "refit's composite on validation)"
            ),
            mo.ui.table(eda_best_df.round(4)),
        ]
    )
    return (eda_best_df,)


@app.cell
def _(eda_best_df):
    eda_best_df.sort_values(by="objective", ascending=False)
    return


@app.cell
def _(EDA_OBJECTIVES, mo, tuner):
    # The objective whose studies the EDA's figures show; the tuner's by default.
    eda_objective = mo.ui.dropdown(
        options=[objective.name for objective in EDA_OBJECTIVES],
        value=tuner.objective_metric.name,
        label="Objective",
    )
    return (eda_objective,)


@app.cell
def _(
    METRICS,
    eda_best_df,
    eda_model1_trials_df,
    eda_model2_joint_trials_df,
    eda_model2_trials_df,
    eda_objective,
    go,
    make_subplots,
    mo,
):
    # Per model and objective, every metric per trial, a panel each on its own
    # scale, the trials in order on a shared axis (the total times probability
    # model by coordinate descent: its total step, then its probability step;
    # by the joint method, like the independent cohorts model, one study): the
    # best trial a red star in every panel. The colors are the dataviz
    # reference palette's (the trials series 1, the best series 8, the rule
    # secondary ink, the chart surface); the trials translucent, so
    # overlapping ones show darker.
    _metric_names = [metric.name for metric in METRICS]
    _surface, _ink = "#fcfcfb", "#52514e"
    eda_figures = {}
    for _model, _trials in (
        ("independent_cohorts", eda_model1_trials_df),
        ("total_times_probability_coordinate_descent", eda_model2_trials_df),
        ("total_times_probability", eda_model2_joint_trials_df),
    ):
        for _objective, _study_trials in _trials.groupby("objective", sort=False):
            _study_trials = _study_trials.reset_index(drop=True)
            _best_id = eda_best_df.loc[
                (eda_best_df["model"] == _model)
                & (eda_best_df["objective"] == _objective),
                "best_id",
            ].iloc[0]
            if _model in ("independent_cohorts", "total_times_probability"):
                _labels = "trial " + _study_trials["id"].astype(str)
                _is_best = _study_trials["id"] == _best_id
                _probability_start = None
            elif _model == "total_times_probability_coordinate_descent":
                _labels = (
                    _study_trials["step"] + " trial " + _study_trials["id"].astype(str)
                )
                _is_best = (_study_trials["step"] == "probability") & (
                    _study_trials["id"] == _best_id
                )
                _probability_start = (_study_trials["step"] == "total").sum()
            else:
                raise ValueError(f"unknown model {_model!r}")
            _best_position = _study_trials.index[_is_best][0]
            _figure = make_subplots(
                rows=len(_metric_names),
                cols=1,
                shared_xaxes=True,
                subplot_titles=_metric_names,
                vertical_spacing=0.04,
            )
            for _row, _metric_name in enumerate(_metric_names, start=1):
                _figure.add_trace(
                    go.Scatter(
                        x=_study_trials.index,
                        y=_study_trials[_metric_name],
                        mode="markers",
                        marker={"size": 5, "color": "#2a78d6", "opacity": 0.6},
                        text=_labels,
                        hovertemplate=f"%{{text}}<br>{_metric_name} %{{y:.4f}}<extra></extra>",
                        name="trials",
                        legendgroup="trials",
                        showlegend=_row == 1,
                    ),
                    row=_row,
                    col=1,
                )
                _figure.add_trace(
                    go.Scatter(
                        x=[_best_position],
                        y=[_study_trials.loc[_best_position, _metric_name]],
                        mode="markers",
                        marker={
                            "symbol": "star",
                            "size": 12,
                            "color": "#e34948",
                            "line": {"width": 2, "color": _surface},
                        },
                        text=[_labels[_best_position]],
                        hovertemplate=f"best: %{{text}}<br>{_metric_name} %{{y:.4f}}<extra></extra>",
                        name=f"best: {_labels[_best_position]}",
                        legendgroup="best",
                        showlegend=_row == 1,
                    ),
                    row=_row,
                    col=1,
                )
            if _probability_start is not None:
                # The rule in every panel; its label once, in the top panel.
                _figure.add_vline(
                    x=_probability_start - 0.5, line_width=1, line_color=_ink
                )
                _figure.add_annotation(
                    text="probability step",
                    x=_probability_start - 0.5,
                    xref="x",
                    y=1,
                    yref="y domain",
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                )
            _figure.update_layout(
                title=f"tuned on {_objective}",
                height=1000,
                template="plotly_white",
                paper_bgcolor=_surface,
                plot_bgcolor=_surface,
                legend={"orientation": "h", "y": -0.06},
                hovermode="closest",
            )
            _figure.update_xaxes(title_text="trial", row=len(_metric_names), col=1)
            eda_figures[_model, _objective] = _figure
    mo.vstack(
        [
            eda_objective,
            mo.md(
                "**The independent cohorts model: every metric per trial** (the "
                "study of the chosen objective; the star its best trial)"
            ),
            eda_figures["independent_cohorts", eda_objective.value],
            mo.md(
                "**The total times probability model by coordinate descent: every "
                "metric per trial** (the total step's trials, then the probability "
                "step's; the star its best trial, in the probability step)"
            ),
            eda_figures[
                "total_times_probability_coordinate_descent", eda_objective.value
            ],
            mo.md(
                "**The total times probability model by the joint method: every "
                "metric per trial** (one study drawing both components; the star its "
                "best trial)"
            ),
            eda_figures["total_times_probability", eda_objective.value],
        ]
    )
    return


@app.cell
def _(METRICS, mo, tuner):
    # The recorded metric on which the EDA's objectives are compared; the
    # tuner's objective by default.
    eda_recorded_metric = mo.ui.dropdown(
        options=[metric.name for metric in METRICS],
        value=tuner.objective_metric.name,
        label="Recorded metric",
    )
    return (eda_recorded_metric,)


@app.cell
def _(
    METRICS,
    eda_best_df,
    eda_model1_trials_df,
    eda_model2_joint_trials_df,
    eda_model2_trials_df,
    eda_recorded_metric,
    go,
    mo,
):
    # Per model and recorded metric, the objectives' studies compared: a box per
    # objective with every trial a point and its best trial a star (the model it
    # picks); the total times probability model's two steps side by side by
    # coordinate descent, its one study by the joint method. The colors are the
    # dataviz reference palette's first three categorical slots, each objective
    # with its own marker; the y-axis spans the bulk of the trials (their 2nd to
    # 98th percentile) and every star.
    _styles = {
        "log_loss": ("#2a78d6", "circle"),
        "poisson_deviance": ("#eb6834", "diamond"),
        "rmse": ("#1baf7a", "triangle-up"),
    }
    _surface, _ink = "#fcfcfb", "#52514e"
    eda_metric_figures = {}
    for _model, _trials in (
        ("independent_cohorts", eda_model1_trials_df),
        ("total_times_probability_coordinate_descent", eda_model2_trials_df),
        ("total_times_probability", eda_model2_joint_trials_df),
    ):
        if _model in ("independent_cohorts", "total_times_probability"):
            _x_column, _boxmode = "objective", "overlay"
            _labels = "trial " + _trials["id"].astype(str)
            _best_candidates = _trials
        elif _model == "total_times_probability_coordinate_descent":
            _x_column, _boxmode = "step", "group"
            _labels = _trials["step"] + " trial " + _trials["id"].astype(str)
            _best_candidates = _trials[_trials["step"] == "probability"]
        else:
            raise ValueError(f"unknown model {_model!r}")
        for _metric in METRICS:
            _figure = go.Figure()
            _star_values = []
            for _objective in _trials["objective"].unique():
                _color, _symbol = _styles[_objective]
                _is_study = _trials["objective"] == _objective
                _best_id = eda_best_df.loc[
                    (eda_best_df["model"] == _model)
                    & (eda_best_df["objective"] == _objective),
                    "best_id",
                ].iloc[0]
                _best_index = _best_candidates.index[
                    (_best_candidates["objective"] == _objective)
                    & (_best_candidates["id"] == _best_id)
                ][0]
                _star_values.append(_trials.loc[_best_index, _metric.name])
                _figure.add_trace(
                    go.Box(
                        x=_trials.loc[_is_study, _x_column],
                        y=_trials.loc[_is_study, _metric.name],
                        name=_objective,
                        boxpoints="all",
                        jitter=0.5,
                        pointpos=0,
                        fillcolor="rgba(0, 0, 0, 0)",
                        line={"color": _color, "width": 2},
                        marker={
                            "color": _color,
                            "symbol": _symbol,
                            "size": 5,
                            "opacity": 0.5,
                        },
                        text=_labels[_is_study],
                        hovertemplate=f"{_objective}: %{{text}}<br>{_metric.name} %{{y:.4f}}<extra></extra>",
                        offsetgroup=_objective,
                        legendgroup=_objective,
                    )
                )
                _figure.add_trace(
                    go.Scatter(
                        x=[_trials.loc[_best_index, _x_column]],
                        y=[_star_values[-1]],
                        mode="markers",
                        marker={
                            "symbol": "star",
                            "size": 14,
                            "color": _color,
                            "line": {"width": 1.5, "color": _ink},
                        },
                        text=[_labels[_best_index]],
                        hovertemplate=f"best of {_objective}: %{{text}}<br>{_metric.name} %{{y:.4f}}<extra></extra>",
                        offsetgroup=_objective,
                        legendgroup=_objective,
                        showlegend=False,
                    )
                )
            _low, _high = _trials[_metric.name].quantile([0.02, 0.98])
            _low, _high = min(_low, *_star_values), max(_high, *_star_values)
            _padding = 0.05 * (_high - _low)
            _figure.update_layout(
                title=f"{_metric.name} by objective",
                height=500,
                template="plotly_white",
                paper_bgcolor=_surface,
                plot_bgcolor=_surface,
                boxmode=_boxmode,
                scattermode="group",
                legend={"orientation": "h", "y": -0.15},
                hovermode="closest",
            )
            _figure.update_yaxes(
                title_text=_metric.name, range=[_low - _padding, _high + _padding]
            )
            eda_metric_figures[_model, _metric.name] = _figure
    mo.vstack(
        [
            eda_recorded_metric,
            mo.md(
                "**The independent cohorts model: the objectives compared on the "
                "chosen metric** (a box per objective's study with every trial; the "
                "star its best trial)"
            ),
            eda_metric_figures["independent_cohorts", eda_recorded_metric.value],
            mo.md(
                "**The total times probability model by coordinate descent: the "
                "objectives compared on the chosen metric** (each step's trials; the "
                "star its best trial, in the probability step)"
            ),
            eda_metric_figures[
                "total_times_probability_coordinate_descent", eda_recorded_metric.value
            ],
            mo.md(
                "**The total times probability model by the joint method: the "
                "objectives compared on the chosen metric** (a box per objective's "
                "study with every trial; the star its best trial)"
            ),
            eda_metric_figures["total_times_probability", eda_recorded_metric.value],
        ]
    )
    return


@app.cell
def _(
    eda_model1_df,
    eda_model1_trials_df,
    eda_model2_df,
    eda_model2_joint_df,
    eda_model2_joint_trials_df,
    eda_model2_trials_df,
    mo,
):
    mo.vstack(
        [
            mo.md(
                "**The objective EDA: the independent cohorts model** (each "
                "objective's best model refit; every metric on validation, per "
                "cohort and for the composite; cv_best: the study's value in its "
                "own objective; seconds: the study's)"
            ),
            mo.ui.table(eda_model1_df.round(4)),
            mo.md(
                "**The total times probability model by coordinate descent** (the "
                "same; cv_best: step 2's value, the pair's, in its own objective; "
                "seconds: the two studies')"
            ),
            mo.ui.table(eda_model2_df.round(4)),
            mo.md(
                "**The total times probability model by the joint method** (the "
                "same; cv_best: the study's value in its own objective; seconds: "
                "the study's)"
            ),
            mo.ui.table(eda_model2_joint_df.round(4)),
            mo.md(
                "**Every trial: the independent cohorts model** (one row per trial "
                "of each objective's study, the whole model it scored: its state, "
                "seconds, cross-validated score on every metric, and each cohort's "
                "feature transformer and hyperparameters)"
            ),
            mo.ui.table(eda_model1_trials_df.round(4)),
            mo.md(
                "**Every trial: the total times probability model by coordinate "
                "descent** (each objective's two studies, each trial by its step and "
                "number: step `total`, each drawn total with the fixed classifier; "
                "step `probability`, step 1's best total with each drawn classifier)"
            ),
            mo.ui.table(eda_model2_trials_df.round(4)),
            mo.md(
                "**Every trial: the total times probability model by the joint "
                "method** (one row per trial of each objective's study, its total and "
                "probabilities drawn together)"
            ),
            mo.ui.table(eda_model2_joint_trials_df.round(4)),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
