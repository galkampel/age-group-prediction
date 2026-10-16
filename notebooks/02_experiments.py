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

    from sklearn.base import clone

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
def _(ExposureTransformer, df):
    targets = ["n_kindergarten", "n_elementary", "n_highschool"]
    exposure_col = "n_apartments"
    groups_col = "neighborhood_id"
    id_col = "building_id"
    X, Y, groups = df.drop(columns=[id_col, groups_col, *targets]), df[targets], df[groups_col]
    # Checks every exposure is finite and > 0; the models leave that to it.
    exposure = ExposureTransformer(exposure_col).fit_transform(df)

    print(X.shape, Y.shape, groups.shape, exposure.shape)
    return X, Y, exposure, groups, id_col


@app.cell
def _():
    return


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
        f"data rows={len(df)} neighborhoods={groups.nunique()} config_hash={config_hash}"
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
    n_trials = 200
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
    return n_splits, n_trials, seed, split_method, test_size, val_size


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
    return (
        X_train,
        X_val,
        Y_train,
        Y_val,
        exposure_train,
        exposure_val,
        split_groups,
        splitter,
        train_index,
    )


@app.cell
def _(n_splits, seed, split_groups, splitter, train_index):
    cv = splitter.cv(n_splits=n_splits, random_state=seed)
    # What cv.split takes: the training rows' groups, or None under random,
    # since KFold warns when given groups (an error under -W error).
    cv_groups = None if split_groups is None else split_groups.iloc[train_index]
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
    # §8.5.4: daycare as an interaction operand. Centered, each product barely
    # correlates with its room share (-0.04 to 0.06 on the training part; 0.85
    # to 0.86 with the uncentered d/2).
    center_scale_by_two_plan = Plan(
        name="center_scale_by_2", transforms=(Center(), DomainScale(scale=2.0))
    )
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
        center_scale_by_two_plan,
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
    center_scale_by_two_plan,
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
    # §6.3: the second choice; §6.5: the composition-stage pair. Daycare is
    # centered as an operand (§8.5.4).
    probability__room_share_x_daycare = FeatureTransformer(
        column_plans=(
            (standardize_plan, ["ses", "avg_household_size", "median_age"]),
            (center_scale_by_tenth_plan, list(ROOM_SHARES)),
            (center_scale_by_two_plan, ["n_daycares_500m"]),
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
            (center_scale_by_two_plan, ["n_daycares_500m"]),
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

    FEATURE_VARIANTS = {
        "tree": tree,
        "tree__ses_quadratic": tree__ses_quadratic,
        "glm_base": glm_base,
        "count__ses_quadratic": count__ses_quadratic,
        "count__daycare_log1p": count__daycare_log1p,
        "count__room_share_x_household_size": count__room_share_x_household_size,
        "count__room_share_x_ses": count__room_share_x_ses,
        "count__room_share_x_household_size_and_ses": (
            count__room_share_x_household_size_and_ses
        ),
        "probability__daycare_log1p": probability__daycare_log1p,
        "probability__room_share_x_median_age": probability__room_share_x_median_age,
        "probability__room_share_x_daycare": probability__room_share_x_daycare,
        "probability__room_share_x_median_age_and_daycare": (
            probability__room_share_x_median_age_and_daycare
        ),
    }
    return (FEATURE_VARIANTS,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Hyperparameter tuning

    One Optuna study per model family, on the composite: the model that is
    scored is the model that is tuned, by the cohort Poisson deviance of
    its cross-validation folds on the training part. A trial chooses, per
    cohort, the exposure on or off, the feature variant, the LightGBM loss
    (Poisson or Gaussian) and the estimator's hyperparameters, under the
    names `cohort_models__<cohort>__<parameter>`. The feature variant is a
    `KeyedParameter`: Optuna searches its names, so `best_params` stays
    plain JSON, and the evaluator resolves a name to its transformer.
    `n_trials = 200`: at 30 a categorical arm starves (a Poisson/Gaussian
    study split 10/20 and did worse than either loss alone); the table of
    trials per arm shows it. The i-th study gets `seed + i`, so the studies
    do not share their startup trials; they run in parallel with joblib.
    Only each family's best combination is refit on the training part and
    scored on training and validation. NB2 tunes its dispersion and its L1
    penalty; a Gaussian loss can predict a negative count, which
    `CountModel` floors, so the trial scores badly instead of stopping the
    study. Model 2 is tuned the same way, one study per pair of a total
    family (the list `MODEL2_TOTALS`, edited after Model 1's table) and a
    classifier family, under `total_model__…` and `probability_model__…`;
    the probability model ignores the exposure, and its calibration is
    searched. The last table shows each model's best family on validation.
    The run cells sit behind buttons: opening the notebook fits nothing.
    """)
    return


@app.cell
def _():
    import dataclasses
    import time
    from collections import Counter

    import optuna
    from joblib import Parallel, delayed
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.linear_model import LogisticRegression, PoissonRegressor

    from age_group_prediction.hyperparameter_tuning import (
        CategoricalParameter,
        CVHyperparameterEvaluator,
        FloatParameter,
        HyperparameterStudy,
        IntParameter,
        KeyedParameter,
    )
    from age_group_prediction.modeling import (
        CohortProbabilityModel,
        CountModel,
        IndependentCohortModels,
        NegativeBinomialRegressor,
        TotalTimesProbabilityModel,
    )
    from age_group_prediction.scoring import (
        COHORT_LOG_LOSS,
        COHORT_POISSON_DEVIANCE,
        POISSON_DEVIANCE,
    )

    # Optuna logs every trial at INFO; the tuning function repeats this in
    # each joblib worker.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        COHORT_LOG_LOSS,
        COHORT_POISSON_DEVIANCE,
        CVHyperparameterEvaluator,
        CategoricalParameter,
        CohortProbabilityModel,
        CountModel,
        Counter,
        FloatParameter,
        HyperparameterStudy,
        IndependentCohortModels,
        IntParameter,
        KeyedParameter,
        LGBMClassifier,
        LGBMRegressor,
        LogisticRegression,
        NegativeBinomialRegressor,
        POISSON_DEVIANCE,
        Parallel,
        PoissonRegressor,
        TotalTimesProbabilityModel,
        dataclasses,
        delayed,
        optuna,
        time,
    )


@app.cell
def _(CategoricalParameter, FloatParameter, IntParameter):
    # The names are set_params keys: estimator__ reaches the model's estimator.
    # No isotonic: its lowest score block can hold no child of a cohort and
    # then maps a building there to probability 0, an infinite log loss
    # (measured 2026-10-15 with the LightGBM classifier).
    CALIBRATION = CategoricalParameter(
        "calibration_method", [None, "temperature", "sigmoid"]
    )
    LGBM_SPACE = [
        IntParameter("estimator__max_depth", 3, 8),
        IntParameter("estimator__num_leaves", 7, 63),
        IntParameter("estimator__min_child_samples", 5, 40),
        FloatParameter("estimator__learning_rate", 0.01, 0.2, log=True),
        IntParameter("estimator__n_estimators", 50, 400),
        FloatParameter("estimator__reg_alpha", 1e-8, 10.0, log=True),
        FloatParameter("estimator__reg_lambda", 1e-8, 10.0, log=True),
        FloatParameter("estimator__min_split_gain", 0.0, 1.0),
        FloatParameter("estimator__subsample", 0.7, 1.0),
        FloatParameter("estimator__colsample_bytree", 0.7, 1.0),
    ]
    SEARCH_SPACES = {
        "lgbm": LGBM_SPACE,
        "poisson_glm": [FloatParameter("estimator__alpha", 1e-4, 10.0, log=True)],
        # The estimated dispersions are 0.12 to 0.18; the penalty's path runs
        # from the unpenalized fit at 1e-3 to all-zero coefficients at 1.
        "nb2": [
            FloatParameter("estimator__dispersion", 0.01, 1.0, log=True),
            FloatParameter("estimator__alpha", 1e-3, 1.0, log=True),
        ],
        "logistic": [
            FloatParameter("estimator__C", 1e-3, 100.0, log=True),
            CALIBRATION,
        ],
        "lgbm_classifier": [*LGBM_SPACE, CALIBRATION],
    }
    return (SEARCH_SPACES,)


@app.cell
def _(
    CategoricalParameter,
    FEATURE_VARIANTS,
    KeyedParameter,
    LGBMClassifier,
    LGBMRegressor,
    LogisticRegression,
    NegativeBinomialRegressor,
    PoissonRegressor,
    SEARCH_SPACES,
    Y,
    seed,
):
    # The trees take n_apartments as a feature and the offset; the GLMs take it
    # as the exposure only (docs/FEATURE_TRANSFORMATIONS.md §3.3, §4.4), and the
    # logistic model needs the scaled plans (it does not converge on `tree`).
    # subsample_freq=1: LightGBM ignores subsample without it; n_jobs=1: one
    # thread per fit.
    COHORTS = list(Y.columns)
    TREE_VARIANTS = ("tree", "tree__ses_quadratic")
    COUNT_GLM_VARIANTS = (
        "glm_base",
        "count__ses_quadratic",
        "count__daycare_log1p",
        "count__room_share_x_household_size",
        "count__room_share_x_ses",
        "count__room_share_x_household_size_and_ses",
    )
    PROBABILITY_GLM_VARIANTS = (
        "glm_base",
        "probability__daycare_log1p",
        "probability__room_share_x_median_age",
        "probability__room_share_x_daycare",
        "probability__room_share_x_median_age_and_daycare",
    )
    # What a trial chooses for one count component besides the estimator's
    # hyperparameters. The variant is searched by name and resolved to its
    # transformer by the evaluator.
    USE_EXPOSURE = CategoricalParameter("use_exposure", [True, False])
    LGBM_OBJECTIVE = CategoricalParameter(
        "estimator__objective", ["poisson", "regression"]
    )
    tree_variant = KeyedParameter(
        "feature_transformer", {name: FEATURE_VARIANTS[name] for name in TREE_VARIANTS}
    )
    count_variant = KeyedParameter(
        "feature_transformer",
        {name: FEATURE_VARIANTS[name] for name in COUNT_GLM_VARIANTS},
    )
    # Each count family: (estimator, the parameters of one count component).
    COUNT_FAMILIES = {
        "lgbm": (
            LGBMRegressor(
                objective="poisson",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
            ),
            [*SEARCH_SPACES["lgbm"], LGBM_OBJECTIVE, USE_EXPOSURE, tree_variant],
        ),
        "poisson_glm": (
            PoissonRegressor(max_iter=1000),
            [*SEARCH_SPACES["poisson_glm"], USE_EXPOSURE, count_variant],
        ),
        "nb2": (
            NegativeBinomialRegressor(),
            [*SEARCH_SPACES["nb2"], USE_EXPOSURE, count_variant],
        ),
    }
    # Each probability family likewise; the probability model ignores the
    # exposure, and its calibration is in the search space.
    probability_variant = KeyedParameter(
        "feature_transformer",
        {name: FEATURE_VARIANTS[name] for name in PROBABILITY_GLM_VARIANTS},
    )
    PROBABILITY_FAMILIES = {
        "logistic": (
            LogisticRegression(max_iter=1000),
            [*SEARCH_SPACES["logistic"], probability_variant],
        ),
        "lgbm": (
            LGBMClassifier(
                objective="multiclass",
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=seed,
            ),
            [*SEARCH_SPACES["lgbm_classifier"], tree_variant],
        ),
    }
    return COHORTS, COUNT_FAMILIES, PROBABILITY_FAMILIES


@app.cell
def _(
    COHORTS,
    COHORT_POISSON_DEVIANCE,
    COUNT_FAMILIES,
    CVHyperparameterEvaluator,
    CountModel,
    IndependentCohortModels,
    cv,
    dataclasses,
):
    def prefixed(prefix, parameters):
        """The same parameters under ``<prefix>__<name>``, the set_params path to a component."""
        return [dataclasses.replace(p, name=f"{prefix}__{p.name}") for p in parameters]

    # Model 1, one evaluator per family: a template with one CountModel per
    # cohort and the component's parameters repeated under each cohort's prefix.
    # Every trial sets each cohort's exposure and variant, so the template's
    # defaults never fit. The evaluator also rebuilds the best model (build_model).
    MODEL1_EVALUATORS = {
        family: CVHyperparameterEvaluator(
            IndependentCohortModels(
                {cohort: CountModel(estimator=estimator) for cohort in COHORTS}
            ),
            [
                parameter
                for cohort in COHORTS
                for parameter in prefixed(f"cohort_models__{cohort}", parameters)
            ],
            cv=cv,
            metric=COHORT_POISSON_DEVIANCE,
        )
        for family, (estimator, parameters) in COUNT_FAMILIES.items()
    }
    return MODEL1_EVALUATORS, prefixed


@app.cell
def _(HyperparameterStudy, optuna, time):
    def tune_model(evaluator, X, y, *, exposure, cv_groups, study_seed, n_trials):
        """One study of ``evaluator``'s model on the rows of ``X``; the result and its seconds.

        An empty search space scores the model as it is, in one trial: more
        trials would repeat the same fit. Runs in a joblib worker, a fresh
        process at Optuna's default INFO verbosity, so it is set here too.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = HyperparameterStudy(
            seed=study_seed, n_trials=n_trials if evaluator.parameters else 1
        )
        start = time.perf_counter()
        result = study.optimize(
            lambda trial: evaluator.evaluate(trial, X, y, cv_groups, exposure=exposure)
        )
        return result, time.perf_counter() - start

    return (tune_model,)


@app.cell
def _(mo):
    run_model1 = mo.ui.run_button(label="Tune Model 1")
    mo.md(
        f"{run_model1} three studies × `n_trials` trials, in parallel; ≈ 5 min at 200."
    )
    return (run_model1,)


@app.cell
def _(
    MODEL1_EVALUATORS,
    Parallel,
    X_train,
    Y_train,
    cv_groups,
    delayed,
    exposure_train,
    mo,
    n_trials,
    run_model1,
    seed,
    tune_model,
):
    mo.stop(not run_model1.value, mo.md("Click the button to tune Model 1."))
    # One process per family; study i gets seed + i.
    model1_runs = Parallel(n_jobs=len(MODEL1_EVALUATORS))(
        delayed(tune_model)(
            evaluator,
            X_train,
            Y_train,
            exposure=exposure_train,
            cv_groups=cv_groups,
            study_seed=seed + i,
            n_trials=n_trials,
        )
        for i, evaluator in enumerate(MODEL1_EVALUATORS.values())
    )
    MODEL1_RESULTS = dict(zip(MODEL1_EVALUATORS, model1_runs, strict=True))
    return (MODEL1_RESULTS,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    CategoricalParameter,
    Counter,
    KeyedParameter,
    POISSON_DEVIANCE,
    pd,
):
    def score_best(
        evaluators,
        results,
        components,
        *,
        X_train,
        Y_train,
        X_val,
        Y_val,
        exposure_train,
        exposure_val,
    ):
        """Refit each family's best combination on the training part; the models and three tables.

        ``results`` maps a family to ``(TuningResult, seconds)``; ``components``
        are the set_params prefixes of the parts a trial chooses for
        (``cohort_models__<cohort>`` or ``total_model`` / ``probability_model``).
        The tables: the Poisson deviance per cohort and of the composite on
        training and validation; the best categorical choices per component;
        the trials per arm of every categorical, so a starved arm is visible
        before a family is declared worse.
        """
        models = {}
        score_rows, choice_rows, arm_rows = [], [], []
        for family, (result, seconds) in results.items():
            evaluator = evaluators[family]
            model = evaluator.build_model(result.best_params).fit(
                X_train, Y_train, exposure=exposure_train
            )
            models[family] = model
            pred_train = model.predict(X_train, exposure=exposure_train)
            pred_val = model.predict(X_val, exposure=exposure_val)
            for cohort in Y_train.columns:
                score_rows.append(
                    {
                        "family": family,
                        "cohort": cohort,
                        "train": model.evaluate(
                            Y_train[cohort], pred_train[cohort], POISSON_DEVIANCE
                        ),
                        "val": model.evaluate(
                            Y_val[cohort], pred_val[cohort], POISSON_DEVIANCE
                        ),
                    }
                )
            score_rows.append(
                {
                    "family": family,
                    "cohort": "composite",
                    "cv_best": result.best_value,
                    "train": model.evaluate(
                        Y_train, pred_train, COHORT_POISSON_DEVIANCE
                    ),
                    "val": model.evaluate(Y_val, pred_val, COHORT_POISSON_DEVIANCE),
                    "seconds": seconds,
                }
            )
            categoricals = [
                p
                for p in evaluator.parameters
                if isinstance(p, (CategoricalParameter, KeyedParameter))
            ]
            for component in components:
                prefix = f"{component}__"
                choice_rows.append(
                    {
                        "family": family,
                        "component": component,
                        **{
                            p.name.removeprefix(prefix): result.best_params[p.name]
                            for p in categoricals
                            if p.name.startswith(prefix)
                        },
                    }
                )
            for p in categoricals:
                counts = Counter(trial.params[p.name] for trial in result.trials)
                arm_rows.extend(
                    {
                        "family": family,
                        "parameter": p.name,
                        "value": value,
                        "n_trials": count,
                    }
                    for value, count in counts.items()
                )
        return (
            models,
            pd.DataFrame(score_rows),
            pd.DataFrame(choice_rows),
            pd.DataFrame(arm_rows),
        )

    return (score_best,)


@app.cell
def _(
    COHORTS,
    MODEL1_EVALUATORS,
    MODEL1_RESULTS,
    X_train,
    X_val,
    Y_train,
    Y_val,
    exposure_train,
    exposure_val,
    score_best,
):
    MODEL1_MODELS, model1_df, model1_choices_df, model1_arms_df = score_best(
        MODEL1_EVALUATORS,
        MODEL1_RESULTS,
        [f"cohort_models__{cohort}" for cohort in COHORTS],
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        exposure_train=exposure_train,
        exposure_val=exposure_val,
    )
    return MODEL1_MODELS, model1_arms_df, model1_choices_df, model1_df


@app.cell
def _(mo, model1_arms_df, model1_choices_df, model1_df):
    mo.vstack(
        [
            mo.md("**Model 1: cohort Poisson deviance** (cv_best: the study's value)"),
            mo.md(f"```text\n{model1_df.round(4).to_string(index=False)}\n```"),
            mo.md("**Best combination per component** (one per cohort)"),
            mo.md(f"```text\n{model1_choices_df.to_string(index=False)}\n```"),
            mo.md("**Trials per arm**"),
            mo.md(f"```text\n{model1_arms_df.to_string(index=False)}\n```"),
        ]
    )


@app.cell
def _():
    # Model 2's total families, a list to edit after Model 1's table above. The
    # pairs tuned are this list × the probability families.
    MODEL2_TOTALS = ["lgbm", "poisson_glm", "nb2"]
    return (MODEL2_TOTALS,)


@app.cell
def _(
    COHORT_POISSON_DEVIANCE,
    COUNT_FAMILIES,
    CVHyperparameterEvaluator,
    CohortProbabilityModel,
    CountModel,
    MODEL2_TOTALS,
    PROBABILITY_FAMILIES,
    TotalTimesProbabilityModel,
    cv,
    prefixed,
):
    # Model 2, one evaluator per (total, probability) pair: the total's
    # parameters under total_model__, the classifier's under probability_model__.
    MODEL2_EVALUATORS = {
        f"{total}__{probability}": CVHyperparameterEvaluator(
            TotalTimesProbabilityModel(
                total_model=CountModel(estimator=COUNT_FAMILIES[total][0]),
                probability_model=CohortProbabilityModel(estimator=estimator),
            ),
            [
                *prefixed("total_model", COUNT_FAMILIES[total][1]),
                *prefixed("probability_model", parameters),
            ],
            cv=cv,
            metric=COHORT_POISSON_DEVIANCE,
        )
        for total in MODEL2_TOTALS
        for probability, (estimator, parameters) in PROBABILITY_FAMILIES.items()
    }
    return (MODEL2_EVALUATORS,)


@app.cell
def _(MODEL2_EVALUATORS, mo):
    run_model2 = mo.ui.run_button(label="Tune Model 2")
    mo.md(
        f"{run_model2} {len(MODEL2_EVALUATORS)} studies × `n_trials` trials, in "
        "parallel; ≈ 12 min at 200 for six."
    )
    return (run_model2,)


@app.cell
def _(
    MODEL2_EVALUATORS,
    Parallel,
    X_train,
    Y_train,
    cv_groups,
    delayed,
    exposure_train,
    mo,
    n_trials,
    run_model2,
    seed,
    tune_model,
):
    mo.stop(not run_model2.value, mo.md("Click the button to tune Model 2."))
    # One process per pair; study i gets seed + i.
    model2_runs = Parallel(n_jobs=len(MODEL2_EVALUATORS))(
        delayed(tune_model)(
            evaluator,
            X_train,
            Y_train,
            exposure=exposure_train,
            cv_groups=cv_groups,
            study_seed=seed + i,
            n_trials=n_trials,
        )
        for i, evaluator in enumerate(MODEL2_EVALUATORS.values())
    )
    MODEL2_RESULTS = dict(zip(MODEL2_EVALUATORS, model2_runs, strict=True))
    return (MODEL2_RESULTS,)


@app.cell
def _(
    MODEL2_EVALUATORS,
    MODEL2_RESULTS,
    X_train,
    X_val,
    Y_train,
    Y_val,
    exposure_train,
    exposure_val,
    score_best,
):
    MODEL2_MODELS, model2_df, model2_choices_df, model2_arms_df = score_best(
        MODEL2_EVALUATORS,
        MODEL2_RESULTS,
        ["total_model", "probability_model"],
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        exposure_train=exposure_train,
        exposure_val=exposure_val,
    )
    return MODEL2_MODELS, model2_arms_df, model2_choices_df, model2_df


@app.cell
def _(mo, model2_arms_df, model2_choices_df, model2_df):
    mo.vstack(
        [
            mo.md("**Model 2: cohort Poisson deviance** (cv_best: the study's value)"),
            mo.md(f"```text\n{model2_df.round(4).to_string(index=False)}\n```"),
            mo.md("**Best combination per component**"),
            mo.md(f"```text\n{model2_choices_df.to_string(index=False)}\n```"),
            mo.md("**Trials per arm**"),
            mo.md(f"```text\n{model2_arms_df.to_string(index=False)}\n```"),
        ]
    )


@app.cell
def _(mo, model1_df, model2_df, pd):
    # Each model's best family on validation, from the composite rows. A fresh
    # index: the two frames' labels overlap, and idxmin + loc would return both.
    _composites = pd.concat(
        [
            model1_df[model1_df["cohort"] == "composite"].assign(model="model1"),
            model2_df[model2_df["cohort"] == "composite"].assign(model="model2"),
        ],
        ignore_index=True,
    )
    best_df = _composites.loc[
        _composites.groupby("model")["val"].idxmin(),
        ["model", "family", "cv_best", "train", "val"],
    ]
    mo.vstack(
        [
            mo.md("**Best family per model** (validation cohort Poisson deviance)"),
            mo.md(f"```text\n{best_df.round(4).to_string(index=False)}\n```"),
        ]
    )
    return (best_df,)


if __name__ == "__main__":
    app.run()
